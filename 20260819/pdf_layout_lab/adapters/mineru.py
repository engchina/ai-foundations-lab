from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext


MINERU_CANDIDATES = [
    ".venv/bin/mineru",
    "mineru",
    ".venv/bin/magic-pdf",
    "magic-pdf",
]
MINERU_JSON_NAMES = ["middle.json", "content_list_v2.json", "content_list.json", "model.json"]
MINERU_CONTENT_KEYS = (
    "title_content",
    "paragraph_content",
    "page_header_content",
    "page_footer_content",
    "page_number_content",
    "table_caption",
    "table_body",
    "table_footnote",
    "image_caption",
    "image_footnote",
)


class MineruAdapter:
    engine_id = "mineru"
    label = "MinerU / MinerU2.5-Pro"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if self.settings.mineru_output_dir and Path(self.settings.mineru_output_dir).expanduser().exists():
            return AdapterAvailability(True, "MINERU_OUTPUT_DIR の既存 JSON 出力を取り込みます。")
        command = resolve_mineru_command(self.settings.mineru_command)
        if command:
            if _is_legacy_magic_pdf(command):
                return AdapterAvailability(True, "magic-pdf コマンドを実行して MinerU JSON 出力を取り込みます。")
            return AdapterAvailability(True, f"mineru コマンドを CPU backend `{self.settings.mineru_backend}` で実行します。")
        return AdapterAvailability(
            False,
            "MinerU CLI が見つかりません。CPU 実行の場合はプロジェクト直下で `.venv/bin/pip install -e '.[mineru]'` を実行するか、MINERU_COMMAND に mineru / magic-pdf のパスを設定してください。",
        )

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        output_root = self._ensure_output(context)
        json_files = _find_mineru_json_files(output_root)
        if not json_files:
            raise RuntimeError("MinerU の JSON 出力が見つかりませんでした。")
        for path in json_files:
            records = self._records_from_file(path, context)
            if records:
                return records
        return []

    def _ensure_output(self, context: AnalysisContext) -> Path:
        if self.settings.mineru_output_dir:
            return Path(self.settings.mineru_output_dir).expanduser()
        output_dir = context.run_dir / "mineru"
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_command = resolve_mineru_command(self.settings.mineru_command)
        if not resolved_command:
            raise RuntimeError(self.availability().message)
        command = build_mineru_command(resolved_command, context.pdf_path, output_dir, self.settings)
        subprocess.run(command, check=True, env=_build_mineru_env(context))
        return output_dir

    def _records_from_file(self, path: Path, context: AnalysisContext) -> list[LayoutRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        page_lookup = {page.page: page for page in context.pages}
        records: list[LayoutRecord] = []
        if isinstance(payload, list):
            records.extend(_records_from_vlm_model_json(payload, page_lookup, self.engine_id, context.min_confidence))
        elif isinstance(payload, dict):
            records.extend(_records_from_dict_payload(payload, page_lookup, self.engine_id, context.min_confidence))
        return records


def resolve_mineru_command(command_setting: str) -> list[str] | None:
    configured = (command_setting or "auto").strip()
    candidates = MINERU_CANDIDATES if configured.lower() == "auto" else [configured]
    for candidate in candidates:
        parts = shlex.split(candidate)
        if not parts:
            continue
        executable = _resolve_executable(parts[0])
        if executable:
            return [executable, *parts[1:]]
    return None


def build_mineru_command(command: list[str], pdf_path: Path, output_dir: Path, settings: Settings) -> list[str]:
    extra_args = shlex.split(settings.mineru_extra_args)
    if _is_legacy_magic_pdf(command):
        cli_args = ["-p", str(pdf_path), "-o", str(output_dir)]
        if settings.mineru_method and "-m" not in extra_args and "--method" not in extra_args:
            cli_args.extend(["-m", settings.mineru_method])
        return [*command, *cli_args, *extra_args]

    cli_args = ["-p", str(pdf_path), "-o", str(output_dir)]
    if settings.mineru_backend and "-b" not in extra_args and "--backend" not in extra_args:
        cli_args.extend(["-b", settings.mineru_backend])
    if settings.mineru_method and "-m" not in extra_args and "--method" not in extra_args:
        cli_args.extend(["-m", settings.mineru_method])
    return [*command, *cli_args, *extra_args]


def _build_mineru_env(context: AnalysisContext) -> dict[str, str]:
    env = os.environ.copy()
    cache_root = (context.settings.output_dir / "_cache" / "mineru").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    # MinerU と周辺ライブラリのキャッシュをプロジェクト配下へ寄せ、ホームディレクトリ権限に依存しないようにする。
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env.setdefault("ORT_DISABLE_TELEMETRY", "1")
    env.setdefault("ONNXRUNTIME_DISABLE_TELEMETRY", "1")
    return env


def _resolve_executable(value: str) -> str | None:
    expanded = Path(value).expanduser()
    if expanded.exists():
        return str(expanded)
    resolved = shutil.which(value)
    return resolved


def _is_legacy_magic_pdf(command: list[str]) -> bool:
    return bool(command) and Path(command[0]).name == "magic-pdf"


def _find_mineru_json_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for name in MINERU_JSON_NAMES:
        found.extend(sorted(root.rglob(f"*{name}")))
    return found


def _records_from_vlm_model_json(payload: list[Any], page_lookup: dict[int, Any], engine_id: str, min_confidence: float) -> list[LayoutRecord]:
    records: list[LayoutRecord] = []
    if _is_flat_content_list(payload):
        return _records_from_flat_content_list(payload, page_lookup, engine_id, min_confidence)
    for fallback_page_index, page_payload in enumerate(payload, start=1):
        page_index = fallback_page_index
        blocks = page_payload
        source_size = None
        if isinstance(page_payload, dict):
            page_index = _page_number_from_payload(page_payload, fallback_page_index)
            blocks = _blocks_from_page_payload(page_payload)
            source_size = _source_size_from_page_payload(page_payload)
        elif isinstance(page_payload, list):
            source_size = _source_size_from_content_list_blocks(page_payload)
        page = page_lookup.get(page_index)
        if not page or not isinstance(blocks, list):
            continue
        for seq, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                continue
            record = _block_to_record(block, page, page_index, seq, engine_id, min_confidence, source_size)
            if record:
                records.append(record)
    return records


def _records_from_dict_payload(payload: dict[str, Any], page_lookup: dict[int, Any], engine_id: str, min_confidence: float) -> list[LayoutRecord]:
    records: list[LayoutRecord] = []
    pdf_info = payload.get("pdf_info")
    if isinstance(pdf_info, list):
        for fallback_page_index, page_payload in enumerate(pdf_info, start=1):
            page_index = _page_number_from_payload(page_payload, fallback_page_index)
            page = page_lookup.get(page_index)
            if not page or not isinstance(page_payload, dict):
                continue
            blocks = _blocks_from_page_payload(page_payload)
            source_size = _source_size_from_page_payload(page_payload)
            for seq, block in enumerate(blocks, start=1):
                record = _block_to_record(block, page, page_index, seq, engine_id, min_confidence, source_size)
                if record:
                    records.append(record)
    elif isinstance(payload.get("pages"), list):
        for page_payload in payload["pages"]:
            page_index = _page_number_from_payload(page_payload, 0)
            page = page_lookup.get(page_index)
            if not page:
                continue
            blocks = _blocks_from_page_payload(page_payload)
            source_size = _source_size_from_page_payload(page_payload)
            for seq, block in enumerate(blocks, start=1):
                record = _block_to_record(block, page, page_index, seq, engine_id, min_confidence, source_size)
                if record:
                    records.append(record)
    return records


def _records_from_flat_content_list(
    payload: list[Any], page_lookup: dict[int, Any], engine_id: str, min_confidence: float
) -> list[LayoutRecord]:
    records: list[LayoutRecord] = []
    counters: dict[int, int] = {}
    for block in payload:
        if not isinstance(block, dict):
            continue
        page_number = _page_number_from_payload(block, 0)
        page = page_lookup.get(page_number)
        if not page:
            continue
        counters[page_number] = counters.get(page_number, 0) + 1
        record = _block_to_record(
            block,
            page,
            page_number,
            counters[page_number],
            engine_id,
            min_confidence,
            (1000.0, 1000.0),
        )
        if record:
            records.append(record)
    return records


def _block_to_record(
    block: dict[str, Any],
    page,
    page_number: int,
    seq: int,
    engine_id: str,
    min_confidence: float,
    source_size: tuple[float, float] | None = None,
) -> LayoutRecord | None:
    bbox = block.get("bbox") or block.get("box")
    if not bbox:
        return None
    confidence = _optional_float(_first_present(block, "score", "confidence"))
    if confidence is not None and confidence < min_confidence:
        return None
    raw_type = str(_first_present(block, "type", "category", "block_type", "label") or "")
    text = _extract_block_text(block)
    return LayoutRecord(
        id=f"{engine_id}-p{page_number}-{seq}",
        engine=engine_id,
        page=page_number,
        seq_no=seq,
        bbox=_bbox_to_page_image(bbox, page, source_size),
        coord_system="image_top_left",
        page_width=page.width,
        page_height=page.height,
        category=normalize_category(raw_type),
        text=str(text),
        confidence=confidence,
        raw_type=raw_type,
        raw=block,
    )


def _is_flat_content_list(payload: list[Any]) -> bool:
    dict_items = [item for item in payload if isinstance(item, dict)]
    return bool(dict_items) and any("page_idx" in item for item in dict_items)


def _blocks_from_page_payload(page_payload: dict[str, Any]) -> list[Any]:
    blocks = (
        page_payload.get("preproc_blocks")
        or page_payload.get("para_blocks")
        or page_payload.get("layout_dets")
        or page_payload.get("blocks")
        or page_payload.get("items")
        or []
    )
    blocks = list(blocks) if isinstance(blocks, list) else []
    # ヘッダー / フッター / 欄外注記は discarded_blocks に入る（他エンジンの Page-header/footer に相当）
    discarded = page_payload.get("discarded_blocks")
    if isinstance(discarded, list):
        blocks.extend(discarded)
    return blocks


def _page_number_from_payload(payload: Any, fallback: int) -> int:
    if not isinstance(payload, dict):
        return fallback
    if payload.get("page") is not None:
        return int(payload["page"])
    if payload.get("page_number") is not None:
        return int(payload["page_number"])
    for key in ("page_idx", "page_no"):
        if payload.get(key) is not None:
            return int(payload[key]) + 1
    page_info = payload.get("page_info")
    if isinstance(page_info, dict):
        return _page_number_from_payload(page_info, fallback)
    return fallback


def _source_size_from_page_payload(payload: dict[str, Any]) -> tuple[float, float] | None:
    for value in (
        payload.get("page_size"),
        payload.get("size"),
        payload.get("page_info"),
        payload,
    ):
        source_size = _source_size_from_value(value)
        if source_size:
            return source_size
    return None


def _source_size_from_content_list_blocks(blocks: list[Any]) -> tuple[float, float] | None:
    if any(isinstance(block, dict) and isinstance(block.get("content"), dict) for block in blocks):
        return (1000.0, 1000.0)
    return None


def _source_size_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _valid_source_size(value[0], value[1])
    if isinstance(value, dict):
        width = _first_present(value, "width", "page_width", "w")
        height = _first_present(value, "height", "page_height", "h")
        return _valid_source_size(width, height)
    return None


def _valid_source_size(width: Any, height: Any) -> tuple[float, float] | None:
    try:
        width_float = float(width)
        height_float = float(height)
    except (TypeError, ValueError):
        return None
    if width_float <= 0 or height_float <= 0:
        return None
    return (width_float, height_float)


def _bbox_to_page_image(bbox: Any, page, source_size: tuple[float, float] | None) -> list[float]:
    values = [float(value) for value in bbox]
    if source_size:
        source_width, source_height = source_size
        values = [
            values[0] * float(page.width) / source_width,
            values[1] * float(page.height) / source_height,
            values[2] * float(page.width) / source_width,
            values[3] * float(page.height) / source_height,
        ]
    elif all(0.0 <= value <= 1.0 for value in values):
        values = [values[0] * page.width, values[1] * page.height, values[2] * page.width, values[3] * page.height]
    return clamp_bbox(values, page.width, page.height)


def _extract_block_text(block: dict[str, Any]) -> str:
    for key in ("text", "html", "content"):
        text = _text_from_value(block.get(key))
        if text:
            return text
    text = _text_from_lines(block.get("lines"))
    if text:
        return text
    # 表などは中身が入れ子の blocks 側にあり、外側の block には text も lines も無い
    text = _text_from_blocks(block.get("blocks"))
    if text:
        return text
    return _text_from_value([block.get("image_caption"), block.get("image_footnote")])


def _text_from_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""
    parts = [_extract_block_text(block) for block in blocks if isinstance(block, dict)]
    parts = [part for part in parts if part]
    # ビューアは <table> で始まる文字列だけを表として描画するため、表本体を先頭に置く
    parts.sort(key=lambda part: not part.lstrip().lower().startswith("<table"))
    return "\n".join(parts)


def _text_from_lines(lines: Any) -> str:
    if not isinstance(lines, list):
        return ""
    line_texts: list[str] = []
    for line in lines:
        if isinstance(line, dict) and isinstance(line.get("spans"), list):
            span_text = "".join(_text_from_value(span) for span in line["spans"])
            text = span_text.strip()
        else:
            text = _text_from_value(line)
        if text:
            line_texts.append(text)
    return "\n".join(line_texts)


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return ""
    if isinstance(value, list):
        parts = [_text_from_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "html", "content"):
            text = _text_from_value(value.get(key))
            if text:
                return text
        parts = [_text_from_value(value.get(key)) for key in MINERU_CONTENT_KEYS]
        text = "\n".join(part for part in parts if part)
        if text:
            return text
        return ""
    return str(value).strip()


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
