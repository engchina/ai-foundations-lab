from __future__ import annotations

import base64
from contextlib import contextmanager
import inspect
import importlib
import importlib.machinery
import json
import os
import re
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pdf_layout_lab import model_pool
from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox
from pdf_layout_lab.prompts import PROMPT_LAYOUT_ALL_EN, PROMPT_PICTURE_MERMAID
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, extra_install_command, has_module


LOCAL_BACKENDS = {"transformers", "hf", "local", "local_hf"}
API_BACKENDS = {"api", "vllm", "openai", "openai_api", "local_vllm"}
_FLASH_ATTN_IMPORT_ERROR = (
    "DOTS_MOCR_ATTN_IMPLEMENTATION=flash_attention_2 には flash_attn が必要です。"
    "`pip install flash_attn` を実行するか、DOTS_MOCR_ATTN_IMPLEMENTATION=sdpa を指定してください。"
)


PICTURE_OCR_MAX_PER_PAGE = 8
PICTURE_OCR_PADDING = 8
PICTURE_OCR_MIN_SIDE = 32
# 図でない画像だと延々と捏造を続けるので Mermaid 生成は短めに打ち切る
PICTURE_MERMAID_MAX_NEW_TOKENS = 1024
PICTURE_MERMAID_MIN_EDGES = 2
PICTURE_MERMAID_MIN_UNIQUE_LABEL_RATIO = 0.6
_MERMAID_STYLE_LINE = re.compile(r"^\s*(style|linkStyle|classDef|class)\b")
# 開き括弧と同じ種類の閉じ括弧までをラベルとみなす（"終了 (完了)" のような括弧入りラベルを壊さない）
_MERMAID_NODE_LABEL = re.compile(r'([A-Za-z_][\w-]*)(?:\[([^\]"]+)\]|\(([^\)"]+)\)|\{([^\}"]+)\})')


def _save_picture_crop(page_picture, record: LayoutRecord, crop_dir: Path) -> Path | None:
    x1, y1, x2, y2 = (float(value) for value in record.bbox)
    left = max(0, int(x1) - PICTURE_OCR_PADDING)
    top = max(0, int(y1) - PICTURE_OCR_PADDING)
    right = min(page_picture.width, int(x2) + PICTURE_OCR_PADDING)
    bottom = min(page_picture.height, int(y2) + PICTURE_OCR_PADDING)
    if right - left < PICTURE_OCR_MIN_SIDE or bottom - top < PICTURE_OCR_MIN_SIDE:
        return None
    crop_path = crop_dir / f"{record.id}.png"
    page_picture.crop((left, top, right, bottom)).save(crop_path)
    return crop_path


def _texts_from_layout_response(response_text: str) -> str:
    """切り出し画像の再解析結果から本文だけを取り出す。

    写真・ロゴ・装飾のように文字が無い画像では空文字を返し、Picture を空のままにする。
    """
    try:
        payload = extract_first_json(response_text)
    except ValueError:
        # JSON にならない応答はそのまま本文として扱う
        return response_text.strip()
    elements = coerce_layout_elements(payload)
    texts = [str(element.get("text") or element.get("content") or "").strip() for element in elements]
    return "\n".join(text for text in texts if text)


def _mermaid_from_response(response_text: str) -> str:
    """Mermaid 生成の応答を検証・整形し、```mermaid フェンス付きで返す。図でない画像への捏造は空文字にする。

    dots.mocr は NONE を返さず、QR コードや印鑑でも同じラベルを繰り返すグラフや style 行の羅列を
    生成するため、辺の数とラベルの重複率で足切りする。
    """
    body = re.sub(r"^\s*```(?:mermaid)?\s*|\s*```\s*$", "", response_text.strip())
    if not re.match(r"\s*(graph|flowchart)\b", body):
        return ""
    lines = [line for line in body.splitlines() if line.strip() and not _MERMAID_STYLE_LINE.match(line)]
    edges = [line for line in lines if "-->" in line]
    labels = [_mermaid_label(match) for match in _MERMAID_NODE_LABEL.finditer("\n".join(lines))]
    if len(edges) < PICTURE_MERMAID_MIN_EDGES or not labels:
        return ""
    if len(set(labels)) < PICTURE_MERMAID_MIN_UNIQUE_LABEL_RATIO * len(labels):
        return ""
    # ラベル内の括弧やパイプで Mermaid の構文が壊れないよう引用符で包む
    quoted = "\n".join(_MERMAID_NODE_LABEL.sub(_quote_mermaid_label, line) for line in lines)
    return f"```mermaid\n{quoted}\n```"


def _mermaid_label(match: re.Match) -> str:
    return next(group for group in match.groups()[1:] if group is not None).strip()


def _quote_mermaid_label(match: re.Match) -> str:
    opener, closer = {2: "[]", 3: "()", 4: "{}"}[match.lastindex]
    return f'{match.group(1)}{opener}"{_mermaid_label(match)}"{closer}'


class DotsMocrAdapter:
    engine_id = "dots_mocr"
    label = "dots.mocr"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        backend = _normalize_backend(self.settings.dots_mocr_backend)
        if backend in API_BACKENDS:
            if not self.settings.dots_mocr_base_url:
                return AdapterAvailability(False, "DOTS_MOCR_BASE_URL が未設定です。")
            return AdapterAvailability(True, "ローカル vLLM / OpenAI 互換 API へ接続します。")
        if backend in LOCAL_BACKENDS:
            missing = [name for name in ("torch", "transformers", "qwen_vl_utils") if not has_module(name)]
            if missing:
                install = extra_install_command("dots")
                return AdapterAvailability(
                    False,
                    f"dots.mocr ローカル実行の依存関係が未インストールです。プロジェクト直下で `{install}` を実行してから再試行してください。不足: {', '.join(missing)}",
                )
            if _explicitly_requires_flash_attn(
                self.settings.dots_mocr_attn_implementation
            ) and not _flash_attn_available():
                return AdapterAvailability(False, _FLASH_ATTN_IMPORT_ERROR)
            device = self.settings.dots_mocr_device or "auto"
            return AdapterAvailability(True, f"Transformers で dots.mocr をローカル実行します（device={device}）。")
        return AdapterAvailability(
            False,
            f"DOTS_MOCR_BACKEND={self.settings.dots_mocr_backend!r} は未対応です。transformers または api を指定してください。",
        )

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        backend = _normalize_backend(self.settings.dots_mocr_backend)
        records: list[LayoutRecord] = []
        for page_image in context.pages:
            page_records: list[LayoutRecord] = []
            response_text = self._infer(Path(page_image.image_path), PROMPT_LAYOUT_ALL_EN, backend, context)
            payload = extract_first_json(response_text)
            elements = coerce_layout_elements(payload)
            for index, element in enumerate(elements, start=1):
                confidence = _optional_float(element.get("confidence") or element.get("score"))
                if confidence is not None and confidence < context.min_confidence:
                    continue
                bbox = element.get("bbox") or element.get("box")
                if not bbox:
                    continue
                category = normalize_category(element.get("category") or element.get("type"))
                text = "" if category == "Picture" else str(element.get("text") or element.get("content") or "")
                page_records.append(
                    LayoutRecord(
                        id=f"dots-p{page_image.page}-{index}",
                        engine=self.engine_id,
                        page=page_image.page,
                        seq_no=index,
                        bbox=clamp_bbox(bbox, page_image.width, page_image.height),
                        coord_system="image_top_left",
                        page_width=page_image.width,
                        page_height=page_image.height,
                        category=category,
                        text=text,
                        confidence=confidence,
                        raw_type=str(element.get("category") or element.get("type") or ""),
                        raw=dict(element),
                    )
                )
            self._ocr_pictures(page_image, page_records, backend, context)
            records.extend(page_records)
        return records

    def _ocr_pictures(
        self,
        page_image,
        records: list[LayoutRecord],
        backend: str,
        context: AnalysisContext,
    ) -> None:
        """公式プロンプトが text を返さない Picture を切り出し、同じ prompt_layout_all_en で本文を取り直す。

        公式 prompt_ocr は図の見出しだけを返すことがあり、prompt_grounding_ocr は bbox の
        座標系が processor 側の resize と合わず別領域を読むため、切り出し画像に対して
        prompt_layout_all_en を使い、要素ごとの text を読み順に連結する。
        """
        targets = [record for record in records if record.category == "Picture" and not record.text.strip()]
        if not targets:
            return

        from PIL import Image

        crop_dir = Path(context.run_dir) / "dots_mocr"
        crop_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(page_image.image_path) as opened:
            page_picture = opened.convert("RGB")
        # ponytail: 1 枚ごとに推論が走るのでページあたりの上限を置く。足りなければ増やす
        for record in targets[:PICTURE_OCR_MAX_PER_PAGE]:
            crop_path = _save_picture_crop(page_picture, record, crop_dir)
            if not crop_path:
                continue
            text = _texts_from_layout_response(self._infer(crop_path, PROMPT_LAYOUT_ALL_EN, backend, context))
            if not text:
                continue
            record.text = text
            record.raw["picture_ocr"] = True
            # 文字のある図だけ Mermaid 化を試み、通れば本文を置き換える（OCR 本文は raw に残す）
            mermaid = _mermaid_from_response(
                self._infer(crop_path, PROMPT_PICTURE_MERMAID, backend, context, max_new_tokens=PICTURE_MERMAID_MAX_NEW_TOKENS)
            )
            if mermaid:
                record.raw["picture_text"] = text
                record.text = mermaid

    def _infer(self, image_path: Path, prompt: str, backend: str, context: AnalysisContext, max_new_tokens: int | None = None) -> str:
        if backend in API_BACKENDS:
            return self._call_vllm(image_path, prompt, max_new_tokens)
        return self._call_transformers(image_path, prompt, context, max_new_tokens)

    def preload(self) -> None:
        """モデルを事前にロードして常駐させる（UI の「ロード」ボタン用）。"""
        if _normalize_backend(self.settings.dots_mocr_backend) in LOCAL_BACKENDS:
            _prepare_runtime_env(self.settings)
            _get_local_runtime(self.settings)

    def _call_transformers(self, image_path: Path, prompt: str, context: AnalysisContext, max_new_tokens: int | None = None) -> str:
        _prepare_runtime_env(context.settings)
        runtime = _get_local_runtime(self.settings)
        return runtime.infer(image_path, prompt, max_new_tokens or self.settings.dots_mocr_max_new_tokens)

    def _call_vllm(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.settings.dots_mocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        if max_new_tokens:
            payload["max_tokens"] = max_new_tokens
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.settings.dots_mocr_base_url.rstrip('/')}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.dots_mocr_api_key or 'EMPTY'}",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.settings.dots_mocr_timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"dots.mocr API に接続できませんでした: {exc}") from exc
        if time.monotonic() - started > self.settings.dots_mocr_timeout_seconds:
            raise RuntimeError("dots.mocr API がタイムアウトしました。")
        return str(result["choices"][0]["message"]["content"])


class _LocalDotsMocrRuntime:
    def __init__(self, settings: Settings):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.device = _resolve_device(torch, settings.dots_mocr_device)
        self.input_device = _resolve_input_device(self.device, settings.dots_mocr_device_map)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        dtype = _resolve_torch_dtype(torch, settings.dots_mocr_torch_dtype, self.device)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        flash_attn_available = _flash_attn_available()
        attn_implementation = _resolve_attn_implementation(
            settings.dots_mocr_attn_implementation,
            flash_attn_available=flash_attn_available,
            device=self.device,
        )
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        device_map = settings.dots_mocr_device_map.strip()
        if device_map:
            model_kwargs["device_map"] = device_map

        with _flash_attn_import_fallback(attn_implementation, flash_attn_available):
            if attn_implementation:
                config = AutoConfig.from_pretrained(settings.dots_mocr_model, trust_remote_code=True)
                _set_vision_attn_implementation(config, attn_implementation)
                model_kwargs["config"] = config
            self.model = AutoModelForCausalLM.from_pretrained(settings.dots_mocr_model, **model_kwargs)
        effective_dtype = dtype or _default_runtime_dtype(torch, self.device)
        _cast_floating_tensors_to_dtype(self.model, effective_dtype)
        _patch_vision_tower_forward(self.model, torch)
        _patch_vision_sdpa_attention(self.model, torch)
        if not device_map:
            self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(settings.dots_mocr_model, trust_remote_code=True, use_fast=True)

    def infer(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.input_device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return str(
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )


def _get_local_runtime(settings: Settings) -> _LocalDotsMocrRuntime:
    signature = (
        settings.dots_mocr_model,
        settings.dots_mocr_device,
        settings.dots_mocr_device_map,
        settings.dots_mocr_torch_dtype,
        settings.dots_mocr_attn_implementation,
    )
    return model_pool.get(
        DotsMocrAdapter.engine_id,
        lambda: _LocalDotsMocrRuntime(settings),
        signature=signature,
    )


def _prepare_runtime_env(settings: Settings) -> None:
    cache_root = (settings.output_dir / "_cache" / "dots_mocr").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _normalize_backend(value: str) -> str:
    return (value or "transformers").strip().lower().replace("-", "_")


def _resolve_device(torch: Any, requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return value


def _resolve_input_device(device: str, device_map: str) -> str:
    if device_map and device_map.strip().lower() == "auto":
        return "cuda" if device == "cuda" else device
    return device


def _normalize_attn_implementation(requested: str) -> str:
    value = (requested or "auto").strip().lower().replace("-", "_")
    aliases = {
        "fa2": "flash_attention_2",
        "flash": "flash_attention_2",
        "flash_attn": "flash_attention_2",
        "flash_attention": "flash_attention_2",
    }
    return aliases.get(value, value)


def _explicitly_requires_flash_attn(requested: str) -> bool:
    return _normalize_attn_implementation(requested) == "flash_attention_2"


def _flash_attn_available() -> bool:
    if not has_module("flash_attn"):
        return False
    try:
        module = importlib.import_module("flash_attn")
    except Exception:
        return False
    return hasattr(module, "flash_attn_varlen_func")


def _resolve_attn_implementation(requested: str, *, flash_attn_available: bool, device: str) -> str | None:
    value = _normalize_attn_implementation(requested)
    if value in {"", "auto"}:
        return None if flash_attn_available and device == "cuda" else "sdpa"
    if value == "flash_attention_2" and not flash_attn_available:
        raise RuntimeError(_FLASH_ATTN_IMPORT_ERROR)
    return value


def _set_vision_attn_implementation(config: Any, attn_implementation: str) -> None:
    vision_config = getattr(config, "vision_config", None)
    if isinstance(vision_config, dict):
        vision_config["attn_implementation"] = attn_implementation
    elif vision_config is not None:
        setattr(vision_config, "attn_implementation", attn_implementation)


@contextmanager
def _flash_attn_import_fallback(attn_implementation: str | None, flash_attn_available: bool):
    if flash_attn_available or attn_implementation == "flash_attention_2":
        yield
        return

    sentinel = object()
    previous = sys.modules.get("flash_attn", sentinel)
    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

    def flash_attn_varlen_func(*args, **kwargs):
        raise RuntimeError(_FLASH_ATTN_IMPORT_ERROR)

    module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = module
    try:
        yield
    finally:
        if previous is sentinel:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = previous


def _resolve_torch_dtype(torch: Any, requested: str, device: str) -> Any | None:
    value = (requested or "auto").strip().lower()
    if value == "auto":
        if device == "cuda":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if bf16_supported else torch.float16
        return None
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_by_name.get(value)


def _default_runtime_dtype(torch: Any, device: str) -> Any:
    if device == "cuda":
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if bf16_supported else torch.float16
    return torch.float32


def _iter_tensors(module: Any, method_name: str):
    method = getattr(module, method_name, None)
    if not callable(method):
        return ()
    try:
        return method(recurse=True)
    except TypeError:
        return method()


def _is_floating_tensor(tensor: Any) -> bool:
    is_floating_point = getattr(tensor, "is_floating_point", None)
    if callable(is_floating_point):
        return bool(is_floating_point())
    dtype = getattr(tensor, "dtype", None)
    return dtype is not None and "float" in str(dtype).lower()


def _cast_tensor_to_dtype(tensor: Any, dtype: Any) -> None:
    if getattr(tensor, "dtype", None) == dtype or getattr(tensor, "is_meta", False):
        return
    data = getattr(tensor, "data", None)
    if data is not None and hasattr(data, "to"):
        tensor.data = data.to(dtype=dtype)
        return
    to = getattr(tensor, "to", None)
    if callable(to):
        to(dtype=dtype)


def _cast_floating_tensors_to_dtype(module: Any, dtype: Any | None) -> None:
    if dtype is None:
        return
    for tensor in _iter_tensors(module, "parameters"):
        if _is_floating_tensor(tensor):
            _cast_tensor_to_dtype(tensor, dtype)
    for tensor in _iter_tensors(module, "buffers"):
        if _is_floating_tensor(tensor):
            _cast_tensor_to_dtype(tensor, dtype)


def _first_floating_dtype(module: Any) -> Any | None:
    for tensor in _iter_tensors(module, "parameters"):
        if _is_floating_tensor(tensor):
            return getattr(tensor, "dtype", None)
    for tensor in _iter_tensors(module, "buffers"):
        if _is_floating_tensor(tensor):
            return getattr(tensor, "dtype", None)
    return None


def _dtype_is(torch: Any, dtype: Any, name: str) -> bool:
    expected = getattr(torch, name, None)
    return expected is not None and dtype == expected


def _patch_vision_tower_forward(model: Any, torch: Any) -> None:
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is None or getattr(vision_tower, "_pdf_layout_lab_dtype_patch", False):
        return
    original_forward = getattr(vision_tower, "forward", None)
    if not callable(original_forward):
        return
    try:
        accepts_bf16 = "bf16" in inspect.signature(original_forward).parameters
    except (TypeError, ValueError):
        accepts_bf16 = False
    if not accepts_bf16:
        return

    def patched_forward(hidden_states, grid_thw=None, *args, **kwargs):
        target_dtype = _first_floating_dtype(vision_tower)
        if target_dtype is not None and hasattr(hidden_states, "to"):
            hidden_states = hidden_states.to(dtype=target_dtype)
        if "bf16" not in kwargs and not args:
            kwargs["bf16"] = _dtype_is(torch, target_dtype, "bfloat16")
        return original_forward(hidden_states, grid_thw, *args, **kwargs)

    setattr(vision_tower, "_pdf_layout_lab_original_forward", original_forward)
    setattr(vision_tower, "_pdf_layout_lab_dtype_patch", True)
    setattr(vision_tower, "forward", patched_forward)


def _patch_vision_sdpa_attention(model: Any, torch: Any) -> None:
    """dots.mocr の SDPA 版 vision attention を、メモリ効率の良いカーネルが選ばれる形に差し替える。

    元実装は q/k/v を 3 次元 [heads, N, dim] で渡し、さらに [1, N, N] の bool マスクを付ける。
    この組み合わせでは PyTorch の flash / efficient カーネルが使えず math 実装へフォールバックし、
    N×N の行列を実体化する（300 DPI の A4 ページで N≈44,000 → 約 90 GiB を要求して OOM になる）。
    4 次元に整形し、画像が 1 枚だけのとき（cu_seqlens が 1 区間）はマスクを付けないことで、
    メモリ使用量を O(N) に抑える。flash_attn がインストール済みの環境ではそもそも呼ばれない。
    """
    vision_tower = getattr(model, "vision_tower", None)
    blocks = getattr(vision_tower, "blocks", None)
    if not blocks:
        return
    attn_cls = type(getattr(blocks[0], "attn", None))
    if attn_cls.__name__ != "VisionSdpaAttention" or getattr(attn_cls, "_pdf_layout_lab_sdpa_patch", False):
        return
    apply_rotary_pos_emb_vision = getattr(sys.modules.get(attn_cls.__module__), "apply_rotary_pos_emb_vision", None)
    if apply_rotary_pos_emb_vision is None:
        return
    F = torch.nn.functional

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        # [N, heads, dim] → [1, heads, N, dim]
        q, k, v = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (q, k, v))
        attention_mask = None
        if len(cu_seqlens) > 2:  # 複数画像のときだけブロック対角マスクが必要
            attention_mask = torch.zeros([1, 1, seq_length, seq_length], device=q.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)

    attn_cls._pdf_layout_lab_original_forward = attn_cls.forward
    attn_cls.forward = forward
    attn_cls._pdf_layout_lab_sdpa_patch = True


def extract_first_json(text: str) -> Any:
    """LLM 応答から最初の JSON オブジェクト/配列を取り出す。"""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    starts = [pos for pos in [stripped.find("{"), stripped.find("[")] if pos >= 0]
    if not starts:
        raise ValueError("JSON 応答が見つかりませんでした。")
    start = min(starts)
    for end in range(len(stripped), start, -1):
        candidate = stripped[start:end].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("JSON 応答を解析できませんでした。")


def coerce_layout_elements(payload: Any) -> list[dict[str, Any]]:
    """トップレベルキーが固定されていない JSON を要素配列へ寄せる。"""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("elements", "layout", "layouts", "results", "items", "blocks"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    dict_items = [value for value in payload.values() if isinstance(value, dict) and ("bbox" in value or "box" in value)]
    if dict_items:
        return dict_items
    if "bbox" in payload or "box" in payload:
        return [payload]
    return []


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
