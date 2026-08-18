from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .adapters import ENGINE_LABELS, ENGINE_ORDER, build_adapters
from .adapters.base import AnalysisContext
from .jsonl import write_jsonl
from .pages import parse_page_range
from .rendering import get_source_page_count, prepare_source_for_analysis
from .schemas import AnalysisRun, EngineStatus, LayoutRecord, write_json
from .settings import Settings


def create_run_id(pdf_path: str | Path, page_range: str, engine_ids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(Path(pdf_path).read_bytes())
    digest.update(page_range.encode("utf-8"))
    digest.update(",".join(engine_ids).encode("utf-8"))
    digest.update(str(time.time_ns()).encode("ascii"))
    return digest.hexdigest()[:16]


def analyze_pdf(
    pdf_path: str | Path,
    page_range: str,
    engine_ids: list[str],
    settings: Settings,
    min_confidence: float = 0.0,
    dpi: int | None = None,
) -> AnalysisRun:
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise FileNotFoundError("PDF / 画像ファイルが見つかりません。")
    adapters = build_adapters(settings)
    selected = [engine for engine in ENGINE_ORDER if engine in engine_ids and engine in adapters]
    if not selected:
        raise ValueError("利用するエンジンを 1 つ以上選択してください。")

    page_count = get_source_page_count(source_path)
    pages_to_run = parse_page_range(page_range, page_count, settings.max_default_pages)
    run_id = create_run_id(source_path, ",".join(str(page) for page in pages_to_run), selected)
    run_dir = settings.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_pdf_path, pages = prepare_source_for_analysis(
        source_path,
        pages_to_run,
        run_dir,
        dpi or settings.render_dpi,
    )
    for page in pages:
        page.image_url = f"/artifacts/{run_id}/pages/{Path(page.image_path).name}"

    context = AnalysisContext(
        pdf_path=Path(run_pdf_path),
        run_dir=run_dir,
        pages=pages,
        settings=settings,
        min_confidence=min_confidence,
    )
    records: list[LayoutRecord] = []
    statuses: list[EngineStatus] = []
    for engine_id in selected:
        adapter = adapters[engine_id]
        availability = adapter.availability()
        if not availability.available:
            statuses.append(
                EngineStatus(
                    engine=engine_id,
                    label=ENGINE_LABELS[engine_id],
                    available=False,
                    message=availability.message,
                    count=0,
                )
            )
            continue
        started = time.monotonic()
        try:
            engine_records = adapter.analyze(context)
        except Exception as exc:
            statuses.append(
                EngineStatus(
                    engine=engine_id,
                    label=ENGINE_LABELS[engine_id],
                    available=False,
                    message=f"解析に失敗しました: {exc}",
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    count=0,
                )
            )
            continue
        records.extend(engine_records)
        statuses.append(
            EngineStatus(
                engine=engine_id,
                label=ENGINE_LABELS[engine_id],
                available=True,
                message="解析が完了しました。",
                elapsed_seconds=round(time.monotonic() - started, 3),
                count=len(engine_records),
            )
        )

    json_path = run_dir / "results.json"
    jsonl_path = run_dir / "results.jsonl"
    viewer_data_path = run_dir / "viewer-data.json"
    run = AnalysisRun(
        run_id=run_id,
        pdf_name=source_path.name,
        pdf_path=run_pdf_path,
        pages=pages,
        records=records,
        statuses=statuses,
        output_dir=str(run_dir),
        json_path=str(json_path),
        jsonl_path=str(jsonl_path),
        viewer_data_path=str(viewer_data_path),
    )
    write_json(json_path, run.to_dict())
    write_jsonl(jsonl_path, records, pages)
    write_json(viewer_data_path, run.viewer_payload())
    return run


def preview_pdf(
    pdf_path: str | Path,
    settings: Settings,
    dpi: int | None = None,
) -> AnalysisRun:
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise FileNotFoundError("PDF / 画像ファイルが見つかりません。")

    page_count = get_source_page_count(source_path)
    page_numbers = list(range(1, page_count + 1))
    if not page_numbers:
        raise ValueError("ファイルにページがありません。")

    run_id = create_run_id(source_path, "preview-all-pages", [])
    run_dir = settings.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_pdf_path, pages = prepare_source_for_analysis(
        source_path,
        page_numbers,
        run_dir,
        dpi or settings.render_dpi,
    )
    for page in pages:
        page.image_url = f"/artifacts/{run_id}/pages/{Path(page.image_path).name}"

    json_path = run_dir / "results.json"
    jsonl_path = run_dir / "results.jsonl"
    viewer_data_path = run_dir / "viewer-data.json"
    run = AnalysisRun(
        run_id=run_id,
        pdf_name=source_path.name,
        pdf_path=run_pdf_path,
        pages=pages,
        records=[],
        statuses=[],
        output_dir=str(run_dir),
        json_path=str(json_path),
        jsonl_path=str(jsonl_path),
        viewer_data_path=str(viewer_data_path),
    )
    write_json(json_path, run.to_dict())
    write_jsonl(jsonl_path, [], pages)
    write_json(viewer_data_path, run.viewer_payload())
    return run


def summarize_run(run: AnalysisRun) -> str:
    lines = [f"### 解析結果", f"- Run ID: `{run.run_id}`", f"- ファイル: `{run.pdf_name}`", f"- ページ数: {len(run.pages)}", ""]
    for status in run.statuses:
        state = "有効" if status.available else "無効"
        elapsed = f" / {status.elapsed_seconds:.3f}s" if status.elapsed_seconds is not None else ""
        lines.append(f"- {status.label}: {state} / {status.count} 件{elapsed} / {status.message}")
    return "\n".join(lines)


def summarize_preview(run: AnalysisRun) -> str:
    return "\n".join(
        [
            "### ファイルプレビュー",
            f"- Run ID: `{run.run_id}`",
            f"- ファイル: `{run.pdf_name}`",
            f"- ページ数: {len(run.pages)}",
            "- 解析はまだ実行されていません。",
        ]
    )
