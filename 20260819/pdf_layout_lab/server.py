from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypeVar

from .adapters import ENGINE_LABELS, ENGINE_ORDER
from .analysis import analyze_pdf, preview_pdf, summarize_preview, summarize_run
from .bootstrap import exec_project_venv_if_available
from .rendering import SUPPORTED_SOURCE_FILE_TYPES
from .settings import get_settings


PresetValue = TypeVar("PresetValue", int, float)

DEFAULT_MIN_CONFIDENCE = 0.5
CONFIDENCE_PRESETS: list[tuple[str, float]] = [
    ("0.25 - 検出重視", 0.25),
    ("0.50 - 標準（既定）", DEFAULT_MIN_CONFIDENCE),
    ("0.75 - 精度重視", 0.75),
]
DPI_PRESETS: list[tuple[str, int]] = [
    ("150 DPI - 低負荷", 150),
    ("200 DPI - バランス", 200),
    ("300 DPI - OCR 標準（既定）", 300),
]

PAGE_SYNC_HEAD = """
<script>
(() => {
  if (window.__pdfLayoutLabPageSyncReady) return;
  window.__pdfLayoutLabPageSyncReady = true;
  window.__pdfLayoutLabSelectedPage = "1";

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type !== "pdf-layout-lab:selected-page") return;

    const page = String(data.page || "").trim();
    if (/^[1-9]\\d*$/.test(page)) {
      window.__pdfLayoutLabSelectedPage = page;
    }
  });
})();
</script>
"""

RUN_PAGE_SELECTION_JS = """
(pdfFile, pageRange, selectedLabels, confidence, dpi) => {
  const readViewerPageInput = () => {
    const frames = Array.from(document.querySelectorAll('iframe[title="PDF / 画像レイアウト比較ビューア"]'));
    for (const frame of frames) {
      try {
        const input = frame.contentDocument?.querySelector('input[aria-label="ページ番号"]');
        const page = input?.value?.trim();
        if (page) return page;
      } catch (error) {
        // Ignore cross-frame access failures and fall back to the posted page value.
      }
    }
    return "";
  };

  const page = readViewerPageInput() || String(window.__pdfLayoutLabSelectedPage || pageRange || "1").trim() || "1";
  return [pdfFile, page, selectedLabels, confidence, dpi];
}
"""

APP_CSS = """
.app-shell { max-width: 1680px; margin: 0 auto; }
.main-title h1 { font-size: 26px; line-height: 1.25; margin-bottom: 4px; }
.main-title p { margin: 0; color: #475569; }
.missing-viewer { padding: 16px; border: 1px solid #fed7aa; background: #fff7ed; border-radius: 8px; color: #7c2d12; }
.missing-viewer code { background: #ffedd5; padding: 2px 5px; border-radius: 4px; }
.setting-help p { margin: 6px 0 2px; color: #475569; font-size: 13px; line-height: 1.45; }
.setting-help strong { color: #334155; }
.preset-radio { margin-top: 2px; }
.settings-row { align-items: flex-start; gap: 16px; margin-top: 10px; }
.setting-card { min-width: 0; }
.gradio-container footer,
footer { display: none !important; }
"""


def _preset_labels(presets: list[tuple[str, int | float]]) -> list[str]:
    return [label for label, _ in presets]


def _preset_label_for_value(presets: list[tuple[str, int | float]], value: int | float | None) -> str | None:
    if value is None:
        return None
    for label, preset_value in presets:
        if isinstance(preset_value, float):
            if abs(float(value) - preset_value) < 1e-9:
                return label
        elif int(value) == preset_value:
            return label
    return None


def _preset_value_for_label(
    presets: list[tuple[str, PresetValue]],
    label: str | None,
    fallback: PresetValue,
) -> PresetValue:
    for preset_label, value in presets:
        if preset_label == label:
            return value
    return fallback


def _file_path(value: Any) -> str:
    if value is None:
        raise ValueError("PDF または画像ファイルをアップロードしてください。")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("name") or value.get("path") or "")
    path = getattr(value, "name", None) or getattr(value, "path", None)
    if not path:
        raise ValueError("PDF / 画像ファイルのパスを取得できませんでした。")
    return str(path)


def _page_range_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        raise ValueError("解析ページにはページ番号を入力してください。")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("解析ページには有効なページ番号を入力してください。")
    if not number.is_integer():
        raise ValueError("解析ページは整数で指定してください。")
    return str(int(number))


def _viewer_frame(run_id: str, viewer_ready: bool) -> str:
    if not viewer_ready:
        return """
        <div class="missing-viewer">
          <h3>React ビューアがまだビルドされていません</h3>
          <p><code>cd viewer && npm install && npm run build</code> を実行してから、アプリを再起動してください。</p>
        </div>
        """
    return f"""
    <iframe
      title="PDF / 画像レイアウト比較ビューア"
      src="/viewer/?run_id={run_id}"
      style="width: 100%; height: 86vh; border: 1px solid #d7dde5; border-radius: 8px; background: #fff;"
    ></iframe>
    """


def _preview_placeholder_html() -> str:
    return (
        "<div style='padding: 16px; color: #475569;'>"
        "PDF または画像をアップロードするとプレビューを表示します。"
        "</div>"
    )


def _error_html(message: str) -> str:
    return f"""
    <div style="padding: 14px 16px; border: 1px solid #fecaca; color: #7f1d1d; background: #fef2f2; border-radius: 8px;">
      <strong>エラー</strong><br />
      {message}
    </div>
    """


def build_gradio_blocks(viewer_ready: bool):
    import gradio as gr

    settings = get_settings()
    label_to_engine = {ENGINE_LABELS[engine]: engine for engine in ENGINE_ORDER}
    choices = [ENGINE_LABELS[engine] for engine in ENGINE_ORDER]
    default_choices = []
    default_dpi = _preset_value_for_label(
        DPI_PRESETS,
        _preset_label_for_value(DPI_PRESETS, settings.render_dpi),
        settings.render_dpi,
    )
    default_dpi_preset = _preset_label_for_value(DPI_PRESETS, default_dpi)

    with gr.Blocks(title="PDF / 画像レイアウト比較ラボ") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                # PDF / 画像レイアウト比較ラボ
                PDF や画像を選択ページだけ解析し、技術ごとの差分を同じページ上の bbox で比較します。
                """,
                elem_classes=["main-title"],
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    pdf_file = gr.File(
                        label="PDF / 画像ファイル",
                        file_types=list(SUPPORTED_SOURCE_FILE_TYPES),
                        type="filepath",
                    )
                with gr.Column(scale=2):
                    engines = gr.CheckboxGroup(
                        label="解析エンジン",
                        choices=choices,
                        value=default_choices,
                    )
                    run_button = gr.Button("解析を実行", variant="primary")
                    summary = gr.Markdown("")
            page_range = gr.Textbox(value="1", visible=False)
            with gr.Row(elem_classes=["settings-row"]):
                with gr.Column(scale=2, min_width=360, elem_classes=["setting-card"]):
                    gr.Markdown(
                        """
                        **信頼度の下限**: 各解析エンジンがファイル上で見つけたテキスト行、表、図、タイトルなどのレイアウト要素に付ける確信度です。0〜1 で表し、下限を上げると誤検出は減りますが、小さな文字や薄い罫線の検出漏れが増える場合があります。
                        """,
                        elem_classes=["setting-help"],
                    )
                    min_confidence_preset = gr.Radio(
                        label="よく使われる信頼度",
                        choices=_preset_labels(CONFIDENCE_PRESETS),
                        value=_preset_label_for_value(CONFIDENCE_PRESETS, DEFAULT_MIN_CONFIDENCE),
                        elem_classes=["preset-radio"],
                    )
                    min_confidence = gr.Slider(
                        label="信頼度の下限（細かく調整）",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=DEFAULT_MIN_CONFIDENCE,
                    )
                with gr.Column(scale=2, min_width=360, elem_classes=["setting-card"]):
                    gr.Markdown(
                        """
                        **ページ画像 DPI**: PDF ページを画像化するときの解像度です。画像アップロード時は元画像の解像度を使います。
                        """,
                        elem_classes=["setting-help"],
                    )
                    dpi_preset = gr.Radio(
                        label="よく使われる DPI",
                        choices=_preset_labels(DPI_PRESETS),
                        value=default_dpi_preset,
                        elem_classes=["preset-radio"],
                    )
                    dpi = gr.Slider(
                        label="ページ画像 DPI（細かく調整）",
                        minimum=72,
                        maximum=300,
                        step=1,
                        value=default_dpi,
                    )
            viewer = gr.HTML(_preview_placeholder_html())

        def preview(pdf_file_value, dpi_value):
            if not pdf_file_value:
                return _preview_placeholder_html(), ""
            try:
                run_result = preview_pdf(
                    pdf_path=_file_path(pdf_file_value),
                    settings=get_settings(),
                    dpi=int(dpi_value or settings.render_dpi),
                )
                return (
                    _viewer_frame(run_result.run_id, viewer_ready),
                    summarize_preview(run_result),
                )
            except Exception as exc:
                return _error_html(str(exc)), f"### プレビューできませんでした\n\n{exc}"

        def run(pdf_file_value, page_range_value, selected_labels, confidence_value, dpi_value):
            try:
                source_path = _file_path(pdf_file_value)
                selected_labels = selected_labels or []
                selected_engine_ids = [label_to_engine[label] for label in selected_labels if label in label_to_engine]
                run_result = analyze_pdf(
                    pdf_path=source_path,
                    page_range=_page_range_value(page_range_value),
                    engine_ids=selected_engine_ids,
                    settings=get_settings(),
                    min_confidence=float(confidence_value or 0),
                    dpi=int(dpi_value or settings.render_dpi),
                )
                return (
                    _viewer_frame(run_result.run_id, viewer_ready),
                    summarize_run(run_result),
                )
            except Exception as exc:
                return _error_html(str(exc)), f"### 解析できませんでした\n\n{exc}"

        def select_confidence_preset(label):
            return _preset_value_for_label(CONFIDENCE_PRESETS, label, DEFAULT_MIN_CONFIDENCE)

        def sync_confidence_preset(value):
            return gr.update(value=_preset_label_for_value(CONFIDENCE_PRESETS, value))

        def select_dpi_preset(label):
            return _preset_value_for_label(DPI_PRESETS, label, default_dpi)

        def sync_dpi_preset(value):
            return gr.update(value=_preset_label_for_value(DPI_PRESETS, value))

        min_confidence_preset.change(
            fn=select_confidence_preset,
            inputs=min_confidence_preset,
            outputs=min_confidence,
        )
        min_confidence.change(
            fn=sync_confidence_preset,
            inputs=min_confidence,
            outputs=min_confidence_preset,
        )
        dpi_preset.change(
            fn=select_dpi_preset,
            inputs=dpi_preset,
            outputs=dpi,
        )
        dpi.change(
            fn=sync_dpi_preset,
            inputs=dpi,
            outputs=dpi_preset,
        )
        pdf_file.change(
            fn=preview,
            inputs=[pdf_file, dpi],
            outputs=[viewer, summary],
        )
        run_button.click(
            fn=run,
            inputs=[pdf_file, page_range, engines, min_confidence, dpi],
            outputs=[viewer, summary],
            js=RUN_PAGE_SELECTION_JS,
        )
    return demo


def create_app():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import gradio as gr

    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    viewer_dist = Path(__file__).resolve().parent.parent / "viewer" / "dist"
    app = FastAPI(title="PDF / 画像レイアウト比較ラボ")
    app.mount("/artifacts", StaticFiles(directory=str(settings.output_dir)), name="artifacts")
    viewer_ready = viewer_dist.exists()
    if viewer_ready:
        app.mount("/viewer", StaticFiles(directory=str(viewer_dist), html=True), name="viewer")
    blocks = build_gradio_blocks(viewer_ready)
    return gr.mount_gradio_app(app, blocks, path="/", css=APP_CSS, head=PAGE_SYNC_HEAD)


def main() -> None:
    exec_project_venv_if_available(Path(__file__).resolve().parent.parent)

    import uvicorn

    settings = get_settings()
    uvicorn.run(create_app(), host=settings.host, port=settings.port)
