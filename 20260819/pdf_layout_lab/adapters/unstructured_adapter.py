from __future__ import annotations

from pdf_layout_lab import model_pool
from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, extra_install_command, has_module, missing_dependency_message, object_to_plain


class UnstructuredAdapter:
    engine_id = "unstructured"
    label = "Unstructured"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if not has_module("unstructured"):
            return AdapterAvailability(False, missing_dependency_message("Unstructured", extra_install_command("unstructured")))
        return AdapterAvailability(True, "high-res partition でレイアウト要素を抽出します。")

    def preload(self) -> None:
        """hi_res 用レイアウトモデルを事前にロードして常駐させる（UI の「ロード」ボタン用）。"""
        self._register_model_cache()
        from unstructured_inference.models.base import get_model

        get_model()

    def _register_model_cache(self) -> None:
        # unstructured_inference はモデルを自前の dict に常駐させるので、解放時にその dict を空にする
        def clear_cache() -> None:
            from unstructured_inference.models import base

            base.models.clear()

        model_pool.get(self.engine_id, lambda: "unstructured_inference", unloader=clear_cache)

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        from unstructured.partition.pdf import partition_pdf

        self._register_model_cache()
        page_numbers = [page.page for page in context.pages]
        elements = partition_pdf(
            filename=str(context.pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
            include_page_breaks=False,
            languages=self.settings.unstructured_ocr_languages,  # Tesseract は先頭を主モデルにするので順序が効く
        )
        page_lookup = {page.page: page for page in context.pages}
        counters: dict[int, int] = {}
        records: list[LayoutRecord] = []
        for element in elements:
            metadata = getattr(element, "metadata", None)
            page_number = int(getattr(metadata, "page_number", 0) or 0)
            if page_number not in page_numbers:
                continue
            page = page_lookup[page_number]
            coords = getattr(metadata, "coordinates", None)
            points = getattr(coords, "points", None)
            if not points:
                continue
            # Unstructured は自前の DPI で描画した PixelSpace の座標を返すので、こちらのページ画像サイズへ合わせる
            system = getattr(coords, "system", None)
            x_scale = page.width / float(getattr(system, "width", 0) or page.width)
            y_scale = page.height / float(getattr(system, "height", 0) or page.height)
            xs = [float(point[0]) * x_scale for point in points]
            ys = [float(point[1]) * y_scale for point in points]
            counters[page_number] = counters.get(page_number, 0) + 1
            confidence = _optional_float(getattr(element, "detection_score", None))
            if confidence is not None and confidence < context.min_confidence:
                continue
            text_as_html = getattr(metadata, "text_as_html", None)
            text = str(text_as_html or getattr(element, "text", "") or "")
            records.append(
                LayoutRecord(
                    id=f"unstructured-p{page_number}-{counters[page_number]}",
                    engine=self.engine_id,
                    page=page_number,
                    seq_no=counters[page_number],
                    bbox=clamp_bbox([min(xs), min(ys), max(xs), max(ys)], page.width, page.height),
                    coord_system="image_top_left",
                    page_width=page.width,
                    page_height=page.height,
                    category=normalize_category(getattr(element, "category", type(element).__name__)),
                    text=text,
                    confidence=confidence,
                    raw_type=str(getattr(element, "category", type(element).__name__)),
                    raw=object_to_plain(element),
                )
            )
        return records


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
