from __future__ import annotations

from pathlib import Path

from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox
from pdf_layout_lab.pymupdf_compat import import_pymupdf
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, has_module, missing_dependency_message


class PyMuPdfAdapter:
    engine_id = "pymupdf"
    label = "PyMuPDF"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if not has_module("pymupdf") and not has_module("fitz"):
            return AdapterAvailability(False, missing_dependency_message("PyMuPDF", ".venv/bin/pip install pymupdf"))
        return AdapterAvailability(True, "利用可能です。PDF 内部の text/image block を抽出します。")

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        fitz = import_pymupdf()

        records: list[LayoutRecord] = []
        page_lookup = {page.page: page for page in context.pages}
        with fitz.open(str(context.pdf_path)) as doc:
            for page_number in sorted(page_lookup):
                page_info = page_lookup[page_number]
                page = doc.load_page(page_number - 1)
                # 既定の flags は画像ブロックを落とすため、スキャン PDF が 0 件になる
                flags = fitz.TEXTFLAGS_BLOCKS | fitz.TEXT_PRESERVE_IMAGES
                blocks = page.get_text("blocks", sort=True, flags=flags)
                for block_index, block in enumerate(blocks, start=1):
                    x1, y1, x2, y2 = block[:4]
                    text = str(block[4] or "").strip()
                    block_type = int(block[6]) if len(block) > 6 else 0
                    category = "Picture" if block_type == 1 else normalize_category("Text")
                    bbox = [
                        float(x1) * page_info.width / page_info.pdf_width,
                        float(y1) * page_info.height / page_info.pdf_height,
                        float(x2) * page_info.width / page_info.pdf_width,
                        float(y2) * page_info.height / page_info.pdf_height,
                    ]
                    if not text and category != "Picture":
                        continue
                    records.append(
                        LayoutRecord(
                            id=f"pymupdf-p{page_number}-{block_index}",
                            engine=self.engine_id,
                            page=page_number,
                            seq_no=block_index,
                            bbox=clamp_bbox(bbox, page_info.width, page_info.height),
                            coord_system="image_top_left",
                            page_width=page_info.width,
                            page_height=page_info.height,
                            category=category,
                            text=text,
                            confidence=None,
                            raw_type=f"block_type_{block_type}",
                            raw={"block": list(block[:7])},
                        )
                    )
        return records


def page_image_path(page_image) -> Path:
    return Path(page_image.image_path)
