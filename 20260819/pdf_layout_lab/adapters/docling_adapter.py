from __future__ import annotations

import os
from typing import Any

from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox, pdf_bottom_left_to_image_top_left
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, extra_install_command, has_module, missing_dependency_message, object_to_plain


class DoclingAdapter:
    engine_id = "docling"
    label = "Docling"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if not has_module("docling"):
            return AdapterAvailability(False, missing_dependency_message("Docling", extra_install_command("docling")))
        return AdapterAvailability(True, f"Docling を `{self.settings.docling_device}` device で実行し、provenance bbox を読み取ります。")

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        _prepare_docling_env(context, self.settings)
        converter = _build_docling_converter(self.settings)
        result = converter.convert(str(context.pdf_path))
        document = result.document
        if hasattr(document, "export_to_dict"):
            payload = document.export_to_dict()
        elif hasattr(document, "model_dump"):
            payload = document.model_dump()
        else:
            payload = object_to_plain(document)
        return self._records_from_payload(payload, context)

    def _records_from_payload(self, payload: dict[str, Any], context: AnalysisContext) -> list[LayoutRecord]:
        page_lookup = {page.page: page for page in context.pages}
        counters: dict[int, int] = {}
        records: list[LayoutRecord] = []
        candidates = []
        for key in ("texts", "tables", "pictures", "groups", "forms", "key_value_items"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        for item in candidates:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or "text")
            text = str(item.get("text") or item.get("orig") or item.get("caption") or "")
            for provenance in item.get("prov", []) or []:
                page_number = int(provenance.get("page_no") or provenance.get("pageNo") or 0)
                page = page_lookup.get(page_number)
                if not page:
                    continue
                bbox_payload = provenance.get("bbox") or {}
                bbox = _docling_bbox_to_image(bbox_payload, page)
                if not bbox:
                    continue
                counters[page_number] = counters.get(page_number, 0) + 1
                records.append(
                    LayoutRecord(
                        id=f"docling-p{page_number}-{counters[page_number]}",
                        engine=self.engine_id,
                        page=page_number,
                        seq_no=counters[page_number],
                        bbox=bbox,
                        coord_system="image_top_left",
                        page_width=page.width,
                        page_height=page.height,
                        category=normalize_category(label),
                        text=text,
                        confidence=None,
                        raw_type=label,
                        raw=item,
                    )
                )
        return records


def _prepare_docling_env(context: AnalysisContext, settings: Settings) -> None:
    cache_root = context.settings.output_dir / "_cache" / "docling"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Docling と OCR / PyTorch 系のキャッシュをプロジェクト配下へ寄せる。
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(cache_root / "easyocr"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("DOCLING_DEVICE", settings.docling_device)
    os.environ.setdefault("DOCLING_NUM_THREADS", str(settings.docling_num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.docling_num_threads))
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("ONNXRUNTIME_DISABLE_TELEMETRY", "1")


def _build_docling_converter(settings: Settings):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    device_value = settings.docling_device.lower()
    device = AcceleratorDevice.CPU if device_value == "cpu" else device_value
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = settings.docling_do_ocr
    pipeline_options.do_table_structure = settings.docling_do_table_structure
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=settings.docling_num_threads, device=device)
    if settings.docling_do_table_structure:
        pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    if settings.docling_do_ocr:
        pipeline_options.ocr_options = RapidOcrOptions()
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})


def _docling_bbox_to_image(bbox: dict[str, Any], page) -> list[float] | None:
    if not bbox:
        return None
    left = bbox.get("l")
    top = bbox.get("t")
    right = bbox.get("r")
    bottom = bbox.get("b")
    if None in (left, top, right, bottom):
        return None
    origin = str(bbox.get("coord_origin") or bbox.get("coordOrigin") or "TOPLEFT").upper()
    raw = [float(left), float(top), float(right), float(bottom)]
    if origin == "BOTTOMLEFT":
        return pdf_bottom_left_to_image_top_left(raw, page.pdf_width, page.pdf_height, page.width, page.height)
    return clamp_bbox(
        [
            raw[0] * page.width / page.pdf_width,
            raw[1] * page.height / page.pdf_height,
            raw[2] * page.width / page.pdf_width,
            raw[3] * page.height / page.pdf_height,
        ],
        page.width,
        page.height,
    )
