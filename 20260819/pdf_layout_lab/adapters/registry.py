from __future__ import annotations

from pdf_layout_lab.settings import Settings

from .base import LayoutAdapter
from .docling_adapter import DoclingAdapter
from .dots_mocr import DotsMocrAdapter
from .mineru import MineruAdapter
from .oci_document import OciDocumentAdapter
from .pp_doclayout_v3 import PpDocLayoutV3Adapter
from .pymupdf_adapter import PyMuPdfAdapter
from .unstructured_adapter import UnstructuredAdapter
from .yolov10_adapter import YoloV10Adapter

ENGINE_ORDER = [
    "oci",
    "mineru",
    "dots_mocr",
    "unstructured",
    "docling",
    "pymupdf",
    "yolov10",
    "pp_doclayout_v3",
]

ENGINE_LABELS = {
    "oci": "OCI Document Understanding",
    "dots_mocr": "dots.mocr",
    "unstructured": "Unstructured",
    "docling": "Docling",
    "pp_doclayout_v3": "PP-DocLayoutV3",
    "mineru": "MinerU / MinerU2.5-Pro",
    "pymupdf": "PyMuPDF",
    "yolov10": "YOLOv10 DocLayNet",
}


def build_adapters(settings: Settings) -> dict[str, LayoutAdapter]:
    adapters: dict[str, LayoutAdapter] = {
        "oci": OciDocumentAdapter(settings),
        "mineru": MineruAdapter(settings),
        "dots_mocr": DotsMocrAdapter(settings),
        "unstructured": UnstructuredAdapter(settings),
        "docling": DoclingAdapter(settings),
        "pymupdf": PyMuPdfAdapter(settings),
        "yolov10": YoloV10Adapter(settings),
        "pp_doclayout_v3": PpDocLayoutV3Adapter(settings),
    }
    enabled = set(ENGINE_ORDER if "all" in settings.enabled_engines else settings.enabled_engines)
    return {engine_id: adapters[engine_id] for engine_id in ENGINE_ORDER if engine_id in enabled}
