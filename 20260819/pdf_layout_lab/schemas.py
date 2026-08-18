from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .categories import CATEGORY_STYLES

CoordSystem = Literal["image_top_left", "pdf_bottom_left", "normalized_top_left"]


@dataclass
class PageImage:
    page: int
    width: int
    height: int
    pdf_width: float
    pdf_height: float
    image_path: str
    image_url: str = ""


@dataclass
class LayoutRecord:
    id: str
    engine: str
    page: int
    seq_no: int
    bbox: list[float]
    coord_system: CoordSystem
    page_width: float
    page_height: float
    category: str
    text: str = ""
    confidence: float | None = None
    raw_type: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineStatus:
    engine: str
    label: str
    available: bool
    message: str
    elapsed_seconds: float | None = None
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisRun:
    run_id: str
    pdf_name: str
    pdf_path: str
    pages: list[PageImage]
    records: list[LayoutRecord]
    statuses: list[EngineStatus]
    output_dir: str
    json_path: str
    jsonl_path: str
    viewer_data_path: str
    source_page_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pdf_name": self.pdf_name,
            "pdf_path": self.pdf_path,
            "pages": [asdict(page) for page in self.pages],
            "source_page_count": self.source_page_count,
            "records": [record.to_dict() for record in self.records],
            "statuses": [status.to_dict() for status in self.statuses],
            "output_dir": self.output_dir,
            "json_path": self.json_path,
            "jsonl_path": self.jsonl_path,
            "viewer_data_path": self.viewer_data_path,
        }

    def viewer_payload(self) -> dict[str, Any]:
        engines = []
        seen = set()
        for status in self.statuses:
            if status.engine in seen:
                continue
            seen.add(status.engine)
            engines.append(status.to_dict())
        return {
            "run_id": self.run_id,
            "pdf_name": self.pdf_name,
            "pages": [asdict(page) for page in self.pages],
            "source_page_count": self.source_page_count,
            "records": [record.to_dict() for record in self.records],
            "engines": engines,
            "category_styles": [asdict(style) for style in CATEGORY_STYLES],
        }


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
