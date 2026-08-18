from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pdf_layout_lab.schemas import LayoutRecord, PageImage
from pdf_layout_lab.settings import Settings


@dataclass(frozen=True)
class AdapterAvailability:
    available: bool
    message: str


@dataclass(frozen=True)
class AnalysisContext:
    pdf_path: Path
    run_dir: Path
    pages: list[PageImage]
    settings: Settings
    min_confidence: float = 0.0


class LayoutAdapter(Protocol):
    engine_id: str
    label: str

    def availability(self) -> AdapterAvailability:
        ...

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        ...


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def missing_dependency_message(package: str, install: str) -> str:
    return f"{package} が未インストールです。プロジェクト直下で `{install}` を実行してから再試行してください。"


def extra_install_command(extra: str) -> str:
    return f".venv/bin/pip install -e '.[{extra}]'"


def object_to_plain(value):
    """SDK モデルなどを JSON 保存しやすい素朴な dict/list へ変換する。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [object_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): object_to_plain(item) for key, item in value.items()}
    if hasattr(value, "swagger_types"):
        result = {}
        for key in getattr(value, "swagger_types", {}):
            result[key] = object_to_plain(getattr(value, key, None))
        return result
    if hasattr(value, "__dict__"):
        return {
            str(key).lstrip("_"): object_to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("__")
        }
    return str(value)
