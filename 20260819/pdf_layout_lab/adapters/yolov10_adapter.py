from __future__ import annotations

import os
from pathlib import Path

from pdf_layout_lab import model_pool
from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, extra_install_command, has_module, missing_dependency_message


DOCLAYNET_LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]


class YoloV10Adapter:
    engine_id = "yolov10"
    label = "YOLOv10 DocLayNet"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if not has_module("ultralytics"):
            return AdapterAvailability(False, missing_dependency_message("ultralytics", extra_install_command("yolo")))
        if not Path(self.settings.yolov10_model_path).expanduser().exists():
            return AdapterAvailability(False, f"YOLOv10_MODEL_PATH の重みが見つかりません: {self.settings.yolov10_model_path}")
        return AdapterAvailability(True, "DocLayNet fine-tuned YOLOv10 重みでレイアウト検出します。")

    def preload(self) -> None:
        """モデルを事前にロードして常駐させる（UI の「ロード」ボタン用）。"""
        self._model(self.settings.output_dir / "_cache" / "yolov10")

    def _model(self, config_root: Path):
        yolo_config_dir = config_root / "ultralytics"
        matplotlib_config_dir = config_root / "matplotlib"
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
        model_path = str(Path(self.settings.yolov10_model_path).expanduser())

        def load():
            from ultralytics import YOLO

            return YOLO(model_path)

        return model_pool.get(self.engine_id, load, signature=model_path)

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        model = self._model(context.run_dir)
        records: list[LayoutRecord] = []
        for page in context.pages:
            results = model.predict(source=page.image_path, conf=max(context.min_confidence, 0.2), iou=0.8, verbose=False)
            boxes = results[0].boxes if results else []
            for idx, box in enumerate(boxes, start=1):
                confidence = float(box.conf.item())
                cls_index = int(box.cls.item())
                label = DOCLAYNET_LABELS[cls_index] if cls_index < len(DOCLAYNET_LABELS) else str(cls_index)
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                records.append(
                    LayoutRecord(
                        id=f"yolov10-p{page.page}-{idx}",
                        engine=self.engine_id,
                        page=page.page,
                        seq_no=idx,
                        bbox=clamp_bbox(xyxy, page.width, page.height),
                        coord_system="image_top_left",
                        page_width=page.width,
                        page_height=page.height,
                        category=normalize_category(label),
                        text="",
                        confidence=confidence,
                        raw_type=label,
                        raw={"label": label, "score": confidence},
                    )
                )
        return records
