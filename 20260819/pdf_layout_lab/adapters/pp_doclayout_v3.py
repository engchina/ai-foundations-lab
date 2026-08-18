from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import clamp_bbox
from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, extra_install_command, has_module, missing_dependency_message


class PpDocLayoutV3Adapter:
    engine_id = "pp_doclayout_v3"
    label = "PP-DocLayoutV3"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if has_module("paddleocr"):
            return AdapterAvailability(True, f"PaddleOCR `{self.settings.pp_doclayout_engine}` engine で PP-DocLayoutV3 を CPU 実行します。")
        if has_module("transformers") and has_module("torch") and _transformers_supports_pp_doclayout_v3():
            return AdapterAvailability(True, "Transformers 経由で PP-DocLayoutV3 を実行します。")
        return AdapterAvailability(
            False,
            missing_dependency_message(
                "PP-DocLayoutV3",
                f"{extra_install_command('pp-doclayout')} または {extra_install_command('paddle')}",
            ),
        )

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        if has_module("paddleocr"):
            return self._analyze_paddleocr(context)
        if has_module("transformers") and has_module("torch") and _transformers_supports_pp_doclayout_v3():
            return self._analyze_transformers(context)
        raise RuntimeError(self.availability().message)

    def _analyze_transformers(self, context: AnalysisContext) -> list[LayoutRecord]:
        _prepare_runtime_env(context, self.settings)
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        model_name = self.settings.pp_doclayout_model
        if model_name == "PP-DocLayoutV3":
            model_name = "PaddlePaddle/PP-DocLayoutV3_safetensors"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForObjectDetection.from_pretrained(model_name)
        model.eval()
        records: list[LayoutRecord] = []
        for page in context.pages:
            image = Image.open(page.image_path).convert("RGB")
            inputs = processor(images=[image], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs,
                threshold=max(context.min_confidence, 0.0),
                target_sizes=torch.tensor([image.size[::-1]]),
            )[0]
            for idx, (score, label_id, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"]), start=1):
                confidence = float(score.item())
                if confidence < context.min_confidence:
                    continue
                label = model.config.id2label[int(label_id.item())]
                records.append(
                    LayoutRecord(
                        id=f"ppdoclayout-p{page.page}-{idx}",
                        engine=self.engine_id,
                        page=page.page,
                        seq_no=idx,
                        bbox=clamp_bbox([float(v) for v in box.tolist()], page.width, page.height),
                        coord_system="image_top_left",
                        page_width=page.width,
                        page_height=page.height,
                        category=normalize_category(label),
                        text="",
                        confidence=confidence,
                        raw_type=str(label),
                        raw={"label": label, "score": confidence},
                    )
                )
        return records

    def _analyze_paddleocr(self, context: AnalysisContext) -> list[LayoutRecord]:
        _prepare_runtime_env(context, self.settings)
        from paddleocr import LayoutDetection

        model = LayoutDetection(model_name=self.settings.pp_doclayout_model, engine=self.settings.pp_doclayout_engine)
        records: list[LayoutRecord] = []
        for page in context.pages:
            output = model.predict(input=str(Path(page.image_path)), batch_size=1, layout_nms=True)
            idx = 1
            for result in output:
                payload = _result_to_payload(result)
                for item in _iter_layout_items(payload):
                    confidence = _optional_float(item.get("score") or item.get("confidence"))
                    if confidence is not None and confidence < context.min_confidence:
                        continue
                    bbox = item.get("coordinate") or item.get("bbox") or item.get("box")
                    if not bbox:
                        continue
                    label = item.get("label") or item.get("category") or item.get("type")
                    records.append(
                        LayoutRecord(
                            id=f"ppdoclayout-p{page.page}-{idx}",
                            engine=self.engine_id,
                            page=page.page,
                            seq_no=idx,
                            bbox=clamp_bbox(bbox, page.width, page.height),
                            coord_system="image_top_left",
                            page_width=page.width,
                            page_height=page.height,
                            category=normalize_category(label),
                            text="",
                            confidence=confidence,
                            raw_type=str(label or ""),
                            raw=dict(item),
                        )
                    )
                    idx += 1
        return records


def _transformers_supports_pp_doclayout_v3() -> bool:
    try:
        import transformers
    except Exception:
        return False
    if hasattr(transformers, "PPDocLayoutV3ForObjectDetection"):
        return True
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        return "pp_doclayout_v3" in CONFIG_MAPPING_NAMES
    except Exception:
        return False


def _prepare_runtime_env(context: AnalysisContext, settings: Settings) -> None:
    cache_root = context.settings.output_dir / "_cache" / "pp_doclayout_v3"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root / "paddlex"))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", settings.pp_doclayout_model_source)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("ONNXRUNTIME_DISABLE_TELEMETRY", "1")


def _result_to_payload(result: Any) -> dict[str, Any]:
    raw_json = getattr(result, "json", None)
    if callable(raw_json):
        raw_json = raw_json()
    if isinstance(raw_json, dict):
        if isinstance(raw_json.get("res"), dict):
            return raw_json["res"]
        return raw_json
    raw_res = getattr(result, "res", None)
    if isinstance(raw_res, dict):
        return raw_res
    if isinstance(result, dict):
        return result
    return {}


def _iter_layout_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("boxes") or payload.get("results") or payload.get("layout_dets") or payload.get("objects") or []
    if isinstance(items, dict):
        items = [items]
    return [item for item in items if isinstance(item, dict)]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
