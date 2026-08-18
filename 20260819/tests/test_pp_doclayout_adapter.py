import sys
import types
import unittest

from pdf_layout_lab.adapters.pp_doclayout_v3 import _iter_layout_items, _result_to_payload


class DummyResult:
    def __init__(self, payload):
        self.res = payload


class PpDocLayoutAdapterTests(unittest.TestCase):
    def test_result_payload_accepts_res_dict(self):
        payload = {"boxes": [{"label": "Text", "coordinate": [1, 2, 3, 4]}]}

        self.assertEqual(_result_to_payload(DummyResult(payload)), payload)

    def test_result_payload_unwraps_json_res(self):
        class JsonResult:
            def json(self):
                return {"res": {"boxes": [{"label": "Text", "coordinate": [1, 2, 3, 4]}]}}

        self.assertIn("boxes", _result_to_payload(JsonResult()))

    def test_iter_layout_items_accepts_common_keys(self):
        payload = {"layout_dets": [{"label": "Table", "bbox": [1, 2, 3, 4]}]}

        self.assertEqual(_iter_layout_items(payload)[0]["label"], "Table")


if __name__ == "__main__":
    unittest.main()


class PpDocLayoutDeviceTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu_without_cuda_provider(self):
        from unittest.mock import patch
        from dataclasses import replace

        from pdf_layout_lab.adapters.pp_doclayout_v3 import _resolve_paddleocr_device
        from pdf_layout_lab.settings import get_settings

        settings = replace(get_settings(), pp_doclayout_device="auto", pp_doclayout_engine="onnxruntime")
        fake_ort = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        with patch("pdf_layout_lab.adapters.pp_doclayout_v3.has_module", return_value=True):
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                self.assertEqual(_resolve_paddleocr_device(settings), "cpu")
        fake_ort_gpu = types.SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with patch("pdf_layout_lab.adapters.pp_doclayout_v3.has_module", return_value=True):
            with patch.dict(sys.modules, {"onnxruntime": fake_ort_gpu}):
                self.assertEqual(_resolve_paddleocr_device(settings), "gpu")
        self.assertEqual(_resolve_paddleocr_device(replace(settings, pp_doclayout_device="cpu")), "cpu")
        self.assertIsNone(_resolve_paddleocr_device(replace(settings, pp_doclayout_engine="paddle_static")))
