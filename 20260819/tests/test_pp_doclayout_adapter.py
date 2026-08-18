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
