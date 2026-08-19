import json
import unittest
from pathlib import Path

from pdf_layout_lab.adapters.base import AnalysisContext
from pdf_layout_lab.adapters.mineru_api import records_from_file_parse_response
from pdf_layout_lab.schemas import PageImage
from pdf_layout_lab.settings import get_settings


class MineruApiAdapterTests(unittest.TestCase):
    def test_parses_middle_json_string_and_scales_bbox(self):
        # /file_parse は middle_json を JSON 文字列で返す。page_size は pt、ページ画像は 2 倍
        middle_json = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [500, 400],
                    "para_blocks": [
                        {"type": "title", "bbox": [50, 40, 250, 80], "lines": [{"spans": [{"type": "text", "content": "见出し"}]}]}
                    ],
                }
            ]
        }
        payload = {"error": None, "results": {"source": {"middle_json": json.dumps(middle_json)}}}
        page = PageImage(page=1, width=1000, height=800, pdf_width=500, pdf_height=400, image_path="page.png")
        context = AnalysisContext(pdf_path=Path("dummy.pdf"), run_dir=Path("."), pages=[page], settings=get_settings())

        records = records_from_file_parse_response(payload, context)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].engine, "mineru_api")
        self.assertEqual(records[0].id, "mineru_api-p1-1")
        self.assertEqual(records[0].bbox, [100.0, 80.0, 500.0, 160.0])
        self.assertEqual(records[0].text, "见出し")

    def test_raises_on_server_error(self):
        page = PageImage(page=1, width=10, height=10, pdf_width=10, pdf_height=10, image_path="page.png")
        context = AnalysisContext(pdf_path=Path("dummy.pdf"), run_dir=Path("."), pages=[page], settings=get_settings())
        with self.assertRaises(RuntimeError):
            records_from_file_parse_response({"error": "boom", "results": {}}, context)


if __name__ == "__main__":
    unittest.main()
