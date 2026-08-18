import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pdf_layout_lab.adapters.mineru import (
    _records_from_dict_payload,
    _records_from_vlm_model_json,
    build_mineru_command,
    resolve_mineru_command,
)
from pdf_layout_lab.schemas import PageImage
from pdf_layout_lab.settings import get_settings


class MineruAdapterTests(unittest.TestCase):
    def setUp(self):
        self.page = PageImage(
            page=1,
            width=2480,
            height=3500,
            pdf_width=595.0,
            pdf_height=841.0,
            image_path="page.png",
        )

    def test_resolves_explicit_local_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            command_path = Path(tmp) / "mineru"
            command_path.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(command_path, 0o755)

            self.assertEqual(resolve_mineru_command(str(command_path)), [str(command_path)])

    def test_builds_cpu_pipeline_command_for_current_cli(self):
        settings = replace(get_settings(), mineru_backend="pipeline", mineru_extra_args="")

        command = build_mineru_command(["mineru"], Path("sample.pdf"), Path("out"), settings)

        self.assertEqual(command, ["mineru", "-p", "sample.pdf", "-o", "out", "-b", "pipeline", "-m", "auto"])

    def test_builds_legacy_magic_pdf_command(self):
        settings = replace(get_settings(), mineru_method="auto", mineru_extra_args="")

        command = build_mineru_command(["magic-pdf"], Path("sample.pdf"), Path("out"), settings)

        self.assertEqual(command, ["magic-pdf", "-p", "sample.pdf", "-o", "out", "-m", "auto"])

    def test_extracts_text_from_middle_json_line_spans(self):
        payload = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [595, 841],
                    "preproc_blocks": [
                        {
                            "score": 0.9,
                            "bbox": [134, 234, 445, 317],
                            "type": "title",
                            "lines": [
                                {"spans": [{"type": "text", "content": "三菱ＵＦＪダイレクト"}]},
                                {"spans": [{"type": "text", "content": "外国送金 入力マニュアル"}]},
                            ],
                        }
                    ],
                }
            ]
        }

        records = _records_from_dict_payload(payload, {1: self.page}, "mineru", 0.0)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "三菱ＵＦＪダイレクト\n外国送金 入力マニュアル")
        self.assertEqual(records[0].category, "Title")
        self.assertEqual(records[0].bbox, [558.5210084033613, 973.8406658739595, 1854.7899159663866, 1319.2627824019025])

    def test_extracts_text_from_content_list_v2_content_dict(self):
        payload = [
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "お手続きにあたり、以下の情報が必要となります。"}
                        ]
                    },
                    "bbox": [277, 554, 705, 574],
                }
            ]
        ]

        records = _records_from_vlm_model_json(payload, {1: self.page}, "mineru", 0.0)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "お手続きにあたり、以下の情報が必要となります。")
        self.assertEqual(records[0].category, "Text")
        self.assertEqual(records[0].bbox, [686.96, 1939.0, 1748.4, 2009.0])

    def test_imports_flat_content_list_json(self):
        payload = [
            {
                "type": "text",
                "text": "必ず弊行へデータを送信される前に、内容に相違がないか再度の確認をお願いします。",
                "bbox": [166, 810, 820, 829],
                "page_idx": 0,
            }
        ]

        records = _records_from_vlm_model_json(payload, {1: self.page}, "mineru", 0.0)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "必ず弊行へデータを送信される前に、内容に相違がないか再度の確認をお願いします。")
        self.assertEqual(records[0].bbox, [411.68, 2835.0, 2033.6, 2901.5])

    def test_imports_model_json_label_records_as_fallback(self):
        payload = [
            {
                "page_info": {"page_no": 0, "width": 1000, "height": 2000},
                "layout_dets": [
                    {
                        "label": "doc_title",
                        "score": 0.8,
                        "bbox": [10, 20, 30, 40],
                    }
                ],
            }
        ]

        records = _records_from_vlm_model_json(payload, {1: self.page}, "mineru", 0.0)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].category, "Title")
        self.assertEqual(records[0].bbox, [24.8, 35.0, 74.4, 70.0])


if __name__ == "__main__":
    unittest.main()


class MineruTableTextTests(unittest.TestCase):
    def test_table_block_uses_nested_table_html(self):
        page = PageImage(page=1, width=2480, height=3500, pdf_width=595.0, pdf_height=841.0, image_path="page.png")
        payload = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [595.0, 841.0],
                    "preproc_blocks": [
                        {
                            "type": "table",
                            "bbox": [88, 472, 496, 750],
                            "blocks": [
                                {
                                    "type": "table_caption",
                                    "bbox": [88, 472, 496, 490],
                                    "lines": [{"spans": [{"type": "text", "content": "表 2"}]}],
                                },
                                {
                                    "type": "table_body",
                                    "bbox": [88, 492, 496, 750],
                                    "lines": [{"spans": [{"type": "table", "html": "<table><tr><td>A</td></tr></table>"}]}],
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        records = _records_from_dict_payload(payload, {1: page}, "mineru", 0.0)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].text.startswith("<table>"))
        self.assertIn("表 2", records[0].text)
