import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pdf_layout_lab.analysis import preview_pdf
from pdf_layout_lab.schemas import PageImage
from pdf_layout_lab.settings import get_settings


class PreviewPdfTests(unittest.TestCase):
    def test_preview_renders_all_pages_without_records_or_engines(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "manual.pdf"
            pdf_path.write_bytes(b"%PDF-preview")
            settings = replace(get_settings(), output_dir=temp_path / "runs", render_dpi=200)

            def fake_prepare(source_path, page_numbers, output_dir, dpi):
                pages_dir = Path(output_dir) / "pages"
                pages_dir.mkdir(parents=True, exist_ok=True)
                pages = [
                    PageImage(
                        page=page,
                        width=100,
                        height=200,
                        pdf_width=50,
                        pdf_height=100,
                        image_path=str(pages_dir / f"page_{page:04d}.png"),
                    )
                    for page in page_numbers
                ]
                return str(Path(output_dir) / "source.pdf"), pages

            with (
                patch("pdf_layout_lab.analysis.get_source_page_count", return_value=3),
                patch("pdf_layout_lab.analysis.prepare_source_for_analysis", side_effect=fake_prepare) as prepare,
            ):
                run = preview_pdf(pdf_path, settings, dpi=144)

            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[1], [1, 2, 3])
            self.assertEqual(prepare.call_args.args[3], 144)
            self.assertEqual([page.page for page in run.pages], [1, 2, 3])
            self.assertEqual(run.records, [])
            self.assertEqual(run.statuses, [])
            self.assertTrue(Path(run.viewer_data_path).exists())
            self.assertEqual(Path(run.jsonl_path).read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
