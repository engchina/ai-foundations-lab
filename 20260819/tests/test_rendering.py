import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pdf_layout_lab.rendering import prepare_source_for_analysis, source_file_kind


class SourceRenderingTests(unittest.TestCase):
    def test_source_file_kind_accepts_pdf_and_images(self):
        self.assertEqual(source_file_kind("sample.PDF"), "pdf")
        self.assertEqual(source_file_kind("scan.JPG"), "image")
        self.assertEqual(source_file_kind("scan.jfif"), "image")
        self.assertEqual(source_file_kind("diagram.gif"), "image")
        self.assertEqual(source_file_kind("page.webp"), "image")

    def test_source_file_kind_rejects_unsupported_extension(self):
        with self.assertRaises(ValueError):
            source_file_kind("notes.txt")

    def test_prepare_source_for_analysis_converts_image_to_single_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "scan.png"
            Image.new("RGBA", (80, 40), (20, 120, 200, 128)).save(source_path)

            source_pdf_path, pages = prepare_source_for_analysis(source_path, [1], tmp_path / "run", dpi=150)

            self.assertTrue(Path(source_pdf_path).exists())
            self.assertEqual(Path(source_pdf_path).name, "source.pdf")
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0].page, 1)
            self.assertEqual(pages[0].width, 80)
            self.assertEqual(pages[0].height, 40)
            self.assertTrue(Path(pages[0].image_path).exists())
            with Image.open(pages[0].image_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (80, 40))

    def test_prepare_source_for_analysis_rejects_image_page_other_than_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "scan.png"
            Image.new("RGB", (10, 10), "white").save(source_path)

            with self.assertRaises(ValueError):
                prepare_source_for_analysis(source_path, [2], tmp_path / "run", dpi=150)


if __name__ == "__main__":
    unittest.main()


class PyMuPdfAdapterTests(unittest.TestCase):
    def _scanned_pdf(self, directory: Path) -> Path:
        """1 ページ全面が 1 枚の画像だけ（テキスト無し）のスキャン PDF を作る。"""
        import pymupdf

        image_path = directory / "scan.png"
        Image.new("RGB", (600, 800), (230, 230, 230)).save(image_path)
        pdf_path = directory / "scanned.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page(width=595, height=842)
            page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(image_path))
            doc.save(str(pdf_path))
        return pdf_path

    def test_image_only_page_returns_picture_record(self):
        from pdf_layout_lab.adapters.base import AnalysisContext
        from pdf_layout_lab.adapters.pymupdf_adapter import PyMuPdfAdapter
        from pdf_layout_lab.schemas import PageImage
        from pdf_layout_lab.settings import get_settings

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pdf_path = self._scanned_pdf(directory)
            page = PageImage(page=1, width=1190, height=1684, pdf_width=595, pdf_height=842, image_path="page.png")
            settings = get_settings()
            records = PyMuPdfAdapter(settings).analyze(
                AnalysisContext(
                    pdf_path=pdf_path,
                    run_dir=directory,
                    pages=[page],
                    settings=settings,
                    min_confidence=0.5,
                )
            )

        self.assertEqual([record.category for record in records], ["Picture"])
