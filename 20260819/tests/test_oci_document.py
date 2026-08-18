import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from pdf_layout_lab.adapters.oci_document import OciDocumentAdapter, _chunk_pages, _language_or_none, _optional_float, _page_ranges_from_pages
from pdf_layout_lab.adapters.base import AnalysisContext
from pdf_layout_lab.schemas import PageImage
from pdf_layout_lab.settings import get_settings


class OciDocumentTests(unittest.TestCase):
    def test_single_page_uses_single_page_syntax(self):
        self.assertEqual(_page_ranges_from_pages([1]), ["1"])

    def test_contiguous_pages_are_compacted(self):
        self.assertEqual(_page_ranges_from_pages([1, 2, 3, 5]), ["1-3", "5"])

    def test_invalid_and_duplicate_pages_are_ignored(self):
        self.assertEqual(_page_ranges_from_pages([0, 2, 2, 1]), ["1-2"])

    def test_pages_are_chunked_for_sync_api_limit(self):
        pages = [
            PageImage(page=index, width=100, height=100, pdf_width=50, pdf_height=50, image_path=f"{index}.png")
            for index in range(1, 7)
        ]
        chunks = _chunk_pages(pages, 5)
        self.assertEqual([[page.page for page in chunk] for chunk in chunks], [[1, 2, 3, 4, 5], [6]])

    def test_auto_language_is_omitted(self):
        self.assertIsNone(_language_or_none("auto"))
        self.assertIsNone(_language_or_none(""))
        self.assertEqual(_language_or_none("JA"), "JA")

    def test_negative_oci_confidence_is_missing_value(self):
        self.assertIsNone(_optional_float(-1.0))
        self.assertEqual(_optional_float(0.75), 0.75)

    def test_inline_request_omits_page_range_after_subset_pdf_is_written(self):
        captured = {}
        fake_oci = _FakeOciModule(captured)
        settings = replace(get_settings(), oci_compartment_id="ocid1.compartment.oc1..example")
        pages = [
            PageImage(page=3, width=100, height=100, pdf_width=50, pdf_height=50, image_path="page.png"),
        ]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subset_pdf = temp_path / "subset.pdf"
            subset_pdf.write_bytes(b"%PDF-1.7\n")
            context = AnalysisContext(
                pdf_path=temp_path / "source.pdf",
                run_dir=temp_path,
                pages=pages,
                settings=settings,
            )

            with patch.dict("sys.modules", {"oci": fake_oci}):
                with patch("pdf_layout_lab.adapters.oci_document._load_oci_config", return_value={}):
                    with patch("pdf_layout_lab.adapters.oci_document._write_subset_pdf", return_value=subset_pdf):
                        OciDocumentAdapter(settings).analyze(context)

        document = captured["details"].document
        self.assertEqual(document.source, "INLINE")
        self.assertEqual(document.data, "JVBERi0xLjcK")
        self.assertIsNone(getattr(document, "page_range", None))

    def test_large_subset_pdf_falls_back_to_single_page_image(self):
        captured = {}
        fake_oci = _FakeOciModule(captured)
        settings = replace(get_settings(), oci_compartment_id="ocid1.compartment.oc1..example")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            page_image = temp_path / "page.png"
            page_image.write_bytes(b"abc")
            subset_pdf = temp_path / "oversized.pdf"
            subset_pdf.write_bytes(b"12345")
            context = AnalysisContext(
                pdf_path=temp_path / "source.pdf",
                run_dir=temp_path,
                pages=[
                    PageImage(page=1, width=100, height=100, pdf_width=50, pdf_height=50, image_path=str(page_image)),
                ],
                settings=settings,
            )

            with patch.dict("sys.modules", {"oci": fake_oci}):
                with patch("pdf_layout_lab.adapters.oci_document.OCI_SYNC_MAX_BYTES", 4):
                    with patch("pdf_layout_lab.adapters.oci_document._load_oci_config", return_value={}):
                        with patch("pdf_layout_lab.adapters.oci_document._write_subset_pdf", return_value=subset_pdf):
                            OciDocumentAdapter(settings).analyze(context)

        self.assertEqual(captured["details"].document.data, "YWJj")


class _FakeDocument:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeClient:
    def __init__(self, captured):
        self.captured = captured

    def analyze_document(self, analyze_document_details):
        self.captured["details"] = analyze_document_details
        return SimpleNamespace(data={"pages": []})


class _FakeOciModule:
    def __init__(self, captured):
        self.config = SimpleNamespace(from_file=lambda *_args, **_kwargs: {})
        self.ai_document = SimpleNamespace(
            AIServiceDocumentClient=lambda _config: _FakeClient(captured),
            models=SimpleNamespace(
                AnalyzeDocumentDetails=_FakeDocument,
                InlineDocumentDetails=_FakeDocument,
                DocumentTextExtractionFeature=_FakeDocument,
                DocumentTableExtractionFeature=_FakeDocument,
            ),
        )


if __name__ == "__main__":
    unittest.main()
