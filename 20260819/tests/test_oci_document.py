import unittest

from pdf_layout_lab.adapters.oci_document import _chunk_pages, _language_or_none, _optional_float, _page_ranges_from_pages
from pdf_layout_lab.schemas import PageImage


class OciDocumentTests(unittest.TestCase):
    def test_single_page_uses_range_syntax(self):
        self.assertEqual(_page_ranges_from_pages([1]), ["1-1"])

    def test_contiguous_pages_are_compacted(self):
        self.assertEqual(_page_ranges_from_pages([1, 2, 3, 5]), ["1-3", "5-5"])

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


if __name__ == "__main__":
    unittest.main()
