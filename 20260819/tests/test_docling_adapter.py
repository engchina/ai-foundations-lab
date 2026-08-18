import unittest
from dataclasses import replace

from pdf_layout_lab.adapters.docling_adapter import _build_docling_converter
from pdf_layout_lab.settings import get_settings


class DoclingAdapterTests(unittest.TestCase):
    def test_builds_cpu_converter(self):
        settings = replace(get_settings(), docling_device="cpu", docling_num_threads=2)

        converter = _build_docling_converter(settings)

        self.assertIsNotNone(converter)


if __name__ == "__main__":
    unittest.main()
