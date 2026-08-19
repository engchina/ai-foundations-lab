import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pdf_layout_lab.adapters.base import AnalysisContext
from pdf_layout_lab.adapters.unstructured_adapter import UnstructuredAdapter
from pdf_layout_lab.schemas import PageImage
from pdf_layout_lab.settings import get_settings


class UnstructuredAdapterTests(unittest.TestCase):
    def test_scales_pixel_space_to_page_image(self):
        # Unstructured は 350 DPI 相当 (2894x1930) の座標を返すが、ページ画像は 300 DPI (2481x1654)
        element = SimpleNamespace(
            text="発票号码",
            category="Header",
            metadata=SimpleNamespace(
                page_number=1,
                coordinates=SimpleNamespace(
                    points=((2484, 180), (2484, 233), (2894, 233), (2894, 180)),
                    system=SimpleNamespace(width=2894, height=1930),
                ),
                text_as_html=None,
            ),
        )
        fake_module = types.ModuleType("unstructured.partition.pdf")
        fake_module.partition_pdf = lambda **kwargs: [element]
        page = PageImage(page=1, width=2481, height=1654, pdf_width=595.28, pdf_height=396.85, image_path="page_0001.png")
        context = AnalysisContext(pdf_path=Path("dummy.pdf"), run_dir=Path("."), pages=[page], settings=get_settings())
        adapter = UnstructuredAdapter(get_settings())

        with patch.dict(sys.modules, {"unstructured": types.ModuleType("unstructured"), "unstructured.partition": types.ModuleType("unstructured.partition"), "unstructured.partition.pdf": fake_module}), \
             patch.object(adapter, "_register_model_cache"):
            records = adapter.analyze(context)

        self.assertEqual(len(records), 1)
        left, top, right, bottom = records[0].bbox
        self.assertAlmostEqual(left, 2484 * 2481 / 2894, places=3)
        self.assertAlmostEqual(top, 180 * 1654 / 1930, places=3)
        self.assertAlmostEqual(right, 2481.0, places=3)
        self.assertAlmostEqual(bottom, 233 * 1654 / 1930, places=3)


if __name__ == "__main__":
    unittest.main()
