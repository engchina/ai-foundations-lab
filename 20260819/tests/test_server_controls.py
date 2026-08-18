import os
import unittest
from unittest.mock import patch

from pdf_layout_lab.server import (
    CONFIDENCE_PRESETS,
    DEFAULT_MIN_CONFIDENCE,
    DPI_PRESETS,
    _page_range_value,
    _preset_label_for_value,
    _preset_labels,
    _preset_value_for_label,
)
from pdf_layout_lab.settings import get_settings


class ServerControlPresetTests(unittest.TestCase):
    def test_default_confidence_uses_standard_preset(self):
        self.assertEqual(DEFAULT_MIN_CONFIDENCE, 0.5)
        self.assertEqual(_preset_label_for_value(CONFIDENCE_PRESETS, DEFAULT_MIN_CONFIDENCE), "0.50 - 標準（既定）")

    def test_default_render_dpi_uses_ocr_standard_preset(self):
        with patch.dict(os.environ, {"PDF_LAYOUT_LAB_RENDER_DPI": "not-an-int"}):
            render_dpi = get_settings().render_dpi
        self.assertEqual(render_dpi, 300)
        self.assertEqual(_preset_label_for_value(DPI_PRESETS, render_dpi), "300 DPI - OCR 標準（既定）")

    def test_preset_labels_and_values_roundtrip(self):
        labels = _preset_labels(CONFIDENCE_PRESETS)
        self.assertIn("0.25 - 検出重視", labels)
        self.assertEqual(_preset_value_for_label(CONFIDENCE_PRESETS, "0.75 - 精度重視", 0.5), 0.75)
        self.assertEqual(_preset_value_for_label(DPI_PRESETS, "missing", 300), 300)

    def test_page_range_value_accepts_number_input(self):
        self.assertEqual(_page_range_value(1), "1")
        self.assertEqual(_page_range_value(2.0), "2")
        self.assertEqual(_page_range_value(None), "")

    def test_page_range_value_rejects_fractional_page(self):
        with self.assertRaises(ValueError):
            _page_range_value(1.5)


if __name__ == "__main__":
    unittest.main()
