import unittest

from pdf_layout_lab.pages import parse_page_range


class PageRangeTests(unittest.TestCase):
    def test_parse_page_range(self):
        self.assertEqual(parse_page_range("1,3-5", 10), [1, 3, 4, 5])

    def test_parse_all_in_japanese(self):
        self.assertEqual(parse_page_range("すべて", 3), [1, 2, 3])

    def test_ignores_out_of_range_pages(self):
        self.assertEqual(parse_page_range("0,2,9", 3), [2])


if __name__ == "__main__":
    unittest.main()
