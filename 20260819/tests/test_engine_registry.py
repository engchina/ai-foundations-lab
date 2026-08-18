from dataclasses import replace
import unittest

from pdf_layout_lab.adapters import ENGINE_LABELS, ENGINE_ORDER, build_adapters
from pdf_layout_lab.settings import get_settings


class EngineRegistryTests(unittest.TestCase):
    def test_mineru_follows_oci_engine_choice(self):
        self.assertEqual(ENGINE_ORDER[:2], ["oci", "mineru"])
        self.assertEqual(
            [ENGINE_LABELS[engine] for engine in ENGINE_ORDER[:2]],
            ["OCI Document Understanding", "MinerU / MinerU2.5-Pro"],
        )

    def test_pp_doclayout_is_last_engine_choice(self):
        self.assertEqual(ENGINE_ORDER[-1], "pp_doclayout_v3")
        self.assertEqual([ENGINE_LABELS[engine] for engine in ENGINE_ORDER][-1], "PP-DocLayoutV3")

    def test_build_adapters_follows_engine_order(self):
        settings = replace(get_settings(), enabled_engines=["all"])

        self.assertEqual(list(build_adapters(settings)), ENGINE_ORDER)


if __name__ == "__main__":
    unittest.main()
