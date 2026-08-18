import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

from pdf_layout_lab.adapters.dots_mocr import (
    DotsMocrAdapter,
    _flash_attn_import_fallback,
    _LocalDotsMocrRuntime,
    _cast_floating_tensors_to_dtype,
    _patch_vision_tower_forward,
    _resolve_attn_implementation,
    coerce_layout_elements,
    extract_first_json,
)
from pdf_layout_lab.settings import get_settings


class DotsMocrParsingTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        payload = extract_first_json('```json\n{"elements": [{"bbox": [1, 2, 3, 4], "category": "Text"}]}\n```')
        self.assertEqual(payload["elements"][0]["category"], "Text")

    def test_accepts_flexible_top_level_keys(self):
        elements = coerce_layout_elements({"layout": [{"bbox": [1, 2, 3, 4], "type": "Table"}]})
        self.assertEqual(elements[0]["type"], "Table")

    def test_transformers_backend_reports_missing_local_dependencies(self):
        with patch.dict("os.environ", {"DOTS_MOCR_BACKEND": "transformers"}):
            settings = get_settings()
        with patch("pdf_layout_lab.adapters.dots_mocr.has_module", return_value=False):
            availability = DotsMocrAdapter(settings).availability()
        self.assertFalse(availability.available)
        self.assertIn("ローカル実行", availability.message)
        self.assertIn(".[dots]", availability.message)

    def test_explicit_flash_attention_reports_missing_flash_attn(self):
        with patch.dict(
            "os.environ",
            {"DOTS_MOCR_BACKEND": "transformers", "DOTS_MOCR_ATTN_IMPLEMENTATION": "flash_attention_2"},
        ):
            settings = get_settings()
        with patch("pdf_layout_lab.adapters.dots_mocr.has_module", side_effect=lambda name: name != "flash_attn"):
            availability = DotsMocrAdapter(settings).availability()
        self.assertFalse(availability.available)
        self.assertIn("flash_attn", availability.message)
        self.assertIn("DOTS_MOCR_ATTN_IMPLEMENTATION=sdpa", availability.message)


class DotsMocrRuntimeTests(unittest.TestCase):
    def test_auto_attention_falls_back_to_sdpa_without_flash_attn(self):
        self.assertEqual(
            _resolve_attn_implementation("auto", flash_attn_available=False, device="cpu"),
            "sdpa",
        )
        self.assertIsNone(_resolve_attn_implementation("auto", flash_attn_available=True, device="cuda"))

    def test_explicit_flash_attention_requires_flash_attn(self):
        with self.assertRaisesRegex(RuntimeError, "flash_attn"):
            _resolve_attn_implementation("flash_attention_2", flash_attn_available=False, device="cuda")

    def test_flash_attn_stub_is_temporary(self):
        previous = sys.modules.pop("flash_attn", None)
        try:
            with _flash_attn_import_fallback("sdpa", flash_attn_available=False):
                self.assertIn("flash_attn", sys.modules)
                self.assertTrue(hasattr(sys.modules["flash_attn"], "flash_attn_varlen_func"))
            self.assertNotIn("flash_attn", sys.modules)
        finally:
            if previous is not None:
                sys.modules["flash_attn"] = previous

    def test_runtime_sets_vision_config_to_sdpa_when_stubbing_flash_attn(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
        fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        fake_torch.bfloat16 = object()
        fake_torch.float16 = object()
        fake_torch.float32 = object()

        fake_qwen = types.ModuleType("qwen_vl_utils")
        fake_qwen.process_vision_info = lambda messages: ([], [])

        captured: dict[str, object] = {}

        class FakeModel:
            def to(self, device):
                captured["to_device"] = device
                return self

            def eval(self):
                captured["eval_called"] = True
                return self

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                captured["config_model_name"] = model_name
                captured["config_kwargs"] = kwargs
                return types.SimpleNamespace(vision_config={"attn_implementation": "flash_attention_2"})

        class FakeAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                captured["model_name"] = model_name
                captured["model_kwargs"] = kwargs
                captured["flash_stub_visible"] = "flash_attn" in sys.modules
                return FakeModel()

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                captured["processor_model_name"] = model_name
                captured["processor_kwargs"] = kwargs
                return object()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = FakeAutoConfig
        fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
        fake_transformers.AutoProcessor = FakeAutoProcessor

        previous_flash = sys.modules.pop("flash_attn", None)
        settings = replace(
            get_settings(),
            dots_mocr_model="fake-dots-mocr",
            dots_mocr_device="cpu",
            dots_mocr_device_map="",
            dots_mocr_torch_dtype="auto",
            dots_mocr_attn_implementation="auto",
        )
        try:
            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "qwen_vl_utils": fake_qwen,
                    "transformers": fake_transformers,
                },
            ):
                with patch("pdf_layout_lab.adapters.dots_mocr.has_module", return_value=False):
                    _LocalDotsMocrRuntime(settings)
        finally:
            if previous_flash is not None:
                sys.modules["flash_attn"] = previous_flash

        model_kwargs = captured["model_kwargs"]
        self.assertEqual(model_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(model_kwargs["config"].vision_config["attn_implementation"], "sdpa")
        self.assertTrue(captured["flash_stub_visible"])
        self.assertEqual(captured["to_device"], "cpu")
        self.assertTrue(captured["eval_called"])

    def test_casts_floating_tensors_to_requested_dtype(self):
        class FakeStorage:
            def __init__(self, dtype):
                self.dtype = dtype

            def to(self, *, dtype):
                return FakeStorage(dtype)

        class FakeTensor:
            def __init__(self, dtype, floating=True):
                self.data = FakeStorage(dtype)
                self.floating = floating

            @property
            def dtype(self):
                return self.data.dtype

            def is_floating_point(self):
                return self.floating

        floating_param = FakeTensor("float32")
        int_param = FakeTensor("int64", floating=False)
        floating_buffer = FakeTensor("float32")

        class FakeModule:
            def parameters(self, recurse=True):
                return [floating_param, int_param]

            def buffers(self, recurse=True):
                return [floating_buffer]

        _cast_floating_tensors_to_dtype(FakeModule(), "bfloat16")

        self.assertEqual(floating_param.dtype, "bfloat16")
        self.assertEqual(floating_buffer.dtype, "bfloat16")
        self.assertEqual(int_param.dtype, "int64")

    def test_vision_forward_uses_module_dtype_instead_of_default_bf16(self):
        fake_torch = types.SimpleNamespace(bfloat16="bfloat16", float16="float16", float32="float32")
        captured: dict[str, object] = {}

        class FakeStorage:
            def __init__(self, dtype):
                self.dtype = dtype

        class FakeParameter:
            def __init__(self, dtype):
                self.data = FakeStorage(dtype)

            @property
            def dtype(self):
                return self.data.dtype

            def is_floating_point(self):
                return True

        class FakeHiddenStates:
            def __init__(self, dtype):
                self.dtype = dtype

            def to(self, *, dtype):
                return FakeHiddenStates(dtype)

        class FakeVisionTower:
            def __init__(self):
                self.param = FakeParameter("float16")

            def parameters(self, recurse=True):
                return [self.param]

            def buffers(self, recurse=True):
                return []

            def forward(self, hidden_states, grid_thw, bf16=True):
                captured["hidden_dtype"] = hidden_states.dtype
                captured["grid_thw"] = grid_thw
                captured["bf16"] = bf16
                return "ok"

        model = types.SimpleNamespace(vision_tower=FakeVisionTower())

        _patch_vision_tower_forward(model, fake_torch)
        result = model.vision_tower.forward(FakeHiddenStates("float32"), "grid")

        self.assertEqual(result, "ok")
        self.assertEqual(captured["hidden_dtype"], "float16")
        self.assertEqual(captured["grid_thw"], "grid")
        self.assertFalse(captured["bf16"])

    def test_vision_forward_keeps_bf16_when_module_is_bf16(self):
        fake_torch = types.SimpleNamespace(bfloat16="bfloat16")
        captured: dict[str, object] = {}

        class FakeParameter:
            dtype = "bfloat16"

            def is_floating_point(self):
                return True

        class FakeHiddenStates:
            def __init__(self, dtype):
                self.dtype = dtype

            def to(self, *, dtype):
                return FakeHiddenStates(dtype)

        class FakeVisionTower:
            def parameters(self, recurse=True):
                return [FakeParameter()]

            def buffers(self, recurse=True):
                return []

            def forward(self, hidden_states, grid_thw, bf16=True):
                captured["hidden_dtype"] = hidden_states.dtype
                captured["bf16"] = bf16
                return "ok"

        model = types.SimpleNamespace(vision_tower=FakeVisionTower())

        _patch_vision_tower_forward(model, fake_torch)
        result = model.vision_tower.forward(FakeHiddenStates("float32"), "grid")

        self.assertEqual(result, "ok")
        self.assertEqual(captured["hidden_dtype"], "bfloat16")
        self.assertTrue(captured["bf16"])


if __name__ == "__main__":
    unittest.main()
