from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """外部依存なしで .env を読み込む。既存の環境変数は上書きしない。"""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    output_dir: Path
    render_dpi: int
    max_default_pages: int
    enabled_engines: list[str]
    dots_mocr_backend: str
    dots_mocr_base_url: str
    dots_mocr_model: str
    dots_mocr_api_key: str
    dots_mocr_timeout_seconds: float
    dots_mocr_device: str
    dots_mocr_device_map: str
    dots_mocr_torch_dtype: str
    dots_mocr_attn_implementation: str
    dots_mocr_max_new_tokens: int
    oci_config_file: str
    oci_profile: str
    oci_compartment_id: str
    oci_document_language: str
    docling_device: str
    docling_num_threads: int
    docling_do_ocr: bool
    docling_do_table_structure: bool
    yolov10_model_path: str
    pp_doclayout_model: str
    pp_doclayout_engine: str
    pp_doclayout_device: str
    pp_doclayout_model_source: str
    mineru_output_dir: str
    mineru_command: str
    mineru_backend: str
    mineru_method: str
    mineru_extra_args: str


def get_settings() -> Settings:
    load_dotenv()
    enabled = os.environ.get(
        "PDF_LAYOUT_LAB_ENABLED_ENGINES",
        "oci,mineru,dots_mocr,unstructured,docling,pymupdf,yolov10,pp_doclayout_v3",
    )
    enabled_engines = [part.strip() for part in enabled.split(",") if part.strip()]
    return Settings(
        host=os.environ.get("PDF_LAYOUT_LAB_HOST", "127.0.0.1"),
        port=_env_int("PDF_LAYOUT_LAB_PORT", 7860),
        output_dir=Path(os.environ.get("PDF_LAYOUT_LAB_OUTPUT_DIR", ".runs")),
        render_dpi=_env_int("PDF_LAYOUT_LAB_RENDER_DPI", 300),
        max_default_pages=_env_int("PDF_LAYOUT_LAB_MAX_DEFAULT_PAGES", 1),
        enabled_engines=enabled_engines,
        dots_mocr_backend=os.environ.get("DOTS_MOCR_BACKEND", "transformers"),
        dots_mocr_base_url=os.environ.get("DOTS_MOCR_BASE_URL", "http://127.0.0.1:8000/v1"),
        dots_mocr_model=os.environ.get("DOTS_MOCR_MODEL", "rednote-hilab/dots.mocr"),
        dots_mocr_api_key=os.environ.get("DOTS_MOCR_API_KEY", ""),
        dots_mocr_timeout_seconds=_env_float("DOTS_MOCR_TIMEOUT_SECONDS", 120.0),
        dots_mocr_device=os.environ.get("DOTS_MOCR_DEVICE", "auto"),
        dots_mocr_device_map=os.environ.get("DOTS_MOCR_DEVICE_MAP", ""),
        dots_mocr_torch_dtype=os.environ.get("DOTS_MOCR_TORCH_DTYPE", "auto"),
        dots_mocr_attn_implementation=os.environ.get("DOTS_MOCR_ATTN_IMPLEMENTATION", "auto"),
        dots_mocr_max_new_tokens=_env_int("DOTS_MOCR_MAX_NEW_TOKENS", 24000),
        oci_config_file=os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"),
        oci_profile=os.environ.get("OCI_PROFILE", "DEFAULT"),
        oci_compartment_id=os.environ.get("OCI_COMPARTMENT_ID", ""),
        oci_document_language=os.environ.get("OCI_DOCUMENT_LANGUAGE", "auto"),
        docling_device=os.environ.get("DOCLING_DEVICE", "cpu"),
        docling_num_threads=_env_int("DOCLING_NUM_THREADS", _env_int("OMP_NUM_THREADS", 4)),
        docling_do_ocr=_env_bool("DOCLING_DO_OCR", True),
        docling_do_table_structure=_env_bool("DOCLING_DO_TABLE_STRUCTURE", True),
        yolov10_model_path=os.environ.get("YOLOV10_MODEL_PATH", "models/yolov10x_best.pt"),
        pp_doclayout_model=os.environ.get("PP_DOCLAYOUT_MODEL", "PP-DocLayoutV3"),
        pp_doclayout_engine=os.environ.get("PP_DOCLAYOUT_ENGINE", "onnxruntime"),
        pp_doclayout_device=os.environ.get("PP_DOCLAYOUT_DEVICE", "auto"),
        pp_doclayout_model_source=os.environ.get("PP_DOCLAYOUT_MODEL_SOURCE", "BOS"),
        mineru_output_dir=os.environ.get("MINERU_OUTPUT_DIR", ""),
        mineru_command=os.environ.get("MINERU_COMMAND", "auto"),
        mineru_backend=os.environ.get("MINERU_BACKEND", "pipeline"),
        mineru_method=os.environ.get("MINERU_METHOD", "auto"),
        mineru_extra_args=os.environ.get("MINERU_EXTRA_ARGS", ""),
    )
