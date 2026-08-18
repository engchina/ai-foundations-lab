from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from .schemas import PageImage
from .pymupdf_compat import import_pymupdf

SourceFileKind = Literal["pdf", "image"]

SUPPORTED_IMAGE_FILE_TYPES = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".tif", ".tiff")
SUPPORTED_SOURCE_FILE_TYPES = (".pdf", *SUPPORTED_IMAGE_FILE_TYPES)
MAX_IMAGE_PDF_PAGE_DIMENSION = 14400.0


def source_file_kind(source_path: str | Path) -> SourceFileKind:
    suffix = Path(source_path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in SUPPORTED_IMAGE_FILE_TYPES:
        return "image"
    supported = ", ".join(SUPPORTED_SOURCE_FILE_TYPES)
    raise ValueError(f"対応しているファイル形式は PDF または画像です。対応拡張子: {supported}")


def get_source_page_count(source_path: str | Path) -> int:
    if source_file_kind(source_path) == "image":
        return 1
    return get_pdf_page_count(source_path)


def prepare_source_for_analysis(
    source_path: str | Path,
    page_numbers: list[int],
    output_dir: str | Path,
    dpi: int,
) -> tuple[str, list[PageImage]]:
    if source_file_kind(source_path) == "image":
        return prepare_image_for_analysis(source_path, page_numbers, output_dir)

    run_pdf_path = copy_pdf_to_run(source_path, output_dir)
    return run_pdf_path, render_pdf_pages(run_pdf_path, page_numbers, output_dir, dpi)


def get_pdf_page_count(pdf_path: str | Path) -> int:
    try:
        fitz = import_pymupdf()
    except ImportError as exc:
        raise RuntimeError("PyMuPDF が未インストールのため PDF を読み込めません。`pip install pymupdf` を実行してください。") from exc

    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def render_pdf_pages(
    pdf_path: str | Path,
    page_numbers: list[int],
    output_dir: str | Path,
    dpi: int,
) -> list[PageImage]:
    """PDF の選択ページを PNG 化し、全エンジン共通の座標基準にする。"""
    try:
        fitz = import_pymupdf()
    except ImportError as exc:
        raise RuntimeError("PyMuPDF が未インストールのため PDF を画像化できません。`pip install pymupdf` を実行してください。") from exc

    pages_dir = Path(output_dir) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[PageImage] = []
    scale = float(dpi) / 72.0
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(str(pdf_path)) as doc:
        for page_number in page_numbers:
            page = doc.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = pages_dir / f"page_{page_number:04d}.png"
            pixmap.save(str(image_path))
            rendered.append(
                PageImage(
                    page=page_number,
                    width=int(pixmap.width),
                    height=int(pixmap.height),
                    pdf_width=float(page.rect.width),
                    pdf_height=float(page.rect.height),
                    image_path=str(image_path),
                )
            )
    return rendered


def copy_pdf_to_run(pdf_path: str | Path, output_dir: str | Path) -> str:
    target = Path(output_dir) / "source.pdf"
    shutil.copyfile(str(pdf_path), target)
    return str(target)


def prepare_image_for_analysis(
    image_path: str | Path,
    page_numbers: list[int],
    output_dir: str | Path,
) -> tuple[str, list[PageImage]]:
    if page_numbers != [1]:
        raise ValueError("画像ファイルは 1 ページとして扱います。解析ページには 1 を指定してください。")

    pages_dir = Path(output_dir) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_image_path = pages_dir / "page_0001.png"
    image = _open_image_as_rgb(image_path)
    width, height = image.size
    image.save(page_image_path)

    source_pdf_path = Path(output_dir) / "source.pdf"
    pdf_width, pdf_height = _write_image_pdf(page_image_path, source_pdf_path, width, height)
    page = PageImage(
        page=1,
        width=width,
        height=height,
        pdf_width=pdf_width,
        pdf_height=pdf_height,
        image_path=str(page_image_path),
    )
    return str(source_pdf_path), [page]


def _open_image_as_rgb(image_path: str | Path):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow が未インストールのため画像を読み込めません。`pip install pillow` を実行してください。") from exc

    try:
        with Image.open(str(image_path)) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                return background
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("画像ファイルを読み込めませんでした。対応形式と破損がないか確認してください。") from exc


def _write_image_pdf(image_path: Path, target_path: Path, width: int, height: int) -> tuple[float, float]:
    try:
        fitz = import_pymupdf()
    except ImportError as exc:
        raise RuntimeError("PyMuPDF が未インストールのため画像 PDF を作成できません。`pip install pymupdf` を実行してください。") from exc

    page_width, page_height = _image_pdf_page_size(width, height)
    with fitz.open() as doc:
        page = doc.new_page(width=page_width, height=page_height)
        page.insert_image(fitz.Rect(0, 0, page_width, page_height), filename=str(image_path))
        doc.save(str(target_path))
    return page_width, page_height


def _image_pdf_page_size(width: int, height: int) -> tuple[float, float]:
    longest_side = max(float(width), float(height))
    if longest_side <= MAX_IMAGE_PDF_PAGE_DIMENSION:
        return float(width), float(height)
    scale = MAX_IMAGE_PDF_PAGE_DIMENSION / longest_side
    return float(width) * scale, float(height) * scale
