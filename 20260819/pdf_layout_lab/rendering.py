from __future__ import annotations

import shutil
from pathlib import Path

from .schemas import PageImage
from .pymupdf_compat import import_pymupdf


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
