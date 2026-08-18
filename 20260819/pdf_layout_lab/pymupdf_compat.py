from __future__ import annotations


def import_pymupdf():
    """PyMuPDF の新しい import 名を優先し、古い fitz 名へフォールバックする。"""
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        import fitz

        return fitz
