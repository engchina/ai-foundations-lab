from __future__ import annotations

import re


def parse_page_range(value: str, page_count: int, default_limit: int = 1) -> list[int]:
    """1,3-5 のようなページ指定を 1-based のページ番号へ変換する。"""
    cleaned = (value or "").strip()
    if not cleaned:
        return list(range(1, min(page_count, default_limit) + 1))
    if cleaned.lower() in {"all", "すべて", "全部"}:
        return list(range(1, page_count + 1))

    pages: set[int] = set()
    for part in re.split(r"[,、\s]+", cleaned):
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    valid_pages = sorted(page for page in pages if 1 <= page <= page_count)
    if not valid_pages:
        raise ValueError(f"有効なページ番号がありません。1 から {page_count} の範囲で指定してください。")
    return valid_pages
