from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from PIL import Image

from pdf_layout_lab.categories import normalize_category
from pdf_layout_lab.coordinates import polygon_to_bbox
from pdf_layout_lab.pymupdf_compat import import_pymupdf
from pdf_layout_lab.schemas import LayoutRecord, PageImage
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext, extra_install_command, has_module, missing_dependency_message, object_to_plain

OCI_SYNC_MAX_PAGES = 5
OCI_SYNC_MAX_BYTES = 8 * 1024 * 1024
OCI_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class OciDocumentAdapter:
    engine_id = "oci"
    label = "OCI Document Understanding"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if not has_module("oci"):
            return AdapterAvailability(False, missing_dependency_message("oci", extra_install_command("oci")))
        if not self.settings.oci_compartment_id:
            return AdapterAvailability(False, "OCI_COMPARTMENT_ID が未設定です。")
        try:
            _load_oci_config(self.settings)
        except RuntimeError as exc:
            return AdapterAvailability(False, str(exc))
        return AdapterAvailability(True, "OCI config/profile 認証で Document Understanding を呼び出します。")

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        import oci

        config = _load_oci_config(self.settings)
        client = oci.ai_document.AIServiceDocumentClient(config)
        records: list[LayoutRecord] = []
        for batch_index, batch_pages in enumerate(_chunk_pages(context.pages, OCI_SYNC_MAX_PAGES), start=1):
            for input_path, page_number_map in _iter_oci_input_documents(
                context.pdf_path,
                batch_pages,
                context.run_dir / "oci",
                batch_index,
            ):
                encoded = base64.b64encode(input_path.read_bytes()).decode("ascii")
                details_kwargs = dict(
                    compartment_id=self.settings.oci_compartment_id,
                    document=oci.ai_document.models.InlineDocumentDetails(
                        source="INLINE",
                        data=encoded,
                    ),
                    features=[
                        oci.ai_document.models.DocumentTextExtractionFeature(feature_type="TEXT_EXTRACTION"),
                        oci.ai_document.models.DocumentTableExtractionFeature(feature_type="TABLE_EXTRACTION"),
                    ],
                )
                language = _language_or_none(self.settings.oci_document_language)
                if language:
                    details_kwargs["language"] = language
                details = oci.ai_document.models.AnalyzeDocumentDetails(**details_kwargs)
                response = client.analyze_document(analyze_document_details=details)
                records.extend(self._records_from_response(object_to_plain(response.data), context, page_number_map))
        return records

    def _records_from_response(
        self,
        payload: dict[str, Any],
        context: AnalysisContext,
        page_number_map: dict[int, int] | None = None,
    ) -> list[LayoutRecord]:
        page_lookup = {page.page: page for page in context.pages}
        records: list[LayoutRecord] = []
        for page_payload in payload.get("pages", []):
            response_page_number = int(page_payload.get("page_number") or page_payload.get("pageNumber") or 0)
            page_number = page_number_map.get(response_page_number, response_page_number) if page_number_map else response_page_number
            page = page_lookup.get(page_number)
            if not page:
                continue
            seq = 1
            for table in page_payload.get("tables", []) or []:
                confidence = _optional_float(table.get("confidence"))
                if confidence is not None and confidence < context.min_confidence:
                    continue
                bbox = _bbox_from_oci_item(table, page.width, page.height)
                if not bbox:
                    continue
                records.append(
                    LayoutRecord(
                        id=f"oci-p{page_number}-table-{seq}",
                        engine=self.engine_id,
                        page=page_number,
                        seq_no=seq,
                        bbox=bbox,
                        coord_system="image_top_left",
                        page_width=page.width,
                        page_height=page.height,
                        category="Table",
                        text=_table_to_html(table),
                        confidence=confidence,
                        raw_type="table",
                        raw=table,
                    )
                )
                seq += 1
            for line in page_payload.get("lines", []) or []:
                confidence = _optional_float(line.get("confidence"))
                if confidence is not None and confidence < context.min_confidence:
                    continue
                bbox = _bbox_from_oci_item(line, page.width, page.height)
                if not bbox:
                    continue
                records.append(
                    LayoutRecord(
                        id=f"oci-p{page_number}-line-{seq}",
                        engine=self.engine_id,
                        page=page_number,
                        seq_no=seq,
                        bbox=bbox,
                        coord_system="image_top_left",
                        page_width=page.width,
                        page_height=page.height,
                        category=normalize_category("Text"),
                        text=str(line.get("text") or ""),
                        confidence=confidence,
                        raw_type="line",
                        raw=line,
                    )
                )
                seq += 1
        return records


def _bbox_from_oci_item(item: dict[str, Any], width: float, height: float) -> list[float] | None:
    polygon = item.get("bounding_polygon") or item.get("boundingPolygon") or {}
    vertices = polygon.get("normalized_vertices") or polygon.get("normalizedVertices") or []
    if not vertices:
        return None
    return polygon_to_bbox(vertices, width, height)


def _load_oci_config(settings: Settings) -> dict[str, Any]:
    import oci

    config_path = str(Path(settings.oci_config_file).expanduser())
    try:
        return oci.config.from_file(config_path, settings.oci_profile)
    except Exception as exc:
        message = (
            f"OCI 設定を読み込めません。OCI_CONFIG_FILE={config_path}、"
            f"OCI_PROFILE={settings.oci_profile} を確認してください。詳細: {exc}"
        )
        raise RuntimeError(message) from exc


def _page_ranges_from_pages(page_numbers) -> list[str]:
    pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not pages:
        return []
    ranges: list[str] = []
    start = pages[0]
    end = pages[0]
    for page in pages[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(_format_page_range(start, end))
        start = end = page
    ranges.append(_format_page_range(start, end))
    return ranges


def _format_page_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _language_or_none(language: str) -> str | None:
    normalized = language.strip()
    if not normalized or normalized.lower() in {"auto", "none", "detect"}:
        return None
    return normalized


def _chunk_pages(pages: list[PageImage], chunk_size: int) -> list[list[PageImage]]:
    return [pages[index : index + chunk_size] for index in range(0, len(pages), chunk_size)]


def _iter_oci_input_documents(
    source_pdf: str | Path,
    pages: list[PageImage],
    output_dir: str | Path,
    batch_index: int,
) -> list[tuple[Path, dict[int, int]]]:
    subset_path = _write_subset_pdf(source_pdf, pages, output_dir, batch_index)
    page_number_map = {index + 1: page.page for index, page in enumerate(pages)}
    if _file_size_within_oci_sync_limit(subset_path):
        return [(subset_path, page_number_map)]
    _unlink_if_exists(subset_path)
    if len(pages) == 1:
        return [(_prepare_single_page_image_for_oci(pages[0], output_dir, batch_index), {1: pages[0].page})]
    inputs: list[tuple[Path, dict[int, int]]] = []
    for offset, page in enumerate(pages, start=1):
        inputs.extend(_iter_oci_input_documents(source_pdf, [page], output_dir, batch_index * 100 + offset))
    return inputs


def _write_subset_pdf(source_pdf: str | Path, pages: list[PageImage], output_dir: str | Path, batch_index: int) -> Path:
    fitz = import_pymupdf()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"batch_{batch_index:04d}.pdf"
    with fitz.open(str(source_pdf)) as source_doc:
        with fitz.open() as subset_doc:
            for page in pages:
                subset_doc.insert_pdf(source_doc, from_page=page.page - 1, to_page=page.page - 1)
            subset_doc.save(str(target_path))
    return target_path


def _prepare_single_page_image_for_oci(page: PageImage, output_dir: str | Path, batch_index: int) -> Path:
    image_path = Path(page.image_path)
    if image_path.suffix.lower() in OCI_IMAGE_SUFFIXES and _file_size_within_oci_sync_limit(image_path):
        return image_path
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"batch_{batch_index:04d}_page_{page.page:04d}.jpg"
    _write_compressed_jpeg_for_oci(image_path, target_path)
    return target_path


def _write_compressed_jpeg_for_oci(source_path: Path, target_path: Path) -> None:
    with Image.open(str(source_path)) as image:
        rgb_image = image.convert("RGB")
        for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
            candidate = rgb_image
            if scale != 1.0:
                width = max(32, int(rgb_image.width * scale))
                height = max(32, int(rgb_image.height * scale))
                candidate = rgb_image.resize((width, height), Image.Resampling.LANCZOS)
            for quality in (90, 80, 70, 60, 50, 40):
                candidate.save(str(target_path), format="JPEG", quality=quality, optimize=True)
                if _file_size_within_oci_sync_limit(target_path):
                    return
    raise RuntimeError(
        f"OCI Document Understanding の同期入力上限 {OCI_SYNC_MAX_BYTES // (1024 * 1024)} MB 以下に画像を圧縮できませんでした。"
        "ページ画像 DPI を下げてから再試行してください。"
    )


def _file_size_within_oci_sync_limit(path: str | Path) -> bool:
    return Path(path).stat().st_size <= OCI_SYNC_MAX_BYTES


def _unlink_if_exists(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _table_to_html(table: dict[str, Any]) -> str:
    rows = []
    for key in ("header_rows", "headerRows", "body_rows", "bodyRows"):
        value = table.get(key) or []
        rows.extend(value)
    if not rows:
        return str(table.get("text") or "")
    row_map: dict[int, dict[int, str]] = {}
    for row in rows:
        for cell in row.get("cells", []) or []:
            row_index = int(cell.get("row_index") or cell.get("rowIndex") or 0)
            column_index = int(cell.get("column_index") or cell.get("columnIndex") or 0)
            row_map.setdefault(row_index, {})[column_index] = str(cell.get("text") or "")
    html_rows = []
    for row_index in sorted(row_map):
        cells = "".join(f"<td>{_escape_html(row_map[row_index].get(column_index, ''))}</td>" for column_index in sorted(row_map[row_index]))
        html_rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric
