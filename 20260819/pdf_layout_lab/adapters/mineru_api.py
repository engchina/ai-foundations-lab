from __future__ import annotations

import json
from typing import Any

from pdf_layout_lab.schemas import LayoutRecord
from pdf_layout_lab.settings import Settings

from .base import AdapterAvailability, AnalysisContext
from .mineru import _records_from_dict_payload

# hybrid-engine の初回呼び出しはモデルのウォームアップで 1〜2 分かかる
MINERU_API_TIMEOUT_SECONDS = 600.0


class MineruApiAdapter:
    """mineru-api サーバー (POST /file_parse) を同期呼び出しするアダプター。"""

    engine_id = "mineru_api"
    label = "MinerU / MinerU2.5-Pro (API)"

    def __init__(self, settings: Settings):
        self.settings = settings

    def availability(self) -> AdapterAvailability:
        if not self.settings.mineru_api_url:
            return AdapterAvailability(False, "MINERU_API_URL に mineru-api サーバーの URL（例: http://127.0.0.1:8887）を設定してください。")
        return AdapterAvailability(True, f"mineru-api `{self.settings.mineru_api_url}` の /file_parse を hybrid-engine / effort=high で呼び出します。")

    def analyze(self, context: AnalysisContext) -> list[LayoutRecord]:
        import httpx  # gradio の依存として常に入っている

        url = self.settings.mineru_api_url.rstrip("/") + "/file_parse"
        data = {
            "backend": "hybrid-engine",
            "effort": "high",
            "return_md": "false",
            "return_middle_json": "true",
        }
        with open(context.pdf_path, "rb") as handle:
            files = {"files": (context.pdf_path.name, handle, "application/pdf")}
            try:
                response = httpx.post(url, data=data, files=files, timeout=MINERU_API_TIMEOUT_SECONDS)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"mineru-api への接続に失敗しました ({url}): {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(f"mineru-api がエラーを返しました (HTTP {response.status_code}): {response.text[:500]}")
        return records_from_file_parse_response(response.json(), context)


def records_from_file_parse_response(payload: dict[str, Any], context: AnalysisContext) -> list[LayoutRecord]:
    """/file_parse のレスポンス (results.<name>.middle_json) を LayoutRecord へ変換する。"""
    if payload.get("error"):
        raise RuntimeError(f"mineru-api の解析が失敗しました: {payload['error']}")
    results = payload.get("results") or {}
    if not results:
        raise RuntimeError("mineru-api のレスポンスに results がありません。")
    page_lookup = {page.page: page for page in context.pages}
    records: list[LayoutRecord] = []
    for result in results.values():
        middle_json = result.get("middle_json")
        if isinstance(middle_json, str):  # サーバーは JSON 文字列で返す
            middle_json = json.loads(middle_json)
        if isinstance(middle_json, dict):
            records.extend(_records_from_dict_payload(middle_json, page_lookup, MineruApiAdapter.engine_id, context.min_confidence))
    return records
