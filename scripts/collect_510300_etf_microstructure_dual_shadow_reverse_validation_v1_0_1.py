"""V1.0.1机械修正：严格处理新浪净值分页的重复边界日。"""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.collect_510300_etf_microstructure_dual_shadow_reverse_validation_v1 as base


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_REVERSE_VALIDATION_V1_0_1"
CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.yaml"
)
FREEZE_MANIFEST = CONFIG_FILE.with_name(
    "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1_manifest.json"
)


def _collect_paginated_nav_corrected(
    *,
    kind: str,
    url: str,
    raw_dir: Path,
    start: Any,
    end: Any,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if kind not in {"nav_eastmoney", "nav_sina"}:
        raise ValueError(f"未知净值来源：{kind}")
    params_builder = (
        base._eastmoney_page_params
        if kind == "nav_eastmoney"
        else base._sina_page_params
    )
    headers = base.EASTMONEY_HEADERS if kind == "nav_eastmoney" else base.DEFAULT_HEADERS

    def validate(payload: dict[str, Any]) -> None:
        if kind == "nav_eastmoney":
            if not isinstance((payload.get("Data") or {}).get("LSJZList"), list):
                raise ValueError("东方财富净值响应缺少LSJZList")
        else:
            result = payload.get("result") or {}
            status = result.get("status") or {}
            if int(status.get("code", -1)) != 0:
                raise ValueError("新浪净值响应状态不是0")
            if not isinstance((result.get("data") or {}).get("data"), list):
                raise ValueError("新浪净值响应缺少data数组")

    def fetch_page(
        page: int,
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        params = params_builder(start, end, page)
        payload, receipt = base._fetch_json_body(
            kind=kind,
            identifier=f"page_{page:04d}",
            raw_path=raw_dir / kind / f"page_{page:04d}.json",
            url=url,
            params=params,
            headers=headers,
            validator=validate,
        )
        return page, payload, receipt

    _, first, first_receipt = fetch_page(1)
    if kind == "nav_eastmoney":
        total = int(first.get("TotalCount") or 0)
        page_size = len((first.get("Data") or {}).get("LSJZList") or [])
    else:
        body = ((first.get("result") or {}).get("data") or {})
        total = int(body.get("total_num") or 0)
        page_size = len(body.get("data") or [])
    if total <= 0 or page_size <= 0:
        raise ValueError(f"{kind}分页元数据无效")
    page_count = math.ceil(total / page_size)
    payloads: dict[int, dict[str, Any]] = {1: first}
    receipts: list[dict[str, Any]] = [first_receipt]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_page, page): page
            for page in range(2, page_count + 1)
        }
        for completed, future in enumerate(as_completed(futures), start=2):
            page, payload, receipt = future.result()
            payloads[page] = payload
            receipts.append(receipt)
            if completed % 25 == 0 or completed == page_count:
                print(f"{kind}采集进度：{completed}/{page_count}页", flush=True)

    page_records: dict[int, list[dict[str, Any]]] = {}
    for page in range(1, page_count + 1):
        payload = payloads[page]
        if kind == "nav_eastmoney":
            page_records[page] = list(
                (payload.get("Data") or {}).get("LSJZList") or []
            )
        else:
            page_records[page] = list(
                (((payload.get("result") or {}).get("data") or {}).get("data") or [])
            )

    if kind == "nav_eastmoney":
        records = [
            row
            for page in range(1, page_count + 1)
            for row in page_records[page]
        ]
        if len(records) != total:
            raise ValueError(f"{kind}记录数{len(records)}不等于total={total}")
        return records, receipts, page_count

    unique_records: list[dict[str, Any]] = []
    seen: dict[str, tuple[dict[str, Any], int, int]] = {}
    duplicate_boundary_count = 0
    for page in range(1, page_count + 1):
        records_on_page = page_records[page]
        for index, record in enumerate(records_on_page):
            date_key = str(record.get("fbrq"))
            if not date_key or date_key == "None":
                raise ValueError("新浪净值记录缺少fbrq")
            if date_key not in seen:
                seen[date_key] = (record, page, index)
                unique_records.append(record)
                continue
            prior_record, prior_page, prior_index = seen[date_key]
            if record != prior_record:
                raise ValueError(f"新浪重复日期内容不一致：{date_key}")
            if not (
                page == prior_page + 1
                and prior_index == len(page_records[prior_page]) - 1
                and index == 0
            ):
                raise ValueError(f"新浪重复日期不是相邻页首尾边界：{date_key}")
            duplicate_boundary_count += 1
    if len(unique_records) != total:
        raise ValueError(
            f"新浪净值去重后唯一日期数{len(unique_records)}不等于total={total}"
        )
    if len(unique_records) + duplicate_boundary_count != sum(
        len(rows) for rows in page_records.values()
    ):
        raise ValueError("新浪净值边界去重数量无法对账")
    print(
        f"nav_sina机械去重：删除{duplicate_boundary_count}个内容完全相同的相邻页边界行，"
        f"保留{len(unique_records)}个唯一日期。",
        flush=True,
    )
    return unique_records, receipts, page_count


def main() -> int:
    base.PROJECT_ID = PROJECT_ID
    base.CONFIG_FILE = CONFIG_FILE
    base.FREEZE_MANIFEST = FREEZE_MANIFEST
    base.__file__ = __file__
    base._collect_paginated_nav = _collect_paginated_nav_corrected
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
