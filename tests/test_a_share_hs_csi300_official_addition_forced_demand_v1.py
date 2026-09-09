from __future__ import annotations

import io
from datetime import date

import pandas as pd
import pytest

from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (
    build_event_ledger,
    commission_cny,
    dividend_cash_cny,
    entry_fill_status,
    extract_effective_date,
    extract_csi300_changes_from_pdf_text,
    extract_excel_csi300_changes,
    extract_html_csi300_changes,
    extract_official_attachment_urls,
    first_trading_day_after,
    normalize_security_code,
    resolve_exit_clock,
    round_board_lot_shares,
    sell_stamp_duty_cny,
    validate_no_result_columns,
)


def test_effective_date_and_next_trading_day_are_point_in_time() -> None:
    html = "<p>本次调整将于 2025 年 12 月 12 日收市后生效。</p><p>2025年11月28日</p>"
    assert extract_effective_date(html, "2025-11-28") == "2025-12-12"
    open_days = [date(2025, 11, 28), date(2025, 12, 1), date(2025, 12, 2)]
    assert first_trading_day_after("2025-11-28", open_days) == "2025-12-01"


def test_official_attachment_allowlist_and_encoded_notice_path() -> None:
    detail = {
        "enclosureList": [
            {
                "fileUrl": "https://oss-ch.csindex.com.cn/notice%2Fofficial.pdf",
                "fileName": "官方名单.pdf",
            }
        ],
        "content": (
            '<a href="https://oss-ch.csindex.com.cn/notice/official.pdf">重复</a>'
            '<a href="https://example.com/not-official.pdf">非官方</a>'
        ),
    }
    result = extract_official_attachment_urls(detail)
    assert result == [
        {
            "url": "https://oss-ch.csindex.com.cn/notice/official.pdf",
            "file_name": "官方名单.pdf",
        }
    ]


def test_html_first_csi300_table_extracts_additions_and_deletions() -> None:
    html = """
    <table>
      <tr><th>调出名单</th><th></th><th>调入名单</th><th></th></tr>
      <tr><th>证券代码</th><th>证券名称</th><th>证券代码</th><th>证券名称</th></tr>
      <tr><td>000001</td><td>旧一</td><td>600001</td><td>新一</td></tr>
      <tr><td>000002</td><td>旧二</td><td>300001</td><td>新二</td></tr>
    </table>
    """
    result = extract_html_csi300_changes(html)
    assert result is not None
    assert [row["security_code"] for row in result["additions"]] == ["600001", "300001"]
    assert [row["security_code"] for row in result["deletions"]] == ["000001", "000002"]


def test_pdf_text_parser_preserves_double_column_boundaries() -> None:
    text = """
    附件：部分指数样本调整名单
    沪深300 指数样本调整名单：
    调出名单 调入名单
    证券代码 证券名称 证券代码 证券名称
    002008 大族激光 000617 中油资本
    002032 苏泊尔 000983 山西焦煤
    中证500 指数样本调整名单：
    000001 旧样本 000002 新样本
    """
    result = extract_csi300_changes_from_pdf_text(text)
    assert [row["security_code"] for row in result["additions"]] == ["000617", "000983"]
    assert [row["security_code"] for row in result["deletions"]] == ["002008", "002032"]


def test_legacy_workbook_first_two_000300_sheets_define_roles() -> None:
    additions = pd.DataFrame(
        [
            ["指数代码", "指数简称", "证券代码", "证券简称"],
            [300, "沪深300", 600001, "新一"],
            [300, "沪深300", 8, "新二"],
        ]
    )
    deletions = pd.DataFrame(
        [
            ["指数代码", "指数简称", "证券代码", "证券简称"],
            [300, "沪深300", 600002, "旧一"],
            [300, "沪深300", 9, "旧二"],
        ]
    )
    reserve = pd.DataFrame([[300, "沪深300", 600003, "备选"]])
    payload = io.BytesIO()
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        additions.to_excel(writer, sheet_name="调入", header=False, index=False)
        deletions.to_excel(writer, sheet_name="调出", header=False, index=False)
        reserve.to_excel(writer, sheet_name="备选名单", header=False, index=False)
    result = extract_excel_csi300_changes(payload.getvalue(), ".xlsx")
    assert [row["security_code"] for row in result["additions"]] == ["600001", "000008"]
    assert [row["security_code"] for row in result["deletions"]] == ["600002", "000009"]
    assert result["source_sheet_additions"] == "调入"
    assert result["source_sheet_deletions"] == "调出"


def test_no_return_event_ledger_has_stable_event_keys_and_clocks() -> None:
    cycles = [
        {
            "cycle_id": "2026H1",
            "announcement_id": 3006137,
            "announcement_title": "关于沪深300指数定期调整结果的公告",
            "announcement_date": "2026-05-29",
            "effective_date": "2026-06-12",
            "source_detail_url": "https://www.csindex.com.cn/detail",
            "source_detail_sha256": "a" * 64,
            "source_format": "PDF",
            "extraction_method": "OFFICIAL_PDF_CSI300_FIRST_SECTION",
            "attachment": {
                "url": "https://oss-ch.csindex.com.cn/notice/list.pdf",
                "sha256": "b" * 64,
                "archive_path": "data/raw/list.pdf",
            },
            "additions": [
                {"security_code": "600001", "security_name": "新一"},
                {"security_code": "000008", "security_name": "新二"},
            ],
        }
    ]
    ledger = build_event_ledger(cycles, [date(2026, 5, 29), date(2026, 6, 1), date(2026, 6, 12)])
    assert ledger["event_key"].tolist() == ["2026H1|600001.SH", "2026H1|000008.SZ"]
    assert set(ledger["entry_date"]) == {"2026-06-01"}
    assert set(ledger["return_evaluation"]) == {"NOT_ALLOWED"}
    assert not any("price" in column.lower() for column in ledger.columns)


def test_mechanics_board_lot_commission_stamp_duty_and_dividend() -> None:
    assert round_board_lot_shares(6_700, 12.30) == 500
    assert round_board_lot_shares(99, 12.30) == 0
    assert commission_cny(10_000) == 5.0
    assert commission_cny(100_000) == 30.0
    assert sell_stamp_duty_cny(10_000, "2023-08-27") == 10.0
    assert sell_stamp_duty_cny(10_000, "2023-08-28") == 5.0
    assert dividend_cash_cny(500, 0.12) == 60.0


def test_mechanics_suspension_limit_up_and_blocked_exit() -> None:
    assert entry_fill_status(suspended=True, one_price_limit_up=False) == "NO_FILL_SUSPENDED"
    assert entry_fill_status(suspended=False, one_price_limit_up=True) == "NO_FILL_ONE_PRICE_LIMIT_UP"
    assert entry_fill_status(suspended=False, one_price_limit_up=False) == "FILL_AT_RAW_OPEN"
    assert resolve_exit_clock(
        "2026-06-12",
        {"2026-06-12": False, "2026-06-15": False, "2026-06-16": True},
    ) == ("EXIT_AT_FIRST_SELLABLE_OPEN", "2026-06-16")
    assert resolve_exit_clock("2026-06-12", {"2026-06-12": False}) == (
        "CENSORED_EXIT_PENDING_FIRST_SELLABLE_OPEN",
        None,
    )


def test_d8_rejects_result_or_price_columns() -> None:
    validate_no_result_columns(["event_key", "return_evaluation"])
    with pytest.raises(ValueError, match="D8"):
        validate_no_result_columns(["event_key", "entry_price"])
    with pytest.raises(ValueError, match="D8"):
        validate_no_result_columns(["event_key", "excess_return"])


def test_security_code_normalisation() -> None:
    assert normalize_security_code(8) == "000008"
    assert normalize_security_code("600001.0") == "600001"
    assert normalize_security_code("证券代码 300001") == "300001"
