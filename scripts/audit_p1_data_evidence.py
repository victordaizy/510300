"""审计沪深300历史成员/权重证据和财务事件历史版本风险。

本脚本只读取 data/ 下的现有输入，并仅写入 reports/audit 与
reports/data_quality。它不修改原始数据，也不把供应商统一回取值认证为严格PIT。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DATE = "20260819"
TIMEZONE = ZoneInfo("Asia/Shanghai")

MEMBERSHIP_FILE = (
    ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
)
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
CURRENT_WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_current_weights.parquet"
OFFICIAL_CURRENT_RAW_FILE = (
    ROOT / "data" / "raw" / "constituents" / "000300_official_snapshot_raw.json"
)
MEMBERSHIP_STATUS_FILE = (
    ROOT / "reports" / "data_quality" / "000300_historical_membership_status.json"
)
CALENDAR_FILE = ROOT / "data" / "raw" / "return_tail" / "options" / "sse_trading_calendar.parquet"
FINANCIAL_FILE = (
    ROOT / "data" / "raw" / "fundamentals" / "csi300_financials_point_in_time_extended_v2.parquet"
)
INDUSTRY_STRATIFICATION_FILE = (
    ROOT / "data" / "raw" / "reference" / "a_share_sw_industry_static_pre_pit.parquet"
)
CHECKPOINT_ROOTS = (
    ROOT / "data" / "raw" / "fundamentals" / "vip_checkpoints",
    ROOT / "data" / "raw" / "fundamentals" / "vip_checkpoints_normalized_2012_2015",
)

AUDIT_DIR = ROOT / "reports" / "audit"
QUALITY_DIR = ROOT / "reports" / "data_quality"
OFFICIAL_PDF_DIR = AUDIT_DIR / "official_csi300_rebalance_documents"

MEMBERSHIP_EVIDENCE_FILE = AUDIT_DIR / f"csi300_membership_event_evidence_{AUDIT_DATE}.csv"
WEIGHT_EVIDENCE_FILE = AUDIT_DIR / f"csi300_weight_snapshot_evidence_{AUDIT_DATE}.csv"
MEMBERSHIP_REPORT_FILE = QUALITY_DIR / f"csi300_point_in_time_membership_evidence_{AUDIT_DATE}.json"
MEMBERSHIP_REPORT_MD_FILE = MEMBERSHIP_REPORT_FILE.with_suffix(".md")
FINANCIAL_SAMPLE_FILE = AUDIT_DIR / f"csi300_financial_vintage_sample_{AUDIT_DATE}.csv"
CHECKPOINT_INVENTORY_FILE = AUDIT_DIR / f"csi300_financial_checkpoint_inventory_{AUDIT_DATE}.csv"
FINANCIAL_REPORT_FILE = QUALITY_DIR / f"csi300_financial_vintage_audit_{AUDIT_DATE}.json"
FINANCIAL_REPORT_MD_FILE = FINANCIAL_REPORT_FILE.with_suffix(".md")

SCOPE_START = pd.Timestamp("2019-12-23")
SCOPE_END = pd.Timestamp("2026-08-14")

# 搜索只使用中证指数官网；附件URL中的时间戳提供官方文件发布日期线索。
OFFICIAL_CHANGE_LISTS: dict[str, dict[str, str]] = {
    "2024-12": {
        "announcement_date": "2024-11-29",
        "url": (
            "https://oss-ch.csindex.com.cn/notice/"
            "20241129172348-%E9%99%84%E4%BB%B6%EF%BC%9A%E9%83%A8%E5%88%86%E6%8C%87%E6%95%B0"
            "%E6%A0%B7%E6%9C%AC%E8%B0%83%E6%95%B4%E5%90%8D%E5%8D%95.pdf"
        ),
    },
    "2025-12": {
        "announcement_date": "2025-11-28",
        "url": (
            "https://oss-ch.csindex.com.cn/notice/"
            "20251128165753-%E9%99%84%E4%BB%B6%EF%BC%9A%E9%83%A8%E5%88%86%E6%8C%87%E6%95%B0"
            "%E6%A0%B7%E6%9C%AC%E8%B0%83%E6%95%B4%E5%90%8D%E5%8D%95.pdf"
        ),
    },
    "2026-06": {
        "announcement_date": "2026-05-29",
        "url": (
            "https://oss-ch.csindex.com.cn/notice/"
            "20260529155822-%E9%99%84%E4%BB%B6%EF%BC%9A%E9%83%A8%E5%88%86%E6%8C%87%E6%95%B0"
            "%E6%A0%B7%E6%9C%AC%E8%B0%83%E6%95%B4%E5%90%8D%E5%8D%95.pdf"
        ),
    },
}

OFFICIAL_METHODOLOGY: dict[str, str] = {
    "document_date": "2023-09-08",
    "url": (
        "https://oss-ch.csindex.com.cn/notice/"
        "20230908165124-%E3%80%8A%E4%B8%AD%E8%AF%81%E6%8C%87%E6%95%B0%E6%9C%89%E9%99%90"
        "%E5%85%AC%E5%8F%B8%E8%82%A1%E7%A5%A8%E6%8C%87%E6%95%B0%E8%AE%A1%E7%AE%97%E4%B8%8E"
        "%E7%BB%B4%E6%8A%A4%E7%BB%86%E5%88%99%E3%80%8B.pdf"
    ),
    "rule": "样本定期调整实施时间为每年六月和十二月第二个星期五的下一交易日",
}
ALLOWED_OFFICIAL_HOSTS = {"oss-ch.csindex.com.cn", "www.csindex.com.cn"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_rows_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].copy().sort_values(columns).reset_index(drop=True)
    content = data.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d").encode("utf-8")
    return sha256_bytes(content)


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _report_period_type(period: pd.Timestamp) -> str:
    return {
        "0331": "Q1",
        "0630": "H1",
        "0930": "Q3",
        "1231": "ANNUAL",
    }.get(period.strftime("%m%d"), "OTHER")


def calendar_day_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    """按各字段展示的自然日计算差值，避免带时区抓取时间与无时区公告日混算。"""

    left_days = pd.to_datetime(
        pd.to_datetime(left, errors="coerce").dt.strftime("%Y-%m-%d"), errors="coerce"
    )
    right_days = pd.to_datetime(
        pd.to_datetime(right, errors="coerce").dt.strftime("%Y-%m-%d"), errors="coerce"
    )
    return (left_days - right_days).dt.days


def expected_regular_effective_date(
    year: int, month: int, trading_dates: Iterable[pd.Timestamp]
) -> pd.Timestamp:
    if month not in {6, 12}:
        raise ValueError("定期调样规则只适用于6月或12月")
    month_days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    month_days = month_days[month_days.month == month]
    fridays = [pd.Timestamp(day).normalize() for day in month_days if day.weekday() == 4]
    second_friday = fridays[1]
    eligible = sorted(pd.Timestamp(day).normalize() for day in trading_dates if pd.Timestamp(day) > second_friday)
    if not eligible:
        raise ValueError(f"交易日历无法计算{year}-{month:02d}定期调样实施日")
    return eligible[0]


def build_membership_events(membership: pd.DataFrame) -> pd.DataFrame:
    data = membership.copy()
    data["opt_in"] = pd.to_datetime(data["opt_in"], errors="coerce")
    data["opt_out"] = pd.to_datetime(data["opt_out"], errors="coerce")
    dates = sorted(
        set(data.loc[data["opt_in"].between(SCOPE_START, SCOPE_END), "opt_in"].dropna())
        | set(data.loc[data["opt_out"].between(SCOPE_START, SCOPE_END), "opt_out"].dropna())
    )
    rows: list[dict[str, Any]] = []
    for date in dates:
        opt_in = sorted(data.loc[data["opt_in"].eq(date), "symbol"].astype(str))
        opt_out = sorted(data.loc[data["opt_out"].eq(date), "symbol"].astype(str))
        change_frame = pd.DataFrame(
            [("IN", symbol) for symbol in opt_in] + [("OUT", symbol) for symbol in opt_out],
            columns=["action", "symbol"],
        )
        rows.append(
            {
                "event_date": pd.Timestamp(date).normalize(),
                "opt_in_count": len(opt_in),
                "opt_out_count": len(opt_out),
                "opt_in_symbols": ";".join(opt_in),
                "opt_out_symbols": ";".join(opt_out),
                "transition_sha256": canonical_rows_sha256(change_frame, ["action", "symbol"]),
            }
        )
    return pd.DataFrame(rows)


def _extract_csi300_changes(pdf_content: bytes) -> tuple[set[str], set[str]]:
    with pdfplumber.open(io.BytesIO(pdf_content)) as document:
        text = "\n".join(page.extract_text() or "" for page in document.pages)
    # 部分官方PDF的中文字体映射缺失，提取后的标题会乱码，但六位证券代码
    # 仍保持可读。官方调样附件把沪深300放在第一页的首个双代码连续块，
    # 后续连续块才是中证500等指数。因此解析结构，不依赖易损的中文字体。
    code_blocks: list[list[tuple[str, str]]] = []
    current_block: list[tuple[str, str]] = []
    for line in text.splitlines():
        codes = re.findall(r"(?<!\d)[036]\d{5}(?!\d)", line)
        if len(codes) == 2:
            current_block.append((codes[0], codes[1]))
        elif current_block:
            code_blocks.append(current_block)
            current_block = []
    if current_block:
        code_blocks.append(current_block)
    if not code_blocks:
        raise ValueError("官方PDF未提取到任何双代码调样表")
    opt_out_codes = {pair[0] for pair in code_blocks[0]}
    opt_in_codes = {pair[1] for pair in code_blocks[0]}
    if not opt_out_codes or len(opt_out_codes) != len(opt_in_codes):
        raise ValueError("官方PDF首个调样表的证券代码结构异常")
    return opt_out_codes, opt_in_codes


def _extract_csi300_codes(pdf_content: bytes) -> set[str]:
    opt_out_codes, opt_in_codes = _extract_csi300_changes(pdf_content)
    return opt_out_codes | opt_in_codes


def fetch_official_change_lists(fetch: bool) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if fetch:
        OFFICIAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    for key, evidence in OFFICIAL_CHANGE_LISTS.items():
        result: dict[str, Any] = {**evidence, "fetch_attempted": fetch}
        if not fetch:
            result.update(
                {
                    "fetch_status": "NOT_REQUESTED",
                    "document_sha256": None,
                    "csi300_codes": [],
                    "csi300_opt_out_codes": [],
                    "csi300_opt_in_codes": [],
                    "retrieved_at": None,
                }
            )
            results[key] = result
            continue
        parsed = urlparse(evidence["url"])
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_OFFICIAL_HOSTS:
            raise ValueError(f"官方证据URL不在允许域名：{evidence['url']}")
        try:
            response = requests.get(evidence["url"], timeout=60, allow_redirects=True)
            response.raise_for_status()
            final_host = urlparse(response.url).hostname
            if final_host not in ALLOWED_OFFICIAL_HOSTS:
                raise ValueError(f"官方证据发生跨域跳转：{response.url}")
            content = response.content
            if not content.startswith(b"%PDF"):
                raise ValueError("官方证据响应不是PDF")
            output = OFFICIAL_PDF_DIR / f"csi300_rebalance_{key}.pdf"
            output.write_bytes(content)
            opt_out_codes, opt_in_codes = _extract_csi300_changes(content)
            result.update(
                {
                    "fetch_status": "SUCCESS",
                    "document_sha256": sha256_bytes(content),
                    "csi300_codes": sorted(opt_out_codes | opt_in_codes),
                    "csi300_opt_out_codes": sorted(opt_out_codes),
                    "csi300_opt_in_codes": sorted(opt_in_codes),
                    "retrieved_at": datetime.now(TIMEZONE).isoformat(),
                    "local_file": output.relative_to(ROOT).as_posix(),
                }
            )
        except Exception as exc:  # 网络证据失败必须进入报告，不能静默回退到二手来源。
            result.update(
                {
                    "fetch_status": "FAILED",
                    "document_sha256": None,
                    "csi300_codes": [],
                    "csi300_opt_out_codes": [],
                    "csi300_opt_in_codes": [],
                    "retrieved_at": datetime.now(TIMEZONE).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        results[key] = result
    return results


def fetch_official_methodology(fetch: bool) -> dict[str, Any]:
    result: dict[str, Any] = {**OFFICIAL_METHODOLOGY, "fetch_attempted": fetch}
    if not fetch:
        result.update(
            {
                "fetch_status": "NOT_REQUESTED",
                "document_sha256": None,
                "retrieved_at": None,
            }
        )
        return result
    parsed = urlparse(OFFICIAL_METHODOLOGY["url"])
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_OFFICIAL_HOSTS:
        raise ValueError(f"官方方法文件URL不在允许域名：{OFFICIAL_METHODOLOGY['url']}")
    try:
        response = requests.get(
            OFFICIAL_METHODOLOGY["url"], timeout=60, allow_redirects=True
        )
        response.raise_for_status()
        final_host = urlparse(response.url).hostname
        if final_host not in ALLOWED_OFFICIAL_HOSTS:
            raise ValueError(f"官方方法文件发生跨域跳转：{response.url}")
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ValueError("官方方法文件响应不是PDF")
        OFFICIAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
        output = OFFICIAL_PDF_DIR / "csi_stock_index_methodology_20230908.pdf"
        output.write_bytes(content)
        result.update(
            {
                "fetch_status": "SUCCESS",
                "document_sha256": sha256_bytes(content),
                "retrieved_at": datetime.now(TIMEZONE).isoformat(),
                "local_file": output.relative_to(ROOT).as_posix(),
            }
        )
    except Exception as exc:  # 方法文件失败必须进入报告，不允许无来源套用日期规则。
        result.update(
            {
                "fetch_status": "FAILED",
                "document_sha256": None,
                "retrieved_at": datetime.now(TIMEZONE).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def audit_membership_and_weights(fetch_official_pdfs: bool) -> dict[str, Any]:
    required = (
        MEMBERSHIP_FILE,
        WEIGHTS_FILE,
        CURRENT_WEIGHTS_FILE,
        OFFICIAL_CURRENT_RAW_FILE,
        MEMBERSHIP_STATUS_FILE,
        CALENDAR_FILE,
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"P1-1缺少输入：{missing}")
    input_hashes_before = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in required}
    membership = pd.read_parquet(MEMBERSHIP_FILE)
    weights = pd.read_parquet(WEIGHTS_FILE)
    current = pd.read_parquet(CURRENT_WEIGHTS_FILE)
    calendar = pd.read_parquet(CALENDAR_FILE)
    membership_status = json.loads(MEMBERSHIP_STATUS_FILE.read_text(encoding="utf-8"))
    current_raw = json.loads(OFFICIAL_CURRENT_RAW_FILE.read_text(encoding="utf-8"))
    trading_column = "trade_date" if "trade_date" in calendar else "date"
    trading_dates = pd.to_datetime(calendar[trading_column], errors="coerce").dropna()
    events = build_membership_events(membership)
    official_documents = fetch_official_change_lists(fetch_official_pdfs)
    official_methodology = fetch_official_methodology(fetch_official_pdfs)

    evidence_rows: list[dict[str, Any]] = []
    active_at_start = membership.loc[
        (pd.to_datetime(membership["opt_in"]) <= SCOPE_START)
        & (
            pd.to_datetime(membership["opt_out"], errors="coerce").isna()
            | (SCOPE_START < pd.to_datetime(membership["opt_out"], errors="coerce"))
        )
    ]
    evidence_rows.append(
        {
            "evidence_type": "BASELINE_STATE",
            "event_date": str(SCOPE_START.date()),
            "opt_in_count": 0,
            "opt_out_count": 0,
            "active_constituent_count": int(active_at_start["symbol"].nunique()),
            "source": membership["source"].iloc[0],
            "source_authority": "THIRD_PARTY_DERIVED_FROM_CLAIMED_OFFICIAL_ANNOUNCEMENTS",
            "source_file_sha256": input_hashes_before[MEMBERSHIP_FILE.relative_to(ROOT).as_posix()],
            "raw_source_sha256": membership_status.get("source_sha256"),
            "retrieved_at": membership_status.get("checked_at"),
            "retrieved_at_basis": "PIPELINE_REPORT_CHECKED_AT_NOT_ROW_LEVEL",
            "official_announcement_url": None,
            "official_announcement_date": None,
            "official_document_sha256": None,
            "official_change_set_verified": False,
            "official_effective_date_verified": False,
            "methodology_url": official_methodology["url"],
            "methodology_document_sha256": official_methodology.get("document_sha256"),
            "methodology_document_retrieved_at": official_methodology.get("retrieved_at"),
            "exception_code": "MISSING_OFFICIAL_BASELINE_SNAPSHOT_ARCHIVE",
        }
    )
    regular_rule_mismatches = 0
    non_trading_event_dates = 0
    change_set_verified_count = 0
    for row in events.to_dict("records"):
        event_date = pd.Timestamp(row["event_date"])
        key = event_date.strftime("%Y-%m")
        official = official_documents.get(key)
        dataset_opt_in_codes = {
            symbol.split(".", 1)[0] for symbol in row["opt_in_symbols"].split(";") if symbol
        }
        dataset_opt_out_codes = {
            symbol.split(".", 1)[0] for symbol in row["opt_out_symbols"].split(";") if symbol
        }
        official_opt_in_codes = (
            set(official.get("csi300_opt_in_codes", [])) if official else set()
        )
        official_opt_out_codes = (
            set(official.get("csi300_opt_out_codes", [])) if official else set()
        )
        change_set_verified = bool(official_opt_in_codes) and (
            dataset_opt_in_codes == official_opt_in_codes
            and dataset_opt_out_codes == official_opt_out_codes
        )
        if change_set_verified:
            change_set_verified_count += 1
        is_regular = event_date.month in {6, 12}
        expected_effective = (
            expected_regular_effective_date(event_date.year, event_date.month, trading_dates)
            if is_regular
            else None
        )
        rule_match = bool(expected_effective == event_date) if expected_effective is not None else None
        if is_regular and not rule_match:
            regular_rule_mismatches += 1
        event_is_trading_day = bool(event_date in set(pd.DatetimeIndex(trading_dates).normalize()))
        if not event_is_trading_day:
            non_trading_event_dates += 1
        exception_codes: list[str] = []
        if official is None:
            exception_codes.append("MISSING_OFFICIAL_ANNOUNCEMENT_ARCHIVE")
        elif official.get("fetch_status") != "SUCCESS":
            exception_codes.append("OFFICIAL_DOCUMENT_FETCH_FAILED")
        elif not change_set_verified:
            exception_codes.append("OFFICIAL_CHANGE_SET_MISMATCH")
        if not event_is_trading_day:
            exception_codes.append("DATASET_EVENT_DATE_IS_NOT_TRADING_DAY")
        if is_regular and not rule_match:
            exception_codes.append("DATASET_DATE_DIFFERS_FROM_METHODOLOGY_EXPECTATION")
        exception_codes.append("OFFICIAL_EFFECTIVE_DATE_NOT_ARCHIVE_VERIFIED")
        evidence_rows.append(
            {
                "evidence_type": "MEMBERSHIP_TRANSITION",
                "event_date": str(event_date.date()),
                "opt_in_count": int(row["opt_in_count"]),
                "opt_out_count": int(row["opt_out_count"]),
                "active_constituent_count": None,
                "transition_sha256": row["transition_sha256"],
                "source": membership["source"].iloc[0],
                "source_authority": "THIRD_PARTY_DERIVED_FROM_CLAIMED_OFFICIAL_ANNOUNCEMENTS",
                "source_file_sha256": input_hashes_before[MEMBERSHIP_FILE.relative_to(ROOT).as_posix()],
                "raw_source_sha256": membership_status.get("source_sha256"),
                "retrieved_at": membership_status.get("checked_at"),
                "retrieved_at_basis": "PIPELINE_REPORT_CHECKED_AT_NOT_ROW_LEVEL",
                "official_announcement_url": official.get("url") if official else None,
                "official_announcement_date": official.get("announcement_date") if official else None,
                "official_document_sha256": official.get("document_sha256") if official else None,
                "official_document_retrieved_at": official.get("retrieved_at") if official else None,
                "official_change_set_verified": change_set_verified,
                "official_effective_date_verified": False,
                "methodology_url": official_methodology["url"],
                "methodology_document_sha256": official_methodology.get("document_sha256"),
                "methodology_document_retrieved_at": official_methodology.get("retrieved_at"),
                "methodology_expected_effective_date": (
                    str(expected_effective.date()) if expected_effective is not None else None
                ),
                "dataset_date_matches_methodology_rule": rule_match,
                "dataset_event_date_is_trading_day": event_is_trading_day,
                "exception_code": ";".join(exception_codes),
            }
        )
    membership_evidence = pd.DataFrame(evidence_rows)

    weights = weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    weights["retrieved_at"] = pd.to_datetime(weights["retrieved_at"], errors="coerce")
    current["effective_date"] = pd.to_datetime(current["effective_date"])
    current_date = current["effective_date"].max()
    current_map = current.set_index("symbol")["weight_pct"].astype(float)
    weight_rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot in weights.groupby("trade_date", sort=True):
        snapshot_map = snapshot.set_index("con_code")["weight"].astype(float)
        latest_crosscheck = snapshot_date == current_date
        common = snapshot_map.index.intersection(current_map.index)
        max_difference = (
            float((snapshot_map.loc[common] - current_map.loc[common]).abs().max())
            if latest_crosscheck and len(common)
            else None
        )
        weight_rows.append(
            {
                "snapshot_date": str(pd.Timestamp(snapshot_date).date()),
                "constituent_count": int(snapshot["con_code"].nunique()),
                "weight_sum_pct": float(snapshot["weight"].sum()),
                "source": ";".join(sorted(snapshot["source"].astype(str).unique())),
                "source_authority": "TUSHARE_VENDOR_NOT_OFFICIAL_ARCHIVE",
                "retrieved_at_min": _iso(snapshot["retrieved_at"].min()),
                "retrieved_at_max": _iso(snapshot["retrieved_at"].max()),
                "source_file_sha256": input_hashes_before[WEIGHTS_FILE.relative_to(ROOT).as_posix()],
                "snapshot_content_sha256": canonical_rows_sha256(
                    snapshot.assign(trade_date=snapshot["trade_date"].dt.strftime("%Y-%m-%d")),
                    ["trade_date", "con_code", "weight"],
                ),
                "official_announcement_url": None,
                "official_announcement_date": None,
                "implementation_date": str(pd.Timestamp(snapshot_date).date()),
                "official_historical_snapshot_verified": False,
                "current_official_api_crosscheck": latest_crosscheck,
                "current_official_set_match": (
                    set(snapshot_map.index) == set(current_map.index) if latest_crosscheck else None
                ),
                "current_official_weight_max_abs_diff_pct": max_difference,
                "current_official_raw_sha256": (
                    input_hashes_before[OFFICIAL_CURRENT_RAW_FILE.relative_to(ROOT).as_posix()]
                    if latest_crosscheck
                    else None
                ),
                "exception_code": "UNIFIED_2026_VENDOR_RETRIEVAL_NO_OFFICIAL_HISTORICAL_ARCHIVE",
            }
        )
    weight_evidence = pd.DataFrame(weight_rows)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    membership_evidence.to_csv(MEMBERSHIP_EVIDENCE_FILE, index=False, encoding="utf-8-sig")
    weight_evidence.to_csv(WEIGHT_EVIDENCE_FILE, index=False, encoding="utf-8-sig")
    input_hashes_after = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in required}
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("P1-1审计期间输入文件发生变化，拒绝出具报告")
    current_official_crosscheck = weight_evidence.loc[
        weight_evidence["current_official_api_crosscheck"].eq(True)  # noqa: E712
    ].iloc[0]
    report = {
        "audit_id": "CSI300_POINT_IN_TIME_MEMBERSHIP_EVIDENCE_20260819",
        "status": "BLOCKED_POINT_IN_TIME_MEMBERSHIP",
        "audited_at": datetime.now(TIMEZONE).isoformat(),
        "scope": {"start": str(SCOPE_START.date()), "end": str(SCOPE_END.date())},
        "membership": {
            "interval_count": int(len(membership)),
            "unique_symbol_count": int(membership["symbol"].nunique()),
            "transition_event_count": int(len(events)),
            "baseline_active_constituent_count": int(active_at_start["symbol"].nunique()),
            "official_change_set_verified_event_count": change_set_verified_count,
            "fully_official_event_count": 0,
            "regular_event_methodology_date_mismatch_count": regular_rule_mismatches,
            "non_trading_event_date_count": non_trading_event_dates,
            "row_level_retrieved_at_present": False,
            "source_is_official": False,
            "evidence_file": MEMBERSHIP_EVIDENCE_FILE.relative_to(ROOT).as_posix(),
        },
        "weights": {
            "row_count": int(len(weights)),
            "snapshot_count": int(weights["trade_date"].nunique()),
            "first_snapshot": str(weights["trade_date"].min().date()),
            "last_snapshot": str(weights["trade_date"].max().date()),
            "official_historical_snapshot_verified_count": 0,
            "source_is_official": False,
            "retrieval_years": sorted(weights["retrieved_at"].dt.year.dropna().astype(int).unique().tolist()),
            "current_official_api_crosscheck": {
                "snapshot_date": current_official_crosscheck["snapshot_date"],
                "set_match": bool(current_official_crosscheck["current_official_set_match"]),
                "maximum_absolute_weight_difference_pct": current_official_crosscheck[
                    "current_official_weight_max_abs_diff_pct"
                ],
                "endpoint_base": current_raw.get("endpoint_base"),
                "retrieved_at": current_raw.get("retrieved_at"),
            },
            "evidence_file": WEIGHT_EVIDENCE_FILE.relative_to(ROOT).as_posix(),
        },
        "official_documents": official_documents,
        "official_methodology": official_methodology,
        "blocking_gaps": [
            "15个成员变更事件均缺少同时包含公告日、实施日和逐项名单的本地官方归档闭环。",
            "第三方成员区间没有逐行retrieved_at、官方公告URL或官方文件哈希。",
            "定期调样事件中存在数据日期与中证方法规则预期实施日不一致。",
            "120个月权重均由Tushare在2026年统一回取；0期具有同期中证官方快照归档。",
            "2026-07-31当前官网快照交叉一致只证明当前期，不能外推此前119期。",
        ],
        "governance": {
            "may_call_membership_fully_official": False,
            "may_call_weights_strict_point_in_time": False,
            "raw_inputs_modified": False,
        },
        "input_sha256": input_hashes_before,
    }
    MEMBERSHIP_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    MEMBERSHIP_REPORT_MD_FILE.write_text(render_membership_markdown(report), encoding="utf-8")
    return report


def deterministic_financial_sample(data: pd.DataFrame, target_size: int = 240) -> pd.DataFrame:
    if target_size < 200:
        raise ValueError("财务版本审计目标样本不得低于200")
    frame = data.copy()
    keys = ["con_code", "report_period", "available_at"]
    if frame[keys].duplicated().any():
        raise ValueError("财务主表存在重复事件主键")
    frame["audit_record_id"] = [
        sha256_bytes("|".join(map(str, values)).encode("utf-8"))
        for values in frame[keys].itertuples(index=False, name=None)
    ]
    frame["sample_rank"] = frame["audit_record_id"]
    reasons: dict[str, set[str]] = defaultdict(set)

    def select_one_per(columns: list[str], reason: str) -> None:
        ordered = frame.sort_values("sample_rank")
        selected = ordered.groupby(columns, dropna=False, sort=True).head(1)
        for record_id in selected["audit_record_id"]:
            reasons[str(record_id)].add(reason)

    select_one_per(["report_year", "period_type"], "YEAR_PERIOD")
    select_one_per(["industry_l1", "period_type"], "INDUSTRY_PERIOD")
    select_one_per(["income_report_type_label"], "INCOME_REPORT_TYPE")
    select_one_per(["balance_report_type_label"], "BALANCE_REPORT_TYPE")
    selected_ids = set(reasons)
    for record_id in frame.sort_values("sample_rank")["audit_record_id"]:
        if len(selected_ids) >= target_size:
            break
        text = str(record_id)
        if text not in selected_ids:
            selected_ids.add(text)
            reasons[text].add("HASH_FILL")
    sample = frame.loc[frame["audit_record_id"].isin(selected_ids)].copy()
    sample["selection_reasons"] = sample["audit_record_id"].map(
        lambda value: ";".join(sorted(reasons[str(value)]))
    )
    return sample.sort_values("audit_record_id").head(target_size).reset_index(drop=True)


def _checkpoint_path(api: str, period: pd.Timestamp) -> Path:
    name = period.strftime("%Y%m%d") + ".parquet"
    if period <= pd.Timestamp("2015-06-30"):
        return CHECKPOINT_ROOTS[1] / "vip" / api / name
    return CHECKPOINT_ROOTS[0] / api / name


def build_checkpoint_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "last_write_time_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=ZoneInfo("UTC")
                    ).isoformat(),
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def audit_financial_vintages(target_size: int = 240) -> dict[str, Any]:
    required = (FINANCIAL_FILE, INDUSTRY_STRATIFICATION_FILE)
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"P1-2缺少输入：{missing}")
    input_hashes_before = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in required}
    financials = pd.read_parquet(FINANCIAL_FILE)
    industry = pd.read_parquet(INDUSTRY_STRATIFICATION_FILE)
    for column in (
        "report_period",
        "available_at",
        "announcement_date",
        "retrieved_at",
        "income_ann_date",
        "balance_ann_date",
        "indicator_ann_date",
    ):
        financials[column] = pd.to_datetime(financials[column], errors="coerce")
    if len(financials) != 45198:
        raise ValueError(f"财务主表行数不是冻结审阅口径45198：{len(financials)}")
    financials["report_year"] = financials["report_period"].dt.year.astype("Int64")
    financials["period_type"] = financials["report_period"].map(_report_period_type)
    mapping = (
        industry[["con_code", "industry_l1"]]
        .drop_duplicates("con_code", keep="last")
        .set_index("con_code")["industry_l1"]
    )
    financials["industry_l1"] = financials["con_code"].map(mapping).fillna("UNMAPPED")
    financials["industry_stratum_source"] = "EXPLANATORY_STATIC_CLASSIFICATION_AUDIT_ONLY"
    financials["income_report_type_label"] = financials["income_report_type"].fillna("MISSING").astype(str)
    financials["balance_report_type_label"] = financials["balance_report_type"].fillna("MISSING").astype(str)
    sample = deterministic_financial_sample(financials, target_size=target_size)
    sample["original_publication_at"] = sample["announcement_date"]
    sample["original_publication_at_basis"] = (
        "PROVIDER_DECLARED_MAX_COMPONENT_ANN_DATE_NOT_ARCHIVE_VERIFIED"
    )
    sample["retrieval_lag_days"] = calendar_day_difference(
        sample["retrieved_at"], sample["original_publication_at"]
    )
    sample["available_after_publication_days"] = calendar_day_difference(
        sample["available_at"], sample["original_publication_at"]
    )
    sample["revision_possible"] = True
    sample["vintage_verified"] = False
    sample["vintage_status"] = "BLOCKED_VENDOR_REVISION_RISK"
    sample["raw_checkpoint_archive_is_contemporaneous"] = False
    for api, prefix in (
        ("income_vip", "income"),
        ("balancesheet_vip", "balance"),
        ("fina_indicator_vip", "indicator"),
    ):
        paths = [_checkpoint_path(api, pd.Timestamp(period)) for period in sample["report_period"]]
        sample[f"{prefix}_checkpoint_path"] = [
            path.relative_to(ROOT).as_posix() if path.exists() else None for path in paths
        ]
        sample[f"{prefix}_checkpoint_sha256"] = [
            sha256_file(path) if path.exists() else None for path in paths
        ]
    output_columns = [
        "audit_record_id",
        "selection_reasons",
        "con_code",
        "industry_l1",
        "industry_stratum_source",
        "report_period",
        "report_year",
        "period_type",
        "income_report_type_label",
        "balance_report_type_label",
        "income_ann_date",
        "balance_ann_date",
        "indicator_ann_date",
        "announcement_date",
        "original_publication_at",
        "original_publication_at_basis",
        "available_at",
        "retrieved_at",
        "retrieval_lag_days",
        "available_after_publication_days",
        "revision_possible",
        "vintage_verified",
        "vintage_status",
        "raw_checkpoint_archive_is_contemporaneous",
        "income_checkpoint_path",
        "income_checkpoint_sha256",
        "balance_checkpoint_path",
        "balance_checkpoint_sha256",
        "indicator_checkpoint_path",
        "indicator_checkpoint_sha256",
        "source",
    ]
    sample_output = sample[output_columns].copy()
    inventory = build_checkpoint_inventory()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    sample_output.to_csv(FINANCIAL_SAMPLE_FILE, index=False, encoding="utf-8-sig")
    inventory.to_csv(CHECKPOINT_INVENTORY_FILE, index=False, encoding="utf-8-sig")
    input_hashes_after = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in required}
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("P1-2审计期间输入文件发生变化，拒绝出具报告")
    source_retrievals = sorted(
        _iso(value) for value in financials["retrieved_at"].dropna().drop_duplicates().sort_values()
    )
    report = {
        "audit_id": "CSI300_FINANCIAL_VINTAGE_AUDIT_20260819",
        "status": "BLOCKED_VENDOR_REVISION_RISK",
        "audited_at": datetime.now(TIMEZONE).isoformat(),
        "population": {
            "row_count": int(len(financials)),
            "symbol_count": int(financials["con_code"].nunique()),
            "first_report_period": str(financials["report_period"].min().date()),
            "last_report_period": str(financials["report_period"].max().date()),
            "report_year_count": int(financials["report_year"].nunique()),
            "industry_stratum_count": int(financials["industry_l1"].nunique()),
            "period_types": sorted(financials["period_type"].unique().tolist()),
            "retrieved_at_values": source_retrievals,
            "source_sha256": input_hashes_before[FINANCIAL_FILE.relative_to(ROOT).as_posix()],
        },
        "sampling": {
            "method": "确定性SHA256分层抽样：年份×季度类型、行业×季度类型、两类供应商报告类型各取哈希最小记录，再按哈希补足。",
            "target_size": target_size,
            "actual_size": int(len(sample_output)),
            "covered_year_count": int(sample_output["report_year"].nunique()),
            "covered_years": sorted(sample_output["report_year"].astype(int).unique().tolist()),
            "covered_industry_count": int(sample_output["industry_l1"].nunique()),
            "covered_period_types": sorted(sample_output["period_type"].unique().tolist()),
            "covered_income_report_types": sorted(
                sample_output["income_report_type_label"].unique().tolist()
            ),
            "covered_balance_report_types": sorted(
                sample_output["balance_report_type_label"].unique().tolist()
            ),
            "sample_file": FINANCIAL_SAMPLE_FILE.relative_to(ROOT).as_posix(),
            "sample_sha256": sha256_file(FINANCIAL_SAMPLE_FILE),
        },
        "audit_results": {
            "retrieved_at_present_count": int(sample_output["retrieved_at"].notna().sum()),
            "original_publication_at_present_count": int(
                sample_output["original_publication_at"].notna().sum()
            ),
            "revision_possible_count": int(sample_output["revision_possible"].sum()),
            "vintage_verified_count": int(sample_output["vintage_verified"].sum()),
            "contemporaneous_raw_checkpoint_count": int(
                sample_output["raw_checkpoint_archive_is_contemporaneous"].sum()
            ),
            "all_three_checkpoint_hashes_present_count": int(
                sample_output[
                    [
                        "income_checkpoint_sha256",
                        "balance_checkpoint_sha256",
                        "indicator_checkpoint_sha256",
                    ]
                ]
                .notna()
                .all(axis=1)
                .sum()
            ),
        },
        "checkpoint_inventory": {
            "file_count": int(len(inventory)),
            "inventory_file": CHECKPOINT_INVENTORY_FILE.relative_to(ROOT).as_posix(),
            "inventory_sha256": sha256_file(CHECKPOINT_INVENTORY_FILE),
        },
        "blocking_gaps": [
            "45,198条事件由2026年统一回取的供应商响应构造，缺少各原公告日同期原始响应。",
            "available_at重建供应商声明的历史可得日，但不能证明该日看到的数值与2026回取值一致。",
            "供应商数据没有revision_id、前值、修订时间或历史vintage快照链。",
            "checkpoint文件证明本次回取输入，不是原公告日同期归档；抽样记录全部revision_possible=true。",
            "original_publication_at来自供应商ann_date字段合并，不是交易所公告原文逐条核验结果。",
        ],
        "governance": {
            "may_call_strict_point_in_time": False,
            "may_call_vintage_verified": False,
            "raw_inputs_modified": False,
            "return_or_model_calculation_performed": False,
        },
        "input_sha256": input_hashes_before,
    }
    FINANCIAL_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    FINANCIAL_REPORT_MD_FILE.write_text(render_financial_markdown(report), encoding="utf-8")
    return report


def render_membership_markdown(report: dict[str, Any]) -> str:
    membership = report["membership"]
    weights = report["weights"]
    gaps = "\n".join(f"- {item}" for item in report["blocking_gaps"])
    return f"""# 沪深300点时成员与权重证据审计

状态：`{report['status']}`

## 结论

- 范围：{report['scope']['start']} 至 {report['scope']['end']}。
- 成员区间共有 {membership['transition_event_count']} 个变更事件；只有 {membership['official_change_set_verified_event_count']} 个事件找到并匹配中证官方变更名单附件，完整官方事件为 0。
- 定期事件日期与方法规则预期不一致 {membership['regular_event_methodology_date_mismatch_count']} 次，非交易日事件日期 {membership['non_trading_event_date_count']} 次。
- 120个月权重全部来自供应商2026年统一回取，历史官方快照验证数为 {weights['official_historical_snapshot_verified_count']}。
- 最新一期当前官网API集合交叉匹配为 {weights['current_official_api_crosscheck']['set_match']}，但不能外推此前119期。

## 阻断缺口

{gaps}

## 证据表

- `{membership['evidence_file']}`
- `{weights['evidence_file']}`

不得将当前材料描述为“全部官方点时成员与权重”。
"""


def render_financial_markdown(report: dict[str, Any]) -> str:
    population = report["population"]
    sampling = report["sampling"]
    results = report["audit_results"]
    gaps = "\n".join(f"- {item}" for item in report["blocking_gaps"])
    return f"""# 沪深300财务事件历史版本审计

状态：`{report['status']}`

## 总体与抽样

- 总体：{population['row_count']:,}条、{population['symbol_count']}只证券、{population['report_year_count']}个报告年份。
- 确定性分层样本：{sampling['actual_size']}条，覆盖{sampling['covered_year_count']}个年份、{sampling['covered_industry_count']}个行业层和{len(sampling['covered_period_types'])}种季度类型。
- 样本中 `revision_possible=true`：{results['revision_possible_count']}条。
- 样本中 `vintage_verified=true`：{results['vintage_verified_count']}条。
- 三类checkpoint哈希齐全：{results['all_three_checkpoint_hashes_present_count']}条。

## 阻断缺口

{gaps}

## 证据表

- `{sampling['sample_file']}`
- `{report['checkpoint_inventory']['inventory_file']}`

本审计不修改原始数据，不把2026年统一回取值称为严格PIT。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="执行P1-1/P1-2数据证据审计")
    parser.add_argument(
        "--fetch-official-pdfs",
        action="store_true",
        help="仅从中证指数官方域名下载已定位的调样附件并计算哈希",
    )
    parser.add_argument("--financial-sample-size", type=int, default=240)
    args = parser.parse_args()
    membership = audit_membership_and_weights(args.fetch_official_pdfs)
    financial = audit_financial_vintages(args.financial_sample_size)
    print(
        json.dumps(
            {
                "membership_status": membership["status"],
                "financial_status": financial["status"],
                "membership_report": MEMBERSHIP_REPORT_FILE.relative_to(ROOT).as_posix(),
                "financial_report": FINANCIAL_REPORT_FILE.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
