"""510300 压力传导危险率 V2 的 G1 历史停牌证据修复核心。

本模块只做三件事：解析交易所官方停复牌记录、将这些记录与父版本全部
点时成分缺口逐行匹配、按原四态契约重新分类。它不接收 BAD10 事件表，
因此数据修复阶段无法按结果或事件日期挑选记录。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from research.stress_transmission_hazard_v2 import classify_constituent_returns


PROGRAM_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2"
EXECUTION_ID = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_HISTORICAL_REMEDIATION_V1"
)

OFFICIAL_INTERVAL_COLUMNS = [
    "exchange",
    "symbol",
    "start_at",
    "resume_at",
    "open_ended",
    "full_day_scope",
    "stop_reason",
    "raw_start",
    "raw_resume",
    "raw_stop_time",
    "raw_record_type",
    "source_kind",
    "source_url",
    "source_sha256",
    "source_record_id",
    "retrieved_at",
]

CLASSIFIER_OUTPUT_COLUMNS = [
    "constituent_return_state",
    "daily_total_shareholder_return",
    "return_is_usable",
    "state_reason",
]

USABLE_PARENT_STATE = "TRADED_VALID"
SUPPLIER_GAP_STATE = "SUPPLIER_MISSING_OR_CONFLICT"
OFFICIAL_SUSPENSION_STATE = "OFFICIAL_SUSPENSION"
ACTION_NONE_CONFIRMED = "NONE_CONFIRMED"


class HistoricalRemediationError(RuntimeError):
    """历史修复输入、来源或输出违反冻结契约。"""


@dataclass(frozen=True)
class RemediationArtifacts:
    """全量缺口匹配与四态重分类结果。"""

    target_evidence: pd.DataFrame
    remediated_classified_returns: pd.DataFrame
    metrics: dict[str, Any]


def sha256_bytes(content: bytes) -> str:
    """返回字节内容的 SHA-256。"""

    return hashlib.sha256(content).hexdigest()


def canonical_sha256(value: Any) -> str:
    """对 JSON 可序列化对象计算稳定摘要。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise HistoricalRemediationError(f"{label}缺少必需列：{missing}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.tz_localize(None).dt.normalize()
    if parsed.isna().any():
        raise HistoricalRemediationError(f"{label}存在不可解析日期")
    return parsed


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    mapped = values.map(
        {
            True: True,
            False: False,
            1: True,
            0: False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
            "1": True,
            "0": False,
        }
    )
    if mapped.isna().any():
        raise HistoricalRemediationError(f"{label}必须是严格布尔值")
    return mapped.astype(bool)


def _normalize_symbol(code: Any, exchange: str) -> str | None:
    match = re.search(r"(?<!\d)(\d{1,6})(?!\d)", str(code).strip())
    if match is None:
        return None
    digits = match.group(1).zfill(6)
    suffix = exchange.strip().upper()
    if suffix not in {"SH", "SZ"}:
        raise HistoricalRemediationError(f"未知交易所：{exchange}")
    return f"{digits}.{suffix}"


def _parse_date_digits(value: Any) -> pd.Timestamp | None:
    text = re.sub(r"\D", "", str(value or ""))
    if len(text) < 8 or text[:4] == "9999":
        return None
    parsed = pd.to_datetime(text[:8], format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _parse_official_timestamp(value: Any) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "nat", "none", "--", "-"}:
        return None
    if text.startswith("9999"):
        return None
    text = text.replace("年", "/").replace("月", "/").replace("日", " ")
    text = re.sub(r"\s+", " ", text).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_localize(None)


def _record_id(prefix: str, row: Mapping[str, Any], source_sha256: str) -> str:
    return f"{prefix}_{canonical_sha256({'row': dict(row), 'source_sha256': source_sha256})[:24]}"


def empty_official_intervals() -> pd.DataFrame:
    """返回具有稳定 schema 的空官方区间表。"""

    return pd.DataFrame(columns=OFFICIAL_INTERVAL_COLUMNS)


def parse_sse_official_response(
    content: bytes,
    *,
    source_url: str,
    retrieved_at: str,
) -> pd.DataFrame:
    """解析上交所 ``GW_PL_JYTS_TFPXX`` 官方 JSON 响应。"""

    source_sha = sha256_bytes(content)
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalRemediationError("上交所停复牌响应不是有效 UTF-8 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        raise HistoricalRemediationError("上交所停复牌响应缺少 result 数组")

    records: list[dict[str, Any]] = []
    for raw in payload["result"]:
        if not isinstance(raw, dict):
            raise HistoricalRemediationError("上交所停复牌 result 含非对象记录")
        product_code = str(raw.get("productCode") or "").strip()
        # 官方页面同时覆盖股票和可转债。当前研究输入只含沪市 A 股；先按
        # 产品身份排除转债、转股代码、基金与 B 股，避免把异类记录解释成股票。
        if re.fullmatch(r"6\d{5}", product_code) is None:
            continue
        symbol = _normalize_symbol(product_code, "SH")
        start_date = _parse_date_digits(raw.get("startStopDate"))
        if symbol is None or start_date is None:
            continue
        stop_time = str(raw.get("stopTime") or "").strip().upper()
        full_day_scope = stop_time in {"", "WH"}
        start_at = start_date + (
            pd.Timedelta(hours=13)
            if stop_time == "PM"
            else pd.Timedelta(hours=9, minutes=30)
        )
        resume_date = _parse_date_digits(raw.get("endStopDate"))
        if resume_date is None:
            resume_at = pd.NaT
        elif resume_date < start_date:
            raise HistoricalRemediationError("上交所停牌结束日早于起始日")
        elif resume_date > start_date:
            # 跨日记录的结束日按恢复交易日开盘处理，因而不把结束日计作
            # 整日停牌。该解释也比把结束日整日记零更保守。
            resume_at = resume_date + pd.Timedelta(hours=9, minutes=30)
        elif stop_time == "AM":
            # AM/PM 都会在下游因 full_day_scope=False 被拒绝；仍保留一个
            # 合法、可审计的盘中区间，避免误把半日记录扩张为整日。
            resume_at = resume_date + pd.Timedelta(hours=13)
        else:
            # 官方历史表把单日整日停牌记为起止同日（常见 WH/LSTP）。
            # 用当日收盘作为右端点，仅允许命中该交易日，不向后扩张。
            resume_at = resume_date + pd.Timedelta(hours=15)
        reason_parts = [
            str(raw.get("stopReason") or "").strip(),
            str(raw.get("endStopReason") or "").strip(),
        ]
        reason = "/".join(part for part in reason_parts if part)
        records.append(
            {
                "exchange": "SH",
                "symbol": symbol,
                "start_at": start_at,
                "resume_at": resume_at,
                "open_ended": resume_date is None,
                "full_day_scope": full_day_scope,
                "stop_reason": reason,
                "raw_start": str(raw.get("startStopDate") or ""),
                "raw_resume": str(raw.get("endStopDate") or ""),
                "raw_stop_time": stop_time,
                "raw_record_type": str(raw.get("type") or "").strip().upper(),
                "source_kind": "SSE_GW_PL_JYTS_TFPXX",
                "source_url": source_url,
                "source_sha256": source_sha,
                "source_record_id": _record_id("SSE_TFP", raw, source_sha),
                "retrieved_at": retrieved_at,
            }
        )
    return normalize_official_intervals(pd.DataFrame(records))


def decode_szse_html(content: bytes) -> str:
    """按深交所历史页常见编码解码，拒绝静默替换乱码。"""

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HistoricalRemediationError("深交所官方 HTML 既不是 UTF-8 也不是 GB18030")


def find_szse_suspension_document_url(content: bytes, *, main_url: str) -> str:
    """从深交所月报主页定位“证券停牌情况”子表。"""

    from urllib.parse import urljoin

    soup = BeautifulSoup(decode_szse_html(content), "html.parser")
    matches = [
        urljoin(main_url, str(anchor.get("href")))
        for anchor in soup.find_all("a", href=True)
        if "证券停牌情况" in anchor.get_text(" ", strip=True)
    ]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise HistoricalRemediationError(
            f"深交所月报应且只能有一个证券停牌情况子表：{main_url} -> {unique}"
        )
    return unique[0]


def parse_szse_official_document(
    content: bytes,
    *,
    source_url: str,
    report_month: str,
    retrieved_at: str,
) -> pd.DataFrame:
    """解析深交所月报中的官方“证券停牌情况”表。"""

    source_sha = sha256_bytes(content)
    soup = BeautifulSoup(decode_szse_html(content), "html.parser")
    records: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) < 5:
                continue
            product_code = str(cells[0]).strip()
            if re.fullmatch(r"[03]\d{5}", product_code) is None:
                continue
            symbol = _normalize_symbol(product_code, "SZ")
            start_at = _parse_official_timestamp(cells[-2])
            if symbol is None or start_at is None:
                continue
            resume_at = _parse_official_timestamp(cells[-1])
            raw = {
                "code": cells[0],
                "name": cells[1],
                "reason": cells[2],
                "start": cells[-2],
                "resume": cells[-1],
                "report_month": report_month,
            }
            records.append(
                {
                    "exchange": "SZ",
                    "symbol": symbol,
                    "start_at": start_at,
                    "resume_at": resume_at if resume_at is not None else pd.NaT,
                    "open_ended": resume_at is None,
                    "full_day_scope": True,
                    "stop_reason": cells[2],
                    "raw_start": cells[-2],
                    "raw_resume": cells[-1],
                    "raw_stop_time": "",
                    "raw_record_type": "SZSE_MONTHLY_SUSPENSION_TABLE_ROW",
                    "source_kind": "SZSE_MONTHLY_SUSPENSION_TABLE",
                    "source_url": source_url,
                    "source_sha256": source_sha,
                    "source_record_id": _record_id("SZSE_TFP", raw, source_sha),
                    "retrieved_at": retrieved_at,
                }
            )
    if not records:
        text = soup.get_text(" ", strip=True)
        if "证券停牌情况" not in text:
            raise HistoricalRemediationError(f"深交所子表不含证券停牌情况标题：{source_url}")
        return empty_official_intervals()
    return normalize_official_intervals(pd.DataFrame(records))


def normalize_official_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """规范并验证官方区间表。"""

    if frame.empty:
        return empty_official_intervals()
    _require_columns(frame, OFFICIAL_INTERVAL_COLUMNS, "官方停复牌区间")
    result = frame[OFFICIAL_INTERVAL_COLUMNS].copy()
    result["exchange"] = result["exchange"].astype("string").str.strip().str.upper()
    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    result["start_at"] = pd.to_datetime(result["start_at"], errors="coerce")
    result["resume_at"] = pd.to_datetime(result["resume_at"], errors="coerce")
    result["open_ended"] = _strict_bool(result["open_ended"], "open_ended")
    result["full_day_scope"] = _strict_bool(result["full_day_scope"], "full_day_scope")
    if result["start_at"].isna().any():
        raise HistoricalRemediationError("官方停复牌区间存在空起始时间")
    if not result["exchange"].isin({"SH", "SZ"}).all():
        raise HistoricalRemediationError("官方停复牌区间混入未知交易所")
    suffix = result["symbol"].str[-2:]
    if not suffix.eq(result["exchange"]).all():
        raise HistoricalRemediationError("官方停复牌区间代码后缀与交易所不一致")
    explicit = ~result["open_ended"]
    if result.loc[explicit, "resume_at"].isna().any():
        raise HistoricalRemediationError("非开放区间缺少复牌时间")
    if (
        result.loc[explicit, "resume_at"]
        .le(result.loc[explicit, "start_at"])
        .any()
    ):
        raise HistoricalRemediationError("官方停复牌区间终点不晚于起点")
    for column in (
        "stop_reason",
        "raw_start",
        "raw_resume",
        "raw_stop_time",
        "raw_record_type",
        "source_kind",
        "source_url",
        "source_sha256",
        "source_record_id",
        "retrieved_at",
    ):
        result[column] = result[column].astype("string").fillna("")
    if result["source_record_id"].eq("").any():
        raise HistoricalRemediationError("官方停复牌区间缺少来源记录 ID")
    result = result.drop_duplicates("source_record_id", keep="first")
    return result.sort_values(
        ["exchange", "symbol", "start_at", "source_record_id"], kind="stable"
    ).reset_index(drop=True)


def combine_official_intervals(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """合并多页官方区间并保持确定性。"""

    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return empty_official_intervals()
    return normalize_official_intervals(pd.concat(nonempty, ignore_index=True))


def _normalize_parent_inputs(
    classified_returns: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(
        classified_returns,
        [
            "date",
            "symbol",
            "source_observed",
            "supplier_conflict",
            "official_suspension",
            "corporate_action_status",
            "corporate_action_evidence_id",
            "previous_unadjusted_close",
            "unadjusted_close",
            *CLASSIFIER_OUTPUT_COLUMNS,
        ],
        "父四态收益",
    )
    _require_columns(membership, ["membership_date", "index_code", "symbol"], "点时成员")
    classified = classified_returns.copy()
    classified["date"] = _normalize_dates(classified["date"], "父四态收益日期")
    classified["symbol"] = classified["symbol"].astype("string").str.strip().str.upper()
    if classified.duplicated(["date", "symbol"]).any():
        raise HistoricalRemediationError("父四态收益存在重复 date/symbol")
    if classified["date"].gt(cutoff).any():
        raise HistoricalRemediationError("父四态收益含历史截止日之后记录")
    for column in ("source_observed", "supplier_conflict", "official_suspension", "return_is_usable"):
        classified[column] = _strict_bool(classified[column], column)

    members = membership.copy()
    members["date"] = _normalize_dates(members["membership_date"], "点时成员日期")
    members["symbol"] = members["symbol"].astype("string").str.strip().str.upper()
    members["index_code"] = members["index_code"].astype("string").str.strip()
    if not members["index_code"].eq("000300").all():
        raise HistoricalRemediationError("点时成员混入非 000300 研究输入")
    if members.duplicated(["date", "symbol"]).any():
        raise HistoricalRemediationError("点时成员存在重复 date/symbol")
    if members["date"].gt(cutoff).any():
        raise HistoricalRemediationError("点时成员含历史截止日之后记录")
    counts = members.groupby("date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise HistoricalRemediationError(
            f"点时成员每日必须为300只：min={int(counts.min())}, max={int(counts.max())}"
        )

    target_keys = members[["date", "symbol"]].merge(
        classified[["date", "symbol", "source_observed", "constituent_return_state"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if target_keys["source_observed"].isna().any():
        raise HistoricalRemediationError("点时成员在父四态收益中存在缺行")
    target = target_keys.loc[~target_keys["source_observed"].astype(bool)].copy()
    if not target["constituent_return_state"].eq(SUPPLIER_GAP_STATE).all():
        raise HistoricalRemediationError("父版本未观测成员日并非全部为供应商缺口态")
    return classified, members, target[["date", "symbol"]]


def _prepare_dividend_action_dates(
    dividend_actions: pd.DataFrame,
) -> set[tuple[str, pd.Timestamp]]:
    _require_columns(dividend_actions, ["ts_code", "ex_date"], "公司行动候选")
    frame = dividend_actions[["ts_code", "ex_date"]].copy()
    frame["symbol"] = frame["ts_code"].astype("string").str.strip().str.upper()
    raw = frame["ex_date"].astype("string").str.replace(r"\D", "", regex=True)
    frame["date"] = pd.to_datetime(raw.where(raw.str.len().eq(8)), format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["date"])
    return set(zip(frame["symbol"].astype(str), frame["date"].map(pd.Timestamp)))


def _prior_valid_observations(
    classified: pd.DataFrame,
) -> dict[str, tuple[list[pd.Timestamp], list[float]]]:
    close = pd.to_numeric(classified["unadjusted_close"], errors="coerce")
    valid = (
        classified["constituent_return_state"].eq(USABLE_PARENT_STATE)
        & classified["source_observed"]
        & close.notna()
        & np.isfinite(close)
        & close.gt(0.0)
    )
    subset = classified.loc[valid, ["date", "symbol"]].assign(close=close.loc[valid])
    result: dict[str, tuple[list[pd.Timestamp], list[float]]] = {}
    for symbol, group in subset.sort_values(["symbol", "date"], kind="stable").groupby("symbol"):
        result[str(symbol)] = (
            [pd.Timestamp(value) for value in group["date"]],
            [float(value) for value in group["close"]],
        )
    return result


def _first_observed_after(
    classified: pd.DataFrame,
) -> dict[str, list[pd.Timestamp]]:
    observed = classified.loc[classified["source_observed"], ["symbol", "date"]]
    return {
        str(symbol): [pd.Timestamp(value) for value in group["date"].sort_values()]
        for symbol, group in observed.groupby("symbol")
    }


def _effective_resume_at(
    row: Any,
    observed_dates: Mapping[str, list[pd.Timestamp]],
) -> pd.Timestamp | None:
    explicit = None if pd.isna(row.resume_at) else pd.Timestamp(row.resume_at)
    dates = observed_dates.get(str(row.symbol), [])
    start_date = pd.Timestamp(row.start_at).normalize()
    position = bisect_right(dates, start_date)
    observed_resume = (
        dates[position] + pd.Timedelta(hours=9, minutes=30)
        if position < len(dates)
        else None
    )
    candidates = [value for value in (explicit, observed_resume) if value is not None]
    return min(candidates) if candidates else None


def _build_exact_interval_matches(
    *,
    target: pd.DataFrame,
    classified: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    normalized = normalize_official_intervals(intervals)
    normalized = normalized.loc[normalized["full_day_scope"]].copy()
    observed_dates = _first_observed_after(classified)
    candidate_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in normalized.itertuples(index=False):
        effective_resume = _effective_resume_at(row, observed_dates)
        candidate_by_symbol.setdefault(str(row.symbol), []).append(
            {
                "start_at": pd.Timestamp(row.start_at),
                "effective_resume_at": effective_resume,
                "source_record_id": str(row.source_record_id),
                "source_kind": str(row.source_kind),
                "source_url": str(row.source_url),
                "source_sha256": str(row.source_sha256),
                "stop_reason": str(row.stop_reason),
                "raw_stop_time": str(row.raw_stop_time),
                "raw_record_type": str(row.raw_record_type),
                "official_start_at": pd.Timestamp(row.start_at),
                "official_resume_at": (
                    pd.NaT if pd.isna(row.resume_at) else pd.Timestamp(row.resume_at)
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    for target_row in target.sort_values(["symbol", "date"], kind="stable").itertuples(index=False):
        date = pd.Timestamp(target_row.date)
        session_open = date + pd.Timedelta(hours=9, minutes=30)
        session_close = date + pd.Timedelta(hours=15)
        matches: list[dict[str, Any]] = []
        for candidate in candidate_by_symbol.get(str(target_row.symbol), []):
            if candidate["start_at"] > session_open:
                continue
            end = candidate["effective_resume_at"]
            if end is not None and end < session_close:
                continue
            matches.append(candidate)
        if matches:
            chosen = sorted(
                matches,
                key=lambda item: (item["start_at"], item["source_record_id"]),
                reverse=True,
            )[0]
            rows.append(
                {
                    "date": date,
                    "symbol": str(target_row.symbol),
                    "official_interval_matched": True,
                    **chosen,
                }
            )
        else:
            rows.append(
                {
                    "date": date,
                    "symbol": str(target_row.symbol),
                    "official_interval_matched": False,
                    "start_at": pd.NaT,
                    "effective_resume_at": pd.NaT,
                    "source_record_id": "",
                    "source_kind": "",
                    "source_url": "",
                    "source_sha256": "",
                    "stop_reason": "",
                    "raw_stop_time": "",
                    "raw_record_type": "",
                    "official_start_at": pd.NaT,
                    "official_resume_at": pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def remediate_official_suspensions(
    *,
    classified_returns: pd.DataFrame,
    membership: pd.DataFrame,
    dividend_actions: pd.DataFrame,
    official_intervals: pd.DataFrame,
    dividend_source_sha256: str,
    historical_cutoff: str,
    expected_target_rows: int | None = None,
) -> RemediationArtifacts:
    """对父版本全部未观测点时成员日进行结果盲的官方停牌匹配。"""

    cutoff = pd.Timestamp(historical_cutoff)
    classified, _members, target = _normalize_parent_inputs(
        classified_returns,
        membership,
        cutoff=cutoff,
    )
    if expected_target_rows is not None and len(target) != int(expected_target_rows):
        raise HistoricalRemediationError(
            f"全量修复目标行数漂移：actual={len(target)}, expected={expected_target_rows}"
        )
    intervals = normalize_official_intervals(official_intervals)
    if intervals["start_at"].dt.normalize().gt(cutoff).any():
        raise HistoricalRemediationError("官方区间含历史截止日之后的起始记录")
    evidence = _build_exact_interval_matches(
        target=target,
        classified=classified,
        intervals=intervals,
    )

    action_dates = _prepare_dividend_action_dates(dividend_actions)
    prior = _prior_valid_observations(classified)
    action_candidate: list[bool] = []
    prior_dates: list[pd.Timestamp | pd.NaT] = []
    prior_closes: list[float] = []
    for row in evidence.itertuples(index=False):
        key = (str(row.symbol), pd.Timestamp(row.date))
        action_candidate.append(key in action_dates)
        dates, closes = prior.get(str(row.symbol), ([], []))
        position = bisect_right(dates, pd.Timestamp(row.date) - pd.Timedelta(nanoseconds=1)) - 1
        if position >= 0:
            prior_dates.append(dates[position])
            prior_closes.append(closes[position])
        else:
            prior_dates.append(pd.NaT)
            prior_closes.append(float("nan"))
    evidence["dividend_action_candidate_on_date"] = action_candidate
    evidence["previous_valid_observed_date"] = prior_dates
    evidence["previous_valid_unadjusted_close"] = prior_closes
    evidence["previous_valid_close_available"] = (
        pd.Series(prior_closes).notna()
        & np.isfinite(np.asarray(prior_closes, dtype=float))
        & pd.Series(prior_closes).gt(0.0)
    )
    evidence["promotable_to_official_suspension"] = (
        evidence["official_interval_matched"].astype(bool)
        & ~evidence["dividend_action_candidate_on_date"].astype(bool)
        & evidence["previous_valid_close_available"].astype(bool)
    )
    evidence["block_reason"] = np.select(
        [
            ~evidence["official_interval_matched"].astype(bool),
            evidence["dividend_action_candidate_on_date"].astype(bool),
            ~evidence["previous_valid_close_available"].astype(bool),
        ],
        [
            "NO_MATCHING_EXCHANGE_OFFICIAL_FULL_DAY_INTERVAL",
            "DIVIDEND_ACTION_CANDIDATE_REQUIRES_SEPARATE_RECONCILIATION",
            "NO_PRIOR_VALID_UNADJUSTED_CLOSE",
        ],
        default="PROMOTABLE_WITH_OFFICIAL_INTERVAL_AND_NO_ACTION_CANDIDATE",
    )
    evidence["target_scope"] = "ALL_PARENT_SOURCE_OBSERVED_FALSE_PIT_MEMBER_DAYS"
    evidence["event_or_label_used_for_selection"] = False
    evidence["missing_return_zero_filled"] = False

    work = classified.copy()
    promote = evidence.loc[evidence["promotable_to_official_suspension"]].copy()
    if not promote.empty:
        promote_keys = promote.set_index(["date", "symbol"])
        work = work.set_index(["date", "symbol"], drop=False)
        for key, row in promote_keys.iterrows():
            official_id = str(row["source_record_id"])
            action_id = (
                "NO_ACTION_CANDIDATE_"
                + canonical_sha256(
                    {
                        "symbol": key[1],
                        "date": pd.Timestamp(key[0]).date().isoformat(),
                        "dividend_source_sha256": dividend_source_sha256,
                    }
                )[:24]
            )
            work.at[key, "previous_unadjusted_close"] = float(
                row["previous_valid_unadjusted_close"]
            )
            work.at[key, "source_observed"] = False
            work.at[key, "supplier_conflict"] = False
            work.at[key, "official_suspension"] = True
            work.at[key, "suspension_evidence_id"] = official_id
            work.at[key, "corporate_action_status"] = ACTION_NONE_CONFIRMED
            work.at[key, "corporate_action_evidence_id"] = action_id
            work.at[key, "cash_distribution_per_pre_event_share"] = 0.0
            work.at[key, "post_to_pre_share_ratio"] = 1.0
            work.at[key, "subscription_cash_outflow_per_pre_event_share"] = 0.0
            work.at[key, "daily_source"] = str(row["source_kind"])
            work.at[key, "previous_observed_date"] = pd.Timestamp(
                row["previous_valid_observed_date"]
            )
            work.at[key, "action_resolution_reason"] = (
                "NO_DIVIDEND_ACTION_CANDIDATE_ON_OFFICIAL_SUSPENSION_DATE"
            )
        work = work.reset_index(drop=True)

    classifier_inputs = work.drop(columns=CLASSIFIER_OUTPUT_COLUMNS)
    remediated = classify_constituent_returns(classifier_inputs)
    remediated = remediated.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    parent_sorted = classified.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    promoted_keys = set(zip(promote["date"], promote["symbol"]))
    untouched = pd.Series(
        [
            (pd.Timestamp(date), str(symbol)) not in promoted_keys
            for date, symbol in zip(parent_sorted["date"], parent_sorted["symbol"])
        ]
    )
    for column in ("constituent_return_state", "return_is_usable", "state_reason"):
        left = parent_sorted.loc[untouched, column].reset_index(drop=True)
        right = remediated.loc[untouched, column].reset_index(drop=True)
        if not left.astype(str).equals(right.astype(str)):
            raise HistoricalRemediationError(f"未触及父行的 {column} 被意外改写")
    parent_return = pd.to_numeric(
        parent_sorted.loc[untouched, "daily_total_shareholder_return"], errors="coerce"
    ).to_numpy(dtype=float)
    new_return = pd.to_numeric(
        remediated.loc[untouched, "daily_total_shareholder_return"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(parent_return, new_return, rtol=0.0, atol=1e-15, equal_nan=True):
        raise HistoricalRemediationError("未触及父行的收益被意外改写")

    promoted_output = remediated.merge(
        promote[["date", "symbol"]], on=["date", "symbol"], how="inner", validate="one_to_one"
    )
    if not promoted_output["constituent_return_state"].eq(OFFICIAL_SUSPENSION_STATE).all():
        raise HistoricalRemediationError("有官方证据的目标行未全部进入官方停牌态")
    if not pd.to_numeric(
        promoted_output["daily_total_shareholder_return"], errors="coerce"
    ).eq(0.0).all():
        raise HistoricalRemediationError("官方停牌态没有严格产生 0 收益")
    unusable = ~remediated["return_is_usable"].astype(bool)
    if remediated.loc[unusable, "daily_total_shareholder_return"].notna().any():
        raise HistoricalRemediationError("不可用四态被填入收益")

    metrics = {
        "target_member_day_count": int(len(evidence)),
        "target_symbol_count": int(evidence["symbol"].nunique()),
        "official_interval_record_count": int(len(intervals)),
        "official_interval_matched_member_day_count": int(
            evidence["official_interval_matched"].sum()
        ),
        "promoted_official_suspension_member_day_count": int(
            evidence["promotable_to_official_suspension"].sum()
        ),
        "blocked_by_action_candidate_count": int(
            evidence["dividend_action_candidate_on_date"].sum()
        ),
        "blocked_by_missing_prior_close_count": int(
            (~evidence["previous_valid_close_available"]).sum()
        ),
        "unmatched_member_day_count": int(
            (~evidence["official_interval_matched"]).sum()
        ),
        "event_or_label_used_for_selection": False,
        "parent_rows_modified_outside_promoted_set": 0,
    }
    return RemediationArtifacts(
        target_evidence=evidence.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True),
        remediated_classified_returns=remediated,
        metrics=metrics,
    )


def extract_szse_month_main_urls(index_content: bytes, *, index_url: str) -> list[str]:
    """从深交所月报索引提取所有月报主页 URL。"""

    from urllib.parse import urljoin

    text = decode_szse_html(index_content)
    relative = re.findall(r"value\s*:\s*['\"](\./t\d{8}_\d+\.html)['\"]", text)
    urls = sorted(set(urljoin(index_url, value) for value in relative))
    if not urls:
        raise HistoricalRemediationError("深交所月报索引未提取到任何月报主页")
    return urls


def extract_szse_report_month(main_content: bytes) -> str:
    """从月报主页标题提取 YYYY-MM。"""

    text = decode_szse_html(main_content)
    match = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*深圳证券交易所市场统计月报", text)
    if match is None:
        raise HistoricalRemediationError("深交所月报主页缺少可解析报告月份")
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"


def utc_or_local_iso_now() -> str:
    """便于调用方注入统一抓取时间；默认返回带时区的本地时间字符串。"""

    return datetime.now().astimezone().isoformat()
