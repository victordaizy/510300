"""510300 非对称压力风险 V1 的点时 M/F/T 数据源准入。

本模块只检查来源身份、点时时钟、结构完整性和缺失处理边界。它不构造
M/F/T 特征值，不读取 BAD10 标签明细，不训练模型，也不读取组合结果。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    RegisteredFileEvidence,
    normalize_project_relative_path,
    read_json_strict,
    validate_registered_inputs,
)


PROGRAM_ID = "510300_ASYMMETRIC_STRESS_HAZARD_V1"
NO_VIEW_RULE = "NO_VIEW_NO_INTERPOLATION"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    source_id: str,
) -> None:
    """拒绝列缺失；不猜测同义列。"""

    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise EvidenceContractError(f"{source_id} 缺少必需列：{missing}")


def _dates(series: pd.Series, *, source_id: str, field: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise EvidenceContractError(f"{source_id}.{field} 含不可解析日期")
    return parsed


def _finite_positive(series: pd.Series, *, source_id: str, field: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        raise EvidenceContractError(
            f"{source_id}.{field} 含空值、非有限值或非正值：{int(invalid.sum())}"
        )
    return values.astype(float)


def _result(
    *,
    source_id: str,
    admitted: bool,
    status: str,
    date_level_no_view_required: bool,
    metrics: Mapping[str, Any],
    availability_clock: str | None = None,
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "admitted": bool(admitted),
        "status": status,
        "whole_source_blocked": not bool(admitted),
        "date_level_no_view_required": bool(date_level_no_view_required),
        "missing_value_rule": NO_VIEW_RULE,
        "availability_clock": availability_clock,
        "metrics": dict(metrics),
        "limitations": list(limitations),
        "feature_values_constructed": False,
    }


def evaluate_etf_price(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    observation_start: str,
    observation_cutoff: str,
) -> tuple[dict[str, Any], pd.DatetimeIndex]:
    source_id = "etf_price"
    require_columns(
        frame, list(contract["required_columns"]), source_id=source_id
    )
    work = frame.loc[:, list(contract["required_columns"])].copy()
    work["date"] = _dates(work["date"], source_id=source_id, field="date")
    work = work.loc[
        work["date"].between(
            pd.Timestamp(observation_start), pd.Timestamp(observation_cutoff)
        )
    ].copy()
    if work.empty:
        raise EvidenceContractError("etf_price 在冻结观察窗内为空")
    if work["date"].duplicated().any():
        raise EvidenceContractError("etf_price 在冻结观察窗内存在重复日期")
    expected_symbol = str(contract["expected_symbol"])
    if set(work["symbol"].astype(str)) != {expected_symbol}:
        raise EvidenceContractError("etf_price 出现非 510300.SH 标的")
    _finite_positive(work["open"], source_id=source_id, field="open")
    _finite_positive(work["close"], source_id=source_id, field="close")
    market_dates = pd.DatetimeIndex(work["date"]).sort_values()
    if market_dates[0] != pd.Timestamp(observation_start):
        raise EvidenceContractError("etf_price 首日与冻结观察起点不一致")
    if market_dates[-1] != pd.Timestamp(observation_cutoff):
        raise EvidenceContractError("etf_price 末日与冻结观察截止日不一致")
    result = _result(
        source_id=source_id,
        admitted=True,
        status="PASS_FROZEN_ETF_PRICE_AND_TRADING_CALENDAR",
        date_level_no_view_required=False,
        availability_clock="SAME_DAY_CLOSE_PUBLIC_OBSERVATION",
        metrics={
            "row_count": int(len(work)),
            "session_count": int(len(market_dates)),
            "first_session": market_dates[0].date().isoformat(),
            "last_session": market_dates[-1].date().isoformat(),
            "duplicate_date_count": 0,
        },
    )
    return result, market_dates


def evaluate_etf_dividends(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = "etf_dividends"
    required = list(contract["required_columns"])
    require_columns(frame, required, source_id=source_id)
    work = frame.loc[:, required].copy()
    if work.empty:
        raise EvidenceContractError("etf_dividends 为空")
    if set(work["symbol"].astype(str)) != {str(contract["expected_symbol"])}:
        raise EvidenceContractError("etf_dividends 出现非 510300.SH 标的")
    for field in ("record_date", "ex_date", "payment_date"):
        work[field] = _dates(work[field], source_id=source_id, field=field)
    values = pd.to_numeric(work["cash_dividend_per_share"], errors="coerce")
    if values.isna().any() or (~np.isfinite(values)).any() or (values <= 0).any():
        raise EvidenceContractError("etf_dividends 现金分红必须为有限正值")
    if (work["record_date"] > work["ex_date"]).any():
        raise EvidenceContractError("etf_dividends 存在登记日晚于除息日")
    if (work["ex_date"] > work["payment_date"]).any():
        raise EvidenceContractError("etf_dividends 存在除息日晚于支付日")
    if work["source"].isna().any() or work["source"].astype(str).str.strip().eq("").any():
        raise EvidenceContractError("etf_dividends 存在空来源")
    if work.duplicated(["symbol", "record_date", "ex_date"]).any():
        raise EvidenceContractError("etf_dividends 存在重复分红记录")
    return _result(
        source_id=source_id,
        admitted=True,
        status="PASS_FROZEN_OFFICIAL_DIVIDEND_LEDGER",
        date_level_no_view_required=False,
        availability_clock="OFFICIAL_ANNOUNCEMENT_AND_CORPORATE_ACTION_DATES",
        metrics={
            "row_count": int(len(work)),
            "first_record_date": work["record_date"].min().date().isoformat(),
            "last_record_date": work["record_date"].max().date().isoformat(),
            "duplicate_record_count": 0,
        },
    )


def _nested_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise EvidenceContractError(
                f"官方原始快照缺少 payload 路径：{dotted_path}"
            )
        current = current[part]
    return current


def evaluate_official_pe(
    frame: pd.DataFrame,
    raw_snapshot: Mapping[str, Any],
    market_dates: pd.DatetimeIndex,
    *,
    contract: Mapping[str, Any],
    raw_contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = "csi300_official_pe"
    required = list(contract["required_columns"])
    require_columns(frame, required, source_id=source_id)
    work = frame.loc[:, required].copy()
    work["date"] = _dates(work["date"], source_id=source_id, field="date")
    work["pe_official"] = _finite_positive(
        work["pe_official"], source_id=source_id, field="pe_official"
    )
    if work["date"].duplicated().any():
        raise EvidenceContractError("csi300_official_pe 存在重复日期")
    if set(work["index_code"].astype(str)) != {str(contract["expected_index_code"])}:
        raise EvidenceContractError("csi300_official_pe 指数代码不一致")
    if set(work["provider_original_field"].astype(str)) != {
        str(contract["provider_field"])
    }:
        raise EvidenceContractError("csi300_official_pe 原始字段不是冻结的 peg")
    if set(work["source"].astype(str)) != {str(contract["expected_source"])}:
        raise EvidenceContractError("csi300_official_pe 来源身份不一致")
    if contract.get("availability_rule") != (
        "NEXT_TRADING_DAY_OPEN_BECAUSE_EXACT_PUBLICATION_TIME_NOT_ARCHIVED"
    ):
        raise EvidenceContractError("csi300_official_pe 可得时钟发生漂移")

    raw_rows = _nested_value(
        raw_snapshot, str(raw_contract["expected_payload_path"])
    )
    if not isinstance(raw_rows, list):
        raise EvidenceContractError("官方 PE 原始 payload 不是列表")
    date_field = str(raw_contract["expected_date_field"])
    value_field = str(raw_contract["expected_value_field"])
    raw_frame = pd.DataFrame(raw_rows)
    require_columns(raw_frame, [date_field, value_field], source_id="official_pe_raw")
    raw_dates = pd.to_datetime(
        raw_frame[date_field].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    raw_values = pd.to_numeric(raw_frame[value_field], errors="coerce")
    if raw_dates.isna().any() or raw_values.isna().any():
        raise EvidenceContractError("官方 PE 原始 payload 含不可解析值")
    raw_compare = pd.DataFrame(
        {"date": raw_dates, "pe_official": raw_values.astype(float)}
    ).sort_values("date", ignore_index=True)
    normalized = work[["date", "pe_official"]].sort_values(
        "date", ignore_index=True
    )
    if len(raw_compare) != len(normalized):
        raise EvidenceContractError("官方 PE 原始 payload 与 parquet 行数不一致")
    if not raw_compare["date"].equals(normalized["date"]):
        raise EvidenceContractError("官方 PE 原始 payload 与 parquet 日期不一致")
    if not np.allclose(
        raw_compare["pe_official"].to_numpy(),
        normalized["pe_official"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise EvidenceContractError("官方 PE 原始 payload 与 parquet 数值不一致")

    source_dates = pd.DatetimeIndex(normalized["date"])
    span = market_dates[
        (market_dates >= source_dates.min())
        & (market_dates <= min(source_dates.max(), market_dates.max()))
    ]
    missing_in_span = span.difference(source_dates)
    if len(missing_in_span):
        raise EvidenceContractError("官方 PE 在自身覆盖区间缺少 ETF 交易日")
    leading = int((market_dates < source_dates.min()).sum())
    trailing = int((market_dates > source_dates.max()).sum())
    return _result(
        source_id=source_id,
        admitted=True,
        status="PASS_OFFICIAL_PE_WITH_CONSERVATIVE_AVAILABILITY_CLOCK",
        date_level_no_view_required=bool(leading or trailing),
        availability_clock=str(contract["availability_rule"]),
        metrics={
            "row_count": int(len(normalized)),
            "first_observation_date": source_dates.min().date().isoformat(),
            "last_observation_date": source_dates.max().date().isoformat(),
            "market_sessions_in_source_span": int(len(span)),
            "missing_market_sessions_in_source_span": 0,
            "leading_no_view_market_sessions": leading,
            "trailing_no_view_market_sessions": trailing,
            "raw_payload_row_count": int(len(raw_compare)),
            "raw_payload_exact_match": True,
        },
        limitations=[
            "EXACT_PUBLICATION_TIME_NOT_ARCHIVED_USE_NEXT_TRADING_DAY_OPEN"
        ],
    )


def evaluate_cgb10y(
    frame: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = "china_10y_government_bond_yield"
    required = list(contract["required_columns"])
    require_columns(frame, required, source_id=source_id)
    work = frame.loc[:, required].copy()
    work["date"] = _dates(work["date"], source_id=source_id, field="date")
    work["cgb_10y"] = _finite_positive(
        work["cgb_10y"], source_id=source_id, field="cgb_10y"
    )
    if work["date"].duplicated().any():
        raise EvidenceContractError("china_10y_government_bond_yield 存在重复日期")
    if set(work["source"].astype(str)) != {str(contract["expected_source"])}:
        raise EvidenceContractError("中国国债 10 年收益率来源身份不一致")
    if contract.get("raw_payload_archive_present") is not False:
        raise EvidenceContractError("中国国债来源的原始 payload 归档声明发生漂移")
    if contract.get("availability_rule") != (
        "NEXT_TRADING_DAY_OPEN_BECAUSE_EXACT_PUBLICATION_TIME_NOT_ARCHIVED"
    ):
        raise EvidenceContractError("中国国债 10 年收益率可得时钟发生漂移")
    source_dates = pd.DatetimeIndex(work["date"].sort_values())
    span = market_dates[
        (market_dates >= source_dates.min())
        & (market_dates <= min(source_dates.max(), market_dates.max()))
    ]
    missing_in_span = span.difference(source_dates)
    if len(missing_in_span):
        raise EvidenceContractError("中国国债 10 年收益率在自身覆盖区间缺少 ETF 交易日")
    leading = int((market_dates < source_dates.min()).sum())
    trailing = int((market_dates > source_dates.max()).sum())
    return _result(
        source_id=source_id,
        admitted=True,
        status="PASS_FROZEN_CHINABOND_10Y_WITH_CONSERVATIVE_CLOCK",
        date_level_no_view_required=bool(leading or trailing),
        availability_clock=str(contract["availability_rule"]),
        metrics={
            "row_count": int(len(work)),
            "first_observation_date": source_dates.min().date().isoformat(),
            "last_observation_date": source_dates.max().date().isoformat(),
            "market_sessions_in_source_span": int(len(span)),
            "missing_market_sessions_in_source_span": 0,
            "leading_no_view_market_sessions": leading,
            "trailing_no_view_market_sessions": trailing,
            "raw_payload_archive_present": False,
        },
        limitations=[
            "RAW_PAYLOAD_NOT_ARCHIVED",
            "EXACT_PUBLICATION_TIME_NOT_ARCHIVED_USE_NEXT_TRADING_DAY_OPEN",
        ],
    )


def evaluate_tsf_first_release(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    source_id = "tsf_stock_yoy_first_release"
    required = list(contract["required_columns"])
    require_columns(frame, required, source_id=source_id)
    work = frame.loc[:, required].copy()
    periods = pd.PeriodIndex(work["reference_period"].astype(str), freq="M")
    if periods.has_duplicates:
        raise EvidenceContractError("TSF 首发 vintage 存在重复参考月")
    expected = pd.period_range(
        str(contract["first_reference_period"]),
        str(contract["last_reference_period"]),
        freq="M",
    )
    if not periods.sort_values().equals(expected):
        raise EvidenceContractError("TSF 首发 vintage 月份不连续或边界漂移")
    published = pd.to_datetime(work["published_at"], errors="coerce", utc=True)
    available = pd.to_datetime(work["available_at"], errors="coerce", utc=True)
    if published.isna().any() or available.isna().any():
        raise EvidenceContractError("TSF 首发 vintage 含不可解析发布时间")
    if not published.equals(available):
        raise EvidenceContractError("TSF available_at 必须等于归档 published_at")
    values = pd.to_numeric(work["first_release_value"], errors="coerce")
    if values.isna().any() or (~np.isfinite(values)).any():
        raise EvidenceContractError("TSF first_release_value 含空值或非有限值")
    if work["latest_revised_value"].notna().any():
        raise EvidenceContractError("TSF 首发契约不得注入修订后历史值")
    if not work["revision_number"].eq(0).all():
        raise EvidenceContractError("TSF 首发契约 revision_number 必须为 0")
    if set(work["latest_revision_status"].astype(str)) != {
        "NOT_SEPARATELY_OBSERVED_NOT_USED"
    }:
        raise EvidenceContractError("TSF 修订状态发生漂移")
    hashes = work["source_hash"].astype(str).str.lower()
    if not hashes.map(lambda value: bool(_SHA256_RE.fullmatch(value))).all():
        raise EvidenceContractError("TSF source_hash 不是完整 SHA-256")
    missing_raw_paths: list[str] = []
    normalized_raw_paths: list[str] = []
    for value in work["raw_path"].astype(str):
        relative = normalize_project_relative_path(value)
        normalized_raw_paths.append(relative)
        if not (project_root / Path(relative)).is_file():
            missing_raw_paths.append(relative)
    if missing_raw_paths:
        raise EvidenceContractError(
            f"TSF 首发 vintage 缺少原始归档文件：{missing_raw_paths[:3]}"
        )
    if contract.get("revised_history_backfill_allowed") is not False:
        raise EvidenceContractError("TSF 修订历史回填禁令发生漂移")
    return _result(
        source_id=source_id,
        admitted=True,
        status="PASS_PBOC_TSF_FIRST_RELEASE_VINTAGE_WITH_LEADING_NO_VIEW",
        date_level_no_view_required=True,
        availability_clock=str(contract["availability_rule"]),
        metrics={
            "row_count": int(len(work)),
            "first_reference_period": str(expected[0]),
            "last_reference_period": str(expected[-1]),
            "contiguous_month_count": int(len(expected)),
            "raw_archive_path_count": int(len(set(normalized_raw_paths))),
            "raw_archive_missing_count": 0,
            "source_hash_format_valid_count": int(len(hashes)),
            "revision_value_used_count": 0,
        },
        limitations=[
            "THREE_MONTH_CHANGE_REQUIRES_LEADING_REFERENCE_MONTHS",
            "DATES_BEFORE_FIRST_AVAILABLE_RELEASE_ARE_NO_VIEW",
        ],
    )


def evaluate_pit_membership(
    frame: pd.DataFrame,
    admission_manifest: Mapping[str, Any],
    market_dates: pd.DatetimeIndex,
    *,
    contract: Mapping[str, Any],
    admission_contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = "pit_csi300_membership"
    required = list(contract["required_columns"])
    require_columns(frame, required, source_id=source_id)
    work = frame.loc[:, required].copy()
    work["membership_date"] = _dates(
        work["membership_date"], source_id=source_id, field="membership_date"
    )
    if work.duplicated(["membership_date", "symbol"]).any():
        raise EvidenceContractError("PIT 沪深 300 成分存在重复 date-symbol")
    if set(work["index_code"].astype(str)) != {str(contract["expected_index_code"])}:
        raise EvidenceContractError("PIT 沪深 300 成分指数代码不一致")
    counts = work.groupby("membership_date", sort=True)["symbol"].nunique()
    expected_count = int(contract["expected_constituent_count_each_session"])
    if not counts.eq(expected_count).all():
        raise EvidenceContractError("PIT 沪深 300 成分并非每个交易日恰好 300 只")
    membership_dates = pd.DatetimeIndex(counts.index)
    if not membership_dates.equals(market_dates):
        missing = market_dates.difference(membership_dates)
        extra = membership_dates.difference(market_dates)
        raise EvidenceContractError(
            f"PIT 成分日历与 510300 冻结日历不一致：missing={len(missing)}, extra={len(extra)}"
        )
    membership_admission = admission_manifest.get("membership_admission")
    if not isinstance(membership_admission, Mapping):
        raise EvidenceContractError("PIT 成分准入收据缺少 membership_admission")
    if membership_admission.get("status") != admission_contract.get(
        "expected_membership_status"
    ):
        raise EvidenceContractError("PIT 成分上游准入状态不一致")
    if int(membership_admission.get("row_count", -1)) != len(work):
        raise EvidenceContractError("PIT 成分上游准入行数不一致")
    if int(membership_admission.get("open_session_count", -1)) != len(counts):
        raise EvidenceContractError("PIT 成分上游准入交易日数不一致")
    if int(
        membership_admission.get("active_constituent_count_each_open_session", -1)
    ) != expected_count:
        raise EvidenceContractError("PIT 成分上游准入日成分数不一致")
    if contract.get("historical_weight_values_used") is not False:
        raise EvidenceContractError("本框架不得使用未准入的历史权重")
    return _result(
        source_id=source_id,
        admitted=True,
        status="PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_NO_HISTORICAL_WEIGHTS_USED",
        date_level_no_view_required=False,
        availability_clock="OFFICIAL_MEMBERSHIP_STATE_EFFECTIVE_BY_SESSION",
        metrics={
            "row_count": int(len(work)),
            "session_count": int(len(counts)),
            "first_session": membership_dates.min().date().isoformat(),
            "last_session": membership_dates.max().date().isoformat(),
            "constituent_count_each_session": expected_count,
            "duplicate_date_symbol_count": 0,
            "market_calendar_exact_match": True,
            "historical_weight_values_used": False,
        },
        limitations=["HISTORICAL_WEIGHTS_NOT_ADMITTED_AND_NOT_REQUIRED_BY_V1"],
    )


def evaluate_constituent_total_return_panel(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    observation_cutoff: str,
) -> tuple[dict[str, Any], tuple[pd.Timestamp, pd.Timestamp]]:
    source_id = "constituent_total_return_panel"
    required = list(contract["required_columns"])
    require_columns(frame, required, source_id=source_id)
    require_columns(
        membership,
        ["membership_date", "symbol"],
        source_id="pit_csi300_membership",
    )
    work = frame.loc[:, required].copy()
    work["date"] = _dates(work["date"], source_id=source_id, field="date")
    if work.duplicated(["date", "con_code"]).any():
        raise EvidenceContractError("成分股总回报面板存在重复 date-symbol")
    _finite_positive(
        work["total_return_close"],
        source_id=source_id,
        field="total_return_close",
    )
    if not work["price_source"].astype(str).str.startswith("tushare_proxy.").all():
        raise EvidenceContractError("成分股价格来源不是冻结的 tushare_proxy 系列")
    if not work["adjustment_source"].astype(str).str.startswith(
        "tushare_proxy."
    ).all():
        raise EvidenceContractError("成分股复权来源不是冻结的 tushare_proxy 系列")
    suspended = work["is_suspended"].fillna(False).astype(bool)
    if not work.loc[suspended, "price_source"].astype(str).str.endswith(
        "forward_fill_suspension"
    ).all():
        raise EvidenceContractError("停牌行没有使用冻结的停牌持平价格语义")

    membership_work = membership.loc[:, ["membership_date", "symbol"]].copy()
    membership_work["membership_date"] = _dates(
        membership_work["membership_date"],
        source_id="pit_csi300_membership",
        field="membership_date",
    )
    panel_start = work["date"].min()
    panel_end = min(work["date"].max(), pd.Timestamp(observation_cutoff))
    membership_window = membership_work.loc[
        membership_work["membership_date"].between(panel_start, panel_end)
    ].copy()
    if membership_window.empty:
        raise EvidenceContractError("成分股总回报面板与 PIT 成分没有重叠窗口")
    joined = membership_window.merge(
        work[["date", "con_code", "total_return_close"]],
        left_on=["membership_date", "symbol"],
        right_on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    joined["valid"] = (
        pd.to_numeric(joined["total_return_close"], errors="coerce").notna()
        & np.isfinite(
            pd.to_numeric(joined["total_return_close"], errors="coerce")
        )
        & (pd.to_numeric(joined["total_return_close"], errors="coerce") > 0)
    )
    valid_counts = joined.groupby("membership_date", sort=True)["valid"].sum()
    full_session_count = int(valid_counts.eq(300).sum())
    no_view_session_count = int(valid_counts.lt(300).sum())
    missing_member_days = int((300 - valid_counts).sum())
    all_membership_dates = pd.DatetimeIndex(
        membership_work["membership_date"].drop_duplicates().sort_values()
    )
    leading_no_view = int((all_membership_dates < panel_start).sum())
    trailing_no_view = int((all_membership_dates > panel_end).sum())
    return (
        _result(
            source_id=source_id,
            admitted=True,
            status="PASS_DATE_LEVEL_NO_VIEW_REQUIRED_CONSTITUENT_RETURN_PANEL",
            date_level_no_view_required=True,
            availability_clock="SAME_DAY_CLOSE_PUBLIC_OBSERVATION",
            metrics={
                "panel_row_count": int(len(work)),
                "panel_first_date": panel_start.date().isoformat(),
                "panel_last_date": work["date"].max().date().isoformat(),
                "evaluated_last_date": panel_end.date().isoformat(),
                "evaluated_session_count": int(len(valid_counts)),
                "full_300_member_session_count": full_session_count,
                "no_view_session_count": no_view_session_count,
                "missing_member_day_count": missing_member_days,
                "minimum_valid_member_count": int(valid_counts.min()),
                "leading_no_view_session_count": leading_no_view,
                "trailing_no_view_session_count": trailing_no_view,
                "suspension_carry_row_count": int(suspended.sum()),
                "duplicate_date_symbol_count": 0,
            },
            limitations=[
                "DATES_BEFORE_PANEL_START_ARE_NO_VIEW",
                "ANY_SESSION_WITH_FEWER_THAN_300_PIT_MEMBER_RETURNS_IS_NO_VIEW",
                "NO_CURRENT_CONSTITUENT_BACKFILL",
            ],
        ),
        (panel_start, panel_end),
    )


def evaluate_pit_industry_intervals(
    intervals: pd.DataFrame,
    raw_intervals: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    raw_contract: Mapping[str, Any],
    coverage_window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, Any]:
    source_id = "pit_industry_intervals"
    required = list(contract["required_columns"])
    raw_required = list(raw_contract["required_columns"])
    require_columns(intervals, required, source_id=source_id)
    require_columns(raw_intervals, raw_required, source_id="pit_industry_intervals_raw")
    work = intervals.loc[:, required].copy()
    raw = raw_intervals.loc[:, raw_required].copy()
    if set(work["classification_usage"].astype(str)) != {
        str(contract["required_classification_usage"])
    }:
        raise EvidenceContractError("行业分类不是 POINT_IN_TIME_INTERVAL")
    if set(work["source"].astype(str)) != {str(contract["source_identity"])}:
        raise EvidenceContractError("行业区间规范化来源身份不一致")
    if set(raw["source"].astype(str)) != {str(contract["source_identity"])}:
        raise EvidenceContractError("行业区间原始来源身份不一致")
    if contract.get("interval_end_semantics") != "OUT_DATE_EXCLUSIVE":
        raise EvidenceContractError("行业 out_date 必须冻结为右开区间")
    work["in_date"] = _dates(work["in_date"], source_id=source_id, field="in_date")
    work["out_date"] = pd.to_datetime(work["out_date"], errors="coerce").dt.normalize()
    invalid_end = work["out_date"].notna() & (work["out_date"] <= work["in_date"])
    if invalid_end.any():
        raise EvidenceContractError("行业区间存在 out_date 不晚于 in_date")
    if work.duplicated(
        ["con_code", "industry_l1_code", "in_date", "out_date"]
    ).any():
        raise EvidenceContractError("行业区间存在重复记录")
    raw["in_date"] = pd.to_datetime(
        raw["in_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    raw["out_date"] = pd.to_datetime(
        raw["out_date"].astype("string"), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    if raw["in_date"].isna().any():
        raise EvidenceContractError("行业原始区间存在不可解析 in_date")
    sentinel = pd.Timestamp("2262-01-01")
    normalized_keys = work.assign(
        out_key=work["out_date"].fillna(sentinel)
    )[["con_code", "industry_l1_code", "in_date", "out_key"]].drop_duplicates()
    raw_keys = raw.assign(out_key=raw["out_date"].fillna(sentinel))[
        ["ts_code", "l1_code", "in_date", "out_key"]
    ].drop_duplicates()
    cross = normalized_keys.merge(
        raw_keys,
        left_on=["con_code", "industry_l1_code", "in_date", "out_key"],
        right_on=["ts_code", "l1_code", "in_date", "out_key"],
        how="left",
        indicator=True,
    )
    raw_match_count = int(cross["_merge"].eq("both").sum())
    if raw_match_count != len(normalized_keys):
        raise EvidenceContractError("规范化行业区间不能逐行追溯到冻结原始快照")

    start, end = coverage_window
    membership_work = membership.loc[:, ["membership_date", "symbol"]].copy()
    membership_work["membership_date"] = _dates(
        membership_work["membership_date"],
        source_id="pit_csi300_membership",
        field="membership_date",
    )
    membership_window = membership_work.loc[
        membership_work["membership_date"].between(start, end)
    ].copy()
    expanded = membership_window.merge(
        work[["con_code", "industry_l1_code", "in_date", "out_date"]],
        left_on="symbol",
        right_on="con_code",
        how="left",
    )
    valid = (
        expanded["in_date"].le(expanded["membership_date"])
        & (
            expanded["out_date"].isna()
            | expanded["out_date"].gt(expanded["membership_date"])
        )
        & expanded["industry_l1_code"].notna()
    )
    valid_keys = expanded.loc[valid, ["membership_date", "symbol"]]
    overlap_counts = valid_keys.groupby(["membership_date", "symbol"]).size()
    overlapping_member_days = int(overlap_counts.gt(1).sum())
    if overlapping_member_days:
        raise EvidenceContractError("行业区间在同一成员交易日发生重叠")
    valid_keys = valid_keys.drop_duplicates()
    covered = membership_window.merge(
        valid_keys.assign(valid=True),
        on=["membership_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    covered["valid"] = covered["valid"].fillna(False).astype(bool)
    counts = covered.groupby("membership_date", sort=True)["valid"].sum()
    full_session_count = int(counts.eq(300).sum())
    no_view_session_count = int(counts.lt(300).sum())
    missing_member_days = int((300 - counts).sum())
    provenance_proven = bool(contract["independent_version_provenance_proven"])
    status = (
        "PASS_PIT_INDUSTRY_INTERVALS_WITH_DATE_LEVEL_NO_VIEW"
        if provenance_proven
        else "BLOCKED_NO_INDEPENDENT_VERSION_PROVEN_PIT_INDUSTRY_PROVENANCE"
    )
    limitations = [
        "ANY_SESSION_WITH_FEWER_THAN_300_PIT_MEMBER_INDUSTRIES_IS_NO_VIEW",
        "NO_CURRENT_INDUSTRY_BACKFILL",
    ]
    if not provenance_proven:
        limitations.insert(0, "PROXY_PAYLOAD_VERSION_PROVENANCE_NOT_INDEPENDENTLY_PROVEN")
    return _result(
        source_id=source_id,
        admitted=provenance_proven,
        status=status,
        date_level_no_view_required=True,
        availability_clock="POINT_IN_TIME_INTERVAL_EFFECTIVE_DATES",
        metrics={
            "normalized_interval_row_count": int(len(work)),
            "raw_interval_row_count": int(len(raw)),
            "normalized_rows_traced_to_raw_snapshot": raw_match_count,
            "coverage_first_session": start.date().isoformat(),
            "coverage_last_session": end.date().isoformat(),
            "evaluated_session_count": int(len(counts)),
            "full_300_member_session_count": full_session_count,
            "no_view_session_count": no_view_session_count,
            "missing_member_day_count": missing_member_days,
            "minimum_valid_member_count": int(counts.min()),
            "overlapping_member_day_count": overlapping_member_days,
            "out_date_is_exclusive": True,
            "independent_version_provenance_proven": provenance_proven,
        },
        limitations=limitations,
    )


def evaluate_missing_required_sources(
    missing_contracts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    dr007 = missing_contracts.get("dr007_daily")
    policy = missing_contracts.get("reverse_repo_policy_rate_7d")
    if not isinstance(dr007, Mapping) or not isinstance(policy, Mapping):
        raise EvidenceContractError("缺少 M2 两项必需来源契约")
    if dr007.get("selected_series") != "DR007":
        raise EvidenceContractError("M2 日度回购利率必须冻结为 DR007")
    forbidden = list(dr007.get("forbidden_substitutes", []))
    if forbidden != ["FDR007", "R007", "FR007", "EXCHANGE_REPO_R_007"]:
        raise EvidenceContractError("M2 DR007 禁止替代清单发生漂移")
    if policy.get("selected_series") != "PBOC_7D_REVERSE_REPO_OPERATION_RATE":
        raise EvidenceContractError("M2 政策利率身份发生漂移")
    if dr007.get("frozen_input_present") is not False:
        raise EvidenceContractError("DR007 来源状态与 missing_required_sources 冲突")
    if policy.get("frozen_input_present") is not False:
        raise EvidenceContractError("7 天逆回购政策利率状态与 missing_required_sources 冲突")
    return {
        "dr007_daily": _result(
            source_id="dr007_daily",
            admitted=False,
            status=str(dr007["failure_status"]),
            date_level_no_view_required=False,
            availability_clock=None,
            metrics={"frozen_input_present": False},
            limitations=[
                "FROZEN_HISTORICAL_DR007_SERIES_ABSENT",
                "FORBIDDEN_SUBSTITUTES=" + ",".join(forbidden),
            ],
        ),
        "reverse_repo_policy_rate_7d": _result(
            source_id="reverse_repo_policy_rate_7d",
            admitted=False,
            status=str(policy["failure_status"]),
            date_level_no_view_required=False,
            availability_clock=None,
            metrics={"frozen_input_present": False},
            limitations=[
                "FROZEN_PBOC_7D_POLICY_RATE_SCHEDULE_ABSENT",
                "NO_INTERPOLATION_OR_INFERRED_CHANGE_DATES",
            ],
        ),
    }


def adjudicate_channel_sources(
    source_results: Mapping[str, Mapping[str, Any]],
    gate_contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    channels: dict[str, dict[str, Any]] = {}
    for channel in ("M1", "M2", "M3", "F1", "F2", "F3", "T1", "T2", "T3"):
        dependency_key = f"{channel}_requires"
        dependencies = [str(value) for value in gate_contract[dependency_key]]
        unknown = [value for value in dependencies if value not in source_results]
        if unknown:
            raise EvidenceContractError(f"{channel} 引用了未知来源：{unknown}")
        blockers = [
            {
                "source_id": source_id,
                "status": str(source_results[source_id]["status"]),
            }
            for source_id in dependencies
            if source_results[source_id].get("admitted") is not True
        ]
        admitted = not blockers
        date_level_no_view = any(
            bool(source_results[source_id].get("date_level_no_view_required"))
            for source_id in dependencies
        )
        channels[channel] = {
            "channel_id": channel,
            "required_sources": dependencies,
            "admitted": admitted,
            "status": (
                "PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW"
                if admitted and date_level_no_view
                else (
                    "PASS_SOURCE_ADMISSION"
                    if admitted
                    else "BLOCKED_REQUIRED_SOURCE_ADMISSION_FAILED"
                )
            ),
            "blockers": blockers,
            "date_level_no_view_required": date_level_no_view,
            "feature_values_constructed": False,
        }
    blocked_channels = [
        channel for channel, result in channels.items() if not result["admitted"]
    ]
    all_admitted = not blocked_channels
    decision = {
        "status": str(
            gate_contract["passed_status"]
            if all_admitted
            else gate_contract["blocked_status"]
        ),
        "all_required_sources_admitted": all_admitted,
        "blocked_channels": blocked_channels,
        "feature_construction_allowed": bool(all_admitted),
        "feature_values_constructed": False,
        "g2_allowed": bool(all_admitted),
        "model_training_allowed": False,
        "portfolio_evaluation_allowed": False,
        "paper_or_shadow_signal_allowed": False,
        "order_generation_allowed": False,
        "live_trading_allowed": False,
        "position_impact": 0,
        "rescue_with_proxy_or_alternative_series_allowed": False,
    }
    return channels, decision


def _read_parquet(
    evidence: RegisteredFileEvidence,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    return pd.read_parquet(
        Path(evidence.resolved_path), columns=list(contract["required_columns"])
    )


def evaluate_source_admission(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    registered_inputs: Mapping[str, RegisteredFileEvidence] | None = None,
) -> dict[str, Any]:
    """对冻结来源做一次准入裁决；不产生任何特征值。"""

    if config.get("program", {}).get("program_id") != PROGRAM_ID:
        raise EvidenceContractError("source admission program_id 不一致")
    if registered_inputs is None:
        registered_inputs = validate_registered_inputs(
            config, project_root=project_root
        )
    contracts = config["source_contract"]
    start = str(config["program"]["observation_start"])
    cutoff = str(config["program"]["observation_cutoff"])

    etf_price_frame = _read_parquet(
        registered_inputs["etf_price"], contracts["etf_price"]
    )
    etf_price_result, market_dates = evaluate_etf_price(
        etf_price_frame,
        contract=contracts["etf_price"],
        observation_start=start,
        observation_cutoff=cutoff,
    )
    dividends = pd.read_csv(
        Path(registered_inputs["etf_dividends"].resolved_path),
        usecols=list(contracts["etf_dividends"]["required_columns"]),
    )
    dividend_result = evaluate_etf_dividends(
        dividends, contract=contracts["etf_dividends"]
    )
    pe_frame = _read_parquet(
        registered_inputs["csi300_official_pe"],
        contracts["csi300_official_pe"],
    )
    raw_snapshot = read_json_strict(
        Path(registered_inputs["csi300_official_raw_snapshot"].resolved_path)
    )
    if not isinstance(raw_snapshot, Mapping):
        raise EvidenceContractError("官方 CSI300 原始快照必须是对象")
    pe_result = evaluate_official_pe(
        pe_frame,
        raw_snapshot,
        market_dates,
        contract=contracts["csi300_official_pe"],
        raw_contract=contracts["csi300_official_raw_snapshot"],
    )
    cgb_result = evaluate_cgb10y(
        _read_parquet(
            registered_inputs["china_10y_government_bond_yield"],
            contracts["china_10y_government_bond_yield"],
        ),
        market_dates,
        contract=contracts["china_10y_government_bond_yield"],
    )
    tsf_result = evaluate_tsf_first_release(
        _read_parquet(
            registered_inputs["tsf_stock_yoy_first_release"],
            contracts["tsf_stock_yoy_first_release"],
        ),
        contract=contracts["tsf_stock_yoy_first_release"],
        project_root=project_root,
    )

    membership_frame = _read_parquet(
        registered_inputs["pit_csi300_membership"],
        contracts["pit_csi300_membership"],
    )
    membership_admission = read_json_strict(
        Path(registered_inputs["pit_csi300_membership_admission"].resolved_path)
    )
    if not isinstance(membership_admission, Mapping):
        raise EvidenceContractError("PIT 成分上游准入收据必须是对象")
    membership_result = evaluate_pit_membership(
        membership_frame,
        membership_admission,
        market_dates,
        contract=contracts["pit_csi300_membership"],
        admission_contract=contracts["pit_csi300_membership_admission"],
    )
    constituent_result, coverage_window = evaluate_constituent_total_return_panel(
        _read_parquet(
            registered_inputs["constituent_total_return_panel"],
            contracts["constituent_total_return_panel"],
        ),
        membership_frame,
        contract=contracts["constituent_total_return_panel"],
        observation_cutoff=cutoff,
    )
    industry_result = evaluate_pit_industry_intervals(
        _read_parquet(
            registered_inputs["pit_industry_intervals"],
            contracts["pit_industry_intervals"],
        ),
        _read_parquet(
            registered_inputs["pit_industry_intervals_raw"],
            contracts["pit_industry_intervals_raw"],
        ),
        membership_frame,
        contract=contracts["pit_industry_intervals"],
        raw_contract=contracts["pit_industry_intervals_raw"],
        coverage_window=coverage_window,
    )

    source_results: dict[str, dict[str, Any]] = {
        "etf_price": etf_price_result,
        "etf_dividends": dividend_result,
        "csi300_official_pe": pe_result,
        "china_10y_government_bond_yield": cgb_result,
        "tsf_stock_yoy_first_release": tsf_result,
        "pit_csi300_membership": membership_result,
        "constituent_total_return_panel": constituent_result,
        "pit_industry_intervals": industry_result,
    }
    source_results.update(
        evaluate_missing_required_sources(config["missing_required_sources"])
    )
    channels, decision = adjudicate_channel_sources(
        source_results, config["admission_gates"]
    )
    return {
        "program_id": PROGRAM_ID,
        "stage_id": str(config["program"]["stage_id"]),
        "source_results": source_results,
        "channel_results": channels,
        "decision": decision,
        "label_artifacts_read": False,
        "feature_values_constructed": False,
        "model_trained": False,
        "portfolio_metrics_read": False,
        "position_generated": False,
        "order_generated": False,
    }

