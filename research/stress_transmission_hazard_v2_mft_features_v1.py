"""510300 压力传导危险率 V2 的点时 M/F/T 特征执行内核。

本模块只构造历史可得特征、来源可用性和 NO_VIEW 账本。它不读取实际标签、
未来收益、模型结果、组合结果、仓位或订单。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.stress_transmission_hazard_v2 import (
    NO_VIEW,
    VIEW_ALLOWED,
    StressTransmissionContractError,
    assert_predictive_names_are_industry_free,
    build_internal_raw_features,
    build_macro_features,
    causal_midrank,
    required_median,
)


SHANGHAI_TIMEZONE = "Asia/Shanghai"
MARKET_OPEN_OFFSET = pd.Timedelta(hours=9, minutes=30)
MARKET_CLOSE_OFFSET = pd.Timedelta(hours=15)

INTERNAL_PERCENTILE_INPUTS: Mapping[str, tuple[str, float]] = {
    "f1_breadth_risk_percentile": ("breadth20", -1.0),
    "f2_leadership_risk_percentile": ("leadership_gap20", 1.0),
    "f3_comovement_risk_percentile": ("comovement20", 1.0),
    "t1_breadth_drop_risk_percentile": ("breadth_drop5", 1.0),
    "t2_tail_diffusion_risk_percentile": ("tail_diffusion5", 1.0),
    "t3_comovement_accel_risk_percentile": ("comovement_accel5", 1.0),
}

MACRO_PERCENTILE_INPUTS: Mapping[str, tuple[str, float]] = {
    "m1_ey_rate_buffer_risk_percentile": ("ey_rate_buffer", -1.0),
    "m2_funding_spread_risk_percentile": ("funding_spread_5d_mean", 1.0),
    "t4_funding_shock_risk_percentile": ("funding_shock5", 1.0),
}

INTERNAL_GATED_COLUMNS = (
    "f1_breadth_risk_percentile",
    "f2_leadership_risk_percentile",
    "f3_comovement_risk_percentile",
    "F",
    "t1_breadth_drop_risk_percentile",
    "t2_tail_diffusion_risk_percentile",
    "t3_comovement_accel_risk_percentile",
)


@dataclass(frozen=True)
class MFTFeatureArtifacts:
    """一次 M/F/T 构建的全部可审计表。"""

    earnings_yield_releases: pd.DataFrame
    china_10y_releases: pd.DataFrame
    dr007_releases: pd.DataFrame
    reverse_repo_7d_releases: pd.DataFrame
    internal_features: pd.DataFrame
    macro_features: pd.DataFrame
    mft_feature_panel: pd.DataFrame
    member_history_ledger: pd.DataFrame
    macro_availability_ledger: pd.DataFrame
    coverage_ledger: pd.DataFrame
    validation: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise StressTransmissionContractError(f"{label}缺少必需列：{missing}")


def _normalize_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise StressTransmissionContractError("日期列含空值或不可解析值")
    return parsed.dt.normalize().astype("datetime64[ns]")


def _market_calendar(market_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(market_dates, errors="coerce")).normalize()
    if calendar.hasnans or calendar.empty:
        raise StressTransmissionContractError("点时市场日历为空或含非法日期")
    calendar = pd.DatetimeIndex(sorted(calendar.unique()))
    return calendar


def _market_clock_utc(
    dates: pd.Series | pd.DatetimeIndex,
    offset: pd.Timedelta,
) -> pd.Series:
    normalized = pd.Series(pd.DatetimeIndex(pd.to_datetime(dates)).normalize())
    local = normalized.dt.tz_localize(SHANGHAI_TIMEZONE) + offset
    return local.dt.tz_convert("UTC")


def conservative_next_market_open(
    observation_dates: pd.Series,
    market_dates: pd.DatetimeIndex,
) -> pd.Series:
    """将观察日映射到严格下一点时市场日 09:30，结果使用 UTC。"""

    observations = _normalize_dates(observation_dates)
    calendar = _market_calendar(market_dates)
    positions = np.searchsorted(
        calendar.to_numpy(dtype="datetime64[ns]"),
        observations.to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    next_dates = pd.Series(pd.NaT, index=observation_dates.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    if valid.any():
        next_dates.loc[valid] = calendar.to_numpy(dtype="datetime64[ns]")[positions[valid]]
    result = pd.Series(pd.NaT, index=observation_dates.index, dtype="datetime64[ns, UTC]")
    if valid.any():
        result.loc[valid] = _market_clock_utc(
            next_dates.loc[valid], MARKET_OPEN_OFFSET
        ).to_numpy()
    return result


def _collapse_one_value_per_date(
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    label: str,
) -> pd.DataFrame:
    work = frame.copy()
    work[date_column] = _normalize_dates(work[date_column])
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    if work[value_column].isna().any() or (~np.isfinite(work[value_column])).any():
        raise StressTransmissionContractError(f"{label}含空值或非有限数值")
    conflicts = work.groupby(date_column)[value_column].nunique(dropna=False).gt(1)
    if conflicts.any():
        examples = [str(x.date()) for x in conflicts.index[conflicts][:5]]
        raise StressTransmissionContractError(f"{label}同日数值冲突：{examples}")
    return (
        work.sort_values(date_column, kind="stable")
        .drop_duplicates(date_column, keep="last")
        .reset_index(drop=True)
    )


def prepare_earnings_yield_releases(
    pe: pd.DataFrame,
    *,
    market_dates: pd.DatetimeIndex,
    source_sha256: str,
) -> pd.DataFrame:
    """把官方 PE 日值转成带保守可得时钟的盈利收益率发布账本。"""

    _require_columns(
        pe,
        ["date", "index_code", "pe_official", "provider_original_field", "source"],
        "沪深300官方市盈率",
    )
    work = pe.loc[pe["index_code"].astype(str).str.strip().eq("000300")].copy()
    if work.empty:
        raise StressTransmissionContractError("沪深300官方市盈率没有 000300 记录")
    if not work["provider_original_field"].astype(str).str.strip().eq("peg").all():
        raise StressTransmissionContractError("官方市盈率原字段身份不是冻结的 peg")
    if not work["source"].astype(str).str.strip().eq("csindex.indexCsiDsPe").all():
        raise StressTransmissionContractError("官方市盈率来源身份漂移")
    work = _collapse_one_value_per_date(
        work,
        date_column="date",
        value_column="pe_official",
        label="沪深300官方市盈率",
    )
    if work["pe_official"].le(0.0).any():
        raise StressTransmissionContractError("官方市盈率必须为正数")
    release = pd.DataFrame(
        {
            "observation_date": work["date"],
            "available_at": conservative_next_market_open(work["date"], market_dates),
            "csi300_earnings_yield": 100.0 / work["pe_official"],
            "original_pe_official": work["pe_official"],
            "source_identity": "csindex.indexCsiDsPe",
            "source_file_sha256": source_sha256,
            "availability_rule": "NEXT_PIT_MARKET_SESSION_OPEN_AFTER_OBSERVATION_DATE",
        }
    )
    return release.dropna(subset=["available_at"]).reset_index(drop=True)


def prepare_china_10y_releases(
    bond: pd.DataFrame,
    *,
    market_dates: pd.DatetimeIndex,
    source_sha256: str,
) -> pd.DataFrame:
    """构造中国10年国债收益率的保守下一交易日发布账本。"""

    _require_columns(bond, ["date", "cgb_10y", "source"], "中国10年国债收益率")
    work = _collapse_one_value_per_date(
        bond,
        date_column="date",
        value_column="cgb_10y",
        label="中国10年国债收益率",
    )
    if work["cgb_10y"].le(0.0).any() or work["cgb_10y"].ge(20.0).any():
        raise StressTransmissionContractError("中国10年国债收益率超出冻结百分比边界")
    if not work["source"].astype(str).str.contains("chinabond", case=False).all():
        raise StressTransmissionContractError("中国10年国债收益率来源身份漂移")
    release = pd.DataFrame(
        {
            "observation_date": work["date"],
            "available_at": conservative_next_market_open(work["date"], market_dates),
            "china_10y_yield": work["cgb_10y"],
            "source_identity": work["source"].astype(str),
            "source_file_sha256": source_sha256,
            "availability_rule": "NEXT_PIT_MARKET_SESSION_OPEN_AFTER_OBSERVATION_DATE",
        }
    )
    return release.dropna(subset=["available_at"]).reset_index(drop=True)


def prepare_dr007_releases(
    dr007: pd.DataFrame,
    *,
    market_dates: pd.DatetimeIndex,
    source_sha256: str,
) -> pd.DataFrame:
    """构造精确 DR007 加权平均利率的下一交易日发布账本。"""

    required = [
        "date",
        "dr007",
        "ts_code",
        "provider_value_field",
        "availability_rule",
        "source_identity",
    ]
    _require_columns(dr007, required, "DR007")
    if not dr007["ts_code"].astype(str).str.strip().eq("DR007.IB").all():
        raise StressTransmissionContractError("DR007 证券代码身份漂移")
    if not dr007["provider_value_field"].astype(str).str.strip().eq("weight").all():
        raise StressTransmissionContractError("DR007 数值字段不是加权价 weight")
    if not dr007["availability_rule"].astype(str).str.strip().eq(
        "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE"
    ).all():
        raise StressTransmissionContractError("DR007 可得时钟漂移")
    work = _collapse_one_value_per_date(
        dr007,
        date_column="date",
        value_column="dr007",
        label="DR007",
    )
    if work["dr007"].le(0.0).any() or work["dr007"].ge(20.0).any():
        raise StressTransmissionContractError("DR007 超出冻结百分比边界")
    release = pd.DataFrame(
        {
            "observation_date": work["date"],
            "available_at": conservative_next_market_open(work["date"], market_dates),
            "dr007": work["dr007"],
            "source_identity": work["source_identity"].astype(str),
            "source_file_sha256": source_sha256,
            "availability_rule": "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE",
        }
    )
    return release.dropna(subset=["available_at"]).reset_index(drop=True)


def _published_at_as_utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise StressTransmissionContractError("央行公告 published_at 含非法时间")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(SHANGHAI_TIMEZONE)
    else:
        parsed = parsed.dt.tz_convert(SHANGHAI_TIMEZONE)
    return parsed.dt.tz_convert("UTC")


def prepare_reverse_repo_7d_releases(
    policy: pd.DataFrame,
    *,
    source_sha256: str,
) -> pd.DataFrame:
    """从央行逐公告原文账本构造7天逆回购政策利率发布账本。"""

    required = [
        "notice_date",
        "published_at",
        "has_seven_day_row",
        "seven_day_rate_percent",
        "parse_status",
        "source_url",
        "raw_sha256",
        "announcement_number",
    ]
    _require_columns(policy, required, "央行7天逆回购公告")
    has_row = policy["has_seven_day_row"].isin([True, 1])
    parsed = policy["parse_status"].astype(str).str.strip().eq(
        "PUBLISHED_7D_OPERATION_RATE"
    )
    work = policy.loc[has_row & parsed].copy()
    if work.empty:
        raise StressTransmissionContractError("央行公告账本没有有效7天逆回购利率")
    work["observation_date"] = _normalize_dates(work["notice_date"])
    work["available_at"] = _published_at_as_utc(work["published_at"])
    work["reverse_repo_7d_rate"] = pd.to_numeric(
        work["seven_day_rate_percent"], errors="coerce"
    )
    if (
        work["reverse_repo_7d_rate"].isna().any()
        or (~np.isfinite(work["reverse_repo_7d_rate"])).any()
        or work["reverse_repo_7d_rate"].le(0.0).any()
        or work["reverse_repo_7d_rate"].ge(20.0).any()
    ):
        raise StressTransmissionContractError("央行7天逆回购利率非法")
    local_available_date = work["available_at"].dt.tz_convert(
        SHANGHAI_TIMEZONE
    ).dt.tz_localize(None).dt.normalize()
    if local_available_date.lt(work["observation_date"]).any():
        raise StressTransmissionContractError("央行公告可得时间早于公告日期")
    key = ["observation_date", "available_at"]
    conflicts = work.groupby(key)["reverse_repo_7d_rate"].nunique(dropna=False).gt(1)
    if conflicts.any():
        raise StressTransmissionContractError("央行同一公告时点存在7天利率冲突")
    work = work.sort_values(key, kind="stable").drop_duplicates(key, keep="last")
    return work.assign(
        source_identity="PBOC_OPEN_MARKET_OPERATION_ANNOUNCEMENT_ARCHIVE",
        source_file_sha256=source_sha256,
        availability_rule="EXACT_ARCHIVED_PUBLISHED_AT_ASIA_SHANGHAI",
    )[
        [
            "observation_date",
            "available_at",
            "reverse_repo_7d_rate",
            "announcement_number",
            "source_url",
            "raw_sha256",
            "source_identity",
            "source_file_sha256",
            "availability_rule",
        ]
    ].reset_index(drop=True)


def normalize_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """把准入成员文件映射为冻结核心所需的 date/symbol schema。"""

    _require_columns(membership, ["membership_date", "index_code", "symbol"], "点时成员")
    frame = membership[["membership_date", "index_code", "symbol"]].rename(
        columns={"membership_date": "date"}
    )
    frame["date"] = _normalize_dates(frame["date"])
    frame["index_code"] = frame["index_code"].astype(str).str.strip()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if not frame["index_code"].eq("000300").all():
        raise StressTransmissionContractError("点时成员混入非 000300 指数")
    if frame.duplicated(["date", "symbol"]).any():
        raise StressTransmissionContractError("点时成员存在重复 date/symbol")
    counts = frame.groupby("date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise StressTransmissionContractError(
            f"点时成员每日不是300只：min={counts.min()}, max={counts.max()}"
        )
    return frame.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def build_member_history_ledger(
    *,
    membership: pd.DataFrame,
    classified_returns: pd.DataFrame,
    lookback_days: int = 20,
    minimum_comovement_member_days: int = 15,
) -> pd.DataFrame:
    """显式记录新成员的公开价格历史与真实成员日历史。"""

    if lookback_days != 20 or minimum_comovement_member_days != 15:
        raise StressTransmissionContractError("新成分历史窗口已冻结为20日/15日")
    members = normalize_membership(membership)
    _require_columns(
        classified_returns,
        ["date", "symbol", "return_is_usable", "daily_total_shareholder_return"],
        "四态收益",
    )
    returns = classified_returns[
        ["date", "symbol", "return_is_usable", "daily_total_shareholder_return"]
    ].copy()
    returns["date"] = _normalize_dates(returns["date"])
    returns["symbol"] = returns["symbol"].astype(str).str.strip().str.upper()
    value = pd.to_numeric(returns["daily_total_shareholder_return"], errors="coerce")
    returns["usable_return"] = value.where(
        returns["return_is_usable"].eq(True) & value.notna() & np.isfinite(value)
    )
    dates = _market_calendar(pd.DatetimeIndex(members["date"]))
    symbols = pd.Index(sorted(members["symbol"].unique()))
    member_mask = (
        members.assign(is_member=True)
        .pivot(index="date", columns="symbol", values="is_member")
        .reindex(index=dates, columns=symbols)
        .eq(True)
    )
    usable = (
        returns.pivot(index="date", columns="symbol", values="usable_return")
        .reindex(index=dates, columns=symbols)
        .notna()
    )
    public_price_count20 = usable.rolling(lookback_days, min_periods=1).sum()
    member_day_count20 = member_mask.astype(int).rolling(
        lookback_days, min_periods=1
    ).sum()
    prior_member = member_mask.shift(1, fill_value=False)
    entered = member_mask & ~prior_member
    current_with_full_public_price = member_mask & public_price_count20.ge(lookback_days)
    current_with_15_member_days = member_mask & member_day_count20.ge(
        minimum_comovement_member_days
    )
    ledger = pd.DataFrame(index=dates)
    ledger["point_in_time_member_count"] = member_mask.sum(axis=1).astype(int)
    ledger["newly_entered_member_count"] = entered.sum(axis=1).astype(int)
    ledger["members_with_full_20d_public_price_history"] = (
        current_with_full_public_price.sum(axis=1).astype(int)
    )
    ledger["members_without_full_20d_public_price_history"] = (
        ledger["point_in_time_member_count"]
        - ledger["members_with_full_20d_public_price_history"]
    )
    ledger["members_with_at_least_15_of_20_pit_member_days"] = (
        current_with_15_member_days.sum(axis=1).astype(int)
    )
    ledger["members_below_15_of_20_pit_member_days"] = (
        ledger["point_in_time_member_count"]
        - ledger["members_with_at_least_15_of_20_pit_member_days"]
    )
    ledger["pre_membership_public_price_history_allowed"] = True
    ledger["nonmember_days_count_as_comovement_history"] = False
    return ledger.reset_index(names="date")


def _numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(left_values, right_values, rtol=0.0, atol=1e-12, equal_nan=True))


def validate_causal_percentile_directions(
    internal_formula: pd.DataFrame,
    macro: pd.DataFrame,
) -> dict[str, bool]:
    """独立重算每个冻结方向的严格历史中秩。"""

    checks: dict[str, bool] = {}
    for output, (raw, sign) in INTERNAL_PERCENTILE_INPUTS.items():
        checks[output] = _numeric_equal(
            internal_formula[output], causal_midrank(sign * internal_formula[raw])
        )
    for output, (raw, sign) in MACRO_PERCENTILE_INPUTS.items():
        checks[output] = _numeric_equal(macro[output], causal_midrank(sign * macro[raw]))
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise StressTransmissionContractError(f"因果分位数或经济方向复核失败：{failed}")
    return checks


def validate_percentile_prefix_invariance(
    internal_formula: pd.DataFrame,
    macro: pd.DataFrame,
) -> dict[str, bool]:
    """用实际样本前缀确认追加未来行不会改写既有分位数。"""

    checks: dict[str, bool] = {}
    for label, frame, specifications in (
        ("internal", internal_formula, INTERNAL_PERCENTILE_INPUTS),
        ("macro", macro, MACRO_PERCENTILE_INPUTS),
    ):
        prefix_length = max(1, int(len(frame) * 0.75))
        for output, (raw, sign) in specifications.items():
            prefix = frame.iloc[:prefix_length]
            recomputed = causal_midrank(sign * prefix[raw])
            checks[f"{label}:{output}"] = _numeric_equal(
                frame[output].iloc[:prefix_length], recomputed
            )
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise StressTransmissionContractError(f"实际样本分位数前缀不稳定：{failed}")
    return checks


def build_macro_availability_ledger(macro: pd.DataFrame) -> pd.DataFrame:
    """核验每个被选宏观值均在 t 日15:00前可得，且参考期不晚于 t。"""

    _require_columns(macro, ["date"], "宏观特征")
    ledger = pd.DataFrame({"date": _normalize_dates(macro["date"])})
    close_utc = _market_clock_utc(ledger["date"], MARKET_CLOSE_OFFSET)
    channels = (
        "csi300_earnings_yield",
        "china_10y_yield",
        "dr007",
        "reverse_repo_7d_rate",
    )
    for channel in channels:
        value = pd.to_numeric(macro[channel], errors="coerce")
        observation = pd.to_datetime(
            macro[f"{channel}_source_observation_date"], errors="coerce"
        ).dt.normalize()
        available = pd.to_datetime(
            macro[f"{channel}_source_available_at"], errors="coerce", utc=True
        )
        ledger[channel] = value
        ledger[f"{channel}_source_observation_date"] = observation
        ledger[f"{channel}_source_available_at"] = available
        ledger[f"{channel}_available_by_close"] = (
            value.notna() & available.notna() & available.le(close_utc)
        )
        ledger[f"{channel}_observation_not_after_date"] = (
            value.notna() & observation.notna() & observation.le(ledger["date"])
        )
        invalid_clock = value.notna() & ~ledger[f"{channel}_available_by_close"]
        invalid_period = value.notna() & ~ledger[
            f"{channel}_observation_not_after_date"
        ]
        if invalid_clock.any() or invalid_period.any():
            raise StressTransmissionContractError(f"{channel} 选择结果使用了未来信息")
    ledger["available_macro_source_count"] = ledger[
        [f"{channel}_available_by_close" for channel in channels]
    ].sum(axis=1)
    ledger["all_four_macro_sources_available"] = ledger[
        "available_macro_source_count"
    ].eq(4)
    return ledger


def _no_view_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["four_state_daily_coverage_state"] != VIEW_ALLOWED:
        reasons.append("FOUR_STATE_DAILY_COVERAGE_FAILED")
    if row["internal_formula_coverage_state"] != VIEW_ALLOWED:
        reasons.append("INTERNAL_20D_60D_OR_COMOVEMENT_COVERAGE_FAILED")
    if pd.isna(row["F"]):
        reasons.append("F_INCOMPLETE_OR_CAUSAL_WARMUP")
    if pd.isna(row["t4_funding_shock_risk_percentile"]):
        reasons.append("T4_FUNDING_SHOCK_INCOMPLETE_OR_CAUSAL_WARMUP")
    if pd.isna(row["T"]):
        reasons.append("T_INCOMPLETE_OR_CAUSAL_WARMUP")
    if pd.isna(row["M"]):
        reasons.append("M_INCOMPLETE_OR_CAUSAL_WARMUP")
    return "VIEW_ALLOWED" if not reasons else ";".join(reasons)


def build_mft_feature_artifacts(
    *,
    membership: pd.DataFrame,
    classified_returns: pd.DataFrame,
    four_state_daily_coverage: pd.DataFrame,
    h00300_total_return_close: pd.DataFrame,
    pe: pd.DataFrame,
    bond: pd.DataFrame,
    dr007: pd.DataFrame,
    reverse_repo_7d: pd.DataFrame,
    source_sha256: Mapping[str, str],
) -> MFTFeatureArtifacts:
    """按冻结规则构造完整 M/F/T 与 NO_VIEW 证据链。"""

    members = normalize_membership(membership)
    market_dates = _market_calendar(pd.DatetimeIndex(members["date"]))
    if market_dates.min() != pd.Timestamp("2015-01-05"):
        raise StressTransmissionContractError("点时市场日历起点漂移")
    if market_dates.max() != pd.Timestamp("2026-08-14"):
        raise StressTransmissionContractError("点时市场日历截止日漂移")

    releases_ey = prepare_earnings_yield_releases(
        pe, market_dates=market_dates, source_sha256=source_sha256["pe"]
    )
    releases_bond = prepare_china_10y_releases(
        bond, market_dates=market_dates, source_sha256=source_sha256["bond"]
    )
    releases_dr007 = prepare_dr007_releases(
        dr007, market_dates=market_dates, source_sha256=source_sha256["dr007"]
    )
    releases_policy = prepare_reverse_repo_7d_releases(
        reverse_repo_7d, source_sha256=source_sha256["policy"]
    )

    _require_columns(h00300_total_return_close, ["date", "symbol", "close"], "H00300")
    benchmark = h00300_total_return_close.loc[
        h00300_total_return_close["symbol"].astype(str).str.strip().eq("H00300"),
        ["date", "close"],
    ].copy()
    benchmark["date"] = _normalize_dates(benchmark["date"])
    benchmark = benchmark.loc[benchmark["date"].isin(market_dates)]
    if benchmark.duplicated("date").any():
        raise StressTransmissionContractError("H00300 同日收盘值重复")
    if set(market_dates).difference(benchmark["date"]):
        raise StressTransmissionContractError("H00300 缺少冻结点时市场日")

    internal_formula = build_internal_raw_features(
        membership=members,
        classified_returns=classified_returns,
        h00300_total_return_close=benchmark,
        member_coverage_minimum=0.98,
        comovement_member_ratio_minimum=0.90,
        comovement_minimum_valid_observations=15,
        lookback_days=20,
        change_days=5,
        tail_volatility_days=60,
        tail_sigma_multiple=1.5,
        expected_members_per_day=300,
    )
    coverage = four_state_daily_coverage.copy()
    _require_columns(coverage, ["date", "aggregation_state"], "四态逐日覆盖")
    coverage["date"] = _normalize_dates(coverage["date"])
    if coverage.duplicated("date").any():
        raise StressTransmissionContractError("四态逐日覆盖存在重复日期")
    internal = internal_formula.merge(
        coverage[["date", "aggregation_state"]].rename(
            columns={"aggregation_state": "four_state_daily_coverage_state"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    internal = internal.rename(
        columns={"internal_coverage_state": "internal_formula_coverage_state"}
    )
    internal_gate = internal["internal_formula_coverage_state"].eq(
        VIEW_ALLOWED
    ) & internal["four_state_daily_coverage_state"].eq(VIEW_ALLOWED)
    internal["internal_feature_state"] = np.where(
        internal_gate, VIEW_ALLOWED, NO_VIEW
    )
    for column in INTERNAL_GATED_COLUMNS:
        internal[f"{column}_before_no_view_gate"] = internal[column]
        internal[column] = internal[column].where(internal_gate)

    macro = build_macro_features(
        market_dates=market_dates,
        earnings_yield_releases=releases_ey,
        china_10y_releases=releases_bond,
        dr007_releases=releases_dr007,
        reverse_repo_7d_releases=releases_policy,
    )
    direction_checks = validate_causal_percentile_directions(internal_formula, macro)
    prefix_checks = validate_percentile_prefix_invariance(internal_formula, macro)
    macro_availability = build_macro_availability_ledger(macro)
    member_history = build_member_history_ledger(
        membership=membership,
        classified_returns=classified_returns,
    )

    panel = (
        internal.merge(macro, on="date", how="left", validate="one_to_one")
        .merge(
            member_history.drop(columns=["point_in_time_member_count"]),
            on="date",
            how="left",
            validate="one_to_one",
        )
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    panel["T"] = required_median(
        panel,
        [
            "t1_breadth_drop_risk_percentile",
            "t2_tail_diffusion_risk_percentile",
            "t3_comovement_accel_risk_percentile",
            "t4_funding_shock_risk_percentile",
        ],
    )
    panel["F_state"] = np.where(
        panel["internal_feature_state"].eq(VIEW_ALLOWED) & panel["F"].notna(),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    panel["M_state"] = np.where(
        panel["macro_state"].eq(VIEW_ALLOWED) & panel["M"].notna(),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    panel["T_state"] = np.where(
        panel["internal_feature_state"].eq(VIEW_ALLOWED) & panel["T"].notna(),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    panel["B2_feature_state"] = np.where(
        panel["F_state"].eq(VIEW_ALLOWED) & panel["T_state"].eq(VIEW_ALLOWED),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    panel["B3_feature_state"] = np.where(
        panel["B2_feature_state"].eq(VIEW_ALLOWED)
        & panel["M_state"].eq(VIEW_ALLOWED),
        VIEW_ALLOWED,
        NO_VIEW,
    )
    panel["MODEL_STATE"] = panel["B3_feature_state"]
    panel["NO_VIEW_REASON"] = panel.apply(_no_view_reason, axis=1)

    predictive_columns = [
        "M",
        "F",
        "T",
        *INTERNAL_PERCENTILE_INPUTS.keys(),
        *MACRO_PERCENTILE_INPUTS.keys(),
    ]
    assert_predictive_names_are_industry_free(predictive_columns)
    if any(
        token in column.casefold()
        for column in panel.columns
        for token in ("industry", "sector", "shenwan")
    ):
        raise StressTransmissionContractError("M/F/T 核心面板出现行业语义字段")
    forbidden_actual_output_tokens = ("bad10", "auc", "sharpe", "drawdown", "nav", "position", "order")
    offending = [
        column
        for column in panel.columns
        if any(token in column.casefold() for token in forbidden_actual_output_tokens)
    ]
    if offending:
        raise StressTransmissionContractError(f"M/F/T 面板出现越权字段：{offending}")
    view = panel["MODEL_STATE"].eq(VIEW_ALLOWED)
    if panel.loc[view, ["M", "F", "T"]].isna().any().any():
        raise StressTransmissionContractError("VIEW_ALLOWED 行缺少 M/F/T")
    if panel.loc[~view, "NO_VIEW_REASON"].eq("VIEW_ALLOWED").any():
        raise StressTransmissionContractError("NO_VIEW 行缺少原因链")

    coverage_ledger = panel[
        [
            "date",
            "point_in_time_member_count",
            "return20_scoreable_member_count",
            "return20_coverage_ratio",
            "tail_scoreable_member_count",
            "tail_coverage_ratio",
            "comovement_scoreable_member_ratio",
            "four_state_daily_coverage_state",
            "internal_formula_coverage_state",
            "internal_feature_state",
            "F_state",
            "M_state",
            "T_state",
            "B2_feature_state",
            "B3_feature_state",
            "MODEL_STATE",
            "NO_VIEW_REASON",
        ]
    ].copy()
    validation: dict[str, Any] = {
        "causal_percentile_direction_checks": direction_checks,
        "actual_prefix_invariance_checks": prefix_checks,
        "all_selected_macro_values_available_by_t_close": bool(
            all(
                macro_availability.loc[
                    macro_availability[channel].notna(),
                    f"{channel}_available_by_close",
                ].all()
                for channel in (
                    "csi300_earnings_yield",
                    "china_10y_yield",
                    "dr007",
                    "reverse_repo_7d_rate",
                )
            )
        ),
        "all_selected_macro_observations_not_after_t": bool(
            all(
                macro_availability.loc[
                    macro_availability[channel].notna(),
                    f"{channel}_observation_not_after_date",
                ].all()
                for channel in (
                    "csi300_earnings_yield",
                    "china_10y_yield",
                    "dr007",
                    "reverse_repo_7d_rate",
                )
            )
        ),
        "industry_fields_in_core": False,
        "same_day_pe_bond_dr007_use": False,
        "missing_feature_zero_fill": False,
        "actual_label_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    if not validation["all_selected_macro_values_available_by_t_close"]:
        raise StressTransmissionContractError("宏观选择存在晚于 t 收盘的记录")
    if not validation["all_selected_macro_observations_not_after_t"]:
        raise StressTransmissionContractError("宏观选择存在晚于 t 的参考期")

    return MFTFeatureArtifacts(
        earnings_yield_releases=releases_ey,
        china_10y_releases=releases_bond,
        dr007_releases=releases_dr007,
        reverse_repo_7d_releases=releases_policy,
        internal_features=internal,
        macro_features=macro,
        mft_feature_panel=panel,
        member_history_ledger=member_history,
        macro_availability_ledger=macro_availability,
        coverage_ledger=coverage_ledger,
        validation=validation,
    )
