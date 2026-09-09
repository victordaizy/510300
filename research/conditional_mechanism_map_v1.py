"""510300 条件机制地图 V1 的无收益状态构造与描述性统计。

本模块不生成仓位、净值、夏普率或交易信号。条件状态先由父项目已经冻结的
CF、DR、RC 面板构造；只有状态冻结后，运行器才允许追加沪深300全收益结果。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


QUADRANTS = (
    "CF_IMPROVING_DR_EASING",
    "CF_IMPROVING_DR_TIGHTENING",
    "CF_DETERIORATING_DR_EASING",
    "CF_DETERIORATING_DR_TIGHTENING",
)

RC_DIRECTIONS = ("RC_SUPPORTIVE", "RC_CONSTRAINING", "RC_NEUTRAL")

ARCHETYPE_FLAG_COLUMNS = {
    "MOMENTUM_FUNDAMENTAL_SUPPORT": "is_momentum_fundamental_support",
    "REVERSAL_ECONOMIC_BASIS": "is_reversal_economic_basis",
    "VALUE_SAFETY_MARGIN": "is_value_safety_margin",
    "TRUE_CONTRACTION_DECLINE": "is_true_contraction_decline",
    "TEMPORARY_LIQUIDITY_SHOCK_CANDIDATE": (
        "is_temporary_liquidity_shock_candidate"
    ),
}

PASS_CF_STATE = "PASS_EXPANDING_POINT_IN_TIME_CF_STATE"
PASS_CF_DATA = "PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT"
PASS_PV_DATA = "PASS_EXPLANATORY_CROSS_SECTIONAL_PRESENT_VALUE_STATE"
PASS_RC_STATE = "PASS_EXPLANATORY_RISK_BEARING_CAPACITY_FACTOR"
PARTIAL_RC_STATE = "PARTIAL_EXPLANATORY_RISK_BEARING_CAPACITY_FACTOR"
COMPLETE_MONTH = "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少必需字段：{missing}")


def normalize_dates(values: pd.Series) -> pd.Series:
    """统一为无时区、纳秒精度的自然日时间戳。"""

    result = pd.to_datetime(values, errors="raise")
    if result.dt.tz is not None:
        result = result.dt.tz_convert(None)
    return result.astype("datetime64[ns]").dt.normalize()


def _validate_unique_dates(frame: pd.DataFrame, column: str, name: str) -> None:
    duplicates = frame.loc[frame[column].duplicated(keep=False), column]
    if not duplicates.empty:
        examples = duplicates.astype(str).head(5).tolist()
        raise ValueError(f"{name} 的 {column} 存在重复值：{examples}")


def _sign_label(value: Any, tolerance: float) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NO_VIEW"
    if float(numeric) > tolerance:
        return "POSITIVE"
    if float(numeric) < -tolerance:
        return "NEGATIVE"
    return "ZERO"


def _quadrant(cf_sign: str, dr_sign: str) -> str:
    mapping = {
        ("POSITIVE", "POSITIVE"): "CF_IMPROVING_DR_EASING",
        ("POSITIVE", "NEGATIVE"): "CF_IMPROVING_DR_TIGHTENING",
        ("NEGATIVE", "POSITIVE"): "CF_DETERIORATING_DR_EASING",
        ("NEGATIVE", "NEGATIVE"): "CF_DETERIORATING_DR_TIGHTENING",
    }
    if "NO_VIEW" in (cf_sign, dr_sign):
        return "NO_VIEW_REQUIRED_CF_OR_DR_STATE"
    return mapping.get((cf_sign, dr_sign), "NEUTRAL_NO_QUADRANT")


def _rc_direction(sign: str) -> str:
    return {
        "POSITIVE": "RC_SUPPORTIVE",
        "NEGATIVE": "RC_CONSTRAINING",
        "ZERO": "RC_NEUTRAL",
        "NO_VIEW": "RC_NO_VIEW",
    }[sign]


def _condition_quality(row: pd.Series) -> str:
    required_numeric = (
        "cf_level",
        "previous_cf_level",
        "equity_risk_premium_expanding_z",
        "previous_equity_risk_premium_expanding_z",
        "risk_bearing_capacity_factor",
        "previous_risk_bearing_capacity_factor",
    )
    if any(pd.isna(row[column]) for column in required_numeric):
        return "NO_VIEW_REQUIRED_PARENT_STATE_MISSING"
    if row["cf_state_status"] != PASS_CF_STATE:
        return "NO_VIEW_CURRENT_CF_STATE_NOT_PASS"
    if row["previous_cf_state_status"] != PASS_CF_STATE:
        return "NO_VIEW_PREVIOUS_CF_STATE_NOT_PASS"
    if row["present_value_data_status"] != PASS_PV_DATA:
        return "NO_VIEW_CURRENT_DR_STATE_NOT_PASS"
    if row["previous_present_value_data_status"] != PASS_PV_DATA:
        return "NO_VIEW_PREVIOUS_DR_STATE_NOT_PASS"
    admitted_rc_statuses = {PASS_RC_STATE, PARTIAL_RC_STATE}
    if row["rc_state_status"] not in admitted_rc_statuses:
        return "NO_VIEW_CURRENT_RC_STATE_NOT_ADMITTED"
    if row["previous_rc_state_status"] not in admitted_rc_statuses:
        return "NO_VIEW_PREVIOUS_RC_STATE_NOT_ADMITTED"

    partial_reasons: list[str] = []
    if row["cf_data_status"] != PASS_CF_DATA:
        partial_reasons.append("CURRENT_CF_DATA")
    if row["previous_cf_data_status"] != PASS_CF_DATA:
        partial_reasons.append("PREVIOUS_CF_DATA")
    if row["rc_state_status"] != PASS_RC_STATE:
        partial_reasons.append("CURRENT_RC_DATA")
    if row["previous_rc_state_status"] != PASS_RC_STATE:
        partial_reasons.append("PREVIOUS_RC_DATA")
    if row["origin_completion_status"] != COMPLETE_MONTH:
        partial_reasons.append("CURRENT_PARTIAL_MONTH")
    if row["previous_origin_completion_status"] != COMPLETE_MONTH:
        partial_reasons.append("PREVIOUS_PARTIAL_MONTH")
    if partial_reasons:
        return "PARTIAL_PARENT_STATE_LIMITATION_" + "_AND_".join(partial_reasons)
    return "PASS_FROZEN_PARENT_STATES"


def build_condition_state_panel(
    cashflow: pd.DataFrame,
    present_value: pd.DataFrame,
    risk_capacity: pd.DataFrame,
    *,
    asof_tolerance_calendar_days: int,
    numerical_zero_tolerance: float,
    rc_interaction_map: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """只使用父项目状态面板构造条件，函数不接收收益数据。"""

    cf_columns = [
        "origin",
        "cf_level",
        "cf_direction",
        "cf_acceleration",
        "cf_breadth",
        "cf_concentration",
        "cf_state_status",
        "cf_data_status",
        "origin_completion_status",
        "cash_flow_expectation_factor",
    ]
    pv_columns = [
        "origin",
        "cross_sectional_expected_return_factor",
        "equity_risk_premium_state",
        "equity_risk_premium_expanding_z",
        "valuation_concentration",
        "duration_compression",
        "present_value_data_status",
    ]
    rc_columns = [
        "date",
        "risk_bearing_capacity_factor",
        "rc_capacity_expanding_median",
        "member_median_return_60d",
        "member_positive_breadth_60d",
        "large_weight_minus_median_member_return_60d",
        "industry_negative_breadth_20d",
        "rc_state_status",
    ]
    _require_columns(cashflow, cf_columns, "CF 面板")
    _require_columns(present_value, pv_columns, "DR 面板")
    _require_columns(risk_capacity, rc_columns, "RC 面板")

    cf = cashflow[cf_columns].copy()
    pv = present_value[pv_columns].copy()
    rc = risk_capacity[rc_columns].copy()
    cf["origin"] = normalize_dates(cf["origin"])
    pv["origin"] = normalize_dates(pv["origin"])
    rc["date"] = normalize_dates(rc["date"])
    _validate_unique_dates(cf, "origin", "CF 面板")
    _validate_unique_dates(pv, "origin", "DR 面板")
    _validate_unique_dates(rc, "date", "RC 面板")

    state = cf.merge(pv, on="origin", how="inner", validate="one_to_one")
    if len(state) != len(cf) or len(state) != len(pv):
        raise ValueError("CF 与 DR 的月度 origin 集合不完全一致")
    state = state.sort_values("origin").reset_index(drop=True)
    state = pd.merge_asof(
        state,
        rc.sort_values("date").rename(columns={"date": "rc_asof_date"}),
        left_on="origin",
        right_on="rc_asof_date",
        direction="backward",
        tolerance=pd.Timedelta(days=int(asof_tolerance_calendar_days)),
    )
    state["rc_asof_age_calendar_days"] = (
        state["origin"] - state["rc_asof_date"]
    ).dt.days
    if (state["rc_asof_date"] > state["origin"]).fillna(False).any():
        raise ValueError("RC as-of 合并读取了 origin 之后的状态")

    previous_columns = [
        "cf_level",
        "cf_state_status",
        "cf_data_status",
        "origin_completion_status",
        "equity_risk_premium_expanding_z",
        "present_value_data_status",
        "risk_bearing_capacity_factor",
        "rc_state_status",
        "rc_asof_date",
    ]
    for column in previous_columns:
        state[f"previous_{column}"] = state[column].shift(1)

    state["cf_news"] = pd.to_numeric(state["cf_level"], errors="coerce").diff()
    state["discount_rate_easing_news"] = -pd.to_numeric(
        state["equity_risk_premium_expanding_z"], errors="coerce"
    ).diff()
    state["risk_capacity_news"] = pd.to_numeric(
        state["risk_bearing_capacity_factor"], errors="coerce"
    ).diff()

    tolerance = float(numerical_zero_tolerance)
    state["cf_news_sign"] = state["cf_news"].map(
        lambda value: _sign_label(value, tolerance)
    )
    state["discount_rate_easing_news_sign"] = state[
        "discount_rate_easing_news"
    ].map(lambda value: _sign_label(value, tolerance))
    state["risk_capacity_news_sign"] = state["risk_capacity_news"].map(
        lambda value: _sign_label(value, tolerance)
    )
    state["cf_dr_quadrant"] = [
        _quadrant(cf_sign, dr_sign)
        for cf_sign, dr_sign in zip(
            state["cf_news_sign"],
            state["discount_rate_easing_news_sign"],
            strict=True,
        )
    ]
    state["rc_direction"] = state["risk_capacity_news_sign"].map(_rc_direction)
    state["condition_quality_status"] = state.apply(_condition_quality, axis=1)

    mechanism_names = {
        "CF_IMPROVING_DR_EASING": "FUNDAMENTAL_AND_DISCOUNT_RATE_EXPANSION",
        "CF_IMPROVING_DR_TIGHTENING": "LATE_CYCLE_DISCOUNT_RATE_SQUEEZE",
        "CF_DETERIORATING_DR_EASING": "POLICY_OR_DISCOUNT_RATE_REPAIR",
        "CF_DETERIORATING_DR_TIGHTENING": "TRUE_CONTRACTION_STATE",
    }
    state["mechanism_state"] = state["cf_dr_quadrant"].map(mechanism_names).fillna(
        state["cf_dr_quadrant"]
    )

    def interaction_role(row: pd.Series) -> str:
        quadrant = row["cf_dr_quadrant"]
        rc_direction = row["rc_direction"]
        if quadrant not in QUADRANTS:
            return "NO_VIEW_OR_NEUTRAL_CF_DR_INTERACTION"
        if rc_direction == "RC_NO_VIEW":
            return "RC_NO_VIEW"
        try:
            return str(rc_interaction_map[quadrant][rc_direction])
        except KeyError as error:
            raise ValueError(
                f"RC 交互映射缺失：{quadrant} × {rc_direction}"
            ) from error

    state["rc_interaction_role"] = state.apply(interaction_role, axis=1)
    state["condition_cell"] = np.where(
        state["cf_dr_quadrant"].isin(QUADRANTS)
        & state["rc_direction"].isin(RC_DIRECTIONS),
        state["cf_dr_quadrant"] + "__" + state["rc_direction"],
        "NO_VIEW_OR_NEUTRAL_CONDITION_CELL",
    )
    state["condition_evaluation_eligible"] = (
        state["cf_dr_quadrant"].isin(QUADRANTS)
        & ~state["condition_quality_status"].str.startswith("NO_VIEW")
    )
    state["rc_interaction_evaluation_eligible"] = (
        state["condition_evaluation_eligible"]
        & state["rc_direction"].isin(RC_DIRECTIONS)
    )
    state["portfolio_evaluation_allowed"] = False
    state["model_position_target"] = "UNSET"
    state["future_return_values_read"] = False
    return state


def attach_total_return_context(
    condition_state: pd.DataFrame,
    total_return: pd.DataFrame,
    *,
    horizons_market_days: Sequence[int],
    trailing_horizons_market_days: Sequence[int],
    observation_cutoff: str | pd.Timestamp,
    maximum_origin_gap_calendar_days: int = 5,
) -> pd.DataFrame:
    """在条件面板已冻结后，追加历史价格语境和未来全收益。"""

    _require_columns(condition_state, ["origin"], "条件状态面板")
    _require_columns(total_return, ["date", "close"], "沪深300全收益序列")
    result = condition_state.copy()
    result["origin"] = normalize_dates(result["origin"])
    daily = total_return[["date", "close"]].copy()
    daily["date"] = normalize_dates(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.loc[
        daily["date"].le(pd.Timestamp(observation_cutoff).normalize())
    ].sort_values("date")
    if daily.empty:
        raise ValueError("沪深300全收益序列在冻结观察截止日前没有数据")
    _validate_unique_dates(daily, "date", "沪深300全收益序列")
    if daily["close"].isna().any() or daily["close"].le(0).any():
        raise ValueError("沪深300全收益 close 存在缺失或非正值")

    dates = pd.DatetimeIndex(daily["date"])
    closes = daily["close"].to_numpy(dtype=float)
    origin_positions: list[int] = []
    origin_dates: list[pd.Timestamp | pd.NaT] = []
    origin_closes: list[float] = []
    origin_statuses: list[str] = []
    for origin in result["origin"]:
        position = int(dates.searchsorted(origin, side="right") - 1)
        if position < 0:
            origin_positions.append(-1)
            origin_dates.append(pd.NaT)
            origin_closes.append(np.nan)
            origin_statuses.append("NO_VIEW_NO_TOTAL_RETURN_ON_OR_BEFORE_ORIGIN")
            continue
        asof_date = dates[position]
        age_days = int((origin - asof_date).days)
        if age_days > int(maximum_origin_gap_calendar_days):
            origin_positions.append(-1)
            origin_dates.append(pd.NaT)
            origin_closes.append(np.nan)
            origin_statuses.append("NO_VIEW_TOTAL_RETURN_ORIGIN_GAP_TOO_OLD")
            continue
        origin_positions.append(position)
        origin_dates.append(asof_date)
        origin_closes.append(float(closes[position]))
        origin_statuses.append("PASS_TOTAL_RETURN_AT_ORIGIN")
    result["total_return_asof_date"] = origin_dates
    result["total_return_close_at_origin"] = origin_closes
    result["total_return_origin_status"] = origin_statuses

    for horizon in trailing_horizons_market_days:
        values: list[float] = []
        statuses: list[str] = []
        start_dates: list[pd.Timestamp | pd.NaT] = []
        for position in origin_positions:
            start = position - int(horizon)
            if position < 0 or start < 0:
                values.append(np.nan)
                statuses.append("CENSORED_NO_TRAILING_HISTORY")
                start_dates.append(pd.NaT)
            else:
                values.append(float(closes[position] / closes[start] - 1.0))
                statuses.append("OBSERVED_PRICE_CONTEXT_ONLY")
                start_dates.append(dates[start])
        result[f"trailing_total_return_{horizon}d"] = values
        result[f"trailing_total_return_{horizon}d_start_date"] = start_dates
        result[f"trailing_total_return_{horizon}d_status"] = statuses

    for horizon in horizons_market_days:
        values = []
        statuses = []
        target_dates: list[pd.Timestamp | pd.NaT] = []
        for position in origin_positions:
            target = position + int(horizon)
            if position < 0 or target >= len(closes):
                values.append(np.nan)
                statuses.append("CENSORED_NO_VIEW")
                target_dates.append(pd.NaT)
            else:
                values.append(float(closes[target] / closes[position] - 1.0))
                statuses.append("OBSERVED_EXPLANATORY_ONLY")
                target_dates.append(dates[target])
        result[f"forward_total_return_{horizon}d"] = values
        result[f"forward_total_return_{horizon}d_target_date"] = target_dates
        result[f"forward_total_return_{horizon}d_status"] = statuses
    result["future_return_values_read"] = True
    return result


def assign_mechanism_archetypes(
    evaluated: pd.DataFrame,
    *,
    numerical_zero_tolerance: float,
) -> pd.DataFrame:
    """按冻结规则标记五类非互斥机制候选。"""

    required = [
        "cf_dr_quadrant",
        "rc_direction",
        "condition_quality_status",
        "equity_risk_premium_expanding_z",
        "trailing_total_return_20d",
        "trailing_total_return_60d",
    ]
    _require_columns(evaluated, required, "已追加收益的条件面板")
    result = evaluated.copy()
    tolerance = float(numerical_zero_tolerance)
    eligible = ~result["condition_quality_status"].str.startswith("NO_VIEW")
    expansion = result["cf_dr_quadrant"].eq("CF_IMPROVING_DR_EASING")
    squeeze = result["cf_dr_quadrant"].eq("CF_IMPROVING_DR_TIGHTENING")
    contraction = result["cf_dr_quadrant"].eq(
        "CF_DETERIORATING_DR_TIGHTENING"
    )
    supportive = result["rc_direction"].eq("RC_SUPPORTIVE")
    constraining = result["rc_direction"].eq("RC_CONSTRAINING")
    not_constraining = result["rc_direction"].isin(
        ["RC_SUPPORTIVE", "RC_NEUTRAL"]
    )
    trailing_20d = pd.to_numeric(
        result["trailing_total_return_20d"], errors="coerce"
    )
    trailing_60d = pd.to_numeric(
        result["trailing_total_return_60d"], errors="coerce"
    )
    erp_z = pd.to_numeric(
        result["equity_risk_premium_expanding_z"], errors="coerce"
    )

    result["is_momentum_fundamental_support"] = (
        eligible & expansion & supportive & trailing_60d.gt(tolerance)
    )
    result["is_reversal_economic_basis"] = (
        eligible & expansion & supportive & trailing_60d.lt(-tolerance)
    )
    result["is_value_safety_margin"] = (
        eligible & expansion & not_constraining & erp_z.ge(-tolerance)
    )
    result["is_true_contraction_decline"] = (
        eligible & contraction & constraining & trailing_20d.lt(-tolerance)
    )
    result["is_temporary_liquidity_shock_candidate"] = (
        eligible & squeeze & constraining & trailing_20d.lt(-tolerance)
    )
    return result


def _reliability_status(
    *,
    condition_origins: int,
    observed_origins: int,
    distinct_years: int,
    observation_coverage: float,
    parent_state_pass_share: float,
    gate: Mapping[str, Any],
) -> str:
    if condition_origins == 0 or observed_origins == 0:
        return "NO_VIEW_NO_OBSERVED_CONDITION_ORIGINS"
    pass_gate = (
        observed_origins >= int(gate["pass_minimum_observed_origins"])
        and distinct_years >= int(gate["pass_minimum_distinct_calendar_years"])
        and observation_coverage >= float(gate["minimum_forward_observation_coverage"])
        and parent_state_pass_share >= float(gate["minimum_parent_state_pass_share"])
    )
    if pass_gate:
        return "PASS_DESCRIPTIVE_COVERAGE"
    partial_gate = (
        observed_origins >= int(gate["partial_minimum_observed_origins"])
        and distinct_years >= int(gate["partial_minimum_distinct_calendar_years"])
    )
    if not partial_gate:
        return "NO_VIEW_INSUFFICIENT_CONDITION_OBSERVATIONS"
    reasons: list[str] = []
    if observation_coverage < float(gate["minimum_forward_observation_coverage"]):
        reasons.append("CENSORED_FORWARD_COVERAGE")
    if parent_state_pass_share < float(gate["minimum_parent_state_pass_share"]):
        reasons.append("PARENT_STATE_LIMITATIONS")
    if observed_origins < int(gate["pass_minimum_observed_origins"]):
        reasons.append("SMALL_SAMPLE")
    if distinct_years < int(gate["pass_minimum_distinct_calendar_years"]):
        reasons.append("SHORT_YEAR_SPAN")
    return "PARTIAL_DESCRIPTIVE_COVERAGE_" + "_AND_".join(reasons)


def _directional_result(mean: float, median: float, expected: str) -> str:
    if expected not in {"POSITIVE", "NEGATIVE"}:
        return "NOT_PRE_REGISTERED_FOR_THIS_VIEW"
    if np.isnan(mean) or np.isnan(median):
        return "NO_VIEW"
    if expected == "POSITIVE":
        if mean > 0 and median > 0:
            return "ALIGNED_WITH_EXPECTED_DIRECTION"
        if mean < 0 and median < 0:
            return "OPPOSITE_TO_EXPECTED_DIRECTION"
    else:
        if mean < 0 and median < 0:
            return "ALIGNED_WITH_EXPECTED_DIRECTION"
        if mean > 0 and median > 0:
            return "OPPOSITE_TO_EXPECTED_DIRECTION"
    return "MIXED_MEAN_MEDIAN_DIRECTION"


def summarize_condition_map(
    evaluated: pd.DataFrame,
    *,
    horizons_market_days: Sequence[int],
    sample_scopes: Mapping[str, Mapping[str, str]],
    reliability_gate: Mapping[str, Any],
    mechanism_archetypes: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """生成四象限、四象限×RC、五类机制候选的固定描述性地图。"""

    _require_columns(
        evaluated,
        [
            "origin",
            "cf_dr_quadrant",
            "rc_direction",
            "condition_quality_status",
            *ARCHETYPE_FLAG_COLUMNS.values(),
            *[f"forward_total_return_{horizon}d" for horizon in horizons_market_days],
        ],
        "已评估条件面板",
    )
    frame = evaluated.copy()
    frame["origin"] = normalize_dates(frame["origin"])
    rows: list[dict[str, Any]] = []

    for scope_id, scope in sample_scopes.items():
        start = pd.Timestamp(scope["start"]).normalize()
        end = pd.Timestamp(scope["end"]).normalize()
        scoped = frame.loc[frame["origin"].between(start, end)].copy()
        valid_quality = ~scoped["condition_quality_status"].str.startswith("NO_VIEW")

        view_groups: list[tuple[str, str, pd.DataFrame, str]] = []
        for quadrant in QUADRANTS:
            group = scoped.loc[valid_quality & scoped["cf_dr_quadrant"].eq(quadrant)]
            view_groups.append(("QUADRANT", quadrant, group, "UNSPECIFIED"))
        for quadrant in QUADRANTS:
            for rc_direction in RC_DIRECTIONS:
                condition_id = f"{quadrant}__{rc_direction}"
                group = scoped.loc[
                    valid_quality
                    & scoped["cf_dr_quadrant"].eq(quadrant)
                    & scoped["rc_direction"].eq(rc_direction)
                ]
                view_groups.append(
                    ("QUADRANT_X_RC", condition_id, group, "UNSPECIFIED")
                )
        for archetype_id, specification in mechanism_archetypes.items():
            flag = ARCHETYPE_FLAG_COLUMNS[archetype_id]
            group = scoped.loc[valid_quality & scoped[flag].fillna(False)]
            view_groups.append(
                (
                    "ARCHETYPE",
                    archetype_id,
                    group,
                    str(specification["expected_forward_direction"]),
                )
            )

        for view_type, condition_id, group, expected_direction in view_groups:
            condition_origins = int(len(group))
            for horizon in horizons_market_days:
                column = f"forward_total_return_{horizon}d"
                numeric = pd.to_numeric(group[column], errors="coerce")
                observed_mask = numeric.notna()
                observed = numeric.loc[observed_mask]
                observed_group = group.loc[observed_mask]
                observed_origins = int(len(observed))
                coverage = (
                    float(observed_origins / condition_origins)
                    if condition_origins
                    else 0.0
                )
                years = int(observed_group["origin"].dt.year.nunique())
                state_pass_share = (
                    float(
                        observed_group["condition_quality_status"]
                        .eq("PASS_FROZEN_PARENT_STATES")
                        .mean()
                    )
                    if observed_origins
                    else 0.0
                )
                values = observed.to_numpy(dtype=float)
                mean = float(np.mean(values)) if observed_origins else np.nan
                median = float(np.median(values)) if observed_origins else np.nan
                reliability = _reliability_status(
                    condition_origins=condition_origins,
                    observed_origins=observed_origins,
                    distinct_years=years,
                    observation_coverage=coverage,
                    parent_state_pass_share=state_pass_share,
                    gate=reliability_gate,
                )
                rows.append(
                    {
                        "sample_scope": scope_id,
                        "view_type": view_type,
                        "condition_id": condition_id,
                        "horizon_market_days": int(horizon),
                        "condition_origin_count": condition_origins,
                        "observed_origin_count": observed_origins,
                        "censored_origin_count": condition_origins - observed_origins,
                        "forward_observation_coverage": coverage,
                        "parent_state_pass_share": state_pass_share,
                        "distinct_calendar_years": years,
                        "first_observed_origin": (
                            observed_group["origin"].min()
                            if observed_origins
                            else pd.NaT
                        ),
                        "last_observed_origin": (
                            observed_group["origin"].max()
                            if observed_origins
                            else pd.NaT
                        ),
                        "mean_total_return": mean,
                        "median_total_return": median,
                        "positive_rate": (
                            float(np.mean(values > 0)) if observed_origins else np.nan
                        ),
                        "loss_rate": (
                            float(np.mean(values < 0)) if observed_origins else np.nan
                        ),
                        "q10_total_return": (
                            float(np.quantile(values, 0.10))
                            if observed_origins
                            else np.nan
                        ),
                        "q25_total_return": (
                            float(np.quantile(values, 0.25))
                            if observed_origins
                            else np.nan
                        ),
                        "q75_total_return": (
                            float(np.quantile(values, 0.75))
                            if observed_origins
                            else np.nan
                        ),
                        "q90_total_return": (
                            float(np.quantile(values, 0.90))
                            if observed_origins
                            else np.nan
                        ),
                        "minimum_total_return": (
                            float(np.min(values)) if observed_origins else np.nan
                        ),
                        "maximum_total_return": (
                            float(np.max(values)) if observed_origins else np.nan
                        ),
                        "expected_direction": expected_direction,
                        "directional_result": _directional_result(
                            mean, median, expected_direction
                        ),
                        "reliability_status": reliability,
                        "overlapping_forward_windows": True,
                        "portfolio_evaluation_allowed": False,
                        "model_position_target": "UNSET",
                    }
                )
    return pd.DataFrame(rows)


def build_archetype_conclusions(
    summary: pd.DataFrame,
    *,
    mechanism_archetypes: Mapping[str, Mapping[str, Any]],
    full_scope_id: str,
    user_scope_id: str,
) -> list[dict[str, Any]]:
    """按预注册主期限检查全样本与 2021+ 样本方向是否重复。"""

    conclusions: list[dict[str, Any]] = []
    for archetype_id, specification in mechanism_archetypes.items():
        horizon = int(specification["primary_evaluation_horizon_market_days"])
        evidence_rows = summary.loc[
            summary["view_type"].eq("ARCHETYPE")
            & summary["condition_id"].eq(archetype_id)
            & summary["horizon_market_days"].eq(horizon)
            & summary["sample_scope"].isin([full_scope_id, user_scope_id])
        ].copy()
        evidence_by_scope: dict[str, Any] = {}
        for _, row in evidence_rows.iterrows():
            evidence_by_scope[str(row["sample_scope"])] = {
                "observed_origin_count": int(row["observed_origin_count"]),
                "distinct_calendar_years": int(row["distinct_calendar_years"]),
                "mean_total_return": (
                    None
                    if pd.isna(row["mean_total_return"])
                    else float(row["mean_total_return"])
                ),
                "median_total_return": (
                    None
                    if pd.isna(row["median_total_return"])
                    else float(row["median_total_return"])
                ),
                "reliability_status": str(row["reliability_status"]),
                "directional_result": str(row["directional_result"]),
            }
        required_rows = [
            evidence_by_scope.get(full_scope_id),
            evidence_by_scope.get(user_scope_id),
        ]
        if any(item is None for item in required_rows):
            conclusion = "NO_VIEW_MISSING_FROZEN_SCOPE_RESULT"
        else:
            aligned = all(
                item["directional_result"] == "ALIGNED_WITH_EXPECTED_DIRECTION"
                for item in required_rows
                if item is not None
            )
            all_pass = all(
                item["reliability_status"] == "PASS_DESCRIPTIVE_COVERAGE"
                for item in required_rows
                if item is not None
            )
            any_no_view = any(
                item["reliability_status"].startswith("NO_VIEW")
                for item in required_rows
                if item is not None
            )
            if aligned and all_pass:
                conclusion = "REPEATED_DESCRIPTIVE_SUPPORT_FULL_AND_2021_PLUS"
            elif aligned and not any_no_view:
                conclusion = "PARTIAL_DIRECTIONAL_SUPPORT_NOT_FULLY_RELIABLE"
            elif any_no_view:
                conclusion = "NO_VIEW_INSUFFICIENT_RELIABLE_OBSERVATIONS"
            else:
                conclusion = "NOT_STABLE_ACROSS_FROZEN_SCOPES"
        conclusions.append(
            {
                "archetype_id": archetype_id,
                "frozen_definition": str(specification["frozen_definition"]),
                "primary_evaluation_horizon_market_days": horizon,
                "expected_forward_direction": str(
                    specification["expected_forward_direction"]
                ),
                "conclusion": conclusion,
                "scope_evidence": evidence_by_scope,
                "strategy_claim_allowed": False,
            }
        )
    return conclusions
