"""510300 总回报成分账本的会计恒等式与覆盖率计算。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少必需列：{missing}")


def build_etf_total_return_series(
    etf_price: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    """按除息日现金分红构造 510300 的日总回报指数。"""

    _require_columns(etf_price, ["date", "close"], "510300 日行情")
    _require_columns(
        dividends,
        ["ex_date", "cash_dividend_per_share"],
        "510300 分红档案",
    )
    price = etf_price[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = (
        price.dropna()
        .drop_duplicates("date", keep="last")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    if (price["close"] <= 0.0).any():
        raise ValueError("510300 日收盘价包含非正值")
    distribution = dividends[["ex_date", "cash_dividend_per_share"]].copy()
    distribution["date"] = pd.to_datetime(
        distribution["ex_date"], errors="coerce"
    ).dt.normalize()
    distribution["cash_dividend_per_share"] = pd.to_numeric(
        distribution["cash_dividend_per_share"], errors="coerce"
    )
    distribution = (
        distribution.dropna(subset=["date", "cash_dividend_per_share"])
        .groupby("date", as_index=True)["cash_dividend_per_share"]
        .sum()
    )
    price["cash_dividend_per_share"] = (
        price["date"].map(distribution).fillna(0.0).astype(float)
    )
    previous = price["close"].shift(1)
    price["daily_total_return_factor"] = (
        price["close"] + price["cash_dividend_per_share"]
    ) / previous
    price.loc[0, "daily_total_return_factor"] = 1.0
    price["etf_total_return_index"] = price[
        "daily_total_return_factor"
    ].cumprod()
    return price


def build_forward_schedule(
    origins: pd.Series,
    market_calendar: pd.Series,
    *,
    horizons_market_days: list[int],
    origin_start: str,
    observation_cutoff: str,
) -> pd.DataFrame:
    """使用 H00300 的实际交易日位置生成精确 60D/120D 目标日期。"""

    calendar = pd.DatetimeIndex(
        pd.to_datetime(market_calendar, errors="coerce")
        .dropna()
        .dt.normalize()
        .unique()
    ).sort_values()
    origin_values = pd.DatetimeIndex(
        pd.to_datetime(origins, errors="coerce").dropna().dt.normalize().unique()
    ).sort_values()
    origin_values = origin_values[
        (origin_values >= pd.Timestamp(origin_start))
        & (origin_values <= pd.Timestamp(observation_cutoff))
    ]
    position = {date: index for index, date in enumerate(calendar)}
    rows: list[dict[str, Any]] = []
    for origin in origin_values:
        if origin not in position:
            rows.append(
                {
                    "origin": origin,
                    "horizon_market_days": None,
                    "target_date": pd.NaT,
                    "schedule_status": "NO_VIEW_ORIGIN_NOT_IN_TOTAL_RETURN_CALENDAR",
                }
            )
            continue
        start_position = position[origin]
        for horizon in horizons_market_days:
            target_position = start_position + int(horizon)
            if target_position >= len(calendar):
                target_date = pd.NaT
                status = "CENSORED_TARGET_BEYOND_AVAILABLE_CALENDAR"
            else:
                target_date = calendar[target_position]
                status = (
                    "PASS_EXACT_MARKET_DAY_TARGET"
                    if target_date <= pd.Timestamp(observation_cutoff)
                    else "CENSORED_TARGET_AFTER_OBSERVATION_CUTOFF"
                )
            rows.append(
                {
                    "origin": origin,
                    "horizon_market_days": int(horizon),
                    "target_date": target_date,
                    "schedule_status": status,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["origin", "horizon_market_days"], kind="stable", na_position="last"
    ).reset_index(drop=True)


def _dated_series(
    frame: pd.DataFrame,
    value_column: str,
    *,
    date_column: str = "date",
) -> pd.Series:
    values = frame[[date_column, value_column]].copy()
    values[date_column] = pd.to_datetime(
        values[date_column], errors="coerce"
    ).dt.normalize()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values.dropna().drop_duplicates(date_column, keep="last")
    return values.set_index(date_column)[value_column].sort_index()


def _component_series(
    frame: pd.DataFrame,
    *,
    code_column: str,
    value_column: str,
) -> pd.Series:
    values = frame[["date", code_column, value_column]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce").dt.normalize()
    values[code_column] = values[code_column].astype(str)
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values.dropna().drop_duplicates(["date", code_column], keep="last")
    return values.set_index(["date", code_column])[value_column].sort_index()


def _lookup_components(
    series: pd.Series,
    date: pd.Timestamp,
    codes: pd.Series,
) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays(
        [np.repeat(pd.Timestamp(date), len(codes)), codes.astype(str).to_numpy()]
    )
    return series.reindex(keys).to_numpy(dtype=float)


def _safe_factor(end: float, start: float) -> float:
    if not np.isfinite(end) or not np.isfinite(start) or start <= 0.0 or end <= 0.0:
        return float("nan")
    return float(end / start)


def _identity_residual(product: float, target: float) -> float:
    if not np.isfinite(product) or not np.isfinite(target) or target <= 0.0:
        return float("nan")
    return float(product / target - 1.0)


def _effective_shares(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    shares = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce")
        candidate = candidate.where(candidate > 0.0)
        shares = shares.fillna(candidate)
    return shares


def _financial_status(coverage: float, pass_minimum: float, partial_minimum: float) -> str:
    if coverage >= pass_minimum:
        return "PASS_FIXED_FINANCIAL_COHORT_COVERAGE"
    if coverage >= partial_minimum:
        return "PARTIAL_FIXED_FINANCIAL_COHORT_COVERAGE"
    return "NO_VIEW_FIXED_FINANCIAL_COHORT_COVERAGE"


def build_component_ledger(
    *,
    schedule: pd.DataFrame,
    price_index: pd.DataFrame,
    total_return_index: pd.DataFrame,
    etf_total_return: pd.DataFrame,
    component_state: pd.DataFrame,
    component_raw_price: pd.DataFrame,
    component_total_return: pd.DataFrame,
    effective_share_columns: list[str],
    market_minimum_weight_coverage: float,
    financial_pass_minimum_weight_coverage: float,
    financial_partial_minimum_weight_coverage: float,
    identity_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造三种口径的成分账本、覆盖率面板和恒等式审计。"""

    _require_columns(schedule, ["origin", "horizon_market_days", "target_date", "schedule_status"], "目标日程")
    _require_columns(price_index, ["date", "close"], "价格指数")
    _require_columns(total_return_index, ["date", "close", "pe_ttm"], "全收益指数")
    _require_columns(etf_total_return, ["date", "etf_total_return_index"], "ETF 总回报序列")
    _require_columns(
        component_state,
        [
            "origin",
            "stock_code",
            "state_weight",
            "parent_net_profit_ttm",
            "income_available_at",
            *effective_share_columns,
        ],
        "成分月度状态",
    )
    _require_columns(component_raw_price, ["date", "stock_code", "raw_close"], "成分原始价格")
    _require_columns(
        component_total_return,
        ["date", "con_code", "total_return_close"],
        "成分总回报价格",
    )

    price_series = _dated_series(price_index, "close")
    total_series = _dated_series(total_return_index, "close")
    pe_series = _dated_series(total_return_index, "pe_ttm")
    etf_series = _dated_series(etf_total_return, "etf_total_return_index")
    raw_series = _component_series(
        component_raw_price, code_column="stock_code", value_column="raw_close"
    )
    component_tr_series = _component_series(
        component_total_return,
        code_column="con_code",
        value_column="total_return_close",
    )
    states = component_state.copy()
    states["origin"] = pd.to_datetime(states["origin"], errors="coerce").dt.normalize()
    states["stock_code"] = states["stock_code"].astype(str)
    states["state_weight"] = pd.to_numeric(states["state_weight"], errors="coerce")
    states["parent_net_profit_ttm"] = pd.to_numeric(
        states["parent_net_profit_ttm"], errors="coerce"
    )
    states["income_available_at"] = pd.to_datetime(
        states["income_available_at"], errors="coerce"
    ).dt.normalize()
    state_origins = pd.DatetimeIndex(states["origin"].dropna().unique()).sort_values()

    ledger_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for schedule_row in schedule.itertuples(index=False):
        if schedule_row.schedule_status != "PASS_EXACT_MARKET_DAY_TARGET":
            continue
        origin = pd.Timestamp(schedule_row.origin)
        target = pd.Timestamp(schedule_row.target_date)
        horizon = int(schedule_row.horizon_market_days)
        index_price_factor = _safe_factor(
            price_series.get(target, np.nan), price_series.get(origin, np.nan)
        )
        index_total_factor = _safe_factor(
            total_series.get(target, np.nan), total_series.get(origin, np.nan)
        )
        etf_factor = _safe_factor(
            etf_series.get(target, np.nan), etf_series.get(origin, np.nan)
        )
        pe_origin = float(pe_series.get(origin, np.nan))
        pe_target = float(pe_series.get(target, np.nan))
        tracking_factor = (
            etf_factor / index_total_factor
            if np.isfinite(etf_factor)
            and np.isfinite(index_total_factor)
            and index_total_factor > 0.0
            else float("nan")
        )
        index_dividend_factor = (
            index_total_factor / index_price_factor
            if np.isfinite(index_total_factor)
            and np.isfinite(index_price_factor)
            and index_price_factor > 0.0
            else float("nan")
        )
        index_earnings_factor = _safe_factor(
            price_series.get(target, np.nan) / pe_target
            if pe_target > 0.0
            else np.nan,
            price_series.get(origin, np.nan) / pe_origin
            if pe_origin > 0.0
            else np.nan,
        )
        index_multiple_factor = _safe_factor(pe_target, pe_origin)
        index_product = (
            index_dividend_factor
            * index_earnings_factor
            * index_multiple_factor
            * tracking_factor
        )
        index_residual = _identity_residual(index_product, etf_factor)
        index_identity_pass = bool(
            np.isfinite(index_residual) and abs(index_residual) <= identity_tolerance
        )
        ledger_rows.append(
            {
                "origin": origin,
                "horizon_market_days": horizon,
                "target_date": target,
                "scope": "ACTUAL_INDEX_CHAINED_SCOPE",
                "total_return_factor": index_total_factor,
                "price_factor": index_price_factor,
                "dividend_component_factor": index_dividend_factor,
                "dividend_component_status": "PASS_INDEX_IMPLIED_DIVIDEND_REINVESTMENT_FACTOR",
                "distribution_and_corporate_action_factor": index_dividend_factor,
                "earnings_growth_component_factor": index_earnings_factor,
                "earnings_growth_component_status": "PARTIAL_CHAINED_AGGREGATE_EARNINGS_PROXY_FROM_INDEX_PE_NOT_PIT_VINTAGE",
                "multiple_repricing_component_factor": index_multiple_factor,
                "multiple_repricing_component_status": "PARTIAL_CHAINED_INDEX_PE_MULTIPLE_NOT_PIT_VINTAGE",
                "membership_and_weight_component_factor": 1.0,
                "membership_and_weight_component_status": "INCLUDED_IN_CHAINED_INDEX_NOT_SEPARATELY_IDENTIFIED",
                "tracking_residual_factor": tracking_factor,
                "tracking_residual_status": "PASS_ETF_CASH_DISTRIBUTION_ADJUSTED_TRACKING_FACTOR",
                "unresolved_price_component_factor": np.nan,
                "identity_formula": "INDEX_DIVIDEND_X_CHAINED_EARNINGS_X_CHAINED_MULTIPLE_X_TRACKING",
                "identity_product_factor": index_product,
                "identity_target_etf_total_return_factor": etf_factor,
                "identity_multiplicative_residual": index_residual,
                "identity_status": "PASS_EXACT_MULTIPLICATIVE_IDENTITY" if index_identity_pass else "FAIL_MULTIPLICATIVE_IDENTITY",
                "market_weight_coverage": 1.0,
                "financial_weight_coverage": 1.0,
                "historical_official_weight_vintage_verified": False,
                "ledger_status": "PARTIAL_ACCOUNTING_CHAIN_EXACT_SOURCE_VINTAGE_LIMITED",
                "return_prediction_allowed": False,
                "portfolio_evaluation_allowed": False,
                "model_position_target": "UNSET",
            }
        )

        origin_state = states.loc[states["origin"].eq(origin)].copy()
        if origin_state.empty:
            coverage_rows.append(
                {
                    "origin": origin,
                    "horizon_market_days": horizon,
                    "target_date": target,
                    "coverage_status": "NO_VIEW_ORIGIN_COMPONENT_STATE_MISSING",
                }
            )
            continue
        origin_state = origin_state.drop_duplicates("stock_code", keep="last")
        weights = origin_state["state_weight"].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError(f"{origin.date()} 成分权重合计无效")
        weights = weights / weight_sum
        codes = origin_state["stock_code"]
        raw_origin = _lookup_components(raw_series, origin, codes)
        raw_target = _lookup_components(raw_series, target, codes)
        tr_origin = _lookup_components(component_tr_series, origin, codes)
        tr_target = _lookup_components(component_tr_series, target, codes)
        market_valid = (
            np.isfinite(raw_origin)
            & np.isfinite(raw_target)
            & np.isfinite(tr_origin)
            & np.isfinite(tr_target)
            & (raw_origin > 0.0)
            & (raw_target > 0.0)
            & (tr_origin > 0.0)
            & (tr_target > 0.0)
        )
        market_coverage = float(weights[market_valid].sum())
        market_count = int(market_valid.sum())
        market_status = (
            "PASS_FIXED_MARKET_COHORT_COVERAGE"
            if market_coverage >= market_minimum_weight_coverage
            else "NO_VIEW_FIXED_MARKET_COHORT_COVERAGE"
        )

        fixed_price_factor = float("nan")
        fixed_tr_factor = float("nan")
        fixed_distribution_factor = float("nan")
        fixed_membership_factor = float("nan")
        fixed_product = float("nan")
        fixed_residual = float("nan")
        if market_coverage >= market_minimum_weight_coverage:
            market_weights = weights[market_valid] / market_coverage
            fixed_price_factor = float(
                np.sum(market_weights * raw_target[market_valid] / raw_origin[market_valid])
            )
            fixed_tr_factor = float(
                np.sum(market_weights * tr_target[market_valid] / tr_origin[market_valid])
            )
            fixed_distribution_factor = fixed_tr_factor / fixed_price_factor
            fixed_membership_factor = index_total_factor / fixed_tr_factor
            fixed_product = (
                fixed_distribution_factor
                * fixed_price_factor
                * fixed_membership_factor
                * tracking_factor
            )
            fixed_residual = _identity_residual(fixed_product, etf_factor)
        fixed_identity_pass = bool(
            np.isfinite(fixed_residual) and abs(fixed_residual) <= identity_tolerance
        )
        ledger_rows.append(
            {
                "origin": origin,
                "horizon_market_days": horizon,
                "target_date": target,
                "scope": "ORIGIN_FIXED_COMPONENTS_AND_WEIGHTS_SCOPE",
                "total_return_factor": fixed_tr_factor,
                "price_factor": fixed_price_factor,
                "dividend_component_factor": np.nan,
                "dividend_component_status": "NO_VIEW_NO_COMPONENT_DISTRIBUTION_ARCHIVE",
                "distribution_and_corporate_action_factor": fixed_distribution_factor,
                "earnings_growth_component_factor": np.nan,
                "earnings_growth_component_status": "NO_VIEW_NOT_SEPARATED_IN_FULL_FIXED_MARKET_COHORT",
                "multiple_repricing_component_factor": np.nan,
                "multiple_repricing_component_status": "NO_VIEW_NOT_SEPARATED_IN_FULL_FIXED_MARKET_COHORT",
                "membership_and_weight_component_factor": fixed_membership_factor,
                "membership_and_weight_component_status": "PARTIAL_MEMBERSHIP_WEIGHT_PLUS_METHOD_AND_PROXY_WEIGHT_RESIDUAL",
                "tracking_residual_factor": tracking_factor,
                "tracking_residual_status": "PASS_ETF_CASH_DISTRIBUTION_ADJUSTED_TRACKING_FACTOR",
                "unresolved_price_component_factor": fixed_price_factor,
                "identity_formula": "DISTRIBUTION_AND_CORPORATE_ACTION_X_UNRESOLVED_PRICE_X_MEMBERSHIP_METHOD_X_TRACKING",
                "identity_product_factor": fixed_product,
                "identity_target_etf_total_return_factor": etf_factor,
                "identity_multiplicative_residual": fixed_residual,
                "identity_status": "PASS_EXACT_IDENTITY_WITH_UNRESOLVED_EARNINGS_MULTIPLE" if fixed_identity_pass else "NO_VIEW_OR_FAIL_FIXED_MARKET_IDENTITY",
                "market_weight_coverage": market_coverage,
                "financial_weight_coverage": np.nan,
                "historical_official_weight_vintage_verified": False,
                "ledger_status": "PARTIAL_PROXY_WEIGHT_AND_PURE_DIVIDEND_NO_VIEW" if fixed_identity_pass else market_status,
                "return_prediction_allowed": False,
                "portfolio_evaluation_allowed": False,
                "model_position_target": "UNSET",
            }
        )

        eligible_targets = state_origins[state_origins <= target]
        financial_target_origin = (
            eligible_targets[-1] if len(eligible_targets) else pd.NaT
        )
        target_state = states.loc[
            states["origin"].eq(financial_target_origin)
        ].drop_duplicates("stock_code", keep="last")
        target_state = target_state.set_index("stock_code")
        origin_financial = origin_state.copy()
        origin_financial["effective_shares_origin"] = _effective_shares(
            origin_financial, effective_share_columns
        )
        target_aligned = target_state.reindex(codes.astype(str).to_numpy())
        target_aligned = target_aligned.reset_index(drop=False).rename(
            columns={"index": "stock_code"}
        )
        target_aligned["effective_shares_target"] = _effective_shares(
            target_aligned, effective_share_columns
        )
        origin_profit = origin_financial["parent_net_profit_ttm"].to_numpy(dtype=float)
        target_profit = pd.to_numeric(
            target_aligned["parent_net_profit_ttm"], errors="coerce"
        ).to_numpy(dtype=float)
        shares_origin = origin_financial["effective_shares_origin"].to_numpy(dtype=float)
        shares_target = target_aligned["effective_shares_target"].to_numpy(dtype=float)
        origin_available = pd.to_datetime(
            origin_financial["income_available_at"], errors="coerce"
        ).to_numpy(dtype="datetime64[ns]")
        target_available = pd.to_datetime(
            target_aligned["income_available_at"], errors="coerce"
        ).to_numpy(dtype="datetime64[ns]")
        financial_valid = (
            market_valid
            & np.isfinite(origin_profit)
            & np.isfinite(target_profit)
            & np.isfinite(shares_origin)
            & np.isfinite(shares_target)
            & (shares_origin > 0.0)
            & (shares_target > 0.0)
            & ~pd.isna(origin_available)
            & ~pd.isna(target_available)
            & (origin_available <= np.datetime64(origin))
            & (target_available <= np.datetime64(target))
        )
        financial_coverage = float(weights[financial_valid].sum())
        financial_count = int(financial_valid.sum())
        financial_status = _financial_status(
            financial_coverage,
            financial_pass_minimum_weight_coverage,
            financial_partial_minimum_weight_coverage,
        )
        cohort_price_factor = float("nan")
        cohort_tr_factor = float("nan")
        cohort_distribution_factor = float("nan")
        earnings_factor = float("nan")
        multiple_factor = float("nan")
        cohort_membership_factor = float("nan")
        cohort_product = float("nan")
        cohort_residual = float("nan")
        aggregate_earnings_origin = float("nan")
        aggregate_earnings_target = float("nan")
        if financial_coverage >= financial_partial_minimum_weight_coverage:
            cohort_weights = weights[financial_valid] / financial_coverage
            cohort_price_factor = float(
                np.sum(
                    cohort_weights
                    * raw_target[financial_valid]
                    / raw_origin[financial_valid]
                )
            )
            cohort_tr_factor = float(
                np.sum(
                    cohort_weights
                    * tr_target[financial_valid]
                    / tr_origin[financial_valid]
                )
            )
            cohort_distribution_factor = cohort_tr_factor / cohort_price_factor
            quantities = cohort_weights / raw_origin[financial_valid]
            eps_origin = origin_profit[financial_valid] / shares_origin[financial_valid]
            eps_target = target_profit[financial_valid] / shares_target[financial_valid]
            aggregate_earnings_origin = float(np.sum(quantities * eps_origin))
            aggregate_earnings_target = float(np.sum(quantities * eps_target))
            if aggregate_earnings_origin > 0.0 and aggregate_earnings_target > 0.0:
                earnings_factor = aggregate_earnings_target / aggregate_earnings_origin
                multiple_factor = cohort_price_factor / earnings_factor
                cohort_membership_factor = index_total_factor / cohort_tr_factor
                cohort_product = (
                    cohort_distribution_factor
                    * earnings_factor
                    * multiple_factor
                    * cohort_membership_factor
                    * tracking_factor
                )
                cohort_residual = _identity_residual(cohort_product, etf_factor)
            else:
                financial_status = "NO_VIEW_NONPOSITIVE_AGGREGATE_EARNINGS_CLAIM"
        cohort_identity_pass = bool(
            np.isfinite(cohort_residual) and abs(cohort_residual) <= identity_tolerance
        )
        ledger_rows.append(
            {
                "origin": origin,
                "horizon_market_days": horizon,
                "target_date": target,
                "scope": "ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE",
                "total_return_factor": cohort_tr_factor,
                "price_factor": cohort_price_factor,
                "dividend_component_factor": np.nan,
                "dividend_component_status": "NO_VIEW_NO_COMPONENT_DISTRIBUTION_ARCHIVE",
                "distribution_and_corporate_action_factor": cohort_distribution_factor,
                "earnings_growth_component_factor": earnings_factor,
                "earnings_growth_component_status": financial_status,
                "multiple_repricing_component_factor": multiple_factor,
                "multiple_repricing_component_status": financial_status,
                "membership_and_weight_component_factor": cohort_membership_factor,
                "membership_and_weight_component_status": "PARTIAL_MEMBERSHIP_WEIGHT_PLUS_COHORT_METHOD_AND_PROXY_WEIGHT_RESIDUAL",
                "tracking_residual_factor": tracking_factor,
                "tracking_residual_status": "PASS_ETF_CASH_DISTRIBUTION_ADJUSTED_TRACKING_FACTOR",
                "unresolved_price_component_factor": np.nan,
                "identity_formula": "DISTRIBUTION_AND_CORPORATE_ACTION_X_EARNINGS_X_MULTIPLE_X_MEMBERSHIP_METHOD_X_TRACKING",
                "identity_product_factor": cohort_product,
                "identity_target_etf_total_return_factor": etf_factor,
                "identity_multiplicative_residual": cohort_residual,
                "identity_status": "PASS_EXACT_MULTIPLICATIVE_IDENTITY_ON_FINANCIAL_COHORT" if cohort_identity_pass else "NO_VIEW_OR_FAIL_FINANCIAL_COHORT_IDENTITY",
                "market_weight_coverage": market_coverage,
                "financial_weight_coverage": financial_coverage,
                "historical_official_weight_vintage_verified": False,
                "ledger_status": financial_status,
                "return_prediction_allowed": False,
                "portfolio_evaluation_allowed": False,
                "model_position_target": "UNSET",
            }
        )
        coverage_rows.append(
            {
                "origin": origin,
                "horizon_market_days": horizon,
                "target_date": target,
                "origin_component_count": int(len(origin_state)),
                "origin_weight_sum_before_normalization": weight_sum,
                "market_valid_component_count": market_count,
                "market_weight_coverage": market_coverage,
                "market_coverage_status": market_status,
                "financial_target_snapshot_origin": financial_target_origin,
                "financial_valid_component_count": financial_count,
                "financial_weight_coverage": financial_coverage,
                "financial_coverage_status": financial_status,
                "aggregate_earnings_claim_origin": aggregate_earnings_origin,
                "aggregate_earnings_claim_target": aggregate_earnings_target,
                "historical_official_weight_vintage_verified": False,
                "pure_component_dividend_archive_available": False,
            }
        )

    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["origin", "horizon_market_days", "scope"], kind="stable"
    ).reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["origin", "horizon_market_days"], kind="stable"
    ).reset_index(drop=True)
    identity = ledger[
        [
            "origin",
            "horizon_market_days",
            "target_date",
            "scope",
            "identity_formula",
            "identity_product_factor",
            "identity_target_etf_total_return_factor",
            "identity_multiplicative_residual",
            "identity_status",
        ]
    ].copy()
    identity["identity_tolerance"] = float(identity_tolerance)
    identity["identity_pass"] = (
        identity["identity_multiplicative_residual"].abs() <= identity_tolerance
    ).fillna(False)
    return ledger, coverage, identity
