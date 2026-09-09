"""510300 结构性权益风险溢价引擎 V1 的纯研究计算。

本模块只提供点时状态构造、无收益标签的潜在因子提取，以及冻结后的
机械周期识别。文件采集、协议哈希和产物写入由运行脚本负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PcaResult:
    factor: pd.Series
    loadings: dict[str, float]
    explained_variance_ratio: float | None
    status: str


def finite_or_none(value: Any) -> float | None:
    """把有限数转换为 float；非有限数保留为明确缺失。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[float, float, int]:
    """返回归一化加权均值、原始权重覆盖率和有效样本数。"""

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = (
        numeric_values.notna()
        & np.isfinite(numeric_values)
        & numeric_weights.notna()
        & np.isfinite(numeric_weights)
        & numeric_weights.gt(0)
    )
    coverage = float(numeric_weights.loc[valid].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage, int(valid.sum())
    value = np.average(
        numeric_values.loc[valid].astype(float),
        weights=numeric_weights.loc[valid].astype(float),
    )
    return float(value), coverage, int(valid.sum())


def normalized_hhi(contributions: pd.Series) -> float:
    """计算非负贡献的标准化 HHI；一家公司独占时返回 1。"""

    values = pd.to_numeric(contributions, errors="coerce")
    values = values.loc[values.notna() & np.isfinite(values) & values.gt(0)]
    if values.empty:
        return 0.0
    if len(values) == 1:
        return 1.0
    shares = values / values.sum()
    raw = float(np.square(shares).sum())
    floor = 1.0 / len(shares)
    return float(np.clip((raw - floor) / (1.0 - floor), 0.0, 1.0))


def _ttm_from_map(
    value_map: dict[pd.Timestamp, float],
    period: pd.Timestamp,
) -> float:
    current = value_map.get(period, np.nan)
    if not np.isfinite(current):
        return np.nan
    if period.month == 12:
        return float(current)
    previous_annual = pd.Timestamp(period.year - 1, 12, 31)
    previous_same = period - pd.DateOffset(years=1)
    annual_value = value_map.get(previous_annual, np.nan)
    previous_same_value = value_map.get(previous_same, np.nan)
    if not np.isfinite(annual_value) or not np.isfinite(previous_same_value):
        return np.nan
    return float(current + annual_value - previous_same_value)


def prepare_statement_vintages(
    facts: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """规范点时财务版本，并验证版本键唯一。"""

    required = {
        "stock_code",
        "report_end",
        "available_at",
        *value_columns,
    }
    missing = sorted(required.difference(facts.columns))
    if missing:
        raise ValueError(f"财务版本表缺字段：{missing}")
    result = facts.copy()
    result["stock_code"] = result["stock_code"].astype(str)
    result["report_end"] = pd.to_datetime(
        result["report_end"], errors="coerce"
    ).dt.normalize()
    result["available_at"] = pd.to_datetime(
        result["available_at"], errors="coerce"
    ).dt.normalize()
    for column in value_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[
        result["report_end"].notna() & result["available_at"].notna()
    ].copy()
    key = ["stock_code", "report_end", "available_at"]
    if result.duplicated(key).any():
        raise ValueError("财务版本键 stock_code/report_end/available_at 不唯一")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def derive_ttm_pair_snapshot(
    facts: pd.DataFrame,
    stock_codes: Iterable[str],
    *,
    origin: pd.Timestamp,
    value_columns: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    """按 origin 选择最新已知版本，并同时构造当期与上年同期 TTM。

    所有 value_columns 必须在同一报告期拥有完整的当前与上年 TTM，避免
    收入、利润或现金流来自不同报告期却被误当作同一状态。
    """

    origin = pd.Timestamp(origin).normalize()
    codes = set(map(str, stock_codes))
    source = facts.loc[
        facts["stock_code"].isin(codes)
        & facts["report_end"].le(origin)
        & facts["available_at"].le(origin)
    ].copy()
    output_columns = [
        "stock_code",
        f"{prefix}_report_end",
        f"{prefix}_available_at",
    ]
    for column in value_columns:
        output_columns.extend(
            [f"{column}_ttm", f"{column}_ttm_prior_year"]
        )
    if source.empty:
        return pd.DataFrame(columns=output_columns)
    source = source.sort_values(
        ["stock_code", "report_end", "available_at"],
        kind="mergesort",
    ).drop_duplicates(["stock_code", "report_end"], keep="last")
    rows: list[dict[str, Any]] = []
    for stock_code, records in source.groupby("stock_code", sort=False):
        maps = {
            column: {
                pd.Timestamp(period): float(value)
                for period, value in zip(
                    records["report_end"], records[column], strict=True
                )
                if pd.notna(value) and np.isfinite(float(value))
            }
            for column in value_columns
        }
        available_map = {
            pd.Timestamp(period): pd.Timestamp(available)
            for period, available in zip(
                records["report_end"], records["available_at"], strict=True
            )
        }
        selected: pd.Timestamp | None = None
        selected_values: dict[str, float] = {}
        for period in sorted(available_map, reverse=True):
            prior_period = period - pd.DateOffset(years=1)
            candidate: dict[str, float] = {}
            complete = True
            for column in value_columns:
                current = _ttm_from_map(maps[column], period)
                prior = _ttm_from_map(maps[column], prior_period)
                if not np.isfinite(current) or not np.isfinite(prior):
                    complete = False
                    break
                candidate[f"{column}_ttm"] = current
                candidate[f"{column}_ttm_prior_year"] = prior
            if complete:
                selected = period
                selected_values = candidate
                break
        if selected is None:
            continue
        rows.append(
            {
                "stock_code": stock_code,
                f"{prefix}_report_end": selected,
                f"{prefix}_available_at": max(
                    available_map.get(selected, pd.Timestamp.min),
                    available_map.get(
                        selected - pd.DateOffset(years=1), pd.Timestamp.min
                    ),
                ),
                **selected_values,
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def derive_balance_snapshot(
    facts: pd.DataFrame,
    stock_codes: Iterable[str],
    *,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """取得 ROE 所需的当期、上年同期、上上年同期权益。"""

    origin = pd.Timestamp(origin).normalize()
    codes = set(map(str, stock_codes))
    source = facts.loc[
        facts["stock_code"].isin(codes)
        & facts["report_end"].le(origin)
        & facts["available_at"].le(origin)
    ].copy()
    columns = [
        "stock_code",
        "balance_report_end",
        "balance_available_at",
        "parent_equity_current",
        "parent_equity_prior_year",
        "parent_equity_two_years_prior",
        "total_assets_current",
        "share_capital_current",
    ]
    if source.empty:
        return pd.DataFrame(columns=columns)
    source = source.sort_values(
        ["stock_code", "report_end", "available_at"],
        kind="mergesort",
    ).drop_duplicates(["stock_code", "report_end"], keep="last")
    rows: list[dict[str, Any]] = []
    for stock_code, records in source.groupby("stock_code", sort=False):
        records = records.set_index("report_end", drop=False)
        for period in sorted(records.index.unique(), reverse=True):
            prior = period - pd.DateOffset(years=1)
            prior_two = period - pd.DateOffset(years=2)
            if prior not in records.index or prior_two not in records.index:
                continue
            current_row = records.loc[period]
            prior_row = records.loc[prior]
            prior_two_row = records.loc[prior_two]
            if isinstance(current_row, pd.DataFrame):
                current_row = current_row.iloc[-1]
            if isinstance(prior_row, pd.DataFrame):
                prior_row = prior_row.iloc[-1]
            if isinstance(prior_two_row, pd.DataFrame):
                prior_two_row = prior_two_row.iloc[-1]
            equity_values = [
                current_row["total_parent_equity"],
                prior_row["total_parent_equity"],
                prior_two_row["total_parent_equity"],
            ]
            if not all(pd.notna(value) and float(value) > 0 for value in equity_values):
                continue
            rows.append(
                {
                    "stock_code": stock_code,
                    "balance_report_end": period,
                    "balance_available_at": max(
                        pd.Timestamp(current_row["available_at"]),
                        pd.Timestamp(prior_row["available_at"]),
                        pd.Timestamp(prior_two_row["available_at"]),
                    ),
                    "parent_equity_current": float(equity_values[0]),
                    "parent_equity_prior_year": float(equity_values[1]),
                    "parent_equity_two_years_prior": float(equity_values[2]),
                    "total_assets_current": finite_or_none(
                        current_row["total_assets"]
                    ),
                    "share_capital_current": finite_or_none(
                        current_row["share_capital"]
                    ),
                }
            )
            break
    return pd.DataFrame(rows, columns=columns)


def clipped_growth(
    current: pd.Series,
    prior: pd.Series,
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    current_numeric = pd.to_numeric(current, errors="coerce")
    prior_numeric = pd.to_numeric(prior, errors="coerce")
    valid = (
        current_numeric.notna()
        & np.isfinite(current_numeric)
        & prior_numeric.notna()
        & np.isfinite(prior_numeric)
        & prior_numeric.gt(0)
    )
    output = pd.Series(np.nan, index=current.index, dtype=float)
    output.loc[valid] = (
        current_numeric.loc[valid] / prior_numeric.loc[valid] - 1.0
    ).clip(lower, upper)
    return output


def build_component_snapshot(
    market_members: pd.DataFrame,
    income_snapshot: pd.DataFrame,
    cash_snapshot: pd.DataFrame,
    balance_snapshot: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    growth_clip: tuple[float, float],
    roe_clip: tuple[float, float],
    roe_change_clip: tuple[float, float],
) -> pd.DataFrame:
    """构造单个 origin 的成分股 CF/DR 点时状态。"""

    required_market = {
        "stock_code",
        "industry_code",
        "industry_name",
        "total_market_cap",
        "market_asof_date",
    }
    missing = sorted(required_market.difference(market_members.columns))
    if missing:
        raise ValueError(f"月度市场截面缺字段：{missing}")
    panel = market_members.copy()
    panel["origin"] = pd.Timestamp(origin).normalize()
    panel = panel.merge(
        income_snapshot, on="stock_code", how="left", validate="one_to_one"
    ).merge(cash_snapshot, on="stock_code", how="left", validate="one_to_one")
    panel = panel.merge(
        balance_snapshot, on="stock_code", how="left", validate="one_to_one"
    )
    panel["total_market_cap"] = pd.to_numeric(
        panel["total_market_cap"], errors="coerce"
    )
    monthly_market_shares = pd.to_numeric(
        panel.get("total_shares", pd.Series(np.nan, index=panel.index)),
        errors="coerce",
    )
    balance_sheet_shares = pd.to_numeric(
        panel.get("share_capital_current", pd.Series(np.nan, index=panel.index)),
        errors="coerce",
    )
    independent_pit_shares = pd.to_numeric(
        panel.get("independent_total_shares", pd.Series(np.nan, index=panel.index)),
        errors="coerce",
    )
    reported_shares = monthly_market_shares.where(monthly_market_shares.gt(0))
    reported_shares = reported_shares.where(
        reported_shares.gt(0), balance_sheet_shares.where(balance_sheet_shares.gt(0))
    )
    reported_shares = reported_shares.where(
        reported_shares.gt(0), independent_pit_shares.where(independent_pit_shares.gt(0))
    )
    panel["state_share_source"] = np.select(
        [
            monthly_market_shares.gt(0),
            balance_sheet_shares.gt(0),
            independent_pit_shares.gt(0),
        ],
        [
            "MONTHLY_MARKET_PANEL_TOTAL_SHARES",
            "EASTMONEY_NOTICE_DATE_SHARE_CAPITAL",
            "INDEPENDENT_AVAILABLE_AT_TOTAL_SHARES_FALLBACK",
        ],
        default="NO_VIEW_TOTAL_SHARES",
    )
    exact_origin_close = pd.to_numeric(
        panel.get("fallback_raw_close", pd.Series(np.nan, index=panel.index)),
        errors="coerce",
    )
    monthly_panel_close = pd.to_numeric(
        panel.get("report_end_close", pd.Series(np.nan, index=panel.index)),
        errors="coerce",
    )
    reported_close = exact_origin_close.where(exact_origin_close.gt(0))
    reported_close = reported_close.where(
        reported_close.gt(0), monthly_panel_close.where(monthly_panel_close.gt(0))
    )
    panel["state_price_source"] = np.select(
        [exact_origin_close.gt(0), monthly_panel_close.gt(0)],
        ["EXACT_ORIGIN_RAW_CLOSE", "MONTHLY_MARKET_PANEL_REPORT_END_CLOSE"],
        default="NO_VIEW_CLOSE",
    )
    fallback_market_cap = (reported_shares * reported_close).where(
        reported_shares.gt(0) & reported_close.gt(0)
    )
    panel["state_market_cap_source"] = np.where(
        panel["total_market_cap"].gt(0),
        "POINT_IN_TIME_MARKET_PANEL_TOTAL_MARKET_CAP",
        np.where(
            fallback_market_cap.gt(0),
            "POINT_IN_TIME_CLOSE_TIMES_REPORTED_TOTAL_SHARES_FALLBACK",
            "NO_VIEW_MARKET_CAP",
        ),
    )
    panel["total_market_cap"] = panel["total_market_cap"].where(
        panel["total_market_cap"].gt(0), fallback_market_cap
    )
    valid_cap = panel["total_market_cap"].gt(0) & np.isfinite(
        panel["total_market_cap"]
    )
    cap_sum = float(panel.loc[valid_cap, "total_market_cap"].sum())
    panel["state_weight"] = np.where(
        valid_cap & (cap_sum > 0), panel["total_market_cap"] / cap_sum, np.nan
    )
    lower, upper = growth_clip
    panel["revenue_yoy"] = clipped_growth(
        panel["total_operating_revenue_ttm"],
        panel["total_operating_revenue_ttm_prior_year"],
        lower=lower,
        upper=upper,
    )
    panel["operating_profit_yoy"] = clipped_growth(
        panel["operating_profit_ttm"],
        panel["operating_profit_ttm_prior_year"],
        lower=lower,
        upper=upper,
    )
    panel["operating_cashflow_yoy"] = clipped_growth(
        panel["operating_cashflow_ttm"],
        panel["operating_cashflow_ttm_prior_year"],
        lower=lower,
        upper=upper,
    )
    current_average_equity = (
        panel["parent_equity_current"] + panel["parent_equity_prior_year"]
    ) / 2.0
    prior_average_equity = (
        panel["parent_equity_prior_year"]
        + panel["parent_equity_two_years_prior"]
    ) / 2.0
    roe_report_aligned = pd.to_datetime(
        panel.get("income_report_end", pd.Series(pd.NaT, index=panel.index)),
        errors="coerce",
    ).eq(
        pd.to_datetime(
            panel.get("balance_report_end", pd.Series(pd.NaT, index=panel.index)),
            errors="coerce",
        )
    )
    panel["roe_report_alignment_status"] = np.where(
        roe_report_aligned,
        "PASS_INCOME_BALANCE_REPORT_END_ALIGNED",
        "NO_VIEW_INCOME_BALANCE_REPORT_END_MISMATCH",
    )
    panel["roe_ttm"] = (
        panel["parent_net_profit_ttm"] / current_average_equity
    ).where(current_average_equity.gt(0) & roe_report_aligned).clip(*roe_clip)
    panel["roe_ttm_prior_year"] = (
        panel["parent_net_profit_ttm_prior_year"] / prior_average_equity
    ).where(prior_average_equity.gt(0) & roe_report_aligned).clip(*roe_clip)
    panel["roe_change"] = (
        panel["roe_ttm"] - panel["roe_ttm_prior_year"]
    ).clip(*roe_change_clip)
    cap = panel["total_market_cap"].where(panel["total_market_cap"].gt(0))
    panel["earnings_to_price"] = panel["parent_net_profit_ttm"] / cap
    panel["book_to_price"] = panel["parent_equity_current"] / cap
    panel["sales_to_price"] = panel["total_operating_revenue_ttm"] / cap
    panel["operating_cashflow_to_price"] = panel["operating_cashflow_ttm"] / cap
    profit_scale = panel["parent_net_profit_ttm"].abs().replace(0, np.nan)
    panel["cashflow_earnings_coverage"] = (
        panel["operating_cashflow_ttm"] / profit_scale
    ).clip(-5.0, 5.0)
    panel["market_snapshot_age_days"] = (
        panel["origin"] - pd.to_datetime(panel["market_asof_date"])
    ).dt.days
    return panel


def aggregate_cashflow_snapshot(
    component: pd.DataFrame,
    *,
    minimum_coverages: dict[str, float],
    minimum_market_cap_members: int,
) -> dict[str, Any]:
    """把成分股状态聚合成六个冻结 CF 核心量。"""

    weights = component["state_weight"]
    revenue, revenue_coverage, revenue_count = weighted_average(
        component["revenue_yoy"], weights
    )
    profit, profit_coverage, profit_count = weighted_average(
        component["operating_profit_yoy"], weights
    )
    cashflow, cashflow_coverage, cashflow_count = weighted_average(
        component["operating_cashflow_yoy"], weights
    )
    roe_change, roe_coverage, roe_count = weighted_average(
        component["roe_change"], weights
    )
    profit_valid = component["operating_profit_yoy"].notna() & weights.gt(0)
    valid_profit_weight = float(weights.loc[profit_valid].sum())
    improvement_weight = float(
        weights.loc[profit_valid & component["operating_profit_yoy"].gt(0)].sum()
    )
    breadth = (
        improvement_weight / valid_profit_weight
        if valid_profit_weight > 0
        else np.nan
    )
    negative_contribution = (
        weights * (-component["operating_profit_yoy"]).clip(lower=0)
    )
    concentration = normalized_hhi(negative_contribution)
    market_cap_member_count = int(weights.notna().sum())
    market_cap_gate = market_cap_member_count >= int(minimum_market_cap_members)
    gates = {
        "revenue": revenue_coverage >= minimum_coverages["revenue"],
        "operating_profit": profit_coverage
        >= minimum_coverages["operating_profit"],
        "operating_cashflow": cashflow_coverage
        >= minimum_coverages["operating_cashflow"],
        "roe_change": roe_coverage >= minimum_coverages["roe_change"],
    }
    if not market_cap_gate:
        status = "NO_VIEW_MARKET_CAP_MEMBER_COVERAGE_BELOW_GATE"
    elif all(gates.values()):
        status = "PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT"
    elif sum(gates.values()) >= 3:
        status = "PARTIAL_PIT_CF_STATE_COVERAGE_BELOW_ONE_OR_MORE_GATES"
    else:
        status = "NO_VIEW_CF_STATE_COVERAGE_INSUFFICIENT"
    return {
        "origin": pd.Timestamp(component["origin"].iloc[0]),
        "membership_count": int(component["stock_code"].nunique()),
        "market_cap_member_count": market_cap_member_count,
        "market_cap_member_gate": int(minimum_market_cap_members),
        "market_cap_gate_pass": market_cap_gate,
        "weighted_revenue_yoy": revenue,
        "weighted_operating_profit_yoy": profit,
        "weighted_operating_cashflow_yoy": cashflow,
        "weighted_roe_change": roe_change,
        "improvement_weight_breadth": breadth,
        "deterioration_concentration": concentration,
        "revenue_weight_coverage": revenue_coverage,
        "operating_profit_weight_coverage": profit_coverage,
        "operating_cashflow_weight_coverage": cashflow_coverage,
        "roe_change_weight_coverage": roe_coverage,
        "revenue_member_count": revenue_count,
        "operating_profit_member_count": profit_count,
        "operating_cashflow_member_count": cashflow_count,
        "roe_change_member_count": roe_count,
        "cf_data_status": status,
        "weight_method": "POINT_IN_TIME_TOTAL_MARKET_CAP_PROXY",
    }


def expanding_zscore(
    values: pd.Series,
    *,
    minimum_periods: int,
    clip: float,
) -> pd.Series:
    """只使用截至当期的历史计算 expanding z-score。"""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    mean = numeric.expanding(min_periods=minimum_periods).mean()
    std = numeric.expanding(min_periods=minimum_periods).std(ddof=1)
    result = (numeric - mean) / std.replace(0, np.nan)
    return result.clip(-clip, clip)


def add_cashflow_state_columns(
    cashflow_panel: pd.DataFrame,
    *,
    minimum_periods: int,
    zscore_clip: float,
) -> pd.DataFrame:
    result = cashflow_panel.sort_values("origin").reset_index(drop=True).copy()
    directions = {
        "weighted_revenue_yoy": 1.0,
        "weighted_operating_profit_yoy": 1.0,
        "weighted_operating_cashflow_yoy": 1.0,
        "weighted_roe_change": 1.0,
        "improvement_weight_breadth": 1.0,
        "deterioration_concentration": -1.0,
    }
    z_columns: list[str] = []
    for column, direction in directions.items():
        target = f"{column}_expanding_z"
        result[target] = expanding_zscore(
            result[column] * direction,
            minimum_periods=minimum_periods,
            clip=zscore_clip,
        )
        z_columns.append(target)
    valid_count = result[z_columns].notna().sum(axis=1)
    result["cf_level"] = result[z_columns].median(axis=1, skipna=True).where(
        valid_count.ge(4)
    )
    result["cf_direction"] = result["cf_level"].diff(3)
    result["cf_acceleration"] = result["cf_direction"].diff(3)
    result["cf_breadth"] = result["improvement_weight_breadth"]
    result["cf_concentration"] = result["deterioration_concentration"]
    result["cf_state_status"] = np.where(
        result["cf_level"].notna(),
        "PASS_EXPANDING_POINT_IN_TIME_CF_STATE",
        "NO_VIEW_CF_STATE_WARMUP_OR_COVERAGE",
    )
    return result


def _industry_percentile_rank(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum_industry_members: int,
) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame[column].notna() & np.isfinite(frame[column])
    counts = frame.loc[valid].groupby("industry_code", observed=True)[
        column
    ].transform("size")
    eligible = counts.ge(minimum_industry_members)
    eligible_index = counts.index[eligible]
    output.loc[eligible_index] = frame.loc[eligible_index].groupby(
        "industry_code", observed=True
    )[column].rank(method="average", pct=True)
    return output


def add_present_value_cross_section(
    component_panel: pd.DataFrame,
    *,
    minimum_industry_members: int,
    winsor_limits: tuple[float, float],
) -> pd.DataFrame:
    """生成行业中性现值排名、质量和久期代理。"""

    result = component_panel.copy()
    ratios = [
        "earnings_to_price",
        "book_to_price",
        "sales_to_price",
        "operating_cashflow_to_price",
    ]
    lower, upper = winsor_limits
    for column in ratios + ["cashflow_earnings_coverage"]:
        clipped = pd.Series(np.nan, index=result.index, dtype=float)
        for _, group in result.groupby("origin", sort=False):
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            if valid.sum() < 20:
                continue
            low = float(values.loc[valid].quantile(lower))
            high = float(values.loc[valid].quantile(upper))
            clipped.loc[group.index] = values.clip(low, high)
        target = f"{column}_winsor"
        result[target] = clipped
        rank_target = f"{column}_industry_rank"
        ranks = pd.Series(np.nan, index=result.index, dtype=float)
        for _, group in result.groupby("origin", sort=False):
            ranks.loc[group.index] = _industry_percentile_rank(
                group.assign(**{column: clipped.loc[group.index]}),
                column,
                minimum_industry_members=minimum_industry_members,
            )
        result[rank_target] = ranks
    rank_columns = [
        f"{column}_industry_rank" for column in ratios
    ]
    result["present_value_rank_mean"] = result[rank_columns].mean(
        axis=1, skipna=True
    ).where(result[rank_columns].notna().sum(axis=1).ge(3))
    quality_rank = result["cashflow_earnings_coverage_industry_rank"]
    result["quality_rank"] = quality_rank
    result["duration_proxy"] = (
        1.0
        - result[
            [
                "book_to_price_industry_rank",
                "sales_to_price_industry_rank",
                "operating_cashflow_to_price_industry_rank",
            ]
        ].mean(axis=1, skipna=True)
    )
    return result


def _tercile(rank: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [rank.le(1.0 / 3.0), rank.le(2.0 / 3.0), rank.gt(2.0 / 3.0)],
            ["LOW", "MID", "HIGH"],
            default=None,
        ),
        index=rank.index,
        dtype="object",
    )


def build_fixed_present_value_portfolios(
    component_panel: pd.DataFrame,
) -> pd.DataFrame:
    """构造冻结的 16 个现值组合；组合不读取任何收益标签。"""

    rows: list[dict[str, Any]] = []
    ratio_specs = {
        "EP": ("earnings_to_price_winsor", "earnings_to_price_industry_rank"),
        "BP": ("book_to_price_winsor", "book_to_price_industry_rank"),
        "SP": ("sales_to_price_winsor", "sales_to_price_industry_rank"),
        "CFP": (
            "operating_cashflow_to_price_winsor",
            "operating_cashflow_to_price_industry_rank",
        ),
    }
    for origin, group in component_panel.groupby("origin", sort=True):
        weight = pd.to_numeric(group["state_weight"], errors="coerce")
        for prefix, (value_column, rank_column) in ratio_specs.items():
            buckets = _tercile(group[rank_column])
            for bucket in ["LOW", "MID", "HIGH"]:
                mask = buckets.eq(bucket)
                value, coverage, count = weighted_average(
                    group.loc[mask, value_column], weight.loc[mask]
                )
                rows.append(
                    {
                        "origin": pd.Timestamp(origin),
                        "portfolio_id": f"{prefix}_{bucket}",
                        "portfolio_state_value": value,
                        "member_count": count,
                        "raw_weight_coverage": coverage,
                        "state_value_definition": value_column,
                    }
                )
        ep_rank = group["earnings_to_price_industry_rank"]
        quality_rank = group["quality_rank"]
        duration = group["duration_proxy"]
        special = {
            "HIGH_EP_HIGH_QUALITY": ep_rank.gt(2 / 3) & quality_rank.gt(2 / 3),
            "HIGH_EP_LOW_QUALITY": ep_rank.gt(2 / 3) & quality_rank.le(1 / 3),
            "LONG_DURATION": duration.gt(2 / 3),
            "SHORT_DURATION": duration.le(1 / 3),
        }
        for portfolio_id, mask in special.items():
            value, coverage, count = weighted_average(
                group.loc[mask, "earnings_to_price_winsor"], weight.loc[mask]
            )
            rows.append(
                {
                    "origin": pd.Timestamp(origin),
                    "portfolio_id": portfolio_id,
                    "portfolio_state_value": value,
                    "member_count": count,
                    "raw_weight_coverage": coverage,
                    "state_value_definition": "earnings_to_price_winsor",
                }
            )
    return pd.DataFrame(rows)


def first_principal_component(
    frame: pd.DataFrame,
    *,
    sign_anchor: pd.Series | None,
    minimum_rows: int,
) -> PcaResult:
    """在不使用收益标签的条件下提取第一主成分。"""

    numeric = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    eligible_columns = [
        column
        for column in numeric.columns
        if int(numeric[column].notna().sum()) >= minimum_rows
        and float(numeric[column].std(ddof=1)) > 0
    ]
    empty = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(eligible_columns) < 2 or len(frame) < minimum_rows:
        return PcaResult(
            factor=empty,
            loadings={},
            explained_variance_ratio=None,
            status="NO_VIEW_PCA_INSUFFICIENT_ROWS_OR_COLUMNS",
        )
    selected = numeric[eligible_columns]
    standardized = (selected - selected.mean()) / selected.std(ddof=1)
    row_coverage = standardized.notna().mean(axis=1)
    matrix = standardized.fillna(0.0).to_numpy(dtype=float)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    loading = vh[0]
    factor_values = matrix @ loading
    factor = pd.Series(factor_values, index=frame.index, dtype=float)
    factor = ((factor - factor.mean()) / factor.std(ddof=1)).where(
        row_coverage.ge(0.75)
    )
    if sign_anchor is not None:
        aligned = pd.concat(
            [factor.rename("factor"), sign_anchor.rename("anchor")], axis=1
        ).dropna()
        if len(aligned) >= 3 and aligned.corr().iloc[0, 1] < 0:
            factor = -factor
            loading = -loading
    total_variance = float(np.square(singular).sum())
    explained = (
        float(singular[0] ** 2 / total_variance) if total_variance > 0 else None
    )
    return PcaResult(
        factor=factor,
        loadings={
            column: float(value)
            for column, value in zip(eligible_columns, loading, strict=True)
        },
        explained_variance_ratio=explained,
        status="PASS_EXPLANATORY_FULL_SAMPLE_PCA_WITHOUT_RETURN_LABELS",
    )


def aggregate_present_value_panel(
    component_panel: pd.DataFrame,
    portfolios: pd.DataFrame,
    *,
    minimum_pca_months: int,
) -> tuple[pd.DataFrame, PcaResult]:
    """聚合横截面现值状态并提取无收益标签的共同因子。"""

    rows: list[dict[str, Any]] = []
    for origin, group in component_panel.groupby("origin", sort=True):
        weights = group["state_weight"]
        aggregates: dict[str, Any] = {"origin": pd.Timestamp(origin)}
        for column in [
            "earnings_to_price",
            "book_to_price",
            "sales_to_price",
            "operating_cashflow_to_price",
        ]:
            value, coverage, count = weighted_average(group[column], weights)
            aggregates[f"cap_weighted_{column}"] = value
            aggregates[f"{column}_weight_coverage"] = coverage
            aggregates[f"{column}_member_count"] = count
        positive_ep_contribution = weights * group["earnings_to_price"].clip(
            lower=0
        )
        aggregates["valuation_concentration"] = normalized_hhi(
            positive_ep_contribution
        )
        long_mask = group["duration_proxy"].gt(2 / 3)
        short_mask = group["duration_proxy"].le(1 / 3)
        long_ep, _, _ = weighted_average(
            group.loc[long_mask, "earnings_to_price"], weights.loc[long_mask]
        )
        short_ep, _, _ = weighted_average(
            group.loc[short_mask, "earnings_to_price"], weights.loc[short_mask]
        )
        aggregates["duration_compression"] = (
            short_ep - long_ep
            if np.isfinite(short_ep) and np.isfinite(long_ep)
            else np.nan
        )
        rows.append(aggregates)
    result = pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)
    pca_ids = [
        f"{prefix}_{bucket}"
        for prefix in ["EP", "BP", "SP", "CFP"]
        for bucket in ["LOW", "MID", "HIGH"]
    ]
    pivot = portfolios.loc[portfolios["portfolio_id"].isin(pca_ids)].pivot(
        index="origin", columns="portfolio_id", values="portfolio_state_value"
    )
    pivot = pivot.reindex(result["origin"])
    anchor = result[
        [
            "cap_weighted_earnings_to_price",
            "cap_weighted_book_to_price",
            "cap_weighted_sales_to_price",
            "cap_weighted_operating_cashflow_to_price",
        ]
    ].apply(lambda column: (column - column.mean()) / column.std(ddof=1)).mean(
        axis=1, skipna=True
    )
    anchor.index = pivot.index
    pca = first_principal_component(
        pivot,
        sign_anchor=anchor,
        minimum_rows=minimum_pca_months,
    )
    result["cross_sectional_expected_return_factor"] = pca.factor.to_numpy()
    result["present_value_factor_status"] = pca.status
    result["dividend_yield"] = np.nan
    result["dividend_yield_status"] = (
        "NO_VIEW_NO_POINT_IN_TIME_COMPONENT_DIVIDEND_ARCHIVE"
    )
    result["direct_cashflow_duration_status"] = "NO_VIEW_PROXY_ONLY"
    return result, pca


def average_pairwise_correlation(returns: pd.DataFrame) -> float:
    """计算有足够共同观测的平均非对角相关系数。"""

    if returns.shape[1] < 2 or returns.shape[0] < 20:
        return np.nan
    correlation = returns.corr(min_periods=max(20, returns.shape[0] // 2))
    values = correlation.to_numpy(dtype=float)
    upper = values[np.triu_indices_from(values, k=1)]
    upper = upper[np.isfinite(upper)]
    return float(upper.mean()) if len(upper) else np.nan


def detect_drawdown_episodes(
    dates: Sequence[pd.Timestamp],
    wealth: Sequence[float],
    *,
    activation_threshold: float,
) -> pd.DataFrame:
    """以完整水下周期机械识别回撤，不允许人工挑年份。"""

    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    values = np.asarray(wealth, dtype=float)
    if len(date_index) != len(values) or len(values) == 0:
        raise ValueError("日期与财富序列长度不一致或为空")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("财富序列必须为有限正数")
    peak_index = 0
    underwater_start: int | None = None
    trough_index = 0
    trough_drawdown = 0.0
    rows: list[dict[str, Any]] = []

    def close_episode(recovery_index: int | None) -> None:
        nonlocal underwater_start, trough_index, trough_drawdown
        if underwater_start is None or trough_drawdown > activation_threshold:
            underwater_start = None
            trough_drawdown = 0.0
            return
        rows.append(
            {
                "peak_date": date_index[peak_index],
                "underwater_start_date": date_index[underwater_start],
                "trough_date": date_index[trough_index],
                "recovery_date": (
                    date_index[recovery_index] if recovery_index is not None else pd.NaT
                ),
                "peak_to_trough_drawdown": float(trough_drawdown),
                "peak_to_trough_market_days": int(trough_index - peak_index),
                "trough_to_recovery_market_days": (
                    int(recovery_index - trough_index)
                    if recovery_index is not None
                    else None
                ),
                "right_censored": recovery_index is None,
            }
        )
        underwater_start = None
        trough_drawdown = 0.0

    for index in range(1, len(values)):
        if values[index] >= values[peak_index]:
            if underwater_start is not None:
                close_episode(index)
            peak_index = index
            trough_index = index
            continue
        drawdown = values[index] / values[peak_index] - 1.0
        if underwater_start is None:
            underwater_start = index
            trough_index = index
            trough_drawdown = drawdown
        elif drawdown < trough_drawdown:
            trough_index = index
            trough_drawdown = drawdown
    if underwater_start is not None:
        close_episode(None)
    return pd.DataFrame(rows)


def detect_sideways_windows(
    dates: Sequence[pd.Timestamp],
    wealth: Sequence[float],
    *,
    lookback: int,
    absolute_return_max: float,
    range_max: float,
) -> pd.DataFrame:
    """识别并合并重叠的固定 252 日横盘窗口。"""

    frame = pd.DataFrame(
        {"date": pd.to_datetime(list(dates)), "wealth": np.asarray(wealth, float)}
    ).sort_values("date")
    qualifying: list[tuple[int, int]] = []
    for end in range(lookback - 1, len(frame)):
        start = end - lookback + 1
        window = frame.iloc[start : end + 1]
        total_return = float(window["wealth"].iloc[-1] / window["wealth"].iloc[0] - 1)
        peak_to_trough_range = float(
            window["wealth"].max() / window["wealth"].min() - 1
        )
        if abs(total_return) <= absolute_return_max and peak_to_trough_range <= range_max:
            qualifying.append((start, end))
    if not qualifying:
        return pd.DataFrame(
            columns=["start_date", "end_date", "market_days", "event_type"]
        )
    merged: list[list[int]] = []
    for start, end in qualifying:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return pd.DataFrame(
        [
            {
                "start_date": frame.iloc[start]["date"],
                "end_date": frame.iloc[end]["date"],
                "market_days": int(end - start + 1),
                "event_type": "LONG_SIDEWAYS",
            }
            for start, end in merged
        ]
    )
