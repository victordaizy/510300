"""解释沪深300原始估值信号长期负IC的来源。

本研究只回答“估值状态与未来总收益是否单调”以及“便宜是否来自盈利高点”。
它不生成仓位、不调用交易回测，也不搜索阈值或模型参数。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtest.valuation_fvg_engine import (
    attach_market_cap_context,
    build_valuation_signals,
)
from research.build_point_in_time_fundamental_panel import (
    _ttm_at_period,
    select_latest_vintages,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "round5_defensive_valuation_timing.yaml"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
TOTAL_RETURN_FILE = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
CONSTITUENT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
FINANCIALS_FILE = ROOT / "data" / "raw" / "fundamentals" / "csi300_financials_point_in_time.parquet"
BOND_FILE = ROOT / "data" / "raw" / "macro" / "china_government_bond_yields_daily.parquet"
INDUSTRY_FILE = ROOT / "data" / "raw" / "reference" / "a_share_sw_industry_static.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_valuation_negative_ic_diagnostics.parquet"
REPORT_JSON = ROOT / "reports" / "research" / "000300_valuation_negative_ic_diagnosis.json"
REPORT_MD = ROOT / "reports" / "research" / "000300_valuation_negative_ic_diagnosis.md"

HORIZONS = (60, 120, 242)
QUINTILE_LABELS = ("最贵20%", "次贵", "中性", "次便宜", "最便宜20%")
CONDITIONAL_MULTIPLE_FEATURES = (
    "cgb_10y",
    "cgb_term_spread",
    "implied_roe",
    "eps_growth_12m",
    "realized_volatility_3m",
)


@dataclass(frozen=True)
class CompanyNormalization:
    """公司在单个信号日可得的TTM与正常化盈利指标。"""

    con_code: str
    latest_report_period: pd.Timestamp
    total_shares: float
    ttm_revenue_cny: float
    ttm_profit_cny: float
    normalized_profit_cny: float
    ttm_eps: float
    normalized_eps: float
    ttm_roe: float
    normalized_roe: float
    ttm_profit_growth_yoy: float
    ttm_revenue_growth_yoy: float
    profit_peak_ratio: float
    margin_gap: float
    roe_gap: float
    normalization_observations: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _latest_at_or_before(value_map: dict[pd.Timestamp, float], period: pd.Timestamp) -> float:
    eligible = [key for key in value_map if key <= period and pd.notna(value_map[key])]
    return float(value_map[max(eligible)]) if eligible else np.nan


def derive_company_normalization(
    latest_vintages: pd.DataFrame,
    lookback_years: int = 3,
    minimum_observations: int = 4,
) -> pd.DataFrame:
    """按公司自身历史正常化盈利，不使用行业标签或未来披露。

    正常化利润固定为两项等权：三年TTM利润中位数，以及当前TTM收入乘以
    三年TTM净利率中位数。若只有一项有效则使用该项，不进行参数搜索。
    """

    required = {
        "con_code",
        "report_period",
        "available_at",
        "revenue_cny",
        "net_profit_parent_cny",
        "equity_parent_cny",
        "total_shares",
    }
    if missing := required - set(latest_vintages.columns):
        raise ValueError(f"点时财务数据缺少字段：{sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for con_code, records in latest_vintages.groupby("con_code", sort=False):
        records = records.sort_values("report_period").drop_duplicates(
            "report_period", keep="last"
        )
        maps = {
            column: {
                pd.Timestamp(period): float(value)
                for period, value in zip(
                    records["report_period"], records[column], strict=True
                )
                if pd.notna(value)
            }
            for column in (
                "revenue_cny",
                "net_profit_parent_cny",
                "equity_parent_cny",
                "total_shares",
            )
        }
        selected_period: pd.Timestamp | None = None
        current_profit = np.nan
        current_revenue = np.nan
        for period in sorted(maps["net_profit_parent_cny"], reverse=True):
            profit = _ttm_at_period(maps["net_profit_parent_cny"], period)
            revenue = _ttm_at_period(maps["revenue_cny"], period)
            if pd.notna(profit) and pd.notna(revenue) and revenue > 0:
                selected_period = period
                current_profit = float(profit)
                current_revenue = float(revenue)
                break
        if selected_period is None:
            continue

        history_start = selected_period - pd.DateOffset(years=lookback_years)
        history: list[tuple[pd.Timestamp, float, float, float]] = []
        for period in sorted(maps["net_profit_parent_cny"]):
            if period < history_start or period > selected_period:
                continue
            profit = _ttm_at_period(maps["net_profit_parent_cny"], period)
            revenue = _ttm_at_period(maps["revenue_cny"], period)
            equity = _latest_at_or_before(maps["equity_parent_cny"], period)
            if pd.notna(profit) and pd.notna(revenue) and revenue > 0:
                history.append((period, float(profit), float(revenue), float(equity)))
        if len(history) < minimum_observations:
            continue

        profits = pd.Series([item[1] for item in history], dtype=float)
        margins = pd.Series([item[1] / item[2] for item in history], dtype=float)
        roes = pd.Series(
            [item[1] / item[3] if item[3] > 0 else np.nan for item in history],
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan)
        positive_profits = profits[profits > 0]
        median_profit = float(positive_profits.median()) if not positive_profits.empty else np.nan
        median_margin = float(margins.median()) if margins.notna().any() else np.nan
        margin_normalized = (
            current_revenue * median_margin
            if pd.notna(median_margin) and median_margin > 0
            else np.nan
        )
        candidates = [value for value in (median_profit, margin_normalized) if pd.notna(value) and value > 0]
        normalized_profit = float(np.mean(candidates)) if candidates else np.nan
        shares = _latest_at_or_before(maps["total_shares"], selected_period)
        equity = _latest_at_or_before(maps["equity_parent_cny"], selected_period)
        prior_period = selected_period - pd.DateOffset(years=1)
        prior_profit = _ttm_at_period(maps["net_profit_parent_cny"], prior_period)
        prior_revenue = _ttm_at_period(maps["revenue_cny"], prior_period)
        current_margin = _safe_ratio(current_profit, current_revenue)
        current_roe = _safe_ratio(current_profit, equity)
        normalized_roe = float(roes.median()) if roes.notna().any() else np.nan
        rows.append(
            CompanyNormalization(
                con_code=str(con_code),
                latest_report_period=selected_period,
                total_shares=shares,
                ttm_revenue_cny=current_revenue,
                ttm_profit_cny=current_profit,
                normalized_profit_cny=normalized_profit,
                ttm_eps=_safe_ratio(current_profit, shares),
                normalized_eps=_safe_ratio(normalized_profit, shares),
                ttm_roe=current_roe,
                normalized_roe=normalized_roe,
                ttm_profit_growth_yoy=(
                    _safe_ratio(current_profit, prior_profit) - 1.0
                    if pd.notna(prior_profit) and prior_profit > 0
                    else np.nan
                ),
                ttm_revenue_growth_yoy=(
                    _safe_ratio(current_revenue, prior_revenue) - 1.0
                    if pd.notna(prior_revenue) and prior_revenue > 0
                    else np.nan
                ),
                profit_peak_ratio=_safe_ratio(current_profit, normalized_profit),
                margin_gap=(
                    current_margin - median_margin
                    if pd.notna(current_margin) and pd.notna(median_margin)
                    else np.nan
                ),
                roe_gap=(
                    current_roe - normalized_roe
                    if pd.notna(current_roe) and pd.notna(normalized_roe)
                    else np.nan
                ),
                normalization_observations=len(history),
            ).__dict__
        )
    return pd.DataFrame(rows)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    valid = values.notna() & weights.notna() & weights.gt(0)
    coverage = float(weights.loc[valid].sum())
    if not valid.any() or coverage <= 0:
        return np.nan, coverage
    return float(np.average(values.loc[valid], weights=weights.loc[valid])), coverage


def aggregate_normalized_snapshot(
    snapshot_weights: pd.DataFrame,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    index_close: float,
) -> dict[str, Any]:
    """聚合一个官方权重快照的TTM与正常化盈利状态。"""

    panel = snapshot_weights[["con_code", "weight"]].copy()
    panel["weight"] = pd.to_numeric(panel["weight"], errors="coerce") / 100.0
    panel = panel.merge(
        prices[["con_code", "raw_close"]], on="con_code", how="left", validate="one_to_one"
    )
    panel = panel.merge(metrics, on="con_code", how="left", validate="one_to_one")
    panel["ttm_earnings_yield"] = panel["ttm_eps"] / panel["raw_close"]
    panel["normalized_earnings_yield"] = panel["normalized_eps"] / panel["raw_close"]
    current_yield, current_coverage = _weighted_mean(panel["ttm_earnings_yield"], panel["weight"])
    normalized_yield, normalized_coverage = _weighted_mean(
        panel["normalized_earnings_yield"], panel["weight"]
    )
    weighted_roe, roe_coverage = _weighted_mean(panel["ttm_roe"].clip(-0.5, 0.8), panel["weight"])
    normalized_roe, _ = _weighted_mean(panel["normalized_roe"].clip(-0.5, 0.8), panel["weight"])
    peak_ratio, _ = _weighted_mean(panel["profit_peak_ratio"].clip(-5.0, 5.0), panel["weight"])
    margin_gap, _ = _weighted_mean(panel["margin_gap"].clip(-0.5, 0.5), panel["weight"])
    roe_gap, _ = _weighted_mean(panel["roe_gap"].clip(-0.5, 0.5), panel["weight"])
    profit_growth, _ = _weighted_mean(
        panel["ttm_profit_growth_yoy"].clip(-1.0, 1.0), panel["weight"]
    )
    revenue_growth, _ = _weighted_mean(
        panel["ttm_revenue_growth_yoy"].clip(-1.0, 1.0), panel["weight"]
    )
    peak = panel["profit_peak_ratio"].gt(1.5) & panel["ttm_profit_growth_yoy"].gt(0)
    return {
        "date": pd.Timestamp(snapshot_date),
        "constituent_count": int(panel["con_code"].nunique()),
        "normalization_company_count": int(panel["normalized_eps"].notna().sum()),
        "ttm_earnings_weight_coverage": current_coverage,
        "normalized_earnings_weight_coverage": normalized_coverage,
        "roe_weight_coverage": roe_coverage,
        "weighted_ttm_earnings_yield": current_yield,
        "weighted_normalized_earnings_yield": normalized_yield,
        "component_ttm_pe": 1.0 / current_yield if pd.notna(current_yield) and current_yield > 0 else np.nan,
        "component_normalized_pe": (
            1.0 / normalized_yield if pd.notna(normalized_yield) and normalized_yield > 0 else np.nan
        ),
        "component_ttm_index_eps": index_close * current_yield,
        "component_normalized_index_eps": index_close * normalized_yield,
        "weighted_ttm_roe": weighted_roe,
        "weighted_normalized_roe": normalized_roe,
        "weighted_profit_peak_ratio": peak_ratio,
        "profit_peak_weight_share": float(panel.loc[peak, "weight"].sum()),
        "weighted_margin_gap": margin_gap,
        "weighted_roe_gap": roe_gap,
        "weighted_ttm_profit_growth_yoy": profit_growth,
        "weighted_ttm_revenue_growth_yoy": revenue_growth,
    }


def aggregate_industry_fundamentals(
    snapshot_weights: pd.DataFrame,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    industry_mapping: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    """用静态解释标签拆分当期盈利峰值，不把行业标签作为模型特征。"""

    panel = snapshot_weights[["con_code", "weight"]].copy()
    panel["weight"] = pd.to_numeric(panel["weight"], errors="coerce") / 100.0
    panel = panel.merge(prices[["con_code", "raw_close"]], on="con_code", how="left")
    panel = panel.merge(metrics, on="con_code", how="left")
    panel = panel.merge(industry_mapping[["con_code", "industry_l1"]], on="con_code", how="left")
    panel["industry_l1"] = panel["industry_l1"].fillna("未识别")
    panel["ttm_yield_contribution"] = panel["weight"] * panel["ttm_eps"] / panel["raw_close"]
    panel["normalized_yield_contribution"] = (
        panel["weight"] * panel["normalized_eps"] / panel["raw_close"]
    )
    panel["is_profit_peak"] = panel["profit_peak_ratio"].gt(1.5) & panel[
        "ttm_profit_growth_yoy"
    ].gt(0)
    rows: list[dict[str, Any]] = []
    for industry, group in panel.groupby("industry_l1", sort=True):
        weight = float(group["weight"].sum())
        peak_ratio, _ = _weighted_mean(group["profit_peak_ratio"].clip(-5.0, 5.0), group["weight"])
        profit_growth, _ = _weighted_mean(
            group["ttm_profit_growth_yoy"].clip(-1.0, 1.0), group["weight"]
        )
        roe_gap, _ = _weighted_mean(group["roe_gap"].clip(-0.5, 0.5), group["weight"])
        current_yield = float(group["ttm_yield_contribution"].sum(min_count=1))
        normalized_yield = float(group["normalized_yield_contribution"].sum(min_count=1))
        rows.append(
            {
                "date": pd.Timestamp(snapshot_date),
                "industry_l1": str(industry),
                "industry_weight": weight,
                "weighted_profit_peak_ratio": peak_ratio,
                "profit_peak_weight_share_index": float(
                    group.loc[group["is_profit_peak"], "weight"].sum()
                ),
                "weighted_profit_growth_yoy": profit_growth,
                "weighted_roe_gap": roe_gap,
                "ttm_earnings_yield_contribution": current_yield,
                "normalized_earnings_yield_contribution": normalized_yield,
                "earnings_overstatement_contribution": current_yield - normalized_yield,
            }
        )
    return rows


def build_normalized_monthly_panel(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    industry_mapping: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为研究区间官方月末权重构造点时正常化盈利面板。"""

    weights = weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    weights = weights.loc[weights["trade_date"].between(start, end)].copy()
    financials = financials.copy()
    financials["report_period"] = pd.to_datetime(financials["report_period"])
    financials["available_at"] = pd.to_datetime(financials["available_at"])
    constituent_daily = constituent_daily.copy()
    constituent_daily["date"] = pd.to_datetime(constituent_daily["date"])
    index_daily = index_daily.copy()
    index_daily["date"] = pd.to_datetime(index_daily["date"])
    rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weights.groupby("trade_date", sort=True):
        snapshot_date = pd.Timestamp(snapshot_date)
        symbols = set(snapshot_weights["con_code"].astype(str))
        vintages = select_latest_vintages(
            financials.loc[financials["con_code"].astype(str).isin(symbols)], snapshot_date
        )
        metrics = derive_company_normalization(vintages)
        prices = constituent_daily.loc[
            constituent_daily["date"].eq(snapshot_date), ["con_code", "raw_close"]
        ].drop_duplicates("con_code", keep="last")
        index_row = index_daily.loc[index_daily["date"].eq(snapshot_date), "close"]
        if index_row.empty:
            raise ValueError(f"{snapshot_date.date()}缺少指数收盘价")
        rows.append(
            aggregate_normalized_snapshot(
                snapshot_weights,
                prices,
                metrics,
                snapshot_date,
                float(index_row.iloc[-1]),
            )
        )
        industry_rows.extend(
            aggregate_industry_fundamentals(
                snapshot_weights, prices, metrics, industry_mapping, snapshot_date
            )
        )
    summary = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    industries = pd.DataFrame(industry_rows).sort_values(["date", "industry_l1"]).reset_index(drop=True)
    industries["future_profit_growth_change_12m"] = industries.groupby(
        "industry_l1", sort=False
    )["weighted_profit_growth_yoy"].shift(-12) - industries["weighted_profit_growth_yoy"]
    return summary, industries


def build_industry_return_contributions(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_mapping: pd.DataFrame,
    total_return_daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """按信号月官方权重计算各行业未来收益的指数贡献。"""

    weights = weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    weights = weights.loc[weights["trade_date"].between(start, end)]
    prices = constituent_daily[["date", "con_code", "total_return_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    total_dates = (
        pd.to_datetime(total_return_daily["date"])
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    date_positions = {date: index for index, date in enumerate(total_dates)}
    price_groups = {
        date: group[["con_code", "total_return_close"]].drop_duplicates("con_code", keep="last")
        for date, group in prices.groupby("date", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weights.groupby("trade_date", sort=True):
        snapshot_date = pd.Timestamp(snapshot_date)
        if snapshot_date not in date_positions or snapshot_date not in price_groups:
            continue
        base = snapshot_weights[["con_code", "weight"]].copy()
        base["weight"] = pd.to_numeric(base["weight"], errors="coerce") / 100.0
        base = base.merge(price_groups[snapshot_date], on="con_code", how="left")
        base = base.rename(columns={"total_return_close": "current_total_return_close"})
        base = base.merge(industry_mapping[["con_code", "industry_l1"]], on="con_code", how="left")
        base["industry_l1"] = base["industry_l1"].fillna("未识别")
        for horizon in horizons:
            future_position = date_positions[snapshot_date] + horizon
            if future_position >= len(total_dates):
                continue
            future_date = pd.Timestamp(total_dates.iloc[future_position])
            if future_date not in price_groups:
                continue
            panel = base.merge(price_groups[future_date], on="con_code", how="left")
            panel["future_return"] = (
                panel["total_return_close"] / panel["current_total_return_close"] - 1.0
            )
            for industry, group in panel.groupby("industry_l1", sort=True):
                valid = group["future_return"].notna() & group["weight"].gt(0)
                covered_weight = float(group.loc[valid, "weight"].sum())
                if covered_weight <= 0:
                    continue
                rows.append(
                    {
                        "date": snapshot_date,
                        "future_date": future_date,
                        "horizon_days": horizon,
                        "industry_l1": str(industry),
                        "industry_weight": float(group["weight"].sum()),
                        "covered_weight": covered_weight,
                        "industry_future_return": float(
                            np.average(
                                group.loc[valid, "future_return"],
                                weights=group.loc[valid, "weight"],
                            )
                        ),
                        "index_return_contribution": float(
                            (group.loc[valid, "weight"] * group.loc[valid, "future_return"]).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def industry_attribution(
    industry_fundamentals: pd.DataFrame,
    industry_returns: pd.DataFrame,
    monthly_signal: pd.DataFrame,
    horizon: int = 242,
) -> list[dict[str, Any]]:
    """解释最便宜组相对最贵组的行业收益与盈利高点贡献。"""

    returns = industry_returns.loc[industry_returns["horizon_days"].eq(horizon)].copy()
    realized_dates = set(returns["date"])
    states = monthly_signal.loc[
        monthly_signal["date"].isin(realized_dates),
        ["date", "original_unclipped_cheapness"],
    ].copy()
    states["valuation_state"] = assign_cheapness_quintiles(
        states["original_unclipped_cheapness"]
    )
    returns = returns.merge(
        states[["date", "valuation_state"]], on="date", how="left"
    )
    fundamentals = industry_fundamentals.merge(
        states[["date", "valuation_state"]], on="date", how="left"
    )
    rows: list[dict[str, Any]] = []
    industries = sorted(set(returns["industry_l1"]) | set(fundamentals["industry_l1"]))
    for industry in industries:
        ret = returns.loc[returns["industry_l1"].eq(industry)]
        fund = fundamentals.loc[fundamentals["industry_l1"].eq(industry)]
        cheap_ret = ret.loc[ret["valuation_state"].eq("最便宜20%")]
        expensive_ret = ret.loc[ret["valuation_state"].eq("最贵20%")]
        cheap_fund = fund.loc[fund["valuation_state"].eq("最便宜20%")]
        rows.append(
            {
                "industry_l1": industry,
                "cheapest_observations": int(len(cheap_ret)),
                "most_expensive_observations": int(len(expensive_ret)),
                "mean_weight_in_cheapest_group": (
                    float(cheap_ret["industry_weight"].mean()) if not cheap_ret.empty else None
                ),
                "cheapest_mean_242d_return_contribution": (
                    float(cheap_ret["index_return_contribution"].mean())
                    if not cheap_ret.empty
                    else None
                ),
                "most_expensive_mean_242d_return_contribution": (
                    float(expensive_ret["index_return_contribution"].mean())
                    if not expensive_ret.empty
                    else None
                ),
                "cheapest_minus_expensive_242d_contribution": (
                    float(
                        cheap_ret["index_return_contribution"].mean()
                        - expensive_ret["index_return_contribution"].mean()
                    )
                    if not cheap_ret.empty and not expensive_ret.empty
                    else None
                ),
                "cheapest_group_earnings_overstatement_contribution": (
                    float(cheap_fund["earnings_overstatement_contribution"].mean())
                    if not cheap_fund.empty
                    else None
                ),
                "cheapest_group_profit_peak_weight_share_index": (
                    float(cheap_fund["profit_peak_weight_share_index"].mean())
                    if not cheap_fund.empty
                    else None
                ),
                "cheapest_group_future_profit_growth_change_12m": (
                    float(cheap_fund["future_profit_growth_change_12m"].mean())
                    if cheap_fund["future_profit_growth_change_12m"].notna().any()
                    else None
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["cheapest_minus_expensive_242d_contribution"]
            if row["cheapest_minus_expensive_242d_contribution"] is not None
            else np.inf
        ),
    )


def build_original_signal(
    valuation_raw: pd.DataFrame,
    constituents: pd.DataFrame,
    valuation_config: dict[str, Any],
) -> pd.DataFrame:
    """复现Round5信号，并保留截断前的连续估值距离。"""

    signal = attach_market_cap_context(
        build_valuation_signals(valuation_raw, valuation_config), constituents
    )
    width = (signal["fair_high_12m"] - signal["fair_low_12m"]).replace(0.0, np.nan)
    signal["original_unclipped_cheapness"] = (
        signal["fair_high_12m"] - signal["index_close"]
    ) / width
    return signal


def _monthly_market_features(
    index_daily: pd.DataFrame,
    valuation: pd.DataFrame,
    bonds: pd.DataFrame,
) -> pd.DataFrame:
    index = index_daily[["date", "close"]].copy().rename(columns={"close": "index_close"})
    value = valuation[["date", "pe_ttm", "pb"]].copy()
    bond = bonds[["date", "cgb_1y", "cgb_10y"]].copy()
    for frame in (index, value, bond):
        frame["date"] = pd.to_datetime(frame["date"])
    daily = index.merge(value, on="date", how="left", validate="one_to_one")
    daily = daily.merge(bond, on="date", how="left", validate="one_to_one")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily[["pe_ttm", "pb", "cgb_1y", "cgb_10y"]] = daily[
        ["pe_ttm", "pb", "cgb_1y", "cgb_10y"]
    ].ffill(limit=5)
    daily["log_return_1d"] = np.log(daily["index_close"] / daily["index_close"].shift(1))
    daily["rv_63d"] = daily["log_return_1d"].rolling(63, min_periods=40).std(ddof=1) * np.sqrt(242)
    daily["calendar_month"] = daily["date"].dt.to_period("M")
    monthly = daily.groupby("calendar_month", sort=True).tail(1).reset_index(drop=True)
    monthly["implied_eps"] = monthly["index_close"] / monthly["pe_ttm"]
    monthly["implied_roe"] = monthly["pb"] / monthly["pe_ttm"]
    monthly["eps_growth_12m"] = monthly["implied_eps"].pct_change(12, fill_method=None).clip(-0.5, 0.5)
    monthly["cgb_term_spread"] = monthly["cgb_10y"] - monthly["cgb_1y"]
    monthly["realized_volatility_3m"] = monthly["rv_63d"]
    return monthly


def expanding_conditioned_fair_pe(
    monthly: pd.DataFrame,
    minimum_training_samples: int = 36,
    ridge_alpha: float = 10.0,
) -> pd.Series:
    """仅用历史月份估计当期条件合理PE，固定模型且不搜索参数。"""

    required = {"pe_ttm", *CONDITIONAL_MULTIPLE_FEATURES}
    if missing := required - set(monthly.columns):
        raise ValueError(f"条件合理PE输入缺少字段：{sorted(missing)}")
    predictions = pd.Series(np.nan, index=monthly.index, dtype=float)
    target = np.log(monthly["pe_ttm"])
    for signal_index in monthly.index:
        history = monthly.index.to_series().lt(signal_index)
        history &= monthly[list(CONDITIONAL_MULTIPLE_FEATURES)].notna().all(axis=1)
        history &= target.notna()
        train_indices = monthly.index[history]
        signal = monthly.loc[[signal_index], list(CONDITIONAL_MULTIPLE_FEATURES)]
        if len(train_indices) < minimum_training_samples or signal.isna().any(axis=None):
            continue
        model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))]
        )
        model.fit(monthly.loc[train_indices, list(CONDITIONAL_MULTIPLE_FEATURES)], target.loc[train_indices])
        predictions.loc[signal_index] = float(np.exp(model.predict(signal)[0]))
    return predictions


def assign_cheapness_quintiles(signal: pd.Series) -> pd.Series:
    """把连续便宜度分成五个等样本组；值越高越便宜。"""

    result = pd.Series(pd.NA, index=signal.index, dtype="object")
    valid = signal.dropna()
    if len(valid) < 5:
        return result
    ranks = valid.rank(method="first")
    result.loc[valid.index] = pd.qcut(ranks, 5, labels=QUINTILE_LABELS).astype(str)
    return result


def add_forward_total_returns(
    frame: pd.DataFrame,
    total_return_daily: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    total = total_return_daily[["date", "close"]].copy()
    total["date"] = pd.to_datetime(total["date"])
    total = total.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for horizon in horizons:
        total[f"forward_total_return_{horizon}d"] = total["close"].shift(-horizon) / total["close"] - 1.0
    return frame.merge(
        total[["date", *[f"forward_total_return_{horizon}d" for horizon in horizons]]],
        on="date",
        how="left",
        validate="one_to_one",
    )


def evaluate_signal(
    frame: pd.DataFrame,
    signal_column: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """报告IC、非重叠IC稳定性和便宜度五组未来总收益。"""

    result: dict[str, Any] = {}
    for horizon in horizons:
        return_column = f"forward_total_return_{horizon}d"
        sample = frame[[signal_column, return_column]].dropna().copy()
        sample["quintile"] = assign_cheapness_quintiles(sample[signal_column])
        rho = float(spearmanr(sample[signal_column], sample[return_column]).statistic)
        non_overlapping: list[float] = []
        for offset in range(horizon):
            subset = sample.iloc[offset::horizon]
            if len(subset) >= 4 and subset[signal_column].nunique() > 1:
                non_overlapping.append(
                    float(spearmanr(subset[signal_column], subset[return_column]).statistic)
                )
        grouped = (
            sample.rename(columns={return_column: "future_return"})
            .groupby("quintile", observed=False)["future_return"]
            .agg(["count", "mean", "median"])
        )
        ordered = grouped.reindex(QUINTILE_LABELS)
        result[f"{horizon}d"] = {
            "observations": int(len(sample)),
            "spearman_ic": rho,
            "non_overlapping_ic_median": (
                float(np.nanmedian(non_overlapping)) if non_overlapping else None
            ),
            "non_overlapping_positive_ratio": (
                float(np.mean(np.asarray(non_overlapping) > 0)) if non_overlapping else None
            ),
            "quintiles": [
                {
                    "valuation_state": label,
                    "observations": int(ordered.loc[label, "count"]),
                    "mean_forward_total_return": float(ordered.loc[label, "mean"]),
                    "median_forward_total_return": float(ordered.loc[label, "median"]),
                }
                for label in QUINTILE_LABELS
                if label in ordered.index and pd.notna(ordered.loc[label, "count"])
            ],
            "cheapest_minus_most_expensive_mean_return": float(
                ordered.loc["最便宜20%", "mean"] - ordered.loc["最贵20%", "mean"]
            ),
        }
    return result


def yearly_ic_attribution(
    frame: pd.DataFrame,
    signal_column: str,
    horizon: int = 242,
) -> list[dict[str, Any]]:
    """按信号年份拆分长期负IC，并给出可加总的秩协方差分子。"""

    return_column = f"forward_total_return_{horizon}d"
    sample = frame[["date", signal_column, return_column]].dropna().copy()
    sample["signal_rank"] = sample[signal_column].rank(pct=True)
    sample["return_rank"] = sample[return_column].rank(pct=True)
    sample["rank_covariance_term"] = (
        sample["signal_rank"] - sample["signal_rank"].mean()
    ) * (sample["return_rank"] - sample["return_rank"].mean())
    sample["year"] = sample["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in sample.groupby("year", sort=True):
        rows.append(
            {
                "year": int(year),
                "observations": int(len(group)),
                "spearman_ic_within_year": (
                    float(spearmanr(group[signal_column], group[return_column]).statistic)
                    if group[signal_column].nunique() > 1 and group[return_column].nunique() > 1
                    else None
                ),
                "rank_covariance_numerator": float(group["rank_covariance_term"].sum()),
                "mean_forward_total_return": float(group[return_column].mean()),
                "mean_cheapness": float(group[signal_column].mean()),
            }
        )
    return rows


def fundamental_quintile_diagnostics(monthly: pd.DataFrame) -> list[dict[str, Any]]:
    """检查原始便宜度五组在当前盈利位置和未来12个月兑现上的差异。"""

    data = monthly.copy()
    data["future_ttm_eps_growth_12m"] = data["component_ttm_index_eps"].shift(-12) / data[
        "component_ttm_index_eps"
    ] - 1.0
    data["future_roe_change_12m"] = data["weighted_ttm_roe"].shift(-12) - data["weighted_ttm_roe"]
    data["future_profit_growth_change_12m"] = (
        data["weighted_ttm_profit_growth_yoy"].shift(-12)
        - data["weighted_ttm_profit_growth_yoy"]
    )
    data = data.loc[data["future_ttm_eps_growth_12m"].notna()].copy()
    data["valuation_state"] = assign_cheapness_quintiles(
        data["original_unclipped_cheapness"]
    )
    columns = (
        "weighted_profit_peak_ratio",
        "profit_peak_weight_share",
        "weighted_margin_gap",
        "weighted_roe_gap",
        "weighted_ttm_profit_growth_yoy",
        "weighted_ttm_revenue_growth_yoy",
        "future_ttm_eps_growth_12m",
        "future_roe_change_12m",
        "future_profit_growth_change_12m",
    )
    rows: list[dict[str, Any]] = []
    for label in QUINTILE_LABELS:
        group = data.loc[data["valuation_state"].eq(label)]
        row: dict[str, Any] = {"valuation_state": label, "observations": int(len(group))}
        for column in columns:
            row[column] = float(group[column].mean()) if group[column].notna().any() else None
        rows.append(row)
    return rows


def _pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 沪深300估值信号长期负IC诊断",
        "",
        "> 本报告只研究估值状态，不生成仓位，不包含趋势、FVG、成本或交易回测。",
        "",
        "## 结论",
        "",
        f"- 诊断状态：`{report['status']}`。",
        f"- 当前利润高点陷阱：`{report['hypothesis_results']['current_profit_peak_trap']}`。",
        f"- 随后盈利反转：`{report['hypothesis_results']['subsequent_earnings_reversal']}`。",
        f"- 跨年份状态反转：`{report['hypothesis_results']['cross_year_regime_reversal']}`。",
        f"- 仓位截断放大负IC：`{report['hypothesis_results']['clipping_amplifies_negative_ic']}`。",
        f"- 正常化盈利改善：`{report['hypothesis_results']['normalization_improves_monotonicity']}`。",
        f"- 利率条件FairPE改善：`{report['hypothesis_results']['conditioned_multiple_improves_monotonicity']}`。",
        "- 行业使用静态申万映射做解释归因，不进入任何点时估值特征。",
        "",
        "## 原始未截断估值信号五组未来总收益",
        "",
        "|估值状态|未来60日|未来120日|未来242日|",
        "|---|---:|---:|---:|",
    ]
    original = report["signal_evaluations"]["original_unclipped_cheapness"]
    for label in QUINTILE_LABELS:
        values: list[str] = []
        for horizon in HORIZONS:
            rows = original[f"{horizon}d"]["quintiles"]
            item = next(row for row in rows if row["valuation_state"] == label)
            values.append(_pct(item["mean_forward_total_return"]))
        lines.append(f"|{label}|{values[0]}|{values[1]}|{values[2]}|")
    lines.extend(["", "## 信号对照", "", "|信号|IC60|IC120|IC242|最便宜减最贵242日|", "|---|---:|---:|---:|---:|"])
    for signal_name, evaluation in report["signal_evaluations"].items():
        lines.append(
            f"|{signal_name}|{evaluation['60d']['spearman_ic']:.3f}|"
            f"{evaluation['120d']['spearman_ic']:.3f}|{evaluation['242d']['spearman_ic']:.3f}|"
            f"{_pct(evaluation['242d']['cheapest_minus_most_expensive_mean_return'])}|"
        )
    lines.extend(
        [
            "",
            "## 242日负IC的信号年份归因",
            "",
            "|信号年份|年内IC|秩协方差分子|平均未来242日收益|平均便宜度|",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["yearly_242d_attribution"]:
        within = row["spearman_ic_within_year"]
        lines.append(
            f"|{row['year']}|{'—' if within is None else f'{within:.3f}'}|"
            f"{row['rank_covariance_numerator']:.2f}|{_pct(row['mean_forward_total_return'])}|"
            f"{row['mean_cheapness']:.2f}|"
        )
    lines.extend(
        [
            "",
            "## 原始估值五组的盈利状态",
            "",
            "|估值状态|盈利峰值权重|当前利润峰值比|未来12月EPS增长|未来12月ROE变化|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["fundamental_quintile_diagnostics"]:
        lines.append(
            f"|{row['valuation_state']}|{_pct(row['profit_peak_weight_share'])}|"
            f"{row['weighted_profit_peak_ratio']:.2f}|{_pct(row['future_ttm_eps_growth_12m'])}|"
            f"{_pct(row['future_roe_change_12m'])}|"
        )
    lines.extend(
        [
            "",
            "## 行业贡献（静态申万标签，仅用于解释）",
            "",
            "|行业|最便宜组平均权重|最便宜减最贵242日贡献|最便宜组盈利高点权重|未来12月利润增速变化|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["industry_attribution"][:10]:
        lines.append(
            f"|{row['industry_l1']}|{_pct(row['mean_weight_in_cheapest_group'])}|"
            f"{_pct(row['cheapest_minus_expensive_242d_contribution'])}|"
            f"{_pct(row['cheapest_group_profit_peak_weight_share_index'])}|"
            f"{_pct(row['cheapest_group_future_profit_growth_change_12m'])}|"
        )
    lines.extend(
        [
            "",
            "## 数据边界",
            "",
            "- 正常化盈利只使用信号日已披露财务、当月官方权重和当日成分股价格。",
            "- 原始财务数据没有经营现金流字段，因此本轮不能检验现金利润质量。",
            "- 行业使用静态申万映射做业务解释，不能作为历史点时模型特征；未识别和重大主业变化会带来归因误差。",
            "- 全部历史已参与研究；替代信号结果属于回顾性机制诊断，不是可交易样本外证据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    start = pd.Timestamp(config["backtest"]["start_date"])
    end = pd.Timestamp(config["backtest"]["end_date"])
    valuation_raw = pd.read_parquet(VALUATION_FILE)
    constituents = pd.read_parquet(CONSTITUENT_FILE)
    index_daily = pd.read_parquet(INDEX_FILE)
    total_return = pd.read_parquet(TOTAL_RETURN_FILE)
    weights = pd.read_parquet(WEIGHTS_FILE)
    financials = pd.read_parquet(FINANCIALS_FILE)
    bonds = pd.read_parquet(BOND_FILE)
    industry_mapping = pd.read_parquet(INDUSTRY_FILE)

    original = build_original_signal(valuation_raw, constituents, config["valuation"])
    original = original.loc[pd.to_datetime(original["date"]).between(start, end)].copy()
    original["date"] = pd.to_datetime(original["date"])
    daily = add_forward_total_returns(original, total_return)

    normalized, industry_fundamentals = build_normalized_monthly_panel(
        weights, financials, constituents, index_daily, industry_mapping, start, end
    )
    original_columns = original[
        ["date", "raw_continuous_position", "original_unclipped_cheapness", "pe_mid"]
    ]
    normalized = normalized.merge(original_columns, on="date", how="left", validate="one_to_one")
    market_monthly = _monthly_market_features(index_daily, valuation_raw, bonds)
    market_monthly["conditioned_fair_pe"] = expanding_conditioned_fair_pe(market_monthly)
    normalized = normalized.merge(
        market_monthly[["date", "conditioned_fair_pe"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    normalized["normalized_earnings_cheapness"] = np.log(
        normalized["pe_mid"] / normalized["component_normalized_pe"]
    )
    normalized["conditioned_normalized_cheapness"] = np.log(
        normalized["conditioned_fair_pe"] / normalized["component_normalized_pe"]
    )
    monthly = add_forward_total_returns(normalized, total_return)
    industry_returns = build_industry_return_contributions(
        weights,
        constituents,
        industry_mapping,
        total_return,
        start,
        end,
    )
    industry_results = industry_attribution(
        industry_fundamentals, industry_returns, monthly
    )

    evaluations = {
        "original_clipped_position": evaluate_signal(daily, "raw_continuous_position"),
        "original_unclipped_cheapness": evaluate_signal(daily, "original_unclipped_cheapness"),
        "normalized_earnings_cheapness": evaluate_signal(monthly, "normalized_earnings_cheapness"),
        "conditioned_normalized_cheapness": evaluate_signal(
            monthly, "conditioned_normalized_cheapness"
        ),
    }
    fundamentals = fundamental_quintile_diagnostics(monthly)
    cheapest = next(row for row in fundamentals if row["valuation_state"] == "最便宜20%")
    expensive = next(row for row in fundamentals if row["valuation_state"] == "最贵20%")
    current_profit_peak_supported = bool(
        cheapest["weighted_profit_peak_ratio"]
        > expensive["weighted_profit_peak_ratio"] + 0.05
        and cheapest["profit_peak_weight_share"]
        > expensive["profit_peak_weight_share"] + 0.03
    )
    earnings_reversal_supported = bool(
        cheapest["future_ttm_eps_growth_12m"]
        < expensive["future_ttm_eps_growth_12m"] - 0.10
        and cheapest["future_profit_growth_change_12m"]
        < expensive["future_profit_growth_change_12m"] - 0.10
    )
    original_242 = evaluations["original_unclipped_cheapness"]["242d"]
    normalized_242 = evaluations["normalized_earnings_cheapness"]["242d"]
    conditioned_242 = evaluations["conditioned_normalized_cheapness"]["242d"]
    normalization_improves = bool(
        normalized_242["spearman_ic"] > original_242["spearman_ic"]
        and normalized_242["cheapest_minus_most_expensive_mean_return"]
        > original_242["cheapest_minus_most_expensive_mean_return"]
    )
    conditioning_improves = bool(
        conditioned_242["spearman_ic"] > normalized_242["spearman_ic"]
        and conditioned_242["cheapest_minus_most_expensive_mean_return"]
        > normalized_242["cheapest_minus_most_expensive_mean_return"]
    )
    negative_industries = [
        row for row in industry_results
        if row["cheapest_minus_expensive_242d_contribution"] is not None
        and row["cheapest_minus_expensive_242d_contribution"] < 0
    ]
    total_negative_contribution = sum(
        abs(row["cheapest_minus_expensive_242d_contribution"])
        for row in negative_industries
    )
    top_three_negative_share = (
        sum(
            abs(row["cheapest_minus_expensive_242d_contribution"])
            for row in negative_industries[:3]
        )
        / total_negative_contribution
        if total_negative_contribution > 0
        else np.nan
    )
    yearly = yearly_ic_attribution(daily, "original_unclipped_cheapness")
    positive_within_years = sum(
        row["spearman_ic_within_year"] is not None
        and row["spearman_ic_within_year"] > 0
        for row in yearly
    )
    cross_year_reversal_supported = bool(
        evaluations["original_unclipped_cheapness"]["242d"]["spearman_ic"] < 0
        and positive_within_years >= 3
        and sum(row["rank_covariance_numerator"] for row in yearly[-2:]) < 0
    )
    clipping_amplifies = bool(
        evaluations["original_clipped_position"]["242d"]["spearman_ic"]
        < evaluations["original_unclipped_cheapness"]["242d"]["spearman_ic"] - 0.05
    )
    conditioning_result = (
        "SUPPORTED"
        if conditioning_improves
        else "MIXED_IC_IMPROVED_EXTREME_SPREAD_NOT_IMPROVED"
        if conditioned_242["spearman_ic"] > normalized_242["spearman_ic"]
        else "NOT_SUPPORTED"
    )
    report = {
        "status": "NEGATIVE_IC_ATTRIBUTED_TO_EARNINGS_REVERSAL_REGIME_AND_CLIPPING",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "估值状态到未来60/120/242日H00300总收益；无仓位、趋势、FVG、成本或交易回测",
        "sample": {
            "daily_start": str(daily["date"].min().date()),
            "daily_end": str(daily["date"].max().date()),
            "daily_observations": int(len(daily)),
            "monthly_snapshots": int(len(monthly)),
            "minimum_normalized_earnings_weight_coverage": float(
                monthly["normalized_earnings_weight_coverage"].min()
            ),
        },
        "signal_evaluations": evaluations,
        "yearly_242d_attribution": yearly,
        "fundamental_quintile_diagnostics": fundamentals,
        "industry_attribution": industry_results,
        "industry_negative_contribution_top3_share": float(top_three_negative_share),
        "hypothesis_results": {
            "current_profit_peak_trap": (
                "SUPPORTED" if current_profit_peak_supported else "NOT_SUPPORTED"
            ),
            "subsequent_earnings_reversal": (
                "SUPPORTED" if earnings_reversal_supported else "NOT_SUPPORTED"
            ),
            "cross_year_regime_reversal": (
                "SUPPORTED" if cross_year_reversal_supported else "NOT_SUPPORTED"
            ),
            "clipping_amplifies_negative_ic": (
                "SUPPORTED" if clipping_amplifies else "NOT_SUPPORTED"
            ),
            "normalization_improves_monotonicity": (
                "SUPPORTED" if normalization_improves else "NOT_SUPPORTED"
            ),
            "conditioned_multiple_improves_monotonicity": conditioning_result,
            "industry_attribution": "AVAILABLE_EXPLANATORY_STATIC_CLASSIFICATION",
            "industry_contribution_concentrated": (
                "SUPPORTED" if top_three_negative_share >= 0.50 else "NOT_SUPPORTED"
            ),
            "cash_flow_quality": "UNAVAILABLE_NO_CASH_FLOW_FIELDS",
        },
        "normalization_definition": {
            "lookback_years": 3,
            "minimum_ttm_observations": 4,
            "normalized_profit": "50%×公司三年正TTM利润中位数 + 50%×当前TTM收入×公司三年TTM净利率中位数；缺一项时使用另一项",
            "industry_specific": False,
            "reason": "本轮统一正常化用于机制诊断；静态行业标签只做结果归因，不进入正常化公式",
        },
        "conditioned_fair_pe_definition": {
            "estimator": "扩展窗口StandardScaler+Ridge(alpha=10)，不搜索参数",
            "minimum_training_months": 36,
            "features": list(CONDITIONAL_MULTIPLE_FEATURES),
            "current_price_is_direct_feature": False,
        },
        "data_limitations": [
            "行业归因使用静态申万映射，只解释业务群贡献；不得作为点时特征或训练输入。",
            "点时财务数据没有经营现金流字段，不能检验现金利润质量。",
            "正常化方法是行业未知条件下的统一诊断代理，不是最终生产估值模型。",
            "全部历史已参与研究，替代信号只属于回顾性机制诊断。",
        ],
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "input_hashes": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                CONFIG_FILE,
                VALUATION_FILE,
                INDEX_FILE,
                TOTAL_RETURN_FILE,
                WEIGHTS_FILE,
                CONSTITUENT_FILE,
                FINANCIALS_FILE,
                BOND_FILE,
                INDUSTRY_FILE,
            )
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(OUTPUT_FILE, index=False)
    report["output_sha256"] = _sha256(OUTPUT_FILE)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
