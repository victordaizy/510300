"""沪深300估值V2：直接估计预期超额总收益，而非未来价格。

信号由股息、正常化盈利增长和部分估值回归三部分组成。全部信号输入在
信号日可得；未来总收益只用于事后评价，不参与参数拟合或信号构造。
"""

from __future__ import annotations

import hashlib
import json
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

from research.build_point_in_time_fundamental_panel import (
    _ttm_at_period,
    select_latest_vintages,
)
from research.diagnose_valuation_negative_ic import (
    BOND_FILE,
    CONSTITUENT_FILE,
    FINANCIALS_FILE,
    INDEX_FILE,
    INDUSTRY_FILE,
    TOTAL_RETURN_FILE,
    VALUATION_FILE,
    WEIGHTS_FILE,
    _safe_ratio,
    _weighted_mean,
    add_forward_total_returns,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "valuation_v2_expected_return.yaml"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_valuation_v2_expected_return.parquet"
REPORT_JSON = ROOT / "reports" / "research" / "000300_valuation_v2_expected_return.json"
REPORT_MD = ROOT / "reports" / "research" / "000300_valuation_v2_expected_return.md"
DIAGNOSTIC_FILE = ROOT / "data" / "features" / "000300_valuation_negative_ic_diagnostics.parquet"

HORIZONS = (60, 120, 242)
EDGE_QUINTILES = ("最低Edge20%", "次低", "中性", "次高", "最高Edge20%")
FAIR_PE_FEATURES = (
    "cgb_10y",
    "cgb_term_spread",
    "quality_roe",
    "growth_expectation",
    "realized_volatility_3m",
)

FINANCIAL_INDUSTRIES = frozenset({"银行", "非银金融"})
CYCLICAL_INDUSTRIES = frozenset(
    {
        "煤炭",
        "钢铁",
        "有色金属",
        "基础化工",
        "石油石化",
        "建筑材料",
        "建筑装饰",
        "机械设备",
        "电力设备",
        "国防军工",
        "交通运输",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalization_method(industry_l1: str | None) -> str:
    """把申万一级行业映射到固定的正常化方法，不从结果反向调参。"""

    industry = str(industry_l1) if pd.notna(industry_l1) else "未识别"
    if industry in FINANCIAL_INDUSTRIES:
        return "FINANCIAL_ROE"
    if industry in CYCLICAL_INDUSTRIES:
        return "CYCLICAL_MARGIN"
    return "OTHER_BLEND"


def _latest_value(value_map: dict[pd.Timestamp, float], period: pd.Timestamp) -> float:
    eligible = [key for key, value in value_map.items() if key <= period and pd.notna(value)]
    return float(value_map[max(eligible)]) if eligible else np.nan


def _median_historical_growth(
    value_map: dict[pd.Timestamp, float],
    periods: list[pd.Timestamp],
) -> float:
    growth: list[float] = []
    for period in periods:
        current = _ttm_at_period(value_map, period)
        prior = _ttm_at_period(value_map, period - pd.DateOffset(years=1))
        if pd.notna(current) and pd.notna(prior) and prior > 0:
            growth.append(float(current / prior - 1.0))
    return float(np.median(growth)) if growth else np.nan


def derive_industry_aware_normalization(
    latest_vintages: pd.DataFrame,
    industry_mapping: pd.DataFrame,
    lookback_years: int = 3,
    minimum_observations: int = 4,
) -> pd.DataFrame:
    """按行业采用固定方法计算公司正常化盈利与预期增长。

    银行和非银使用历史ROE与当前权益；周期行业使用历史利润率与当前收入；
    其他行业使用历史利润和历史利润率两种锚的等权。所有历史财务必须已在
    信号日披露，行业标签不参与任何数据驱动估计。
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
    mapping = (
        industry_mapping[["con_code", "industry_l1"]]
        .drop_duplicates("con_code", keep="last")
        .set_index("con_code")["industry_l1"]
        .to_dict()
    )
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
            if not history_start <= period <= selected_period:
                continue
            profit = _ttm_at_period(maps["net_profit_parent_cny"], period)
            revenue = _ttm_at_period(maps["revenue_cny"], period)
            equity = _latest_value(maps["equity_parent_cny"], period)
            if pd.notna(profit) and pd.notna(revenue) and revenue > 0:
                history.append((period, float(profit), float(revenue), float(equity)))
        if len(history) < minimum_observations:
            continue

        periods = [item[0] for item in history]
        profits = pd.Series([item[1] for item in history], dtype=float)
        margins = pd.Series([item[1] / item[2] for item in history], dtype=float)
        roes = pd.Series(
            [item[1] / item[3] if item[3] > 0 else np.nan for item in history],
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan)
        positive_profits = profits[profits > 0]
        median_profit = (
            float(positive_profits.median()) if not positive_profits.empty else np.nan
        )
        median_margin = float(margins.median()) if margins.notna().any() else np.nan
        median_roe = float(roes.median()) if roes.notna().any() else np.nan
        current_equity = _latest_value(maps["equity_parent_cny"], selected_period)
        shares = _latest_value(maps["total_shares"], selected_period)
        margin_anchor = (
            current_revenue * median_margin
            if pd.notna(median_margin) and median_margin > 0
            else np.nan
        )
        roe_anchor = (
            current_equity * median_roe
            if pd.notna(current_equity)
            and current_equity > 0
            and pd.notna(median_roe)
            and median_roe > 0
            else np.nan
        )
        blend_candidates = [
            value for value in (median_profit, margin_anchor) if pd.notna(value) and value > 0
        ]
        blend_anchor = float(np.mean(blend_candidates)) if blend_candidates else np.nan

        industry = str(mapping.get(str(con_code), "未识别"))
        method = normalization_method(industry)
        primary = {
            "FINANCIAL_ROE": roe_anchor,
            "CYCLICAL_MARGIN": margin_anchor,
            "OTHER_BLEND": blend_anchor,
        }[method]
        normalized_profit = (
            float(primary)
            if pd.notna(primary) and primary > 0
            else float(blend_anchor)
            if pd.notna(blend_anchor) and blend_anchor > 0
            else np.nan
        )

        current_growth = _safe_ratio(
            current_revenue,
            _ttm_at_period(
                maps["revenue_cny"], selected_period - pd.DateOffset(years=1)
            ),
        )
        current_growth = current_growth - 1.0 if pd.notna(current_growth) else np.nan
        historical_growth = _median_historical_growth(
            maps["revenue_cny"], periods
        )
        current_weight = 0.25 if method in {"FINANCIAL_ROE", "CYCLICAL_MARGIN"} else 0.50
        growth_candidates: list[tuple[float, float]] = []
        if pd.notna(current_growth):
            growth_candidates.append((float(current_growth), current_weight))
        if pd.notna(historical_growth):
            growth_candidates.append((float(historical_growth), 1.0 - current_weight))
        expected_growth = (
            float(np.average(
                [item[0] for item in growth_candidates],
                weights=[item[1] for item in growth_candidates],
            ))
            if growth_candidates
            else np.nan
        )
        expected_growth = float(np.clip(expected_growth, -0.10, 0.15))
        rows.append(
            {
                "con_code": str(con_code),
                "industry_l1": industry,
                "normalization_method": method,
                "latest_report_period": selected_period,
                "normalized_profit_cny": normalized_profit,
                "normalized_eps": _safe_ratio(normalized_profit, shares),
                "normalized_roe": _safe_ratio(normalized_profit, current_equity),
                "ttm_eps": _safe_ratio(current_profit, shares),
                "ttm_profit_cny": current_profit,
                "profit_peak_ratio": _safe_ratio(current_profit, normalized_profit),
                "current_revenue_growth_yoy": current_growth,
                "historical_revenue_growth_median": historical_growth,
                "expected_earnings_growth_annual": expected_growth,
                "normalization_observations": len(history),
            }
        )
    return pd.DataFrame(rows)


def aggregate_v2_snapshot(
    snapshot_weights: pd.DataFrame,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    index_close: float,
) -> dict[str, Any]:
    """把公司正常化结果按当月官方权重聚合到指数层。"""

    panel = snapshot_weights[["con_code", "weight"]].copy()
    panel["weight"] = pd.to_numeric(panel["weight"], errors="coerce") / 100.0
    panel = panel.merge(
        prices[["con_code", "raw_close"]], on="con_code", how="left", validate="one_to_one"
    )
    panel = panel.merge(metrics, on="con_code", how="left", validate="one_to_one")
    panel["normalized_earnings_yield"] = panel["normalized_eps"] / panel["raw_close"]
    panel["ttm_earnings_yield"] = panel["ttm_eps"] / panel["raw_close"]
    normalized_yield, normalized_coverage = _weighted_mean(
        panel["normalized_earnings_yield"], panel["weight"]
    )
    ttm_yield, ttm_coverage = _weighted_mean(panel["ttm_earnings_yield"], panel["weight"])
    normalized_roe, roe_coverage = _weighted_mean(
        panel["normalized_roe"].clip(-0.5, 0.8), panel["weight"]
    )
    expected_growth, growth_coverage = _weighted_mean(
        panel["expected_earnings_growth_annual"], panel["weight"]
    )
    peak_ratio, _ = _weighted_mean(
        panel["profit_peak_ratio"].clip(-5.0, 5.0), panel["weight"]
    )
    method_weights = (
        panel.dropna(subset=["normalization_method"])
        .groupby("normalization_method", sort=False)["weight"]
        .sum()
        .to_dict()
    )
    return {
        "date": pd.Timestamp(snapshot_date),
        "constituent_count": int(panel["con_code"].nunique()),
        "normalization_company_count": int(panel["normalized_eps"].notna().sum()),
        "normalized_earnings_weight_coverage": normalized_coverage,
        "ttm_earnings_weight_coverage": ttm_coverage,
        "normalized_roe_weight_coverage": roe_coverage,
        "expected_growth_weight_coverage": growth_coverage,
        "weighted_normalized_earnings_yield": normalized_yield,
        "weighted_ttm_earnings_yield": ttm_yield,
        "component_normalized_pe": (
            1.0 / normalized_yield
            if pd.notna(normalized_yield) and normalized_yield > 0
            else np.nan
        ),
        "component_ttm_pe": (
            1.0 / ttm_yield if pd.notna(ttm_yield) and ttm_yield > 0 else np.nan
        ),
        "component_normalized_index_eps": index_close * normalized_yield,
        "weighted_normalized_roe": normalized_roe,
        "expected_earnings_growth_annual": expected_growth,
        "weighted_profit_peak_ratio": peak_ratio,
        "financial_method_weight": float(method_weights.get("FINANCIAL_ROE", 0.0)),
        "cyclical_method_weight": float(method_weights.get("CYCLICAL_MARGIN", 0.0)),
        "other_method_weight": float(method_weights.get("OTHER_BLEND", 0.0)),
    }


def build_v2_monthly_fundamentals(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    industry_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """使用每个官方月末权重快照构造点时V2基本面。"""

    weights = weights.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    financials = financials.copy()
    financials["report_period"] = pd.to_datetime(financials["report_period"])
    financials["available_at"] = pd.to_datetime(financials["available_at"])
    constituent_daily = constituent_daily.copy()
    constituent_daily["date"] = pd.to_datetime(constituent_daily["date"])
    available_price_dates = set(constituent_daily["date"].drop_duplicates())
    weights = weights.loc[weights["trade_date"].isin(available_price_dates)].copy()
    index_daily = index_daily.copy()
    index_daily["date"] = pd.to_datetime(index_daily["date"])
    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weights.groupby("trade_date", sort=True):
        signal_date = pd.Timestamp(snapshot_date)
        symbols = set(snapshot_weights["con_code"].astype(str))
        vintages = select_latest_vintages(
            financials.loc[financials["con_code"].astype(str).isin(symbols)],
            signal_date,
        )
        metrics = derive_industry_aware_normalization(vintages, industry_mapping)
        prices = constituent_daily.loc[
            constituent_daily["date"].eq(signal_date), ["con_code", "raw_close"]
        ].drop_duplicates("con_code", keep="last")
        index_row = index_daily.loc[index_daily["date"].eq(signal_date), "close"]
        if index_row.empty:
            continue
        rows.append(
            aggregate_v2_snapshot(
                snapshot_weights,
                prices,
                metrics,
                signal_date,
                float(index_row.iloc[-1]),
            )
        )
    if not rows:
        raise ValueError("没有生成V2正常化盈利快照")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_market_monthly(
    index_daily: pd.DataFrame,
    total_return_daily: pd.DataFrame,
    valuation: pd.DataFrame,
    bonds: pd.DataFrame,
) -> pd.DataFrame:
    """构造长历史月频宏观特征及只使用过去数据的股息率代理。"""

    index = index_daily[["date", "close"]].copy().rename(columns={"close": "index_close"})
    total = total_return_daily[["date", "close"]].copy().rename(
        columns={"close": "total_return_close"}
    )
    value = valuation[["date", "pe_ttm", "pb"]].copy()
    bond = bonds[["date", "cgb_1y", "cgb_10y"]].copy()
    for frame in (index, total, value, bond):
        frame["date"] = pd.to_datetime(frame["date"])
    daily = index.merge(total, on="date", how="left", validate="one_to_one")
    daily = daily.merge(value, on="date", how="left", validate="one_to_one")
    daily = daily.merge(bond, on="date", how="left", validate="one_to_one")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily[["total_return_close", "pe_ttm", "pb", "cgb_1y", "cgb_10y"]] = daily[
        ["total_return_close", "pe_ttm", "pb", "cgb_1y", "cgb_10y"]
    ].ffill(limit=5)
    daily["log_return_1d"] = np.log(daily["index_close"] / daily["index_close"].shift(1))
    daily["realized_volatility_3m"] = (
        daily["log_return_1d"].rolling(63, min_periods=40).std(ddof=1) * np.sqrt(242)
    )
    total_growth = daily["total_return_close"] / daily["total_return_close"].shift(242)
    price_growth = daily["index_close"] / daily["index_close"].shift(242)
    daily["trailing_dividend_yield_annual"] = (total_growth / price_growth - 1.0).clip(
        0.0, 0.10
    )
    daily["implied_eps"] = daily["index_close"] / daily["pe_ttm"]
    daily["implied_roe"] = daily["pb"] / daily["pe_ttm"]
    daily["calendar_month"] = daily["date"].dt.to_period("M")
    monthly = daily.groupby("calendar_month", sort=True).tail(1).copy().reset_index(drop=True)
    monthly["eps_growth_12m"] = monthly["implied_eps"].pct_change(
        12, fill_method=None
    ).clip(-0.5, 0.5)
    monthly["cgb_term_spread"] = monthly["cgb_10y"] - monthly["cgb_1y"]
    monthly = monthly.replace([np.inf, -np.inf], np.nan)
    return monthly


def _fair_pe_model(alpha: float = 10.0) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    )


def expanding_conditioned_fair_pe_v2(
    market_monthly: pd.DataFrame,
    fundamentals: pd.DataFrame,
    minimum_training_samples: int = 48,
    ridge_alpha: float = 10.0,
) -> pd.DataFrame:
    """严格扩展窗口估计FairPE；当前价格和当前PE都不进入评分特征。"""

    historical = market_monthly.copy().sort_values("date").reset_index(drop=True)
    historical["quality_roe"] = historical["implied_roe"]
    historical["growth_expectation"] = historical["eps_growth_12m"]
    historical["target_log_pe"] = np.log(historical["pe_ttm"])
    rows: list[dict[str, Any]] = []
    market_by_date = historical.set_index("date")
    for signal in fundamentals.sort_values("date").itertuples(index=False):
        signal_date = pd.Timestamp(signal.date)
        if signal_date not in market_by_date.index:
            continue
        current = market_by_date.loc[signal_date]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[-1]
        train = historical.loc[historical["date"].lt(signal_date)].dropna(
            subset=[*FAIR_PE_FEATURES, "target_log_pe"]
        )
        if len(train) < minimum_training_samples:
            continue
        score = pd.DataFrame(
            [
                {
                    "cgb_10y": current["cgb_10y"],
                    "cgb_term_spread": current["cgb_term_spread"],
                    "quality_roe": signal.weighted_normalized_roe,
                    "growth_expectation": signal.expected_earnings_growth_annual,
                    "realized_volatility_3m": current["realized_volatility_3m"],
                }
            ]
        )
        if score[list(FAIR_PE_FEATURES)].isna().any(axis=None):
            continue
        model = _fair_pe_model(ridge_alpha).fit(
            train[list(FAIR_PE_FEATURES)], train["target_log_pe"]
        )
        raw_log_pe = float(model.predict(score[list(FAIR_PE_FEATURES)])[0])
        lower, upper = train["target_log_pe"].quantile([0.10, 0.90])
        predicted_log_pe = float(np.clip(raw_log_pe, lower, upper))
        rows.append(
            {
                "date": signal_date,
                "conditioned_fair_pe_v2": float(np.exp(predicted_log_pe)),
                "raw_conditioned_fair_pe_v2": float(np.exp(raw_log_pe)),
                "fair_pe_training_samples": int(len(train)),
                "fair_pe_training_last_date": pd.Timestamp(train["date"].max()),
                "fair_pe_lower_bound": float(np.exp(lower)),
                "fair_pe_upper_bound": float(np.exp(upper)),
            }
        )
    return pd.DataFrame(rows)


def horizon_return_components(
    annual_dividend_yield: float,
    annual_earnings_growth: float,
    fair_pe: float,
    current_normalized_pe: float,
    annual_cash_rate: float,
    horizon_days: int,
    trading_days_per_year: int = 242,
) -> dict[str, float]:
    """按可乘关系分解指定期限的预期总收益与超额收益。"""

    fraction = horizon_days / trading_days_per_year
    dividend = (1.0 + annual_dividend_yield) ** fraction - 1.0
    growth = (1.0 + annual_earnings_growth) ** fraction - 1.0
    convergence = 1.0 - 2.0 ** (-fraction)
    rerating_log = convergence * np.log(fair_pe / current_normalized_pe)
    rerating = float(np.exp(rerating_log) - 1.0)
    expected_total = (1.0 + dividend) * (1.0 + growth) * (1.0 + rerating) - 1.0
    cash = (1.0 + annual_cash_rate) ** fraction - 1.0
    return {
        "expected_dividend_return": float(dividend),
        "expected_earnings_growth_return": float(growth),
        "expected_rerating_return": rerating,
        "rerating_convergence": float(convergence),
        "expected_total_return": float(expected_total),
        "cash_return": float(cash),
        "expected_excess_total_return": float(expected_total - cash),
    }


def build_expected_return_panel(
    fundamentals: pd.DataFrame,
    fair_pe: pd.DataFrame,
    market_monthly: pd.DataFrame,
    total_return_daily: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """生成V2信号面板；未来收益列只作为标签追加。"""

    market_columns = [
        "date",
        "index_close",
        "pe_ttm",
        "pb",
        "cgb_1y",
        "cgb_10y",
        "cgb_term_spread",
        "realized_volatility_3m",
        "trailing_dividend_yield_annual",
    ]
    panel = fundamentals.merge(fair_pe, on="date", how="inner", validate="one_to_one")
    panel = panel.merge(
        market_monthly[market_columns], on="date", how="left", validate="one_to_one"
    )
    panel["annual_cash_rate"] = (panel["cgb_1y"] / 100.0).clip(lower=0.0)
    panel["fair_value_index_v2"] = (
        panel["component_normalized_index_eps"] * panel["conditioned_fair_pe_v2"]
    )
    panel["full_rerating_space"] = (
        panel["conditioned_fair_pe_v2"] / panel["component_normalized_pe"] - 1.0
    )
    for horizon in horizons:
        components = panel.apply(
            lambda row: horizon_return_components(
                float(row["trailing_dividend_yield_annual"]),
                float(row["expected_earnings_growth_annual"]),
                float(row["conditioned_fair_pe_v2"]),
                float(row["component_normalized_pe"]),
                float(row["annual_cash_rate"]),
                horizon,
            ),
            axis=1,
            result_type="expand",
        )
        for column in components:
            panel[f"{column}_{horizon}d"] = components[column]
    panel = add_forward_total_returns(panel, total_return_daily, horizons)
    for horizon in horizons:
        panel[f"realized_excess_total_return_{horizon}d"] = (
            panel[f"forward_total_return_{horizon}d"] - panel[f"cash_return_{horizon}d"]
        )
    return panel.sort_values("date").reset_index(drop=True)


def assign_edge_quintiles(signal: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=signal.index, dtype="object")
    valid = signal.dropna()
    if len(valid) < 5:
        return result
    ranks = valid.rank(method="first")
    result.loc[valid.index] = pd.qcut(ranks, 5, labels=EDGE_QUINTILES).astype(str)
    return result


def _non_overlapping_ics(
    sample: pd.DataFrame,
    signal_column: str,
    target_column: str,
    horizon_days: int,
) -> list[float]:
    spacing_months = max(1, int(round(horizon_days / 21.0)))
    values: list[float] = []
    for offset in range(spacing_months):
        subset = sample.iloc[offset::spacing_months]
        if (
            len(subset) >= 4
            and subset[signal_column].nunique() > 1
            and subset[target_column].nunique() > 1
        ):
            values.append(
                float(spearmanr(subset[signal_column], subset[target_column]).statistic)
            )
    return values


def evaluate_expected_edge(
    panel: pd.DataFrame,
    horizon_days: int,
) -> dict[str, Any]:
    """在相同的已兑现样本内评价IC、五组单调性与校准误差。"""

    signal_column = f"expected_excess_total_return_{horizon_days}d"
    target_column = f"realized_excess_total_return_{horizon_days}d"
    sample = panel[["date", signal_column, target_column]].dropna().copy()
    if len(sample) < 5:
        return {"horizon_days": horizon_days, "observations": int(len(sample))}
    sample["quintile"] = assign_edge_quintiles(sample[signal_column])
    grouped = (
        sample.groupby("quintile", observed=False)[target_column]
        .agg(["count", "mean", "median"])
        .reindex(EDGE_QUINTILES)
    )
    non_overlapping = _non_overlapping_ics(
        sample, signal_column, target_column, horizon_days
    )
    expected_positive = sample[signal_column].gt(0)
    realized_positive = sample[target_column].gt(0)
    calibration_error = sample[signal_column] - sample[target_column]
    quintile_means = grouped["mean"].to_numpy(dtype=float)
    return {
        "horizon_days": horizon_days,
        "observations": int(len(sample)),
        "first_signal_date": str(sample["date"].min().date()),
        "last_realized_signal_date": str(sample["date"].max().date()),
        "spearman_ic": float(
            spearmanr(sample[signal_column], sample[target_column]).statistic
        ),
        "pearson_correlation": float(sample[signal_column].corr(sample[target_column])),
        "non_overlapping_ic_median": (
            float(np.median(non_overlapping)) if non_overlapping else None
        ),
        "non_overlapping_positive_ratio": (
            float(np.mean(np.asarray(non_overlapping) > 0)) if non_overlapping else None
        ),
        "mean_absolute_calibration_error": float(calibration_error.abs().mean()),
        "expected_positive_precision": (
            float(realized_positive.loc[expected_positive].mean())
            if expected_positive.any()
            else None
        ),
        "quintiles": [
            {
                "edge_state": label,
                "observations": int(grouped.loc[label, "count"]),
                "mean_realized_excess_return": float(grouped.loc[label, "mean"]),
                "median_realized_excess_return": float(grouped.loc[label, "median"]),
            }
            for label in EDGE_QUINTILES
        ],
        "highest_minus_lowest_mean_realized_excess_return": float(
            grouped.loc["最高Edge20%", "mean"] - grouped.loc["最低Edge20%", "mean"]
        ),
        "strictly_monotonic_quintile_means": bool(
            np.all(np.diff(quintile_means) > 0)
        ),
    }


def evaluate_reference_signal(
    panel: pd.DataFrame,
    signal_column: str,
    horizon_days: int,
) -> dict[str, Any]:
    target_column = f"realized_excess_total_return_{horizon_days}d"
    sample = panel[[signal_column, target_column]].dropna()
    if len(sample) < 4:
        return {"observations": int(len(sample)), "spearman_ic": None}
    return {
        "observations": int(len(sample)),
        "spearman_ic": float(
            spearmanr(sample[signal_column], sample[target_column]).statistic
        ),
    }


def chronological_stability(
    panel: pd.DataFrame,
    horizon_days: int = 242,
) -> list[dict[str, Any]]:
    signal = f"expected_excess_total_return_{horizon_days}d"
    target = f"realized_excess_total_return_{horizon_days}d"
    sample = panel[["date", signal, target]].dropna().sort_values("date").reset_index(drop=True)
    if sample.empty:
        return []
    split = len(sample) // 2
    rows: list[dict[str, Any]] = []
    for label, subset in (("前半段", sample.iloc[:split]), ("后半段", sample.iloc[split:])):
        rows.append(
            {
                "period": label,
                "observations": int(len(subset)),
                "start": str(subset["date"].min().date()),
                "end": str(subset["date"].max().date()),
                "spearman_ic": float(spearmanr(subset[signal], subset[target]).statistic),
                "mean_expected_edge": float(subset[signal].mean()),
                "mean_realized_edge": float(subset[target].mean()),
            }
        )
    return rows


def component_ablation(
    panel: pd.DataFrame,
    horizon_days: int,
) -> list[dict[str, Any]]:
    """诊断每一段预期收益对实际超额收益的排序贡献。"""

    target = f"realized_excess_total_return_{horizon_days}d"
    columns = {
        "股息": f"expected_dividend_return_{horizon_days}d",
        "正常化盈利增长": f"expected_earnings_growth_return_{horizon_days}d",
        "估值重估": f"expected_rerating_return_{horizon_days}d",
        "预期总收益": f"expected_total_return_{horizon_days}d",
        "现金收益": f"cash_return_{horizon_days}d",
        "预期超额总收益": f"expected_excess_total_return_{horizon_days}d",
    }
    rows: list[dict[str, Any]] = []
    for label, column in columns.items():
        sample = panel[[column, target]].dropna()
        rows.append(
            {
                "component": label,
                "observations": int(len(sample)),
                "spearman_ic_vs_realized_edge": float(
                    spearmanr(sample[column], sample[target]).statistic
                ),
            }
        )
    return rows


def growth_forecast_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    """单独检验盈利增长估计是否预测未来正常化盈利，而非拿收益作标签。"""

    data = panel.sort_values("date").copy()
    data["future_normalized_eps_growth_12m"] = (
        data["component_normalized_index_eps"].shift(-12)
        / data["component_normalized_index_eps"]
        - 1.0
    )
    sample = data[
        ["expected_earnings_growth_annual", "future_normalized_eps_growth_12m"]
    ].dropna()
    return {
        "observations": int(len(sample)),
        "spearman_ic": float(
            spearmanr(
                sample["expected_earnings_growth_annual"],
                sample["future_normalized_eps_growth_12m"],
            ).statistic
        ),
        "pearson_correlation": float(
            sample["expected_earnings_growth_annual"].corr(
                sample["future_normalized_eps_growth_12m"]
            )
        ),
        "interpretation": "增长项应首先预测未来正常化盈利；其与股票收益可因估值已反映增长而反向。",
    }


def _pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 沪深300估值V2：预期超额总收益",
        "",
        "> 本研究直接估计股息、正常化盈利增长、估值重估与现金回报，不预测未来价格，也不生成仓位。",
        "",
        "## 结论",
        "",
        f"- 研究状态：`{report['status']}`。",
        f"- 最低正常化盈利权重覆盖：{_pct(report['sample']['minimum_normalized_earnings_weight_coverage'])}。",
        "- FairPE使用严格扩展窗口；当期价格和当期PE不进入评分特征。",
        "- 行业标签只路由预先固定的盈利正常化方法，不用行业未来收益拟合参数。",
        f"- 242日后半段IC：{report['chronological_stability_242d'][1]['spearman_ic']:.3f}；时间稳定性门槛：`{report['acceptance']['positive_ic_both_chronological_halves_242d']}`。",
        "- 全部历史已被研究，结果只属于回顾性机制证据，不授权仓位或交易。",
        "",
        "## V2预期Edge验证",
        "",
        "|期限|样本|IC|非重叠IC中位数|最高减最低Edge组实际超额|平均绝对校准误差|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["evaluations"]:
        lines.append(
            f"|{item['horizon_days']}日|{item['observations']}|"
            f"{item.get('spearman_ic', float('nan')):.3f}|"
            f"{item.get('non_overlapping_ic_median', float('nan')):.3f}|"
            f"{_pct(item.get('highest_minus_lowest_mean_realized_excess_return'))}|"
            f"{_pct(item.get('mean_absolute_calibration_error'))}|"
        )
    lines.extend(
        [
            "",
            "## 与估值诊断信号的同样本IC对照",
            "",
            "|期限|原统一正常化便宜度|原条件正常化便宜度|V2预期Edge|",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in report["reference_comparison"]:
        lines.append(
            f"|{row['horizon_days']}日|{row['normalized_earnings_cheapness_ic']:.3f}|"
            f"{row['conditioned_normalized_cheapness_ic']:.3f}|{row['v2_expected_edge_ic']:.3f}|"
        )
    lines.extend(
        [
            "",
            "## 收益组件消融",
            "",
            "|期限|组件|对实际Edge的IC|",
            "|---:|---|---:|",
        ]
    )
    for horizon in report["component_ablation"]:
        for row in horizon["components"]:
            lines.append(
                f"|{horizon['horizon_days']}日|{row['component']}|"
                f"{row['spearman_ic_vs_realized_edge']:.3f}|"
            )
    lines.extend(
        [
            "",
            "## 242日Edge五组",
            "",
            "|Edge状态|样本|平均实际超额收益|中位实际超额收益|",
            "|---|---:|---:|---:|",
        ]
    )
    evaluation_242 = next(
        item for item in report["evaluations"] if item["horizon_days"] == 242
    )
    for row in evaluation_242["quintiles"]:
        lines.append(
            f"|{row['edge_state']}|{row['observations']}|"
            f"{_pct(row['mean_realized_excess_return'])}|"
            f"{_pct(row['median_realized_excess_return'])}|"
        )
    lines.extend(
        [
            "",
            "## 最新信号分解",
            "",
            f"- 信号日：{report['latest_signal']['date']}。",
            f"- 正常化PE：{report['latest_signal']['component_normalized_pe']:.2f}；条件FairPE：{report['latest_signal']['conditioned_fair_pe_v2']:.2f}。",
            f"- 年化股息代理：{_pct(report['latest_signal']['trailing_dividend_yield_annual'])}；预期盈利增长：{_pct(report['latest_signal']['expected_earnings_growth_annual'])}。",
            "",
            "|期限|股息|盈利增长|估值重估|预期总收益|现金收益|预期Edge|",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["latest_signal"]["horizons"]:
        lines.append(
            f"|{row['horizon_days']}日|{_pct(row['expected_dividend_return'])}|"
            f"{_pct(row['expected_earnings_growth_return'])}|"
            f"{_pct(row['expected_rerating_return'])}|{_pct(row['expected_total_return'])}|"
            f"{_pct(row['cash_return'])}|{_pct(row['expected_excess_total_return'])}|"
        )
    lines.extend(
        [
            "",
            "## 正常化方法",
            "",
            "- 银行、非银金融：当前权益 × 三年TTM ROE中位数。",
            "- 周期行业：当前TTM收入 × 三年TTM净利率中位数。",
            "- 其他行业：三年正TTM利润中位数与当前收入 × 三年净利率中位数等权。",
            "- 盈利增长对当前收入增速做行业相关收缩，并固定截断在−10%至+15%。",
            "",
            "## 证据边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _latest_signal_payload(panel: pd.DataFrame) -> dict[str, Any]:
    row = panel.sort_values("date").iloc[-1]
    horizons = []
    for horizon in HORIZONS:
        horizons.append(
            {
                "horizon_days": horizon,
                "expected_dividend_return": float(
                    row[f"expected_dividend_return_{horizon}d"]
                ),
                "expected_earnings_growth_return": float(
                    row[f"expected_earnings_growth_return_{horizon}d"]
                ),
                "expected_rerating_return": float(
                    row[f"expected_rerating_return_{horizon}d"]
                ),
                "expected_total_return": float(row[f"expected_total_return_{horizon}d"]),
                "cash_return": float(row[f"cash_return_{horizon}d"]),
                "expected_excess_total_return": float(
                    row[f"expected_excess_total_return_{horizon}d"]
                ),
            }
        )
    return {
        "date": str(pd.Timestamp(row["date"]).date()),
        "component_normalized_index_eps": float(row["component_normalized_index_eps"]),
        "component_normalized_pe": float(row["component_normalized_pe"]),
        "conditioned_fair_pe_v2": float(row["conditioned_fair_pe_v2"]),
        "fair_value_index_v2": float(row["fair_value_index_v2"]),
        "trailing_dividend_yield_annual": float(row["trailing_dividend_yield_annual"]),
        "expected_earnings_growth_annual": float(row["expected_earnings_growth_annual"]),
        "annual_cash_rate": float(row["annual_cash_rate"]),
        "fair_pe_training_samples": int(row["fair_pe_training_samples"]),
        "horizons": horizons,
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    index_daily = pd.read_parquet(INDEX_FILE)
    total_return = pd.read_parquet(TOTAL_RETURN_FILE)
    valuation = pd.read_parquet(VALUATION_FILE)
    bonds = pd.read_parquet(BOND_FILE)
    fundamentals = build_v2_monthly_fundamentals(
        pd.read_parquet(WEIGHTS_FILE),
        pd.read_parquet(FINANCIALS_FILE),
        pd.read_parquet(CONSTITUENT_FILE),
        index_daily,
        pd.read_parquet(INDUSTRY_FILE),
    )
    market_monthly = build_market_monthly(index_daily, total_return, valuation, bonds)
    fair_pe = expanding_conditioned_fair_pe_v2(market_monthly, fundamentals)
    panel = build_expected_return_panel(
        fundamentals, fair_pe, market_monthly, total_return
    )

    if DIAGNOSTIC_FILE.exists():
        diagnostic = pd.read_parquet(DIAGNOSTIC_FILE)[
            ["date", "normalized_earnings_cheapness", "conditioned_normalized_cheapness"]
        ].copy()
        diagnostic["date"] = pd.to_datetime(diagnostic["date"])
        panel = panel.merge(diagnostic, on="date", how="left", validate="one_to_one")
    else:
        panel["normalized_earnings_cheapness"] = np.nan
        panel["conditioned_normalized_cheapness"] = np.nan

    evaluations = [evaluate_expected_edge(panel, horizon) for horizon in HORIZONS]
    reference_comparison = []
    for evaluation in evaluations:
        horizon = evaluation["horizon_days"]
        reference_comparison.append(
            {
                "horizon_days": horizon,
                "normalized_earnings_cheapness_ic": evaluate_reference_signal(
                    panel, "normalized_earnings_cheapness", horizon
                )["spearman_ic"],
                "conditioned_normalized_cheapness_ic": evaluate_reference_signal(
                    panel, "conditioned_normalized_cheapness", horizon
                )["spearman_ic"],
                "v2_expected_edge_ic": evaluation["spearman_ic"],
            }
        )
    all_positive = all(
        item.get("spearman_ic", -1.0) > 0
        and item.get("highest_minus_lowest_mean_realized_excess_return", -1.0) > 0
        for item in evaluations
    )
    long_horizon = next(item for item in evaluations if item["horizon_days"] == 242)
    stability_supported = (
        long_horizon.get("non_overlapping_ic_median") is not None
        and long_horizon["non_overlapping_ic_median"] > 0
    )
    chronology = chronological_stability(panel)
    chronological_stability_supported = bool(
        len(chronology) == 2 and all(item["spearman_ic"] > 0 for item in chronology)
    )
    monotonic_all_horizons = all(
        item.get("strictly_monotonic_quintile_means", False) for item in evaluations
    )
    report = {
        "status": (
            "RETROSPECTIVE_MECHANISM_SUPPORTED_FORWARD_FREEZE_REQUIRED"
            if all_positive
            and stability_supported
            and chronological_stability_supported
            and monotonic_all_horizons
            else "RETROSPECTIVE_RANKING_POSITIVE_BUT_TIME_UNSTABLE_NOT_READY"
        ),
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "objective": config["model"]["objective"],
        "sample": {
            "start": str(panel["date"].min().date()),
            "end": str(panel["date"].max().date()),
            "monthly_snapshots": int(len(panel)),
            "minimum_normalized_earnings_weight_coverage": float(
                panel["normalized_earnings_weight_coverage"].min()
            ),
            "minimum_expected_growth_weight_coverage": float(
                panel["expected_growth_weight_coverage"].min()
            ),
        },
        "signal_definition": config,
        "evaluations": evaluations,
        "reference_comparison": reference_comparison,
        "component_ablation": [
            {
                "horizon_days": horizon,
                "components": component_ablation(panel, horizon),
            }
            for horizon in HORIZONS
        ],
        "growth_forecast_diagnostics": growth_forecast_diagnostics(panel),
        "chronological_stability_242d": chronology,
        "latest_signal": _latest_signal_payload(panel),
        "acceptance": {
            "positive_ic_all_horizons": all_positive,
            "positive_non_overlapping_242d_ic_median": stability_supported,
            "positive_ic_both_chronological_halves_242d": chronological_stability_supported,
            "strictly_monotonic_quintiles_all_horizons": monotonic_all_horizons,
            "trading_use_authorized": False,
            "next_required_evidence": "冻结规则与数据指纹后的真正前瞻样本",
        },
        "limitations": [
            "申万行业映射是静态标签，不是历史点时行业分类；本轮仅用于固定方法路由。",
            "点时财务没有经营现金流字段，尚不能加入现金利润质量过滤。",
            "只有60个月成分股价格与点时聚合历史，242日已兑现样本更少且相互重叠。",
            "FairPE以历史市场PE为训练目标，但信号日价格和信号日PE不进入评分特征。",
            "全部历史已参与研究，任何通过都只能说明机制值得冻结，不能称为样本外Alpha。",
        ],
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "input_hashes": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                CONFIG_FILE,
                INDEX_FILE,
                TOTAL_RETURN_FILE,
                VALUATION_FILE,
                BOND_FILE,
                WEIGHTS_FILE,
                FINANCIALS_FILE,
                CONSTITUENT_FILE,
                INDUSTRY_FILE,
            )
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_FILE, index=False)
    report["output_sha256"] = _sha256(OUTPUT_FILE)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
