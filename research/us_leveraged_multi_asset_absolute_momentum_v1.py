"""美国上市杠杆多资产绝对动量V1：冻结信号、真实股数账本与可见期评价。"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "us_leveraged_multi_asset_absolute_momentum_v1.yaml"
CONSERVATIVE_EVALUATOR_CONFIG = (
    ROOT / "config" / "qdii_cross_market_discount_reversion_zero_variance_v2.yaml"
)


@dataclass
class UsEtfPosition:
    """基础与压力账本共享的真实整数股持仓。"""

    ticker: str
    shares: int
    entry_date: pd.Timestamp
    last_raw_close: float
    latest_prior_median_turnover_usd: float
    pending_exit: bool = False
    exit_signal_date: pd.Timestamp | None = None
    delayed_exit_days: int = 0


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取并校验候选合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("美国杠杆多资产合同必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    """防止目标、产品、成本或研究边界被静默弱化。"""

    failures: list[str] = []
    protocol = contract.get("protocol", {})
    objective = contract.get("objective", {})
    account = contract.get("account", {})
    universe = contract.get("universe", {})
    signal = contract.get("signal", {})
    portfolio = contract.get("portfolio", {})
    costs = contract.get("costs", {})
    risk = contract.get("risk", {})
    gates = contract.get("visible_gates", {})
    safety = contract.get("safety", {})

    if protocol.get("candidate_id") != "US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V5":
        failures.append("parent_protocol_id")
    expected_numbers = {
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "cash_annual_rate": 0.015,
    }
    for key, expected in expected_numbers.items():
        try:
            actual = float(account.get(key, float("nan")))
        except (TypeError, ValueError):
            actual = float("nan")
        if actual != expected:
            failures.append(f"account_{key}")
    if float(objective.get("minimum_annualized_net_excess", float("nan"))) != 0.40:
        failures.append("objective_excess")
    if float(objective.get("minimum_strategy_net_sharpe", float("nan"))) != 1.50:
        failures.append("objective_sharpe")
    if universe.get("fixed_tickers") != ["UPRO", "TQQQ", "TMF", "UGL"]:
        failures.append("fixed_tickers")
    if int(signal.get("momentum_lookback_observed_days", 0)) != 252:
        failures.append("momentum_lookback")
    if int(signal.get("trend_sma_observed_days", 0)) != 200:
        failures.append("trend_sma")
    if int(signal.get("target_asset_count", 0)) != 2:
        failures.append("target_asset_count")
    if int(portfolio.get("maximum_positions", 0)) != 2:
        failures.append("maximum_positions")
    if float(portfolio.get("target_weight_per_selected_asset", float("nan"))) != 0.50:
        failures.append("target_weight")
    if not bool(portfolio.get("target_security_gross_notional_includes_stress_cost_reserve")):
        failures.append("stress_cost_reserve")
    expected_base = (
        float(costs.get("user_transaction_fee_rate_per_leg", float("nan")))
        + float(costs.get("conservative_other_trading_cost_rate_per_leg", float("nan")))
        + float(costs.get("base_slippage_bps_per_leg", float("nan"))) / 10000.0
        + float(costs.get("base_market_impact_bps_per_leg", float("nan"))) / 10000.0
    )
    expected_stress = (
        float(costs.get("user_transaction_fee_rate_per_leg", float("nan")))
        + float(costs.get("conservative_other_trading_cost_rate_per_leg", float("nan")))
        + float(costs.get("stress_slippage_bps_per_leg", float("nan"))) / 10000.0
        + float(costs.get("stress_market_impact_bps_per_leg", float("nan"))) / 10000.0
    )
    if not np.isclose(expected_base, 0.0017, atol=1e-15):
        failures.append("base_security_cost")
    if not np.isclose(expected_stress, 0.0047, atol=1e-15):
        failures.append("stress_security_cost")
    if float(costs.get("base_total_security_cost_bps_per_leg", float("nan"))) != 17.0:
        failures.append("base_total_cost_label")
    if float(costs.get("stress_total_security_cost_bps_per_leg", float("nan"))) != 47.0:
        failures.append("stress_total_cost_label")
    if float(gates.get("minimum_annualized_net_excess", float("nan"))) != 0.40:
        failures.append("gate_excess")
    if float(gates.get("minimum_strategy_net_sharpe", float("nan"))) != 1.50:
        failures.append("gate_sharpe")
    if int(gates.get("annualization_trading_days", 0)) != 252:
        failures.append("annualization_days")
    if int(gates.get("bootstrap", {}).get("repetitions", 0)) != 5000:
        failures.append("bootstrap_repetitions")
    if float(risk.get("maximum_gross_exposure", float("nan"))) != 1.0:
        failures.append("gross_exposure")
    if any(
        bool(risk.get(name, True))
        for name in (
            "account_borrowing_allowed",
            "account_margin_allowed",
            "short_sale_allowed",
            "derivative_position_allowed",
        )
    ):
        failures.append("account_leverage_or_short")
    if any(
        bool(safety.get(name, True))
        for name in (
            "paper_position_generation",
            "shadow_signal_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety")
    if failures:
        raise ValueError(f"美国杠杆多资产合同被弱化或损坏：{sorted(set(failures))}")


def security_cost_rate(contract: dict[str, Any], scenario: str) -> float:
    """返回基础或压力情景的单边证券交易总成本率。"""

    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知情景：{scenario}")
    costs = contract["costs"]
    rate = (
        float(costs["user_transaction_fee_rate_per_leg"])
        + float(costs["conservative_other_trading_cost_rate_per_leg"])
        + float(costs[f"{scenario}_slippage_bps_per_leg"]) / 10000.0
        + float(costs[f"{scenario}_market_impact_bps_per_leg"]) / 10000.0
    )
    expected = float(costs[f"{scenario}_total_security_cost_bps_per_leg"]) / 10000.0
    if not np.isclose(rate, expected, atol=1e-15):
        raise DataContractError(f"{scenario}证券成本分项与冻结总成本不一致")
    return rate


def fx_spread_rate(contract: dict[str, Any], scenario: str) -> float:
    """返回单次人民币与美元兑换点差率。"""

    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知情景：{scenario}")
    return float(
        contract["costs"][f"{scenario}_fx_conversion_spread_bps_per_conversion"]
    ) / 10000.0


def _normalize_date_column(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = (
        pd.to_datetime(result[column], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    return result


def load_visible_inputs(
    contract: dict[str, Any],
    *,
    signal_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只读取可见输入；冻结覆盖阶段不读取开盘、汇率、基准或封存文件。"""

    inputs = contract["inputs"]
    status = json.loads((ROOT / inputs["source_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY":
        raise DataContractError(f"采集状态未通过：{status.get('status')}")
    if status.get("strategy_total_return_or_rank_computed") is not False:
        raise DataContractError("采集阶段越权计算了策略收益或排名")
    if status.get("portfolio_nav_or_target_gate_computed") is not False:
        raise DataContractError("采集阶段越权计算了组合净值或目标门")

    master = pd.read_parquet(ROOT / inputs["product_master"])
    signal_columns = [
        "ticker",
        "date",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "signal_total_return_index",
    ]
    full_columns = [
        "ticker",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "qfq_factor",
        "adjust",
        "split_ratio_at_open",
        "cash_distribution_per_post_event_share_usd",
        "signal_total_return_index",
    ]
    panel_columns = signal_columns if signal_only else full_columns
    panel = pd.read_parquet(ROOT / inputs["visible_panel"], columns=panel_columns)
    panel = _normalize_date_column(panel)
    visible_end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    if panel.empty or panel["date"].max() > visible_end:
        raise DataContractError("可见价格面板为空或越过可见期末")
    if panel[["ticker", "date"]].duplicated().any():
        raise DataContractError("可见价格面板存在重复主键")
    if set(panel["ticker"].unique()) != set(contract["universe"]["fixed_tickers"]):
        raise DataContractError("可见价格面板未覆盖固定四只产品")

    if signal_only:
        fx = pd.DataFrame(columns=["date", "cny_per_usd"])
        benchmark = pd.DataFrame(columns=["date", "close"])
        fx_columns: list[str] = []
        benchmark_columns: list[str] = []
    else:
        fx = _normalize_date_column(pd.read_parquet(ROOT / inputs["visible_fx"]))
        benchmark = _normalize_date_column(
            pd.read_parquet(ROOT / inputs["visible_benchmark"])
        )
        fx_columns = list(fx.columns)
        benchmark_columns = list(benchmark.columns)
        if fx.empty or benchmark.empty:
            raise DataContractError("可见汇率或基准为空")
        if fx["date"].max() > visible_end or benchmark["date"].max() > visible_end:
            raise DataContractError("可见汇率或基准越过可见期末")

    audit = {
        "source_status": status["status"],
        "panel_columns_read": panel_columns,
        "fx_columns_read": fx_columns,
        "benchmark_columns_read": benchmark_columns,
        "visible_panel_path": inputs["visible_panel"],
        "sealed_panel_or_benchmark_read": False,
        "future_open_or_strategy_return_read": False if signal_only else True,
        "signal_only": signal_only,
    }
    return panel, master, fx, benchmark, audit


def _common_calendar(panel: pd.DataFrame, tickers: list[str]) -> pd.DatetimeIndex:
    counts = panel.groupby("date", observed=True)["ticker"].nunique()
    common = counts.loc[counts.eq(len(tickers))].index
    return pd.DatetimeIndex(common).sort_values().unique()


def build_monthly_selections(
    panel: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """仅用月末收盘及更早信息形成选择和下一共同交易日映射。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    tickers = list(contract["universe"]["fixed_tickers"])
    required = {
        "ticker",
        "date",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "signal_total_return_index",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise DataContractError(f"信号面板缺少字段：{missing}")
    data = _normalize_date_column(panel)
    data = data.loc[data["ticker"].isin(tickers)].sort_values(["ticker", "date"])
    if data[["ticker", "date"]].duplicated().any():
        raise DataContractError("信号面板存在重复主键")
    common = _common_calendar(data, tickers)
    if len(common) < 2:
        raise DataContractError("共同美国交易日不足")

    calendar = pd.DataFrame({"signal_date": common})
    calendar["month"] = calendar["signal_date"].dt.to_period("M")
    month_ends = calendar.groupby("month", observed=True)["signal_date"].max().sort_values()
    common_series = pd.Series(common)
    next_mapping = pd.Series(common_series.shift(-1).values, index=common_series.values)
    schedule = pd.DataFrame({"signal_date": month_ends.values})
    schedule["execution_date"] = schedule["signal_date"].map(next_mapping)
    schedule = schedule.dropna(subset=["execution_date"])
    schedule = schedule.loc[
        schedule["execution_date"].between(start_ts, end_ts)
    ].sort_values("execution_date").reset_index(drop=True)
    if schedule.empty:
        raise DataContractError("冻结可见期没有月频执行日")

    lookback = int(contract["signal"]["momentum_lookback_observed_days"])
    trend_window = int(contract["signal"]["trend_sma_observed_days"])
    vol_window = int(contract["signal"]["volatility_lookback_observed_days"])
    turnover_window = int(contract["universe"]["turnover_lookback_observed_days"])
    annualization = int(contract["signal"]["volatility_annualization_days"])
    feature_parts: list[pd.DataFrame] = []
    for ticker, group in data.groupby("ticker", observed=True, sort=True):
        item = group.copy().sort_values("date").reset_index(drop=True)
        tri = pd.to_numeric(item["signal_total_return_index"], errors="coerce")
        item["observed_bar_count"] = np.arange(1, len(item) + 1)
        item["momentum_252"] = tri / tri.shift(lookback) - 1.0
        item["trend_sma_200"] = tri.rolling(trend_window, min_periods=trend_window).mean()
        security_return = tri.pct_change(fill_method=None)
        item["realized_volatility_60"] = (
            security_return.rolling(vol_window, min_periods=vol_window).std(ddof=1)
            * np.sqrt(annualization)
        )
        item["prior_median_dollar_turnover_20"] = (
            pd.to_numeric(item["raw_dollar_turnover_usd"], errors="coerce")
            .shift(1)
            .rolling(turnover_window, min_periods=turnover_window)
            .median()
        )
        feature_parts.append(item)
    all_features = pd.concat(feature_parts, ignore_index=True)
    features = all_features.loc[
        all_features["date"].isin(schedule["signal_date"])
    ].copy()
    features = features.merge(schedule, left_on="date", right_on="signal_date", validate="many_to_one")
    minimum_bars = int(contract["universe"]["minimum_observed_bars_for_signal"])
    minimum_turnover = float(
        contract["universe"]["minimum_prior_median_dollar_turnover_usd"]
    )
    features["eligible"] = (
        features["observed_bar_count"].ge(minimum_bars)
        & features["momentum_252"].gt(0.0)
        & features["signal_total_return_index"].gt(features["trend_sma_200"])
        & features["realized_volatility_60"].notna()
        & features["prior_median_dollar_turnover_20"].ge(minimum_turnover)
        & features["raw_close"].gt(0.0)
        & features["raw_volume"].gt(0.0)
    )
    eligible = features.loc[features["eligible"]].copy()
    eligible.sort_values(
        ["signal_date", "momentum_252", "realized_volatility_60", "ticker"],
        ascending=[True, False, True, True],
        kind="mergesort",
        inplace=True,
    )
    eligible["selection_rank"] = eligible.groupby("signal_date", observed=True).cumcount() + 1
    target_count = int(contract["signal"]["target_asset_count"])
    selections = eligible.loc[eligible["selection_rank"].le(target_count)].copy()
    selection_columns = [
        "signal_date",
        "execution_date",
        "ticker",
        "selection_rank",
        "momentum_252",
        "realized_volatility_60",
        "prior_median_dollar_turnover_20",
    ]
    selections = selections[selection_columns].sort_values(
        ["execution_date", "selection_rank", "ticker"]
    ).reset_index(drop=True)
    feature_columns = [
        "signal_date",
        "execution_date",
        "ticker",
        "observed_bar_count",
        "momentum_252",
        "trend_sma_200",
        "realized_volatility_60",
        "prior_median_dollar_turnover_20",
        "eligible",
    ]
    features = features[feature_columns].sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    eligible_counts = eligible.groupby("signal_date", observed=True)["ticker"].nunique()
    audit = {
        "common_trading_day_count": int(len(common)),
        "signal_date_count": int(len(schedule)),
        "selection_row_count": int(len(selections)),
        "dates_with_at_least_one_eligible_asset": int(eligible_counts.ge(1).sum()),
        "dates_with_two_eligible_assets": int(eligible_counts.ge(2).sum()),
        "eligible_asset_date_rows": int(len(eligible)),
        "first_signal_date": schedule["signal_date"].min().date().isoformat(),
        "last_signal_date": schedule["signal_date"].max().date().isoformat(),
        "first_execution_date": schedule["execution_date"].min().date().isoformat(),
        "last_execution_date": schedule["execution_date"].max().date().isoformat(),
        "signal_columns_used": sorted(required),
        "future_open_or_strategy_return_read": False,
        "sealed_replication_read": False,
    }
    return selections, schedule, features, audit


def prepare_daily_context(
    panel: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """把严格滞后的汇率和已收盘H00300映射到共同美国交易日。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    tickers = list(contract["universe"]["fixed_tickers"])
    common = _common_calendar(_normalize_date_column(panel), tickers)
    calendar = pd.DataFrame({"date": common})
    calendar = calendar.loc[calendar["date"].le(end_ts)].copy()
    if calendar.empty or calendar["date"].min() >= start_ts:
        raise DataContractError("评价开始日前缺少共同交易日，无法计算首日基准收益")

    fx_data = _normalize_date_column(fx)[["date", "cny_per_usd"]].copy()
    fx_data["cny_per_usd"] = pd.to_numeric(fx_data["cny_per_usd"], errors="coerce")
    fx_data = fx_data.dropna().loc[fx_data["cny_per_usd"].gt(0.0)]
    fx_data = fx_data.drop_duplicates("date", keep="last").sort_values("date")
    context = pd.merge_asof(
        calendar.sort_values("date"),
        fx_data.rename(columns={"date": "fx_source_date"}),
        left_on="date",
        right_on="fx_source_date",
        direction="backward",
        allow_exact_matches=False,
    )
    context["fx_age_calendar_days"] = (context["date"] - context["fx_source_date"]).dt.days
    maximum_fx_age = int(contract["currency"]["maximum_fx_staleness_calendar_days"])
    evaluation_mask = context["date"].between(start_ts, end_ts)
    if context.loc[evaluation_mask, ["fx_source_date", "cny_per_usd"]].isna().any().any():
        raise DataContractError("评价日存在无法映射的严格滞后美元汇率")
    if context.loc[evaluation_mask, "fx_age_calendar_days"].gt(maximum_fx_age).any():
        raise DataContractError("评价日严格滞后美元汇率超过冻结陈旧上限")

    benchmark_data = _normalize_date_column(benchmark)[["date", "close"]].copy()
    benchmark_data["close"] = pd.to_numeric(benchmark_data["close"], errors="coerce")
    benchmark_data = benchmark_data.dropna().loc[benchmark_data["close"].gt(0.0)]
    benchmark_data = benchmark_data.drop_duplicates("date", keep="last").sort_values("date")
    context = pd.merge_asof(
        context.sort_values("date"),
        benchmark_data.rename(
            columns={"date": "benchmark_source_date", "close": "benchmark_close"}
        ),
        left_on="date",
        right_on="benchmark_source_date",
        direction="backward",
        allow_exact_matches=True,
    )
    context["benchmark_total_return"] = context["benchmark_close"].pct_change(fill_method=None)
    context = context.loc[evaluation_mask].reset_index(drop=True)
    if context[["cny_per_usd", "benchmark_close", "benchmark_total_return"]].isna().any().any():
        raise DataContractError("评价日汇率、基准或首日基准收益缺失")
    return context


def _market_row(
    market: pd.DataFrame,
    date: pd.Timestamp,
    ticker: str,
) -> pd.Series | None:
    try:
        row = market.loc[(date, ticker)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        raise DataContractError(f"市场面板主键重复：{date.date()} {ticker}")
    return row


def _valid_execution_row(row: pd.Series | None) -> bool:
    if row is None:
        return False
    open_price = float(row.get("raw_open", np.nan))
    volume = float(row.get("raw_volume", np.nan))
    return bool(np.isfinite(open_price) and open_price > 0.0 and np.isfinite(volume) and volume > 0.0)


def run_portfolio_backtest(
    selections: pd.DataFrame,
    schedule: pd.DataFrame,
    signal_features: pd.DataFrame,
    panel: pd.DataFrame,
    context: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按真实未复权价格、整数股、拆分与税后分配运行双情景账本。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    calendar = pd.DatetimeIndex(context["date"])
    if calendar.empty or calendar[0] != start_ts or calendar[-1] != end_ts:
        raise DataContractError("日度上下文未精确覆盖冻结评价首尾")
    if context["date"].duplicated().any():
        raise DataContractError("日度上下文日期重复")
    market_frame = _normalize_date_column(panel)
    market = market_frame.set_index(["date", "ticker"]).sort_index()
    selected_by_execution = {
        pd.Timestamp(date): group.sort_values(["selection_rank", "ticker"])
        for date, group in selections.groupby("execution_date", observed=True)
    }
    schedule_by_execution = {
        pd.Timestamp(row.execution_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
    }
    features_by_key = {
        (pd.Timestamp(row.execution_date), str(row.ticker)): row
        for row in signal_features.itertuples(index=False)
    }

    base_cost = security_cost_rate(contract, "base")
    stress_cost = security_cost_rate(contract, "stress")
    costs = {"base": base_cost, "stress": stress_cost}
    fx_spreads = {
        "base": fx_spread_rate(contract, "base"),
        "stress": fx_spread_rate(contract, "stress"),
    }
    initial_capital = float(contract["account"]["initial_capital_cny"])
    first_fx = float(context.iloc[0]["cny_per_usd"])
    cash = {
        scenario: initial_capital / first_fx * (1.0 - fx_spreads[scenario])
        for scenario in ("base", "stress")
    }
    positions: dict[str, UsEtfPosition] = {}
    prior_nav_cny = {"base": initial_capital, "stress": initial_capital}
    daily_cash_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (
        1.0 / int(contract["visible_gates"]["annualization_trading_days"])
    ) - 1.0
    target_weight = float(contract["portfolio"]["target_weight_per_selected_asset"])
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usd"])
    maximum_capacity_fraction = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]
    )
    withholding = float(contract["corporate_actions"]["cash_distribution_withholding_rate"])

    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    entry_transaction_count = 0
    exit_transaction_count = 0
    resize_sell_transaction_count = 0
    blocked_entry_count = 0
    blocked_exit_attempt_count = 0
    capacity_block_count = 0
    stale_mark_count = 0
    split_event_count = 0
    fractional_cash_in_lieu_count = 0
    distribution_event_count = 0
    maximum_observed_order_capacity_fraction = 0.0
    maximum_position_count = 0
    maximum_gross_exposure = 0.0
    maximum_net_exposure = 0.0

    def capacity_fraction(gross: float, median_turnover: float) -> float:
        if not np.isfinite(median_turnover) or median_turnover <= 0.0:
            return float("inf")
        return gross / median_turnover

    def record_trade(
        *,
        date: pd.Timestamp,
        signal_date: pd.Timestamp,
        ticker: str,
        side: str,
        reason: str,
        shares: int,
        open_price: float,
        prior_median: float,
        delayed_exit_days: int = 0,
    ) -> None:
        nonlocal maximum_observed_order_capacity_fraction
        gross = float(shares) * open_price
        cap_fraction = capacity_fraction(gross, prior_median)
        if np.isfinite(cap_fraction):
            maximum_observed_order_capacity_fraction = max(
                maximum_observed_order_capacity_fraction, cap_fraction
            )
        if side == "BUY":
            for scenario in ("base", "stress"):
                cash[scenario] -= gross * (1.0 + costs[scenario])
        elif side == "SELL":
            for scenario in ("base", "stress"):
                cash[scenario] += gross * (1.0 - costs[scenario])
        else:
            raise ValueError(f"未知交易方向：{side}")
        if min(cash.values()) < -1e-8:
            raise DataContractError(f"{date.date()}交易后现金为负")
        trade_rows.append(
            {
                "trade_date": date,
                "signal_date": signal_date,
                "ticker": ticker,
                "side": side,
                "reason": reason,
                "shares": int(shares),
                "raw_open_usd": open_price,
                "gross_notional_usd": gross,
                "base_cost_usd": gross * base_cost,
                "stress_cost_usd": gross * stress_cost,
                "prior_median_dollar_turnover_20_usd": prior_median,
                "capacity_fraction": cap_fraction,
                "delayed_exit_days": int(delayed_exit_days),
            }
        )

    for day_index, context_row in enumerate(context.itertuples(index=False)):
        date = pd.Timestamp(context_row.date)
        fx_today = float(context_row.cny_per_usd)
        if day_index > 0:
            for scenario in ("base", "stress"):
                cash[scenario] *= 1.0 + daily_cash_rate

        entitled_distributions: list[tuple[int, float]] = []
        for ticker in list(positions):
            position = positions[ticker]
            row = _market_row(market, date, ticker)
            if row is None:
                continue
            ratio = float(row.get("split_ratio_at_open", 1.0))
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise DataContractError(f"{date.date()} {ticker}拆分比率无效")
            if not np.isclose(ratio, 1.0, rtol=0.0, atol=1e-12):
                exact_shares = position.shares * ratio
                whole_shares = int(math.floor(exact_shares + 1e-10))
                fractional = max(0.0, exact_shares - whole_shares)
                if fractional > 1e-10:
                    open_price = float(row.get("raw_open", np.nan))
                    if not np.isfinite(open_price) or open_price <= 0.0:
                        raise DataContractError(f"{date.date()} {ticker}零股现金替代缺少有效开盘")
                    for scenario in ("base", "stress"):
                        cash[scenario] += fractional * open_price
                    fractional_cash_in_lieu_count += 1
                position.shares = whole_shares
                split_event_count += 1
                if position.shares == 0:
                    del positions[ticker]
                    continue
            distribution = float(
                row.get("cash_distribution_per_post_event_share_usd", 0.0)
            )
            if not np.isfinite(distribution) or distribution < -1e-12:
                raise DataContractError(f"{date.date()} {ticker}现金分配无效")
            if distribution > 0.0 and position.shares > 0:
                entitled_distributions.append((position.shares, distribution))
                distribution_event_count += 1

        execution_signal = schedule_by_execution.get(date)
        selected_today = selected_by_execution.get(
            date,
            pd.DataFrame(
                columns=[
                    "signal_date",
                    "execution_date",
                    "ticker",
                    "selection_rank",
                    "prior_median_dollar_turnover_20",
                ]
            ),
        )
        if execution_signal is not None:
            selected_set = set(selected_today["ticker"].astype(str))
            for ticker, position in positions.items():
                feature = features_by_key.get((date, ticker))
                if feature is not None:
                    position.latest_prior_median_turnover_usd = float(
                        feature.prior_median_dollar_turnover_20
                    )
                if ticker not in selected_set and not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = execution_signal
                    position.delayed_exit_days = 0
        if date == end_ts:
            terminal_signal = date
            for position in positions.values():
                if not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = terminal_signal
                    position.delayed_exit_days = 0

        exited_tickers_today: set[str] = set()
        capacity_pass_today = True
        for ticker in list(positions):
            position = positions[ticker]
            if not position.pending_exit:
                continue
            row = _market_row(market, date, ticker)
            if not _valid_execution_row(row):
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                continue
            open_price = float(row["raw_open"])
            gross = position.shares * open_price
            cap_fraction = capacity_fraction(
                gross, position.latest_prior_median_turnover_usd
            )
            if cap_fraction > maximum_capacity_fraction + 1e-15:
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                capacity_block_count += 1
                capacity_pass_today = False
                continue
            record_trade(
                date=date,
                signal_date=position.exit_signal_date or date,
                ticker=ticker,
                side="SELL",
                reason="TERMINAL_EXIT" if date == end_ts else "SELECTION_EXIT",
                shares=position.shares,
                open_price=open_price,
                prior_median=position.latest_prior_median_turnover_usd,
                delayed_exit_days=position.delayed_exit_days,
            )
            exit_transaction_count += 1
            exited_tickers_today.add(ticker)
            del positions[ticker]

        if execution_signal is not None and date != end_ts:
            reference_nav_cny = min(prior_nav_cny.values())
            reference_nav_usd = reference_nav_cny / fx_today
            target_gross_per_asset = (
                reference_nav_usd * target_weight / (1.0 + stress_cost)
            )
            selected_records = {
                str(row.ticker): row for row in selected_today.itertuples(index=False)
            }

            for ticker, selection in selected_records.items():
                if ticker not in positions or positions[ticker].pending_exit:
                    continue
                row = _market_row(market, date, ticker)
                if not _valid_execution_row(row):
                    continue
                open_price = float(row["raw_open"])
                target_shares = int(math.floor(target_gross_per_asset / open_price))
                position = positions[ticker]
                position.latest_prior_median_turnover_usd = float(
                    selection.prior_median_dollar_turnover_20
                )
                excess_shares = max(0, position.shares - target_shares)
                gross = excess_shares * open_price
                if excess_shares <= 0 or gross < minimum_trade:
                    continue
                cap_fraction = capacity_fraction(
                    gross, position.latest_prior_median_turnover_usd
                )
                if cap_fraction > maximum_capacity_fraction + 1e-15:
                    capacity_block_count += 1
                    capacity_pass_today = False
                    continue
                record_trade(
                    date=date,
                    signal_date=execution_signal,
                    ticker=ticker,
                    side="SELL",
                    reason="MONTHLY_RESIZE_SELL",
                    shares=excess_shares,
                    open_price=open_price,
                    prior_median=position.latest_prior_median_turnover_usd,
                )
                position.shares -= excess_shares
                resize_sell_transaction_count += 1
                if position.shares == 0:
                    del positions[ticker]

            for ticker, selection in sorted(
                selected_records.items(), key=lambda item: (item[1].selection_rank, item[0])
            ):
                if ticker in exited_tickers_today:
                    continue
                current_shares = positions[ticker].shares if ticker in positions else 0
                if ticker in positions and positions[ticker].pending_exit:
                    continue
                row = _market_row(market, date, ticker)
                if not _valid_execution_row(row):
                    blocked_entry_count += 1
                    continue
                open_price = float(row["raw_open"])
                target_shares = int(math.floor(target_gross_per_asset / open_price))
                desired_shares = max(0, target_shares - current_shares)
                if desired_shares <= 0:
                    continue
                affordable = min(
                    int(math.floor(cash[scenario] / (open_price * (1.0 + costs[scenario]))))
                    for scenario in ("base", "stress")
                )
                buy_shares = min(desired_shares, max(0, affordable))
                gross = buy_shares * open_price
                if buy_shares <= 0 or gross < minimum_trade:
                    blocked_entry_count += 1
                    continue
                prior_median = float(selection.prior_median_dollar_turnover_20)
                cap_fraction = capacity_fraction(gross, prior_median)
                if cap_fraction > maximum_capacity_fraction + 1e-15:
                    blocked_entry_count += 1
                    capacity_block_count += 1
                    capacity_pass_today = False
                    continue
                record_trade(
                    date=date,
                    signal_date=execution_signal,
                    ticker=ticker,
                    side="BUY",
                    reason="MONTHLY_ENTRY_OR_RESIZE_BUY",
                    shares=buy_shares,
                    open_price=open_price,
                    prior_median=prior_median,
                )
                if ticker in positions:
                    positions[ticker].shares += buy_shares
                    positions[ticker].latest_prior_median_turnover_usd = prior_median
                else:
                    positions[ticker] = UsEtfPosition(
                        ticker=ticker,
                        shares=buy_shares,
                        entry_date=date,
                        last_raw_close=float(row["raw_close"]),
                        latest_prior_median_turnover_usd=prior_median,
                    )
                entry_transaction_count += 1

        for shares, distribution in entitled_distributions:
            net_cash = shares * distribution * (1.0 - withholding)
            for scenario in ("base", "stress"):
                cash[scenario] += net_cash

        position_value_usd = 0.0
        quality_complete_today = True
        for ticker, position in positions.items():
            row = _market_row(market, date, ticker)
            close_price = np.nan if row is None else float(row.get("raw_close", np.nan))
            if not np.isfinite(close_price) or close_price <= 0.0:
                close_price = position.last_raw_close
                quality_complete_today = False
                stale_mark_count += 1
            else:
                position.last_raw_close = close_price
            position_value_usd += position.shares * close_price

        if date == end_ts and positions:
            raise DataContractError(
                f"评价期末仍有无法退出的美国ETF持仓：{sorted(positions)}"
            )
        nav_usd = {
            scenario: cash[scenario] + position_value_usd
            for scenario in ("base", "stress")
        }
        if min(nav_usd.values()) <= 0.0:
            raise DataContractError(f"{date.date()}组合净值非正")
        reference_nav_usd_close = min(nav_usd.values())
        gross_exposure = position_value_usd / reference_nav_usd_close
        net_exposure = gross_exposure
        maximum_position_count = max(maximum_position_count, len(positions))
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_net_exposure = max(maximum_net_exposure, abs(net_exposure))
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-12:
            raise DataContractError(f"{date.date()}毛敞口超过冻结上限")
        if abs(net_exposure) > float(contract["risk"]["maximum_absolute_net_exposure"]) + 1e-12:
            raise DataContractError(f"{date.date()}净敞口超过冻结上限")

        if date == end_ts:
            nav_cny = {
                scenario: cash[scenario] * fx_today * (1.0 - fx_spreads[scenario])
                for scenario in ("base", "stress")
            }
        else:
            nav_cny = {
                scenario: nav_usd[scenario] * fx_today
                for scenario in ("base", "stress")
            }
        daily_rows.append(
            {
                "trade_date": date,
                "strategy_base_net_return": nav_cny["base"] / prior_nav_cny["base"] - 1.0,
                "strategy_stress_net_return": nav_cny["stress"] / prior_nav_cny["stress"] - 1.0,
                "benchmark_total_return": float(context_row.benchmark_total_return),
                "base_nav_cny": nav_cny["base"],
                "stress_nav_cny": nav_cny["stress"],
                "cny_per_usd_lagged": fx_today,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "position_count": len(positions),
                "quality_complete": quality_complete_today,
                "capacity_pass": capacity_pass_today,
            }
        )
        prior_nav_cny = nav_cny

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    audit = {
        "initial_capital_cny": initial_capital,
        "final_base_nav_cny": float(daily["base_nav_cny"].iloc[-1]),
        "final_stress_nav_cny": float(daily["stress_nav_cny"].iloc[-1]),
        "entry_transaction_count": int(entry_transaction_count),
        "exit_transaction_count": int(exit_transaction_count),
        "resize_sell_transaction_count": int(resize_sell_transaction_count),
        "blocked_entry_count": int(blocked_entry_count),
        "blocked_exit_attempt_count": int(blocked_exit_attempt_count),
        "capacity_block_count": int(capacity_block_count),
        "stale_mark_count": int(stale_mark_count),
        "split_event_count_for_held_positions": int(split_event_count),
        "fractional_cash_in_lieu_count": int(fractional_cash_in_lieu_count),
        "distribution_event_count_for_held_positions": int(distribution_event_count),
        "maximum_observed_order_capacity_fraction": float(
            maximum_observed_order_capacity_fraction
        ),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_net_exposure),
        "maximum_position_count": int(maximum_position_count),
        "base_security_cost_rate_per_leg": base_cost,
        "stress_security_cost_rate_per_leg": stress_cost,
        "base_fx_spread_rate_per_conversion": fx_spreads["base"],
        "stress_fx_spread_rate_per_conversion": fx_spreads["stress"],
        "terminal_position_count": int(daily.iloc[-1]["position_count"]),
        "sealed_replication_read": False,
    }
    return daily, trades, audit


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def render_markdown(report: dict[str, Any]) -> str:
    """生成面向决策的中文结果摘要。"""

    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    portfolio = report["data_audit"]["portfolio"]
    return "\n".join(
        [
            "# 美国杠杆多资产绝对动量V1可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            "## 核心结果",
            "",
            f"- 可见期：{report['period']['start']} 至 {report['period']['end']}。",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}。",
            f"- 压力策略净CAGR：{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础年化净超额：{metrics['base_annualized_excess']:.2%}。",
            f"- 压力年化净超额：{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础净夏普：{metrics['base_strategy_net_sharpe']:.3f}。",
            f"- 压力净夏普：{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 基础/压力最大回撤：{metrics['base_maximum_drawdown']:.2%}/{metrics['stress_maximum_drawdown']:.2%}。",
            f"- 买入/卖出交易数：{portfolio['entry_transaction_count']}/{portfolio['exit_transaction_count']}。",
            "",
            "## 40个百分点与高夏普硬门",
            "",
            f"- 基础净超额至少40个百分点：{gates['base_annualized_excess_at_least_40pct']}。",
            f"- 压力净超额至少40个百分点：{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础净夏普至少1.50：{gates['base_strategy_sharpe_at_least_1_5']}。",
            f"- 压力净夏普至少1.50：{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部可见期门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            "## 证据边界与决策",
            "",
            "汇率使用中行折算价历史表而非可成交即期汇率；资本利得税未建模，现金分配统一扣30%。",
            f"{report['decision']['next_step']}。",
            "",
            "封存公式族复验、Paper、Shadow、订单、券商连接和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """仅运行一次冻结可见期，不读取封存公式族复验。"""

    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, fx, benchmark, input_audit = load_visible_inputs(
        contract, signal_only=False
    )
    selections, schedule, features, signal_audit = build_monthly_selections(
        panel,
        contract,
        start=start,
        end=end,
    )
    context = prepare_daily_context(
        panel,
        fx,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        selections,
        schedule,
        features,
        panel,
        context,
        contract,
        start=start,
        end=end,
    )
    correction = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        contract,
        correction,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    status = (
        "VISIBLE_PASS_FORMULA_FAMILY_REPLICATION_AUTHORIZED_NOT_OPENED"
        if passed
        else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
    )
    report = {
        "schema_version": "1.0.0",
        "report_id": "US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": contract["objective"],
        "manifest_verification": manifest_verification,
        "data_audit": {
            "inputs": input_audit,
            "product_master_rows": int(len(master)),
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "daily_context": {
                "rows": int(len(context)),
                "first_date": context["date"].min().date().isoformat(),
                "last_date": context["date"].max().date().isoformat(),
                "maximum_fx_age_calendar_days": int(context["fx_age_calendar_days"].max()),
                "strictly_lagged_fx": bool((context["fx_source_date"] < context["date"]).all()),
            },
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "formula_family_replication_conditionally_eligible": passed,
            "sealed_replication_authorized": passed,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": (
                "可见期全部通过；另行验证清单后才可打开封存公式族复验，当前仍未打开"
                if passed
                else "冻结拒绝本候选，不打开封存期，不修改参数救回，并继续登记新的独立机制"
            ),
        },
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_selections"], selections)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
