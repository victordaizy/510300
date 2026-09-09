"""数字资产现货20日趋势与56日波动缩放周频组合V1。"""

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
CONFIG = ROOT / "config" / "digital_asset_spot_volatility_scaled_trend_v1.yaml"
CONSERVATIVE_EVALUATOR_CONFIG = (
    ROOT / "config" / "qdii_cross_market_discount_reversion_zero_variance_v2.yaml"
)


@dataclass
class SpotPosition:
    """基础与压力账本共享的现货数量。"""

    product_id: str
    quantity: float
    entry_date: pd.Timestamp
    last_close: float
    latest_prior_median_turnover_usd: float
    pending_exit: bool = False
    exit_signal_date: pd.Timestamp | None = None
    delayed_exit_days: int = 0


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("数字资产现货趋势合同必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    """防止目标、信号、费用或研究边界被弱化。"""

    failures: list[str] = []
    protocol = contract.get("protocol", {})
    account = contract.get("account", {})
    universe = contract.get("universe", {})
    signal = contract.get("signal", {})
    costs = contract.get("costs", {})
    gates = contract.get("visible_gates", {})
    risk = contract.get("risk", {})
    safety = contract.get("safety", {})
    if protocol.get("candidate_id") != "DIGITAL_ASSET_SPOT_VOLATILITY_SCALED_TREND_V1":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V7":
        failures.append("parent_protocol_id")
    expected_account = {
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "cash_annual_rate": 0.015,
    }
    for key, expected in expected_account.items():
        if float(account.get(key, float("nan"))) != expected:
            failures.append(f"account_{key}")
    if universe.get("fixed_products") != ["BTC-USD", "ETH-USD"]:
        failures.append("fixed_products")
    if int(signal.get("momentum_lookback_calendar_days", 0)) != 20:
        failures.append("momentum_lookback")
    if int(signal.get("covariance_lookback_calendar_days", 0)) != 56:
        failures.append("covariance_lookback")
    if float(signal.get("portfolio_target_annualized_volatility", float("nan"))) != 0.60:
        failures.append("target_volatility")
    if int(signal.get("volatility_annualization_days", 0)) != 365:
        failures.append("signal_annualization")
    expected_base = (
        float(costs.get("user_transaction_fee_rate_per_leg", float("nan")))
        + float(costs.get("coinbase_taker_fee_rate_per_leg", float("nan")))
        + float(costs.get("base_slippage_bps_per_leg", float("nan"))) / 10000.0
        + float(costs.get("base_market_impact_bps_per_leg", float("nan"))) / 10000.0
    )
    expected_stress = (
        float(costs.get("user_transaction_fee_rate_per_leg", float("nan")))
        + float(costs.get("coinbase_taker_fee_rate_per_leg", float("nan")))
        + float(costs.get("stress_slippage_bps_per_leg", float("nan"))) / 10000.0
        + float(costs.get("stress_market_impact_bps_per_leg", float("nan"))) / 10000.0
    )
    if not np.isclose(expected_base, 0.0076, atol=1e-15):
        failures.append("base_cost")
    if not np.isclose(expected_stress, 0.0106, atol=1e-15):
        failures.append("stress_cost")
    if float(costs.get("base_total_spot_cost_bps_per_leg", float("nan"))) != 76.0:
        failures.append("base_cost_label")
    if float(costs.get("stress_total_spot_cost_bps_per_leg", float("nan"))) != 106.0:
        failures.append("stress_cost_label")
    if float(gates.get("minimum_annualized_net_excess", float("nan"))) != 0.40:
        failures.append("excess_gate")
    if float(gates.get("minimum_strategy_net_sharpe", float("nan"))) != 1.50:
        failures.append("sharpe_gate")
    if int(gates.get("annualization_trading_days", 0)) != 365:
        failures.append("evaluation_annualization")
    if int(gates.get("bootstrap", {}).get("repetitions", 0)) != 5000:
        failures.append("bootstrap_repetitions")
    if any(
        bool(risk.get(name, True))
        for name in (
            "account_borrowing_allowed",
            "account_margin_allowed",
            "short_sale_allowed",
            "derivative_position_allowed",
            "staking_or_lending_allowed",
        )
    ):
        failures.append("risk")
    if any(bool(safety.get(name, True)) for name in safety):
        failures.append("safety")
    if failures:
        raise ValueError(f"数字资产现货趋势合同被弱化或损坏：{sorted(set(failures))}")


def spot_cost_rate(contract: dict[str, Any], scenario: str) -> float:
    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知情景：{scenario}")
    costs = contract["costs"]
    rate = (
        float(costs["user_transaction_fee_rate_per_leg"])
        + float(costs["coinbase_taker_fee_rate_per_leg"])
        + float(costs[f"{scenario}_slippage_bps_per_leg"]) / 10000.0
        + float(costs[f"{scenario}_market_impact_bps_per_leg"]) / 10000.0
    )
    expected = float(costs[f"{scenario}_total_spot_cost_bps_per_leg"]) / 10000.0
    if not np.isclose(rate, expected, atol=1e-15):
        raise DataContractError(f"{scenario}现货成本分项与冻结总成本不一致")
    return rate


def fx_spread_rate(contract: dict[str, Any], scenario: str) -> float:
    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知情景：{scenario}")
    return float(
        contract["costs"][f"{scenario}_fx_conversion_spread_bps_per_conversion"]
    ) / 10000.0


def _normalize_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = (
        pd.to_datetime(result["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    return result


def load_visible_inputs(
    contract: dict[str, Any],
    *,
    signal_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只读取可见期；信号覆盖阶段不读取开盘、汇率或基准。"""

    inputs = contract["inputs"]
    status = json.loads((ROOT / inputs["source_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY":
        raise DataContractError(f"数字资产采集状态未通过：{status.get('status')}")
    if status.get("strategy_total_return_or_rank_computed") is not False:
        raise DataContractError("采集阶段越权计算策略收益或排名")
    if status.get("complete_utc_daily_calendar") is not True:
        raise DataContractError("Coinbase UTC日历不完整")
    master = pd.read_parquet(ROOT / inputs["product_master"])
    signal_columns = [
        "product_id",
        "date",
        "close",
        "volume",
        "dollar_turnover_usd",
    ]
    full_columns = [
        "product_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_turnover_usd",
    ]
    columns = signal_columns if signal_only else full_columns
    panel = _normalize_date(
        pd.read_parquet(ROOT / inputs["visible_panel"], columns=columns)
    )
    visible_end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    if panel.empty or panel["date"].max() > visible_end:
        raise DataContractError("数字资产可见面板为空或越界")
    if panel[["product_id", "date"]].duplicated().any():
        raise DataContractError("数字资产可见面板主键重复")
    if set(panel["product_id"].unique()) != set(contract["universe"]["fixed_products"]):
        raise DataContractError("数字资产可见面板未覆盖固定产品")
    if signal_only:
        fx = pd.DataFrame(columns=["date", "cny_per_usd"])
        benchmark = pd.DataFrame(columns=["date", "close"])
        fx_columns: list[str] = []
        benchmark_columns: list[str] = []
    else:
        fx = _normalize_date(pd.read_parquet(ROOT / inputs["visible_fx"]))
        benchmark = _normalize_date(pd.read_parquet(ROOT / inputs["visible_benchmark"]))
        fx_columns = list(fx.columns)
        benchmark_columns = list(benchmark.columns)
        if fx.empty or benchmark.empty:
            raise DataContractError("数字资产可见汇率或基准为空")
        if fx["date"].max() > visible_end or benchmark["date"].max() > visible_end:
            raise DataContractError("数字资产可见汇率或基准越过期末")
    audit = {
        "source_status": status["status"],
        "panel_columns_read": columns,
        "fx_columns_read": fx_columns,
        "benchmark_columns_read": benchmark_columns,
        "sealed_inputs_read": False,
        "future_open_or_strategy_return_read": False if signal_only else True,
        "signal_only": signal_only,
    }
    return panel, master, fx, benchmark, audit


def _common_calendar(panel: pd.DataFrame, products: list[str]) -> pd.DatetimeIndex:
    counts = panel.groupby("date", observed=True)["product_id"].nunique()
    return pd.DatetimeIndex(counts.loc[counts.eq(len(products))].index).sort_values().unique()


def build_weekly_targets(
    panel: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """仅用完整周日收盘及更早数据形成下一周一目标权重。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    products = list(contract["universe"]["fixed_products"])
    required = {"product_id", "date", "close", "volume", "dollar_turnover_usd"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise DataContractError(f"数字资产信号面板缺少字段：{missing}")
    data = _normalize_date(panel).sort_values(["product_id", "date"])
    common = _common_calendar(data, products)
    if len(common) < 58:
        raise DataContractError("数字资产共同日历不足58天")
    expected = pd.date_range(common.min(), common.max(), freq="D")
    if not common.equals(expected):
        raise DataContractError("数字资产共同UTC日历存在缺日")
    sunday = common[common.weekday == 6]
    schedule = pd.DataFrame({"signal_date": sunday})
    schedule["execution_date"] = schedule["signal_date"] + pd.Timedelta(days=1)
    schedule = schedule.loc[
        schedule["execution_date"].between(start_ts, end_ts)
        & schedule["execution_date"].isin(common)
    ].reset_index(drop=True)
    if schedule.empty:
        raise DataContractError("可见期没有周频执行日")

    momentum_window = int(contract["signal"]["momentum_lookback_calendar_days"])
    covariance_window = int(contract["signal"]["covariance_lookback_calendar_days"])
    annualization = int(contract["signal"]["volatility_annualization_days"])
    turnover_window = int(contract["universe"]["turnover_lookback_calendar_days"])
    feature_parts: list[pd.DataFrame] = []
    return_parts: list[pd.DataFrame] = []
    for product_id, group in data.groupby("product_id", observed=True, sort=True):
        item = group.copy().sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(item["close"], errors="coerce")
        returns = close.pct_change(fill_method=None)
        item["observed_bar_count"] = np.arange(1, len(item) + 1)
        item["momentum_20"] = close / close.shift(momentum_window) - 1.0
        item["realized_volatility_56"] = (
            returns.rolling(covariance_window, min_periods=covariance_window).std(ddof=1)
            * np.sqrt(annualization)
        )
        item["prior_median_dollar_turnover_20"] = (
            pd.to_numeric(item["dollar_turnover_usd"], errors="coerce")
            .shift(1)
            .rolling(turnover_window, min_periods=turnover_window)
            .median()
        )
        feature_parts.append(item)
        return_parts.append(
            pd.DataFrame({"date": item["date"], "product_id": product_id, "return": returns})
        )
    all_features = pd.concat(feature_parts, ignore_index=True)
    returns_long = pd.concat(return_parts, ignore_index=True)
    returns_wide = returns_long.pivot(index="date", columns="product_id", values="return").sort_index()
    features = all_features.loc[all_features["date"].isin(schedule["signal_date"])].copy()
    features = features.merge(
        schedule, left_on="date", right_on="signal_date", validate="many_to_one"
    )
    minimum_bars = int(contract["universe"]["minimum_observed_daily_bars_for_signal"])
    minimum_turnover = float(
        contract["universe"]["minimum_prior_median_dollar_turnover_usd"]
    )
    features["eligible"] = (
        features["observed_bar_count"].ge(minimum_bars)
        & features["momentum_20"].gt(0.0)
        & features["realized_volatility_56"].gt(0.0)
        & features["prior_median_dollar_turnover_20"].ge(minimum_turnover)
        & features["close"].gt(0.0)
        & features["volume"].gt(0.0)
    )

    target_rows: list[dict[str, Any]] = []
    target_vol = float(contract["signal"]["portfolio_target_annualized_volatility"])
    minimum_covariance = int(
        contract["signal"]["covariance_minimum_complete_observations"]
    )
    for schedule_row in schedule.itertuples(index=False):
        signal_date = pd.Timestamp(schedule_row.signal_date)
        execution_date = pd.Timestamp(schedule_row.execution_date)
        candidates = features.loc[
            features["signal_date"].eq(signal_date) & features["eligible"]
        ].copy()
        if candidates.empty:
            continue
        active = candidates["product_id"].astype(str).tolist()
        history = returns_wide.loc[:signal_date, active].tail(covariance_window)
        if len(history) < minimum_covariance or history.isna().any().any():
            continue
        inverse_vol = 1.0 / candidates.set_index("product_id")["realized_volatility_56"]
        preliminary = inverse_vol / inverse_vol.sum()
        covariance = history.cov(ddof=1) * annualization
        vector = preliminary.reindex(active).to_numpy(dtype=float)
        covariance_array = covariance.reindex(index=active, columns=active).to_numpy(dtype=float)
        portfolio_variance = float(vector @ covariance_array @ vector)
        if not np.isfinite(portfolio_variance) or portfolio_variance <= 0.0:
            continue
        preliminary_vol = float(np.sqrt(portfolio_variance))
        scale = min(1.0, target_vol / preliminary_vol)
        for product_id in active:
            row = candidates.loc[candidates["product_id"].eq(product_id)].iloc[0]
            weight = float(preliminary.loc[product_id] * scale)
            target_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "product_id": product_id,
                    "momentum_20": float(row["momentum_20"]),
                    "realized_volatility_56": float(row["realized_volatility_56"]),
                    "prior_median_dollar_turnover_20": float(
                        row["prior_median_dollar_turnover_20"]
                    ),
                    "preliminary_weight": float(preliminary.loc[product_id]),
                    "preliminary_portfolio_volatility": preliminary_vol,
                    "volatility_scale": scale,
                    "target_weight": weight,
                }
            )
    targets = pd.DataFrame(target_rows)
    if targets.empty:
        targets = pd.DataFrame(
            columns=[
                "signal_date",
                "execution_date",
                "product_id",
                "momentum_20",
                "realized_volatility_56",
                "prior_median_dollar_turnover_20",
                "preliminary_weight",
                "preliminary_portfolio_volatility",
                "volatility_scale",
                "target_weight",
            ]
        )
    targets = targets.sort_values(["execution_date", "product_id"]).reset_index(drop=True)
    feature_columns = [
        "signal_date",
        "execution_date",
        "product_id",
        "observed_bar_count",
        "momentum_20",
        "realized_volatility_56",
        "prior_median_dollar_turnover_20",
        "eligible",
    ]
    features = features[feature_columns].sort_values(["signal_date", "product_id"]).reset_index(drop=True)
    gross_by_date = targets.groupby("signal_date", observed=True)["target_weight"].sum()
    if gross_by_date.gt(float(contract["signal"]["maximum_portfolio_gross_weight"]) + 1e-12).any():
        raise DataContractError("信号目标权重超过冻结毛敞口")
    audit = {
        "complete_calendar_day_count": int(len(common)),
        "signal_date_count": int(len(schedule)),
        "target_row_count": int(len(targets)),
        "signal_dates_with_positive_target_weight": int(targets["signal_date"].nunique()),
        "maximum_target_gross_weight": float(gross_by_date.max()) if len(gross_by_date) else 0.0,
        "first_signal_date": schedule["signal_date"].min().date().isoformat(),
        "last_signal_date": schedule["signal_date"].max().date().isoformat(),
        "future_open_or_strategy_return_read": False,
        "sealed_replication_read": False,
    }
    return targets, schedule, features, audit


def prepare_daily_context(
    panel: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """映射严格滞后官方汇率和最近H00300收盘到UTC日历。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    common = _common_calendar(_normalize_date(panel), contract["universe"]["fixed_products"])
    calendar = pd.DataFrame({"date": common[common <= end_ts]})
    if calendar.empty or calendar["date"].min() >= start_ts:
        raise DataContractError("评价开始日前缺少UTC日历，无法计算首日基准收益")
    fx_data = _normalize_date(fx)[["date", "cny_per_usd"]].copy()
    fx_data["cny_per_usd"] = pd.to_numeric(fx_data["cny_per_usd"], errors="coerce")
    fx_data = fx_data.dropna().loc[lambda x: x["cny_per_usd"].gt(0.0)]
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
    evaluation_mask = context["date"].between(start_ts, end_ts)
    maximum_age = int(contract["currency"]["maximum_fx_staleness_calendar_days"])
    if context.loc[evaluation_mask, ["fx_source_date", "cny_per_usd"]].isna().any().any():
        raise DataContractError("数字资产评价日存在无法映射的严格滞后官方汇率")
    if context.loc[evaluation_mask, "fx_age_calendar_days"].gt(maximum_age).any():
        raise DataContractError("数字资产评价日官方汇率超过冻结陈旧上限")
    benchmark_data = _normalize_date(benchmark)[["date", "close"]].copy()
    benchmark_data["close"] = pd.to_numeric(benchmark_data["close"], errors="coerce")
    benchmark_data = benchmark_data.dropna().loc[lambda x: x["close"].gt(0.0)]
    benchmark_data = benchmark_data.drop_duplicates("date", keep="last").sort_values("date")
    context = pd.merge_asof(
        context,
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
        raise DataContractError("数字资产评价日汇率、基准或首日收益缺失")
    return context


def _market_row(market: pd.DataFrame, date: pd.Timestamp, product_id: str) -> pd.Series | None:
    try:
        row = market.loc[(date, product_id)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        raise DataContractError(f"市场主键重复：{date.date()} {product_id}")
    return row


def _valid_execution(row: pd.Series | None) -> bool:
    if row is None:
        return False
    open_price = float(row.get("open", np.nan))
    volume = float(row.get("volume", np.nan))
    return bool(np.isfinite(open_price) and open_price > 0.0 and np.isfinite(volume) and volume > 0.0)


def _floor_quantity(quantity: float, increment: float) -> float:
    return math.floor((quantity + increment * 1e-8) / increment) * increment


def run_portfolio_backtest(
    targets: pd.DataFrame,
    schedule: pd.DataFrame,
    signal_features: pd.DataFrame,
    panel: pd.DataFrame,
    context: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """运行现货分数数量、双成本情景与人民币净值账本。"""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if context.empty or pd.Timestamp(context.iloc[0]["date"]) != start_ts or pd.Timestamp(context.iloc[-1]["date"]) != end_ts:
        raise DataContractError("数字资产日度上下文未精确覆盖评价首尾")
    market = _normalize_date(panel).set_index(["date", "product_id"]).sort_index()
    targets_by_execution = {
        pd.Timestamp(date): group.sort_values("product_id")
        for date, group in targets.groupby("execution_date", observed=True)
    }
    schedule_by_execution = {
        pd.Timestamp(row.execution_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
    }
    features_by_key = {
        (pd.Timestamp(row.execution_date), str(row.product_id)): row
        for row in signal_features.itertuples(index=False)
    }
    costs = {scenario: spot_cost_rate(contract, scenario) for scenario in ("base", "stress")}
    fx_spreads = {scenario: fx_spread_rate(contract, scenario) for scenario in ("base", "stress")}
    initial_capital = float(contract["account"]["initial_capital_cny"])
    first_fx = float(context.iloc[0]["cny_per_usd"])
    cash = {
        scenario: initial_capital / first_fx * (1.0 - fx_spreads[scenario])
        for scenario in ("base", "stress")
    }
    positions: dict[str, SpotPosition] = {}
    prior_nav_cny = {"base": initial_capital, "stress": initial_capital}
    daily_cash_rate = (1.0 + float(contract["account"]["cash_annual_rate"])) ** (
        1.0 / int(contract["visible_gates"]["annualization_trading_days"])
    ) - 1.0
    increment = float(contract["portfolio"]["quantity_increment"])
    minimum_trade = float(contract["portfolio"]["minimum_trade_notional_usd"])
    maximum_capacity = float(
        contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]
    )
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    entry_count = 0
    exit_count = 0
    resize_sell_count = 0
    blocked_entry_count = 0
    blocked_exit_count = 0
    capacity_block_count = 0
    stale_mark_count = 0
    maximum_capacity_fraction = 0.0
    maximum_gross_exposure = 0.0
    maximum_position_count = 0

    def order_capacity(gross: float, median: float) -> float:
        if not np.isfinite(median) or median <= 0.0:
            return float("inf")
        return gross / median

    def execute(
        *,
        date: pd.Timestamp,
        signal_date: pd.Timestamp,
        product_id: str,
        side: str,
        reason: str,
        quantity: float,
        open_price: float,
        median: float,
        delayed_exit_days: int = 0,
    ) -> None:
        nonlocal maximum_capacity_fraction
        gross = quantity * open_price
        fraction = order_capacity(gross, median)
        if np.isfinite(fraction):
            maximum_capacity_fraction = max(maximum_capacity_fraction, fraction)
        if side == "BUY":
            for scenario in ("base", "stress"):
                cash[scenario] -= gross * (1.0 + costs[scenario])
        elif side == "SELL":
            for scenario in ("base", "stress"):
                cash[scenario] += gross * (1.0 - costs[scenario])
        else:
            raise ValueError(f"未知方向：{side}")
        if min(cash.values()) < -1e-7:
            raise DataContractError(f"{date.date()}现货交易后现金为负")
        trade_rows.append(
            {
                "trade_date": date,
                "signal_date": signal_date,
                "product_id": product_id,
                "side": side,
                "reason": reason,
                "quantity": quantity,
                "open_usd": open_price,
                "gross_notional_usd": gross,
                "base_cost_usd": gross * costs["base"],
                "stress_cost_usd": gross * costs["stress"],
                "prior_median_dollar_turnover_20_usd": median,
                "capacity_fraction": fraction,
                "delayed_exit_days": delayed_exit_days,
            }
        )

    for day_index, context_row in enumerate(context.itertuples(index=False)):
        date = pd.Timestamp(context_row.date)
        fx_today = float(context_row.cny_per_usd)
        if day_index > 0:
            for scenario in ("base", "stress"):
                cash[scenario] *= 1.0 + daily_cash_rate
        signal_date = schedule_by_execution.get(date)
        todays_targets = targets_by_execution.get(
            date,
            pd.DataFrame(
                columns=[
                    "product_id",
                    "target_weight",
                    "prior_median_dollar_turnover_20",
                ]
            ),
        )
        target_map = {
            str(row.product_id): row for row in todays_targets.itertuples(index=False)
        }
        if signal_date is not None:
            for product_id, position in positions.items():
                feature = features_by_key.get((date, product_id))
                if feature is not None:
                    position.latest_prior_median_turnover_usd = float(
                        feature.prior_median_dollar_turnover_20
                    )
                if product_id not in target_map and not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = signal_date
                    position.delayed_exit_days = 0
        if date == end_ts:
            for position in positions.values():
                if not position.pending_exit:
                    position.pending_exit = True
                    position.exit_signal_date = date
                    position.delayed_exit_days = 0

        capacity_pass_today = True
        exited_today: set[str] = set()
        for product_id in list(positions):
            position = positions[product_id]
            if not position.pending_exit:
                continue
            row = _market_row(market, date, product_id)
            if not _valid_execution(row):
                position.delayed_exit_days += 1
                blocked_exit_count += 1
                continue
            open_price = float(row["open"])
            gross = position.quantity * open_price
            fraction = order_capacity(gross, position.latest_prior_median_turnover_usd)
            if fraction > maximum_capacity + 1e-15:
                position.delayed_exit_days += 1
                blocked_exit_count += 1
                capacity_block_count += 1
                capacity_pass_today = False
                continue
            execute(
                date=date,
                signal_date=position.exit_signal_date or date,
                product_id=product_id,
                side="SELL",
                reason="TERMINAL_EXIT" if date == end_ts else "TREND_EXIT",
                quantity=position.quantity,
                open_price=open_price,
                median=position.latest_prior_median_turnover_usd,
                delayed_exit_days=position.delayed_exit_days,
            )
            exit_count += 1
            exited_today.add(product_id)
            del positions[product_id]

        if signal_date is not None and date != end_ts:
            reference_nav_usd = min(prior_nav_cny.values()) / fx_today
            desired_quantities: dict[str, float] = {}
            for product_id, target in target_map.items():
                row = _market_row(market, date, product_id)
                if not _valid_execution(row):
                    continue
                open_price = float(row["open"])
                target_notional = (
                    reference_nav_usd
                    * float(target.target_weight)
                    / (1.0 + costs["stress"])
                )
                desired_quantities[product_id] = _floor_quantity(
                    target_notional / open_price, increment
                )

            for product_id, position in list(positions.items()):
                if position.pending_exit:
                    continue
                desired = desired_quantities.get(product_id, 0.0)
                excess = _floor_quantity(max(0.0, position.quantity - desired), increment)
                row = _market_row(market, date, product_id)
                if excess <= 0.0 or not _valid_execution(row):
                    continue
                open_price = float(row["open"])
                gross = excess * open_price
                if gross < minimum_trade:
                    continue
                fraction = order_capacity(gross, position.latest_prior_median_turnover_usd)
                if fraction > maximum_capacity + 1e-15:
                    capacity_block_count += 1
                    capacity_pass_today = False
                    continue
                execute(
                    date=date,
                    signal_date=signal_date,
                    product_id=product_id,
                    side="SELL",
                    reason="WEEKLY_RESIZE_SELL",
                    quantity=excess,
                    open_price=open_price,
                    median=position.latest_prior_median_turnover_usd,
                )
                position.quantity = _floor_quantity(position.quantity - excess, increment)
                resize_sell_count += 1
                if position.quantity <= increment / 2:
                    del positions[product_id]

            for product_id in sorted(target_map):
                if product_id in exited_today:
                    continue
                if product_id in positions and positions[product_id].pending_exit:
                    continue
                row = _market_row(market, date, product_id)
                if not _valid_execution(row):
                    blocked_entry_count += 1
                    continue
                desired = desired_quantities.get(product_id, 0.0)
                current = positions[product_id].quantity if product_id in positions else 0.0
                quantity = _floor_quantity(max(0.0, desired - current), increment)
                open_price = float(row["open"])
                affordable = min(
                    _floor_quantity(
                        cash[scenario] / (open_price * (1.0 + costs[scenario])),
                        increment,
                    )
                    for scenario in ("base", "stress")
                )
                quantity = min(quantity, affordable)
                gross = quantity * open_price
                if quantity <= 0.0 or gross < minimum_trade:
                    continue
                median = float(target_map[product_id].prior_median_dollar_turnover_20)
                fraction = order_capacity(gross, median)
                if fraction > maximum_capacity + 1e-15:
                    blocked_entry_count += 1
                    capacity_block_count += 1
                    capacity_pass_today = False
                    continue
                execute(
                    date=date,
                    signal_date=signal_date,
                    product_id=product_id,
                    side="BUY",
                    reason="WEEKLY_ENTRY_OR_RESIZE_BUY",
                    quantity=quantity,
                    open_price=open_price,
                    median=median,
                )
                if product_id in positions:
                    positions[product_id].quantity = _floor_quantity(
                        positions[product_id].quantity + quantity, increment
                    )
                    positions[product_id].latest_prior_median_turnover_usd = median
                else:
                    positions[product_id] = SpotPosition(
                        product_id=product_id,
                        quantity=quantity,
                        entry_date=date,
                        last_close=float(row["close"]),
                        latest_prior_median_turnover_usd=median,
                    )
                entry_count += 1

        position_value_usd = 0.0
        quality_complete_today = True
        for product_id, position in positions.items():
            row = _market_row(market, date, product_id)
            close = np.nan if row is None else float(row.get("close", np.nan))
            if not np.isfinite(close) or close <= 0.0:
                close = position.last_close
                quality_complete_today = False
                stale_mark_count += 1
            else:
                position.last_close = close
            position_value_usd += position.quantity * close
        if date == end_ts and positions:
            raise DataContractError(
                f"评价期末仍有无法退出的数字资产现货持仓：{sorted(positions)}"
            )
        nav_usd = {
            scenario: cash[scenario] + position_value_usd
            for scenario in ("base", "stress")
        }
        if min(nav_usd.values()) <= 0.0:
            raise DataContractError(f"{date.date()}数字资产组合净值非正")
        reference_close_nav = min(nav_usd.values())
        gross_exposure = position_value_usd / reference_close_nav
        if gross_exposure > float(contract["risk"]["maximum_gross_exposure"]) + 1e-12:
            raise DataContractError(f"{date.date()}数字资产毛敞口超过100%")
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_position_count = max(maximum_position_count, len(positions))
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
                "net_exposure": gross_exposure,
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
        "entry_transaction_count": int(entry_count),
        "exit_transaction_count": int(exit_count),
        "resize_sell_transaction_count": int(resize_sell_count),
        "blocked_entry_count": int(blocked_entry_count),
        "blocked_exit_attempt_count": int(blocked_exit_count),
        "capacity_block_count": int(capacity_block_count),
        "stale_mark_count": int(stale_mark_count),
        "maximum_order_capacity_fraction": float(maximum_capacity_fraction),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_gross_exposure),
        "maximum_position_count": int(maximum_position_count),
        "base_spot_cost_rate_per_leg": costs["base"],
        "stress_spot_cost_rate_per_leg": costs["stress"],
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
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    portfolio = report["data_audit"]["portfolio"]
    return "\n".join(
        [
            "# 数字资产现货波动缩放趋势V1可见期结果",
            "",
            f"状态：`{report['status']}`",
            "",
            "## 核心结果",
            "",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']:.3f}/{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 基础/压力最大回撤：{metrics['base_maximum_drawdown']:.2%}/{metrics['stress_maximum_drawdown']:.2%}。",
            f"- 买入/卖出交易数：{portfolio['entry_transaction_count']}/{portfolio['exit_transaction_count']}。",
            "",
            "## 40个百分点与高夏普硬门",
            "",
            f"- 基础/压力净超额至少40个百分点：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力净夏普至少1.50：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。",
            "",
            "## 决策",
            "",
            f"{report['decision']['next_step']}。",
            "",
            "封存复验、Paper、Shadow、订单、交易所连接和实盘均未打开。",
            "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """运行一次冻结可见期，不读取封存期。"""

    validate_contract(contract)
    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, fx, benchmark, input_audit = load_visible_inputs(
        contract, signal_only=False
    )
    targets, schedule, features, signal_audit = build_weekly_targets(
        panel, contract, start=start, end=end
    )
    context = prepare_daily_context(
        panel, fx, benchmark, contract, start=start, end=end
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        targets,
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
        "report_id": "DIGITAL_ASSET_SPOT_VOLATILITY_SCALED_TREND_V1_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": {
            "initial_capital_cny": contract["account"]["initial_capital_cny"],
            "minimum_annualized_net_excess": contract["visible_gates"]["minimum_annualized_net_excess"],
            "minimum_strategy_net_sharpe": contract["visible_gates"]["minimum_strategy_net_sharpe"],
            "user_transaction_fee_rate_per_leg": contract["account"]["user_transaction_fee_rate_per_leg"],
        },
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
                else "冻结拒绝本候选，不打开封存期，不修改产品、信号、波动目标或成本救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_targets"], targets)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
