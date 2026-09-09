"""BTC现货长周期趋势与波动缩放V2。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    atomic_json,
    atomic_parquet,
    atomic_text,
    prepare_daily_context,
    run_portfolio_backtest,
)
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "btc_spot_long_horizon_trend_v2.yaml"


def _normalize_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    return result


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BTC长周期趋势合同必须是YAML对象")
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    protocol = contract.get("protocol", {})
    account = contract.get("account", {})
    universe = contract.get("universe", {})
    signal = contract.get("signal", {})
    costs = contract.get("costs", {})
    gates = contract.get("visible_gates", {})
    if protocol.get("candidate_id") != "BTC_SPOT_LONG_HORIZON_TREND_V2":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V11":
        failures.append("parent_protocol_id")
    if float(account.get("initial_capital_cny", float("nan"))) != 500000.0:
        failures.append("initial_capital")
    if float(account.get("user_transaction_fee_rate_per_leg", float("nan"))) != 0.0001:
        failures.append("user_fee")
    if universe.get("fixed_products") != ["BTC-USD"]:
        failures.append("fixed_products")
    expected_signal = {
        "trend_sma_calendar_days": 200,
        "momentum_lookback_calendar_days": 126,
        "realized_volatility_lookback_calendar_days": 60,
        "volatility_annualization_days": 365,
    }
    for key, value in expected_signal.items():
        if int(signal.get(key, 0)) != value:
            failures.append(key)
    if float(signal.get("portfolio_target_annualized_volatility", float("nan"))) != 0.40:
        failures.append("target_volatility")
    base_cost = (
        float(costs.get("user_transaction_fee_rate_per_leg", float("nan")))
        + float(costs.get("coinbase_taker_fee_rate_per_leg", float("nan")))
        + float(costs.get("base_slippage_bps_per_leg", float("nan"))) / 10000.0
        + float(costs.get("base_market_impact_bps_per_leg", float("nan"))) / 10000.0
    )
    stress_cost = (
        float(costs.get("user_transaction_fee_rate_per_leg", float("nan")))
        + float(costs.get("coinbase_taker_fee_rate_per_leg", float("nan")))
        + float(costs.get("stress_slippage_bps_per_leg", float("nan"))) / 10000.0
        + float(costs.get("stress_market_impact_bps_per_leg", float("nan"))) / 10000.0
    )
    if not np.isclose(base_cost, 0.0076, atol=1e-15):
        failures.append("base_cost")
    if not np.isclose(stress_cost, 0.0106, atol=1e-15):
        failures.append("stress_cost")
    if float(gates.get("minimum_annualized_net_excess", float("nan"))) != 0.40:
        failures.append("excess_gate")
    if float(gates.get("minimum_strategy_net_sharpe", float("nan"))) != 1.50:
        failures.append("sharpe_gate")
    if int(gates.get("bootstrap", {}).get("repetitions", 0)) != 5000:
        failures.append("bootstrap")
    if any(bool(value) for value in contract.get("safety", {}).values()):
        failures.append("safety")
    if any(
        bool(contract.get("risk", {}).get(name, True))
        for name in (
            "account_borrowing_allowed",
            "account_margin_allowed",
            "short_sale_allowed",
            "derivative_position_allowed",
            "staking_or_lending_allowed",
        )
    ):
        failures.append("risk")
    if failures:
        raise ValueError(f"BTC长周期趋势合同被弱化或损坏：{sorted(set(failures))}")


def load_visible_inputs(
    contract: dict[str, Any],
    *,
    signal_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = contract["inputs"]
    status = json.loads((ROOT / inputs["source_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS_INPUT_ACQUISITION_ONLY":
        raise DataContractError(f"数字资产采集状态未通过：{status.get('status')}")
    if status.get("strategy_total_return_or_rank_computed") is not False:
        raise DataContractError("采集阶段越权计算策略收益或排名")
    if status.get("complete_utc_daily_calendar") is not True:
        raise DataContractError("Coinbase UTC日历不完整")
    products = contract["universe"]["fixed_products"]
    master = pd.read_parquet(ROOT / inputs["product_master"])
    master = master.loc[master["product_id"].isin(products)].reset_index(drop=True)
    signal_columns = ["product_id", "date", "close", "volume", "dollar_turnover_usd"]
    full_columns = [
        "product_id", "date", "open", "high", "low", "close", "volume",
        "dollar_turnover_usd",
    ]
    columns = signal_columns if signal_only else full_columns
    panel = pd.read_parquet(ROOT / inputs["visible_panel"], columns=columns)
    panel = _normalize_date(panel.loc[panel["product_id"].isin(products)])
    visible_end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    if panel.empty or panel["date"].max() > visible_end:
        raise DataContractError("BTC可见面板为空或越界")
    if panel[["product_id", "date"]].duplicated().any():
        raise DataContractError("BTC可见面板主键重复")
    if set(panel["product_id"].unique()) != set(products) or len(master) != 1:
        raise DataContractError("BTC可见面板或产品主表不完整")
    if signal_only:
        fx = pd.DataFrame(columns=["date", "cny_per_usd"])
        benchmark = pd.DataFrame(columns=["date", "close"])
    else:
        fx = _normalize_date(pd.read_parquet(ROOT / inputs["visible_fx"]))
        benchmark = _normalize_date(pd.read_parquet(ROOT / inputs["visible_benchmark"]))
        if fx.empty or benchmark.empty:
            raise DataContractError("BTC可见汇率或基准为空")
        if fx["date"].max() > visible_end or benchmark["date"].max() > visible_end:
            raise DataContractError("BTC可见汇率或基准越界")
    audit = {
        "source_status": status["status"],
        "panel_columns_read": columns,
        "fx_columns_read": [] if signal_only else list(fx.columns),
        "benchmark_columns_read": [] if signal_only else list(benchmark.columns),
        "sealed_inputs_read": False,
        "future_open_or_strategy_return_read": False if signal_only else True,
        "signal_only": signal_only,
    }
    return panel, master, fx, benchmark, audit


def build_weekly_targets(
    panel: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    data = _normalize_date(panel).sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(data["date"]).sort_values().unique()
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    if not dates.equals(expected):
        raise DataContractError("BTC共同UTC日历存在缺日")
    sunday = dates[dates.weekday == 6]
    schedule = pd.DataFrame({"signal_date": sunday})
    schedule["execution_date"] = schedule["signal_date"] + pd.Timedelta(days=1)
    schedule = schedule.loc[
        schedule["execution_date"].between(start_ts, end_ts)
        & schedule["execution_date"].isin(dates)
    ].reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")
    returns = close.pct_change(fill_method=None)
    sma_days = int(contract["signal"]["trend_sma_calendar_days"])
    momentum_days = int(contract["signal"]["momentum_lookback_calendar_days"])
    volatility_days = int(contract["signal"]["realized_volatility_lookback_calendar_days"])
    turnover_days = int(contract["universe"]["turnover_lookback_calendar_days"])
    annualization = int(contract["signal"]["volatility_annualization_days"])
    data["observed_bar_count"] = np.arange(1, len(data) + 1)
    data["sma_200"] = close.rolling(sma_days, min_periods=sma_days).mean()
    data["momentum_126"] = close / close.shift(momentum_days) - 1.0
    data["realized_volatility_60"] = (
        returns.rolling(volatility_days, min_periods=volatility_days).std(ddof=1)
        * np.sqrt(annualization)
    )
    data["prior_median_dollar_turnover_20"] = (
        pd.to_numeric(data["dollar_turnover_usd"], errors="coerce")
        .shift(1).rolling(turnover_days, min_periods=turnover_days).median()
    )
    features = data.loc[data["date"].isin(schedule["signal_date"])].copy()
    features = features.merge(
        schedule, left_on="date", right_on="signal_date", validate="one_to_one"
    )
    features["eligible"] = (
        features["observed_bar_count"].ge(
            int(contract["universe"]["minimum_observed_daily_bars_for_signal"])
        )
        & features["close"].gt(features["sma_200"])
        & features["momentum_126"].gt(0.0)
        & features["realized_volatility_60"].gt(0.0)
        & features["prior_median_dollar_turnover_20"].ge(
            float(contract["universe"]["minimum_prior_median_dollar_turnover_usd"])
        )
        & features["volume"].gt(0.0)
    )
    target_vol = float(contract["signal"]["portfolio_target_annualized_volatility"])
    target_features = features.loc[features["eligible"]].copy()
    target_features["target_weight"] = np.minimum(
        1.0, target_vol / target_features["realized_volatility_60"]
    )
    target_columns = [
        "signal_date", "execution_date", "product_id", "sma_200", "momentum_126",
        "realized_volatility_60", "prior_median_dollar_turnover_20", "target_weight",
    ]
    targets = target_features[target_columns].sort_values("execution_date").reset_index(drop=True)
    feature_columns = [
        "signal_date", "execution_date", "product_id", "observed_bar_count", "sma_200",
        "momentum_126", "realized_volatility_60",
        "prior_median_dollar_turnover_20", "eligible",
    ]
    features = features[feature_columns].sort_values("signal_date").reset_index(drop=True)
    audit = {
        "complete_calendar_day_count": int(len(dates)),
        "signal_date_count": int(len(schedule)),
        "target_row_count": int(len(targets)),
        "signal_dates_with_positive_target_weight": int(targets["signal_date"].nunique()),
        "maximum_target_gross_weight": float(targets["target_weight"].max()) if len(targets) else 0.0,
        "first_signal_date": schedule["signal_date"].min().date().isoformat(),
        "last_signal_date": schedule["signal_date"].max().date().isoformat(),
        "future_open_or_strategy_return_read": False,
        "sealed_replication_read": False,
    }
    return targets, schedule, features, audit


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["evaluation"]["metrics"]
    gates = report["evaluation"]["gates"]
    return "\n".join(
        [
            "# BTC现货长周期趋势V2可见期结果", "",
            f"状态：`{report['status']}`", "",
            f"- H00300全收益CAGR：{metrics['benchmark_total_return_cagr']:.2%}。",
            f"- 基础/压力策略净CAGR：{metrics['strategy_base_net_cagr']:.2%}/{metrics['strategy_stress_net_cagr']:.2%}。",
            f"- 基础/压力年化净超额：{metrics['base_annualized_excess']:.2%}/{metrics['stress_annualized_excess']:.2%}。",
            f"- 基础/压力净夏普：{metrics['base_strategy_net_sharpe']:.3f}/{metrics['stress_strategy_net_sharpe']:.3f}。",
            f"- 基础/压力40个百分点门：{gates['base_annualized_excess_at_least_40pct']}/{gates['stress_annualized_excess_at_least_40pct']}。",
            f"- 基础/压力1.50夏普门：{gates['base_strategy_sharpe_at_least_1_5']}/{gates['stress_strategy_sharpe_at_least_1_5']}。",
            f"- 全部门通过：{report['evaluation']['all_visible_gates_pass']}。", "",
            f"{report['decision']['next_step']}。", "",
            "封存复验、Paper、Shadow、订单、交易所连接和实盘均未打开。", "",
        ]
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    panel, master, fx, benchmark, input_audit = load_visible_inputs(contract)
    targets, schedule, features, signal_audit = build_weekly_targets(
        panel, contract, start=start, end=end
    )
    context = prepare_daily_context(panel, fx, benchmark, contract, start=start, end=end)
    daily, trades, portfolio_audit = run_portfolio_backtest(
        targets, schedule, features, panel, context, contract, start=start, end=end
    )
    correction = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily, contract, correction,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    report = {
        "schema_version": "1.0.0",
        "report_id": "BTC_SPOT_LONG_HORIZON_TREND_V2_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": (
            "VISIBLE_PASS_FORMULA_FAMILY_REPLICATION_AUTHORIZED_NOT_OPENED"
            if passed else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
        ),
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
                if passed else "冻结拒绝本候选，不打开封存期，不修改200/126/60日窗口、波动目标、产品或成本救回"
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
