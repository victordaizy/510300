"""Oracle 信息预算 V1.0.1 的保守验收与 Atlas 最终状态固化。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import oracle_information_budget_v1 as base  # noqa: E402
from backtest.engine import BacktestCosts  # noqa: E402
from known_state_policy_oracle_v1 import (  # noqa: E402
    STATE_BEAR,
    STATE_BULL,
    STATE_RANGE,
)
from three_state_trend_router_v1_0_1 import ContractError  # noqa: E402


PROJECT_ID = "510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE"
CONFIG_PATH = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_manifest.json"
PROGRAM_CORRECTION_MANIFEST_PATH = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_program_correction_manifest.json"
PROGRAM_CORRECTION_2_MANIFEST_PATH = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_program_correction_2_manifest.json"
ProgressCallback = Callable[[str], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_scalar(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    measurement = config["frozen_measurement"]
    checks = {
        "project_id": protocol["project_id"] == PROJECT_ID,
        "version": protocol["version"] == "1.0.1",
        "source_v1_known": protocol["source_v1_outputs_known"] is True,
        "source_audit_known": protocol["source_robustness_audit_known"] is True,
        "new_metrics_unseen": protocol[
            "conservative_scenario_metrics_computed_before_freeze"
        ]
        is False,
        "target": float(measurement["target_net_sharpe"]) == 1.2,
        "conservative_definition": measurement[
            "conservative_scenario_sharpe_definition"
        ]
        == "MIN_OF_MONTHLY_20D_INTERVAL_AND_LO_HAC",
        "lag": int(measurement["lo_bartlett_max_lag"]) == 20,
        "seed": int(measurement["random_seed"]) == 51030020260831,
        "repetitions": int(measurement["repetitions_per_random_cell"]) == 500,
        "joint_up": measurement["joint_grid"]["captured_bull_counts"]
        == [4, 8, 12, 16, 20, 23],
        "joint_range": measurement["joint_grid"]["false_range_counts"]
        == [0, 5, 10, 20, 40, 97],
        "joint_down": measurement["joint_grid"]["false_bear_counts"]
        == [0, 1, 2, 4, 8, 21],
        "timing": measurement["timing_error_days"] == list(range(11)),
        "minimum_prior_manifests": int(
            config["selection_bias"]["minimum_prior_manifest_count"]
        )
        == 357,
        "preexisting_up20_protocol_declared": config[
            "preexisting_up20_protocol"
        ]["state_at_acceptance_freeze"]
        == "FROZEN_BEFORE_MODEL_OUTCOME_CALCULATION",
        "preexisting_up20_result_absent": config["preexisting_up20_protocol"][
            "result_existed_at_acceptance_freeze"
        ]
        is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ContractError(f"V1.0.1保守验收配置异常：{failed}")
    rescue_fields = (
        "ordinary_result_rescue_after_result",
        "error_grid_rescue_after_result",
        "random_seed_rescue_after_result",
        "adversarial_rule_rescue_after_result",
        "timing_grid_rescue_after_result",
        "serial_metric_rescue_after_result",
        "advancement_gate_rescue_after_result",
    )
    if any(protocol[field] != "forbidden" for field in rescue_fields):
        raise ContractError("V1.0.1结果后救援开关没有全部冻结")
    required_fields = [
        "MIN_UP_RECALL_FOR_SHARPE_1_2",
        "MIN_UP_PRECISION_FOR_SHARPE_1_2",
        "MAX_DOWN_FALSE_LONG_RATE",
        "MAX_RANGE_FALSE_LONG_RATE",
        "MAX_ENTRY_DELAY_DAYS",
        "MAX_EXIT_DELAY_DAYS",
        "MIN_ORACLE_RETURN_CAPTURE",
        "SHARPE_1_2_FEASIBLE_REGION",
    ]
    if config["reported_budget_fields"] != required_fields:
        raise ContractError("V1.0.1必须交付的八个预算字段发生变化")
    governance = config["governance"]
    prohibited = (
        "realistic_prediction_model_evaluated",
        "atlas_router_created",
        "up20_prediction_model_created",
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_enabled",
        "order_generation",
        "broker_connection",
        "live_trading_authorized",
    )
    if any(bool(governance[field]) for field in prohibited):
        raise ContractError("V1.0.1不得创建模型、信号、仓位、订单、连接或实盘授权")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("V1.0.1保守验收冻结清单不存在")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("V1.0.1冻结清单项目编号不匹配")
    if manifest.get("state") != "FROZEN_BEFORE_CONSERVATIVE_SCENARIO_METRICS":
        raise ContractError("V1.0.1冻结清单状态不允许计算")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("V1.0.1冻结后配置发生漂移")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        corrected_relative = "research/oracle_information_budget_v1_0_1_acceptance.py"
        if set(mismatches) != {corrected_relative}:
            raise ContractError(f"V1.0.1冻结文件发生漂移：{mismatches}")
        if not PROGRAM_CORRECTION_MANIFEST_PATH.exists():
            raise ContractError("V1.0.1第一程序修正清单不存在")
        if not PROGRAM_CORRECTION_2_MANIFEST_PATH.exists():
            raise ContractError("V1.0.1第二程序修正清单不存在，禁止接受代码哈希变化")
        correction_1 = json.loads(
            PROGRAM_CORRECTION_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        correction = json.loads(
            PROGRAM_CORRECTION_2_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        correction_checks = {
            "project": correction.get("project_id") == PROJECT_ID,
            "state": correction.get("state")
            == "FROZEN_DETERMINISTIC_NAN_COMPARATOR_CORRECTION_BEFORE_RERUN",
            "base_manifest": correction.get("base_manifest_sha256")
            == sha256_file(MANIFEST_PATH),
            "prior_correction_manifest": correction.get(
                "prior_correction_manifest_sha256"
            )
            == sha256_file(PROGRAM_CORRECTION_MANIFEST_PATH),
            "old_code": correction.get("old_code_sha256")
            == correction_1.get("corrected_code_sha256"),
            "new_code": correction.get("corrected_code_sha256")
            == mismatches[corrected_relative]["actual"],
            "scope": correction.get("correction_scope")
            == "DETERMINISTIC_AUDIT_COMPARATOR_TREATS_NAN_AS_EQUAL_ONLY_WHEN_BOTH_SIDES_ARE_NAN",
            "no_outputs": correction.get("formal_outputs_existed_before_correction")
            is False,
        }
        failed_correction = sorted(
            name for name, passed in correction_checks.items() if not passed
        )
        if failed_correction:
            raise ContractError(
                f"V1.0.1程序修正清单验证失败：{failed_correction}"
            )
    if manifest.get("source_v1_outputs_known") is not True:
        raise ContractError("V1.0.1没有如实声明原V1结果已知")
    if manifest.get("source_robustness_audit_known") is not True:
        raise ContractError("V1.0.1没有如实声明稳健性审计已知")
    if manifest.get("conservative_scenario_metrics_computed_before_freeze") is not False:
        raise ContractError("V1.0.1冻结前新保守场景指标可见性声明失败")
    return manifest


def verify_source_hashes(config: dict[str, Any]) -> None:
    failures: dict[str, dict[str, str]] = {}
    for section in ("source_v1", "source_atlas", "preexisting_up20_protocol"):
        for record in config[section].values():
            if not isinstance(record, dict) or "path" not in record:
                continue
            path = project_path(record["path"])
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != record["sha256"]:
                failures[record["path"]] = {
                    "expected": record["sha256"],
                    "actual": actual,
                }
    if failures:
        raise ContractError(f"V1.0.1源产物哈希不符合冻结合同：{failures}")


def _annualized_sharpe_matrix(
    returns: np.ndarray, periods_per_year: float
) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    means = np.mean(values, axis=1)
    volatility = np.std(values, axis=1, ddof=1)
    return np.divide(
        means * math.sqrt(periods_per_year),
        volatility,
        out=np.full(len(values), np.nan, dtype=float),
        where=volatility > 0,
    )


def simulate_dense_signals_with_conservative_metrics(
    signals: np.ndarray,
    context: base.StudyContext,
    base_config: dict[str, Any],
    acceptance_config: dict[str, Any],
) -> pd.DataFrame:
    """精确复刻原批量账户，并保存日收益以计算逐场景保守夏普。"""

    signal_array = np.asarray(signals, dtype=np.int8)
    engine = context.engine
    if signal_array.ndim != 2 or signal_array.shape[1] != len(engine.dates):
        raise ValueError("保守验收信号矩阵形状不正确")
    if not np.isin(signal_array, [-1, 0, 1]).all():
        raise ValueError("保守验收信号只能取-1、0、1")
    scenario_count, observation_count = signal_array.shape
    initial_cash = float(base_config["execution"]["initial_capital_cny"])
    costs: BacktestCosts = base._stress_costs(base_config)
    cash = np.full(scenario_count, initial_cash, dtype=float)
    shares = np.zeros(scenario_count, dtype=np.int64)
    dividend_receivable = np.zeros(scenario_count, dtype=float)
    current_target = np.zeros(scenario_count, dtype=np.int8)
    previous_equity = np.full(scenario_count, initial_cash, dtype=float)
    equity_peak = previous_equity.copy()
    maximum_drawdown = np.zeros(scenario_count, dtype=float)
    return_sum = np.zeros(scenario_count, dtype=float)
    return_square_sum = np.zeros(scenario_count, dtype=float)
    exposure_sum = np.zeros(scenario_count, dtype=float)
    trade_count = np.zeros(scenario_count, dtype=np.int32)
    total_explicit_cost = np.zeros(scenario_count, dtype=float)
    total_slippage_cost = np.zeros(scenario_count, dtype=float)
    daily_returns = np.zeros((scenario_count, observation_count), dtype=float)
    scheduled_payments: dict[int, np.ndarray] = {}
    buy_price_multiplier = 1.0 + costs.slippage_bps / 10000.0
    sell_price_multiplier = 1.0 - costs.slippage_bps / 10000.0

    for day_index in range(observation_count):
        shares_at_start = shares.copy()
        for payment_index, amount in engine.dividend_events_by_ex_index.get(
            day_index, ()
        ):
            entitlement = shares_at_start.astype(float) * amount
            dividend_receivable += entitlement
            if payment_index is not None:
                existing = scheduled_payments.get(payment_index)
                if existing is None:
                    scheduled_payments[payment_index] = entitlement.copy()
                else:
                    existing += entitlement

        if day_index == 0:
            trade_allowed = np.zeros(scenario_count, dtype=bool)
        else:
            previous_signal = signal_array[:, day_index - 1]
            trade_allowed = previous_signal >= 0
            current_target = np.where(
                trade_allowed, previous_signal, current_target
            ).astype(np.int8)

        open_price = float(engine.opens[day_index])
        close_price = float(engine.closes[day_index])
        open_equity = cash + shares.astype(float) * open_price + dividend_receivable
        buy_price = open_price * buy_price_multiplier
        sell_price = open_price * sell_price_multiplier
        target_value = current_target.astype(float) * open_equity
        reference_price = np.where(
            target_value >= shares.astype(float) * open_price,
            buy_price,
            sell_price,
        )
        desired_shares = (
            np.floor(target_value / reference_price / costs.lot_size).astype(np.int64)
            * costs.lot_size
        )
        desired_shares = np.maximum(desired_shares, 0)
        quantity = desired_shares - shares
        quantity = np.where(trade_allowed, quantity, 0).astype(np.int64)
        commission = np.zeros(scenario_count, dtype=float)
        slippage_cost = np.zeros(scenario_count, dtype=float)

        buy_mask = quantity > 0
        while buy_mask.any():
            notional_all = quantity.astype(float) * buy_price
            candidate_commission = np.maximum(
                costs.minimum_commission_cny,
                notional_all * costs.commission_rate,
            )
            unaffordable = buy_mask & (
                notional_all + candidate_commission > cash + 1e-9
            )
            if not unaffordable.any():
                break
            quantity[unaffordable] -= costs.lot_size
            buy_mask = quantity > 0
        if buy_mask.any():
            notional = quantity[buy_mask].astype(float) * buy_price
            commission[buy_mask] = np.maximum(
                costs.minimum_commission_cny,
                notional * costs.commission_rate,
            )
            cash[buy_mask] -= notional + commission[buy_mask]
            shares[buy_mask] += quantity[buy_mask]
            slippage_cost[buy_mask] = (
                buy_price - open_price
            ) * quantity[buy_mask].astype(float)

        sell_mask = quantity < 0
        if sell_mask.any():
            sell_quantity = np.minimum(-quantity, shares_at_start)
            sell_quantity = (
                sell_quantity // costs.lot_size * costs.lot_size
            ).astype(np.int64)
            sell_mask = sell_mask & (sell_quantity > 0)
            if sell_mask.any():
                notional = sell_quantity[sell_mask].astype(float) * sell_price
                commission[sell_mask] = np.maximum(
                    costs.minimum_commission_cny,
                    notional * costs.commission_rate,
                )
                stamp_duty = notional * costs.stamp_duty_rate
                cash[sell_mask] += notional - commission[sell_mask] - stamp_duty
                shares[sell_mask] -= sell_quantity[sell_mask]
                slippage_cost[sell_mask] = (
                    open_price - sell_price
                ) * sell_quantity[sell_mask].astype(float)
                commission[sell_mask] += stamp_duty

        executed = buy_mask | sell_mask
        trade_count += executed.astype(np.int32)
        total_explicit_cost += commission
        total_slippage_cost += slippage_cost

        payment = scheduled_payments.pop(day_index, None)
        if payment is not None:
            cash += payment
            dividend_receivable -= payment
            dividend_receivable[np.abs(dividend_receivable) < 1e-9] = 0.0

        close_equity = cash + shares.astype(float) * close_price + dividend_receivable
        if day_index == 0:
            daily_return = np.zeros(scenario_count, dtype=float)
        else:
            daily_return = close_equity / previous_equity - 1.0
        daily_returns[:, day_index] = daily_return
        return_sum += daily_return
        return_square_sum += daily_return * daily_return
        equity_peak = np.maximum(equity_peak, close_equity)
        drawdown = close_equity / equity_peak - 1.0
        maximum_drawdown = np.minimum(maximum_drawdown, drawdown)
        exposure_sum += np.divide(
            shares.astype(float) * close_price,
            close_equity,
            out=np.zeros_like(close_equity),
            where=close_equity != 0,
        )
        previous_equity = close_equity

    variance = (
        return_square_sum - return_sum * return_sum / observation_count
    ) / (observation_count - 1)
    variance = np.maximum(variance, 0.0)
    daily_std = np.sqrt(variance)
    daily_mean = return_sum / observation_count
    annualized_volatility = daily_std * math.sqrt(242.0)
    ordinary_sharpe = np.divide(
        daily_mean * 242.0,
        annualized_volatility,
        out=np.full(scenario_count, np.nan, dtype=float),
        where=annualized_volatility > 0,
    )
    total_return = previous_equity / initial_cash - 1.0
    elapsed_days = max((engine.dates[-1] - engine.dates[0]).days, 1)
    cagr = np.where(
        total_return > -1.0,
        np.power(1.0 + total_return, 365.25 / elapsed_days) - 1.0,
        -1.0,
    )

    measurement = acceptance_config["frozen_measurement"]
    centered = daily_returns - np.mean(daily_returns, axis=1, keepdims=True)
    gamma0 = np.mean(centered * centered, axis=1)
    long_run_variance = gamma0.copy()
    maximum_lag = int(measurement["lo_bartlett_max_lag"])
    for lag in range(1, maximum_lag + 1):
        covariance = np.mean(centered[:, lag:] * centered[:, :-lag], axis=1)
        weight = 1.0 - lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    lo_hac_sharpe = np.divide(
        daily_mean
        * math.sqrt(float(measurement["trading_days_per_year"])),
        np.sqrt(long_run_variance),
        out=np.full(scenario_count, np.nan, dtype=float),
        where=long_run_variance > 0,
    )

    safe_returns = np.clip(daily_returns, -0.999999999999, None)
    log_returns = np.log1p(safe_returns)
    month_labels = np.asarray(engine.dates.to_period("M").astype(str))
    month_starts = np.flatnonzero(
        np.concatenate(([True], month_labels[1:] != month_labels[:-1]))
    )
    monthly_log_returns = np.add.reduceat(log_returns, month_starts, axis=1)
    monthly_returns = np.expm1(monthly_log_returns)
    monthly_sharpe = _annualized_sharpe_matrix(
        monthly_returns, float(measurement["monthly_periods_per_year"])
    )

    complete_blocks = context.blocks.loc[context.blocks["complete"]]
    interval_indices = np.vstack(
        [
            np.arange(int(anchor) + 1, int(anchor) + 21, dtype=int)
            for anchor in complete_blocks["anchor_position"]
        ]
    )
    interval_log_returns = log_returns[:, interval_indices].sum(axis=2)
    interval_returns = np.expm1(interval_log_returns)
    interval_sharpe = _annualized_sharpe_matrix(
        interval_returns,
        float(measurement["trading_days_per_year"])
        / float(measurement["interval_days"]),
    )
    conservative_sharpe = np.min(
        np.vstack((monthly_sharpe, interval_sharpe, lo_hac_sharpe)), axis=0
    )

    return pd.DataFrame(
        {
            "stress_net_sharpe": ordinary_sharpe,
            "monthly_compounded_return_sharpe": monthly_sharpe,
            "twenty_day_interval_return_sharpe": interval_sharpe,
            "lo_hac_adjusted_sharpe_q20": lo_hac_sharpe,
            "conservative_scenario_sharpe": conservative_sharpe,
            "stress_total_return": total_return,
            "stress_cagr": cagr,
            "stress_annualized_volatility": annualized_volatility,
            "stress_max_drawdown": maximum_drawdown,
            "average_exposure": exposure_sum / observation_count,
            "trade_count": trade_count.astype(int),
            "ending_equity": previous_equity,
            "total_explicit_cost_cny": total_explicit_cost,
            "total_slippage_cost_cny": total_slippage_cost,
            "total_execution_cost_cny": total_explicit_cost + total_slippage_cost,
        }
    )


def _attach_classification_and_gates(
    metrics: pd.DataFrame,
    context: base.StudyContext,
    base_config: dict[str, Any],
    acceptance_config: dict[str, Any],
) -> pd.DataFrame:
    output = metrics.copy()
    source_stress = context.source_result["threshold_results"]["0.05"]["policies"][
        "STATE_ROUTER_RANGE_NO_T"
    ]["stress"]
    oracle_log_wealth = math.log1p(float(source_stress["total_return"]))
    output["oracle_net_log_wealth_capture"] = np.log1p(
        output["stress_total_return"].clip(lower=-0.999999999999)
    ) / oracle_log_wealth
    all_long_mdd = abs(
        float(
            context.source_result["threshold_results"]["0.05"]["policies"][
                "ALL_LONG"
            ]["stress"]["max_drawdown"]
        )
    )
    drawdown_limit = (
        float(
            acceptance_config["frozen_measurement"][
                "maximum_drawdown_ratio_vs_source_all_long"
            ]
        )
        * all_long_mdd
    )
    target = float(acceptance_config["frozen_measurement"]["target_net_sharpe"])
    output["drawdown_limit"] = drawdown_limit
    output["ordinary_sharpe_gate_pass"] = output["stress_net_sharpe"] >= target
    output["conservative_sharpe_gate_pass"] = (
        output["conservative_scenario_sharpe"] >= target
    )
    output["positive_return_gate_pass"] = output["stress_total_return"] > 0.0
    output["drawdown_gate_pass"] = (
        output["stress_max_drawdown"].abs() <= drawdown_limit
    )
    output["conservative_deterministic_budget_pass"] = (
        output["conservative_sharpe_gate_pass"]
        & output["positive_return_gate_pass"]
        & output["drawdown_gate_pass"]
    )
    return output


RANDOM_METRIC_COLUMNS = (
    "stress_net_sharpe",
    "monthly_compounded_return_sharpe",
    "twenty_day_interval_return_sharpe",
    "lo_hac_adjusted_sharpe_q20",
    "conservative_scenario_sharpe",
    "stress_total_return",
    "stress_cagr",
    "stress_max_drawdown",
    "average_exposure",
    "trade_count",
    "oracle_net_log_wealth_capture",
)


def _summarize_random_cell(
    family: str,
    cell_id: str,
    classification: dict[str, Any],
    draws: pd.DataFrame,
    acceptance_config: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "cell_id": cell_id,
        "repetitions": int(len(draws)),
        **classification,
    }
    for column in RANDOM_METRIC_COLUMNS:
        values = draws[column].to_numpy(dtype=float)
        row[f"{column}_q05"] = float(np.quantile(values, 0.05))
        row[f"{column}_median"] = float(np.quantile(values, 0.50))
        row[f"{column}_q95"] = float(np.quantile(values, 0.95))
    row["probability_ordinary_sharpe_at_least_1_2"] = float(
        draws["ordinary_sharpe_gate_pass"].mean()
    )
    row["probability_conservative_sharpe_at_least_1_2"] = float(
        draws["conservative_sharpe_gate_pass"].mean()
    )
    row["probability_positive_return"] = float(
        draws["positive_return_gate_pass"].mean()
    )
    row["probability_drawdown_within_limit"] = float(
        draws["drawdown_gate_pass"].mean()
    )
    measurement = acceptance_config["frozen_measurement"]
    row["random_conservative_budget_pass"] = bool(
        row["conservative_scenario_sharpe_q05"]
        >= float(measurement["target_net_sharpe"])
        and row["probability_conservative_sharpe_at_least_1_2"]
        >= float(measurement["random_pass_probability_minimum"])
        and row["stress_total_return_q05"] > 0.0
        and row["probability_drawdown_within_limit"]
        >= float(measurement["random_drawdown_pass_probability_minimum"])
    )
    return row


def _assert_reproduces_v1_random(
    corrected: dict[str, Any], source: pd.Series
) -> None:
    columns = (
        "stress_net_sharpe_q05",
        "stress_net_sharpe_median",
        "stress_net_sharpe_q95",
        "stress_total_return_q05",
        "stress_total_return_median",
        "stress_total_return_q95",
        "stress_max_drawdown_q05",
        "stress_max_drawdown_median",
        "stress_max_drawdown_q95",
        "average_exposure_q05",
        "average_exposure_median",
        "average_exposure_q95",
        "oracle_net_log_wealth_capture_q05",
        "oracle_net_log_wealth_capture_median",
        "oracle_net_log_wealth_capture_q95",
    )
    failures: dict[str, dict[str, float]] = {}
    for column in columns:
        corrected_value = float(corrected[column])
        source_value = float(source[column])
        both_nan = math.isnan(corrected_value) and math.isnan(source_value)
        if both_nan:
            continue
        if not math.isclose(
            corrected_value,
            source_value,
            rel_tol=0.0,
            abs_tol=5e-12,
        ):
            failures[column] = {
                "corrected": corrected_value,
                "source": source_value,
            }
    if failures:
        raise ContractError(
            f"保守验收未能复刻V1随机单元{corrected['cell_id']}：{failures}"
        )


def _random_cells(base_config: dict[str, Any]) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    random_cfg = base_config["random_error_budget"]
    axis_cells: list[tuple[Any, ...]] = []
    for captured in range(
        int(random_cfg["recall_axis"]["captured_bull_count_start"]),
        int(random_cfg["recall_axis"]["captured_bull_count_end"]) + 1,
    ):
        axis_cells.append(("RECALL_ONLY", f"UP_{captured:02d}", captured, 0, 0, 101))
    for false_range in range(
        int(random_cfg["range_axis"]["false_range_count_start"]),
        int(random_cfg["range_axis"]["false_range_count_end"]) + 1,
    ):
        axis_cells.append(
            (
                "RANGE_CONTAMINATION",
                f"RANGE_{false_range:02d}",
                23,
                false_range,
                0,
                102,
            )
        )
    for false_bear in range(
        int(random_cfg["down_axis"]["false_bear_count_start"]),
        int(random_cfg["down_axis"]["false_bear_count_end"]) + 1,
    ):
        axis_cells.append(
            (
                "DOWN_CONTAMINATION",
                f"DOWN_{false_bear:02d}",
                23,
                0,
                false_bear,
                103,
            )
        )
    joint_cells: list[tuple[Any, ...]] = []
    for captured in random_cfg["joint_grid"]["captured_bull_counts"]:
        for false_range in random_cfg["joint_grid"]["false_range_counts"]:
            for false_bear in random_cfg["joint_grid"]["false_bear_counts"]:
                joint_cells.append(
                    (
                        "JOINT_GRID",
                        f"UP_{captured:02d}_R_{false_range:02d}_D_{false_bear:02d}",
                        int(captured),
                        int(false_range),
                        int(false_bear),
                        104,
                    )
                )
    return axis_cells, joint_cells


def run_random_conservative_experiments(
    context: base.StudyContext,
    base_config: dict[str, Any],
    acceptance_config: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis_cells, joint_cells = _random_cells(base_config)
    source_axis = pd.read_parquet(
        project_path(acceptance_config["source_v1"]["random_axis_summary"]["path"])
    ).set_index("cell_id")
    source_joint = pd.read_parquet(
        project_path(acceptance_config["source_v1"]["random_joint_summary"]["path"])
    ).set_index("cell_id")
    repetitions = int(
        acceptance_config["frozen_measurement"]["repetitions_per_random_cell"]
    )
    seed = int(acceptance_config["frozen_measurement"]["random_seed"])
    bull_positions = np.flatnonzero(context.states == STATE_BULL)
    range_positions = np.flatnonzero(context.states == STATE_RANGE)
    bear_positions = np.flatnonzero(context.states == STATE_BEAR)

    def execute(
        cells: list[tuple[Any, ...]], source: pd.DataFrame, label: str
    ) -> pd.DataFrame:
        summaries: list[dict[str, Any]] = []
        for index, (
            family,
            cell_id,
            captured,
            false_range,
            false_bear,
            family_code,
        ) in enumerate(cells, start=1):
            predictions = base.sample_exact_predictions(
                len(context.blocks),
                bull_positions,
                range_positions,
                bear_positions,
                int(captured),
                int(false_range),
                int(false_bear),
                repetitions,
                base._seed_sequence(
                    seed,
                    int(family_code),
                    int(captured),
                    int(false_range),
                    int(false_bear),
                ),
            )
            signals = base.predictions_to_dense_signals(predictions, context)
            metrics = simulate_dense_signals_with_conservative_metrics(
                signals, context, base_config, acceptance_config
            )
            metrics = _attach_classification_and_gates(
                metrics, context, base_config, acceptance_config
            )
            classification = base._classification_fields(
                int(captured), int(false_range), int(false_bear), context
            )
            summary = _summarize_random_cell(
                str(family), str(cell_id), classification, metrics, acceptance_config
            )
            _assert_reproduces_v1_random(summary, source.loc[str(cell_id)])
            summaries.append(summary)
            if progress is not None and (index % 20 == 0 or index == len(cells)):
                progress(f"{label}保守场景：{index}/{len(cells)}单元完成")
        return pd.DataFrame(summaries)

    return (
        execute(axis_cells, source_axis, "随机逐轴"),
        execute(joint_cells, source_joint, "随机联合"),
    )


def _adversarial_cells(base_config: dict[str, Any]) -> list[tuple[str, str, int, int, int]]:
    cells: list[tuple[str, str, int, int, int]] = []
    for captured in range(24):
        cells.append(("RECALL_ONLY", f"UP_{captured:02d}", captured, 0, 0))
    for false_range in range(98):
        cells.append(
            ("RANGE_CONTAMINATION", f"RANGE_{false_range:02d}", 23, false_range, 0)
        )
    for false_bear in range(22):
        cells.append(
            ("DOWN_CONTAMINATION", f"DOWN_{false_bear:02d}", 23, 0, false_bear)
        )
    random_cfg = base_config["random_error_budget"]
    for captured in random_cfg["joint_grid"]["captured_bull_counts"]:
        for false_range in random_cfg["joint_grid"]["false_range_counts"]:
            for false_bear in random_cfg["joint_grid"]["false_bear_counts"]:
                cells.append(
                    (
                        "JOINT_GRID",
                        f"UP_{captured:02d}_R_{false_range:02d}_D_{false_bear:02d}",
                        int(captured),
                        int(false_range),
                        int(false_bear),
                    )
                )
    return cells


def _assert_reproduces_v1_deterministic(
    corrected: pd.DataFrame,
    source: pd.DataFrame,
    keys: list[str],
) -> None:
    columns = (
        "stress_net_sharpe",
        "stress_total_return",
        "stress_cagr",
        "stress_max_drawdown",
        "average_exposure",
        "trade_count",
        "oracle_net_log_wealth_capture",
    )
    left = corrected.sort_values(keys).reset_index(drop=True)
    right = source.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ContractError(f"V1确定性场景键或行数无法复刻：{keys}")
    failures: dict[str, float] = {}
    for column in columns:
        if column == "trade_count":
            passed = left[column].astype(int).equals(right[column].astype(int))
            difference = float(
                np.max(np.abs(left[column].to_numpy() - right[column].to_numpy()))
            )
        else:
            left_values = left[column].to_numpy(dtype=float)
            right_values = right[column].to_numpy(dtype=float)
            both_nan = np.isnan(left_values) & np.isnan(right_values)
            one_nan_only = np.isnan(left_values) ^ np.isnan(right_values)
            finite_differences = np.abs(left_values[~both_nan] - right_values[~both_nan])
            difference = (
                float(np.max(finite_differences))
                if len(finite_differences) > 0
                else 0.0
            )
            passed = not bool(one_nan_only.any()) and difference <= 5e-12
        if not passed:
            failures[column] = difference
    if failures:
        raise ContractError(f"V1确定性场景账户复刻失败：{failures}")


def run_adversarial_conservative_experiments(
    context: base.StudyContext,
    base_config: dict[str, Any],
    acceptance_config: dict[str, Any],
) -> pd.DataFrame:
    basis = pd.read_parquet(
        project_path(acceptance_config["source_v1"]["label_execution_basis"]["path"])
    )
    indexed = basis.set_index("block_id")
    executable = "etf_executable_open_to_next_execution_open_total_return"
    block_id_to_position = {
        int(block_id): position
        for position, block_id in enumerate(context.blocks["block_id"].astype(int))
    }

    def order_for(state: str) -> np.ndarray:
        frame = indexed.loc[indexed["state"].eq(state)].sort_values(
            executable, ascending=True
        )
        return np.array(
            [block_id_to_position[int(block_id)] for block_id in frame.index],
            dtype=int,
        )

    bull_order = order_for(STATE_BULL)
    range_order = order_for(STATE_RANGE)
    bear_order = order_for(STATE_BEAR)
    cells = _adversarial_cells(base_config)
    predictions = np.vstack(
        [
            base._nested_adversarial_predictions(
                len(context.blocks),
                bull_order,
                captured,
                range_order,
                false_range,
                bear_order,
                false_bear,
            )
            for _, _, captured, false_range, false_bear in cells
        ]
    )
    signals = base.predictions_to_dense_signals(predictions, context)
    metrics = simulate_dense_signals_with_conservative_metrics(
        signals, context, base_config, acceptance_config
    )
    metadata = pd.DataFrame(
        [
            {
                "mode": "ADVERSARIAL",
                "family": family,
                "cell_id": cell_id,
                **base._classification_fields(
                    captured, false_range, false_bear, context
                ),
            }
            for family, cell_id, captured, false_range, false_bear in cells
        ]
    )
    output = _attach_classification_and_gates(
        pd.concat([metadata, metrics], axis=1),
        context,
        base_config,
        acceptance_config,
    )
    source = pd.read_parquet(
        project_path(acceptance_config["source_v1"]["adversarial_summary"]["path"])
    )
    _assert_reproduces_v1_deterministic(output, source, ["family", "cell_id"])
    return output


def run_timing_conservative_experiments(
    context: base.StudyContext,
    base_config: dict[str, Any],
    acceptance_config: dict[str, Any],
) -> pd.DataFrame:
    events = base.build_bull_events(context)
    timing_cfg = base_config["timing_and_persistence_budget"]
    definitions = (
        ("ENTRY_DELAY", timing_cfg["entry_delay_days"]),
        ("EARLY_EXIT", timing_cfg["early_exit_days"]),
        ("LATE_EXIT", timing_cfg["late_exit_days"]),
    )
    signals: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for family, values in definitions:
        for value in values:
            signals.append(
                base.build_timing_signal(context, events, str(family), int(value))
            )
            metadata.append(
                {
                    "mode": "DETERMINISTIC_TIMING",
                    "family": str(family),
                    "error_days": int(value),
                    "continuous_bull_event_count": int(len(events)),
                }
            )
    metrics = simulate_dense_signals_with_conservative_metrics(
        np.vstack(signals), context, base_config, acceptance_config
    )
    output = _attach_classification_and_gates(
        pd.concat([pd.DataFrame(metadata), metrics], axis=1),
        context,
        base_config,
        acceptance_config,
    )
    source = pd.read_parquet(
        project_path(acceptance_config["source_v1"]["timing_summary"]["path"])
    )
    _assert_reproduces_v1_deterministic(output, source, ["family", "error_days"])
    return output


def _prefix_maximum_passing_count(
    frame: pd.DataFrame, count_column: str, pass_column: str
) -> int | None:
    ordered = frame.sort_values(count_column)
    maximum: int | None = None
    for row in ordered.itertuples(index=False):
        if not bool(getattr(row, pass_column)):
            break
        maximum = int(getattr(row, count_column))
    return maximum


def _suffix_minimum_passing_count(
    frame: pd.DataFrame, count_column: str, pass_column: str
) -> int | None:
    ordered = frame.sort_values(count_column, ascending=False)
    minimum: int | None = None
    for row in ordered.itertuples(index=False):
        if not bool(getattr(row, pass_column)):
            break
        minimum = int(getattr(row, count_column))
    return minimum


def _timing_boundary(timing: pd.DataFrame, family: str) -> dict[str, Any]:
    frame = timing.loc[timing["family"].eq(family)].sort_values("error_days")
    maximum = _prefix_maximum_passing_count(
        frame, "error_days", "conservative_deterministic_budget_pass"
    )
    grid = frame["error_days"].astype(int).tolist()
    all_grid_passed = bool(
        frame["conservative_deterministic_budget_pass"].astype(bool).all()
    )
    if maximum is None:
        return {
            "point_estimate_days": None,
            "lower_bound_days": None,
            "display": "NO_PASS_AT_ZERO",
            "boundary_status": "NOT_ESTABLISHED",
            "tested_grid_days": grid,
        }
    if all_grid_passed:
        return {
            "point_estimate_days": None,
            "lower_bound_days": int(maximum),
            "display": f">={maximum}",
            "boundary_status": "RIGHT_CENSORED_AT_TEST_GRID_MAXIMUM",
            "tested_grid_days": grid,
        }
    return {
        "point_estimate_days": int(maximum),
        "lower_bound_days": int(maximum),
        "display": str(maximum),
        "boundary_status": "IDENTIFIED_WITHIN_TEST_GRID",
        "tested_grid_days": grid,
    }


def build_feasible_region(
    random_joint: pd.DataFrame, adversarial: pd.DataFrame
) -> pd.DataFrame:
    random_columns = [
        "cell_id",
        "captured_bull_count",
        "false_range_count",
        "false_bear_count",
        "up_recall",
        "up_precision",
        "range_false_long_rate",
        "down_false_long_rate",
        "conservative_scenario_sharpe_q05",
        "conservative_scenario_sharpe_median",
        "probability_conservative_sharpe_at_least_1_2",
        "random_conservative_budget_pass",
    ]
    adversarial_joint = adversarial.loc[adversarial["family"].eq("JOINT_GRID")]
    adversarial_columns = [
        "cell_id",
        "conservative_scenario_sharpe",
        "monthly_compounded_return_sharpe",
        "twenty_day_interval_return_sharpe",
        "lo_hac_adjusted_sharpe_q20",
        "conservative_deterministic_budget_pass",
    ]
    output = random_joint[random_columns].merge(
        adversarial_joint[adversarial_columns],
        on="cell_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_random", "_adversarial"),
    )
    if len(output) != 216:
        raise ContractError(f"保守联合可行区域应有216格，实际{len(output)}格")
    output = output.rename(
        columns={
            "conservative_scenario_sharpe_q05": "random_conservative_sharpe_q05",
            "conservative_scenario_sharpe_median": "random_conservative_sharpe_median",
            "conservative_scenario_sharpe": "adversarial_conservative_sharpe",
            "monthly_compounded_return_sharpe": "adversarial_monthly_sharpe",
            "twenty_day_interval_return_sharpe": "adversarial_twenty_day_sharpe",
            "lo_hac_adjusted_sharpe_q20": "adversarial_lo_hac_sharpe_q20",
            "conservative_deterministic_budget_pass": "adversarial_conservative_budget_pass",
        }
    )
    output["conservative_intersection_pass"] = (
        output["random_conservative_budget_pass"].astype(bool)
        & output["adversarial_conservative_budget_pass"].astype(bool)
    )
    return output.sort_values(
        ["captured_bull_count", "false_range_count", "false_bear_count"]
    ).reset_index(drop=True)


def _precision_record(frame: pd.DataFrame, pass_column: str) -> dict[str, Any] | None:
    passing = frame.loc[
        frame[pass_column].astype(bool) & frame["up_precision"].notna()
    ].sort_values(
        ["up_precision", "up_recall", "down_false_long_rate"],
        ascending=[True, True, True],
    )
    if passing.empty:
        return None
    row = passing.iloc[0]
    return {
        key: _json_scalar(row[key])
        for key in (
            "cell_id",
            "up_precision",
            "up_recall",
            "range_false_long_rate",
            "down_false_long_rate",
            "captured_bull_count",
            "false_range_count",
            "false_bear_count",
        )
    }


def derive_information_budget(
    random_axis: pd.DataFrame,
    random_joint: pd.DataFrame,
    adversarial: pd.DataFrame,
    timing: pd.DataFrame,
    feasible_region: pd.DataFrame,
) -> dict[str, Any]:
    random_recall = random_axis.loc[random_axis["family"].eq("RECALL_ONLY")]
    random_range = random_axis.loc[
        random_axis["family"].eq("RANGE_CONTAMINATION")
    ]
    random_down = random_axis.loc[
        random_axis["family"].eq("DOWN_CONTAMINATION")
    ]
    adversarial_recall = adversarial.loc[adversarial["family"].eq("RECALL_ONLY")]
    adversarial_range = adversarial.loc[
        adversarial["family"].eq("RANGE_CONTAMINATION")
    ]
    adversarial_down = adversarial.loc[
        adversarial["family"].eq("DOWN_CONTAMINATION")
    ]
    random_recall_count = _suffix_minimum_passing_count(
        random_recall, "captured_bull_count", "random_conservative_budget_pass"
    )
    adversarial_recall_count = _suffix_minimum_passing_count(
        adversarial_recall,
        "captured_bull_count",
        "conservative_deterministic_budget_pass",
    )
    random_range_count = _prefix_maximum_passing_count(
        random_range, "false_range_count", "random_conservative_budget_pass"
    )
    adversarial_range_count = _prefix_maximum_passing_count(
        adversarial_range,
        "false_range_count",
        "conservative_deterministic_budget_pass",
    )
    random_down_count = _prefix_maximum_passing_count(
        random_down, "false_bear_count", "random_conservative_budget_pass"
    )
    adversarial_down_count = _prefix_maximum_passing_count(
        adversarial_down,
        "false_bear_count",
        "conservative_deterministic_budget_pass",
    )
    entry = _timing_boundary(timing, "ENTRY_DELAY")
    exit_delay = _timing_boundary(timing, "LATE_EXIT")
    early_exit = _timing_boundary(timing, "EARLY_EXIT")

    capture_candidates: list[float] = []
    passed_random_recall = random_recall.loc[
        random_recall["random_conservative_budget_pass"].astype(bool),
        "oracle_net_log_wealth_capture_q05",
    ]
    capture_candidates.extend(float(value) for value in passed_random_recall)
    passed_adversarial_recall = adversarial_recall.loc[
        adversarial_recall["conservative_deterministic_budget_pass"].astype(bool),
        "oracle_net_log_wealth_capture",
    ]
    capture_candidates.extend(float(value) for value in passed_adversarial_recall)
    passed_timing = timing.loc[
        timing["conservative_deterministic_budget_pass"].astype(bool),
        "oracle_net_log_wealth_capture",
    ]
    capture_candidates.extend(float(value) for value in passed_timing)
    minimum_capture = min(capture_candidates) if capture_candidates else None

    random_passing = feasible_region.loc[
        feasible_region["random_conservative_budget_pass"].astype(bool)
    ]
    adversarial_passing = feasible_region.loc[
        feasible_region["adversarial_conservative_budget_pass"].astype(bool)
    ]
    intersection = feasible_region.loc[
        feasible_region["conservative_intersection_pass"].astype(bool)
    ]
    return {
        "definitions": {
            "conservative_scenario_sharpe": "月频、原Oracle完整20日区间与Lo/HAC(q=20)夏普三者最小值",
            "precision": "只能连同对应召回和两类误入率读取，不是可独立套用的通用标量阈值",
            "exit_delay": "上涨事件预定结束后继续持有；若冻结网格顶端仍通过则报告右删失下界",
        },
        "MIN_UP_RECALL_FOR_SHARPE_1_2": {
            "random_95pct_robust": {
                "captured_bull_blocks": random_recall_count,
                "rate": (
                    random_recall_count / 23
                    if random_recall_count is not None
                    else None
                ),
            },
            "adversarial": {
                "captured_bull_blocks": adversarial_recall_count,
                "rate": (
                    adversarial_recall_count / 23
                    if adversarial_recall_count is not None
                    else None
                ),
            },
        },
        "MIN_UP_PRECISION_FOR_SHARPE_1_2": {
            "random_95pct_robust_joint_grid": _precision_record(
                random_joint, "random_conservative_budget_pass"
            ),
            "adversarial_joint_grid": _precision_record(
                feasible_region, "adversarial_conservative_budget_pass"
            ),
            "conservative_intersection": _precision_record(
                feasible_region, "conservative_intersection_pass"
            ),
        },
        "MAX_DOWN_FALSE_LONG_RATE": {
            "random_95pct_robust": {
                "false_bear_blocks": random_down_count,
                "rate": random_down_count / 21 if random_down_count is not None else None,
            },
            "adversarial": {
                "false_bear_blocks": adversarial_down_count,
                "rate": (
                    adversarial_down_count / 21
                    if adversarial_down_count is not None
                    else None
                ),
            },
        },
        "MAX_RANGE_FALSE_LONG_RATE": {
            "random_95pct_robust": {
                "false_range_blocks": random_range_count,
                "rate": (
                    random_range_count / 97
                    if random_range_count is not None
                    else None
                ),
            },
            "adversarial": {
                "false_range_blocks": adversarial_range_count,
                "rate": (
                    adversarial_range_count / 97
                    if adversarial_range_count is not None
                    else None
                ),
            },
        },
        "MAX_ENTRY_DELAY_DAYS": entry,
        "MAX_EXIT_DELAY_DAYS": exit_delay,
        "EARLY_EXIT_TOLERANCE_DAYS_DIAGNOSTIC": early_exit,
        "MIN_ORACLE_RETURN_CAPTURE": minimum_capture,
        "SHARPE_1_2_FEASIBLE_REGION": {
            "grid_cell_count": int(len(feasible_region)),
            "random_95pct_robust_passing_cell_count": int(len(random_passing)),
            "adversarial_passing_cell_count": int(len(adversarial_passing)),
            "conservative_intersection_passing_cell_count": int(len(intersection)),
            "random_passing_cell_ids": random_passing["cell_id"].tolist(),
            "adversarial_passing_cell_ids": adversarial_passing["cell_id"].tolist(),
            "conservative_intersection_cell_ids": intersection["cell_id"].tolist(),
        },
    }


def adjudicate(
    budget: dict[str, Any], acceptance_config: dict[str, Any], source_result: dict[str, Any]
) -> dict[str, Any]:
    gate = acceptance_config["nondegenerate_advancement_gate"]
    recall = budget["MIN_UP_RECALL_FOR_SHARPE_1_2"]["adversarial"]
    down = budget["MAX_DOWN_FALSE_LONG_RATE"]["adversarial"]
    range_budget = budget["MAX_RANGE_FALSE_LONG_RATE"]["adversarial"]
    entry = budget["MAX_ENTRY_DELAY_DAYS"]
    source_serial = source_result["robustness"]["serial_correlation"]
    source_conservative = min(
        float(source_serial["monthly_compounded_return_sharpe"]),
        float(source_serial["twenty_day_interval_return_sharpe"]),
        float(source_serial["lo_adjusted_annualized_sharpe"]),
    )
    checks = {
        "perfect_oracle_conservative_sharpe_at_least_1_2": source_conservative
        >= 1.2,
        "adversarial_up_recall_not_near_perfect": recall["rate"] is not None
        and recall["rate"] <= float(gate["maximum_required_adversarial_up_recall"]),
        "adversarial_down_error_tolerance_nonzero": down["false_bear_blocks"]
        is not None
        and down["false_bear_blocks"]
        >= int(gate["minimum_adversarial_down_false_long_blocks_tolerated"]),
        "adversarial_range_error_tolerance_material": range_budget[
            "false_range_blocks"
        ]
        is not None
        and range_budget["false_range_blocks"]
        >= int(gate["minimum_adversarial_range_false_long_blocks_tolerated"]),
        "entry_delay_tolerance": entry["lower_bound_days"] is not None
        and entry["lower_bound_days"]
        >= int(gate["minimum_entry_delay_days_tolerated"]),
        "conservative_joint_region_nonempty": budget["SHARPE_1_2_FEASIBLE_REGION"][
            "conservative_intersection_passing_cell_count"
        ]
        > 0,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "nondegenerate_conservative_feasible_region": passed,
        "gates": checks,
        "source_ordinary_daily_sharpe": float(
            source_serial["ordinary_daily_annualized_sharpe"]
        ),
        "source_conservative_sharpe": source_conservative,
        "source_conservative_to_ordinary_ratio": source_conservative
        / float(source_serial["ordinary_daily_annualized_sharpe"]),
        "realistic_up20_forecast": "NOT_EVALUATED",
        "up20_protocol_freeze_eligibility": (
            "ELIGIBLE_NOT_CREATED"
            if passed
            else "STOPPED_ORACLE_VALUE_NOT_REALISTICALLY_ACCESSIBLE"
        ),
        "historical_tradable_strategy_target_achieved": False,
        "verified_forward_target_achieved": False,
        "position_output": "NONE",
        "live_trading_authorized": False,
    }


def build_near_miss_ledger(acceptance_config: dict[str, Any]) -> pd.DataFrame:
    leading = pd.read_csv(
        project_path(acceptance_config["source_atlas"]["leading_results"]["path"])
    )

    def exact(label: str, event: str, feature: str) -> pd.Series:
        selected = leading.loc[
            leading["label_system"].eq(label)
            & leading["event_type"].eq(event)
            & leading["feature"].eq(feature)
        ]
        if len(selected) != 1:
            raise ContractError(
                f"Atlas观察账本源行不唯一：{label}/{event}/{feature}"
            )
        return selected.iloc[0]

    hhi_start = exact(
        "LABEL_A_DIRECTIONAL_CHANGE", "UP_START", "contribution_hhi20"
    )
    hhi_end = exact(
        "LABEL_A_DIRECTIONAL_CHANGE", "UP_END", "contribution_hhi20"
    )
    price_b = exact("LABEL_B_FUTURE_PATH", "DOWN_START", "price_trend_z120")
    price_a = exact(
        "LABEL_A_DIRECTIONAL_CHANGE", "DOWN_START", "price_trend_z120"
    )
    common = {
        "status": "FORWARD_OBSERVATION_ONLY",
        "position_impact": 0,
        "current_signal": "NONE",
        "model_eligible": False,
        "combination_allowed": False,
        "window_change_allowed": False,
        "threshold_change_allowed": False,
        "label_change_allowed": False,
        "auto_promotion": False,
        "return_evaluation": "NOT_ALLOWED",
        "recheck_protocol": "ORIGINAL_V1_CODE_LABELS_WINDOWS_THRESHOLDS_ONLY",
        "recheck_trigger": "LABEL_B_INDEPENDENT_UP_EVENTS_AT_LEAST_6_AND_DOWN_EVENTS_AT_LEAST_6",
        "current_label_b_independent_up_events": 4,
        "current_label_b_independent_down_events": 5,
        "minimum_new_up_events_needed": 2,
        "minimum_new_down_events_needed": 1,
    }

    def source_fields(row: pd.Series) -> dict[str, Any]:
        return {
            "label_system": str(row["label_system"]),
            "event_type": str(row["event_type"]),
            "feature": str(row["feature"]),
            "module": str(row["module"]),
            "original_expected_direction": str(row["expected_direction"]),
            "event_observations": int(row["event_observations"]),
            "positive_direction_count": int(row["positive_direction_count"]),
            "direction_fraction": float(row["direction_fraction"]),
            "one_sided_sign_pvalue": float(row["one_sided_sign_pvalue"]),
            "median_expected_direction_z": float(
                row["median_expected_direction_z"]
            ),
            "median_onset_relative_day": float(row["median_onset_relative_day"]),
            "maximum_single_event_absolute_share": float(
                row["maximum_single_event_absolute_share"]
            ),
            "early_late_direction_agrees": bool(row["early_late_direction_agrees"]),
            "original_passed": bool(row["passed"]),
        }

    rows = [
        {
            "hypothesis_id": "ATLAS_NM_CONTRIBUTION_HHI20_UP_START_A",
            "original_window_days": 20,
            "failure_reasons": "MEDIAN_Z_BELOW_0_50;SIGN_PVALUE_ABOVE_0_05",
            "cross_label_replication": "FAILED",
            "cross_label_direction_fraction": None,
            "cross_label_sign_pvalue": None,
            "cross_label_median_expected_direction_z": None,
            **source_fields(hhi_start),
            **common,
        },
        {
            "hypothesis_id": "ATLAS_NM_CONTRIBUTION_HHI20_UP_END_A",
            "original_window_days": 20,
            "failure_reasons": "SIGN_PVALUE_ABOVE_0_05",
            "cross_label_replication": "FAILED",
            "cross_label_direction_fraction": None,
            "cross_label_sign_pvalue": None,
            "cross_label_median_expected_direction_z": None,
            **source_fields(hhi_end),
            **common,
        },
        {
            "hypothesis_id": "ATLAS_NM_PRICE_TREND_Z120_DOWN_START_B",
            "original_window_days": 120,
            "failure_reasons": "EVENT_COUNT_BELOW_6;CROSS_LABEL_NONREPLICATION",
            "cross_label_replication": "FAILED_LABEL_A_DOWN_START",
            "cross_label_direction_fraction": float(price_a["direction_fraction"]),
            "cross_label_sign_pvalue": float(price_a["one_sided_sign_pvalue"]),
            "cross_label_median_expected_direction_z": float(
                price_a["median_expected_direction_z"]
            ),
            **source_fields(price_b),
            **common,
        },
    ]
    return pd.DataFrame(rows)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def render_result_markdown(report: dict[str, Any]) -> str:
    budget = report["information_budget"]
    adjudication = report["adjudication"]
    recall = budget["MIN_UP_RECALL_FOR_SHARPE_1_2"]
    down = budget["MAX_DOWN_FALSE_LONG_RATE"]
    range_budget = budget["MAX_RANGE_FALSE_LONG_RATE"]
    precision = budget["MIN_UP_PRECISION_FOR_SHARPE_1_2"]
    feasible = budget["SHARPE_1_2_FEASIBLE_REGION"]

    def percent(value: Any) -> str:
        return "NOT_ESTABLISHED" if value is None else f"{float(value):.1%}"

    return "\n".join(
        [
            "# 510300 Oracle 信息预算 V1.0.1 保守验收",
            "",
            f"- 最终状态：`{report['status']}`",
            "- 证据等级：已知原V1结果后的保守合同修正；不是盲测、不是现实预测器回测。",
            f"- 完美Oracle普通日频夏普：`{adjudication['source_ordinary_daily_sharpe']:.3f}`。",
            f"- 完美Oracle保守夏普（三种调整口径最小值）：`{adjudication['source_conservative_sharpe']:.3f}`。",
            "",
            "## 八个冻结预算字段",
            "",
            "| 字段 | 随机95%稳健 | 对抗/保守 |",
            "|---|---:|---:|",
            f"| `MIN_UP_RECALL_FOR_SHARPE_1_2` | {percent(recall['random_95pct_robust']['rate'])} | {percent(recall['adversarial']['rate'])} |",
            f"| `MIN_UP_PRECISION_FOR_SHARPE_1_2` | {percent(None if precision['random_95pct_robust_joint_grid'] is None else precision['random_95pct_robust_joint_grid']['up_precision'])} | {percent(None if precision['conservative_intersection'] is None else precision['conservative_intersection']['up_precision'])} |",
            f"| `MAX_DOWN_FALSE_LONG_RATE` | {percent(down['random_95pct_robust']['rate'])} | {percent(down['adversarial']['rate'])} |",
            f"| `MAX_RANGE_FALSE_LONG_RATE` | {percent(range_budget['random_95pct_robust']['rate'])} | {percent(range_budget['adversarial']['rate'])} |",
            f"| `MAX_ENTRY_DELAY_DAYS` | — | {budget['MAX_ENTRY_DELAY_DAYS']['display']} |",
            f"| `MAX_EXIT_DELAY_DAYS` | — | {budget['MAX_EXIT_DELAY_DAYS']['display']} |",
            f"| `MIN_ORACLE_RETURN_CAPTURE` | — | {percent(budget['MIN_ORACLE_RETURN_CAPTURE'])} |",
            f"| `SHARPE_1_2_FEASIBLE_REGION` | {feasible['random_95pct_robust_passing_cell_count']}/{feasible['grid_cell_count']}格 | 交集{feasible['conservative_intersection_passing_cell_count']}/{feasible['grid_cell_count']}格 |",
            "",
            "精确率必须与同一单元的召回率、震荡误入率和大跌误入率共同读取。退出延迟若显示`>=10`，含义是冻结网格内右删失，不是已识别精确上限。",
            "",
            "## 裁决",
            "",
            f"- 非退化保守可行区域：`{str(adjudication['nondegenerate_conservative_feasible_region']).upper()}`。",
            f"- UP20协议资格：`{adjudication['up20_protocol_freeze_eligibility']}`。",
            "- 现实UP20预测器：`NOT_EVALUATED`。",
            "- 仓位输出：`NONE`；实盘授权：`FALSE`。",
            "",
        ]
    )


def render_project_state_markdown(state: dict[str, Any]) -> str:
    rows = [
        (key, value)
        for key, value in state["project_states"].items()
    ]
    lines = [
        "# 510300 Atlas 后项目状态 V1",
        "",
        "| 对象 | 正式状态 |",
        "|---|---|",
    ]
    lines.extend(f"| `{key}` | `{value}` |" for key, value in rows)
    lines.extend(
        [
            "",
            "- Atlas 原始冻结结果未改写；本文件只把完成态映射为最终治理裁决 `REJECTED_FROZEN`。",
            "- 两个 near-miss 只保留原方向、原窗口和原门槛，`POSITION_IMPACT=0`，不得组合或自动晋级。",
            "- 两套标签是方法论上独立定义，不是统计独立样本。",
            "- 未建立状态路由器、现实UP20模型、Paper/Shadow信号、仓位、订单或实盘连接。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(
    write: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    verify_source_hashes(config)
    base_config = base.load_config()
    base.validate_manifest(base_config)
    context = base.load_context(base_config)
    source_result = json.loads(
        project_path(config["source_v1"]["result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    atlas_result = json.loads(
        project_path(config["source_atlas"]["result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    notify: ProgressCallback | None = print if progress else None
    if notify is not None:
        notify("[1/6] 原V1与Atlas冻结哈希通过；开始随机逐轴保守夏普。")
    random_axis, random_joint = run_random_conservative_experiments(
        context, base_config, config, notify
    )
    if notify is not None:
        notify("[2/6] 随机场景完成；开始对抗性保守夏普。")
    adversarial = run_adversarial_conservative_experiments(
        context, base_config, config
    )
    if notify is not None:
        notify("[3/6] 对抗场景完成；开始进入与退出延迟保守夏普。")
    timing = run_timing_conservative_experiments(context, base_config, config)
    feasible = build_feasible_region(random_joint, adversarial)
    budget = derive_information_budget(
        random_axis, random_joint, adversarial, timing, feasible
    )
    adjudication = adjudicate(budget, config, source_result)
    near_miss = build_near_miss_ledger(config)
    status = (
        "CONSERVATIVE_INFORMATION_BUDGET_NONDEGENERATE_UP20_PROTOCOL_FREEZE_ELIGIBLE_NOT_TRADABLE"
        if adjudication["nondegenerate_conservative_feasible_region"]
        else "ORACLE_VALUE_NOT_REALISTICALLY_ACCESSIBLE_STOP_UP20_FORECAST_BRANCH"
    )
    if notify is not None:
        notify("[4/6] 八个预算字段与完整216格可行区域已导出。")

    report: dict[str, Any] = {
        "schema_version": "1.0.1",
        "project_id": PROJECT_ID,
        "status": status,
        "evidence_class": "POST_RESULT_CONSERVATIVE_CONTRACT_CORRECTION_NOT_BLIND_NOT_TRADABLE",
        "generated_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "source_v1_outputs_known": True,
            "source_robustness_audit_known": True,
            "conservative_scenario_metrics_computed_before_freeze": False,
            "program_correction_manifest_path": PROGRAM_CORRECTION_MANIFEST_PATH.relative_to(
                ROOT
            ).as_posix(),
            "program_correction_manifest_sha256": sha256_file(
                PROGRAM_CORRECTION_MANIFEST_PATH
            ),
            "program_correction_2_manifest_path": PROGRAM_CORRECTION_2_MANIFEST_PATH.relative_to(
                ROOT
            ).as_posix(),
            "program_correction_2_manifest_sha256": sha256_file(
                PROGRAM_CORRECTION_2_MANIFEST_PATH
            ),
        },
        "source_v1_boundary": {
            "source_status": source_result["status"],
            "source_result_unchanged": True,
            "source_ordinary_budget_is_preserved_but_not_accepted_as_conservative": True,
            "error_grid_unchanged": True,
            "random_seed_unchanged": True,
            "adversarial_rule_unchanged": True,
            "timing_grid_unchanged": True,
        },
        "preexisting_up20_protocol_boundary": {
            "project_id": "510300_UP20_RARE_EVENT_FORECAST_V1",
            "manifest_path": config["preexisting_up20_protocol"]["manifest"][
                "path"
            ],
            "manifest_sha256": config["preexisting_up20_protocol"]["manifest"][
                "sha256"
            ],
            "protocol_frozen_before_acceptance_correction": True,
            "model_outcome_existed_at_acceptance_freeze": False,
            "accepted_as_authorized_successor": False,
            "disposition": "QUARANTINED_PREMATURE_PROTOCOL_DO_NOT_RUN_FROM_THIS_ADJUDICATION",
        },
        "measurement": {
            "target_net_sharpe": 1.2,
            "conservative_scenario_sharpe_definition": config[
                "frozen_measurement"
            ]["conservative_scenario_sharpe_definition"],
            "ordinary_account_replay_tolerance": 5e-12,
            "random_repetitions_per_cell": 500,
            "joint_grid_cells": 216,
        },
        "oracle_robustness_audit": source_result["robustness"],
        "information_budget": budget,
        "adjudication": adjudication,
        "governance": {
            **config["governance"],
            "return_evaluation_for_realistic_model": "NOT_ALLOWED_NO_REALISTIC_MODEL",
            "current_signal": "NONE",
        },
    }

    paths = {key: project_path(value) for key, value in config["paths"].items()}
    if write:
        _atomic_parquet(paths["random_axis_summary"], random_axis)
        _atomic_parquet(paths["random_joint_summary"], random_joint)
        _atomic_parquet(paths["adversarial_summary"], adversarial)
        _atomic_parquet(paths["timing_summary"], timing)
        _atomic_csv(paths["feasible_region"], feasible)
        _atomic_csv(paths["near_miss_ledger"], near_miss)
        report["artifacts"] = {
            key: {
                "path": config["paths"][key],
                "bytes": paths[key].stat().st_size,
                "sha256": sha256_file(paths[key]),
            }
            for key in (
                "random_axis_summary",
                "random_joint_summary",
                "adversarial_summary",
                "timing_summary",
                "feasible_region",
                "near_miss_ledger",
            )
        }
        _atomic_json(paths["result_json"], report)
        _atomic_text(paths["result_markdown"], render_result_markdown(report))
        if notify is not None:
            notify("[5/6] 保守验收结果与观察账本已原子写入。")

        up20_state = (
            "ELIGIBLE_FOR_PROTOCOL_FREEZE_NOT_CREATED"
            if adjudication["nondegenerate_conservative_feasible_region"]
            else "STOPPED_ORACLE_VALUE_NOT_REALISTICALLY_ACCESSIBLE"
        )
        project_state = {
            "schema_version": "1.0.0",
            "project_id": "510300_POST_ATLAS_PROJECT_STATE_V1",
            "generated_at_asia_shanghai": datetime.now(
                ZoneInfo(config["protocol"]["timezone"])
            ).isoformat(),
            "source_hashes": {
                "atlas_result": sha256_file(
                    project_path(config["source_atlas"]["result"]["path"])
                ),
                "atlas_manifest": sha256_file(
                    project_path(config["source_atlas"]["manifest"]["path"])
                ),
                "information_budget_v1_result": sha256_file(
                    project_path(config["source_v1"]["result"]["path"])
                ),
                "conservative_acceptance_result": sha256_file(paths["result_json"]),
            },
            "atlas_source_status": atlas_result["status"],
            "project_states": {
                "510300_KNOWN_STATE_POLICY_ORACLE_V1_0_1": "CONDITIONAL_POLICY_VALUE_PASS",
                "510300_TREND_LIFECYCLE_ATLAS_V1": "REJECTED_FROZEN",
                "510300_TREND_LIFECYCLE_FACTOR_ROUTER_V1": "SKIPPED_NOT_CREATED",
                "ATLAS_FORWARD_MATURITY_MONITOR": "OBSERVATION_ONLY",
                "AUTO_PROMOTION_ON_EVENT_COUNT": "FALSE",
                "ATLAS_NEAR_MISS_HYPOTHESES": "FORWARD_OBSERVATION_ONLY",
                "HISTORICAL_OFFICIAL_WEIGHT_ARCHIVE": "DATA_FOUNDATION_SEPARATE",
                "TIME_VARYING_FACTOR_EFFECTIVENESS": "UNVERIFIED_HYPOTHESIS",
                "510300_ORACLE_INFORMATION_BUDGET_V1": "COMPLETED_ORDINARY_CANDIDATE_NOT_CONSERVATIVE_ACCEPTED",
                "510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE": status,
                "510300_UP20_DECISION_FORECAST_V1": up20_state,
                "510300_UP20_RARE_EVENT_FORECAST_V1": "QUARANTINED_PREMATURE_PROTOCOL_FROZEN_RESULT_NOT_RUN_NOT_ACCEPTED",
                "RETURN_EVALUATION_REALISTIC_MODEL": "NOT_ALLOWED_NO_REALISTIC_MODEL",
                "POSITION_OUTPUT": "NONE",
                "LIVE_TRADING_AUTHORIZED": "FALSE",
            },
            "label_independence": {
                "methodologically_independent_definitions": True,
                "statistically_independent_samples": False,
                "same_h00300_return_path": True,
            },
            "near_miss_ledger": report["artifacts"]["near_miss_ledger"],
            "forbidden_actions": [
                "ROUND_0_475_TO_0_50",
                "LOWER_EVENT_GATE_FROM_6_TO_5",
                "PROMOTE_SINGLE_LABEL",
                "COMBINE_CONTRIBUTION_HHI20_AND_PRICE_TREND_Z120",
                "TRAIN_ON_RETROSPECTIVE_SEVEN_STATE_PANEL",
                "CHANGE_LABEL_HORIZON",
                "CHANGE_LEAD_OR_BASELINE_WINDOW",
                "RELABEL_HISTORICAL_WEIGHTS_AS_STRICT_OFFICIAL_PIT",
                "RELABEL_FAIL_AS_INCONCLUSIVE",
            ],
            "governance": config["governance"],
        }
        _atomic_json(paths["project_state_json"], project_state)
        _atomic_text(
            paths["project_state_markdown"],
            render_project_state_markdown(project_state),
        )
        if notify is not None:
            notify("[6/6] Atlas最终裁决和全项目状态已固化。")
    return _json_safe(report)


__all__ = [
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "PROJECT_ID",
    "build_feasible_region",
    "derive_information_budget",
    "load_config",
    "run_study",
    "simulate_dense_signals_with_conservative_metrics",
    "validate_manifest",
]
