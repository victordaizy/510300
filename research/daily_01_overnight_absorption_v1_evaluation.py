"""DAILY_01冻结预测检验：标签、HAC、区块自助与一次性判定。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from research.daily_01_overnight_absorption_v1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "daily_01_overnight_absorption_v1_evaluation.yaml"


def load_evaluation_config(path: Path = CONFIG_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_parent_protocol(root: Path, evaluation_config: dict) -> dict:
    protocol = evaluation_config["evaluation_protocol"]
    manifest_path = root / protocol["parent_protocol_manifest"]
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != protocol["parent_protocol_manifest_sha256"]:
        raise ValueError("父协议清单哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in {**manifest["frozen_files"], **manifest["data_files"]}.items():
        path = root / relative
        if not path.exists() or sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"父协议冻结文件偏离：{mismatches}")
    if manifest.get("predictive_outcomes_computed") is not False:
        raise ValueError("父协议不是收益计算前冻结状态")
    return manifest


def _maximum_buy_quantity(
    cash: float,
    execution_price: float,
    lot_size: int,
    minimum_shares: int,
    commission_rate: float,
    minimum_commission: float,
) -> int:
    quantity = int(np.floor(cash / execution_price / lot_size) * lot_size)
    while quantity >= minimum_shares:
        notional = quantity * execution_price
        commission = max(notional * commission_rate, minimum_commission)
        if notional + commission <= cash + 1e-9:
            return quantity
        quantity -= lot_size
    return 0


def _label_one_horizon(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_index: int,
    horizon: int,
    parent_config: dict,
) -> dict[str, Any] | None:
    entry_index = signal_index + 1
    end_index = signal_index + horizon
    if end_index >= len(market):
        return None
    execution = parent_config["execution"]
    entry_row = market.iloc[entry_index]
    end_row = market.iloc[end_index]
    slippage = float(execution["slippage_bps_per_leg"]) / 10000.0
    buy_price = float(entry_row["open"]) * (1.0 + slippage)
    sell_price = float(end_row["close"]) * (1.0 - slippage)
    initial_cash = float(execution["initial_cash_cny"])
    commission_rate = float(execution["commission_rate_per_leg"])
    minimum_commission = float(execution["minimum_commission_cny_per_leg"])
    quantity = _maximum_buy_quantity(
        initial_cash,
        buy_price,
        int(execution["lot_size_shares"]),
        int(execution["minimum_ordinary_trade_shares"]),
        commission_rate,
        minimum_commission,
    )
    if quantity == 0:
        return None
    buy_notional = quantity * buy_price
    buy_commission = max(buy_notional * commission_rate, minimum_commission)
    residual_cash = initial_cash - buy_notional - buy_commission
    cash_growth = (1.0 + float(parent_config["prediction_contract"]["cash_annual_rate"])) ** (
        horizon / float(execution["trading_days_per_year"])
    )
    residual_cash_end = residual_cash * cash_growth

    entry_date = pd.Timestamp(entry_row["date"])
    end_date = pd.Timestamp(end_row["date"])
    entitled = dividends.loc[
        pd.to_datetime(dividends["record_date"]).between(entry_date, end_date)
    ]
    distribution_cash = float(entitled["cash_dividend_per_share"].sum()) * quantity
    sell_notional = quantity * sell_price
    sell_commission = max(sell_notional * commission_rate, minimum_commission)
    ending_wealth = residual_cash_end + distribution_cash + sell_notional - sell_commission
    net_account_return = ending_wealth / initial_cash - 1.0
    cash_return = cash_growth - 1.0
    return {
        "entry_date": entry_date,
        "end_date": end_date,
        "quantity": quantity,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "buy_commission": buy_commission,
        "sell_commission": sell_commission,
        "distribution_cash": distribution_cash,
        "net_account_return": net_account_return,
        "cash_return": cash_return,
        "net_excess_vs_cash": net_account_return - cash_return,
    }


def build_forward_labels(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    parent_config: dict,
    evaluation_config: dict,
) -> pd.DataFrame:
    """构造冻结D5/D20账户标签，不改变特征或信号。"""

    result = features.copy().sort_values("date").reset_index(drop=True)
    horizons = [int(value) for value in evaluation_config["labels"]["horizons_trading_days"]]
    for horizon in horizons:
        records = [
            _label_one_horizon(result, dividends, index, horizon, parent_config)
            for index in range(len(result))
        ]
        result[f"target_d{horizon}_net_excess"] = [
            np.nan if record is None else record["net_excess_vs_cash"] for record in records
        ]
        result[f"target_d{horizon}_entry_date"] = [
            pd.NaT if record is None else record["entry_date"] for record in records
        ]
        result[f"target_d{horizon}_end_date"] = [
            pd.NaT if record is None else record["end_date"] for record in records
        ]
    return result


def newey_west_regression(y: np.ndarray, x: np.ndarray, lag: int) -> dict:
    """计算OLS系数与Bartlett核Newey-West协方差。"""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(y) != len(x):
        raise ValueError("回归输入维度不一致")
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[valid]
    x = x[valid]
    if len(y) <= x.shape[1]:
        raise ValueError("回归有效样本不足")
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    scores = x * residual[:, None]
    meat = scores.T @ scores
    maximum_lag = min(int(lag), len(y) - 1)
    for offset in range(1, maximum_lag + 1):
        weight = 1.0 - offset / (maximum_lag + 1.0)
        gamma = scores[offset:].T @ scores[:-offset]
        meat += weight * (gamma + gamma.T)
    covariance = inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0,
    )
    return {
        "observations": int(len(y)),
        "coefficients": beta.tolist(),
        "standard_errors": standard_error.tolist(),
        "t_values": t_value.tolist(),
    }


def circular_block_bootstrap_difference(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    random_seed: int,
) -> dict:
    """对固定有利/不利分组执行循环移动块均值差自助法。"""

    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=int)
    valid = np.isfinite(values) & np.isin(groups, [-1, 0, 1])
    values = values[valid]
    groups = groups[valid]
    if not (np.any(groups == 1) and np.any(groups == -1)):
        raise ValueError("区块自助缺少有利组或不利组")
    generator = np.random.default_rng(random_seed)
    sample_size = len(values)
    differences: list[float] = []
    block_count = int(np.ceil(sample_size / block_length))
    offsets = np.arange(block_length)
    for _ in range(repetitions):
        starts = generator.integers(0, sample_size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % sample_size).ravel()[:sample_size]
        sampled_values = values[indices]
        sampled_groups = groups[indices]
        if np.any(sampled_groups == 1) and np.any(sampled_groups == -1):
            differences.append(
                float(sampled_values[sampled_groups == 1].mean() - sampled_values[sampled_groups == -1].mean())
            )
    if not differences:
        raise ValueError("区块自助未生成有效重复")
    array = np.asarray(differences)
    return {
        "valid_repetitions": int(len(array)),
        "lower_95": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.50)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def _group_summary(data: pd.DataFrame, target: str) -> dict:
    favorable = data.loc[data["evaluation_group"].eq(1), target]
    unfavorable = data.loc[data["evaluation_group"].eq(-1), target]
    return {
        "favorable_observations": int(favorable.notna().sum()),
        "unfavorable_observations": int(unfavorable.notna().sum()),
        "favorable_mean": float(favorable.mean()),
        "unfavorable_mean": float(unfavorable.mean()),
        "top_minus_bottom_mean": float(favorable.mean() - unfavorable.mean()),
        "favorable_positive_rate": float(favorable.gt(0).mean()),
        "unfavorable_positive_rate": float(unfavorable.gt(0).mean()),
    }


def evaluate_factor(labeled: pd.DataFrame, evaluation_config: dict) -> dict:
    """执行预注册因子门槛；不运行策略资产曲线。"""

    sample_config = evaluation_config["sample"]
    statistics = evaluation_config["statistics"]
    data = labeled.copy().sort_values("date").reset_index(drop=True)
    candidate = data["entry_sign_eligible"] & data["absorption_percentile"].notna()
    data["evaluation_group"] = 0
    data.loc[
        candidate
        & data["absorption_percentile"].ge(float(sample_config["favorable_group_percentile_at_least"])),
        "evaluation_group",
    ] = 1
    data.loc[
        candidate
        & data["absorption_percentile"].le(float(sample_config["unfavorable_group_percentile_at_most"])),
        "evaluation_group",
    ] = -1

    d20 = _group_summary(data, "target_d20_net_excess")
    d5 = _group_summary(data, "target_d5_net_excess")
    bootstrap_source = data.loc[data["target_d20_net_excess"].notna()].copy()
    bootstrap = circular_block_bootstrap_difference(
        bootstrap_source["target_d20_net_excess"].to_numpy(),
        bootstrap_source["evaluation_group"].to_numpy(),
        block_length=int(statistics["block_bootstrap_length_trading_days"]),
        repetitions=int(statistics["block_bootstrap_repetitions"]),
        random_seed=int(statistics["random_seed"]),
    )

    regression_sample = data.loc[
        candidate
        & data["target_d20_net_excess"].notna()
        & data["past_total_return_5d"].notna()
        & data["absorption_raw"].notna()
    ].copy()
    design = np.column_stack(
        [
            np.ones(len(regression_sample)),
            regression_sample["past_total_return_5d"].to_numpy(dtype=float),
            regression_sample["absorption_raw"].to_numpy(dtype=float),
        ]
    )
    regression = newey_west_regression(
        regression_sample["target_d20_net_excess"].to_numpy(dtype=float),
        design,
        int(statistics["hac_lag_trading_days"]),
    )
    regression["coefficient_names"] = ["constant", "h001_past_total_return_5d", "daily_01_absorption_raw"]

    chronological = regression_sample.sort_values("date").reset_index(drop=True)
    split = len(chronological) // 2
    half_summaries = []
    for name, half in (("first_half", chronological.iloc[:split]), ("second_half", chronological.iloc[split:])):
        half_grouped = half.loc[half["evaluation_group"].isin([-1, 1])]
        summary = _group_summary(half_grouped, "target_d20_net_excess")
        summary["id"] = name
        half_summaries.append(summary)

    mature_enter_events = int(
        (data["signal_action"].eq("ENTER") & data["target_d20_net_excess"].notna()).sum()
    )
    minimum_group = int(sample_config["minimum_group_observations"])
    gate_checks = {
        "minimum_mature_enter_events": mature_enter_events >= int(sample_config["minimum_mature_enter_events"]),
        "minimum_group_observations": d20["favorable_observations"] >= minimum_group and d20["unfavorable_observations"] >= minimum_group,
        "both_chronological_halves_positive": all(item["top_minus_bottom_mean"] > 0 for item in half_summaries),
        "d20_top_minus_bottom_positive": d20["top_minus_bottom_mean"] > 0,
        "d20_block_bootstrap_lower_bound_positive": bootstrap["lower_95"] > 0,
        "incremental_hac_coefficient_positive": regression["coefficients"][2] > 0,
        "d5_direction_not_reversed": d5["top_minus_bottom_mean"] >= 0,
    }
    passed = all(gate_checks.values())
    decision = evaluation_config["decision"]
    return {
        "status": decision["pass_status"] if passed else decision["fail_status"],
        "alpha_pass": False,
        "strategy_backtest_authorized": False,
        "sample": {
            "rows": int(len(data)),
            "candidate_rows": int(candidate.sum()),
            "mature_enter_events": mature_enter_events,
            "first_valid_candidate_date": None if not candidate.any() else str(pd.to_datetime(data.loc[candidate, "date"]).min().date()),
            "last_valid_candidate_date": None if not candidate.any() else str(pd.to_datetime(data.loc[candidate, "date"]).max().date()),
        },
        "d20": d20,
        "d5": d5,
        "chronological_halves": half_summaries,
        "block_bootstrap_d20_top_minus_bottom": bootstrap,
        "incremental_hac_regression": regression,
        "gate_checks": gate_checks,
        "known_prior_trial_lower_bound_before_daily_01": int(statistics["known_prior_trial_lower_bound_before_daily_01"]),
        "known_prior_trial_lower_bound_including_daily_01": int(statistics["known_prior_trial_lower_bound_before_daily_01"]) + 1,
        "multiple_testing_limits": {
            "white_reality_check": "NOT_ESTIMABLE_SINGLE_CANDIDATE",
            "hansen_spa": "NOT_ESTIMABLE_SINGLE_CANDIDATE",
            "pbo": "NOT_ESTIMABLE_SINGLE_CANDIDATE",
        },
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
