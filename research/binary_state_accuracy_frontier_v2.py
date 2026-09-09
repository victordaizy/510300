"""510300满仓/空仓短周期识别精度前沿V2。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.binary_state_feasibility_v1 import (
    CostModel,
    build_perfect_block_states,
    load_and_audit_inputs as load_v1_inputs,
    load_config as load_v1_config,
    validate_manifest as validate_v1_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_binary_state_accuracy_frontier_v2.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_binary_state_accuracy_frontier_v2_manifest.json"


@dataclass(frozen=True)
class BatchSimulationResult:
    """多条二元状态路径的逐日权益和成交统计。"""

    equity: np.ndarray
    total_execution_cost: np.ndarray
    trade_leg_count: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    objective = config["objective"]
    frontier = config["accuracy_frontier"]
    if protocol["study_id"] != "510300_BINARY_STATE_ACCURACY_FRONTIER_V2":
        raise ValueError("研究编号不匹配")
    if protocol["is_strategy_candidate"] or protocol["is_trading_signal"]:
        raise ValueError("精度前沿不得伪装成策略或交易信号")
    if protocol["live_trading_authorized"]:
        raise ValueError("精度前沿不得授权实盘")
    if scope["execution_asset"] != "510300.SH":
        raise ValueError("唯一执行资产必须是510300.SH")
    if list(scope["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("允许持有资产必须严格为510300与现金")
    if list(scope["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if scope["leverage_allowed"] or scope["short_selling_allowed"] or scope["derivatives_allowed"]:
        raise ValueError("禁止杠杆、卖空和衍生品")
    if objective["benchmark_id"] != "H00300":
        raise ValueError("主基准必须是H00300全收益指数")
    if float(objective["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额目标必须固定为20个百分点")
    if float(objective["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动超额中位数目标必须固定为20个百分点")
    if [int(value) for value in frontier["block_horizons_trading_days"]] != [1, 5]:
        raise ValueError("V2只能检验1日和5日块")
    if not frontier["evaluate_all_calendar_offsets"]:
        raise ValueError("必须遍历全部日历错位")
    if frontier["final_incomplete_block_policy"] != "EXCLUDE":
        raise ValueError("末尾不完整块必须排除")
    if frontier["cost_scenario"] != "STRESS":
        raise ValueError("精度前沿必须使用压力成本")
    random_contract = frontier["random_exact_count_errors"]
    if int(random_contract["repetitions"]) != 100:
        raise ValueError("每个随机错误网格必须固定100次")
    if random_contract["count_rounding"] != "ROUND_HALF_UP":
        raise ValueError("错误数量必须使用ROUND_HALF_UP")
    recall_grid = [float(value) for value in random_contract["bad_block_recall_grid"]]
    false_exit_grid = [float(value) for value in random_contract["false_exit_rate_grid"]]
    if recall_grid != sorted(set(recall_grid)) or recall_grid[-1] != 1.0:
        raise ValueError("坏块召回率网格必须递增、唯一并包含100%")
    if false_exit_grid != sorted(set(false_exit_grid)) or false_exit_grid[0] != 0.0:
        raise ValueError("误退出率网格必须递增、唯一并包含0%")
    severity_grid = [float(value) for value in frontier["severity_ranked_oracle"]["bad_block_coverage_grid"]]
    if severity_grid != sorted(set(severity_grid)) or severity_grid[0] != 0.0 or severity_grid[-1] != 1.0:
        raise ValueError("严重度神谕覆盖网格必须递增、唯一并包含0%和100%")


def validate_manifest(root: Path = ROOT, path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("V2冻结清单不存在，禁止读取历史结果")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("study_id") != "510300_BINARY_STATE_ACCURACY_FRONTIER_V2":
        raise ValueError("V2冻结清单研究编号不匹配")
    drift: list[dict[str, str | None]] = []
    for section in ("tracked_files", "data_files"):
        for relative, expected in manifest.get(section, {}).items():
            target = root / relative
            actual = sha256_file(target) if target.exists() else None
            if actual != expected:
                drift.append({"文件": relative, "预期": expected, "实际": actual})
    if drift:
        raise ValueError(f"V2冻结清单漂移：{json.dumps(drift, ensure_ascii=False)}")
    return manifest


def load_source_inputs(
    root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """复用已冻结V1的数据契约，并确认V2没有改变核心目标与成本。"""

    source = config["source_contract"]
    v1_config = load_v1_config(root / source["v1_config"])
    v1_manifest = validate_v1_manifest(root, root / source["v1_manifest"])
    for section in ("scope", "objective", "costs"):
        if config[section] != v1_config[section]:
            raise ValueError(f"V2的{section}偏离已冻结V1")
    market, dividends, data_audit = load_v1_inputs(root, v1_config)
    expected_files = {
        v1_config["data_contracts"]["etf_market"]["file"],
        v1_config["data_contracts"]["benchmark_total_return"]["file"],
        v1_config["data_contracts"]["cash_distributions"]["file"],
    }
    actual_files = {
        source["etf_market"],
        source["benchmark_total_return"],
        source["cash_distributions"],
    }
    if actual_files != expected_files:
        raise ValueError("V2来源文件与V1数据契约不一致")
    return market, dividends, data_audit, v1_manifest


def stress_cost_model(config: dict[str, Any]) -> CostModel:
    costs = config["costs"]
    return CostModel(
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(costs["stress_slippage_bps_per_leg"]),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def round_half_up(value: float) -> int:
    if value < 0.0:
        raise ValueError("ROUND_HALF_UP只接受非负数")
    return int(np.floor(value + 0.5))


def build_exact_count_predictions(
    true_states: np.ndarray,
    *,
    bad_block_recall: float,
    false_exit_rate: float,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """构造错误数量严格固定、错误位置随机的块预测矩阵。"""

    true_values = np.asarray(true_states, dtype=np.int8)
    if true_values.ndim != 1 or not set(np.unique(true_values)).issubset({0, 1}):
        raise ValueError("真实块状态必须是一维0/1数组")
    if not 0.0 <= bad_block_recall <= 1.0 or not 0.0 <= false_exit_rate <= 1.0:
        raise ValueError("召回率与误退出率必须位于[0,1]")
    if repetitions <= 0:
        raise ValueError("重复次数必须为正整数")
    bad_indices = np.flatnonzero(true_values == 0)
    good_indices = np.flatnonzero(true_values == 1)
    correct_bad_count = round_half_up(len(bad_indices) * bad_block_recall)
    false_good_count = round_half_up(len(good_indices) * false_exit_rate)
    predictions = np.ones((repetitions, len(true_values)), dtype=np.int8)
    for repetition in range(repetitions):
        if correct_bad_count:
            selected_bad = rng.choice(bad_indices, size=correct_bad_count, replace=False)
            predictions[repetition, selected_bad] = 0
        if false_good_count:
            selected_good = rng.choice(good_indices, size=false_good_count, replace=False)
            predictions[repetition, selected_good] = 0
    return predictions


def build_severity_ranked_predictions(
    blocks: pd.DataFrame,
    *,
    bad_block_coverage: float,
) -> tuple[np.ndarray, float, int]:
    """利用未来损失排序，只退出最严重坏块；仅用于不可实现上界。"""

    if not 0.0 <= bad_block_coverage <= 1.0:
        raise ValueError("坏块覆盖率必须位于[0,1]")
    required = {"oracle_state", "cash_gross_factor", "full_gross_factor", "block_index"}
    missing = sorted(required.difference(blocks.columns))
    if missing:
        raise ValueError(f"块标签缺少列：{missing}")
    predictions = np.ones(len(blocks), dtype=np.int8)
    bad = blocks.loc[blocks["oracle_state"] == 0].copy()
    bad["opportunity_loss"] = (
        bad["cash_gross_factor"].astype(float) - bad["full_gross_factor"].astype(float)
    ).clip(lower=0.0)
    selected_count = round_half_up(len(bad) * bad_block_coverage)
    ranked = bad.sort_values(
        ["opportunity_loss", "block_index"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = ranked.head(selected_count)
    if not selected.empty:
        predictions[selected["block_index"].to_numpy(dtype=int)] = 0
    total_loss = float(bad["opportunity_loss"].sum())
    captured_loss = float(selected["opportunity_loss"].sum())
    capture_ratio = 0.0 if total_loss <= 0.0 else captured_loss / total_loss
    return predictions, float(capture_ratio), int(selected_count)


def expand_block_predictions(predictions: np.ndarray, *, horizon: int) -> np.ndarray:
    """把每块一个预测扩展为逐日目标状态，第0行固定为空仓。"""

    values = np.asarray(predictions, dtype=np.int8)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("块预测必须是二维0/1矩阵")
    if horizon <= 0:
        raise ValueError("块周期必须为正整数")
    repeated = np.repeat(values.T, horizon, axis=0)
    states = np.zeros((1 + len(repeated), values.shape[0]), dtype=np.int8)
    states[1:, :] = repeated
    return states


def _vector_commission(notional: np.ndarray, costs: CostModel) -> np.ndarray:
    return np.where(
        notional > 0.0,
        np.maximum(float(costs.minimum_commission), notional * float(costs.commission_rate)),
        0.0,
    )


def simulate_binary_batch(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    desired_states: np.ndarray,
    *,
    costs: CostModel,
    initial_capital: float,
    reinvest_paid_dividends: bool,
) -> BatchSimulationResult:
    """向量化复刻V1逐日成交、整手、分红应收与现金计息规则。"""

    data = market.reset_index(drop=True)
    states = np.asarray(desired_states, dtype=np.int8)
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    if states.ndim != 2 or states.shape[0] != len(data):
        raise ValueError("目标状态矩阵行数必须与市场数据一致")
    if not set(np.unique(states)).issubset({0, 1}):
        raise ValueError("目标状态只能为0或1")
    if np.any(states[0, :] != 0):
        raise ValueError("首行是前一收盘信号日，必须从现金开始")
    if initial_capital <= 0.0:
        raise ValueError("初始资金必须为正数")
    required_market = {"date", "etf_open", "etf_close"}
    missing_market = sorted(required_market.difference(data.columns))
    if missing_market:
        raise ValueError(f"市场数据缺少列：{missing_market}")

    dates = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    ex_events: dict[int, list[tuple[int | None, float]]] = {}
    for event in dividends.to_dict("records"):
        ex_date = pd.Timestamp(event["ex_date"]).normalize()
        if ex_date not in date_to_index:
            continue
        payment_date = pd.Timestamp(event["payment_date"]).normalize()
        payment_index = date_to_index.get(payment_date)
        ex_events.setdefault(date_to_index[ex_date], []).append(
            (payment_index, float(event["cash_dividend_per_share"]))
        )

    path_count = states.shape[1]
    cash = np.full(path_count, float(initial_capital), dtype=float)
    shares = np.zeros(path_count, dtype=float)
    receivable = np.zeros(path_count, dtype=float)
    payment_schedule: dict[int, np.ndarray] = {}
    equity = np.empty((len(data), path_count), dtype=float)
    total_cost = np.zeros(path_count, dtype=float)
    trade_legs = np.zeros(path_count, dtype=np.int64)
    cash_growth = 1.0 + float(costs.cash_annual_rate) / int(costs.trading_days_per_year)
    slippage_rate = float(costs.slippage_bps) / 10000.0
    lot_size = float(costs.lot_size)

    for index in range(len(data)):
        if index > 0:
            cash *= cash_growth
        shares_at_start = shares.copy()
        for payment_index, dividend_per_share in ex_events.get(index, []):
            entitlement = shares_at_start * dividend_per_share
            receivable += entitlement
            if payment_index is not None:
                if payment_index in payment_schedule:
                    payment_schedule[payment_index] += entitlement
                else:
                    payment_schedule[payment_index] = entitlement.copy()

        target = states[index, :]
        raw_open = float(data.loc[index, "etf_open"])
        sell_mask = (target == 0) & (shares > 0.0)
        if np.any(sell_mask):
            sell_quantity = shares[sell_mask].copy()
            execution_price = raw_open * (1.0 - slippage_rate)
            notional = sell_quantity * execution_price
            commission = _vector_commission(notional, costs)
            cash[sell_mask] += notional - commission
            shares[sell_mask] = 0.0
            total_cost[sell_mask] += commission + sell_quantity * (raw_open - execution_price)
            trade_legs[sell_mask] += 1

        eligible_to_buy = target == 1
        if not reinvest_paid_dividends:
            eligible_to_buy &= shares == 0.0
        if np.any(eligible_to_buy):
            execution_price = raw_open * (1.0 + slippage_rate)
            quantity = np.zeros(path_count, dtype=float)
            affordable_lots = np.floor(
                cash[eligible_to_buy]
                / (execution_price * (1.0 + float(costs.commission_rate)))
                / lot_size
            )
            quantity[eligible_to_buy] = np.maximum(affordable_lots, 0.0) * lot_size
            while True:
                notional_all = quantity * execution_price
                commission_all = _vector_commission(notional_all, costs)
                overspent = (quantity > 0.0) & (notional_all + commission_all > cash + 1e-9)
                if not np.any(overspent):
                    break
                quantity[overspent] = np.maximum(quantity[overspent] - lot_size, 0.0)
            buy_mask = quantity > 0.0
            if np.any(buy_mask):
                notional = quantity[buy_mask] * execution_price
                commission = _vector_commission(notional, costs)
                cash[buy_mask] -= notional + commission
                shares[buy_mask] += quantity[buy_mask]
                total_cost[buy_mask] += commission + quantity[buy_mask] * (execution_price - raw_open)
                trade_legs[buy_mask] += 1

        payment_today = payment_schedule.pop(index, None)
        if payment_today is not None:
            cash += payment_today
            receivable -= payment_today
            receivable[np.abs(receivable) < 1e-9] = 0.0
        equity[index, :] = cash + shares * float(data.loc[index, "etf_close"]) + receivable

    if not np.isfinite(equity).all() or np.any(equity <= 0.0):
        raise ValueError("批量模拟产生非法权益")
    return BatchSimulationResult(
        equity=equity,
        total_execution_cost=total_cost,
        trade_leg_count=trade_legs,
    )


def build_benchmark_equity(market: pd.DataFrame, initial_capital: float) -> np.ndarray:
    close = pd.to_numeric(market["benchmark_close"], errors="raise").to_numpy(dtype=float)
    if len(close) < 2 or not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ValueError("基准序列不足或包含非法值")
    return float(initial_capital) * close / close[0]


def _rolling_annualized_return(
    equity: np.ndarray,
    *,
    window: int,
    trading_days_per_year: int,
) -> np.ndarray:
    values = np.asarray(equity, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(values) < window:
        return np.empty((0, values.shape[1]), dtype=float)
    log_equity = np.log(values)
    log_returns = np.zeros_like(log_equity)
    log_returns[1:, :] = np.diff(log_equity, axis=0)
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(log_returns, axis=0)])
    rolling_sum = cumulative[window:, :] - cumulative[:-window, :]
    return np.expm1(rolling_sum / float(window) * int(trading_days_per_year))


def summarize_batch(
    result: BatchSimulationResult,
    desired_states: np.ndarray,
    benchmark_equity: np.ndarray,
    *,
    objective: dict[str, Any],
) -> pd.DataFrame:
    equity = np.asarray(result.equity, dtype=float)
    states = np.asarray(desired_states, dtype=np.int8)
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    benchmark = np.asarray(benchmark_equity, dtype=float)
    if equity.shape != states.shape or len(benchmark) != len(equity):
        raise ValueError("批量权益、状态与基准长度不一致")
    periods = len(equity) - 1
    trading_days = int(objective["annualization_trading_days"])
    strategy_cagr = np.expm1(
        (np.log(equity[-1, :]) - np.log(equity[0, :])) / periods * trading_days
    )
    benchmark_cagr = float(
        np.expm1((np.log(benchmark[-1]) - np.log(benchmark[0])) / periods * trading_days)
    )
    annualized_excess = strategy_cagr - benchmark_cagr
    window = int(objective["rolling_window_trading_days"])
    strategy_rolling = _rolling_annualized_return(
        equity,
        window=window,
        trading_days_per_year=trading_days,
    )
    benchmark_rolling = _rolling_annualized_return(
        benchmark,
        window=window,
        trading_days_per_year=trading_days,
    )
    if len(strategy_rolling) == 0:
        rolling_median = np.full(equity.shape[1], np.nan)
        rolling_minimum = np.full(equity.shape[1], np.nan)
        rolling_hit_ratio = np.full(equity.shape[1], np.nan)
    else:
        rolling_excess = strategy_rolling - benchmark_rolling
        rolling_target = float(objective["minimum_rolling_excess_median"])
        rolling_median = np.median(rolling_excess, axis=0)
        rolling_minimum = np.min(rolling_excess, axis=0)
        rolling_hit_ratio = np.mean(rolling_excess >= rolling_target, axis=0)
    peaks = np.maximum.accumulate(equity, axis=0)
    maximum_drawdown = np.min(equity / peaks - 1.0, axis=0)
    state_changes = np.sum(np.diff(states, axis=0) != 0, axis=0)
    annual_target = float(objective["minimum_annualized_excess"])
    rolling_target = float(objective["minimum_rolling_excess_median"])
    both_gates = (annualized_excess >= annual_target) & (rolling_median >= rolling_target)
    return pd.DataFrame(
        {
            "path_index": np.arange(equity.shape[1], dtype=int),
            "strategy_cagr": strategy_cagr,
            "benchmark_cagr": benchmark_cagr,
            "annualized_excess": annualized_excess,
            "rolling_242d_excess_median": rolling_median,
            "rolling_242d_excess_minimum": rolling_minimum,
            "rolling_242d_target_hit_ratio": rolling_hit_ratio,
            "maximum_drawdown": maximum_drawdown,
            "average_policy_state": np.mean(states, axis=0),
            "state_change_count": state_changes.astype(int),
            "trade_leg_count": result.trade_leg_count.astype(int),
            "total_execution_cost_cny": result.total_execution_cost,
            "annualized_excess_gate": annualized_excess >= annual_target,
            "rolling_median_gate": rolling_median >= rolling_target,
            "both_20pct_gates": both_gates,
        }
    )


def _quantile(values: pd.Series, probability: float) -> float:
    return float(pd.to_numeric(values, errors="raise").quantile(probability))


def aggregate_random_trials(trials: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "horizon_trading_days",
        "calendar_offset",
        "target_bad_block_recall",
        "target_false_exit_rate",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in trials.groupby(keys, sort=True):
        horizon, offset, recall, false_exit = key
        rows.append(
            {
                "horizon_trading_days": int(horizon),
                "calendar_offset": int(offset),
                "target_bad_block_recall": float(recall),
                "target_false_exit_rate": float(false_exit),
                "repetitions": int(len(group)),
                "realized_bad_block_recall": float(group["realized_bad_block_recall"].median()),
                "realized_false_exit_rate": float(group["realized_false_exit_rate"].median()),
                "bad_opportunity_loss_capture_median": float(
                    group["bad_opportunity_loss_capture"].median()
                ),
                "annualized_excess_p10": _quantile(group["annualized_excess"], 0.10),
                "annualized_excess_median": float(group["annualized_excess"].median()),
                "annualized_excess_p90": _quantile(group["annualized_excess"], 0.90),
                "rolling_242d_excess_median_p10": _quantile(
                    group["rolling_242d_excess_median"], 0.10
                ),
                "rolling_242d_excess_median_median": float(
                    group["rolling_242d_excess_median"].median()
                ),
                "rolling_242d_excess_median_p90": _quantile(
                    group["rolling_242d_excess_median"], 0.90
                ),
                "both_20pct_gate_success_ratio": float(group["both_20pct_gates"].mean()),
                "average_policy_state_median": float(group["average_policy_state"].median()),
                "state_change_count_median": float(group["state_change_count"].median()),
                "total_execution_cost_cny_median": float(
                    group["total_execution_cost_cny"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _first_required_recall(
    aggregate: pd.DataFrame,
    *,
    horizon: int,
    false_exit: float,
    objective: dict[str, Any],
) -> dict[str, Any]:
    subset = aggregate.loc[
        (aggregate["horizon_trading_days"] == horizon)
        & np.isclose(aggregate["target_false_exit_rate"], false_exit)
    ]
    annual_target = float(objective["minimum_annualized_excess"])
    rolling_target = float(objective["minimum_rolling_excess_median"])
    primary: float | None = None
    secondary: float | None = None
    diagnostics: list[dict[str, Any]] = []
    for recall in sorted(subset["target_bad_block_recall"].unique()):
        cell = subset.loc[np.isclose(subset["target_bad_block_recall"], recall)]
        offset_pass = (
            (cell["annualized_excess_median"] >= annual_target)
            & (cell["rolling_242d_excess_median_median"] >= rolling_target)
        )
        median_offset_pass = bool(
            float(cell["annualized_excess_median"].median()) >= annual_target
            and float(cell["rolling_242d_excess_median_median"].median()) >= rolling_target
        )
        all_offset_pass = bool(offset_pass.all())
        diagnostics.append(
            {
                "bad_block_recall": float(recall),
                "offsets_passing_both_median_gates": int(offset_pass.sum()),
                "offset_count": int(len(cell)),
                "all_offsets_pass": all_offset_pass,
                "median_offset_pass": median_offset_pass,
                "worst_offset_annualized_excess_median": float(
                    cell["annualized_excess_median"].min()
                ),
                "worst_offset_rolling_excess_median": float(
                    cell["rolling_242d_excess_median_median"].min()
                ),
            }
        )
        if secondary is None and median_offset_pass:
            secondary = float(recall)
        if primary is None and all_offset_pass:
            primary = float(recall)
    return {
        "horizon_trading_days": int(horizon),
        "false_exit_rate": float(false_exit),
        "required_bad_block_recall_every_offset": primary,
        "required_bad_block_recall_median_offset": secondary,
        "grid_diagnostics": diagnostics,
    }


def _first_required_severity_coverage(
    severity: pd.DataFrame,
    *,
    horizon: int,
    objective: dict[str, Any],
) -> dict[str, Any]:
    subset = severity.loc[severity["horizon_trading_days"] == horizon]
    annual_target = float(objective["minimum_annualized_excess"])
    rolling_target = float(objective["minimum_rolling_excess_median"])
    primary: float | None = None
    secondary: float | None = None
    diagnostics: list[dict[str, Any]] = []
    for coverage in sorted(subset["target_bad_block_coverage"].unique()):
        cell = subset.loc[np.isclose(subset["target_bad_block_coverage"], coverage)]
        offset_pass = (
            (cell["annualized_excess"] >= annual_target)
            & (cell["rolling_242d_excess_median"] >= rolling_target)
        )
        median_offset_pass = bool(
            float(cell["annualized_excess"].median()) >= annual_target
            and float(cell["rolling_242d_excess_median"].median()) >= rolling_target
        )
        all_offset_pass = bool(offset_pass.all())
        diagnostics.append(
            {
                "target_bad_block_coverage": float(coverage),
                "realized_bad_block_coverage_median": float(
                    cell["realized_bad_block_coverage"].median()
                ),
                "bad_opportunity_loss_capture_median": float(
                    cell["bad_opportunity_loss_capture"].median()
                ),
                "offsets_passing_both_gates": int(offset_pass.sum()),
                "offset_count": int(len(cell)),
                "all_offsets_pass": all_offset_pass,
                "median_offset_pass": median_offset_pass,
                "worst_offset_annualized_excess": float(cell["annualized_excess"].min()),
                "worst_offset_rolling_excess_median": float(
                    cell["rolling_242d_excess_median"].min()
                ),
            }
        )
        if secondary is None and median_offset_pass:
            secondary = float(coverage)
        if primary is None and all_offset_pass:
            primary = float(coverage)
    return {
        "horizon_trading_days": int(horizon),
        "required_target_bad_block_coverage_every_offset": primary,
        "required_target_bad_block_coverage_median_offset": secondary,
        "grid_diagnostics": diagnostics,
    }


def evaluate_frontiers(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frontier = config["accuracy_frontier"]
    random_contract = frontier["random_exact_count_errors"]
    severity_contract = frontier["severity_ranked_oracle"]
    horizons = [int(value) for value in frontier["block_horizons_trading_days"]]
    repetitions = int(random_contract["repetitions"])
    rng = np.random.default_rng(int(random_contract["random_seed"]))
    costs = stress_cost_model(config)
    initial_capital = float(config["scope"]["initial_capital_cny"])
    reinvest = bool(config["scope"]["full_state_reinvests_paid_cash_dividends"])
    random_rows: list[pd.DataFrame] = []
    severity_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        offsets = range(horizon) if frontier["evaluate_all_calendar_offsets"] else range(1)
        for offset in offsets:
            shifted = market.iloc[offset:].reset_index(drop=True)
            _, blocks, last_end = build_perfect_block_states(
                shifted,
                dividends,
                horizon=horizon,
                cash_annual_rate=float(config["costs"]["cash_annual_rate"]),
                trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
            )
            sample = shifted.iloc[: last_end + 1].reset_index(drop=True)
            if len(sample) != 1 + len(blocks) * horizon:
                raise ValueError("完整块展开长度与样本不一致")
            labels = blocks.copy()
            labels.insert(0, "calendar_offset", int(offset))
            labels.insert(0, "horizon_trading_days", int(horizon))
            labels["global_start_index"] = labels["start_index"].astype(int) + offset
            labels["global_end_index"] = labels["end_index"].astype(int) + offset
            labels["bad_opportunity_loss"] = (
                labels["cash_gross_factor"].astype(float)
                - labels["full_gross_factor"].astype(float)
            ).clip(lower=0.0)
            block_frames.append(labels)

            benchmark_equity = build_benchmark_equity(sample, initial_capital)
            true_states = blocks["oracle_state"].to_numpy(dtype=np.int8)
            bad_mask = true_states == 0
            good_mask = true_states == 1
            bad_losses = labels.loc[bad_mask, "bad_opportunity_loss"].to_numpy(dtype=float)
            total_bad_loss = float(bad_losses.sum())

            for recall in [float(value) for value in random_contract["bad_block_recall_grid"]]:
                for false_exit in [float(value) for value in random_contract["false_exit_rate_grid"]]:
                    predictions = build_exact_count_predictions(
                        true_states,
                        bad_block_recall=recall,
                        false_exit_rate=false_exit,
                        repetitions=repetitions,
                        rng=rng,
                    )
                    states = expand_block_predictions(predictions, horizon=horizon)
                    simulation = simulate_binary_batch(
                        sample,
                        dividends,
                        states,
                        costs=costs,
                        initial_capital=initial_capital,
                        reinvest_paid_dividends=reinvest,
                    )
                    metrics = summarize_batch(
                        simulation,
                        states,
                        benchmark_equity,
                        objective=config["objective"],
                    )
                    if bad_mask.any():
                        realized_recall = np.mean(predictions[:, bad_mask] == 0, axis=1)
                        captured = np.sum(
                            (predictions[:, bad_mask] == 0) * bad_losses.reshape(1, -1),
                            axis=1,
                        )
                        loss_capture = (
                            np.zeros(repetitions, dtype=float)
                            if total_bad_loss <= 0.0
                            else captured / total_bad_loss
                        )
                    else:
                        realized_recall = np.full(repetitions, np.nan)
                        loss_capture = np.full(repetitions, np.nan)
                    realized_false_exit = (
                        np.mean(predictions[:, good_mask] == 0, axis=1)
                        if good_mask.any()
                        else np.full(repetitions, np.nan)
                    )
                    metrics.insert(0, "repetition", np.arange(repetitions, dtype=int))
                    metrics.insert(0, "target_false_exit_rate", float(false_exit))
                    metrics.insert(0, "target_bad_block_recall", float(recall))
                    metrics.insert(0, "calendar_offset", int(offset))
                    metrics.insert(0, "horizon_trading_days", int(horizon))
                    metrics["block_count"] = int(len(blocks))
                    metrics["bad_block_count"] = int(bad_mask.sum())
                    metrics["good_block_count"] = int(good_mask.sum())
                    metrics["realized_bad_block_recall"] = realized_recall
                    metrics["realized_false_exit_rate"] = realized_false_exit
                    metrics["bad_opportunity_loss_capture"] = loss_capture
                    metrics["sample_start_date"] = sample["date"].iloc[0]
                    metrics["sample_end_date"] = sample["date"].iloc[-1]
                    random_rows.append(metrics)

            for coverage in [
                float(value) for value in severity_contract["bad_block_coverage_grid"]
            ]:
                prediction, loss_capture, selected_count = build_severity_ranked_predictions(
                    blocks,
                    bad_block_coverage=coverage,
                )
                states = expand_block_predictions(prediction, horizon=horizon)
                simulation = simulate_binary_batch(
                    sample,
                    dividends,
                    states,
                    costs=costs,
                    initial_capital=initial_capital,
                    reinvest_paid_dividends=reinvest,
                )
                metrics = summarize_batch(
                    simulation,
                    states,
                    benchmark_equity,
                    objective=config["objective"],
                ).iloc[0]
                bad_count = int(bad_mask.sum())
                severity_rows.append(
                    {
                        "horizon_trading_days": int(horizon),
                        "calendar_offset": int(offset),
                        "target_bad_block_coverage": float(coverage),
                        "selected_bad_block_count": int(selected_count),
                        "bad_block_count": bad_count,
                        "good_block_count": int(good_mask.sum()),
                        "realized_bad_block_coverage": (
                            np.nan if bad_count == 0 else selected_count / bad_count
                        ),
                        "bad_opportunity_loss_capture": float(loss_capture),
                        "sample_start_date": sample["date"].iloc[0],
                        "sample_end_date": sample["date"].iloc[-1],
                        **{
                            column: metrics[column]
                            for column in metrics.index
                            if column != "path_index"
                        },
                    }
                )

    random_trials = pd.concat(random_rows, ignore_index=True)
    random_aggregates = aggregate_random_trials(random_trials)
    severity = pd.DataFrame(severity_rows)
    block_labels = pd.concat(block_frames, ignore_index=True)
    random_requirements = [
        _first_required_recall(
            random_aggregates,
            horizon=horizon,
            false_exit=false_exit,
            objective=config["objective"],
        )
        for horizon in horizons
        for false_exit in [float(value) for value in random_contract["false_exit_rate_grid"]]
    ]
    severity_requirements = [
        _first_required_severity_coverage(
            severity,
            horizon=horizon,
            objective=config["objective"],
        )
        for horizon in horizons
    ]
    perfect_rows = random_aggregates.loc[
        np.isclose(random_aggregates["target_bad_block_recall"], 1.0)
        & np.isclose(random_aggregates["target_false_exit_rate"], 0.0)
    ]
    perfect_summary: list[dict[str, Any]] = []
    for horizon, group in perfect_rows.groupby("horizon_trading_days", sort=True):
        perfect_summary.append(
            {
                "horizon_trading_days": int(horizon),
                "offset_count": int(len(group)),
                "annualized_excess_median_across_offsets": float(
                    group["annualized_excess_median"].median()
                ),
                "annualized_excess_worst_offset": float(group["annualized_excess_median"].min()),
                "rolling_excess_median_across_offsets": float(
                    group["rolling_242d_excess_median_median"].median()
                ),
                "rolling_excess_worst_offset": float(
                    group["rolling_242d_excess_median_median"].min()
                ),
            }
        )
    summary = {
        "random_exact_count_requirements": random_requirements,
        "severity_ranked_oracle_requirements": severity_requirements,
        "perfect_accuracy_check": perfect_summary,
    }
    return random_trials, random_aggregates, severity, block_labels, summary


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    v1_manifest: dict[str, Any],
    data_audit: dict[str, Any],
    random_trials: pd.DataFrame,
    random_aggregates: pd.DataFrame,
    severity: pd.DataFrame,
    block_labels: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "BINARY_SHORT_HORIZON_ACCURACY_REQUIREMENT_QUANTIFIED",
        "goal_achieved": False,
        "reason_goal_not_achieved": "全部标签与严重度排序均使用未来结果，只能约束候选所需能力，不能构成可实现策略证据",
        "historical_evidence_label": config["protocol"]["historical_evidence_label"],
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(MANIFEST_PATH),
            "frozen_at": manifest["frozen_at"],
        },
        "upstream_v1_manifest": {
            "path": config["source_contract"]["v1_manifest"],
            "frozen_at": v1_manifest["frozen_at"],
        },
        "scope": config["scope"],
        "objective": config["objective"],
        "costs": config["costs"],
        "data_audit": data_audit,
        "row_counts": {
            "random_trials": int(len(random_trials)),
            "random_aggregates": int(len(random_aggregates)),
            "severity_ranked_oracle": int(len(severity)),
            "block_labels": int(len(block_labels)),
        },
        "frontier_summary": summary,
        "governance": {
            "strategy_candidate_created": False,
            "current_state_signal_created": False,
            "position_mapping_enabled": False,
            "paper_or_shadow_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }


def _format_optional_percentage(value: float | None) -> str:
    return "网格内未达到" if value is None else f"{float(value):.1%}"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["frontier_summary"]
    lines = [
        "# 510300 满仓/空仓短周期识别精度前沿 V2",
        "",
        f"- 状态：`{report['status']}`",
        "- 目标达成：`false`（未来方向和未来严重度只用于诊断）",
        f"- 数据截止：`{report['data_audit']['historical_cutoff']}`",
        "- 仓位集合：`{0%, 100%}`",
        "- 成本：1bp佣金/腿 + 15bp压力滑点/腿；现金年化1.5%。",
        "",
        "## 随机等数量错误前沿",
        "",
        "主口径要求每个日历错位的100次模拟中位数都同时通过两项20%门槛；次口径只要求日历错位中位数通过。",
        "",
        "| 周期 | 误退出率 | 主口径最低坏块召回率 | 次口径最低坏块召回率 |",
        "|---:|---:|---:|---:|",
    ]
    for item in summary["random_exact_count_requirements"]:
        lines.append(
            "| {h}日 | {f:.0%} | {primary} | {secondary} |".format(
                h=item["horizon_trading_days"],
                f=item["false_exit_rate"],
                primary=_format_optional_percentage(
                    item["required_bad_block_recall_every_offset"]
                ),
                secondary=_format_optional_percentage(
                    item["required_bad_block_recall_median_offset"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 未来严重度排序上界",
            "",
            "该上界提前知道每个坏块的损失严重度，只退出最危险的部分，并且从不误退出好块。",
            "",
            "| 周期 | 主口径最低坏块覆盖率 | 次口径最低坏块覆盖率 | 达到主口径时损失捕获中位数 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in summary["severity_ranked_oracle_requirements"]:
        required = item["required_target_bad_block_coverage_every_offset"]
        capture: float | None = None
        if required is not None:
            matching = [
                row
                for row in item["grid_diagnostics"]
                if np.isclose(row["target_bad_block_coverage"], required)
            ]
            if matching:
                capture = float(matching[0]["bad_opportunity_loss_capture_median"])
        lines.append(
            "| {h}日 | {primary} | {secondary} | {capture} |".format(
                h=item["horizon_trading_days"],
                primary=_format_optional_percentage(required),
                secondary=_format_optional_percentage(
                    item["required_target_bad_block_coverage_median_offset"]
                ),
                capture="不适用" if capture is None else f"{capture:.1%}",
            )
        )
    lines.extend(
        [
            "",
            "## 完美准确率复核",
            "",
            "| 周期 | 年化超额错位中位数 | 年化超额最差错位 | 滚动超额错位中位数 | 滚动超额最差错位 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["perfect_accuracy_check"]:
        lines.append(
            "| {h}日 | {a:.2%} | {aw:.2%} | {r:.2%} | {rw:.2%} |".format(
                h=item["horizon_trading_days"],
                a=item["annualized_excess_median_across_offsets"],
                aw=item["annualized_excess_worst_offset"],
                r=item["rolling_excess_median_across_offsets"],
                rw=item["rolling_excess_worst_offset"],
            )
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 本结果只决定下一独立候选需要达到的事件频率、召回率与误退出上限。",
            "- 未来方向标签、未来严重度排序和完美准确率都不得作为候选业绩。",
            "- 本研究未生成当前满仓/空仓判断、Paper/Shadow、订单或实盘授权。",
        ]
    )
    return "\n".join(lines) + "\n"
