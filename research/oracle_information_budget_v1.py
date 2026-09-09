"""510300未来20日上涨机会的信息质量预算与Oracle稳健性实验。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from backtest.engine import BacktestCosts  # noqa: E402
from known_state_policy_oracle_v1 import (  # noqa: E402
    STATE_BEAR,
    STATE_BULL,
    STATE_CENSORED,
    STATE_RANGE,
    classify_oracle_state,
    load_config as load_source_oracle_config,
    validate_manifest as validate_source_oracle_manifest,
)
from three_state_trend_router_v1_0_1 import (  # noqa: E402
    ContractError,
    load_inputs as load_daily_inputs,
)


CONFIG_PATH = ROOT / "config" / "510300_oracle_information_budget_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_oracle_information_budget_v1_manifest.json"
PROJECT_ID = "510300_ORACLE_INFORMATION_BUDGET_V1"
COMPLETE_STATES = (STATE_BULL, STATE_BEAR, STATE_RANGE)


@dataclass(frozen=True)
class EngineInputs:
    """批量账户回测所需的固定市场数组。"""

    dates: pd.DatetimeIndex
    opens: np.ndarray
    closes: np.ndarray
    dividend_events_by_ex_index: dict[int, tuple[tuple[int | None, float], ...]]
    dividend_per_ex_index: np.ndarray


@dataclass(frozen=True)
class StudyContext:
    """冻结输入解包后的研究上下文。"""

    market: pd.DataFrame
    total_return_index: pd.DataFrame
    dividends: pd.DataFrame
    engine: EngineInputs
    blocks: pd.DataFrame
    anchor_positions: np.ndarray
    states: np.ndarray
    source_result: dict[str, Any]
    source_stress_ledger: pd.DataFrame
    source_stress_trades: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    random_cfg = config["random_error_budget"]
    source = config["source_oracle"]
    checks = {
        "project_id": config["protocol"]["project_id"] == PROJECT_ID,
        "source_known": config["protocol"]["source_oracle_outcomes_already_known"]
        is True,
        "new_outputs_unseen": config["protocol"][
            "information_budget_outputs_computed_before_freeze"
        ]
        is False,
        "target": config["research_question"]["target_net_sharpe"] == 1.2,
        "up20": config["research_question"]["prediction_target"] == "UP20",
        "asset": config["scope"]["execution_asset"] == "510300.SH",
        "holdings": config["scope"]["allowed_holdings"]
        == ["510300.SH", "CASH_CNY"],
        "source_threshold": source["threshold"] == 0.05,
        "source_horizon": source["horizon_trading_days"] == 20,
        "state_counts": source["expected_state_counts"]
        == {STATE_BULL: 23, STATE_BEAR: 21, STATE_RANGE: 97},
        "random_repetitions": random_cfg["repetitions_per_cell"] == 500,
        "joint_up": random_cfg["joint_grid"]["captured_bull_counts"]
        == [4, 8, 12, 16, 20, 23],
        "joint_range": random_cfg["joint_grid"]["false_range_counts"]
        == [0, 5, 10, 20, 40, 97],
        "joint_down": random_cfg["joint_grid"]["false_bear_counts"]
        == [0, 1, 2, 4, 8, 21],
        "offsets": config["robustness_annex"]["block_phase_offsets"]
        == list(range(20)),
        "entry_delays": config["timing_and_persistence_budget"][
            "entry_delay_days"
        ]
        == list(range(11)),
        "stress_slippage": config["execution"]["stress_costs"][
            "slippage_bps_per_leg"
        ]
        == 10.0,
        "prior_manifests": config["selection_bias"]["prior_manifest_count"] == 355,
        "total_trials": config["selection_bias"][
            "expected_total_trial_count_including_current"
        ]
        == 356,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ContractError(f"信息预算冻结配置异常：{failed}")
    forbidden_scope = (
        "leverage_allowed",
        "short_selling_allowed",
        "derivatives_execution_allowed",
        "prediction_model_evaluated",
        "paper_or_live_signal_allowed",
    )
    if any(bool(config["scope"][key]) for key in forbidden_scope):
        raise ContractError("信息预算不得授权杠杆、卖空、衍生品、预测模型或交易信号")
    rescue_fields = (
        "error_grid_rescue_after_result",
        "random_seed_rescue_after_result",
        "adversarial_rule_rescue_after_result",
        "timing_grid_rescue_after_result",
        "robustness_rule_rescue_after_result",
    )
    if any(config["protocol"][key] != "forbidden" for key in rescue_fields):
        raise ContractError("信息预算结果后救援开关没有全部冻结")
    if config["reported_budget_fields"] != [
        "MIN_UP_PRECISION_FOR_SHARPE_1_2",
        "MIN_UP_RECALL_FOR_SHARPE_1_2",
        "MAX_DOWN_FALSE_LONG_RATE",
        "MAX_RANGE_FALSE_LONG_RATE",
        "MAX_ENTRY_DELAY_DAYS",
        "MIN_ORACLE_RETURN_CAPTURE",
        "MAX_ACCEPTABLE_WHIPSAW_LEGS",
    ]:
        raise ContractError("必须报告的信息预算字段发生变化")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("信息预算冻结清单不存在，禁止计算预算结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("信息预算清单项目编号不匹配")
    if manifest.get("state") != "FROZEN_BEFORE_INFORMATION_BUDGET_CALCULATION":
        raise ContractError("信息预算清单状态不允许计算")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("信息预算冻结后配置发生漂移")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"信息预算冻结文件发生漂移：{mismatches}")
    if manifest.get("source_oracle_outcomes_already_known") is not True:
        raise ContractError("信息预算没有如实声明源Oracle结果已经可见")
    if manifest.get("information_budget_outputs_computed_before_freeze") is not False:
        raise ContractError("信息预算冻结前新结果可见性声明失败")
    return manifest


def _verify_declared_file_hashes(config: dict[str, Any]) -> None:
    records: list[dict[str, Any]] = []
    for key in ("manifest", "result", "blocks", "stress_ledger", "stress_trades"):
        records.append(config["source_oracle"][key])
    for key in (
        "etf_daily",
        "index_daily",
        "benchmark_total_return",
        "dividends",
        "backtest_engine",
    ):
        records.append(config["inputs"][key])
    failures: dict[str, dict[str, str]] = {}
    for record in records:
        path = _project_path(record["path"])
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != record["sha256"]:
            failures[record["path"]] = {
                "expected": record["sha256"],
                "actual": actual,
            }
    if failures:
        raise ContractError(f"信息预算源文件哈希不符合冻结合同：{failures}")


def prepare_engine_inputs(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> EngineInputs:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    market = prices.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.loc[market["date"].between(start, end)].sort_values("date")
    market = market.reset_index(drop=True)
    if len(market) < 2 or market["date"].duplicated().any():
        raise ContractError("批量回测行情日期不足或重复")
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(market["close"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(opens).all() or not np.isfinite(closes).all():
        raise ContractError("批量回测行情存在非有限价格")
    if (opens <= 0).any() or (closes <= 0).any():
        raise ContractError("批量回测行情存在非正价格")
    dates = pd.DatetimeIndex(market["date"])
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    events = dividends.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="raise").dt.normalize()
    events["payment_date"] = pd.to_datetime(
        events["payment_date"], errors="raise"
    ).dt.normalize()
    events["cash_dividend_per_share"] = pd.to_numeric(
        events["cash_dividend_per_share"], errors="coerce"
    )
    by_ex: dict[int, list[tuple[int | None, float]]] = {}
    dividend_per_ex = np.zeros(len(dates), dtype=float)
    for row in events.itertuples(index=False):
        ex_index = date_to_index.get(pd.Timestamp(row.ex_date))
        if ex_index is None:
            continue
        payment_index = date_to_index.get(pd.Timestamp(row.payment_date))
        amount = float(row.cash_dividend_per_share)
        by_ex.setdefault(ex_index, []).append((payment_index, amount))
        dividend_per_ex[ex_index] += amount
    frozen_events = {key: tuple(value) for key, value in by_ex.items()}
    return EngineInputs(
        dates=dates,
        opens=opens,
        closes=closes,
        dividend_events_by_ex_index=frozen_events,
        dividend_per_ex_index=dividend_per_ex,
    )


def load_context(config: dict[str, Any]) -> StudyContext:
    _verify_declared_file_hashes(config)
    source_config = load_source_oracle_config()
    validate_source_oracle_manifest(source_config)
    etf, _index, total_return_index, dividends, _audit = load_daily_inputs(source_config)
    start = config["dates"]["evaluation_start"]
    end = config["dates"]["evaluation_end"]
    market = etf.loc[pd.to_datetime(etf["date"]).between(start, end)].copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.sort_values("date").reset_index(drop=True)
    engine = prepare_engine_inputs(market, dividends, start, end)
    block_path = _project_path(config["source_oracle"]["blocks"]["path"])
    blocks = pd.read_parquet(block_path)
    blocks = blocks.loc[np.isclose(blocks["threshold"].astype(float), 0.05)].copy()
    blocks = blocks.sort_values("block_id").reset_index(drop=True)
    blocks["signal_date"] = pd.to_datetime(blocks["signal_date"]).dt.normalize()
    if len(blocks) != 142:
        raise ContractError(f"主Oracle区间应为142行，实际{len(blocks)}行")
    if int(blocks["complete"].sum()) != int(
        config["source_oracle"]["expected_complete_blocks"]
    ):
        raise ContractError("主Oracle完整区间数量不符")
    state_counts = {
        state: int((blocks.loc[blocks["complete"], "state"] == state).sum())
        for state in COMPLETE_STATES
    }
    if state_counts != config["source_oracle"]["expected_state_counts"]:
        raise ContractError(f"主Oracle状态数量不符：{state_counts}")
    anchors = blocks["anchor_position"].to_numpy(dtype=int)
    if (anchors < 0).any() or (anchors >= len(engine.dates)).any():
        raise ContractError("主Oracle锚点越出评价日历")
    if blocks["signal_date"].tolist() != [engine.dates[index] for index in anchors]:
        raise ContractError("主Oracle锚点日期与执行日历不一致")
    source_result = json.loads(
        _project_path(config["source_oracle"]["result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source_sharpe = float(source_result["adjudication"]["primary_stress_net_sharpe"])
    if not math.isclose(
        source_sharpe,
        float(config["source_oracle"]["expected_primary_stress_net_sharpe"]),
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise ContractError("主Oracle压力夏普与冻结预期不一致")
    source_ledger = pd.read_parquet(
        _project_path(config["source_oracle"]["stress_ledger"]["path"])
    )
    source_ledger["date"] = pd.to_datetime(source_ledger["date"]).dt.normalize()
    source_trades = pd.read_parquet(
        _project_path(config["source_oracle"]["stress_trades"]["path"])
    )
    if len(source_ledger) != len(engine.dates):
        raise ContractError("主Oracle压力净值长度与评价日历不一致")
    if source_ledger["date"].tolist() != list(engine.dates):
        raise ContractError("主Oracle压力净值日期与评价日历不一致")
    return StudyContext(
        market=market,
        total_return_index=total_return_index,
        dividends=dividends,
        engine=engine,
        blocks=blocks,
        anchor_positions=anchors,
        states=blocks["state"].astype(str).to_numpy(),
        source_result=source_result,
        source_stress_ledger=source_ledger,
        source_stress_trades=source_trades,
    )


def _stress_costs(config: dict[str, Any]) -> BacktestCosts:
    values = config["execution"]["stress_costs"]
    return BacktestCosts(
        commission_rate=float(values["commission_rate_per_leg"]),
        minimum_commission_cny=float(values["minimum_commission_cny_per_leg"]),
        stamp_duty_rate=float(values["stamp_duty_rate"]),
        slippage_bps=float(values["slippage_bps_per_leg"]),
        lot_size=int(config["execution"]["lot_size_shares"]),
        cash_annual_rate=float(config["execution"]["cash_annual_rate"]),
    )


def simulate_dense_signals(
    signals: np.ndarray,
    engine: EngineInputs,
    initial_cash: float,
    costs: BacktestCosts,
) -> pd.DataFrame:
    """逐日批量复刻正式引擎；-1表示该收盘日没有新信号。"""

    signal_array = np.asarray(signals, dtype=np.int8)
    if signal_array.ndim != 2 or signal_array.shape[1] != len(engine.dates):
        raise ValueError("批量信号矩阵形状不正确")
    if not np.isin(signal_array, [-1, 0, 1]).all():
        raise ValueError("批量信号只能取-1、0、1")
    scenario_count, observation_count = signal_array.shape
    cash = np.full(scenario_count, float(initial_cash), dtype=float)
    shares = np.zeros(scenario_count, dtype=np.int64)
    dividend_receivable = np.zeros(scenario_count, dtype=float)
    current_target = np.zeros(scenario_count, dtype=np.int8)
    previous_equity = np.full(scenario_count, float(initial_cash), dtype=float)
    equity_peak = previous_equity.copy()
    maximum_drawdown = np.zeros(scenario_count, dtype=float)
    return_sum = np.zeros(scenario_count, dtype=float)
    return_square_sum = np.zeros(scenario_count, dtype=float)
    exposure_sum = np.zeros(scenario_count, dtype=float)
    trade_count = np.zeros(scenario_count, dtype=np.int32)
    total_explicit_cost = np.zeros(scenario_count, dtype=float)
    total_slippage_cost = np.zeros(scenario_count, dtype=float)
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
            notional = quantity.astype(float) * buy_price
            candidate_commission = np.maximum(
                costs.minimum_commission_cny,
                notional * costs.commission_rate,
            )
            unaffordable = buy_mask & (notional + candidate_commission > cash + 1e-9)
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

        close_equity = (
            cash + shares.astype(float) * close_price + dividend_receivable
        )
        if day_index == 0:
            daily_return = np.zeros(scenario_count, dtype=float)
        else:
            daily_return = close_equity / previous_equity - 1.0
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

    if observation_count <= 1:
        variance = np.zeros(scenario_count, dtype=float)
    else:
        variance = (
            return_square_sum - return_sum * return_sum / observation_count
        ) / (observation_count - 1)
        variance = np.maximum(variance, 0.0)
    daily_std = np.sqrt(variance)
    daily_mean = return_sum / observation_count
    annualized_volatility = daily_std * math.sqrt(242.0)
    sharpe = np.divide(
        daily_mean * 242.0,
        annualized_volatility,
        out=np.full(scenario_count, np.nan, dtype=float),
        where=annualized_volatility > 0,
    )
    total_return = previous_equity / float(initial_cash) - 1.0
    elapsed_days = max((engine.dates[-1] - engine.dates[0]).days, 1)
    cagr = np.where(
        total_return > -1.0,
        np.power(1.0 + total_return, 365.25 / elapsed_days) - 1.0,
        -1.0,
    )
    return pd.DataFrame(
        {
            "stress_net_sharpe": sharpe,
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


def predictions_to_dense_signals(
    predictions: np.ndarray, context: StudyContext
) -> np.ndarray:
    prediction_array = np.asarray(predictions, dtype=np.int8)
    if prediction_array.ndim != 2 or prediction_array.shape[1] != len(context.blocks):
        raise ValueError("区间预测矩阵形状不正确")
    if not np.isin(prediction_array, [0, 1]).all():
        raise ValueError("区间预测只能取0或1")
    signals = np.full(
        (prediction_array.shape[0], len(context.engine.dates)), -1, dtype=np.int8
    )
    signals[:, context.anchor_positions] = prediction_array
    return signals


def simulate_predictions(
    predictions: np.ndarray,
    context: StudyContext,
    config: dict[str, Any],
    batch_size: int = 4096,
) -> pd.DataFrame:
    prediction_array = np.asarray(predictions, dtype=np.int8)
    outputs: list[pd.DataFrame] = []
    costs = _stress_costs(config)
    initial = float(config["execution"]["initial_capital_cny"])
    for start in range(0, len(prediction_array), batch_size):
        batch = prediction_array[start : start + batch_size]
        signals = predictions_to_dense_signals(batch, context)
        outputs.append(simulate_dense_signals(signals, context.engine, initial, costs))
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def perfect_predictions(context: StudyContext) -> np.ndarray:
    return (context.states == STATE_BULL).astype(np.int8)


def verify_fast_engine(context: StudyContext, config: dict[str, Any]) -> dict[str, Any]:
    metrics = simulate_predictions(perfect_predictions(context)[None, :], context, config)
    actual = metrics.iloc[0]
    source = context.source_result["threshold_results"]["0.05"]["policies"][
        "STATE_ROUTER_RANGE_NO_T"
    ]["stress"]
    comparisons = {
        "stress_net_sharpe": (
            float(actual["stress_net_sharpe"]),
            float(source["sharpe_zero_cash_rate"]),
        ),
        "stress_total_return": (
            float(actual["stress_total_return"]),
            float(source["total_return"]),
        ),
        "stress_max_drawdown": (
            float(actual["stress_max_drawdown"]),
            float(source["max_drawdown"]),
        ),
        "average_exposure": (
            float(actual["average_exposure"]),
            float(source["average_exposure"]),
        ),
        "trade_count": (int(actual["trade_count"]), int(source["trade_count"])),
    }
    failures: dict[str, Any] = {}
    for name, (left, right) in comparisons.items():
        if name == "trade_count":
            passed = left == right
        else:
            passed = math.isclose(left, right, rel_tol=0.0, abs_tol=5e-12)
        if not passed:
            failures[name] = {"fast": left, "source": right}
    if failures:
        raise ContractError(f"批量账户引擎未能复刻原Oracle压力结果：{failures}")
    return {
        "status": "PASS_EXACT_SOURCE_ORACLE_REPLAY_WITHIN_5E_12",
        "comparisons": {
            name: {"fast": left, "source": right}
            for name, (left, right) in comparisons.items()
        },
    }


def build_label_execution_basis(context: StudyContext) -> pd.DataFrame:
    engine = context.engine
    tri = context.total_return_index[["date", "close"]].copy()
    tri["date"] = pd.to_datetime(tri["date"], errors="raise").dt.normalize()
    tri = tri.drop_duplicates("date").set_index("date")
    tri_close = pd.to_numeric(tri.reindex(engine.dates)["close"], errors="coerce")
    if tri_close.isna().any():
        raise ContractError("标签执行基差缺少H00300点位")
    rows: list[dict[str, Any]] = []
    for block in context.blocks.loc[context.blocks["complete"]].itertuples(index=False):
        anchor = int(block.anchor_position)
        label_end = anchor + 20
        entry = anchor + 1
        exit_index = anchor + 21
        if exit_index >= len(engine.dates):
            executable_return = math.nan
            exit_overnight = math.nan
        else:
            executable_dividend = float(
                engine.dividend_per_ex_index[entry + 1 : exit_index + 1].sum()
            )
            executable_return = float(
                (engine.opens[exit_index] + executable_dividend) / engine.opens[entry]
                - 1.0
            )
            exit_overnight = float(
                (
                    engine.opens[exit_index]
                    + engine.dividend_per_ex_index[exit_index]
                )
                / engine.closes[label_end]
                - 1.0
            )
        label_like_dividend = float(
            engine.dividend_per_ex_index[anchor + 1 : label_end + 1].sum()
        )
        etf_same_endpoint = float(
            (engine.closes[label_end] + label_like_dividend)
            / engine.closes[anchor]
            - 1.0
        )
        post_signal_dividend = float(
            engine.dividend_per_ex_index[entry + 1 : label_end + 1].sum()
        )
        post_signal_to_label_close = float(
            (engine.closes[label_end] + post_signal_dividend) / engine.opens[entry]
            - 1.0
        )
        h00300_return = float(tri_close.iloc[label_end] / tri_close.iloc[anchor] - 1.0)
        rows.append(
            {
                "block_id": int(block.block_id),
                "state": str(block.state),
                "signal_date": engine.dates[anchor],
                "label_end_date": engine.dates[label_end],
                "execution_entry_date": engine.dates[entry],
                "execution_exit_date": (
                    engine.dates[exit_index] if exit_index < len(engine.dates) else pd.NaT
                ),
                "h00300_label_close_to_close_return": h00300_return,
                "etf_same_endpoint_close_to_close_total_return": etf_same_endpoint,
                "etf_post_signal_open_to_label_close_total_return": post_signal_to_label_close,
                "etf_executable_open_to_next_execution_open_total_return": executable_return,
                "entry_untradeable_overnight_price_return": float(
                    engine.opens[entry] / engine.closes[anchor] - 1.0
                ),
                "exit_actual_overnight_total_return": exit_overnight,
                "execution_minus_label_return": executable_return - h00300_return,
                "tracking_same_endpoint_minus_label_return": etf_same_endpoint
                - h00300_return,
                "timing_open_to_open_minus_same_endpoint_return": executable_return
                - etf_same_endpoint,
            }
        )
    return pd.DataFrame(rows)


def _seed_sequence(base_seed: int, family_code: int, *values: int) -> np.random.SeedSequence:
    low = int(base_seed % (2**32))
    high = int((base_seed // (2**32)) % (2**32))
    return np.random.SeedSequence([low, high, family_code, *map(int, values)])


def sample_exact_predictions(
    block_count: int,
    bull_positions: np.ndarray,
    range_positions: np.ndarray,
    bear_positions: np.ndarray,
    captured_bulls: int,
    false_ranges: int,
    false_bears: int,
    repetitions: int,
    seed_sequence: np.random.SeedSequence,
) -> np.ndarray:
    groups = (
        (np.asarray(bull_positions, dtype=int), int(captured_bulls)),
        (np.asarray(range_positions, dtype=int), int(false_ranges)),
        (np.asarray(bear_positions, dtype=int), int(false_bears)),
    )
    predictions = np.zeros((int(repetitions), int(block_count)), dtype=np.int8)
    rng = np.random.default_rng(seed_sequence)
    rows = np.arange(repetitions)[:, None]
    for positions, count in groups:
        if count < 0 or count > len(positions):
            raise ValueError("状态内固定抽样数量越界")
        if count == 0:
            continue
        if count == len(positions):
            predictions[:, positions] = 1
            continue
        scores = rng.random((repetitions, len(positions)))
        chosen_local = np.argpartition(scores, count - 1, axis=1)[:, :count]
        predictions[rows, positions[chosen_local]] = 1
    return predictions


def _classification_fields(
    captured_bulls: int,
    false_ranges: int,
    false_bears: int,
    context: StudyContext,
) -> dict[str, Any]:
    bull_total = int((context.states == STATE_BULL).sum())
    range_total = int((context.states == STATE_RANGE).sum())
    bear_total = int((context.states == STATE_BEAR).sum())
    predicted_long = captured_bulls + false_ranges + false_bears
    return {
        "captured_bull_count": int(captured_bulls),
        "false_range_count": int(false_ranges),
        "false_bear_count": int(false_bears),
        "up_recall": float(captured_bulls / bull_total),
        "range_false_long_rate": float(false_ranges / range_total),
        "down_false_long_rate": float(false_bears / bear_total),
        "up_precision": (
            float(captured_bulls / predicted_long) if predicted_long > 0 else math.nan
        ),
        "predicted_long_block_count": int(predicted_long),
    }


def _attach_capture_and_pass(
    frame: pd.DataFrame,
    context: StudyContext,
    config: dict[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
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
    dd_limit = float(
        config["budget_pass_rules"]["maximum_drawdown_ratio_vs_source_all_long"]
    ) * all_long_mdd
    output["drawdown_limit"] = dd_limit
    output["sharpe_gate_pass"] = output["stress_net_sharpe"] >= float(
        config["budget_pass_rules"]["stress_net_sharpe_minimum"]
    )
    output["positive_return_gate_pass"] = output["stress_total_return"] > 0.0
    output["drawdown_gate_pass"] = output["stress_max_drawdown"].abs() <= dd_limit
    output["deterministic_budget_pass"] = (
        output["sharpe_gate_pass"]
        & output["positive_return_gate_pass"]
        & output["drawdown_gate_pass"]
    )
    return output


def _summarize_random_draws(
    draws: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    grouping = ["family", "cell_id"]
    metric_columns = (
        "stress_net_sharpe",
        "stress_total_return",
        "stress_cagr",
        "stress_max_drawdown",
        "average_exposure",
        "trade_count",
        "oracle_net_log_wealth_capture",
    )
    constant_columns = (
        "captured_bull_count",
        "false_range_count",
        "false_bear_count",
        "up_recall",
        "range_false_long_rate",
        "down_false_long_rate",
        "up_precision",
        "predicted_long_block_count",
    )
    rows: list[dict[str, Any]] = []
    rules = config["budget_pass_rules"]
    for keys, group in draws.groupby(grouping, sort=False):
        row: dict[str, Any] = {
            "family": keys[0],
            "cell_id": keys[1],
            "repetitions": int(len(group)),
        }
        for column in constant_columns:
            value = group[column].iloc[0]
            row[column] = int(value) if column.endswith("count") else float(value)
        for column in metric_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_q05"] = float(np.quantile(values, 0.05))
            row[f"{column}_median"] = float(np.quantile(values, 0.50))
            row[f"{column}_q95"] = float(np.quantile(values, 0.95))
        row["probability_sharpe_at_least_1_2"] = float(
            group["sharpe_gate_pass"].mean()
        )
        row["probability_positive_return"] = float(
            group["positive_return_gate_pass"].mean()
        )
        row["probability_drawdown_within_limit"] = float(
            group["drawdown_gate_pass"].mean()
        )
        row["random_budget_pass"] = bool(
            row["stress_net_sharpe_q05"]
            >= float(rules["random_sharpe_fifth_percentile_minimum"])
            and row["probability_sharpe_at_least_1_2"]
            >= float(rules["random_sharpe_pass_probability_minimum"])
            and row["stress_total_return_q05"] > 0.0
            and row["probability_drawdown_within_limit"]
            >= float(rules["random_drawdown_pass_probability_minimum"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_random_error_experiments(
    context: StudyContext,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config["random_error_budget"]
    repetitions = int(cfg["repetitions_per_cell"])
    base_seed = int(cfg["seed"])
    bull_positions = np.flatnonzero(context.states == STATE_BULL)
    range_positions = np.flatnonzero(context.states == STATE_RANGE)
    bear_positions = np.flatnonzero(context.states == STATE_BEAR)

    def generate(cells: Iterable[tuple[str, str, int, int, int, int]]) -> tuple[np.ndarray, pd.DataFrame]:
        prediction_chunks: list[np.ndarray] = []
        metadata_chunks: list[pd.DataFrame] = []
        for family, cell_id, captured, false_range, false_bear, family_code in cells:
            predictions = sample_exact_predictions(
                len(context.blocks),
                bull_positions,
                range_positions,
                bear_positions,
                captured,
                false_range,
                false_bear,
                repetitions,
                _seed_sequence(
                    base_seed, family_code, captured, false_range, false_bear
                ),
            )
            fields = _classification_fields(
                captured, false_range, false_bear, context
            )
            metadata = pd.DataFrame(
                {
                    "family": np.repeat(family, repetitions),
                    "cell_id": np.repeat(cell_id, repetitions),
                    "repetition": np.arange(repetitions, dtype=int),
                    **{
                        key: np.repeat(value, repetitions)
                        for key, value in fields.items()
                    },
                }
            )
            prediction_chunks.append(predictions)
            metadata_chunks.append(metadata)
        return np.vstack(prediction_chunks), pd.concat(metadata_chunks, ignore_index=True)

    axis_cells: list[tuple[str, str, int, int, int, int]] = []
    for captured in range(
        int(cfg["recall_axis"]["captured_bull_count_start"]),
        int(cfg["recall_axis"]["captured_bull_count_end"]) + 1,
    ):
        axis_cells.append(
            ("RECALL_ONLY", f"UP_{captured:02d}", captured, 0, 0, 101)
        )
    for false_range in range(
        int(cfg["range_axis"]["false_range_count_start"]),
        int(cfg["range_axis"]["false_range_count_end"]) + 1,
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
        int(cfg["down_axis"]["false_bear_count_start"]),
        int(cfg["down_axis"]["false_bear_count_end"]) + 1,
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
    axis_predictions, axis_meta = generate(axis_cells)
    axis_metrics = simulate_predictions(axis_predictions, context, config)
    axis_draws = _attach_capture_and_pass(
        pd.concat([axis_meta, axis_metrics], axis=1), context, config
    )
    axis_summary = _summarize_random_draws(axis_draws, config)

    joint_cells: list[tuple[str, str, int, int, int, int]] = []
    for captured in cfg["joint_grid"]["captured_bull_counts"]:
        for false_range in cfg["joint_grid"]["false_range_counts"]:
            for false_bear in cfg["joint_grid"]["false_bear_counts"]:
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
    joint_predictions, joint_meta = generate(joint_cells)
    joint_metrics = simulate_predictions(joint_predictions, context, config)
    joint_draws = _attach_capture_and_pass(
        pd.concat([joint_meta, joint_metrics], axis=1), context, config
    )
    joint_summary = _summarize_random_draws(joint_draws, config)
    return axis_draws, axis_summary, joint_draws, joint_summary


def _nested_adversarial_predictions(
    block_count: int,
    captured_order: np.ndarray,
    captured_count: int,
    range_order: np.ndarray,
    range_count: int,
    bear_order: np.ndarray,
    bear_count: int,
) -> np.ndarray:
    prediction = np.zeros(block_count, dtype=np.int8)
    prediction[captured_order[:captured_count]] = 1
    prediction[range_order[:range_count]] = 1
    prediction[bear_order[:bear_count]] = 1
    return prediction


def run_adversarial_error_experiments(
    context: StudyContext,
    config: dict[str, Any],
    basis: pd.DataFrame,
) -> pd.DataFrame:
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
    random_cfg = config["random_error_budget"]
    cells: list[tuple[str, str, int, int, int]] = []
    for captured in range(24):
        cells.append(("RECALL_ONLY", f"UP_{captured:02d}", captured, 0, 0))
    for false_range in range(98):
        cells.append(
            (
                "RANGE_CONTAMINATION",
                f"RANGE_{false_range:02d}",
                23,
                false_range,
                0,
            )
        )
    for false_bear in range(22):
        cells.append(
            (
                "DOWN_CONTAMINATION",
                f"DOWN_{false_bear:02d}",
                23,
                0,
                false_bear,
            )
        )
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
    predictions = np.vstack(
        [
            _nested_adversarial_predictions(
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
    metadata_rows: list[dict[str, Any]] = []
    for family, cell_id, captured, false_range, false_bear in cells:
        metadata_rows.append(
            {
                "mode": "ADVERSARIAL",
                "family": family,
                "cell_id": cell_id,
                **_classification_fields(captured, false_range, false_bear, context),
            }
        )
    metrics = simulate_predictions(predictions, context, config)
    output = pd.concat([pd.DataFrame(metadata_rows), metrics], axis=1)
    return _attach_capture_and_pass(output, context, config)


def build_baseline_signals(context: StudyContext) -> np.ndarray:
    signals = np.full(len(context.engine.dates), -1, dtype=np.int8)
    signals[context.anchor_positions] = perfect_predictions(context)
    return signals


def build_bull_events(context: StudyContext) -> list[dict[str, Any]]:
    bull_positions = np.flatnonzero(context.states == STATE_BULL)
    if len(bull_positions) == 0:
        return []
    groups: list[list[int]] = [[int(bull_positions[0])]]
    for position in bull_positions[1:]:
        position = int(position)
        previous = groups[-1][-1]
        anchors_adjacent = (
            int(context.anchor_positions[position])
            - int(context.anchor_positions[previous])
            == 20
        )
        if position == previous + 1 and anchors_adjacent:
            groups[-1].append(position)
        else:
            groups.append([position])
    events: list[dict[str, Any]] = []
    for event_id, positions in enumerate(groups):
        start_anchor = int(context.anchor_positions[positions[0]])
        transition_anchor = int(context.anchor_positions[positions[-1]]) + 20
        if transition_anchor >= len(context.engine.dates):
            raise ContractError("连续上涨事件缺少可执行退出锚点")
        events.append(
            {
                "event_id": int(event_id),
                "block_positions": positions,
                "block_ids": [
                    int(context.blocks.iloc[position]["block_id"])
                    for position in positions
                ],
                "start_anchor_position": start_anchor,
                "transition_anchor_position": transition_anchor,
                "start_signal_date": context.engine.dates[start_anchor],
                "transition_signal_date": context.engine.dates[transition_anchor],
                "block_count": int(len(positions)),
            }
        )
    return events


def build_timing_signal(
    context: StudyContext,
    events: list[dict[str, Any]],
    error_type: str,
    error_days: int,
) -> np.ndarray:
    if error_days < 0:
        raise ValueError("时序误差天数不能为负数")
    signal = build_baseline_signals(context)
    if error_days == 0:
        return signal
    day_count = len(signal)
    for event in events:
        start = int(event["start_anchor_position"])
        transition = int(event["transition_anchor_position"])
        if error_type == "ENTRY_DELAY":
            signal[start] = 0
            delayed_signal = start + error_days
            if delayed_signal < transition and delayed_signal < day_count:
                signal[delayed_signal] = 1
        elif error_type == "EARLY_EXIT":
            early_signal = transition - error_days
            if early_signal <= start:
                raise ContractError("提前退出误差越过上涨事件起点")
            signal[early_signal] = 0
        elif error_type == "LATE_EXIT":
            signal[transition] = 1
            late_signal = transition + error_days
            if late_signal < day_count:
                signal[late_signal] = 0
        else:
            raise ValueError(f"未知时序误差类型：{error_type}")
    return signal


def run_timing_experiments(
    context: StudyContext,
    config: dict[str, Any],
) -> pd.DataFrame:
    events = build_bull_events(context)
    timing_cfg = config["timing_and_persistence_budget"]
    signals: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    definitions = (
        ("ENTRY_DELAY", timing_cfg["entry_delay_days"]),
        ("EARLY_EXIT", timing_cfg["early_exit_days"]),
        ("LATE_EXIT", timing_cfg["late_exit_days"]),
    )
    for error_type, values in definitions:
        for value in values:
            signals.append(build_timing_signal(context, events, error_type, int(value)))
            metadata.append(
                {
                    "mode": "DETERMINISTIC_TIMING",
                    "family": error_type,
                    "error_days": int(value),
                    "continuous_bull_event_count": int(len(events)),
                }
            )
    signal_matrix = np.vstack(signals)
    metrics = simulate_dense_signals(
        signal_matrix,
        context.engine,
        float(config["execution"]["initial_capital_cny"]),
        _stress_costs(config),
    )
    return _attach_capture_and_pass(
        pd.concat([pd.DataFrame(metadata), metrics], axis=1), context, config
    )


def _target_after_close_from_sparse_signal(signal: np.ndarray) -> np.ndarray:
    current = 0
    targets = np.zeros(len(signal), dtype=np.int8)
    for index, value in enumerate(signal):
        if value >= 0:
            current = int(value)
        targets[index] = current
    return targets


def build_whipsaw_candidates(context: StudyContext) -> pd.DataFrame:
    baseline = build_baseline_signals(context)
    targets = _target_after_close_from_sparse_signal(baseline)
    rows: list[dict[str, Any]] = []
    engine = context.engine
    for signal_position in range(0, len(targets) - 2):
        if targets[signal_position] != 0 or targets[signal_position + 1] != 0:
            continue
        entry = signal_position + 1
        exit_index = signal_position + 2
        gross_return = float(
            (
                engine.opens[exit_index]
                + engine.dividend_per_ex_index[exit_index]
            )
            / engine.opens[entry]
            - 1.0
        )
        rows.append(
            {
                "signal_position": int(signal_position),
                "entry_date": engine.dates[entry],
                "exit_date": engine.dates[exit_index],
                "one_day_executable_gross_return": gross_return,
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ContractError("没有可用的非上涨一日错误往返窗口")
    return candidates


def _greedy_nonoverlapping_positions(
    ordered_positions: Iterable[int], required_count: int
) -> list[int]:
    selected: list[int] = []
    blocked: set[int] = set()
    for raw_position in ordered_positions:
        position = int(raw_position)
        if position in blocked:
            continue
        selected.append(position)
        blocked.update((position - 1, position, position + 1))
        if len(selected) >= required_count:
            break
    if len(selected) < required_count:
        raise ContractError(
            f"互不重叠的一日错误往返不足：需要{required_count}，实际{len(selected)}"
        )
    return selected


def _signals_with_whipsaws(
    baseline: np.ndarray, positions: Iterable[int]
) -> np.ndarray:
    signal = baseline.copy()
    for raw_position in positions:
        position = int(raw_position)
        signal[position] = 1
        signal[position + 1] = 0
    return signal


def _summarize_whipsaw_random(
    draws: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules = config["budget_pass_rules"]
    for count, group in draws.groupby("planned_round_trips", sort=True):
        row: dict[str, Any] = {
            "mode": "RANDOM_PRIORITY_GREEDY",
            "planned_round_trips": int(count),
            "planned_extra_legs": int(count) * 2,
            "repetitions": int(len(group)),
        }
        for column in (
            "stress_net_sharpe",
            "stress_total_return",
            "stress_max_drawdown",
            "average_exposure",
            "trade_count",
            "actual_incremental_trade_legs",
            "oracle_net_log_wealth_capture",
        ):
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_q05"] = float(np.quantile(values, 0.05))
            row[f"{column}_median"] = float(np.quantile(values, 0.50))
            row[f"{column}_q95"] = float(np.quantile(values, 0.95))
        row["probability_sharpe_at_least_1_2"] = float(
            group["sharpe_gate_pass"].mean()
        )
        row["probability_drawdown_within_limit"] = float(
            group["drawdown_gate_pass"].mean()
        )
        row["random_budget_pass"] = bool(
            row["stress_net_sharpe_q05"]
            >= float(rules["random_sharpe_fifth_percentile_minimum"])
            and row["probability_sharpe_at_least_1_2"]
            >= float(rules["random_sharpe_pass_probability_minimum"])
            and row["stress_total_return_q05"] > 0.0
            and row["probability_drawdown_within_limit"]
            >= float(rules["random_drawdown_pass_probability_minimum"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_whipsaw_experiments(
    context: StudyContext,
    config: dict[str, Any],
) -> pd.DataFrame:
    timing_cfg = config["timing_and_persistence_budget"]
    candidates = build_whipsaw_candidates(context)
    baseline = build_baseline_signals(context)
    baseline_metrics = simulate_dense_signals(
        baseline[None, :],
        context.engine,
        float(config["execution"]["initial_capital_cny"]),
        _stress_costs(config),
    ).iloc[0]
    baseline_trade_count = int(baseline_metrics["trade_count"])
    maximum_count = int(timing_cfg["whipsaw_adversarial_round_trip_count_end"])
    ordered = candidates.sort_values(
        "one_day_executable_gross_return", ascending=True
    )["signal_position"].astype(int)
    adversarial_positions = _greedy_nonoverlapping_positions(ordered, maximum_count)
    adversarial_signals = np.vstack(
        [
            _signals_with_whipsaws(baseline, adversarial_positions[:count])
            for count in range(
                int(timing_cfg["whipsaw_adversarial_round_trip_count_start"]),
                maximum_count + 1,
            )
        ]
    )
    adversarial_metrics = simulate_dense_signals(
        adversarial_signals,
        context.engine,
        float(config["execution"]["initial_capital_cny"]),
        _stress_costs(config),
    )
    adversarial_counts = np.arange(
        int(timing_cfg["whipsaw_adversarial_round_trip_count_start"]),
        maximum_count + 1,
        dtype=int,
    )
    adversarial = pd.concat(
        [
            pd.DataFrame(
                {
                    "mode": "ADVERSARIAL_GREEDY",
                    "planned_round_trips": adversarial_counts,
                    "planned_extra_legs": adversarial_counts * 2,
                    "repetitions": 1,
                }
            ),
            adversarial_metrics,
        ],
        axis=1,
    )
    adversarial["actual_incremental_trade_legs"] = (
        adversarial["trade_count"] - baseline_trade_count
    )
    adversarial = _attach_capture_and_pass(adversarial, context, config)

    random_counts = [
        int(value) for value in timing_cfg["whipsaw_random_round_trip_counts"]
    ]
    repetitions = int(timing_cfg["whipsaw_random_repetitions_per_cell"])
    candidate_positions = candidates["signal_position"].to_numpy(dtype=int)
    random_signals: list[np.ndarray] = []
    random_metadata: list[dict[str, Any]] = []
    base_seed = int(config["random_error_budget"]["seed"])
    for count in random_counts:
        rng = np.random.default_rng(_seed_sequence(base_seed, 205, count))
        for repetition in range(repetitions):
            if count == 0:
                selected: list[int] = []
            else:
                priority = rng.permutation(candidate_positions)
                selected = _greedy_nonoverlapping_positions(priority, count)
            random_signals.append(_signals_with_whipsaws(baseline, selected))
            random_metadata.append(
                {
                    "mode": "RANDOM_PRIORITY_GREEDY_DRAW",
                    "planned_round_trips": count,
                    "planned_extra_legs": count * 2,
                    "repetition": repetition,
                }
            )
    random_metrics = simulate_dense_signals(
        np.vstack(random_signals),
        context.engine,
        float(config["execution"]["initial_capital_cny"]),
        _stress_costs(config),
    )
    random_draws = pd.concat([pd.DataFrame(random_metadata), random_metrics], axis=1)
    random_draws["actual_incremental_trade_legs"] = (
        random_draws["trade_count"] - baseline_trade_count
    )
    random_draws = _attach_capture_and_pass(random_draws, context, config)
    random_summary = _summarize_whipsaw_random(random_draws, config)
    common_columns = sorted(set(adversarial.columns).union(random_summary.columns))
    return pd.concat(
        [
            adversarial.reindex(columns=common_columns),
            random_summary.reindex(columns=common_columns),
        ],
        ignore_index=True,
    )


def run_phase_offset_robustness(
    context: StudyContext,
    config: dict[str, Any],
) -> pd.DataFrame:
    engine = context.engine
    tri = context.total_return_index[["date", "close"]].copy()
    tri["date"] = pd.to_datetime(tri["date"], errors="raise").dt.normalize()
    tri_close = pd.to_numeric(
        tri.drop_duplicates("date").set_index("date").reindex(engine.dates)["close"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(tri_close).all():
        raise ContractError("相位稳健性缺少H00300全收益点位")
    threshold = float(config["robustness_annex"]["phase_threshold"])
    signal_rows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for offset in config["robustness_annex"]["block_phase_offsets"]:
        offset = int(offset)
        signal = np.full(len(engine.dates), -1, dtype=np.int8)
        counts = {STATE_BULL: 0, STATE_BEAR: 0, STATE_RANGE: 0}
        complete_count = 0
        censored_count = 0
        for anchor in range(offset, len(engine.dates), 20):
            if anchor + 20 < len(engine.dates):
                future_return = float(tri_close[anchor + 20] / tri_close[anchor] - 1.0)
                state = classify_oracle_state(future_return, threshold)
                counts[state] += 1
                complete_count += 1
                signal[anchor] = 1 if state == STATE_BULL else 0
            else:
                censored_count += 1
                signal[anchor] = 0
        signal_rows.append(signal)
        metadata.append(
            {
                "offset": offset,
                "complete_block_count": complete_count,
                "censored_block_count": censored_count,
                "bull_block_count": counts[STATE_BULL],
                "bear_block_count": counts[STATE_BEAR],
                "range_block_count": counts[STATE_RANGE],
            }
        )
    metrics = simulate_dense_signals(
        np.vstack(signal_rows),
        context.engine,
        float(config["execution"]["initial_capital_cny"]),
        _stress_costs(config),
    )
    return _attach_capture_and_pass(
        pd.concat([pd.DataFrame(metadata), metrics], axis=1), context, config
    )


def _annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float | None:
    values = np.asarray(returns, dtype=float)
    if len(values) < 2:
        return None
    volatility = float(np.std(values, ddof=1))
    if volatility <= 0:
        return None
    return float(np.mean(values) / volatility * math.sqrt(periods_per_year))


def _lo_adjusted_sharpe(
    returns: np.ndarray, periods_per_year: int, maximum_lag: int
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    centered = values - np.mean(values)
    gamma0 = float(np.mean(centered * centered))
    long_run_variance = gamma0
    autocorrelations: list[float] = []
    for lag in range(1, maximum_lag + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        weight = 1.0 - lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
        autocorrelations.append(covariance / gamma0 if gamma0 > 0 else math.nan)
    adjusted = (
        float(np.mean(values) * math.sqrt(periods_per_year / long_run_variance))
        if long_run_variance > 0
        else None
    )
    return {
        "maximum_lag": int(maximum_lag),
        "bartlett_long_run_variance": float(long_run_variance),
        "lo_adjusted_annualized_sharpe": adjusted,
        "autocorrelations_lag_1_to_q": autocorrelations,
    }


def serial_correlation_metrics(
    context: StudyContext, config: dict[str, Any]
) -> dict[str, Any]:
    ledger = context.source_stress_ledger.copy()
    returns = ledger["daily_return"].to_numpy(dtype=float)
    serial_cfg = config["robustness_annex"]["serial_correlation"]
    monthly = (
        ledger.assign(month=ledger["date"].dt.to_period("M"))
        .groupby("month", sort=True)["daily_return"]
        .apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
        .to_numpy(dtype=float)
    )
    interval_returns: list[float] = []
    for block in context.blocks.loc[context.blocks["complete"]].itertuples(index=False):
        anchor = int(block.anchor_position)
        path = returns[anchor + 1 : anchor + 21]
        if len(path) != 20:
            raise ContractError("20日组合收益路径长度异常")
        interval_returns.append(float(np.prod(1.0 + path) - 1.0))
    lo = _lo_adjusted_sharpe(
        returns,
        int(serial_cfg["trading_days_per_year"]),
        int(serial_cfg["lo_bartlett_max_lag"]),
    )
    return {
        "daily_observations": int(len(returns)),
        "ordinary_daily_annualized_sharpe": _annualized_sharpe(returns, 242.0),
        "monthly_observations": int(len(monthly)),
        "monthly_compounded_return_sharpe": _annualized_sharpe(monthly, 12.0),
        "twenty_day_interval_observations": int(len(interval_returns)),
        "twenty_day_interval_return_sharpe": _annualized_sharpe(
            np.asarray(interval_returns), 242.0 / 20.0
        ),
        **lo,
    }


def run_event_concentration(
    context: StudyContext,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = build_bull_events(context)
    returns = context.source_stress_ledger["daily_return"].to_numpy(dtype=float)
    total_log_wealth = float(np.log1p(returns).sum())
    event_rows: list[dict[str, Any]] = []
    event_paths: list[np.ndarray] = []
    for event in events:
        entry = int(event["start_anchor_position"]) + 1
        exit_index = int(event["transition_anchor_position"]) + 1
        if exit_index >= len(returns):
            exit_index = len(returns) - 1
        path = returns[entry : exit_index + 1].copy()
        event_paths.append(path)
        log_contribution = float(np.log1p(path).sum())
        event_rows.append(
            {
                **event,
                "execution_entry_date": context.engine.dates[entry],
                "execution_exit_date": context.engine.dates[exit_index],
                "path_observations": int(len(path)),
                "event_net_return": float(np.prod(1.0 + path) - 1.0),
                "event_log_wealth_contribution": log_contribution,
                "share_of_total_log_wealth": (
                    log_contribution / total_log_wealth
                    if total_log_wealth != 0
                    else math.nan
                ),
            }
        )
    event_table = pd.DataFrame(event_rows).sort_values(
        "event_log_wealth_contribution", ascending=False
    )
    if event_table.empty:
        raise ContractError("连续上涨事件表为空")
    perfect = perfect_predictions(context)
    removal_rows: list[dict[str, Any]] = []
    ranked_events = event_table["event_id"].astype(int).tolist()
    event_by_id = {int(event["event_id"]): event for event in events}
    for remove_count in config["robustness_annex"]["concentration"][
        "remove_largest_event_counts"
    ]:
        prediction = perfect.copy()
        removed_ids = ranked_events[: int(remove_count)]
        for event_id in removed_ids:
            prediction[event_by_id[event_id]["block_positions"]] = 0
        metrics = simulate_predictions(prediction[None, :], context, config).iloc[0]
        removal_rows.append(
            {
                "remove_largest_event_count": int(remove_count),
                "removed_event_ids": removed_ids,
                "stress_net_sharpe": float(metrics["stress_net_sharpe"]),
                "stress_total_return": float(metrics["stress_total_return"]),
                "stress_max_drawdown": float(metrics["stress_max_drawdown"]),
                "trade_count": int(metrics["trade_count"]),
            }
        )

    bootstrap_cfg = config["robustness_annex"]["concentration"]
    repetitions = int(bootstrap_cfg["event_bootstrap_repetitions"])
    rng = np.random.default_rng(int(bootstrap_cfg["event_bootstrap_seed"]))
    sampled = rng.integers(0, len(event_paths), size=(repetitions, len(event_paths)))
    path_sums = np.array([path.sum() for path in event_paths], dtype=float)
    path_square_sums = np.array(
        [np.square(path).sum() for path in event_paths], dtype=float
    )
    sampled_sums = path_sums[sampled].sum(axis=1)
    sampled_square_sums = path_square_sums[sampled].sum(axis=1)
    fixed_n = len(returns)
    variances = (
        sampled_square_sums - sampled_sums * sampled_sums / fixed_n
    ) / (fixed_n - 1)
    variances = np.maximum(variances, 0.0)
    bootstrap_sharpes = np.divide(
        sampled_sums / fixed_n * 242.0,
        np.sqrt(variances) * math.sqrt(242.0),
        out=np.full(repetitions, np.nan),
        where=variances > 0,
    )
    event_returns = np.array(
        [float(np.prod(1.0 + path) - 1.0) for path in event_paths], dtype=float
    )
    concentration = {
        "independent_bull_event_count": int(len(events)),
        "total_oracle_bull_block_count": int((context.states == STATE_BULL).sum()),
        "largest_event_share_of_total_log_wealth": float(
            event_table.iloc[0]["share_of_total_log_wealth"]
        ),
        "largest_two_event_share_of_total_log_wealth": float(
            event_table.head(2)["event_log_wealth_contribution"].sum()
            / total_log_wealth
        ),
        "event_return_mean": float(event_returns.mean()),
        "event_return_median": float(np.median(event_returns)),
        "event_return_win_rate": float((event_returns > 0).mean()),
        "event_return_mean_to_std": (
            float(event_returns.mean() / event_returns.std(ddof=1))
            if len(event_returns) > 1 and event_returns.std(ddof=1) > 0
            else None
        ),
        "remove_largest_events": removal_rows,
        "event_bootstrap": {
            "repetitions": repetitions,
            "sharpe_q05": float(np.quantile(bootstrap_sharpes, 0.05)),
            "sharpe_median": float(np.quantile(bootstrap_sharpes, 0.50)),
            "sharpe_q95": float(np.quantile(bootstrap_sharpes, 0.95)),
            "probability_sharpe_at_least_1_2": float(
                np.mean(bootstrap_sharpes >= 1.2)
            ),
        },
    }
    return event_table, concentration


def _suffix_minimum_passing_count(
    frame: pd.DataFrame, count_column: str, pass_column: str
) -> int | None:
    ordered = frame.sort_values(count_column).reset_index(drop=True)
    passes = ordered[pass_column].astype(bool).to_numpy()
    counts = ordered[count_column].astype(int).to_numpy()
    for index in range(len(ordered)):
        if passes[index:].all():
            return int(counts[index])
    return None


def _prefix_maximum_passing_count(
    frame: pd.DataFrame, count_column: str, pass_column: str
) -> int | None:
    ordered = frame.sort_values(count_column).reset_index(drop=True)
    last: int | None = None
    for row in ordered.itertuples(index=False):
        if not bool(getattr(row, pass_column)):
            break
        last = int(getattr(row, count_column))
    return last


def derive_information_budget(
    context: StudyContext,
    config: dict[str, Any],
    random_axis_summary: pd.DataFrame,
    random_joint_summary: pd.DataFrame,
    adversarial: pd.DataFrame,
    timing: pd.DataFrame,
    whipsaw: pd.DataFrame,
) -> dict[str, Any]:
    random_recall = random_axis_summary.loc[
        random_axis_summary["family"].eq("RECALL_ONLY")
    ]
    random_range = random_axis_summary.loc[
        random_axis_summary["family"].eq("RANGE_CONTAMINATION")
    ]
    random_down = random_axis_summary.loc[
        random_axis_summary["family"].eq("DOWN_CONTAMINATION")
    ]
    adversarial_recall = adversarial.loc[adversarial["family"].eq("RECALL_ONLY")]
    adversarial_range = adversarial.loc[
        adversarial["family"].eq("RANGE_CONTAMINATION")
    ]
    adversarial_down = adversarial.loc[
        adversarial["family"].eq("DOWN_CONTAMINATION")
    ]
    random_min_recall_count = _suffix_minimum_passing_count(
        random_recall, "captured_bull_count", "random_budget_pass"
    )
    adversarial_min_recall_count = _suffix_minimum_passing_count(
        adversarial_recall, "captured_bull_count", "deterministic_budget_pass"
    )
    random_max_range_count = _prefix_maximum_passing_count(
        random_range, "false_range_count", "random_budget_pass"
    )
    adversarial_max_range_count = _prefix_maximum_passing_count(
        adversarial_range, "false_range_count", "deterministic_budget_pass"
    )
    random_max_down_count = _prefix_maximum_passing_count(
        random_down, "false_bear_count", "random_budget_pass"
    )
    adversarial_max_down_count = _prefix_maximum_passing_count(
        adversarial_down, "false_bear_count", "deterministic_budget_pass"
    )
    feasible_joint = random_joint_summary.loc[
        random_joint_summary["random_budget_pass"].astype(bool)
        & random_joint_summary["up_precision"].notna()
    ].sort_values(
        ["up_precision", "up_recall", "down_false_long_rate"],
        ascending=[True, True, True],
    )
    precision_record = (
        None
        if feasible_joint.empty
        else {
            key: _json_scalar(feasible_joint.iloc[0][key])
            for key in (
                "cell_id",
                "up_precision",
                "up_recall",
                "range_false_long_rate",
                "down_false_long_rate",
                "captured_bull_count",
                "false_range_count",
                "false_bear_count",
                "stress_net_sharpe_q05",
                "probability_sharpe_at_least_1_2",
            )
        }
    )
    entry = timing.loc[timing["family"].eq("ENTRY_DELAY")]
    maximum_entry_delay = _prefix_maximum_passing_count(
        entry, "error_days", "deterministic_budget_pass"
    )
    adversarial_whipsaw = whipsaw.loc[
        whipsaw["mode"].eq("ADVERSARIAL_GREEDY")
    ]
    maximum_round_trips = _prefix_maximum_passing_count(
        adversarial_whipsaw,
        "planned_round_trips",
        "deterministic_budget_pass",
    )
    capture_candidates: list[float] = []
    for frame, pass_column, capture_column in (
        (
            random_recall,
            "random_budget_pass",
            "oracle_net_log_wealth_capture_q05",
        ),
        (
            adversarial_recall,
            "deterministic_budget_pass",
            "oracle_net_log_wealth_capture",
        ),
        (timing, "deterministic_budget_pass", "oracle_net_log_wealth_capture"),
    ):
        passed = frame.loc[frame[pass_column].astype(bool), capture_column]
        capture_candidates.extend(
            [float(value) for value in passed if pd.notna(value)]
        )
    minimum_capture = min(capture_candidates) if capture_candidates else None
    return {
        "definitions": {
            "precision": "区间级TP/(TP+震荡FP+大跌FP)，仅在冻结联合随机网格内取95%稳健门通过单元的最低值",
            "recall": "零误入逐轴中从该捕获数起所有更高捕获数均通过的后缀边界",
            "false_long": "捕获全部上涨且另一类误入为零时，从0起所有更低误入数均通过的前缀边界",
            "entry_delay": "连续上涨事件起点统一延迟，0日起连续通过的最大天数",
            "return_capture": "通过的召回或时序场景相对完美Oracle压力净对数财富的最低捕获比例",
            "whipsaw_legs": "对抗性一日错误往返从0起连续通过的最大计划交易腿数",
        },
        "MIN_UP_PRECISION_FOR_SHARPE_1_2": precision_record,
        "MIN_UP_RECALL_FOR_SHARPE_1_2": {
            "random_95pct_robust": {
                "captured_bull_blocks": random_min_recall_count,
                "rate": (
                    random_min_recall_count / 23
                    if random_min_recall_count is not None
                    else None
                ),
            },
            "adversarial": {
                "captured_bull_blocks": adversarial_min_recall_count,
                "rate": (
                    adversarial_min_recall_count / 23
                    if adversarial_min_recall_count is not None
                    else None
                ),
            },
        },
        "MAX_DOWN_FALSE_LONG_RATE": {
            "random_95pct_robust": {
                "false_bear_blocks": random_max_down_count,
                "rate": (
                    random_max_down_count / 21
                    if random_max_down_count is not None
                    else None
                ),
            },
            "adversarial": {
                "false_bear_blocks": adversarial_max_down_count,
                "rate": (
                    adversarial_max_down_count / 21
                    if adversarial_max_down_count is not None
                    else None
                ),
            },
        },
        "MAX_RANGE_FALSE_LONG_RATE": {
            "random_95pct_robust": {
                "false_range_blocks": random_max_range_count,
                "rate": (
                    random_max_range_count / 97
                    if random_max_range_count is not None
                    else None
                ),
            },
            "adversarial": {
                "false_range_blocks": adversarial_max_range_count,
                "rate": (
                    adversarial_max_range_count / 97
                    if adversarial_max_range_count is not None
                    else None
                ),
            },
        },
        "MAX_ENTRY_DELAY_DAYS": maximum_entry_delay,
        "MIN_ORACLE_RETURN_CAPTURE": minimum_capture,
        "MAX_ACCEPTABLE_WHIPSAW_LEGS": (
            maximum_round_trips * 2 if maximum_round_trips is not None else None
        ),
        "MAX_ACCEPTABLE_WHIPSAW_ROUND_TRIPS": maximum_round_trips,
    }


def robustness_summary(
    context: StudyContext,
    basis: pd.DataFrame,
    offsets: pd.DataFrame,
    serial: dict[str, Any],
    concentration: dict[str, Any],
) -> dict[str, Any]:
    bull_basis = basis.loc[basis["state"].eq(STATE_BULL)]
    correlation = float(
        bull_basis[
            [
                "h00300_label_close_to_close_return",
                "etf_executable_open_to_next_execution_open_total_return",
            ]
        ].corr().iloc[0, 1]
    )
    return {
        "phase_offsets": {
            "count": int(len(offsets)),
            "stress_net_sharpe_minimum": float(offsets["stress_net_sharpe"].min()),
            "stress_net_sharpe_q05": float(
                np.quantile(offsets["stress_net_sharpe"], 0.05)
            ),
            "stress_net_sharpe_median": float(
                np.quantile(offsets["stress_net_sharpe"], 0.50)
            ),
            "stress_net_sharpe_maximum": float(offsets["stress_net_sharpe"].max()),
            "all_offsets_at_least_1_2": bool(
                (offsets["stress_net_sharpe"] >= 1.2).all()
            ),
        },
        "label_execution_basis": {
            "complete_block_count": int(len(basis)),
            "bull_block_count": int(len(bull_basis)),
            "bull_label_return_mean": float(
                bull_basis["h00300_label_close_to_close_return"].mean()
            ),
            "bull_executable_return_mean": float(
                bull_basis[
                    "etf_executable_open_to_next_execution_open_total_return"
                ].mean()
            ),
            "bull_execution_minus_label_mean": float(
                bull_basis["execution_minus_label_return"].mean()
            ),
            "bull_label_execution_correlation": correlation,
            "bull_entry_untradeable_overnight_mean": float(
                bull_basis["entry_untradeable_overnight_price_return"].mean()
            ),
            "bull_exit_actual_overnight_mean": float(
                bull_basis["exit_actual_overnight_total_return"].mean()
            ),
            "bull_entry_untradeable_overnight_log_sum": float(
                np.log1p(
                    bull_basis["entry_untradeable_overnight_price_return"]
                ).sum()
            ),
            "bull_executable_return_log_sum": float(
                np.log1p(
                    bull_basis[
                        "etf_executable_open_to_next_execution_open_total_return"
                    ]
                ).sum()
            ),
        },
        "serial_correlation": serial,
        "event_concentration": concentration,
    }


def adjudicate_advancement(
    budget: dict[str, Any],
    robustness: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate_cfg = config["conservative_forecast_research_advancement_gate"]
    adversarial_recall = budget["MIN_UP_RECALL_FOR_SHARPE_1_2"]["adversarial"][
        "rate"
    ]
    adversarial_down = budget["MAX_DOWN_FALSE_LONG_RATE"]["adversarial"][
        "false_bear_blocks"
    ]
    adversarial_range = budget["MAX_RANGE_FALSE_LONG_RATE"]["adversarial"][
        "false_range_blocks"
    ]
    remove_one = robustness["event_concentration"]["remove_largest_events"][0][
        "stress_net_sharpe"
    ]
    gates = {
        "adversarial_up_recall_not_near_perfect": (
            adversarial_recall is not None
            and adversarial_recall
            <= float(gate_cfg["maximum_required_adversarial_up_recall"])
        ),
        "adversarial_down_error_tolerance_nonzero": (
            adversarial_down is not None
            and adversarial_down
            >= int(gate_cfg["minimum_adversarial_down_false_long_blocks_tolerated"])
        ),
        "adversarial_range_error_tolerance_material": (
            adversarial_range is not None
            and adversarial_range
            >= int(gate_cfg["minimum_adversarial_range_false_long_blocks_tolerated"])
        ),
        "entry_delay_tolerance": (
            budget["MAX_ENTRY_DELAY_DAYS"] is not None
            and budget["MAX_ENTRY_DELAY_DAYS"]
            >= int(gate_cfg["minimum_entry_delay_days_tolerated"])
        ),
        "phase_offset_robustness": robustness["phase_offsets"][
            "stress_net_sharpe_q05"
        ]
        >= float(gate_cfg["phase_offset_fifth_percentile_stress_sharpe_minimum"]),
        "lo_adjusted_sharpe": robustness["serial_correlation"][
            "lo_adjusted_annualized_sharpe"
        ]
        >= float(gate_cfg["lo_adjusted_stress_sharpe_minimum"]),
        "remove_largest_event": remove_one
        >= float(gate_cfg["remove_largest_one_event_stress_sharpe_minimum"]),
        "event_bootstrap_fifth_percentile": robustness["event_concentration"][
            "event_bootstrap"
        ]["sharpe_q05"]
        >= float(gate_cfg["event_bootstrap_fifth_percentile_sharpe_minimum"]),
    }
    passed = bool(all(gates.values()))
    return {
        "conservative_up20_forecast_research_advancement_gate_passed": passed,
        "gates": gates,
        "interpretation": (
            "历史信息预算在冻结的随机、对抗和稳健性门下足够宽，可以启动低自由度UP20预测器研究"
            if passed
            else "完美信息价值仍成立，但至少一项冻结预算或稳健性门显示现实可实现空间偏窄"
        ),
        "realistic_up20_forecast": "NOT_EVALUATED",
        "historical_tradable_strategy_target_achieved": False,
        "verified_forward_target_achieved": False,
        "goal_achieved": False,
        "live_trading_authorized": False,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if pd.isna(value):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_scalar(value)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}%}"


def render_markdown(report: dict[str, Any]) -> str:
    budget = report["information_budget"]
    precision = budget["MIN_UP_PRECISION_FOR_SHARPE_1_2"]
    recall = budget["MIN_UP_RECALL_FOR_SHARPE_1_2"]
    down = budget["MAX_DOWN_FALSE_LONG_RATE"]
    range_budget = budget["MAX_RANGE_FALSE_LONG_RATE"]
    robustness = report["robustness"]
    adjudication = report["adjudication"]
    lines = [
        "# 510300 未来20日上涨机会信息质量预算 V1 结果",
        "",
        f"- 状态：`{report['status']}`",
        "- 研究性质：已知源Oracle结果后的信息降级与稳健性诊断；不是预测模型，不是可交易策略。",
        "- 源上界：未来20日上涨区间完美已知时，压力净夏普1.740。",
        "- 目标：反推压力净夏普仍不低于1.2时可承受的分类、时序与信号反复误差。",
        "",
        "## 核心工程预算",
        "",
        "| 预算 | 随机95%稳健边界 | 对抗边界 |",
        "|---|---:|---:|",
        f"| 最低上涨召回率 | {_pct(recall['random_95pct_robust']['rate'])}（{recall['random_95pct_robust']['captured_bull_blocks']}/23） | {_pct(recall['adversarial']['rate'])}（{recall['adversarial']['captured_bull_blocks']}/23） |",
        f"| 最大大跌误入率 | {_pct(down['random_95pct_robust']['rate'])}（{down['random_95pct_robust']['false_bear_blocks']}/21） | {_pct(down['adversarial']['rate'])}（{down['adversarial']['false_bear_blocks']}/21） |",
        f"| 最大震荡误入率 | {_pct(range_budget['random_95pct_robust']['rate'])}（{range_budget['random_95pct_robust']['false_range_blocks']}/97） | {_pct(range_budget['adversarial']['rate'])}（{range_budget['adversarial']['false_range_blocks']}/97） |",
        "",
        f"- 冻结联合随机网格最低通过精确率：`{_pct(None if precision is None else precision['up_precision'])}`。",
        f"- 最大连续可承受入场延迟：`{budget['MAX_ENTRY_DELAY_DAYS']}`个交易日。",
        f"- 通过场景最低Oracle净对数财富捕获：`{_pct(budget['MIN_ORACLE_RETURN_CAPTURE'])}`。",
        f"- 对抗性一日错误往返最大连续可承受额外交易腿：`{budget['MAX_ACCEPTABLE_WHIPSAW_LEGS']}`腿。",
        "",
        "精确率不是通用单一门槛；上值只适用于冻结联合网格，并必须与对应召回率、震荡误入率和大跌误入率一起读取。",
        "",
        "## Oracle稳健性附件",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
        f"| 20种起点相位压力夏普中位数 | {_fmt(robustness['phase_offsets']['stress_net_sharpe_median'])} |",
        f"| 20种相位压力夏普5%分位数 / 最差值 | {_fmt(robustness['phase_offsets']['stress_net_sharpe_q05'])} / {_fmt(robustness['phase_offsets']['stress_net_sharpe_minimum'])} |",
        f"| 普通日度 / 月度 / 20日区间夏普 | {_fmt(robustness['serial_correlation']['ordinary_daily_annualized_sharpe'])} / {_fmt(robustness['serial_correlation']['monthly_compounded_return_sharpe'])} / {_fmt(robustness['serial_correlation']['twenty_day_interval_return_sharpe'])} |",
        f"| Lo/HAC调整夏普（q=20） | {_fmt(robustness['serial_correlation']['lo_adjusted_annualized_sharpe'])} |",
        f"| 独立连续上涨事件数 | {robustness['event_concentration']['independent_bull_event_count']} |",
        f"| 最大一轮 / 两轮对总对数财富贡献 | {_pct(robustness['event_concentration']['largest_event_share_of_total_log_wealth'])} / {_pct(robustness['event_concentration']['largest_two_event_share_of_total_log_wealth'])} |",
        f"| 事件bootstrap夏普5%分位数 | {_fmt(robustness['event_concentration']['event_bootstrap']['sharpe_q05'])} |",
        "",
        "## 标签—执行基差",
        "",
        f"在23个大涨标签区间中，H00300标签平均收益为`{_pct(robustness['label_execution_basis']['bull_label_return_mean'])}`，510300从T+1开盘到下一次执行开盘的含分红可执行毛收益平均为`{_pct(robustness['label_execution_basis']['bull_executable_return_mean'])}`，平均差为`{_pct(robustness['label_execution_basis']['bull_execution_minus_label_mean'])}`。",
        f"不可交易的入场隔夜平均收益为`{_pct(robustness['label_execution_basis']['bull_entry_untradeable_overnight_mean'])}`；该项已单独量化，没有被当作可执行收益。",
        "",
        "## 晋级判断",
        "",
        f"- 保守UP20预测研究晋级门：`{'PASS' if adjudication['conservative_up20_forecast_research_advancement_gate_passed'] else 'FAIL'}`。",
        f"- 解释：{adjudication['interpretation']}。",
        "- 现实UP20预测器：`NOT_EVALUATED`。",
        "- 历史可交易策略夏普1.2目标：`NOT_ACHIEVED`。",
        "- Paper、Shadow、持仓、订单和实盘授权：全部为`FALSE`。",
        "",
        "## 正确使用方式",
        "",
        "本结果把现实预测器需要达到的性能转化为冻结工程门槛。它不允许拿未来20日标签生成历史交易信号，也不能证明任何技术因子已经具有该预测质量。",
        "",
    ]
    return "\n".join(lines)


def run_study(write: bool = True, progress: bool = False) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    context = load_context(config)
    if progress:
        print("[1/8] 源Oracle合同与输入哈希通过，复核批量账户引擎。", flush=True)
    engine_replay = verify_fast_engine(context, config)
    basis = build_label_execution_basis(context)
    if progress:
        print("[2/8] 运行随机分类误差预算：71,000个逐轴样本与108,000个联合网格样本。", flush=True)
    random_axis_draws, random_axis_summary, random_joint_draws, random_joint_summary = (
        run_random_error_experiments(context, config)
    )
    if progress:
        print("[3/8] 运行漏掉最佳上涨、误入最差震荡和最差大跌的对抗预算。", flush=True)
    adversarial = run_adversarial_error_experiments(context, config, basis)
    if progress:
        print("[4/8] 运行0至10日进入、提前退出和延后退出预算。", flush=True)
    timing = run_timing_experiments(context, config)
    if progress:
        print("[5/8] 运行随机与对抗性一日错误往返预算。", flush=True)
    whipsaw = run_whipsaw_experiments(context, config)
    if progress:
        print("[6/8] 运行20种阶段起点相位及标签—执行基差检查。", flush=True)
    offsets = run_phase_offset_robustness(context, config)
    serial = serial_correlation_metrics(context, config)
    if progress:
        print("[7/8] 合并连续上涨事件，执行集中度、剔除最大事件和事件bootstrap。", flush=True)
    event_table, concentration = run_event_concentration(context, config)
    budget = derive_information_budget(
        context,
        config,
        random_axis_summary,
        random_joint_summary,
        adversarial,
        timing,
        whipsaw,
    )
    robust = robustness_summary(context, basis, offsets, serial, concentration)
    adjudication = adjudicate_advancement(budget, robust, config)
    status = (
        "INFORMATION_BUDGET_QUANTIFIED_CONSERVATIVE_GATE_PASS_PROCEED_TO_UP20_FORECAST_RESEARCH_NOT_TRADABLE"
        if adjudication[
            "conservative_up20_forecast_research_advancement_gate_passed"
        ]
        else "INFORMATION_BUDGET_QUANTIFIED_CONSERVATIVE_GATE_FAILED_REALISTIC_SPACE_NARROW_OR_FRAGILE_NOT_TRADABLE"
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "research_question": config["research_question"],
        "source_oracle_boundary": {
            "source_project_id": config["source_oracle"]["project_id"],
            "source_oracle_outcomes_already_known": True,
            "source_conditional_policy_value": "PASS",
            "source_primary_stress_net_sharpe": float(
                config["source_oracle"]["expected_primary_stress_net_sharpe"]
            ),
            "source_result_unchanged": True,
            "future_information_used": True,
            "prediction_model_evaluated": False,
            "not_a_tradable_signal": True,
        },
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
            "trading_days": int(len(context.engine.dates)),
            "complete_oracle_blocks": int(context.blocks["complete"].sum()),
            "state_counts": {
                state: int((context.states == state).sum())
                for state in COMPLETE_STATES
            },
        },
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "source_oracle_outcomes_already_known": True,
            "information_budget_outputs_computed_before_freeze": False,
        },
        "engine_replay": engine_replay,
        "simulation_counts": {
            "random_axis_draws": int(len(random_axis_draws)),
            "random_joint_draws": int(len(random_joint_draws)),
            "adversarial_scenarios": int(len(adversarial)),
            "timing_scenarios": int(len(timing)),
            "whipsaw_rows": int(len(whipsaw)),
            "phase_offsets": int(len(offsets)),
            "event_bootstrap_repetitions": int(
                config["robustness_annex"]["concentration"][
                    "event_bootstrap_repetitions"
                ]
            ),
        },
        "information_budget": budget,
        "robustness": robust,
        "adjudication": adjudication,
        "governance": config["governance"],
        "artifacts": {},
    }
    if write:
        path_map = {key: _project_path(value) for key, value in config["paths"].items()}
        artifact_frames = {
            "random_axis_draws": random_axis_draws,
            "random_axis_summary": random_axis_summary,
            "random_joint_draws": random_joint_draws,
            "random_joint_summary": random_joint_summary,
            "adversarial_summary": adversarial,
            "timing_summary": timing,
            "whipsaw_summary": whipsaw,
            "offset_robustness": offsets,
            "label_execution_basis": basis,
            "event_concentration": event_table,
        }
        for key, frame in artifact_frames.items():
            path = path_map[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        report["artifacts"] = {
            key: _artifact_record(path_map[key]) for key in artifact_frames
        }
        result_json = path_map["result_json"]
        result_markdown = path_map["result_markdown"]
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result_markdown.write_text(render_markdown(_json_safe(report)), encoding="utf-8")
    if progress:
        print("[8/8] 信息预算、稳健性附件与晋级判断已完成。", flush=True)
    return _json_safe(report)
