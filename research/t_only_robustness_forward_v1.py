"""冻结 T_ONLY 的执行压力测试器与前瞻影子账本核心。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from research.graph_regime_martin_turtle_v2 import CostModel, _events_by_date
from research.weekly_daily_technical_v1 import _target_with_minimum_gate


ROOT = Path(__file__).resolve().parents[1]
STRESS_CONFIG_FILE = ROOT / "config" / "t_only_forward_v1.yaml"


def load_stress_config(path: Path = STRESS_CONFIG_FILE) -> dict[str, Any]:
    """读取并检查压力测试及前瞻协议。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("T_ONLY 前瞻配置必须是 YAML 对象")
    scenario_ids = [item["id"] for item in config["stress_test"]["scenarios"]]
    expected = [
        "BASELINE_REPLICATION",
        "DELAY_1_EXTRA_BAR",
        "SLIPPAGE_10BP",
        "SLIPPAGE_20BP",
        "COMMISSION_2X",
        "COMMISSION_3X",
        "MISS_10PCT_ENTRIES",
    ]
    if scenario_ids != expected:
        raise ValueError("压力场景集合或顺序与冻结协议不一致")
    governance = config["governance"]
    if any(
        governance[key]
        for key in (
            "live_position_mapping_enabled",
            "order_generation_enabled",
            "broker_connection_enabled",
        )
    ):
        raise ValueError("安全开关必须保持关闭")
    if config["protocol"]["parameter_changes_after_freeze"] is not False:
        raise ValueError("冻结后禁止依据结果修改参数")
    return config


def _stress_cost_model(
    base_config: dict[str, Any], *, commission_multiplier: float, slippage_bps: float
) -> CostModel:
    execution = base_config["price_and_execution"]
    return CostModel(
        commission_rate=float(execution["commission_rate_per_leg"])
        * commission_multiplier,
        minimum_commission=float(execution["minimum_commission_cny_per_leg"])
        * commission_multiplier,
        slippage_rate=slippage_bps / 10_000.0,
        multiplier=1.0,
    )


def simulate_t_only(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    base_config: dict[str, Any],
    *,
    scenario_id: str,
    execution_delay_bars: int = 0,
    commission_multiplier: float = 1.0,
    slippage_bps: float = 5.0,
    blocked_entry_signal_dates: Iterable[pd.Timestamp] = (),
    allow_terminal_pending_signal: bool = False,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """运行 T_ONLY；零额外延迟时应逐日复现父协议实现。"""

    if execution_delay_bars < 0:
        raise ValueError("额外执行延迟不得为负数")
    if commission_multiplier < 0.0 or slippage_bps < 0.0:
        raise ValueError("佣金倍数和滑点不得为负数")
    frame = features.sort_values("date").reset_index(drop=True).copy()
    execution = base_config["price_and_execution"]
    trend = base_config["trend_module"]
    target_exposures = [float(value) for value in execution["target_exposures"]]
    lot_size = int(execution["lot_size_shares"])
    minimum_notional = float(execution["minimum_normal_trade_notional_cny"])
    costs = _stress_cost_model(
        base_config,
        commission_multiplier=commission_multiplier,
        slippage_bps=slippage_bps,
    )
    daily_cash_rate = (1.0 + float(execution["cash_annual_rate"])) ** (
        1.0 / int(execution["trading_days_per_year"])
    ) - 1.0
    blocked_dates = {pd.Timestamp(value).normalize() for value in blocked_entry_signal_dates}

    record_events = _events_by_date(dividends, "record_date")
    ex_events = _events_by_date(dividends, "ex_date")
    payment_events = _events_by_date(dividends, "payment_date")
    entitlements: dict[pd.Timestamp, float] = {}
    receivables_by_payment: dict[pd.Timestamp, float] = defaultdict(float)

    cash = float(execution["initial_capital_cny"])
    shares = 0
    receivable = 0.0
    pending: dict[str, Any] | None = None
    mode = "CASH"
    tier = 0
    frozen_atr = np.nan
    last_fill_adjusted = np.nan
    holding_days = 0
    active_cycle: dict[str, Any] | None = None
    ledger_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        date = pd.Timestamp(row["date"]).normalize()
        for event in ex_events.get(date, []):
            amount = entitlements.get(event.ex_date, 0.0)
            receivable += amount
            receivables_by_payment[event.payment_date] += amount
        if date in payment_events:
            amount = receivables_by_payment.pop(date, 0.0)
            cash += amount
            receivable -= amount

        shares_before = shares
        mode_before = mode
        executed_reason: str | None = None
        day_commission = 0.0
        day_slippage = 0.0
        day_notional = 0.0

        pending_waiting = pending is not None and index < int(pending["due_index"])
        if pending is not None and index >= int(pending["due_index"]):
            executed_reason = str(pending["reason"])
            target = float(pending["target_exposure"])
            cash_after, shares_after, trade, rejection, preview_notional = (
                _target_with_minimum_gate(
                    target_exposure=target,
                    raw_open=float(row["open"]),
                    cash=cash,
                    shares=shares,
                    receivable=receivable,
                    lot_size=lot_size,
                    costs=costs,
                    minimum_notional=minimum_notional,
                    allow_small_liquidation=bool(
                        execution["risk_liquidation_exempt_from_minimum_notional"]
                    ),
                )
            )
            if rejection is not None:
                skipped_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "signal_date": pending["signal_date"],
                        "execution_date": date,
                        "reason": executed_reason,
                        "rejection": rejection,
                        "target_exposure": target,
                        "preview_raw_notional": preview_notional,
                    }
                )
            else:
                cash, shares = cash_after, shares_after
                if trade is not None:
                    day_commission = float(trade["commission"])
                    day_slippage = float(trade["slippage_cost"])
                    day_notional = float(trade["raw_notional"])
                    execution_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "signal_date": pending["signal_date"],
                            "execution_date": date,
                            "reason": executed_reason,
                            "mode_before": mode_before,
                            "side": trade["side"],
                            "trade_shares": int(trade["shares"]),
                            "raw_open": float(row["open"]),
                            "execution_price": float(trade["execution_price"]),
                            "raw_notional": day_notional,
                            "commission": day_commission,
                            "slippage_cost": day_slippage,
                            "target_exposure": target,
                            "shares_after": shares,
                        }
                    )
                    if executed_reason == "TREND_ENTRY":
                        mode, tier, holding_days = "TREND", 1, 0
                        frozen_atr = float(pending["atr14"])
                        last_fill_adjusted = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                    elif executed_reason.startswith("TREND_ADD_"):
                        mode = "TREND"
                        tier = int(executed_reason.rsplit("_", 1)[-1])
                        last_fill_adjusted = float(trade["execution_price"]) * float(
                            row["adjustment_factor"]
                        )
                    elif target == 0.0:
                        mode, tier, holding_days = "CASH", 0, 0
                        frozen_atr = np.nan
                        last_fill_adjusted = np.nan
            pending = None
            pending_waiting = False

        if shares_before == 0 and shares > 0:
            active_cycle = {
                "scenario_id": scenario_id,
                "entry_date": date,
                "entry_reason": executed_reason,
                "entry_equity": cash + shares * float(row["open"]) + receivable,
            }
        elif shares_before > 0 and shares == 0 and active_cycle is not None:
            exit_equity = cash + receivable
            cycle_rows.append(
                {
                    **active_cycle,
                    "exit_date": date,
                    "exit_reason": executed_reason,
                    "exit_equity": exit_equity,
                    "net_pnl": exit_equity - float(active_cycle["entry_equity"]),
                    "holding_calendar_days": (date - active_cycle["entry_date"]).days,
                }
            )
            active_cycle = None

        cash *= 1.0 + daily_cash_rate
        equity = cash + shares * float(row["close"]) + receivable
        exposure = shares * float(row["close"]) / equity if equity > 0 else np.nan
        for event in record_events.get(date, []):
            entitlements[event.ex_date] = shares * float(event.cash_dividend_per_share)
        if shares > 0:
            holding_days += 1

        next_signal: dict[str, Any] | None = None
        weekly_state = str(row["weekly_state_at_close"])
        adjusted_close = float(row["adjusted_close"])
        atr14 = float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
        hh20 = float(row["hh20"]) if pd.notna(row["hh20"]) else np.nan
        ll10 = float(row["ll10"]) if pd.notna(row["ll10"]) else np.nan
        due_index = index + 1 + execution_delay_bars
        has_execution_day = due_index < len(frame) or (
            allow_terminal_pending_signal and due_index == len(frame)
        )

        def order(reason: str, target_tier: int) -> dict[str, Any]:
            return {
                "signal_date": date,
                "reason": reason,
                "target_exposure": (
                    target_exposures[target_tier - 1] if target_tier > 0 else 0.0
                ),
                "atr14": atr14,
                "due_index": due_index,
            }

        if not pending_waiting and pending is None and has_execution_day:
            if mode == "TREND":
                if weekly_state == "W_BEAR":
                    next_signal = order("TREND_EXIT_WEEKLY_BEAR", 0)
                elif np.isfinite(ll10) and adjusted_close < ll10:
                    next_signal = order("TREND_EXIT_DONCHIAN_10", 0)
                elif (
                    np.isfinite(frozen_atr)
                    and adjusted_close
                    <= last_fill_adjusted
                    - float(trend["stop_from_last_fill_atr"]) * frozen_atr
                ):
                    next_signal = order("TREND_EXIT_2ATR_STOP", 0)
                elif (
                    weekly_state == "W_BULL"
                    and tier < int(trend["maximum_tier"])
                    and np.isfinite(frozen_atr)
                    and adjusted_close
                    >= last_fill_adjusted
                    + float(trend["add_atr_interval"]) * frozen_atr
                ):
                    next_signal = order(f"TREND_ADD_{tier + 1}", tier + 1)
            elif (
                weekly_state == "W_BULL"
                and np.isfinite(hh20)
                and adjusted_close > hh20
                and np.isfinite(atr14)
            ):
                if date in blocked_dates:
                    missed_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "signal_date": date,
                            "reason": "TREND_ENTRY",
                            "disposition": "MISSED_BY_FROZEN_STRESS",
                        }
                    )
                else:
                    next_signal = order("TREND_ENTRY", 1)
        if next_signal is not None:
            pending = next_signal

        ledger_rows.append(
            {
                "scenario_id": scenario_id,
                "date": date,
                "equity": equity,
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "exposure": exposure,
                "mode": mode,
                "tier": tier,
                "holding_days": holding_days,
                "weekly_state_at_close": weekly_state,
                "day_commission": day_commission,
                "day_slippage": day_slippage,
                "day_raw_notional": day_notional,
                "executed_reason": executed_reason,
                "next_signal_reason": next_signal["reason"] if next_signal else None,
            }
        )

    return {
        "ledger": pd.DataFrame(ledger_rows),
        "executions": pd.DataFrame(execution_rows),
        "skipped_orders": pd.DataFrame(skipped_rows),
        "missed_signals": pd.DataFrame(missed_rows),
        "cycles": pd.DataFrame(cycle_rows),
        "state": {
            "ending_mode": mode,
            "ending_tier": tier,
            "open_cycle": active_cycle is not None,
            "pending_signal_suppressed": pending is not None,
            "pending_signal": (
                {
                    "signal_date": pending["signal_date"],
                    "reason": pending["reason"],
                    "target_exposure": pending["target_exposure"],
                }
                if pending is not None
                else None
            ),
        },
    }


def select_missed_entry_dates(
    baseline_executions: pd.DataFrame, *, fraction: float, random_seed: int
) -> list[pd.Timestamp]:
    """按固定种子从基线实际入场信号中抽取遗漏日期。"""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("遗漏比例必须位于 0 至 1")
    dates = (
        baseline_executions.loc[
            baseline_executions["reason"].eq("TREND_ENTRY"), "signal_date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not dates or fraction == 0.0:
        return []
    sample_size = max(1, int(np.ceil(len(dates) * fraction)))
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(len(dates), size=sample_size, replace=False)
    return sorted(pd.Timestamp(dates[index]).normalize() for index in selected)


def replication_audit(
    parent: dict[str, Any], replica: dict[str, Any], *, tolerance: float = 1e-9
) -> dict[str, Any]:
    """核验专用执行器是否逐日复现父 T_ONLY。"""

    parent_ledger = parent["ledger"].reset_index(drop=True)
    replica_ledger = replica["ledger"].reset_index(drop=True)
    if len(parent_ledger) != len(replica_ledger):
        return {"status": "FAIL", "reason": "LEDGER_LENGTH_MISMATCH"}
    dates_equal = parent_ledger["date"].equals(replica_ledger["date"])
    max_equity_difference = float(
        np.max(np.abs(parent_ledger["equity"] - replica_ledger["equity"]))
    )
    shares_equal = parent_ledger["shares"].equals(replica_ledger["shares"])
    parent_exec = parent["executions"].reset_index(drop=True)
    replica_exec = replica["executions"].reset_index(drop=True)
    event_columns = [
        "signal_date",
        "execution_date",
        "reason",
        "side",
        "trade_shares",
        "shares_after",
    ]
    executions_equal = len(parent_exec) == len(replica_exec) and parent_exec[
        event_columns
    ].equals(replica_exec[event_columns])
    status = (
        "PASS"
        if dates_equal
        and shares_equal
        and executions_equal
        and max_equity_difference <= tolerance
        else "FAIL"
    )
    return {
        "status": status,
        "dates_equal": bool(dates_equal),
        "shares_equal": bool(shares_equal),
        "executions_equal": bool(executions_equal),
        "maximum_absolute_equity_difference_cny": max_equity_difference,
        "tolerance_cny": tolerance,
        "ledger_rows": int(len(parent_ledger)),
        "execution_legs": int(len(parent_exec)),
    }


def profitable_cycle_concentration(cycles: pd.DataFrame) -> float | None:
    """返回最大单个盈利周期占全部正盈利的比例。"""

    if cycles.empty or "net_pnl" not in cycles:
        return None
    positive = pd.to_numeric(cycles["net_pnl"], errors="coerce").dropna()
    positive = positive.loc[positive > 0.0]
    if positive.empty:
        return None
    return float(positive.max() / positive.sum())


__all__ = [
    "ROOT",
    "load_stress_config",
    "profitable_cycle_concentration",
    "replication_audit",
    "select_missed_entry_dates",
    "simulate_t_only",
]
