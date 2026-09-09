"""把开发期有效的期权风险信息转成低换手的510300二元状态机。

本版本与V1单日脉冲分开：V1已被永久淘汰，V2只检验预先限定的固定延续和
滞回解除机制。期权仍只提供信息，实际资产只能是510300或人民币现金。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import global_liquidity_regime_5d_discovery_v0 as market_module
import option_information_binary_discovery_v1 as v1
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_OPTION_INFORMATION_STATE_MACHINE_DISCOVERY_V2"
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_information_state_machine_discovery_v2"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_information_state_machine_discovery_v2"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "discovery"
    / "510300_option_information_state_machine_discovery_v2.json"
)


@dataclass(frozen=True)
class StatePolicy:
    policy_type: str
    score_column: str
    trigger: float
    recovery: float | None = None
    hold_days: int | None = None


def _prepare_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not v1.OUTPUT_FEATURES.is_file():
        raise FileNotFoundError(f"缺少V1开发期特征：{v1.OUTPUT_FEATURES}")
    features = pd.read_parquet(v1.OUTPUT_FEATURES)
    features["date"] = pd.to_datetime(features["date"], errors="raise").dt.normalize()
    if features.empty or features["date"].max() > v1.DEVELOPMENT_CUTOFF:
        raise ValueError("V1特征为空或包含2024年及以后数据")
    features = features.sort_values("date").drop_duplicates("date", keep="last")

    market_module.DEVELOPMENT_CUTOFF = v1.DEVELOPMENT_CUTOFF
    dividends = market_module._load_dividends()
    market = market_module._load_market(dividends).copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.loc[market["date"].le(v1.DEVELOPMENT_CUTOFF)].sort_values("date")
    frame = market.merge(features, on="date", how="left", validate="one_to_one").reset_index(drop=True)

    gb = frame["rank252_gamma_balance"]
    gs = frame["rank252_gamma_signed"]
    db = frame["rank252_delta_balance"]
    frame["core_broad_count"] = pd.concat([gb.ge(0.80), gs.ge(0.80), db.ge(0.80)], axis=1).sum(axis=1)
    frame["core_narrow_count"] = pd.concat([gb.ge(0.85), gs.ge(0.85), db.ge(0.85)], axis=1).sum(axis=1)
    frame["core_median_rank"] = pd.concat([gb, gs, db], axis=1).median(axis=1, skipna=False)
    audit = {
        "feature_rows": int(len(features)),
        "feature_first_date": features["date"].min().date().isoformat(),
        "feature_last_date": features["date"].max().date().isoformat(),
        "selection_rows": int(frame["date"].between(v1.SELECTION_START, v1.SELECTION_END).sum()),
        "post_2023_rows_read": 0,
        "v1_feature_sha256": v1._sha256(v1.OUTPUT_FEATURES),
    }
    return frame, dividends, audit


def _make_policies() -> dict[str, StatePolicy]:
    policies: dict[str, StatePolicy] = {}
    for score_name, trigger in [("core_broad_count", 2.0), ("core_narrow_count", 2.0)]:
        label = "BROAD" if score_name == "core_broad_count" else "NARROW"
        for hold_days in [2, 3, 5]:
            policies[f"FIXED_CORE_{label}_VOTE2_HOLD{hold_days:02d}"] = StatePolicy(
                "FIXED_HOLD", score_name, trigger, hold_days=hold_days
            )
    for hold_days in [2, 3, 5]:
        policies[f"FIXED_GAMMA_BALANCE_TOP10_HOLD{hold_days:02d}"] = StatePolicy(
            "FIXED_HOLD", "rank252_gamma_balance", 0.90, hold_days=hold_days
        )

    for trigger, recoveries in [(0.80, [0.50, 0.60, 0.70]), (0.85, [0.50, 0.60, 0.70]), (0.90, [0.60, 0.70])]:
        for recovery in recoveries:
            policies[
                f"HYST_CORE_MEDIAN_TRIG{int(trigger * 100):02d}_REC{int(recovery * 100):02d}"
            ] = StatePolicy(
                "HYSTERESIS",
                "core_median_rank",
                trigger,
                recovery=recovery,
            )
    for recovery in [0.50, 0.60, 0.70]:
        policies[f"HYST_GAMMA_BALANCE_TRIG90_REC{int(recovery * 100):02d}"] = StatePolicy(
            "HYSTERESIS",
            "rank252_gamma_balance",
            0.90,
            recovery=recovery,
        )
    return policies


def _states_for_range(
    frame: pd.DataFrame,
    first_execution_index: int,
    last_execution_index: int,
    spec: StatePolicy,
) -> tuple[np.ndarray, pd.DataFrame]:
    anchor_index = first_execution_index - 1
    states = np.zeros(last_execution_index - anchor_index + 1, dtype=np.int8)
    previous_state = 1
    remaining_cash_days = 0
    decision_rows: list[dict[str, Any]] = []

    for execution_index in range(first_execution_index, last_execution_index + 1):
        signal_index = execution_index - 1
        signal_date = pd.Timestamp(frame.loc[signal_index, "date"])
        execution_date = pd.Timestamp(frame.loc[execution_index, "date"])
        score = pd.to_numeric(pd.Series([frame.loc[signal_index, spec.score_column]]), errors="coerce").iloc[0]
        finite = bool(pd.notna(score) and np.isfinite(float(score)))
        trigger_now = bool(finite and float(score) >= spec.trigger)

        if spec.policy_type == "FIXED_HOLD":
            if trigger_now:
                remaining_cash_days = max(remaining_cash_days, int(spec.hold_days or 1))
            if remaining_cash_days > 0:
                state = 0
                remaining_cash_days -= 1
            else:
                state = 1
        elif spec.policy_type == "HYSTERESIS":
            if previous_state == 1:
                state = 0 if trigger_now else 1
            else:
                recovery = float(spec.recovery if spec.recovery is not None else spec.trigger)
                state = 1 if finite and float(score) < recovery else 0
        else:
            raise ValueError(f"未知状态机类型：{spec.policy_type}")

        states[execution_index - anchor_index] = state
        decision_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "score": v1._safe_float(score),
                "trigger_now": trigger_now,
                "state": state,
            }
        )
        previous_state = state
    return states, pd.DataFrame(decision_rows)


def _evaluate(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy_name: str,
    spec: StatePolicy,
    start_perturbation: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_indices = np.flatnonzero(
        frame["date"].between(v1.SELECTION_START, v1.SELECTION_END).to_numpy()
    )
    first_execution_index = int(selection_indices[start_perturbation])
    last_execution_index = int(selection_indices[-1])
    anchor_index = first_execution_index - 1
    if anchor_index < 0:
        raise ValueError("缺少首个执行日的前一交易日信号锚点")
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states, decisions = _states_for_range(frame, first_execution_index, last_execution_index, spec)

    benchmark = build_benchmark_ledger(sample, market_module.INITIAL_CAPITAL_CNY)
    base_ledger, base_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=market_module.BASE_COSTS,
        initial_capital=market_module.INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    stress_ledger, stress_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=market_module.STRESS_COSTS,
        initial_capital=market_module.INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    base = summarize_path(base_ledger, base_trades, benchmark, objective=market_module.OBJECTIVE)
    stress = summarize_path(stress_ledger, stress_trades, benchmark, objective=market_module.OBJECTIVE)
    row: dict[str, Any] = {
        "policy": policy_name,
        "policy_type": spec.policy_type,
        "score_column": spec.score_column,
        "trigger": spec.trigger,
        "recovery": spec.recovery,
        "hold_days": spec.hold_days,
        "start_perturbation": start_perturbation,
        "cash_day_share": float(decisions["state"].eq(0).mean()),
        "cash_day_count": int(decisions["state"].eq(0).sum()),
        "execution_day_count": int(len(decisions)),
    }
    for prefix, summary in [("base", base), ("stress", stress)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
    decisions.insert(0, "policy", policy_name)
    decisions.insert(1, "start_perturbation", start_perturbation)
    return row, decisions


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in metrics.groupby("policy", sort=True):
        first = group.iloc[0]
        stress_annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        stress_rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        base_annual = pd.to_numeric(group["base_annualized_excess"], errors="coerce")
        base_rolling = pd.to_numeric(group["base_rolling_242d_excess_median"], errors="coerce")
        passed = bool(
            len(group) == len(v1.START_PERTURBATIONS)
            and group["stress_both_20pct_gates"].astype(bool).all()
        )
        rows.append(
            {
                "policy": policy,
                "policy_type": str(first["policy_type"]),
                "score_column": str(first["score_column"]),
                "trigger": float(first["trigger"]),
                "recovery": v1._safe_float(first["recovery"]),
                "hold_days": None if pd.isna(first["hold_days"]) else int(first["hold_days"]),
                "base_annualized_excess_minimum": v1._safe_float(base_annual.min()),
                "base_annualized_excess_median": v1._safe_float(base_annual.median()),
                "base_rolling_242d_excess_minimum": v1._safe_float(base_rolling.min()),
                "stress_annualized_excess_minimum": v1._safe_float(stress_annual.min()),
                "stress_annualized_excess_median": v1._safe_float(stress_annual.median()),
                "stress_annualized_excess_maximum": v1._safe_float(stress_annual.max()),
                "stress_rolling_242d_excess_minimum": v1._safe_float(stress_rolling.min()),
                "stress_rolling_242d_excess_median": v1._safe_float(stress_rolling.median()),
                "cash_day_share_median": float(group["cash_day_share"].median()),
                "stress_trade_leg_count_median": v1._safe_float(group["stress_trade_leg_count"].median()),
                "stress_perturbation_pass_ratio": float(group["stress_both_20pct_gates"].mean()),
                "development_gate": passed,
            }
        )
    rows.sort(
        key=lambda row: row["stress_annualized_excess_median"]
        if row["stress_annualized_excess_median"] is not None
        else -float("inf"),
        reverse=True,
    )
    return rows


def main() -> int:
    frame, dividends, frame_audit = _prepare_frame()
    policies = _make_policies()
    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for policy_name, spec in policies.items():
        for perturbation in v1.START_PERTURBATIONS:
            metrics, decisions = _evaluate(frame, dividends, policy_name, spec, perturbation)
            metric_rows.append(metrics)
            decision_frames.append(decisions)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    frontier = _aggregate(metrics)
    passed = [row for row in frontier if row["development_gate"]]
    status = (
        "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
        if passed
        else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
    )
    v1._atomic_parquet(metrics, OUTPUT_METRICS)
    v1._atomic_parquet(decisions, OUTPUT_DECISIONS)
    payload = {
        "status": status,
        "project_id": PROJECT_ID,
        "parent_branch": {
            "project_id": v1.PROJECT_ID,
            "status": "DEVELOPMENT_REJECTED",
            "report": str(v1.OUTPUT_REPORT.relative_to(PROJECT_ROOT)),
            "report_sha256": v1._sha256(v1.OUTPUT_REPORT),
        },
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "option_market_is_information_only": True,
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
        },
        "development_contract": {
            "source_data_ceiling": v1.DEVELOPMENT_CUTOFF.date().isoformat(),
            "selection_period": [v1.SELECTION_START.date().isoformat(), v1.SELECTION_END.date().isoformat()],
            "candidate_family": "固定2/3/5日延续或风险分位滞回解除",
            "candidate_count": len(policies),
            "start_perturbations": v1.START_PERTURBATIONS,
            "stress_costs": {
                "commission_rate_per_leg": market_module.STRESS_COSTS.commission_rate,
                "slippage_bps_per_leg": market_module.STRESS_COSTS.slippage_bps,
                "cash_annual_rate": market_module.STRESS_COSTS.cash_annual_rate,
                "lot_size": market_module.STRESS_COSTS.lot_size,
            },
            "all_five_starts_and_both_20pct_gates_required": True,
        },
        "passed_candidate_count": len(passed),
        "best_candidate": frontier[0],
        "passed_candidates": passed,
        "frontier": frontier,
        "frame_audit": frame_audit,
        "artifacts": {
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
            "decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)),
        },
        "boundaries": {
            "validation_or_holdout_rows_read": 0,
            "orders": "DISABLED",
            "broker": "DISABLED",
            "live_trading": "NOT_AUTHORIZED",
        },
    }
    v1._atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(policies),
                "passed_candidate_count": len(passed),
                "best_candidate": frontier[0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
