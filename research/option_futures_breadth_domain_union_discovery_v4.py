"""将重复的IF基差信号合并为一个风险域后的510300二元并集开发检验。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import global_liquidity_regime_5d_discovery_v0 as market_module
import option_futures_breadth_union_discovery_v3 as v3
import option_information_binary_discovery_v1 as option_v1
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_OPTION_FUTURES_BREADTH_DOMAIN_UNION_DISCOVERY_V4"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_option_futures_breadth_domain_union_discovery_v4.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_futures_breadth_domain_union_discovery_v4"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_option_futures_breadth_domain_union_discovery_v4"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "discovery"
    / "510300_option_futures_breadth_domain_union_discovery_v4.json"
)


def _prepare() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame, dividends, audit = v3._build_market_frame()
    frame["domain_breadth"] = frame["trigger_breadth_concentration_low20"].fillna(False)
    frame["domain_basis"] = (
        frame["trigger_basis_level_low10"].fillna(False)
        | frame["trigger_basis_change_low05"].fillna(False)
    )
    frame["domain_futures_oi"] = frame["trigger_futures_oi_change_low15"].fillna(False)
    frame["domain_option_put_call_volume"] = frame[
        "trigger_put_call_volume_low10"
    ].fillna(False)
    domain_columns = [
        "domain_breadth",
        "domain_basis",
        "domain_futures_oi",
        "domain_option_put_call_volume",
    ]
    frame["orthogonal_domain_vote_count"] = frame[domain_columns].sum(axis=1)
    audit["domain_count"] = len(domain_columns)
    audit["basis_signals_count_as_one_domain"] = True
    audit["v3_report_sha256"] = option_v1._sha256(v3.OUTPUT_REPORT)
    return frame, dividends, audit


def _state_vector(
    frame: pd.DataFrame,
    first_execution_index: int,
    last_execution_index: int,
    minimum_votes: int | None,
    include_option_state: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    anchor_index = first_execution_index - 1
    states = np.zeros(last_execution_index - anchor_index + 1, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for execution_index in range(first_execution_index, last_execution_index + 1):
        signal_index = execution_index - 1
        option_cash = bool(frame.loc[execution_index, "option_hysteresis_cash_state"])
        raw_votes = frame.loc[signal_index, "orthogonal_domain_vote_count"]
        votes = 0 if pd.isna(raw_votes) else int(raw_votes)
        domain_cash = bool(minimum_votes is not None and votes >= minimum_votes)
        state = 0 if ((include_option_state and option_cash) or domain_cash) else 1
        states[execution_index - anchor_index] = state
        rows.append(
            {
                "signal_date": pd.Timestamp(frame.loc[signal_index, "date"]),
                "execution_date": pd.Timestamp(frame.loc[execution_index, "date"]),
                "option_cash": option_cash,
                "orthogonal_domain_vote_count": votes,
                "domain_cash": domain_cash,
                "state": state,
            }
        )
    return states, pd.DataFrame(rows)


def _evaluate(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy: str,
    minimum_votes: int | None,
    include_option_state: bool,
    perturbation: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_indices = np.flatnonzero(
        frame["date"].between(v3.SELECTION_START, v3.SELECTION_END).to_numpy()
    )
    first_execution_index = int(selection_indices[perturbation])
    last_execution_index = int(selection_indices[-1])
    anchor_index = first_execution_index - 1
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states, decisions = _state_vector(
        frame,
        first_execution_index,
        last_execution_index,
        minimum_votes,
        include_option_state,
    )
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
        "policy": policy,
        "minimum_orthogonal_domain_votes": minimum_votes,
        "include_option_state": include_option_state,
        "start_perturbation": perturbation,
        "cash_day_count": int(decisions["state"].eq(0).sum()),
        "cash_day_share": float(decisions["state"].eq(0).mean()),
        "execution_day_count": int(len(decisions)),
    }
    for prefix, summary in [("base", base), ("stress", stress)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
    decisions.insert(0, "policy", policy)
    decisions.insert(1, "start_perturbation", perturbation)
    return row, decisions


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, group in metrics.groupby("policy", sort=True):
        first = group.iloc[0]
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        base_annual = pd.to_numeric(group["base_annualized_excess"], errors="coerce")
        base_rolling = pd.to_numeric(group["base_rolling_242d_excess_median"], errors="coerce")
        passed = bool(
            len(group) == len(v3.START_PERTURBATIONS)
            and group["stress_both_20pct_gates"].astype(bool).all()
        )
        rows.append(
            {
                "policy": policy,
                "minimum_orthogonal_domain_votes": None
                if pd.isna(first["minimum_orthogonal_domain_votes"])
                else int(first["minimum_orthogonal_domain_votes"]),
                "include_option_state": bool(first["include_option_state"]),
                "base_annualized_excess_minimum": option_v1._safe_float(base_annual.min()),
                "base_annualized_excess_median": option_v1._safe_float(base_annual.median()),
                "base_rolling_242d_excess_minimum": option_v1._safe_float(base_rolling.min()),
                "stress_annualized_excess_minimum": option_v1._safe_float(annual.min()),
                "stress_annualized_excess_median": option_v1._safe_float(annual.median()),
                "stress_annualized_excess_maximum": option_v1._safe_float(annual.max()),
                "stress_rolling_242d_excess_minimum": option_v1._safe_float(rolling.min()),
                "stress_rolling_242d_excess_median": option_v1._safe_float(rolling.median()),
                "cash_day_share_median": float(group["cash_day_share"].median()),
                "stress_trade_leg_count_median": option_v1._safe_float(
                    group["stress_trade_leg_count"].median()
                ),
                "stress_perturbation_pass_ratio": float(
                    group["stress_both_20pct_gates"].mean()
                ),
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
    frame, dividends, audit = _prepare()
    policies: dict[str, tuple[int | None, bool]] = {
        "OPTION_HYSTERESIS_ONLY": (None, True),
        "FOUR_DOMAIN_VOTE2_ONLY": (2, False),
        "OPTION_OR_FOUR_DOMAIN_VOTE1": (1, True),
        "OPTION_OR_FOUR_DOMAIN_VOTE2": (2, True),
        "OPTION_OR_FOUR_DOMAIN_VOTE3": (3, True),
    }
    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for policy, (minimum_votes, include_option) in policies.items():
        for perturbation in v3.START_PERTURBATIONS:
            metrics, decisions = _evaluate(
                frame,
                dividends,
                policy,
                minimum_votes,
                include_option,
                perturbation,
            )
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

    feature_columns = [
        "date",
        "option_hysteresis_cash_state",
        "domain_breadth",
        "domain_basis",
        "domain_futures_oi",
        "domain_option_put_call_volume",
        "orthogonal_domain_vote_count",
    ]
    option_v1._atomic_parquet(frame[feature_columns], OUTPUT_FEATURES)
    option_v1._atomic_parquet(metrics, OUTPUT_METRICS)
    option_v1._atomic_parquet(decisions, OUTPUT_DECISIONS)
    payload = {
        "status": status,
        "project_id": PROJECT_ID,
        "parent_v3": {
            "status": "DEVELOPMENT_REJECTED_3_OF_5_STARTS_PASSED",
            "report": str(v3.OUTPUT_REPORT.relative_to(PROJECT_ROOT)),
            "report_sha256": option_v1._sha256(v3.OUTPUT_REPORT),
        },
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "information_only_assets": ["510300期权", "IF0股指期货", "全A横截面"],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
        },
        "development_contract": {
            "data_ceiling": v3.DEVELOPMENT_CUTOFF.date().isoformat(),
            "selection_period": [v3.SELECTION_START.date().isoformat(), v3.SELECTION_END.date().isoformat()],
            "start_perturbations": v3.START_PERTURBATIONS,
            "candidate_count": len(policies),
            "primary_rule": "期权Gamma滞回制度为空仓，或四个正交风险域至少两票触发",
            "domain_definitions": {
                "breadth": "全A头部成交集中度滚动低20%",
                "basis": "IF基差水平滚动低10%或五日变化滚动低5%，合并只计一票",
                "futures_open_interest": "IF五日持仓变化滚动低15%",
                "option_put_call_volume": "期权认沽认购成交量比滚动低10%",
            },
            "stress_costs": {
                "commission_rate_per_leg": market_module.STRESS_COSTS.commission_rate,
                "slippage_bps_per_leg": market_module.STRESS_COSTS.slippage_bps,
                "cash_annual_rate": market_module.STRESS_COSTS.cash_annual_rate,
                "lot_size": market_module.STRESS_COSTS.lot_size,
            },
            "gate": "五个起点在压力成本下年化净超额和滚动242日超额中位数均不低于20%",
        },
        "passed_candidate_count": len(passed),
        "best_candidate": frontier[0],
        "passed_candidates": passed,
        "frontier": frontier,
        "frame_audit": audit,
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)),
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
            "decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)),
        },
        "evidence_limit": "历史研究输入已被多轮查看；开发通过只允许冻结并进入前向Shadow，不能直接授权实盘。",
        "boundaries": {
            "post_2023_rows_read": 0,
            "position_mapping": "DISABLED",
            "orders": "DISABLED",
            "broker": "DISABLED",
            "live_trading": "NOT_AUTHORIZED",
        },
    }
    option_v1._atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(policies),
                "passed_candidate_count": len(passed),
                "best_candidate": frontier[0],
                "passed_candidates": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
