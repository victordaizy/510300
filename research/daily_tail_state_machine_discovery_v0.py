"""把五日风险排序转成逐日执行的二元仓位状态机。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from binary_state_feasibility_v1 import build_benchmark_ledger, simulate_binary_path, summarize_path
from global_liquidity_regime_5d_discovery_v0 import (
    BASE_COSTS,
    DEVELOPMENT_CUTOFF,
    INITIAL_CAPITAL_CNY,
    OBJECTIVE,
    PROJECT_ROOT,
    SELECTION_START,
    STRESS_COSTS,
    _atomic_json,
    _atomic_parquet,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)
from multi_domain_severity_rank_5d_discovery_v0 import _fit_ecdf, _rolling_last_percentile


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")
START_PERTURBATIONS = list(range(5))

DIRECT_SCORE_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_direct_avoidable_loss_rank_5d_discovery_v0.parquet"
)
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_daily_tail_state_machine_discovery_v0.parquet"
OUTPUT_OFFSETS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_tail_state_machine_discovery_v0"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_tail_state_machine_discovery_v0"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "discovery" / "510300_daily_tail_state_machine_discovery_v0.json"

SOURCE_SCORE_COLUMNS = [
    "risk_rolling252_gbr_q90_loss_all",
    "risk_rolling252_ridge_signed_all_a",
    "risk_fit_ecdf_hgb_severe20_all",
]


@dataclass(frozen=True)
class PolicySpec:
    policy_type: str
    score_column: str
    trigger: float
    recovery: float | None = None
    hold_days: int | None = None
    crossing_only: bool = False


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    scores = pd.read_parquet(
        DIRECT_SCORE_FILE,
        columns=["date", "target_end_date", "signed_cash_advantage5", "avoidable_loss5", *SOURCE_SCORE_COLUMNS],
    )
    scores = _normalize_dates(scores)
    frame = market.merge(scores, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)

    fit_mask = (
        frame["date"].between(FIT_START, FIT_END)
        & frame["target_end_date"].le(FIT_END)
        & frame["signed_cash_advantage5"].notna()
    )
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("逐日状态机的拟合段或选择段覆盖不足")

    frame["risk_pre_consensus_mean3"] = frame[SOURCE_SCORE_COLUMNS].mean(axis=1)
    frame["risk_pre_consensus_max3"] = frame[SOURCE_SCORE_COLUMNS].max(axis=1)
    frame["risk_fit_ecdf_consensus_mean3"] = _fit_ecdf(frame["risk_pre_consensus_mean3"], fit_mask)
    frame["risk_fit_ecdf_consensus_max3"] = _fit_ecdf(frame["risk_pre_consensus_max3"], fit_mask)
    frame["risk_rolling252_consensus_mean3"] = _rolling_last_percentile(
        frame["risk_pre_consensus_mean3"]
    )
    frame["risk_rolling252_consensus_max3"] = _rolling_last_percentile(
        frame["risk_pre_consensus_max3"]
    )
    score_columns = [
        *SOURCE_SCORE_COLUMNS,
        "risk_fit_ecdf_consensus_mean3",
        "risk_rolling252_consensus_mean3",
        "risk_rolling252_consensus_max3",
    ]
    audit = {
        "rows": int(len(frame)),
        "purged_fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "selection_missing_by_score": {
            column: int(frame.loc[selection_mask, column].isna().sum()) for column in score_columns
        },
    }
    return frame, score_columns, audit


def _make_policies(score_columns: list[str]) -> dict[str, PolicySpec]:
    policies: dict[str, PolicySpec] = {}
    for score_column in score_columns:
        score_name = score_column.replace("risk_rolling252_", "").replace("risk_fit_ecdf_", "")
        for trigger in [0.95, 0.90]:
            tail = int(round((1.0 - trigger) * 100))
            policies[f"DAILY_{score_name.upper()}_TAIL{tail:02d}_PULSE"] = PolicySpec(
                "PULSE", score_column, trigger
            )
            for hold_days in [3, 5, 10]:
                policies[
                    f"DAILY_{score_name.upper()}_TAIL{tail:02d}_HOLD{hold_days:02d}"
                ] = PolicySpec("FIXED_HOLD", score_column, trigger, hold_days=hold_days)
            policies[f"DAILY_{score_name.upper()}_TAIL{tail:02d}_CROSS_HOLD05"] = PolicySpec(
                "FIXED_HOLD", score_column, trigger, hold_days=5, crossing_only=True
            )
            for recovery in [0.70, 0.50]:
                policies[
                    f"DAILY_{score_name.upper()}_TAIL{tail:02d}_HYST_REC{int(recovery * 100):02d}"
                ] = PolicySpec("HYSTERESIS", score_column, trigger, recovery=recovery)
    return policies


def _states_for_range(
    frame: pd.DataFrame,
    first_execution_index: int,
    last_execution_index: int,
    spec: PolicySpec,
) -> tuple[np.ndarray, pd.DataFrame]:
    anchor_index = first_execution_index - 1
    sample_length = last_execution_index - anchor_index + 1
    states = np.zeros(sample_length, dtype=np.int8)
    previous_state = 1
    remaining_cash_days = 0
    previous_score = float("nan")
    rows: list[dict[str, Any]] = []

    for absolute_execution_index in range(first_execution_index, last_execution_index + 1):
        signal_index = absolute_execution_index - 1
        row = frame.iloc[signal_index]
        score = float(row[spec.score_column])
        finite = bool(np.isfinite(score))
        trigger_now = bool(finite and score >= spec.trigger)
        if spec.crossing_only:
            trigger_now = bool(
                trigger_now
                and (not np.isfinite(previous_score) or previous_score < spec.trigger)
            )

        if spec.policy_type == "PULSE":
            state = 0 if trigger_now else 1
        elif spec.policy_type == "FIXED_HOLD":
            if trigger_now:
                remaining_cash_days = int(spec.hold_days or 1)
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
                state = 1 if (finite and score < recovery) else 0
        else:
            raise ValueError(f"未知状态机类型：{spec.policy_type}")

        sample_index = absolute_execution_index - anchor_index
        states[sample_index] = state
        rows.append(
            {
                "signal_date": pd.Timestamp(row["date"]),
                "execution_date": pd.Timestamp(frame.loc[absolute_execution_index, "date"]),
                "state": state,
                "trigger_now": trigger_now,
                "risk_score": _safe_float(score),
                "future5_full_factor": _safe_float(row["future5_full_factor"]),
                "future5_cash_factor": _safe_float(row["future5_cash_factor"]),
                "avoidable_loss5": _safe_float(row["avoidable_loss5"]),
            }
        )
        previous_state = state
        previous_score = score

    return states, pd.DataFrame(rows)


def _evaluate_policy_start(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    policy_name: str,
    spec: PolicySpec,
    start_perturbation: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_indices = np.flatnonzero(
        frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF).to_numpy()
    )
    first_execution_index = int(selection_indices[start_perturbation])
    last_execution_index = int(selection_indices[-1])
    if first_execution_index <= 0:
        raise ValueError("逐日状态机缺少前一交易日信号锚点")
    anchor_index = first_execution_index - 1
    sample = frame.iloc[anchor_index : last_execution_index + 1][
        ["date", "etf_open", "etf_close", "benchmark_close"]
    ].reset_index(drop=True)
    states, decisions = _states_for_range(frame, first_execution_index, last_execution_index, spec)
    benchmark = build_benchmark_ledger(sample, INITIAL_CAPITAL_CNY)
    base_ledger, base_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=BASE_COSTS,
        initial_capital=INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    stress_ledger, stress_trades = simulate_binary_path(
        sample,
        dividends,
        states,
        costs=STRESS_COSTS,
        initial_capital=INITIAL_CAPITAL_CNY,
        reinvest_paid_dividends=True,
    )
    base_summary = summarize_path(base_ledger, base_trades, benchmark, objective=OBJECTIVE)
    stress_summary = summarize_path(stress_ledger, stress_trades, benchmark, objective=OBJECTIVE)

    cash = decisions["state"].eq(0)
    bad5 = pd.to_numeric(decisions["avoidable_loss5"], errors="coerce").fillna(0.0) > 0.0
    total_loss = float(pd.to_numeric(decisions["avoidable_loss5"], errors="coerce").fillna(0.0).sum())
    captured_loss = float(
        pd.to_numeric(decisions.loc[cash, "avoidable_loss5"], errors="coerce").fillna(0.0).sum()
    )
    row: dict[str, Any] = {
        "policy": policy_name,
        "policy_type": spec.policy_type,
        "score_column": spec.score_column,
        "trigger": spec.trigger,
        "recovery": spec.recovery,
        "hold_days": spec.hold_days,
        "crossing_only": spec.crossing_only,
        "start_perturbation": start_perturbation,
        "execution_day_count": int(len(decisions)),
        "cash_day_count": int(cash.sum()),
        "cash_day_share": float(cash.mean()),
        "state_change_count": int(decisions["state"].diff().fillna(0).ne(0).sum()),
        "overlapping_bad5_recall": _safe_float(cash[bad5].mean()),
        "overlapping_good5_false_exit": _safe_float(cash[~bad5].mean()),
        "overlapping_bad5_loss_capture": _safe_float(captured_loss / total_loss if total_loss > 0 else np.nan),
    }
    for prefix, summary in [("base", base_summary), ("stress", stress_summary)]:
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
        every_pass = bool(
            len(group) == len(START_PERTURBATIONS) and group["stress_both_20pct_gates"].all()
        )
        rows.append(
            {
                "policy": policy,
                "policy_type": str(first["policy_type"]),
                "score_column": str(first["score_column"]),
                "trigger": float(first["trigger"]),
                "recovery": _safe_float(first["recovery"]),
                "hold_days": None if pd.isna(first["hold_days"]) else int(first["hold_days"]),
                "crossing_only": bool(first["crossing_only"]),
                "perturbation_count": int(len(group)),
                "base_annualized_excess_minimum": _safe_float(base_annual.min()),
                "base_annualized_excess_median": _safe_float(base_annual.median()),
                "base_rolling_242d_excess_minimum": _safe_float(base_rolling.min()),
                "base_rolling_242d_excess_median": _safe_float(base_rolling.median()),
                "stress_annualized_excess_minimum": _safe_float(stress_annual.min()),
                "stress_annualized_excess_median": _safe_float(stress_annual.median()),
                "stress_annualized_excess_maximum": _safe_float(stress_annual.max()),
                "stress_rolling_242d_excess_minimum": _safe_float(stress_rolling.min()),
                "stress_rolling_242d_excess_median": _safe_float(stress_rolling.median()),
                "stress_every_perturbation_both_20pct_gates": every_pass,
                "stress_perturbation_pass_ratio": float(group["stress_both_20pct_gates"].mean()),
                "cash_day_share_median": float(group["cash_day_share"].median()),
                "overlapping_bad5_recall_median": _safe_float(group["overlapping_bad5_recall"].median()),
                "overlapping_good5_false_exit_median": _safe_float(
                    group["overlapping_good5_false_exit"].median()
                ),
                "overlapping_bad5_loss_capture_median": _safe_float(
                    group["overlapping_bad5_loss_capture"].median()
                ),
                "stress_trade_leg_count_median": _safe_float(group["stress_trade_leg_count"].median()),
                "development_gate": every_pass,
            }
        )
    rows.sort(
        key=lambda item: item["stress_annualized_excess_median"]
        if item["stress_annualized_excess_median"] is not None
        else -float("inf"),
        reverse=True,
    )
    return rows


def main() -> int:
    if not DIRECT_SCORE_FILE.exists():
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": [str(DIRECT_SCORE_FILE)]}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, score_columns, frame_audit = _load_frame()
    policies = _make_policies(score_columns)
    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for policy_name, spec in policies.items():
        for perturbation in START_PERTURBATIONS:
            metrics, decisions = _evaluate_policy_start(
                frame,
                dividends,
                policy_name,
                spec,
                perturbation,
            )
            metric_rows.append(metrics)
            decision_frames.append(decisions)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    aggregates = _aggregate(metrics)
    best = aggregates[0]
    passed = bool(best["development_gate"])

    output_columns = [
        "date",
        "target_end_date",
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "signed_cash_advantage5",
        "avoidable_loss5",
        *score_columns,
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(metrics, OUTPUT_OFFSETS)
    _atomic_parquet(decisions, OUTPUT_DECISIONS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_DAILY_TAIL_STATE_MACHINE_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "state_meanings": {"0": "CASH_CNY", "1": "FULL_510300"},
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "decision_frequency": "每交易日",
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "start_perturbations_trading_days": START_PERTURBATIONS,
            "policy_families": ["PULSE", "FIXED_HOLD", "HYSTERESIS"],
            "trigger_percentiles": [0.95, 0.90],
            "fixed_hold_days": [3, 5, 10],
            "hysteresis_recovery_percentiles": [0.70, 0.50],
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个起点扰动的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "frame_audit": frame_audit,
        "score_columns": score_columns,
        "policy_count": int(len(policies)),
        "best_development_policy": best,
        "all_policy_aggregates": aggregates,
        "all_start_perturbation_metrics": metrics.to_dict("records"),
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(DIRECT_SCORE_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(DIRECT_SCORE_FILE)
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_perturbation_metrics": str(OUTPUT_OFFSETS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "逐日状态机通过开发门槛；先冻结状态转移、输入哈希与成本，再读取2021年后的独立验证。"
            if passed
            else "逐日触发、固定持有与迟滞复位均未达到20%开发门槛；保留前沿并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "policy_count": len(policies),
                "best_development_policy": best,
                "validation_2021_plus_loaded": False,
                "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
