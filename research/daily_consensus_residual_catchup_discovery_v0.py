"""在高精度两票共识之外补充全球冲击后的国内补跌风险。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from daily_tail_state_machine_discovery_v0 import (
    PolicySpec,
    START_PERTURBATIONS,
    _aggregate,
    _evaluate_policy_start,
)
from direct_one_day_cash_advantage_rank_discovery_v0 import _add_one_day_diagnostics
from global_liquidity_regime_5d_discovery_v0 import (
    DEVELOPMENT_CUTOFF,
    PROJECT_ROOT,
    SELECTION_START,
    _atomic_json,
    _atomic_parquet,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")
GLOBAL_TAIL_COVERAGES = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

DIRECT_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_direct_one_day_cash_advantage_rank_discovery_v0.parquet"
)
GLOBAL_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_global_contagion_state_machine_discovery_v0.parquet"
)
ALL_A_FILE = PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_consensus_residual_catchup_discovery_v0.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_consensus_residual_catchup_discovery_v0"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_consensus_residual_catchup_discovery_v0"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_daily_consensus_residual_catchup_discovery_v0.json"
)

DIRECT_COLUMNS = [
    "date",
    "target1_end_date",
    "one_day_open_total_return",
    "signed_cash_advantage1",
    "avoidable_loss1",
    "avoidable_loss5",
    "risk_rolling252_global_selloff1_baseline",
    "risk_rolling252_rf_signed_all",
    "risk_rolling252_hgb_signed_all",
    "risk_rolling252_gbr_q90_loss_all",
    "risk_rolling252_elastic_signed_all",
]
GLOBAL_COLUMNS = ["date", "risk_rolling252_vix_level", "risk_rolling252_contagion_joint"]
ALL_A_COLUMNS = ["date", "etf_return_1d", "etf_return_20d", "down_amount_share"]


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_frame() -> tuple[pd.DataFrame, float, dict[str, Any]]:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    direct = _normalize_dates(pd.read_parquet(DIRECT_FILE, columns=DIRECT_COLUMNS))
    global_scores = _normalize_dates(pd.read_parquet(GLOBAL_FILE, columns=GLOBAL_COLUMNS))
    all_a = _normalize_dates(pd.read_parquet(ALL_A_FILE, columns=ALL_A_COLUMNS))
    frame = market.merge(direct, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(global_scores, on="date", how="left", validate="one_to_one")
    frame = frame.merge(all_a, on="date", how="left", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)
    fit_mask = frame["date"].between(FIT_START, FIT_END)
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("补跌残差信号的拟合段或选择段覆盖不足")
    down_amount_fit_median = float(frame.loc[fit_mask, "down_amount_share"].median())
    audit = {
        "rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "selection_first_date": frame.loc[selection_mask, "date"].min().date().isoformat(),
        "selection_last_date": frame.loc[selection_mask, "date"].max().date().isoformat(),
    }
    return frame, down_amount_fit_median, audit


def _base_consensus(frame: pd.DataFrame) -> pd.Series:
    members = [
        frame["risk_rolling252_rf_signed_all"] >= 0.97,
        frame["risk_rolling252_hgb_signed_all"] >= 0.96,
        frame["risk_rolling252_gbr_q90_loss_all"] >= 0.96,
        frame["risk_rolling252_elastic_signed_all"] >= 0.99,
        frame["risk_rolling252_vix_level"] >= 0.95,
        frame["risk_rolling252_contagion_joint"] >= 0.95,
    ]
    return (pd.concat(members, axis=1).sum(axis=1) >= 2).fillna(False)


def _condition_masks(frame: pd.DataFrame, down_amount_fit_median: float) -> dict[str, pd.Series]:
    return {
        "TREND20_POS": frame["etf_return_20d"] >= 0.0,
        "ETF_GREEN": frame["etf_return_1d"] >= 0.0,
        "TREND20_POS_AND_GREEN": (frame["etf_return_20d"] >= 0.0) & (frame["etf_return_1d"] >= 0.0),
        "DOWN_AMOUNT_BELOW_FIT_MEDIAN": frame["down_amount_share"] < down_amount_fit_median,
        "TREND20_POS_OR_GREEN": (frame["etf_return_20d"] >= 0.0) | (frame["etf_return_1d"] >= 0.0),
    }


def _build_policies(
    frame: pd.DataFrame,
    down_amount_fit_median: float,
) -> tuple[dict[str, PolicySpec], dict[str, dict[str, Any]], list[str]]:
    base = _base_consensus(frame)
    conditions = _condition_masks(frame, down_amount_fit_median)
    policies: dict[str, PolicySpec] = {}
    metadata: dict[str, dict[str, Any]] = {}
    signal_columns: list[str] = []

    frame["signal_base_consensus"] = base.astype(float)
    signal_columns.append("signal_base_consensus")
    policies["CATCHUP_BASE_CONSENSUS"] = PolicySpec("PULSE", "signal_base_consensus", 0.5)
    metadata["CATCHUP_BASE_CONSENSUS"] = {
        "global_tail_coverage": None,
        "residual_condition": "NONE",
    }

    counter = 0
    for coverage in GLOBAL_TAIL_COVERAGES:
        global_tail = frame["risk_rolling252_global_selloff1_baseline"] >= 1.0 - coverage
        for condition_name, condition in conditions.items():
            counter += 1
            signal_column = f"signal_catchup_{counter:03d}"
            frame[signal_column] = (base | (global_tail & condition)).astype(float)
            signal_columns.append(signal_column)
            policy_name = f"CATCHUP_GLOBAL_TOP{int(coverage * 100):02d}_{condition_name}"
            policies[policy_name] = PolicySpec("PULSE", signal_column, 0.5)
            metadata[policy_name] = {
                "global_tail_coverage": coverage,
                "residual_condition": condition_name,
            }
    return policies, metadata, signal_columns


def main() -> int:
    required = [DIRECT_FILE, GLOBAL_FILE, ALL_A_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, down_amount_fit_median, frame_audit = _load_frame()
    policies, policy_metadata, signal_columns = _build_policies(frame, down_amount_fit_median)
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
            metrics.update(policy_metadata[policy_name])
            metrics, decisions = _add_one_day_diagnostics(metrics, decisions, frame)
            metric_rows.append(metrics)
            decision_frames.append(decisions)

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    aggregates = _aggregate(metrics)
    meta = metrics.groupby("policy", sort=False).first()[["global_tail_coverage", "residual_condition"]]
    one_day = metrics.groupby("policy", sort=False)[
        ["bad1_recall", "good1_false_exit", "avoidable_loss1_capture", "selected_signed_cash_advantage1_sum"]
    ].median()
    for aggregate in aggregates:
        aggregate.update(
            {
                "global_tail_coverage": _safe_float(meta.loc[aggregate["policy"], "global_tail_coverage"]),
                "residual_condition": str(meta.loc[aggregate["policy"], "residual_condition"]),
                "bad1_recall_median": _safe_float(one_day.loc[aggregate["policy"], "bad1_recall"]),
                "good1_false_exit_median": _safe_float(
                    one_day.loc[aggregate["policy"], "good1_false_exit"]
                ),
                "avoidable_loss1_capture_median": _safe_float(
                    one_day.loc[aggregate["policy"], "avoidable_loss1_capture"]
                ),
                "selected_signed_cash_advantage1_sum_median": _safe_float(
                    one_day.loc[aggregate["policy"], "selected_signed_cash_advantage1_sum"]
                ),
            }
        )
    best = aggregates[0]
    passed = bool(best["development_gate"])

    output_columns = [
        "date",
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "one_day_open_total_return",
        "signed_cash_advantage1",
        "avoidable_loss1",
        "avoidable_loss5",
        *signal_columns,
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(metrics, OUTPUT_METRICS)
    _atomic_parquet(decisions, OUTPUT_DECISIONS)
    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_DAILY_CONSENSUS_RESIDUAL_CATCHUP_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "decision_frequency": "每交易日",
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "base_rule": "六个独立风险信号至少两个触发",
            "global_tail_coverages": GLOBAL_TAIL_COVERAGES,
            "residual_conditions": list(_condition_masks(frame, down_amount_fit_median)),
            "down_amount_fit_median": down_amount_fit_median,
            "start_perturbations_trading_days": START_PERTURBATIONS,
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个起点扰动的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "frame_audit": frame_audit,
        "policy_count": int(len(policies)),
        "best_development_policy": best,
        "all_policy_aggregates": aggregates,
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path) for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_perturbation_metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "共识加补跌残差通过开发门槛；冻结共识、残差条件、输入哈希和成本后再读取独立验证。"
            if passed
            else "共识加补跌残差仍未达到20%开发门槛；保留条件前沿并继续搜索。"
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
