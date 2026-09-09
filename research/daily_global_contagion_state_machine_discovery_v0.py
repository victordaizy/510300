"""以前一夜全球市场与流动性冲击驱动510300逐日二元状态。"""

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
from global_liquidity_regime_5d_discovery_v0 import (
    DEVELOPMENT_CUTOFF,
    PROJECT_ROOT,
    SELECTION_START,
    _atomic_json,
    _atomic_parquet,
    _load_dividends,
    _load_market,
    _sha256,
)
from multi_domain_severity_rank_5d_discovery_v0 import _fit_ecdf, _rolling_last_percentile


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")

GLOBAL_FILE = PROJECT_ROOT / "data" / "features" / "510300_global_liquidity_regime_5d_discovery_v0.parquet"
OUTPUT_FEATURES = (
    PROJECT_ROOT / "data" / "features" / "510300_daily_global_contagion_state_machine_discovery_v0.parquet"
)
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_global_contagion_state_machine_discovery_v0"
    / "start_perturbation_metrics.parquet"
)
OUTPUT_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_daily_global_contagion_state_machine_discovery_v0"
    / "daily_decisions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_daily_global_contagion_state_machine_discovery_v0.json"
)

RAW_COLUMNS = [
    "vix_log_level",
    "vix_ret1",
    "vix_ret5",
    "us_mean_ret1",
    "us_mean_ret5",
    "eu_mean_ret1",
    "eu_mean_ret5",
    "asia_mean_ret1",
    "asia_mean_ret5",
    "global_min_ret1",
    "fx_basket_weak1",
    "fx_basket_weak5",
    "credit_spread_d1",
    "credit_spread_d5",
    "score_mechanism_risk",
    "vix_observation_date",
    "spx_observation_date",
    "dax_observation_date",
    "nikkei_observation_date",
]


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _load_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    dividends = _load_dividends()
    market = _normalize_dates(_load_market(dividends))
    global_features = pd.read_parquet(GLOBAL_FILE, columns=["date", *RAW_COLUMNS])
    global_features = _normalize_dates(global_features)
    frame = market.merge(global_features, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)
    frame["target_end_date"] = frame["date"].shift(-5)
    frame["signed_cash_advantage5"] = frame["future5_cash_factor"] - frame["future5_full_factor"]
    frame["avoidable_loss5"] = frame["signed_cash_advantage5"].clip(lower=0.0)
    frame["vix_level"] = np.exp(pd.to_numeric(frame["vix_log_level"], errors="coerce"))
    frame["raw_vix_shock"] = np.maximum(
        pd.to_numeric(frame["vix_ret1"], errors="coerce"),
        pd.to_numeric(frame["vix_ret5"], errors="coerce") / np.sqrt(5.0),
    )
    frame["raw_global_selloff1"] = -pd.concat(
        [
            frame["us_mean_ret1"],
            frame["eu_mean_ret1"],
            frame["asia_mean_ret1"],
            frame["global_min_ret1"],
        ],
        axis=1,
    ).min(axis=1)
    frame["raw_global_selloff5"] = -pd.concat(
        [frame["us_mean_ret5"], frame["eu_mean_ret5"], frame["asia_mean_ret5"]],
        axis=1,
    ).mean(axis=1)

    fit_mask = (
        frame["date"].between(FIT_START, FIT_END)
        & frame["target_end_date"].le(FIT_END)
        & frame["signed_cash_advantage5"].notna()
    )
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("全球传染状态机的拟合段或选择段覆盖不足")

    raw_risks = {
        "global_mechanism": "score_mechanism_risk",
        "vix_level": "vix_level",
        "vix_shock": "raw_vix_shock",
        "global_selloff1": "raw_global_selloff1",
        "global_selloff5": "raw_global_selloff5",
        "fx_weak5": "fx_basket_weak5",
        "credit_widen5": "credit_spread_d5",
    }
    component_percentiles: list[str] = []
    for name, raw_column in raw_risks.items():
        column = f"risk_rolling252_{name}"
        frame[column] = _rolling_last_percentile(frame[raw_column])
        component_percentiles.append(column)

    frame["risk_pre_contagion_mean5"] = frame[
        [
            "risk_rolling252_vix_level",
            "risk_rolling252_vix_shock",
            "risk_rolling252_global_selloff1",
            "risk_rolling252_global_selloff5",
            "risk_rolling252_fx_weak5",
        ]
    ].mean(axis=1)
    frame["risk_pre_contagion_joint"] = np.minimum(
        frame[["risk_rolling252_vix_level", "risk_rolling252_vix_shock"]].max(axis=1),
        frame[["risk_rolling252_global_selloff1", "risk_rolling252_global_selloff5"]].max(axis=1),
    )
    frame["risk_rolling252_contagion_mean5"] = _rolling_last_percentile(
        frame["risk_pre_contagion_mean5"]
    )
    frame["risk_rolling252_contagion_joint"] = _rolling_last_percentile(
        frame["risk_pre_contagion_joint"]
    )
    percentile_scores = [
        "risk_rolling252_global_mechanism",
        "risk_rolling252_vix_level",
        "risk_rolling252_vix_shock",
        "risk_rolling252_global_selloff1",
        "risk_rolling252_contagion_mean5",
        "risk_rolling252_contagion_joint",
    ]

    observation_age: dict[str, dict[str, float | None]] = {}
    for column in ["vix_observation_date", "spx_observation_date", "dax_observation_date", "nikkei_observation_date"]:
        observation = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        age = (frame["date"] - observation).dt.days
        selected = age.loc[selection_mask].dropna()
        observation_age[column] = {
            "minimum_calendar_days": float(selected.min()) if not selected.empty else None,
            "median_calendar_days": float(selected.median()) if not selected.empty else None,
            "maximum_calendar_days": float(selected.max()) if not selected.empty else None,
        }
    audit = {
        "rows": int(len(frame)),
        "purged_fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "selection_observation_age": observation_age,
        "selection_missing_by_score": {
            column: int(frame.loc[selection_mask, column].isna().sum()) for column in percentile_scores
        },
    }
    return frame, percentile_scores, audit


def _add_percentile_policies(
    policies: dict[str, PolicySpec],
    score_column: str,
) -> None:
    score_name = score_column.replace("risk_rolling252_", "")
    for trigger in [0.95, 0.90]:
        tail = int(round((1.0 - trigger) * 100))
        prefix = f"GLOBAL_{score_name.upper()}_TAIL{tail:02d}"
        policies[f"{prefix}_PULSE"] = PolicySpec("PULSE", score_column, trigger)
        for hold_days in [3, 5, 10]:
            policies[f"{prefix}_HOLD{hold_days:02d}"] = PolicySpec(
                "FIXED_HOLD", score_column, trigger, hold_days=hold_days
            )
        for recovery in [0.70, 0.50]:
            policies[f"{prefix}_HYST_REC{int(recovery * 100):02d}"] = PolicySpec(
                "HYSTERESIS", score_column, trigger, recovery=recovery
            )


def _make_policies(percentile_scores: list[str]) -> dict[str, PolicySpec]:
    policies: dict[str, PolicySpec] = {}
    for score_column in percentile_scores:
        _add_percentile_policies(policies, score_column)

    for level in [25.0, 30.0, 40.0]:
        prefix = f"GLOBAL_VIX_LEVEL_{int(level):02d}"
        policies[f"{prefix}_PULSE"] = PolicySpec("PULSE", "vix_level", level)
        for hold_days in [3, 5, 10]:
            policies[f"{prefix}_HOLD{hold_days:02d}"] = PolicySpec(
                "FIXED_HOLD", "vix_level", level, hold_days=hold_days
            )
        policies[f"{prefix}_HYST_REC20"] = PolicySpec(
            "HYSTERESIS", "vix_level", level, recovery=20.0
        )

    for selloff in [0.02, 0.03]:
        prefix = f"GLOBAL_SELLOFF1_{int(selloff * 100):02d}PCT"
        policies[f"{prefix}_PULSE"] = PolicySpec("PULSE", "raw_global_selloff1", selloff)
        for hold_days in [3, 5]:
            policies[f"{prefix}_HOLD{hold_days:02d}"] = PolicySpec(
                "FIXED_HOLD", "raw_global_selloff1", selloff, hold_days=hold_days
            )
    return policies


def main() -> int:
    if not GLOBAL_FILE.exists():
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": [str(GLOBAL_FILE)]}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, percentile_scores, frame_audit = _load_frame()
    policies = _make_policies(percentile_scores)
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
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "signed_cash_advantage5",
        "avoidable_loss5",
        "vix_level",
        "raw_vix_shock",
        "raw_global_selloff1",
        "raw_global_selloff5",
        "fx_basket_weak5",
        "credit_spread_d5",
        *percentile_scores,
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
        "project_id": "510300_DAILY_GLOBAL_CONTAGION_STATE_MACHINE_DISCOVERY_V0",
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
            "start_perturbations_trading_days": START_PERTURBATIONS,
            "percentile_triggers": [0.95, 0.90],
            "fixed_vix_levels": [25.0, 30.0, 40.0],
            "fixed_global_selloff_thresholds": [0.02, 0.03],
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个起点扰动的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "frame_audit": frame_audit,
        "percentile_scores": percentile_scores,
        "policy_count": int(len(policies)),
        "best_development_policy": best,
        "all_policy_aggregates": aggregates,
        "all_start_perturbation_metrics": metrics.to_dict("records"),
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(GLOBAL_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(GLOBAL_FILE)
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_perturbation_metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_decisions": str(OUTPUT_DECISIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "全球传染逐日状态机通过开发门槛；冻结时区对齐、状态转移与输入哈希后再读取独立验证。"
            if passed
            else "全球传染逐日状态机未达到20%开发门槛；保留前沿并继续搜索。"
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
