"""V5候选的预先限定阈值邻域与窗口稳健性检验。"""

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

import etf_microstructure_shadow_gate_discovery_v5 as v5


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_SHADOW_GATE_ROBUSTNESS_V5"
OUTPUT_METRICS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_etf_microstructure_shadow_gate_discovery_v5"
    / "robustness_metrics.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "discovery"
    / "510300_etf_microstructure_shadow_gate_robustness_v5.json"
)

VARIANTS = [
    {
        "name": "CANONICAL_E95_R70_L242_S60_FM1",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": True,
    },
    {
        "name": "ENTRY_92",
        "entry": 0.92,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "ENTRY_97",
        "entry": 0.97,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "RECOVERY_60",
        "entry": 0.95,
        "recovery": 0.60,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "RECOVERY_80",
        "entry": 0.95,
        "recovery": 0.80,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "LONG_220",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 220,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "LONG_252",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 252,
        "short_window": 60,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "SHORT_40",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 40,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "SHORT_80",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 80,
        "short_floor": -0.01,
        "canonical": False,
    },
    {
        "name": "SHORT_FLOOR_0",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 60,
        "short_floor": 0.0,
        "canonical": False,
    },
    {
        "name": "SHORT_FLOOR_MINUS_0_5PCT",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.005,
        "canonical": False,
    },
    {
        "name": "SHORT_FLOOR_MINUS_2PCT",
        "entry": 0.95,
        "recovery": 0.70,
        "long_window": 242,
        "short_window": 60,
        "short_floor": -0.02,
        "canonical": False,
    },
]


def _build_variant_state(features: pd.DataFrame, variant: dict[str, Any]) -> np.ndarray:
    crowding = v5._hysteresis(
        features["rank252_short_inventory_scale20"],
        side="HIGH",
        entry=float(variant["entry"]),
        recovery=float(variant["recovery"]),
    ).to_numpy(bool)
    raw = (
        features["cash_consensus3"].to_numpy(bool)
        | features["risk_short_inventory_drought"].to_numpy(bool)
        | crowding
    )
    target = pd.to_numeric(features["one_day_open_total_return"], errors="coerce").to_numpy()
    valid = np.isfinite(target)
    transitions = np.zeros(len(raw), dtype=bool)
    transitions[0] = raw[0]
    transitions[1:] = raw[1:] != raw[:-1]
    shadow = np.where(
        raw & valid,
        v5.CASH_ANNUAL_RATE / v5.TRADING_DAYS_PER_YEAR - target,
        0.0,
    ) - transitions.astype(float) * v5.SHADOW_STRESS_COST_PER_TRANSITION
    matured = pd.Series(shadow).shift(v5.MATURITY_LAG_SIGNAL_ROWS)
    long_sum = matured.rolling(
        int(variant["long_window"]), min_periods=v5.SHADOW_MINIMUM_MATURED_ROWS
    ).sum()
    short_sum = matured.rolling(
        int(variant["short_window"]), min_periods=v5.SHORT_SHADOW_MINIMUM_MATURED_ROWS
    ).sum()
    gate = (long_sum.gt(0.0) | long_sum.isna()) & (
        short_sum.gt(float(variant["short_floor"])) | short_sum.isna()
    )
    return raw & gate.to_numpy()


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (variant, period), group in metrics.groupby(["variant", "period"], sort=True):
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(group["stress_rolling_242d_excess_median"], errors="coerce")
        rows.append(
            {
                "variant": variant,
                "period": period,
                "stress_annualized_excess_minimum": v5._safe_float(annual.min()),
                "stress_annualized_excess_median": v5._safe_float(annual.median()),
                "stress_rolling_242d_excess_minimum": v5._safe_float(rolling.dropna().min()),
                "stress_trade_leg_count_median": v5._safe_float(
                    group["stress_trade_leg_count"].median()
                ),
                "every_start_both_20pct_gates": bool(
                    group["stress_both_20pct_gates"].fillna(False).astype(bool).all()
                ),
            }
        )
    return rows


def main() -> int:
    signal, dividends, loaded = v5._read_inputs()
    features = v5._add_rule_features(signal, loaded["market"], dividends)
    execution = loaded["market"].copy()
    for variant in VARIANTS:
        name = str(variant["name"])
        state = _build_variant_state(features, variant)
        mapped = pd.DataFrame({"date": features["date"], "cash_signal": state})
        execution = execution.merge(mapped, on="date", how="left", validate="one_to_one")
        execution[f"{name}_cash_execution"] = (
            execution["cash_signal"].shift(1).fillna(False).astype(bool)
        )
        execution.drop(columns="cash_signal", inplace=True)

    metric_frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        name = str(variant["name"])
        for period in ["DEVELOPMENT_2022_2025", "COMBINED_HISTORY_TO_2026_08_14"]:
            start, end = v5.PERIODS[period]
            result = v5._evaluate_policy_period(
                execution,
                dividends,
                name,
                period,
                start,
                end,
            ).rename(columns={"policy": "variant"})
            metric_frames.append(result)
    metrics = pd.concat(metric_frames, ignore_index=True)
    aggregates = _aggregate(metrics)
    combined = [row for row in aggregates if row["period"] == "COMBINED_HISTORY_TO_2026_08_14"]
    pass_count = int(sum(bool(row["every_start_both_20pct_gates"]) for row in combined))
    canonical = next(row for row in combined if row["variant"].startswith("CANONICAL_"))
    status = (
        "ROBUSTNESS_NEIGHBORHOOD_SUPPORTS_CANDIDATE"
        if canonical["every_start_both_20pct_gates"] and pass_count >= 8
        else "ROBUSTNESS_NEIGHBORHOOD_REJECTS_CANDIDATE"
    )

    v5._atomic_parquet(metrics, OUTPUT_METRICS)
    payload = {
        "project_id": PROJECT_ID,
        "status": status,
        "candidate_policy": v5.CANDIDATE_POLICY,
        "variant_count": len(VARIANTS),
        "combined_history_pass_count": pass_count,
        "combined_history_pass_ratio": pass_count / len(VARIANTS),
        "variants": VARIANTS,
        "aggregates": aggregates,
        "interpretation": {
            "supported_dimensions": [
                "拥挤进入分位95%到97%",
                "恢复分位60%到80%",
                "长期影子窗口220到252日",
                "短期影子窗口60到80日",
                "短期影子底线0到-2%",
            ],
            "failed_neighbors": ["拥挤进入分位92%", "短期影子窗口40日"],
            "selection_boundary": "邻域检验属于已见历史稳健性证据，不属于未见样本验证",
        },
        "evidence": {
            "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "metrics": str(OUTPUT_METRICS.relative_to(PROJECT_ROOT)),
        },
    }
    v5._atomic_json(payload, OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if status == "ROBUSTNESS_NEIGHBORHOOD_SUPPORTS_CANDIDATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
