"""V1.0.1机械修正：复制只读NumPy视图后再设置现金锚点。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import frozen_microstructure_sharpe_audit_v1 as base


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_CONFIG = (
    ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_1_correction.json"
)
CORRECTION_MANIFEST = (
    ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_1_manifest.json"
)


def validate_correction_manifest() -> dict[str, Any]:
    if not CORRECTION_MANIFEST.exists():
        raise FileNotFoundError("V1.0.1机械修正清单不存在")
    manifest = json.loads(CORRECTION_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("implementation_frozen", False):
        raise ValueError("V1.0.1机械修正尚未冻结")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in manifest["tracked_files"].items():
        path = ROOT / relative
        actual = "MISSING" if not path.exists() else base.sha256_file(path)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"V1.0.1机械修正哈希漂移：{json.dumps(mismatches, ensure_ascii=False)}")
    return manifest


def _evaluate_period_corrected(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    period_name: str,
    period: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    """与V1逐行等价；唯一机械改动是to_numpy(copy=True)。"""

    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    selection = np.flatnonzero(market["date"].between(start, end).to_numpy())
    offsets = config["execution"]["start_perturbations_trading_days"]
    if len(selection) <= max(offsets):
        raise ValueError(f"{period_name}不足以执行五个起点")
    last_index = int(selection[-1])
    initial_capital = float(config["execution"]["initial_capital_cny"])
    lot_size = int(config["execution"]["lot_size_shares"])
    trading_days = int(config["execution"]["annualization_trading_days"])
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for offset in offsets:
        first_index = int(selection[offset])
        anchor_index = first_index - 1
        if anchor_index < 0:
            raise ValueError(f"{period_name}缺少首个执行日前的现金锚点")
        sample = market.iloc[anchor_index : last_index + 1].reset_index(drop=True).copy()
        states = pd.to_numeric(sample["execution_state"], errors="coerce")
        if states.iloc[1:].isna().any():
            raise ValueError(f"{period_name}起点{offset}存在冻结执行状态缺失")
        desired = states.fillna(0).astype(np.int8).to_numpy(copy=True)
        desired[0] = 0
        buy_hold_desired = np.ones(len(sample), dtype=np.int8)
        buy_hold_desired[0] = 0
        buy_hold, buy_hold_trades = base.simulate(
            sample,
            dividends,
            buy_hold_desired,
            costs=config["costs"]["base"],
            initial_capital=initial_capital,
            lot_size=lot_size,
        )
        scenario_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        for scenario in ["base", "double"]:
            ledger, trades = base.simulate(
                sample,
                dividends,
                desired,
                costs=config["costs"][scenario],
                initial_capital=initial_capital,
                lot_size=lot_size,
            )
            summary = base._path_summary(
                ledger,
                trades,
                buy_hold,
                trading_days=trading_days,
            )
            row: dict[str, Any] = {
                "period": period_name,
                "gate_eligible": bool(period["gate_eligible"]),
                "start_perturbation": int(offset),
                "scenario": scenario,
                "first_execution_date": str(sample["date"].iloc[1].date()),
                "last_execution_date": str(sample["date"].iloc[-1].date()),
                **summary,
            }
            rows.append(row)
            scenario_artifacts[scenario] = (ledger, trades, buy_hold)
        artifacts[f"{period_name}:{offset}"] = scenario_artifacts["base"]
        if buy_hold_trades.empty:
            raise ValueError("买入持有基准未产生初始买入，数据或成本合同异常")
    return rows, artifacts


def main() -> int:
    correction_manifest = validate_correction_manifest()
    original_write_outputs = base.write_outputs

    def write_outputs_with_correction(
        config: dict[str, Any],
        report: dict[str, Any],
        metrics: pd.DataFrame,
        artifacts: dict[str, pd.DataFrame],
    ) -> None:
        report["correction_v1_0_1"] = {
            "project_id": correction_manifest["project_id"],
            "status": "MECHANICAL_READ_ONLY_ARRAY_COPY_CORRECTION",
            "manifest_sha256": base.sha256_file(CORRECTION_MANIFEST),
            "only_change": "to_numpy(copy=True)",
            "formula_or_parameter_changes": 0,
            "metrics_computed_before_correction_freeze": False,
        }
        original_write_outputs(config, report, metrics, artifacts)

    base._evaluate_period = _evaluate_period_corrected
    base.write_outputs = write_outputs_with_correction
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
