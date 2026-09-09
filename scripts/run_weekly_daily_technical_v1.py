"""运行510300周线状态—日线执行V1冻结历史检验。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.weekly_daily_technical_v1 import (
    build_features,
    h00300_metrics,
    load_config,
    load_inputs,
    performance_metrics,
    period_return,
    simulate_static_exposure,
    simulate_variant,
)


MANIFEST = ROOT / "config" / "weekly_daily_technical_v1_freeze_manifest.json"
DATA_GATE = ROOT / "reports" / "data_quality" / "weekly_daily_technical_v1_data_gate.json"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "weekly_daily_technical_v1"
REPORT_JSON = ROOT / "reports" / "backtest" / "weekly_daily_technical_v1.json"
REPORT_MARKDOWN = ROOT / "reports" / "backtest" / "weekly_daily_technical_v1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MANIFEST.exists() or not DATA_GATE.exists():
        raise RuntimeError("NO_VIEW：V1冻结清单或数据闸门缺失")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data_gate = json.loads(DATA_GATE.read_text(encoding="utf-8"))
    if manifest.get("return_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：冻结清单未授权收益计算")
    if data_gate.get("status") != "PASS" or data_gate.get("return_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：数据闸门未通过")
    for section in ("protocol_files", "implementation_files", "input_files"):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：冻结文件指纹变化：{relative_path}")
    return manifest, data_gate


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _markdown(report: dict[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    def num(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    lines = [
        "# 510300周线状态—日线执行V1冻结历史检验",
        "",
        f"- 数据截止：{report['data_cutoff']}",
        f"- 证据标签：{report['evidence_label']}",
        f"- 当前视图：{report['current_view']}",
        "- 安全边界：仅历史研究；真实仓位映射、订单生成和券商连接全部关闭。",
        "",
        "| 轨道 | 期末权益 | 总收益 | CAGR | 最大回撤 | Sharpe | 平均仓位 | 闭合周期 | 成本拖累 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id, evidence in report["variants"].items():
        base = evidence["base"]
        lines.append(
            f"| {variant_id} | {base['ending_equity_cny']:.2f} | {pct(base['total_return'])} | "
            f"{pct(base['cagr'])} | {pct(base['maximum_drawdown'])} | {num(base['sharpe'])} | "
            f"{pct(base['average_exposure'])} | {evidence['closed_cycles']} | "
            f"{evidence['cost_drag_cny']:.2f}元 |"
        )
    lines.extend(
        [
            "",
            "## 基准",
            "",
            f"- 510300含分红买入持有：总收益{pct(report['buy_hold']['total_return'])}，CAGR {pct(report['buy_hold']['cagr'])}，最大回撤{pct(report['buy_hold']['maximum_drawdown'])}，Sharpe {num(report['buy_hold']['sharpe'])}。",
            f"- H00300全收益指数：总收益{pct(report['h00300']['total_return'])}，CAGR {pct(report['h00300']['cagr'])}，Sharpe {num(report['h00300']['sharpe'])}。",
            "",
            "## 纪律",
            "",
            "该历史样本参与了策略设计，只能用于发现和否证。V1参数不得依据本报告结果调整；若失败，处理为`REJECT_WITHOUT_PARAMETER_RESCUE`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifest, data_gate = _verify_frozen_inputs()
    config = load_config()
    market, benchmark, dividends = load_inputs(ROOT, config)
    features, weekly = build_features(market, dividends, config)
    cash_rate = float(config["price_and_execution"]["cash_annual_rate"])
    initial = float(config["price_and_execution"]["initial_capital_cny"])

    results: dict[str, dict[float, dict[str, Any]]] = {}
    for variant in config["variants"]:
        variant_id = variant["id"]
        results[variant_id] = {}
        for multiplier in config["evaluation"]["cost_multipliers"]:
            results[variant_id][float(multiplier)] = simulate_variant(
                features,
                dividends,
                config,
                variant_id,
                cost_multiplier=float(multiplier),
            )

    buy_hold_ledger = simulate_static_exposure(features, dividends, config, 1.0)
    cash_ledger = simulate_static_exposure(features, dividends, config, 0.0)
    buy_hold = performance_metrics(buy_hold_ledger, cash_rate)
    cash_metrics = performance_metrics(cash_ledger, cash_rate)
    h00300 = h00300_metrics(
        benchmark,
        pd.Timestamp(features["date"].iloc[0]),
        pd.Timestamp(features["date"].iloc[-1]),
        initial,
    )

    report: dict[str, Any] = {
        "project_id": config["protocol"]["project_id"],
        "data_cutoff": str(features["date"].iloc[-1].date()),
        "evidence_label": config["protocol"]["evidence_label"],
        "true_forward_start": config["protocol"]["true_forward_start"],
        "current_view": "NO_VIEW_STALE_DATA_CUTOFF_2026-08-14",
        "buy_hold": buy_hold,
        "h00300": h00300,
        "cash": cash_metrics,
        "weekly_state_counts": weekly["state"].value_counts().to_dict(),
        "variants": {},
        "governance": config["governance"],
        "freeze_manifest_sha256": sha256(MANIFEST),
        "data_gate_sha256": sha256(DATA_GATE),
        "data_gate_status": data_gate["status"],
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_DIRECTORY / "daily_features.parquet", index=False)
    weekly.to_csv(OUTPUT_DIRECTORY / "weekly_state_audit.csv", index=False)
    buy_hold_ledger.to_parquet(OUTPUT_DIRECTORY / "buy_hold_ledger.parquet", index=False)

    all_executions: list[pd.DataFrame] = []
    all_cycles: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    for variant in config["variants"]:
        variant_id = variant["id"]
        base = results[variant_id][1.0]
        gross = results[variant_id][0.0]
        double = results[variant_id][2.0]
        base_metrics = performance_metrics(base["ledger"], cash_rate)
        gross_metrics = performance_metrics(gross["ledger"], cash_rate)
        double_metrics = performance_metrics(double["ledger"], cash_rate)
        matched_ledger = simulate_static_exposure(
            features, dividends, config, float(base_metrics["average_exposure"])
        )
        matched_metrics = performance_metrics(matched_ledger, cash_rate)
        periods = []
        for period in config["evaluation"]["predefined_periods"]:
            periods.append(
                {
                    **period,
                    "strategy_return": period_return(base["ledger"], period["start"], period["end"]),
                    "buy_hold_return": period_return(buy_hold_ledger, period["start"], period["end"]),
                }
            )
        cycles = base["cycles"]
        executions = base["executions"]
        skipped = base["skipped_orders"]
        evidence = {
            "base": base_metrics,
            "zero_cost": gross_metrics,
            "double_cost": double_metrics,
            "matched_exposure": matched_metrics,
            "cost_drag_cny": float(gross_metrics["ending_equity_cny"] - base_metrics["ending_equity_cny"]),
            "timing_contribution_vs_matched_total_return": float(base_metrics["total_return"] - matched_metrics["total_return"]),
            "closed_cycles": int(len(cycles)),
            "execution_legs": int(len(executions)),
            "minimum_notional_rejections": int(len(skipped)),
            "predefined_periods": periods,
            "ending_state_suppressed": base["state"],
            "decision": "DESCRIPTIVE_HISTORICAL_ONLY",
        }
        report["variants"][variant_id] = evidence
        prefix = variant_id.lower()
        base["ledger"].to_parquet(OUTPUT_DIRECTORY / f"{prefix}_daily_ledger.parquet", index=False)
        base["executions"].to_csv(OUTPUT_DIRECTORY / f"{prefix}_executions.csv", index=False)
        base["skipped_orders"].to_csv(OUTPUT_DIRECTORY / f"{prefix}_skipped_orders.csv", index=False)
        base["transitions"].to_csv(OUTPUT_DIRECTORY / f"{prefix}_transitions.csv", index=False)
        base["cycles"].to_csv(OUTPUT_DIRECTORY / f"{prefix}_closed_cycles.csv", index=False)
        matched_ledger.to_parquet(OUTPUT_DIRECTORY / f"{prefix}_matched_exposure_ledger.parquet", index=False)
        if not executions.empty:
            all_executions.append(executions)
        if not cycles.empty:
            all_cycles.append(cycles)
        if not skipped.empty:
            all_skipped.append(skipped)

    if all_executions:
        pd.concat(all_executions, ignore_index=True).to_csv(
            OUTPUT_DIRECTORY / "all_execution_events.csv", index=False
        )
    if all_cycles:
        pd.concat(all_cycles, ignore_index=True).to_csv(
            OUTPUT_DIRECTORY / "all_closed_cycles.csv", index=False
        )
    if all_skipped:
        pd.concat(all_skipped, ignore_index=True).to_csv(
            OUTPUT_DIRECTORY / "all_skipped_orders.csv", index=False
        )

    safe_report = _safe(report)
    REPORT_JSON.write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MARKDOWN.write_text(_markdown(safe_report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "data_cutoff": safe_report["data_cutoff"],
                "current_view": safe_report["current_view"],
                "variants": {
                    key: {
                        "ending_equity_cny": value["base"]["ending_equity_cny"],
                        "cagr": value["base"]["cagr"],
                        "maximum_drawdown": value["base"]["maximum_drawdown"],
                        "sharpe": value["base"]["sharpe"],
                        "closed_cycles": value["closed_cycles"],
                    }
                    for key, value in safe_report["variants"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

