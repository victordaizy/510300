"""运行极端两级类马丁与旧V2轨道对照。"""

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

from research.extreme_two_level_martin_v1 import (
    VARIANTS,
    load_config,
    simulate_variant,
)
from research.graph_regime_martin_turtle_v2 import (
    build_features,
    load_config as load_source_config,
    load_inputs,
    performance_metrics,
)


MANIFEST = ROOT / "config" / "extreme_two_level_martin_v1_implementation_manifest.json"
SOURCE_REPORT = ROOT / "reports" / "backtest" / "graph_regime_martin_turtle_v2.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "extreme_two_level_martin_v1"
REPORT_JSON = ROOT / "reports" / "backtest" / "extreme_two_level_martin_v1.json"
REPORT_MD = ROOT / "reports" / "backtest" / "extreme_two_level_martin_v1.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：类马丁实现尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("return_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：冻结清单未授权收益计算")
    for section in ("implementation_files", "input_files"):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：冻结指纹变化：{relative_path}")
    return manifest


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


def _fmt(value: float | None, spec: str) -> str:
    return "N/A" if value is None else format(value, spec)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 510300极端两级类马丁V1历史对照",
        "",
        "- 第一层50%，再下跌2个固定入场ATR后第二层100%。",
        "- 下跌3ATR硬失效；确认下跌趋势时不补仓而退出。",
        "- 证据标签：`POST_V2_HISTORICALLY_CONTAMINATED_CANDIDATE`。",
        "",
        "| 轨道 | 类马丁闭合周期 | 第二层满仓 | CAGR | 最大回撤 | Sharpe | 显式成本 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id, item in report["variants"].items():
        metrics = item["metrics"]
        lines.append(
            "| {variant} | {cycles} | {full} | {cagr} | {drawdown} | {sharpe} | {cost}元 | {decision} |".format(
                variant=variant_id,
                cycles=item["module_closed_cycles"],
                full=item["second_layer_full_position_count"],
                cagr=_fmt(metrics["cagr"], ".2%"),
                drawdown=_fmt(metrics["maximum_drawdown"], ".2%"),
                sharpe=_fmt(metrics["sharpe"], ".3f"),
                cost=_fmt(metrics["commission_cny"] + metrics["slippage_cny"], ".2f"),
                decision=item["decision"],
            )
        )
    lines.extend(
        [
            "",
            "少于10个类马丁闭合周期只能判为证据不足。本报告不产生真实仓位或订单。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifest = verify_freeze()
    config = load_config()
    source_config = load_source_config()
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    market, _, dividends = load_inputs(ROOT, source_config)
    features, _ = build_features(market, dividends, source_config)
    cash_rate = float(source_config["price_and_execution"]["cash_annual_rate"])
    comparisons = config["evaluation"]["primary_comparisons"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    all_executions: list[pd.DataFrame] = []
    all_cycles: list[pd.DataFrame] = []
    for variant_id in VARIANTS:
        result = simulate_variant(features, dividends, source_config, variant_id)
        metrics = performance_metrics(result["ledger"], cash_rate)
        executions = result["executions"]
        cycles = result["cycles"]
        module_cycles = cycles.loc[cycles["entry_reason"].eq("EXTREME_LAYER_1")]
        layer1_count = int(executions["reason"].eq("EXTREME_LAYER_1").sum())
        layer2_count = int(executions["reason"].eq("EXTREME_LAYER_2").sum())
        transition_count = int(executions["reason"].eq("EXTREME_TO_TURTLE").sum())
        buy = executions.loc[executions["side"].eq("BUY")].copy()
        if not buy.empty:
            buy["one_way_explicit_cost_bps"] = 10_000.0 * (
                buy["commission"] + buy["slippage_cost"]
            ) / buy["raw_notional"]
        source_variant = comparisons[variant_id]
        source = source_report["variants"][source_variant]
        source_base = source["base"]
        module_count = int(len(module_cycles))
        variants[variant_id] = {
            "source_comparison_variant": source_variant,
            "layer1_count": layer1_count,
            "second_layer_full_position_count": layer2_count,
            "turtle_transition_count": transition_count,
            "module_closed_cycles": module_count,
            "total_closed_cycles": int(len(cycles)),
            "module_win_rate": float(module_cycles["net_pnl"].gt(0.0).mean())
            if module_count
            else None,
            "module_net_pnl_cny": float(module_cycles["net_pnl"].sum())
            if module_count
            else 0.0,
            "mean_pnl_before_full_cny": float(module_cycles["pnl_before_full"].mean())
            if module_cycles["pnl_before_full"].notna().any()
            else None,
            "mean_pnl_after_full_cny": float(module_cycles["pnl_after_full"].mean())
            if module_cycles["pnl_after_full"].notna().any()
            else None,
            "maximum_buy_one_way_explicit_cost_bps": float(
                buy["one_way_explicit_cost_bps"].max()
            )
            if not buy.empty
            else None,
            "metrics": metrics,
            "source_metrics": source_base,
            "ending_equity_difference_vs_source_cny": metrics["ending_equity_cny"]
            - source_base["ending_equity_cny"],
            "cagr_difference_vs_source": metrics["cagr"] - source_base["cagr"],
            "decision": "INSUFFICIENT_EVIDENCE"
            if module_count < int(config["evaluation"]["minimum_closed_cycles"])
            else "HISTORICAL_DIAGNOSTIC_ONLY",
        }
        prefix = variant_id.lower()
        result["ledger"].to_parquet(OUTPUT_DIR / f"{prefix}_daily_ledger.parquet", index=False)
        executions.to_csv(OUTPUT_DIR / f"{prefix}_executions.csv", index=False)
        cycles.to_csv(OUTPUT_DIR / f"{prefix}_closed_cycles.csv", index=False)
        all_executions.append(executions)
        all_cycles.append(cycles)
    pd.concat(all_executions, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_execution_events.csv", index=False
    )
    pd.concat(all_cycles, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_closed_cycles.csv", index=False
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "evidence_label": config["protocol"]["evidence_label"],
        "data_cutoff": config["protocol"]["source_data_cutoff"],
        "true_forward_start": None,
        "module": config["module"],
        "variants": variants,
        "t1_secondary_benchmark": source_report["variants"]["T1_TURTLE_GRAPH_EXIT"]["base"],
        "implementation_manifest_sha256": sha256(MANIFEST),
        "freeze_stage": manifest["freeze_stage"],
        "live_position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    safe = _safe(report)
    REPORT_JSON.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(safe), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "summary": {
                    key: {
                        "module_cycles": value["module_closed_cycles"],
                        "full_positions": value["second_layer_full_position_count"],
                        "decision": value["decision"],
                    }
                    for key, value in safe["variants"].items()
                },
                "safety": {
                    "live_position_mapping_enabled": False,
                    "order_generation_enabled": False,
                    "broker_connection_enabled": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
