"""运行V2.1小单成本门历史对照。"""

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

from research.graph_regime_cost_gate_v2_1 import (
    load_cost_gate_config,
    simulate_variant_cost_gated,
)
from research.graph_regime_martin_turtle_v2 import (
    VARIANTS,
    build_features,
    load_config,
    load_inputs,
    performance_metrics,
)


MANIFEST = ROOT / "config" / "graph_regime_cost_gate_v2_1_manifest.json"
SOURCE_REPORT = ROOT / "reports" / "backtest" / "graph_regime_martin_turtle_v2.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "graph_regime_cost_gate_v2_1"
REPORT_JSON = ROOT / "reports" / "backtest" / "graph_regime_cost_gate_v2_1.json"
REPORT_MD = ROOT / "reports" / "backtest" / "graph_regime_cost_gate_v2_1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：V2.1成本门尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("return_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：V2.1冻结清单不允许计算收益")
    for section in ("frozen_files", "input_files"):
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


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 510300图形状态策略V2.1小单成本门对照",
        "",
        "- 证据标签：`POST_RESULT_EXECUTION_OVERLAY_HISTORICALLY_CONTAMINATED`",
        "- 成本门：买入或加仓的单边显式成本不得超过10bp；卖出与减仓全部豁免。",
        "- 推导最低买入金额：10000元。",
        f"- 总拦截买单：{report['total_blocked_buy_orders']}笔。",
        "- 本报告是结果产生后的执行成本敏感性，不是新的样本外alpha证据。",
        "",
        "| 轨道 | 拦截买单 | 原闭合交易 | 门后闭合交易 | 原CAGR | 门后CAGR | 期末权益变化 | 显式成本变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id, item in report["variants"].items():
        lines.append(
            "| {variant} | {blocked} | {old_trades} | {new_trades} | {old_cagr} | "
            "{new_cagr} | {equity_delta}元 | {cost_delta}元 |".format(
                variant=variant_id,
                blocked=item["blocked_buy_orders"],
                old_trades=item["source_closed_trades"],
                new_trades=item["cost_gated_closed_trades"],
                old_cagr=_fmt(item["source_cagr"], ".2%"),
                new_cagr=_fmt(item["cost_gated_cagr"], ".2%"),
                equity_delta=_fmt(item["ending_equity_difference_cny"], ".2f"),
                cost_delta=_fmt(item["explicit_cost_difference_cny"], ".2f"),
            )
        )
    lines.extend(
        [
            "",
            "显式成本变化定义为“成本门后成本－V2原成本”，负数代表节省。成本门不产生真实仓位或订单。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifest = verify_freeze()
    cost_gate_config = load_cost_gate_config()
    source_config = load_config()
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    market, _, dividends = load_inputs(ROOT, source_config)
    features, _ = build_features(market, dividends, source_config)
    cash_rate = float(source_config["price_and_execution"]["cash_annual_rate"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    all_blocked: list[pd.DataFrame] = []
    all_executions: list[pd.DataFrame] = []
    for variant_id in VARIANTS:
        result = simulate_variant_cost_gated(
            features,
            dividends,
            source_config,
            cost_gate_config,
            variant_id,
        )
        metrics = performance_metrics(result["ledger"], cash_rate)
        source = source_report["variants"][variant_id]
        source_base = source["base"]
        blocked = result["blocked_orders"]
        executions = result["executions"]
        gated_cost = metrics["commission_cny"] + metrics["slippage_cny"]
        source_cost = source_base["commission_cny"] + source_base["slippage_cny"]
        variants[variant_id] = {
            "blocked_buy_orders": int(len(blocked)),
            "blocked_intended_notional_cny": float(
                blocked["intended_raw_notional"].sum()
            )
            if not blocked.empty
            else 0.0,
            "blocked_estimated_one_way_cost_cny": float(
                (blocked["estimated_commission"] + blocked["estimated_slippage_cost"]).sum()
            )
            if not blocked.empty
            else 0.0,
            "source_closed_trades": int(source["closed_trades"]),
            "cost_gated_closed_trades": int(len(result["cycles"])),
            "source_ending_equity_cny": source_base["ending_equity_cny"],
            "cost_gated_ending_equity_cny": metrics["ending_equity_cny"],
            "ending_equity_difference_cny": metrics["ending_equity_cny"]
            - source_base["ending_equity_cny"],
            "source_cagr": source_base["cagr"],
            "cost_gated_cagr": metrics["cagr"],
            "cagr_difference": metrics["cagr"] - source_base["cagr"],
            "source_maximum_drawdown": source_base["maximum_drawdown"],
            "cost_gated_maximum_drawdown": metrics["maximum_drawdown"],
            "source_explicit_cost_cny": source_cost,
            "cost_gated_explicit_cost_cny": gated_cost,
            "explicit_cost_difference_cny": gated_cost - source_cost,
            "state_transition_count": int(
                executions["reason"]
                .isin(["MARTIN_TO_TURTLE", "PRING_TO_TURTLE"])
                .sum()
            )
            if not executions.empty
            else 0,
            "metrics": metrics,
            "interpretation": "IMPROVED_ENDING_EQUITY"
            if metrics["ending_equity_cny"] > source_base["ending_equity_cny"] + 0.01
            else "WORSE_ENDING_EQUITY"
            if metrics["ending_equity_cny"] < source_base["ending_equity_cny"] - 0.01
            else "UNCHANGED",
        }
        prefix = variant_id.lower()
        result["ledger"].to_parquet(OUTPUT_DIR / f"{prefix}_daily_ledger.parquet", index=False)
        result["executions"].to_csv(OUTPUT_DIR / f"{prefix}_executions.csv", index=False)
        result["cycles"].to_csv(OUTPUT_DIR / f"{prefix}_closed_cycles.csv", index=False)
        result["blocked_orders"].to_csv(
            OUTPUT_DIR / f"{prefix}_blocked_orders.csv", index=False
        )
        if not blocked.empty:
            all_blocked.append(blocked)
        if not executions.empty:
            all_executions.append(executions)

    blocked_audit = pd.concat(all_blocked, ignore_index=True) if all_blocked else pd.DataFrame()
    execution_audit = (
        pd.concat(all_executions, ignore_index=True) if all_executions else pd.DataFrame()
    )
    blocked_audit.to_csv(OUTPUT_DIR / "all_blocked_buy_orders.csv", index=False)
    execution_audit.to_csv(OUTPUT_DIR / "all_executions.csv", index=False)
    report = {
        "project_id": cost_gate_config["protocol"]["project_id"],
        "version": cost_gate_config["protocol"]["version"],
        "evidence_label": cost_gate_config["protocol"]["evidence_label"],
        "data_cutoff": cost_gate_config["protocol"]["source_data_cutoff"],
        "true_forward_start": None,
        "cost_gate": cost_gate_config["cost_gate"],
        "total_blocked_buy_orders": int(len(blocked_audit)),
        "variants": variants,
        "manifest_sha256": sha256(MANIFEST),
        "source_report_sha256": sha256(SOURCE_REPORT),
        "freeze_stage": manifest["freeze_stage"],
        "live_position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    safe = _safe(report)
    REPORT_JSON.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(_markdown(safe), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "total_blocked_buy_orders": safe["total_blocked_buy_orders"],
                "interpretations": {
                    key: value["interpretation"] for key, value in safe["variants"].items()
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
