"""运行510300图形状态马丁-海龟V2冻结历史检验。"""

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

from research.graph_regime_martin_turtle_v2 import (
    VARIANTS,
    build_features,
    evaluate_results,
    load_config,
    load_inputs,
    simulate_variant,
)


IMPLEMENTATION_MANIFEST = (
    ROOT
    / "config"
    / "graph_regime_martin_turtle_v2_implementation_manifest_reporting_fix.json"
)
DATA_GATE_REPORT = ROOT / "reports" / "data_quality" / "donchian_discovery_v1_data_gate.json"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "graph_regime_v2"
REPORT_JSON = ROOT / "reports" / "backtest" / "graph_regime_martin_turtle_v2.json"
REPORT_MARKDOWN = ROOT / "reports" / "backtest" / "graph_regime_martin_turtle_v2.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_freeze() -> dict[str, Any]:
    if not IMPLEMENTATION_MANIFEST.exists():
        raise RuntimeError("NO_VIEW：V2实现尚未冻结，禁止计算历史收益")
    manifest = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("return_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：实现冻结清单未授权历史收益计算")
    for section in ("implementation_files", "input_files"):
        for relative_path, expected_hash in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected_hash:
                raise RuntimeError(f"NO_VIEW：冻结后文件指纹变化：{relative_path}")
    return manifest


def _verify_data_gate() -> dict[str, Any]:
    if not DATA_GATE_REPORT.exists():
        raise RuntimeError("NO_VIEW：数据闸门报告缺失")
    report = json.loads(DATA_GATE_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("return_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：数据闸门未通过")
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _transition_summary(
    results: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_id, frames in results.items():
        cycles = frames["cycles"]
        if cycles.empty:
            continue
        grouped = cycles.groupby(
            ["entry_reason", "switch_reason", "exit_reason"],
            dropna=False,
        )
        for keys, group in grouped:
            entry_reason, switch_reason, exit_reason = keys
            rows.append(
                {
                    "variant_id": variant_id,
                    "entry_reason": entry_reason,
                    "switch_reason": switch_reason,
                    "exit_reason": exit_reason,
                    "closed_cycles": int(len(group)),
                    "net_pnl_cny": float(group["net_pnl"].sum()),
                    "mean_pnl_cny": float(group["net_pnl"].mean()),
                    "win_rate": float(group["net_pnl"].gt(0.0).mean()),
                    "pnl_to_switch_cny": float(group["pnl_to_switch"].sum(min_count=1))
                    if group["pnl_to_switch"].notna().any()
                    else None,
                    "pnl_after_switch_cny": float(
                        group["pnl_after_switch"].sum(min_count=1)
                    )
                    if group["pnl_after_switch"].notna().any()
                    else None,
                }
            )
    return pd.DataFrame(rows)


def _markdown_report(report: dict[str, Any], event_count: dict[str, int]) -> str:
    def metric(value: Any, format_spec: str) -> str:
        if value is None:
            return "N/A"
        return format(float(value), format_spec)

    lines = [
        "# 510300图形状态马丁-海龟V2冻结历史检验",
        "",
        f"- 数据截止：{report['data_cutoff']}",
        f"- 证据标签：{report['evidence_label']}",
        f"- 背离事件：底背离 {event_count['bottom']} 个，顶背离 {event_count['top']} 个",
        f"- 注册候选通过数：{report['registered_pass_count']} / 5",
        "- 安全边界：仅研究/纸面账本；真实仓位映射、订单生成、券商连接均关闭。",
        "",
        "| 轨道 | 决策 | 闭合交易 | CAGR | 最大回撤 | Sharpe | 平均仓位 | 时机贡献 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id, evidence in report["variants"].items():
        base = evidence["base"]
        lines.append(
            "| {variant} | {decision} | {trades} | {cagr} | {drawdown} | "
            "{sharpe} | {exposure} | {timing} |".format(
                variant=variant_id,
                decision=evidence["decision"],
                trades=evidence["closed_trades"],
                cagr=metric(base["cagr"], ".2%"),
                drawdown=metric(base["maximum_drawdown"], ".2%"),
                sharpe=metric(base["sharpe"], ".3f"),
                exposure=metric(base["average_exposure"], ".2%"),
                timing=metric(evidence["timing_contribution_after_cost"], ".2%"),
            )
        )
    lines.extend(
        [
            "",
            "## 基准",
            "",
            "- 510300含分红买入持有：CAGR {cagr:.2%}，最大回撤 {drawdown:.2%}，Sharpe {sharpe:.3f}。".format(
                cagr=report["buy_hold"]["cagr"],
                drawdown=report["buy_hold"]["maximum_drawdown"],
                sharpe=report["buy_hold"]["sharpe"],
            ),
            "- H00300全收益指数：CAGR {cagr:.2%}，Sharpe {sharpe:.3f}。".format(
                cagr=report["h00300"]["cagr"],
                sharpe=report["h00300"]["sharpe"],
            ),
            "",
            "## 判定纪律",
            "",
            "任何失败轨道都按 `REJECT_WITHOUT_PARAMETER_RESCUE` 处理；交易不足30次则为 `INSUFFICIENT_EVIDENCE`。本报告不构成真实交易指令。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    implementation_manifest = _verify_freeze()
    data_gate = _verify_data_gate()
    config = load_config()
    market, benchmark, dividends = load_inputs(ROOT, config)
    features, events = build_features(market, dividends, config)

    base_results: dict[str, dict[str, pd.DataFrame]] = {}
    stress_results: dict[str, dict[str, pd.DataFrame]] = {}
    for variant_id in VARIANTS:
        base_results[variant_id] = simulate_variant(
            features, dividends, config, variant_id, cost_multiplier=1.0
        )
        stress_results[variant_id] = simulate_variant(
            features, dividends, config, variant_id, cost_multiplier=2.0
        )

    report, _ = evaluate_results(
        features,
        dividends,
        benchmark,
        config,
        base_results,
        stress_results,
    )
    report["implementation_manifest_sha256"] = sha256(IMPLEMENTATION_MANIFEST)
    report["data_gate_report_sha256"] = sha256(DATA_GATE_REPORT)
    report["data_gate_status"] = data_gate["status"]
    report["implementation_freeze_stage"] = implementation_manifest["freeze_stage"]
    report["divergence_events"] = {
        "total": int(len(events)),
        "bottom": int(events["event_type"].eq("BOTTOM").sum()),
        "top": int(events["event_type"].eq("TOP").sum()),
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_DIRECTORY / "daily_features.parquet", index=False)
    events.to_csv(OUTPUT_DIRECTORY / "divergence_event_audit.csv", index=False)
    all_executions: list[pd.DataFrame] = []
    all_cycles: list[pd.DataFrame] = []
    for variant_id, frames in base_results.items():
        prefix = variant_id.lower()
        frames["ledger"].to_parquet(
            OUTPUT_DIRECTORY / f"{prefix}_daily_ledger.parquet", index=False
        )
        frames["executions"].to_csv(
            OUTPUT_DIRECTORY / f"{prefix}_executions.csv", index=False
        )
        frames["cycles"].to_csv(
            OUTPUT_DIRECTORY / f"{prefix}_closed_cycles.csv", index=False
        )
        if not frames["executions"].empty:
            all_executions.append(frames["executions"])
        if not frames["cycles"].empty:
            all_cycles.append(frames["cycles"])
    pd.concat(all_executions, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "all_execution_events.csv", index=False
    )
    pd.concat(all_cycles, ignore_index=True).to_csv(
        OUTPUT_DIRECTORY / "all_closed_cycles.csv", index=False
    )
    _transition_summary(base_results).to_csv(
        OUTPUT_DIRECTORY / "state_transition_pnl.csv", index=False
    )

    safe_report = _json_safe(report)
    REPORT_JSON.write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MARKDOWN.write_text(
        _markdown_report(safe_report, safe_report["divergence_events"]),
        encoding="utf-8",
    )
    concise = {
        "report": str(REPORT_JSON),
        "events": safe_report["divergence_events"],
        "registered_pass_count": safe_report["registered_pass_count"],
        "decisions": {
            key: value["decision"] for key, value in safe_report["variants"].items()
        },
        "safety": {
            "live_position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
