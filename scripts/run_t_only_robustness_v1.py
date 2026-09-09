"""运行冻结 T_ONLY 历史压力测试。"""

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
    block_bootstrap_mean_interval,
    circular_shift_percentile,
)
from research.t_only_robustness_forward_v1 import (
    load_stress_config,
    profitable_cycle_concentration,
    replication_audit,
    select_missed_entry_dates,
    simulate_t_only,
)
from research.weekly_daily_technical_v1 import (
    build_features,
    load_config,
    load_inputs,
    performance_metrics,
    simulate_static_exposure,
    simulate_variant,
)


MANIFEST = ROOT / "config" / "t_only_forward_v1_freeze_manifest.json"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "t_only_robustness_v1"
REPORT_JSON = ROOT / "reports" / "backtest" / "t_only_robustness_v1.json"
REPORT_MARKDOWN = ROOT / "reports" / "backtest" / "t_only_robustness_v1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> dict[str, Any]:
    """验证压力协议、实现和父冻结清单没有变化。"""

    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：T_ONLY 压力测试冻结清单缺失")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("stress_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：冻结清单未授权压力计算")
    for section in (
        "protocol_files",
        "implementation_files",
        "parent_files",
        "parent_input_files",
    ):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：冻结文件指纹变化：{relative_path}")
    return manifest


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown(report: dict[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    def num(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    lines = [
        "# 510300 T_ONLY 冻结历史压力测试 V1",
        "",
        f"- 数据截止：{report['data_cutoff']}",
        f"- 基线复现：{report['baseline_replication']['status']}",
        "- 证据边界：历史稳健性否证，不是样本外收益，不产生真实仓位或订单。",
        "",
        "| 场景 | 期末权益 | 总收益 | CAGR | 最大回撤 | Sharpe | 成交腿 | 正收益是否保留 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario_id, evidence in report["scenarios"].items():
        metrics = evidence["metrics"]
        lines.append(
            f"| {scenario_id} | {metrics['ending_equity_cny']:.2f} | "
            f"{pct(metrics['total_return'])} | {pct(metrics['cagr'])} | "
            f"{pct(metrics['maximum_drawdown'])} | {num(metrics['sharpe'])} | "
            f"{evidence['execution_legs']} | {evidence['positive_total_return']} |"
        )
    statistics = report["statistical_placebos"]
    lines.extend(
        [
            "",
            "## 路径脆弱性",
            "",
            f"- 循环位移百分位：{num(statistics['circular_shift_percentile'])}。",
            f"- 相对同平均仓位基准的年化日收益差区块自助法 95% 区间："
            f"{pct(statistics['annualized_incremental_return_bootstrap_95pct'][0])} 至 "
            f"{pct(statistics['annualized_incremental_return_bootstrap_95pct'][1])}。",
            f"- 最大单个盈利周期占全部正盈利：{pct(report['baseline_profitable_cycle_concentration'])}。",
            "",
            "## 纪律",
            "",
            "这些结果只能暴露执行假设脆弱性。参数不得据此调整；真正验证只来自 2026-08-19 以后冻结规则的影子样本。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    manifest = verify_manifest()
    stress_config = load_stress_config()
    base_config = load_config()
    market, _, dividends = load_inputs(ROOT, base_config)
    features, _ = build_features(market, dividends, base_config)
    cash_rate = float(base_config["price_and_execution"]["cash_annual_rate"])

    parent = simulate_variant(features, dividends, base_config, "T_ONLY")
    replica = simulate_t_only(
        features,
        dividends,
        base_config,
        scenario_id="BASELINE_REPLICATION",
    )
    audit = replication_audit(parent, replica)
    if audit["status"] != "PASS":
        raise RuntimeError("NO_VIEW_BASELINE_REPLICATION_FAILED")

    stress = stress_config["stress_test"]
    missed_dates = select_missed_entry_dates(
        parent["executions"],
        fraction=0.10,
        random_seed=int(stress["missed_entry_seed"]),
    )
    results: dict[str, dict[str, Any]] = {}
    for scenario in stress["scenarios"]:
        scenario_id = str(scenario["id"])
        blocked_dates = missed_dates if float(scenario["missed_entry_fraction"]) > 0 else []
        result = simulate_t_only(
            features,
            dividends,
            base_config,
            scenario_id=scenario_id,
            execution_delay_bars=int(scenario["execution_delay_bars"]),
            commission_multiplier=float(scenario["commission_multiplier"]),
            slippage_bps=float(scenario["slippage_bps"]),
            blocked_entry_signal_dates=blocked_dates,
        )
        metrics = performance_metrics(result["ledger"], cash_rate)
        results[scenario_id] = {
            "parameters": scenario,
            "metrics": metrics,
            "execution_legs": int(len(result["executions"])),
            "closed_cycles": int(len(result["cycles"])),
            "skipped_orders": int(len(result["skipped_orders"])),
            "missed_signals": int(len(result["missed_signals"])),
            "positive_total_return": bool(metrics["total_return"] > 0.0),
            "profitable_cycle_concentration": profitable_cycle_concentration(
                result["cycles"]
            ),
        }
        result["ledger"].to_parquet(
            OUTPUT_DIRECTORY / f"{scenario_id.lower()}_daily_ledger.parquet",
            index=False,
        )
        result["executions"].to_csv(
            OUTPUT_DIRECTORY / f"{scenario_id.lower()}_executions.csv", index=False
        )
        result["cycles"].to_csv(
            OUTPUT_DIRECTORY / f"{scenario_id.lower()}_closed_cycles.csv", index=False
        )
        result["missed_signals"].to_csv(
            OUTPUT_DIRECTORY / f"{scenario_id.lower()}_missed_signals.csv", index=False
        )

    baseline = replica["ledger"]
    average_exposure = float(results["BASELINE_REPLICATION"]["metrics"]["average_exposure"])
    matched = simulate_static_exposure(
        features, dividends, base_config, average_exposure
    )
    matched_metrics = performance_metrics(matched, cash_rate)
    shift = stress["circular_shift"]
    circular_percentile = circular_shift_percentile(
        baseline["exposure"],
        features["adjusted_close"],
        repetitions=int(shift["repetitions"]),
        minimum_shift=int(shift["minimum_shift_days"]),
        random_seed=int(shift["random_seed"]),
    )
    incremental = baseline["equity"].pct_change(fill_method=None) - matched[
        "equity"
    ].pct_change(fill_method=None)
    bootstrap = stress["block_bootstrap"]
    bootstrap_interval = block_bootstrap_mean_interval(
        incremental,
        repetitions=int(bootstrap["repetitions"]),
        block_length=int(bootstrap["block_length_days"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    double_total_cost = simulate_variant(
        features, dividends, base_config, "T_ONLY", cost_multiplier=2.0
    )
    double_total_cost_metrics = performance_metrics(
        double_total_cost["ledger"], cash_rate
    )

    scenario_returns = {
        scenario_id: float(evidence["metrics"]["total_return"])
        for scenario_id, evidence in results.items()
    }
    worst_scenario = min(scenario_returns, key=scenario_returns.get)
    report = safe(
        {
            "project_id": stress_config["protocol"]["project_id"],
            "evidence_label": "HISTORICAL_ROBUSTNESS_DISCOVERY_ONLY",
            "data_cutoff": str(features["date"].iloc[-1].date()),
            "manifest_sha256": sha256(MANIFEST),
            "parent_manifest_sha256": manifest["parent_files"][
                "config/weekly_daily_technical_v1_freeze_manifest.json"
            ],
            "baseline_replication": audit,
            "scenarios": results,
            "missed_entry_signal_dates": missed_dates,
            "matched_exposure": matched_metrics,
            "double_total_cost": double_total_cost_metrics,
            "baseline_profitable_cycle_concentration": profitable_cycle_concentration(
                replica["cycles"]
            ),
            "statistical_placebos": {
                "circular_shift_percentile": circular_percentile,
                "circular_shift_parameters": shift,
                "annualized_incremental_return_bootstrap_95pct": bootstrap_interval,
                "block_bootstrap_parameters": bootstrap,
            },
            "summary": {
                "all_stress_scenarios_positive": all(
                    value > 0.0 for value in scenario_returns.values()
                ),
                "worst_total_return_scenario": worst_scenario,
                "worst_total_return": scenario_returns[worst_scenario],
                "double_total_cost_positive": bool(
                    double_total_cost_metrics["total_return"] > 0.0
                ),
                "historical_decision": "DESCRIPTIVE_ONLY_NO_PARAMETER_RESCUE",
            },
            "safety": stress_config["governance"],
        }
    )
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MARKDOWN.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告：{REPORT_MARKDOWN}")
    return 0


if __name__ == "__main__":
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(main())
