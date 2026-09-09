"""一次性运行DAILY_02历史回测并封存结果。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_02_tom_liquidity_drift_v1 import (
    audit_and_load_inputs,
    build_event_trade_table,
    build_open_intervals,
    build_static_targets,
    build_strategy_targets,
    circular_block_bootstrap_mean,
    cost_model_snapshot,
    event_gross_returns,
    load_config,
    newey_west_event_regression,
    profit_factor,
    run_account_path,
    run_month_matched_placebo,
    sha256_file,
    subperiod_summary,
    summarize_account_path,
    yearly_contribution_share,
)


MANIFEST = ROOT / "config" / "daily_02_tom_liquidity_drift_v1_manifest.json"


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError("冻结清单不存在，禁止运行")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stored_hash = manifest.get("manifest_content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_content_sha256", None)
    if stored_hash != canonical_hash(unsigned):
        raise ValueError("冻结清单内容哈希不一致")
    if manifest.get("historical_run_completed"):
        raise RuntimeError("一次性历史运行已经完成，禁止再次运行")
    for path, expected in manifest["frozen_files"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"冻结实现发生变化：{path}")
    for path, expected in manifest["data_files"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"冻结数据发生变化：{path}")
    if manifest["project_id"] != config["protocol"]["project_id"]:
        raise ValueError("冻结清单项目编号不一致")
    return manifest


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"无法序列化：{type(value)!r}")


def markdown_report(report: dict[str, Any]) -> str:
    base = report["account_paths"]["base_strategy"]
    static = report["account_paths"]["base_static_60"]
    stress = report["account_paths"]["stress_strategy"]
    lines = [
        "# DAILY_02月末—月初流动性漂移历史回测",
        "",
        f"- 状态：`{report['decision']}`",
        f"- 数据截止：`{report['data_audit']['data_cutoff']}`",
        f"- 完整月度事件：{report['data_audit']['complete_event_episodes']}",
        "- 分钟线：未使用",
        "- 仓位、订单、券商：未授权",
        "",
        "## 核心结果",
        "",
        "|指标|结果|门槛|通过|",
        "|---|---:|---:|---|",
        f"|HAC事件系数|{report['factor_statistics']['hac']['event_coefficient']:.6%}|>0且单侧p≤5%|{report['gates']['hac_event_effect']}|",
        f"|HAC单侧p|{report['factor_statistics']['hac']['event_one_sided_p_value']:.4f}|≤0.05|{report['gates']['hac_event_effect']}|",
        f"|基础盈利因子|{report['event_execution']['base_profit_factor']:.3f}|≥1.25|{report['gates']['base_profit_factor']}|",
        f"|压力盈利因子|{report['event_execution']['stress_profit_factor']:.3f}|≥1.10|{report['gates']['stress_profit_factor']}|",
        f"|基础策略夏普|{base['sharpe_excess_cash']:.3f}|≥0.80|{report['gates']['base_strategy_sharpe']}|",
        f"|相对静态60%夏普改善|{report['comparisons']['base_sharpe_improvement_vs_static_60']:.3f}|≥0.15|{report['gates']['sharpe_improvement_vs_static_60']}|",
        f"|基础CAGR|{base['cagr']:.2%}|静态60% {static['cagr']:.2%}|{report['gates']['base_cagr_vs_static_60']}|",
        f"|压力CAGR|{stress['cagr']:.2%}|高于压力静态60%|{report['gates']['stress_cagr_vs_static_60']}|",
        f"|随机四日分位|{report['factor_statistics']['placebo']['placebo_percentile']:.3f}|≥0.95|{report['gates']['placebo_percentile']}|",
        "",
        "## 账户路径",
        "",
        "|路径|CAGR|夏普|最大回撤|平均仓位|交易次数|总执行成本|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("base_strategy", "策略-基础成本"),
        ("base_static_60", "静态60%-基础成本"),
        ("base_buy_hold", "98.5%买入持有"),
        ("stress_strategy", "策略-压力成本"),
        ("stress_static_60", "静态60%-压力成本"),
    ):
        item = report["account_paths"][key]
        lines.append(
            f"|{label}|{item['cagr']:.2%}|{item['sharpe_excess_cash']:.3f}|{item['max_drawdown']:.2%}|{item['average_exposure']:.2%}|{item['trade_count']}|{item['total_execution_cost_cny']:.2f}元|"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            report["decision_explanation"],
            "",
            "本结果只决定历史候选是否通过。无论通过或失败，均不生成当前仓位、订单或实盘指令。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    config = load_config()
    manifest = verify_manifest(config)
    market, dividends, episodes, audit = audit_and_load_inputs(ROOT, config)
    if audit["status"] != "PASS":
        raise RuntimeError(f"数据闸门未通过：{audit['errors']}")
    execution = config["account_and_execution"]
    event = config["event_definition"]
    evaluation = config["evaluation"]
    gates_config = config["gates"]
    targets = build_strategy_targets(
        market,
        episodes,
        float(execution["normal_target_exposure"]),
        float(execution["event_target_exposure"]),
    )
    static_targets = build_static_targets(market, float(config["benchmarks"]["static_60_exposure"]))
    buy_hold_targets = build_static_targets(market, float(config["benchmarks"]["buy_hold_exposure"]))

    paths: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, path_targets, slippage in (
        ("base_strategy", targets, float(execution["base_slippage_bps"])),
        ("base_static_60", static_targets, float(execution["base_slippage_bps"])),
        ("base_buy_hold", buy_hold_targets, float(execution["base_slippage_bps"])),
        ("stress_strategy", targets, float(execution["stress_slippage_bps"])),
        ("stress_static_60", static_targets, float(execution["stress_slippage_bps"])),
    ):
        paths[name] = run_account_path(market, dividends, path_targets, execution, slippage_bps=slippage)
    path_summaries = {
        name: summarize_account_path(ledger, trades, execution)
        for name, (ledger, trades) in paths.items()
    }

    base_events = build_event_trade_table(
        market, dividends, episodes, execution, slippage_bps=float(execution["base_slippage_bps"])
    )
    stress_events = build_event_trade_table(
        market, dividends, episodes, execution, slippage_bps=float(execution["stress_slippage_bps"])
    )
    intervals = build_open_intervals(market, dividends, episodes)
    hac = newey_west_event_regression(
        intervals,
        float(execution["cash_annual_rate"]),
        int(execution["trading_days_per_year"]),
        int(evaluation["hac_lag_trading_days"]),
    )
    bootstrap = circular_block_bootstrap_mean(
        base_events["net_excess_return"].to_numpy(dtype=float),
        block_length=int(evaluation["bootstrap_block_length_episodes"]),
        repetitions=int(evaluation["bootstrap_repetitions"]),
        random_seed=int(evaluation["random_seed"]),
    )
    gross_events = event_gross_returns(intervals)
    placebo, placebo_distribution = run_month_matched_placebo(
        intervals,
        episodes,
        gross_events,
        repetitions=int(evaluation["placebo_repetitions"]),
        random_seed=int(evaluation["random_seed"]),
    )
    periods = subperiod_summary(base_events, config)
    maximum_year_share, annual_contributions = yearly_contribution_share(base_events)
    base_pf = profit_factor(base_events["net_excess_pnl"])
    stress_pf = profit_factor(stress_events["net_excess_pnl"])
    if base_pf is None or stress_pf is None:
        raise RuntimeError("盈利因子无法计算")
    base_strategy = path_summaries["base_strategy"]
    base_static = path_summaries["base_static_60"]
    stress_strategy = path_summaries["stress_strategy"]
    stress_static = path_summaries["stress_static_60"]
    sharpe_improvement = float(base_strategy["sharpe_excess_cash"] - base_static["sharpe_excess_cash"])
    base_cagr_difference = float(base_strategy["cagr"] - base_static["cagr"])
    stress_cagr_difference = float(stress_strategy["cagr"] - stress_static["cagr"])
    gate_checks = {
        "hac_event_effect": hac["event_coefficient"] > 0 and hac["event_one_sided_p_value"] <= float(gates_config["hac_one_sided_p_maximum"]),
        "bootstrap_lower_positive": bootstrap["lower_95"] > float(gates_config["bootstrap_net_event_mean_lower_95_minimum"]),
        "base_profit_factor": base_pf >= float(gates_config["base_profit_factor_minimum"]),
        "stress_profit_factor": stress_pf >= float(gates_config["stress_profit_factor_minimum"]),
        "base_strategy_sharpe": float(base_strategy["sharpe_excess_cash"]) >= float(gates_config["base_strategy_sharpe_minimum"]),
        "sharpe_improvement_vs_static_60": sharpe_improvement >= float(gates_config["sharpe_improvement_vs_static_60_minimum"]),
        "base_cagr_vs_static_60": base_cagr_difference > float(gates_config["base_cagr_difference_vs_static_60_minimum"]),
        "stress_cagr_vs_static_60": stress_cagr_difference > float(gates_config["stress_cagr_difference_vs_static_60_minimum"]),
        "positive_subperiods": sum(item["positive"] for item in periods) >= int(gates_config["positive_subperiods_minimum"]),
        "placebo_percentile": placebo["placebo_percentile"] >= float(gates_config["placebo_percentile_minimum"]),
        "year_contribution_concentration": maximum_year_share is not None and maximum_year_share <= float(gates_config["maximum_single_year_positive_contribution_share"]),
    }
    passed = all(gate_checks.values())
    decision = "PASS_HISTORICAL_GATES_PAPER_PROTOCOL_ONLY" if passed else "REJECT_HISTORICAL_STOP_NO_RESCUE"
    failed_gates = [name for name, passed_gate in gate_checks.items() if not passed_gate]
    report = {
        "project_id": config["protocol"]["project_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "decision": decision,
        "decision_explanation": (
            "全部冻结门槛通过；这只允许另行设计纸面观察协议，不授权仓位或订单。"
            if passed
            else f"冻结门槛失败：{', '.join(failed_gates)}。按协议停止，禁止修改窗口或增加过滤器救援。"
        ),
        "failed_gates": failed_gates,
        "data_audit": audit,
        "event_definition": event,
        "cost_models": {
            "base": cost_model_snapshot(execution, float(execution["base_slippage_bps"])),
            "stress": cost_model_snapshot(execution, float(execution["stress_slippage_bps"])),
        },
        "factor_statistics": {"hac": hac, "bootstrap": bootstrap, "placebo": placebo},
        "event_execution": {
            "episodes": int(len(base_events)),
            "base_profit_factor": float(base_pf),
            "stress_profit_factor": float(stress_pf),
            "base_total_net_excess_pnl": float(base_events["net_excess_pnl"].sum()),
            "stress_total_net_excess_pnl": float(stress_events["net_excess_pnl"].sum()),
            "subperiods": periods,
            "positive_subperiod_count": int(sum(item["positive"] for item in periods)),
            "annual_net_excess_pnl": annual_contributions,
            "maximum_single_year_positive_contribution_share": maximum_year_share,
        },
        "account_paths": path_summaries,
        "comparisons": {
            "base_sharpe_improvement_vs_static_60": sharpe_improvement,
            "base_cagr_difference_vs_static_60": base_cagr_difference,
            "stress_cagr_difference_vs_static_60": stress_cagr_difference,
            "base_sharpe_improvement_vs_buy_hold": float(base_strategy["sharpe_excess_cash"] - path_summaries["base_buy_hold"]["sharpe_excess_cash"]),
        },
        "gates": gate_checks,
        "governance": {
            "minute_data_used": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }

    outputs = config["outputs"]
    for path in outputs.values():
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    paths["base_strategy"][0].to_parquet(ROOT / outputs["base_ledger"], index=False)
    paths["base_strategy"][1].to_parquet(ROOT / outputs["base_trades"], index=False)
    paths["stress_strategy"][0].to_parquet(ROOT / outputs["stress_ledger"], index=False)
    paths["stress_strategy"][1].to_parquet(ROOT / outputs["stress_trades"], index=False)
    combined_episodes = base_events.merge(
        stress_events[["episode_id", "net_excess_return", "net_excess_pnl"]].rename(
            columns={"net_excess_return": "stress_net_excess_return", "net_excess_pnl": "stress_net_excess_pnl"}
        ),
        on="episode_id",
        how="left",
        validate="one_to_one",
    ).merge(gross_events, left_on="episode_id", right_on="event_episode_id", how="left", validate="one_to_one")
    combined_episodes.to_parquet(ROOT / outputs["episodes"], index=False)
    placebo_distribution.to_parquet(ROOT / outputs["placebo"], index=False)
    json_path = ROOT / outputs["report_json"]
    markdown_path = ROOT / outputs["report_markdown"]
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")

    manifest.update(
        {
            "historical_run_completed": True,
            "historical_run_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "real_event_returns_seen_after_freeze": True,
            "strategy_backtest_computed_after_freeze": True,
            "historical_result_status": "PASS" if passed else "REJECT",
            "decision": decision,
            "failed_gates": failed_gates,
            "result_files": {
                path: sha256_file(ROOT / path)
                for path in outputs.values()
            },
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        }
    )
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps({"decision": decision, "failed_gates": failed_gates, "report": outputs["report_markdown"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
