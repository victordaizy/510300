"""执行510300方向切换V1的唯一一次冻结历史诊断与IF机制检验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.direction_switch_common_v1 import (
    CONFIG_PATH,
    MANIFEST_PATH,
    ROOT,
    atomic_json,
    atomic_text,
    generated_at,
    load_config,
    sha256_file,
    verify_manifest,
)
from research.direction_switch_diagnostics_v1 import (
    load_diagnostic_inputs,
    write_diagnostic_artifacts,
)
from research.if_forced_flow_state_v1 import write_if_artifacts


def _render_umbrella(report: dict[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    if_report = report["if_forced_flow"]
    oracle = diagnostics["oracle"]
    frontier = diagnostics["skill_frontier"]
    lines = [
        "# 510300研究方向切换 V1 最终报告",
        "",
        f"最终状态：`{report['status']}`",
        "",
        "本轮已按冻结顺序完成宏观分支封存与事后归因、收益解剖、受约束Oracle、预测能力前沿，以及IF基差残差×持仓冲击×耗竭机制门。",
        "",
        "## 方向切换结果",
        "",
        f"- 宏观分支：`{diagnostics['macro_post_mortem']['status']}`；模型保持拒绝，不允许营救。",
        f"- 收益解剖：`{diagnostics['return_anatomy']['status']}`。",
        f"- Oracle有限搜索族最高净夏普率：{oracle['best_overall_search_family_sharpe']:.4f}。",
        f"- 预测能力前沿：{frontier['passing_cells']}/{frontier['grid_cells']}个冻结网格格子通过全部目标门。",
        f"- IF机制：`{if_report['status']}`。",
        "",
        "## 授权边界",
        "",
        f"- 夏普率1.2目标是否已实现：`{str(report['goal_achieved']).lower()}`",
        f"- 历史收益是否允许解释：`{report['return_evaluation']}`",
        "- Paper/Shadow信号：关闭",
        "- 仓位映射：关闭",
        "- 订单生成：关闭",
        "- 券商连接：关闭",
        "- 实盘交易：未授权",
        "",
        "负结果不会通过改窗口、阈值、方向或追加宏观/估值/趋势/期权/广度因子营救。",
        "",
    ]
    return "\n".join(lines)


def _consume_manifest(result_hashes: dict[str, str]) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if int(manifest["historical_runs_consumed"]) != 0:
        raise ValueError("历史运行次数已经消耗")
    manifest["historical_runs_consumed"] = 1
    manifest["first_completed_run_at"] = generated_at()
    manifest["result_sha256"] = result_hashes
    atomic_json(manifest, MANIFEST_PATH)


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if config_path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许冻结协议入口")
    config = load_config(config_path)
    manifest = verify_manifest(config)
    artifacts = config["artifacts"]
    report_path = ROOT / artifacts["umbrella_report_json"]
    if report_path.exists():
        raise FileExistsError("冻结总报告已存在，禁止重复读取历史结果")
    diagnostics_inputs = load_diagnostic_inputs(config)
    diagnostics = write_diagnostic_artifacts(config, diagnostics_inputs)
    if_result = write_if_artifacts(config)
    if_report = if_result["report"]
    if if_report["mechanism"]["passed"]:
        status = if_report["status"]
        return_evaluation = "COMPLETED_ONCE_AFTER_BOTH_MECHANISM_GATES_PASS"
    else:
        status = "REJECTED_FROZEN_IF_FORCED_FLOW_MECHANISM_GATE_FAILED_NO_RESCUE"
        return_evaluation = "NOT_ALLOWED_FOR_PORTFOLIO"
    report = {
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "status": status,
        "generated_at": generated_at(),
        "protocol_sha256": manifest["protocol_sha256"],
        "pre_run_freeze_manifest_sha256": sha256_file(MANIFEST_PATH),
        "diagnostics": {
            "macro_post_mortem": diagnostics["macro_post_mortem"],
            "return_anatomy": diagnostics["return_anatomy"],
            "oracle": diagnostics["oracle"],
            "skill_frontier": diagnostics["skill_frontier"],
        },
        "if_forced_flow": if_report,
        "return_evaluation": return_evaluation,
        "historical_run_count": 1,
        "goal_net_sharpe": 1.20,
        "goal_achieved": False,
        "goal_achieved_reason": (
            "历史发现或诊断不能替代至少252个新交易日且不少于3个完整压力周期的Shadow验证；当前不授权仓位或交易。"
        ),
        "paper_or_shadow_enabled": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
        "no_parameter_rescue": True,
        "artifact_hashes": {
            **diagnostics["artifact_hashes"],
            **if_result["artifact_hashes"],
        },
    }
    atomic_json(report, report_path)
    markdown_path = ROOT / artifacts["umbrella_report_markdown"]
    atomic_text(_render_umbrella(report), markdown_path)
    final_hashes = {
        **report["artifact_hashes"],
        artifacts["umbrella_report_json"]: sha256_file(report_path),
        artifacts["umbrella_report_markdown"]: sha256_file(markdown_path),
    }
    _consume_manifest(final_hashes)
    print(
        json.dumps(
            {
                "status": status,
                "goal_achieved": False,
                "oracle_best_search_family_sharpe": diagnostics["oracle"][
                    "best_overall_search_family_sharpe"
                ],
                "skill_frontier_passing_cells": diagnostics["skill_frontier"][
                    "passing_cells"
                ],
                "if_mechanism_status": if_report["mechanism"]["status"],
                "if_portfolio_status": if_report["portfolio"]["status"],
                "report_sha256": final_hashes[artifacts["umbrella_report_json"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
