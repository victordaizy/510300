"""运行冻结的 510300 有锚稀疏均值回归网格 V1 第一阶段。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.anchored_sparse_mean_reversion_grid_v1 import (  # noqa: E402
    CONFIG_PATH,
    atomic_json,
    atomic_parquet,
    atomic_text,
    attach_event_outcomes,
    audit_inputs,
    blocked_report,
    build_minute_features,
    build_report,
    evaluate_phase_a,
    extract_events,
    feature_output_columns,
    load_config,
    load_inputs,
    project_path,
    render_markdown,
    sha256_file,
)
from scripts.freeze_510300_anchored_sparse_mean_reversion_grid_v1 import (  # noqa: E402
    verify,
)


def _authoritative_result_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(config["paths"][key])
        for key in (
            "input_audit_json",
            "result_json",
            "result_markdown",
            "feature_table",
            "event_table",
            "event_outcome_table",
            "gate_receipt",
        )
    ]


def _blocked_markdown(report: dict[str, Any]) -> str:
    failures = report["input_audit"].get("failures", [])
    return "\n".join(
        [
            "# 510300 有锚稀疏均值回归网格 V1：输入阻断",
            "",
            "- 状态：`BLOCKED_INPUT_CONTRACT`",
            "- 收益评价：`NOT_ALLOWED`",
            f"- 失败输入段：{', '.join(failures) if failures else '未知'}",
            "- 未构造事件、未来收益、网格净值、仓位或订单。",
            "- 不回填、不替代锚点、不改参数；只能按同一合同建立机械修正版。",
            "",
        ]
    )


def _failure_payload(error: Exception, verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "report_id": "510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_PHASE_A",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "FAILED_EXECUTION_CONTRACT",
        "goal_achieved": False,
        "return_evaluation": "NOT_COMPLETED",
        "phase_b_grid_backtest": "NOT_ALLOWED",
        "error_type": type(error).__name__,
        "error": str(error),
        "manifest_verification": verification,
        "order_generation": "DISABLED",
        "broker_connection": "DISABLED",
        "live_trading_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 510300 有锚稀疏网格第一阶段")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verification = verify()
    if args.verify_only:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 0 if verification["failure_count"] == 0 else 1
    if verification["failure_count"]:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 1

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if config_path.resolve() != CONFIG_PATH.resolve():
        print(
            json.dumps(
                {
                    "状态": "REFUSED_UNFROZEN_CONFIG_PATH",
                    "冻结配置": CONFIG_PATH.relative_to(ROOT).as_posix(),
                    "请求配置": str(config_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    config = load_config(config_path)
    preexisting = [
        path.relative_to(ROOT).as_posix()
        for path in _authoritative_result_paths(config)
        if path.exists()
    ]
    if preexisting:
        print(
            json.dumps(
                {
                    "状态": "REFUSED_AUTHORITATIVE_RESULT_OVERWRITE",
                    "既有产物": preexisting,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    result_json = project_path(config["paths"]["result_json"])
    result_markdown = project_path(config["paths"]["result_markdown"])
    try:
        inputs = load_inputs(config)
        input_audit = audit_inputs(inputs, config)
        atomic_json(project_path(config["paths"]["input_audit_json"]), input_audit)
        if not input_audit["passed"]:
            report = blocked_report(input_audit, config, verification)
            atomic_json(result_json, report)
            atomic_text(result_markdown, _blocked_markdown(report))
            print(
                json.dumps(
                    {
                        "状态": report["status"],
                        "收益评价": report["return_evaluation"],
                        "第二阶段": "NOT_ALLOWED",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2

        panel = build_minute_features(inputs, config)
        events = extract_events(panel, config)
        outcomes = attach_event_outcomes(events, panel, config)
        evaluation = evaluate_phase_a(outcomes, panel, config)
        report = build_report(
            inputs,
            input_audit,
            panel,
            events,
            outcomes,
            evaluation,
            config,
            verification,
        )

        atomic_parquet(
            project_path(config["paths"]["feature_table"]),
            panel[feature_output_columns()].copy(),
        )
        atomic_parquet(project_path(config["paths"]["event_table"]), events)
        atomic_parquet(project_path(config["paths"]["event_outcome_table"]), outcomes)
        atomic_json(result_json, report)
        atomic_text(result_markdown, render_markdown(report))

        output_hashes: dict[str, Any] = {}
        for key in (
            "input_audit_json",
            "result_json",
            "result_markdown",
            "feature_table",
            "event_table",
            "event_outcome_table",
        ):
            path = project_path(config["paths"][key])
            output_hashes[key] = {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        gate_receipt = {
            "project_id": config["protocol"]["project_id"],
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "status": evaluation["status"],
            "phase_a_passed": bool(evaluation["phase_a_passed"]),
            "phase_b_grid_backtest": evaluation["phase_b_portfolio_backtest"],
            "failed_gates": evaluation["failed_gates"],
            "output_hashes": output_hashes,
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        }
        atomic_json(project_path(config["paths"]["gate_receipt"]), gate_receipt)
        print(
            json.dumps(
                {
                    "状态": evaluation["status"],
                    "第一阶段通过": evaluation["phase_a_passed"],
                    "第二阶段网格回测": evaluation["phase_b_portfolio_backtest"],
                    "成本合格45分钟完整事件数": evaluation["metrics"][
                        "primary_cost_grid_qualified_completed_event_count"
                    ],
                    "基础成本后平均事件收益": evaluation["metrics"][
                        "base_mean_net_event_return"
                    ],
                    "Bootstrap_95%下界": evaluation["bootstrap"].get("lower_bound"),
                    "未通过门槛": evaluation["failed_gates"],
                    "实盘授权": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except Exception as error:
        failure = _failure_payload(error, verification)
        if not result_json.exists():
            atomic_json(result_json, failure)
        if not result_markdown.exists():
            atomic_text(
                result_markdown,
                "\n".join(
                    [
                        "# 510300 有锚稀疏均值回归网格 V1：执行失败",
                        "",
                        "- 状态：`FAILED_EXECUTION_CONTRACT`",
                        f"- 错误类型：`{type(error).__name__}`",
                        f"- 错误：{error}",
                        "- 第二阶段未运行；未生成仓位、订单或实盘授权。",
                        "",
                    ]
                ),
            )
        print(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
