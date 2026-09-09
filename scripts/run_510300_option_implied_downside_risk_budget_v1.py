"""运行 510300 期权隐含下行风险预算 V1 的阶段 B/C 数据任务。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.option_implied_downside_risk_budget_v1 import (  # noqa: E402
    MODEL_ID,
    build_client,
    load_protocol,
    resolve_ephemeral_token,
    run_data_adjudication,
    run_full_acquisition,
    run_permission_probe,
    sanitize_text,
    verify_frozen_manifests,
)


def parse_arguments() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(
        description=(
            "只执行 510300_OPTION_IMPLIED_DOWNSIDE_RISK_BUDGET_V1 的权限探针、"
            "原始下载和 G0/G1 数据准入；不读取未来收益。"
        )
    )
    parser.add_argument(
        "mode",
        choices=("verify", "probe", "acquire", "adjudicate", "replay-check"),
        help="执行模式",
    )
    parser.add_argument(
        "--token-environment-name",
        default="TUSHARE_PROXY_TOKEN",
        help="临时凭据环境变量名；未设置时使用隐藏交互输入",
    )
    return parser.parse_args()


def concise_result(mode: str, payload: dict) -> dict:
    """生成终端可读且不含凭据的摘要。"""

    result = {
        "model_id": MODEL_ID,
        "mode": mode,
        "state": payload.get("state", "PASS_INTEGRITY_ONLY"),
        "future_return_reads": int(payload.get("future_return_reads", 0)),
        "portfolio_evaluation_run": bool(
            payload.get("portfolio_evaluation_run", False)
        ),
        "position_impact": int(payload.get("position_impact", 0)),
    }
    if mode == "probe":
        result.update(
            {
                "failed_core_apis": payload.get("failed_core_apis", []),
                "etf_mins_available": payload.get("etf_mins_available"),
                "core_strategy_data_acquisition_allowed": payload.get(
                    "core_strategy_data_acquisition_allowed"
                ),
            }
        )
    elif mode == "acquire":
        result.update(
            {
                "start_date": payload.get("start_date"),
                "end_date": payload.get("end_date"),
                "open_trade_dates": payload.get("open_trade_dates"),
                "partition_counts": payload.get("partition_counts", {}),
                "row_counts": payload.get("row_counts", {}),
            }
        )
    elif mode in {"adjudicate", "replay-check"}:
        result.update(
            {
                "replay_state": payload.get("replay_state"),
                "g0_prediction_pass": payload.get("g0", {}).get(
                    "prediction_data_admission_pass"
                ),
                "g0_final_execution_pass": payload.get("g0", {}).get(
                    "final_execution_data_admission_pass"
                ),
                "g1_pass": payload.get("g1", {}).get("pass"),
                "valid_surface_days": payload.get("g1", {}).get(
                    "valid_surface_days"
                ),
                "overall_valid_coverage": payload.get("g1", {}).get(
                    "overall_valid_coverage"
                ),
                "decision_fingerprint": payload.get("decision_fingerprint"),
            }
        )
    return result


def main() -> int:
    """验证冻结状态后执行指定阶段。"""

    arguments = parse_arguments()
    token = ""
    try:
        protocol = load_protocol(PROJECT_ROOT)
        protocol_manifest, implementation_manifest = verify_frozen_manifests(
            PROJECT_ROOT
        )
        if arguments.mode == "verify":
            payload = {
                "state": "PASS_PROTOCOL_AND_IMPLEMENTATION_INTEGRITY",
                "protocol_freeze_id": protocol_manifest["freeze_id"],
                "implementation_freeze_id": implementation_manifest["freeze_id"],
                "future_return_reads": 0,
                "portfolio_evaluation_run": False,
                "position_impact": 0,
            }
        elif arguments.mode == "probe":
            token = resolve_ephemeral_token(arguments.token_environment_name)
            client = build_client(protocol, token)
            payload = run_permission_probe(
                PROJECT_ROOT,
                protocol,
                client,
                protocol_manifest,
                implementation_manifest,
            )
        elif arguments.mode == "acquire":
            token = resolve_ephemeral_token(arguments.token_environment_name)
            client = build_client(protocol, token)
            payload = run_full_acquisition(PROJECT_ROOT, protocol, client)
        elif arguments.mode == "adjudicate":
            payload = run_data_adjudication(PROJECT_ROOT, protocol)
        else:
            report_path = PROJECT_ROOT / protocol["planned_outputs"][
                "g0_g1_adjudication"
            ]
            if not report_path.is_file():
                raise FileNotFoundError("缺少首次 G0/G1 裁决，不能执行新进程重放")
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            expected = previous.get("decision_fingerprint")
            if not expected:
                raise RuntimeError("首次 G0/G1 裁决缺少 decision_fingerprint")
            payload = run_data_adjudication(
                PROJECT_ROOT,
                protocol,
                replay_expected_fingerprint=str(expected),
            )
        print(json.dumps(concise_result(arguments.mode, payload), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "mode": arguments.mode,
                    "state": "PROGRAM_FAILED",
                    "error_type": type(exc).__name__,
                    "error_message": sanitize_text(exc, token),
                    "future_return_reads": 0,
                    "portfolio_evaluation_run": False,
                    "position_impact": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
