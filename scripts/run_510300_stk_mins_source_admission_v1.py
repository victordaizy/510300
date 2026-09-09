"""运行510300 STK_MINS来源准入V1。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.stk_mins_source_admission_v1 import (  # noqa: E402
    DATA_MODEL_ID,
    build_client,
    load_protocol,
    resolve_ephemeral_token,
    run_full_acquisition,
    run_permission_probe,
    run_source_adjudication,
    sanitize_text,
    verify_frozen_manifests,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集并裁决510300的STK_MINS一分钟数据；不读取策略收益。"
    )
    parser.add_argument(
        "mode",
        choices=("verify", "probe", "acquire", "adjudicate", "replay-check"),
    )
    parser.add_argument(
        "--token-environment-name",
        default="TUSHARE_PROXY_TOKEN",
        help="临时凭据环境变量名；未设置时隐藏交互输入",
    )
    parser.add_argument(
        "--event-ledger",
        default="",
        help="可选的预收益事件账本，用于最终事件窗口覆盖门",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    token = ""
    try:
        protocol = load_protocol(PROJECT_ROOT)
        protocol_manifest, implementation_manifest = verify_frozen_manifests(PROJECT_ROOT)
        event_ledger = (
            PROJECT_ROOT / Path(arguments.event_ledger) if arguments.event_ledger else None
        )
        if arguments.mode == "verify":
            payload = {
                "state": "PASS_PROTOCOL_AND_IMPLEMENTATION_INTEGRITY",
                "protocol_freeze_id": protocol_manifest["freeze_id"],
                "implementation_freeze_id": implementation_manifest["freeze_id"],
                "strategy_return_reads": 0,
                "position_impact": 0,
            }
        elif arguments.mode == "probe":
            token = resolve_ephemeral_token(arguments.token_environment_name)
            payload = run_permission_probe(
                PROJECT_ROOT,
                protocol,
                build_client(protocol, token),
                protocol_manifest,
                implementation_manifest,
            )
        elif arguments.mode == "acquire":
            token = resolve_ephemeral_token(arguments.token_environment_name)
            payload = run_full_acquisition(
                PROJECT_ROOT,
                protocol,
                build_client(protocol, token),
            )
        elif arguments.mode == "adjudicate":
            payload = run_source_adjudication(
                PROJECT_ROOT,
                protocol,
                event_ledger_path=event_ledger,
            )
        else:
            report_path = PROJECT_ROOT / Path(protocol["storage"]["source_adjudication"])
            if not report_path.is_file():
                raise FileNotFoundError("缺少首次来源裁决，不能运行重放检查")
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            payload = run_source_adjudication(
                PROJECT_ROOT,
                protocol,
                event_ledger_path=event_ledger,
                replay_expected_fingerprint=str(previous["decision_fingerprint"]),
            )
        concise = {
            "data_model_id": DATA_MODEL_ID,
            "mode": arguments.mode,
            "state": payload.get("state"),
            "general_source_pass": payload.get("general_source_pass"),
            "event_window_gate_pass": payload.get("event_window_gate_pass"),
            "decision_fingerprint": payload.get("decision_fingerprint"),
            "strategy_return_reads": int(payload.get("strategy_return_reads", 0)),
            "position_impact": int(payload.get("position_impact", 0)),
        }
        print(json.dumps(concise, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "data_model_id": DATA_MODEL_ID,
                    "mode": arguments.mode,
                    "state": "PROGRAM_FAILED",
                    "error_type": type(exc).__name__,
                    "error_message": sanitize_text(exc, token),
                    "strategy_return_reads": 0,
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
