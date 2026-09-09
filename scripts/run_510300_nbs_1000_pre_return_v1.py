"""运行510300国家统计局10时策略的预收益准入。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.nbs_1000_pre_return_admission_v1 import (  # noqa: E402
    MODEL_ID,
    acquire_official_schedules,
    adjudicate_g0_g1,
    build_pre_return_event_ledger,
    load_protocol,
    run_all_pre_return,
    verify_frozen_manifests,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集NBS官方日程并裁决G0/G1；禁止读取事件收益。"
    )
    parser.add_argument(
        "mode",
        choices=("verify", "acquire", "build-ledger", "adjudicate", "run", "replay-check"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        protocol = load_protocol(PROJECT_ROOT)
        protocol_manifest, implementation_manifest = verify_frozen_manifests(
            PROJECT_ROOT
        )
        if arguments.mode == "verify":
            payload = {
                "state": "PASS_NBS_PRE_RETURN_PROTOCOL_AND_IMPLEMENTATION_INTEGRITY",
                "protocol_freeze_id": protocol_manifest["freeze_id"],
                "implementation_freeze_id": implementation_manifest["freeze_id"],
                "event_return_reads": 0,
                "position_impact": 0,
            }
        elif arguments.mode == "acquire":
            payload = acquire_official_schedules(PROJECT_ROOT, protocol)
        elif arguments.mode == "build-ledger":
            _, payload = build_pre_return_event_ledger(PROJECT_ROOT, protocol)
        elif arguments.mode == "adjudicate":
            payload = adjudicate_g0_g1(PROJECT_ROOT, protocol)
        elif arguments.mode == "run":
            payload = run_all_pre_return(PROJECT_ROOT, protocol)
        else:
            report_path = PROJECT_ROOT / Path(protocol["outputs"]["g0_g1"])
            if not report_path.is_file():
                raise FileNotFoundError("缺少首次G0/G1裁决，不能重放")
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            payload = adjudicate_g0_g1(
                PROJECT_ROOT,
                protocol,
                replay_expected_fingerprint=str(previous["decision_fingerprint"]),
            )
        concise = {
            "model_id": MODEL_ID,
            "mode": arguments.mode,
            "state": payload.get("state"),
            "g0_pass": payload.get("g0_pass"),
            "g1_pass": payload.get("g1_pass"),
            "return_evaluation": payload.get("return_evaluation", "NOT_ALLOWED"),
            "event_return_reads": int(payload.get("event_return_reads", 0)),
            "model_training_run": bool(payload.get("model_training_run", False)),
            "position_impact": int(payload.get("position_impact", 0)),
        }
        print(json.dumps(concise, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "mode": arguments.mode,
                    "state": "PROGRAM_FAILED",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1500],
                    "return_evaluation": "NOT_ALLOWED",
                    "event_return_reads": 0,
                    "model_training_run": False,
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
