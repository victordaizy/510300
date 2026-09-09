"""运行已冻结的510300机制图谱V1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from mechanism_atlas_v1 import ContractError, run_atlas  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300机制图谱V1")
    parser.add_argument("--no-write", action="store_true", help="复算但不写产物")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_atlas(write=not arguments.no_write)
    except (ContractError, FileNotFoundError, ValueError, np.linalg.LinAlgError) as exc:
        payload = {
            "status": (
                "NO_VIEW_DATA_OR_PROTOCOL_CONTRACT_FAILED"
                if isinstance(exc, ContractError)
                else "FAILED_MECHANISM_ATLAS_PROGRAM"
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "state_panel_rows": report["state_panel"]["rows"],
        "case_count": report["matching"]["balance"]["case_count"],
        "matched_control_count": report["matching"]["balance"][
            "assigned_control_count"
        ],
        "canonical_case_count": report["chronology"]["canonical_case_count"],
        "return_evaluation": report["adjudication"]["return_evaluation"],
        "net_sharpe": report["adjudication"]["net_sharpe"],
        "target_achieved": report["adjudication"]["target_achieved"],
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
