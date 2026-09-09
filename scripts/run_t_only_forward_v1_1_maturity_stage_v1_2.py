"""运行冻结 T-only V1.1 计算并发布 V1.2 成熟度公开状态。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t_only_forward_v1_1 import main as run_frozen_v1_1
from scripts.t_only_forward_maturity_only_v1_2 import (
    retire_legacy_run_status,
    sanitize_existing_outputs,
)


def run_stage(*, sanitize_existing_only: bool) -> dict[str, object]:
    if not sanitize_existing_only:
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            exit_code = int(run_frozen_v1_1())
        if exit_code != 0:
            raise RuntimeError(f"冻结 V1.1 计算失败，退出码：{exit_code}")

    snapshot = sanitize_existing_outputs(
        run_completeness={
            "status": "STAGE_COMPLETE_PENDING_DAILY_RUN_AUDIT",
            "complete": None,
            "run_id": None,
        }
    )
    retire_legacy_run_status(snapshot)
    return {
        "status": snapshot["status"],
        "current_view": snapshot["current_view"],
        "as_of_market_date": snapshot["as_of_market_date"],
        "new_trading_days": snapshot["new_trading_days"],
        "closed_cycles": snapshot["closed_cycles"],
        "data_completeness": snapshot["completeness"]["data"]["status"],
        "ledger_completeness": snapshot["completeness"]["ledger"]["status"],
        "evaluation_status": snapshot["evaluation_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 T-only V1.1 并仅发布成熟度与完整性"
    )
    parser.add_argument("--sanitize-existing-only", action="store_true")
    arguments = parser.parse_args()
    try:
        result = run_stage(sanitize_existing_only=arguments.sanitize_existing_only)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "current_view": "NO_VIEW_OPERATIONAL_FAILURE",
                    "failure_class": "PROGRAM_FAILED",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
