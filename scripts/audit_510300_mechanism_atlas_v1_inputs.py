"""审计510300机制图谱V1输入；不读取新的机制结果。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from mechanism_atlas_v1 import (  # noqa: E402
    ContractError,
    audit_inputs,
    load_config,
    load_inputs,
    write_input_audit,
)


def main() -> int:
    try:
        config = load_config()
        inputs = load_inputs(config)
        audit = audit_inputs(config, inputs)
        path = write_input_audit(config, audit)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        payload = {
            "status": "NO_VIEW_DATA_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = {
        "project_id": audit["project_id"],
        "status": audit["status"],
        "passed": audit["passed"],
        "common_window": audit["common_window"],
        "anchor_events": audit["anchor_events"],
        "input_audit": path.relative_to(ROOT).as_posix(),
        "strategy_backtest_run": False,
        "net_sharpe": "NOT_COMPUTED",
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

