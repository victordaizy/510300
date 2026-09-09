"""执行尾部风险门控数据审计；不会读取未来收益或运行回测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.return_tail_data_feasibility import run_feasibility_audit


def main() -> int:
    report = run_feasibility_audit(ROOT)
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "branch_status": report["branch_status"],
        "hypotheses_allowed_for_return_test": report[
            "hypotheses_allowed_for_return_test"
        ],
        "hypotheses_terminated": report["hypotheses_terminated"],
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

