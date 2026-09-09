"""运行价格补采后的点时估值V2 V1.1审计。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.point_in_time_valuation_v2_post_acquisition import (
    load_config,
    run_post_acquisition_audit,
    write_artifacts,
)


def main() -> int:
    config = load_config()
    report, coverage = run_post_acquisition_audit(config)
    write_artifacts(report, coverage, config)
    print(
        json.dumps(
            {
                "status": report["overall_status"],
                "branch_status": report["branch_status"],
                "five_year": report["history_gates"]["five_year"],
                "artifacts": report["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
