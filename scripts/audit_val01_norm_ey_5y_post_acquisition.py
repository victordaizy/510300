"""运行 VAL01_NORM_EY_5Y 财务扩展后的数据闸门。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_norm_ey_5y_financial_extension import load_config
from research.val01_norm_ey_5y_post_acquisition_audit import (
    run_post_acquisition_audit,
    write_artifacts,
)


def main() -> int:
    config = load_config()
    report, coverage, normalized_panel = run_post_acquisition_audit(config)
    write_artifacts(report, coverage, normalized_panel, config)
    print(
        json.dumps(
            {
                "status": report["overall_status"],
                "branch_status": report["branch_status"],
                "five_year_gate": report["history_gates"]["five_year"],
                "normalized_metric_panel": report["normalized_metric_panel"],
                "independent_cross_check": report["independent_cross_check"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["overall_status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
