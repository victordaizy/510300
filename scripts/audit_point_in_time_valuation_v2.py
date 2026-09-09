"""运行沪深300点时估值V2可重建性审计；不计算收益或IC。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.point_in_time_valuation_v2_audit import load_config, run_audit, write_artifacts


def main() -> int:
    config = load_config()
    report, coverage = run_audit(config)
    write_artifacts(report, coverage, config)
    summary = {
        "status": report["overall_status"],
        "formal_valuation_run_status": report["formal_valuation_run_status"],
        "snapshot_count": int(len(coverage)),
        "five_year": report["history_gates"]["five_year"],
        "seven_year": report["history_gates"]["seven_year"],
        "r5_engineering_replay": report["r5_replay"]["engineering_replay_status"],
        "report_json": config["artifacts"]["report_json"],
        "report_markdown": config["artifacts"]["report_markdown"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
