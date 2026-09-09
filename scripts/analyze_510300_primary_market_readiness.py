"""生成510300 PCF/IOPV前向样本成熟度报告。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.primary_market_forward_readiness import evaluate_forward_readiness


CONFIG_FILE = PROJECT_ROOT / "config" / "primary_market_forward.yaml"


def _resolve(value: str) -> Path:
    return PROJECT_ROOT / Path(value)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    pcf_file = _resolve(config["outputs"]["pcf_daily_file"])
    iopv_file = _resolve(config["outputs"]["iopv_snapshot_file"])
    if not pcf_file.exists() or not iopv_file.exists():
        print("缺少PCF或IOPV前向数据，暂时无法生成成熟度报告", file=sys.stderr)
        return 1

    readiness = evaluate_forward_readiness(
        pd.read_parquet(pcf_file),
        pd.read_parquet(iopv_file),
        minimum_full_coverage_days=int(config["readiness"]["minimum_full_coverage_days"]),
        recommended_full_coverage_days=int(
            config["readiness"]["recommended_full_coverage_days"]
        ),
        minimum_snapshots_per_day=int(config["readiness"]["minimum_snapshots_per_day"]),
        first_unseen_evaluation_full_coverage_days=int(
            config["readiness"]["first_unseen_evaluation_full_coverage_days"]
        ),
        replication_full_coverage_days=int(
            config["readiness"]["replication_full_coverage_days"]
        ),
        timezone=str(config["collector"]["timezone"]),
        maximum_iopv_staleness_seconds=float(
            config["quality"]["maximum_clock_delay_seconds_during_session"]
        ),
        maximum_negative_clock_skew_seconds=float(
            config["quality"]["maximum_negative_clock_skew_seconds"]
        ),
    )
    payload = {
        "generated_at": datetime.now(ZoneInfo(config["collector"]["timezone"])).isoformat(),
        "collector_id": config["collector"]["collector_id"],
        **readiness,
        "governance": {
            "automatic_position_mapping_authorized": False,
            "automatic_ordering_authorized": False,
        },
    }
    output_file = _resolve(config["outputs"]["readiness_report_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
