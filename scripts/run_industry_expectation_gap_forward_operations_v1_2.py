"""运行只读内容寻址输入的行业预期差 V1.2 状态链。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_industry_expectation_gap_forward_operations_v1 as base
from scripts.industry_outcome_snapshot_v1_2 import (
    ROOT,
    load_verified_snapshot_manifest,
)


CONFIG_PATH = ROOT / "config" / "industry_expectation_gap_forward_operations_v1_2.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="行业预期差 V1.2 前瞻运行监控")
    parser.add_argument("--as-of", help="可复现运行时点")
    parser.add_argument("--evaluate-if-ready", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("行业 V1.2 运行配置顶层必须是对象")
    snapshot, by_role = load_verified_snapshot_manifest(config)
    runtime_config = copy.deepcopy(config)
    for role in ("constituent_total_return_daily", "etf_total_return_daily"):
        runtime_config["inputs"][role] = by_role[role].relative_to(ROOT).as_posix()
    report = base.build_status(
        runtime_config,
        base._parse_as_of(args.as_of),
        args.evaluate_if_ready,
    )
    report["operations_version"] = config["version"]
    report["input_snapshot"] = {
        "snapshot_id": snapshot["snapshot_id"],
        "immutable_manifest_path": snapshot["immutable_manifest_path"],
        "shared_latest_consumed_by_evaluation": False,
        "records": snapshot["records"],
    }
    base._atomic_write(
        ROOT / config["outputs"]["status_json"],
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    base._atomic_write(
        ROOT / config["outputs"]["status_markdown"],
        base._markdown(report),
    )
    print(f"行业预期差 V1.2 状态：{report['status']}")
    print(f"输入快照：{snapshot['snapshot_id']}")
    return 1 if report["operations_health"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
