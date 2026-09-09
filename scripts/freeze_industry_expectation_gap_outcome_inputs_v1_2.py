"""生成行业预期差 V1.2 内容寻址结果输入快照。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.industry_outcome_snapshot_v1_2 import ROOT, freeze_inputs


DEFAULT_CONFIG = (
    ROOT / "config" / "industry_expectation_gap_forward_operations_v1_2.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结行业预期差 V1.2 结果输入")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("行业输入快照配置顶层必须是对象")
    payload = freeze_inputs(
        config,
        datetime.now(ZoneInfo(str(config["timezone"]))),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
