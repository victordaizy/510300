"""运行V3_FORWARD_1日终前瞻记录、到期解析和固定频率监控。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v3_forward_validation import run_forward_cycle
from scripts.refresh_v3_forward_inputs import refresh_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行V3_FORWARD_1日终前瞻周期")
    parser.add_argument(
        "--date",
        help="仅用于当日运行或受控测试的信号日，格式YYYY-MM-DD；禁止历史回填",
    )
    parser.add_argument(
        "--force-report",
        action="store_true",
        help="强制生成监控报告；正常日运行每20个新信号自动生成",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="仅供测试：跳过联网刷新；自动日运行不得使用",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    signal_date = pd.Timestamp(args.date) if args.date else None
    if not args.skip_refresh:
        refresh = refresh_all(signal_date)
        if refresh["status"] != "SUCCESS":
            print(json.dumps(refresh, ensure_ascii=False, indent=2, default=str))
            return 0
    result = run_forward_cycle(
        signal_date=signal_date,
        generated_at=now,
        enforce_same_local_date=True,
        force_report=args.force_report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
