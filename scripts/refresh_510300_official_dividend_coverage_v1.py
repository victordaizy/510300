"""核验两处官方记录后更新分红覆盖；不生成预测或抓取价格。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import official_dividend_coverage_refresh_v1 as source


def main() -> int:
    parser = argparse.ArgumentParser(description="维护510300官方分红覆盖凭证")
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--if-origin", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = source.verify_runtime(ROOT, args.expected_manifest_sha256)
    if args.verify_only:
        print(json.dumps({"status": "PASS_OFFICIAL_DIVIDEND_MAINTENANCE_RUNTIME", "source_fetches": 0, "coverage_updated": False}, ensure_ascii=False, indent=2))
        return 0
    if args.if_origin and not source.is_origin_window(ROOT, config, datetime.now(source.TZ)):
        print(json.dumps({"status": "NOT_ORIGIN_WINDOW_NO_FETCH_NO_UPDATE", "source_fetches": 0, "coverage_updated": False}, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(source.refresh(ROOT, config, args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
