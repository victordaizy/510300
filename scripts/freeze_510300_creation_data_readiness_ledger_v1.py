from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.creation_data_readiness_ledger_v1 import freeze_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结510300一级市场数据成熟度账V1")
    parser.add_argument(
        "--config",
        default="config/510300_creation_data_readiness_ledger_v1.yaml",
        help="协议配置路径（相对项目根目录）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze_bundle(ROOT, ROOT / args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

