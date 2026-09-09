from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.creation_data_readiness_ledger_v1 import (
    append_forward_day,
    initialize_readiness_ledger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="维护510300一级市场数据成熟度账V1")
    parser.add_argument("action", choices=("initialize", "append-day"))
    parser.add_argument(
        "--config",
        default="config/510300_creation_data_readiness_ledger_v1.yaml",
        help="协议配置路径（相对项目根目录）",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="冻结清单内容摘要；不匹配时停止",
    )
    parser.add_argument("--payload", help="append-day所需的JSON回执路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    if args.action == "initialize":
        result = initialize_readiness_ledger(ROOT, config_path, args.expected_manifest_sha256)
    else:
        if not args.payload:
            raise SystemExit("append-day必须提供--payload")
        payload_path = Path(args.payload)
        if not payload_path.is_absolute():
            payload_path = ROOT / payload_path
        result = append_forward_day(
            ROOT,
            config_path,
            args.expected_manifest_sha256,
            payload_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

