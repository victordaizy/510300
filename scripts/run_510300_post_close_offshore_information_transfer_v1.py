from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.post_close_offshore_information_transfer_v1 import (
    initialize_forward_ledgers,
    mature_observation,
    record_signal_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="维护510300盘后离岸信息传导V1纯前向三账")
    parser.add_argument(
        "action",
        choices=("initialize", "record-window", "mature"),
        help="初始化空账、记录当日信号窗口、或在下一交易日成熟观测",
    )
    parser.add_argument(
        "--config",
        default="config/510300_post_close_offshore_information_transfer_v1.yaml",
        help="协议配置路径（相对项目根目录）",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="冻结清单内容摘要；不匹配时停止",
    )
    parser.add_argument("--payload", help="record-window或mature所需的JSON回执路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    if args.action == "initialize":
        result = initialize_forward_ledgers(ROOT, config_path, args.expected_manifest_sha256)
    else:
        if not args.payload:
            raise SystemExit("record-window和mature必须提供--payload")
        payload_path = Path(args.payload)
        if not payload_path.is_absolute():
            payload_path = ROOT / payload_path
        if args.action == "record-window":
            result = record_signal_window(
                ROOT,
                config_path,
                args.expected_manifest_sha256,
                payload_path,
            )
        else:
            result = mature_observation(
                ROOT,
                config_path,
                args.expected_manifest_sha256,
                payload_path,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

