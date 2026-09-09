from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.derivative_pressure_release_v2_aggregate_oi import run_one_shot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行且仅执行一次510300衍生品压力释放V2聚合OI研究")
    parser.add_argument(
        "--config",
        default="config/510300_derivative_pressure_release_v2_aggregate_oi.yaml",
        help="协议配置路径（相对项目根目录）",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="冻结清单内容摘要；不匹配时在读取研究结果前停止",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = run_one_shot(ROOT, ROOT / args.config, args.expected_manifest_sha256)
    print(json.dumps(output["status_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

