from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.episodic_alpha_library_v1 import run_library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行已冻结的510300短周期情景Alpha库V1发现性研究")
    parser.add_argument(
        "--config",
        default="config/510300_episodic_alpha_library_v1.yaml",
        help="协议配置路径（相对项目根目录）",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="冻结清单的内容摘要；不匹配时在读取收益前停止",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_library(ROOT, ROOT / args.config, args.expected_manifest_sha256)
    print(json.dumps(result["status_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
