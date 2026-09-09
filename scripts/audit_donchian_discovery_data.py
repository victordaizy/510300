"""审计唐奇安发现协议输入；只产生数据质量证据，不计算收益。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.donchian_discovery_data_audit import audit_all, load_registry


DEFAULT_OUTPUT = ROOT / "reports" / "data_quality" / "donchian_discovery_v1_data_gate.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="审计510300唐奇安发现协议数据闸门")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON证据输出路径",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="只在终端打印，不写报告文件",
    )
    args = parser.parse_args()
    result = audit_all(ROOT, load_registry())
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if not args.no_write:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"数据闸门报告：{output}")
    print(payload)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

