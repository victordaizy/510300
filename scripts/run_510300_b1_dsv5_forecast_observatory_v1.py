"""只读风险观察器入口，生产运行始终使用真实本机时钟。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.b1_dsv5_forecast_observatory_v1 import register, tick


def main() -> None:
    parser = argparse.ArgumentParser(description="登记或运行B1 DSV5只读观察器")
    parser.add_argument("phase", choices=["register", "tick", "status"])
    args = parser.parse_args()
    result = register(ROOT) if args.phase == "register" else tick(ROOT, allow_fetch=args.phase == "tick")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
