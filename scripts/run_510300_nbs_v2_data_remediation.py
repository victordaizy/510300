"""运行 NBS 数据复核或小范围供应商重取，不读取事件收益。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.nbs_v2_common import clean
from research.nbs_v2_data_remediation import audit, freeze, probe


def main() -> None:
    parser = argparse.ArgumentParser(description="NBS 数据修复取证，保留冻结门槛")
    parser.add_argument("action", choices=["freeze", "audit", "probe"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    result = {"freeze": freeze, "audit": audit, "probe": probe}[args.action](ROOT)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
