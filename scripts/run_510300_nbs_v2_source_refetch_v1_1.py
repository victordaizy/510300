"""冻结、重取并审核 NBS 原始分钟数据；不越过 G0 读取收益。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.nbs_v2_common import clean
from research.nbs_v2_source_refetch_v1_1 import acquire, audit, freeze, probe


def main() -> None:
    parser = argparse.ArgumentParser(description="使用新授权临时凭据完成全部异常月份的来源重取")
    parser.add_argument("action", choices=["freeze", "probe", "acquire", "audit"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    result = {"freeze": freeze, "probe": probe, "acquire": acquire, "audit": audit}[args.action](ROOT)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
