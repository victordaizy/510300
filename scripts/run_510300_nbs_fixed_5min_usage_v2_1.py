"""冻结并执行已获明确授权的 NBS 五分钟用途合同。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.nbs_v2_common import ContractError, clean
from research.nbs_fixed_5min_usage_v2_1 import freeze, run_g2, run_g3, source_admission


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="NBS 五分钟用途准入及逐级执行")
    parser.add_argument("action", choices=["freeze", "source", "g2", "g3"])
    args = parser.parse_args()
    try:
        result = {"freeze": freeze, "source": source_admission, "g2": run_g2, "g3": run_g3}[args.action](ROOT)
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))
        return 0
    except ContractError as exc:
        print(f"冻结合同停止：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
