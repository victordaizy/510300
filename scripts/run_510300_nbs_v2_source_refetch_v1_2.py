"""续取同供应商全部异常月份，失败时输出清晰状态。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.nbs_v2_common import ContractError, clean
from research.nbs_v2_source_refetch_v1_2 import acquire, audit, freeze


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="NBS 异常月份来源续取")
    parser.add_argument("action", choices=["freeze", "acquire", "audit"])
    args = parser.parse_args()
    try:
        result = {"freeze": freeze, "acquire": acquire, "audit": audit}[args.action](ROOT)
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))
        return 0
    except ContractError as exc:
        print(f"来源阶段停止：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
