"""冻结或运行 NBS G2；入口自动检查来源准入。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.nbs_v2_common import ContractError, clean
from research.nbs_v2_g2_engine import freeze, run


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="NBS G2 严格前序预测")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    try:
        print(json.dumps(clean({"freeze": freeze, "run": run}[args.action](ROOT)), ensure_ascii=False, indent=2))
        return 0
    except ContractError as exc:
        print(f"NBS G2 停止：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
