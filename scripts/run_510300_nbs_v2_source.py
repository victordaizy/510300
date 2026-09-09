"""NBS V2 无收益来源准入入口。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.nbs_v2_common import clean
from research.stk_mins_source_admission_v2 import freeze, run


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结或运行NBS V2无收益来源合同")
    parser.add_argument("phase", choices=["freeze", "source"])
    args = parser.parse_args()
    result = freeze(ROOT) if args.phase == "freeze" else run(ROOT)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
