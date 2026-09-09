"""价格路径DSV5政策：阶段化入口，支持Windows直接文件运行。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.price_path_dsv5_risk_budget_policy_v1 import (
    ContractError, clean_json, register_forward, run_gates, run_history, shadow_tick,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行冻结DSV5政策指定阶段")
    parser.add_argument("stage", choices=["register-forward", "gates", "history", "shadow-tick"])
    args = parser.parse_args()
    methods = {"register-forward": register_forward, "gates": run_gates,
               "history": run_history, "shadow-tick": shadow_tick}
    try:
        result = methods[args.stage](ROOT)
    except (ContractError, FileExistsError, FileNotFoundError) as exc:
        print(json.dumps({"状态": "停止执行", "原因": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
