"""冻结V1.0.1技术修正的正式入口。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import price_path_dsv5_risk_budget_policy_v1 as base
from research.price_path_dsv5_risk_budget_policy_v1_0_1 import activate


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结DSV5政策V1.0.1阶段入口")
    parser.add_argument("stage", choices=["register-forward", "gates", "history", "shadow-tick"])
    args = parser.parse_args()
    activate()
    methods = {"register-forward": base.register_forward, "gates": base.run_gates,
               "history": base.run_history, "shadow-tick": base.shadow_tick}
    try:
        result = methods[args.stage](ROOT)
    except (base.ContractError, FileExistsError, FileNotFoundError) as exc:
        print(json.dumps({"状态": "停止执行", "原因": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(base.clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
