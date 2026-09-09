"""只运行模型层重建数据闸门；本脚本不计算任何策略收益。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_layer_rebuild_v1 import (
    audit_and_load_inputs,
    build_valuation_engineering_audit,
    load_config,
)


def main() -> int:
    config = load_config()
    datasets, audit = audit_and_load_inputs(ROOT, config)
    _, valuation_engineering = build_valuation_engineering_audit(datasets, config)
    payload = {**audit, "valuation_engineering_only": valuation_engineering}
    output = ROOT / config["artifacts"]["data_gate_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
