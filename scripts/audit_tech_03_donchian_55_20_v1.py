"""只运行TECH_03数据闸门，不计算任何收益。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tech_03_donchian_55_20_v1 import audit_and_load_inputs, load_config


OUTPUT = ROOT / "reports" / "data_quality" / "tech_03_donchian_55_20_v1_data_gate.json"


def main() -> int:
    config = load_config()
    _, _, audit = audit_and_load_inputs(ROOT, config)
    audit["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    audit["position_mapping_enabled"] = False
    audit["order_generation_enabled"] = False
    audit["broker_connection_enabled"] = False
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
