"""运行DAILY_01只读数据闸门，不计算特征分布或未来收益。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_01_overnight_absorption_v1 import audit_and_load_inputs, load_config


OUTPUT = ROOT / "reports" / "data_quality" / "daily_01_overnight_absorption_v1_data_gate.json"


def main() -> int:
    _, _, audit = audit_and_load_inputs(ROOT, load_config())
    audit["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
