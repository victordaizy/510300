"""冻结十因子V1唯一允许的报告字段重名修复。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_csi300_ten_factor_alpha_v1 import (
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
)
from scripts.run_csi300_ten_factor_alpha_v1_reporting_fix import (
    corrected_prediction_diagnostics,
)


FIX_FILE = ROOT / "scripts" / "run_csi300_ten_factor_alpha_v1_reporting_fix.py"
FIX_MANIFEST = ROOT / "config" / "csi300_ten_factor_alpha_v1_reporting_fix_manifest.json"


def main() -> int:
    original = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if original.get("state") != "FROZEN_BEFORE_OUTCOME_READ":
        raise RuntimeError("原始十因子冻结清单状态错误")
    changed = [
        relative
        for relative in FROZEN_FILES
        if not (ROOT / relative).exists()
        or sha256(ROOT / relative) != original["frozen_files"][relative]
    ]
    if changed:
        raise RuntimeError(f"原始冻结文件已变化：{changed}")
    sample_predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-02")] * 3,
            "con_code": ["A", "B", "C"],
            "model_output": ["PREDICTION_AUDIT_ONLY"] * 3,
            "predicted_excess_log_return_10d": [0.03, 0.02, 0.01],
            "future_excess_log_return_10d": [0.02, 0.01, -0.01],
            "prediction_rank": [1, 2, 3],
        }
    )
    diagnostic = corrected_prediction_diagnostics(sample_predictions, pd.DataFrame())
    if diagnostic["matured_evaluation_rows"] != 3:
        raise AssertionError("报告修复自测失败")
    manifest = {
        "project_id": original["project_id"],
        "state": "REPORTING_ONLY_FIX_FROZEN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "original_manifest_sha256": sha256(MANIFEST_FILE),
        "original_frozen_files_unchanged": True,
        "permitted_change": "预测表已含future_excess_log_return_10d时禁止重复合并，仅修复报告诊断",
        "factor_change": False,
        "model_change": False,
        "target_change": False,
        "trade_change": False,
        "reporting_fix_file": sha256(FIX_FILE),
        "self_test": diagnostic,
    }
    FIX_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"状态": manifest["state"], "原冻结文件未变": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
