"""V1.0.3机械修正：完整绑定报告边界的Python布尔字面量。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import frozen_microstructure_sharpe_audit_v1_0_2 as correction_v1_0_2


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_MANIFEST = (
    ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_3_manifest.json"
)


def validate_correction_manifest() -> dict[str, Any]:
    if not CORRECTION_MANIFEST.exists():
        raise FileNotFoundError("V1.0.3机械修正清单不存在")
    manifest = json.loads(CORRECTION_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("implementation_frozen", False):
        raise ValueError("V1.0.3机械修正尚未冻结")
    original_module = correction_v1_0_2.correction_v1_0_1.base
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in manifest["tracked_files"].items():
        path = ROOT / relative
        actual = "MISSING" if not path.exists() else original_module.sha256_file(path)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"V1.0.3机械修正哈希漂移：{json.dumps(mismatches, ensure_ascii=False)}")
    return manifest


def main() -> int:
    manifest = validate_correction_manifest()
    original_module = correction_v1_0_2.correction_v1_0_1.base
    original_module.true = True
    original_module.false = False
    original_write_outputs = original_module.write_outputs

    def write_outputs_with_correction(
        config: dict[str, Any],
        report: dict[str, Any],
        metrics: pd.DataFrame,
        artifacts: dict[str, pd.DataFrame],
    ) -> None:
        report["correction_v1_0_3"] = {
            "project_id": manifest["project_id"],
            "status": "MECHANICAL_COMPLETE_BOOLEAN_LITERAL_BINDING",
            "manifest_sha256": original_module.sha256_file(CORRECTION_MANIFEST),
            "only_change": "父模块命名空间true = True且false = False",
            "formula_or_parameter_changes": 0,
            "metrics_computed_in_memory_before_correction_freeze": True,
            "metric_values_read_by_operator_before_correction_freeze": False,
        }
        original_write_outputs(config, report, metrics, artifacts)

    original_module.write_outputs = write_outputs_with_correction
    return correction_v1_0_2.main()


if __name__ == "__main__":
    raise SystemExit(main())
