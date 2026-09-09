"""固定专家双影子并集发现V1.0.1机械日期边界修正版。

V1在任何绩效输出产生前，因把2025-12-31执行日误当作还需2025-12-31
新信号而失败。本修正版只将专家信号截点设为2025-12-30；该信号仍按原合同
在2025-12-31开盘执行。六个策略、双影子门、成本、基准和目标均不改变。
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import fixed_expert_dual_shadow_union_discovery_v1 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_FIXED_EXPERT_DUAL_SHADOW_UNION_DISCOVERY_V1_0_1"
CORRECTION_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1_0_1_correction.json"
)
CORRECTION_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1_0_1_manifest.json"
)
CORRECTION_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_fixed_expert_dual_shadow_union_discovery_v1_0_1_receipt.json"
)

_ORIGINAL_LOAD_CONTRACT = base._load_contract
_ORIGINAL_LOAD_EXPERT_PANEL = base._load_expert_panel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".v1_0_1.tmp")
    if temporary.exists():
        raise FileExistsError(f"修正版临时文件已存在：{temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_correction() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(CORRECTION_CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(CORRECTION_MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(CORRECTION_RECEIPT.read_text(encoding="utf-8"))
    if any(item.get("project_id") != PROJECT_ID for item in [config, manifest, receipt]):
        raise ValueError("V1.0.1修正合同项目编号不匹配")
    if receipt.get("status") != "MECHANICAL_CORRECTION_FROZEN_BEFORE_FIRST_CORRECTED_RUN":
        raise ValueError("V1.0.1修正冻结收据状态无效")
    if receipt.get("manifest_sha256") != _sha256(CORRECTION_MANIFEST):
        raise ValueError("V1.0.1修正清单哈希与冻结收据不匹配")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in manifest["tracked_files"].items():
        path = PROJECT_ROOT / relative
        actual = _sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"V1.0.1修正合同哈希漂移：{mismatches}")
    return config, manifest, receipt


def _corrected_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _verify_correction()
    return _ORIGINAL_LOAD_CONTRACT()


def _corrected_expert_panel(
    config: dict[str, Any], market: Any
) -> tuple[Any, dict[str, Any]]:
    correction, _, _ = _verify_correction()
    local_config = copy.deepcopy(config)
    expert_signal_end = correction["correction"]["expert_signal_end"]
    execution_end = correction["correction"]["execution_evaluation_end"]
    if config["periods"]["common_discovery_end"] != execution_end:
        raise ValueError("V1.0.1执行评估终点与基础合同不一致")
    local_config["periods"]["common_discovery_end"] = expert_signal_end
    panel, audit = _ORIGINAL_LOAD_EXPERT_PANEL(local_config, market)
    audit["correction_version"] = "V1.0.1"
    audit["expert_signal_end"] = expert_signal_end
    audit["execution_evaluation_end"] = execution_end
    audit["missing_value_fill"] = False
    return panel, audit


def main() -> int:
    correction, manifest, receipt = _verify_correction()
    base._load_contract = _corrected_contract
    base._load_expert_panel = _corrected_expert_panel
    base.__file__ = str(Path(__file__).resolve())
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exit_code = base.main()
    payload = json.loads(base.OUTPUT_REPORT.read_text(encoding="utf-8"))
    payload["correction_v1_0_1"] = {
        "project_id": PROJECT_ID,
        "status": correction["status"],
        "config": _relative(CORRECTION_CONFIG),
        "config_sha256": _sha256(CORRECTION_CONFIG),
        "manifest": _relative(CORRECTION_MANIFEST),
        "manifest_sha256": _sha256(CORRECTION_MANIFEST),
        "receipt": _relative(CORRECTION_RECEIPT),
        "receipt_sha256": _sha256(CORRECTION_RECEIPT),
        "implementation": _relative(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "tracked_file_count": int(len(manifest["tracked_files"])),
        "frozen_at_asia_shanghai": receipt["frozen_at_asia_shanghai"],
        "expert_signal_end": correction["correction"]["expert_signal_end"],
        "execution_evaluation_end": correction["correction"][
            "execution_evaluation_end"
        ],
        "missing_value_fill": False,
        "formula_or_threshold_changes": 0,
    }
    payload["freeze"]["implementation_sha256"] = _sha256(Path(__file__).resolve())
    payload["freeze"]["base_v1_implementation_sha256"] = _sha256(
        PROJECT_ROOT / "research" / "fixed_expert_dual_shadow_union_discovery_v1.py"
    )
    _atomic_json(payload, base.OUTPUT_REPORT)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
