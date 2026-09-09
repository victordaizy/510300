"""冻结V1.0.1分页机械修正及其全部既有原始响应哈希。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_REVERSE_VALIDATION_V1_0_1"
CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.yaml"
)
PRIOR_CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1.yaml"
)
MANIFEST_FILE = CONFIG_FILE.with_name(
    "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1_manifest.json"
)
RECEIPT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1_freeze_receipt.json"
)
PRIOR_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_validation"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1"
    / "snapshots"
    / "20260828T124544_0800"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _write_new_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"冻结证据已存在，拒绝覆盖：{path}")
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(content)


def _assert_unchanged_contract(
    current: dict[str, Any], prior: dict[str, Any]
) -> list[str]:
    exact_sections = [
        "scope",
        "dates",
        "candidate_contract",
        "inputs_at_freeze",
        "objective",
        "seam_crosscheck",
        "governance",
    ]
    for section in exact_sections:
        if current[section] != prior[section]:
            raise ValueError(f"V1.0.1非机械地修改了冻结章节：{section}")
    for source in ["fund_share", "margin"]:
        if current["acquisition"][source] != prior["acquisition"][source]:
            raise ValueError(f"V1.0.1修改了非目标采集协议：{source}")
    for key in ["primary", "secondary", "maximum_unit_nav_absolute_difference"]:
        if current["acquisition"]["nav"][key] != prior["acquisition"]["nav"][key]:
            raise ValueError(f"V1.0.1修改了净值采集核心字段：{key}")
    return exact_sections


def main() -> int:
    if MANIFEST_FILE.exists() or RECEIPT_FILE.exists():
        raise FileExistsError("V1.0.1冻结清单或收据已存在，拒绝重复冻结")
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    prior_config = yaml.safe_load(PRIOR_CONFIG_FILE.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ValueError("V1.0.1配置项目编号不匹配")
    unchanged_sections = _assert_unchanged_contract(config, prior_config)

    prior_status_path = PROJECT_ROOT / config["mechanical_correction"][
        "prior_failure_status"
    ]
    prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
    if (
        prior_status.get("status") != "PARTIAL_SUCCESS_RESUMABLE"
        or prior_status.get("error")
        != "nav_sina记录数2238不等于total=2227"
        or int(prior_status.get("raw_response_files_preserved", -1)) != 2346
    ):
        raise ValueError("V1失败收据与机械修正声明不一致")

    tracked_relatives = [
        "config/510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.yaml",
        "scripts/collect_510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.py",
        "research/evaluate_510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.py",
        "scripts/freeze_510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.py",
        "config/510300_etf_microstructure_dual_shadow_reverse_validation_v1.yaml",
        "config/510300_etf_microstructure_dual_shadow_reverse_validation_v1_manifest.json",
        "scripts/collect_510300_etf_microstructure_dual_shadow_reverse_validation_v1.py",
        "research/evaluate_510300_etf_microstructure_dual_shadow_reverse_validation_v1.py",
        "config/510300_etf_microstructure_dual_shadow_v1.yaml",
        "config/510300_etf_microstructure_dual_shadow_v1_manifest.json",
        "research/validate_etf_microstructure_dual_shadow_v1_frozen.py",
        "research/binary_state_feasibility_v1.py",
        "research/global_liquidity_regime_5d_discovery_v0.py",
    ]
    tracked_files: dict[str, str] = {}
    for relative in tracked_relatives:
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"冻结跟踪文件不存在：{relative}")
        tracked_files[relative] = _sha256(path)

    historical_files: dict[str, str] = {}
    for relative in config["inputs_at_freeze"].values():
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"冻结输入文件不存在：{relative}")
        historical_files[relative] = _sha256(path)
    historical_files[_relative(prior_status_path)] = _sha256(prior_status_path)
    raw_files = sorted(
        path
        for path in (PRIOR_RUN_DIR / "raw_responses").rglob("*.json")
        if path.is_file()
    )
    if len(raw_files) != 2346:
        raise ValueError(f"V1原始响应文件数不是2346：{len(raw_files)}")
    for path in raw_files:
        historical_files[_relative(path)] = _sha256(path)

    frozen_at = datetime.now(TIMEZONE).isoformat()
    manifest = {
        "project_id": PROJECT_ID,
        "state": "FROZEN_MECHANICAL_CORRECTION_BEFORE_EXTERNAL_RESULT_COMPUTATION",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": frozen_at,
        "candidate_name": config["protocol"]["candidate_name"],
        "evidence_class": config["protocol"]["evidence_class"],
        "prefreeze_exposure": config["protocol"]["prefreeze_exposure"],
        "candidate_feature_or_return_result_computed_before_freeze": False,
        "mechanical_correction": config["mechanical_correction"],
        "unchanged_contract_sections_verified": unchanged_sections,
        "tracked_files": tracked_files,
        "historical_input_files_at_freeze": historical_files,
        "prior_raw_response_file_count": int(len(raw_files)),
        "periods": config["dates"]["periods"],
        "objective": config["objective"],
        "governance": config["governance"],
    }
    _write_new_json(manifest, MANIFEST_FILE)
    receipt = {
        "project_id": PROJECT_ID,
        "status": "FROZEN_MECHANICAL_CORRECTION_PASS",
        "frozen_at_asia_shanghai": frozen_at,
        "manifest": _relative(MANIFEST_FILE),
        "manifest_sha256": _sha256(MANIFEST_FILE),
        "config": _relative(CONFIG_FILE),
        "config_sha256": _sha256(CONFIG_FILE),
        "tracked_file_count": int(len(tracked_files)),
        "historical_input_file_count": int(len(historical_files)),
        "prior_raw_response_file_count": int(len(raw_files)),
        "external_candidate_results_present_at_freeze": False,
        "unchanged_contract_sections_verified": unchanged_sections,
        "permitted_change": config["mechanical_correction"]["permitted_change"],
        "next_entrypoint": (
            ".venv\\Scripts\\python.exe scripts\\"
            "collect_510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1.py "
            "--run-id <immutable_run_id>"
        ),
        "boundaries": {
            "research_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _write_new_json(receipt, RECEIPT_FILE)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
