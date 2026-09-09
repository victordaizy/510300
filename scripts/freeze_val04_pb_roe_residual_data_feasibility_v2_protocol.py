"""冻结 VAL-04 申万点时行业 V2 数据协议。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    load_config,
    sha256_file,
    verify_input_hashes,
)


CONFIG_FILE = ROOT / "config" / "val04_pb_roe_residual_data_feasibility_v2.yaml"
FROZEN_FILES = (
    "config/val04_pb_roe_residual_data_feasibility_v2.yaml",
    "docs/VAL04_PB_ROE_RESIDUAL_DATA_FEASIBILITY_V2_ADDENDUM.md",
    "research/val04_pb_roe_residual_data_feasibility_v1.py",
    "scripts/freeze_val04_pb_roe_residual_data_feasibility_v2_protocol.py",
    "scripts/audit_val04_pb_roe_residual_data_feasibility_v2.py",
    "tests/test_val04_pb_roe_residual_data_feasibility_v1.py",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def validate_supplier_report(config: dict) -> dict:
    contract = config["data_contracts"]["sw_industry_acquisition_report"]
    report = json.loads((ROOT / contract["file"]).read_text(encoding="utf-8"))
    phase = report.get("phases", {}).get("industry", {})
    rules = config["industry_source_acceptance"]
    checks = {
        "overall_status": report.get("status") == rules["required_acquisition_status"],
        "industry_phase_status": phase.get("status")
        == rules["required_industry_phase_status"],
        "member_day_coverage": float(phase.get("coverage_ratio", 0.0))
        >= float(rules["minimum_supplier_member_day_coverage"]),
        "overlap_ratio": float(phase.get("overlap_ratio", 1.0))
        <= float(rules["maximum_supplier_overlap_ratio"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"申万点时行业供应商报告未通过：{checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "coverage_ratio": phase["coverage_ratio"],
        "overlap_ratio": phase["overlap_ratio"],
    }


def main() -> int:
    config = load_config(CONFIG_FILE)
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    input_audit = verify_input_hashes(config)
    if input_audit["status"] != "PASS":
        raise RuntimeError("V2 输入哈希漂移，禁止冻结")
    supplier_audit = validate_supplier_report(config)
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_DATA_FEASIBILITY_PROTOCOL_NO_RETURN_READING",
        "frozen_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "parent_v1_status": config["protocol"]["parent_result"],
        "revision_reason": config["protocol"]["revision_reason"],
        "frozen_files": {
            relative: {
                "sha256": sha256_file(ROOT / relative),
                "size_bytes": (ROOT / relative).stat().st_size,
            }
            for relative in FROZEN_FILES
        },
        "input_hash_audit": input_audit,
        "supplier_audit": supplier_audit,
        "frozen_candidate_definition": config["frozen_candidate_definition"],
        "quality_gates": config["quality_gates"],
        "governance": config["governance"],
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    output = ROOT / config["artifacts"]["protocol_manifest"]
    atomic_text(output, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "状态": manifest["status"],
                "协议清单": str(output),
                "内容哈希": manifest["manifest_content_sha256"],
                "供应商审计": supplier_audit,
                "V1文件已修改": False,
                "未来收益读取": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
