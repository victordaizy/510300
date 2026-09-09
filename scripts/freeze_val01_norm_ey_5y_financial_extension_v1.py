"""冻结 VAL01_NORM_EY_5Y 财务扩展与可重建性数据闸门结果。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_norm_ey_5y_financial_extension import (
    load_config,
    sha256_file,
    verify_frozen_inputs,
)


IMPLEMENTATION_FILES = (
    "config/val01_norm_ey_5y_financial_extension_v1.yaml",
    "docs/VAL01_NORM_EY_5Y_FINANCIAL_EXTENSION_V1_PROTOCOL.md",
    "research/val01_norm_ey_5y_financial_extension.py",
    "research/val01_norm_ey_5y_post_acquisition_audit.py",
    "scripts/download_val01_norm_ey_5y_financial_extension.py",
    "scripts/audit_val01_norm_ey_5y_post_acquisition.py",
    "scripts/freeze_val01_norm_ey_5y_financial_extension_protocol.py",
    "scripts/freeze_val01_norm_ey_5y_financial_extension_v1.py",
    "tests/test_val01_norm_ey_5y_financial_extension.py",
    "tests/test_val01_norm_ey_5y_post_acquisition_audit.py",
)

DATA_AND_EVIDENCE_FILES = (
    "data/raw/fundamentals/csi300_financials_point_in_time_history_extension_2012_2015.parquet",
    "data/raw/fundamentals/csi300_financials_point_in_time_extended_v2.parquet",
    "data/audit/000300_point_in_time_valuation_v2_snapshot_coverage_v1_2.parquet",
    "data/audit/000300_val01_norm_ey_5y_metric_audit_panel_v1.parquet",
    "reports/data_quality/000300_normalized_earnings_financial_extension_v1.json",
    "reports/data_quality/000300_normalized_earnings_financial_extension_v1.md",
    "reports/data_quality/000300_point_in_time_valuation_v2_reconstructibility_v1_2.json",
    "reports/data_quality/000300_point_in_time_valuation_v2_reconstructibility_v1_2.md",
)

EXPECTED_CHECKPOINT_COUNTS = {
    "vip/income_vip": 13,
    "vip/balancesheet_vip": 13,
    "vip/fina_indicator_vip": 13,
    "single_security_history/income_vip": 88,
    "single_security_history/balancesheet_vip": 96,
    "single_security_history/fina_indicator_vip": 95,
    "overlap_validation/income_vip": 1,
    "overlap_validation/balancesheet_vip": 1,
    "overlap_validation/fina_indicator_vip": 1,
}

EXPECTED_BRANCH_STATUS = {
    "VAL01_RAW_EY_5Y": "REJECTED_PREDICTIVE_SCREEN_NOT_REOPENED",
    "VAL01_NORM_EY_5Y": "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY",
    "VAL02_NORM_EY_SPREAD_5Y": "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY",
    "VAL01_NORM_EY_7Y": "BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED",
    "VAL02_NORM_EY_SPREAD_7Y": "BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED",
}


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _assert_artifacts_exist() -> None:
    missing = [
        path
        for path in (*IMPLEMENTATION_FILES, *DATA_AND_EVIDENCE_FILES)
        if not (ROOT / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"冻结所需文件缺失：{missing}")


def _checkpoint_inventory(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"检查点目录不存在：{directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    inventory = [
        {
            "file": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    counts: dict[str, int] = {}
    for relative, expected in EXPECTED_CHECKPOINT_COUNTS.items():
        target = directory / Path(relative)
        actual = len(list(target.glob("*.parquet"))) if target.is_dir() else 0
        counts[relative] = actual
        if actual != expected:
            raise RuntimeError(
                f"检查点数量不一致：{relative} 预期 {expected}，实际 {actual}"
            )
    return {
        "directory": directory.relative_to(ROOT).as_posix(),
        "file_count": len(inventory),
        "content_sha256": _canonical_hash({"files": inventory}),
        "required_group_counts": counts,
        "files": inventory,
    }


def _run_tests(arguments: list[str]) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        *arguments,
        "-q",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "冻结前测试失败：\n"
            f"命令：{command}\n标准输出：\n{result.stdout}\n错误输出：\n{result.stderr}"
        )
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def _assert_governance(governance: dict[str, Any]) -> None:
    prohibited_true = [key for key, value in governance.items() if value is True]
    if prohibited_true:
        raise RuntimeError(f"数据冻结发现越权动作：{prohibited_true}")


def _assert_reports(
    acquisition: dict[str, Any], post_audit: dict[str, Any]
) -> None:
    if acquisition.get("status") != "PASS_TARGET_SCOPE_CURRENT_ARCHIVE_CAVEAT":
        raise RuntimeError(f"采集状态不允许冻结：{acquisition.get('status')}")
    if post_audit.get("overall_status") != (
        "PASS_VAL01_NORM_EY_5Y_AND_VAL02_SPREAD_DATA_ONLY"
    ):
        raise RuntimeError(
            f"采后审计状态不允许冻结：{post_audit.get('overall_status')}"
        )
    if post_audit.get("branch_status") != EXPECTED_BRANCH_STATUS:
        raise RuntimeError("分支状态与冻结合同不一致")
    gate = post_audit["history_gates"]["five_year"]
    if gate.get("required_snapshot_count") != 60:
        raise RuntimeError("5年闸门要求的快照数不是60")
    if gate.get("local_weight_snapshot_count") != 60:
        raise RuntimeError("本地权重快照未达到60个月")
    failure_lists = (
        "missing_weight_months",
        "price_coverage_failed_months",
        "ttm_coverage_failed_months",
        "normalized_coverage_failed_months",
        "cgb_10y_exact_date_failed_months",
    )
    nonempty = {key: gate.get(key) for key in failure_lists if gate.get(key)}
    if nonempty:
        raise RuntimeError(f"5年可重建闸门仍有失败月份：{nonempty}")
    minimum = gate["minimum_local_coverages"][
        "normalized_metric_weight_coverage"
    ]
    if float(minimum) < 0.90:
        raise RuntimeError(f"正常化盈利最低权重覆盖低于90%：{minimum}")
    _assert_governance(acquisition["governance"])
    _assert_governance(post_audit["governance"])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    config = load_config()
    _assert_artifacts_exist()
    acquisition_path = config["artifacts"]["acquisition_report_json"]
    post_audit_path = config["artifacts"]["post_acquisition_report_json"]
    acquisition = _load_json(acquisition_path)
    post_audit = _load_json(post_audit_path)
    _assert_reports(acquisition, post_audit)

    frozen_inputs = verify_frozen_inputs(config)
    if frozen_inputs.get("status") != "PASS":
        raise RuntimeError("原冻结输入哈希不一致")

    checkpoint_directory = ROOT / config["artifacts"]["checkpoint_directory"]
    checkpoint_inventory = _checkpoint_inventory(checkpoint_directory)
    targeted_tests = _run_tests(
        [
            "tests/test_val01_norm_ey_5y_financial_extension.py",
            "tests/test_val01_norm_ey_5y_post_acquisition_audit.py",
        ]
    )
    full_suite = _run_tests([])

    payload: dict[str, Any] = {
        "project_id": config["protocol"]["project_id"],
        "protocol_version": config["protocol"]["version"],
        "manifest_version": "1.1.0",
        "status": "FROZEN_DATA_RECONSTRUCTIBILITY_PASS_NO_RETURN_EVALUATION",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "acquisition_status": acquisition["status"],
        "post_acquisition_status": post_audit["overall_status"],
        "branch_status": post_audit["branch_status"],
        "five_year_gate": post_audit["history_gates"]["five_year"],
        "independent_cross_check": post_audit["independent_cross_check"],
        "implementation_files": {
            path: sha256_file(ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "data_and_evidence_files": {
            path: {
                "size_bytes": (ROOT / path).stat().st_size,
                "sha256": sha256_file(ROOT / path),
            }
            for path in DATA_AND_EVIDENCE_FILES
        },
        "new_checkpoint_inventory": checkpoint_inventory,
        "frozen_input_audit": frozen_inputs,
        "test_results": {
            "targeted": targeted_tests,
            "full_suite": full_suite,
        },
        "governance": {
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "raw_ey_rejected_branch_reopened": False,
            "seven_year_branch_unblocked": False,
            "old_protocol_manifest_overwritten": False,
        },
        "meaning": (
            "只冻结5年正常化盈利及正常化盈利利差的数据可重建性；"
            "不表示预测有效，不包含收益、IC、仓位、订单或券商连接。"
        ),
        "next_allowed_step": (
            "另行冻结VAL01_NORM_EY_5Y与VAL02_NORM_EY_SPREAD_5Y模型卡；"
            "在目标、持有期、T+1可捕获部分、成本、基准和拆分验证合同冻结前，"
            "不得开展正式收益评估。"
        ),
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    manifest = ROOT / config["artifacts"]["manifest"]
    _atomic_write_json(manifest, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "manifest": manifest.relative_to(ROOT).as_posix(),
                "manifest_content_sha256": payload["manifest_content_sha256"],
                "five_year_minimum_normalized_coverage": payload["five_year_gate"]
                ["minimum_local_coverages"]
                ["normalized_metric_weight_coverage"],
                "test_results": payload["test_results"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
