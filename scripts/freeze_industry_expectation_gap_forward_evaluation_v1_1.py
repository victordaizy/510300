"""冻结行业预期差前瞻评价 V1.1 的精确恢复执行层。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_expectation_gap_forward_evaluation import (  # noqa: E402
    ForwardEvaluationError,
    file_sha256,
    verify_hash_manifest,
)
from research.industry_expectation_gap_frozen_input_recovery import (  # noqa: E402
    validate_recovery_config,
    verify_manifest_with_declared_recoveries,
)


CONFIG_RELATIVE = "config/industry_expectation_gap_forward_evaluation_v1_1_recovery.yaml"
FROZEN_FILES = [
    CONFIG_RELATIVE,
    "docs/INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_1_RECOVERY_ADDENDUM.md",
    "docs/incidents/INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_20260819_FROZEN_INPUT.md",
    "paper/industry_expectation_gap_v1/frozen_inputs/2026-08-18_primary_market_readiness.json",
    "reports/audit/industry_expectation_gap_forward_evaluation_v1_20260819_frozen_input.json",
    "research/industry_expectation_gap_frozen_input_recovery.py",
    "scripts/freeze_industry_expectation_gap_forward_evaluation_v1_1.py",
    "scripts/recover_industry_expectation_gap_v1_frozen_readiness.py",
    "scripts/run_industry_expectation_gap_forward_evaluation_v1_1.py",
    "tests/test_industry_expectation_gap_frozen_input_recovery.py",
]


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ForwardEvaluationError(f"路径越出工作区：{relative}") from exc
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ForwardEvaluationError(f"YAML 顶层必须是对象：{path}")
    return value


def _git_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_once(path: Path, content: str) -> str:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ForwardEvaluationError(f"恢复清单已存在且内容不同：{path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "CREATED"


def main() -> int:
    config = _load_yaml(_path(CONFIG_RELATIVE))
    validate_recovery_config(config)
    parent = config["parent_evaluation"]
    if file_sha256(_path(parent["config"])) != parent["config_sha256"]:
        raise ForwardEvaluationError("原评价配置哈希不一致")
    verify_hash_manifest(ROOT, parent["manifest"], parent["manifest_sha256"])

    frozen = config["frozen_prediction"]
    verification = verify_manifest_with_declared_recoveries(
        ROOT,
        manifest_relative_path=frozen["manifest"],
        expected_manifest_sha256=frozen["manifest_sha256"],
        recoveries=config["recoveries"],
    )
    for field, hash_field, label in [
        ("result", "result_sha256", "原预测结果"),
        ("ledger", "ledger_sha256", "原预测账本"),
    ]:
        if file_sha256(_path(frozen[field])) != frozen[hash_field]:
            raise ForwardEvaluationError(f"{label}哈希不一致")

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.date().isoformat() != "2026-08-19":
        raise ForwardEvaluationError("V1.1 恢复执行层只允许在事故当日冻结")
    first_maturity = datetime.fromisoformat("2026-11-18T15:00:00+08:00")
    if now >= first_maturity:
        raise ForwardEvaluationError("恢复执行层不得在首个成熟日后冻结")
    output_dir = _path(config["outputs"]["directory"])
    forbidden_outputs = [
        output_dir / "industry_expectation_gap_forward_evaluation_v1_2026-08-19.json",
        output_dir / "industry_expectation_gap_forward_evaluation_v1_1_2026-08-19.json",
    ]
    existing = [str(path) for path in forbidden_outputs if path.exists()]
    if existing:
        raise ForwardEvaluationError(f"冻结前不应已有 8 月 19 日评价输出：{existing}")

    frozen_files: list[dict[str, Any]] = []
    for relative in sorted(FROZEN_FILES):
        path = _path(relative)
        if not path.is_file():
            raise ForwardEvaluationError(f"待冻结文件不存在：{relative}")
        frozen_files.append(
            {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        )
    manifest = {
        "manifest_version": "INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_1_RECOVERY_MANIFEST",
        "status": "FROZEN_RECOVERY_ADDENDUM_AFTER_FIRST_FORWARD_OBSERVATION",
        "frozen_at": now.isoformat(timespec="seconds"),
        "purpose": "EXACT_FROZEN_INPUT_RECOVERY_ONLY",
        "parent_evaluation": parent,
        "frozen_prediction": frozen,
        "frozen_files": frozen_files,
        "prediction_manifest_verification": verification,
        "pre_evaluation_state": {
            "entry_date_has_started": True,
            "first_horizon_matured": False,
            "partial_forward_return_read": False,
            "partial_forward_return_output": False,
            "v1_2026_08_19_output_exists": False,
            "v1_1_2026_08_19_output_exists": False,
        },
        "constraints": config["constraints"],
        "authorization": config["safety"],
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(_git_text("status", "--porcelain")),
            "all_recovery_files_tracked": False,
        },
    }
    manifest_path = _path(config["outputs"]["manifest"])
    state = _write_once(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"V1.1 恢复清单：{state}")
    print(f"输出：{manifest_path}")
    print(f"原清单当前精确对象：{verification['frozen_file_count'] - verification['recovered_target_count']}")
    print(f"精确恢复对象：{verification['recovered_target_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
