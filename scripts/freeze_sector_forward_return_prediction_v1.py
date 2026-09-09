"""在读取历史未来收益前冻结板块未来贡献预测 V1。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "sector_forward_return_prediction_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "sector_forward_return_prediction_v1_manifest.json"
FROZEN_FILES = (
    "CONTEXT.md",
    "docs/SECTOR_FORWARD_RETURN_PREDICTION_V1_SPEC.md",
    "config/sector_forward_return_prediction_v1.yaml",
    "research/sector_future_return_prediction.py",
    "scripts/freeze_sector_forward_return_prediction_v1.py",
    "scripts/run_sector_forward_return_prediction_v1.py",
    "tests/test_sector_future_return_prediction.py",
)


def sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _validate_contract(contract: dict) -> None:
    protocol = contract["protocol"]
    model = contract["model"]
    governance = contract["governance"]
    if protocol["state"] != "DISCOVERY_ONLY":
        raise ValueError("预测协议只能处于 DISCOVERY_ONLY")
    if protocol["candidate_budget"] != 1:
        raise ValueError("板块预测 V1 候选预算必须严格等于 1")
    if protocol["true_forward_start"] is not None:
        raise ValueError("历史评价冻结时 true_forward_start 必须为空")
    if not protocol["candidate_addition_after_freeze_forbidden"]:
        raise ValueError("冻结后必须禁止追加候选")
    if not protocol["threshold_change_after_freeze_forbidden"]:
        raise ValueError("冻结后必须禁止修改门槛")
    if model["candidate_id"] != "S1_SECTOR_FUNDAMENTAL_RIDGE_V1":
        raise ValueError("只允许冻结唯一候选 S1")
    if model["estimator"] != "RIDGE_FIXED" or float(model["alpha"]) != 10.0:
        raise ValueError("S1 必须使用固定 alpha=10 的 Ridge")
    if model["fit_intercept"] is not False:
        raise ValueError("S1 必须固定 fit_intercept=false")
    if model["feature_selection_allowed"] or model["hyperparameter_search_allowed"]:
        raise ValueError("冻结前后均不得做特征或超参数搜索")
    forbidden_true = {
        "historical_run_may_emit_forecast_eligible",
        "driver_episode_features_allowed",
        "x1_features_allowed",
        "r6_reselection_allowed",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    }
    enabled = sorted(key for key in forbidden_true if governance.get(key) is True)
    if enabled:
        raise ValueError(f"研究安全开关不得启用：{enabled}")
    if governance["missing_input_output"] != "NO_VIEW":
        raise ValueError("缺失输入的唯一输出必须是 NO_VIEW")


def _validate_data_status(contract: dict) -> dict:
    status_path = ROOT / contract["inputs"]["sector_data_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "READY_FOR_TARGET_FREEZE":
        raise RuntimeError("板块点时数据没有通过目标冻结门")
    if status.get("future_return_read") is not False:
        raise RuntimeError("数据门运行时已经读取未来收益，禁止冻结")
    if status.get("predictive_model_fitted") is not False:
        raise RuntimeError("数据门运行时已经拟合预测模型，禁止冻结")
    panel_relative = contract["inputs"]["sector_panel"]
    expected = status.get("output_hashes", {}).get(panel_relative)
    if expected is None or sha256(ROOT / panel_relative) != expected:
        raise RuntimeError("板块点时面板与数据门状态哈希不一致")
    return status


def main() -> int:
    if not CONTRACT_FILE.exists():
        raise FileNotFoundError(CONTRACT_FILE)
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    _validate_contract(contract)

    missing_frozen = [path for path in FROZEN_FILES if not (ROOT / path).exists()]
    if missing_frozen:
        raise FileNotFoundError(f"冻结文件缺失：{missing_frozen}")
    input_paths = [str(value) for value in contract["inputs"].values()]
    missing_inputs = [path for path in input_paths if not (ROOT / path).exists()]
    if missing_inputs:
        raise FileNotFoundError(f"冻结输入缺失：{missing_inputs}")
    data_status = _validate_data_status(contract)

    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_sector_future_return_prediction.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError(
            "板块预测冻结测试失败：\n" + test.stdout + "\n" + test.stderr
        )

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "state": "FROZEN_BEFORE_HISTORICAL_EVALUATION",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "historical_contamination_cutoff": contract["protocol"][
            "historical_contamination_cutoff"
        ],
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(status.stdout.strip()),
        "test_command": (
            f"{sys.executable} -m pytest "
            "tests/test_sector_future_return_prediction.py -q"
        ),
        "test_result": test.stdout.strip(),
        "frozen_files": {path: sha256(ROOT / path) for path in FROZEN_FILES},
        "input_files": {path: sha256(ROOT / path) for path in input_paths},
        "parent_data_status": {
            "status": data_status["status"],
            "checked_at": data_status["checked_at"],
            "snapshot_count": data_status["snapshot_count"],
        },
        "candidate": {
            "candidate_id": contract["model"]["candidate_id"],
            "candidate_budget": contract["protocol"]["candidate_budget"],
            "alpha": contract["model"]["alpha"],
            "numeric_features": contract["model"]["numeric_features"],
        },
        "governance": contract["governance"],
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(MANIFEST_FILE)
    print(
        json.dumps(
            {
                "状态": manifest["state"],
                "候选": manifest["candidate"]["candidate_id"],
                "冻结文件数": len(manifest["frozen_files"]),
                "冻结输入数": len(manifest["input_files"]),
                "测试": manifest["test_result"],
                "清单": str(MANIFEST_FILE),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
