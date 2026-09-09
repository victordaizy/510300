"""在读取2024结果前冻结IVS迁移V1验证器。"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "config" / "510300_option_ivs_transfer_v1_validation_2024.yaml"
MANIFEST = ROOT / "config" / "510300_option_ivs_transfer_v1_validation_2024_manifest.json"
FROZEN_FILES = (
    "config/510300_option_ivs_transfer_v1_validation_2024.yaml",
    "config/510300_option_ivs_transfer_v1_manifest.json",
    "reports/discovery/510300_option_ivs_transfer_v1_dev.json",
    "data/features/510300_option_ivs_transfer_v1_dev_predictions.parquet",
    "research/option_ivs_transfer_v1_validation_2024.py",
    "scripts/freeze_510300_option_ivs_transfer_v1_validation_2024.py",
    "tests/test_510300_option_ivs_transfer_v1_validation_2024.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "510300_OPTION_IVS_TRANSFER_V1_VALIDATION_2024":
        raise ValueError("IVS 2024验证协议ID漂移")
    if config["protocol"]["formula_or_threshold_change_allowed"]:
        raise ValueError("2024验证不得修改公式或阈值")
    if config["final_holdout"]["access_authorized"]:
        raise ValueError("2024验证不得授权最终留出")
    if config["governance"]["final_holdout_access"] != "FORBIDDEN":
        raise ValueError("最终留出访问必须禁止")
    for key, hash_key in (
        ("formula_manifest", "formula_manifest_sha256"),
        ("development_result", "development_result_sha256"),
        ("development_predictions", "development_predictions_sha256"),
    ):
        path = ROOT / config["protocol"][key]
        if sha256(path) != config["protocol"][hash_key]:
            raise RuntimeError(f"2024验证依赖漂移：{path}")
    development = json.loads(
        (ROOT / config["protocol"]["development_result"]).read_text(encoding="utf-8")
    )
    if development["status"] != "DEVELOPMENT_TRANSFER_PASS_VALIDATION_NOT_AUTHORIZED":
        raise RuntimeError("开发期状态未通过，禁止冻结2024验证")
    if not development["gates"]["development_transfer_pass"]:
        raise RuntimeError("开发期门槛未通过，禁止冻结2024验证")
    from scripts.freeze_510300_option_ivs_transfer_v1 import verify_protocol as verify_formula

    verify_formula()
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.exists():
        raise RuntimeError("IVS 2024验证协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("IVS 2024验证冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"IVS 2024验证冻结哈希漂移：{drift}")
    return config, manifest


def main() -> int:
    config = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"IVS 2024验证冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("510300期权IVS迁移V1的2024验证器已冻结，全部哈希一致。")
        return 0
    for key in ("result_json", "result_markdown", "predictions"):
        if (ROOT / config["paths"][key]).exists():
            raise RuntimeError("2024验证结果已存在，禁止事后冻结")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_ivs_transfer_v1_validation_2024.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"2024验证冻结前测试失败：\n{detail}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "validation_outcomes_read_at_freeze": False,
        "final_holdout_access_authorized": False,
        "formula_manifest_sha256": config["protocol"]["formula_manifest_sha256"],
        "development_result_sha256": config["protocol"]["development_result_sha256"],
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": config["governance"],
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
