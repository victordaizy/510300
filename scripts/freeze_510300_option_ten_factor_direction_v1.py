"""在开发期模型输出前冻结510300期权十因子方向公式。"""

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
CONFIG = ROOT / "config" / "510300_option_ten_factor_direction_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_ten_factor_direction_v1_manifest.json"
FROZEN_FILES = (
    "config/510300_option_ten_factor_direction_v1.yaml",
    "config/510300_option_trade_envelope_v1_manifest.json",
    "research/option_ten_factor_direction_v1.py",
    "scripts/freeze_510300_option_ten_factor_direction_v1.py",
    "tests/test_510300_option_ten_factor_direction_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "510300_OPTION_TEN_FACTOR_DIRECTION_V1":
        raise ValueError("十因子方向协议ID漂移")
    columns = config["factors"]["columns"]
    if len(columns) != config["protocol"]["used_factor_count"] or len(columns) > 10:
        raise ValueError("因子数量违反不超过10个的冻结约束")
    if config["model"]["hyperparameter_search_allowed"]:
        raise ValueError("冻结模型不得搜索超参数")
    if config["model"]["feature_selection_allowed"]:
        raise ValueError("冻结模型不得事后选因子")
    if config["model"]["threshold_search_allowed"]:
        raise ValueError("冻结模型不得搜索方向阈值")
    if config["dates"]["validation_or_holdout_read_authorized"]:
        raise ValueError("本协议不得读取验证或最终留出")
    if config["governance"]["validation_or_holdout_access"] != "FORBIDDEN":
        raise ValueError("验证和最终留出访问必须禁止")
    upstream = ROOT / config["protocol"]["upstream_vehicle_manifest"]
    if sha256(upstream) != config["protocol"]["upstream_vehicle_manifest_sha256"]:
        raise RuntimeError("上游车辆冻结清单不一致")
    for item in config["inputs"].values():
        if sha256(ROOT / item["path"]) != str(item["sha256"]).lower():
            raise RuntimeError(f"十因子冻结输入不一致：{item['path']}")
    from research.option_trade_envelope_v1 import assert_input_hashes
    from scripts.freeze_510300_option_trade_envelope_v1 import (
        verify_protocol as verify_vehicle_protocol,
    )

    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.exists():
        raise RuntimeError("十因子方向协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("十因子冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"十因子冻结文件哈希漂移：{drift}")
    return config, manifest


def main() -> int:
    config = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"十因子冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("510300期权十因子方向V1已冻结，全部哈希一致。")
        return 0
    for key in ("result_json", "result_markdown", "predictions"):
        if (ROOT / config["paths"][key]).exists():
            raise RuntimeError("开发期模型输出已经存在，禁止事后冻结公式")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_ten_factor_direction_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"十因子方向冻结前测试失败：\n{detail}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "development_model_outcomes_read_at_freeze": False,
        "validation_or_holdout_read_authorized": False,
        "factor_count": len(config["factors"]["columns"]),
        "factor_columns": config["factors"]["columns"],
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
