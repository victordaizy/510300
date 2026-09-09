"""在510300开发期结果前冻结50ETF文献IVS迁移公式。"""

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
CONFIG = ROOT / "config" / "510300_option_ivs_transfer_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_ivs_transfer_v1_manifest.json"
FROZEN_FILES = (
    "config/510300_option_ivs_transfer_v1.yaml",
    "config/510300_option_trade_envelope_v1_manifest.json",
    "config/510300_option_technical_rule_discovery_v1_manifest.json",
    "research/option_ivs_transfer_v1.py",
    "scripts/freeze_510300_option_ivs_transfer_v1.py",
    "tests/test_510300_option_ivs_transfer_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "510300_OPTION_IVS_TRANSFER_V1":
        raise ValueError("IVS迁移协议ID漂移")
    if config["protocol"]["factor_count"] > 10:
        raise ValueError("IVS迁移因子超过10个")
    if config["research_basis"]["direct_generalization_claim"]:
        raise ValueError("不得把50ETF文献直接冒充510300证据")
    if config["model"]["parameter_search_allowed"] or config["model"]["threshold_search_allowed"]:
        raise ValueError("IVS迁移不得搜索参数或阈值")
    if config["dates"]["validation_or_holdout_access_authorized"]:
        raise ValueError("开发协议不得读取验证或最终留出")
    if config["governance"]["validation_or_holdout_access"] != "FORBIDDEN":
        raise ValueError("验证与最终留出访问必须禁止")
    for key, hash_key in (
        ("upstream_vehicle_manifest", "upstream_vehicle_manifest_sha256"),
        ("data_loader_manifest", "data_loader_manifest_sha256"),
    ):
        path = ROOT / config["protocol"][key]
        if sha256(path) != config["protocol"][hash_key]:
            raise RuntimeError(f"IVS迁移依赖清单漂移：{path}")
    from research.option_trade_envelope_v1 import assert_input_hashes
    from scripts.freeze_510300_option_technical_rule_discovery_v1 import (
        verify_protocol as verify_data_loader_protocol,
    )
    from scripts.freeze_510300_option_trade_envelope_v1 import (
        verify_protocol as verify_vehicle_protocol,
    )

    verify_data_loader_protocol()
    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.exists():
        raise RuntimeError("IVS迁移协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("IVS迁移冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"IVS迁移冻结哈希漂移：{drift}")
    return config, manifest


def main() -> int:
    config = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"IVS迁移冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("510300期权IVS外部规则迁移V1已冻结，全部哈希一致。")
        return 0
    for key in ("result_json", "result_markdown", "monthly_predictions"):
        if (ROOT / config["paths"][key]).exists():
            raise RuntimeError("IVS迁移开发结果已存在，禁止事后冻结")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_ivs_transfer_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"IVS迁移冻结前测试失败：\n{detail}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "development_outcomes_read_at_freeze": False,
        "validation_or_holdout_access_authorized": False,
        "factor_count": config["protocol"]["factor_count"],
        "external_rule_source": config["research_basis"]["source_url"],
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
