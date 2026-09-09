"""冻结40交易日深度实值期权时间序列动量V1。"""

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
CONFIG = ROOT / "config" / "510300_option_long_horizon_tsmom_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_long_horizon_tsmom_v1_manifest.json"
FROZEN_FILES = (
    "config/510300_option_long_horizon_tsmom_v1.yaml",
    "config/510300_option_trade_envelope_v1_manifest.json",
    "config/510300_option_technical_rule_discovery_v1_manifest.json",
    "research/option_long_horizon_tsmom_v1.py",
    "scripts/freeze_510300_option_long_horizon_tsmom_v1.py",
    "tests/test_510300_option_long_horizon_tsmom_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "510300_OPTION_LONG_HORIZON_TSMOM_V1":
        raise ValueError("长周期动量协议ID漂移")
    if config["protocol"]["factor_count"] > 10:
        raise ValueError("长周期策略因子超过10个")
    if config["factor"]["parameter_search_allowed"]:
        raise ValueError("长周期动量不得搜索参数")
    if config["dates"]["post_development_access_authorized"]:
        raise ValueError("开发协议不得访问后续区间")
    if config["governance"]["post_development_access"] != "FORBIDDEN":
        raise ValueError("开发期后访问必须禁止")
    upstream = ROOT / config["protocol"]["upstream_vehicle_manifest"]
    if sha256(upstream) != config["protocol"]["upstream_vehicle_manifest_sha256"]:
        raise RuntimeError("上游车辆冻结清单漂移")
    loader_manifest = ROOT / config["protocol"]["data_loader_manifest"]
    if sha256(loader_manifest) != config["protocol"]["data_loader_manifest_sha256"]:
        raise RuntimeError("数据加载工具冻结清单漂移")
    from research.option_trade_envelope_v1 import assert_input_hashes
    from scripts.freeze_510300_option_trade_envelope_v1 import (
        verify_protocol as verify_vehicle_protocol,
    )

    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    from scripts.freeze_510300_option_technical_rule_discovery_v1 import (
        verify_protocol as verify_data_loader_protocol,
    )

    verify_data_loader_protocol()
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.exists():
        raise RuntimeError("长周期动量协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("长周期冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"长周期冻结哈希漂移：{drift}")
    return config, manifest


def main() -> int:
    config = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"长周期冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("510300期权长周期时间序列动量V1已冻结，全部哈希一致。")
        return 0
    for key in ("result_json", "result_markdown"):
        if (ROOT / config["paths"][key]).exists():
            raise RuntimeError("长周期开发结果已存在，禁止事后冻结")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_long_horizon_tsmom_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"长周期冻结前测试失败：\n{detail}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "development_outcomes_read_at_freeze": False,
        "post_development_access_authorized": False,
        "factor_count": config["protocol"]["factor_count"],
        "holding_trading_days": config["schedule"]["holding_trading_days"],
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
