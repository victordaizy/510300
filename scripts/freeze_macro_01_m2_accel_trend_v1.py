"""冻结MACRO-01协议、实现和输入哈希；冻结前不得读取真实未来收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "macro_01_m2_accel_trend_v1.yaml"
MANIFEST = ROOT / "config" / "macro_01_m2_accel_trend_v1_manifest.json"
FROZEN_FILES = [
    "config/macro_01_m2_accel_trend_v1.yaml",
    "docs/510300_MACRO_01_M2_ACCEL_TREND_V1_SPEC.md",
    "research/daily_macro_01_m2_accel_trend_v1.py",
    "scripts/download_china_money_supply_monthly.py",
    "scripts/run_macro_01_m2_accel_trend_v1.py",
    "tests/test_macro_01_m2_accel_trend_v1.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError(f"冻结清单已存在，禁止覆盖：{MANIFEST}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for output in config["outputs"].values():
        if (ROOT / output).exists():
            raise FileExistsError(f"历史输出已存在，禁止在冻结前覆盖：{output}")
    frozen_hashes = {}
    for relative in FROZEN_FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"待冻结文件缺失：{relative}")
        frozen_hashes[relative] = sha256_file(path)
    input_hashes = {}
    for contract in config["data_contracts"].values():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"冻结输入缺失：{contract['file']}")
        actual = sha256_file(path)
        if actual != contract["sha256"]:
            raise ValueError(f"冻结输入哈希不匹配：{contract['file']}，实际{actual}")
        input_hashes[contract["file"]] = actual
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_state": config["protocol"]["state"],
        "frozen_file_sha256": frozen_hashes,
        "input_file_sha256": input_hashes,
        "historical_run_completed": False,
        "real_510300_future_returns_seen_before_freeze": False,
        "m2_factor_outcomes_seen_before_freeze": False,
        "data_distributions_seen_before_freeze": [
            "M2原始月份范围、余额、同比及8个官方抽查",
            "510300行情与分红数据质量",
        ],
        "known_data_limitation": "M2为当前历史版本，不是逐月公告快照；通过也不能授权Paper或实盘。",
        "parameter_search_allowed": False,
        "rescue_after_result_forbidden": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "manifest": str(MANIFEST.relative_to(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

