"""在首个有效前向盘口之前冻结O1至O4完整数学实现。"""

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
CONFIG = ROOT / "config" / "510300_option_surface_signal_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_surface_signal_v1_manifest.json"
FROZEN_FILES = (
    "config/510300_option_surface_signal_v1.yaml",
    "docs/510300_OPTION_SURFACE_SIGNAL_V1_SPEC.md",
    "config/return_tail_hypothesis_registry.yaml",
    "config/510300_option_forward_orderbook_v1_protocol_manifest.json",
    "research/option_surface_signal_v1.py",
    "scripts/freeze_510300_option_surface_signal_v1.py",
    "tests/test_510300_option_surface_signal_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["project_id"] != "510300_OPTION_SURFACE_SIGNAL_V1":
        raise ValueError("曲面公式协议ID漂移")
    if list(protocol["candidate_ids"]) != ["O1", "O2", "O3", "O4"]:
        raise ValueError("曲面公式只允许O1至O4")
    if int(protocol["factor_count"]) != 4 or int(protocol["factor_count"]) > 10:
        raise ValueError("冻结因子数量必须为4且不超过10")
    if str(protocol["earliest_input_date"]) != "2026-08-19":
        raise ValueError("曲面公式最早输入日漂移")
    if config["gates"]["raw_factor_calculation_before_orderbook_gate"]:
        raise ValueError("盘口门槛前不得计算真实因子")
    if config["gates"]["return_test_in_this_protocol"]:
        raise ValueError("本协议不得运行收益检验")
    governance = config["governance"]
    expected = {
        "parameter_search": "FORBIDDEN",
        "factor_addition": "FORBIDDEN",
        "historical_bid_ask_backfill": "FORBIDDEN",
        "position_mapping": "DISABLED",
        "order_generation": "DISABLED",
        "broker_connection": "DISABLED",
        "live_trading": "NOT_AUTHORIZED",
    }
    if governance != expected:
        raise ValueError("曲面公式治理边界漂移")
    upstream = yaml.safe_load(
        (ROOT / protocol["upstream_hypothesis_registry"]).read_text(encoding="utf-8")
    )
    definitions = {item["id"] for item in upstream["hypotheses"]}
    if not set(protocol["candidate_ids"]).issubset(definitions):
        raise ValueError("上游注册表缺少O1至O4")
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.exists():
        raise RuntimeError("曲面公式协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("曲面公式冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"曲面公式冻结哈希漂移：{drift}")
    return config, manifest


def _run_tests() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_surface_signal_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"曲面公式冻结前测试失败：\n{detail}")


def main() -> int:
    config = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"曲面公式冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("510300期权曲面公式V1冻结清单已存在，全部哈希一致。")
        return 0
    forward_start = ROOT / "reports/forward/510300_option_orderbook_v1/forward_start_status.json"
    if forward_start.exists():
        raise RuntimeError("首个有效前向盘口已经到达，禁止事后冻结公式")
    output = ROOT / config["paths"]["raw_output"]
    if output.exists():
        raise RuntimeError("冻结前已存在真实因子输出，拒绝冻结")
    _run_tests()
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "earliest_input_date": str(config["protocol"]["earliest_input_date"]),
        "candidate_ids": list(config["protocol"]["candidate_ids"]),
        "factor_count": int(config["protocol"]["factor_count"]),
        "first_valid_forward_capture_at_freeze": None,
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "sources": {
            "model_free_variance": "https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf",
            "arbitrage_free_svi": "https://arxiv.org/abs/1204.0646",
            "har_rv": "https://doi.org/10.1093/jjfinec/nbp001",
        },
        "governance": config["governance"],
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
