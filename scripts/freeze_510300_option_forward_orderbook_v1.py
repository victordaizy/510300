"""冻结并验证510300期权前向收盘盘口V1协议。"""

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
CONFIG = ROOT / "config" / "510300_option_forward_orderbook_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_forward_orderbook_v1_protocol_manifest.json"
FROZEN_FILES = (
    "config/510300_option_forward_orderbook_v1.yaml",
    "docs/510300_OPTION_FORWARD_ORDERBOOK_V1_SPEC.md",
    "config/return_tail_hypothesis_registry.yaml",
    "research/option_research_data_acquisition.py",
    "research/return_tail_supplemental_acquisition.py",
    "scripts/download_csi300_all_etf_momentum_v1.py",
    "scripts/collect_510300_option_forward_orderbook_v1.py",
    "scripts/audit_510300_option_forward_orderbook_v1.py",
    "scripts/freeze_510300_option_forward_orderbook_v1.py",
    "tests/test_510300_option_forward_orderbook_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract() -> dict:
    if not CONFIG.exists():
        raise FileNotFoundError(f"协议配置不存在：{CONFIG}")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _assert_contract(contract: dict) -> dict:
    protocol = contract["protocol"]
    capture = contract["capture"]
    governance = contract["governance"]
    if protocol["project_id"] != "510300_OPTION_FORWARD_ORDERBOOK_V1":
        raise ValueError("协议ID不匹配")
    candidate_ids = list(protocol["candidate_ids"])
    if candidate_ids != ["O1", "O2", "O3", "O4"]:
        raise ValueError("本协议只允许冻结O1至O4")
    if int(protocol["factor_count"]) != len(candidate_ids) or len(candidate_ids) > 10:
        raise ValueError("因子数量必须与候选一致且不超过10")
    if str(protocol["historical_contamination_cutoff"]) != "2026-08-18":
        raise ValueError("历史污染截止日漂移")
    if str(protocol["earliest_permitted_capture_date"]) != "2026-08-19":
        raise ValueError("最早前向采集日漂移")
    if capture["window_start"] != "14:58:30" or capture["window_end"] != "15:05:30":
        raise ValueError("采集窗口漂移")
    if not capture["append_only_date_files"] or not capture["overwrite_forbidden"]:
        raise ValueError("前向文件必须只追加且禁止覆盖")
    forbidden = (
        governance["historical_backfill_forbidden"],
        not governance["signal_calculation_before_data_gate"],
        not governance["return_test_before_data_gate"],
        governance["position_mapping"] == "DISABLED",
        governance["order_generation"] == "DISABLED",
        governance["broker_connection"] == "DISABLED",
        governance["live_trading"] == "NOT_AUTHORIZED",
    )
    if not all(forbidden):
        raise ValueError("研究、交易或回填治理边界漂移")
    upstream = ROOT / protocol["upstream_registry"]
    registry = yaml.safe_load(upstream.read_text(encoding="utf-8"))
    definitions = {item["id"] for item in registry["hypotheses"]}
    if not set(candidate_ids).issubset(definitions):
        raise ValueError("上游注册表缺少冻结候选")
    if registry["protocol"]["true_forward_start"] is not None:
        raise ValueError("上游注册表不得事后回填true_forward_start")
    return registry


def verify_protocol() -> tuple[dict, dict]:
    """逐文件验证冻结指纹；供采集器与审计器在每次运行前调用。"""

    contract = load_contract()
    _assert_contract(contract)
    if not MANIFEST.exists():
        raise RuntimeError("协议尚未冻结，禁止采集")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("project_id") != contract["protocol"]["project_id"]:
        raise RuntimeError("冻结清单项目ID不匹配")
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"冻结文件哈希漂移：{drift}")
    upstream = ROOT / contract["protocol"]["upstream_registry"]
    if sha256(upstream) != manifest["upstream_registry_sha256"]:
        raise RuntimeError("上游O1至O4注册表哈希漂移")
    return contract, manifest


def _assert_no_prefreeze_forward_data(contract: dict) -> None:
    paths = contract["paths"]
    for key in ("snapshot_directory", "master_snapshot_directory"):
        directory = ROOT / paths[key]
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"冻结前已存在前向数据，拒绝冻结：{directory}")
    start = ROOT / paths["forward_start_status"]
    if start.exists():
        raise RuntimeError(f"冻结前已存在前向起点，拒绝冻结：{start}")


def _run_tests() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_forward_orderbook_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-4000:]
        raise RuntimeError(f"冻结前测试未通过：\n{detail}")


def main() -> int:
    contract = load_contract()
    registry = _assert_contract(contract)
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("510300期权前向盘口V1冻结清单已存在，全部哈希一致。")
        return 0
    _assert_no_prefreeze_forward_data(contract)
    _run_tests()
    payload = {
        "project_id": contract["protocol"]["project_id"],
        "state": "COLLECTOR_FROZEN_BEFORE_FIRST_VALID_FORWARD_CAPTURE",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "historical_contamination_cutoff": str(contract["protocol"]["historical_contamination_cutoff"]),
        "earliest_permitted_capture_date": str(contract["protocol"]["earliest_permitted_capture_date"]),
        "candidate_ids": list(contract["protocol"]["candidate_ids"]),
        "factor_count": int(contract["protocol"]["factor_count"]),
        "upstream_project_id": registry["protocol"]["project_id"],
        "upstream_registry_sha256": sha256(ROOT / contract["protocol"]["upstream_registry"]),
        "first_valid_forward_capture": None,
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": contract["governance"],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
