"""在收益计算前冻结图形状态马丁-海龟V2实现与输入快照。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "graph_regime_martin_turtle_v2_implementation_manifest.json"
PROTOCOL_MANIFEST = ROOT / "config" / "graph_regime_martin_turtle_v2_protocol_manifest.json"
DATA_GATE = ROOT / "reports" / "data_quality" / "donchian_discovery_v1_data_gate.json"
IMPLEMENTATION_FILES = (
    "research/graph_regime_martin_turtle_v2.py",
    "scripts/run_graph_regime_martin_turtle_v2.py",
    "scripts/freeze_graph_regime_v2_implementation.py",
    "tests/test_graph_regime_martin_turtle_v2.py",
)
INPUT_FILES = (
    "data/raw/r6/510300_daily.parquet",
    "data/raw/r6/H00300_total_return_daily.parquet",
    "data/reference/510300_dividends.csv",
    "data/reference/510300_dividends_coverage.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _verify_protocol_manifest() -> dict:
    if not PROTOCOL_MANIFEST.exists():
        raise FileNotFoundError("V2协议冻结清单缺失")
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    for relative_path, expected_hash in protocol["frozen_files"].items():
        path = ROOT / relative_path
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"V2协议冻结后发生变化：{relative_path}")
    return protocol


def _verify_data_gate() -> dict:
    if not DATA_GATE.exists():
        raise FileNotFoundError("数据闸门报告缺失")
    report = json.loads(DATA_GATE.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("return_calculation_allowed") is not True:
        raise RuntimeError("数据闸门未通过，禁止冻结实现并计算收益")
    return report


def _stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "frozen_at"}


def main() -> int:
    protocol = _verify_protocol_manifest()
    data_gate = _verify_data_gate()
    missing = [
        path
        for path in (*IMPLEMENTATION_FILES, *INPUT_FILES)
        if not (ROOT / path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"实现冻结文件缺失：{missing}")
    payload = {
        "project_id": protocol["project_id"],
        "version": protocol["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "IMPLEMENTATION_FROZEN_HISTORICAL_TEST_ALLOWED",
        "protocol_manifest_sha256": sha256(PROTOCOL_MANIFEST),
        "data_gate_report_sha256": sha256(DATA_GATE),
        "data_gate_generated_at": data_gate["generated_at"],
        "data_cutoff": data_gate["data_cutoff"],
        "return_calculation_allowed": True,
        "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
        "true_forward_start": None,
        "implementation_files": {
            path: sha256(ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "input_files": {path: sha256(ROOT / path) for path in INPUT_FILES},
        "candidate_ids": protocol["candidate_ids"],
        "external_frozen_baseline": protocol["external_frozen_baseline"],
        "governance": {
            "live_position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if _stable(existing) != _stable(payload):
            raise RuntimeError("实现或输入指纹已变化，禁止覆盖原冻结清单")
        print("V2实现冻结清单已存在，指纹一致。")
        return 0
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
