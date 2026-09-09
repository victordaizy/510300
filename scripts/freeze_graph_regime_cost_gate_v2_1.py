"""在收益计算前冻结V2.1小单成本门、实现与输入。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "graph_regime_cost_gate_v2_1_manifest.json"
FROZEN_FILES = (
    "docs/510300_GRAPH_REGIME_COST_GATE_V2_1_SPEC.md",
    "config/graph_regime_cost_gate_v2_1.yaml",
    "research/graph_regime_cost_gate_v2_1.py",
    "scripts/run_graph_regime_cost_gate_v2_1.py",
    "scripts/freeze_graph_regime_cost_gate_v2_1.py",
    "tests/test_graph_regime_cost_gate_v2_1.py",
)
INPUT_FILES = (
    "config/graph_regime_martin_turtle_v2.yaml",
    "research/graph_regime_martin_turtle_v2.py",
    "config/graph_regime_martin_turtle_v2_implementation_manifest_reporting_fix.json",
    "reports/backtest/graph_regime_martin_turtle_v2.json",
    "reports/data_quality/donchian_discovery_v1_data_gate.json",
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


def _stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "frozen_at"}


def main() -> int:
    config = yaml.safe_load(
        (ROOT / "config" / "graph_regime_cost_gate_v2_1.yaml").read_text(
            encoding="utf-8"
        )
    )
    gate = config["cost_gate"]
    if float(gate["maximum_one_way_explicit_cost_bps"]) != 10.0:
        raise ValueError("成本门不得偏离单边10bp")
    if float(gate["derived_minimum_buy_notional_cny"]) != 10_000.0:
        raise ValueError("最低买入金额不得偏离10000元")
    if gate["sell_and_exposure_reduction_exempt"] is not True:
        raise ValueError("卖出与减仓必须豁免")
    if config["protocol"]["true_forward_start"] is not None:
        raise ValueError("真正前向起点必须保持null")
    data_gate = json.loads(
        (ROOT / "reports" / "data_quality" / "donchian_discovery_v1_data_gate.json").read_text(
            encoding="utf-8"
        )
    )
    if data_gate.get("status") != "PASS":
        raise RuntimeError("数据闸门未通过")
    missing = [
        path for path in (*FROZEN_FILES, *INPUT_FILES) if not (ROOT / path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"V2.1冻结文件缺失：{missing}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "COST_GATE_IMPLEMENTATION_FROZEN_HISTORICAL_TEST_ALLOWED",
        "evidence_label": config["protocol"]["evidence_label"],
        "data_cutoff": config["protocol"]["source_data_cutoff"],
        "true_forward_start": None,
        "return_calculation_allowed": True,
        "cost_gate": gate,
        "frozen_files": {path: sha256(ROOT / path) for path in FROZEN_FILES},
        "input_files": {path: sha256(ROOT / path) for path in INPUT_FILES},
        "governance": config["governance"],
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if _stable(existing) != _stable(payload):
            raise RuntimeError("V2.1成本门指纹已变化，禁止覆盖")
        print("V2.1成本门冻结清单已存在，指纹一致。")
        return 0
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
