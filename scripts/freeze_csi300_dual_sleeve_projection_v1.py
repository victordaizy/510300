"""在读取双时段投影结果前冻结父公式投影。"""

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
CONFIG_FILE = ROOT / "config" / "csi300_dual_sleeve_projection_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "csi300_dual_sleeve_projection_v1_manifest.json"
FROZEN_FILES = (
    "config/csi300_dual_sleeve_projection_v1.yaml",
    "config/final_alpha_strategy.yaml",
    "docs/CSI300_DUAL_SLEEVE_PROJECTION_V1_SPEC.md",
    "research/csi300_dual_sleeve_projection_v1.py",
    "research/small_account_cross_sectional.py",
    "backtest/run_final_alpha_strategy.py",
    "scripts/run_csi300_dual_sleeve_projection_v1.py",
    "tests/test_csi300_dual_sleeve_projection_v1.py",
    "tests/test_small_account_cross_sectional.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if contract["factor_budget"]["used"] != 8 or contract["factor_budget"]["maximum"] != 10:
        raise ValueError("父公式投影必须沿用8因子且不超过10")
    if contract["protocol"]["parameter_search_allowed"] or contract["projection"]["holdings"] != 3:
        raise ValueError("禁止参数搜索且投影必须为三仓")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    outputs = contract["outputs"]
    existing = [value for value in outputs.values() if (ROOT / value).exists()]
    if existing:
        raise RuntimeError(f"冻结前已有投影结果：{existing}")
    inputs = contract["inputs"]
    input_keys = ["recent_member_panel", "recent_index", "recent_benchmark", "recent_breadth", "external_status", "external_member_panel", "external_index", "external_benchmark"]
    absent = [inputs[key] for key in input_keys if not (ROOT / inputs[key]).exists()]
    if absent:
        raise FileNotFoundError(f"输入缺失：{absent}")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_csi300_dual_sleeve_projection_v1.py", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": contract["protocol"]["project_id"], "state": "FROZEN_BEFORE_DUAL_PERIOD_PROJECTION_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "input_files": {inputs[key]: sha256(ROOT / inputs[key]) for key in input_keys},
        "recent_component_history_tree": tree_sha256(ROOT / inputs["recent_component_history_cache"]),
        "test_result": test.stdout.strip(),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "父公式因子": 8, "投影持仓": 3, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
