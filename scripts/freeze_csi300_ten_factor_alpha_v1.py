"""在首次读取十因子未来收益前冻结协议、实现和输入。"""

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
CONFIG_FILE = ROOT / "config" / "csi300_ten_factor_alpha_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "csi300_ten_factor_alpha_v1_manifest.json"
FROZEN_FILES = (
    "config/csi300_ten_factor_alpha_v1.yaml",
    "docs/CSI300_TEN_FACTOR_NONLINEAR_ALPHA_V1_SPEC.md",
    "research/csi300_ten_factor_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/freeze_csi300_ten_factor_alpha_v1.py",
    "scripts/run_csi300_ten_factor_alpha_v1.py",
    "tests/test_csi300_ten_factor_alpha_v1.py",
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
    budget = contract["factor_budget"]
    if budget["used"] != 10 or budget["maximum"] != 10:
        raise ValueError("本协议必须固定使用10个因子且上限为10")
    if len(budget["feature_columns"]) != 10 or len(set(budget["feature_columns"])) != 10:
        raise ValueError("十因子列必须恰好10个且不重复")
    model = contract["model"]
    if model["hyperparameter_search_allowed"] or model["feature_selection_allowed"] or model["interaction_search_allowed"]:
        raise ValueError("冻结协议禁止模型搜索")
    if not contract["governance"]["failed_model_parameter_rescue_forbidden"]:
        raise ValueError("失败后必须禁止参数补救")
    unsafe = [
        key
        for key in (
            "paper_signal_enabled",
            "position_mapping_enabled",
            "order_generation_enabled",
            "broker_connection_enabled",
            "live_trading_authorized",
        )
        if contract["governance"].get(key) is True
    ]
    if unsafe:
        raise ValueError(f"安全开关非法启用：{unsafe}")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    inputs = contract["inputs"]
    file_inputs = {
        key: value for key, value in inputs.items() if key != "component_history_cache"
    }
    missing_inputs = [value for value in file_inputs.values() if not (ROOT / value).exists()]
    if missing_inputs:
        raise FileNotFoundError(f"输入缺失：{missing_inputs}")
    history_dir = ROOT / inputs["component_history_cache"]
    history_files = sorted(history_dir.glob("*.parquet"))
    if len(history_files) != 493:
        raise ValueError(f"完整证券历史文件数不是493：{len(history_files)}")
    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_csi300_ten_factor_alpha_v1.py",
            "tests/test_small_account_cross_sectional.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError("冻结前测试失败：\n" + test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "state": "FROZEN_BEFORE_OUTCOME_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "historical_contamination_cutoff": contract["protocol"]["historical_contamination_cutoff"],
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "input_files": {value: sha256(ROOT / value) for value in file_inputs.values()},
        "component_history_cache": tree_sha256(history_dir),
        "component_history_file_count": len(history_files),
        "test_command": f"{sys.executable} -m pytest tests/test_csi300_ten_factor_alpha_v1.py tests/test_small_account_cross_sectional.py -q",
        "test_result": test.stdout.strip(),
        "governance": contract["governance"],
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"状态": manifest["state"], "因子": 10, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

