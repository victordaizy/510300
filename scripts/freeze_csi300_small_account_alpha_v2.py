"""在运行2万元历史重缩放前冻结规则与实现。"""

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
CONFIG_FILE = ROOT / "config" / "csi300_small_account_alpha_v2.yaml"
MANIFEST_FILE = ROOT / "config" / "csi300_small_account_alpha_v2_manifest.json"
FROZEN_FILES = (
    "config/csi300_small_account_alpha_v2.yaml",
    "docs/CSI300_SMALL_ACCOUNT_ALPHA_V2_SPEC.md",
    "research/small_account_cross_sectional.py",
    "scripts/freeze_csi300_small_account_alpha_v2.py",
    "scripts/run_csi300_small_account_alpha_v2.py",
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
    files = sorted(directory.glob("*.parquet"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if config["factor_budget"]["used"] > config["factor_budget"]["maximum"]:
        raise ValueError("因子数量超过10个硬上限")
    if config["selection"]["parameter_search_allowed"]:
        raise ValueError("历史重缩放禁止参数搜索")
    if config["account"]["minimum_trade_notional_cny"] != 5000:
        raise ValueError("单笔最低成交额必须为5000元")
    if config["account"]["initial_cash_cny"] != 20000:
        raise ValueError("初始资金必须为20000元")
    if config["governance"]["historical_result_label"] != "HISTORICALLY_CONTAMINATED_RETROSPECTIVE":
        raise ValueError("历史污染标签不得移除")
    unsafe = [
        key
        for key in (
            "paper_signal_enabled",
            "position_mapping_enabled",
            "order_generation_enabled",
            "broker_connection_enabled",
            "live_trading_authorized",
        )
        if config["governance"].get(key) is True
    ]
    if unsafe:
        raise ValueError(f"安全开关非法启用：{unsafe}")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    inputs = config["inputs"]
    file_inputs = {
        key: value for key, value in inputs.items() if key != "component_history_cache"
    }
    missing_inputs = [value for value in file_inputs.values() if not (ROOT / value).exists()]
    if missing_inputs:
        raise FileNotFoundError(f"输入文件缺失：{missing_inputs}")
    history_dir = ROOT / inputs["component_history_cache"]
    history_files = sorted(history_dir.glob("*.parquet"))
    if len(history_files) != 493:
        raise ValueError(f"完整证券历史必须恰有493个文件，实际{len(history_files)}")

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError("冻结前测试失败：\n" + test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "state": "FROZEN_BEFORE_SMALL_ACCOUNT_RERUN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "historical_contamination_cutoff": config["protocol"]["historical_contamination_cutoff"],
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "input_files": {value: sha256(ROOT / value) for value in file_inputs.values()},
        "component_history_cache": tree_sha256(history_dir),
        "component_history_file_count": len(history_files),
        "test_command": f"{sys.executable} -m pytest tests/test_small_account_cross_sectional.py -q",
        "test_result": test.stdout.strip(),
        "governance": config["governance"],
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"状态": manifest["state"], "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

