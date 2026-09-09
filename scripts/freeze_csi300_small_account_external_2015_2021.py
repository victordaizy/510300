"""在读取早期外部时段策略收益前冻结父公式、执行和新数据。"""

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
CONFIG_FILE = ROOT / "config" / "csi300_small_account_external_2015_2021.yaml"
MANIFEST_FILE = ROOT / "config" / "csi300_small_account_external_2015_2021_manifest.json"
FROZEN_FILES = (
    "config/csi300_small_account_external_2015_2021.yaml",
    "docs/CSI300_SMALL_ACCOUNT_EXTERNAL_2015_2021_SPEC.md",
    "scripts/freeze_csi300_small_account_external_2015_2021.py",
    "scripts/run_csi300_small_account_external_2015_2021.py",
    "tests/test_csi300_small_account_external_2015_2021.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if contract["factor_budget"]["used"] > contract["factor_budget"]["maximum"]:
        raise ValueError("外部验证因子超过10个")
    if not contract["factor_budget"]["parent_formula_unchanged"]:
        raise ValueError("外部验证必须保持父公式不变")
    if contract["selection"]["parameter_search_allowed"]:
        raise ValueError("外部时段禁止参数搜索")
    if not contract["governance"]["parameter_rescue_after_result_forbidden"]:
        raise ValueError("外部结果后必须禁止参数补救")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    inputs = contract["inputs"]
    input_paths = list(inputs.values())
    missing_inputs = [value for value in input_paths if not (ROOT / value).exists()]
    if missing_inputs:
        raise FileNotFoundError(f"外部验证输入缺失：{missing_inputs}")
    data_status = json.loads(
        (ROOT / inputs["external_data_status"]).read_text(encoding="utf-8")
    )
    if data_status.get("status") != "PASS":
        raise RuntimeError("外部验证数据门不是PASS")
    if data_status.get("minimum_members_per_day") != 300 or data_status.get("maximum_members_per_day") != 300:
        raise RuntimeError("外部验证每日成员数不是300")
    if data_status.get("evaluation_missing_price_rows") != 0:
        raise RuntimeError("外部验证正式区间存在价格缺口")
    parent = json.loads((ROOT / inputs["parent_manifest"]).read_text(encoding="utf-8"))
    if parent.get("state") != "FROZEN_BEFORE_SMALL_ACCOUNT_RERUN":
        raise RuntimeError("父策略冻结清单状态错误")
    parent_config_hash = sha256(ROOT / inputs["parent_config"])
    if parent["frozen_files"].get(inputs["parent_config"]) != parent_config_hash:
        raise RuntimeError("父策略配置已偏离冻结哈希")
    parent_implementation = inputs["parent_factor_implementation"]
    if parent["input_files"].get(parent_implementation) != sha256(ROOT / parent_implementation):
        raise RuntimeError("父策略因子实现已偏离冻结哈希")
    parent_execution = inputs["parent_execution_implementation"]
    if parent["frozen_files"].get(parent_execution) != sha256(ROOT / parent_execution):
        raise RuntimeError("父策略执行实现已偏离冻结哈希")
    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_csi300_small_account_external_2015_2021.py",
            "tests/test_small_account_cross_sectional.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError("外部验证冻结测试失败：\n" + test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_EXTERNAL_PERIOD_RETURN_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "input_files": {value: sha256(ROOT / value) for value in input_paths},
        "parent_formula_unchanged": True,
        "data_gate": data_status,
        "test_result": test.stdout.strip(),
        "governance": contract["governance"],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "每日成员": 300, "价格缺口": 0, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

