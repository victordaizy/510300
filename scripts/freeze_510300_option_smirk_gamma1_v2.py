"""在V2首次产生交易结果前冻结唯一的数据覆盖修正。"""

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
CONFIG = ROOT / "config" / "510300_option_smirk_gamma1_v2.yaml"
MANIFEST = ROOT / "config" / "510300_option_smirk_gamma1_v2_manifest.json"
V1_RESULT = ROOT / "reports" / "discovery" / "510300_option_smirk_gamma1_v1_dev.json"
FROZEN_FILES = (
    "config/510300_option_smirk_gamma1_v1.yaml",
    "research/option_smirk_gamma1_v1.py",
    "config/510300_option_smirk_gamma1_v2.yaml",
    "research/option_smirk_gamma1_v2.py",
    "scripts/freeze_510300_option_smirk_gamma1_v2.py",
    "tests/test_510300_option_smirk_gamma1_v2.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> tuple[dict, dict]:
    override = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if override["protocol"]["project_id"] != "510300_OPTION_SMIRK_GAMMA1_V2":
        raise ValueError("偏度V2协议ID漂移")
    if not override["protocol"]["correction_only"]:
        raise ValueError("V2只能是覆盖修正")
    if int(override["protocol"]["factor_count"]) != 1:
        raise ValueError("V2不得改变因子数")
    if override["governance"]["all_non_coverage_parameters_changed_from_v1"]:
        raise ValueError("V2不得改变非覆盖参数")
    if override["governance"]["parameter_search"] != "FORBIDDEN":
        raise ValueError("V2禁止参数搜索")
    if not V1_RESULT.is_file():
        raise FileNotFoundError("V1结果不存在，无法证明零交易覆盖故障")
    prior = json.loads(V1_RESULT.read_text(encoding="utf-8"))
    if int(prior.get("executed_trades", -1)) != 0:
        raise RuntimeError("V1已经读取实际交易结果，不允许用V2修正规则")
    return override, prior


def verify_protocol() -> tuple[dict, dict]:
    override, _ = _load_and_assert()
    if not MANIFEST.is_file():
        raise RuntimeError("偏度V2协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("偏度V2冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"偏度V2冻结文件哈希漂移：{drift}")
    from research.option_smirk_gamma1_v2 import load_config

    return load_config(), manifest


def main() -> int:
    override, prior = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"偏度V2冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("风险中性偏度Gamma1 V2已冻结，全部哈希一致。")
        return 0
    for output in (override["paths"]["result_json"], override["paths"]["result_markdown"]):
        if (ROOT / output).exists():
            raise RuntimeError("V2结果已存在，禁止事后冻结覆盖修正")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_smirk_gamma1_v2.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"偏度V2冻结前测试失败：\n{detail}")
    payload = {
        "project_id": override["protocol"]["project_id"],
        "state": override["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "prior_version_executed_trades_at_freeze": prior["executed_trades"],
        "development_trade_outcomes_read_at_freeze": False,
        "validation_or_holdout_read_authorized": False,
        "correction_only": True,
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": override["governance"],
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
