"""在读取开发期收益前冻结风险中性偏度迁移协议。"""

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
CONFIG = ROOT / "config" / "510300_option_smirk_gamma1_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_smirk_gamma1_v1_manifest.json"
FROZEN_FILES = (
    "config/510300_option_smirk_gamma1_v1.yaml",
    "research/option_smirk_gamma1_v1.py",
    "scripts/freeze_510300_option_smirk_gamma1_v1.py",
    "tests/test_510300_option_smirk_gamma1_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "510300_OPTION_SMIRK_GAMMA1_V1":
        raise ValueError("偏度协议ID漂移")
    if int(config["protocol"]["factor_count"]) != 1:
        raise ValueError("V1必须且只能包含一个偏度因子")
    if int(config["protocol"]["factor_count"]) > 10:
        raise ValueError("因子数超过用户限定")
    if config["dates"]["validation_or_holdout_read_authorized"]:
        raise ValueError("开发协议不得读取验证或最终留出")
    if config["governance"]["parameter_search"] != "FORBIDDEN":
        raise ValueError("偏度迁移禁止参数搜索")
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != str(item["sha256"]).lower():
            raise RuntimeError(f"冻结输入不存在或哈希不一致：{name}")
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.is_file():
        raise RuntimeError("偏度迁移协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("偏度迁移冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"偏度迁移冻结文件哈希漂移：{drift}")
    return config, manifest


def main() -> int:
    config = _load_and_assert()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"偏度迁移冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("风险中性偏度Gamma1 V1已冻结，全部哈希一致。")
        return 0
    for output in (config["paths"]["result_json"], config["paths"]["result_markdown"]):
        if (ROOT / output).exists():
            raise RuntimeError("开发结果已存在，禁止事后冻结偏度规则")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_510300_option_smirk_gamma1_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"偏度迁移冻结前测试失败：\n{detail}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "development_outcomes_read_at_freeze": False,
        "validation_or_holdout_read_authorized": False,
        "factor_count": config["protocol"]["factor_count"],
        "input_hashes": {name: item["sha256"] for name, item in config["inputs"].items()},
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": config["governance"],
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
