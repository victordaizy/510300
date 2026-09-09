"""在股票母表与任何收益下载前冻结哈希盲测全协议。"""

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
CONFIG_FILE = ROOT / "config" / "a_share_hash_holdout_alpha_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "a_share_hash_holdout_alpha_v1_protocol_manifest.json"
FROZEN_FILES = (
    "config/a_share_hash_holdout_alpha_v1.yaml",
    "docs/A_SHARE_HASH_HOLDOUT_ALPHA_V1_SPEC.md",
    "research/a_share_hash_holdout_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/download_a_share_hash_holdout_training_v1.py",
    "scripts/train_a_share_hash_holdout_alpha_v1.py",
    "scripts/freeze_a_share_hash_holdout_protocol_v1.py",
    "scripts/freeze_a_share_hash_holdout_model_v1.py",
    "scripts/download_a_share_hash_holdout_evaluation_v1.py",
    "scripts/run_a_share_hash_holdout_evaluation_v1.py",
    "tests/test_a_share_hash_holdout_alpha_v1.py",
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
    if contract["factor_budget"]["used"] != 10 or contract["factor_budget"]["maximum"] != 10:
        raise ValueError("双重盲测必须恰好10因子")
    if contract["model"]["hyperparameter_search_allowed"] or not contract["model"]["holdout_paths_forbidden_during_training"]:
        raise ValueError("禁止调参且训练阶段必须隔离盲测路径")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    paths = contract["paths"]
    forbidden = [paths[key] for key in ("master", "training_panel", "training_benchmark", "holdout_panel", "holdout_benchmark", "trained_model", "result_json")]
    existing = [name for name in forbidden if (ROOT / name).exists()]
    if existing:
        raise RuntimeError(f"协议冻结前已有母表、收益、模型或结果：{existing}")
    training_source = (ROOT / "scripts/train_a_share_hash_holdout_alpha_v1.py").read_text(encoding="utf-8")
    if "holdout_panel" in training_source or "holdout_checkpoint_directory" in training_source:
        raise RuntimeError("训练脚本引用盲测收益路径")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_a_share_hash_holdout_alpha_v1.py", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": contract["protocol"]["project_id"], "state": "FROZEN_BEFORE_STOCK_MASTER_AND_RETURNS",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "training_script_holdout_path_reference": False, "test_result": test.stdout.strip(),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "因子": 10, "盲测路径隔离": True, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
