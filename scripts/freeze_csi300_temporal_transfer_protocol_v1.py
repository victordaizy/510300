"""在读取早期个股未来收益前冻结时间迁移协议与两阶段实现。"""

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
CONFIG_FILE = ROOT / "config" / "csi300_temporal_transfer_alpha_v1.yaml"
PROTOCOL_MANIFEST = ROOT / "config" / "csi300_temporal_transfer_alpha_v1_protocol_manifest.json"
FROZEN_FILES = (
    "config/csi300_temporal_transfer_alpha_v1.yaml",
    "docs/CSI300_TEMPORAL_TRANSFER_ALPHA_V1_SPEC.md",
    "research/csi300_temporal_transfer_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/freeze_csi300_temporal_transfer_protocol_v1.py",
    "scripts/train_csi300_temporal_transfer_alpha_v1.py",
    "scripts/freeze_csi300_temporal_transfer_model_v1.py",
    "scripts/run_csi300_temporal_transfer_evaluation_v1.py",
    "tests/test_csi300_temporal_transfer_alpha_v1.py",
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
        raise ValueError("时间迁移必须恰好10个因子")
    if contract["model"]["hyperparameter_search_allowed"]:
        raise ValueError("时间迁移禁止超参数搜索")
    if not contract["model"]["recent_label_read_during_training_forbidden"]:
        raise ValueError("训练阶段必须禁止读取近期标签")
    if not contract["governance"]["trained_model_freeze_required_before_evaluation"]:
        raise ValueError("评估前必须二次冻结模型")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"协议冻结文件缺失：{missing}")
    training_inputs = list(contract["training_inputs"].values())
    missing_inputs = [value for value in training_inputs if not (ROOT / value).exists()]
    if missing_inputs:
        raise FileNotFoundError(f"训练输入缺失：{missing_inputs}")
    training_source = (ROOT / "scripts/train_csi300_temporal_transfer_alpha_v1.py").read_text(encoding="utf-8")
    if "evaluation_inputs" in training_source:
        raise RuntimeError("训练脚本不得引用evaluation_inputs")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_csi300_temporal_transfer_alpha_v1.py", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if test.returncode != 0:
        raise RuntimeError("协议冻结测试失败：\n" + test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_TRAINING_OUTCOME_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "training_inputs": {value: sha256(ROOT / value) for value in training_inputs},
        "training_script_recent_input_reference": False,
        "test_result": test.stdout.strip(),
    }
    PROTOCOL_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "因子": 10, "训练脚本读取近期输入": False, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

