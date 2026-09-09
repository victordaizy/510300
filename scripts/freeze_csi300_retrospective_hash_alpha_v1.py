"""在计算代码留出组收益前冻结补充验证协议。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "csi300_retrospective_hash_alpha_v1.yaml"
FROZEN_FILES = (
    "config/csi300_retrospective_hash_alpha_v1.yaml",
    "docs/CSI300_RETROSPECTIVE_HASH_ALPHA_V1_SPEC.md",
    "research/csi300_retrospective_hash_alpha_v1.py",
    "scripts/freeze_csi300_retrospective_hash_alpha_v1.py",
    "scripts/train_csi300_retrospective_hash_alpha_v1.py",
    "scripts/run_csi300_retrospective_hash_alpha_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    if manifest_path.exists():
        raise FileExistsError(f"协议清单已经存在，禁止覆盖：{manifest_path}")
    input_files = list(contract["inputs"].values())
    missing = [name for name in (*FROZEN_FILES, *input_files) if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"冻结输入缺失：{missing}")
    manifest = {
        "state": "FROZEN_BEFORE_HASH_SUBGROUP_FEATURE_AND_RETURN_EVALUATION",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "evidence_class": contract["protocol"]["evidence_class"],
        "permanent_blind_claim_allowed": False,
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "input_files": {name: sha256(ROOT / name) for name in input_files},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
