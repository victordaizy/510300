"""在桶2未来收益下载前冻结单因子低波动公式。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.freeze_a_share_bucket1_time_holdout_formula_v1 as implementation  # noqa: E402


CONFIG_FILE = ROOT / "config" / "a_share_bucket2_time_holdout_lowvol_v1.yaml"
FROZEN_FILES = (
    "config/a_share_bucket2_time_holdout_lowvol_v1.yaml",
    "docs/A_SHARE_BUCKET2_TIME_HOLDOUT_LOWVOL_V1_SPEC.md",
    "research/a_share_hash_holdout_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/freeze_a_share_bucket1_time_holdout_formula_v1.py",
    "scripts/download_a_share_bucket1_time_holdout_formula_v1.py",
    "scripts/run_a_share_bucket1_time_holdout_formula_v1.py",
    "scripts/freeze_a_share_bucket2_time_holdout_lowvol_v1.py",
    "scripts/download_a_share_bucket2_time_holdout_lowvol_v1.py",
    "scripts/run_a_share_bucket2_time_holdout_lowvol_v1.py",
    "tests/test_a_share_hash_holdout_alpha_v1.py",
    "tests/test_small_account_cross_sectional.py",
)


def sha256(path: Path) -> str:
    return implementation.sha256(path)


def tree_sha256(directory: Path) -> str:
    return implementation.tree_sha256(directory)


def verify_protocol() -> tuple[dict, dict]:
    implementation.CONFIG_FILE = CONFIG_FILE
    implementation.FROZEN_FILES = FROZEN_FILES
    contract, manifest = implementation.verify_protocol()
    if manifest.get("future_holdout_bucket") != 2:
        raise RuntimeError("桶2协议清单的未来留出桶错误")
    return contract, manifest


def main() -> int:
    implementation.CONFIG_FILE = CONFIG_FILE
    implementation.FROZEN_FILES = FROZEN_FILES
    result = implementation.main()
    contract = __import__("yaml").safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["future_holdout_bucket"] = 2
    manifest.pop("future_bucket1_returns_downloaded_before_freeze", None)
    manifest["future_bucket2_returns_downloaded_before_freeze"] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
