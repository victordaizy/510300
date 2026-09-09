"""在下载冻结资产池行情前冻结协议与实现。"""

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
CONFIG_FILE = ROOT / "config" / "csi300_etf_rotation_alpha_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "csi300_etf_rotation_alpha_v1_protocol_manifest.json"
FROZEN_FILES = (
    "config/csi300_etf_rotation_alpha_v1.yaml",
    "docs/CSI300_ETF_ROTATION_ALPHA_V1_SPEC.md",
    "research/csi300_etf_rotation_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/download_csi300_etf_rotation_v1.py",
    "scripts/freeze_csi300_etf_rotation_protocol_v1.py",
    "scripts/run_csi300_etf_rotation_alpha_v1.py",
    "tests/test_csi300_etf_rotation_alpha_v1.py",
    "tests/test_small_account_cross_sectional.py",
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
        raise ValueError("因子超过10个")
    if contract["formula"]["hyperparameter_search_allowed"]:
        raise ValueError("冻结候选禁止搜索参数")
    if abs(sum(map(float, contract["formula"]["weights"].values())) - 1.0) > 1e-12:
        raise ValueError("因子权重之和必须为1")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    output_paths = [contract["inputs"]["etf_panel"], contract["inputs"]["benchmark"], contract["outputs"]["result_json"]]
    existing = [name for name in output_paths if (ROOT / name).exists()]
    if existing:
        raise RuntimeError(f"首次冻结前已经存在候选行情或结果：{existing}")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_csi300_etf_rotation_alpha_v1.py", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_MARKET_DATA_DOWNLOAD",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "test_result": test.stdout.strip(),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "因子数": contract["factor_budget"]["used"], "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
