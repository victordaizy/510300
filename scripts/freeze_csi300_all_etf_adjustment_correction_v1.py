"""冻结全ETF V1的数据修复；公式必须与原冻结候选逐字一致。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MANIFEST = ROOT / "config" / "csi300_all_etf_momentum_alpha_v1_protocol_manifest.json"
CORRECTION_MANIFEST = ROOT / "config" / "csi300_all_etf_momentum_alpha_v1r_correction_manifest.json"
CORRECTED_PANEL = ROOT / "data" / "raw" / "all_etf_momentum_v1r" / "etf_total_return_panel_tushare_adj.parquet"
CORRECTION_STATUS = ROOT / "reports" / "data_quality" / "csi300_all_etf_momentum_v1r_status.json"
CORRECTION_RESULT = ROOT / "reports" / "backtest" / "csi300_all_etf_momentum_alpha_v1r.json"
FROZEN_FILES = (
    "config/csi300_all_etf_momentum_alpha_v1.yaml",
    "research/csi300_all_etf_momentum_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "docs/CSI300_ALL_ETF_MOMENTUM_V1_ADJUSTMENT_POSTMORTEM.md",
    "scripts/download_csi300_all_etf_fund_adj_correction_v1.py",
    "scripts/freeze_csi300_all_etf_adjustment_correction_v1.py",
    "scripts/run_csi300_all_etf_momentum_alpha_v1r.py",
    "scripts/run_csi300_etf_rotation_alpha_v1.py",
    "tests/test_csi300_all_etf_momentum_alpha_v1.py",
    "tests/test_small_account_cross_sectional.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    original = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    if original.get("state") != "FROZEN_BEFORE_FUND_MASTER_AND_RETURN_DOWNLOAD":
        raise RuntimeError("原候选冻结清单无效")
    original_formula_files = (
        "config/csi300_all_etf_momentum_alpha_v1.yaml",
        "research/csi300_all_etf_momentum_alpha_v1.py",
        "research/small_account_cross_sectional.py",
        "tests/test_csi300_all_etf_momentum_alpha_v1.py",
        "tests/test_small_account_cross_sectional.py",
    )
    changed_formula = [name for name in original_formula_files if sha256(ROOT / name) != original["frozen_files"][name]]
    if changed_formula:
        raise RuntimeError(f"原冻结公式或执行器已变化：{changed_formula}")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"修复冻结文件缺失：{missing}")
    existing = [str(path) for path in (CORRECTED_PANEL, CORRECTION_STATUS, CORRECTION_RESULT) if path.exists()]
    if existing:
        raise RuntimeError(f"修复冻结前已经存在完整修复数据或结果：{existing}")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_csi300_all_etf_momentum_alpha_v1.py", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + "\n" + test.stderr)
    manifest = {
        "project_id": "CSI300_ALL_ETF_MOMENTUM_ALPHA_V1R_DATA_CORRECTED",
        "state": "FROZEN_DATA_CORRECTION_ONLY_FORMULA_UNCHANGED",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "original_manifest_sha256": sha256(ORIGINAL_MANIFEST),
        "formula_files_equal_original_freeze": True,
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "test_result": test.stdout.strip(),
    }
    CORRECTION_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "公式未变": True, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
