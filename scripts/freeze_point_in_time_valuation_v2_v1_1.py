"""冻结价格补采与点时估值V2 V1.1审计证据。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.point_in_time_valuation_v2_audit import canonical_directory_hash, sha256_file
from research.point_in_time_valuation_v2_post_acquisition import (
    load_config,
    verify_hashes,
)


FROZEN_FILES = (
    "config/point_in_time_valuation_price_backfill.yaml",
    "research/point_in_time_valuation_price_acquisition.py",
    "scripts/download_point_in_time_valuation_prices.py",
    "config/point_in_time_valuation_v2_audit_v1_1.yaml",
    "research/point_in_time_valuation_v2_post_acquisition.py",
    "scripts/audit_point_in_time_valuation_v2_v1_1.py",
    "scripts/freeze_point_in_time_valuation_v2_v1_1.py",
    "tests/test_point_in_time_valuation_price_acquisition.py",
)

OUTPUT_FILES = (
    "data/raw/constituents/000300_historical_weight_snapshot_prices.parquet",
    "reports/data_quality/000300_valuation_price_backfill.json",
    "reports/data_quality/000300_valuation_price_backfill.md",
    "data/audit/000300_point_in_time_valuation_v2_snapshot_coverage_v1_1.parquet",
    "reports/data_quality/000300_point_in_time_valuation_v2_reconstructibility_v1_1.json",
    "reports/data_quality/000300_point_in_time_valuation_v2_reconstructibility_v1_1.md",
)


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m", "pytest",
        "tests/test_point_in_time_valuation_price_acquisition.py",
        "tests/test_point_in_time_valuation_v2_audit.py",
        "-q",
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"冻结前测试失败：\n{result.stdout}\n{result.stderr}")
    return {"command": command, "return_code": 0, "stdout": result.stdout.strip()}


def main() -> int:
    config = load_config()
    if verify_hashes(config)["status"] != "PASS":
        raise RuntimeError("V1.1输入哈希不一致")
    missing = [path for path in (*FROZEN_FILES, *OUTPUT_FILES) if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    acquisition = json.loads(
        (ROOT / "reports/data_quality/000300_valuation_price_backfill.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (ROOT / config["artifacts"]["report_json"]).read_text(encoding="utf-8")
    )
    if acquisition["status"] != "PASS_FIVE_YEAR_SNAPSHOT_PRICES":
        raise RuntimeError("价格补采未通过")
    if audit["branch_status"]["VAL01_RAW_EY_5Y"] != "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY":
        raise RuntimeError("五年原始EY数据闸门未通过")
    if audit["branch_status"]["VAL01_NORM_EY_5Y"] == "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY":
        raise RuntimeError("标准化EY被意外放行")
    if any(audit["governance"].values()):
        raise RuntimeError("审计治理边界异常")
    cache_dir = ROOT / "data/raw/constituents/.tushare_valuation_price_cache"
    cache_hash, cache_files = canonical_directory_hash(cache_dir)
    if len(cache_files) != 437:
        raise RuntimeError(f"价格缓存应为437个文件，实际{len(cache_files)}")
    tests = _run_tests()
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "prior_v1_0_manifest": {
            "file": "config/point_in_time_valuation_v2_audit_manifest.json",
            "sha256": sha256_file(ROOT / "config/point_in_time_valuation_v2_audit_manifest.json"),
            "mutated": False,
        },
        "frozen_files": {path: sha256_file(ROOT / path) for path in FROZEN_FILES},
        "output_files": {path: sha256_file(ROOT / path) for path in OUTPUT_FILES},
        "cache": {
            "directory": "data/raw/constituents/.tushare_valuation_price_cache",
            "file_count": len(cache_files),
            "content_sha256": cache_hash,
        },
        "status_at_freeze": {
            "price_acquisition": acquisition["status"],
            "price_minimum_weight_coverage": acquisition["snapshot_validation"]["minimum_price_weight_coverage"],
            "price_missing_rows": acquisition["snapshot_validation"]["missing_price_row_count"],
            "price_future_rows": acquisition["snapshot_validation"]["future_price_row_count"],
            "cross_source_exact_match_ratio": acquisition["cross_source_overlap"]["exact_match_ratio"],
            **audit["branch_status"],
        },
        "governance": audit["governance"],
        "test_result": tests,
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    manifest = ROOT / config["artifacts"]["manifest"]
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
