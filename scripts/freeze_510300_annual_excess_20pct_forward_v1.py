"""冻结或验证 510300 年化净超额 20%前瞻目标。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "510300_annual_excess_20pct_forward_v1_manifest.json"
TRACKED_FILES = [
    "config/510300_annual_excess_20pct_forward_v1.yaml",
    "docs/510300_ANNUAL_EXCESS_20PCT_FORWARD_V1_SPEC.md",
    "research/annual_excess_20pct_forward_v1.py",
    "scripts/build_510300_annual_excess_20pct_forward_v1.py",
    "scripts/freeze_510300_annual_excess_20pct_forward_v1.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    return {relative: sha256(ROOT / relative) for relative in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze() -> dict:
    files = current_hashes()
    payload = {
        "schema_version": "1.0.0",
        "manifest_id": "510300_ANNUAL_EXCESS_20PCT_FORWARD_V1_MANIFEST",
        "status": "FROZEN_BEFORE_2026_08_27_FORWARD_WINDOW",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "target_annualized_net_excess": 0.20,
        "tracked_file_count": len(files),
        "tracked_files": files,
        "content_sha256": content_hash(files),
        "tests_are_runtime_inputs": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing.get("tracked_files") != files:
            raise RuntimeError("冻结清单已存在且内容不同，禁止覆盖")
        return existing
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def verify() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError("目标冻结清单尚未创建")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live = current_hashes()
    failures = []
    if manifest.get("status") != "FROZEN_BEFORE_2026_08_27_FORWARD_WINDOW":
        failures.append("status")
    if float(manifest.get("target_annualized_net_excess", -1.0)) != 0.20:
        failures.append("target_annualized_net_excess")
    if manifest.get("tracked_files") != live:
        failures.append("tracked_files")
    if manifest.get("content_sha256") != content_hash(live):
        failures.append("content_sha256")
    return {
        "status": "PASS_20PCT_TARGET_MANIFEST_VERIFIED" if not failures else "FAILED_20PCT_TARGET_MANIFEST",
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(MANIFEST),
        "content_sha256": content_hash(live),
        "tracked_file_count": len(live),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证510300年化净超额20%目标")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.mode == "verify" and result["failure_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

