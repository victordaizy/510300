"""构建单一自包含研究ZIP并在新解压目录只读复核。"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_sequential_patterns_regime_v1"
DELIVERY = ROOT / "deliverables/510300_连续形态与状态启停_V1_GPT审阅_20260924.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if DELIVERY.exists():
        raise RuntimeError("同名最终包已存在，不覆盖已交付快照。")
    for name in ("package_sequential_patterns_regime_v1.py", "verify_sequential_patterns_delivery_v1.py", "finalize_sequential_patterns_regime_v1.py"):
        shutil.copy2(ROOT / "scripts" / name, STUDY / "code" / name)
    files = [p for p in STUDY.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in ("FILE_INDEX.csv", "delivery_receipt.json") and p.suffix != ".pyc"]
    files.sort(key=lambda p: p.relative_to(STUDY).as_posix())
    index = []
    for path in files:
        payload = path.read_bytes()
        index.append({"path": path.relative_to(STUDY).as_posix(), "bytes": len(payload), "sha256": sha(payload)})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(index)
    index_bytes = buffer.getvalue().encode("utf-8-sig")
    (STUDY / "FILE_INDEX.csv").write_bytes(index_bytes)
    build = DELIVERY.with_suffix(".building.zip")
    if build.exists():
        raise RuntimeError("已有building文件，需核实其来源后再继续，不能覆盖。")
    with zipfile.ZipFile(build, "x", zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for path, item in zip(files, index):
            z.write(path, item["path"])
        z.writestr("FILE_INDEX.csv", index_bytes)
    extraction = ROOT / "deliverables/verification" / ("sequential_patterns_v1_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    extraction.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(build) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names)) == len(index) + 1
        assert set(names) == {r["path"] for r in index} | {"FILE_INDEX.csv"}
        for item in index:
            raw = z.read(item["path"])
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
        z.extractall(extraction)
    result = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), str(extraction / "code/verify_sequential_patterns_delivery_v1.py"), "--root", str(extraction)],
                            capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, cwd=extraction)
    log = DELIVERY.with_suffix(".verification.txt")
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("新解压目录复算失败，保留building文件与日志。\n" + result.stdout + result.stderr)
    assert build.stat().st_size < 450000000
    build.replace(DELIVERY)
    receipt = {"packaged_at": datetime.now().astimezone().isoformat(), "path": str(DELIVERY),
               "bytes": DELIVERY.stat().st_size, "sha256": sha(DELIVERY.read_bytes()),
               "zip_members": len(index) + 1, "indexed_members": len(index),
               "crc": "PASS", "duplicates": "NONE", "index_size_hash": "PASS",
               "fresh_extraction": str(extraction), "saved_recomputation": "PASS",
               "verification_log": str(log), "new_accounts_during_verification": 0,
               "new_fits_during_verification": 0, "new_random_samples": 0, "downloads": 0,
               "current_view": "NO_VIEW", "strict_forward_observations": 0,
               "external_review_completed": False, "goal_achieved": False}
    DELIVERY.with_suffix(".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
