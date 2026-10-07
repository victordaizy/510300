"""打包实际训练产物，并在新解压目录复算保存结果。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml
import threadpoolctl

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/510300_absorption_available_members_training_v1"
TARGET = ROOT / "deliverables/510300_已有成分吸收率实际训练_V1_GPT审阅_20260922.zip"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    temporary = TARGET.with_suffix(".building.zip")
    extracted = REPORT / "verification/fresh_extract"
    require(not TARGET.exists() and not temporary.exists() and not extracted.exists(), "交付已存在，不覆盖。")
    summary = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    require(summary["prequential_ridge_fits"] == 393 and summary["account_scenarios"] == 20, "训练产物计数不同。")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "scipy": scipy.__version__, "PyYAML": yaml.__version__, "threadpoolctl": threadpoolctl.__version__}
    write_json(REPORT / "runtime.json", runtime)
    (REPORT / "requirements.txt").write_text("\n".join(f"{k}=={runtime[k]}" for k in ["numpy", "pandas", "pyarrow", "scipy", "PyYAML", "threadpoolctl"]) + "\ntzdata\n", encoding="utf-8")
    members = {p.relative_to(REPORT).as_posix(): p.read_bytes() for p in REPORT.rglob("*")
               if p.is_file() and "verification" not in p.relative_to(REPORT).parts and p.name != "delivery_receipt.json"}
    members["code/package_absorption_available_members_training_v1.py"] = Path(__file__).read_bytes()
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, data in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(data), "sha256": digest(data)})
    members["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "ZIP重复成员或CRC有误。")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) == {r["path"] for r in rows} | {"FILE_INDEX.csv"}, "成员集合与索引不同。")
        for row in rows:
            value = archive.read(row["path"])
            require(len(value) == int(row["bytes"]) and digest(value) == row["sha256"], "成员大小或哈希有误。")
        for name in names:
            destination = (extracted / name).resolve()
            require(destination.is_relative_to(extracted.resolve()), "成员路径超出解压目录。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    result = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "code/absorption_available_members_monthly_training_v1.py"), "verify", "--root", str(extracted)], check=True, capture_output=True, text=True, encoding="utf-8")
    recomputation = json.loads(result.stdout)
    write_json(REPORT / "verification/保存结果复算.json", recomputation)
    temporary.replace(TARGET)
    receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(TARGET),
               "bytes": TARGET.stat().st_size, "sha256": digest(TARGET.read_bytes()),
               "members": len(members), "indexed_files": len(members) - 1,
               "structural_checks": "PASS_CRC_DUPLICATES_INDEX_SIZE_HASH",
               "fresh_extraction_saved_output_recomputation": recomputation,
               "ridge_fits_including_final": 396, "mean_snapshots": 132, "entry_predictions": 16, "monthly_predictions": 508,
               "training_round_account_scenarios": 20,
               "external_review_completed": False, "independent_validation_established": False,
               "goal_achieved": False, "orders_authorized": False}
    write_json(REPORT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
