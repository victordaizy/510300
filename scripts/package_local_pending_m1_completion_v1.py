"""打包本地待评价事件的补齐结果，并从新解压目录复算。"""
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

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/510300_local_pending_m1_completion_v1"
TARGET = ROOT / "deliverables/510300_待评价月份历史模型输出补齐_V1_1_GPT审阅_20260922.zip"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    temporary = TARGET.with_suffix(".building.zip")
    extracted = REPORT / "verification/fresh_extract_v1_1"
    require(not TARGET.exists() and not temporary.exists() and not extracted.exists(), "交付已存在，不覆盖。")
    summary = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    require(summary["reconstructed_events"] == 1 and summary["saved_model_outputs"] == 3 and summary["current_target_net_sharpe"] == 1.3, "本轮记录或目标不同。")
    for name in ["研究结论.md", "00_README_FIRST.md", "用户需求.md", "GPT审阅提示词.md", "EXCLUSIONS.md", "goal_continuation_state.json", "local_runtime_observation.json", "delivery_revision.json"]:
        require((REPORT / name).is_file(), f"缺少完整交付文件：{name}")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    write_json(REPORT / "runtime.json", runtime)
    (REPORT / "requirements.txt").write_text("\n".join(f"{k}=={runtime[k]}" for k in ["numpy", "pandas", "pyarrow"]) + "\ntzdata\n", encoding="utf-8")
    members = {p.relative_to(REPORT).as_posix(): p.read_bytes() for p in REPORT.rglob("*")
               if p.is_file() and "verification" not in p.relative_to(REPORT).parts
               and "__pycache__" not in p.relative_to(REPORT).parts
               and p.relative_to(REPORT).as_posix() != "delivery_receipt.json"}
    for name in ["package_local_pending_m1_completion_v1.py", "finish_local_pending_m1_completion_v1.py"]:
        members["code/" + name] = (ROOT / "scripts" / name).read_bytes()
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
    result = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "code/local_pending_m1_completion_v1.py"), "verify", "--root", str(extracted)], check=True, capture_output=True, text=True, encoding="utf-8")
    recomputation = json.loads(result.stdout)
    require(recomputation["old_events_reproduced"] == 16 and recomputation["saved_model_outputs"] == 3, "解压复算记录不一致。")
    write_json(REPORT / "verification/保存结果复算.json", recomputation)
    temporary.replace(TARGET)
    receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(TARGET),
               "bytes": TARGET.stat().st_size, "sha256": digest(TARGET.read_bytes()),
               "members": len(members), "indexed_files": len(members) - 1,
               "structural_checks": "PASS_CRC_DUPLICATES_INDEX_SIZE_HASH",
               "fresh_extraction_saved_output_recomputation": recomputation,
               "reconstructed_events": 1, "saved_model_outputs": 3, "new_independent_forward_predictions": 0,
               "target_net_sharpe": 1.3, "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0,
               "external_review_completed": False, "independent_validation_established": False,
               "goal_achieved": False, "orders_authorized": False}
    write_json(REPORT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
