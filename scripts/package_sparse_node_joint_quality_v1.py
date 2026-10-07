"""生成自含的节点联合质量交付包，并从新解压目录复算。"""
import csv
from datetime import datetime
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
REPORT = ROOT / "reports/research/510300_sparse_node_joint_quality_v1"
TARGET = ROOT / "deliverables/510300_少数节点胜率盈亏比夏普联合评价_V1_GPT审阅_20260922.zip"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    temporary = TARGET.with_suffix(".building.zip")
    extracted = REPORT / "verification/fresh_extract"
    require(not TARGET.exists() and not temporary.exists() and not extracted.exists(), "交付已存在，不覆盖。")
    for name in ["00_README_FIRST.md", "研究结论.md", "节点训练任务书.md", "用户需求.md", "GPT审阅提示词.md",
                 "指标口径.md", "EXCLUSIONS.md", "node_training_contract.json", "mandate_after.json", "authority_update.json"]:
        require((REPORT / name).is_file(), f"缺少交付文件：{name}")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    write_json(REPORT / "runtime.json", runtime)
    (REPORT / "requirements.txt").write_text("\n".join(f"{k}=={runtime[k]}" for k in ["numpy", "pandas", "pyarrow"]) + "\n", encoding="utf-8")
    members = {p.relative_to(REPORT).as_posix(): p.read_bytes() for p in REPORT.rglob("*")
               if p.is_file() and "verification" not in p.relative_to(REPORT).parts
               and "__pycache__" not in p.relative_to(REPORT).parts and p.name != "delivery_receipt.json"}
    for name in ["finish_sparse_node_joint_quality_v1.py", "package_sparse_node_joint_quality_v1.py"]:
        members["code/" + name] = (ROOT / "scripts" / name).read_bytes()
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, value in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(value), "sha256": digest(value)})
    members["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "ZIP存在重复或CRC错误。")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) == {r["path"] for r in rows} | {"FILE_INDEX.csv"}, "索引集合不同。")
        for row in rows:
            value = archive.read(row["path"])
            require(len(value) == int(row["bytes"]) and digest(value) == row["sha256"], "文件大小或哈希不同。")
        for name in names:
            destination = (extracted / name).resolve()
            require(destination.is_relative_to(extracted.resolve()), "解压路径超出交付目录。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    result = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "code/sparse_node_joint_quality_v1.py"),
                             "verify", "--root", str(extracted)], check=True, capture_output=True, text=True, encoding="utf-8")
    recomputation = json.loads(result.stdout)
    require(recomputation["account_metric_records"] == 170 and recomputation["saved_cycle_rows"] == 1496, "解压复算范围不同。")
    write_json(REPORT / "verification/保存账户联合指标复算.json", recomputation)
    temporary.replace(TARGET)
    receipt = {"completed_at": datetime.now().astimezone().isoformat(), "archive": str(TARGET),
               "bytes": TARGET.stat().st_size, "sha256": digest(TARGET.read_bytes()),
               "members": len(members), "indexed_files": len(members) - 1,
               "structural_checks": "PASS_CRC_DUPLICATES_INDEX_SIZE_HASH",
               "fresh_extraction_saved_recomputation": recomputation,
               "external_review_completed": False, "independent_validation_established": False,
               "new_fits": 0, "new_accounts": 0, "new_downloads": 0, "goal_achieved": False, "orders_authorized": False}
    write_json(REPORT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
