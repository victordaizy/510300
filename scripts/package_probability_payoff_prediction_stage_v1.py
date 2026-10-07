"""打包预测阶段实际训练及新解压保存结果复算。"""
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
import scipy
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_probability_payoff_prediction_stage_v1"
TARGET = ROOT / "deliverables/510300_先训练概率与盈亏幅度_V1_GPT审阅_20260924.zip"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    building, extracted = TARGET.with_suffix(".building.zip"), OUT / "verification/fresh_extract"
    require(not TARGET.exists() and not building.exists() and not extracted.exists(), "已有交付，不覆盖。")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    require(summary["new_conditional_mean_fits"] == 444 and summary["prediction_rows"] == 660, "实际训练范围不同。")
    for name in ["00_README_FIRST.md", "训练结果.md", "stage_decision.json", "authority_update.json", "用户要求.md", "GPT审阅提示词.md", "EXCLUSIONS.md"]:
        require((OUT / name).is_file(), f"交付文件缺失：{name}")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": scipy.__version__, "pyarrow": pyarrow.__version__, "PyYAML": yaml.__version__}
    save(OUT / "runtime.json", runtime)
    (OUT / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in runtime.items() if k != "python") + "\ntzdata\n", encoding="utf-8")
    members = {p.relative_to(OUT).as_posix(): p.read_bytes() for p in OUT.rglob("*")
               if p.is_file() and "verification" not in p.relative_to(OUT).parts
               and "__pycache__" not in p.relative_to(OUT).parts and p.name != "delivery_receipt.json"}
    for name in ["finish_probability_payoff_prediction_stage_v1.py", "package_probability_payoff_prediction_stage_v1.py"]:
        members["code/" + name] = (ROOT / "scripts" / name).read_bytes()
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, value in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(value), "sha256": digest(value)})
    members["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "重复成员或CRC错误。")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) == {r["path"] for r in rows} | {"FILE_INDEX.csv"}, "索引集合不同。")
        for row in rows:
            value = archive.read(row["path"])
            require(len(value) == int(row["bytes"]) and digest(value) == row["sha256"], "索引大小或哈希不同。")
        for name in names:
            destination = (extracted / name).resolve()
            require(destination.is_relative_to(extracted.resolve()), "解压目录超出范围。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    result = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "code/probability_payoff_prediction_stage_v1.py"),
                             "verify", "--root", str(extracted)], check=True, capture_output=True, text=True, encoding="utf-8")
    verification = json.loads(result.stdout)
    require(verification["status"] == "PASS_SAVED_PREDICTION_STAGE_RECOMPUTATION", "保存预测复算失败。")
    save(OUT / "verification/保存模型与预测复算.json", verification)
    building.replace(TARGET)
    receipt = {"completed_at": datetime.now().astimezone().isoformat(), "archive": str(TARGET),
               "bytes": TARGET.stat().st_size, "sha256": digest(TARGET.read_bytes()), "members": len(members), "indexed_files": len(members)-1,
               "structural_checks": "PASS_CRC_DUPLICATES_INDEX_SIZE_HASH", "fresh_extraction_saved_recomputation": verification,
               "actual_new_conditional_mean_fits": 444, "new_probability_fits": 0, "new_nodes": 0, "new_accounts": 0, "new_downloads": 0,
               "external_review_completed": False, "independent_validation_established": False, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
