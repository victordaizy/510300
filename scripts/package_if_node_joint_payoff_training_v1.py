"""打包联合训练结果，在新解压目录只复算保存参数和账簿。"""
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml
import matplotlib


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_if_node_joint_payoff_training_v1"
TARGET = ROOT / "deliverables/510300_节点胜率与条件盈亏实际训练_V1_GPT审阅_20260924.zip"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    building, extracted = TARGET.with_suffix(".building.zip"), OUT / "verification/fresh_extract_verified"
    require(not TARGET.exists() and not building.exists() and not extracted.exists(), "已存在交付，不覆盖。")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    require(summary["logistic_fits_including_final"] == 222 and summary["conditional_magnitude_fits_including_final"] == 444, "实际拟合数量不同。")
    for name in ["00_README_FIRST.md", "训练结果.md", "用户要求.md", "GPT审阅提示词.md", "EXCLUSIONS.md", "branch_decision.json", "最新频率要求与逐年评价.md", "annual_frequency_requirement.json", "annual_frequency_summary.json"]:
        require((OUT / name).is_file(), f"缺少交付文件：{name}")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": scipy.__version__, "pyarrow": pyarrow.__version__, "PyYAML": yaml.__version__, "matplotlib": matplotlib.__version__}
    save(OUT / "runtime.json", runtime)
    (OUT / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in runtime.items() if k != "python") + "\ntzdata\n", encoding="utf-8")
    members = {p.relative_to(OUT).as_posix(): p.read_bytes() for p in OUT.rglob("*")
               if p.is_file() and "verification" not in p.relative_to(OUT).parts
               and "__pycache__" not in p.relative_to(OUT).parts and p.name != "delivery_receipt.json"}
    for name in ["finish_if_node_joint_payoff_training_v1.py", "package_if_node_joint_payoff_training_v1.py"]:
        members["code/" + name] = (ROOT / "scripts" / name).read_bytes()
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, data in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(data), "sha256": digest(data)})
    members["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "重复成员或CRC错误。")
        indexed = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) == {r["path"] for r in indexed} | {"FILE_INDEX.csv"}, "索引集合错误。")
        for r in indexed:
            value = archive.read(r["path"])
            require(len(value) == int(r["bytes"]) and digest(value) == r["sha256"], "成员大小或哈希错误。")
        for name in names:
            target = (extracted / name).resolve()
            require(target.is_relative_to(extracted.resolve()), "成员超出解压目录。")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    result = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "code/if_node_joint_payoff_training_v1.py"),
                             "verify", "--root", str(extracted)], check=True, capture_output=True, text=True, encoding="utf-8")
    verification = json.loads(result.stdout)
    require(verification["status"] == "PASS_SAVED_JOINT_MODELS_PREDICTIONS_AND_ACCOUNTS", "保存结果复算失败。")
    frequency_result = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "code/annual_node_frequency_requirement_v1.py"),
                                       "verify", "--root", str(extracted)], check=True, capture_output=True, text=True, encoding="utf-8")
    frequency_verification = json.loads(frequency_result.stdout)
    require(frequency_verification["status"] == "PASS_SAVED_ANNUAL_FREQUENCY_RECOMPUTATION", "新增频率要求复算失败。")
    save(OUT / "verification/保存模型与账户复算.json", verification)
    save(OUT / "verification/逐年频率复算.json", frequency_verification)
    building.replace(TARGET)
    receipt = {"completed_at": datetime.now().astimezone().isoformat(), "archive": str(TARGET),
               "bytes": TARGET.stat().st_size, "sha256": digest(TARGET.read_bytes()),
               "members": len(members), "indexed_files": len(members)-1,
               "structural_checks": "PASS_CRC_DUPLICATES_INDEX_SIZE_HASH", "fresh_extraction_saved_recomputation": verification,
               "annual_frequency_verification": frequency_verification,
               "research_logistic_fits": 222, "research_conditional_magnitude_fits": 444, "research_empirical_updates": 111,
               "research_accounts": summary["account_scenarios"], "new_downloads": 0,
               "external_review_completed": False, "independent_validation_established": False,
               "goal_achieved": False, "orders_authorized": False}
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
