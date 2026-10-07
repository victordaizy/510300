"""生成本轮有限资料审阅包，并执行独立解压复算。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/510300_lpr_expectation_identification_v1"
TARGET = ROOT / "deliverables/510300_LPR事前预期识别_V1_GPT审阅_20260922.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    building = TARGET.with_suffix(".building.zip")
    extract = REPORT / "verification/fresh_extract"
    require(not TARGET.exists() and not building.exists() and not extract.exists(), "已存在交付或解压目录，请另立版本。")
    members = {p.relative_to(REPORT).as_posix(): p.read_bytes() for p in REPORT.rglob("*") if p.is_file() and "verification" not in p.relative_to(REPORT).parts and p.name != "delivery_receipt.json"}
    members["code/lpr_expectation_identification_v1.py"] = (ROOT / "research/lpr_expectation_identification_v1.py").read_bytes()
    members["code/package_lpr_expectation_identification_v1.py"] = Path(__file__).read_bytes()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, data in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(data), "sha256": digest(data)})
    members["FILE_INDEX.csv"] = stream.getvalue().encode("utf-8-sig")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    extract.mkdir(parents=True)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "ZIP成员或CRC有误。")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) == {r["path"] for r in index} | {"FILE_INDEX.csv"}, "索引成员集合不符。")
        for row in index:
            data = archive.read(row["path"])
            require(len(data) == int(row["bytes"]) and digest(data) == row["sha256"], "索引大小或哈希不符。")
        for name in names:
            destination = (extract / name).resolve()
            require(destination.is_relative_to(extract.resolve()), "ZIP路径超出解压目录。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    completed = subprocess.run([sys.executable, "-X", "utf8", str(extract / "code/lpr_expectation_identification_v1.py"), "verify", "--root", str(extract)], check=True, capture_output=True, text=True, encoding="utf-8")
    verification = json.loads(completed.stdout)
    (REPORT / "verification/recomputation.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    building.replace(TARGET)
    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(TARGET), "bytes": TARGET.stat().st_size,
        "sha256": digest(TARGET.read_bytes()), "members": len(members), "indexed_files": len(members) - 1,
        "zip_crc_duplicates_index_size_hash": "PASS", "fresh_extraction_recomputation": verification,
        "verification_scope": "保存资料身份、事前时钟比较、数值区间、全部84月及汇总；没有认证网络首版或证券预测结果",
        "new_equity_labels": 0, "new_models": 0, "new_accounts": 0, "external_review_completed": False, "whole_goal_achieved": False,
    }
    (REPORT / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
