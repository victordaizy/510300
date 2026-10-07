"""为第二轮已完成研究打包；只读保存输出，不运行策略或统计抽样。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import zipfile


PROJECT = Path(__file__).absolute().parents[3]
STUDY = Path(__file__).resolve().parent
DELIVERABLES = PROJECT / "deliverables"
STEM = "510300_第二轮两项结构比较_results_run2_20261002"
SOURCE_NAME = "510300_纯日线低频政策_results_run1_20261002.zip"
SOURCE_HASH = "07056707ea79b6e70d677ba85af7dfa3c7a73ee7957164fc44db14d2ae97b8e1"
EXPECTED_PROGRAM = {
    "reference_engine.py": "b8a6b9b85ef9aa7735b0ae7e192469338e8dbc65ef89d39d3bd3293be4b5f421",
    "reference_rules.json": "5320e8e3ffef98b6b397b2df681f07e00b4351130aaf1fbdc0ae9e627fbd5a96",
    "run_next.py": "c6604d8c240b24d10d9a89ce99154d42636ac6c9884a188583b59879a09d33c4",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def main() -> None:
    DELIVERABLES.mkdir(exist_ok=True)
    final = DELIVERABLES / f"{STEM}.zip"
    building = DELIVERABLES / f"{STEM}.building.zip"
    receipt_path = DELIVERABLES / f"{STEM}_交付校验.json"
    hash_path = DELIVERABLES / f"{STEM}.sha256"
    prompt_path = DELIVERABLES / f"{STEM}_给Pro的审阅提示词.md"
    if any(path.exists() for path in (final, building, receipt_path, hash_path, prompt_path)):
        raise FileExistsError("交付目标已存在，不覆盖已有包或记录")

    source = DELIVERABLES / SOURCE_NAME
    if digest(source) != SOURCE_HASH:
        raise ValueError("首轮直接输入ZIP身份变化")
    for name, expected in EXPECTED_PROGRAM.items():
        if digest(STUDY / "launch" / name) != expected:
            raise ValueError(f"收到程序或配置发生变化：{name}")

    raw_index = json.loads((STUDY / "analysis/original_results_file_index.json").read_text(encoding="utf-8-sig"))
    raw_files = sorted(path for path in (STUDY / "results_run2").rglob("*") if path.is_file())
    if len(raw_files) != 137 or len(raw_index) != 137:
        raise ValueError("原始结果文件总数应为137")
    expected_raw = {entry["path"]: entry for entry in raw_index}
    for path in raw_files:
        relative = path.relative_to(STUDY / "results_run2").as_posix()
        entry = expected_raw[relative]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"原始结果发生变化：{relative}")

    with (STUDY / "results_run2/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    roles = dict(Counter(row["study_role"] for row in summary))
    if roles != {"ORIGINAL_CONTROL_REPLAY": 24, "NEW_FIXED_POLICY": 8}:
        raise ValueError("32账户角色矩阵不一致")

    members: dict[str, Path | bytes] = {}

    def add(name: str, value: Path | bytes) -> None:
        if name in members or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"成员路径非法或重复：{name}")
        members[name] = value

    for directory in ("launch", "received", "results_run2", "analysis"):
        for path in sorted((STUDY / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path.relative_to(STUDY).as_posix(), path)
    root_names = [
        "authority_and_input_freeze.json", "local_execution_receipt.json", "real_run_claim.json",
        "self_test_local_receipt.json", "self_test_stdout.txt", "self_test_stderr.txt",
        "run_stdout.txt", "run_stderr.txt", "analyze_saved_results.py", "plot_saved_results.py",
        "build_review_package.py",
    ]
    for name in root_names:
        add(name, STUDY / name)
    add("README.md", STUDY / "review_readme.md")
    add("PRO_REVIEW_PROMPT.md", STUDY / "pro_review_prompt.md")
    for name in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md"):
        add(f"docs_snapshot/{name}", PROJECT / "docs" / name)
    add(f"source_archives/{SOURCE_NAME}", source)
    with zipfile.ZipFile(source) as source_zip:
        for name in ("readable/normalized_prices.csv", "readable/normalized_dividends.csv"):
            add(name, source_zip.read(name))

    created = timestamp()
    scope = {
        "created_at": created,
        "scope": "SECOND_FIXED_STRUCTURAL_COMPARISON_COMPLETE_SAVED_RESULTS",
        "source_zip_sha256": SOURCE_HASH,
        "original_replays": 24,
        "new_accounts": 8,
        "role_counts": roles,
        "raw_results_files": 137,
        "raw_results_original_hashes_retained": True,
        "launch_original_hashes_retained": EXPECTED_PROGRAM,
        "joint_target_passes": 0,
        "new_market_accounts_during_packaging": 0,
        "new_models_during_packaging": 0,
        "new_random_draws_during_packaging": 0,
        "network_requests": 0,
        "uploaded": False,
        "external_review_completed": False,
        "independent_validation": False,
        "actual_fill_verified": False,
        "promoted": [],
        "docs_snapshot": "两份共享文档打包前快照，不含本ZIP完成后追加的交付记录；其他研究线链接未递归包含。",
        "source_archive_reason": "收到程序固定核对首轮ZIP的原始SHA并读取其24对照及两CSV；属于直接输入依赖。",
        "excluded": ["更早47.17MB路线图审阅ZIP", "项目运行环境和缓存", "其他分支的递归数据和代码", "认证信息"],
        "integrity_scope": "ZIP成员、CRC、索引、大小和哈希；不是安全/隐私审计、外部审阅或金融有效性证明。",
    }
    add("PACKAGE_SCOPE.json", (json.dumps(scope, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    indexed = []
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, source_value in sorted(members.items()):
            payload = source_value.read_bytes() if isinstance(source_value, Path) else source_value
            indexed.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            method = zipfile.ZIP_STORED if name.startswith("source_archives/") else zipfile.ZIP_DEFLATED
            archive.writestr(name, payload, compress_type=method)
        index_stream = io.StringIO(newline="")
        writer = csv.DictWriter(index_stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(indexed)
        archive.writestr("FILE_INDEX.csv", index_stream.getvalue().encode("utf-8-sig"))

    with zipfile.ZipFile(building) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC失败：{bad}")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(members) | {"FILE_INDEX.csv"}:
            raise ValueError("ZIP成员集合不一致")
        saved_index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if len(saved_index) != len(indexed):
            raise ValueError("索引数量不一致")
        for row in saved_index:
            payload = archive.read(row["path"])
            if len(payload) != int(row["bytes"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError(f"成员大小或哈希失败：{row['path']}")

    # 再核对直接输入和原始结果；打包只复制，不改研究输出。
    if digest(source) != SOURCE_HASH:
        raise ValueError("打包期间直接输入变化")
    for name, expected in EXPECTED_PROGRAM.items():
        if digest(STUDY / "launch" / name) != expected:
            raise ValueError("打包期间收到程序发生变化")
    for entry in raw_index:
        if digest(STUDY / "results_run2" / entry["path"]) != entry["sha256"]:
            raise ValueError("打包期间原始结果变化")

    building.replace(final)
    final_hash = digest(final)
    receipt = {
        "status": "PASS_SAVED_DELIVERY_INTEGRITY",
        "created_at": created,
        "verified_at": timestamp(),
        "zip": str(final),
        "bytes": final.stat().st_size,
        "sha256": final_hash,
        "members": len(names),
        "indexed_members": len(saved_index),
        "zip_crc": "PASS",
        "member_unique_and_complete": "PASS",
        "indexed_member_bytes_sha256": "PASS",
        "raw_results_files_unchanged": 137,
        "original_source_zip_unchanged": True,
        "received_program_rules_unchanged": True,
        "accounts": 32,
        "roles": roles,
        "source_zip_sha256": SOURCE_HASH,
        "new_market_accounts_during_packaging": 0,
        "new_models_during_packaging": 0,
        "new_random_draws_during_packaging": 0,
        "network_requests": 0,
        "uploaded": False,
        "external_review_completed": False,
        "independent_validation": False,
        "actual_fill_verified": False,
        "candidate_promoted": False,
        "scope": scope["integrity_scope"],
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hash_path.write_text(f"{final_hash}  {final.name}\n", encoding="utf-8")
    prompt_path.write_bytes((STUDY / "pro_review_prompt.md").read_bytes())
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
