"""完整交付来源修复、V1否定和V2重算，在全新解压目录只读验证。"""
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2"
INVALID_V1 = ROOT / "reports/research/510300_factor96_financial_parser_repair_v1"
PRIOR = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
FINAL = ROOT / "deliverables/510300_96候选库_财报原文与合并口径修复V2_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (2035910021, "d60d8f2086efe872e66119a27d46c24e6ceef111e49af5a6046f90866b0be252")
VERIFY = "scripts/verify_factor96_financial_parser_scope_v2.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stream_identity(stream):
    checksum, size = hashlib.sha256(), 0
    while part := stream.read(1024 * 1024):
        checksum.update(part)
        size += len(part)
    return size, checksum.hexdigest()


def identity(path):
    with path.open("rb") as stream:
        return stream_identity(stream)


def main():
    assert not FINAL.exists() and not (OUT / "delivery_receipt.json").exists()
    assert read(OUT / "round_status.json")["cumulative_executed_account_scenarios"] == 584
    assert read(OUT / "source_repair_result.json")["all_confirmed_regressions_passed"]
    assert read(INVALID_V1 / "partial_batch_closure.json")["v1_replay"] == "NOT_RUN"
    prior_receipt = read(PRIOR / "delivery_receipt.json")
    prior_zip = Path(prior_receipt["zip_path"])
    assert (prior_receipt["bytes"], prior_receipt["sha256"]) == PRIOR_IDENTITY
    assert identity(prior_zip) == PRIOR_IDENTITY
    files = {}
    for base in [OUT, INVALID_V1]:
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.name not in {"delivery_receipt.json", "fresh_extraction_verification.log"}:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ["factor96_financial_parser_repair_v1", "factor96_financial_parser_batch_v1", "factor96_financial_parser_replay_v1",
                 "factor96_financial_row_parser_v1", "factor96_financial_row_parser_v2", "factor96_financial_parser_batch_v2",
                 "factor96_financial_parser_replay_v2", "factor96_earnings_cashflow_measurement_v1"]:
        files["research/" + name + ".py"] = ROOT / "research" / (name + ".py")
    for name in ["test_factor96_financial_parser_repair_v1", "test_factor96_financial_parser_scope_v2", "test_factor96_financial_parser_handoff_v2"]:
        files["tests/" + name + ".py"] = ROOT / "tests" / (name + ".py")
    for name in [VERIFY, "scripts/report_factor96_financial_parser_scope_v2.py", "scripts/package_factor96_financial_parser_scope_v2.py",
                 "scripts/verify_factor96_financial_parser_repair_v1.py"]:
        files[name] = ROOT / name
    for base in [ROOT / "data/raw/cninfo/factor96_financial_parser_repair_v1", ROOT / "data/raw/cninfo/factor96_financial_parser_scope_v2"]:
        for path in base.rglob("*"):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + prior_zip.name] = prior_zip
    files["history/prior_delivery_receipt.json"] = PRIOR / "delivery_receipt.json"
    for base, freeze_names in [(OUT, ["batch_freeze.json", "replay_freeze.json", "result_freeze.json"]),
                              (INVALID_V1, ["initial_freeze.json", "batch_freeze.json", "replay_freeze.json"])]:
        for name in freeze_names:
            for row in read(base / name)["files"]:
                path = base / row["path"]
                assert path.relative_to(ROOT).as_posix() in files
                assert identity(path) == (row["bytes"], row["sha256"]), row["path"]
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本阶段修复财报原文提取。V1在批量复核中确认新合并口径错误并终止；V2沿原2112份/2529字段范围复核并按原公式重算。所有失败、未知和旧包身份保留。T11未运行、新账户0。附件是参考材料。\n"
    nav = """# 510300财报原文与合并口径修复V2

目标尚未实现。T11未运行。本包保存V1的新来源反证、批次终止、V2接管和同公式测量，不能把源修复、测试通过或机械完整事件数当作成本后夏普1.2的证据。

阅读顺序：
1. 02_研究结论.md与01_GPT_REVIEW_PROMPT.txt。
2. reports/research/510300_factor96_financial_parser_scope_v2/中的round_status.json、source_repair_result.json和measurement_replay_result.json。
3. 同目录batch_protocol.json、batch_freeze.json、replay_protocol.json、replay_freeze.json及result_freeze.json。V2是在V1真实原文反例出现后制定，不能声称事前未知。
4. V2的source_handoff/：原回执、终止原因及接管边界。旧失败或中断没有重复请求；V2只继续原计划未尝试URL。
5. V2的fetch_attempts/、fetch_receipts/、parsed_documents/与batch_targets.json；原PDF位于data/raw/cninfo/下对应V1或V2目录，引用路径保持。失败的部分响应也保留，不能作为已准入PDF。
6. V2的field_change_ledger.parquet、repaired_verified_facts.parquet、repaired_formula_dependencies.parquet、repaired_member_report_measurements.parquet、repaired_daily_member_measurements.parquet、repaired_daily_coverage.parquet及年度覆盖表。
7. reports/research/510300_factor96_financial_parser_repair_v1/中的parser_semantic_invalidation_addendum.json、partial_batch_closure.json及spot_review/：白云山公司现金流778464955.25元不能替代合并585185023.09元，V1未执行原定覆盖重算。该目录的早先运行中检查点仅表示当时时刻。
8. 测试日志和code/：原七反例、V2合并/独立报表回归与请求接管测试。V1旧结果保留，不作有效字段来源。
9. reference_library/、sources/与USER_REQUEST.md：96因子、18策略、用户原始附件和请求。附件中的建议不构成新增权限。
10. history/：上一轮2035910021字节ZIP原样保留，SHA-256 d60d8f2086efe872e66119a27d46c24e6ceef111e49af5a6046f90866b0be252。更早失败沿历史链保存，未重新包装成成功。

FILE_INDEX.csv是本包除自身之外所有成员的权威索引。核验范围外旧财务字段仍沿用原档案，历史首次版本及行业实际可用时刻没有因此得到证明。外部审阅NOT_PERFORMED，当前市场NO_VIEW，无订单权限。本包为完整本地证据档案，未上传；其大小不表示任何外部平台能够接收。
"""
    exclusions = "范围,原因\nT11收益与账户,尚未固定行业ETF聚合和首日反应规则\n原财务档案全部PDF,本阶段只处理冻结2112份来源路径\n全字段人工核对,只保存明确原文反例与定向抽查不能宣称全样本人工验证\n未完成失败中断的请求重试,沿原冻结每URL至多一次原则保持未知\n历史首次原始版本证明,本次同哈希不能替代历史首次抓取\n外部审阅上传及交易,未执行\n"
    reproduce = ("在全新解压根目录只读重算：\n\npython " + VERIFY + " --root reports/research/510300_factor96_financial_parser_scope_v2\n\n"
        "所需包见REPRODUCTION_ENVIRONMENT.json。该命令核对全部原PDF哈希及保存候选，重解析七份真实回归PDF，重建来源字段和固定公式覆盖，不联网、不训练、不创建账户。它不重新解析2112份所有表格，也不声称人工核验全部源字段。不要运行研究驱动的freeze、run、prepare、replay或report命令覆盖已保存快照。\n")
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (OUT / "研究结论.md").read_bytes(), "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
        "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"),
        "REPRODUCTION_ENVIRONMENT.json": json.dumps({"python": platform.python_version(), "platform": platform.system(),
            "packages": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "pdfplumber", "pypdfium2", "requests", "pytest"]}}, ensure_ascii=False, indent=2).encode("utf-8")}
    assert set(files).isdisjoint(extras)
    total_bytes = sum(path.stat().st_size for path in files.values()) + sum(len(v) for v in extras.values())
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    extraction_parent = ROOT / "data/audit/factor96_financial_parser_scope_v2"
    extraction_parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(FINAL.parent).free > total_bytes + 1024 ** 3
    assert shutil.disk_usage(extraction_parent).free > total_bytes + 1024 ** 3
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = FINAL.with_suffix(".building.zip")
    print(f"写入{len(files) + len(extras)}个成员，保留全部原文与V1失败，已有压缩二进制不重复压缩。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            method = zipfile.ZIP_STORED if path.suffix.lower() in {".pdf", ".zip", ".parquet", ".png", ".xlsx"} else zipfile.ZIP_DEFLATED
            archive.write(path, name, compress_type=method)
        for name, payload in sorted(extras.items()):
            archive.writestr(name, payload)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names)) == len(files) + len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {row["path"] for row in rows} == set(names) - {"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_identity(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(FINAL)
    destination = extraction_parent / ("fresh_verification_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(FINAL) as archive:
        assert all((destination / name).resolve().is_relative_to(destination.resolve()) for name in archive.namelist())
        archive.extractall(destination)
    print("完整索引、CRC和原文身份核对通过，开始全新解压目录的只读重算。", flush=True)
    command = [sys.executable, "-X", "utf8", str(destination / VERIFY), "--root", str(destination / OUT.relative_to(ROOT))]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT / "fresh_extraction_verification.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    assert process.returncode == 0, process.stdout + process.stderr
    verified = json.loads(process.stdout.strip().splitlines()[-1])
    assert verified["status"] == "PASS_SAVED_V2_SCOPE_REPAIR_AND_UNCHANGED_MEASUREMENT"
    assert verified["new_accounts"] == verified["network_requests"] == 0
    size, checksum = identity(FINAL)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(FINAL), "bytes": size, "sha256": checksum,
        "members": len(files) + len(extras), "indexed_members": len(rows), "crc_duplicate_index_size_sha256": "PASS",
        "frozen_input_coverage": "PASS", "fresh_extraction": str(destination), "saved_output_recomputation": verified,
        "prior_zip_bytes": PRIOR_IDENTITY[0], "prior_zip_sha256": PRIOR_IDENTITY[1], "new_accounts_this_round": 0,
        "formal_accounts": 432, "invalid_implementation_accounts": 152, "executed_accounts": 584,
        "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED", "orders_authorized": False}
    with (OUT / "delivery_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
