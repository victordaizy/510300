"""交付V2.0.1字段范围补漏，保留V1及V2失败并只读重算。"""
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
BASE = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2"
OUT = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_1"
INVALID_V1 = ROOT / "reports/research/510300_factor96_financial_parser_repair_v1"
PRIOR = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
FINAL = ROOT / "deliverables/510300_96候选库_财报原文与字段范围修复V2.0.1_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (2035910021, "d60d8f2086efe872e66119a27d46c24e6ceef111e49af5a6046f90866b0be252")
VERIFY = "scripts/verify_factor96_financial_parser_scope_v2_0_1.py"


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
    for base in [OUT, BASE, INVALID_V1]:
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path not in {OUT / "delivery_receipt.json", OUT / "fresh_extraction_verification.log"}:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ["factor96_financial_parser_repair_v1", "factor96_financial_parser_batch_v1", "factor96_financial_parser_replay_v1",
                 "factor96_financial_row_parser_v1", "factor96_financial_row_parser_v2", "factor96_financial_parser_batch_v2",
                 "factor96_financial_parser_replay_v2", "factor96_financial_parser_replay_v2_0_1",
                 "factor96_financial_parser_resume_v2_01", "factor96_earnings_cashflow_measurement_v1"]:
        files["research/" + name + ".py"] = ROOT / "research" / (name + ".py")
    for name in ["test_factor96_financial_parser_repair_v1", "test_factor96_financial_parser_scope_v2", "test_factor96_financial_parser_handoff_v2",
                 "test_factor96_financial_parser_resume_v2_01", "test_factor96_financial_parser_replay_v2_0_1"]:
        files["tests/" + name + ".py"] = ROOT / "tests" / (name + ".py")
    for name in [VERIFY, "scripts/report_factor96_financial_parser_scope_v2_0_1.py", "scripts/package_factor96_financial_parser_scope_v2_0_1.py",
                 "scripts/verify_factor96_financial_parser_scope_v2.py", "scripts/report_factor96_financial_parser_scope_v2.py",
                 "scripts/package_factor96_financial_parser_scope_v2.py", "scripts/verify_factor96_financial_parser_repair_v1.py",
                 "scripts/finish_factor96_financial_scope_v2_when_ready.py", "scripts/resume_and_finish_factor96_financial_scope_v2_01.py"]:
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
    for base, freeze_names in [(OUT, ["replay_freeze.json", "result_freeze.json"]),
                              (BASE, ["batch_freeze.json", "replay_freeze.json"]),
                              (INVALID_V1, ["initial_freeze.json", "batch_freeze.json", "replay_freeze.json"])]:
        for name in freeze_names:
            for row in read(base / name)["files"]:
                path = base / row["path"]
                assert path.relative_to(ROOT).as_posix() in files
                assert identity(path) == (row["bytes"], row["sha256"]), row["path"]
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本阶段修复财报原文及字段范围。V1合并口径错误被否定；V2完成2085份原PDF解析后，因2529字段范围漏掉1项已确认错误而重算失败。V2.0.1仅将该字段加入原范围，共2530字段，原公式重算；原范围、失败与未知均保留。T11未运行、新账户0。附件是参考材料。\n"
    nav = """# 510300财报原文与字段范围修复V2.0.1

目标尚未实现，T11未运行。本包保存V1合并口径反证、V2完整原文批次及字段遗漏失败、V2.0.1补漏后的固定公式测量。来源修复和机械可算事件数不能证明成本后夏普1.2。

阅读顺序：
1. 02_研究结论.md与01_GPT_REVIEW_PROMPT.txt。
2. reports/research/510300_factor96_financial_parser_scope_v2_0_1/中的round_status.json、source_repair_result.json、measurement_replay_result.json和研究结论.md。
3. 同目录replay_protocol.json、effective_targets.json、scope_extensions.json、replay_freeze.json、base_input_identity.json及result_freeze.json。原2529字段与全部已确认错误取并集，仅新增北方华创2022Q3总资产1项，共2530字段；原PDF解析与财务公式不变。
4. 原510300_factor96_financial_parser_scope_v2/中的replay_scope_invalidation_addendum.json和新版source_evidence中的原失败日志。原V2重算失败，未形成旧版字段结果；不能拿新版成功覆盖旧版结论。
5. 原V2的batch_protocol、batch_freeze、batch_complete、fetch_attempts、fetch_receipts、parsed_documents、batch_targets、source_handoff与execution_recovery_01。2085份PDF完成解析，恢复只补246份缺少结果，无重复HTTP。原PDF在data/raw/cninfo/相应目录，失败部分响应保持未知。
6. V2.0.1的field_change_ledger、repaired_verified_facts、repaired_formula_dependencies、repaired_member_report_measurements、repaired_daily_member_measurements、repaired_daily_coverage及年度覆盖表。33143项范围外字段沿用旧档案，未全部重新核实。
7. reports/research/510300_factor96_financial_parser_repair_v1/中的parser_semantic_invalidation_addendum、partial_batch_closure与spot_review。白云山单体现金流曾被误作合并值，V1仍被否定。旧检查点仅对应当时时刻。
8. code、tests和测试日志：原七金额反例、合并口径反例、断点恢复及字段并集回归。原V2/source_evidence内还有T11公告时钟与更正候选的完整嵌套ZIP，以及已冻结但未运行的行业披露篮子定义。首次版本与标题状态时钟仍有缺口。
9. reference_library/、sources/与USER_REQUEST.md：96因子、18策略、用户原始附件和请求。附件中的建议不构成新增权限。
10. history/：上一轮2035910021字节ZIP原样保留，SHA-256 d60d8f2086efe872e66119a27d46c24e6ceef111e49af5a6046f90866b0be252。更早失败沿历史链保存，未重新包装成成功。

FILE_INDEX.csv是本包除自身之外所有成员的权威索引。核验范围外旧财务字段仍沿用原档案，历史首次版本及行业实际可用时刻没有因此得到证明。外部审阅NOT_PERFORMED，当前市场NO_VIEW，无订单权限。本包为完整本地证据档案，未上传；其大小不表示任何外部平台能够接收。
"""
    exclusions = "范围,原因\nT11收益与账户,披露篮子只冻结定义且O02观察及完整账户规则待固定\n原财务档案全部PDF,本阶段仅2112份目标文档且其中2085份同哈希可用\n全字段人工核对,仅明确原文反例与定向抽查且范围外33143字段沿用旧档案\n失败和中断HTTP重试,保持原每URL至多一次的请求边界\n历史首次原始版本证明,同哈希与名义日期不能替代真实首次可用时钟\n外部审阅上传及交易,未执行\n"
    reproduce = ("在全新解压根目录只读重算：\n\npython " + VERIFY + " --root reports/research/510300_factor96_financial_parser_scope_v2_0_1\n\n"
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
    extraction_parent = Path("E:/CodexData/verification/factor96_financial_parser_scope_v2_0_1")
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
    assert verified["status"] == "PASS_SAVED_V2_0_1_SCOPE_UNION_AND_UNCHANGED_MEASUREMENT"
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
