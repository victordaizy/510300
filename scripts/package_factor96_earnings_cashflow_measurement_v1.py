"""交付财务测量及原页反证，原样保留历史包，在全新目录重算及复核否定结论。"""
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_repurchase_missing_originals_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
FINAL = ROOT / "deliverables/510300_96候选库_盈利现金质量测量与七项原文反证_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (1986562962, "5596a27ddb6dae42fbb79fc9c05e0051f37401d41343a46e0820cf5586db8226")
VERIFY = "scripts/verify_factor96_earnings_cashflow_measurement_v1.py"
EXPECTED = "PASS_SAVED_FINANCIAL_MEASUREMENT_AND_SOURCE_REJECTION"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stream_identity(stream):
    size, checksum = 0, hashlib.sha256()
    while piece := stream.read(1024 * 1024):
        size += len(piece)
        checksum.update(piece)
    return size, checksum.hexdigest()


def identity(path):
    with path.open("rb") as stream:
        return stream_identity(stream)


def main():
    assert not FINAL.exists() and not (OUT / "delivery_receipt.json").exists()
    result, status = read(OUT / "result.json"), read(OUT / "round_status.json")
    invalidation = read(OUT / "source_invalidation_addendum.json")
    assert result["new_accounts"] == result["new_returns"] == 0
    assert invalidation["current_dataset_admission"] == "BLOCKED_CONFIRMED_SEMANTIC_EXTRACTION_ERRORS"
    assert invalidation["confirmed_wrong_fields"] == 7 and invalidation["affected_daily_joint_member_rows"] == 979
    assert status["cumulative_executed_account_scenarios"] == 584 and not status["goal_achieved"]
    assert read(OUT / "measurement_recomputation_receipt.json")["status"] == "PASS_SAVED_FINANCIAL_QUARTER_TTM_AND_CLOCKS"
    assert read(OUT / "anomaly_evidence/semantic_verification_receipt.json")["confirmed_wrong_fields"] == 7
    previous = read(PREVIOUS / "delivery_receipt.json")
    prior_zip = Path(previous["zip_path"])
    print("核对上轮ZIP原始身份。", flush=True)
    assert (previous["bytes"], previous["sha256"]) == PRIOR_IDENTITY and identity(prior_zip) == PRIOR_IDENTITY
    files = {}
    for path in OUT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not (path.parent == OUT and path.name in ["delivery_receipt.json", "fresh_extraction_verification.log"]):
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ["research/factor96_earnings_cashflow_measurement_v1.py", "research/factor96_earnings_cashflow_anomalies_v1.py",
                 "tests/test_factor96_earnings_cashflow_measurement_v1.py", "scripts/report_factor96_earnings_cashflow_measurement_v1.py",
                 VERIFY, "scripts/package_factor96_earnings_cashflow_measurement_v1.py"]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + prior_zip.name] = prior_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    for base, frozen in [(OUT, "freeze.json"), (OUT / "anomaly_evidence", "review_freeze.json")]:
        for row in read(base / frozen)["files"]:
            path = base / row["path"]
            assert path.relative_to(ROOT).as_posix() in files
            assert identity(path) == (row["bytes"], row["sha256"])
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本轮转入T11财务测量；25个测量前冻结文件、441735条依赖及逐日覆盖完成，但六份同哈希原PDF确认七个语义提取错误。原测量不覆盖，补充来源否定与影响清单；0新账户，目标仍未达成。附件始终是参考材料，非另行授权。\n"
    nav = """# 510300：盈利与现金质量测量及七项原文反证

目标仍未实现，T11未运行。必须先看source_invalidation_addendum.json：本轮按档案计算出的6977个两因子完整事件不能当作有效样本。六份原公告确认七个字段错误，影响17份报告、其中16份机械完整报告，以及979条逐日成员记录。未命中这七个字段不等于正确。

阅读顺序：
1. 02_研究结论.md：结论、七个金额对照、影响下界及停止条件。
2. 本轮目录source_invalidation_addendum.json：来源准入失败与后续修正边界。
3. protocol.json、freeze.json、source_manifest.json：读取本轮财务数值前冻结的计算及来源。
4. inputs/：原始106948条财务事实档案、提取回执、文档队列、公告元数据、历史成员、行业和日历。不是全部原PDF。
5. verified_facts.parquet、member_report_measurements.parquet、formula_dependencies.parquet、daily_member_measurements.parquet、daily_coverage.parquet、missing_field_keys.parquet、yearly_*及result.json：不可覆盖的机械测量与缺口。
6. anomaly_evidence/protocol_addendum.json和request_freeze.json：发现异常后、取得六份原文前的限定计划；该计划不是事前未知异常的预注册。
7. anomaly_evidence/pdf/、receipts/、text/、visual/：六份原文、六次直接HTTP请求、89页全文、12张原页渲染及10页目视记录。六份PDF均与旧档案SHA-256相同。
8. anomaly_evidence/confirmed_field_contradictions.json、affected_*、review_freeze.json：七个字段19个片段定位、36条公式依赖、17份报告、979条逐日观测。
9. prefreeze_test_receipt.json、measurement_recomputation_receipt.json：8项公式/时序测试与保存输入重算；不表示源字段语义正确。
10. program_before/、program_after/：候选进度与权限的前后快照。原库已完成7个固定问题，加一个日更变体共8个问题；累计432正式加152否定实现共584账户情景不变。
11. history/：上轮1986562962字节ZIP原样保存，SHA-256 5596a27ddb6dae42fbb79fc9c05e0051f37401d41343a46e0820cf5586db8226。更早研究与失败沿历史链保留。

上述本轮目录是reports/research/510300_factor96_earnings_cashflow_measurement_v1/。本轮直接用到的来源、代码、结果和六份原文无需展开history/即可读取。原附件HTML、Excel、粘贴文本见sources/，候选注册表见reference_library/。

本轮未读取市场收益、未建立T11账户、未计算修正后绩效。L02采用两个同季历史值，L04采用真实TTM，但少数股东口径、重述和行业时钟限制仍存在；当前名义公告日不能证明当年首次版本。O02与行业标准化后的ETF聚合未计算。

FILE_INDEX.csv覆盖除自身之外所有文件。结构核查与保存测量重算通过，可以同时存在来源语义准入失败。本包外的delivery_receipt.json记录新解压核查及最终ZIP身份。仅本地交付，外部审阅NOT_PERFORMED，当前市场NO_VIEW，无任何订单权限，原PCF/IOPV计划继续暂停。
"""
    exclusions = "范围,原因\nT11账户与收益,原始字段出现已确认的语义提取错误\n修正后策略收益,尚未建立新的来源版本和交易协议\n按极端值自动缩尾或删除,会掩盖来源问题且未被协议授权\n全部11892份原PDF,旧档案不保留全文本轮仅取得六份\nO02及行业归一化后的ETF信号,尚未冻结具体入场映射\n当前聚合财报填缺,可能包含事后修订\n历史首次发布时间证明,本次文件同哈希不构成当年首次抓取\n全档案无其他错误,本轮七个字段为已知影响下界\n外部审阅上传及交易,未执行\n"
    reproduce = "全新解压目录只读复算：\n\npython " + VERIFY + " --root " + OUT.relative_to(ROOT).as_posix() + "\n\n需要pandas、numpy、pyarrow、pypdfium2，版本见REPRODUCTION_ENVIRONMENT.json。命令不联网、不创建账户、不读取市场收益、不修改输入或重随机抽样。它重算保存测量，再从六份PDF重新提取19个证据片段并重建错误传播清单。不要运行prepare-freeze、run、collect或report来覆盖原研究。只需核对原文反证时可追加--source-addendum-only。\n"
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (OUT / "研究结论.md").read_bytes(), "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
        "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"),
        "REPRODUCTION_ENVIRONMENT.json": json.dumps({"python": platform.python_version(), "platform": platform.system(),
            "packages": {p: version(p) for p in ["pandas", "numpy", "pyarrow", "pypdfium2", "requests", "pytest"]}}, ensure_ascii=False, indent=2).encode("utf-8")}
    assert set(files).isdisjoint(extras)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = FINAL.with_suffix(".building.zip")
    print(f"写入{len(files) + len(extras)}个成员，历史ZIP不重压缩。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == ".zip" else zipfile.ZIP_DEFLATED)
        for name, payload in sorted(extras.items()):
            archive.writestr(name, payload)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in rows} == set(archive.namelist()) - {"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_identity(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(FINAL)
    destination = ROOT / "outputs" / ("factor96_earnings_measurement_verify_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(FINAL) as archive:
        assert all((destination / name).resolve().is_relative_to(destination.resolve()) for name in archive.namelist())
        archive.extractall(destination)
    print("CRC、索引及文件身份通过；在全新解压目录重算季度/TTM、时钟和七个原页反证。", flush=True)
    command = [sys.executable, "-X", "utf8", str(destination / VERIFY), "--root", str(destination / OUT.relative_to(ROOT))]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT / "fresh_extraction_verification.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    assert process.returncode == 0, process.stdout + process.stderr
    verified = json.loads(process.stdout.strip().splitlines()[-1])
    assert verified["status"] == EXPECTED and verified["new_accounts"] == verified["network_requests"] == 0
    size, checksum = identity(FINAL)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(FINAL), "bytes": size, "sha256": checksum,
        "members": len(files) + len(extras), "indexed_members": len(rows), "crc_duplicate_index_size_sha256": "PASS",
        "frozen_input_coverage": "PASS", "fresh_extraction": str(destination), "saved_output_recomputation": verified,
        "source_semantic_admission": "BLOCKED_CONFIRMED_SEMANTIC_EXTRACTION_ERRORS",
        "prior_zip_bytes": PRIOR_IDENTITY[0], "prior_zip_sha256": PRIOR_IDENTITY[1],
        "new_accounts_this_round": 0, "formal_accounts": 432, "invalid_implementation_accounts": 152, "executed_accounts": 584,
        "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED", "orders_authorized": False}
    with (OUT / "delivery_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
