"""交付七个回购原方案来源包，保留旧包身份并在全新目录只读核查。"""
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
OUT = ROOT / "reports/research/510300_factor96_repurchase_missing_originals_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_repurchase_change_chain_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
FINAL = ROOT / "deliverables/510300_96候选库_七个回购原方案补证_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (1977876526, "0d95e5a3d2d1deb3ee3ba5ebca95f012e017ed4c47d01bb53470ea69efee7f69")
VERIFY = "scripts/verify_factor96_repurchase_missing_originals_v1.py"
EXPECTED = "PASS_SAVED_ORIGINAL_ROOTS_AND_LIFECYCLE_CLOCKS"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stream_identity(stream):
    size, checksum = 0, hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        checksum.update(chunk)
    return size, checksum.hexdigest()


def identity(path):
    with path.open("rb") as stream:
        return stream_identity(stream)


def main():
    assert not FINAL.exists() and not (OUT / "delivery_receipt.json").exists()
    result, status = read(OUT / "result.json"), read(OUT / "round_status.json")
    assert result["new_accounts"] == result["new_returns"] == 0 and result["collector_direct_http_requests"] == 20
    assert status["cumulative_executed_account_scenarios"] == 584 and not status["goal_achieved"]
    assert read(OUT / "saved_verification_receipt.json")["status"] == EXPECTED
    previous = read(PREVIOUS / "delivery_receipt.json")
    prior_zip = Path(previous["zip_path"])
    print("核对上轮ZIP原始身份。", flush=True)
    assert (previous["bytes"], previous["sha256"]) == PRIOR_IDENTITY and identity(prior_zip) == PRIOR_IDENTITY
    files = {}
    for path in OUT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not (path.parent == OUT and path.name in ["delivery_receipt.json", "fresh_extraction_verification.log"]):
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ["research/factor96_repurchase_missing_originals_v1.py", "research/factor96_repurchase_missing_originals_analysis_v1.py", "tests/test_factor96_repurchase_missing_originals_v1.py",
                 "scripts/report_factor96_repurchase_missing_originals_v1.py", VERIFY,
                 "scripts/package_factor96_repurchase_missing_originals_v1.py"]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + prior_zip.name] = prior_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    for item in read(OUT / "analysis_freeze.json")["files"]:
        path = OUT / item["path"]
        assert path.relative_to(ROOT).as_posix() in files
        assert identity(path) == (item["bytes"], item["sha256"])
    for item in read(OUT / "prior_account_evidence/manifest.json")["files"]:
        assert identity(OUT / item["path"]) == (item["bytes"], item["sha256"])
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本轮按七个已明确原方案批准日完成7个窗口查询，采集程序20次直接HTTP请求取得13份PDF；七个既已定位缺口已补证，仍无交易因子准入或新账户，T12仍NOT_RUN。用户目标与账户合同不变，附件是参考材料。\n"
    nav = """# 510300：七个回购原方案补证

目标仍未实现。七个既已识别目标的原方案已对应；31份变更候选仍未完整复核。7个新日期窗口、22条目录、13份新PDF、7个确认原方案、5份后续事项和1个背景引用排除。0新账户、0收益计算、0交易因子准入，T12仍NOT_RUN。

阅读顺序：
1. 02_研究结论.md：七个方案、不同后续状态、剩余缺口及停止条件。
2. reports/research/510300_factor96_repurchase_missing_originals_v1/protocol.json、source_request_freeze.json：查询前固定的7个日期窗口；local_search_evidence.json记录本地缺口及此前未查询证据。
3. 同目录catalogues.json、selected_pdf_targets.json、raw/、receipts/、text/、documents.json：20个采集HTTP原响应和13份新PDF。13份PDF不会全计作独立原方案。
4. root_review_cards.json、original_root_ledger.json、七个原方案与变更关联.csv：七个方案身份与预算、用途，大全2023-0XX编号保留未知。
5. local_sources/、followup_review_cards.json、followup_ledger.json：13份复用原文；天山已注销、闻泰已用于转股，与另外3份注销安排分开；一个2026新方案的背景引用排除。
6. existing_original_sources/：多方案变更另涉及的4个早先已知方案原件；不计作本轮新采集。
7. updated_change_ledger.json、asof_event_queries.json：54份台账的新补充快照及68个查询；旧冻结台账未改。20份公告对应22个方案、24条关联，31份未复核。
8. analysis_protocol.json、analysis_freeze.json、prefreeze_test_receipt.json、saved_verification_receipt.json：142个冻结文件，7项边界测试，72条短语、114处页内定位和只读核查。
9. visual/与visual_review_receipt.json：三页原图及核对结论。
10. prior_account_evidence/：最近日更变体失败的保存结果；本轮未重跑。
11. history/：上轮1,977,876,526字节ZIP原样保存，SHA-256 0d95e5a3d2d1deb3ee3ba5ebca95f012e017ed4c47d01bb53470ea69efee7f69。更早账本和失败沿旧快照链保留。

除根目录文件和history/外，上述本轮文件都位于reports/research/510300_factor96_repurchase_missing_originals_v1/。本轮新增语义关联的原件直接可读，无需解开历史ZIP；54份旧候选的完整原件沿history/保存。附件HTML、Excel和粘贴文本在sources/，96/18注册表在reference_library/。

20次只指采集程序直接HTTP请求；额外网页入口浏览不混入此计数。自由流通、完整执行与用途链、首次历史发布版本尚未完成。累计432正式+152否定实现=584账户情景，不是独立策略数。执行用途569已知、379未知计数不变。

FILE_INDEX.csv覆盖除自身外全部文件，最终ZIP身份在包外交付回执。结构检查、保存输出重算与时钟检查不证明外部审阅或策略有效性。仅510300.SH/CASH_CNY研究，PCF/IOPV计划继续暂停，当前市场NO_VIEW，独立前向0，外部审阅NOT_PERFORMED，无订单权限。本包仅本地交付。
"""
    exclusions = "范围,原因\nT12收益检验,完整用途链和自由流通分母仍缺\n31份候选的完整对象复核,本轮仅七个已定位目标\n大全公告号的推测修正,源PDF印为2023-0XX保留未知\n计划日期自动变已完成,不符合原文证据\n后续登记的独立确认,三份仅有注销安排的公告未补该确认\n历史首次发布版本,现有抓取及名义日期不能证明\n全指数完整回购覆盖,七个目标不是全体原方案\n市场价格与旧账户重新运行,本轮没有涉及\n外部审阅及上传与真实订单,未执行\n"
    reproduce = ("全新解压目录的只读核查命令：\n\npython " + VERIFY + " --root " + OUT.relative_to(ROOT).as_posix()
                 + "\n\n该核查仅需Python标准库，不请求网络、不拟合、不追加账户、不修改输入。若另检查原研究代码环境，pandas和pytest版本见REPRODUCTION_ENVIRONMENT.json。不要重新运行prepare、freeze、run或report；保存文件采用不可覆盖设计。旧账户在原快照中已有核查回执，本轮不重新计算账户。\n")
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
              "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
              "02_研究结论.md": (OUT / "研究结论.md").read_bytes(),
              "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"), "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"),
              "REPRODUCTION_ENVIRONMENT.json": json.dumps({"python": platform.python_version(), "platform": platform.system(),
                  "read_only_verifier": "Python标准库", "packages": {p: version(p) for p in ["pandas", "pytest", "pypdfium2"]}},
                  ensure_ascii=False, indent=2).encode("utf-8")}
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
    destination = ROOT / "outputs" / ("factor96_repurchase_change_verify_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(FINAL) as archive:
        assert all((destination / name).resolve().is_relative_to(destination.resolve()) for name in archive.namelist())
        archive.extractall(destination)
    print("CRC、索引与文件身份通过；新解压目录只读核查七个原方案与68个时序查询。", flush=True)
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
               "prior_zip_bytes": PRIOR_IDENTITY[0], "prior_zip_sha256": PRIOR_IDENTITY[1],
               "new_accounts_this_round": 0, "formal_accounts": 432, "invalid_implementation_accounts": 152,
               "executed_accounts": 584, "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED",
               "orders_authorized": False, "prior_account_reverification": "NOT_RERUN_UNCHANGED_PRIOR_ZIP_AND_RECEIPT_PRESERVED"}
    with (OUT / "delivery_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
