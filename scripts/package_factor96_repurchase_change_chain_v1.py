"""交付回购变更来源包，保留旧包身份并在全新目录只读核查。"""
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
OUT = ROOT / "reports/research/510300_factor96_repurchase_change_chain_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_weight_float_source_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
FINAL = ROOT / "deliverables/510300_96候选库_回购变更对象与审批状态_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (1958123478, "20cdd3fbddeeffd4e8454ad7821fafa7fb48c5962102e4dbeee327392210c170")
VERIFY = "scripts/verify_factor96_repurchase_change_chain_v1.py"
EXPECTED = "PASS_SAVED_CHANGE_FIELDS_EVIDENCE_AND_ASOF_QUERIES"


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
    assert result["new_accounts"] == result["new_returns"] == result["new_network_requests"] == 0
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
    for name in ["research/factor96_repurchase_change_chain_v1.py", "tests/test_factor96_repurchase_change_chain_v1.py",
                 "scripts/report_factor96_repurchase_change_chain_v1.py", VERIFY,
                 "scripts/package_factor96_repurchase_change_chain_v1.py"]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + prior_zip.name] = prior_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    for item in read(OUT / "freeze.json")["files"]:
        path = OUT / item["path"]
        assert path.relative_to(ROOT).as_posix() in files
        assert identity(path) == (item["bytes"], item["sha256"])
    for item in read(OUT / "prior_account_evidence/manifest.json")["files"]:
        assert identity(OUT / item["path"]) == (item["bytes"], item["sha256"])
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本轮仅复核已有回购公告中的实际变更对象、审批状态、数量和版本时序；0新网络请求、0新账户，T12仍NOT_RUN。用户目标与现行账户合同未改变。附件是参考材料，不是操作指令。\n"
    nav = """# 510300：回购变更对象与审批状态

目标尚未实现。54份候选全保留，23份逐条复核；17份确认对应15个原方案，相比上一版新增16条关联。3份公告中提到的4个方案仅为背景，已拒绝自动关联。0新账户、0收益计算、0网络请求，T12仍NOT_RUN，所有来源字段尚未准入交易因子。

阅读顺序：
1. 02_研究结论.md：本轮结论、具体语义修正、时间范围与下一步停止条件。
2. reports/research/510300_factor96_repurchase_change_chain_v1/protocol.json、freeze.json、prefreeze_test_receipt.json：研究范围、259个冻结文件与9项边界测试。
3. review_cards.json、change_ledger.json、逐公告变更对象与审批状态.csv：23张人工复核卡、全部54条台账；31份尚未完整复核、7个明确目标缺原方案，均未删去。
4. inputs/raw、inputs/text、inputs/receipts、inputs/documents.json：80份来源PDF、逐页文本、原始收据与哈希。91条短语、135处页内定位可追溯到原文。
5. visual/与visual_review_receipt.json：3页数量、对象、预算疑点的渲染核对。
6. asof_change_queries.json、calendar_scope.json：34个时序查询；泰格同日两份文件保留歧义，2026信息不回填2021至2025。
7. prior_account_evidence/：最近失败的日更变体结果与112行账户指标，沿用旧结果未重跑；main20万元净夏普约-0.0070。
8. identified_missing_originals.json、01_GPT_REVIEW_PROMPT.txt：7个明确缺口、批判性复核要求和有限下一步。
9. history/：上轮1,958,123,478字节ZIP逐字节保留，SHA-256 20cdd3fbddeeffd4e8454ad7821fafa7fb48c5962102e4dbeee327392210c170。更早完整账本及失败沿原快照链保留。

上述同目录文件均位于reports/research/510300_factor96_repurchase_change_chain_v1/。本轮来源复核无需解开历史ZIP。原始HTML、Excel、粘贴文本与96/18注册表在sources/及reference_library/。

累计432正式+152否定实现=584账户情景，不是独立策略数。既有执行用途569已知、379未知的口径不变。外部审阅NOT_PERFORMED、独立前向0、当前市场NO_VIEW。仅510300.SH和CASH_CNY研究模拟，采集计划保持暂停，无订单权限。

FILE_INDEX.csv覆盖除自身外的全部文件。结构检查和新目录只读核查不证明外部审阅、完整历史时钟或策略有效性。最终ZIP身份在包外delivery_receipt.json。本包仅本地交付，无既定外部上传体积上限。
"""
    exclusions = "范围,原因\n新增策略与收益计算,来源条件尚未补齐且本轮只作字段整理\n31份候选完整对象复核,无既有原方案候选关系而本轮未逐条完成\n7个已明确目标的缺失原方案,未取得或未定位原件\n完整后续批准登记和修订链,未建立\n历史首次发布时间证明,现有known_at仅沿用保守元数据\n自由流通分母,仍缺且不以普通流通口径替代\n同日多版本唯一先后顺序,证据不足保持歧义\n2026公告回填历史主期,不符合时序\n旧账户再次回测,旧结果和原包身份保留未重跑\n外部审阅及上传和订单执行,未执行\n"
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
    print("CRC、索引与文件身份通过；新解压目录只读核查54条台账。", flush=True)
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
