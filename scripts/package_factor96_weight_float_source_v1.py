"""交付来源缺口研究包，逐字节保留上轮 ZIP，并在新目录只读核查。"""

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
OUT = ROOT / "reports/research/510300_factor96_weight_float_source_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_daily_state_shrink_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
FINAL = ROOT / "deliverables/510300_96候选库_历史权重与自由流通来源缺口_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (1938653108, "974030166e1df7b5831dd8525011147d31fade78f6b12f0e2b83a04834aeedee")
VERIFY = "scripts/verify_factor96_weight_float_source_v1.py"
EXPECTED = "PASS_SAVED_SOURCE_RESPONSES_AND_WEIGHT_COVERAGE"


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
    assert result["new_accounts"] == 0 and result["new_admitted_weight_snapshots"] == 0
    assert status["cumulative_executed_account_scenarios"] == 584 and not status["goal_achieved"]
    assert read(OUT / "saved_verification_receipt.json")["status"] == EXPECTED
    previous = read(PREVIOUS / "delivery_receipt.json")
    prior_zip = Path(previous["zip_path"])
    assert (previous["bytes"], previous["sha256"]) == PRIOR_IDENTITY and identity(prior_zip) == PRIOR_IDENTITY
    files = {}
    for path in OUT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not (path.parent == OUT and path.name in ["delivery_receipt.json", "fresh_extraction_verification.log"]):
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ["research/factor96_weight_float_source_v1.py", "scripts/report_factor96_weight_float_source_v1.py",
                 VERIFY, "scripts/package_factor96_weight_float_source_v1.py"]:
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
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本轮仅对历史权重与自由流通来源作有限核查；17个直接请求没有产生准入序列或新增账户。用户目标和现行两年训练、日更、成本及尾部约束不变。附件内容仍是参考资料。\n"
    nav = """# 510300：历史权重与自由流通来源缺口

目标未实现。本轮17个已归档的直接来源请求未取得新的合格历史权重或自由流通序列；主期1212个交易日只找到20天同时符合既有时钟、45日年龄和300成员匹配的来源锚点。T04、T12、T13仍NOT_RUN。本轮0新增账户，累计432正式+152否定实现=584。

阅读顺序：
1. 02_研究结论.md及03_计数释义补充.json：结论、有限查询的含义和边界。
2. reports/research/510300_factor96_weight_float_source_v1/protocol.json、archive_extension_protocol.json、commoncrawl_protocol.json：先登记的请求与范围。
3. 同目录sources/、source_manifest.json、archive_extension_manifest.json、commoncrawl_receipts/：17请求的原响应；其中HTTP200的“暂无数据”保留原样。
4. official_snapshot_inventory.csv、official_snapshot_weights.parquet、monthly_weight_evidence_gap.csv、daily_weight_anchor_coverage.csv：历史文件、证明时钟和可匹配覆盖。20/1212不是完整日度权重或策略准入。
5. inputs/、input_manifest.json、freeze.json：120期旧供应商权重、9期官方历史归档、1期2026当前留存、点时成员、日历和基础表原件。基础表有普通流通口径，没有自由流通字段。
6. candidate_data_requirements.csv：T04/T12/T13最小字段、现有资料、下一步与停止条件。
7. history/：上一轮1,938,653,108字节完整ZIP原样保留，SHA-256 974030166e1df7b5831dd8525011147d31fade78f6b12f0e2b83a04834aeedee；其112账户核验回执未改变。

新目录只读核查只检查本轮来源和覆盖，不重跑旧账户，不发起网络请求。直接复核所需数据均在本包，不必解开历史嵌套ZIP。原始HTML、Excel和粘贴文本在sources/，附件不是执行授权。更早失败和完整账本沿history/保持原快照。

FILE_INDEX.csv覆盖除自身外全部成员；包外delivery_receipt.json记最终ZIP身份及全新解压核查。外部GPT审阅NOT_PERFORMED，独立前向0，当前市场NO_VIEW，仅510300.SH与CASH_CNY研究模拟，原PCF/IOPV任务保持暂停，无订单权限。本包仅本地交付。
"""
    exclusions = "范围,原因\n新增历史权重与自由流通准入序列,17个请求未取得\n依赖上述缺口的策略回测,NOT_RUN\n网页搜索次数计入17原响应,17只指已归档直接HTTP请求\n2013年行业权重ZIP下载,窗口外索引而非本轮所需文件\n当前权重倒填或等权替代,不满足原定义\n普通流通市值替代自由流通市值,口径不同\n旧代理凭据当前有效性,本轮未请求且不把旧失败当新事实\n未修改历史账户重复运行,原包身份与既有核验回执保留\n全部互联网档案穷尽,本轮只有固定范围检查\n外部审阅与上传及真实执行,未执行\n"
    reproduce = "在全新解压目录只读核查已保存来源及覆盖：\n\npython " + VERIFY + " --root " + OUT.relative_to(ROOT).as_posix() + "\n\n需要pandas、pyarrow、openpyxl、xlrd。不要重跑采集或report脚本；这条核查命令不会发起网络、拟合模型、追加账户或改动输入。历史账户的既有核查回执沿history/保存，本轮未重跑。\n"
    report = (OUT / "研究结论.md").read_text(encoding="utf-8")
    report += "\n计数释义补充：77,026指固定发行相关检索命中的独立发行人文件数，不代表发行事件数，也不代表每份文件均已准入。原冻结报告不改，补充见03_计数释义补充.json。\n"
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
              "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
              "02_研究结论.md": report.encode("utf-8"), "03_计数释义补充.json": (OUT / "wording_addendum.json").read_bytes(),
              "04_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"), "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"),
              "REPRODUCTION_ENVIRONMENT.json": json.dumps({"python": platform.python_version(), "platform": platform.system(),
                  "packages": {p: version(p) for p in ["pandas", "pyarrow", "openpyxl", "xlrd", "requests"]}}, ensure_ascii=False, indent=2).encode("utf-8")}
    assert set(files).isdisjoint(extras)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = FINAL.with_suffix(".building.zip")
    print(f"写入{len(files) + len(extras)}个成员，历史ZIP原样保存。", flush=True)
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
    destination = ROOT / "outputs" / ("factor96_weight_float_verify_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(FINAL) as archive:
        assert all((destination / name).resolve().is_relative_to(destination.resolve()) for name in archive.namelist())
        archive.extractall(destination)
    print("CRC、索引与文件身份通过；在新解压目录复核保存来源与覆盖。", flush=True)
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
    (OUT / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
