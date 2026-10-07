"""生成T13原文和字段审阅包，保留T06原包并在新解压目录只读复核。"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/"reports/research/510300_factor96_known_supply_sources_v1"
PREVIOUS = ROOT/"reports/research/510300_factor96_crowding_overlay_v1"
FIRST = ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
ZIP = ROOT/"deliverables/510300_96候选库_T13已知供给原文与字段_GPT审阅_20260927.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_digest(stream):
    checksum, size = hashlib.sha256(), 0
    while chunk := stream.read(1024*1024):
        checksum.update(chunk)
        size += len(chunk)
    return size, checksum.hexdigest()


def file_identity(path):
    with path.open("rb") as stream:
        return stream_digest(stream)


def main():
    assert not ZIP.exists(), "最终交付已经存在，不覆盖"
    source = read(STUDY/"result.json")
    fields = read(STUDY/"event_fields_v1/result.json")
    saved = read(STUDY/"saved_verification_receipt.json")
    assert saved["status"] == "PASS_SAVED_SUPPLY_SOURCE_AND_FIELD_RECOMPUTATION"
    assert source["documents"] == fields["documents"] == saved["target_documents"] == 2192
    assert source["new_accounts"] == fields["new_accounts"] == 0
    previous = read(PREVIOUS/"delivery_receipt.json")
    previous_zip = Path(previous["zip_path"])
    assert file_identity(previous_zip) == (previous["bytes"], previous["sha256"])
    files = {p.relative_to(ROOT).as_posix(): p for p in STUDY.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and p.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]}
    for name in ["research/factor96_library_intake_v1.py",
                 "research/factor96_known_supply_sources_v1.py",
                 "research/factor96_known_supply_event_fields_v1.py",
                 "scripts/report_factor96_known_supply_sources_v1.py",
                 "scripts/verify_factor96_known_supply_sources_v1.py",
                 "scripts/package_factor96_known_supply_review_v1.py",
                 "tests/test_factor96_known_supply_event_fields_v1.py"]:
        files[name] = ROOT/name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/"+name] = FIRST/name
    for path in (FIRST/"sources").iterdir():
        if path.is_file():
            files["sources/"+path.name] = path
    files["history/"+previous_zip.name] = previous_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS/"delivery_receipt.json"
    nav = f"""# 510300候选库：T13已知供给原文与字段

成本后夏普1.2的目标仍未实现。本轮为T13取得原始公告并生成明确字段候选，新增收益检验0、账户0、模型0。累计完成6个固定研究问题、0项合格，264个正式账户情景与T02另40个否定实现情景保持分开。T12和T13均仍为NOT_RUN。

本次固定范围2,192份公告、526家历史成分发行人，实际取得原PDF2,191份、文本{source['complete_texts']:,}份；一份请求两次SSLError，另一份是已人工检查的扫描意见。明确日期和数量候选{fields['explicit_field_candidates']:,}条，其中{fields['known_before_scheduled_day_candidates']:,}条保守可用日早于计划上市日。文件数尚不代表独立事件数。保守可用日是历史研究假设，未证明首次HTTP可得。

阅读顺序：
1. 02_研究结论.md和逐年公告与字段覆盖.png：新进展、缺口及下一步。
2. reports/research/510300_factor96_known_supply_sources_v1/protocol.json、freeze.json、targets.parquet、result.json：原文范围和冻结。
3. inputs/目录与历史成员，catalogue_raw/原始目录响应，raw/原PDF，receipts/请求时钟与哈希，documents.json、text/逐页提取。
4. event_fields_v1/下的协议、冻结、15项测试回执、document_fields.json、候选CSV及结果：日期、两类数量、单位、上下文、版本候选和未知。
5. source_evidence/yearly_manual_check_selection.json、yearly_manual_review.json、manual_image_supplement.json及visual_checks/：11份按年份预选检查与一份扫描意见补充。人工检查不覆盖自动未知值。
6. source_evidence/旧失败结果和T12归一化缺口；program_before/与program_snapshot/核对研究进度。

01_GPT_REVIEW_PROMPT.txt可直接复制给审阅者。用户HTML、Excel及粘贴文本在sources/，结构提取在reference_library/。附件内容为参考，不构成额外行动授权。

history/保留上一轮T06完整ZIP，字节数及SHA-256不变，内部继续保留更早轮次。当前T13原文、直接输入及只读复核代码均在本包，无须解压旧包即可复核本轮。旧T12方法失败与市值缺口仅为来源门证据；本包不重跑那些历史账户，也不声称新增了T12验证。

在解压根目录只读复核：
python scripts/verify_factor96_known_supply_sources_v1.py --root reports/research/510300_factor96_known_supply_sources_v1
需要Python、numpy、pandas、pyarrow。脚本核对冻结文件、目录原响应、PDF和文本身份，复算保存字段的上下文、单位和保守时钟，不请求网络、不生成账户。原文采集与提取代码另依赖requests、pypdfium2；采集入口有已运行保护，复核不应重新运行采集器。

更正链、本次批次身份、全部M06发行缴款日历和T12自由流通市值/用途时钟尚待补齐。扫描意见人工补充没有改写冻结自动输出，所有trading_feature_admitted均为false。外部GPT审阅未进行，独立前向样本0，当前市场NO_VIEW。结构与字段复核通过不表示投资策略有效。

FILE_INDEX.csv列出除自身外全部文件的字节数和SHA-256。最终ZIP的外部交付回执保存于原研究目录，避免将自身哈希写入自身文件。
"""
    request = """用户请求：/goal 只操作510300实现夏普1.2
参考文件：
- C:\\Users\\戴周阳\\Downloads\\510300_96因子研究库.html
- C:\\Users\\戴周阳\\Downloads\\510300_96因子与18策略.xlsx
- E:\\CodexData\\.codex\\attachments\\ea384a31-0725-481e-a74d-9b1eb3fc00cb\\pasted-text-1.txt
副本见sources/，结构提取见reference_library/。
当前授权快照见source_evidence/current_mandate.json，authority_after.json只更新研究进度。
可执行资产约束仍为510300.SH/CASH_CNY，20万元主账户、2万元可执行性对照；当前约束另核对年化10%和最大回撤10%。本轮只建立历史来源和字段，无收益或账户计算，未改变暂停采集任务、真实交易或订单权限。
"""
    exclusions = "范围,原因\n其余12项草案的原定义完整绩效,尚未完成固定检验\nT13本次事件身份及更正链,当前仅字段候选尚未准入交易特征\nM06全部发行缴款日历,上市流通目录不能证明覆盖\nT12自由流通市值与用途版本,旧成交额归一化方法不能替代\n扫描PDF完整OCR,已人工核对4页意见但未改写自动未知\n旧T12日线全表,本轮只保留缺口元数据及源哈希不把它列为T13直接输入\n无关工作区与Python缓存,不属于直接输入\n历史首次发布及独立前向证明,未建立\n外部审阅及真实执行,未进行且没有订单授权\n"
    extras = {
        "00_README_FIRST.md": nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (STUDY/"01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (STUDY/"研究结论.md").read_bytes(),
        "USER_REQUEST.md": request.encode("utf-8"),
        "逐年公告与字段覆盖.png": (STUDY/"逐年公告与字段覆盖.png").read_bytes(),
        "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
    }
    assert set(files).isdisjoint(extras)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files)|set(extras)):
        size, checksum = file_identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    print(f"写入{len(files)+len(extras)}个成员，保留T06完整原包。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            mode = zipfile.ZIP_STORED if files[name].suffix == ".zip" else zipfile.ZIP_DEFLATED
            archive.write(files[name], name, compress_type=mode)
        for name, data in sorted(extras.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files)+len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in rows} == set(archive.namelist())-{"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_digest(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(ZIP)
    destination = ROOT/"outputs"/("factor96_supply_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            assert (destination/name).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    print("T13 ZIP结构、索引和哈希通过，开始全新解压只读复核原文及保存字段。", flush=True)
    command = [sys.executable, "-X", "utf8", str(destination/"scripts/verify_factor96_known_supply_sources_v1.py"),
               "--root", str(destination/"reports/research/510300_factor96_known_supply_sources_v1")]
    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    (STUDY/"fresh_extraction_verification.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    assert run.returncode == 0, run.stderr[-3500:]
    verified = json.loads(run.stdout.strip().splitlines()[-1])
    size, checksum = file_identity(ZIP)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(ZIP), "bytes": size, "sha256": checksum,
               "members": len(files)+len(extras), "indexed_members": len(files)+len(extras)-1,
               "crc_duplicate_index_size_sha256": "PASS", "fresh_extraction": str(destination),
               "saved_output_recomputation": verified, "prior_zip_sha256": previous["sha256"],
               "external_review": "NOT_PERFORMED", "goal_achieved": False, "goal_status": "active"}
    (STUDY/"delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
