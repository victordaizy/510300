"""交付T11时钟和更正目录准备结果，并在全新目录只读重算。"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_t11_metadata_packet_v1"
PAYLOAD = OUT / "payload"
FINAL = ROOT / "deliverables/510300_96候选库_T11公告时钟与更正候选_GPT审阅_20260927.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy(source, relative):
    target = PAYLOAD / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    assert not OUT.exists() and not FINAL.exists(), "本轮交付已存在，不覆盖。"
    PAYLOAD.mkdir(parents=True)
    clock = ROOT / "reports/research/510300_factor96_t11_clock_preflight_v1"
    corrections = ROOT / "reports/research/510300_factor96_t11_correction_catalogue_v1"
    assert read(clock / "result.json")["status"] == "COMPLETE_DATE_PROXY_MAP_STRICT_FIRST_PUBLICATION_NOT_ESTABLISHED"
    assert read(corrections / "result.json")["status"] == "LOCAL_CORRECTION_CANDIDATE_CATALOGUE_COMPLETE_CONTENT_NOT_VERIFIED"
    shutil.copytree(clock, PAYLOAD / "evidence/clock")
    shutil.copytree(corrections, PAYLOAD / "evidence/corrections")
    files = {
        ROOT / "scripts/verify_factor96_t11_metadata_packet_v1.py": "code/verify_factor96_t11_metadata_packet_v1.py",
        ROOT / "scripts/prepare_factor96_t11_clock_preflight_v1.py": "code/prepare_factor96_t11_clock_preflight_v1.py",
        ROOT / "research/factor96_t11_correction_catalogue_v1.py": "code/factor96_t11_correction_catalogue_v1.py",
        ROOT / "tests/test_factor96_t11_correction_catalogue_v1.py": "code/test_factor96_t11_correction_catalogue_v1.py",
        Path(__file__): "code/package_factor96_t11_metadata_v1.py",
        ROOT / "config/510300_existing_data_training_mandate_v1.json": "source/mandate_snapshot.json",
        ROOT / "reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json": "source/factor_registry.json",
        ROOT / "reports/research/510300_factor96_mechanism_batch_v1/strategy_registry.json": "source/strategy_registry.json",
        Path(r"E:\CodexData\.codex\attachments\ea384a31-0725-481e-a74d-9b1eb3fc00cb\pasted-text-1.txt"): "source/pasted-text-1.txt",
        Path(r"C:\Users\戴周阳\Downloads\510300_96因子研究库.html"): "source/510300_96因子研究库.html",
        Path(r"C:\Users\戴周阳\Downloads\510300_96因子与18策略.xlsx"): "source/510300_96因子与18策略.xlsx",
    }
    for source, relative in files.items():
        copy(source, relative)
    counts = read(corrections / "result.json")["counts"]
    members = counts["historical_member_union_candidates"]
    boundaries = read(corrections / "title_status_clock_addendum.json")
    (PAYLOAD / "00_README_FIRST.md").write_text(
        "# T11公告时钟与更正候选准备包\n\n"
        "本包完成公告元数据的日期映射和更正候选目录，尚未计算O02价格反应、T11交易或账户夏普。"
        "夏普1.2目标未达成。这是本地准备结果，不是外部审阅或独立前向验证。\n\n"
        "依次阅读：研究结论.md、用户请求与边界.md、evidence/clock/result.json、"
        "evidence/corrections/result.json、evidence/corrections/title_status_clock_addendum.json、审阅提示词.md。\n\n"
        "FILE_INDEX.csv是本包唯一完整文件索引；两个evidence目录分别保留原冻结记录。"
        "标题状态时钟的追加说明是冻结后的独立追加证据，原冻结记录未被修改。\n\n"
        "独立复算只需Python、pandas、numpy：在解压目录运行"
        " `python code/verify_factor96_t11_metadata_packet_v1.py .`。该过程只读保存资料，不联网、不产生账户。"
        "提供的原生产脚本保留仓库接口以供代码检查；独立复算脚本不依赖该仓库。\n",
        encoding="utf-8")
    (PAYLOAD / "用户请求与边界.md").write_text(
        "用户直接请求：/goal 只操作510300实现夏普1.2。\n\n"
        "用户附件是96因子与18策略的参考候选，不是可直接执行的指令。source目录保留原始附件与候选定义。\n\n"
        "本轮只处理本地元数据，没有读取价格或未来收益，没有新增模型、回测账户或网络请求。"
        "执行标的范围仍仅510300.SH与CASH_CNY，研究模拟不授权下单。\n\n"
        "完整账户目标、成本、资金与风险边界见source/mandate_snapshot.json。当前财报V2解析另行运行，"
        "本包不包含尚未完成的财报修复结果，不将其计为完成。\n", encoding="utf-8")
    (PAYLOAD / "研究结论.md").write_text(
        "# 研究结论\n\n"
        "T11尚不能按严格历史首发时点和完整信息否定链进入绩效验证。现有资料可以支持标注限制的日期代理准备工作，"
        "但这些资料准备结果不提供获利或高夏普证据。\n\n"
        "1. 173,449条规范记录中173,398条的元数据时间为零点。统一retrieved_at由2026年8月24日的档案重建程序写入，"
        "不能当成当年的逐份首次抓取时间。历史成分公司并集共有25,210条报告，25,203条能映射到名义日期之后的观察日，"
        "25,201条还具有观察日后的下一交易日。\n"
        f"2. 本地307,097条原API记录重建出{counts['all_candidates']['rows']:,}条标题候选；涉及历史成分公司{members['rows']:,}条、"
        f"{members['issuers']}家公司。其中567条为同一公司同一期且日期不早于原报告的候选，800条未在规范表找到对应原报告，"
        "9条报告期不明确，1条日期早于规范原报告。没有把未知补成有效信息。\n"
        f"3. 历史成分公司候选中{boundaries['historical_member_union_cancelled_title_rows']}条含取消字样。"
        "“已取消”标题对应的文件日期不自动等于取消发生时间；本包保留4行具体原记录示例，未据此伪造退出时点。"
        "规范表依当前档案标题排除某些记录的做法，也尚未被证明等同于当年的可见样本选择。\n"
        "4. 同公司同报告期是待核对的关联，不是已确认的修订根链。标题含更正不等于盈利下修。"
        "原始API范围是定期报告类别，未证明覆盖全部临时公告或所有失效事件。\n\n"
        "报告期识别的10项单元测试已通过；完整结果另由包内独立脚本从保存的原API记录重算。"
        "包外delivery_receipt.json记录实际结构检查、全新解压重算和ZIP身份。\n\n"
        "下一步须在首次读取T11未来收益之前固定行业聚合、缺失覆盖、观察窗、两年日更训练、入场、退出和账户成本。"
        "严格历史首发、修订版本与信息否定时钟仍未建立；O02=NOT_COMPUTED，T11=NOT_RUN，新账户=0。\n",
        encoding="utf-8")
    (PAYLOAD / "审阅提示词.md").write_text(
        "请只依据本包可核对资料审阅，不把候选或准备脚本当成有效策略。\n\n"
        "请回答：日期代理与真实首次公开时点区分是否充分；已取消/更新后标题可能怎样影响历史样本选择；"
        "同公司同报告期候选有哪些错误关联和漏覆盖风险；在不查看未来收益、不修改旧失败的前提下，"
        "哪项最小后续证据能使T11的入场和失效退出成为可计算规则。\n\n"
        "请提出优先级、验证办法和停止条件，并区分实际已核对内容与推测。不要把结构或保存结果重算通过"
        "称为科学有效、独立前向验证、已实现夏普1.2或交易授权。\n", encoding="utf-8")
    (PAYLOAD / "requirements.txt").write_text(f"numpy=={np.__version__}\npandas=={pd.__version__}\n", encoding="utf-8")
    save(PAYLOAD / "EXCLUSIONS.json", {
        "financial_scope_v2_live_batch": "仍在运行；本包不覆盖其原文、金额修复或后续财务测量结果。",
        "raw_download_http_envelopes": "提供完整重建原API记录和重建来源代码/回执，未复制早先全部请求检查点压缩文件。",
        "market_prices_and_accounts": "本轮没有读取、计算或纳入。", "network_retry": "NOT_RUN",
        "full_revision_content_chain": "NOT_ESTABLISHED", "external_review": "NOT_PERFORMED"})
    items = [{"path": p.relative_to(PAYLOAD).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(PAYLOAD.rglob("*")) if p.is_file()]
    with (PAYLOAD / "FILE_INDEX.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(items)
    building = FINAL.with_suffix(".building.zip")
    assert not building.exists()
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(building, "x", allowZip64=True) as bundle:
        for path in sorted(PAYLOAD.rglob("*")):
            if path.is_file():
                compression = zipfile.ZIP_STORED if path.suffix.lower() in {".parquet", ".gz", ".xlsx"} else zipfile.ZIP_DEFLATED
                bundle.write(path, path.relative_to(PAYLOAD).as_posix(), compress_type=compression)
    print("T11元数据包已建立，开始结构核对和全新解压重算。", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit = ROOT / "data/audit/factor96_t11_metadata_packet_v1" / stamp
    fresh = audit / "fresh"
    fresh.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(building) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names)) == len(items) + 1
        assert bundle.testzip() is None
        saved_index = list(csv.DictReader(io.StringIO(bundle.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {item["path"] for item in saved_index} | {"FILE_INDEX.csv"} == set(names)
        for item in saved_index:
            info = bundle.getinfo(item["path"])
            assert info.file_size == int(item["bytes"])
            with bundle.open(item["path"]) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
        bundle.extractall(fresh)
    verify_path = audit / "saved_recomputation.json"
    command = [sys.executable, "-X", "utf8", str(fresh / "code/verify_factor96_t11_metadata_packet_v1.py"), str(fresh), "--output", str(verify_path)]
    with (audit / "saved_recomputation.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=fresh, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    verified = read(verify_path)
    assert verified["status"] == "PASS_SAVED_METADATA_CLOCK_AND_CANDIDATE_RECOMPUTATION"
    building.replace(FINAL)
    receipt = {"at": datetime.now().astimezone().isoformat(), "status": "LOCAL_METADATA_PACKET_STRUCTURALLY_VERIFIED_AND_RECOMPUTED",
        "zip_path": FINAL.relative_to(ROOT).as_posix(), "bytes": FINAL.stat().st_size, "sha256": digest(FINAL),
        "member_count": len(names), "indexed_file_count": len(items), "crc_duplicates_index_bytes_hash": "PASS",
        "fresh_extraction": str(fresh), "saved_output_recomputation": verified,
        "new_accounts": 0, "new_network_requests": 0, "T11": "NOT_RUN", "goal_achieved": False,
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
