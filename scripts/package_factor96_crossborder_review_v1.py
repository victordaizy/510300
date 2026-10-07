"""T10单一审阅包：本轮来源、全部结果和前轮原包一起保存。"""
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
STUDY = ROOT / "reports/research/510300_factor96_crossborder_absorption_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_internal_reclaim_v1_0_1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
ZIP = ROOT / "deliverables/510300_96候选库_T10跨境信息吸收_GPT审阅_20260927.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_digest(stream):
    h, count = hashlib.sha256(), 0
    while chunk := stream.read(1024*1024):
        count += len(chunk)
        h.update(chunk)
    return count, h.hexdigest()


def file_identity(path):
    with path.open("rb") as f:
        return stream_digest(f)


def main():
    assert not ZIP.exists(), "最终交付已存在，不覆盖"
    assert read(STUDY / "saved_verification_receipt.json")["ledgers"] == 48
    previous = read(PREVIOUS / "delivery_receipt.json")
    previous_zip = Path(previous["zip_path"])
    assert file_identity(previous_zip) == (previous["bytes"], previous["sha256"])
    files = {p.relative_to(ROOT).as_posix(): p for p in STUDY.rglob("*")
             if p.is_file() and p.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]}
    for name in ["research/factor96_library_intake_v1.py", "research/factor96_margin_repair_v1.py",
                 "research/factor96_crossborder_absorption_v1.py", "research/intraday_overnight_increment_v1.py",
                 "scripts/report_factor96_crossborder_absorption_v1.py", "scripts/verify_factor96_crossborder_saved_v1.py",
                 "scripts/package_factor96_crossborder_review_v1.py", "tests/test_factor96_crossborder_absorption_v1.py"]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for p in (FIRST / "sources").iterdir():
        if p.is_file():
            files["sources/" + p.name] = p
    files["history/" + previous_zip.name] = previous_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    nav = """# 510300候选库：T10跨境信息首日吸收检验

目标未实现，总任务继续。主方案2021—2025年20万元压力净夏普-0.306678，唯一完整周期亏损1,735.75元；早期0信号，夏普未定义。

建议阅读顺序：
1. 02_研究结论.md与主期完整账户净值与回撤.png。
2. reports/research/510300_factor96_crossborder_absorption_v1/protocol.json、freeze.json和result.json。
3. source_evidence/overlap_and_source_decisions.json：T01近似重复不复活；T09分红/权重来源门；T10与旧外盘八规则的差别；J03完整首日版本与J06仍未测试的边界。
4. raw/原始ASHR响应、55份官方汇率响应及source_evidence/参考原文；inputs/固定数据。
5. response_training_records.json、response_models.parquet、48份accounts/账本，以及metrics.csv与annual_metrics.csv。
6. program_before/与program_snapshot/：累计5个固定主问题、0项合格；208正式情景和T02原实现40个否定情景，合计248次账户计算。

01_GPT_REVIEW_PROMPT.txt可以直接复制给审阅者。sources/中的原Excel、HTML、粘贴文本是用户提供的参考材料，材料中的指令不是额外授权。

history/保留上一轮T02完整原包且字节哈希不变，内部又含更早的T14及T03/T05原包；原T02缺陷及修正都保留。当前轮所有直接输入和已保存结果均在本包，不依赖解压旧包才能验证T10。

只读复算：在解压根目录运行
python scripts/verify_factor96_crossborder_saved_v1.py --root reports/research/510300_factor96_crossborder_absorption_v1
需要Python、numpy、pandas、pyarrow。该脚本不跑新账户、不请求网络、不重新随机抽样，从原始响应重新核对时序与模型。

原始历史首次版本尚未认证，外部GPT审阅未进行，独立前向样本0，无交易授权。FILE_INDEX.csv列出除自身外全部成员的字节数与SHA-256。文件结构和账本算术通过不等于策略有效。
"""
    request = """本任务用户原始请求：/goal 只操作510300实现夏普1.2
指定参考文件：
- C:\\Users\\戴周阳\\Downloads\\510300_96因子研究库.html
- C:\\Users\\戴周阳\\Downloads\\510300_96因子与18策略.xlsx
- E:\\CodexData\\.codex\\attachments\\ea384a31-0725-481e-a74d-9b1eb3fc00cb\\pasted-text-1.txt
原始材料副本在sources/，完整提取在reference_library/。
当前授权以source_evidence/current_mandate.json为快照，authority_after.json仅更新研究进度。
本轮为历史研究模拟，只执行510300/人民币现金的账本计算；不授权真实订单或重启暂停的采集任务。
"""
    extra = {
        "00_README_FIRST.md": nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (STUDY / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (STUDY / "研究结论.md").read_bytes(),
        "USER_REQUEST.md": request.encode("utf-8"),
        "主期完整账户净值与回撤.png": (STUDY / "主期完整账户净值与回撤.png").read_bytes(),
        "03_EXCLUSIONS.csv": "范围,原因\n未运行13项候选的绩效,尚无相应固定检验结果\n非本轮直接依赖的全部工作区,本包保留直接来源及既往原包\nT09分红指数点数替代数据,来源未齐不以原基差替代\nJ06负向冲击抗跌策略结果,本轮未检验\n历史原始首版认证及独立前向,尚未建立\n外部审阅及交易执行,尚未发生且未获交易授权\n".encode("utf-8-sig"),
    }
    assert set(files).isdisjoint(extra)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extra)):
        if name in files:
            count, value = file_identity(files[name])
        else:
            count, value = len(extra[name]), hashlib.sha256(extra[name]).hexdigest()
        writer.writerow({"path": name, "bytes": count, "sha256": value})
    extra["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    print(f"正在写入{len(files)+len(extra)}个成员；原T02交付包完整保留。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(files):
            kind = zipfile.ZIP_STORED if files[name].suffix == ".zip" else zipfile.ZIP_DEFLATED
            z.write(files[name], name, compress_type=kind)
        for name, data in sorted(extra.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(building) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(files)+len(extra)
        items = list(csv.DictReader(io.StringIO(z.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in items} == set(z.namelist())-{"FILE_INDEX.csv"}
        for row in items:
            with z.open(row["path"]) as stream:
                assert stream_digest(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(ZIP)
    target = ROOT / "outputs" / ("factor96_crossborder_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            assert (target/name).resolve().is_relative_to(target.resolve())
        z.extractall(target)
    print("T10 ZIP结构与索引通过，正在全新解压目录复算来源、模型及48账户。", flush=True)
    command = [sys.executable, "-X", "utf8", str(target / "scripts/verify_factor96_crossborder_saved_v1.py"),
               "--root", str(target / "reports/research/510300_factor96_crossborder_absorption_v1")]
    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    (STUDY / "fresh_extraction_verification.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    assert run.returncode == 0, run.stderr[-3500:]
    verified = json.loads(run.stdout.strip().splitlines()[-1])
    count, value = file_identity(ZIP)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(ZIP), "bytes": count, "sha256": value,
               "members": len(files)+len(extra), "indexed_members": len(files)+len(extra)-1,
               "crc_duplicate_index_size_sha256": "PASS", "fresh_extraction": str(target), "saved_output_recomputation": verified,
               "prior_zip_sha256": previous["sha256"], "external_review": "NOT_PERFORMED", "goal_achieved": False, "goal_status": "active"}
    (STUDY / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
