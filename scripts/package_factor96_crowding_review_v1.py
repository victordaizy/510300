"""生成T06单一审阅包，保留T10原包并在全新解压目录复算。"""
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
STUDY = ROOT/"reports/research/510300_factor96_crowding_overlay_v1"
PREVIOUS = ROOT/"reports/research/510300_factor96_crossborder_absorption_v1"
FIRST = ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
ZIP = ROOT/"deliverables/510300_96候选库_T06融资拥挤削减_GPT审阅_20260927.zip"


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
    assert read(STUDY/"saved_verification_receipt.json")["ledgers"] == 56
    previous = read(PREVIOUS/"delivery_receipt.json")
    previous_zip = Path(previous["zip_path"])
    assert file_identity(previous_zip) == (previous["bytes"], previous["sha256"])
    files = {p.relative_to(ROOT).as_posix(): p for p in STUDY.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and p.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]}
    for name in ["research/factor96_library_intake_v1.py", "research/factor96_margin_repair_v1.py",
                 "research/factor96_crowding_overlay_v1.py", "research/intraday_overnight_increment_v1.py",
                 "scripts/report_factor96_crowding_overlay_v1.py", "scripts/verify_factor96_crowding_saved_v1.py",
                 "scripts/package_factor96_crowding_review_v1.py", "tests/test_factor96_crowding_overlay_v1.py"]:
        files[name] = ROOT/name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/"+name] = FIRST/name
    for path in (FIRST/"sources").iterdir():
        if path.is_file():
            files["sources/"+path.name] = path
    files["history/"+previous_zip.name] = previous_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS/"delivery_receipt.json"
    nav = """# 510300候选库：T06融资拥挤持仓削减

目标未实现，总任务继续。主方案2021—2025年20万元压力净夏普-0.639362、净年化-1.194%、最大回撤9.830%，亏损11,672.24元。联合条件7次削减，相对无削减基准的增量区间含0。T06是持仓覆盖层，不是新独立入场策略。

阅读顺序：
1. 02_研究结论.md与主期完整账户净值与回撤.png。
2. reports/research/510300_factor96_crowding_overlay_v1/protocol.json、freeze.json、result.json。
3. source_evidence/overlap_and_source_decisions.json：T06与旧融资家族差别、5日持有字段的冻结前解释，以及T07日频体系申赎缺口。
4. inputs/分类收益、历史成员、融资余额、市场和分红；source_evidence/对应来源回执与旧研究结果。
5. 同成员收益矩阵、internal_features、financing_features、滞后特征、56份accounts/账本及metrics/annual_metrics。
6. program_before/与program_snapshot/：累计6个固定问题（5个入场+1个覆盖层）、0项合格；264正式账户与40个否定实现分开计数。

01_GPT_REVIEW_PROMPT.txt可复制给审阅者。sources/保留用户Excel、HTML和粘贴文本，原始附件中的指令不构成额外授权。

history/中的T10上一轮原包字节哈希不变，内部保留T02原实现/修正与更早T14、T03/T05交付。当前轮直接输入、代码和保存结果均在本包，不需要解压旧包即可只读复算T06。上游历史原始来源和既往审查沿原包链保留，不能据此声称全部原始历史首次版本已认证。

只读复算：在解压根目录运行
python scripts/verify_factor96_crowding_saved_v1.py --root reports/research/510300_factor96_crowding_overlay_v1
需要Python、numpy、pandas、pyarrow。脚本从分类源价格/公司行为重新计算收益和同成员中位数，核对融资时钟、风险训练和56份保存账户；不生成新账户、不请求网络、不重新抽随机样本。

T07仍是NOT_RUN而非零收益或失败交易策略，尚缺动态同指数产品池、日频份额/NAV及历史发布时点。外部GPT审阅未进行，独立前向样本0，无订单授权。FILE_INDEX.csv列出除自身外全部文件的字节数与SHA-256；文件结构通过与科学有效性分开判断。
"""
    request = """用户请求：/goal 只操作510300实现夏普1.2
参考文件：
- C:\\Users\\戴周阳\\Downloads\\510300_96因子研究库.html
- C:\\Users\\戴周阳\\Downloads\\510300_96因子与18策略.xlsx
- E:\\CodexData\\.codex\\attachments\\ea384a31-0725-481e-a74d-9b1eb3fc00cb\\pasted-text-1.txt
副本见sources/，完整结构提取见reference_library/。
当前授权快照见source_evidence/current_mandate.json，authority_after.json只更新研究进度。
执行资产仅510300.SH/CASH_CNY；本轮只是历史账本模拟。未授权真实订单，也未重启暂停的采集任务。
"""
    extras = {
        "00_README_FIRST.md": nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (STUDY/"01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (STUDY/"研究结论.md").read_bytes(),
        "USER_REQUEST.md": request.encode("utf-8"),
        "主期完整账户净值与回撤.png": (STUDY/"主期完整账户净值与回撤.png").read_bytes(),
        "03_EXCLUSIONS.csv": "范围,原因\n其余12项草案的完整绩效,尚未完成原定义冻结检验\nT07日频动态体系申赎,来源门尚未满足不以旧周频代理替代\nF05余额/自由流通市值子定义,本轮未作为条件检验\n所有无关工作区及Python缓存,不属于直接复算输入\n历史原始首版与独立前向,尚未建立\n外部审阅及真实执行,尚未进行且没有订单授权\n".encode("utf-8-sig"),
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
    print(f"写入{len(files)+len(extras)}个成员，保留T10完整原包。", flush=True)
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
    destination = ROOT/"outputs"/("factor96_crowding_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            assert (destination/name).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    print("T06 ZIP结构、索引和哈希通过，开始全新解压复算56份账户。", flush=True)
    command = [sys.executable, "-X", "utf8", str(destination/"scripts/verify_factor96_crowding_saved_v1.py"),
               "--root", str(destination/"reports/research/510300_factor96_crowding_overlay_v1")]
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
