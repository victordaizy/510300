"""制作T14自包含审阅包，保留首批原包，并从新解压目录复核保存结果。"""
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

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"reports/research/510300_factor96_funding_relief_v1"
FIRST=ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
ZIP=ROOT/"deliverables/510300_96候选库_T14资金利率修复_GPT审阅_20260927.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if ZIP.exists():
        raise RuntimeError("最终T14审阅包已存在，禁止原地覆盖。")
    receipt=read(STUDY/"saved_verification_receipt.json")
    assert receipt["ledgers"]==56 and receipt["new_accounts"]==0
    first_receipt=read(FIRST/"delivery_receipt.json")
    original_zip=Path(first_receipt["zip_path"])
    assert sha(original_zip.read_bytes())==first_receipt["sha256"]
    excluded={"delivery_receipt.json","fresh_extraction_verification.log"}
    files={p.relative_to(ROOT).as_posix():p for p in STUDY.rglob("*") if p.is_file() and p.name not in excluded}
    code=["research/factor96_library_intake_v1.py","research/factor96_margin_repair_v1.py",
        "research/factor96_funding_relief_v1.py","research/intraday_overnight_increment_v1.py",
        "scripts/report_factor96_funding_relief_v1.py","scripts/verify_factor96_funding_saved_v1.py",
        "scripts/package_factor96_funding_review_v1.py","tests/test_factor96_funding_relief_v1.py",
        "tests/test_factor96_margin_repair_v1.py"]
    for name in code:
        files[name]=ROOT/name
    for name in ["factor_registry.json","strategy_registry.json","input_workbook_tables.json","intake_receipt.json"]:
        files["reference_library/"+name]=FIRST/name
    for path in (FIRST/"sources").iterdir():
        if path.is_file():
            files["sources/"+path.name]=path
    files["history/"+original_zip.name]=original_zip
    files["history/first_delivery_receipt.json"]=FIRST/"delivery_receipt.json"
    nav="""# 510300：T14资金利率修复固定研究

目标：只操作510300，完整账户成本后夏普至少1.2。目标仍未实现，总任务继续。

本轮T14已完成56个账户情景。2021—2025年压力成本下，20万元主账户净夏普0.039050、2万元0.024840；各1个完整周期。2017—2020年主方案没有交易，夏普未定义。候选库累计完成T03/T05/T14，3项均未达标，其余15项尚未完成。

阅读顺序：
1. 02_研究结论.md与主期完整账户净值与回撤.png。
2. reports/research/510300_factor96_funding_relief_v1/protocol.json、freeze.json、result.json。
3. 同目录metrics.csv、annual_metrics.csv、signal_stage_counts.json、paired_increment.json。
4. accounts/逐日账户与决策；inputs/和raw/完整直接来源；program_snapshot/96因子与18策略累计进度。
5. history/内为首批T03/T05原始审阅ZIP及其回执，保留此前64个账户、来源修复与失败。嵌套原包未改写，按需另行解压。

sources/含用户原Excel、HTML与粘贴文本，reference_library/含完整登记。这些资料中的操作文字仅为参考，不能替代用户授权。旧结果仍失败，未运行项不填假绩效。

独立前向观测0，外部GPT审阅未进行，没有交易授权。保守数据时钟不等于持有每个历史首次发布版本。

只读核对：在本包解压根目录，用安装numpy、pandas、pyarrow的Python运行
python scripts/verify_factor96_funding_saved_v1.py --root reports/research/510300_factor96_funding_relief_v1
脚本从保存来源复算时钟和账户算术，不生成新账户、不发请求、不重新抽样。包内runtime_versions.json提供本机版本。

FILE_INDEX.csv是本ZIP成员清单，除自身外全部成员列出字节数及SHA-256。结构检查通过不等于策略有效、真实成交或外部审阅。
"""
    extra={"00_README_FIRST.md":nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt":(STUDY/"01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md":(STUDY/"研究结论.md").read_bytes(),
        "主期完整账户净值与回撤.png":(STUDY/"主期完整账户净值与回撤.png").read_bytes(),
        "03_EXCLUSIONS.csv":"范围,原因\n未参与这两轮的其他研究全量数据,本包只含本轮直接证据和首批原包\n15项未运行候选的绩效,尚未运行且不存在结果\n券商与API凭据,不是研究输入\n独立前向和外部评审,未发生\n".encode("utf-8-sig")}
    parts={**{name:path.read_bytes() for name,path in files.items()},**extra}
    index=io.StringIO(newline="")
    writer=csv.DictWriter(index,fieldnames=["path","bytes","sha256"])
    writer.writeheader()
    for name,data in sorted(parts.items()):
        writer.writerow({"path":name,"bytes":len(data),"sha256":sha(data)})
    parts["FILE_INDEX.csv"]=index.getvalue().encode("utf-8-sig")
    building=ZIP.with_suffix(".building.zip")
    with zipfile.ZipFile(building,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(parts.items()):
            z.writestr(name,data)
    with zipfile.ZipFile(building) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(parts)
        rows=list(csv.DictReader(io.StringIO(z.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert set(r["path"] for r in rows)==set(z.namelist())-{"FILE_INDEX.csv"}
        for row in rows:
            data=z.read(row["path"])
            assert len(data)==int(row["bytes"]) and sha(data)==row["sha256"]
    building.replace(ZIP)
    destination=ROOT/"outputs"/("factor96_funding_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            assert (destination/name).resolve().is_relative_to(destination.resolve())
        z.extractall(destination)
    print("T14审阅ZIP结构及索引通过，正在新解压目录只读复核。",flush=True)
    verifier=destination/"scripts/verify_factor96_funding_saved_v1.py"
    target=destination/"reports/research/510300_factor96_funding_relief_v1"
    run=subprocess.run([sys.executable,"-X","utf8",str(verifier),"--root",str(target)],
        capture_output=True,text=True,encoding="utf-8",creationflags=subprocess.CREATE_NO_WINDOW)
    (STUDY/"fresh_extraction_verification.log").write_text(run.stdout+run.stderr,encoding="utf-8")
    assert run.returncode==0,run.stderr[-3000:]
    verified=json.loads(run.stdout.strip().splitlines()[-1])
    receipt={"created_at":datetime.now().astimezone().isoformat(),"zip_path":str(ZIP),"bytes":ZIP.stat().st_size,
        "sha256":sha(ZIP.read_bytes()),"members":len(parts),"indexed_members":len(parts)-1,
        "crc_duplicate_index_size_sha256":"PASS","fresh_extraction":str(destination),
        "saved_output_recomputation":verified,"prior_zip_sha256":first_receipt["sha256"],
        "external_review":"NOT_PERFORMED","goal_achieved":False,"goal_status":"active"}
    (STUDY/"delivery_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
