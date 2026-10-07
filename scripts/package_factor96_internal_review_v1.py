"""T02原始缺陷与修正结果同包保存，新解压目录复核正式40账户。"""
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
STUDY=ROOT/"reports/research/510300_factor96_internal_reclaim_v1_0_1"
ORIGINAL=ROOT/"reports/research/510300_factor96_internal_reclaim_v1"
PREVIOUS=ROOT/"reports/research/510300_factor96_funding_relief_v1"
FIRST=ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
ZIP=ROOT/"deliverables/510300_96候选库_T02内部背离_GPT审阅_20260927.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert not ZIP.exists(),"最终包已存在，不覆盖"
    assert read(STUDY/"saved_verification_receipt.json")["ledgers"]==40
    previous=read(PREVIOUS/"delivery_receipt.json")
    prior_zip=Path(previous["zip_path"])
    assert sha(prior_zip.read_bytes())==previous["sha256"]
    files={}
    for folder in [STUDY,ORIGINAL]:
        files.update({p.relative_to(ROOT).as_posix():p for p in folder.rglob("*")
            if p.is_file() and p.name not in ["delivery_receipt.json","fresh_extraction_verification.log"]})
    code=["research/factor96_library_intake_v1.py","research/factor96_margin_repair_v1.py",
        "research/factor96_internal_reclaim_v1.py","research/factor96_internal_reclaim_v1_0_1.py",
        "research/intraday_overnight_increment_v1.py","scripts/report_factor96_internal_reclaim_v1.py",
        "scripts/verify_factor96_internal_saved_v1.py","scripts/package_factor96_internal_review_v1.py",
        "tests/test_factor96_internal_reclaim_v1.py","tests/test_factor96_internal_reclaim_v1_0_1.py",
        "tests/test_factor96_margin_repair_v1.py"]
    for name in code:
        files[name]=ROOT/name
    for name in ["factor_registry.json","strategy_registry.json","input_workbook_tables.json","intake_receipt.json"]:
        files["reference_library/"+name]=FIRST/name
    for p in (FIRST/"sources").iterdir():
        if p.is_file():
            files["sources/"+p.name]=p
    files["history/"+prior_zip.name]=prior_zip
    files["history/prior_delivery_receipt.json"]=PREVIOUS/"delivery_receipt.json"
    nav="""# 510300候选库：T02二次试低与内部背离

目标仍未实现，总任务继续。正式V1.0.1主方案2021—2025年20万元压力净夏普-0.616675，2万元-0.704129；各11个完整周期。晚一日敏感性夏普0.540980，仍未达标且不晋升为主方案。

必须区分：
- reports/research/510300_factor96_internal_reclaim_v1_0_1/：正式修正40个账户、报告、事件与来源。
- reports/research/510300_factor96_internal_reclaim_v1/：原40个实现缺陷账户，官方停牌记录错误排除，结果已否定。先读verification_failure_01.json，不能拿原值作正式结论。

阅读顺序：02_研究结论.md → 正式目录protocol.json/freeze.json/result.json → implementation_diff.patch与原失败 → metrics/annual_metrics/events/accounts → 原始价、成员和公司行为证据 → program_snapshot累计进度。

本包保存本轮全部80个实际执行账户，包括40个否定实现结果。候选库累计4个已完成固定候选，0项合格；160个当前正式情景加40个否定实现情景，共200个实际执行账户。账户情景数量不能当独立策略数量。

history/保留上一轮T14完整原包，其内部history/又保留首批T03/T05原包。原包字节与先前交付一致。sources/为用户原Excel、HTML和粘贴文本，只作为参考材料。全部直接输入、两版代码、冻结文件和65份补充官方成员文件可在本包查看。

只读复核正式结果：从解压根目录运行
python scripts/verify_factor96_internal_saved_v1.py --root reports/research/510300_factor96_internal_reclaim_v1_0_1
需要Python以及numpy、pandas、pyarrow。脚本从原始价/公司行为按每个21日窗口重算，不运行策略、不新增抽样或请求网络。

外部GPT审阅未进行，独立前向样本0，无交易授权。FILE_INDEX.csv列出除自身外的所有成员大小与SHA-256。结构与算术验证不是经济有效性证明。
"""
    extra={"00_README_FIRST.md":nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt":(STUDY/"01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md":(STUDY/"研究结论.md").read_bytes(),
        "主期完整账户净值与回撤.png":(STUDY/"主期完整账户净值与回撤.png").read_bytes(),
        "03_EXCLUSIONS.csv":"范围,原因\n非本轮依赖的其他研究原始全库,本包保留直接来源和以前两轮原包\n14项未完成候选的绩效,不存在可据以宣称通过的结果\n券商或API凭据,非研究输入\n独立前向或外部审阅,尚未发生\n".encode("utf-8-sig")}
    parts={**{name:p.read_bytes() for name,p in files.items()},**extra}
    index=io.StringIO(newline="")
    writer=csv.DictWriter(index,fieldnames=["path","bytes","sha256"])
    writer.writeheader()
    for name,data in sorted(parts.items()):
        writer.writerow({"path":name,"bytes":len(data),"sha256":sha(data)})
    parts["FILE_INDEX.csv"]=index.getvalue().encode("utf-8-sig")
    building=ZIP.with_suffix(".building.zip")
    print(f"正在保存{len(parts)}个成员，原实现与修正结果同时保留。",flush=True)
    with zipfile.ZipFile(building,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(parts.items()):
            z.writestr(name,data)
    with zipfile.ZipFile(building) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(parts)
        items=list(csv.DictReader(io.StringIO(z.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in items}==set(z.namelist())-{"FILE_INDEX.csv"}
        for row in items:
            data=z.read(row["path"])
            assert len(data)==int(row["bytes"]) and sha(data)==row["sha256"]
    building.replace(ZIP)
    target=ROOT/"outputs"/("factor96_internal_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    target.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            assert (target/name).resolve().is_relative_to(target.resolve())
        z.extractall(target)
    print("T02 ZIP结构与成员索引通过，正在全新目录复核正式保存结果。",flush=True)
    verifier=target/"scripts/verify_factor96_internal_saved_v1.py"
    study=target/"reports/research/510300_factor96_internal_reclaim_v1_0_1"
    run=subprocess.run([sys.executable,"-X","utf8",str(verifier),"--root",str(study)],capture_output=True,
        text=True,encoding="utf-8",creationflags=subprocess.CREATE_NO_WINDOW)
    (STUDY/"fresh_extraction_verification.log").write_text(run.stdout+run.stderr,encoding="utf-8")
    assert run.returncode==0,run.stderr[-3000:]
    verified=json.loads(run.stdout.strip().splitlines()[-1])
    receipt={"at":datetime.now().astimezone().isoformat(),"zip_path":str(ZIP),"bytes":ZIP.stat().st_size,
        "sha256":sha(ZIP.read_bytes()),"members":len(parts),"indexed_members":len(parts)-1,
        "crc_duplicate_index_size_sha256":"PASS","fresh_extraction":str(target),"saved_output_recomputation":verified,
        "invalid_implementation_accounts_preserved":40,"prior_zip_sha256":previous["sha256"],
        "external_review":"NOT_PERFORMED","goal_achieved":False,"goal_status":"active"}
    (STUDY/"delivery_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
