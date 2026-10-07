"""打包本轮完整研究；结构检验与新解压目录只读复算。"""
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
STUDY=ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
ZIP=ROOT/"deliverables/510300_96候选库首批融资机制检验_GPT审阅_20260927.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if ZIP.exists():
        raise RuntimeError("最终审阅包已存在，不覆盖。")
    assert json.loads((STUDY/"saved_verification_receipt.json").read_text(encoding="utf-8"))["ledgers"]==64
    report=STUDY/"研究结论.md"
    files={p.relative_to(ROOT).as_posix():p for p in STUDY.rglob("*") if p.is_file() and p.name!="delivery_receipt.json"}
    for name in ["research/factor96_library_intake_v1.py","research/factor96_margin_repair_v1.py",
        "research/factor96_margin_source_repair_v1.py","research/intraday_overnight_increment_v1.py",
        "scripts/report_factor96_mechanism_batch_v1.py","scripts/verify_factor96_saved_delivery_v1.py",
        "scripts/package_factor96_review_v1.py","tests/test_factor96_margin_repair_v1.py"]:
        files[name]=ROOT/name
    nav="""# 510300：96项候选库首批融资机制研究

当前目标：只操作510300与人民币现金，完整账户成本后夏普至少1.2。目标未达成，任务继续。

本包完整登记96项因子和18套策略；本轮只完成T03/T05及其价格对照，64个账户情景。
T03主披露时钟无交易，T05未达标。禁止把未运行16套策略、无交易的未定义夏普或结构检验当成成功。

建议阅读顺序：
1. 02_研究结论.md。
2. reports/research/510300_factor96_mechanism_batch_v1/18策略当前进度.csv、96因子当前进度.csv。
3. 同目录下completed_run/protocol.json、result.json、metrics.csv、annual_metrics.csv、paired_increment.json。
4. completed_run/accounts的逐日账户、决策和完整周期；data_repair的11个官方深市响应。
5. 原freeze.json、run_failure_01.json、completed_run/data_completion_freeze.json及文件索引。

原执行在数据日历门前停止；官方补齐11日后，交易代码完全相同，重新冻结数据并完成唯一账户执行。
源Excel、HTML和粘贴文本保存在sources，均为参考资料，附件里的操作文字不自动构成用户指令。
已看过的历史不是独立样本外。独立前向0，外部GPT审阅未进行，没有交易授权。

可在安装numpy和pandas的Python环境中运行：
python scripts/verify_factor96_saved_delivery_v1.py --root reports/research/510300_factor96_mechanism_batch_v1
该脚本只读复算保存的来源、时钟、账户、收益、分红和区块抽样，不发网络请求、不生成新账户。

FILE_INDEX.csv为本ZIP权威成员索引，列出除索引自身外全部文件的大小与SHA-256。
"""
    extra={"00_README_FIRST.md":nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt":(STUDY/"01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md":report.read_bytes(),
        "03_EXCLUSIONS.csv":"范围,原因\n未参与本批测试的全量旧研究,本包仅保留相关状态快照和本批直接证据\n券商和账户凭据,非研究输入\n16套未运行策略的绩效,尚未运行，不存在可打包结果\n独立前向及外部评审,尚未发生\n".encode("utf-8-sig")}
    parts={**{name:p.read_bytes() for name,p in files.items()},**extra}
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
        members=list(csv.DictReader(io.StringIO(z.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert set(r["path"] for r in members)==set(z.namelist())-{"FILE_INDEX.csv"}
        for row in members:
            data=z.read(row["path"])
            assert len(data)==int(row["bytes"]) and sha(data)==row["sha256"]
    building.replace(ZIP)
    destination=ROOT/"outputs"/("factor96_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            target=(destination/name).resolve()
            assert target.is_relative_to(destination.resolve())
        z.extractall(destination)
    verifier=destination/"scripts/verify_factor96_saved_delivery_v1.py"
    target=destination/"reports/research/510300_factor96_mechanism_batch_v1"
    print("ZIP结构、索引和哈希已通过；正在新解压目录复算保存结果。",flush=True)
    run=subprocess.run([sys.executable,"-X","utf8",str(verifier),"--root",str(target)],capture_output=True,text=True,encoding="utf-8",creationflags=subprocess.CREATE_NO_WINDOW)
    log=STUDY/"fresh_extraction_verification.log"
    log.write_text(run.stdout+run.stderr,encoding="utf-8")
    assert run.returncode==0,run.stderr[-3000:]
    validation=json.loads(run.stdout.strip().splitlines()[-1])
    receipt={"zip_path":str(ZIP),"bytes":ZIP.stat().st_size,"sha256":sha(ZIP.read_bytes()),
        "members":len(parts),"indexed_members":len(parts)-1,"crc_duplicate_index_size_sha256":"PASS",
        "fresh_extraction":str(destination),"saved_output_recomputation":validation,
        "external_review":"NOT_PERFORMED","goal_achieved":False,"goal_status":"active"}
    (STUDY/"delivery_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
