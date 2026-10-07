"""两轮研究打包及解压后保存结果复算，不重训或生成新账户。"""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_probability_payoff_progress_20260924"
TARGET=ROOT/"deliverables/510300_概率盈亏训练与逐步扩点_两轮结果_20260924.zip"
STAGES=[("01_日样本训练与扩点",ROOT/"reports/research/510300_dense_probability_payoff_nodes_v1","dense_probability_payoff_nodes_v1.py"),
        ("02_固定节点分布校准",ROOT/"reports/research/510300_node_distribution_calibration_v1","node_distribution_calibration_v1.py")]


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    building=TARGET.with_suffix(".building.zip")
    extracted=OUT/"verification/fresh_extract"
    require(not TARGET.exists() and not building.exists() and not extracted.exists(),"已有交付或临时解压，不能覆盖。")
    state=json.loads((OUT/"continuation_status.json").read_text(encoding="utf-8"))
    require(state["new_accounts"]==32 and state["new_saved_fitted_heads_total"]==1676 and not state["goal_achieved"],"两轮实际训练范围或目标状态不同。")
    runtime={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
             "pyarrow":pyarrow.__version__,"PyYAML":yaml.__version__,"matplotlib":matplotlib.__version__}
    save(OUT/"runtime.json",runtime)
    (OUT/"requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in runtime.items() if k!="python")+"\ntzdata\n",encoding="utf-8")
    members={}
    for prefix,folder in [("",OUT),*[(name,path) for name,path,_ in STAGES]]:
        for path in folder.rglob("*"):
            relative=path.relative_to(folder)
            if not path.is_file() or "verification" in relative.parts or "__pycache__" in relative.parts or path.name=="delivery_receipt.json":
                continue
            name=(prefix+"/" if prefix else "")+relative.as_posix()
            require(name not in members,"打包成员重名。")
            members[name]=path.read_bytes()
    for name in [Path(__file__).name,"finish_probability_payoff_progress_20260924.py"]:
        members["code/"+name]=(ROOT/"scripts"/name).read_bytes()
    index=io.StringIO(newline="")
    writer=csv.DictWriter(index,["path","bytes","sha256"])
    writer.writeheader()
    for name,value in sorted(members.items()):
        writer.writerow({"path":name,"bytes":len(value),"sha256":sha(value)})
    members["FILE_INDEX.csv"]=index.getvalue().encode("utf-8-sig")
    with zipfile.ZipFile(building,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,value in sorted(members.items()):
            archive.writestr(name,value)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(building) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)) and archive.testzip() is None,"ZIP重复成员或CRC错误。")
        indexed=list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names)=={r["path"] for r in indexed}|{"FILE_INDEX.csv"},"索引成员覆盖不同。")
        for row in indexed:
            value=archive.read(row["path"])
            require(len(value)==int(row["bytes"]) and sha(value)==row["sha256"],"索引大小或哈希错误。")
        for name in names:
            destination=(extracted/name).resolve()
            require(destination.is_relative_to(extracted.resolve()),"解压路径越界。")
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(name))
    verifications=[]
    for prefix,_,script in STAGES:
        folder=extracted/prefix
        before={p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in folder.rglob("*") if p.is_file()}
        result=subprocess.run([sys.executable,"-X","utf8",str(folder/"code"/script),"verify","--root",str(folder)],
            check=True,capture_output=True,text=True,encoding="utf-8",timeout=300)
        receipt=json.loads(result.stdout)
        require(receipt["status"].startswith("PASS_SAVED_") and receipt["new_fits"]==receipt["new_accounts"]==receipt["new_downloads"]==0,"保存结果复算失败或产生新研究。")
        after={p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        require(before==after,"复算修改了保存数据或结果。")
        receipt["stage"]=prefix
        verifications.append(receipt)
        print(f"{prefix}：解压后保存参数、预测、现金流和年度次数复算通过。",flush=True)
    save(OUT/"verification/fresh_saved_recomputation.json",verifications)
    building.replace(TARGET)
    receipt={"completed_at":datetime.now().astimezone().isoformat(),"archive":str(TARGET),"bytes":TARGET.stat().st_size,
        "sha256":sha(TARGET.read_bytes()),"members":len(members),"indexed_files":len(members)-1,
        "structural_checks":"PASS_CRC_DUPLICATES_INDEX_SIZE_HASH","fresh_extraction_saved_recomputation":verifications,
        "recomputation_did_not_change_packaged_data":True,"new_saved_fitted_heads":1676,"new_accounts_in_research":32,
        "reused_parent_accounts":4,"new_downloads":0,"external_review_completed":False,"independent_validation_established":False,
        "goal_achieved":False,"background_training_running":False,"orders_authorized":False}
    save(OUT/"delivery_receipt.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
