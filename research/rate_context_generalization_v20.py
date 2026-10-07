"""固定全体货币发布月的事前利率来源对照，保留原宏观状态和失败。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_rate_context_generalization_v20"
V19=ROOT/"reports/research/510300_real_yield_company_exposure_v19"
YEARS=list(range(2018,2027))
CUTOFF="2026-09-11T23:59:59+08:00"


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def source_specs():
    rows=[]
    for y in YEARS:
        for kind,stem,data in [("nominal","Treasury","daily_treasury_yield_curve"),("real","Treasury_TIPS","daily_treasury_real_yield_curve")]:
            name=f"{stem}_{y}.xml"
            reused=V19/"sources"/name
            if not reused.exists() and kind=="nominal" and y==2022:
                reused=ROOT/"reports/research/510300_external_discount_clock_v15/sources"/name
            rows.append({"name":name,"year":y,"kind":kind,
                         "reuse":str(reused) if reused.exists() else None,
                         "url":f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data={data}&field_tdr_date_value={y}"})
    return rows


def freeze():
    if OUT.exists():
        raise RuntimeError("第20轮已存在，不覆盖冻结结果。")
    for sub in ["inputs","sources","results","figures","code"]:
        (OUT/sub).mkdir(parents=True)
    protocol={"at":now(),"study_id":"510300_RATE_CONTEXT_GENERALIZATION_V20",
      "previous_turn":"PROGRESS：V19完成三窗口利率来源与六家公司传导；全窗口变化尚不能证明起点可预测。",
      "question":"V19的实际利率来源差异能否在原全部月份成立？起点已知的利率结构与持有期才发生的结构，分别和收益有何关系？原信贷订单、价格位置、估值及下行波动背景是否可比？",
      "scope":"原104个月、84旧口径与20新口径分列；103个完整E0和原E1状态不变。全部原ETF结果已知，新增公开历史利率仅用于机制外推诊断，不能称独立验证。",
      "cutoff":CUTOFF,"primary_features":"原21点判断时已知的同日10年名义CMT、TIPS实际CMT及两者差额；各自相对前20条Treasury记录变化。连续记录跨年，缺失不前填。",
      "clocks":"美国数据日美东23:59:59假定可见；另整个来源延迟一条记录。不是历史首版认证。原判断时点晚于行情截止的2026年8月行保留NO_VIEW。",
      "source_budget":"2018至2026九个年份，两种Treasury原XML；已有5份复用，其余每份请求一次，不追加新模型、数据商或替代期限。",
      "primary_comparison":"旧M1全部可用月份中，事前名义10年近20记录上升时，实际TIPS上升组减未上升组的原E0后20日平均收益差。正负边界固定为0；原报告精度下恰为0单独标注，未上升包含0。不挑阈值。",
      "complete_groups":"保存名义升/未升与TIPS升/未升四种组合及各自均值、中位数、正收益、最差路径、逐年覆盖；不能从中另选最好组。",
      "context":"沿用原信贷订单联合状态，保存所有状态，不重定义、不投票。另对原11个共同支持月完整列出同样四种组合，仅作小样本诊断。各组同时列出剪刀差水平及变化、当期/基数项、信贷分项、订单、资金价格、此前20/60日涨幅、原估值、内部参与和下行波动，检查背景是否相近。",
      "delay_checks":"分组始终用原21点信息；E1收益不使用E1新增信息重新筛选。来源滞后一记录再完整复算。另保存E0/E1开盘可见利率，作为信息变化说明，不选最好开盘。",
      "after_origin":"原起点到原E0/E1终点，按各时点可见利率计算同日来源变化；只作为事后解释。保存事前与事后方向转移表及全部跨年失败，严禁回填起点条件。",
      "continuous_diagnostic":"分旧新制度分别列事前与事后名义/TIPS/差额变化和E0/E1收益、最差路径的Spearman相关；保留全部相关，不筛选，不拟合或预测。",
      "uncertainty":"唯一主比较及其E1/来源延迟对照采用固定6个统计月连续循环区块、2000次重抽、随机种子20260930的95%描述区间；两组任一少于8个月或全体不到4个自然年不计算区间。年度缺失保留在日历位置。未经多重检验校正，不用于策略晋级。",
      "original_horizons":"主20个A股交易日；原E0/E1均保留，分红留现金、毛收益，非账户；不改止盈止损或持有期限。",
      "limits":"已看过收益的历史解释，无随机冲击识别、无独立验证。国际利率与国内盈利存在共同驱动，相关不能直接当因果。V14旧模型及冻结失败保持原样；本轮不重估或修救。",
      "new_models":0,"new_accounts":0,"orders_authorized":False,"goal_achieved":False,"global_mandate_modified":False}
    save("protocol.json",protocol)
    shutil.copy2(OUT/"protocol.json",ROOT/"config/510300_rate_context_generalization_v20.json")
    mapping={
      "monthly.csv":ROOT/"reports/research/510300_balance_source_comparability_v14/results/104个月_完整输入与原结果.csv",
      "expectations.csv":ROOT/"reports/research/510300_information_change_transmission_v6/results/104个月_原因盈利定价与预期差完整连接.csv",
      "market.csv":V19/"inputs/market.csv",
      "prior_v14_result.json":ROOT/"reports/research/510300_balance_source_comparability_v14/result.json",
      "prior_v19_completion.json":V19/"completion.json"}
    inputs=[]
    for name,p in mapping.items():
        shutil.copy2(p,OUT/"inputs"/name)
        inputs.append({"name":name,"source":str(p),"sha256":digest(p)})
    specs=source_specs()
    save("freeze.json",{"at":now(),"protocol_sha256":digest(OUT/"protocol.json"),"inputs":inputs,"sources":specs})
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    print(json.dumps({"状态":"已固定全体104月与唯一主比较","新来源数":sum(r["reuse"] is None for r in specs),"复用数":sum(r["reuse"] is not None for r in specs)},ensure_ascii=False))


def fetch(spec):
    p=OUT/"sources"/spec["name"]
    receipt={**spec,"retrieved_at":now(),"historical_first_vintage_verified":False}
    try:
        if spec["reuse"]:
            shutil.copy2(spec["reuse"],p)
            receipt["status"]="REUSED_EXISTING_RAW"
        else:
            response=requests.get(spec["url"],timeout=(12,45),headers={"User-Agent":"Mozilla/5.0"})
            receipt.update(http_status=response.status_code,final_url=response.url)
            response.raise_for_status()
            data=response.content
            root=ET.fromstring(data)
            entries=root.findall("{http://www.w3.org/2005/Atom}entry")
            assert len(entries)>=150,(spec["name"],len(entries))
            field="BC_10YEAR" if spec["kind"]=="nominal" else "TC_10YEAR"
            assert all(e.find(".//{http://schemas.microsoft.com/ado/2007/08/dataservices}"+field) is not None for e in entries)
            p.write_bytes(data)
            receipt["status"]="SAVED_RAW_PENDING_REVIEW"
        receipt.update(bytes=p.stat().st_size,sha256=digest(p))
    except Exception as exc:
        receipt.update(status="SOURCE_FAILED_NO_RETRY",error=str(exc)[:260])
    save("sources/"+spec["name"]+".receipt.json",receipt)
    return receipt


def collect():
    if (OUT/"source_receipts.json").exists():
        raise RuntimeError("固定来源请求已执行，不重试。")
    frozen=json.loads((OUT/"freeze.json").read_text("utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(fetch,frozen["sources"]))
    save("source_receipts.json",receipts)
    print(json.dumps([{k:r.get(k) for k in ["name","status","bytes","error"]} for r in receipts],ensure_ascii=False))


if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ["freeze","collect"]:
        raise SystemExit("参数：freeze 或 collect")
    freeze() if sys.argv[1]=="freeze" else collect()
