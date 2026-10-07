"""固定原三窗口的实际利率分解与六家公司经营暴露。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "reports/research/510300_real_yield_company_exposure_v19"
V18=ROOT / "reports/research/510300_expectation_rates_internal_v18"
V13=ROOT / "reports/research/510300_bank_profit_credit_bridge_v13"
CASES=["2020-11","2020-12","2021-01"]
COMPANIES=["600519.SH","601318.SH","000858.SZ","600036.SH","000333.SZ","600276.SH"]
DOCS=[
 {"name":f"Treasury_TIPS_{y}.xml","kind":"TIPS","url":f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value={y}"} for y in [2020,2021]
]+[
 {"name":"DKW_updates.csv","kind":"MODEL","url":"https://www.federalreserve.gov/econres/notes/feds-notes/DKW-updates.csv"},
 {"name":"DKW方法与发布说明.html","kind":"METHOD","url":"https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html"},
 {"name":"600519_2020Q3.pdf","kind":"COMPANY","symbol":"600519.SH","published_date":"2020-10-26","url":"https://static.cninfo.com.cn/finalpage/2020-10-26/1208611044.PDF"},
 {"name":"601318_2020Q3.pdf","kind":"COMPANY","symbol":"601318.SH","published_date":"2020-10-28","url":"https://static.cninfo.com.cn/finalpage/2020-10-28/1208625889.PDF"},
 {"name":"000858_2020Q3.pdf","kind":"COMPANY","symbol":"000858.SZ","published_date":"2020-10-30","url":"https://static.cninfo.com.cn/finalpage/2020-10-30/1208647843.PDF"},
 {"name":"000333_2020Q3.pdf","kind":"COMPANY","symbol":"000333.SZ","published_date":"2020-10-31","url":"https://static.cninfo.com.cn/finalpage/2020-10-31/1208664350.PDF"},
 {"name":"600276_2020Q3.pdf","kind":"COMPANY","symbol":"600276.SH","published_date":"2020-10-20","url":"https://static.cninfo.com.cn/finalpage/2020-10-20/1208584798.PDF"},
 {"name":"601318_2020全年报告.pdf","kind":"COMPANY","symbol":"601318.SH","document_date":"2021-02-03","published_date":None,"url":"https://www.pingan.com/app_upload/file/official/2020annualreport.pdf","clock_note":"官网原件，报告日期不当作首发证明；全年通稿另按2021年2月3日末可见，完整报告待原发布目录核对。"},
 {"name":"601318_20210203全年通稿.html","kind":"COMPANY_NEWS","symbol":"601318.SH","published_date":"2021-02-03","url":"https://www.pingan.cn/zh/common/cn_news/1612348496784.shtml"},
]


def now():
    return datetime.now().astimezone().isoformat()


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def freeze():
    if OUT.exists():
        raise RuntimeError("本轮已经存在，不覆盖。")
    for d in ["inputs","sources","results","figures","code"]:
        (OUT/d).mkdir(parents=True)
    p={"at":now(),"study_id":"510300_REAL_YIELD_COMPANY_EXPOSURE_V19","previous_turn":"PROGRESS：V18已完成预期、69个利率时点和原权重贡献。",
       "cases":CASES,"question":"原三窗口的名义长端利率变化来自TIPS收益率还是通胀补偿？当时原公司资料怎样约束借款成本、投资收益、负债和股价折现的解释？",
       "selection":"固定V18全部三窗口和原E0/E1时钟，ETF及公司收益已知；不另按收益挑时段。",
       "companies":COMPANIES,"company_selection":"V18各期原权重前五恰为同五家公司，再取各期医药生物最大原权重恒瑞；六家均保留，不按涨跌决定取舍。",
       "company_documents":"六家2020三季报作为三个起点均已可见背景；招商全年快报与平安全年通稿按自身发布时间另列。平安完整年报的金融敏感性须核对发布目录，目录未确认时只作回顾结构材料。不是全部公告目录或最新全部盈利预期。",
       "raw_decomposition":"同日、同10年期限的Treasury名义CMT减TIPS实际CMT；差额称通胀补偿代理，不等于纯预期通胀。保留5年、7年、20年、30年原始值供追溯，但不从中选最佳期限。",
       "clocks":"沿V18美国记录日美东23:59:59假定可见和额外滞后一条记录；2020/2021连续记录跨年差分。名义和实际须同日匹配，不前向填充缺失。",
       "source_budget":"两个固定年份TIPS XML，1份DKW当前CSV及方法页，5份新三季报、平安1份全年报告和1份同日公司通稿；复用招商2份和恒瑞2020H1。发布目录最多各1次原始查询；失败保留。",
       "model_role":"DKW当前版本仅为回顾机制诊断；官方模型是月度更新且允许修订，当前拟合值不是2021年当时可用输入。仅按原端点分解10年实际预期短率、实际期限溢价、通胀预期、通胀风险溢价和TIPS流动性项；缺列保留未知，不估新模型。",
       "causal_limits":"债券收益率拆解不是股票收益因果分摊；不把美国TIPS变化直接代入人民币公司债务成本或公司披露的统一平移情景。公司利率敏感性有各自资产、负债、币种和会计口径。",
       "windows":"保存69个原时点和60个交易日全部路径；按原三窗完整比较，未新增收益组、预测门槛或账户。",
       "new_models":0,"new_accounts":0,"orders_authorized":False,"goal_achieved":False,"independent_validation":False,"global_mandate_modified":False}
    save("protocol.json",p)
    shutil.copy2(OUT/"protocol.json",ROOT/"config/510300_real_yield_company_exposure_v19.json")
    specs={
     "inputs/monthly.csv":V18/"results/三个连续月份_经营预期利率与内部传导.csv",
     "inputs/snapshots.csv":V18/"results/69个观察时点_当时可见利率及延后记录.csv",
     "inputs/daily_paths.csv":V18/"results/原60个交易日_ETF波动与利率时序.csv",
     "inputs/stocks.csv":V18/"results/900个原成员_收益贡献与缺失.csv",
     "inputs/market.csv":V18/"inputs/market.csv",
     "sources/Treasury_2020.xml":V18/"sources/Treasury_2020.xml",
     "sources/Treasury_2021.xml":V18/"sources/Treasury_2021.xml",
     "sources/600036_2020Q3.pdf":V13/"sources/招商银行_2020三季报.pdf",
     "sources/600036_2020Q3.pdf.receipt.json":V13/"sources/招商银行_2020三季报.pdf.receipt.json",
     "sources/600036_2020快报.pdf":V13/"sources/招商银行_2020业绩快报.pdf",
     "sources/600036_2020快报.pdf.receipt.json":V13/"sources/招商银行_2020业绩快报.pdf.receipt.json",
     "sources/600276_2020H1.pdf":ROOT/"reports/research/510300_constituent_external_bridge_v16/sources/600276_2020H1.pdf",
    }
    receipts=[]
    for name,src in specs.items():
        shutil.copy2(src,OUT/name)
        receipts.append({"name":name,"source":str(src.relative_to(ROOT)),"sha256":digest(src)})
    save("freeze.json",{"at":now(),"protocol_sha256":digest(OUT/"protocol.json"),"inputs":receipts,"documents":DOCS})
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    print(json.dumps({"已固定":p["study_id"],"公司数":len(COMPANIES),"新来源上限":len(DOCS)},ensure_ascii=False))


def fetch(d):
    dst=OUT/"sources"/d["name"]
    r={**d,"retrieved_at":now(),"historical_first_vintage_verified":False}
    try:
        response=requests.get(d["url"],timeout=(12,45),headers={"User-Agent":"Mozilla/5.0"})
        r.update(http_status=response.status_code,final_url=response.url)
        response.raise_for_status()
        data=response.content
        if dst.suffix==".pdf":
            assert data.startswith(b"%PDF")
        elif dst.suffix==".xml":
            assert len(ET.fromstring(data).findall("{http://www.w3.org/2005/Atom}entry"))>200
        elif dst.suffix==".csv":
            assert len(data)>10000 and b"<html" not in data[:1000].lower()
        else:
            txt=BeautifulSoup(data,"html.parser").get_text("\n",strip=True)
            assert len(txt)>1000
            dst.with_suffix(".txt").write_text(txt,encoding="utf-8")
        dst.write_bytes(data)
        r.update(status="SAVED_PENDING_NUMERIC_REVIEW",bytes=len(data),sha256=digest(dst))
    except Exception as exc:
        r.update(status="FAILED_SOURCE_REQUEST",error=str(exc)[:250])
    save("sources/"+dst.name+".receipt.json",r)
    return r


def collect():
    if (OUT/"source_receipts.json").exists():
        raise RuntimeError("来源请求已执行。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,DOCS))
    save("source_receipts.json",results)
    print(json.dumps([{k:r.get(k) for k in ["name","status","bytes","error"]} for r in results],ensure_ascii=False))


if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ["freeze","collect"]:
        raise SystemExit("参数应为freeze或collect。")
    freeze() if sys.argv[1]=="freeze" else collect()
