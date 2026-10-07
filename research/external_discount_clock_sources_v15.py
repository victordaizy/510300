"""固定两个既有病例，保存外部利率及官方公告的时间证据。"""
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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_external_discount_clock_v15"
INPUTS = {
    "monthly.csv": "reports/research/510300_balance_source_comparability_v14/results/104个月_完整输入与原结果.csv",
    "daily_paths.csv": "reports/research/510300_information_change_transmission_v6/results/原E0二十日_全部逐日贡献与波动变化.csv",
    "fx.parquet": "data/raw/macro/510300_macro_stress_2015_v2/usdcny_midpoint_daily_2015_2026.parquet",
    "china_bonds.parquet": "data/raw/macro/china_government_bond_yields_daily.parquet",
    "market.csv": "reports/research/510300_macro_transmission_context_v4/inputs/market.csv",
}
DOCS = []
for ymd in ["20200729", "20200916", "20220727", "20220921"]:
    DOCS.append({"name": f"Fed_{ymd}.html", "url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm",
                 "local_time": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} 14:00:00", "kind": "FOMC"})
for mdy in ["09112020", "10132020", "08102022", "09132022", "10132022"]:
    DOCS.append({"name": f"CPI_{mdy}.html", "url": f"https://www.bls.gov/news.release/archives/cpi_{mdy}.htm",
                 "local_time": f"{mdy[4:]}-{mdy[:2]}-{mdy[2:4]} 08:30:00", "kind": "CPI"})
for ymd in ["20200610", "20200916", "20220615", "20220921"]:
    DOCS.append({"name": f"SEP_{ymd}.pdf", "url": f"https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl{ymd}.pdf",
                 "local_time": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} 14:00:00", "kind": "SEP"})
for year in [2020, 2022]:
    DOCS.append({"name": f"Treasury_{year}.xml", "url": f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}", "kind": "TREASURY"})
DOCS += [
    {"name": "Treasury_曲线定义.html", "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve", "kind": "DEFINITION"},
    {"name": "Treasury_XML说明.html", "url": "https://home.treasury.gov/treasury-daily-interest-rate-xml-feed", "kind": "DEFINITION"},
]


def now():
    return datetime.now().astimezone().isoformat()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮范围已固定，不覆盖。")
    for folder in ["inputs", "sources", "results", "figures", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": "510300_EXTERNAL_DISCOUNT_CLOCK_V15",
        "previous_turn_classification": "PROGRESS：V14完成同月份可比性、固定背景条件关联及累计/单月补充。",
        "question": "国内货币与信用改善之时，外部无风险利率及人民币中间价的方向是否相同？起点信息与后来新公告如何区分？",
        "cases": ["2020-08", "2022-08"], "selection": "沿V14已经固定并解释的两个8月病例，不另选涨跌最明显事件；价格结果已见，属于历史机制诊断。",
        "windows": "原E0二十日及E1结果不变。外部价格观察覆盖各快照前40个自然日至原E0退出日；此前20记录变化与全部20日路径保留。",
        "announcements": "两个原窗口内全部美联储利率决定和美国CPI发布，另含起点之前最近一期；经济预测用起点最新一期与窗口内新一期。不是完整全球或国内消息目录。",
        "source_budget": "13份FOMC/CPI/SEP官方文件，2个固定年份美国财政部XML，2份定义；BLS原HTML失败可逐文件试一次同发行者PDF，其余失败保留。",
        "clocks": {"official_announcements": "按原文件明确公布时间，用America/New_York转Asia/Shanghai；不以美国日期当中国日期。",
                   "treasury": "约美东15:30报价形成的日度CMT；本轮以美国数据日23:59:59作为研究可用时刻假设，再列额外滞后一个美债记录日的敏感性。该假设不是历史首发回执。中国同日收盘不使用随后才形成的美国当天数值。",
                   "fx": "既有中国外汇交易中心人民币中间价，按原09:15时钟；非在岸收盘、离岸价或可成交汇率，无历史不可变首版认证。",
                   "china_bonds": "沿原研究，数据日之后下一个A股开盘保守可用，非同刻跨国无套利价差。"},
        "decomposition": "政策/CPI节点前后按同一原入场金额分段贡献，节点按真正公布时间映射首个中国交易日；不把后段涨跌全部归因于节点。分清起点前、起点后入场前、入场后。",
        "missing": "无合格事前市场预期时不计算意外，不将SEP参与者预测当市场一致预期，不将美国利率或FX直接换成指数盈利损失。",
        "new_models": 0, "new_accounts": 0, "orders_authorized": False, "goal_achieved": False, "independent_validation": False}
    save("protocol.json", protocol)
    shutil.copy2(OUT / "protocol.json", ROOT / "config/510300_external_discount_clock_v15.json")
    receipts = []
    for name, source in INPUTS.items():
        src = ROOT / source
        shutil.copy2(src, OUT / "inputs" / name)
        receipts.append({"name": name, "source": source, "sha256": sha256(src.read_bytes()).hexdigest()})
    save("freeze.json", {"at": now(), "protocol_sha256": sha256((OUT / "protocol.json").read_bytes()).hexdigest(), "inputs": receipts, "documents": DOCS})
    print(json.dumps({"已固定": protocol["study_id"], "病例": protocol["cases"], "官方文件范围": len(DOCS)}, ensure_ascii=False))


def fetch(item):
    path = OUT / "sources" / item["name"]
    receipt = {**item, "retrieved_at": now(), "first_vintage_authenticated": False}
    try:
        response = requests.get(item["url"], timeout=(12, 40), headers={"User-Agent": "Mozilla/5.0 research-document-reader"})
        receipt.update({"http_status": response.status_code, "final_url": response.url})
        if response.status_code != 200:
            receipt["status"] = "FAILED_HTTP"
        else:
            content = response.content
            if item["name"].endswith(".pdf"):
                assert content.startswith(b"%PDF"), "返回内容不是PDF"
            elif item["kind"] == "TREASURY":
                root = ET.fromstring(content)
                assert len(root.findall("{http://www.w3.org/2005/Atom}entry")) > 200, "美债XML行数不完整"
            else:
                text = BeautifulSoup(content, "html.parser").get_text("\n", strip=True)
                assert len(text) > 2000 and "Access Denied" not in text[:300], "返回内容不是原公告"
                path.with_suffix(".txt").write_text(text, encoding="utf-8")
            path.write_bytes(content)
            receipt.update({"status": "SAVED", "bytes": len(content), "sha256": sha256(content).hexdigest()})
    except Exception as exc:
        receipt.update({"status": "FAILED_REQUEST_OR_CONTENT", "error": str(exc)[:240]})
    if "local_time" in item:
        receipt["expected_published_at_china"] = pd.Timestamp(item["local_time"]).tz_localize("America/New_York").tz_convert("Asia/Shanghai").isoformat()
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def collect():
    if (OUT / "source_receipts.json").exists():
        raise RuntimeError("本轮原始请求已完成，不重复请求。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, DOCS))
    save("source_receipts.json", receipts)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps([{k: r.get(k) for k in ["name", "status", "http_status", "bytes", "error"]} for r in receipts], ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["freeze", "collect"]:
        raise SystemExit("使用方式：脚本后添加freeze或collect。")
    freeze() if sys.argv[1] == "freeze" else collect()
