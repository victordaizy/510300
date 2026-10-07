"""固定三个连续货币月份，保存预期、利率与内部贡献的研究范围。"""
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
OUT = ROOT / "reports/research/510300_expectation_rates_internal_v18"
CASES = ["2020-11", "2020-12", "2021-01"]
V6 = ROOT / "reports/research/510300_information_change_transmission_v6"
V15 = ROOT / "reports/research/510300_external_discount_clock_v15"
V16 = ROOT / "reports/research/510300_constituent_external_bridge_v16"
DOCS = []
for ymd in ["20201105", "20201216", "20210127"]:
    DOCS.append({"name": f"Fed_{ymd}.html", "kind": "FOMC",
                 "url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm",
                 "local_time": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} 14:00:00"})
for mdy in ["11122020", "12102020", "01132021", "02102021", "03102021"]:
    DOCS.append({"name": f"CPI_{mdy}.pdf", "kind": "CPI",
                 "url": f"https://www.bls.gov/news.release/archives/cpi_{mdy}.pdf",
                 "local_time": f"{mdy[4:]}-{mdy[:2]}-{mdy[2:4]} 08:30:00"})
DOCS.append({"name": "Treasury_2021.xml", "kind": "TREASURY",
             "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=2021"})


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    if OUT.exists():
        raise RuntimeError("本轮目录已经存在，不覆盖。")
    for folder in ["inputs", "sources", "results", "figures", "code"]:
        (OUT / folder).mkdir(parents=True)
    protocol = {
        "at": now(), "study_id": "510300_EXPECTATION_RATES_INTERNAL_V18", "cases": CASES,
        "question": "经营共同支持状态延续时，新增货币信息、国内外利率与原始成分权重怎样连接不同价格路径？",
        "selection": "V17已明确的2020年11月至2021年1月连续三期；其ETF结果已知，属于历史机制解释，不是独立验证。",
        "windows": "保持原E0、E1二十日。原E0全部60个交易日均保留，不按后来利率走势重新择时。",
        "money": "沿用V2同口径对数当期/基数修订项、V4企业及住户中长期信贷和订单、V6事前预期；2020年11月合格预期缺失继续保留，不新增调查来源填补。",
        "expectation": "2021年1月11日及2月8日报告原第12页复核；M1、M2各调查汇总值之差只是剪刀差预期代理，非逐个预测者剪刀差的汇总。贷款与社融只在同页期间一致时比对。",
        "source_budget": "新增3份FOMC决定、5份CPI原公告PDF、1份2021年财政部XML；复用2020年XML和既有国内利率、中间价。PDF失效保留失败，不无限搜索。",
        "announcements": "三窗口内及各起点前最近的FOMC利率决定和美国CPI；不是全部新闻目录。2021年3月17日FOMC在最后窗口退出之后，不提前使用。",
        "clocks": {
            "announcements": "原明确美东发布时间转上海时区；2021年2月CPI档案有2月11日重发说明，另保守按重发日美东23:59:59可用，不认证首版数值。",
            "treasury": "美国数据日美东23:59:59假定可用，并另列再滞后一条美债记录；2020/2021连续记录直接跨年计算20条差，不按年份断开。不是首发回执。",
            "china_bond": "数据日之后下一A股开盘保守可用。",
            "fixing": "既有FDR007/FR007上午定盘，按原中午12时可见；不是全天加权DR007/R007。近20条与前20条均保留。",
            "fx": "中国外汇交易中心中间价及原09:15时钟，非CNH即期成交价。"},
        "constituents": "每期起点300只成员、起点前最近一期既有历史指数权重归一；起点前最近行业参考。权重不是ETF实际持仓，行业首版未认证。",
        "missing_constituents": "覆盖检查已见2020年11月窗口3只股票4条不合格，2021年1月窗口1只股票1条不合格；按整只股票的原20日路径是否完整判断可计算性，全部300只列出，缺失不填零不按收益剔除。",
        "contribution": "可计算股票收益乘原300只权重，不重分配缺失权重；总和仅为已覆盖贡献。另对可计算部分权重归一展示子篮子波动和广度，不能称完整ETF归因。",
        "top5": "按各起点原始权重前五固定，不按后续收益选择；全体行业保留。",
        "cash": "收盘至收盘的固定股数、分红留现金、送转调整股数；另19日入场收盘口径，不拿成分收盘收益直接等同ETF原开盘收益。",
        "risk": "子篮子日收益方差协方差贡献和负日平方贡献均恒等复核；单股负贡献允许。波动不是未来方向规则。",
        "limits": "名义长债利率未拆实际利率、通胀预期、期限溢价；未取得资金流或预期现金流冲击识别，时序与共同变化不等于唯一因果。",
        "new_models": 0, "new_accounts": 0, "orders_authorized": False, "goal_achieved": False,
        "independent_validation": False, "global_mandate_modified": False,
    }
    save("protocol.json", protocol)
    shutil.copy2(OUT / "protocol.json", ROOT / "config/510300_expectation_rates_internal_v18.json")
    copies = {
        "monthly.csv": V16 / "inputs/monthly.csv",
        "expectations.csv": V6 / "results/104个月_原因盈利定价与预期差完整连接.csv",
        "daily_paths.csv": V16 / "inputs/daily_paths.csv",
        "market.csv": V16 / "inputs/market.csv",
        "funding.parquet": ROOT / "reports/research/510300_funding_quantity_price_bridge_v12/results/逐日定盘与当时政策利率.parquet",
        "china_bonds.parquet": V15 / "inputs/china_bonds.parquet",
        "fx.parquet": V15 / "inputs/fx.parquet",
        "forecast_sources.json": V6 / "inputs/forecast_sources.json",
    }
    records = []
    for name, src in copies.items():
        target = OUT / "inputs" / name
        shutil.copy2(src, target)
        records.append({"name": name, "source": str(src.relative_to(ROOT)), "sha256": digest(target)})
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month").loc[CASES]
    origin_dates = pd.to_datetime(monthly.observation_date)
    paths = pd.read_csv(OUT / "inputs/daily_paths.csv")
    used_dates = pd.to_datetime(paths.loc[paths.stat_month.isin(CASES), "date"])
    for name, datecol in [("membership", "membership_date"), ("weights", "trade_date"), ("industry", "origin"), ("returns", "date")]:
        src = V16 / "inputs" / f"{name}.parquet"
        columns = ["origin", "stock_code", "industry_name"] if name == "industry" else None
        frame = pd.read_parquet(src, columns=columns)
        frame[datecol] = pd.to_datetime(frame[datecol])
        if name == "membership":
            part = frame[frame[datecol].isin(origin_dates)].copy()
            symbols = set(part.symbol)
            rule = "只截取三个原观察日全部成员。"
        elif name in ["weights", "industry"]:
            selected_dates = [frame.loc[frame[datecol] < day, datecol].max() for day in origin_dates]
            part = frame[frame[datecol].isin(selected_dates)].copy()
            rule = "分别截取三个观察日前最近日期；行业仅存代码和名称，不使用财务列。"
        else:
            part = frame[frame.date.isin(used_dates) & frame.symbol.isin(symbols)].copy()
            rule = "按全部原E0日期和三次成员并集截取；保留全部缺失和冲突状态。"
        dst = OUT / "inputs" / f"{name}.parquet"
        part.to_parquet(dst, index=False)
        records.append({"name": dst.name, "source": str(src.relative_to(ROOT)), "source_sha256": digest(src),
                        "sha256": digest(dst), "rows": len(part), "extraction_rule": rule})
    reused = {
        "Treasury_2020.xml": V15 / "sources/Treasury_2020.xml",
        "Treasury_2020.xml.receipt.json": V15 / "sources/Treasury_2020.xml.receipt.json",
        "Bls210111.pdf": ROOT / "reports/research/510300_money_consensus_source_extension_v1/raw/Bls210111.pdf",
        "Bls210208.pdf": V6 / "sources/Bls210208.pdf",
    }
    for name, src in reused.items():
        dst = OUT / "sources" / name
        shutil.copy2(src, dst)
        records.append({"name": "../sources/" + name, "source": str(src.relative_to(ROOT)), "sha256": digest(dst)})
    save("freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": records, "documents": DOCS})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"范围已固定": CASES, "新增原文请求上限": len(DOCS), "原始路径文件数": len(records)}, ensure_ascii=False))


def fetch(item):
    dst = OUT / "sources" / item["name"]
    receipt = {**item, "retrieved_at": now(), "historical_first_vintage_verified": False}
    try:
        r = requests.get(item["url"], timeout=(12, 40), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update(http_status=r.status_code, final_url=r.url)
        r.raise_for_status()
        if item["name"].endswith(".pdf"):
            assert r.content.startswith(b"%PDF"), "返回内容不是PDF。"
        elif item["kind"] == "TREASURY":
            tree = ET.fromstring(r.content)
            assert len(tree.findall("{http://www.w3.org/2005/Atom}entry")) > 200
        else:
            txt = BeautifulSoup(r.content, "html.parser").get_text("\n", strip=True)
            assert "For release at 2:00 p.m. EST" in txt
            dst.with_suffix(".txt").write_text(txt, encoding="utf-8")
        dst.write_bytes(r.content)
        receipt.update(status="SAVED", bytes=len(r.content), sha256=digest(dst))
    except Exception as exc:
        receipt.update(status="FAILED_SOURCE_REQUEST", error=str(exc)[:250])
    save("sources/" + dst.name + ".receipt.json", receipt)
    return receipt


def collect():
    if (OUT / "source_receipts.json").exists():
        raise RuntimeError("原文请求已执行，不重复请求。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, DOCS))
    save("source_receipts.json", rows)
    print(json.dumps([{k: r.get(k) for k in ["name", "status", "bytes", "error"]} for r in rows], ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["freeze", "collect"]:
        raise SystemExit("参数应为freeze或collect。")
    freeze() if sys.argv[1] == "freeze" else collect()
