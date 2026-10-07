"""追查2018年消费权重财报的组成、公布顺序及510300历史剩余收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pdfplumber
import numpy as np
import pandas as pd
import requests

from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_consumer_earnings_transmission_v1"
PARENT = ROOT / "reports/research/510300_historical_relief_price_transmission_v1"
STUDY = "510300_HISTORICAL_CONSUMER_EARNINGS_TRANSMISSION_V1"
SOURCES = [
    {"id": "maotai_q3", "symbol": "600519.SH", "name": "贵州茅台", "publication_date": "2018-10-29", "url": "https://static.cninfo.com.cn/finalpage/2018-10-29/1205549057.PDF"},
    {"id": "yili_q3", "symbol": "600887.SH", "name": "伊利股份", "publication_date": "2018-10-31", "url": "https://static.cninfo.com.cn/finalpage/2018-10-31/1205561719.PDF"},
    {"id": "yanghe_q3", "symbol": "002304.SZ", "name": "洋河股份", "publication_date": "2018-10-27", "url": "https://static.cninfo.com.cn/finalpage/2018-10-27/1205543056.PDF"},
    {"id": "wuliangye_q3", "symbol": "000858.SZ", "name": "五粮液", "publication_date": "2018-10-29", "url": "https://static.cninfo.com.cn/finalpage/2018-10-29/1205545060.PDF"},
]
SUPPORTS = [
    {"id": "maotai_h1", "symbol": "600519.SH", "name": "贵州茅台", "publication_date": "2018-08-02", "url": "https://static.cninfo.com.cn/finalpage/2018-08-02/1205249471.PDF", "role": "半年度收入和回款，拆出单季度变化；不是新增事件。"},
    {"id": "maotai_2017q3", "symbol": "600519.SH", "name": "贵州茅台", "publication_date": "2017-10-26", "url": "https://static.cninfo.com.cn/finalpage/2017-10-26/1204071279.PDF", "role": "核对2017年同期增长基数及可比口径；不是新增事件。"},
    {"id": "yanghe_h1", "symbol": "002304.SZ", "name": "洋河股份", "publication_date": "2018-08-30", "url": "https://pdf.dfcfw.com/pdf/H2_AN201808291184493630_1.pdf", "extract_pages": [1, 15, 75], "role": "公司半年报全文的公开镜像；核对事前1至9月业绩预告和税费核算说明。归档镜像名称为8月29日，保守按次日公开；不是新增交易事件。"},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, np.generic):
        return clean(x.item())
    if x is pd.NA or x is pd.NaT or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return x


def save(name, x):
    (OUT / name).write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已保存本轮设定，不覆盖。")
    prior = json.loads((PARENT / "protocol.json").read_text(encoding="utf-8"))
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "research_mode": "HISTORICAL_ONLY",
        "previous_goal_turn_classification": "PROGRESS_HISTORICAL_RELIEF_TRANSMISSION_AND_DELAYED_REMAINING_RETURN",
        "question": "2018年食品饮料权重下跌与财报公布孰先孰后，收入、利润、回款和基数分别怎样变化；正式公开以后510300还剩何种收益？",
        "sample_selection": "上一轮已见负贡献较大的四个食品饮料公司，事后原因追查，不是独立样本，也不代表行业全体。保留四个案例和所有公布日，不按本轮后续收益删选。",
        "sources": SOURCES, "some_headline_financial_numbers_already_seen": True,
        "before_new_event_returns": True, "independent_validation": False,
        "fixed_prior_window": ["2018-10-19", "2018-10-26"],
        "statement_analysis": "区分营业收入与含金融子公司利息的营业总收入；区分累计与单季度；分解毛利、费用、税费与金融现金项目。同比增长变化不是已测量的市场预期差。",
        "clock": "公开归档日期按日末处理，下一实际交易日开盘；如公司文件落款或互联网传播更早，只记录，不悄悄提前交易时点。",
        "event_hold_sessions": [5, 20], "traded_asset": "510300.SH", "other_assets": "OBSERVATION_ONLY",
        "costs": prior["costs"],
        "overlap": "相同入场日期的报告合并为一个ETF收益节点，不能把同一ETF交易复制成多次独立成功。",
        "company_return_role": "固定已见观察窗口收益及单股收盘路径只作原因定位，不模拟单股交易。",
        "new_full_accounts": 0, "new_parameters_fitted": 0, "new_prospective_forecasts": 0,
        "goal_achieved": False, "orders_authorized": False,
        "reuse_2024": "复用510300_historical_huijin_ownership_transmission_v1；季末持有增量不能还原2月逐日买入，旧披露策略失败不重新拟合。",
    })
    print("已固定四家公司及5日、20日期限；2024年持有研究只复用。", flush=True)


def fetch():
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)
    old_index = {r["id"]: r for r in json.loads((OUT / "source_index.json").read_text(encoding="utf-8"))} if (OUT / "source_index.json").exists() else {}
    def one(spec):
        if spec["id"] in old_index and old_index[spec["id"]].get("status") == "PDF_TEXT_ACQUIRED" and (raw/(spec["id"]+"_pages.json")).exists():
            return old_index[spec["id"]]
        row = dict(spec, retrieved_at=now())
        if spec["id"] in old_index:
            row["previous_attempt"] = old_index[spec["id"]]
        path = raw / (spec["id"]+".pdf")
        try:
            if path.exists():
                body = path.read_bytes()
                row["mode"] = "REUSED_THIS_STUDY_FILE"
            else:
                response = requests.get(spec["url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                row["http_status"] = response.status_code
                response.raise_for_status()
                body = response.content
                assert body.startswith(b"%PDF"), "返回内容不是PDF。"
                path.write_bytes(body)
                row["mode"] = "NEW_PUBLIC_SOURCE"
            with pdfplumber.open(BytesIO(body)) as doc:
                total_pages = len(doc.pages)
                pages = [{"page": i, "text": doc.pages[i-1].extract_text(layout=False) or ""} for i in spec.get("extract_pages", range(1, total_pages+1))]
            issuer = "江苏洋河酒厂" if spec["id"].startswith("yanghe_") else spec["name"]
            assert issuer in "".join(p["text"] for p in pages[:3]).replace(" ", ""), "PDF前页未核实发行人名称。"
            (raw/(spec["id"]+".txt")).write_text("\n".join(f"===== 第{p['page']}页 =====\n{p['text']}" for p in pages), encoding="utf-8")
            save("raw/"+spec["id"]+"_pages.json", pages)
            row.update(status="PDF_TEXT_ACQUIRED", pages=total_pages, extracted_page_count=len(pages), local_pdf=str(path.resolve()), sha256=hashlib.sha256(body).hexdigest())
        except (requests.RequestException, AssertionError, RuntimeError) as exc:
            row.update(status="SOURCE_REQUEST_FAILED", error=str(exc))
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, cfg["sources"]+SUPPORTS))
    save("supporting_sources.json", {"recorded_at": now(), "purpose": "仅补充基数及分季计算，不新增交易事件或收益期限。", "sources": SUPPORTS})
    save("source_index.json", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def statement_facts():
    """按已读原表逐行录入；页码为PDF物理页码，保留列顺序及金额单位。"""
    rows = []
    pages = {}
    for spec in SOURCES+SUPPORTS:
        pages[spec["id"]] = {p["page"]: p["text"] for p in json.loads((OUT/"raw"/(spec["id"]+"_pages.json")).read_text(encoding="utf-8"))}

    def add(key, source, page, label, columns, values, unit="CNY"):
        text = re.sub(r"\s", "", pages[source][page]).replace(",", "")
        for value in values:
            assert f"{value:.2f}" in text, f"{source}第{page}页没有原值：{value}"
        rows.append({"key": key, "source_id": source, "pdf_page": page, "row_label": label,
                     "columns": columns.split("|"), "values": values, "unit": unit,
                     "method": "原表列顺序人工核对，程序核对数字存在；不是自动推断字段。"})
        return np.array(values, dtype=float)

    col4 = "2018Q3|2017Q3|2018YTD_Q3|2017YTD_Q3"
    col2 = "2018YTD_Q3|2017YTD_Q3"
    qcol2 = "2018Q3|2017Q3"
    mrev = add("m_revenue", "maotai_q3", 13, "营业收入", col4, [18844960092.90,18260437281.05,52241669986.01,42450467500.01])
    mprofit = add("m_profit", "maotai_q3", 15, "归属于母公司所有者的净利润", col4, [8969366937.67,8732986054.39,24733552720.33,19983846984.10])
    mhrev = add("m_h1_revenue", "maotai_h1", 5, "营业收入", "2018H1|2017H1", [33396709893.11,24190030218.96])
    mhprofit = add("m_h1_profit", "maotai_h1", 5, "归属于上市公司股东的净利润", "2018H1|2017H1", [15764185782.66,11250860929.71])
    moldrev = add("m_prior_revenue", "maotai_2017q3", 13, "营业收入", "2017Q3|2016Q3|2017YTD_Q3|2016YTD_Q3", [18260437281.05,8458649358.99,42450467500.01,26631884032.43])
    moldprofit = add("m_prior_profit", "maotai_2017q3", 14, "归属于母公司所有者的净利润", "2017Q3|2016Q3|2017YTD_Q3|2016YTD_Q3", [8732986054.39,3662940626.12,19983846984.10,12465577764.14])
    mprepay = add("m_prepay_q3", "maotai_q3", 10, "预收款项", "2018-09-30|2017-12-31", [11167533769.62,14429106902.38])
    mhprepay = add("m_prepay_h1", "maotai_h1", 24, "预收款项", "2018-06-30|2017-12-31", [9940315208.28,14429106902.38])

    mcash_specs = [
        ("goods",18,"销售商品、提供劳务收到的现金",[57619330306.41,48901003501.12],[34611486701.83,28152402920.57],1),
        ("deposits",18,"客户存款和同业存放款项净增加额",[8229882388.12,1119829586.02],[2533121205.32,-1989660145.95],1),
        ("interest_in",18,"收取利息、手续费及佣金的现金",[2583282270.91,1902470710.34],[1711480390.14,1258038556.05],1),
        ("other_in",18,"收到其他与经营活动有关的现金",[594104103.42,579053392.17],[488566388.95,433271070.21],1),
        ("purchases",18,"购买商品、接受劳务支付的现金",[3676356107.55,3385873042.93],[2276635677.35,2145743960.98],-1),
        ("loans",18,"客户贷款及垫款净增加额",[10000000.00,-32393350.80],[10000000.00,167606649.20],-1),
        ("bank_placements",18,"存放中央银行和同业款项净增加额",[2149145701.58,828004708.44],[-7379011885.99,546566507.44],-1),
        ("interest_out",18,"支付利息、手续费及佣金的现金",[86293015.37,117837629.18],[42180573.87,64186151.31],-1),
        ("staff",18,"支付给职工以及为职工支付的现金",[5116193340.71,4411221320.90],[3972027077.20,3405303499.19],-1),
        ("tax",19,"支付的各项税费",[27578911212.84,19059728251.36],[21509649014.42,13284387765.88],-1),
        ("other_out",19,"支付其他与经营活动有关的现金",[2188413775.27,1945318389.45],[1178143267.20,1304897456.52],-1),
    ]
    mcash = {}
    for key, page, label, ytd, half, sign in mcash_specs:
        a = add("m_cash_ytd_"+key, "maotai_q3", page, label, col2, ytd)
        b = add("m_cash_h1_"+key, "maotai_h1", 30, label, "2018H1|2017H1", half)
        mcash[key] = {"ytd": a, "h1": b, "q3": a-b, "sign": sign}
    mcfo = add("m_cfo_ytd", "maotai_q3", 19, "经营活动产生的现金流量净额", col2, [28221285915.54,22786767198.19])
    mhcfo = add("m_cfo_h1", "maotai_h1", 30, "经营活动产生的现金流量净额", "2018H1|2017H1", [17735030962.19,6935360410.36])

    irev = add("i_revenue", "yili_q3", 15, "营业收入", col4, [21257756990.76,18825129256.65,60846276714.34,52126895563.29])
    icost = add("i_cost", "yili_q3", 15, "营业成本", col4, [13647777423.82,11775353354.57,37927214178.17,32358565330.69])
    isales = add("i_sales", "yili_q3", 15, "销售费用", col4, [5163977150.91,4164186004.48,15333885574.52,11812041928.04])
    iprofit = add("i_profit", "yili_q3", 16, "归属于母公司所有者的净利润", col4, [1601987824.20,1573156230.86,5047818747.67,4937267331.79])

    yrevq = add("y_revenue_q3", "yanghe_q3", 23, "营业收入", qcol2, [6423080855.11,5347834958.10])
    yprofitq = add("y_profit_q3", "yanghe_q3", 24, "归属于母公司所有者的净利润", qcol2, [2033763138.67,1673401613.10])
    yrevy = add("y_revenue_ytd", "yanghe_q3", 27, "营业收入", col2, [20965660630.72,16878326032.28])
    ycost = add("y_cost_ytd", "yanghe_q3", 27, "营业成本", col2, [5653402852.96,6371076317.86])
    ytax = add("y_tax_ytd", "yanghe_q3", 27, "税金及附加", col2, [3371234585.87,557558898.83])
    yprofity = add("y_profit_ytd", "yanghe_q3", 3, "归属于上市公司股东的净利润", "2018YTD_Q3", [7038754909.89])
    yguidance = add("y_guidance", "yanghe_h1", 15, "2018年1至9月归母净利润预告", "lower|upper|2017YTD_Q3_base", [669799.45,725616.07,558166.21], "CNY_10000")*10000

    wrevq = add("w_revenue_q3", "wuliangye_q3", 15, "营业收入", qcol2, [7829106181.73,6356468969.14])
    wprofitq = add("w_profit_q3", "wuliangye_q3", 16, "归属于母公司所有者的净利润", qcol2, [2384160895.83,1993257688.85])
    wcash_specs = [
        ("goods",22,"销售商品、提供劳务收到的现金",[26438870647.47,21956722427.07],1),
        ("refund",22,"收到的税费返还",[4336017.86,25329875.74],1),
        ("other_in",22,"收到其他与经营活动有关的现金",[1056252658.69,925030795.92],1),
        ("purchases",22,"购买商品、接受劳务支付的现金",[5630219819.98,4416529916.94],-1),
        ("staff",23,"支付给职工以及为职工支付的现金",[3733232035.69,2664960126.43],-1),
        ("tax",23,"支付的各项税费",[12154156688.99,7814824936.87],-1),
        ("other_out",23,"支付其他与经营活动有关的现金",[2104980744.91,1917602720.85],-1),
    ]
    wcash = {k: {"values": add("w_cash_"+k,"wuliangye_q3",p,l,col2,v), "sign": s}
             for k,p,l,v,s in wcash_specs}
    wcfo = add("w_cfo_ytd", "wuliangye_q3", 23, "经营活动产生的现金流量净额", col2, [3876870034.45,6093165397.64])

    yoy = lambda a: float(a[0]/a[1]-1)
    deltap = lambda a: float((a[0]-a[1])*100)
    financial = ["deposits","interest_in","loans","bank_placements","interest_out"]
    cashblocks = []
    casherrors = []
    for period,cfo in [("ytd",mcfo),("h1",mhcfo),("q3",mcfo-mhcfo)]:
        fin = sum(mcash[k][period]*mcash[k]["sign"] for k in financial)
        rest = sum(v[period]*v["sign"] for k,v in mcash.items() if k not in financial)
        casherrors.append(float(np.max(np.abs(fin+rest-cfo))))
        cashblocks.append({"period": period, "financial_statement_lines_cny": fin.tolist(),
                           "remaining_statement_lines_cny": rest.tolist(), "cfo_cny": cfo.tolist(),
                           "remaining_lines_are_liquor_segment_cashflow": False})
    q3block = next(x for x in cashblocks if x["period"] == "q3")
    q3cash_change = {k: float(v[0]-v[1]) for k,v in q3block.items() if k.endswith("_cny")}
    wdecomp = [{"item": k, "signed_change_cny": float(v["sign"]*(v["values"][0]-v["values"][1]))} for k,v in wcash.items()]
    mechanisms = {
        "maotai": {
            "q3_revenue_yoy": yoy(mrev[:2]), "q3_parent_profit_yoy": yoy(mprofit[:2]),
            "h1_revenue_yoy": yoy(mhrev), "h1_parent_profit_yoy": yoy(mhprofit),
            "prior_q3_revenue_yoy": yoy(moldrev[:2]), "prior_q3_parent_profit_yoy": yoy(moldprofit[:2]),
            "q3_revenue_two_year_cagr": float((mrev[0]/moldrev[1])**0.5-1),
            "q3_goods_cash_yoy": yoy(mcash["goods"]["q3"]),
            "ytd_cfo_yoy": yoy(mcfo), "q3_cfo_yoy": yoy(mcfo-mhcfo),
            "prepayment_q3_change_cny": float(mprepay[0]-mhprepay[0]),
            "cash_blocks": cashblocks,
            "q3_cash_change_decomposition_cny": q3cash_change,
            "causal_limit": "增长放缓和高基数均真实；财务公司存款及银行存放影响合并现金流。销售收现还包含结算和预收变化，不等于终端动销。没有量价及渠道库存证据，不能将增速放缓全部归因为需求下降。",
        },
        "yili": {
            "q3_revenue_yoy": yoy(irev[:2]), "q3_parent_profit_yoy": yoy(iprofit[:2]),
            "q3_gross_margins": (1-icost[:2]/irev[:2]).tolist(),
            "q3_gross_margin_change_pp": deltap(1-icost[:2]/irev[:2]),
            "q3_sales_expense_ratios": (isales[:2]/irev[:2]).tolist(),
            "q3_sales_expense_ratio_change_pp": deltap(isales[:2]/irev[:2]),
            "causal_limit": "成本率与销售费用率上升是报表层面的利润传导，不足以继续归因到某种原奶价格、促销竞争或终端数量；这些上游解释本轮未证实。",
        },
        "yanghe": {
            "q3_revenue_yoy": yoy(yrevq), "q3_parent_profit_yoy": yoy(yprofitq),
            "ytd_gross_margins": (1-ycost/yrevy).tolist(),
            "ytd_gross_margin_change_pp": deltap(1-ycost/yrevy),
            "ytd_margin_after_cost_and_tax": (1-(ycost+ytax)/yrevy).tolist(),
            "ytd_margin_after_cost_and_tax_change_pp": deltap(1-(ycost+ytax)/yrevy),
            "guidance_bounds_cny": yguidance[:2].tolist(), "actual_ytd_parent_profit_cny": float(yprofity[0]),
            "actual_inside_prior_company_guidance": bool(yguidance[0] <= yprofity[0] <= yguidance[1]),
            "guidance_is_market_consensus": False,
            "guidance_available_by_conservative_date": "2018-08-30",
            "tax_change_already_disclosed_in_half_year_report": True,
            "upstream_cause": "报告记载2017年9月起由委托加工改为自行生产，消费税从成本改列税金及附加；2017年5月起最低计税价格规则也发生变化，核定比例由50%至70%统一为60%。不能把后者一律表述成每家公司加税10个百分点。",
            "causal_limit": "收入减成本及税金比例是提高可比性的诊断量，不是还原后的精确经济毛利率；仍含真实税负、产品结构等变化。未获得同期市场一致预期。",
        },
        "wuliangye": {
            "q3_revenue_yoy": yoy(wrevq), "q3_parent_profit_yoy": yoy(wprofitq),
            "ytd_cfo_yoy": yoy(wcfo), "goods_cash_yoy": yoy(wcash["goods"]["values"]),
            "cfo_change_cny": float(wcfo[0]-wcfo[1]),
            "cash_change_decomposition": wdecomp,
            "upstream_cause": "公司解释银行承兑汇票结算增加，税费和采购支付增加；现金流量表还显示工资支出上升。销售收现本身仍增长，净现金流下降不能直接替代销量下降。",
            "causal_limit": "票据增加可能体现结算安排或经销商融资约束；本轮无法分辨两者的占比。现金税费是实际支付，分解后不能在估值或账户中当作可以忽略的支出。",
        },
    }
    errors = {
        "maotai_revenue_half_plus_quarter_equals_ytd_max_abs_cny": float(np.max(np.abs(mhrev+mrev[:2]-mrev[2:]))),
        "maotai_profit_half_plus_quarter_equals_ytd_max_abs_cny": float(np.max(np.abs(mhprofit+mprofit[:2]-mprofit[2:]))),
        "maotai_cash_identity_max_abs_cny": max(casherrors),
        "wuliangye_cash_identity_max_abs_cny": float(np.max(np.abs(sum(v["values"]*v["sign"] for v in wcash.values())-wcfo))),
        "wuliangye_cash_change_decomposition_abs_cny": abs(sum(x["signed_change_cny"] for x in wdecomp)-(wcfo[0]-wcfo[1])),
    }
    assert all(v < 0.05 for v in errors.values()), errors
    save("财报原值及页码.json", rows)
    save("经营利润现金与预告分解.json", mechanisms)
    return mechanisms, errors, len(rows)


def archive_event_returns(cfg):
    market = pd.read_parquet(ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    assert market["date"].is_unique, "ETF行情存在重复日期。"
    dividends = pd.read_csv(ROOT/"data/reference/510300_dividends.csv")
    dividends["record_date"] = pd.to_datetime(dividends["record_date"])
    dividends = dividends.loc[dividends.symbol == "510300.SH"]
    calendar = pd.DatetimeIndex(market.date)
    groups = {}
    clocks = []
    signed_dates = {"maotai_q3": "2018-10-26", "yili_q3": "2018-10-29", "yanghe_q3": None, "wuliangye_q3": "2018-10-29"}
    for s in cfg["sources"]:
        idx = int(calendar.searchsorted(pd.Timestamp(s["publication_date"]), side="right"))
        groups.setdefault(idx, []).append(s)
        clocks.append({"source_id": s["id"], "issuer": s["name"], "signed_date": signed_dates[s["id"]],
                       "archive_date": s["publication_date"], "proxy_entry": calendar[idx].date().isoformat(),
                       "first_public_timestamp_verified": False,
                       "clock_limit": "按归档日期日末取得文件的保守回放，不是经核实的首次公开时点。文件落款不直接当作公开时点。"})
    rows = []
    for idx, specs in sorted(groups.items()):
        for horizon in cfg["event_hold_sessions"]:
            exit_idx = idx+horizon
            entry, end = market.iloc[idx], market.iloc[exit_idx]
            entitled = dividends.loc[(dividends.record_date >= entry.date)&(dividends.record_date < end.date),"cash_dividend_per_share"].sum()
            calc = round_trip(entry.open,end.open,float(entitled),cfg["costs"])
            rows.append({"node_id": f"ARCHIVE_{entry.date:%Y%m%d}", "issuers": [s["name"] for s in specs],
                         "source_ids": [s["id"] for s in specs], "entry_date": entry.date.date().isoformat(),
                         "exit_date": end.date.date().isoformat(), "holding_sessions": horizon,
                         "entry_index": idx, "exit_index": exit_idx,
                         "entry_open": float(entry.open), "exit_open": float(end.open),
                         "dividends_per_share": float(entitled),
                         "gross_return": float((end.open+entitled)/entry.open-1), **calc,
                         "illustrative_full_capital_pnl_ratio": calc["net_pnl_cny"]/cfg["costs"]["illustrative_full_capital_cny"]})
    prior = pd.read_parquet(PARENT/"固定成分分段回报.parquet")
    prior = prior.loc[(prior.event_id=="P18_OCT")&(prior.segment=="FIRST_FIVE")&prior.symbol.isin([s["symbol"] for s in cfg["sources"]])].copy()
    assert len(prior)==4 and prior.symbol.is_unique, "上一轮四家公司观察窗不完整。"
    prior["name"] = prior.symbol.map({s["symbol"]:s["name"] for s in cfg["sources"]})
    prior_rows = prior[["symbol","name","weight","total_return","contribution"]].to_dict("records")
    save("公开归档与落款日期.json", clocks)
    save("三个归档节点510300剩余收益.json", rows)
    save("复用四公司事前观察窗.json", {"window": cfg["fixed_prior_window"], "rows": prior_rows,
         "weight_sum": float(prior.weight.sum()), "contribution_sum": float(prior.contribution.sum()),
         "use_limit": "使用上一轮固定2018年9月末权重解释10月19至26日走势；不把权重历史首次发布时间已核实作为前提，不构成信号。"})
    return rows, clocks, prior_rows


def render_figure(mechanisms, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axs = plt.subplots(1,3,figsize=(15,5.7),layout="constrained")
    colors = ["#245b78","#c87843"]
    labels = ["茅台","伊利","洋河","五粮液"]
    keys = ["maotai","yili","yanghe","wuliangye"]
    x = np.arange(4)
    for j,(field,label) in enumerate([("q3_revenue_yoy","单季营业收入"),("q3_parent_profit_yoy","单季归母利润")]):
        values = [mechanisms[k][field]*100 for k in keys]
        bars = axs[0].bar(x+(j-.5)*.34,values,width=.34,color=colors[j],label=label)
        axs[0].bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    axs[0].set_xticks(x,labels)
    axs[0].set_ylim(0,30)
    axs[0].set_ylabel("2018年第三季度同比（%）")
    axs[0].set_title("此前同跌，经营变化不同",loc="left",fontweight="bold")
    axs[0].legend(frameon=False,loc="upper left")
    y=mechanisms["yanghe"]
    vals=[y["ytd_gross_margin_change_pp"],y["ytd_margin_after_cost_and_tax_change_pp"]]
    bars=axs[1].bar([0,1],vals,color=colors,width=.5)
    axs[1].bar_label(bars,fmt="%+.2f",padding=4)
    axs[1].set_xticks([0,1],["原毛利率变化","再扣税金及附加"])
    axs[1].axhline(0,color="#999999",lw=.7)
    axs[1].set_ylim(-5,14)
    axs[1].set_ylabel("2018年前三季度同比变化（百分点）")
    axs[1].set_title("洋河：列报位置影响因子方向",loc="left",fontweight="bold")
    axs[1].text(.03,.93,"消费税从成本转列税金\n此事项已在半年报披露",transform=axs[1].transAxes,va="top",fontsize=9)
    dates=sorted({r["entry_date"] for r in events})
    for j,h in enumerate([5,20]):
        vals=[next(r["net_return"]*100 for r in events if r["entry_date"]==d and r["holding_sessions"]==h) for d in dates]
        bars=axs[2].bar(np.arange(3)+(j-.5)*.34,vals,width=.34,color=colors[j],label=f"{h}个交易日")
        axs[2].bar_label(bars,fmt="%+.2f",padding=3,fontsize=9)
    axs[2].set_xticks(np.arange(3),[d[5:] for d in dates])
    axs[2].set_ylim(-2,10)
    axs[2].axhline(0,color="#999999",lw=.7)
    axs[2].set_ylabel("510300参考交易成本后收益（%）")
    axs[2].set_title("归档后入场，保留全部三个节点",loc="left",fontweight="bold")
    axs[2].legend(frameon=False,loc="upper right")
    fig.suptitle("2018年消费权重：先拆变化的来源，再看公开后还剩什么",fontsize=17,fontweight="bold")
    fig.supxlabel("四个已见下跌案例的历史追查；三个ETF节点互相重叠，不是独立验证或完整账户。\n交易按归档日后下一开盘；首次公开时点未核实。每笔预算10万元，收益分母为实际投入。",fontsize=10)
    fig.savefig(OUT/"消费权重经营组成与剩余收益.png",dpi=160)
    plt.close(fig)


def write_report(result, m, events, clocks, prior):
    fmt=lambda v:f"{v*100:+.2f}%"
    file="历史发现_消费权重为何同跌却不同因.md"
    lines=["# 历史发现：2018年消费权重同跌，却不能套用同一盈利因子", "",
        "本轮只做历史原因追查和固定期限的收益计算。四家公司来自上一轮已经知道跌幅的消费权重案例，不能视为事前选出的独立样本。目标仍是20万元完整账户成本后夏普不低于1.2；本轮没有达标账户，也没有改动旧策略。", "",
        "最有用的新发现是：同样叫‘盈利增长’、‘毛利率’或‘经营现金流’，背后分别可能是高基数、成本费用挤压、税费核算迁移、票据结算和金融子公司的资金往来。先查清哪条机制在变，因子的方向才有经济含义。", "",
        "## 一、此前同跌，三季报给出的经营变化不同", "",
        "|公司|10月19至26日已见收益|第三季度营业收入同比|第三季度归母利润同比|", "|---|---:|---:|---:|" ]
    keys={"600519.SH":"maotai","600887.SH":"yili","002304.SZ":"yanghe","000858.SZ":"wuliangye"}
    for r in prior:
        v=m[keys[r["symbol"]]]
        lines.append(f"|{r['name']}|{fmt(r['total_return'])}|{fmt(v['q3_revenue_yoy'])}|{fmt(v['q3_parent_profit_yoy'])}|")
    lines += ["", "四家公司采用上一轮固定的9月末权重，合计6.995%，对该段指数篮子回报的贡献约−0.685个百分点。该贡献只定位价格变化发生在哪里，没有证明哪份报告造成了下跌。", "",
        "## 二、往上游追一层，哪些结论可以保留", "",
        f"**茅台：增长放缓、高基数和合并现金口径需要同时看。** 半年收入同比{fmt(m['maotai']['h1_revenue_yoy'])}、归母利润同比{fmt(m['maotai']['h1_parent_profit_yoy'])}；第三季度分别只增{fmt(m['maotai']['q3_revenue_yoy'])}和{fmt(m['maotai']['q3_parent_profit_yoy'])}。但2017年同季收入曾增{fmt(m['maotai']['prior_q3_revenue_yoy'])}、归母利润曾增{fmt(m['maotai']['prior_q3_parent_profit_yoy'])}。高基数解释比较尺度，不能据此否认减速，也不能证明终端消费下降。", "",
        f"将累计表减去半年表，第三季度销售商品收现同比仍增{fmt(m['maotai']['q3_goods_cash_yoy'])}，合并经营现金流却下降{abs(m['maotai']['q3_cfo_yoy'])*100:.2f}%；预收款余额较6月末增加{m['maotai']['prepayment_q3_change_cny']/1e8:.2f}亿元。财务公司吸收存款、利息以及银行存放资金影响合并现金流，因此不能用合并现金流增速直接给白酒动销打分。本轮把金融列单独加总并与剩余列相加核回原表；剩余列仍含共同税费等，不能冒称白酒分部现金流。", "",
        f"具体分解为：单季经营现金流同比减少{abs(m['maotai']['q3_cash_change_decomposition_cny']['cfo_cny'])/1e8:.2f}亿元，其中存款、贷款、银行存放及利息相关列合计减少{abs(m['maotai']['q3_cash_change_decomposition_cny']['financial_statement_lines_cny'])/1e8:.2f}亿元，剩余表内项目合计增加{m['maotai']['q3_cash_change_decomposition_cny']['remaining_statement_lines_cny']/1e8:.2f}亿元。这里有可复算的金额恒等关系；它解释现金流指标的构成，不证明白酒终端需求一定改善。", "",
        f"**伊利：收入增长没有同幅转成利润。** 第三季度毛利率由{m['yili']['q3_gross_margins'][1]*100:.2f}%降至{m['yili']['q3_gross_margins'][0]*100:.2f}%，下降{abs(m['yili']['q3_gross_margin_change_pp']):.2f}个百分点；销售费用率由{m['yili']['q3_sales_expense_ratios'][1]*100:.2f}%升至{m['yili']['q3_sales_expense_ratios'][0]*100:.2f}%。这是利润转换受压的直接证据。原奶、包装材料、促销竞争或产品组合各占多少，本轮原文还不足以判断；不把合理猜测写成原因已证实。", "",
        f"**洋河：表面毛利率改善，存在税费核算迁移。** 前三季度原毛利率上升{m['yanghe']['ytd_gross_margin_change_pp']:.2f}个百分点；将税金及附加也扣除后，收入剩余比例反而下降{abs(m['yanghe']['ytd_margin_after_cost_and_tax_change_pp']):.2f}个百分点。公司已在半年报解释，2017年9月由委托加工改为自行生产，消费税从生产成本改列税金及附加；最低计税价格规则也曾改变。将成本与税金合看可以减少列报位置干扰，但不能完全剔除真实税负、品类结构等变化。", "",
        "更关键的是，这项核算变化早已公开；半年报还预计2018年前三季度归母利润增长20%—30%，实际增长26.10%，处于公司原预告范围内。因此不能把‘利润同比增长降低’直接认定为相对公司预告的负面意外。公司预告不是市场一致预期，本轮没有测得市场预期差。", "",
        f"**五粮液：现金流下降并不等于销售收现下降。** 前三季度销售收现同比{fmt(m['wuliangye']['goods_cash_yoy'])}，经营现金流净额却同比{fmt(m['wuliangye']['ytd_cfo_yoy'])}，金额减少{abs(m['wuliangye']['cfo_change_cny'])/1e8:.2f}亿元。税费现金支出增加{abs(next(x['signed_change_cny'] for x in m['wuliangye']['cash_change_decomposition'] if x['item']=='tax'))/1e8:.2f}亿元，采购和职工支出也增加。公司还说明承兑汇票结算增加。结算工具变化和支付节奏必须单独分析，但真实税费支出不能从账户或估值中凭空删去。", "",
        "## 三、保留信息时点，再看已经实现的剩余收益", "",
        "四份季报归档日期均晚于前述10月19至26日观察窗。文件落款日期不代表首次公开日期，本轮未核实交易所精确首次发布时间。下面严格执行计算前固定的保守口径：归档日结束后，下一个交易日开盘买入510300，5或20个交易日后开盘卖出。它回答‘按此归档时点取得文件还剩多少’，不能冒充首次公告的全部可交易反应。", "",
        "|相关报告|归档后参考入场|5日成本后收益|20日成本后收益|", "|---|---|---:|---:|" ]
    for d in sorted({r["entry_date"] for r in events}):
        a=next(r for r in events if r["entry_date"]==d and r["holding_sessions"]==5)
        b=next(r for r in events if r["entry_date"]==d and r["holding_sessions"]==20)
        lines.append(f"|{'、'.join(a['issuers'])}|{d}|{fmt(a['net_return'])}|{fmt(b['net_return'])}|")
    lines += ["", "每笔示例预算10万元，佣金单边万四且最低5元、滑点单边0.1%、千分之一元价位向不利方向取整、100份整手；分红按登记日权益计算。收益分母为实际投入，不是20万元全账户。相同入场日的两家公司合并成一个节点，三个节点仍处于同一时期且大量重叠，不能当作三次独立检验，也不计算有误导性的事件夏普。", "",
        "## 四、这轮改变了什么判断", "",
        "- 不保留‘食品饮料跌了，所以全行业收入利润都恶化’的统一叙事；四家公司的变化来源不同。",
        "- 不把洋河原始毛利率上升直接解释为竞争力提升；需合看成本与税金，也需识别此前已公开的信息。",
        "- 不把合并现金流下降直接替代终端需求下降；先拆金融子公司、票据与税费支付。",
        "- 预期差先找当时的比较基准。增长低于前一季度不等于低于公司预告，更不等于低于市场一致预期。",
        "- 宏观或政策作用到指数时，要经过权重公司、风险溢价及库存承接等渠道；本轮未估计每个渠道的因果效应，不能用三组正负收益反推筛选条件。", "",
        "2024年持有人资料沿用已完成研究：季末增持与单日现金买盘不能互换，实际持有人身份及数额公开晚于2月的价格变化；原有披露策略未达标，不重新拟合。", "",
        "最值得继续的历史问题是：在原始公告时点已知的预告、经营解释与渠道资料中，哪些信息已经消化、哪些确实修正了盈利预期。优先补清2018年伊利成本费用增加的上游原因和茅台量价/渠道结算的可识别部分，不因三次ETF收益临时增加交易规则。", "",
        f"本轮核对{result['financial_source_rows']}条原表数列，复算分季恒等关系、经营现金流加总和六条交易收益；未运行新的完整账户、参数搜索或前瞻任务。夏普1.2目标仍未实现。", "",
        "![经营组成与剩余收益](消费权重经营组成与剩余收益.png)", "",
        "## 原文与可复算资料", "" ]
    for spec in SOURCES+SUPPORTS:
        lines.append(f"- [{spec['name']}：{spec['id']}]({spec['url']})；本地原件 `raw/{spec['id']}.pdf`。")
    lines += ["", "计算脚本：`research/historical_consumer_earnings_transmission_v1.py analyze`。原表列、物理页码与原值保存在`财报原值及页码.json`；现金分解和限制保存在`经营利润现金与预告分解.json`；各交易的开盘价、股数、费用及退出日保存在`三个归档节点510300剩余收益.json`。", "",
        "报告是历史机制发现，不是独立收益验证。已见样本、未核实的首次公开时间、缺少同期市场一致预期，以及未识别的终端需求与渠道库存，是结论的实际边界。", ""]
    (OUT/file).write_text("\n".join(lines),encoding="utf-8")
    return file


def analyze():
    cfg=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    source_index=json.loads((OUT/"source_index.json").read_text(encoding="utf-8"))
    assert len(source_index)==7 and all(r["status"]=="PDF_TEXT_ACQUIRED" for r in source_index)
    mechanisms, errors, nrows=statement_facts()
    events, clocks, prior=archive_event_returns(cfg)
    assert len(events)==6 and len({r["node_id"] for r in events})==3
    assert all(r["exit_index"]-r["entry_index"]==r["holding_sessions"] for r in events)
    checks={"status":"PASS_BOUNDED_SOURCE_ARITHMETIC_AND_EVENT_CLOCK_CHECKS",**errors,
            "financial_source_rows":nrows,"unique_etf_nodes":3,"event_return_rows":6,
            "formal_first_public_timestamp_verified":False,"independent_validation":False}
    save("calculation_checks.json",checks)
    result={"study_id":STUDY,"completed_at":now(),
            "status":"HISTORICAL_EARNINGS_COMPONENTS_GUIDANCE_AND_ARCHIVE_RETURNS_COMPLETED",
            "classification":"PROGRESS_HISTORICAL_CONSUMER_EARNINGS_CAUSES_AND_EXPECTATION_ANCHOR",
            "research_mode":"HISTORICAL_ONLY","primary_company_pdfs":7,"selected_companies":4,
            "financial_source_rows":nrows,"unique_etf_nodes":3,"event_return_rows":6,
            "known_downmove_selected_sample":True,"independent_validation":False,
            "new_full_accounts":0,"new_parameters_fitted":0,"new_prospective_tasks":0,
            "newly_achieved_sharpe":None,"goal_achieved":False,"orders_authorized":False,
            "mechanism_findings":mechanisms,
            "latest_2024_source_reused":"reports/research/510300_historical_huijin_ownership_transmission_v1/result.json",
            "old_failures_reclassified":False,
            "next_historical_question":"追溯伊利成本费用挤压、茅台量价及结算变化的同期公司解释和公开时点；区分相对公司预告与市场预期的变化，不按本轮ETF收益生成规则。"}
    render_figure(mechanisms,events)
    result["report"]=write_report(result,mechanisms,events,clocks,prior)
    save("result.json",result)
    print(json.dumps({"状态":result["status"],"原值行数":nrows,"必要核对":checks,
                      "三个节点收益":[{k:r[k] for k in ["entry_date","exit_date","holding_sessions","net_return"]} for r in events],
                      "机制分解":mechanisms},ensure_ascii=False,indent=2),flush=True)


def record_progress():
    """仅登记已经完成的本轮发现；保留旧失败、暂停项及交易权限。"""
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    checks=json.loads((OUT/"calculation_checks.json").read_text(encoding="utf-8"))
    events=json.loads((OUT/"三个归档节点510300剩余收益.json").read_text(encoding="utf-8"))
    residuals=[]
    for r in events:
        recomputed=(r["sell_price"]-r["buy_price"])*r["shares"]-r["commissions_cny"]+r["dividend_entitlement_cny"]
        residuals.append(abs(recomputed-r["net_pnl_cny"]))
        assert r["paid_cny"]<=100000 and r["shares"]%100==0 and r["exit_date"]>r["entry_date"]
        assert abs(r["net_pnl_cny"]/r["paid_cny"]-r["net_return"])<1e-12
    assert max(residuals)<0.01
    checks["saved_event_pnl_recomputation_max_abs_cny"]=max(residuals)
    checks["budget_lots_and_next_day_exit_checks"]=True
    checks["figure_visually_reviewed"]=True
    save("calculation_checks.json",checks)
    prefix="reports/research/510300_historical_consumer_earnings_transmission_v1/"
    stamp=now()
    config_paths=[ROOT/"config/510300_historical_cause_discovery_v1.json",ROOT/"config/510300_existing_data_training_mandate_v1.json"]
    receipt={"recorded_at":stamp,"study_id":STUDY,"research_mode":"HISTORICAL_ONLY",
             "user_authority":"不做前瞻性，只做历史检验和发现；减少繁琐检验程序，先找到确定性和规律。",
             "result":prefix+"result.json","goal_service_status_observed":"active","goal_achieved":False,
             "changes":[],"old_rejections_retained":True,"orders_authorized":False}
    for path in config_paths:
        cfg=json.loads(path.read_text(encoding="utf-8"))
        before=dict(cfg)
        if path.name=="510300_historical_cause_discovery_v1.json":
            cfg.update(latest_completed_study=prefix+"result.json",latest_report=prefix+result["report"],updated_at=stamp)
        else:
            cfg.update(current_round=STUDY,latest_progress_receipt=prefix+"result.json",
                latest_continuation_report=prefix+result["report"],latest_historical_report=prefix+result["report"],
                latest_historical_consumer_earnings_transmission=prefix+"result.json",
                latest_continuation_classification=result["classification"],current_driver_continuation_classification=result["classification"],
                current_driver_consecutive_blocked_goal_turns=0,latest_historical_diagnostic_at=stamp,
                latest_goal_service_status="active",latest_goal_service_status_observed_at=stamp,
                goal_status="active",goal_achieved=False,local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                last_research_result="完成2018年四个已见消费权重案例的53条原表数列拆解：核对高基数、税费列报、金融现金流与票据支付；洋河实际利润处于原公司预告区间。归档后三个重叠ETF节点的5日净收益为3.42%、5.64%、1.61%，20日为-1.13%、2.22%、-0.02%。未形成达标账户，不按这些结果新设规则。",
                last_source_result="取得7份公司原始财报PDF，其中洋河半年报为公司原文的公开镜像；核对53条原表数列。2024年持有研究复用既有结果。首次公开精确时间及市场一致预期仍未核实。",
                next_research_question=result["next_historical_question"])
        receipt["changes"].append({"path":str(path),"fields":{k:{"before":before.get(k),"after":v} for k,v in cfg.items() if before.get(k)!=v}})
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save("authority_update.json",receipt)
    print("已登记本轮历史发现；六条保存收益核回，目标仍为进行中且未达标。",flush=True)


def main():
    parser = argparse.ArgumentParser(description="历史消费权重盈利传导")
    parser.add_argument("mode", choices=["prepare", "fetch", "analyze", "record-progress"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "fetch":
        fetch()
    elif args.mode == "analyze":
        analyze()
    else:
        record_progress()


if __name__ == "__main__":
    main()
