"""逐页核对固定十份中报的数值与传导说明，不把公司自述升级为因果估计。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_external_bridge_v16"
UNITS = {"元": 1, "万元": 10000, "人民币百万元": 1000000, "%": None}
SPECS = [
    ("2020-08", "600519.SH", "600519_2020H1.pdf", "元", [
        ("营业收入", 5, "43,952,804,663.50", "39,487,788,339.74", 11.31),
        ("归母净利润", 5, "22,601,655,085.51", "19,951,025,609.22", 13.29),
        ("经营现金净额", 5, "12,620,963,440.03", "24,086,920,146.43", -47.60)]),
    ("2020-08", "601318.SH", "601318_2020H1.pdf", "人民币百万元", [
        ("营业收入", 10, "629,956", "639,155", -1.4),
        ("归母净利润", 10, "68,683", "97,676", -29.7),
        ("经营现金净额", 10, "130,740", "181,853", -28.1)]),
    ("2020-08", "000858.SZ", "000858_2020H1.pdf", "元", [
        ("营业收入", 5, "30,767,525,988.40", "27,151,044,852.75", 13.32),
        ("归母净利润", 6, "10,855,391,967.86", "9,335,637,379.26", 16.28),
        ("经营现金净额", 6, "1,168,132,703.99", "8,365,812,629.04", -86.04)]),
    ("2020-08", "600036.SH", "600036_2020H1_巨潮.pdf", "人民币百万元", [
        ("营业收入", 7, "148,353", "138,301", 7.27),
        ("归母净利润", 7, "49,788", "50,612", -1.63),
        ("经营现金净额", 7, "254,254", "-55,023", None)]),
    ("2020-08", "600276.SH", "600276_2020H1.pdf", "元", [
        ("营业收入", 5, "11,308,881,923.08", "10,026,300,034.60", 12.79),
        ("归母净利润", 5, "2,661,917,189.96", "2,412,461,799.21", 10.34),
        ("经营现金净额", 5, "3,376,609,108.89", "1,452,490,116.70", 132.47)]),
    ("2022-08", "600519.SH", "600519_2022H1_巨潮.pdf", "元", [
        ("营业收入", 5, "57,616,866,647.29", "49,087,277,796.01", 17.38),
        ("归母净利润", 5, "29,793,908,077.83", "24,653,985,551.58", 20.85),
        ("经营现金净额", 5, "-11,163,941.30", "21,719,470,732.97", None)]),
    ("2022-08", "300750.SZ", "300750_2022H1.pdf", "万元", [
        ("营业收入", 7, "11,297,125.79", "4,407,456.06", 156.32),
        ("归母净利润", 7, "816,803.46", "448,378.76", 82.17),
        ("经营现金净额", 7, "1,868,234.30", "2,574,205.97", -27.42)]),
    ("2022-08", "601318.SH", "601318_2022H1.pdf", "人民币百万元", [
        ("营业收入", 10, "612,102", "635,649", -3.7),
        ("归母净利润", 10, "60,273", "58,005", 3.9),
        ("经营现金净额", 10, "318,100", "19,466", 1534.1)]),
    ("2022-08", "600036.SH", "600036_2022H1.pdf", "人民币百万元", [
        ("营业收入", 7, "179,091", "168,749", 6.13),
        ("归母净利润", 7, "69,420", "61,150", 13.52),
        ("经营现金净额", 7, "130,624", "6,322", 1966.18)]),
    ("2022-08", "000858.SZ", "000858_2022H1.pdf", "元", [
        ("营业收入", 7, "41,222,377,583.11", "36,751,547,826.70", 12.17),
        ("归母净利润", 7, "15,098,936,573.76", "13,200,371,746.73", 14.38),
        ("经营现金净额", 7, "1,887,074,640.92", "8,707,487,982.81", -78.33)]),
]

# 原数和原页同时保存；分母与比较期间单列，避免把季度变化当同比。
EXTRA = [
    ("600519_2020H1.pdf", 7, "主营业务境内收入", "4,281,023.63", "万元", "当期"),
    ("600519_2020H1.pdf", 7, "主营业务国外收入", "110,140.78", "万元", "当期"),
    ("600519_2022H1_巨潮.pdf", 8, "主营业务境内收入", "5,549,714.28", "万元", "当期"),
    ("600519_2022H1_巨潮.pdf", 8, "主营业务国外收入", "206,610.61", "万元", "当期"),
    ("600036_2020H1_巨潮.pdf", 9, "净利息收益率", "2.50", "%", "2020H1"),
    ("600036_2020H1_巨潮.pdf", 9, "净利息收益率", "2.70", "%", "2019H1"),
    ("600036_2022H1.pdf", 9, "净利息收益率", "2.44", "%", "2022H1"),
    ("600036_2022H1.pdf", 9, "净利息收益率", "2.49", "%", "2021H1"),
    ("600036_2022H1.pdf", 15, "净利息收益率", "2.37", "%", "2022Q2"),
    ("600036_2022H1.pdf", 15, "净利息收益率", "2.51", "%", "2022Q1"),
    ("600036_2022H1.pdf", 9, "不良贷款率", "0.95", "%", "2022-06-30"),
    ("600036_2022H1.pdf", 9, "不良贷款率", "0.91", "%", "2021-12-31"),
    ("601318_2020H1.pdf", 10, "扣非归母利润", "68,879", "人民币百万元", "2020H1"),
    ("601318_2020H1.pdf", 10, "扣非归母利润", "87,446", "人民币百万元", "2019H1"),
    ("601318_2020H1.pdf", 10, "税收政策引起的一次性损益调整", "10,453", "人民币百万元", "2019H1"),
    ("601318_2020H1.pdf", 26, "新业务价值", "31,031", "人民币百万元", "2020H1"),
    ("601318_2020H1.pdf", 26, "新业务价值", "41,052", "人民币百万元", "2019H1"),
    ("601318_2022H1.pdf", 26, "新业务价值", "19,573", "人民币百万元", "2022H1"),
    ("601318_2022H1.pdf", 26, "新业务价值", "27,387", "人民币百万元", "2021H1原口径"),
    ("601318_2022H1.pdf", 26, "可比假设下新业务价值降幅", "20.3", "%", "2021年底假设及方法重列比较"),
    ("601318_2022H1.pdf", 43, "年化净投资收益率", "3.9", "%", "2022H1"),
    ("601318_2022H1.pdf", 43, "年化净投资收益率", "3.8", "%", "2021H1"),
    ("601318_2022H1.pdf", 43, "年化总投资收益率", "3.1", "%", "2022H1"),
    ("601318_2022H1.pdf", 43, "年化总投资收益率", "3.5", "%", "2021H1"),
    ("601318_2022H1.pdf", 68, "集团内含价值", "1,441,261", "人民币百万元", "基准假设"),
    ("601318_2022H1.pdf", 68, "集团内含价值", "1,502,910", "人民币百万元", "投资收益率和风险贴现率同时增加50bp"),
    ("601318_2022H1.pdf", 68, "集团内含价值", "1,373,870", "人民币百万元", "投资收益率和风险贴现率同时减少50bp"),
    ("300750_2022H1.pdf", 14, "境外电池业务收入", "2,225,441.18", "万元", "2022H1非全部海外业务"),
    ("300750_2022H1.pdf", 14, "整体毛利率", "18.68", "%", "2022H1"),
    ("300750_2022H1.pdf", 14, "整体毛利率同比降幅", "8.58", "%", "个百分点"),
    ("300750_2022H1.pdf", 13, "动力电池毛利率", "15.04", "%", "2022H1"),
    ("300750_2022H1.pdf", 13, "动力电池毛利率同比降幅", "7.96", "%", "个百分点"),
    ("300750_2022H1.pdf", 135, "计入财务费用的汇兑损益", "-6,248.72", "万元", "2022H1负值为收益"),
    ("300750_2022H1.pdf", 135, "计入财务费用的汇兑损益", "16,223.62", "万元", "2021H1正值为费用"),
    ("300750_2022H1.pdf", 140, "美元货币资金人民币折算", "2,661,496.71", "万元", "2022-06-30并非净敞口"),
    ("300750_2022H1.pdf", 141, "美元长期借款人民币折算", "1,017,448.24", "万元", "2022-06-30并非全部美元负债"),
    ("300750_2022H1.pdf", 141, "现金流量套期工具公允价值变动计入其他综合收益", "-63,296.03", "万元", "2022-06-30并非全部汇兑损益"),
]

QUAL = [
    ("600519_2020H1.pdf", 7, "经营现金下降说明", ["客户存款和同业存放款项净增加额减少"], "经营现金净额下降含财务公司客户存款及同业资金影响，不宜直接等同白酒回款恶化。"),
    ("600519_2022H1_巨潮.pdf", 7, "经营现金下降说明", ["客户存款和同业存放款项净增加额减少", "存放中央银行和同业款项净增加额增加"], "合并经营现金净额变化受到财务子公司存放与吸收资金变化影响，不能直接解释成酒类经营亏损。"),
    ("600519_2022H1_巨潮.pdf", 80, "直接浮息负债", ["无以浮动利率计息的负债"], "报告期末没有浮息负债；这不排除股票折现率、需求和外币净投资风险。"),
    ("000858_2020H1.pdf", 10, "经营现金下降说明", ["本年春节较早", "上交递延税金"], "公司把回款跨年和税金支付列为经营现金减少原因。"),
    ("000858_2022H1.pdf", 7, "经营现金下降说明", ["降低预收款中现金收取比例", "减少经销商资金压力", "银行承兑汇票到期收现额度较高"], "公司调整预收现金比例及订单管理以缓解经销商资金压力，并存在上年票据到期收现基数影响。"),
    ("000858_2022H1.pdf", 14, "地区分类边界", ["公司未直接出口酒类产品", "客户注册地"], "酒类经进出口公司出口，地区按客户注册地划分；不能把国内地区表当作最终海外需求占比为零。"),
    ("600036_2022H1.pdf", 36, "存款与息差传导", ["企业资金活化不足", "对公活期存款增长受限", "居民投资向定期储蓄转化", "融资需求不足"], "管理层说明企业结算活期增长受限、居民资金定期化；贷款需求、结构及定价与存款成本共同影响息差。这是单家银行证据，不是全国M1变化的归因金额。"),
    ("601318_2022H1.pdf", 26, "新业务价值比较口径", ["20.3%", "去年年底假设及方法重述"], "原列报数同比下降28.5%；按2021年末假设和方法重列的同比下降20.3%，两者不能混用。"),
    ("601318_2022H1.pdf", 68, "利率情景边界", ["投资收益率和风险贴现率每年增加50个基点"], "敏感性同时改变投资回报与风险贴现假设，不是美债利率单独冲击，更不是市场股价弹性。"),
    ("601318_2020H1.pdf", 10, "利润同比基数", ["一次性调整", "10,453"], "2019H1包含税收政策引起的一次性损益调整；2020H1扣非利润仍下降21.2%，并非净利润降幅全由基数解释。"),
    ("300750_2022H1.pdf", 13, "收入与成本", ["部分上游材料价", "格上涨造成成本增加"], "收入扩张同时伴随上游成本上涨；利润增长不等于单位利润率改善。"),
    ("300750_2022H1.pdf", 141, "外汇套期", ["远期结售汇合约", "部分预期采购交易支出"], "外币资金、借款和部分预期采购套期同时存在，仅凭汇率方向或一个账户不能判定综合收益。"),
    ("300750_2022H1.pdf", 147, "利率风险", ["长期银行借款及应付债券", "目前并未采取利率对冲政策"], "公司披露长期有息负债的利率风险；不能把美债收益率变化直接施加到全部人民币债务。"),
    ("600276_2020H1.pdf", 92, "直接借款风险", ["本集团无借款"], "公司表示没有借款；直接借款成本通道与股票折现、需求通道应分别讨论。"),
]


def compact(text):
    return "".join(text.split()).replace("，", ",").replace("−", "-")


def value(token):
    return float(token.replace(",", ""))


def main():
    if (OUT / "results/company_facts_receipt.json").exists():
        raise RuntimeError("公司事实表已冻结，不重复覆盖。")
    receipts = {x["name"]: x for x in json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8")) if x["status"].startswith("SAVED")}
    texts = {name: json.loads((OUT / "sources" / name).with_suffix(".pages.json").read_text(encoding="utf-8")) for name in receipts}
    top = pd.read_csv(OUT / "results/事前前五名_收益与风险贡献.csv").set_index(["stat_month", "symbol"])
    cases = pd.read_csv(OUT / "results/两个病例_完整篮子与510300同口径比较.csv").set_index("stat_month")
    mapping = {x[2]: (x[0], x[1]) for x in SPECS}

    def provenance(name, page, tokens):
        receipt = receipts[name]
        assert sha256((OUT / "sources" / name).read_bytes()).hexdigest() == receipt["sha256"]
        text = compact(texts[name][page - 1]["text"])
        for token in tokens:
            assert compact(token) in text, (name, page, token)
        case, symbol = mapping[name]
        assert receipt["published_date"] < cases.loc[case, "origin_date"]
        return {"stat_month": case, "symbol": symbol, "name": top.loc[(case, symbol), "name"],
                "document": name, "pdf_page": page, "published_date": receipt["published_date"],
                "available_at_upper_bound": receipt["published_date"] + "T23:59:59+08:00",
                "source_url": receipt["url"], "source_sha256": receipt["sha256"],
                "first_vintage_authenticated": False, "used_as": "起点前已披露经营背景，非起点新消息"}

    financial, scalars, statements = [], [], []
    for case, symbol, doc, unit, data in SPECS:
        for field, page, current_token, prior_token, published_yoy in data:
            row = provenance(doc, page, [current_token, prior_token])
            current, prior = value(current_token), value(prior_token)
            yoy = (current / prior - 1) * 100 if published_yoy is not None else None
            if yoy is not None:
                tolerance = .051 if symbol == "601318.SH" else .011
                assert abs(yoy - published_yoy) < tolerance, (doc, field, yoy, published_yoy)
            row.update({"field": field, "source_unit": unit, "current_raw_token": current_token, "prior_raw_token": prior_token,
                        "current_value": current, "prior_value": prior, "current_cny": current * UNITS[unit], "prior_cny": prior * UNITS[unit],
                        "comparison": "本年上半年与上年上半年", "reported_yoy_percent": published_yoy, "recomputed_yoy_percent": yoy,
                        "growth_status": "RECOMPUTED" if yoy is not None else "SOURCE_NOT_APPLICABLE",
                        "cashflow_comparable_to_industrial_sales": False if symbol in ["601318.SH", "600036.SH", "600519.SH"] else True})
            financial.append(row)
    for doc, page, field, token, unit, period in EXTRA:
        row = provenance(doc, page, [token])
        row.update({"field": field, "period_or_scenario": period, "source_unit": unit, "raw_token": token, "value": value(token),
                    "value_cny": value(token) * UNITS[unit] if UNITS[unit] is not None else None})
        scalars.append(row)
    for doc, page, field, tokens, interpretation in QUAL:
        row = provenance(doc, page, tokens)
        row.update({"field": field, "source_check_phrases": "；".join(tokens), "interpretation": interpretation,
                    "evidence_type": "公司原报告及管理层说明", "causal_effect_identified": False})
        statements.append(row)
    f = pd.DataFrame(financial)
    merged = top.reset_index()
    for field, column in [("营业收入", "revenue_yoy_percent"), ("归母净利润", "net_profit_yoy_percent"), ("经营现金净额", "cashflow_yoy_percent")]:
        part = f[f.field == field][["stat_month", "symbol", "reported_yoy_percent", "current_cny", "prior_cny"]]
        part = part.rename(columns={"reported_yoy_percent": column, "current_cny": column + "_current_cny", "prior_cny": column + "_prior_cny"})
        merged = merged.merge(part, on=["stat_month", "symbol"], validate="1:1")
    s = pd.DataFrame(scalars)
    def scalar(doc, field, period=None):
        a = s[(s.document == doc) & (s.field == field)]
        if period is not None:
            a = a[a.period_or_scenario == period]
        assert len(a) == 1
        return float(a.iloc[0].value)
    derived = []
    for doc in ["600519_2020H1.pdf", "600519_2022H1_巨潮.pdf"]:
        domestic = scalar(doc, "主营业务境内收入")
        foreign = scalar(doc, "主营业务国外收入")
        derived.append({"document": doc, "metric": "国外收入占主营业务收入百分比", "value": foreign / (foreign + domestic) * 100,
                        "numerator": foreign, "denominator": foreign + domestic, "boundary": "主营业务口径，非合并营业收入；亦非净汇率敞口"})
    revenue = f[(f.document == "300750_2022H1.pdf") & (f.field == "营业收入")].iloc[0].current_value
    foreign = scalar("300750_2022H1.pdf", "境外电池业务收入")
    derived.append({"document": "300750_2022H1.pdf", "metric": "境外电池业务占合并营业收入百分比", "value": foreign / revenue * 100,
                    "numerator": foreign, "denominator": revenue, "boundary": "只含表中境外电池业务；不称全部海外收入比例"})
    for case in ["2020-08", "2022-08"]:
        a = merged[merged.stat_month == case]
        derived.append({"document": "固定前五名全部", "metric": case + "利润增长且后续下跌公司数", "value": int(((a.net_profit_yoy_percent > 0) & (a.stock_return20_cc < 0)).sum()),
                        "numerator": None, "denominator": 5, "boundary": "事前前五名描述；非所有成分盈利，非盈利预测修订，也无独立检验"})
    assert len(financial) == 30 and len(merged) == 10 and len(scalars) == 37 and len(statements) == 14
    for name, frame in [("十份中报_30项财务成对原数.csv", f), ("公司传导_37项原始数值.csv", s),
                        ("公司传导_14项原文定位与解释.csv", pd.DataFrame(statements)), ("事前前五名_经营背景与后续收益.csv", merged),
                        ("公司数据_有明确分母的推导.csv", pd.DataFrame(derived))]:
        frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.12g")
    receipt = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "SOURCE_PAGE_TOKENS_AND_REPORTED_GROWTH_CHECKED",
               "company_documents": 10, "financial_pairs": len(financial), "scalar_facts": len(scalars), "qualitative_source_locators": len(statements),
               "cohort_selected_from_pre_origin_weights": True, "future_reports_used": 0,
               "all_source_dates_before_origin": True, "first_vintage_authenticated": False,
               "goal_achieved": False, "independent_validation": False}
    (OUT / "results/company_facts_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("公司事实表完成：10份中报、30项成对财务数字、37项专项数值、14项原因说明。")
    print(merged[["stat_month", "name", "net_profit_yoy_percent", "stock_return20_cc"]].to_string(index=False))
    print(pd.DataFrame(derived).to_string(index=False))


if __name__ == "__main__":
    main()
