"""以指数样本集合比较历史收入、利润与宏观工业统计，不生成交易信号。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.macro_earnings_pricing_bridge_v5 import clean

OUT = ROOT / "reports/research/510300_historical_index_earnings_population_v1"
OLD = ROOT / "reports/research/510300_macro_earnings_pricing_bridge_v5"
FIELDS = ["total_operating_revenue", "operating_profit", "parent_net_profit"]
PERIODS = [
    {"period": "2022Q1", "report_end": "2022-03-31", "cutoff": "2022-04-30"},
    {"period": "2022H1", "report_end": "2022-06-30", "cutoff": "2022-08-31"},
    {"period": "2022Q1Q3", "report_end": "2022-09-30", "cutoff": "2022-10-31"},
    {"period": "2022Y", "report_end": "2022-12-31", "cutoff": "2023-04-30"},
    {"period": "2023Q1", "report_end": "2023-03-31", "cutoff": "2023-04-30"},
]
SOURCES = [
    {"id": "nbs_2022h1", "url": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901534.html", "expect": "2022年1—6月份全国规模以上工业企业利润增长1.0%"},
    {"id": "nbs_2022y", "url": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901735.html", "expect": "2022年全国规模以上工业企业利润下降4.0%"},
    {"id": "nbs_2023q1", "url": "https://www.stats.gov.cn/sj/zxfb/202304/t20230427_1939091.html", "expect": "2023年1—3月份全国规模以上工业企业利润下降21.4%"},
    {"id": "nbs_2023janfeb_cause", "url": "https://www.stats.gov.cn/sj/sjjd/202303/t20230327_1937984.html", "expect": "下拉工业利润18.6个百分点"},
    {"id": "sse_2022q1", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20220430_5701776.shtml", "expect": "2022年第一季度"},
    {"id": "sse_2022h1", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20220831_5707932.shtml", "expect": "2022年上半年"},
    {"id": "sse_2022q3", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20221031_5710966.shtml", "expect": "2022年前三季度"},
    {"id": "sse_2022y_2023q1", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20230429_5720867.shtml", "expect": "2023年第一季度"},
]


def now():
    return datetime.now().astimezone().isoformat()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "study_id": "510300_HISTORICAL_INDEX_EARNINGS_POPULATION_V1", "created_at": now(),
        "question": "2022至2023年初的经营恢复，能否映射到沪深300样本总体名义收入和利润；工业统计、金融和非金融为何不同？",
        "periods": PERIODS, "unit": "指数当时样本集合及全部行业，不筛选具体公司病例",
        "observation": "每个固定截止日之后首个已有交易日；当日300个历史样本，本期和同比期用相同集合。只接入名义公告日早于观察日的字段。",
        "comparator_population": "全部五个观察日成分名单的交集也全量汇总，用于描述换样差异；不挑结果、不选择其中之一作为策略。",
        "fields": FIELDS, "period_measure": "累计报告期同比及同一集合下本期累计减前一季度累计的单季同比。第一季度不另减。缺一项保留缺失。",
        "aggregation": "全部、银行、非银金融、非金融、行业缺失及全部行业分别汇总；含亏损。组间利润增量除全部组上年同期利润得到增长贡献。不是指数价格权重贡献或EPS。",
        "margin_identity": "仅非金融同组计算归母净利/收入；对称分解：利润变化=收入变化×两期平均净利率+净利率变化×两期平均收入。是会计恒等式，不识别量价成本各自因果。",
        "vintage": "复用V5保存的东方财富当前值；保留晚更新数、原来源hash。只是有修订和会计口径风险的回顾重建，不能认证首次披露值，不进入历史交易信号。",
        "weights": "仅列最近早于观察日的旧参考权重和覆盖，不归一化，不用它给利润加权，不声称官方历史原版本。",
        "external_sources": "官方工业利润与沪市主板披露总结只是各自总体事实；不当作沪深300总体替代，不回填公告前。",
        "deduplication": {"V5": "已有2018至2026观察点的滚动利润与名义时钟重建；本轮只新增五个一致报告期和单季的集合比较。", "valuation_repricing": "2019P/PE隐含分母与真实盈利已被区分；不重复声称P/PE为EPS。", "PMI": "上一轮两个候选均REJECTED_FROZEN，本轮不改规则、不加筛选。"},
        "new_accounts": 0, "new_models": 0, "orders": 0,
        "future_forecasts": False, "independent_validation": False, "goal_achieved": False,
        "stop": "没有一致首次版本及预期差证据时，只给机制发现；不泛搜公司财报，不为此调整旧PMI账户。",
    }
    if (OUT / "protocol.json").exists():
        raise RuntimeError("既定方案已存在，不覆盖。")
    save(OUT / "protocol.json", protocol)
    inputs = [OLD / "inputs" / n for n in ["income.parquet", "membership.parquet", "monthly_state.parquet", "weights.parquet"]]
    save(OUT / "input_receipts.json", [{"path": rel(p), "sha256": digest(p), "bytes": p.stat().st_size} for p in inputs])
    save(OUT / "source_plan.json", SOURCES)
    save(OUT / "freeze.json", {"protocol_sha256": digest(OUT / "protocol.json"), "created_at": now()})
    print("已保存五个固定报告期的指数总体研究方案。")


def fetch_one(item):
    folder = OUT / "sources"
    folder.mkdir(exist_ok=True)
    try:
        response = requests.get(item["url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        path = folder / (item["id"] + ".html")
        path.write_bytes(response.content)
        content = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)
        text_path = folder / (item["id"] + ".txt")
        text_path.write_text(content, encoding="utf-8")
        if "".join(item["expect"].split()) not in "".join(content.split()):
            raise ValueError("原文校对词未出现，保留原件，不接纳数值。")
        return {**item, "status": "OK", "retrieved_at": now(), "final_url": response.url, "path": rel(path), "text_path": rel(text_path), "sha256": digest(path)}
    except (requests.RequestException, ValueError) as error:
        return {**item, "status": "FAILED", "retrieved_at": now(), "error": str(error)}


def collect():
    plan = read(OUT / "source_plan.json")
    additional = OUT / "additional_source_plan.json"
    if additional.exists():
        plan += read(additional)["sources"]
    assert len({x["id"] for x in plan}) == len(plan)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch_one, plan))
    save(OUT / "source_manifest.json", rows)
    print(json.dumps({"原文取得": sum(x["status"] == "OK" for x in rows), "总数": len(rows)}, ensure_ascii=False))


def review_sources():
    rows = read(OUT / "source_manifest.json")
    for row in rows:
        path = OUT / "sources" / (row["id"] + ".html")
        if not path.exists():
            continue
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        text = soup.get_text(" ", strip=True)
        expected = "".join(row["expect"].split())
        if expected not in "".join(text.split()):
            continue
        if row["status"] != "OK":
            row["prior_validation"] = {"status": row["status"], "error": row.get("error")}
            row.update(status="OK", validation_correction="原网页数字之间含排版空白；忽略空白后核对原词，未重取原件。", path=rel(path), text_path=rel(path.with_suffix(".txt")), sha256=digest(path))
            row.pop("error", None)
        body = soup.select_one(".TRS_Editor") or soup.select_one(".article") or soup.select_one(".allZoom")
        if body:
            path.with_name(row["id"] + "_body.txt").write_text(body.get_text(" ", strip=True), encoding="utf-8")
        if row["id"].startswith("nbs_"):
            match = re.search(r"202[023]/\d\d/\d\d(?:\s+\d\d:\d\d)?", text)
        else:
            match = re.search(r"202[023]-\d\d-\d\d", text)
        row["displayed_publication"] = match.group() if match else None
        if row["id"] == "mof_insurance_standard_2020":
            metadata = soup.find("meta", attrs={"name": "PubDate"})
            if metadata:
                row["displayed_publication"] = metadata["content"]
                row["publication_precision"] = "SECOND_METADATA"
                row["known_at_upper_bound_cn"] = pd.Timestamp(metadata["content"]).tz_localize("Asia/Shanghai").isoformat()
                row["document_date"] = "2020-12-19"
                row["effective_date_specified_population"] = "2023-01-01"
        if row["id"] == "nbs_2022q3":
            row["document_date"] = "2022-10-27"
            row["publication_precision"] = "UNRESOLVED_DOCUMENT_DATE_ONLY"
            row["known_at_upper_bound_cn"] = None
        if match:
            stamp = pd.Timestamp(match.group())
            row["publication_precision"] = "MINUTE" if ":" in match.group() else "DAY"
            if row["publication_precision"] == "DAY":
                stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            row["known_at_upper_bound_cn"] = stamp.tz_localize("Asia/Shanghai").isoformat()
        row["clock_use"] = "仅解释资料；不构成新交易信号，后发布的总结不回填较早月份。"
    save(OUT / "source_manifest.json", rows)
    print("已按原网页确认排版空白与公开时间。")


def source_facts():
    review_sources()
    manifest = {x["id"]: x for x in read(OUT / "source_manifest.json")}
    macro = []
    for period, key in [("2022Q1", "nbs_2022q1"), ("2022H1", "nbs_2022h1"), ("2022Q1Q3", "nbs_2022q3"), ("2022Y", "nbs_2022y"), ("2023Q1", "nbs_2023q1")]:
        source = manifest[key]
        assert source["status"] == "OK"
        soup = BeautifulSoup((ROOT / source["path"]).read_bytes(), "html.parser")
        body = soup.select_one(".TRS_Editor")
        selected = None
        for table in body.find_all("table"):
            parsed = pd.read_html(StringIO(str(table)), header=None)[0]
            if parsed.shape[1] == 7 and str(parsed.iat[0, 1]).startswith("营业收入"):
                selected = parsed
                break
        assert selected is not None
        parsed_rows = 0
        for _, values in selected.iterrows():
            name = "".join(str(values.iloc[0]).split()).replace("其中：", "")
            if name not in ["总计", "采矿业", "制造业", "电力、热力、燃气及水生产和供应业"]:
                continue
            nums = [float(str(v).replace(",", "")) for v in values.iloc[1:]]
            r, rg, c, cg, p, pg = nums
            macro.append({"period": period, "group": name, "revenue_100m_cny": r, "revenue_yoy_pct": rg, "cost_100m_cny": c, "cost_yoy_pct": cg, "pretax_profit_100m_cny": p, "pretax_profit_yoy_pct": pg, "calculated_pretax_margin_pct": p / r * 100, "source_id": key, "url": source["url"], "displayed_publication": source["displayed_publication"], "source_sha256": source["sha256"], "scope": "规上工业法人单位；税前利润与可比统计口径，不是指数利润"})
            parsed_rows += 1
        assert parsed_rows == 4
    save(OUT / "official_industrial_facts.json", macro)
    sse = [
        {"period": "2022Q1", "source_id": "sse_2022q1", "revenue_100m_cny": 120300, "revenue_yoy_pct": 12, "net_profit_100m_cny": 11300, "net_profit_yoy_pct": 8, "anchor": "2022年第一季度，沪市主板公司实现营业收入12.03万亿元，净利润1.13万亿元，扣非后净利润1.08万亿元，同比增幅分别为12%、8%、10%"},
        {"period": "2022H1", "source_id": "sse_2022h1", "revenue_100m_cny": 247700, "revenue_yoy_pct": 9, "net_profit_100m_cny": 23300, "net_profit_yoy_pct": 6, "anchor": "2022年上半年，沪市主板公司合计实现营业收入24.77万亿元，净利润2.33万亿元，扣非后净利润2.24万亿元，同比增长9%、6%和8%"},
        {"period": "2022Q1Q3", "source_id": "sse_2022q3", "revenue_100m_cny": 372300, "revenue_yoy_pct": 8, "net_profit_100m_cny": 33900, "net_profit_yoy_pct": 5, "anchor": "2022年前三季度，沪市主板公司合计实现营业收入37.23万亿元、净利润3.39万亿元、扣非后净利润3.27万亿元，同比分别增长8%、5%和6%"},
        {"period": "2022Y", "source_id": "sse_2022y_2023q1", "revenue_100m_cny": 505500, "revenue_yoy_pct": 6, "net_profit_100m_cny": 41600, "net_profit_yoy_pct": 2, "anchor": "2022年，沪市主板公司合计实现营业收入50.55万亿元，同比增长6%；净利润4.16万亿元，同比增长2%"},
        {"period": "2023Q1", "source_id": "sse_2022y_2023q1", "revenue_100m_cny": 121600, "revenue_yoy_pct": 4, "net_profit_100m_cny": 12000, "net_profit_yoy_pct": 5, "anchor": "沪市主板公司第一季度合计实现营业收入12.16万亿元，净利润1.20万亿元，同比分别增长4%、5%"},
    ]
    for fact in sse:
        source = manifest[fact["source_id"]]
        content = (ROOT / source["text_path"]).read_text(encoding="utf-8")
        assert "".join(fact["anchor"].split()) in "".join(content.split()), fact["period"]
        fact.update(url=source["url"], source_sha256=source["sha256"], displayed_publication=source["displayed_publication"], scope="沪市主板披露总体，不是沪深300；摘要舍入数与两期样本不应直接连乘")
    save(OUT / "official_sse_facts.json", sse)
    explanations = [
        {"id": "demand_price_cost", "source_id": "nbs_2023janfeb_cause", "date": "2023-03-27", "fact": "1至2月规上工业收入下降1.3%，PPI同比下降1.1%；收入比成本降得快，毛利下降对利润增速拖累18.6个百分点。", "limit": "统计局工业总体的解释，不是沪深300的因果贡献；价格、收入与毛利存在重叠，不能把这几项相加。"},
        {"id": "insurance_accounting", "source_id": "mof_insurance_standard_2020", "date": "2020-12-24", "fact": "规定境内外同时上市及所述境外上市企业自2023年1月1日起执行新保险合同准则。", "limit": "仅确认统一制度变化；未逐一核实聚合器两期值采用何种重述，不能把非银利润增幅全归为经营改善或全归为准则。"},
    ]
    for fact in explanations:
        fact["url"] = manifest[fact["source_id"]]["url"]
    save(OUT / "explanatory_facts.json", explanations)
    print("已提取五期工业与沪市主板总体事实，保留各自统计范围。")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
    a = pd.read_parquet(OUT / "aggregates.parquet")
    d = a[(a.population == "DYNAMIC_300") & (a.measure == "YTD")]
    periods = [x["period"] for x in PERIODS]
    labels = ["2022Q1", "2022H1", "2022前三季", "2022全年", "2023Q1"]
    x = np.arange(5)
    industrial = pd.DataFrame(read(OUT / "official_industrial_facts.json"))
    total = industrial[industrial.group.eq("总计")].set_index("period")
    sse = pd.DataFrame(read(OUT / "official_sse_facts.json")).set_index("period")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8))
    colors = ["#1c5573", "#bd652c", "#347c5c"]
    ax = axes[0, 0]
    for offset, values, label, color in [(-.25, total.loc[periods].pretax_profit_yoy_pct, "规上工业利润总额", colors[0]), (0, sse.loc[periods].net_profit_yoy_pct, "沪市主板净利润", colors[1]), (.25, d[d.group.eq("全部")].set_index("period").loc[periods].parent_net_profit_yoy*100, "指数样本归母利润（回顾重建）", colors[2])]:
        ax.bar(x+offset, values, .24, label=label, color=color)
    ax.set_title("统计范围不同，利润方向可以相反", loc="left", fontweight="bold")
    ax.set_ylabel("累计利润同比 / %")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[0, 1]
    q = d[(d.period == "2023Q1") & (d.group_type == "金融分层")].set_index("group").loc[["银行", "非银金融", "非金融"]]
    bars = ax.bar(q.index, q.profit_growth_contribution_pp, color=colors)
    for bar, value in zip(bars, q.profit_growth_contribution_pp):
        ax.text(bar.get_x()+bar.get_width()/2, value+.10, f"{value:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 5.8)
    ax.set_title("2023Q1：合计增速6.51%的行业来源", loc="left", fontweight="bold")
    ax.set_ylabel("对样本归母利润增速的贡献 / 百分点")
    ax.text(.02,.93,"利润金额贡献；不等于指数涨幅贡献",transform=ax.transAxes,fontsize=9)
    ax = axes[1, 0]
    q = d[d.group.eq("非金融")].set_index("period").loc[periods]
    ax.plot(x, q.total_operating_revenue_yoy*100, "o-", label="收入同比", color=colors[0])
    ax.plot(x, q.parent_net_profit_yoy*100, "s-", label="归母利润同比", color=colors[1])
    ax.set_xticks(x, labels)
    ax.set_ylabel("累计同比 / %")
    ax.set_title("非金融：利润增速持续低于收入增速", loc="left", fontweight="bold")
    ax.legend()
    ax = axes[1, 1]
    q = a[(a.measure == "QUARTER") & (a.group == "非金融")]
    for population, label, color in [("DYNAMIC_300", "各观察日指数样本", colors[0]), ("COMMON_MEMBERSHIP", "五个观察日共同成分（事后对照）", colors[1])]:
        values = q[q.population.eq(population)].set_index("period").loc[periods].parent_net_profit_yoy*100
        ax.plot(x, values, "o-", label=label, color=color)
        for j, value in enumerate(values):
            if population == "DYNAMIC_300":
                ax.annotate(f"{value:.1f}%", (j, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, ["2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1"])
    ax.set_ylabel("单季归母利润同比 / %")
    ax.set_title("单季转折：四季度走弱，一季度恢复增长", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes.flat:
        ax.axhline(0, color="#6b737b", lw=.7)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("从宏观经营恢复到指数盈利：先区分总体、报告期与行业结构", fontsize=17, fontweight="bold", y=.98)
    fig.text(.045,.025,"指数样本值含后续修订，会计口径未完成首版认证；图示为历史描述，不是交易信号。累计期互相重叠。\n官方工业、沪市主板和指数样本的覆盖与利润口径不同；指数财务观察日最晚为2023年5月4日。",fontsize=9,color="#505962")
    fig.tight_layout(rect=[.02,.075,.99,.94], h_pad=3)
    path=OUT / "指数盈利_总体行业与利润率.png"
    fig.savefig(path,dpi=170,facecolor="white")
    plt.close(fig)
    print("已生成总体、行业贡献与利润率比较图。")


def publish():
    a = pd.read_parquet(OUT / "aggregates.parquet")
    base = a[(a.population == "DYNAMIC_300") & (a.measure == "YTD")]
    industry = pd.DataFrame(read(OUT / "official_industrial_facts.json"))
    macro = industry[industry.group.eq("总计")].set_index("period")
    sse = pd.DataFrame(read(OUT / "official_sse_facts.json")).set_index("period")
    sources = {x["id"]: x for x in read(OUT / "source_manifest.json")}
    labels = {"2022Q1": "2022一季度", "2022H1": "2022上半年", "2022Q1Q3": "2022前三季度", "2022Y": "2022全年", "2023Q1": "2023一季度"}
    source_ids = {"2022Q1": "nbs_2022q1", "2022H1": "nbs_2022h1", "2022Q1Q3": "nbs_2022q3", "2022Y": "nbs_2022y", "2023Q1": "nbs_2023q1"}
    summary_rows, margin_rows, quarter_rows = [], [], []
    table = ["| 报告期 | 规上工业利润总额同比 | 沪市主板净利润同比 | 指数样本归母利润同比* | 指数非金融归母利润同比* |", "|---|---:|---:|---:|---:|"]
    margin_table = ["| 报告期 | 非金融收入同比 | 非金融归母利润同比 | 同组上年净利/收入 | 本期净利/收入 |", "|---|---:|---:|---:|---:|"]
    quarter_table = ["| 单季度 | 当时成分中的非金融利润同比 | 共同成分中的非金融利润同比 |", "|---|---:|---:|"]
    for spec, quarter in zip(PERIODS, ["2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1"]):
        period = spec["period"]
        q = base[(base.period == period) & (base.group_type != "行业")].set_index("group")
        total, nonfinancial = q.loc["全部"], q.loc["非金融"]
        summary_rows.append({"period": period, "industrial_pretax_yoy": macro.loc[period, "pretax_profit_yoy_pct"]/100, "sse_net_profit_yoy": sse.loc[period, "net_profit_yoy_pct"]/100, "component_parent_profit_yoy": total.parent_net_profit_yoy, "component_nonfinancial_profit_yoy": nonfinancial.parent_net_profit_yoy, "component_valid_members": int(total.valid_members), "late_updated_members": int(total.late_updated_members)})
        table.append(f"| {labels[period]} | {macro.loc[period, 'pretax_profit_yoy_pct']:+.1f}% | {sse.loc[period, 'net_profit_yoy_pct']:+.0f}% | {total.parent_net_profit_yoy:+.2%} | {nonfinancial.parent_net_profit_yoy:+.2%} |")
        margin_table.append(f"| {labels[period]} | {nonfinancial.total_operating_revenue_yoy:+.2%} | {nonfinancial.parent_net_profit_yoy:+.2%} | {nonfinancial.parent_net_margin_prior:.2%} | {nonfinancial.parent_net_margin_current:.2%} |")
        margin_rows.append(nonfinancial.to_dict())
        query = a[(a.period == period) & (a.measure == "QUARTER") & (a.group == "非金融")].set_index("population")
        dynamic = query.loc["DYNAMIC_300", "parent_net_profit_yoy"]
        common = query.loc["COMMON_MEMBERSHIP", "parent_net_profit_yoy"]
        quarter_rows.append({"period": quarter, "dynamic_nonfinancial_profit_yoy": dynamic, "common_nonfinancial_profit_yoy": common})
        quarter_table.append(f"| {quarter} | {dynamic:+.2%} | {common:+.2%} |")
    q1 = base[(base.period == "2023Q1") & (base.group_type != "行业")].set_index("group")
    total, bank, nonbank, nf = (q1.loc[name] for name in ["全部", "银行", "非银金融", "非金融"])
    contribution_share = nonbank.parent_net_profit_change/total.parent_net_profit_change
    bank_profit_share = bank.parent_net_profit_current/total.parent_net_profit_current
    layers = ["| 分层 | 同比利润变化/亿元 | 对全部利润增速贡献/百分点 | 自身归母利润同比 |", "|---|---:|---:|---:|"]
    for name in ["银行", "非银金融", "非金融"]:
        row = q1.loc[name]
        layers.append(f"| {name} | {row.parent_net_profit_change/1e8:+.2f} | {row.profit_growth_contribution_pp:+.2f} | {row.parent_net_profit_yoy:+.2%} |")
    findings = {
        "classification": "PROGRESS_FIXED_INDEX_POPULATIONS_AND_NOMINAL_PROFIT_TRANSMISSION",
        "summary": summary_rows, "nonfinancial_margin_identity": margin_rows, "single_quarter_comparison": quarter_rows,
        "2023q1_nonbank_share_of_profit_increase": contribution_share,
        "2023q1_bank_share_of_group_profit": bank_profit_share,
        "2023q1_bank_unversioned_reference_weight": bank.reference_weight_sum,
        "inference": "总量经营恢复与指数盈利改善的强度不同，行业利润变化也相互抵消；2023Q1回顾重建的正增长不能据此推定当时未反映的指数收益。",
        "causal_links_unidentified": ["数量、售价、各项成本对指数公司利润的精确贡献", "非银投资收益、保险经营与会计准则变化的各自贡献", "利润金额变化到自由流通加权指数价格的映射", "公告前全市场一致预期与已反映程度"],
        "trading_use": "NOT_ADMITTED_CURRENT_VALUE_AND_EXPECTATION_GAPS",
        "parent_candidates": {"PMI_POSITIVE_SURPRISE": "REJECTED_FROZEN", "PMI_SURPRISE_DEMAND_IMPROVEMENT": "REJECTED_FROZEN"},
    }
    save(OUT / "mechanism_findings.json", findings)
    def link(path, label):
        return f"[{label}](<{Path(path).as_posix()}>)"
    report_path = OUT / "历史发现_指数总体盈利与宏观传导.md"
    figure = OUT / "指数盈利_总体行业与利润率.png"
    official_sources = "、".join(f"[{labels[p]}]({sources[source_ids[p]]['url']})" for p in labels)
    sse_sources = "、".join(f"[{labels[p]}]({sse.loc[p, 'url']})" for p in ["2022Q1", "2022H1", "2022Q1Q3", "2022Y"])
    report = f"""# 历史发现：宏观经营恢复怎样映射到指数总体盈利

**本轮新增的结论是：宏观工业利润、指数样本合计利润和指数价格，对同一轮恢复可以表现出不同方向和幅度。** 2023年一季度，官方规上工业利润下降21.4%，沪市主板净利润增长5%；旧库回顾重建的沪深300样本归母利润约增6.51%，其中非银金融贡献4.47个百分点，非金融样本只增2.46%。总体增长并不等于所有权重行业共同恢复；利润金额占比也不等于指数价格权重。

研究围绕指数整体，固定覆盖2022年一季度、半年、前三季度、全年和2023年一季度，五期全部保留，另作同集合单季分解。本轮没有扩展个股案例、拟合阈值或新增账户。旧PMI两个失败规则维持原结论。

先说明数据身份：**下文带星号的指数样本财务，是V5旧库当前值的回顾重建，全部有效记录至少有一个依赖后来更新。** 它们适合定位历史传导问题，未认证当时首次披露和两期会计口径，不能成为交易输入。官方工业统计与沪市主板总结各自保留统计范围，也不冒充指数财务。

**经营总体与指数样本的差别有多大。**

{chr(10).join(table)}

工业数据为规模以上工业法人单位的利润总额；沪市主板是交易所披露总体；指数列为当时历史成分公司集团归母利润的金额汇总，含亏损。三列的税前/税后、公司范围和样本处理不同，因此表格用于比较，不能直接相减称为某项冲击的效应。工业同比取各次公告的可比口径，不拿去年公开金额相除替代。累计期重叠，也不能当成五次独立试验。

官方工业来源：{official_sources}。沪市主板来源：{sse_sources}；最后一份同时覆盖2022年报和2023年一季度。

**2023年一季度的增长，主要落在哪些行业。**

{chr(10).join(layers)}

指数有效样本共299家，合计归母利润从同组上年约{total.parent_net_profit_prior/1e8:,.2f}亿元变为{total.parent_net_profit_current/1e8:,.2f}亿元，增量{total.parent_net_profit_change/1e8:.2f}亿元。非银金融约占这项增量的{contribution_share:.2%}。这是汇总结果的来源，不是非银“造成了”相应比例的指数涨幅。

非金融内部也互相抵消：电力设备、食品饮料、公用事业对全部样本利润增速分别约贡献+1.48、+0.86、+0.61个百分点；电子、基础化工、石油石化、建筑材料分别约贡献-0.88、-0.74、-0.52、-0.51个百分点。所有行业都在保存表内，没有只保留支持恢复的一组。

**数量恢复之后，还要经过售价和利润率。**

{chr(10).join(margin_table)}

以上每期本期和上期使用同一集合；各观察日之间会换样，所以不同表行的同比基数不能直接串接。净利/收入只是同组归母利润与营业总收入之比，不是毛利率，也没有把银行或保险收入强行套进制造业利润率。

2023年一季度非金融样本收入增4.47%，归母利润增2.46%，净利/收入由6.20%降至6.08%，下降约0.12个百分点。对称会计分解中，收入项对应利润增加{nf.revenue_term/1e8:.2f}亿元，利润率项对应减少{abs(nf.margin_term)/1e8:.2f}亿元，两者合为{nf.parent_net_profit_change/1e8:.2f}亿元。公式为“收入变化×两期平均净利率＋净利率变化×两期平均收入”；它只分解金额，不能把收入项直接叫作销量贡献，或把利润率项全部归为原料成本。

工业总体的当期原文给出更具体的传导背景：1至2月营收同比下降1.3%，降得比成本快；PPI同比下降1.1%，价格压力与需求未完全恢复并存。统计局估计毛利下降拖累工业利润增速18.6个百分点。这些因素彼此重叠，不能再相加，也不能把工业总体的18.6点套到沪深300。[统计局2023年3月27日说明]({sources['nbs_2023janfeb_cause']['url']})。

到整个一季度，官方工业营收仍降0.5%，成本反而增0.6%；制造业利润下降29.4%，采矿业下降5.8%，公用事业相关工业增长33.2%。这组事实支持“恢复速度、名义售价和利润兑现需要分别观察”。与此同时，指数样本非金融利润仍为正增长，因此也不支持把工业利润负增长直接等同于指数盈利崩塌。[一季度工业原表]({sources['nbs_2023q1']['url']})。

**累计报表会掩盖哪一季正在变化。**

{chr(10).join(quarter_table)}

每个单季都按同公司集合计算，二至四季度以本期累计减前一季度累计，第一季度直接使用季度数。共同成分是五个观察日名单的258家公司交集，再按各期已有行业分类区分金融和非金融；它是事后构成对照，不是2022年已经知道的名单，也不构成独立样本。两种集合都出现四季度转弱、一季度恢复增长，说明该定性轮廓并非只靠本轮换样出现。共同成分不能消除财务后来修订和行业分类变化。

**从公司金额总和到指数收益，还缺一层权重和定价。**

银行在这组一季度归母利润金额中约占{bank_profit_share:.2%}，而旧4月28日快照中的参考指数权重仅约{bank.reference_weight_sum:.2%}。后者尚未认证历史原版本，作为量级参考保留。两类占比的含义不同：集团利润包含全部股东经济规模，指数价格按调整股本和价格构造，还受A/H股分配、母子公司合并范围、指数维护影响。因此本轮不把公司合计利润当作指数EPS，不用利润贡献推算指数应涨几个点。

对非银金融还要保留两种竞争解释：金融资产价格变化可能反过来改变投资相关损益；会计规则变化可能改变列报和比较基数。财政部原通知明确所述境内外上市企业自2023年1月1日起执行新保险合同准则。这里仅确认制度事实，未识别其对本轮非银增速的净影响，也未认证聚合器2022和2023字段采用同一重述基础。[财政部原通知]({sources['mof_insurance_standard_2020']['url']})。不能把58.99%的回顾增速全部命名为新增经营景气或全部解释成会计变化。

**这些信息什么时候才可能知道。**

五期指数财务的统一观察日依次为2022年5月5日、9月1日、11月1日，以及2023年5月4日、5月4日；仅按名义公告早于观察日接入。这种时钟仍没有消除后续修订。2023年1至2月工业解释在3月27日公布，一季度工业结果在4月27日公布，沪市主板一季度总结在4月29日公布。它们不能回填到2月1日或3月2日的PMI买入时刻。

上一轮两个PMI账户期间截止2023年3月31日。季度财务在此之后才集中确认，所以本轮是后续兑现的回顾解释，不声称当时已经拥有这些数字。原美国CPI、国内政策和PMI账户均未改变，也没有将某个盈利阶段拼进新策略。

资料中保留一个时间缺口：2022前三季度工业页面只给成文日期10月27日，未把它伪写成已核实的发布时间。财政部通知则分别记录2020年12月19日成文、12月24日网页发布和2023年1月1日适用范围内生效。所有原件来自目前取得的历史页面，并非本机当年保存的不可变版本。

**本轮完成了什么，仍然缺什么。**

完成：五个固定报告期、两类集合、全部行业的656行汇总；1500个公司报告期位置，累计与单季两种表达共3000行；11份官方原网页；收入与利润率的金额分解及组间贡献复算。五期有效财务覆盖依次299、300、299、300、299家公司，缺失不补零；早期未知行业分别保留，未塞入非金融。全部有效财务依赖均存在晚于观察日更新，原权重仅参考。复算只说明计算与保存数据一致。

已得到的实用约束是：先确认宏观变量影响的是哪一组总体，再区分数量、名义收入、利润率和利润发生期间，最后才研究指数权重与市场已反映的部分。仅“PMI回升”或“合计利润转正”，都不足以填补当时的预期差证据。这里没有识别新增可交易优势，净夏普和年化收益为NOT_COMPUTED；原夏普1.2与年化10%目标仍未实现。

下一项历史问题将转向已有数据中2022年1月至2023年3月的指数整体重估、行业价格抵消与资金约束，先检查此前是否已做同等比较；不继续泛搜公司财报，也不以本轮解释向旧PMI规则追加条件。

{link(figure, '查看总体、行业与利润率图')}。

数据入口：{link(OUT / 'aggregates.csv', '五期全行业汇总')}；{link(OUT / 'mechanism_findings.json', '关键数字与未识别环节')}；{link(OUT / 'official_industrial_facts.json', '工业原表提取')}；{link(OUT / 'official_sse_facts.json', '沪市主板原文提取')}；{link(OUT / 'source_manifest.json', '来源与公开时间')}；{link(ROOT / 'research/historical_index_earnings_population_v1.py', '完整研究脚本')}。

完成时间：{now()}。
"""
    report_path.write_text(report, encoding="utf-8")
    result = read(OUT / "result.json")
    summary = "五期指数样本与官方工业、沪市主板总体完成对照。2023Q1回顾重建样本归母利润+6.51%，非银贡献4.47点，非金融+2.46%且净利率同比下降；单季与共同成分对照保留四季度弱、一季度恢复轮廓。全部财务有后续更新，未作为交易输入；工业口径和利润金额权重均不能替代指数收益。"
    next_question = "核对已有2022年1月至2023年3月指数整体估值、行业价格与资金约束研究的覆盖；缺失处以全部固定月份及同一公开时钟比较共同重估和行业抵消，不泛搜个股财报，不向失败PMI规则追加阈值。"
    result.update(status="COMPLETED_HISTORICAL_MECHANISM_STUDY_NO_TRADING_CANDIDATE", completed_at=now(), classification=findings["classification"], discovery=summary, next_historical_question=next_question, report=rel(report_path), figure=rel(figure), figure_visually_reviewed=True, source_count=len(sources), aggregate_rows=len(a), component_rows=3000, distinct_company_period_positions=1500, net_sharpe=None, net_cagr=None, account_status="NOT_RUN_NO_ADMITTED_TRADING_CANDIDATE", metric_status="NOT_COMPUTED", report_status="WRITTEN_PENDING_FINAL_REVIEW", goal_achieved=False)
    save(OUT / "result.json", result)
    output_files = [report_path, OUT / "components.parquet", OUT / "aggregates.parquet", OUT / "official_industrial_facts.json", OUT / "official_sse_facts.json", OUT / "mechanism_findings.json", figure]
    save(OUT / "research_receipt.json", {"created_at": now(), "script": rel(Path(__file__)), "script_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json"), "output_hashes": {rel(p): digest(p) for p in output_files}, "calculation_checks": rel(OUT / "calculation_checks.json"), "goal_achieved": False})
    print(json.dumps({"报告": rel(report_path), "新增账户": 0, "目标完成": False}, ensure_ascii=False))


def compute():
    assert digest(OUT / "protocol.json") == read(OUT / "freeze.json")["protocol_sha256"]
    raw = pd.read_parquet(OLD / "inputs/income.parquet")
    membership = pd.read_parquet(OLD / "inputs/membership.parquet")
    states = pd.read_parquet(OLD / "inputs/monthly_state.parquet")
    weights = pd.read_parquet(OLD / "inputs/weights.parquet")
    for col in ["report_end", "available_at", "update_date"]:
        raw[col] = pd.to_datetime(raw[col]).dt.normalize()
    assert not raw.duplicated(["stock_code", "report_end"]).any()
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    calendar = pd.DatetimeIndex(sorted(membership.membership_date.unique()))
    states["origin"] = pd.to_datetime(states.origin)
    weights["trade_date"] = pd.to_datetime(weights.trade_date)
    frames, clocks = [], []
    for spec in PERIODS:
        date = calendar[calendar.searchsorted(pd.Timestamp(spec["cutoff"]), side="right")]
        codes = membership.loc[membership.membership_date.eq(date), "symbol"].tolist()
        assert len(codes) == len(set(codes)) == 300
        industry_date = states.loc[states.origin.lt(date), "origin"].max()
        industry = states[states.origin.eq(industry_date)].set_index("stock_code")
        weight_date = weights.loc[weights.trade_date.lt(date), "trade_date"].max()
        reference = weights[weights.trade_date.eq(weight_date)].set_index("con_code")
        assert (date - industry_date).days <= 35 and (date - weight_date).days <= 62
        frame = pd.DataFrame({"stock_code": codes, "period": spec["period"], "observation_date": date})
        frame["industry"] = frame.stock_code.map(industry.industry_name).fillna("行业缺失")
        frame["finance_group"] = frame.industry.where(frame.industry.isin(["银行", "非银金融", "行业缺失"]), "非金融")
        frame["industry_date"] = industry_date
        frame["weight_date"] = weight_date
        frame["reference_weight"] = frame.stock_code.map(reference.weight) / 100
        frame["weight_status"] = "UNVERSIONED_REFERENCE_ONLY"
        end = pd.Timestamp(spec["report_end"])
        quarter_prior = end - pd.offsets.QuarterEnd(1) if end.month != 3 else None
        requested = {"current": end, "prior": end - pd.DateOffset(years=1)}
        if quarter_prior is not None:
            requested.update(previous_quarter=quarter_prior, prior_previous_quarter=quarter_prior - pd.DateOffset(years=1))
        for prefix, period in requested.items():
            sub = raw[raw.report_end.eq(period) & raw.available_at.lt(date)].set_index("stock_code")
            for col in FIELDS + ["available_at", "update_date", "source_sha256"]:
                frame[prefix + "_" + col] = frame.stock_code.map(sub[col])
            frame[prefix + "_report_end"] = period
        for kind in ["YTD", "QUARTER"]:
            q = frame.copy()
            q["measure"] = kind
            dependencies = ["current", "prior"]
            if kind == "QUARTER" and quarter_prior is not None:
                dependencies += ["previous_quarter", "prior_previous_quarter"]
            for field in FIELDS:
                q["value_" + field] = q["current_" + field]
                q["base_" + field] = q["prior_" + field]
                if kind == "QUARTER" and quarter_prior is not None:
                    q["value_" + field] -= q["previous_quarter_" + field]
                    q["base_" + field] -= q["prior_previous_quarter_" + field]
            required = [side + field for side in ["value_", "base_"] for field in FIELDS]
            q["valid"] = q[required].notna().all(axis=1)
            q["late_updated"] = q[[p + "_update_date" for p in dependencies]].gt(date).any(axis=1)
            q["latest_dependency_notice"] = q[[p + "_available_at" for p in dependencies]].max(axis=1)
            q["latest_dependency_update"] = q[[p + "_update_date" for p in dependencies]].max(axis=1)
            q["source_version"] = "RETROSPECTIVE_CURRENT_AGGREGATOR_VALUE_NOT_FIRST_DISCLOSURE"
            frames.append(q)
        clocks.append({**spec, "observation_date": date, "industry_date": industry_date, "reference_weight_date": weight_date})
    components = pd.concat(frames, ignore_index=True)
    common = set.intersection(*(set(x.stock_code) for _, x in components[components.measure.eq("YTD")].groupby("period")))
    components["common_membership"] = components.stock_code.isin(common)
    components.to_parquet(OUT / "components.parquet", index=False)
    save(OUT / "observation_clocks.json", clocks)
    rows = []
    for (period, measure), part in components.groupby(["period", "measure"], sort=False):
        for population in ["DYNAMIC_300", "COMMON_MEMBERSHIP"]:
            base = part if population == "DYNAMIC_300" else part[part.common_membership]
            aggregate_base = base.loc[base.valid, "base_parent_net_profit"].sum(min_count=1)
            groups = [("全部", "全部", base)]
            groups += [("金融分层", name, group) for name, group in base.groupby("finance_group")]
            groups += [("行业", name, group) for name, group in base.groupby("industry")]
            for kind, name, sub in groups:
                valid = sub[sub.valid]
                row = {"period": period, "measure": measure, "population": population, "group_type": kind, "group": name, "members": len(sub), "valid_members": len(valid), "missing_members": len(sub) - len(valid), "observation_date": part.observation_date.iloc[0], "late_updated_members": int(valid.late_updated.sum()), "reference_weight_sum": sub.reference_weight.sum(min_count=1), "valid_reference_weight_sum": valid.reference_weight.sum(min_count=1)}
                for field in FIELDS:
                    current = valid["value_" + field].sum(min_count=1)
                    previous = valid["base_" + field].sum(min_count=1)
                    row[field + "_current"] = current
                    row[field + "_prior"] = previous
                    row[field + "_change"] = current - previous
                    row[field + "_yoy"] = current / previous - 1 if previous > 0 else np.nan
                change = row["parent_net_profit_change"]
                row["profit_growth_contribution_pp"] = change / aggregate_base * 100 if aggregate_base > 0 else np.nan
                row["profit_increasing_fraction"] = (valid.value_parent_net_profit > valid.base_parent_net_profit).mean()
                row["profitable_fraction"] = valid.value_parent_net_profit.gt(0).mean()
                if name == "非金融" and len(valid):
                    r1, r0 = row["total_operating_revenue_current"], row["total_operating_revenue_prior"]
                    p1, p0 = row["parent_net_profit_current"], row["parent_net_profit_prior"]
                    m1, m0 = p1 / r1, p0 / r0
                    row.update(parent_net_margin_current=m1, parent_net_margin_prior=m0, revenue_term=(r1-r0)*(m1+m0)/2, margin_term=(m1-m0)*(r1+r0)/2)
                    row["margin_identity_error"] = change-row["revenue_term"]-row["margin_term"]
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_parquet(OUT / "aggregates.parquet", index=False)
    result.to_csv(OUT / "aggregates.csv", encoding="utf-8-sig", index=False)
    identities = []
    for keys, part in result.groupby(["period", "measure", "population"], sort=False):
        all_row = part[part.group_type.eq("全部")].iloc[0]
        layers = part[part.group_type.eq("金融分层")]
        error = layers.parent_net_profit_change.sum()-all_row.parent_net_profit_change
        pp_error = layers.profit_growth_contribution_pp.sum()-all_row.parent_net_profit_yoy*100
        assert abs(error) < 0.02 and abs(pp_error) < 1e-8
        identities.append({"period_measure_population": list(keys), "profit_change_error_cny": error, "growth_contribution_error_pp": pp_error})
    assert components[components.valid].latest_dependency_notice.lt(components[components.valid].observation_date).all()
    assert result.margin_identity_error.dropna().abs().max() < 0.02
    save(OUT / "calculation_checks.json", {"status": "PASS_SAVED_AGGREGATION_IDENTITIES_ONLY", "component_rows": len(components), "aggregate_rows": len(result), "common_membership_count": len(common), "identities": identities, "max_margin_error_cny": result.margin_identity_error.abs().max(), "first_vintage_verified": False})
    save(OUT / "result.json", {"study_id": read(OUT / "protocol.json")["study_id"], "status": "COMPUTED_PENDING_INTERPRETATION", "created_at": now(), "periods": len(PERIODS), "common_membership_count": len(common), "new_accounts": 0, "new_models": 0, "goal_achieved": False, "reconstructed_values_are_tradable": False, "first_vintage_verified": False})
    print(result.loc[result.population.eq("DYNAMIC_300") & result.measure.eq("YTD") & result.group_type.ne("行业"), ["period", "group", "valid_members", "late_updated_members", "total_operating_revenue_yoy", "parent_net_profit_yoy", "profit_growth_contribution_pp"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指数样本总体盈利的固定报告期研究")
    parser.add_argument("action", choices=["prepare", "collect", "compute", "facts", "plot", "publish"])
    action = parser.parse_args().action
    {"prepare": prepare, "collect": collect, "compute": compute, "facts": source_facts, "plot": plot, "publish": publish}[action]()
