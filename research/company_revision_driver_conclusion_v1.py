"""分解同机构盈利预测的构成变化，保留经营指引与预测代理的不同口径。"""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from bs4 import BeautifulSoup

from company_expectation_anchor_conclusion_v1 import aggregate_table, displayed_details


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_company_revision_driver_v1"
STAMP = datetime.now(timezone(timedelta(hours=8))).isoformat()
IVAN_Q1 = "https://www.ivanhoemines.com/news-stories/news-release/ivanhoe-mines-issues-2026-first-quarter-financial-results-overview-of-operations-and-exploration-activities/"
IVAN_Q2 = "https://www.ivanhoemines.com/news-stories/news-release/ivanhoe-mines-issues-2026-second-quarter-financial-results-overview-of-operations-and-exploration-activities/"
IVAN_JAN = "https://www.ivanhoemines.com/news-stories/news-release/ivanhoe-mines-provides-2025-production-results-2026-production-guidance/"
NAMES = {"300750": "宁德时代", "601899": "紫金矿业"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def line_series(text, label):
    token = r"(?:\([\d,.]+\)|-?[\d,.]+)"
    pattern = re.escape(label) + r"\s+" + r"\s+".join(["(" + token + ")"] * 5)
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"第3页未找到完整的五年序列：{label}")
    values = []
    for raw in match.groups():
        negative = raw.startswith("(")
        value = float(raw.strip("()").replace(",", ""))
        values.append(-value if negative else value)
    return dict(zip(["2024", "2025", "2026", "2027", "2028"], values))


def pct(new, old):
    return (new / old - 1) * 100


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮比较已完成，不覆盖原件基准。")
    src = read(OUT / "source_result.json")
    broker_scope = read(OUT / "broker_document_scope.json")
    docs = read(OUT / "broker_document_result.json")
    doc_urls = {row["key"]: row["url"] for row in docs["receipts"]}
    labels = {
        "revenue": "营业收入", "cost": "营业成本", "finance_expense": "财务费用",
        "income_tax": "所得税费用", "minority": "少数股东损益", "parent_profit": "归属于母公司净利润",
        "cfo": "经营活动现金流", "capex": "资本开支",
    }
    parsed = {}
    for code in NAMES:
        parsed[code] = {}
        for version in ["previous", "latest"]:
            key = code + "_" + version
            pages = docs["responses"][key]
            page = next(row["text"] for row in pages if row["page"] == 3)
            current_labels = dict(labels)
            if code == "601899":
                current_labels.update(income_tax="所得税", parent_profit="归属母公司净利润", capex="资本支出")
            else:
                current_labels.update(tax_surcharge="营业税金及附加", selling="销售费用", admin="管理费用", research="研发费用", investment="投资收益", impair_and_fair="值变动", other_income="其他收入", eps="每股收益", free_cashflow="企业自由现金流")
            fields = {field: line_series(page, label) for field, label in current_labels.items()}
            fields["gross_profit"] = {year: fields["revenue"][year] - fields["cost"][year] for year in fields["revenue"]}
            fields["gross_margin_pct"] = {year: fields["gross_profit"][year] / fields["revenue"][year] * 100 for year in fields["revenue"]}
            first_page = pages[0]["text"]
            date = re.search(r"(2026)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", first_page)
            if not date or code not in first_page:
                raise ValueError("研报日期或证券代码未匹配原件。")
            pair = next(row for row in broker_scope["pairs"] if row["code"] == code)
            parsed[code][version] = {
                "fields": fields, "financial_unit": "百万元；EPS为元；比率为百分数",
                "document_date": f"{date[1]}-{int(date[2]):02d}-{int(date[3]):02d}",
                "vendor_directory_publication_date": pair[version]["publishDate"],
                "source_url": doc_urls[key], "source_pdf": "sources/" + key + ".pdf", "financial_table_pdf_page": 3,
            }
    catl_old, catl_new = [parsed["300750"][v]["fields"] for v in ["previous", "latest"]]
    zijin_old, zijin_new = [parsed["601899"][v]["fields"] for v in ["previous", "latest"]]

    def delta(old, new, field):
        return (new[field]["2026"] - old[field]["2026"]) / 100

    catl_components = [
        {"name": "营业毛利", "delta_cny_100million": delta(catl_old, catl_new, "gross_profit")},
        {"name": "费用与税附", "delta_cny_100million": -sum(delta(catl_old, catl_new, field) for field in ["tax_surcharge", "selling", "admin", "research", "finance_expense"])},
        {"name": "投资收益", "delta_cny_100million": delta(catl_old, catl_new, "investment")},
        {"name": "减值及公允", "delta_cny_100million": delta(catl_old, catl_new, "impair_and_fair")},
        {"name": "其他收入", "delta_cny_100million": delta(catl_old, catl_new, "other_income")},
        {"name": "所得税及少数", "delta_cny_100million": -sum(delta(catl_old, catl_new, field) for field in ["income_tax", "minority"])},
    ]
    catl_reported_delta = delta(catl_old, catl_new, "parent_profit")
    catl_residual = catl_reported_delta - sum(row["delta_cny_100million"] for row in catl_components)
    catl_components.append({"name": "表内差额", "delta_cny_100million": catl_residual})
    zijin_components = [
        {"name": "营业毛利", "delta_cny_100million": delta(zijin_old, zijin_new, "gross_profit")},
        {"name": "财务费用", "delta_cny_100million": -delta(zijin_old, zijin_new, "finance_expense")},
        {"name": "所得税", "delta_cny_100million": -delta(zijin_old, zijin_new, "income_tax")},
        {"name": "少数股东损益", "delta_cny_100million": -delta(zijin_old, zijin_new, "minority")},
    ]
    zijin_reported_delta = delta(zijin_old, zijin_new, "parent_profit")
    if abs(sum(row["delta_cny_100million"] for row in zijin_components) - zijin_reported_delta) > 0.02 or abs(catl_residual) > 0.02:
        raise ValueError("盈利修正桥接差额超过表内百万元级尾差，先检查原表。")
    broad = read(ROOT / "reports/research/510300_broad_index_driver_bridge_v1/reviewed_facts.json")
    h1 = {"300750": broad["catl"]["h1_parent_net_profit_cny_thousand"] / 100000, "601899": 39169807243 / 100000000}
    current = []
    for code, name in NAMES.items():
        receipt = next(row for row in src["receipts"] if row["key"] == "ths_" + code)
        soup = BeautifulSoup((OUT / f"sources/ths_{code}.html").read_bytes(), "html.parser")
        text = soup.get_text(" ", strip=True)
        date = re.search(r"截至\s*(\d{4}-\d{2}-\d{2})", text)
        if not date:
            raise ValueError("当期预测代理没有标注日期。")
        tables = soup.find_all("table")
        profits = aggregate_table(tables[1])
        mean = next(row for row in profits if row["year"] == 2026)["mean"]
        current.append({
            "code": code, "name": name, "page_as_of": date[1], "received_at": receipt["received_at"], "source_url": receipt["url"],
            "source_kind": "CURRENT_VENDOR_PROXY_NOT_COMPLETE_MARKET_CONSENSUS", "net_profit_aggregates": profits,
            "visible_detail_rows": displayed_details(tables[2]), "h1_actual_parent_profit_cny_100million": h1[code],
            "conditional_h2_profit_cny_100million": mean - h1[code], "conditional_h2_vs_h1_pct": pct(mean - h1[code], h1[code]),
            "comparability_condition": "汇总字段名为净利润；减去公司归母净利润仅在二者同口径的假设下成立，未逐一取得所有机构原件。",
            "can_infer_revision_from_single_snapshot": False,
        })
    save(OUT / "current_vendor_forecast_baseline.json", {"recorded_at": STAMP, "companies": current})
    catl_h1_tax_million = 1324939 / 1000
    catl_h1_rev_million = 276916580 / 1000
    catl_implied_h2_tax_ratio = (catl_new["tax_surcharge"]["2026"] - catl_h1_tax_million) / (catl_new["revenue"]["2026"] - catl_h1_rev_million) * 100
    mine_h1 = 71417 + 64328
    mine_h2_required = [290000 - mine_h1, 310000 - mine_h1]
    facts = {
        "recorded_at": STAMP, "financial_forecasts": parsed,
        "catl_2026_revision_components": catl_components,
        "catl_2026_parent_profit_change_cny_100million": catl_reported_delta,
        "catl_report_table_residual_cny_100million": catl_residual,
        "catl_2026_revenue_revision_pct": pct(catl_new["revenue"]["2026"], catl_old["revenue"]["2026"]),
        "catl_2026_gross_margin_revision_pp": catl_new["gross_margin_pct"]["2026"] - catl_old["gross_margin_pct"]["2026"],
        "catl_2026_cfo_revision_cny_100million": delta(catl_old, catl_new, "cfo"),
        "catl_2026_capex_additional_cash_use_cny_100million": -delta(catl_old, catl_new, "capex"),
        "catl_broker_implied_h2_tax_ratio_pct": catl_implied_h2_tax_ratio,
        "batt1_comparison_limit": "BATT1是Q3对H1的经营判断；研报倒算为H2，期限不同，不能直接做同口径超预期检验。",
        "zijin_2026_revision_components": zijin_components,
        "zijin_2026_parent_profit_change_cny_100million": zijin_reported_delta,
        "zijin_2026_parent_profit_revision_pct": pct(zijin_new["parent_profit"]["2026"], zijin_old["parent_profit"]["2026"]),
        "zijin_2026_gross_margin_revision_pp": zijin_new["gross_margin_pct"]["2026"] - zijin_old["gross_margin_pct"]["2026"],
        "comparison_limits": ["各一机构、两报告，只识别该机构已披露预测变化，不代表市场整体。", "预测表的会计差额不是已识别的实际经营因果贡献。", "原件落款与供应商发布日分别保留，最早市场可得时刻未核实，不构造历史事件收益。", "报告维持净利润不能推断内部量价成本和现金流预测没变。"],
        "kamoa_guidance": {
            "january_document_date": "2026-01-15", "january_2026_range_tonnes": [380000, 420000],
            "january_scope": "精矿含铜量", "q1_document_date": "2026-05-06", "q1_2026_range_tonnes": [290000, 330000],
            "q1_change_referenced_date": "2026-03-31", "q2_document_date": "2026-07-29", "q2_2026_range_tonnes": [290000, 310000],
            "q2_scope": "阳极铜、粗铜及可销售炉渣精矿含铜，100%项目口径；表格标题另简称阳极铜或粗铜。",
            "zijin_h1_stated_range_tonnes": broad["zijin"]["kamoa_annual_plan_revised_tonnes"],
            "prior_range_status": "紫金半年报所载29至33万吨保留为其原文，后续项目经营比较需并列艾芬豪29至31万吨；尚未完整调和产品范围差异。",
            "q1_production_tonnes": 71417, "q2_production_tonnes": 64328, "h1_production_tonnes": mine_h1,
            "conditional_h2_to_meet_latest_guidance_tonnes": mine_h2_required,
            "conditional_h2_increase_over_h1_pct": [pct(value, mine_h1) for value in mine_h2_required],
            "conditional_h2_midpoint_increase_pct": pct(sum(mine_h2_required) / 2, mine_h1),
            "h2_ramp_already_in_guidance": True,
            "inventory_note": "年初约5万吨、6月末约4万吨含铜库存；年末目标2.5至3万吨。销量、冶炼产量与矿山新开采量需分别解释。",
            "operating_constraints": ["岩土及水文条件影响巷道开发", "矿山原料供应限制冶炼进一步爬坡", "卡库拉新区域回采预计2027H2开始", "入选品位和回收率影响有效含铜产出"],
            "q2_c1_usd_per_lb": 2.84, "q1_c1_usd_per_lb": 2.58,
            "q2_smelter_cost_usd_per_lb": 0.41, "q2_acid_credit_usd_per_lb": 0.39,
            "q2_acid_realized_price_usd_per_tonne": 465, "july_august_acid_contract_price_usd_per_tonne": 840,
            "byproduct_root_cause": "公司将高硫酸价格联系到全球供应链和霍尔木兹运输受限；未独立估计各原因贡献。",
            "cost_accounting_note": "Q1部分冶炼成本资本化；Q2经营成本上升不全是实际效率恶化；C1含副产品抵扣且不等于营业成本或归母净利润。",
            "source_urls": [IVAN_JAN, IVAN_Q1, IVAN_Q2],
            "new_september_guidance_change_verified": False,
        },
        "etf_exposure": {"weight_date": "2026-08-31", "catl_pct": 3.66, "zijin_pct": 2.11, "combined_pct": 5.77,
                         "hypothetical_both_share_prices_rise_pct": 10, "hypothetical_static_index_contribution_pp": 0.577,
                         "hypothetical_is_forecast": False, "earnings_revision_equal_stock_return_assumed": False},
        "new_strategy_accounts": 0, "new_return_tests": 0,
    }
    save(OUT / "reviewed_facts.json", facts)
    save(OUT / "prior_guidance_clarification.json", {
        "recorded_at": STAMP, "prior_file": "reports/research/510300_broad_index_driver_bridge_v1/reviewed_facts.json",
        "prior_value_is_true_to_its_source": True, "clarification": facts["kamoa_guidance"]["prior_range_status"],
        "research_effect": "后续不以超越旧33万吨上限或仅出现环比恢复判定超预期；优先核对最新可比产品口径与经营路径。",
        "no_backdated_forecast_change": True,
    })

    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    matplotlib.rcParams["font.family"] = font.get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14.1, 6.7), sharey=True)
    for ax, components, total, title in [
        (axes[0], catl_components, catl_reported_delta, "国信：宁德2026年预测｜4月22日 → 7月31日"),
        (axes[1], zijin_components, zijin_reported_delta, "华龙：紫金2026年预测｜3月24日 → 8月25日"),
    ]:
        cumulative = 0.0
        for i, row in enumerate(components):
            value = row["delta_cny_100million"]
            bottom = min(cumulative, cumulative + value)
            ax.bar(i, abs(value), bottom=bottom, color="#318674" if value >= 0 else "#bd5f4d", width=0.66)
            ax.text(i, bottom + abs(value) + 3, f"{value:+.2f}", ha="center", fontsize=9)
            if i < len(components) - 1:
                ax.plot([i + 0.33, i + 0.67], [cumulative + value] * 2, color="#9ca8ad", lw=0.8)
            cumulative += value
        i = len(components)
        ax.bar(i, abs(total), bottom=min(0, total), color="#2c5674", width=0.66)
        ax.text(i, max(total, 0) + 3, f"{total:+.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(i + 1), [row["name"] for row in components] + ["归母变化"], rotation=25, ha="right", fontsize=9)
        ax.axhline(0, color="#7a8790", lw=0.9)
        ax.grid(axis="y", color="#e7ebee")
        ax.set_axisbelow(True)
        ax.set_ylim(-115, 132)
        ax.set_title(title, fontsize=11, pad=16)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("预测变动对归母利润的会计桥接（亿元）")
    fig.suptitle("净利润预测的表面变化，可能隐藏不同的经营假设", x=0.065, ha="left", fontsize=17, y=0.97)
    fig.text(0.065, 0.055, "每家公司仅比较同一机构的两份研报；这是预测表的构成差额，不是实际经营结果或已识别因果效应。\n宁德费用与税附含销售、管理、研发、财务费用及税金；表内差额约0.01亿元保留。", fontsize=10, color="#53636d")
    fig.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.25, wspace=0.11)
    chart_name = "盈利预测变化的内部构成.png"
    fig.savefig(OUT / chart_name, dpi=170, facecolor="white")
    plt.close(fig)

    comparison_fields = [("revenue", "营业收入"), ("gross_profit", "营业毛利"), ("parent_profit", "归母净利润"), ("cfo", "经营活动现金流")]
    catl_table = "\n".join(f"| {label} | {catl_old[field]['2026']/100:.2f} | {catl_new[field]['2026']/100:.2f} | {delta(catl_old, catl_new, field):+.2f} |" for field, label in comparison_fields)
    catl_table += f"\n| 资本开支现金支出 | {-catl_old['capex']['2026']/100:.2f} | {-catl_new['capex']['2026']/100:.2f} | {-delta(catl_old, catl_new, 'capex'):+.2f} |"
    proxy_table = "\n".join(f"| {row['name']} | {row['net_profit_aggregates'][0]['institution_count']} | {row['net_profit_aggregates'][0]['mean']:.2f} | {row['h1_actual_parent_profit_cny_100million']:.2f} | {row['conditional_h2_profit_cny_100million']:.2f} | {row['conditional_h2_vs_h1_pct']:+.1f}% |" for row in current)
    report_name = "盈利预测为何改变及剩余预期差.md"
    report = f"""# 盈利预测为何改变，以及哪些变化已经在预期里

记录时间：{STAMP}。沿用宁德时代、紫金矿业研究对象，取得同机构相邻研报原件、当期供应商预测快照和卡莫阿合营方经营更新。本轮没有收益回测，不把本次才读到的旧消息当9月新事件。

**主要发现：宁德的归母预测保持不变，但其收入、毛利与经营现金流预测已下调，其他项目抵消了利润影响；紫金的一组归母上修，财务表中主要表现为毛利率假设提高。卡莫阿的下半年恢复本来就在经营指引里，单纯恢复不能直接称为超预期。**

**宁德：维持盈利预测，内部假设仍可显著改变。**

国信证券4月22日与7月31日两份原件，对2026—2028年归母净利润均预测963.51、1200.89、1408.87亿元。2026年EPS从21.11变为20.83元，而归母金额不变；原件注明按最新总股本摊薄，不能把每股数值变化当作总盈利下修。这里只识别分母或每股口径的影响，不追认某一具体增发事件。

| 2026年预测，亿元 | 4月22日原件 | 7月31日原件 | 后版减前版 |
| --- | ---: | ---: | ---: |
{catl_table}

营业毛利预测减少99.95亿元，由费用与税附预测变化约+56.06亿元、投资收益+35亿元、减值及公允价值变动+27亿元等抵消；其他收入预测减少24亿元，所得税和少数股东损益变化合计约+5.88亿元，表内差额约{catl_residual:.2f}亿元。费用下降、减值减少也可能对应真实经营改善，不能一概称作利润操纵或一次性收益；不过，未来要验证的支撑项显然不再只是电池销量。

这些是分析师模型内部的假设变化，不是公司已经发生了相同金额的损失或增益。原报告叙述提到规模效应、产品设计、产业链布局应对原料压力，但没有逐项量化售价、产品结构和原料成本各自的修正贡献，不能由会计桥接冒充经济因果分解。证据：[国信4月原件]({doc_urls['300750_previous']})、[国信7月原件]({doc_urls['300750_latest']})，两份PDF第1、3页。

税负方面，7月报告预测全年税金及附加30.88亿元、营业收入6176.49亿元；减去H1实际后，条件倒算H2税费比例约{catl_implied_h2_tax_ratio:.4f}%。该表没有给出电池消费税的独立税基、抵扣和转嫁假设，无法据此断言已经充分计入或完全漏算。原BATT1只判断Q3比例是否高于H1的0.4785%；它与H2期限不同，命中不等于超过市场预期，也不据本次材料改写原卡。

**紫金：总利润上修，需要查清模型到底改了哪一层。**

华龙两份原件落款3月24日和8月25日，2026年归母预测由754.08升至826.46亿元，上调{facts['zijin_2026_parent_profit_revision_pct']:.2f}%。但两版营业收入预测均为4496.24亿元，营业成本预测由3141.69降至3033.79亿元，对应毛利增加107.90亿元、毛利率提高约{facts['zijin_2026_gross_margin_revision_pp']:.2f}个百分点；财务费用减少1.29亿元，被所得税增加19.55亿元和少数股东损益增加17.26亿元部分抵消，归母增量72.38亿元。

因此，后续可验证的重点是这个毛利假设能否兑现。报告叙述涉及金属价格、项目增量和副产品价值，但表格未单列各矿山量价成本模型；营业成本总额减少不能直接等同于矿山每吨实际成本下降，也不能把归母上调9.6%直接变成股价应涨9.6%。证据：[华龙3月原件]({doc_urls['601899_previous']})、[华龙8月原件]({doc_urls['601899_latest']})，两份PDF第1—3页。

![盈利预测构成](<{(OUT / chart_name).as_posix()}>)

**卡莫阿：恢复的速度和来源，比“是否恢复”更有区分力。**

1月15日指引为38—42万吨精矿含铜；5月6日更新援引3月31日调整后的29—33万吨阳极铜或粗铜含铜；7月29日进一步收紧为29—31万吨，并在经营口径中含可销售炉渣精矿。不同产品阶段、冶炼损耗和库存变化不能混作纯矿山产量变化。[1月公告]({IVAN_JAN})、[一季报更新]({IVAN_Q1})、[二季报更新]({IVAN_Q2})。

上一轮紫金半年报确实载29—33万吨，原件事实保留；后续不能将它当唯一项目基准。双方范围尚未完全调和，已在[补充说明](prior_guidance_clarification.json)并列。艾芬豪披露的H1合计13.5745万吨，对应29—31万吨全年目标，H2需约{mine_h2_required[0]/10000:.4f}—{mine_h2_required[1]/10000:.4f}万吨，即较H1约增加{pct(mine_h2_required[0],mine_h1):.1f}%—{pct(mine_h2_required[1],mine_h1):.1f}%。这个条件算式说明恢复已经包含在该指引中，不是我新增的产量预测。

进一步追原因：岩土、水文和巷道开发约束高品位矿区进入时间；冶炼能力还受精矿供应限制；库存去化能抬升销售兑现，而不等于新增开采。Q2公司披露冶炼成本0.41美元/磅、副产品硫酸抵扣0.39美元/磅，7—8月硫酸合同价格高于Q2；公司把高酸价联系到供应链及霍尔木兹运输受限。供给扰动因此既可能提高柴油和运输成本，也能通过副产品提高收益。C1现金成本仍由Q1的2.58升至Q2的2.84美元/磅，不能只看抵扣就说总成本下降；Q1部分冶炼成本资本化也影响季度比较。来源为上述二季报更新，均为公司表述及当时假设。

这条机制产生的条件判断是：如果供给运输恢复，副产品高价收益也可能回落；如果铜销售增加主要来自去库存，不能外推矿山长期供给能力。以上尚未证明9月发生了相同变化，7月已知的复产计划也不构成本轮新发现的市场惊喜。

**把当前盈利门槛摆出来，避免用过去高增长代替未来比较。**

| 公司 | 供应商2026年机构数 | 全年净利润均值/亿元 | H1归母实际/亿元 | 条件所需H2/亿元 | H2相对H1 |
| --- | ---: | ---: | ---: | ---: | ---: |
{proxy_table}

以上为9月29日取得的[宁德预测页面](https://basic.10jqka.com.cn/300750/worth.html)与[紫金预测页面](https://basic.10jqka.com.cn/601899/worth.html)，六个月内报告的供应商汇总；不是9月每家机构重新评估后的同口径市场共识。供应商“净利润”与公司归母可比是假设，表中H2仅为条件算式。原件只核对各一机构，不能用它代表全部机构。两个目录覆盖不同，不能因本次研报目录未列9月报告就称9月没有更新。

**对510300的实际用途。**

宁德、紫金8月末权重合计5.77%。即使假设两股各上涨10%、其他成分不变，静态量级也只有约0.577个百分点；这是股价情景算例，未假定盈利修正与股价一一对应，更不是ETF涨幅预测。研究这两家公司可以理解工业和材料压力，但还不足以单独建立完整ETF的方向优势。

下一步的轻量判断应同时看“修正方向”和“支撑构成”：宁德关注毛利与经营现金流的变化是否需要越来越多其他收益抵消；紫金关注毛利和产量假设是否兑现、恢复是否超过已含恢复的基准；再看相似变化是否扩散到其他重要权重。每项变化仍要考虑估值和价格吸收，不能仅按净利润预测升降投票。

本轮原F1/F2/F3、FIN1和BATT1不改，没有增加新预测卡、策略账户或收益测试。夏普1.2与净年化10%的目标仍未达到。**新增的是已核对的预测构成和可区分的经营原因，当前可成交价后的指数净优势仍未证明。**

来源范围说明：最初目录使用stockCode参数，实际返回全市场条目，已保留为不可用于选样的响应；改用code参数后核对证券身份再选原件。华龙原件落款3月24日、8月25日，与供应商目录3月25日、8月28日不同，分别保存，未将落款冒充已核实的最早公开时间。公司公告目录本次只返回30条，未声称全覆盖。仅检查了关键原件页面、预测表算式与数据日期，没有扩展为冗长审计。
"""
    (OUT / report_name).write_text(report, encoding="utf-8")
    result = {
        "study_id": "510300_COMPANY_REVISION_DRIVER_V1", "recorded_at": STAMP,
        "previous_goal_turn_classification": "PROGRESS_BROAD_INDEX_PRICE_PRESSURE_AND_COMPANY_UNIT_ECONOMICS",
        "continuation_classification": "PROGRESS_PAIRED_FORECAST_COMPONENT_REVISIONS_AND_OPERATING_BASELINE",
        "status": "FORECAST_COMPOSITION_IDENTIFIED_INDEX_RETURN_EDGE_UNPROVEN",
        "concrete_progress": ["同机构原件配对后分解宁德盈利维持背后的毛利和现金流变化", "量化紫金盈利上调依赖毛利率假设", "补充卡莫阿29至31万吨经营指引及已含恢复门槛", "建立两家公司当期供应商预测比较基准"],
        "new_accounts": 0, "new_strategy_return_tests": 0, "new_forecast_cards": 0,
        "existing_account_counts": {"admitted": 800, "executed": 952}, "existing_forecasts_unchanged": True,
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "goal_achieved": False, "goal_status": "active", "not_a_blocked_turn": True, "orders_authorized": False,
        "report": report_name, "next_question": "观察盈利修正是否从毛利和现金流改善扩散到更多指数权重，并将9月30日宏观发布与原F1/F2比较；经营恢复须超过已包含恢复的基准。",
    }
    save(OUT / "result.json", result)
    rel = OUT.relative_to(ROOT).as_posix()
    p = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(p)
    protocol.update({"latest_concrete_diagnostic": rel + "/result.json", "latest_paired_forecast_revision": rel + "/reviewed_facts.json",
                     "latest_company_forecast_proxy_extension": rel + "/current_vendor_forecast_baseline.json",
                     "forecast_composition_rule": "归母预测不变仍需观察毛利、费用、投资、减值和现金流构成；预测金额差额不等于实际经济原因贡献。",
                     "recovery_expectation_rule": "产量恢复和成本改善需比较既有指引已包含的恢复路径；生产阶段、销售去库存和新开采量分开。", "updated_at": STAMP})
    save(p, protocol)
    p = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(p)
    mandate.update({"current_round": result["study_id"], "latest_progress_receipt": rel + "/result.json", "latest_continuation_report": rel + "/" + report_name,
                    "latest_continuation_classification": result["continuation_classification"], "latest_paired_forecast_revision": rel + "/result.json",
                    "last_research_result": "宁德同机构归母预测不变但毛利预测减少99.95亿元、经营现金流减少92.50亿元；紫金归母上调72.38亿元主要对应毛利假设提高。卡莫阿29至31万吨指引已包含H2恢复。ETF净优势未证。",
                    "last_source_result": "四份同机构配对研报PDF、两个供应商快照、三个项目方经营更新已保存；目录过滤错误和发布时间差异已明确，未伪造历史共识。",
                    "latest_driver_diagnostic_at": STAMP, "next_research_question": result["next_question"]})
    save(p, mandate)
    p = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- COMPANY_REVISION_DRIVER_V1_20260929 -->

## 2026-09-29 盈利预测构成与经营恢复基准

取得四份同机构配对原件，宁德归母预测不变而毛利预测减少99.95亿元、经营现金流减少92.50亿元，其他项目变化抵消；紫金归母预测上调72.38亿元主要对应毛利假设提高。补充艾芬豪卡莫阿29至31万吨指引，保留紫金半年报29至33万吨原文并区分范围；恢复已在指引中。新增两股当前预测代理，不将其当历史共识。原预测卡不改，账户和收益测试新增均0，旧800/952与失败不变。目标active且未实现，本轮PROGRESS。

详见[盈利预测为何改变及剩余预期差](<{(OUT / report_name).as_posix()}>)。

"""
    p.write_text(head + p.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"报告": str(OUT / report_name), "宁德毛利预测变动亿元": catl_components[0]["delta_cny_100million"],
                      "宁德归母预测变动亿元": catl_reported_delta, "紫金归母预测变动亿元": zijin_reported_delta,
                      "卡莫阿指引要求H2产量吨": mine_h2_required, "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
