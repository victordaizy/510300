"""连接行业价格、公司量价成本和税制时点，形成可推翻的经营判断。"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_broad_index_driver_bridge_v1"
STAMP = datetime.now(timezone(timedelta(hours=8))).isoformat()
BASE, END = "2026-08-31", "2026-09-28"
WEIGHT_PATH = ROOT / "reports/research/510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json"
PREVIOUS_PATH = ROOT / "reports/research/510300_current_price_transmission_v1/reviewed_facts.json"
CATL_URL = "https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-07-25/8e6750da-b178-4be9-a74a-4b1d8fc42c58.PDF"
ZIJIN_URL = "https://www.zjky.cn/upload/file/2026/08/21/e978b5218f824602a4db3f5b20f4dd6f.pdf"
TAX_URL = "https://szs.mof.gov.cn/zhengcefabu/202607/t20260717_3993743.htm"
QA_URL = "https://www.chinatax.gov.cn/chinatax/c102414/c5252006/content.html"
PLAN_URL = "https://www.miit.gov.cn/gyhxxhb/jgsj/dzxxsnew/zcwj/art/2026/art_6c2161b398414389a1bcc655d2113fa0.html"
DIVIDEND_URL = "https://static.cninfo.com.cn/finalpage/2026-08-03/1225456323.PDF"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(after, before):
    return (after / before - 1) * 100


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮已完成，不覆盖结果或重新登记预测。")
    scope = read(OUT / "scope.json")
    source = read(OUT / "source_result.json")
    prior = read(PREVIOUS_PATH)
    weights = read(WEIGHT_PATH)
    previous_rows = {row["symbol"]: row for row in prior["prices"]}
    sector_weights = dict(weights["sector_weights_pct"])
    sector_weights["金融和房地产"] = sector_weights.pop("金融") + sector_weights.pop("房地产")
    dates = [row["date"] for row in source["responses"]["sh000908_raw"]["rows"]]
    if dates[0] != BASE or dates[-1] != END or len(set(dates)) != len(dates):
        raise ValueError("固定窗口端点或日期唯一性不符。")

    def prices(symbol, mode="raw"):
        value = source["responses"].get(symbol + "_" + mode)
        if value is None:
            return None
        rows = value["rows"]
        if [row["date"] for row in rows] != dates or any(row["close"] <= 0 for row in rows):
            raise ValueError(f"{symbol}的{mode}价格日期或数值不符。")
        return rows

    def secondary_close(symbol, end_price):
        quote = source["responses"]["sina_previous_close"].get(symbol)
        if not quote or quote["date"] != "2026-09-29" or quote["previous_close"] <= 0:
            return {"status": "NOT_AVAILABLE", "reason": "次日昨收字段无效，不把零价格当真实收盘。", "quote": quote}
        if abs(end_price - quote["previous_close"]) > 0.01:
            raise ValueError(f"{symbol}有效次日昨收与日线不一致。")
        return {"status": "AGREES_WITHIN_QUOTE_PRECISION", "quote": quote}

    sector_rows = []
    for code, name in scope["sectors"].items():
        symbol = "sh" + code
        rows = prices(symbol)
        first, prev, last = rows[0], rows[-2], rows[-1]
        period = pct(last["close"], first["close"])
        w0 = sector_weights[name]
        sector_rows.append({
            "symbol": symbol, "name": name, "start_close": first["close"], "previous_close": prev["close"],
            "end_close": last["close"], "start_weight_pct": w0, "period_price_change_pct": period,
            "period_fixed_weight_contribution_pp": w0 * period / 100,
            "last_session_price_change_pct": pct(last["close"], prev["close"]),
            "previous_session_drifted_weight_numerator": w0 * prev["close"] / first["close"],
            "secondary_close": secondary_close(symbol, last["close"]),
        })
    drift_total = sum(row["previous_session_drifted_weight_numerator"] for row in sector_rows)
    for row in sector_rows:
        fraction = row["previous_session_drifted_weight_numerator"] / drift_total
        row["previous_session_drifted_weight_proxy_pct"] = fraction * 100
        row["last_session_drifted_weight_contribution_pp"] = fraction * row["last_session_price_change_pct"]

    stock_rows = []
    for item in weights["top10"]:
        code, name, weight = item["symbol"], item["name"], item["weight_pct"]
        symbol = ("sh" if code.startswith("6") else "sz") + code
        if symbol in previous_rows:
            old = previous_rows[symbol]
            row = {k: old[k] for k in ["symbol", "name", "start_close", "end_close", "raw_price_change_pct", "vendor_adjusted_price_change_pct", "last_session_price_change_pct"]}
            row["source"] = PREVIOUS_PATH.relative_to(ROOT).as_posix()
            row["secondary_close_status"] = "AGREES_WITHIN_QUOTE_PRECISION_PREVIOUS_ROUND"
        else:
            raw, adjusted = prices(symbol), prices(symbol, "qfq")
            quote_check = secondary_close(symbol, raw[-1]["close"])
            if quote_check["status"] != "AGREES_WITHIN_QUOTE_PRECISION":
                raise ValueError(f"{name}缺少有效昨收核对。")
            row = {
                "symbol": symbol, "name": name, "start_close": raw[0]["close"], "end_close": raw[-1]["close"],
                "raw_price_change_pct": pct(raw[-1]["close"], raw[0]["close"]),
                "vendor_adjusted_price_change_pct": None if adjusted is None else pct(adjusted[-1]["close"], adjusted[0]["close"]),
                "last_session_price_change_pct": pct(raw[-1]["close"], raw[-2]["close"]),
                "source": "source_result.json", "secondary_close_status": quote_check["status"],
            }
        row["weight_20260831_pct"] = weight
        row["period_fixed_weight_price_contribution_pp"] = weight * row["raw_price_change_pct"] / 100
        row["vendor_adjusted_is_exact_total_return"] = False
        stock_rows.append(row)

    selected = {row["name"]: row for row in sector_rows}
    period_total = sum(row["period_fixed_weight_contribution_pp"] for row in sector_rows)
    daily_total = sum(row["last_session_drifted_weight_contribution_pp"] for row in sector_rows)
    main_period = sum(selected[name]["period_fixed_weight_contribution_pp"] for name in ["原材料", "工业", "信息技术"])
    main_daily = sum(selected[name]["last_session_drifted_weight_contribution_pp"] for name in ["信息技术", "通信服务"])
    benchmark = previous_rows["sh000300"]
    top10_contribution = sum(row["period_fixed_weight_price_contribution_pp"] for row in stock_rows)
    catl_tax, catl_revenue = 1324939, 276916580
    tax_ratio_pct = catl_tax / catl_revenue * 100
    catl_income = read(OUT / "sources/catl_h1_income_pages.json")
    consolidated = next(row["text"] for row in catl_income if row["page"] == 77)
    if "1,324,939" not in consolidated or "276,916,580" not in consolidated:
        raise ValueError("宁德合并利润表基准未匹配原件。")

    forecast = {
        "id": "BATT1", "recorded_at": STAMP, "company": "300750.SZ", "target_period": "2026Q3单季",
        "statement": "宁德时代2026年第三季度单季合并税金及附加占营业收入比例，高于2026年上半年的同口径比例。",
        "h1_baseline": {"currency": "CNY", "unit": "千元", "taxes_and_surcharges": catl_tax, "revenue": catl_revenue, "ratio_pct": tax_ratio_pct},
        "measure": "(三季报合并1至9月税金及附加－1至6月税金及附加)/(三季报合并1至9月营业收入－1至6月营业收入)*100",
        "evaluation": {"true": "Q3单季比例严格大于H1同口径比例", "false": "Q3单季比例小于或等于H1同口径比例", "not_scorable": "任一同口径合并项目缺失、分母非正或无法处理口径变化"},
        "restatement_rule": "如三季报明确重述H1数据，另存原始及重述值，用一致口径重新计算比较基准并披露；不更换指标、季度或方向。",
        "rationale": "9月1日部分电池开始征收消费税；公司H1国内业务占比较高，但实际税基、抵扣、出口结构及客户分担尚未量化。",
        "alternatives": ["出口与免税产品结构变化", "外购已税原料扣除", "确认时点", "其他税金附加的变化抵消"],
        "identification_limit": "税金及附加包含其他项目；方向命中不能单独识别消费税的因果贡献或实际利润损失。",
        "timing_limit": "9月29日登记，季度大部分已经发生；属于发布前经营判断，不是整季事前预测。",
        "market_consensus_for_this_metric": None, "stock_return_prediction": False,
        "probability": None, "status": "PENDING_PUBLICATION", "source_url": CATL_URL,
    }
    save(OUT / "company_tax_forecast_card.json", forecast)
    facts = {
        "recorded_at": STAMP, "base_close": BASE, "end_close": END, "previous_session": dates[-2],
        "dates_per_series": len(dates), "sectors": sector_rows, "top10_stocks": stock_rows,
        "rounded_initial_sector_weight_sum_pct": sum(sector_weights.values()),
        "period_approximate_contribution_sum_pp": period_total,
        "index_period_price_change_pct": benchmark["raw_price_change_pct"],
        "period_approximation_minus_index_pp": period_total - benchmark["raw_price_change_pct"],
        "materials_industrials_information_period_contribution_pp": main_period,
        "last_session_approximate_contribution_sum_pp": daily_total,
        "index_last_session_price_change_pct": benchmark["last_session_price_change_pct"],
        "information_communication_last_session_contribution_pp": main_daily,
        "top10_fixed_initial_weight_contribution_pp": top10_contribution,
        "method": {
            "period": "8月末四舍五入权重百分数×行业价格变化百分数/100；100.1%权重不为贴合指数重新归一化。",
            "daily": "8月末权重乘截至9月24日价格比例后归一化，再乘9月24日至28日行业涨跌幅；仅为权重漂移近似。",
            "limits": "两者均未逐股重建样本调整、股本、自由流通量及指数维护，不是官方精确贡献，更不是宏观原因的因果份额。",
            "scope": "000914已合并金融与房地产，不再叠加房地产指数。",
            "vendor_coverage": "10个行业日线齐全；仅2个行业有有效次日昨收核对，另外8个保留未交叉确认。新增5股昨收全部一致。宁德前复权接口失败，保留缺失。",
        },
        "catl": {
            "h1_revenue_cny_thousand": catl_revenue, "h1_parent_net_profit_cny_thousand": 43284002,
            "h1_operating_cashflow_cny_thousand": 60216851, "h1_operating_cashflow_yoy_pct": 2.61,
            "power_battery_revenue_yoy_pct": 46.02, "power_battery_gross_margin_pct": 20.63,
            "power_battery_gross_margin_change_pp": -1.78, "storage_revenue_yoy_pct": 87.54,
            "storage_gross_margin_pct": 23.96, "storage_gross_margin_change_pp": -1.56,
            "overseas_revenue_share_pct": 31.46, "h1_capacity_utilization_pct": 94.86,
            "overseas_revenue_is_taxable_export_base": False, "h1_operating_metrics_are_september_metrics": False,
            "unit_profit_tax_note": "合并利润表中税金及附加与营业成本单列；不能把2%消费税直接说成毛利率必降2个百分点。",
            "known_dividend_ex_date": "2026-08-10", "known_dividend_in_price_window": False,
            "complete_corporate_action_coverage_claimed": False, "source_url": CATL_URL,
        },
        "battery_policy": {
            "announcement_published": "2026-07-17", "consumption_tax_effective": "2026-09-01",
            "initial_rate_pct_for_covered_products": 2, "next_rate_pct": 4, "next_rate_date": "2027-09-01",
            "vat_export_rebate_rate_from_20260401_pct": 6, "vat_export_rebate_cancellation_date": "2027-01-01",
            "vat_details_source": QA_URL, "original_vat_notice_fetch_status": "HTTP_502_WEB_TIMEOUT_NOT_RETRIED",
            "tax_base_and_relief": "应税产品、内外销、原料已纳税扣除、特定产品免税认证分别核对；出口消费税退免与增值税出口退税不是同一税。",
            "timing_hypothesis": "2027年初出口退税取消可能激励2026年末提前合规交付报关；未证明已发生或能带来长期终端需求增长。",
            "new_plan_page_published_at": "2026-09-28T15:07:00+08:00", "new_plan_document_date": "2026-09-14",
            "plan_clock_limit": "已取得页面晚于当日收盘，不能用它证明盘中因果；其他渠道首次发布时间未核实。",
            "new_plan_url": PLAN_URL, "tax_source_url": TAX_URL,
        },
        "zijin": {
            "h1_mined_gold_kg": 46702, "h1_mined_gold_yoy_pct": 13.4,
            "h1_disclosed_mined_copper_tonnes": 534407, "h1_disclosed_mined_copper_yoy_pct": -5.7,
            "h1_mined_copper_excluding_kamoa_tonnes": 480032, "h1_mined_copper_excluding_kamoa_yoy_pct": 4.8,
            "copper_scope": "公司口径包括卡莫阿权益产量，不能当作全部矿山100%实物产量。",
            "kamoa_annual_plan_old_tonnes": [380000, 420000], "kamoa_annual_plan_revised_tonnes": [290000, 330000],
            "company_copper_plan_impact_tonnes": [-57000, -22000],
            "copper_concentrate_sale_price_cny_per_tonne": 82624,
            "copper_concentrate_unit_cost_cny_per_tonne": 24480,
            "copper_concentrate_unit_cost_yoy_pct": 16, "copper_concentrate_gross_margin_pct": 70.37,
            "product_table_scope": "价格成本产品表不含非控股公司数据；不与含权益产量的总量口径直接相乘。",
            "cost_reasons_from_company": ["矿石品位变化", "柴油药剂", "运输距离", "矿权出让收益金", "副产品列报口径"],
            "taxes_and_surcharges_yoy_pct": 67.48, "tax_reason_from_company": "主要为资源税增加",
            "september_gold_copper_and_real_yield_shock_measured": False,
            "h1_disclosure_date": "2026-08-21", "source_url": ZIJIN_URL,
        },
        "current_executable_etf_price_obtained": False,
        "mechanism_inference": "当前重点是税后单位盈利和复产约束能否改善；报告提供可区分的路径，不把阶段跌幅全部归因于这些旧信息。",
    }
    save(OUT / "reviewed_facts.json", facts)

    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    matplotlib.rcParams["font.family"] = font.get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    ordered = sorted(sector_rows, key=lambda row: row["period_fixed_weight_contribution_pp"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.1), sharey=True)
    fields = ["period_fixed_weight_contribution_pp", "last_session_drifted_weight_contribution_pp"]
    titles = ["8月31日 → 9月28日｜月初固定权重", "9月24日 → 9月28日｜前日漂移权重近似"]
    for ax, field, title in zip(axes, fields, titles):
        values = [row[field] for row in ordered]
        bars = ax.barh(range(len(ordered)), values, color=["#af4d40" if v < 0 else "#318674" for v in values], height=0.62)
        ax.bar_label(bars, labels=[f"{v:+.2f}" for v in values], padding=5, fontsize=10)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_yticks(range(len(ordered)), [row["name"] for row in ordered])
        ax.axvline(0, color="#6f7982", lw=0.7)
        ax.grid(axis="x", color="#e6eaee", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("对指数价格变化的近似贡献（百分点）")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlim(min(values) * 1.20, max(0.18, max(values) * 2.0))
    axes[0].invert_yaxis()
    fig.suptitle("阶段不同，指数压力的主要来源也不同", fontsize=17, x=0.11, ha="left", y=0.97)
    fig.text(0.11, 0.06, "行业完整覆盖且不重叠；左右横轴尺度不同。8月末权重因四舍五入合计100.1%。\n权重×价格仅用于量级分析，未逐股重建指数维护，也不识别政策或宏观因素的因果贡献。", fontsize=10, color="#57626d")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.85, bottom=0.19, wspace=0.16)
    chart_name = "行业压力的阶段差异.png"
    fig.savefig(OUT / chart_name, dpi=170, facecolor="white")
    plt.close(fig)

    sector_table = "\n".join(f"| {row['name']} | {row['start_weight_pct']:.1f}% | {row['period_price_change_pct']:+.2f}% | {row['period_fixed_weight_contribution_pp']:+.3f} | {row['last_session_drifted_weight_contribution_pp']:+.3f} |" for row in ordered)
    stock_table = "\n".join(f"| {row['name']} | {row['weight_20260831_pct']:.2f}% | {row['raw_price_change_pct']:+.2f}% | {row['period_fixed_weight_price_contribution_pp']:+.3f} |" for row in stock_rows)
    report_name = "全行业压力与下一阶段驱动.md"
    report = f"""# 从行业压力追到量、价、成本、税负和预期修正

记录时间：{STAMP}。价格窗口沿用2026年8月31日至9月28日收盘；最后一交易日为9月24日至28日。研究先按完整行业组和既定前十大权重覆盖，再对工业、原材料的重要公司追原因。选择宁德和紫金是在看到行业表现之后，属于知情诊断，不是事前选股或收益检验。

**判断已推进到具体约束：工业与材料的盈利压力不能被最新一日的科技政策故事代替；宁德应看销量能否覆盖单位利润及税负变化，紫金应看金属价格能否覆盖矿山产量和成本变化。由这些约束产生下一项可推翻的判断，再与市场既有预期比较。**

**先定位问题发生在哪里。**

| 行业 | 8月末权重 | 全窗口价格变化 | 全窗口近似贡献/百分点 | 最后一日近似贡献/百分点 |
| --- | ---: | ---: | ---: | ---: |
{sector_table}

工业、原材料、信息技术全窗口合计{main_period:+.3f}个百分点，约占同期指数净跌幅的{main_period / benchmark['raw_price_change_pct'] * 100:.1f}%；信息技术与通信服务最后一日合计{main_daily:+.3f}个百分点，约占该日行业近似净贡献的{main_daily / daily_total * 100:.1f}%。这是价格压力分布，不是上述行业的某项基本面原因已经解释了相同比例跌幅。

行业全窗口近似合计{period_total:+.4f}个百分点，沪深300价格指数实际{benchmark['raw_price_change_pct']:+.4f}%，两者接近不等于正式归因成立。全窗口使用8月末四舍五入权重，合计100.1%，未为贴近结果调权重；最后一日则先按价格漂移估计9月24日权重并归一化，未用9月28日权重回填。样本、股本及指数维护未逐股重建。000914同时包括金融、房地产，未与房地产另行叠加。资料见[官方金融地产指数说明](sources/000914_factsheet.pdf)和[8月末权重依据](../510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json)。

![行业压力图](<{(OUT / chart_name).as_posix()}>)

10个行业各20条日线，端点和日期一致；仅医药、金融地产有有效次日昨收核对，另外8个供应商字段为过期日期或零，保留未交叉确认。新增5股昨收均核对一致。行业原始日线来自行情供应商，权重来自指数公司，未冒充交易所全量成交数据。

**前十大权重补齐后，也不能只讲个别明星公司的故事。**

| 公司 | 8月末权重 | 全窗口原价变化 | 固定权重近似贡献/百分点 |
| --- | ---: | ---: | ---: |
{stock_table}

十股合计23.18%，原价近似贡献{top10_contribution:+.3f}个百分点。这里只给价格指数量级，个股不构成新交易品种。平安窗口内派息、药明供应商复权序列有差别，不能把原价跌幅当持有人总收益；平安的现金股息已在[上一轮](../510300_current_price_transmission_v1/价格反应与新增信息.md)单列。宁德前复权请求失败，继续保留缺失。已查到的宁德中期派息除息日为8月10日，不在本窗口；这笔分红不能解释9月价格变化，但也不借此宣称穷尽所有公司行为。[宁德分红原件]({DIVIDEND_URL})，PDF第2页。

**宁德时代：需求增长、成本传导、税负与收入确认时点要一起看。**

上半年动力电池收入增长46.02%，毛利率20.63%、下降1.78个百分点；储能收入增长87.54%，毛利率23.96%、下降1.56个百分点。产能利用率94.86%是H1数据，不是9月订单证明。现金流同比仅增2.61%，本轮未完整识别其具体回款及付款构成，不能直接称回款恶化。公司提到长协、资源布局、套期保值和价格联动管理原料成本；能转嫁多少仍取决于合同与客户议价。[宁德半年报]({CATL_URL})，PDF第15—17、29页。

下一阶段至少有三条不同路径：终端交付量增加；产品结构、售价与材料成本改变单位利润；税制及内外销结构改变税后收益。只见收入加速，无法推断后两项同步改善。

7月17日已经公布的部分电池消费税，9月1日起按2%执行，2027年9月1日起升至4%。这不是9月新公布的意外。外购已税原料扣除、自产自用、出口退免及特定产品认证条件会改变实际税负；不能把合并总收入乘2%作为损失，也不能把境外收入31.46%直接当符合某项出口税收口径的税基。消费税列在“税金及附加”，本报告合并利润表与营业成本单列，不能说毛利率机械下降2个百分点。[财政部公告]({TAX_URL})、[税务总局问答]({QA_URL})。

另一条时间线是增值税出口退税：问答第7项确认，指定电池产品2026年4月1日起从9%降为6%，2027年1月1日起取消；出口消费税退免与此是不同税种。年末提前交付报关是可能的行为反应，尚未证明实际发生。若以后看到四季度出货加速，需要区分终端需求增加与订单前移，后者不能外推成2027年的持续增长。原增值税公告独立链接本次获取失败，保留失败，现有日期依据为税务总局正式问答。

工信部新规划同时涉及应用拓展、产能预警和价格竞争规范，政策支持的现金与订单落地仍需观察。已取得页面发布时间为9月28日15:07，晚于当日A股收盘；成文日期9月14日不等于市场当时已知，其他渠道更早披露尚未核实，不能拿这张页面去解释当日盘中价格。[规划原文]({PLAN_URL})。

本轮登记一条经营判断BATT1：**宁德2026Q3单季“合并税金及附加/营业收入”高于H1的{tax_ratio_pct:.4f}%。** 以三季报1—9月减去1—6月计算单季。H1两项目分别为1,324,939和276,916,580千元，来自PDF第77页合并利润表；第79页是母公司，不能混用。出口、抵扣、结构与其他税金变化都可能推翻判断。季度大部分已经发生，因此这是结果公布前的经营判断；不是整季事前预测，也不是股价预测。税负比例命中同样不能识别消费税的净因果贡献。规则保存于[原始判断卡](company_tax_forecast_card.json)。

**紫金矿业：金属涨价只是乘法的一项，矿山恢复是另一项。**

H1公司披露矿产铜53.44万吨、同比下降5.7%；剔除卡莫阿后为48.00万吨、增长4.8%。卡莫阿淹井及复产爬坡使其全年计划由38—42万吨调至29—33万吨，公司披露对自身矿产铜计划影响约减少2.2—5.7万吨。总体产量口径含卡莫阿权益产量，不能当全部矿山100%实物产量。[紫金半年报]({ZIJIN_URL})，PDF第10页。

铜精矿售价由60,354升至82,624元/吨，单位成本也由21,104升至24,480元/吨、上升16%。公司将成本变化联系到矿石品位、柴油药剂、运距、矿权出让收益金及副产品列报；毛利率仍升至70.37%，是价格与成本共同作用的结果。税金及附加增长67.48%，公司解释主要为资源税增加。产品价成本表不含非控股公司，不能直接乘前述含权益产量的总数测利润。来源：同一半年报PDF第13—15页。

未来判断应拆成：全球需求和矿山供给如何改变金属价格；复产进度如何改变可售数量；品位、能源、运输如何改变单位成本；税费、少数股东权益等如何影响归母收益。半年报8月已公开，9月看到旧数字不构成新惊喜。本轮未取得新的金铜现货、实际利率或市场矿产量预期，不能把9月跌幅归结为已识别的某个金属价格冲击，也不能由“股价跌得多”断定低估。

**未来1至4周，用改变盈利路径的证据更新判断。**

| 新信息及其原因 | 更可能改变什么 | 推翻或削弱原判断的事实 | 如何进入510300判断 |
| --- | --- | --- | --- |
| 电池交付增长同时伴随单位利润稳定，新增税负可转嫁或被效率抵消 | 盈利增长质量可能改善 | 仅靠降价放量，或出口前移而后续订单减少 | 先比较已有盈利预期，不能把销量增长再当额外独立利好 |
| 电池税负增加且客户不承担，产品价格竞争仍强 | 收入高增长仍可能伴随利润率压力 | 实际免抵税、产品结构或价格分担使净负担很小 | 改变工业部分盈利路径；当前价格已反映多少仍要比较 |
| 铜矿复产快于已有计划、品位或成本改善 | 同样金属价格下可售量和利润增加 | 复产延误、成本上升或金属价格走弱 | 材料业盈利能否改善，需要和下游成本压力一起看 |
| 大宗价格上涨来自终端需求恢复 | 资源端量价可能同升，下游有机会转嫁 | 新订单不增，涨价主要来自供给受损 | 对指数影响取决于资源利润增量与下游利润挤压，不能直接看商品方向 |
| 大宗价格上涨来自供给扰动 | 上游有价格收益，下游承受成本；受损矿山自身也可能减产 | 供给迅速恢复或下游有强议价能力 | 同一商品因子可产生相反公司效应，应查约束实际落在哪一端 |
| 跨季资金利率回落，但订单、信用需求与公司指引未变 | 资金成本改善，盈利驱动未必跟随 | 订单、贷款需求、利润率同步超出原预期 | 只记录一次流动性改善，不把与其同源的多个价格当多份证据 |

这些是条件规律的候选，不是已经验证的稳定超额收益。新的实用重点是“造成当前压力的约束是否松动、松动是否超过原预期”：信息技术和通信需要政策范围与订单新证据；工业需要税后单位盈利；材料需要量价成本；金融与实体需求继续看原F1/F2/F3及FIN1。行业贡献只是决定研究注意力，不按跌幅临时挑因子、拼接交易阶段。

这轮没有跑长历史检验、收益参数网格或新策略账户。原宏观预测和FIN1不改，BATT1单独登记。**夏普1.2仍未实现，当前价后的ETF相对现金净收益优势尚未识别；本轮净夏普、净年化收益均未计算。** 现在已明确后续最该追的变量及其原因，不用增加大量指标来代替缺失的预期修正证据。
"""
    (OUT / report_name).write_text(report, encoding="utf-8")
    result = {
        "study_id": "510300_BROAD_INDEX_DRIVER_BRIDGE_V1", "recorded_at": STAMP,
        "previous_goal_turn_classification": "PROGRESS_PRICE_RESPONSE_POLICY_SCOPE_AND_COMPLETED_BUYBACK",
        "continuation_classification": "PROGRESS_BROAD_INDEX_PRICE_PRESSURE_AND_COMPANY_UNIT_ECONOMICS",
        "status": "DRIVER_PRIORITIES_IDENTIFIED_EXPECTATION_REVISION_AND_INDEX_EDGE_UNPROVEN",
        "concrete_progress": ["补齐十个不重叠行业与前十大权重价格", "区分月度和最后一日压力", "拆开宁德收入毛利税负及出口时点", "拆开紫金金属价格产量复产成本及资源税", "登记BATT1发布前经营判断"],
        "new_accounts": 0, "new_strategy_return_tests": 0, "strategy_rule_created": False,
        "existing_account_counts": {"admitted": 800, "executed": 952},
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "existing_forecasts_unchanged": True, "new_forecast_card": "company_tax_forecast_card.json",
        "current_index_return_forecast_made": False, "goal_achieved": False, "goal_status": "active",
        "not_a_blocked_turn": True, "orders_authorized": False, "report": report_name,
        "next_question": "宁德的税后单位盈利、紫金的产量成本，是否出现相对市场既有预期的真实修正；9月30日宏观新信息按原F1/F2核对，不重写。",
    }
    save(OUT / "result.json", result)
    rel = OUT.relative_to(ROOT).as_posix()
    p = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(p)
    protocol.update({
        "latest_concrete_diagnostic": rel + "/result.json", "latest_broad_index_driver_bridge": rel + "/reviewed_facts.json",
        "latest_company_tax_forecast": rel + "/company_tax_forecast_card.json",
        "unit_earnings_rule": "销量、售价、单位成本、税负与内外销范围分开；增长不等于单位利润改善，税金及附加不混作毛利。",
        "policy_time_rule": "成文、公开、生效和执行时点分开；晚于收盘的已取得网页不能用来证明当日盘中因果。",
        "sector_coverage_rule": "完整行业覆盖避免热点股替代整个指数；权重漂移近似与官方精确归因分开，不把价格贡献当原因贡献。",
        "updated_at": STAMP,
    })
    save(p, protocol)
    p = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(p)
    mandate.update({
        "current_round": result["study_id"], "latest_progress_receipt": rel + "/result.json",
        "latest_continuation_report": rel + "/" + report_name,
        "latest_continuation_classification": result["continuation_classification"],
        "latest_broad_index_driver_bridge": rel + "/result.json", "latest_company_tax_forecast": rel + "/company_tax_forecast_card.json",
        "last_research_result": "工业、材料、信息技术约贡献阶段-4.663个百分点，最后一日信息与通信约-1.701个百分点；补齐宁德量价税负和紫金产量成本原因，新增BATT1，剩余ETF净优势未证。",
        "last_source_result": "行业及五股日线、指数资料、宁德和紫金财报、电池税制与规划原件已保存；8个行业昨收无效、宁德前复权失败及原出口退税公告失败均保留。",
        "latest_driver_diagnostic_at": STAMP, "next_research_question": result["next_question"],
    })
    save(p, mandate)
    p = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- BROAD_INDEX_DRIVER_BRIDGE_V1_20260929 -->

## 2026-09-29 行业压力与单位盈利原因

完整行业量级分析显示工业、原材料、信息技术全窗口近似贡献{main_period:+.3f}个百分点，最后一日信息与通信约{main_daily:+.3f}个百分点。拆开宁德的交付、单位利润、税负和出口时点，以及紫金的价格、复产、品位成本和资源税；旧信息不冒充新惊喜，价格贡献不冒充因果贡献。新增BATT1发布前经营判断，原F1/F2/F3及FIN1不改。新增账户0、收益测试0，旧800/952及失败不变，目标active且未实现，本轮PROGRESS。

详见[全行业压力与下一阶段驱动](<{(OUT / report_name).as_posix()}>)。

"""
    p.write_text(head + p.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"报告": str(OUT / report_name), "阶段三行业近似贡献百分点": main_period,
                      "最后一日两行业近似贡献百分点": main_daily, "前十大近似贡献百分点": top10_contribution,
                      "BATT1基准百分比": tax_ratio_pct, "状态": result["status"], "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
