"""整理已读取的原因、公司财务和价格；不重新联网、不生成交易账户。"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_weight_company_driver_bridge_20260929"
REPORT_NAME = "上游支出原因与权重公司预期差.md"
STUDY = "510300_WEIGHT_COMPANY_DRIVER_BRIDGE_20260929"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("本轮结论已经保存；新增证据应另存记录，不覆盖本轮判断。")
    recorded_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    prices = read_json(OUT / "prices/result.json")
    weights = read_json(ROOT / "reports/research/510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json")
    price_by_symbol = {r["symbol"][2:]: r for r in prices["comparisons"]}
    weight_by_symbol = {r["symbol"]: r["weight_pct"] for r in weights["top10"]}
    companies = [
        {
            "symbol": "300308", "name": "中际旭创", "source_file": "innolight_h1.pdf", "financial_pdf_page": 8,
            "cause_pdf_pages": [23, 25, 26, 27, 31],
            "revenue": 41777861795.03, "prior_revenue": 14789074837.80,
            "parent_profit": 13651149693.27, "prior_parent_profit": 3995115384.70,
            "operating_cashflow": 1799674832.02, "prior_operating_cashflow": 3218463287.30,
            "inventory": 19825909263.51, "previous_year_end_inventory": 12680707624.68,
            "receivables": 15000893993.58, "previous_year_end_receivables": 6276876022.07,
            "reported_cause": "800G/1.6T出货和产品结构推动收入利润；经营现金流下降主要因采购商品付款增加。",
            "next_discriminator": "高速产品交付和良率能否兑现备货，同时观察应收回款、存货消化和利润率；不能只看收入同比。",
        },
        {
            "symbol": "300502", "name": "新易盛", "source_file": "eoptolink_h1.pdf", "financial_pdf_page": 7,
            "cause_pdf_pages": [13, 14, 15],
            "revenue": 20909746962.20, "prior_revenue": 10437170265.50,
            "parent_profit": 7529168039.39, "prior_parent_profit": 3942294268.37,
            "operating_cashflow": 1616408524.15, "prior_operating_cashflow": 952686853.97,
            "inventory": 11655392245.35, "previous_year_end_inventory": 7234333468.99,
            "receivables": 7548269870.50, "previous_year_end_receivables": 4437509481.09,
            "reported_cause": "销售回款和税费返还增加经营现金流；按在手及预期订单、备货周期增加库存；汇兑损失影响财务费用。",
            "next_discriminator": "800G、1.6T交付和产能效率是否高于已公开的下半年放量指引；同步看单价、成本、汇兑及回款。",
        },
        {
            "symbol": "688256", "name": "寒武纪", "source_file": "cambricon_h1.pdf", "financial_pdf_page": 7,
            "cause_pdf_pages": [8, 19, 20, 21, 22, 23],
            "revenue": 5995573619.61, "prior_revenue": 2880643471.09,
            "parent_profit": 2310912080.33, "prior_parent_profit": 1038082568.57,
            "operating_cashflow": 311314642.46, "prior_operating_cashflow": 911150321.73,
            "inventory": 8247509484.00, "previous_year_end_inventory": 4943532510.26,
            "receivables": 222016836.34, "previous_year_end_receivables": 670643223.27,
            "prepayments": 2913927071.43, "previous_year_end_prepayments": 744539708.47,
            "reported_cause": "采购和税金支出压低经营现金流；收回旧应收款，同时预付采购、原材料及委托加工物资增加。",
            "next_discriminator": "芯片供应、软件适配、客户验收和采购款转为交付的效率；不能将北美光模块需求直接映射为该公司订单。",
        },
    ]
    for c in companies:
        p = price_by_symbol[c["symbol"]]
        c["financial_period"] = "2026H1与2025H1同一中报列示口径"
        c["balance_comparison"] = "2026-06-30与2025年末；存量变化，不是当期现金流逐项归因"
        c["currency_unit"] = "CNY"
        c["weight_pct_asof_20260831"] = weight_by_symbol[c["symbol"]]
        c["close_20260928"] = p["tencent_unadjusted_close"]
        c["price_crosscheck"] = p["status"]
        for metric in ("revenue", "parent_profit", "operating_cashflow"):
            c[metric + "_yoy_pct"] = (c[metric] / c["prior_" + metric] - 1) * 100

    total_weight = sum(c["weight_pct_asof_20260831"] for c in companies)
    sensitivity = {
        "assumptions_only_not_predictions": True,
        "same_forward_earnings_horizon_required": True,
        "earnings_forecast_revision_fraction": 0.10,
        "valuation_multiple_revision_fraction": -0.10,
        "illustrative_stock_price_change_fraction": 1.10 * 0.90 - 1,
        "three_stock_weight_pct_asof_20260831": total_weight,
        "other_constituents_unchanged": True,
        "fixed_weight_index_contribution_pp": total_weight * (1.10 * 0.90 - 1),
        "ten_percent_common_stock_rise_index_contribution_pp": total_weight * 0.10,
        "transaction_costs_and_etf_tracking_not_included": True,
    }
    facts = {
        "recorded_at": recorded_at,
        "companies": companies,
        "microsoft": {
            "fy26q3_calendar_2026_capex_usd_billion": 190,
            "fy26q3_component_price_impact_usd_billion": 25,
            "fy26q4_calendar_2026_capex_usd_billion": 175,
            "headline_change_fraction": 175 / 190 - 1,
            "cause": "数据中心和办公楼预计使用年限15年改为25年，更多未来租赁从融资租赁转为经营租赁；后者不计入该资本支出口径。管理层称排除此影响后的投资预期未变。",
            "not_gpu_depreciation_life_change": True,
            "real_demand_evidence": "FY26Q4 Azure收入增长43%；公司称需求超过可用产能，效率和较早交付提升收入。",
            "forward_management_anchor": "FY27Q1 Azure恒定汇率收入增速约45%；属于公司指引，不是一致预期或真实采购数量。",
            "named_customer_relationship_for_selected_companies_confirmed": False,
            "raw_html_download": "HTTP_403_PRESERVED",
            "usable_source": "WEB_PRIMARY_TEXT_EXTRACT",
        },
        "eoptolink_september_guidance": {
            "meeting_date": "2026-09-11",
            "reported_current_year_delivery": ["800G", "1.6T"],
            "npo_batch_delivery_expected_year": 2027,
            "npo_larger_scale_expected_year": 2028,
            "is_management_expectation_not_verified_orders": True,
            "quarterly_unit_guidance_disclosed": False,
            "source": "eoptolink_september_ir.pdf，第1至3页",
        },
        "prices": prices["comparisons"],
        "price_snapshot_is_consensus_or_fair_value": False,
        "sensitivity": sensitivity,
    }
    observations = {
        "recorded_at": recorded_at,
        "window": {"start": "2026-09-29", "end": "2026-10-26"},
        "kind": "条件观察表，不是已量化的新增收益预测，也不是策略规则",
        "existing_macro_forecasts": "../510300_current_driver_outlook_20260929/forecast_cards.json",
        "existing_macro_forecasts_edited": False,
        "rows": [
            {"issue": "上游资本开支", "base_interpretation": "本次190至175的下调由口径变化解释，不能据此确认设备需求转弱。", "upward_evidence": "同口径设备数量、可用容量及实际需求上修，并可传递到相关供应商。", "downward_evidence": "实际采购取消或推迟、短寿命设备采购减少，同时排除单纯租赁分类和交付时点。"},
            {"issue": "光模块", "base_interpretation": "800G/1.6T景气和下半年放量已属公开指引；当前没有数量化超预期证据。", "upward_evidence": "相对发布前预期，交付数量、良率、利润率或回款出现正向新修正。", "downward_evidence": "交付推迟、单价或良率压低利润，或备货无法转成订单及回款。"},
            {"issue": "寒武纪", "base_interpretation": "采购和备货扩张与收入增长并存，现金流下降不单独代表客户违约。", "upward_evidence": "预付材料和委托加工转为可验收产品，交付及回款超出事前基准。", "downward_evidence": "供应、适配或客户验收迟延，库存减值或采购成本压缩利润。"},
            {"issue": "510300整体", "base_interpretation": "三家公司已知权重仅7.08%，其余成分与贴现率可抵消此方向。", "upward_evidence": "权重盈利预期的净上修及资金条件共同支持，且市场价格留下足够空间。", "downward_evidence": "盈利变化已被价格计入，或外部利率、国内需求及其他权重股抵消。"},
        ],
        "evaluation": "区分公开指引、市场事前预期和实际值；没有当期可比数据则保持未判断，不把未披露当零、不事后延长窗口。",
        "report_release_dates_verified": False,
        "note": "不承诺三季报均在此窗口内发布；窗口内优先使用新出现的可核实公告。",
    }
    rows = "\n".join(
        f"|{c['name']}|{c['revenue_yoy_pct']:+.2f}%|{c['parent_profit_yoy_pct']:+.2f}%|{c['operating_cashflow']/1e8:.2f}亿元（{c['operating_cashflow_yoy_pct']:+.2f}%）|{c['reported_cause']}|"
        for c in companies
    )
    report = f"""# 上游支出原因与权重公司预期差

记录于{recorded_at}。本轮结论是：**资本开支、现金流、库存这些因子必须按变化原因解释；已得到几项能排除错误方向的依据，但尚未证明当期价格中存在正的预期差，不能据此给510300建立买入规则。**

研究沿已选的中际旭创、新易盛、寒武纪展开，选择依据是8月末已知指数权重与需求链，不按后续股价筛选。这里只研究它们对510300的影响，不改变交易资产范围。本轮新增四份公司PDF、复核两份微软原站交流记录；没有扩大因子网格或做五年十年相关性筛选。

## 1. 资本开支下调，先追问到底改了什么

微软FY26Q3交流给出的2026日历年资本开支预期约1900亿美元，其中约250亿美元来自元器件涨价影响。因此金额增长本来就不能全部解释为设备数量增长，也不能简单相减就得到经核实的实物投资量。[微软FY26Q3原文](https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3)

FY26Q4交流把口径下的预期改为约1750亿美元，表面下降{abs(facts['microsoft']['headline_change_fraction']):.2%}。解释是数据中心和办公楼的预计使用年限由15年改为25年，使更多未来租赁改列经营租赁、从该资本支出口径排除；公司明确称排除此影响的投资预期未变。这里不是GPU使用年限调整。同时公司称需求仍超过可用产能，效率改善和较早交付也能提高收入。**这支持排除“金额下调就意味着设备需求下降”的推断，不证明未来所有采购都不会减少。**[微软FY26Q4原文](https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q4)

实际需要区分四件事：买多少设备、每单位多少钱、什么时候到货投入使用、以什么口径列账。长期预算还要经过采购、供应和部署约束，才能成为供应商收入。微软案例用于说明上游机制，本轮未证实微软与所选三家公司之间的具体客户订单关系。

## 2. 同是高增长，现金流变化原因不同

下表均为各公司2026半年报中的合并口径，收入和归母净利润同比，上半年经营现金流净额及同比。经营现金流与归母利润属于不同报表口径，未据两者比值制作“质量排名”。

|公司|收入同比|归母净利润同比|经营现金流净额及同比|原文给出的关键原因|
|---|---:|---:|---:|---|
{rows}

中际旭创期末存货198.26亿元、应收150.01亿元，分别高于上年末126.81亿元、62.77亿元。公司把收入利润增长与800G/1.6T出货、产品结构和效率联系起来，而经营现金流下降主要指向采购付款。后续需要看这些投入是否按期转换成交付和回款；不能从备货增加直接认定未来订单确定，也不能从现金流下降直接认定需求崩塌。[半年报第8、23、26—27页](https://static.cninfo.com.cn/finalpage/2026-08-22/1225491753.PDF)

新易盛经营现金流增加，原因为销售回款及税费返还；存货由72.34亿元升至116.55亿元，公司解释为按在手及预期订单、备货周期备货。预期订单仍会变化，税费返还也不等于客户新增需求。财务费用中的汇兑损失另会影响利润，不能把收入增速直接当成净利润增速。[半年报第7、14—15页](https://static.cninfo.com.cn/finalpage/2026-08-25/1225499406.PDF)

寒武纪应收账款由6.71亿元降至2.22亿元，原文解释为收回旧款；预付款由7.45亿元升至29.14亿元，存货由49.44亿元升至82.48亿元，主要涉及采购、原材料和委托加工。经营现金流下降主要来自采购及税金增加。它面临芯片制造、封装、软件适配和客户验收等约束，不应直接套用北美光模块的需求链。[半年报第7—8、19—23页](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF)

以上资产负债数字比较的是6月底和上年末存量。存量变化包含赊购、付款、收入确认等，不是已经完成的现金流逐项分解。公司解释是重要证据，但本轮没有独立识别每个原因的贡献比例。

## 3. 时间决定哪个因子有用

新易盛9月11日交流称，今年交付主要为800G和1.6T，1.6T下半年放量预期已公开；NPO预计2027年有批量交付需求、2028年规模进一步提升。公司没有给出本轮所需的季度出货数量、毛利率和单独客户预算指引。投资者提问中的猜测不作为公司的确认。[交流记录第1—3页](https://static.cninfo.com.cn/finalpage/2026-09-12/1225562013.PDF)

因此未来几周更有用的是现有产品交付、良率、单位成本、回款及其相对预期的变化。NPO可影响远期估值，但不能直接充当2026年四季度已实现收入。微软FY27Q1 Azure约45%的恒定汇率增速属于管理层已公开指引；即使兑现，也不自动构成超出市场预期的新信息。

因子用途取决于当时瓶颈：订单不足时看终端需求和预算；需求强而供给紧时看交付、良率和关键料件；投入已发生时看验收、回款与存货消化；业绩已被普遍预期时，看新增盈利修正和贴现率。阶段以事前可观察的约束判断，不能用后来涨跌给过去补标签。

## 4. 已有价格，还没有完整的价格预期

截至2026年9月28日收盘，供应商报价为：510300 **4.417元**，中际旭创**815.00元**，新易盛**399.70元**，寒武纪**1020.00元**。腾讯当日收盘与新浪9月29日报价中的昨收一致；原始响应和日期存于[价格记录](prices/result.json)。这是供应商报价核对，并非交易所官方确认、独立定价来源或9月29日收盘数据。未用这段未复权序列计算总回报或事件收益。

一个价格不能唯一倒推出市场的盈利预测、估值倍数和风险补偿。目前没有同一时点同一未来盈利期的一致预期。本轮不把半年EPS乘二当年度预测，不把管理层指引叫做市场一致预期，也不声称上述股价便宜或昂贵。

用于理解的代数关系是：在同一未来盈利口径、股本等条件可比时，价格变化约为“盈利预测修正倍数×估值倍数修正倍数−1”。假设盈利预期上修10%、估值倍数下修10%，结果仍是**−1%**，不是上涨10%。这只是敏感性算例，不是给任一股票设定的真实预测或目标价。

8月末官方权重为3.62%、2.13%、1.33%，合计{total_weight:.2f}%。在权重固定、其他成分不变的假设下，三股同时上涨10%仅贡献约{sensitivity['ten_percent_common_stock_rise_index_contribution_pp']:.3f}个百分点；若同时出现上述−1%价格变动则贡献约{sensitivity['fixed_weight_index_contribution_pp']:.4f}个百分点。实际还有其余成分、权重漂移、ETF跟踪和交易成本。**三家公司基本面良好，仍不足以推出510300有可交易的正收益。**[前轮官方权重记录](../510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json)

## 5. 未来1—4周如何更新判断

观察窗保持9月29日至10月26日。以下是条件观察，不伪装成已量化的新增预测；未查实财报发布日期，不假定三季报都在窗口内发布。没有新可比数据就保留未判断，不把未披露当零，也不事后延长窗口。

|正在判断的原因|会加强判断的新事实|会推翻或削弱判断的新事实|
|---|---|---|
|上游金额下调主要是口径变化|同口径实际采购和需求仍兑现，并出现可核实的供应商传导|实际采购取消、推迟或短寿命设备采购减少，且不能由口径或到货时点解释|
|光模块处于交付约束阶段|交付、良率或回款高于发布前的可比预期|数量或验收落后，单价、成本、汇兑压低净利润，备货无法消化|
|寒武纪现金消耗包含扩张投入|预付和委托加工转换成交付、验收与回款|供应或适配受阻、验收延迟，成本和减值吞噬增长|
|局部盈利能否形成指数机会|主要权重公司的盈利修正与资金条件共同支持，价格留下空间|其余成分或贴现率抵消，或相关增长早已被价格计入|

完整观察项保存在[next_observations.json](next_observations.json)。前轮已记录的9月PMI订单分化、购进与出厂价格差、节后资金三项前瞻判断保持原样，未来结果尚未产生；本轮不追加一套事后胜率。

这轮的实用成果是：能明确排除“支出金额下降就看空”“现金流下降就判需求坏”“技术规划等于近期收入”这三种失真传导。接下来最值得补的是**同一事件发布前的可比预期、实际修正，以及它对指数主要权重的净影响**，而不是继续堆叠高增长公司的描述。

## 6. 记录范围与目标状态

本轮只读取半年报与原因相关的页面，四份PDF初次抽取各前50页或全文不足50页的部分；六个财务及交流页面已目视确认单位、合并口径、正负号。中际旭创经营现金流是同比下降44.08%，不沿用搜索摘要中相反的描述；新易盛使用合并现金流16.16亿元，不混入母公司现金流。

微软直接下载的HTTP 403失败仍保留；可用证据来自web工具读取原站的文本摘录，文件明确标注不是原始HTML。原PDF及原行情响应保留，临时页面图片在核对后清理。不以资料可读或数值复算代替策略有效性。

新增策略账户0，新增收益测试0。夏普、CAGR均未计算；本轮没有证明成本后夏普达到1.2。目标继续active，旧失败与交易风险预算保持原状。研究中形成的经营判断不等同于实际现金仓位或下单指令。
"""
    save_json(OUT / "reviewed_facts.json", facts)
    save_json(OUT / "next_observations.json", observations)
    (OUT / REPORT_NAME).write_text(report, encoding="utf-8")
    result = {
        "study_id": STUDY, "recorded_at": recorded_at,
        "previous_goal_turn_classification": "PROGRESS_CURRENT_MACRO_AND_INDEX_WEIGHT_BRIDGE",
        "continuation_classification": "PROGRESS_REAL_DRIVERS_COMPANY_TRANSMISSION_AND_CURRENT_PRICE_ANCHORS",
        "status": "CAUSES_SEPARATED_PRICE_EXPECTATION_GAP_NOT_IDENTIFIED",
        "concrete_progress": ["区分微软支出调整的租赁口径与实物需求", "核对三家公司不同现金流原因", "区分2026交付与2027至2028技术规划", "取得4个标的9月28日收盘价并核对供应商昨收", "形成下一阶段可推翻判断的观察项"],
        "new_origin_pdfs": 4, "upstream_origin_web_documents": 2,
        "new_accounts": 0, "new_strategy_return_tests": 0, "strategy_rule_created": False,
        "current_index_return_forecast_made": False, "forward_consensus_obtained": False,
        "current_price_snapshot_obtained": True, "net_sharpe": None, "net_cagr": None,
        "goal_achieved": False, "goal_status": "active", "orders_authorized": False,
        "not_a_blocked_turn": True, "report": REPORT_NAME,
        "next_question": "在真实信息更新时锁定同口径的事前预期、盈利修正与主要权重净影响，检验是否存在剩余交易空间；不把旧公开增长当新惊喜。",
        "existing_macro_forecast_cards_unchanged": True,
    }
    save_json(OUT / "result.json", result)
    save_json(OUT / "source_review.json", {
        "recorded_at": recorded_at,
        "pdf_visual_pages": {"innolight_h1.pdf": [8, 26], "eoptolink_h1.pdf": [7], "cambricon_h1.pdf": [7, 23], "eoptolink_september_ir.pdf": [2]},
        "pdf_text_scope": "已保存各前50页或不足50页的全文，仅逐项复核引用页面；不声称阅读全部四份报告。",
        "microsoft_web_text_files": ["microsoft_web_open.txt", "microsoft_decomposition_web_extract.txt", "microsoft_final_cause_extract.txt"],
        "raw_html_status": "直接下载失败，未覆盖原失败收据；web原站文本可用于本次归因。",
        "search_summary_correction": "中际旭创经营现金流同比-44.08%，以原PDF为准。",
        "cashflow_scope": "新易盛使用合并口径1616408524.15元，排除母公司现金流。",
        "validation_scope": "事实方向、单位、口径与算术；不构成策略检验。",
    })
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read_json(mandate_path)
    mandate.update({
        "current_round": STUDY,
        "latest_progress_receipt": str((OUT / "result.json").relative_to(ROOT)).replace("\\", "/"),
        "last_research_result": "完成上游支出原因、三家公司经营现金流与下一阶段交付传导；微软1900至1750亿美元调整主要解释为租赁口径，三家公司现金流原因不同。取得9月28日价格，但未识别一致预期与可交易正预期差；新增账户0、目标未实现。",
        "latest_continuation_report": str((OUT / REPORT_NAME).relative_to(ROOT)).replace("\\", "/"),
        "latest_continuation_classification": result["continuation_classification"],
        "latest_new_public_price_source": str((OUT / "prices/result.json").relative_to(ROOT)).replace("\\", "/"),
        "last_source_result": "新增4份原PDF、2份微软原站web文本；直接HTML请求403保留。腾讯收盘与新浪昨收核对4个标的9月28日报价，非交易所原始行情。",
        "next_research_question": result["next_question"],
        "latest_driver_diagnostic_at": recorded_at,
        "latest_weight_company_driver_bridge": str((OUT / "result.json").relative_to(ROOT)).replace("\\", "/"),
        "goal_status": "active", "goal_achieved": False,
    })
    save_json(mandate_path, mandate)
    protocol_path = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read_json(protocol_path)
    protocol.update({
        "latest_concrete_diagnostic": str((OUT / "result.json").relative_to(ROOT)).replace("\\", "/"),
        "latest_weight_company_driver_bridge": str((OUT / "reviewed_facts.json").relative_to(ROOT)).replace("\\", "/"),
        "current_price_expectation_bridge": "PRICE_SNAPSHOT_OBTAINED_FORWARD_EXPECTATION_GAP_NOT_IDENTIFIED",
        "current_price_snapshot": str((OUT / "prices/result.json").relative_to(ROOT)).replace("\\", "/"),
        "current_index_return_forecast_made": False, "updated_at": recorded_at,
        "accounting_volume_timing_rule": "先区分金额的数量、单价、交付时点和会计口径；经营现金流再追到付款、回款、税金和备货原因。",
    })
    save_json(protocol_path, protocol)
    status_path = ROOT / "RESEARCH_STATUS.md"
    previous = status_path.read_text(encoding="utf-8-sig")
    prefix = f"""<!-- WEIGHT_COMPANY_DRIVER_BRIDGE_20260929 -->

## 2026-09-29 上游支出原因、权重公司经营与当期价格

微软1900至1750亿美元资本开支调整主要涉及租赁分类，不能直接当作实物需求下降；三家公司的采购、回款和备货分别解释现金流变化。新易盛今年800G/1.6T交付与2027年以后的NPO规划分开。取得9月28日510300及三家公司供应商收盘价并核对昨收，但没有同口径市场事前预期，尚未识别可交易的正预期差。新增账户0、收益测试0，旧账户800/952及失败不变；夏普目标未实现，目标active。本轮分类PROGRESS。

详见[上游支出原因与权重公司预期差](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_weight_company_driver_bridge_20260929/{REPORT_NAME}>)。前轮三项宏观预测保持原样，后续按真实新发布信息核对。

"""
    status_path.write_text(prefix + previous, encoding="utf-8")
    print("已保存原因分解、公司事实、当期价格及后续观察，未创建策略账户；目标尚未实现。")
    print(str(OUT / REPORT_NAME))


if __name__ == "__main__":
    main()
