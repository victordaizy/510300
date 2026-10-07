"""将金融权重的盈利变化拆到可延续的原因，并登记待公布经营结果判断。"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_financial_driver_bridge_v1"
STAMP = datetime.now(timezone(timedelta(hours=8))).isoformat()
CMB_URL = "https://static.cninfo.com.cn/finalpage/2026-08-29/1225530237.PDF"
PA_URL = "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2026/2026-8/2026-08-21/12512669.PDF"
LPR_URL = "https://jrj.sh.gov.cn/SCGK194/20260920/736cd13499494746bebda910dd12cc65.html"
LPR_HISTORY_URL = "https://www.psbc.com/cn/common/bjfw/dkscbjllcx/202507/t20250721_352102.html"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮结论和预测已保存，后续新信息另记，不覆盖事前判断。")
    cmb = {
        "symbol": "600036", "name": "招商银行", "unit": "人民币百万元；利率为百分数",
        "scope": "集团口径，2026H1与报告所列2025H1；余额比较另列2025年末",
        "net_interest_income": 112022, "prior_net_interest_income": 106085,
        "net_interest_income_volume_contribution": 8279,
        "net_interest_income_rate_contribution": -2342,
        "volume_rate_interaction_allocation": "依公司口径计入规模因素，不是外生冲击的严格因果效应",
        "net_interest_margin_pct": 1.83, "prior_net_interest_margin_pct": 1.88,
        "q2_net_interest_margin_pct": 1.82, "q1_net_interest_margin_pct": 1.83,
        "earning_asset_yield_pct": 2.82, "prior_earning_asset_yield_pct": 3.14,
        "interest_bearing_liability_cost_pct": 1.05, "prior_liability_cost_pct": 1.35,
        "average_earning_assets": 12369944, "prior_average_earning_assets": 11358612,
        "retail_loan_balance": 3680028, "prior_year_end_retail_loan_balance": 3720191,
        "retail_npl_pct": 1.16, "prior_year_end_retail_npl_pct": 1.06,
        "corporate_npl_pct": 0.78, "prior_year_end_corporate_npl_pct": 0.89,
        "total_npl_pct": 0.94, "prior_year_end_total_npl_pct": 0.94,
        "wealth_management_fee_income": 16192, "prior_wealth_management_fee_income": 12797,
        "fund_distribution_income_yoy_pct": 61.40,
        "management_causes": ["上年二季度LPR下调后的贷款重定价", "有效信贷特别是零售需求不足", "新投债券收益率低于到期债券", "较低存款成本基数约束后续降成本空间"],
        "management_forward_anchor": "公司已经预期下半年净利息收益率降幅进一步收窄；不是研究新增的独有信息。",
        "source_url": CMB_URL, "pdf_pages": [14, 15, 16, 23, 24, 38, 39, 43],
    }
    cmb["net_interest_income_change"] = cmb["net_interest_income"] - cmb["prior_net_interest_income"]
    cmb["net_interest_income_yoy_pct"] = (cmb["net_interest_income"] / cmb["prior_net_interest_income"] - 1) * 100
    cmb["average_earning_asset_yoy_pct"] = (cmb["average_earning_assets"] / cmb["prior_average_earning_assets"] - 1) * 100
    cmb["retail_loan_vs_year_end_pct"] = (cmb["retail_loan_balance"] / cmb["prior_year_end_retail_loan_balance"] - 1) * 100
    if cmb["net_interest_income_change"] != cmb["net_interest_income_volume_contribution"] + cmb["net_interest_income_rate_contribution"]:
        raise ValueError("公司净利息收入量价分解未能与同比变化相等。")
    pa = {
        "symbol": "601318", "name": "中国平安", "unit": "人民币百万元；收益率为百分数",
        "scope": "2026H1与报告所列2025H1；归母与含少数股东口径分别保留",
        "parent_net_profit": 92585, "prior_parent_net_profit": 68047,
        "parent_operating_profit": 84196, "prior_parent_operating_profit": 77732,
        "asset_management_parent_operating_profit": 9172,
        "prior_asset_management_parent_operating_profit": 2723,
        "group_short_term_investment_fluctuation": 4166,
        "prior_group_short_term_investment_fluctuation": -4126,
        "group_oneoff_and_other_adjustment": 4297,
        "prior_group_oneoff_and_other_adjustment": -5571,
        "adjustments_are_before_minority_attribution": True,
        "management_operating_profit_definition": "剔除指定短期投资波动及管理层认为非日常的一次性重大项目等；适用业务的投资回报率锁定为4.0%，不等于所有业务未来实际回报4.0%。",
        "oneoff_causes": "美元、港元可转债转股权价值重估等；上年还含平安健康并表损益。不能将管理层剔除项目视作没有经济影响。",
        "insurance_net_investment_income": 78012, "prior_insurance_net_investment_income": 92823,
        "insurance_total_investment_income": 136942, "prior_insurance_total_investment_income": 96216,
        "insurance_comprehensive_investment_income": 117373, "prior_insurance_comprehensive_investment_income": 157809,
        "insurance_comprehensive_yield_nonannualized_pct": 2.1,
        "prior_insurance_comprehensive_yield_nonannualized_pct": 3.1,
        "comprehensive_yield_reported_cause": "公司称高股息及低波动权益资产表现不及上年同期。不能据此拆出每类资产的精确因果贡献。",
        "investment_return_scope_note": "保险资金组合与集团归母利润覆盖不同；净投资、总投资、综合投资不是可互换指标。综合收益率计算还剔除指定债权投资公允价值变化。",
        "new_business_value": 24847, "prior_new_business_value": 22335,
        "life_new_contract_service_margin": 29811, "prior_life_new_contract_service_margin": 25209,
        "life_new_business_margin_pct": 9.8, "prior_life_new_business_margin_pct": 10.7,
        "q2_standard_premium_nbvm_qoq_change_pp": 3.5,
        "nbvm_is_not_csm_new_business_margin": True,
        "product_response": "公司将分红险转型与低利率环境、浮动收益供给及资产负债韧性联系；不由此推出所有存量保证成本已下降。",
        "source_url": PA_URL, "pdf_pages": [38, 39, 44, 46, 47, 56, 57, 58, 66, 67, 68],
    }
    pa["parent_net_profit_change"] = pa["parent_net_profit"] - pa["prior_parent_net_profit"]
    pa["parent_operating_profit_change"] = pa["parent_operating_profit"] - pa["prior_parent_operating_profit"]
    pa["parent_nonoperating_change"] = pa["parent_net_profit_change"] - pa["parent_operating_profit_change"]
    pa["parent_nonoperating_share_of_net_profit_change_pct"] = pa["parent_nonoperating_change"] / pa["parent_net_profit_change"] * 100
    pa["asset_management_parent_operating_profit_change"] = pa["asset_management_parent_operating_profit"] - pa["prior_asset_management_parent_operating_profit"]
    weights = read_json(ROOT / "reports/research/510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json")
    lookup = {row["symbol"]: row for row in weights["top10"]}
    bridge = {
        "weight_date": weights["data_date"], "financial_sector_weight_pct": weights["sector_weights_pct"]["金融"],
        "company_weights_pct": {code: lookup[code]["weight_pct"] for code in ["600036", "601318"]},
        "two_company_direct_weight_pct": sum(lookup[code]["weight_pct"] for code in ["600036", "601318"]),
        "prior_three_company_direct_weight_pct": 7.08,
        "distinct_five_company_direct_weight_pct": round(7.08 + sum(lookup[code]["weight_pct"] for code in ["600036", "601318"]), 2),
        "not_a_full_financial_sector_or_index_profit_forecast": True,
        "unknown_other_weights_not_assumed_zero_impact": True,
        "same_shock_channels_are_not_independent_evidence": True,
        "price_feedback_rule": "市场上涨可能经投资和代销影响金融公司利润；反馈可以继续，但不能循环证明新的独立基本面利好。",
    }
    facts = {"recorded_at": STAMP, "cmb": cmb, "pingan": pa, "index_bridge": bridge,
             "lpr": {"current_public_one_year_pct": 3.0, "current_public_five_year_plus_pct": 3.5,
                     "unchanged_in_2026_monthly_bank_table": True, "government_reprint_url": LPR_URL,
                     "bank_history_url": LPR_HISTORY_URL, "is_september_new_rate_cut": False}}
    save(OUT / "reviewed_facts.json", facts)

    card = {
        "id": "FIN1", "recorded_at": STAMP, "symbol": "600036",
        "type": "PRE_RELEASE_BUSINESS_NOWCAST_NOT_RETURN_FORECAST",
        "forecast": "招商银行待公布的2026三季报中，集团前三季净利息收入同比为正，集团零售贷款期末余额仍低于2025年末。",
        "components": [
            {"id": "FIN1A", "metric": "集团2026年1至9月净利息收入/同表2025年1至9月净利息收入-1", "condition": ">0"},
            {"id": "FIN1B", "metric": "集团2026年9月末零售贷款余额/同口径2025年末余额-1", "condition": "<0"},
        ],
        "reason": "H1净利息收入增长由规模抵消利率拖累；零售贷款已收缩，需求与信用卡结构仍承压，尚缺广泛修复证据。",
        "alternatives": ["生息资产规模放缓或息差再下压，使净利息收入转为下降", "零售投放、需求或经营策略改变，使余额在三季度反超年末"],
        "evaluation": "仅在2026三季报首次正式公开后，用同报告可比集团口径逐项核对；若只公布本行或缺少零售字段，该项未能评价，不替换。",
        "joint_hit_rule": "两项均成立才记联合命中；分别记录，不当两份独立样本。",
        "known_before_forecast": "已读取2026H1原报告，未观察未来三季报；当季大部分经营已发生，因此是发布前判断，不称完整季度的提前预测。",
        "is_new_management_guidance": False, "is_market_consensus_gap": False,
        "probability": None, "status": "PENDING_FUTURE_PUBLICATION",
        "strategy_rule_created": False, "automation_created": False,
    }
    save(OUT / "company_forecast_card.json", card)

    report = f"""# 金融权重的盈利原因与指数传导

记录：{STAMP}。上轮分类经已保存result核实为PROGRESS；本轮继续研究利率、需求和市场反馈怎样影响510300。执行范围及风险预算不变。

当前证据更支持“增长来源分化”，不足以支持金融业已经全面经营复苏。招商银行收入增长依靠规模对冲息差压力，零售与对公方向不同；平安较快净利润增长含显著投资波动和非日常项目影响。两家公司都有资本市场传导，不能把股市上涨及其带来的盈利同时当成两份独立原因。

**招行：量在补价，零售与对公分化。**

集团H1净利息收入1,120.22亿元，增加59.37亿元。原报告给出的分解为：规模因素+82.79亿元，利率因素−23.42亿元，合计+59.37亿元。公司把量价共同变化计入规模因素；这是财务分解，不是外生利率冲击的因果估计。

生息资产平均余额同比增加约{cmb['average_earning_asset_yoy_pct']:.1f}%；资产平均收益率下降32bp，负债成本下降30bp，净利息收益率由1.88%降到1.83%。所以“净利息收入增长”不能直接解释成贷款定价能力修复，更不能由此推断利率下降对银行收入一定有利。

进一步追原因，原文同时列出：上年降息后的贷款重定价、零售有效需求不足、新投债券收益低于到期债券，以及低存款成本基数令后续降成本空间有限。由此形成可观察的约束：贷款及债券收益下调的速度，与负债重定价节省成本的速度，谁更快；同时能否增加质量合适的生息资产。

集团零售贷款余额较年末下降1.08%，零售不良率由1.06%升至1.16%；公司贷款不良率由0.89%降至0.78%，总不良率仍为0.94%。总指标稳定下面有不同方向，不能把总不良率未变当作所有客户质量都稳定。

集团财富管理手续费增长26.53%，其中代理基金收入增长61.40%，公司解释涉及权益基金保有规模和销量提升。这可能受资本市场活跃支持，不能全部作为独立于股市的实体需求修复证据。

来源：[招行半年报，PDF第14—16、24、38—39页]({CMB_URL})。第38页已披露管理层预期下半年息差降幅收窄；它是已有公开基准，后续仅“降幅收窄”不自动成为新惊喜。

**平安：把净利润增长还原为不同来源。**

| 归母口径，亿元 | 2026H1 | 2025H1 | 增量 |
| --- | ---: | ---: | ---: |
| 净利润 | 925.85 | 680.47 | 245.38 |
| 营运利润 | 841.96 | 777.32 | 64.64 |
| 两者差额 | 83.89 | −96.85 | 180.74 |

按公司营运利润定义，净利润增量中约{pa['parent_nonoperating_share_of_net_profit_change_pct']:.1f}%来自营运利润之外项目的变化。定义涉及指定寿险业务短期投资波动及管理层认为非日常的一次性重大项目等；其中包含可转债转股权价值重估，上年还含平安健康并表损益。不能将36.1%的净利润增速直接外推成常态经营增速，也不能因为这些项目被管理层剔除就认为没有经济影响。

集团归母营运利润增加64.64亿元，资管分部增加64.49亿元，增量明显集中。该分部包含证券、信托、租赁及其他资管业务；本轮没有把全部分部增量强行归因于市场上涨。公司对证券业务披露净利润增长31.5%，还需要将客户规模、交易、费用和资产质量分别解释。

保险资金组合的净投资收益下降16.0%，总投资收益增长42.3%，综合投资收益下降25.6%，非年化综合收益率由3.1%降至2.1%。公司将综合收益率下降主要联系到高股息、低波动权益资产表现不及上年同期；其中净投资收益还包含利息、分红、租金及联营损益，不能将其下降全部归因于利率。不同指标覆盖的收益项目和会计位置不同，保险组合又不等于整个集团。不能拿其中一个增长率代表全部投资能力改善，也不能用某个下降率否定全部利润改善。

低利率还影响负债和产品选择。公司将分红险转型与低利率环境及资产负债韧性联系；新业务合同服务边际增加18.3%，但新业务利润率由10.7%降至9.8%。销量、产品结构和单位价值需要一起看，新业务增长不等于当前全部利润或存量负债成本立即改善。

也保留边际改善证据：公司演示第14页披露，第二季度按标准保费计算的新业务价值率比第一季度提高3.5个百分点。这是后续产品量价可能改善的线索；它与上面的合同服务边际口径新业务利润率不同，不能互相替代。后续观察第三季度是否延续这项改善，而非只沿用全年或上半年同比作判断。

来源：[平安中期报告，PDF第38—39、44、46—47、56—58、66—68页]({PA_URL})；[公司中期业绩演示，第44—45页](https://static.cninfo.com.cn/finalpage/2026-08-21/1225487472.PDF)同时列示利润勾稽。原报告人民币百万元已换算为亿元；归母金额未与含少数股东调整额混加。

**利率水平、利率变化、旧政策的滞后传导分别判断。**

当前公开的1年期及5年期以上LPR为3.0%、3.5%，银行公布的2026年月度表中保持不变。招行说明的贷款重定价与上年降息有关，不能包装成本月发生新降息。LPR保持稳定也不保证贷款实际收益率不变，贷款组合、期限、合同重定价、竞争及债券再投资仍会影响资产收益。

来源：[上海市委金融办转载央行公布值]({LPR_URL})、[邮储银行LPR历史表]({LPR_HISTORY_URL})。两处是同一根利率公布的信息，不计作独立宏观冲击。

| 下一阶段的主导变化 | 支持盈利的传导 | 可能抵消的传导 | 下一项区分证据 |
| --- | --- | --- | --- |
| 国内真实融资需求修复 | 合理定价的贷款投放、回款和资产质量改善；保险销售及单位价值可能改善 | 利率上行或成本回升可能压低部分资产估值 | 零售和企业订单/信贷结构、信用成本、新业务量价 |
| 跨季短期资金压力缓和 | 同业融资和流动性条件改善 | 不能保证贷款需求、长期再投资收益或经营利润同步改善 | 已登记DR007判断，以及独立的融资需求变化 |
| 主要是资本市场上涨 | 投资估值、交易及财富管理收入受益，反馈可能延续 | 已经发生的市场收益或重估不能反复算作新的独立消息 | 后续客户资金、交易活动及盈利预测修正是否持续 |
| 债券收益率因需求偏弱而下降 | 部分存量资产估值和融资成本受益 | 再投资收益下行、息差压力、信用风险可能抵消 | 降收益来自政策供给还是需求收缩，资产与负债重定价速度 |

这是条件传导表，不是给每个分支填入未经估计的上涨概率。对510300的净方向仍需相应权重和价格反映程度；利率只说明一个中间变量，来源不同可以改变方向。

8月末官方权重中金融占19.9%，招行1.89%、平安2.27%，两家合计4.16%。加上上轮三家科技相关公司，直接观察对象合计11.24%。这不是“已经解释指数11.24%的涨跌”，更不能把两家公司代表整个19.9%的金融。其余成分影响保留未知，不能当成零；指数权重也不是指数利润占比。

与前轮三项宏观预测的连接：若节后DR007压力缓和，只能先确认资金条件正常化；若非制造业订单仍弱，不能由制造业改善直接推出居民和全部服务需求已经修复。若后续需求改善超出这些公开基准，才有理由上调更广泛的盈利路径。既有宏观预测原样保留。

**新增一条事前经营判断，检验上述原因能否区分待公布结果。**

FIN1：招行2026三季报集团前三季净利息收入同比为正，同时零售贷款期末余额仍低于2025年末。依据是规模与价格、对公与零售的分化延续；可能被资产增长放缓、息差再受压或零售投放转强推翻。

这是9月29日登记的待发布经营结果判断，当季大部分经营已发生，不能称为提前预测了整个季度。公布后按同报告集团可比口径逐项核对，两项都成立才记联合命中；缺字段不以本行口径替代。该判断没有概率、股价目标或仓位，也没有证明超出市场预期。即使命中，也不等于夏普目标得到验证。

记录：[公司预测卡](company_forecast_card.json)、[原因与数字](reviewed_facts.json)。本轮取得3份公司PDF和2个LPR页面，只做核心原件与加总核对；没有新增收益测试、账户或长期参数搜索。旧800个有效/952个已运行账户与既有失败不变。净夏普、净年化未计算，目标active且未达成。
"""
    (OUT / "金融权重的盈利原因与指数传导.md").write_text(report, encoding="utf-8")
    result = {
        "study_id": "510300_FINANCIAL_DRIVER_BRIDGE_V1", "recorded_at": STAMP,
        "previous_goal_turn_classification": "PROGRESS_PRIOR_EXPECTATION_AND_CONDITIONAL_EARNINGS_HURDLES",
        "continuation_classification": "PROGRESS_FINANCIAL_EARNINGS_CAUSES_AND_PRE_RELEASE_JUDGMENT",
        "status": "FINANCIAL_TRANSMISSION_SEPARATED_INDEX_NET_EDGE_UNPROVEN",
        "concrete_progress": ["招行收入量价分解", "总资产质量下的对公零售分化", "平安归母净利润与营运利润增量勾稽", "利率滞后与市场反馈分清", "登记FIN1待公布经营判断"],
        "new_company_pdfs": 3, "new_rate_html_pages": 2,
        "new_accounts": 0, "new_strategy_return_tests": 0, "strategy_rule_created": False,
        "existing_account_counts": {"admitted": 800, "executed": 952},
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "current_index_return_forecast_made": False, "new_business_forecast": "company_forecast_card.json",
        "existing_macro_forecasts_unchanged": True, "goal_achieved": False, "goal_status": "active",
        "orders_authorized": False, "not_a_blocked_turn": True,
        "report": "金融权重的盈利原因与指数传导.md",
        "next_question": "将真实需求改善、流动性正常化和既有市场上涨造成的盈利反馈分开，在新发布信息上比较预期修正与已反映价格。",
    }
    save(OUT / "result.json", result)
    p = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read_json(p)
    protocol.update({
        "latest_concrete_diagnostic": "reports/research/510300_financial_driver_bridge_v1/result.json",
        "latest_financial_driver_bridge": "reports/research/510300_financial_driver_bridge_v1/reviewed_facts.json",
        "latest_company_business_forecast": "reports/research/510300_financial_driver_bridge_v1/company_forecast_card.json",
        "market_feedback_rule": "市场变化经投资收益、销售与交易活动影响公司利润；反馈不是独立于原市场变化的新根证据，是否可预测后续收益另行判断。",
        "financial_profit_scope_rule": "集团与本行、归母与少数股东、营运利润与净利润、资产收益与负债成本分别传导，不混加。",
        "updated_at": STAMP,
    })
    save(p, protocol)
    p = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read_json(p)
    mandate.update({
        "current_round": result["study_id"], "latest_progress_receipt": "reports/research/510300_financial_driver_bridge_v1/result.json",
        "latest_continuation_report": "reports/research/510300_financial_driver_bridge_v1/金融权重的盈利原因与指数传导.md",
        "latest_continuation_classification": result["continuation_classification"],
        "last_research_result": "招行净利息收入由规模抵消利率拖累，零售仍弱；平安净利润增量约74%来自营运利润以外项目变化。金融传导分清并登记FIN1，未识别指数净优势，新增账户0。",
        "last_source_result": "3份公司PDF、2个LPR页面已保存，关键量价及归母利润表已核对。",
        "latest_financial_driver_bridge": "reports/research/510300_financial_driver_bridge_v1/result.json",
        "latest_driver_diagnostic_at": STAMP, "next_research_question": result["next_question"],
    })
    save(p, mandate)
    p = ROOT / "RESEARCH_STATUS.md"
    head = """<!-- FINANCIAL_DRIVER_BRIDGE_V1_20260929 -->

## 2026-09-29 金融权重的盈利原因与指数传导

招行净利息收入增加59.37亿元，规模贡献82.79亿元、利率拖累23.42亿元；总不良率稳定下，零售与对公表现分化。平安归母净利润增量245.38亿元，营运利润增量64.64亿元，差额180.74亿元；不能将投资、一次性变化与持续经营混同。区分旧利率重定价、当前LPR及市场反馈。新增FIN1待公布经营判断，原宏观预测不改。新增账户0、收益测试0，旧800/952与失败不变，夏普1.2目标active且未实现。本轮PROGRESS。

详见[金融权重的盈利原因与指数传导](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_financial_driver_bridge_v1/金融权重的盈利原因与指数传导.md>)。

"""
    p.write_text(head + p.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"状态": result["status"], "净利息收入增量亿元": cmb["net_interest_income_change"]/100,
                      "平安营运外项目增量占比百分比": pa["parent_nonoperating_share_of_net_profit_change_pct"],
                      "前瞻经营判断": card["forecast"], "报告": str(OUT / result["report"]),
                      "新增账户": 0, "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
