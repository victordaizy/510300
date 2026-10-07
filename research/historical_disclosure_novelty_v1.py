"""比较原三例正式报告与此前公开信息，不计算新交易收益。"""

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_disclosure_novelty_v1"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    source = json.loads((OUT / "reviewed_source_facts.json").read_text(encoding="utf-8"))
    scope = json.loads((OUT / "scope.json").read_text(encoding="utf-8"))
    financial = pd.read_parquet(ROOT / source["target_financial_amount_source"])
    financial["announcement_id"] = financial.announcement_id.astype(str)

    def fact(identity, metric):
        rows = financial[financial.announcement_id.eq(identity) & financial.metric_id.eq(metric)]
        if len(rows) != 1:
            raise ValueError(f"财务事实不唯一：{identity}，{metric}")
        return float(rows.verified_value.iloc[0])

    # 金额与刚读取的原公告交叉核对；只检查本次实际用于相减的字段。
    for key in ["muyuan_2024q1", "commodity_city_2025h1", "shangji_2022h1"]:
        row = source[key]
        for field, metric in [("parent_profit", "PARENT_NET_PROFIT_YTD"), ("operating_cash_flow", "OPERATING_CASH_FLOW_YTD")]:
            if abs(row[field] - fact(row["announcement_id"], metric)) > .011:
                raise ValueError(f"原公告与继承金额不一致：{key}，{field}")

    guidance, q1 = source["muyuan_guidance"], source["muyuan_2024q1"]
    actual_h1 = fact("1220788951", "PARENT_NET_PROFIT_YTD")
    actual_q2 = actual_h1 - q1["parent_profit"]
    implied_q2_low = guidance["parent_profit_low"] - q1["parent_profit"]
    implied_q2_high = guidance["parent_profit_high"] - q1["parent_profit"]
    sample = pd.read_parquet(ROOT / "reports/research/510300_historical_cash_quality_causes_v1/全部49条公司财务变化.parquet")
    row = sample[sample.announcement_id.eq("1220788951")].iloc[0]
    if abs(q1["assets"] - float(row.previous_quarter_assets)) > .011:
        raise ValueError("原季度资产与因子分母不同。")
    score = lambda profit: (profit / q1["assets"] - float(row.seasonal_mean)) / float(row.seasonal_sd)
    if abs(actual_q2 - float(row.quarter_profit)) > .011 or abs(score(actual_q2) - float(row.L02)) > 1e-9:
        raise ValueError("原盈利历史偏离分数不能重建。")
    muyuan = {
        "guidance_nominal_date": guidance["nominal_publication_date"], "report_nominal_date": "2024-08-03",
        "guidance_lead_calendar_days": 23, "parent_profit_guidance_low_cny": guidance["parent_profit_low"],
        "parent_profit_guidance_high_cny": guidance["parent_profit_high"], "actual_h1_parent_profit_cny": actual_h1,
        "actual_within_management_range": guidance["parent_profit_low"] <= actual_h1 <= guidance["parent_profit_high"],
        "actual_vs_management_midpoint_fraction": actual_h1 / ((guidance["parent_profit_low"] + guidance["parent_profit_high"]) / 2) - 1,
        "q1_parent_profit_already_public_cny": q1["parent_profit"],
        "implied_q2_profit_low_cny": implied_q2_low, "implied_q2_profit_high_cny": implied_q2_high,
        "actual_q2_profit_cny": actual_q2, "L02_guidance_low": score(implied_q2_low),
        "L02_guidance_high": score(implied_q2_high), "L02_actual": float(row.L02),
        "L02_range_status": "DESCRIPTIVE_RECONSTRUCTION_USING_INHERITED_HISTORICAL_BASELINE_NOT_NEW_TRADING_SIGNAL",
        "market_consensus_obtained": False, "cashflow_fully_preannounced": False,
    }
    quarters = []
    for key, target, name in [("commodity_city_2025h1", "1224710669", "小商品城"), ("shangji_2022h1", "1214728625", "上机数控")]:
        prior = source[key]
        target_cash = fact(target, "OPERATING_CASH_FLOW_YTD")
        target_profit = fact(target, "PARENT_NET_PROFIT_YTD")
        current = next(s for s in scope["cases"] if s["report_id"] == target)
        quarters.append({"company": name, "target_announcement_id": target,
                         "prior_nominal_date": prior["nominal_publication_date"],
                         "target_nominal_date": current["report_nominal_date"],
                         "prior_lead_calendar_days": (pd.Timestamp(current["report_nominal_date"]) - pd.Timestamp(prior["nominal_publication_date"])).days,
                         "h1_operating_cashflow_cny": prior["operating_cash_flow"], "ytd9m_operating_cashflow_cny": target_cash,
                         "q3_operating_cashflow_cny": target_cash - prior["operating_cash_flow"],
                         "h1_share_of_ytd9m_cashflow": prior["operating_cash_flow"] / target_cash,
                         "q3_share_of_ytd9m_cashflow": 1 - prior["operating_cash_flow"] / target_cash,
                         "h1_parent_profit_cny": prior["parent_profit"], "ytd9m_parent_profit_cny": target_profit,
                         "q3_parent_profit_cny": target_profit - prior["parent_profit"],
                         "cash_flow_period_bridge_is_information_surprise": False})
    save(OUT / "三例信息增量计算.json", {"muyuan": muyuan, "quarter_bridges": quarters})
    pd.DataFrame(quarters).to_csv(OUT / "两例累计现金到单季现金.csv", index=False, encoding="utf-8-sig")
    screen = json.loads((OUT / "metadata_screen_summary.json").read_text(encoding="utf-8"))
    result = {
        "study_id": scope["study_id"], "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_PRIOR_INFORMATION_AND_NEW_PERIOD_AMOUNTS_IDENTIFIED",
        "classification": "PROGRESS_HISTORICAL_DISCLOSURE_NOVELTY_AND_FACTOR_SEMANTICS",
        "previous_goal_turn_classification": scope["previous_goal_turn_classification"],
        "reviewed_prior_official_documents": 5, "case_count": 3,
        "metadata_screen": screen, "muyuan": muyuan, "quarter_bridges": quarters,
        "supported_findings": [
            "牧原正式中报利润落在此前管理层预告区间，主要经营原因已公开；不能将扭亏方向当作中报首次出现的信息。",
            "牧原预告结合已公开一季报即可推导二季度利润范围；原L02极高分主要衡量历史同季偏离，不是相对市场预期的 surprise。",
            "小商品城的招商收款机制在半年报已公开，三季度新增现金数额仍然很大；机制已知不等于后续金额已知。",
            "上机数控此前半年报已披露规模扩张和经营现金增加，前三季累计现金多数属于已公开半年期间。"
        ],
        "not_established": ["历史市场一致预期", "全部相关消息已充分计价", "财报新消息的独立价格因果效应", "可交易的510300净收益优势"],
        "old_accounts_unchanged": True, "new_accounts": 0, "new_return_tests": 0,
        "new_fitted_parameters": 0, "new_forecast_cards": 0, "goal_achieved": False,
        "wait_for_future_data": False, "independent_validation": False,
        "report": "reports/research/510300_historical_disclosure_novelty_v1/历史发现_财报高分与真正新增信息.md",
        "next_research_question": "转向有同期公开预期基准、且能改变指数共同盈利或折现条件的历史事件；优先检验事件的信息增量，不继续把少量公司季节性高分作为宽基新利好。"
    }
    save(OUT / "result.json", result)
    report(source, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def report(source, result):
    m = result["muyuan"]
    city, shangji = result["quarter_bridges"]
    lines = [
        "# 历史发现：财报高分与真正新增信息", "",
        "2026年9月30日。继续检查此前按财务构成选出的牧原2024半年报、小商品城2025三季报、上机数控2022三季报。对应后续收益已在上一轮见过，本轮不新增收益测试，不根据结果挑选案例。", "",
        "**三例都能找到正式目标报告之前的相关经营线索，但信息的新旧程度不同。经营方向、业绩区间、现金明细以及新季度金额必须分开；历史同比或季节性高分不能直接命名为市场预期差。**", "",
        "## 牧原：极高的历史分数，在中报前已有公开利润范围", "",
        "2024年7月6日的六月销售简报已披露商品猪均价17.73元/公斤，五月为15.52元/公斤；同时披露半年生猪销量3238.8万头。这是经营先行信息，不等于合并利润或现金流。7月11日业绩预告进一步给出归母利润7亿—9亿元，原因包括出栏量、销售均价同比上升及养殖成本同比下降。[六月销售简报，第1—2页](https://static.cninfo.com.cn/finalpage/2024-07-06/1220542009.PDF)、[半年业绩预告，第1页](https://static.cninfo.com.cn/finalpage/2024-07-11/1220602782.PDF)。", "",
        f"8月3日名义日期的正式中报归母利润为{m['actual_h1_parent_profit_cny']/1e8:.4f}亿元，在预告区间内，较区间中点高{m['actual_vs_management_midpoint_fraction']*100:.2f}%。这个中点只是管理层区间的算术中点，不能当作市场一致预期。", "",
        f"4月已公开的一季报归母利润为{m['q1_parent_profit_already_public_cny']/1e8:.4f}亿元，因此7月11日就可以由累计预告减去一季度，得到二季度归母利润约{m['implied_q2_profit_low_cny']/1e8:.4f}亿—{m['implied_q2_profit_high_cny']/1e8:.4f}亿元。正式结果为{m['actual_q2_profit_cny']/1e8:.4f}亿元。[牧原一季报，第2页](https://static.cninfo.com.cn/finalpage/2024-04-27/1219861327.PDF)、[牧原半年报](https://static.cninfo.com.cn/finalpage/2024-08-03/1220788951.PDF)。", "",
        "| 同一原L02公式的描述性重建 | 数值 |", "|---|---:|",
        f"| 用预告下限推导的二季度利润 | {m['L02_guidance_low']:.2f} |",
        f"| 用预告上限推导的二季度利润 | {m['L02_guidance_high']:.2f} |",
        f"| 正式中报利润计算出的原分数 | {m['L02_actual']:.2f} |", "",
        "这里沿用原样本的季度资产和两年同季基准，仅做指标语义检查，未新建7月交易信号。结果表明，正式报告时的极高L02不是其相对7月预告的新增偏离，更不是184倍市场惊喜。原分数更准确的解释是‘相对历史同季的盈利偏离’，其量级又受到仅两个历史点的小标准差放大。", "",
        "预告没有给出完整经营现金流与调节明细，因此只能确认利润方向、范围及主要经营原因已被披露，不能据此说中报全是旧消息、现金流毫无新增内容，或股价已完全反映。", "",
        "## 小商品城：招商收款机制已知，新增季度金额仍大", "",
        "8月18日名义日期的半年报已明确把现金流增加与全球数贸中心时尚珠宝行业招商收款联系起来，并披露首批商位招商定位及后续批次进展。半年经营现金净额13.83亿元，上年同期1.14亿元；这条原因在三季报前已经公开。[2025年半年报，第6、12、20页](https://static.cninfo.com.cn/finalpage/2025-08-18/1224501021.PDF)。", "",
        f"前三季累计经营现金净额{city['ytd9m_operating_cashflow_cny']/1e8:.2f}亿元，减去已披露半年金额，可得三季度单季{city['q3_operating_cashflow_cny']/1e8:.2f}亿元，占前三季累计金额{city['q3_share_of_ytd9m_cashflow']*100:.2f}%。所以，三季度新增现金数额仍然很大。这里计算的是期间构成，不能把这个占比叫作市场未知信息占比。[2025年三季报，第2、9页](https://static.cninfo.com.cn/finalpage/2025-10-15/1224710669.PDF)。", "",
        "需要保留的竞争解释是：市场可能知道项目收款方向，但对最终金额、确认收入时点或其持续性仍有不同预期。本轮没有取得当时统一的金额预期基准，因此无法判定该金额究竟高于还是低于市场预期。", "",
        "## 上机数控：累计现金大部分来自早已报告的半年期间", "",
        "8月30日名义日期的半年报已披露单晶硅产能30GW、上半年出货约15GW，并将收入及现金流增加归于业务规模扩大。这些资料早于10月目标三季报。[2022年半年报，第5、9、10页](https://static.cninfo.com.cn/finalpage/2022-08-30/1214442231.PDF)。", "",
        f"上半年经营现金净额{shangji['h1_operating_cashflow_cny']/1e8:.2f}亿元，前三季累计{shangji['ytd9m_operating_cashflow_cny']/1e8:.2f}亿元，三季度单季约{shangji['q3_operating_cashflow_cny']/1e8:.2f}亿元。半年期间金额占前三季累计{shangji['h1_share_of_ytd9m_cashflow']*100:.2f}%。这说明不能把累计27亿元当作本次才发生或才披露的全新经营现金；但三季度利润、现金及资产的实际新增数仍须单独评价。[2022年三季报，第3、10—11页](https://static.cninfo.com.cn/finalpage/2022-10-10/1214728625.PDF)。", "",
        "## 全样本的先行披露线索", "",
        "在原49条报告中，同公司、同报告期且严格早于目标报告名义日期的预告/快报标题候选覆盖22条；原公司自身同时通过条件的30条中，有15条存在这类候选。候选原文共24项，包含一组取消、补充和更新版本。除三例涉及的已核对材料外，其余候选没有逐份核对金额和版本，不能直接认定为22次重复消息，更不能从中计算预期差胜率。", "",
        "小商品城与上机数控对应三季报在该本地索引中没有找到同期间预告标题，只能记录此次检索结果；半年报中的相关经营信息仍然存在。未找到预告不等于没有其他公开信息，也不等于市场毫无预期。", "",
        "## 对继续研究的约束与方向", "",
        "本轮把原财务分数的经济含义说得更准确：先问消息是否新增，再判断它改变了哪些经营量及哪些指数成分。历史同比改善、管理层预告中点偏离和市场一致预期差是三种不同量，不能混用。只有原因相近、信息增量口径也相近的事件，才适合比较当时价格之后的剩余收益。", "",
        "此次发现支持停止把原23个少量公司披露篮子的季节性高分直接视为宽基新利好。旧账户失败不改，也不通过删除有预告案例来重跑或挽救它。下一步历史研究优先选择有同期公开预期基准、能够改变指数共同盈利或折现条件的事件。", "",
        "五份更早原公告已保存并核对所用金额；历史名义日期与真实精确首见时间仍有区别。原先按名义日期后一个交易日测量的收益保留其时钟限制，不能改称准确捕捉全部首日反应。未建立历史一致预期、完整信息计价或独立价格因果效应；本轮新增账户、收益测试、拟合参数、前瞻判断均为0。完整账户费用后夏普1.2及年化10%目标仍未实现。", "",
        "## 明细与复算", "",
    ]
    for label, path in [
        ("三例信息增量计算", OUT / "三例信息增量计算.json"),
        ("原公告金额与页码", OUT / "reviewed_source_facts.json"),
        ("49条报告的先行披露候选", OUT / "49条报告的先行披露候选.csv"),
        ("候选原文索引", OUT / "先行披露候选原文索引.csv"),
        ("原公告文件记录", OUT / "sources/receipts.json"),
        ("计算脚本", ROOT / "research/historical_disclosure_novelty_v1.py"),
    ]:
        lines.append(f"- [{label}](<{path.as_posix()}>)")
    (OUT / "历史发现_财报高分与真正新增信息.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
