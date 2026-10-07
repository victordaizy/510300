"""保存本轮经营原因研究结果并更新入口；原预测和账户结果保持原样。"""

import json

import company_revision_driver_sources_v1 as source
from consumer_health_driver_sources_v1 import OUT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮结果已保存，不重复更新入口。")
    report_name = "消费医药因子背后的原因与兑现条件.md"
    if not (OUT / report_name).exists():
        raise SystemExit("缺少本轮研究正文，停止更新入口。")
    facts = read(OUT / "reviewed_facts.json")
    receipts = read(OUT / "source_result.json")["receipts"] + read(OUT / "policy_context_result.json")["receipts"]
    if len(receipts) != 8 or any(r.get("http_status") != 200 or "失败" in r.get("status", "") for r in receipts):
        raise SystemExit("原件取得结果与正文不一致，停止写入结论。")
    mt = facts["moutai"]["derived"]
    cash_parts = sum(mt[k] for k in ["deposit_flow_delta_cny_100million", "interbank_flow_delta_cny_100million", "sales_cash_delta_cny_100million", "remaining_cashflow_changes_cny_100million"])
    if abs(cash_parts - mt["cfo_delta_cny_100million"]) > 1e-8:
        raise SystemExit("现金流分解未闭合。")
    weights = read(source.ROOT / "reports/research/510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json")
    subset = sum(r["weight_pct"] for r in weights["top10"] if r["symbol"] in ["600519", "000333", "603259"])
    if abs(subset - facts["new_three_weight_pct"]) > 1e-10:
        raise SystemExit("沿用权重的加总与正文不一致。")
    stamp = source.now()
    next_question = "停止扩展公司名单，把既定权重的需求定价、汇率风险补偿和交付现金流归并为少数情景；识别超过既有指引的新信息如何形成当前价格下的510300净收益赔率。"
    result = {
        "study_id": "510300_CONSUMER_HEALTH_DRIVER_BRIDGE_V1", "recorded_at": stamp,
        "previous_goal_turn_classification": "PROGRESS_PAIRED_FORECAST_COMPONENT_REVISIONS_AND_OPERATING_BASELINE",
        "continuation_classification": "PROGRESS_FACTOR_MEANING_CAUSES_AND_TOP10_OPERATING_COVERAGE",
        "status": "PHASE_DEPENDENT_FACTOR_INTERPRETATION_IDENTIFIED_INDEX_EDGE_UNPROVEN",
        "concrete_progress": [
            "茅台现金流增量拆到同业款项和吸收存款，预收款变化追到渠道模式与投放政策",
            "美的扣非分类差异与实际毛利、相关汇兑衍生损益及营运资本分别解释",
            "药明持续经营毛利改善与已上调指引形成未来兑现基准，订单存量不提前当收入",
            "完成既定十大权重经营线索分析并收住公司范围，不虚称全指数盈利覆盖",
        ],
        "sources_obtained": 8, "new_accounts": 0, "new_strategy_return_tests": 0, "new_forecast_cards": 0,
        "existing_account_counts": {"admitted": 800, "executed": 952},
        "existing_forecasts_unchanged": True, "historical_rejections_unchanged": True,
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "goal_achieved": False, "goal_status": "active", "not_a_blocked_turn": True,
        "orders_authorized": False, "report": report_name, "next_question": next_question,
        "main_limit": "已识别旧披露中的经营原因和后续基准，未识别9月新预期差或当前可成交价后的指数净优势。",
    }
    source.save(OUT / "result.json", result)
    rel = OUT.relative_to(source.ROOT).as_posix()
    path = source.ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(path)
    protocol.update({
        "latest_concrete_diagnostic": rel + "/result.json",
        "latest_consumer_health_driver_bridge": rel + "/reviewed_facts.json",
        "factor_meaning_change_rule": "销售模式、合并范围、损益分类及资金配置改变时，先重建因子含义；不自动将预收款、扣非或现金流变化解释为需求变化。",
        "hedging_scope_rule": "货币项目与衍生收益按敞口、风险类型及期限匹配；相关损益合计不冒充精确对冲净效果。",
        "order_conversion_rule": "订单存量、履约确认和回款分开；后期项目与产能效率的改善须比较已上调指引。",
        "company_scope_next_stage": "既定十大权重的经营线索已补齐，集中做共享驱动与指数赔率，不继续无边扩张公司名单。",
        "updated_at": stamp,
    })
    source.save(path, protocol)
    path = source.ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update({
        "current_round": result["study_id"], "latest_progress_receipt": rel + "/result.json",
        "latest_continuation_report": rel + "/" + report_name,
        "latest_continuation_classification": result["continuation_classification"],
        "latest_consumer_health_driver_bridge": rel + "/result.json",
        "last_research_result": "茅台现金流增量主要对应金融业务两项流量变化，美的扣非分类与真实毛利压力并存，药明持续经营改善且新指引已要求H2收入296至316亿元；既定十大经营线索补齐，ETF净优势未证。",
        "last_source_result": "取得三份半年报、一份既有季度指引、两份公司经营网页及两份政策背景原文；核对关键表页，未开展新收益测试。",
        "latest_driver_diagnostic_at": stamp, "next_research_question": next_question,
    })
    source.save(path, mandate)
    path = source.ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- CONSUMER_HEALTH_DRIVER_BRIDGE_V1_20260929 -->

## 2026-09-29 消费医药因子含义与未来兑现条件

补齐既定十大权重经营线索。茅台经营现金流增量的99.36%可由两项金融业务现金流变动算术解释；美的扣非分类影响与毛利压力并存；药明持续经营毛利率提高7.80个百分点，8月指引已要求H2收入296.03至316.03亿元。三者不构成一致盈利扩张，订单与现金、指引与预期差分开。公司范围收束，转向共享驱动与指数赔率。原预测不改，新增账户与收益测试均0，旧800/952与失败不变；目标active且未实现，本轮PROGRESS。

详见[消费医药因子背后的原因与兑现条件](<{(OUT / report_name).as_posix()}>)。

"""
    path.write_text(head + path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"报告": str(OUT / report_name), "状态": result["continuation_classification"], "新增账户": 0, "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
