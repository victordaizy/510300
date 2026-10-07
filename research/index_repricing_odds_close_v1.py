"""登记本轮估值观察和条件赔率结果，保留来源缺口与未计算表现。"""

from index_repricing_odds_sources_v1 import ROOT, OUT, now, save
from index_repricing_odds_calculate_v1 import read


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮已登记，不重复更新入口。")
    facts = read(OUT / "reviewed_facts.json")
    report_name = "从共同驱动到当前价格的收益边界.md"
    if not (OUT / report_name).exists() or not (OUT / "盈利修正与估值压缩的费用边界.png").exists():
        raise SystemExit("报告或图表不完整，停止更新入口。")
    previous = read(ROOT / "reports/research/510300_consumer_health_driver_bridge_v1/result.json")
    if previous["continuation_classification"] != "PROGRESS_FACTOR_MEANING_CAUSES_AND_TOP10_OPERATING_COVERAGE":
        raise SystemExit("上一轮状态与本轮范围不一致。")
    stamp = now()
    save(OUT / "source_admission.json", {
        "recorded_at": stamp,
        "usable": ["csi300_indicator", "csi300_perf价格字段", "etf_intraday时点报价"],
        "not_reconciled": ["市盈率1、市盈率2、接口peg及月度简报滚动市盈率的完整口径"],
        "calendar": {"status": "SEARCH_SNIPPET_ONLY_NOT_ADMITTED_AS_VERIFIED_EXPECTATION",
                     "url": "https://www.jpmhkwarrants.com/zh-cn/market-statistics/news/type/world/newsID/229734",
                     "additional_search_url": "https://jpmhkwarrants.com/zh_cn/market-statistics/news/type/cn/newsID/210391",
                     "manufacturing_headline_candidate": 50.1, "nonmanufacturing_headline_candidate": 49.3,
                     "direct_html_result": "只有站点通用框架，未取得文章正文；不能称已核实市场共识。",
                     "not_a_new_orders_expectation": True},
        "fred": {"DGS10": "ONE_REQUEST_READ_TIMEOUT", "DFII10": "ONE_REQUEST_READ_TIMEOUT", "current_rate_decomposition": "NOT_COMPUTED", "automatic_retries": False},
        "quote": {"date": facts["quote"]["date"], "time": facts["quote"]["time"], "not_close_or_fill": True},
    })
    next_question = "沿已出现的9月利率、政策或订单事件，定向取得发布前预期与发布后价格反应，判断远期增长下修、风险补偿和资金约束哪一项发生了变化，以及是否仍有未完成的重估；不扩展公司名单。"
    result = {
        "study_id": "510300_INDEX_REPRICING_ODDS_V1", "recorded_at": stamp,
        "previous_goal_turn_classification": previous["continuation_classification"],
        "continuation_classification": "PROGRESS_COMMON_DRIVERS_VALUATION_COMPARISON_AND_CONDITIONAL_NET_ODDS",
        "status": "PRICE_EARNINGS_MULTIPLE_BOUNDARY_COMPUTED_POSITIVE_EXPECTATION_UNPROVEN",
        "concrete_progress": ["取得中证每日PE1/PE2和同窗口价格，隔离未调和口径", "把十大权重经营原因归并为需求定价、交付效率和资金要求回报三条驱动", "以带时点盘口及原费用计算盈利修正能抵御的估值压缩边界", "明确宏观预期只有搜索线索、利率组成未取得，未伪造预期差和概率"],
        "new_accounts": 0, "new_strategy_return_tests": 0, "new_forecast_cards": 0,
        "conditional_cost_examples": 4, "new_fitted_parameters": 0,
        "existing_account_counts": previous["existing_account_counts"],
        "original_forecasts_unchanged": True, "historical_rejections_unchanged": True,
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "goal_status": "active", "goal_achieved": False, "not_a_blocked_turn": True,
        "orders_authorized": False, "report": report_name, "next_question": next_question,
    }
    save(OUT / "result.json", result)
    rel = OUT.relative_to(ROOT).as_posix()
    path = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(path)
    protocol.update({"latest_concrete_diagnostic": rel + "/result.json", "latest_repricing_odds": rel + "/reviewed_facts.json",
                     "valuation_comparability_rule": "总股本、计算用股本、滚动及接口字段口径未调和前，不混接估值或声称指数盈利修正；比值恒等式不识别风险溢价因果贡献。",
                     "conditional_odds_rule": "盈利修正与倍数变化共同决定价格情景，计入报价、费用和尾部风险；盈亏平衡概率不是估计胜率，情景损益不是策略表现。",
                     "current_source_admission": rel + "/source_admission.json", "updated_at": stamp})
    save(path, protocol)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update({"current_round": result["study_id"], "latest_progress_receipt": rel + "/result.json",
                    "latest_continuation_report": rel + "/" + report_name, "latest_continuation_classification": result["continuation_classification"],
                    "latest_repricing_odds": rel + "/result.json",
                    "last_research_result": "9月1至28日指数价格与同表PE2降幅接近，不能据此识别风险溢价。带时点盘口压力费用下，3%盈利上修约能抵御2.60%倍数下降；情景概率和指数净优势未证。",
                    "last_source_result": "取得官方估值表、行情接口和ETF盘中快照；公开PMI预期仍为摘要线索，两个FRED请求超时，未重试或计算缺失值。",
                    "latest_driver_diagnostic_at": stamp, "next_research_question": next_question})
    save(path, mandate)
    path = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- INDEX_REPRICING_ODDS_V1_20260929 -->

## 2026-09-29 共同驱动、估值变化与条件收益边界

9月1至28日指数价格下降5.87%，每日表计算用股本PE2下降5.89%；不同估值口径未调和，不把分母代理或倍数变化当因果识别。已将经营原因归并为三条共享驱动，以9月29日14:19:30盘口和原费用计算条件损益：3%盈利上修约能抵御2.60%倍数压缩。未估计真实情景概率；四组费用算例不算策略账户。PMI预期只取得搜索摘要，美债两序列超时，缺口保留。原预测与失败不改，新增账户和收益测试均0；目标active且未实现，本轮PROGRESS。

详见[从共同驱动到当前价格的收益边界](<{(OUT / report_name).as_posix()}>)。

"""
    path.write_text(head + path.read_text(encoding="utf-8"), encoding="utf-8")
    print("已保存研究结果和统一入口；原风险预算、预测卡与策略账户未修改。")


if __name__ == "__main__":
    main()
