"""登记本轮原因链和真实进展，不修改旧预测及表现。"""

from company_revision_driver_sources_v1 import now, save
from september_rate_shock_calculate_v1 import read
from september_rate_shock_sources_v1 import ROOT, OUT


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮已经登记。")
    facts = read(OUT / "reviewed_facts.json")
    report = "利率变化背后的约束与剩余预期差.md"
    if not (OUT / report).exists():
        raise SystemExit("缺少本轮报告。")
    previous = read(ROOT / "reports/research/510300_index_repricing_odds_v1/result.json")
    stamp = now()
    next_question = "用下一次真实新信息区分订单增强、供给约束缓解和需求转弱；国内既有F1/F2/F3尚待发布或观察，保留原预测，优先判断是否出现尚未被价格反映的约束变化。"
    result = {
        "study_id": "510300_SEPTEMBER_RATE_SHOCK_V1", "recorded_at": stamp,
        "previous_goal_turn_classification": previous["continuation_classification"],
        "continuation_classification": "PROGRESS_EVENT_REAL_YIELD_DECOMPOSITION_AND_EXPECTATION_TIMING",
        "status": "CAUSES_AND_REPRICING_WINDOW_REFINED_REMAINING_NET_EDGE_UNPROVEN",
        "concrete_progress": [
            "从美联储同表补上名义和实际收益率；事件日15个基点变化由13加2构成",
            "取得有正文和发布时间的历史预告代理，并保留事后更新及调查缺口",
            "定位需求、有效供给和成本约束；辨别当前强度与一年期预期",
            "对齐北京时间和A股交易日，隔离公布前跌幅与公布后反应",
        ],
        "new_accounts": 0, "new_strategy_return_tests": 0, "new_forecast_cards": 0, "new_fitted_parameters": 0,
        "existing_account_counts": previous["existing_account_counts"],
        "original_forecasts_unchanged": True, "historical_rejections_unchanged": True,
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "goal_status": "active", "goal_achieved": False, "not_a_blocked_turn": True,
        "orders_authorized": False, "report": report, "next_question": next_question,
        "source_observation_cutoff": facts["rates"]["last_observation"],
    }
    save(OUT / "result.json", result)
    rel = OUT.relative_to(ROOT).as_posix()
    path = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(path)
    protocol.update({"latest_concrete_diagnostic": rel + "/result.json", "latest_external_rate_event": rel + "/reviewed_facts.json", "updated_at": stamp})
    save(path, protocol)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update({
        "current_round": result["study_id"], "latest_progress_receipt": rel + "/result.json",
        "latest_continuation_report": rel + "/" + report,
        "latest_continuation_classification": result["continuation_classification"],
        "last_research_result": "9月23日10年期名义收益率上行15bp，其中TIPS实际13bp、差额2bp；需求与有效供给约束支持政策路径重估解释，但不识别纯期限或政策份额。A股首个反应窗口为24日，当前净优势仍未证。",
        "last_source_result": "取得美联储H.15、官员讲话及日历和公开预告正文；调查原件未取，预告事后更新。S&P官方PDF仅取得网页工具文本，直连403。",
        "latest_driver_diagnostic_at": stamp, "next_research_question": next_question,
    })
    save(path, mandate)
    path = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- SEPTEMBER_RATE_SHOCK_V1_20260929 -->

## 2026-09-29 美国需求约束、实际收益率与事件时序

取得美联储同表数据，9月22至23日10年名义收益率增加15bp，实际13bp、差额2bp；至25日累计21、20、1bp。需求加快与交付、人员和成本约束并存，未来一年产出预期未变。公开预告正文取得，但属于事后重建代理。PMI公布在23日A股收盘后；24日510300低开0.2614%，日内下降1.3761%，不将共现视作单一因果。原预测与失败不改，新增账户和收益测试0；目标active且未实现，本轮PROGRESS。

详见[利率变化背后的约束与剩余预期差](<{(OUT / report).as_posix()}>)。

"""
    path.write_text(head + path.read_text(encoding="utf-8"), encoding="utf-8")
    print("本轮结果及入口已登记；目标未实现，原预测和风险预算保持原样。")


if __name__ == "__main__":
    main()
