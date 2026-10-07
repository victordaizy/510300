"""登记发布前基准和资金原因，本轮不改变原预测。"""

from company_revision_driver_sources_v1 import now, save
from domestic_event_baseline_calculate_v1 import read
from domestic_event_baseline_sources_v1 import ROOT, OUT


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮已经登记。")
    facts = read(OUT / "reviewed_facts.json")
    report = "数据公布前的预期与资金原因.md"
    if not (OUT / report).exists():
        raise SystemExit("本轮报告尚未保存。")
    previous = read(ROOT / "reports/research/510300_september_rate_shock_v1/result.json")
    stamp = now()
    next_question = "在真实新信息到达后按已存基准评价原F1/F2/F3；资金数据先取得官方最终值，PMI按需求、交付和成本构成解释，再比较新信息与实际价格。"
    result = {
        "study_id": "510300_DOMESTIC_EVENT_BASELINE_V1", "recorded_at": stamp,
        "previous_goal_turn_classification": previous["continuation_classification"],
        "continuation_classification": "PROGRESS_PRE_EVENT_EXPECTATIONS_AND_FUNDING_ROLLOVER_DECOMPOSITION",
        "status": "PUBLIC_EXPECTATIONS_CAPTURED_NEW_RELEASES_PENDING_NET_EDGE_UNPROVEN",
        "concrete_progress": [
            "发布前保存制造业50.1及非制造业49.3公开预期，分开TE模型50.0",
            "分解7至8月PMI构成，识别新订单贡献及配送逆向作用",
            "核对9月29日7890亿元操作与6960亿元到期，得到930亿元已知逆回购本金净增",
            "提前确定9月30日7065亿元已知到期本金，并区分政策容量与实际执行",
        ],
        "new_accounts": 0, "new_strategy_return_tests": 0, "new_forecast_cards": 0, "new_fitted_parameters": 0,
        "existing_account_counts": previous["existing_account_counts"],
        "original_forecasts_unchanged": True, "historical_rejections_unchanged": True,
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "goal_status": "active", "goal_achieved": False, "not_a_blocked_turn": True,
        "live_handle": None, "verified_wait": False,
        "orders_authorized": False, "report": report, "next_question": next_question,
    }
    save(OUT / "result.json", result)
    rel = OUT.relative_to(ROOT).as_posix()
    path = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(path)
    protocol.update({"latest_concrete_diagnostic": rel + "/result.json",
                     "latest_pre_release_expectations": rel + "/reviewed_facts.json",
                     "latest_funding_rollover_diagnostic": rel + "/reviewed_facts.json",
                     "current_pmi_expectation_status": facts["public_expectation_snapshot"]["status"],
                     "current_source_admission": rel + "/reviewed_facts.json", "updated_at": stamp})
    save(path, protocol)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update({"current_round": result["study_id"], "latest_progress_receipt": rel + "/result.json",
                    "latest_continuation_report": rel + "/" + report,
                    "latest_continuation_classification": result["continuation_classification"],
                    "last_research_result": "事前保存PMI公开预期，原分项预测不改。9月29日所核对逆回购本金净增930亿元，明日已知到期7065亿元；不把续作当全额新增或股票买盘。当前净优势未证。",
                    "last_source_result": "取得6份央行原公告和TE日历原网页；Investing正文由网页工具保存。货币网响应为业务错误，DR007未取得，没有替代。",
                    "latest_driver_diagnostic_at": stamp, "next_research_question": next_question})
    save(path, mandate)
    path = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- DOMESTIC_EVENT_BASELINE_V1_20260929 -->

## 2026-09-29 国内发布前预期与跨季续作原因

事前保存制造业50.1、非制造业49.3公开预期，分开TE模型50.0；用官方分项重建7至8月PMI改善0.585点，其中订单贡献0.630点、配送加快贡献-0.090点。央行29日7890亿元操作对应6960亿元已知到期，净增930亿元；30日已知到期7065亿元。容量、续作与股票买盘分开。原F1/F2/F3仍待未来信息，官方最终DR007未取。新增账户和收益测试0；目标active且未实现，本轮PROGRESS，不是已核实的后台等待。

详见[数据公布前的预期与资金原因](<{(OUT / report).as_posix()}>)。

"""
    path.write_text(head + path.read_text(encoding="utf-8"), encoding="utf-8")
    print("已登记发布前基准和资金分解，原预测及风险预算保持不变。")


if __name__ == "__main__":
    main()
