"""登记快速事件代理的负面初筛结论，并明确无效方向不继续细化。"""
import numpy as np
import pandas as pd

from research.factor96_rapid_capital_proxy_v1 import OUT, PRIOR, ROOT, read, save, core
from scripts.record_factor96_remaining_changes_v1 import update


def main():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    table = pd.read_csv(OUT / "all_account_metrics.csv")
    assert len(table) == 16
    for row in table.itertuples():
        ledger = pd.read_parquet(ROOT / row.ledger_path)
        values = core.metrics(ledger, row.capital)
        for key in ("cagr", "max_drawdown", "end_equity", "commission", "slippage"):
            assert np.isclose(values[key], getattr(row, key), rtol=1e-10, atol=1e-8)
        if pd.notna(row.net_sharpe):
            assert np.isclose(values["net_sharpe"], row.net_sharpe, atol=1e-10)
    coverage = read(OUT / "coverage_diagnostic.json")
    primary = table[(table.capital == 200000) & (table.cost == "STRESS")].copy()
    prior = pd.read_csv(PRIOR / "all_account_metrics.csv")
    previous_primary = prior[(prior.capital == 200000) & (prior.cost == "STRESS")]
    combined = pd.concat([previous_primary, primary], ignore_index=True)
    combined.to_csv(OUT / "本轮全部方向20万元压力结果.csv", index=False, encoding="utf-8-sig")
    names = {"REPURCHASE_HIGH": "回购活跃代理", "REPURCHASE_LOW": "回购低活跃对照",
             "SUPPLY_RELIEF": "已知配股供给退潮代理", "SUPPLY_INCREASE": "已知配股供给增加对照"}
    lines = ["最终目标继续是只交易510300及人民币现金，实现完整账户成本后夏普至少1.2，并联合检验年化10%和回撤约束。",
        "用户要求抓大放小，先简单检验可行性；无效则换收益机制。用户随后明确继续只交易510300。本轮已经依照这个优先级完成10张卡的20个简单方向，以及回购、供给的4个事件代理方向，共104个账户场景，全部保留。",
        "这次新增16个事件代理账户，使用同一2017至2025年样本、持有期、成本和风险合同。20万元压力成本结果如下：",
        "| 代理方向 | 净夏普 | 年化收益 | 最大回撤 | 入场次数 |",
        "|---|---:|---:|---:|---:|"]
    for row in primary.sort_values("net_sharpe", ascending=False).itertuples():
        sharpe = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "无交易"
        lines.append(f"| {names[row.policy]} | {sharpe} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.entries} |")
    lines += ["回购活跃代理只有13次入场，年化约0.52%；供给退潮代理为负收益。均未通过事前初筛线，未达到最终联合目标，当前不支持进一步投入这些代理的公告细节。",
        "回购按正的新增实施披露批次计数；供给按当时已经公开、未来20交易日内拟上市的配股事件计数。都使用当时成分，保留披露时钟，不将集团金额或未核实限售量加入。这些是所收集数据里的事件代理，零不等于全市场没有事件，资料覆盖仍不完整，不能由本轮否定所有回购或供给机制。",
        "之前价格及融资单变量初筛20个方向的最佳压力成本夏普仅0.184。相关性诊断中，量价效率差与未来10日收益的Spearman约-0.116，日内相对隔夜强弱约0.102，但对应简单账户没有通过；不能仅凭相关系数继续投入。",
        "本轮动作：结束上述固定简单规则的深化工作，完整保存负面结果；公告逐条补证继续暂缓。下一轮只交易510300，换到ETF申赎需求、期现交易压力等不同的经济问题，先盘点能立即检验的既有数据；不因数据不足就自动建设大库。若新方向也没有足够账户效果，再与用户重新选题。",
        "104个账户没有任何正式达标候选。所有结果属于已见历史探索，独立前向观察仍为0，没有下单权限，也没有交付包。"]
    (OUT / "研究方向与初筛结论.md").write_text("\n\n".join(lines[:3]) + "\n\n" + "\n".join(lines[3:9]) + "\n\n" + "\n\n".join(lines[9:]) + "\n", encoding="utf-8")
    save(OUT / "saved_verification_receipt.json", {"at": core.now(), "status": "PASS_16_SAVED_ACCOUNT_RECOMPUTATIONS",
        "accounts": 16, "max_accounting_error": float(table.max_identity_error.max()),
        "coverage_diagnostic": coverage, "classification": result["classification"],
        "external_review": "NOT_PERFORMED", "goal_achieved": False})
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / "status.json", program / "strategy_progress.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    objects = [read(p) for p in paths]; status, strategies, mandate = objects
    before = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    assert status["latest_round"] == "510300_FACTOR96_RAPID_FEASIBILITY_V1"
    relative = OUT.relative_to(ROOT).as_posix()
    note = "新增16个回购与供给代理账户。20万元压力成本回购活跃Sharpe0.419655/CAGR0.5152%/13次入场，供给退潮Sharpe-0.558016/CAGR-0.7510%。均不值得当前深化；连同前批共104个快速账户，正式达标0。用户确认仍只交易510300，无效就换收益机制；公告补证暂缓。"
    for r in strategies:
        if r["id"] in ("T12", "T13"):
            r["rapid_proxy_feasibility"] = {"status": "NO_PROMISING_FIXED_SIMPLE_PROXY",
                "evidence_path": relative + "/result.json", "strict_original_strategy_validated": False,
                "source_detail_deferred_by_user": True}
    status.update(at=core.now(), latest_round=result["study_id"], latest_result=relative + "/result.json",
        latest_progress_receipt=relative + "/saved_verification_receipt.json", last_research_result=note,
        current_research_phase="RESEARCH_DIRECTION_RESET_AFTER_NEGATIVE_SIMPLE_SCREENS",
        latest_continuation_classification="PROGRESS_16_CAPITAL_PROXY_ACCOUNTS_AND_BRANCH_STOP_DECISION",
        admitted_account_scenarios_this_round=16, invalid_implementation_accounts_this_round=0,
        cumulative_admitted_account_scenarios=status["cumulative_admitted_account_scenarios"]+16,
        cumulative_executed_account_scenarios=status["cumulative_executed_account_scenarios"]+16,
        rapid_screen_direction_tests=status.get("rapid_screen_direction_tests", 0)+4,
        last_completed_account_experiment=result["study_id"], last_completed_account_result=relative + "/result.json",
        next_independent_source_action="DISCUSS_DIFFERENT_510300_MECHANISM_AND_USE_READY_DATA_FIRST",
        research_priority="RAPID_FEASIBILITY_FIRST", source_detail_work_deferred_by_user=True,
        stopped_simple_research_branches=["20_PRICE_AND_MARGIN_SINGLE_FACTOR_DIRECTIONS", "4_CAPITAL_BEHAVIOR_COUNT_PROXIES"],
        qualified_candidates=[], goal_status="active", goal_achieved=False, delivery_package_required=False)
    mandate.update(current_round=result["study_id"], current_protocol=relative + "/protocol.json",
        latest_progress_receipt=relative + "/saved_verification_receipt.json", last_research_result=note,
        latest_continuation_report=relative + "/研究方向与初筛结论.md",
        latest_research_direction_instruction="最终按实现目标决定去留，无效则换收益机制；用户确认仍只交易510300。",
        research_execution_state=status["current_research_phase"], research_priority="RAPID_FEASIBILITY_FIRST",
        source_detail_work_deferred_by_user=True, goal_status="active", goal_achieved=False)
    for path, obj in zip(paths, objects):
        update(path, core.clean(obj))
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": core.now(), "before": before,
        "after": [{"path": str(p), "sha256": core.digest(p)} for p in paths],
        "new_account_scenarios": 16, "goal_achieved": False, "delivery_package_created": False})
    print("已登记104个快速账户的合并结论：无正式达标，停止这些简单规则的深化，保持510300范围。")


if __name__ == "__main__":
    main()
