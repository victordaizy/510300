"""保存快速初筛的简短结论及当前优先级，不制作交付包。"""
import json
import numpy as np
import pandas as pd

from research.factor96_rapid_feasibility_v1 import ROOT, OUT, IDS, read, save, core, REQUEST
from scripts.record_factor96_remaining_changes_v1 import update


def main():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    metrics = pd.read_csv(OUT / "all_account_metrics.csv")
    assert len(metrics) == 88 and result["causal_prefix_check"] == "PASS"
    recomputed = 0
    for row in metrics.itertuples():
        ledger = pd.read_parquet(ROOT / row.ledger_path)
        values = core.metrics(ledger, row.capital)
        for key in ("cagr", "max_drawdown", "end_equity", "commission", "slippage"):
            assert np.isclose(values[key], getattr(row, key), rtol=1e-10, atol=1e-8), (row.policy, key)
        if pd.notna(row.net_sharpe):
            assert np.isclose(values["net_sharpe"], row.net_sharpe, atol=1e-10)
        recomputed += 1
    primary = metrics[(metrics.capital == 200000) & (metrics.cost == "STRESS")]
    signals = primary[~primary.policy.isin(["CASH", "BUY_HOLD_50"])].sort_values("net_sharpe", ascending=False)
    benchmark = primary[primary.policy == "BUY_HOLD_50"].iloc[0]
    names = {r["id"]: r["name"] for r in read(OUT / "original_factor_cards.json")}
    lines = ["已改为先简单检验可行性，再展开有成效的方向。原公告逐条补证暂缓。",
        "本轮10张因子卡、20个高低方向，使用2017年1月3日至2025年12月31日2186个交易日。统一10个开盘间隔持有期，20万元与2万元、基础与压力成本，共88个账户。所有现金日、分红、整手、T+1、既有仓位与风险预算均保留。",
        "本轮20万元压力成本下，20个方向中仅3个年化收益为正，没有方向通过预先设定的进一步研究初筛线，更未达到最终夏普1.2与年化10%的联合目标。以下是本轮排序最靠前的简单用法，不是推荐策略。",
        "| 简单用法 | 压力成本夏普 | 年化收益 | 最大回撤 | 入场次数 |\n|---|---:|---:|---:|---:|"]
    for row in signals.head(5).itertuples():
        key, side = row.policy.split("_")
        label = names[key] + ("高状态" if side == "HIGH" else "低状态")
        lines.append(f"| {label} | {row.net_sharpe:.3f} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.entries} |")
    lines += [f"同期半仓买入持有基准：夏普{benchmark.net_sharpe:.3f}，年化{benchmark.cagr:.2%}，最大回撤{benchmark.max_drawdown:.2%}；基准也未达到联合目标。",
        "最好的一条P01高状态在20万元压力成本下年化仅0.53%，平均仓位约9.65%；在2万元账户中年化为负，最低佣金及整手限制的影响更明显。不能只用较好资金规模的结果替代完整评价。",
        "结论适用于这20条固定简单规则；不证明相应经济机制在所有设计中都无效。A06仅筛查速度维度，距离维度另存；多维卡没有被冒充为已完成原始策略。所有历史已经属于探索样本，因果计算和分段报告不等于独立前向验证。",
        "已重新读取88份账户账本核算收益、成本和回撤，并用截断未来输入重算截至2021年初的全部特征与信号，一致。没有参数网格、没有换窗补救、没有交付包。下一步对已存回购和资本供给资料做简单代理检验，优先得到机制层面的收益证据；暂不回到逐条公告补证。"]
    # 表格各行必须相邻；其余段落保留空行。
    text = "\n\n".join(lines[:3]) + "\n\n" + "\n".join(lines[3:9]) + "\n\n" + "\n\n".join(lines[9:]) + "\n"
    (OUT / "初筛结论.md").write_text(text, encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / "status.json", program / "factor_progress.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    objects = [read(p) for p in paths]; status, factors, mandate = objects
    before = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    old_counts = {key: status[key] for key in ("cumulative_executed_account_scenarios", "cumulative_admitted_account_scenarios", "cumulative_invalid_implementation_account_scenarios")}
    relative = OUT.relative_to(ROOT).as_posix()
    note = "按用户新优先级完成10张因子卡20方向的88个快速账户。20万元压力成本最佳P01_HIGH夏普0.183514、年化0.5264%、回撤8.4597%，无候选值得进一步展开，无联合达标。公告细节补证暂缓，继续已有信息类数据简单检验。"
    for factor in factors:
        if factor["id"] in IDS:
            block = signals[signals.policy.str.startswith(factor["id"] + "_")]
            factor["rapid_feasibility_screen"] = {"status": "COMPLETE_NO_PROMISING_SIMPLE_DIRECTION",
                "evidence_path": relative + "/result.json", "original_factor_definition_fully_validated": False,
                "direction_results": block[["policy", "net_sharpe", "cagr", "max_drawdown"]].to_dict("records")}
    status.update(at=core.now(), latest_round=result["study_id"], latest_result=relative + "/result.json",
        latest_progress_receipt=relative + "/saved_verification_receipt.json", last_research_result=note,
        latest_continuation_classification="PROGRESS_88_RAPID_ACCOUNT_SCENARIOS_AND_NEGATIVE_FEASIBILITY_RESULT",
        current_research_phase="RAPID_FEASIBILITY_FIRST_EXISTING_INFORMATION_PROXIES",
        research_priority="RAPID_FEASIBILITY_FIRST", source_detail_work_deferred_by_user=True,
        admitted_account_scenarios_this_round=88, invalid_implementation_accounts_this_round=0,
        cumulative_admitted_account_scenarios=old_counts["cumulative_admitted_account_scenarios"]+88,
        cumulative_executed_account_scenarios=old_counts["cumulative_executed_account_scenarios"]+88,
        rapid_screen_direction_tests=status.get("rapid_screen_direction_tests", 0)+20,
        last_completed_account_experiment=result["study_id"], last_completed_account_result=relative + "/result.json",
        next_independent_source_action="RAPID_SCREEN_EXISTING_REPURCHASE_AND_SUPPLY_PROXIES",
        qualified_candidates=[], goal_status="active", goal_achieved=False, delivery_package_required=False)
    for key in list(status):
        if key.endswith("_this_round") and key.startswith(("new_", "source_field_candidates", "reused_")):
            status[key] = 0
    mandate.update(current_round=result["study_id"], current_protocol=relative + "/protocol.json",
        latest_progress_receipt=relative + "/saved_verification_receipt.json", last_research_result=note,
        latest_continuation_report=relative + "/初筛结论.md", latest_research_priority_instruction=REQUEST,
        research_priority="RAPID_FEASIBILITY_FIRST", source_detail_work_deferred_by_user=True,
        research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    save(OUT / "saved_verification_receipt.json", {"at": core.now(), "status": "PASS_SAVED_LEDGER_RECOMPUTATION_AND_CAUSAL_FEATURE_PREFIX",
        "recomputed_accounts": recomputed, "max_accounting_error": result["accounting_identity_max_error"],
        "new_network_requests": 0, "external_review": "NOT_PERFORMED", "goal_achieved": False})
    for path, obj in zip(paths, objects):
        update(path, core.clean(obj))
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": core.now(), "before": before,
        "after": [{"path": str(p), "sha256": core.digest(p)} for p in paths], "prior_account_counts": old_counts,
        "new_account_scenarios": 88, "goal_achieved": False, "delivery_package_created": False})
    print("88个账户已登记，初筛无候选通过；研究优先级保持先可行性、后细节。")


if __name__ == "__main__":
    main()
