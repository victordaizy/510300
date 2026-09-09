"""复用四账户必要核对，尾部最优性用独立排序与次梯度检查。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "scripts/finalize_round83_20260908.py"
    require(not destination.exists(), "第83轮核对已建立")
    source = (ROOT / "scripts/finalize_round82_20260908.py").read_text(encoding="utf-8")
    source = source.split('    status = "COMPLETED_MIN_VARIANCE_MAIN_IMPROVEMENT_EARLY_WEAKER_TARGET_NOT_MET"')[0]
    source = source.replace("research.two_policy_min_variance_v1", "research.two_policy_tail_loss_v1")
    source = source.replace("510300最小方差预算_第82轮_20260908", "510300尾部损失预算_第83轮_20260908")
    source = source.replace("最小方差预算_结果及中文规则.md", "尾部损失预算_结果及中文规则.md")
    source = source.replace("510300_AFTER_MIN_VARIANCE_TAIL_LOSS_20260908.md", "510300_AFTER_TAIL_LOSS_WEALTH_BUDGET_20260908.md")
    source = source.replace('== 81 and not OUT.exists()', '== 82 and not OUT.exists()').replace("第82轮前序", "第83轮前序")
    start = source.index("                    covariance = np.cov(values,")
    end = source.index('                require(status == row.risk_status', start)
    source = source[:start]+'''                    status = "TAIL_LOSS_BUDGET_AVAILABLE"
                    require(row.linear_programs == 3, "完整尾部最优区间未求出")
                    lower, upper = row.optimal_budget_lower, row.optimal_budget_upper
                    require(0 <= lower <= upper+1e-8 and upper <= 1, "尾部最优预算区间无效")
                    weights = np.array([np.clip(previous[0], lower, upper), 1-np.clip(previous[0], lower, upper)])
                    w = weights[0]
                    mass = len(values)*(1-cfg["tail_confidence"])
                    def tail_at(budget):
                        losses = -(values @ [budget, 1-budget])
                        ordered = np.sort(losses)[::-1]
                        whole = int(np.floor(mass))
                        return float((ordered[:whole].sum()+(mass-whole)*ordered[whole])/mass)
                    observed = tail_at(w)
                    np.testing.assert_allclose([row.selected_tail_loss, row.optimal_tail_loss, tail_at(lower), tail_at(upper)], observed, atol=1e-8, rtol=0)
                    require(observed <= tail_at(previous[0])+1e-8, "最小化后的尾部损失反而变大")
                    losses = -(values @ weights)
                    derivative = values[:, 1]-values[:, 0]
                    threshold = np.sort(losses)[::-1][int(np.floor(mass))]
                    strict = losses > threshold+1e-12
                    tied = np.abs(losses-threshold) <= 1e-12
                    remaining = mass-int(strict.sum())
                    require(-1e-10 <= remaining <= tied.sum()+1e-10, "尾部质量未完整计入")
                    def fill_derivatives(ordered):
                        k = int(np.floor(remaining))
                        fraction = remaining-k
                        return ordered[:k].sum()+(fraction*ordered[k] if fraction > 1e-12 else 0.)
                    tie_derivatives = np.sort(derivative[tied])
                    low_gradient = (derivative[strict].sum()+fill_derivatives(tie_derivatives))/mass
                    high_gradient = (derivative[strict].sum()+fill_derivatives(tie_derivatives[::-1]))/mass
                    require((w <= 1e-8 and high_gradient >= -1e-8) or (w >= 1-1e-8 and low_gradient <= 1e-8) or
                        (1e-8 < w < 1-1e-8 and low_gradient <= 1e-8 and high_gradient >= -1e-8), "预算未满足凸尾部损失的全局最优条件")
''' + source[end:]
    source = source.replace('"KEY_MONTHLY_OPTIMUM_AND_COMPLETE_ACCOUNTS_CHECKED"', '"KEY_EMPIRICAL_TAIL_GLOBAL_OPTIMUM_AND_COMPLETE_ACCOUNTS_CHECKED"')
    source += '''    status = "COMPLETED_TAIL_LOSS_BUDGET_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False, "position_impact": 0,
        "decision": "两段夏普、年化均下降；主所有27次有效优化都给急跌全预算，使学习机会消失。停止尾部目标，不扫描置信度、窗口或附加收益项。"}, exclusive=True)
    NEXT.write_text("# 第83轮完成，下一项采用随参考累计净值变化的预算\\n\\n"
        "83主基础／压力夏普0.967884／0.943632，较早0.406921／0.370732，四项均低于82及76。主27次有效优化全为急跌预算100%，只剩64持仓收盘、11笔成交；较早32次有效优化中16次急跌全额、5次为零，246持仓收盘29成交。177次线性优化成功，12历史不足及原零波动处理每段照旧。此方向关闭，不改变95%尾部、窗口、预算上下限或新增预期收益项救回。\\n\\n"
        "下一84先固定单项累计财富预算：两条原基础参考净值从各自评价起点等额开始，预算每天随已实现累计净值的相对大小自然变化；不估计方差、协方差或尾部，不排名选单条、不截最近一年、不引入学习率。合成目标仍是预算乘两条原持有状态，下一开盘实际执行。只有方向，未登记或运行；需要明确缺失参考净值时保留无观点及此前预算，检查时钟和完整账户。\\n\\n"
        "有界查重未见这两条策略的同一累计净值配比实现。已读原自适应分配和45过去一年赚钱才进入、58日历月度启停，以及47第三信号，旧失败保留；本项没有把旧年度窗口缩短或调择优阈值。只复用原净值和信号即可，继续不补EPS和慢来源。\\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第83轮：按最差5%日损失分配预算", "",
        "主评价基础／压力净夏普0.968／0.944，较早历史0.407／0.371，四项都比第82轮及第76轮低。目标未达成，这条尾部优化方向关闭，不调整尾部比例救回。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["主基础年化2.97%、回撤3.58%，较早年化2.86%、回撤11.07%。主所有27次有效预算优化均选择急跌策略100%，学习预算被压到零，实际仅64个持仓收盘、5买6卖，持仓机会明显减少。较早32次有效更新中，16次急跌全额、5次急跌为零，实际246个持仓收盘、14买15卖。两费用没有整笔未成交。", "",
        "该结果解释了本轮为何不能保留：稀少交易在已见完整日损失样本里容易显得尾部风险小，优化会偏向不经常持仓的规则。最小化样本亏损并没有带来更好的真实收益风险表现，不能仅因求解器成功就接受策略。", "",
        "主80个月首中12次历史不足、41次原零波动，27次成功；较早60次中12次不足、16次零波动、32次成功。59个窗口合计177次线性优化，没有求解失败。预算排序复算、并列区间和凸函数最优条件已经检查，未重复运行优化器。", "",
        "## 完整中文规则与原因子", ""] + (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", "原两条策略全部进入、八因子学习、持有、退出及再次进入规则见同目录《沿用的两条策略全部中文规则.md》，原逐月系数直接复用已有文件。", "",
        f"12项必要测试通过；{len(checks)}次月度时钟及预算检查，四个新账户和{len(cycles)}个含分红周期已核对；完整保存因子、目标、现金、费用和成交随附。下一累计净值预算尚未登记或运行。", ""]
    DOCUMENT.write_text("\\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT / "沿用的两条策略全部中文规则.md")
    coefficient = ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coefficient, OUT / coefficient.name)
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 83, "study": result["study_id"], "title": "两条原策略最差5%日损失预算", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "risk_window_optimizations": 59, "linear_programs": 177,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND83_COMPLETE_TAIL_LOSS_REJECTED_TARGET_NOT_MET",
        count_warning="83轮，357不同设置，373已评价来源版本，378登记含5旧未运行，1270主评价记录；无效来源保留。",
        next_work={"status": "REFERENCE_CUMULATIVE_WEALTH_BUDGET_NOT_REGISTERED", "focus": "两条原策略随已实现累计净值自然变化的预算", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="81来源修正、82最小方差和83尾部风险均完成账户、必要核对与简洁交付；84累计财富仅方向。")
    index["deliveries"].append({"created_at": now(), "type": "TAIL_LOSS_ROUND83_FAST_CHINESE_RESULTS", "rounds": [83], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "相对76差额": differences, "集中": concentration}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
'''
    destination.write_text(source, encoding="utf-8")
    print("第83轮必要核对脚本已建立，不重复优化或重跑账户。", flush=True)


if __name__ == "__main__":
    main()
