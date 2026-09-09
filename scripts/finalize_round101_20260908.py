"""核对保存成交額方向与四账户，简洁交付第101轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.directional_turnover_recovery_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300成交额方向恢复_第101轮_20260908"
DOCUMENT = OUT / "成交额方向恢复_结果及中文规则.md"
NEXT = ROOT / "docs/510300_AFTER_DIRECTIONAL_TURNOVER_RECOVERY_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 100 and not OUT.exists(), "第101轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时需另作验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "directional_turnover_factors.parquet")
    require(pd.DatetimeIndex(f.date).equals(pd.DatetimeIndex(data.date)), "成交额方向完整日历不同")
    require(np.isfinite(data[["high", "low", "close", "dividend", "amount"]]).all().all(), "当前保存核对需针对实际缺失重新定义全局递推")
    units, wealth = 1., []
    for row in data.itertuples():
        cash = units*row.dividend
        wealth.append(units*(row.high+row.low+row.close)/3+cash)
        units += cash/row.close
    positive, negative = np.full(len(data), np.nan), np.full(len(data), np.nan)
    for t in range(1, len(data)):
        change = wealth[t]-wealth[t-1]
        direction = 0 if abs(change) <= cfg["numerical_tie_tolerance"]*max(1., abs(wealth[t]), abs(wealth[t-1])) else int(np.sign(change))
        require(direction == f.adjusted_price_direction.iloc[t], "全局含分红财富与局部方向不同")
        positive[t] = data.amount.iloc[t] if direction > 0 else 0.
        negative[t] = data.amount.iloc[t] if direction < 0 else 0.
    score, checks = np.full(len(data), np.nan), []
    for t in range(cfg["flow_period"], len(data)):
        pos, neg = sum(positive[t-cfg["flow_period"]+1:t+1]), sum(negative[t-cfg["flow_period"]+1:t+1])
        score[t] = 100*pos/(pos+neg) if pos+neg > 0 else np.nan
        np.testing.assert_allclose([pos, neg, score[t]], [f.positive_amount_window_cny.iloc[t], f.negative_amount_window_cny.iloc[t], f.up_amount_share.iloc[t]], atol=1e-12, rtol=0, equal_nan=True)
        checks.append({"date": data.date.iloc[t], "independent_positive_amount_cny": pos, "independent_negative_amount_cny": neg, "independent_up_amount_share": score[t]})
    prior = np.r_[np.nan, score[:-1]]
    np.testing.assert_allclose(score, f.up_amount_share, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_array_equal(np.isfinite(prior) & np.isfinite(score) & (prior <= cfg["recovery_threshold"]) & (score > cfg["recovery_threshold"]) & (score < cfg["balance_threshold"]), f.raw_entry.astype(bool))
    np.testing.assert_array_equal(np.isfinite(score) & (score >= cfg["balance_threshold"]), f.raw_exit)
    accounts, cycles = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])), "实际账本与完整原点日历不同")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "不是下一开盘执行")
            np.testing.assert_allclose(decisions.up_amount_share, f.up_amount_share.iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]-1, ledger.net_return, atol=1e-12, rtol=0)
            require(decisions.loc[decisions.requested_quantity.gt(0), "raw_entry"].eq(1).all(), "无穿越因子却请求买入")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in complete)
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            require(all(value_equal(actual[k], saved[k]) for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]), "保存账户指标不同")
            accounts.append({"period": period, "cost": cost, "cycles": len(complete), "losing_cycles": sum(c["net_profit"] < 0 for c in complete), **actual})
    for name, rows in [("saved_independent_turnover_checks.csv", checks), ("saved_account_checks.csv", accounts), ("saved_actual_cycles.csv", cycles)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    a = pd.read_csv(RESEARCH / "earlier_diagnostic/BASE" / f"{PRIMARY}_cycles.csv")
    b = pd.read_csv(RESEARCH / "earlier_diagnostic/STRESS" / f"{PRIMARY}_cycles.csv")
    diff = a.merge(b, on="cycle_id", suffixes=("_base", "_stress"))
    diff = diff[diff.exit_date_base.ne(diff.exit_date_stress)]
    require(len(diff) == 1 and diff.entry_date_base.iloc[0] == "2018-07-02" and diff.exit_date_base.iloc[0] == "2018-07-06" and diff.exit_date_stress.iloc[0] == "2018-07-05", "费用触发实际退出差异与保存发现不同")
    diff.to_csv(RESEARCH / "费用导致的实际退出日期差异.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_DIVIDEND_DIRECTION_REAL_AMOUNT_AND_ACTUAL_ACCOUNTS_CHECKED", "independent_full_windows": len(checks), "new_accounts_checked": len(accounts), "complete_actual_cycles": len(cycles), "cost_driven_exit_date_difference_cycles": len(diff), "new_models_or_account_replays_during_check": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_DIRECTIONAL_TURNOVER_RECOVERY_NO_LOCAL_IMPROVEMENT_NO_SHARPE_TARGET"
    decision = "第101轮未达标。主基础／压力夏普0.022／负0.045，较早0.454／0.441；主基础年化0.015%、回撤9.93%，较早基础年化1.466%、回撤9.02%。成交额方向从低位恢复并未带来足够收益，关闭本次固定规则，不改窗口、门槛或方向救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第101轮之后：一项浮盈与回撤交互输入", "", decision, "", "101一设置四新账户、五必要测试3.96秒、账户约7.40秒。3442完整成交额方向窗口、14预热原点；主11周期22成交93持股收盘，较早6周期12成交，基础44收盘、压力43收盘。两费用共用输入，压力早2018年7月2日一笔交易更早达到自身含费用4%止损，于7月5日退出，基础7月6日退出；因此压力回撤略小，不是更改费用。全部来源、方向、34周期及完整账户已核，不重跑。", "", "100之后保存状态诊断已经完成：reports/research/510300_saved_reference_own_exit_states_20260908/result.json。四账户1074个持股收盘中，486个同一纯学习单次实际买入且已有成熟模型的可比较状态，预测符号与两日退出确认均零差异。其余408个无成熟模型、180个涉及其他进入或部分成交。没有计算反事实收益，暂不开发实际状态替换。此前方向说明提到第81轮时须准确理解：第81轮来源修正是参考入场锚点采用每日真实shares，不是将第91轮退出模型改为合并账户自己的状态。", "", "急跌专用退出学习暂不启动：第91轮2013年起连续PANIC_ONLY保存参考只有12个完整周期，不在这条路线上降低样本要求或套用另一进入策略的模型。第31轮原退出训练三条进入是趋势反弹、偏离均值后回升、D60日内隔夜，不含急跌专用模型。", "", "下一102准备一个局部因素修改：原D60学习退出的八个输入保留，只加一项‘浮盈回撤交互’，等于当前实际含分红浮盈的正部乘当前周期回撤。浮盈为正时，模型可以给回撤不同的边际作用；处于亏损时该已知乘积为零。任何原输入未知则该项仍未知，不能把未知浮盈当零。它是连续交互，不添加市场窗口、盈亏门槛、交易规则或新数据。限定研究退出与继续价值的实现和文档检索未找到此明确交互的已运行版本；不声称整个未索引工作区从未有类似模型。", "", "继续使用原第31轮all_reference_samples.parquet的D60_INTRA成熟样本、原月首训练日程、最近20完整周期、至少10周期100状态、等周期权重、标准化后截断至正负5、岭惩罚1和原两次负预测退出。当前尚未建立该第九输入、训练模型、登记配置或运行102账户。只拟合这个有限设置，复用原32学习退出、原自然退出、91局部候选和买持保存对照。若出现局部改善才进一步考虑预算层，不先生成组合。原模型和已失败非线性家族原结果保留，不重新选其参数。", "", "仅新输入构造、时钟、标准化及真实退出必要测试后冻结并直接跑账户；暂不下载、生成GPT包或长系数文档。所有进出规则继续用中文。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第101轮：成交额涨跌分布恢复", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 结果说明", "", "主基础仅93个持股收盘、11个完整周期，平均股票比例5.78%；较早基础44个持股收盘、6个完整周期，平均股票比例3.60%。交易较少没有自动产生更高夏普，两段基础年化都低于同期买入持有。", "", "较早压力费用最大回撤8.68%，略低于基础9.02%。保存交易显示2018年7月2日进入的周期，由于自身费用更高，压力于7月5日止损、基础于7月6日止损。持仓和费用会共同改变实际触发日；不能只从费用高低假设账户回撤的顺序。", "", "五项必要测试通过；3442个完整方向窗口、四账户共34个完整周期已核对。仅复用旧对照，没有重训旧模型、重跑旧账户或制作审阅包。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 101, "study": result["study_id"], "title": "成交额涨跌分布恢复后的进入退出", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 8
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND101_DIRECTIONAL_TURNOVER_NO_SHARPE_TARGET_FULL_GOAL_NOT_MET", count_warning="101轮，377不同设置，393已评价来源版本，398登记含5旧未运行，1452主评价记录。", next_work={"status": "PROFIT_DRAWDOWN_INTERACTION_METHOD_PREPARED_NOT_REGISTERED", "focus": "保留原八项退出及全部旧训练规则，只检查一项浮盈正部乘周期回撤的连续交互", "source": str(NEXT.relative_to(ROOT))}, process_state_note="101四新账户及五必要测试完成；102只有具体单因子设计，尚未训练、登记或运行账户。")
    index["deliveries"].append({"created_at": now(), "type": "DIRECTIONAL_TURNOVER_RECOVERY_ROUND101_FAST_CHINESE_RESULTS", "rounds": [101], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "账户周期": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
