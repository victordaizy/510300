"""核对限价、旧库存、真实成交周期及两种情景，关闭第110轮。"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.standing_profit_limit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, ASSUMPTIONS
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.panic_learned_equal_blend_v1 import panic_folder
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table

OUT = ROOT / "deliverables/510300事前止盈限价_第110轮_20260908"
DOCUMENT = OUT / "事前止盈限价_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_STANDING_PROFIT_LIMIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 109 and not OUT.exists(), "110前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标点估计不能直接关闭为未达标")
    data = pd.read_parquet(ROOT / cfg["features"]).set_index("date")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    checks, cycles, differences, fills, pairs = [], [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost_id, cost in cfg["costs"].items():
            folder = RESEARCH / period / cost_id
            old = pd.read_parquet(folder / "PANIC_ONLY_ledger.parquet")
            old_cycles = pd.DataFrame([{"entry_date": str(pd.Timestamp(c["entry_date"]).date()), "exit_date": str(pd.Timestamp(c["exit_date"]).date()),
                                        "net_profit_cny": c["net_profit"]} for c in saved_cycles(old, dividends, cfg)])
            scenario_ledgers = []
            for model, assumption in ASSUMPTIONS.items():
                ledger = pd.read_parquet(folder / f"{model}_ledger.parquet")
                native = pd.read_csv(folder / f"{model}_cycles.csv")
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(old.date)), "限价账户完整日历改变")
                limit_rows = ledger[ledger.standing_limit_price.notna()]
                for row in limit_rows.itertuples():
                    c = native[pd.to_datetime(native.entry_date).le(row.date) & pd.to_datetime(native.exit_date).ge(row.date)].iloc[0]
                    require(pd.Timestamp(c.entry_date) < row.date and row.sellable_before == row.shares_before, "止盈限价使用当日新买份额")
                    require(abs(row.entry_cost_before-c.entry_cost_cny) < 1e-9, "限价没有使用自身买入成本")
                    recognized = ledger.loc[ledger.date.between(pd.Timestamp(c.entry_date), row.date), "dividend_recognized"].sum()
                    require(abs(recognized-row.known_cycle_dividends_before) < 1e-8, "止盈使用未知或不同周期分红")
                    raw = (row.entry_cost_before*1.06-recognized)/row.shares_before
                    expected = math.ceil(raw/cfg["tick"]-1e-10)*cfg["tick"]
                    require(abs(expected-row.standing_limit_price) < 1e-10 and row.standing_limit_quantity == -row.shares_before, "事前限价或份额不符")
                    threshold = row.standing_reference_threshold
                    after_slip = math.floor(threshold*(1-cost["slippage"])/cfg["tick"]+1e-10)*cfg["tick"]
                    require(after_slip >= expected-1e-10, "原滑点将卖价推到限价以下")
                    if row.execution_clock == "INTRADAY_CONDITIONAL_PROFIT_LIMIT":
                        observed = data.loc[row.date]
                        require(row.open < threshold and observed.high >= threshold+cfg["tick"]-1e-10, "日内条件没有事前穿价支持")
                        require(np.isfinite(observed.volume) and observed.volume > 0 and row.shares_before <= .01*observed.volume, "日内情景容量不符")
                        if assumption == "CLOSE_STILL_ABOVE":
                            require(observed.close >= threshold+cfg["tick"]-1e-10, "保守情景没有收盘支持")
                        require(row.filled_quantity == -row.shares_before and row.fill_price >= expected-1e-10, "实际限价卖出金额或可卖份额不符")
                        fills.append({"period": period, "cost": cost_id, "model": model, "date": row.date, "entry_date": c.entry_date,
                                      "limit_price": expected, "reference_price": threshold, "fill_price": row.fill_price, "shares": row.shares_before})
                previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
                measured = summarize(ledger, cfg)
                saved = metric(result, model, period, cost_id)
                for key in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "trade_count"]:
                    require(abs(measured[key]-saved[key]) < 1e-9, "保存止盈绩效不能复算")
                crows = saved_cycles(ledger, dividends, cfg)
                cycles.extend({"period": period, "cost": cost_id, "model": model, **c} for c in crows)
                checks.append({"period": period, "cost": cost_id, "model": model, "limit_days": len(limit_rows), "cycles": len(crows), "net_sharpe": measured["net_sharpe"]})
                differences.append({"period": period, "cost": cost_id, "model": model, "terminal_nav_difference": float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                                    **{f"{c}_difference": float(ledger[c].sum()-old[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
                paired = native.merge(old_cycles, on="entry_date", suffixes=("_new", "_old"), validate="one_to_one")
                require(len(paired) == len(native) == len(old_cycles), "本轮出现额外进入，需另解释")
                pairs.extend({"period": period, "cost": cost_id, "model": model, "entry_date": r.entry_date, "new_exit": r.exit_date_new, "old_exit": r.exit_date_old,
                              "new_profit": r.net_profit_cny_new, "old_profit": r.net_profit_cny_old, "profit_difference": r.net_profit_cny_new-r.net_profit_cny_old} for r in paired.itertuples())
                if period == "earlier_diagnostic":
                    for key in ["shares", "cash", "equity", "filled_quantity", "commission", "slippage_cost"]:
                        np.testing.assert_allclose(ledger[key], old[key], atol=1e-10, rtol=0)
                scenario_ledgers.append(ledger)
            for key in ["shares", "cash", "equity", "filled_quantity", "commission", "slippage_cost"]:
                np.testing.assert_allclose(scenario_ledgers[0][key], scenario_ledgers[1][key], atol=0, rtol=0)
    require(len(fills) == 4 and all(str(pd.Timestamp(r["date"]).date()) == "2020-02-17" for r in fills), "本轮条件成交不止原单一日期")
    for name, rows in [("saved_account_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_profit_differences.csv", differences), ("saved_limit_fills.csv", fills), ("saved_cycle_pairs.csv", pairs)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_STANDING_LIMIT_AND_ACCOUNT_CHECKS_COMPLETE", "standing_limit_days": sum(r["limit_days"] for r in checks),
               "new_account_metrics_recomputed": len(checks), "complete_cycles": len(cycles), "conditional_fill_records": len(fills), "distinct_conditional_fill_dates": 1,
               "two_execution_scenarios_same_actual_paths": True, "earlier_accounts_identical_to_original": True, "new_accounts_or_models": 0,
               "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_SINGLE_PROFIT_EXIT_CHANGE_LOWER_RETURN_NOT_TARGET"
    decision = "第110轮未达标。两成交情景的实际路径相同：主基础／压力夏普1.124／1.106，较早负0.107／负0.139。主基础年化2.23%，低于原急跌策略2.38%和买入持有3.63%；较早年化负1.02%，与原策略完全相同。主历史只发生三次完整交易，新增止盈只改变2020年2月的一次退出，不能据此认定稳定改善。关闭本次固定止盈设置。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第110轮之后：稀少机会与更多状态样本", "", decision, "",
        "一个原6%止盈设置、两固定成交假设、两时期两费用，共8新账户；7必要测试4.18秒、核心运行2.283秒、零新模型或参考。主每账户3周期、27持仓收盘、25挂单日；早8周期、40持仓收盘、32挂单日。八账户228挂单日、44完整周期、四个条件成交记录都已核；两情景路径相同，较早与原账户一致。", "",
        "唯一条件成交日期2020-02-17：原2020-02-05进入，本来2月18日开盘退出，现提前一天。基础限价4.014、参考4.017、成交4.014；压力限价4.017、参考4.022、成交4.017。首周期基础利润从14067.61降至11956.03元；后两次进入退出日期一样，因本金更低，后两次利润也小幅下降。主终值基础少2302.87元、压力少2010.35元。没有依据改善，停止此限价家族，不调6%、1%容量或穿价/收盘条件。", "",
        "本轮查重确认第26轮已有缩量回调转升和放量大跌后缩量回升；旧第五轮固定底仓强制每日往返已被用户废止；旧30盘中亏损保护和27限价买入同样保留失败。不得把这些换名称或时间点作为新方向。", "",
        "下一候选方向是用每日可观测涨跌状态，直接估计从下一开盘到再下一开盘的条件收益；目的为减少对少数成熟交易周期的依赖。尚未证明这一方案区别于已有模型，不能登记成新研究。先核对research/direct_one_day_cash_advantage_rank_discovery_v0.py、旧research/dynamic_continuation_exit_v1.py和旧三状态HMM的定义与终止范围。当前有界检索已发现旧单日开盘收益模型和动态继续价值，不可直接称为全新。", "",
        "只有确认不同机制后才固定观察状态、月度日程、完整近期样本、下限、标签成熟时点及进出场成本逻辑。支持核对可只读日期和状态计数，不先拟合或计算新目标收益。不得把收盘到收盘的未来收益混作下一开盘买入后的收益。第111轮目前未选定、未登记、未训练，也没有运行会话。", "",
        "109条件方差预测及普通波动对照已关闭；108共享退出已关闭。EPS、公募、慢源仍暂停，不做GPT包或额外安全审计。研究目标和510300现金权限保持，goal继续active。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第110轮：事前止盈限价", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "新增机制只把2020年2月5日进入的周期，从2月18日开盘退出改为2月17日盘中条件退出。基础费用下该周期利润由14067.61元降至11956.03元；之后两次交易日期不变，但本金减少使后续利润略低。较早历史没有任何新增止盈成交。两种成交情景一致只说明本次历史路径重合，不证明真实限价排队和成交已验证。", "",
        "七项关键测试通过，八个新账户核心计算约2.28秒；旧库存、事前限价、原滑点、分红权益及44个完整周期保存值已核。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in ASSUMPTIONS:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 110, "study": result["study_id"], "title": "原急跌回升事前止盈限价与两成交情景", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND110_PROFIT_LIMIT_NO_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="110轮，387不同设置，403已评价来源版本，408登记含5旧未运行，1538主评价记录。",
                 next_work={"status": "OBSERVED_RETURN_STATE_METHOD_DUPLICATION_CHECK_PENDING", "focus": "核对旧单日开盘收益模型与每日观察状态方案的差异", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="110一个策略两成交情景、七必要测试、八新账户与保存核对交付完成；111未登记。")
    index["deliveries"].append({"created_at": now(), "type": "STANDING_PROFIT_LIMIT_ROUND110_FAST_CHINESE_RESULTS", "rounds": [110], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
