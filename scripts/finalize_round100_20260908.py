"""独立核对保存高低点日期及完整账户，快速关闭第100轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.aroon_time_entry_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300高低点时间_第100轮_20260908"
DOCUMENT = OUT / "高低点时间_结果及中文规则.md"
NEXT = ROOT / "docs/510300_AFTER_AROON_TIME_ENTRY_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 99 and not OUT.exists(), "第100轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时需另作验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "aroon_factors.parquet")
    require(pd.DatetimeIndex(f.date).equals(pd.DatetimeIndex(data.date)), "极值因子完整日历不同")
    checks = []
    for t in range(len(data)):
        if t < cfg["extrema_period"]:
            require(f.factor_status.iloc[t] == "NO_VIEW_INSUFFICIENT_EXTREMA_WINDOW" and pd.isna(f.aroon_up.iloc[t]) and pd.isna(f.aroon_down.iloc[t]), "不完整极值窗口产生了观点")
            continue
        window = data.iloc[t-cfg["extrema_period"]:t+1]
        raw = window[["high", "low", "close", "dividend"]].to_numpy(float)
        valid = np.isfinite(raw).all() and (raw[:, :3] > 0).all() and (raw[:, 3] >= 0).all() and (raw[:, 0] >= raw[:, 2]).all() and (raw[:, 2] >= raw[:, 1]).all()
        require(valid == (f.factor_status.iloc[t] == "AROON_TIME_FACTORS_AVAILABLE"), "极值窗口有效性不同")
        if not valid:
            continue
        # 按逐日现金再投资递推，不调用因子的向量化份额函数。
        units, highs, lows = 1., [], []
        for row in window.itertuples():
            cash = units*row.dividend
            highs.append(units*row.high+cash)
            lows.append(units*row.low+cash)
            units += cash/row.close
        ages = []
        for values, extreme in [(highs, max(highs)), (lows, min(lows))]:
            ages.append(next(age for age, value in enumerate(reversed(values)) if abs(value-extreme) <= cfg["numerical_tie_tolerance"]*max(1., abs(extreme))))
        up, down = [100.*(cfg["extrema_period"]-age)/cfg["extrema_period"] for age in ages]
        np.testing.assert_array_equal(ages, [f.high_age_trading_days.iloc[t], f.low_age_trading_days.iloc[t]])
        np.testing.assert_allclose([up, down], [f.aroon_up.iloc[t], f.aroon_down.iloc[t]], atol=0, rtol=0)
        require(int(up >= cfg["entry_up_threshold"] and down <= cfg["entry_down_threshold"]) == f.raw_entry.iloc[t] and bool(down > up) == f.raw_exit.iloc[t], "保存高低点进出条件不同")
        checks.append({"date": data.date.iloc[t], "independent_high_age": ages[0], "independent_low_age": ages[1], "independent_up": up, "independent_down": down})
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
            np.testing.assert_allclose(decisions.aroon_up, f.aroon_up.iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(decisions.aroon_down, f.aroon_down.iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]-1, ledger.net_return, atol=1e-12, rtol=0)
            require(decisions.loc[decisions.requested_quantity.gt(0), "raw_entry"].eq(1).all(), "无进入因子却请求买入")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in complete)
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            require(all(value_equal(actual[k], saved[k]) for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]), "保存账户指标不同")
            accounts.append({"period": period, "cost": cost, "cycles": len(complete), "losing_cycles": sum(c["net_profit"] < 0 for c in complete), **actual})
    for name, rows in [("saved_independent_extrema_checks.csv", checks), ("saved_account_checks.csv", accounts), ("saved_actual_cycles.csv", cycles)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_DIVIDEND_EXTREMA_DATES_AND_ACTUAL_ACCOUNTS_CHECKED", "independent_extrema_windows": len(checks), "new_accounts_checked": len(accounts), "complete_actual_cycles": len(cycles), "new_models_or_account_replays_during_check": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_AROON_TIME_SIGNAL_NO_LOCAL_IMPROVEMENT_NO_SHARPE_TARGET"
    decision = "第100轮未达标。主基础／压力夏普0.064／0.018，较早0.454／0.410；主基础年化只有0.107%、回撤22.98%，主压力年化为负0.389%。高低点出现时间信号没有改善主结果，结束这一固定进入退出设置，不改25周期、70与30门槛或翻转方向救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第100轮之后：先检查局部候选的实际退出状态", "", decision, "", "100一设置四新账户、五必要测试3.30秒、账户约5.85秒，无新预测模型或参考账户。3431完整极值窗口、25预热原点；主24周期48成交、较早21周期42成交，各费用路径完整。所有完整日期、分红、费用保留，不继续邻近指标筛选。", "", "下一步优先用第91轮保存实际账本与连续学习参考，检查新资金的实际进入日期、含分红成本和周期高点与参考状态的差异，以及这些差异是否改变已有八因子模型的退出预测。该工作只评价保存状态与保存模型，不生成反事实夏普、重训或重跑旧账户。若合并账户有部分增减仓，必须先明确可比较的实际单位状态，不能把不同数量下的总市值变动当成盈亏或回撤。只有可识别且有实际判断差异时再登记单一局部改动，否则立即放弃。", "", "已有第81轮分批进入修正使用实际份额和实际退出状态；新检查不能把该修正重复称为新机制。第91轮组合本来按连续参考意向分预算，参考与实际不同不自动属于错误。", "", "本次有界查重还发现研究已包括固定共享专家跟踪（contextual_expert_tracking_v1.py）、旧负向Page-Hinkley事件衰退与第60轮持仓CUSUM，不重跑这些旧算法或换名称。仍暂停EPS、公募和慢来源，不制作GPT包或长模型文档。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第100轮：高低点时间进入与明确退出", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 结果说明", "", "主基础平均股票占比38.87%，625个持股收盘，24个完整周期；较早44.14%，539个持股收盘，21个周期。主三个阶段基础夏普分别约0.215、负0.069、负0.033，收益没有跨阶段延续。压力情景算术日均收益仍略正、复利年化略负，两者口径不同，可以同时出现。", "", "五项必要测试、3431个分红调整极值日期及四账户90个完整周期已经核对。没有重新训练或重跑旧账户。阶段结果只解释本次固定设置，不据此挑选起止日。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 100, "study": result["study_id"], "title": "高低点时间进入与明确自然退出", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 8
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND100_AROON_TIME_SIGNAL_NO_SHARPE_TARGET_FULL_GOAL_NOT_MET", count_warning="100轮，376不同设置，392已评价来源版本，397登记含5旧未运行，1444主评价记录。", next_work={"status": "SAVED_REFERENCE_VS_OWN_EXIT_STATE_DIAGNOSTIC_NOT_NEW_STRATEGY", "focus": "只查第91轮实际周期与参考周期状态是否造成已有退出模型判断差异", "source": str(NEXT.relative_to(ROOT))}, process_state_note="100四新账户及五必要测试完成；下一仅保存状态诊断，尚无101新设置或收益。")
    index["deliveries"].append({"created_at": now(), "type": "AROON_TIME_ENTRY_ROUND100_FAST_CHINESE_RESULTS", "rounds": [100], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "账户周期": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
