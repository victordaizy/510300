"""核对保存的成熟均值、费用进出场和完整账户，简洁交付第111轮。"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.observed_return_state_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, MODES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal
from scripts.review_round74_saved import saved_cycles

OUT = ROOT / "deliverables/510300涨跌状态进出场_第111轮_20260908"
DOCUMENT = OUT / "涨跌状态进出场_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_OBSERVED_RETURN_STATE_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 110 and not OUT.exists(), "111前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "有新方案达到点目标，不能直接归档未达标")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "观察状态冻结文件改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    np.testing.assert_allclose(((data.close+data.dividend)/data.previous_close-1).iloc[1:], data.total_simple.iloc[1:], atol=1e-12, rtol=0)
    labels = pd.read_parquet(RESEARCH / "open_interval_labels.parquet")
    require(labels.origin_index.to_list() == list(range(1, len(data)-2)), "单日目标完整行历不符")
    dates = pd.DatetimeIndex(data.date)
    for row in labels.itertuples():
        i = row.origin_index
        require(row.origin == dates[i] and row.entry_index == i+1 and row.exit_index == i+2, "开盘区间标签时钟不符")
        earned = [e for e in dividends.itertuples() if dates[i+1] <= e.record_date < dates[i+2]]
        mature = max([dates[i+2], *[e.ex_date for e in earned]])
        require(row.maturity_date == mature and row.maturity_index == dates.searchsorted(mature), "分红成熟时间不符")
        np.testing.assert_allclose([row.observed_state, row.price_return, row.new_dividend_yield],
            [int(data.total_simple.iloc[i] >= 0), data.open.iloc[i+2]/data.open.iloc[i+1]-1, sum(e.cash_dividend_per_share for e in earned)/data.open.iloc[i+1]], atol=1e-12, rtol=0)
    models = json.loads((RESEARCH / "saved_state_estimates.json").read_text(encoding="utf-8"))["monthly_estimates"]
    schedule = json.loads((ROOT / cfg["model_schedule"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in schedule], "原月度日程改变")
    for model in models:
        t = model["fit_index"]
        expected = labels.set_index("origin_index").loc[t-243:t-2]
        require(len(expected) == 242 and model["first_origin_index"] == t-243 and model["last_origin_index"] == t-2, "成熟窗口不符")
        require(expected.maturity_index.le(t).all() and model["latest_maturity_index"] == expected.maturity_index.max() and model["status"] == "ESTIMATION_COMPLETE", "实际完整模型或成熟边界不符")
        for state in ["0", "1", "pooled"]:
            group = expected if state == "pooled" else expected[expected.observed_state.eq(int(state))]
            if state != "pooled":
                require(len(group) == model["counts"][state] and len(group) >= 60, "状态样本数不符")
            np.testing.assert_allclose([model["estimates"][state][k] for k in ["price_return", "new_dividend_yield"]], [group.price_return.mean(), group.new_dividend_yield.mean()], atol=1e-12, rtol=0)
    cycles, accounts, differences = [], [], []
    requests_checked, suppressed_terminal_entries = 0, 0
    fit_indexes = np.array([m["fit_index"] for m in models])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost_id, cost in cfg["costs"].items():
            def fee(q, price):
                return max(q*price*cost["commission"], cost["minimum"]) if q else 0.
            def price(raw, side):
                if cost["slippage"] == 0:
                    return raw
                ticks = raw*(1+side*cost["slippage"])/cfg["tick"]
                return (math.ceil(ticks-1e-10) if side > 0 else math.floor(ticks+1e-10))*cfg["tick"]
            pair = {}
            for model_id in MODES:
                folder = RESEARCH / period / cost_id
                ledger = pd.read_parquet(folder / f"{model_id}_ledger.parquet")
                decisions = pd.read_parquet(folder / f"{model_id}_decisions.parquet")
                own = ledger.set_index("date")
                locked = False
                for row in decisions.itertuples():
                    t = row.origin_index
                    require(row.execution_date == dates[t+1], "观察状态执行没有延至下一开盘")
                    shares, cash = (int(own.loc[row.origin, "shares"]), float(own.loc[row.origin, "cash"])) if row.origin in own.index else (0, cfg["initial_capital"])
                    if shares == 0:
                        locked = False
                    record = models[np.searchsorted(fit_indexes, t, side="right")-1]
                    state = int(data.total_simple.iloc[t] >= 0)
                    key = str(state) if model_id == PRIMARY else "pooled"
                    prediction = record["estimates"][key]
                    require(row.signal_state == "STATE_EXPECTATION_AVAILABLE" and row.observed_state == state and row.estimate_fit_index == record["fit_index"], "实际预测时钟或状态不符")
                    np.testing.assert_allclose([row.predicted_price_return, row.predicted_dividend_yield], [prediction["price_return"], prediction["new_dividend_yield"]], atol=1e-12, rtol=0)
                    close = data.close.iloc[t]
                    later = price(close*(1+prediction["price_return"]), -1)
                    div = close*prediction["new_dividend_yield"]
                    if shares:
                        immediate = price(close, -1)
                        gain = shares*(later+div)-fee(shares, later)-(shares*immediate-fee(shares, immediate))
                        require(abs(gain-row.estimated_continue_gain_cny) < 1e-7 and pd.isna(row.estimated_entry_gain_cny), "持有继续价值不符")
                        locked = locked or gain <= 0
                        q, target = (-shares, 0.) if locked else (0, 1.)
                        require(row.economic_quantity == shares, "持仓经济比较没有用实际份额")
                    else:
                        buy = price(close, 1)
                        affordable = math.floor((cash+1e-9)/buy/cfg["lot"])*cfg["lot"]
                        while affordable and affordable*buy+fee(affordable, buy) > cash+1e-8:
                            affordable -= cfg["lot"]
                        require(row.economic_quantity == affordable, "空仓经济比较没有用实际现金")
                        gain = affordable*(later+div)-fee(affordable, later)-(affordable*buy+fee(affordable, buy)) if affordable else np.nan
                        np.testing.assert_allclose(gain, row.estimated_entry_gain_cny, atol=1e-7, rtol=0, equal_nan=True)
                        q, target = (affordable, 1.) if affordable and gain > 0 else (0, 0.)
                    require(row.requested_quantity == q and row.reference_weight == target and row.pending_exit_locked == locked, "实际进出场请求或退出锁定不符")
                    require(own.loc[row.execution_date, "requested_quantity"] == (0 if row.execution_date == ledger.date.iloc[-1] and shares == 0 else -shares if row.execution_date == ledger.date.iloc[-1] else q), "实际开盘请求与期末结算不符")
                    suppressed_terminal_entries += int(row.execution_date == ledger.date.iloc[-1] and q > 0)
                    requests_checked += 1
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
                previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
                np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
                measured = summarize(ledger, cfg)
                stored = metric(result, model_id, period, cost_id)
                for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost", "trade_count"]:
                    require(value_equal(measured[k], stored[k]), "观察状态保存绩效或未定义夏普不符")
                complete = saved_cycles(ledger, dividends, cfg)
                cycles.extend({"period": period, "cost": cost_id, "model": model_id, **c} for c in complete)
                accounts.append({"period": period, "cost": cost_id, "model": model_id, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"],
                                 "gross_price_and_dividend_profit": float(ledger.price_pnl.sum()+ledger.dividend_recognized.sum()), "explicit_cost": float(ledger.commission.sum()+ledger.slippage_cost.sum()), "net_profit": float(ledger.equity.iloc[-1]-cfg["initial_capital"])})
                pair[model_id] = ledger
            a, b = pair[MODES[0]], pair[MODES[1]]
            differences.append({"period": period, "cost": cost_id, "terminal_nav_difference": float(a.equity.iloc[-1]-b.equity.iloc[-1]), **{f"{k}_difference": float(a[k].sum()-b[k].sum()) for k in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts), ("saved_state_vs_pooled_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_MATURE_STATE_ESTIMATES_AND_COST_DECISIONS_CHECKED", "open_interval_labels": len(labels), "monthly_estimation_batches": len(models),
               "actual_decision_requests": requests_checked, "terminal_priority_suppressed_entry_requests": suppressed_terminal_entries, "new_account_metrics_recomputed": len(accounts), "complete_cycles": len(cycles), "new_accounts_or_models": 0,
               "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_OBSERVED_STATE_AND_POOLED_RETURN_RULES_NOT_TARGET"
    decision = "第111轮未达标。涨跌状态方案主历史基础／压力净夏普为-0.222／-0.139，较早为0.047／0.084；主基础年化-2.80%、最大回撤27.03%，较早年化-1.17%、最大回撤43.47%。不分状态对照主基础净夏普-0.202，主压力全程现金、零波动，夏普无法计算；较早基础／压力为-0.086／-0.413。关闭本次两种均值进出场，不更改窗口或方向挽救。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    next_text = ["# 第111轮之后：直接检验已有价量的另一种关系", "", decision, "",
        "141次月度联合估计、8个新账户约9.74秒；7项必要测试通过。全部月度窗口可用，评价期没有预测缺失或未成交。完整账户包含零持仓天数，压力对照主历史的夏普未定义。主基础涨跌方案51个周期、102次成交，费用和滑点约1.50万元；但费用前价格及分红合计也亏损，不能只归咎交易成本。", "",
        "主基础有52个进入意图但仅51次买入，差额为终点日的准备信号被既定终点开盘结算优先级抑制，不是丢失成交。不同成本会改变经济进入门槛及路径，基础与压力不能视为同一仓位上简单扣费。", "",
        "下一候选方向是简易波动指标：价格高低点中间位置的日移动，乘当天高低价区间，再除以实际成交份额，观察14日平均值由非正转正及转回非正。它同时使用中间位置变化、日内区间和成交量，区别于旧收盘位置资金流、成交额涨跌占比、加权均线和绝对收益除成交额。限定research和docs查找EOM、EMV、Ease of Movement、简易波动等，未命中旧实现；这只证明本次有界查重未发现同项。", "",
        "已读取TradingView及thinkorswim官方指标说明，前者给出14日均值定义，后者给出中点变化乘区间除量和零线方向用法。来源不证明本项目可以获利；除息连续尺度、缺失状态、退出锁定及真实下一开盘仍需本项目明确并测试。暂未登记、计算新因子或运行第112轮。", "",
        "https://www.tradingview.com/support/solutions/43000502256-ease-of-movement-eom/", "https://toslc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/E-F/EaseOfMovement", "",
        "111、110、109、108及更早失败保持关闭。继续优先现成行情，不补EPS及其他慢源，不准备GPT审阅包，不扩大交易证券或实盘权限；完整目标尚未实现。"]
    NEXT.write_text("\n".join(next_text)+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第111轮：涨跌状态进出场", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "141次联合估计和8个新账户用时9.74秒，7项必要测试通过。主基础方案51个完整周期、102次成交，主压力2个周期、4次成交；成本会改变进入选择。所有年份均保留现金日，主压力合并对照全现金的夏普保持未定义。各方案两期年化均未超过买入持有。", "",
        "账户核对包括分红权利成熟、连续242行、141个当时可得均值、11292个收盘决策、真实账户净值和122个完整周期。主基础52个进入意图中，终点日1个按预定结算规则不再买入，其余51个成交；不能把它算未成交订单。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    record = {"round": 111, "study": result["study_id"], "title": "每日涨跌观察状态的成熟开盘均值进出场", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "model_views_per_estimation": 2, "evaluated_candidate_source_runs": 2, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND111_OBSERVED_STATE_NOT_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="111轮，389不同设置，405已评价来源版本，410登记含5旧未运行，1548主评价记录。",
                 next_work={"status": "EASE_OF_MOVEMENT_DEFINITION_CHECKED_NOT_REGISTERED", "focus": "固定中点变化乘区间除量的零线进入退出", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="111七项必要测试、141次联合估计、八新账户与122周期保存核对完成；112仅定义与有界查重，尚未登记或回测。")
    index["deliveries"].append({"created_at": now(), "type": "OBSERVED_STATE_ROUND111_FAST_CHINESE_RESULTS", "rounds": [111], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "账户": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
