"""复核简易波动原始计算和四个实际账户，交付简短中文结果。"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.ease_of_movement_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal
from scripts.review_round74_saved import saved_cycles

OUT = ROOT / "deliverables/510300简易波动进出场_第112轮_20260908"
DOCUMENT = OUT / "简易波动进出场_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_EASE_OF_MOVEMENT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 111 and not OUT.exists(), "112前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "新策略有点值达到目标，不能直接归档失败")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "简易波动冻结输入改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "factors.parquet")
    require(pd.DatetimeIndex(f.date).equals(pd.DatetimeIndex(data.date)), "简易波动完整日历不符")
    require(f.input_valid.all(), "实际原始输入存在缺失，需保留并核对其覆盖")
    wealth = np.r_[1., np.cumprod(((data.close+data.dividend)/data.previous_close).iloc[1:].to_numpy())]
    np.testing.assert_allclose(wealth, data.wealth, atol=1e-12, rtol=1e-12)
    midpoint, raw = [], []
    for i, row in enumerate(data.itertuples()):
        high = wealth[i]*(row.high+row.dividend)/(row.close+row.dividend)
        low = wealth[i]*(row.low+row.dividend)/(row.close+row.dividend)
        midpoint.append((high+low)/2)
        raw.append((midpoint[i]-midpoint[i-1])*(high-low)*100000000./row.volume if i else np.nan)
        np.testing.assert_allclose([f.wealth_high.iloc[i], f.wealth_low.iloc[i], f.wealth_midpoint.iloc[i]], [high, low, midpoint[i]], atol=1e-12, rtol=0)
    expected = pd.Series(raw).rolling(14, min_periods=14).mean()
    np.testing.assert_allclose(f.movement_per_volume, raw, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_allclose(f.ease_of_movement, expected, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_array_equal(f.entry_cross, expected.gt(0)&expected.shift().le(0))
    np.testing.assert_array_equal(f.exit_nonpositive, expected.le(0))
    accounts, cycles = [], []
    requests, terminal_buys, terminal_sells = 0, 0, 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost_id, cost in cfg["costs"].items():
            folder = RESEARCH / period / cost_id
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            own = ledger.set_index("date")
            locked = False
            for row in decisions.itertuples():
                t = row.origin_index
                require(row.execution_date == data.date.iloc[t+1], "信号未按下一开盘执行")
                shares, cash = (int(own.loc[row.origin, "shares"]), float(own.loc[row.origin, "cash"])) if row.origin in own.index else (0, cfg["initial_capital"])
                if shares == 0:
                    locked = False
                require(row.ease_of_movement == f.ease_of_movement.iloc[t] and row.entry_cross == f.entry_cross.iloc[t] and row.exit_nonpositive == f.exit_nonpositive.iloc[t], "账户未使用当时保存因子")
                require(row.signal_state == "EASE_OF_MOVEMENT_AVAILABLE", "实际评价存在无指标日，需按具体状态处理")
                quantity, target = 0, 0.
                if shares:
                    locked = locked or expected.iloc[t] <= 0
                    quantity, target = (-shares, 0.) if locked else (0, 1.)
                elif f.entry_cross.iloc[t]:
                    price = math.ceil(data.close.iloc[t]*(1+cost["slippage"])/cfg["tick"]-1e-10)*cfg["tick"]
                    quantity = math.floor((cash+1e-9)/price/cfg["lot"])*cfg["lot"]
                    while quantity and quantity*price+max(quantity*price*cost["commission"], cost["minimum"]) > cash+1e-8:
                        quantity -= cfg["lot"]
                    target = 1. if quantity else 0.
                require(row.requested_quantity == quantity and row.reference_weight == target and row.pending_exit_locked == locked, "零线上穿、实际份额或退出锁定不符")
                terminal = row.execution_date == ledger.date.iloc[-1]
                require(own.loc[row.execution_date, "requested_quantity"] == (-shares if terminal else quantity), "下一开盘或终点优先结算不符")
                terminal_buys += int(terminal and quantity > 0)
                terminal_sells += int(terminal and shares > 0 and quantity == 0)
                requests += 1
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            measured = summarize(ledger, cfg)
            stored = metric(result, PRIMARY, period, cost_id)
            for k in ["net_sharpe", "annualized_return", "cumulative_return", "max_drawdown", "mean_exposure", "commission", "slippage_cost", "trade_count"]:
                require(value_equal(measured[k], stored[k]), "简易波动保存指标不符")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost_id, **c} for c in complete)
            accounts.append({"period": period, "cost": cost_id, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"],
                             "gross_price_and_dividend_profit": float(ledger.price_pnl.sum()+ledger.dividend_recognized.sum()), "explicit_cost": float(ledger.commission.sum()+ledger.slippage_cost.sum()), "net_profit": float(ledger.equity.iloc[-1]-cfg["initial_capital"])})
    pd.DataFrame(cycles).to_csv(RESEARCH / "saved_actual_cycles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(accounts).to_csv(RESEARCH / "saved_account_checks.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_MOVEMENT_FACTORS_AND_REAL_ACCOUNTS_CHECKED", "raw_factor_rows": len(f), "actual_decision_requests": requests,
               "terminal_priority_suppressed_entries": terminal_buys, "terminal_priority_added_exits": terminal_sells, "new_account_metrics_recomputed": len(accounts), "complete_cycles": len(cycles),
               "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_EASE_OF_MOVEMENT_LOW_RETURN_AND_LARGE_DRAWDOWN_NOT_TARGET"
    decision = "第112轮未达标。主历史基础／压力净夏普0.094／-0.022，较早0.192／0.062。基础主年化0.38%、最大回撤41.17%；较早年化1.74%、最大回撤38.97%。两时期、两档成本的年化均低于买入持有，不保留本次十四日简易波动零线策略。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第112轮之后：继续压缩重复研究", "", decision, "",
        "本轮6项测试3.33秒，一个设置、4个真实账户用时2.45秒，零训练、零新参考。3456行因子、5646个决策、272个完整周期已核。主每档73周期146成交，较早63周期126成交；评价期零缺失、零未成交。主终点各抑制1个进入意图，较早终点各额外完成1次强制结算退出，全部符合事前终点规则。", "",
        "本次主基础费用前价格及分红利润约2.81万元，费用及滑点约2.30万元，剩余净利润约0.51万元；不能把低成本下的毛利润当成可执行优势。本轮不调平滑周期、零线、价量权重、止损或过滤条件挽救。", "",
        "111也已完成关闭：两设置8账户，141次联合估计9.74秒；其主基础毛贡献已经亏损，不能把失败统一归为成本。两轮共13项测试、12个新账户，均未达到完整目标。", "",
        "下一步先核对已存在的分钟方法和来源边界。本次查重已经发现research/intraday_entry_indicator.py和intraday_technical_factors.py完整实现前30分钟ORB、VWAP斜率、EMA8/21、ADX14及分时相对成交量；震荡分支还包括VWAP偏离和RSI7。不得把开盘区间突破或这些组合当作新方案。第43轮下午退出、第75轮下午进入的失败也保持关闭，不能挪几分钟重新包装。", "",
        "旧MINUTE_DATA.md描述短公开接口，INTRADAY_ENTRY_INDICATOR.md已经指向后来补入的五年1分钟代理来源；有两个时期说明，不能拿旧文档的123日结论替代当前文件范围。下一步只读当前分钟文件元数据和旧方法索引，核实实际覆盖、时间标注及完整账户研究边界；不重新做大规模分钟审计或下载。如缺乏较早分钟支持，明确保留不足，不把原日线回退算分钟验证。", "",
        "尚未选定或登记第113轮，不声称分钟新模型已经运行。只有发现和旧方法结构不同、现成数据足以支持的新方案才继续冻结与真实账户评价；否则立即换方向。EPS、公募及慢源继续不优先，不准备GPT包，不扩大证券或实盘权限，目标保持未完成。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第112轮：简易波动进出场", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "四个新账户约2.45秒完成，6项必要测试通过。主每档73个完整周期、146次成交，较早63个周期、126次成交。没有缺失预测或未成交请求。主基础费用前利润约2.81万元，扣除约2.30万元费用及滑点后剩约0.51万元，波动和回撤仍然较大。", "",
        "主期各74个进入意图中，最后一个被预定终点结算抑制；较早各62个常规退出意图另加一次终点结算，构成63次实际卖出。没有删除零持仓日，也没有把期末准备信号当真实成交。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    record = {"round": 112, "study": result["study_id"], "title": "中点移动乘区间除实际成交量的零线进出场", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND112_MOVEMENT_NOT_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="112轮，390不同设置，406已评价来源版本，411登记含5旧未运行，1556主评价记录。",
                 next_work={"status": "MINUTE_METHOD_DEDUP_AND_CURRENT_SOURCE_BOUNDARY_PENDING", "focus": "复用现成分钟范围，跳过旧ORB与下午时钟微调", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="111至112共13项测试、141次联合估计、12个新账户及394周期核对交付完成；113仅方向核对，尚未选定或登记。")
    index["deliveries"].append({"created_at": now(), "type": "EASE_OF_MOVEMENT_ROUND112_FAST_CHINESE_RESULTS", "rounds": [112], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "账户": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
