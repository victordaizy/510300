"""核对曲折度、因果市场状态与四账户，完成简洁中文交付。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.choppiness_state_entry_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.choppiness_state_entry_inputs_v1 import factor_frame
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

OUT = ROOT / "deliverables/510300价格曲折度状态切换_第105轮_20260908"
DOCUMENT = OUT / "价格曲折度状态切换_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_CHOPPINESS_STATE_ENTRY_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 104 and not OUT.exists(), "105前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接关闭")
    data = pd.read_parquet(ROOT / cfg["features"])
    factors = pd.read_parquet(RESEARCH / "choppiness_factors.parquet")
    high = data.wealth*(data.high+data.dividend)/(data.close+data.dividend)
    low = data.wealth*(data.low+data.dividend)/(data.close+data.dividend)
    tr = np.maximum.reduce([(high-low).to_numpy(), (high-data.wealth.shift()).abs().to_numpy(), (low-data.wealth.shift()).abs().to_numpy()])
    path = pd.Series(tr).rolling(14).sum()
    span = high.rolling(14).max()-low.rolling(14).min()
    score = 100*np.log10(path/span.where(span > 0))/np.log10(14)
    np.testing.assert_allclose(factors.choppiness, score, atol=1e-12, rtol=0, equal_nan=True)
    state = None
    for row in factors.itertuples():
        if pd.isna(row.choppiness):
            state = None
        elif row.choppiness < 38.2:
            state = 1
        elif row.choppiness > 61.8:
            state = 2
        require((state is None and pd.isna(row.market_state)) or state == row.market_state, "曲折度市场状态不符合因果记忆")
        t = row.Index
        allowed = bool(data.feature_valid.iloc[t])
        trend = state == 1 and data.sma120.iloc[t] > 0 and data.wealth.iloc[t] > row.previous_high20
        rebound = state == 2 and data.z20.iloc[t] < -1.5 and t > 0 and data.wealth.iloc[t] > data.wealth.iloc[t-1]
        expected = 1 if allowed and trend else 2 if allowed and rebound else 0
        require(row.raw_entry == expected, "曲折度两模式进入条件不符")
        require(row.raw_exit_trend == (state == 2 or data.sma120.iloc[t] <= 0 or data.wealth.iloc[t] < row.previous_low10), "趋势退出条件不符")
        require(row.raw_exit_range == (state == 1 or data.z20.iloc[t] >= 0), "震荡退出条件不符")
    early = data[data.date.le(cfg["earlier_terminal"])].copy()
    pd.testing.assert_frame_equal(factors.iloc[:len(early)], factor_frame(early))
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, accounts = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            for row in native.itertuples():
                origin = int(row.entry_index)-1
                require(factors.raw_entry.iloc[origin] == row.mode and data.date.iloc[origin] == pd.Timestamp(row.entry_origin), "实际进入未按前一收盘模式")
                require(row.holding_intervals >= 1, "实际退出违反次日可卖")
            m, expected = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(m[field], expected[field]), "保存完整账户指标错误")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "trend_cycles": int(native["mode"].eq(1).sum()), "range_cycles": int(native["mode"].eq(2).sum()),
                             "unknown_factor_origins": int(decisions.factor_status.ne("CHOPPINESS_STATE_AVAILABLE").sum())})
    pd.DataFrame(cycles).to_csv(RESEARCH / "saved_actual_cycles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(accounts).to_csv(RESEARCH / "saved_account_checks.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_CHOPPINESS_STATES_ENTRY_EXIT_AND_ACCOUNTS_VERIFIED", "factor_rows_checked": len(factors),
               "earlier_prefix_rows_checked": len(early), "new_account_metrics_checked": 4, "complete_cycles": len(cycles),
               "new_accounts_or_models": 0, "source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第105轮未达标。主基础／压力夏普为负0.187／负0.254，较早为0.610／0.549；主基础年化负2.55%、最大回撤33.15%，较早年化6.55%、最大回撤11.16%。曲折度切换没有形成稳定高夏普，关闭此固定设置，不调整周期、状态界限或原进出场阈值救回。"
    status = "COMPLETED_CHOPPINESS_STATE_MAIN_NEGATIVE_NOT_TARGET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第105轮之后：同一真实持仓状态的非线性继续价值", "", decision, "",
        "105五项必要测试4.77秒，四个新账户实际运行约1.42秒，零拟合和参考账户。主35周期70成交421持股收盘，早24周期48成交342收盘；无缺失观点原点或未成交。3456行曲折度与因果状态、1852行早期前缀、四账户118周期已核，不重复运行。", "",
        "104成熟预测误差混合也已关闭：66校准中51内点、7旧端点、8新端点，主252收盘有156新权重正，但实际混合预测零变号，四账本完全等于32。主夏普0.705/0.641，早0.748/0.725。禁止继续调预测混合目标、窗口或确认天数。", "",
        "下一106拟检验原八项实际持仓状态能否用非线性提升树给出更有效的继续价值。限定检索已确认全局日收益分配/分类研究用过直方图提升树，不得称这一算法从未试过；research/adaptive_allocation_v1.py的make_model附近保存80轮、学习率0.05、7叶、每叶60样本、正则10、禁随机提前停止的原算法设置。先精读这处和原31的退出训练，核对本持仓自然退出目标是否已有同一提升实现；没有重复才登记一个同参数的条件继续价值任务，不能改旧全球模型阈值重跑。", "",
        "拟复用原31 D60_INTRA 1461状态34自然周期、原八项FEATURES和同141月度日程；最近20成熟完整周期、十周期一百状态下限、周期等权，继续价值目标和基础费用标签不改。零新参考；预期114个成熟拟合，早2017年3月以前仍无模型。它不凭空补足较早样本，不宣称可以解决全段早期问题。原D60进入、6%亏损8%追踪60日、连续两次负预测、再现条件与两日冷却不变。只改变非线性条件预测映射；输出完整模型文件及简洁中文因子和规则，避免重复展开每棵树长报告。", "",
        "下一步先核对本任务重复性和既有算法精确设置，再用原完整周期因果训练，必要测试后冻结并实际运行。若训练代价显著高于预期，先说明真实进度，不编造结果；不加新源或参数网格。目前106尚未登记、训练或生成账户。只执行510300及现金，EPS/公募慢源与GPT包暂停，完整目标仍未实现。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第105轮：价格曲折度状态切换", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "五项必要测试通过，四个完整账户约1.42秒；主35个实际周期，较早24个，两种费用均无未成交。因子、实际进入模式、完整日历和净值已核。本轮主历史亏损，停止对本规则继续调参。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".json", ".csv"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 105, "study": result["study_id"], "title": "价格路径曲折度的趋势震荡状态切换", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND105_CHOPPINESS_NO_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="105轮，381不同设置，397已评价来源版本，402登记含5旧未运行，1488主评价记录。",
                 next_work={"status": "NONLINEAR_CONDITIONAL_CONTINUATION_DIRECTION_NOT_REGISTERED", "focus": "同一成熟自然周期八项实际状态的非线性继续价值，先限定查重", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="104及105共12必要测试、66校准、8新账户及保存核对交付完成；无新预测拟合或参考账户。106待登记。")
    index["deliveries"].append({"created_at": now(), "type": "CHOPPINESS_STATE_ROUND105_FAST_CHINESE_RESULTS", "rounds": [105], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
