"""用另一种积分表示核对预算，并核对保存的实际账本与周期。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from research.universal_reference_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, P91, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.simple_signal_blend_v1 import decision_state
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300固定混合增长预算_第97轮_20260908"
DOCUMENT = OUT / "固定混合增长预算_结果及中文规则.md"
NEXT = ROOT / "docs/510300_AFTER_UNIVERSAL_REFERENCE_BUDGET_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 96 and not OUT.exists(), "第97轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时需另作验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "continuous_budget_factors.parquet")
    first_ref = int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    for model, prefix in [("PANIC_ONLY", "panic"), ("REARM_RIDGE", "learned")]:
        ledger = pd.read_parquet(P91 / "continuous_references/BASE" / f"{model}_ledger.parquet")
        decisions = pd.read_parquet(P91 / "continuous_references/BASE" / f"{model}_decisions.parquet")
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first_ref:])), "连续父账户日历不符")
        np.testing.assert_allclose(f[f"{prefix}_reference_return"].iloc[first_ref:], ledger.net_return, atol=0, rtol=0)
        np.testing.assert_allclose(f[f"{prefix}_state"], decision_state(data, decisions), atol=0, rtol=0, equal_nan=True)
    # 独立固定128节点积分仅检查保存预算，节点不参与策略或挑选参数。
    x, quadrature = leggauss(128)
    b, measure = (x+1)/2, quadrature/2
    log_growth = np.zeros(len(b))
    checks = []
    for t in range(first_ref-1, len(data)-1):
        if t >= first_ref:
            a, c = f.panic_reference_return.iloc[t], f.learned_reference_return.iloc[t]
            require(np.isfinite([a, c]).all() and min(a, c) > -1, "本来源连续增长缺失")
            log_growth += np.log1p(b*a+(1-b)*c)
        mass = measure*np.exp(log_growth-log_growth.max())
        expected = float(np.dot(b, mass)/mass.sum())
        error = abs(expected-f.panic_budget.iloc[t])
        require(error < 1e-11, "独立积分与保存精确预算不符")
        expected_target = expected*f.panic_state.iloc[t]+(1-expected)*f.learned_state.iloc[t]
        require(abs(expected_target-f.target.iloc[t]) < 1e-11, "预算与原持有状态没有一致合成")
        checks.append({"date": data.date.iloc[t], "panic_budget": f.panic_budget.iloc[t], "independent_integral": expected, "absolute_difference": error})
    require(pd.isna(f.target.iloc[-1]), "终点开盘后生成了新目标")
    accounts, cycles = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])), "实际账本与完整原点日历不符")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "不是下一开盘执行")
            np.testing.assert_allclose(decisions.reference_weight, f.target.iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]-1, ledger.net_return, atol=1e-12, rtol=0)
            own = ledger.set_index("date")
            for row in decisions.itertuples():
                shares = 0 if row.origin_index == first-1 else int(own.loc[row.origin, "shares"])
                equity = cfg["initial_capital"] if row.origin_index == first-1 else float(own.loc[row.origin, "equity"])
                price = float(frame.close.iloc[row.origin_index])
                actual = shares*price/equity
                target = row.reference_weight
                expected_qty = 0 if pd.isna(target) or (target > 0 and abs(target-actual) < cfg["weight_band"] and shares > 0) else -shares if target == 0 else int(np.floor(target*equity/price/cfg["lot"]))*cfg["lot"]-shares
                require(expected_qty == row.requested_quantity, "目标整手或十个百分点带宽未按自身账户计算")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in complete)
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            require(all(value_equal(actual[k], saved[k]) for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]), "保存账户指标不符")
            accounts.append({"period": period, "cost": cost, "cycles": len(complete), "losing_cycles": sum(c["net_profit"] < 0 for c in complete), **actual})
    for name, rows in [("saved_independent_integral_checks.csv", checks), ("saved_account_checks.csv", accounts), ("saved_actual_cycles.csv", cycles)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_CONTINUOUS_SOURCES_INTEGRALS_AND_ACTUAL_ACCOUNTS_CHECKED", "budget_origins_checked": len(checks), "maximum_independent_integral_error": max(r["absolute_difference"] for r in checks), "new_accounts_checked": len(accounts), "complete_actual_cycles": len(cycles), "new_models_or_account_replays_during_check": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_UNIVERSAL_MIXTURE_POSITIVE_RETURN_EXCESS_BUT_SHARPE_BELOW_TARGET"
    decision = "第97轮四个新账户完成。主基础／压力夏普0.887／0.821，较早0.625／0.594，均不足1.2。主基础年化4.209%、较早4.509%，分别高于买入持有0.583和0.613个百分点；这是两段点估计超额，未证明稳定超额。保留收益线索，关闭本轮固定通用混合规则，不改先验、连续起点、财富指数、窗口或父策略救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第97轮之后：直接比较收益与风险的预算", "", decision, "", "本轮没有新预测训练或参考账户。精确正系数积分汇总所有固定比例，四真实账户与16个保存对照均完成。六项必要测试3.22秒通过，账户运行6.24秒；模型、来源与旧策略不重跑。主急跌预算41.46%至43.37%，早43.00%至47.68%，主要预算仍给学习分支。主52成交262持股收盘；早33成交339持股收盘；两段两成本均无未成交或目标缺失。主年化超额BASE/STRESS为0.583456/0.268346个百分点，早0.613064/0.392296个百分点。", "", "原91只最小化完整日方差，收益均值只用于去中心化；97按累计财富增长汇总，没有对当前收益与方差同时求最优。这留下一个直接且无需预测训练的问题：同一对连续参考策略，按已知过去均值和协方差选择非负预算，能否比单看风险或累计财富更接近目标？", "", "下一98优先检查两参考策略的非负切点预算（最大过去净收益均值除以波动）。已有界搜索未找到同一切点配置，旧5是市场因子直接效用训练，旧89是从固定父组合中选择；进一步只读相关文件确认机制区别，不展开全部历史。不改变91窗口或原信号，仅对新均值与协方差目标预先登记有限一个方法、现金选择及缺失处理。先完成必要公式与时钟测试，再冻结与真实账户评价；尚未登记98设置或读取新账户收益。", "", "输入直接复用reports/research/510300_continuous_reference_min_variance_v1/evaluation_factors.parquet：date、panic_reference_return、learned_reference_return、panic_state、learned_state；3456日，参考首日2013-06-03，准备收盘2013-05-31，完整旧父账本在continuous_references/BASE。97独立核对已经确认字段等于实际父账本与原决策，不重复来源整理。参考终点开盘收益不进入新CLOSE预算，早期只读同一连续来源的当前前缀。", "", "加快动作：使用已有费用、事件账户和保存对照；只有局部改善或异常才加针对性诊断，不生成长报告或GPT包。目标保持active；EPS、公募及慢来源继续暂停。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第97轮：固定混合增长预算的结果", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 结果的含义", "", "新方法没有新增预测训练，四个新实际账户的计算约6秒。主历史平均股票占比9.80%，主基础最大回撤6.19%；较早平均股票占比15.33%，最大回撤11.27%。主52笔成交、较早33笔成交，均没有未成交或缺失目标。", "", "两个阶段的年化收益都超过同期买入持有，但新方法的主夏普低于第91轮，较早也远低于1.2。预算随全历史财富缓慢变化，主历史急跌份额约41.5%至43.4%，较早43.0%至47.7%；它没有提供足够强的风险收益改善。不能把两段正超额点估计称为已经稳定有效。", "", f"六项必要测试通过；独立积分检查覆盖{len(checks)}个预算原点，四个新账户共{len(cycles)}个完整交易周期已核对。完整每日预算、成交、分红费用和周期明细见同目录CSV。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 97, "study": result["study_id"], "title": "全部固定混合比例的累计增长汇总预算", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND97_RETURN_EXCESS_WITHOUT_TARGET_SHARPE_FULL_GOAL_NOT_MET", count_warning="97轮，373不同设置，389已评价来源版本，394登记含5旧未运行，1416主评价记录。", next_work={"status": "CONTINUOUS_REFERENCE_TANGENCY_METHOD_CHECK_NOT_REGISTERED", "focus": "同一连续参考的均值与风险共同决定非负预算", "source": str(NEXT.relative_to(ROOT))}, process_state_note="97四新账户及六必要测试完成，没有新增模型拟合或参考；98切点预算尚未登记。")
    index["deliveries"].append({"created_at": now(), "type": "UNIVERSAL_REFERENCE_BUDGET_ROUND97_FAST_CHINESE_RESULTS", "rounds": [97], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "账户周期": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
