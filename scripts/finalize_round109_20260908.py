"""核对条件方差递推、共同风险映射与八个保存账户后简洁交付。"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.conditional_variance_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, P91, PRIMARY, MODELS
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table
from scripts.review_round74_saved import saved_cycles

OUT = ROOT / "deliverables/510300预测波动仓位比较_第109轮_20260908"
DOCUMENT = OUT / "预测波动仓位比较_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_CONDITIONAL_VARIANCE_BUDGET_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 108 and not OUT.exists(), "109前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标不能直接归档未达标")
    data = pd.read_parquet(ROOT / cfg["features"])
    raw_returns = (data.close+data.dividend)/data.previous_close-1
    np.testing.assert_allclose(raw_returns.iloc[1:], data.total_simple.iloc[1:], atol=1e-12, rtol=0)
    rv20 = data.total_simple.rolling(20).std(ddof=1)*np.sqrt(cfg["annual_days"])
    np.testing.assert_allclose(rv20, data.vol20, atol=1e-12, rtol=0, equal_nan=True)
    models = json.loads((RESEARCH / "saved_variance_models.json").read_text(encoding="utf-8"))["models"]
    schedule = json.loads((ROOT / cfg["model_schedule"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in schedule], "条件方差改动原月度拟合日程")
    checks = []
    for record in models:
        require(record["status"] == "FIT_COMPLETE", "本轮实际存在未完成模型，需按具体状态核对")
        t = record["fit_index"]
        left = max(1, t-cfg["training_window"]+1)
        require(left == record["training_start_index"] and t-left+1 == record["training_rows"] and record["latest_observed_return_index"] == t, "方差训练窗口或已知时点不符")
        r = data.total_simple.iloc[left:t+1].to_numpy(float)*100.
        theta = np.asarray(record["theta"])
        require(-12 <= theta[0] <= 6 and np.all((theta[1:] >= -12) & (theta[1:] <= 12)), "求解参数越过冻结范围")
        initial = float(np.mean(r*r))
        scores = np.exp(np.r_[0., theta[1:]])
        weights = scores/scores.sum()
        omega, alpha, beta = initial*np.exp(theta[0]), .999*weights[1], .999*weights[2]
        m = record["model"]
        np.testing.assert_allclose([m["omega"], m["alpha"], m["beta"], m["initial_variance_percent_squared"]], [omega, alpha, beta, initial], atol=1e-12, rtol=1e-12)
        require(omega > 0 and 0 <= alpha+beta < .999, "方差参数违反正值与平稳限制")
        h = [initial]
        for shock in r[:-1]:
            h.append(omega+alpha*shock*shock+beta*h[-1])
        h = np.asarray(h)
        objective = .5*np.mean(np.log(h)+r*r/h)
        expected_next = omega+alpha*r[-1]**2+beta*h[-1]
        require(abs(objective-record["objective"]) < 1e-10 and objective <= record["initial_objective"]+1e-10, "标量准似然复算与保存目标不符")
        np.testing.assert_allclose([m["variance_at_fit_percent_squared"], m["next_variance_percent_squared"]], [h[-1], expected_next], atol=1e-10, rtol=1e-12)
        checks.append({"fit_origin": record["fit_origin"], "observations": len(r), "omega": omega, "alpha": alpha, "beta": beta,
                       "objective_error": abs(objective-record["objective"]), "next_variance_error": abs(expected_next-m["next_variance_percent_squared"]), "iterations": record["iterations"]})
    require(len(checks) == 141 and result["failed_fits"] == 0, "条件方差训练数量不符")
    forecasts = pd.read_parquet(RESEARCH / "variance_forecasts.parquet")
    by_index = {m["fit_index"]: m for m in models}
    expected = np.full(len(data), np.nan)
    m, v = None, np.nan
    for t in range(len(data)):
        if t in by_index:
            m = by_index[t]["model"]
            v = m["next_variance_percent_squared"]
        elif m is not None:
            shock = data.total_simple.iloc[t]*100.
            v = m["omega"]+m["alpha"]*shock*shock+m["beta"]*v if np.isfinite(shock) else np.nan
        expected[t] = v
    np.testing.assert_allclose(forecasts.next_variance_percent_squared, expected, atol=1e-10, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(forecasts.forecast_annual_volatility, np.sqrt(expected*242)/100, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_allclose(forecasts.realized_annual_volatility20, rv20, atol=1e-12, rtol=0, equal_nan=True)
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, accounts, differences = [], [], []
    requests_checked = 0
    for period in ["evaluation", "earlier_diagnostic"]:
        factors = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        parent = pd.read_parquet(P91 / f"{period}_factors.parquet")
        require(pd.DatetimeIndex(factors.date).equals(pd.DatetimeIndex(parent.date)), "原目标完整日历不同")
        np.testing.assert_allclose(factors.parent_target, parent.target, atol=0, rtol=0, equal_nan=True)
        for model, vol_name in [(MODELS[0], "forecast_annual_volatility"), (MODELS[1], "realized_annual_volatility20")]:
            multiplier = 1.
            expected_targets = []
            for row in factors.itertuples():
                vol = getattr(row, vol_name)
                if np.isfinite(vol) and vol > 0:
                    multiplier = min(1., .10/vol)
                expected_targets.append(row.parent_target*multiplier if np.isfinite(row.parent_target) else np.nan)
            np.testing.assert_allclose(factors[model], expected_targets, atol=1e-12, rtol=0, equal_nan=True)
            for cost in cfg["costs"]:
                folder = RESEARCH / period / cost
                ledger = pd.read_parquet(folder / f"{model}_ledger.parquet")
                decisions = pd.read_parquet(folder / f"{model}_decisions.parquet")
                parent_ledger = pd.read_parquet(folder / "CONTINUOUS_REFERENCE_MIN_VARIANCE_ledger.parquet")
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(parent_ledger.date)), "新旧风险账户日历不同")
                own = ledger.set_index("date")
                np.testing.assert_allclose(decisions.reference_weight, factors[model].iloc[decisions.origin_index], atol=1e-12, rtol=0, equal_nan=True)
                for row in decisions.itertuples():
                    require(row.execution_date == data.date.iloc[row.origin_index+1], "风险调整不是下一开盘")
                    if row.origin in own.index:
                        state = own.loc[row.origin]
                        equity, shares = state.equity, int(state.shares)
                    else:
                        require(row.origin == decisions.origin.iloc[0], "非准备日缺少实际账户状态")
                        equity, shares = cfg["initial_capital"], 0
                    price = data.close.iloc[row.origin_index]
                    target = row.reference_weight
                    if np.isfinite(target):
                        desired = math.floor(target*equity/price/cfg["lot"])*cfg["lot"]
                        if target > 0 and shares > 0 and abs(target-shares*price/equity) < cfg["weight_band"]:
                            desired = shares
                        require(row.requested_quantity == desired-shares, "风险预算没有使用本账户净值或带宽")
                    else:
                        require(row.requested_quantity == 0, "未知目标生成了新调整")
                    requests_checked += 1
                previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
                measured = summarize(ledger, cfg)
                saved = metric(result, model, period, cost)
                for key in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost", "trade_count"]:
                    require(abs(measured[key]-saved[key]) < 1e-9, "风险账户保存绩效不符")
                cycle_rows = saved_cycles(ledger, dividends, cfg)
                cycles.extend({"period": period, "cost": cost, "model": model, **c} for c in cycle_rows)
                accounts.append({"period": period, "cost": cost, "model": model, "days": len(ledger), "cycles": len(cycle_rows), "net_sharpe": measured["net_sharpe"]})
                differences.append({"period": period, "cost": cost, "model": model, "parent": "CONTINUOUS_REFERENCE_MIN_VARIANCE",
                                    "terminal_nav_difference": float(ledger.equity.iloc[-1]-parent_ledger.equity.iloc[-1]),
                                    **{f"{c}_difference": float(ledger[c].sum()-parent_ledger[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_variance_fit_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts), ("saved_profit_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_CONDITIONAL_VARIANCE_AND_SHARED_EXECUTION_CHECKS_COMPLETE", "scalar_variance_models": len(checks), "forecast_calendar_rows": len(forecasts),
               "actual_decision_requests": requests_checked, "new_account_metrics_recomputed": len(accounts), "complete_cycles": len(cycles), "new_accounts_or_models": 0,
               "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_CONDITIONAL_VARIANCE_INFERIOR_TO_SIMPLE_CONTROL_NOT_TARGET"
    decision = "第109轮未达标。预测波动主基础／压力夏普1.011／0.938，较早0.561／0.525；普通二十日波动对照为1.032／0.966和0.777／0.745，预测模型两段均更差。原91主夏普1.233／1.174，但较早0.431／0.395且缺乏稳定超额；本轮两种缩减都降低主夏普。预测波动主基础年化1.24%、最大回撤1.54%；普通对照年化1.47%、回撤1.79%。风险减少没有转化为完整目标，关闭本次固定缩减结构。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第109轮之后：减少无效改造", "", decision, "",
        "本轮六项必要测试3.58秒；141个条件方差模型、两设置共八个实际账户约2.59秒。全部模型成功，最多31迭代；零新参考、零未成交、评价期零波动模型缺失、零目标缺失。保存的标量方差递推、准似然、每日预测、同风险乘数、实际净值带宽订单和完整周期已核。", "",
        "原单纯二十日波动缩减第29轮及入场定预算第41轮保持旧结果，不将此对照算新预测发现。本轮新假设是动态条件方差估计，预测风险更复杂没有胜过同规则普通波动。两种风险缩减都使主基础年化从原91的3.36%降至约1.2%至1.5%，不得再调目标波动、估计窗口、非对称项、系数界限或带宽救回。", "",
        "108共享继续价值已关闭：主0.526/0.460，早0.618/0.594；138模型加四账户5.20秒。106提升、107隔夜下行、104误差混合和105曲折度同样不重开。", "",
        "去重补充：此前99记录中称HAR只有协议引用不完整；本次明确找到research/har_volatility_challenger.py的旧多尺度波动预测实现，不能当作未试新方法。GARCH本轮已实际完成，不再启动不同阶数。假跌破收复第26轮、压缩突破第54轮、回踩确认第55轮、旧背离/图结构及Aroon均已做过。", "",
        "下一步回到产生收益的进入机会。只检索现有成交量、人民币成交额与价格的跨日先后关系：是否存在不同于旧放量收强、加权均线、跌破收复和回踩突破的独立事件定义。若已做过立即跳过，不把加一个成交量条件包装为新策略。先完成有界查重与已有输入支持，再只冻结一个不同机制；尚未选定、登记或训练第110轮。", "",
        "已完成108及109共十二项测试、279次新拟合、十二个新评价账户。只做必要研究检查和简洁中文结果，不准备GPT包或额外安全审计；不补EPS、公募、慢源或扩大证券权限。完整目标未实现，继续研究。"])+"\n", encoding="utf-8")
    parent_rules = (ROOT / cfg["parent_rules"]).read_text(encoding="utf-8")
    underlying = parent_rules.split("## 两条策略如何产生原始信号\n", 1)[1].split("## 新增风险因子", 1)[0].strip()
    underlying = underlying.replace("全部逐月中文数值随结果提供", "逐月数值使用原第31轮保存文件")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第109轮：预测波动仓位比较", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "141次拟合与八账户共约2.59秒，六项必要测试通过。两种风险方法在相同原目标和交易约束下比较；预测模型没有胜过普通波动对照。主平均股票占比仅2.35%和2.73%，年化收益均低于买入持有；较早普通波动对照夏普上升，同时年化降至2.00%，也没有稳定超额。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:], "", "## 附：底层两条参考的全部中文因子与进出场", "", underlying,
        "", "上述底层规则只说明复用的参考路径；本轮没有重新训练原八因子模型。参考连续起点及最小方差预算以本文件原股票目标一节为准，不使用第76轮冷启动或倒数波动预算。"])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in MODELS:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 109, "study": result["study_id"], "title": "预测波动与普通波动的同规则仓位比较", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 2, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND109_VARIANCE_MODEL_NO_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="109轮，386不同设置，402已评价来源版本，407登记含5旧未运行，1528主评价记录。",
                 next_work={"status": "PRICE_VOLUME_EVENT_ORDER_NOVELTY_NOT_YET_VERIFIED", "focus": "已有价量跨日事件的有界查重，避免再调风险缩减", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="108及109共十二测试、279新拟合、十二新账户与保存核对交付完成；110尚未选定或登记。")
    index["deliveries"].append({"created_at": now(), "type": "CONDITIONAL_VARIANCE_ROUND109_FAST_CHINESE_RESULTS", "rounds": [109], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
