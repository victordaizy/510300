"""只核对自身净值、峰值递推、目标和完整保存账户，交付第94轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.own_cushion_risk_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, PARENTS
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300自身回撤控制_第94轮_20260908"
DOCUMENT = OUT / "自身回撤控制_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_OWN_CUSHION_RISK_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 93 and not OUT.exists(), "第94轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    checks, cycles, differences, annual_differences, concentration = [], [], [], [], []
    origin_count = 0
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for model, (parent_model, parent_folder, _) in PARENTS.items():
            parent_target = pd.read_parquet(parent_folder / f"{period}_factors.parquet").target.iloc[first-1:-1].to_numpy()
            for cost_id in cfg["costs"]:
                ledger = pd.read_parquet(RESEARCH / period / cost_id / f"{model}_ledger.parquet")
                decisions = pd.read_parquet(RESEARCH / period / cost_id / f"{model}_decisions.parquet")
                old = pd.read_parquet(RESEARCH / period / cost_id / f"{parent_model}_ledger.parquet")
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(old.date)), "完整交易日历不符")
                require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])) and pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "收盘决策与下一开盘时钟不符")
                previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
                high = np.maximum.accumulate(previous)
                floor = cfg["floor_fraction"]*high
                cushion = np.maximum(previous-floor, 0)
                scale = np.clip(cfg["cushion_multiplier"]*cushion/previous, 0, 1)
                expected = parent_target*scale
                for name, values in [("origin_equity", previous), ("high_water", high), ("floor_value", floor), ("cushion_value", cushion), ("risk_scale", scale), ("parent_target", parent_target), ("target", expected), ("reference_weight", expected)]:
                    np.testing.assert_allclose(decisions[name], values, atol=1e-10, rtol=0, equal_nan=True)
                np.testing.assert_allclose(ledger.decision_floor_value, floor, atol=1e-10, rtol=0)
                np.testing.assert_array_equal(ledger.below_previous_decision_floor, ledger.equity.to_numpy() < floor)
                shares = np.r_[0, ledger.shares.iloc[:-1]].astype(int)
                price = frame.close.iloc[first-1:-1].to_numpy()
                desired = (np.floor(expected*previous/price/cfg["lot"])*cfg["lot"]).astype(int)
                preserve = (expected > 0) & (shares > 0) & (abs(expected-shares*price/previous) < cfg["weight_band"])
                desired[preserve] = shares[preserve]
                np.testing.assert_array_equal(decisions.requested_quantity, desired-shares)
                require(ledger.shares.iloc[-1] == 0 and not ledger.terminal_unliquidated.iloc[-1], "新账户终点仍有未结算份额")
                origin_count += len(decisions)
                actual, stored = summarize(ledger, cfg), metric(result, model, period, cost_id)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                    require(abs(actual[field]-stored[field]) < 1e-9, "保存自身回撤控制指标不符")
                checks.append({"period": period, "cost": cost_id, "model": model, "origins_checked": len(decisions), "minimum_risk_scale": float(scale.min()), **actual})
                group = saved_cycles(ledger, dividends, cfg)
                cycles.extend({"period": period, "cost": cost_id, "model": model, **c} for c in group)
                total = sum(c["net_profit"] for c in group)
                top = sorted((c["net_profit"] for c in group if c["net_profit"] > 0), reverse=True)[:3]
                concentration.append({"period": period, "cost": cost_id, "model": model, "cycles": len(group), "losing_cycles": sum(c["net_profit"] < 0 for c in group), "net_profit": total, "largest_three_vs_net_profit": sum(top)/total if total > 0 else None})
                def compare(left, right):
                    return {"net_profit_difference": float(left.pnl.sum()-right.pnl.sum()), **{f"{field}_difference": float(left[field].sum()-right[field].sum()) for field in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}}
                delta = compare(ledger, old)
                require(abs(delta["net_profit_difference"]-delta["price_pnl_difference"]-delta["dividend_recognized_difference"]+delta["commission_difference"]+delta["slippage_cost_difference"]) < 1e-6, "相对原父策略的损益差额不守恒")
                differences.append({"period": period, "cost": cost_id, "model": model, "parent": parent_model, **delta})
                for year, group in ledger.groupby(ledger.date.dt.year):
                    annual_differences.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **compare(group, old[old.date.dt.year.eq(year)])})
    for name, rows in [("saved_account_and_cushion_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_parent_differences.csv", differences), ("saved_yearly_parent_differences.csv", annual_differences), ("saved_profit_concentration.csv", concentration)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    acceptance = []
    for model in PARENTS:
        ms = [metric(result, model, period, cost) for period in ["evaluation", "earlier_diagnostic"] for cost in cfg["costs"]]
        acceptance.append({"model": model, "all_four_point_sharpes_at_least_1_2": all(m["meets_point_target"] for m in ms), "all_four_annual_excess_positive": all(m["annualized_return_excess_vs_buy_hold"] > 0 for m in ms), "independent_validation": "NOT_ESTABLISHED"})
    require(not any(a["all_four_point_sharpes_at_least_1_2"] and a["all_four_annual_excess_positive"] for a in acceptance), "出现全面点目标通过时应继续完整验收")
    receipt = {"verified_at": now(), "status": "OWN_NAV_HIGH_WATER_TARGET_REQUEST_AND_ACCOUNT_CHECKED", "own_account_metrics_checked": len(checks), "decision_origins_checked": origin_count, "complete_actual_cycles": len(cycles), "new_accounts_or_model_fits": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_OWN_CUSHION_DRAWDOWN_REDUCED_SHARPE_NOT_IMPROVED"
    decision = "两个固定应用降低回撤，同时损失了更多收益，没有改善夏普。原91加控制主基础／压力为1.231／1.172，较早为0.295／0.257；原92加控制主基础／压力为0.897／0.829，较早为0.491／0.452。原91主基础1.233局部候选保留。本轮结束，不调底线比例、倍数、带宽或换父策略抢救结果。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "candidate_gates": acceptance, "historical_point_target_met": True, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第94轮后续：避免继续围绕退出和预算微调", "", decision, "", "本次连续完成93与94：指数前史补充给出更早的141个模型，却把早期盈利持仓截短；自身回撤控制八个新账户又显示亏损后的减仓同时压低随后收益。两组结果均未解决入场条件在不同市场状态下的差异。", "", "下一95优先从已经保存的每次进入事件出发，建立不含未来信息的市场状态及进入机会清单。先核对旧entry_path、entry_path_coverage、directional和contextual研究的标签与动作范围，直接复用可用资料；只有存在明确未试的新进入决策机制才登记。不是继续给93换权重，或给94换底线和倍数。", "", "最快路线仍为既有免费日线因子与原账户模块；EPS、申购及其他慢来源保持暂停。不把诊断中盈利最多的历史年份直接设成新开关，不按收益看完后再划新验证区间。若需要新模型，只使用当时已完成的进入机会标签；未来尚未结束的事件仍无观点。主、较早、费用、整手、分红、全部现金日和独立验证状态继续保留。", "", "当前完成94轮，369个不同设置、385个已评价来源版本、390个登记来源版本（含5个旧未运行）、1386个主评价记录。第95轮尚未登记、训练或产生新账户；不能把这一方向写成正在计算收益。持续研究保持，完整目标尚未完成。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第94轮：自身回撤控制能否改善夏普", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 实际结论", "", "本轮使用现有第91、92轮信号，各加同一层控制：以自己账户历史最高净值的80%作为参考底线，净值与底线之间的差额乘五决定原策略能使用的资金比例。两项应用事先固定，共八个新账户，零新增来源、零模型训练、零新参考账户。", "", "第91轮加控制后，主基础最大回撤从2.31%降至2.25%，但年化从3.36%降至3.30%，夏普从1.233降至1.231。较早回撤从11.07%降至9.89%，年化却从2.93%降至1.58%，夏普从0.431降至0.295。较小回撤并不自动带来较高夏普。", "", "第92轮加控制后，主基础回撤从5.70%降至5.24%，年化从4.13%降至3.73%，夏普从0.916降至0.897；较早回撤从10.91%降至9.45%，年化从3.85%降至2.64%，夏普从0.584降至0.491。控制层在净值尚未恢复时持续减少资金，也减少了后续盈利机会的收益。", "", "八个账户均未出现参考底线突破，也没有比例归零或原信号缺失。这只描述本次历史路径，不能保证未来不会因跳空或无法成交越过底线。主基础1.231仍是局部点结果：压力费用与较早历史没有达到1.2，年化超额也未同时转正。", "", "## 全部中文因子及进入退出规则", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:], "", f"五项必要测试通过，随后核对八个自身账户、{origin_count}次收盘的净值高点与目标及自身份额请求，以及{len(cycles)}个完整交易周期。测试中关闭保护的合成样例与原事件账户含分红记账完全相同；正式结果没有改变已冻结保护比例。每月原八项模型中文系数随附。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for path in RESEARCH.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            shutil.copy2(path, OUT / path.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    shutil.copy2(ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT / "沿用的两条策略全部中文规则.md")
    coefficients = ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coefficients, OUT / coefficients.name)
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in PARENTS:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 94, "study": result["study_id"], "title": "原91与92的自身净值高点回撤控制", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 2, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND94_DRAWDOWN_REDUCED_SHARPE_NOT_IMPROVED_FULL_GOAL_NOT_MET", count_warning="94轮，369不同设置，385已评价来源版本，390登记含5旧未运行，1386主评价记录；无效来源保留。", next_work={"status": "ENTRY_CONTEXT_AND_PRIOR_LABELS_DIAGNOSTIC_NOT_REGISTERED", "focus": "核对旧进入研究标签和动作范围，复用进入事件资料，辨别有实质差异的新进入机制", "source": str(NEXT.relative_to(ROOT))}, process_state_note="93与94的十二个新账户及必要核对完成；94没有改善夏普，95为进入状态诊断方向，尚未登记或训练。")
    index["deliveries"].append({"created_at": now(), "type": "OWN_CUSHION_RISK_ROUND94_FAST_CHINESE_RESULTS", "rounds": [94], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "对原父策略差额": differences, "验收": acceptance}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
