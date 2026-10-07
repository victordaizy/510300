"""原85资格的风险预算来源解释；逐条复算已有决定与买入，不运行账户。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import daily_supply_test_v1 as risk_base
from research.trend_expansion_phase_study_v1 import load, OUT as PRIOR
from research.joint_onset_clock_diagnostic_v1 import OUT as CLOCK, ERAS
from research.point_account_nr7_inputs_v1 import fill, fee, open_blocked
from research.point_first_passage_study_v1 import read, write_json, require, digest, now

OUT = ROOT / "reports/research/510300_joint_onset_budget_attribution_v1"


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "原预算归因已冻结。")
    require(read(CLOCK / "summary.json")["technical_decision"] == "TECH.R166", "时钟解释未完成。")
    sources = {}
    for s in read(CLOCK / "protocol.json")["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "旧来源改变。")
        sources[s["path"]] = s["sha256"]
    paths = [Path(__file__), CLOCK / "summary.json", CLOCK / "allocation_identity_summary.json",
             CLOCK / "results/全部85资格及原实际周期_触发时钟与A覆盖.parquet",
             CLOCK / "results/原投入金额与周期回报_全部资金恒等式.parquet",
             ROOT / "research/entry_volatility_sizing_v1.py", ROOT / "research/volatility_capped_entry_v1.py",
             ROOT / "research/self_performance_entry_v1.py"]
    for slug in ["510300_entry_volatility_sizing_v1", "510300_volatility_capped_entry_v1", "510300_self_performance_entry_v1"]:
        paths.append(ROOT / "reports/research" / slug / "acceptance_outcome.json")
    for p in paths:
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_ORIGINAL_JOINT_ONSET_BUDGET_ATTRIBUTION_V1", "technical_decision": "TECH.R167",
        "hypothesis": "原已发生投入金额与回报的协方差为负，可能涉及前期回撤余量或ES约束压缩后续赢家；还原所有预算来源，不能据后验收益放宽预算。",
        "scope": "全部85原资格/两费用；原决定和84实际买入逐条复算，83完成/1开放/1限价未成交保留，不运行新的资金路径。",
        "known_before_entry": "信号收盘原自身现金/应收/NAV/峰值与已知ES，原50%初始意向及固定cap_quantity；不使用未来结果重定预算。当前开盘执行约束单独标为execution_known。",
        "binding": "原点预定份额加100份会违反的仓位/ES/跳空预算项完整集合，七非空集合全部报告；跳空预算由原5%或半剩余回撤余量或两者同值决定，不能任选一主因。",
        "execution": "复算原下一开盘限价、原预算和现金缩减；BASE/STRESS费用及原risk_ok的STRESS压力评估均不改；匹配168实际买入份额/价格/费用。",
        "metric_role": "原周期子集和预算归因，四原时期两费用全部56单元；不是更换预算、等额/零费账户或新策略Sharpe。",
        "no_rescue": "不改变50%、ES、跳空或回撤预算、不把亏损预算类过滤重跑、不救回R165，不恢复旧简单波动率/半仓/自身盈利过滤。",
        "necessary_new_tests": 0, "required_saved_checks": "全部170原决定与168实际买入精确复算、原点早于执行、输入哈希及原周期合计保持。",
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "sources": [{"path": k, "sha256": v} for k,v in sorted(sources.items())],
    }, exclusive=True)
    print("全体原预算来源解释已固定，不修改预算或运行新账户。", flush=True)


def binding(q, raw, budget):
    exit_cost = q*(raw-fill(raw, -1, "STRESS"))+fee(q*fill(raw, -1, "STRESS"), "STRESS")
    position = q*raw > budget["position_budget"]+1e-8
    es = q*raw*budget["es95"]+exit_cost > budget["es_budget"]+1e-8
    gap = .1*q*raw+exit_cost > budget["gap_budget"]+1e-8
    require(bool(position or es or gap) == (not risk_base.risk_ok(q, raw, budget)), "预算项解释与原风险约束不同。")
    return int(position)+2*int(es)+4*int(gap)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "原预算归因已开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "冻结预算来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, _ = load()
    price = data.set_index("date")
    events = pd.read_parquet(CLOCK / "results/全部85资格及原实际周期_触发时钟与A覆盖.parquet")
    records, saved_checks = [], []
    for period in ("2015_2019", "2020_2026"):
        for cost in ("BASE", "STRESS"):
            folder = PRIOR / f"results/accounts/{period}/{cost}/JOINT_PHASE_START"
            daily = pd.read_parquet(folder / "daily.parquet").set_index("date")
            daily["known_peak"] = np.maximum.accumulate(np.r_[200000., daily.equity.to_numpy()])[1:]
            decisions = pd.read_parquet(folder / "decisions.parquet").set_index("origin")
            orders = pd.read_parquet(folder / "orders.parquet")
            buys = orders.loc[orders.side.eq("BUY")].set_index("origin")
            local = events.loc[events.period.eq(period) & events.cost.eq(cost)]
            for event in local.itertuples():
                origin, decision = event.origin, decisions.loc[event.origin]
                require(origin in daily.index, "原点没有真实资金状态，不以未来状态回填。")
                state = daily.loc[origin]
                require(state.shares == 0 and decision.shares_before == 0, "原生出生预算并非空仓状态。")
                raw = float(price.loc[origin, "close"])
                nav, peak, es = float(state.equity), float(state.known_peak), float(decision.known_es95)
                budget = risk_base.limits(nav, peak, es)
                maximum = int(.5*nav/raw//100)*100
                capped = risk_base.cap_quantity(raw, budget, maximum)
                require(capped == int(decision.desired_shares), "原点预算与已保存决定不同。")
                mask = binding(capped+100, raw, budget)
                require(mask > 0, "原点多一手没有违反预算，份额解释不完整。")
                base_gap, dd_gap = .05*nav, .5*max(0., nav-.9*peak)
                source = "REMAINING_DD_MARGIN" if dd_gap < base_gap-1e-8 else "BASE_AND_DD_TIED"
                execute_date = decision.execution_date
                i = int(data.index[data.date.eq(execute_date)][0])
                opening = float(price.loc[execute_date, "open"])
                blocked = open_blocked(data, i, 1)
                expected = 0
                if not blocked:
                    current = daily.loc[execute_date]
                    open_nav = nav+float(current.dividend_accrual)
                    open_budget = risk_base.limits(open_nav, peak, es)
                    expected = risk_base.cap_quantity(opening, open_budget, capped)
                    px = fill(opening, 1, cost)
                    while expected:
                        debit = expected*px+fee(expected*px, cost)
                        if debit <= state.cash+1e-8 and risk_base.risk_ok(expected, opening, open_budget, buying=True):
                            break
                        expected -= 100
                actual = int(buys.loc[origin, "quantity"]) if origin in buys.index else 0
                require(expected == actual, "原次开预算、现金与买入数量复算不符。")
                if actual:
                    order = buys.loc[origin]
                    require(order.fill_price == fill(opening, 1, cost)
                            and abs(order.commission-fee(actual*order.fill_price, cost)) < 1e-10, "原买入价格或费用不符。")
                records.append({
                    "origin": origin, "period": period, "cost": cost, "known_nav": nav, "known_peak": peak,
                    "known_drawdown": 1-nav/peak, "known_es95": es, "known_initial_50pct_shares": maximum,
                    "known_desired_shares": capped, "known_position_budget": budget["position_budget"],
                    "known_es_budget": budget["es_budget"], "known_gap_budget": budget["gap_budget"],
                    "known_gap_budget_source": source, "known_binding_mask": mask,
                    "execution_date": execute_date, "execution_open": opening, "execution_open_blocked": blocked,
                    "actual_buy_quantity": actual, "actual_origin_to_buy_notional_ratio": actual*opening/nav,
                    "execution_status": event.execution_status, "actual_status": event.actual_status,
                    "actual_net_return": event.actual_net_return, "actual_net_pnl": event.actual_net_pnl,
                    "birth_clock": event.birth_clock,
                })
            saved_checks.append({"period": period, "cost": cost, "matched_origin_decisions": len(local),
                                 "matched_actual_buys": len(buys), "new_accounts": 0})
    frame = pd.DataFrame(records)
    require(len(frame)==170 and int(frame.actual_buy_quantity.gt(0).sum())==168, "170决定或168既有买入缺失。")
    require(frame.origin.lt(frame.execution_date).all(), "原预算出现同时或未来决定。")
    table("全部原资格_事前预算来源与实际开盘份额", frame)
    stats = []
    for cost in ("BASE", "STRESS"):
        for era,(start,end) in ERAS.items():
            local = frame.loc[frame.cost.eq(cost) & frame.origin.between(start,end)]
            for mask in range(1,8):
                group = local.loc[local.known_binding_mask.eq(mask)]
                closed = group.loc[group.actual_status.eq("COMPLETE")]
                stats.append({
                    "era": era, "cost": cost, "binding_mask": mask, "known_origins": len(group),
                    "actual_buys": int(group.actual_buy_quantity.gt(0).sum()), "complete_cycles": len(closed),
                    "open_cycles": int(group.execution_status.eq("ACTUAL_OPEN").sum()),
                    "unexecuted_origins": int(group.execution_status.eq("REJECTED_AT_ACTUAL_OPEN").sum()),
                    "mean_known_drawdown": float(group.known_drawdown.mean()) if len(group) else np.nan,
                    "mean_actual_entry_exposure": float(group.actual_origin_to_buy_notional_ratio.mean()) if len(group) else np.nan,
                    "mean_cycle_net_return": float(closed.actual_net_return.mean()) if len(closed) else np.nan,
                    "original_complete_pnl_cny": float(closed.actual_net_pnl.sum()), "new_policy": False,
                })
    stats = pd.DataFrame(stats)
    table("原预算集合归因_全部56单元", stats)
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "执行期间冻结预算来源改变。")
    write_json(OUT / "summary.json", {
        "at": now(), "study": protocol["study"], "technical_decision": "TECH.R167",
        "status": "COMPLETED_FIXED_ORIGINAL_BUDGET_ATTRIBUTION_NO_POLICY_CHANGE",
        "origin_decisions_verified": 170, "actual_buy_quantities_verified": 168,
        "unique_known_origins": 85, "unique_actual_cycles": 84, "complete_cycles_per_cost": 83,
        "open_cycles_per_cost": 1, "unexecuted_origins_per_cost": 1, "all_group_cells": len(stats),
        "checks": saved_checks, "frozen_sources": len(protocol["sources"]),
        "pressure_binding_counts": frame.loc[frame.cost.eq("STRESS")].known_binding_mask.value_counts().to_dict(),
        "pressure_gap_budget_sources": frame.loc[frame.cost.eq("STRESS")].known_gap_budget_source.value_counts().to_dict(),
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "old_risk_budgets_preserved": True, "latest_financial_strategy_decision_preserved": "TECH.R165",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(frame.loc[frame.cost.eq("STRESS") & frame.origin.isin(pd.to_datetime(["2015-02-13", "2019-01-18", "2020-06-16", "2024-09-30"]))].to_string(index=False), flush=True)
    print("全体170原预算决定和168既有买入精确复算，预算集合56单元完成；未运行新账户或修改风险参数。", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="只解释原预算压缩与原成交份额，不改变金融规则。")
    parser.add_argument("command",choices=("freeze","run"))
    args=parser.parse_args()
    {"freeze":freeze,"run":run}[args.command]()
