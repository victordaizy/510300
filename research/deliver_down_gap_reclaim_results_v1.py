"""只读一次测量保存结果，交付全部收复资格、真实点位和原四案例。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.down_gap_reclaim_study_v1 import OUT, EXPLANATION, PRIMARY, PERIODS, COSTS, table
from research.down_gap_reclaim_explanation_v1 import KNOWN_TABLE, CASE_TABLE, MAP_TABLE
from research.deliver_downtrend_break_results_v1 import case_windows, capital_identity, percent, number
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def run():
    require(not (OUT / "delivery_receipt.json").exists(), "向下缺口收复结果已经交付。")
    financial, explanation = read(OUT / "summary.json"), read(EXPLANATION / "summary.json")
    require(financial["technical_decision"] == "TECH.R185" and financial["new_primary_accounts"] == 4, "四新账户尚未完成。")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES", "保存核对尚未完成。")
    for protocol in (OUT / "protocol.json", EXPLANATION / "protocol.json"):
        for source in read(protocol)["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源改变。")
    known = pd.read_parquet(EXPLANATION / f"results/{KNOWN_TABLE}.parquet")
    mapped = pd.read_parquet(EXPLANATION / f"results/{MAP_TABLE}.parquet")
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    qualifications, actual, exits, counts, identities = [], [], [], [], []
    for period, (start, end) in PERIODS.items():
        for cost in COSTS:
            folder = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            trades, decisions, orders, rejects = [pd.read_parquet(folder / f"{name}.parquet")
                                                  for name in ("trades", "decisions", "orders", "rejections")]
            require(trades.entry_origin.is_unique, "同一收复对应多个实际周期。")
            population = known.loc[known.reclaim_event & known.date.isin(decisions.origin)]
            statuses = {}
            for event in population.itertuples(index=False):
                d = decisions.loc[decisions.origin.eq(event.date)]
                require(len(d) == 1, "收复没有唯一当时决定。")
                d = d.iloc[0]
                t = trades.loc[trades.entry_origin.eq(event.date)]
                rejected = rejects.loc[rejects.origin.eq(event.date)] if len(rejects) else rejects
                row = event._asdict()
                row.update(period=period, cost=cost, desired_shares=int(d.desired_shares), shares_before=int(d.shares_before),
                           planned_execution_date=d.execution_date, decision_reason=d.reason, known_es95=d.known_es95,
                           source_weight=d.source_weight, rejection_reason=None)
                if len(t):
                    trade = t.iloc[0]
                    status = str(trade.status)
                    row.update(cycle_id=int(trade.cycle_id), entry_date=trade.entry_date, exit_date=trade.exit_date,
                               actual_net_return=trade.net_return, actual_net_pnl=trade.net_pnl,
                               entry_quantity=int(trade.entry_quantity), actual_entry_raw=float(trade.entry_raw),
                               actual_entry_fill=float(trade.entry_price), actual_buy_debit=float(trade.buy_debit))
                else:
                    if pd.isna(d.execution_date):
                        status = "TERMINAL_NO_NEXT_SESSION_NO_EXECUTION"
                    elif int(d.shares_before) > 0:
                        status = "ALREADY_HOLDING_OR_LOCKED_EXIT_NO_NEW_CYCLE"
                    elif len(rejected):
                        require(len(rejected) == 1, "同一收复有多个未解释拒绝。")
                        status = "CANCELLED_"+str(rejected.reason.iloc[0])
                        row["rejection_reason"] = str(rejected.reason.iloc[0])
                    else:
                        require(int(d.desired_shares) == 0, "未执行的正请求没有开盘解释。")
                        status = "NO_SIZE_OR_UNKNOWN_PRIOR_INFORMATION"
                    row.update(cycle_id=-1, entry_date=pd.NaT, exit_date=pd.NaT, actual_net_return=np.nan,
                               actual_net_pnl=np.nan, entry_quantity=0, actual_entry_raw=np.nan,
                               actual_entry_fill=np.nan, actual_buy_debit=np.nan)
                parent = known.iloc[int(event.reclaim_reference_index)]
                require(bool(parent.new_down_gap) and parent.date < event.date, "收复锚不是此前已知完整下跌缺口。")
                require(str(parent.observed_anchor_id) == str(event.reclaim_anchor_id), "没有引用最新出生锚。")
                require(int(event.known_cash_close_ticks) > int(event.reclaim_upper_ticks), "收复未严格越过上沿。")
                row["execution_status"] = status
                statuses[status] = statuses.get(status, 0)+1
                qualifications.append(row)
            require(sum(statuses.values()) == len(population), "全体资格未还原。")
            for trade in trades.itertuples(index=False):
                observed = known.loc[known.date.eq(trade.entry_origin)]
                require(len(observed) == 1 and bool(observed.reclaim_event.iloc[0]), "实际入场没有已知首次收复。")
                event = observed.iloc[0]
                require(int(trade.fixed_reclaim_floor_ticks) == int(trade.entry_reclaim_floor_ticks) == int(event.reclaim_floor_ticks), "固定原锚下沿被改变。")
                row = {**trade._asdict(), "period": period, "cost": cost}
                row.update({"entry_known_"+key: value for key, value in event.to_dict().items()})
                row["entry_known_initial_raw_floor_at_origin"] = int(event.reclaim_floor_ticks)*.001-float(event.cash_shift)
                sold = orders.loc[orders.cycle_id.eq(trade.cycle_id) & orders.side.eq("SELL")]
                final = sold.iloc[-1] if trade.status == "COMPLETE" and len(sold) else None
                row.update(final_exit_raw=final.raw_open if final is not None else np.nan,
                           final_exit_fill=final.fill_price if final is not None else np.nan)
                actual.append(row)
                if final is not None:
                    decision = decisions.loc[decisions.origin.eq(final.origin)].iloc[0]
                    at_exit = known.loc[known.date.eq(final.origin)].iloc[0]
                    floor = int(decision.known_current_reclaim_floor_ticks)
                    require(final.date > trade.entry_date and final.origin < final.date, "退出违反次开或T+1。")
                    if final.reason == "DOWN_GAP_RECLAIM_FAILED_KNOWN_CLOSE":
                        require(bool(at_exit.current_quote_known) and int(at_exit.known_cash_close_ticks) <= floor, "收盘退出没有已知固定线失败。")
                    elif final.reason == "DOWN_GAP_RECLAIM_NEW_DOWN_GAP":
                        require(bool(at_exit.new_down_gap) and int(at_exit.known_cash_close_ticks) > floor, "新缺口退出没有严格形态或未按失败优先。")
                    exits.append({"period": period, "cost": cost, "cycle_id": int(trade.cycle_id),
                                  "entry_origin": trade.entry_origin, "exit_origin": final.origin, "exit_date": final.date,
                                  "exit_reason": final.reason, "known_fixed_floor_ticks": floor,
                                  "exit_close_ticks": int(at_exit.known_cash_close_ticks), "new_down_gap": bool(at_exit.new_down_gap),
                                  "actual_net_return": trade.net_return, "actual_net_pnl": trade.net_pnl})
            counts.append({"period": period, "cost": cost, "reclaim_origins": len(population), "actual_cycles": len(trades),
                           "completed": int(trades.status.eq("COMPLETE").sum()), "open": int(trades.status.ne("COMPLETE").sum()),
                           "execution_status_counts": statuses})
            identities.append(capital_identity(trades, period, cost))
    qualifications, actual, exits = pd.DataFrame(qualifications), pd.DataFrame(actual), pd.DataFrame(exits)
    cases = pd.read_parquet(EXPLANATION / f"results/{CASE_TABLE}.parquet")
    case_actual = case_windows(actual, cases)
    table("全部收复资格_原锚当时量价及执行状态", qualifications)
    table("全部实际进出点位_已知收复量价及真实资金", actual)
    table("四案例相交全部实际周期_不筛收益", case_actual)
    table("全部完成退出_固定线或新向下缺口", exits)
    table("全部四场景完成周期_实际资金恒等式", pd.DataFrame(identities))
    primary = metrics.loc[metrics.policy.eq(PRIMARY)]
    passes = sum(bool(row["economic_gate_passed"]) for row in financial["comparisons"])
    quality = bool(primary.p_times_b.gt(1).all() & primary.standard_expectancy_loss_units.gt(0).all())
    accepted = bool(financial["all_four_economic_gates_passed"] and financial["historical_stability_gate_passed"])
    admitted = mapped.loc[mapped.admitted]
    next_action = (
        "对所有保存收复周期及原49上涨作一次完整位置/资金路径归因：原缺口出生到首次收复、已知固定线或新缺口退出、"
        "实际股数/已发生股息/成本/最后次开时钟、全部失败和开放，以及没有收复的原上涨覆盖。"
        "最高收盘财富只解释，不是新增止盈；不据2024好案例加量/MACD过滤，不从失败修改最新锚/消耗或退出。"
        "归因后新金融准入须有实质不同信息及完整用途、真正新样本或来源/实现错误。")
    boundary = {
        "at": now(), "status": "FIXED_DOWN_GAP_RECLAIM_COMPLETED_NO_NEW_ADMITTED_FINANCIAL_CANDIDATE",
        "after_decisions": ["TECH.R183", "TECH.R184", "TECH.R185"], "fixed_policy_status": financial["status"],
        "next_required_input": next_action, "next_new_account_strategy": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "new_admitted_unrun_candidates": 0, "new_accounts_in_next_saved_diagnostic": 0,
        "no_parameter_rescue": True, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    }
    write_json(OUT / "next_all_reclaim_lifecycle_diagnostic_proposal.json", boundary, exclusive=True)
    lines = ["# 向下整日缺口收复：具体量价、实际多头点位与完整收益夏普", "",
             "依用户要求，TECH.R183先展开原具体上涨及全体失败，再以已固定完整用途反推收盘后可知点位；R184登记、R185原四场景实际测量已经完成。", "",
             "2019直到3月18日回调收复才确认，日DIF/上一周柱正而日柱负，量0.975倍，并没有提前识别1月启动。2020年4月2日收复时量0.741倍、日柱正而DIF/上一周柱负；6月1日另一个收复时日柱/上一周柱仍负。2024年9月24日收复此前9月9日缺口，量3.338倍、日柱正而DIF/上一周柱负，价格修复在慢动量转正前可知。2015年7月9日则在原短反弹后，日周动量负、RV比3.620，不能凭一次收复判整个下行结束。", "",
             "[四例全部当时数值、阶段解释和图](../510300_down_gap_reclaim_explanation_v1/具体上涨与点位反推.md)。日线量价不能证明主动买盘或上涨因果，指标符号不自动形成过滤。", "",
             explanation["future_complete_policy_intent_fixed"], "",
             f"全3488原点、112原锚（2015起91）、当期{explanation['reclaims_since_2015']}首次收复、全原锚终态{explanation['all_anchor_terminal_counts']}均保留；原61分段/49上涨及2846旧标签/2826成熟/20删失/9尾部保持。"
             f"原49上涨底峰间无首次收复{int(admitted.reclaims_bottom_to_peak.eq(0).sum())}段，确认至峰无收复{int(admitted.reclaims_confirm_to_peak.eq(0).sum())}段；覆盖只是事后对齐。", "",
             "## 实际完整账户", "",
             "原20万元、252日、完整现金日、分红应收/到账、T+1/100份/.001、两费用/最低佣金5元及ES/跳空/DD预算保持。原BASE买入也按原STRESS风险检查；形态只在收盘可知，下一开盘尝试一次。B为实际完成盈利周期平均净回报/亏损周期平均净损失，pB与标准期望pB−q均列，不以初始失效距离代替真实B。", "",
             "| 时期 | 费用 | 净年化 | 净夏普 | 最大回撤 | 完成/开放 | 胜率 | B | pB | pB−q | 完整年均次数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary.itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {percent(row.net_cagr,4)} | {number(row.net_sharpe)} | {percent(row.max_drawdown)} | {row.completed_cycles}/{row.unfinished_cycles} | {percent(row.win_rate)} | {number(row.payoff)} | {number(row.p_times_b)} | {number(row.standard_expectancy_loss_units)} | {number(row.average_full_year_cycles,2)} |")
    lines.extend(["", "| 时期 | 费用 | 原控制 | 净年化 | 净夏普 | 完整年均次数 |", "|---|---|---|---:|---:|---:|"])
    for row in metrics.loc[metrics.policy.ne(PRIMARY)].itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {row.policy} | {percent(row.net_cagr,4)} | {number(row.net_sharpe)} | {number(row.average_full_year_cycles,2)} |")
    lines.extend(["", f"四单场景经济门通过{passes}，整体={financial['all_four_economic_gates_passed']}，历史稳定门={financial['historical_stability_gate_passed']}；实际点位pB四场景均>1且标准期望正={quality}。固定完整用途裁决：{financial['status']}。20/252日配对区块各2000/固定种子，全部差值区间保存在summary.json，不选择有利尺度。", "",
                  "| 时期 | 费用 | 资格 | 实际周期 | 完成/开放 | 全部执行状态 |", "|---|---|---:|---:|---:|---|"])
    for row in counts:
        lines.append(f"| {row['period']} | {row['cost']} | {row['reclaim_origins']} | {row['actual_cycles']} | {row['completed']}/{row['open']} | {row['execution_status_counts']} |")
    lines.extend(["", "## 原四案例与真实执行", "",
                  "下表保留与固定案例观察窗相交的全部压力周期，包含窗口前已经进入的周期。开放只截至本周期所属账户末日，未成交/开放不填入完成胜率；未按原事后低点或最高点交易。", "",
                  "| 原案例 | 收复确认 | 实际进场 | 实际退出 | 净回报 | 净金额/元 | 状态 |", "|---|---|---|---|---:|---:|---|"])
    for row in case_actual.loc[case_actual.cost.eq("STRESS")].itertuples(index=False):
        exit_date = "未完成" if pd.isna(row.exit_date) else row.exit_date.strftime("%Y-%m-%d")
        lines.append(f"| {row.original_episode_id} | {row.entry_origin:%Y-%m-%d} | {row.entry_date:%Y-%m-%d} | {exit_date} | {percent(row.net_return)} | {number(row.net_pnl,2)} | {row.status} |")
    lines.extend(["", f"全部{len(qualifications)}资格行、{len(actual)}实际周期、{len(exits)}完成退出以及四资金恒等式已交付。Σ实际支出×周期净回报只还原原实际资金，不另生成等额或忽略仓位的账户。", "",
                  "## 裁决和下一步", "",
                  ("接受本完整用途历史开发增量，但不自动替换原策略，仍未建立独立验证。" if accepted else "该固定完整用途未通过全部经济和稳定门，拒绝作为完整模型提高并关闭。保留2024等真实点位的开发事实，不能据赢家加量/MACD/RV、改锚/退出、选时期或混A营救。"), "",
                  "四输入及五账户必要测试首轮通过、四真实前缀一致、八原A/纯价格日账/订单/周期精确复现、四主账户一次运行及保存资金/库存/次开/T+1/指标核对通过。四图已查看；解释表格仅根据原保存行重排并保留初稿，定义和结果不变。0拟合/训练标签/采集。", "",
                  "全部已用历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED；固定规则及历史改进不等于去除过拟合，完整收益夏普目标尚未完成。", "",
                  next_action, "", "原A/POINT十二真实前瞻及1008实际交易日计划保持，未将本历史形态自动加入前瞻。", "",
                  "- [全部资格和执行](results/全部收复资格_原锚当时量价及执行状态.csv)",
                  "- [全部实际点位和当时指标](results/全部实际进出点位_已知收复量价及真实资金.csv)",
                  "- [原四案例全部相交周期](results/四案例相交全部实际周期_不筛收益.csv)",
                  "- [全部退出](results/全部完成退出_固定线或新向下缺口.csv)",
                  "- [十二账户共同比较](results/完整账户共同口径比较.csv)、[逐年收益与次数](results/逐年净收益与实际周期次数.csv)",
                  "- [冻结](protocol.json)、[结果及全部区间](summary.json)、[保存核对](saved_result_verification.json)",
                  "- [下一完整归因提案](next_all_reclaim_lifecycle_diagnostic_proposal.json)", ""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_RECLAIM_POINTS_CASES_EXITS_AND_SAVED_IDENTITIES_DELIVERED",
        "technical_decision": "TECH.R185", "scenario_counts": counts, "all_qualification_rows": len(qualifications),
        "all_actual_cycle_rows": len(actual), "all_completed_cycles": int(actual.status.eq("COMPLETE").sum()),
        "all_open_cycles": int(actual.status.ne("COMPLETE").sum()), "all_four_actual_point_quality_passed": quality,
        "individual_economic_gate_passes": passes, "overall_economic_gate_passed": financial["all_four_economic_gates_passed"],
        "historical_stability_gate_passed": financial["historical_stability_gate_passed"], "actual_complete_exit_rows": len(exits),
        "capital_identity_cells": 4, "original_episodes_preserved": len(mapped), "original_admitted_waves": len(admitted),
        "admitted_waves_without_bottom_to_peak_reclaim": int(admitted.reclaims_bottom_to_peak.eq(0).sum()),
        "admitted_waves_without_confirm_to_peak_reclaim": int(admitted.reclaims_confirm_to_peak.eq(0).sum()),
        "case_cycle_rows": len(case_actual), "open_case_windows_clipped_at_own_account_end": True,
        "new_accounts_in_delivery": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), OUT / "summary.json", OUT / "saved_result_verification.json", EXPLANATION / "summary.json", report]],
    }, exclusive=True)
    print(f"全部{len(qualifications)}资格、{len(actual)}真实周期与原四案例已交付；无重跑。", flush=True)


if __name__ == "__main__":
    run()
