"""只读四保存账户，交付全部资格、实际进出及具体量价解释。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.full_range_reversal_study_v1 import OUT, EXPLANATION, PRIMARY, PERIODS, COSTS, table
from research.full_range_reversal_explanation_v1 import KNOWN_TABLE, CASE_TABLE, MAP_TABLE
from research.deliver_downtrend_break_results_v1 import case_windows, capital_identity, percent, number
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def run():
    require(not (OUT / "delivery_receipt.json").exists(), "整日反转结果已经交付。")
    financial, explanation = read(OUT / "summary.json"), read(EXPLANATION / "summary.json")
    require(financial["technical_decision"] == "TECH.R181" and financial["new_primary_accounts"] == 4,
            "四新账户尚未全部完成。")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES",
            "保存账户核对尚未完成。")
    for path in (OUT / "protocol.json", EXPLANATION / "protocol.json"):
        for source in read(path)["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "本轮冻结来源改变。")
    known = pd.read_parquet(EXPLANATION / f"results/{KNOWN_TABLE}.parquet")
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    qualifications, actual, exits, counts, identities = [], [], [], [], []
    for period, (start, end) in PERIODS.items():
        for cost in COSTS:
            folder = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            trades = pd.read_parquet(folder / "trades.parquet")
            decisions = pd.read_parquet(folder / "decisions.parquet")
            orders = pd.read_parquet(folder / "orders.parquet")
            rejects = pd.read_parquet(folder / "rejections.parquet")
            require(trades.entry_origin.is_unique, "同一出生对应多个实际周期。")
            population = known.loc[known.bullish_range_reversal & known.date.isin(decisions.origin)]
            statuses = {}
            for event in population.itertuples(index=False):
                decision = decisions.loc[decisions.origin.eq(event.date)]
                require(len(decision) == 1, "事件缺少唯一当时决定。")
                d = decision.iloc[0]
                t = trades.loc[trades.entry_origin.eq(event.date)]
                rejected = rejects.loc[rejects.origin.eq(event.date)] if len(rejects) else rejects
                row = event._asdict()
                row.update(period=period, cost=cost, desired_shares=int(d.desired_shares), shares_before=int(d.shares_before),
                           planned_execution_date=d.execution_date, decision_reason=d.reason,
                           known_es95=d.known_es95, source_weight=d.source_weight, rejection_reason=None)
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
                        status = "ALREADY_HOLDING_NO_NEW_CYCLE"
                    elif len(rejected):
                        require(len(rejected) == 1, "同一出生发生多个未解释的开盘拒绝。")
                        status = "CANCELLED_"+str(rejected.reason.iloc[0])
                        row["rejection_reason"] = str(rejected.reason.iloc[0])
                    else:
                        require(int(d.desired_shares) == 0, "未成交请求缺少信息或份额解释。")
                        status = "NO_SIZE_OR_UNKNOWN_PRIOR_INFORMATION"
                    row.update(cycle_id=-1, entry_date=pd.NaT, exit_date=pd.NaT, actual_net_return=np.nan,
                               actual_net_pnl=np.nan, entry_quantity=0, actual_entry_raw=np.nan,
                               actual_entry_fill=np.nan, actual_buy_debit=np.nan)
                require(int(event.reversal_reference_index) == int(event.origin_index)-1,
                        "整日区间引用没有来自上一真实日。")
                require(int(event.known_cash_low_ticks) < int(event.previous_cash_low_ticks)
                        and int(event.known_cash_close_ticks) > int(event.previous_cash_high_ticks),
                        "出生没有严格跨越完整上一日区间。")
                row["execution_status"] = status
                statuses[status] = statuses.get(status, 0)+1
                qualifications.append(row)
            require(sum(statuses.values()) == len(population), "全体资格状态未还原。")
            for trade in trades.itertuples(index=False):
                observed = known.loc[known.date.eq(trade.entry_origin)]
                require(len(observed) == 1 and bool(observed.bullish_range_reversal.iloc[0]),
                        "实际入场没有已知整日反转。")
                row = {**trade._asdict(), "period": period, "cost": cost}
                row.update({"entry_known_"+key: value for key, value in observed.iloc[0].to_dict().items()})
                row["entry_known_initial_raw_floor_at_origin"] = (
                    int(observed.reversal_floor_ticks.iloc[0])*.001-float(observed.cash_shift.iloc[0]))
                sold = orders.loc[orders.cycle_id.eq(trade.cycle_id) & orders.side.eq("SELL")]
                final = sold.iloc[-1] if trade.status == "COMPLETE" and len(sold) else None
                row.update(final_exit_raw=final.raw_open if final is not None else np.nan,
                           final_exit_fill=final.fill_price if final is not None else np.nan)
                actual.append(row)
                if final is not None:
                    decision = decisions.loc[decisions.origin.eq(final.origin)].iloc[0]
                    event = known.loc[known.date.eq(final.origin)].iloc[0]
                    floor = int(decision.known_current_reversal_floor_ticks)
                    require(final.date > trade.entry_date and final.origin < final.date, "退出违反次开或T+1。")
                    if final.reason == "FULL_RANGE_REVERSAL_FAILED_KNOWN_CLOSE":
                        require(bool(event.current_quote_known) and int(event.known_cash_close_ticks) <= floor,
                                "收盘线退出没有已知线失败。")
                    elif final.reason == "FULL_RANGE_REVERSAL_OPPOSING_CLOSE":
                        require(bool(event.bearish_range_reversal) and int(event.known_cash_close_ticks) > floor,
                                "相反事件退出没有严格形态或没有遵循失效优先顺序。")
                    exits.append({"period": period, "cost": cost, "cycle_id": int(trade.cycle_id),
                                  "entry_origin": trade.entry_origin, "exit_origin": final.origin,
                                  "exit_date": final.date, "exit_reason": final.reason, "known_floor_ticks": floor,
                                  "exit_close_ticks": int(event.known_cash_close_ticks),
                                  "bearish_range_reversal": bool(event.bearish_range_reversal),
                                  "actual_net_return": trade.net_return, "actual_net_pnl": trade.net_pnl,
                                  "post_result_description_only": True})
            counts.append({"period": period, "cost": cost, "reversal_origins": len(population),
                           "actual_cycles": len(trades), "completed": int(trades.status.eq("COMPLETE").sum()),
                           "open": int(trades.status.ne("COMPLETE").sum()), "execution_status_counts": statuses})
            identities.append(capital_identity(trades, period, cost))
    qualifications, actual, exits = pd.DataFrame(qualifications), pd.DataFrame(actual), pd.DataFrame(exits)
    cases = pd.read_parquet(EXPLANATION / f"results/{CASE_TABLE}.parquet")
    case_actual = case_windows(actual, cases)
    mapped = pd.read_parquet(EXPLANATION / f"results/{MAP_TABLE}.parquet")
    table("全部整日反转资格_当时量价及真实执行状态", qualifications)
    table("全部实际进出点位_已知整日反转量价及真实资金", actual)
    table("四案例相交全部实际周期_不筛收益", case_actual)
    table("全部完成退出_收盘线或相反事件", exits)
    table("全部四场景完成周期_实际资金恒等式", pd.DataFrame(identities))
    primary = metrics.loc[metrics.policy.eq(PRIMARY)]
    economic_passes = sum(bool(row["economic_gate_passed"]) for row in financial["comparisons"])
    point_quality = bool(primary.p_times_b.gt(1).all() & primary.standard_expectancy_loss_units.gt(0).all())
    accepted = bool(financial["all_four_economic_gates_passed"] and financial["historical_stability_gate_passed"])
    boundary = {
        "at": now(), "status": "FIXED_RANGE_REVERSAL_COMPLETED_NO_NEW_ADMITTED_FINANCIAL_CANDIDATE",
        "after_decisions": ["TECH.R179", "TECH.R180", "TECH.R181"], "fixed_policy_status": financial["status"],
        "accepted_fact": "局部收盘强反转的时钟不同于日周动量转正；早期修复、下跌途中失败、上涨后段和急涨无信号均已展开，不能只保留2019赢家。",
        "next_required_input": "先对全98出生及全部保存周期核对出生位置、只抬事件低点、相反退出/收盘线失败、实际资金及次开差额；同时保留原49上涨无该形态的覆盖，0新账户。诊断完成后只有实质不同信息及完整用途、真实新样本或来源/实现错误可进一步准入。",
        "no_parameter_rescue": True,
        "not_authorized_as_new_policy": ["据结果挑早期/年份/赢家", "增量MACD/量/RV过滤", "改变区间参考或收盘/反向退出", "放宽原预算或混A"],
        "next_new_account_strategy": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "new_admitted_unrun_candidates": 0, "new_accounts_in_next_saved_diagnostic": 0,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    }
    write_json(OUT / "next_all_reversal_lifecycle_diagnostic_proposal.json", boundary, exclusive=True)
    lines = ["# 整日区间反转：具体上涨解释、全部点位及真实账户", "",
             "本轮依用户要求先解释具体上涨及失败的量价和日周指标，再反推当时可知点位。TECH.R179解释、R180登记和R181四实际账户已完成，原冻结策略保持。", "",
             "## 具体上涨及当时指标", "",
             "2019-01-04跨昨低且收盘跨昨高：相对量1.623，日DIF−0.054674、日柱−0.015543、上一完整周柱−0.008664，RV比0.868。价格先局部反转、长周期动量仍弱；01-17相反形态又给出失效。02-22日周动量已正时再次同向；04-03、04-16、04-19高位也有同向事件，不能合并为同一个早期买点。", "",
             "2020-03-10同向事件的相对量1.983，日DIF+0.018313、日柱几乎0、上一周柱负，RV比1.669；随后仍下跌。05-06同向事件日动量已正而上一周柱仍负、相对量1.174，06-11有相反形态。7月急涨没有新的同向完整反转，局部反转不是所有趋势启动的必要条件。", "",
             "2024-09-24及月底急涨没有该形态，直到10-18才出现：收盘4.016、事件低点3.846、量1.305、日DIF+0.162373、日柱+0.013743、上一周柱+0.144025、RV比4.907。它是急涨后的高波动整理信号，不是提前识别9月启动。", "",
             "2015原06-29/30短反弹没有该形态，固定观察窗后段07-09出现：收盘3.809、相对量1.143，日DIF/日柱/上一周柱均负、RV比3.620，较长周期下行和高波动仍在。7月事件与原案例18并非同一原分段，仅在固定观察窗中相交。", "",
             "[四例及反推全文](../510300_full_range_reversal_explanation_v1/具体上涨与点位反推.md)。形态只在收盘后确认，日线无法给出日内极值先后或主动买卖流，也不证明上涨的因果。", "",
             "固定完整用途：LOW<昨LOW且CLOSE>昨HIGH出生后次开一次；初始线为事件LOW，开盘已到/低于线取消；持有新同向形态只抬事件低点，已知CLOSE<=线或HIGH>昨HIGH且CLOSE<昨LOW的相反形态次合法开退出。未知不强退、风险只减，无加仓/混A/固定获利目标/时间退出/量或MACD过滤。", "",
             f"全3488日、2855当期原点、{explanation['bullish_reversals_since_2015']}同向/{explanation['bearish_reversals_observation_and_long_exit_only']}相反事件保留。全部98出生路径已失效，其中42是在线上出现相反事件；路径不是实际完成交易。原61分段/49正式上涨、2846旧标签/2826成熟/20删失/9尾部保持。", "",
             "## 完整账户结果", "",
             "原20万元、252日、完整现金日、股息、次开、T+1/100份/.001、两费用/最低佣金5元及ES/跳空/DD预算不变。B为实际完成盈利周期平均净回报除以亏损周期平均损失，pB和标准期望pB−q分别报告；计划失效距离不代替实际B。", "",
             "| 时期 | 费用 | 净年化 | 净夏普 | 最大回撤 | 完成/开放 | 胜率 | B | pB | pB−q | 完整年均次数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary.itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {percent(row.net_cagr,4)} | {number(row.net_sharpe)} | {percent(row.max_drawdown)} | {row.completed_cycles}/{row.unfinished_cycles} | {percent(row.win_rate)} | {number(row.payoff)} | {number(row.p_times_b)} | {number(row.standard_expectancy_loss_units)} | {number(row.average_full_year_cycles,2)} |")
    lines.extend(["", f"四单场景经济门通过{economic_passes}个，整体={financial['all_four_economic_gates_passed']}，历史稳定门={financial['historical_stability_gate_passed']}；实际pB四场景均>1且标准期望正={point_quality}。完整裁决：{financial['status']}。", "",
                  "| 时期 | 费用 | 原控制 | 净年化 | 净夏普 | 完整年均次数 |", "|---|---|---|---:|---:|---:|"])
    for row in metrics.loc[metrics.policy.ne(PRIMARY)].itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {row.policy} | {percent(row.net_cagr,4)} | {number(row.net_sharpe)} | {number(row.average_full_year_cycles,2)} |")
    lines.extend(["", "| 时期 | 费用 | 资格 | 实际周期 | 完成/开放 | 全部状态 |", "|---|---|---:|---:|---:|---|"])
    for row in counts:
        lines.append(f"| {row['period']} | {row['cost']} | {row['reversal_origins']} | {row['actual_cycles']} | {row['completed']}/{row['open']} | {row['execution_status_counts']} |")
    lines.extend(["", f"全部{len(qualifications)}资格行、{len(actual)}实际周期行及{len(exits)}完成退出逐一保留。未成交与开放不填入完成胜率；原首日前一收盘及期末无次开分列，案例开放观察只到自身账户末日。", "",
                  "四案例相交的全部压力成本实际周期如下，包含跨越观察窗边缘的已有周期；不是事后按低点和最高点成交。", "",
                  "| 原案例 | 出生 | 实际入场 | 实际退出 | 净回报 | 净金额/元 | 状态 |", "|---|---|---|---|---:|---:|---|"])
    for row in case_actual.loc[case_actual.cost.eq("STRESS")].itertuples(index=False):
        end_date = "未完成" if pd.isna(row.exit_date) else row.exit_date.strftime("%Y-%m-%d")
        lines.append(f"| {row.original_episode_id} | {row.entry_origin:%Y-%m-%d} | {row.entry_date:%Y-%m-%d} | {end_date} | {percent(row.net_return)} | {number(row.net_pnl,2)} | {row.status} |")
    lines.extend(["", "Σ实际支出×周期净回报的均值项与资金对应项只还原原账本，不生成等额账户。全部四资金恒等式见保存表，实际投入不同，点位等权收益不能替代全账户收益。", "",
                  "## 裁决、验证和下一步", "",
                  ("只接受本用途的历史开发增量，尚无独立验证，原策略和前瞻不自动替换。" if accepted else "该固定完整用途未通过全部经济与稳定门，拒绝作为完整模型改善并关闭。早期局部点位或单项夏普改善仍可归档，不能选时期或根据好分组加过滤救回。"), "",
                  "九项必要测试、四真实前缀、八A/纯价格日账/周期/订单精确复现，四新主账户一次测量，保存资金/库存/次开/T+1/股息/成本/指标核对通过。首次输入测试仅测试例除息算术差.001，修正前主结果读取0，原逻辑未改；首次失败和测试源码保存。未联网采集、未拟合或新建训练标签。", "",
                  "全部历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED，global DSR/PBO NOT_COMPUTED；不能宣称去除过拟合或已完成收益夏普目标。没有实盘订单、空头或期权收益。", "",
                  boundary["next_required_input"], "进一步策略必须先核旧完整用途并单独登记，已准入待跑新金融候选0。原A/POINT十二真实前瞻值与1008实际交易日计划保持。", "",
                  "- [全部资格及执行状态](results/全部整日反转资格_当时量价及真实执行状态.csv)",
                  "- [全部真实进出及已知指标/资金](results/全部实际进出点位_已知整日反转量价及真实资金.csv)",
                  "- [四案例全部实际周期](results/四案例相交全部实际周期_不筛收益.csv)",
                  "- [全部失效退出](results/全部完成退出_收盘线或相反事件.csv)",
                  "- [十二账户比较](results/完整账户共同口径比较.csv)、[全部逐年收益及次数](results/逐年净收益与实际周期次数.csv)",
                  "- [冻结](protocol.json)、[全部结果及区间](summary.json)、[保存核对](saved_result_verification.json)",
                  "- [下一全体归因](next_all_reversal_lifecycle_diagnostic_proposal.json)", ""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_RANGE_REVERSAL_POINTS_CASES_EXITS_AND_SAVED_IDENTITIES_DELIVERED",
        "technical_decision": "TECH.R181", "scenario_counts": counts,
        "all_qualification_rows": len(qualifications), "all_actual_cycle_rows": len(actual),
        "all_completed_cycles": int(actual.status.eq("COMPLETE").sum()),
        "all_open_cycles": int(actual.status.ne("COMPLETE").sum()),
        "all_four_actual_point_quality_passed": point_quality, "individual_economic_gate_passes": economic_passes,
        "overall_economic_gate_passed": financial["all_four_economic_gates_passed"],
        "historical_stability_gate_passed": financial["historical_stability_gate_passed"],
        "actual_complete_exit_rows": len(exits), "capital_identity_cells": len(identities),
        "original_episodes_preserved": len(mapped), "original_admitted_waves": int(mapped.admitted.sum()),
        "case_cycle_rows": len(case_actual), "open_case_windows_clipped_at_own_account_end": True,
        "new_accounts_in_delivery": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in [
            Path(__file__), ROOT / "research/deliver_downtrend_break_results_v1.py", OUT / "summary.json",
            OUT / "saved_result_verification.json", EXPLANATION / "summary.json", report]],
    }, exclusive=True)
    print(f"全部{len(qualifications)}资格、{len(actual)}实际周期及四案例已交付；没有重跑账户。", flush=True)


if __name__ == "__main__":
    run()
