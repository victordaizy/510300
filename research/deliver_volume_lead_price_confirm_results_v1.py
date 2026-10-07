"""交付全部保存点位和实际账户，补具体阶段的动量时钟解释，不修改策略。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.volume_lead_price_confirm_study_v1 import OUT, EXPLANATION, PRIMARY, PERIODS, COSTS, table
from research.volume_lead_price_confirm_explanation_v1 import KNOWN_TABLE, ATLAS
from research.deliver_downtrend_break_results_v1 import case_windows, capital_identity, percent, number
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def run():
    require(not (OUT / "delivery_receipt.json").exists(), "量价先后研究已经交付，不覆盖。")
    financial, explanation = read(OUT / "summary.json"), read(EXPLANATION / "summary.json")
    require(financial["technical_decision"] == "TECH.R189" and financial["new_primary_accounts"] == 4, "四主账户尚未实际完成。")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES",
            "四保存账户尚未核对通过。")
    known = pd.read_parquet(EXPLANATION / f"results/{KNOWN_TABLE}.parquet")
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    qualifications, actual, exits, counts, identities = [], [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            trades, decisions, orders, rejects = [pd.read_parquet(folder / f"{name}.parquet")
                                                  for name in ("trades", "decisions", "orders", "rejections")]
            require(trades.entry_origin.is_unique, "一个确认对应多个实际周期。")
            population = known.loc[known.price_after_volume_event & known.date.isin(decisions.origin)]
            statuses = {}
            for event in population.itertuples(index=False):
                selected = decisions.loc[decisions.origin.eq(event.date)]
                require(len(selected) == 1, "确认缺少唯一当时决定。")
                decision = selected.iloc[0]
                entered = trades.loc[trades.entry_origin.eq(event.date)]
                rejected = rejects.loc[rejects.origin.eq(event.date)] if len(rejects) else rejects
                row = event._asdict()
                row.update(period=period, cost=cost, planned_execution_date=decision.execution_date,
                           desired_shares=int(decision.desired_shares), shares_before=int(decision.shares_before),
                           decision_reason=decision.reason, known_es95=decision.known_es95, rejection_reason=None)
                if len(entered):
                    trade = entered.iloc[0]
                    status = str(trade.status)
                    row.update(cycle_id=int(trade.cycle_id), entry_date=trade.entry_date, exit_date=trade.exit_date,
                               actual_net_return=trade.net_return, actual_net_pnl=trade.net_pnl,
                               entry_quantity=int(trade.entry_quantity), actual_entry_raw=float(trade.entry_raw),
                               actual_buy_debit=float(trade.buy_debit))
                else:
                    if pd.isna(decision.execution_date):
                        status = "TERMINAL_NO_NEXT_SESSION_NO_EXECUTION"
                    elif int(decision.shares_before) > 0:
                        status = "ALREADY_HOLDING_OR_LOCKED_EXIT_NO_NEW_CYCLE"
                    elif len(rejected):
                        require(len(rejected) == 1, "同一确认有多个未解释拒绝。")
                        status = "CANCELLED_"+str(rejected.reason.iloc[0])
                        row["rejection_reason"] = str(rejected.reason.iloc[0])
                    else:
                        require(int(decision.desired_shares) == 0, "未执行正请求缺少解释。")
                        status = "NO_SIZE_OR_UNKNOWN_PRIOR_INFORMATION"
                    row.update(cycle_id=-1, entry_date=pd.NaT, exit_date=pd.NaT, actual_net_return=np.nan,
                               actual_net_pnl=np.nan, entry_quantity=0, actual_entry_raw=np.nan, actual_buy_debit=np.nan)
                require(0 <= event.lead_index < event.origin_index and event.reference_confirmation_index <= event.lead_index,
                        "没有已知参考或准备先于价格确认。")
                require(event.known_cash_close_ticks > event.reference_high_ticks and event.cumulative_signed_volume > event.reference_cumulative_volume,
                        "价格确认没有严格超过同锚价量。")
                row["execution_status"] = status
                statuses[status] = statuses.get(status, 0)+1
                qualifications.append(row)
            for trade in trades.itertuples(index=False):
                observed = known.loc[known.date.eq(trade.entry_origin)]
                require(len(observed) == 1 and bool(observed.price_after_volume_event.iloc[0]), "实际买入缺少当时价格确认。")
                event = observed.iloc[0]
                require(int(trade.fixed_volume_floor_ticks) == int(trade.entry_volume_floor_ticks) == int(event.fixed_low_ticks), "准备低点被改变。")
                require(int(trade.fixed_reference_high_ticks) == int(event.reference_high_ticks)
                        and int(trade.fixed_reference_volume) == int(event.reference_cumulative_volume), "持有参考价量被改变。")
                row = {**trade._asdict(), "period": period, "cost": cost,
                       **{"entry_known_"+key: value for key, value in event.to_dict().items()}}
                row["entry_known_initial_raw_floor"] = int(event.fixed_low_ticks)*.001-float(event.cash_shift)
                actual.append(row)
                sold = orders.loc[orders.cycle_id.eq(trade.cycle_id) & orders.side.eq("SELL")]
                if trade.status == "COMPLETE":
                    require(len(sold) > 0, "完成周期没有实际卖出。")
                    final = sold.iloc[-1]
                    at_exit = known.loc[known.date.eq(final.origin)].iloc[0]
                    require(final.date > trade.entry_date and final.origin < final.date, "退出违反次开或T+1。")
                    low_failed = bool(at_exit.current_quote_known and at_exit.known_cash_close_ticks <= trade.fixed_volume_floor_ticks)
                    joint_failed = bool(at_exit.current_quote_known and at_exit.cumulative_volume_known
                                        and at_exit.cumulative_signed_volume <= trade.fixed_reference_volume
                                        and at_exit.known_cash_close_ticks <= trade.fixed_reference_high_ticks)
                    if final.reason == PRIMARY+"_FIXED_LOW_FAILED_CLOSE":
                        require(low_failed, "低点退出没有已知固定低点失效。")
                    elif final.reason == PRIMARY+"_REFERENCE_VOLUME_AND_PRICE_FAILED_CLOSE":
                        require(joint_failed and not low_failed, "共同退出未满足两项或未按低点优先。")
                    exits.append({"period": period, "cost": cost, "cycle_id": int(trade.cycle_id),
                                  "entry_origin": trade.entry_origin, "exit_origin": final.origin, "exit_date": final.date,
                                  "exit_reason": final.reason, "fixed_low_failed": low_failed, "joint_reference_failed": joint_failed,
                                  "actual_net_return": trade.net_return, "actual_net_pnl": trade.net_pnl})
            counts.append({"period": period, "cost": cost, "confirmation_origins": len(population),
                           "actual_cycles": len(trades), "completed": int(trades.status.eq("COMPLETE").sum()),
                           "open": int(trades.status.ne("COMPLETE").sum()), "execution_status_counts": statuses})
            identities.append(capital_identity(trades, period, cost))
    qualifications, actual, exits = pd.DataFrame(qualifications), pd.DataFrame(actual), pd.DataFrame(exits)
    cases = pd.read_parquet(EXPLANATION / "results/四原案例逐日全部量价及量先恢复状态.parquet")
    case_actual = case_windows(actual, cases)
    table("全部确认资格_当时量价与真实执行状态", qualifications)
    table("全部实际进出点位_准备确认及真实资金", actual)
    table("四案例相交全部实际周期_不筛收益", case_actual)
    table("全部完成退出_固定低点或参考量价共同失效", exits)
    table("四场景完成周期真实资金恒等式", pd.DataFrame(identities))
    episodes = pd.read_parquet(ATLAS / "results/上涨段全集.parquet")
    transitions, snapshots = [], []
    for identifier in (18, 37, 42, 55):
        ep = episodes.loc[episodes.episode_id.eq(identifier)].iloc[0]
        local = known.loc[known.origin_index.between(ep.bottom_idx, ep.peak_idx)]
        for field in ("daily_hist", "daily_dif", "weekly_hist"):
            changed = known[field].gt(0) & known[field].shift(1).le(0)
            for day in local.loc[changed.loc[local.index]].itertuples(index=False):
                transitions.append({"original_episode_id": identifier, "field": field, "date": day.date,
                                    "value": getattr(day, field), "previous_complete_week_end": day.weekly_last_date,
                                    "not_a_new_signal_or_rule": True, "retrospective_episode_alignment_only": True})
        selected_dates = set(local.iloc[[0, -1]].date)
        selected_dates.update(row["date"] for row in transitions if row["original_episode_id"] == identifier)
        selected_dates.update(local.loc[local.volume_lead_event | local.price_after_volume_event, "date"])
        saved = local.loc[local.date.isin(selected_dates)].copy()
        saved["original_episode_id"], saved["retrospective_alignment_only"] = identifier, True
        snapshots.append(saved)
    table("四具体上涨日周指标全部跨零时钟_只解释", pd.DataFrame(transitions))
    table("四具体上涨底峰跨零准备确认原值_只解释", pd.concat(snapshots, ignore_index=True))
    primary = metrics.loc[metrics.policy.eq(PRIMARY)]
    passes = sum(bool(row["economic_gate_passed"]) for row in financial["comparisons"])
    quality = bool(primary.p_times_b.gt(1).all() and primary.standard_expectancy_loss_units.gt(0).all())
    accepted = bool(financial["all_four_economic_gates_passed"] and financial["historical_stability_gate_passed"])
    next_action = "本次全部准备、确认、实际占用/取消及低频长持原因已交付；不再为该失败用途增加指标、改确认/锚/退出或时间目标。下一金融实验尚未定义，先核对实质不同的已知信息及完整用途，或等待真正新样本/确认真实来源实现错误；没有通过准入卡和原失败核对前不运行新策略。原A/POINT真实前瞻1008实际交易日计划保持。"
    boundary = {"at": now(), "status": "FIXED_VOLUME_PRICE_SEQUENCE_CLOSED_NO_NEW_ADMITTED_FINANCIAL_CANDIDATE",
                "after_decisions": ["TECH.R187", "TECH.R188", "TECH.R189"], "fixed_policy_status": financial["status"],
                "next_action": next_action, "new_admitted_unrun_candidates": 0, "no_parameter_rescue": True,
                "project_wide_halt": False, "goal_achieved": False}
    write_json(OUT / "next_information_admission_boundary.json", boundary, exclusive=True)
    lines = ["# 具体上涨解释与量先价后完整用途的实际结果", "",
             "R187先解释量价及全部准备失败，R188登记唯一完整用途，R189的原四场景账户已经实际完成。该轮没有达到同时提高原A收益、夏普和交易机会的目标。", "",
             "## 四段上涨如何解释", "",
             "2019年1月至4月：1月8日日MACD柱转正，1月14日开始可用的上一完整周柱转正，1月18日DIF转正，是短动量、完整周背景、较慢趋势陆续恢复。全部跨零日期均保存，不能把底部或这些指标当成已验证进场。累计量和价格多数同步改善，固定的‘量先价后’准备没有识别1月启动；3月22日反而出现准备、25日量领先失效，没有价格确认。", "",
             "2020年3月至7月：4月1日日柱转正、4月23日DIF转正，但完整周柱到6月8日才转正；修复不是所有指标同时确认。3月27日及4月9/21/22日的量准备各自失效或被替换；5月28日量恢复、29日价格才确认。当时相对量分别0.929/0.833倍，日柱和上一完整周柱均仍负，所以不能靠‘全部转正、同时放量’解释这次确认。实际6月1日进场后按固定规则持有到2022年10月31日，并非7月最高点卖出。", "",
             "2024年9月至10月：9月23日收盘3.283、相对量0.736、RV比0.917，累计量已越过此前已确认高点日的量，而价格仍在参考高价下方；日柱刚转正，DIF和上一完整周柱仍负。9月24日收盘3.427、量3.338、RV比1.574，价格越过同一高点，形成收盘后确认；DIF到26日、完整周柱到30日才转正。这里可以在慢指标恢复前识别价格修复，但这不是证明机构主动吸筹或上涨因果。实际25日买入、截至2026年9月30日仍未完成，不能把后来最高涨幅或未完成回报计入胜率/B。", "",
             "2015年6月反弹：6月24日已经有量恢复准备，25日量领先消失，没有价格确认；29至30日短反弹时日DIF、日柱、上一完整周柱仍负，波动较高。本用途没有新增反弹进场。与此案例相交的真实持仓实际早在3月13日建立，不能把3月信号改写成6月反弹信号。", "",
             "[四图、全部当时数值和所有准备失败](../510300_volume_lead_price_confirm_explanation_v1/具体上涨与点位反推.md)。跨零只是指标时钟解释，不是新筛选条件；底峰只是事后对齐。", "",
             f"全部3488日、{explanation['all_reference_anchors']}高点锚、{explanation['all_volume_leads']}量先恢复、{explanation['all_price_confirmations']}全历史价格确认；2015起{explanation['volume_leads_since_2015']}准备/{explanation['price_confirmations_since_2015']}确认。"
             f"全部锚终态{explanation['all_anchor_terminal_counts']}。原49上涨中{explanation['waves_without_confirmation_bottom_to_peak']}段底峰间无确认、{explanation['waves_without_confirmation_confirm_to_peak']}段5%确认至峰间无确认。"
             "原61分段/2846标签/2826成熟/20删失/9尾部完整保留。", "",
             "## 实际净账户与点位质量", "",
             "20万元、252日、全部现金日、50%上限、原ES/跳空/DD预算、T+1/100份/.001刻度/股息应收及到账、两费用/最低佣金5均保持；原BASE买入仍通过原STRESS风险检查。B是完成周期平均盈利净回报除平均亏损净回报绝对值，pB与标准期望pB−q分别列，不以初始止损空间或未完成收益替代。", "",
             "| 时期 | 费用 | 净年化 | 净夏普 | 最大回撤 | 完成/开放 | 胜率 | B | pB | pB−q | 完整年均次数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary.itertuples(index=False):
        lines.append(f"| {r.period} | {r.cost} | {percent(r.net_cagr,4)} | {number(r.net_sharpe)} | {percent(r.max_drawdown)} | {r.completed_cycles}/{r.unfinished_cycles} | {percent(r.win_rate)} | {number(r.payoff)} | {number(r.p_times_b)} | {number(r.standard_expectancy_loss_units)} | {number(r.average_full_year_cycles,2)} |")
    lines.extend(["", "| 时期 | 费用 | 原控制 | 净年化 | 净夏普 | 完整年均次数 |", "|---|---|---|---:|---:|---:|"])
    for r in metrics.loc[metrics.policy.ne(PRIMARY)].itertuples(index=False):
        lines.append(f"| {r.period} | {r.cost} | {r.policy} | {percent(r.net_cagr,4)} | {number(r.net_sharpe)} | {number(r.average_full_year_cycles,2)} |")
    lines.extend(["", f"四单场景经济门通过{passes}，整体门={financial['all_four_economic_gates_passed']}，历史稳定门={financial['historical_stability_gate_passed']}；四场景实际pB均>1且标准期望正={quality}。20/252日配对区块各2000、种子固定，全部比较/尺度区间保存在summary.json，不选有利尺度。", "",
                  "近期两费用pB>1，但只有3个完成周期，其中1胜2负；早期7完成为1胜6负，pB<1。全日历年化均正，是按原实际资金和开放持仓计价得到的结果；不能据此把负的等周期点位标准期望改成正。四场景收益夏普均低于原A，完整年均次数1.4/0.5也低于原A4.4/4.1667，且长期持有占用了后来的新确认。", "",
                  "| 时期 | 费用 | 确认资格 | 实际周期 | 完成/开放 | 全部执行状态 |", "|---|---|---:|---:|---:|---|"])
    for r in counts:
        lines.append(f"| {r['period']} | {r['cost']} | {r['confirmation_origins']} | {r['actual_cycles']} | {r['completed']}/{r['open']} | {r['execution_status_counts']} |")
    lines.extend(["", "## 原四案例相交真实压力周期", "",
                  "全部相交周期保留，含窗口前已建立持仓；开放周期不超所属账户末日，不按未来底峰进出。", "",
                  "| 案例 | 原确认 | 实际进场 | 实际退出 | 完成净回报 | 净金额/元 | 状态 |", "|---|---|---|---|---:|---:|---|"])
    for r in case_actual.loc[case_actual.cost.eq("STRESS")].itertuples(index=False):
        exit_date = "截至所属账户末日未完成" if pd.isna(r.exit_date) else r.exit_date.strftime("%Y-%m-%d")
        lines.append(f"| {r.original_episode_id} | {r.entry_origin:%Y-%m-%d} | {r.entry_date:%Y-%m-%d} | {exit_date} | {percent(r.net_return)} | {number(r.net_pnl,2)} | {r.status} |")
    lines.extend(["", "2019固定窗口没有相交实际周期。2020的原上涨识别能成立，但原完整退出到2022才成立，不能按2020最高点包装盈亏比；2024的开放周期不能列成已获利完成交易。", "",
                  f"全部{len(qualifications)}资格行、{len(actual)}实际周期/{len(exits)}完成退出、四资金恒等式已经交付，未新增账户。6输入/6账户必要测试首轮通过、4真实前缀、8原控制精确复现、4新主账户一次测量和保存核对通过。", "",
                  "## 裁决与下一步", "",
                  ("接受此完整用途的历史开发增量，但不替换原策略；仍缺独立证据。" if accepted else "拒绝此固定完整用途作为提高原收益夏普的策略，并关闭。保留2024提前一日准备/次日价格确认和2020中段修复的开发事实；不能据好案例改时间退出、抬低点、加量比/MACD/RV筛选或混A营救。"), "",
                  "现有历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，收益夏普整体目标及去过拟合未完成。", "", next_action, "",
                  "- [全部真实点位及已知资料](results/全部实际进出点位_准备确认及真实资金.csv)",
                  "- [全部确认资格及执行](results/全部确认资格_当时量价与真实执行状态.csv)",
                  "- [四段日周指标全部跨零时钟](results/四具体上涨日周指标全部跨零时钟_只解释.csv)",
                  "- [十二账户比较](results/完整账户共同口径比较.csv)、[逐年收益与次数](results/逐年净收益与实际周期次数.csv)",
                  "- [冻结](protocol.json)、[实际结果及全部区间](summary.json)、[保存核对](saved_result_verification.json)", ""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    paths = [Path(__file__), report, OUT / "summary.json", OUT / "protocol.json", OUT / "saved_result_verification.json",
             OUT / "next_information_admission_boundary.json", OUT / "results/全部实际进出点位_准备确认及真实资金.parquet",
             OUT / "results/全部确认资格_当时量价与真实执行状态.parquet", OUT / "results/四场景完成周期真实资金恒等式.parquet"]
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_SAVED_VOLUME_PRICE_POINTS_AND_CONCRETE_PHASES_DELIVERED",
        "all_qualification_rows": len(qualifications), "all_actual_cycle_rows": len(actual), "complete_exit_rows": len(exits),
        "case_cycle_rows_two_costs": len(case_actual), "scenario_counts": counts, "capital_identity_cells": len(identities),
        "max_capital_identity_error": max(abs(r["identity_error_cny"]) for r in identities),
        "individual_economic_gate_passes": passes, "all_four_actual_point_quality_passed": quality,
        "full_policy_accepted": accepted, "new_admitted_unrun_candidates": 0, "new_accounts_in_delivery": 0,
        "next_action": next_action, "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print(f"R189全部{len(qualifications)}资格、{len(actual)}实际周期、{len(exits)}完成退出及四段动量时钟已交付；4场景经济门通过{passes}。", flush=True)


if __name__ == "__main__":
    run()
