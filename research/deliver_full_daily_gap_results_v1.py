"""交付全部日线完整缺口资格、真实周期、原四案例和扣费后的全账户结果。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.full_daily_gap_study_v1 import OUT, EXPLANATION, PRIMARY, PERIODS, COSTS, table
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.deliver_downtrend_break_results_v1 import case_windows, capital_identity, percent, number


def run():
    require(not (OUT / "delivery_receipt.json").exists(), "全体缺口结果已经交付，不重复。")
    financial = read(OUT / "summary.json")
    explanation = read(EXPLANATION / "summary.json")
    require(financial["technical_decision"] == "TECH.R177" and financial["new_primary_accounts"] == 4,
            "四缺口账户未完成。")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES",
            "四保存账户尚未核对。")
    known = pd.read_parquet(EXPLANATION / "results/全部3488已知缺口与量价_前2015A未知保留.parquet")
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    qualifications, actual, scenario_counts, identities, exit_rows = [], [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            trades = pd.read_parquet(folder / "trades.parquet")
            decisions = pd.read_parquet(folder / "decisions.parquet")
            orders = pd.read_parquet(folder / "orders.parquet")
            rejects = pd.read_parquet(folder / "rejections.parquet")
            population = known.loc[known.gap_event & known.date.isin(decisions.origin)]
            require(trades.entry_origin.is_unique, "一个缺口出生对应多个新周期。")
            counts = {}
            for event in population.itertuples(index=False):
                selected = decisions.loc[decisions.origin.eq(event.date)]
                require(len(selected) == 1, "完整缺口没有唯一原点决定。")
                decision = selected.iloc[0]
                trade = trades.loc[trades.entry_origin.eq(event.date)]
                rejected = rejects.loc[rejects.origin.eq(event.date)] if len(rejects) else pd.DataFrame()
                row = event._asdict()
                row.update(period=period, cost=cost, desired_shares=int(decision.desired_shares),
                           shares_before=int(decision.shares_before), decision_reason=decision.reason,
                           source_weight=decision.source_weight, known_es95=decision.known_es95,
                           planned_execution_date=decision.execution_date,
                           known_current_gap_floor_ticks=int(decision.known_current_gap_floor_ticks),
                           A_inventory_role="SEPARATE_ORIGINAL_A_SAVED_ACCOUNT_NOT_GAP_ACCOUNT_INVENTORY",
                           rejection_reason=None)
                row["known_initial_raw_gap_floor_at_origin"] = int(event.gap_lower_ticks)*.001-float(event.cash_shift)
                if len(trade):
                    require(len(trade) == 1, "同原点周期不唯一。")
                    t = trade.iloc[0]
                    status = str(t.status)
                    row.update(cycle_id=int(t.cycle_id), entry_date=t.entry_date, exit_date=t.exit_date,
                               actual_net_return=t.net_return, actual_net_pnl=t.net_pnl,
                               entry_quantity=int(t.entry_quantity), actual_entry_raw=float(t.entry_raw),
                               actual_entry_fill=float(t.entry_price), actual_buy_debit=float(t.buy_debit))
                else:
                    if pd.isna(decision.execution_date):
                        status = "NO_NEXT_OPEN_WITHIN_OWN_ACCOUNT_PERIOD"
                    elif int(decision.shares_before) > 0:
                        status = "ALREADY_HOLDING_NO_NEW_CYCLE"
                    elif len(rejected):
                        require(len(rejected) == 1, "未成交原点有多个开盘拒绝。")
                        status = "CANCELLED_" + str(rejected.reason.iloc[0])
                        row["rejection_reason"] = str(rejected.reason.iloc[0])
                    else:
                        require(int(decision.desired_shares) == 0, "未成交请求没有资金或信息解释。")
                        status = "NO_SIZE_OR_UNKNOWN_PRIOR_INFORMATION"
                    row.update(cycle_id=-1, entry_date=pd.NaT, exit_date=pd.NaT,
                               actual_net_return=np.nan, actual_net_pnl=np.nan, entry_quantity=0,
                               actual_entry_raw=np.nan, actual_entry_fill=np.nan, actual_buy_debit=np.nan)
                row["execution_status"] = status
                counts[status] = counts.get(status, 0)+1
                require(int(event.gap_lower_reference_index) == int(event.origin_index)-1,
                        "缺口下沿不来自相邻上一完整日。")
                require(int(event.known_cash_low_ticks) > int(event.gap_lower_ticks), "出生时整日区间已经接触。")
                qualifications.append(row)
            require(sum(counts.values()) == len(population), "全部资格状态未还原。")
            for trade in trades.itertuples(index=False):
                observed = known.loc[known.date.eq(trade.entry_origin)]
                require(len(observed) == 1 and bool(observed.gap_event.iloc[0]), "真实买入没有已知完整缺口。")
                row = trade._asdict()
                row.update(period=period, cost=cost)
                row.update({"entry_known_"+key: value for key, value in observed.iloc[0].to_dict().items()})
                row["entry_known_initial_raw_gap_floor_at_origin"] = int(observed.gap_lower_ticks.iloc[0])*.001-float(observed.cash_shift.iloc[0])
                sold = orders.loc[orders.cycle_id.eq(trade.cycle_id) & orders.side.eq("SELL")]
                final = sold.iloc[-1] if trade.status == "COMPLETE" and len(sold) else None
                row.update(final_exit_raw=final.raw_open if final is not None else np.nan,
                           final_exit_fill=final.fill_price if final is not None else np.nan)
                actual.append(row)
                if final is not None:
                    decision = decisions.loc[decisions.origin.eq(final.origin)].iloc[0]
                    event = known.loc[known.date.eq(final.origin)].iloc[0]
                    floor = int(decision.known_current_gap_floor_ticks)
                    require(final.date > trade.entry_date and final.origin < final.date, "退出违反次开或T+1。")
                    if final.reason == "FULL_DAILY_UP_GAP_FILLED_DAILY_LOW":
                        require(bool(event.current_quote_known) and int(event.known_cash_low_ticks) <= floor,
                                "回补退出没有已知日低点触及。")
                    exit_rows.append({"period":period,"cost":cost,"cycle_id":int(trade.cycle_id),
                                      "entry_origin":trade.entry_origin,"exit_origin":final.origin,
                                      "exit_date":final.date,"exit_reason":final.reason,
                                      "gap_floor_ticks":floor,"exit_day_low_ticks":int(event.known_cash_low_ticks),
                                      "exit_day_close_ticks":int(event.known_cash_close_ticks),
                                      "close_still_above_floor":bool(event.current_quote_known and int(event.known_cash_close_ticks)>floor),
                                      "actual_net_return":trade.net_return,"actual_net_pnl":trade.net_pnl,
                                      "post_result_description_only":True})
            scenario_counts.append({"period":period,"cost":cost,"full_gap_origins":len(population),
                                    "actual_cycles":len(trades),"completed":int(trades.status.eq("COMPLETE").sum()),
                                    "open":int(trades.status.ne("COMPLETE").sum()),"execution_status_counts":counts})
            identities.append(capital_identity(trades, period, cost))
    qualifications, actual, exits = pd.DataFrame(qualifications), pd.DataFrame(actual), pd.DataFrame(exit_rows)
    cases = pd.read_parquet(EXPLANATION / "results/四案例逐日完整量价及整日缺口.parquet")
    case_actual = case_windows(actual, cases)
    mapped = pd.read_parquet(EXPLANATION / "results/原61分段及49正式波段_完整日线缺口覆盖.parquet")
    table("全部完整缺口资格_原点量价A覆盖及真实执行状态", qualifications)
    table("全部实际进出点位_已知缺口量价及真实资金", actual)
    table("四案例相交的全部实际周期_不筛收益", case_actual)
    table("全部完成退出_日低点回补和收盘区别", exits)
    table("全部四场景完成周期_实际资金恒等式", pd.DataFrame(identities))
    primary = metrics.loc[metrics.policy.eq(PRIMARY)]
    economic_passes = sum(bool(row["economic_gate_passed"]) for row in financial["comparisons"])
    point_quality = bool(primary.p_times_b.gt(1).all() & primary.standard_expectancy_loss_units.gt(0).all())
    boundary = {
        "at":now(),"status":"FIXED_FULL_GAP_MEASUREMENT_COMPLETED_NO_NEW_ADMITTED_FINANCIAL_CANDIDATE",
        "after_decisions":["TECH.R175","TECH.R176","TECH.R177"],
        "accepted_fact":"整日区间分离及低点回补可在部分修复早期到达；全体实际进出及失败已记录，不能用单次上涨解释提升完整账户。",
        "closed_policy":financial["status"],"no_parameter_rescue":True,
        "not_authorized_as_new_policy":["按本结果加量/MACD/RV过滤","改缺口大小/回补窗口","改预算/费用/时期","混A或拼接旧突破退出"],
        "next_required_input":"先归因全部已保存周期的出生、后续缺口抬高下沿、日低点回补和次开成交时钟，分清早期修复、上涨中途和晚期追入；这种诊断不直接形成新过滤或策略。之后需要实质不同信息及完整用途、真正新样本或真实实现错误，才能再次准入。",
        "fixed_diagnostic_population":"全部97出生、四真实账户、194资格和所有完成/开放/取消/已有持仓；原61/49段、旧标签及失败保留。",
        "next_new_account_strategy":"NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "new_admitted_unrun_candidates":0,"new_accounts_in_next_saved_diagnostic":0,
        "independent_validation":"NOT_ESTABLISHED","goal_achieved":False,
        "scope_limit":"本规则关闭不是全项目停止，原真实前瞻未变。历史先后解释不是独立新样本或真实订单流。",
    }
    write_json(OUT / "next_all_gap_lifecycle_diagnostic_proposal.json", boundary, exclusive=True)
    lines = ["# 完整日线缺口：具体上涨、当时量价及真实账户结果","",
             "本轮按先解释具体上涨与失败、再反推当时可知点位的顺序完成TECH.R175解释、R176登记和R177唯一完整账户检验。全部3488日、2015起97完整向上缺口、原61分段/49正式波段及四案例保留，向下91缺口只观察。指标描述价格、成交活跃度和波动，不能据此推断真实主动买卖流或造成上涨的因果。","",
             "固定点位：当前完整日线现金平移low严格大于上一完整日high，收盘后才可确认，次真实开一次买入；次开已到或低于缺口下沿取消。初始下沿为上一日high，持有中新的完整缺口只将下沿提高，已知日low触及下沿即回补、次合法开退出，收盘在线上也不撤销。未知自身不退出、原风险只减；无加仓、A混合、2R、20日或期末人工清仓。整数.001刻度避免除息相等边界被浮点误判。","",
             "## 具体上涨及指标顺序","",
             "2019-01-09完整缺口出现在春季上涨较早阶段：相对前20日成交量中位数1.400，日MACD柱+0.013402，而日DIF−0.038431、上一完整周柱−0.007990仍负，RV20/前252日中位数0.821。它描述价格区间已向上分离、短期动量修复先到而长周期指标滞后；不是全部指标同向才有点位。该日原A库存和目标均0。实际压力01-10入、03-11出净+18.51%；随后04-01另一个缺口04-02入、04-29出净−2.35%，失败同样保留。","",
             "2020-03-25完整缺口到达修复初期：相对量1.666，日DIF−0.113406、日柱−0.063373及上一完整周柱−0.077174均负，RV比2.131，属于高波动修复。实际压力03-26入、04-22出净+2.25%，只覆盖部分修复。05-26、06-01、06-24及07-06后续缺口分别产生净负周期，不能把波段后来整体大涨算成这些入点均盈利。06-01事件次06-02入、06-12出净−0.43%；07-06放量约2.962并日周动量同向，但次07-07追入至08-21出仍净−1.35%。","",
             "2024-09-25才形成首个完整向上缺口，比09-24上涨发生晚一日。相对量2.654、日柱+0.046265而DIF−0.010426、上一周柱−0.053959仍负，RV比1.619；09-27后续缺口伴随日动量转正和波动扩张。实际压力09-26入、2025-03-11出净+14.83%，不能按事后10月最高点卖出。09-25原A已有21800份及已知目标31.92%，该信息不等于原A完全漏掉此段。","",
             "2015原06-29/30单日反弹没有该点位；固定案例窗后来07-10形成完整缺口，此时日DIF/日柱/上一完整周柱均负，RV比4.033、相对量0.916。压力07-13入、07-28出净−10.28%。缺口在剧烈波动中同样会失败，不能只保留2019和2024赢家。原固定波段仅用于事后对齐，不是策略知道的底和顶。","",
             f"全部{explanation['completed_gap_fill_paths']}条出生路径中，{explanation['known_filled_but_close_above_floor']}条在日低点回补时收盘仍在线上；本规则的失效时钟用当时日低点。出生路径描述和实际账户周期分别报告，持仓中的新缺口抬线不等于另开一笔交易。","",
             "各初始下沿按事件日原报价坐标换回：2019-01-09为3.110元，2020-03-25为3.623元，2024-09-25为3.429元。它们是当时上一完整日高点的现金平移失效线，之后仅按固定后续完整缺口提高及当时除息转换，不能回看最高点选择卖价；实际入场原开盘与成交价格见全周期表。","",
             "## 全部实际结果及接受裁决","",
             "20万元、252日、所有现金日、真实次开、T+1/100份/.001、两费用/最低佣金5元及原ES/跳空/DD预算不变。下表B为实际完成盈利周期平均净收益率除以亏损周期平均损失幅度；pB与标准期望pB−q分别报告，计划失效距离不代替实际B。","",
             "| 时期 | 费用 | 净年化 | 净夏普 | 最大回撤 | 完成/开放 | 胜率 | B | pB | pB−q | 完整年均完成次数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary.itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {percent(row.net_cagr,4)} | {number(row.net_sharpe)} | {percent(row.max_drawdown)} | {row.completed_cycles}/{row.unfinished_cycles} | {percent(row.win_rate)} | {number(row.payoff)} | {number(row.p_times_b)} | {number(row.standard_expectancy_loss_units)} | {number(row.average_full_year_cycles,2)} |")
    lines.extend(["",f"四个场景中{economic_passes}个通过共同经济门，整体经济门={financial['all_four_economic_gates_passed']}、历史稳定门={financial['historical_stability_gate_passed']}。较早两费用pB低于1；近期两费用pB虽超过1、收益夏普较R173提高，但仍显著低于原A，不接受为完整模型改进。以下八原控制均保持原结果，不混用10万元/242日旧口径。","",
                  "| 时期 | 费用 | 比较基准 | 原净年化 | 原净夏普 | 原完整年均次数 |","|---|---|---|---:|---:|---:|"])
    for row in metrics.loc[metrics.policy.ne(PRIMARY)].itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {row.policy} | {percent(row.net_cagr,4)} | {number(row.net_sharpe)} | {number(row.average_full_year_cycles,2)} |")
    lines.extend(["","| 时期 | 费用 | 缺口资格 | 实际周期 | 完成/开放 | 全部执行状态 |","|---|---|---:|---:|---:|---|"])
    for row in scenario_counts:
        lines.append(f"| {row['period']} | {row['cost']} | {row['full_gap_origins']} | {row['actual_cycles']} | {row['completed']}/{row['open']} | {row['execution_status_counts']} |")
    lines.extend(["",f"全部{len(qualifications)}资格行和{len(actual)}实际周期行分别记录成交、开放、取消及已有持仓。原首日前一收盘也被核对，本次2014-12-31和2019-12-31均没有缺口；期末没有次开时不虚构成交或完成收益。开放周期不进入完成胜率，案例观察只到所属账户末日。原2846旧标签/2826成熟/20删失和9尾部不变，旧20日标签pB约0.758不是动态回补账户的真实交易质量。","",
                  "全体实际完成利润的恒等拆解ΣQr=ΣQ×平均r+Σ[(Q−平均Q)(r−平均r)]，Q为真实买入支出。两项只共同还原保存账本，不能把均值项称为等额配仓策略。","",
                  "| 时期 | 费用 | 均值项/元 | 资金收益对应项/元 | 实际完成净利润/元 | 最大实际盈利原点及金额 |","|---|---|---:|---:|---:|---|"])
    for row in identities:
        lines.append(f"| {row['period']} | {row['cost']} | {row['mean_return_component_cny']:.2f} | {row['allocation_covariance_component_cny']:.2f} | {row['actual_completed_pnl_cny']:.2f} | {row['largest_actual_winner_origin']:%Y-%m-%d} / {row['largest_actual_winner_net_pnl']:.2f}元 |")
    lines.extend(["","## 验证、限制与下一步","",
                  "四输入和四必要账户行为测试通过，四真实前缀一致，八共同A/纯价格日账/订单/周期精确复现，四新主账户一次运行并通过保存指标/资金/库存/T+1核对。共同对照首次发现市场日期ms与冻结输入ns不同，经济日期完全相同，仅新金融接口转ns后恢复原账户分辨率，旧源码和原缺口输入未改。首次失败保留；首次备份因异步复制实际捕获修正后源码，时序纠正和明确标注的重建副本单独保留，不能称首次源码快照。修正时新主金融运行/结果读取0，未据收益改参。","",
                  "有限旧用途核对区分十五分钟三柱不平衡、一分钟扫低FVG、旧RR3开盘取消和R173收盘突破；原终态保持，不声称全项目所有缺口表达都未试。日线极值和当日成交量是价格量的观察，缺少逐笔主动买卖及Level2，不能叫真实订单流验证。","",
                  "全部历史DEVELOPMENT_CALIBRATION，来源first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED，global DSR/PBO NOT_COMPUTED。提前固定完整用途和未来不改写过去的检查不能认证已去过拟合。全四场景经济及稳定门未过，R177固定政策拒绝关闭，原A、R173及其他旧终态和真实前瞻保持。目标仍active、未达到；没有实盘或期权收益。","",
                  "下一具体实验只对全部已保存周期作出生、后续缺口抬下沿、日低点回补及次开成交时钟的归因，解释2019较早修复、2020反复中断和2024持续持有的区别。全97出生和四账户/所有失败/开放保留，不挑赢家形成MACD/量/RV过滤，不改回补线、费用或预算救回本规则。该诊断不跑新账户；之后只有实质不同信息与完整用途、真正新样本或真实实现错误才进一步准入，已准入待跑新金融候选0。原A/POINT真实前瞻十二值与1008实际交易日计划保持。","",
                  "## 文件入口","",
                  "- [全体已知缺口、指标和执行状态](results/全部完整缺口资格_原点量价A覆盖及真实执行状态.csv)",
                  "- [全部真实进出点位与资金](results/全部实际进出点位_已知缺口量价及真实资金.csv)",
                  "- [四例相交的全部实际周期](results/四案例相交的全部实际周期_不筛收益.csv)",
                  "- [全部完成退出的低点/收盘区别](results/全部完成退出_日低点回补和收盘区别.csv)",
                  "- [十二共同账户结果](results/完整账户共同口径比较.csv)、[全部逐年收益及次数](results/逐年净收益与实际周期次数.csv)",
                  "- [四资金恒等式](results/全部四场景完成周期_实际资金恒等式.csv)",
                  "- [R175完整解释](../510300_full_daily_gap_explanation_v1/summary.json)、[R176登记](protocol.json)、[R177全结果及区间](summary.json)",
                  "- [必要测试](tests_receipt.json)、[共同对照](control_preflight.json)、[保存核对](saved_result_verification.json)",
                  "- [首次共同对照失败](initial_control_preflight_failure.json)、[备份时序纠正](initial_archive_timing_correction.json)",
                  "- [下一全体时钟归因](next_all_gap_lifecycle_diagnostic_proposal.json)",""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at":now(),"status":"PASS_ALL_FULL_GAP_POINTS_CASES_EXITS_AND_SAVED_IDENTITIES_DELIVERED",
        "technical_decision":"TECH.R177","scenario_counts":scenario_counts,
        "all_qualification_rows":len(qualifications),"all_actual_cycle_rows":len(actual),
        "all_completed_cycles":int(actual.status.eq("COMPLETE").sum()),"all_open_cycles":int(actual.status.ne("COMPLETE").sum()),
        "all_four_actual_point_quality_passed":point_quality,"individual_economic_gate_passes":economic_passes,
        "overall_economic_gate_passed":financial["all_four_economic_gates_passed"],
        "historical_stability_gate_passed":financial["historical_stability_gate_passed"],
        "actual_complete_exit_rows":len(exits),"actual_low_fill_exit_close_above_floor":int((exits.exit_reason.eq("FULL_DAILY_UP_GAP_FILLED_DAILY_LOW") & exits.close_still_above_floor).sum()),
        "capital_identity_cells":len(identities),"original_episodes_preserved":len(mapped),
        "original_admitted_waves":int(mapped.admitted.sum()),"case_cycle_rows":len(case_actual),
        "open_case_windows_clipped_at_own_account_end":True,"new_accounts_in_delivery":0,
        "new_fits":0,"new_training_labels":0,"new_market_requests":0,"goal_achieved":False,
        "sources":[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in
                   [Path(__file__),ROOT/"research/deliver_downtrend_break_results_v1.py",OUT/"summary.json",
                    OUT/"saved_result_verification.json",EXPLANATION/"summary.json",report]],
    }, exclusive=True)
    print(f"全{len(qualifications)}资格行、{len(actual)}实际周期行、四例及全体退出已交付；未重跑账户。", flush=True)


if __name__ == "__main__":
    run()
