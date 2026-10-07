"""一次交付R182全体位置与财富归因，保留金融R181及全部原前瞻。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.full_range_reversal_lifecycle_attribution_v1 import OUT, FINANCIAL, EXPLANATION, STATE, PERIODS
from research.finalize_downtrend_break_state_v1 import FORWARD, DOCS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def run():
    require(not (OUT / "delivery_receipt.json").exists() and not (OUT / "project_state_update_receipt.json").exists(),
            "R182交付及长期事实已经写入，不重复。")
    summary, protocol = read(OUT / "summary.json"), read(OUT / "protocol.json")
    require(summary["status"] == "COMPLETED_ALL_SAVED_REVERSAL_LIFECYCLE_COVERAGE_AND_EXIT_WEALTH_IDENTITIES",
            "全体诊断尚未完成。")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结诊断输入改变。")
    cycles = pd.read_parquet(OUT / "results/全部142周期_已知抬线失效及真实资金时钟.parquet")
    cases = pd.read_parquet(OUT / "results/四原案例全部相交压力周期_真实时钟与资金.parquet")
    coverage = pd.read_parquet(OUT / "results/原61分段及49正式上涨_两种原区间覆盖不筛选.parquet")
    original = pd.read_parquet(FINANCIAL / "results/全部实际进出点位_已知整日反转量价及真实资金.parquet")
    grouped = pd.read_parquet(OUT / "results/全部六组两时期两费用_最高账面与最终结果仅解释.parquet")
    require(len(cycles) == 142 and len(cases) == 7 and len(grouped) == 24 and int(coverage.admitted.sum()) == 49,
            "全体保存数量与原总体不同。")
    for row in cycles.itertuples(index=False):
        reference = original.loc[original.period.eq(row.period) & original.cost.eq(row.cost) & original.cycle_id.eq(row.cycle_id)]
        require(len(reference) == 1, "诊断没有唯一原金融周期。")
        expected = reference.iloc[0]
        for key in ("entry_origin", "entry_date", "entry_quantity", "status"):
            require(getattr(row, key) == expected[key], "诊断改变周期身份或库存。")
        if row.status == "COMPLETE":
            require(abs(row.actual_net_return-expected.net_return) <= 1e-12 and abs(row.actual_net_pnl-expected.net_pnl) <= 1e-8,
                    "诊断改变原实际收益。")
        else:
            require(pd.isna(row.actual_net_return) and pd.isna(row.actual_net_pnl), "开放周期被填完成收益。")
        require(row.cycle_observed_until <= pd.Timestamp(PERIODS[row.period][1]), "案例延伸到另一账户时期。")
    for scenario in summary["all_scenario_diagnostics"]:
        selected = cycles.loc[cycles.period.eq(scenario["period"]) & cycles.cost.eq(scenario["cost"]) & cycles.status.eq("COMPLETE")]
        reconstructed = float((selected.exit_origin_marked_pnl_cny+selected.final_open_gap_gross_cny+
                               selected.dividend_accrual_after_final_origin_cny-selected.final_sell_commission-selected.final_sell_slippage).sum())
        require(abs(reconstructed-selected.actual_net_pnl.sum()) <= 1e-6, "全部完成财富身份不符。")
    early = next(row for row in summary["all_scenario_diagnostics"] if row["period"] == "2015_2019" and row["cost"] == "STRESS")
    recent = next(row for row in summary["all_scenario_diagnostics"] if row["period"] == "2020_2026" and row["cost"] == "STRESS")
    metrics = pd.read_parquet(FINANCIAL / "results/完整账户共同口径比较.parquet")
    old_a = metrics.loc[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS") & metrics.policy.eq("A_SAVED_WEIGHT")].iloc[0]
    a_gap = float(old_a.completed_cycle_net_pnl-recent["completed_pnl_cny"])
    last_clock_loss = float(-recent["final_open_gap_gross_cny"]+recent["final_exit_friction_cny"])
    pressure = cycles.loc[cycles.cost.eq("STRESS") & cycles.status.eq("COMPLETE")]
    nonpositive_losses = int(pressure.final_result_group.eq("NONPOSITIVE_HOLDING_CLOSE_FINAL_LOSS").sum())
    next_action = (
        "本次全部保存归因完成，不再分组调参。下一先提出实质不同、当时可观察的日线点位信息与完整用途卡，"
        "明确对象/可知时钟/原失败区别/全体反例，再核旧实际结果决定金融准入；也可是真正新样本或来源/实现错误。"
        "保留急涨无反转、下跌中反转及最大赢家集中等事实，不把未来底峰、最高财富或后来赢家变成交易输入。")
    boundary = {
        "at": now(), "status": "ALL_FIXED_RANGE_REVERSAL_LIFECYCLE_DIAGNOSTICS_COMPLETE_NO_NEW_FINANCIAL_CANDIDATE",
        "latest_diagnostic": "TECH.R182", "latest_actual_financial": "TECH.R181",
        "accepted_new_facts": {
            "all_cycles": 142, "holding_decisions": 1884, "exit_wealth_identities": 140,
            "admitted_waves_without_bottom_to_peak_birth": 21, "admitted_waves_without_confirmation_to_peak_birth": 26,
            "pressure_ever_positive_then_loss": 28, "pressure_never_positive_then_loss": nonpositive_losses,
            "early_largest_winner_share_of_net_pnl": early["largest_positive_share_of_net_pnl"],
            "recent_final_exit_clock_difference_cny": last_clock_loss, "recent_actual_difference_to_A_cny": a_gap,
        },
        "finite_rejected_explanations": ["单日完整反转是所有上涨的必要启动", "曾盈利说明已有可稳定实现的止盈点", "最后卖出时钟差解释全部对A差距", "早期单项夏普提高足以接受跨期模型"],
        "next_required_input": next_action, "closed_current_policy": "R181固定完整反转金融拒绝保持",
        "no_additional_same_diagnostic_split_planned": True, "new_admitted_unrun_candidates": 0,
        "next_new_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "new_accounts": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "scope_limit": "本用途及固定诊断结束不等于全项目暂停；原A/POINT真实前瞻保持。",
    }
    write_json(OUT / "next_information_admission_boundary.json", boundary, exclusive=True)
    lines = ["# 整日反转：全体上涨覆盖、真实资金及退出时钟", "",
             "TECH.R182只读R181保存结果，完成全部98出生、142实际周期、1884持有决定、2024逐日财富记录与140完成退出身份。四原案例全部7个压力周期、原61分段/49正式上涨及全部失败/开放保留；0新账户/拟合/训练标签/采集。", "",
             "## 具体上涨与失败", "",
             "2019-01-04价格局部转强而日周MACD仍负；实际01-07入、01-17相反整日形态、01-18出，持有最高真实收盘账面+936.69元，最后+768.45元。它先于后来春季上涨，但相反形态使本周期提前结束，不能把之后总涨幅给它。02-22日周动量已正时再次入点，02-25入场后04-03、04-19两个同向事件提高低点线；04-22收盘跌线，23日出，最高收盘账面6468.25元、最终4996.89元。", "",
             "2020-03-10量1.983倍、日柱近零、上一周仍负的同向反转后，实际持有各收盘均未盈利，03-13跌线、16日出，最终−3934.13元。05-06同向反转的周期曾真实收盘盈利462.55元，05-22跌线时已经账面−1650.25元，25日出−1779.80元。原7月急涨没有新的同向形态，不能由这些早期点位自动得到后段利润。", "",
             "2024九月急涨没有该形态；10-18形态的日周动量均正而RV比4.907，实际10-21入。11-07新同向形态抬线，11-15跌线、18日出；最高实际收盘账面413.70元，最终62.90元。它位于急涨后的整理，不是九月启动点；真实投入和已减份额的财富均保持，不能用较大假想资金替代。", "",
             "2015固定案例窗含06-04出生的原已有周期及07-09出生的后续周期。前者06-15相反形态退出，曾收盘+1812.45元、最终−1257.61元；后者07-10入、08-18相反形态、19日出，曾收盘+3597.02元、最终−1231.28元，两周期都没提高初始事件低点线。7月形态属于原另一分段，与案例18固定窗相交，不能混称同一上涨段。", "",
             "| 原案例 | 出生 | 失效原点 | 实际退出 | 抬线次数 | 最高真实收盘账面/元 | 退出原点账面/元 | 次开价差/元 | 最终净利润/元 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in cases.itertuples(index=False):
        lines.append(f"| {row.original_episode_id} | {row.entry_origin:%Y-%m-%d} | {row.first_known_exit_signal_origin:%Y-%m-%d} | {row.exit_date:%Y-%m-%d} | {row.floor_raise_count} | {row.best_observed_holding_close_pnl_cny:.2f} | {row.exit_origin_marked_pnl_cny:.2f} | {row.final_open_gap_gross_cny:.2f} | {row.actual_net_pnl:.2f} |")
    lines.extend(["", "最高真实收盘账面损益包括已发生买卖摩擦、真实减仓及当时股息应收，是事后路径值，不是能提前知道的最高价卖点。事件低点线仅在整数现金坐标提高；除息后换回原报价线可下降而经济线没有下降。", "",
                  "## 全体覆盖与利润来源", "",
                  "49段正式上涨中21段在原底至峰全区间没有同向反转，26段在原确认至峰区间没有。两区间均沿原图谱，不择有利定义；底和峰是事后标签，不能作为事前过滤。98出生中72与原上升段相交，26没有相交，未知保留。原61分段中的排除段保持排除，样本范围外不补零。", "",
                  f"压力70完成中24盈利/46亏损；52曾在真实持有收盘盈利，其中28最终亏损；另{nonpositive_losses}个持有收盘从未盈利、最终均亏损。早期12盈利/12曾盈利最终亏/5从未盈利亏，近期12/16/13。BASE曾盈利最终亏分别13/17，同样报告；六事后组×四场景24单元全保留，不能将后来分组作为入场或止盈门。", "",
                  "| 时期 | 费用 | 完成/开放 | 曾盈利最终亏损 | 收盘线/相反退出 | 只跌抬高线 | 次开价差/元 | 最后卖出摩擦/元 | 全部完成净利润/元 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary["all_scenario_diagnostics"]:
        lines.append(f"| {row['period']} | {row['cost']} | {row['completed']}/{row['open']} | {row['completed_positive_close_then_loss']} | {row['close_floor_exits']}/{row['opposing_reversal_exits']} | {row['exit_on_raised_floor_only']} | {row['final_open_gap_gross_cny']:.2f} | {row['final_exit_friction_cny']:.2f} | {row['completed_pnl_cny']:.2f} |")
    lines.extend(["", "140条完成退出的净利润−最后卖出前原点财富 = 原点剩余份额×(真实卖出日原开盘−原点收盘)+其后该周期股息应收−最后卖出佣金−滑点。全部逐笔和四场景合计还原，最大残差小于2e−11元；本次后续股息项为0，但原登记权利及应收已逐笔计入。", "",
                  f"近期STRESS最后次开价差{recent['final_open_gap_gross_cny']:.2f}元、最后卖出摩擦{recent['final_exit_friction_cny']:.2f}元，净时钟差{last_clock_loss:.2f}元；原A完成净利润{old_a.completed_cycle_net_pnl:.2f}元与反转账户{recent['completed_pnl_cny']:.2f}元差{a_gap:.2f}元。原最终卖出的时钟差不足以单独解释全部差距。身份用已有实际数量，并非收盘清仓策略或任意新退出的收益上界。", "",
                  "| 时期 | 费用 | 最大赢家出生 | 该周期净利润/元 | 占全部正利润 | 占完成净利润 |",
                  "|---|---|---|---:|---:|---:|"])
    for row in summary["all_scenario_diagnostics"]:
        net_share = row["largest_positive_share_of_net_pnl"]
        net_label = "未知（净利润非正）" if net_share is None else f"{net_share:.2%}"
        lines.append(f"| {row['period']} | {row['cost']} | {pd.Timestamp(row['largest_positive_cycle_origin']):%Y-%m-%d} | {row['largest_positive_cycle_pnl_cny']:.2f} | {row['largest_positive_share_of_positive_pnl']:.2%} | {net_label} |")
    lines.extend(["", "早期压力最大赢家2015-03-09贡献17887.90元，完成净利润16722.46元，占106.97%。这是原利润来源的算术集中度，没有删除该周期重跑账户，也没有据它改变样本或拟合。不能把早期夏普单项提高当成跨期稳健。", "",
                  "## 裁决及下一步", "",
                  "接受全体原点、抬线、收盘/相反失效、真实库存/资金/股息身份与覆盖的解释。排除‘单日反转是所有上涨的必要启动’、‘曾盈利证明能事前稳定止盈’和‘最后卖出时钟差能解释全部对A差距’。R181完整政策拒绝保持，没有新的收益夏普提升。", "",
                  next_action, "新金融未定义或登记，待跑0。原A/POINT十二真实前瞻和1008实际交易日计划保持，不自动映射历史形态为新前瞻或实盘。", "",
                  "全部历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED；去过拟合和提高收益夏普的完整目标未达，目标active。44冻结来源、四图已查看，0新必要测试/账户/拟合/训练标签/采集；本次保存身份核对不称新金融回测。", "",
                  "- [全部142周期及退出身份](results/全部142周期_已知抬线失效及真实资金时钟.csv)",
                  "- [全部1884持有原点与原决定核对](results/全部持有原点_低点更新收盘相反失效与原决定核对.csv)",
                  "- [全部逐日库存、现金、股息及财富](results/全部周期逐日真实库存现金应收及收盘财富.csv)",
                  "- [原61分段/49正式上涨两种区间覆盖](results/原61分段及49正式上涨_两种原区间覆盖不筛选.csv)",
                  "- [全98出生位置及未知](results/全部98出生_原上涨段位置仅事后对齐.csv)",
                  "- [六组全部24单元](results/全部六组两时期两费用_最高账面与最终结果仅解释.csv)",
                  "- [四案例全部7压力周期](results/四原案例全部相交压力周期_真实时钟与资金.csv)",
                  "- [冻结](protocol.json)、[全体诊断](summary.json)、[下一信息准入](next_information_admission_boundary.json)", ""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_SAVED_REVERSAL_CYCLES_COVERAGE_WEALTH_AND_FOUR_CHARTS_DELIVERED",
        "technical_decision": "TECH.R182", "new_accounts": 0, "all_actual_cycle_rows": 142,
        "holding_origins_verified": 1884, "complete_exit_identities_verified": 140,
        "original_admitted_waves": 49, "case_cycle_rows": 7, "charts_inspected": 4,
        "latest_financial_preserved": "TECH.R181", "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), OUT / "protocol.json", OUT / "summary.json", report]],
    }, exclusive=True)
    update_state(summary, boundary, report)


def update_state(summary, boundary, report):
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R181" and state["latest_financial_strategy_decision"] == "TECH.R181",
            "接手状态改变，不能覆盖其他阶段。")
    forward, next_experiment, trials = {key: state[key] for key in FORWARD}, state["next_experiment"], state["actual_candidate_trials"]
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    marker = "> 技术线当前事实 TECH.R182（优先于下方技术线历史快照，2026-10-05）："
    require(all(marker not in text for text in texts.values()), "长期事实已有本轮标记。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    relative = OUT.relative_to(ROOT).as_posix()
    overview = (
        "TECH.R182全98出生、142真实周期/1884持有决定/2024财富记录/140退出身份完成，0新账户。"
        "49正式上涨中21底至峰无同向形态、26确认至峰无形态，98出生72在原上升段/26不相交未知；底峰位置只事后对齐，不作交易门。"
        "压力70完成24赢46输，28曾收盘盈利最终亏损、18持有收盘从未盈利最终亏损。41收盘线退出/29相反事件退出，10只跌抬高线；24事后组全部保留。"
        "早期压力最大赢家2015-03-09净17887.90元，占完成净利润16722.46元的106.97%，没有删除重跑。"
        "2019两周期真实收盘最高936.69/6468.25元、最终768.45/4996.89元；2020-03-10周期各持有收盘均负，05-06曾+462.55元最终−1779.80元；"
        "2024-10-18晚期形态最高真实收盘+413.70元最终+62.90元，不覆盖9月启动。2015两个相交周期曾盈利最终负，全部失败保留。"
        "近期压力最后次开价差−607.00元/卖出摩擦1436.30元，时钟差2043.30元，对原A完成净利润差76735.50元，不能单独解释全部差距或当新退出上界。"
        "140退出身份最大残差<2e−11元，股息权利按原登记库存还原，7压力案例周期/四图已查看，44来源保持。"
        "本固定诊断完成，不切组、改窗口/退出、补量/MACD/RV或混A营救R181；下一实质不同当时信息及完整用途卡先核旧实际结果，或真实新样本/实现来源错误。"
        "待跑新金融0；最新技术R182、金融R181/登记R180、预测R158/退出R145保持。原12真实前瞻及1008实际交易日计划与其他分支保持。"
        "收益夏普/独立/去过拟合未达，DEVELOPMENT_CALIBRATION/first-vintage NOT_CERTIFIED/global DSR/PBO NOT_COMPUTED；目标active，本轮及前轮progress，阻塞0。")
    top = marker+(
        "全98出生/142周期、1884持有决定和140退出财富身份完成，0新账户。49正式上涨21底至峰无形态/26确认至峰无形态；压力70完成28曾盈利最终亏损/18从未收盘盈利最终亏损。"
        "早最大赢家贡献106.97%完成净利润；近期最后卖出时钟差2043.30元，对A净差76735.50元，不能单独解释差距或推新退出。"
        "固定全体诊断结束，不救R181；下一不同信息完整用途准入，待跑0。金融R181/登记R180与原前瞻保持，目标未达且active。"
        f" [全部覆盖、具体量价与真实资金](../{relative}/研究结果与下一步.md)。\n\n")
    decision = (
        "\n## TECH.R182：全体反转覆盖、生命周期与真实财富归因（2026-10-05）\n\n"
        "**假设**：先核对具体上涨与失败、完整反转覆盖及真实抬线/失效/资金时钟，区分入点覆盖不足、曾盈利回吐和次开摩擦，不把早期单项夏普改善视为跨期优势。\n\n"
        "**验证方法**：只读R181四保存账户，全98出生、142周期/140完成/2开放、1884持有决定、2024逐日真实财富、140退出身份及六事后组×四场景。原61分段/49正式上涨同时报告底至峰及确认至峰两区间；逐周期匹配真实库存、只抬事件低点、收盘线失败/相反事件及最终成交，股息权利按登记日库存。四图全部7压力周期相交，开放只观察自身账户末日。44来源冻结，0新账户/拟合/训练标签/采集/必要测试。\n\n"
        "**结果**："+overview+"\n\n"
        "**为什么接受/拒绝**：接受全体可知时钟、原真实财富身份及回顾性覆盖/集中度解释；排除‘单日强反转是所有上涨必要启动’、‘曾盈利说明能事前稳定止盈’、‘原最后卖出时钟差解释全部对A差距’及‘早夏普单项提高即可接受跨期模型’。最高财富/原底峰位置不是输入或可成交退出，真实数量的财富差额不是任意新策略上界。R181固定完整政策拒绝、原A及所有旧终态保持，没有新收益夏普提升。\n\n"
        "**是否需要重新验证/下一具体实验**：本固定全体诊断一次完成，不重复切组或增加量/MACD/RV、改变窗口/退出/预算/费用、删赢家或混A救回。下一信息及完整用途卡先核旧实际结果，或真实新样本/来源实现错误，才决定金融准入；当前未定义/登记新金融、待跑0。原A/POINT真实前瞻及1008实际交易日计划保持，历史重复不创造独立样本。\n\n"
        f"依据：[全体报告](../{relative}/研究结果与下一步.md)、[冻结](../{relative}/protocol.json)、[全部保存结果](../{relative}/summary.json)、"
        f"[全部真实周期](../{relative}/results/全部142周期_已知抬线失效及真实资金时钟.csv)、[覆盖](../{relative}/results/原61分段及49正式上涨_两种原区间覆盖不筛选.csv)、"
        f"[下一准入边界](../{relative}/next_information_admission_boundary.json)。\n")
    for name, text in texts.items():
        first, remainder = text.split("\n", 1)
        appended = decision if name != "PROJECT_STATE.md" else "\n## 技术线TECH.R182全体保存归因更新（2026-10-05）\n\n"+overview+f"\n\n[全体证据](../{relative}/研究结果与下一步.md)。\n"
        (ROOT / "docs" / name).write_text(first+"\n\n"+top+remainder.lstrip("\n")+"\n"+appended, encoding="utf-8", newline="\n")
    legacy = {key: value for key, value in state.items() if key.startswith("current_phase_")}
    state["archived_previous_current_phase_fields_before_R182"] = {"at": now(), "fields": legacy}
    for key in legacy:
        del state[key]
    work = {"scope": "TECH_R182_ALL_SAVED_RANGE_REVERSAL_LIFECYCLE_COVERAGE_ATTRIBUTION", "new_accounts": 0,
            "all_births": 98, "all_actual_cycles": 142, "holding_decisions": 1884, "cycle_wealth_rows": 2024,
            "exit_identities": 140, "original_episodes": 61, "original_admitted_waves": 49, "retrospective_groups": 24,
            "charts": 4, "case_cycles": 7, "frozen_sources": 44, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0}
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": relative, "latest_result": relative+"/summary.json",
        "latest_report": relative+"/研究结果与下一步.md", "latest_research_report": relative+"/研究结果与下一步.md",
        "latest_research_status": summary["status"], "latest_progress": overview, "latest_overall_summary": overview,
        "latest_continuation_outcome": overview, "current_study": "510300_FULL_RANGE_REVERSAL_SAVED_LIFECYCLE_ATTRIBUTION_V1",
        "current_phase": "ALL_FIXED_REVERSAL_COVERAGE_AND_EXIT_WEALTH_IDENTITIES_COMPLETED_NO_NEW_FINANCIAL_POLICY",
        "current_direction": "先解释具体上涨/失败，再全体真实位置、失效与资金归因；固定诊断完成，下一不同信息完整用途准入。",
        "current_priority": boundary["next_required_input"], "next_research_action": boundary["next_required_input"],
        "next_available_action": boundary["next_required_input"], "next_research_question": boundary["next_required_input"],
        "next_research_plan": relative+"/next_information_admission_boundary.json",
        "next_experiment_status": "FULL_FIXED_DIAGNOSTIC_COMPLETED_NEW_INFORMATION_PURPOSE_CARD_REQUIRED",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_candidate_field_status": "NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_R182_COMPLETE_DIAGNOSTIC",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R182", "latest_actual_model_decision": "TECH.R181",
        "latest_financial_strategy_decision": "TECH.R181", "latest_registration_decision": "TECH.R180",
        "latest_financial_registration_decision": "TECH.R180", "latest_actual_prediction_model_decision": "TECH.R158",
        "latest_original_exit_decision": "TECH.R145", "latest_continuation_receipt": relative+"/delivery_receipt.json",
        "latest_range_reversal_lifecycle_attribution": relative+"/summary.json",
        "latest_range_reversal_all_lifecycle_points": relative+"/results/全部142周期_已知抬线失效及真实资金时钟.csv",
        "latest_range_reversal_original_wave_coverage": relative+"/results/原61分段及49正式上涨_两种原区间覆盖不筛选.csv",
        "latest_new_information_admission_boundary": relative+"/next_information_admission_boundary.json",
        "new_accounts_this_continuation": 0, "new_accounts_in_current_phase": 0, "new_primary_accounts_this_continuation": 0,
        "new_financial_candidate_accounts_this_continuation": 0, "new_investment_account_evaluations_this_continuation": 0,
        "new_strategy_accounts_this_continuation": 0, "new_strategy_configurations_this_continuation": 0,
        "new_financial_result_computed_this_continuation": False, "new_account_return_sharpe": "NOT_RUN_THIS_SAVED_DIAGNOSTIC_LATEST_FINANCIAL_R181_RETAINED",
        "new_model_fits_this_continuation": 0, "historical_model_refits_this_continuation": 0,
        "new_return_labels_this_continuation": 0, "new_training_labels_this_continuation": 0,
        "new_model_training_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
        "current_information_mechanisms_admitted_this_continuation": 0, "current_fields_admitted_this_continuation": 0,
        "new_market_information_admitted": False, "new_method_admitted": "NONE_SAVED_DIAGNOSTIC_ONLY_FIXED_POLICY_CLOSED",
        "internal_reference_replays_this_continuation": 0, "saved_account_controls_replayed_this_continuation": 0,
        "saved_accounts_checked_this_continuation": 4, "necessary_tests_passed_this_continuation": 0,
        "necessary_tests_passed_in_current_phase": 0, "necessary_test_executions_this_continuation": 0,
        "actual_prefix_checks_this_continuation": 0, "pre_result_input_test_fixture_repairs": 0,
        "code_files_added_this_continuation": ["research/full_range_reversal_lifecycle_attribution_v1.py", "research/deliver_full_range_reversal_lifecycle_v1.py"],
        "current_goal_turn_actual_work": work, "current_phase_trial_accounting": work,
        "current_phase_source_freeze_count": 44, "current_phase_known_daily_rows": 3488, "current_phase_required_directions": ["LONG"],
        "current_phase_information_scope": "SAVED_R181_COVERAGE_HOLDING_CLOCKS_REAL_WEALTH_NOT_A_NEW_POLICY",
        "actual_candidate_trials_role": "LATEST_FINANCIAL_R181_TRIAL_ACCOUNTING_PRESERVED_NOT_R182_NEW_ACCOUNTS",
        "return_and_sharpe_changed_this_continuation": False, "return_and_sharpe_improved_this_continuation": False,
        "returns_and_sharpe_improved": False, "current_goal_turn_classification": "progress", "previous_goal_turn_classification": "progress",
        "previous_goal_turn_classification_reason": "R179—R181完成新完整用途及四真实账户，跨期拒绝证据改变下一步，不是状态重述。",
        "goal_turn_progress_classification": "ALL_SAVED_REVERSAL_COVERAGE_LIFECYCLE_AND_EXIT_WEALTH_IDENTITIES_COMPLETED",
        "current_goal_turn_classification_reason": "21/26正式上涨无形态、28曾盈利后亏/18从未盈利、最大赢家106.97%贡献及全部140退出身份改变对覆盖/退出/集中度的解释。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None,
        "blocked_audit_key": None, "blocking_decision": None, "goal_status": "active", "goal_tool_status_confirmed": "active",
        "goal_achieved": False, "overfitting_removed": False, "whole_model_overfitting_removed": False,
        "global_DSR_PBO": "NOT_COMPUTED", "independent_validation_status": "NOT_ESTABLISHED",
        "current_unmet_evidence": "R181经济/稳定门拒绝不变，R182完成覆盖/财富诊断无新策略；收益夏普/独立/去过拟合未达，下一新信息完整用途尚未准入。",
        "validation_method_this_continuation": "全142原周期身份/实际收益一致、1884持有决定和140退出财富身份、原49段两区间覆盖、24全组、四图7案例全交付；0新金融重跑。",
    })
    require({key: state[key] for key in FORWARD} == forward and state["next_experiment"] == next_experiment
            and state["actual_candidate_trials"] == trials, "原真实前瞻、1008日计划或R181金融试验账改变。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_diagnostic": "TECH.R182", "latest_financial_preserved": "TECH.R181", "registration_preserved": "TECH.R180",
        "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "forward_values_unchanged": forward, "existing_forward_next_experiment_unchanged": next_experiment,
        "actual_financial_trial_accounting_unchanged": trials, "goal_status": "active", "goal_achieved": False,
        "new_accounts": 0, "new_admitted_unrun_candidates": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__), STATE, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("R182全部覆盖、真实资金和四图交付，四事实一次更新；金融R181和原前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    run()
