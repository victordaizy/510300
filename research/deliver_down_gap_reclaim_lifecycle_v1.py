"""一次交付R186全体覆盖与财富归因，保留R185金融裁决和原真实前瞻。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.down_gap_reclaim_lifecycle_attribution_v1 import OUT, FINANCIAL, EXPLANATION, STATE, PERIODS
from research.finalize_downtrend_break_state_v1 import FORWARD, DOCS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def run():
    require(not (OUT / "delivery_receipt.json").exists() and not (OUT / "project_state_update_receipt.json").exists(),
            "R186交付及事实更新已完成，不重复。")
    summary = read(OUT / "summary.json")
    require(summary["status"] == "COMPLETED_ALL_SAVED_RECLAIM_COVERAGE_FIXED_FAILURE_AND_WEALTH_IDENTITIES", "全体归因未完成。")
    for name in ("protocol.json", "repair_protocol.json"):
        for item in read(OUT / name)["sources"]:
            require(digest(ROOT / item["path"]) == item["sha256"], "原冻结或隔离日期恢复来源改变。")
    cycles = pd.read_parquet(OUT / "results/全部102周期_原锚固定线失效及真实资金时钟.parquet")
    cases = pd.read_parquet(OUT / "results/四原案例全部相交压力周期_真实时钟与资金.parquet")
    coverage = pd.read_parquet(OUT / "results/原61分段及49上涨_两种原区间覆盖不筛选.parquet")
    grouped = pd.read_parquet(OUT / "results/全部六组两时期两费用_最高财富与最终结果仅解释.parquet")
    original = pd.read_parquet(FINANCIAL / "results/全部实际进出点位_已知收复量价及真实资金.parquet")
    require(len(cycles) == 102 and len(cases) == 7 and len(grouped) == 24 and int(coverage.admitted.sum()) == 49, "原全体数量不同。")
    for row in cycles.itertuples(index=False):
        expected = original.loc[original.period.eq(row.period) & original.cost.eq(row.cost) & original.cycle_id.eq(row.cycle_id)]
        require(len(expected) == 1, "诊断周期没有唯一R185周期。")
        expected = expected.iloc[0]
        for key in ("entry_origin", "entry_date", "entry_quantity", "status"):
            require(getattr(row, key) == expected[key], "诊断改变真实周期身份或份额。")
        require(int(row.fixed_floor_ticks) == int(expected.entry_reclaim_floor_ticks) == int(expected.fixed_reclaim_floor_ticks), "固定线与原交易不一致。")
        if row.status == "COMPLETE":
            require(abs(row.actual_net_return-expected.net_return) <= 1e-12 and abs(row.actual_net_pnl-expected.net_pnl) <= 1e-8, "原完成收益改变。")
            reconstructed = row.exit_origin_marked_pnl_cny+row.final_open_gap_gross_cny+row.dividend_accrual_after_final_origin_cny-row.final_sell_commission-row.final_sell_slippage
            require(abs(reconstructed-row.actual_net_pnl) <= 1e-6, "退出财富身份不同。")
        else:
            require(pd.isna(row.actual_net_return) and pd.isna(row.actual_net_pnl), "开放被填完成结果。")
        require(row.cycle_observed_until <= pd.Timestamp(PERIODS[row.period][1]), "开放延伸到其他时期。")
    early = next(row for row in summary["all_scenario_diagnostics"] if row["period"] == "2015_2019" and row["cost"] == "STRESS")
    recent = next(row for row in summary["all_scenario_diagnostics"] if row["period"] == "2020_2026" and row["cost"] == "STRESS")
    metrics = pd.read_parquet(FINANCIAL / "results/完整账户共同口径比较.parquet")
    a = metrics.loc[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS") & metrics.policy.eq("A_SAVED_WEIGHT")].iloc[0]
    a_difference = float(a.completed_cycle_net_pnl-recent["completed_pnl_cny"])
    final_clock_difference = float(recent["final_exit_friction_cny"]-recent["final_open_gap_gross_cny"]-recent["post_final_origin_dividend_cny"])
    pressure = cycles.loc[cycles.cost.eq("STRESS") & cycles.status.eq("COMPLETE")]
    never_positive_loss = int(pressure.final_result_group.eq("NONPOSITIVE_HOLDING_CLOSE_FINAL_LOSS").sum())
    never_positive_win = int(pressure.final_result_group.eq("NONPOSITIVE_HOLDING_CLOSE_FINAL_WIN").sum())
    next_action = (
        "本次全部收复归因完成，不再追加同一政策分组或退出搜索。下一先提出实质不同、当时可观察的日线信息及完整用途卡，"
        "说明它在具体上涨/失败段的机制、可知时钟、与已拒绝完整用途的区别以及全部反例；核旧实际结果后才决定金融准入。"
        "也可是真实新样本或来源/实现错误。保留原A优势、收复覆盖不足及浮盈回吐，不能把原最高财富、未来底峰、"
        "2024放量值或已见分组作为新交易过滤，不营救R185/R181/R177及其他冻结终态。")
    boundary = {
        "at": now(), "status": "ALL_FIXED_RECLAIM_LIFECYCLE_DIAGNOSTICS_COMPLETE_NO_NEW_FINANCIAL_CANDIDATE",
        "latest_diagnostic": "TECH.R186", "latest_actual_financial": "TECH.R185",
        "accepted_new_facts": {"first_reclaims": 60, "all_cycles": 102, "holding_decisions": 2186, "exit_wealth_identities": 100,
                               "admitted_waves_without_bottom_to_peak_reclaim": 19, "admitted_waves_without_confirmation_to_peak_reclaim": 28,
                               "pressure_positive_close_then_loss": 20, "pressure_never_positive_then_loss": never_positive_loss,
                               "pressure_never_positive_then_win": never_positive_win,
                               "recent_largest_winner_share_of_net_pnl": recent["largest_positive_share_of_net_pnl"],
                               "recent_original_final_exit_clock_difference_cny": final_clock_difference,
                               "recent_original_completed_profit_difference_to_A_cny": a_difference},
        "finite_rejected_explanations": ["单次缺口收复是所有上涨的必要启动", "持有收盘从未盈利就必然最终亏损",
                                         "原最后卖出时钟差解释全部对A完成净利润差距", "曾浮盈或2024成功说明已有事前稳定止盈规则",
                                         "近期pB正即可接受跨期收益夏普目标"],
        "next_required_input": next_action, "closed_current_policy": "R185固定完整收复政策拒绝保持",
        "no_additional_same_diagnostic_split_planned": True, "new_admitted_unrun_candidates": 0,
        "next_new_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE", "new_accounts": 0,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "scope_limit": "本固定用途和归因结束不等于全项目暂停；原A/POINT真实前瞻保持。",
    }
    write_json(OUT / "next_information_admission_boundary.json", boundary, exclusive=True)
    lines = ["# 向下缺口收复：全体覆盖、固定失效与真实财富归因", "",
             "TECH.R186只读R185保存结果，全60首次收复/120资格、102周期、2186持有原点和2286财富记录完成；100退出财富身份、全部失败/取消/2开放、原61/49段与四案例全部7个压力周期保留。0新账户/拟合/训练标签/采集。", "",
             "## 具体上涨与失败路径", "",
             "2024年9月24日放量收复先于DIF和上一完整周柱转正，真实9月25日进入；10月8日真实收盘账面最高15679.93元。固定线没有随着上涨上移，2025年4月7日出现新的完整向下缺口时，收盘周期账面仍3454.23元；4月8日实际出场净3412.85元（+5.65%）。它确实提供了启动阶段的已知修复点，但不能据此按事后10月最高点退出或断定存在稳定止盈。", "",
             "2020年4月2日修复时量0.741倍，日柱正但DIF/上一周柱负；实际4月3日进入，5月22日新向下缺口触发已知失效，25日出场净1497.86元。6月1日另一缺口收复，2日进场后仍有短暂账面亏损；7月急涨时日周动量与量上升，7月13日持有收盘最高14168.65元，8月20日新向下缺口、21日出场净12483.58元。它保留了大部分已发生盈利；窗口前3月2日收复周期则曾账面+1239.29元最终亏2361.71元，必须一起解释。", "",
             "2019年3月18日确认是春季趋势中回调收复而非1月启动。真实3月19日进、25日跌固定线、26日出；全部持有收盘中最好的财富仍−159.27元，最终−600.41元，无法用‘只因卖晚了回吐’解释。窗口前2018年12月周期亦保留，最高持有收盘−348.91元、最终−425.66元。", "",
             "2015年7月9日收复时日周动量仍负、RV比3.620，真实10日进入；7月23日持有收盘曾+3336.32元。8月21日收盘跌固定线时已经账面−2543.58元，24日次开价差−989元、卖出摩擦46.46元，最终−3579.04元（−9.88%）。长周期下行中的局部修复可以很快浮盈，也可以再次失效。", "",
             "## 全体资金和退出时钟", "",
             "| 时期 | 费用 | 完成/开放 | 曾收盘盈利最终亏 | 从未收盘盈利最终亏 | 固定线/新缺口退出 | 最后开盘价差/元 | 最后卖出摩擦/元 | 完成净利润/元 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["all_scenario_diagnostics"]:
        lines.append(f"| {row['period']} | {row['cost']} | {row['completed']}/{row['open']} | {row['completed_positive_close_then_loss']} | {row['never_positive_close_final_loss']} | {row['close_floor_exits']}/{row['new_down_gap_exits']} | {row['final_open_gap_gross_cny']:.2f} | {row['final_exit_friction_cny']:.2f} | {row['completed_pnl_cny']:.2f} |")
    lines.extend(["", f"压力50完成中20曾收盘盈利最终亏、{never_positive_loss}从未持有收盘盈利最终亏；另有{never_positive_win}个持有收盘从未盈利但最终实际盈利的周期，原最后开盘成交可能超过此前持有收盘财富。不能将未浮盈等同于必亏。全部六组×四场景24格及空组质量未知保持，开放不混入完成组。", "",
                  f"近期压力最大赢家是2020-06-01确认周期，净12483.58元，占全部正利润{recent['largest_positive_share_of_positive_pnl']:.2%}、完成净利润{recent['largest_positive_share_of_net_pnl']:.2%}；早期净利润负，最大赢家/净利润占比未知。未删除最大赢家或重跑账户。", "",
                  f"近期压力原最后次开价差+{recent['final_open_gap_gross_cny']:.2f}元、卖出摩擦{recent['final_exit_friction_cny']:.2f}元，相对原最后卖出前收盘财富的净差{final_clock_difference:.2f}元。对原A完成净利润差{a_difference:.2f}元；这一原账本时钟项只解释部分差额。没有计算改成收盘卖出的账户，也不是任意止盈策略的收益上界。", "",
                  "实际完成净利润−最终卖出前原点财富=该原点实际余量×(真实原开盘−原点原收盘)+后来该周期已发生股息应收−最终佣金−滑点。100退出身份最大残差<2e−11元，股息权利按原登记库存、逐次风险减仓和实际买卖现金还原。", "",
                  "## 上涨覆盖与全体案例", "",
                  "原49正式上涨中19段底峰之间没有首次收复、28段原5%确认至峰之间没有首次收复；全60收复43在原上涨分段中、17无分段匹配。原分段、未来底峰和最高财富均仅事后解释，不是交易输入。两区间同时报告，不选有利范围。", "",
                  "| 原案例 | 已知收复 | 真实进场 | 真实退出 | 最高持有收盘财富/元 | 最后退出原点财富/元 | 最后开盘价差/元 | 实际净利润/元 |",
                  "|---|---|---|---|---:|---:|---:|---:|"])
    for row in cases.itertuples(index=False):
        lines.append(f"| {row.original_episode_id} | {row.entry_origin:%Y-%m-%d} | {row.entry_date:%Y-%m-%d} | {row.exit_date:%Y-%m-%d} | {row.best_observed_holding_close_pnl_cny:.2f} | {row.exit_origin_marked_pnl_cny:.2f} | {row.final_open_gap_gross_cny:.2f} | {row.actual_net_pnl:.2f} |")
    for path in summary["charts"]:
        lines.extend(["", f"![真实周期与当时指标](./{Path(path).name})", ""])
    lines.extend(["## 裁决、限制和下一步", "",
                  "接受全体覆盖、固定线和已知新缺口失效、真实库存/财富及最后时钟身份解释；R185固定完整金融用途拒绝不变。近期pB正和2024启动识别不代替原A比较或早期负结果。高浮盈回吐本身没有给出事前可靠退出规则，不根据本结果改固定线、MACD/量/RV、时期或混A营救。", "",
                  "首次纯归因脚本因日期设为索引后删除列、输出出生日期时读取缺失列而停止，尚未保存诊断表、0新账户。原代码/原冻结/首次开始标记和错误保存；隔离恢复仅保留与原索引同值日期列，并核对完全一致。原算法、分组、固定线、账户和收益未修改，46原冻结来源与6恢复绑定来源均核对。四图已查看。", "",
                  "全部历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，完整收益夏普/去过拟合目标未达。", "",
                  next_action, "", "原A/POINT十二前瞻值和1008实际交易日计划保持，待跑新金融0。", "",
                  "- [全部102真实周期](results/全部102周期_原锚固定线失效及真实资金时钟.csv)",
                  "- [全部持有原点核对](results/全部持有原点_固定线新缺口及原决定核对.csv)",
                  "- [逐日实际财富](results/全部周期逐日库存现金股息及收盘财富.csv)",
                  "- [原61/49段两区间覆盖](results/原61分段及49上涨_两种原区间覆盖不筛选.csv)",
                  "- [全部60收复位置](results/全部60首次收复_原上涨位置仅事后对齐.csv)",
                  "- [六组×四场景](results/全部六组两时期两费用_最高财富与最终结果仅解释.csv)",
                  "- [冻结](protocol.json)、[全体结果](summary.json)、[日期恢复](repair_protocol.json)、[首次失败](initial_execution_failure.json)",
                  "- [下一准入边界](next_information_admission_boundary.json)", ""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    relative = OUT.relative_to(ROOT).as_posix()
    state_hash, state = digest(STATE), read(STATE)
    require(state["latest_technical_decision"] == "TECH.R185" and state["latest_financial_strategy_decision"] == "TECH.R185", "其他工作已更新阶段，不覆盖。")
    forward, next_experiment, trials = {key: state[key] for key in FORWARD}, state["next_experiment"], state["actual_candidate_trials"]
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    doc_hashes = {name: digest(ROOT / "docs" / name) for name in DOCS}
    marker = "> 技术线当前事实 TECH.R186（优先于下方技术线历史快照，2026-10-05）："
    require(all(marker not in text for text in texts.values()), "事实已有本期记录。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    overview = (
        "完成R186全体收复保存归因：60首次收复/120资格、102实际周期（100完成/2开放）、2186持有原点、2286财富记录和100最后次开身份，0新账户。"
        "原49上涨19底峰间无收复/28确认至峰无收复，43收复在原分段/17无匹配，未来底峰/最高财富只事后观察。"
        "压力50完成20曾盈利最终亏、15从未持有收盘盈利最终亏，另1从未收盘盈利最终赢，不能从未浮盈推必亏；24完整事后组保留。"
        "2015-07-09周期最高真实收盘3336.32元、失效原点−2543.58元、次开−989元/摩擦46.46元、最终−3579.04元；"
        "2019-03-18周期全部持有收盘未盈利、最终−600.41元。2020-04-02修复最高3858.24元最终1497.86元，"
        "06-01修复最高14168.65元最终12483.58元，捕捉7月上涨但不按最高点退出。"
        "2024-09-24量3.338、日柱先正而慢动量负的入点，最高10-08真实收盘15679.93元，2025-04-07新向下缺口时仍3454.23元，04-08出净3412.85元；原线固定没有按未来峰值抬高。"
        f"近期压力最大赢家占全部正利润{recent['largest_positive_share_of_positive_pnl']:.4%}/完成净利润{recent['largest_positive_share_of_net_pnl']:.4%}，早净负时净占比未知，不删赢家。"
        f"近期原最后次开+972.20元/摩擦3356.88元、净时钟差{final_clock_difference:.2f}元，对原A完成净利润差{a_difference:.2f}元，"
        "仅为原数量/退出日期的财富身份，不是收盘退出账户或任意新止盈上界。全部100身份最大残差<2e−11元、固定线/原决定/登记股息及实际净收益一致，7压力案例/四图已查看。"
        "首次日期列移索引错误在首周期输出时停止、未存诊断表、0新账户；原代码/冻结/首次标记和错误保留，隔离恢复只保留索引同值日期列，原算法、政策和收益不改。46原来源/6恢复绑定来源核对。"
        "本固定全体归因完成，不再切组/调退出、补量/MACD/RV或混A营救；下一实质不同当时信息+完整用途先核旧实际结果，或真实新样本/来源实现错误。"
        "最新技术R186、金融R185/登记R184、预测R158/原退出R145保持，金融试验账和12真实前瞻/1008实际交易日计划及其他分支保持；待跑新金融0。"
        "收益夏普/独立/去过拟合未达、DEVELOPMENT_CALIBRATION/first-vintage NOT_CERTIFIED/global DSR/PBO NOT_COMPUTED；目标active/本轮及前轮progress/阻塞0。")
    top = marker+(
        "全60收复/102周期、2186持有原点及100退出财富身份完成，0新账户。49上涨19底峰间/28确认峰间无收复；压力50完成20曾盈利最终亏、15从未收盘盈利最终亏、另1从未收盘盈利最终赢。"
        "2024最高真实收盘15679.93元最终3412.85元，近期最大赢家占完成净利润47.64%；最后时钟差2384.68元，对A完成净利润差31643.82元，仅部分原财富身份。"
        "固定全体归因结束，不营救R185，下一不同信息完整用途准入、待跑0。金融R185/登记R184及原前瞻保持，完整目标未达且active。"
        f" [全体覆盖、具体量价和真实财富](../{relative}/研究结果与下一步.md)。\n\n")
    decision = (
        "\n\n## TECH.R186：全体收复覆盖、固定失效与真实财富归因（2026-10-05）\n\n"
        "**假设**：具体上涨中修复的时间位置、原锚固定线/新向下缺口失效、实际库存/风险减仓和最后次开时钟，能解释为何近期pB正而低于A、早期负；最高收盘财富不是事前退出输入。\n\n"
        "**验证方法**：只读R185四保存账户、全60收复/120资格/102周期（100完成/2开放），逐个持有原点匹配原固定线和决定、实际库存/买卖净现金、登记股息与日收盘财富；原61/49段两固定区间、六组×四场景全体，四图全部7压力案例周期。46原冻结来源，首次日期列错误后6恢复来源绑定，0新账户/拟合/标签/采集/新增必要测试。\n\n"
        "**结果**："+overview+"\n\n"
        "**为什么接受/拒绝**：接受全体已知失效时钟和真实财富身份、覆盖及集中度解释；排除‘收复是所有上涨必要启动’、‘从未持有收盘浮盈就必然最终亏’、‘原最后次开差解释全部对A完成净利润差’、‘曾浮盈说明已有事前可稳定执行止盈’及‘近期pB正即可完成跨期目标’。固定金融R185拒绝保持，没有计算不同退出或改参账户，收益夏普尚未提升。\n\n"
        "**是否需要重新验证/下一具体实验**：本固定全体归因一次完成，不追加分组/窗口/退出/量MACD/RV/预算费用或混A。只有实质不同信息及完整用途、真实新样本或来源/实现错误才进一步准入；必须先核旧实际用途及结果，当前待跑新金融0。原A/POINT真实前瞻和1008实际交易日计划保持；原代码日期列错误及隔离同值修复证据保留，历史重复不创造独立验证。\n\n"
        f"依据：[全体报告](../{relative}/研究结果与下一步.md)、[保存结果](../{relative}/summary.json)、[原冻结](../{relative}/protocol.json)、"
        f"[恢复冻结](../{relative}/repair_protocol.json)、[下一准入边界](../{relative}/next_information_admission_boundary.json)。\n")
    for name, text in texts.items():
        path = ROOT / "docs" / name
        require(digest(path) == doc_hashes[name], "事实文件同时有其他修改，不覆盖。")
        first, remainder = text.split("\n", 1)
        appended = decision if name.startswith("RESEARCH_DECISIONS") else "\n\n## 技术线TECH.R186全体保存归因更新（2026-10-05）\n\n"+overview+f"\n\n[全体证据](../{relative}/研究结果与下一步.md)。\n"
        path.write_text(first+"\n\n"+top+remainder.lstrip("\n")+appended, encoding="utf-8", newline="\n")
    work = {"scope": "TECH_R186_ALL_SAVED_RECLAIM_COVERAGE_AND_WEALTH_ATTRIBUTION", "new_accounts": 0,
            "first_reclaims": 60, "all_actual_cycles": 102, "holding_decisions": 2186, "cycle_wealth_rows": 2286,
            "exit_identities": 100, "original_episodes": 61, "original_admitted_waves": 49, "retrospective_groups": 24,
            "charts": 4, "case_cycles": 7, "frozen_sources": 46, "repair_binding_sources": 6,
            "initial_implementation_failure_preserved": 1, "isolated_date_representation_repairs": 1,
            "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0}
    legacy = {key: value for key, value in state.items() if key.startswith("current_phase_")}
    state["archived_previous_current_phase_fields_before_R186"] = {"at": now(), "fields": legacy}
    for key in legacy:
        del state[key]
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": relative, "latest_result": relative+"/summary.json",
        "latest_report": relative+"/研究结果与下一步.md", "latest_research_report": relative+"/研究结果与下一步.md",
        "latest_research_status": summary["status"], "latest_progress": overview, "latest_overall_summary": overview,
        "latest_continuation_outcome": overview, "current_study": "510300_DOWN_GAP_RECLAIM_SAVED_LIFECYCLE_ATTRIBUTION_V1",
        "current_phase": "ALL_FIXED_RECLAIM_COVERAGE_AND_EXIT_WEALTH_IDENTITIES_COMPLETED_NO_NEW_FINANCIAL_POLICY",
        "current_direction": "具体上涨/失败、最新缺口收复覆盖与真实固定线/库存/财富归因完成，下一不同信息完整用途准入。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_question": next_action, "next_research_plan": relative+"/next_information_admission_boundary.json",
        "next_experiment_status": "FULL_FIXED_RECLAIM_DIAGNOSTIC_COMPLETED_DIFFERENT_INFORMATION_PURPOSE_REQUIRED",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE", "next_candidate_field_status": "NO_NEW_ADMITTED_FINANCIAL_POLICY_AFTER_R186_DIAGNOSTIC",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R186", "latest_actual_model_decision": "TECH.R185", "latest_financial_strategy_decision": "TECH.R185",
        "latest_registration_decision": "TECH.R184", "latest_financial_registration_decision": "TECH.R184",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145",
        "latest_continuation_receipt": relative+"/delivery_receipt.json", "latest_reclaim_lifecycle_attribution": relative+"/summary.json",
        "latest_reclaim_all_lifecycle_points": relative+"/results/全部102周期_原锚固定线失效及真实资金时钟.csv",
        "latest_reclaim_original_wave_coverage": relative+"/results/原61分段及49上涨_两种原区间覆盖不筛选.csv",
        "latest_new_information_admission_boundary": relative+"/next_information_admission_boundary.json",
        "new_accounts_this_continuation": 0, "new_accounts_in_current_phase": 0, "new_primary_accounts_this_continuation": 0,
        "new_financial_candidate_accounts_this_continuation": 0, "new_investment_account_evaluations_this_continuation": 0,
        "new_strategy_accounts_this_continuation": 0, "new_strategy_configurations_this_continuation": 0, "new_financial_result_computed_this_continuation": False,
        "new_account_return_sharpe": "NOT_RUN_THIS_SAVED_DIAGNOSTIC_LATEST_FINANCIAL_R185_RETAINED",
        "new_model_fits_this_continuation": 0, "historical_model_refits_this_continuation": 0, "new_return_labels_this_continuation": 0,
        "new_training_labels_this_continuation": 0, "new_model_training_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
        "current_information_mechanisms_admitted_this_continuation": 0, "current_fields_admitted_this_continuation": 0,
        "new_market_information_admitted": False, "new_method_admitted": "NONE_SAVED_DIAGNOSTIC_ONLY_FIXED_RECLAIM_POLICY_CLOSED",
        "internal_reference_replays_this_continuation": 0, "saved_account_controls_replayed_this_continuation": 0,
        "saved_accounts_checked_this_continuation": 4, "necessary_tests_passed_this_continuation": 0, "necessary_tests_passed_in_current_phase": 0,
        "necessary_test_executions_this_continuation": 0, "actual_prefix_checks_this_continuation": 0, "pre_result_input_test_fixture_repairs": 0,
        "code_files_added_this_continuation": ["research/down_gap_reclaim_lifecycle_attribution_v1.py", "research/repair_down_gap_reclaim_lifecycle_date_v1.py", "research/deliver_down_gap_reclaim_lifecycle_v1.py"],
        "current_goal_turn_actual_work": work, "current_phase_trial_accounting": work, "current_phase_source_freeze_count": 46,
        "current_phase_known_daily_rows": 3488, "current_phase_required_directions": ["LONG"],
        "current_phase_information_scope": "SAVED_R185_COVERAGE_FIXED_FAILURE_AND_REAL_WEALTH_NOT_A_NEW_POLICY",
        "current_phase_isolated_date_representation_repairs": 1,
        "actual_candidate_trials_role": "LATEST_FINANCIAL_R185_TRIAL_ACCOUNTING_PRESERVED_NOT_R186_NEW_ACCOUNTS",
        "return_and_sharpe_changed_this_continuation": False, "return_and_sharpe_improved_this_continuation": False, "returns_and_sharpe_improved": False,
        "current_goal_turn_classification": "progress", "previous_goal_turn_classification": "progress",
        "previous_goal_turn_classification_reason": "R183—R185完成新具体解释及四实际账户，近期pB正而原A更高/早期负，跨期拒绝证据改变下一步。",
        "goal_turn_progress_classification": "ALL_SAVED_RECLAIM_COVERAGE_FIXED_FAILURE_AND_WEALTH_IDENTITIES_COMPLETED",
        "current_goal_turn_classification_reason": "19/28正式上涨无收复、20曾盈利最终亏及1未浮盈最终赢、最大赢家47.64%和100财富身份改变对覆盖、回吐及时钟差的解释。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "blocked_reason": None, "blocked_audit_key": None, "blocking_decision": None,
        "goal_status": "active", "goal_tool_status_confirmed": "active", "goal_achieved": False, "overfitting_removed": False,
        "whole_model_overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED", "independent_validation_status": "NOT_ESTABLISHED",
        "current_unmet_evidence": "R185经济/稳定门拒绝不变，R186全体覆盖/财富归因没有新策略；收益夏普/独立/去过拟合未达，下一不同完整用途尚未准入。",
        "validation_method_this_continuation": "全部102周期身份/原实际收益不变，2186持有决定/100退出财富身份、原49上涨两区间覆盖/24完整组/四图7案例交付；首次日期失败及隔离同值恢复保存。",
    })
    require({key: state[key] for key in FORWARD} == forward and state["next_experiment"] == next_experiment
            and state["actual_candidate_trials"] == trials, "原前瞻、1008日计划或R185金融试验账改变。")
    require(digest(STATE) == state_hash, "当前状态同时被其他工作修改，不覆盖。")
    write_json(STATE, state)
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_RECLAIM_COVERAGE_FIXED_FAILURE_WEALTH_AND_FOUR_CHARTS_DELIVERED",
        "technical_decision": "TECH.R186", "all_actual_cycles_checked": 102, "holding_decisions": 2186, "exit_identities": 100,
        "case_cycles": 7, "charts_visually_inspected": summary["charts"], "new_accounts": 0,
        "first_date_column_failure_preserved": True, "original_frozen_code_and_protocol_unchanged": True,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in [Path(__file__), report, OUT / "summary.json", OUT / "repair_protocol.json"]],
    }, exclusive=True)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_diagnostic": "TECH.R186", "latest_financial_preserved": "TECH.R185", "registration_preserved": "TECH.R184",
        "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145", "forward_values_unchanged": forward,
        "existing_forward_next_experiment_unchanged": next_experiment, "actual_financial_trial_accounting_unchanged": trials,
        "goal_status": "active", "goal_achieved": False, "new_accounts": 0, "new_admitted_unrun_candidates": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in [Path(__file__), STATE, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("R186全体覆盖/财富和四图交付，四长期事实一次更新，金融R185及原前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    run()
