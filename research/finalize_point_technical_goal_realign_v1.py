"""保存技术点位纠偏、已有财务结果及全交易图谱；不修改冻结策略。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.point_technical_goal_realign_v1 import OUT

STATE_DIR = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
MARKER = "2026-10-03：按用户纠偏回到日周线技术点位（TECH.R146—R148）"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percentage(value):
    return "未知" if pd.isna(value) else f"{value:.2%}"


def number(value):
    return "未知" if pd.isna(value) else f"{value:.3f}"


def run() -> None:
    require(not (OUT / "fact_recording_receipt.json").exists(), "本次纠偏事实已记录，不重复。")
    summary = read(OUT / "summary.json")
    require(summary["status"] == "COMPLETED_TECHNICAL_TRADE_MAP_DESCRIPTION", "实际点位整理尚未完成。")
    goal = read(OUT / "actual_goal_tool_result.json")
    require(goal["goal"]["status"] == "active", "目标工具当前状态未确认继续。")
    source_out = ROOT / "reports/research/510300_point_notice_reason_information_intake_v1"
    source_result = read(source_out / "summary.json")
    require(not (ROOT / "reports/research/510300_point_full_notice_reason_design_v1/protocol.json").exists(),
            "未执行公告分支的状态已变化，应先读取实际记录。")
    accounts = pd.read_parquet(OUT / "原压力成本完整账户指标.parquet")
    groups = pd.read_parquet(OUT / "固定MACD背景_全部胜负对照.parquet")
    account_lines = ["| 原历史时期 | 候选 | 净年化 | 全日历净夏普 | 最大回撤 | 净胜率 | 实际净盈亏比 | 净pB | 完整周期 |",
                     "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in accounts.itertuples(index=False):
        account_lines.append(f"| {row.period} | {row.policy} | {percentage(row.net_cagr)} | {number(row.net_sharpe)} | "
                             f"{percentage(row.max_drawdown)} | {percentage(row.win_rate)} | {number(row.payoff)} | "
                             f"{number(row.p_times_b)} | {row.completed_cycles} |")
    group_lines = ["| 原时期 | 入场前MACD背景 | 完整交易 | 盈利/亏损 | 实际净pB | 原周期平均净回报 |",
                   "|---|---|---:|---:|---:|---:|"]
    for row in groups.itertuples(index=False):
        if row.context_group not in ["ALL_ORIGINAL_COMPLETED_CYCLES", "UNKNOWN_ORIGINAL_INDICATOR_CONTEXT"]:
            group_lines.append(f"| {row.period} | {row.context_group} | {row.completed_cycles} | {row.wins}/{row.losses} | "
                               f"{number(row.p_times_actual_net_b)} | {percentage(row.mean_original_cycle_net_return)} |")
    report = "\n".join([
        "# 回到510300技术点位：本次纠偏与真实研究结果", "",
        "用户指出“你做的偏了”。偏移发生在研究单位：近期不断转向央行公告来源和原退出模型的字段增量，"
        "没有直接产生更好的入场点位或更高的账户收益。技术主线现回到日线、此前完整周、MACD、量价与波动，"
        "以实际多头交易的净胜率、实际盈亏比、完整账户净收益和夏普评价。次数为软目标，空头后置。", "",
        "## 已经完成什么", "",
        "本次原样复用A的压力成本账户，将全部55个实际周期逐笔对齐入场前一个交易日的技术背景，"
        "54个完整周期全部保留胜负、1个开放周期单列。原MACD12/26/9、EMA20、相对量、相对波动和此前完整周定义完全复用。"
        "三处按时间固定的前缀核对通过，没有取入场当天未来收盘或未完成周；分组完整且互斥，原pB与原净损益对齐。"
        "新增拟合、账户回测、交易、收益标签、联网均0，本次没有提高收益或夏普的新策略。", "",
        "逐笔入口：[全部实际点位与入场前指标](全部实际点位_入场前日周线量价.csv)。", "",
        "## 目前实际财务结果", "",
        "以下直接摘录原已完成比较，20万元、252日年化、空仓日计入、现金比较率0，压力成本为单边万四佣金、千一滑点、最低5元。"
        "原风险上限和次日开盘执行保留。两个时期各自启动账户，不拼接优胜版本。",
        "", *account_lines, "",
        "A近期优于B，但早期A/B都没有满足用户的净pB>1额外门槛。原A保留仓位强弱相对二值点位的近期净年化3.12%→3.99%、夏普0.880→1.217，"
        "是已保存的历史机制改善；没有因此建立独立有效性。完整年平均次数A早期4.40、近期2020—2025为4.17。", "",
        "## 为什么不能只追求高胜率或指标同步转强", "",
        *group_lines, "",
        "较早时期日柱正/周柱非正组7笔中6笔盈利，但净pB只有0.315；近期日周柱均正组6笔中5笔盈利，净pB也只有0.567。"
        "一笔大亏损能抵消许多小盈利。反过来，较早日柱非正/周柱正组只有2/7盈利，却有较大的平均盈利。"
        "这些小组仅3—10笔、属于已经看过的开发资料，不能挑最好的组设过滤器、计算伪造的过滤后夏普，或宣布稳定模式。", "",
        "分组pB使用原周期净回报net_return=净损益/累计买入支出；标准亏损单位期望为pB−实际亏损率q。"
        "分组净现金利润还受原仓位大小影响，两种统计不能混为同一个风险分母。", "",
        "## 回到原目标后的取舍", "",
        "1. 已拒绝NR7增次、统一正目标、按年份拼A/B，以及已失败MACD、EMA、周锚与量价固定表达的调参营救。"
        "NR7虽增加周期，却降低收益和夏普；成交订单不能代替完整交易次数。",
        "2. 原路径诊断没有发现普遍延迟多日退出。较早/近期压力亏损周期的中间持有贡献为−25,876.20/−12,726.04元；"
        "盈利周期的大部分利润也在中间阶段，不能由事后亏损组推出统一提前退出。",
        "3. 暂停在本技术任务继续推进公告理由分支。TECH.R146已完成来源支持，65已知/1442未知、27成熟月零理由支持，"
        "仅拒绝固定来源对原退出两系数模型的必要资格。TECH.R147只写了未登记程序，未运行950原件抽取或金融模型；用户纠偏后不继续。"
        "它的首轮9项解析/时钟测试5通过4失败，问题在全未知时空时间列的时区比较，未覆盖原R146实跑结果，不能冒充金融失败或金融通过。",
        "4. 115成熟月可识别和原预测MSE门属于已冻结退出增量合同；不自动变成全部技术点位研究的前置条件。"
        "原合同内全部失败维持原裁决，也不能跳过其失败门去补算经济收益来营救。", "",
        "## 下一步具体实验", "",
        "下一步仍是提案，尚未登记或回测：在本次全交易表和既有全部空仓失败对照上，先明确一种入场或持有失效的技术机制，"
        "核对它与旧失败是否存在实质区别，再冻结唯一完整首版。完整规则必须在入场前确定触发、结构失效位置、下一开盘执行和退出，"
        "把计划收益风险比与最终实际盈亏比分开。主评价直接比较原共同账户的净年化、全日历净夏普、回撤、净pB、完整年度次数及最长等待。"
        "不能只报告指标解释、预测MSE或入场胜率。", "",
        "新增机会需要检查与原A持仓时间重叠和合并后实际账户；所有失败对照和未完成周期保留，不只选成功上涨。"
        "简单再加MACD、量或周线转强条件不因本次纠偏自动成为新机制。当前没有已接纳的新数值候选，不把方案文本写成已取得策略改善。", "",
        "旧A/B前瞻和E03保持其原登记，不等待宏观资料才允许技术研究；E03的1008实际交易日终点只属于原仓位机制比较，"
        "不约束全部新实验，也不自动验证整个策略。全部既有历史仍是开发校准资料，独立验证未成立，正式DSR/PBO未计算，去过拟合与收益夏普目标未完成。", "",
        "直接来源：[本次描述定义](description_plan.json)、[实际结果](summary.json)、"
        "[原A/B共同账户比较](../510300_point_second_weight_comparison_v1/研究结论.md)、"
        "[原路径诊断](../510300_point_weight_path_bottleneck_v1/研究解释与下一步.md)、"
        "[原长空仓与失败对照](../510300_point_long_wait_diagnostic_v1/研究结论.md)。", "",
    ])
    (OUT / "研究纠偏与当前结果.md").write_text(report, encoding="utf-8")
    next_experiment = {
        "at": now(), "status": "PROPOSED_NOT_REGISTERED_NOT_RUN", "user_steering": "你做的偏了",
        "priority": "TECHNICAL_ENTRY_OR_HOLD_INVALIDATION_FULL_ACCOUNT_COMPARISON",
        "current_new_numeric_candidates": 0,
        "steps": ["从全部原胜负点位及全部空仓失败对照明确一种事前可知技术机制，先与旧合同核对实质区别。",
                  "只冻结一种完整首版：入场、已知失效位置、退出、原成本资金风险执行及未完成周期口径。",
                  "比较独立完整账户的净年化、全日历净夏普、回撤、实际净pB/标准期望、频率、最长等待；新增点位另检合并A账户。",
                  "固定早期与近期均保留，历史只开发，另明确独立验证；不挑组、挑年或调失败参数。"],
        "not_a_global_entry_gate": "原115成熟月秩/MSE及E03终点仅约束所属冻结合同。",
        "financial_candidate_registered": False, "new_fits": 0, "new_accounts": 0,
        "goal_achieved": False,
    }
    write_json(OUT / "next_experiment.json", next_experiment, exclusive=True)
    state_path = STATE_DIR / "state.json"
    old_state_bytes = state_path.read_bytes()
    (OUT / "state_before_realign.json").write_bytes(old_state_bytes)
    state = json.loads(old_state_bytes.decode("utf-8-sig"))
    protected_keys = [key for key in state if "forward" in key or "prospective" in key
                      or key in ["earliest_future_exchange_session", "next_new_close_eligible_at",
                                 "registered_candidate_intents", "accepted_independent_candidates", "current_validated_candidates"]]
    protected = {key: state[key] for key in protected_keys}
    scope = {"at": now(), "user_quote": "你做的偏了", "interpretation": "停止本技术任务向公告来源和机械追加退出字段漂移，回到日周线技术量价点位和实际账户目标。",
             "goal_tool_status": goal["goal"]["status"], "original_strategy_and_all_rejections_preserved": True,
             "other_branches_authority_unchanged": True, "source_R146_actual_result": source_result,
             "source_R147": "NOT_REGISTERED_NOT_RUN_STOPPED_AFTER_USER_STEERING",
             "current_trade_map": summary, "next_experiment": next_experiment}
    write_json(STATE_DIR / "technical_goal_realignment_scope_20261003.json", scope, exclusive=True)
    banner = ("2026-10-03按用户“你做的偏了”纠偏：本任务停止推进公告理由来源分支，回到510300日周线技术量价点位、"
              "实际胜率/盈亏比及同资金成本风险的净收益/全日历夏普。TECH.R148已将原A压力账户55周期（54完整、1开放）"
              "对齐入场前指标，三处前缀检查通过、0拟合/账户/交易/标签/联网；高胜率组仍可能净pB不足1，分组只描述、不选最优过滤器。"
              "TECH.R146为已完成来源支持拒绝，不是金融收益裁决；R147未登记未执行，纠偏后不继续。"
              "最新实际金融模型仍R145拒绝，原13表达与全部冻结失败保持。原退出115月秩/MSE门不扩大到所有技术实验。"
              "独立验证、去过拟合和收益夏普目标未完成；目标工具active。下一完整技术点位实验仅提案，0新候选已接纳。")
    section = "\n\n## " + MARKER + "\n\n" + banner + "\n\n" + (
        "**假设**：原日周线、量价及波动背景需要在实际全部胜负点位上解释，并最终落实为净优势完整交易。\n\n"
        "**验证方法**：原压力A账户55周期全保留，入场前原交易日指标，原MACD/EMA/量价/波动及此前完整周定义；"
        "54完整周期分两原时期、四固定背景，全未知单列；三处时间前缀及原pB/损益对齐。\n\n"
        "**结果**：较早日柱正/周柱非正6胜1负但净pB0.315；近期日周柱均正5胜1负但净pB0.567。"
        "组仅3—10笔、历史开发且受原仓位影响，未建立新过滤器或账户提升。0新拟合/账户/交易/标签/联网。\n\n"
        "**接受/拒绝原因**：接受回到实际点位与账户目标、原交易对齐和高胜率不充分的描述；不接受任何组为有效策略。"
        "原MSE/完整115月资格合同及失败保留，只限定原退出增量用途，不自动约束其他独立技术机制。\n\n"
        "**是否需要重新验证**：不重跑原失败、不改其门；新完整点位候选先说明实质新机制并单独冻结，再按原资金/成本/风险比较。"
        "下一实验未登记，独立证据与去过拟合仍未成立。\n\n"
        "**公告分支的实际状态**：R146原1823对象映射1507状态为65已知/1442未知、27成熟月零支持，"
        "仅固定来源支持失败；R147写了隔离程序但协议未登记、950原件未抽取、金融未运行，首轮9测试5通过4失败"
        "为全未知时间列时区错误，用户纠偏后停止推进，不记成金融模型失败或合格待跑。其他宏观分支权限不受本条影响。\n\n"
        "直接证据：[研究纠偏与当前结果](../reports/research/510300_technical_goal_realign_20261003/研究纠偏与当前结果.md)、"
        "[实际点位表](../reports/research/510300_technical_goal_realign_20261003/全部实际点位_入场前日周线量价.csv)、"
        "[下一实验提案](../reports/research/510300_technical_goal_realign_20261003/next_experiment.json)、"
        "[R146实际来源结果](../reports/research/510300_point_notice_reason_information_intake_v1/summary.json)。\n")
    doc_receipts = []
    for relative in ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
                     "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        path = ROOT / relative
        content_bytes = path.read_bytes()
        content = content_bytes.decode("utf-8-sig")
        require(MARKER not in content, "纠偏条目已存在，应先读取。")
        lines = content.splitlines()
        if relative.endswith("_TECHNICAL_LINE.md"):
            prefixes = ("> 用户允许新增隔离实验", "> 依据当前协议、代码、结果及Git历史；最新TECH.")
        else:
            prefixes = ("> 技术线本轮最新TECH.",)
        indices = [i for i, line in enumerate(lines) if line.startswith(prefixes)]
        require(len(indices) == 1, "技术线顶部摘要位置不唯一。")
        lines[indices[0]] = "> 技术线纠偏更新：" + banner
        updated = "\n".join(lines).rstrip() + section
        require(path.read_bytes() == content_bytes, "共享文档刚被其他任务更新，请读取后接续。")
        path.write_text(updated, encoding="utf-8")
        doc_receipts.append({"path": relative, "sha256": digest(path)})
    state["previous_blocking_decision_before_user_realignment"] = state.get("blocking_decision")
    state["previous_blocked_audit_count_before_user_realignment"] = state.get("blocked_audit_count")
    state.update({"updated_at": now(), "status": "research_active", "goal_status": "active",
                  "goal_tool_status_confirmed": "active", "goal_achieved": False,
                  "latest_user_instruction": "你做的偏了", "current_phase": "TECHNICAL_ENTRY_TRADE_QUALITY_REALIGNMENT",
                  "current_study": summary["study"], "latest_completed_study": summary["study"],
                  "latest_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "latest_report": (OUT / "研究纠偏与当前结果.md").relative_to(ROOT).as_posix(),
                  "latest_research_status": summary["status"], "latest_technical_decision": "TECH.R148",
                  "latest_actual_model_decision": "TECH.R145", "latest_model_decision": "TECH.R145",
                  "latest_progress": banner, "current_direction": "原实际点位与账户质量的技术诊断；不再在本任务推进公告理由来源。",
                  "next_research_question": "哪一种事前技术机制能改善入场或持有失效识别，并在原共同账户提高净收益和净夏普？",
                  "next_experiment_status": "PROPOSED_NOT_REGISTERED_NOT_RUN",
                  "next_research_action": (OUT / "next_experiment.json").relative_to(ROOT).as_posix(),
                  "next_strategy_increment_status": "NO_NEW_NUMERIC_STRATEGY_REGISTERED",
                  "source_R147_status": "NOT_REGISTERED_NOT_RUN_STOPPED_AFTER_USER_STEERING",
                  "latest_source_intake": source_out.relative_to(ROOT).as_posix(),
                  "user_goal_realignment_scope": "technical_goal_realignment_scope_20261003.json",
                  "previous_goal_turn_classification": state.get("goal_turn_classification", "no_progress"),
                  "goal_turn_classification": "progress", "blocked_audit_count": 0,
                  "blocking_decision": None, "live_own_process_handle": None, "verified_wait": False,
                  "new_model_fits_this_continuation": 0, "new_accounts_this_continuation": 0,
                  "new_strategy_accounts_this_continuation": 0, "new_accounts_in_current_phase": 0,
                  "new_investment_account_evaluations_this_continuation": 0,
                  "new_stock_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
                  "current_economic_stage_status": "NOT_RUN_NO_NEW_STRATEGY_REGISTERED",
                  "current_study_prediction_gate_status": "NOT_APPLICABLE_DESCRIPTIVE_TRADE_MAP",
                  "current_trade_map_cycles": 55, "current_trade_map_complete_cycles": 54,
                  "current_trade_map_open_cycles": 1, "saved_original_account_metrics_quoted": 4,
                  "return_and_sharpe_changed_this_continuation": False,
                  "overfit_removed": False})
    require({key: state[key] for key in protected_keys} == protected, "原前瞻值被改变。")
    require(state_path.read_bytes() == old_state_bytes, "当前任务状态被其他写入改变。")
    write_json(state_path, state)
    write_json(OUT / "fact_recording_receipt.json", {
        "at": now(), "status": "RECORDED_USER_REALIGNMENT_AND_ACTUAL_TRADE_MAP",
        "documents": doc_receipts, "state_sha256": digest(state_path),
        "goal_tool_status": "active", "prior_blocked_state_archived": True,
        "protected_forward_values_preserved": protected,
        "latest_actual_financial_decision": "TECH.R145", "new_strategy_improvement": False,
        "other_project_branches_preserved": True, "goal_achieved": False,
    }, exclusive=True)
    print("纠偏、55个原交易周期及当前真实财务结果已写入报告、四份项目事实文件和本任务状态；原策略及前瞻登记保持。")


if __name__ == "__main__":
    run()
