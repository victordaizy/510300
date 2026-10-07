"""从全部已有真实周期解释信息顺序，附保存盈亏但不声称新策略胜率。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import actual_holding_information_format_v1_0_1 as repair
from research.write_support_price_acceptance_report_v1 import date_label, markdown

parent, ROOT, OUT, SOURCE = repair.parent, repair.ROOT, repair.OUT, repair.SOURCE
REPORT = OUT / "全部真实交易_新增量价宏观信息与延续失败顺序.md"


def sequence_order(row):
    accept, failure = row.first_price_acceptance_while_holding, row.first_entry_low_failure_while_holding
    if pd.isna(accept):
        return "NO_ACCEPTANCE_NO_LOW_FAILURE" if pd.isna(failure) else "LOW_FAILURE_WITHOUT_PRICE_ACCEPTANCE"
    if pd.isna(failure):
        return "PRICE_ACCEPTANCE_WITHOUT_LOW_FAILURE"
    return "LOW_FAILURE_BEFORE_PRICE_ACCEPTANCE" if failure < accept else "LOW_FAILURE_AFTER_PRICE_ACCEPTANCE"


def frozen_exact():
    old, new = parent.read(SOURCE / "protocol.json"), parent.read(OUT / "protocol.json")
    for item in [*old["sources"], *new["files"]]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原来源、动作或已保存序列改变。")
    return {"original_frozen_files_exact": len(old["sources"]), "completion_files_exact": len(new["files"])}


def figure(observed, cycles, accounts):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    cases = [("2015_2019", "2015-05-18", "2015-07-10", "2015下跌：支持与价格接受仍会失败"),
             ("2015_2019", "2018-12-17", "2019-03-29", "2019修复：原A和支持原型进入钟不同"),
             ("2020_2026", "2020-03-20", "2020-07-20", "2020修复到重新加速：旧锚消费后还有价格腿"),
             ("2020_2026", "2024-09-02", "2024-10-31", "2024重定价：原A三持仓收盘后已退出"),
             ("2020_2026", "2021-11-29", "2021-12-31", "2021反例：原确认周四天在进入前"),
             ("2020_2026", "2023-06-01", "2023-07-10", "2023反例：周抬高不等于新增周接受")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    for ax, (period, begin, end, title) in zip(axes.ravel(), cases):
        chunk = observed.loc[observed.date.between(begin, end)]
        ax.plot(chunk.date.to_numpy(dtype="datetime64[ns]"), chunk.ac, color="#2b3f50", linewidth=1.5, label="现金平移收盘")
        for policy, color, marker in [(parent.POLICIES[0], "#236a97", "^"), (parent.POLICIES[1], "#a46435", "o")]:
            trades = cycles.loc[cycles.period.eq(period) & cycles.cost.eq("STRESS") & cycles.policy.eq(policy)]
            selected = trades.loc[trades.entry_date.between(begin, end)]
            ys = observed.set_index("date").ac.reindex(pd.DatetimeIndex(selected.entry_date)).to_numpy()
            ax.scatter(selected.entry_date.to_numpy(dtype="datetime64[ns]"), ys, color=color, marker=marker, s=35,
                       label=parent.previous.NAMES[policy] + "实际进入", zorder=5)
            exits = trades.loc[trades.exit_date.between(begin, end)]
            ey = observed.set_index("date").ac.reindex(pd.DatetimeIndex(exits.exit_date)).to_numpy()
            ax.scatter(exits.exit_date.to_numpy(dtype="datetime64[ns]"), ey, color=color, marker="x", s=32, label=parent.previous.NAMES[policy] + "实际退出", zorder=5)
            for _, trade in selected.iterrows():
                if pd.notna(trade.first_wholly_new_week_acceptance_while_holding) and begin <= date_label(trade.first_wholly_new_week_acceptance_while_holding) <= end:
                    ax.axvline(trade.first_wholly_new_week_acceptance_while_holding, color=color, linestyle=":", alpha=.7, linewidth=.8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.18)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.legend(fontsize=7, frameon=False, loc="best", ncol=1)
    fig.suptitle("全部原案例与确认反例：实际进入、退出和买入后新完整周\n点与叉标实际成交日期，纵坐标对齐当日现金平移收盘；虚线为持仓内首次新增整周接受", fontsize=12)
    target = OUT / "figures/原四段上涨下跌与两确认反例_真实时钟.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return {"path": repair.relative(target), "sha256": parent.digest(target), "panels": 6}


def main():
    if REPORT.exists() or (OUT / "post_run_diagnosis.json").exists():
        raise RuntimeError("完整解释已保存，不重复写入。")
    summary = parent.read(OUT / "summary.json")
    if len(summary["prefix_checks"]) != 19 or not all(item["all_states_and_clocks_exact"] for item in summary["prefix_checks"]):
        raise ValueError("完整时序核对尚未完成。")
    exact = frozen_exact()
    panel = pd.read_parquet(SOURCE / "results/全部八账户_真实进入后的新增信息与退出后序列.parquet")
    cycles = pd.read_parquet(SOURCE / "results/全部实际周期与开放_新增证据和原净结果.parquet")
    cycles["sequence_order"] = cycles.apply(sequence_order, axis=1)
    complete = cycles.loc[cycles.status.eq("COMPLETE")]
    pressure = complete.loc[complete.cost.eq("STRESS")]
    core = pressure.loc[pressure.policy.eq(parent.POLICIES[0])]
    primary = pressure.loc[pressure.policy.eq(parent.POLICIES[1])]
    if len(cycles) != 128 or len(core) != 54 or len(primary) != 9 or len(panel) != 11268:
        raise ValueError("完整已有账户范围与实际序列不匹配。")
    criteria = ["post_entry_price_acceptance_while_holding", "wholly_new_complete_week_acceptance_while_holding",
                "new_support_publication_while_holding", "entry_day_low_failure_while_holding"]
    groups = []
    for (period, cost, policy), frame in complete.groupby(["period", "cost", "policy"], sort=False):
        for criterion in criteria:
            for value in (True, False):
                selected = frame.loc[frame[criterion].eq(value)]
                groups.append({"period": period, "cost": cost, "policy": policy, "criterion": criterion, "present": value,
                    "saved_completed_cycles": len(selected), "saved_positive_outcomes": int(selected.net_pnl.gt(0).sum()),
                    "saved_negative_outcomes": int(selected.net_pnl.lt(0).sum()),
                    "role": "POST_ENTRY_OBSERVATION_CONDITION_NO_NEW_STRATEGY_WIN_RATE"})
    orders = []
    for (period, cost, policy, sequence), frame in complete.groupby(["period", "cost", "policy", "sequence_order"], sort=False):
        orders.append({"period": period, "cost": cost, "policy": policy, "sequence_order": sequence,
            "saved_completed_cycles": len(frame), "saved_positive_outcomes": int(frame.net_pnl.gt(0).sum()),
            "saved_negative_outcomes": int(frame.net_pnl.lt(0).sum()),
            "role": "FULL_SEQUENCE_AFTER_THE_FACT_NOT_AVAILABLE_AT_FIRST_LOW_FAILURE"})
    groups, orders = pd.DataFrame(groups), pd.DataFrame(orders)
    repair.table("全部观察条件与原结果标注_不作新策略胜率", groups)
    repair.table("全部低点失败与接受先后顺序_原结果不作预测标签", orders)
    repair.table("全部128保存周期与信息先后完整解释", cycles)
    recovered = core.loc[core.entry_day_low_failure_while_holding & core.net_pnl.gt(0)]
    slow_missing = core.loc[~core.wholly_new_complete_week_acceptance_while_holding & core.net_pnl.gt(0)]
    never_accepted = core.loc[core.sequence_order.eq("LOW_FAILURE_WITHOUT_PRICE_ACCEPTANCE")]
    before = core.loc[core.sequence_order.eq("LOW_FAILURE_BEFORE_PRICE_ACCEPTANCE")]
    after = core.loc[core.sequence_order.eq("LOW_FAILURE_AFTER_PRICE_ACCEPTANCE")]
    whole = core.loc[core.wholly_new_complete_week_acceptance_while_holding]
    coverage = pd.read_parquet(OUT / "results/全部持有与退出后_价格信息角色及未知覆盖.parquet")
    covered = coverage.loc[coverage.cost.eq("STRESS")]
    observed = pd.read_parquet(parent.DAILY)
    image = figure(observed, cycles, parent.load_accounts())
    diagnosis = {"at": parent.previous.parent.original.now(), "decision": parent.RESULT, **exact,
        "all_sequence_rows": len(panel), "all_saved_cycles_with_cost_replicates": len(cycles),
        "all_saved_completed_with_cost_replicates": len(complete), "all_saved_open_with_cost_replicates": int(cycles.status.eq("RIGHT_CENSORED").sum()),
        "same_economic_events_not_independent_cost_replications": True, "all_pressure_A_completed_cycles": len(core),
        "all_pressure_A_open_cycles": 1, "all_pressure_primary_completed_cycles": len(primary),
        "A_price_acceptance_during_hold": int(core.post_entry_price_acceptance_while_holding.sum()),
        "A_wholly_post_entry_week_acceptance_during_hold": len(whole), "A_whole_week_existing_positive": int(whole.net_pnl.gt(0).sum()),
        "A_whole_week_existing_negative": int(whole.net_pnl.lt(0).sum()),
        "A_positive_without_new_whole_week": len(slow_missing), "A_recent_positive_without_new_whole_week": int(slow_missing.period.eq("2020_2026").sum()),
        "A_low_failure_but_existing_positive": len(recovered), "A_low_before_acceptance_cycles": len(before),
        "A_low_before_acceptance_existing_positive": int(before.net_pnl.gt(0).sum()), "A_low_after_acceptance_cycles": len(after),
        "A_low_after_acceptance_existing_positive": int(after.net_pnl.gt(0).sum()), "A_low_without_acceptance_cycles": len(never_accepted),
        "A_low_without_acceptance_existing_negative": int(never_accepted.net_pnl.lt(0).sum()),
        "known_only_after_sequence_finished_not_a_first_failure_forecast": True,
        "A_new_support_publication_during_hold_cycles": int(core.new_support_publication_while_holding.sum()),
        "all_no_new_publication_as_no_policy": False, "whole_week_survival_selection_limit": True,
        "sparse_news_gate_or_uniform_week_entry_gate_admitted": False,
        "new_strategy_return_sharpe": "NOT_COMPUTED", "new_strategy_win_rate": "NOT_COMPUTED",
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "independent_validation": "NOT_ESTABLISHED", "old_actual_financial_decision": "TECH.R232",
        "next_materially_different_purpose": "原A保持进入来源与目标，实际进入后价格接受/失败和新周结构决定持有阶段；宏观新公开信息撤销延续资格，未知保持CORE。",
        "goal_achieved": False}
    parent.write(OUT / "post_run_diagnosis.json", diagnosis)
    proposal = {"at": parent.previous.parent.original.now(), "status": "PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN",
        "name": "CORE_ENTRY_ACTUAL_ACCEPTANCE_FAILURE_AND_NEW_INFORMATION_CARRY",
        "question": "保持原A完整进入及权重来源，通过买入后已发生的接受/失效顺序和新增信息改变持有，能否提高完整收益与夏普？",
        "reason": "政策单次触发过少；短赢家不等待新周；先低后接受的修复与接受后失效须区别；不同作用在真实进入后才能使用。",
        "entry_and_weights": "原A保存目标来源/大小/0与未知、10个百分点调仓带，原完整账户风险与费用；不重训或改源。",
        "holding_mechanism": "CORE保持原动作；实际进入日高低在其收盘后固定。严格后来价格接受才启用接受后结构保护；形成全部买日起的新完整周和已知参与证据可晋延续，允许其状态独立于旧目标归零。",
        "macro_role": "旧水平与空记录不强制判好坏；买后真正新公布订单恶化或已记录反向操作可撤销延续资格。宣布/旧背景新观察/确认和共同源分开。",
        "unknown_role": "不足以确认延续就保持原CORE，不能新发入场或把未知当收紧。",
        "failure_and_episode": "价格接受后的结构失效执行退出；真实强制退出消费本原来源正目标段，防旧正目标立即重复入场；下一原来源0→正才重启。各动作细节须在金融前唯一固定。",
        "necessary_controls": ["同CORE进入和新增结构持有但不使用参与/宏观信息", "同CORE进入和接受后保护但不延长原退出", "原A保存完整账户", "原R212阶段保存完整账户"],
        "financial_scope": "一主政策两新增归因对照×两独立时期×两费用，完整现金/股息/风险/整手/T+1/次开/自然终态；不得借描述组18/20直接报胜率。",
        "financial_acceptance": "完整净CAGR/Sharpe均正且提高、DD<=10%、实际pB>1及标准净期望>0、原20/252块全部区间；交易频率软目标。",
        "before_admission": "完成唯一状态转换、保护位、锁定与未知规则、必要源时序测试和原账户精确对照后冻结；当前不金融运行。",
        "no_rescue": "不是修改R232周纯度或窗口，不拼接早期新原型/近期A，不把后验低点失败分类当事前标签。",
        "financial_admission": "NOT_ADMITTED", "new_financial_runs": 0, "independent_validation": "NOT_ESTABLISHED"}
    parent.write(OUT / "next_core_actual_acceptance_carry_proposal.json", proposal)
    shown = cycles.loc[cycles.cost.eq("STRESS")].copy()
    shown["policy_name"] = shown.policy.map(parent.previous.NAMES)
    shown_groups = groups.loc[groups.cost.eq("STRESS")].copy()
    shown_groups["policy_name"] = shown_groups.policy.map(parent.previous.NAMES)
    shown_orders = orders.loc[orders.cost.eq("STRESS")].copy()
    shown_orders["policy_name"] = shown_orders.policy.map(parent.previous.NAMES)
    texts = ["# 全部真实交易：买入后的新增量价宏观信息与延续、失败顺序",
        f"TECH.R233登记、R234结果；{summary['at']}。完整3488原观察日/65来源身份，8已保存账户、11268持仓与退出后日行、128费用复本周期（126完成/2开放），原19关键日整段前缀精确。费用复本不是128独立机会；压力原A54完成/1开放、支持原型9完成全部解释。0新金融/拟合/行情/训练目标；最新实际金融仍R232拒绝，本轮新收益与夏普NOT_COMPUTED。",
        "## 解释改变了什么",
        "上一完整周已可知，未必全部形成于实际买入后。本轮固定真实进入日高低，分别描述后来价格接受、低点失败及完整买后周结构；把持有期间、真实退出后分开。真正新公布、旧支持首次观察、实施目标确认、同文件多个节点分开；量、行业、宏观各有角色，未建立统一线性评分。",
        f"原A54完成中36曾发生价格接受，20曾形成新增完整周接受，但其原结果18正2负，不能当新策略90%胜率：确认要在实际持有中等到，有存续与价格路径选择，进入时尚未知道会不会确认。另13原A盈利没有新完整周，其中近期10笔；2020Jul1—Jul7净利13302.43、2024Sep25—Sep30净利11030.84分别只有4/3持仓收盘。统一等待新周会错过这些快上涨，而对R232只改变周线纯度仍未金融检验。",
        "## 价格接受、失败的先后比统一低点止损更有信息",
        f"原A7笔先低点失败、后来价格接受，原结果6正1负；7笔先接受、后来低点失败，2正5负；11笔低点失败后在真实持有中没有接受，全部原结果负。最后一类直到持有结束才知道‘后来没有接受’，第一次低点失败时不能预测它。所有类别仅是保存原结果标注，无新交易概率或pB。",
        markdown(shown_orders, {"period": "时期", "policy_name": "原账户", "sequence_order": "完整观察顺序", "saved_completed_cycles": "原完成", "saved_positive_outcomes": "原结果正", "saved_negative_outcomes": "原结果负"}),
        "原A共有8笔低点失败却最终盈利，不能简单把进入日低当所有阶段统一强制止损。两筆先接受后失败仍有原正结果，所以接受后保护也必须核对提前卖出的真实价格、再次现金进入和费用。下面列出全部8例，没有挑其中赢家证明策略。",
        markdown(recovered, {"period": "时期", "entry_date": "实际进入", "exit_date": "实际退出", "net_pnl": "原净损益元", "first_entry_low_failure_while_holding": "首次低点失败", "first_price_acceptance_while_holding": "首次价格接受", "sequence_order": "顺序"}, dates=("entry_date", "exit_date", "first_entry_low_failure_while_holding", "first_price_acceptance_while_holding")),
        "## 不同上涨段的实际解释",
        "2015：原AJun3进入、Jun23退出净亏5715.56元，期间有价格接受和买后完整周接受，随后低点失败。支持信息及两个价格确认都不是保证。原AJan6亏7409.70、Jan22亏739.99均未在持有中出现价格接受；另一方面Feb11长周期净利21060.90与其他修复全保留，不能只从崩跌例给所有时期制定退出。",
        "2019：支持原型Jan10进入、Mar27退出净利16105.95；原A直到Feb28才进入，Mar6退出净利1295.87。两者数量、原来源和期间不同，不能相减叫持有增量。Jan初日MACD与行业参与先修复，PMI新订单仍49.7/变化−0.7、周柱负。之后2019May31原A先低点失败后价格接受，最终3579.22正收益，修复途中弱点和已经接受后的失败不同。",
        "2020：支持原型Apr3实际进入来自Mar30操作锚、Apr2价格决定，Apr3 16:57:32公告只能Apr7首次ETF观察。Apr14已退出，6月支持锚仍已消费；Jun2是旧阶段修复决定，原A并非Jun2入场，直到Jul1才有对应真实周期。原AJul7退出净利13302.43，快速加速段没有等待全买后周。原阶段Jun2盈利10756.16和原A不混同。",
        "2024：上午宣布→Sep24价格量响应→原ASep25次开进入→Sep30退出，原净利11030.84、三实际持仓收盘，原退出理由按保存父目标零。支持原型Sep26进入、Oct17退出净利7747.24。原AOct9重新进入、Oct10净亏2808.61，原型则一直持有；快速重新定价、后续失效与新进入风险必须联合解释，不能只延长Sep30那笔而忽略后来资金路径。",
        "2021/2023：R232在Dec13’21/Jun19’23得到原抬高周确认，四日来自买前、一日来自买日起；其真实持有结束前均没有新增完整买后周接受，分别亏2686.01/2058.31。2020Feb18和2025May13两亏损同样没有新整周接受。但是‘后来未确认’是持有终点描述，不能假称已找到四笔可提前避免的亏损；否则只是用存续选择胜者。",
        "## 新公布、旧观察和未知的覆盖",
        f"原A54完成只有{int(core.new_support_publication_while_holding.sum())}笔在持有中遇到本有限集合的真正新支持公布。没有节点不代表当天无政策；要求每个进入/持有都遇到新政策会再次过度稀疏。原资金/融资新公布多数是日常统计更新，不当独立资金流冲击。PMI新订单是调查背景，源龄超限仍未知，不靠前值填补。行业逐点成员/分类可能变化，未认证原进入固定组传播，不直接比较比例差作新扩散。",
        markdown(covered, {"period": "时期", "policy": "账户", "phase": "实际阶段", "daily_rows": "日行", "industry_known_rows": "行业已知", "orders_known_rows": "订单已知", "funding_known_rows": "资金已知", "margin_known_rows": "融资已知", "new_support_publication_events": "新支持公布", "old_support_new_observation_events": "旧支持新观察", "target_operation_confirmation_events": "目标操作确认"}),
        "## 全部观察条件与原结果",
        markdown(shown_groups, {"period": "时期", "policy_name": "账户", "criterion": "观察条件", "present": "曾发生", "saved_completed_cycles": "原完成", "saved_positive_outcomes": "原结果正", "saved_negative_outcomes": "原结果负"}),
        "各行条件都在买入后才出现，既有盈亏只是附注；不同条件集合重叠，不能相加为机会数或新模型正确数。原开放周期不归为胜者/败者。基础和压力全部128行及64条件行在结果表保存；上表仅展示压力解释，并未据费用选择新策略。",
        "## 下一唯一完整机制",
        "最值得继续的是保留原A进入与目标权重，把真实进入后的接受、失效、完整新周和信息参与用于持有阶段转换。CORE先保留原动作；价格被接受后启用相应结构保护，新完整周与已知参与证据可支持延续；旧目标归零时是否延续由已形成的状态决定。买后新公布宏观恶化/反向操作可以撤销延续资格，未知保持原CORE；强制退出消费原正目标段，防立即重复旧信号。",
        "这套提案PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN。需要先固定唯一保护位、转换、锁定/再启动及未知细节，必要测试和原A精确对照，再比较主政策、无新增信息的同持有动作、无延长的同接受后保护、原A/原阶段。完整账户要包含提前退出影响、释放现金新进入、风险/费用/股息，才能判断收益与夏普。不是仅修R232周纯度、根据低点失败组反选门或拼接旧净值。",
        "## 时序、技术接续与独立验证",
        f"7必要测试通过；初次两测试的样例把当日当完整周末，被守卫拒绝后仅修登记前样例。原冻结一次处理全部8序列/128周期，首个前缀因全NaT时间列推断秒/纳秒类型不同退出。原失败和输入留档；另立1.0.1在临时比较副本规范声明的时间列，每个原时间值/索引和其他全部特征精确；有限无时区公布钟仍拒绝。2必要时间测试、19整段前缀完成，未重新生成保存全序列或运行金融。原{exact['original_frozen_files_exact']}输入和{exact['completion_files_exact']}接续文件精确。",
        "所有历史已知且多次研究选择未校正，本轮不能说去过拟合或独立验证成功。最新实际金融R232拒绝，目标active、尚未实现；原13独立前瞻/E03及其他旧策略与失败保持。",
        "## 全部压力实际周期与开放",
        markdown(shown, {"period": "时期", "policy_name": "原账户", "cycle_id": "周期", "entry_date": "进入", "exit_date": "退出", "status": "完成/开放", "net_pnl": "原净损益元", "actual_holding_close_rows": "持仓收盘", "first_price_acceptance_while_holding": "首次接受", "first_entry_low_failure_while_holding": "首次低点失败", "first_wholly_new_week_acceptance_while_holding": "首次新周接受"}, dates=("entry_date", "exit_date", "first_price_acceptance_while_holding", "first_entry_low_failure_while_holding", "first_wholly_new_week_acceptance_while_holding")),
        "六案例图的点与叉标实际成交日，纵坐标对齐当日现金平移收盘；实际开盘成交价、滑点和佣金见原订单表。",
        "## 可复查证据",
        "[原用途](../../../../docs/510300_ACTUAL_HOLDING_INFORMATION_V1.md)、[原冻结协议](../protocol.json)、[原时间类型失败](../prefix_storage_failure.json)、[接续协议](protocol.json)、[实际完成](summary.json)、[完整归因](post_run_diagnosis.json)、[下一不同整体机制](next_core_actual_acceptance_carry_proposal.json)。原八完整账户按原协议逐路径保存，全部序列与128周期在上级results，前缀/覆盖/条件/顺序在本results。",
        "![六原案例和真实时钟](figures/原四段上涨下跌与两确认反例_真实时钟.png)"]
    with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n\n".join(texts) + "\n")
    parent.write(OUT / "delivery_receipt.json", {"at": parent.previous.parent.original.now(), "report": repair.relative(REPORT),
        "report_sha256": parent.digest(REPORT), "figure": image, "figure_actually_viewed": "NOT_YET",
        "all_saved_sequence_rows": len(panel), "all_saved_cycles_with_cost_replicates": len(cycles),
        "all_pressure_A_completed": len(core), "all_pressure_primary_completed": len(primary),
        "all_conditions_rows": len(groups), "all_sequence_order_rows": len(orders), "frozen": frozen_exact(),
        "new_financial_runs": 0, "new_strategy_return_sharpe": "NOT_COMPUTED"})
    print("R234完整解释与六案例图已保存：全部128复本周期，不以后验条件报新策略胜率。", flush=True)


if __name__ == "__main__":
    main()
