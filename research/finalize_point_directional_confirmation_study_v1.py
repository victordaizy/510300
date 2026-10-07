"""整理方向确认完整账户结果；不重跑金融策略，不据结果改变规则。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, now, require
from research.point_directional_confirmation_study_v1 import OUT, PERIODS, COSTS, CONTROLS, read, table
from research.upward_episode_anatomy_v1 import save_json


def pct(value):
    return f"{value:.2%}" if np.isfinite(value) else "不可估计"


def number(value):
    return f"{value:.3f}" if np.isfinite(value) else "不可估计"


def account(period, cost, mode):
    path = CONTROLS / period / cost / "A_SAVED_WEIGHT" if mode == "A_SAVED_WEIGHT" else OUT / "results/accounts" / period / cost / mode
    return {name: pd.read_parquet(path / f"{name}.parquet") for name in ["daily", "trades", "orders"]}


def decomposition():
    rows, native_rows, core_rows = [], [], []
    for period in PERIODS:
        for cost in COSTS:
            original = account(period, cost, "A_SAVED_WEIGHT")
            combined = account(period, cost, "A_PLUS_DIRECTIONAL")
            a, b = original["daily"], combined["daily"]
            gross_delta = float((b.price_pnl + b.dividend_accrual).sum() - (a.price_pnl + a.dividend_accrual).sum())
            fees_delta = float(b.commission.sum() - a.commission.sum())
            slip_delta = float(b.slippage.sum() - a.slippage.sum())
            nav_delta = float(b.equity.iloc[-1] - a.equity.iloc[-1])
            require(abs(nav_delta - gross_delta + fees_delta + slip_delta) < 1e-6, "实际损益差桥不成立。")
            trades = combined["trades"].copy()
            complete = trades.status.eq("COMPLETE")
            trades["marked_net_pnl"] = trades.sell_net_cny + trades.dividend_cny - trades.buy_debit
            unfinished = trades.loc[~complete]
            require(len(unfinished) <= 1, "同一资金账户有多个未结束实际持仓。")
            if len(unfinished):
                trades.loc[unfinished.index[0], "marked_net_pnl"] += float(b.shares.iloc[-1] * b.close.iloc[-1])
            require(abs(trades.marked_net_pnl.sum() - (b.equity.iloc[-1] - 200000.)) < 1e-6,
                    "新完整/未完成交易不能还原全部净财富。")
            native = trades.loc[trades.source.eq("DIRECTIONAL")]
            native_marked = float(native.marked_net_pnl.sum())
            core_marked = float(trades.loc[trades.source.eq("CORE_WEIGHT"), "marked_net_pnl"].sum())
            original_net = float(a.equity.iloc[-1] - 200000.)
            require(abs(native_marked + core_marked - original_net - nav_delta) < 1e-6, "来源净财富差桥不成立。")
            native_rows.extend({"period": period, "cost": cost, **r} for r in native.to_dict("records"))
            old_core = original["trades"]
            new_core = trades.loc[trades.source.eq("CORE_WEIGHT")]
            old_entry = set(pd.to_datetime(old_core.entry_date))
            new_entry = set(pd.to_datetime(new_core.entry_date))
            for r in old_core.to_dict("records"):
                same = new_core.loc[new_core.entry_date.eq(r["entry_date"])]
                core_rows.append({"period": period, "cost": cost, "original_entry_date": r["entry_date"],
                                  "original_exit_date": r.get("exit_date"), "same_entry_in_combined": len(same) == 1,
                                  "combined_exit_date": same.exit_date.iloc[0] if len(same) == 1 else pd.NaT,
                                  "same_exit_when_entry_matched": bool(len(same) == 1 and same.exit_date.iloc[0] == r.get("exit_date"))})
            rows.append({"period": period, "cost": cost, "ending_equity_delta": nav_delta,
                         "gross_at_actual_quantities_delta": gross_delta, "commission_delta": fees_delta,
                         "slippage_delta": slip_delta, "new_source_marked_net_pnl": native_marked,
                         "core_source_marked_net_pnl_delta": core_marked - original_net,
                         "new_source_complete_cycles": int(native.status.eq("COMPLETE").sum()),
                         "new_source_open_cycles": int(native.status.ne("COMPLETE").sum()),
                         "original_core_entry_cycles": len(old_entry), "combined_core_entry_cycles": len(new_entry),
                         "original_entry_dates_not_present": len(old_entry - new_entry),
                         "additional_core_entry_dates": len(new_entry - old_entry),
                         "comparison_role": "两个真实账户的资金来源归因，非同份额或提前退出反事实。"})
    table("实际账户损益与来源差分解", pd.DataFrame(rows))
    table("组合新增来源全部实际交易", pd.DataFrame(native_rows))
    table("原核心入场日期逐例对照", pd.DataFrame(core_rows))
    return rows, pd.DataFrame(native_rows)


def plot():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex="col")
    for column, period in enumerate(PERIODS):
        for mode, label, color in [("A_SAVED_WEIGHT", "原A", "#2463a6"),
                                   ("DIRECTIONAL_ONLY", "方向确认单独", "#7b8795"),
                                   ("A_PLUS_DIRECTIONAL", "原A加方向确认", "#d46b32")]:
            d = account(period, "STRESS", mode)["daily"]
            axes[0, column].plot(d.date, d.equity / 200000., label=label, color=color, lw=1.25)
            axes[1, column].plot(d.date, 100 * d.drawdown, label=label, color=color, lw=1.1)
        axes[0, column].set_title("2015—2019" if column == 0 else "2020—2026-09-30")
        axes[0, column].legend(fontsize=9, frameon=False)
        axes[1, column].xaxis.set_major_locator(mdates.YearLocator(1 if column == 0 else 2))
        axes[1, column].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for row in range(2):
            axes[row, column].grid(alpha=.2)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("全账户净值（初始=1）")
    axes[1, 0].set_ylabel("从账户高点回撤（%）")
    fig.suptitle("5%方向确认完整交易：次数增加，净收益与夏普下降", fontsize=15)
    fig.text(.5, .01, "20万元 · 压力成本 · 原风险预算 · 252日年化 · 含全部现金日 · 历史开发结果", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .03, 1, .95])
    fig.savefig(OUT / "完整账户净值与回撤.png", dpi=160)
    plt.close(fig)


def record_facts(headline):
    link = "../reports/research/510300_point_directional_confirmation_study_v1/研究结果与下一步.md"
    for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        path = ROOT / "docs" / name
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        require("## 2026-10-04：方向确认完整账户实际失败（TECH.R153—R154）" not in text, "本次决策已记录，不重复追加。")
        banner = f"> 技术线最新实际金融裁决：{headline} [完整结果]({link})。\n"
        lines = text.splitlines(keepends=True)
        lines.insert(2, banner + "\n")
        appendix = f"\n\n## 2026-10-04：方向确认完整账户实际失败（TECH.R153—R154）\n\n{headline}\n\n"
        appendix += "**假设**：原5%上涨确认后至反向确认之间仍有净延续优势，能补原A空仓并提高完整账户。\n\n"
        appendix += "**验证方法**：只一个已有阈值完整规则，确认日决定/次开入、反向确认/次开出；0模型拟合/训练标签，不加MACD、量、波动过滤或2R。原A四账户精确重放、6必要测试；单独及加A两用途×两时期×两费用共8新账户，原资金与风险、252日、现金日和股息保持。\n\n"
        appendix += "**结果**：四场景整体经济门和历史稳定性门失败；新增实际账户与资金来源差已还原。原确认前极值没有注入成交日期。\n\n"
        appendix += "**接受/拒绝原因**：接受原定义不同完整用途及真实账户检验，拒绝本固定规则改善收益夏普的主张；事件时钟正确和交易增多不建立净优势。\n\n"
        appendix += "**是否需要重新验证**：固定5%确认规则结束，不改阈值、过滤、退出、组合、时期或费用救回。历史仍开发；独立验证、去过拟合和完整目标未达，原前瞻与全部旧失败保持。\n\n"
        appendix += f"依据：[唯一协议]({link.replace('研究结果与下一步.md', 'protocol.json')})、[实际结果]({link.replace('研究结果与下一步.md', 'summary.json')})、[完整报告]({link})。\n"
        new = "".join(lines) + appendix
        require(path.read_bytes() == raw, "共享文档记录期间改变，停止覆盖。")
        path.write_text(new, encoding="utf-8")


def run():
    require(not (OUT / "fact_recording_receipt.json").exists(), "本次结果已经整理，不重跑。")
    result = read(OUT / "summary.json")
    require(not result["all_period_cost_economic_gates_passed"], "此固定整理器仅处理本次已失败的实际结果。")
    rows, native = decomposition()
    comparison = pd.read_csv(OUT / "results/完整账户共同口径比较.csv")
    events = pd.read_parquet(OUT / "results/全部方向变化与确认事件.parquet")
    native_unique = int(native.event_id.nunique())
    plot()
    lines = ["# 5%方向确认：完整账户结果与下一步", "", "本轮真正运行8份新候选账户，固定规则失败。原A四份对照精确一致，6必要测试通过，0收益模型拟合/新训练标签；确认之后的交易次数增加，完整账户收益与夏普下降。", "",
             "数据3488日，2012-05-28至2026-09-30；原两个时期分别以20万元现金启动，原风险预算、252日、基础/压力成本保持。未将过去低点或高点当作可交易日期。", "",
             "| 时期 | 成本 | 用途 | 净年化 | 净夏普 | 最大回撤 | 完整周期 | 实际净pB |", "|---|---|---|---:|---:|---:|---:|---:|"]
    names = {"A_SAVED_WEIGHT": "原A", "DIRECTIONAL_ONLY": "方向确认单独", "A_PLUS_DIRECTIONAL": "原A加方向确认"}
    for r in comparison.itertuples(index=False):
        lines.append(f"| {r.period} | {r.cost} | {names[r.policy]} | {pct(r.net_cagr)} | {number(r.net_sharpe)} | {pct(r.max_drawdown)} | {r.completed_cycles} | {number(r.p_times_b)} |")
    lines += ["", f"全3488日有{int(events.entry_event.sum())}个向上确认，包含正式账户开始前的事件；不能全算新交易。八账户中的组合来源共有{native_unique}个不同事件身份，费用副本不是独立样本。", "",
              "| 时期/费用 | 期末权益差 | 新来源含末端持仓净贡献 | 原核心来源净贡献差 | 原核心入场日期未保留数 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['period']}/{r['cost']} | {r['ending_equity_delta']:,.2f}元 | {r['new_source_marked_net_pnl']:,.2f}元 | {r['core_source_marked_net_pnl_delta']:,.2f}元 | {r['original_entry_dates_not_present']} |")
    lines += ["", "净差严格等于实际份额下毛损益差减佣金/滑点差。新来源完整及未完成持仓与核心来源的标记净损益也严格还原全账户。核心入场日期未保留说明合并后的实际路径不同；不能把表中金额冒充被错过交易的同份额反事实利润。", "",
              "方向确认是一种价格事件时钟。所读[原论文](https://repository.essex.ac.uk/33750/1/s10462-022-10307-0.pdf)的实验为10分钟外汇、多阈值与优化算法，与本次日线510300单规则不同。文献和确认时钟正确均不证明本策略盈利。", "",
              "固定20/252日配对区块各2000次的四场景区间保存在原summary；经济与稳定性门均失败。当前历史全是开发资料，独立验证NOT_ESTABLISHED、全搜索DSR/PBO NOT_COMPUTED、去过拟合未完成。", "",
              "原A在当前技术线较早压力1.84%/0.438/净pB0.611、近期3.99%/1.217/1.073继续作为历史开发对照。本固定5%规则关闭，不改3%/7%、追加MACD/量/波动过滤、提前退出、改组合或选年份救回。下一实质不同完整机制尚未登记，新准入待跑0，目标active未完成。", "",
              "冻结前曾把对照回执status错读成passed，登记失败及随后缺协议调用均在候选运行前结束，候选账户0；修正真实字段后才冻结，一次完成8账户。原失败源码和registration_fix_receipt保留，交易规则未因收益修改。", "",
              "直接文件：[唯一协议](protocol.json)、[6测试](tests_receipt.json)、[原A精确对照](control_preflight.json)、[实际结果](summary.json)、[完整账户比较](results/完整账户共同口径比较.csv)、[实际资金与库存核对](results/实际账户资金库存核对.csv)、[来源损益差](results/实际账户损益与来源差分解.csv)、[核心入场逐例](results/原核心入场日期逐例对照.csv)。", "",
              "![压力账户净值与回撤](完整账户净值与回撤.png)"]
    (OUT / "研究结果与下一步.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    early = comparison.query("period=='2015_2019' and cost=='STRESS' and policy=='A_PLUS_DIRECTIONAL'").iloc[0]
    recent = comparison.query("period=='2020_2026' and cost=='STRESS' and policy=='A_PLUS_DIRECTIONAL'").iloc[0]
    headline = (f"2026-10-04已完成TECH.R153—R154原5%方向确认的不同完整账户用途：1固定规则、4原A精确对照、8新账户、6必要测试，0模型拟合/训练标签。全日线61向上确认；较早压力周期22→{int(early.completed_cycles)}、净年化1.84%→{pct(early.net_cagr)}、夏普0.438→{number(early.net_sharpe)}、净pB{number(early.p_times_b)}；近期32→{int(recent.completed_cycles)}、3.99%→{pct(recent.net_cagr)}、1.217→{number(recent.net_sharpe)}、净pB{number(recent.p_times_b)}。四场景经济及稳定性门均失败，固定规则关闭，不调阈值或加过滤救回。独立验证/去过拟合及目标未达，目标active，新准入待跑0；原前瞻和全部旧失败保持。")
    state_path = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
    raw = state_path.read_bytes()
    state = json.loads(raw.decode("utf-8"))
    protected = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session", "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at", "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check", "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation"]
    before = {k: state.get(k) for k in protected}
    (OUT / "state_before_recording_R154.json").write_bytes(raw)
    state.update(updated_at=now(), status="research_active", goal_status="active", latest_technical_decision="TECH.R154", latest_actual_model_decision="TECH.R154", latest_model_decision="TECH.R154", latest_financial_strategy_decision="TECH.R154", current_study=result["study"], current_phase="COMPLETED_FIXED_DIRECTIONAL_CONFIRMATION_ACCOUNT_REJECTION", latest_result=str((OUT / "summary.json").relative_to(ROOT)), latest_report=str((OUT / "研究结果与下一步.md").relative_to(ROOT)), latest_financial_strategy_result=str((OUT / "summary.json").relative_to(ROOT)), goal_turn_classification="progress", previous_goal_turn_classification="progress", blocked_audit_count=0, blocking_decision=None, live_own_process_handle=None, verified_wait=False, new_accounts_this_continuation=8, new_accounts_in_current_phase=8, new_investment_account_evaluations_this_continuation=8, necessary_tests_passed_this_continuation=6, saved_account_controls_replayed_this_continuation=4, return_and_sharpe_changed_this_continuation=True, return_and_sharpe_improved_this_continuation=False, current_full_account_economic_gate_passed=False, current_structural_signal_events=int(events.entry_event.sum()), current_executed_unique_new_entry_events=native_unique, current_admitted_unrun_numeric_candidates=0, goal_achieved=False, overfit_removed=False, latest_continuation_outcome=headline, latest_continuation_receipt=str((OUT / "summary.json").relative_to(ROOT)), next_available_action="该固定5%完整账户规则已关闭；不同完整技术机制尚未登记，无已准入待跑新候选，不改阈值/过滤/退出营救。")
    state["current_goal_turn_actual_work"] = {"new_accounts": 8, "original_control_replays": 4, "necessary_tests_passed": 6, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "new_financial_rules_tested": 1}
    require({k: state.get(k) for k in protected} == before, "前瞻保护值改变。")
    require(state_path.read_bytes() == raw, "共享状态在结果记录期间改变。")
    save_json(state_path, state)
    record_facts(headline)
    save_json(OUT / "event_account_decomposition.json", {"at": now(), "rows": rows, "combined_unique_event_ids": native_unique})
    save_json(OUT / "next_research_route.json", {"at": now(), "latest_financial_decision": "TECH.R154", "closed_fixed_rule": "UP_FIVE_PERCENT_CONFIRMATION_DOWN_FIVE_PERCENT_EXIT", "current_admitted_unrun_numeric_candidates": 0, "next_status": "PROPOSED_DIFFERENT_COMPLETE_TECHNICAL_MECHANISM_NOT_REGISTERED_NOT_RUN", "financial_goal_achieved": False})
    save_json(OUT / "fact_recording_receipt.json", {"at": now(), "headline": headline, "latest_actual_financial_decision": "TECH.R154", "preserved_forward_values": before, "new_candidate_accounts": 8, "goal_status": "active", "goal_achieved": False})
    print(headline)


if __name__ == "__main__":
    run()
