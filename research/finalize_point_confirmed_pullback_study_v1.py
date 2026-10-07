"""记录完整技术点位的实际失败、交易空间和账户损益；保存结果不重跑。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import upward_episode_anatomy_v1 as common
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.point_confirmed_pullback_study_v1 import OUT, CURRENT, WEIGHT, CONTROLS, PERIODS
from research.adaptive_allocation_v1 import normalize_dividends

STATE_DIR = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
MARKER = "2026-10-04：完整缩量回调点位实验实际失败（TECH.R149—R150）"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_table(name, frame):
    frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
    frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def plots(data, trade):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    colors = {"A_SAVED_WEIGHT": "#2563a5", "PULLBACK_ONLY": "#a8a29e", "A_PLUS_PULLBACK": "#c05a35"}
    names = {"A_SAVED_WEIGHT": "原A", "PULLBACK_ONLY": "新点位单独", "A_PLUS_PULLBACK": "原A加新点位"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", constrained_layout=True)
    for column, period in enumerate(PERIODS):
        for mode in names:
            path = CONTROLS / period / "STRESS/A_SAVED_WEIGHT/daily.parquet" if mode == "A_SAVED_WEIGHT" else (
                OUT / "results/accounts" / period / "STRESS" / mode / "daily.parquet")
            frame = pd.read_parquet(path)
            axes[0, column].plot(frame.date, frame.equity / 200000., color=colors[mode], label=names[mode], lw=1.5)
            axes[1, column].plot(frame.date, frame.drawdown * 100., color=colors[mode], lw=1.2)
        axes[0, column].set_title("2015—2019" if period == "2015_2019" else "2020—2026年9月30日")
        axes[0, column].set_ylabel("完整账户净值")
        axes[1, column].set_ylabel("距账户高点的回撤（%）")
        axes[0, column].legend(loc="upper left", frameon=False)
        for axis in axes[:, column]:
            axis.grid(alpha=.18)
    fig.suptitle("固定缩量回调点位没有改善原A：相同20万元、风险与压力成本")
    fig.savefig(OUT / "完整账户净值与回撤.png", dpi=170)
    plt.close(fig)
    part = data.loc[data.date.between(pd.Timestamp(trade.entry_date) - pd.Timedelta(days=45),
                                      pd.Timestamp(trade.exit_date) + pd.Timedelta(days=14))].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1, 1.2], constrained_layout=True)
    axes[0].plot(part.date, part.ac, color="#2563a5", lw=1.5, label="含当时已生效股息的前向平移价")
    axes[0].hlines(trade.target_index, trade.entry_origin, trade.exit_date, color="#16836a", ls="--", label="入场前锁定的前高目标")
    axes[0].hlines(trade.stop_index, trade.entry_origin, trade.exit_date, color="#c05a35", ls="--", label="入场前锁定的结构失效")
    axes[0].axvspan(trade.entry_date, trade.exit_date, alpha=.07, color="#2563a5")
    entry = data.loc[data.date.eq(trade.entry_date)].iloc[0]
    exit_row = data.loc[data.date.eq(trade.exit_date)].iloc[0]
    axes[0].scatter([trade.entry_date, trade.exit_date],
                    [trade.entry_raw + entry.cash_shift, exit_row.open + exit_row.cash_shift],
                    marker="o", color=["#16836a", "#c05a35"], zorder=4)
    axes[0].annotate("次日开盘入场", (trade.entry_date, trade.entry_raw + entry.cash_shift), xytext=(7, -22), textcoords="offset points")
    axes[0].annotate("确认失效后下一开盘退出", (trade.exit_date, exit_row.open + exit_row.cash_shift), xytext=(-155, -22), textcoords="offset points")
    axes[0].set_ylabel("前向平移价")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].bar(part.date, part.amount / 1e8, color="#9aa5b1", width=1.2)
    axes[1].set_ylabel("成交额（亿元）")
    axes[2].bar(part.date, part.daily_hist, color=np.where(part.daily_hist >= 0, "#16836a", "#c05a35"), width=1.2, label="日MACD柱")
    axes[2].plot(part.date, part.weekly_hist, color="#526a94", label="此前完整周MACD柱")
    axes[2].axhline(0, color="#777777", lw=.6)
    axes[2].set_ylabel("MACD背景")
    axes[2].legend(loc="upper right", frameon=False)
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle("唯一新增点位：计划净空间约2.07R，最终结构失效；MACD与量仅作原背景")
    fig.savefig(OUT / "唯一新增点位_价格量能与MACD.png", dpi=170)
    plt.close(fig)


def run():
    require(not (OUT / "fact_recording_receipt.json").exists(), "本实际裁决已记录，不重复。")
    result = read(OUT / "summary.json")
    require(result["status"] == "REJECTED_FIXED_CONFIRMED_PULLBACK_FULL_ACCOUNT_GATE_FAILED", "须按实际金融结果改写报告。")
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    all_signals = pd.read_parquet(OUT / "results/全部回调阶段与首次转强事件.parquet")
    records, origins, funnels, bridges = [], [], [], []
    for period, (start, end) in PERIODS.items():
        subset = all_signals.loc[all_signals.date.between(start, end)]
        counts = subset.event_status.value_counts().to_dict()
        funnels.append({"period": period, "original_sessions": len(subset), "status_counts": counts,
                        "first_turns_evaluated": sum(counts.get(key, 0) for key in ["STRUCTURAL_ENTRY_EVENT", "REMAINING_ROOM_BELOW_TWO_R", "NO_PRIOR_AMOUNT_CONTRACTION"]),
                        "structural_events": int(subset.entry_event.sum())})
        for cost in ["BASE", "STRESS"]:
            for mode in ["PULLBACK_ONLY", "A_PLUS_PULLBACK"]:
                directory = OUT / "results/accounts" / period / cost / mode
                trades = pd.read_parquet(directory / "trades.parquet")
                native = trades.loc[trades.source.eq("PULLBACK")].copy()
                if len(native):
                    native.insert(0, "period", period)
                    native.insert(1, "cost", cost)
                    native.insert(2, "policy", mode)
                    records.append(native)
                decisions = pd.read_parquet(directory / "decisions.parquet")
                actual = decisions.loc[decisions.entry_event].copy()
                actual.insert(0, "period", period)
                actual.insert(1, "cost", cost)
                actual.insert(2, "policy", mode)
                origins.append(actual)
            base = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost) & metrics.policy.eq("A_SAVED_WEIGHT")].iloc[0]
            combo = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost) & metrics.policy.eq("A_PLUS_PULLBACK")].iloc[0]
            trades = pd.read_parquet(OUT / "results/accounts" / period / cost / "A_PLUS_PULLBACK/trades.parquet")
            core = trades.loc[trades.source.eq("CORE_WEIGHT")].reset_index(drop=True)
            saved = pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/trades.parquet")
            require(len(core) == len(saved), "本次核心周期身份需要另作解释。")
            pd.testing.assert_frame_equal(core[["entry_date", "exit_date"]], saved[["entry_date", "exit_date"]].reset_index(drop=True))
            native_pnl = float(trades.loc[trades.source.eq("PULLBACK"), "net_pnl"].sum())
            core_pnl_delta = float(core.net_pnl.sum() - saved.net_pnl.sum())
            gross_delta = float(combo.gross_at_actual_quantities_pnl - base.gross_at_actual_quantities_pnl)
            fee_delta = float(combo.total_commission + combo.total_slippage - base.total_commission - base.total_slippage)
            net_delta = float(combo.ending_equity - base.ending_equity)
            require(abs(net_delta - gross_delta + fee_delta) < 1e-6, "实际净损益桥不一致。")
            if period == "2020_2026":
                require(abs(native_pnl + core_pnl_delta - net_delta) < 1e-6, "新增及核心损益没有解释净值差。")
            bridges.append({"period": period, "cost": cost, "net_equity_delta_cny": net_delta,
                            "gross_pnl_delta_at_actual_quantities_cny": gross_delta,
                            "actual_fee_delta_cny": fee_delta, "new_native_cycle_net_pnl_cny": native_pnl,
                            "original_core_cycle_net_pnl_delta_cny": core_pnl_delta,
                            "core_original_entry_exit_dates_preserved": True,
                            "scope": "实际两账户路径，不是删除交易后的反事实策略"})
    native = pd.concat(records, ignore_index=True)
    save_table("全部新增实际点位_四账户用途副本非独立样本", native)
    save_table("全部结构事件_逐账户实际进入判定", pd.concat(origins, ignore_index=True))
    save_table("实际账户损益差分解", pd.DataFrame(bridges))
    write_json(OUT / "event_funnel_and_account_bridge.json", {"at": now(), "period_funnels": funnels,
               "account_bridges": bridges, "unique_executed_native_events": int(native.event_id.nunique()),
               "all_costs_and_uses_are_same_event_copies": True}, exclusive=True)
    prices = pd.read_parquet(CURRENT / "inputs/candidate_prices.parquet")
    dividends = normalize_dividends(pd.read_csv(WEIGHT / "inputs/dividends.csv"))
    data, _ = common.features(prices, dividends)
    trade = native.loc[native.period.eq("2020_2026") & native.cost.eq("STRESS") & native.policy.eq("A_PLUS_PULLBACK")].iloc[0]
    plots(data, trade)
    account_lines = ["| 原时期 | 压力成本方案 | 净年化 | 净夏普 | 最大回撤 | 完整周期 | 实际净pB | 完整年均次数 |",
                     "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.loc[metrics.cost.eq("STRESS")].itertuples(index=False):
        sharpe = f"{row.net_sharpe:.3f}" if np.isfinite(row.net_sharpe) else "不可估计"
        pb = f"{row.p_times_b:.3f}" if np.isfinite(row.p_times_b) else "不可估计"
        account_lines.append(f"| {row.period} | {row.policy} | {row.net_cagr:.2%} | {sharpe} | {row.max_drawdown:.2%} | "
                             f"{row.completed_cycles} | {pb} | {row.average_full_year_cycles:.2f} |")
    recent = next(row for row in bridges if row["period"] == "2020_2026" and row["cost"] == "STRESS")
    entry = data.loc[data.date.eq(trade.entry_date)].iloc[0]
    report = "\n".join([
        "# 完整技术点位实验：缩量回调后转强没有改善收益与夏普", "",
        "本轮已实际完成一个完整点位假说及8候选账户，结论为固定拒绝。它使近期A账户多完成1笔交易，"
        "但降低收益、夏普和实际净pB；早期没有新增交易。没有沿用公告来源或退出预测MSE作为本轮主评价。", "",
        "## 固定规则与旧研究的区别", "",
        "沿用原C04已知上涨段与回调阶段，前一日阶段平均成交额低于阶段前5日均额；"
        "首次收盘超过前日最高价时只判断一次。回调已知最低价减一刻度为失效，已确认前高收盘为目标，"
        "收盘计划与下一开盘实际报价的扣费计划收益风险比均至少2。失效/目标锁定，收盘确认后下一可卖开盘退出。", "",
        "旧供给测试使用跌破前20日支撑后收复、第二次回测的单日缩量缩幅及合成2R目标；"
        "旧力度研究使用2/13平滑力度；旧C04是加入原退出模型的字段函数。本轮研究首次入场事件和完整账户，"
        "这些旧合同及失败均保留，不改其参数或补算其失败账户。", "",
        "统一20万元、原50%上限/ES5/跳空/回撤风险、252日全日历夏普、现金比较率0、T+1和登记/应收/支付，"
        "两时期分别启动；两成本都完成。6项必要测试通过，原A四个账户的逐日金额、份额与订单精确复现；新账户资金库存恒等核对通过。", "",
        *account_lines, "",
        "近期基础成本同样恶化：原A净年化4.24%/夏普1.290，组合3.83%/1.107。"
        "早期两个费用场景完全相同，无法证明严格改善；单独账户0交易时夏普不可估计，1亏0赢时平均盈利和盈亏比不可估计，不能填0或无穷大。", "",
        "## 交易为何如此少", "",
        f"早期{funnels[0]['first_turns_evaluated']}个首次转强中，35个剩余原价空间不到2R、4个没有事前缩量，只剩1个结构事件；"
        "2018-12-12的原价计划比2.50，扣压力费用约1.74，收盘计划即取消。",
        f"近期{funnels[1]['first_turns_evaluated']}个首次转强中，29个空间不足、17个没有事前缩量，只剩2个结构事件。"
        "2024-10-18形成入场；2024-11-20事件发生时仍持有上一笔，按原协议忽略，不另计交易、不补仓换身份。", "",
        "以上转强数只属于可定义的原回调阶段，未知/非阶段日及重复转强分别保留。原全部3488日的结构事件共3个，"
        "正式两时期覆盖2855个原交易日；没有删掉失败触发后重选时间点。", "",
        "## 唯一新增交易的计划与实际", "",
        f"2024-10-18收盘触发，10月21日开盘原价{trade.entry_raw:.3f}入场；"
        f"入场时原价失效约{trade.stop_index-entry.cash_shift:.3f}、原价目标约{trade.target_index-entry.cash_shift:.3f}，"
        f"扣费计划比{trade.entry_net_reward_risk:.3f}。2025-01-13按结构失效退出，共{int(trade.holding_sessions)}个持有交易区间。"
        f"最终净回报{trade.net_return:.2%}，组合内这笔净损益{trade.net_pnl:,.2f}元。", "",
        "这笔交易中日线/完整周MACD和量价背景可逐日查看，但它们不是本轮新增过滤器。"
        "中间曾有浮盈只是事后路径，不能将最高收盘当作可事前选中的退出，也不能看过这一笔后追加保护阈值营救。", "",
        f"近期压力期末少{abs(recent['net_equity_delta_cny']):,.2f}元：实际份额毛损益少{abs(recent['gross_pnl_delta_at_actual_quantities_cny']):,.2f}元，"
        f"费用多{recent['actual_fee_delta_cny']:,.2f}元。新增交易净亏{abs(recent['new_native_cycle_net_pnl_cny']):,.2f}元，"
        f"其余原核心交易净损益再少{abs(recent['original_core_cycle_net_pnl_delta_cny']):,.2f}元。"
        "原A的32笔核心进出日期全部一致，不能把差异归因于被新仓位挤掉原A入场；损失改变了随后可用资本、风险余量和真实份额。", "",
        "## 不确定性及后续取舍", "",
        "预先固定20/252日循环区块、各2000次配对净收益/夏普增量区间均已保存。"
        "早期增量恒0，近期点值为负且描述区间跨0；区间不能把失败升级为支持，也没有解决历史反复使用或全局多重选择。", "",
        "本固定规则结束，不改2R、金额基准、确认根数、目标/止损、退出或组合方式营救。"
        "它只否定这一完整用途，不能从一笔亏损否定全部缩量回调、MACD或技术分析。"
        "后续高盈亏比研究应先证明事前的上涨延续或净优势，不能只用最近前高的几何距离代替概率。"
        "下一不同完整技术假说必须与旧失败有实质区别再冻结；当前没有已接纳的新数值候选，不宣称待跑模型已经就绪。", "",
        "原A/两前瞻登记不变，真实新账户日和完成点位仍0。本轮历史全部DEVELOPMENT_CALIBRATION，"
        "独立验证未成立、正式DSR/PBO未计算，去过拟合和完整收益夏普目标未完成。", "",
        "![完整账户净值与回撤](完整账户净值与回撤.png)", "",
        "![唯一新增点位的价格、成交额与MACD](唯一新增点位_价格量能与MACD.png)", "",
        "直接结果：[协议](protocol.json)、[实际裁决与区间](summary.json)、"
        "[完整账户比较](results/完整账户共同口径比较.csv)、[全部新增实际点位](results/全部新增实际点位_四账户用途副本非独立样本.csv)、"
        "[全部结构事件的实际判定](results/全部结构事件_逐账户实际进入判定.csv)、"
        "[账户损益差分解](results/实际账户损益差分解.csv)、[原A精确复现](control_preflight.json)、[六测试回执](tests_receipt.json)。", "",
    ])
    (OUT / "研究结果与下一步.md").write_text(report, encoding="utf-8")
    route = {"at": now(), "latest_actual_financial_decision": "TECH.R150",
             "closed_candidate": "CONFIRMED_PULLBACK_FIRST_TURN_FIXED_PRIOR_HIGH_TWO_R",
             "current_admitted_unrun_numeric_candidates": 0,
             "next_status": "PROPOSED_DIFFERENT_TECHNICAL_MECHANISM_NOT_REGISTERED_NOT_RUN",
             "next_question": "哪种事前可知的价格/量价延续机制能提供真实净优势，且不同于全部旧失败并改善完整账户？",
             "constraints": "本固定规则结束，不改空间/退出/组合救失败；下一假说先明确机制及旧用途，唯一冻结后直接账户比较。原技术/风险/独立验证与前瞻门保持。",
             "financial_goal_achieved": False}
    write_json(OUT / "next_research_route.json", route, exclusive=True)
    banner = ("2026-10-04已完成TECH.R149—R150唯一完整技术点位实验：原已确认上涨段缩量回调首次转强、原前高目标及扣费计划至少2R，"
              "原A精确对照4账户，实际新候选8账户，6必要测试通过，0学习拟合/新训练标签/联网。3488日只有3结构事件；"
              "早期1个被扣费空间取消，近期2个只有1笔进入、另一笔发生在持仓中。近期压力原A净年化3.99%/夏普1.217/净pB1.073，"
              "加候选降为3.58%/1.036/0.927，周期32→33，期末少6609.02元；早期账户完全相同。唯一新交易2024-10-21至2025-01-13、59区间，"
              "净回报−5.67%、组合净亏4767.87元。原32核心进出日期全同，剩余差来自资本/风险余量/份额，不是挤掉原入场或单纯费用。"
              "整体REJECTED_FIXED，不换参数/目标/退出营救。原R145退出增量及所有失败保持；下一实质不同技术机制未登记，已准入待跑候选0。"
              "独立验证、去过拟合及完整目标未达，目标active；原前瞻12状态值不改。")
    section = "\n\n## " + MARKER + "\n\n" + banner + "\n\n" + (
        "**假设**：事前缩量回调后首次转强及固定目标空间，能形成净优势并补原A空仓。\n\n"
        "**验证方法**：唯一完整规则，原两时期/两成本、20万元/252日/原风险与执行；单独及加A八账户，原A四账户精确复现。"
        "计划与下一开盘扣费比均>=2，目标失效锁定，收盘确认下一开盘退出；六必要时钟/现金测试，资金库存核对通过。\n\n"
        "**结果**：早期没有新增交易；近期唯一新交易亏损、净收益夏普与净pB降低，主门四场景均失败。"
        "20/252日各2000配对区间保存，原32核心进出日期全同，净损益桥精确解释期末差。费用档和单独/组合是同一个事件副本，不算独立样本。\n\n"
        "**接受/拒绝原因**：接受执行、事件稀缺与实际账户差事实；拒绝本固定规则改善收益夏普或增次质量的主张。"
        "原前高空间不能代替上涨延续概率；0交易/没有盈利的盈亏比不可估计，不填0或无穷。\n\n"
        "**是否需要重新验证**：本固定完整规则结束，不改确认、阶段、金额基准、2R、目标失效、退出、成本或组合救失败。"
        "后续不同技术机制另立唯一合同；历史仍开发，独立验证与去过拟合未完成。原C04/A04/供给/力度/NR7及学习退出拒绝保持。\n\n"
        "直接证据：[研究结果](../reports/research/510300_point_confirmed_pullback_study_v1/研究结果与下一步.md)、"
        "[实际裁决](../reports/research/510300_point_confirmed_pullback_study_v1/summary.json)、"
        "[完整账户](../reports/research/510300_point_confirmed_pullback_study_v1/results/完整账户共同口径比较.csv)、"
        "[唯一规则](../reports/research/510300_point_confirmed_pullback_study_v1/protocol.json)。\n")
    for relative in ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md", "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        path = ROOT / relative
        before = path.read_bytes()
        text = before.decode("utf-8-sig")
        require(MARKER not in text, "裁决条目已存在。")
        lines = text.splitlines()
        indices = [i for i, line in enumerate(lines) if line.startswith("> 技术线纠偏更新：")]
        require(len(indices) == 1, "本任务顶部事实位置改变，请先读取。")
        lines[indices[0]] = "> 技术线最新完整点位实验：" + banner
        require(path.read_bytes() == before, "共享文档正在被其他任务更新。")
        path.write_text("\n".join(lines).rstrip() + section, encoding="utf-8")
    state_path = STATE_DIR / "state.json"
    before = state_path.read_bytes()
    (OUT / "state_before_recording_R150.json").write_bytes(before)
    state = json.loads(before.decode("utf-8-sig"))
    protected_keys = [key for key in state if "forward" in key or "prospective" in key or key in [
        "earliest_future_exchange_session", "next_new_close_eligible_at", "registered_candidate_intents", "accepted_independent_candidates", "current_validated_candidates"]]
    protected = {key: state[key] for key in protected_keys}
    state.update({"updated_at": now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
                  "goal_tool_status_confirmed": "active", "previous_goal_turn_classification": state.get("goal_turn_classification"),
                  "goal_turn_classification": "progress", "blocked_audit_count": 0,
                  "latest_completed_study": result["study"], "current_study": result["study"],
                  "latest_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "latest_report": (OUT / "研究结果与下一步.md").relative_to(ROOT).as_posix(),
                  "latest_research_status": result["status"], "latest_progress": banner,
                  "latest_technical_decision": "TECH.R150", "latest_actual_model_decision": "TECH.R150",
                  "latest_model_decision": "TECH.R150", "latest_actual_prediction_model_decision": "TECH.R145",
                  "latest_financial_strategy_decision": "TECH.R150",
                  "current_phase": "COMPLETED_FIXED_TECHNICAL_PULLBACK_ACCOUNT_REJECTION",
                  "current_direction": "本固定完整点位拒绝；下一不同技术机制尚未登记。",
                  "next_research_action": (OUT / "next_research_route.json").relative_to(ROOT).as_posix(),
                  "next_experiment_status": route["next_status"], "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY",
                  "new_model_fits_this_continuation": 0, "new_candidate_coefficient_estimations_this_continuation": 0,
                  "new_accounts_this_continuation": 8, "new_strategy_accounts_this_continuation": 8,
                  "new_accounts_in_current_phase": 8, "new_investment_account_evaluations_this_continuation": 8,
                  "saved_account_controls_replayed_this_continuation": 4, "saved_original_account_metrics_quoted": 4,
                  "necessary_tests_passed_in_current_phase": 6, "new_model_training_labels_this_continuation": 0,
                  "new_stock_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
                  "current_candidate_trials": {"candidate_configurations": 1, "complete_account_uses": 2,
                                               "periods": 2, "costs": 2, "actual_candidate_accounts": 8,
                                               "original_control_replays": 4, "new_model_fits": 0,
                                               "independent_validation": "NOT_ESTABLISHED"},
                  "current_study_prediction_gate_status": "NOT_APPLICABLE_FULL_TECHNICAL_ACCOUNT_TEST",
                  "current_economic_stage_status": "COMPLETED_FIXED_ACCOUNT_GATE_FAILED",
                  "current_full_account_economic_gate_passed": False,
                  "account_return_sharpe_this_continuation": "COMPUTED_AND_FIXED_INCREMENT_REJECTED",
                  "return_and_sharpe_changed_this_continuation": True, "return_and_sharpe_improved_this_continuation": False,
                  "current_executed_unique_new_entry_events": 1, "current_structural_signal_events": 3,
                  "live_own_process_handle": None, "verified_wait": False, "overfit_removed": False})
    require({key: state[key] for key in protected_keys} == protected and state_path.read_bytes() == before,
            "前瞻值或当前任务状态发生并发变化。")
    write_json(state_path, state)
    write_json(OUT / "fact_recording_receipt.json", {"at": now(), "status": "RECORDED_ACTUAL_FULL_ACCOUNT_R150_REJECTION",
               "actual_candidate_accounts": 8, "control_replays": 4, "necessary_tests_passed": 6,
               "unique_native_trade_events": 1, "latest_actual_prediction_model_decision": "TECH.R145",
               "protected_forward_values_preserved": protected, "goal_status": "active",
               "new_strategy_validated": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}, exclusive=True)
    print("实际八账户、事件漏斗、唯一交易与损益差已记录；四份事实文档已接续，原冻结策略与前瞻保持。")


if __name__ == "__main__":
    run()
