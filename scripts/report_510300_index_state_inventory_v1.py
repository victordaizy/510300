"""从保存的纯指数结果生成中文结论与图，并同步当前研究范围。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_index_state_inventory_daily_v1"
NAMES = {"HISTORY": "共同历史均值", "PRICE_DOWN_ONLY": "价格状态，仅下跌后增仓",
         "PRICE": "价格状态，全部阶段", "MACRO": "价格状态＋信贷与资金利差", "BUY_HOLD": "买入持有"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cycles(ledger):
    result, active = [], None
    last_shares, last_equity = 0, 200000.
    for row in ledger.itertuples():
        if last_shares == 0 and row.shares > 0:
            active = {"entry_date": row.date, "starting_equity": last_equity}
        if active is not None and row.shares == 0:
            result.append({**active, "exit_date": row.date, "equity_pnl": row.equity - active["starting_equity"]})
            active = None
        last_shares, last_equity = row.shares, row.equity
    return pd.DataFrame(result), active


def main():
    result, verification = read(OUT / "result.json"), read(OUT / "verification.json")
    metrics = {m["policy"]: m for m in result["all_account_metrics"] if m["cost"] == "STRESS"}
    ledgers = {policy: pd.read_parquet(OUT / "accounts/STRESS" / f"{policy}_ledger.parquet") for policy in NAMES}
    rolling = pd.read_parquet(OUT / "results/all_rolling_two_years.parquet")
    decisions = pd.read_parquet(OUT / "accounts/STRESS/MACRO_decisions.parquet")
    predictions = pd.read_parquet(OUT / "results/predictions.parquet")
    models = read(OUT / "results/saved_models.json")
    macro_models = [m for m in models if m["model"] == "MACRO"]
    macro_split_days = sum(any(index >= 3 for index in m["tree"]["feature"] if index >= 0) for m in macro_models)
    root_macro_days = sum(m["tree"]["feature"][0] >= 3 for m in macro_models)
    pred_metrics = {m["model"]: m for m in result["prediction_metrics"]}
    primary = result["primary"]
    comparison = result["account_increments"][0]
    accounts_pass = [m["policy"] for m in metrics.values() if m["historical_point_targets_met"] and m["policy"] != "BUY_HOLD"]
    windows_pass = int(rolling.loc[rolling.policy.ne("BUY_HOLD"), "joint_targets"].sum())
    complete, unfinished = cycles(ledgers["MACRO"])
    complete.to_parquet(OUT / "results/descriptive_inventory_cycles.parquet", index=False)
    pnl = complete.equity_pnl if len(complete) else pd.Series(dtype=float)
    positive = pnl[pnl > 0]
    negative = -pnl[pnl < 0]
    interpretation = {"complete_inventory_cycles": len(complete), "unfinished_cycle": None if unfinished is None else str(unfinished["entry_date"]),
                      "cycle_win_rate": float((pnl > 0).mean()) if len(pnl) else None,
                      "average_winning_cycle_equity_change": float(positive.mean()) if len(positive) else None,
                      "average_losing_cycle_equity_change": float(negative.mean()) if len(negative) else None,
                      "cycle_values_include_contemporaneous_dividend_receivable_accounting": True,
                      "macro_split_days": macro_split_days, "macro_root_split_days": root_macro_days,
                      "macro_MSE_relative_change_vs_price": pred_metrics["MACRO"]["MSE"] / pred_metrics["PRICE"]["MSE"] - 1,
                      "stress_historical_candidates": accounts_pass, "stress_rolling_two_year_joint_pass": windows_pass,
                      "new_models_or_accounts": 0, "goal_achieved": False}
    save(OUT / "saved_interpretation.json", interpretation)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.3), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"HISTORY": "#8C939B", "PRICE_DOWN_ONLY": "#2B9283", "PRICE": "#377CC1", "MACRO": "#C65736"}
    for policy, color in colors.items():
        ledger = ledgers[policy]
        axes[0].plot(ledger.date, ledger.equity / 10000, color=color, label=NAMES[policy], lw=1.7 if policy == "MACRO" else 1.15)
        axes[1].plot(ledger.date, ledger.drawdown * 100, color=color, lw=1.2)
    axes[0].set_title("510300纯指数账户｜最近两年训练，每日更新，压力成本")
    axes[0].set_ylabel("完整账户权益（万元）")
    axes[0].axhline(20, ls="--", color="#555555", lw=.7)
    axes[0].legend(fontsize=9, loc="lower left", ncol=2)
    axes[1].set_ylabel("从峰值回撤（%）")
    axes[1].axhline(10, ls="--", color="#555555", lw=.8)
    axes[1].invert_yaxis()
    for axis in axes:
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.085, .015, "四组采用相同50%仓位上限和尾部预算；图中为已反复研究的历史，无独立验证。", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(OUT / "纯指数压力账户净值与回撤.png", dpi=160)
    plt.close(fig)
    p = primary
    lines = [
        "当前范围只有510300指数ETF和现金。按用户最新要求停止期权后，已经完成本轮纯指数研究，没有找到符合目标的策略。",
        "",
        f"主压力账户从20万元变为{p['ending_equity']:,.2f}元，成本后夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、最大回撤{p['max_drawdown_magnitude']:.2%}。目标仍为夏普至少1.2、年化至少10%、最大回撤不超过10%，三项必须同时达到。",
        "",
        f"本轮实际完成{result['daily_distribution_updates']:,}次逐日条件分布更新，其中{result['new_tree_fits']:,}次树拟合，覆盖{result['unique_prediction_days']:,}个不同预测日，以及10条基础/压力完整账户。每一天的训练范围严格为最近两个日历年，标签退出开盘必须早于当天；没有用最后样本日预先卖出。",
        "", "压力成本下的同覆盖比较：", "",
    ]
    for policy in ["HISTORY", "PRICE_DOWN_ONLY", "PRICE", "MACRO"]:
        m = metrics[policy]
        lines.append(f"- {NAMES[policy]}：夏普{m['net_sharpe']:.3f}，年化{m['annualized_return']:.2%}，最大回撤{m['max_drawdown_magnitude']:.2%}。")
    bh = metrics["BUY_HOLD"]
    lines.extend([
        "", f"全仓买入持有作为市场基准，压力夏普{bh['net_sharpe']:.3f}、年化{bh['annualized_return']:.2%}、最大回撤{bh['max_drawdown_magnitude']:.2%}。该基准不受策略50%仓位上限及主动尾部预算约束，因此回撤差异不能直接当择时优势。",
        "", f"加入宏观后的年化算术收益相对价格模型增加{comparison['annual_arithmetic_increment'] * 100:.3f}个百分点，20日区块95%区间[{comparison['ci95'][0] * 100:.3f}, {comparison['ci95'][1] * 100:.3f}]个百分点。宏观增量资格为{result['macro_increment_gate_met']}；点值少亏不能证明可靠优势。",
        "", f"宏观模型的五日收益预测MSE较价格模型变化{interpretation['macro_MSE_relative_change_vs_price']:.2%}。{len(macro_models):,}个宏观日模型中，{macro_split_days:,}个至少有一次资金变量分裂，{root_macro_days:,}个根节点使用资金变量。这只能说明模型使用了它们，不能证明这些分裂是因果机制。",
        "", f"四个研究账户共有{windows_pass}个滚动两年窗口同时达到三项目标。每个模型{pred_metrics['MACRO']['n']:,}个成熟五日预测相互重叠，不能当同等数量独立交易或宏观公告。主模型固定五日相位的尾部预警为{result['tail_monitor_alerts'].get('MACRO', 0)}次；预警次数不等于收益规律通过检验。",
        "",
        "三个价格变量分别是此前五日的标准化抛压、二十日趋势和短长实现波动率之比。新增资料只有已公布的社融存量同比三个月变化、DR007相对政策利率的差。两项均用实际已有可用时钟，不把它们称为中性利率缺口或NFCI；期权报价、隐波、权利金和合约均不参与本轮模型、覆盖条件或账户。",
        "",
        "每次最多四个状态，每个叶至少126个历史标签；状态条件、样本索引和分布统计均保留。使用完整叶分布估计收益、波动和下侧尾部，再枚举合法100份目标仓位。PRICE_DOWN_ONLY与PRICE复用同一预测，只比较是否限制为下跌后增仓；它不改变旧冻结研究。",
        "",
        "账户最高持仓50%；五日条件ES95预算为权益2.5%，额外保留10%标的跳空情景的5%权益预算及距90%权益峰值余量的一半。每日可减持，买入后遵守T+1，涨跌停可拒绝成交。压力费率每侧万分之四、最低5元，加每侧千分之一滑点和不利价格单位取整。风险预算是事前限制，不是最大回撤保证。",
        "", f"主账户平均敞口{p['mean_exposure']:.2%}，累计成本{p['cost_cny']:,.2f}元，最差单日{p['worst_day']:.2%}，日收益偏度{p['daily_skew']:.3f}。末端持仓={p['terminal_position']}，末端清算储备{p['terminal_reserve_cny']:.2f}元。风险控制降低了暴露，也降低了收益容量，不能把低波动直接当成高夏普。",
        "", f"保存的{verification['saved_distributions_recomputed']:,}次分布已复算，两个历史截断和未来标签扰动检查通过；10个账户的现金、持仓、分红应收、费用与T+1份额相符，新增买入逐笔符合事前尾部预算。检查用于确认实现，没有产生新预测或新账户。",
        "",
        "本轮在同一已见历史上进行了有限的状态与资金增量比较。结果仍为未达标，不能继续据这份结果改树深、换窗口、选择最好阶段或放大仓位来宣告高夏普。当前最需要的是能够在相同形态和风险条件下区分后续路径的独立信息证据；没有建立这项增量之前，不把每日更新或状态复杂度当作优势。",
        "",
        "共同历史截至2026-08-14；没有新增行情采集，当前市场观点仍为NO_VIEW。期权后续研究已撤回，四腿程序未运行。研究源码、协议、账本和保存预测在本目录，未制作审核ZIP或用户汇总表格。",
    ])
    (OUT / "研究结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    mandate.update(current_round=STUDY if (STUDY := result["study_id"]) else "", latest_integrated_experiment=STUDY,
                   current_protocol=(OUT / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(OUT / "result.json").relative_to(ROOT).as_posix(),
                   last_research_result=f"纯指数两年每日状态研究完成：主压力夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、回撤{p['max_drawdown_magnitude']:.2%}；目标未达。",
                   goal_achieved=False)
    save(mandate_path, mandate)
    state_path = ROOT / "reports/research/510300_integrated_research_continuation_20260924/current_status.json"
    state = read(state_path)
    completed = state.get("completed_followup_studies", [])
    if STUDY not in completed:
        completed.append(STUDY)
    state.update(updated_at=timestamp, completed_followup_studies=completed, latest_user_requested_study=OUT.relative_to(ROOT).as_posix(),
                 latest_user_requested_study_status=result["status"], latest_research_workflow="INDEX_ONLY_FIXED_STATE_EXPERIMENT_COMPLETED_NO_QUALIFIED_STRATEGY",
                 latest_index_primary=p, latest_index_distribution_updates=result["daily_distribution_updates"],
                 latest_index_new_accounts=10, latest_goal_turn_classification="PROGRESS_INDEX_ONLY_STATE_COMPARISON_COMPLETED",
                 same_condition_consecutive_no_progress_goal_turns=0, goal_achieved=False,
                 remaining_research_question="纯指数状态模型和既有宏观资料仍未提供可靠正期望；需有别于已失败用途的独立信息证据，不能凭改参数宣布高夏普。")
    save(state_path, state)
    report = state_path.parent / "最新研究结论.md"
    paragraph = (f"2026-09-25最新纯指数进展：已按用户‘不做期权了，我只做指数’恢复510300与现金范围。"
                 f"新增{result['daily_distribution_updates']:,}次两年每日条件分布更新、10条完整账户；主压力夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、回撤{p['max_drawdown_magnitude']:.2%}，高夏普目标仍未完成。"
                 f"宏观相对价格增量区间[{comparison['ci95'][0] * 100:.3f}, {comparison['ci95'][1] * 100:.3f}]个百分点。"
                 "详见[本轮纯指数结论](../510300_index_state_inventory_daily_v1/研究结论.md)。本轮未使用期权资料或资产，期权新实验停止。\n\n---\n\n")
    if not report.read_text(encoding="utf-8").startswith("2026-09-25最新纯指数进展"):
        report.write_text(paragraph + report.read_text(encoding="utf-8"), encoding="utf-8")
    print("纯指数中文结论、净值图和最新研究状态已保存，目标保持未完成。")


if __name__ == "__main__":
    main()
