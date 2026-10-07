"""还原已保存真实点位、资金与原A重合，并报告顺序规则的有限结论。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.point_fresh_repair_order_study_v1 import OUT, CONTROL, PERIODS, COSTS, rules, load, read, write_json, digest, now, table
from research.point_account_nr7_inputs_v1 import metrics, PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def number(value, percent=False):
    if value is None or not np.isfinite(float(value)):
        return "不可估计"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def main():
    require_path = OUT / "saved_result_verification.json"
    if require_path.exists():
        raise ValueError("保存账户与顺序点位核对已经完成。")
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    for source in protocol["sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("冻结来源改变："+source["path"])
    data, _, _, parents = load()
    sequence = pd.read_parquet(OUT / "results/全部原点的当时已知修复顺序.parquet")
    pd.testing.assert_frame_equal(sequence, rules.signals(data), check_exact=True)
    for date in ["2019-01-09", "2020-04-07", "2024-09-24"]:
        i = int(np.flatnonzero(data.date.eq(date))[0])
        prefix = rules.signals(data.iloc[:i+1])
        pd.testing.assert_frame_equal(sequence.iloc[:i+1].reset_index(drop=True), prefix, check_exact=True)
    stats = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    points, checks, concentrations, accounts = [], [], [], {}
    for period in PERIODS:
        old_daily = pd.read_parquet(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet").set_index("date")
        old_target = parents[period].set_index("origin")[PARENT_A]
        for cost in COSTS:
            for policy in rules.POLICIES:
                path = OUT / f"results/accounts/{period}/{cost}/{policy}"
                account = {name: pd.read_parquet(path / f"{name}.parquet") for name in ["daily", "orders", "trades", "decisions", "rejections"]}
                account["terminal"] = read(path / "terminal.json")
                accounts[(period, cost, policy)] = account
                check = verify_account(account)
                checks.append({"period": period, "cost": cost, "policy": policy, **check})
                actual = metrics(account)
                saved = stats.loc[stats.period.eq(period) & stats.cost.eq(cost) & stats.policy.eq(policy)].iloc[0]
                for name in ["net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff", "p_times_b", "ending_equity"]:
                    np.testing.assert_allclose(actual[name], saved[name], rtol=1e-12, atol=1e-8, equal_nan=True)
        primary = accounts[(period, "STRESS", "FRESH_ORDERED_REPAIR")]
        completed = primary["trades"].loc[primary["trades"].status.eq("COMPLETE")]
        profits = completed.loc[completed.net_pnl.gt(0), "net_pnl"]
        net = float(completed.net_pnl.sum())
        largest = float(profits.max()) if len(profits) else np.nan
        concentrations.append({"period": period, "completed_cycles": len(completed), "winning_cycles": len(profits),
                               "net_pnl_cny": net, "largest_winner_cny": largest,
                               "largest_winner_divided_by_total_net_pnl": largest/net if net > 0 else np.nan,
                               "new_account_after_removing_trade_computed": False})
        for trade in completed.itertuples(index=False):
            i = int(np.flatnonzero(data.date.eq(trade.entry_origin))[0])
            r, x = sequence.iloc[i], data.iloc[i]
            start, v, m = int(r.known_under_start), int(r.current_volume_positive_start), int(r.current_macd_positive_start)
            if not (r.ordered_repair and 0 <= start <= v < m < i < trade.entry_idx):
                raise ValueError("实际点位没有当时成立的量→MACD→价格顺序。")
            if not (data.up_volume_balance5.iloc[v-1] <= 0 and data.daily_hist.iloc[m-1] <= 0):
                raise ValueError("当前正段没有真实已知出生。")
            exit_orders = primary["orders"].loc[primary["orders"].cycle_id.eq(trade.cycle_id) & primary["orders"].date.eq(trade.exit_date)]
            exit_origin = exit_orders.origin.iloc[0]
            k = int(np.flatnonzero(data.date.eq(exit_origin))[0])
            if trade.exit_reason == "REPAIR_PRICE_CONFIRMATION_FAILED_CLOSE" and data.ac.iloc[k] > data.ema20.iloc[k]:
                raise ValueError("实际价格失效退出没有已知收盘确认。")
            prior = old_daily.loc[trade.entry_origin]
            points.append({"period": period, "cycle_id": trade.cycle_id,
                           "known_below_ema_since": data.date.iloc[start], "current_volume_positive_since": data.date.iloc[v],
                           "current_macd_positive_since": data.date.iloc[m], "origin": trade.entry_origin,
                           "entry_date": trade.entry_date, "exit_decision_origin": exit_origin, "exit_date": trade.exit_date,
                           "holding_sessions": trade.holding_sessions, "net_return": trade.net_return, "net_pnl_cny": trade.net_pnl,
                           "entry_raw": trade.entry_raw, "initial_nominal_exposure": trade.entry_quantity*trade.entry_raw/trade.entry_equity,
                           "relative_volume": x.relative_volume, "rv_ratio": x.rv_ratio, "weekly_macd_hist": x.weekly_hist,
                           "previous_complete_week": x.weekly_last_date, "A_shares_at_origin": int(prior.shares),
                           "A_exposure_at_origin": float(prior.exposure), "A_known_target_at_origin": float(old_target.loc[trade.entry_origin]),
                           "hypothetical_combined_account_computed": False})
    points = pd.DataFrame(points)
    if len(points) != 9:
        raise ValueError("保存实际点位数与原完成交易不同。")
    table("全部实际点位_量MACD价格顺序及A持仓", points)
    table("实际利润集中度", pd.DataFrame(concentrations))
    maximum_error = max(item["maximum_accounting_error"] for item in checks)
    overlap = {period: {"points": len(group), "A_already_held": int(group.A_shares_at_origin.gt(0).sum()),
                       "A_was_flat": int(group.A_shares_at_origin.eq(0).sum())} for period, group in points.groupby("period")}
    write_json(require_path, {"at": now(), "status": "PASS_SAVED_SEQUENCE_POINT_AND_EIGHT_ACCOUNT_RECOMPUTATION",
               "unchanged_frozen_sources": len(protocol["sources"]), "all_origin_sequence_rows": len(sequence),
               "three_case_prefix_checks": 3, "actual_complete_points": len(points), "saved_accounts_checked": 8,
               "maximum_accounting_error_cny": maximum_error, "original_A_position_context": overlap,
               "new_fits_or_account_runs": 0, "independent_validation": "NOT_ESTABLISHED"}, exclusive=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for ax, (origin, left, right, title) in zip(axes, [("2020-04-07", "2020-03-15", "2020-08-01", "2020：识别了早期修复，但实际退出后还有后段上涨"),
                                                     ("2024-09-24", "2024-09-01", "2024-12-05", "2024：早段确认的实际进出与回撤路径")]):
        part = data.loc[data.date.between(left, right)]
        ax.plot(part.date, part.close, color="#2e485e", linewidth=1.6, label="原始日收盘")
        ax.plot(part.date, part.ema20-part.cash_shift, color="#bc852f", linewidth=1.2, label="EMA20（同原价单位）")
        p = points.loc[points.origin.eq(origin)].iloc[0]
        native = accounts[("2020_2026", "STRESS", "FRESH_ORDERED_REPAIR")]["orders"]
        orders = native.loc[native.cycle_id.eq(p.cycle_id)]
        for side, marker, color, label in [("BUY", "^", "#b43738", "实际次开买入"), ("SELL", "v", "#23806a", "实际次开卖出/风险减仓")]:
            subset = orders.loc[orders.side.eq(side)]
            ax.scatter(subset.date, subset.fill_price, marker=marker, color=color, s=42, label=label, zorder=4)
        ax.axvspan(p.entry_date, p.exit_date, color="#d8b350", alpha=.09)
        ax.set_title(title+f"｜该笔净回报{p.net_return:+.2%}")
        ax.set_ylabel("价格 / 元")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.savefig(OUT / "具体上涨解释与实际可成交点位.png", dpi=170)
    plt.close(fig)

    lines = ["# 从具体上涨反推：量→MACD→价格确认的完整结果", "",
             "这条固定顺序比单纯价格上穿得到更好的历史点位质量，但交易很少、盈利集中，完整账户没有同时超过原A的收益与夏普。TECH.R161拒绝该固定完整政策替代A，不否定有限顺序解释，也不宣称整个技术分析无效。", "",
             "## 原假设与可识别时钟", "",
             "2019/2020/2024案例依次出现当前量方向正段、当前日MACD正段和价格上穿EMA20。事后低点不是规则起点：只用当时已经发生的连续均线下区间，并且量和MACD正段必须由此前真实非正值转入、当前仍未失效。严格要求量先于MACD、MACD先于价格；没有同日/倒序的补救设置。", "",
             "同一完整交易逻辑：空仓时次真实开盘尝试进场；买后价格收盘重新<=当日EMA20，次合法开盘退出，风险只减仓。不使用固定2R、20日到期、波动或放量筛选。周线仅作此前完整周背景，沿用原130周预热；没有拟合或新训练标签。", "",
             "## 真实完整账户", "",
             "各时期初始20万元、510300与现金、252日年化/现金0，原风险上限与T+1/100份/.001刻度/最低佣金5元；两费用全部保留。原A四账户日净值/份额/订单精确一致，八个新政策账户一次运行；频率是软目标。", "",
             "| 时期 | 费用 | 政策 | 净年化 | 净夏普 | 回撤 | 完成次数 | 实际胜率 | 实际B | 实际pB |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"A_SAVED_WEIGHT": "原A", "FRESH_ORDERED_REPAIR": "当前修复顺序", "PRICE_CONFIRMATION": "纯价格上穿"}
    for r in stats.itertuples(index=False):
        lines.append(f"| {r.period} | {r.cost} | {names[r.policy]} | {number(r.net_cagr, True)} | {number(r.net_sharpe)} | {number(r.max_drawdown, True)} | {r.completed_cycles} | {number(r.win_rate, True)} | {number(r.payoff)} | {number(r.p_times_b)} |")
    lines.extend(["", "顺序规则相对纯价格政策的收益/夏普历史点值四场景均提高，实际pB四场景均>1；但是较早净年化低于A，近期净年化和夏普均低于A。全部整体门和历史稳定性门未通过，不能只以对较弱价格对照的改进宣布完成目标。20/252日各2000配对区间全部保存在summary.json。", "",
                  "## 全部九个真实点位", "",
                  "| 量当前正段开始 | 日MACD当前正段开始 | 价格确认 | 次开入场 | 退出 | 持有区间 | 实际净回报 | A当时持仓 |", "|---|---|---|---|---|---:|---:|---|"])
    for r in points.itertuples(index=False):
        lines.append(f"| {r.current_volume_positive_since:%Y-%m-%d} | {r.current_macd_positive_since:%Y-%m-%d} | {r.origin:%Y-%m-%d} | {r.entry_date:%Y-%m-%d} | {r.exit_date:%Y-%m-%d} | {r.holding_sessions} | {r.net_return:+.2%} | {'已持有' if r.A_shares_at_origin else '空仓'} |")
    lines.extend(["", "这些是压力费用的九个实际周期；基础/压力、原20日结果标签和原上涨段不是更多独立交易。2020年案例虽然整个事后上涨段涨38.54%，该固定规则2020-04-08进、04-14出，实际−0.45%。识别早段改善并不保证持有到后段加速。2019和2024真实赢家分别+17.89%和+15.62%，也不是从事后低点买到高点的39.13%/36.68%。", "",
                  "![具体上涨与实际点位](具体上涨解释与实际可成交点位.png)", "",
                  "## 盈亏比好看但为何还不能接受", "",
                  "较早只有2笔、1盈1亏；近期7笔仅1盈6亏，实际胜率14.29%。近期pB2.513来自平均赢家约为平均亏损17.59倍，标准亏损单位期望1.656；高pB与高胜率是不同事实。由于仅一个赢家，不能把B17.59当作稳定未来分布。", ""])
    for c in concentrations:
        lines.append(f"{c['period']}最大的唯一赢家{c['largest_winner_cny']:,.2f}元，占全段净损益{c['largest_winner_divided_by_total_net_pnl']:.2%}；其他交易合计抵减盈利。这是原真实交易损益比例，不是移除赢家后重跑的新账户。")
    lines.append("")
    for period, value in overlap.items():
        lines.append(f"{period}的{value['points']}个信号原点，原A已持有{value['A_already_held']}个、空仓{value['A_was_flat']}个。重合事实只描述已有库存；未将两账户收益相加，未计算或授权新的组合政策。")
    lines.extend(["", "顺序规则的完整年平均完成次数较早0.4、近期1.0；没有账户回撤停机，不是交易次数被年度配额限制。它过滤了很多价格上穿，但不足以提供足够连续的账户收益。", "",
                  "## 旧20日标签与真实交易必须分开", "",
                  "原186成熟价格事件中固定顺序只选9个；原20日标签整体正回报比例55.56%，平均正/负幅度2.724、标签pB1.513。2020—2023的3个标签pB仅0.802。实际价格失效退出后的近期胜率则为14.29%。旧固定终点标签不是可执行收盘成交、实际pB或策略夏普；完整账户已直接运行，没有用原退出MSE或该诊断门阻止。", "",
                  "## 处置与下一问题", "",
                  "保留当前修复顺序能从案例变为当时可识别规则、其有限历史点位质量及集中度事实；拒绝该完整独立政策已经改善原A的收益夏普。固定规则关闭，不改变顺序/EMA/MACD/量窗口/退出、加过滤/阈值、改成本或选年份营救。", "",
                  "下一问题回到具体路径：早段修复与后段加速之间发生了什么、哪些再次确认在当时可知、原A是否已经覆盖这些时段。先解释全体成功与失败路径，再判断是否存在不同完整机制；不能仅因2020这一个上涨案例退出过早而调长R161持有期，或因两笔赢家改仓位。下一金融实验尚未登记。", "",
                  f"五项必要测试通过，八保存账户与全部3488原点顺序、三案例前缀和九实际点位时钟已核对，最大资金误差{maximum_error:.3g}元，冻结来源{len(protocol['sources'])}未改。报告阶段0新账户/拟合。", "",
                  "全部已用历史为DEVELOPMENT_CALIBRATION，价量与股息供应历史首版未认证；global DSR/PBO仍NOT_COMPUTED，独立验证/去过拟合未成立，完整目标未达。原失败与前瞻保持，0实盘授权。", "",
                  "[完整账户比较](results/完整账户共同口径比较.csv) · [全部实际点位与A库存](results/全部实际点位_量MACD价格顺序及A持仓.csv) · [逐年次数及净回报](results/逐年净收益与实际交易次数.csv) · [利润集中度](results/实际利润集中度.csv) · [唯一协议](protocol.json) · [实际裁决](summary.json) · [保存结果核对](saved_result_verification.json)", ""])
    (OUT / "研究结果与下一步.md").write_text("\n".join(lines), encoding="utf-8")
    print("已还原全部九个实际点位、原A持仓重合及利润集中；八账户/3488原点保存结果核对通过，无新金融运行。", flush=True)


if __name__ == "__main__":
    main()
