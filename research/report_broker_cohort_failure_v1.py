"""复算已保存账户、解释额外退出并交付图文；不重新模拟政策。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import broker_cohort_failure_study_v1 as study

ROOT, OUT = study.ROOT, study.OUT
POLICIES = (study.CONTROL_A, study.CONTROL_STAGE, *study.candidate.POLICIES)
COLORS = dict(zip(POLICIES, ("#414b5f", "#8593b0", "#167765", "#c58136", "#9c5880")))


def number(value, pattern=".5f"):
    return format(value, pattern) if value is not None and pd.notna(value) else "UNKNOWN"


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("本用途图文已交付，不重复复算或追加。")
    summary = study.read(OUT / "summary.json")
    metrics = pd.read_parquet(OUT / "results/二十完整账户_同资金风险成本比较.parquet")
    accounts, saved_checks, information_checks, decompositions, source_rows, contexts, figures = {}, [], [], [], [], [], []
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for period in study.parent.PERIODS:
        for cost in study.parent.COSTS:
            fig, axes = plt.subplots(2, 1, figsize=(12.7, 7.8), sharex=True,
                gridspec_kw={"height_ratios": [2, 1]})
            for policy in POLICIES:
                result = study.saved_account(period, cost, policy)
                accounts[(period, cost, policy)] = result
                stats, _ = study.statistics(result, period, cost, policy)
                saved = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost) & metrics.policy.eq(policy)].iloc[0]
                for key, value in stats.items():
                    if value is None:
                        if pd.notna(saved[key]):
                            raise AssertionError("未知指标被填成已知。")
                    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
                        np.testing.assert_allclose(value, saved[key], atol=1e-10, rtol=0, equal_nan=True)
                    elif isinstance(value, (bool, np.bool_)) and bool(value) != bool(saved[key]):
                        raise AssertionError("保存风险状态不一致。")
                checked = study.parent.verify_account(result)
                if policy in study.candidate.POLICIES:
                    saved_checks.append({"period": period, "cost": cost, "policy": policy, **checked})
                    # 保存源日期直接复核；不构造新信号或重新模拟账户。
                    check = result["failure_checks"]
                    if check.cycle_id.duplicated().any() or not check.relative_session.eq(5).all():
                        raise AssertionError("保存检查周期重复或时点改变。")
                    if len(check) and not (check.cohort_source_date.lt(check.entry_date).all()
                            and check.observation_source_date.lt(check.decision_date).all()):
                        raise AssertionError("保存成分源时钟不在决策之前。")
                    information_checks.append({"period": period, "cost": cost, "policy": policy,
                        "checks": len(check), "known": int(check.cohort_view_allowed.sum()),
                        "unknown": int((~check.cohort_view_allowed).sum()), "extra_requests": int(check.extra_exit.sum())})
                gross = float(stats["gross_at_actual_quantities_pnl"])
                friction = float(stats["total_commission"] + stats["total_slippage"])
                decompositions.append({"period": period, "cost": cost, "policy": policy,
                    "gross_pnl_cny": gross, "friction_cny": friction,
                    "net_pnl_cny": float(stats["ending_equity"] - 200000),
                    "gross_no_friction_cagr_fixed_actual_quantities": (1 + gross / 200000) ** (252 / len(result["daily"])) - 1,
                    "counterfactual_role": "固定实际份额与时点的费用解释，不是另一可执行策略",
                    "open_pnl_cny": stats["open_pnl_cny"], "unpaid_dividend_cny": stats["unpaid_dividend_cny"],
                    "rejections": len(result["rejections"]) if "rejections" in result else None,
                    "risk_stopped": stats["account_risk_stopped"]})
                daily = result["daily"]
                axes[0].plot(daily.date, daily.equity / 200000, label=study.NAMES[policy],
                    color=COLORS[policy], linewidth=1.7 if policy == study.candidate.POLICIES[0] else 1.25,
                    linestyle="--" if policy == study.candidate.POLICIES[1] else "-")
                axes[1].plot(daily.date, -100 * daily.drawdown, color=COLORS[policy], linewidth=1.1,
                    linestyle="--" if policy == study.candidate.POLICIES[1] else "-")
            axes[0].legend(loc="upper left", ncol=2, fontsize=9)
            axes[0].set_ylabel("完整净值 / 初始20万元")
            axes[1].set_ylabel("账户回撤 / %")
            for axis in axes:
                axis.grid(axis="y", alpha=.22)
            axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.suptitle(f"{period} · {cost}：固定组失效与两个价格对照、原阶段及原A\n全账户、全部现金日和摩擦；终点未平仓自然计价", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .94))
            path = OUT / "figures" / f"{period}_{cost}_二十万完整账户.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=140)
            plt.close(fig)
            figures.append(study.relative(path))
            primary = accounts[(period, cost, study.candidate.POLICIES[0])]
            old = accounts[(period, cost, study.CONTROL_STAGE)]["trades"]
            for source, group in primary["trades"].groupby("source"):
                source_rows.append({"period": period, "cost": cost, "source": source,
                    **study.parent.measurements.trade_statistics(group)})
            context = primary["failure_checks"].merge(primary["trades"][["cycle_id", "status", "exit_date", "exit_reason", "net_return", "net_pnl"]],
                on="cycle_id", how="left", validate="one_to_one")
            old_map = old.set_index("entry_date")
            for row in context.to_dict("records"):
                old_cycle = old_map.loc[row["entry_date"]] if row["entry_date"] in old_map.index else None
                row.update(period=period, cost=cost, same_entry_in_R212=old_cycle is not None,
                    original_same_entry_status=old_cycle["status"] if old_cycle is not None else None,
                    original_same_entry_exit=old_cycle["exit_date"] if old_cycle is not None else pd.NaT,
                    original_same_entry_net_return=old_cycle["net_return"] if old_cycle is not None else np.nan,
                    original_same_entry_net_pnl=old_cycle["net_pnl"] if old_cycle is not None else np.nan,
                    original_same_entry_source=old_cycle["source"] if old_cycle is not None else None)
                contexts.append(row)
    study.table("十二保存账户交付复算", pd.DataFrame(saved_checks))
    study.table("十二保存检查时序与未知复核", pd.DataFrame(information_checks))
    study.table("二十账户实际数量毛收益与摩擦分解", pd.DataFrame(decompositions))
    study.table("主政策全部进入来源_完成与未完成分开", pd.DataFrame(source_rows))
    context_frame = pd.DataFrame(contexts)
    study.table("主政策全部第五日检查与原同进入上下文", context_frame)
    write_report(summary, metrics, pd.DataFrame(decompositions), pd.DataFrame(source_rows), context_frame, figures, accounts)
    study.write(OUT / "delivery_receipt.json", {"at": study.parent.original.now(),
        "status": "SAVED_ACCOUNTS_RECOMPUTED_REPORT_AND_FIGURES_WRITTEN",
        "saved_accounts_recomputed": len(saved_checks), "all_saved_metric_rows_recomputed": len(metrics),
        "saved_information_check_accounts": len(information_checks), "figures": figures,
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "goal_achieved": False})
    print("12保存账户、20指标行及全部第五日记录复算通过；4图和完整结论已写入，没有重跑政策。", flush=True)


def write_report(summary, metrics, decomposition, sources, contexts, figures, accounts):
    passed = sum(row["economic_passed"] for row in summary["gates"])
    lines = ["# 真实进入后的固定组与价格失效：完整金融结果", "",
        "2026-10-05，TECH.R215登记、TECH.R216结果。八必要测试通过；原R212四账户五账表及终态复现，登记前另一次核验因无列空表序列化类型差异终止，记录保留；只规范核对双方0行0列的索引，没有改变账户。12新账户一次运行，20行指标与12保存新账户复算。原A仅读取之前核验保存账户，本轮没有重跑原A。", "",
        f"裁决：{summary['status']}，四经济门{passed}/4，历史稳定门{'通过' if summary['historical_stability_passed'] else '失败'}；独立NOT_ESTABLISHED、去过拟合未建立，提高完整收益夏普目标未实现。全部历史是开发/校准，五日来自已知R214结果，不能称事前盲测。", "",
        "## 比较口径与全部结果", "",
        "每个时期20万元，510300.SH/CASH_CNY、原三路线进入和宏观/周线角色、原风险与整手/T+1/股息规则。主政策只在实际买入后的第5个观察收盘检查一次；两固定组均未过半上涨且ETF同源区间总回报非正才新增退出，次合法开盘执行；原退出优先，未知保留原规则且不重试。分组依据实际进入前一源日，不能把原R214信号锚点13亏损直接算为本政策节省。", "",
        "同覆盖ETF对照保留同成分源资格而只看ETF价格；全日期ETF对照忽略成分覆盖。三个政策都不依据未来结果改变进入，较早退出产生不同资金占用和后续进入，所以完整账户比较才是主证据。", "",
        "| 时期 | 费用 | 政策 | 净年化 | 全日历夏普 | 最大回撤 | 完成 | 胜/负 | B | pB | 标准净期望 | 完整年均次数 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {study.NAMES[row.policy]} | {number(row.net_cagr,'.4%')} | {number(row.net_sharpe)} | {number(row.max_drawdown,'.4%')} | {row.completed_cycles} | {row.wins}/{row.losses} | {number(row.payoff)} | {number(row.p_times_b)} | {number(row.standard_expectancy_loss_units)} | {row.average_full_year_cycles:.3f} |")
    lines += ["", "B为实际完成净周期的平均盈利/平均亏损绝对值，p为已完成胜率；标准亏损单位期望pB−(1−p)。用户pB>1保留为更严格要求；计划2R、毛收益或零亏损时无穷大不能替代。完整年2015—2019/2020—2025，2026单列部分年；所有空仓日进入252年化与夏普。", "",
        "## 成分信息的增量与全部不确定性", "",
        "| 时期/费用 | 对照 | 年化差百分点 | 夏普差 | 块长 | 年化差95%百分点 | 夏普差95% |",
        "|---|---|---:|---:|---:|---|---|"]
    for comparison in summary["comparisons"]:
        for item in comparison["intervals"]:
            c, s = item["cagr_delta_95"], item["sharpe_delta_95"]
            lines.append(f"| {comparison['period']}/{comparison['cost']} | {study.NAMES[comparison['control']]} | {100*comparison['cagr_delta']:.5f} | {comparison['sharpe_delta']:.5f} | {item['block']} | [{100*c[0]:.5f}, {100*c[1]:.5f}] | [{s[0]:.5f}, {s[1]:.5f}] |")
    lines += ["", "每尺度2000次，固定种子510300154。两尺度全部保留，区间是历史描述，不修正全项目多重选择，也不产生独立证据。主政策严格高于同覆盖ETF对照才可能支持成分方向增量；同覆盖与全日期对照的差别仍包含覆盖资格和资金路径，不能全部归因于组方向。", "",
        "## 第五日检查、实际退出与反例", "",
        "| 时期/费用 | 政策 | 检查 | 源已知/未知 | 新请求 | 实际额外卖单 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in summary["timing_checks"]:
        lines.append(f"| {row['period']}/{row['cost']} | {study.NAMES[row['policy']]} | {row['checks']} | {row['known_checks']}/{row['unknown_checks']} | {row['extra_requests']} | {row['actual_extra_sell_orders']} |")
    lines += ["", "这里只检查实际仍持有至固定日的周期；此前已经退出不能用于额外政策收益。execution_date列是下一计划开盘，真实执行看订单；涨跌停等原约束继续作用。未知不是看空或退出。", "",
        "| 主政策压力进入 | 决策/可用源日 | 两组上涨下界 | 同源ETF回报 | 新退出/原退出 | 本周期净回报 | 原同进入净回报 |",
        "|---|---|---|---:|---|---:|---:|"]
    extra = contexts.loc[contexts.cost.eq("STRESS") & contexts.extra_exit]
    for row in extra.itertuples(index=False):
        old_exit = str(row.original_same_entry_exit.date()) if pd.notna(row.original_same_entry_exit) else "UNKNOWN"
        lines.append(f"| {row.entry_date.date()} | {row.decision_date.date()}/{row.observation_source_date.date()} | {number(row.leaders_positive_lower,'.2%')}/{number(row.followers_positive_lower,'.2%')} | {number(row.etf_lagged_five_return,'.3%')} | {row.exit_date.date() if pd.notna(row.exit_date) else '未完成'}/{old_exit} | {number(row.net_return,'.3%')} | {number(row.original_same_entry_net_return,'.3%')} |")
    lines += ["", "同进入对比只用于说明；份额、部分减仓、佣金和后续账户路径可能不同，不能将逐笔回报差相加当作全账户收益。无原同日进入时明确未知。原13亏损上下文与本次新交易数不同。", "",
        "## 2024具体上涨、量价指标与持有", ""]
    for policy in (study.CONTROL_STAGE, *study.candidate.POLICIES):
        trades = accounts[("2020_2026", "STRESS", policy)]["trades"]
        chosen = trades.loc[trades.entry_origin.eq(pd.Timestamp("2024-09-24"))]
        if len(chosen):
            row = chosen.iloc[0]
            lines.append(f"- {study.NAMES[policy]}：2024-09-24原放量重新定价信号，{row.entry_date.date()}实际进入，{row.exit_date.date() if pd.notna(row.exit_date) else '自然未完成'}退出，净周期回报{number(row.net_return,'.4%')}，净损益{number(row.net_pnl,'.2f')}元。")
        else:
            lines.append(f"- {study.NAMES[policy]}：原2024-09-24信号未形成同一实际进入，保留该资金占用差异。")
    lines += ["", "原启动解释：ETF收盘突破前20日收盘高、成交量/前20中位至少1.5、日MACD柱转强构成可见重新定价；慢宏观未被要求同步转强，后续完整周柱、融资与上一完整周结构位决定晋级和失效。成分扩散只在滞后源钟上可见，后来多数上行不能回填原开盘。2024局部成功仍不能证明其他年份或整体有效。2019修复、2020恢复及2015未知详见R214保留的全部原案例，不依据本轮结果改锚点或单独选赢家。", "",
        "## 信号、摩擦、账户约束与未完成", "",
        "| 压力时期/政策 | 实际数量毛损益 | 摩擦 | 净损益 | 加回摩擦年化 | 开仓净损益 | 拒单/停买 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in decomposition.loc[decomposition.cost.eq("STRESS")].itertuples(index=False):
        lines.append(f"| {row.period}/{study.NAMES[row.policy]} | {row.gross_pnl_cny:.2f}元 | {row.friction_cny:.2f}元 | {row.net_pnl_cny:.2f}元 | {row.gross_no_friction_cagr_fixed_actual_quantities:.4%} | {number(row.open_pnl_cny,'.2f')}元 | {number(row.rejections,'.0f')}/{row.risk_stopped} |")
    lines += ["", "费用解释保持已经执行的份额、时点及风险，不能直接当成无费策略。原A未保存open_pnl_cny，显式UNKNOWN；完整净值含自然持仓。原阶段/新政策的应收与未完成均保留，不末日强平。净周期回报与账户回撤分开。", ""]
    for row in metrics.loc[metrics.policy.eq(study.candidate.POLICIES[0]) & metrics.cost.eq("STRESS")].itertuples(index=False):
        lines.append(f"- {row.period}：平均暴露{row.mean_exposure:.2%}、最坏日{row.worst_day:.4%}、最坏完成周期{number(row.worst_trade,'.4%')}、最大盈利集中{number(row.largest_winner_fraction_of_all_wins,'.2%')}、最长空仓{row.longest_flat_sessions}日、原价双边成交额/初资{row.gross_turnover_over_initial_capital:.3f}、未完成{row.unfinished_cycles}、应收{row.unpaid_dividend_cny:.2f}元。")
    lines += ["", "## 决策边界与下一步", "",
        "本配置如果未通过，拒绝其晋升，不据个别时期改五日、覆盖、两组阈值、来源滞后、路线或费用。描述事实可保留，策略收益必须以全部四场景与完整对照为准。小资金成交规模和调仓灵活性是真实账户条件，不能补足不存在的预测优势。", "",
        "下一方向须先解释这个失效政策为什么起效或无效：成分是否只是重复ETF自身价格、首次五日判断是否已错过原退出、是否误伤后来恢复、来源未知是否主导。之后才能提出实质不同的信息/机制与完整对照；行业主线与宽基传导仍需先解决分类版本和来源钟，不能事后拼接产业赢家。当前无已准入待跑的新用途，独立前瞻13字段和E03计划保持。", "",
        "源：原3488技术宏观观察、历史成员/已分类修复回报及覆盖；成员与回报末日2026-08-14，缺口不前填。首次供应商实际可得/首版NOT_CERTIFIED；用保守源钟不等于完成首版认证。0新拟合、标签或行情请求。", "",
        "[冻结协议](protocol.json) · [全部结果与区间](summary.json) · [20账户完整指标](results/二十完整账户_同资金风险成本比较.csv) · [逐年统计](results/全部逐年实际完成周期与账户收益.csv) · [全部第五日记录](results/全部实际周期第五日未知与额外请求.csv) · [全部同进入上下文](results/主政策全部第五日检查与原同进入上下文.csv) · [毛收益费用](results/二十账户实际数量毛收益与摩擦分解.csv) · [R214原上涨图文](../510300_broker_fixed_cohort_propagation_v1/固定组传播_全部结果与下一实验.md)"]
    for path in figures:
        lines += ["", f"![完整账户](figures/{path.split('/')[-1]})"]
    (OUT / "真实进入后固定组失效_完整结果与反例.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
