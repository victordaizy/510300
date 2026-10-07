"""解释已完成的支持信息账户，不重跑金融、不修改冻结输入或动作。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import point_account_nr7_inputs_v1 as measurements
from research import support_price_acceptance_acceptance_v1_0_1 as acceptance
from research import support_price_acceptance_study_v1 as study

ROOT, OUT, SOURCE = study.ROOT, acceptance.OUT, study.OUT
PRIMARY = study.inputs.POLICIES[0]
REPORT = OUT / "支持信息价格接受_完整账户结果与持仓失败解释.md"
COLORS = {PRIMARY: "#186a9c", study.CONTROL_A: "#965520", study.CONTROL_STAGE: "#848c94",
          study.inputs.POLICIES[1]: "#b8454c", study.inputs.POLICIES[2]: "#63814d"}
CONTEXT = ["daily_hist", "weekly_hist", "log_relative_volume", "funding_known", "funding_gap_pp",
           "funding_gap_change5", "margin_known", "financing_net_change5", "orders_known",
           "pmi_orders_level", "pmi_orders_change", "industry_view_allowed", "industry_positive5_fraction",
           "leader_mean_return5", "industry_source_age_days", "support_information_arrived",
           "support_new_node_ids", "reverse_operation_arrived", "reverse_node_ids",
           "announced_target_confirmation_ids", "support_activation_allowed", "support_entry_event",
           "support_observation_reason", "support_setup_date", "support_watch_alive_after_decision"]


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def table(name: str, frame: pd.DataFrame) -> None:
    path = OUT / "results" / name
    if path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists():
        raise RuntimeError("描述表已存在，不覆盖本次完整结果。")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def check_frozen() -> dict:
    original = study.read(SOURCE / "protocol.json")
    comparison = study.read(OUT / "protocol.json")
    for item in [*original["sources"], *comparison["files"]]:
        if study.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结输入、配置或已完成账户改变，停止描述归档。")
    return {"original_frozen_inputs_exact": len(original["sources"]),
            "saved_comparison_files_exact": len(comparison["files"]),
            "new_account_replays": 0, "new_fits": 0, "new_market_requests": 0}


def date_label(value) -> str:
    return "未知" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def number(value, digits=4, percent=False) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return f"{float(value):.{digits}%}" if percent else f"{float(value):.{digits}f}"


def markdown(frame: pd.DataFrame, columns: dict[str, str], percent=(), dates=()) -> str:
    rows = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for record in frame.to_dict("records"):
        values = []
        for key in columns:
            value = record[key]
            if key in dates:
                displayed = date_label(value)
            elif key in percent:
                displayed = number(value, 4, True)
            elif isinstance(value, (float, np.floating)):
                displayed = number(value)
            elif pd.isna(value):
                displayed = "UNKNOWN"
            elif isinstance(value, (bool, np.bool_)):
                displayed = "是" if value else "否"
            else:
                displayed = str(value)
            values.append(displayed.replace("|", " / ").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def holding_description(observed: pd.DataFrame, accounts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = observed.set_index("date", verify_integrity=True)
    cycles, contexts = [], []
    for period in study.PERIODS:
        account = accounts[(period, "STRESS", PRIMARY)]
        for trade in account["trades"].to_dict("records"):
            if trade["status"] != "COMPLETE":
                raise ValueError("本主账户存在未完成周期，不能把开放损益当完整收益。")
            entry, promotion, exit_date = map(pd.Timestamp, (trade["entry_date"], trade["promotion_date"], trade["exit_date"]))
            if pd.isna(promotion):
                raise ValueError("实际主账户周期未晋级，描述需保留未确认分支。")
            row = indexed.loc[promotion]
            last = pd.Timestamp(row.complete_week_last)
            week_start = last.to_period("W-FRI").start_time.normalize()
            week = observed.loc[observed.date.between(week_start, last)]
            before_count = int(week.date.lt(entry).sum())
            after_count = int(week.date.ge(entry).sum())
            prior_holding = int(account["daily"].date.ge(entry).mul(account["daily"].date.lt(promotion)).sum())
            learned = observed.loc[observed.date.ge(entry) & observed.date.le(promotion)]
            holding = observed.loc[observed.date.ge(entry) & observed.date.lt(exit_date)]
            cycles.append({"period": period, **trade,
                "promotion_complete_week_first": week.date.iloc[0], "promotion_complete_week_last": last,
                "week_rows_before_actual_entry": before_count, "week_rows_on_or_after_entry": after_count,
                "actual_holding_closes_before_promotion": prior_holding,
                "week_information_is_fully_post_entry": before_count == 0,
                "new_support_ids_entry_through_promotion": "|".join(learned.loc[learned.support_information_arrived, "support_new_node_ids"]),
                "new_reverse_ids_entry_through_promotion": "|".join(learned.loc[learned.reverse_operation_arrived, "reverse_node_ids"]),
                "new_support_ids_during_actual_holding": "|".join(holding.loc[holding.support_information_arrived, "support_new_node_ids"]),
                "role": "ALL_ACTUAL_CYCLES_POST_RUN_DESCRIPTION_NO_RULE_SELECTION"})
            points = (("SUPPORT_ACTIVATION", trade["setup_date"]), ("ACCEPTANCE_DECISION", trade["entry_origin"]),
                      ("ACTUAL_ENTRY_CLOSE", entry), ("CARRY_PROMOTION", promotion),
                      ("EXIT_DECISION", account["decisions"].loc[account["decisions"].execution_date.eq(exit_date) & account["decisions"].desired_shares.eq(0), "origin"].iloc[0]))
            for role, when in points:
                context = indexed.loc[pd.Timestamp(when)]
                contexts.append({"period": period, "cycle_id": trade["cycle_id"], "date": pd.Timestamp(when),
                    "point_role": role, "actual_cycle_net_pnl": trade["net_pnl"],
                    **{key: context[key] for key in CONTEXT},
                    "pmi_new_orders_original_level": context.pmi_orders_level + 50,
                    "relative_volume": np.exp(context.log_relative_volume),
                    "role": "PAST_AVAILABLE_CONTEXT_OUTCOME_DESCRIPTIVE_NOT_ENTRY_TARGET"})
    return pd.DataFrame(cycles), pd.DataFrame(contexts)


def baseline_opportunities(observed: pd.DataFrame, accounts: dict) -> pd.DataFrame:
    indexed = observed.set_index("date", verify_integrity=True)
    rows = []
    for period in study.PERIODS:
        daily = accounts[(period, "STRESS", PRIMARY)]["daily"].set_index("date")
        for trade in accounts[(period, "STRESS", study.CONTROL_A)]["trades"].to_dict("records"):
            origin = pd.Timestamp(trade["entry_origin"])
            context = indexed.loc[origin]
            entry = pd.Timestamp(trade["entry_date"])
            rows.append({"period": period, **trade,
                "new_primary_holding_at_A_entry_close": int(daily.loc[entry, "shares"]) > 0,
                **{key: context[key] for key in CONTEXT},
                "pmi_new_orders_original_level": context.pmi_orders_level + 50,
                "relative_volume": np.exp(context.log_relative_volume),
                "role": "ALL_ORIGINAL_A_ACTUAL_CYCLES_AND_OPEN_NO_WINNER_SELECTION"})
    return pd.DataFrame(rows)


def cost_diagnosis(accounts: dict, metrics: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    diagnoses, curves = [], []
    for period in study.PERIODS:
        account = accounts[(period, "STRESS", PRIMARY)]
        daily = account["daily"]
        row = metrics.loc[metrics.period.eq(period) & metrics.cost.eq("STRESS") & metrics.policy.eq(PRIMARY)].iloc[0]
        fees = float((daily.commission + daily.slippage).sum())
        gross_curve = 200000 + (daily.price_pnl + daily.dividend_accrual).cumsum()
        gross_returns = gross_curve / gross_curve.shift(fill_value=200000) - 1
        stats = measurements.return_statistics(gross_returns)
        if not np.allclose(gross_curve - (daily.commission + daily.slippage).cumsum(), daily.equity, rtol=0, atol=1e-6):
            raise ValueError("固定原数量毛财富不能逐日还原净财富。")
        rejects = account["rejections"]
        decisions = account["decisions"]
        t = account["trades"]
        positive = t.loc[t.net_pnl.gt(0), "net_pnl"]
        diagnoses.append({"period": period, "actual_net_pnl": row.ending_equity - 200000,
            "fixed_actual_quantities_gross_pnl": float(gross_curve.iloc[-1] - 200000), "actual_fees": fees,
            "fees_fraction_of_gross": fees / (gross_curve.iloc[-1] - 200000),
            "fixed_quantity_zero_fee_explanatory_cagr": stats["net_cagr"],
            "fixed_quantity_zero_fee_explanatory_sharpe": stats["net_sharpe"],
            "zero_fee_role": "EXPLANATORY_FIXED_QUANTITY_PATH_NOT_EXECUTABLE_STRATEGY",
            "largest_winner_cny_fraction": float(positive.max() / positive.sum()),
            "all_completed_actual_cycles": len(t), "all_promoted_cycles": int(t.promotion_date.notna().sum()),
            "losing_promoted_cycles": int((t.net_pnl.lt(0) & t.promotion_date.notna()).sum()),
            "actual_rejections": len(rejects), "actual_account_risk_stop": bool(daily.risk_stopped.any()),
            "actual_risk_reduction_decisions": int(decisions.reason.eq("PRIOR_CLOSE_RISK_REDUCTION").sum()),
            "accepted_events_consumed_while_already_holding": int((decisions.entry_event & decisions.shares_before.gt(0)).sum())})
        curves.append(pd.DataFrame({"date": daily.date, "period": period, "actual_net_equity": daily.equity,
            "fixed_actual_quantity_zero_fee_explanatory_wealth": gross_curve,
            "role": "ACCOUNTING_BOUND_NOT_NEW_ACCOUNT_OR_PARAMETER_RESCUE"}))
    return diagnoses, pd.concat(curves, ignore_index=True)


def figures(observed: pd.DataFrame, accounts: dict, cycles: pd.DataFrame) -> list[dict]:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    receipts = []
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), constrained_layout=True)
    for ax, period in zip(axes, study.PERIODS):
        for policy in [PRIMARY, *study.CONTROLS]:
            d = accounts[(period, "STRESS", policy)]["daily"]
            ax.plot(d.date.to_numpy(dtype="datetime64[ns]"), d.equity / 10000,
                    label=study.NAMES[policy], color=COLORS[policy], linewidth=2 if policy == PRIMARY else 1.25)
        ax.set(title=f"{period.replace('_', '—')}：压力费用，20万元完整现金账户", ylabel="账户财富 / 万元")
        ax.grid(alpha=.18)
        ax.legend(ncol=3, frameon=False, fontsize=9)
    fig.suptitle("不同阶段的动作已经改变，但新原型近期未改善原A\n全部历史为开发资料；收益含现金、费用、股息及自然期末持仓", fontsize=13)
    target = folder / "两个时期压力费用_全部五账户.png"
    fig.savefig(target, dpi=145)
    plt.close(fig)
    receipts.append({"path": relative(target), "sha256": study.digest(target), "panels": 2})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.1), constrained_layout=True)
    chosen = [("2019-01-10", "2018-12-17", "2019-04-05"), ("2024-09-26", "2024-09-02", "2024-11-01"),
              ("2021-12-10", "2021-11-29", "2022-01-07"), ("2023-06-16", "2023-06-01", "2023-07-10")]
    for ax, (entry, begin, end) in zip(axes.ravel(), chosen):
        cycle = cycles.loc[cycles.entry_date.eq(pd.Timestamp(entry))].iloc[0]
        chunk = observed.loc[observed.date.between(begin, end)]
        decisions = accounts[(cycle.period, "STRESS", PRIMARY)]["decisions"]
        stops = decisions.loc[decisions.origin.between(begin, end) & decisions.shares_before.gt(0)]
        ax.plot(chunk.date.to_numpy(dtype="datetime64[ns]"), chunk.ac, color="#273d50", linewidth=1.7, label="现金平移收盘")
        ax.step(stops.origin.to_numpy(dtype="datetime64[ns]"), stops.structural_stop, where="post", color="#b36c35", label="当时已知结构止损")
        for when, label, color in [(cycle.setup_date, "激活", "#737d86"), (cycle.entry_date, "实际进入", "#186a9c"),
                                   (cycle.promotion_date, "晋延续", "#735e9d"), (cycle.exit_date, "实际退出", "#b8454c")]:
            ax.axvline(pd.Timestamp(when), color=color, linestyle="--", linewidth=.9, alpha=.8)
            ax.text(pd.Timestamp(when), .99, label, transform=ax.get_xaxis_transform(), rotation=90,
                    color=color, va="top", ha="right", fontsize=8)
        ax.set_title(f"{date_label(cycle.entry_date)}进入：净损益{cycle.net_pnl:+,.2f}元\n确认周入场前{cycle.week_rows_before_actual_entry}日 / 入场日起{cycle.week_rows_on_or_after_entry}日", fontsize=10)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.grid(alpha=.18)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("上涨段与失败段都经历延续确认：确认的实际信息构成\n固定2019、2024原案例；2021、2023为持仓后仅一个收盘即确认的全部两例，事后归因不作买点标签", fontsize=12)
    target = folder / "具体上涨与持仓确认反例_已知结构位.png"
    fig.savefig(target, dpi=145)
    plt.close(fig)
    receipts.append({"path": relative(target), "sha256": study.digest(target), "panels": 4})
    return receipts


def main() -> None:
    if REPORT.exists() or (OUT / "post_run_diagnosis.json").exists():
        raise RuntimeError("完整描述已保存，不重复本次归档。")
    exact = check_frozen()
    summary = study.read(OUT / "summary.json")
    if (summary["decision"] != "TECH.R232" or summary["original_new_accounts_completed"] != 12
            or summary["all_economic_gates_passed"] or summary["historical_stability_passed"]):
        raise ValueError("实际完整终态与失败归因用途不同。")
    observed = pd.read_parquet(SOURCE / "results/全部3488支持信息与价格接受状态.parquet")
    metrics = pd.DataFrame(summary["metrics"])
    intervals = pd.read_parquet(OUT / "results/全部四场景四对照两尺度净增量区间.parquet")
    if len(metrics) != 20 or len(intervals) != 32 or len(observed) != 3488:
        raise ValueError("完整账户、固定来源日历或全部区间缺失。")
    accounts = {(period, cost, policy): acceptance.saved_account(period, cost, policy)
                for period in study.PERIODS for cost in study.COSTS for policy in [PRIMARY, *study.CONTROLS]}
    cycles, contexts = holding_description(observed, accounts)
    original = baseline_opportunities(observed, accounts)
    complete_original = original.loc[original.status.eq("COMPLETE")]
    if len(cycles) != 9 or len(complete_original) != 54:
        raise ValueError("原A全部完成周期或新主账户实际周期不完整。")
    fees, gross_curves = cost_diagnosis(accounts, metrics)
    points = observed.loc[observed.support_entry_event, ["date", "support_setup_date", "support_setup_source_ids", *CONTEXT]].copy()
    points = points.loc[:, ~points.columns.duplicated()]
    actual_origins = set(cycles.entry_origin)
    points["primary_stress_actual_entry"] = points.date.isin(actual_origins)
    activation = observed.loc[observed.support_activation_allowed, ["date", "support_new_node_ids", "support_unique_common_sources", "support_observation_reason", "support_setup_date", "support_watch_alive_after_decision"]]
    key_dates = pd.read_parquet(study.KEYS).date
    keys = observed.loc[observed.date.isin(key_dates), ["date", "ac", *CONTEXT]].copy()
    source_qualifications = pd.read_parquet(SOURCE / "results/全部65来源用途资格_操作前值与非激活保留.parquet")
    if len(points) != 12 or len(activation) != 26 or len(keys) != 19 or len(source_qualifications) != 65:
        raise ValueError("固定范围的点位、激活、关键日或来源资格不完整。")
    for name, frame in [("主政策全部九实际压力周期_周线确认信息构成", cycles), ("全部四十五持仓节点_当时可知量价宏观行业", contexts),
                        ("原A全部真实周期与开放_新原型机会状态", original), ("全部十二支持价格接受点位_已持仓消费保留", points),
                        ("全部二十六支持激活日_不等于实际进入", activation), ("原十九关键日_新信息状态完整比较", keys),
                        ("固定原数量零费用解释边界_不作策略", gross_curves)]:
        table(name, frame)
    frozen_after = check_frozen()
    pictures = figures(observed, accounts, cycles)
    recent = cycles.loc[cycles.period.eq("2020_2026")]
    short_confirm = cycles.loc[cycles.actual_holding_closes_before_promotion.eq(1)]
    no_watch = complete_original.loc[complete_original.support_observation_reason.eq("NO_ACTIVE_OBSERVATION")]
    daily_june = observed.loc[observed.date.eq(pd.Timestamp("2020-06-02"))].iloc[0]
    relevant_a = complete_original.loc[complete_original.entry_origin.eq(pd.Timestamp("2020-06-02"))]
    diagnosis = {"at": study.parent.original.now(), "registration": "TECH.R231", "decision": "TECH.R232",
        "purpose": "POST_RUN_FULL_ACTUAL_HOLDING_AND_ALL_A_OPPORTUNITIES_NO_PARAMETER_RESCUE",
        **exact, "frozen_inputs_after_description_exact": frozen_after,
        "all_economic_scenarios_passed": sum(row["economic_passed"] for row in summary["gates"]),
        "all_historical_stability_passed": summary["historical_stability_passed"],
        "all_new_account_metrics": 12, "all_reused_control_account_metrics": 8,
        "all_saved_interval_rows": len(intervals), "original_support_activation_days": len(activation),
        "original_support_price_accepted_points": len(points), "original_price_only_points": int(observed.price_entry_event.sum()),
        "all_primary_stress_completed_cycles": len(cycles), "all_primary_stress_promoted_cycles": int(cycles.promotion_date.notna().sum()),
        "recent_completed_cycles": len(recent), "recent_wins": int(recent.net_pnl.gt(0).sum()),
        "recent_promoted_losses": int(recent.net_pnl.lt(0).sum()),
        "confirmation_after_one_actual_holding_close": short_confirm[["entry_date", "promotion_date", "week_rows_before_actual_entry", "week_rows_on_or_after_entry", "net_pnl"]].to_dict("records"),
        "promotion_uses_any_pre_entry_week_rows": int(cycles.week_rows_before_actual_entry.gt(0).sum()),
        "week_contains_pre_entry_data_is_frozen_rule_not_storage_bug": True,
        "no_week_purity_or_waiting_parameter_change_admitted": True,
        "all_A_completed_cycles": len(complete_original), "all_A_cycles_including_open": len(original),
        "A_completed_cycle_origins_without_active_support_watch": len(no_watch),
        "A_completed_cycles_with_primary_flat_at_A_entry_close": int((~complete_original.new_primary_holding_at_A_entry_close).sum()),
        "A_no_watch_scope": "固定来源没有活动观察，不等于全国无政策或无正期望机会",
        "june_2020_fixed_original_case": {"date": daily_june.date, "watch_state": daily_june.support_observation_reason,
            "source_watch_alive": daily_june.support_watch_alive_after_decision,
            "A_actual_cycles_at_that_origin": relevant_a[["entry_origin", "entry_date", "exit_date", "net_pnl", "status"]].to_dict("records")},
        "cost_signal_account_constraint_attribution": fees,
        "first_period_actual_payoff": "UNKNOWN_NO_LOSING_SAMPLE_THREE_CYCLES_NOT_VALIDATED_100_PERCENT",
        "recent_standard_expectancy_positive_but_user_p_times_B_gate_failed": True,
        "source_coverage": "RECORDED_POSITIVE_EVIDENCE_ONLY_EMPTY_NOT_POLICY_ABSENCE",
        "complete_policy_announcement_coverage": "NOT_ESTABLISHED", "first_vintage": "NOT_CERTIFIED",
        "new_training_labels": 0, "history_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False}
    study.write(OUT / "post_run_diagnosis.json", diagnosis)
    proposal = {"at": study.parent.original.now(), "status": "PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN",
        "name": "ACTUAL_ENTRY_NEW_CONTINUATION_EVIDENCE_AND_ALL_A_OPPORTUNITY_SEQUENCE",
        "question": "原A全部实际进入与持仓期间，哪些新增可知价格、参与扩散及支持信息代表延续或失败；旧锚消费后后续修复如何解释？",
        "mechanism_difference": "从单次政策触发原型改为实际进入后的新增证据与机会序列；不是增加入场指标过滤或缩短旧等待参数。",
        "complete_description_first": "原A全部54完成及开放、新原型全部9周期和原19关键日；每日可知信息时序，不只挑赢的上涨段。",
        "new_definition_required_before_finance": "信息角色、实际进入后的新结构、阶段切换与失效、旧状态结束后的新机会身份及未知分支须唯一事前固定。",
        "no_shortcut": "不能直接改R232周线纯度、等待天数、新闻年限或来源；不能早期新原型/近期原A拼接，不把A未持有全部当新增买点。",
        "frequency": "软目标；更密价格对照已实际失败，不能靠增加交易替代毛优势。",
        "source_coverage": "有限本地已记录支持信息；缺失不当无政策，不以全国新闻全集无限延期。",
        "complete_account_required": "若描述支持不同动作，唯一整体账户与同进入静态/同动作无新增信息/原A/原阶段完整对照；同风险费用全部时期。",
        "acceptance": "完整净CAGR和Sharpe增量、DD<=10%、实际pB>1且标准净期望>0及固定两尺度全部区间，历史开发非独立。",
        "financial_admission": "NOT_ADMITTED", "new_financial_runs": 0, "independent_validation": "NOT_ESTABLISHED"}
    study.write(OUT / "next_actual_holding_information_proposal.json", proposal)
    pressure = metrics.loc[metrics.cost.eq("STRESS")].copy()
    pressure["policy_name"] = pressure.policy.map(study.NAMES)
    all_metrics = metrics.copy()
    all_metrics["policy_name"] = all_metrics.policy.map(study.NAMES)
    annual = pd.read_parquet(SOURCE / "results/全部逐年实际次数与净收益.parquet")
    annual = annual.loc[annual.cost.eq("STRESS") & annual.policy.eq(PRIMARY)]
    gate_table = pd.DataFrame(summary["gates"])
    cycle_display = cycles.copy()
    cycle_display["确认周构成"] = cycle_display.week_rows_before_actual_entry.astype(str) + "前 / " + cycle_display.week_rows_on_or_after_entry.astype(str) + "入场起"
    paragraphs = [
        "# 支持信息、价格接受与延续：完整账户结果及失败解释",
        f"TECH.R231登记、TECH.R232结果；{summary['at']}。唯一配置已完成12新账户、复用8原A/原阶段账户，共20账户和全部32增量区间。本配置拒绝：四经济场景0/4、历史稳定失败。目标仍active，尚未提高近期收益和夏普；此次归因不修改冻结动作。",
        "## 结论与券商启发的实际转化",
        "已把‘策略会变’落实为观察→价格接受→EARLY→CARRY→失效退出的状态，而非指标静态多数投票。支持信息、真实买入、完整周确认、反向操作分别使用自身时钟。小资金的选择与容量优势仍需靠实际净优势实现。本实验早期改善、近期基本持平于现金且显著低于原A；不是全部阶段分析无效，只拒绝这套固定完整配置。",
        markdown(pressure, {"period": "时期", "policy_name": "账户", "net_cagr": "净年化", "net_sharpe": "净夏普", "max_drawdown": "最大回撤", "completed_cycles": "完成", "win_rate": "胜率", "payoff": "实际净B", "p_times_b": "p×B", "standard_expectancy_loss_units": "标准净期望", "average_full_year_cycles": "完整年均次数"}, percent=("net_cagr", "max_drawdown", "win_rate")),
        "两独立时期各从20万元开始，全部现金日计入252日年化；不把两时期拼接。近期新主账户胜率2/6=33.33%，实际净B=2.3729，p×B=0.79097，标准亏损单位期望pB−q=+0.12430。标准净期望正与用户要求pB>1是两个门，不能将0.79097说成数学期望负，也不能忽略更严格pB门。早期3/3盈利，没有亏损样本，B/pB/标准亏损单位期望不定义，保留UNKNOWN，不宣传可靠100%胜率或用计划2R替代。",
        "## 唯一规则、来源身份与实际执行",
        "固定R230的65来源节点、25已保存操作原文；九明确人民币降准宣布/同期政府转载/首次资本工具公告纳入，操作低于前一保存操作可激活、较高可失效；前值未知及外汇/考核/实施进度/迟发回顾不自动激活。已宣布目标与后来匹配操作只确认，不再次激活；同文件多节点不独立加分。全国政策全集、历史首版及最早宣布均未认证，空节点是未知，不是无政策。",
        "新支持首次落入原ETF16:00决定钟时固定现金平移高低价；当天不进入，后来收盘严格越固定高且日MACD柱正才次开申请。锚接受或失效就消费，不因已有持仓或拒单再试；新支持不重写活锚。真实进入后EARLY锚低止损；日柱负且低EMA20也退出。上一完整自然周低与收盘均高于此前周且最后日不早于实际进入，晋CARRY，结构位仅抬高；不再要求周MACD/PMI正。此确认周允许含入场前交易日，是冻结规则，不是日期格式缺陷。",
        "与同价格动作不使用支持信息、同支持进入固定1ATR止损/2ATR目标/20持仓收盘、原A和原阶段完整比较。每账户最大50%、成熟ES5预算2.5%、10%跳空预算与剩余回撤余量、DD10%停买；整手100、.001报价、最低佣金5、T+1、股息应收/到账和自然期末持仓。基础佣金/滑点0.0002/0.0005，压力0.0004/0.001；原风险预算和下一开盘执行保持。",
        f"范围完整：3488来源观察日、26支持激活日、12价格接受点位；压力主账户只有9实际完成周期。早期6接受点位中3在已有持仓时消费，不能计为6笔成交；近期6点位均成交。无政策价格对照全部406点位，完整账户早期45/近期64完成周期却都为负，次数更多没有带来质量改善。原19关键日及全部65来源资格保留。",
        markdown(points, {"date": "价格接受决定", "support_setup_date": "固定观察锚", "support_setup_source_ids": "支持身份", "primary_stress_actual_entry": "主账户压力成交"}, dates=("date", "support_setup_date")),
        "## 全部实际持仓与确认信息构成",
        markdown(cycle_display, {"period": "时期", "setup_date": "支持观察", "entry_origin": "接受决定", "entry_date": "实际进入", "promotion_date": "晋延续", "actual_holding_closes_before_promotion": "晋级前持仓收盘", "确认周构成": "确认周信息", "exit_date": "实际退出", "net_pnl": "实际净损益元"}, dates=("setup_date", "entry_origin", "entry_date", "promotion_date", "exit_date")),
        f"9个周期全部晋延续，其中近期4个后来亏损；{int(cycles.week_rows_before_actual_entry.gt(0).sum())}个确认周包含入场前交易日。2021-12-10、2023-06-16均在仅1个实际持仓收盘后下一周一确认，确认周4日早于买入、1日为买入日，两笔净亏2686.01和2058.31元。这说明当前确认条件缺少足以区分延续与失败的证据；不能事后直接把‘全部周日均在买入后’或多等几天当已验证改进。全部45持仓节点同时保存当时日/周MACD、量、资金、融资、PMI新订单、行业参与和源龄；已知与未知分开。",
        "2019原成功段：Jan7才从Jan5政府保存页得到支持观察，Jan9价格接受、Jan10次开买入；此时日柱正、完整周柱负，PMI新订单49.7/环比−0.7。Jan14晋延续、Mar27退出，实际净利16105.95元。宏观水平和周柱不必同步为正，此例支持阶段动作的研究，但不是保证其他修复赚钱。它与旧2019周期7978.96元数量、时点及持有路径不同，不能把差直接称为单因素增量。",
        "2024原成功段：Sep24上午同源R01/C01宣布，固定当日高低；Sep25接受、Sep26次开进入、Sep30晋延续、Oct17退出，净利7747.24元。宣布额度不当真实股票流入，同源工具不作为两票独立确认。旧原周期3776.25元同样只是不同账户路径的参照。随后失效用当时结构位，不能从最高点倒选退出。",
        "2020Apr3实际买入来自Mar30 RATE18支持锚、Apr2价格决定，不来自Apr3 16:57:32降准公告；该公告首次ETF决定Apr7才可知。Apr13晋延续、Apr14次开退出，净利645.45元。Feb18那一笔虽也晋级，却在Mar2净亏2242.24元；‘政策支持+突破+晋级’仍不能覆盖全部风险。",
        "2025May7新宣布后May12接受、May13进入、May19晋级、May28净亏1150.68元。2017Apr5和2026May11原两假启动被此原型避开，但避开两个已知反例不能替代近期6笔和完整账户的检验，不能据此宣称多因子模型已经稳健。",
        "## 信号、费用和账户约束分别归因",
        markdown(pd.DataFrame(fees), {"period": "时期", "fixed_actual_quantities_gross_pnl": "原数量毛损益元", "actual_fees": "佣金加滑点元", "actual_net_pnl": "净财富增量元", "fixed_quantity_zero_fee_explanatory_cagr": "零费用解释年化", "fixed_quantity_zero_fee_explanatory_sharpe": "零费用解释夏普", "actual_rejections": "实际拒单", "actual_risk_reduction_decisions": "风险减仓决定"}, percent=("fixed_quantity_zero_fee_explanatory_cagr",)),
        f"近期毛损益1457.80元，费用1202.36元，净增255.44元；费用占毛优势约{fees[1]['fees_fraction_of_gross']:.2%}。在保持实际数量、时间及持有路径的零费用解释边界下，年化约{fees[1]['fixed_quantity_zero_fee_explanatory_cagr']:.4%}、夏普{fees[1]['fixed_quantity_zero_fee_explanatory_sharpe']:.4f}，仍远低于原A。失败既有摩擦，也有毛优势弱、机会少；不能仅降佣金或放松风险来救配置。此零费用序列仅财富分解，不是新可执行策略，因为未重新计算资金、风险及数量。",
        f"近期盈利金额约{fees[1]['largest_winner_cny_fraction']:.2%}来自2024一笔；其余盈利只有645.45元，四亏损共8137.25元。新主账户两个时期无拒单、无DD停买，风险减仓决定分别{fees[0]['actual_risk_reduction_decisions']}/{fees[1]['actual_risk_reduction_decisions']}次。账户限制存在，但不是全部失败的原因。平均资金曝光早期约2.90%、近期0.98%，完整年均交易0.6/1.0；零成交完整年份按下面逐年表保留，2026未完年不混入完整年均。不能只报持仓期间Sharpe来遮掩长期现金。",
        markdown(pressure.loc[pressure.policy.eq(PRIMARY)], {"period": "时期", "worst_day": "最差账户日", "worst_trade": "最差实际周期", "largest_winner_fraction_of_all_wins": "最大赢家回报集中度", "mean_exposure": "平均曝光", "gross_turnover_over_initial_capital": "毛换手/初始资本", "open_pnl_cny": "开放损益元", "unpaid_dividend_cny": "应收股息元"}, percent=("worst_day", "worst_trade", "largest_winner_fraction_of_all_wins", "mean_exposure")),
        "上表集中度按实际周期回报计算；前述92%左右按盈利人民币金额计算，两者定义不同。早期‘最差实际周期’为三笔中的最低盈利，不是损失尾部估计。新主账户期末全部空仓/开放损益0；原A保存终态没有可核对的开放周期账面分解，指标表保留UNKNOWN，全账户财富仍包含其原库存和应收，不能私填0。",
        "## 原A全部机会与单次激活的覆盖问题",
        f"原A54完成周期及{len(original)-54}开放周期全部保存，{len(no_watch)}个完成周期的进入决定日为NO_ACTIVE_OBSERVATION；{int((~complete_original.new_primary_holding_at_A_entry_close).sum())}个A周期买入当日新主账户收盘为空仓。这里没有只挑原A赢家，也不把无活动观察解释为无政策。比较对象是实际资金路径，不能直接把54−9称作新增有效交易数。",
        f"固定2020年6月上涨案例，Jun2旧阶段修复的进入决定日对应支持状态{daily_june.support_observation_reason}，当时旧支持锚已消费，原型不能重复使用旧支持开启观察。原A在Jun2没有对应进入决定，实际查询保留空集；旧修复阶段该周期盈利10756.16元属于旧阶段账户，不当原A收益复制。单次消费限制确实漏掉该后续腿，但应先解释新价格/参与结构，而不是事后延长公告寿命或无限重用旧锚。",
        markdown(annual, {"period": "时期", "year": "自然年", "full_year": "完整年", "completed_cycles": "完成交易", "calendar_year_return": "该年净收益"}, percent=("calendar_year_return",)),
        "## 全部经济门和历史不确定性",
        markdown(gate_table, {"period": "时期", "cost": "费用", "positive_cagr_sharpe": "收益夏普正", "drawdown_within_10pct": "DD门", "actual_pB_above_1": "pB门", "actual_standard_EV_positive": "标准期望门", "beats_A_SAVED_WEIGHT": "超过A", "beats_R212_CONTROL": "超过阶段", "beats_PRICE_ACCEPTANCE_CARRY_ONLY": "超过无支持", "beats_SUPPORT_PRICE_ACCEPTANCE_FIXED_EXIT": "超过固定退", "economic_passed": "全部经济门"}),
        "收益前固定的四场景四对照、20和252日循环块各2000次，全部32区间保存。早期压力相对A年化点增量+1.1372个百分点、夏普+0.6719，但两尺度区间都跨0；近期压力相对A年化−3.9712个百分点、夏普−1.1968，两尺度全部区间全负。原型相对弱价格对照可以显著改善，但没有证明优于已有较好A。",
        markdown(intervals, {"period": "时期", "cost": "费用", "control": "对照", "block": "块日数", "cagr_delta": "年化点增量", "cagr_delta_lower": "年化95下", "cagr_delta_upper": "年化95上", "sharpe_delta": "夏普点增量", "sharpe_delta_lower": "夏普95下", "sharpe_delta_upper": "夏普95上"}, percent=("cagr_delta", "cagr_delta_lower", "cagr_delta_upper")),
        "这些区间仅描述原开发历史，历史反复选择未校正；没有独立验证、没有‘已去过拟合’认证。过去研究失败均保留，不能早期用新原型、近期用A拼接成收益，也不能选择基础费用或某个有利时期掩盖压力失败。",
        "## 技术接续和长期事实边界",
        "原金融一次运行已完成全部12账户及20指标、现金/时序复算，随后第一个对照日期断言因ms/ns存储类型不同退出；原失败、冻结账户和代码保存。另立1.0.1只在16个比较对的临时日期序列规范为ns，并严格逐值、索引检查，接续全部区间；0账户重跑、0金融规则改动。必要检查8原测试+2日期测试，19整段前缀、8原保存对照、12财富复算/12持仓时序通过。63原冻结输入和80接续冻结文件（含12账户72文件）精确保持。",
        "## 下一步具体问题与实验边界",
        "下一优先问题是原A全部实际进入及持仓期间的新增延续证据，以及支持锚消费后的后续机会身份。当前已提供54完成及开放、9新周期、45节点完整描述底表；继续按实际买入钟拆新增价格结构、量/参与扩散、资金/融资/订单/政策不同角色与未知，不只解释赢家。若能形成与单次政策触发实质不同的机制，再唯一固定阶段切换/失败/再进入动作和原账户对照后检验。",
        "不将更多过滤或更密交易自动视为改善，不直接改本冻结周线条件/窗口/来源；不是重新运行R232。新提案PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN，0已准入待跑金融；完整金融目标未实现，继续研究权限保持。原13独立前瞻/E03及旧策略保持；没有当日市场意见或执行授权。",
        "## 全部20账户指标",
        markdown(all_metrics, {"period": "时期", "cost": "费用", "policy_name": "账户", "net_cagr": "净年化", "net_sharpe": "净夏普", "max_drawdown": "DD", "completed_cycles": "完成", "p_times_b": "p×B", "standard_expectancy_loss_units": "标准期望", "total_commission": "佣金元", "total_slippage": "滑点元", "open_pnl_cny": "开放损益元"}, percent=("net_cagr", "max_drawdown")),
        "## 可复查文件",
        "[原冻结用途](../../../../docs/510300_SUPPORT_PRICE_ACCEPTANCE_V1.md)、[原协议](../protocol.json)、[原日期失败](../acceptance_storage_failure.json)、[1.0.1接续协议](protocol.json)、[实际金融结果](summary.json)、[全归因](post_run_diagnosis.json)、[下一不同机制提案](next_actual_holding_information_proposal.json)。完整CSV及Parquet在results；原12账户逐日/订单/周期/决定/拒单/终态在上级accounts，原A和阶段通过原协议逐路径引用。",
        "![两个时期完整账户](figures/两个时期压力费用_全部五账户.png)",
        "![具体上涨与持仓反例](figures/具体上涨与持仓确认反例_已知结构位.png)"]
    with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n\n".join(paragraphs) + "\n")
    study.write(OUT / "delivery_receipt.json", {"at": study.parent.original.now(), "report": relative(REPORT),
        "report_sha256": study.digest(REPORT), "figures": pictures, "figures_actually_viewed": "NOT_YET",
        "all_actual_metrics": len(metrics), "all_interval_rows": len(intervals), "all_primary_cycles": len(cycles),
        "all_A_completed_cycles": len(complete_original), "all_A_including_open": len(original),
        "all_points": len(points), "all_activation_days": len(activation), "all_original_key_days": len(keys),
        "all_holding_context_rows": len(contexts), "account_replays_this_description": 0, "frozen": check_frozen()})
    print("R232全部20账户、32区间、9真实周期、原A全部周期和45持仓信息节点归档；仅描述，无金融重跑。", flush=True)


if __name__ == "__main__":
    main()
