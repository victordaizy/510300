"""从唯一实际结果输出完整报告、归因和图；不重跑金融或选择参数。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import core_actual_acceptance_study_v1 as study

ROOT, OUT = study.ROOT, study.OUT
REPORT = "原A进入后接受保护与新增延续_完整金融结果.md"
COLORS = {study.CONTROL_A: "#425466", study.CONTROL_STAGE: "#b0a092",
    study.inputs.POLICIES[0]: "#c5443a", study.inputs.POLICIES[1]: "#2878a0", study.inputs.POLICIES[2]: "#36a086"}


def frozen_exact():
    protocol = study.read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if study.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结来源改变："+item["path"])
    return len(protocol["sources"])


def load_account(period, cost, policy):
    if policy in (study.CONTROL_A, study.CONTROL_STAGE):
        return study.controls.saved_account(period, cost, policy)
    folder = OUT / "accounts" / period / cost / policy
    result = {name: pd.read_parquet(folder / f"{name}.parquet") for name in (*study.ACCOUNT_TABLES, "holding_evidence")}
    result["terminal"] = study.read(folder / "terminal.json")
    return result


def fmt(value, percent=False):
    if value is None or pd.isna(value):
        return "UNKNOWN"
    return f"{value:.4%}" if percent else f"{value:.6f}"


def markdown(frame):
    names = list(frame.columns)
    lines = ["| " + " | ".join(names) + " |", "|"+"---|"*len(names)]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("UNKNOWN" if pd.isna(v) else str(v).replace("|", "/").replace("\n", " ") for v in values) + " |")
    return "\n".join(lines)


def finance_inventory():
    files = []
    for period in study.PERIODS:
        for cost in study.COSTS:
            for policy in study.inputs.POLICIES:
                folder = OUT / "accounts" / period / cost / policy
                for name in (*study.ACCOUNT_TABLES, "holding_evidence"):
                    p = folder / f"{name}.parquet"
                    files.append({"path": study.relative(p), "sha256": study.digest(p)})
                p = folder / "terminal.json"
                files.append({"path": study.relative(p), "sha256": study.digest(p)})
    if len(files) != 84:
        raise ValueError("十二完整账户七文件不全。")
    return files


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("完整报告已输出，不重复。")
    frozen = frozen_exact()
    summary = study.read(OUT / "summary.json")
    if summary["new_accounts"] != 12 or len(summary["metrics"]) != 20 or len(summary["comparisons"]) != 16:
        raise ValueError("完整金融范围不足。")
    inventory = finance_inventory()
    metrics = pd.DataFrame(summary["metrics"])
    accounts = {(p,c,k): load_account(p,c,k) for p in study.PERIODS for c in study.COSTS
                for k in (study.CONTROL_A, study.CONTROL_STAGE, *study.inputs.POLICIES)}
    cycles, evidence, diagnoses = [], [], []
    for (period, cost, policy), result in accounts.items():
        t = result["trades"].copy()
        t.insert(0, "policy", policy)
        t.insert(0, "cost", cost)
        t.insert(0, "period", period)
        cycles.append(t)
        if policy in study.inputs.POLICIES:
            h = result["holding_evidence"].copy()
            for name, value in (("policy", policy), ("cost", cost), ("period", period)):
                h.insert(0, name, value)
            evidence.append(h)
            d = result["daily"]
            gross_equity = 200000.+(d.price_pnl+d.dividend_accrual).cumsum().to_numpy(float)
            gross_returns = gross_equity/np.r_[200000., gross_equity[:-1]]-1
            gross_stats = study.parent.measurements.return_statistics(gross_returns)
            stats = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost) & metrics.policy.eq(policy)].iloc[0]
            fees = float(d.commission.sum()+d.slippage.sum())
            gross = float((d.price_pnl+d.dividend_accrual).sum())
            reasons = result["orders"].loc[result["orders"].side.eq("SELL"), "reason"].value_counts().to_dict() if len(result["orders"]) else {}
            diagnoses.append({"period": period, "cost": cost, "policy": policy,
                "gross_actual_quantity_pnl": gross, "commission_and_slippage": fees,
                "fee_share_of_positive_gross": fees/gross if gross > 0 else None,
                "net_total_pnl": float(d.equity.iloc[-1]-200000),
                "fixed_quantity_zero_fee_explanatory_cagr": gross_stats["net_cagr"],
                "fixed_quantity_zero_fee_explanatory_sharpe": gross_stats["net_sharpe"],
                "counterfactual_is_not_executable_strategy": True,
                "mean_exposure": stats.mean_exposure, "open_pnl_cny": stats.open_pnl_cny,
                "actual_exit_reasons": reasons,
                "cycles_accepted": int(t.first_price_acceptance.notna().sum()),
                "cycles_promoted": int(t.promotion_date.notna().sum()),
                "cycles_extension_revoked": int(t.extension_revoked.sum()),
                "true_new_order_adverse_decisions": int(h.orders_fresh_deterioration.sum()),
                "true_new_reverse_decisions": int(h.reverse_fresh_operation.sum()),
                "participation_known_holding_decisions": int(h.participation_known.sum()),
                "consumed_episode_flat_decisions": int(result["decisions"].reason.eq("FORCED_EXIT_CONSUMED_SAME_PARENT_EPISODE").sum())})
    pairs, matching, match_counts = [], [], []
    for period in study.PERIODS:
        for cost in study.COSTS:
            primary = accounts[(period,cost,study.inputs.POLICIES[0])]
            protection = accounts[(period,cost,study.inputs.POLICIES[2])]
            fields = ("equity","cash","shares","net_return","commission","slippage")
            same = all(primary["daily"][field].equals(protection["daily"][field]) for field in fields)
            decision = primary["decisions"]
            extended = decision.holding_stage.eq("CARRY") & decision.shares_before.gt(0) & (decision.source_weight.eq(0)|decision.source_weight.isna())
            pairs.append({"period":period,"cost":cost,"information_and_protection_financial_sequences_exact":same,
                "main_promoted_cycles":int(primary["trades"].promotion_date.notna().sum()),
                "main_revoked_cycles":int(primary["trades"].extension_revoked.sum()),
                "main_parent_zero_or_unknown_carry_decisions":int(extended.sum())})
            original = accounts[(period,cost,study.CONTROL_A)]["trades"]
            columns = ["entry_date","entry_origin","exit_date","status","buy_debit","net_pnl","net_return"]
            for policy in study.inputs.POLICIES:
                a, b = original[columns].copy(), accounts[(period,cost,policy)]["trades"][columns].copy()
                for field in ("entry_date","entry_origin","exit_date"):
                    a[field] = pd.to_datetime(a[field]).astype("datetime64[ns]")
                    b[field] = pd.to_datetime(b[field]).astype("datetime64[ns]")
                joined = a.merge(b,on="entry_date",how="outer",suffixes=("_original_A","_new_policy"),indicator=True,validate="one_to_one")
                joined["entry_identity_role"] = joined.pop("_merge").astype(str)
                for field,value in (("policy",policy),("cost",cost),("period",period)):
                    joined.insert(0,field,value)
                matching.append(joined)
                both = joined.entry_identity_role.eq("both")
                match_counts.append({"period":period,"cost":cost,"policy":policy,"matched_entry_dates":int(both.sum()),
                    "original_only_entry_dates":int(joined.entry_identity_role.eq("left_only").sum()),
                    "new_only_entry_dates":int(joined.entry_identity_role.eq("right_only").sum()),
                    "original_positive_became_new_negative":int((both & joined.net_pnl_original_A.gt(0) & joined.net_pnl_new_policy.lt(0)).sum()),
                    "original_negative_became_new_positive":int((both & joined.net_pnl_original_A.lt(0) & joined.net_pnl_new_policy.gt(0)).sum())})
    study.table("全部四场景_参与宏观延续相对仅保护的实际作用",pd.DataFrame(pairs))
    study.table("全部原A与新政策实际进入匹配_完整现金数量反馈",pd.concat(matching,ignore_index=True))
    study.table("全部十二新账户原A进入保留及盈亏变号",pd.DataFrame(match_counts))
    all_cycles, all_evidence = pd.concat(cycles, ignore_index=True), pd.concat(evidence, ignore_index=True)
    study.table("全部二十账户实际周期与开放_费用复本不作独立机会", all_cycles)
    study.table("全部十二账户真实持有收盘_接受延续撤销和失效", all_evidence)
    study.table("全部十二账户毛优势费用持有及退出归因", pd.DataFrame(diagnoses))
    pressure = metrics.loc[metrics.cost.eq("STRESS")].copy()
    rows = [{"时期": r.period, "账户": study.NAMES[r.policy], "净年化": fmt(r.net_cagr, True), "净夏普": fmt(r.net_sharpe),
        "最大回撤": fmt(r.max_drawdown, True), "完成": r.completed_cycles, "赢/亏": f"{r.wins}/{r.losses}",
        "实际净pB": fmt(r.p_times_b), "标准净期望": fmt(r.standard_expectancy_loss_units), "完整年均次数": fmt(r.average_full_year_cycles)}
        for r in pressure.itertuples()]
    result_table = markdown(pd.DataFrame(rows))
    gate_table = markdown(pd.DataFrame(summary["gates"]))
    interval_table = pd.read_parquet(OUT / "results/全部四场景四对照两尺度净增量区间.parquet")
    if len(interval_table) != 32:
        raise ValueError("全部两尺度增量区间不足32。")
    pictures = OUT / "figures"
    pictures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140)
    for column, period in enumerate(study.PERIODS):
        for policy in (study.CONTROL_A, study.CONTROL_STAGE, *study.inputs.POLICIES):
            d = accounts[(period, "STRESS", policy)]["daily"]
            axes[0,column].plot(d.date, d.equity/10000, color=COLORS[policy], label=study.NAMES[policy], lw=1.4)
            axes[1,column].plot(d.date, d.drawdown*100, color=COLORS[policy], lw=1.2)
        axes[0,column].set_title(period+" 压力费用完整账户")
        axes[0,column].set_ylabel("净资产（万元）")
        axes[1,column].set_ylabel("收盘回撤（%）")
        for ax in axes[:,column]:
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8, loc="upper left")
    fig.suptitle("原A进入来源保持，实际接受保护与延续改变持有：完整现金日均纳入", fontsize=14)
    fig.tight_layout(rect=(0,.025,1,.96))
    fig.text(.02,.012,"两段时期各从20万元现金开始；终点保留自然持仓；历史全部为开发，图不代表独立验证。", fontsize=9)
    equity_path = pictures / "完整净资产与回撤_两时期全部五政策.png"
    fig.savefig(equity_path)
    plt.close(fig)
    observed = pd.read_parquet(study.OBSERVED)
    cases = [("2015-05-18","2015-07-10","2015确认后下跌"),("2018-12-17","2019-03-29","2019修复"),
        ("2020-03-20","2020-07-20","2020修复与快上涨"),("2024-09-02","2024-10-31","2024重新定价与失效"),
        ("2021-11-29","2021-12-31","2021旧周确认反例"),("2023-06-01","2023-07-10","2023旧周确认反例")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=140)
    for ax, (start,end,title) in zip(axes.ravel(),cases):
        segment = observed.loc[observed.date.between(start,end)].set_index("date")
        ax.plot(segment.index, segment.ac, color="#333c48", lw=1.3, label="现金平移收盘")
        period = "2015_2019" if pd.Timestamp(start).year < 2020 else "2020_2026"
        for policy in (study.CONTROL_A, *study.inputs.POLICIES):
            t = accounts[(period,"STRESS",policy)]["trades"]
            for field, marker in (("entry_date","^"),("exit_date","v")):
                dates = pd.to_datetime(t[field]).dropna()
                dates = dates.loc[dates.between(start,end) & dates.isin(segment.index)]
                ax.scatter(dates, segment.loc[dates,"ac"], color=COLORS[policy], marker=marker, s=42,
                    label=study.NAMES[policy]+("买" if field=="entry_date" else "卖"), alpha=.8)
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.tick_params(axis="x", labelrotation=22)
    axes[0,0].legend(fontsize=7, ncol=2)
    fig.suptitle("原六个案例：同进入来源后的真实成交日期与量价路径", fontsize=14)
    fig.tight_layout(rect=(0,.035,1,.96))
    fig.text(.02,.012,"三角标记是实际买卖日期，纵坐标对齐该日现金平移收盘；实际次开成交价在原订单。未用收盘冒充开盘成交。", fontsize=9)
    case_path = pictures / "原六段具体量价_实际接受保护与延续成交日.png"
    fig.savefig(case_path)
    plt.close(fig)
    full_metric_rows = [{"时期":r.period,"费用":r.cost,"账户":study.NAMES[r.policy],"净年化":fmt(r.net_cagr,True),
        "净夏普":fmt(r.net_sharpe),"回撤":fmt(r.max_drawdown,True),"净pB":fmt(r.p_times_b),"净期望":fmt(r.standard_expectancy_loss_units),
        "最差日":fmt(r.worst_day,True),"最差周期":fmt(r.worst_trade,True),"赢家集中":fmt(r.largest_winner_fraction_of_all_wins,True),
        "平均曝光":fmt(r.mean_exposure,True),"周转/初始资金":fmt(r.gross_turnover_over_initial_capital),"佣金":fmt(r.total_commission),
        "滑点":fmt(r.total_slippage),"开放损益元":fmt(r.open_pnl_cny)} for r in metrics.itertuples()]
    main_cycles = all_cycles.loc[all_cycles.cost.eq("STRESS") & all_cycles.policy.eq(study.inputs.POLICIES[0])].copy()
    case_cols = ["period","cycle_id","entry_origin","entry_date","exit_date","status","net_pnl","net_return",
                 "first_price_acceptance","promotion_date","first_extension_revocation","exit_reason"]
    for name in ("entry_origin","entry_date","exit_date","first_price_acceptance","promotion_date","first_extension_revocation"):
        main_cycles[name] = pd.to_datetime(main_cycles[name]).dt.strftime("%Y-%m-%d")
    for name in ("net_pnl","net_return"):
        main_cycles[name] = main_cycles[name].map(lambda v: fmt(v, name=="net_return"))
    text = f"""# 原A进入后接受保护与新增延续：完整金融结果

TECH.R235登记、R236实际结果；{summary['at']}。唯一主用途与两个新增归因对照、两时期/两费用共12新完整账户，8保存原A/原阶段对照，20完整指标及全部32两尺度区间。十一必要测试、原19整段来源/目标段前缀、4原A适配器逐日订单周期精确、8旧指标、12现金与时序核对。登记前时间格式失败及首次全日表保留；0参数网格/拟合/新行情。全部截至Sep30历史已看，独立验证未建立。

实际终态：{summary['status']}。四经济门通过{sum(g['economic_passed'] for g in summary['gates'])}/4，全部历史稳定区间通过={summary['historical_stability_passed']}。目标未实现；不得用后验18/20当新策略90%胜率，也不把任何一时期点值进步称去过拟合。

## 固定的不同机制

原A源权重/0/未知及10pp调仓带保持。实际进入日高低收盘才固定，严格后来价格接受才启用低点保护；此前破低不额外退出。完整买后周、当前接受和已知ETF上涨成交量余额/行业多数及领先收益支持时晋CARRY，周低保护只升不降，原0/未知允许风险减仓后继续持有。主的真新订单恶化/保存反向操作撤销延续而不强制原正目标退出。结构真实清仓消费原正目标段，下一明确零后新正才重启；新金融包括所有随后现金和进入反馈。

价格对照不使用参与/宏观，保护对照不延长。行业为逐点已知参与，不是固定组新扩散；ETF价量代理不是净资金流。旧宏观背景与未知不强制判坏，实际公布和首次观察分开。动作已在收益前唯一固定，结果后不改。

## 压力费用完整比较

{result_table}

## 全部经济门与两尺度不确定性

{gate_table}

{markdown(interval_table)}

## 所有时期费用的风险、费用与集中度

{markdown(pd.DataFrame(full_metric_rows))}

## 全部新账户持有、费用与动作归因

{markdown(pd.DataFrame(diagnoses))}

零费用数值仅用已发生数量固定的财富解释，不重算现金和数量，不是可执行候选，也不用于通过经济门。实际净收益、费用和原对照同时保留，减少损失与放弃盈利都必须进入完整账户。

## 多信息延续到底有没有实际改变持有

{markdown(pd.DataFrame(pairs))}

{markdown(pd.DataFrame(match_counts))}

原A和新策略的进入身份逐个外连接，未匹配也保留。相同进入日期不等于相同现金和数量；以前退出的反馈会改变后续权重数量，不能把单周期人民币差当纯持有贡献。早期主与仅保护金融序列在两费用均精确相同，主虽晋级2周期但原零/未知时延续决定为0；新增参与/宏观没有提供额外金融增量。近期主8晋级/8撤销、42原零/未知延续收盘，延续实际发生，但主压力年化/夏普仍低于仅保护，不能因状态复杂就称改善。

## 压力主政策全部真实周期及开放

{markdown(main_cycles[case_cols])}

原A和全部新政策两费用的所有周期保存在“全部二十账户实际周期与开放”表，费用复本不是独立机会。未来成交只影响以后；价格接受、晋级、撤销发生在当时收盘，表中完成净损益不用于状态判断。开放周期保留自然终态、不贴输赢标签。

## 六段具体上涨、下跌及反例

2015、2019、2020、2024上涨/下跌及2021/2023确认反例均显示全部真实日期，未挑赢家。2020Jun2是原阶段修复，不能当原A进入；原A真正Jul1快上涨和2024Sep25快上涨按原源进入，不要求买前等新周。延长持有必须纳入原2024Oct9重新进入及后续账户路径，不能只挑Sep30延长那一笔。2019不同进入/仓位不相减当纯持有增量。

![完整净资产与回撤](figures/{equity_path.name})

![六段真实交易日期](figures/{case_path.name})

图标记纵坐标是当日现金平移收盘，真实成交为此前收盘决定、次开订单价。来源全国覆盖及首版不认证，全部历史为开发，参数选择校正NOT_COMPUTED，独立验证NOT_ESTABLISHED；本配置终态只约束本用途，原冻结策略与原E03前瞻保持。
"""
    report_path = OUT / REPORT
    report_path.write_text(text, encoding="utf-8")
    study.write(OUT / "post_run_diagnosis.json", {"at":study.parent.original.now(),"decision":study.RESULT,
        "scope":"ALL_TWENTY_ACCOUNTS_TWELVE_NEW_FIXED_POLICIES_AND_ALL_SOURCE_CASES",
        "economic_gates_passed":sum(g["economic_passed"] for g in summary["gates"]),"historical_stability_passed":summary["historical_stability_passed"],
        "all_account_cycles_with_cost_replicates":len(all_cycles),"all_new_holding_evidence_rows":len(all_evidence),
        "pressure_main_completed":int(main_cycles.status.eq("COMPLETE").sum()),"pressure_main_open":int(main_cycles.status.eq("RIGHT_CENSORED").sum()),
        "all_new_account_diagnoses":diagnoses,"information_vs_protection_pairs":pairs,"all_original_A_entry_matches":match_counts,
        "new_financial_runs_for_report":0,"new_fits":0,"goal_achieved":False})
    if frozen_exact() != frozen or finance_inventory() != inventory:
        raise ValueError("报告输出改变了冻结来源或金融账户。")
    study.write(OUT / "delivery_receipt.json", {"at":study.parent.original.now(),"report":study.relative(report_path),
        "report_sha256":study.digest(report_path),"frozen_source_files_exact":frozen,"new_account_files_exact":84,
        "new_account_files":inventory,"all_metrics":20,"all_interval_rows":32,"pressure_main_cycles":len(main_cycles),
        "figures":[{"path":study.relative(p),"sha256":study.digest(p)} for p in (equity_path,case_path)],
        "figures_actually_viewed":False,"new_financial_runs_for_delivery":0})
    print("二十账户、完整周期/持有/费用归因和两图保存；金融未重跑，图待实际查看。", flush=True)


if __name__ == "__main__":
    main()
