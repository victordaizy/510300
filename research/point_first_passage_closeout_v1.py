"""读取已保存结果，核对模型时钟及实际点位并形成报告；不拟合或重跑账户。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import point_first_passage_inputs_v1 as learning
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account
from research.point_first_passage_study_v1 import OUT, CONTROL, PERIODS, COSTS, POLICIES, read, now, write_json, table


def require(condition, message):
    if not condition:
        raise ValueError(message)


def display(value, percent=False, precision=3):
    if value is None or not np.isfinite(float(value)):
        return "不可估计"
    return f"{value:.2%}" if percent else f"{value:.{precision}f}"


def saved_account(period, cost, policy):
    directory = OUT / f"results/accounts/{period}/{cost}/{policy}"
    result = {name: pd.read_parquet(directory / f"{name}.parquet") for name in ["daily", "orders", "trades", "decisions", "rejections"]}
    result["terminal"] = read(directory / "terminal.json")
    return result


def main():
    require(not (OUT / "saved_result_verification.json").exists(), "保存结果核对已完成，不重复运行。")
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    for source in protocol["sources"]:
        actual = hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
        require(actual == source["sha256"], "冻结来源改变："+source["path"])
    data = pd.read_parquet(OUT / "results/原点全部技术特征.parquet")
    outcomes = pd.read_parquet(OUT / "results/原点首次边界参考结果.parquet")
    forecasts = pd.read_parquet(OUT / "results/全部事前模型预测.parquet")
    stats = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    predictions = pd.read_parquet(OUT / "results/首次事件事前预测表现.parquet")
    records = [read(path) for path in sorted((OUT / "models").glob("*.json"))]
    require(len(records) == summary["monthly_fit_records"] == 142, "实际月度模型记录不完整。")
    by_index = {record["fit_index"]: record for record in records}
    maturity_checks, max_probability_error, max_quality_error = 0, 0., 0.
    for record in records:
        i = record["fit_index"]
        eligible = outcomes.status.eq("MATURE_REFERENCE") & outcomes.mature_idx.le(i)
        eligible &= outcomes.origin_index.ge(max(0, i-learning.WINDOW+1))
        eligible &= data.first_passage_feature_known.to_numpy(bool)
        expected = outcomes.loc[eligible].sort_values("origin_index")
        require(expected.origin_index.astype(int).tolist() == record["training_origins"], "保存模型的成熟训练成员与原合同不符。")
        require(record["latest_mature_idx"] == int(expected.mature_idx.max()), "最新标签成熟日期不符。")
        maturity_checks += len(expected)
    available = forecasts.loc[forecasts.status.eq("AVAILABLE")]
    for row in available.itertuples(index=False):
        require(int(row.fit_index) <= int(row.origin_index), "预测使用未来模型。")
        model = by_index[int(row.fit_index)]["model"]
        require(model is not None, "可用预测没有实际模型。")
        values = data.iloc[int(row.origin_index)][learning.FEATURES].to_numpy(float)
        p = learning.probability(model, values, row.policy)
        observed = np.array([row.p_LOSS, row.p_PROFIT, row.p_TIMEOUT])
        max_probability_error = max(max_probability_error, float(np.max(np.abs(p-observed))))
        require(abs(p.sum()-1) < 1e-12 and (p > 0).all(), "事件概率不合法。")
        quality = learning.quality(model, p)
        for name, value in quality.items():
            actual = getattr(row, name)
            require(np.isfinite(value) == np.isfinite(actual), "保存质量估计的缺失状态改变。")
            if np.isfinite(value):
                max_quality_error = max(max_quality_error, abs(value-actual))
        expected_entry = quality["predicted_p_times_b"] > 1 and quality["predicted_net_expectation"] > 0
        require(bool(row.entry_event) == bool(expected_entry), "入场标记与冻结预计质量门不同。")
    require(max_probability_error < 1e-12 and max_quality_error < 1e-12, "保存预测不能从当时模型与特征还原。")

    checks, points, opportunities, annual, loaded = [], [], [], [], {}
    for period, (start, end) in PERIODS.items():
        for cost in COSTS:
            for policy in POLICIES:
                account = saved_account(period, cost, policy)
                loaded[(period, cost, policy)] = account
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(account)})
                actual_stats = metrics(account)
                saved = stats.loc[stats.period.eq(period) & stats.cost.eq(cost) & stats.policy.eq(policy)].iloc[0]
                for name in ["net_cagr", "net_sharpe", "max_drawdown", "win_rate", "payoff", "p_times_b", "ending_equity", "completed_cycles"]:
                    np.testing.assert_allclose(actual_stats[name], saved[name], rtol=1e-12, atol=1e-8, equal_nan=True)
                if cost == "STRESS":
                    dec = account["decisions"]
                    known = forecasts.loc[forecasts.policy.eq(policy) & forecasts.origin_index.isin(data.index[data.date.isin(dec.origin)])]
                    opportunities.append({"period": period, "policy": policy,
                                          "available_forecast_origins": int(known.status.eq("AVAILABLE").sum()),
                                          "estimated_quality_signal_origins": int(dec.entry_event.sum()),
                                          "signals_while_already_holding": int((dec.entry_event & dec.shares_before.gt(0)).sum()),
                                          "new_buy_requests": int((dec.execution_date.notna() & dec.desired_shares.gt(dec.shares_before)).sum()),
                                          "actual_entry_cycles": len(account["trades"]),
                                          "stopped": account["terminal"]["stopped"],
                                          "open_rejections": len(account["rejections"]),
                                          "maximum_predicted_p_times_b": float(known.predicted_p_times_b.max())})
            if cost == "STRESS":
                baseline = pd.read_parquet(CONTROL / period / cost / "A_SAVED_WEIGHT/trades.parquet")
                for policy, trades in [("A_SAVED_WEIGHT", baseline), *[(p, loaded[(period, cost, p)]["trades"]) for p in POLICIES]]:
                    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1):
                        completed = trades.loc[trades.status.eq("COMPLETE")]
                        n = int(completed.exit_date.dt.year.eq(year).sum()) if len(completed) else 0
                        annual.append({"period": period, "year": year, "policy": policy,
                                       "full_calendar_year": year < 2026, "completed_cycles": n})
        pressure = loaded[(period, "STRESS", "TECHNICAL_LOGIT")]
        for trade in pressure["trades"].itertuples(index=False):
            origin = int(np.flatnonzero(data.date.eq(trade.entry_origin))[0])
            f = forecasts.loc[forecasts.origin_index.eq(origin) & forecasts.policy.eq("TECHNICAL_LOGIT")].iloc[0]
            x = data.iloc[origin]
            require(f.entry_event and f.fit_index <= origin < trade.entry_idx <= trade.exit_idx, "真实进出点位时钟或当时预计质量不合格。")
            item = {"period": period, "cycle_id": trade.cycle_id, "origin": trade.entry_origin,
                    "entry_date": trade.entry_date, "exit_date": trade.exit_date,
                    "holding_sessions": trade.holding_sessions, "exit_reason": trade.exit_reason,
                    "predicted_p_times_b": f.predicted_p_times_b, "predicted_win_probability": f.predicted_win_probability,
                    "predicted_payoff": f.predicted_payoff, "predicted_net_expectation": f.predicted_net_expectation,
                    "actual_net_return": trade.net_return, "actual_net_pnl_cny": trade.net_pnl,
                    "prior_completed_week_last_date": x.weekly_last_date,
                    "relative_volume": float(np.exp(x.log_relative_volume)), "rv_ratio": float(np.exp(x.log_rv_ratio)),
                    "fit_index": int(f.fit_index), "reference_event_class": outcomes.event_class.iloc[origin],
                    "reference_net_return": outcomes.reference_net_return.iloc[origin]}
            item.update({name: float(x[name]) for name in learning.FEATURES})
            points.append(item)
    require(len(points) == 11, "真实点位数与完成账户不符。")
    point_frame, opportunity_frame = pd.DataFrame(points), pd.DataFrame(opportunities)
    table("实际进出点位与事前指标", point_frame)
    table("预计信号与实际交易次数", opportunity_frame)
    table("同完整自然年交易次数", pd.DataFrame(annual))
    maximum_error = max(check["maximum_accounting_error"] for check in checks)
    write_json(OUT / "saved_result_verification.json", {
        "at": now(), "status": "PASS_SAVED_MODEL_CLOCK_POINT_AND_ACCOUNT_RECOMPUTATION",
        "frozen_sources_unchanged": len(protocol["sources"]), "saved_model_records": len(records),
        "mature_member_clock_checks": maturity_checks, "saved_available_forecasts_recomputed": len(available),
        "maximum_probability_error": max_probability_error, "maximum_quality_error": max_quality_error,
        "saved_candidate_accounts_checked": 8, "actual_pressure_entry_points": 11,
        "maximum_accounting_error_cny": maximum_error,
        "new_fits_or_account_runs": 0, "independent_validation": "NOT_ESTABLISHED",
        "zero_cash_variance_sharpe": "NOT_ESTIMABLE_NO_SUBSTITUTION_WITH_ZERO",
        "original_A_early_frequency_note": "共同口径原metrics的较早年均只取2015—2018；同完整自然年附表统一2015—2019，原22完成周期年均4.4。仅报告分母纠正，不改原账户或终态。"}, exclusive=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ax, (period, _) in zip(axes, PERIODS.items()):
        baseline = pd.read_parquet(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet")
        ax.plot(baseline.date, baseline.equity/10000, label="原A", color="#286843", linewidth=1.6)
        for policy, label, color in [("TECHNICAL_LOGIT", "技术事件模型", "#ac3635"), ("MATURE_FREQUENCY", "历史频率对照（全现金）", "#7c8288")]:
            d = loaded[(period, "STRESS", policy)]["daily"]
            ax.plot(d.date, d.equity/10000, label=label, color=color, linewidth=1.5)
        ax.set_title(period.replace("_", "—")+"：压力成本完整账户（各自初始20万元）")
        ax.set_ylabel("权益 / 万元")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.savefig(OUT / "压力成本完整账户权益.png", dpi=170)
    plt.close(fig)

    lines = ["# 日周线技术事件模型：实际点位与完整账户结果", "",
             "本固定模型未提高收益、夏普或有效交易次数，TECH.R158终态拒绝。不是独立验证；没有可以准入的新策略。", "",
             "## 回到用户的研究问题", "",
             "对象是510300日线与此前已完成周线，多头优先。次数是软目标，实际扣费p×B>1与正标准期望是硬质量要求；最终需要提高同资金、同费用和风险口径的完整账户收益及全日历夏普。指标能解释过去上涨，仍需另外检验能否在入场前识别净优势。", "",
             "本项采用一个固定学习模型：日/周MACD、均线距离、五日动量、相对量、有向量、波动率比和当日实体效率，预测下一开盘入场后先到收盘盈利边界、亏损边界还是20个收盘到期。初始上2ATR/下1ATR是本项几何定义，不是用户通用2R门槛，也不保证实际平均盈亏比。", "",
             "模型只用拟合时已经退出且权益已成熟的历史参考结果，每月拟合一次；未知保留。对照使用同一个成熟训练池的事件频率。买后固定边界，收盘触发、次合法开盘成交，风险只能减仓。原A保持精确执行对照，不与新模型混合。", "",
             "## 完整账户", "",
             "各账户初始20万元，资产510300.SH/CASH_CNY；252日年化、现金收益0、原50%与ES/跳空/回撤风险限制、T+1、100份一手、0.001报价、最低佣金5元。费用两档全部报告，终点没有人工清仓。", "",
             "| 时期 | 费用 | 政策 | 净年化 | 净夏普 | 最大回撤 | 完成交易 | 胜率 | 实际B | 实际pB |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"A_SAVED_WEIGHT": "原A", "TECHNICAL_LOGIT": "技术事件模型", "MATURE_FREQUENCY": "历史频率对照"}
    for row in stats.itertuples(index=False):
        lines.append(f"| {row.period} | {row.cost} | {names[row.policy]} | {display(row.net_cagr, True)} | {display(row.net_sharpe)} | {display(row.max_drawdown, True)} | {row.completed_cycles} | {display(row.win_rate, True)} | {display(row.payoff)} | {display(row.p_times_b)} |")
    lines.extend(["", "两时期、两费用的整体经济门全部失败。无特征频率对照没有信号、全现金；净收益为0，夏普及交易质量不可估计，不能填成0或无穷。新模型两期均没有触发回撤停止，不能将交易稀少归因于账户停机。", "",
                  "同完整自然年次数另存附表：原A较早22次、2015—2019年均4.4；技术模型6次、年均1.2，其中四年无交易。近期技术模型5次，2020—2025年均0.83，其中三年无交易。共同指标原A较早旧年均字段只取2015—2018，本报告不据该字段作频率比较。", "",
                  "## 所有真实点位", "",
                  "以下为压力成本的全部11次真实进场；两成本是同一政策副本，不算22个独立点位。完整八项事前指标、前完整周日期、锁定模型、预估胜率/盈亏比及参考事件另存CSV。", "",
                  "| 决定日 | 次开进场 | 退出 | 持有区间 | 日MACD/ATR | 周MACD/周波动单位 | 相对量 | 预计pB | 实际净收益率 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in point_frame.itertuples(index=False):
        lines.append(f"| {row.origin:%Y-%m-%d} | {row.entry_date:%Y-%m-%d} | {row.exit_date:%Y-%m-%d} | {row.holding_sessions} | {row.daily_hist_atr:.3f} | {row.weekly_hist_atr:.3f} | {row.relative_volume:.2f} | {row.predicted_p_times_b:.3f} | {row.actual_net_return:+.2%} |")
    lines.extend(["", "## 为什么拒绝", "",
                  "所有真实进场在决定日都满足模型预计pB>1，但实际较早pB仅0.192、近期仅0.138。近期胜率60%仍不够：平均盈利约为平均亏损的0.231倍，标准亏损单位期望−0.262。估计质量门没有兑现成实际质量。", ""])
    for period in PERIODS:
        row = stats.loc[stats.period.eq(period) & stats.cost.eq("STRESS") & stats.policy.eq("TECHNICAL_LOGIT")].iloc[0]
        lines.append(f"{period}实际份额的价格与股息毛损益{row.gross_at_actual_quantities_pnl:+,.2f}元，佣金{row.total_commission:,.2f}元、滑点{row.total_slippage:,.2f}元，最终账户损益{row.ending_equity-200000:+,.2f}元。毛损益已经为负，降低费用不能解释或证明信号有效；这是既有现金流解释，不是另一个可执行策略。")
    lines.extend(["", "事前类别预测同样没有建立稳定信息优势：", "",
                  "| 时期 | 样本口径 | 技术模型logloss | 频率logloss | 技术Brier | 频率Brier |", "|---|---|---:|---:|---:|---:|"])
    for period in PERIODS:
        for group in ["ALL_OVERLAPPING", "FIXED_EVERY_TWENTY_ORIGINS"]:
            a = predictions.loc[predictions.period.eq(period) & predictions.policy.eq("TECHNICAL_LOGIT") & predictions["sample"].eq(group)].iloc[0]
            b = predictions.loc[predictions.period.eq(period) & predictions.policy.eq("MATURE_FREQUENCY") & predictions["sample"].eq(group)].iloc[0]
            lines.append(f"| {period} | {group} | {a.multiclass_logloss:.4f} | {b.multiclass_logloss:.4f} | {a.multiclass_brier:.4f} | {b.multiclass_brier:.4f} |")
    lines.extend(["", "这些误差越低越好。两期全成熟原点的两项误差都比同池频率更差；固定20原点子样本不支持两期稳定改善。模型误差只是解释，未阻止八个完整账户实际运行。重叠原点和142月度拟合不是独立样本或142个独立策略。", "",
                  "20日和252日配对区块各2000次全部保存在summary.json；相对原A未形成收益与夏普的双正下界。全现金对照方差为0，其夏普差及相应区间不可估计，原警告与空值保留。", "",
                  "## 结果可信度与边界", "",
                  f"六项必要测试通过、四个原A账户金额/份额/订单精确一致。保存142个模型成熟成员时钟、{len(available)}个可用预测和八个候选账户已核对；概率和质量复算最大误差{max_probability_error:.3g}/{max_quality_error:.3g}，资金库存最大误差{maximum_error:.3g}元。登记前夹具缺列与日期精度工程失败保留，不作金融失败。报告阶段新增拟合/账户运行0。", "",
                  "全部历史至2026-09-30已用于开发；原价量和股息供应历史首版未认证。按经济时钟排除未来标签不等于物理PIT认证。参考标签不含账户持仓冲突、动态风险减仓和停板延期的全部路径，因此不能用参考pB替代实际账户pB。global DSR/PBO仍NOT_COMPUTED，独立验证/去过拟合未建立。原失败及12项前瞻状态保持。", "",
                  "## 处置与接下来的问题", "",
                  "本固定模型终态拒绝，不修改类别、ATR边界、20日到期、训练窗口、八项指标、C、入场门、成本或时期营救；不按部分有利年份或交易选模型。A/B共享核心和辅助，不能将A/B切换自动当作新信息。", "",
                  "用户最新指出方向偏离，具体偏离处仍待其说明。继续按既有要求把具体上涨和失败点位的事前证据与完整交易结果连起来；已有49段上涨图谱不重新算作新研究，不从这11笔亏损后验筛一个获利规则。目前没有已冻结待跑的新候选，目标未达且保持active。", "",
                  "[压力成本权益图](压力成本完整账户权益.png) · [全部实际点位](results/实际进出点位与事前指标.csv) · [完整账户比较](results/完整账户共同口径比较.csv) · [同完整自然年次数](results/同完整自然年交易次数.csv) · [信号与交易次数](results/预计信号与实际交易次数.csv) · [固定协议](protocol.json) · [保存结果核对](saved_result_verification.json)", "",
                  "实现参考：[scikit-learn LogisticRegression 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。文档只支持分类、正则和样本权重实现，交易标签与规则为本地设计，不提供策略收益证明。", ""])
    (OUT / "研究结果与下一步.md").write_text("\n".join(lines), encoding="utf-8")
    print("已从保存结果完成142模型时钟、5712预测、八账户及全部11真实点位核对；无新拟合或账户运行。", flush=True)


if __name__ == "__main__":
    main()
