"""只读取已保存的 V3 结果，复核必要恒等式并形成历史研究结论。"""
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
STUDY = ROOT / "reports/research/510300_pattern_daily_state_learning_v3"
RESULTS = STUDY / "results"
REVIEW = STUDY / "completion_review"
MODELS = ["MATURE_MEAN", "BACKGROUND", "PHASE_VOLUME"]
NAMES = {"MATURE_MEAN": "历史平均", "BACKGROUND": "趋势与波动", "PHASE_VOLUME": "再加转强速度与成交量"}
CAPITAL = 200000.0
REPORT = STUDY / "历史发现_指数价量状态增量.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_close(left, right, tolerance=1e-7) -> float:
    error = float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f"已存结果复核不一致，最大绝对误差={error}")
    return error


def main() -> None:
    REVIEW.mkdir(exist_ok=True)
    source_summary = read_json(STUDY / "summary.json")
    scope = read_json(STUDY / "current_scope_note_20261001.json")
    comparison = pd.read_csv(RESULTS / "account_comparison.csv")
    prediction = pd.read_csv(RESULTS / "prediction_comparison.csv").set_index("model")
    intervals = pd.read_csv(RESULTS / "paired_intervals.csv").set_index("measure")
    controls = pd.read_parquet(STUDY / "inputs/controls.parquet").set_index("signal_idx")
    if not controls.index.is_unique:
        raise ValueError("控制样本存在重复信号索引。")
    features = pd.read_parquet(STUDY / "inputs/features.parquet")
    forecasts = pd.read_parquet(RESULTS / "daily_forecasts.parquet")
    evaluation = pd.read_csv(RESULTS / "prediction_evaluation.csv")
    signal_predictions = pd.read_csv(RESULTS / "signal_predictions.csv")

    # 只检查会改变结论的时序、资金恒等式及保存指标，不新增参数搜索。
    fitted_models = [read_json(p) for p in sorted((STUDY / "models").glob("*.json"))]
    for fitted in fitted_models:
        pool = controls.loc[fitted["train_signal_indices"]]
        if not (pool.exit_idx <= fitted["fit_idx"]).all():
            raise ValueError("训练样本包含尚未退出的标签。")
        if int(pool.exit_idx.max()) != fitted["latest_mature_exit_idx"]:
            raise ValueError("已存训练标签成熟时间不一致。")
        if len(pool) != fitted["n_train"] or len(pool) < 252:
            raise ValueError("固定训练样本数量不一致。")
        if int(pool.index.min()) < fitted["fit_idx"] - 504:
            raise ValueError("训练样本超出原窗口。")
    if not (forecasts.fit_idx <= forecasts.idx).all():
        raise ValueError("历史预测引用了未来拟合。")
    schedule = pd.read_csv(RESULTS / "fit_schedule.csv")
    if not (schedule.fit_idx.diff().dropna() == 21).all():
        raise ValueError("已存拟合时钟与原规则不一致。")
    for model in MODELS:
        require_close(((evaluation[model] - evaluation.net_return) ** 2).mean(), prediction.loc[model, "mse"], 1e-12)

    reviews, years, stress_ledgers, trades_by_model = [], [], {}, {}
    for model in MODELS:
        for cost in ["BASE", "STRESS"]:
            folder = RESULTS / "accounts" / model / cost
            ledger = pd.read_parquet(folder / "daily.parquet")
            trades = pd.read_csv(folder / "trades.csv")
            saved = read_json(folder / "metrics.json")
            equity = ledger.equity_cny.to_numpy(float)
            returns = equity / np.r_[CAPITAL, equity[:-1]] - 1
            equity_error = require_close(equity, ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny)
            require_close(returns, ledger.daily_return, 1e-12)
            if (ledger.cash_cny < -1e-7).any() or (ledger.receivable_cny < -1e-7).any() or (ledger.shares % 100 != 0).any():
                raise ValueError("账户现金、应收或整手数量不符合原账簿规则。")
            if not (trades.exit_idx > trades.entry_idx).all():
                raise ValueError("保存交易存在当天买入当天卖出。")
            if not (trades.entry_idx > trades.signal_idx).all():
                raise ValueError("保存交易早于收盘信号。")
            pnl = trades.quantity * (trades.exit_price - trades.entry_price) + trades.dividend_cny - trades.entry_fee - trades.exit_fee
            require_close(pnl, trades.net_pnl)
            require_close(trades.entry_fee.sum() + trades.exit_fee.sum(), ledger.fees_cny.sum())
            if saved["open_position"]:
                raise ValueError("本次已存结果出现未预计的终点持仓，需要明确追加解释。")
            require_close(trades.net_pnl.sum(), equity[-1] - CAPITAL)
            cash_delta = pd.Series(0.0, index=ledger.idx.astype(int))
            shares_delta = pd.Series(0, index=ledger.idx.astype(int), dtype="int64")
            for trade in trades.itertuples():
                cash_delta.loc[int(trade.entry_idx)] -= trade.quantity * trade.entry_price + trade.entry_fee
                cash_delta.loc[int(trade.exit_idx)] += trade.quantity * trade.exit_price - trade.exit_fee
                shares_delta.loc[int(trade.entry_idx)] += int(trade.quantity)
                shares_delta.loc[int(trade.exit_idx)] -= int(trade.quantity)
            require_close(CAPITAL + cash_delta.cumsum().to_numpy() + ledger.dividend_paid_cny.cumsum(), ledger.cash_cny)
            require_close(shares_delta.cumsum().to_numpy(), ledger.shares)
            require_close((ledger.dividend_accrual_cny - ledger.dividend_paid_cny).cumsum(), ledger.receivable_cny)
            adjusted = returns.copy()
            adjusted[-1] = (equity[-1] - saved["terminal_haircut_cny"]) / equity[-2] - 1
            nav = np.r_[CAPITAL, CAPITAL * np.cumprod(1 + adjusted)]
            computed = {
                "net_cagr": float((nav[-1] / CAPITAL) ** (252 / len(ledger)) - 1),
                "net_sharpe": float(adjusted.mean() / adjusted.std(ddof=1) * np.sqrt(252)),
                "max_drawdown": float(-(nav / np.maximum.accumulate(nav) - 1).min()),
                "net_profit_cny": float(nav[-1] - CAPITAL),
            }
            for key, value in computed.items():
                require_close(value, saved[key])
            reviews.append({"model": model, "cost": cost, "status": "PASS_SAVED_METRICS_AND_BOOK_IDENTITIES", "equity_identity_max_error_cny": equity_error, "rows": len(ledger), "completed_cycles": len(trades), "maximum_close_exposure": float(ledger.exposure.max())})
            if cost == "STRESS":
                stress_ledgers[model] = ledger
                trades["entry_year"] = trades.entry_date.str[:4].astype(int)
                trades["net_return_on_entry_cost"] = trades.net_pnl / (trades.quantity * trades.entry_price + trades.entry_fee)
                trades_by_model[model] = trades
                for year in range(2021, 2027):
                    block = trades[trades.entry_year == year]
                    signals = signal_predictions[signal_predictions.signal_date.str[:4] == str(year)]
                    years.append({"model": model, "year": year, "full_calendar_year": year < 2026, "confirmed_signals": len(signals), "positive_confirmations": int((signals[model] > 0).sum()), "completed_cycles": len(block), "meets_five_in_full_year": bool(len(block) >= 5) if year < 2026 else None, "saved_trade_net_pnl_cny": float(block.net_pnl.sum())})
    yearly = pd.DataFrame(years)
    yearly.to_csv(REVIEW / "annual_frequency.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reviews).to_csv(REVIEW / "saved_results_checks.csv", index=False, encoding="utf-8-sig")

    phase = trades_by_model["PHASE_VOLUME"]
    background = trades_by_model["BACKGROUND"]
    largest = phase.loc[phase.net_pnl.idxmax()]
    ratio = float(largest.net_pnl / phase.net_pnl.sum())
    winners = phase.loc[phase.net_return_on_entry_cost > 0, "net_return_on_entry_cost"]
    losers = phase.loc[phase.net_return_on_entry_cost < 0, "net_return_on_entry_cost"]
    payoff = float(winners.mean() / -losers.mean())
    keys = ["signal_id", "entry_date", "exit_date", "quantity", "net_pnl", "net_return_on_entry_cost"]
    paired = background[keys].merge(phase[keys], on="signal_id", how="outer", suffixes=("_background", "_phase"), indicator=True)
    paired["saved_pnl_difference_cny"] = paired.net_pnl_phase.fillna(0) - paired.net_pnl_background.fillna(0)
    paired.to_csv(REVIEW / "saved_trade_comparison.csv", index=False, encoding="utf-8-sig")
    require_close(paired.saved_pnl_difference_cny.sum(), phase.net_pnl.sum() - background.net_pnl.sum())

    old = pd.read_csv(STUDY / "inputs/v2_comparison.csv")
    old = old[(old.variant == "V1_UNCHANGED") & (old.policy == "PATTERN_ONLY") & (old.cost == "STRESS")].iloc[0]
    stress = comparison[comparison.cost == "STRESS"].set_index("model")
    mse_change = float(prediction.loc["PHASE_VOLUME", "mse"] / prediction.loc["BACKGROUND", "mse"] - 1)
    sharpe_interval = intervals.loc["SHARPE_PHASE_MINUS_BACKGROUND"]
    boundary = "V3按原冻结参数完成并保留失败；不调整窗口、阈值、特征或有利子期。当前半仓及尾部风险合同账户未运行，本结果不能据此认定其实盘表现。"
    result = {
        "study_id": "510300_PATTERN_DAILY_STATE_LEARNING_V3_COMPLETION_REVIEW",
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "COMPLETED_FROZEN_V3_NO_CONFIRMED_INCREMENT_NO_GOAL_PASS",
        "goal_turn_classification": "PROGRESS_FROZEN_INDEX_STATE_INCREMENT_RESOLVED",
        "previous_goal_turn_classification": scope["previous_goal_turn_classification"],
        "consecutive_blocked_goal_turns": 0,
        "research_unit": "510300整体日线价量状态，非个股择优",
        "period": ["2021-01-04", "2026-09-16"],
        "new_fitted_models": source_summary["new_fitted_models"],
        "refit_dates": source_summary["refit_dates"],
        "new_legacy_accounts": 6,
        "current_risk_contract_accounts": 0,
        "new_20000_cny_accounts": 0,
        "parameter_grids": 0,
        "new_external_sources": 0,
        "minute_data_used": False,
        "new_prospective_forecasts_enabled": False,
        "prediction_mse_relative_change_phase_vs_background": mse_change,
        "prediction_comparison": prediction.reset_index().to_dict("records"),
        "legacy_pressure_accounts": stress.reset_index()[["model", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff_ratio", "average_exposure"]].to_dict("records"),
        "phase_largest_trade": {"signal_id": largest.signal_id, "entry_date": largest.entry_date, "exit_date": largest.exit_date, "net_pnl_cny": float(largest.net_pnl), "fraction_of_total_saved_net_pnl": ratio},
        "phase_other_saved_trades_net_pnl_cny": float(phase.net_pnl.sum() - largest.net_pnl),
        "concentration_is_saved_attribution_not_removed_trade_strategy": True,
        "phase_realized_payoff_by_return": payoff,
        "payoff_definition": "平均正的单笔净收益率除以平均负的单笔净收益率绝对值；净收益率以买入本金及买入费为分母。原账户payoff_ratio按金额计算，二者不同。",
        "phase_full_year_completed_cycles": {str(year): int((phase.entry_year == year).sum()) for year in range(2021, 2026)},
        "phase_partial_2026_completed_cycles": int((phase.entry_year == 2026).sum()),
        "current_acceptance": {"net_sharpe_target": 1.2, "net_cagr_target": 0.10, "max_drawdown_target": 0.10, "maximum_position_fraction": 0.5, "each_full_calendar_year_minimum_cycles": 5, "status": "NOT_RUN_UNDER_CURRENT_CONTRACT"},
        "necessary_verification": {"mature_training_labels": "PASS", "fitted_model_files": len(fitted_models), "saved_accounts_and_metrics": "PASS", "account_count": 6, "daily_ledger_rows": int(sum(item["rows"] for item in reviews)), "execution_realizability": "NOT_ESTABLISHED_BY_LEGACY_DAILY_SIMULATION"},
        "known_legacy_execution_limits": scope["known_legacy_execution_limits"],
        "independence": "历史已经用于多轮研究；按历史时间重放不等于独立样本。1377个五日标签相互重叠。",
        "branch_boundary": boundary,
        "goal_achieved": False,
        "orders_authorized": False,
        "current_view": "NO_VIEW",
        "position_target": "UNSET",
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
    }
    save_json(REVIEW / "result.json", result)

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        import matplotlib.font_manager as font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), gridspec_kw={"width_ratios": [1.4, 1]})
    colors = {"MATURE_MEAN": "#90969f", "BACKGROUND": "#49769b", "PHASE_VOLUME": "#c87736"}
    for model in MODELS:
        ledger = stress_ledgers[model]
        axes[0].plot(pd.to_datetime(ledger.date), (ledger.equity_cny - CAPITAL) / 10000, label=NAMES[model], color=colors[model], linewidth=1.8)
    axes[0].axvspan(pd.Timestamp(largest.entry_date), pd.Timestamp(largest.exit_date), color="#c87736", alpha=.13)
    axes[0].set(title="原账户累计净利润：2024年单笔贡献突出", ylabel="累计净利润（万元）", xlabel="历史日期")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=.2)
    year_values = np.arange(2021, 2027)
    for j, model in enumerate(["BACKGROUND", "PHASE_VOLUME"]):
        counts = yearly[yearly.model == model].set_index("year").loc[year_values, "completed_cycles"]
        bars = axes[1].bar(year_values + (j - .5) * .3, counts, width=.3, color=colors[model], label=NAMES[model])
        bars[-1].set_hatch("///")
    axes[1].axhline(5, color="#a63232", linestyle="--", linewidth=1.3, label="当前完整年度要求：5次")
    axes[1].set_xticks(year_values, [str(year) if year < 2026 else "2026\n截至9/16" for year in year_values])
    axes[1].set(title="自然完成次数：完整年度均不足5次", ylabel="周期数（按入场年归属）", ylim=(0, 6))
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("510300历史固定版本：价量变化未提供可确认增量", fontsize=16, y=.98)
    fig.text(.5, .02, "原近满仓、开盘成交假设下的压力成本诊断；不是当前半仓及尾部风险合同的验收结果。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .94))
    figure_path = REVIEW / "V3_历史账户与年度机会.png"
    fig.savefig(figure_path, dpi=155, facecolor="white")
    plt.close(fig)

    rows = [f"| 原形态、不加状态模型（旧结果） | {old.net_cagr:.2%} | {old.net_sharpe:.3f} | {old.max_drawdown:.2%} | {int(old.completed_cycles)} |"]
    for model in MODELS:
        row = stress.loc[model]
        rows.append(f"| {NAMES[model]} | {row.net_cagr:.2%} | {row.net_sharpe:.3f} | {row.max_drawdown:.2%} | {int(row.completed_cycles)} |")
    year_lines = []
    for year in range(2021, 2027):
        selected = yearly[yearly.year == year].set_index("model")
        label = str(year) if year < 2026 else "2026截至9月16日（不完整）"
        year_lines.append(f"| {label} | {int(selected.loc['PHASE_VOLUME','confirmed_signals'])} | {int(selected.loc['BACKGROUND','completed_cycles'])} | {int(selected.loc['PHASE_VOLUME','completed_cycles'])} |")
    text = f"""# 510300：历史价量状态有没有增量？

**结论：本次固定版本没有证明‘转强速度与成交活跃度’带来稳定的额外收益信息，夏普1.2目标仍未达到。** 这是指数ETF整体状态的历史研究，不涉及个股筛选，也没有生成当前市场观点。

原V3在2026年9月24日冻结，本次按原参数只执行一次。1384个交易日来自2021年1月4日至2026年9月16日；没有为争取成功延长样本、调参或挑选子期。研究对象是原42个形态确认，以及当时已经成熟的普通日五日收益。模型每21个交易日使用此前504个交易日的成熟样本重新拟合，共66个拟合时点、132个模型文件；这是固定规则的历史滚动拟合，不是132套待挑选策略。

三个比较对象分别是历史收益平均值、趋势与相对波动背景，以及在背景之上增加三日转强速度与相对成交量。全部为510300日线价格和成交量的代理指标；它们没有直接测量政策原因、投资者持仓或真实资金净流入。

## 直接结果

下表都是原20万元、近满仓结构的完整历史账户，包含空仓日和压力成本。现金收益及夏普无风险收益扣减均按原代码取0，年化使用252个交易日。旧形态行来自冻结时复制的旧结果，仅作描述性参照，未据此重新选择策略。

| 原固定账户 | 净年化 | 净夏普 | 最大回撤 | 自然完成周期 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

加入两个变量后，夏普相对背景模型增加{stress.loc['PHASE_VOLUME','net_sharpe']-stress.loc['BACKGROUND','net_sharpe']:.3f}；原先固定的20日区块配对重抽样给出的该差值95%区间为[{sharpe_interval.lower95:.3f}, {sharpe_interval.upper95:.3f}]，不能据此确认稳定提升。区间评价的是已保存的账户日收益，不是在每个重抽样中重新拟合和交易，也没有消除历史研究选择的影响。

更直接的检验方向相反：1377个成熟五日标签上，背景模型的均方误差为{prediction.loc['BACKGROUND','mse']:.9f}，增加价量变化后为{prediction.loc['PHASE_VOLUME','mse']:.9f}，**误差扩大{mse_change:.2%}**。均方误差较小的还是简单历史平均值（{prediction.loc['MATURE_MEAN','mse']:.9f}）。原区块区间中“背景误差−新增变量误差”全为负。这个结论限定在本模型、本标签和该历史样本内，不能扩大为所有阶段中成交量都无效；1377个重叠标签也不等于1377次独立事件。

## 高胜率、较高赔率为何仍未达到账户目标

新增变量账户压力成本下11笔中7笔盈利，胜率63.64%；按单笔净收益率计算的已实现赔率为{payoff:.2f}，按原代码的金额口径则是{stress.loc['PHASE_VOLUME','payoff_ratio']:.2f}。这两个赔率定义不同，均不是买入前可承诺的赔率。

该账户累计净利润{phase.net_pnl.sum():,.2f}元，其中{largest.entry_date}至{largest.exit_date}一笔为{largest.net_pnl:,.2f}元，占{ratio:.2%}；其余10笔已保存利润合计{phase.net_pnl.sum()-largest.net_pnl:,.2f}元。背景模型也抓到了同一笔，因而不能把这笔行情解释成新增价量变量独有的能力。这是已存交易金额归因，**没有删除该笔后重新复利，也不是删除大行情后的策略业绩**。

原42次形态确认经模型启停后，只剩11个完成周期，平均持仓敞口约{stress.loc['PHASE_VOLUME','average_exposure']:.2%}（开仓时接近满仓）。结果改善同时伴随着很少交易和高度集中的利润，尚未形成足够重复的年度机会。

| 历史年份 | 原形态确认数 | 趋势与波动完成次数 | 再加价量变化完成次数 |
|---|---:|---:|---:|
{chr(10).join(year_lines)}

2021—2025五个完整自然年全部低于当前每年至少5个自然完成周期的要求。2026是不完整年度，另列已发生次数，不认定全年失败。年度门槛只评价最终策略；不因这个节点稀少而人工补交易，也不从有利年份提取新开关。

## 信号、成本和账户约束分别判断

信号方面，新增变量未改善全部普通日标签的预测误差，账户微小增量也未得到原区块比较的支持。成本方面，改为原基础成本档，新增变量账户夏普仍只有{float(comparison[(comparison.model=='PHASE_VOLUME') & (comparison.cost=='BASE')].iloc[0].net_sharpe):.3f}，所以在两档既定成本之间，降低费用没有解决缺口。账户方面，原版本并未施加当前50%仓位和尾部风险合同；不能把现有低收益解释成当前半仓限制造成的，也不能把原满仓结果当成当前账户表现的数学上界。

本次必要复核覆盖132个模型的成熟标签时间、6个账户共8304条日账的现金/股份/应收恒等式、买卖先后以及保存的净值指标，均一致。**这只说明结果能从已存账目复算，不代表原成交假设已被证明。**

原代码观察实际开盘价是否越过失效线，并以这个开盘价计算数量，再按同一开盘价加滑点成交；还用当日总成交量辅助判定可成交。日线无法证明这些数量能在该开盘时点实际成交。当前50%仓位、五日ES、缺口预算和盘前确定订单的账户，以及2万元比较账户，本次均未运行。旧协议中的年度次数取消和独立前向要求不改变当前历史研究及年度次数授权。

本分支结论保留为失败，不调模型、不改窗口、不反号、不拼接年份。下一步若提出新的指数原因机制，应有新增的原因观测；本次价量结果不构成继续堆叠技术变量的依据。目标状态保持未实现。

![历史账户和年度机会](<{figure_path.as_posix()}>)

原始结果：[summary.json](summary.json)；原冻结协议：[protocol.json](protocol.json)；本次范围：[current_scope_note_20261001.json](current_scope_note_20261001.json)；可机读复核：[completion_review/result.json](completion_review/result.json)；逐年次数：[annual_frequency.csv](completion_review/annual_frequency.csv)；已存交易对照：[saved_trade_comparison.csv](completion_review/saved_trade_comparison.csv)。
"""
    REPORT.write_text(text, encoding="utf-8")
    print("已完成历史V3必要复核与说明；6个原账户未达目标，当前风险合同账户未运行。")
    print(json.dumps({"报告": str(REPORT), "净夏普": float(stress.loc["PHASE_VOLUME", "net_sharpe"]), "预测误差增加": mse_change, "最大单笔占总净利": ratio, "目标已达成": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
