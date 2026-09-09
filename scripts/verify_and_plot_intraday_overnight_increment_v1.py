"""只读取本轮冻结结果，核对账户算术并绘图；不拟合或重跑策略。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_intraday_overnight_increment_v1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    config = json.loads((ROOT / "config/510300_intraday_overnight_increment_v1.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT / "receipt.json").read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        check(sha(ROOT / name) == identity["sha256"], f"冻结文件变化：{name}")
    for name, identity in receipt["output_files"].items():
        check(sha(OUT / name) == identity["sha256"], f"原始运行输出变化：{name}")
    comparison = pd.read_csv(OUT / "comparison.csv").set_index(["scenario", "model"])
    dividends = pd.read_csv(ROOT / config["inputs"]["dividends"], parse_dates=["record_date", "ex_date", "payment_date"])
    prices = pd.read_parquet(ROOT / config["inputs"]["prices"]).set_index("date")
    reports = {}
    saved = {}
    for (scenario, model), metrics in comparison.iterrows():
        name = f"{scenario}_{model}"
        ledger = pd.read_csv(OUT / "ledgers" / f"{name}.csv", parse_dates=["date"])
        trades = pd.read_csv(OUT / "trades" / f"{name}.csv", parse_dates=["date", "origin"])
        saved[name] = ledger
        check(ledger.date.is_monotonic_increasing and not ledger.date.duplicated().any(), "日账日期不连续有序")
        dates = ledger.date
        expected_dates = prices.index[(prices.index >= dates.iloc[0]) & (prices.index <= dates.iloc[-1])]
        check(pd.DatetimeIndex(dates).equals(expected_dates), "账户跳过了评价交易日")
        quantities = trades.groupby("date").filled_quantity.sum().reindex(dates, fill_value=0).to_numpy(float)
        rebuilt_shares = np.cumsum(quantities)
        check(np.array_equal(rebuilt_shares, ledger.shares), "成交份额不能还原持仓")
        check((ledger.shares >= 0).all() and (ledger.shares % 100 == 0).all(), "持仓超范围或非整手")
        before = np.r_[0, rebuilt_shares[:-1]]
        check((np.maximum(-quantities, 0) <= before).all(), "卖出了当天新买份额")
        check((trades.origin < trades.date).all(), "交易不在信息日之后")
        holding_on_record = ledger.set_index("date").shares
        entitlement = []
        for event in dividends.itertuples():
            count = int(holding_on_record.get(event.record_date, 0))
            if count:
                entitlement.append((event.ex_date, event.payment_date, count * float(event.cash_dividend_per_share)))
        cash_spent = trades.assign(spent=trades.filled_quantity * trades.fill_price.fillna(0) + trades.commission).groupby("date").spent.sum().reindex(dates, fill_value=0).to_numpy(float)
        rebuilt_cash, rebuilt_receivable = [], []
        for position, row in enumerate(ledger.itertuples()):
            is_close = row.mark_clock == "CLOSE"
            paid = sum(amount for ex, payment, amount in entitlement if payment < row.date or (payment == row.date and is_close))
            outstanding = sum(amount for ex, payment, amount in entitlement if ex <= row.date and (payment > row.date or (payment == row.date and not is_close)))
            rebuilt_cash.append(config["initial_capital"] - float(cash_spent[:position + 1].sum()) + paid)
            rebuilt_receivable.append(outstanding)
        max_cash_error = float(np.max(np.abs(np.array(rebuilt_cash) - ledger.cash)))
        max_receivable_error = float(np.max(np.abs(np.array(rebuilt_receivable) - ledger.dividend_receivable)))
        marks = prices.loc[dates, "close"].to_numpy(dtype=float, copy=True)
        marks[-1] = float(prices.loc[dates.iloc[-1], "open"])
        reconstructed = np.array(rebuilt_cash) + rebuilt_shares * marks + np.array(rebuilt_receivable)
        max_nav_error = float(np.max(np.abs(reconstructed - ledger.equity)))
        check(max_cash_error < 1e-6 and max_receivable_error < 1e-8 and max_nav_error < 1e-6, "独立现金、应收或净值算术不符")
        check((np.array(rebuilt_cash) >= -1e-7).all(), "独立现金账出现透支")
        net = ledger.net_return.to_numpy(float)
        cumulative = float(np.prod(1 + net) - 1)
        check(abs(cumulative - metrics.cumulative_return) < 1e-10, "净累计收益不符")
        check(abs(config["initial_capital"] * (1 + cumulative) - reconstructed[-1]) < 1e-6, "日收益与期末财富不一致")
        check(abs(float(trades.commission.sum() + trades.slippage_cost.sum()) - metrics.total_cost_cny) < 1e-7, "费用总和不符")
        check(abs(ledger.pnl.sum() - (reconstructed[-1] - config["initial_capital"])) < 1e-6, "净损益不能加总")
        cost = config["costs"][scenario]
        filled = trades.loc[trades.filled_quantity.ne(0)].copy()
        expected_commission = np.maximum(filled.filled_quantity.abs() * filled.fill_price * cost["commission"], cost["minimum"])
        max_commission_error = float(np.max(np.abs(expected_commission - filled.commission))) if len(filled) else 0.0
        check(max_commission_error < 1e-7, "每笔佣金不符合冻结费率")
        expected_slip = filled.filled_quantity.abs() * (filled.fill_price - filled.open_price).abs()
        check(np.allclose(expected_slip, filled.slippage_cost, atol=1e-7), "滑点费用不是实际价差")
        for row in filled.itertuples():
            check(abs(row.open_price - prices.loc[row.date, "open"]) < 1e-10, "成交基准不是当日原始开盘")
            check(abs(row.filled_quantity) <= abs(row.requested_quantity), "执行增加了已冻结的份额请求")
        reports[name] = {"days": len(ledger), "start": str(dates.iloc[0].date()), "end_open": str(dates.iloc[-1].date()),
                         "max_independent_cash_error_cny": max_cash_error, "max_independent_receivable_error_cny": max_receivable_error,
                         "max_independent_equity_error_cny": max_nav_error, "max_commission_error_cny": max_commission_error,
                         "t_plus_one_and_lots": True, "cash_nonnegative": True, "terminal_shares": int(ledger.shares.iloc[-1])}
    samples = pd.read_csv(OUT / "conditional_samples.csv", parse_dates=["origin", "entry_date", "exit_date"])
    models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    prediction_errors = {}
    for name, model in models.items():
        x = samples[model["features"]].to_numpy(float)
        recomputed = ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["slopes"]) + model["intercept"]
        error = float(np.max(np.abs(recomputed - samples[f"prediction_{name}"])))
        prediction_errors[name] = error
        check(error < 1e-12, "预测与冻结参数不符")
        training = samples.loc[samples.train_eligible]
        check((training.exit_date <= pd.Timestamp(config["train_cutoff"])).all(), "训练标签跨界")
        check(np.allclose(training[model["features"]].mean().to_numpy(), model["mean"], atol=1e-12), "标准化使用了训练外均值")
        check(np.allclose(training[model["features"]].std(ddof=0).to_numpy(), model["scale"], atol=1e-12), "标准化使用了训练外方差")
    path_differences = {}
    for model in ["M0", "M1"]:
        base = pd.read_csv(OUT / "decisions" / f"BASE_{model}.csv", parse_dates=["origin"])
        stress = pd.read_csv(OUT / "decisions" / f"STRESS_{model}.csv", parse_dates=["origin"])
        paired = base.merge(stress, on="origin", suffixes=("_base", "_stress"), validate="one_to_one")
        action_different = paired.action_base != paired.action_stress
        substantial = (paired.reference_weight_base - paired.reference_weight_stress).abs() > 0.20
        examples = paired.loc[substantial, ["origin", "action_base", "action_stress", "reference_weight_base", "reference_weight_stress"]].head(5).copy()
        examples["origin"] = examples.origin.dt.strftime("%Y-%m-%d")
        path_differences[model] = {"decision_origins": len(paired), "different_action_labels": int(action_different.sum()),
                                   "exposure_difference_over_20_percentage_points": int(substantial.sum()), "examples": examples.to_dict("records")}
    proof = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "PASS_RESULT_ARITHMETIC_AND_DELIVERY_CHECKS",
             "scope": "只核对冻结文件、结果算术和交付，不拟合、不重跑策略、不做安全性审计",
             "frozen_files_unchanged": len(manifest["files"]), "original_output_files_unchanged": len(receipt["output_files"]),
             "account_checks": reports, "saved_parameter_prediction_max_errors": prediction_errors,
             "base_stress_decision_path_differences": path_differences,
             "new_models_fitted": 0, "new_strategy_runs": 0, "final_model_not_created_after_rejection": not (OUT / "final_full_history_models.json").exists()}
    (OUT / "delivery_checks.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_chart(saved)
    note = ["# 本轮交付核对与成本情景解释", "", "一次性主运行完成，状态为 `STOP_REPRESENTATION_NO_PARAMETER_RESCUE`。", "",
            f"原冻结的{len(manifest['files'])}项文件及原运行回执列明的{len(receipt['output_files'])}项输出均一致。九个模拟账户已独立用成交、登记日份额和分红付款日重建现金、应收及净值；没有重拟合或重跑策略。", "",
            "压力情景的M1收益高于基础情景，是因为费用事先进入了Q评分，影响是否交易和切换时点；它并不是在同一成交路径上只多扣一次费用。不得将这一差别解释为高费用创造收益，也不能选择压力情景替代基础情景完成验收。", ""]
    for model, info in path_differences.items():
        note.append(f"{model}：{info['decision_origins']}个收盘决策原点中，{info['different_action_labels']}个动作标签不同；{info['exposure_difference_over_20_percentage_points']}个原点的参考暴露相差超过20个百分点。")
    note.extend(["", "M1基础成本下净累计-4.32%、压力下+9.29%，都低于对应买入持有；基础相对M0为负，训练到评价的D20条件方向反转。按预先规则停止，不继续最终全历史拟合，不启用前向信号。", "",
                 "`comparison_chart.png`只绘制已保存日账，`delivery_checks.json`保存独立核对结果。原始主运行报告和回执原样保留。", ""])
    (OUT / "DELIVERY_NOTE.md").write_text("\n".join(note), encoding="utf-8")
    extras = [OUT / "comparison_chart.png", OUT / "comparison_chart.pdf", OUT / "delivery_checks.json", OUT / "DELIVERY_NOTE.md", Path(__file__).resolve()]
    identity = {"checked_at": proof["checked_at"], "primary_receipt_sha256": sha(OUT / "receipt.json"),
                "supplement_files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in extras}}
    (OUT / "delivery_receipt.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"核对状态": proof["status"], "模拟账户": len(reports), "冻结文件": len(manifest["files"]), "原始输出": len(receipt["output_files"]), "绘图": str(OUT / "comparison_chart.png"), "重拟合或重跑": False}, ensure_ascii=False))


def draw_chart(ledgers: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    check(font_path.exists(), "缺少绘图中文字体")
    font = FontProperties(fname=str(font_path))
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharey=True, facecolor="#f8f8f5")
    labels = {"BUY_HOLD": "510300 买入持有（分红现金留存）", "M0": "M0 · 总收益 + 波动", "M1": "M1 · 再加日内—隔夜分歧"}
    colors = {"BUY_HOLD": "#75818a", "M0": "#167b8c", "M1": "#b24a36"}
    for ax, scenario, title in zip(axes, ["BASE", "STRESS"], ["基础成本", "压力成本"]):
        ax.set_facecolor("#f8f8f5")
        for model in labels:
            ledger = ledgers[f"{scenario}_{model}"]
            times = ledger.date + pd.to_timedelta(15, unit="h")
            times.iloc[-1] = ledger.date.iloc[-1] + pd.Timedelta(hours=9, minutes=30)
            times = pd.concat([pd.Series([ledger.date.iloc[0] + pd.Timedelta(hours=9, minutes=30)]), times], ignore_index=True)
            wealth = np.r_[1.0, ledger.equity.to_numpy(float) / 200000]
            ax.plot(times, wealth, color=colors[model], linewidth=1.8 if model != "M1" else 2.1, label=labels[model])
            ax.annotate(f"{wealth[-1]:.3f}", (times.iloc[-1], wealth[-1]), xytext=(5, 0), textcoords="offset points", va="center", color=colors[model], fontsize=10)
        ax.set_title(title, loc="left", fontproperties=font, fontsize=13, pad=12)
        ax.axhline(1, color="#bcc2c3", linewidth=0.7, linestyle="--")
        ax.grid(axis="y", color="#d8dcd9", linewidth=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#c3c9c7")
        ax.margins(x=0.04)
    axes[0].set_ylabel("模拟净值（初始本金 = 1）", fontproperties=font)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False, prop=font)
    fig.suptitle("日内—隔夜分歧：固定模型的连续历史回放", x=0.055, ha="left", fontproperties=font, fontsize=18)
    fig.text(0.055, 0.89, "2020-01-02 开盘至 2026-08-14 开盘 · 2019年训练截止 · 评价期间不重训", fontproperties=font, fontsize=10, color="#556267")
    fig.text(0.055, 0.02, "费用参与预先冻结的交易评分，因此基础与压力情景的交易路径可能不同。图示为已观察历史中的模拟结果。", fontproperties=font, fontsize=9, color="#556267")
    fig.subplots_adjust(left=0.06, right=0.96, top=0.80, bottom=0.20, wspace=0.12)
    fig.savefig(OUT / "comparison_chart.png", dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(OUT / "comparison_chart.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
