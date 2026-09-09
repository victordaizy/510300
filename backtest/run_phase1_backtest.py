"""运行第一阶段冻结候选与双基准回测。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
MODEL_FILE = PROJECT_ROOT / "config" / "phase1_frozen_candidate.yaml"
ETF_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
TOTAL_RETURN_INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "000300_market_state_daily.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "backtest"
LEDGER_DIR = PROJECT_ROOT / "data" / "processed" / "backtest"


def build_hysteresis_target(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    data = features[["date", model["feature"]]].copy().sort_values("date")
    data["date"] = pd.to_datetime(data["date"])
    low = float(model["rules"]["enter_or_hold_full_when_lte"])
    high = float(model["rules"]["exit_or_hold_cash_when_gte"])
    if not 0 <= low < high <= 1:
        raise ValueError("估值滞回阈值必须满足0 <= low < high <= 1")
    state = float(model["initial_state"])
    targets: list[float] = []
    for value in data[model["feature"]]:
        if pd.isna(value):
            targets.append(state)
            continue
        if value <= low:
            state = 1.0
        elif value >= high:
            state = 0.0
        targets.append(state)
    data["target_position"] = targets
    return data[["date", "target_position"]]


def theoretical_etf_total_return_series(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    market = prices.loc[prices["date"].between(start_date, end_date)].copy().sort_values("date")
    if len(market) < 2:
        raise ValueError("基准区间至少需要两个交易日")
    entry_date = market["date"].iloc[1]
    entry_price = float(market["open"].iloc[1])
    shares = 1.0 / entry_price
    event_map = (
        dividends.loc[
            (dividends["ex_date"] > entry_date) & (dividends["ex_date"] <= end_date)
        ]
        .groupby("ex_date")["cash_dividend_per_share"]
        .sum()
        .to_dict()
    )
    rows = []
    for row in market.iloc[1:].itertuples(index=False):
        date = pd.Timestamp(row.date)
        dividend = float(event_map.get(date, 0.0))
        if dividend:
            shares += shares * dividend / float(row.close)
        rows.append({"date": date, "equity": shares * float(row.close), "dividend_per_share": dividend})
    result = pd.DataFrame(rows)
    result["equity"] /= result["equity"].iloc[0] / (float(market["close"].iloc[1]) / entry_price)
    result["daily_return"] = result["equity"].pct_change().fillna(0.0)
    return result


def index_total_return(
    index_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    frame = index_data.loc[index_data["date"].between(start_date, end_date)].sort_values("date")
    if len(frame) < 2:
        return float("nan")
    return float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)


def annual_returns(equity: pd.DataFrame, initial_value: float) -> dict[str, float]:
    data = equity[["date", "equity"]].copy()
    data["year"] = data["date"].dt.year
    year_ends = data.groupby("year", sort=True)["equity"].last()
    output: dict[str, float] = {}
    previous = initial_value
    for year, value in year_ends.items():
        output[str(int(year))] = float(value / previous - 1.0)
        previous = float(value)
    return output


def summarize_theoretical_benchmark(series: pd.DataFrame) -> dict:
    total_return = float(series["equity"].iloc[-1] - 1.0)
    elapsed_days = max((series["date"].iloc[-1] - series["date"].iloc[0]).days, 1)
    daily_return = series["daily_return"]
    volatility = float(daily_return.std(ddof=1) * np.sqrt(242))
    drawdown = series["equity"] / series["equity"].cummax() - 1.0
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": float(daily_return.mean() * 242 / volatility) if volatility > 0 else None,
        "max_drawdown": float(drawdown.min()),
    }


def render_markdown(report: dict) -> str:
    full = report["periods"]["full_five_years"]
    strategy = full["strategy"]
    etf = full["benchmark_a_etf_total_return"]
    realistic = full["benchmark_a_tradable_buy_hold"]
    next_valid_oos = report["model_governance"].get("next_valid_oos") or "等待正式候选冻结后的下一个交易日"
    lines = [
        "# 510300 第一阶段五年回测（旧探索产物，未获批准）", "",
        "> 治理状态：`EXPLORATORY_NOT_APPROVED`。本报告生成于正式Target与因子讨论之前，不代表项目策略；当前正式结果以第二轮选择决定为准。", "",
        "## 结论", "",
        f"- 旧探索规则：`{report['model_id']}`，五年净收益 {strategy['total_return']:.2%}，"
        f"510300分红再投资基准 {etf['total_return']:.2%}，超额 {full['excess_vs_etf_total_return']:.2%}。",
        f"- 策略最大回撤 {strategy['max_drawdown']:.2%}，基准最大回撤 {etf['max_drawdown']:.2%}。",
        f"- 现实整手买入持有净收益 {realistic['total_return']:.2%}；沪深300全收益指数同期 {full['benchmark_b_h00300_total_return']:.2%}。",
        f"- 15bp滑点压力测试净收益 {full['stress_15bps']['total_return']:.2%}；策略验收状态：`{report['strategy_gate']['status']}`。",
        f"- 2025-08-01至2026-08-11不是严格未见样本，只能作为受污染回顾测试；{next_valid_oos}。", "",
        "## 固定口径", "",
        "- 信号：t日收盘形成；成交：最早t+1开盘。",
        "- 股票ETF按T+1；100份整手；仅做多；不使用杠杆。",
        f"- 成本：佣金 {report['costs']['commission_rate']:.4%}，最低{report['costs']['minimum_commission_cny']:.2f}元，"
        f"基础滑点 {report['costs']['slippage_bps']:.1f}bp，ETF印花税 {report['costs']['stamp_duty_rate']:.4%}。",
        "- 分红在除息日确认为应收、发放日转为现金；基准A按分红再投资计算。",
        "- H00300数据源只含收盘点位，基准B采用区间首尾收盘比较，仅作跟踪差异诊断。",
        "- 年度20%目标定义为自然年度策略收益减510300分红再投资收益，不作为训练损失函数。", "",
        "## 分区结果", "",
        "|区间|治理状态|策略收益|ETF总回报|超额|策略最大回撤|交易次数|平均仓位|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["periods"].items():
        lines.append(
            f"|{name}|{item['governance']}|{item['strategy']['total_return']:.2%}|"
            f"{item['benchmark_a_etf_total_return']['total_return']:.2%}|{item['excess_vs_etf_total_return']:.2%}|"
            f"{item['strategy']['max_drawdown']:.2%}|{item['strategy']['trade_count']}|"
            f"{item['strategy']['average_exposure']:.2%}|"
        )
    lines += ["", "## 自然年度结果（完整五年，仅回顾性）", "", "|年份|策略|ETF总回报|超额|达到20%超额|", "|---|---:|---:|---:|---|" ]
    for year, item in full["calendar_years"].items():
        lines.append(
            f"|{year}|{item['strategy']:.2%}|{item['etf_total_return']:.2%}|{item['excess']:.2%}|"
            f"{item['meets_20pct_excess']}|"
        )
    lines += ["", "## 锚定式Walk-forward（回顾性）", "", "|折|训练截止|隔离交易日|测试区间|策略收益|ETF总回报|超额|", "|---|---|---:|---|---:|---:|---:|"]
    for fold in report["anchored_walk_forward"]:
        lines.append(
            f"|{fold['fold']}|{fold['train_end']}|{fold['embargo_trading_days']}|"
            f"{fold['test_start']}至{fold['test_end']}|{fold['strategy_total_return']:.2%}|"
            f"{fold['etf_total_return']:.2%}|{fold['excess']:.2%}|"
        )
    lines += [
        "", "## 研究判断", "",
        f"- 当前策略门槛：{report['strategy_gate']['status']}。"
        f"{report['strategy_gate']['reason']}",
        "- 验证区间与回顾测试区间没有持续同向超额，因此基础Edge尚未成立，不进入FVG和PV/LC优化。",
        "- 趋势、波动率没有因本次回测而改方向或调阈值；如需继续，只能预注册新假设并等待新的前向样本。",
        "- 详细净值和逐笔成交保存在 `data/processed/backtest`。", "",
    ]
    return "\n".join(lines)


def run_walk_forward(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    initial_cash: float,
    costs: BacktestCosts,
) -> list[dict]:
    calendar = prices["date"].sort_values().reset_index(drop=True)
    definitions = [
        ("WF1", "2022-12-30", "2023-12-29"),
        ("WF2", "2023-12-29", "2024-12-31"),
        ("WF3", "2024-12-31", "2025-06-30"),
    ]
    output = []
    for fold, train_end_text, test_end_text in definitions:
        train_end = calendar.loc[calendar <= pd.Timestamp(train_end_text)].max()
        train_index = int(calendar.loc[calendar == train_end].index[0])
        test_start_index = train_index + 21
        if test_start_index >= len(calendar):
            continue
        test_start = pd.Timestamp(calendar.iloc[test_start_index])
        test_end = min(pd.Timestamp(test_end_text), pd.Timestamp(calendar.max()))
        ledger, trades = run_long_cash_backtest(
            prices, dividends, targets, initial_cash, costs, test_start, test_end
        )
        theoretical = theoretical_etf_total_return_series(prices, dividends, test_start, test_end)
        strategy = summarize_backtest(ledger, trades, initial_cash)
        benchmark = summarize_theoretical_benchmark(theoretical)
        output.append(
            {
                "fold": fold,
                "train_end": str(pd.Timestamp(train_end).date()),
                "embargo_trading_days": 20,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "strategy_total_return": strategy["total_return"],
                "etf_total_return": benchmark["total_return"],
                "excess": strategy["total_return"] - benchmark["total_return"],
            }
        )
    return output


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    model = yaml.safe_load(MODEL_FILE.read_text(encoding="utf-8"))
    prices = pd.read_parquet(ETF_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    features = pd.read_parquet(FEATURE_FILE)
    features["date"] = pd.to_datetime(features["date"])
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    total_return_index = pd.read_parquet(TOTAL_RETURN_INDEX_FILE)
    total_return_index["date"] = pd.to_datetime(total_return_index["date"])
    targets = build_hysteresis_target(features, model)
    buy_hold_targets = targets.assign(target_position=1.0)

    costs_config = settings["backtest"]
    costs = BacktestCosts(
        commission_rate=float(costs_config["commission_rate"]),
        minimum_commission_cny=float(costs_config["minimum_commission_cny"]),
        stamp_duty_rate=float(costs_config["stamp_duty_rate"]),
        slippage_bps=float(costs_config["slippage_bps_base"]),
        lot_size=int(costs_config["lot_size"]),
        cash_annual_rate=float(costs_config["cash_annual_rate"]),
    )
    stress_costs = BacktestCosts(
        commission_rate=costs.commission_rate,
        minimum_commission_cny=costs.minimum_commission_cny,
        stamp_duty_rate=costs.stamp_duty_rate,
        slippage_bps=float(costs_config["slippage_bps_stress"]),
        lot_size=costs.lot_size,
        cash_annual_rate=costs.cash_annual_rate,
    )
    initial_cash = float(costs_config["initial_cash"])
    split = settings["research_split"]
    periods = {
        "development": (split["development_start"], split["development_end"], "模型开发"),
        "validation": (split["validation_start"], split["validation_end"], "顺序验证"),
        "retrospective_test": (split["final_holdout_start"], split["final_holdout_end"], "受污染，不用于选参"),
        "full_five_years": (settings["project"]["start_date"], "2026-08-11", "完整回顾，不是样本外"),
    }

    report_periods: dict[str, dict] = {}
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_strategy_ledger = pd.DataFrame()
    full_benchmark_series = pd.DataFrame()
    for name, (start_text, end_text, governance) in periods.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        strategy_ledger, strategy_trades = run_long_cash_backtest(
            prices, dividends, targets, initial_cash, costs, start, end
        )
        buy_hold_ledger, buy_hold_trades = run_long_cash_backtest(
            prices, dividends, buy_hold_targets, initial_cash, costs, start, end
        )
        theoretical = theoretical_etf_total_return_series(prices, dividends, start, end)
        strategy_summary = summarize_backtest(strategy_ledger, strategy_trades, initial_cash)
        realistic_summary = summarize_backtest(buy_hold_ledger, buy_hold_trades, initial_cash)
        theoretical_summary = summarize_theoretical_benchmark(theoretical)
        item = {
            "governance": governance,
            "strategy": strategy_summary,
            "benchmark_a_etf_total_return": theoretical_summary,
            "benchmark_a_tradable_buy_hold": realistic_summary,
            "benchmark_b_h00300_total_return": index_total_return(total_return_index, start, end),
            "excess_vs_etf_total_return": strategy_summary["total_return"] - theoretical_summary["total_return"],
        }
        report_periods[name] = item
        strategy_ledger.to_parquet(LEDGER_DIR / f"{name}_strategy_ledger.parquet", index=False)
        strategy_trades.to_parquet(LEDGER_DIR / f"{name}_strategy_trades.parquet", index=False)
        if name == "full_five_years":
            full_strategy_ledger = strategy_ledger
            full_benchmark_series = theoretical
            stress_ledger, stress_trades = run_long_cash_backtest(
                prices, dividends, targets, initial_cash, stress_costs, start, end
            )
            report_periods[name]["stress_15bps"] = summarize_backtest(
                stress_ledger, stress_trades, initial_cash
            )

    full_strategy_years = annual_returns(full_strategy_ledger, initial_cash)
    benchmark_for_years = full_benchmark_series.copy()
    benchmark_for_years["equity"] *= initial_cash
    full_benchmark_years = annual_returns(benchmark_for_years, initial_cash)
    year_rows = {}
    for year in sorted(set(full_strategy_years).intersection(full_benchmark_years)):
        excess = full_strategy_years[year] - full_benchmark_years[year]
        year_rows[year] = {
            "strategy": full_strategy_years[year],
            "etf_total_return": full_benchmark_years[year],
            "excess": excess,
            "meets_20pct_excess": bool(excess >= float(settings["objective"]["annual_excess_target"])),
        }
    report_periods["full_five_years"]["calendar_years"] = year_rows

    walk_forward = run_walk_forward(prices, dividends, targets, initial_cash, costs)
    retrospective_excess = report_periods["retrospective_test"]["excess_vs_etf_total_return"]
    all_years_hit = any(item["meets_20pct_excess"] for item in year_rows.values())
    gate_pass = retrospective_excess > 0 and all_years_hit
    strategy_gate = {
        "status": "PASS" if gate_pass else "FAIL",
        "reason": (
            "回顾测试超额为正且至少一个自然年度达到20%超额。"
            if gate_pass
            else "受污染回顾区间超额为负，且没有自然年度达到20%超额；不能宣称基础Edge成立。"
        ),
    }
    report = {
        "execution_status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "model_id": model["model_id"],
        "model_governance": model["governance"],
        "costs": costs.__dict__,
        "objective": settings["objective"],
        "strategy_gate": strategy_gate,
        "anchored_walk_forward": walk_forward,
        "periods": report_periods,
    }
    (OUTPUT_DIR / "phase1_five_year_backtest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "phase1_five_year_backtest.md").write_text(render_markdown(report), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(11, 6))
    strategy_plot = full_strategy_ledger.copy()
    strategy_plot["normalized"] = strategy_plot["equity"] / initial_cash
    axis.plot(strategy_plot["date"], strategy_plot["normalized"], label="估值滞回策略（成本后）", linewidth=1.8)
    axis.plot(full_benchmark_series["date"], full_benchmark_series["equity"], label="510300分红再投资", linewidth=1.8)
    axis.set_title("510300 第一阶段五年回测")
    axis.set_ylabel("累计净值")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "phase1_five_year_equity_curve.png", dpi=160)
    plt.close(figure)
    print(f"回测报告：{OUTPUT_DIR / 'phase1_five_year_backtest.md'}")
    print(f"净值图：{OUTPUT_DIR / 'phase1_five_year_equity_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
