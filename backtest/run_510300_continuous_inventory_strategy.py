"""运行只交易510300的连续估值库存、短波战术偏移与FVG单边执行策略。"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.valuation_fvg_engine import (
    StrategyCosts,
    aggregate_minute_to_5m,
    attach_market_cap_context,
    build_short_wave,
    build_tactical_positions,
    build_valuation_signals,
    costs_with_slippage,
    detect_fvg_zones,
    run_inventory_backtest,
    schedule_continuous_positions,
    summarize_inventory_backtest,
)

CONFIG_FILE = ROOT / "config" / "510300_valuation_fvg_strategy.yaml"
DAILY_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
MINUTE_FILE = ROOT / "data" / "raw" / "market" / "510300_1m_tushare_raw.parquet"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
CONSTITUENT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
BENCHMARK_FILE = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
REPORT_DIR = ROOT / "reports" / "backtest"
OUTPUT_DIR = ROOT / "data" / "processed" / "510300_continuous_inventory"
PAPER_DIR = ROOT / "paper"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _costs(config: dict[str, Any]) -> StrategyCosts:
    cost = config["costs"]
    account = config["account"]
    return StrategyCosts(
        commission_rate=float(cost["commission_rate"]),
        minimum_commission_cny=float(cost["minimum_commission_cny"]),
        stamp_duty_rate=float(cost["stamp_duty_rate"]),
        slippage_bps_per_leg=float(cost["base_slippage_bps_per_leg"]),
        price_tick_cny=float(cost["price_tick_cny"]),
        lot_size=int(account["lot_size"]),
        cash_annual_rate=float(account["cash_annual_rate"]),
    )


def _benchmark(
    raw: pd.DataFrame, dates: pd.Series, initial_cash: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    source = raw[["date", "close"]].copy()
    source["date"] = pd.to_datetime(source["date"])
    frame = frame.merge(source.rename(columns={"close": "benchmark_close"}), on="date")
    frame["equity"] = initial_cash * frame["benchmark_close"] / frame["benchmark_close"].iloc[0]
    elapsed = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days, 1)
    total = float(frame["equity"].iloc[-1] / initial_cash - 1.0)
    returns = frame["equity"].pct_change(fill_method=None).fillna(0.0)
    drawdown = frame["equity"] / frame["equity"].cummax() - 1.0
    summary = {
        "total_return": total,
        "cagr": float((1.0 + total) ** (365.25 / elapsed) - 1.0),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(242)),
        "max_drawdown": float(drawdown.min()),
    }
    return frame, summary


def _forward_values(latest: pd.Series, config: dict[str, Any], fraction: float) -> dict[str, float]:
    eps = float(latest["implied_eps"]) * (1.0 + float(latest["eps_growth_12m"])) ** fraction
    bps = float(latest["implied_bps"]) * (1.0 + float(latest["bps_growth_12m"])) ** fraction
    earnings_weight = float(config["earnings_anchor_weight"])
    book_weight = float(config["book_anchor_weight"])
    return {
        label: (
            earnings_weight * eps * float(latest[f"pe_{label}"])
            + book_weight * bps * float(latest[f"pb_{label}"])
        )
        for label in ("low", "mid", "high")
    }


def _current_signal(
    signals: pd.DataFrame,
    ledger: pd.DataFrame,
    latest_etf_close: float,
    initial_cash: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    latest = signals.dropna(subset=["desired_position"]).iloc[-1]
    ledger_latest = ledger.iloc[-1]
    date = pd.Timestamp(latest["date"])
    twelve = _forward_values(latest, config["valuation"], 1.0)
    december = pd.Timestamp(date.year, 12, 31)
    if december <= date:
        december = pd.Timestamp(date.year + 1, 12, 31)
    december_values = _forward_values(
        latest, config["valuation"], (december - date).days / 365.25
    )
    etf_scale = latest_etf_close / float(latest["index_close"])
    desired_position = float(latest["desired_position"])
    reference_shares = int(
        np.floor(desired_position * initial_cash / latest_etf_close / 100.0)
    ) * 100
    simulated_shares = int(ledger_latest["shares"])
    share_gap = reference_shares - simulated_shares
    if abs(share_gap) * latest_etf_close < float(config["fvg"]["minimum_trade_notional_cny"]):
        next_action = "库存差额未达到经济成交额，继续持有"
    elif share_gap > 0:
        next_action = f"等待看涨FVG中点回补，计划买入约{share_gap}份"
    else:
        next_action = f"等待看跌FVG中点回补，计划卖出约{abs(share_gap)}份旧库存"
    return {
        "signal_date": str(date.date()),
        "effective_from": "下一交易日；没有FVG则延续等待，不追价",
        "instrument": "510300.SH",
        "index_close": float(latest["index_close"]),
        "etf_close": latest_etf_close,
        "aggregate_market_cap_cny": float(latest["aggregate_market_cap_cny"]),
        "raw_continuous_valuation_position": float(latest["raw_continuous_position"]),
        "base_position": float(latest["base_position"]),
        "wave_z": float(latest["wave_z"]),
        "tactical_adjustment": float(latest["tactical_adjustment"]),
        "desired_position": desired_position,
        "paper_actual_position": float(ledger_latest["actual_position"]),
        "paper_shares": simulated_shares,
        "reference_account_cny": initial_cash,
        "reference_target_shares": reference_shares,
        "next_action": next_action,
        "forced_end_of_day_restoration": False,
        "minimum_single_leg_notional_cny": float(config["fvg"]["minimum_trade_notional_cny"]),
        "twelve_month_index_fair_value": twelve,
        "twelve_month_etf_price_proxy": {key: value * etf_scale for key, value in twelve.items()},
        "december_target_date": str(december.date()),
        "december_index_fair_value": december_values,
        "december_etf_price_proxy": {
            key: value * etf_scale for key, value in december_values.items()
        },
    }


def _year_table(
    strategy: pd.DataFrame,
    strategic_only: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    frame = strategy[["date", "equity"]].rename(columns={"equity": "strategy"})
    frame = frame.merge(
        strategic_only[["date", "equity"]].rename(columns={"equity": "strategic_only"}),
        on="date",
    ).merge(benchmark[["date", "equity"]].rename(columns={"equity": "benchmark"}), on="date")
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year"):
        if len(group) < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(group["strategy"].iloc[-1] / group["strategy"].iloc[0] - 1.0),
                "strategic_only_return": float(
                    group["strategic_only"].iloc[-1] / group["strategic_only"].iloc[0] - 1.0
                ),
                "benchmark_return": float(group["benchmark"].iloc[-1] / group["benchmark"].iloc[0] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def _plot(
    strategy: pd.DataFrame,
    strategic_only: pd.DataFrame,
    stress: pd.DataFrame,
    benchmark: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for frame, label in (
        (strategy, "连续库存+短波+FVG（5bp）"),
        (strategic_only, "连续估值库存（无战术偏移）"),
        (stress, "连续库存压力（15bp）"),
    ):
        axes[0].plot(frame["date"], frame["equity"] / frame["equity"].iloc[0], label=label)
    axes[0].plot(
        benchmark["date"], benchmark["equity"] / benchmark["equity"].iloc[0], label="沪深300全收益"
    )
    axes[0].set_ylabel("归一化净值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(strategy["date"], strategy["desired_position"], label="目标库存", alpha=0.8)
    axes[1].plot(strategy["date"], strategy["actual_position"], label="实际库存", alpha=0.8)
    axes[1].set_ylabel("仓位")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _report(
    config: dict[str, Any],
    summaries: dict[str, Any],
    current: dict[str, Any],
    ledger: pd.DataFrame,
    yearly: pd.DataFrame,
    hashes: dict[str, str],
) -> str:
    base = summaries["base"]
    strategic = summaries["strategic_only"]
    stress = summaries["stress"]
    benchmark = summaries["benchmark"]
    status = ledger["execution_status"].value_counts().to_dict()
    lines = [
        "# 510300连续库存—短波—FVG单边执行策略",
        "",
        "> 本版只交易510300。仓位为0%至100%连续值；FVG成交后允许跨日持有，不再强制日终恢复核心份额，也不为了凑一次T而支付第二条腿成本。",
        "",
        "## 当前信号",
        "",
        f"- 信号日：{current['signal_date']}；连续估值基础仓位：**{current['base_position']:.1%}**。",
        f"- 长短波残差z值：{current['wave_z']:.3f}；战术偏移：{current['tactical_adjustment']:+.1%}；最终目标库存：**{current['desired_position']:.1%}**。",
        f"- 2万元参考账户目标约{current['reference_target_shares']}份；纸面历史库存为{current['paper_shares']}份。下一动作：{current['next_action']}。",
        f"- 12个月510300估值代理：低 {current['twelve_month_etf_price_proxy']['low']:.3f} / 中 {current['twelve_month_etf_price_proxy']['mid']:.3f} / 高 {current['twelve_month_etf_price_proxy']['high']:.3f}。",
        "",
        "## 完整规则",
        "",
        "1. 每10个交易日更新一次基础仓位：`clip((合理总市值上沿-当前总市值)/(上沿-下沿), 0, 1)`。它可以是任意0%至100%中间值。",
        "2. 短周期使用5日涨跌减去同期20日趋势的5/20，再除以20日实现波动率×√5。z低于-0.75增加库存，z高于0.75减少库存，最大战术偏移10%。",
        "3. 目标库存高于实际库存时，只等看涨FVG中点回补买入；目标低于实际库存时，只等看跌FVG中点回补卖出。没有FVG就继续持有，不在开盘追价。",
        "4. 默认每日一条库存调整腿；卖出不超过开盘旧库存，满足T+1。若同日后续出现新的反向FVG，且按已知成交价可锁定至少10元净利润，允许第二条腿完成可选日内T；否则库存跨日。",
        "5. 单边成交额至少2000元，且佣金不得超过成交额0.25%；小库存差继续累积，避免最低5元佣金反复侵蚀。",
        "",
        "## FVG与成本漏斗",
        "",
        f"- 原始标准FVG：10,952个。基础成本下，库存模型实际执行{base['single_leg_trade_count']}条成交腿，其中买入{base['buy_leg_count']}条、卖出{base['sell_leg_count']}条。",
        f"- 可选同日T共{base['optional_intraday_roundtrip_count']}次，成交时锁定净利润合计{base['locked_intraday_roundtrip_profit_cny']:.2f}元；其余库存均不强制回补。",
        f"- 因库存差低于经济成交额而等待：{status.get('目标差额成交额低于最低经济成交金额', 0)}天；没有匹配可成交FVG：{status.get('当日没有匹配并可成交的FVG中点回补', 0)}天。",
        "- 这里不再统计“当天往返次数”，因为一买一卖可能跨越数日；强行日内配对会错误增加成本。",
        "",
        "## 五年历史回顾",
        "",
        "| 情景 | CAGR | 相对基准年化差 | 最大回撤 | 平均仓位 | 单边成交 | 显式佣金 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 连续库存+短波+FVG 5bp | {_pct(base['cagr'])} | {_pct(base['cagr'] - benchmark['cagr'])} | {_pct(base['max_drawdown'])} | {_pct(base['average_exposure'])} | {base['single_leg_trade_count']} | {base['explicit_cost_cny']:.2f}元 |",
        f"| 仅连续估值库存 | {_pct(strategic['cagr'])} | {_pct(strategic['cagr'] - benchmark['cagr'])} | {_pct(strategic['max_drawdown'])} | {_pct(strategic['average_exposure'])} | {strategic['single_leg_trade_count']} | {strategic['explicit_cost_cny']:.2f}元 |",
        f"| 压力15bp | {_pct(stress['cagr'])} | {_pct(stress['cagr'] - benchmark['cagr'])} | {_pct(stress['max_drawdown'])} | {_pct(stress['average_exposure'])} | {stress['single_leg_trade_count']} | {stress['explicit_cost_cny']:.2f}元 |",
        f"| 沪深300全收益 | {_pct(benchmark['cagr'])} | 0.00% | {_pct(benchmark['max_drawdown'])} | 100.00% | — | — |",
        "",
        "短波战术层相对仅连续估值库存的历史增量必须单独观察；全部历史已参与规则研究，因此这仍不是严格样本外证明，也没有达到年化超额20%。",
        "",
        "## 自然年度",
        "",
        "| 年份 | 完整策略 | 仅基础库存 | 全收益基准 |",
        "|---:|---:|---:|---:|",
    ]
    for row in yearly.itertuples(index=False):
        lines.append(
            f"| {row.year} | {_pct(row.strategy_return)} | {_pct(row.strategic_only_return)} | {_pct(row.benchmark_return)} |"
        )
    lines.extend(
        [
            "",
            "## 口径与治理",
            "",
            "市值层是300只点时成员总市值与整体PE/PB构成的聚合代理，不是逐家公司DCF。分钟FVG采用保守不利滑点和最低佣金。规则冻结后应纸面前向运行，当前不授权自动下单。",
            "",
            "## 数据指纹",
            "",
        ]
    )
    lines.extend(f"- `{name}`：`{digest}`" for name, digest in hashes.items())
    return "\n".join(lines) + "\n"


def main() -> None:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    daily = pd.read_parquet(DAILY_FILE)
    minute = pd.read_parquet(MINUTE_FILE)
    valuation = pd.read_parquet(VALUATION_FILE)
    constituents = pd.read_parquet(CONSTITUENT_FILE)
    benchmark_raw = pd.read_parquet(BENCHMARK_FILE)
    dividends = pd.read_csv(DIVIDEND_FILE)

    valuation_features = attach_market_cap_context(
        build_valuation_signals(valuation, config["valuation"]), constituents
    )
    wave = build_short_wave(daily, config["short_wave"])
    signals = valuation_features.merge(
        wave[
            [
                "date",
                "wave_log_return",
                "long_wave_log_return",
                "wave_residual",
                "realized_volatility",
                "wave_z",
                "t_direction",
            ]
        ],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    signals = schedule_continuous_positions(
        signals, int(config["valuation"]["rebalance_every_trading_days"])
    )
    signals = build_tactical_positions(signals, config["tactical_inventory"])
    strategic_signals = signals.copy()
    strategic_signals["tactical_adjustment"] = 0.0
    strategic_signals["desired_position"] = strategic_signals["base_position"]

    bars = aggregate_minute_to_5m(minute, int(config["fvg"]["source_records_per_bar"]))
    zones = detect_fvg_zones(bars, float(config["fvg"]["minimum_gap_bps"]))
    costs = _costs(config)
    initial_cash = float(config["backtest"]["initial_cash_cny"])
    start, end = config["backtest"]["start_date"], config["backtest"]["end_date"]
    ledger, trades = run_inventory_backtest(
        daily, dividends, signals, bars, zones, initial_cash, costs, config["fvg"], start, end
    )
    strategic_ledger, strategic_trades = run_inventory_backtest(
        daily,
        dividends,
        strategic_signals,
        bars,
        zones,
        initial_cash,
        costs,
        config["fvg"],
        start,
        end,
    )
    stress_costs = costs_with_slippage(
        costs, float(config["costs"]["stress_slippage_bps_per_leg"])
    )
    stress_ledger, stress_trades = run_inventory_backtest(
        daily,
        dividends,
        signals,
        bars,
        zones,
        initial_cash,
        stress_costs,
        config["fvg"],
        start,
        end,
    )
    strategic_stress_ledger, strategic_stress_trades = run_inventory_backtest(
        daily,
        dividends,
        strategic_signals,
        bars,
        zones,
        initial_cash,
        stress_costs,
        config["fvg"],
        start,
        end,
    )
    benchmark, benchmark_summary = _benchmark(benchmark_raw, ledger["date"], initial_cash)
    summaries = {
        "base": summarize_inventory_backtest(ledger, trades, initial_cash),
        "strategic_only": summarize_inventory_backtest(
            strategic_ledger, strategic_trades, initial_cash
        ),
        "strategic_only_stress": summarize_inventory_backtest(
            strategic_stress_ledger, strategic_stress_trades, initial_cash
        ),
        "stress": summarize_inventory_backtest(stress_ledger, stress_trades, initial_cash),
        "benchmark": benchmark_summary,
    }
    current = _current_signal(
        signals, ledger, float(daily.sort_values("date").iloc[-1]["close"]), initial_cash, config
    )
    yearly = _year_table(ledger, strategic_ledger, benchmark)
    hashes = {
        path.name: _sha256(path)
        for path in (
            DAILY_FILE,
            MINUTE_FILE,
            VALUATION_FILE,
            CONSTITUENT_FILE,
            BENCHMARK_FILE,
            DIVIDEND_FILE,
            CONFIG_FILE,
        )
    }
    payload = {
        "strategy": config["strategy"],
        "summaries": summaries,
        "current_signal": current,
        "execution_status_counts": ledger["execution_status"].value_counts().to_dict(),
        "raw_fvg_events": int(len(zones)),
        "raw_fvg_days": int(zones["date"].nunique()),
        "data_hashes": hashes,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(OUTPUT_DIR / "daily_inventory_ledger.parquet", index=False)
    stress_ledger.to_parquet(OUTPUT_DIR / "stress_inventory_ledger.parquet", index=False)
    strategic_ledger.to_parquet(OUTPUT_DIR / "strategic_only_ledger.parquet", index=False)
    strategic_stress_ledger.to_parquet(
        OUTPUT_DIR / "strategic_only_stress_ledger.parquet", index=False
    )
    trades.to_csv(OUTPUT_DIR / "single_leg_trades.csv", index=False, encoding="utf-8-sig")
    signals.to_parquet(OUTPUT_DIR / "continuous_signals.parquet", index=False)
    yearly.to_csv(OUTPUT_DIR / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "510300_continuous_inventory_strategy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PAPER_DIR / "510300_latest_signal.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "510300_continuous_inventory_strategy.md").write_text(
        _report(config, summaries, current, ledger, yearly, hashes), encoding="utf-8"
    )
    _plot(
        ledger,
        strategic_ledger,
        stress_ledger,
        benchmark,
        REPORT_DIR / "510300_continuous_inventory_equity_curve.png",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
