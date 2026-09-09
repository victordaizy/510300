"""运行并固化只交易510300的估值-短波动-FVG完整策略。"""

from __future__ import annotations

import hashlib
import json
import sys
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
    audit_fvg_funnel,
    attach_market_cap_context,
    build_short_wave,
    build_valuation_signals,
    costs_with_slippage,
    detect_fvg_zones,
    run_integrated_backtest,
    summarize_strategy,
)

CONFIG_FILE = ROOT / "config" / "510300_valuation_fvg_strategy.yaml"
DAILY_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
MINUTE_FILE = ROOT / "data" / "raw" / "market" / "510300_1m_tushare_raw.parquet"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
CONSTITUENT_PANEL_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
BENCHMARK_FILE = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
REPORT_DIR = ROOT / "reports" / "backtest"
OUTPUT_DIR = ROOT / "data" / "processed" / "510300_valuation_fvg"
PAPER_DIR = ROOT / "paper"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def _benchmark_series(
    benchmark: pd.DataFrame, dates: pd.Series, initial_cash: float
) -> pd.DataFrame:
    result = pd.DataFrame({"date": pd.to_datetime(dates)})
    source = benchmark[["date", "close"]].copy()
    source["date"] = pd.to_datetime(source["date"])
    result = result.merge(source.rename(columns={"close": "benchmark_close"}), on="date", how="left")
    if result["benchmark_close"].isna().any():
        raise ValueError("基准在回测交易日存在缺口")
    result["benchmark_equity"] = (
        initial_cash * result["benchmark_close"] / result["benchmark_close"].iloc[0]
    )
    return result


def _series_summary(equity: pd.Series, dates: pd.Series) -> dict[str, float]:
    elapsed_days = max((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days, 1)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total,
        "cagr": float((1.0 + total) ** (365.25 / elapsed_days) - 1.0),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(242)),
        "max_drawdown": float(drawdown.min()),
    }


def _forward_fair_values(
    latest: pd.Series,
    valuation_config: dict[str, Any],
    horizon_fraction: float,
) -> dict[str, float]:
    eps_base = float(latest["implied_eps"])
    bps_base = float(latest["implied_bps"])
    eps_growth = float(latest["eps_growth_12m"])
    bps_growth = float(latest["bps_growth_12m"])
    projected_eps = eps_base * (1.0 + eps_growth) ** horizon_fraction
    projected_bps = bps_base * (1.0 + bps_growth) ** horizon_fraction
    earnings_weight = float(valuation_config["earnings_anchor_weight"])
    book_weight = float(valuation_config["book_anchor_weight"])
    return {
        label: (
            earnings_weight * projected_eps * float(latest[f"pe_{label}"])
            + book_weight * projected_bps * float(latest[f"pb_{label}"])
        )
        for label in ("low", "mid", "high")
    }


def _current_signal(
    features: pd.DataFrame,
    latest_etf_close: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    latest = features.dropna(subset=["target_position"]).iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    december = pd.Timestamp(year=latest_date.year, month=12, day=31)
    if december <= latest_date:
        december = pd.Timestamp(year=latest_date.year + 1, month=12, day=31)
    december_fraction = (december - latest_date).days / 365.25
    december_values = _forward_fair_values(latest, config["valuation"], december_fraction)
    twelve_month_values = _forward_fair_values(latest, config["valuation"], 1.0)
    etf_scale = latest_etf_close / float(latest["index_close"])
    current_market_cap = float(latest["aggregate_market_cap_cny"])
    market_cap_scale = current_market_cap / float(latest["index_close"])
    direction = str(latest["t_direction"])
    do_t_reason = (
        "短波动未越过阈值；不寻找日内FVG"
        if direction == "NONE"
        else f"已确定{direction}方向，但仍须等待盘中合格FVG，当前不能预先成交"
    )
    return {
        "signal_date": str(latest_date.date()),
        "effective_time": "下一交易日开盘",
        "instrument": "510300.SH",
        "target_position": float(latest["target_position"]),
        "valuation_state": str(latest["valuation_state"]),
        "index_close": float(latest["index_close"]),
        "etf_close": latest_etf_close,
        "wave_z": float(latest["wave_z"]),
        "t_direction": direction,
        "do_t_now": False,
        "do_t_now_reason": do_t_reason,
        "current_aggregate_market_cap_cny": current_market_cap,
        "aggregate_member_count": int(latest["member_count"]),
        "december_target_date": str(december.date()),
        "december_index_fair_value": december_values,
        "december_market_cap_fair_value_cny": {
            key: value * market_cap_scale for key, value in december_values.items()
        },
        "december_etf_price_proxy": {
            key: value * etf_scale for key, value in december_values.items()
        },
        "twelve_month_index_fair_value": twelve_month_values,
        "twelve_month_market_cap_fair_value_cny": {
            key: value * market_cap_scale for key, value in twelve_month_values.items()
        },
        "twelve_month_etf_price_proxy": {
            key: value * etf_scale for key, value in twelve_month_values.items()
        },
        "etf_proxy_warning": "按最新ETF/指数价格比例换算，仅用于仓位决策参考，不是ETF净值承诺",
    }


def _year_table(
    strategy: pd.DataFrame,
    valuation_only: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    combined = strategy[["date", "equity"]].rename(columns={"equity": "strategy"})
    combined = combined.merge(
        valuation_only[["date", "equity"]].rename(columns={"equity": "valuation_only"}),
        on="date",
    ).merge(benchmark[["date", "benchmark_equity"]], on="date")
    combined["year"] = combined["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, frame in combined.groupby("year"):
        if len(frame) < 2:
            continue
        row = {"year": int(year), "observations": int(len(frame))}
        for column in ("strategy", "valuation_only", "benchmark_equity"):
            row[f"{column}_return"] = float(frame[column].iloc[-1] / frame[column].iloc[0] - 1.0)
        row["strategy_excess"] = row["strategy_return"] - row["benchmark_equity_return"]
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(
    base: pd.DataFrame,
    valuation_only: pd.DataFrame,
    stress: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_file: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(base["date"], base["equity"] / base["equity"].iloc[0], label="估值+FVG（5bp）")
    axes[0].plot(
        valuation_only["date"],
        valuation_only["equity"] / valuation_only["equity"].iloc[0],
        label="仅估值仓位",
    )
    axes[0].plot(stress["date"], stress["equity"] / stress["equity"].iloc[0], label="估值+FVG（15bp）")
    axes[0].plot(
        benchmark["date"],
        benchmark["benchmark_equity"] / benchmark["benchmark_equity"].iloc[0],
        label="沪深300全收益",
    )
    axes[0].set_ylabel("归一化净值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].fill_between(base["date"], base["drawdown"], 0.0, alpha=0.35, label="策略回撤")
    axes[1].set_ylabel("回撤")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_report(
    config: dict[str, Any],
    summaries: dict[str, Any],
    current: dict[str, Any],
    year_table: pd.DataFrame,
    t_trades: pd.DataFrame,
    fvg_funnel: dict[str, Any],
    data_hashes: dict[str, str],
) -> str:
    base = summaries["base"]
    valuation_only = summaries["valuation_only"]
    stress = summaries["stress"]
    benchmark = summaries["benchmark"]
    twelve = current["twelve_month_index_fair_value"]
    twelve_etf = current["twelve_month_etf_price_proxy"]
    twelve_cap = current["twelve_month_market_cap_fair_value_cny"]
    december = current["december_index_fair_value"]
    lines = [
        "# 510300估值—短波动—5分钟FVG完整策略",
        "",
        "> 结论：策略交易范围只有510300，不含成分股、期权、期货、融资或杠杆。历史结果没有达到年化超额20%，因此状态是研究完成、等待真正前向验证，而不是收益达标。",
        "",
        "## 当前决策",
        "",
        f"- 信号日：{current['signal_date']}；下一交易日开盘目标为**{current['valuation_state']}（{current['target_position']:.0%}）**。",
        f"- 沪深300现值：{current['index_close']:.2f}；12个月估值带：低 {twelve['low']:.2f} / 中 {twelve['mid']:.2f} / 高 {twelve['high']:.2f}。",
        f"- 信号日300只点时成员总市值：{current['current_aggregate_market_cap_cny'] / 1e12:.2f}万亿元；对应12个月合理总市值：低 {twelve_cap['low'] / 1e12:.2f} / 中 {twelve_cap['mid'] / 1e12:.2f} / 高 {twelve_cap['high'] / 1e12:.2f}万亿元。",
        f"- 按最新价格比例换算的510300代理区间：低 {twelve_etf['low']:.3f} / 中 {twelve_etf['mid']:.3f} / 高 {twelve_etf['high']:.3f}；当前收盘 {current['etf_close']:.3f}。",
        f"- 到{current['december_target_date']}的指数代理估值：低 {december['low']:.2f} / 中 {december['mid']:.2f} / 高 {december['high']:.2f}。",
        f"- 5日短波动z值：{current['wave_z']:.3f}；方向：{current['t_direction']}。当前不做T，因为短波动没有越过±{config['short_wave']['z_threshold']}。",
        "",
        "## 三层规则",
        "",
        "1. **看长：决定隔夜仓位。** 汇总信号日300只点时成员总市值，用总市值/PE反推整体盈利、总市值/PB反推整体净资产；以过去252日年度增速中位数外推12个月，以过去1210日PE/PB的30%、50%、70%分位构造低/中/高合理总市值和指数估值。现值低于低估值满仓，高于高估值空仓，中间半仓。收盘计算，下一交易日开盘执行。",
        "2. **做短：决定半仓日做T方向。** 前一收盘5日对数涨跌除以20日实现波动率×√5；z≥0.75只允许先卖后买，z≤-0.75只允许先买后卖，区间内不做T。",
        "3. **执行：5分钟FVG只找位置。** 三根K线形成标准FVG，等待中点被触及，下一根5分钟K线开盘成交；目标为FVG近端边界，止损为另一边界，最多8根K线，不跨午休、不隔夜、每天最多一次。目标空间不足完整往返成本2倍时放弃。",
        "",
        "## T+1与仓位约束",
        "",
        "- 只有目标半仓且开盘调仓后核心份额全部为隔夜旧仓时，才允许做T。",
        "- 先买后卖：买入的是新份额，随后卖出的必须是旧底仓；先卖后买：先卖旧底仓，再买回同量新份额。",
        "- 日终份额必须回到半仓核心份额。空仓不能做T；由空仓当天新建的半仓也不能做T。",
        "",
        "## FVG筛选漏斗",
        "",
        "> 重要口径：候选FVG、有效回补触发与完成经济门槛的实际往返不是同一个数量。此前把最后一项简称为“合格FVG”属于错误表述，现已修正。",
        "",
        "| 层级 | 事件数 | 覆盖交易日 | 含义 |",
        "|---|---:|---:|---|",
        f"| 原始标准FVG | {fvg_funnel['raw_fvg_events']} | {fvg_funnel['raw_fvg_days']} | 全部5分钟三柱不平衡 |",
        f"| 半仓、短波动有方向、满足T+1 | — | {fvg_funnel['t_plus_one_eligible_days']} | 当日具备做T账户条件 |",
        f"| 与短波动方向匹配 | {fvg_funnel['direction_matched_events']} | {fvg_funnel['direction_matched_days']} | FVG方向候选 |",
        f"| 12根K线内回补中点 | {fvg_funnel['midpoint_retracement_events']} | {fvg_funnel['midpoint_retracement_days']} | 有效FVG触发 |",
        f"| 回补后下一根开盘仍在缺口内 | {fvg_funnel['next_open_inside_events']} | {fvg_funnel['next_open_inside_days']} | 原实现的额外执行限制 |",
        f"| FVG自身宽度覆盖2倍成本 | {fvg_funnel['economic_gate_events']} | {fvg_funnel['economic_gate_days']} | 原实现的经济门槛，不代表FVG是否合格 |",
        "",
        "诊断显示，FVG自身宽度不应被当成整笔T的止盈空间；FVG只负责触发，退出应由独立的短周期目标定义。该退出层尚无可靠净边际，因此实际T往返继续单列，不再用它代表FVG数量。",
        "",
        "## 五年历史回顾",
        "",
        "| 情景 | CAGR | 相对全收益年化差 | 最大回撤 | 平均仓位 | 实际T往返 | T净损益 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 基础5bp | {_format_pct(base['cagr'])} | {_format_pct(base['cagr'] - benchmark['cagr'])} | {_format_pct(base['max_drawdown'])} | {_format_pct(base['average_exposure'])} | {base['t_roundtrip_count']} | {base['t_net_pnl_cny']:.2f}元 |",
        f"| 仅估值仓位 | {_format_pct(valuation_only['cagr'])} | {_format_pct(valuation_only['cagr'] - benchmark['cagr'])} | {_format_pct(valuation_only['max_drawdown'])} | {_format_pct(valuation_only['average_exposure'])} | 0 | 0.00元 |",
        f"| 压力15bp | {_format_pct(stress['cagr'])} | {_format_pct(stress['cagr'] - benchmark['cagr'])} | {_format_pct(stress['max_drawdown'])} | {_format_pct(stress['average_exposure'])} | {stress['t_roundtrip_count']} | {stress['t_net_pnl_cny']:.2f}元 |",
        f"| 沪深300全收益 | {_format_pct(benchmark['cagr'])} | 0.00% | {_format_pct(benchmark['max_drawdown'])} | 100.00% | — | — |",
        "",
        f"基础成本下实际完成全部旧退出规则的T往返只有{len(t_trades)}笔；这不是合格FVG数量。有效中点回补触发覆盖{fvg_funnel['midpoint_retracement_days']}天。直接将这些触发全部交易，在日收盘、反向FVG或波动止盈三种预先检查的退出方式下均无法覆盖成本，因此没有用放宽门槛伪造收益。",
        "",
        "## 为什么不能承诺年化超额20%",
        "",
        "该约束下组合最多100%持有同一只ETF，没有杠杆，也不做空。在强上涨年份，空仓或半仓无法稳定比满仓标的多赚20个百分点；唯一额外来源是日内T，但现有FVG在真实最低佣金下没有足够证据。把目标写成承诺会篡改研究结论。",
        "",
        "## 自然年度审计",
        "",
        "| 年份 | 策略 | 仅估值 | 全收益基准 | 策略超额 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in year_table.itertuples(index=False):
        lines.append(
            f"| {row.year} | {_format_pct(row.strategy_return)} | {_format_pct(row.valuation_only_return)} | "
            f"{_format_pct(row.benchmark_equity_return)} | {_format_pct(row.strategy_excess)} |"
        )
    lines.extend(
        [
            "",
            "## 估值口径限制",
            "",
            "当前可回测数据包含每日300只点时成员总市值与沪深300整体PE/PB，但不含每家公司的历史公告时点自由现金流，因此本版是指数聚合盈利/净资产估值代理，不是逐家公司DCF。总市值未按指数自由流通权重重新加权，只作为一致口径的市值尺度。以后补齐点时财报数据，可以替换估值层，但不得把两种模型混称。ETF价格区间按最新ETF/指数价格比换算，也不等同于基金净值保证。",
            "",
            "## 治理状态",
            "",
            "全部历史已参与规则研究，没有严格样本外。规则从冻结后的下一交易日起只做纸面前向记录；至少积累30笔合格FVG往返前不重新调参，也不授权自动下单。",
            "",
            "## 数据指纹",
            "",
        ]
    )
    lines.extend(f"- `{name}`：`{digest}`" for name, digest in data_hashes.items())
    return "\n".join(lines) + "\n"


def main() -> None:
    config = _load_config()
    daily = pd.read_parquet(DAILY_FILE)
    minute = pd.read_parquet(MINUTE_FILE)
    valuation = pd.read_parquet(VALUATION_FILE)
    constituent_panel = pd.read_parquet(CONSTITUENT_PANEL_FILE)
    benchmark_raw = pd.read_parquet(BENCHMARK_FILE)
    dividends = pd.read_csv(DIVIDEND_FILE)

    valuation_features = attach_market_cap_context(
        build_valuation_signals(valuation, config["valuation"]), constituent_panel
    )
    wave_features = build_short_wave(daily, config["short_wave"])
    signals = valuation_features.merge(
        wave_features[["date", "wave_z", "t_direction"]], on="date", how="inner", validate="one_to_one"
    )
    bars = aggregate_minute_to_5m(minute, int(config["fvg"]["source_records_per_bar"]))
    zones = detect_fvg_zones(bars, float(config["fvg"]["minimum_gap_bps"]))
    costs = _costs(config)
    initial_cash = float(config["backtest"]["initial_cash_cny"])
    start = config["backtest"]["start_date"]
    end = config["backtest"]["end_date"]

    base_ledger, base_core, base_t = run_integrated_backtest(
        daily, dividends, signals, bars, zones, initial_cash, costs, config["fvg"], start, end, True
    )
    valuation_ledger, valuation_core, valuation_t = run_integrated_backtest(
        daily, dividends, signals, bars, zones, initial_cash, costs, config["fvg"], start, end, False
    )
    stress_costs = costs_with_slippage(costs, float(config["costs"]["stress_slippage_bps_per_leg"]))
    stress_ledger, stress_core, stress_t = run_integrated_backtest(
        daily, dividends, signals, bars, zones, initial_cash, stress_costs, config["fvg"], start, end, True
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    benchmark_summary = _series_summary(benchmark["benchmark_equity"], benchmark["date"])
    summaries = {
        "base": summarize_strategy(base_ledger, base_core, base_t, initial_cash),
        "valuation_only": summarize_strategy(valuation_ledger, valuation_core, valuation_t, initial_cash),
        "stress": summarize_strategy(stress_ledger, stress_core, stress_t, initial_cash),
        "benchmark": benchmark_summary,
    }
    fvg_funnel = audit_fvg_funnel(
        valuation_ledger, bars, zones, costs, config["fvg"]
    )
    current = _current_signal(signals, float(daily.sort_values("date").iloc[-1]["close"]), config)
    year_table = _year_table(base_ledger, valuation_ledger, benchmark)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    base_ledger.to_parquet(OUTPUT_DIR / "daily_ledger.parquet", index=False)
    base_core.to_csv(OUTPUT_DIR / "core_trades.csv", index=False, encoding="utf-8-sig")
    base_t.to_csv(OUTPUT_DIR / "fvg_t_trades.csv", index=False, encoding="utf-8-sig")
    signals.to_parquet(OUTPUT_DIR / "valuation_and_wave_signals.parquet", index=False)
    year_table.to_csv(OUTPUT_DIR / "yearly_returns.csv", index=False, encoding="utf-8-sig")

    data_hashes = {
        DAILY_FILE.name: _sha256(DAILY_FILE),
        MINUTE_FILE.name: _sha256(MINUTE_FILE),
        VALUATION_FILE.name: _sha256(VALUATION_FILE),
        CONSTITUENT_PANEL_FILE.name: _sha256(CONSTITUENT_PANEL_FILE),
        BENCHMARK_FILE.name: _sha256(BENCHMARK_FILE),
        DIVIDEND_FILE.name: _sha256(DIVIDEND_FILE),
        CONFIG_FILE.name: _sha256(CONFIG_FILE),
    }
    payload = {
        "strategy": config["strategy"],
        "summaries": summaries,
        "current_signal": current,
        "fvg_funnel": fvg_funnel,
        "actual_t_roundtrip_sample_size": int(len(base_t)),
        "data_hashes": data_hashes,
    }
    (REPORT_DIR / "510300_valuation_fvg_strategy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PAPER_DIR / "510300_latest_signal.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "fvg_funnel.json").write_text(
        json.dumps(fvg_funnel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _build_report(
        config, summaries, current, year_table, base_t, fvg_funnel, data_hashes
    )
    (REPORT_DIR / "510300_valuation_fvg_strategy.md").write_text(report, encoding="utf-8")
    _plot(
        base_ledger,
        valuation_ledger,
        stress_ledger,
        benchmark,
        REPORT_DIR / "510300_valuation_fvg_equity_curve.png",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from backtest.run_510300_continuous_inventory_strategy import main as continuous_main

    continuous_main()
