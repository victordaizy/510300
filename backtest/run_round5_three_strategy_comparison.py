"""比较波段持仓、固定底仓频繁做T，以及估值库存与做T组合。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtest.run_round5_defensive_valuation_timing as round5
from backtest.engine import BacktestCosts, run_long_cash_backtest
from backtest.valuation_anchor_t_engine import (
    AnchorTResult,
    build_fixed_anchor_features,
    run_anchor_t_overlay,
    summarize_anchor_t,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "round5_valuation_anchor_t_overlay.yaml"
REPORT_JSON = ROOT / "reports" / "backtest" / "round5_three_strategy_comparison.json"
REPORT_MD = ROOT / "reports" / "backtest" / "round5_three_strategy_comparison.md"
REPORT_PNG = ROOT / "reports" / "backtest" / "round5_three_strategy_comparison.png"
OUTPUT_DIR = ROOT / "data" / "processed" / "round5_three_strategy_comparison"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backtest_costs(config: dict[str, Any], slippage_bps: float) -> BacktestCosts:
    costs = config["costs"]
    return BacktestCosts(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_rate=float(costs["stamp_duty_rate"]),
        slippage_bps=float(slippage_bps),
        lot_size=int(config["account"]["lot_size"]),
        cash_annual_rate=0.015,
    )


def _static_half_targets(calendar: pd.Series) -> pd.DataFrame:
    targets = pd.DataFrame({"date": pd.to_datetime(calendar), "target_position": 0.5})
    targets["trade_allowed"] = False
    targets.loc[targets.index[0], "trade_allowed"] = True
    targets["risk_off_override"] = False
    targets["signal_reason"] = "固定50%底仓首次建仓"
    return targets


def _metric_frame(
    name: str,
    scenario: str,
    ledger: pd.DataFrame,
    initial_cash: float,
    average_exposure_column: str,
    t_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = ledger.copy()
    if "daily_return" not in frame:
        frame["daily_return"] = frame["equity"].pct_change(fill_method=None).fillna(0.0)
    if "drawdown" not in frame:
        frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    elapsed_days = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days, 1)
    total_return = float(frame["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    volatility = float(frame["daily_return"].std(ddof=1) * np.sqrt(242))
    return {
        "strategy_name": name,
        "scenario": scenario,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": (
            float(frame["daily_return"].mean() * 242 / volatility) if volatility > 0 else None
        ),
        "max_drawdown": float(frame["drawdown"].min()),
        "average_exposure": float(frame[average_exposure_column].mean()),
        "ending_equity": float(frame["equity"].iloc[-1]),
        "intraday_t": t_metrics
        or {
            "round_trip_count": 0,
            "raw_gross_pnl_cny": 0.0,
            "net_pnl_cny": 0.0,
            "commission_cny": 0.0,
            "slippage_and_tick_cost_cny": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        },
    }


def _attach_benchmark_metrics(
    summary: dict[str, Any],
    ledger: pd.DataFrame,
    h00300: pd.DataFrame,
    etf_buy_hold: pd.DataFrame,
) -> dict[str, Any]:
    result = dict(summary)
    h_summary = round5._index_summary(h00300, float(h00300["equity"].iloc[0]))
    etf_cagr = _metric_frame(
        "510300买入持有", "BASE_COST", etf_buy_hold, float(etf_buy_hold["equity"].iloc[0]), "actual_position"
    )["cagr"]
    result["annualized_excess_vs_h00300"] = result["cagr"] - h_summary["cagr"]
    result["annualized_excess_vs_etf_buy_hold"] = result["cagr"] - etf_cagr
    result["rolling_242d_excess_vs_h00300"] = round5._rolling_excess(ledger, h00300)
    result["rolling_242d_excess_vs_etf_buy_hold"] = round5._rolling_excess(
        ledger, etf_buy_hold
    )
    return result


def _overlay_summary(result: AnchorTResult, initial_cash: float) -> dict[str, Any]:
    raw = summarize_anchor_t(result, initial_cash)
    raw["average_exposure"] = raw.pop("average_base_exposure")
    return raw


def _gate(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    excess = float(summary["annualized_excess_vs_h00300"])
    rolling = float(summary["rolling_242d_excess_vs_h00300"]["median"])
    minimum = float(evaluation["annualized_excess_minimum"])
    challenge = float(evaluation["annualized_excess_challenge"])
    rolling_minimum = float(evaluation["rolling_242d_excess_median_minimum"])
    rolling_challenge = float(evaluation["rolling_242d_excess_median_challenge"])
    return {
        "minimum_status": (
            "PASS_RETROSPECTIVE"
            if excess >= minimum and rolling >= rolling_minimum
            else "FAIL_RETROSPECTIVE"
        ),
        "challenge_status": (
            "PASS_RETROSPECTIVE"
            if excess >= challenge and rolling >= rolling_challenge
            else "FAIL_RETROSPECTIVE"
        ),
        "annualized_excess": excess,
        "rolling_242d_excess_median": rolling,
        "minimum_requirements": {"annualized_excess": minimum, "rolling_median": rolling_minimum},
        "challenge_requirements": {
            "annualized_excess": challenge,
            "rolling_median": rolling_challenge,
        },
    }


def _annual_analysis(
    ledgers: dict[str, pd.DataFrame],
    t_trades: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    merged: pd.DataFrame | None = None
    for name, ledger in ledgers.items():
        values = ledger[["date", "equity"]].rename(columns={"equity": name})
        merged = values if merged is None else merged.merge(values, on="date", validate="one_to_one")
    if merged is None:
        return []
    merged["year"] = merged["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in merged.groupby("year", sort=True):
        row: dict[str, Any] = {"year": int(year), "observations": int(len(group))}
        for name in ledgers:
            row[f"{name}_return"] = float(group[name].iloc[-1] / group[name].iloc[0] - 1.0)
        for name, trades in t_trades.items():
            if trades.empty:
                year_trades = trades
            else:
                year_trades = trades.loc[pd.to_datetime(trades["trade_date"]).dt.year.eq(year)]
            row[f"{name}_t_round_trips"] = int(len(year_trades))
            row[f"{name}_t_net_pnl_cny"] = (
                float(year_trades["net_pnl_cny"].sum()) if not year_trades.empty else 0.0
            )
        rows.append(row)
    return rows


def _t_stability(trades: pd.DataFrame, annual: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    complete = [item for item in annual if item["observations"] >= 230]
    counts = [item[f"{prefix}_t_round_trips"] for item in complete]
    profitable = [item[f"{prefix}_t_net_pnl_cny"] > 0.0 for item in complete]
    return {
        "complete_years": [item["year"] for item in complete],
        "minimum_round_trips": min(counts) if counts else 0,
        "maximum_round_trips": max(counts) if counts else 0,
        "profitable_complete_year_ratio": float(np.mean(profitable)) if profitable else 0.0,
        "total_round_trips": int(len(trades)),
    }


def _render(payload: dict[str, Any]) -> str:
    s = payload["summaries"]
    pct = lambda value: f"{value:.2%}"
    rows = []
    for key, label in (
        ("h00300", "H00300全收益"),
        ("etf_buy_hold", "510300买入持有"),
        ("swing", "波段持仓"),
        ("t_only", "固定50%底仓频繁T"),
        ("combined", "估值库存+频繁T"),
        ("t_only_stress", "固定底仓T（15bp）"),
        ("combined_stress", "组合策略（15bp）"),
    ):
        item = s[key]
        rows.append(
            f"| {label} | {pct(item['cagr'])} | {pct(item.get('annualized_excess_vs_h00300', 0.0))} | "
            f"{pct(item['max_drawdown'])} | {item['sharpe_zero_cash_rate']:.2f} | "
            f"{pct(item.get('average_exposure', 1.0))} | {item['intraday_t']['round_trip_count']} | "
            f"{item['intraday_t']['net_pnl_cny']:.2f} |"
        )
    lines = [
        "# 第五轮三策略比较：波段、频繁做T与组合",
        "",
        "> 所有方案使用同一2万元起点、同一510300分钟行情和成交成本。V2固定锚规则在查看结果前冻结，没有网格搜索。全部五年历史已被研究，结论仍是回顾性证据。",
        "",
        "## 核心结果",
        "",
        "| 方案 | CAGR | 对H00300年化超额 | 最大回撤 | Sharpe | 平均隔夜仓位 | T回合 | T净收益（元） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 直接结论",
        "",
        f"- 波段持仓相对H00300年化超额：**{pct(s['swing']['annualized_excess_vs_h00300'])}**。",
        f"- 固定底仓做T相对H00300年化超额：**{pct(s['t_only']['annualized_excess_vs_h00300'])}**；其中纯T累计净收益 **{s['t_only']['intraday_t']['net_pnl_cny']:.2f}元**。",
        f"- 组合策略相对H00300年化超额：**{pct(s['combined']['annualized_excess_vs_h00300'])}**；估值库存之上的纯T累计净收益 **{s['combined']['intraday_t']['net_pnl_cny']:.2f}元**。",
        f"- 组合策略滚动242日超额中位数：**{pct(s['combined']['rolling_242d_excess_vs_h00300']['median'])}**。最低10%门槛：**{payload['gates']['combined']['minimum_status']}**；15%挑战线：**{payload['gates']['combined']['challenge_status']}**。",
        f"- 组合策略每回合实际毛收益/成本/净收益均值：**{payload['t_economics']['observed_gross_per_round_cny']:.2f} / {payload['t_economics']['observed_cost_per_round_cny']:.2f} / {payload['t_economics']['observed_net_per_round_cny']:.2f}元**。",
        f"- 若交易次数不变，要让组合达到10%年化超额，平均每回合至少需要约 **{payload['t_economics']['required_net_per_round_for_10pct_excess_cny']:.2f}元净收益**；达到15%约需 **{payload['t_economics']['required_net_per_round_for_15pct_excess_cny']:.2f}元**。",
        "",
        "## 做T规则",
        "",
        "1. 前一交易日未复权收盘价是全天固定锚，避免累计VWAP在趋势日持续移动。",
        "2. 偏离固定锚达到双倍往返成本，且连续三条记录与最新一分钟同时转向锚时，下一分钟开盘成交。",
        "3. 回到固定锚、止损、持有满60条记录或14:49触发退出，仍在下一分钟开盘成交。",
        "4. 每天最多一个回合；买低后只能卖等量旧仓，卖高后买回相同份额，日终核心库存不变。",
        "5. 固定底仓方案使用50%隔夜仓位；组合方案由估值—趋势—风险模型决定隔夜仓位。核心调仓日不做T。",
        "",
        "## 自然年度",
        "",
        "| 年份 | 波段 | 固定底仓T | 组合 | H00300 | 固定底仓T净收益 | 组合T净收益 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["annual"]:
        lines.append(
            f"| {item['year']} | {pct(item['swing_return'])} | {pct(item['t_only_return'])} | "
            f"{pct(item['combined_return'])} | {pct(item['h00300_return'])} | "
            f"{item['t_only_t_net_pnl_cny']:.2f} | {item['combined_t_net_pnl_cny']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 治理边界",
            "",
            "- T收益与底仓收益完全分账；只有`intraday_t.net_pnl_cny`属于赚波动的钱。",
            "- 一分钟数据没有盘口队列，回测采用信号后下一分钟开盘并加不利滑点，不能证明限价单真实排队成交率。",
            "- 当前结果不授权自动下单；任何下一版必须更换经济机制或增加真正前向数据，不能在本五年历史上微调阈值。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths = {key: ROOT / value for key, value in config["inputs"].items() if key.endswith("_file")}
    audit = json.loads(paths["data_audit_file"].read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("一分钟做T数据审计未通过")
    required_round5 = [
        paths["base_ledger_file"],
        paths["base_trades_file"],
        round5.OUTPUT_DIR / "enhanced_stress_ledger.parquet",
        round5.OUTPUT_DIR / "enhanced_stress_trades.csv",
    ]
    if any(not path.exists() for path in required_round5):
        raise FileNotFoundError("缺少第五轮波段策略产物，请先运行防守/估值回测")

    minute = pd.read_parquet(paths["minute_file"])
    daily = pd.read_parquet(paths["daily_file"])
    dividends = pd.read_csv(paths["dividend_file"])
    h00300_raw = pd.read_parquet(paths["h00300_file"])
    enhanced = pd.read_parquet(paths["base_ledger_file"])
    enhanced_trades = pd.read_csv(paths["base_trades_file"])
    enhanced_stress = pd.read_parquet(round5.OUTPUT_DIR / "enhanced_stress_ledger.parquet")
    enhanced_stress_trades = pd.read_csv(round5.OUTPUT_DIR / "enhanced_stress_trades.csv")
    features = build_fixed_anchor_features(minute, daily)

    start = pd.Timestamp(enhanced["date"].min())
    end = pd.Timestamp(enhanced["date"].max())
    initial_cash = float(config["account"]["initial_cash_cny"])
    calendar = pd.to_datetime(enhanced["date"])
    static_targets = _static_half_targets(calendar)
    static_base, static_base_trades = run_long_cash_backtest(
        daily,
        dividends,
        static_targets,
        initial_cash,
        _backtest_costs(config, float(config["costs"]["base_slippage_bps_per_leg"])),
        start,
        end,
    )
    static_stress, static_stress_trades = run_long_cash_backtest(
        daily,
        dividends,
        static_targets,
        initial_cash,
        _backtest_costs(config, float(config["costs"]["stress_slippage_bps_per_leg"])),
        start,
        end,
    )
    buy_hold_targets = round5._build_buy_hold_targets(calendar)
    buy_hold, buy_hold_trades = run_long_cash_backtest(
        daily,
        dividends,
        buy_hold_targets,
        initial_cash,
        _backtest_costs(config, float(config["costs"]["base_slippage_bps_per_leg"])),
        start,
        end,
    )
    h00300 = round5._benchmark_ledger(h00300_raw, enhanced["date"], initial_cash)

    def trade_dates(frame: pd.DataFrame) -> set[pd.Timestamp]:
        return set() if frame.empty else set(pd.to_datetime(frame["date"]).dt.normalize())

    t_only = run_anchor_t_overlay(
        features,
        static_base,
        trade_dates(static_base_trades),
        config,
        "BASE_COST",
        float(config["costs"]["base_slippage_bps_per_leg"]),
        "固定50%底仓频繁T",
    )
    combined = run_anchor_t_overlay(
        features,
        enhanced,
        trade_dates(enhanced_trades),
        config,
        "BASE_COST",
        float(config["costs"]["base_slippage_bps_per_leg"]),
        "估值库存+频繁T",
    )
    t_only_stress = run_anchor_t_overlay(
        features,
        static_stress,
        trade_dates(static_stress_trades),
        config,
        "STRESS_15BP",
        float(config["costs"]["stress_slippage_bps_per_leg"]),
        "固定50%底仓频繁T",
    )
    combined_stress = run_anchor_t_overlay(
        features,
        enhanced_stress,
        trade_dates(enhanced_stress_trades),
        config,
        "STRESS_15BP",
        float(config["costs"]["stress_slippage_bps_per_leg"]),
        "估值库存+频繁T",
    )

    h_summary = round5._index_summary(h00300, initial_cash)
    h_summary.update(
        {
            "strategy_name": "H00300全收益",
            "scenario": "BENCHMARK",
            "average_exposure": 1.0,
            "intraday_t": {"round_trip_count": 0, "net_pnl_cny": 0.0},
        }
    )
    buy_hold_summary = _metric_frame(
        "510300买入持有", "BASE_COST", buy_hold, initial_cash, "actual_position"
    )
    buy_hold_summary["annualized_excess_vs_h00300"] = buy_hold_summary["cagr"] - h_summary["cagr"]
    swing_summary = _metric_frame(
        "波段持仓", "BASE_COST", enhanced, initial_cash, "actual_position"
    )
    swing_summary = _attach_benchmark_metrics(swing_summary, enhanced, h00300, buy_hold)

    summaries = {
        "h00300": h_summary,
        "etf_buy_hold": buy_hold_summary,
        "swing": swing_summary,
        "t_only": _attach_benchmark_metrics(
            _overlay_summary(t_only, initial_cash), t_only.ledger, h00300, buy_hold
        ),
        "combined": _attach_benchmark_metrics(
            _overlay_summary(combined, initial_cash), combined.ledger, h00300, buy_hold
        ),
        "t_only_stress": _attach_benchmark_metrics(
            _overlay_summary(t_only_stress, initial_cash), t_only_stress.ledger, h00300, buy_hold
        ),
        "combined_stress": _attach_benchmark_metrics(
            _overlay_summary(combined_stress, initial_cash), combined_stress.ledger, h00300, buy_hold
        ),
    }
    annual = _annual_analysis(
        {
            "swing": enhanced,
            "t_only": t_only.ledger,
            "combined": combined.ledger,
            "h00300": h00300,
        },
        {"t_only": t_only.trades, "combined": combined.trades},
    )
    payload = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "strategy": config["strategy"],
        "summaries": summaries,
        "gates": {
            key: _gate(summaries[key], config) for key in ("swing", "t_only", "combined")
        },
        "t_stability": {
            "t_only": _t_stability(t_only.trades, annual, "t_only"),
            "combined": _t_stability(combined.trades, annual, "combined"),
        },
        "t_economics": {},
        "annual": annual,
        "data_hashes": {name: _sha256(path) for name, path in {"config": CONFIG_FILE, **paths}.items()},
    }
    elapsed_years = max((combined.ledger["date"].iloc[-1] - combined.ledger["date"].iloc[0]).days / 365.25, 1e-9)
    combined_rounds = max(int(len(combined.trades)), 1)
    combined_t = summaries["combined"]["intraday_t"]
    h_cagr = summaries["h00300"]["cagr"]
    required_10_end = initial_cash * (1.0 + h_cagr + float(config["evaluation"]["annualized_excess_minimum"])) ** elapsed_years
    required_15_end = initial_cash * (1.0 + h_cagr + float(config["evaluation"]["annualized_excess_challenge"])) ** elapsed_years
    observed_cost = combined_t["commission_cny"] + combined_t["slippage_and_tick_cost_cny"]
    payload["t_economics"] = {
        "observed_gross_per_round_cny": combined_t["raw_gross_pnl_cny"] / combined_rounds,
        "observed_cost_per_round_cny": observed_cost / combined_rounds,
        "observed_net_per_round_cny": combined_t["net_pnl_cny"] / combined_rounds,
        "required_net_per_round_for_10pct_excess_cny": max(required_10_end - summaries["swing"]["ending_equity"], 0.0) / combined_rounds,
        "required_net_per_round_for_15pct_excess_cny": max(required_15_end - summaries["swing"]["ending_equity"], 0.0) / combined_rounds,
        "assumption": "保持当前组合T回合数不变，T净收益直接叠加到波段策略期末权益，未假设复投",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    for prefix, result in (
        ("t_only", t_only),
        ("combined", combined),
        ("t_only_stress", t_only_stress),
        ("combined_stress", combined_stress),
    ):
        result.ledger.to_parquet(OUTPUT_DIR / f"{prefix}_ledger.parquet", index=False)
        result.trades.to_csv(OUTPUT_DIR / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_render(payload), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for frame, label in (
        (enhanced, "波段持仓"),
        (t_only.ledger, "固定50%底仓频繁T"),
        (combined.ledger, "估值库存+频繁T"),
        (buy_hold, "510300买入持有"),
        (h00300, "H00300全收益"),
    ):
        axes[0].plot(frame["date"], frame["equity"] / initial_cash, label=label)
    axes[0].set_ylabel("归一化净值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(combined.ledger["date"], combined.ledger["cumulative_t_net_pnl_cny"], label="组合策略累计T净收益")
    axes[1].plot(t_only.ledger["date"], t_only.ledger["cumulative_t_net_pnl_cny"], label="固定底仓累计T净收益")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("做T累计净收益（元）")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(REPORT_PNG, dpi=160)
    plt.close(figure)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
