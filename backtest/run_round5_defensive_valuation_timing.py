"""运行第五轮：防守型趋势择时基线与估值战略仓位增强策略。"""

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

from backtest.defensive_valuation_timing_engine import (
    build_model_positions,
    build_timing_features,
    schedule_asymmetric_execution,
)
from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest
from backtest.valuation_fvg_engine import attach_market_cap_context, build_valuation_signals


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "round5_defensive_valuation_timing.yaml"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
H00300_FILE = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
CONSTITUENT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "round5_defensive_valuation_timing"
REPORT_DIR = ROOT / "reports" / "backtest"
PAPER_DIR = ROOT / "paper"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _costs(config: dict[str, Any], slippage_bps: float) -> BacktestCosts:
    account = config["account"]
    costs = config["costs"]
    return BacktestCosts(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_rate=float(costs["stamp_duty_rate"]),
        slippage_bps=float(slippage_bps),
        lot_size=int(account["lot_size"]),
        cash_annual_rate=float(account["cash_annual_rate"]),
    )


def _benchmark_ledger(
    benchmark: pd.DataFrame,
    calendar: pd.Series,
    initial_cash: float,
) -> pd.DataFrame:
    frame = benchmark[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates("date").set_index("date").reindex(pd.DatetimeIndex(calendar))
    if frame["close"].isna().any():
        missing = frame.index[frame["close"].isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"H00300无法覆盖策略交易日历：{missing[:5]}")
    frame = frame.reset_index().rename(columns={"index": "date"})
    frame["daily_return"] = frame["close"].pct_change(fill_method=None).fillna(0.0)
    frame["equity"] = initial_cash * (1.0 + frame["daily_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def _index_summary(ledger: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(242))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": (
            float(ledger["daily_return"].mean() * 242 / volatility) if volatility > 0 else None
        ),
        "max_drawdown": float(ledger["drawdown"].min()),
        "ending_equity": float(ledger["equity"].iloc[-1]),
    }


def _rolling_excess(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: int = 242,
) -> dict[str, Any]:
    left = strategy.set_index("date")["equity"]
    right = benchmark.set_index("date")["equity"]
    excess = left.pct_change(window, fill_method=None) - right.pct_change(
        window, fill_method=None
    )
    valid = excess.dropna()
    return {
        "window_trading_days": window,
        "median": float(valid.median()),
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "positive_ratio": float(valid.gt(0.0).mean()),
        "at_least_10pct_ratio": float(valid.ge(0.10).mean()),
        "at_least_15pct_ratio": float(valid.ge(0.15).mean()),
        "observations": int(len(valid)),
    }


def _slippage_cost(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    price_gap = (trades["execution_price"] - trades["open_price"]).abs()
    return float((price_gap * trades["quantity"]).sum())


def _strategy_summary(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    h00300: pd.DataFrame,
    etf_buy_hold: pd.DataFrame,
) -> dict[str, Any]:
    summary = summarize_backtest(ledger, trades, initial_cash)
    h00300_summary = _index_summary(h00300, initial_cash)
    etf_summary = summarize_backtest(etf_buy_hold, pd.DataFrame(), initial_cash)
    summary.update(
        {
            "annualized_excess_vs_h00300": summary["cagr"] - h00300_summary["cagr"],
            "annualized_excess_vs_etf_buy_hold": summary["cagr"] - etf_summary["cagr"],
            "rolling_242d_excess_vs_h00300": _rolling_excess(ledger, h00300),
            "rolling_242d_excess_vs_etf_buy_hold": _rolling_excess(ledger, etf_buy_hold),
            "modeled_slippage_cost_cny": _slippage_cost(trades),
            "risk_off_override_trade_count": (
                int(trades["risk_off_override"].sum()) if not trades.empty else 0
            ),
        }
    )
    return summary


def _annual_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, frame in frames.items():
        values = frame[["date", "equity"]].rename(columns={"equity": name})
        merged = values if merged is None else merged.merge(values, on="date", validate="one_to_one")
    if merged is None:
        return pd.DataFrame()
    merged["year"] = merged["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in merged.groupby("year", sort=True):
        row: dict[str, Any] = {"year": int(year), "observations": int(len(group))}
        for name in frames:
            row[name] = float(group[name].iloc[-1] / group[name].iloc[0] - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_buy_hold_targets(calendar: pd.Series) -> pd.DataFrame:
    targets = pd.DataFrame({"date": pd.to_datetime(calendar), "target_position": 1.0})
    targets["trade_allowed"] = False
    targets.loc[targets.index[0], "trade_allowed"] = True
    targets["risk_off_override"] = False
    targets["signal_reason"] = "买入持有首次建仓"
    return targets


def _execution_sensitivity(
    etf: pd.DataFrame,
    dividends: pd.DataFrame,
    scheduled_targets: pd.DataFrame,
    daily_model_targets: pd.DataFrame,
    initial_cash: float,
    base_costs: BacktestCosts,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """区分信号能力与小账户成交门槛影响，不据此回选参数。"""

    rows: list[dict[str, Any]] = []
    for threshold in (0.0, 2_000.0, 4_000.0, 6_000.0, 8_000.0, 10_000.0):
        ledger, trades = run_long_cash_backtest(
            etf,
            dividends,
            scheduled_targets,
            initial_cash,
            base_costs,
            start,
            end,
            threshold,
        )
        summary = summarize_backtest(ledger, trades, initial_cash)
        rows.append(
            {
                "scenario": f"真实成本_普通成交门槛{threshold:.0f}元",
                "cagr": summary["cagr"],
                "max_drawdown": summary["max_drawdown"],
                "trade_count": summary["trade_count"],
                "commission_cny": summary["total_explicit_cost_cny"],
                "average_exposure": summary["average_exposure"],
            }
        )
    frictionless = BacktestCosts(
        commission_rate=0.0,
        minimum_commission_cny=0.0,
        stamp_duty_rate=0.0,
        slippage_bps=0.0,
        lot_size=1,
        cash_annual_rate=base_costs.cash_annual_rate,
    )
    for scenario, targets in (
        ("零摩擦_非对称周度执行", scheduled_targets),
        ("零摩擦_每日执行离散模型", daily_model_targets),
    ):
        ledger, trades = run_long_cash_backtest(
            etf, dividends, targets, initial_cash, frictionless, start, end, 0.0
        )
        summary = summarize_backtest(ledger, trades, initial_cash)
        rows.append(
            {
                "scenario": scenario,
                "cagr": summary["cagr"],
                "max_drawdown": summary["max_drawdown"],
                "trade_count": summary["trade_count"],
                "commission_cny": summary["total_explicit_cost_cny"],
                "average_exposure": summary["average_exposure"],
            }
        )
    return rows


def _render_report(payload: dict[str, Any], annual: pd.DataFrame) -> str:
    summaries = payload["summaries"]
    benchmark = summaries["h00300"]
    etf = summaries["etf_buy_hold"]
    baseline = summaries["baseline_base_cost"]
    enhanced = summaries["enhanced_base_cost"]
    stress = summaries["enhanced_stress_cost"]
    pct = lambda value: f"{value:.2%}"
    lines = [
        "# 第五轮：510300防守择时与估值增强对照",
        "",
        "> 本轮使用一次冻结参数，不做历史网格搜索。全部历史均已被查看，结论属于回顾性证据，不是严格样本外证明。",
        "",
        "## 核心结果",
        "",
        "| 方案 | CAGR | 对H00300年化超额 | 对ETF持有年化超额 | 最大回撤 | Sharpe | 平均仓位 | 成交笔数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| H00300全收益 | {pct(benchmark['cagr'])} | 0.00% | — | {pct(benchmark['max_drawdown'])} | {benchmark['sharpe_zero_cash_rate']:.2f} | 100.00% | — |",
        f"| 510300买入持有 | {pct(etf['cagr'])} | {pct(etf['cagr'] - benchmark['cagr'])} | 0.00% | {pct(etf['max_drawdown'])} | {etf['sharpe_zero_cash_rate']:.2f} | {pct(etf['average_exposure'])} | {etf['trade_count']} |",
        f"| 防守趋势基线（5bp） | {pct(baseline['cagr'])} | {pct(baseline['annualized_excess_vs_h00300'])} | {pct(baseline['annualized_excess_vs_etf_buy_hold'])} | {pct(baseline['max_drawdown'])} | {baseline['sharpe_zero_cash_rate']:.2f} | {pct(baseline['average_exposure'])} | {baseline['trade_count']} |",
        f"| 估值增强（5bp） | {pct(enhanced['cagr'])} | {pct(enhanced['annualized_excess_vs_h00300'])} | {pct(enhanced['annualized_excess_vs_etf_buy_hold'])} | {pct(enhanced['max_drawdown'])} | {enhanced['sharpe_zero_cash_rate']:.2f} | {pct(enhanced['average_exposure'])} | {enhanced['trade_count']} |",
        f"| 估值增强压力成本（15bp） | {pct(stress['cagr'])} | {pct(stress['annualized_excess_vs_h00300'])} | {pct(stress['annualized_excess_vs_etf_buy_hold'])} | {pct(stress['max_drawdown'])} | {stress['sharpe_zero_cash_rate']:.2f} | {pct(stress['average_exposure'])} | {stress['trade_count']} |",
        "",
        "## 本轮回答",
        "",
        f"- 估值中枢相对防守基线的年化增量：**{pct(payload['valuation_incremental_cagr'])}**。",
        f"- 估值增强相对H00300的滚动242日超额中位数：**{pct(enhanced['rolling_242d_excess_vs_h00300']['median'])}**；为正比例：**{pct(enhanced['rolling_242d_excess_vs_h00300']['positive_ratio'])}**。",
        f"- 滚动242日超额达到10%/15%的比例分别为：**{pct(enhanced['rolling_242d_excess_vs_h00300']['at_least_10pct_ratio'])} / {pct(enhanced['rolling_242d_excess_vs_h00300']['at_least_15pct_ratio'])}**。",
        f"- 估值增强最大回撤较ETF买入持有变化：**{pct(enhanced['max_drawdown'] - etf['max_drawdown'])}**（正值代表回撤收窄）。",
        f"- 2万元账户普通成交门槛为1万元；估值增强共成交{enhanced['trade_count']}笔，其中风险事件小额降仓豁免{enhanced['risk_off_override_trade_count']}笔。",
        f"- 基础成本下显式佣金{enhanced['total_explicit_cost_cny']:.2f}元，模拟滑点成本{enhanced['modeled_slippage_cost_cny']:.2f}元；15bp压力下CAGR变化{pct(stress['cagr'] - enhanced['cagr'])}。",
        "",
        "## 冻结规则",
        "",
        "1. 防守基线：60/200日连续趋势得分映射到20%—100%基础仓位；风险层不与趋势仓位相乘，而是最多减20个百分点。",
        "2. 风险层：同时使用RV5/RV20、RV20/RV60和20日下行波动占比；单日跌幅或波动结构触发危机条件时允许立即降仓。",
        "3. 估值增强：历史点时指数聚合估值先给20%—100%战略仓位，趋势最多确认±20个百分点，再扣除风险惩罚。",
        "4. 执行层：模型连续输出映射到10%仓位档；普通信号每5个交易日复核一次，恢复仓位每次最多10个百分点。",
        "5. 成交层：t日收盘形成信号，t+1开盘成交；100份整手、最低5元佣金、基础5bp/压力15bp滑点。",
        "6. 折溢价层：历史交易时刻IOPV不可得，因此停用。它只保留为未来限价/暂缓执行条件，不作为隔夜收益预测因子。",
        "",
        "## 自然年度",
        "",
        "| 年份 | 防守基线 | 估值增强 | ETF买入持有 | H00300全收益 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.year} | {pct(row.baseline)} | {pct(row.enhanced)} | {pct(row.etf_buy_hold)} | {pct(row.h00300)} |"
        )
    lines.extend(
        [
            "",
            "## 数据与治理边界",
            "",
            "- 主要Alpha基准是H00300；510300买入持有是实际可交易基准。当前仓库没有同口径的000888与000001历史序列，因此未把它们混入本轮核心判断。",
            "- 估值模型是指数聚合盈利/净资产代理，不是逐家公司DCF；参数与结构已受既有历史研究影响。",
            "- 当前配置不授权自动下单。下一步应冻结代码、配置和数据指纹后进行纸面前向验证。",
            "",
            "## 执行敏感性（诊断，不用于回选参数）",
            "",
            "| 场景 | CAGR | 最大回撤 | 成交笔数 | 平均仓位 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in payload["execution_sensitivity"]:
        lines.append(
            f"| {item['scenario']} | {pct(item['cagr'])} | {pct(item['max_drawdown'])} | {item['trade_count']} | {pct(item['average_exposure'])} |"
        )
    lines.extend(
        [
            "",
            "零摩擦每日执行仍未接近10%年化超额，说明主要瓶颈是信号收益能力，而非佣金或1万元成交门槛。不同门槛形成不同持仓路径，1万元门槛的较好结果不能解释为单调成本优势。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    etf = pd.read_parquet(ETF_FILE)
    index = pd.read_parquet(INDEX_FILE)
    h00300_raw = pd.read_parquet(H00300_FILE)
    valuation_raw = pd.read_parquet(VALUATION_FILE)
    constituents = pd.read_parquet(CONSTITUENT_FILE)
    dividends = pd.read_csv(DIVIDEND_FILE)

    timing = build_timing_features(index, config)
    valuation = attach_market_cap_context(
        build_valuation_signals(valuation_raw, config["valuation"]), constituents
    )
    baseline_model = build_model_positions(timing, config)
    enhanced_model = build_model_positions(timing, config, valuation)
    baseline_targets = schedule_asymmetric_execution(baseline_model, config)
    enhanced_targets = schedule_asymmetric_execution(enhanced_model, config)

    start = pd.Timestamp(config["backtest"]["start_date"])
    end = pd.Timestamp(config["backtest"]["end_date"])
    etf_calendar = etf.loc[pd.to_datetime(etf["date"]).between(start, end), "date"]
    calendar_frame = pd.DataFrame({"date": pd.to_datetime(etf_calendar)})
    baseline_targets = calendar_frame.merge(baseline_targets, on="date", how="left", validate="one_to_one")
    enhanced_targets = calendar_frame.merge(enhanced_targets, on="date", how="left", validate="one_to_one")
    daily_model_targets = calendar_frame.merge(enhanced_model, on="date", how="left", validate="one_to_one")
    daily_model_targets["target_position"] = daily_model_targets["discrete_model_position"]
    daily_model_targets["trade_allowed"] = True
    daily_model_targets["risk_off_override"] = True
    daily_model_targets["signal_reason"] = "诊断：每日执行离散模型"
    required_target_columns = ["target_position", "trade_allowed", "risk_off_override"]
    if baseline_targets[required_target_columns].isna().any().any():
        raise ValueError("防守基线无法完整覆盖ETF交易日历")
    if enhanced_targets[required_target_columns].isna().any().any():
        raise ValueError("估值增强模型无法完整覆盖ETF交易日历")

    initial_cash = float(config["backtest"]["initial_cash_cny"])
    minimum_notional = float(config["execution"]["minimum_normal_trade_notional_cny"])
    base_costs = _costs(config, float(config["costs"]["base_slippage_bps_per_leg"]))
    stress_costs = _costs(config, float(config["costs"]["stress_slippage_bps_per_leg"]))
    baseline_ledger, baseline_trades = run_long_cash_backtest(
        etf, dividends, baseline_targets, initial_cash, base_costs, start, end, minimum_notional
    )
    enhanced_ledger, enhanced_trades = run_long_cash_backtest(
        etf, dividends, enhanced_targets, initial_cash, base_costs, start, end, minimum_notional
    )
    stress_ledger, stress_trades = run_long_cash_backtest(
        etf, dividends, enhanced_targets, initial_cash, stress_costs, start, end, minimum_notional
    )
    buy_hold_targets = _build_buy_hold_targets(etf_calendar)
    buy_hold_ledger, buy_hold_trades = run_long_cash_backtest(
        etf, dividends, buy_hold_targets, initial_cash, base_costs, start, end
    )
    h00300 = _benchmark_ledger(h00300_raw, baseline_ledger["date"], initial_cash)

    h00300_summary = _index_summary(h00300, initial_cash)
    buy_hold_summary = summarize_backtest(buy_hold_ledger, buy_hold_trades, initial_cash)
    summaries = {
        "h00300": h00300_summary,
        "etf_buy_hold": buy_hold_summary,
        "baseline_base_cost": _strategy_summary(
            baseline_ledger, baseline_trades, initial_cash, h00300, buy_hold_ledger
        ),
        "enhanced_base_cost": _strategy_summary(
            enhanced_ledger, enhanced_trades, initial_cash, h00300, buy_hold_ledger
        ),
        "enhanced_stress_cost": _strategy_summary(
            stress_ledger, stress_trades, initial_cash, h00300, buy_hold_ledger
        ),
    }
    annual = _annual_table(
        {
            "baseline": baseline_ledger,
            "enhanced": enhanced_ledger,
            "etf_buy_hold": buy_hold_ledger,
            "h00300": h00300,
        }
    )
    latest = enhanced_targets.iloc[-1]
    execution_sensitivity = _execution_sensitivity(
        etf,
        dividends,
        enhanced_targets,
        daily_model_targets,
        initial_cash,
        base_costs,
        start,
        end,
    )
    hashes = {
        path.name: _sha256(path)
        for path in (CONFIG_FILE, ETF_FILE, INDEX_FILE, H00300_FILE, VALUATION_FILE, CONSTITUENT_FILE, DIVIDEND_FILE)
    }
    payload = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "strategy": config["strategy"],
        "summaries": summaries,
        "valuation_incremental_cagr": (
            summaries["enhanced_base_cost"]["cagr"] - summaries["baseline_base_cost"]["cagr"]
        ),
        "execution_sensitivity": execution_sensitivity,
        "acceptance_gate": {
            "minimum_annualized_excess": float(config["backtest"]["annualized_excess_minimum"]),
            "challenge_annualized_excess": float(config["backtest"]["annualized_excess_challenge"]),
            "minimum_rolling_242d_excess_median": float(
                config["backtest"]["rolling_242d_excess_median_minimum"]
            ),
            "challenge_rolling_242d_excess_median": float(
                config["backtest"]["rolling_242d_excess_median_challenge"]
            ),
            "minimum_status": (
                "PASS_RETROSPECTIVE"
                if summaries["enhanced_base_cost"]["annualized_excess_vs_h00300"]
                >= float(config["backtest"]["annualized_excess_minimum"])
                and summaries["enhanced_base_cost"]["rolling_242d_excess_vs_h00300"]["median"]
                >= float(config["backtest"]["rolling_242d_excess_median_minimum"])
                else "FAIL_RETROSPECTIVE"
            ),
            "challenge_status": (
                "PASS_RETROSPECTIVE"
                if summaries["enhanced_base_cost"]["annualized_excess_vs_h00300"]
                >= float(config["backtest"]["annualized_excess_challenge"])
                and summaries["enhanced_base_cost"]["rolling_242d_excess_vs_h00300"]["median"]
                >= float(config["backtest"]["rolling_242d_excess_median_challenge"])
                else "FAIL_RETROSPECTIVE"
            ),
        },
        "diagnostics": {
            "trend_risk_stress_correlation": float(
                baseline_targets[["trend_score", "risk_stress_score"]].corr().iloc[0, 1]
            ),
            "crisis_event_days": int(baseline_targets["crisis_event"].sum()),
            "premium_discount_layer": config["execution"]["premium_discount_layer"],
            "normal_minimum_trade_notional_cny": minimum_notional,
        },
        "latest_signal": {
            "signal_date": str(pd.Timestamp(latest["date"]).date()),
            "strategic_position": float(latest["strategic_position"]),
            "trend_score": float(latest["trend_score"]),
            "risk_penalty": float(latest["risk_penalty"]),
            "continuous_model_position": float(latest["continuous_model_position"]),
            "target_position": float(latest["target_position"]),
            "trade_allowed": bool(latest["trade_allowed"]),
            "is_review_day": bool(latest["is_review_day"]),
            "risk_off_override": bool(latest["risk_off_override"]),
            "allow_position_increase": bool(latest["allow_position_increase"]),
            "allow_position_decrease": bool(latest["allow_position_decrease"]),
            "signal_reason": str(latest["signal_reason"]),
            "automatic_ordering_authorized": False,
        },
        "data_hashes": hashes,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    baseline_targets.to_parquet(OUTPUT_DIR / "baseline_signals.parquet", index=False)
    enhanced_targets.to_parquet(OUTPUT_DIR / "enhanced_signals.parquet", index=False)
    baseline_ledger.to_parquet(OUTPUT_DIR / "baseline_ledger.parquet", index=False)
    baseline_trades.to_csv(OUTPUT_DIR / "baseline_trades.csv", index=False, encoding="utf-8-sig")
    enhanced_ledger.to_parquet(OUTPUT_DIR / "enhanced_ledger.parquet", index=False)
    enhanced_trades.to_csv(OUTPUT_DIR / "enhanced_trades.csv", index=False, encoding="utf-8-sig")
    stress_ledger.to_parquet(OUTPUT_DIR / "enhanced_stress_ledger.parquet", index=False)
    stress_trades.to_csv(OUTPUT_DIR / "enhanced_stress_trades.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUTPUT_DIR / "annual_returns.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "round5_defensive_valuation_timing.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "round5_defensive_valuation_timing.md").write_text(
        _render_report(payload, annual), encoding="utf-8"
    )
    (PAPER_DIR / "round5_latest_signal.json").write_text(
        json.dumps(payload["latest_signal"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for frame, label in (
        (baseline_ledger, "防守趋势基线"),
        (enhanced_ledger, "估值战略仓位增强"),
        (buy_hold_ledger, "510300买入持有"),
        (h00300, "H00300全收益"),
    ):
        axes[0].plot(frame["date"], frame["equity"] / initial_cash, label=label)
    axes[0].set_ylabel("归一化净值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(enhanced_targets["date"], enhanced_targets["continuous_model_position"], label="连续模型仓位", alpha=0.7)
    axes[1].step(enhanced_targets["date"], enhanced_targets["target_position"], label="执行目标仓位", where="post")
    axes[1].set_ylabel("仓位")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "round5_defensive_valuation_timing.png", dpi=160)
    plt.close(figure)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
