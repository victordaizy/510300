"""对R5目标执行独立的2万元、25%档、1000份门槛端到端历史回测。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.defensive_valuation_timing_engine import (
    build_model_positions,
    build_timing_features,
    schedule_asymmetric_execution,
)
from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest
from backtest.small_account_execution import quantize_position
from backtest.valuation_fvg_engine import attach_market_cap_context, build_valuation_signals


ROOT = Path(__file__).resolve().parents[1]
R5_CONFIG_FILE = ROOT / "config" / "round5_defensive_valuation_timing.yaml"
ACCOUNT_CONFIG_FILE = ROOT / "config" / "small_account_20000.yaml"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
CONSTITUENT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "round5_small_account_execution"
REPORT_JSON = ROOT / "reports" / "backtest" / "round5_small_account_execution.json"
REPORT_MD = ROOT / "reports" / "backtest" / "round5_small_account_execution.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_small_account_targets(
    index: pd.DataFrame,
    valuation_raw: pd.DataFrame,
    constituents: pd.DataFrame,
    etf_calendar: pd.Series,
    r5_config: dict[str, Any],
    account_config: dict[str, Any],
) -> pd.DataFrame:
    """重建R5历史目标，再在执行层独立映射到25%档位。"""

    timing = build_timing_features(index, r5_config)
    valuation = attach_market_cap_context(
        build_valuation_signals(valuation_raw, r5_config["valuation"]), constituents
    )
    model = build_model_positions(timing, r5_config, valuation)
    scheduled = schedule_asymmetric_execution(model, r5_config)
    calendar = pd.DataFrame({"date": pd.to_datetime(etf_calendar)})
    targets = calendar.merge(scheduled, on="date", how="left", validate="one_to_one")
    step = float(account_config["execution"]["position_grid_step"])
    targets["r5_target_position"] = targets["target_position"]
    targets["target_position"] = targets["target_position"].map(
        lambda value: quantize_position(value, step) if pd.notna(value) else value
    )
    required = ["target_position", "trade_allowed", "risk_off_override"]
    if targets[required].isna().any().any():
        raise ValueError("R5小账户目标无法完整覆盖回测交易日历")
    return targets


def _buy_hold_targets(calendar: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame({"date": pd.to_datetime(calendar), "target_position": 1.0})
    result["trade_allowed"] = False
    result.loc[result.index[0], "trade_allowed"] = True
    result["risk_off_override"] = False
    result["signal_reason"] = "买入持有首次建仓"
    return result


def _render(payload: dict[str, Any]) -> str:
    strategy = payload["strategy_summary"]
    benchmark = payload["buy_hold_summary"]
    pct = lambda value: f"{value:.2%}"
    return "\n".join(
        [
            "# R5两万元外层执行独立历史回测",
            "",
            "> 本报告从R5逐日目标重新执行25%仓位档、100份整手、普通调仓至少1000份、最低佣金和滑点；不继承底层R5绩效数字。",
            "",
            "| 指标 | R5外层执行 | 510300含分红买入持有 |",
            "|---|---:|---:|",
            f"| CAGR | {pct(strategy['cagr'])} | {pct(benchmark['cagr'])} |",
            f"| Sharpe | {strategy['sharpe_zero_cash_rate']:.3f} | {benchmark['sharpe_zero_cash_rate']:.3f} |",
            f"| 最大回撤 | {pct(strategy['max_drawdown'])} | {pct(benchmark['max_drawdown'])} |",
            f"| 平均仓位 | {pct(strategy['average_exposure'])} | {pct(benchmark['average_exposure'])} |",
            f"| 成交笔数 | {strategy['trade_count']} | {benchmark['trade_count']} |",
            f"| 显式成本 | {strategy['total_explicit_cost_cny']:.2f}元 | {benchmark['total_explicit_cost_cny']:.2f}元 |",
            "",
            f"外层执行相对买入持有年化差：{pct(payload['annualized_excess_vs_buy_hold'])}。",
            "",
            "治理边界：仅纸面研究，不连接券商，不提交订单。",
        ]
    )


def main() -> int:
    r5_config = yaml.safe_load(R5_CONFIG_FILE.read_text(encoding="utf-8"))
    account_config = yaml.safe_load(ACCOUNT_CONFIG_FILE.read_text(encoding="utf-8"))
    etf = pd.read_parquet(ETF_FILE)
    index = pd.read_parquet(INDEX_FILE)
    valuation = pd.read_parquet(VALUATION_FILE)
    constituents = pd.read_parquet(CONSTITUENT_FILE)
    dividends = pd.read_csv(DIVIDEND_FILE)
    start = pd.Timestamp(r5_config["backtest"]["start_date"])
    end = pd.Timestamp(r5_config["backtest"]["end_date"])
    calendar = etf.loc[pd.to_datetime(etf["date"]).between(start, end), "date"]
    targets = build_small_account_targets(
        index, valuation, constituents, calendar, r5_config, account_config
    )
    execution = account_config["execution"]
    costs = BacktestCosts(
        commission_rate=float(execution["commission_rate"]),
        minimum_commission_cny=float(execution["minimum_commission_cny"]),
        stamp_duty_rate=0.0,
        slippage_bps=float(execution["slippage_bps_per_leg"]),
        lot_size=int(execution["lot_size"]),
        cash_annual_rate=float(r5_config["account"]["cash_annual_rate"]),
    )
    initial_cash = float(account_config["account"]["reference_equity_cny"])
    ledger, trades = run_long_cash_backtest(
        etf,
        dividends,
        targets,
        initial_cash,
        costs,
        start,
        end,
        minimum_trade_shares=int(execution["minimum_normal_trade_shares"]),
    )
    buy_hold_ledger, buy_hold_trades = run_long_cash_backtest(
        etf,
        dividends,
        _buy_hold_targets(calendar),
        initial_cash,
        costs,
        start,
        end,
    )
    strategy_summary = summarize_backtest(ledger, trades, initial_cash)
    buy_hold_summary = summarize_backtest(buy_hold_ledger, buy_hold_trades, initial_cash)
    payload = {
        "strategy": "510300_SMALL_ACCOUNT_PAPER_V1_R5_EXECUTION_AUDIT",
        "independent_outer_execution": True,
        "execution": {
            "position_grid_step": float(execution["position_grid_step"]),
            "lot_size": int(execution["lot_size"]),
            "minimum_normal_trade_shares": int(execution["minimum_normal_trade_shares"]),
            "minimum_commission_cny": float(execution["minimum_commission_cny"]),
            "commission_rate": float(execution["commission_rate"]),
            "slippage_bps_per_leg": float(execution["slippage_bps_per_leg"]),
            "t_plus_one": True,
        },
        "strategy_summary": strategy_summary,
        "buy_hold_summary": buy_hold_summary,
        "annualized_excess_vs_buy_hold": strategy_summary["cagr"] - buy_hold_summary["cagr"],
        "data_hashes": {
            path.name: _sha256(path)
            for path in (
                R5_CONFIG_FILE,
                ACCOUNT_CONFIG_FILE,
                ETF_FILE,
                INDEX_FILE,
                VALUATION_FILE,
                CONSTITUENT_FILE,
                DIVIDEND_FILE,
            )
        },
        "governance": {
            "paper_only": True,
            "automatic_ordering_authorized": False,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(OUTPUT_DIR / "targets.parquet", index=False)
    ledger.to_parquet(OUTPUT_DIR / "ledger.parquet", index=False)
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(_render(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
