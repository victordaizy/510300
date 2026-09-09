"""510300满仓/空仓日内利弗莫尔关键点固定候选筛选 V1。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_intraday_binary_livermore_screen_v1.yaml"


@dataclass(frozen=True)
class CostModel:
    """单一成本情景。"""

    scenario: str
    commission_rate: float
    minimum_commission: float
    slippage_bps: float
    cash_annual_rate: float
    trading_days_per_year: int
    lot_size: int

    def commission(self, notional: float) -> float:
        if notional <= 0.0:
            return 0.0
        return max(self.minimum_commission, notional * self.commission_rate)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置文件不是映射结构")
    return config


def validate_protocol(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    objective = config["objective"]
    evaluation = config["evaluation"]
    family = config["candidate_family"]
    costs = config["costs"]

    if protocol["study_id"] != "510300_INTRADAY_BINARY_LIVERMORE_SCREEN_V1":
        raise ValueError("研究ID不匹配")
    if protocol["version"] != "1.0.0":
        raise ValueError("研究版本不匹配")
    if protocol["no_parameter_rescue"] is not True:
        raise ValueError("必须冻结为不允许参数救援")
    if list(scope["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("允许持有资产必须严格为510300和现金")
    if list(scope["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if any(
        bool(scope[key])
        for key in ("leverage_allowed", "short_selling_allowed", "derivatives_allowed")
    ):
        raise ValueError("禁止杠杆、卖空和衍生品")
    if bool(scope["same_bar_execution_allowed"]):
        raise ValueError("禁止同柱成交")
    if float(scope["initial_capital_cny"]) != 500000.0:
        raise ValueError("初始资金必须固定为50万元")
    if objective["benchmark_id"] != "H00300":
        raise ValueError("主基准必须是H00300")
    if float(objective["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额门槛必须固定为20个百分点")
    if float(objective["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动超额中位数门槛必须固定为20个百分点")
    if list(evaluation["start_date_perturbations_trading_days"]) != [0, 1, 2, 3, 4]:
        raise ValueError("起点扰动必须固定为0至4个交易日")
    if list(evaluation["cost_scenarios"]) != ["BASE", "STRESS"]:
        raise ValueError("成本情景必须固定为基础和压力")
    if int(family["candidate_count"]) != len(family["candidates"]):
        raise ValueError("候选数量声明与候选清单不一致")
    if int(family["candidate_count"]) != 13:
        raise ValueError("V1候选数量必须固定为13")
    candidate_ids = [item["id"] for item in family["candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("候选ID存在重复")
    if int(costs["lot_size_shares"]) != 100:
        raise ValueError("交易单位必须固定为100份")
    if float(costs["etf_stamp_duty_rate"]) != 0.0:
        raise ValueError("本协议ETF印花税必须固定为0")


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError("缺少冻结清单，禁止先看候选收益")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_state = "FROZEN_BEFORE_FIRST_CANDIDATE_PERFORMANCE_EVALUATION"
    if manifest.get("state") != expected_state:
        raise ValueError("冻结清单状态无效")
    tracked = manifest.get("tracked_files", {})
    for relative_path, expected_hash in tracked.items():
        actual_hash = sha256_file(ROOT / relative_path)
        if actual_hash != expected_hash:
            raise ValueError(f"冻结后文件漂移：{relative_path}")
    return {
        "path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(manifest_path),
        "created_at": manifest.get("created_at"),
        "state": manifest.get("state"),
    }


def _required_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name}缺少字段：{missing}")


def load_and_audit_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contracts = config["data_contracts"]
    bars_contract = contracts["intraday_bars"]
    factor_contract = contracts["factor_file"]
    benchmark_contract = contracts["benchmark_total_return"]
    dividend_contract = contracts["cash_distributions"]
    quality_contract = contracts["data_quality_receipt"]

    bars_path = ROOT / bars_contract["file"]
    factor_path = ROOT / factor_contract["file"]
    benchmark_path = ROOT / benchmark_contract["file"]
    dividend_path = ROOT / dividend_contract["file"]
    quality_path = ROOT / quality_contract["file"]

    expected_hashes = {
        bars_contract["file"]: bars_contract["required_sha256"],
        factor_contract["file"]: factor_contract["required_sha256"],
        benchmark_contract["file"]: benchmark_contract["required_sha256"],
        dividend_contract["file"]: dividend_contract["required_sha256"],
        quality_contract["file"]: quality_contract["required_sha256"],
    }
    actual_hashes = {relative: sha256_file(ROOT / relative) for relative in expected_hashes}
    hash_mismatches = {
        relative: {"expected": expected, "actual": actual_hashes[relative]}
        for relative, expected in expected_hashes.items()
        if actual_hashes[relative] != expected
    }
    if hash_mismatches:
        raise ValueError(f"输入哈希门失败：{json.dumps(hash_mismatches, ensure_ascii=False)}")

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("status") != quality_contract["required_status"]:
        raise ValueError("分钟数据质量回执状态不满足冻结要求")

    bars = pd.read_parquet(bars_path)
    _required_columns(bars, bars_contract["required_columns"], "15分钟行情")
    bars = bars.copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="raise").dt.normalize()
    bars["bar_end"] = pd.to_datetime(bars["bar_end"], errors="raise")
    bars.sort_values("bar_end", inplace=True)
    bars.reset_index(drop=True, inplace=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    bars_per_day = bars.groupby("trade_date", sort=True).size()
    bar_checks = {
        "row_count": len(bars) == int(bars_contract["required_rows"]),
        "trading_day_count": bars["trade_date"].nunique()
        == int(bars_contract["required_trading_days"]),
        "bars_per_day": bool(
            (bars_per_day == int(bars_contract["required_bars_per_day"])).all()
        ),
        "first_bar_end": bars["bar_end"].iloc[0].isoformat()
        == bars_contract["required_first_bar_end"],
        "last_bar_end": bars["bar_end"].iloc[-1].isoformat()
        == bars_contract["required_last_bar_end"],
        "symbol": set(bars["symbol"].astype(str)) == {bars_contract["required_symbol"]},
        "unique_bar_end": not bars["bar_end"].duplicated().any(),
        "positive_prices": bool((bars[["open", "high", "low", "close"]] > 0.0).all().all()),
        "ohlc_consistency": bool(
            (bars["high"] >= bars[["open", "close", "low"]].max(axis=1)).all()
            and (bars["low"] <= bars[["open", "close", "high"]].min(axis=1)).all()
        ),
        "nonnegative_volume_amount": bool((bars[["volume", "amount"]] >= 0.0).all().all()),
    }
    if not all(bar_checks.values()):
        raise ValueError(f"15分钟行情数据门失败：{bar_checks}")

    factors = pd.read_parquet(factor_path)
    _required_columns(factors, factor_contract["required_columns"], "技术因子")
    factors = factors[factor_contract["required_columns"]].copy()
    factors["bar_end"] = pd.to_datetime(factors["bar_end"], errors="raise")
    factors.sort_values("bar_end", inplace=True)
    factors.reset_index(drop=True, inplace=True)
    factor_checks = {
        "row_count": len(factors) == int(factor_contract["required_rows"]),
        "unique_bar_end": not factors["bar_end"].duplicated().any(),
        "same_bar_axis": factors["bar_end"].equals(bars["bar_end"]),
    }
    if not all(factor_checks.values()):
        raise ValueError(f"技术因子数据门失败：{factor_checks}")
    bars = bars.merge(factors, on="bar_end", how="left", validate="one_to_one")

    benchmark = pd.read_parquet(benchmark_path)
    _required_columns(benchmark, benchmark_contract["required_columns"], "H00300基准")
    benchmark = benchmark[["date", "close", "symbol"]].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise").dt.normalize()
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="raise")
    benchmark.sort_values("date", inplace=True)
    benchmark.reset_index(drop=True, inplace=True)
    if set(benchmark["symbol"].astype(str)) != {benchmark_contract["required_symbol"]}:
        raise ValueError("H00300基准代码不匹配")
    if benchmark["date"].duplicated().any() or (benchmark["close"] <= 0.0).any():
        raise ValueError("H00300基准日期重复或收盘价非法")
    required_dates = set(bars["trade_date"])
    missing_benchmark_dates = sorted(required_dates.difference(set(benchmark["date"])))
    if missing_benchmark_dates:
        raise ValueError(f"H00300缺少分钟样本交易日：{missing_benchmark_dates[:5]}")

    dividends = pd.read_csv(dividend_path)
    _required_columns(dividends, dividend_contract["required_columns"], "现金分红")
    if len(dividends) != int(dividend_contract["required_event_count"]):
        raise ValueError("现金分红事件数不匹配")
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if dividends["ex_date"].duplicated().any() or (
        dividends["cash_dividend_per_share"] <= 0.0
    ).any():
        raise ValueError("现金分红事件重复或金额非法")

    audit = {
        "status": "PASS",
        "input_hashes": actual_hashes,
        "intraday_bars": {
            "checks": bar_checks,
            "rows": int(len(bars)),
            "trading_days": int(bars["trade_date"].nunique()),
            "first_bar_end": bars["bar_end"].iloc[0].isoformat(),
            "last_bar_end": bars["bar_end"].iloc[-1].isoformat(),
        },
        "factor_file": {"checks": factor_checks, "rows": int(len(factors))},
        "benchmark": {
            "rows": int(len(benchmark)),
            "missing_intraday_dates": 0,
        },
        "dividends": {"event_count": int(len(dividends))},
        "quality_receipt_status": quality.get("status"),
        "source_caveats": quality.get("warnings", []),
    }
    return bars, benchmark, dividends, audit


def build_candidate_states(bars: pd.DataFrame, candidate: dict[str, Any]) -> np.ndarray:
    """仅用当前及更早柱构造目标状态；返回的状态在下一柱开盘执行。"""

    candidate_type = candidate["type"]
    close = bars["close"].astype(float)
    states = np.zeros(len(bars), dtype=np.int8)

    if candidate_type == "EMA_CROSS_STATE":
        fast = close.ewm(span=int(candidate["fast_span_bars"]), adjust=False).mean()
        slow = close.ewm(span=int(candidate["slow_span_bars"]), adjust=False).mean()
        valid_from = int(candidate["slow_span_bars"]) - 1
        states[valid_from:] = (fast.iloc[valid_from:] > slow.iloc[valid_from:]).astype(np.int8)
        return states

    if candidate_type not in {"DONCHIAN_HYSTERESIS", "DONCHIAN_ADX_PIVOT"}:
        raise ValueError(f"未知候选类型：{candidate_type}")

    entry_lookback = int(candidate["entry_lookback_bars"])
    exit_lookback = int(candidate["exit_lookback_bars"])
    prior_high = bars["high"].rolling(entry_lookback, min_periods=entry_lookback).max().shift(1)
    prior_low = bars["low"].rolling(exit_lookback, min_periods=exit_lookback).min().shift(1)
    state = 0
    for index in range(len(bars)):
        current_close = float(close.iloc[index])
        if state == 0 and pd.notna(prior_high.iloc[index]):
            entry = current_close > float(prior_high.iloc[index])
            if entry and candidate_type == "DONCHIAN_ADX_PIVOT":
                adx = bars.loc[index, "adx_14bar"]
                plus_di = bars.loc[index, "plus_di_14bar"]
                minus_di = bars.loc[index, "minus_di_14bar"]
                entry = bool(
                    pd.notna(adx)
                    and float(adx) >= float(candidate["entry_adx_minimum"])
                    and (
                        not bool(candidate["require_positive_directional_index"])
                        or (
                            pd.notna(plus_di)
                            and pd.notna(minus_di)
                            and float(plus_di) > float(minus_di)
                        )
                    )
                )
            if entry:
                state = 1
        elif state == 1 and pd.notna(prior_low.iloc[index]):
            if current_close < float(prior_low.iloc[index]):
                state = 0
        states[index] = state
    return states


def cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    if scenario not in {"BASE", "STRESS"}:
        raise ValueError("成本情景必须为BASE或STRESS")
    costs = config["costs"]
    slippage_key = (
        "base_slippage_bps_per_leg" if scenario == "BASE" else "stress_slippage_bps_per_leg"
    )
    return CostModel(
        scenario=scenario,
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(costs[slippage_key]),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def _dividend_maps(
    dividends: pd.DataFrame,
) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    ex_map: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for record in dividends.to_dict("records"):
        ex_map.setdefault(pd.Timestamp(record["ex_date"]), []).append(record)
    return ex_map


def _maximum_affordable_quantity(cash: float, raw_open: float, costs: CostModel) -> tuple[float, float, float]:
    execution_price = raw_open * (1.0 + costs.slippage_bps / 10000.0)
    approximate = int(
        cash / (execution_price * (1.0 + costs.commission_rate)) / costs.lot_size
    )
    quantity = float(max(approximate, 0) * costs.lot_size)
    while quantity > 0.0:
        notional = quantity * execution_price
        commission = costs.commission(notional)
        if notional + commission <= cash + 1e-9:
            return quantity, execution_price, commission
        quantity -= costs.lot_size
    return 0.0, execution_price, 0.0


def simulate_candidate(
    bars: pd.DataFrame,
    dividends: pd.DataFrame,
    target_signals: np.ndarray,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    costs: CostModel,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """执行下一柱开盘成交、整手、现金分红与T+1约束的二元状态账本。"""

    if len(bars) != len(target_signals):
        raise ValueError("目标状态长度与分钟行情不一致")
    if not set(np.unique(target_signals)).issubset({0, 1}):
        raise ValueError("目标状态只能为0或1")
    in_window = bars["trade_date"].between(start_date, end_date)
    indices = np.flatnonzero(in_window.to_numpy())
    if len(indices) == 0:
        raise ValueError("模拟区间没有分钟行情")
    first_index = int(indices[0])
    last_index = int(indices[-1])
    if first_index == 0:
        raise ValueError("模拟起点必须有前一根信号柱")
    prior_dates = bars.loc[: first_index - 1, "trade_date"]
    anchor_date = pd.Timestamp(prior_dates.iloc[-1])

    cash = float(initial_capital)
    shares = 0.0
    receivable = 0.0
    last_buy_date: pd.Timestamp | None = None
    current_date: pd.Timestamp | None = None
    payment_schedule: dict[pd.Timestamp, float] = {}
    ex_map = _dividend_maps(dividends)
    ledger_rows: list[dict[str, Any]] = [
        {
            "date": anchor_date,
            "equity": float(initial_capital),
            "shares": 0.0,
            "cash": float(initial_capital),
            "dividend_receivable": 0.0,
            "actual_state": 0,
            "desired_state_at_close": int(target_signals[first_index - 1]),
            "daily_commission": 0.0,
            "daily_slippage_cost": 0.0,
            "blocked_t1_exit_count": 0,
        }
    ]
    trade_rows: list[dict[str, Any]] = []
    daily_commission = 0.0
    daily_slippage = 0.0
    daily_blocked = 0
    total_blocked = 0
    total_reinvestment_buys = 0

    for index in range(first_index, last_index + 1):
        row = bars.iloc[index]
        date = pd.Timestamp(row["trade_date"])
        new_day = current_date is None or date != current_date
        if new_day:
            cash *= 1.0 + costs.cash_annual_rate / costs.trading_days_per_year
            current_date = date
            daily_commission = 0.0
            daily_slippage = 0.0
            daily_blocked = 0
            shares_at_open = shares
            for event in ex_map.get(date, []):
                entitlement = shares_at_open * float(event["cash_dividend_per_share"])
                if entitlement > 0.0:
                    receivable += entitlement
                    payment_date = pd.Timestamp(event["payment_date"])
                    payment_schedule[payment_date] = payment_schedule.get(payment_date, 0.0) + entitlement

        signal_index = index - 1
        desired_state = int(target_signals[signal_index])
        raw_open = float(row["open"])
        signal_asof = pd.Timestamp(bars.iloc[signal_index]["bar_end"])
        execution_time = pd.Timestamp(row["bar_end"]) - pd.Timedelta(minutes=15)

        if desired_state == 0 and shares > 0.0:
            if last_buy_date is not None and last_buy_date == date:
                daily_blocked += 1
                total_blocked += 1
            else:
                quantity = shares
                execution_price = raw_open * (1.0 - costs.slippage_bps / 10000.0)
                notional = quantity * execution_price
                commission = costs.commission(notional)
                slippage_cost = quantity * (raw_open - execution_price)
                cash += notional - commission
                shares = 0.0
                daily_commission += commission
                daily_slippage += slippage_cost
                trade_rows.append(
                    {
                        "date": date,
                        "signal_asof": signal_asof,
                        "execution_time": execution_time,
                        "side": "SELL",
                        "reason": "STATE_EXIT_TO_CASH",
                        "raw_open": raw_open,
                        "execution_price": execution_price,
                        "quantity": quantity,
                        "execution_notional": notional,
                        "commission": commission,
                        "slippage_cost": slippage_cost,
                        "scenario": costs.scenario,
                    }
                )
        elif desired_state == 1:
            quantity, execution_price, commission = _maximum_affordable_quantity(
                cash, raw_open, costs
            )
            if quantity > 0.0:
                notional = quantity * execution_price
                slippage_cost = quantity * (execution_price - raw_open)
                reason = "STATE_ENTRY_FULL" if shares == 0.0 else "FULL_STATE_CASH_REINVESTMENT"
                if shares > 0.0:
                    total_reinvestment_buys += 1
                cash -= notional + commission
                shares += quantity
                last_buy_date = date
                daily_commission += commission
                daily_slippage += slippage_cost
                trade_rows.append(
                    {
                        "date": date,
                        "signal_asof": signal_asof,
                        "execution_time": execution_time,
                        "side": "BUY",
                        "reason": reason,
                        "raw_open": raw_open,
                        "execution_price": execution_price,
                        "quantity": quantity,
                        "execution_notional": notional,
                        "commission": commission,
                        "slippage_cost": slippage_cost,
                        "scenario": costs.scenario,
                    }
                )

        is_day_end = index == last_index or pd.Timestamp(bars.iloc[index + 1]["trade_date"]) != date
        if is_day_end:
            payable_dates = [payment_date for payment_date in payment_schedule if payment_date <= date]
            payment_today = float(sum(payment_schedule.pop(payment_date) for payment_date in payable_dates))
            if payment_today > 0.0:
                cash += payment_today
                receivable -= payment_today
                if abs(receivable) < 1e-8:
                    receivable = 0.0
            close = float(row["close"])
            equity = cash + shares * close + receivable
            ledger_rows.append(
                {
                    "date": date,
                    "equity": equity,
                    "shares": shares,
                    "cash": cash,
                    "dividend_receivable": receivable,
                    "actual_state": int(shares > 0.0),
                    "desired_state_at_close": int(target_signals[index]),
                    "daily_commission": daily_commission,
                    "daily_slippage_cost": daily_slippage,
                    "blocked_t1_exit_count": daily_blocked,
                }
            )

    ledger = pd.DataFrame(ledger_rows)
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.normalize()
    ledger["daily_return"] = ledger["equity"].pct_change(fill_method=None).fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    trades = pd.DataFrame(trade_rows)
    diagnostics = {
        "t_plus_one_blocked_exit_attempts": int(total_blocked),
        "t_plus_one_violations": 0,
        "full_state_reinvestment_buy_legs": int(total_reinvestment_buys),
        "anchor_date": anchor_date.date().isoformat(),
        "actual_start_date": pd.Timestamp(ledger["date"].iloc[1]).date().isoformat(),
        "actual_end_date": pd.Timestamp(ledger["date"].iloc[-1]).date().isoformat(),
    }
    return ledger, trades, diagnostics


def _geometric_annual_return(returns: pd.Series, trading_days: int) -> float:
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("收益序列为空或包含非法值")
    return float(np.expm1(np.log1p(values).mean() * trading_days))


def summarize_period(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    *,
    period_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    objective: dict[str, Any],
) -> dict[str, Any]:
    daily = ledger[ledger["date"].between(start_date, end_date)].copy()
    if daily.empty:
        raise ValueError(f"时期{period_id}没有策略账本")
    merged = daily.merge(benchmark_returns, on="date", how="left", validate="one_to_one")
    if merged["benchmark_return"].isna().any():
        raise ValueError(f"时期{period_id}缺少基准收益")
    trading_days = int(objective["annualization_trading_days"])
    strategy_cagr = _geometric_annual_return(merged["daily_return"], trading_days)
    benchmark_cagr = _geometric_annual_return(merged["benchmark_return"], trading_days)
    annualized_excess = strategy_cagr - benchmark_cagr
    window = int(objective["rolling_window_trading_days"])
    strategy_rolling = np.expm1(
        np.log1p(merged["daily_return"].astype(float)).rolling(window).sum()
        * trading_days
        / window
    )
    benchmark_rolling = np.expm1(
        np.log1p(merged["benchmark_return"].astype(float)).rolling(window).sum()
        * trading_days
        / window
    )
    rolling_excess = (strategy_rolling - benchmark_rolling).dropna()
    rolling_median = None if rolling_excess.empty else float(rolling_excess.median())
    normalized = merged["equity"] / float(merged["equity"].iloc[0])
    drawdown = normalized / normalized.cummax() - 1.0
    period_trades = trades
    if not trades.empty:
        period_trades = trades[trades["date"].between(start_date, end_date)]
    execution_cost = 0.0
    if not period_trades.empty:
        execution_cost = float(
            period_trades["commission"].sum() + period_trades["slippage_cost"].sum()
        )
    annual_target = float(objective["minimum_annualized_excess"])
    rolling_target = float(objective["minimum_rolling_excess_median"])
    both_gate = bool(
        annualized_excess >= annual_target
        and rolling_median is not None
        and rolling_median >= rolling_target
    )
    return {
        "period_id": period_id,
        "actual_start_date": pd.Timestamp(merged["date"].iloc[0]).date().isoformat(),
        "actual_end_date": pd.Timestamp(merged["date"].iloc[-1]).date().isoformat(),
        "observations": int(len(merged)),
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": annualized_excess,
        "rolling_242d_excess_median": rolling_median,
        "rolling_242d_excess_minimum": None
        if rolling_excess.empty
        else float(rolling_excess.min()),
        "rolling_242d_target_hit_ratio": None
        if rolling_excess.empty
        else float((rolling_excess >= rolling_target).mean()),
        "maximum_drawdown": float(drawdown.min()),
        "average_actual_state": float(merged["actual_state"].mean()),
        "trade_leg_count": int(len(period_trades)),
        "execution_cost_cny": execution_cost,
        "t_plus_one_blocked_exit_attempts": int(merged["blocked_t1_exit_count"].sum()),
        "annualized_excess_gate": bool(annualized_excess >= annual_target),
        "rolling_median_gate": bool(
            rolling_median is not None and rolling_median >= rolling_target
        ),
        "both_20pct_gates": both_gate,
    }


def build_benchmark_returns(benchmark: pd.DataFrame) -> pd.DataFrame:
    result = benchmark[["date", "close"]].copy()
    result["benchmark_return"] = result["close"].pct_change(fill_method=None)
    return result[["date", "benchmark_return"]]


def trading_date_with_offset(
    dates: list[pd.Timestamp], base_start: pd.Timestamp, offset: int
) -> pd.Timestamp:
    eligible = [date for date in dates if date >= base_start]
    if offset < 0 or offset >= len(eligible):
        raise ValueError("起点扰动超出交易日范围")
    return pd.Timestamp(eligible[offset])


def evaluate_all(
    config: dict[str, Any],
    bars: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    evaluation = config["evaluation"]
    objective = config["objective"]
    initial_capital = float(config["scope"]["initial_capital_cny"])
    all_dates = sorted(pd.Timestamp(value) for value in bars["trade_date"].unique())
    base_start = pd.Timestamp(evaluation["base_start_date"])
    end_date = pd.Timestamp(evaluation["end_date"])
    periods = evaluation["periods"]
    benchmark_returns = build_benchmark_returns(benchmark)
    period_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    state_map: dict[str, np.ndarray] = {}

    for candidate in config["candidate_family"]["candidates"]:
        candidate_id = candidate["id"]
        target_states = build_candidate_states(bars, candidate)
        state_map[candidate_id] = target_states
        for scenario in evaluation["cost_scenarios"]:
            model = cost_model(config, scenario)
            for offset in evaluation["start_date_perturbations_trading_days"]:
                simulation_start = trading_date_with_offset(all_dates, base_start, int(offset))
                ledger, trades, diagnostics = simulate_candidate(
                    bars,
                    dividends,
                    target_states,
                    start_date=simulation_start,
                    end_date=end_date,
                    costs=model,
                    initial_capital=initial_capital,
                )
                diagnostics_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario": scenario,
                        "start_offset": int(offset),
                        **diagnostics,
                    }
                )
                for period in periods:
                    requested_start = max(pd.Timestamp(period["start_date"]), simulation_start)
                    summary = summarize_period(
                        ledger,
                        trades,
                        benchmark_returns,
                        period_id=period["id"],
                        start_date=requested_start,
                        end_date=pd.Timestamp(period["end_date"]),
                        objective=objective,
                    )
                    period_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "candidate_type": candidate["type"],
                            "scenario": scenario,
                            "start_offset": int(offset),
                            "simulation_start_date": simulation_start.date().isoformat(),
                            **summary,
                        }
                    )

    results = pd.DataFrame(period_rows)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    return results, diagnostics_frame, state_map


def summarize_candidates(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate_frame in results.groupby("candidate_id", sort=True):
        stress = candidate_frame[candidate_frame["scenario"] == "STRESS"]
        base = candidate_frame[candidate_frame["scenario"] == "BASE"]
        if len(stress) != 15 or len(base) != 15:
            raise ValueError(f"候选{candidate_id}未形成固定的30条时期结果")
        if stress["rolling_242d_excess_median"].isna().any():
            raise ValueError(f"候选{candidate_id}存在不足242日的冻结时期")
        stress_min_annual = float(stress["annualized_excess"].min())
        stress_min_rolling = float(stress["rolling_242d_excess_median"].min())
        rank_score = min(stress_min_annual, stress_min_rolling)
        hard_pass = bool(stress["both_20pct_gates"].all())
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            and (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        stress_combined_zero = stress[
            (stress["period_id"] == "COMBINED") & (stress["start_offset"] == 0)
        ].iloc[0]
        base_combined_zero = base[
            (base["period_id"] == "COMBINED") & (base["start_offset"] == 0)
        ].iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": stress["candidate_type"].iloc[0],
                "hard_pass": hard_pass,
                "positive_stress_shadow": positive_shadow,
                "rank_score": rank_score,
                "stress_min_annualized_excess": stress_min_annual,
                "stress_min_rolling_242d_excess_median": stress_min_rolling,
                "stress_combined_annualized_excess_offset0": float(
                    stress_combined_zero["annualized_excess"]
                ),
                "stress_combined_rolling_median_offset0": float(
                    stress_combined_zero["rolling_242d_excess_median"]
                ),
                "base_combined_annualized_excess_offset0": float(
                    base_combined_zero["annualized_excess"]
                ),
                "base_combined_rolling_median_offset0": float(
                    base_combined_zero["rolling_242d_excess_median"]
                ),
                "stress_trade_leg_count_offset0": int(stress_combined_zero["trade_leg_count"]),
                "stress_execution_cost_cny_offset0": float(
                    stress_combined_zero["execution_cost_cny"]
                ),
                "stress_average_actual_state_offset0": float(
                    stress_combined_zero["average_actual_state"]
                ),
                "stress_maximum_drawdown_offset0": float(
                    stress_combined_zero["maximum_drawdown"]
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.sort_values(
        [
            "hard_pass",
            "positive_stress_shadow",
            "rank_score",
            "stress_trade_leg_count_offset0",
            "stress_execution_cost_cny_offset0",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True, True],
        inplace=True,
    )
    summary.reset_index(drop=True, inplace=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"不可序列化类型：{type(value)!r}")


def _percent(value: float | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2%}"


def render_markdown(report: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# 510300 满仓/空仓日内利弗莫尔关键点筛选 V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 冻结候选：{len(summary)}个；硬通过：{int(summary['hard_pass'].sum())}个；压力成本全时期为正：{int(summary['positive_stress_shadow'].sum())}个。",
        "- 约束：任何时刻只允许510300满仓或现金空仓；信号后下一根15分钟柱开盘执行；强制T+1；50万元、100份整手。",
        "- 目标：压力成本下，前段、后段、合并段及五个起点的年化净超额和滚动242日超额中位数全部不低于20%。",
        "- 证据标签：全部历史此前已被查看，本报告只能用于否决或决定是否值得另行前向冻结。",
        "",
        "## 固定候选排名",
        "",
        "| 排名 | 候选 | 硬通过 | 全时期为正 | 压力最差年化超额 | 压力最差滚动中位数 | 合并压力年化超额 | 合并压力滚动中位数 | 压力交易腿 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {rank} | {candidate_id} | {hard} | {positive} | {min_a} | {min_r} | {comb_a} | {comb_r} | {legs} |".format(
                rank=int(row["rank"]),
                candidate_id=row["candidate_id"],
                hard="是" if row["hard_pass"] else "否",
                positive="是" if row["positive_stress_shadow"] else "否",
                min_a=_percent(row["stress_min_annualized_excess"]),
                min_r=_percent(row["stress_min_rolling_242d_excess_median"]),
                comb_a=_percent(row["stress_combined_annualized_excess_offset0"]),
                comb_r=_percent(row["stress_combined_rolling_median_offset0"]),
                legs=int(row["stress_trade_leg_count_offset0"]),
            )
        )
    top = summary.iloc[0]
    lines.extend(
        [
            "",
            "## 决策",
            "",
            f"- 排名第一：`{top['candidate_id']}`；压力最差年化超额 {_percent(float(top['stress_min_annualized_excess']))}，压力最差滚动中位数 {_percent(float(top['stress_min_rolling_242d_excess_median']))}。",
            f"- 结果：`{report['decision']['next_action']}`",
            "- 本筛选不生成Paper/Shadow动作、订单或实盘授权。",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    validate_protocol(config)
    freeze = verify_freeze_manifest(config)
    bars, benchmark, dividends, input_audit = load_and_audit_inputs(config)

    results, diagnostics, state_map = evaluate_all(config, bars, benchmark, dividends)
    summary = summarize_candidates(results)
    top = summary.iloc[0]
    hard_pass_count = int(summary["hard_pass"].sum())
    positive_count = int(summary["positive_stress_shadow"].sum())
    if hard_pass_count > 0:
        status = config["governance"]["result_label_if_passed"]
        next_action = "冻结排名第一的历史通过候选，另建不共享结果的新前向协议；当前仍不得交易"
    else:
        status = config["governance"]["result_label_if_rejected"]
        next_action = "保留本固定家族否决记录，不修改参数救援；转向尚未检验的正交信息源"

    artifacts = config["artifacts"]
    output_dir = ROOT / artifacts["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(ROOT / artifacts["period_results_parquet"], index=False)
    summary.to_parquet(ROOT / artifacts["candidate_summary_parquet"], index=False)

    top_states = state_map[str(top["candidate_id"])]
    top_ledger, top_trades, top_diagnostics = simulate_candidate(
        bars,
        dividends,
        top_states,
        start_date=pd.Timestamp(config["evaluation"]["base_start_date"]),
        end_date=pd.Timestamp(config["evaluation"]["end_date"]),
        costs=cost_model(config, "STRESS"),
        initial_capital=float(config["scope"]["initial_capital_cny"]),
    )
    top_ledger.to_parquet(ROOT / artifacts["top_candidate_daily_ledger_parquet"], index=False)
    top_trades.to_parquet(ROOT / artifacts["top_candidate_trades_parquet"], index=False)

    top_periods = results[
        (results["candidate_id"] == top["candidate_id"])
        & (results["scenario"] == "STRESS")
        & (results["start_offset"] == 0)
    ].to_dict("records")
    report = {
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(),
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "historical_evidence_label": config["protocol"]["historical_evidence_label"],
        "freeze": freeze,
        "scope": config["scope"],
        "objective": config["objective"],
        "costs": config["costs"],
        "input_audit": input_audit,
        "evaluation": config["evaluation"],
        "candidate_definitions": config["candidate_family"]["candidates"],
        "candidate_count": int(len(summary)),
        "hard_pass_count": hard_pass_count,
        "positive_stress_shadow_count": positive_count,
        "ranking": summary.to_dict("records"),
        "top_candidate": {
            "candidate_id": str(top["candidate_id"]),
            "summary": top.to_dict(),
            "stress_offset0_periods": top_periods,
            "execution_diagnostics": top_diagnostics,
        },
        "execution_diagnostics": diagnostics.to_dict("records"),
        "decision": {
            "hard_target_achieved": bool(hard_pass_count > 0),
            "eligible_for_composition": bool(positive_count > 0),
            "next_action": next_action,
            "paper_or_shadow_authorized": False,
            "live_trading_authorized": False,
        },
        "artifact_hashes": {},
    }

    report_path = ROOT / artifacts["report_json"]
    markdown_path = ROOT / artifacts["report_markdown"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report, summary), encoding="utf-8")

    artifact_paths = [
        artifacts["candidate_summary_parquet"],
        artifacts["period_results_parquet"],
        artifacts["top_candidate_daily_ledger_parquet"],
        artifacts["top_candidate_trades_parquet"],
        artifacts["report_markdown"],
    ]
    report["artifact_hashes"] = {
        relative: sha256_file(ROOT / relative) for relative in artifact_paths
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "状态": status,
                "硬通过候选数": hard_pass_count,
                "压力成本全时期为正候选数": positive_count,
                "排名第一": str(top["candidate_id"]),
                "压力最差年化超额": float(top["stress_min_annualized_excess"]),
                "压力最差滚动中位数": float(
                    top["stress_min_rolling_242d_excess_median"]
                ),
                "报告": artifacts["report_json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行510300满仓/空仓日内关键点固定筛选")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
