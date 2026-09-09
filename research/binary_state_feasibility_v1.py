"""510300 满仓/空仓二元状态的完美方向可行性前沿。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_binary_state_feasibility_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_binary_state_feasibility_v1_manifest.json"


@dataclass(frozen=True)
class CostModel:
    """二元组合的逐腿成本。"""

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
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    objective = config["objective"]
    frontier = config["oracle_frontier"]
    if protocol["study_id"] != "510300_BINARY_STATE_FEASIBILITY_V1":
        raise ValueError("研究编号不匹配")
    if protocol["is_strategy_candidate"] or protocol["is_trading_signal"]:
        raise ValueError("可行性前沿不得伪装成策略或交易信号")
    if protocol["live_trading_authorized"]:
        raise ValueError("本研究不得授权实盘")
    if scope["execution_asset"] != "510300.SH":
        raise ValueError("唯一执行资产必须是510300.SH")
    if list(scope["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("允许持有资产必须严格为510300与现金")
    if list(scope["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if scope["leverage_allowed"] or scope["short_selling_allowed"] or scope["derivatives_allowed"]:
        raise ValueError("禁止杠杆、卖空和衍生品")
    if objective["benchmark_id"] != "H00300":
        raise ValueError("主基准必须是H00300")
    if float(objective["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额目标必须固定为20个百分点")
    if float(objective["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动超额中位数目标必须固定为20个百分点")
    horizons = [int(value) for value in frontier["block_horizons_trading_days"]]
    if horizons != [1, 5, 20, 60, 121, 242]:
        raise ValueError("神谕周期集合偏离冻结定义")
    if frontier["final_incomplete_block_policy"] != "EXCLUDE":
        raise ValueError("末尾不完整块必须排除")
    accuracy = frontier["accuracy_frontier"]
    if int(accuracy["canonical_horizon_trading_days"]) != 20:
        raise ValueError("精度前沿必须固定使用20日块")
    if accuracy["cost_scenario"] != "STRESS":
        raise ValueError("精度前沿必须使用压力成本")


def validate_manifest(root: Path = ROOT, path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ValueError("冻结清单不存在，禁止读取历史收益")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("study_id") != "510300_BINARY_STATE_FEASIBILITY_V1":
        raise ValueError("冻结清单研究编号不匹配")
    if not manifest.get("implementation_frozen", False):
        raise ValueError("实现尚未冻结，禁止读取历史收益")
    mismatches: list[str] = []
    for group in ("tracked_files", "data_files"):
        for relative, expected in manifest[group].items():
            file_path = root / relative
            actual = sha256_file(file_path) if file_path.exists() else None
            if actual != expected:
                mismatches.append(relative)
    if mismatches:
        raise ValueError(f"冻结清单哈希不一致：{mismatches}")
    return manifest


def _normalize_date(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result.sort_values(column).reset_index(drop=True)


def _audit_market_contract(frame: pd.DataFrame, contract: dict[str, Any], *, price_columns: Iterable[str]) -> dict[str, Any]:
    missing = sorted(set(contract["required_columns"]).difference(frame.columns))
    dates = pd.to_datetime(frame["date"], errors="coerce")
    symbols = sorted(str(value) for value in frame["symbol"].dropna().unique())
    invalid_prices = 0
    for column in price_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_prices += int((values.isna() | (values <= 0.0)).sum())
    checks = {
        "rows": int(len(frame)),
        "required_rows": int(contract["required_rows"]),
        "first_date": None if dates.empty else dates.min().date().isoformat(),
        "last_date": None if dates.empty else dates.max().date().isoformat(),
        "duplicate_dates": int(dates.duplicated().sum()),
        "symbols": symbols,
        "missing_columns": missing,
        "invalid_prices": int(invalid_prices),
    }
    checks["status"] = "PASS" if all(
        [
            checks["rows"] == checks["required_rows"],
            checks["first_date"] == contract["required_first_date"],
            checks["last_date"] == contract["required_last_date"],
            checks["duplicate_dates"] == 0,
            checks["symbols"] == [contract["required_symbol"]],
            not checks["missing_columns"],
            checks["invalid_prices"] == 0,
        ]
    ) else "FAIL"
    return checks


def load_and_audit_inputs(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contracts = config["data_contracts"]
    etf_path = root / contracts["etf_market"]["file"]
    benchmark_path = root / contracts["benchmark_total_return"]["file"]
    dividend_path = root / contracts["cash_distributions"]["file"]

    etf = _normalize_date(pd.read_parquet(etf_path), "date")
    benchmark = _normalize_date(pd.read_parquet(benchmark_path), "date")
    dividends = pd.read_csv(dividend_path)
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)

    etf_audit = _audit_market_contract(
        etf, contracts["etf_market"], price_columns=("open", "high", "low", "close")
    )
    benchmark_audit = _audit_market_contract(
        benchmark, contracts["benchmark_total_return"], price_columns=("close",)
    )
    dividend_contract = contracts["cash_distributions"]
    dividend_audit = {
        "event_count": int(len(dividends)),
        "first_ex_date": dividends["ex_date"].min().date().isoformat(),
        "last_ex_date": dividends["ex_date"].max().date().isoformat(),
        "duplicate_ex_dates": int(dividends["ex_date"].duplicated().sum()),
        "nonpositive_cash_dividends": int((dividends["cash_dividend_per_share"] <= 0.0).sum()),
        "missing_columns": sorted(set(dividend_contract["required_columns"]).difference(dividends.columns)),
    }
    dividend_audit["status"] = "PASS" if all(
        [
            dividend_audit["event_count"] == int(dividend_contract["required_event_count"]),
            dividend_audit["first_ex_date"] == dividend_contract["required_first_ex_date"],
            dividend_audit["last_ex_date"] == dividend_contract["required_last_ex_date"],
            dividend_audit["duplicate_ex_dates"] == 0,
            dividend_audit["nonpositive_cash_dividends"] == 0,
            not dividend_audit["missing_columns"],
        ]
    ) else "FAIL"

    etf_dates = set(etf["date"])
    benchmark_dates = set(benchmark["date"])
    allowed_extra = {
        pd.Timestamp(value) for value in contracts["benchmark_total_return"]["allowed_non_execution_dates"]
    }
    missing_benchmark = sorted(value.date().isoformat() for value in etf_dates.difference(benchmark_dates))
    extra_benchmark = benchmark_dates.difference(etf_dates)
    calendar_audit = {
        "missing_benchmark_execution_dates": missing_benchmark,
        "extra_benchmark_dates": sorted(value.date().isoformat() for value in extra_benchmark),
        "extra_dates_exactly_allowed": extra_benchmark == allowed_extra,
    }
    calendar_audit["status"] = "PASS" if not missing_benchmark and calendar_audit["extra_dates_exactly_allowed"] else "FAIL"

    merged = etf[["date", "open", "high", "low", "close"]].rename(
        columns={
            "open": "etf_open",
            "high": "etf_high",
            "low": "etf_low",
            "close": "etf_close",
        }
    ).merge(
        benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    merged_invalid = int(merged[["etf_open", "etf_high", "etf_low", "etf_close", "benchmark_close"]].isna().any(axis=1).sum())
    audit = {
        "status": "PASS" if all(
            item["status"] == "PASS"
            for item in (etf_audit, benchmark_audit, dividend_audit, calendar_audit)
        ) and merged_invalid == 0 else "NO_VIEW_DATA_GATE_FAILED",
        "historical_cutoff": config["protocol"]["historical_cutoff"],
        "merged_rows": int(len(merged)),
        "merged_invalid_rows": merged_invalid,
        "etf_market": etf_audit,
        "benchmark_total_return": benchmark_audit,
        "cash_distributions": dividend_audit,
        "calendar": calendar_audit,
        "input_hashes": {
            contracts["etf_market"]["file"]: sha256_file(etf_path),
            contracts["benchmark_total_return"]["file"]: sha256_file(benchmark_path),
            contracts["cash_distributions"]["file"]: sha256_file(dividend_path),
        },
    }
    if audit["status"] != "PASS":
        raise ValueError(f"数据门失败：{json.dumps(audit, ensure_ascii=False)}")
    return merged.reset_index(drop=True), dividends, audit


def _dividend_factor_for_block(dividends: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    eligible = dividends.loc[(dividends["ex_date"] > start) & (dividends["ex_date"] <= end)]
    return float(eligible["cash_dividend_per_share"].sum())


def build_perfect_block_states(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    horizon: int,
    cash_annual_rate: float,
    trading_days_per_year: int,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """为完整非重叠块生成完美方向状态；第0行只作为前一收盘信号日。"""

    if horizon <= 0:
        raise ValueError("决策周期必须为正整数")
    if len(market) <= horizon:
        raise ValueError("市场数据不足一个完整决策块及前一信号日")
    states = np.zeros(len(market), dtype=np.int8)
    blocks: list[dict[str, Any]] = []
    last_end = 0
    for start_index in range(1, len(market), horizon):
        end_index = start_index + horizon - 1
        if end_index >= len(market):
            break
        start_date = pd.Timestamp(market.loc[start_index, "date"])
        end_date = pd.Timestamp(market.loc[end_index, "date"])
        dividend_per_share = _dividend_factor_for_block(dividends, start_date, end_date)
        full_factor = (
            float(market.loc[end_index, "etf_close"]) + dividend_per_share
        ) / float(market.loc[start_index, "etf_open"])
        cash_factor = (1.0 + cash_annual_rate / trading_days_per_year) ** horizon
        target_state = int(full_factor > cash_factor)
        states[start_index : end_index + 1] = target_state
        blocks.append(
            {
                "block_index": int(len(blocks)),
                "start_index": int(start_index),
                "end_index": int(end_index),
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": int(horizon),
                "etf_open": float(market.loc[start_index, "etf_open"]),
                "etf_end_close": float(market.loc[end_index, "etf_close"]),
                "cash_dividend_per_share": dividend_per_share,
                "full_gross_factor": float(full_factor),
                "cash_gross_factor": float(cash_factor),
                "oracle_state": target_state,
                "bad_block": bool(target_state == 0),
            }
        )
        last_end = end_index
    if not blocks:
        raise ValueError("没有形成完整决策块")
    return states[: last_end + 1], pd.DataFrame(blocks), last_end


def _event_maps(dividends: pd.DataFrame) -> tuple[dict[pd.Timestamp, list[dict[str, Any]]], dict[pd.Timestamp, float]]:
    ex_map: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    payment_map: dict[pd.Timestamp, float] = {}
    for record in dividends.to_dict("records"):
        ex_date = pd.Timestamp(record["ex_date"])
        payment_date = pd.Timestamp(record["payment_date"])
        ex_map.setdefault(ex_date, []).append(record)
        payment_map[payment_date] = payment_map.get(payment_date, 0.0) + float(record["cash_dividend_per_share"])
    return ex_map, payment_map


def simulate_binary_path(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    desired_states: np.ndarray,
    *,
    costs: CostModel,
    initial_capital: float,
    reinvest_paid_dividends: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在每个交易日开盘把组合调整到0/1状态，现金分红最早下一开盘再投资。"""

    data = market.reset_index(drop=True).copy()
    if len(data) != len(desired_states):
        raise ValueError("目标状态长度必须与市场数据一致")
    unique_states = sorted(int(value) for value in np.unique(desired_states))
    if not set(unique_states).issubset({0, 1}):
        raise ValueError("目标状态只能为0或1")
    if int(desired_states[0]) != 0:
        raise ValueError("首行是前一收盘信号日，必须从现金开始")

    ex_map, _ = _event_maps(dividends)
    payment_schedule: dict[pd.Timestamp, float] = {}
    cash = float(initial_capital)
    shares = 0.0
    receivable = 0.0
    ledger_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    previous_date: pd.Timestamp | None = None

    for index, row in data.iterrows():
        date = pd.Timestamp(row["date"])
        if previous_date is not None:
            cash *= 1.0 + costs.cash_annual_rate / costs.trading_days_per_year

        shares_at_start = shares
        entitlement_today = 0.0
        for event in ex_map.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            entitlement_today += entitlement
            payment_date = pd.Timestamp(event["payment_date"])
            payment_schedule[payment_date] = payment_schedule.get(payment_date, 0.0) + entitlement
        receivable += entitlement_today

        target_state = int(desired_states[index])
        raw_open = float(row["etf_open"])
        side = "NONE"
        trade_reason = "NO_TRADE"
        quantity = 0.0
        execution_price = np.nan
        execution_notional = 0.0
        commission = 0.0
        slippage_cost = 0.0

        if target_state == 0 and shares > 0.0:
            side = "SELL"
            trade_reason = "STATE_EXIT_TO_CASH"
            quantity = shares
            execution_price = raw_open * (1.0 - costs.slippage_bps / 10000.0)
            execution_notional = quantity * execution_price
            commission = costs.commission(execution_notional)
            cash += execution_notional - commission
            shares = 0.0
            slippage_cost = quantity * (raw_open - execution_price)
        elif target_state == 1:
            can_reinvest = shares == 0.0 or reinvest_paid_dividends
            if can_reinvest:
                execution_price = raw_open * (1.0 + costs.slippage_bps / 10000.0)
                affordable = int(cash / (execution_price * (1.0 + costs.commission_rate)) / costs.lot_size)
                quantity = float(max(affordable, 0) * costs.lot_size)
                while quantity > 0.0:
                    execution_notional = quantity * execution_price
                    commission = costs.commission(execution_notional)
                    if execution_notional + commission <= cash + 1e-9:
                        break
                    quantity -= costs.lot_size
                if quantity > 0.0:
                    side = "BUY"
                    trade_reason = "STATE_ENTRY_FULL" if shares == 0.0 else "FULL_STATE_CASH_REINVESTMENT"
                    cash -= execution_notional + commission
                    shares += quantity
                    slippage_cost = quantity * (execution_price - raw_open)

        if side in {"BUY", "SELL"}:
            trade_rows.append(
                {
                    "date": date,
                    "side": side,
                    "trade_reason": trade_reason,
                    "target_state": target_state,
                    "raw_open": raw_open,
                    "execution_price": float(execution_price),
                    "quantity": float(quantity),
                    "execution_notional": float(execution_notional),
                    "commission": float(commission),
                    "slippage_cost": float(slippage_cost),
                    "shares_before_trade": float(shares_at_start),
                    "shares_after_trade": float(shares),
                }
            )

        payment_today = float(payment_schedule.pop(date, 0.0))
        if payment_today:
            cash += payment_today
            receivable -= payment_today
            if abs(receivable) < 1e-9:
                receivable = 0.0
        equity = cash + shares * float(row["etf_close"]) + receivable
        ledger_rows.append(
            {
                "date": date,
                "policy_state": target_state,
                "shares": float(shares),
                "cash": float(cash),
                "dividend_receivable": float(receivable),
                "dividend_entitlement_today": float(entitlement_today),
                "dividend_payment_today": float(payment_today),
                "etf_open": raw_open,
                "etf_close": float(row["etf_close"]),
                "equity": float(equity),
                "daily_commission": float(commission),
                "daily_slippage_cost": float(slippage_cost),
            }
        )
        previous_date = date

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    return ledger, pd.DataFrame(trade_rows)


def build_benchmark_ledger(market: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    result = market[["date", "benchmark_close"]].reset_index(drop=True).copy()
    result["equity"] = initial_capital * result["benchmark_close"] / float(result.loc[0, "benchmark_close"])
    result["daily_return"] = result["equity"].pct_change().fillna(0.0)
    result["equity_peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["equity_peak"] - 1.0
    return result


def geometric_cagr(returns: pd.Series, trading_days_per_year: int) -> float:
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if len(values) <= 1 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("收益序列不足或包含非法值")
    return float(np.expm1(np.log1p(values[1:]).mean() * trading_days_per_year))


def rolling_annualized_excess(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int,
    trading_days_per_year: int,
) -> pd.Series:
    left = np.log1p(strategy_returns.astype(float))
    right = np.log1p(benchmark_returns.astype(float))
    strategy_annual = np.expm1(left.rolling(window).mean() * trading_days_per_year)
    benchmark_annual = np.expm1(right.rolling(window).mean() * trading_days_per_year)
    return strategy_annual - benchmark_annual


def summarize_path(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    objective: dict[str, Any],
) -> dict[str, Any]:
    merged = ledger[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_benchmark"),
        validate="one_to_one",
    )
    trading_days = int(objective["annualization_trading_days"])
    strategy_cagr = geometric_cagr(merged["daily_return_strategy"], trading_days)
    benchmark_cagr = geometric_cagr(merged["daily_return_benchmark"], trading_days)
    annualized_excess = strategy_cagr - benchmark_cagr
    rolling = rolling_annualized_excess(
        merged["daily_return_strategy"],
        merged["daily_return_benchmark"],
        window=int(objective["rolling_window_trading_days"]),
        trading_days_per_year=trading_days,
    ).dropna()
    rolling_median = None if rolling.empty else float(rolling.median())
    executed = trades if not trades.empty else pd.DataFrame()
    total_cost = 0.0
    if not executed.empty:
        total_cost = float(executed["commission"].sum() + executed["slippage_cost"].sum())
    state_changes = int((ledger["policy_state"].diff().fillna(0.0) != 0.0).sum())
    target = float(objective["minimum_annualized_excess"])
    rolling_target = float(objective["minimum_rolling_excess_median"])
    return {
        "start_date": ledger["date"].iloc[0].date().isoformat(),
        "end_date": ledger["date"].iloc[-1].date().isoformat(),
        "observations": int(len(ledger)),
        "strategy_cagr": float(strategy_cagr),
        "benchmark_cagr": float(benchmark_cagr),
        "annualized_excess": float(annualized_excess),
        "rolling_242d_excess_median": rolling_median,
        "rolling_242d_excess_minimum": None if rolling.empty else float(rolling.min()),
        "rolling_242d_target_hit_ratio": None if rolling.empty else float((rolling >= rolling_target).mean()),
        "maximum_drawdown": float(ledger["drawdown"].min()),
        "average_policy_state": float(ledger["policy_state"].mean()),
        "state_change_count": state_changes,
        "trade_leg_count": int(len(executed)),
        "total_execution_cost_cny": total_cost,
        "annualized_excess_gate": bool(annualized_excess >= target),
        "rolling_median_gate": bool(rolling_median is not None and rolling_median >= rolling_target),
        "both_20pct_gates": bool(
            annualized_excess >= target and rolling_median is not None and rolling_median >= rolling_target
        ),
    }


def _cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    costs = config["costs"]
    if scenario not in {"BASE", "STRESS"}:
        raise ValueError("成本情景必须为BASE或STRESS")
    slippage = costs["base_slippage_bps_per_leg"] if scenario == "BASE" else costs["stress_slippage_bps_per_leg"]
    return CostModel(
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(slippage),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def evaluate_oracle_offsets(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    horizons = [int(value) for value in config["oracle_frontier"]["block_horizons_trading_days"]]
    initial_capital = float(config["scope"]["initial_capital_cny"])
    reinvest = bool(config["scope"]["full_state_reinvests_paid_cash_dividends"])
    base_costs = _cost_model(config, "BASE")
    stress_costs = _cost_model(config, "STRESS")
    rows: list[dict[str, Any]] = []
    canonical_blocks = pd.DataFrame()

    for horizon in horizons:
        for offset in range(horizon):
            shifted = market.iloc[offset:].reset_index(drop=True)
            if len(shifted) <= horizon:
                continue
            states, blocks, last_end = build_perfect_block_states(
                shifted,
                dividends,
                horizon=horizon,
                cash_annual_rate=float(config["costs"]["cash_annual_rate"]),
                trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
            )
            sample = shifted.iloc[: last_end + 1].reset_index(drop=True)
            benchmark = build_benchmark_ledger(sample, initial_capital)
            base_ledger, base_trades = simulate_binary_path(
                sample,
                dividends,
                states,
                costs=base_costs,
                initial_capital=initial_capital,
                reinvest_paid_dividends=reinvest,
            )
            stress_ledger, stress_trades = simulate_binary_path(
                sample,
                dividends,
                states,
                costs=stress_costs,
                initial_capital=initial_capital,
                reinvest_paid_dividends=reinvest,
            )
            base_summary = summarize_path(base_ledger, base_trades, benchmark, objective=config["objective"])
            stress_summary = summarize_path(stress_ledger, stress_trades, benchmark, objective=config["objective"])
            row: dict[str, Any] = {
                "horizon_trading_days": horizon,
                "calendar_offset": offset,
                "block_count": int(len(blocks)),
                "bad_block_count": int((blocks["oracle_state"] == 0).sum()),
                "full_block_count": int((blocks["oracle_state"] == 1).sum()),
                "sample_start_date": sample["date"].iloc[0],
                "sample_end_date": sample["date"].iloc[-1],
            }
            for prefix, summary in (("base", base_summary), ("stress", stress_summary)):
                for key, value in summary.items():
                    if key not in {"start_date", "end_date"}:
                        row[f"{prefix}_{key}"] = value
            rows.append(row)
            if horizon == 20 and offset == 0:
                canonical_blocks = blocks.copy()

    offset_frame = pd.DataFrame(rows)
    if offset_frame.empty:
        raise ValueError("神谕错位评价没有产生任何结果")
    summaries: dict[str, Any] = {}
    for horizon, group in offset_frame.groupby("horizon_trading_days", sort=True):
        stress_ranked = group.sort_values("stress_annualized_excess").reset_index(drop=True)
        representative = stress_ranked.iloc[len(stress_ranked) // 2]
        payload: dict[str, Any] = {
            "horizon_trading_days": int(horizon),
            "offset_count": int(len(group)),
            "block_count_median": float(group["block_count"].median()),
            "bad_block_count_median": float(group["bad_block_count"].median()),
            "full_block_count_median": float(group["full_block_count"].median()),
            "base_annualized_excess": _distribution_summary(group["base_annualized_excess"]),
            "stress_annualized_excess": _distribution_summary(group["stress_annualized_excess"]),
            "base_rolling_242d_excess_median": _distribution_summary(group["base_rolling_242d_excess_median"]),
            "stress_rolling_242d_excess_median": _distribution_summary(group["stress_rolling_242d_excess_median"]),
            "base_both_20pct_gate_offset_ratio": float(group["base_both_20pct_gates"].mean()),
            "stress_both_20pct_gate_offset_ratio": float(group["stress_both_20pct_gates"].mean()),
            "stress_average_policy_state_median": float(group["stress_average_policy_state"].median()),
            "stress_state_change_count_median": float(group["stress_state_change_count"].median()),
            "representative_median_stress_offset": int(representative["calendar_offset"]),
            "representative_median_stress_annualized_excess": float(representative["stress_annualized_excess"]),
            "representative_median_stress_rolling_242d_excess": float(representative["stress_rolling_242d_excess_median"]),
        }
        summaries[str(int(horizon))] = payload
    return offset_frame, summaries, canonical_blocks


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"observations": 0}
    return {
        "observations": int(len(numeric)),
        "minimum": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.90)),
        "maximum": float(numeric.max()),
    }


def _states_from_predictions(length: int, blocks: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    if len(blocks) != len(predictions):
        raise ValueError("预测数量必须与决策块数量一致")
    states = np.zeros(length, dtype=np.int8)
    for block, prediction in zip(blocks.itertuples(index=False), predictions, strict=True):
        states[int(block.start_index) : int(block.end_index) + 1] = int(prediction)
    return states


def evaluate_accuracy_frontier(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    canonical_blocks: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    accuracy = config["oracle_frontier"]["accuracy_frontier"]
    horizon = int(accuracy["canonical_horizon_trading_days"])
    if canonical_blocks.empty or not (canonical_blocks["trading_days"] == horizon).all():
        raise ValueError("固定20日决策块缺失或周期不匹配")
    last_end = int(canonical_blocks["end_index"].max())
    sample = market.iloc[: last_end + 1].reset_index(drop=True)
    benchmark = build_benchmark_ledger(sample, float(config["scope"]["initial_capital_cny"]))
    stress_costs = _cost_model(config, "STRESS")
    rng = np.random.default_rng(int(accuracy["random_seed"]))
    raw_rows: list[dict[str, Any]] = []
    true_states = canonical_blocks["oracle_state"].to_numpy(dtype=np.int8)
    bad_mask = true_states == 0
    good_mask = true_states == 1

    for recall in [float(value) for value in accuracy["bad_block_recall_grid"]]:
        for false_exit in [float(value) for value in accuracy["false_exit_rate_grid"]]:
            for repetition in range(int(accuracy["repetitions"])):
                predictions = np.ones(len(true_states), dtype=np.int8)
                predictions[bad_mask] = (rng.random(int(bad_mask.sum())) >= recall).astype(np.int8)
                predictions[good_mask] = (rng.random(int(good_mask.sum())) >= false_exit).astype(np.int8)
                states = _states_from_predictions(len(sample), canonical_blocks, predictions)
                ledger, trades = simulate_binary_path(
                    sample,
                    dividends,
                    states,
                    costs=stress_costs,
                    initial_capital=float(config["scope"]["initial_capital_cny"]),
                    reinvest_paid_dividends=bool(config["scope"]["full_state_reinvests_paid_cash_dividends"]),
                )
                summary = summarize_path(ledger, trades, benchmark, objective=config["objective"])
                raw_rows.append(
                    {
                        "bad_block_recall": recall,
                        "false_exit_rate": false_exit,
                        "repetition": repetition,
                        "realized_bad_block_recall": float((predictions[bad_mask] == 0).mean()) if bad_mask.any() else np.nan,
                        "realized_false_exit_rate": float((predictions[good_mask] == 0).mean()) if good_mask.any() else np.nan,
                        "annualized_excess": summary["annualized_excess"],
                        "rolling_242d_excess_median": summary["rolling_242d_excess_median"],
                        "both_20pct_gates": summary["both_20pct_gates"],
                        "average_policy_state": summary["average_policy_state"],
                        "state_change_count": summary["state_change_count"],
                    }
                )
    raw = pd.DataFrame(raw_rows)
    aggregates: list[dict[str, Any]] = []
    for (recall, false_exit), group in raw.groupby(["bad_block_recall", "false_exit_rate"], sort=True):
        aggregates.append(
            {
                "bad_block_recall": float(recall),
                "false_exit_rate": float(false_exit),
                "repetitions": int(len(group)),
                "annualized_excess": _distribution_summary(group["annualized_excess"]),
                "rolling_242d_excess_median": _distribution_summary(group["rolling_242d_excess_median"]),
                "both_20pct_gate_success_ratio": float(group["both_20pct_gates"].mean()),
                "average_policy_state_median": float(group["average_policy_state"].median()),
                "state_change_count_median": float(group["state_change_count"].median()),
            }
        )

    required_by_false_exit: dict[str, float | None] = {}
    target = float(config["objective"]["minimum_annualized_excess"])
    rolling_target = float(config["objective"]["minimum_rolling_excess_median"])
    for false_exit in sorted(raw["false_exit_rate"].unique()):
        required: float | None = None
        for item in aggregates:
            if item["false_exit_rate"] != float(false_exit):
                continue
            if (
                item["annualized_excess"].get("median", -np.inf) >= target
                and item["rolling_242d_excess_median"].get("median", -np.inf) >= rolling_target
            ):
                required = float(item["bad_block_recall"])
                break
        required_by_false_exit[f"{float(false_exit):.2f}"] = required
    frontier = {
        "canonical_horizon_trading_days": horizon,
        "block_count": int(len(canonical_blocks)),
        "bad_block_count": int(bad_mask.sum()),
        "good_block_count": int(good_mask.sum()),
        "required_bad_block_recall_for_median_both_20pct_gates_by_false_exit_rate": required_by_false_exit,
    }
    return raw, aggregates, frontier


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    data_audit: dict[str, Any],
    offset_results: pd.DataFrame,
    horizon_summaries: dict[str, Any],
    accuracy_aggregates: list[dict[str, Any]],
    accuracy_frontier: dict[str, Any],
) -> dict[str, Any]:
    feasible_horizons = [
        int(horizon)
        for horizon, payload in horizon_summaries.items()
        if payload["stress_annualized_excess"]["median"] >= float(config["objective"]["minimum_annualized_excess"])
        and payload["stress_rolling_242d_excess_median"]["median"] >= float(config["objective"]["minimum_rolling_excess_median"])
    ]
    if feasible_horizons:
        status = "BINARY_TARGET_MATHEMATICALLY_FEASIBLE_ONLY_WITH_PERFECT_FORESIGHT"
    else:
        status = "BINARY_TARGET_BELOW_PERFECT_BLOCK_DIRECTION_FRONTIER"
    best_row = offset_results.sort_values("stress_annualized_excess", ascending=False).iloc[0]
    return {
        "schema_version": "1.0.0",
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": status,
        "goal_achieved": False,
        "reason_goal_not_achieved": "本研究使用完美未来方向标签，只能证明数学空间，不能构成可实现策略证据",
        "historical_evidence_label": config["protocol"]["historical_evidence_label"],
        "manifest": {
            "path": "config/510300_binary_state_feasibility_v1_manifest.json",
            "sha256": sha256_file(MANIFEST_PATH),
            "frozen_at": manifest["frozen_at"],
        },
        "scope": config["scope"],
        "objective": config["objective"],
        "costs": config["costs"],
        "data_audit": data_audit,
        "horizon_summaries": horizon_summaries,
        "perfect_foresight_feasible_horizons": feasible_horizons,
        "best_single_calendar_offset_diagnostic": {
            "horizon_trading_days": int(best_row["horizon_trading_days"]),
            "calendar_offset": int(best_row["calendar_offset"]),
            "stress_annualized_excess": float(best_row["stress_annualized_excess"]),
            "stress_rolling_242d_excess_median": float(best_row["stress_rolling_242d_excess_median"]),
            "selection_warning": "仅为神谕上界诊断，不得作为候选周期或起点",
        },
        "accuracy_frontier": accuracy_frontier,
        "accuracy_grid": accuracy_aggregates,
        "governance": {
            "strategy_candidate_created": False,
            "current_state_signal_created": False,
            "position_mapping_enabled": False,
            "paper_or_shadow_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 510300 满仓/空仓二元状态可行性前沿 V1",
        "",
        f"- 状态：`{report['status']}`",
        "- 目标达成：`false`（完美未来方向只用于诊断，不能作为策略证据）",
        f"- 数据截止：`{report['data_audit']['historical_cutoff']}`",
        "- 仓位集合：`{0%, 100%}`",
        "- 主目标：基础与压力成本后相对 H00300 年化净超额及 242 日滚动超额中位数均不低于 20%。",
        "",
        "## 完美方向块神谕",
        "",
        "| 决策周期 | 错位数 | 压力年化超额中位数 | 压力滚动242日超额中位数 | 压力双门通过错位比例 | 平均满仓比例 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in sorted(report["horizon_summaries"], key=int):
        item = report["horizon_summaries"][horizon]
        lines.append(
            "| {h} | {n} | {annual:.2%} | {rolling:.2%} | {ratio:.2%} | {exposure:.2%} |".format(
                h=horizon,
                n=item["offset_count"],
                annual=item["stress_annualized_excess"]["median"],
                rolling=item["stress_rolling_242d_excess_median"]["median"],
                ratio=item["stress_both_20pct_gate_offset_ratio"],
                exposure=item["stress_average_policy_state_median"],
            )
        )
    feasible = report["perfect_foresight_feasible_horizons"]
    lines.extend(
        [
            "",
            "## 20日坏状态识别精度",
            "",
            "下表给出在不同误退出率下，使两项20%门槛的模拟中位数同时达标所需的最低坏块召回率；`未达到`表示即使网格中的100%召回也没有同时达标。",
            "",
            "| 误退出率 | 最低坏块召回率 |",
            "|---:|---:|",
        ]
    )
    required = report["accuracy_frontier"]["required_bad_block_recall_for_median_both_20pct_gates_by_false_exit_rate"]
    for false_exit, recall in required.items():
        recall_text = "未达到" if recall is None else f"{recall:.0%}"
        lines.append(f"| {float(false_exit):.0%} | {recall_text} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 完美方向下可跨多数日历错位同时达到两项20%门槛的周期：`{feasible}`。",
            "- 这不是候选收益；它只量化下一模型必须达到的召回率与误退出容忍度。",
            "- 本研究未生成当前满仓/空仓判断、Paper/Shadow、订单或实盘授权。",
        ]
    )
    return "\n".join(lines) + "\n"

