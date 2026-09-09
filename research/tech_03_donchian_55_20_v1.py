"""TECH_03 唐奇安55/20：冻结信号、双层账本与统计审计。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis, norm, skew


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "tech_03_donchian_55_20_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "tech_03_donchian_55_20_v1_manifest.json"


@dataclass(frozen=True)
class CostModel:
    """每腿成本；理论层的最低佣金为零。"""

    commission_rate: float
    minimum_commission: float
    slippage_bps: float
    cash_annual_rate: float
    trading_days_per_year: int
    lot_size: int | None
    minimum_trade_shares: int

    def commission(self, execution_notional: float) -> float:
        if execution_notional <= 0:
            return 0.0
        return max(self.minimum_commission, execution_notional * self.commission_rate)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol"]["project_id"] != "TECH_03_DONCHIAN_55_20_V1":
        raise ValueError("项目编号不匹配")
    variants = [
        (item["id"], item["entry_lookback_trading_days"], item["exit_lookback_trading_days"])
        for item in config["variants"]
    ]
    expected = [
        ("TECH_03_DONCHIAN_55_20_V1", 55, 20),
        ("TECH_03_DONCHIAN_100_50_ROBUSTNESS_V1", 100, 50),
    ]
    if variants != expected:
        raise ValueError("候选集合或窗口偏离冻结定义")
    registry = config["trial_registry"]
    if registry["registered_trial_count"] != 3:
        raise ValueError("累计试验数必须固定为3")
    if registry["return_tested_trial_count_after_run"] != 2:
        raise ValueError("本次收益试验数必须固定为2")
    if config["governance"]["parameter_search_allowed"]:
        raise ValueError("冻结协议禁止参数搜索")
    for switch in ("position_mapping_enabled", "order_generation_enabled", "broker_connection_enabled"):
        if config["governance"][switch]:
            raise ValueError(f"安全开关不得启用：{switch}")
    if config["protocol"]["true_forward_start"] is not None:
        raise ValueError("历史诊断阶段不得伪造真实样本外起点")


def validate_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_FILE.exists():
        raise ValueError("实现清单不存在，禁止历史收益计算")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ValueError("实现清单项目编号不匹配")
    if not manifest.get("implementation_frozen", False):
        raise ValueError("实现尚未冻结，禁止历史收益计算")
    mismatches: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        path = root / relative
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            mismatches.append(relative)
    for relative, expected in manifest["data_files"].items():
        path = root / relative
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"冻结哈希不匹配：{mismatches}")
    return manifest


def _normalize_dates(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result.sort_values(column).reset_index(drop=True)


def _audit_market_frame(
    root: Path,
    contract: dict[str, Any],
    symbol: str,
    require_ohlc: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / contract["file"]
    if not path.exists():
        raise ValueError(f"数据文件不存在：{contract['file']}")
    actual_hash = sha256_file(path)
    frame = _normalize_dates(pd.read_parquet(path))
    missing = sorted(set(contract["required_columns"]).difference(frame.columns))
    checks: dict[str, Any] = {
        "file": contract["file"],
        "sha256": actual_hash,
        "sha256_matches": actual_hash == contract["sha256"],
        "rows": int(len(frame)),
        "rows_match": len(frame) == int(contract["required_rows"]),
        "first_date": str(frame["date"].min().date()),
        "last_date": str(frame["date"].max().date()),
        "dates_match": (
            str(frame["date"].min().date()) == contract["required_first_date"]
            and str(frame["date"].max().date()) == contract["required_last_date"]
        ),
        "missing_columns": missing,
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "symbols": sorted(frame["symbol"].dropna().astype(str).unique().tolist()),
    }
    numeric_columns = ["close"] if not require_ohlc else ["open", "high", "low", "close"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    checks["missing_or_nonpositive_prices"] = int((numeric.isna() | (numeric <= 0)).sum().sum())
    invalid_ohlc = 0
    if require_ohlc:
        invalid = (
            (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        )
        invalid_ohlc = int(invalid.sum())
    checks["invalid_ohlc_rows"] = invalid_ohlc
    checks["status"] = "PASS" if all(
        [
            checks["sha256_matches"],
            checks["rows_match"],
            checks["dates_match"],
            not missing,
            checks["duplicate_dates"] == 0,
            checks["symbols"] == [symbol],
            checks["missing_or_nonpositive_prices"] == 0,
            invalid_ohlc == 0,
        ]
    ) else "BLOCKED"
    return frame, checks


def _audit_distributions(
    root: Path, contract: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / contract["file"]
    coverage_path = root / contract["coverage_file"]
    frame = pd.read_csv(path)
    for column in ("record_date", "ex_date", "payment_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["cash_dividend_per_share"] = pd.to_numeric(
        frame["cash_dividend_per_share"], errors="raise"
    )
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(path)
    coverage_hash = sha256_file(coverage_path)
    checks = {
        "file": contract["file"],
        "sha256": actual_hash,
        "coverage_sha256": coverage_hash,
        "event_count": int(len(frame)),
        "first_ex_date": str(frame["ex_date"].min().date()),
        "last_ex_date": str(frame["ex_date"].max().date()),
        "duplicate_ex_dates": int(frame["ex_date"].duplicated().sum()),
        "nonpositive_amounts": int((frame["cash_dividend_per_share"] <= 0).sum()),
        "coverage_complete": bool(coverage.get("complete_history_confirmed")),
        "coverage_start": coverage.get("coverage_start"),
        "coverage_end": coverage.get("coverage_end"),
        "coverage_distribution_hash_matches": coverage.get("distribution_file_sha256") == actual_hash,
    }
    checks["status"] = "PASS" if all(
        [
            actual_hash == contract["sha256"],
            coverage_hash == contract["coverage_sha256"],
            len(frame) == int(contract["required_event_count"]),
            checks["first_ex_date"] == contract["required_first_ex_date"],
            checks["last_ex_date"] == contract["required_last_ex_date"],
            checks["duplicate_ex_dates"] == 0,
            checks["nonpositive_amounts"] == 0,
            checks["coverage_complete"],
            checks["coverage_start"] == contract["coverage_start"],
            checks["coverage_end"] == contract["coverage_end"],
            checks["coverage_distribution_hash_matches"],
        ]
    ) else "BLOCKED"
    return frame.sort_values("ex_date").reset_index(drop=True), checks


def audit_and_load_inputs(
    root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contracts = config["data_contracts"]
    etf, etf_audit = _audit_market_frame(
        root, contracts["etf_market"], "510300.SH", require_ohlc=True
    )
    index, index_audit = _audit_market_frame(
        root, contracts["price_index"], "000300.SH", require_ohlc=True
    )
    total_return, total_return_audit = _audit_market_frame(
        root, contracts["total_return_index"], "H00300", require_ohlc=False
    )
    dividends, distribution_audit = _audit_distributions(root, contracts["distributions"])

    etf_dates = set(etf["date"])
    index_dates = set(index["date"])
    total_return_dates = set(total_return["date"])
    allowed_extra = {
        pd.Timestamp(value) for value in contracts["total_return_index"]["allowed_non_execution_dates"]
    }
    calendar_checks = {
        "etf_equals_price_index_calendar": etf_dates == index_dates,
        "missing_total_return_execution_dates": sorted(
            str(value.date()) for value in etf_dates.difference(total_return_dates)
        ),
        "extra_total_return_dates": sorted(
            str(value.date()) for value in total_return_dates.difference(etf_dates)
        ),
        "extra_total_return_dates_allowed": total_return_dates.difference(etf_dates) == allowed_extra,
    }

    index_columns = index[["date", "open", "high", "low", "close"]].rename(
        columns={column: f"price_index_{column}" for column in ("open", "high", "low", "close")}
    )
    tri_columns = total_return[["date", "close"]].rename(columns={"close": "h00300_close"})
    etf_columns = etf[["date", "open", "high", "low", "close"]].rename(
        columns={column: f"etf_{column}" for column in ("open", "high", "low", "close")}
    )
    merged = etf_columns.merge(index_columns, on="date", validate="one_to_one")
    merged = merged.merge(tri_columns, on="date", validate="one_to_one")
    merged["tr_scale_factor"] = merged["h00300_close"] / merged["price_index_close"]
    for column in ("open", "high", "low", "close"):
        merged[f"signal_{column}"] = merged[f"price_index_{column}"] * merged["tr_scale_factor"]
    close_error = (merged["signal_close"] - merged["h00300_close"]).abs()
    synthetic_invalid = (
        (merged["signal_high"] < merged[["signal_open", "signal_close", "signal_low"]].max(axis=1))
        | (merged["signal_low"] > merged[["signal_open", "signal_close", "signal_high"]].min(axis=1))
    )
    synthetic_checks = {
        "method": config["signal_contract"]["total_return_ohlc_method"],
        "official_h00300_ohlc_claimed": False,
        "rows": int(len(merged)),
        "positive_scale_factor": bool((merged["tr_scale_factor"] > 0).all()),
        "maximum_close_identity_error": float(close_error.max()),
        "close_identity_within_1e_10": bool((close_error <= 1e-10).all()),
        "invalid_ohlc_rows": int(synthetic_invalid.sum()),
    }
    synthetic_checks["status"] = "PASS" if all(
        [
            synthetic_checks["positive_scale_factor"],
            synthetic_checks["close_identity_within_1e_10"],
            synthetic_checks["invalid_ohlc_rows"] == 0,
        ]
    ) else "BLOCKED"
    branches = {
        "etf_market": etf_audit,
        "price_index": index_audit,
        "total_return_index": total_return_audit,
        "distributions": distribution_audit,
        "synthetic_total_return_ohlc": synthetic_checks,
    }
    gate_pass = (
        all(item["status"] == "PASS" for item in branches.values())
        and calendar_checks["etf_equals_price_index_calendar"]
        and not calendar_checks["missing_total_return_execution_dates"]
        and calendar_checks["extra_total_return_dates_allowed"]
    )
    audit = {
        "project_id": config["protocol"]["project_id"],
        "data_cutoff": config["protocol"]["historical_contamination_cutoff"],
        "status": "PASS" if gate_pass else "NO_VIEW",
        "return_calculation_allowed": bool(gate_pass),
        "branches": branches,
        "calendar": calendar_checks,
    }
    return merged.sort_values("date").reset_index(drop=True), dividends, audit


def build_signals(market: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    entry_days = int(variant["entry_lookback_trading_days"])
    exit_days = int(variant["exit_lookback_trading_days"])
    data = market.copy().sort_values("date").reset_index(drop=True)
    data["entry_channel"] = data["signal_high"].shift(1).rolling(entry_days).max()
    data["exit_channel"] = data["signal_low"].shift(1).rolling(exit_days).min()
    data["target_position"] = 0.0
    data["signal_action"] = "HOLD"
    data["channel_start_date"] = pd.NaT
    data["channel_end_date"] = pd.NaT
    data["channel_extreme_date"] = pd.NaT
    data["channel_value"] = np.nan
    state = 0.0
    for index in range(len(data)):
        if state == 0.0 and np.isfinite(data.at[index, "entry_channel"]):
            if float(data.at[index, "signal_close"]) > float(data.at[index, "entry_channel"]):
                state = 1.0
                data.at[index, "signal_action"] = "ENTER"
                start = index - entry_days
                window = data.loc[start : index - 1, "signal_high"]
                extreme_index = int(window.idxmax())
                data.at[index, "channel_start_date"] = data.at[start, "date"]
                data.at[index, "channel_end_date"] = data.at[index - 1, "date"]
                data.at[index, "channel_extreme_date"] = data.at[extreme_index, "date"]
                data.at[index, "channel_value"] = float(window.max())
        elif state == 1.0 and np.isfinite(data.at[index, "exit_channel"]):
            if float(data.at[index, "signal_close"]) < float(data.at[index, "exit_channel"]):
                state = 0.0
                data.at[index, "signal_action"] = "EXIT"
                start = index - exit_days
                window = data.loc[start : index - 1, "signal_low"]
                extreme_index = int(window.idxmin())
                data.at[index, "channel_start_date"] = data.at[start, "date"]
                data.at[index, "channel_end_date"] = data.at[index - 1, "date"]
                data.at[index, "channel_extreme_date"] = data.at[extreme_index, "date"]
                data.at[index, "channel_value"] = float(window.min())
        data.at[index, "target_position"] = state
    data["trade_allowed"] = data["signal_action"].ne("HOLD")
    data["signal_reason"] = data["signal_action"].map(
        {"ENTER": "55/20或100/50通道进入", "EXIT": "55/20或100/50通道退出", "HOLD": "状态维持"}
    )
    first_eligible = data["entry_channel"].first_valid_index()
    if first_eligible is None:
        raise ValueError(f"{variant['id']}没有足够的进入通道历史")
    result = data.loc[first_eligible:].copy().reset_index(drop=True)
    result["variant_id"] = variant["id"]
    return result


def _dividend_maps(dividends: pd.DataFrame) -> tuple[dict[pd.Timestamp, list[dict]], dict[pd.Timestamp, list[dict]]]:
    records = dividends.to_dict("records")
    ex_map: dict[pd.Timestamp, list[dict]] = {}
    pay_map: dict[pd.Timestamp, list[dict]] = {}
    for item in records:
        ex_map.setdefault(pd.Timestamp(item["ex_date"]), []).append(item)
        pay_map.setdefault(pd.Timestamp(item["payment_date"]), []).append(item)
    return ex_map, pay_map


def simulate_portfolio(
    signals: pd.DataFrame,
    dividends: pd.DataFrame,
    costs: CostModel,
    initial_cash: float,
    fractional: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅在状态改变时执行；信号在下一交易日开盘生效。"""

    data = signals.copy().sort_values("date").reset_index(drop=True)
    ex_map, _ = _dividend_maps(dividends)
    signal_by_date = data.set_index("date").to_dict("index")
    cash = float(initial_cash)
    shares = 0.0
    receivable = 0.0
    scheduled: dict[pd.Timestamp, float] = {}
    previous_date: pd.Timestamp | None = None
    ledgers: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if previous_date is not None:
            cash *= 1.0 + costs.cash_annual_rate / costs.trading_days_per_year
        shares_at_start = shares
        entitlement_today = 0.0
        for event in ex_map.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            entitlement_today += entitlement
            payment_date = pd.Timestamp(event["payment_date"])
            scheduled[payment_date] = scheduled.get(payment_date, 0.0) + entitlement
        receivable += entitlement_today

        signal = signal_by_date.get(previous_date, {}) if previous_date is not None else {}
        action = str(signal.get("signal_action", "HOLD"))
        target = float(signal.get("target_position", 0.0 if shares == 0 else 1.0))
        cash_before_trade = cash
        shares_before_trade = shares
        open_equity = cash + shares * float(row.etf_open) + receivable
        side = "NONE"
        quantity = 0.0
        raw_open = float(row.etf_open)
        execution_price = np.nan
        execution_notional = 0.0
        commission = 0.0
        slippage_cost = 0.0
        blocked_reason: str | None = None
        if action == "ENTER" and shares == 0:
            side = "BUY"
            execution_price = raw_open * (1.0 + costs.slippage_bps / 10000.0)
            if fractional:
                quantity = cash / (execution_price * (1.0 + costs.commission_rate))
            else:
                assert costs.lot_size is not None
                quantity = float(int(cash / execution_price / costs.lot_size) * costs.lot_size)
                while quantity > 0:
                    execution_notional = quantity * execution_price
                    commission = costs.commission(execution_notional)
                    if execution_notional + commission <= cash + 1e-9:
                        break
                    quantity -= costs.lot_size
                if quantity < costs.minimum_trade_shares:
                    blocked_reason = "BELOW_MINIMUM_ORDINARY_TRADE_SHARES"
                    quantity = 0.0
            if quantity > 0:
                execution_notional = quantity * execution_price
                commission = costs.commission(execution_notional)
                cash -= execution_notional + commission
                shares += quantity
                slippage_cost = quantity * (execution_price - raw_open)
            else:
                side = "BLOCKED"
        elif action == "EXIT" and shares > 0:
            side = "SELL"
            execution_price = raw_open * (1.0 - costs.slippage_bps / 10000.0)
            quantity = shares
            if not fractional and quantity < costs.minimum_trade_shares:
                blocked_reason = "BELOW_MINIMUM_ORDINARY_TRADE_SHARES"
                side = "BLOCKED"
                quantity = 0.0
            if quantity > 0:
                execution_notional = quantity * execution_price
                commission = costs.commission(execution_notional)
                cash += execution_notional - commission
                shares -= quantity
                slippage_cost = quantity * (raw_open - execution_price)

        if side in {"BUY", "SELL"}:
            trades.append(
                {
                    "variant_id": row.variant_id,
                    "signal_date": previous_date,
                    "execution_date": date,
                    "signal_action": action,
                    "side": side,
                    "raw_open": raw_open,
                    "execution_price": float(execution_price),
                    "quantity": float(quantity),
                    "execution_notional": float(execution_notional),
                    "commission": float(commission),
                    "slippage_cost": float(slippage_cost),
                    "cash_before_trade": float(cash_before_trade),
                    "cash_after_trade": float(cash),
                    "shares_before_trade": float(shares_before_trade),
                    "shares_after_trade": float(shares),
                    "t_plus_one_sellable_before_trade": float(shares_at_start),
                    "target_position": target,
                }
            )
        elif side == "BLOCKED":
            trades.append(
                {
                    "variant_id": row.variant_id,
                    "signal_date": previous_date,
                    "execution_date": date,
                    "signal_action": action,
                    "side": side,
                    "raw_open": raw_open,
                    "execution_price": float(execution_price),
                    "quantity": 0.0,
                    "execution_notional": 0.0,
                    "commission": 0.0,
                    "slippage_cost": 0.0,
                    "cash_before_trade": float(cash_before_trade),
                    "cash_after_trade": float(cash),
                    "shares_before_trade": float(shares_before_trade),
                    "shares_after_trade": float(shares),
                    "t_plus_one_sellable_before_trade": float(shares_at_start),
                    "target_position": target,
                    "blocked_reason": blocked_reason,
                }
            )

        payment_today = float(scheduled.pop(date, 0.0))
        if payment_today:
            cash += payment_today
            receivable -= payment_today
            if abs(receivable) < 1e-12:
                receivable = 0.0
        equity = cash + shares * float(row.etf_close) + receivable
        ledgers.append(
            {
                "variant_id": row.variant_id,
                "date": date,
                "signal_date_used": previous_date,
                "signal_action_used": action,
                "target_position": target,
                "shares": float(shares),
                "cash": float(cash),
                "dividend_receivable": float(receivable),
                "dividend_entitlement_today": float(entitlement_today),
                "dividend_payment_today": float(payment_today),
                "etf_open": raw_open,
                "etf_close": float(row.etf_close),
                "equity": float(equity),
                "actual_position": float(shares * float(row.etf_close) / equity) if equity else 0.0,
                "daily_commission": float(commission),
                "daily_slippage_cost": float(slippage_cost),
                "blocked_reason": blocked_reason,
            }
        )
        previous_date = date
    ledger = pd.DataFrame(ledgers)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    return ledger, pd.DataFrame(trades)


def build_buy_hold_signals(signals: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    result = signals.copy()
    result["variant_id"] = variant_id
    result["target_position"] = 1.0
    result["signal_action"] = "HOLD"
    result.loc[result.index[0], "signal_action"] = "ENTER"
    result["trade_allowed"] = result["signal_action"].ne("HOLD")
    return result


def build_h00300_benchmark(
    signals: pd.DataFrame,
    initial_wealth: float,
    cash_annual_rate: float,
    trading_days: int,
) -> pd.DataFrame:
    data = signals.copy().reset_index(drop=True)
    equity = np.full(len(data), float(initial_wealth), dtype=float)
    if len(data) >= 2:
        equity[1] = initial_wealth * (1.0 + cash_annual_rate / trading_days) * float(
            data.loc[1, "signal_close"] / data.loc[1, "signal_open"]
        )
        for index in range(2, len(data)):
            equity[index] = equity[index - 1] * float(
                data.loc[index, "signal_close"] / data.loc[index - 1, "signal_close"]
            )
    result = pd.DataFrame({"date": data["date"], "equity": equity})
    result["daily_return"] = result["equity"].pct_change().fillna(0.0)
    result["actual_position"] = np.where(result.index == 0, 0.0, 1.0)
    result["equity_peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["equity_peak"] - 1.0
    return result


def build_fixed_mix_benchmark(
    signals: pd.DataFrame,
    dividends: pd.DataFrame,
    exposure: float,
    initial_wealth: float,
    cash_annual_rate: float,
    trading_days: int,
) -> pd.DataFrame:
    data = signals.copy().reset_index(drop=True)
    dividend_by_ex = dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    etf_returns = np.zeros(len(data), dtype=float)
    cash_daily = cash_annual_rate / trading_days
    if len(data) >= 2:
        overnight_cash_growth = 1.0 + cash_daily
        intraday_return = float(data.loc[1, "etf_close"] / data.loc[1, "etf_open"] - 1.0)
        etf_returns[1] = overnight_cash_growth * (1.0 + intraday_return) - 1.0
    for index in range(2, len(data)):
        distribution = float(dividend_by_ex.get(pd.Timestamp(data.loc[index, "date"]), 0.0))
        etf_returns[index] = float(
            (data.loc[index, "etf_close"] + distribution) / data.loc[index - 1, "etf_close"] - 1.0
        )
    returns = exposure * etf_returns + (1.0 - exposure) * cash_daily
    returns[0] = 0.0
    equity = initial_wealth * np.cumprod(1.0 + returns)
    result = pd.DataFrame(
        {
            "date": data["date"],
            "equity": equity,
            "daily_return": returns,
            "actual_position": exposure,
        }
    )
    result["equity_peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["equity_peak"] - 1.0
    return result


def _longest_drawdown_days(equity: pd.Series) -> int:
    underwater = equity < equity.cummax() - 1e-12
    longest = current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def summarize_ledger(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_wealth: float,
    cash_annual_rate: float,
    trading_days: int,
    minimum_commission: float,
) -> dict[str, Any]:
    returns = ledger["daily_return"].astype(float)
    observations = max(len(returns) - 1, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_wealth - 1.0)
    cagr = float((1.0 + total_return) ** (trading_days / observations) - 1.0)
    volatility = float(returns.iloc[1:].std(ddof=1) * np.sqrt(trading_days))
    excess = returns.iloc[1:] - cash_annual_rate / trading_days
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(trading_days)) if excess.std(ddof=1) > 0 else None
    downside = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt((downside**2).mean()) * np.sqrt(trading_days))
    sortino = float(excess.mean() * trading_days / downside_deviation) if downside_deviation > 0 else None
    max_drawdown = float(ledger["drawdown"].min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else None
    executed = (
        trades.loc[trades["side"].isin(["BUY", "SELL"])].copy()
        if "side" in trades.columns
        else trades.copy()
    )
    total_commission = float(executed["commission"].sum()) if not executed.empty else 0.0
    total_slippage = float(executed["slippage_cost"].sum()) if not executed.empty else 0.0
    turnover = float(executed["execution_notional"].sum() / ledger["equity"].mean()) if not executed.empty else 0.0
    minimum_count = 0
    if minimum_commission > 0 and not executed.empty:
        minimum_count = int(np.isclose(executed["commission"], minimum_commission, atol=1e-9).sum())
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "observations": int(len(ledger)),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_excess_cash": sharpe,
        "sortino_excess_cash": sortino,
        "maximum_drawdown": max_drawdown,
        "calmar": calmar,
        "longest_drawdown_recovery_trading_days": _longest_drawdown_days(ledger["equity"]),
        "average_exposure": float(ledger["actual_position"].mean()),
        "trade_count": int(len(executed)),
        "turnover_over_average_equity": turnover,
        "commission_cny_or_wealth_units": total_commission,
        "slippage_cny_or_wealth_units": total_slippage,
        "total_execution_cost_cny_or_wealth_units": total_commission + total_slippage,
        "minimum_commission_trade_count": minimum_count,
        "minimum_commission_trade_ratio": float(minimum_count / len(executed)) if len(executed) else 0.0,
        "ending_equity": float(ledger["equity"].iloc[-1]),
    }


def relative_metrics(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    trading_days: int,
    rolling_windows: list[int],
) -> tuple[dict[str, Any], dict[int, pd.DataFrame]]:
    merged = strategy[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]], on="date", suffixes=("_strategy", "_benchmark"), validate="one_to_one"
    )
    returns_s = merged["daily_return_strategy"].astype(float)
    returns_b = merged["daily_return_benchmark"].astype(float)
    active = returns_s - returns_b
    count = max(len(merged) - 1, 1)
    cagr_s = float(np.prod(1.0 + returns_s) ** (trading_days / count) - 1.0)
    cagr_b = float(np.prod(1.0 + returns_b) ** (trading_days / count) - 1.0)
    active_std = active.iloc[1:].std(ddof=1)
    information_ratio = float(active.iloc[1:].mean() / active_std * np.sqrt(trading_days)) if active_std > 0 else None
    up = returns_b > 0
    down = returns_b < 0
    upside = float(returns_s[up].sum() / returns_b[up].sum()) if returns_b[up].sum() != 0 else None
    downside = float(returns_s[down].sum() / returns_b[down].sum()) if returns_b[down].sum() != 0 else None
    relative_wealth = ((1.0 + returns_s) / (1.0 + returns_b)).cumprod()
    active_drawdown = relative_wealth / relative_wealth.cummax() - 1.0
    rolling_payload: dict[str, Any] = {}
    rolling_frames: dict[int, pd.DataFrame] = {}
    for window in rolling_windows:
        values = (
            (1.0 + returns_s).rolling(window).apply(np.prod, raw=True)
            / (1.0 + returns_b).rolling(window).apply(np.prod, raw=True)
            - 1.0
        )
        frame = pd.DataFrame({"date": merged["date"], "rolling_active_return": values}).dropna()
        rolling_frames[window] = frame
        rolling_payload[str(window)] = {
            "observations": int(len(frame)),
            "median": float(frame["rolling_active_return"].median()) if len(frame) else None,
            "positive_ratio": float((frame["rolling_active_return"] > 0).mean()) if len(frame) else None,
            "minimum": float(frame["rolling_active_return"].min()) if len(frame) else None,
            "maximum": float(frame["rolling_active_return"].max()) if len(frame) else None,
        }
    result = {
        "annualized_active_return": cagr_s - cagr_b,
        "information_ratio": information_ratio,
        "upside_capture": upside,
        "downside_capture": downside,
        "active_maximum_drawdown": float(active_drawdown.min()),
        "daily_relative_win_rate": float((returns_s.iloc[1:] > returns_b.iloc[1:]).mean()),
        "rolling_active_returns": rolling_payload,
    }
    return result, rolling_frames


def build_cycles(
    trades: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty or "side" not in trades.columns:
        return pd.DataFrame()
    executed = trades.loc[trades["side"].isin(["BUY", "SELL"])].sort_values("execution_date")
    if executed.empty:
        return pd.DataFrame()
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(market["date"])}
    cycles: list[dict[str, Any]] = []
    entry: pd.Series | None = None
    for _, trade in executed.iterrows():
        if trade["side"] == "BUY":
            if entry is not None:
                raise ValueError("出现未闭合进入后再次买入")
            entry = trade
            continue
        if entry is None:
            raise ValueError("出现没有对应进入的卖出")
        cycles.append(_cycle_row(entry, trade, market, dividends, date_to_index, closed=True))
        entry = None
    if entry is not None:
        cycles.append(_cycle_row(entry, None, market, dividends, date_to_index, closed=False))
    return pd.DataFrame(cycles)


def _cycle_row(
    entry: pd.Series,
    exit_trade: pd.Series | None,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    date_to_index: dict[pd.Timestamp, int],
    closed: bool,
) -> dict[str, Any]:
    entry_date = pd.Timestamp(entry["execution_date"])
    end_date = pd.Timestamp(exit_trade["execution_date"]) if closed and exit_trade is not None else pd.Timestamp(market["date"].iloc[-1])
    quantity = float(entry["quantity"])
    eligible = dividends.loc[(dividends["ex_date"] > entry_date) & (dividends["ex_date"] <= end_date)]
    dividend_per_share = float(eligible["cash_dividend_per_share"].sum())
    dividend_cash = quantity * dividend_per_share
    entry_index = date_to_index[entry_date]
    end_index = date_to_index[end_date]
    window = market.iloc[entry_index : end_index + 1]
    signal_entry_open = float(market.loc[entry_index, "signal_open"])
    if closed and exit_trade is not None:
        gross_pnl = quantity * (float(exit_trade["raw_open"]) - float(entry["raw_open"])) + dividend_cash
        net_pnl = quantity * (float(exit_trade["execution_price"]) - float(entry["execution_price"])) + dividend_cash - float(entry["commission"]) - float(exit_trade["commission"])
        exit_signal_date = pd.Timestamp(exit_trade["signal_date"])
        exit_execution_price = float(exit_trade["execution_price"])
        exit_commission = float(exit_trade["commission"])
    else:
        gross_pnl = quantity * (float(market.loc[end_index, "etf_close"]) - float(entry["raw_open"])) + dividend_cash
        net_pnl = quantity * (float(market.loc[end_index, "etf_close"]) - float(entry["execution_price"])) + dividend_cash - float(entry["commission"])
        exit_signal_date = pd.NaT
        exit_execution_price = np.nan
        exit_commission = 0.0
    return {
        "variant_id": entry["variant_id"],
        "closed": bool(closed),
        "entry_signal_date": pd.Timestamp(entry["signal_date"]),
        "entry_execution_date": entry_date,
        "exit_signal_date": exit_signal_date,
        "exit_execution_date": end_date if closed else pd.NaT,
        "mark_date_if_open": end_date if not closed else pd.NaT,
        "quantity": quantity,
        "entry_raw_open": float(entry["raw_open"]),
        "entry_execution_price": float(entry["execution_price"]),
        "exit_execution_price": exit_execution_price,
        "entry_commission": float(entry["commission"]),
        "exit_commission": exit_commission,
        "dividend_cash": dividend_cash,
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "holding_trading_days": int(end_index - entry_index + 1),
        "mae_h00300_total_return": float(window["signal_low"].min() / signal_entry_open - 1.0),
        "mfe_h00300_total_return": float(window["signal_high"].max() / signal_entry_open - 1.0),
    }


def build_event_audit(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    cycles: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    events = signals.loc[signals["signal_action"].isin(["ENTER", "EXIT"])].copy()
    trade_map = {pd.Timestamp(row.signal_date): row for row in trades.itertuples(index=False)}
    entry_cycle = {}
    exit_cycle = {}
    if not cycles.empty:
        entry_cycle = {pd.Timestamp(row.entry_signal_date): row for row in cycles.itertuples(index=False)}
        exit_cycle = {
            pd.Timestamp(row.exit_signal_date): row
            for row in cycles.itertuples(index=False)
            if pd.notna(row.exit_signal_date)
        }
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(signals["date"])}
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        signal_date = pd.Timestamp(event.date)
        trade = trade_map.get(signal_date)
        cycle = entry_cycle.get(signal_date) if event.signal_action == "ENTER" else exit_cycle.get(signal_date)
        row: dict[str, Any] = {
            "variant_id": event.variant_id,
            "signal_type": event.signal_action,
            "signal_date": signal_date,
            "channel_start_date": event.channel_start_date,
            "channel_end_date": event.channel_end_date,
            "channel_extreme_date": event.channel_extreme_date,
            "price_index_raw_close": float(event.price_index_close),
            "synthetic_total_return_close": float(event.signal_close),
            "channel_value": float(event.channel_value),
            "execution_status": "NO_NEXT_OPEN" if signal_date == pd.Timestamp(signals["date"].iloc[-1]) else "NO_EXECUTION",
            "execution_date": pd.NaT,
            "etf_raw_open": np.nan,
            "slippage_adjusted_price": np.nan,
            "shares": np.nan,
            "notional": np.nan,
            "commission": np.nan,
            "cash_before_trade": np.nan,
            "cash_after_trade": np.nan,
            "holding_trading_days": np.nan,
            "dividend_cash": np.nan,
            "gross_pnl": np.nan,
            "net_pnl": np.nan,
            "mae_h00300_total_return": np.nan,
            "mfe_h00300_total_return": np.nan,
        }
        if trade is not None:
            row.update(
                {
                    "execution_status": trade.side,
                    "execution_date": pd.Timestamp(trade.execution_date),
                    "etf_raw_open": float(trade.raw_open),
                    "slippage_adjusted_price": float(trade.execution_price),
                    "shares": float(trade.quantity),
                    "notional": float(trade.execution_notional),
                    "commission": float(trade.commission),
                    "cash_before_trade": float(trade.cash_before_trade),
                    "cash_after_trade": float(trade.cash_after_trade),
                }
            )
            execution_index = date_to_index[pd.Timestamp(trade.execution_date)]
            signal_open = float(signals.loc[execution_index, "signal_open"])
            for horizon in horizons:
                horizon_index = execution_index + horizon - 1
                row[f"forward_h00300_total_return_{horizon}d"] = (
                    float(signals.loc[horizon_index, "signal_close"] / signal_open - 1.0)
                    if horizon_index < len(signals)
                    else np.nan
                )
        else:
            for horizon in horizons:
                row[f"forward_h00300_total_return_{horizon}d"] = np.nan
        if cycle is not None:
            row.update(
                {
                    "holding_trading_days": int(cycle.holding_trading_days),
                    "dividend_cash": float(cycle.dividend_cash),
                    "gross_pnl": float(cycle.gross_pnl),
                    "net_pnl": float(cycle.net_pnl),
                    "mae_h00300_total_return": float(cycle.mae_h00300_total_return),
                    "mfe_h00300_total_return": float(cycle.mfe_h00300_total_return),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def annual_relative_table(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    merged = strategy[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]], on="date", suffixes=("_strategy", "_benchmark"), validate="one_to_one"
    )
    merged["year"] = merged["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in merged.groupby("year", sort=True):
        strategy_return = float(np.prod(1.0 + group["daily_return_strategy"]) - 1.0)
        benchmark_return = float(np.prod(1.0 + group["daily_return_benchmark"]) - 1.0)
        active_return = float((1.0 + strategy_return) / (1.0 + benchmark_return) - 1.0)
        rows.append(
            {
                "year": int(year),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "active_return": active_return,
            }
        )
    return pd.DataFrame(rows)


def period_diagnostics(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    periods: list[dict[str, Any]],
    trading_days: int,
) -> dict[str, Any]:
    merged = strategy[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]], on="date", suffixes=("_strategy", "_benchmark"), validate="one_to_one"
    )
    period_rows = []
    for period in periods:
        frame = merged.loc[merged["date"].between(period["start"], period["end"])]
        relative = float(
            np.prod((1.0 + frame["daily_return_strategy"]) / (1.0 + frame["daily_return_benchmark"])) - 1.0
        ) if len(frame) else None
        period_rows.append({**period, "observations": int(len(frame)), "active_return": relative})
    annual = annual_relative_table(strategy, benchmark)
    best_year = int(annual.loc[annual["active_return"].idxmax(), "year"])
    without = merged.loc[merged["date"].dt.year.ne(best_year)]
    n = max(len(without) - 1, 1)
    without_best = float(
        np.prod((1.0 + without["daily_return_strategy"]) / (1.0 + without["daily_return_benchmark"])) ** (trading_days / n) - 1.0
    )
    positive_logs = np.log1p(annual.loc[annual["active_return"] > 0, "active_return"])
    concentration = float(positive_logs.max() / positive_logs.sum()) if positive_logs.sum() > 0 else None
    return {
        "predefined_periods": period_rows,
        "positive_predefined_period_count": int(sum(item["active_return"] is not None and item["active_return"] > 0 for item in period_rows)),
        "best_active_year_removed": best_year,
        "annualized_active_return_without_best_year": without_best,
        "maximum_positive_active_year_contribution_share": concentration,
        "annual": annual,
    }


def multiple_testing_diagnostics(
    active_returns: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    names = list(active_returns)
    merged: pd.DataFrame | None = None
    for name, frame in active_returns.items():
        current = frame[["date", "active_return"]].rename(columns={"active_return": name})
        merged = current if merged is None else merged.merge(current, on="date", how="inner", validate="one_to_one")
    assert merged is not None
    matrix = merged[names].to_numpy(float)[1:]
    n = len(matrix)
    settings = config["evaluation"]["multiple_testing"]
    repetitions = int(settings["repetitions"])
    block = int(settings["circular_block_length_trading_days"])
    rng = np.random.default_rng(int(settings["random_seed"]))
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1)
    observed_rc = float(np.sqrt(n) * means.max())
    observed_spa = float(np.max(np.sqrt(n) * means / np.where(stds > 0, stds, np.inf)))
    centered = matrix - means
    rc_boot = np.empty(repetitions)
    spa_boot = np.empty(repetitions)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sample = centered[indices]
        sample_means = sample.mean(axis=0)
        rc_boot[repetition] = np.sqrt(n) * sample_means.max()
        sample_stds = sample.std(axis=0, ddof=1)
        spa_boot[repetition] = np.max(
            np.sqrt(n) * sample_means / np.where(sample_stds > 0, sample_stds, np.inf)
        )
    primary = matrix[:, names.index("TECH_03_DONCHIAN_55_20_V1")]
    sr_daily = float(primary.mean() / primary.std(ddof=1)) if primary.std(ddof=1) > 0 else 0.0
    trial_count = int(config["trial_registry"]["registered_trial_count"])
    expected_max_z = float(norm.ppf((trial_count - 0.375) / (trial_count + 0.25)))
    sr0_daily = expected_max_z / np.sqrt(max(n - 1, 1))
    sample_skew = float(skew(primary, bias=False))
    sample_kurtosis = float(kurtosis(primary, fisher=False, bias=False))
    denominator = np.sqrt(
        max(1e-12, 1.0 - sample_skew * sr_daily + ((sample_kurtosis - 1.0) / 4.0) * sr_daily**2)
    )
    dsr_z = float((sr_daily - sr0_daily) * np.sqrt(max(n - 1, 1)) / denominator)
    minimum_candidates = int(settings["pbo_minimum_comparable_candidates"])
    return {
        "registered_trial_count": trial_count,
        "return_tested_candidate_count": len(names),
        "common_observations": int(n),
        "candidate_ids": names,
        "white_reality_check": {
            "method": "CIRCULAR_BLOCK_BOOTSTRAP_CENTERED_ACTIVE_RETURNS",
            "block_length": block,
            "repetitions": repetitions,
            "observed_statistic": observed_rc,
            "p_value": float((1 + np.sum(rc_boot >= observed_rc)) / (repetitions + 1)),
        },
        "hansen_spa": {
            "method": "STUDENTIZED_CIRCULAR_BLOCK_BOOTSTRAP_APPROXIMATION",
            "block_length": block,
            "repetitions": repetitions,
            "observed_statistic": observed_spa,
            "p_value": float((1 + np.sum(spa_boot >= observed_spa)) / (repetitions + 1)),
        },
        "deflated_sharpe_ratio": {
            "basis": "PRIMARY_ACTIVE_DAILY_RETURNS_VS_510300_BUY_HOLD",
            "registered_trial_count": trial_count,
            "annualized_active_sharpe": sr_daily * np.sqrt(config["evaluation"]["annualization_days"]),
            "expected_maximum_null_daily_sharpe": sr0_daily,
            "skew": sample_skew,
            "kurtosis": sample_kurtosis,
            "probability": float(norm.cdf(dsr_z)),
        },
        "pbo": {
            "status": "NOT_ESTIMABLE" if len(names) < minimum_candidates else "NOT_IMPLEMENTED_UNEXPECTED",
            "comparable_candidate_count": len(names),
            "minimum_required": minimum_candidates,
            "reason": "实际可比较收益候选少于3，CSCV排名退化" if len(names) < minimum_candidates else None,
            "value": None,
        },
        "parameter_neighborhood": {
            "status": "NOT_RUN_PROHIBITED_BY_FROZEN_PARAMETER_BUDGET",
            "fixed_robustness_variant": "TECH_03_DONCHIAN_100_50_ROBUSTNESS_V1",
        },
    }


def make_cost_model(config: dict[str, Any], layer: str, stress: bool) -> CostModel:
    execution = config["execution"]
    multiplier = float(execution["stress_cost_multiplier"]) if stress else 1.0
    if layer == "theoretical":
        return CostModel(
            commission_rate=float(execution["commission_rate_per_leg"]) * multiplier,
            minimum_commission=0.0,
            slippage_bps=float(execution["slippage_bps_per_leg"]) * multiplier,
            cash_annual_rate=float(execution["cash_annual_rate"]),
            trading_days_per_year=int(execution["trading_days_per_year"]),
            lot_size=None,
            minimum_trade_shares=0,
        )
    return CostModel(
        commission_rate=float(execution["commission_rate_per_leg"]) * multiplier,
        minimum_commission=float(execution["minimum_commission_cny_per_leg"]) * multiplier,
        slippage_bps=float(execution["slippage_bps_per_leg"]) * multiplier,
        cash_annual_rate=float(execution["cash_annual_rate"]),
        trading_days_per_year=int(execution["trading_days_per_year"]),
        lot_size=int(execution["small_account"]["lot_size_shares"]),
        minimum_trade_shares=int(execution["small_account"]["minimum_ordinary_trade_shares"]),
    )


def active_return_frame(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    merged = strategy[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]], on="date", suffixes=("_strategy", "_benchmark"), validate="one_to_one"
    )
    merged["active_return"] = merged["daily_return_strategy"] - merged["daily_return_benchmark"]
    return merged[["date", "active_return"]]


def evaluate_decision(
    config: dict[str, Any],
    primary: dict[str, Any],
    robustness: dict[str, Any],
    multiple_testing: dict[str, Any],
) -> dict[str, Any]:
    gates = config["evaluation"]["historical_gates"]
    rel = primary["small_account_base"]["relative_vs_510300"]
    stress_rel = primary["small_account_stress"]["relative_vs_510300"]
    robust_rel = robustness["small_account_base"]["relative_vs_510300"]
    periods = primary["period_diagnostics"]
    mt = multiple_testing
    checks = {
        "annualized_net_excess_at_least_1_5pct": rel["annualized_active_return"] >= float(gates["annualized_net_excess_vs_510300_minimum"]),
        "information_ratio_at_least_0_35": rel["information_ratio"] is not None and rel["information_ratio"] >= float(gates["information_ratio_minimum"]),
        "positive_predefined_periods_at_least_3": periods["positive_predefined_period_count"] >= int(gates["positive_predefined_periods_minimum"]),
        "double_cost_excess_positive": stress_rel["annualized_active_return"] > 0,
        "fixed_100_50_robustness_excess_positive": robust_rel["annualized_active_return"] > 0,
        "remove_best_year_excess_positive": periods["annualized_active_return_without_best_year"] > 0,
        "single_positive_year_contribution_not_over_50pct": periods["maximum_positive_active_year_contribution_share"] is not None and periods["maximum_positive_active_year_contribution_share"] <= float(gates["maximum_positive_active_year_contribution_share"]),
        "white_reality_check_5pct": mt["white_reality_check"]["p_value"] <= float(config["evaluation"]["multiple_testing"]["significance_level"]),
        "hansen_spa_5pct": mt["hansen_spa"]["p_value"] <= float(config["evaluation"]["multiple_testing"]["significance_level"]),
        "deflated_sharpe_probability_95pct": mt["deflated_sharpe_ratio"]["probability"] >= float(config["evaluation"]["multiple_testing"]["deflated_sharpe_probability_minimum"]),
    }
    historical_screen_pass = all(checks.values())
    strategy_summary = primary["small_account_base"]["strategy"]
    benchmark_summary = primary["small_account_base"]["buy_hold"]
    risk_gates = config["evaluation"]["risk_overlay_gates"]
    sharpe_improvement = (
        strategy_summary["sharpe_excess_cash"] - benchmark_summary["sharpe_excess_cash"]
        if strategy_summary["sharpe_excess_cash"] is not None and benchmark_summary["sharpe_excess_cash"] is not None
        else None
    )
    drawdown_reduction = 1.0 - abs(strategy_summary["maximum_drawdown"]) / abs(benchmark_summary["maximum_drawdown"])
    risk_checks = {
        "sharpe_improvement_at_least_0_15": sharpe_improvement is not None and sharpe_improvement >= float(risk_gates["sharpe_improvement_minimum"]),
        "drawdown_reduction_at_least_20pct": drawdown_reduction >= float(risk_gates["drawdown_reduction_minimum"]),
        "upside_capture_at_least_70pct": rel["upside_capture"] is not None and rel["upside_capture"] >= float(risk_gates["upside_capture_minimum"]),
    }
    risk_overlay_pass = all(risk_checks.values()) and not checks["annualized_net_excess_at_least_1_5pct"]
    if historical_screen_pass:
        decision = "HISTORICAL_SCREEN_PASS_FORWARD_REQUIRED"
    elif risk_overlay_pass:
        decision = "RISK_OVERLAY_CANDIDATE_HISTORICAL_ONLY"
    else:
        decision = "REJECT_HISTORICAL"
    return {
        "decision": decision,
        "alpha_pass": False,
        "alpha_pass_blockers": [
            "TRUE_OOS_NOT_STARTED",
            "PBO_NOT_ESTIMABLE_WITH_TWO_RETURN_TESTED_CANDIDATES",
            "NEIGHBOR_PARAMETER_GRID_PROHIBITED",
        ],
        "historical_screen_pass": historical_screen_pass,
        "historical_gate_checks": checks,
        "risk_overlay_historical_pass": risk_overlay_pass,
        "risk_overlay_checks": risk_checks,
        "sharpe_improvement": sharpe_improvement,
        "drawdown_reduction": drawdown_reduction,
    }
