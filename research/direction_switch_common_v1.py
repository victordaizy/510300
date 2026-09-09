"""510300研究方向切换V1的公共数据、冻结校验与二元仓位仿真工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_direction_switch_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_direction_switch_v1_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if value is pd.NA:
        return None
    return value


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=json_default,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_replace_bytes(path, encoded + b"\n")


def atomic_text(payload: str, path: Path) -> None:
    _atomic_replace_bytes(path, payload.encode("utf-8"))


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许执行冻结的510300方向切换V1协议")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("协议顶层必须为对象")
    protocol = payload["protocol"]
    if protocol["study_id"] != "510300_DIRECTION_SWITCH_V1":
        raise ValueError("研究编号不匹配")
    if protocol["state"] != "PREREGISTERED_BEFORE_NEW_OUTCOMES":
        raise ValueError("协议未处于预注册冻结状态")
    if protocol["trade_assets"] != ["510300.SH", "CASH_CNY"]:
        raise ValueError("资产边界必须严格为510300和人民币现金")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("协议必须禁止参数营救")
    if bool(protocol["live_trading_authorized"]):
        raise ValueError("实盘授权必须关闭")
    return payload


def verify_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("冻结清单不存在")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["freeze_state"] != "PREREGISTERED_BEFORE_NEW_OUTCOMES":
        raise ValueError("冻结清单状态无效")
    actual_protocol_hash = sha256_file(CONFIG_PATH)
    if actual_protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("冻结协议哈希漂移")
    if manifest["new_outcomes_read_before_freeze"] is not False:
        raise ValueError("冻结清单记录显示预注册前已读取新结果")
    if int(manifest["historical_runs_consumed"]) != 0:
        raise ValueError("冻结清单已消耗历史运行次数")
    for relative, expected in manifest["input_sha256"].items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"冻结输入不存在：{relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"冻结输入哈希漂移：{relative}")
    implementation_hashes = manifest.get("implementation_sha256", {})
    if not implementation_hashes:
        raise ValueError("冻结清单缺少实现哈希")
    for relative, expected in implementation_hashes.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"冻结实现不存在：{relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"冻结实现哈希漂移：{relative}")
    for name, item in config["inputs"].items():
        path = ROOT / item["file"]
        if not path.exists():
            raise FileNotFoundError(f"协议输入不存在：{name} -> {item['file']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"协议输入哈希漂移：{name}")
    return manifest


def required_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少字段：{missing}")


def normalize_market(frame: pd.DataFrame, expected_symbol: str) -> pd.DataFrame:
    required_columns(
        frame,
        ["date", "open", "high", "low", "close", "volume", "symbol"],
        expected_symbol,
    )
    market = frame.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market.sort_values("date", kind="mergesort", inplace=True)
    market.reset_index(drop=True, inplace=True)
    if market["date"].duplicated().any():
        raise ValueError(f"{expected_symbol}存在重复日期")
    symbols = set(market["symbol"].dropna().astype(str).unique())
    if symbols != {expected_symbol}:
        raise ValueError(f"代码不匹配：期望{expected_symbol}，实际{sorted(symbols)}")
    for column in ["open", "high", "low", "close", "volume"]:
        market[column] = pd.to_numeric(market[column], errors="raise")
    if market[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{expected_symbol}价格存在空值")
    if (market[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError(f"{expected_symbol}价格存在非正值")
    if (market["volume"] < 0.0).any():
        raise ValueError(f"{expected_symbol}成交量存在负值")
    if (
        (market["high"] < market[["open", "close", "low"]].max(axis=1))
        | (market["low"] > market[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError(f"{expected_symbol}存在OHLC逻辑错误")
    return market


def normalize_dividends(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns(
        frame,
        [
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
        ],
        "510300分红",
    )
    dividends = frame.copy()
    if set(dividends["symbol"].astype(str).unique()) != {"510300.SH"}:
        raise ValueError("分红输入混入非510300代码")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(
            dividends[column], errors="raise"
        ).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if dividends["ex_date"].duplicated().any():
        raise ValueError("分红除息日重复")
    if (dividends["cash_dividend_per_share"] < 0.0).any():
        raise ValueError("分红金额存在负值")
    if (dividends["payment_date"] < dividends["ex_date"]).any():
        raise ValueError("分红发放日早于除息日")
    return dividends.sort_values("ex_date", kind="mergesort").reset_index(drop=True)


def prepare_total_return_market(
    market: pd.DataFrame, dividends: pd.DataFrame
) -> pd.DataFrame:
    output = market.copy()
    cash_by_ex_date = dividends.groupby("ex_date")["cash_dividend_per_share"].sum()
    output["cash_dividend_per_share"] = output["date"].map(cash_by_ex_date).fillna(0.0)
    output["previous_close"] = output["close"].shift(1)
    output["price_return"] = output["close"].pct_change()
    output["total_return"] = (
        (output["close"] + output["cash_dividend_per_share"])
        / output["previous_close"]
        - 1.0
    )
    output["overnight_contribution"] = (
        output["open"]
        + output["cash_dividend_per_share"]
        - output["previous_close"]
    ) / output["previous_close"]
    output["intraday_contribution"] = (
        output["close"] - output["open"]
    ) / output["previous_close"]
    output["overnight_standalone_return"] = (
        (output["open"] + output["cash_dividend_per_share"])
        / output["previous_close"]
        - 1.0
    )
    output["intraday_standalone_return"] = output["close"] / output["open"] - 1.0
    valid = output["total_return"].notna()
    identity_error = (
        output.loc[valid, "total_return"]
        - output.loc[valid, "overnight_contribution"]
        - output.loc[valid, "intraday_contribution"]
    ).abs()
    if not identity_error.empty and float(identity_error.max()) > 1e-12:
        raise ValueError("总收益的隔夜与日内贡献恒等式失败")
    output["total_wealth_index"] = (1.0 + output["total_return"].fillna(0.0)).cumprod()
    return output


def rolling_prior_percentile(
    values: pd.Series, window: int, minimum_observations: int
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    for index, current in enumerate(numeric):
        if not np.isfinite(current):
            continue
        start = max(0, index - int(window))
        history = numeric[start:index]
        history = history[np.isfinite(history)]
        if len(history) < int(minimum_observations):
            continue
        less = np.count_nonzero(history < current)
        equal = np.count_nonzero(history == current)
        result[index] = (less + 0.5 * equal) / len(history)
    return pd.Series(result, index=values.index, dtype=float)


def cagr_from_total_return(total_return: float, elapsed_days: int) -> float:
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (365.25 / max(int(elapsed_days), 1)) - 1.0)


def index_cagr(
    dates: pd.Series, levels: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="raise").dt.normalize(),
            "level": pd.to_numeric(levels, errors="raise"),
        }
    ).sort_values("date", kind="mergesort")
    selected = frame.loc[frame["date"].between(start, end)].copy()
    if len(selected) < 2:
        raise ValueError("指数基准区间不足两个观测")
    total = float(selected["level"].iloc[-1] / selected["level"].iloc[0] - 1.0)
    elapsed = int((selected["date"].iloc[-1] - selected["date"].iloc[0]).days)
    return cagr_from_total_return(total, elapsed)


def _payment_index_map(
    dates: pd.Series, dividends: pd.DataFrame
) -> tuple[np.ndarray, dict[int, list[int]]]:
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(dates).dt.normalize())
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(normalized_dates)}
    ex_cash = np.zeros(len(normalized_dates), dtype=float)
    payments: dict[int, list[int]] = {}
    for row in dividends.itertuples(index=False):
        ex_date = pd.Timestamp(row.ex_date).normalize()
        payment_date = pd.Timestamp(row.payment_date).normalize()
        if ex_date not in date_to_index:
            continue
        ex_index = date_to_index[ex_date]
        ex_cash[ex_index] += float(row.cash_dividend_per_share)
        payment_index = int(normalized_dates.searchsorted(payment_date, side="left"))
        if payment_index >= len(normalized_dates):
            continue
        payments.setdefault(payment_index, []).append(ex_index)
    return ex_cash, payments


def simulate_binary_execution_targets(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    execution_targets: np.ndarray,
    execution: dict[str, Any],
    *,
    metric_start_index: int,
    commission_rate: float | None = None,
    slippage_bps: float | None = None,
    return_daily: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """批量、逐日、整手模拟510300/现金二元目标仓位。

    `execution_targets[:, i]` 是第i日开盘执行后的目标；调用方必须把收盘信号
    先移到下一交易日。算法与项目日线引擎采用相同的整手、最低佣金、滑点、
    除息确认和发放日现金规则。
    """

    if execution_targets.ndim == 1:
        targets = execution_targets.reshape(1, -1).astype(float, copy=False)
    elif execution_targets.ndim == 2:
        targets = execution_targets.astype(float, copy=False)
    else:
        raise ValueError("执行目标必须是一维或二维数组")
    if targets.shape[1] != len(market):
        raise ValueError("执行目标列数与行情行数不一致")
    if not np.isin(targets, [0.0, 1.0]).all():
        raise ValueError("本研究只允许0或1二元仓位")
    if not 0 <= metric_start_index < len(market):
        raise ValueError("指标起点越界")

    scenario_count, day_count = targets.shape
    initial_cash = float(execution["initial_capital_cny"])
    commission = float(
        execution["commission_rate_per_leg"]
        if commission_rate is None
        else commission_rate
    )
    minimum_commission = float(execution["minimum_commission_cny_per_leg"])
    slippage = float(
        execution["slippage_bps_per_leg"]
        if slippage_bps is None
        else slippage_bps
    )
    lot_size = int(execution["lot_size_shares"])
    stamp_duty_rate = float(execution["stamp_duty_rate"])
    cash_rate = float(execution["cash_annual_rate"])

    opens = market["open"].to_numpy(dtype=float)
    closes = market["close"].to_numpy(dtype=float)
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    ex_cash, payment_map = _payment_index_map(dates, dividends)

    cash = np.full(scenario_count, initial_cash, dtype=float)
    shares = np.zeros(scenario_count, dtype=np.int64)
    receivable = np.zeros(scenario_count, dtype=float)
    scheduled_by_ex_index: dict[int, np.ndarray] = {}
    previous_equity = np.full(scenario_count, initial_cash, dtype=float)
    equity_peak = np.full(scenario_count, initial_cash, dtype=float)
    maximum_drawdown = np.zeros(scenario_count, dtype=float)
    sum_return = np.zeros(scenario_count, dtype=float)
    sum_return_squared = np.zeros(scenario_count, dtype=float)
    metric_count = 0
    exposure_sum = np.zeros(scenario_count, dtype=float)
    trade_count = np.zeros(scenario_count, dtype=np.int64)
    cash_entry_count = np.zeros(scenario_count, dtype=np.int64)
    turnover_notional = np.zeros(scenario_count, dtype=float)
    explicit_cost = np.zeros(scenario_count, dtype=float)
    slippage_cost = np.zeros(scenario_count, dtype=float)
    daily_returns = (
        np.zeros((scenario_count, day_count - metric_start_index), dtype=np.float64)
        if return_daily
        else None
    )

    for day in range(day_count):
        if day > 0 and cash_rate != 0.0:
            cash *= 1.0 + cash_rate / 242.0
        shares_at_start = shares.copy()
        if ex_cash[day] != 0.0:
            entitlement = shares_at_start.astype(float) * ex_cash[day]
            receivable += entitlement
            scheduled_by_ex_index[day] = entitlement

        open_price = opens[day]
        open_equity = cash + shares.astype(float) * open_price + receivable
        target = targets[:, day]
        buy_price = open_price * (1.0 + slippage / 10000.0)
        sell_price = open_price * (1.0 - slippage / 10000.0)
        currently_below_target = target * open_equity >= shares.astype(float) * open_price
        reference = np.where(currently_below_target, buy_price, sell_price)
        desired_shares = (
            np.floor(target * open_equity / reference / lot_size).astype(np.int64)
            * lot_size
        )
        desired_shares = np.maximum(desired_shares, 0)
        quantity = desired_shares - shares

        buy_mask = quantity > 0
        if buy_mask.any():
            buy_quantity = np.where(buy_mask, quantity, 0).astype(np.int64)
            while True:
                notional = buy_quantity.astype(float) * buy_price
                buy_commission = np.where(
                    buy_quantity > 0,
                    np.maximum(minimum_commission, notional * commission),
                    0.0,
                )
                unaffordable = (buy_quantity > 0) & (notional + buy_commission > cash + 1e-9)
                if not unaffordable.any():
                    break
                buy_quantity[unaffordable] -= lot_size
            notional = buy_quantity.astype(float) * buy_price
            buy_commission = np.where(
                buy_quantity > 0,
                np.maximum(minimum_commission, notional * commission),
                0.0,
            )
            cash -= notional + buy_commission
            shares += buy_quantity
            traded = buy_quantity > 0
            trade_count += traded.astype(np.int64)
            turnover_notional += notional
            explicit_cost += buy_commission
            slippage_cost += buy_quantity.astype(float) * (buy_price - open_price)

        sell_mask = quantity < 0
        if sell_mask.any():
            sell_quantity = np.where(sell_mask, -quantity, 0).astype(np.int64)
            sell_quantity = np.minimum(sell_quantity, shares_at_start)
            sell_quantity = sell_quantity // lot_size * lot_size
            notional = sell_quantity.astype(float) * sell_price
            sell_commission = np.where(
                sell_quantity > 0,
                np.maximum(minimum_commission, notional * commission),
                0.0,
            )
            stamp_duty = notional * stamp_duty_rate
            cash += notional - sell_commission - stamp_duty
            shares -= sell_quantity
            traded = sell_quantity > 0
            trade_count += traded.astype(np.int64)
            cash_entry_count += (traded & (shares == 0)).astype(np.int64)
            turnover_notional += notional
            explicit_cost += sell_commission + stamp_duty
            slippage_cost += sell_quantity.astype(float) * (open_price - sell_price)

        for ex_index in payment_map.get(day, []):
            payment = scheduled_by_ex_index.pop(ex_index, np.zeros(scenario_count))
            cash += payment
            receivable -= payment
        receivable[np.abs(receivable) < 1e-9] = 0.0
        equity = cash + shares.astype(float) * closes[day] + receivable

        if day >= metric_start_index:
            daily = equity / previous_equity - 1.0
            sum_return += daily
            sum_return_squared += daily * daily
            metric_count += 1
            equity_peak = np.maximum(equity_peak, equity)
            drawdown = equity / equity_peak - 1.0
            maximum_drawdown = np.minimum(maximum_drawdown, drawdown)
            exposure_sum += np.where(equity > 0.0, shares * closes[day] / equity, 0.0)
            if daily_returns is not None:
                daily_returns[:, day - metric_start_index] = daily
        previous_equity = equity

    if metric_count < 2:
        raise ValueError("模拟指标区间不足两个交易日")
    variance = (
        sum_return_squared - (sum_return * sum_return) / metric_count
    ) / (metric_count - 1)
    variance = np.maximum(variance, 0.0)
    annualized_volatility = np.sqrt(variance) * math.sqrt(242.0)
    annualized_mean = sum_return / metric_count * 242.0
    sharpe = np.divide(
        annualized_mean,
        annualized_volatility,
        out=np.full(scenario_count, np.nan),
        where=annualized_volatility > 0.0,
    )
    total_return = previous_equity / initial_cash - 1.0
    start_date = pd.Timestamp(dates.iloc[metric_start_index])
    end_date = pd.Timestamp(dates.iloc[-1])
    elapsed_days = max(int((end_date - start_date).days), 1)
    cagr = np.array(
        [cagr_from_total_return(float(value), elapsed_days) for value in total_return]
    )
    metrics = pd.DataFrame(
        {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": annualized_volatility,
            "net_sharpe": sharpe,
            "max_drawdown": maximum_drawdown,
            "average_exposure": exposure_sum / metric_count,
            "trade_count": trade_count,
            "cash_entry_count": cash_entry_count,
            "turnover_notional_cny": turnover_notional,
            "turnover_over_initial_capital": turnover_notional / initial_cash,
            "explicit_cost_cny": explicit_cost,
            "slippage_cost_cny": slippage_cost,
            "ending_equity_cny": previous_equity,
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "observations": metric_count,
        }
    )
    return metrics, daily_returns


def execution_targets_from_intervals(
    dates: pd.Series,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    metric_start_index: int,
) -> np.ndarray:
    normalized = pd.DatetimeIndex(pd.to_datetime(dates).dt.normalize())
    target = np.ones(len(normalized), dtype=float)
    target[:metric_start_index] = 0.0
    for start, end in intervals:
        start_value = pd.Timestamp(start).normalize()
        end_value = pd.Timestamp(end).normalize()
        mask = (normalized >= start_value) & (normalized < end_value)
        target[mask] = 0.0
    return target


def maximum_round_trips_per_year(
    dates: pd.Series, execution_targets: np.ndarray
) -> dict[int, int]:
    normalized_dates = pd.to_datetime(dates).reset_index(drop=True)
    target = np.asarray(execution_targets, dtype=float)
    previous = np.r_[target[0], target[:-1]]
    entries = (target == 0.0) & (previous == 1.0)
    result: dict[int, int] = {}
    for year, count in pd.Series(entries.astype(int)).groupby(normalized_dates.dt.year).sum().items():
        result[int(year)] = int(count)
    return result


def holm_bonferroni(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = [(name, float(value)) for name, value in p_values.items() if value is not None]
    valid.sort(key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    total = len(valid)
    for rank, (name, value) in enumerate(valid):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def generated_at() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
