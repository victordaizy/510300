"""510300月末—月初流动性漂移：冻结事件、回测与否证统计。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

from backtest.engine import BacktestCosts, run_long_cash_backtest
from scripts.quality_check_daily import check_daily_data, check_metadata_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "daily_02_tom_liquidity_drift_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_event_episodes(market: pd.DataFrame) -> pd.DataFrame:
    """只根据日期构造最后一日到下月第四日开盘的事件，不读取收益。"""

    dates = pd.DatetimeIndex(pd.to_datetime(market["date"], errors="raise").sort_values().unique())
    by_month: dict[pd.Period, list[pd.Timestamp]] = {}
    for date in dates:
        by_month.setdefault(date.to_period("M"), []).append(pd.Timestamp(date))
    rows: list[dict[str, Any]] = []
    for month in sorted(by_month):
        next_month = month + 1
        if next_month not in by_month or len(by_month[next_month]) < 4:
            continue
        next_dates = by_month[next_month]
        rows.append(
            {
                "episode_id": f"TOM_{month}",
                "month": str(month),
                "entry_date": by_month[month][-1],
                "first_date": next_dates[0],
                "second_date": next_dates[1],
                "third_date": next_dates[2],
                "exit_date": next_dates[3],
            }
        )
    return pd.DataFrame(rows)


def active_interval_map(episodes: pd.DataFrame) -> dict[pd.Timestamp, str]:
    result: dict[pd.Timestamp, str] = {}
    for row in episodes.itertuples(index=False):
        for column in ("entry_date", "first_date", "second_date", "third_date"):
            date = pd.Timestamp(getattr(row, column))
            if date in result:
                raise ValueError(f"换月事件区间重叠：{date.date()}")
            result[date] = str(row.episode_id)
    return result


def build_strategy_targets(
    market: pd.DataFrame,
    episodes: pd.DataFrame,
    normal_exposure: float,
    event_exposure: float,
) -> pd.DataFrame:
    """目标写在前一交易日，下一开盘执行。"""

    dates = list(pd.to_datetime(market["date"], errors="raise"))
    active = active_interval_map(episodes)
    rows: list[dict[str, Any]] = []
    for index, date in enumerate(dates):
        if index + 1 >= len(dates):
            next_date = None
            next_active = False
        else:
            next_date = dates[index + 1]
            next_active = next_date in active
        current_active = date in active
        if next_active and not current_active:
            reason = "TOM_ENTER"
        elif next_active:
            reason = "TOM_HOLD"
        elif current_active:
            reason = "TOM_EXIT"
        else:
            reason = "NORMAL_50"
        rows.append(
            {
                "date": pd.Timestamp(date),
                "target_position": float(event_exposure if next_active else normal_exposure),
                "trade_allowed": True,
                "signal_reason": reason,
                "next_execution_date": pd.Timestamp(next_date) if next_date is not None else pd.NaT,
                "next_event_episode_id": active.get(pd.Timestamp(next_date)) if next_date is not None else None,
            }
        )
    return pd.DataFrame(rows)


def build_static_targets(market: pd.DataFrame, exposure: float) -> pd.DataFrame:
    """首次建仓后不主动再平衡的可执行静态对照。"""

    dates = pd.to_datetime(market["date"], errors="raise")
    result = pd.DataFrame(
        {
            "date": dates,
            "target_position": float(exposure),
            "trade_allowed": False,
            "signal_reason": "STATIC_HOLD",
        }
    )
    if not result.empty:
        result.loc[result.index[0], "trade_allowed"] = True
        result.loc[result.index[0], "signal_reason"] = "STATIC_INITIAL_BUILD"
    return result


def audit_and_load_inputs(
    root: Path = ROOT, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只做数据与日期支持度审计，不计算价格收益。"""

    config = config or load_config()
    contracts = config["data_contracts"]
    errors: list[dict[str, Any]] = []
    for name, contract in contracts.items():
        path = root / contract["file"]
        if not path.exists():
            errors.append({"code": "MISSING_DATA_FILE", "contract": name})
        elif sha256_file(path) != contract["sha256"]:
            errors.append({"code": "DATA_HASH_MISMATCH", "contract": name})

    market = pd.read_parquet(root / contracts["etf_market"]["file"])
    metadata = json.loads((root / contracts["etf_market_metadata"]["file"]).read_text(encoding="utf-8"))
    cross = json.loads((root / contracts["etf_market_cross_check"]["file"]).read_text(encoding="utf-8"))
    coverage = json.loads((root / contracts["distribution_coverage"]["file"]).read_text(encoding="utf-8"))
    settings = yaml.safe_load((root / "config" / "settings.yaml").read_text(encoding="utf-8"))
    data_errors, data_warnings = check_daily_data(market, settings, metadata)
    errors.extend(data_errors)
    errors.extend(
        check_metadata_contract(
            market,
            metadata,
            sha256_file(root / contracts["etf_market"]["file"]),
        )
    )
    market_contract = contracts["etf_market"]
    missing_columns = sorted(set(market_contract["required_columns"]).difference(market.columns))
    if missing_columns:
        errors.append({"code": "MISSING_MARKET_COLUMNS", "columns": missing_columns})
    dates = pd.to_datetime(market["date"], errors="coerce")
    if len(market) != int(market_contract["required_rows"]):
        errors.append({"code": "MARKET_ROW_COUNT_MISMATCH", "actual": int(len(market))})
    if str(dates.min().date()) != market_contract["required_first_date"]:
        errors.append({"code": "MARKET_FIRST_DATE_MISMATCH"})
    if str(dates.max().date()) != market_contract["required_last_date"]:
        errors.append({"code": "MARKET_LAST_DATE_MISMATCH"})
    if int(metadata.get("schema_version", -1)) != int(contracts["etf_market_metadata"]["required_schema_version"]):
        errors.append({"code": "METADATA_SCHEMA_VERSION_MISMATCH"})
    if cross.get("status") != contracts["etf_market_cross_check"]["required_status"]:
        errors.append({"code": "CROSS_SOURCE_STATUS_FAILED"})
    if int(cross.get("overlap_rows", -1)) != int(contracts["etf_market_cross_check"]["required_overlap_rows"]):
        errors.append({"code": "CROSS_SOURCE_ROWS_MISMATCH"})
    if cross.get("primary_sha256") != market_contract["sha256"]:
        errors.append({"code": "CROSS_SOURCE_PRIMARY_HASH_MISMATCH"})
    if not bool(coverage.get("complete_history_confirmed")):
        errors.append({"code": "DIVIDEND_HISTORY_NOT_CONFIRMED"})
    if coverage.get("coverage_end") != contracts["distribution_coverage"]["required_coverage_end"]:
        errors.append({"code": "DIVIDEND_COVERAGE_END_MISMATCH"})

    dividends = pd.read_csv(root / contracts["distributions"]["file"])
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise")
    if len(dividends) != int(contracts["distributions"]["required_event_count"]):
        errors.append({"code": "DIVIDEND_EVENT_COUNT_MISMATCH", "actual": int(len(dividends))})

    cutoff = pd.Timestamp(config["protocol"]["historical_evaluation_end"])
    evaluation_market = market.loc[dates.le(cutoff)].copy().reset_index(drop=True)
    if len(evaluation_market) != int(market_contract["evaluation_rows_through_cutoff"]):
        errors.append({"code": "EVALUATION_ROW_COUNT_MISMATCH", "actual": int(len(evaluation_market))})
    episodes = build_event_episodes(evaluation_market)
    if len(episodes) != int(config["event_definition"]["expected_complete_episodes"]):
        errors.append({"code": "EPISODE_COUNT_MISMATCH", "actual": int(len(episodes))})
    period_counts: dict[str, int] = {}
    for period in config["evaluation"]["subperiods"]:
        mask = pd.to_datetime(episodes["entry_date"]).between(period["start"], period["end"])
        count = int(mask.sum())
        period_counts[period["id"]] = count
        if count != int(period["expected_episodes"]):
            errors.append(
                {"code": "SUBPERIOD_EPISODE_COUNT_MISMATCH", "period": period["id"], "actual": count}
            )
    audit = {
        "status": "PASS" if not errors else "NO_VIEW",
        "data_cutoff": str(cutoff.date()),
        "market_rows": int(len(market)),
        "evaluation_rows": int(len(evaluation_market)),
        "evaluation_first_date": str(pd.to_datetime(evaluation_market["date"]).min().date()),
        "evaluation_last_date": str(pd.to_datetime(evaluation_market["date"]).max().date()),
        "complete_event_episodes": int(len(episodes)),
        "subperiod_episode_counts": period_counts,
        "errors": errors,
        "warnings": data_warnings,
        "real_event_returns_computed": False,
        "strategy_backtest_computed": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    return evaluation_market, dividends, episodes, audit


def build_open_intervals(
    market: pd.DataFrame, dividends: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    """构造开盘到下一开盘的含分红收益；分红归属于权益登记日持有人。"""

    data = market.sort_values("date").reset_index(drop=True).copy()
    data["date"] = pd.to_datetime(data["date"])
    dividend_by_record = (
        dividends.groupby("record_date")["cash_dividend_per_share"].sum().to_dict()
    )
    active = active_interval_map(episodes)
    rows: list[dict[str, Any]] = []
    for index in range(len(data) - 1):
        date = pd.Timestamp(data.loc[index, "date"])
        next_date = pd.Timestamp(data.loc[index + 1, "date"])
        dividend = float(dividend_by_record.get(date, 0.0))
        open_price = float(data.loc[index, "open"])
        next_open = float(data.loc[index + 1, "open"])
        rows.append(
            {
                "date": date,
                "next_date": next_date,
                "month": str(date.to_period("M")),
                "open_total_return": (next_open + dividend) / open_price - 1.0,
                "event_episode_id": active.get(date),
                "is_event_interval": date in active,
            }
        )
    return pd.DataFrame(rows)


def _commission(notional: float, execution: dict[str, Any]) -> float:
    return max(
        float(execution["minimum_commission_cny"]),
        notional * float(execution["commission_rate"]),
    )


def build_event_trade_table(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    episodes: pd.DataFrame,
    execution: dict[str, Any],
    *,
    slippage_bps: float,
) -> pd.DataFrame:
    """以冻结的9,700元增量袖带逐事件计算可执行净损益。"""

    prices = market.set_index(pd.to_datetime(market["date"]))
    capital = float(execution["active_sleeve_test_capital_cny"])
    lot = int(execution["lot_size_shares"])
    daily_cash = float(execution["cash_annual_rate"]) / int(execution["trading_days_per_year"])
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        entry_date = pd.Timestamp(episode.entry_date)
        exit_date = pd.Timestamp(episode.exit_date)
        raw_buy = float(prices.loc[entry_date, "open"])
        raw_sell = float(prices.loc[exit_date, "open"])
        buy_price = raw_buy * (1.0 + slippage_bps / 10000.0)
        sell_price = raw_sell * (1.0 - slippage_bps / 10000.0)
        quantity = int(np.floor(capital / buy_price / lot)) * lot
        while quantity > 0:
            buy_notional = quantity * buy_price
            buy_commission = _commission(buy_notional, execution)
            if buy_notional + buy_commission <= capital + 1e-9:
                break
            quantity -= lot
        if quantity <= 0:
            raise ValueError(f"事件袖带资金不足以买入整手：{episode.episode_id}")
        buy_notional = quantity * buy_price
        sell_notional = quantity * sell_price
        buy_commission = _commission(buy_notional, execution)
        sell_commission = _commission(sell_notional, execution)
        held_dividends = dividends.loc[
            pd.to_datetime(dividends["record_date"]).between(entry_date, exit_date, inclusive="left"),
            "cash_dividend_per_share",
        ].sum()
        dividend_cash = quantity * float(held_dividends)
        cash_remainder = capital - buy_notional - buy_commission
        cash_remainder_end = cash_remainder * (1.0 + daily_cash) ** 4
        end_wealth = cash_remainder_end + sell_notional - sell_commission + dividend_cash
        cash_benchmark = capital * (1.0 + daily_cash) ** 4
        excess_pnl = end_wealth - cash_benchmark
        rows.append(
            {
                **{column: getattr(episode, column) for column in episodes.columns},
                "slippage_bps": float(slippage_bps),
                "capital_cny": capital,
                "quantity": quantity,
                "entry_open": raw_buy,
                "exit_open": raw_sell,
                "buy_execution_price": buy_price,
                "sell_execution_price": sell_price,
                "buy_commission": buy_commission,
                "sell_commission": sell_commission,
                "dividend_cash": dividend_cash,
                "end_wealth": end_wealth,
                "net_return": end_wealth / capital - 1.0,
                "net_excess_return": excess_pnl / capital,
                "net_excess_pnl": excess_pnl,
            }
        )
    return pd.DataFrame(rows)


def profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").dropna().to_numpy(dtype=float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0:
        return None if gains == 0 else float("inf")
    return gains / losses


def newey_west_event_regression(
    intervals: pd.DataFrame, cash_annual_rate: float, trading_days: int, lag: int
) -> dict[str, float]:
    y = intervals["open_total_return"].to_numpy(dtype=float) - cash_annual_rate / trading_days
    event = intervals["is_event_interval"].astype(float).to_numpy()
    x = np.column_stack([np.ones(len(y)), event])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((2, 2), dtype=float)
    for index in range(len(y)):
        vector = x[index][:, None]
        meat += residual[index] ** 2 * (vector @ vector.T)
    for distance in range(1, lag + 1):
        weight = 1.0 - distance / (lag + 1.0)
        cross = np.zeros((2, 2), dtype=float)
        for index in range(distance, len(y)):
            cross += residual[index] * residual[index - distance] * np.outer(x[index], x[index - distance])
        meat += weight * (cross + cross.T)
    covariance = xtx_inv @ meat @ xtx_inv
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    coefficient = float(beta[1])
    t_value = coefficient / standard_error if standard_error > 0 else float("inf")
    return {
        "event_coefficient": coefficient,
        "event_standard_error_hac": standard_error,
        "event_t_value": t_value,
        "event_one_sided_p_value": float(norm.sf(t_value)),
        "event_interval_observations": int(event.sum()),
        "non_event_interval_observations": int(len(event) - event.sum()),
    }


def circular_block_bootstrap_mean(
    values: np.ndarray, *, block_length: int, repetitions: int, random_seed: int
) -> dict[str, float | int]:
    numeric = np.asarray(values, dtype=float)
    if len(numeric) < block_length:
        raise ValueError("区块长度超过事件样本")
    rng = np.random.default_rng(random_seed)
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sample: list[float] = []
        while len(sample) < len(numeric):
            start = int(rng.integers(0, len(numeric)))
            sample.extend(numeric[(start + offset) % len(numeric)] for offset in range(block_length))
        means[repetition] = float(np.mean(sample[: len(numeric)]))
    return {
        "repetitions": repetitions,
        "block_length_episodes": block_length,
        "sample_mean": float(np.mean(numeric)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def event_gross_returns(intervals: pd.DataFrame) -> pd.DataFrame:
    event = intervals.loc[intervals["is_event_interval"]].copy()
    grouped = event.groupby("event_episode_id", sort=False)["open_total_return"]
    result = grouped.apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
    return result.rename("gross_event_return").reset_index()


def run_month_matched_placebo(
    intervals: pd.DataFrame,
    episodes: pd.DataFrame,
    actual_event_returns: pd.DataFrame,
    *,
    repetitions: int,
    random_seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """每个事件月随机抽一个不重叠的连续四区间窗口。"""

    candidates: dict[str, list[float]] = {}
    for episode in episodes.itertuples(index=False):
        month = str(episode.month)
        frame = intervals.loc[intervals["month"].eq(month)].reset_index(drop=True)
        returns: list[float] = []
        for start in range(0, len(frame) - 3):
            block = frame.iloc[start : start + 4]
            if not bool(block["is_event_interval"].any()):
                returns.append(float(np.prod(1.0 + block["open_total_return"].to_numpy(dtype=float)) - 1.0))
        if not returns:
            raise ValueError(f"月份没有合格四日安慰剂窗口：{month}")
        candidates[str(episode.episode_id)] = returns
    rng = np.random.default_rng(random_seed)
    placebo_means = np.empty(repetitions, dtype=float)
    episode_ids = episodes["episode_id"].astype(str).tolist()
    for repetition in range(repetitions):
        selected = [values[int(rng.integers(0, len(values)))] for values in (candidates[key] for key in episode_ids)]
        placebo_means[repetition] = float(np.mean(selected))
    actual_mean = float(actual_event_returns["gross_event_return"].mean())
    percentile = float(np.mean(placebo_means <= actual_mean))
    distribution = pd.DataFrame({"repetition": np.arange(repetitions), "placebo_mean_return": placebo_means})
    return (
        {
            "repetitions": repetitions,
            "actual_event_mean_return": actual_mean,
            "placebo_mean_return": float(placebo_means.mean()),
            "placebo_percentile": percentile,
            "placebo_95_threshold": float(np.quantile(placebo_means, 0.95)),
        },
        distribution,
    )


def run_account_path(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    execution: dict[str, Any],
    *,
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    costs = BacktestCosts(
        commission_rate=float(execution["commission_rate"]),
        minimum_commission_cny=float(execution["minimum_commission_cny"]),
        stamp_duty_rate=float(execution["stamp_duty_rate"]),
        slippage_bps=float(slippage_bps),
        lot_size=int(execution["lot_size_shares"]),
        cash_annual_rate=float(execution["cash_annual_rate"]),
    )
    return run_long_cash_backtest(
        market,
        dividends,
        targets,
        float(execution["initial_capital_cny"]),
        costs,
        pd.to_datetime(market["date"]).min(),
        pd.to_datetime(market["date"]).max(),
        minimum_trade_notional_cny=float(execution["minimum_trade_notional_cny"]),
        minimum_trade_shares=int(execution["minimum_trade_shares"]),
    )


def summarize_account_path(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    execution: dict[str, Any],
) -> dict[str, float | int | str | None]:
    returns = ledger["daily_return"].iloc[1:].to_numpy(dtype=float)
    cash_daily = float(execution["cash_annual_rate"]) / int(execution["trading_days_per_year"])
    excess = returns - cash_daily
    volatility = float(np.std(excess, ddof=1) * np.sqrt(int(execution["trading_days_per_year"])))
    sharpe = float(np.mean(excess) * int(execution["trading_days_per_year"]) / volatility) if volatility > 0 else None
    initial = float(execution["initial_capital_cny"])
    ending = float(ledger["equity"].iloc[-1])
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = ending / initial - 1.0
    cagr = float((ending / initial) ** (365.25 / elapsed_days) - 1.0)
    return {
        "start_date": str(pd.Timestamp(ledger["date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(ledger["date"].iloc[-1]).date()),
        "observations": int(len(ledger)),
        "ending_equity": ending,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_excess_volatility": volatility,
        "sharpe_excess_cash": sharpe,
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_exposure": float(ledger["actual_position"].mean()),
        "trade_count": int(len(trades)),
        "explicit_cost_cny": float(ledger["daily_explicit_cost_cny"].sum()),
        "slippage_cost_cny": float(ledger["daily_slippage_cost_cny"].sum()),
        "total_execution_cost_cny": float(ledger["daily_total_execution_cost_cny"].sum()),
    }


def yearly_contribution_share(event_table: pd.DataFrame) -> tuple[float | None, dict[str, float]]:
    annual = event_table.assign(year=pd.to_datetime(event_table["entry_date"]).dt.year).groupby("year")["net_excess_pnl"].sum()
    positive = annual.clip(lower=0.0)
    denominator = float(positive.sum())
    share = float(positive.max() / denominator) if denominator > 0 else None
    return share, {str(index): float(value) for index, value in annual.items()}


def subperiod_summary(event_table: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(event_table["entry_date"])
    for period in config["evaluation"]["subperiods"]:
        frame = event_table.loc[dates.between(period["start"], period["end"])]
        total = float(frame["net_excess_pnl"].sum())
        rows.append(
            {
                "id": period["id"],
                "episodes": int(len(frame)),
                "net_excess_pnl": total,
                "mean_net_excess_return": float(frame["net_excess_return"].mean()),
                "positive": bool(total > 0),
            }
        )
    return rows


def cost_model_snapshot(execution: dict[str, Any], slippage_bps: float) -> dict[str, Any]:
    costs = BacktestCosts(
        commission_rate=float(execution["commission_rate"]),
        minimum_commission_cny=float(execution["minimum_commission_cny"]),
        stamp_duty_rate=float(execution["stamp_duty_rate"]),
        slippage_bps=float(slippage_bps),
        lot_size=int(execution["lot_size_shares"]),
        cash_annual_rate=float(execution["cash_annual_rate"]),
    )
    return asdict(costs)
