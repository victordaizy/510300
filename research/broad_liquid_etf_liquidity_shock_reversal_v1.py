"""广泛流动 ETF 流动性冲击反转候选的冻结研究实现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from research.multi_asset_annual_excess_40pct_high_sharpe_v1 import (
    _paired_block_bootstrap,
    _rolling_metrics,
    annualized_return,
    annualized_sharpe,
    maximum_drawdown,
)


class DataContractError(RuntimeError):
    """输入、持仓或成交路径不符合冻结合同。"""


@dataclass
class EtfPosition:
    """共享于基础和压力现金账本的同一 ETF 实物持仓。"""

    con_code: str
    shares: int
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    entry_raw_open: float
    entry_total_return_open: float
    entry_gross_notional: float
    prior_median_amount_20: float
    signal_date: pd.Timestamp
    shock_score: float
    amount_ratio: float
    last_total_return_close: float
    delayed_exit_days: int = 0


def load_contract(path: Path) -> dict[str, Any]:
    """读取并校验冻结候选合同。"""

    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """防止目标、费用、信号或执行边界被静默弱化。"""

    protocol = contract.get("protocol", {})
    account = contract.get("account", {})
    partition = contract.get("historical_partition", {})
    universe = contract.get("universe", {})
    signal = contract.get("signal", {})
    portfolio = contract.get("portfolio", {})
    costs = contract.get("costs", {})
    risk = contract.get("risk", {})
    gates = contract.get("visible_gates", {})
    safety = contract.get("safety", {})
    failures: list[str] = []

    if protocol.get("candidate_id") != "BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_V1":
        failures.append("candidate_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V1":
        failures.append("parent_protocol_id")
    if protocol.get("lane") != "BROAD_LIQUID_ETF_CROSS_ASSET":
        failures.append("lane")
    if bool(protocol.get("prior_failed_formula_reused", True)):
        failures.append("prior_failed_formula_reused")
    if bool(protocol.get("prior_failed_family_parameter_rescue", True)):
        failures.append("prior_failed_family_parameter_rescue")
    if bool(protocol.get("outcome_used_to_choose_signal_parameters", True)):
        failures.append("outcome_used_to_choose_signal_parameters")
    if float(account.get("initial_capital_cny", -1.0)) != 500_000.0:
        failures.append("initial_capital_cny")
    if float(account.get("user_transaction_fee_rate_per_leg", -1.0)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    if not bool(partition.get("sealed_replication_may_open_only_after_visible_pass")):
        failures.append("sealed_replication_gate")
    if bool(partition.get("retrospective_history_can_verify_target", True)):
        failures.append("retrospective_history_can_verify_target")
    if int(universe.get("minimum_observed_history_days", -1)) != 252:
        failures.append("minimum_observed_history_days")
    if float(universe.get("minimum_prior_median_amount_20_cny", -1.0)) != 100_000_000.0:
        failures.append("minimum_prior_median_amount_20_cny")
    if int(signal.get("lookback_return_trading_days", -1)) != 5:
        failures.append("lookback_return_trading_days")
    if int(signal.get("volatility_lookback_trading_days", -1)) != 60:
        failures.append("volatility_lookback_trading_days")
    if float(signal.get("maximum_shock_score", 0.0)) != -1.5:
        failures.append("maximum_shock_score")
    if float(signal.get("minimum_signal_amount_to_prior_median_ratio", -1.0)) != 1.0:
        failures.append("minimum_signal_amount_to_prior_median_ratio")
    if int(portfolio.get("maximum_positions", -1)) != 5:
        failures.append("maximum_positions")
    if float(portfolio.get("target_fraction_per_position", -1.0)) != 0.20:
        failures.append("target_fraction_per_position")
    if int(portfolio.get("holding_period_trading_days_open_to_open", -1)) != 5:
        failures.append("holding_period")
    if float(portfolio.get("maximum_pairwise_correlation", -1.0)) != 0.95:
        failures.append("maximum_pairwise_correlation")
    if portfolio.get("blocked_exit_policy") != "CARRY_UNTIL_FIRST_ELIGIBLE_OPEN":
        failures.append("blocked_exit_policy")
    if portfolio.get("terminal_open_positions_policy") != "FAIL_NO_FORCED_OR_FICTITIOUS_EXIT":
        failures.append("terminal_open_positions_policy")
    if float(costs.get("etf_user_fee_rate_per_leg", -1.0)) != 0.0001:
        failures.append("etf_user_fee_rate_per_leg")
    if float(costs.get("conservative_exchange_handling_fee_rate_per_leg", -1.0)) != 0.00004:
        failures.append("exchange_handling_fee_rate")
    if not bool(costs.get("omitted_cost_is_failure")):
        failures.append("omitted_cost_is_failure")
    if float(risk.get("maximum_gross_exposure", -1.0)) != 1.0:
        failures.append("maximum_gross_exposure")
    if float(risk.get("maximum_absolute_net_exposure", -1.0)) != 1.0:
        failures.append("maximum_absolute_net_exposure")
    if bool(risk.get("short_sale_allowed", True)) or bool(risk.get("leverage_allowed", True)):
        failures.append("long_only_unlevered")
    if float(gates.get("minimum_annualized_net_excess", -1.0)) != 0.40:
        failures.append("minimum_annualized_net_excess")
    if float(gates.get("minimum_strategy_net_sharpe", -1.0)) != 1.50:
        failures.append("minimum_strategy_net_sharpe")
    if not bool(gates.get("base_and_stress_must_both_pass")):
        failures.append("base_and_stress_must_both_pass")
    if any(
        bool(safety.get(key, True))
        for key in (
            "paper_position_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety")
    if failures:
        raise ValueError(f"ETF反转候选合同被弱化或损坏：{sorted(failures)}")


def load_etf_inputs(panel_path: Path, master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取所需 ETF 行情字段和基金主表。"""

    panel_columns = {
        "date",
        "con_code",
        "total_return_open",
        "total_return_close",
        "raw_open",
        "raw_close",
        "is_suspended",
        "amount",
    }
    master_columns = {
        "ts_code",
        "name",
        "status",
        "list_date",
        "delist_date",
    }
    panel = pd.read_parquet(panel_path)
    master = pd.read_parquet(master_path)
    panel_missing = panel_columns.difference(panel.columns)
    master_missing = master_columns.difference(master.columns)
    if panel_missing:
        raise DataContractError(f"ETF总收益面板缺少字段：{sorted(panel_missing)}")
    if master_missing:
        raise DataContractError(f"基金主表缺少字段：{sorted(master_missing)}")
    panel = panel[sorted(panel_columns)].copy()
    master = master[sorted(master_columns)].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce").dt.normalize()
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce").dt.normalize()
    if panel["date"].isna().any() or panel.duplicated(["con_code", "date"]).any():
        raise DataContractError("ETF总收益面板含无效日期或重复主键")
    if master["ts_code"].isna().any() or master["ts_code"].duplicated().any():
        raise DataContractError("基金主表含空代码或重复代码")
    return panel, master


def load_benchmark_series(total_return_path: Path, price_path: Path) -> pd.DataFrame:
    """以价格指数开收盘比例把全收益指数收盘桥接为开盘。"""

    total = pd.read_parquet(total_return_path)
    price = pd.read_parquet(price_path)
    if not {"date", "close"}.issubset(total.columns):
        raise DataContractError("H00300全收益输入缺少date或close")
    if not {"date", "open", "close"}.issubset(price.columns):
        raise DataContractError("沪深300价格指数输入缺少date、open或close")
    total = total[["date", "close"]].copy()
    price = price[["date", "open", "close"]].copy()
    total["date"] = pd.to_datetime(total["date"], errors="coerce").dt.normalize()
    price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
    if total["date"].isna().any() or price["date"].isna().any():
        raise DataContractError("基准输入含无效日期")
    if total["date"].duplicated().any() or price["date"].duplicated().any():
        raise DataContractError("基准输入含重复日期")
    total = total.rename(columns={"close": "benchmark_close"})
    price = price.rename(columns={"open": "price_open", "close": "price_close"})
    data = total.merge(price, on="date", how="inner", validate="one_to_one").sort_values("date")
    numeric = ["benchmark_close", "price_open", "price_close"]
    if data[numeric].isna().any().any() or (data[numeric] <= 0.0).any().any():
        raise DataContractError("基准桥接字段缺失或非正")
    data["benchmark_open"] = data["benchmark_close"] * data["price_open"] / data["price_close"]
    return data[["date", "benchmark_open", "benchmark_close"]].reset_index(drop=True)


def _calendar_successor_maps(
    calendar: pd.DatetimeIndex,
    holding_period: int,
) -> tuple[dict[pd.Timestamp, pd.Timestamp], dict[pd.Timestamp, pd.Timestamp]]:
    entry_by_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    exit_by_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    for index, signal_date in enumerate(calendar):
        entry_index = index + 1
        exit_index = entry_index + holding_period
        if exit_index >= len(calendar):
            continue
        entry_by_signal[pd.Timestamp(signal_date)] = pd.Timestamp(calendar[entry_index])
        exit_by_signal[pd.Timestamp(signal_date)] = pd.Timestamp(calendar[exit_index])
    return entry_by_signal, exit_by_signal


def build_liquidity_shock_signals(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只用信号日及此前数据构造下一开盘可执行候选。"""

    validate_contract(contract)
    universe = contract["universe"]
    signal_rules = contract["signal"]
    portfolio = contract["portfolio"]
    data = panel.copy().sort_values(["con_code", "date"]).reset_index(drop=True)
    data = data.loc[data["date"].le(end)].copy()
    group = data.groupby("con_code", sort=False)
    data["log_close"] = np.log(data["total_return_close"].where(data["total_return_close"].gt(0.0)))
    data["log_return_1d"] = group["log_close"].diff()
    return_lookback = int(signal_rules["lookback_return_trading_days"])
    data["log_return_5d"] = data["log_close"] - group["log_close"].shift(return_lookback)
    volatility_lookback = int(signal_rules["volatility_lookback_trading_days"])
    data["volatility_prior_60"] = group["log_return_1d"].transform(
        lambda values: values.shift(1).rolling(
            volatility_lookback,
            min_periods=volatility_lookback,
        ).std(ddof=1)
    )
    amount_lookback = int(signal_rules["amount_lookback_trading_days"])
    data["prior_median_amount_20"] = group["amount"].transform(
        lambda values: values.shift(1).rolling(
            amount_lookback,
            min_periods=amount_lookback,
        ).median()
    )
    observed = (
        data["total_return_close"].gt(0.0)
        & data["raw_close"].gt(0.0)
        & data["amount"].gt(0.0)
        & ~data["is_suspended"].fillna(True)
    )
    data["observed_signal_bar"] = observed
    data["observed_history_count"] = observed.astype(int).groupby(data["con_code"]).cumsum()
    data["prior_raw_close"] = group["raw_close"].shift(1)
    data["shock_score"] = data["log_return_5d"] / (
        data["volatility_prior_60"] * np.sqrt(float(return_lookback))
    )
    data["amount_ratio"] = data["amount"] / data["prior_median_amount_20"]

    metadata = master.rename(columns={"ts_code": "con_code"}).copy()
    allowed_statuses = set(str(value) for value in universe["master_statuses"])
    metadata = metadata.loc[metadata["status"].astype(str).isin(allowed_statuses)].copy()
    data = data.merge(
        metadata[["con_code", "name", "status", "list_date", "delist_date"]],
        on="con_code",
        how="inner",
        validate="many_to_one",
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]).sort_values().unique())
    entry_by_signal, exit_by_signal = _calendar_successor_maps(
        calendar,
        int(portfolio["holding_period_trading_days_open_to_open"]),
    )
    point_in_time = data["date"].ge(data["list_date"])
    point_in_time &= data["delist_date"].isna() | data["date"].lt(data["delist_date"])
    signal_eligible = (
        data["date"].isin(entry_by_signal)
        & point_in_time
        & data["observed_history_count"].ge(int(universe["minimum_observed_history_days"]))
        & data["observed_signal_bar"]
        & data["raw_close"].ge(float(universe["minimum_raw_price_cny"]))
        & data["prior_median_amount_20"].ge(
            float(universe["minimum_prior_median_amount_20_cny"])
        )
        & data["shock_score"].le(float(signal_rules["maximum_shock_score"]))
        & data["log_return_1d"].lt(0.0)
        & data["log_return_5d"].lt(0.0)
        & data["amount_ratio"].ge(
            float(signal_rules["minimum_signal_amount_to_prior_median_ratio"])
        )
    )
    signals = data.loc[
        signal_eligible,
        [
            "date",
            "con_code",
            "name",
            "raw_close",
            "shock_score",
            "log_return_1d",
            "log_return_5d",
            "volatility_prior_60",
            "prior_median_amount_20",
            "amount_ratio",
        ],
    ].rename(columns={"date": "signal_date", "raw_close": "signal_raw_close"})
    signals["entry_date"] = signals["signal_date"].map(entry_by_signal)
    signals["scheduled_exit_date"] = signals["signal_date"].map(exit_by_signal)
    signals = signals.loc[
        signals["entry_date"].between(start, end, inclusive="both")
        & signals["scheduled_exit_date"].le(end)
    ].copy()

    entry = data[
        [
            "date",
            "con_code",
            "raw_open",
            "total_return_open",
            "amount",
            "is_suspended",
        ]
    ].rename(
        columns={
            "date": "entry_date",
            "raw_open": "entry_raw_open",
            "total_return_open": "entry_total_return_open",
            "amount": "entry_amount",
            "is_suspended": "entry_is_suspended",
        }
    )
    signals = signals.merge(
        entry,
        on=["entry_date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    signals["entry_gap"] = signals["entry_raw_open"] / signals["signal_raw_close"] - 1.0
    entry_eligible = (
        signals["entry_raw_open"].ge(float(universe["minimum_raw_price_cny"]))
        & signals["entry_total_return_open"].gt(0.0)
        & signals["entry_amount"].gt(0.0)
        & ~signals["entry_is_suspended"].fillna(True)
        & signals["entry_gap"].gt(float(portfolio["entry_open_gap_lower_bound"]))
        & signals["entry_gap"].lt(float(portfolio["entry_open_gap_upper_bound"]))
    )
    raw_signal_count = int(len(signals))
    signals = signals.loc[entry_eligible].copy()
    signals.sort_values(
        ["entry_date", "shock_score", "prior_median_amount_20", "con_code"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    signals.reset_index(drop=True, inplace=True)

    market = data[
        [
            "date",
            "con_code",
            "total_return_open",
            "total_return_close",
            "raw_open",
            "raw_close",
            "prior_raw_close",
            "amount",
            "is_suspended",
        ]
    ].copy()
    return_wide = data.pivot(index="date", columns="con_code", values="log_return_1d").sort_index()
    by_year = signals.groupby(signals["signal_date"].dt.year).size()
    audit = {
        "signal_rows_before_entry_filter": raw_signal_count,
        "executable_signal_rows": int(len(signals)),
        "signal_day_count": int(signals["signal_date"].nunique()),
        "entry_day_count": int(signals["entry_date"].nunique()),
        "unique_etf_count": int(signals["con_code"].nunique()),
        "signal_rows_by_year": {str(int(year)): int(count) for year, count in by_year.items()},
        "first_signal_date": None
        if signals.empty
        else signals["signal_date"].min().date().isoformat(),
        "last_signal_date": None
        if signals.empty
        else signals["signal_date"].max().date().isoformat(),
        "future_return_used_in_signal_construction": False,
    }
    return signals, market, return_wide, audit


def etf_cost_rate(contract: dict[str, Any], *, scenario: str) -> float:
    """返回单条 ETF 成交腿的冻结总比例成本。"""

    if scenario not in {"base", "stress"}:
        raise ValueError(f"未知成本场景：{scenario}")
    costs = contract["costs"]
    slippage = float(costs[f"{scenario}_slippage_bps_per_leg"]) / 10_000.0
    impact = float(costs[f"{scenario}_market_impact_bps_per_leg"]) / 10_000.0
    return (
        float(costs["etf_user_fee_rate_per_leg"])
        + float(costs["conservative_exchange_handling_fee_rate_per_leg"])
        + slippage
        + impact
        + float(costs["stamp_duty_rate"])
    )


def _pair_correlation(
    return_wide: pd.DataFrame,
    left: str,
    right: str,
    signal_date: pd.Timestamp,
    *,
    lookback: int,
    minimum_observations: int,
) -> float | None:
    """使用信号日及此前的配对观测计算历史相关性。"""

    if left not in return_wide.columns or right not in return_wide.columns:
        return None
    window = return_wide.loc[return_wide.index <= signal_date, [left, right]].tail(lookback)
    paired = window.dropna()
    if len(paired) < minimum_observations:
        return None
    correlation = float(paired[left].corr(paired[right]))
    return correlation if np.isfinite(correlation) else None


def run_portfolio_backtest(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    return_wide: pd.DataFrame,
    contract: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按下一开盘成交、五日持有和受阻退出规则构建逐日净值。"""

    validate_contract(contract)
    account = contract["account"]
    universe = contract["universe"]
    portfolio = contract["portfolio"]
    risk = contract["risk"]
    calendar_frame = benchmark.loc[
        benchmark["date"].between(start, end, inclusive="both")
    ].sort_values("date").reset_index(drop=True)
    if calendar_frame.empty:
        raise DataContractError("评价区间没有基准交易日")
    market_index = market.set_index(["con_code", "date"]).sort_index()
    if not market_index.index.is_unique:
        raise DataContractError("ETF行情主键重复")
    signals_by_entry = {
        pd.Timestamp(date): group.sort_values(
            ["shock_score", "prior_median_amount_20", "con_code"],
            ascending=[True, False, True],
        ).copy()
        for date, group in signals.groupby("entry_date", sort=True)
    } if not signals.empty else {}

    initial_capital = float(account["initial_capital_cny"])
    cash = {"base": initial_capital, "stress": initial_capital}
    prior_nav = {"base": initial_capital, "stress": initial_capital}
    daily_cash_factor = (1.0 + float(account["cash_annual_rate"])) ** (
        1.0 / int(contract["visible_gates"]["annualization_trading_days"])
    )
    positions: list[EtfPosition] = []
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prior_benchmark_close: float | None = None
    maximum_capacity_fraction = 0.0
    maximum_gross_exposure = 0.0
    maximum_net_exposure = 0.0
    maximum_position_count = 0
    blocked_exit_attempt_count = 0
    stale_mark_count = 0
    correlation_rejection_count = 0
    missing_correlation_rejection_count = 0
    entry_candidate_count = 0
    entry_transaction_count = 0
    exit_transaction_count = 0

    def get_market_row(code: str, date: pd.Timestamp) -> pd.Series | None:
        key = (code, date)
        if key not in market_index.index:
            return None
        row = market_index.loc[key]
        if isinstance(row, pd.DataFrame):
            raise DataContractError(f"ETF行情主键不唯一：{code} {date.date()}")
        return row

    def marked_value(position: EtfPosition, date: pd.Timestamp, field: str) -> tuple[float, bool]:
        row = get_market_row(position.con_code, date)
        if row is not None:
            value = float(row[field])
            if np.isfinite(value) and value > 0.0:
                return (
                    position.entry_gross_notional
                    * value
                    / position.entry_total_return_open,
                    False,
                )
        return (
            position.entry_gross_notional
            * position.last_total_return_close
            / position.entry_total_return_open,
            True,
        )

    for day_index, benchmark_row in calendar_frame.iterrows():
        date = pd.Timestamp(benchmark_row["date"])
        if day_index > 0:
            cash["base"] *= daily_cash_factor
            cash["stress"] *= daily_cash_factor

        remaining_positions: list[EtfPosition] = []
        for position in positions:
            if date < position.scheduled_exit_date:
                remaining_positions.append(position)
                continue
            row = get_market_row(position.con_code, date)
            exit_allowed = False
            exit_gap: float | None = None
            if row is not None:
                prior_raw_close = float(row["prior_raw_close"])
                raw_open = float(row["raw_open"])
                total_return_open = float(row["total_return_open"])
                if np.isfinite(prior_raw_close) and prior_raw_close > 0.0:
                    exit_gap = raw_open / prior_raw_close - 1.0
                exit_allowed = bool(
                    not bool(row["is_suspended"])
                    and float(row["amount"]) > 0.0
                    and np.isfinite(raw_open)
                    and raw_open > 0.0
                    and np.isfinite(total_return_open)
                    and total_return_open > 0.0
                    and exit_gap is not None
                    and exit_gap > float(portfolio["exit_open_gap_lower_bound"])
                )
            if not exit_allowed:
                position.delayed_exit_days += 1
                blocked_exit_attempt_count += 1
                remaining_positions.append(position)
                continue
            gross, used_stale = marked_value(position, date, "total_return_open")
            if used_stale:
                raise DataContractError(
                    f"允许退出但缺少有效复权开盘：{position.con_code} {date.date()}"
                )
            for scenario in ("base", "stress"):
                cash[scenario] += gross * (1.0 - etf_cost_rate(contract, scenario=scenario))
            exit_transaction_count += 1
            trade_rows.append(
                {
                    "trade_date": date,
                    "side": "SELL",
                    "con_code": position.con_code,
                    "shares": position.shares,
                    "gross_notional_cny": gross,
                    "signal_date": position.signal_date,
                    "scheduled_exit_date": position.scheduled_exit_date,
                    "delayed_exit_days": position.delayed_exit_days,
                    "shock_score": position.shock_score,
                    "amount_ratio": position.amount_ratio,
                    "capacity_fraction": gross / position.prior_median_amount_20,
                    "open_gap": exit_gap,
                }
            )
        positions = remaining_positions

        opening_position_value = 0.0
        for position in positions:
            value, _ = marked_value(position, date, "total_return_open")
            opening_position_value += value
        reference_open_nav = min(
            cash["base"] + opening_position_value,
            cash["stress"] + opening_position_value,
        )
        if reference_open_nav <= 0.0:
            raise DataContractError(f"开盘参考净值非正：{date.date()}")

        candidates = signals_by_entry.get(date, pd.DataFrame())
        active_codes = [position.con_code for position in positions]
        selected_today: list[str] = []
        available_slots = int(portfolio["maximum_positions"]) - len(positions)
        if available_slots > 0 and not candidates.empty:
            for candidate in candidates.itertuples(index=False):
                entry_candidate_count += 1
                code = str(candidate.con_code)
                if code in active_codes or code in selected_today:
                    continue
                correlated = False
                insufficient_pair = False
                for peer in active_codes + selected_today:
                    correlation = _pair_correlation(
                        return_wide,
                        code,
                        peer,
                        pd.Timestamp(candidate.signal_date),
                        lookback=int(
                            portfolio["diversification_correlation_lookback_trading_days"]
                        ),
                        minimum_observations=int(
                            portfolio["diversification_minimum_pair_observations"]
                        ),
                    )
                    if correlation is None:
                        insufficient_pair = True
                        break
                    if correlation >= float(portfolio["maximum_pairwise_correlation"]):
                        correlated = True
                        break
                if insufficient_pair:
                    missing_correlation_rejection_count += 1
                    continue
                if correlated:
                    correlation_rejection_count += 1
                    continue
                row = get_market_row(code, date)
                if row is None:
                    continue
                raw_open = float(row["raw_open"])
                total_return_open = float(row["total_return_open"])
                target = reference_open_nav * float(portfolio["target_fraction_per_position"])
                board_lot = int(portfolio["board_lot_shares"])
                shares = int(np.floor(target / raw_open / board_lot) * board_lot)
                if shares <= 0:
                    continue
                gross = shares * raw_open
                if gross < float(portfolio["minimum_trade_notional_cny"]):
                    continue
                capacity_fraction = gross / float(candidate.prior_median_amount_20)
                if capacity_fraction > float(
                    universe["maximum_order_fraction_of_prior_median_amount"]
                ) + 1e-12:
                    raise DataContractError(f"目标金额超过冻结容量：{code} {date.date()}")
                stress_rate = etf_cost_rate(contract, scenario="stress")
                if gross * (1.0 + stress_rate) > cash["stress"]:
                    affordable_lots = int(
                        np.floor(cash["stress"] / (raw_open * (1.0 + stress_rate)) / board_lot)
                    )
                    shares = affordable_lots * board_lot
                    gross = shares * raw_open
                    capacity_fraction = gross / float(candidate.prior_median_amount_20)
                if shares <= 0 or gross < float(portfolio["minimum_trade_notional_cny"]):
                    continue
                for scenario in ("base", "stress"):
                    cash[scenario] -= gross * (1.0 + etf_cost_rate(contract, scenario=scenario))
                if min(cash.values()) < -1e-8:
                    raise DataContractError(f"ETF买入导致现金为负：{code} {date.date()}")
                positions.append(
                    EtfPosition(
                        con_code=code,
                        shares=shares,
                        entry_date=date,
                        scheduled_exit_date=pd.Timestamp(candidate.scheduled_exit_date),
                        entry_raw_open=raw_open,
                        entry_total_return_open=total_return_open,
                        entry_gross_notional=gross,
                        prior_median_amount_20=float(candidate.prior_median_amount_20),
                        signal_date=pd.Timestamp(candidate.signal_date),
                        shock_score=float(candidate.shock_score),
                        amount_ratio=float(candidate.amount_ratio),
                        last_total_return_close=total_return_open,
                    )
                )
                selected_today.append(code)
                entry_transaction_count += 1
                maximum_capacity_fraction = max(maximum_capacity_fraction, capacity_fraction)
                trade_rows.append(
                    {
                        "trade_date": date,
                        "side": "BUY",
                        "con_code": code,
                        "shares": shares,
                        "gross_notional_cny": gross,
                        "signal_date": pd.Timestamp(candidate.signal_date),
                        "scheduled_exit_date": pd.Timestamp(candidate.scheduled_exit_date),
                        "delayed_exit_days": 0,
                        "shock_score": float(candidate.shock_score),
                        "amount_ratio": float(candidate.amount_ratio),
                        "capacity_fraction": capacity_fraction,
                        "open_gap": float(candidate.entry_gap),
                    }
                )
                if len(selected_today) >= available_slots:
                    break

        stock_close_value = 0.0
        for position in positions:
            row = get_market_row(position.con_code, date)
            close_value, used_stale = marked_value(position, date, "total_return_close")
            if used_stale:
                stale_mark_count += 1
            elif row is not None:
                total_return_close = float(row["total_return_close"])
                if np.isfinite(total_return_close) and total_return_close > 0.0:
                    position.last_total_return_close = total_return_close
            stock_close_value += close_value

        if date == pd.Timestamp(calendar_frame["date"].iloc[-1]) and positions:
            codes = sorted(position.con_code for position in positions)
            raise DataContractError(f"评价区间末仍有未退出持仓：{codes}")
        nav = {
            scenario: cash[scenario] + stock_close_value
            for scenario in ("base", "stress")
        }
        if min(nav.values()) <= 0.0:
            raise DataContractError(f"组合净值非正：{date.date()}")
        reference_nav = min(nav.values())
        gross_exposure = stock_close_value / reference_nav
        net_exposure = gross_exposure
        maximum_gross_exposure = max(maximum_gross_exposure, gross_exposure)
        maximum_net_exposure = max(maximum_net_exposure, abs(net_exposure))
        maximum_position_count = max(maximum_position_count, len(positions))
        if gross_exposure > float(risk["maximum_gross_exposure"]) + 1e-12:
            raise DataContractError(f"毛敞口超过上限：{date.date()} {gross_exposure}")
        if abs(net_exposure) > float(risk["maximum_absolute_net_exposure"]) + 1e-12:
            raise DataContractError(f"净敞口超过上限：{date.date()} {net_exposure}")

        benchmark_close = float(benchmark_row["benchmark_close"])
        if prior_benchmark_close is None:
            benchmark_return = benchmark_close / float(benchmark_row["benchmark_open"]) - 1.0
        else:
            benchmark_return = benchmark_close / prior_benchmark_close - 1.0
        daily_rows.append(
            {
                "trade_date": date,
                "strategy_base_net_return": nav["base"] / prior_nav["base"] - 1.0,
                "strategy_stress_net_return": nav["stress"] / prior_nav["stress"] - 1.0,
                "benchmark_total_return": benchmark_return,
                "base_nav_cny": nav["base"],
                "stress_nav_cny": nav["stress"],
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "position_count": len(positions),
                "quality_complete": True,
                "capacity_pass": True,
            }
        )
        prior_nav = nav
        prior_benchmark_close = benchmark_close

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    audit = {
        "initial_capital_cny": initial_capital,
        "final_base_nav_cny": float(daily["base_nav_cny"].iloc[-1]),
        "final_stress_nav_cny": float(daily["stress_nav_cny"].iloc[-1]),
        "entry_candidate_count": int(entry_candidate_count),
        "entry_transaction_count": int(entry_transaction_count),
        "exit_transaction_count": int(exit_transaction_count),
        "correlation_rejection_count": int(correlation_rejection_count),
        "missing_correlation_rejection_count": int(missing_correlation_rejection_count),
        "blocked_exit_attempt_count": int(blocked_exit_attempt_count),
        "stale_mark_count": int(stale_mark_count),
        "maximum_capacity_fraction": float(maximum_capacity_fraction),
        "maximum_gross_exposure": float(maximum_gross_exposure),
        "maximum_net_exposure": float(maximum_net_exposure),
        "maximum_position_count": int(maximum_position_count),
        "historical_bid_ask_observed": False,
        "execution_cost_status": "MODELED_BASE_AND_STRESS_COSTS",
    }
    return daily, trades, audit


def evaluate_historical_returns(
    daily: pd.DataFrame,
    contract: dict[str, Any],
    *,
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """按父协议的 40 个百分点和高夏普门槛评价历史路径。"""

    required = {
        "trade_date",
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
        "gross_exposure",
        "net_exposure",
        "quality_complete",
        "capacity_pass",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise DataContractError(f"历史收益缺少字段：{sorted(missing)}")
    data = daily.copy().sort_values("trade_date").reset_index(drop=True)
    if data.empty or data["trade_date"].duplicated().any():
        raise DataContractError("历史收益为空或日期重复")
    numeric = [
        "strategy_base_net_return",
        "strategy_stress_net_return",
        "benchmark_total_return",
        "gross_exposure",
        "net_exposure",
    ]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise DataContractError("历史收益或敞口含非有限值")
    if (data[["strategy_base_net_return", "strategy_stress_net_return", "benchmark_total_return"]] <= -1.0).any().any():
        raise DataContractError("历史日收益不得小于等于-100%")
    gates_config = contract["visible_gates"]
    if bool(gates_config["data_quality_complete_required"]) and not data[
        "quality_complete"
    ].astype(bool).all():
        raise DataContractError("历史路径存在不完整数据")
    if bool(gates_config["capacity_pass_required"]) and not data["capacity_pass"].astype(bool).all():
        raise DataContractError("历史路径容量失败")

    trading_days = int(gates_config["annualization_trading_days"])
    cash_rate = float(contract["account"]["cash_annual_rate"])
    benchmark_returns = data["benchmark_total_return"].astype(float)
    base_returns = data["strategy_base_net_return"].astype(float)
    stress_returns = data["strategy_stress_net_return"].astype(float)
    benchmark_cagr = annualized_return(benchmark_returns, trading_days)
    base_cagr = annualized_return(base_returns, trading_days)
    stress_cagr = annualized_return(stress_returns, trading_days)
    base_excess = base_cagr - benchmark_cagr
    stress_excess = stress_cagr - benchmark_cagr
    base_sharpe = annualized_sharpe(
        base_returns,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
    )
    stress_sharpe = annualized_sharpe(
        stress_returns,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
    )
    window = int(gates_config["rolling_window_trading_days"])
    base_rolling_excess, base_rolling_sharpe = _rolling_metrics(
        base_returns,
        benchmark_returns,
        window=window,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
    )
    stress_rolling_excess, stress_rolling_sharpe = _rolling_metrics(
        stress_returns,
        benchmark_returns,
        window=window,
        trading_days_per_year=trading_days,
        cash_annual_rate=cash_rate,
    )
    bootstrap = gates_config["bootstrap"]
    repetitions = (
        int(bootstrap_repetitions_override)
        if bootstrap_repetitions_override is not None
        else int(bootstrap["repetitions"])
    )
    common_bootstrap = {
        "repetitions": repetitions,
        "block_length": int(bootstrap["block_length_trading_days"]),
        "trading_days_per_year": trading_days,
        "cash_annual_rate": cash_rate,
        "lower_quantile": float(bootstrap["lower_quantile"]),
        "upper_quantile": 1.0 - float(bootstrap["lower_quantile"]),
    }
    base_bootstrap = _paired_block_bootstrap(
        base_returns,
        benchmark_returns,
        random_seed=int(bootstrap["random_seed"]),
        **common_bootstrap,
    )
    stress_bootstrap = _paired_block_bootstrap(
        stress_returns,
        benchmark_returns,
        random_seed=int(bootstrap["random_seed"]) + 1,
        **common_bootstrap,
    )

    target = float(gates_config["minimum_annualized_net_excess"])
    sharpe_target = float(gates_config["minimum_strategy_net_sharpe"])
    block_length = int(gates_config["rolling_window_trading_days"])
    year_blocks: list[dict[str, Any]] = []
    for block_start in range(0, len(data) - block_length + 1, block_length):
        block_stop = block_start + block_length
        benchmark_block = benchmark_returns.iloc[block_start:block_stop]
        base_block = base_returns.iloc[block_start:block_stop]
        stress_block = stress_returns.iloc[block_start:block_stop]
        benchmark_block_return = annualized_return(benchmark_block, trading_days)
        base_block_return = annualized_return(base_block, trading_days)
        stress_block_return = annualized_return(stress_block, trading_days)
        base_block_sharpe = annualized_sharpe(
            base_block,
            trading_days_per_year=trading_days,
            cash_annual_rate=cash_rate,
        )
        stress_block_sharpe = annualized_sharpe(
            stress_block,
            trading_days_per_year=trading_days,
            cash_annual_rate=cash_rate,
        )
        qualified = bool(
            base_block_return - benchmark_block_return >= target
            and stress_block_return - benchmark_block_return >= target
            and base_block_sharpe >= sharpe_target
            and stress_block_sharpe >= sharpe_target
        )
        year_blocks.append(
            {
                "start": pd.Timestamp(data["trade_date"].iloc[block_start]).date().isoformat(),
                "end": pd.Timestamp(data["trade_date"].iloc[block_stop - 1]).date().isoformat(),
                "base_annualized_excess": base_block_return - benchmark_block_return,
                "stress_annualized_excess": stress_block_return - benchmark_block_return,
                "base_sharpe": base_block_sharpe,
                "stress_sharpe": stress_block_sharpe,
                "target_qualified": qualified,
            }
        )
    qualified_blocks = sum(bool(item["target_qualified"]) for item in year_blocks)
    base_rolling_excess_median = float(base_rolling_excess.median())
    stress_rolling_excess_median = float(stress_rolling_excess.median())
    base_rolling_sharpe_median = float(base_rolling_sharpe.median())
    stress_rolling_sharpe_median = float(stress_rolling_sharpe.median())
    excess_floor = float(bootstrap["minimum_excess_lower_bound"])
    bootstrap_sharpe_floor = float(bootstrap["minimum_sharpe_lower_bound"])
    gates = {
        "base_annualized_excess_at_least_40pct": base_excess >= target,
        "stress_annualized_excess_at_least_40pct": stress_excess >= target,
        "base_strategy_sharpe_at_least_1_5": base_sharpe >= sharpe_target,
        "stress_strategy_sharpe_at_least_1_5": stress_sharpe >= sharpe_target,
        "base_rolling_excess_median_at_least_40pct": base_rolling_excess_median
        >= float(gates_config["minimum_rolling_excess_median"]),
        "stress_rolling_excess_median_at_least_40pct": stress_rolling_excess_median
        >= float(gates_config["minimum_rolling_excess_median"]),
        "base_rolling_sharpe_median_at_least_1_5": base_rolling_sharpe_median
        >= float(gates_config["minimum_rolling_sharpe_median"]),
        "stress_rolling_sharpe_median_at_least_1_5": stress_rolling_sharpe_median
        >= float(gates_config["minimum_rolling_sharpe_median"]),
        "base_bootstrap_excess_lower_bound_positive": base_bootstrap[
            "annualized_excess_interval_95pct"
        ][0]
        > excess_floor,
        "stress_bootstrap_excess_lower_bound_positive": stress_bootstrap[
            "annualized_excess_interval_95pct"
        ][0]
        > excess_floor,
        "base_bootstrap_sharpe_lower_bound_at_least_1": base_bootstrap[
            "strategy_sharpe_interval_95pct"
        ][0]
        >= bootstrap_sharpe_floor,
        "stress_bootstrap_sharpe_lower_bound_at_least_1": stress_bootstrap[
            "strategy_sharpe_interval_95pct"
        ][0]
        >= bootstrap_sharpe_floor,
        "minimum_non_overlapping_year_blocks": len(year_blocks)
        >= int(gates_config["minimum_non_overlapping_242d_blocks"]),
        "minimum_target_qualified_year_blocks": qualified_blocks
        >= int(gates_config["minimum_target_qualified_blocks"]),
        "data_quality_complete": bool(data["quality_complete"].astype(bool).all()),
        "capacity_pass": bool(data["capacity_pass"].astype(bool).all()),
    }
    return {
        "eligible_historical_day_count": int(len(data)),
        "first_trade_date": pd.Timestamp(data["trade_date"].iloc[0]).date().isoformat(),
        "last_trade_date": pd.Timestamp(data["trade_date"].iloc[-1]).date().isoformat(),
        "metrics": {
            "benchmark_total_return_cagr": benchmark_cagr,
            "strategy_base_net_cagr": base_cagr,
            "strategy_stress_net_cagr": stress_cagr,
            "base_annualized_excess": base_excess,
            "stress_annualized_excess": stress_excess,
            "base_strategy_net_sharpe": base_sharpe,
            "stress_strategy_net_sharpe": stress_sharpe,
            "base_maximum_drawdown": maximum_drawdown(base_returns),
            "stress_maximum_drawdown": maximum_drawdown(stress_returns),
            "base_rolling_242d_excess_median": base_rolling_excess_median,
            "stress_rolling_242d_excess_median": stress_rolling_excess_median,
            "base_rolling_242d_sharpe_median": base_rolling_sharpe_median,
            "stress_rolling_242d_sharpe_median": stress_rolling_sharpe_median,
            "base_bootstrap": base_bootstrap,
            "stress_bootstrap": stress_bootstrap,
            "non_overlapping_years": year_blocks,
            "target_qualified_year_block_count": qualified_blocks,
        },
        "gates": gates,
        "all_visible_gates_pass": bool(all(gates.values())),
    }
