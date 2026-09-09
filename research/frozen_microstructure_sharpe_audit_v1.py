"""既有510300微观结构冻结仓位序列的2万元净夏普审计。

本模块只复算既有、已见结果的冻结状态，不发现新规则，不生成信号、持仓或订单。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1.yaml"
MANIFEST_PATH = (
    PROJECT_ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_manifest.json"
)


def sha256_file(path: Path) -> str:
    """计算文件的SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取并校验冻结审计配置。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload["protocol"]["project_id"] != "510300_FROZEN_MICROSTRUCTURE_SHARPE_AUDIT_V1":
        raise ValueError("审计项目ID不匹配")
    if payload["scope"]["execution_asset"] != "510300.SH":
        raise ValueError("执行资产必须严格为510300.SH")
    if payload["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ValueError("允许持仓必须严格为510300和现金")
    if payload["scope"]["allowed_states"] != [0, 1]:
        raise ValueError("仓位状态必须严格为0或1")
    if float(payload["execution"]["initial_capital_cny"]) != 20000.0:
        raise ValueError("初始资金必须冻结为2万元")
    if int(payload["execution"]["lot_size_shares"]) != 100:
        raise ValueError("交易单位必须冻结为100股")
    if float(payload["execution"]["cash_annual_rate"]) != 0.0:
        raise ValueError("现金收益必须冻结为0")
    if payload["execution"]["start_perturbations_trading_days"] != [0, 1, 2, 3, 4]:
        raise ValueError("起点扰动必须严格为0至4交易日")
    return payload


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """验证实现、配置和全部输入哈希未漂移。"""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("冻结清单不存在，禁止运行审计")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("implementation_frozen", False):
        raise ValueError("冻结清单未标记implementation_frozen")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in manifest["tracked_files"].items():
        path = PROJECT_ROOT / Path(relative)
        actual = "MISSING" if not path.exists() else sha256_file(path)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    for relative, expected in manifest["input_files"].items():
        path = PROJECT_ROOT / Path(relative)
        actual = "MISSING" if not path.exists() else sha256_file(path)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"冻结哈希漂移，禁止审计：{json.dumps(mismatches, ensure_ascii=False)}")
    if sha256_file(CONFIG_PATH) != manifest["config_sha256"]:
        raise ValueError("配置哈希与冻结清单不一致")
    return manifest


def _normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result.sort_values(column).drop_duplicates(column, keep="last").reset_index(drop=True)


def _combine_market(
    historical_path: Path,
    post_cutoff_path: Path,
    *,
    price_prefix: str,
) -> pd.DataFrame:
    historical = _normalize_date(pd.read_parquet(historical_path))
    post_cutoff = _normalize_date(pd.read_parquet(post_cutoff_path))
    combined = _normalize_date(pd.concat([historical, post_cutoff], ignore_index=True))
    earliest = pd.Timestamp("2021-01-01")
    combined = combined.loc[combined["date"].ge(earliest)].copy()
    required = {"date", "open", "close"} if price_prefix == "etf" else {"date", "close"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"{price_prefix}行情缺少字段：{sorted(missing)}")
    for column in required.difference({"date"}):
        combined[column] = pd.to_numeric(combined[column], errors="raise")
        if combined[column].isna().any() or (combined[column] <= 0.0).any():
            raise ValueError(f"{price_prefix}行情字段{column}存在无效值")
    return combined


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """只形成2021年及以后可进入审计的行情、状态和分红。"""

    inputs = config["inputs"]
    signals = _normalize_date(
        pd.read_parquet(PROJECT_ROOT / inputs["frozen_signal_states"]["path"])
    )
    state_column = inputs["frozen_signal_states"]["state_column"]
    if state_column not in signals.columns:
        raise ValueError("冻结信号文件缺少state_final")
    signals = signals.loc[:, ["date", state_column]].copy()
    if signals["date"].min() < pd.Timestamp(config["scope"]["earliest_allowed_source_date"]):
        raise ValueError("冻结信号文件包含2021年以前日期，违反本审计范围")
    states = pd.to_numeric(signals[state_column], errors="raise")
    if not set(states.astype(int).unique()).issubset({0, 1}):
        raise ValueError("冻结状态出现0/1以外值")
    signals[state_column] = states.astype(np.int8)

    etf = _combine_market(
        PROJECT_ROOT / inputs["historical_etf_market"]["path"],
        PROJECT_ROOT / inputs["post_cutoff_etf_market"]["path"],
        price_prefix="etf",
    )
    benchmark = _combine_market(
        PROJECT_ROOT / inputs["historical_benchmark"]["path"],
        PROJECT_ROOT / inputs["post_cutoff_benchmark"]["path"],
        price_prefix="benchmark",
    )
    market = etf.loc[:, ["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    market = market.merge(
        benchmark.loc[:, ["date", "close"]].rename(columns={"close": "benchmark_close"}),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    market = market.merge(signals, on="date", how="left", validate="one_to_one")
    market["execution_state"] = market[state_column].shift(1)
    market = market.loc[market["date"].le(pd.Timestamp(config["dates"]["primary_evaluation_end"]))]
    market = market.reset_index(drop=True)

    dividends = pd.read_csv(PROJECT_ROOT / inputs["dividends"]["path"])
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends = dividends.loc[
        dividends["ex_date"].ge(pd.Timestamp(config["scope"]["earliest_allowed_source_date"]))
    ].copy()

    required_start = min(
        pd.Timestamp(period["start"]) for period in config["dates"]["periods"].values()
    )
    required_end = max(
        pd.Timestamp(period["end"]) for period in config["dates"]["periods"].values()
    )
    coverage = market.loc[market["date"].between(required_start, required_end)].copy()
    if coverage.empty or coverage["date"].min() > required_start or coverage["date"].max() < required_end:
        raise ValueError("合并行情未覆盖全部预声明评价期")
    if coverage["execution_state"].isna().any():
        missing_dates = coverage.loc[coverage["execution_state"].isna(), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"评价期冻结执行状态缺失：{missing_dates.tolist()[:10]}")

    audit = {
        "signal_rows": int(len(signals)),
        "signal_first_date": str(signals["date"].min().date()),
        "signal_last_date": str(signals["date"].max().date()),
        "market_rows_2021plus": int(len(market)),
        "market_first_date_read_into_audit": str(market["date"].min().date()),
        "market_last_date": str(market["date"].max().date()),
        "evaluation_rows": int(len(coverage)),
        "pre_2021_rows_entering_metrics": 0,
        "dividend_events_2021plus": int(len(dividends)),
    }
    return market, dividends, signals, audit


def _event_maps(dividends: pd.DataFrame) -> tuple[dict[pd.Timestamp, list[dict[str, Any]]], dict[pd.Timestamp, float]]:
    ex_map: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    payment_map: dict[pd.Timestamp, float] = {}
    for row in dividends.to_dict("records"):
        ex_date = pd.Timestamp(row["ex_date"])
        payment_date = pd.Timestamp(row["payment_date"])
        ex_map.setdefault(ex_date, []).append(row)
        payment_map[payment_date] = payment_map.get(payment_date, 0.0) + float(
            row["cash_dividend_per_share"]
        )
    return ex_map, payment_map


def _commission(notional: float, costs: dict[str, Any]) -> float:
    if notional <= 0.0:
        return 0.0
    return max(
        float(costs["minimum_commission_cny_per_leg"]),
        notional * float(costs["commission_rate_per_leg"]),
    )


def simulate(
    sample: pd.DataFrame,
    dividends: pd.DataFrame,
    desired_states: np.ndarray,
    *,
    costs: dict[str, Any],
    initial_capital: float,
    lot_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在T+1开盘按0/1目标状态执行并逐日记录净值。"""

    if len(sample) != len(desired_states) or len(sample) < 2:
        raise ValueError("样本和目标状态长度不匹配或不足")
    if int(desired_states[0]) != 0:
        raise ValueError("首行必须是现金锚点")
    if not set(np.unique(desired_states).astype(int)).issubset({0, 1}):
        raise ValueError("目标状态必须为0或1")

    ex_map, _ = _event_maps(dividends)
    cash = float(initial_capital)
    shares = 0.0
    receivable = 0.0
    payment_schedule: dict[pd.Timestamp, float] = {}
    ledger_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    slippage_fraction = float(costs["slippage_bps_per_leg"]) / 10000.0

    for index, row in sample.reset_index(drop=True).iterrows():
        date = pd.Timestamp(row["date"])
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
        quantity = 0.0
        execution_price = math.nan
        execution_notional = 0.0
        commission = 0.0
        slippage_cost = 0.0

        if target_state == 0 and shares > 0.0:
            side = "SELL"
            quantity = shares
            execution_price = raw_open * (1.0 - slippage_fraction)
            execution_notional = quantity * execution_price
            commission = _commission(execution_notional, costs)
            cash += execution_notional - commission
            shares = 0.0
            slippage_cost = quantity * (raw_open - execution_price)
        elif target_state == 1:
            execution_price = raw_open * (1.0 + slippage_fraction)
            quantity = float(max(int(cash / execution_price / lot_size), 0) * lot_size)
            while quantity > 0.0:
                execution_notional = quantity * execution_price
                commission = _commission(execution_notional, costs)
                if execution_notional + commission <= cash + 1e-9:
                    break
                quantity -= lot_size
            if quantity > 0.0:
                side = "BUY"
                cash -= execution_notional + commission
                shares += quantity
                slippage_cost = quantity * (execution_price - raw_open)

        if side != "NONE":
            trade_rows.append(
                {
                    "date": date,
                    "side": side,
                    "target_state": target_state,
                    "raw_open": raw_open,
                    "execution_price": float(execution_price),
                    "quantity": float(quantity),
                    "execution_notional": float(execution_notional),
                    "commission": float(commission),
                    "slippage_cost": float(slippage_cost),
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
                "benchmark_close": float(row["benchmark_close"]),
                "equity": float(equity),
                "daily_commission": float(commission),
                "daily_slippage_cost": float(slippage_cost),
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    return ledger, pd.DataFrame(trade_rows)


def _sharpe(returns: pd.Series, trading_days: int) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return None
    volatility = float(values.std(ddof=1))
    if volatility <= 0.0:
        return None
    return float(values.mean() / volatility * math.sqrt(trading_days))


def _cagr(returns: pd.Series, trading_days: int) -> float:
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if len(values) < 2 or (values <= -1.0).any() or not np.isfinite(values).all():
        raise ValueError("收益序列不足或包含非法值")
    return float(np.expm1(np.log1p(values[1:]).mean() * trading_days))


def _complete_round_trips_by_year(states: pd.DataFrame) -> dict[str, int]:
    active = False
    counts: dict[str, int] = {}
    for row in states.sort_values("date").itertuples(index=False):
        state = int(row.policy_state)
        if state == 0 and not active:
            active = True
        elif state == 1 and active:
            year = str(pd.Timestamp(row.date).year)
            counts[year] = counts.get(year, 0) + 1
            active = False
    return counts


def _path_summary(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    buy_hold: pd.DataFrame,
    *,
    trading_days: int,
) -> dict[str, Any]:
    strategy_cagr = _cagr(ledger["daily_return"], trading_days)
    buy_hold_cagr = _cagr(buy_hold["daily_return"], trading_days)
    benchmark_returns = ledger["benchmark_close"].pct_change().fillna(0.0)
    benchmark_cagr = _cagr(benchmark_returns, trading_days)
    round_trips = _complete_round_trips_by_year(
        ledger.loc[:, ["date", "policy_state"]]
    )
    buy_hold_drawdown = float(buy_hold["drawdown"].min())
    strategy_drawdown = float(ledger["drawdown"].min())
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "observations": int(len(ledger)),
        "net_sharpe_zero_cash_rate": _sharpe(ledger["daily_return"], trading_days),
        "strategy_cagr": strategy_cagr,
        "h00300_total_return_cagr": benchmark_cagr,
        "buy_hold_510300_net_cagr": buy_hold_cagr,
        "annualized_excess_vs_h00300": float(strategy_cagr - benchmark_cagr),
        "annualized_excess_vs_buy_hold": float(strategy_cagr - buy_hold_cagr),
        "maximum_drawdown": strategy_drawdown,
        "buy_hold_maximum_drawdown": buy_hold_drawdown,
        "maximum_drawdown_ratio_vs_buy_hold": (
            float(abs(strategy_drawdown) / abs(buy_hold_drawdown))
            if buy_hold_drawdown < 0.0
            else None
        ),
        "cash_day_share": float((ledger["policy_state"] == 0).mean()),
        "trade_leg_count": int(len(trades)),
        "total_commission_cny": (
            0.0 if trades.empty else float(trades["commission"].sum())
        ),
        "total_slippage_cost_cny": (
            0.0 if trades.empty else float(trades["slippage_cost"].sum())
        ),
        "complete_round_trips_by_year": round_trips,
        "maximum_complete_round_trips_in_any_year": max(round_trips.values(), default=0),
    }


def _year_robustness(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    *,
    trading_days: int,
) -> dict[str, Any]:
    years = sorted(int(value) for value in strategy["date"].dt.year.unique())
    leave_one_out = {
        str(year): _sharpe(
            strategy.loc[strategy["date"].dt.year.ne(year), "daily_return"], trading_days
        )
        for year in years
    }
    aligned = strategy.loc[:, ["date", "daily_return"]].merge(
        buy_hold.loc[:, ["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    aligned["log_excess"] = np.log1p(aligned["daily_return_strategy"]) - np.log1p(
        aligned["daily_return_buy_hold"]
    )
    contributions = aligned.groupby(aligned["date"].dt.year)["log_excess"].sum()
    total = float(contributions.sum())
    maximum_share = None
    if total > 0.0:
        maximum_positive = max((float(value) for value in contributions if value > 0.0), default=0.0)
        maximum_share = float(maximum_positive / total)
    return {
        "leave_one_calendar_year_out_sharpes": leave_one_out,
        "yearly_log_excess_contributions_vs_buy_hold": {
            str(int(year)): float(value) for year, value in contributions.items()
        },
        "total_log_excess_vs_buy_hold": total,
        "maximum_single_year_share_of_total_log_excess": maximum_share,
    }


def _evaluate_period(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    period_name: str,
    period: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    selection = np.flatnonzero(market["date"].between(start, end).to_numpy())
    offsets = config["execution"]["start_perturbations_trading_days"]
    if len(selection) <= max(offsets):
        raise ValueError(f"{period_name}不足以执行五个起点")
    last_index = int(selection[-1])
    initial_capital = float(config["execution"]["initial_capital_cny"])
    lot_size = int(config["execution"]["lot_size_shares"])
    trading_days = int(config["execution"]["annualization_trading_days"])
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for offset in offsets:
        first_index = int(selection[offset])
        anchor_index = first_index - 1
        if anchor_index < 0:
            raise ValueError(f"{period_name}缺少首个执行日前的现金锚点")
        sample = market.iloc[anchor_index : last_index + 1].reset_index(drop=True).copy()
        states = pd.to_numeric(sample["execution_state"], errors="coerce")
        if states.iloc[1:].isna().any():
            raise ValueError(f"{period_name}起点{offset}存在冻结执行状态缺失")
        desired = states.fillna(0).astype(np.int8).to_numpy()
        desired[0] = 0
        buy_hold_desired = np.ones(len(sample), dtype=np.int8)
        buy_hold_desired[0] = 0
        buy_hold, buy_hold_trades = simulate(
            sample,
            dividends,
            buy_hold_desired,
            costs=config["costs"]["base"],
            initial_capital=initial_capital,
            lot_size=lot_size,
        )
        scenario_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        for scenario in ["base", "double"]:
            ledger, trades = simulate(
                sample,
                dividends,
                desired,
                costs=config["costs"][scenario],
                initial_capital=initial_capital,
                lot_size=lot_size,
            )
            summary = _path_summary(
                ledger,
                trades,
                buy_hold,
                trading_days=trading_days,
            )
            row: dict[str, Any] = {
                "period": period_name,
                "gate_eligible": bool(period["gate_eligible"]),
                "start_perturbation": int(offset),
                "scenario": scenario,
                "first_execution_date": str(sample["date"].iloc[1].date()),
                "last_execution_date": str(sample["date"].iloc[-1].date()),
                **summary,
            }
            rows.append(row)
            scenario_artifacts[scenario] = (ledger, trades, buy_hold)
        artifacts[f"{period_name}:{offset}"] = scenario_artifacts["base"]
        if buy_hold_trades.empty:
            raise ValueError("买入持有基准未产生初始买入，数据或成本合同异常")
    return rows, artifacts


def _gate_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["historical_gates"]
    drawdown_ratio = row["maximum_drawdown_ratio_vs_buy_hold"]
    return {
        "net_sharpe_at_least_1_20": bool(
            row["net_sharpe_zero_cash_rate"] is not None
            and row["net_sharpe_zero_cash_rate"] >= float(gates["net_sharpe_minimum"])
        ),
        "annualized_excess_vs_h00300_positive": bool(row["annualized_excess_vs_h00300"] > 0.0),
        "annualized_excess_vs_buy_hold_positive": bool(row["annualized_excess_vs_buy_hold"] > 0.0),
        "maximum_drawdown_at_most_75pct_buy_hold": bool(
            drawdown_ratio is not None
            and drawdown_ratio <= float(gates["maximum_drawdown_ratio_vs_buy_hold"])
        ),
        "maximum_round_trips_per_year_at_most_10": bool(
            row["maximum_complete_round_trips_in_any_year"]
            <= int(gates["maximum_complete_round_trips_in_any_calendar_year"])
        ),
    }


def evaluate(config: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    market, dividends, signals, input_audit = load_inputs(config)
    rows: list[dict[str, Any]] = []
    artifact_map: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for period_name, period in config["dates"]["periods"].items():
        period_rows, period_artifacts = _evaluate_period(
            market, dividends, config, period_name, period
        )
        rows.extend(period_rows)
        artifact_map.update(period_artifacts)
    metrics = pd.DataFrame(rows)

    primary_name = config["dates"]["primary_period"]
    temporal_name = config["dates"]["temporal_replication_period"]
    primary_base = metrics.loc[
        metrics["period"].eq(primary_name) & metrics["scenario"].eq("base")
    ].copy()
    primary_double = metrics.loc[
        metrics["period"].eq(primary_name) & metrics["scenario"].eq("double")
    ].copy()
    temporal_base = metrics.loc[
        metrics["period"].eq(temporal_name) & metrics["scenario"].eq("base")
    ].copy()
    if len(primary_base) != 5 or len(primary_double) != 5 or len(temporal_base) != 5:
        raise ValueError("预声明五起点评价行不完整")

    primary_gates_by_start: list[dict[str, Any]] = []
    for row in primary_base.to_dict("records"):
        gates = _gate_row(row, config)
        primary_gates_by_start.append(
            {
                "start_perturbation": int(row["start_perturbation"]),
                "gates": gates,
                "passed": bool(all(gates.values())),
            }
        )
    all_primary_pass = all(item["passed"] for item in primary_gates_by_start)

    primary_start0 = artifact_map[f"{primary_name}:0"]
    year_robustness = _year_robustness(
        primary_start0[0],
        primary_start0[2],
        trading_days=int(config["execution"]["annualization_trading_days"]),
    )
    robustness_config = config["robustness_gates"]
    leave_one_values = year_robustness["leave_one_calendar_year_out_sharpes"].values()
    year_share = year_robustness["maximum_single_year_share_of_total_log_excess"]
    robustness_gates = {
        "all_five_start_double_cost_sharpes_at_least_0_90": bool(
            primary_double["net_sharpe_zero_cash_rate"].notna().all()
            and (
                primary_double["net_sharpe_zero_cash_rate"]
                >= float(robustness_config["double_cost_sharpe_minimum"])
            ).all()
        ),
        "leave_one_calendar_year_out_sharpe_positive": bool(
            leave_one_values
            and all(value is not None and value > 0.0 for value in leave_one_values)
        ),
        "single_year_log_excess_share_at_most_50_percent": bool(
            year_share is not None
            and year_share
            <= float(robustness_config["maximum_single_year_share_of_total_log_excess"])
        ),
        "temporal_replication_all_five_starts_excess_vs_both_positive": bool(
            (temporal_base["annualized_excess_vs_h00300"] > 0.0).all()
            and (temporal_base["annualized_excess_vs_buy_hold"] > 0.0).all()
        ),
    }
    all_robustness_pass = all(robustness_gates.values())
    passed = bool(all_primary_pass and all_robustness_pass)
    status = (
        config["adjudication"]["historical_candidate_status_if_all_gates_pass"]
        if passed
        else config["adjudication"]["failed_status"]
    )

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    reverse_report = json.loads(
        (PROJECT_ROOT / config["inputs"]["reverse_validation_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    forward_status = json.loads(
        (PROJECT_ROOT / config["inputs"]["candidate_forward_status"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "status": status,
        "passed": passed,
        "evidence_class": config["protocol"]["evidence_class"],
        "candidate_project_id": config["protocol"]["candidate_project_id"],
        "candidate_name": config["protocol"]["candidate_name"],
        "interpretation": config["protocol"]["interpretation"],
        "scope": config["scope"],
        "execution_contract": config["execution"],
        "costs": config["costs"],
        "input_audit": input_audit,
        "freeze": {
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "config_sha256": manifest["config_sha256"],
            "tracked_file_count": len(manifest["tracked_files"]),
            "input_file_count": len(manifest["input_files"]),
            "hash_mismatches": {},
        },
        "primary_period": primary_name,
        "primary_base_start_summaries": records(primary_base),
        "primary_double_cost_start_summaries": records(primary_double),
        "temporal_replication_base_start_summaries": records(temporal_base),
        "primary_gates_by_start": primary_gates_by_start,
        "primary_all_five_starts_pass": all_primary_pass,
        "year_robustness_primary_start0": year_robustness,
        "robustness_gates": robustness_gates,
        "robustness_all_pass": all_robustness_pass,
        "candidate_existing_evidence": {
            "reverse_validation_status": reverse_report.get("status"),
            "reverse_validation_passed": reverse_report.get("passed"),
            "forward_status": forward_status.get("status"),
            "true_prospective_claim_rows": forward_status.get("coverage", {}).get(
                "prospective_after_freeze_signal_row_count"
            ),
        },
        "adjudication": {
            "goal_achieved": false,
            "reason_goal_not_achieved_even_if_historical_audit_passed": "该序列是历史后选候选，且真正前瞻证据不足；本审计不能替代原宏观事件门或严格前向验证",
            "macro_candidate_replacement_allowed": false,
            "parameter_rescue_after_result": "FORBIDDEN",
        },
        "boundaries": {
            "historical_rows_were_seen_before_this_audit": true,
            "new_candidate_discovered": false,
            "frozen_signal_changed": false,
            "pre_2021_rows_entering_metrics": 0,
            "research_only": true,
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": false,
        },
    }
    artifact_frames = {
        "primary_ledger": primary_start0[0],
        "primary_trades": primary_start0[1],
    }
    return report, metrics, artifact_frames


def write_outputs(
    config: dict[str, Any],
    report: dict[str, Any],
    metrics: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame],
) -> None:
    paths = config["paths"]
    result_path = PROJECT_ROOT / paths["result"]
    metrics_path = PROJECT_ROOT / paths["metrics"]
    ledger_path = PROJECT_ROOT / paths["primary_ledger"]
    trades_path = PROJECT_ROOT / paths["primary_trades"]
    for path in [result_path, metrics_path, ledger_path, trades_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError("审计结果已存在，冻结协议禁止重复运行或覆盖")
    metrics.to_parquet(metrics_path, index=False)
    artifacts["primary_ledger"].to_parquet(ledger_path, index=False)
    artifacts["primary_trades"].to_parquet(trades_path, index=False)
    report["artifacts"] = {
        "metrics": {"path": paths["metrics"], "sha256": sha256_file(metrics_path)},
        "primary_ledger": {
            "path": paths["primary_ledger"],
            "sha256": sha256_file(ledger_path),
        },
        "primary_trades": {
            "path": paths["primary_trades"],
            "sha256": sha256_file(trades_path),
        },
    }
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    config = load_config()
    manifest = validate_manifest(config)
    report, metrics, artifacts = evaluate(config, manifest)
    write_outputs(config, report, metrics, artifacts)
    print(f"审计完成：{report['status']}")
    print(f"历史后选审计通过：{report['passed']}")
    print("实盘授权：False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
