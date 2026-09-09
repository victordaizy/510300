"""GammaOIPressure公开因子的510300单合约开发期迁移检验。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_option_gamma_oi_pressure_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_input_hashes(config: dict) -> None:
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入不存在：{name}={path}")
        if sha256(path) != str(item["sha256"]).lower():
            raise RuntimeError(f"冻结输入哈希漂移：{name}")


def compute_gamma_oi_pressure(frame: pd.DataFrame, epsilon: float) -> pd.Series:
    """按冻结公式计算因子；调用方负责构造同合约上一交易日持仓量。"""

    turnover_cny = frame["turnover_10k_cny"].astype(float) * 10000.0
    price_impact = (frame["high"].astype(float) - frame["low"].astype(float)) / (
        turnover_cny + epsilon
    )
    oi_pressure = np.tanh(
        (
            frame["open_interest"].astype(float)
            - frame["lag1_open_interest"].astype(float)
        )
        / (frame["volume"].astype(float) + 1.0)
    )
    delta_direction = np.tanh(frame["delta"].astype(float))
    convexity = np.log1p(
        frame["underlying_close"].astype(float).pow(2)
        * frame["gamma"].astype(float).abs()
    )
    decay_scale = np.sqrt(1.0 + frame["theta"].astype(float).abs())
    maturity_scale = np.sqrt(1.0 + frame["dte_years"].astype(float))
    factor = (
        -price_impact
        * oi_pressure
        * delta_direction
        * convexity
        / decay_scale
        / maturity_scale
    )
    return factor.replace([np.inf, -np.inf], np.nan)


def load_development_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在Parquet读取层截断，禁止把验证期和最终留出期载入内存。"""

    cutoff = pd.Timestamp(config["dates"]["development_end"])
    eod = pd.read_parquet(
        ROOT / config["inputs"]["option_eod"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    risk = pd.read_parquet(
        ROOT / config["inputs"]["option_risk"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    benchmark = pd.read_parquet(
        ROOT / config["inputs"]["benchmark"]["path"],
        filters=[("date", "<=", cutoff)],
    )
    eod["trade_date"] = pd.to_datetime(eod["trade_date"]).dt.normalize()
    risk["trade_date"] = pd.to_datetime(risk["trade_date"]).dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    for name, frame, column in (
        ("option_eod", eod, "trade_date"),
        ("option_risk", risk, "trade_date"),
        ("benchmark", benchmark, "date"),
    ):
        if frame.empty or frame[column].max() > cutoff:
            raise RuntimeError(f"{name}越过冻结开发期截止日")
    if eod.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("期权日线存在重复合约日期")
    if risk.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("期权风险指标存在重复合约日期")
    risk_columns = [
        "trade_date",
        "contract_code",
        "delta",
        "theta",
        "gamma",
        "implied_volatility",
    ]
    panel = eod.merge(
        risk[risk_columns],
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
    )
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"]).dt.normalize()
    panel["dte"] = (panel["expiry_date"] - panel["trade_date"]).dt.days
    panel["dte_years"] = panel["dte"] / 365.2425
    panel["moneyness"] = panel["strike"] / panel["underlying_close"]
    panel["signal_premium_per_contract"] = panel["close"] * panel["contract_unit"]
    panel = panel.sort_values(["contract_code", "trade_date"], kind="mergesort")
    panel["lag1_open_interest"] = panel.groupby("contract_code", sort=False)[
        "open_interest"
    ].shift(1)
    panel["lag1_contract_date"] = panel.groupby("contract_code", sort=False)[
        "trade_date"
    ].shift(1)
    benchmark = (
        benchmark[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    previous_market_date = pd.Series(
        benchmark["date"].shift(1).to_numpy(), index=benchmark["date"]
    )
    panel["expected_lag1_date"] = panel["trade_date"].map(previous_market_date)
    panel.loc[
        panel["lag1_contract_date"].ne(panel["expected_lag1_date"]),
        "lag1_open_interest",
    ] = np.nan
    panel["gamma_oi_pressure"] = compute_gamma_oi_pressure(
        panel, float(config["factor"]["epsilon"])
    )
    return panel, benchmark


def build_schedule(benchmark: pd.DataFrame, config: dict) -> list[dict[str, pd.Timestamp]]:
    start = pd.Timestamp(config["dates"]["feature_start"])
    end = pd.Timestamp(config["dates"]["development_end"])
    dates = benchmark.loc[benchmark["date"].between(start, end), "date"].tolist()
    schedule: list[dict[str, pd.Timestamp]] = []
    index = 0
    while index + 2 < len(dates):
        schedule.append(
            {
                "signal_date": dates[index],
                "entry_date": dates[index + 1],
                "exit_date": dates[index + 2],
            }
        )
        index += 2
    if schedule and max(item["exit_date"] for item in schedule) > end:
        raise RuntimeError("日程越过开发截止日")
    return schedule


def select_contract(signal_rows: pd.DataFrame, config: dict) -> dict[str, Any] | None:
    rule = config["contract_selection"]
    rows = signal_rows.copy()
    rows = rows.loc[
        ~rows["is_adjusted"].astype(bool)
        & rows["contract_unit"].eq(int(rule["contract_unit_required"]))
        & rows["dte"].between(
            int(rule["minimum_calendar_days_to_expiry"]),
            int(rule["maximum_calendar_days_to_expiry"]),
        )
        & rows["moneyness"].between(
            float(rule["minimum_moneyness"]), float(rule["maximum_moneyness"])
        )
        & rows["volume"].ge(float(rule["minimum_signal_day_volume"]))
        & rows["open_interest"].ge(float(rule["minimum_signal_day_open_interest"]))
        & rows["signal_premium_per_contract"].between(
            float(rule["minimum_signal_premium_per_contract_cny"]),
            float(rule["maximum_signal_premium_per_contract_cny"]),
        )
        & rows["gamma_oi_pressure"].notna()
    ].copy()
    if rows.empty:
        return None
    rows = rows.sort_values(
        ["gamma_oi_pressure", "volume", "contract_code"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    row = rows.iloc[0]
    quantity = math.ceil(
        float(rule["target_signal_notional_cny"])
        / float(row["signal_premium_per_contract"])
    )
    quantity = min(quantity, int(rule["maximum_contracts"]))
    return {
        "contract_code": str(row["contract_code"]),
        "option_type": str(row["option_type"]),
        "quantity": int(quantity),
        "contract_unit": int(row["contract_unit"]),
        "factor": float(row["gamma_oi_pressure"]),
        "signal_close": float(row["close"]),
        "signal_volume": float(row["volume"]),
        "signal_open_interest": float(row["open_interest"]),
        "signal_delta": float(row["delta"]),
        "signal_gamma": float(row["gamma"]),
        "signal_theta": float(row["theta"]),
        "signal_dte": int(row["dte"]),
    }


@dataclass(frozen=True)
class TradeResult:
    status: str
    executed: bool
    entry_notional_cny: float
    exit_notional_cny: float
    fees_cny: float
    pnl_cny: float


def execute_trade(
    selection: dict[str, Any] | None,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    indexed_panel: pd.DataFrame,
    capital_cny: float,
    config: dict,
) -> TradeResult:
    if selection is None:
        return TradeResult("NO_CONTRACT", False, 0.0, 0.0, 0.0, 0.0)
    key_entry = (entry_date, selection["contract_code"])
    if key_entry not in indexed_panel.index:
        return TradeResult("NO_ENTRY_ROW", False, 0.0, 0.0, 0.0, 0.0)
    entry = indexed_panel.loc[key_entry]
    if isinstance(entry, pd.DataFrame):
        raise ValueError("入场键不唯一")
    entry_price = float(entry["high"])
    if not math.isfinite(entry_price) or entry_price <= 0 or float(entry["volume"]) <= 0:
        return TradeResult("NO_ENTRY_TRADE", False, 0.0, 0.0, 0.0, 0.0)
    quantity = int(selection["quantity"])
    unit = int(selection["contract_unit"])
    fee_per_contract = float(
        config["execution_envelope"]["option_fee_cny_per_contract_per_leg"]
    )
    entry_notional = entry_price * unit * quantity
    buy_fee = fee_per_contract * quantity
    minimum = float(config["execution_envelope"]["minimum_opening_trade_notional_cny"])
    if entry_notional < minimum:
        return TradeResult("SMALL_OPENING_TRADE_REJECTED", False, entry_notional, 0.0, 0.0, 0.0)
    if entry_notional + buy_fee > capital_cny:
        return TradeResult("INSUFFICIENT_CASH", False, entry_notional, 0.0, 0.0, 0.0)
    key_exit = (exit_date, selection["contract_code"])
    if key_exit not in indexed_panel.index:
        exit_notional = 0.0
        status = "MISSING_EXIT_MARKED_ZERO"
    else:
        exit_row = indexed_panel.loc[key_exit]
        if isinstance(exit_row, pd.DataFrame):
            raise ValueError("退出键不唯一")
        exit_price = float(exit_row["low"])
        if (
            not math.isfinite(exit_price)
            or exit_price <= 0
            or float(exit_row["volume"]) <= 0
        ):
            exit_notional = 0.0
            status = "NO_EXIT_TRADE_MARKED_ZERO"
        else:
            exit_notional = exit_price * unit * quantity
            status = "EXECUTED_WORST_TRADE_ENVELOPE"
    fees = 2.0 * fee_per_contract * quantity
    pnl = exit_notional - entry_notional - fees
    return TradeResult(status, True, entry_notional, exit_notional, fees, pnl)


def cagr(final_value: float, initial_value: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    if final_value <= 0:
        return -1.0
    return (final_value / initial_value) ** (365.2425 / days) - 1.0


def run_development(config: dict, panel: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    schedule = build_schedule(benchmark, config)
    if not schedule:
        raise RuntimeError("开发期没有可评估日程")
    panel_by_date = {
        date: rows for date, rows in panel.groupby("trade_date", sort=False)
    }
    indexed = panel.set_index(["trade_date", "contract_code"]).sort_index()
    benchmark_close = benchmark.set_index("date")["close"]
    initial = float(config["execution_envelope"]["initial_cash_cny"])
    capital = initial
    ledger: list[dict[str, Any]] = []
    executed_entry_notionals: list[float] = []
    status_counts: dict[str, int] = {}
    for period, dates in enumerate(schedule, start=1):
        signal_rows = panel_by_date.get(dates["signal_date"], pd.DataFrame())
        selection = select_contract(signal_rows, config)
        capital_before = capital
        trade = execute_trade(
            selection,
            dates["entry_date"],
            dates["exit_date"],
            indexed,
            capital_before,
            config,
        )
        if trade.executed:
            executed_entry_notionals.append(trade.entry_notional_cny)
            capital = max(0.0, capital_before + trade.pnl_cny)
        status_counts[trade.status] = status_counts.get(trade.status, 0) + 1
        ledger.append(
            {
                "period": period,
                **{key: str(value.date()) for key, value in dates.items()},
                "selection": selection,
                "trade": asdict(trade),
                "capital_before": capital_before,
                "capital_after": capital,
            }
        )
        if capital <= 0:
            break
    start = schedule[0]["entry_date"]
    end = schedule[min(len(ledger), len(schedule)) - 1]["exit_date"]
    benchmark_initial = float(benchmark_close.loc[start])
    benchmark_final = float(benchmark_close.loc[end])
    strategy_cagr = cagr(capital, initial, start, end)
    benchmark_cagr = cagr(benchmark_final, benchmark_initial, start, end)
    excess = strategy_cagr - benchmark_cagr
    minimum = float(config["execution_envelope"]["minimum_opening_trade_notional_cny"])
    all_minimum = bool(executed_entry_notionals) and all(
        notional >= minimum for notional in executed_entry_notionals
    )
    minimum_count = int(config["evaluation"]["minimum_executed_trades"])
    threshold = float(config["evaluation"]["annualized_excess_threshold"])
    pass_gate = (
        excess >= threshold
        and len(executed_entry_notionals) >= minimum_count
        and all_minimum
    )
    return {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "scheduled_periods": len(schedule),
        "evaluated_periods": len(ledger),
        "loaded_option_max_date": str(panel["trade_date"].max().date()),
        "loaded_benchmark_max_date": str(benchmark["date"].max().date()),
        "initial_cash_cny": initial,
        "final_equity_cny": capital,
        "strategy_total_return": capital / initial - 1.0,
        "strategy_cagr": strategy_cagr,
        "benchmark_total_return": benchmark_final / benchmark_initial - 1.0,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": excess,
        "executed_trades": len(executed_entry_notionals),
        "status_counts": status_counts,
        "all_executed_openings_at_least_5000": all_minimum,
        "research_gate": {
            "annualized_excess_at_least_20pct": excess >= threshold,
            "executed_trade_count_at_least_minimum": len(executed_entry_notionals)
            >= minimum_count,
            "minimum_opening_notional_pass": all_minimum,
            "eligible_for_locked_validation": pass_gate,
        },
        "ledger": ledger,
    }


def write_outputs(result: dict, config: dict, manifest: dict) -> None:
    status = (
        "DEV_PASS_ELIGIBLE_FOR_SEPARATELY_FROZEN_VALIDATION"
        if result["research_gate"]["eligible_for_locked_validation"]
        else "DEV_REJECTED_FROZEN_NO_PARAMETER_RESCUE"
    )
    payload = {
        "project_id": config["protocol"]["project_id"],
        "status": status,
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "DEVELOPMENT_ONLY_NO_2024_PLUS_ROWS_LOADED",
        "deployable_strategy": False,
        **result,
        "safety": config["governance"],
    }
    json_path = ROOT / config["paths"]["result_json"]
    markdown_path = ROOT / config["paths"]["result_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    markdown = "\n".join(
        [
            "# 510300 GammaOIPressure V1开发期迁移结果",
            "",
            f"- 状态：`{status}`",
            f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
            f"- 策略年化：{payload['strategy_cagr']:.2%}",
            f"- H00300全收益基准年化：{payload['benchmark_cagr']:.2%}",
            f"- 几何年化超额：{payload['annualized_excess']:.2%}",
            f"- 期末权益：{payload['final_equity_cny']:.2f}元",
            f"- 实际成交：{payload['executed_trades']}笔",
            f"- 所有成交开仓不少于5000元：{payload['all_executed_openings_at_least_5000']}",
            "- 2024验证期与2025后最终留出期：未载入",
            "",
            "## 证据边界",
            "",
            "本结果是公开横截面因子的单合约迁移，不是原论文复制。入场使用次日最高价、",
            "退出使用再下一日最低价并收取双边期权费用；历史买卖盘不可见，因此仍不是",
            "真实历史成交证明，也不授权仓位、订单或实盘。",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_option_gamma_oi_pressure_v1 import verify_protocol

    config, manifest = verify_protocol()
    assert_input_hashes(config)
    panel, benchmark = load_development_data(config)
    result = run_development(config, panel, benchmark)
    write_outputs(result, config, manifest)
    summary = {key: value for key, value in result.items() if key != "ledger"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
