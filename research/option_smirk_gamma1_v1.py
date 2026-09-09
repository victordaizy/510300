"""风险中性偏度Gamma1对510300认购期权收益的开发期迁移检验。"""

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
CONFIG = ROOT / "config" / "510300_option_smirk_gamma1_v1.yaml"
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
        if not path.is_file() or sha256(path) != str(item["sha256"]).lower():
            raise RuntimeError(f"冻结输入不存在或哈希漂移：{name}")


def weighted_smirk_fit(xi: np.ndarray, iv: np.ndarray, volume: np.ndarray) -> tuple[float, float, float] | None:
    finite = np.isfinite(xi) & np.isfinite(iv) & np.isfinite(volume) & (volume > 0)
    xi = xi[finite]
    iv = iv[finite]
    volume = volume[finite]
    if len(np.unique(xi)) < 3:
        return None
    design = np.column_stack([np.ones(len(xi)), xi, xi**2])
    root_weight = np.sqrt(volume)
    weighted_design = design * root_weight[:, None]
    weighted_iv = iv * root_weight
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_iv, rcond=None)
    if rank < 3 or not np.isfinite(coefficients).all() or coefficients[0] <= 0:
        return None
    alpha0, alpha1, alpha2 = coefficients
    return float(alpha0), float(alpha1 / alpha0), float(alpha2 / alpha0)


def fit_expiry_smirk(rows: pd.DataFrame, config: dict) -> dict[str, float] | None:
    rule = config["smirk_construction"]
    rows = rows.loc[
        ~rows["is_adjusted"].astype(bool)
        & rows["contract_unit"].eq(int(config["contract_selection"]["contract_unit_required"]))
        & rows["volume"].gt(0)
        & rows["close"].gt(0)
        & rows["implied_volatility"].between(
            float(rule["implied_volatility_minimum"]),
            float(rule["implied_volatility_maximum"]),
        )
    ].copy()
    if rows.empty:
        return None
    trade_date = pd.Timestamp(rows["trade_date"].iloc[0])
    expiry = pd.Timestamp(rows["expiry_date"].iloc[0])
    dte = int((expiry - trade_date).days)
    if dte <= 0:
        return None
    paired = rows.pivot_table(
        index="strike", columns="option_type", values="close", aggfunc="first"
    ).dropna(subset=["C", "P"])
    if paired.empty:
        return None
    paired["gap"] = (paired["C"] - paired["P"]).abs()
    anchor_strike = float(
        paired.sort_values(["gap", "strike"], kind="mergesort").index[0]
    )
    tau = dte / 365.2425
    rate = float(rule["risk_free_annual_rate"])
    anchor = paired.loc[anchor_strike]
    forward = anchor_strike + math.exp(rate * tau) * (float(anchor["C"]) - float(anchor["P"]))
    if not math.isfinite(forward) or forward <= 0:
        return None
    rows["forward_distance"] = (rows["strike"] - forward).abs()
    closest = rows.loc[rows["forward_distance"].eq(rows["forward_distance"].min())]
    atm_iv = float(closest["implied_volatility"].mean())
    if not math.isfinite(atm_iv) or atm_iv <= 0:
        return None
    if bool(rule["use_otm_options_only"]):
        rows = rows.loc[
            (rows["option_type"].eq("P") & rows["strike"].lt(forward))
            | (rows["option_type"].eq("C") & rows["strike"].ge(forward))
        ].copy()
    rows = rows.sort_values(
        ["strike", "volume", "contract_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("strike", keep="first")
    if rows["strike"].nunique() < int(rule["minimum_distinct_strikes_per_expiry"]):
        return None
    xi = np.log(rows["strike"].to_numpy(float) / forward) / (atm_iv * math.sqrt(tau))
    fit = weighted_smirk_fit(
        xi,
        rows["implied_volatility"].to_numpy(float),
        rows["volume"].to_numpy(float),
    )
    if fit is None:
        return None
    gamma0, gamma1, gamma2 = fit
    return {
        "dte": float(dte),
        "forward": forward,
        "atm_iv": atm_iv,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "strike_count": int(rows["strike"].nunique()),
    }


def interpolate_constant_maturity(rows: pd.DataFrame, target_dte: int) -> float | None:
    rows = rows.dropna(subset=["dte", "gamma1"]).sort_values("dte", kind="mergesort")
    exact = rows.loc[rows["dte"].eq(float(target_dte))]
    if not exact.empty:
        return float(exact.iloc[0]["gamma1"])
    below = rows.loc[rows["dte"].lt(float(target_dte))]
    above = rows.loc[rows["dte"].gt(float(target_dte))]
    if below.empty or above.empty:
        return None
    left = below.iloc[-1]
    right = above.iloc[0]
    weight = (target_dte - float(left["dte"])) / (
        float(right["dte"]) - float(left["dte"])
    )
    return float(left["gamma1"] + weight * (right["gamma1"] - left["gamma1"]))


def load_development_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            raise RuntimeError(f"{name}越过冻结开发截止日")
    panel = eod.merge(
        risk[["trade_date", "contract_code", "delta", "implied_volatility"]],
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
    )
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"]).dt.normalize()
    panel["dte"] = (panel["expiry_date"] - panel["trade_date"]).dt.days
    panel["signal_premium_per_contract"] = panel["close"] * panel["contract_unit"]
    benchmark = (
        benchmark[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return panel, benchmark


def build_smirk_features(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (trade_date, expiry_date), rows in panel.groupby(
        ["trade_date", "expiry_date"], sort=True
    ):
        fitted = fit_expiry_smirk(rows, config)
        if fitted is not None:
            records.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "expiry_date": pd.Timestamp(expiry_date),
                    **fitted,
                }
            )
    expiry_features = pd.DataFrame(records)
    if expiry_features.empty:
        raise RuntimeError("没有可拟合的波动率曲线")
    target = int(config["smirk_construction"]["target_constant_maturity_calendar_days"])
    daily_records = []
    for trade_date, rows in expiry_features.groupby("trade_date", sort=True):
        value = interpolate_constant_maturity(rows, target)
        daily_records.append({"trade_date": trade_date, "gamma1_30d": value})
    daily = pd.DataFrame(daily_records).sort_values("trade_date").reset_index(drop=True)
    window = int(config["signal"]["trailing_window_trading_days"])
    quantile = float(config["signal"]["lower_quantile"])
    lag = int(config["signal"]["threshold_lag_trading_days"])
    daily["gamma1_lower_threshold"] = (
        daily["gamma1_30d"].shift(lag).rolling(window, min_periods=window).quantile(quantile)
    )
    daily["buy_call_signal"] = daily["gamma1_30d"].le(
        daily["gamma1_lower_threshold"]
    ) & daily["gamma1_lower_threshold"].notna()
    return daily


def build_schedule(benchmark: pd.DataFrame, config: dict) -> list[dict[str, pd.Timestamp]]:
    start = pd.Timestamp(config["dates"]["strategy_start"])
    end = pd.Timestamp(config["dates"]["development_end"])
    holding = int(config["schedule"]["holding_trading_days"])
    dates = benchmark.loc[benchmark["date"].between(start, end), "date"].tolist()
    schedule = []
    index = 0
    while index + 1 + holding < len(dates):
        schedule.append(
            {
                "signal_date": dates[index],
                "entry_date": dates[index + 1],
                "exit_date": dates[index + 1 + holding],
            }
        )
        index += 1 + holding
    return schedule


def select_call(signal_rows: pd.DataFrame, config: dict) -> dict[str, Any] | None:
    rule = config["contract_selection"]
    rows = signal_rows.loc[
        signal_rows["option_type"].eq(str(rule["option_type"]))
        & ~signal_rows["is_adjusted"].astype(bool)
        & signal_rows["contract_unit"].eq(int(rule["contract_unit_required"]))
        & signal_rows["dte"].between(
            int(rule["minimum_calendar_days_to_expiry"]),
            int(rule["maximum_calendar_days_to_expiry"]),
        )
        & signal_rows["delta"].abs().between(
            float(rule["absolute_delta_minimum"]),
            float(rule["absolute_delta_maximum"]),
        )
        & signal_rows["volume"].ge(float(rule["minimum_signal_day_volume"]))
        & signal_rows["open_interest"].ge(float(rule["minimum_signal_day_open_interest"]))
        & signal_rows["signal_premium_per_contract"].between(
            float(rule["minimum_signal_premium_per_contract_cny"]),
            float(rule["maximum_signal_premium_per_contract_cny"]),
        )
    ].copy()
    if rows.empty:
        return None
    rows["delta_distance"] = (
        rows["delta"].abs() - float(rule["target_absolute_delta"])
    ).abs()
    rows["dte_distance"] = (
        rows["dte"] - int(rule["target_calendar_days_to_expiry"])
    ).abs()
    rows = rows.sort_values(
        ["delta_distance", "dte_distance", "volume", "contract_code"],
        ascending=[True, True, False, True],
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
        "quantity": int(quantity),
        "contract_unit": int(row["contract_unit"]),
        "signal_delta": float(row["delta"]),
        "signal_dte": int(row["dte"]),
        "signal_close": float(row["close"]),
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
    entry_key = (entry_date, selection["contract_code"])
    if entry_key not in indexed_panel.index:
        return TradeResult("NO_ENTRY_ROW", False, 0.0, 0.0, 0.0, 0.0)
    entry = indexed_panel.loc[entry_key]
    if isinstance(entry, pd.DataFrame):
        raise ValueError("入场键不唯一")
    entry_price = float(entry["high"])
    if not math.isfinite(entry_price) or entry_price <= 0 or float(entry["volume"]) <= 0:
        return TradeResult("NO_ENTRY_TRADE", False, 0.0, 0.0, 0.0, 0.0)
    quantity = int(selection["quantity"])
    unit = int(selection["contract_unit"])
    fee = float(config["execution_envelope"]["option_fee_cny_per_contract_per_leg"])
    entry_notional = entry_price * unit * quantity
    minimum = float(config["execution_envelope"]["minimum_opening_trade_notional_cny"])
    if entry_notional < minimum:
        return TradeResult("SMALL_OPENING_TRADE_REJECTED", False, entry_notional, 0.0, 0.0, 0.0)
    if entry_notional + fee * quantity > capital_cny:
        return TradeResult("INSUFFICIENT_CASH", False, entry_notional, 0.0, 0.0, 0.0)
    exit_key = (exit_date, selection["contract_code"])
    if exit_key not in indexed_panel.index:
        exit_notional = 0.0
        status = "MISSING_EXIT_MARKED_ZERO"
    else:
        exit_row = indexed_panel.loc[exit_key]
        if isinstance(exit_row, pd.DataFrame):
            raise ValueError("退出键不唯一")
        exit_price = float(exit_row["low"])
        if not math.isfinite(exit_price) or exit_price <= 0 or float(exit_row["volume"]) <= 0:
            exit_notional = 0.0
            status = "NO_EXIT_TRADE_MARKED_ZERO"
        else:
            exit_notional = exit_price * unit * quantity
            status = "EXECUTED_WORST_TRADE_ENVELOPE"
    fees = 2.0 * fee * quantity
    return TradeResult(
        status,
        True,
        entry_notional,
        exit_notional,
        fees,
        exit_notional - entry_notional - fees,
    )


def cagr(final_value: float, initial_value: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    if final_value <= 0:
        return -1.0
    return (final_value / initial_value) ** (365.2425 / days) - 1.0


def run_development(
    config: dict,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    schedule = build_schedule(benchmark, config)
    if not schedule:
        raise RuntimeError("开发期没有日程")
    feature_index = features.set_index("trade_date")
    panel_by_date = {
        date: rows for date, rows in panel.groupby("trade_date", sort=False)
    }
    indexed_panel = panel.set_index(["trade_date", "contract_code"]).sort_index()
    benchmark_close = benchmark.set_index("date")["close"]
    initial = float(config["execution_envelope"]["initial_cash_cny"])
    capital = initial
    ledger: list[dict[str, Any]] = []
    notionals: list[float] = []
    status_counts: dict[str, int] = {}
    signal_count = 0
    for period, dates in enumerate(schedule, start=1):
        signal_row = feature_index.loc[dates["signal_date"]] if dates["signal_date"] in feature_index.index else None
        buy_signal = bool(signal_row["buy_call_signal"]) if signal_row is not None else False
        if buy_signal:
            signal_count += 1
            selection = select_call(
                panel_by_date.get(dates["signal_date"], pd.DataFrame()), config
            )
            trade = execute_trade(
                selection,
                dates["entry_date"],
                dates["exit_date"],
                indexed_panel,
                capital,
                config,
            )
        else:
            selection = None
            trade = TradeResult("CASH_NO_EXTREME_SKEW", False, 0.0, 0.0, 0.0, 0.0)
        capital_before = capital
        if trade.executed:
            notionals.append(trade.entry_notional_cny)
            capital = max(0.0, capital + trade.pnl_cny)
        status_counts[trade.status] = status_counts.get(trade.status, 0) + 1
        ledger.append(
            {
                "period": period,
                **{key: str(value.date()) for key, value in dates.items()},
                "gamma1_30d": None if signal_row is None else float(signal_row["gamma1_30d"]),
                "gamma1_lower_threshold": None
                if signal_row is None or pd.isna(signal_row["gamma1_lower_threshold"])
                else float(signal_row["gamma1_lower_threshold"]),
                "buy_call_signal": buy_signal,
                "selection": selection,
                "trade": asdict(trade),
                "capital_before": capital_before,
                "capital_after": capital,
            }
        )
        if capital <= 0:
            break
    start = schedule[0]["entry_date"]
    end = schedule[len(ledger) - 1]["exit_date"]
    benchmark_initial = float(benchmark_close.loc[start])
    benchmark_final = float(benchmark_close.loc[end])
    strategy_cagr = cagr(capital, initial, start, end)
    benchmark_cagr = cagr(benchmark_final, benchmark_initial, start, end)
    excess = strategy_cagr - benchmark_cagr
    minimum_count = int(config["evaluation"]["minimum_executed_trades"])
    threshold = float(config["evaluation"]["annualized_excess_threshold"])
    minimum_notional = float(config["execution_envelope"]["minimum_opening_trade_notional_cny"])
    all_minimum = bool(notionals) and all(value >= minimum_notional for value in notionals)
    gate = excess >= threshold and len(notionals) >= minimum_count and all_minimum
    return {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "scheduled_periods": len(schedule),
        "evaluated_periods": len(ledger),
        "loaded_option_max_date": str(panel["trade_date"].max().date()),
        "loaded_benchmark_max_date": str(benchmark["date"].max().date()),
        "feature_days": int(features["gamma1_30d"].notna().sum()),
        "extreme_skew_signals": signal_count,
        "executed_trades": len(notionals),
        "status_counts": status_counts,
        "initial_cash_cny": initial,
        "final_equity_cny": capital,
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": excess,
        "all_executed_openings_at_least_5000": all_minimum,
        "research_gate": {
            "annualized_excess_at_least_20pct": excess >= threshold,
            "executed_trade_count_at_least_minimum": len(notionals) >= minimum_count,
            "minimum_opening_notional_pass": all_minimum,
            "eligible_for_locked_validation": gate,
        },
        "ledger": ledger,
    }


def write_outputs(result: dict, features: pd.DataFrame, config: dict, manifest: dict) -> None:
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
    feature_path = ROOT / config["paths"]["feature_parquet"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    features.to_parquet(feature_path, index=False)
    markdown = "\n".join(
        [
            "# 510300风险中性偏度Gamma1 V1开发期结果",
            "",
            f"- 状态：`{status}`",
            f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
            f"- 策略年化：{payload['strategy_cagr']:.2%}",
            f"- H00300全收益基准年化：{payload['benchmark_cagr']:.2%}",
            f"- 几何年化超额：{payload['annualized_excess']:.2%}",
            f"- 极端偏度信号：{payload['extreme_skew_signals']}次",
            f"- 实际成交：{payload['executed_trades']}笔",
            f"- 期末权益：{payload['final_equity_cny']:.2f}元",
            "- 2024验证期与2025后最终留出期：未载入",
            "",
            "本分支是50ETF Delta中性论文向300ETF单合约只做多的迁移，不是原论文复制。",
            "历史买卖盘不可见，入场按次日最高价、退出按第20日最低价并计双边费用。",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_option_smirk_gamma1_v1 import verify_protocol

    config, manifest = verify_protocol()
    assert_input_hashes(config)
    panel, benchmark = load_development_data(config)
    features = build_smirk_features(panel, config)
    result = run_development(config, panel, benchmark, features)
    write_outputs(result, features, config, manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "ledger"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
