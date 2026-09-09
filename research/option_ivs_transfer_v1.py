"""把50ETF文献的持仓量加权IVS以冻结公式迁移到510300。"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.option_trade_envelope_v1 import (  # noqa: E402
    assert_input_hashes,
    evaluate_leg,
    select_contract,
)
from research.option_technical_rule_discovery_v1 import load_sources  # noqa: E402
from scripts.freeze_510300_option_technical_rule_discovery_v1 import (  # noqa: E402
    verify_protocol as verify_data_loader_protocol,
)
from scripts.freeze_510300_option_trade_envelope_v1 import (  # noqa: E402
    verify_protocol as verify_vehicle_protocol,
)


CONFIG = ROOT / "config" / "510300_option_ivs_transfer_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def build_daily_ivs(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    rule = config["ivs"]
    rows = panel.loc[
        ~panel["is_adjusted"].astype(bool)
        & panel["dte"].ge(int(rule["minimum_calendar_days_to_expiry"]))
        & panel["open_interest"].ge(int(rule["minimum_open_interest_each_side"]))
        & panel["implied_volatility"].between(
            float(rule["minimum_iv"]), float(rule["maximum_iv"])
        )
    ].copy()
    keys = ["trade_date", "expiry_date", "strike"]
    calls = rows.loc[rows["option_type"].eq("C"), keys + ["implied_volatility", "open_interest"]]
    calls = calls.rename(columns={"implied_volatility": "call_iv", "open_interest": "call_oi"})
    puts = rows.loc[rows["option_type"].eq("P"), keys + ["implied_volatility", "open_interest"]]
    puts = puts.rename(columns={"implied_volatility": "put_iv", "open_interest": "put_oi"})
    if calls.duplicated(keys).any() or puts.duplicated(keys).any():
        raise ValueError("标准期权同日同到期同执行价的认购或认沽不唯一")
    pairs = calls.merge(puts, on=keys, how="inner", validate="one_to_one")
    pairs["pair_open_interest"] = pairs["call_oi"] + pairs["put_oi"]
    pairs["weighted_spread"] = (
        pairs["call_iv"] - pairs["put_iv"]
    ) * pairs["pair_open_interest"]
    grouped = pairs.groupby("trade_date", sort=True)
    daily = grouped.agg(
        pair_count=("strike", "size"),
        total_pair_open_interest=("pair_open_interest", "sum"),
        weighted_spread_sum=("weighted_spread", "sum"),
    ).reset_index(names="date")
    daily["ivs"] = daily["weighted_spread_sum"] / daily["total_pair_open_interest"]
    if not np.isfinite(daily["ivs"]).all():
        raise ValueError("IVS存在非有限值")
    return daily.sort_values("date").reset_index(drop=True)


def build_monthly_rows(
    benchmark: pd.DataFrame,
    daily_ivs: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    market = benchmark.sort_values("date").reset_index(drop=True).copy()
    market["month"] = market["date"].dt.to_period("M")
    month_ends = market.groupby("month", sort=True).tail(1)[["month", "date", "close"]]
    month_ends = month_ends.rename(columns={"date": "signal_date", "close": "signal_close"})
    dates = market["date"].tolist()
    position = {date: index for index, date in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    ivs_dates = daily_ivs["date"].to_numpy(dtype="datetime64[ns]")
    for index in range(len(month_ends) - 1):
        current = month_ends.iloc[index]
        following = month_ends.iloc[index + 1]
        signal_date = pd.Timestamp(current["signal_date"])
        next_signal_date = pd.Timestamp(following["signal_date"])
        entry_index = position[signal_date] + 1
        if entry_index >= len(dates):
            continue
        entry_date = dates[entry_index]
        exit_date = next_signal_date
        iv_position = int(np.searchsorted(ivs_dates, np.datetime64(signal_date), side="right") - 1)
        if iv_position < 0:
            continue
        iv_row = daily_ivs.iloc[iv_position]
        iv_date = pd.Timestamp(iv_row["date"])
        staleness = (signal_date - iv_date).days
        ivs_value = (
            float(iv_row["ivs"])
            if staleness <= int(config["ivs"]["maximum_signal_staleness_calendar_days"])
            else np.nan
        )
        entry_close = float(market.loc[market["date"].eq(entry_date), "close"].iloc[0])
        exit_close = float(following["signal_close"])
        rows.append(
            {
                "month": str(current["month"]),
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ivs_date": iv_date,
                "ivs_staleness_calendar_days": staleness,
                "ivs": ivs_value,
                "forward_return": exit_close / entry_close - 1.0,
            }
        )
    return pd.DataFrame(rows)


def expanding_predictions(monthly: pd.DataFrame, config: dict) -> pd.DataFrame:
    first_month = pd.Period(config["dates"]["first_strategy_signal_month"], freq="M")
    development_end = pd.Timestamp(config["dates"]["development_end"])
    minimum = int(config["model"]["minimum_mature_training_months"])
    rows: list[dict[str, Any]] = []
    for current in monthly.to_dict("records"):
        signal_date = pd.Timestamp(current["signal_date"])
        if signal_date.to_period("M") < first_month or signal_date > development_end:
            continue
        training = monthly.loc[
            monthly["exit_date"].le(signal_date)
            & monthly["ivs"].notna()
            & monthly["forward_return"].notna()
        ].copy()
        base = {**current, "training_months": len(training)}
        if not math.isfinite(float(current["ivs"])):
            rows.append({**base, "status": "NO_VIEW_STALE_IVS", "alpha": np.nan, "beta": np.nan, "forecast": np.nan})
            continue
        if len(training) < minimum:
            rows.append({**base, "status": "NO_MODEL_MINIMUM_TRAINING", "alpha": np.nan, "beta": np.nan, "forecast": np.nan})
            continue
        design = np.column_stack([np.ones(len(training)), training["ivs"].to_numpy(float)])
        alpha, beta_raw = np.linalg.lstsq(design, training["forward_return"].to_numpy(float), rcond=None)[0]
        beta = min(float(beta_raw), 0.0)
        if beta == 0.0:
            alpha = float(training["forward_return"].mean())
        forecast = float(alpha + beta * float(current["ivs"]))
        rows.append(
            {
                **base,
                "status": "MODEL_READY",
                "alpha": float(alpha),
                "beta_raw": float(beta_raw),
                "beta": beta,
                "forecast": forecast,
                "side": "C" if forecast >= 0 else "P",
                "latest_training_exit": training["exit_date"].max(),
            }
        )
    result = pd.DataFrame(rows)
    ready = result.loc[result["status"].eq("MODEL_READY")]
    if len(ready) and ready["latest_training_exit"].gt(ready["signal_date"]).any():
        raise RuntimeError("IVS模型训练使用了信号日后才成熟的标签")
    return result


def evaluate_strategy(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict,
) -> tuple[dict, list[dict[str, Any]]]:
    panel_by_date = {date: rows for date, rows in panel.groupby("trade_date", sort=False)}
    indexed_panel = panel.set_index(["trade_date", "contract_code"]).sort_index()
    benchmark_close = benchmark.set_index("date")["close"]
    initial = float(config["execution_envelope"]["initial_cash_cny"])
    capital = initial
    minimum_passes: list[bool] = []
    correct: list[bool] = []
    ledger: list[dict[str, Any]] = []
    ready_predictions = predictions.loc[predictions["status"].eq("MODEL_READY")]
    for row in predictions.to_dict("records"):
        before = capital
        if row["status"] != "MODEL_READY":
            ledger.append(
                {
                    "signal_date": str(pd.Timestamp(row["signal_date"]).date()),
                    "status": row["status"],
                    "capital_before": before,
                    "capital_after": before,
                }
            )
            continue
        side = str(row["side"])
        signal_date = pd.Timestamp(row["signal_date"])
        entry_date = pd.Timestamp(row["entry_date"])
        exit_date = pd.Timestamp(row["exit_date"])
        signal_rows = panel_by_date.get(signal_date, pd.DataFrame())
        selection = select_contract(signal_rows, side, config)
        leg = evaluate_leg(
            selection, entry_date, exit_date, indexed_panel, capital, config
        )
        capital = max(0.0, capital + leg.pnl_cny)
        realized_up = float(row["forward_return"]) >= 0
        correct.append((side == "C") == realized_up)
        if leg.executed:
            minimum_passes.append(leg.entry_minimum_pass)
        ledger.append(
            {
                "signal_date": str(signal_date.date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "ivs": float(row["ivs"]),
                "forecast": float(row["forecast"]),
                "beta": float(row["beta"]),
                "side": side,
                "direction_correct": correct[-1],
                "contract_code": None if selection is None else selection["contract_code"],
                "quantity": None if selection is None else selection["quantity"],
                "status": leg.status,
                "capital_before": before,
                "pnl_cny": leg.pnl_cny,
                "capital_after": capital,
                "entry_notional_cny": leg.entry_notional_cny,
                "entry_minimum_pass": leg.entry_minimum_pass,
            }
        )
    if ready_predictions.empty:
        raise RuntimeError("IVS迁移开发期没有模型可用月份")
    start = pd.Timestamp(ready_predictions.iloc[0]["entry_date"])
    end = pd.Timestamp(ready_predictions.iloc[-1]["exit_date"])
    days = max((end - start).days, 1)
    strategy_cagr = -1.0 if capital <= 0 else (capital / initial) ** (365.2425 / days) - 1.0
    benchmark_total = float(benchmark_close.loc[end] / benchmark_close.loc[start] - 1.0)
    benchmark_cagr = (1.0 + benchmark_total) ** (365.2425 / days) - 1.0
    excess = strategy_cagr - benchmark_cagr
    minimum_gate = bool(minimum_passes) and all(minimum_passes)
    threshold = float(config["evaluation"]["annualized_excess_minimum"])
    metrics = {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "prediction_month_count": len(predictions),
        "model_ready_month_count": len(ready_predictions),
        "direction_accuracy": float(np.mean(correct)) if correct else None,
        "call_signal_count": int(ready_predictions["side"].eq("C").sum()),
        "put_signal_count": int(ready_predictions["side"].eq("P").sum()),
        "initial_cash_cny": initial,
        "final_equity_cny": capital,
        "strategy_total_return": capital / initial - 1.0,
        "strategy_cagr": strategy_cagr,
        "benchmark_total_return": benchmark_total,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": excess,
        "executed_opening_count": len(minimum_passes),
        "all_opening_trades_at_least_5000": minimum_gate,
        "gates": {
            "development_annualized_excess_at_least_20pct": excess >= threshold,
            "minimum_opening_trade_gate": minimum_gate,
            "development_transfer_pass": excess >= threshold and minimum_gate,
        },
    }
    return metrics, ledger


def write_outputs(
    metrics: dict,
    ledger: list[dict[str, Any]],
    predictions: pd.DataFrame,
    config: dict,
    manifest: dict,
) -> None:
    payload = {
        "project_id": config["protocol"]["project_id"],
        "status": (
            "DEVELOPMENT_TRANSFER_PASS_VALIDATION_NOT_AUTHORIZED"
            if metrics["gates"]["development_transfer_pass"]
            else "DEVELOPMENT_TRANSFER_REJECTED_FROZEN"
        ),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "EXTERNAL_50ETF_RULE_510300_DEVELOPMENT_TRANSFER_ONLY",
        "factor_count": 1,
        **metrics,
        "ledger": ledger,
        "safety": config["governance"],
    }
    json_path = ROOT / config["paths"]["result_json"]
    markdown_path = ROOT / config["paths"]["result_markdown"]
    prediction_path = ROOT / config["paths"]["monthly_predictions"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    predictions.to_parquet(prediction_path, index=False)
    markdown_path.write_text(
        "\n".join(
            [
                "# 510300期权IVS外部规则迁移V1开发结果",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 证据：`{payload['evidence_label']}`",
                f"- 因子数：{payload['factor_count']} / 10",
                f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
                f"- 模型可用月：{payload['model_ready_month_count']} / {payload['prediction_month_count']}",
                f"- 方向准确率：{payload['direction_accuracy']:.2%}",
                f"- 策略年化：{payload['strategy_cagr']:.2%}",
                f"- H00300年化：{payload['benchmark_cagr']:.2%}",
                f"- 年化超额：{payload['annualized_excess']:.2%}",
                f"- 期末权益：{payload['final_equity_cny']:.2f}元",
                "",
                f"- 年化超额至少20%：{payload['gates']['development_annualized_excess_at_least_20pct']}",
                f"- 开仓至少5000元：{payload['gates']['minimum_opening_trade_gate']}",
                f"- 开发迁移通过：{payload['gates']['development_transfer_pass']}",
                "",
                "原论文对象为50ETF，本结果只检验同一公式能否迁移到510300，不能借用原论文绩效。",
                "2024验证与2025后最终留出未授权读取；历史成交价包络不等于历史买卖盘。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    from scripts.freeze_510300_option_ivs_transfer_v1 import verify_protocol

    config, manifest = verify_protocol()
    verify_data_loader_protocol()
    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    cutoff = pd.Timestamp(config["dates"]["development_end"])
    panel, benchmark = load_sources(cutoff, vehicle_config)
    daily_ivs = build_daily_ivs(panel, config)
    monthly = build_monthly_rows(benchmark, daily_ivs, config)
    predictions = expanding_predictions(monthly, config)
    metrics, ledger = evaluate_strategy(predictions, panel, benchmark, config)
    write_outputs(metrics, ledger, predictions, config, manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
