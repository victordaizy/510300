"""IF期货、300指数与300ETF期权七因子桥梁的开发期检验。"""

from __future__ import annotations

import hashlib
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from research.option_trade_envelope_v1 import (
    build_schedule,
    cagr,
    evaluate_leg,
    select_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_futures_option_bridge_v1.yaml"
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


def load_development_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(config["dates"]["development_end"])
    futures = pd.read_parquet(
        ROOT / config["inputs"]["futures"]["path"], filters=[("date", "<=", cutoff)]
    )
    spot = pd.read_parquet(
        ROOT / config["inputs"]["spot_index"]["path"], filters=[("date", "<=", cutoff)]
    )
    stats = pd.read_parquet(
        ROOT / config["inputs"]["option_statistics"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    eod = pd.read_parquet(
        ROOT / config["inputs"]["option_eod"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    risk = pd.read_parquet(
        ROOT / config["inputs"]["option_risk"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    benchmark = pd.read_parquet(
        ROOT / config["inputs"]["benchmark"]["path"], filters=[("date", "<=", cutoff)]
    )
    futures["date"] = pd.to_datetime(futures["date"]).dt.normalize()
    spot["date"] = pd.to_datetime(spot["date"]).dt.normalize()
    stats["trade_date"] = pd.to_datetime(stats["trade_date"]).dt.normalize()
    eod["trade_date"] = pd.to_datetime(eod["trade_date"]).dt.normalize()
    risk["trade_date"] = pd.to_datetime(risk["trade_date"]).dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    for name, frame, column in (
        ("futures", futures, "date"),
        ("spot", spot, "date"),
        ("option_statistics", stats, "trade_date"),
        ("option_eod", eod, "trade_date"),
        ("option_risk", risk, "trade_date"),
        ("benchmark", benchmark, "date"),
    ):
        if frame.empty or frame[column].max() > cutoff:
            raise RuntimeError(f"{name}越过冻结开发期")
    features = build_features(futures, spot, stats, benchmark, config)
    panel = eod.merge(
        risk[["trade_date", "contract_code", "delta", "implied_volatility"]],
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
    )
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"]).dt.normalize()
    panel["dte"] = (panel["expiry_date"] - panel["trade_date"]).dt.days
    panel["premium_per_contract"] = panel["close"] * panel["contract_unit"]
    benchmark = (
        benchmark[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return features, panel, benchmark


def build_features(
    futures: pd.DataFrame,
    spot: pd.DataFrame,
    stats: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    frame = futures[
        ["date", "close", "volume", "open_interest"]
    ].rename(
        columns={
            "close": "futures_close",
            "volume": "futures_volume",
            "open_interest": "futures_open_interest",
        }
    )
    frame = frame.merge(
        spot[["date", "close"]].rename(columns={"close": "spot_close"}),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        stats[
            [
                "trade_date",
                "call_volume",
                "put_volume",
                "call_open_interest",
                "put_open_interest",
            ]
        ].rename(columns={"trade_date": "date"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["basis"] = np.log(frame["futures_close"] / frame["spot_close"])
    frame["basis_z60"] = (
        frame["basis"] - frame["basis"].rolling(60, min_periods=60).mean()
    ) / frame["basis"].rolling(60, min_periods=60).std(ddof=1)
    frame["basis_change5"] = frame["basis"] - frame["basis"].shift(5)
    frame["futures_open_interest_log_change5"] = np.log(
        frame["futures_open_interest"] / frame["futures_open_interest"].shift(5)
    )
    log_volume = np.log1p(frame["futures_volume"])
    frame["futures_volume_z20"] = (
        log_volume - log_volume.rolling(20, min_periods=20).mean()
    ) / log_volume.rolling(20, min_periods=20).std(ddof=1)
    frame["futures_minus_spot_momentum5"] = np.log(
        frame["futures_close"] / frame["futures_close"].shift(5)
    ) - np.log(frame["spot_close"] / frame["spot_close"].shift(5))
    frame["option_log_put_call_volume"] = np.log(
        (frame["put_volume"] + 1.0) / (frame["call_volume"] + 1.0)
    )
    frame["option_log_put_call_open_interest"] = np.log(
        (frame["put_open_interest"] + 1.0)
        / (frame["call_open_interest"] + 1.0)
    )
    benchmark_lookup = benchmark[["date", "close"]].drop_duplicates("date").sort_values("date")
    dates = benchmark_lookup["date"].tolist()
    closes = benchmark_lookup["close"].astype(float).to_numpy()
    date_to_position = {date: position for position, date in enumerate(dates)}
    entry_lag = int(config["target"]["entry_lag_trading_days"])
    holding = int(config["target"]["holding_trading_days"])
    labels = []
    exits = []
    for date in frame["date"]:
        position = date_to_position.get(date)
        if position is None or position + entry_lag + holding >= len(dates):
            labels.append(np.nan)
            exits.append(pd.NaT)
        else:
            entry_position = position + entry_lag
            exit_position = entry_position + holding
            forward_return = closes[exit_position] / closes[entry_position] - 1.0
            labels.append(float(forward_return >= float(config["target"]["positive_when_return_at_least"])))
            exits.append(dates[exit_position])
    frame["target_up"] = labels
    frame["target_exit_date"] = pd.to_datetime(exits)
    return frame


def fit_predict_probability(
    training: pd.DataFrame,
    signal_row: pd.Series,
    config: dict,
) -> float | None:
    factors = list(config["factors"])
    minimum = int(config["model"]["minimum_mature_training_rows"])
    training = training.dropna(subset=["target_up"]).copy()
    if len(training) < minimum or training["target_up"].nunique() < 2:
        return None
    imputer = SimpleImputer(strategy="median")
    train_x = imputer.fit_transform(training[factors])
    signal_x = imputer.transform(signal_row.to_frame().T[factors])
    model = RandomForestClassifier(
        n_estimators=int(config["model"]["n_estimators"]),
        max_depth=int(config["model"]["max_depth"]),
        min_samples_leaf=int(config["model"]["min_samples_leaf"]),
        max_features=str(config["model"]["max_features"]),
        class_weight=str(config["model"]["class_weight"]),
        random_state=int(config["model"]["random_state"]),
        n_jobs=1,
    )
    model.fit(train_x, training["target_up"].astype(int))
    class_index = list(model.classes_).index(1)
    return float(model.predict_proba(signal_x)[0, class_index])


def run_development(
    config: dict,
    features: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    schedule = build_schedule(benchmark, config)
    if not schedule:
        raise RuntimeError("开发期没有非重叠日程")
    feature_index = features.set_index("date")
    panel_by_date = {
        date: rows for date, rows in panel.groupby("trade_date", sort=False)
    }
    indexed_panel = panel.set_index(["trade_date", "contract_code"]).sort_index()
    benchmark_close = benchmark.set_index("date")["close"]
    initial = float(config["execution_envelope"]["initial_cash_cny"])
    capital = initial
    ledger: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    executed_notionals: list[float] = []
    ready_directions = []
    for period, dates in enumerate(schedule, start=1):
        signal_date = dates["signal_date"]
        if signal_date not in feature_index.index:
            probability = None
        else:
            signal_row = feature_index.loc[signal_date]
            training = features.loc[
                features["target_exit_date"].notna()
                & features["target_exit_date"].lt(signal_date)
            ]
            probability = fit_predict_probability(training, signal_row, config)
        if probability is None:
            side = "CASH_NOT_READY"
        elif probability >= float(config["model"]["call_probability_threshold"]):
            side = "C"
        elif probability <= float(config["model"]["put_probability_threshold"]):
            side = "P"
        else:
            side = "CASH_LOW_CONFIDENCE"
        underlying_return = (
            float(benchmark_close.loc[dates["exit_date"]])
            / float(benchmark_close.loc[dates["entry_date"]])
            - 1.0
        )
        if probability is not None:
            ready_directions.append((probability >= 0.5) == (underlying_return >= 0))
        capital_before = capital
        if side in {"C", "P"}:
            selection = select_contract(
                panel_by_date.get(signal_date, pd.DataFrame()), side, config
            )
            trade = evaluate_leg(
                selection,
                dates["entry_date"],
                dates["exit_date"],
                indexed_panel,
                capital,
                config,
            )
            if trade.executed and not trade.entry_minimum_pass:
                status = "SMALL_OPENING_TRADE_REJECTED"
                executed = False
                pnl = 0.0
            else:
                status = trade.status
                executed = trade.executed
                pnl = trade.pnl_cny
            if executed:
                executed_notionals.append(trade.entry_notional_cny)
                capital = max(0.0, capital + pnl)
        else:
            selection = None
            status = side
            executed = False
            pnl = 0.0
            trade = None
        prediction = {
            "period": period,
            **{key: value for key, value in dates.items()},
            "probability_up": probability,
            "side": side,
            "underlying_return": underlying_return,
        }
        predictions.append(prediction)
        ledger.append(
            {
                **{key: (str(value.date()) if isinstance(value, pd.Timestamp) else value) for key, value in prediction.items()},
                "selection": selection,
                "status": status,
                "executed": executed,
                "pnl_cny": pnl,
                "capital_before": capital_before,
                "capital_after": capital,
                "entry_notional_cny": 0.0 if trade is None else trade.entry_notional_cny,
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
    threshold = float(config["evaluation"]["annualized_excess_threshold"])
    minimum_trades = int(config["evaluation"]["minimum_executed_trades"])
    all_minimum = bool(executed_notionals) and all(
        value >= float(config["execution_envelope"]["minimum_opening_trade_notional_cny"])
        for value in executed_notionals
    )
    gate = excess >= threshold and len(executed_notionals) >= minimum_trades and all_minimum
    result = {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "scheduled_periods": len(schedule),
        "evaluated_periods": len(ledger),
        "loaded_feature_max_date": str(features["date"].max().date()),
        "loaded_option_max_date": str(panel["trade_date"].max().date()),
        "ready_predictions": len(ready_directions),
        "direction_accuracy": float(np.mean(ready_directions)) if ready_directions else None,
        "executed_trades": len(executed_notionals),
        "initial_cash_cny": initial,
        "final_equity_cny": capital,
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": excess,
        "all_executed_openings_at_least_5000": all_minimum,
        "research_gate": {
            "annualized_excess_at_least_20pct": excess >= threshold,
            "executed_trade_count_at_least_minimum": len(executed_notionals) >= minimum_trades,
            "minimum_opening_notional_pass": all_minimum,
            "eligible_for_locked_validation": gate,
        },
        "ledger": ledger,
    }
    return result, pd.DataFrame(predictions)


def write_outputs(
    result: dict,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict,
    manifest: dict,
) -> None:
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
    prediction_path = ROOT / config["paths"]["prediction_parquet"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    features.to_parquet(feature_path, index=False)
    predictions.to_parquet(prediction_path, index=False)
    markdown_path.write_text(
        "\n".join(
            [
                "# 510300期货—期权—现货桥梁V1开发期结果",
                "",
                f"- 状态：`{status}`",
                f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
                f"- 可预测周期：{payload['ready_predictions']}",
                f"- 方向准确率：{payload['direction_accuracy']}",
                f"- 实际成交：{payload['executed_trades']}笔",
                f"- 策略年化：{payload['strategy_cagr']:.2%}",
                f"- H00300全收益基准年化：{payload['benchmark_cagr']:.2%}",
                f"- 几何年化超额：{payload['annualized_excess']:.2%}",
                f"- 期末权益：{payload['final_equity_cny']:.2f}元",
                "- 2024验证期与2025后最终留出期：未载入",
                "",
                "IF0主连仅作为不可交易信息因子。期权按次日最高价买入、退出日最低价卖出，",
                "并计双边费用与5000元开仓门槛；历史买卖盘不可见，结果不是成交证明。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_futures_option_bridge_v1 import verify_protocol

    config, manifest = verify_protocol()
    assert_input_hashes(config)
    features, panel, benchmark = load_development_data(config)
    result, predictions = run_development(config, features, panel, benchmark)
    write_outputs(result, features, predictions, config, manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "ledger"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
