"""不超过十因子的510300期权方向模型V1：开发期严格走步检验。"""

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
from sklearn.ensemble import HistGradientBoostingClassifier

from research.option_trade_envelope_v1 import (
    ROOT,
    assert_input_hashes as assert_vehicle_input_hashes,
    build_schedule,
    evaluate_leg,
    load_development_data,
    select_contract,
)


CONFIG = ROOT / "config" / "510300_option_ten_factor_direction_v1.yaml"
VEHICLE_CONFIG = ROOT / "config" / "510300_option_trade_envelope_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_vehicle_config() -> dict:
    return yaml.safe_load(VEHICLE_CONFIG.read_text(encoding="utf-8"))


def assert_inputs(config: dict) -> None:
    manifest_path = ROOT / config["protocol"]["upstream_vehicle_manifest"]
    if sha256(manifest_path) != config["protocol"]["upstream_vehicle_manifest_sha256"]:
        raise RuntimeError("上游期权车辆冻结清单哈希漂移")
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != str(item["sha256"]).lower():
            raise RuntimeError(f"十因子冻结输入缺失或哈希漂移：{name}")


def _load_filtered(path: Path, date_column: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[(date_column, "<=", cutoff)])
    frame[date_column] = pd.to_datetime(frame[date_column]).dt.normalize()
    if frame.empty or frame[date_column].max() > cutoff:
        raise RuntimeError(f"输入越过开发期截止日：{path}")
    return frame


def build_iv_features(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    rule = config["iv_cross_section"]
    rows = panel.loc[
        ~panel["is_adjusted"].astype(bool)
        & panel["dte"].between(int(rule["minimum_dte"]), int(rule["maximum_dte"]))
        & panel["volume"].ge(float(rule["minimum_contract_volume"]))
        & panel["implied_volatility"].between(
            float(rule["minimum_iv"]), float(rule["maximum_iv"])
        )
        & panel["delta"].notna()
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["date", "iv_skew_25d", "atm_iv_30d"])
    expiry = (
        rows[["trade_date", "expiry_date", "dte"]]
        .drop_duplicates()
        .assign(distance=lambda frame: (frame["dte"] - int(rule["target_dte"])).abs())
        .sort_values(["trade_date", "distance", "dte", "expiry_date"], kind="mergesort")
        .drop_duplicates("trade_date", keep="first")
        [["trade_date", "expiry_date"]]
    )
    chosen = rows.merge(expiry, on=["trade_date", "expiry_date"], how="inner")
    absolute_delta = chosen["delta"].abs()
    call_25 = chosen.loc[
        chosen["option_type"].eq("C")
        & absolute_delta.between(
            float(rule["delta25_minimum"]), float(rule["delta25_maximum"])
        )
    ].groupby("trade_date")["implied_volatility"].median()
    put_25 = chosen.loc[
        chosen["option_type"].eq("P")
        & absolute_delta.between(
            float(rule["delta25_minimum"]), float(rule["delta25_maximum"])
        )
    ].groupby("trade_date")["implied_volatility"].median()
    atm = chosen.loc[
        absolute_delta.between(
            float(rule["atm_delta_minimum"]), float(rule["atm_delta_maximum"])
        )
    ].groupby("trade_date")["implied_volatility"].median()
    result = pd.concat(
        [(put_25 - call_25).rename("iv_skew_25d"), atm.rename("atm_iv_30d")], axis=1
    ).reset_index(names="date")
    return result.sort_values("date").reset_index(drop=True)


def build_factor_frame(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    statistics: pd.DataFrame,
    etf: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    factors = benchmark[["date", "close"]].copy().sort_values("date")
    close = factors["close"].astype(float)
    log_return = np.log(close).diff()
    factors["f01_total_return_momentum_5d"] = np.log(close / close.shift(5))
    factors["f02_total_return_momentum_20d"] = np.log(close / close.shift(20))
    factors["f03_total_return_momentum_120d"] = np.log(close / close.shift(120))
    factors["f04_drawdown_60d"] = close / close.rolling(60, min_periods=60).max() - 1.0
    rv20 = log_return.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(242.0)
    rv120 = log_return.rolling(120, min_periods=120).std(ddof=1) * math.sqrt(242.0)
    factors["f05_realized_volatility_20d"] = rv20
    factors["f06_realized_volatility_ratio_20_120"] = np.log(rv20 / rv120)
    etf = etf[["date", "high", "low"]].drop_duplicates("date").sort_values("date")
    etf["range_square"] = np.log(etf["high"] / etf["low"]).pow(2)
    etf["f07_parkinson_range_volatility_20d"] = np.sqrt(
        242.0 / (4.0 * math.log(2.0))
        * etf["range_square"].rolling(20, min_periods=20).mean()
    )
    factors = factors.merge(
        etf[["date", "f07_parkinson_range_volatility_20d"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    stats = statistics[["trade_date", "call_volume", "put_volume"]].copy()
    stats = stats.rename(columns={"trade_date": "date"}).drop_duplicates("date")
    stats["f08_log_put_call_volume_ratio"] = np.log(
        (pd.to_numeric(stats["put_volume"], errors="coerce") + 1.0)
        / (pd.to_numeric(stats["call_volume"], errors="coerce") + 1.0)
    )
    factors = factors.merge(
        stats[["date", "f08_log_put_call_volume_ratio"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    iv = build_iv_features(panel, config)
    factors = factors.merge(iv, on="date", how="left", validate="one_to_one")
    factors["f09_25delta_put_call_iv_skew"] = factors["iv_skew_25d"]
    factors["f10_atm_iv_minus_realized_volatility"] = (
        factors["atm_iv_30d"] - factors["f05_realized_volatility_20d"]
    )
    horizon = int(config["target"]["holding_trading_days"])
    entry_lag = int(config["target"]["entry_lag_trading_days"])
    factors["target_start_date"] = factors["date"].shift(-entry_lag)
    factors["target_end_date"] = factors["date"].shift(-(entry_lag + horizon))
    factors["forward_entry_exit_return"] = (
        close.shift(-(entry_lag + horizon)) / close.shift(-entry_lag) - 1.0
    )
    factors["target_positive"] = factors["forward_entry_exit_return"].ge(0).astype(float)
    factors.loc[factors["target_end_date"].isna(), "target_positive"] = np.nan
    expected = list(config["factors"]["columns"])
    if len(expected) != 10 or set(expected).difference(factors.columns):
        raise RuntimeError("十因子列集合不等于冻结定义")
    factors["nonmissing_factor_count"] = factors[expected].notna().sum(axis=1)
    return factors.drop(columns=["iv_skew_25d", "atm_iv_30d"])


def mature_training_rows(
    factors: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    columns = list(config["factors"]["columns"])
    minimum_nonmissing = int(config["factors"]["minimum_nonmissing_per_row"])
    return factors.loc[
        factors["target_end_date"].le(signal_date)
        & factors["target_positive"].notna()
        & factors["nonmissing_factor_count"].ge(minimum_nonmissing),
        ["date", "target_end_date", "target_positive", *columns],
    ].copy()


def model_from_config(config: dict) -> HistGradientBoostingClassifier:
    model = config["model"]
    return HistGradientBoostingClassifier(
        learning_rate=float(model["learning_rate"]),
        max_iter=int(model["max_iter"]),
        max_leaf_nodes=int(model["max_leaf_nodes"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        l2_regularization=float(model["l2_regularization"]),
        max_bins=int(model["max_bins"]),
        early_stopping=bool(model["early_stopping"]),
        random_state=int(model["random_state"]),
    )


def walkforward_predictions(
    factors: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    config: dict,
) -> pd.DataFrame:
    columns = list(config["factors"]["columns"])
    minimum_rows = int(config["model"]["minimum_mature_training_rows"])
    minimum_nonmissing = int(config["factors"]["minimum_nonmissing_per_row"])
    half_life = float(config["model"]["training_weight_half_life_rows"])
    rows: list[dict[str, Any]] = []
    indexed = factors.set_index("date", drop=False)
    for signal_date in signal_dates:
        signal = indexed.loc[signal_date]
        if isinstance(signal, pd.DataFrame):
            raise ValueError("因子日期不唯一")
        training = mature_training_rows(factors, signal_date, config)
        base = {
            "signal_date": signal_date,
            "training_rows": len(training),
            "latest_training_target_end": (
                training["target_end_date"].max() if len(training) else pd.NaT
            ),
            "realized_forward_return": signal["forward_entry_exit_return"],
            "realized_positive": signal["target_positive"],
        }
        if len(training) < minimum_rows:
            rows.append({**base, "status": "NO_MODEL_MINIMUM_TRAINING", "probability_up": np.nan})
            continue
        if int(signal["nonmissing_factor_count"]) < minimum_nonmissing:
            rows.append({**base, "status": "NO_MODEL_SIGNAL_FEATURES", "probability_up": np.nan})
            continue
        if training["target_positive"].nunique() < 2:
            rows.append({**base, "status": "NO_MODEL_SINGLE_CLASS", "probability_up": np.nan})
            continue
        age = np.arange(len(training) - 1, -1, -1, dtype=float)
        weights = np.power(0.5, age / half_life)
        estimator = model_from_config(config)
        estimator.fit(training[columns], training["target_positive"].astype(int), sample_weight=weights)
        probability = float(estimator.predict_proba(signal[columns].to_frame().T)[0, 1])
        rows.append({**base, "status": "MODEL_READY", "probability_up": probability})
    result = pd.DataFrame(rows)
    if len(result) and result["latest_training_target_end"].notna().any():
        leaked = result["latest_training_target_end"].gt(result["signal_date"])
        if leaked.any():
            raise RuntimeError("模型训练标签越过信号日")
    return result


def evaluate_predictions(
    predictions: pd.DataFrame,
    factors: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    model_config: dict,
    vehicle_config: dict,
) -> tuple[dict, list[dict[str, Any]]]:
    schedule = build_schedule(benchmark, vehicle_config)
    schedule_by_signal = {row["signal_date"]: row for row in schedule}
    panel_by_date = {date: rows for date, rows in panel.groupby("trade_date", sort=False)}
    indexed_panel = panel.set_index(["trade_date", "contract_code"]).sort_index()
    benchmark_close = benchmark.set_index("date")["close"]
    initial = float(model_config["execution"]["initial_cash_cny"])
    capital = initial
    threshold = float(model_config["model"]["action_threshold_probability"])
    ledger: list[dict[str, Any]] = []
    direction_pairs: list[tuple[int, int]] = []
    brier_pairs: list[tuple[float, int]] = []
    minimum_passes: list[bool] = []
    for row in predictions.to_dict("records"):
        dates = schedule_by_signal[pd.Timestamp(row["signal_date"])]
        probability = row["probability_up"]
        if not math.isfinite(float(probability)):
            ledger.append(
                {
                    **{key: str(value.date()) for key, value in dates.items()},
                    "status": row["status"],
                    "capital_before": capital,
                    "capital_after": capital,
                }
            )
            continue
        side = "C" if float(probability) >= threshold else "P"
        signal_rows = panel_by_date.get(dates["signal_date"], pd.DataFrame())
        selection = select_contract(signal_rows, side, vehicle_config)
        leg = evaluate_leg(
            selection,
            dates["entry_date"],
            dates["exit_date"],
            indexed_panel,
            capital,
            vehicle_config,
        )
        before = capital
        capital = max(0.0, capital + leg.pnl_cny)
        realized = int(float(row["realized_positive"]))
        predicted = int(side == "C")
        direction_pairs.append((predicted, realized))
        brier_pairs.append((float(probability), realized))
        if leg.executed:
            minimum_passes.append(leg.entry_minimum_pass)
        ledger.append(
            {
                **{key: str(value.date()) for key, value in dates.items()},
                "status": leg.status,
                "probability_up": float(probability),
                "side": side,
                "contract_code": None if selection is None else selection["contract_code"],
                "quantity": None if selection is None else selection["quantity"],
                "realized_positive": realized,
                "direction_correct": predicted == realized,
                "capital_before": before,
                "pnl_cny": leg.pnl_cny,
                "capital_after": capital,
                "entry_notional_cny": leg.entry_notional_cny,
                "entry_minimum_pass": leg.entry_minimum_pass,
            }
        )
    start = schedule[0]["entry_date"]
    end = schedule[-1]["exit_date"]
    days = max((end - start).days, 1)
    strategy_cagr = -1.0 if capital <= 0 else (capital / initial) ** (365.2425 / days) - 1.0
    benchmark_total = float(benchmark_close.loc[end] / benchmark_close.loc[start] - 1.0)
    benchmark_cagr = (1.0 + benchmark_total) ** (365.2425 / days) - 1.0
    annualized_excess = strategy_cagr - benchmark_cagr
    accuracy = (
        float(np.mean([predicted == realized for predicted, realized in direction_pairs]))
        if direction_pairs
        else None
    )
    brier = (
        float(np.mean([(probability - realized) ** 2 for probability, realized in brier_pairs]))
        if brier_pairs
        else None
    )
    all_minimum = bool(minimum_passes) and all(minimum_passes)
    threshold_excess = float(model_config["evaluation"]["annualized_excess_minimum"])
    metrics = {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "signal_count": len(schedule),
        "model_ready_signal_count": len(direction_pairs),
        "call_signal_count": sum(int(row.get("side") == "C") for row in ledger),
        "put_signal_count": sum(int(row.get("side") == "P") for row in ledger),
        "direction_accuracy": accuracy,
        "brier_score": brier,
        "initial_cash_cny": initial,
        "final_equity_cny": capital,
        "strategy_total_return": capital / initial - 1.0,
        "strategy_cagr": strategy_cagr,
        "benchmark_total_return": benchmark_total,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": annualized_excess,
        "all_opening_trades_at_least_5000": all_minimum,
        "gates": {
            "development_annualized_excess_at_least_20pct": annualized_excess
            >= threshold_excess,
            "minimum_opening_trade_gate": all_minimum,
            "development_formula_pass": annualized_excess >= threshold_excess and all_minimum,
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
            "DEVELOPMENT_FORMULA_PASS_VALIDATION_NOT_AUTHORIZED"
            if metrics["gates"]["development_formula_pass"]
            else "DEVELOPMENT_FORMULA_REJECTED_FROZEN"
        ),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "DEVELOPMENT_WALK_FORWARD_NO_VALIDATION_OR_HOLDOUT_READ",
        "factor_count": len(config["factors"]["columns"]),
        **metrics,
        "ledger": ledger,
        "safety": config["governance"],
    }
    json_path = ROOT / config["paths"]["result_json"]
    markdown_path = ROOT / config["paths"]["result_markdown"]
    prediction_path = ROOT / config["paths"]["predictions"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    predictions.to_parquet(prediction_path, index=False)
    markdown_path.write_text(
        "\n".join(
            [
                "# 510300期权十因子方向模型V1开发期结果",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 证据：`{payload['evidence_label']}`",
                f"- 因子数：{payload['factor_count']} / 10",
                f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
                f"- 模型可用信号：{payload['model_ready_signal_count']} / {payload['signal_count']}",
                f"- 方向准确率：{payload['direction_accuracy']:.2%}" if payload["direction_accuracy"] is not None else "- 方向准确率：NO_VIEW",
                f"- 策略几何年化：{payload['strategy_cagr']:.2%}",
                f"- H00300几何年化：{payload['benchmark_cagr']:.2%}",
                f"- 年化超额：{payload['annualized_excess']:.2%}",
                f"- 期末权益：{payload['final_equity_cny']:.2f}元",
                "",
                "## 门槛",
                "",
                f"- 开发期年化超额至少20%：{payload['gates']['development_annualized_excess_at_least_20pct']}",
                f"- 所有开仓至少5000元：{payload['gates']['minimum_opening_trade_gate']}",
                f"- 开发公式通过：{payload['gates']['development_formula_pass']}",
                "",
                "## 边界",
                "",
                "模型只使用在信号日已经成熟的标签逐次重训；2024验证期和2025后最终留出期未读取。",
                "成交仍是次日最高价买入、退出日最低价卖出的历史成交价包络，并非历史买卖盘证明。",
                "开发期通过也不自动授权验证、仓位、订单或实盘。开发期失败后禁止在同一历史上调整本模型。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_option_ten_factor_direction_v1 import verify_protocol

    config, manifest = verify_protocol()
    assert_inputs(config)
    from scripts.freeze_510300_option_trade_envelope_v1 import (
        verify_protocol as verify_vehicle_protocol,
    )

    vehicle_config, _ = verify_vehicle_protocol()
    assert_vehicle_input_hashes(vehicle_config)
    panel, benchmark = load_development_data(vehicle_config)
    cutoff = pd.Timestamp(config["dates"]["development_end"])
    statistics = _load_filtered(
        ROOT / config["inputs"]["option_daily_statistics"]["path"], "trade_date", cutoff
    )
    etf = _load_filtered(ROOT / config["inputs"]["etf_daily"]["path"], "date", cutoff)
    factors = build_factor_frame(panel, benchmark, statistics, etf, config)
    schedule = build_schedule(benchmark, vehicle_config)
    predictions = walkforward_predictions(
        factors, [row["signal_date"] for row in schedule], config
    )
    metrics, ledger = evaluate_predictions(
        predictions, factors, panel, benchmark, config, vehicle_config
    )
    write_outputs(metrics, ledger, predictions, config, manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
