"""使用标准HAR式多尺度特征预测510300未来20日实现波动率。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.short_horizon_direction_volatility import add_future_realized_volatility, evaluate_volatility


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "har_volatility_challenger.yaml"
REGISTERED_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
MARKET_STATE_FILE = ROOT / "data" / "features" / "000300_market_state_daily.parquet"
BREADTH_FILE = ROOT / "data" / "features" / "000300_official_weighted_breadth_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_har_volatility_20d_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_har_volatility_20d_report.json"

HAR_FEATURES = [
    "log_rv_5",
    "log_rv_20",
    "log_rv_60",
    "log_parkinson_rv_20",
    "log_component_return_dispersion_20",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_har_data(
    registered: pd.DataFrame,
    market_state: pd.DataFrame,
    breadth: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    base = registered[["date", "etf_close"]].copy()
    state = market_state[
        ["date", "signal_rv_5", "signal_rv_20", "signal_rv_60", "signal_parkinson_rv_20"]
    ].copy()
    internal = breadth[["date", "official_weighted_return_dispersion_1d"]].copy()
    for frame in (base, state, internal):
        frame["date"] = pd.to_datetime(frame["date"])
    data = base.merge(state, on="date", how="inner", validate="one_to_one")
    data = data.merge(internal, on="date", how="inner", validate="one_to_one")
    data["component_return_dispersion_20"] = data[
        "official_weighted_return_dispersion_1d"
    ].rolling(20, min_periods=20).mean() * np.sqrt(242)
    mapping = {
        "log_rv_5": "signal_rv_5",
        "log_rv_20": "signal_rv_20",
        "log_rv_60": "signal_rv_60",
        "log_parkinson_rv_20": "signal_parkinson_rv_20",
        "log_component_return_dispersion_20": "component_return_dispersion_20",
    }
    for output, source in mapping.items():
        data[output] = np.log(data[source].where(data[source].gt(0)))
    return add_future_realized_volatility(data, horizon).replace([np.inf, -np.inf], np.nan)


def _model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def walk_forward_har(
    data: pd.DataFrame,
    horizon: int = 20,
    minimum_training_samples: int = 504,
    refit_interval: int = 5,
) -> pd.DataFrame:
    frame = data.sort_values("date").reset_index(drop=True)
    target = f"future_realized_volatility_{horizon}d"
    model: Pipeline | None = None
    fit_index: int | None = None
    training_count = 0
    latest_training_label_end = pd.NaT
    rows: list[dict[str, object]] = []
    for signal_index in range(len(frame)):
        last_train_signal = signal_index - horizon
        if last_train_signal < 0:
            continue
        train_mask = frame.index.to_series().le(last_train_signal)
        train_mask &= frame[HAR_FEATURES].notna().all(axis=1) & frame[target].gt(0)
        train_indices = frame.index[train_mask]
        if len(train_indices) < minimum_training_samples:
            continue
        if frame.loc[[signal_index], HAR_FEATURES].isna().any(axis=None):
            continue
        if fit_index is None or signal_index - fit_index >= refit_interval:
            model = _model().fit(
                frame.loc[train_indices, HAR_FEATURES],
                np.log(frame.loc[train_indices, target]),
            )
            fit_index = signal_index
            training_count = int(len(train_indices))
            latest_training_label_end = pd.to_datetime(
                frame.loc[train_indices, f"volatility_label_end_date_{horizon}d"]
            ).max()
        if model is None or fit_index is None:
            continue
        predicted = float(
            np.exp(model.predict(frame.loc[[signal_index], HAR_FEATURES])[0])
        )
        current = frame.loc[signal_index]
        rows.append(
            {
                "signal_date": pd.Timestamp(current["date"]),
                "model_fit_date": pd.Timestamp(frame.loc[fit_index, "date"]),
                "latest_training_label_end_date": latest_training_label_end,
                "training_sample_count": training_count,
                "predicted_realized_volatility_20d": predicted,
                "persistence_baseline_volatility_20d": float(current["signal_rv_20"]),
                "actual_realized_volatility_20d": current[target],
                "is_realized": bool(pd.notna(current[target])),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def main() -> int:
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    if registry["status"] != "PRE_REGISTERED_BEFORE_OUTCOME_TEST":
        raise ValueError("HAR波动模型必须先登记再检验")
    data = prepare_har_data(
        pd.read_parquet(REGISTERED_FILE),
        pd.read_parquet(MARKET_STATE_FILE),
        pd.read_parquet(BREADTH_FILE),
    )
    forecasts = walk_forward_har(data)
    if forecasts.empty:
        raise ValueError("没有生成HAR未来波动预测")
    evaluation = evaluate_volatility(forecasts)
    realized = forecasts.loc[forecasts["is_realized"]].copy().reset_index(drop=True)
    midpoint = len(realized) // 2
    halves = {
        "first_half": evaluate_volatility(realized.iloc[:midpoint].copy()),
        "second_half": evaluate_volatility(realized.iloc[midpoint:].copy()),
    }
    gate = registry["acceptance_gate"]
    checks = {
        "minimum_realized_forecasts": evaluation["realized_forecast_count"] >= gate["minimum_realized_forecasts"],
        "overall_mae": evaluation["model_mae"] < evaluation["persistence_baseline_mae"],
        "overall_qlike": evaluation["model_qlike"] < evaluation["persistence_baseline_qlike"],
        "correlation": evaluation["model_actual_correlation"] >= gate["minimum_model_actual_correlation"],
        "each_half_mae": all(
            item["model_mae"] < item["persistence_baseline_mae"] for item in halves.values()
        ),
        "each_half_qlike": all(
            item["model_qlike"] < item["persistence_baseline_qlike"] for item in halves.values()
        ),
        "non_overlapping_mae": evaluation["non_overlapping_cohort_mae_improvement_median"] > 0,
    }
    all_passed = bool(all(checks.values()))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_FILE, index=False)
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    latest = forecasts.iloc[-1]
    report = {
        "status": "HAR_VOLATILITY_ALL_GATES_PASSED" if all_passed else "HAR_VOLATILITY_REJECTED",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "registry_status_before_test": registry["status"],
        "features": registry["features"],
        "model": registry["model"],
        "acceptance_gate": gate,
        "gate_checks": checks,
        "evaluation": evaluation,
        "chronological_halves": halves,
        "latest_forecast": {
            "signal_date": str(pd.Timestamp(latest["signal_date"]).date()),
            "predicted_realized_volatility_20d": float(latest["predicted_realized_volatility_20d"]),
        },
        "trading_use_authorized": False,
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
        "registry_sha256": _sha256(REGISTRY_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
