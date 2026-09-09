"""独立预测510300未来20日净收益方向概率与实现波动率。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTERED_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
BREADTH_FILE = ROOT / "data" / "features" / "000300_official_weighted_breadth_daily.parquet"
MARKET_STATE_FILE = ROOT / "data" / "features" / "000300_market_state_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_direction_volatility_20d_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "510300_direction_volatility_20d_report.json"

DIRECTION_FEATURES = [
    "factor_etf_total_return_5d",
    "factor_index_return_20d",
    "factor_index_ma20_over_ma60",
    "factor_index_range_position_60d",
    "context_if_minus_index_return_5d",
    "official_weighted_advancer_share_1d",
    "official_weighted_positive_momentum_share_20d",
    "official_weighted_above_ma20_share",
    "official_weighted_above_ma60_share",
    "official_weighted_advancer_share_1d_change_5d",
]

VOLATILITY_FEATURES = [
    "signal_rv_5",
    "signal_rv_20",
    "signal_rv_60",
    "signal_rv_120",
    "signal_downside_rv_20",
    "signal_parkinson_rv_20",
    "signal_vol_term_20_120",
    "official_weighted_return_dispersion_1d",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_future_realized_volatility(data: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """用t+1至t+h的收盘对数收益构造年化实现波动率标签。"""

    result = data.sort_values("date").reset_index(drop=True).copy()
    log_return = np.log(result["etf_close"] / result["etf_close"].shift(1))
    result[f"future_realized_volatility_{horizon}d"] = (
        log_return.rolling(horizon, min_periods=horizon).std(ddof=1).shift(-horizon) * np.sqrt(242)
    )
    result[f"volatility_label_end_date_{horizon}d"] = result["date"].shift(-horizon)
    return result


def prepare_model_data(
    registered: pd.DataFrame,
    breadth: pd.DataFrame,
    market_state: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """合并方向、广度和波动特征，所有输入均以t日收盘为截止点。"""

    base = registered.copy()
    official = breadth.copy()
    state = market_state.copy()
    for frame in (base, official, state):
        frame["date"] = pd.to_datetime(frame["date"])
    breadth_columns = ["date", *[column for column in DIRECTION_FEATURES if column.startswith("official_")], "official_weighted_return_dispersion_1d"]
    state_columns = ["date", *VOLATILITY_FEATURES[:-1]]
    data = base.merge(official[breadth_columns], on="date", how="inner", validate="one_to_one")
    data = data.merge(state[state_columns], on="date", how="inner", validate="one_to_one")
    data = add_future_realized_volatility(data, horizon)
    data["direction_label"] = data[f"exec_total_return_{horizon}d_net"].gt(0).where(
        data[f"exec_total_return_{horizon}d_net"].notna()
    )
    return data.replace([np.inf, -np.inf], np.nan)


def _direction_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000),
            ),
        ]
    )


def _volatility_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])


def walk_forward_models(
    data: pd.DataFrame,
    horizon: int = 20,
    minimum_training_samples: int = 504,
    refit_interval: int = 5,
) -> pd.DataFrame:
    """扩展窗口滚动预测，训练标签必须已在模型拟合日兑现。"""

    frame = data.sort_values("date").reset_index(drop=True).copy()
    volatility_target = f"future_realized_volatility_{horizon}d"
    rows: list[dict[str, object]] = []
    direction_model: Pipeline | None = None
    volatility_model: Pipeline | None = None
    baseline_probability = np.nan
    fit_index: int | None = None
    training_count = 0
    latest_training_label_end = pd.NaT
    for signal_index in range(len(frame)):
        last_train_signal = signal_index - horizon
        if last_train_signal < 0:
            continue
        train_mask = frame.index.to_series().le(last_train_signal)
        train_mask &= frame[DIRECTION_FEATURES + VOLATILITY_FEATURES].notna().all(axis=1)
        train_mask &= frame["direction_label"].notna() & frame[volatility_target].gt(0)
        train_indices = frame.index[train_mask]
        signal_features_valid = not frame.loc[
            [signal_index], DIRECTION_FEATURES + VOLATILITY_FEATURES
        ].isna().any(axis=None)
        if len(train_indices) < minimum_training_samples or not signal_features_valid:
            continue
        should_refit = fit_index is None or signal_index - fit_index >= refit_interval
        if should_refit:
            y_direction = frame.loc[train_indices, "direction_label"].astype(int)
            if y_direction.nunique() < 2:
                continue
            direction_model = _direction_model().fit(
                frame.loc[train_indices, DIRECTION_FEATURES], y_direction
            )
            volatility_model = _volatility_model().fit(
                frame.loc[train_indices, VOLATILITY_FEATURES],
                np.log(frame.loc[train_indices, volatility_target]),
            )
            baseline_probability = float(y_direction.mean())
            fit_index = signal_index
            training_count = int(len(train_indices))
            latest_training_label_end = pd.to_datetime(
                frame.loc[train_indices, f"label_end_date_{horizon}d"]
            ).max()
        if direction_model is None or volatility_model is None or fit_index is None:
            continue
        direction_probability = float(
            direction_model.predict_proba(frame.loc[[signal_index], DIRECTION_FEATURES])[0, 1]
        )
        predicted_volatility = float(
            np.exp(volatility_model.predict(frame.loc[[signal_index], VOLATILITY_FEATURES])[0])
        )
        current = frame.loc[signal_index]
        rows.append(
            {
                "signal_date": pd.Timestamp(current["date"]),
                "model_fit_date": pd.Timestamp(frame.loc[fit_index, "date"]),
                "latest_training_label_end_date": latest_training_label_end,
                "training_sample_count": training_count,
                "direction_probability_positive_20d_net": direction_probability,
                "expanding_base_rate_probability": baseline_probability,
                "predicted_realized_volatility_20d": predicted_volatility,
                "persistence_baseline_volatility_20d": float(current["signal_rv_20"]),
                "actual_direction_positive_20d_net": current["direction_label"],
                "actual_total_return_20d_net": current[f"exec_total_return_{horizon}d_net"],
                "actual_realized_volatility_20d": current[volatility_target],
                "label_end_date": current[f"label_end_date_{horizon}d"],
                "is_realized": bool(
                    pd.notna(current["direction_label"]) and pd.notna(current[volatility_target])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def _calibration_slope(probability: pd.Series, outcome: pd.Series) -> float:
    clipped = probability.clip(1e-6, 1 - 1e-6)
    logit_probability = np.log(clipped / (1.0 - clipped)).to_numpy().reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
        logit_probability, outcome.astype(int)
    )
    return float(model.coef_[0, 0])


def evaluate_direction(forecasts: pd.DataFrame, horizon: int = 20) -> dict[str, object]:
    realized = forecasts.loc[forecasts["is_realized"]].copy().reset_index(drop=True)
    probability = realized["direction_probability_positive_20d_net"]
    baseline_probability = realized["expanding_base_rate_probability"]
    outcome = realized["actual_direction_positive_20d_net"].astype(int)
    model_brier = float(brier_score_loss(outcome, probability))
    baseline_brier = float(brier_score_loss(outcome, baseline_probability))
    bins = pd.qcut(probability, q=5, duplicates="drop")
    calibration = (
        realized.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(
            observations=("actual_direction_positive_20d_net", "size"),
            mean_probability=("direction_probability_positive_20d_net", "mean"),
            realized_hit_rate=("actual_direction_positive_20d_net", "mean"),
            mean_net_return=("actual_total_return_20d_net", "mean"),
        )
        .reset_index(drop=True)
        .to_dict("records")
    )
    cohort_auc: list[float] = []
    for offset in range(horizon):
        subset = realized.iloc[offset::horizon]
        if len(subset) >= 10 and subset["actual_direction_positive_20d_net"].nunique() == 2:
            cohort_auc.append(
                float(
                    roc_auc_score(
                        subset["actual_direction_positive_20d_net"].astype(int),
                        subset["direction_probability_positive_20d_net"],
                    )
                )
            )
    top_bottom_return = (
        float(calibration[-1]["mean_net_return"] - calibration[0]["mean_net_return"])
        if len(calibration) >= 2
        else np.nan
    )
    auc = float(roc_auc_score(outcome, probability))
    brier_skill = 1.0 - model_brier / baseline_brier
    passed = bool(
        len(realized) >= 500
        and brier_skill > 0
        and auc >= 0.53
        and top_bottom_return > 0
        and len(cohort_auc) >= 15
        and np.mean(np.asarray(cohort_auc) > 0.5) >= 0.60
    )
    return {
        "realized_forecast_count": int(len(realized)),
        "positive_base_rate": float(outcome.mean()),
        "brier_score": model_brier,
        "expanding_base_rate_brier_score": baseline_brier,
        "brier_skill_score": float(brier_skill),
        "log_loss": float(log_loss(outcome, probability)),
        "auc": auc,
        "accuracy_at_0_5": float(((probability >= 0.5).astype(int) == outcome).mean()),
        "calibration_slope": _calibration_slope(probability, outcome),
        "probability_quintiles": calibration,
        "top_minus_bottom_mean_net_return": top_bottom_return,
        "non_overlapping_cohort_auc_median": float(np.median(cohort_auc)),
        "non_overlapping_cohort_auc_above_half_share": float(
            np.mean(np.asarray(cohort_auc) > 0.5)
        ),
        "direction_gate_passed": passed,
    }


def _qlike(actual: pd.Series, predicted: pd.Series) -> float:
    ratio = actual / predicted.clip(lower=1e-8)
    return float((ratio - np.log(ratio) - 1.0).mean())


def evaluate_volatility(forecasts: pd.DataFrame, horizon: int = 20) -> dict[str, object]:
    realized = forecasts.loc[forecasts["is_realized"]].copy().reset_index(drop=True)
    actual = realized["actual_realized_volatility_20d"]
    predicted = realized["predicted_realized_volatility_20d"]
    baseline = realized["persistence_baseline_volatility_20d"]
    enhanced_mae = float((predicted - actual).abs().mean())
    baseline_mae = float((baseline - actual).abs().mean())
    enhanced_qlike = _qlike(actual, predicted)
    baseline_qlike = _qlike(actual, baseline)
    cohort_improvements: list[float] = []
    for offset in range(horizon):
        subset = realized.iloc[offset::horizon]
        if len(subset) >= 10:
            cohort_improvements.append(
                float(
                    (subset["persistence_baseline_volatility_20d"] - subset["actual_realized_volatility_20d"]).abs().mean()
                    - (subset["predicted_realized_volatility_20d"] - subset["actual_realized_volatility_20d"]).abs().mean()
                )
            )
    passed = bool(
        len(realized) >= 500
        and enhanced_mae < baseline_mae
        and enhanced_qlike < baseline_qlike
        and float(predicted.corr(actual)) >= 0.50
        and np.median(cohort_improvements) > 0
    )
    return {
        "realized_forecast_count": int(len(realized)),
        "model_mae": enhanced_mae,
        "persistence_baseline_mae": baseline_mae,
        "model_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "persistence_baseline_rmse": float(np.sqrt(np.mean((baseline - actual) ** 2))),
        "model_qlike": enhanced_qlike,
        "persistence_baseline_qlike": baseline_qlike,
        "model_actual_correlation": float(predicted.corr(actual)),
        "persistence_actual_correlation": float(baseline.corr(actual)),
        "non_overlapping_cohort_mae_improvement_median": float(np.median(cohort_improvements)),
        "volatility_gate_passed": passed,
    }


def main() -> int:
    data = prepare_model_data(
        pd.read_parquet(REGISTERED_FILE),
        pd.read_parquet(BREADTH_FILE),
        pd.read_parquet(MARKET_STATE_FILE),
    )
    forecasts = walk_forward_models(data)
    if forecasts.empty:
        raise ValueError("没有生成短周期方向/波动预测")
    direction = evaluate_direction(forecasts)
    volatility = evaluate_volatility(forecasts)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_FILE, index=False)
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    latest = forecasts.iloc[-1]
    report = {
        "status": (
            "DIRECTION_AND_VOLATILITY_GATES_PASSED"
            if direction["direction_gate_passed"] and volatility["volatility_gate_passed"]
            else "DIRECTION_GATE_FAILED_VOLATILITY_GATE_PASSED"
            if not direction["direction_gate_passed"] and volatility["volatility_gate_passed"]
            else "DIRECTION_GATE_PASSED_VOLATILITY_GATE_FAILED"
            if direction["direction_gate_passed"] and not volatility["volatility_gate_passed"]
            else "DIRECTION_AND_VOLATILITY_GATES_FAILED"
        ),
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "design": {
            "direction_target": "t日收盘信号，t+1开盘买入，t+20收盘卖出，含分红并扣基础成本后的净收益是否大于0",
            "volatility_target": "t+1至t+20收盘对数收益标准差年化",
            "minimum_training_samples": 504,
            "refit_interval_trading_days": 5,
            "direction_estimator": "固定C=0.1的标准化L2 Logistic，不搜索超参数",
            "volatility_estimator": "固定alpha=10的标准化Ridge预测对数波动率，不搜索超参数",
            "direction_features": DIRECTION_FEATURES,
            "volatility_features": VOLATILITY_FEATURES,
            "separation_rule": "方向概率与未来波动率分别拟合、分别验收；波动率不生成方向。",
        },
        "direction_evaluation": direction,
        "volatility_evaluation": volatility,
        "latest_forecast": {
            "signal_date": str(pd.Timestamp(latest["signal_date"]).date()),
            "direction_probability_positive_20d_net": float(latest["direction_probability_positive_20d_net"]),
            "predicted_realized_volatility_20d": float(latest["predicted_realized_volatility_20d"]),
            "training_sample_count": int(latest["training_sample_count"]),
        },
        "trading_use_authorized": False,
        "limitations": [
            "全部历史已参与研究，只能称为pseudo-OOS。",
            "20日标签高度重叠，因此同时报告20组错位非重叠诊断。",
            "方向与波动层通过前仍不得映射现货仓位。",
        ],
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
