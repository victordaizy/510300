"""分别预测未来指数EPS与条件PE，形成指定目标月合理价值分布。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
VALUATION_FILE = ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
BOND_FILE = ROOT / "data" / "raw" / "macro" / "china_government_bond_yields_daily.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "000300_conditional_fair_value_forecasts.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "000300_conditional_fair_value_report.json"

FEATURE_COLUMNS = [
    "log_eps",
    "log_pe",
    "implied_roe",
    "eps_growth_12m",
    "cgb_10y",
    "cgb_term_spread",
    "momentum_3m",
    "momentum_6m",
    "realized_volatility_3m",
]


@dataclass(frozen=True)
class ForecastMetrics:
    """单一预测期限的滚动样本外评价。"""

    horizon_months: int
    forecast_count: int
    first_signal_date: str | None
    last_realized_signal_date: str | None
    median_absolute_percentage_error: float | None
    mean_absolute_percentage_error: float | None
    direction_accuracy: float | None
    interval_80_coverage: float | None
    median_predicted_interval_width: float | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_monthly_features(
    index_daily: pd.DataFrame,
    etf_daily: pd.DataFrame,
    valuation: pd.DataFrame,
    bonds: pd.DataFrame,
) -> pd.DataFrame:
    """使用每个自然月最后一个已完成交易日构造指数级特征。"""

    index = index_daily[["date", "close"]].copy().rename(columns={"close": "index_close"})
    etf = etf_daily[["date", "close"]].copy().rename(columns={"close": "etf_close"})
    value = valuation[["date", "pe_ttm", "pb"]].copy()
    bond = bonds[["date", "cgb_1y", "cgb_10y"]].copy()
    for frame in (index, etf, value, bond):
        frame["date"] = pd.to_datetime(frame["date"])
    daily = index.merge(value, on="date", how="left", validate="one_to_one")
    daily = daily.merge(bond, on="date", how="left", validate="one_to_one")
    daily = daily.merge(etf, on="date", how="left", validate="one_to_one")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily[["pe_ttm", "pb", "cgb_1y", "cgb_10y"]] = daily[
        ["pe_ttm", "pb", "cgb_1y", "cgb_10y"]
    ].ffill(limit=5)
    daily["log_return_1d"] = np.log(daily["index_close"] / daily["index_close"].shift(1))
    daily["rv_63d"] = daily["log_return_1d"].rolling(63, min_periods=40).std(ddof=1) * np.sqrt(242)
    daily["calendar_month"] = daily["date"].dt.to_period("M")
    monthly = daily.groupby("calendar_month", sort=True).tail(1).copy().reset_index(drop=True)
    monthly["implied_eps"] = monthly["index_close"] / monthly["pe_ttm"]
    monthly["log_eps"] = np.log(monthly["implied_eps"])
    monthly["log_pe"] = np.log(monthly["pe_ttm"])
    monthly["implied_roe"] = monthly["pb"] / monthly["pe_ttm"]
    monthly["eps_growth_12m"] = monthly["implied_eps"].pct_change(12, fill_method=None).clip(-0.5, 0.5)
    monthly["cgb_term_spread"] = monthly["cgb_10y"] - monthly["cgb_1y"]
    monthly["momentum_3m"] = monthly["index_close"].pct_change(3, fill_method=None)
    monthly["momentum_6m"] = monthly["index_close"].pct_change(6, fill_method=None)
    monthly["realized_volatility_3m"] = monthly["rv_63d"]
    monthly["etf_index_ratio"] = monthly["etf_close"] / monthly["index_close"]
    monthly["target_month_base"] = monthly["calendar_month"]
    numeric = list(dict.fromkeys([
        "index_close", "etf_close", "pe_ttm", "pb", "implied_eps",
        "cgb_1y", "cgb_10y", *FEATURE_COLUMNS,
    ]))
    monthly[numeric] = monthly[numeric].replace([np.inf, -np.inf], np.nan)
    return monthly


def _model() -> Pipeline:
    """固定正则强度，避免在短历史上搜索超参数。"""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def _target_month_end(signal_date: pd.Timestamp, horizon_months: int) -> pd.Timestamp:
    period = signal_date.to_period("M") + horizon_months
    return period.to_timestamp(how="end").normalize()


def fit_one_forecast(
    monthly: pd.DataFrame,
    signal_index: int,
    horizon_months: int,
    minimum_training_samples: int,
) -> dict[str, object] | None:
    """在指定信号行只使用当时已经实现的训练标签。"""

    target_log_eps = monthly["log_eps"].shift(-horizon_months)
    target_log_pe = monthly["log_pe"].shift(-horizon_months)
    last_train_signal = signal_index - horizon_months
    if last_train_signal < 0:
        return None
    candidate = monthly.index.to_series().le(last_train_signal)
    train_mask = candidate & monthly[FEATURE_COLUMNS].notna().all(axis=1)
    train_mask &= target_log_eps.notna() & target_log_pe.notna()
    train_indices = monthly.index[train_mask]
    if len(train_indices) < minimum_training_samples:
        return None
    signal_features = monthly.loc[[signal_index], FEATURE_COLUMNS]
    if signal_features.isna().any(axis=None):
        return None
    x_train = monthly.loc[train_indices, FEATURE_COLUMNS]
    eps_model = _model().fit(x_train, target_log_eps.loc[train_indices])
    pe_model = _model().fit(x_train, target_log_pe.loc[train_indices])
    predicted_log_eps = float(eps_model.predict(signal_features)[0])
    predicted_log_pe = float(pe_model.predict(signal_features)[0])
    joint_residual = (
        target_log_eps.loc[train_indices] - eps_model.predict(x_train)
        + target_log_pe.loc[train_indices] - pe_model.predict(x_train)
    )
    quantiles = joint_residual.quantile([0.10, 0.50, 0.90])
    predicted_log_index = predicted_log_eps + predicted_log_pe
    fair_index_bear = float(np.exp(predicted_log_index + quantiles.loc[0.10]))
    fair_index_base = float(np.exp(predicted_log_index + quantiles.loc[0.50]))
    fair_index_bull = float(np.exp(predicted_log_index + quantiles.loc[0.90]))
    signal = monthly.loc[signal_index]
    ratio = float(signal["etf_index_ratio"]) if pd.notna(signal["etf_index_ratio"]) else np.nan
    target_index = signal_index + horizon_months
    realized = target_index < len(monthly)
    actual_index = float(monthly.loc[target_index, "index_close"]) if realized else np.nan
    return {
        "signal_date": pd.Timestamp(signal["date"]),
        "target_date": (
            pd.Timestamp(monthly.loc[target_index, "date"])
            if realized
            else _target_month_end(pd.Timestamp(signal["date"]), horizon_months)
        ),
        "horizon_months": horizon_months,
        "training_sample_count": int(len(train_indices)),
        "current_index": float(signal["index_close"]),
        "current_etf": float(signal["etf_close"]) if pd.notna(signal["etf_close"]) else np.nan,
        "predicted_eps": float(np.exp(predicted_log_eps)),
        "predicted_fair_pe": float(np.exp(predicted_log_pe)),
        "raw_predicted_index": float(np.exp(predicted_log_index)),
        "fair_index_bear": fair_index_bear,
        "fair_index_base": fair_index_base,
        "fair_index_bull": fair_index_bull,
        "fair_etf_bear": fair_index_bear * ratio,
        "fair_etf_base": fair_index_base * ratio,
        "fair_etf_bull": fair_index_bull * ratio,
        "bear_probability": 0.20,
        "base_probability": 0.60,
        "bull_probability": 0.20,
        "expected_fair_index": 0.20 * fair_index_bear + 0.60 * fair_index_base + 0.20 * fair_index_bull,
        "expected_fair_etf": ratio * (0.20 * fair_index_bear + 0.60 * fair_index_base + 0.20 * fair_index_bull),
        "actual_target_index": actual_index,
        "actual_target_etf": (
            float(monthly.loc[target_index, "etf_close"])
            if realized and pd.notna(monthly.loc[target_index, "etf_close"])
            else np.nan
        ),
        "is_realized": realized,
        "mapping_note": "510300价格使用信号日ETF/指数比率静态映射；未单独预测目标日前基金现金分红。",
        "calibration_source": "TRAINING_IN_SAMPLE_FALLBACK",
        "calibration_sample_count": int(len(joint_residual)),
    }


def calibrate_with_prior_oos_errors(
    forecasts: pd.DataFrame,
    minimum_calibration_samples: int = 20,
) -> pd.DataFrame:
    """只用信号日之前已兑现的滚动样本外误差校准价格区间。"""

    result = forecasts.sort_values(["horizon_months", "signal_date"]).reset_index(drop=True).copy()
    for horizon, indices in result.groupby("horizon_months", sort=False).groups.items():
        history_indices = list(indices)
        for row_index in history_indices:
            signal_date = pd.Timestamp(result.at[row_index, "signal_date"])
            prior = result.loc[
                result.index.isin(history_indices)
                & result["is_realized"]
                & result["actual_target_index"].notna()
                & result["target_date"].le(signal_date)
                & (result.index.to_numpy() < row_index)
            ].copy()
            if len(prior) < minimum_calibration_samples:
                continue
            signed_log_error = np.log(
                prior["actual_target_index"] / prior["raw_predicted_index"]
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(signed_log_error) < minimum_calibration_samples:
                continue
            quantiles = signed_log_error.quantile([0.10, 0.50, 0.90])
            raw = float(result.at[row_index, "raw_predicted_index"])
            bear = float(raw * np.exp(quantiles.loc[0.10]))
            base = float(raw * np.exp(quantiles.loc[0.50]))
            bull = float(raw * np.exp(quantiles.loc[0.90]))
            ratio = float(result.at[row_index, "current_etf"] / result.at[row_index, "current_index"])
            result.at[row_index, "fair_index_bear"] = bear
            result.at[row_index, "fair_index_base"] = base
            result.at[row_index, "fair_index_bull"] = bull
            result.at[row_index, "fair_etf_bear"] = bear * ratio
            result.at[row_index, "fair_etf_base"] = base * ratio
            result.at[row_index, "fair_etf_bull"] = bull * ratio
            result.at[row_index, "expected_fair_index"] = 0.20 * bear + 0.60 * base + 0.20 * bull
            result.at[row_index, "expected_fair_etf"] = ratio * (
                0.20 * bear + 0.60 * base + 0.20 * bull
            )
            result.at[row_index, "calibration_source"] = "PRIOR_PSEUDO_OOS_ERRORS"
            result.at[row_index, "calibration_sample_count"] = int(len(signed_log_error))
    return result


def walk_forward_forecasts(
    monthly: pd.DataFrame,
    horizons: tuple[int, ...] = (4, 6, 12),
    minimum_training_samples: int = 48,
) -> pd.DataFrame:
    """为多个目标月距生成扩展窗口伪样本外预测。"""

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        for signal_index in range(len(monthly)):
            forecast = fit_one_forecast(
                monthly,
                signal_index,
                horizon,
                minimum_training_samples,
            )
            if forecast is not None:
                rows.append(forecast)
    raw = pd.DataFrame(rows).sort_values(["horizon_months", "signal_date"]).reset_index(drop=True)
    return calibrate_with_prior_oos_errors(raw)


def evaluate_forecasts(forecasts: pd.DataFrame, horizon_months: int) -> ForecastMetrics:
    realized = forecasts.loc[
        forecasts["horizon_months"].eq(horizon_months)
        & forecasts["is_realized"]
        & forecasts["actual_target_index"].notna()
    ].copy()
    if realized.empty:
        return ForecastMetrics(horizon_months, 0, None, None, None, None, None, None, None)
    percentage_error = realized["fair_index_base"] / realized["actual_target_index"] - 1.0
    predicted_return = realized["fair_index_base"] / realized["current_index"] - 1.0
    actual_return = realized["actual_target_index"] / realized["current_index"] - 1.0
    direction_accuracy = float((np.sign(predicted_return) == np.sign(actual_return)).mean())
    coverage = realized["actual_target_index"].between(
        realized["fair_index_bear"], realized["fair_index_bull"]
    )
    interval_width = realized["fair_index_bull"] / realized["fair_index_bear"] - 1.0
    return ForecastMetrics(
        horizon_months=horizon_months,
        forecast_count=int(len(realized)),
        first_signal_date=str(realized["signal_date"].min().date()),
        last_realized_signal_date=str(realized["signal_date"].max().date()),
        median_absolute_percentage_error=float(percentage_error.abs().median()),
        mean_absolute_percentage_error=float(percentage_error.abs().mean()),
        direction_accuracy=direction_accuracy,
        interval_80_coverage=float(coverage.mean()),
        median_predicted_interval_width=float(interval_width.median()),
    )


def evaluate_calibrated_subset(forecasts: pd.DataFrame, horizon_months: int) -> dict[str, object]:
    """单独评价已有足够历史样本校准的预测，避免与回退区间混淆。"""

    subset = forecasts.loc[
        forecasts["horizon_months"].eq(horizon_months)
        & forecasts["is_realized"]
        & forecasts["actual_target_index"].notna()
        & forecasts["calibration_source"].eq("PRIOR_PSEUDO_OOS_ERRORS")
    ].copy()
    if subset.empty:
        return {"horizon_months": horizon_months, "forecast_count": 0}
    error = subset["fair_index_base"] / subset["actual_target_index"] - 1.0
    predicted_return = subset["fair_index_base"] / subset["current_index"] - 1.0
    actual_return = subset["actual_target_index"] / subset["current_index"] - 1.0
    return {
        "horizon_months": horizon_months,
        "forecast_count": int(len(subset)),
        "median_absolute_percentage_error": float(error.abs().median()),
        "direction_accuracy": float((np.sign(predicted_return) == np.sign(actual_return)).mean()),
        "interval_80_coverage": float(
            subset["actual_target_index"].between(
                subset["fair_index_bear"], subset["fair_index_bull"]
            ).mean()
        ),
        "median_predicted_interval_width": float(
            (subset["fair_index_bull"] / subset["fair_index_bear"] - 1.0).median()
        ),
    }


def main() -> int:
    monthly = build_monthly_features(
        pd.read_parquet(INDEX_FILE),
        pd.read_parquet(ETF_FILE),
        pd.read_parquet(VALUATION_FILE),
        pd.read_parquet(BOND_FILE),
    )
    forecasts = walk_forward_forecasts(monthly)
    if forecasts.empty:
        raise ValueError("没有生成任何条件合理价值预测")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(OUTPUT_FILE, index=False)
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    metrics = [evaluate_forecasts(forecasts, horizon) for horizon in (4, 6, 12)]
    calibrated_metrics = [
        evaluate_calibrated_subset(forecasts, horizon) for horizon in (4, 6, 12)
    ]
    latest = (
        forecasts.sort_values("signal_date")
        .groupby("horizon_months", as_index=False)
        .tail(1)
    )
    four_month_calibrated = next(
        item for item in calibrated_metrics if item["horizon_months"] == 4
    )
    direction_gate_passed = bool(
        four_month_calibrated.get("forecast_count", 0) >= 20
        and four_month_calibrated.get("direction_accuracy", 0.0) >= 0.55
    )
    report = {
        "status": (
            "PSEUDO_OOS_BASELINE_PASSED_DIRECTION_GATE"
            if direction_gate_passed
            else "REJECTED_AS_ALPHA_SIGNAL_DIRECTION_GATE_FAILED"
        ),
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "model": {
            "fundamental_target": "目标月沪深300隐含EPS",
            "multiple_target": "目标月沪深300PE_TTM",
            "estimator": "固定alpha=10的标准化Ridge；不搜索超参数",
            "interval": "训练期EPS与PE联合对数残差的10%/50%/90%分位",
            "interval_calibration": "有至少20个已兑现预测后，改用信号日前已兑现的pseudo-OOS联合误差分位",
            "training_rule": "信号月只使用目标结果已在该月前实现的训练样本",
        },
        "metrics": [asdict(metric) for metric in metrics],
        "calibrated_subset_metrics": calibrated_metrics,
        "acceptance_gate": {
            "minimum_calibrated_forecasts": 20,
            "minimum_direction_accuracy": 0.55,
            "four_month_direction_gate_passed": direction_gate_passed,
            "trading_use_authorized": False,
        },
        "latest_forecasts": latest[
            [
                "signal_date", "target_date", "horizon_months", "current_etf",
                "predicted_eps", "predicted_fair_pe", "fair_etf_bear",
                "fair_etf_base", "fair_etf_bull", "expected_fair_etf",
                "training_sample_count",
            ]
        ].to_dict("records"),
        "limitations": [
            "当前指数级条件估值MVP未通过方向门槛，不得用于现货仓位。",
            "达到20个历史已兑现预测后改用先前pseudo-OOS误差校准；更早预测仍回退训练内残差。",
            "510300价格映射尚未单独预测基金分红与跟踪误差。",
            "所有现有历史均已参与研究，只能称为pseudo-OOS。",
        ],
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
