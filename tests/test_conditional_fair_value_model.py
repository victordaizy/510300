"""条件合理价值模型测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.conditional_fair_value_model import (
    FEATURE_COLUMNS,
    build_monthly_features,
    calibrate_with_prior_oos_errors,
    fit_one_forecast,
    walk_forward_forecasts,
)


def test_日频输入可构造无重复字段的月频特征() -> None:
    dates = pd.bdate_range("2020-01-02", periods=400)
    index = pd.DataFrame({"date": dates, "close": 3000.0 + np.arange(len(dates))})
    etf = pd.DataFrame({"date": dates, "close": 3.0 + np.arange(len(dates)) / 1000.0})
    valuation = pd.DataFrame({"date": dates, "pe_ttm": 12.0, "pb": 1.5})
    bonds = pd.DataFrame({"date": dates, "cgb_1y": 2.0, "cgb_10y": 3.0})
    result = build_monthly_features(index, etf, valuation, bonds)
    assert result.columns.is_unique
    assert result["log_eps"].notna().all()


def _monthly(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=rows, freq="ME")
    trend = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "calendar_month": dates.to_period("M"),
            "index_close": 3000.0 + trend * 20.0,
            "etf_close": (3000.0 + trend * 20.0) / 1000.0,
            "etf_index_ratio": 0.001,
            "log_eps": np.log(250.0 + trend * 1.5),
            "log_pe": np.log(12.0 + np.sin(trend / 10.0)),
            "implied_roe": 0.12 + trend * 0.0001,
            "eps_growth_12m": 0.05,
            "cgb_10y": 3.0 - trend * 0.005,
            "cgb_term_spread": 0.5,
            "momentum_3m": 0.02,
            "momentum_6m": 0.04,
            "realized_volatility_3m": 0.18,
        }
    )
    return frame


def test_训练样本必须在信号日前已经实现目标() -> None:
    monthly = _monthly()
    forecast = fit_one_forecast(monthly, signal_index=70, horizon_months=6, minimum_training_samples=20)
    assert forecast is not None
    assert forecast["training_sample_count"] == 65
    assert forecast["target_date"] == monthly.loc[76, "date"]


def test_未来未实现目标仍能输出指定目标月() -> None:
    monthly = _monthly()
    forecast = fit_one_forecast(monthly, signal_index=89, horizon_months=4, minimum_training_samples=20)
    assert forecast is not None
    assert not forecast["is_realized"]
    assert forecast["target_date"] == pd.Timestamp("2025-10-31")
    assert forecast["fair_etf_bear"] < forecast["fair_etf_base"] < forecast["fair_etf_bull"]


def test_多期限滚动预测字段完整且不缺特征() -> None:
    monthly = _monthly()
    assert not monthly[FEATURE_COLUMNS].isna().any().any()
    forecasts = walk_forward_forecasts(monthly, horizons=(4, 12), minimum_training_samples=20)
    assert set(forecasts["horizon_months"]) == {4, 12}
    assert np.allclose(
        forecasts[["bear_probability", "base_probability", "bull_probability"]].sum(axis=1),
        1.0,
    )


def test_校准只使用信号日前已经兑现的样本外误差() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    forecasts = pd.DataFrame(
        {
            "signal_date": dates,
            "target_date": dates + pd.offsets.MonthEnd(2),
            "horizon_months": 2,
            "is_realized": True,
            "actual_target_index": [110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
            "raw_predicted_index": 100.0,
            "current_index": 100.0,
            "current_etf": 1.0,
            "fair_index_bear": 90.0,
            "fair_index_base": 100.0,
            "fair_index_bull": 110.0,
            "fair_etf_bear": 0.9,
            "fair_etf_base": 1.0,
            "fair_etf_bull": 1.1,
            "expected_fair_index": 100.0,
            "expected_fair_etf": 1.0,
            "calibration_source": "TRAINING_IN_SAMPLE_FALLBACK",
            "calibration_sample_count": 1,
        }
    )
    result = calibrate_with_prior_oos_errors(forecasts, minimum_calibration_samples=2)
    assert result.loc[2, "calibration_source"] == "TRAINING_IN_SAMPLE_FALLBACK"
    assert result.loc[4, "calibration_source"] == "PRIOR_PSEUDO_OOS_ERRORS"
    assert result.loc[4, "calibration_sample_count"] == 3
