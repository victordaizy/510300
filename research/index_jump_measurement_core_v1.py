"""分开上午与下午的指数五分钟变差测量，仅供次日研究判断。"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS = {
    "AM": pd.date_range("2000-01-01 09:35", "2000-01-01 11:25", freq="5min").strftime("%H:%M:%S").tolist(),
    "PM": pd.date_range("2000-01-01 13:05", "2000-01-01 14:55", freq="5min").strftime("%H:%M:%S").tolist(),
}
ENDPOINTS = [clock for session in SESSIONS.values() for clock in session]
VALUE_COLUMNS = ["sampled_rv", "sampled_bv", "jump_proxy", "jump_fraction"]


def measure(minute: pd.DataFrame) -> pd.DataFrame:
    """缺少端点、非正价格或零变差时保留NO_VIEW，不填造价格。"""
    frame = minute[["ts_code", "trade_time", "close"]].copy()
    if set(frame.ts_code) != {"000300.SH"}:
        raise ValueError("本测量仅接受沪深300指数000300.SH。")
    frame["trade_time"] = pd.to_datetime(frame.trade_time).astype("datetime64[ns]")
    frame["source_date"] = frame.trade_time.dt.normalize()
    frame["clock"] = frame.trade_time.dt.strftime("%H:%M:%S")
    selected = frame[frame.clock.isin(ENDPOINTS)]
    if selected.duplicated(["source_date", "clock"]).any():
        raise ValueError("同一日期及端点存在重复记录，不能静默选择。")
    dates = pd.DatetimeIndex(sorted(frame.source_date.unique()), name="source_date")
    prices = selected.pivot(index="source_date", columns="clock", values="close").reindex(index=dates, columns=ENDPOINTS)
    good = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    rows = pd.DataFrame(index=dates)
    rows["observed_endpoints"] = prices.notna().sum(axis=1)
    rows["complete_positive_endpoints"] = good
    rows["sampled_rv"] = np.nan
    rows["sampled_bv"] = np.nan
    if good.any():
        rv, bv = np.zeros(int(good.sum())), np.zeros(int(good.sum()))
        for clocks in SESSIONS.values():
            # 两段分别差分，午间与隔夜跳空不会混入相邻收益乘积。
            returns = np.diff(np.log(prices.loc[good, clocks].to_numpy(float)), axis=1)
            n = returns.shape[1]
            assert n == 22
            rv += np.square(returns).sum(axis=1)
            bv += (np.pi / 2) * n / (n - 1) * (np.abs(returns[:, 1:]) * np.abs(returns[:, :-1])).sum(axis=1)
        rows.loc[good, "sampled_rv"] = rv
        rows.loc[good, "sampled_bv"] = bv
    rows["jump_proxy"] = (rows.sampled_rv - rows.sampled_bv).clip(lower=0)
    rows["jump_fraction"] = rows.jump_proxy / rows.sampled_rv.where(rows.sampled_rv.gt(0))
    rows["measurement_known"] = good & rows.sampled_rv.gt(0) & np.isfinite(rows.jump_fraction)
    rows["measurement_status"] = np.where(rows.measurement_known, "KNOWN_RECONSTRUCTED", "NO_VIEW")
    rows["variation_available_at"] = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=30)
    return rows.reset_index()


def scale_measurement(measurement: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """分母只取源日前20个日收益平方均值；当日与未来价格不进入分母。"""
    assert market.date.is_monotonic_increasing and not market.date.duplicated().any()
    variance = pd.DataFrame({"source_date": market.date,
        "source_prior_variance20": market.total_log.pow(2).rolling(20, min_periods=20).mean().shift(1)})
    result = measurement.merge(variance, on="source_date", how="left", validate="one_to_one")
    good = result.measurement_known & result.source_prior_variance20.gt(0)
    result["log_intraday_rv_ratio"] = np.nan
    result.loc[good, "log_intraday_rv_ratio"] = np.log(result.loc[good, "sampled_rv"] / result.loc[good, "source_prior_variance20"])
    result["feature_known"] = good & np.isfinite(result.log_intraday_rv_ratio)
    return result


def implementation_checks() -> dict:
    """使用具有已知答案的路径验证测量与时间隔离。"""
    day = pd.Timestamp("2024-02-01")
    clocks = [day + pd.Timedelta(clock) for clock in ENDPOINTS]
    uniform = np.concatenate([4000 * np.exp(np.arange(23) * .001)] * 2)
    sample = pd.DataFrame({"ts_code": "000300.SH", "trade_time": clocks, "close": uniform})
    measured = measure(sample)
    np.testing.assert_allclose(measured.sampled_rv, [44 * .001**2], atol=1e-16, rtol=0)
    np.testing.assert_allclose(measured.sampled_bv, [np.pi / 2 * 44 * .001**2], atol=1e-16, rtol=0)
    assert measured.jump_fraction.iloc[0] == 0
    scaled = sample.copy()
    scaled["close"] *= 17
    np.testing.assert_allclose(measure(scaled)[VALUE_COLUMNS], measured[VALUE_COLUMNS], atol=1e-14, rtol=0)
    lunch = sample.copy()
    lunch.loc[lunch.trade_time.dt.hour.ge(13), "close"] *= 2
    np.testing.assert_allclose(measure(lunch)[VALUE_COLUMNS], measured[VALUE_COLUMNS], atol=1e-14, rtol=0)
    jump = sample.copy()
    jump["close"] = 4000.
    jump.loc[jump.index.isin(range(11, 23)), "close"] = 4000 * np.exp(-.03)
    one = measure(jump)
    np.testing.assert_allclose(one.sampled_rv, [.03**2], atol=1e-16, rtol=0)
    assert one.jump_fraction.iloc[0] == 1
    flat = sample.copy()
    flat["close"] = 4000.
    assert not measure(flat).measurement_known.iloc[0]
    assert not measure(sample.iloc[1:]).measurement_known.iloc[0]
    future = sample.copy()
    future["trade_time"] += pd.Timedelta(days=1)
    future["close"] *= np.exp(np.linspace(0, 1, len(future)))
    pd.testing.assert_frame_equal(measure(pd.concat([sample, future], ignore_index=True)).iloc[:1], measured)
    dates = pd.date_range(day - pd.Timedelta(days=40), periods=60)
    market = pd.DataFrame({"date": dates, "total_log": np.linspace(.001, .01, len(dates))})
    before = scale_measurement(measured, market)
    changed = market.copy()
    changed.loc[changed.date.ge(day), "total_log"] = 99.
    pd.testing.assert_frame_equal(scale_measurement(measured, changed), before)
    return {"known_uniform_path": True, "single_jump_path": True, "price_unit_invariance": True,
        "no_lunch_or_overnight_return": True, "zero_rv_and_missing_endpoint_NO_VIEW": True,
        "future_minute_exclusion": True, "normalizer_excludes_source_day_and_future": True}
