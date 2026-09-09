"""用已发生现金分配和滞后可用资金利率产生明确的进出场目标。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def checked_dates(values, name):
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{name}日期缺失、重复或倒序")
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError(f"{name}日期须为无时区的完整自然日")
    return dates


def known_funding(dates, rates, max_age_days=7):
    """每条利率推迟到其日期之后的首个股票交易日开盘才可使用。"""
    dates = checked_dates(dates, "交易日历")
    source_dates = checked_dates(rates.date, "资金利率")
    values = pd.to_numeric(rates.dr007, errors="raise").to_numpy(float)
    if np.isinf(values).any() or max_age_days < 0:
        raise ValueError("利率无穷大或时效限制为负")
    positions = dates.searchsorted(source_dates, side="right")
    included = positions < len(dates)
    source = pd.DataFrame({"dr_date": source_dates[included], "dr_annual_rate": values[included] / 100,
        "dr_available_at": dates[positions[included]] + pd.Timedelta(hours=9, minutes=30)})
    source = source.drop_duplicates("dr_available_at", keep="last")
    origins = pd.DataFrame({"date": dates, "origin_time": dates + pd.Timedelta(hours=15)})
    joined = pd.merge_asof(origins, source, left_on="origin_time", right_on="dr_available_at", direction="backward")
    joined["dr_age_days"] = (joined.date - joined.dr_date).dt.days
    joined["dr_valid"] = np.isfinite(joined.dr_annual_rate) & joined.dr_age_days.between(0, max_age_days)
    known = joined.dr_date.notna()
    if not (joined.loc[known, "dr_date"] < joined.loc[known, "date"]).all():
        raise ValueError("使用了同日或未来利率")
    return joined


def distribution_frame(data, dividends, rates, coverage_start, coverage_end, max_age_days=7):
    dates = checked_dates(data.date, "行情")
    closes = pd.to_numeric(data.close, errors="raise").to_numpy(float)
    if np.isinf(closes).any() or np.any(closes[np.isfinite(closes)] <= 0):
        raise ValueError("收盘价非正或无穷大")
    events = checked_dates(dividends.ex_date, "除息事件")
    amounts = pd.to_numeric(dividends.cash_dividend_per_share, errors="raise").to_numpy(float)
    if not np.isfinite(amounts).all() or np.any(amounts < 0):
        raise ValueError("每份现金分红缺失、无穷大或为负")
    left = pd.DatetimeIndex([date - pd.DateOffset(years=1) for date in dates])
    complete = (left >= pd.Timestamp(coverage_start)) & (dates <= pd.Timestamp(coverage_end))
    first = events.searchsorted(left, side="right")
    last = events.searchsorted(dates, side="right")
    cumulative = np.r_[0., amounts.cumsum()]
    cash = np.where(complete, cumulative[last] - cumulative[first], np.nan)
    counts = np.where(complete, last - first, np.nan)
    result = known_funding(dates, rates, max_age_days)
    result["close"] = closes
    result["window_start_exclusive"] = left
    result["dividend_coverage_valid"] = complete
    result["past_year_distribution_per_share"] = cash
    result["past_year_distribution_count"] = counts
    result["distribution_yield"] = cash / closes
    valid = np.isfinite(result.distribution_yield) & result.dr_valid
    result["distribution_funding_spread"] = (result.distribution_yield - result.dr_annual_rate).where(valid)
    result["source_state"] = np.select([~complete, ~np.isfinite(closes), ~result.dr_valid],
        ["NO_VIEW_DIVIDEND_COVERAGE", "NO_VIEW_PRICE", "NO_VIEW_FUNDING_MISSING_OR_STALE"], default="VIEW")
    result["target"] = np.where(valid, (result.distribution_funding_spread > 0).astype(float), np.nan)
    return result
