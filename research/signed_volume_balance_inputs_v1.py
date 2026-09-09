"""按含分红日涨跌方向汇总真实成交份额；不是公募申购或净资金流。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def signed_volume_balance(data, window=20, flat_tolerance=1e-12):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and window >= 2, "有符号成交量的日历或窗口无效")
    price = data[["close", "previous_close", "dividend"]].to_numpy(float)
    volume = data.volume.to_numpy(float)
    known = np.isfinite(price).all(axis=1) & (price[:, :2] > 0).all(axis=1) & (price[:, 2] >= 0) & np.isfinite(volume) & (volume >= 0)
    change = price[:, 0]+price[:, 2]-price[:, 1]
    scale = np.maximum.reduce([np.ones(len(data)), np.abs(price[:, 0]+price[:, 2]), np.abs(price[:, 1])])
    direction = np.where(known, np.where(np.abs(change) <= flat_tolerance*scale, 0., np.sign(change)), np.nan)
    signed = pd.Series(np.where(known, volume*direction, np.nan))
    valid_volume = pd.Series(np.where(known, volume, np.nan))
    numerator = signed.rolling(window, min_periods=window).sum()
    denominator = valid_volume.rolling(window, min_periods=window).sum()
    complete = pd.Series(known.astype(int)).rolling(window, min_periods=window).sum().eq(window)
    available = complete & denominator.gt(0)
    balance = (numerator/denominator.where(denominator.gt(0))).where(available)
    require(balance.dropna().between(-1.-1e-12, 1.+1e-12).all(), "有符号成交份额占比越界")
    status = np.where(available, "SIGNED_VOLUME_BALANCE_AVAILABLE", np.where(complete, "NO_VIEW_ZERO_WINDOW_VOLUME", "NO_VIEW_INCOMPLETE_VOLUME_WINDOW"))
    return pd.DataFrame({"date": dates, "adjusted_return_direction": direction, "volume_shares": volume,
                         "signed_volume_shares": signed, "signed_volume_window": numerator, "total_volume_window": denominator,
                         "signed_volume_balance20": balance, "factor_status": status})
