"""用原始报价和分红的有理数比较严格财富均线，消除相等边界的浮点伪突破。"""
from fractions import Fraction
import numpy as np
import pandas as pd


def exact_wealth_ma_relation(market, window=20):
    if window < 2:
        raise ValueError("均线窗口至少为两个交易日")
    prices = market.close.to_numpy(dtype=float)
    dividends = market.dividend.to_numpy(dtype=float)
    finite = np.isfinite(prices)
    if np.any(prices[finite] <= 0):
        raise ValueError("原始价格必须为正")
    ticks = np.rint(prices[finite]*1000)
    if not np.allclose(prices[finite]*1000, ticks, rtol=0, atol=1e-7):
        raise ValueError("原始510300报价不在0.001元价位，不进行隐式修正")
    p = [Fraction(int(round(v*1000)), 1000) if np.isfinite(v) else None for v in prices]
    d = [Fraction(str(float(v))) if np.isfinite(v) else None for v in dividends]
    output = [pd.NA]*len(market)
    for t in range(window-1, len(market)):
        first = t-window+1
        if any(v is None for v in p[first:t+1]) or any(v is None for v in d[first+1:t+1]):
            continue
        relative, total = Fraction(1), Fraction(1)
        for j in range(first+1, t+1):
            relative *= (p[j]+d[j])/p[j-1]
            total += relative
        difference = window*relative-total
        output[t] = 1 if difference > 0 else (-1 if difference < 0 else 0)
    return pd.Series(output, index=market.index, dtype="Int8", name="ma20_exact_relation")


def exact_price_long(market):
    return exact_wealth_ma_relation(market).eq(1).fillna(False).astype(bool).rename("price_long")
