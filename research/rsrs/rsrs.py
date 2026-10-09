"""510300 RSRS 因子：仅生成日线指标和次日交易意图，不执行订单。

输入为按交易日期升序排列的 pandas.DataFrame，至少含 high、low。
数据须为一致复权口径；复权方式及其当日可知性需在外部审计。
价格指标在当日收盘后才可知，最早下一个交易日执行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsrs(
    bars: pd.DataFrame,
    regression_window: int = 18,
    standard_window: int = 600,
) -> pd.DataFrame:
    """标准化RSRS: 对窗口内 high = alpha + beta * low 做 OLS。

    输出 beta、r_squared、zscore、right_skew_rsrs。所有滚动窗
    只用当前位置及更早数据；初始不足样本保留NaN，禁止补零。
    """
    if regression_window < 3 or standard_window < 20:
        raise ValueError("regression_window>=3 且 standard_window>=20")
    if not {"high", "low"}.issubset(bars.columns):
        raise ValueError("需要 high、low 两列")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("交易日索引必须唯一且升序")
    hi = pd.to_numeric(bars["high"], errors="coerce").astype(float)
    lo = pd.to_numeric(bars["low"], errors="coerce").astype(float)
    valid = hi.notna() & lo.notna() & np.isfinite(hi) & np.isfinite(lo) & (hi > 0) & (lo > 0) & (hi >= lo)
    hi = hi.where(valid)
    lo = lo.where(valid)
    n, m = regression_window, standard_window
    cov = hi.rolling(n, min_periods=n).cov(lo)
    var = lo.rolling(n, min_periods=n).var()
    beta = (cov / var.where(var > 1e-14)).replace([np.inf, -np.inf], np.nan)
    corr = hi.rolling(n, min_periods=n).corr(lo)
    r_squared = corr.pow(2).clip(lower=0, upper=1)
    mean = beta.rolling(m, min_periods=m).mean()
    std = beta.rolling(m, min_periods=m).std(ddof=0)
    zscore = ((beta - mean) / std.where(std > 1e-14)).replace([np.inf, -np.inf], np.nan)
    result = pd.DataFrame(index=bars.index)
    result["beta"] = beta
    result["r_squared"] = r_squared
    result["zscore"] = zscore
    result["right_skew_rsrs"] = beta * zscore * r_squared
    return result


def rsrs_target_weight(
    factor: pd.Series,
    enter: float = 0.7,
    exit: float = -0.7,
    long_weight: float = 1.0,
) -> pd.Series:
    """双阈值、无做空、滞回状态机；NaN期间禁开新仓、已持仓不自动清仓。

    当日输出为收盘后目标意图，非当日可交易仓位。
    阈值仅作为预注册初始候选，未经过510300收益验证。
    """
    if not (-np.inf < exit < enter < np.inf):
        raise ValueError("exit 必须小于 enter")
    if not (0 <= long_weight <= 1):
        raise ValueError("long_weight 必须在0到1之间")
    state = 0.0
    weights = []
    for value in factor:
        if pd.notna(value) and np.isfinite(value):
            if value >= enter:
                state = float(long_weight)
            elif value <= exit:
                state = 0.0
        weights.append(state)
    return pd.Series(weights, index=factor.index, name="target_after_close")
