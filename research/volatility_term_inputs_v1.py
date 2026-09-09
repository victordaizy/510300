"""同一美国源日期的两个波动指数，在中国开盘前形成完整状态。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def close_source(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    dates = pd.to_datetime(raw["DATE"], format="%m/%d/%Y", errors="raise").astype("datetime64[ns]")
    values = pd.to_numeric(raw["CLOSE"], errors="raise").astype(float)
    if len(dates) == 0 or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("源日期必须非空、有效、不重复且递增")
    if (values.notna() & (~np.isfinite(values) | values.le(0))).any():
        raise ValueError("已给出的波动指数收盘值必须有限且大于零")
    return pd.DataFrame({"source_date": dates, name: values})


def paired_sources(vix: pd.DataFrame, vix9d: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    month, short = close_source(vix, "vix_close"), close_source(vix9d, "vix9d_close")
    common_start = max(month.source_date.iloc[0], short.source_date.iloc[0])
    pair = month.merge(short, on="source_date", how="outer", validate="one_to_one").sort_values("source_date")
    pair = pair[pair.source_date.between(common_start, pd.Timestamp(cutoff))].reset_index(drop=True)
    if pair.empty:
        raise ValueError("固定历史终点以内没有共同支持区间")
    local_close = pd.DatetimeIndex(pair.source_date) + pd.Timedelta(hours=17)
    pair["available_at"] = local_close.tz_localize("America/New_York").tz_convert("Asia/Shanghai").tz_localize(None).astype("datetime64[ns]")
    return pair


def preopen_frame(stock_dates, pair: pd.DataFrame, maximum_age_days: float = 7.) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(stock_dates)).astype("datetime64[ns]")
    if len(dates) < 2 or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or not dates.equals(dates.normalize()):
        raise ValueError("中国交易日历必须至少两日、无缺失重复、递增且只有日期")
    if not np.isfinite(maximum_age_days) or maximum_age_days <= 0:
        raise ValueError("数据年龄上限必须为有限正数")
    source = pair.copy()
    source["available_at"] = pd.to_datetime(source.available_at).astype("datetime64[ns]")
    if source.available_at.isna().any() or source.available_at.duplicated().any() or not source.available_at.is_monotonic_increasing:
        raise ValueError("来源可用时刻必须有效、唯一且递增")
    left = pd.DataFrame({"date": dates[:-1], "execution_date": dates[1:], "decision_time": dates[1:] + pd.Timedelta(hours=9)})
    states = pd.merge_asof(left, source, left_on="decision_time", right_on="available_at", direction="backward", allow_exact_matches=True)
    states["source_age_days"] = (states.decision_time - states.available_at).dt.total_seconds() / 86400.
    exists = states.source_date.notna()
    complete = states[["vix_close", "vix9d_close"]].notna().all(axis=1)
    timely = states.source_age_days.between(0, maximum_age_days, inclusive="both")
    valid = exists & complete & timely
    states["source_state"] = np.select([~exists, ~complete, ~timely], ["BEFORE_SOURCE", "MISSING_PAIR", "STALE"], default="VIEW")
    states["term_ratio"] = (states.vix9d_close / states.vix_close).where(valid)
    states["target"] = (states.vix9d_close < states.vix_close).astype(float).where(valid)
    states["policy_reason"] = np.select([~valid, states.target.eq(1)], ["资料不足或过期，保留实际份额", "九天预期波动低于30天，持有预算"], default="九天预期波动达到或超过30天，退出预算")
    states = states.set_index("date").reindex(dates).rename_axis("date").reset_index()
    states.loc[len(states) - 1, ["source_state", "policy_reason"]] = ["NOT_USED_NO_NEXT_EXECUTION", "历史终点之后无执行日，不作新决定"]
    return states
