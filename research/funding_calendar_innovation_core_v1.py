"""在当前两年窗口内，逐个历史时点重建事前资金利率预测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATE_COLUMNS = ["lag1", "mean20"]
CALENDAR_COLUMNS = ["month_first3", "month_last5", "quarter_last5"]
ALL_COLUMNS = [*RATE_COLUMNS, *CALENDAR_COLUMNS]
MIN_FUNDING_TRAIN = 126
RIDGE_ALPHA = 10.


def prepare(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.sort_values("date").reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
    assert frame.date.is_unique and frame.date.is_monotonic_increasing
    frame["source_idx"] = np.arange(len(frame))
    values = frame.fdr007_percent.astype(float)
    assert np.isfinite(values).all()
    frame["lag1"] = values.shift(1)
    frame["mean20"] = values.shift(1).rolling(20, min_periods=20).mean()
    remaining = frame.date.dt.days_in_month - frame.date.dt.day
    frame["month_first3"] = frame.date.dt.day.le(3).astype(float)
    frame["month_last5"] = remaining.lt(5).astype(float)
    frame["quarter_last5"] = (remaining.lt(5) & frame.date.dt.month.isin([3, 6, 9, 12])).astype(float)
    return frame


def prefixes(frame: pd.DataFrame):
    raw = frame[ALL_COLUMNS].to_numpy(float)
    # 前20行没有滞后输入；任何拟合都从窗口起点后20行开始，因此这些占位不入样本。
    x = np.nan_to_num(raw, nan=0.)
    y = frame.fdr007_percent.to_numpy(float)
    def accumulated(a):
        return np.concatenate([np.zeros((1, *a.shape[1:])), np.cumsum(a, axis=0)], axis=0)
    return {"x": raw, "y": y, "sum_x": accumulated(x), "sum_y": accumulated(y),
            "sum_xx": accumulated(x[:, :, None] * x[:, None, :]), "sum_xy": accumulated(x * y[:, None])}


def forecast_curve(frame, sufficient, lower, last_source):
    """每个j的预测只拟合[lower,j)；原始滞后数据也不能早于lower。"""
    raw_start = int(frame.date.searchsorted(pd.Timestamp(lower), side="left"))
    train_start = raw_start + 20
    targets = np.arange(train_start + MIN_FUNDING_TRAIN, int(last_source) + 1, dtype=int)
    if not len(targets):
        return {"source_indices": targets, "raw_start": raw_start, "train_start": train_start,
                "AR": np.array([]), "CALENDAR": np.array([])}
    count = targets - train_start
    sums = {name: sufficient[name][targets] - sufficient[name][train_start]
            for name in ["sum_x", "sum_y", "sum_xx", "sum_xy"]}
    mean_x = sums["sum_x"] / count[:, None]
    mean_y = sums["sum_y"] / count
    covariance = sums["sum_xx"] - count[:, None, None] * mean_x[:, :, None] * mean_x[:, None, :]
    variance = np.maximum(np.diagonal(covariance, axis1=1, axis2=2) / (count - 1)[:, None], 0.)
    scale = np.sqrt(variance)
    scale[scale < 1e-10] = 1.
    centered_xy = sums["sum_xy"] - count[:, None] * mean_x * mean_y[:, None]
    result = {"source_indices": targets, "raw_start": raw_start, "train_start": train_start}
    for name, width in [("AR", 2), ("CALENDAR", 5)]:
        sd = scale[:, :width]
        gram = covariance[:, :width, :width] / (sd[:, :, None] * sd[:, None, :])
        rhs = centered_xy[:, :width] / sd
        beta = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(width)[None, :, :], rhs[:, :, None]).squeeze(-1)
        now_x = (sufficient["x"][targets, :width] - mean_x[:, :width]) / sd
        forecast = mean_y + np.sum(now_x * beta, axis=1)
        result[name] = forecast
        result[name + "_beta"] = beta
        result[name + "_mean"] = mean_x[:, :width]
        result[name + "_scale"] = sd
        result[name + "_intercept"] = mean_y
    return result


def checks():
    dates = pd.bdate_range("2020-01-01", periods=1000)
    remaining = dates.days_in_month - dates.day
    rate = 2 + .2 * (remaining < 5) + .1 * np.sin(np.arange(1000) / 17)
    source = pd.DataFrame({"date": dates, "fdr007_percent": rate})
    frame = prepare(source)
    lower, last = dates[300], 900
    before = forecast_curve(frame, prefixes(frame), lower, last)
    current = len(before["source_indices"]) - 1
    # 当前待预测数值变化不得改变自身预测，后续数值同样不得影响任何过去预测。
    mutated = source.copy()
    mutated.loc[last:, "fdr007_percent"] += 7.
    after_frame = prepare(mutated)
    after = forecast_curve(after_frame, prefixes(after_frame), lower, last)
    for key in ["AR", "CALENDAR"]:
        np.testing.assert_allclose(before[key], after[key], atol=1e-10, rtol=0)
    # 两年起点之前的利率变化不得通过滞后输入偷偷进入拟合。
    mutated = source.copy()
    mutated.loc[:299, "fdr007_percent"] += 7.
    after_frame = prepare(mutated)
    after = forecast_curve(after_frame, prefixes(after_frame), lower, last)
    for key in ["AR", "CALENDAR"]:
        np.testing.assert_allclose(before[key], after[key], atol=1e-8, rtol=0)
    # 与原始训练切片直接求解比较，覆盖批量充分统计量的实现。
    for target in [446, 600, 900]:
        k = int(np.flatnonzero(before["source_indices"] == target)[0])
        raw = frame.loc[320:target-1, ALL_COLUMNS].to_numpy(float)
        y = frame.fdr007_percent.iloc[320:target].to_numpy(float)
        mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=1)
        sd[sd < 1e-10] = 1.
        z = (raw - mean) / sd
        beta = np.linalg.solve(z.T @ z + RIDGE_ALPHA * np.eye(5), z.T @ (y-y.mean()))
        expected = y.mean() + ((frame.loc[target, ALL_COLUMNS].to_numpy(float)-mean)/sd) @ beta
        np.testing.assert_allclose(expected, before["CALENDAR"][k], atol=1e-10, rtol=0)
    assert before["source_indices"][current] == last
    return {"current_and_future_rate_excluded_from_own_forecast": True,
            "all_raw_rate_inputs_inside_current_two_year_window": True,
            "batch_solution_matches_direct_training_slices": 3,
            "calendar_requires_no_future_trading_dates": True}


if __name__ == "__main__":
    print(checks())
