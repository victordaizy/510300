"""按分红连续的典型价方向汇总实际人民币成交额。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def factor_frame(data, period=14, recovery=20., balance=50., tolerance=1e-12):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and period >= 2, "成交额方向日历或周期错误")
    raw = data[["high", "low", "close", "dividend", "amount"]].to_numpy(float)
    price_valid = np.isfinite(raw[:, :4]).all(axis=1) & (raw[:, :3] > 0).all(axis=1) & (raw[:, 3] >= 0) & (raw[:, 0] >= raw[:, 2]) & (raw[:, 2] >= raw[:, 1])
    amount_valid = np.isfinite(raw[:, 4]) & (raw[:, 4] >= 0)
    typical = raw[:, :3].mean(axis=1)
    before, after = np.full(len(data), np.nan), np.full(len(data), np.nan)
    direction = np.full(len(data), np.nan)
    pair_valid = np.zeros(len(data), bool)
    for t in range(1, len(data)):
        pair_valid[t] = price_valid[t-1] and price_valid[t] and amount_valid[t]
        if not pair_valid[t]:
            continue
        # 消去两个相邻日共有的指标份额，避免从全历史补一条财富路径。
        before[t] = typical[t-1]+raw[t-1, 3]
        after[t] = (1+raw[t-1, 3]/raw[t-1, 2])*(typical[t]+raw[t, 3])
        change = after[t]-before[t]
        direction[t] = 0. if abs(change) <= tolerance*max(1., abs(before[t]), abs(after[t])) else float(np.sign(change))
    positive = np.where(pair_valid, np.where(direction > 0, raw[:, 4], 0.), np.nan)
    negative = np.where(pair_valid, np.where(direction < 0, raw[:, 4], 0.), np.nan)
    pos = pd.Series(positive).rolling(period, min_periods=period).sum().to_numpy()
    neg = pd.Series(negative).rolling(period, min_periods=period).sum().to_numpy()
    known = pd.Series(pair_valid.astype(int)).rolling(period, min_periods=period).sum().eq(period).to_numpy()
    total = pos+neg
    score = np.full(len(data), np.nan)
    has_directional_amount = known & (total > 0)
    score[has_directional_amount] = 100*pos[has_directional_amount]/total[has_directional_amount]
    require((score[has_directional_amount] >= 0).all() and (score[has_directional_amount] <= 100).all(), "成交额方向占比越界")
    status = np.asarray(np.where(has_directional_amount, "DIRECTIONAL_TURNOVER_AVAILABLE", np.where(known, "NO_VIEW_NO_DIRECTIONAL_TURNOVER", "NO_VIEW_INCOMPLETE_DIRECTIONAL_WINDOW")), dtype=object)
    status[:period] = "NO_VIEW_INSUFFICIENT_DIRECTIONAL_WINDOW"
    previous = np.r_[np.nan, score[:-1]]
    enter = np.isfinite(previous) & np.isfinite(score) & (previous <= recovery) & (score > recovery) & (score < balance)
    exit_flag = np.isfinite(score) & (score >= balance)
    starts = [dates[t-period] if t >= period else pd.NaT for t in range(len(data))]
    return pd.DataFrame({"date": dates, "factor_status": status, "price_window_start": starts,
        "raw_typical_price": typical, "amount_cny": raw[:, 4], "local_previous_typical_wealth": before, "local_current_typical_wealth": after,
        "adjusted_price_direction": direction, "positive_amount_cny": positive, "negative_amount_cny": negative,
        "positive_amount_window_cny": pos, "negative_amount_window_cny": neg, "up_amount_share": score,
        "previous_up_amount_share": previous, "raw_entry": enter.astype(int), "raw_exit": exit_flag})


def rules(factors):
    return {"entry": factors.raw_entry.to_numpy(int), "exit": {1: factors.raw_exit.to_numpy(bool)}}


def attach_factor_context(decisions, factors):
    saved = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
    missing = saved.factor_status.ne("DIRECTIONAL_TURNOVER_AVAILABLE")
    require((saved.loc[missing, "requested_quantity"] <= 0).all(), "成交额方向无观点时请求新增买入")
    hold = missing & saved.requested_quantity.eq(0)
    saved.loc[hold, "reference_weight"] = np.nan
    saved.loc[hold, "action"] = "成交额方向无观点，保持已有份额；原价格及期限退出继续"
    return saved
