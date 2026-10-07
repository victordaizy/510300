"""日周线价量的首次边界事件学习，只使用拟合时已经成熟的完整参考结果。"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from research.point_account_nr7_inputs_v1 import fill, fee

FEATURES = ["daily_hist_atr", "weekly_hist_atr", "ema_distance_atr", "momentum5_scaled",
            "log_relative_volume", "up_volume_balance5", "log_rv_ratio", "body_efficiency"]
CLASSES = ("LOSS", "PROFIT", "TIMEOUT")
HORIZON = 20
WINDOW = 756
MINIMUM_ROWS = 252


def require(condition, message):
    if not condition:
        raise ValueError(message)


def features(prices, dividends):
    d = prices.reset_index(drop=True).copy()
    require(d.date.is_unique and d.date.is_monotonic_increasing, "日线日期不完整或不唯一。")
    require(np.isfinite(d[["open", "high", "low", "close", "volume"]].to_numpy(float)).all(), "原价量字段有缺失。")
    require((d[["open", "high", "low", "close"]] > 0).all().all() and d.volume.gt(0).all(), "价格或成交量不为正。")
    cash = dividends.groupby("ex_date").cash_dividend_per_share.sum()
    d["dividend"] = d.date.map(cash).fillna(0.)
    d["cash_shift"] = d.dividend.cumsum()
    d["ac"] = d.close + d.cash_shift
    high, low, previous = d.high + d.cash_shift, d.low + d.cash_shift, d.ac.shift()
    tr = pd.concat([high-low, (high-previous).abs(), (low-previous).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14, min_periods=14).mean()
    atr = d.atr14.where(d.atr14.gt(0))
    econ = np.log((d.close+d.dividend)/d.close.shift())
    rv = econ.rolling(20, min_periods=20).std(ddof=1)
    ema12 = d.ac.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = d.ac.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema12-ema26
    hist = dif-dif.ewm(span=9, adjust=False, min_periods=9).mean()
    d["daily_hist_atr"] = hist/atr
    d["ema_distance_atr"] = (d.ac-d.ac.ewm(span=20, adjust=False, min_periods=20).mean())/atr
    d["momentum5_scaled"] = econ.rolling(5, min_periods=5).sum()/(rv*np.sqrt(5))
    d["log_relative_volume"] = np.log(d.volume/d.volume.shift().rolling(20, min_periods=20).median())
    direction_volume = np.sign(econ)*d.volume
    d["up_volume_balance5"] = direction_volume.rolling(5, min_periods=5).sum()/d.volume.rolling(5, min_periods=5).sum()
    d["log_rv_ratio"] = np.log(rv/rv.shift().rolling(252, min_periods=252).median())
    d["body_efficiency"] = (d.close-d.open)/(d.high-d.low).where(d.high.gt(d.low))
    periods = d.date.dt.to_period("W-FRI")
    weeks = d.groupby(periods).agg(weekly_ac=("ac", "last"), weekly_last_date=("date", "last"))
    wdif = weeks.weekly_ac.ewm(span=12, adjust=False, min_periods=12).mean()-weeks.weekly_ac.ewm(span=26, adjust=False, min_periods=26).mean()
    weeks["weekly_hist"] = wdif-wdif.ewm(span=9, adjust=False, min_periods=9).mean()
    weeks["weekly_available_date"] = (weeks.index.to_timestamp(how="end").normalize()+pd.Timedelta(days=1)).astype(d.date.dtype)
    known = pd.merge_asof(d[["date"]], weeks.reset_index(drop=True)[["weekly_available_date", "weekly_last_date", "weekly_hist"]],
                          left_on="date", right_on="weekly_available_date", direction="backward")
    d["weekly_last_date"] = known.weekly_last_date
    d["weekly_available_date"] = known.weekly_available_date
    d["weekly_hist_atr"] = known.weekly_hist/(atr*np.sqrt(5))
    d["first_passage_feature_known"] = np.isfinite(d[FEATURES].to_numpy(float)).all(axis=1)
    known_week = d.weekly_last_date.notna()
    require(d.loc[known_week, "weekly_last_date"].lt(d.loc[known_week, "date"]).all(), "周线包含本周未完成信息。")
    return d


def labels(data, dividends):
    """日收盘首达上2ATR/下1ATR/20收盘到期，下一真实开盘后才成熟。"""
    d = data.reset_index(drop=True)
    rows = []
    ex_indices = {date: idx for idx, date in enumerate(d.date)}
    for origin in range(len(d)):
        row = {"origin_index": origin, "origin": d.date.iloc[origin], "status": "UNMATURED",
               "entry_idx": origin+1, "exit_idx": np.nan, "mature_idx": np.nan, "event_class": None,
               "reference_net_return": np.nan, "barrier_close_idx": np.nan}
        if not np.isfinite(d.atr14.iloc[origin]) or d.atr14.iloc[origin] <= 0 or origin+1 >= len(d):
            row["status"] = "NO_REFERENCE_ORIGIN_OR_FUTURE_OPEN"
            rows.append(row)
            continue
        entry = origin+1
        level = float(d.open.iloc[entry]+d.cash_shift.iloc[entry])
        distance = float(d.atr14.iloc[origin])
        end = min(entry+HORIZON, len(d))
        barrier, kind = None, None
        for t in range(entry, end):
            if d.ac.iloc[t] <= level-distance:
                barrier, kind = t, "LOSS"
                break
            if d.ac.iloc[t] >= level+2*distance:
                barrier, kind = t, "PROFIT"
                break
        if barrier is None and entry+HORIZON <= len(d):
            barrier, kind = entry+HORIZON-1, "TIMEOUT"
        if barrier is None or barrier+1 >= len(d):
            rows.append(row)
            continue
        exit_index = barrier+1
        eligible = dividends.loc[dividends.record_date.ge(d.date.iloc[entry]) & dividends.record_date.lt(d.date.iloc[exit_index])]
        maturity = exit_index
        if len(eligible):
            ex = eligible.ex_date.max()
            if ex not in ex_indices:
                row["status"] = "UNMATURED_OWNED_DIVIDEND"
                rows.append(row)
                continue
            maturity = max(maturity, ex_indices[ex])
        quantity = int(100000/float(d.open.iloc[entry])//100)*100
        require(quantity > 0, "十万元参考原价名义份额不足一手。")
        buy = fill(float(d.open.iloc[entry]), 1, "STRESS")
        sell = fill(float(d.open.iloc[exit_index]), -1, "STRESS")
        debit = quantity*buy+fee(quantity*buy, "STRESS")
        credit = quantity*sell-fee(quantity*sell, "STRESS")+quantity*float(eligible.cash_dividend_per_share.sum())
        row.update(status="MATURE_REFERENCE", exit_idx=exit_index, mature_idx=maturity, event_class=kind,
                   reference_net_return=(credit-debit)/debit, barrier_close_idx=barrier)
        rows.append(row)
    return pd.DataFrame(rows)


def training_pool(data, outcomes, fit_index):
    known = data.first_passage_feature_known.to_numpy(bool)
    mask = outcomes.status.eq("MATURE_REFERENCE") & outcomes.mature_idx.le(fit_index)
    mask &= outcomes.origin_index.ge(max(0, fit_index-WINDOW+1))
    mask &= known[outcomes.origin_index.to_numpy(int)]
    pool = outcomes.loc[mask].copy().sort_values("origin_index")
    if len(pool):
        require(pool.mature_idx.le(fit_index).all() and pool.origin_index.lt(pool.entry_idx).all(), "训练读取未成熟未来结果。")
        concurrency = np.zeros(fit_index+1)
        for row in pool.itertuples(index=False):
            concurrency[int(row.entry_idx):int(row.exit_idx)] += 1
        weights = [float(np.mean(1/concurrency[int(row.entry_idx):int(row.exit_idx)])) for row in pool.itertuples(index=False)]
        require(np.isfinite(weights).all() and (np.asarray(weights) > 0).all(), "重叠标签唯一性权重无效。")
        pool["uniqueness_weight"] = weights
        pool["fit_weight"] = pool.uniqueness_weight/pool.uniqueness_weight.mean()
    else:
        pool["uniqueness_weight"] = pd.Series(dtype=float)
        pool["fit_weight"] = pd.Series(dtype=float)
    return pool


def fit_at(data, outcomes, index):
    pool = training_pool(data, outcomes, index)
    counts = pool.event_class.value_counts()
    record = {"fit_index": int(index), "fit_date": str(data.date.iloc[index].date()),
              "training_origins": pool.origin_index.astype(int).tolist(), "training_rows": len(pool),
              "class_counts": {name: int(counts.get(name, 0)) for name in CLASSES},
              "latest_mature_idx": int(pool.mature_idx.max()) if len(pool) else None,
              "sum_raw_uniqueness_weights": float(pool.uniqueness_weight.sum()),
              "uniqueness_not_independent_event_count": True, "status": "NO_VIEW_TRAINING_SUPPORT", "model": None}
    if len(pool) < MINIMUM_ROWS or any(counts.get(name, 0) < 10 for name in CLASSES):
        return record
    x = data.iloc[pool.origin_index.to_numpy(int)][FEATURES].to_numpy(float)
    w = pool.fit_weight.to_numpy(float)
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale[scale <= 1e-12] = 1.
    z = np.clip((x-mean)/scale, -5., 5.)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, tol=1e-8).fit(z, pool.event_class, sample_weight=w)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        record["status"] = "NO_VIEW_FIT_NOT_CONVERGED"
        return record
    require(tuple(fitted.classes_) == CLASSES, "三类模型身份顺序改变。")
    payoffs = []
    for name in CLASSES:
        part = pool.loc[pool.event_class.eq(name)]
        rw, y = part.fit_weight.to_numpy(float), part.reference_net_return.to_numpy(float)
        payoffs.append({"positive_probability": float(np.average(y > 0, weights=rw)),
                        "negative_probability": float(np.average(y < 0, weights=rw)),
                        "positive_contribution": float(np.average(np.maximum(y, 0), weights=rw)),
                        "negative_contribution": float(np.average(np.maximum(-y, 0), weights=rw))})
    model = {"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "clip": 5.,
             "classes": list(CLASSES), "coef": fitted.coef_.tolist(), "intercept": fitted.intercept_.tolist(),
             "mature_frequency": [float(pool.loc[pool.event_class.eq(name), "fit_weight"].sum()/w.sum()) for name in CLASSES],
             "class_payoffs": payoffs}
    record.update(status="FIT_COMPLETE", model=model)
    return record


def probability(model, values, kind):
    if kind == "MATURE_FREQUENCY":
        return np.asarray(model["mature_frequency"])
    require(kind == "TECHNICAL_LOGIT", "未知固定预测用途。")
    z = np.clip((np.asarray(values)-model["mean"])/model["scale"], -model["clip"], model["clip"])
    return softmax(np.asarray(model["coef"])@z+model["intercept"])


def quality(model, probabilities):
    pay = model["class_payoffs"]
    win = float(probabilities@np.array([v["positive_probability"] for v in pay]))
    loss = float(probabilities@np.array([v["negative_probability"] for v in pay]))
    positive = float(probabilities@np.array([v["positive_contribution"] for v in pay]))
    negative = float(probabilities@np.array([v["negative_contribution"] for v in pay]))
    gain_mean = positive/win if win > 0 else np.nan
    loss_mean = negative/loss if loss > 0 else np.nan
    payoff = gain_mean/loss_mean if np.isfinite(loss_mean) and loss_mean > 0 else np.nan
    return {"predicted_win_probability": win, "predicted_loss_probability": loss,
            "predicted_payoff": payoff, "predicted_p_times_b": win*payoff,
            "predicted_net_expectation": positive-negative}


def forecast(data, outcomes):
    periods = data.date.dt.to_period("M")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp("2015-01-05")))[0])-1
    schedule = sorted({first, *(int(i) for i in np.flatnonzero(periods.ne(periods.shift())) if i > first)})
    records, rows = [], []
    current = None
    cuts = set(schedule)
    for i in range(len(data)):
        if i in cuts:
            current = fit_at(data, outcomes, i)
            records.append(current)
        for kind in ["TECHNICAL_LOGIT", "MATURE_FREQUENCY"]:
            row = {"date": data.date.iloc[i], "origin_index": i, "policy": kind,
                   "fit_index": current["fit_index"] if current else np.nan,
                   "status": "NO_VIEW_NO_MODEL", "entry_event": False, "event_id": None,
                   "atr": float(data.atr14.iloc[i]), "stop_index": np.nan, "target_index": np.nan,
                   **{key: np.nan for key in ["predicted_win_probability", "predicted_loss_probability", "predicted_payoff",
                                             "predicted_p_times_b", "predicted_net_expectation", "p_LOSS", "p_PROFIT", "p_TIMEOUT"]}}
            if current and current["status"] == "FIT_COMPLETE" and data.first_passage_feature_known.iloc[i]:
                p = probability(current["model"], data.iloc[i][FEATURES].to_numpy(float), kind)
                row.update(status="AVAILABLE", **quality(current["model"], p), **dict(zip(["p_"+name for name in CLASSES], p)))
                row["entry_event"] = bool(row["predicted_p_times_b"] > 1. and row["predicted_net_expectation"] > 0.)
                row["event_id"] = f"{kind}_{i}" if row["entry_event"] else None
            rows.append(row)
    return records, pd.DataFrame(rows)
