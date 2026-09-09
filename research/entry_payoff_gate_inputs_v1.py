"""从保存单次进入账户构建完整交易标签，按原信号组成熟后学习。"""
from __future__ import annotations
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.intraday_overnight_increment_v1 import require

FEATURES = ["mom5", "mom20", "sma120", "vol20", "vol_ratio", "z20", "rsi2", "close_location", "d60_factor"]
CN = ["五日含分红涨跌", "二十日含分红涨跌", "一百二十日均线偏离", "二十日波动", "二十日相对六十日波动", "二十日均值偏离分数", "两日相对强弱", "当日收盘位置", "六十日日内相对隔夜强弱"]


def full_entry_label(path, dividends, initial_capital):
    """现金、股票及应收计入终值；退出后才确认的已有权益额外计入一次。"""
    require(len(path) > 1 and path.date.is_monotonic_increasing and path.date.is_unique, "单次进入路径不完整")
    first, last = path.iloc[0], path.iloc[-1]
    require(first.filled_quantity > 0 and first.shares_before == 0 and last.filled_quantity < 0 and last.shares == 0, "标签路径不是从实际买入至全部退出")
    require(path.filled_quantity.gt(0).sum() == 1 and path.filled_quantity.lt(0).sum() == 1, "原单次参考路径出现追加或部分退出")
    np.testing.assert_allclose(path.cash+path.shares*path.mark+path.dividend_receivable, path.equity, atol=1e-6, rtol=0)
    expected_pnl = float(path.price_pnl.sum()+path.dividend_recognized.sum()-path.commission.sum()-path.slippage_cost.sum())
    require(abs(expected_pnl-(last.equity-initial_capital)) < 1e-6, "完整进入路径价格分红费用未与终值守恒")
    owners = path.set_index("date").shares
    extra, maturity = 0., pd.Timestamp(last.date)
    for event in dividends.itertuples():
        owned = int(owners.get(event.record_date, 0))
        if owned > 0 and event.ex_date > last.date:
            extra += owned*event.cash_dividend_per_share
            maturity = max(maturity, event.ex_date)
    final = float(last.equity)+extra
    return {"target": final/initial_capital-1, "observed_final_equity": float(last.equity), "extra_owned_dividend_after_exit": extra,
        "complete_final_equity": final, "economic_maturity_date": maturity, "entry_date": first.date, "natural_exit_date": last.date}


def choose_rows(samples, fit_date, recent_groups):
    mature = samples[samples.training_maturity_date.notna() & samples.training_maturity_date.le(pd.Timestamp(fit_date))]
    order = mature[["episode_id", "training_maturity_date"]].drop_duplicates().sort_values(["training_maturity_date", "episode_id"])
    ids = order.tail(recent_groups).episode_id.to_list()
    rows = mature[mature.episode_id.isin(ids)].copy().sort_values(["episode_id", "path_id"])
    rows["sample_weight"] = 1/rows.groupby("episode_id").path_id.transform("count") if len(rows) else pd.Series(dtype=float)
    return rows, ids


def fit_model(rows, cfg):
    x, y, weights = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), "进入模型训练因子或完整周期标签缺失")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale <= 1e-12] = 1
    design = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    fitted = Ridge(alpha=cfg["ridge_alpha"], solver="svd", fit_intercept=True).fit(design, y, sample_weight=weights)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "clip": cfg["feature_clip"]}


def predict(model, values):
    z = np.clip((np.asarray(values)-np.array(model["mean"]))/np.array(model["scale"]), -model["clip"], model["clip"])
    return float(model["intercept"] + z@np.array(model["coefficients"]))


def entry_views(data, d60, models):
    fits = [m["fit_index"] for m in models]
    rows = []
    for t in range(len(data)):
        idx = bisect_right(fits, t)-1
        stored = models[idx] if idx >= 0 else None
        values = [float(data[key].iloc[t]) for key in FEATURES[:-1]] + [float(d60.iloc[t])]
        state, estimate = "NO_VIEW_NO_MATURE_ENTRY_MODEL", float("nan")
        if stored and stored["status"] == "FIT_COMPLETE":
            require(stored["fit_index"] <= t and pd.Timestamp(stored["latest_training_maturity"]) <= data.date.iloc[stored["fit_index"]], "进入模型使用了未来完整交易")
            if np.isfinite(values).all():
                state, estimate = "ENTRY_MODEL_PREDICTION_AVAILABLE", predict(stored["model"], values)
            else:
                state = "NO_VIEW_INCOMPLETE_ENTRY_CONTEXT"
        rows.append({"date": data.date.iloc[t], "entry_model_status": state, "predicted_entry_return": estimate,
            "entry_fit_origin": data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "entry_accepted_by_model": bool(np.isfinite(estimate) and estimate > 0), **dict(zip(FEATURES, values))})
    return pd.DataFrame(rows)
