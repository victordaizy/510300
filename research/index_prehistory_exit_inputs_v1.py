"""把指数上市前价格周期作为训练类比样本，不构造上市前ETF账户。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import FEATURES

SOURCE_ROLE = "INDEX_PRICE_CONTINUATION_LABEL_NOT_TRADABLE_ETF_RETURN"


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy().reset_index(drop=True)
    require(data.date.is_monotonic_increasing and data.date.is_unique, "指数日期必须唯一递增")
    require(np.isfinite(data[["open", "high", "low", "close"]]).all().all(), "指数价格缺失")
    require(data[["open", "high", "low", "close"]].gt(0).all().all(), "指数价格必须为正")
    previous = data.close.shift(1)
    simple = data.close / previous - 1
    total_log = np.log(data.close / previous)
    data["overnight_log"] = np.log(data.open / previous)
    data["intraday_log"] = np.log(data.close / data.open)
    for window in [5, 20]:
        data[f"mom{window}"] = total_log.rolling(window).sum()
    data["sma120"] = data.close / data.close.rolling(120).mean() - 1
    data["vol20"] = simple.rolling(20).std(ddof=1) * np.sqrt(242)
    # 沿用原ETF完整因子的最长预热：252个既往日收益；不使用指数成交量。
    data["feature_valid"] = total_log.rolling(252).count().eq(252) & np.isfinite(data[["mom5", "mom20", "sma120", "vol20"]]).all(axis=1)
    difference = data.intraday_log - data.overnight_log
    scale = difference.rolling(60).std(ddof=1) * np.sqrt(60)
    data["d60"] = difference.rolling(60).sum() / scale.replace(0, np.nan)
    data["entry_condition"] = data.d60.gt(1) & data.d60.shift(1).gt(1) & data.feature_valid
    data["price_exit_condition"] = data.d60.lt(0) & data.d60.shift(1).lt(0)
    return data


def natural_price_episodes(data: pd.DataFrame, specification: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """收盘判定、下一开盘开始或结束；截止日未自然结束的周期不给标签。"""
    mode = specification["modes"].get("1", specification["modes"].get(1))
    cycle, pending, last_exit, cycle_number = None, None, -1000000, 0
    episodes, samples = [], []
    for t, row in enumerate(data.itertuples()):
        if pending == "ENTER":
            cycle_number += 1
            cycle = {"source_cycle_id": cycle_number, "entry_index": t, "entry_date": row.date,
                     "entry_origin": data.date.iloc[t - 1], "entry_price": float(row.open), "peak_price": float(row.open), "states": []}
        elif pending == "EXIT":
            require(cycle is not None, "指数自然退出缺少在途周期")
            ended = {k: v for k, v in cycle.items() if k != "states"}
            ended.update(exit_index=t, mature_date=row.date, exit_price=float(row.open), status="NATURALLY_CLOSED")
            episodes.append(ended)
            for state in cycle["states"]:
                early = int(state["origin_index"]) + 1
                if early >= t:
                    continue
                target = float(row.open / data.open.iloc[early] - 1)
                samples.append({**state, "signal": "D60_INTRA", "cycle_id": cycle_number,
                    "early_exit_index": early, "early_exit_date": data.date.iloc[early], "exit_index": t,
                    "mature_date": row.date, "target": target, "source_role": SOURCE_ROLE})
            cycle, last_exit = None, t
        pending = None
        if cycle is not None:
            cycle["peak_price"] = max(cycle["peak_price"], float(row.close))
            gain = row.close / cycle["entry_price"] - 1
            drawdown = row.close / cycle["peak_price"] - 1
            held = t - cycle["entry_index"] + 1
            reasons = []
            if row.price_exit_condition:
                reasons.append("原日内强弱退出")
            if gain <= -mode["loss"]:
                reasons.append("价格较进入开盘下跌百分之六")
            if drawdown <= -mode["trail"]:
                reasons.append("价格从周期高点回落百分之八")
            if held >= mode["days"]:
                reasons.append("达到六十个持仓收盘")
            if reasons:
                cycle["exit_origin"] = row.date
                cycle["exit_reasons"] = "；".join(reasons)
                pending = "EXIT"
            else:
                values = [np.log1p(held), gain, drawdown, 1., row.mom5, row.mom20, row.sma120, row.vol20]
                require(np.isfinite(values).all(), "指数周期状态缺少必要因子")
                cycle["states"].append({"origin_index": t, "origin": row.date, **dict(zip(FEATURES, values))})
        elif row.entry_condition and t - last_exit >= specification["cooldown"]:
            # 训练参考沿用第31轮自然周期；实际第93轮执行仍要求第32轮重新出现进入条件。
            pending = "ENTER"
    if cycle is not None:
        unfinished = {k: v for k, v in cycle.items() if k != "states"}
        unfinished.update(exit_index=None, mature_date=pd.NaT, exit_price=None, status="NO_VIEW_UNCLOSED_AT_SOURCE_CUTOFF")
        episodes.append(unfinished)
    sample_columns = ["signal", "cycle_id", "origin_index", "origin", "early_exit_index", "early_exit_date", "exit_index", "mature_date", "target", "source_role", *FEATURES]
    return pd.DataFrame(episodes), pd.DataFrame(samples, columns=sample_columns)


def pool_samples(index_samples: pd.DataFrame, etf_samples: pd.DataFrame, etf_dates) -> pd.DataFrame:
    dates = pd.DatetimeIndex(etf_dates)
    require(dates.is_unique and dates.is_monotonic_increasing, "ETF拟合日历必须唯一递增")
    blocks = []
    for name, frame in [("INDEX_PRICE", index_samples), ("ETF_REFERENCE", etf_samples)]:
        local = frame.copy()
        if local.empty:
            continue
        require((local.origin < local.early_exit_date).all() and (local.early_exit_date < local.mature_date).all(), "训练周期的产生和成熟时钟错误")
        if name == "INDEX_PRICE":
            require((local.origin < dates[0]).all() and (local.mature_date < dates[0]).all(), "指数训练周期超出ETF上市前范围")
        else:
            require((local.origin >= dates[0]).all(), "ETF参考训练日期不符")
        local["source"] = name
        local["source_cycle_id"] = local.cycle_id.astype(int)
        local["cycle_id"] = [f"{name}_{value:06d}" for value in local.source_cycle_id]
        local["source_origin_index"] = local.origin_index.astype(int)
        local["source_exit_index"] = local.exit_index.astype(int)
        local["exit_index"] = dates.searchsorted(local.mature_date, side="right") - 1
        if name == "ETF_REFERENCE":
            require(np.array_equal(local.exit_index, local.source_exit_index), "ETF原成熟下标与日期不一致")
        else:
            require(local.exit_index.eq(-1).all(), "上市前成熟周期必须早于ETF首行")
        local["source_role"] = SOURCE_ROLE if name == "INDEX_PRICE" else "ETF_REFERENCE_NET_CONTINUATION_LABEL"
        local["row_id"] = local.cycle_id + "_" + local.origin.dt.strftime("%Y%m%d")
        blocks.append(local)
    require(bool(blocks), "没有可登记的训练样本")
    combined = pd.concat(blocks, ignore_index=True).sort_values(["mature_date", "cycle_id", "origin"]).reset_index(drop=True)
    require(combined.row_id.is_unique, "训练来源与周期行编号冲突")
    require(combined.groupby("cycle_id").mature_date.nunique().eq(1).all(), "同一周期存在不同成熟日")
    return combined


def choose_mature_samples(samples: pd.DataFrame, fit_date, recent_cycles: int) -> tuple[pd.DataFrame, list[str]]:
    mature = samples[samples.mature_date <= pd.Timestamp(fit_date)]
    cycle_order = mature[["cycle_id", "mature_date"]].drop_duplicates().sort_values(["mature_date", "cycle_id"])
    chosen_ids = cycle_order.tail(recent_cycles).cycle_id.to_list()
    rows = mature[mature.cycle_id.isin(chosen_ids)].copy().sort_values(["cycle_id", "origin"])
    rows["sample_weight"] = 1 / rows.groupby("cycle_id").row_id.transform("count") if len(rows) else pd.Series(dtype=float)
    return rows, chosen_ids


def monthly_schedule(data: pd.DataFrame, earlier_start: str) -> list[int]:
    anchor = int(np.flatnonzero(data.date.ge(earlier_start))[0]) - 1
    month_first = data.date.dt.to_period("M").ne(data.date.shift(1).dt.to_period("M"))
    return sorted(set([anchor] + [int(t) for t in np.flatnonzero(month_first) if anchor <= t < len(data) - 1]))
