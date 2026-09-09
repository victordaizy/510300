"""第81轮来源修正：用每日实际持仓维持参考入场基准，交易规则不变。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def reference_entry_frame(data, ledger, first):
    """逐行记录已经发生的参考成交，不用未来买入价回填。"""
    dates = pd.DatetimeIndex(data.date)
    require(pd.DatetimeIndex(ledger.date).equals(dates[first:]), "参考成交记录缺少完整账户日历")
    opened = data.wealth.shift(1)*(data.open+data.dividend)/data.previous_close
    entry = np.full(len(data), np.nan)
    buy = np.zeros(len(data), bool)
    active = np.zeros(len(data), bool)
    entry_date = [pd.NaT]*len(data)
    anchor, origin = np.nan, pd.NaT
    for t, row in enumerate(ledger.itertuples(), first):
        if row.shares_before == 0 and row.filled_quantity > 0:
            anchor, origin, buy[t] = float(opened.iloc[t]), dates[t], True
        if row.shares > 0:
            active[t], entry[t], entry_date[t] = True, anchor, origin
        else:
            anchor, origin = np.nan, pd.NaT
    return pd.DataFrame({"date": data.date.to_numpy(), "wealth_close": data.wealth.to_numpy(float),
        "reference_new_buy": buy, "reference_holding": active, "reference_entry_wealth_open": entry,
        "reference_entry_date": pd.to_datetime(entry_date)})


def staged_targets(inputs, reference_state, first):
    state = np.asarray(reference_state, float)
    require(len(inputs) == len(state) and (np.isnan(state) | np.isin(state, [0., 1.])).all(), "参考目标不是零一或缺失")
    require(0 < first < len(inputs), "分批账户起点无准备日")
    target = np.full(len(state), np.nan)
    stages = np.full(len(state), np.nan)
    upgraded = np.zeros(len(state), bool)
    statuses = np.full(len(state), "NO_VIEW_OUTSIDE_DECISION_PERIOD", dtype=object)
    stage = 0.
    for t in range(first-1, len(state)-1):
        row = inputs.iloc[t]
        # 已发生的新参考买入标记新周期，即使本日其他资料不足也重置阶段。
        if row.reference_new_buy:
            stage = .5
        if not np.isfinite(state[t]) or not np.isfinite(row.wealth_close):
            statuses[t] = "NO_VIEW_KEEP_EXISTING_SHARES_AND_STAGE"
        elif state[t] == 0:
            stage, target[t], statuses[t] = 0., 0., "REFERENCE_EXIT_ZERO_TARGET"
        else:
            if stage == 0:
                stage = .5
            if stage == .5 and row.reference_holding and np.isfinite(row.reference_entry_wealth_open) and row.wealth_close > row.reference_entry_wealth_open:
                stage, upgraded[t] = 1., True
            target[t] = stage
            statuses[t] = "FULL_AFTER_PRICE_CONFIRMATION" if stage == 1 else "HALF_WAIT_PRICE_OR_REFERENCE_FILL"
        stages[t] = stage
    result = inputs.copy()
    result["reference_state"], result["stage"], result["target"] = state, stages, target
    result["price_upgrade"], result["stage_status"] = upgraded, statuses
    return result
