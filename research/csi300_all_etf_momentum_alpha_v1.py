"""点时全ETF宇宙的冻结九因子非线性轮动。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = (
    "momentum_252_skip_21",
    "momentum_126_skip_21",
    "momentum_63_skip_21",
    "momentum_20",
    "breakout_252",
    "trend_efficiency_120",
    "volatility_adjusted_momentum_126",
    "drawdown_60",
    "liquidity_20",
)


def build_all_etf_targets(panel: pd.DataFrame, master: pd.DataFrame, benchmark: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """按点时上市状态、历史长度和流动性构造下一开盘信号。"""

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    meta = master.copy()
    meta["list_date"] = pd.to_datetime(meta["list_date"], errors="coerce")
    meta["delist_date"] = pd.to_datetime(meta["delist_date"], errors="coerce")
    if data[["date", "con_code"]].duplicated().any() or meta["ts_code"].duplicated().any():
        raise ValueError("ETF行情或母表存在重复键")
    periods, universe = contract["periods"], contract["universe"]
    start, end = pd.Timestamp(periods["evaluation_start"]), pd.Timestamp(periods["evaluation_end"])
    bench_dates = pd.to_datetime(benchmark["date"])
    calendar = pd.DatetimeIndex(sorted(bench_dates[(bench_dates >= pd.Timestamp(periods["warmup_start"])) & (bench_dates <= end)].unique()))
    evaluation_calendar = calendar[(calendar >= start) & (calendar <= end)]
    signal_dates = evaluation_calendar[:: int(periods["rebalance_every_trading_days"])]
    defensive = str(universe["defensive_asset"])
    risk_symbols = sorted(code for code in meta["ts_code"].astype(str).unique() if code != defensive)

    def wide(column: str) -> pd.DataFrame:
        return data.pivot(index="date", columns="con_code", values=column).reindex(index=calendar, columns=risk_symbols)

    observed_close = wide("total_return_close")
    close = observed_close.ffill()
    amount = wide("amount").fillna(0.0)
    observed = observed_close.notna() & amount.gt(0.0)
    history_count = observed.cumsum()
    log_return = np.log(close.divide(close.shift(1))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility_60 = log_return.rolling(60, min_periods=60).std()
    momentum_126_log = np.log(close.divide(close.shift(126)))
    raw = {
        "momentum_252_skip_21": np.log(close.shift(21).divide(close.shift(252))),
        "momentum_126_skip_21": np.log(close.shift(21).divide(close.shift(126))),
        "momentum_63_skip_21": np.log(close.shift(21).divide(close.shift(63))),
        "momentum_20": np.log(close.divide(close.shift(20))),
        "breakout_252": close.divide(close.rolling(252, min_periods=252).max()).subtract(1.0),
        "trend_efficiency_120": np.log(close.divide(close.shift(120))).abs().divide(log_return.abs().rolling(120, min_periods=120).sum()),
        "volatility_adjusted_momentum_126": momentum_126_log.divide(volatility_60.mul(np.sqrt(126))),
        "drawdown_60": close.divide(close.rolling(60, min_periods=60).max()).subtract(1.0),
        "liquidity_20": np.log(amount.rolling(20, min_periods=20).mean()),
    }
    ma200_gap = close.divide(close.rolling(200, min_periods=200).mean()).subtract(1.0)
    average_amount20 = amount.rolling(20, min_periods=20).mean()
    available = history_count.ge(int(universe["minimum_history_days"])) & observed & average_amount20.ge(float(universe["minimum_20d_average_amount_cny"]))

    pieces = []
    for symbol in risk_symbols:
        listed = pd.Timestamp(meta.loc[meta["ts_code"].eq(symbol), "list_date"].iloc[0])
        delisted = meta.loc[meta["ts_code"].eq(symbol), "delist_date"].iloc[0]
        symbol_frame = pd.DataFrame({"date": signal_dates, "con_code": symbol})
        for factor, frame in raw.items():
            symbol_frame[factor] = frame.loc[signal_dates, symbol].to_numpy(float)
        symbol_frame["distance_to_ma200"] = ma200_gap.loc[signal_dates, symbol].to_numpy(float)
        symbol_frame["average_amount_20d"] = average_amount20.loc[signal_dates, symbol].to_numpy(float)
        symbol_frame["history_count"] = history_count.loc[signal_dates, symbol].to_numpy(float)
        point_in_time = signal_dates >= listed
        if pd.notna(delisted):
            point_in_time &= signal_dates < pd.Timestamp(delisted)
        symbol_frame["eligible"] = available.loc[signal_dates, symbol].to_numpy(bool) & point_in_time
        pieces.append(symbol_frame)
    features = pd.concat(pieces, ignore_index=True)
    weights = contract["formula"]["linear_weights"]
    for factor in FACTOR_COLUMNS:
        features[f"rank_{factor}"] = features.loc[features["eligible"]].groupby("date")[factor].rank(pct=True)
    features["score"] = sum(features[f"rank_{factor}"].fillna(0.0) * float(weights[factor]) for factor in FACTOR_COLUMNS)
    interactions = contract["formula"]["nonlinear_interactions"]
    features["score"] += float(interactions["momentum_efficiency_rank_product"]) * features["rank_momentum_126_skip_21"].fillna(0.0) * features["rank_trend_efficiency_120"].fillna(0.0)
    features["score"] += float(interactions["medium_short_momentum_rank_product"]) * features["rank_momentum_63_skip_21"].fillna(0.0) * features["rank_momentum_20"].fillna(0.0)
    features["absolute_trend_pass"] = features["eligible"] & features["distance_to_ma200"].gt(0.0) & features["momentum_126_skip_21"].gt(0.0)
    features.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)

    previous: str | None = None
    retention = int(contract["formula"]["retention_rank"])
    targets = []
    for date, frame in features.groupby("date", sort=True):
        eligible = frame.loc[frame["absolute_trend_pass"]].sort_values(["score", "con_code"], ascending=[False, True])
        ranks = dict(zip(eligible["con_code"].astype(str), range(1, len(eligible) + 1), strict=True))
        if previous in ranks and ranks[previous] <= retention:
            selected, regime = previous, "RISK_RETAINED_TOP3"
        elif not eligible.empty:
            selected, regime = str(eligible.iloc[0]["con_code"]), "RISK_TOP_NONLINEAR_SCORE"
        else:
            selected, regime = defensive, "DEFENSIVE_NO_RISK_TREND"
        targets.append({"signal_date": pd.Timestamp(date), "con_code": selected, "selection_rank": 1, "regime": regime})
        previous = selected if selected != defensive else None
    return features.reset_index(drop=True), pd.DataFrame(targets), evaluation_calendar
