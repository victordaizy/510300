"""冻结的八因子跨资产 ETF 轮动信号。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = (
    "momentum_252_skip_21",
    "momentum_126_skip_21",
    "momentum_63_skip_21",
    "momentum_20",
    "low_volatility_20",
    "low_volatility_60",
    "drawdown_252",
    "distance_to_ma200",
)


def build_rotation_targets(panel: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """只用当日及过去的总收益价格形成冻结信号。"""

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    if data[["date", "con_code"]].duplicated().any():
        raise ValueError("ETF面板存在重复证券日期")
    periods, universe = contract["periods"], contract["universe"]
    all_symbols = [*universe["risk_assets"].keys(), *universe["defensive_asset"].keys()]
    calendar = pd.DatetimeIndex(sorted(data["date"].unique()))
    close = data.pivot(index="date", columns="con_code", values="total_return_close").reindex(index=calendar, columns=all_symbols)
    returns = close.pct_change(fill_method=None)
    raw = {
        "momentum_252_skip_21": close.shift(21).divide(close.shift(252)).subtract(1.0),
        "momentum_126_skip_21": close.shift(21).divide(close.shift(126)).subtract(1.0),
        "momentum_63_skip_21": close.shift(21).divide(close.shift(63)).subtract(1.0),
        "momentum_20": close.divide(close.shift(20)).subtract(1.0),
        "low_volatility_20": -returns.rolling(20, min_periods=20).std(),
        "low_volatility_60": -returns.rolling(60, min_periods=60).std(),
        "drawdown_252": close.divide(close.rolling(252, min_periods=252).max()).subtract(1.0),
        "distance_to_ma200": close.divide(close.rolling(200, min_periods=200).mean()).subtract(1.0),
    }
    start, end = pd.Timestamp(periods["evaluation_start"]), pd.Timestamp(periods["evaluation_end"])
    evaluation_calendar = calendar[(calendar >= start) & (calendar <= end)]
    signal_dates = evaluation_calendar[:: int(periods["rebalance_every_trading_days"])]
    risk_assets = list(universe["risk_assets"].keys())
    rows: list[pd.DataFrame] = []
    for name, frame in raw.items():
        long = frame.reindex(index=signal_dates, columns=risk_assets).rename_axis("date").reset_index().melt(
            id_vars="date", var_name="con_code", value_name=name
        )
        rows.append(long)
    features = rows[0]
    for frame in rows[1:]:
        features = features.merge(frame, on=["date", "con_code"], how="outer", validate="one_to_one")
    grouped = features.groupby("date", sort=False)
    weights = contract["formula"]["weights"]
    score = np.zeros(len(features), dtype=float)
    for factor in FACTOR_COLUMNS:
        rank_column = f"rank_{factor}"
        features[rank_column] = grouped[factor].rank(pct=True)
        score += features[rank_column].fillna(0.0).to_numpy(float) * float(weights[factor])
    features["score"] = score
    features["factor_available_count"] = features[list(FACTOR_COLUMNS)].notna().sum(axis=1)
    features["absolute_trend_pass"] = (
        features["factor_available_count"].eq(len(FACTOR_COLUMNS))
        & features["distance_to_ma200"].gt(0.0)
        & features["momentum_126_skip_21"].gt(0.0)
    )
    features.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)
    features["eligible_rank"] = features.loc[features["absolute_trend_pass"]].groupby("date").cumcount().add(1)

    defensive = next(iter(universe["defensive_asset"].keys()))
    retention_rank = int(contract["formula"]["retention_rank"])
    previous: str | None = None
    targets: list[dict] = []
    for date, frame in features.groupby("date", sort=True):
        eligible = frame.loc[frame["absolute_trend_pass"]].sort_values(["score", "con_code"], ascending=[False, True])
        rank_map = dict(zip(eligible["con_code"].astype(str), range(1, len(eligible) + 1), strict=True))
        if previous in rank_map and rank_map[previous] <= retention_rank:
            selected, regime = previous, "RISK_RETAINED"
        elif not eligible.empty:
            selected, regime = str(eligible.iloc[0]["con_code"]), "RISK_TOP_SCORE"
        else:
            selected, regime = defensive, "DEFENSIVE_NO_RISK_TREND"
        targets.append({"signal_date": pd.Timestamp(date), "con_code": selected, "selection_rank": 1, "regime": regime})
        previous = selected if selected != defensive else None
    return features.reset_index(drop=True), pd.DataFrame(targets), evaluation_calendar
