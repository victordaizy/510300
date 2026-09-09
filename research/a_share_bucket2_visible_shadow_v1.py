"""桶2固定低波公式的可见Shadow计算核心。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.a_share_hash_holdout_alpha_v1 import (
    FEATURE_COLUMNS,
    ModelRules,
    build_features,
)


class VisibleShadowError(ValueError):
    """Shadow输入或固定规则不满足要求。"""


@dataclass(frozen=True)
class SignalSelection:
    signal_date: pd.Timestamp
    con_code: str
    score: float
    raw_vol20: float
    raw_close: float
    average_amount20: float
    eligible_count: int


def model_rules(config: dict, signal_start: pd.Timestamp, signal_end: pd.Timestamp) -> ModelRules:
    periods = config["periods"]
    universe = config["universe"]
    return ModelRules(
        signal_start=pd.Timestamp(signal_start),
        signal_end=pd.Timestamp(signal_end),
        rebalance_step=int(periods["rebalance_every_trading_days"]),
        horizon=int(periods["target_horizon_trading_days"]),
        minimum_amount=float(universe["minimum_20d_average_amount_cny"]),
        maximum_one_lot=float(universe["maximum_signal_price_for_one_lot_cny"]),
        learning_rate=0.01,
        max_iter=1,
        max_leaf_nodes=2,
        min_samples_leaf=2,
        l2_regularization=0.0,
        max_bins=2,
        random_state=20260819,
        target_clip=(-0.5, 0.5),
        half_life_days=1095.75,
    )


def extend_total_return_history(
    prior_panel: pd.DataFrame,
    raw_increment: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """用前收链接增量行情，并保持旧面板的总收益价格尺度。"""

    required_prior = {
        "date",
        "con_code",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "pre_close",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
        "volume",
        "amount",
    }
    required_increment = {
        "date",
        "con_code",
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "volume",
        "amount",
    }
    if missing := required_prior.difference(prior_panel.columns):
        raise VisibleShadowError(f"旧面板缺少字段：{sorted(missing)}")
    if missing := required_increment.difference(raw_increment.columns):
        raise VisibleShadowError(f"增量行情缺少字段：{sorted(missing)}")
    prior = prior_panel.copy()
    increment = raw_increment.copy()
    prior["date"] = pd.to_datetime(prior["date"])
    increment["date"] = pd.to_datetime(increment["date"])
    if prior[["date", "con_code"]].duplicated().any():
        raise VisibleShadowError("旧面板存在重复证券日期")
    if increment[["date", "con_code"]].duplicated().any():
        raise VisibleShadowError("增量行情存在重复证券日期")
    prior.sort_values(["con_code", "date"], inplace=True)
    increment.sort_values(["con_code", "date"], inplace=True)
    prior_last = prior.groupby("con_code", sort=False).tail(1).set_index("con_code")
    output_rows: list[pd.DataFrame] = []
    for code, frame in increment.groupby("con_code", sort=False):
        frame = frame.copy().sort_values("date")
        previous = prior_last.loc[code] if code in prior_last.index else None
        if previous is not None:
            frame = frame.loc[frame["date"].gt(pd.Timestamp(previous["date"]))].copy()
        if frame.empty:
            continue
        if previous is None:
            factor = 1.0
            previous_raw_close: float | None = None
            previous_total_close: float | None = None
        else:
            factor = float(previous["total_return_close"]) / float(previous["raw_close"])
            previous_raw_close = float(previous["raw_close"])
            previous_total_close = float(previous["total_return_close"])
        factors: list[float] = []
        identity_differences: list[float] = []
        for row in frame.itertuples(index=False):
            if previous_raw_close is not None:
                factor *= previous_raw_close / float(row.pre_close)
            factors.append(factor)
            current_total_close = float(row.raw_close) * factor
            if previous_total_close is None:
                identity_differences.append(0.0)
            else:
                observed = current_total_close / previous_total_close - 1.0
                expected = float(row.raw_close) / float(row.pre_close) - 1.0
                identity_differences.append(abs(observed - expected))
            previous_raw_close = float(row.raw_close)
            previous_total_close = current_total_close
        frame["linked_adjustment_factor"] = factors
        for stem in ("open", "high", "low", "close"):
            frame[f"total_return_{stem}"] = frame[f"raw_{stem}"] * frame["linked_adjustment_factor"]
        frame["return_identity_difference"] = identity_differences
        if frame["return_identity_difference"].max() > tolerance:
            raise VisibleShadowError(f"{code}总收益链接恒等式超过容差")
        frame["is_suspended"] = frame["volume"].fillna(0.0).le(0.0)
        frame["price_source"] = "tushare.daily"
        frame["adjustment_source"] = "tushare.daily_pre_close_link"
        output_rows.append(frame)
    if not output_rows:
        return prior.reset_index(drop=True)
    extended = pd.concat([prior, *output_rows], ignore_index=True, sort=False)
    extended.sort_values(["con_code", "date"], inplace=True)
    if extended[["date", "con_code"]].duplicated().any():
        raise VisibleShadowError("扩展面板产生重复证券日期")
    return extended.reset_index(drop=True)


def is_scheduled_signal_date(
    calendar: pd.DatetimeIndex,
    first_signal_date: pd.Timestamp,
    candidate_date: pd.Timestamp,
    step: int,
) -> bool:
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    first = pd.Timestamp(first_signal_date)
    candidate = pd.Timestamp(candidate_date)
    if first not in dates or candidate not in dates:
        return False
    first_index = int(dates.get_loc(first))
    candidate_index = int(dates.get_loc(candidate))
    distance = candidate_index - first_index
    return distance >= 0 and distance % int(step) == 0


def select_signal(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: dict,
) -> tuple[SignalSelection, pd.DataFrame, pd.DatetimeIndex]:
    """按旧实现的完整资格条件选择桶2最低20日波动证券。"""

    date = pd.Timestamp(signal_date)
    bucket = int(config["split"]["fixed_bucket"])
    bucket_master = master.loc[master["split_bucket"].eq(bucket)].copy()
    bucket_codes = set(bucket_master["ts_code"].astype(str))
    bucket_panel = panel.loc[panel["con_code"].astype(str).isin(bucket_codes)].copy()
    features, calendar = build_features(
        bucket_panel,
        bucket_master,
        benchmark,
        model_rules(config, date, date),
    )
    ready = features.loc[
        features["date"].eq(date)
        & features["signal_output"].eq("SIGNAL_READY")
        & features["raw_close"].ge(float(config["universe"]["minimum_signal_price_cny"]))
    ].copy()
    if ready.empty:
        raise VisibleShadowError(f"{date.date()}没有满足固定条件的桶2候选")
    if ready[list(FEATURE_COLUMNS)].isna().any().any():
        raise VisibleShadowError("候选存在旧10特征缺失")
    ready["score"] = ready[FEATURE_COLUMNS[4]]
    ready.sort_values(["score", "con_code"], ascending=[False, True], inplace=True)
    row = ready.iloc[0]
    selection = SignalSelection(
        signal_date=date,
        con_code=str(row["con_code"]),
        score=float(row["score"]),
        raw_vol20=float(row["raw_vol20"]),
        raw_close=float(row["raw_close"]),
        average_amount20=float(row["average_amount20"]),
        eligible_count=int(len(ready)),
    )
    return selection, features, calendar


def target_frame(selection: SignalSelection) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": selection.signal_date,
                "con_code": selection.con_code,
                "score": selection.score,
                "raw_vol20": selection.raw_vol20,
                "eligible_count": selection.eligible_count,
                "selection_rank": 1,
                "regime": "BUCKET2_VISIBLE_SHADOW_LOWVOL20",
            }
        ]
    )


__all__ = [
    "SignalSelection",
    "VisibleShadowError",
    "extend_total_return_history",
    "is_scheduled_signal_date",
    "model_rules",
    "select_signal",
    "target_frame",
]
