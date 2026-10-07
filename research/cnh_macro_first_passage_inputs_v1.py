"""同一成熟成员池上的技术、宏观和定盘偏离三模型，原金融规则不变。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from research import macro_technical_first_passage_inputs_v1 as parent

TECH = list(parent.TECH)
MACRO = list(parent.MACRO)
FX = ["fixing_deviation_log_bp", "fixing_deviation_change5_log_bp"]
POLICIES = ("TECH_COMMON_TREE", "MACRO_TECH_TREE", "FX_MACRO_TECH_TREE")
FEATURES = dict(zip(POLICIES, [TECH, TECH + MACRO, TECH + MACRO + FX]))
SEED = parent.SEED
require = parent.require


def attach_fx(macro_data, fx):
    """只接纳原来源合同已知且在收盘决定前可得的两个字段。"""
    data = macro_data.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    data["macro_joint_features_known"] = data.joint_features_known
    source = fx.copy()
    source["date"] = pd.to_datetime(source.date).astype("datetime64[ns]")
    require(source.date.is_unique, "外汇观察日期重复，不能择版本。")
    require(pd.DatetimeIndex(source.date).equals(pd.DatetimeIndex(data.date)), "外汇和原3488日历不一致，不能缩样本。")
    source = source.rename(columns={"conservative_available_at": "fx_available_at",
                                    "both_fields_known": "source_fx_both_fields_known"})
    columns = ["date", "fx_stat_date", "cnh_bid_close", "fixing_value", "fixing_available_at",
               "bar_end_utc", "fx_available_at", "source_age_calendar_days",
               "source_fx_both_fields_known", *FX]
    data = data.merge(source[columns], on="date", how="left", validate="one_to_one")
    data["fx_stat_date"] = pd.to_datetime(data.fx_stat_date).astype("datetime64[ns]")
    data["fx_available_at"] = parent.time_shanghai(data.fx_available_at)
    data["fx_features_known"] = (data.source_fx_both_fields_known.eq(True)
        & data.fx_stat_date.lt(data.date) & data.fx_available_at.le(data.decision_time)
        & data.source_age_calendar_days.between(0, 7)
        & np.isfinite(data[FX].to_numpy(float)).all(axis=1))
    data.loc[~data.fx_features_known, FX] = np.nan
    known = data.fx_features_known
    require(np.allclose(data.loc[known, FX[0]],
                        10000. * np.log(data.loc[known, "cnh_bid_close"] / data.loc[known, "fixing_value"]),
                        rtol=1e-12, atol=1e-10), "偏离单位或分子分母改变。")
    data["joint_features_known"] = data.macro_joint_features_known & data.fx_features_known
    return data


def common_pool(data, outcomes, index):
    return parent.common_pool(data, outcomes, index)


def schedule(data):
    month = data.date.dt.to_period("M")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp("2015-01-05")))[0]) - 1
    return sorted({first, *(int(i) for i in np.flatnonzero(month.ne(month.shift())) if i > first)})


def pool_record(data, pool, index):
    counts = pool.event_class.value_counts()
    supported = len(pool) >= parent.original.MINIMUM_ROWS and all(counts.get(name, 0) >= 10 for name in parent.original.CLASSES)
    return {"fit_index": int(index), "fit_date": data.date.iloc[index],
            "training_origins": pool.origin_index.astype(int).tolist(),
            "training_fit_weights": pool.fit_weight.astype(float).tolist(),
            "training_rows": len(pool),
            "latest_mature_idx": int(pool.mature_idx.max()) if len(pool) else None,
            "class_counts": {name: int(counts.get(name, 0)) for name in parent.original.CLASSES},
            "sum_raw_uniqueness_weights": float(pool.uniqueness_weight.sum()),
            "common_support_sufficient": bool(supported),
            "uniqueness_is_not_independent_sample_count": True,
            "status": "NO_VIEW_COMMON_TRAINING_SUPPORT", "models": {}}


def fit_three(data, outcomes, index):
    pool = common_pool(data, outcomes, index)
    record = pool_record(data, pool, index)
    if not record["common_support_sufficient"]:
        return record
    payoffs = parent.class_payoffs(pool)
    for policy, columns in FEATURES.items():
        x = data.iloc[pool.origin_index.to_numpy(int)][columns].to_numpy(float)
        require(np.isfinite(x).all(), "共同成熟池仍含未知，不允许填零或按模型删行。")
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=SEED)
        tree.fit(x, pool.event_class, sample_weight=pool.fit_weight.to_numpy(float))
        record["models"][policy] = parent.serialize(tree, columns, payoffs)
    record["status"] = "FIT_COMPLETE"
    return record


def forecast(data, outcomes):
    cuts = set(schedule(data))
    records, rows, current = [], [], None
    quality_columns = ["predicted_win_probability", "predicted_loss_probability", "predicted_payoff",
                       "predicted_p_times_b", "predicted_net_expectation"]
    for index in range(len(data)):
        if index in cuts:
            current = fit_three(data, outcomes, index)
            records.append(current)
        for policy in POLICIES:
            row = {"date": data.date.iloc[index], "origin_index": index, "policy": policy,
                   "fit_index": current["fit_index"] if current else np.nan,
                   "status": "NO_VIEW_NO_MODEL", "entry_event": False, "event_id": None,
                   "atr": float(data.atr14.iloc[index]), "stop_index": np.nan, "target_index": np.nan,
                   "score": np.nan, "leaf": np.nan, "leaf_train_rows": np.nan,
                   "technical_features_on_path": "", "macro_features_on_path": "", "fx_features_on_path": "",
                   **{key: np.nan for key in quality_columns + ["p_LOSS", "p_PROFIT", "p_TIMEOUT"]}}
            if current and current["status"] == "FIT_COMPLETE":
                row["status"] = "NO_VIEW_CURRENT_INFORMATION"
                if bool(data.joint_features_known.iloc[index]):
                    model = current["models"][policy]
                    probabilities, leaf, used = parent.predict(model, data.iloc[index][FEATURES[policy]].to_numpy(float))
                    row.update(status="AVAILABLE", **parent.original.quality(model, probabilities),
                               **dict(zip(["p_" + name for name in parent.original.CLASSES], probabilities)))
                    row.update(score=100. * row["predicted_win_probability"], leaf=leaf,
                               leaf_train_rows=model["node_rows"][leaf],
                               technical_features_on_path="|".join(dict.fromkeys(name for name in used if name in TECH)),
                               macro_features_on_path="|".join(dict.fromkeys(name for name in used if name in MACRO)),
                               fx_features_on_path="|".join(dict.fromkeys(name for name in used if name in FX)))
                    row["entry_event"] = bool(row["predicted_p_times_b"] > 1. and row["predicted_net_expectation"] > 0.)
                    row["event_id"] = f"{policy}_{index}" if row["entry_event"] else None
            rows.append(row)
    return records, pd.DataFrame(rows)
