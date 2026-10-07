"""宏观与日周线联合首次边界评分，保留公布时钟和未成熟标签。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from research import point_first_passage_inputs_v1 as original

TECH = list(original.FEATURES)
MACRO = ["pmi_orders_level", "pmi_orders_change", "funding_gap_pp", "funding_gap_change5",
         "financing_net_change5", "financing_buy_activity"]
POLICIES = ("TECH_COMMON_TREE", "MACRO_TECH_TREE")
FEATURES = {POLICIES[0]: TECH, POLICIES[1]: TECH + MACRO}
SEED = 510300191


def require(condition, message):
    if not condition:
        raise ValueError(message)


def time_shanghai(values):
    return pd.to_datetime(values, utc=True).dt.tz_convert("Asia/Shanghai").astype("datetime64[ns, Asia/Shanghai]")


def views(data, orders, funding, margin):
    d = data.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ns]")
    require(d.date.is_unique and d.date.is_monotonic_increasing, "观察日历不唯一或未排序。")
    d["decision_time"] = d.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    o = orders.copy().sort_values("reference_period").reset_index(drop=True)
    require(o.reference_period.is_unique, "PMI统计月份重复，不能事后择版本。")
    months = pd.PeriodIndex(o.reference_period, freq="M").asi8
    continuous = pd.Series(months).diff().eq(1)
    o["orders_available_at"] = time_shanghai(o.available_at)
    o["pmi_orders_level"] = o.first_release_value - 50.
    o["pmi_orders_change"] = o.first_release_value.diff().where(continuous)
    o = o.rename(columns={"reference_period": "orders_reference_period", "source_url": "orders_source_url"})
    columns = ["orders_available_at", "orders_reference_period", "orders_source_url", "pmi_orders_level", "pmi_orders_change"]
    require(o.orders_available_at.notna().all() and o.orders_available_at.is_unique, "PMI公布时点缺失或重复。")
    d = pd.merge_asof(d, o[columns].sort_values("orders_available_at"), left_on="decision_time",
                      right_on="orders_available_at", direction="backward")
    age = (d.decision_time - d.orders_available_at).dt.total_seconds() / 86400.
    d["orders_known"] = age.between(0., 60.) & d[["pmi_orders_level", "pmi_orders_change"]].notna().all(axis=1)
    d.loc[~d.orders_known, ["pmi_orders_level", "pmi_orders_change"]] = np.nan

    f = funding.copy()
    f["date"] = pd.to_datetime(f.date).astype("datetime64[ns]")
    f["fund_stat_date"] = pd.to_datetime(f.fund_stat_date).astype("datetime64[ns]")
    f["funding_available_at"] = time_shanghai(f.available_at)
    f["funding_policy_known_at"] = time_shanghai(f.policy_known_at)
    require(f.date.is_unique, "资金观察日期重复。")
    f = f.rename(columns={"fund_known": "original_fund_known", "gap_pp": "funding_gap_pp"})
    d = d.merge(f[["date", "fund_stat_date", "funding_available_at", "funding_policy_known_at",
                   "original_fund_known", "funding_gap_pp", "dr007", "rate"]], on="date", how="left", validate="one_to_one")
    d["funding_known"] = (d.original_fund_known.eq(True) & d.fund_stat_date.lt(d.date)
                          & d.funding_available_at.le(d.decision_time) & d.funding_policy_known_at.le(d.decision_time)
                          & np.isfinite(d[["funding_gap_pp", "dr007", "rate"]].to_numpy(float)).all(axis=1))
    require(np.allclose(d.loc[d.funding_known, "funding_gap_pp"],
                        d.loc[d.funding_known, "dr007"] - d.loc[d.funding_known, "rate"], atol=1e-10), "资金差的单位或定义改变。")
    d.loc[~d.funding_known, "funding_gap_pp"] = np.nan
    d["funding_gap_change5"] = d.funding_gap_pp.diff(5).where(d.funding_gap_pp.rolling(6, min_periods=6).count().eq(6))

    m = margin.copy()
    m["date"] = pd.to_datetime(m.date).astype("datetime64[ns]")
    m = m.sort_values("date").reset_index(drop=True)
    require(m.date.is_unique, "融资统计日重复。")
    require(m.market_rzye.gt(0).all() and m.market_rzmre.gt(0).all(), "融资金额不为正，须先核实来源。")
    m = d[["date"]].merge(m[["date", "market_rzye", "market_rzmre"]], on="date", how="left", validate="one_to_one")
    m["financing_net_change5"] = m.market_rzye.pct_change(5, fill_method=None).where(m.market_rzye.rolling(6, min_periods=6).count().eq(6))
    m["financing_buy_activity"] = np.log(m.market_rzmre.rolling(5, min_periods=5).mean()
                                         / m.market_rzmre.rolling(60, min_periods=60).mean())
    aligned = d[["date"]].merge(m[["date", "financing_net_change5", "financing_buy_activity"]], on="date", how="left", validate="one_to_one")
    d[["financing_net_change5", "financing_buy_activity"]] = aligned[["financing_net_change5", "financing_buy_activity"]].shift(1)
    d["margin_stat_date"] = d.date.shift(1).where(aligned.financing_buy_activity.shift(1).notna())
    d["margin_available_at"] = (d.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)).where(d.margin_stat_date.notna())
    d["margin_known"] = (d.margin_stat_date.lt(d.date) & d.margin_available_at.le(d.decision_time)
                         & np.isfinite(d[["financing_net_change5", "financing_buy_activity"]].to_numpy(float)).all(axis=1))
    d["macro_features_known"] = (d.orders_known & d.funding_known & d.margin_known
                                  & np.isfinite(d[MACRO].to_numpy(float)).all(axis=1))
    d["joint_features_known"] = d.first_passage_feature_known & d.macro_features_known
    return d


def common_pool(data, outcomes, fit_index):
    pool = original.training_pool(data, outcomes, fit_index)
    indices = pool.origin_index.to_numpy(int)
    pool = pool.loc[data.joint_features_known.iloc[indices].to_numpy(bool)].copy()
    if len(pool):
        concurrent = np.zeros(fit_index + 1)
        for r in pool.itertuples(index=False):
            concurrent[int(r.entry_idx):int(r.exit_idx)] += 1
        w = np.asarray([np.mean(1. / concurrent[int(r.entry_idx):int(r.exit_idx)]) for r in pool.itertuples(index=False)])
        require(np.isfinite(w).all() and (w > 0).all(), "共同训练标签唯一性权重无效。")
        pool["uniqueness_weight"] = w
        pool["fit_weight"] = w / w.mean()
        require(pool.mature_idx.le(fit_index).all(), "共同训练使用了未来才成熟的标签。")
    return pool


def class_payoffs(pool):
    values = []
    for name in original.CLASSES:
        part = pool.loc[pool.event_class.eq(name)]
        w, y = part.fit_weight.to_numpy(float), part.reference_net_return.to_numpy(float)
        values.append({"positive_probability": float(np.average(y > 0, weights=w)),
                       "negative_probability": float(np.average(y < 0, weights=w)),
                       "positive_contribution": float(np.average(np.maximum(y, 0), weights=w)),
                       "negative_contribution": float(np.average(np.maximum(-y, 0), weights=w))})
    return values


def serialize(tree, columns, payoffs):
    t = tree.tree_
    require(tuple(tree.classes_) == original.CLASSES, "三类事件顺序改变。")
    return {"features": columns, "classes": list(original.CLASSES), "class_payoffs": payoffs,
            "children_left": t.children_left.tolist(), "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
            "value": t.value[:, 0, :].tolist(), "node_rows": t.n_node_samples.tolist()}


def predict(model, vector):
    x = np.asarray(vector, dtype=np.float32)
    require(np.isfinite(x).all(), "评分向量含未知，不能填零。")
    node, used = 0, []
    while model["children_left"][node] != -1:
        feature = model["feature"][node]
        used.append(model["features"][feature])
        node = (model["children_left"][node] if float(x[feature]) <= model["threshold"][node]
                else model["children_right"][node])
    values = np.asarray(model["value"][node], float)
    return values / values.sum(), node, used


def fit_pair(data, outcomes, index):
    pool = common_pool(data, outcomes, index)
    counts = pool.event_class.value_counts()
    record = {"fit_index": int(index), "fit_date": data.date.iloc[index], "training_origins": pool.origin_index.astype(int).tolist(),
              "training_rows": len(pool), "latest_mature_idx": int(pool.mature_idx.max()) if len(pool) else None,
              "class_counts": {name: int(counts.get(name, 0)) for name in original.CLASSES},
              "sum_raw_uniqueness_weights": float(pool.uniqueness_weight.sum()),
              "status": "NO_VIEW_COMMON_TRAINING_SUPPORT", "models": {}}
    if len(pool) < original.MINIMUM_ROWS or any(counts.get(name, 0) < 10 for name in original.CLASSES):
        return record
    payoffs = class_payoffs(pool)
    for policy, columns in FEATURES.items():
        x = data.iloc[pool.origin_index.to_numpy(int)][columns].to_numpy(float)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=SEED)
        tree.fit(x, pool.event_class, sample_weight=pool.fit_weight.to_numpy(float))
        record["models"][policy] = serialize(tree, columns, payoffs)
    record["status"] = "FIT_COMPLETE"
    return record


def forecast(data, outcomes):
    month = data.date.dt.to_period("M")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp("2015-01-05")))[0]) - 1
    cuts = {first, *(int(i) for i in np.flatnonzero(month.ne(month.shift())) if i > first)}
    records, rows, current = [], [], None
    quality_columns = ["predicted_win_probability", "predicted_loss_probability", "predicted_payoff",
                       "predicted_p_times_b", "predicted_net_expectation"]
    for i in range(len(data)):
        if i in cuts:
            current = fit_pair(data, outcomes, i)
            records.append(current)
        for policy in POLICIES:
            row = {"date": data.date.iloc[i], "origin_index": i, "policy": policy,
                   "fit_index": current["fit_index"] if current else np.nan, "status": "NO_VIEW_NO_MODEL",
                   "entry_event": False, "event_id": None, "atr": float(data.atr14.iloc[i]),
                   "stop_index": np.nan, "target_index": np.nan, "score": np.nan,
                   "leaf": np.nan, "leaf_train_rows": np.nan, "macro_features_on_path": "",
                   **{k: np.nan for k in quality_columns + ["p_LOSS", "p_PROFIT", "p_TIMEOUT"]}}
            if current and current["status"] == "FIT_COMPLETE":
                row["status"] = "NO_VIEW_CURRENT_INFORMATION"
                if bool(data.joint_features_known.iloc[i]):
                    model = current["models"][policy]
                    p, leaf, used = predict(model, data.iloc[i][FEATURES[policy]].to_numpy(float))
                    row.update(status="AVAILABLE", **original.quality(model, p),
                               **dict(zip(["p_" + name for name in original.CLASSES], p)))
                    row.update(score=100. * row["predicted_win_probability"], leaf=leaf,
                               leaf_train_rows=model["node_rows"][leaf],
                               macro_features_on_path="|".join(dict.fromkeys(x for x in used if x in MACRO)))
                    row["entry_event"] = bool(row["predicted_p_times_b"] > 1. and row["predicted_net_expectation"] > 0.)
                    row["event_id"] = f"{policy}_{i}" if row["entry_event"] else None
            rows.append(row)
    return records, pd.DataFrame(rows)
