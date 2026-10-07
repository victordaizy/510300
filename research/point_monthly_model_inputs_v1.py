"""退出模型的自然成熟样本与原月度拟合制度，不从未完成周期生成训练标签。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.learned_cycle_exit_v1 import FEATURES, continuation_label, state_values, training_rows, fit_one
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.simple_intraday_protection_v1 import make_rules
from research.training_reference_observation_v1 import observe_training_reference

SAMPLE_COLUMNS = ["signal", "cycle_id", "origin_index", "origin", "early_exit_index", "early_exit_date", "exit_index",
                  "mature_date", "target", "extra_dividend_cny", "reference_quantity", *FEATURES]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mature_samples(data, dividends, cfg, decisions, cycles):
    samples = []
    require(not cycles.exit_reasons.str.contains("研究终点", regex=False).any(), "训练观察混入了人工终点退出。")
    for cycle in cycles.to_dict("records"):
        if pd.isna(cycle.get("exit_date")):
            continue
        end = int(np.flatnonzero(data.date.eq(pd.Timestamp(cycle["exit_date"])))[0])
        states = decisions.loc[decisions.learning_cycle_id.eq(cycle["cycle_id"]) & decisions.requested_quantity.eq(0)]
        for row in states.to_dict("records"):
            t = int(row["origin_index"])
            if t+1 >= end or not np.isfinite([row[k] for k in FEATURES]).all():
                continue
            target, extra = continuation_label(data, dividends, cycle["entry_quantity"], t+1, end, cfg["costs"]["BASE"], cfg["tick"])
            samples.append({"signal": "D60_INTRA", "cycle_id": int(cycle["cycle_id"]), "origin_index": t, "origin": data.date.iloc[t],
                            "early_exit_index": t+1, "early_exit_date": data.date.iloc[t+1], "exit_index": end,
                            "mature_date": data.date.iloc[end], "target": target, "extra_dividend_cny": extra,
                            "reference_quantity": int(cycle["entry_quantity"]), **{k: row[k] for k in FEATURES}})
    result = pd.DataFrame(samples, columns=SAMPLE_COLUMNS)
    if len(result):
        require(result.origin_index.lt(result.early_exit_index).all() and result.early_exit_index.lt(result.exit_index).all(), "样本时钟不是先状态后标签。")
        require(not result.duplicated(["cycle_id", "origin_index"]).any(), "训练状态重复。")
    return result


def reference_samples(data, dividends, cfg, next_date):
    def recorder(t, cycle, current_value, peak_value):
        return {"learning_cycle_id": cycle["cycle_id"], "learned_exit_requested": False,
                **dict(zip(FEATURES, state_values(data, t, cycle, current_value, peak_value)))}
    ledger, decisions, cycles = observe_training_reference(data, dividends, cfg, cfg["costs"]["BASE"], cfg["reference_start"],
        make_rules(data)["D60_INTRA"], cfg["candidate_specs"]["D60_INTRA"], recorder, next_execution_date=next_date)
    samples = mature_samples(data, dividends, cfg, decisions, cycles)
    open_ids = cycles.loc[cycles.exit_date.isna(), "cycle_id"].to_list()
    unfinished = decisions.loc[decisions.learning_cycle_id.isin(open_ids) & decisions.requested_quantity.eq(0),
                               ["origin", "origin_index", "learning_cycle_id", *FEATURES]].copy()
    unfinished["status"] = "UNMATURED_NO_TRAINING_LABEL"
    return {"ledger": ledger, "decisions": decisions, "cycles": cycles, "samples": samples, "unfinished_states": unfinished}


def monthly_schedule(data, earlier_start):
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(earlier_start)))[0])-1
    require(first >= 0, "月度模型缺少初始收盘。")
    new_month = data.date.dt.to_period("M").ne(data.date.shift().dt.to_period("M"))
    # 最后真实日若恰为月首，其已完成收盘也有权按原制度拟合。
    return sorted({first, *(int(t) for t in np.flatnonzero(new_month) if first <= t < len(data))})


def fit_monthly_pair(data, samples, ordinary_cfg, within_cfg):
    for key in ("recent_cycles", "minimum_cycles", "minimum_rows", "feature_columns", "feature_clip", "ridge_alpha"):
        require(ordinary_cfg[key] == within_cfg[key], "两类原模型的共同训练制度不同：" + key)
    require(ordinary_cfg["recent_cycles"] == 20 and ordinary_cfg["minimum_cycles"] == 10 and ordinary_cfg["minimum_rows"] == 100,
            "原成熟样本门槛改变。")
    ordinary, within, membership = [], [], []
    for t in monthly_schedule(data, ordinary_cfg["earlier_start"]):
        rows, ids = training_rows(samples, t, ordinary_cfg)
        eligible = len(ids) >= ordinary_cfg["minimum_cycles"] and len(rows) >= ordinary_cfg["minimum_rows"]
        require(not len(rows) or rows.exit_index.le(t).all(), "月度拟合读取未成熟周期。")
        basic = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5),
                 "status": "FIT_COMPLETE" if eligible else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS",
                 "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows),
                 "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                 "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None}
        ordinary.append({"signal": "D60_INTRA", "kind": "RIDGE", **basic, "model": fit_one(rows, "RIDGE", ordinary_cfg) if eligible else None})
        missing = int((~np.isfinite(rows[FEATURES].to_numpy(float)).all(axis=1)).sum())
        status, failure, model = basic["status"], None, None
        if eligible and missing:
            status = "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"
        elif eligible:
            try:
                model = fit_within_cycle_exit(rows, within_cfg)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                status, failure = "NO_VIEW_MODEL_FIT_FAILED", str(error)
        within.append({**basic, "status": status, "eligible_for_fit": eligible, "failure": failure, "missing_feature_rows": missing, "model": model})
        if eligible:
            membership.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                               "sample_weight": float(r.sample_weight)} for r in rows.itertuples())
    return {"ordinary": ordinary, "within": within, "membership": pd.DataFrame(membership)}
