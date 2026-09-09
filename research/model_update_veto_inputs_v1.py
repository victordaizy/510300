"""用同一原参考状态的最新模型负面分歧否决当前父正目标段。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.entry_vintage_exit_inputs_v1 import prediction_identity
from research.within_cycle_exit_inputs_v1 import FEATURES, within_cycle_prediction
from research.intraday_overnight_increment_v1 import require

PRIMARY = "MODEL_UPDATE_VETO"
MODELS = ["TREND_NOISE_REFERENCE_BLEND"]


def model_update_veto_frames(data, parents_by_cost, signals_by_cost, records, cfg, start):
    require(cfg["combination"] == "FIXED_NONNEGATIVE_LATEST_NEGATIVE_VETO_UNTIL_PARENT_ZERO", "模型更新分歧或解除规则改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    dates, indices = pd.DatetimeIndex(data.date), np.arange(first-1, len(data)-1)
    require(set(signals_by_cost) == set(cfg["costs"]), "固定预测来源费用集合不同")
    fit_indices = [record["fit_index"] for record in records]
    require(fit_indices == sorted(set(fit_indices)), "保存模型拟合索引不唯一递增")
    by_date = {pd.Timestamp(record["fit_origin"]): record for record in records}
    require(len(by_date) == len(records), "保存模型日期重复")
    identities = {record["fit_index"]: prediction_identity(record) for record in records}
    summaries = []
    for cost, frame in frames.items():
        signals = signals_by_cost[cost]
        require(signals.source_cost.eq(cost).all() and np.array_equal(signals.origin_index, indices) and
            pd.DatetimeIndex(signals.origin).equals(dates[indices]) and pd.DatetimeIndex(signals.execution_date).equals(dates[indices+1]), "固定预测费用或当日时钟不同")
        shares = signals.reference_current_shares.to_numpy(float)
        require((np.isfinite(shares) & (shares >= 0) & (shares == np.floor(shares))).all(), "原参考实际份额未知或非法")
        target, latest_predictions, disagreement = [np.full(len(data), np.nan) for _ in range(3)]
        states = np.full(len(data), "OUTSIDE_DECISION_CALENDAR", dtype=object)
        latest_dates = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        latest_identities = np.full(len(data), None, dtype=object)
        locked, started, reset = [np.zeros(len(data), bool) for _ in range(3)]
        parent = frame[MODELS[0]+"_parent_target"].to_numpy(float)
        veto, comparisons, changed_identities = False, 0, 0
        for row in signals.itertuples():
            t = int(row.origin_index)
            if np.isnan(parent[t]):
                states[t] = "PARENT_UNKNOWN_KEEP_VETO"
            elif parent[t] == 0:
                reset[t], veto = veto, False
                target[t], states[t] = 0., "PARENT_ZERO_RESET_VETO"
            elif veto:
                target[t], states[t] = 0., "VETO_LOCKED_UNTIL_PARENT_ZERO"
            elif row.reference_current_shares == 0:
                require(pd.isna(row.learning_status) and pd.isna(row.continuation_prediction), "原参考空仓时混入固定持仓预测")
                target[t], states[t] = parent[t], "REFERENCE_FLAT_USE_PARENT_WITHOUT_COMPARISON"
            elif pd.isna(row.learning_status) or row.learning_status not in {"PREDICTION_AVAILABLE", "NO_VIEW_NO_MATURE_MODEL"}:
                states[t] = "NO_VIEW_FIXED_SOURCE_FAILED_OR_UNKNOWN"
            else:
                metadata = [row.learning_fit_origin, row.model_selection_index, row.model_selection_origin, row.fixed_prediction_identity, row.log_holding_days]
                if any(pd.isna(value) for value in metadata):
                    states[t] = "NO_VIEW_INCOMPLETE_FIXED_MODEL_METADATA"
                else:
                    selected = float(row.model_selection_index)
                    require(np.isfinite(selected) and selected == int(selected) and 0 <= selected <= t, "原固定模型选择索引非法或在未来")
                    selected = int(selected)
                    require(pd.Timestamp(row.model_selection_origin) == dates[selected], "原选择日期不对应行情索引")
                    fixed = by_date.get(pd.Timestamp(row.learning_fit_origin))
                    if fixed is None:
                        states[t] = "NO_VIEW_MISSING_FIXED_MODEL"
                    else:
                        require(fixed["latest_exit_index"] <= fixed["fit_index"] <= selected and
                            pd.Timestamp(fixed["fit_time"]) <= dates[selected]+pd.Timedelta(hours=15, minutes=5), "固定模型使用未来版本或训练周期")
                        require(dates[fixed["fit_index"]] == pd.Timestamp(fixed["fit_origin"]) and
                            row.fixed_prediction_identity == identities[fixed["fit_index"]], "原固定模型日期或参数身份不同")
                        require(np.isfinite(row.log_holding_days) and np.isclose(row.log_holding_days, np.log1p(t-selected+1), atol=1e-12, rtol=1e-12), "原持仓天数不对应首次实际进入收盘")
                        if row.learning_status == "NO_VIEW_NO_MATURE_MODEL":
                            require(fixed["status"] == "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS" and fixed["model"] is None and pd.isna(row.continuation_prediction), "原明确无成熟模型状态不一致")
                            target[t], states[t] = parent[t], "FIXED_NO_MATURE_MODEL_USE_PARENT_WITHOUT_COMPARISON"
                        else:
                            require(fixed["status"] == "FIT_COMPLETE" and fixed["model"] is not None, "原可用预测不是成熟固定模型")
                            values = np.array([getattr(row, feature) for feature in FEATURES], float)
                            require((np.isnan(values) | np.isfinite(values)).all() and
                                (pd.isna(row.continuation_prediction) or np.isfinite(row.continuation_prediction)), "原八因素或固定预测包含无穷值")
                            position = bisect_right(fit_indices, t)-1
                            if position < 0 or not np.isfinite(values).all() or pd.isna(row.continuation_prediction):
                                states[t] = "NO_VIEW_INCOMPLETE_COMPARISON_INPUT"
                            else:
                                latest = records[position]
                                require(latest["latest_exit_index"] <= latest["fit_index"] <= t and fixed["fit_index"] <= latest["fit_index"], "最近模型读取未来训练或早于固定版")
                                require(pd.Timestamp(latest["fit_time"]) <= dates[t]+pd.Timedelta(hours=15, minutes=5) and
                                    dates[latest["fit_index"]] == pd.Timestamp(latest["fit_origin"]), "最近模型时钟或日期索引不符")
                                if latest["status"] != "FIT_COMPLETE" or latest["model"] is None:
                                    states[t] = "NO_VIEW_LATEST_MODEL_UNAVAILABLE"
                                else:
                                    model = latest["model"]
                                    require(all(len(model[key]) == 8 and np.isfinite(model[key]).all() for key in ["mean", "scale", "coefficients"]) and
                                        (np.asarray(model["scale"]) > 0).all() and np.isfinite(model["intercept"]) and
                                        np.isfinite(model["feature_clip"]) and model["feature_clip"] > 0, "最近保存模型参数非法")
                                    value = within_cycle_prediction(latest["model"], values)
                                    require(np.isfinite(value), "最近模型预测不是有限数")
                                    latest_predictions[t], latest_dates[t] = value, dates[latest["fit_index"]].to_datetime64()
                                    latest_identities[t] = identities[latest["fit_index"]]
                                    comparisons += 1
                                    changed_identities += int(latest_identities[t] != row.fixed_prediction_identity)
                                    if latest_identities[t] == row.fixed_prediction_identity:
                                        require(np.isclose(value, row.continuation_prediction, atol=1e-12, rtol=1e-12), "同参数同状态却不能复算固定预测")
                                    disagreement[t] = float(row.continuation_prediction >= 0 and value < 0)
                                    if disagreement[t] == 1:
                                        veto, started[t], target[t], states[t] = True, True, 0., "NEGATIVE_MODEL_UPDATE_STARTED_VETO"
                                    else:
                                        target[t], states[t] = parent[t], "COMPARABLE_PREDICTIONS_NO_NEGATIVE_UPDATE"
            locked[t] = veto
        frame["comparison_status"] = states
        frame["latest_continuation_prediction"] = latest_predictions
        frame["latest_model_fit_origin"] = latest_dates
        frame["latest_prediction_identity"] = latest_identities
        frame["negative_update_disagreement"] = disagreement
        frame["veto_active"] = locked
        frame["veto_started"] = started
        frame["veto_reset"] = reset
        for column in ["reference_current_shares", "learning_status", "continuation_prediction", "learning_fit_origin", "model_selection_index",
            "model_selection_origin", "fixed_prediction_identity", *FEATURES]:
            name = "fixed_continuation_prediction" if column == "continuation_prediction" else "source_"+column
            frame[name] = signals.set_index("origin_index")[column].reindex(np.arange(len(data))).to_numpy()
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(eligible), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "new_latest_predictions": comparisons, "comparisons_with_different_parameter_identity": changed_identities,
            "veto_episodes_started": int(started.sum()), "veto_resets_at_parent_zero": int(reset.sum()), "veto_active_origins": int(locked.sum()),
            "veto_still_active_at_last_origin": veto, "comparison_states": eligible.comparison_status.value_counts().to_dict(),
            "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
