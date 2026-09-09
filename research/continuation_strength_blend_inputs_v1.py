"""复用原参考当前收盘预测分配预算，保留无模型与未知状态。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.entry_vintage_exit_inputs_v1 import prediction_identity
from research.intraday_overnight_increment_v1 import require

PRIMARY = "CONTINUATION_STRENGTH_BLEND"
MODELS = ["VINTAGE_REFERENCE_RISK", "TREND_NOISE_REFERENCE_BLEND"]


def current_close_signals(decisions, ledger, data, start, cost):
    """只连接判断当日的实际收盘份额，唯一准备收盘来自初始现金。"""
    first = int(np.flatnonzero(data.date.ge(start))[0])
    indices = np.arange(first-1, len(data)-1)
    dates = pd.DatetimeIndex(data.date)
    require(np.array_equal(decisions.origin_index, indices), "保存预测索引不完整")
    require(pd.DatetimeIndex(decisions.origin).equals(dates[indices]) and
        pd.DatetimeIndex(decisions.execution_date).equals(dates[indices+1]), "保存预测不是当前收盘到下一开盘")
    require(pd.DatetimeIndex(ledger.date).equals(dates[first:]), "原参考持仓账本日历不完整")
    require(ledger.mark_clock.iloc[:-1].eq("CLOSE").all() and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL", "原参考持仓不是当日收盘标记")
    shares = decisions.origin.map(ledger.set_index("date").shares)
    require(shares.isna().sum() == 1 and pd.isna(shares.iloc[0]), "准备收盘以外缺少原参考实际份额")
    shares.iloc[0] = 0.
    return decisions.copy().assign(reference_current_shares=shares.to_numpy(float), source_cost=cost)


def continuation_strength_frames(data, parents_by_cost, signals_by_cost, records, cfg, start):
    require(cfg["combination"] == "POSITIVE_CONTINUATION_OVER_CONTINUATION_PLUS_REMAINING_NOISE" and
        cfg["maximum_holding_days"] == 60 and cfg["noise_window"] == 20 and cfg["annual_days"] == 242,
        "继续持有预算或期限定义改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    require(set(signals_by_cost) == set(cfg["costs"]), "保存预测费用集合不同")
    expected_vol = data.total_simple.rolling(20, min_periods=20).std(ddof=1)*np.sqrt(242)
    np.testing.assert_allclose(data.vol20, expected_vol, atol=1e-12, rtol=1e-12, equal_nan=True)
    volatility = data.vol20.to_numpy(float)
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0))).all(), "原波动出现负值或无穷值")
    dates, indices = pd.DatetimeIndex(data.date), np.arange(first-1, len(data)-1)
    by_day = {pd.Timestamp(record["fit_origin"]): record for record in records}
    require(len(by_day) == len(records), "保存模型日期重复")
    identities = {date: prediction_identity(record) for date, record in by_day.items()}
    summaries = []
    for cost, frame in frames.items():
        signals = signals_by_cost[cost]
        require(signals.source_cost.eq(cost).all(), "继续预测混用费用")
        require(np.array_equal(signals.origin_index, indices) and pd.DatetimeIndex(signals.origin).equals(dates[indices]) and
            pd.DatetimeIndex(signals.execution_date).equals(dates[indices+1]), "继续预测时钟或日历改变")
        current = signals.reference_current_shares.to_numpy(float)
        require((np.isfinite(current) & (current >= 0) & (current == np.floor(current))).all(), "原参考实际份额未知或非法")
        budget, age, horizon, noise, advantage = [np.full(len(data), np.nan) for _ in range(5)]
        state = np.full(len(data), "OUTSIDE_DECISION_CALENDAR", dtype=object)
        for row in signals.itertuples():
            t = int(row.origin_index)
            if row.reference_current_shares == 0:
                require(pd.isna(row.learning_status) and pd.isna(row.continuation_prediction), "原参考空仓时混入持仓预测")
                budget[t], state[t] = 0., "REFERENCE_FLAT_USE143_WITHOUT_LEARNING"
                continue
            if pd.isna(row.learning_status) or row.learning_status not in {"PREDICTION_AVAILABLE", "NO_VIEW_NO_MATURE_MODEL"}:
                state[t] = "NO_VIEW_UNRECOGNIZED_OR_FAILED_SOURCE_STATE"
                continue
            metadata = [row.model_selection_index, row.model_selection_origin, row.log_holding_days, row.learning_fit_origin, row.fixed_prediction_identity]
            if any(pd.isna(value) for value in metadata):
                state[t] = "NO_VIEW_INCOMPLETE_MODEL_METADATA"
                continue
            selected = float(row.model_selection_index)
            require(np.isfinite(selected) and selected == int(selected) and 0 <= selected <= t, "原模型选择索引非法或在未来")
            selected = int(selected)
            require(pd.Timestamp(row.model_selection_origin) == dates[selected], "原模型选择日期与索引不一致")
            record = by_day.get(pd.Timestamp(row.learning_fit_origin))
            if record is None:
                state[t] = "NO_VIEW_MISSING_MODEL_RECORD"
                continue
            require(pd.Timestamp(record["fit_time"]) <= dates[selected]+pd.Timedelta(hours=15, minutes=5) <= dates[t]+pd.Timedelta(hours=15, minutes=5),
                "原模型拟合或选择晚于当时判断")
            require(record["latest_exit_index"] <= record["fit_index"] <= selected and dates[record["fit_index"]] == pd.Timestamp(record["fit_origin"]),
                "原模型使用未来版本或未结束训练周期")
            require(row.fixed_prediction_identity == identities[pd.Timestamp(row.learning_fit_origin)], "原固定模型身份不同")
            age[t] = t-selected+1
            require(np.isfinite(row.log_holding_days) and np.isclose(row.log_holding_days, np.log1p(age[t]), atol=1e-12, rtol=1e-12), "原持仓天数不对应首次实际进入收盘")
            if row.learning_status == "NO_VIEW_NO_MATURE_MODEL":
                require(record["status"] == "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS" and record["model"] is None and
                    pd.isna(row.continuation_prediction), "原明确无成熟模型状态不一致")
                budget[t], state[t] = 0., "NO_MATURE_MODEL_USE143_WITHOUT_LEARNING"
                continue
            require(record["status"] == "FIT_COMPLETE" and record["model"] is not None, "原可用预测没有成熟固定模型")
            prediction = float(row.continuation_prediction)
            if np.isnan(prediction) or np.isnan(volatility[t]) or pd.isna(row.vol20):
                state[t] = "NO_VIEW_INCOMPLETE_PREDICTION_OR_VOLATILITY"
                continue
            require(np.isfinite(prediction) and np.isfinite(row.vol20) and row.vol20 >= 0, "原预测或保存波动非法")
            require(np.isclose(row.vol20, volatility[t], atol=1e-12, rtol=1e-12), "原预测波动与判断当日市场不同")
            horizon[t] = max(cfg["maximum_holding_days"]-age[t], 1.)
            noise[t] = volatility[t]*np.sqrt(horizon[t]/cfg["annual_days"])
            advantage[t] = max(prediction, 0.)
            total = advantage[t]+noise[t]
            budget[t] = advantage[t]/total if total > 0 else 0.
            state[t] = "PREDICTION_AMPLITUDE_BUDGET_AVAILABLE"
        frame["allocation_status"] = state
        frame["reference_holding_days"] = age
        frame["remaining_comparison_days"] = horizon
        frame["remaining_noise_amplitude"] = noise
        frame["positive_continuation_advantage"] = advantage
        frame["reference_budget131"] = budget
        frame["reference_budget143"] = 1-budget
        for column in ["reference_current_shares", "learning_status", "continuation_prediction", "learning_fit_origin",
            "model_selection_index", "model_selection_origin", "fixed_prediction_identity", "log_holding_days", "vol20"]:
            frame[column] = signals.set_index("origin_index")[column].reindex(np.arange(len(data))).to_numpy()
        x, y = (frame[model+"_parent_target"].to_numpy(float) for model in MODELS)
        known = np.isfinite(budget) & np.isfinite(x) & np.isfinite(y)
        target = np.full(len(data), np.nan)
        target[known] = np.clip(budget[known]*x[known]+(1-budget[known])*y[known], 0, 1)
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(indices), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "allocation_states": eligible.allocation_status.value_counts().to_dict(),
            "mean_131_budget": float(eligible.reference_budget131.mean()), "maximum_131_budget": float(eligible.reference_budget131.max()),
            "positive_131_budget_origins": int(eligible.reference_budget131.gt(0).sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
