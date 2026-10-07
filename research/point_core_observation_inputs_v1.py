"""完整收盘核心信号计算链：保留既有规则，所有父状态在同一已知日历上生成。"""
from __future__ import annotations

from bisect import bisect_right
import numpy as np
import pandas as pd

from research import reference_observation_accounts_v1 as observer
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.learned_cycle_exit_v1 import ExitController
from research.simple_intraday_protection_v1 import make_rules as learned_rules
from research.simple_volume_reversal_v1 import make_rules as panic_rules
from research.downside_reference_risk_inputs_v1 import moment_risk_frame
from research.joint_downside_reference_pair_inputs_v1 import minimum_joint_downside
from research.model_support_reference_router_inputs_v1 import validate_support_records
from research.trend_noise_reference_blend_inputs_v1 import market_amplitudes
from research.point_state_reconstruction_v1 import reconstruct_market, compose, MODELS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def intent_array(data, decisions, start, next_date):
    dates = pd.DatetimeIndex(data.date)
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    expected_origins = dates[first-1:]
    execution = dates[first:].append(pd.DatetimeIndex([next_date]))
    require(pd.DatetimeIndex(decisions.origin).equals(expected_origins), "观察参考缺少连续收盘意向。")
    require(pd.DatetimeIndex(decisions.execution_date).equals(execution), "观察参考计划执行时钟不同。")
    require(np.array_equal(decisions.origin_index, np.arange(first-1, len(data))), "观察参考的原始日历索引不同。")
    values = decisions.reference_weight.to_numpy(float)
    require((np.isnan(values) | np.isin(values, [0., 1.])).all(), "底层单次进入参考意向不是零一状态。")
    result = np.full(len(data), np.nan)
    result[first-1:] = values
    return result


def return_array(data, ledger, start):
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])), "参考收益日历不完整。")
    require(ledger.mark_clock.eq("CLOSE").all(), "参考收益包含人工终点开盘收益。")
    values = ledger.net_return.to_numpy(float)
    require((np.isfinite(values) & (values > -1)).all(), "参考收盘收益存在无效值。")
    result = np.full(len(data), np.nan)
    result[first:] = values
    return result


def observe_variance_budget(dates, returns, states, first, window=242):
    """原两策略最小方差预算，唯一边界差异是最后真实收盘也参与计算。"""
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(returns, float), np.asarray(states, float)
    require(dates.is_unique and dates.is_monotonic_increasing, "方差预算需要递增完整日期。")
    require(returns.shape == states.shape == (len(dates), 2), "方差预算的收益与状态形状不同。")
    require(1 <= first < len(dates) and window >= 2, "方差预算起点或窗口无效。")
    require((np.isnan(states) | np.isin(states, [0., 1.])).all(), "方差预算底层状态非法。")
    weights, sd = np.array([.5, .5]), np.full(2, np.nan)
    covariance = difference_variance = raw = np.nan
    attempted = successful = window_start = pd.NaT
    count, status, rows = 0, "INITIAL_EQUAL_BUDGET_NO_RISK_ESTIMATE", []
    for t, date in enumerate(dates):
        outside, scheduled = t < first-1, False
        if not outside and t >= first and date.to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, date
            count, window_start = min(window, t-first+1), dates[max(first, t-window+1)]
            values = returns[t-count+1:t+1]
            sd = np.full(2, np.nan)
            covariance = difference_variance = raw = np.nan
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                sd = np.std(values, axis=0, ddof=1)
                if not np.isfinite(sd).all() or (sd <= 0).any():
                    status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    centered = values-values.mean(axis=0)
                    covariance = float((centered[:, 0]*centered[:, 1]).sum()/(window-1))
                    difference_variance = float(np.var(values[:, 0]-values[:, 1], ddof=1))
                    if not np.isfinite(difference_variance) or difference_variance <= 0:
                        status = "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                    else:
                        raw = float((sd[1]**2-covariance)/difference_variance)
                        first_weight = float(np.clip(raw, 0, 1))
                        weights = np.array([first_weight, 1-first_weight])
                        successful, status = date, "MIN_VARIANCE_BUDGET_AVAILABLE"
        known = np.isfinite(states[t]).all() and not outside
        rows.append({"date": date, "panic_budget": np.nan if outside else weights[0], "learned_budget": np.nan if outside else weights[1],
                     "panic_sd": np.nan if outside else sd[0], "learned_sd": np.nan if outside else sd[1],
                     "reference_covariance": np.nan if outside else covariance, "difference_variance": np.nan if outside else difference_variance,
                     "raw_panic_budget": np.nan if outside else raw, "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status,
                     "risk_update_scheduled": scheduled, "risk_attempt_origin": pd.NaT if outside else attempted,
                     "last_successful_risk_origin": pd.NaT if outside else successful, "risk_window_start": pd.NaT if outside else window_start,
                     "risk_window_observations": 0 if outside else count, "panic_state": states[t, 0], "learned_state": states[t, 1],
                     "target": float(states[t] @ weights) if known else np.nan})
    return pd.DataFrame(rows)


def observe_joint_budget(dates, returns, first, window=242):
    """原共同下行预算，全部输入均为真实收盘参考收益。"""
    dates, returns = pd.DatetimeIndex(dates), np.asarray(returns, float)
    require(returns.shape == (len(dates), 2) and dates.is_unique and dates.is_monotonic_increasing, "共同下行输入日历或形状不同。")
    require(1 <= first < len(dates) and window >= 2, "共同下行起点或窗口无效。")
    weight, risk, prior_risk, derivative = .5, np.nan, np.nan, np.nan
    attempted = successful = window_start = pd.NaT
    count, status, rows = 0, "INITIAL_EQUAL_BUDGET_NO_DOWNSIDE_ESTIMATE", []
    for t, date in enumerate(dates):
        outside, scheduled = t < first-1, False
        if not outside and t >= first and date.to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, date
            count = min(window, t-first+1)
            window_start = dates[t-count+1]
            values = returns[t-count+1:t+1]
            risk = prior_risk = derivative = np.nan
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                choice = minimum_joint_downside(values, weight)
                weight, risk = choice["downside_budget"], choice["joint_downside_second_moment"]
                prior_risk, derivative = choice["previous_budget_downside_second_moment"], choice["downside_gradient"]
                status, successful = choice["optimizer_status"], date
        rows.append({"date": date, "downside_budget": np.nan if outside else weight, "continuous_budget": np.nan if outside else 1-weight,
                     "joint_downside_second_moment": np.nan if outside else risk, "previous_budget_downside_second_moment": np.nan if outside else prior_risk,
                     "downside_gradient": np.nan if outside else derivative, "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status,
                     "risk_update_scheduled": scheduled, "risk_attempt_origin": pd.NaT if outside else attempted,
                     "last_successful_risk_origin": pd.NaT if outside else successful, "risk_window_start": pd.NaT if outside else window_start,
                     "risk_window_observations": 0 if outside else count,
                     "downside_reference_return": returns[t, 0], "continuous_reference_return": returns[t, 1]})
    return pd.DataFrame(rows)


def weighted_targets(first_values, second_values, first_weights):
    values = np.column_stack([np.asarray(first_values, float), np.asarray(second_values, float)])
    weights = np.column_stack([np.asarray(first_weights, float), 1-np.asarray(first_weights, float)])
    require(values.shape == weights.shape, "组合目标与权重形状不同。")
    # 即使未知来源的预算为零，也保持原规则的完整资料要求。
    return np.sum(weights*values, axis=1)


def support_selection(data, models, first):
    times = validate_support_records(data, models)
    rows = []
    for t, date in enumerate(data.date):
        outside = t < first-1
        clock = date.normalize()+pd.Timedelta(hours=15, minutes=5)
        k = bisect_right(times, clock)-1 if not outside else -1
        record = models[k] if k >= 0 else None
        mature = record is not None and record["status"] == "FIT_COMPLETE"
        rows.append({"date": date, "decision_time": clock if not outside else pd.NaT,
                     "selected_parent": None if outside else "JOINT_DOWNSIDE_REFERENCE_PAIR" if mature else "VINTAGE_REFERENCE_RISK",
                     "model_support_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else record["status"] if record else "NO_RECORD_AVAILABLE",
                     "support_fit_origin": pd.Timestamp(record["fit_origin"]) if record else pd.NaT,
                     "support_fit_index": record["fit_index"] if record else np.nan})
    return pd.DataFrame(rows)


def continuous_references(data, panic_data, dividends, cfg, models31, next_date):
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(panic_data.date)), "连续参考价格日历不同。")
    controller = ExitController(data, models31, cfg["confirmation_days"])
    ridge = observer.observe_rearmed_exit(data, dividends, cfg, cfg["costs"]["BASE"], cfg["reference_start"],
                                         learned_rules(data)["D60_INTRA"], cfg["learned_spec"], controller, next_execution_date=next_date)
    panic = observer.observe_price_policy(panic_data, dividends, cfg, cfg["costs"]["BASE"], cfg["reference_start"],
                                         panic_rules(panic_data)[0]["V6_PANIC_RECOVERY"], cfg["panic_spec"], next_execution_date=next_date)
    return {"PANIC_ONLY": panic, "REARM_RIDGE": ridge}


def assemble_chain(data, dividends, configs, models114, start, next_date, continuous):
    """不读取已存核心目标；从参考持仓、原风险规则和模型时钟重建全部来源。"""
    cfg128, cfg91, cfg132 = (configs[k] for k in ("entry_vintage_exit", "continuous_reference_min_variance", "downside_reference_risk"))
    cfg137, cfg142 = configs["joint_downside_reference_pair"], configs["trend_noise_reference_blend"]
    require(cfg91["risk_window"] == cfg137["risk_window"] == 242, "不得改变连续或共同风险窗口。")
    require(cfg137["downside_benchmark"] == 0., "共同下行基准改变。")
    require(cfg128["costs"] == cfg91["costs"] == cfg132["costs"] == cfg137["costs"] == cfg142["costs"], "来源费用口径不同。")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    reference_first = int(np.flatnonzero(data.date.ge(pd.Timestamp(cfg91["reference_start"])))[0])
    base_returns, states = [], []
    for name in ("PANIC_ONLY", "REARM_RIDGE"):
        ledger, decisions = continuous[name][:2]
        base_returns.append(return_array(data, ledger, cfg91["reference_start"]))
        states.append(intent_array(data, decisions, cfg91["reference_start"], next_date))
    variance = observe_variance_budget(data.date, np.column_stack(base_returns), np.column_stack(states), reference_first)
    risk = moment_risk_frame(data, cfg132)
    known_vol = np.isfinite(data.vol20) & data.vol20.gt(0)
    multiplier = pd.Series(np.where(known_vol, np.minimum(1., .1/data.vol20), np.nan)).ffill().fillna(1.).to_numpy()
    down_multiplier = risk.DOWNSIDE_REFERENCE_RISK_multiplier.to_numpy()
    references, factors = {}, {}
    rule = learned_rules(data)["D60_INTRA"]
    for cost_id in cfg128["costs"]:
        controller = EntryVintageExitController(data, models114, cfg128["confirmation_days"])
        ref = observer.observe_rearmed_exit(data, dividends, cfg128, cfg128["costs"][cost_id], start, rule, cfg128["specification"], controller,
                                            next_execution_date=next_date)
        references["ENTRY_VINTAGE_" + cost_id] = ref
        intent = intent_array(data, ref[1], start, next_date)
        factors[cost_id] = pd.DataFrame({"date": data.date, "origin_index": np.arange(len(data)), "ENTRY_VINTAGE_EXIT": intent,
                                        "VINTAGE_REFERENCE_RISK": intent*multiplier, "DOWNSIDE_REFERENCE_RISK": intent*down_multiplier,
                                        "CONTINUOUS_REFERENCE_MIN_VARIANCE": variance.target.to_numpy(),
                                        "ordinary_risk_multiplier": multiplier, "downside_risk_multiplier": down_multiplier})
    joint_returns = []
    for model, cfg in (("DOWNSIDE_REFERENCE_RISK", cfg132), ("CONTINUOUS_REFERENCE_MIN_VARIANCE", cfg91)):
        account = observer.observe_event_account(data, dividends, cfg, cfg["costs"]["BASE"], start, model,
                                                 targets=factors["BASE"][model].to_numpy(), event_mask=np.ones(len(data), bool), next_execution_date=next_date)
        references[model + "_BASE"] = account
        joint_returns.append(return_array(data, account[0], start))
    joint = observe_joint_budget(data.date, np.column_stack(joint_returns), first)
    support = support_selection(data, models114, first)
    positive, noise = market_amplitudes(data, cfg142)
    known = np.isfinite(positive) & np.isfinite(noise)
    budget = np.full(len(data), np.nan)
    denom = positive+noise
    budget[known] = np.divide(positive[known], denom[known], out=np.zeros(known.sum()), where=denom[known] > 0)
    market = reconstruct_market(data, dividends)
    for cost_id, frame in factors.items():
        frame["JOINT_DOWNSIDE_REFERENCE_PAIR"] = weighted_targets(frame.DOWNSIDE_REFERENCE_RISK, frame.CONTINUOUS_REFERENCE_MIN_VARIANCE, joint.downside_budget)
        selected = support.selected_parent
        routed = np.full(len(data), np.nan)
        for model in ("VINTAGE_REFERENCE_RISK", "JOINT_DOWNSIDE_REFERENCE_PAIR"):
            mask = selected.eq(model)
            routed[mask] = frame.loc[mask, model]
        frame["MODEL_SUPPORT_REFERENCE_ROUTER"] = routed
        available = known & np.isfinite(frame.VINTAGE_REFERENCE_RISK) & np.isfinite(routed)
        core = np.full(len(data), np.nan)
        core[available] = np.clip(budget[available]*frame.loc[available, "VINTAGE_REFERENCE_RISK"]+(1-budget[available])*routed[available], 0, 1)
        frame["TREND_NOISE_REFERENCE_BLEND"] = core
        frame["trend_noise_budget"] = budget
        frame["positive_trend120"], frame["noise120"] = positive, noise
        frame["downside_budget"], frame["panic_budget"] = joint.downside_budget, variance.panic_budget
        frame["selected_parent"], frame["support_fit_origin"] = selected, support.support_fit_origin
        frame["auxiliary_raw"] = market.aux_target
        frame["runs_direction"], frame["lag_direction"] = market.runs_direction, market.lag_direction
        frame["drawdown60"] = market.drawdown60
        for model in MODELS:
            gate = np.where(market.drawdown60.notna(), market.drawdown60.gt(-.05).astype(float), np.nan) if model.startswith("CORE_") else market.lag_direction.to_numpy()
            target, effective = compose(core, market.aux_target, gate)
            frame[model] = target
            frame[model + "_effective_auxiliary"] = effective
        frame["cost"] = cost_id
    return {"factors": factors, "references": references, "variance_budget": variance, "joint_budget": joint,
            "support": support, "market": market, "risk": risk}
