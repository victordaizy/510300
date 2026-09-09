"""只用当时价格估计平滑趋势，保存季度估计与每日滤波状态。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.intraday_overnight_increment_v1 import require

DEFAULT_SETTINGS = {"training_observations": 504, "entry_z": 1.0, "log_variance_bounds": [-30.0, 0.0],
    "initial_variance_floor": 1e-8, "initial_slope_noise_ratio": .001, "initial_observation_noise_ratio": .5,
    "optimizer_maxiter": 100, "optimizer_ftol": 1e-9, "optimizer_gtol": 1e-5, "optimizer_maxls": 20}


def predict_state(state, slope_noise):
    level, slope, p00, p01, p11 = map(float, state)
    return (level + slope, slope, p00 + 2 * p01 + p11, p01 + p11, p11 + slope_noise)


def filter_step(state, observation, slope_noise, observation_noise):
    level, slope, p00, p01, p11 = map(float, state)
    require(np.isfinite(state).all() and slope_noise > 0 and observation_noise > 0, "滤波状态或噪声无效")
    require(p00 > 0 and p11 > 0 and p00 * p11 - p01 * p01 >= -1e-12 * max(p00 * p11, 1e-30), "滤波协方差无效")
    a, b = p00 + 2 * p01 + p11, p01 + p11
    variance = a + observation_noise
    require(variance > 0 and np.isfinite(observation), "观测或创新方差无效")
    innovation = float(observation) - level - slope
    k0, k1 = a / variance, b / variance
    remain = 1 - k1
    # 将斜率协方差写成前一状态及两种噪声的传播，避免大数直接相减。
    next_p11 = k1 * k1 * (p00 + observation_noise) - 2 * k1 * remain * p01 + remain * remain * p11 + slope_noise
    updated = (level + slope + k0 * innovation, slope + k1 * innovation,
        a * observation_noise / variance, b * observation_noise / variance, next_p11)
    require(np.isfinite(updated).all() and updated[2] > 0 and updated[4] > 0, "滤波更新产生非法协方差")
    return updated, innovation, variance


def filter_likelihood(observations, slope_noise, observation_noise):
    """首价格作为初始均值，后续价格的高斯创新条件似然。"""
    y = np.asarray(observations, float)
    require(y.ndim == 1 and len(y) >= 2 and np.isfinite(y).all(), "训练价格必须完整且至少两条")
    require(slope_noise > 0 and observation_noise > 0, "估计噪声必须为正")
    level, slope, p00, p01, p11 = float(y[0]), 0., 1., 0., 1.
    objective = 0.
    for observation in y[1:].tolist():
        a, b = p00 + 2 * p01 + p11, p01 + p11
        variance = a + observation_noise
        if not math.isfinite(variance) or variance <= 0:
            return float("inf"), None
        innovation = observation - level - slope
        objective += .5 * (math.log(2 * math.pi) + math.log(variance) + innovation * innovation / variance)
        k0, k1 = a / variance, b / variance
        remain = 1 - k1
        next_p11 = k1 * k1 * (p00 + observation_noise) - 2 * k1 * remain * p01 + remain * remain * p11 + slope_noise
        level, slope, p00, p01, p11 = (level + slope + k0 * innovation, slope + k1 * innovation,
            a * observation_noise / variance, b * observation_noise / variance, next_p11)
        if not all(math.isfinite(x) for x in (level, slope, p00, p01, p11)) or p00 <= 0 or p11 <= 0:
            return float("inf"), None
    return objective, (level, slope, p00, p01, p11)


def fit_noise(observations, settings):
    y = np.asarray(observations, float)
    require(np.isfinite(y).all() and len(y) >= 2, "季度估计输入不完整")
    log_low, log_high = settings["log_variance_bounds"]
    variance = max(float(np.var(np.diff(y), ddof=1 if len(y) > 2 else 0)), settings["initial_variance_floor"])
    initial = np.clip(np.log([variance * settings["initial_slope_noise_ratio"], variance * settings["initial_observation_noise_ratio"]]), log_low, log_high)

    def objective(parameters):
        q, r = np.exp(parameters)
        value, _ = filter_likelihood(y, float(q), float(r))
        return value

    initial_objective = objective(initial)
    result = minimize(objective, initial, method="L-BFGS-B", bounds=[(log_low, log_high)] * 2,
        options={"maxiter": settings["optimizer_maxiter"], "ftol": settings["optimizer_ftol"], "gtol": settings["optimizer_gtol"], "maxls": settings["optimizer_maxls"]})
    q, r = map(float, np.exp(result.x))
    value, state = filter_likelihood(y, q, r)
    covariance_valid = state is not None and state[2] * state[4] - state[3] ** 2 >= -1e-12 * max(state[2] * state[4], 1e-30)
    valid = bool(result.success and np.isfinite(value) and state is not None and covariance_valid and value <= initial_objective + 1e-6)
    return {"success": valid, "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
        "optimizer_message": str(result.message), "optimizer_iterations": int(result.nit), "objective_evaluations": int(result.nfev),
        "initial_objective": initial_objective, "objective": value, "slope_noise": q, "observation_noise": r,
        "state": list(state) if state is not None else None, "training_observations": len(y),
        "conditional_likelihood_observations": len(y) - 1, "parameter_at_bound": bool(np.any(np.isclose(result.x, log_low, atol=1e-6, rtol=0) | np.isclose(result.x, log_high, atol=1e-6, rtol=0)))}


def policy_target(slope, slope_uncertainty, previous_target, entry_z=1.):
    require(np.isfinite(slope) and np.isfinite(slope_uncertainty) and slope_uncertainty > 0, "趋势或不确定性无效")
    require(previous_target is None or previous_target in (0., 1.), "上一目标必须为零一或未初始化")
    if slope <= 0:
        return 0., "趋势非正，退出或空仓"
    if slope > entry_z * slope_uncertainty:
        return 1., "正向趋势超过进入边界"
    return (0. if previous_target is None else float(previous_target)), "趋势仍正但未过进入边界，保持上一目标"


def walk_forward_states(data, settings, fitter=fit_noise, progress=None):
    dates = pd.DatetimeIndex(data.date)
    require(len(dates) > 0 and dates.is_monotonic_increasing and not dates.has_duplicates, "每日价格日期必须非空并严格递增")
    wealth = data.wealth.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0))).all(), "财富价格必须为正或明确缺失")
    logs = np.log(wealth)
    quarters = dates.to_period("Q")
    scheduled = np.r_[True, quarters[1:] != quarters[:-1]]
    state, parameters, last_fit, intent = None, None, None, None
    window = settings["training_observations"]
    require(isinstance(window, int) and window >= 2 and settings["entry_z"] > 0, "训练窗口或进入边界无效")
    rows, fits = [], []
    for t, day in enumerate(dates):
        update_status = "本日不更新参数"
        consumed = False
        if scheduled[t]:
            left = t - window + 1
            record = {"origin_index": t, "origin": day, "decision_time": day + pd.Timedelta(hours=15),
                "training_start_index": max(left, 0), "training_end_index": t,
                "training_start": dates[max(left, 0)], "training_end": day, "requested_observations": window}
            if left < 0:
                record.update(status="NO_VIEW_INSUFFICIENT_HISTORY", success=False)
            elif not np.isfinite(logs[left:t + 1]).all():
                record.update(status="SKIPPED_INCOMPLETE_HISTORY", success=False)
            else:
                estimated = fitter(logs[left:t + 1].copy(), settings)
                record.update(estimated)
                if estimated["success"]:
                    candidate = tuple(estimated["state"])
                    q, r = estimated["slope_noise"], estimated["observation_noise"]
                    require(len(candidate) == 5 and np.isfinite(candidate).all() and candidate[2] > 0 and candidate[4] > 0 and q > 0 and r > 0, "成功估计没有合法末端滤波状态")
                    state, parameters, last_fit = candidate, (q, r), t
                    consumed = True
                    record["status"] = "FIT_CONVERGED"
                else:
                    record["status"] = "FIT_FAILED_PRIOR_MODEL_RETAINED" if state is not None else "NO_VIEW_FIT_FAILED"
            fits.append(record)
            update_status = record["status"]
            if progress is not None:
                progress(record)
        row = {"date": day, "origin_index": t, "log_wealth": logs[t], "fit_update_status": update_status,
            "last_fit_origin_index": last_fit, "last_fit_origin": dates[last_fit] if last_fit is not None else pd.NaT,
            "fit_age_trading_days": t - last_fit if last_fit is not None else np.nan,
            "quarter_fit_consumed_current_observation": consumed, "daily_filter_updates": 0,
            "level": np.nan, "slope": np.nan, "level_variance": np.nan, "level_slope_covariance": np.nan,
            "slope_variance": np.nan, "slope_uncertainty": np.nan, "slope_z": np.nan,
            "innovation": np.nan, "innovation_variance": np.nan,
            "slope_noise": parameters[0] if parameters else np.nan, "observation_noise": parameters[1] if parameters else np.nan,
            "target": np.nan, "source_state": "NO_VIEW_NO_MATURE_MODEL", "policy_reason": "无成熟模型，不形成新目标"}
        if state is not None:
            if not np.isfinite(logs[t]):
                state = predict_state(state, parameters[0])
                row.update(source_state="NO_VIEW_MISSING_OBSERVATION", policy_reason="当日价格缺失，只推进内部时间，不形成新目标")
            else:
                if not consumed:
                    state, innovation, variance = filter_step(state, logs[t], *parameters)
                    row.update(innovation=innovation, innovation_variance=variance, daily_filter_updates=1)
                uncertainty = math.sqrt(state[4])
                intent, reason = policy_target(state[1], uncertainty, intent, settings["entry_z"])
                row.update(level=state[0], slope=state[1], level_variance=state[2], level_slope_covariance=state[3], slope_variance=state[4],
                    slope_uncertainty=uncertainty, slope_z=state[1] / uncertainty, target=intent,
                    source_state="MODEL_VIEW_AVAILABLE", policy_reason=reason)
        rows.append(row)
    return pd.DataFrame(rows), fits
