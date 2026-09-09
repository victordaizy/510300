"""用已知收盘收益估计条件方差，在现有策略目标外增加同口径风险缩减。"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import softmax
from research.intraday_overnight_increment_v1 import require

PERSISTENCE_LIMIT = .999
MODELS = ["CONDITIONAL_VARIANCE_BUDGET", "ROLLING_VARIANCE_BUDGET_CONTROL"]


def parameters(theta, initial_variance):
    weights = softmax(np.r_[0., np.asarray(theta)[1:]])
    return float(initial_variance*np.exp(theta[0])), float(PERSISTENCE_LIMIT*weights[1]), float(PERSISTENCE_LIMIT*weights[2])


def variance_path(returns_percent, omega, alpha, beta, initial_variance):
    r = np.asarray(returns_percent, float)
    require(len(r) > 0 and np.isfinite(r).all() and initial_variance > 0, "条件方差需要完整已知收益与正初始方差")
    h = np.empty(len(r))
    h[0] = initial_variance
    if len(r) > 1:
        h[1:], _ = lfilter([1.], [1., -beta], omega+alpha*r[:-1]**2, zi=[beta*initial_variance])
    return h


def objective_gradient(theta, returns_percent, initial_variance):
    r = np.asarray(returns_percent, float)
    omega, alpha, beta = parameters(theta, initial_variance)
    h = variance_path(r, omega, alpha, beta, initial_variance)
    require(np.isfinite(h).all() and (h > 0).all(), "条件方差递推出现无效数值")
    objective = .5*np.mean(np.log(h)+r*r/h)
    derivatives = np.zeros((len(r), 3))
    if len(r) > 1:
        derivatives[1:, 0] = lfilter([1.], [1., -beta], np.ones(len(r)-1))
        derivatives[1:, 1] = lfilter([1.], [1., -beta], r[:-1]**2)
        derivatives[1:, 2] = lfilter([1.], [1., -beta], h[:-1])
    score = .5*(1./h-r*r/h**2)/len(r)
    raw_gradient = derivatives.T@score
    cross = -alpha*beta/PERSISTENCE_LIMIT
    jacobian = np.array([[omega, 0., 0.], [0., alpha*(1-alpha/PERSISTENCE_LIMIT), cross], [0., cross, beta*(1-beta/PERSISTENCE_LIMIT)]])
    return float(objective), jacobian.T@raw_gradient


def fit_variance(returns, cfg):
    r = np.asarray(returns, float)*100.
    require(np.isfinite(r).all(), "训练窗口缺失，禁止删行拟合")
    initial = float(np.mean(r*r))
    if initial <= 0:
        return {"status": "NO_VIEW_ZERO_TRAINING_VARIANCE", "model": None}
    theta = np.array([np.log(.05), np.log(.05/(PERSISTENCE_LIMIT-.95)), np.log(.90/(PERSISTENCE_LIMIT-.95))])
    fitted = minimize(objective_gradient, theta, args=(r, initial), jac=True, method="L-BFGS-B", bounds=[(-12., 6.), (-12., 12.), (-12., 12.)],
                      options={"maxiter": cfg["maximum_iterations"], "ftol": 1e-10, "gtol": 1e-6, "maxls": 20})
    record = {"solver_success": bool(fitted.success), "solver_message": str(fitted.message), "iterations": int(fitted.nit), "objective": float(fitted.fun),
              "initial_objective": objective_gradient(theta, r, initial)[0], "raw_gradient_maximum": float(np.max(np.abs(fitted.jac))), "theta": fitted.x.tolist()}
    if not fitted.success or not np.isfinite(fitted.fun):
        return {**record, "status": "NO_VIEW_VARIANCE_FIT_FAILED", "model": None}
    omega, alpha, beta = parameters(fitted.x, initial)
    path = variance_path(r, omega, alpha, beta, initial)
    model = {"omega": omega, "alpha": alpha, "beta": beta, "initial_variance_percent_squared": initial, "variance_at_fit_percent_squared": float(path[-1]),
             "next_variance_percent_squared": float(omega+alpha*r[-1]**2+beta*path[-1])}
    return {**record, "status": "FIT_COMPLETE", "model": model}


def train_schedule(data, schedule, cfg):
    records = []
    for old in schedule:
        t = int(old["fit_index"])
        left = max(1, t-cfg["training_window"]+1)
        rows = data.total_simple.iloc[left:t+1].to_numpy(float)
        base = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "training_start_index": left, "training_rows": len(rows), "latest_observed_return_index": t}
        if len(rows) < cfg["minimum_training_rows"]:
            fitted = {"status": "NO_VIEW_INSUFFICIENT_VARIANCE_HISTORY", "model": None}
        elif not np.isfinite(rows).all():
            fitted = {"status": "NO_VIEW_INCOMPLETE_VARIANCE_WINDOW", "model": None}
        else:
            fitted = fit_variance(rows, cfg)
        records.append({**base, **fitted})
    return records


def forecast_frame(data, records, annual_days=242):
    by_index = {r["fit_index"]: r for r in records}
    require(len(by_index) == len(records), "条件方差拟合日期重复")
    saved = []
    model, next_variance, fit_index = None, np.nan, None
    for t, row in enumerate(data.itertuples()):
        status = "NO_VIEW_NO_VARIANCE_MODEL"
        if t in by_index:
            r = by_index[t]
            require(r["latest_observed_return_index"] <= t, "条件方差使用未来收益拟合")
            model, fit_index = r["model"], t
            next_variance = model["next_variance_percent_squared"] if model else np.nan
            status = "CONDITIONAL_VARIANCE_AVAILABLE" if model else r["status"]
        elif model is not None:
            if np.isfinite(row.total_simple) and np.isfinite(next_variance):
                next_variance = model["omega"]+model["alpha"]*(row.total_simple*100.)**2+model["beta"]*next_variance
                status = "CONDITIONAL_VARIANCE_AVAILABLE"
            else:
                next_variance = np.nan
                status = "NO_VIEW_MISSING_RECURSIVE_RETURN"
        volatility = np.sqrt(next_variance*annual_days)/100. if np.isfinite(next_variance) and next_variance > 0 else np.nan
        saved.append({"date": row.date, "variance_status": status, "variance_fit_index": fit_index, "next_variance_percent_squared": next_variance,
                      "forecast_annual_volatility": volatility, "realized_annual_volatility20": row.vol20})
    return pd.DataFrame(saved)


def budget_frame(parent, forecasts, target_volatility=.10):
    require(pd.DatetimeIndex(parent.date).equals(pd.DatetimeIndex(forecasts.date)), "条件方差与原参考目标日历不符")
    result = forecasts.copy()
    result["parent_target"] = parent.target.to_numpy(float)
    for name, column in [(MODELS[0], "forecast_annual_volatility"), (MODELS[1], "realized_annual_volatility20")]:
        last_multiplier = 1.
        targets, multipliers, statuses = [], [], []
        for target, volatility in zip(parent.target, forecasts[column], strict=True):
            if np.isfinite(volatility) and volatility > 0:
                last_multiplier = min(1., target_volatility/volatility)
                status = "RISK_MULTIPLIER_AVAILABLE"
            else:
                status = "NO_VIEW_KEEP_PREVIOUS_EXPLICIT_RISK_MULTIPLIER"
            if np.isfinite(target):
                require(0 <= target <= 1, "原策略目标超出不融资范围")
                value = float(target)*last_multiplier
            else:
                value, status = np.nan, "NO_VIEW_PARENT_TARGET"
            targets.append(value)
            multipliers.append(last_multiplier)
            statuses.append(status)
        result[name] = targets
        result[f"{name}_multiplier"] = multipliers
        result[f"{name}_status"] = statuses
    return result
