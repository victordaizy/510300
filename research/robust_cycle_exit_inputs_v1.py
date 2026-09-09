"""完整成熟周期上的联合胡伯回归，保留等周期基础权重与新周期平均截距。"""
from bisect import bisect_right

import numpy as np
from scipy.optimize import minimize

from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import state_values
from research.market_path_exit_inputs_v1 import FEATURES, CN
from research.market_path_state_v1 import MarketPathTracker


def weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    require(len(values) > 0 and values.shape == weights.shape and np.isfinite(values).all() and np.isfinite(weights).all() and np.all(weights > 0), "加权中位数的数值或权重无效")
    order = np.argsort(values, kind="stable")
    k = np.searchsorted(np.cumsum(weights[order]), weights.sum()/2, side="left")
    return float(values[order[min(int(k), len(order)-1)]])


def original_residual_scale(rows, initial):
    require(initial["kind"] == "MARKET_PATH_WITHIN_CYCLE_RIDGE" and initial["features"] == FEATURES, "稳健初始化必须是原117的固定成熟模型")
    x, y, w = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w > 0).all(), "稳健原始样本缺失或权重无效，禁止删行")
    z = np.clip((x-initial["mean"])/initial["scale"], -initial["feature_clip"], initial["feature_clip"])
    intercepts = {int(g["cycle_id"]): float(g["cycle_intercept"]) for g in initial["cycle_intercepts"]}
    require(set(rows.cycle_id.astype(int)) == set(intercepts), "稳健起点不是同一组完整成熟周期")
    residual = y-z@np.asarray(initial["coefficients"])-np.array([intercepts[int(c)] for c in rows.cycle_id])
    median = weighted_median(residual, w)
    mad = weighted_median(np.abs(residual-median), w)
    return {"median": median, "mad": mad, "scale": mad/0.6744897501960817, "residual": residual, "z": z}


def objective_gradient(parameters, z, y, weights, groups, delta, alpha):
    beta, intercepts = parameters[:z.shape[1]], parameters[z.shape[1]:]
    residual = z@beta+intercepts[groups]-y
    absolute = np.abs(residual)
    loss = np.where(absolute <= delta, .5*residual**2, delta*(absolute-.5*delta))
    influence = weights*np.clip(residual, -delta, delta)
    objective = float(weights@loss+.5*alpha*(beta@beta))
    gradient = np.r_[z.T@influence+alpha*beta, np.bincount(groups, weights=influence, minlength=len(intercepts))]
    return objective, gradient


def fit_robust_cycle(rows, cfg, initial):
    start = original_residual_scale(rows, initial)
    if not np.isfinite(start["scale"]) or start["scale"] <= 1e-12:
        raise RuntimeError("稳健残差尺度不足，不能改尺度完成拟合")
    delta = cfg["huber_constant"]*start["scale"]
    ids = sorted(rows.cycle_id.astype(int).unique())
    index = {cid: j for j, cid in enumerate(ids)}
    groups = np.array([index[int(cid)] for cid in rows.cycle_id])
    w, y, z = rows.sample_weight.to_numpy(float), rows.target.to_numpy(float), start["z"]
    np.testing.assert_allclose(np.bincount(groups, weights=w), 1., atol=1e-12, rtol=0)
    old = {int(g["cycle_id"]): g for g in initial["cycle_intercepts"]}
    theta = np.r_[initial["coefficients"], [old[cid]["cycle_intercept"] for cid in ids]]
    args = (z, y, w, groups, delta, cfg["ridge_alpha"])
    initial_objective = objective_gradient(theta, *args)[0]
    fitted = minimize(objective_gradient, theta, args=args, jac=True, method="L-BFGS-B",
                      options={"maxiter": cfg["maximum_iterations"], "ftol": cfg["objective_tolerance"], "gtol": cfg["gradient_tolerance"], "maxls": cfg["maximum_line_search_steps"]})
    objective, gradient = objective_gradient(fitted.x, *args)
    if not fitted.success or not np.isfinite(fitted.x).all() or np.abs(gradient).max() > cfg["accepted_gradient_bound"]:
        raise RuntimeError("联合胡伯回归未满足冻结收敛条件："+str(fitted.message))
    require(objective <= initial_objective+1e-12, "稳健目标值高于原固定起点")
    beta, intercepts = fitted.x[:8], fitted.x[8:]
    effects = [{**old[cid], "cycle_intercept": float(intercepts[j]), "original_square_intercept": float(old[cid]["cycle_intercept"])} for j, cid in enumerate(ids)]
    residual = z@beta+intercepts[groups]-y
    return {"kind": "MARKET_PATH_JOINT_CYCLE_HUBER", "features": FEATURES.copy(), "mean": initial["mean"], "scale": initial["scale"],
            "coefficients": beta.tolist(), "intercept": float(intercepts.mean()), "feature_clip": initial["feature_clip"], "cycle_intercepts": effects,
            "huber_constant": cfg["huber_constant"], "huber_delta": delta, "original_residual_median": start["median"],
            "original_residual_mad": start["mad"], "original_residual_scale": start["scale"], "initial_huber_objective": initial_objective,
            "final_huber_objective": objective, "maximum_gradient": float(np.abs(gradient).max()), "iterations": int(fitted.nit),
            "limited_influence_rows": int((np.abs(residual) > delta).sum()), "limited_influence_base_weight": float(w[np.abs(residual) > delta].sum()),
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def robust_prediction(model, values):
    require(model["kind"] == "MARKET_PATH_JOINT_CYCLE_HUBER" and model["features"] == FEATURES, "联合胡伯模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "联合胡伯预测需要完整八状态")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class RobustCycleExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models, self.confirmation_days = data, models, confirmation_days
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "联合胡伯月份必须唯一递增")
        self.market_path, self.cycle_id, self.negative_count = MarketPathTracker(data), None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = np.asarray(state_values(self.data, t, cycle, current_value, peak_value), float)
        account_return, account_drawdown = float(x[1]), float(x[2])
        market = self.market_path.observe(t, cycle, current_value)
        x[1:3] = [market["market_cycle_return"], market["market_cycle_drawdown"]]
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "联合胡伯退出读取未来周期或模型")
            if np.isfinite(x).all():
                value, status = robust_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                "account_cycle_return": account_return, "account_cycle_drawdown": account_drawdown, **market, **dict(zip(FEATURES, x))}
