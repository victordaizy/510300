"""在原八项持仓状态上用固定高斯核学习继续收益。"""
from __future__ import annotations
from bisect import bisect_right
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values


def similarities(x, centers, gamma):
    x=np.atleast_2d(np.asarray(x,dtype=float)); centers=np.atleast_2d(np.asarray(centers,dtype=float))
    distance=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*x@centers.T,0.)
    return np.exp(-gamma*distance)


def fit_kernel_exit(rows, config):
    x=rows[FEATURES].to_numpy(float); y=rows.target.to_numpy(float); weights=rows.sample_weight.to_numpy(float)
    require(len(rows)>0 and np.isfinite(x).all() and np.isfinite(y).all(), "原八项核训练输入或标签不完整")
    require(np.isfinite(weights).all() and (weights>0).all(), "完整周期训练权重不合法")
    mean=np.average(x,axis=0,weights=weights)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)); scale=np.where(scale>1e-12,scale,1.)
    centers=np.clip((x-mean)/scale,-config["feature_clip"],config["feature_clip"])
    kernel=similarities(centers,centers,config["kernel_gamma"])
    sqrtw=np.sqrt(weights)
    regularized=sqrtw[:,None]*kernel*sqrtw[None,:]+config["ridge_alpha"]*np.eye(len(rows))
    factor=cho_factor(regularized,lower=True,check_finite=True)
    solved=cho_solve(factor,np.column_stack([sqrtw*y,sqrtw]),check_finite=True)
    denominator=float(sqrtw@solved[:,1]);require(denominator>0,"核回归截距方程分母非正")
    intercept=float(sqrtw@solved[:,0]/denominator)
    coefficients=sqrtw*(solved[:,0]-intercept*solved[:,1])
    if not (np.isfinite(coefficients).all() and np.isfinite(intercept)):
        raise FloatingPointError("核状态模型求解出现非有限数")
    return {"kind":"WEIGHTED_RBF_CONTINUATION_WITH_INTERCEPT","features":FEATURES.copy(),"mean":mean.tolist(),"scale":scale.tolist(),
        "centers":centers.tolist(),"dual_coefficients":coefficients.tolist(),"intercept":intercept,"feature_clip":config["feature_clip"],
        "kernel_gamma":config["kernel_gamma"],"ridge_alpha":config["ridge_alpha"],
        "training_origin_indexes":rows.origin_index.astype(int).tolist(),"training_cycle_ids":rows.cycle_id.astype(int).tolist()}


def kernel_prediction(model, values):
    require(model["kind"]=="WEIGHTED_RBF_CONTINUATION_WITH_INTERCEPT" and model["features"]==FEATURES,"核状态模型身份不符")
    normalized=np.clip((np.asarray(values)-model["mean"])/model["scale"],-model["feature_clip"],model["feature_clip"])
    values=similarities(normalized,model["centers"],model["kernel_gamma"])[0]
    return float(model["intercept"]+values@np.asarray(model["dual_coefficients"]))


def kernel_chinese_formula(model):
    lines=[f"本月截距{model['intercept']:.12f}，已成熟参考状态{len(model['centers'])}条。八项先按本月均值及标准差标准化并限制在负五至正五。与每个参考状态逐项相减、平方、相加，乘负八分之一后取自然指数，得到相似度。每个相似度乘对应训练系数后相加，再加截距。全部参考状态与系数列于同目录中文表格。", ""]
    lines.extend(f"- {name}：均值{mean:.12f}，标准差{scale:.12f}。" for name,mean,scale in zip(CN,model["mean"],model["scale"],strict=True))
    return lines


class KernelExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [r["fit_index"] for r in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "核状态模型月度时点必须唯一且递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        estimate, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "核状态退出模型使用未来周期或模型")
            estimate, status = kernel_prediction(stored["model"], values), "PREDICTION_AVAILABLE"
            require(np.isfinite(estimate), "核状态模型预测不是有限数")
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        elif stored and stored["status"] == "NO_VIEW_MODEL_FIT_FAILED":
            status = "NO_VIEW_MODEL_FIT_FAILED"
        self.negative_count = self.negative_count + 1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": estimate,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
