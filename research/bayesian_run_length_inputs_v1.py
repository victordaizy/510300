"""逐日保留全部行情持续时间的概率，只用已观察收益预测下一日均值。"""
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from research.intraday_overnight_increment_v1 import require

PRIMARY = "BAYESIAN_RUN_LENGTH"
CONTROL = "BAYESIAN_EXPANDING"


class GaussianRunLengthFilter:
    def __init__(self, prior, hazard):
        self.prior, self.hazard, self.count = dict(prior), float(hazard), 0
        require(0 <= self.hazard < 1, "阶段结束先验必须在零与一之间")
        require(np.isfinite(list(prior.values())).all() and prior["kappa"] > 0 and prior["alpha"] > 1 and prior["beta"] > 0,
                "正态逆伽马先验无效")
        self.mu, self.kappa, self.alpha, self.beta = [np.array([prior[k]], float) for k in ["mu", "kappa", "alpha", "beta"]]
        self.log_probability = np.array([0.])

    def predictive_log_density(self, value):
        require(np.isfinite(value), "观察收益不是有限值")
        scale_squared = self.beta*(self.kappa+1)/(self.alpha*self.kappa)
        degrees = 2*self.alpha
        return (gammaln((degrees+1)/2)-gammaln(degrees/2)-.5*np.log(degrees*np.pi*scale_squared)
                -(degrees+1)/2*np.log1p((value-self.mu)**2/(degrees*scale_squared)))

    def update(self, value):
        density = self.predictive_log_density(value)
        evidence = float(logsumexp(self.log_probability+density))
        kappa = self.kappa+1
        mu = (self.kappa*self.mu+value)/kappa
        alpha = self.alpha+.5
        beta = self.beta+self.kappa*(value-self.mu)**2/(2*kappa)
        if self.hazard:
            old_message = self.log_probability+density
            probabilities = np.r_[logsumexp(old_message)+np.log(self.hazard), old_message+np.log1p(-self.hazard)]
            probabilities -= logsumexp(probabilities)
            mu, kappa, alpha, beta = [np.r_[self.prior[key], arr] for key, arr in zip(["mu", "kappa", "alpha", "beta"], [mu, kappa, alpha, beta])]
            lengths = np.arange(len(probabilities))
        else:
            probabilities, lengths = np.array([0.]), np.array([self.count+1])
        weights = np.exp(probabilities)
        prediction = float(weights@mu)
        variance = float(weights@(beta*(kappa+1)/(kappa*(alpha-1))+(mu-prediction)**2))
        require(np.isfinite([prediction, variance, evidence]).all() and variance > 0, "阶段预测或方差不是有效有限值")
        require(np.isfinite(probabilities).all() and abs(weights.sum()-1) < 1e-10, "持续时间概率归一失败")
        require(np.isfinite(np.r_[mu, kappa, alpha, beta]).all() and (beta > 0).all(), "共轭更新参数无效")
        self.mu, self.kappa, self.alpha, self.beta, self.log_probability = mu, kappa, alpha, beta, probabilities
        self.count += 1
        return {"raw_forward_mean": prediction, "predictive_variance": variance, "log_evidence": evidence,
                "run_zero_probability": float(weights[0]) if self.hazard else 0.,
                "expected_run_length": float(weights@lengths), "most_probable_run_length": int(lengths[np.argmax(weights)]),
                "posterior_branches": len(weights), "observations": self.count}

    def final_parameters(self):
        return {"hazard": self.hazard, "observations": self.count,
                **{key: getattr(self, key).tolist() for key in ["mu", "kappa", "alpha", "beta", "log_probability"]}}


def sequential_forecasts(data, cfg, method):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and len(dates) > 1, "逐日预测日历必须完整递增")
    values = data.total_log.to_numpy(float)
    require(np.isnan(values[0]), "保存日收益首行应是没有前收盘的准备行")
    require(method in (PRIMARY, CONTROL), "未知阶段预测设置")
    filt = GaussianRunLengthFilter(cfg["prior"], cfg["hazard"] if method == PRIMARY else 0.)
    rows = [{"date": dates[0], "origin_index": 0, "prediction": np.nan, "model_status": "NO_VIEW_INITIAL_RETURN", "observations": 0}]
    parts, offsets, failed = [np.array([0.])], [0, 1], None
    for t in range(1, len(values)):
        info, probability = {}, np.array([], float)
        if failed is None and not np.isfinite(values[t]):
            failed = "NO_VIEW_SOURCE_GAP_REMAINDER"
        if failed is None:
            try:
                info = filt.update(float(values[t]))
                probability = filt.log_probability.copy()
            except (ValueError, FloatingPointError, OverflowError) as error:
                failed = "NO_VIEW_NUMERICAL_REMAINDER"
                info = {"failure": str(error)}
        status = failed or ("PREDICTION_AVAILABLE" if filt.count >= cfg["warmup_observations"] else "NO_VIEW_WARMUP")
        rows.append({"date": dates[t], "origin_index": t, "prediction": info.get("raw_forward_mean", np.nan) if status == "PREDICTION_AVAILABLE" else np.nan,
                     "model_status": status, "observations": filt.count, **info})
        parts.append(probability)
        offsets.append(offsets[-1]+len(probability))
    archive = {"offsets": np.asarray(offsets, np.int64), "log_probabilities": np.concatenate(parts)}
    return pd.DataFrame(rows), archive, {"method": method, "status": failed or "FILTER_COMPLETE", **filt.final_parameters()}


class MeanConfirmationController:
    def __init__(self, forecasts):
        self.values = forecasts.prediction.to_numpy(float)
        self.statuses = forecasts.model_status.to_numpy()

    def __call__(self, t, mode):
        current = float(self.values[t])
        previous = float(self.values[t-1]) if t else np.nan
        detail = {"account_mode": mode, "current_forward_mean": current, "previous_forward_mean": previous,
                  "model_status": str(self.statuses[t]), "policy_action": None}
        if mode == 2:
            return {**detail, "policy_action": 0, "decision_status": "LOCKED_EXIT_CONTINUES"}
        if not np.isfinite([current, previous]).all():
            return {**detail, "decision_status": "NO_VIEW_TWO_COMPLETE_FORECASTS"}
        if min(current, previous) > 0:
            return {**detail, "policy_action": 1, "decision_status": "TWO_POSITIVE_FORECASTS"}
        if max(current, previous) < 0:
            return {**detail, "policy_action": 0, "decision_status": "TWO_NEGATIVE_FORECASTS"}
        return {**detail, "policy_action": int(mode == 1), "decision_status": "MIXED_OR_ZERO_KEEP_ACTUAL_ASSET"}
