"""有限起点的保存净收益虚拟组合优化，仅用于研究筛选。"""
import math
import time

import numpy as np
from scipy.optimize import minimize

from research.intraday_overnight_increment_v1 import require


def fit_static_mixtures(matrices, starts):
    require(len(matrices) == 4 and len(starts) == 2, '虚拟组合必须是四场景和两个固定起点')
    n = matrices[0].shape[1]
    require(all(m.ndim == 2 and m.shape[1] == n and np.isfinite(m).all() for m in matrices), '虚拟收益矩阵无效')
    means = [m.mean(axis=0) for m in matrices]
    covariances = [np.cov(m, rowvar=False, ddof=1) for m in matrices]

    def scores(weights):
        values, gradients = [], []
        for matrix, mean, cov in zip(matrices, means, covariances):
            returns = matrix @ weights
            vol = math.sqrt(max(float(weights @ cov @ weights), 1e-24))
            average = float(mean @ weights)
            annual = float(np.expm1(np.log1p(returns).sum()*242/len(returns)))
            values.extend([average/vol*np.sqrt(242)/1.2, annual/.1])
            gradients.extend([np.sqrt(242)/1.2*(mean/vol-average*(cov @ weights)/vol**3),
                              (annual+1)*242/len(returns)*(matrix.T @ (1/(1+returns)))/.1])
        return np.array(values), np.array(gradients)

    def constraint(x):
        return scores(x[:-1])[0]-x[-1]

    def jacobian(x):
        return np.column_stack([scores(x[:-1])[1], -np.ones(8)])

    solutions = []
    for label, initial in starts:
        initial = np.asarray(initial, float)
        require(initial.shape == (n,) and np.all(initial >= 0) and abs(initial.sum()-1) < 1e-10, '固定求解起点无效')
        began = time.perf_counter()
        x0 = np.r_[initial, min(scores(initial)[0])-1e-5]
        fit = minimize(lambda x: -x[-1], x0, method='SLSQP', jac=lambda x: np.r_[np.zeros(n), -1.],
            bounds=[(0., 1.)]*n+[(None, 3.)],
            constraints=[{'type': 'eq', 'fun': lambda x: x[:-1].sum()-1,
                          'jac': lambda x: np.r_[np.ones(n), 0.]},
                         {'type': 'ineq', 'fun': constraint, 'jac': jacobian}],
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        weights = np.maximum(fit.x[:-1], 0.)
        weights /= weights.sum()
        values, _ = scores(weights)
        solutions.append({'start': label, 'success': bool(fit.success), 'message': str(fit.message),
            'iterations': int(fit.nit), 'objective_evaluations': int(fit.nfev), 'weights': weights.tolist(),
            'minimum_joint_ratio': float(min(values)), 'metric_ratios': values.tolist(),
            'seconds': time.perf_counter()-began})
    return solutions
