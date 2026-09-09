"""按未复利累计路径优化条件回撤，保留三次线性规划的可复核解。"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from research.intraday_overnight_increment_v1 import require
from research.two_policy_tail_loss_optimizer_v1 import empirical_tail_loss


def drawdown_path(returns):
    values = np.asarray(returns, float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "回撤收益序列必须非空完整")
    cumulative = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0., cumulative])[1:]
    return peaks-cumulative


def drawdown_program(values, confidence=.95):
    """变量依次为急跌预算、尾部分界、每日最高位置和每日超额回撤。"""
    values = np.asarray(values, float)
    require(values.ndim == 2 and values.shape[1] == 2 and len(values) >= 2 and np.isfinite(values).all(), "回撤优化的参考收益无效")
    require(0 < confidence < 1, "条件回撤置信水平无效")
    n = len(values)
    difference = np.cumsum(values[:, 0]-values[:, 1])
    baseline = np.cumsum(values[:, 1])
    zero = sparse.csr_matrix((n, 1))
    identity = sparse.eye(n, format="csr")
    peak_above_path = sparse.hstack([sparse.csr_matrix(difference[:, None]), zero, -identity, sparse.csr_matrix((n, n))], format="csr")
    monotone = sparse.diags([np.ones(n-1), -np.ones(n-1)], [0, 1], shape=(n-1, n), format="csr")
    peak_nondecreasing = sparse.hstack([sparse.csr_matrix((n-1, 2)), monotone, sparse.csr_matrix((n-1, n))], format="csr")
    excess_above_drawdown = sparse.hstack([sparse.csr_matrix(-difference[:, None]), sparse.csr_matrix(-np.ones((n, 1))), identity, -identity], format="csr")
    matrix = sparse.vstack([peak_above_path, peak_nondecreasing, excess_above_drawdown], format="csr")
    rhs = np.r_[-baseline, np.zeros(n-1), baseline]
    objective = np.r_[0., 1., np.zeros(n), np.repeat(1/(n*(1-confidence)), n)]
    bounds = [(0., 1.)]+[(0., None)]*(2*n+1)
    return objective, matrix, rhs, bounds


def saved_solution(answer):
    record = {"success": bool(answer.success), "status": int(answer.status), "message": str(answer.message),
              "objective": None if answer.fun is None else float(answer.fun),
              "x": None if answer.x is None else answer.x.tolist()}
    for name in ["ineqlin", "eqlin", "lower", "upper"]:
        values = getattr(getattr(answer, name, None), "marginals", None)
        record[name+"_marginals"] = None if values is None else np.asarray(values, float).tolist()
    return record


def optimal_drawdown_budget(values, previous, confidence=.95):
    require(np.isfinite(previous) and 0 <= previous <= 1, "此前预算无效")
    values = np.asarray(values, float)
    objective, matrix, rhs, bounds = drawdown_program(values, confidence)
    common = {"A_ub": matrix, "b_ub": rhs, "bounds": bounds, "method": "highs",
              "options": {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9, "ipm_optimality_tolerance": 1e-10}}
    answer = {"status": "NO_VIEW_OPTIMIZER_FAILURE_KEEP_BUDGET", "panic_budget": float(previous), "optimal_drawdown": np.nan,
              "selected_drawdown": np.nan, "optimal_budget_lower": np.nan, "optimal_budget_upper": np.nan,
              "linear_programs": 1, "certificate": {}}
    solved = linprog(objective, **common)
    answer["certificate"]["minimum_drawdown"] = saved_solution(solved)
    if not solved.success:
        return answer
    direction = np.zeros(len(objective)); direction[0] = 1.
    left = linprog(direction, A_eq=objective[None, :], b_eq=[solved.fun], **common)
    right = linprog(-direction, A_eq=objective[None, :], b_eq=[solved.fun], **common)
    answer["linear_programs"] = 3
    answer["certificate"].update(minimum_budget=saved_solution(left), maximum_budget=saved_solution(right))
    if not left.success or not right.success:
        return answer
    lower, upper = float(np.clip(left.x[0], 0., 1.)), float(np.clip(right.x[0], 0., 1.))
    if lower > upper+1e-8:
        return answer
    if lower > upper:
        lower = upper = float(np.clip(solved.x[0], 0., 1.))
    selected = float(np.clip(previous, lower, upper))
    risk = empirical_tail_loss(drawdown_path(values @ [selected, 1-selected]), confidence)
    if not np.isfinite(risk) or abs(risk-solved.fun) > 1e-8:
        return answer
    answer.update(status="CONDITIONAL_DRAWDOWN_BUDGET_AVAILABLE", panic_budget=selected, optimal_drawdown=float(solved.fun),
                  selected_drawdown=risk, optimal_budget_lower=lower, optimal_budget_upper=upper)
    return answer
