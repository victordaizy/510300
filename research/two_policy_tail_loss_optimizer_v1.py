"""经验尾部损失的线性优化与确定性并列预算选择。"""
import numpy as np
from scipy.optimize import linprog
from research.intraday_overnight_increment_v1 import require


def empirical_tail_loss(losses, confidence=.95):
    values = np.sort(np.asarray(losses, float))[::-1]
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "尾部损失样本无效")
    require(0 < confidence < 1, "尾部置信水平无效")
    mass = len(values)*(1-confidence)
    if round(mass) > 0 and abs(mass-round(mass)) < 1e-12:
        mass = float(round(mass))
    whole = int(np.floor(mass))
    fraction = mass-whole
    boundary = fraction*values[whole] if fraction > 0 else 0.
    return float((values[:whole].sum()+boundary)/mass)


def optimal_tail_budget(values, previous, confidence=.95):
    """只优化已给历史窗口，预算和为一且非负；并列时保留最近预算。"""
    values = np.asarray(values, float)
    require(values.ndim == 2 and values.shape[1] == 2 and len(values) >= 2 and np.isfinite(values).all(), "尾部优化的两条收益无效")
    require(0 <= previous <= 1 and 0 < confidence < 1, "此前预算或置信水平无效")
    n = len(values)
    objective = np.r_[0., 1., np.repeat(1/(n*(1-confidence)), n)]
    constraints = np.column_stack([values[:, 1]-values[:, 0], -np.ones(n), -np.eye(n)])
    bounds = [(0., 1.), (None, None)]+[(0., None)]*n
    common = {"A_ub": constraints, "b_ub": values[:, 1], "bounds": bounds, "method": "highs",
        "options": {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9, "ipm_optimality_tolerance": 1e-10}}
    answer = {"status": "NO_VIEW_OPTIMIZER_FAILURE_KEEP_BUDGET", "panic_budget": float(previous), "optimal_tail_loss": np.nan,
        "selected_tail_loss": np.nan, "optimal_budget_lower": np.nan, "optimal_budget_upper": np.nan, "linear_programs": 1}
    solved = linprog(objective, **common)
    if not solved.success:
        return answer
    # 固定最小损失目标，求全部最优预算的左右端点，不人为添加换手惩罚。
    direction = np.r_[1., np.zeros(n+1)]
    left = linprog(direction, A_eq=objective[None, :], b_eq=[solved.fun], **common)
    right = linprog(-direction, A_eq=objective[None, :], b_eq=[solved.fun], **common)
    answer["linear_programs"] = 3
    if not left.success or not right.success:
        return answer
    lower, upper = float(np.clip(left.x[0], 0., 1.)), float(np.clip(right.x[0], 0., 1.))
    if lower > upper+1e-8:
        return answer
    if lower > upper:
        lower = upper = float(np.clip(solved.x[0], 0., 1.))
    selected = float(np.clip(previous, lower, upper))
    loss = empirical_tail_loss(-(values @ [selected, 1-selected]), confidence)
    if abs(loss-solved.fun) > 1e-8:
        return answer
    answer.update(status="TAIL_LOSS_BUDGET_AVAILABLE", panic_budget=selected, optimal_tail_loss=float(solved.fun),
        selected_tail_loss=loss, optimal_budget_lower=lower, optimal_budget_upper=upper)
    return answer
