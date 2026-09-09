"""复用月度最小方差预算，以同费用连续目标生成新组合。"""
import numpy as np
from research.intraday_overnight_increment_v1 import require
from research.recent_reference_selection_inputs_v1 import aligned_reference_inputs, MODELS
from research.two_policy_min_variance_inputs_v1 import budget_frame

PRIMARY = "COVARIANCE_REFERENCE_PAIR"


def covariance_reference_frames(data, base_ledgers, parents_by_cost, cfg, start):
    """风险预算仅估计一次；传入常数状态只提取既有控制器预算。"""
    require(cfg["risk_window"] == 242, "两参考共同风险窗口改变")
    first, returns, target_arrays = aligned_reference_inputs(data, base_ledgers, parents_by_cost, {**cfg, "selection_window": cfg["risk_window"]}, start)
    original = budget_frame(data.date, returns, np.ones_like(returns), first, window=cfg["risk_window"])
    risk = original.drop(columns=["panic_state", "learned_state", "target"]).rename(columns={
        "panic_budget": "downside_budget", "learned_budget": "continuous_budget", "panic_sd": "downside_sd", "learned_sd": "continuous_sd",
        "raw_panic_budget": "raw_downside_budget"}).copy()
    risk["origin_index"] = np.arange(len(data))
    risk["risk_cost"] = "BASE"
    risk["downside_reference_return"] = returns[:, 0]
    risk["continuous_reference_return"] = returns[:, 1]
    weights = risk[["downside_budget", "continuous_budget"]].to_numpy(float)
    frames, summaries = {}, []
    for cost_id, targets in target_arrays.items():
        factors = risk.copy()
        factors["target_cost"] = cost_id
        factors["downside_parent_target"] = targets[:, 0]
        factors["continuous_parent_target"] = targets[:, 1]
        known = np.isfinite(targets).all(axis=1) & np.isfinite(weights).all(axis=1)
        final_target = np.full(len(data), np.nan)
        final_target[known] = np.sum(weights[known]*targets[known], axis=1)
        factors["target"] = final_target
        eligible = factors.iloc[first-1:-1]
        frames[cost_id] = factors
        summaries.append({"cost": cost_id, "decision_origins": len(eligible), "monthly_attempts": int(factors.risk_update_scheduled.sum()),
            "budget_changes": int(eligible.downside_budget.diff().abs().gt(1e-12).sum()),
            "downside_full_budget_origins": int(eligible.downside_budget.eq(1).sum()), "continuous_full_budget_origins": int(eligible.downside_budget.eq(0).sum()),
            "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
            "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
