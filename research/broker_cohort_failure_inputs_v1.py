"""真实进入后的固定组失效适配器；原阶段引擎在独立模块中运行。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker_fixed_cohort_inputs_v1 as cohort_inputs
from research import broker_stage_policy_inputs_v1 as stage

POLICIES = ("COHORT_AND_PRICE_FAILURE", "PRICE_FAILURE_SAME_COVERAGE", "PRICE_FAILURE_ALL_DAYS")
NAMES = {POLICIES[0]: "固定组与ETF共同失效", POLICIES[1]: "同资料覆盖仅ETF失效", POLICIES[2]: "全部日期仅ETF失效"}
CHECK_LAG = 5
EPS = 1e-12
CHECK_COLUMNS = ["cycle_id", "entry_date", "decision_date", "execution_date", "relative_session", "cohort_source_date",
    "observation_source_date", "cohort_anchor_status", "cohort_eligible_count", "cohort_state", "cohort_view_allowed",
    "leaders_positive_lower", "leaders_positive_upper", "followers_positive_lower", "followers_positive_upper",
    "etf_lagged_five_return", "etf_price_known", "etf_price_failed", "original_exit", "extra_exit", "extra_reason", "policy"]


class FailureController:
    def __init__(self, panel, data, policy, original_exit):
        if policy not in POLICIES:
            raise ValueError("未知固定组持有失效政策。")
        self.panel, self.data, self.policy, self.original_exit = panel, data, policy, original_exit
        self.etf_logs = np.log((data.close + data.dividend) / data.close.shift()).to_numpy(float)
        self.checks = []

    def __call__(self, active, row, i, original_policy):
        original_reason = self.original_exit(active, row, i, original_policy)
        entry_idx = int(active["entry_idx"])
        if i - entry_idx != CHECK_LAG:
            return original_reason
        # 真实进入日作为新锚点；名单只依赖进入前一日，进入条件不依赖成分源。
        cohort = cohort_inputs.make_cohort(self.panel, entry_idx)
        observed = cohort_inputs.observe(self.panel, cohort, i)
        price_values = self.etf_logs[entry_idx:i]
        price_known = len(price_values) == CHECK_LAG and np.isfinite(price_values).all()
        price_return = float(np.expm1(price_values.sum())) if price_known else np.nan
        price_failed = bool(price_known and price_return <= EPS)
        source_known = bool(observed["view_allowed"] and observed["descriptive_propagation_state"] != "UNKNOWN_MAJORITY_BOUNDARY")
        both_weak = observed["descriptive_propagation_state"] == "NEITHER_POSITIVE_MAJORITY"
        if self.policy == POLICIES[0]:
            extra = source_known and both_weak and price_failed
        elif self.policy == POLICIES[1]:
            extra = source_known and price_failed
        else:
            extra = price_failed
        extra = bool(extra and original_reason is None)
        reason = f"{self.policy}_FIVE_SOURCE_CLOSES_FAILED" if extra else None
        self.checks.append({"cycle_id": int(active["cycle_id"]), "entry_date": active["entry_date"], "decision_date": row.date,
            "execution_date": self.data.date.iloc[i + 1] if i + 1 < len(self.data) else pd.NaT, "relative_session": i - entry_idx,
            "cohort_source_date": observed["cohort_source_date"], "observation_source_date": observed["observation_source_date"],
            "cohort_anchor_status": cohort.status, "cohort_eligible_count": cohort.eligible_count,
            "cohort_state": observed["descriptive_propagation_state"], "cohort_view_allowed": source_known,
            "leaders_positive_lower": observed["leaders_positive_lower"], "leaders_positive_upper": observed["leaders_positive_upper"],
            "followers_positive_lower": observed["followers_positive_lower"], "followers_positive_upper": observed["followers_positive_upper"],
            "etf_lagged_five_return": price_return, "etf_price_known": bool(price_known), "etf_price_failed": price_failed,
            "original_exit": original_reason, "extra_exit": extra, "extra_reason": reason, "policy": self.policy})
        return original_reason if original_reason is not None else reason


def private_engine():
    spec = importlib.util.spec_from_file_location("_510300_private_stage_cohort_failure_v1", Path(stage.__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法创建独立的原阶段引擎模块。")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def account(data, dividends, risks, panel, policy, cost, start):
    if len(data) > len(panel.dates) or not np.array_equal(pd.to_datetime(data.date).astype("datetime64[ns]").to_numpy(), panel.dates[:len(data)].to_numpy()):
        raise ValueError("账户与成分信息的原日历/行索引不一致。")
    engine = private_engine()
    controller = None
    if policy != "R212_CONTROL":
        controller = FailureController(panel, data, policy, engine.exit_decision)
        engine.exit_decision = controller
    result = engine.account(data, dividends, risks, "STAGE_ENTRY_AND_EXIT", cost, start)
    result["failure_checks"] = pd.DataFrame(controller.checks if controller is not None else [], columns=CHECK_COLUMNS)
    return result
