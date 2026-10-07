"""已用历史的否证统计及独立验收准入；不产生新交易策略。"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from research.point_account_nr7_inputs_v1 import return_statistics, trade_statistics


def completed(trades):
    return trades.loc[trades.status.eq("COMPLETE")].copy()


def concentration(trades):
    t = completed(trades)
    if t.empty:
        raise ValueError("利润集中度没有完成周期。")
    ordered = t.sort_values(["net_pnl", "entry_date"], ascending=[False, True])
    net = float(t.net_pnl.sum())
    positive = float(t.loc[t.net_pnl.gt(0), "net_pnl"].sum())
    top = ordered.iloc[0]
    remaining = t.drop(index=top.name)
    base, without = trade_statistics(t), trade_statistics(remaining)
    return {
        "complete_cycles": len(t), "completed_net_pnl": net, "positive_cycle_pnl": positive,
        "largest_profit_entry": top.entry_date, "largest_profit_exit": top.exit_date,
        "largest_profit_cny": float(top.net_pnl),
        "top1_fraction_of_net_profit": float(top.net_pnl / net) if net > 0 else np.nan,
        "top3_fraction_of_net_profit": float(ordered.head(3).net_pnl.sum() / net) if net > 0 else np.nan,
        "top5_fraction_of_net_profit": float(ordered.head(5).net_pnl.sum() / net) if net > 0 else np.nan,
        "top1_fraction_of_positive_profit": float(top.net_pnl / positive) if positive > 0 else np.nan,
        "original_p_times_b": base["p_times_b"], "original_mean_cycle_return": base["mean_cycle_net_return"],
        "without_largest_cycle_p_times_b": without["p_times_b"],
        "without_largest_cycle_mean_return": without["mean_cycle_net_return"],
        "scope": "完成周期集中度与删最大盈利一次的解释统计；不是新的账户或交易过滤规则。",
    }


def leave_one_year(daily, trades):
    rows = []
    t = completed(trades)
    for year in sorted(daily.date.dt.year.unique()):
        r = daily.loc[~daily.date.dt.year.eq(year), "net_return"]
        remaining = t.loc[~t.exit_date.dt.year.eq(year)]
        rows.append({"excluded_year": int(year), **return_statistics(r), **trade_statistics(remaining),
                     "scope": "日收益剔除该日历年；周期指标按退出年剔除。只作敏感性统计，不拼成可执行账户。"})
    return pd.DataFrame(rows)


def point_stats(values):
    x = np.asarray(values, float)
    positive, negative = x[x > 0], x[x < 0]
    if not len(x):
        return np.nan, np.nan
    pb = (len(positive) / len(x)) * (positive.mean() / -negative.mean()) if len(positive) and len(negative) else np.nan
    return float(pb), float(x.mean())


def calendar_year_trade_bootstrap(trades, first_year, last_year, replications=5000):
    """完整退出年份成组抽取，保留组内交易、亏损及空组；不假设每笔独立。"""
    t = completed(trades)
    years = list(range(first_year, last_year + 1))
    groups = [t.loc[t.exit_date.dt.year.eq(y), "net_return"].to_numpy(float) for y in years]
    rng = np.random.default_rng(20261001)
    rows = []
    for _ in range(replications):
        chosen = rng.integers(0, len(years), len(years))
        sample = np.concatenate([groups[j] for j in chosen])
        pb, mean = point_stats(sample)
        rows.append((pb, mean, len(sample)))
    a = np.asarray(rows)
    valid_pb = np.isfinite(a[:, 0])
    return {
        "replications": replications, "calendar_groups": len(years),
        "zero_trade_calendar_groups": sum(len(x) == 0 for x in groups),
        "undefined_p_times_b_replications": int((~valid_pb).sum()),
        "p_times_b_lower_2_5": float(np.nanquantile(a[:, 0], .025)) if valid_pb.any() else np.nan,
        "p_times_b_upper_97_5": float(np.nanquantile(a[:, 0], .975)) if valid_pb.any() else np.nan,
        "mean_cycle_return_lower_2_5": float(np.nanquantile(a[:, 1], .025)),
        "mean_cycle_return_upper_97_5": float(np.nanquantile(a[:, 1], .975)),
        "resampling_role": "年份组很少且历史已用于选择，只是描述区间，不是成功概率或独立验证。",
    }, a


def paired_block_effect(a, b, block=252, replications=2000):
    if len(a) != len(b) or not len(a):
        raise ValueError("配对账户长度不同或为空。")
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("配对账户有缺失。")
    n = len(a)
    rng = np.random.default_rng(20261001)
    values = []
    for _ in range(replications):
        start = rng.integers(0, n, size=math.ceil(n / block))
        indices = ((start[:, None] + np.arange(block)) % n).ravel()[:n]
        x, y = a[indices], b[indices]
        sx = np.sqrt(252) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 1e-14 else np.nan
        sy = np.sqrt(252) * y.mean() / y.std(ddof=1) if y.std(ddof=1) > 1e-14 else np.nan
        cx = np.expm1(252 * np.log1p(x).mean())
        cy = np.expm1(252 * np.log1p(y).mean())
        values.append((sy - sx, cy - cx))
    z = np.asarray(values)
    return {"block_trading_days": block, "replications": replications,
            "undefined_sharpe_difference": int((~np.isfinite(z[:, 0])).sum()),
            "sharpe_delta_lower_2_5": float(np.nanquantile(z[:, 0], .025)),
            "sharpe_delta_upper_97_5": float(np.nanquantile(z[:, 0], .975)),
            "cagr_delta_lower_2_5": float(np.quantile(z[:, 1], .025)),
            "cagr_delta_upper_97_5": float(np.quantile(z[:, 1], .975))}, z


def independent_claim_gate(evidence):
    """检查验收依据是否具备独立性；数字达线、重采样和历史切分均不能绕过。"""
    reasons = []
    required = {
        "candidate_frozen_at", "last_outcome_used_for_design", "evaluation_first_origin",
        "evaluation_last_origin", "actual_decisions_registered_before_execution",
        "full_calendar_coverage", "same_frozen_version", "evaluation_schedule_fixed_before_first_origin",
        "required_information_met", "uncertainty_gate_passed", "economic_gate_passed",
        "prospective_completed_cycles", "candidate_role", "legacy_terminal_rejection",
    }
    missing = required.difference(evidence)
    if missing:
        reasons.append("缺少必需证据字段：" + ",".join(sorted(missing)))
    if evidence.get("legacy_terminal_rejection"):
        reasons.append("候选有未撤销的终止裁决。")
    if evidence.get("candidate_role") != "PREDECLARED_PROSPECTIVE_CANDIDATE":
        reasons.append("当前身份是开发或机制诊断，未登记为该独立评价的固定候选。")
    try:
        frozen = pd.Timestamp(evidence["candidate_frozen_at"])
        first = pd.Timestamp(evidence["evaluation_first_origin"])
        last = pd.Timestamp(evidence["evaluation_last_origin"])
        used = pd.Timestamp(evidence["last_outcome_used_for_design"])
        if any(pd.isna(x) for x in (frozen, first, last, used)):
            raise ValueError("日期缺失")
        # 需要精确到实际时刻，不能用只到日期的模糊登记证明事前冻结。
        if frozen.tzinfo is None or first.tzinfo is None or last.tzinfo is None or used.tzinfo is None:
            reasons.append("冻结、已用历史和评价边界必须带真实时区时刻。")
        else:
            if not frozen < first:
                reasons.append("冻结时刻没有早于独立评价第一个决策时刻。")
            if not used < first:
                reasons.append("评价期间包含已用于设计或筛选的历史。")
            if first > last:
                reasons.append("评价时间顺序无效。")
    except (KeyError, TypeError, ValueError):
        reasons.append("冻结或评价时间证据缺失、无法解析。")
    flags = {
        "actual_decisions_registered_before_execution": "缺少完整的真实事前决定登记。",
        "full_calendar_coverage": "缺少全部交易日（含现金日）的连续覆盖。",
        "same_frozen_version": "评价期间没有保持同一固定版本。",
        "evaluation_schedule_fixed_before_first_origin": "没有在评价开始前固定验收时点及信息量设计。",
        "required_information_met": "事前规定的信息量尚未满足。",
        "uncertainty_gate_passed": "不确定性要求未通过或尚未计算。",
        "economic_gate_passed": "收益、夏普和实际交易质量等经济要求未同时通过。",
    }
    for key, message in flags.items():
        if evidence.get(key) is not True:
            reasons.append(message)
    cycles = evidence.get("prospective_completed_cycles")
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
        reasons.append("尚无合格的独立完整交易周期。")
    reasons = list(dict.fromkeys(reasons))
    return {"independent_promotion_allowed": not reasons,
            "classification": "INDEPENDENT_EVIDENCE_GATES_SATISFIED" if not reasons else "NOT_INDEPENDENTLY_VALIDATED",
            "reasons": reasons, "historical_numeric_metrics_unchanged": True,
            "a_pass_is_conditional_on_the_supplied_evidence_being_verified": True}


def require_independent_claim(evidence):
    result = independent_claim_gate(evidence)
    if not result["independent_promotion_allowed"]:
        raise ValueError("不能宣布去除过拟合或独立达标：" + "；".join(result["reasons"]))
    return result
