"""从已保存完整账户计算零现金收益下的稀疏夏普上界；不训练、不回测。"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import sparse_event_training_core_v3 as io

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sparse_sharpe_support_bound_v2"


def sharpe_cap(n, k, annual):
    """n个完整日历收益中，最多k项非零时，样本标准差口径的绝对夏普上界。"""
    if n < 2 or not 0 <= k <= n or annual <= 0:
        raise ValueError("交易日、支持天数或年化因子非法。")
    if k == 0 or k == n:
        return None
    return math.sqrt(annual * k * (n - 1) / (n * (n - k)))


def minimum_support(n, annual, target):
    value = Fraction(str(target)) ** 2 * n * n / (annual * (n - 1) + Fraction(str(target)) ** 2 * n)
    return (value.numerator + value.denominator - 1) // value.denominator


def required_active_ratio(n, k, annual, target):
    """持有日均值/持有日总体标准差的必要比值；不解释为胜率。"""
    fraction = k / n
    denominator = annual * (n - 1) / n * fraction - target**2 * (1 - fraction)
    return target / math.sqrt(denominator) if denominator > 0 else None


def formula_checks(cfg):
    target = cfg["target_net_sharpe"]
    cases = []
    for n, k in [(1849, 15), (1849, 18), (2420, 200), (2420, 250)]:
        returns = np.r_[np.full(k, 0.001), np.zeros(n - k)]
        measured = float(returns.mean() / returns.std(ddof=1) * math.sqrt(242))
        bound = sharpe_cap(n, k, 242)
        io.require(abs(measured - bound) < 1e-12, "等正收益序列未达到解析上界。")
        minimum = minimum_support(n, 242, target)
        io.require(sharpe_cap(n, minimum, 242) >= target, "最少天数不够。")
        io.require(sharpe_cap(n, minimum - 1, 242) < target, "最少天数不是最小整数。")
        cases.append({"n": n, "k": k, "analytical_cap": bound, "constructed_vector_sharpe": measured, "target_net_sharpe": target, "minimum_support_for_current_target": minimum})
    n, k = 2420, 200
    ratio = required_active_ratio(n, k, 242, target)
    active_mean = 0.001
    active_sigma = active_mean / ratio
    returns = np.r_[np.tile([active_mean - active_sigma, active_mean + active_sigma], k // 2), np.zeros(n - k)]
    actual = float(returns.mean() / returns.std(ddof=1) * math.sqrt(242))
    io.require(abs(actual - target) < 1e-12, "持有日信噪比关系不一致。")
    return {"status": "PASS_ANALYTICAL_IDENTITIES", "cases": cases, "required_ratio_identity_sharpe": actual,
        "synthetic_math_vectors_only": True, "market_paths_generated": 0, "accounts_generated": 0}


def freeze(out):
    io.require(not (out / "freeze.json").exists(), "本轮已固定，不覆盖。")
    cfg = io.load(out / "protocol.json")
    for study in cfg["studies"]:
        source = ROOT / "reports/research" / study
        destination = out / "inputs" / study
        destination.mkdir(parents=True)
        copies = {"daily.csv": "results/完整逐日账户.csv", "metrics.csv": "results/账户指标.csv",
            "protocol.json": "protocol.json", "summary.json": "summary.json", "delivery_receipt.json": "delivery_receipt.json"}
        for name, relative in copies.items():
            shutil.copy2(source / relative, destination / name)
    (out / "code").mkdir()
    for source in [Path(__file__), Path(io.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    shutil.copy2(ROOT / "config/510300_existing_data_training_mandate_v1.json", out / "inputs/mandate.json")
    paths = [out / "protocol.json", *[p for p in (out / "inputs").rglob("*") if p.is_file()], *list((out / "code").iterdir()), *[p for p in (out / "authority_update").rglob("*") if p.is_file()]]
    io.save(out / "freeze.json", {"frozen_at": io.now(), "saved_accounts_already_evaluated": True,
        "new_parameter_search": False, "identities": {p.relative_to(out).as_posix(): io.digest(p) for p in paths}})
    print("近期七项研究的完整保存账户与上界定义已固定。", flush=True)


def check_inputs(out):
    for name, expected in io.load(out / "freeze.json")["identities"].items():
        io.require(io.digest(out / name) == expected, f"保存来源变化：{name}")


def compute(out, cfg):
    mandate = io.load(out / "inputs/mandate.json")
    io.require(cfg["target_net_sharpe"] == mandate["target_net_sharpe"] and cfg["target_max_drawdown"] == mandate["target_max_drawdown"], "协议与本轮用户目标不一致。")
    rows, studies = [], []
    total_daily_rows = 0
    for study in cfg["studies"]:
        base = out / "inputs" / study
        source_cfg = io.load(base / "protocol.json")
        annual, capital = source_cfg["annual_days"], source_cfg["capital_cny"]
        io.require(annual == 242 and capital == 200000, "来源口径不是242日和20万元。")
        daily = pd.read_csv(base / "daily.csv")
        metrics = pd.read_csv(base / "metrics.csv")
        keys = [name for name in ["model", "timing", "policy", "scenario"] if name in daily]
        total_daily_rows += len(daily)
        count = 0
        for identity, frame in daily.groupby(keys, sort=True, dropna=False):
            identity = identity if isinstance(identity, tuple) else (identity,)
            item = dict(zip(keys, identity))
            frame = frame.sort_values("date").reset_index(drop=True)
            metric_mask = pd.Series(True, index=metrics.index)
            for key, value in item.items():
                metric_mask &= metrics[key] == value
            matched = metrics[metric_mask]
            io.require(len(matched) == 1, "账户与保存指标不能唯一对应。")
            metric = matched.iloc[0]
            n = len(frame)
            io.require(not frame.date.duplicated().any(), "账户日期重复。")
            equity = frame.equity_cny.to_numpy(float)
            returns = equity / np.r_[capital, equity[:-1]] - 1
            io.require(np.allclose(returns, frame.daily_return, atol=1e-12, rtol=0), "保存日收益与净值不一致。")
            exposed = frame.any_exposure.to_numpy(bool)
            k = int(exposed.sum())
            nonzero = int(np.count_nonzero(returns))
            inactive_nonzero = int(np.count_nonzero(returns[~exposed]))
            io.require(inactive_nonzero == 0, "空仓日存在非零收益，不能应用本轮零现金假设。")
            support_cap = sharpe_cap(n, nonzero, annual)
            exposure_cap = sharpe_cap(n, k, annual)
            volatility = float(returns.std(ddof=1))
            sharpe = float(returns.mean() / volatility * math.sqrt(annual)) if volatility > 1e-14 else None
            if sharpe is None:
                io.require(pd.isna(metric.net_sharpe), "零波动保存夏普必须未定义。")
            else:
                io.require(abs(sharpe - metric.net_sharpe) < 1e-10, "全账户保存夏普不同。")
                if support_cap is not None:
                    io.require(abs(sharpe) <= support_cap + 1e-10, "保存收益违反稀疏支持上界。")
            benchmark = item.get("policy") == "BENCHMARK" or item.get("model", item.get("timing")) in {"CASH", "BUY_AND_HOLD", "BUY_HOLD"}
            if k == 0:
                status = "ALL_CASH_SHARPE_UNDEFINED"
            elif exposure_cap is not None and exposure_cap < cfg["target_net_sharpe"]:
                status = "FIXED_EXPOSURE_SUPPORT_CAP_BELOW_TARGET"
            else:
                status = "SUPPORT_ALLOWS_TARGET_BUT_NO_PASS_IMPLIED"
            peak = np.maximum.accumulate(np.r_[capital, equity])[1:]
            drawdown = float(-(equity / peak - 1).min())
            io.require(abs(drawdown - metric.max_drawdown) < 1e-12, "保存回撤不同。")
            path_hash = hashlib.sha256(frame[["date", "equity_cny", "any_exposure"]].to_csv(index=False, float_format="%.17g").encode("utf-8")).hexdigest()
            active = returns[exposed]
            active_sigma = float(active.std(ddof=0)) if k else None
            rows.append({"study": study, "model": item.get("model", item.get("timing")), "policy": item.get("policy", "FROZEN_SINGLE_POLICY"),
                "scenario": item["scenario"], "benchmark": benchmark, "start": frame.date.iloc[0], "end": frame.date.iloc[-1],
                "n_calendar_days": n, "exposure_days": k, "nonzero_return_days": nonzero,
                "inactive_nonzero_return_days": inactive_nonzero, "opportunities": int(metric.opportunities),
                "exposure_days_per_year": k / n * annual, "zero_cash_day_fraction": 1 - k / n,
                "actual_sharpe": sharpe, "actual_max_drawdown": drawdown, "ending_equity_cny": equity[-1],
                "cap_with_same_exposure_days": exposure_cap, "cap_with_realized_nonzero_days": support_cap,
                "minimum_nonzero_days_for_target": minimum_support(n, annual, cfg["target_net_sharpe"]),
                "required_active_mean_to_population_std": required_active_ratio(n, k, annual, cfg["target_net_sharpe"]),
                "actual_active_mean_to_population_std": float(active.mean()) / active_sigma if active_sigma and active_sigma > 1e-14 else None,
                "support_status": status,
                "source_saved_target_net_sharpe": source_cfg["target_net_sharpe"],
                "source_saved_numeric_targets_pass": bool(metric.meets_numeric_targets),
                "current_target_net_sharpe": cfg["target_net_sharpe"],
                "current_numeric_targets_pass": bool(sharpe is not None and sharpe >= cfg["target_net_sharpe"] and drawdown <= cfg["target_max_drawdown"]),
                "path_sha256": path_hash})
            count += 1
        studies.append({"study": study, "account_records": count, "saved_daily_rows": len(daily)})
    return pd.DataFrame(rows), pd.DataFrame(studies), total_daily_rows


def reference_scenarios(cfg):
    n, years = 2420, 10
    rows = []
    for annual_opportunities, holding_days in [(1, 1), (1, 5), (4, 1), (4, 5), (5, 5), (4, 10)]:
        k = annual_opportunities * holding_days * years
        rows.append({"years": years, "n_calendar_days": n, "opportunities_per_year": annual_opportunities,
            "holding_days_per_opportunity": holding_days, "exposure_days": k,
            "zero_cash_day_fraction": 1 - k / n, "theoretical_cap": sharpe_cap(n, k, 242),
            "required_active_mean_to_population_std": required_active_ratio(n, k, 242, cfg["target_net_sharpe"]),
            "nonoverlap_assumption": True, "not_a_recommended_trade_quota": True})
    return pd.DataFrame(rows)


def run(out):
    io.require(not (out / "summary.json").exists(), "本轮已完成，不覆盖。")
    check_inputs(out)
    cfg = io.load(out / "protocol.json")
    checks = formula_checks(cfg)
    results, studies, total_daily = compute(out, cfg)
    for name, frame in [("保存账户的稀疏上界", results), ("来源研究范围", studies), ("机会频率数学示例", reference_scenarios(cfg))]:
        io.export(out / "results" / f"{name}.csv", frame)
    io.save(out / "formula_checks.json", checks)
    strategies = results[~results.benchmark]
    summary = {"study_id": cfg["study_id"], "completed_at": io.now(), "status": "COMPLETED_SAVED_ACCOUNT_SUPPORT_BOUND_ANALYSIS",
        "studies": len(studies), "account_records_including_controls": len(results),
        "strategy_account_records": len(strategies), "unique_saved_paths_including_controls": results.path_sha256.nunique(),
        "saved_daily_rows": total_daily, "support_status_counts_strategy_records": strategies.support_status.value_counts().to_dict(),
        "target_net_sharpe": cfg["target_net_sharpe"], "target_max_drawdown": cfg["target_max_drawdown"],
        "numeric_target_pass_strategy_records": int(strategies.current_numeric_targets_pass.sum()),
        "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0, "goal_achieved": False,
        "orders_authorized": False, "previous_goal_turn_classification": "PROGRESS_TARGET_CHANGED_BY_USER_AND_EXISTING_RESULTS_REASSESSED"}
    io.save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    display = strategies[(strategies.scenario == "STRESS") & (strategies.exposure_days > 0)]
    print(display[["study", "model", "policy", "n_calendar_days", "exposure_days", "actual_sharpe", "cap_with_same_exposure_days", "support_status"]].to_string(index=False))


def verify(out):
    check_inputs(out)
    cfg = io.load(out / "protocol.json")
    formula_checks(cfg)
    results, studies, total_daily = compute(out, cfg)
    for name, expected in [("保存账户的稀疏上界", results), ("来源研究范围", studies), ("机会频率数学示例", reference_scenarios(cfg))]:
        actual = pd.read_csv(out / "results" / f"{name}.csv")
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, rtol=1e-10, atol=1e-12)
    return {"status": "PASS_SAVED_RETURN_IDENTITIES_AND_ANALYTICAL_SUPPORT_BOUNDS", "account_records": len(results),
        "saved_daily_rows": total_daily, "target_net_sharpe": cfg["target_net_sharpe"],
        "strategy_numeric_target_pass_records": int(results.loc[~results.benchmark, "current_numeric_targets_pass"].sum()),
        "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="既有账户的稀疏夏普数学上界。")
    parser.add_argument("command", choices=["freeze", "calculate", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "calculate": run}[args.command](args.root)


if __name__ == "__main__":
    main()
