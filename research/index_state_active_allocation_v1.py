"""固定检验：只能给当日已有信号的参考策略分配指数仓位。"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.index_state_strategy_value_v1 as parent
import research.index_state_strategy_value_saved_completion_v1 as source
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, now

OUT = ROOT / "reports/research/510300_index_state_active_allocation_v1"
STUDY = "510300_INDEX_STATE_ACTIVE_ALLOCATION_V1"


def active_allocation(mu, covariance, available):
    """沿同一凹目标枚举允许的约束面；当日无信号的参考只能给零权重。"""
    matrix = .5 * covariance + .5 * np.diag(np.diag(covariance)) + np.eye(3) * 1e-10
    permitted = np.flatnonzero(available).tolist()
    best, score = np.zeros(3), 0.
    for size in range(1, len(permitted) + 1):
        for subset in combinations(permitted, size):
            ids = list(subset)
            c, m = matrix[np.ix_(ids, ids)], mu[ids]
            inverse_mu = np.linalg.solve(c, m)
            inverse_one = np.linalg.solve(c, np.ones(size))
            multiplier = (inverse_mu.sum() - 4.) / inverse_one.sum()
            for candidate in [inverse_mu / 4., (inverse_mu - multiplier * inverse_one) / 4.]:
                if candidate.min() < -1e-10 or candidate.sum() > 1 + 1e-10:
                    continue
                weights = np.zeros(3)
                weights[ids] = np.maximum(candidate, 0)
                if weights.sum() > 1:
                    weights /= weights.sum()
                value = float(weights @ mu - 2 * weights @ matrix @ weights)
                if value > score + 1e-12:
                    best, score = weights, value
    assert np.all(best[~available] == 0)
    return best, score, matrix


def checks():
    rng = np.random.default_rng(20261002)
    count = 0
    for _ in range(40):
        values = rng.normal(size=(20, 3)) * .01
        covariance = values.T @ values / len(values)
        mu = rng.normal(size=3) * .002
        a, av, _ = active_allocation(mu, covariance, np.ones(3, dtype=bool))
        b, bv, _ = parent.allocation(mu, covariance)
        np.testing.assert_allclose(a, b, atol=1e-10)
        assert abs(av - bv) < 1e-12
        z, zv, _ = active_allocation(mu, covariance, np.zeros(3, dtype=bool))
        assert not z.any() and zv == 0
        count += 1
    return {"all_available_matches_parent": count, "none_available_cash": count}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("当日可用策略分配已经固定。")
    source.verify_sources(source.OUT)
    assert (source.OUT / "completion_receipt.json").exists()
    verification = checks()
    for name in ["code", "inputs", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    shutil.copytree(source.OUT / "references", root / "references")
    for name in ["expert_value_labels.parquet", "expert_current_targets.parquet", "allocation_predictions.parquet", "saved_allocations.json"]:
        shutil.copy2(source.OUT / "results" / name, root / "inputs" / name)
    saved = read(root / "inputs/saved_allocations.json")
    structural = []
    for policy in ["HISTORY", "PRICE_HMM", "MACRO_HMM"]:
        rows = [r for r in saved if r["policy"] == policy]
        weights = np.array([r["weights"] for r in rows])
        targets = np.array([r["expert_current_targets"] for r in rows])
        mu = np.array([r["expert_mu5"] for r in rows])
        available = targets.sum(axis=1) > 0
        idle = ((weights * targets).sum(axis=1) < 1e-10) & available
        structural.append({"policy": policy, "signal_days": int(available.sum()),
                           "signal_but_zero_allocation_days": int(idle.sum()),
                           "idle_despite_positive_available_expert_mean": int((idle & ((mu * targets) > 0).any(axis=1)).sum()),
                           "mean_inactive_expert_weight_on_signal_days": float((weights * (1 - targets)).sum(axis=1)[available].mean())})
    protocol = {**read(parent.OUT / "protocol.json"), "study_id": STUDY, "at": now(),
        "question": "若分配只允许当日已有参考持仓信号的策略，是否比允许给暂时空仓参考配置权重更有利。",
        "motivation": structural,
        "interpretation": "旧策略给当日空仓参考分配资金可能具有等待未来机会的经济价值，不能直接认定为错误。本轮只检验新增当日可用动作约束。",
        "fixed_change": "参考当前目标为0则优化权重固定为0；当前目标为1才允许优化。保持原三策略五日均值与协方差、收缩和惩罚，未改训练样本或状态。全无可用信号或无正效用时现金。",
        "forecast_label_limit": "沿用原无条件参考五日价值，尚未按当前持仓阶段条件化收益标签；此实验不能证明解决全部价值标注问题。",
        "new_hidden_model_fits": 0, "new_statistical_fits": 0, "new_optimizations": len(saved),
        "new_reference_accounts": 0, "new_full_accounts": 6, "recomputed_equal_accounts": 2,
        "new_parameter_grid": 0, "checks": verification,
        "primary_comparison": "主宏观状态政策对其父政策；原价格与无状态对照同样加可用约束，等权对照逐列重现。20日块2000次seed20261002。"}
    save(root / "protocol.json", protocol, True)
    files = local_import_closure({Path(__file__)})
    files.update(ROOT / name for name in read(parent.OUT / "freeze.json")["sources"])
    files.update(p for p in source.OUT.rglob("*") if p.is_file())
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(root / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)},
         "copies": {p.relative_to(root).as_posix(): digest(p) for folder in ["inputs", "references"] for p in (root / folder).rglob("*") if p.is_file()}}, True)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(path, mandate)
    print("当日可用动作约束已固定，所有旧预测复用，未重新估计状态或收益。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)
    for name, expected in record["copies"].items():
        assert digest(root / name) == expected


def learn_active(root, labels, targets):
    frame = pd.read_parquet(root / "inputs/allocation_predictions.parquet")
    allocations = read(root / "inputs/saved_allocations.json")
    updated, raw_targets = [], {}
    for record in allocations:
        mu, covariance = np.array(record["expert_mu5"]), np.array(record["expert_covariance"])
        target = np.array(record["expert_current_targets"])
        assert np.isin(target, [0., 1.]).all()
        weights, utility, regularized = active_allocation(mu, covariance, target > 0)
        raw = float(np.clip(weights @ target, 0., 1.))
        raw_targets[record["idx"], record["policy"]] = raw
        updated.append({**record, "parent_weights": record["weights"], "parent_raw_target": record["raw_target"],
                        "weights": weights.tolist(), "cash_weight": float(np.clip(1 - weights.sum(), 0, 1)),
                        "utility": utility, "regularized_covariance": regularized.tolist(), "raw_target": raw,
                        "available_experts": (target > 0).tolist()})
    for idx, row in frame.iterrows():
        if row.model != "EQUAL_EXPERTS":
            frame.loc[idx, "raw_target"] = raw_targets[int(row.idx), row.model]
    assert frame.raw_target.between(0, 1).all()
    frame.to_parquet(root / "results/allocation_predictions.parquet", index=False)
    save(root / "results/saved_allocations.json", updated, True)
    return frame, updated


def run(root):
    # 原运行结尾的无约束求解器不适用于新增动作约束；独立验证仍然保留，改为在受限坐标求解。
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    h, e = parent.inventory.modules(parent.state.OUT)
    d, x, frame, dividends = parent.inventory.inputs(parent.state.OUT, e)
    data = pd.read_parquet(parent.LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(d.date.iloc[-1])].reset_index(drop=True)
    np.testing.assert_array_equal(data.date.to_numpy(dtype="datetime64[ns]"), d.date.to_numpy(dtype="datetime64[ns]"))
    for column in ["open", "high", "low", "close"]:
        np.testing.assert_array_equal(data[column], d[column])
    labels, targets = source.saved_references(root, data, dividends)
    predictions, allocations = learn_active(root, labels, targets)
    from types import SimpleNamespace
    proxy = SimpleNamespace(**vars(h))
    proxy.plan = lambda engine, account, ref, peak, forecast, pressure: parent.account_plan(h, engine, account, ref, peak, forecast, pressure)
    proxy.risk_valid = parent.state.risk_valid
    accounts, measurements = {}, []
    for cost in h.COSTS:
        folder = root / "accounts" / cost
        folder.mkdir()
        for policy in parent.POLICIES:
            ledger, decisions = parent.inventory.simulate(proxy, e, d, x, dividends, predictions, policy, cost)
            ledger.to_parquet(folder / f"{policy}_ledger.parquet", index=False)
            decisions.to_parquet(folder / f"{policy}_decisions.parquet", index=False)
            accounts[policy, cost] = ledger
            measurement = {"policy": policy, "cost": cost, **h.summary(e, ledger)}
            measurement["historical_point_targets_met"] = measurement["net_sharpe"] is not None and measurement["net_sharpe"] >= 1.2 and measurement["annualized_return"] >= .1 and measurement["max_drawdown"] >= -.1
            measurements.append(measurement)
            assert ledger.accounting_error.abs().max() < 1e-6 and ledger.cash.ge(-1e-7).all()
            assert (-ledger.filled_quantity.clip(upper=0)).le(ledger.sellable_before).all()
            for row in decisions.loc[decisions.filled_quantity.gt(0)].to_dict("records"):
                quantity = int(ledger.loc[ledger.date.eq(row["date"]), "shares"].iloc[0])
                assert parent.state.risk_valid(e, quantity, row["actual_open"], row, h.COSTS["STRESS"])
            if policy == "EQUAL_EXPERTS":
                pd.testing.assert_frame_equal(ledger, pd.read_parquet(source.OUT / "accounts" / cost / f"{policy}_ledger.parquet"))
            print(f"当日可用 {cost}/{policy}：夏普{measurement['net_sharpe']}，年化{measurement['annualized_return']:.2%}，回撤{measurement['max_drawdown']:.2%}。", flush=True)
    from scipy.optimize import minimize
    checked = 0
    for record in allocations[::max(1, len(allocations) // 30)]:
        active = np.array(record["available_experts"], dtype=bool)
        mu, covariance = np.array(record["expert_mu5"]), np.array(record["regularized_covariance"])
        weights = np.array(record["weights"])
        if active.any():
            fitted = minimize(lambda w: -float(w @ mu - 2 * w @ covariance @ w), np.zeros(3),
                              jac=lambda w: -mu + 4 * covariance @ w, method="SLSQP",
                              bounds=[(0, 1) if allowed else (0, 0) for allowed in active],
                              constraints={"type": "ineq", "fun": lambda w: 1 - w.sum(), "jac": lambda w: -np.ones(3)},
                              options={"ftol": 1e-12, "maxiter": 200})
            assert fitted.success and abs(float(weights @ mu - 2 * weights @ covariance @ weights) + fitted.fun) < 1e-8
        else:
            assert not weights.any()
        checked += 1
    comparisons, rng = [], np.random.default_rng(20261002)
    for control, ledger in [("PARENT_MACRO_HMM", pd.read_parquet(source.OUT / "accounts/STRESS/MACRO_HMM_ledger.parquet")),
                            *[(policy, accounts[policy, "STRESS"]) for policy in ["PRICE_HMM", "HISTORY", "EQUAL_EXPERTS"]]]:
        actual = accounts[parent.PRIMARY, "STRESS"]
        pd.testing.assert_series_equal(actual.date, ledger.date)
        difference = actual.net_return.to_numpy(float) - ledger.net_return.to_numpy(float)
        n, samples = len(difference), []
        for _ in range(2000):
            starts = rng.integers(0, n, size=int(np.ceil(n / 20)))
            indices = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
            samples.append(float(difference[indices].mean() * 252))
        comparisons.append({"left": parent.PRIMARY, "right": control, "annual_arithmetic_difference": float(difference.mean() * 252),
                            "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)), "selection_adjusted": False})
    primary = next(row for row in measurements if row["policy"] == parent.PRIMARY and row["cost"] == "STRESS")
    verification = {"at": now(), "saved_optimizers_cross_checked": checked, "all_inactive_weights_zero": True,
                    "accounts_checked": 8, "unchanged_equal_accounts_exactly_reproduced": 2,
                    "sources_verified": True, "statistical_models_refitted": 0}
    save(root / "verification.json", verification, True)
    save(root / "results/account_metrics.json", measurements, True)
    save(root / "result.json", {"study_id": STUDY, "at": now(), "primary": primary, "all_accounts": measurements,
        "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_ACTIVE_STATE_ALLOCATION",
        "comparisons": comparisons, "verification": verification, "new_reference_accounts": 0,
        "new_full_accounts": 6, "recomputed_equal_accounts": 2, "new_optimizations": len(allocations),
        "new_statistical_fits": 0, "goal_achieved": False, "new_independent_observations": 0,
        "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    print("当日动作可用性比较完成，旧等待策略及全部对照保留。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="当日信号可用性对状态策略分配的固定增量")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
