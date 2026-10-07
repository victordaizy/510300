"""让持续状态条件化具体策略的收益，而不是直接预测指数涨跌。"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.index_persistent_state_daily_v1 as state
import research.index_state_inventory_daily_v1 as inventory
import research.selected_mix_daily_two_year_v1 as daily
import research.selected_mix_risk_before_band_v1 as execution
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, now, LATEST
from research.post_selection_continuous_accounts_v1 import simulate_policy
from research.simple_intraday_protection_v1 import make_rules
from research.simple_volume_reversal_v1 import make_rules as volume_rules
from research.adaptive_allocation_v1 import normalize_dividends

STUDY = "510300_INDEX_STATE_STRATEGY_VALUE_V1"
OUT = ROOT / "reports/research/510300_index_state_strategy_value_v1"
EXPERTS = ["D60_INTRA", "R2_Z_CONFIRM", "PANIC_ONLY"]
POLICIES = ["EQUAL_EXPERTS", "HISTORY", "PRICE_HMM", "MACRO_HMM"]
PRIMARY = "MACRO_HMM"


def allocation(mu, covariance):
    """解三个参考策略与现金的固定凹二次规划；枚举约束面不是策略参数搜索。"""
    covariance = .5 * covariance + .5 * np.diag(np.diag(covariance)) + np.eye(3) * 1e-10
    best, best_score = np.zeros(3), 0.
    for size in range(1, 4):
        for subset in combinations(range(3), size):
            ids = list(subset)
            c, m = covariance[np.ix_(ids, ids)], mu[ids]
            inverse_mu = np.linalg.solve(c, m)
            inverse_one = np.linalg.solve(c, np.ones(size))
            multiplier = (inverse_mu.sum() - 4.) / inverse_one.sum()
            candidates = [inverse_mu / 4., (inverse_mu - multiplier * inverse_one) / 4.]
            for candidate in candidates:
                if candidate.min() < -1e-10 or candidate.sum() > 1 + 1e-10:
                    continue
                weights = np.zeros(3)
                weights[ids] = np.maximum(candidate, 0)
                if weights.sum() > 1:
                    weights /= weights.sum()
                score = float(weights @ mu - 2 * weights @ covariance @ weights)
                if score > best_score + 1e-12:
                    best, best_score = weights, score
    return best, best_score, covariance


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("状态与策略价值研究已固定。")
    for name in ["code", "inputs", "references", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {
        "study_id": STUDY, "at": now(), "primary": PRIMARY,
        "question": "相同状态信息用于判断三类既有策略的收益，而非标的直接收益，是否能形成成本后账户优势。",
        "experts": EXPERTS, "expert_status": "三类固定价格参考，未被称为独立有效Alpha；不加入原学习退出或旧85/15模型。",
        "reference_clock": "按原价格规则和压力成本，从原2013参考起点连续重建；不因评价起点重置已有参考策略状态。",
        "value_label": "参考账户在当日开盘交易前权益至第五个后续开盘交易前权益的净收益。除去第一个开盘前的隔夜涨跌，不把入场前跳空计入可获得收益。参考过程内的合法成交和分红全部保留。",
        "information": "复用已保存两年日更HISTORY、PRICE_HMM、MACRO_HMM的相同成熟标签权重，隐藏状态不重新训练、不改状态参数。仅将被加权的结果换为具体策略的五日收益。",
        "allocation": "每天计算三策略条件均值与协方差，协方差固定50%收缩到对角并加1e-10数值对角项；最大化w均值-2*w协方差w，w>=0且总和<=1，余量为现金。无正效用则现金。",
        "actual_account": "将三个参考当天计划仓位按上述权重合成一个510300目标，之后统一尾部预算和10个百分点交易带；不是把三条净值相加。",
        "common_risk": "四个分配政策统一使用同一MACRO_HMM的五日指数尾部分布；比较只隔离状态对策略分配的作用。PRICE_HMM是价格信息分配对照，不是完全不含宏观风险信息的账户。",
        "policies": {"EQUAL_EXPERTS": "三类固定等权参考", "HISTORY": "不分状态的策略价值分配",
                     "PRICE_HMM": "价格状态条件策略价值分配", "MACRO_HMM": "宏观持续状态条件策略价值分配"},
        "risk": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "period": ["2021-01-04", "2026-08-14"], "capital": 200000, "annual_days": 252,
        "targets": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "new_reference_accounts": 3, "new_full_accounts": 8, "new_hidden_models": 0,
        "comparison": "主MACRO_HMM对PRICE_HMM、HISTORY及EQUAL_EXPERTS；完整同日账户20日区块2000次，种子20261001。",
        "duplicate_boundary": "区别于直接预测指数涨跌，以及过去仅按总体均值/风险配置的旧参考分配；本次固定检验持续状态与各具体策略未来收益的关系。",
        "new_parameter_grid": 0, "new_market_collection": False, "goal_achieved": False, "orders_authorized": False,
        "evidence_limit": "同一段历史反复研究后的新固定开发比较，不是独立验证。"}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update(ROOT / name for name in read(state.OUT / "freeze.json")["sources"])
    sources.update([state.OUT / "results/training_weights.parquet", state.OUT / "results/predictions.parquet",
                    LATEST / "candidate_features.parquet", ROOT / "data/reference/510300_dividends.csv",
                    ROOT / "config/510300_learned_cycle_exit_v1.json", ROOT / "config/510300_continuous_reference_min_variance_v1.json"])
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
                                "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("状态条件策略价值实验已固定，三个参考合成一个实际指数账户。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def references(root, data, dividends):
    cfg = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    panic = read(ROOT / "config/510300_continuous_reference_min_variance_v1.json")
    price_rules = make_rules(data)
    price_rules["PANIC_ONLY"] = volume_rules(data)[0]["V6_PANIC_RECOVERY"]
    targets = pd.DataFrame({"idx": np.arange(len(data)), "date": data.date})
    labels = pd.DataFrame({"idx": np.arange(len(data) - 5), "date": data.date.iloc[:-5].to_numpy(),
                           "exit_idx": np.arange(5, len(data)), "exit_date": data.date.iloc[5:].to_numpy()})
    for expert in EXPERTS:
        spec = panic["panic_spec"] if expert == "PANIC_ONLY" else cfg["candidate_specs"][expert]
        ledger, decisions, cycles, checkpoint = simulate_policy(data, dividends, cfg, cfg["costs"]["STRESS"],
            cfg["reference_start"], price_rules[expert], spec, next_execution_date="2026-08-17")
        folder = root / "references" / expert
        folder.mkdir()
        ledger.to_parquet(folder / "ledger.parquet", index=False)
        decisions.to_parquet(folder / "decisions.parquet", index=False)
        cycles.to_parquet(folder / "cycles.parquet", index=False)
        save(folder / "checkpoint.json", checkpoint, True)
        # 收盘权益逆推至当日交易前开盘权益，包含已确认股息但不包含当日买卖费用。
        nav_open = ledger.equity - ledger.shares * (ledger.mark - ledger.open) + ledger.commission + ledger.slippage_cost
        assert np.isfinite(nav_open).all() and nav_open.gt(0).all()
        global_open = pd.Series(np.nan, index=np.arange(len(data)))
        indices = data.date.searchsorted(ledger.date)
        global_open.iloc[indices] = nav_open.to_numpy(float)
        labels[expert] = global_open.shift(-5).iloc[:-5].to_numpy() / global_open.iloc[:-5].to_numpy() - 1
        target = np.full(len(data), np.nan)
        decision_index = decisions.origin_index.to_numpy(int)
        usable = decision_index + 1 < len(data)
        target[decision_index[usable] + 1] = decisions.reference_weight.to_numpy(float)[usable]
        targets[expert] = target
        print(f"压力参考{expert}完成，开盘权益标签排除入场前跳空。", flush=True)
    labels.to_parquet(root / "results/expert_value_labels.parquet", index=False)
    targets.to_parquet(root / "results/expert_current_targets.parquet", index=False)
    return labels.set_index("idx", drop=False), targets.set_index("idx", drop=False)


def learn_allocations(root, labels, targets):
    source = pd.read_parquet(state.OUT / "results/predictions.parquet")
    risk = source.loc[source.model.eq("MACRO_HMM")].set_index("idx")
    memberships = pd.read_parquet(state.OUT / "results/training_weights.parquet")
    memberships = memberships.loc[memberships.model.isin(POLICIES)]
    predictions, records = [], []
    for (idx, policy), group in memberships.groupby(["idx", "model"], sort=False):
        tail = risk.loc[idx]
        rows = labels.loc[group.training_idx]
        assert rows.date.ge(tail.lower_bound).all() and rows.exit_date.lt(tail.date).all()
        returns, probability = rows[EXPERTS].to_numpy(float), group.weight.to_numpy(float)
        assert np.isfinite(returns).all() and abs(probability.sum() - 1) < 1e-10
        mu = probability @ returns
        centered = returns - mu
        covariance = (centered * probability[:, None]).T @ centered / (1 - probability @ probability)
        weights, utility, regularized = allocation(mu, covariance)
        expert_target = targets.loc[idx, EXPERTS].to_numpy(float)
        assert np.isfinite(expert_target).all()
        raw = float(weights @ expert_target)
        predictions.append({**tail.to_dict(), "idx": int(idx), "model": policy, "raw_target": raw})
        records.append({"idx": int(idx), "date": tail.date, "policy": policy, "training_rows": len(rows),
                        "latest_exit": rows.exit_date.max(), "lower_bound": tail.lower_bound,
                        "expert_mu5": mu.tolist(), "expert_covariance": covariance.tolist(),
                        "regularized_covariance": regularized.tolist(), "weights": weights.tolist(),
                        "cash_weight": float(1 - weights.sum()), "utility": utility,
                        "expert_current_targets": expert_target.tolist(), "raw_target": raw})
    for idx, tail in risk.iterrows():
        raw = float(targets.loc[idx, EXPERTS].mean())
        predictions.append({**tail.to_dict(), "idx": int(idx), "model": "EQUAL_EXPERTS", "raw_target": raw})
    frame = pd.DataFrame(predictions).sort_values(["idx", "model"]).reset_index(drop=True)
    frame.to_parquet(root / "results/allocation_predictions.parquet", index=False)
    save(root / "results/saved_allocations.json", records, True)
    return frame, records


def account_plan(h, e, account, ref, peak, forecast, pressure):
    nav = account.value(ref)
    chosen = execution.fixed_choice(account.shares, nav, peak, ref, float(forecast.raw_target), False,
                                    float(forecast.es95), h.COSTS["STRESS"], .001, 100)
    q = chosen["target_shares"]
    return {**chosen, **h.risk_limits(nav, peak, forecast.es95), "risk_shares_before": account.shares,
            "requested_quantity": q - account.shares, "planned_weight": q * ref / nav,
            "reference_price": ref, "decision_equity": nav, "allocation_source_target": forecast.raw_target}


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    h, e = inventory.modules(state.OUT)
    d, x, frame, dividends = inventory.inputs(state.OUT, e)
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(d.date.iloc[-1])].reset_index(drop=True)
    np.testing.assert_array_equal(data.date.to_numpy(dtype="datetime64[ns]"), d.date.to_numpy(dtype="datetime64[ns]"))
    for column in ["open", "high", "low", "close"]:
        np.testing.assert_array_equal(data[column], d[column])
    labels, targets = references(root, data, dividends)
    predictions, allocations = learn_allocations(root, labels, targets)
    proxy = SimpleNamespace(**vars(h))
    proxy.plan = lambda engine, account, ref, peak, forecast, pressure: account_plan(h, engine, account, ref, peak, forecast, pressure)
    proxy.risk_valid = state.risk_valid
    accounts, measurements, yearly = {}, [], []
    for cost in h.COSTS:
        folder = root / "accounts" / cost
        folder.mkdir()
        for policy in POLICIES:
            ledger, decisions = inventory.simulate(proxy, e, d, x, dividends, predictions, policy, cost)
            ledger.to_parquet(folder / f"{policy}_ledger.parquet", index=False)
            decisions.to_parquet(folder / f"{policy}_decisions.parquet", index=False)
            accounts[policy, cost] = ledger
            m = {"policy": policy, "cost": cost, **h.summary(e, ledger)}
            m["historical_point_targets_met"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["annualized_return"] >= .1 and m["max_drawdown"] >= -.1
            measurements.append(m)
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"policy": policy, "cost": cost, "year": int(year), **e.return_metrics(group.net_return, 252)})
            assert ledger.accounting_error.abs().max() < 1e-6 and ledger.cash.ge(-1e-7).all()
            assert (-ledger.filled_quantity.clip(upper=0)).le(ledger.sellable_before).all()
            for row in decisions.loc[decisions.filled_quantity.gt(0)].to_dict("records"):
                quantity = int(ledger.loc[ledger.date.eq(row["date"]), "shares"].iloc[0])
                assert state.risk_valid(e, quantity, row["actual_open"], row, h.COSTS["STRESS"])
            print(f"状态策略价值 {cost}/{policy}：夏普{m['net_sharpe']}，年化{m['annualized_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    save(root / "results/account_metrics.json", measurements, True)
    save(root / "results/yearly_metrics.json", yearly, True)
    comparisons = []
    rng = np.random.default_rng(20261001)
    for control in ["PRICE_HMM", "HISTORY", "EQUAL_EXPERTS"]:
        a, b = accounts[PRIMARY, "STRESS"], accounts[control, "STRESS"]
        pd.testing.assert_series_equal(a.date, b.date)
        values = a.net_return.to_numpy(float) - b.net_return.to_numpy(float)
        n, samples = len(values), []
        for _ in range(2000):
            starts = rng.integers(0, n, size=int(np.ceil(n / 20)))
            indices = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
            samples.append(float(values[indices].mean() * 252))
        comparisons.append({"left": PRIMARY, "right": control, "annual_arithmetic_difference": float(values.mean() * 252),
                            "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)),
                            "selection_adjusted": False})
    # 独立求解器只核对固定优化问题，不产生或挑选另一组账户。
    from scipy.optimize import minimize
    checked = 0
    for record in read(root / "results/saved_allocations.json")[::max(1, len(allocations) // 15)]:
        mu, covariance = np.array(record["expert_mu5"]), np.array(record["regularized_covariance"])
        expected = np.array(record["weights"])
        fitted = minimize(lambda w: -float(w @ mu - 2 * w @ covariance @ w), np.zeros(3),
                          jac=lambda w: -mu + 4 * covariance @ w, method="SLSQP", bounds=[(0, 1)] * 3,
                          constraints={"type": "ineq", "fun": lambda w: 1 - w.sum(), "jac": lambda w: -np.ones(3)},
                          options={"ftol": 1e-12, "maxiter": 200})
        assert fitted.success
        score = float(expected @ mu - 2 * expected @ covariance @ expected)
        assert abs(score + fitted.fun) < 1e-8
        checked += 1
    verification = {"at": now(), "saved_optimizers_cross_checked": checked, "accounts_checked": 8,
                    "training_maturity_and_two_year_window_checked": len(allocations),
                    "exact_common_market_prices": True, "reference_open_labels_exclude_first_gap": True}
    save(root / "verification.json", verification, True)
    primary = next(row for row in measurements if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    save(root / "result.json", {"study_id": STUDY, "at": now(), "primary": primary, "all_accounts": measurements,
        "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_STATE_STRATEGY_VALUE",
        "comparisons": comparisons, "verification": verification, "new_reference_accounts": 3, "new_full_accounts": 8,
        "daily_allocation_fits": len(allocations), "new_hidden_model_fits": 0,
        "goal_achieved": False, "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    print("状态条件策略价值实验完成，所有对照和实际账户已保存。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="状态与固定价格策略收益关系的每日学习")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "研究尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
