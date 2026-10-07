"""纯指数的两年每日持续状态分布：固定双状态模型，比较宏观增量与状态记忆。"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.index_state_inventory_daily_v1 as previous
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, now

OUT = ROOT / "reports/research/510300_index_persistent_state_daily_v1"
PARENT = previous.OUT
STUDY = "510300_INDEX_PERSISTENT_STATE_DAILY_V1"
POLICIES = ["HISTORY", "PRICE_HMM", "MACRO_EMISSION", "MACRO_HMM"]
PRIMARY = "MACRO_HMM"
FEATURES = {"PRICE": previous.PRICE, "MACRO": previous.MACRO}
ITERATIONS = 40
VARIANCE_FLOOR = .05
PSEUDOCOUNT = 1.
BATCH_SIZE = 64


def emissions(z, means, variance):
    residual = z[:, :, None, :] - means[:, None, :, :]
    logs = -.5 * (np.log(2 * np.pi * variance[:, None, :, :]) + residual * residual / variance[:, None, :, :]).sum(axis=-1)
    shifts = logs.max(axis=2)
    density = np.maximum(np.exp(logs - shifts[:, :, None]), 1e-250)
    return density, shifts


def forward_backward(z, valid, initial, transition, means, variance):
    """沿时间递推、沿独立窗口并行；填充行不产生观察或转移。"""
    b, shifts = emissions(z, means, variance)
    batch, length, _ = b.shape
    alpha = np.empty_like(b)
    scales = np.ones((batch, length))
    prior = initial.copy()
    for t in range(length):
        if t:
            prior = np.einsum("bi,bij->bj", alpha[:, t - 1], transition)
        weighted = prior * b[:, t]
        denom = weighted.sum(axis=1)
        posterior = weighted / denom[:, None]
        alpha[:, t] = np.where(valid[:, t, None], posterior, alpha[:, t - 1] if t else initial)
        scales[:, t] = np.where(valid[:, t], denom, 1.)
    beta = np.ones_like(b)
    transitions = np.zeros_like(transition)
    for t in range(length - 2, -1, -1):
        next_term = b[:, t + 1] * beta[:, t + 1]
        candidate = np.einsum("bij,bj->bi", transition, next_term) / scales[:, t + 1, None]
        beta[:, t] = np.where(valid[:, t + 1, None], candidate, 1.)
        xi = alpha[:, t, :, None] * transition * next_term[:, None, :] / scales[:, t + 1, None, None]
        transitions += xi * valid[:, t + 1, None, None]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    gamma *= valid[:, :, None]
    likelihood = ((np.log(scales) + shifts) * valid).sum(axis=1)
    return likelihood, alpha, gamma, transitions


def fit_batch(windows, columns):
    counts = np.array([len(w) for w in windows], dtype=int)
    batch, length, dimensions = len(windows), int(counts.max()), len(columns)
    z = np.zeros((batch, length, dimensions))
    valid = np.arange(length)[None, :] < counts[:, None]
    means, variance = np.zeros((batch, 2, dimensions)), np.ones((batch, 2, dimensions))
    centers, scales = np.zeros((batch, dimensions)), np.ones((batch, dimensions))
    initial = np.empty((batch, 2))
    transition = np.empty((batch, 2, 2))
    for k, window in enumerate(windows):
        raw = window[columns].to_numpy(float)
        centers[k], scales[k] = raw.mean(axis=0), raw.std(axis=0, ddof=1)
        scales[k] = np.where(scales[k] > 1e-12, scales[k], 1.)
        values = np.clip((raw - centers[k]) / scales[k], -5, 5)
        z[k, :len(window)] = values
        # 仅按已知二十日趋势的中位数初始化，不读取任何后续收益。
        trend = values[:, columns.index("trend20")]
        states = (trend >= np.median(trend)).astype(int)
        if len(np.unique(states)) != 2:
            states = (np.arange(len(states)) >= len(states) // 2).astype(int)
        for state in range(2):
            selected = values[states == state]
            means[k, state] = selected.mean(axis=0)
            variance[k, state] = np.maximum(selected.var(axis=0), VARIANCE_FLOOR)
        initial[k] = (np.bincount(states, minlength=2) + PSEUDOCOUNT) / (len(states) + 2 * PSEUDOCOUNT)
        count = np.full((2, 2), PSEUDOCOUNT)
        np.add.at(count, (states[:-1], states[1:]), 1.)
        transition[k] = count / count.sum(axis=1, keepdims=True)
    initial_likelihood = None
    for iteration in range(ITERATIONS):
        ll, _, gamma, counts_transition = forward_backward(z, valid, initial, transition, means, variance)
        if iteration == 0:
            initial_likelihood = ll.copy()
        mass = gamma.sum(axis=1)
        assert (mass > 1e-10).all()
        initial = np.maximum(gamma[:, 0], 1e-12)
        initial /= initial.sum(axis=1, keepdims=True)
        transition = counts_transition + PSEUDOCOUNT
        transition /= transition.sum(axis=2, keepdims=True)
        means = np.einsum("btk,btd->bkd", gamma, z) / mass[:, :, None]
        second = np.einsum("btk,btd->bkd", gamma, z * z) / mass[:, :, None]
        variance = np.maximum(second - means * means, VARIANCE_FLOOR)
    ll, alpha, gamma, _ = forward_backward(z, valid, initial, transition, means, variance)
    density, _ = emissions(z, means, variance)
    stationary = np.stack([transition[:, 1, 0], transition[:, 0, 1]], axis=1)
    stationary /= stationary.sum(axis=1, keepdims=True)
    independent = density * stationary[:, None, :]
    independent /= independent.sum(axis=2, keepdims=True)
    result = []
    for k, count in enumerate(counts):
        result.append({"features": list(columns), "standard_mean": centers[k].tolist(), "standard_scale": scales[k].tolist(),
                       "initial": initial[k].tolist(), "transition": transition[k].tolist(),
                       "means": means[k].tolist(), "variance": variance[k].tolist(),
                       "initial_log_likelihood": float(initial_likelihood[k]), "final_log_likelihood": float(ll[k]),
                       "smoothed_state_mass": gamma[k, :count].sum(axis=0).tolist(),
                       "filtered": alpha[k, :count], "independent": independent[k, :count]})
    return result


def weighted_statistics(y, weights):
    y, weights = np.asarray(y, float), np.asarray(weights, float)
    weights = weights / weights.sum()
    effective = 1 / float(weights @ weights)
    fallback = effective < 126
    if fallback:
        weights = np.full(len(y), 1 / len(y))
    mu = float(weights @ y)
    variance = float(weights @ ((y - mu) ** 2) / (1 - weights @ weights))
    order = np.argsort(y, kind="stable")
    sorted_y, sorted_w = y[order], weights[order]
    cumulative = sorted_w.cumsum()
    q05 = float(sorted_y[min(int(np.searchsorted(cumulative, .05)), len(y) - 1)])
    included = np.clip(.05 - np.r_[0., cumulative[:-1]], 0., sorted_w)
    es = max(0., -float(included @ sorted_y / .05))
    return {"mu5": mu, "variance5": variance, "q05": q05, "es95": es,
            "win_probability": float(weights[y > 0].sum()), "weight_effective_rows_before_fallback": effective,
            "weight_effective_rows": 1 / float(weights @ weights), "distribution_fallback": fallback}, weights


def predict_window(window, model, memory):
    day = window.date.iloc[-1]
    mature = window.exit_date.lt(day) & window.gross_return5.notna()
    probabilities = model["filtered"] if memory else model["independent"]
    historical = probabilities[mature.to_numpy()]
    current = probabilities[-1]
    mass = historical.sum(axis=0)
    assert (mass > 1e-12).all()
    weights = (historical / mass[None, :] * current[None, :]).sum(axis=1)
    stats, used = weighted_statistics(window.loc[mature, "gross_return5"].to_numpy(float), weights)
    return stats, used, current, mature


def mechanics():
    z = np.array([[[.2], [-.6], [.8], [.1]], [[.2], [-.6], [0.], [0.]]])
    valid = np.array([[True] * 4, [True, True, False, False]])
    pi = np.array([[.4, .6]] * 2)
    transition = np.array([[[.8, .2], [.1, .9]]] * 2)
    means = np.array([[[-.4], [.7]]] * 2)
    variance = np.array([[[.5], [.9]]] * 2)
    ll, alpha, gamma, xi = forward_backward(z, valid, pi, transition, means, variance)
    for k, length in enumerate([4, 2]):
        total, occupancy, count = 0., np.zeros((length, 2)), np.zeros((2, 2))
        for states in product(range(2), repeat=length):
            probability = pi[k, states[0]]
            for t, state in enumerate(states):
                v = variance[k, state, 0]
                probability *= np.exp(-.5 * (z[k, t, 0] - means[k, state, 0]) ** 2 / v) / np.sqrt(2 * np.pi * v)
                if t:
                    probability *= transition[k, states[t - 1], state]
            total += probability
            for t, state in enumerate(states):
                occupancy[t, state] += probability
                if t:
                    count[states[t - 1], state] += probability
        np.testing.assert_allclose(ll[k], np.log(total), atol=1e-12)
        np.testing.assert_allclose(gamma[k, :length], occupancy / total, atol=1e-12)
        np.testing.assert_allclose(xi[k], count / total, atol=1e-12)
    np.testing.assert_allclose(alpha[0, :2], alpha[1, :2], atol=1e-12)
    stats, _ = weighted_statistics(np.r_[-.2, np.zeros(199)], np.full(200, .005))
    assert abs(stats["es95"] - .02) < 1e-12
    return {"state_path_enumeration_likelihood_and_smoothing": True, "padding_adds_no_transitions": True,
            "filter_does_not_read_later_observations": True, "exact_weighted_five_percent_tail": True}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("持续状态实验已经固定。")
    checked = mechanics()
    for name in ["code", "inputs", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    files = {"inputs/market.parquet": PARENT / "inputs/market.parquet",
             "inputs/decision_information.parquet": PARENT / "results/decision_information.parquet",
             "inputs/mature_labels.parquet": PARENT / "inputs/mature_labels.parquet",
             "inputs/dividends.csv": PARENT / "inputs/dividends.csv",
             "code/account_engine.py": PARENT / "code/account_engine.py",
             "code/distribution_primitives.py": PARENT / "code/distribution_primitives.py"}
    for dest, source in files.items():
        shutil.copy2(source, root / dest)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"] and mandate["training_window_calendar_years"] == 2
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {
        "study_id": STUDY, "at": now(), "primary": PRIMARY, "assets": ["510300.SH", "CASH_CNY"],
        "question": "两年每日更新的持续状态能否改善纯指数条件收益分布，宏观信息与状态记忆是否有实际账户增量。",
        "mechanism": "相似当前价格可来自不同的持续状态；用历史状态概率及其转移更新今天的条件分布。不能把隐藏状态名称当成已知牛熊或真实资金意图。",
        "features": FEATURES, "macro_definition": "社融存量同比的三月变化及DR007减七日逆回购政策利率，沿用保存可用时钟；不是NFCI，也不是名义利率减中性利率。",
        "period": ["2021-01-04", "2026-08-14"], "capital": 200000, "annual_days": 252,
        "targets": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "training": "每个09:00决策只使用当日向前两日历年内的已知共同观察，包括今天已知特征；收益标签退出日必须严格早于今天，最少252行。共同观察若不连续，当天NO_VIEW。",
        "hidden_model": {"states": 2, "covariance": "diagonal", "EM_iterations": ITERATIONS,
                         "initialization": "同窗二十日趋势中位数单次确定初始化", "variance_floor_standardized": VARIANCE_FLOOR,
                         "transition_pseudocount_per_cell": PSEUDOCOUNT, "standardization": "同窗均值及样本标准差、裁剪5",
                         "no_previous_window_warm_start": True, "batch_dimension": "仅并行计算互不共享参数的每日窗口"},
        "policies": {"HISTORY": "共同成熟标签等权经验分布", "PRICE_HMM": "仅三价格变量的双状态模型和前向过滤概率",
                     "MACRO_EMISSION": "与MACRO_HMM相同拟合模型，使用平稳概率乘当日发射似然，不继承昨日状态概率",
                     PRIMARY: "五变量双状态模型，沿时间前向过滤；以当前状态概率混合各状态的成熟五日收益分布"},
        "distribution": "状态历史样本仅按前向过滤概率加权，不按看过后来观察的平滑后验分组。有效加权行数小于126则回到同窗等权分布。精确取下侧5%概率质量计算ES。",
        "cashflow": "五日开盘至开盘含新增登记股息；原20万元库存效用和T+1引擎。目标50%、五日ES2.5%、10%跳空5%及回撤余量；本轮明确将调仓费一并计入风险预算。",
        "entry_gap": "09:00确定方向与最多份额，09:30开盘只允许因风险预算缩小买入；不足现金和T+1仍由原引擎处理。",
        "comparison": "MACRO_HMM对PRICE_HMM为宏观增量，MACRO_HMM对MACRO_EMISSION为状态记忆增量；与HISTORY比较整个条件化过程。20日区块2000次，种子20260930，全部账户日。",
        "duplicate_boundary": ["旧HMM只在2012-2014拟合后冻结、三状态牛熊映射满仓或空仓；本次是两年每日估计的条件完整收益分布，不使用旧参数或状态序列。",
                               "旧两年日更浅树仅按当前截面特征分叶；本次明确检验历史状态过滤的额外作用，同日发射对照用于识别记忆影响。"],
        "account_count": 8, "parameter_search": False, "mechanism_checks": checked,
        "new_market_collection": False, "orders_authorized": False, "goal_achieved": False,
        "evidence_class": "此前已研究历史上的固定开发实验，没有独立留出"}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)}) | set(files.values())
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
                                "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)},
                                "copied_files": {dest: digest(root / dest) for dest in files}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("双状态持续性实验已固定：两年每日拟合、价格及宏观对照、相同账户尾部预算。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)
    for name, expected in record["copied_files"].items():
        assert digest(root / name) == expected


def risk_valid(e, quantity, ref, limits, cost):
    if quantity == 0:
        return True
    old = int(limits["risk_shares_before"])
    change = quantity - old
    change_px = e.fill_price(ref, 1 if change > 0 else -1, cost, .001)
    change_cost = abs(change) * abs(change_px - ref) + e.commission(change, change_px, cost)
    exit_px = e.fill_price(ref, -1, cost, .001)
    exit_cost = quantity * (ref - exit_px) + e.commission(quantity, exit_px, cost)
    notional = quantity * ref
    return (notional <= limits["position_budget_cny"] - .5 * change_cost + 1e-8
            and notional * limits["es95"] + exit_cost + change_cost <= limits["es_budget_cny"] + 1e-8
            and .1 * notional + exit_cost + change_cost <= limits["gap_budget_cny"] + 1e-8)


def strict_plan(h, e, account, ref, peak, prediction, pressure):
    nav = account.value(ref)
    limits = {**h.risk_limits(nav, peak, prediction.es95), "risk_shares_before": account.shares}
    cost = h.COSTS["STRESS"]
    maximum = int(limits["position_budget_cny"] / ref / 100) * 100
    maximum = min(maximum, account.shares + e.affordable_quantity(account.cash, e.fill_price(ref, 1, cost, .001), cost, 100))
    best = None
    for target in range(0, maximum + 1, 100):
        if not risk_valid(e, target, ref, limits, cost):
            continue
        change = target - account.shares
        px = e.fill_price(ref, 1 if change > 0 else -1, cost, .001)
        adjust = abs(change) * abs(px - ref) + e.commission(change, px, cost)
        exit_px = e.fill_price(ref, -1, cost, .001)
        exit_cost = target * (ref - exit_px) + e.commission(target, exit_px, cost)
        weight = target * ref / nav
        score = weight * prediction.mu5 - 2 * weight * weight * prediction.variance5 - (adjust + exit_cost) / nav
        row = {"target_shares": target, "requested_quantity": change, "planned_weight": weight, "score": score,
               "estimated_adjustment_cost": adjust, "reserved_exit_cost": exit_cost, "reference_price": ref,
               "decision_equity": nav, "mu5": prediction.mu5, "variance5": prediction.variance5, **limits}
        if best is None or score > best["score"] + 1e-12:
            best = row
    assert best is not None
    return best


def learn(root, frame, h):
    windows, receipts = [], []
    for current in frame.loc[frame.date.ge(h.START)].itertuples():
        lower = current.date - pd.DateOffset(years=2)
        window = frame.loc[frame.date.ge(lower) & frame.date.le(current.date)].copy()
        mature = window.exit_date.lt(current.date) & window.gross_return5.notna() & window.common_known
        usable = bool(window.common_known.all() and mature.sum() >= 252 and window.idx.diff().dropna().eq(1).all())
        receipts.append({"idx": current.idx, "date": current.date, "lower_bound": lower, "observation_rows": len(window),
                         "mature_rows": int(mature.sum()), "status": "UPDATED" if usable else "NO_VIEW"})
        if usable:
            windows.append(window)
    predictions, models, memberships = [], [], []
    for offset in range(0, len(windows), BATCH_SIZE):
        batch = windows[offset:offset + BATCH_SIZE]
        fitted = {name: fit_batch(batch, columns) for name, columns in FEATURES.items()}
        for k, window in enumerate(batch):
            day, idx = window.date.iloc[-1], int(window.idx.iloc[-1])
            mature = window.exit_date.lt(day) & window.gross_return5.notna()
            y = window.loc[mature, "gross_return5"].to_numpy(float)
            common = {"idx": idx, "date": day, "lower_bound": day - pd.DateOffset(years=2),
                      "training_count": len(y), "latest_training_exit": window.loc[mature, "exit_date"].max()}
            for name in FEATURES:
                model = fitted[name][k]
                models.append({**common, "family": name, "observation_indices": window.idx.to_list(),
                               "training_indices": window.loc[mature, "idx"].to_list(),
                               **{key: value for key, value in model.items() if key not in {"filtered", "independent"}}})
            for policy in POLICIES:
                if policy == "HISTORY":
                    stats, weights = weighted_statistics(y, np.ones(len(y)))
                    state = np.array([np.nan, np.nan])
                else:
                    family = "PRICE" if policy == "PRICE_HMM" else "MACRO"
                    stats, weights, state, _ = predict_window(window, fitted[family][k], policy != "MACRO_EMISSION")
                predictions.append({**common, "model": policy, "state_probability_0": state[0],
                                    "state_probability_1": state[1], **stats})
                memberships.extend({"idx": idx, "model": policy, "training_idx": int(train_idx), "weight": float(weight)}
                                   for train_idx, weight in zip(window.loc[mature, "idx"], weights))
        print(f"持续状态每日模型完成：{min(offset + BATCH_SIZE, len(windows))}/{len(windows)}个决策日。", flush=True)
    result = pd.DataFrame(predictions)
    result.to_parquet(root / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/update_receipts.parquet", index=False)
    pd.DataFrame(memberships).to_parquet(root / "results/training_weights.parquet", index=False)
    save(root / "results/saved_models.json", models, True)
    return result, models


def verify(root, frame, predictions, models, h, e):
    indexed = frame.set_index("idx", drop=False)
    weights = pd.read_parquet(root / "results/training_weights.parquet")
    lookup = predictions.set_index(["idx", "model"])
    count = 0
    for (idx, policy), group in weights.groupby(["idx", "model"], sort=False):
        rows = indexed.loc[group.training_idx]
        origin = lookup.loc[(idx, policy)]
        assert rows.date.ge(origin.lower_bound).all() and rows.exit_date.lt(origin.date).all()
        assert abs(group.weight.sum() - 1) < 1e-10 and group.weight.gt(0).all()
        stats, _ = weighted_statistics(rows.gross_return5, group.weight)
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(stats[key], origin[key], atol=1e-11, rtol=0)
        count += 1
    filter_checks = 0
    for model in models[::max(1, len(models) // 12)]:
        rows = indexed.loc[model["observation_indices"]]
        assert rows.date.ge(pd.Timestamp(model["lower_bound"])).all() and rows.date.le(pd.Timestamp(model["date"])).all()
        z = np.clip((rows[model["features"]].to_numpy(float) - model["standard_mean"]) / model["standard_scale"], -5, 5)[None]
        valid = np.ones(z.shape[:2], bool)
        _, filtered, _, _ = forward_backward(z, valid, np.array(model["initial"])[None],
            np.array(model["transition"])[None], np.array(model["means"])[None], np.array(model["variance"])[None])
        p = lookup.loc[(model["idx"], model["family"] + "_HMM")]
        np.testing.assert_allclose(filtered[0, -1], [p.state_probability_0, p.state_probability_1], atol=1e-11)
        filter_checks += 1
    perturbations = 0
    for date in ["2022-06-30", "2024-09-30", "2026-08-14"]:
        window = frame.loc[frame.date.ge(pd.Timestamp(date) - pd.DateOffset(years=2)) & frame.date.le(date)].copy()
        original = fit_batch([window], FEATURES["MACRO"])[0]
        changed = window.copy()
        changed.loc[changed.exit_date.ge(pd.Timestamp(date)) | changed.exit_date.isna(), "gross_return5"] = -999.
        a = predict_window(window, original, True)[0]
        b = predict_window(changed, original, True)[0]
        assert a == b
        perturbations += 1
    for cost in h.COSTS:
        for policy in POLICIES:
            ledger = pd.read_parquet(root / "accounts" / cost / f"{policy}_ledger.parquet")
            decisions = pd.read_parquet(root / "accounts" / cost / f"{policy}_decisions.parquet")
            np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable - ledger.terminal_exit_reserve, atol=1e-6)
            assert ledger.accounting_error.abs().max() < 1e-6 and ledger.cash.ge(-1e-7).all()
            assert (-ledger.filled_quantity.clip(upper=0)).le(ledger.sellable_before).all()
            for row in decisions.loc[decisions.filled_quantity.gt(0)].to_dict("records"):
                actual = int(ledger.loc[ledger.date.eq(row["date"]), "shares"].iloc[0])
                assert risk_valid(e, actual, row["actual_open"], row, h.COSTS["STRESS"])
    receipt = {"at": now(), "saved_weighted_distributions_recomputed": count, "saved_state_filters_recomputed": filter_checks,
               "immature_label_perturbations": perturbations, "full_accounts_checked": 8, "mechanism_checks": mechanics()}
    save(root / "verification.json", receipt, True)
    return receipt


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    h, e = previous.modules(root)
    d, x, frame, dividends = previous.inputs(root, e)
    predictions, models = learn(root, frame, h)
    # 复用原账户引擎，只补齐事前风险检查的调仓成本，四政策使用同一实现。
    proxy = SimpleNamespace(**vars(h))
    proxy.plan = lambda engine, account, ref, peak, forecast, pressure: strict_plan(h, engine, account, ref, peak, forecast, pressure)
    proxy.risk_valid = risk_valid
    accounts, measurements, yearly, rolling = {}, [], [], []
    for cost in h.COSTS:
        folder = root / "accounts" / cost
        folder.mkdir()
        for policy in POLICIES:
            ledger, decisions = previous.simulate(proxy, e, d, x, dividends, predictions, policy, cost)
            ledger.to_parquet(folder / f"{policy}_ledger.parquet", index=False)
            decisions.to_parquet(folder / f"{policy}_decisions.parquet", index=False)
            accounts[policy, cost] = ledger
            measure = {"policy": policy, "cost": cost, **h.summary(e, ledger)}
            measure["historical_point_targets_met"] = (measure["net_sharpe"] is not None and measure["net_sharpe"] >= 1.2
                and measure["annualized_return"] >= .1 and measure["max_drawdown"] >= -.1)
            measurements.append(measure)
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"policy": policy, "cost": cost, "year": int(year), **e.return_metrics(group.net_return, 252)})
            dates = pd.DatetimeIndex(ledger.date)
            if cost == "STRESS":
                for i, start in enumerate(dates):
                    end = int(dates.searchsorted(start + pd.DateOffset(years=2)))
                    if end >= len(dates):
                        break
                    m = e.return_metrics(ledger.net_return.iloc[i:end + 1], 252)
                    rolling.append({"policy": policy, "start": start, "end": dates[end], **m,
                                    "joint_pass": m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["annualized_return"] >= .1 and m["max_drawdown"] >= -.1})
            print(f"持续状态 {cost}/{policy}：夏普{measure['net_sharpe']}，年化{measure['annualized_return']:.2%}，回撤{measure['max_drawdown']:.2%}。", flush=True)
    save(root / "results/account_metrics.json", measurements, True)
    save(root / "results/yearly_metrics.json", yearly, True)
    pd.DataFrame(rolling).to_parquet(root / "results/rolling_two_years.parquet", index=False)
    scored = predictions.merge(frame[["idx", "exit_date", "gross_return5"]], on="idx", validate="many_to_one")
    scored["squared_error"] = (scored.gross_return5 - scored.mu5) ** 2
    scored["tail_breach"] = scored.gross_return5 < scored.q05
    scored.to_parquet(root / "results/scored_predictions.parquet", index=False)
    score_summary = [{"model": policy, "mature_predictions": int(rows.gross_return5.notna().sum()),
                      "MSE": float(rows.squared_error.mean()),
                      "tail_breach_rate": float(rows.loc[rows.gross_return5.notna(), "tail_breach"].mean())}
                     for policy, rows in scored.groupby("model")]
    comparisons = []
    rng = np.random.default_rng(20260930)
    for control in ["PRICE_HMM", "MACRO_EMISSION", "HISTORY"]:
        a, b = accounts[PRIMARY, "STRESS"], accounts[control, "STRESS"]
        pd.testing.assert_series_equal(a.date, b.date)
        difference = a.net_return.to_numpy(float) - b.net_return.to_numpy(float)
        means, n = [], len(difference)
        for _ in range(2000):
            starts = rng.integers(0, n, size=int(np.ceil(n / 20)))
            indices = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
            means.append(float(difference[indices].mean() * 252))
        comparisons.append({"left": PRIMARY, "right": control, "annual_arithmetic_difference": float(difference.mean() * 252),
                            "lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975)),
                            "selection_adjusted": False})
    monitor = scored.loc[scored.gross_return5.notna() & ((scored.idx - int(predictions.idx.min())) % 5).eq(0)].copy()
    monitor = monitor.sort_values(["model", "idx"])
    monitor["breach_rate60"] = monitor.groupby("model").tail_breach.transform(lambda s: s.rolling(60, min_periods=60).mean())
    monitor["observation_alert"] = monitor.breach_rate60.gt(.15)
    monitor.to_parquet(root / "results/nonoverlap_tail_monitor.parquet", index=False)
    verification = verify(root, frame, predictions, read(root / "results/saved_models.json"), h, e)
    primary = next(row for row in measurements if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    result = {"study_id": STUDY, "at": now(), "primary": primary, "all_accounts": measurements,
              "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_PERSISTENT_STATE_POLICY",
              "new_hidden_model_fits": len(models), "EM_updates_per_model": ITERATIONS, "daily_predictions": len(predictions),
              "new_full_accounts": 8, "comparisons": comparisons, "prediction_metrics": score_summary,
              "verification": verification, "tail_monitor_alerts": monitor.groupby("model").observation_alert.sum().to_dict(),
              "rolling_two_year_joint_passes": {policy: sum(row["joint_pass"] for row in rolling if row["policy"] == policy) for policy in POLICIES},
              "goal_achieved": False, "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(root / "result.json", result, True)
    print("两年每日持续状态实验完成，主结果和全部对照已保存。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="仅指数的两年每日持续状态与宏观增量研究")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "研究尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
