"""直接学习风险收益与换仓成本的有限策略研究，仅历史模拟。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from threadpoolctl import threadpool_limits

from research.adaptive_allocation_v1 import ROOT, identity, target_request, eligible_training, save_account, summarize
from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, fill_price, execute_order, normalize_dividends,
    digest, now, write_json, require, block_indices, return_metrics, interval,
)

STUDY = "510300_DIRECT_POLICY_UTILITY_V1"
OUT = ROOT / "reports/research/510300_direct_policy_utility_v1"
CONFIG = ROOT / "config/510300_direct_policy_utility_v1.json"
MANIFEST = ROOT / "config/510300_direct_policy_utility_v1_manifest.json"
PARENT = ROOT / "reports/research/510300_overnight_global_information_v1"
ENSEMBLES = {
    "U1_PRIMARY_FULL_SHARPE": ("FULL", "SHARPE"),
    "U2_PRICE_SHARPE": ("PRICE", "SHARPE"),
    "U3_FULL_MEANVAR": ("FULL", "MEANVAR"),
    "U4_PRICE_MEANVAR": ("PRICE", "MEANVAR"),
}


def specification() -> dict:
    config = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config.update({
        "study_id": STUDY, "primary": "U1_PRIMARY_FULL_SHARPE", "model_start": "2019-12-31",
        "minimum_train_samples": 500, "smoothing_alpha": 0.2, "feature_clip": 5.0,
        "coefficient_bound": 2.0, "intercept_bound": 4.0, "l2_penalty": 0.05,
        "turnover_smoothing_epsilon": 0.001, "annual_volatility_floor": 0.02,
        "training_turnover_cost": 0.0014, "mean_variance_gamma": 10.0,
        "optimizer": {"method": "L-BFGS-B", "maxiter": 250, "maxfun": 2000,
                      "ftol": 1e-9, "gtol": 1e-6, "maxls": 40},
        "training_single_start": "ALL_ZERO_PARAMETERS_EVERY_QUARTER",
        "training_objective_is_continuous_surrogate": True,
        "optimizer_failure_rule": "FINITE_NONWORSENING_LIMIT_RESULT_ALLOWED_OTHERWISE_NO_VIEW",
        "decision_clock": "NEXT_CHINA_SESSION_09_00_ASIA_SHANGHAI",
        "historical_vendor_first_delivery_proven": False,
        "training_carry_initial_weight": 0.5,
        "models": [{"id": f"{group}_{utility}_{window}", "group": group, "utility": utility,
                    "train_window": None if window == "EXPANDING" else 756}
                   for group in ("PRICE", "FULL") for utility in ("SHARPE", "MEANVAR")
                   for window in ("EXPANDING", "ROLL756")],
        "meta_names": ENSEMBLES, "rule_names": {},
    })
    return config


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "第五轮已经冻结，禁止覆盖")
    config = specification()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, config, exclusive=True)
    paths = {
        CONFIG, Path(__file__), ROOT / "docs/510300_DIRECT_POLICY_UTILITY_V1_PROTOCOL.md",
        ROOT / "tests/test_direct_policy_utility_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
        PARENT / "features.parquet", PARENT / "labels.parquet", PARENT / "source_receipt.json",
        PARENT / "source_clock_alignment.parquet",
    }
    for name in ("510300_adaptive_allocation_v1", "510300_overnight_global_information_v1"):
        path = ROOT / f"config/{name}_manifest.json"
        paths.add(path)
        paths.update(ROOT / item["path"] for item in json.loads(path.read_text(encoding="utf-8"))["files"])
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [identity(p) for p in sorted(paths)],
               "own_candidate_returns_read_before_freeze": False, "underlying_history_previously_observed": True,
               "prior_rounds_known": 4, "finite_candidates": 12,
               "method_reference": "https://people.idsia.ch/~juergen/rnnaissance2003talks/MoodySaffellTNN01.pdf",
               "implementation_reference": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html",
               "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}, exclusive=True)
    print(json.dumps({"状态": "第五轮已冻结", "清单哈希": digest(MANIFEST)}, ensure_ascii=False), flush=True)


def normalized_design(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
    return np.column_stack((np.ones(len(x)), np.clip((x - mean) / scale, -clip, clip)))


def smooth_policy(design: np.ndarray, theta: np.ndarray, alpha: float,
                  segment_starts: np.ndarray, initial: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    probability = expit(design @ theta)
    raw_gradient = probability[:, None] * (1 - probability[:, None]) * design
    weights = np.empty(len(probability))
    gradients = np.empty_like(raw_gradient)
    ends = np.r_[segment_starts[1:], len(probability)]
    for start, end in zip(segment_starts, ends):
        weights[start:end], _ = lfilter([alpha], [1, -(1 - alpha)], probability[start:end], zi=[(1 - alpha) * initial])
        gradients[start:end], _ = lfilter([alpha], [1, -(1 - alpha)], raw_gradient[start:end],
                                          axis=0, zi=np.zeros((1, design.shape[1])))
    return weights, gradients


def objective(theta: np.ndarray, design: np.ndarray, returns: np.ndarray,
              segment_starts: np.ndarray, utility: str, config: dict) -> tuple[float, np.ndarray]:
    weights, gradients = smooth_policy(design, theta, config["smoothing_alpha"], segment_starts,
                                      config["training_carry_initial_weight"])
    previous = np.r_[0.0, weights[:-1]]
    previous_gradient = np.vstack((np.zeros(design.shape[1]), gradients[:-1]))
    previous[segment_starts] = 0.0
    previous_gradient[segment_starts] = 0.0
    delta, delta_gradient = weights - previous, gradients - previous_gradient
    eps = config["turnover_smoothing_epsilon"]
    smooth_absolute = np.sqrt(delta * delta + eps * eps)
    turnover = smooth_absolute - eps
    turnover_gradient = delta[:, None] / smooth_absolute[:, None] * delta_gradient
    net = weights * returns - config["training_turnover_cost"] * turnover
    net_gradient = gradients * returns[:, None] - config["training_turnover_cost"] * turnover_gradient
    # 每个训练连续片段的最后一个持有期结束时支付退出近似成本，避免遗忘退出费用。
    ends = np.r_[segment_starts[1:] - 1, len(weights) - 1]
    for end in ends:
        magnitude = np.sqrt(weights[end] ** 2 + eps ** 2)
        net[end] -= config["training_turnover_cost"] * (magnitude - eps)
        net_gradient[end] -= config["training_turnover_cost"] * weights[end] / magnitude * gradients[end]
    annual = config["annual_days"]
    mean, mean_gradient = float(net.mean()), net_gradient.mean(axis=0)
    centered = net - mean
    variance = float(centered @ centered / (len(net) - 1))
    variance_gradient = 2 * (centered[:, None] * net_gradient).sum(axis=0) / (len(net) - 1)
    if utility == "SHARPE":
        denominator = np.sqrt(variance + config["annual_volatility_floor"] ** 2 / annual)
        reward = np.sqrt(annual) * mean / denominator
        reward_gradient = np.sqrt(annual) * (mean_gradient / denominator - mean * variance_gradient / (2 * denominator ** 3))
    elif utility == "MEANVAR":
        gamma = config["mean_variance_gamma"]
        reward = annual * (mean - gamma / 2 * variance)
        reward_gradient = annual * (mean_gradient - gamma / 2 * variance_gradient)
    else:
        raise ValueError("未知训练目标")
    penalty = config["l2_penalty"] * float(theta @ theta)
    return float(-reward + penalty), -reward_gradient + 2 * config["l2_penalty"] * theta


def fit_policy(x: np.ndarray, returns: np.ndarray, origins: np.ndarray, utility: str, config: dict) -> dict:
    require(len(x) >= config["minimum_train_samples"], "成熟训练样本不足")
    require(np.isfinite(x).all() and np.isfinite(returns).all(), "训练输入存在缺失")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = normalized_design(x, mean, scale, config["feature_clip"])
    starts = np.r_[0, np.flatnonzero(np.diff(origins) != 1) + 1]
    zero = np.zeros(design.shape[1])
    initial_loss = objective(zero, design, returns, starts, utility, config)[0]
    bounds = [(-config["intercept_bound"], config["intercept_bound"])] + [(-config["coefficient_bound"], config["coefficient_bound"])] * x.shape[1]
    settings = dict(config["optimizer"])
    method = settings.pop("method")
    with threadpool_limits(limits=1):
        result = minimize(objective, zero, args=(design, returns, starts, utility, config), jac=True,
                          method=method, bounds=bounds, options=settings)
    usable = bool(np.isfinite(result.x).all() and np.isfinite(result.fun) and result.fun <= initial_loss + 1e-10
                  and (result.success or result.status == 1))
    return {"theta": result.x, "mean": mean, "scale": scale, "usable": usable,
            "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
            "optimizer_message": str(result.message), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "initial_loss": initial_loss,
            "final_loss": float(result.fun), "training_segments": len(starts)}


def predict_sequence(model: dict, x: np.ndarray, valid: np.ndarray, previous: float, config: dict) -> tuple[np.ndarray, float]:
    output = np.full(len(x), np.nan)
    if not model["usable"]:
        return output, previous
    indexes = np.flatnonzero(valid & np.isfinite(x).all(axis=1))
    if len(indexes):
        design = normalized_design(x[indexes], model["mean"], model["scale"], config["feature_clip"])
        raw = expit(design @ model["theta"])
        alpha = config["smoothing_alpha"]
        for index, target in zip(indexes, raw):
            previous = (1 - alpha) * previous + alpha * float(target)
            output[index] = previous
    return output, previous


def train_all(data: pd.DataFrame, labels: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    source = json.loads((PARENT / "source_receipt.json").read_text(encoding="utf-8"))
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0]) - 1
    quarter = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [first] + [t for t in range(first + 1, len(data) - 1) if quarter[t] != quarter[t - 1]] + [len(data) - 1]
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(labels.date)), "特征与成熟收益日期不一致")
    y = labels.Y1.to_numpy(float)
    predictions, receipts, parameters = {}, [], []
    (OUT / "final_models").mkdir(exist_ok=True)
    for number, item in enumerate(config["models"], 1):
        columns = source["price_features"] + (source["external_features"] if item["group"] == "FULL" else [])
        x, targets = data[columns].to_numpy(float), np.full(len(data), np.nan)
        previous = config["training_carry_initial_weight"]
        for t, end in zip(cuts[:-1], cuts[1:]):
            train = eligible_training(data, t, 1, item["train_window"])
            require((train + 2 <= t).all(), "训练持有期退出越过拟合日")
            model = fit_policy(x[train], y[train], train, item["utility"], config)
            targets[t:end], previous = predict_sequence(model, x[t:end], data.feature_valid.iloc[t:end].to_numpy(), previous, config)
            receipts.append({"model": item["id"], "group": item["group"], "utility": item["utility"],
                             "train_window": item["train_window"], "fit_origin": data.date.iloc[t],
                             "first_train_origin": data.date.iloc[train[0]], "last_train_origin": data.date.iloc[train[-1]],
                             "last_label_exit": data.date.iloc[train[-1] + 2], "train_rows": len(train),
                             "last_prediction_origin": data.date.iloc[end - 1],
                             "training_origin_sha256": hashlib.sha256(train.astype(np.int64).tobytes()).hexdigest(),
                             "training_array_sha256": hashlib.sha256(x[train].tobytes() + y[train].tobytes()).hexdigest(),
                             **{key: value for key, value in model.items() if key not in ("theta", "mean", "scale")}})
            parameters.append({"model": item["id"], "fit_origin": str(data.date.iloc[t]), "features": columns,
                               "theta": model["theta"].tolist(), "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
                               "usable": model["usable"]})
        joblib.dump({"model": model, "features": columns, "specification": item, "last_fit_receipt": receipts[-1]},
                     OUT / "final_models" / f"{item['id']}.joblib", compress=3)
        predictions[item["id"]] = targets
        print(f"直接仓位学习已完成 {number}/{len(config['models'])}：{label(item['id'])}", flush=True)
    for key, (group, utility) in ENSEMBLES.items():
        predictions[key] = np.mean(np.column_stack([predictions[f"{group}_{utility}_{w}"] for w in ("EXPANDING", "ROLL756")]), axis=1)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "targets.parquet", index=False)
    write_json(OUT / "all_quarter_parameters.json", parameters)
    return predictions


def simulate_policy(data: pd.DataFrame, dividends: pd.DataFrame, config: dict, cost: dict,
                    model_id: str, targets: np.ndarray | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0])
    last, anchor = len(data) - 1, first - 1
    account = Account(config["initial_capital"])
    dates = pd.DatetimeIndex(data.date)
    values = {name: data[name].to_numpy() for name in ("open", "close", "previous_close", "dividend", "feature_valid")}
    events = dividends.to_dict("records")
    previous_nav, previous_mark = config["initial_capital"], float(values["close"][anchor])
    records, decisions = [], []

    def decide(t: int) -> dict:
        close = float(values["close"][t])
        if model_id == "BUY_HOLD":
            quantity = affordable_quantity(account.cash, fill_price(close, 1, cost, config["tick"]), cost, config["lot"]) if t == anchor else 0
            decision = {"requested_quantity": quantity, "reference_weight": 1.0, "action": "买入持有", "view": "有效"}
        elif not values["feature_valid"][t] or targets is None or not np.isfinite(targets[t]):
            decision = {"requested_quantity": 0, "reference_weight": None, "action": "无有效判断，保持已有份额", "view": "NO_VIEW"}
        else:
            decision = target_request(account, close, float(targets[t]), config)
            decision["view"] = "有效"
        decisions.append({"origin": dates[t], "decision_time": dates[t + 1] + pd.Timedelta(hours=9),
                          "execution_date": dates[t + 1], "simulation_only": True, **decision})
        return decision

    pending = decide(anchor)
    for day in range(first, last + 1):
        date, op, cl = dates[day], float(values["open"][day]), float(values["close"][day])
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for key, event in enumerate(events):
            if event["ex_date"] == date:
                value = account.entitlements.get(key, 0) * event["cash_dividend_per_share"]
                account.receivables[key] = value
                recognized += value
            if event["payment_date"] < date and key in account.receivables:
                value = account.receivables.pop(key)
                account.cash += value
                paid += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        pretrade = account.value(op)
        execution = execute_order(account, quantity, op, float(values["previous_close"][day]), float(values["dividend"][day]), day, cost, config)
        mark = op if terminal else cl
        if not terminal:
            for key, event in enumerate(events):
                if event["payment_date"] == date and key in account.receivables:
                    value = account.receivables.pop(key)
                    account.cash += value
                    paid += value
                if event["record_date"] == date:
                    account.entitlements[key] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"直接仓位学习账户财富不守恒：{model_id}，{date}")
        account.assert_valid()
        records.append({"date": date, "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / pretrade, "view_at_order_origin": pending["view"], **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)


def label(key: str) -> str:
    names = {"U1_PRIMARY_FULL_SHARPE": "主方案：全部信息、直接夏普、双窗口共识",
             "U2_PRICE_SHARPE": "价格信息、直接夏普、双窗口共识",
             "U3_FULL_MEANVAR": "全部信息、风险收益、双窗口共识",
             "U4_PRICE_MEANVAR": "价格信息、风险收益、双窗口共识", "BUY_HOLD": "买入持有"}
    if key in names:
        return names[key]
    group, utility, window = key.split("_")
    return f"{'全部信息' if group == 'FULL' else '价格信息'}、{'直接夏普' if utility == 'SHARPE' else '风险收益'}、{'扩展样本' if window == 'EXPANDING' else '近756个交易日'}"


def run(expected: str) -> None:
    require(digest(MANIFEST) == expected, "第五轮冻结清单哈希不符")
    for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        path = ROOT / item["path"]
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], f"冻结文件变化：{item['path']}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    data = pd.read_parquet(PARENT / "features.parquet")
    labels = pd.read_parquet(PARENT / "labels.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    alignment = pd.read_parquet(PARENT / "source_clock_alignment.parquet")
    valid = alignment.available_at.notna()
    require((alignment.loc[valid, "available_at"] <= alignment.loc[valid, "decision_time"]).all(), "海外信息穿越决策时点")
    targets = train_all(data, labels, config)
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for cost_id, cost in config["costs"].items():
        ledgers = {}
        for key in list(targets) + ["BUY_HOLD"]:
            ledger, decisions = simulate_policy(data, dividends, config, cost, key, targets.get(key))
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            ledgers[key] = ledger
        benchmark = summarize(ledgers["BUY_HOLD"], config)
        for key, ledger in ledgers.items():
            item = {"cost": cost_id, "model": key, **summarize(ledger, config)}
            item["annualized_return_excess_vs_buy_hold"] = item["annualized_return"] - benchmark["annualized_return"]
            item["meets_point_target"] = item["net_sharpe"] is not None and item["net_sharpe"] >= 1.2
            metrics.append(item)
            for year, part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(part, config)})
            for name, start, end in (("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                                     ("2024—终点", "2024-01-01", config["data_cutoff"])):
                part = ledger.loc[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": key, "era": name, **summarize(part, config)})
        primary = ledgers[config["primary"]].net_return.to_numpy()
        compare = ledgers["U2_PRICE_SHARPE"].net_return.to_numpy()
        benchmark_returns = ledgers["BUY_HOLD"].net_return.to_numpy()
        rng = np.random.default_rng(config["random_seed"])
        samples = {"primary_sharpe": [], "annualized_arithmetic_excess_vs_buy_hold": [], "annualized_arithmetic_increment_vs_price_policy": []}
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(primary), config["bootstrap_day_block"])
            value = return_metrics(primary[ix], config["annual_days"])["net_sharpe"]
            samples["primary_sharpe"].append(np.nan if value is None else value)
            samples["annualized_arithmetic_excess_vs_buy_hold"].append(float((primary[ix] - benchmark_returns[ix]).mean() * 242))
            samples["annualized_arithmetic_increment_vs_price_policy"].append(float((primary[ix] - compare[ix]).mean() * 242))
        uncertainty[cost_id] = {**{key + "_95_interval": interval(value) for key, value in samples.items()}, "multiple_search_adjusted": False}
        print(f"第五轮 {cost_id} 的 13 个完整账户已完成", flush=True)
    # 预先登记的零费用诊断仅解释费用损耗，绝不用作达标依据。
    zero_cost = {"commission": 0.0, "minimum": 0.0, "slippage": 0.0}
    diagnostic, decisions = simulate_policy(data, dividends, config, zero_cost, config["primary"], targets[config["primary"]])
    save_account(OUT / "diagnostic/ZERO_COST", config["primary"], diagnostic, decisions)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    primary_rows = frame.loc[frame.model == config["primary"]]
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    reached = bool(primary_rows.meets_point_target.all())
    training = pd.read_csv(OUT / "training_receipts.csv")
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary_rows.to_dict("records"), "post_selected_best_base": best.to_dict(),
              "number_of_candidates": 12, "benchmark_count": 1, "evaluation_accounts": 26, "diagnostic_accounts": 1,
              "trained_model_count": 8, "quarterly_fits": len(training), "usable_fits": int(training.usable.sum()),
              "optimizer_converged_fits": int(training.optimizer_success.sum()),
              "primary_point_target_in_both_costs": reached,
              "zero_cost_primary_diagnostic": summarize(diagnostic, config),
              "historical_vendor_first_delivery_proven": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
              "training_objective_is_not_real_account_performance": True, "uncertainty": uncertainty,
              "position_impact": 0, "validated_live_strategy": "NONE"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    lines = ["# 第五轮：直接学习风险收益与换仓成本的实际结果", "",
             f"预先指定主方案基础夏普 **{primary_rows.iloc[0].net_sharpe:.4f}**，压力夏普 **{primary_rows.iloc[1].net_sharpe:.4f}**。目标为 1.2。",
             "", "本轮只改变学习目标与仓位生成方式。训练使用连续近似，最终成绩来自含整手、分红、T+1 和真实费用的完整账户。",
             "", f"完成 12 个登记候选、26 个评价账户和 1 个零费用诊断账户；{len(training)} 次季度拟合中，{int(training.usable.sum())} 次按事前规则可用。",
             "", "|费用|策略中文名称|夏普率|年化收益|最大回撤|成交次数|", "|---|---|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"|{'基础' if row.cost == 'BASE' else '压力'}|{label(row.model)}|{row.net_sharpe:.4f}|{row.annualized_return:.2%}|{row.max_drawdown:.2%}|{row.trade_count}|")
    lines += ["", f"事后最高为“{label(best.model)}”，基础夏普 {best.net_sharpe:.4f}。这不是主方案替换或独立验证。",
              "", f"主方案零费用诊断夏普为 {result['zero_cost_primary_diagnostic']['net_sharpe']:.4f}，只解释摩擦，不作为目标验收口径。",
              "", "所有原有失败记录保留；本轮完整历史已被反复观察，不能当成未见样本。未来免费接口实时交付能力尚未证明。"]
    (OUT / "第五轮全部候选_实际结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第五轮直接仓位学习：冻结或按冻结身份完成研究")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        require(bool(args.manifest_sha256), "必须传入事前冻结清单哈希")
        run(args.manifest_sha256)
    else:
        parser.error("请指定冻结或执行")
