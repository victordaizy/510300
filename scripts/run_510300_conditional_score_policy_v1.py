"""一次冻结、开发、主评价和交付；直接文件运行兼容Windows。"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import LinearConstraint, differential_evolution

from research.intraday_overnight_increment_v1 import (
    digest, fill_price, normalize_dividends, now, require, safe_json, write_json,
)
from research.conditional_score_policy_v1 import (
    FULL_COLUMNS, assert_batch_parity, batch_simulate, design, features, metrics,
    period, score, simulate_reference,
)

STEM = "510300_conditional_score_policy_v1"
CONFIG_PATH = ROOT / f"config/{STEM}.json"
PROTOCOL = ROOT / "docs/510300_CONDITIONAL_SCORE_POLICY_V1_PROTOCOL.md"
USER_TEXT = ROOT / "docs/510300_CONDITIONAL_SCORE_POLICY_V1_USER_TEXT_20260905.txt"
IMPLEMENTATION = [CONFIG_PATH, PROTOCOL, USER_TEXT, Path(__file__),
                  ROOT / "research/conditional_score_policy_v1.py",
                  ROOT / "research/intraday_overnight_increment_v1.py",
                  ROOT / "tests/test_510300_conditional_score_policy_v1.py"]


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def event(output: Path, name: str, **values) -> None:
    row = {"time": now(), "event": name, **values}
    with (output / "stage_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(safe_json(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    print(f"{row['time']} {name} {json.dumps(safe_json(values), ensure_ascii=False)}", flush=True)


def freeze(config: dict, output: Path) -> None:
    require(not (output / "freeze_manifest.json").exists(), "已经冻结；禁止覆盖原冻结")
    require(not (output / "run_claim.json").exists(), "已存在运行声明")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"study_id": config["study_id"], "created_at": now(),
                "state": "PRE_RETURN_FROZEN", "new_model_fits": 0, "main_evaluation_read": False,
                "implementation": [identity(p) for p in IMPLEMENTATION],
                "inputs": {key: identity(ROOT / value) for key, value in config["inputs"].items()},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__},
                "authorization": "本轮用户原文明确授权独立连续评分持仓历史研究；只生成模拟账户",
                "source_limit": "历史已被多轮研究观察，不能称为全新盲测"}
    for source in IMPLEMENTATION:
        target = output / "frozen_sources" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for key, source in config["inputs"].items():
        target = output / "frozen_inputs" / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    write_json(output / "freeze_manifest.json", manifest, exclusive=True)
    event(output, "协议代码输入冻结完成", manifest_sha256=digest(output / "freeze_manifest.json"))


def verify_freeze(output: Path) -> dict:
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in [*manifest["implementation"], *manifest["inputs"].values()]:
        require(identity(ROOT / item["path"]) == item, f"冻结文件发生变化：{item['path']}")
    return manifest


def load_data(config: dict, end: str, output: Path, stage: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """底层读取附带日期过滤，开发进程不拿到2020年后行情数值。"""
    if pd.Timestamp(end) > pd.Timestamp(config["train_cutoff"]):
        lock_path = output / "model_freeze.json"
        require(lock_path.exists(), "主评价行情读取前缺少模型冻结")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for item in lock["development_artifacts"]:
            require(identity(ROOT / item["path"]) == item, "主评价前开发产物哈希变化")
    event(output, "行情日期过滤读取", phase=stage, maximum_date=end,
          main_evaluation=bool(pd.Timestamp(end) > pd.Timestamp(config["train_cutoff"])))
    inputs = {name: ROOT / path for name, path in config["inputs"].items()}
    receipt = json.loads(inputs["price_receipt"].read_text(encoding="utf-8-sig"))
    coverage = json.loads(inputs["dividend_coverage"].read_text(encoding="utf-8-sig"))
    require(receipt["status"] == "PASS" and receipt["canonical_price"]["output_sha256"] == digest(inputs["prices"]), "既有修正价格凭证不匹配")
    require(coverage["complete_history_confirmed"] and coverage["distribution_file_sha256"] == digest(inputs["dividends"]), "分红覆盖凭证不匹配")
    require(pd.Timestamp(coverage["coverage_end"]) >= pd.Timestamp(config["data_cutoff"]), "分红覆盖不足")
    dividends = normalize_dividends(pd.read_csv(inputs["dividends"]))
    require(len(dividends) == coverage["event_count"], "分红事件数量不符")
    calendar = pd.read_parquet(inputs["calendar"], columns=["trade_date", "is_open"])
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    require(not dates.duplicated().any(), "交易日历重复")
    dates = dates.sort_values()
    warmup = dates[dates < pd.Timestamp(config["train_start"])][-config["warmup_trading_days"]:]
    require(len(warmup) == config["warmup_trading_days"], "暖启动日历不足")
    lower, upper = warmup[0], pd.Timestamp(end)
    raw = pd.read_parquet(inputs["prices"], filters=[("date", ">=", lower), ("date", "<=", upper)])
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    require(not raw.date.duplicated().any(), "行情日期重复")
    require(set(raw.symbol) == {"510300.SH"}, "标的超出授权")
    require(set(raw.amount_unit) == {"CNY"} and set(raw.volume_unit) == {"share"}, "金额或份额单位不符合冻结合同")
    expected = dates[(dates >= lower) & (dates <= upper)]
    require(set(raw.date).issubset(set(expected)), "行情出现非交易日")
    data = raw.set_index("date").reindex(expected).rename_axis("date").reset_index()
    require(np.isfinite(data.close).all() and (data.close > 0).all(), "缺少可信收盘估值，不能发布完整经济结论")
    for row in dividends.itertuples():
        if lower <= row.ex_date <= upper:
            require(row.record_date in expected and row.ex_date in expected, "分红登记或除息不在日历")
            require(expected[expected.get_loc(row.ex_date) - 1] == row.record_date, "登记至除息存在未定义的权益日")
    result = features(data, dividends, config["scale_floor"])
    main = result.loc[result.date >= pd.Timestamp(config["train_start"])]
    report = {"phase": stage, "maximum_permitted_read_date": end, "first_date": data.date.min(),
              "last_date": data.date.max(), "calendar_rows": len(data), "research_rows": len(main),
              "warmup_rows": len(warmup), "no_view_rows": int((~main.feature_valid).sum()),
              "missing_close_rows": 0, "amount_unit": "CNY", "distribution_events": len(dividends),
              "status": "PASS_REQUIRED_DATA_CONTRACT", "main_data_available_to_training": False}
    write_json(output / f"data_contract_{stage}.json", report, exclusive=True)
    result.to_parquet(output / f"features_{stage}.parquet", index=False)
    return result, dividends, report


def initial_population(dimensions: int, seed: int, members: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = [[-2.0] + [0.0] * (dimensions - 1), [0.0] * dimensions]
    while len(values) < members:
        row = np.r_[rng.uniform(-2, 2), rng.uniform(0, 2, dimensions - 1)]
        if row[1:].sum() > 6 or (dimensions == 6 and row[3] > row[4] / 2):
            continue
        values.append(row)
    return np.asarray(values)


def fit(p, model: str, gamma: float, penalty: float, config: dict, training_end_year: int) -> dict:
    dimensions = 6 if model == "FULL" else 4
    x = design(p.data, model)
    valid = p.data.feature_valid.to_numpy(bool)
    opt = config["optimizer"]
    seed = opt["seed"] + training_end_year
    count = 0

    def objective(candidates: np.ndarray) -> np.ndarray:
        nonlocal count
        count += candidates.shape[1]
        if candidates.shape[1] == 0:
            return np.empty(0)
        require(count <= opt["maximum_candidate_evaluations_per_fit"], "优化候选预算超限")
        values = 100 / (1 + np.exp(-(x @ candidates)))
        values[~valid, :] = np.nan
        statistics = batch_simulate(p, values, config)
        utility = np.min(statistics["mu"] - gamma / 2 * statistics["variance"], axis=0)
        loss = -utility + penalty * np.sum(candidates[1:] ** 2, axis=0)
        turnover = np.max(statistics["turnover"], axis=0)
        return np.where(turnover <= config["maximum_annual_two_way_turnover"] + 1e-12,
                        loss, 1000.0 + turnover)

    constraint_matrix = np.ones((1, dimensions))
    constraint_matrix[0, 0] = 0
    lower, upper = [-np.inf], [6.0]
    if model == "FULL":
        structure = np.array([[0, 0, 0, 1, -0.5, 0]])
        constraint_matrix = np.vstack([constraint_matrix, structure])
        lower.append(-np.inf)
        upper.append(0.0)
    started = time.monotonic()
    result = differential_evolution(
        objective, [(-2, 2)] + [(0, 2)] * (dimensions - 1), strategy=opt["strategy"],
        maxiter=opt["maxiter"], mutation=tuple(opt["mutation"]), recombination=opt["recombination"],
        tol=opt["tol"], atol=opt["atol"], polish=False, updating="deferred", vectorized=True,
        workers=1, rng=np.random.default_rng(seed),
        init=initial_population(dimensions, seed, opt["population_members"]),
        constraints=(LinearConstraint(constraint_matrix, lower, upper),),
    )
    coefficients = result.x
    stats = batch_simulate(p, score(p.data, coefficients, model), config)
    require(np.max(stats["turnover"]) <= config["maximum_annual_two_way_turnover"] + 1e-10, "优化终解违反换手约束")
    require(np.isfinite(result.fun) and result.fun < 1000, "没有可行终解")
    require(np.max(constraint_matrix @ coefficients - np.asarray(upper)) <= 1e-10, "优化终解违反系数结构")
    objective_value = float(np.min(stats["mu"] - gamma / 2 * stats["variance"]) - penalty * np.sum(coefficients[1:] ** 2))
    require(abs(objective_value + result.fun) < 1e-10, "优化目标复算不一致")
    names = ["intercept", "T", "Q", "B", "D", "R"] if model == "FULL" else ["intercept", "T", "Q", "V_PLUS"]
    return {"model": model, "gamma": gamma, "lambda": penalty, "coefficients": coefficients.tolist(),
            "named_coefficients": dict(zip(names, coefficients.tolist())),
            "training_first_day": p.data.date.iloc[1], "training_last_day": p.data.date.iloc[-1],
            "training_days": p.n, "objective": objective_value, "training_statistics": stats,
            "seed": seed, "generations": int(result.nit), "candidate_evaluations": count,
            "vectorized_function_calls": int(result.nfev), "converged": bool(result.success),
            "termination": "CONVERGED" if result.success else "FIXED_BUDGET_EXHAUSTED_BEST_FEASIBLE",
            "optimizer_message": str(result.message), "seconds": time.monotonic() - started}


def select_configuration(rows: list[dict], model: str, config: dict) -> tuple[dict, list[dict]]:
    ranking = []
    for gamma in config["gamma_grid"]:
        for penalty in config["lambda_grid"]:
            matched = [row for row in rows if row["model"] == model and row["gamma"] == gamma and row["lambda"] == penalty]
            require(len(matched) == 3, "配置缺少完整三个验证年")
            utilities = np.asarray([row["validation"]["STRESS"]["utility_gamma5"] for row in matched])
            turnover_ok = all(row["validation"][cost]["annualized_two_way_turnover"] <= config["maximum_annual_two_way_turnover"] + 1e-12
                              for row in matched for cost in ("BASE", "STRESS"))
            ranking.append({"model": model, "gamma": gamma, "lambda": penalty,
                            "validation_years": config["validation_years"], "stress_utility_by_year": utilities,
                            "mean_stress_utility": float(utilities.mean()),
                            "std_stress_utility": float(utilities.std(ddof=1)),
                            "selection_score": float(utilities.mean() - config["validation_dispersion_penalty"] * utilities.std(ddof=1)),
                            "turnover_eligible": turnover_ok})
    eligible = [row for row in ranking if row["turnover_eligible"]]
    require(bool(eligible), f"{model}九组配置全部被验证换手约束阻断")
    best_score = max(row["selection_score"] for row in eligible)
    ties = [row for row in eligible if abs(row["selection_score"] - best_score) <= 1e-12]
    selected = max(ties, key=lambda row: (row["lambda"], row["gamma"]))
    for row in ranking:
        row["selected"] = row is selected
    return selected, ranking


def develop(data: pd.DataFrame, dividends: pd.DataFrame, config: dict, output: Path) -> dict:
    directory = output / "development"
    directory.mkdir(parents=True, exist_ok=True)
    results, ranking, selected_models, parity = [], [], {}, []
    for model in ("FULL", "SIMPLE"):
        for gamma in config["gamma_grid"]:
            for penalty in config["lambda_grid"]:
                for year in config["validation_years"]:
                    name = f"{model}_g{gamma:g}_l{penalty:g}_validate_{year}"
                    event(output, "内层拟合开始", fit=name, train_end=f"{year-1}-12-31")
                    train = period(data, dividends, config["train_start"], f"{year-1}-12-31")
                    fitted = fit(train, model, gamma, penalty, config, year - 1)
                    validation = period(data, dividends, f"{year}-01-01", f"{year}-12-31")
                    values = score(validation.data, np.asarray(fitted["coefficients"]), model)
                    validation_metrics = {}
                    check = assert_batch_parity(validation, values, config, model)
                    for cost in ("BASE", "STRESS"):
                        ledger, _, _ = simulate_reference(validation, values, config["costs"][cost], config, model)
                        validation_metrics[cost] = metrics(ledger, config)
                        ledger.to_parquet(directory / f"{name}_{cost}_validation.parquet", index=False)
                    fitted["validation_year"] = year
                    fitted["validation"] = validation_metrics
                    fitted["account_engine_parity"] = check
                    results.append(fitted)
                    write_json(directory / f"{name}.json", fitted, exclusive=True)
                    event(output, "内层拟合完成", fit=name, candidates=fitted["candidate_evaluations"],
                          generations=fitted["generations"], termination=fitted["termination"],
                          stress_validation_utility=validation_metrics["STRESS"]["utility_gamma5"],
                          elapsed_seconds=round(fitted["seconds"], 2))
        selected, model_ranking = select_configuration(results, model, config)
        ranking.extend(model_ranking)
        write_json(directory / f"{model}_selection.json", {"selected": selected, "all_nine": model_ranking}, exclusive=True)
        event(output, "内层选择锁定", model=model, gamma=selected["gamma"], penalty=selected["lambda"],
              selection_score=selected["selection_score"], main_evaluation_read=False)
        train = period(data, dividends, config["train_start"], config["train_cutoff"])
        fitted = fit(train, model, selected["gamma"], selected["lambda"], config, 2019)
        fitted["selection"] = selected
        values = score(train.data, np.asarray(fitted["coefficients"]), model)
        check = assert_batch_parity(train, values, config, model)
        fitted["account_engine_parity"] = check
        parity.append({"model": model, "period": "2015-2019", "errors": check})
        selected_models[model] = fitted
        write_json(directory / f"{model}_final_fit.json", fitted, exclusive=True)
        event(output, "主历史模型重估完成", model=model, coefficients=fitted["named_coefficients"],
              termination=fitted["termination"], main_evaluation_read=False)
    write_json(directory / "all_inner_results.json", {"fits": results}, exclusive=True)
    pd.DataFrame(ranking).to_csv(directory / "configuration_ranking.csv", index=False, encoding="utf-8-sig")
    flat = []
    for row in results:
        for cost in ("BASE", "STRESS"):
            flat.append({"model": row["model"], "gamma": row["gamma"], "lambda": row["lambda"],
                         "validation_year": row["validation_year"], "cost": cost,
                         "training_objective": row["objective"], "candidate_evaluations": row["candidate_evaluations"],
                         "termination": row["termination"], **row["named_coefficients"], **row["validation"][cost]})
    pd.DataFrame(flat).to_csv(directory / "all_inner_results.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "selected_models.json", selected_models, exclusive=True)
    artifacts = sorted(p for p in directory.iterdir() if p.is_file())
    lock = {"created_at": now(), "state": "COEFFICIENTS_FROZEN_BEFORE_MAIN_READ", "models": selected_models,
            "inner_fits": len(results), "final_history_fits": 2,
            "distinct_gamma_lambda_pairs": 9, "main_data_used_for_selection": False,
            "total_candidate_evaluations": sum(row["candidate_evaluations"] for row in results) + sum(row["candidate_evaluations"] for row in selected_models.values()),
            "development_artifacts": [identity(p) for p in artifacts], "training_parity": parity}
    write_json(output / "model_freeze.json", lock, exclusive=True)
    event(output, "主评价前模型冻结完成", model_freeze_sha256=digest(output / "model_freeze.json"),
          total_candidate_evaluations=lock["total_candidate_evaluations"])
    return lock


def robust_statistics(ledgers: dict[str, pd.DataFrame], config: dict) -> dict:
    keys = list(ledgers)
    returns = np.column_stack([ledgers[key].net_return.to_numpy(float) for key in keys])
    n, count = returns.shape
    annual = config["annual_days"]
    rng = np.random.default_rng(config["bootstrap_seed"])
    boot = np.full((config["bootstrap_repetitions"], count, 4), np.nan)
    for rep in range(config["bootstrap_repetitions"]):
        starts = rng.integers(0, n, size=math.ceil(n / config["bootstrap_block_days"]))
        indices = ((starts[:, None] + np.arange(config["bootstrap_block_days"])) % n).reshape(-1)[:n]
        selected = returns[indices]
        mu, sigma = selected.mean(axis=0) * annual, selected.std(axis=0, ddof=1) * math.sqrt(annual)
        boot[rep, :, 0] = np.divide(mu, sigma, out=np.full(count, np.nan), where=sigma > 1e-8)
        boot[rep, :, 1] = np.expm1(np.log1p(selected).sum(axis=0) * annual / n)
        boot[rep, :, 2] = mu - 2.5 * sigma * sigma
        boot[rep, :, 3] = mu
    result = {"method": "20日联合循环移动区块；给定冻结模型的样本不确定性，不修正历次研究选择",
              "block_days": config["bootstrap_block_days"], "repetitions": config["bootstrap_repetitions"],
              "seed": config["bootstrap_seed"], "models": {}, "paired_increments": {}}
    for j, key in enumerate(keys):
        r = returns[:, j]
        centered = r - r.mean()
        long_variance = float(centered @ centered / n)
        for lag in range(1, config["hac_lags"] + 1):
            long_variance += 2 * (1 - lag / (config["hac_lags"] + 1)) * float(centered[lag:] @ centered[:-lag] / n)
        value = {"sharpe_95_interval": np.nanquantile(boot[:, j, 0], [0.025, 0.975]),
                 "cagr_95_interval": np.quantile(boot[:, j, 1], [0.025, 0.975]),
                 "hac_lags": config["hac_lags"], "hac_long_run_daily_variance": long_variance,
                 "hac_adjusted_sharpe": r.mean() * math.sqrt(annual / long_variance) if long_variance > 1e-15 else None,
                 "hac_mean_t": r.mean() * math.sqrt(n / long_variance) if long_variance > 1e-15 else None}
        result["models"][key] = value
    for cost in ("BASE", "STRESS"):
        for opponent in ("SIMPLE", "BUY_HOLD", "NO_REPAIR"):
            left, right = keys.index(f"FULL_{cost}"), keys.index(f"{opponent}_{cost}")
            delta = boot[:, left] - boot[:, right]
            result["paired_increments"][f"FULL_MINUS_{opponent}_{cost}"] = {
                "sharpe_difference_95_interval": np.nanquantile(delta[:, 0], [0.025, 0.975]),
                "cagr_difference_95_interval": np.quantile(delta[:, 1], [0.025, 0.975]),
                "utility_difference_95_interval": np.quantile(delta[:, 2], [0.025, 0.975])}
    return result


def fixed_path_costs(p, base_ledger: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """在基础实际份额/日期上纯改费用，不反馈到决策；现金可行性单独报告。"""
    prices = p.data.iloc[1:].reset_index(drop=True)
    quantity = base_ledger.filled_quantity.to_numpy(int)
    opens = prices.open.to_numpy(float)
    delta_fee = np.zeros(len(quantity))
    delta_slip = np.zeros(len(quantity))
    limit_infeasible = np.zeros(len(quantity), dtype=bool)
    for i, q in enumerate(quantity):
        if not q:
            continue
        fees, slips = {}, {}
        for name in ("BASE", "STRESS"):
            cost = config["costs"][name]
            px = fill_price(float(opens[i]), 1 if q > 0 else -1, cost, config["tick"])
            fees[name] = max(abs(q) * px * cost["commission"], cost["minimum"])
            slips[name] = abs(q) * abs(px - opens[i])
            if name == "STRESS":
                basis = float(prices.previous_close.iloc[i] - prices.dividend.iloc[i])
                upper = math.floor(basis * (1 + config["limit_fraction"]) / config["tick"] + .5 + 1e-9) * config["tick"]
                lower = math.floor(basis * (1 - config["limit_fraction"]) / config["tick"] + .5 + 1e-9) * config["tick"]
                limit_infeasible[i] = (q > 0 and (opens[i] >= upper - 1e-9 or px > upper + 1e-9)) or (q < 0 and (opens[i] <= lower + 1e-9 or px < lower - 1e-9))
        delta_fee[i] = fees["STRESS"] - fees["BASE"]
        delta_slip[i] = slips["STRESS"] - slips["BASE"]
    cumulative = np.cumsum(delta_fee + delta_slip)
    result = pd.DataFrame({"date": base_ledger.date, "fixed_filled_quantity": quantity,
                           "extra_commission": delta_fee, "extra_slippage": delta_slip,
                           "cumulative_extra_cost": cumulative, "base_equity": base_ledger.equity,
                           "fixed_path_stress_equity": base_ledger.equity - cumulative,
                           "fixed_path_stress_cash": base_ledger.cash - cumulative,
                           "fixed_path_stress_cash_after_execution": base_ledger.cash_after_execution - cumulative,
                           "stress_limit_infeasible": limit_infeasible})
    result["fixed_path_stress_return"] = result.fixed_path_stress_equity / result.fixed_path_stress_equity.shift(1).fillna(config["initial_capital"]) - 1
    info = {"extra_commission": float(delta_fee.sum()), "extra_slippage": float(delta_slip.sum()),
            "fixed_path_total_extra_cost": float(cumulative[-1]),
            "fixed_path_stress_terminal_equity": float(result.fixed_path_stress_equity.iloc[-1]),
            "negative_cash_days": int((result.fixed_path_stress_cash_after_execution < -1e-7).sum()),
            "minimum_counterfactual_cash": float(result.fixed_path_stress_cash_after_execution.min()),
            "counterfactual_cash_feasible": bool((result.fixed_path_stress_cash_after_execution >= -1e-7).all()),
            "directional_limit_infeasible_days": int(limit_infeasible.sum()),
            "counterfactual_execution_feasible": bool((result.fixed_path_stress_cash_after_execution >= -1e-7).all() and not limit_infeasible.any()),
            "interpretation": "固定基础成交份额的费用归因；若现金不足，不能当作可执行策略收益"}
    return result, info


def evaluate(data: pd.DataFrame, dividends: pd.DataFrame, locked: dict, config: dict, output: Path) -> dict:
    directory = output / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    p = period(data, dividends, config["evaluation_start"], config["data_cutoff"])
    models = locked["models"]
    ledgers, economics, decision_tables, annual_rows, checks = {}, {}, {}, [], {}
    for model in ("BUY_HOLD", "SIMPLE", "FULL", "NO_REPAIR"):
        coefficients = np.asarray(models["FULL" if model == "NO_REPAIR" else model]["coefficients"]) if model != "BUY_HOLD" else None
        values = score(p.data, coefficients, model) if coefficients is not None else np.full(len(p.data), 100.0)
        if model != "BUY_HOLD":
            checks[model] = assert_batch_parity(p, values, config, model)
        for cost in ("BASE", "STRESS"):
            key = f"{model}_{cost}"
            ledger, trades, decisions = simulate_reference(p, values, config["costs"][cost], config, model)
            ledger["model"], ledger["cost"] = model, cost
            decisions["model"], decisions["cost"] = model, cost
            ledgers[key], decision_tables[key] = ledger, decisions
            economics[key] = metrics(ledger, config)
            ledger.to_parquet(directory / f"{key}_ledger.parquet", index=False)
            ledger.to_csv(directory / f"{key}_ledger.csv", index=False, encoding="utf-8-sig")
            trades.to_csv(directory / f"{key}_trades.csv", index=False, encoding="utf-8-sig")
            decisions.to_csv(directory / f"{key}_decisions.csv", index=False, encoding="utf-8-sig")
            for year, subset in ledger.groupby(ledger.date.dt.year, sort=True):
                value = metrics(subset, config)
                annual_rows.append({"model": model, "cost": cost, "year": year,
                                    "contribution_cny": float(subset.pnl.sum()),
                                    "contribution_over_initial_equity": float(subset.pnl.sum() / config["initial_capital"]), **value})
        if model != "BUY_HOLD":
            left = decision_tables[f"{model}_BASE"][["score", "target_weight"]].to_numpy(float)
            right = decision_tables[f"{model}_STRESS"][["score", "target_weight"]].to_numpy(float)
            require(np.allclose(left, right, atol=0, rtol=0, equal_nan=True), "费用情景原始分数/目标发生变化")
    friction = {}
    for model in ("BUY_HOLD", "SIMPLE", "FULL", "NO_REPAIR"):
        table, info = fixed_path_costs(p, ledgers[f"{model}_BASE"], config)
        actual_stress = economics[f"{model}_STRESS"]["ending_equity"]
        info["actual_stress_terminal_equity"] = actual_stress
        info["actual_stress_minus_fixed_path_stress_equity"] = actual_stress - info["fixed_path_stress_terminal_equity"]
        info["path_difference_interpretation"] = "剩余项包含资金约束、份额舍入、触发变化及相应分红路径，不能称为纯费用"
        table.to_csv(directory / f"{model}_fixed_path_costs.csv", index=False, encoding="utf-8-sig")
        friction[model] = info
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(directory / "annual_contributions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"model_cost": key, **value} for key, value in economics.items()]).to_csv(directory / "economic_comparison.csv", index=False, encoding="utf-8-sig")
    decisions = pd.concat([decision_tables[key] for key in ("FULL_BASE", "FULL_STRESS")], ignore_index=True)
    execution = pd.concat([ledgers[key] for key in ("FULL_BASE", "FULL_STRESS")], ignore_index=True)
    joined = decisions.merge(execution[["date", "cost", "model", "shares", "cash", "equity", "exposure", "filled_quantity", "execution_status"]],
                             left_on=["origin", "cost", "model"], right_on=["date", "cost", "model"], how="left", validate="one_to_one")
    joined.to_csv(output / "每日决策表.csv", index=False, encoding="utf-8-sig")
    event(output, "主评价连续账户完成", calendar_days=p.n, model_cost_paths=len(ledgers))
    statistics = robust_statistics(ledgers, config)
    gates, increments = {}, {}
    for cost in ("BASE", "STRESS"):
        full, simple, hold = (economics[f"{model}_{cost}"] for model in ("FULL", "SIMPLE", "BUY_HOLD"))
        gates[f"{cost}_sharpe_at_least_1_2"] = full["net_sharpe"] is not None and full["net_sharpe"] >= config["high_sharpe_target"]
        gates[f"{cost}_cagr_above_buy_hold"] = full["annualized_return"] > hold["annualized_return"]
        gates[f"{cost}_cagr_above_simple"] = full["annualized_return"] > simple["annualized_return"]
        gates[f"{cost}_utility_above_simple"] = full["utility_gamma5"] > simple["utility_gamma5"]
        gates[f"{cost}_turnover_at_most_12"] = full["annualized_two_way_turnover"] <= config["maximum_annual_two_way_turnover"] + 1e-12
        increments[cost] = {"full_minus_simple_cagr": full["annualized_return"] - simple["annualized_return"],
                            "full_minus_simple_utility": full["utility_gamma5"] - simple["utility_gamma5"],
                            "full_minus_buy_hold_cagr": full["annualized_return"] - hold["annualized_return"],
                            "full_minus_no_repair_cagr": full["annualized_return"] - economics[f"NO_REPAIR_{cost}"]["annualized_return"]}
    gates["all_account_identities"] = all(value["maximum_accounting_error"] < 1e-6 for value in economics.values())
    passed = all(gates.values())
    result = {"study_id": config["study_id"], "created_at": now(),
              "state": "HISTORICAL_ACCEPTANCE_PASSED_RESEARCH_ONLY" if passed else "REJECTED_FROZEN_CONDITIONAL_SCORE_POLICY_V1",
              "acceptance_passed": passed, "gates": gates, "economics": economics, "increments": increments,
              "main_start": p.data.date.iloc[1], "main_end": p.data.date.iloc[-1], "main_calendar_days": p.n,
              "frozen_coefficients": {key: value["named_coefficients"] for key, value in models.items()},
              "selected_configurations": {key: {"gamma": value["gamma"], "lambda": value["lambda"]} for key, value in models.items()},
              "fixed_path_cost_attribution": friction, "robust_statistics": statistics,
              "account_engine_parity": checks, "terminal_valuation": "截止日收盘估值；未扣未来退出费用",
              "evidence_class": config["evidence_class"], "future_data_used_for_training": False,
              "main_evaluation_used_for_selection": False, "position_impact": 0,
              "live_trading_authorized": False, "order_generation_authorized": False,
              "paper_shadow_authorized": False, "retained_future_model": None,
              "failed_branch_rescue_allowed": False}
    if passed:
        event(output, "历史验收通过_拟合研究留存模型", future_data_training=False)
        full = models["FULL"]
        retained = fit(period(data, dividends, config["train_start"], config["data_cutoff"]),
                       "FULL", full["gamma"], full["lambda"], config, pd.Timestamp(config["data_cutoff"]).year)
        write_json(output / "retained_research_model.json", retained, exclusive=True)
        result["retained_future_model"] = identity(output / "retained_research_model.json")
    write_json(output / "result.json", result, exclusive=True)
    plot_paths(ledgers, config, output)
    write_report(result, locked, annual, config, output)
    event(output, "研究裁决完成", state=result["state"], acceptance_passed=passed)
    return result


def plot_paths(ledgers: dict, config: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = [font.name for font in font_manager.fontManager.ttflist]
    plt.rcParams["font.sans-serif"] = [name for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"] if name in fonts]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"FULL": "#b54b32", "SIMPLE": "#30688f", "BUY_HOLD": "#505854", "NO_REPAIR": "#ad923c"}
    labels = {"FULL": "完整评分", "SIMPLE": "简单评分", "BUY_HOLD": "买入持有", "NO_REPAIR": "完整去修复"}
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 0.8]})
    fig.patch.set_facecolor("#faf9f5")
    for col, cost in enumerate(("BASE", "STRESS")):
        for model in ("BUY_HOLD", "SIMPLE", "NO_REPAIR", "FULL"):
            ledger = ledgers[f"{model}_{cost}"]
            nav = ledger.equity.to_numpy(float) / config["initial_capital"]
            drawdown = nav / np.maximum.accumulate(np.r_[1, nav])[1:] - 1
            axes[0, col].plot(ledger.date, nav, label=labels[model], color=colors[model], linewidth=1.8 if model == "FULL" else 1.1)
            axes[1, col].plot(ledger.date, drawdown * 100, color=colors[model], linewidth=1.0)
        full = ledgers[f"FULL_{cost}"]
        axes[2, col].fill_between(full.date, full.exposure * 100, color=colors["FULL"], alpha=0.35)
        axes[0, col].set_title("基础成本" if cost == "BASE" else "压力成本", loc="left", fontweight="bold")
        axes[0, col].legend(ncol=2, frameon=False, fontsize=9)
        axes[0, col].set_ylabel("净权益 / 初始资金")
        axes[1, col].set_ylabel("回撤（%）")
        axes[2, col].set_ylabel("完整评分暴露（%）")
        for row in range(3):
            axes[row, col].grid(alpha=0.15)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    fig.suptitle("510300 连续条件评分持仓 V1｜2020—2026-08-14 冻结历史回放", x=0.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.07, 0.025, "包含所有交易日、实际费用与独立分红现金；历史曾被研究观察。图示为模拟账户，非实际持仓。", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.95))
    fig.savefig(output / "连续账户比较.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def pct(value: float) -> str:
    return f"{value:.2%}"


def fmt(value: float | None) -> str:
    return "不可验收" if value is None else f"{value:.3f}"


def write_report(result: dict, locked: dict, annual: pd.DataFrame, config: dict, output: Path) -> None:
    economics = result["economics"]
    full_base, full_stress = economics["FULL_BASE"], economics["FULL_STRESS"]
    lines = ["# 510300 连续条件评分持仓 V1 研究结果", "",
             f"**裁决：{'历史验收通过，仅限研究证据' if result['acceptance_passed'] else '未通过验收，完整负结果冻结保留'}。** 完整模型基础净夏普{fmt(full_base['net_sharpe'])}，压力净夏普{fmt(full_stress['net_sharpe'])}；验收线始终为1.2。", "",
             f"主评价为{pd.Timestamp(result['main_start']).date()}至{pd.Timestamp(result['main_end']).date()}，共{result['main_calendar_days']}个连续交易日。初始20万元，只模拟510300.SH和人民币现金；2015—2019完成训练，主评价期不重训。末日收盘估值，终值仍可持有ETF，未扣未来退出费用。年化统一242天，现金利息为零。", "",
             "本轮使用已存在且已核验到2026-08-14的行情和分红，因此截止日早于本次执行日期。数据为历史下载/更正版本，不是逐日原始发布快照；这段历史已被多次研究观察，不能称为完全未见样本或未来收益证明。", "",
             "## 完整经济比较", "",
             "| 模型 | 成本 | 净夏普 | 复合年化净收益 | 算术年化净收益 | 年化波动 | 最大回撤 | 平均暴露 | 年化双向换手 | 总摩擦/元 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"BUY_HOLD": "买入持有", "SIMPLE": "简单评分", "FULL": "完整评分", "NO_REPAIR": "完整去修复"}
    for model in names:
        for cost in ("BASE", "STRESS"):
            m = economics[f"{model}_{cost}"]
            lines.append(f"| {names[model]} | {'基础' if cost == 'BASE' else '压力'} | {fmt(m['net_sharpe'])} | {pct(m['annualized_return'])} | {pct(m['annualized_arithmetic_mean'])} | {pct(m['annualized_volatility'])} | {pct(m['max_drawdown'])} | {pct(m['mean_exposure'])} | {m['annualized_two_way_turnover']:.2f} | {m['total_friction']:,.2f} |")
    lines += ["", "年化双向换手为每日成交额/成交前权益的合计乘242/交易日数，全仓买入和全仓卖出各计1倍。佣金和滑点含最低收费及不利价位舍入。", "", "## 冻结模型与开发", "",
              f"同一九组gamma/lambda配置用于完整和简单两个模型，各3个内层验证年，共54次内层拟合和2次2015—2019重估；总实际候选评估{locked['total_candidate_evaluations']:,}次。未追加种子、窗口、交易规则或优化预算。", ""]
    for model in ("FULL", "SIMPLE"):
        fitted = locked["models"][model]
        lines += [f"**{names[model]}**：gamma={fitted['gamma']:g}，lambda={fitted['lambda']:g}；内层选择分数{fitted['selection']['selection_score']:.6f}；最终优化状态`{fitted['termination']}`。", "",
                  "| 系数 | 冻结值 |", "|---|---:|"]
        lines.extend(f"| {key} | {value:.9f} |" for key, value in fitted["named_coefficients"].items())
        lines.append("")
    lines += ["预算耗尽时使用已登记预算内最优可行解，不把它称为全局最优；无后续种子救援。消融仅把完整模型R系数置零，其余保持不变，不重训。", "",
              "## 验收与增量", "", "| 固定验收项 | 结果 |", "|---|---|"]
    lines.extend(f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in result["gates"].items())
    for cost in ("BASE", "STRESS"):
        delta = result["increments"][cost]
        lines += ["", f"{'基础' if cost == 'BASE' else '压力'}成本：完整相对简单的复合年化差{delta['full_minus_simple_cagr']*100:+.3f}个百分点，固定gamma=5效用差{delta['full_minus_simple_utility']:+.6f}；相对买入持有的复合年化差{delta['full_minus_buy_hold_cagr']*100:+.3f}个百分点。保留修复项相对删除修复项的复合年化差{delta['full_minus_no_repair_cagr']*100:+.3f}个百分点。"]
    lines += ["", "## 年度贡献", "", "| 年份 | 基础完整净收益 | 压力完整净收益 | 基础买入持有 | 压力买入持有 | 基础完整净损益/元 |", "|---|---:|---:|---:|---:|---:|"]
    for year in sorted(annual.year.unique()):
        def row(model, cost):
            return annual.loc[(annual.year == year) & (annual.model == model) & (annual.cost == cost)].iloc[0]
        b, s, hb, hs = row("FULL", "BASE"), row("FULL", "STRESS"), row("BUY_HOLD", "BASE"), row("BUY_HOLD", "STRESS")
        lines.append(f"| {year} | {pct(b.cumulative_return)} | {pct(s.cumulative_return)} | {pct(hb.cumulative_return)} | {pct(hs.cumulative_return)} | {b.contribution_cny:,.2f} |")
    lines += ["", "2026年为截至8月14日的部分年度；年度净收益未再年化。每年损益衔接同一账户，未拼接内层模型。", "", "## 固定路径费用及不确定性", ""]
    attr = result["fixed_path_cost_attribution"]["FULL"]
    lines += [f"完整模型基础实际成交份额在压力费用下，额外佣金{attr['extra_commission']:,.2f}元、额外滑点{attr['extra_slippage']:,.2f}元，合计{attr['fixed_path_total_extra_cost']:,.2f}元。压力账户真实终值与固定份额压力反事实终值相差{attr['actual_stress_minus_fixed_path_stress_equity']:,.2f}元，后者包含账户路径变化，不能归为纯费用。反事实现金不足日数{attr['negative_cash_days']}，最低现金{attr['minimum_counterfactual_cash']:,.2f}元。", ""]
    for cost in ("BASE", "STRESS"):
        stats = result["robust_statistics"]["models"][f"FULL_{cost}"]
        lo, hi = stats["sharpe_95_interval"]
        delta = result["robust_statistics"]["paired_increments"][f"FULL_MINUS_SIMPLE_{cost}"]
        dlo, dhi = delta["cagr_difference_95_interval"]
        lines.append(f"{'基础' if cost == 'BASE' else '压力'}完整净夏普的20日区块95%区间[{lo:.3f}, {hi:.3f}]；相对简单复合年化差区间[{dlo*100:.3f}, {dhi*100:.3f}]个百分点。20阶Newey-West校正夏普{fmt(stats['hac_adjusted_sharpe'])}、均值t值{fmt(stats['hac_mean_t'])}。")
    lines += ["", "上述2000次区块抽样使用相同日期联合抽取所有模型，只刻画给定模型和样本的不确定性，不是未来达标概率，也不清除历次搜索造成的选择偏差。", "", "## 复现与交付", "",
              "- `每日决策表.csv`：五项分数、总分、原始目标、执行规则调整目标、实际份额/暴露、未交易原因、请求原点与下一执行日期；末日请求仅作模拟记录，不执行到样本外。",
              "- `development/all_inner_results.csv`及JSON：54次内层完整结果；`configuration_ranking.csv`：两模型各九组排名；`model_freeze.json`：主评价读取前的模型与开发文件哈希。",
              "- `evaluation/economic_comparison.csv`、`annual_contributions.csv`、8条完整账户/成交/决策路径及固定费用路径；`result.json`：所有指标、验收项与统计区间。",
              "- `freeze_manifest.json`、`run_claim.json`、`stage_events.jsonl`、`frozen_sources/`、`frozen_inputs/`：本轮协议、一次运行声明、阶段证据和小型原始输入副本。",
              "", "训练批量实现与既有账户引擎在54个验证终解、两个2015—2019最终模型以及主评价模型逐日核对，股数和成交份额要求完全一致，现金/权益误差小于1e-6元。真实路径使用既有Account和execute_order；没有导入旧失败模型的预测或持仓。", "",
              f"状态为`{result['state']}`。{'按冻结保留条件另存一次全历史研究模型。' if result['acceptance_passed'] else '未满足保留条件，不拟合全历史未来模型，不晋升简单或消融模型，不营救本分支。'} `POSITION_IMPACT=0`，未创建Paper/Shadow、自动化、券商连接或真实订单。", "",
              "![连续账户比较](连续账户比较.png)", "",
              "具体公式、时钟、成本、优化规则与方法来源见冻结协议；本轮只执行必要的合同、会计和复现检查。", ""]
    (output / "研究报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="510300连续条件评分持仓V1冻结研究")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    if args.command == "freeze":
        freeze(config, output)
        return 0
    verify_freeze(output)
    write_json(output / "run_claim.json", {"created_at": now(), "run_number": 1,
               "state": "ONE_HISTORICAL_RUN_CONSUMED", "main_evaluation_read": False}, exclusive=True)
    try:
        event(output, "一次历史运行开始", study_id=config["study_id"])
        data, dividends, _ = load_data(config, config["train_cutoff"], output, "development")
        locked = develop(data, dividends, config, output)
        del data
        verify_freeze(output)
        data, dividends, _ = load_data(config, config["data_cutoff"], output, "main")
        result = evaluate(data, dividends, locked, config, output)
        verify_freeze(output)
        write_json(output / "execution_receipt.json", {"created_at": now(), "status": "COMPLETED",
                   "study_state": result["state"], "result": identity(output / "result.json"),
                   "model_freeze": identity(output / "model_freeze.json"), "historical_run_count": 1,
                   "main_evaluation_used_for_selection": False, "position_impact": 0}, exclusive=True)
        return 0
    except Exception as exc:
        event(output, "程序失败_不自动追加研究运行", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "failure_receipt.json", {"created_at": now(), "state": "PROGRAM_FAILED",
                   "error": str(exc), "traceback": traceback.format_exc()}, exclusive=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
