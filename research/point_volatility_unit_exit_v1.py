"""隔离的继续收益单位实验：风险单位拟合、原收益单位检验，保留冻结对照。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import types

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, state_values, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController, prediction_identity


OUT = ROOT / "reports/research/510300_point_volatility_unit_exit_v1"
CURRENT = Path("reports/research/510300_point_current_observation_20261001")
WEIGHT = Path("reports/research/510300_point_second_weight_comparison_v1/inputs")
STUDY = "510300_POINT_VOLATILITY_UNIT_EXIT_V1"
CONFIG_NAMES = ("entry_vintage_exit", "continuous_reference_min_variance", "downside_reference_risk",
                "joint_downside_reference_pair", "trend_noise_reference_blend")


def fit_volatility_unit(rows, cfg):
    """只改变拟合标签的单位；保持八项输入、周期等权、岭惩罚和截断。"""
    sigma = rows.vol20.to_numpy(float)
    require(np.isfinite(sigma).all() and (sigma > 0).all(), "风险单位拟合缺少正的当时波动，不删行或补值。")
    normalized = rows.copy()
    normalized["target"] = rows.target.to_numpy(float) / sigma
    model = fit_within_cycle_exit(normalized, cfg)
    return model


def predict_original_return(model, values):
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all() and x[-1] > 0, "风险单位预测缺少原八项或当时正波动。")
    return within_cycle_prediction(model, x) * float(x[-1])


class VolatilityUnitController(EntryVintageExitController):
    """原实际买入日首次收盘锁定版本；只替换预测的单位及恢复步骤。"""

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "风险单位模型必须在实际入场首次收盘选择。")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t) - 1
            self.record = self.models[k] if k >= 0 else None
            self.identity = "VOLATILITY_UNIT:" + prediction_identity(self.record)
            self.negative_count = 0
        x = state_values(self.data, t, cycle, current_value, peak_value)
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        record = self.record
        if record and record["status"] == "FIT_COMPLETE":
            require(record["target_unit"] == "RETURN_PER_KNOWN_ANNUAL_VOL20", "风险单位模型身份不同。")
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index <= t, "风险单位模型时钟读取了未来。")
            if np.isfinite(x).all() and x[-1] > 0:
                value, status = predict_original_return(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES_OR_VOLATILITY"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count + 1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status,
                "continuation_prediction": value, "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "model_selection_index": self.selection_index, "model_selection_origin": self.data.date.iloc[self.selection_index],
                "fixed_prediction_identity": self.identity, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days,
                "target_unit": "RETURN_PER_KNOWN_ANNUAL_VOL20", **dict(zip(FEATURES, x))}


def model_at_entry(records, entry_index):
    indexes = [r["fit_index"] for r in records]
    require(indexes == sorted(set(indexes)), "月度模型时点非唯一递增。")
    k = bisect_right(indexes, int(entry_index)) - 1
    return records[k] if k >= 0 else None


def paired_predictions(samples, cycles, originals, candidates):
    """两个版本对同一自然参考周期的同一原点预测，买入首次收盘锁定版本。"""
    entry = cycles.set_index("cycle_id").entry_index.to_dict()
    rows = []
    for row in samples.itertuples():
        selection = int(entry[row.cycle_id])
        original = model_at_entry(originals, selection)
        candidate = model_at_entry(candidates, selection)
        old, new, status = np.nan, np.nan, "NO_VIEW_NO_MODEL_AT_REFERENCE_ENTRY"
        x = np.array([getattr(row, name) for name in FEATURES], float)
        if original and original["status"] == "FIT_COMPLETE":
            require(candidate and candidate["fit_index"] == original["fit_index"], "同原点模型版本不匹配。")
            require(original["latest_exit_index"] <= original["fit_index"] <= selection <= row.origin_index < row.early_exit_index < row.exit_index,
                    "预测、版本选择或标签成熟时钟错误。")
            old = within_cycle_prediction(original["model"], x)
            if candidate["status"] == "FIT_COMPLETE" and np.isfinite(x).all() and x[-1] > 0:
                new = predict_original_return(candidate["model"], x)
                status = "PAIRED_PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_CANDIDATE_MISSING_UNIT_OR_MODEL"
        rows.append({"cycle_id": int(row.cycle_id), "origin": row.origin, "origin_index": int(row.origin_index),
                     "entry_index": selection, "fit_index": original["fit_index"] if original else None,
                     "fit_origin": original["fit_origin"] if original else None, "mature_date": row.mature_date,
                     "target_original_return": float(row.target), "known_vol20": float(row.vol20),
                     "baseline_prediction": old, "candidate_prediction": new, "status": status})
    return pd.DataFrame(rows)


def paired_year_bootstrap(cycles, seed=51030067, iterations=5000):
    """整年块重采样；只作开发资料敏感性检查，不宣称年份独立。"""
    years = sorted(cycles.year.unique())
    if len(years) < 2:
        return {"status": "NOT_COMPUTED_FEWER_THAN_TWO_YEAR_BLOCKS", "lower_95": None, "upper_95": None}
    groups = [cycles.loc[cycles.year.eq(year), "raw_mse_improvement"].to_numpy(float) for year in years]
    rng = np.random.default_rng(seed)
    values = np.array([np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(iterations)])
    return {"status": "COMPUTED_DEVELOPMENT_YEAR_BLOCK_SENSITIVITY", "years": [int(y) for y in years],
            "iterations": iterations, "lower_95": float(np.quantile(values, .025)), "upper_95": float(np.quantile(values, .975)),
            "independence_established": False}


def source_paths():
    paths = [CURRENT / "results/training_reference/samples.parquet", CURRENT / "results/training_reference/cycles.parquet",
             CURRENT / "inputs/within_models.json", CURRENT / "inputs/ordinary_models.json", CURRENT / "inputs/candidate_features.parquet",
             CURRENT / "inputs/dividends.csv", CURRENT / "inputs/config/within_cycle_exit.json",
             Path("reports/research/510300_point_core_observation_v1/inputs/within_models.json"),
             Path("reports/research/510300_point_core_observation_v1/results/earlier_diagnostic/STRESS/full_factors.parquet"),
             CURRENT / "results/STRESS/full_factors.parquet", WEIGHT / "prices.parquet", WEIGHT / "risks.parquet"]
    paths.extend(CURRENT / f"inputs/config/{name}.json" for name in CONFIG_NAMES)
    for name in ("PANIC_ONLY", "REARM_RIDGE"):
        paths.extend(CURRENT / f"results/continuous/{name}_{kind}.parquet" for kind in ("ledger", "decisions", "cycles"))
    for period in ("2015_2019", "2020_2026"):
        for cost in ("BASE", "STRESS"):
            folder = WEIGHT / f"controls/{period}/{cost}/A_SAVED_WEIGHT"
            paths.extend(folder / f"{name}.parquet" for name in ("daily", "orders", "trades", "decisions"))
    paths.extend(Path(f"research/{name}.py") for name in (
        "point_volatility_unit_exit_v1", "point_account_cashflow_state_v1", "point_core_observation_inputs_v1",
        "point_weight_information_inputs_v1", "point_account_nr7_inputs_v1", "daily_supply_test_v1",
        "entry_vintage_exit_inputs_v1", "within_cycle_exit_inputs_v1", "learned_cycle_exit_v1",
        "reference_observation_accounts_v1", "point_state_reconstruction_v1", "point_binary_rebalance_diagnostic_v1",
        "point_second_weight_comparison_v1", "upward_episode_anatomy_v1"))
    paths.extend([Path("tests/test_point_volatility_unit_exit_v1.py"),
                  Path("reports/research/510300_point_method_intake_20261002/prior_direction_review.json")])
    return paths


def freeze():
    require(not OUT.exists(), "风险单位实验已存在，不覆盖冻结研究。")
    sources = [{"path": path.as_posix(), "sha256": digest(ROOT / path)} for path in source_paths()]
    protocol = {
        "study": STUDY, "frozen_at": now(), "linked_question": "E05_新拟合单位方法，非E02新增市场字段",
        "hypothesis": "原继续收益百分比在不同波动环境的幅度差异可能影响有限样本估计；以原点已知vol20缩放训练标签并恢复原收益单位可能改善同日预测。",
        "single_change": "y_unit=y_original/known_vol20；使用原八项周期内岭拟合；预测原收益=known_current_vol20*fitted_unit_prediction。",
        "original_economic_target_unchanged": "原自然参考退出相对下一开盘退出的含权益费用继续收益；不改自然标签终点，不使用未来波动或未来持有天数缩放。",
        "fixed_training": {"recent_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100,
                           "ridge_alpha": 1., "feature_clip": 5., "each_cycle_total_weight": 1.},
        "model_clock": "原保存142个月首时点及原成熟成员；实际买入首次收盘固定版本，两日严格负预测确认；无成熟模型时原价格保护保留。",
        "unit_failure": "任一训练vol20缺失/非正则该模型NO_VIEW，不删训练行，不设置新波动下限。",
        "prediction_control": "全部保存FIT_COMPLETE模型按原标签复算，参数误差<=1e-12；原模型作为冻结对照。预先3个纯控制复算已一致，记入控制工作，不算新候选效果。",
        "prediction_gate": "同一自然参考原点、同一入场版本，按自然周期等权计算原收益单位MSE；2015-2019与2020-2026各自候选MSE严格低于原模型且整年块重采样改进95%下界>0；缺旧预测保留NO_VIEW。",
        "uncertainty": {"bootstrap_iterations": 5000, "seed": 51030067, "block": "整年，同一自然周期不拆行", "history_role": "DEVELOPMENT_CALIBRATION_NOT_INDEPENDENT"},
        "economic_stage_if_gate_passes": "只替换A核心固定版本退出预测，普通及急跌来源不改；重建所有受影响参考/风险预算并通过原共同执行器。BASE账户继续使用STRESS来源目标，与原四A对照逐表核对。",
        "economic_contract": "两独立20万元时期，252日年化，50%上限、10百分点调仓带、原ES/跳空/回撤预算、T+1/100份/0.001刻度、原两费用和权益；无强制终点退出。",
        "economic_gate": "四场景净CAGR和全日历净Sharpe同时严格提高，实际完成周期pB>1且平均净收益>0；回撤不得超过10%；完整逐年次数及毛损益/费用均报告。",
        "failure_exit": "预测门失败即不运行经济账户；经济门失败则拒绝固定版本，不换波动窗口、alpha、阈值、缩放指数或入场版本营救。原策略和E03完全保留。",
        "method_limit": "此法改变均值函数与拟合损失的尺度，并非已证明的误差方差估计或纯WLS；当时收益波动不自动等于继续收益预测误差标准差。",
        "method_reference": "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html，仅用于误差方差权重的定义边界，不是本方法盈利证据。",
        "new_market_information": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "orders_authorized": False, "sources": sources,
    }
    OUT.mkdir(parents=True)
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"protocol_sha256": digest(OUT / "protocol.json"), "sources": sources}, exclusive=True)
    print("E05风险单位方法已冻结；尚未读取候选预测或账户收益。", flush=True)


def check():
    saved = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT / "protocol.json") == saved["protocol_sha256"], "风险单位协议变化。")
    for item in saved["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结来源变化：" + item["path"])
    return len(saved["sources"])


def save_table(name, frame):
    folder = OUT / "results"
    folder.mkdir(exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")


def fit_and_evaluate():
    require(not (OUT / "prediction_summary.json").exists(), "本固定预测实验已有结果，不重跑。")
    check()
    samples = pd.read_parquet(ROOT / CURRENT / "results/training_reference/samples.parquet")
    cycles = pd.read_parquet(ROOT / CURRENT / "results/training_reference/cycles.parquet")
    original = json.loads((ROOT / CURRENT / "inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    cfg = json.loads((ROOT / CURRENT / "inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    for key, expected in (("recent_cycles", 20), ("minimum_cycles", 10), ("minimum_rows", 100), ("ridge_alpha", 1.), ("feature_clip", 5.)):
        require(cfg[key] == expected, "原拟合设置发生变化：" + key)
    candidate, verification = [], []
    for record in original:
        rows, ids = training_rows(samples, record["fit_index"], cfg)
        require(ids == record["training_cycles"] and len(rows) == record["training_rows"], "原成熟训练成员无法复现。")
        new = {**record, "target_unit": "RETURN_PER_KNOWN_ANNUAL_VOL20", "model": None}
        if record["status"] == "FIT_COMPLETE":
            require(rows.exit_index.le(record["fit_index"]).all(), "训练标签没有成熟。")
            baseline = fit_within_cycle_exit(rows, cfg)
            error = max(float(np.max(np.abs(np.asarray(baseline[k]) - np.asarray(record["model"][k])))) for k in ("mean", "scale", "coefficients", "intercept"))
            require(error <= 1e-12, "保存对照模型不能复算。")
            verification.append({"fit_origin": record["fit_origin"], "rows": len(rows), "cycles": len(ids), "max_model_error": error})
            sigma = rows.vol20.to_numpy(float)
            if np.isfinite(sigma).all() and (sigma > 0).all():
                new["model"] = fit_volatility_unit(rows, cfg)
            else:
                new["status"] = "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"
        candidate.append(new)
    write_json(OUT / "candidate_models.json", {"models": candidate, "evidence_class": "DEVELOPMENT_VOLATILITY_UNIT_FITS"}, exclusive=True)
    predictions = paired_predictions(samples, cycles, original, candidate)
    available = predictions.loc[predictions.status.eq("PAIRED_PREDICTION_AVAILABLE")].copy()
    available["baseline_sq_error"] = (available.baseline_prediction - available.target_original_return).pow(2)
    available["candidate_sq_error"] = (available.candidate_prediction - available.target_original_return).pow(2)
    available["baseline_unit_sq_error"] = available.baseline_sq_error / available.known_vol20.pow(2)
    available["candidate_unit_sq_error"] = available.candidate_sq_error / available.known_vol20.pow(2)
    entry_dates = cycles.set_index("cycle_id").entry_date
    available["entry_date"] = available.cycle_id.map(entry_dates)
    available["year"] = available.entry_date.dt.year
    # 跨评价边界的周期按原点分时期；本源没有跨2019/2020持仓周期，仍显式分组。
    available["period"] = np.where(available.origin.lt(pd.Timestamp("2020-01-01")), "2015_2019", "2020_2026")
    by_cycle = available.groupby(["period", "cycle_id"]).agg(rows=("origin", "size"), year=("year", "first"),
        baseline_mse=("baseline_sq_error", "mean"), candidate_mse=("candidate_sq_error", "mean"),
        baseline_unit_mse=("baseline_unit_sq_error", "mean"), candidate_unit_mse=("candidate_unit_sq_error", "mean")).reset_index()
    by_cycle["raw_mse_improvement"] = by_cycle.baseline_mse - by_cycle.candidate_mse
    summaries = []
    for period in ("2015_2019", "2020_2026"):
        group = by_cycle.loc[by_cycle.period.eq(period)]
        ci = paired_year_bootstrap(group)
        old, new = float(group.baseline_mse.mean()), float(group.candidate_mse.mean())
        passed = len(group) > 0 and new < old and ci["lower_95"] is not None and ci["lower_95"] > 0
        summaries.append({"period": period, "natural_cycles": len(group), "prediction_rows": int(group.rows.sum()),
                          "baseline_raw_mse": old, "candidate_raw_mse": new, "raw_mse_improvement": old - new,
                          "relative_raw_mse_change": new / old - 1 if old > 0 else None,
                          "baseline_unit_mse": float(group.baseline_unit_mse.mean()), "candidate_unit_mse": float(group.candidate_unit_mse.mean()),
                          "year_block_sensitivity": ci, "passes_prediction_gate": bool(passed)})
    passed = all(item["passes_prediction_gate"] for item in summaries)
    save_table("同原点配对预测", predictions)
    save_table("逐自然周期误差", by_cycle)
    save_table("原模型参数复算", pd.DataFrame(verification))
    summary = {"study": STUDY, "at": now(), "status": "PREDICTION_GATE_PASSED_ECONOMIC_STAGE_PENDING" if passed else "REJECTED_FIXED_VOLATILITY_UNIT_METHOD_PREDICTION_GATE_FAILED",
               "prediction_gate_passed": passed, "periods": summaries, "no_view_rows": int(predictions.status.ne("PAIRED_PREDICTION_AVAILABLE").sum()),
               "model_clock_count": len(original), "baseline_recomputation_fits": len(verification), "pre_freeze_control_recomputations": 3,
               "new_candidate_fits": sum(r["model"] is not None for r in candidate), "maximum_baseline_model_error": max(v["max_model_error"] for v in verification),
               "new_strategy_accounts": 0, "account_return_sharpe": "NOT_COMPUTED", "new_bars": 0,
               "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "sources_unchanged": check()}
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(summary["status"], flush=True)
    for item in summaries:
        print(f"{item['period']}：{item['natural_cycles']}周期，原收益MSE变化{item['relative_raw_mse_change']:.6%}，预测门{item['passes_prediction_gate']}。", flush=True)


def economic_stage():
    """预测门通过才接通原完整来源链与共同账户，避免读取失败方法账户来救回。"""
    require(not (OUT / "economic_summary.json").exists(), "经济阶段已有结果，不重跑。")
    check()
    pred = json.loads((OUT / "prediction_summary.json").read_text(encoding="utf-8"))
    require(pred["prediction_gate_passed"], "预测增量门失败，经济账户阶段禁止运行。")
    from research import point_core_observation_inputs_v1 as engine
    from research.point_weight_information_inputs_v1 import weight_account
    from research.point_account_nr7_inputs_v1 import PARENT_A, metrics as account_metrics
    from research.point_second_weight_comparison_v1 import annual_rows
    from research import upward_episode_anatomy_v1 as common
    from research.adaptive_allocation_v1 import normalize_dividends
    data = pd.read_parquet(ROOT / CURRENT / "inputs/candidate_features.parquet")
    prices = pd.read_parquet(ROOT / WEIGHT / "prices.parquet")
    risks = pd.read_parquet(ROOT / WEIGHT / "risks.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / CURRENT / "inputs/dividends.csv"))
    execution_data, _ = common.features(prices, dividends)
    configs = {name: json.loads((ROOT / CURRENT / f"inputs/config/{name}.json").read_text(encoding="utf-8")) for name in CONFIG_NAMES}
    originals = json.loads((ROOT / CURRENT / "inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    candidate = json.loads((OUT / "candidate_models.json").read_text(encoding="utf-8"))["models"]
    # 新函数独立持有原函数全局表的副本，不修改原模块或磁盘源文件。
    globals_copy = {**engine.assemble_chain.__globals__, "EntryVintageExitController": VolatilityUnitController}
    modified_chain = types.FunctionType(engine.assemble_chain.__code__, globals_copy, "assemble_volatility_unit_chain", engine.assemble_chain.__defaults__, engine.assemble_chain.__closure__)
    metrics, checks, annual = [], [], []
    for period, start, end, next_date in (("2015_2019", "2015-01-05", "2019-12-31", "2020-01-02"),
                                          ("2020_2026", "2020-01-02", "2026-09-30", "2026-10-08")):
        frame = data.loc[data.date.le(pd.Timestamp(end))].copy().reset_index(drop=True)
        continuous = {}
        for name in ("PANIC_ONLY", "REARM_RIDGE"):
            ledger, decisions, cycles = [pd.read_parquet(ROOT / CURRENT / f"results/continuous/{name}_{kind}.parquet") for kind in ("ledger", "decisions", "cycles")]
            continuous[name] = (ledger.loc[ledger.date.le(pd.Timestamp(end))], decisions.loc[decisions.origin.le(pd.Timestamp(end))], cycles)
        original_models = [r for r in originals if r["fit_index"] < len(frame)]
        candidate_models = [r for r in candidate if r["fit_index"] < len(frame)]
        baseline = engine.assemble_chain(frame, dividends, configs, original_models, start, pd.Timestamp(next_date), continuous)
        changed = modified_chain(frame, dividends, configs, candidate_models, start, pd.Timestamp(next_date), continuous)
        saved_path = Path("reports/research/510300_point_core_observation_v1/results/earlier_diagnostic/STRESS/full_factors.parquet") if period == "2015_2019" else CURRENT / "results/STRESS/full_factors.parquet"
        saved = pd.read_parquet(ROOT / saved_path)
        np.testing.assert_allclose(baseline["factors"]["STRESS"][PARENT_A], saved[PARENT_A], atol=1e-11, rtol=0, equal_nan=True)
        for name, chain in (("CONTROL", baseline), ("VOLATILITY_UNIT", changed)):
            parents = chain["factors"]["STRESS"][["date", PARENT_A]].rename(columns={"date": "origin"})
            for cost in ("BASE", "STRESS"):
                account = weight_account(execution_data.loc[execution_data.date.le(pd.Timestamp(end))].reset_index(drop=True), dividends, parents, risks, cost, start)
                if name == "CONTROL":
                    folder = ROOT / WEIGHT / f"controls/{period}/{cost}/A_SAVED_WEIGHT"
                    for table_name in ("daily", "orders", "trades", "decisions"):
                        pd.testing.assert_frame_equal(account[table_name], pd.read_parquet(folder / (table_name + ".parquet")), check_exact=False, atol=1e-9, rtol=0)
                    checks.append({"period": period, "cost": cost, "saved_account_match": True})
                folder = OUT / f"accounts/{period}/{cost}/{name}"
                folder.mkdir(parents=True)
                for table_name, table in account.items():
                    if isinstance(table, pd.DataFrame):
                        table.to_parquet(folder / (table_name + ".parquet"), index=False)
                    else:
                        write_json(folder / (table_name + ".json"), table)
                y = annual_rows(account, period, name, cost)
                m = account_metrics(account)
                counts = [row["completed_cycles"] for row in y if row["full_year"]]
                m["average_full_year_cycles"] = float(np.mean(counts))
                m["zero_trade_full_years"] = sum(n == 0 for n in counts)
                annual.extend(y)
                metrics.append({"period": period, "cost": cost, "candidate": name, **m})
        save_table("候选完整来源_" + period, changed["factors"]["STRESS"])
    save_table("共同账户比较", pd.DataFrame(metrics))
    save_table("逐年净收益与次数", pd.DataFrame(annual))
    comparisons = []
    for period in ("2015_2019", "2020_2026"):
        for cost in ("BASE", "STRESS"):
            pair = {r["candidate"]: r for r in metrics if r["period"] == period and r["cost"] == cost}
            old, new = pair["CONTROL"], pair["VOLATILITY_UNIT"]
            delta_cagr, delta_sharpe = new["net_cagr"] - old["net_cagr"], new["net_sharpe"] - old["net_sharpe"]
            quality = new["p_times_b"] > 1 and new["mean_cycle_net_return"] > 0
            passed = delta_cagr > 0 and delta_sharpe > 0 and quality and new["max_drawdown"] <= .1
            comparisons.append({"period": period, "cost": cost, "net_cagr_delta": delta_cagr,
                                "net_sharpe_delta": delta_sharpe, "trade_quality_pass": bool(quality), "joint_pass": bool(passed)})
    joint_pass = all(r["joint_pass"] for r in comparisons)
    write_json(OUT / "economic_summary.json", {"study": STUDY, "at": now(), "status": "HISTORICAL_JOINT_GATE_PASS_INDEPENDENT_VALIDATION_PENDING" if joint_pass else "REJECTED_FIXED_VOLATILITY_UNIT_METHOD_ECONOMIC_GATE_FAILED",
               "metrics": metrics, "control_checks": checks, "new_strategy_accounts": 4,
               "comparisons": comparisons, "historical_joint_gate_pass": joint_pass,
               "new_control_replays": 4, "new_internal_reference_replays": 16, "sources_unchanged": check(),
               "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}, exclusive=True)
    print("四个候选共同账户已完成；完整目标仍须按冻结经济门和独立证据判定。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="隔离风险单位继续收益方法")
    parser.add_argument("action", choices=("freeze", "predict", "account", "check"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "predict":
        fit_and_evaluate()
    elif action == "account":
        economic_stage()
    else:
        print(f"冻结来源未变：{check()}份。", flush=True)


if __name__ == "__main__":
    main()
