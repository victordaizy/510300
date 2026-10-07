"""隔离A03创新高频率与加速：先冻结原成员准入，再检验两列条件退出增量。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES, state_values, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController, prediction_identity
from research.point_volatility_unit_exit_v1 import CURRENT, model_at_entry, source_paths as original_source_paths


STUDY = "510300_POINT_A03_EXIT_PREDICTION_V1"
OUT = ROOT / "reports/research/510300_point_a03_exit_prediction_v1"
FIELDS = ["new_high_frequency20", "new_high_frequency_acceleration5_15"]
INTAKE = OUT / "intake"
FEATURES = BASE_FEATURES + FIELDS
KIND = "A03_WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE"
PERIODS = [("2015_2019", "2015-01-05", "2019-12-31"),
           ("2020_2026", "2020-01-02", "2026-09-30")]


def validate_wealth_clock(data):
    """绑定原收盘和当日已除息现金的前向财富；不接受回溯价格或未知插值。"""
    require(data.symbol.eq("510300.SH").all(), "A03标的身份不同。")
    raw = data[["close", "dividend", "previous_close", "total_simple", "wealth"]].to_numpy(float)
    close, dividend, previous, returns, wealth = raw.T
    require(np.isfinite(close).all() and (close > 0).all(), "A03原收盘缺失或非正。")
    require(np.isfinite(dividend).all() and (dividend >= 0).all(), "A03现金除息事件缺失或为负。")
    require(np.isfinite(previous[1:]).all() and (previous[1:] > 0).all(), "A03前收盘缺失或非正。")
    np.testing.assert_allclose(previous[1:], close[:-1], atol=1e-12, rtol=0)
    expected = (close[1:]+dividend[1:])/previous[1:]-1
    require(np.isfinite(returns[1:]).all(), "A03非初始经济收益缺失，不填零。")
    np.testing.assert_allclose(returns[1:], expected, atol=1e-12, rtol=0)
    require(np.isfinite(wealth).all() and (wealth > 0).all(), "A03前向财富缺失或非正。")
    # 仅首个市场点归一为一；后续全部使用已发生的原价及当日现金，不填其他缺口。
    reconstructed = np.r_[1., np.cumprod(1+expected)]
    np.testing.assert_allclose(wealth, reconstructed, atol=1e-12, rtol=1e-12)


def a03_information_field(data):
    """每个严格20日新高只用当日及过去；两列共同需要40完整财富点。"""
    dates = pd.DatetimeIndex(data.date).astype("datetime64[ns]")
    require(not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
            and (dates.dayofweek <= 4).all(), "A03市场日期必须唯一递增且为工作日。")
    require(data.symbol.eq("510300.SH").all(), "A03标的身份不同。")
    wealth = pd.Series(data.wealth.to_numpy(float))
    finite_positive = np.isfinite(wealth) & wealth.gt(0)
    price = wealth.where(finite_positive)
    previous_high = price.shift(1).rolling(20, min_periods=20).max()
    known_event = finite_positive & previous_high.notna()
    new_high = price.gt(previous_high).astype(float).where(known_event)
    frequency = new_high.rolling(20, min_periods=20).mean()
    recent = new_high.rolling(5, min_periods=5).mean()
    earlier = new_high.shift(5).rolling(15, min_periods=15).mean()
    acceleration = recent-earlier
    complete = frequency.notna() & acceleration.notna()
    frequency, acceleration = frequency.where(complete), acceleration.where(complete)
    require(frequency.dropna().between(0., 1.).all()
            and acceleration.dropna().between(-1.-1e-12, 1.+1e-12).all(), "A03频率/加速超出范围。")
    indexes = np.arange(len(data))
    return pd.DataFrame({"date": dates, "origin_index": indexes,
        "known_forward_wealth": price.to_numpy(), "strict_prior20_wealth_high": previous_high.to_numpy(),
        "known_new_high_event": new_high.to_numpy(), FIELDS[0]: frequency.to_numpy(),
        "recent5_new_high_frequency": recent.to_numpy(), "earlier15_new_high_frequency": earlier.to_numpy(),
        FIELDS[1]: acceleration.to_numpy(),
        "status": np.where(complete, "A03_FORTY_COMPLETE_WEALTH_POINTS_AVAILABLE",
                           "NO_VIEW_INCOMPLETE_FORTY_WEALTH_POINTS_OR_NEW_HIGH_EVENTS"),
        "valid_daily_wealth": finite_positive.to_numpy(),
        "prior20_start_index": np.where(indexes >= 20, indexes-20, -1),
        "full_field_earliest_source_index": np.where(indexes >= 39, indexes-39, -1),
        "latest_source_index": indexes})


def fit_a03_within(rows, cfg):
    """原固定周期截距岭回归增加已登记两列；不删行，不改变权重或正则化。"""
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(),
            "A03完整训练输入缺失，禁止删行或补值。")
    require(np.isfinite(weights).all() and (weights > 0).all(), "A03周期训练权重无效。")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    dx, dy, groups = np.empty_like(z), np.empty_like(y), []
    ids = rows.cycle_id.to_numpy(int)
    for cycle_id in sorted(set(ids)):
        mask = ids == cycle_id
        require(abs(weights[mask].sum()-1) < 1e-12, "每个自然成熟周期必须等总权重一。")
        mz = np.average(z[mask], axis=0, weights=weights[mask])
        my = float(np.average(y[mask], weights=weights[mask]))
        dx[mask], dy[mask] = z[mask]-mz, y[mask]-my
        groups.append({"cycle_id": int(cycle_id), "rows": int(mask.sum()),
                       "standardized_feature_mean": mz.tolist(), "target_mean": my})
    fitted = Ridge(alpha=cfg["ridge_alpha"], solver="svd", fit_intercept=False)
    fitted.fit(dx, dy, sample_weight=weights)
    require(np.isfinite(fitted.coef_).all(), "A03回归系数无效。")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"] -
                                         np.asarray(group["standardized_feature_mean"])@fitted.coef_)
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(np.mean([g["cycle_intercept"] for g in groups])),
            "feature_clip": cfg["feature_clip"], "cycle_intercepts": groups,
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def a03_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "A03模型身份不同。")
    x = np.asarray(values, float)
    require(x.shape == (10,) and np.isfinite(x).all(), "A03预测必须有完整十项状态。")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + z@np.asarray(model["coefficients"]))


class A03EntryController(EntryVintageExitController):
    """实际入场首次收盘锁定A03版本；未知不换模型，两次负预测确认退出。"""

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "A03须在实际首次买入收盘选择模型。")
            self.cycle_id, self.selection_index, self.negative_count = cycle["cycle_id"], t, 0
            k = bisect_right(self.indexes, t)-1
            self.record = self.models[k] if k >= 0 else None
            self.identity = "A03:" + prediction_identity(self.record)
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index,
                "A03本笔固定选择时点或实际周期改变。")
        x = np.r_[state_values(self.data, t, cycle, current_value, peak_value), self.data.loc[t, FIELDS].to_numpy(float)]
        record = self.record
        prediction, status = None, "NO_VIEW_NO_MATURE_MODEL_AT_ENTRY"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index <= t,
                    "A03读取未来成熟周期或月度版本。")
            if np.isfinite(x).all():
                prediction, status = a03_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_TEN_FEATURES"
        elif record and record["status"] in {"NO_VIEW_INCOMPLETE_TRAINING_FEATURES", "NO_VIEW_MODEL_FIT_FAILED"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if prediction is not None and prediction < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status,
                "continuation_prediction": prediction,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "model_selection_index": self.selection_index,
                "model_selection_origin": self.data.date.iloc[self.selection_index],
                "fixed_prediction_identity": self.identity,
                "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, x))}


def save_table(name, frame):
    folder = OUT / "results"
    folder.mkdir(exist_ok=True)
    frame.to_parquet(folder/f"{name}.parquet", index=False)
    frame.to_csv(folder/f"{name}.csv", index=False, encoding="utf-8-sig")


def check():
    frozen = json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "A03协议在冻结后改变。")
    for source in frozen["sources"]:
        require(digest(ROOT/source["path"]) == source["sha256"], "A03冻结来源变化："+source["path"])
    return len(frozen["sources"])


def sources(include_intake=False):
    # 仅复用原路径清单，禁止调用原失败策略入口。
    paths = [path for path in original_source_paths() if path not in [
        Path("research/point_volatility_unit_exit_v1.py"), Path("tests/test_point_volatility_unit_exit_v1.py"),
        Path("reports/research/510300_point_method_intake_20261002/prior_direction_review.json")]]
    paths += [Path("research/point_volatility_unit_exit_v1.py"), Path(__file__).relative_to(ROOT),
              Path("tests/test_point_a03_exit_prediction_v1.py"), OUT.relative_to(ROOT)/"prior_and_source_review.json",
              OUT.relative_to(ROOT)/"tests_receipt.json"]
    if include_intake:
        paths += [path.relative_to(ROOT) for path in INTAKE.rglob("*") if path.is_file()]
    return sorted(set(paths), key=lambda path: path.as_posix())



def freeze():
    require(not (OUT/"freeze.json").exists(), "A03已冻结，不覆盖或重复登记。")
    check_intake()
    prior = json.loads((OUT/"prior_and_source_review.json").read_text(encoding="utf-8"))
    intake = json.loads((INTAKE/"summary.json").read_text(encoding="utf-8"))
    require(prior["conditional_purpose_and_source_clock_bound"] and intake["support_gate_passed"],
            "A03完整原成员或条件用途未准入，禁止注册模型。")
    require(intake["supported_original_natural_rows"] == 1507
            and intake["fully_supported_original_mature_months"] == 115, "A03完整原成员数量不同。")
    tests = json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["passed"] and tests["tests"] == 7
            and tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT/"tests/test_point_a03_exit_prediction_v1.py"),
            "A03必要测试或代码身份改变。")
    protocol = {"study": STUDY, "frozen_at": now(), "candidate_configurations": 1,
        "hypothesis": "严格新高事件的重复频率与近期加速是否增加原八项持仓继续价值信息。",
        "information_role": "EXISTING_NEW_HIGH_FREQUENCY_BLOCK_NEW_CONDITIONAL_EXIT_USE_NOT_NEW_PHYSICAL_SOURCE",
        "increment": "原八项加固定A03频率与加速两列共十项，不恢复旧分位政策或突破入场。",
        "fields": FIELDS, "formula": prior["field_binding"], "zero_missing": prior["missing_rule"],
        "unchanged": "原经济标签、全部成熟成员、最近20周期/10周期100行、周期等权、alpha1/clip5、周期截距、入场固定版本、两负确认。",
        "model": "两列各自沿原加权标准化/clip5线性加入，不新增交互、阈值或入场规则。",
        "source_clock": "j仅当日与此前20完整财富点判新高；完整两列用40点截至当日，收盘后可知、下一合法开盘执行，不读未来分红/价格。",
        "prediction_gate": "同自然原点原收益单位周期等权MSE，两时期原可用预测全部配对且MSE严格改善，入场年份块5000次/seed51030090改进95%下界均>0。",
        "evaluation_periods": PERIODS,
        "fit_cache": "完整有序十项/标签/成员/权重/设置完全相同才复用，失败不重试。",
        "economic_gate_if_prediction_passes": "另冻结应用；原20万元252日全日历BASE/STRESS两时期，净CAGR与净夏普均高于A、实际净pB>1、平均周期收益>0、回撤<=10%；次数软目标。",
        "old_failures": "原A03高低分位、单一突破/回踩/A01/其他固定失败保持。",
        "no_rescue": prior["no_rescue"], "feature_regression_estimation_calls": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "orders_authorized": False, "minute_data_used": False, "goal_achieved": False}
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    frozen = [{"path": path.as_posix(), "sha256": digest(ROOT/path)} for path in sources(True)]+prior["direct_sources"]
    unique = {item["path"]: item for item in frozen}
    write_json(OUT/"freeze.json", {"frozen_at": now(), "protocol_sha256": digest(OUT/"protocol.json"),
                                 "sources": list(unique.values())}, exclusive=True)
    print(json.dumps({"状态": "A03固定两列模型已冻结", "来源": check(), "新增收益模型拟合": 0}, ensure_ascii=False))


def training_identity(rows, cfg):
    columns = ["cycle_id", "origin_index", "exit_index", "target", "sample_weight"] + FEATURES
    values = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(np.uint64).tobytes()
    return hashlib.sha256(values + json.dumps({"ridge_alpha": cfg["ridge_alpha"],
                       "feature_clip": cfg["feature_clip"]}, sort_keys=True).encode("utf-8")).hexdigest()


def improvement_interval(values):
    years = sorted(values.entry_year.unique())
    if len(years) < 2:
        return {"status": "NOT_COMPUTED_FEWER_THAN_TWO_YEAR_BLOCKS", "low": None, "high": None}
    groups = [values.loc[values.entry_year.eq(y), "improvement"].to_numpy(float) for y in years]
    rng = np.random.default_rng(51030090)
    estimates = np.array([np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]).mean()
                          for _ in range(5000)])
    return {"status": "COMPUTED_DEVELOPMENT_YEAR_BLOCK_SENSITIVITY", "years": [int(y) for y in years],
            "iterations": 5000, "seed": 51030090, "low": float(np.quantile(estimates, .025)),
            "high": float(np.quantile(estimates, .975)), "independence_established": False}


def fit_and_evaluate():
    require(not (OUT/"PREDICTION_STARTED.json").exists(), "A03预测阶段已开始，不重跑或覆写。")
    check()
    write_json(OUT/"PREDICTION_STARTED.json", {"at": now()}, exclusive=True)
    data = pd.read_parquet(ROOT/CURRENT/"inputs/candidate_features.parquet")
    validate_wealth_clock(data)
    factor = pd.read_parquet(INTAKE/"results/日线A03事前字段.parquet")
    require(len(factor) == len(data) == 3488, "A03冻结字段日历不同。")
    save_table("日线A03事前字段", factor)
    samples = pd.read_parquet(ROOT/CURRENT/"results/training_reference/samples.parquet")
    cycles = pd.read_parquet(ROOT/CURRENT/"results/training_reference/cycles.parquet").set_index("cycle_id")
    originals = json.loads((ROOT/CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    cfg = json.loads((ROOT/CURRENT/"inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    require(cfg["recent_cycles"] == 20 and cfg["minimum_cycles"] == 10 and cfg["minimum_rows"] == 100
            and cfg["ridge_alpha"] == 1. and cfg["feature_clip"] == 5., "原固定训练合同不同。")
    indexes = samples.origin_index.to_numpy(int)
    require(pd.DatetimeIndex(samples.origin).equals(pd.DatetimeIndex(data.date.iloc[indexes])),
            "自然样本原点与字段日历不同。")
    samples[FIELDS] = factor[FIELDS].iloc[indexes].to_numpy()
    cache, candidates, checks, fits = {}, [], [], []
    for original in originals:
        record = {k: copy.deepcopy(v) for k, v in original.items() if k != "model"}
        record["model"] = None
        if original["status"] == "FIT_COMPLETE":
            rows, ids = training_rows(samples, original["fit_index"], cfg)
            require(ids == original["training_cycles"] and len(rows) == original["training_rows"],
                    "A03改变了原成熟训练成员。")
            key = training_identity(rows, cfg)
            first = key not in cache
            if first:
                control = fit_within_cycle_exit(rows, cfg)
                failure, model = None, None
                if np.isfinite(rows[FEATURES].to_numpy(float)).all():
                    try:
                        model = fit_a03_within(rows, cfg)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                        failure = str(error)
                else:
                    failure = "完整训练的A03字段缺失；不删行或补值。"
                cache[key] = {"control": control, "model": model, "failure": failure,
                              "first_fit_origin": original["fit_origin"]}
            stored = cache[key]
            errors = [abs(stored["control"]["intercept"] - original["model"]["intercept"])]
            errors += [float(np.max(np.abs(np.asarray(stored["control"][k])-np.asarray(original["model"][k]))))
                       for k in ("mean", "scale", "coefficients")]
            require(max(errors) <= 1e-12, "未还原保存原八项模型。")
            record.update(model=copy.deepcopy(stored["model"]),
                          status="FIT_COMPLETE" if stored["model"] else "NO_VIEW_MODEL_FIT_FAILED",
                          failure=stored["failure"], input_identity=key,
                          first_estimation_origin=stored["first_fit_origin"], reused=not first)
            checks.append({"fit_origin": original["fit_origin"], "max_parameter_error": max(errors),
                           "input_identity": key, "first_control_estimation": first})
            fits.append({"fit_origin": original["fit_origin"], "status": record["status"],
                         "input_identity": key, "first_estimation": first, "training_rows": len(rows),
                         "training_cycles": len(ids), "latest_training_exit_index": int(rows.exit_index.max()),
                         "fit_index": original["fit_index"], "missing_block_rows": int(rows[FIELDS].isna().any(axis=1).sum()),
                         "block_coefficients": json.dumps(stored["model"]["coefficients"][-2:]) if stored["model"] else None})
        candidates.append(record)
    write_json(OUT/"candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    paired = []
    for row in samples.itertuples():
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old_record, new_record = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        old, new, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        x = [getattr(row, k) for k in BASE_FEATURES]
        extra = np.asarray([getattr(row, name) for name in FIELDS], float)
        if old_record and old_record["status"] == "FIT_COMPLETE":
            require(old_record["latest_exit_index"] <= old_record["fit_index"] <= entry <= row.origin_index
                    < row.early_exit_index < row.exit_index, "原预测/入场/标签成熟时钟错误。")
            old = within_cycle_prediction(old_record["model"], x)
            if new_record and new_record["status"] == "FIT_COMPLETE" and np.isfinite(extra).all():
                require(new_record["fit_index"] == old_record["fit_index"], "配对月度版本不同。")
                new, status = a03_prediction(new_record["model"], x+extra.tolist()), "PAIRED_PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_CANDIDATE_MODEL_OR_TEN_FEATURE_BLOCK"
        paired.append({"cycle_id": int(row.cycle_id), "origin": row.origin,
                       "origin_index": int(row.origin_index), "entry_index": entry,
                       "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                       "target": float(row.target), **dict(zip(FIELDS, extra)), "baseline_prediction": old,
                       "candidate_prediction": new, "status": status})
    paired = pd.DataFrame(paired)
    periods, cycle_tables = [], []
    for name, start, end in PERIODS:
        group = paired.loc[paired.origin.between(pd.Timestamp(start), pd.Timestamp(end))]
        original_known = group.baseline_prediction.notna()
        matched = group.loc[original_known & group.candidate_prediction.notna()].copy()
        matched["old_error"] = (matched.baseline_prediction-matched.target)**2
        matched["new_error"] = (matched.candidate_prediction-matched.target)**2
        per_cycle = matched.groupby("cycle_id").agg(baseline_mse=("old_error", "mean"),
            candidate_mse=("new_error", "mean"), entry_year=("entry_year", "first"),
            rows=("origin_index", "size")).reset_index()
        per_cycle["improvement"] = per_cycle.baseline_mse-per_cycle.candidate_mse
        per_cycle["period"] = name
        cycle_tables.append(per_cycle)
        interval = improvement_interval(per_cycle)
        baseline = float(per_cycle.baseline_mse.mean()) if len(per_cycle) else None
        candidate = float(per_cycle.candidate_mse.mean()) if len(per_cycle) else None
        complete = len(matched) == int(original_known.sum()) and bool(original_known.any())
        passed = bool(complete and candidate < baseline and interval["low"] is not None and interval["low"] > 0)
        periods.append({"period": name, "original_available_rows": int(original_known.sum()),
                        "paired_rows": len(matched), "cycles": len(per_cycle), "complete_pair_coverage": complete,
                        "baseline_raw_return_mse": baseline, "candidate_raw_return_mse": candidate,
                        "relative_mse_change": candidate/baseline-1 if baseline else None,
                        "improvement_interval": interval, "prediction_gate_passed": passed})
    passed = all(p["prediction_gate_passed"] for p in periods)
    failed = sum(c["model"] is None for c in cache.values())
    paired_known = paired.loc[paired.baseline_prediction.notna() & paired.candidate_prediction.notna()]
    signs = int(((paired_known.baseline_prediction < 0) != (paired_known.candidate_prediction < 0)).sum())
    accounting = {"candidate_configurations": 1, "monthly_records": len(candidates),
                  "mature_monthly_records": len(checks), "distinct_training_inputs": len(cache),
                  "new_candidate_fit_attempts": len(cache), "successful_candidate_fits": len(cache)-failed,
                  "failed_distinct_candidate_fits": failed, "reused_monthly_records": len(checks)-len(cache),
                  "control_coefficient_reestimations": len(cache), "control_monthly_parameter_checks": len(checks),
                  "total_candidate_and_control_fit_calls": 2*len(cache),
                  "feature_OLS_calls_during_prediction": 0,
                  "frozen_intake_feature_OLS_calls": 0,
                  "independent_trials": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED"}
    save_table("同自然原点配对预测", paired)
    save_table("逐周期预测误差", pd.concat(cycle_tables, ignore_index=True))
    save_table("原八项模型复算", pd.DataFrame(checks))
    save_table("逐月成熟训练及A03两列", pd.DataFrame(fits))
    write_json(OUT/"trial_accounting.json", accounting, exclusive=True)
    summary = {"study": STUDY, "completed_at": now(),
        "status": "PREDICTION_GATE_PASS_ECONOMIC_APPLICATION_PENDING" if passed else
                  "REJECTED_FIXED_A03_FIELD_PREDICTION_GATE_FAILED",
        "prediction_gate_passed": passed, "periods": periods, "trial_accounting": accounting,
        "original_parameter_max_error": max(c["max_parameter_error"] for c in checks),
        "all_pair_rows": len(paired_known), "canonical_prediction_sign_changes": signs,
        "canonical_prediction_max_absolute_change": float(np.max(np.abs(
            paired_known.candidate_prediction-paired_known.baseline_prediction))) if len(paired_known) else None,
        "candidate_no_view_rows": int(paired.candidate_prediction.isna().sum()),
        "new_strategy_accounts": 0, "account_return_sharpe": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT/"prediction_summary.json", summary, exclusive=True)
    write_json(OUT/"economic_stage_status.json", {"at": now(),
        "status": "NOT_RUN_PREDICTION_PASS_APPLICATION_PENDING" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
        "new_strategy_accounts": 0, "account_return_sharpe": "NOT_COMPUTED"}, exclusive=True)
    files = sorted((OUT/"results").glob("*")) + [OUT/"candidate_models.json", OUT/"prediction_summary.json",
             OUT/"trial_accounting.json", OUT/"economic_stage_status.json"]
    write_json(OUT/"prediction_verification_receipt.json", {"at": now(),
        "status": "PASS_SAVED_ORIGINAL_MODELS_AND_FROZEN_SOURCES", "sources_unchanged": check(),
        "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files]}, exclusive=True)
    print(json.dumps({"status": summary["status"], "periods": periods, "trial_accounting": accounting,
                      "符号变化": signs}, ensure_ascii=False), flush=True)




METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
FIELD_INPUT_COLUMNS = ["date", "symbol", "close", "previous_close", "dividend",
                       "total_simple", "wealth", "total_log"]
INTAKE_TABLES = ("日线A03事前字段", "原自然成员A03支持", "原月度成熟成员A03支持")


def freeze_intake():
    require(not (INTAKE/"freeze.json").exists(), "A03数量阶段已冻结，禁止覆盖。")
    tests = json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["passed"] and tests["tests"] == 7
            and tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT/"tests/test_point_a03_exit_prediction_v1.py"),
            "A03必要测试或代码身份尚未通过。")
    prior = json.loads((OUT/"prior_and_source_review.json").read_text(encoding="utf-8"))
    require(prior["conditional_purpose_and_source_clock_bound"], "A03条件用途与源时钟尚未绑定。")
    INTAKE.mkdir(exist_ok=False)
    protocol = {"study": STUDY+"_FIELD_INTAKE", "frozen_at": now(),
        "purpose": "先核对原完整成员，不读取收益标签或拟合继续收益模型。",
        "formula": prior["field_binding"], "missing_rule": prior["missing_rule"],
        "support_rule": prior["support_rule"], "no_rescue": prior["no_rescue"],
        "fixed_field_operationalizations": 1, "return_model_configurations": 0,
        "new_return_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "feature_regression_accounting": prior["field_regression_accounting"],
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(INTAKE/"protocol.json", protocol, exclusive=True)
    frozen = [{"path": path.as_posix(), "sha256": digest(ROOT/path)} for path in sources()]+prior["direct_sources"]
    unique = {item["path"]: item for item in frozen}
    write_json(INTAKE/"freeze.json", {"frozen_at": now(), "protocol_sha256": digest(INTAKE/"protocol.json"),
                                    "sources": list(unique.values())}, exclusive=True)
    print(json.dumps({"状态": "A03字段操作定义已冻结", "来源": check_intake(), "新增收益模型拟合": 0}, ensure_ascii=False))


def check_intake():
    frozen = json.loads((INTAKE/"freeze.json").read_text(encoding="utf-8"))
    require(digest(INTAKE/"protocol.json") == frozen["protocol_sha256"], "A03数量协议改变。")
    for item in frozen["sources"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "A03数量来源改变："+item["path"])
    return len(frozen["sources"])


def derive_intake():
    data = pd.read_parquet(ROOT/CURRENT/"inputs/candidate_features.parquet", columns=FIELD_INPUT_COLUMNS)
    samples = pd.read_parquet(ROOT/CURRENT/"results/training_reference/samples.parquet", columns=METADATA)
    records = json.loads((ROOT/CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "A03原快照数量改变。")
    validate_wealth_clock(data)
    field = a03_information_field(data)
    indexes = samples.origin_index.to_numpy(int)
    require(np.array_equal(pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]"),
                           field.date.iloc[indexes].to_numpy(dtype="datetime64[ns]")), "A03原点日期不同。")
    natural = field.iloc[indexes].reset_index(drop=True)
    natural.insert(1, "cycle_id", samples.cycle_id.to_numpy(int))
    rows = []
    values = field[FIELDS].to_numpy(float)
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"]
                and len(selected) == record["training_rows"]
                and selected.cycle_id.nunique() == len(ids), "A03原训练成员改变。")
        require((selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "A03原周期尚未成熟。")
        finite = np.isfinite(values[selected.origin_index.to_numpy(int)]).all(axis=1)
        rows.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict), "original_training_rows": len(selected),
            "original_training_cycles": len(ids), "supported_rows": int(finite.sum()),
            "missing_rows": int((~finite).sum()), "all_original_members_supported": bool(finite.all())})
    monthly = pd.DataFrame(rows)
    ready = monthly.loc[monthly.original_model_available]
    require(len(ready) == 115, "A03原可用月份改变。")
    supported = bool(natural[FIELDS].notna().all(axis=1).all() and ready.all_original_members_supported.all())
    facts = {"support_gate_passed": supported, "daily_rows": len(field),
        "finite_daily_rows": int(field[FIELDS].notna().all(axis=1).sum()),
        "daily_status_counts": {str(key): int(value) for key, value in field.status.value_counts().items()},
        "new_feature_OLS_estimation_calls": 0,
        "rank_deficient_feature_OLS_calls": 0,
        "original_natural_rows": len(natural), "supported_original_natural_rows": int(natural[FIELDS].notna().all(axis=1).sum()),
        "original_monthly_records": len(monthly), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(monthly)-len(ready),
        "field_ranges": {name: [float(natural[name].min()), float(natural[name].max())] for name in FIELDS}}
    return (field, natural, monthly), facts


def run_intake():
    count = check_intake()
    write_json(INTAKE/"ADMISSION_STARTED.json", {"at": now(), "new_return_model_fits": 0}, exclusive=True)
    tables, facts = derive_intake()
    folder = INTAKE/"results"
    folder.mkdir(exist_ok=False)
    for name, table in zip(INTAKE_TABLES, tables):
        table.to_parquet(folder/(name+".parquet"), index=False)
        table.to_csv(folder/(name+".csv"), index=False, encoding="utf-8-sig")
    summary = {"study": STUDY+"_FIELD_INTAKE", "completed_at": now(),
        "status": "ADMITTED_A03_COMPLETE_ORIGINAL_MEMBER_SUPPORT_ONLY_NO_RETURN_MODEL" if facts["support_gate_passed"]
                  else "REJECTED_FIXED_A03_TWO_FIELD_BLOCK_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT",
        "frozen_sources_unchanged": count, **facts, "new_return_model_configurations": 0,
        "new_return_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(INTAKE/"summary.json", summary, exclusive=True)
    require(check_intake() == count, "A03数量运行后来源改变。")
    print(json.dumps({"状态": summary["status"], "原成员支持": facts["supported_original_natural_rows"],
        "完整可用月": facts["fully_supported_original_mature_months"],
        "字段回归估计": facts["new_feature_OLS_estimation_calls"], "新增收益模型拟合": 0}, ensure_ascii=False))


def verify_intake():
    count = check_intake()
    tables, facts = derive_intake()
    for name, rebuilt in zip(INTAKE_TABLES, tables):
        saved = pd.read_parquet(INTAKE/"results"/(name+".parquet"))
        pd.testing.assert_frame_equal(saved, rebuilt, check_exact=True)
    summary = json.loads((INTAKE/"summary.json").read_text(encoding="utf-8"))
    require(all(summary[key] == value for key, value in facts.items()), "A03数量摘要复算不同。")
    receipt = {"at": now(), "status": "PASS_SAVED_A03_FIELD_AND_ORIGINAL_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "tables_and_support_exactly_equal": True,
        "daily_rows": len(tables[0]), "natural_rows": len(tables[1]), "monthly_rows": len(tables[2]),
        "feature_OLS_recomputation_calls": facts["new_feature_OLS_estimation_calls"],
        "new_return_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0}
    write_json(INTAKE/"saved_output_recomputation_receipt.json", receipt, exclusive=True)
    print(json.dumps({"状态": receipt["status"], "字段回归复算": receipt["feature_OLS_recomputation_calls"],
                      "新增收益模型拟合": 0}, ensure_ascii=False))


def verify_saved_predictions():
    count = check()
    samples = pd.read_parquet(ROOT/CURRENT/"results/training_reference/samples.parquet")
    cycles = pd.read_parquet(ROOT/CURRENT/"results/training_reference/cycles.parquet").set_index("cycle_id")
    originals = json.loads((ROOT/CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    candidates = json.loads((OUT/"candidate_models.json").read_text(encoding="utf-8"))["models"]
    factor = pd.read_parquet(INTAKE/"results/日线A03事前字段.parquet")
    pd.testing.assert_frame_equal(factor, pd.read_parquet(OUT/"results/日线A03事前字段.parquet"), check_exact=True)
    samples[FIELDS] = factor[FIELDS].iloc[samples.origin_index.to_numpy(int)].to_numpy()
    paired = []
    for row in samples.itertuples():
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old_record, new_record = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        old, new, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        x, extra = [getattr(row, name) for name in BASE_FEATURES], np.asarray([getattr(row, name) for name in FIELDS], float)
        if old_record and old_record["status"] == "FIT_COMPLETE":
            require(old_record["latest_exit_index"] <= old_record["fit_index"] <= entry <= row.origin_index
                    < row.early_exit_index < row.exit_index, "A03保存预测读取未来版本或错标签时钟。")
            old = within_cycle_prediction(old_record["model"], x)
            if new_record and new_record["status"] == "FIT_COMPLETE" and np.isfinite(extra).all():
                require(new_record["fit_index"] == old_record["fit_index"], "A03保存配对版本不同。")
                new, status = a03_prediction(new_record["model"], x+extra.tolist()), "PAIRED_PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_CANDIDATE_MODEL_OR_TEN_FEATURE_BLOCK"
        paired.append({"cycle_id": int(row.cycle_id), "origin": row.origin, "origin_index": int(row.origin_index),
            "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
            "target": float(row.target), **dict(zip(FIELDS, extra)), "baseline_prediction": old,
            "candidate_prediction": new, "status": status})
    paired = pd.DataFrame(paired)
    pd.testing.assert_frame_equal(paired, pd.read_parquet(OUT/"results/同自然原点配对预测.parquet"), check_exact=True)
    summary = json.loads((OUT/"prediction_summary.json").read_text(encoding="utf-8"))
    cycle_tables = []
    for name, start, end in PERIODS:
        group = paired.loc[paired.origin.between(pd.Timestamp(start), pd.Timestamp(end))]
        original_known = group.baseline_prediction.notna()
        matched = group.loc[original_known & group.candidate_prediction.notna()].copy()
        matched["old_error"] = (matched.baseline_prediction-matched.target)**2
        matched["new_error"] = (matched.candidate_prediction-matched.target)**2
        per_cycle = matched.groupby("cycle_id").agg(baseline_mse=("old_error", "mean"),
            candidate_mse=("new_error", "mean"), entry_year=("entry_year", "first"), rows=("origin_index", "size")).reset_index()
        per_cycle["improvement"] = per_cycle.baseline_mse-per_cycle.candidate_mse
        per_cycle["period"] = name
        cycle_tables.append(per_cycle)
        interval = improvement_interval(per_cycle)
        saved_period = next(item for item in summary["periods"] if item["period"] == name)
        require(interval == saved_period["improvement_interval"], "A03年度块区间复算不同。")
        require(float(per_cycle.baseline_mse.mean()) == saved_period["baseline_raw_return_mse"]
                and float(per_cycle.candidate_mse.mean()) == saved_period["candidate_raw_return_mse"], "A03保存MSE复算不同。")
    rebuilt = pd.concat(cycle_tables, ignore_index=True)
    pd.testing.assert_frame_equal(rebuilt, pd.read_parquet(OUT/"results/逐周期预测误差.parquet"), check_exact=True)
    receipt = {"at": now(), "status": "PASS_SAVED_A03_PREDICTIONS_CYCLE_ERRORS_AND_INTERVAL_RECOMPUTATION",
        "frozen_sources_unchanged": count, "all_original_prediction_rows": len(paired), "paired_cycle_rows": len(rebuilt),
        "predictions_and_intervals_exactly_equal": True, "new_return_model_fits": 0,
        "feature_OLS_calls": 0, "new_return_labels": 0, "new_strategy_accounts": 0}
    write_json(OUT/"saved_output_recomputation_receipt.json", receipt, exclusive=True)
    print(json.dumps({"状态": receipt["status"], "配对表原行": len(paired), "新拟合": 0}, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description="A03隔离固定增量研究")
    parser.add_argument("action", choices=["intake-freeze", "intake-run", "intake-verify", "freeze", "run", "verify"])
    args = parser.parse_args()
    actions = {"intake-freeze": freeze_intake, "intake-run": run_intake, "intake-verify": verify_intake,
               "freeze": freeze, "run": fit_and_evaluate, "verify": verify_saved_predictions}
    actions[args.action]()



if __name__ == "__main__":
    main()
