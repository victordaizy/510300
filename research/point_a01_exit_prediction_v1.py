"""既有A01路径字段用于原退出条件增量；不改原冻结策略、标签或成熟成员。"""
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


STUDY = "510300_POINT_A01_EXIT_PREDICTION_V1"
OUT = ROOT / "reports/research/510300_point_a01_exit_prediction_v1"
FIELD = "signed_path_efficiency20"
FEATURES = BASE_FEATURES + [FIELD]
KIND = "A01_WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE"
PERIODS = [("2015_2019", "2015-01-05", "2019-12-31"),
           ("2020_2026", "2020-01-02", "2026-09-30")]


def validate_wealth_clock(data):
    """绑定原收盘和当日已除息现金的前向财富；不接受回溯价格或未知插值。"""
    require(data.symbol.eq("510300.SH").all(), "A01标的身份不同。")
    raw = data[["close", "dividend", "previous_close", "total_simple", "wealth"]].to_numpy(float)
    close, dividend, previous, returns, wealth = raw.T
    require(np.isfinite(close).all() and (close > 0).all(), "A01原收盘缺失或非正。")
    require(np.isfinite(dividend).all() and (dividend >= 0).all(), "A01现金除息事件缺失或为负。")
    require(np.isfinite(previous[1:]).all() and (previous[1:] > 0).all(), "A01前收盘缺失或非正。")
    np.testing.assert_allclose(previous[1:], close[:-1], atol=1e-12, rtol=0)
    expected = (close[1:]+dividend[1:])/previous[1:]-1
    require(np.isfinite(returns[1:]).all(), "A01非初始经济收益缺失，不填零。")
    np.testing.assert_allclose(returns[1:], expected, atol=1e-12, rtol=0)
    require(np.isfinite(wealth).all() and (wealth > 0).all(), "A01前向财富缺失或非正。")
    # 仅首个市场点归一为一；后续全部使用已发生的原价及当日现金，不填其他缺口。
    reconstructed = np.r_[1., np.cumprod(1+expected)]
    np.testing.assert_allclose(wealth, reconstructed, atol=1e-12, rtol=1e-12)


def directional_path_efficiency(data):
    """原A01有向20区间效率；21财富点完整且绝对路径为正才有观察。"""
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_unique and dates.is_monotonic_increasing, "A01日期必须唯一递增。")
    wealth = pd.to_numeric(data.wealth, errors="coerce").astype(float)
    finite = np.isfinite(wealth.to_numpy())
    require((wealth.to_numpy()[finite] > 0).all(), "A01有限财富必须为正。")
    valid = wealth.where(finite)
    complete = valid.rolling(21, min_periods=21).count().eq(21)
    path = valid.diff().abs().rolling(20, min_periods=20).sum()
    net = valid-valid.shift(20)
    available = complete & path.gt(0) & np.isfinite(path)
    result = (net/path).where(available)
    require(result.dropna().between(-1-1e-12, 1+1e-12).all(), "A01效率越出数学范围。")
    index = np.arange(len(data))
    status = np.full(len(data), "NO_VIEW_20_INTERVAL_WARMUP", dtype=object)
    warmed = index >= 20
    status[warmed & ~complete.to_numpy()] = "NO_VIEW_INCOMPLETE_21_WEALTH_POINTS"
    status[warmed & complete.to_numpy() & ~available.to_numpy()] = "NO_VIEW_ZERO_OR_INVALID_PATH"
    status[available.to_numpy()] = "A01_AVAILABLE"
    return pd.DataFrame({"date": dates, "known_forward_wealth": valid.to_numpy(),
                         "window_net_wealth_change": net.to_numpy(), "window_absolute_wealth_path": path.to_numpy(),
                         FIELD: result.to_numpy(), "status": status,
                         "window_start_index": np.where(warmed, index-20, -1), "latest_source_index": index})


def fit_a01_within(rows, cfg):
    """原固定周期截距岭回归增加一项；不删缺失行，不改变周期等权和正则化。"""
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(),
            "A01完整训练输入缺失，禁止删行或补值。")
    require(np.isfinite(weights).all() and (weights > 0).all(), "A01周期训练权重无效。")
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
    require(np.isfinite(fitted.coef_).all(), "A01回归系数无效。")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"] -
                                         np.asarray(group["standardized_feature_mean"])@fitted.coef_)
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(np.mean([g["cycle_intercept"] for g in groups])),
            "feature_clip": cfg["feature_clip"], "cycle_intercepts": groups,
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def a01_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "A01模型身份不同。")
    x = np.asarray(values, float)
    require(x.shape == (9,) and np.isfinite(x).all(), "A01预测必须有完整九项状态。")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + z@np.asarray(model["coefficients"]))


class A01EntryController(EntryVintageExitController):
    """实际入场首次收盘锁定A01版本；未知不换模型，两次负预测确认退出。"""

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "A01须在实际首次买入收盘选择模型。")
            self.cycle_id, self.selection_index, self.negative_count = cycle["cycle_id"], t, 0
            k = bisect_right(self.indexes, t)-1
            self.record = self.models[k] if k >= 0 else None
            self.identity = "A01:" + prediction_identity(self.record)
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index,
                "A01本笔固定选择时点或实际周期改变。")
        x = np.r_[state_values(self.data, t, cycle, current_value, peak_value), self.data[FIELD].iloc[t]]
        record = self.record
        prediction, status = None, "NO_VIEW_NO_MATURE_MODEL_AT_ENTRY"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index <= t,
                    "A01读取未来成熟周期或月度版本。")
            if np.isfinite(x).all():
                prediction, status = a01_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_NINE_FEATURES"
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
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "A01协议在冻结后改变。")
    for source in frozen["sources"]:
        require(digest(ROOT/source["path"]) == source["sha256"], "A01冻结来源变化："+source["path"])
    return len(frozen["sources"])


def sources():
    # 复用原共同账户/标签所需路径清单；仅登记文件，不调用已失败方法的实验入口。
    paths = [p for p in original_source_paths() if p not in [
        Path("research/point_volatility_unit_exit_v1.py"), Path("tests/test_point_volatility_unit_exit_v1.py"),
        Path("reports/research/510300_point_method_intake_20261002/prior_direction_review.json")]]
    paths += [Path("research/point_volatility_unit_exit_v1.py"), Path(__file__).relative_to(ROOT),
              Path("tests/test_point_a01_exit_prediction_v1.py"), OUT.relative_to(ROOT)/"prior_and_source_review.json",
              OUT.relative_to(ROOT)/"tests_receipt.json"]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    require(not (OUT/"freeze.json").exists(), "A01已冻结，不覆盖或重复登记。")
    prior = json.loads((OUT/"prior_and_source_review.json").read_text(encoding="utf-8"))
    require(prior["status"] == "ADMITTED_EXISTING_A01_FIELD_FOR_NEW_CONDITIONAL_EXIT_INCREMENT_DEVELOPMENT",
            "A01旧表达、用途或来源准入未通过。")
    require(prior["original_natural_rows_supported"] == 1507 and prior["original_mature_months_supported"] == 115,
            "A01完整原成员数量未通过。")
    tests = json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["passed"] and tests["tests"] == 8 and tests["module_sha256"] == digest(Path(__file__)),
            "A01八项必要测试或模块身份未通过。")
    require(tests["test_sha256"] == digest(ROOT/"tests/test_point_a01_exit_prediction_v1.py"), "A01测试来源身份改变。")
    protocol = {
        "study": STUDY, "frozen_at": now(), "candidate_configurations": 1,
        "hypothesis": "既有有向路径效率相对原八项是否提供原持仓继续收益的条件增量。",
        "information_role": "EXISTING_SIGNED_ER20_NEW_CONDITIONAL_EXIT_USE_NOT_NEW_PHYSICAL_SOURCE",
        "increment": "原八项加入含分红前向财富有向ER20一项，不运行旧三家族策略或T01。",
        "field": FIELD, "price_coordinate": "原close和当日现金收益前向累乘的wealth，起点1，不使用未来回溯复权。",
        "formula": "(W[t]-W[t-20])/sum(abs(W[j]-W[j-1]),j=t-19..t)，包括当日20差分，21点完整。",
        "zero_missing": "真实零净变化/正路径为0；真零路径或不足完整窗口为NO_VIEW，不填零或删原成员。",
        "unchanged": "原经济标签、全部成熟成员、最近20周期/10周期100行、周期等权、alpha1/clip5、原周期截距和入场固定版本、两负确认。",
        "ninth_model": "原加权标准化及周期内岭；新周期截距为成熟周期截距均值，单字段线性加入，不新增交互或入场规则。",
        "source_clock": "各日完整收盘后可知；仍是截至9月30日的开发材料，无新行情。",
        "prediction_gate": "同自然原点原收益单位、周期等权MSE；两时期原预测全部配对且MSE严格改善，入场整年块5000次/seed51030080的改进95%下界均>0。",
        "evaluation_periods": PERIODS,
        "fit_cache": "原完整有序训练输入及9项、标签、周期权重与设置相同则复用；失败缓存，不重复尝试。",
        "economic_gate_if_prediction_passes": "另冻结应用阶段；原20万元/252日全日历/BASE及STRESS、两时期，净CAGR与净夏普均高于A，实际净pB>1、平均周期收益>0、回撤<=10%；频率软目标。",
        "old_failures": "旧日更三家族及T01失败保留；其共同或有限用途不冒充该原八项条件增量的独立字段失败，也不据本次结论覆盖旧策略。",
        "no_rescue": "不因失败改20窗口、财富坐标、方向、未知赋值、成员、模型、标签、正则、确认或时期；不恢复旧策略/系数/仓位映射。",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "orders_authorized": False, "minute_data_used": False, "goal_achieved": False,
    }
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    frozen_sources = [{"path": p.as_posix(), "sha256": digest(ROOT/p)} for p in sources()]+prior["direct_sources"]
    unique = {item["path"]: item for item in frozen_sources}
    write_json(OUT/"freeze.json", {"frozen_at": now(), "protocol_sha256": digest(OUT/"protocol.json"),
                                  "sources": list(unique.values())}, exclusive=True)
    print(f"A01既有字段的条件退出用途已冻结，{check()}项来源；尚未拟合或运行新账户。", flush=True)


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
    rng = np.random.default_rng(51030080)
    estimates = np.array([np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]).mean()
                          for _ in range(5000)])
    return {"status": "COMPUTED_DEVELOPMENT_YEAR_BLOCK_SENSITIVITY", "years": [int(y) for y in years],
            "iterations": 5000, "seed": 51030080, "low": float(np.quantile(estimates, .025)),
            "high": float(np.quantile(estimates, .975)), "independence_established": False}


def fit_and_evaluate():
    require(not (OUT/"PREDICTION_STARTED.json").exists(), "A01预测阶段已开始，不重跑或覆写。")
    check()
    write_json(OUT/"PREDICTION_STARTED.json", {"at": now()}, exclusive=True)
    data = pd.read_parquet(ROOT/CURRENT/"inputs/candidate_features.parquet")
    validate_wealth_clock(data)
    factor = directional_path_efficiency(data)
    save_table("日线A01事前字段", factor)
    samples = pd.read_parquet(ROOT/CURRENT/"results/training_reference/samples.parquet")
    cycles = pd.read_parquet(ROOT/CURRENT/"results/training_reference/cycles.parquet").set_index("cycle_id")
    originals = json.loads((ROOT/CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    cfg = json.loads((ROOT/CURRENT/"inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    require(cfg["recent_cycles"] == 20 and cfg["minimum_cycles"] == 10 and cfg["minimum_rows"] == 100
            and cfg["ridge_alpha"] == 1. and cfg["feature_clip"] == 5., "原固定训练合同不同。")
    indexes = samples.origin_index.to_numpy(int)
    require(pd.DatetimeIndex(samples.origin).equals(pd.DatetimeIndex(data.date.iloc[indexes])),
            "自然样本原点与字段日历不同。")
    samples[FIELD] = factor[FIELD].iloc[indexes].to_numpy()
    cache, candidates, checks, fits = {}, [], [], []
    for original in originals:
        record = {k: copy.deepcopy(v) for k, v in original.items() if k != "model"}
        record["model"] = None
        if original["status"] == "FIT_COMPLETE":
            rows, ids = training_rows(samples, original["fit_index"], cfg)
            require(ids == original["training_cycles"] and len(rows) == original["training_rows"],
                    "A01改变了原成熟训练成员。")
            key = training_identity(rows, cfg)
            first = key not in cache
            if first:
                control = fit_within_cycle_exit(rows, cfg)
                failure, model = None, None
                if np.isfinite(rows[FEATURES].to_numpy(float)).all():
                    try:
                        model = fit_a01_within(rows, cfg)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                        failure = str(error)
                else:
                    failure = "完整训练的A01字段缺失；不删行或补值。"
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
                         "fit_index": original["fit_index"], "missing_ninth_rows": int(rows[FIELD].isna().sum()),
                         "ninth_coefficient": stored["model"]["coefficients"][-1] if stored["model"] else None})
        candidates.append(record)
    write_json(OUT/"candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    paired = []
    for row in samples.itertuples():
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old_record, new_record = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        old, new, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        x = [getattr(row, k) for k in BASE_FEATURES]
        extra = float(getattr(row, FIELD))
        if old_record and old_record["status"] == "FIT_COMPLETE":
            require(old_record["latest_exit_index"] <= old_record["fit_index"] <= entry <= row.origin_index
                    < row.early_exit_index < row.exit_index, "原预测/入场/标签成熟时钟错误。")
            old = within_cycle_prediction(old_record["model"], x)
            if new_record and new_record["status"] == "FIT_COMPLETE" and np.isfinite(extra):
                require(new_record["fit_index"] == old_record["fit_index"], "配对月度版本不同。")
                new, status = a01_prediction(new_record["model"], x+[extra]), "PAIRED_PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_CANDIDATE_MODEL_OR_NINTH_FEATURE"
        paired.append({"cycle_id": int(row.cycle_id), "origin": row.origin,
                       "origin_index": int(row.origin_index), "entry_index": entry,
                       "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                       "target": float(row.target), FIELD: extra, "baseline_prediction": old,
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
                  "independent_trials": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED"}
    save_table("同自然原点配对预测", paired)
    save_table("逐周期预测误差", pd.concat(cycle_tables, ignore_index=True))
    save_table("原八项模型复算", pd.DataFrame(checks))
    save_table("逐月成熟训练及第九项", pd.DataFrame(fits))
    write_json(OUT/"trial_accounting.json", accounting, exclusive=True)
    summary = {"study": STUDY, "completed_at": now(),
        "status": "PREDICTION_GATE_PASS_ECONOMIC_APPLICATION_PENDING" if passed else
                  "REJECTED_FIXED_A01_FIELD_PREDICTION_GATE_FAILED",
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


def main():
    parser = argparse.ArgumentParser(description="A01日线成交分布单字段退出预测增量")
    parser.add_argument("action", choices=["freeze", "predict", "check"])
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "predict":
        fit_and_evaluate()
    else:
        print(f"A01冻结来源未变：{check()}份。", flush=True)


if __name__ == "__main__":
    main()
