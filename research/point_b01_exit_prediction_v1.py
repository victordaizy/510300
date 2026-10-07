"""隔离的B01五日均线偏离信息块：两列固定输入，不改原冻结策略。"""
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
from research.point_a01_exit_prediction_v1 import validate_wealth_clock
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController, prediction_identity
from research.point_volatility_unit_exit_v1 import CURRENT, model_at_entry, source_paths as original_source_paths


STUDY = "510300_POINT_B01_EXIT_PREDICTION_V1"
OUT = ROOT / "reports/research/510300_point_b01_exit_prediction_v1"
FIELDS = ["ma5_deviation", "ma5_deviation_rv20"]
FEATURES = BASE_FEATURES + FIELDS
KIND = "B01_WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE"
PERIODS = [("2015_2019", "2015-01-05", "2019-12-31"),
           ("2020_2026", "2020-01-02", "2026-09-30")]


def validate_market_clock(data):
    """原财富及经济log收益绑定原价和当日已除息现金。"""
    validate_wealth_clock(data)
    returns = data.total_simple.to_numpy(float)[1:]
    logs = data.total_log.to_numpy(float)[1:]
    require(np.isfinite(logs).all(), "B01经济log收益缺失。")
    np.testing.assert_allclose(logs, np.log1p(returns), atol=1e-12, rtol=0)


def b01_information_block(data):
    """五财富点偏离和20日经济log波动标准化；所有未知原样保留。"""
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_unique and dates.is_monotonic_increasing, "B01日期必须唯一递增。")
    wealth = pd.to_numeric(data.wealth, errors="coerce").astype(float)
    finite = np.isfinite(wealth.to_numpy())
    require((wealth.to_numpy()[finite] > 0).all(), "B01有限财富必须为正。")
    valid = wealth.where(finite)
    complete5 = valid.rolling(5, min_periods=5).count().eq(5)
    average5 = valid.rolling(5, min_periods=5).mean()
    raw = (valid/average5-1).where(complete5)
    logs = pd.to_numeric(data.total_log, errors="coerce").astype(float)
    logs = logs.where(np.isfinite(logs.to_numpy()))
    # 首个财富点没有前一市场区间，不以初始归一值补收益。
    if len(logs):
        logs.iloc[0] = np.nan
    complete21 = valid.rolling(21, min_periods=21).count().eq(21)
    complete_returns = logs.rolling(20, min_periods=20).count().eq(20)
    rv = logs.rolling(20, min_periods=20).std(ddof=1)
    supported = complete21 & complete_returns & rv.gt(0) & raw.notna()
    normalized = (raw/(rv*np.sqrt(5))).where(supported)
    index = np.arange(len(data))
    raw_status = np.where(complete5, "MA5_DEVIATION_AVAILABLE", "NO_VIEW_INCOMPLETE_FIVE_WEALTH_POINTS")
    block_status = np.full(len(data), "NO_VIEW_INCOMPLETE_21_WEALTH_POINTS_OR_20_RETURNS", dtype=object)
    block_status[(complete21 & complete_returns & ~rv.gt(0)).to_numpy()] = "NO_VIEW_ZERO_RETURN_VOLATILITY"
    block_status[supported.to_numpy()] = "B01_BLOCK_AVAILABLE"
    require(np.isfinite(normalized.dropna().to_numpy()).all(), "B01标准化字段出现无穷值。")
    return pd.DataFrame({"date": dates, "origin_index": index, "known_forward_wealth": valid.to_numpy(),
        "known_economic_log_return": logs.to_numpy(), "inclusive_five_point_wealth_mean": average5.to_numpy(),
        "daily_log_std20": rv.to_numpy(), FIELDS[0]: raw.to_numpy(), FIELDS[1]: normalized.to_numpy(),
        "raw_status": raw_status, "field_status": block_status,
        "ma5_start_index": np.where(index >= 4, index-4, -1),
        "rv20_wealth_start_index": np.where(index >= 20, index-20, -1), "latest_source_index": index})


def fit_b01_within(rows, cfg):
    """原固定周期截距岭回归增加已登记两列；不删行，不改变权重或正则化。"""
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(),
            "B01完整训练输入缺失，禁止删行或补值。")
    require(np.isfinite(weights).all() and (weights > 0).all(), "B01周期训练权重无效。")
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
    require(np.isfinite(fitted.coef_).all(), "B01回归系数无效。")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"] -
                                         np.asarray(group["standardized_feature_mean"])@fitted.coef_)
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(np.mean([g["cycle_intercept"] for g in groups])),
            "feature_clip": cfg["feature_clip"], "cycle_intercepts": groups,
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def b01_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "B01模型身份不同。")
    x = np.asarray(values, float)
    require(x.shape == (10,) and np.isfinite(x).all(), "B01预测必须有完整十项状态。")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + z@np.asarray(model["coefficients"]))


class B01EntryController(EntryVintageExitController):
    """实际入场首次收盘锁定B01版本；未知不换模型，两次负预测确认退出。"""

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "B01须在实际首次买入收盘选择模型。")
            self.cycle_id, self.selection_index, self.negative_count = cycle["cycle_id"], t, 0
            k = bisect_right(self.indexes, t)-1
            self.record = self.models[k] if k >= 0 else None
            self.identity = "B01:" + prediction_identity(self.record)
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index,
                "B01本笔固定选择时点或实际周期改变。")
        x = np.r_[state_values(self.data, t, cycle, current_value, peak_value), self.data.loc[t, FIELDS].to_numpy(float)]
        record = self.record
        prediction, status = None, "NO_VIEW_NO_MATURE_MODEL_AT_ENTRY"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index <= t,
                    "B01读取未来成熟周期或月度版本。")
            if np.isfinite(x).all():
                prediction, status = b01_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_ELEVEN_FEATURES"
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
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "B01协议在冻结后改变。")
    for source in frozen["sources"]:
        require(digest(ROOT/source["path"]) == source["sha256"], "B01冻结来源变化："+source["path"])
    return len(frozen["sources"])


def sources():
    """登记原模型/经济合同和本次有限旧用途来源，不运行旧失败方法。"""
    paths = [p for p in original_source_paths() if p not in [
        Path("research/point_volatility_unit_exit_v1.py"), Path("tests/test_point_volatility_unit_exit_v1.py"),
        Path("reports/research/510300_point_method_intake_20261002/prior_direction_review.json")]]
    paths += [Path("research/point_volatility_unit_exit_v1.py"), Path("research/point_a01_exit_prediction_v1.py"),
        Path("research/point_c02_exit_prediction_v1.py"), Path(__file__).relative_to(ROOT),
        Path("tests/test_point_b01_exit_prediction_v1.py"), OUT.relative_to(ROOT)/"prior_and_source_review.json",
        OUT.relative_to(ROOT)/"source_definition_review.json", OUT.relative_to(ROOT)/"tests_receipt.json",
        OUT.relative_to(ROOT)/"原训练成员两列数量支持.csv"]
    review = json.loads((OUT/"source_definition_review.json").read_text(encoding="utf-8"))
    paths += [Path(item["path"]) for item in review["sources"]]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    require(not (OUT/"freeze.json").exists(), "B01已冻结，不覆盖或重复登记。")
    prior = json.loads((OUT/"prior_and_source_review.json").read_text(encoding="utf-8"))
    require(prior["status"] == "ADMITTED_EXISTING_B01_TWO_FIELD_BLOCK_FOR_CONDITIONAL_EXIT_DEVELOPMENT",
        "B01旧用途或完整成员准入未通过。")
    require(prior["original_natural_rows_supported"] == 1507 and prior["original_mature_months_supported"] == 115,
        "B01改变原完整成员数量。")
    tests = json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["passed"] and tests["tests"] == 9 and tests["module_sha256"] == digest(Path(__file__)),
        "B01九项必要测试或代码身份未通过。")
    require(tests["test_sha256"] == digest(ROOT/"tests/test_point_b01_exit_prediction_v1.py"), "B01测试来源改变。")
    protocol = {"study": STUDY, "frozen_at": now(), "candidate_configurations": 1,
        "hypothesis": "既有五日均线偏离及波动标准化相对原八项是否提供继续自然持仓收益的条件增量。",
        "information_role": "EXISTING_MA5_BASELINE_TWO_FIELDS_NEW_CONDITIONAL_USE_NOT_NEW_PHYSICAL_SOURCE",
        "single_change": "原八项同时加入原B01卡raw和标准化两列，共10项；不按效果选版本。",
        "feature_definition": {"fields": FIELDS, "P": "原close/previous_close和当日已除息现金前向累乘财富W，初始1。",
            "raw": "W[t]/mean(W[t-4..t])-1，五完整财富点含当日，跨四差分，有符号不截负。",
            "rv20": "r=原total_log=log1p(total_simple)，r[t-19..t]20完整经济日收益样本std，ddof1含当日不年化；完整21财富点。",
            "normalized": "raw/(rv20*sqrt5)，sqrt5保持原卡。",
            "missing": "真实raw0保留0；缺少5/21财富点、20收益或rv20为0则相应字段/完整块NO_VIEW，不补0或删原行。",
            "application": "两列各自沿原周期等权标准化/clip5线性加入；无交互、阈值、压分、替换原八项或新增入场规则。"},
        "fixed_training": {"last_mature_cycles": 20, "minimum_cycles": 10, "minimum_rows": 100,
            "each_cycle_total_weight": 1., "ridge_alpha": 1., "feature_clip": 5.},
        "unchanged": "原经济标签、全部成熟成员、周期截距、新周期截距等均值、原入场固定版本及两负确认。",
        "source_clock": "完整当日收盘后可知，全部历史至2026-09-30仍开发校准；不使用分钟数据或新增行情。",
        "prediction_gate": "同自然原点原收益单位、周期等权MSE；两时期原预测全部配对且MSE严格改善，入场整年块5000次/seed51030082的改进95%下界都>0。",
        "evaluation_periods": PERIODS, "fit_cache": "完整有序10项、标签、成员、周期权重和设置相同才复用，失败缓存不重复尝试。",
        "economic_gate_if_prediction_passes": "另冻结应用；原20万元252日全日历/BASE及STRESS两时期，新净CAGR和夏普均高于A、实际净pB>1/净平均周期>0/回撤<=10%，频率软目标。",
        "old_failures": "旧R2、均值反弹辅助、锚定分钟反转、多尺度趋势和压力5原用途及所有技术冻结失败保留。",
        "no_rescue": "失败不改5/20窗口、坐标、sqrt5、方向、维度、交互、模型、标签、成员、门、时期或成本，不补跑失败账户。",
        "trial_counting": "一个配置，完整不同成熟输入各一次候选/原对照；复用月份、测试及费用档不是独立试验。",
        "global_DSR_PBO": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    source_list = [{"path": p.as_posix(), "sha256": digest(ROOT/p)} for p in sources()]
    write_json(OUT/"freeze.json", {"frozen_at": now(), "protocol_sha256": digest(OUT/"protocol.json"),
        "sources": source_list}, exclusive=True)
    print(json.dumps({"status": "FROZEN_B01_TWO_FIELD_CONDITIONAL_EXIT_NOT_RUN", "sources": len(source_list)}, ensure_ascii=False), flush=True)


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
    rng = np.random.default_rng(51030082)
    estimates = np.array([np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]).mean()
                          for _ in range(5000)])
    return {"status": "COMPUTED_DEVELOPMENT_YEAR_BLOCK_SENSITIVITY", "years": [int(y) for y in years],
            "iterations": 5000, "seed": 51030082, "low": float(np.quantile(estimates, .025)),
            "high": float(np.quantile(estimates, .975)), "independence_established": False}


def fit_and_evaluate():
    require(not (OUT/"PREDICTION_STARTED.json").exists(), "B01预测阶段已开始，不重跑或覆写。")
    check()
    write_json(OUT/"PREDICTION_STARTED.json", {"at": now()}, exclusive=True)
    data = pd.read_parquet(ROOT/CURRENT/"inputs/candidate_features.parquet")
    validate_market_clock(data)
    factor = b01_information_block(data)
    save_table("日线B01事前字段", factor)
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
                    "B01改变了原成熟训练成员。")
            key = training_identity(rows, cfg)
            first = key not in cache
            if first:
                control = fit_within_cycle_exit(rows, cfg)
                failure, model = None, None
                if np.isfinite(rows[FEATURES].to_numpy(float)).all():
                    try:
                        model = fit_b01_within(rows, cfg)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                        failure = str(error)
                else:
                    failure = "完整训练的B01两列缺失；不删行或补值。"
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
                         "fit_index": original["fit_index"], "missing_market_rows": int(rows[FIELDS].isna().any(axis=1).sum()),
                         "market_coefficients": json.dumps(stored["model"]["coefficients"][-3:]) if stored["model"] else None})
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
                new, status = b01_prediction(new_record["model"], x+extra.tolist()), "PAIRED_PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_CANDIDATE_MODEL_OR_MARKET_BLOCK"
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
                  "independent_trials": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED"}
    save_table("同自然原点配对预测", paired)
    save_table("逐周期预测误差", pd.concat(cycle_tables, ignore_index=True))
    save_table("原八项模型复算", pd.DataFrame(checks))
    save_table("逐月成熟训练及B01信息块", pd.DataFrame(fits))
    write_json(OUT/"trial_accounting.json", accounting, exclusive=True)
    summary = {"study": STUDY, "completed_at": now(),
        "status": "PREDICTION_GATE_PASS_ECONOMIC_APPLICATION_PENDING" if passed else
                  "REJECTED_FIXED_B01_INFORMATION_BLOCK_PREDICTION_GATE_FAILED",
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
    parser = argparse.ArgumentParser(description="B01五日均线偏离固定两列退出预测增量")
    parser.add_argument("action", choices=["freeze", "predict", "check"])
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "predict":
        fit_and_evaluate()
    else:
        print(f"B01冻结来源未变：{check()}份。", flush=True)


if __name__ == "__main__":
    main()
