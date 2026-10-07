"""固定入场前ATR尺度的持仓回撤：完整原成员准入后做一次继续价值增量比较。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit, within_cycle_prediction
from research.point_volatility_unit_exit_v1 import model_at_entry
from research.point_p02_exit_prediction_v1 import improvement_interval, PERIODS
from research.weekly_daily_technical_v1 import wilder_atr


CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_p03_entry_atr_exit_prediction_v1"
INTAKE = OUT / "intake"
FIELD = "drawdown_in_frozen_entry_atr"
FEATURES = BASE_FEATURES + [FIELD]
KIND = "P03_ENTRY_ATR_WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE"
STUDY = "510300_POINT_P03_ENTRY_ATR_EXIT_PREDICTION_V1"
META = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date", "cycle_drawdown", "log_holding_days"]


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def table(folder: Path, name: str, frame: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f"{name}.parquet", index=False)
    frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def field_values(data: pd.DataFrame, cycles: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只用entry-1的原ATR、已实际发生的entry开盘及当下保存回撤。"""
    required = {"date", "symbol", "open", "high", "low", "close"}
    require(required.issubset(data.columns), "P03原日线字段缺失")
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and dates.is_unique and dates.is_monotonic_increasing and not dates.hasnans,
            "P03原日历必须唯一递增")
    require(data.symbol.eq("510300.SH").all(), "P03标的不同")
    require(not cycles.cycle_id.duplicated().any(), "P03自然周期身份重复")
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "P03原持仓状态重复")
    raw = data[["open", "high", "low", "close"]].to_numpy(float)
    valid = np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1)
    valid &= (raw[:, 1] >= raw[:, 2]) & (raw[:, 1] >= np.maximum(raw[:, 0], raw[:, 3]))
    valid &= raw[:, 2] <= np.minimum(raw[:, 0], raw[:, 3])
    # 保持旧ATR14数学定义；不更改旧数据或把除息跳空解释成纯经济噪声。
    atr = wilder_atr(data.high.where(valid), data.low.where(valid), data.close.where(valid), 14)
    atr = atr.where(valid)
    daily = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)),
                          "raw_ohlc_valid": valid, "raw_wilder_atr14": atr.to_numpy(float)})
    entries = cycles.set_index("cycle_id")
    values = []
    for row in states.itertuples(index=False):
        require(row.cycle_id in entries.index, "P03状态无实际入场周期")
        entry = int(entries.loc[row.cycle_id, "entry_index"])
        origin = int(row.origin_index)
        require(0 <= entry <= origin < len(data), "P03读取未发生的入场或原点")
        require(pd.Timestamp(row.origin) == dates[origin], "P03保存状态日期不同")
        require(np.isclose(row.log_holding_days, np.log1p(origin-entry+1), atol=1e-12, rtol=0),
                "P03持仓时间与真实入场不同")
        prior_atr = float(atr.iloc[entry-1]) if entry > 0 else np.nan
        entry_open = float(data.open.iloc[entry])
        known = entry > 0 and valid[entry] and np.isfinite(prior_atr) and prior_atr > 0
        fraction = prior_atr / entry_open if known else np.nan
        known = bool(known and np.isfinite(fraction) and fraction > 0 and np.isfinite(row.cycle_drawdown))
        value = float(row.cycle_drawdown / fraction) if known else np.nan
        if known:
            require(row.cycle_drawdown <= 1e-12 and np.isfinite(value), "P03回撤或ATR尺度非法")
        values.append({"cycle_id": int(row.cycle_id), "origin_index": origin, "origin": dates[origin],
                       "entry_index": entry, "entry_date": dates[entry], "atr_latest_source_index": entry-1,
                       "atr_known_at": dates[entry-1] if entry > 0 else pd.NaT,
                       "entry_open": entry_open, "frozen_entry_atr14": prior_atr,
                       "frozen_entry_atr_fraction": fraction, "cycle_drawdown": row.cycle_drawdown,
                       FIELD: value, "field_available": known,
                       "status": "ENTRY_ATR_SCALED_DRAWDOWN_AVAILABLE" if known else "NO_VIEW_ENTRY_ATR_OR_CURRENT_DRAWDOWN"})
    return daily, pd.DataFrame(values)


def fit_increment(rows: pd.DataFrame, cfg: dict) -> dict:
    require(cfg["ridge_alpha"] == 1. and cfg["feature_clip"] == 5., "P03原正则化或尺度改变")
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "P03完整训练缺失，禁止删行或补值")
    require(np.isfinite(weights).all() and (weights > 0).all(), "P03原周期权重非法")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -5., 5.)
    dx, dy, groups = np.empty_like(z), np.empty_like(y), []
    ids = rows.cycle_id.to_numpy(int)
    for cycle_id in sorted(set(ids)):
        mask = ids == cycle_id
        require(np.allclose(weights[mask], 1./mask.sum(), atol=1e-14, rtol=0), "P03每周期必须等权一")
        mz, my = np.average(z[mask], axis=0, weights=weights[mask]), float(np.average(y[mask], weights=weights[mask]))
        dx[mask], dy[mask] = z[mask]-mz, y[mask]-my
        groups.append({"cycle_id": int(cycle_id), "rows": int(mask.sum()),
                       "standardized_feature_mean": mz.tolist(), "target_mean": my})
    fitted = Ridge(alpha=1., solver="svd", fit_intercept=False).fit(dx, dy, sample_weight=weights)
    require(np.isfinite(fitted.coef_).all(), "P03模型系数非法")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"]-np.asarray(group["standardized_feature_mean"])@fitted.coef_)
    return {"kind": KIND, "features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(np.mean([g["cycle_intercept"] for g in groups])),
            "feature_clip": 5., "cycle_intercepts": groups,
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def predict(model: dict, values: list | np.ndarray) -> float:
    require(model["kind"] == KIND and model["features"] == FEATURES, "P03模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (9,) and np.isfinite(x).all(), "P03预测需要完整九项")
    z = np.clip((x-np.asarray(model["mean"]))/np.asarray(model["scale"]), -5., 5.)
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


def sources() -> list[Path]:
    relative = ["research/point_p03_entry_atr_exit_prediction_v1.py", "tests/test_point_p03_entry_atr_exit_prediction_v1.py",
                "research/weekly_daily_technical_v1.py", "research/learned_cycle_exit_v1.py",
                "research/within_cycle_exit_inputs_v1.py", "research/point_p02_exit_prediction_v1.py",
                "research/point_volatility_unit_exit_v1.py",
                "reports/research/510300_point_next_information_intake_20261002/G1_P03_N04_P04_prior_definition_plan.json",
                "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json"]
    paths = [ROOT / p for p in relative]
    paths += [CURRENT / p for p in ["inputs/candidate_features.parquet", "inputs/within_models.json",
                                  "inputs/config/within_cycle_exit.json", "results/training_reference/samples.parquet",
                                  "results/training_reference/cycles.parquet"]]
    paths += [OUT / "prior_and_source_review.json", OUT / "tests_receipt.json"]
    return paths


def check(folder: Path) -> int:
    frozen = read(folder / "freeze.json")
    require(digest(folder / "protocol.json") == frozen["protocol_sha256"], "P03冻结协议改变")
    for row in frozen["sources"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "P03冻结来源变化："+row["path"])
    return len(frozen["sources"])


def freeze_intake() -> None:
    require(not INTAKE.exists(), "P03字段阶段已登记，禁止重复")
    prior, tests = read(OUT / "prior_and_source_review.json"), read(OUT / "tests_receipt.json")
    require(prior["conditional_purpose_and_source_clock_bound"] and tests["passed"] and tests["tests"] == 7,
            "P03不同用途、源合同或必要测试未完成")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_p03_entry_atr_exit_prediction_v1.py"), "P03必要测试代码变化")
    INTAKE.mkdir()
    protocol = {"at": now(), "study": STUDY+"_INTAKE", "technical_decision": "TECH.R112",
                "field": FIELD, "formula": prior["formula"], "scope": prior["scope"],
                "clock": "entry-1收盘的ATR14固定，真实entry开盘已发生，当前回撤仅到原点收盘；下一合法开盘才可应用。",
                "missing": "无入场前完整ATR、ATR<=0或当前回撤缺失保持NO_VIEW，不填0/epsilon、删周期或换窗口。",
                "support": "原1507状态及115可用月的全部原训练成员完整；142月原27无模型保持。",
                "return_labels_fits_predictions_accounts": 0, "network_requests": 0,
                "history_role": "DEVELOPMENT_CALIBRATION", "goal_achieved": False}
    write_json(INTAKE / "protocol.json", protocol, exclusive=True)
    write_json(INTAKE / "freeze.json", {"at": now(), "protocol_sha256": digest(INTAKE / "protocol.json"),
              "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources()]}, exclusive=True)
    print("P03入场前ATR回撤定义及数量阶段已冻结；尚未计算收益预测。")


def calculate_intake() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    data = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=["date", "symbol", "open", "high", "low", "close"])
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=META)
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet", columns=["cycle_id", "entry_index"])
    records = read(CURRENT / "inputs/within_models.json")["models"]
    require(len(data) == 3488 and len(states) == 1507 and len(records) == 142, "P03原快照数量改变")
    daily, field = field_values(data, cycles, states)
    monthly = []
    for record in records:
        selected = states.loc[states.cycle_id.isin(record["training_cycles"])]
        require(len(selected) == record["training_rows"] and selected.cycle_id.nunique() == len(record["training_cycles"]),
                "P03原训练成员数量改变")
        require((selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(), "P03原标签未成熟")
        mask = field.loc[field.cycle_id.isin(record["training_cycles"]), "field_available"]
        available = isinstance(record["model"], dict)
        monthly.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
                        "original_model_available": available, "training_cycles": len(record["training_cycles"]),
                        "training_rows": len(selected), "supported_rows": int(mask.sum()),
                        "missing_rows": int((~mask).sum()),
                        "all_original_members_supported": bool(mask.all()) if available else None})
    monthly = pd.DataFrame(monthly)
    ready = monthly.loc[monthly.original_model_available]
    require(len(ready) == 115, "P03原可用模型月数改变")
    passed = bool(field.field_available.all() and ready.all_original_members_supported.all())
    summary = {"status": "PASS_COMPLETE_ORIGINAL_MEMBERS_ENTRY_ATR_FIELD" if passed else "NOT_ADMITTED_ENTRY_ATR_COMPLETE_MEMBER_GATE_FAILED",
               "technical_decision": "TECH.R112", "market_rows": 3488, "natural_rows": 1507,
               "supported_original_natural_rows": int(field.field_available.sum()),
               "original_mature_months": 115, "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
               "original_unknown_months_preserved": 27, "support_gate_passed": passed,
               "monthly_metadata_checks": 142, "feature_ols_calls": 0, "new_return_labels_fits_predictions_accounts": 0,
               "source_clock": "LOCAL_FROZEN_DAILY_HISTORY_PRE_ENTRY_ATR_COMPONENT_NOT_NEW_PHYSICAL_SOURCE",
               "strict_source_first_seen_independence": "NOT_ESTABLISHED", "goal_achieved": False}
    return daily, field, monthly, summary


def run_intake() -> None:
    require(not (INTAKE / "run_started.json").exists(), "P03数量阶段已开始，不重试")
    check(INTAKE)
    write_json(INTAKE / "run_started.json", {"at": now()}, exclusive=True)
    daily, field, monthly, summary = calculate_intake()
    for name, frame in [("日线原ATR14", daily), ("原自然成员入场ATR回撤", field), ("原月度成熟成员支持", monthly)]:
        table(INTAKE / "results", name, frame)
    write_json(INTAKE / "summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))


def freeze_prediction() -> None:
    require(not (OUT / "freeze.json").exists(), "P03固定模型已冻结，不覆盖")
    check(INTAKE)
    summary = read(INTAKE / "summary.json")
    require(summary["support_gate_passed"] and summary["supported_original_natural_rows"] == 1507
            and summary["fully_supported_original_mature_months"] == 115, "P03全部成员未准入")
    protocol = {"at": now(), "study": STUDY, "technical_decision": "TECH.R113", "candidate_configurations": 1,
                "hypothesis": "固定入场前真实波幅尺度的当前回撤是否增加原八项继续价值信息。",
                "features": FEATURES, "added_field": FIELD, "formula": read(OUT / "prior_and_source_review.json")["formula"],
                "scope": "P03回撤尺度组件，非完整卡认证；不重新使用旧高点时间方向或改变实际入场及预算。",
                "training": "原最近20完整成熟周期、至少10周期100行、周期等权、alpha1/clip5、周期固定截距及训练周期截距等均值。",
                "prediction_clock": "实际入场首收盘锁定最近可用原月度版本，保持原8项、目标、全部成员及成熟钟。",
                "prediction_gate": "两固定时期原可用预测完整配对、周期等权原收益MSE严格改善及入场年份块5000次seed51030099改进95%下界均>0。",
                "periods": PERIODS, "new_labels": 0, "economic_gate": "通过后另冻结原20万元252日BASE/STRESS两时期账户；净CAGR与净夏普均高于A，实际净pB>1、标准净期望>0、最大回撤<=10%。次数为软目标。",
                "no_rescue": "不变ATR14、公式、方向、窗口、阈值、目标、权重、正则化、成员、费用或筛子期营救。",
                "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "goal_achieved": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    paths = sources() + [p for p in INTAKE.rglob("*") if p.is_file()]
    unique = sorted(set(paths))
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in unique]}, exclusive=True)
    print("P03唯一固定九项模型已冻结；准备同原八项模型做双期比较。")


def training_identity(rows: pd.DataFrame, cfg: dict) -> str:
    columns = ["cycle_id", "origin_index", "exit_index", "target", "sample_weight"]+FEATURES
    raw = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(np.uint64).tobytes()
    return hashlib.sha256(raw+json.dumps({"ridge_alpha": cfg["ridge_alpha"], "feature_clip": cfg["feature_clip"]}, sort_keys=True).encode()).hexdigest()


def run_prediction() -> None:
    require(not (OUT / "PREDICTION_STARTED.json").exists(), "P03预测已开始，禁止重复")
    check(OUT)
    write_json(OUT / "PREDICTION_STARTED.json", {"at": now()}, exclusive=True)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    fields = pd.read_parquet(INTAKE / "results/原自然成员入场ATR回撤.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]].to_numpy(), fields[["cycle_id", "origin_index"]].to_numpy()),
            "P03源状态顺序不同")
    states[FIELD] = fields[FIELD].to_numpy(float)
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require(cfg["recent_cycles"] == 20 and cfg["minimum_cycles"] == 10 and cfg["minimum_rows"] == 100
            and cfg["ridge_alpha"] == 1. and cfg["feature_clip"] == 5., "P03原训练规则改变")
    cache, candidates, checks, fit_receipts = {}, [], [], []
    for original in originals:
        record = copy.deepcopy(original)
        record["model"] = None
        if original["status"] == "FIT_COMPLETE":
            rows, ids = training_rows(states, original["fit_index"], cfg)
            require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "P03改变原训练成员")
            key = training_identity(rows, cfg)
            first = key not in cache
            if first:
                control = fit_within_cycle_exit(rows, cfg)
                candidate, failure = None, None
                try:
                    candidate = fit_increment(rows, cfg)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    failure = str(error)
                cache[key] = {"control": control, "model": candidate, "failure": failure,
                              "first_fit_origin": original["fit_origin"]}
            stored = cache[key]
            error = max(float(np.max(np.abs(np.asarray(stored["control"][name])-np.asarray(original["model"][name]))))
                        for name in ("mean", "scale", "coefficients", "intercept"))
            require(error <= 1e-12, "P03原八项模型参数无法还原")
            record.update(model=copy.deepcopy(stored["model"]), input_identity=key, reused=not first,
                          status="FIT_COMPLETE" if stored["model"] else "NO_VIEW_MODEL_FIT_FAILED",
                          failure=stored["failure"], first_estimation_origin=stored["first_fit_origin"])
            checks.append({"fit_origin": original["fit_origin"], "parameter_error": error, "first_control_estimation": first})
            fit_receipts.append({"fit_origin": original["fit_origin"], "fit_index": original["fit_index"],
                                 "training_rows": len(rows), "training_cycles": len(ids),
                                 "latest_training_exit_index": int(rows.exit_index.max()), "input_identity": key,
                                 "first_estimation": first, "status": record["status"]})
        candidates.append(record)
    write_json(OUT / "candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    paired = []
    for row in states.itertuples(index=False):
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old_record, new_record = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        old, new, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        values = [getattr(row, name) for name in FEATURES]
        if old_record and old_record["status"] == "FIT_COMPLETE":
            require(old_record["latest_exit_index"] <= old_record["fit_index"] <= entry <= row.origin_index
                    < row.early_exit_index < row.exit_index, "P03原预测或标签成熟时钟错误")
            old = within_cycle_prediction(old_record["model"], values[:-1])
            if new_record and new_record["status"] == "FIT_COMPLETE":
                require(new_record["fit_index"] == old_record["fit_index"], "P03月度配对版本不同")
                new, status = predict(new_record["model"], values), "PAIRED_PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_CANDIDATE_MODEL"
        paired.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
                       "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                       "target": float(row.target), FIELD: getattr(row, FIELD), "baseline_prediction": old,
                       "candidate_prediction": new, "status": status})
    paired = pd.DataFrame(paired)
    periods, tables = [], []
    for name, start, end in PERIODS:
        group = paired.loc[paired.origin.between(pd.Timestamp(start), pd.Timestamp(end))]
        known = group.baseline_prediction.notna()
        matched = group.loc[known & group.candidate_prediction.notna()].copy()
        matched["old_error"] = (matched.baseline_prediction-matched.target)**2
        matched["new_error"] = (matched.candidate_prediction-matched.target)**2
        per_cycle = matched.groupby("cycle_id").agg(baseline_mse=("old_error", "mean"), candidate_mse=("new_error", "mean"),
                                                   entry_year=("entry_year", "first"), rows=("origin_index", "size")).reset_index()
        per_cycle["improvement"] = per_cycle.baseline_mse-per_cycle.candidate_mse
        per_cycle["period"] = name
        tables.append(per_cycle)
        interval = improvement_interval(per_cycle)
        baseline = float(per_cycle.baseline_mse.mean()) if len(per_cycle) else None
        candidate = float(per_cycle.candidate_mse.mean()) if len(per_cycle) else None
        complete = len(matched) == int(known.sum()) and bool(known.any())
        passed = bool(complete and candidate < baseline and interval["low"] is not None and interval["low"] > 0)
        periods.append({"period": name, "original_available_rows": int(known.sum()), "paired_rows": len(matched),
                        "cycles": len(per_cycle), "complete_pair_coverage": complete, "baseline_raw_return_mse": baseline,
                        "candidate_raw_return_mse": candidate, "relative_mse_change": candidate/baseline-1 if baseline else None,
                        "improvement_interval": interval, "prediction_gate_passed": passed})
    passed = all(row["prediction_gate_passed"] for row in periods)
    known = paired.loc[paired.baseline_prediction.notna() & paired.candidate_prediction.notna()]
    signs = int(((known.baseline_prediction < 0) != (known.candidate_prediction < 0)).sum())
    failed = sum(row["model"] is None for row in cache.values())
    accounting = {"candidate_configurations": 1, "monthly_records": len(candidates), "mature_monthly_records": len(checks),
                  "distinct_training_inputs": len(cache), "new_candidate_fit_attempts": len(cache),
                  "successful_candidate_fits": len(cache)-failed, "failed_candidate_fits": failed,
                  "reused_monthly_records": len(checks)-len(cache), "control_coefficient_reestimations": len(cache),
                  "control_monthly_parameter_checks": len(checks), "total_fit_calls": 2*len(cache),
                  "original_unknown_months_preserved": sum(row["model"] is None for row in originals),
                  "new_return_labels": 0, "new_strategy_accounts": 0, "global_DSR_PBO": "NOT_COMPUTED"}
    for name, frame in [("同自然原点配对预测", paired), ("逐周期预测误差", pd.concat(tables, ignore_index=True)),
                        ("原八项模型参数核对", pd.DataFrame(checks)), ("逐月成熟训练", pd.DataFrame(fit_receipts))]:
        table(OUT / "results", name, frame)
    write_json(OUT / "trial_accounting.json", accounting, exclusive=True)
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R113",
               "status": "PREDICTION_GATE_PASSED_APPLICATION_PENDING" if passed else "REJECTED_FIXED_ENTRY_ATR_DRAWDOWN_PREDICTION_GATE_FAILED",
               "periods": periods, "prediction_gate_passed": passed, "paired_sign_changes_not_trades": signs,
               "original_unknown_prediction_rows": int(paired.baseline_prediction.isna().sum()),
               "trial_accounting": accounting, "history_role": "DEVELOPMENT_CALIBRATION",
               "independent_validation": "NOT_ESTABLISHED", "overall_overfit_removed": False,
               "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED", "goal_achieved": False}
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    write_json(OUT / "economic_stage_status.json", {"at": now(), "status": "NOT_RUN_PREDICTION_PASS_APPLICATION_PENDING" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
               "new_strategy_accounts": 0, "account_return_sharpe": "NOT_COMPUTED"}, exclusive=True)
    print(json.dumps({"状态": summary["status"], "双期结果": periods, "拟合记账": accounting, "符号变化非交易": signs}, ensure_ascii=False))


def verify() -> None:
    check(INTAKE)
    daily, field, monthly, summary = calculate_intake()
    require(summary == read(INTAKE / "summary.json"), "P03保存数量结论不同")
    for name, expected in [("日线原ATR14", daily), ("原自然成员入场ATR回撤", field), ("原月度成熟成员支持", monthly)]:
        pd.testing.assert_frame_equal(pd.read_parquet(INTAKE / f"results/{name}.parquet"), expected, check_exact=True)
    prediction_checks = 0
    if (OUT / "prediction_summary.json").exists():
        check(OUT)
        states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
        states[FIELD] = field[FIELD].to_numpy(float)
        originals = read(CURRENT / "inputs/within_models.json")["models"]
        candidates = read(OUT / "candidate_models.json")["models"]
        saved = pd.read_parquet(OUT / "results/同自然原点配对预测.parquet")
        for row, state in zip(saved.itertuples(index=False), states.itertuples(index=False), strict=True):
            require(row.cycle_id == state.cycle_id and row.origin_index == state.origin_index and row.target == state.target,
                    "P03保存原成员或目标改变")
            if row.status == "PAIRED_PREDICTION_AVAILABLE":
                old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
                require(old["fit_index"] == new["fit_index"] <= row.entry_index <= row.origin_index,
                        "P03保存入场锁定版本时钟不同")
                x = [getattr(state, name) for name in FEATURES]
                require(np.isclose(row.baseline_prediction, within_cycle_prediction(old["model"], x[:-1]), atol=1e-14, rtol=0)
                        and np.isclose(row.candidate_prediction, predict(new["model"], x), atol=1e-14, rtol=0), "P03保存预测不同")
                prediction_checks += 1
        report = read(OUT / "prediction_summary.json")
        cycle_table = pd.read_parquet(OUT / "results/逐周期预测误差.parquet")
        for period in report["periods"]:
            block = cycle_table.loc[cycle_table.period.eq(period["period"])]
            require(improvement_interval(block) == period["improvement_interval"], "P03保存区间不同")
            require(np.isclose(block.baseline_mse.mean(), period["baseline_raw_return_mse"], atol=1e-16, rtol=0)
                    and np.isclose(block.candidate_mse.mean(), period["candidate_raw_return_mse"], atol=1e-16, rtol=0), "P03保存周期MSE不同")
        if not report["prediction_gate_passed"]:
            require(not (OUT / "accounts").exists() and read(OUT / "economic_stage_status.json")["new_strategy_accounts"] == 0,
                    "P03预测失败仍出现新账户")
    write_json(OUT / "saved_verification_receipt.json", {"at": now(),
               "status": "PASS_SAVED_ENTRY_ATR_FIELD_AND_FIXED_ENTRY_PREDICTIONS",
               "intake_rows": [len(daily), len(field), len(monthly)], "saved_paired_predictions_checked": prediction_checks,
               "new_model_fits": 0, "new_labels_accounts_network": 0,
               "bootstrap_recomputation_not_new_trials": 10000 if prediction_checks else 0,
               "scope": "源字段/完整成员及保存预测和周期误差核对，未重拟合；不证明独立验证或金融改善。"}, exclusive=True)
    print(f"P03保存字段与{prediction_checks}个入场固定配对预测核对通过；无重拟合或新账户。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="入场前ATR尺度回撤的固定增量实验")
    parser.add_argument("operation", choices=["freeze_intake", "run_intake", "freeze_prediction", "run_prediction", "verify"])
    args = parser.parse_args()
    {"freeze_intake": freeze_intake, "run_intake": run_intake, "freeze_prediction": freeze_prediction,
     "run_prediction": run_prediction, "verify": verify}[args.operation]()
