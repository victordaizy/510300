"""510300相对510500原价格状态的固定两系数持有退出比较。"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.point_optional_residual_model_v1 import (
    design, system, fit as optional_fit, predict as optional_predict, identity as optional_identity, KIND)
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_volatility_unit_exit_v1 import model_at_entry

CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OUT = ROOT / "reports/research/510300_point_relative_price_optional_correction_v1"
OTHER = ROOT / "reports/research/510300_csi300_csi500_relative_price_increment_v1"
COMPARISON = ROOT / "data/raw/all_etf_momentum_v1r/fund_daily_checkpoints/510500_SH.parquet"
MASTER = ROOT / "data/raw/all_etf_momentum_v1/fund_master.parquet"
FIELDS = ["RELATIVE_RAW_PRICE_RETURN_5", "RELATIVE_RAW_PRICE_RETURN_20"]
STUDY = "510300_POINT_RELATIVE_PRICE_OPTIONAL_CORRECTION_V1"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def table(name, frame):
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")

def validate_raw(raw):
    require(raw.ndim == 2 and raw.shape[1] == 2, "相对价格固定两列。")
    require((np.isfinite(raw) | np.isnan(raw)).all(), "无穷值不是合法缺失，不填补。")
    return np.isfinite(raw).all(axis=1)

def validate_dates(frame):
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    require(len(dates) > 0 and not dates.hasnans and dates.is_unique
            and dates.is_monotonic_increasing and dates.tz is None, "原日期必须非空、递增、唯一且无时区。")
    require(dates.equals(dates.normalize()), "原交易日不得夹带盘中时间。")
    return dates

def build_field(stock, comparison):
    dates, other_dates = validate_dates(stock), validate_dates(comparison)
    require(stock.symbol.eq("510300.SH").all(), "执行标的必须为510300.SH。")
    require(comparison.ts_code.eq("510500.SH").all(), "观察来源必须为510500.SH，不能替代为000905.SZ。")
    primary = stock.close.to_numpy(float)
    require(np.isfinite(primary).all() and (primary > 0).all(), "原510300收盘价非法。")
    other_raw = comparison.ts_close.to_numpy(float)
    require((np.isfinite(other_raw) | np.isnan(other_raw)).all()
            and (other_raw[np.isfinite(other_raw)] > 0).all(), "原510500报价非法，缺价保持NaN。")
    secondary = pd.Series(other_raw, index=other_dates).reindex(dates).to_numpy(float)
    daily = pd.DataFrame({"date": dates, "original_510300_close": primary,
                          "original_510500_close": secondary})
    for field, window in zip(FIELDS, [5, 20], strict=True):
        p = pd.Series(primary)
        q = pd.Series(secondary)
        complete = q.notna().rolling(window + 1, min_periods=window + 1).sum().eq(window + 1)
        value = (p / p.shift(window) - 1.) - (q / q.shift(window) - 1.)
        daily[field] = value.where(complete).to_numpy(float)
    daily["known_at"] = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    daily["latest_source_index"] = np.arange(len(daily))
    daily["earliest_5_source_index"] = np.arange(len(daily)) - 5
    daily["earliest_20_source_index"] = np.arange(len(daily)) - 20
    daily["auxiliary_available"] = validate_raw(daily[FIELDS].to_numpy(float))
    return daily

def align(states, daily):
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "原自然身份重复。")
    indices = states.origin_index.to_numpy(int)
    require((indices >= 0).all() and (indices < len(daily)).all(), "原自然索引越界。")
    field = daily.iloc[indices].reset_index(drop=True).copy()
    require(np.array_equal(pd.to_datetime(states.origin).to_numpy(dtype="datetime64[ns]"),
                           pd.to_datetime(field.date).to_numpy(dtype="datetime64[ns]")), "原日期与索引不一致。")
    field.insert(0, "cycle_id", states.cycle_id.to_numpy(int))
    field.insert(1, "origin_index", indices)
    field["origin_at"] = pd.DatetimeIndex(field.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    require((field.known_at <= field.origin_at).all(), "价格公开钟晚于原决定钟。")
    field["optional_status"] = np.where(field.auxiliary_available, "KNOWN_OPTIONAL_RELATIVE_RAW_PRICE",
                                        "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK")
    return field

def load_fields():
    stock = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=["date", "close", "symbol"])
    comparison = pd.read_parquet(COMPARISON, columns=["date", "ts_close", "ts_code"])
    master = pd.read_parquet(MASTER)
    identities = master.loc[master.ts_code.isin(["510300.SH", "510500.SH"])]
    require(len(identities) == 2 and identities.ts_code.is_unique, "两个ETF主表身份缺失或重复。")
    daily = build_field(stock, comparison)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    require(len(stock) == 3488 and len(states) == 1507, "原日线或自然母集变化。")
    field = align(states, daily)
    other = pd.read_parquet(OTHER / "两ETF原价与固定过去相对状态.parquet")
    overlap = daily.set_index("date").loc[pd.DatetimeIndex(other.date)]
    require(np.array_equal(overlap.original_510300_close.to_numpy(float), other.original_510300_close.to_numpy(float)),
            "另一用途与原技术线510300原价不一致，不拼接来源。")
    np.testing.assert_allclose(overlap.original_510500_close.to_numpy(float), other.original_510500_close.to_numpy(float),
                               rtol=0, atol=0, equal_nan=True)
    old_fields = ["510300相对510500原价格五日变化差", "510300相对510500原价格二十日变化差"]
    np.testing.assert_allclose(overlap[FIELDS].to_numpy(float), other[old_fields].to_numpy(float),
                               rtol=0, atol=1e-13, equal_nan=True)
    summary = {"status": "PASS_ORIGINAL_ETF_RELATIVE_RAW_PRICE_OPTIONAL_FUNCTION_ONLY",
               "daily_rows": len(daily), "comparison_source_rows": len(comparison),
               "comparison_source_start": str(comparison.date.min().date()),
               "comparison_source_end": str(comparison.date.max().date()),
               "same_fixed_E70_fields_overlap_rows": len(other),
               "known_daily_rows": int(daily.auxiliary_available.sum()), "original_state_rows": len(field),
               "optional_known_rows": int(field.auxiliary_available.sum()),
               "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
               "partial_known_original_state_rows": int((field[FIELDS].notna().any(axis=1) & ~field.auxiliary_available).sum()),
               "physical_first_vintage_verified": False, "other_branch_E71_failure_reversed": False,
               "complete_relative_price_all_member_field_admission": bool(field.auxiliary_available.all()),
               "source_clock": "RECONSTRUCTED_SESSION_CLOSE_1500_ORIGINAL_DECISION_1505",
               "history_role": "DEVELOPMENT_CALIBRATION", "new_market_requests": 0}
    return field, daily, summary

def fit(rows, core):
    validate_raw(rows[FIELDS].to_numpy(float))
    return optional_fit(rows, core, FIELDS)

def predict(core, model, base_values, raw_values):
    validate_raw(np.asarray(raw_values, float).reshape(1, -1))
    return optional_predict(core, model, base_values, raw_values, FIELDS)

def identity(rows, core):
    return optional_identity(rows, core, FIELDS)

def paths():
    own = [ROOT / "research/point_relative_price_optional_correction_v1.py",
           ROOT / "tests/test_point_relative_price_optional_correction_v1.py", COMPARISON, MASTER,
           OUT / "build_isolated_module.py", OUT / "prior_definition_and_function_addendum.json", OUT / "tests_receipt.json",
           ROOT / "research/point_optional_residual_model_v1.py", ROOT / "research/point_macro_optional_correction_v1.py",
           ROOT / "research/point_account_cashflow_state_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
           ROOT / "research/within_cycle_exit_inputs_v1.py", ROOT / "research/point_volatility_unit_exit_v1.py",
           ROOT / "research/point_p02_exit_prediction_v1.py", ROOT / "research/point_d02_optional_correction_v1.py",
           CURRENT / "inputs/within_models.json", CURRENT / "inputs/config/within_cycle_exit.json",
           CURRENT / "results/training_reference/samples.parquet", CURRENT / "results/training_reference/cycles.parquet",
           CURRENT / "inputs/candidate_features.parquet",
           OTHER / "protocol_source.json", OTHER / "source_result.json", OTHER / "两个ETF身份与原价格来源.json",
           OTHER / "两ETF原价与固定过去相对状态.parquet", OTHER / "实际旧用途与两个ETF观察身份.json",
           OTHER / "protocol.json", OTHER / "result.json", OTHER / "固定失败归因_信号费用与原叶.json",
           ROOT / "config/csi300_etf_rotation_alpha_v1.yaml", ROOT / "reports/backtest/csi300_etf_rotation_alpha_v1.json",
           ROOT / "config/510300_cross_etf_forced_flow_binary_screen_v1_candidates.yaml",
           ROOT / "config/510300_stress_transmission_and_exhaustion_atlas_v1.yaml",
           ROOT / "research/stress_transmission_and_exhaustion_atlas_v1.py",
           ROOT / "reports/research/510300_stress_transmission_and_exhaustion_atlas_v1.json",
           ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json"]
    return sorted(set(own))

def check():
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "相对价格金融协议改变。")
    for item in frozen["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结源改变：" + item["path"])
    return len(frozen["sources"])

def qualify():
    require(not (OUT / "source_protocol.json").exists(), "唯一源与设计资格已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["module_sha256"] == digest(ROOT / "research/point_relative_price_optional_correction_v1.py")
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_relative_price_optional_correction_v1.py"),
            "必要测试或版本不符。")
    sources = paths()
    protocol = {"at": now(), "study": STUDY, "technical_decision": "TECH.R143", "fields": FIELDS,
                "source_clock": "固定原510300/510500未复权价差，过去5/20原交易日，收盘15:00在15:05前；首版未认证。",
                "source_gate": "全部原3488日/1507状态保留；缺任何过去窗口原价保持未知；原115成熟月两列周期内设计标准SVD秩均2。",
                "rank_rule": "sqrt(w)*Dx的标准SVD，tol=max(shape)*float64机器精度*最大奇异值；不按结果选择容差。",
                "target_column_read": False, "new_model_fits": 0,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources]}
    write_json(OUT / "source_protocol.json", protocol, exclusive=True)
    field, daily, summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    raw = field[FIELDS].to_numpy(float)
    models = read(CURRENT / "inputs/within_models.json")["models"]
    months = []
    for record in models:
        if record["model"] is None:
            continue
        mask = states.cycle_id.isin(record["training_cycles"]).to_numpy()
        selected = states.loc[mask]
        require(len(selected) == record["training_rows"] and (selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(), "原成熟成员或时钟改变。")
        ids = selected.cycle_id.to_numpy(int)
        unique, counts = np.unique(ids, return_counts=True)
        count_map = dict(zip(unique, counts, strict=True))
        weights = np.asarray([1 / count_map[i] for i in ids], float)
        x = raw[mask]
        known = np.isfinite(x).all(axis=1)
        phi = np.zeros_like(x)
        if known.any():
            mean = np.average(x[known], axis=0, weights=weights[known])
            sd = np.sqrt(np.average((x[known] - mean)**2, axis=0, weights=weights[known]))
            z = np.clip((x[known] - mean) / np.where(sd > 1e-12, sd, 1.), -5., 5.)
            phi[known] = z - np.average(z, axis=0, weights=weights[known])
        dx = np.empty_like(phi)
        for cycle in unique:
            selected_cycle = ids == cycle
            dx[selected_cycle] = phi[selected_cycle] - np.average(phi[selected_cycle], axis=0, weights=weights[selected_cycle])
        matrix = np.sqrt(weights[:, None]) * dx
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = float(max(matrix.shape) * np.finfo(float).eps * singular[0])
        rank = int((singular > tolerance).sum())
        months.append({"fit_origin": record["fit_origin"], "original_training_rows": len(selected),
                       "known_optional_training_rows": int(known.sum()), "rank": rank,
                       "minimum_singular_value": float(singular[-1]), "svd_tolerance": tolerance})
    require(len(months) == 115, "原115成熟月改变。")
    passed = all(item["rank"] == 2 for item in months)
    result = {"at": now(), "technical_decision": "TECH.R143",
              "status": "PASS_ORIGINAL_RELATIVE_PRICE_SOURCE_AND_TWO_COLUMN_DESIGN_NO_TARGET" if passed
                        else "REJECTED_FIXED_RELATIVE_PRICE_TWO_COLUMN_DESIGN_NOT_IDENTIFIABLE",
              "source_summary": summary, "ready_months": len(months),
              "identifiable_months": sum(item["rank"] == 2 for item in months), "monthly_support": months,
              "target_column_read": False, "new_model_fits": 0, "new_accounts": 0, "new_network_requests": 0,
              "design_gate_passed": passed, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    for item in protocol["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "资格执行中源改变。")
    table("资格_原相对价格日线字段", daily)
    table("资格_全部原状态相对价格字段", field)
    write_json(OUT / "source_and_design_qualification.json", result, exclusive=True)
    print(json.dumps({key:value for key,value in result.items() if key != "monthly_support"}, ensure_ascii=False))

def freeze():
    require(not (OUT / "freeze.json").exists(), "唯一相对价格金融方案已冻结，不重复。")
    qualified = read(OUT / "source_and_design_qualification.json")
    require(qualified["design_gate_passed"] and not qualified["target_column_read"], "源/设计资格失败或已读目标。")
    original = read(OUT / "source_protocol.json")
    for item in original["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "源资格后对象改变。")
    prior = read(OUT / "prior_definition_and_function_addendum.json")
    require(prior["other_branch_E71_failure_preserved"] and prior["different_function_bound"], "用途区别未成立。")
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R144", "result_decision": "TECH.R145",
                "candidate_configurations": 1, "fields": FIELDS, "hypothesis": prior["hypothesis"],
                "different_use": prior["different_use"], "source": original["source_clock"],
                "field": "(510300原close/过去5或20日close-1)-(510500原ts_close/相同原交易日close-1)；两固定窗口全价完整才有值，不是纯风格因果或资金流。",
                "members_and_fit": "原八项/尺度/截距固定，原全部行/目标/周期总权重1及142月115可用27未知、最近20/至少10周期100行和入场锁定不变；联合已知标准化clip正负5中心化，缺失修正不活动。设计与原目标减固定核心的残差各周期内中心化，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1/新截距0。缺原值精确回原核心；核心未知双方未知。",
                "prediction_gate": "完整原双期配对、周期等权原MSE均严格下降且5000原入场年块seed51030099改善95%下界均>0。",
                "periods": PERIODS,
                "economic_gate": "通过预测门后另冻结20万元252日BASE/STRESS双期完整账户：净CAGR/净夏普均高于A、实际净pB>1、标准净期望>0、DD<=10%、次数软目标；原50%上限/10pp带宽及全部风险限制保持。",
                "no_rescue": prior["no_rescue"], "other_branch_E71_failure_preserved": True,
                "normalizer_recomputation": "冻结前统一按原fit的rows[FIELDS].to_numpy(float)[known]选择和归约；保留1e-13，不修改原父模块。",
                "history_role": "DEVELOPMENT_CALIBRATION", "physical_first_vintage_verified": False,
                "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
                "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    sources = paths() + [OUT / "source_protocol.json", OUT / "source_and_design_qualification.json",
                        OUT / "results/资格_原相对价格日线字段.parquet", OUT / "results/资格_全部原状态相对价格字段.parquet"]
    frozen = {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
              "sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in sorted(set(sources))]}
    write_json(OUT / "freeze.json", frozen, exclusive=True)
    print("唯一固定相对价格金融协议已冻结；未读新标签或拟合。")

def run() -> None:
    require(not (OUT / "RUN_STARTED.json").exists(), "唯一RELATIVE_PRICE可选实验已开始，不重复")
    count = check()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "frozen_sources": count}, exclusive=True)
    field, daily, source_summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "ridge_alpha", "feature_clip"]]
        == [20, 10, 100, 1., 5.], "RELATIVE_PRICE原训练口径改变")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    table("原相对价格日线字段", daily)
    table("全部原状态相对价格可选字段", field)
    write_json(OUT / "source_summary.json", source_summary, exclusive=True)
    cache, candidates, receipts = {}, [], []
    for old in originals:
        record = copy.deepcopy(old)
        record["auxiliary_model"] = None
        if old["status"] == "FIT_COMPLETE":
            rows = original_training(states, old, cfg)
            key = identity(rows, old["model"])
            first = key not in cache
            if first:
                cache[key] = {"model": fit(rows, old["model"]), "fit_origin": old["fit_origin"]}
            record.update(auxiliary_model=copy.deepcopy(cache[key]["model"]), input_identity=key, reused=not first,
                first_auxiliary_estimation_origin=cache[key]["fit_origin"])
            receipts.append({"fit_origin": old["fit_origin"], "training_rows": len(rows),
                "training_cycles": len(old["training_cycles"]), "input_identity": key,
                "known_training_rows": record["auxiliary_model"]["known_training_rows"],
                "unknown_training_rows_retained": record["auxiliary_model"]["unknown_training_rows_retained"],
                "first_estimation": first})
        candidates.append(record)
    write_json(OUT / "candidate_models.json", {"at": now(), "models": candidates}, exclusive=True)
    predictions = []
    for row in states.itertuples(index=False):
        cycle = cycles.loc[row.cycle_id]
        entry = int(cycle.entry_index)
        old, new = model_at_entry(originals, entry), model_at_entry(candidates, entry)
        a, b, status = np.nan, np.nan, "NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        if old and old["status"] == "FIT_COMPLETE":
            require(old["latest_exit_index"] <= old["fit_index"] <= entry <= row.origin_index < row.early_exit_index < row.exit_index,
                "RELATIVE_PRICE原预测成熟钟改变")
            require(new and new["fit_index"] == old["fit_index"] and new["model"] == old["model"], "RELATIVE_PRICE原模型或入场锁定改变")
            a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(row, k) for k in BASE_FEATURES],
                [getattr(row, k) for k in FIELDS])
        predictions.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
            "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
            "target": float(row.target), "baseline_prediction": a, "candidate_prediction": b,
            "auxiliary_available": bool(row.auxiliary_available), "status": status})
    paired = pd.DataFrame(predictions)
    periods, losses = period_results(paired)
    known = paired.loc[paired.baseline_prediction.notna()]
    require(len(originals) == 142 and len(receipts) == 115 and len(states) == 1507 and len(known) == 1010
        and paired.baseline_prediction.isna().sum() == 497, "RELATIVE_PRICE原完整比较成员改变")
    passed = all(row["prediction_gate_passed"] for row in periods)
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R145",
        "status": "PASS_FIXED_RELATIVE_PRICE_OPTIONAL_PREDICTION_GATE" if passed else "REJECTED_FIXED_RELATIVE_PRICE_OPTIONAL_PREDICTION_GATE_FAILED",
        "periods": periods, "prediction_gate_passed": passed,
        "accounting": {"candidate_configurations": 1, "auxiliary_design_columns": 2, "original_monthly_records": 142,
            "available_monthly_records": 115, "original_unknown_months_preserved": 27,
            "distinct_auxiliary_training_inputs": len(cache), "auxiliary_coefficient_estimations": len(cache),
            "monthly_cache_reuses": len(receipts)-len(cache), "core_model_reestimations": 0,
            "new_return_labels": 0, "new_accounts": 0, "new_market_requests": 0},
        "all_natural_rows": len(paired), "paired_available_predictions": len(known), "original_unknown_predictions_preserved": 497,
        "exact_fallback_available_predictions": int(known.status.eq("EXACT_CORE_FALLBACK").sum()),
        "prediction_sign_changes": int(((known.baseline_prediction < 0) != (known.candidate_prediction < 0)).sum()),
        "economic_stage": "READY_FOR_SEPARATE_FIXED_ACCOUNT_REGISTRATION" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
        "net_sharpe": None, "net_cagr": None, "other_branch_E71_failure_reversed": False,
        "complete_relative_price_all_member_field_admission": False, "physical_first_vintage_verified": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    for name, frame in [("全部原状态相对价格固定预测", paired), ("原周期等权误差", losses), ("原成熟训练与复用", pd.DataFrame(receipts))]:
        table(name, frame)
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))

def verify() -> None:
    count = check()
    field, daily, source_summary = load_fields()
    pd.testing.assert_frame_equal(daily, pd.read_parquet(OUT / "results/原相对价格日线字段.parquet"), check_exact=True)
    pd.testing.assert_frame_equal(field, pd.read_parquet(OUT / "results/全部原状态相对价格可选字段.parquet"), check_exact=True)
    require(source_summary == read(OUT / "source_summary.json"), "RELATIVE_PRICE来源摘要不符")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    candidates = read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    verified, gradients, means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "RELATIVE_PRICE改变原模型或成员")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "RELATIVE_PRICE补原未知月份")
            continue
        rows = original_training(states, old, cfg)
        key = identity(rows, old["model"])
        require(key == new["input_identity"], "RELATIVE_PRICE训练指纹不符")
        if key not in verified:
            model = new["auxiliary_model"]
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["training_rows"] == len(rows) and model["unknown_training_rows_retained"] == int((~known).sum()),
                "RELATIVE_PRICE丢原未知训练行")
            if known.any():
                raw = rows[FIELDS].to_numpy(float)[known]
                w = rows.sample_weight.to_numpy(float)[known]
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw-mu)**2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw-mu)/sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0)
                    and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                    and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "RELATIVE_PRICE尺度不能还原")
            dx, dy, w = system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = float(np.max(np.abs(dx.T@(w*(dx@beta-dy))+beta)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=w))))
            require(gradient < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "RELATIVE_PRICE方程或截距不符")
            gradients.append(gradient)
            means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/全部原状态相对价格固定预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "RELATIVE_PRICE原配对身份变化")
    error, fallbacks = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "RELATIVE_PRICE原标签变化")
        old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "RELATIVE_PRICE补原未知预测")
            continue
        a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES],
            [getattr(state, k) for k in FIELDS])
        require(status == row.status, "RELATIVE_PRICE预测分支不符")
        error = max(error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
        if status == "EXACT_CORE_FALLBACK":
            require(a == b == row.baseline_prediction == row.candidate_prediction, "RELATIVE_PRICE缺输入不是精确原预测")
            fallbacks += 1
    periods, losses = period_results(paired)
    require(periods == read(OUT / "prediction_summary.json")["periods"] and error == 0., "RELATIVE_PRICE预测或区间不符")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权误差.parquet"), check_exact=True)
    result = {"at": now(), "status": "PASS_SAVED_RELATIVE_PRICE_ORIGINAL_ETF_RAW_PRICE_FIELDS_FIXED_CORE_ALL_MEMBER_TWO_EQUATION_PREDICTION_RECOMPUTATION",
        "frozen_sources": count, "daily_rows": len(daily), "natural_field_rows": len(field), "monthly_records": len(candidates),
        "distinct_normal_equations": len(verified), "predictions_checked": len(paired), "cycle_losses_checked": len(losses),
        "exact_fallback_predictions": fallbacks, "maximum_prediction_error": error,
        "maximum_normal_equation_gradient": max(gradients), "maximum_global_design_mean": max(means),
        "new_model_fits": 0, "new_accounts": 0, "other_branch_E71_failure_reversed": False,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "verification.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description="原相对价格固定两系数隔离研究")
    parser.add_argument("action", choices=["qualify", "freeze", "run", "verify"])
    arguments = parser.parse_args()
    {"qualify": qualify, "freeze": freeze, "run": run, "verify": verify}[arguments.action]()

if __name__ == "__main__":
    main()
