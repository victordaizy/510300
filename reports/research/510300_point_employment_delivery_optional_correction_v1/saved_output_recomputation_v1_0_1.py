"""独立复算修复：精确采用原fit的二维布尔索引归约，原容差/模型/结果保持。"""
import numpy as np
import pandas as pd
from research.point_employment_delivery_optional_correction_v1 import (
    CURRENT, OUT, FIELDS, BASE_FEATURES, check, load_fields, read, original_training,
    identity, system, design, model_at_entry, predict, period_results, require, now, write_json,
)
from research.point_account_cashflow_state_v1 import digest

def verify_v1_0_1() -> None:
    count = check()
    field, daily, source_summary = load_fields()
    pd.testing.assert_frame_equal(daily, pd.read_parquet(OUT / "results/原月度就业配送字段.parquet"), check_exact=True)
    pd.testing.assert_frame_equal(field, pd.read_parquet(OUT / "results/全部原状态就业配送可选字段.parquet"), check_exact=True)
    require(source_summary == read(OUT / "source_summary.json"), "EMPLOYMENT_DELIVERY来源摘要不符")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    candidates = read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    verified, gradients, means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "EMPLOYMENT_DELIVERY改变原模型或成员")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "EMPLOYMENT_DELIVERY补原未知月份")
            continue
        rows = original_training(states, old, cfg)
        key = identity(rows, old["model"])
        require(key == new["input_identity"], "EMPLOYMENT_DELIVERY训练指纹不符")
        if key not in verified:
            model = new["auxiliary_model"]
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["training_rows"] == len(rows) and model["unknown_training_rows_retained"] == int((~known).sum()),
                "EMPLOYMENT_DELIVERY丢原未知训练行")
            if known.any():
                raw = rows[FIELDS].to_numpy(float)[known]
                w = rows.sample_weight.to_numpy(float)[known]
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw-mu)**2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw-mu)/sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0)
                    and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                    and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "EMPLOYMENT_DELIVERY尺度不能还原")
            dx, dy, w = system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = float(np.max(np.abs(dx.T@(w*(dx@beta-dy))+beta)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=w))))
            require(gradient < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "EMPLOYMENT_DELIVERY方程或截距不符")
            gradients.append(gradient)
            means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/全部原状态就业配送固定预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "EMPLOYMENT_DELIVERY原配对身份变化")
    error, fallbacks = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "EMPLOYMENT_DELIVERY原标签变化")
        old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "EMPLOYMENT_DELIVERY补原未知预测")
            continue
        a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES],
            [getattr(state, k) for k in FIELDS])
        require(status == row.status, "EMPLOYMENT_DELIVERY预测分支不符")
        error = max(error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
        if status == "EXACT_CORE_FALLBACK":
            require(a == b == row.baseline_prediction == row.candidate_prediction, "EMPLOYMENT_DELIVERY缺输入不是精确原预测")
            fallbacks += 1
    periods, losses = period_results(paired)
    require(periods == read(OUT / "prediction_summary.json")["periods"] and error == 0., "EMPLOYMENT_DELIVERY预测或区间不符")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权误差.parquet"), check_exact=True)
    result = {"at": now(), "status": "PASS_SAVED_EMPLOYMENT_DELIVERY_ORIGINAL_LOCAL_FIELDS_FIXED_CORE_ALL_MEMBER_TWO_EQUATION_PREDICTION_RECOMPUTATION",
        "frozen_sources": count, "monthly_reports": len(daily), "natural_field_rows": len(field), "monthly_records": len(candidates),
        "distinct_normal_equations": len(verified), "predictions_checked": len(paired), "cycle_losses_checked": len(losses),
        "exact_fallback_predictions": fallbacks, "maximum_prediction_error": error,
        "maximum_normal_equation_gradient": max(gradients), "maximum_global_design_mean": max(means),
        "new_model_fits": 0, "new_accounts": 0, "other_branch_E65_failure_reversed": False,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    result.update(verification_version="v1_0_1_same_original_fit_array_layout", initial_verification_command_failed=True, normalizer_tolerance_preserved=1e-13, normalizer_same_fit_layout_difference=0., repair_program="reports/research/510300_point_employment_delivery_optional_correction_v1/saved_output_recomputation_v1_0_1.py", repair_program_sha256=digest(OUT / "saved_output_recomputation_v1_0_1.py"), initial_failure_record="initial_verification_failure_and_diagnosis.json")
    write_json(OUT / "verification.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    verify_v1_0_1()
