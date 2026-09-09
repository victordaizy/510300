"""只复用保存模型和成熟样本，核对周期内相邻残差相关。"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import training_rows, FEATURES
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_cycle_serial_error_preflight_20260909"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"


def main():
    require(not (OUT / "result.json").exists() and not (OUT / "INPUTS_BOUND.json").exists(), "相邻误差输入核对已经执行")
    config_path = ROOT / "config/510300_within_cycle_exit_v1.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    c128 = ROOT / "config/510300_entry_vintage_exit_v1.json"
    binding = json.loads(c128.read_text(encoding="utf-8"))
    known = {str(Path(row["path"])): row["sha256"] for row in binding["frozen_files"]}
    models_path = P114 / "saved_models.json"
    require(digest(models_path) == known[str(models_path.relative_to(ROOT))], "原114模型不再等于128绑定版本")
    source_path = ROOT / cfg["samples"]
    original_known = {str(Path(row["path"])): row["sha256"] for row in cfg["frozen_files"]}
    require(digest(source_path) == original_known[str(source_path.relative_to(ROOT))], "原自然周期样本改变")
    paths = [Path(__file__), config_path, c128, models_path, source_path, ROOT / cfg["features"], P114 / "extended_reference_samples.parquet",
        P114 / "training_memberships.parquet", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
        ROOT / "docs/510300_CYCLE_SERIAL_ERROR_NEXT_20260909.md", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "reports/research/510300_saved_state_failure_diagnostic_v1/saved_verification_receipt.json"]
    write_json(OUT / "INPUTS_BOUND.json", {"recorded_at": now(), "new_strategy_registered": False,
        "correlation_bounds": [-.99, .99], "files": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in paths]}, exclusive=True)
    raw = pd.read_parquet(source_path)
    samples = raw[raw.signal.eq("D60_INTRA")].copy()
    stored = pd.read_parquet(P114 / "extended_reference_samples.parquet")
    pd.testing.assert_frame_equal(samples.reset_index(drop=True), stored.reset_index(drop=True))
    data = pd.read_parquet(ROOT / cfg["features"])
    models = json.loads(models_path.read_text(encoding="utf-8"))["models"]
    memberships = pd.read_parquet(P114 / "training_memberships.parquet")
    monthly, cycle_details, identities = [], [], set()
    for record in models:
        t = int(record["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == record["training_cycles"] and len(rows) == record["training_rows"], "原月度成熟成员不同")
        require(pd.Timestamp(record["fit_time"]) == data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5), "原月度模型时钟不同")
        require(not len(rows) or rows.exit_index.le(t).all(), "相邻误差读取尚未成熟周期")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (record["status"] == "FIT_COMPLETE"), "原月份支持状态改变")
        if not eligible:
            monthly.append({"fit_origin": record["fit_origin"], "fit_index": t, "status": "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS",
                "training_rows": len(rows), "training_cycles": len(ids), "raw_rho": None, "bounded_rho": None})
            continue
        original = memberships[memberships.fit_index.eq(t)].sort_values(["cycle_id", "origin_index"])
        require(np.array_equal(original[["cycle_id", "origin_index", "exit_index"]], rows[["cycle_id", "origin_index", "exit_index"]]), "保存训练逐行成员不同")
        np.testing.assert_allclose(original.sample_weight, rows.sample_weight, atol=0, rtol=0)
        model = record["model"]
        require(model["features"] == FEATURES and model["kind"] == "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "原残差模型类型改变")
        x, y = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float)
        require(np.isfinite(x).all() and np.isfinite(y).all(), "相邻误差原必要样本缺失")
        z = np.clip((x-np.asarray(model["mean"]))/np.asarray(model["scale"]), -model["feature_clip"], model["feature_clip"])
        beta = np.asarray(model["coefficients"])
        intercepts = {int(g["cycle_id"]): float(g["cycle_intercept"]) for g in model["cycle_intercepts"]}
        residual = y-z@beta-np.array([intercepts[int(cycle)] for cycle in rows.cycle_id])
        numerator, denominator, pairs, gaps, unpaired_cycles = 0., 0., 0, 0, 0
        for cycle in sorted(ids):
            mask = rows.cycle_id.eq(cycle).to_numpy()
            errors = residual[mask]
            origins = rows.origin_index.to_numpy()[mask]
            require(abs(math.fsum(errors)/len(errors)) < 1e-12, "原周期截距不能还原零均值残差")
            adjacent = np.diff(origins) == 1
            k = int(adjacent.sum())
            gaps += int((np.diff(origins) != 1).sum())
            if k:
                previous, current = errors[:-1][adjacent], errors[1:][adjacent]
                a = math.fsum(current*previous)/k
                b = math.fsum(previous**2)/k
                numerator += a
                denominator += b
            else:
                a, b = 0., 0.
                unpaired_cycles += 1
            pairs += k
            cycle_details.append({"fit_origin": record["fit_origin"], "cycle_id": int(cycle), "state_rows": len(errors), "adjacent_pairs": k,
                "pair_mean_cross_product": a, "pair_mean_previous_square": b})
        require(denominator > 0 and pairs > 0, "原残差不能识别相邻相关，停止此输入路线")
        rho = numerator/denominator
        bounded = float(np.clip(rho, -.99, .99))
        payload = {"source_model": model, "members": rows[["cycle_id", "origin_index"]].to_numpy(int).tolist()}
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        seen = identity in identities
        identities.add(identity)
        monthly.append({"fit_origin": record["fit_origin"], "fit_index": t, "status": "SAVED_RESIDUAL_CORRELATION_AVAILABLE",
            "training_rows": len(rows), "training_cycles": len(ids), "adjacent_pairs": pairs, "nonadjacent_gaps": gaps, "cycles_without_pairs": unpaired_cycles,
            "raw_rho": rho, "bounded_rho": bounded, "bound_active": bounded != rho, "source_input_identity": identity,
            "earlier_identical_input_exists": seen})
    available = [row for row in monthly if row["status"] == "SAVED_RESIDUAL_CORRELATION_AVAILABLE"]
    require(len(models) == 141 and len(available) == 114 and len(samples) == 1461, "相邻残差原支持覆盖不同")
    pd.DataFrame(monthly).to_csv(OUT / "saved_monthly_residual_correlation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cycle_details).to_csv(OUT / "saved_cycle_pair_moments.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": now(), "status": "INPUTS_READY_ORIGINAL_EIGHT_FEATURES_AND_MATURE_CLOCKS_PRESERVED",
        "source_state_rows": len(samples), "monthly_records": len(models), "residual_correlation_estimates": len(available),
        "no_mature_model_months": len(models)-len(available), "distinct_source_input_sets": len(identities),
        "first_mature_origin": available[0]["fit_origin"], "rho_min": min(row["raw_rho"] for row in available), "rho_max": max(row["raw_rho"] for row in available),
        "rho_median": float(np.median([row["raw_rho"] for row in available])), "bound_active_months": sum(row["bound_active"] for row in available),
        "monthly_nonadjacent_gap_total": sum(row["nonadjacent_gaps"] for row in available), "new_exit_model_fits": 0, "new_exit_predictions": 0,
        "new_accounts": 0, "new_reference_accounts": 0, "new_strategy_registered": False, "goal_achieved": False, "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 151, "相关误差输入核对的总索引不同")
    index["next_work"].update(status="CYCLE_SERIAL_ERROR_INPUTS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)),
        distinct_source_input_sets=len(identities), new_exit_model_fits=0, new_accounts=0)
    index["updated_at"] = now()
    write_json(index_path, index)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
