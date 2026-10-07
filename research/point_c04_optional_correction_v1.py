"""C04回调成交额比的独立单系数可选残差；原完整第九项拒绝保持。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.point_c04_information_intake_v1 import build_field, FIELD
from research.point_optional_residual_model_v1 import design, system
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_volatility_unit_exit_v1 import model_at_entry


CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OLD = ROOT / "reports/research/510300_point_c04_information_intake_v1"
OUT = ROOT / "reports/research/510300_point_c04_optional_correction_v1"
PROPOSAL = ROOT / "reports/research/510300_point_j01_optional_correction_v1/next_C04_stage_optional_residual_proposal.json"
FIELDS = [FIELD]
KIND = "FIXED_CORE_OPTIONAL_C04_SINGLE_WITHIN_CYCLE_RESIDUAL_RIDGE"
STUDY = "510300_POINT_C04_OPTIONAL_CORRECTION_V1"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name: str, frame: pd.DataFrame) -> None:
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def align(states: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "C04原状态身份重复")
    indices = states.origin_index.to_numpy(int)
    require((indices >= 0).all() and (indices < len(daily)).all(), "C04原状态越界")
    field = daily.iloc[indices].reset_index(drop=True).copy()
    require(np.array_equal(indices, field.origin_index.to_numpy(int))
        and np.array_equal(pd.to_datetime(states.origin).to_numpy(dtype="datetime64[ns]"),
                           pd.to_datetime(field.date).to_numpy(dtype="datetime64[ns]")), "C04原状态日期或索引变化")
    field.insert(0, "cycle_id", states.cycle_id.to_numpy(int))
    field["origin_at"] = pd.DatetimeIndex(field.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    field["auxiliary_available"] = np.isfinite(field[FIELD].to_numpy(float))
    require(np.array_equal(field.auxiliary_available, field.field_status.eq("FIELD_AVAILABLE")), "C04原可用状态与值不一致")
    known = field.auxiliary_available
    for key in ["low_confirmation_index", "high_confirmation_index", "phase_start_index", "latest_source_index"]:
        require(field.loc[known, key].le(field.loc[known, "origin_index"]).all(), "C04阶段引用未来确认或来源")
    field["optional_status"] = np.where(known, "KNOWN_OPTIONAL_C04_STAGE_INPUT",
        "NO_VIEW_OPTIONAL_STAGE_EXACT_CORE_FALLBACK")
    return field


def fit(rows: pd.DataFrame, core: dict) -> dict:
    """保持全部原行，仅以一维固定岭闭式解估计辅助系数。"""
    require(len(rows) > 0 and not rows.duplicated(["cycle_id", "origin_index"]).any(), "C04原训练身份为空或重复")
    require(np.isfinite(rows[BASE_FEATURES + ["target", "sample_weight"]].to_numpy(float)).all()
        and (rows.sample_weight > 0).all(), "C04原训练值缺失或权重非法")
    raw = rows[FIELDS].to_numpy(float)
    known = np.isfinite(raw).all(axis=1)
    require(np.array_equal(known, rows.auxiliary_available.to_numpy(bool)), "C04训练原值与分支不符")
    mean, scale, center = np.zeros(1), np.ones(1), np.zeros(1)
    if known.any():
        weights = rows.sample_weight.to_numpy(float)[known]
        mean = np.average(raw[known], axis=0, weights=weights)
        sd = np.sqrt(np.average((raw[known]-mean)**2, axis=0, weights=weights))
        scale = np.where(sd > 1e-12, sd, 1.)
        center = np.average(np.clip((raw[known]-mean)/scale, -5., 5.), axis=0, weights=weights)
    model = {"kind": KIND, "features": FIELDS, "identified": bool(known.any()), "mean": mean.tolist(),
        "scale": scale.tolist(), "clip_center": center.tolist(), "global_intercept": 0.,
        "ridge_alpha": 1., "feature_clip": 5., "training_rows": len(rows),
        "known_training_rows": int(known.sum()), "unknown_training_rows_retained": int((~known).sum())}
    dx, dy, weights = system(rows, core, model)
    gram = float(np.sum(weights*dx[:, 0]**2)+1.)
    rhs = float(np.sum(weights*dx[:, 0]*dy))
    beta = rhs/gram
    require(np.isfinite(beta), "C04辅助系数非法")
    mean_error = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=weights))))
    require(mean_error < 1e-12, "C04修正增加全局截距")
    model.update(coefficients=[beta], normal_equation_gradient_max=abs(gram*beta-rhs),
        global_design_mean_max=mean_error)
    return model


def predict(core: dict, model: dict, base: list | np.ndarray, raw: list | np.ndarray) -> tuple[float, float, str]:
    require(model["kind"] == KIND and model["features"] == FIELDS, "C04修正模型身份不同")
    baseline = within_cycle_prediction(core, base)
    value = np.asarray(raw, float)
    require(value.shape == (1,), "C04固定一个可选原值")
    if not np.isfinite(value).all() or not model["identified"]:
        return baseline, baseline, "EXACT_CORE_FALLBACK"
    phi = np.clip((value-np.asarray(model["mean"]))/np.asarray(model["scale"]), -5., 5.)-np.asarray(model["clip_center"])
    return baseline, baseline+float(phi@np.asarray(model["coefficients"])), "OPTIONAL_CORRECTION_AVAILABLE"


def identity(rows: pd.DataFrame, core: dict) -> str:
    columns = ["cycle_id", "origin_index", "exit_index", "target", "sample_weight", "auxiliary_available"] + BASE_FEATURES + FIELDS
    payload = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(np.uint64).tobytes()
    metadata = {"core": core, "fields": FIELDS, "kind": KIND, "alpha": 1., "clip": 5.}
    return hashlib.sha256(payload+json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_fields() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stock = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet")
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=["cycle_id", "origin_index", "origin"])
    daily = build_field(stock)
    pd.testing.assert_frame_equal(daily, pd.read_parquet(OLD / "results/日线C04阶段及成交额比.parquet"), check_exact=True)
    field = align(states, daily)
    summary = {"status": "PASS_SAME_R76_C04_FIELDS_OPTIONAL_SINGLE_FUNCTION_ONLY", "daily_rows": len(daily),
        "known_daily_rows": int(daily[FIELD].notna().sum()), "original_state_rows": len(field),
        "optional_known_rows": int(field.auxiliary_available.sum()), "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
        "raw_original_status_counts": field.field_status.value_counts().to_dict(),
        "same_as_R76_saved_fields": True, "old_R76_direct_ninth_field_support_gate_passed": False,
        "complete_C04_all_member_field_admission": False, "physical_first_vintage_verified": False,
        "source_clock": "ORIGINAL_CLOSE_1505_CONFIRMED_SWING_AND_PHASE_NO_BACKFILL",
        "history_role": "DEVELOPMENT_CALIBRATION", "new_market_requests": 0}
    require([len(stock), len(states), summary["known_daily_rows"], summary["optional_known_rows"]]
        == [3488, 1507, 1121, 502], "C04原字段或完整成员数量改变")
    return field, daily, summary


def paths() -> list[Path]:
    own = [Path(__file__), ROOT / "tests/test_point_c04_optional_correction_v1.py", PROPOSAL,
        OUT / "prior_definition_and_function_addendum.json", OUT / "tests_receipt.json",
        ROOT / "research/point_optional_residual_model_v1.py", ROOT / "research/point_macro_optional_correction_v1.py",
        ROOT / "research/point_account_cashflow_state_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
        ROOT / "research/within_cycle_exit_inputs_v1.py", ROOT / "research/point_volatility_unit_exit_v1.py",
        ROOT / "research/point_p02_exit_prediction_v1.py", OLD / "freeze.json", OLD / "protocol.json",
        OLD / "summary.json", OLD / "prior_and_source_review.json", OLD / "saved_output_recomputation_receipt.json",
        OLD / "results/日线C04阶段及成交额比.parquet", CURRENT / "inputs/candidate_features.parquet",
        CURRENT / "inputs/within_models.json", CURRENT / "inputs/config/within_cycle_exit.json",
        CURRENT / "results/training_reference/samples.parquet", CURRENT / "results/training_reference/cycles.parquet",
        ROOT / "reports/research/510300_daily_supply_test_v1/result.json"]
    old_freeze = read(OLD / "freeze.json")
    require(digest(OLD / "protocol.json") == old_freeze["protocol_sha256"], "R76原协议变化")
    for record in old_freeze["sources"]:
        path = ROOT / record["path"]
        require(digest(path) == record["sha256"], "R76原冻结来源变化：" + record["path"])
        own.append(path)
    return sorted(set(own))


def check() -> int:
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "C04可选协议变化")
    for record in frozen["sources"]:
        require(digest(ROOT / record["path"]) == record["sha256"], "C04可选冻结对象变化：" + record["path"])
    return len(frozen["sources"])


def freeze() -> None:
    require(not (OUT / "freeze.json").exists(), "唯一C04可选协议已冻结，不覆盖")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] >= 8 and tests["module_sha256"] == digest(Path(__file__))
        and tests["test_sha256"] == digest(ROOT / "tests/test_point_c04_optional_correction_v1.py"), "C04必要测试或版本不符")
    prior = read(OUT / "prior_definition_and_function_addendum.json")
    require(prior["old_R76_direct_rejection_preserved"] and prior["separate_optional_single_function_bound"]
        and not prior["old_direct_support_gate_promoted"], "C04不同函数合同或旧拒绝不符")
    sources = paths()
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R124", "result_decision": "TECH.R125",
        "candidate_configurations": 1, "fields": FIELDS,
        "hypothesis": "原固定回调成交额比的阶段内单系数可选残差，是否增加原自然继续价值的跨期信息。",
        "field": "完全复用R76原确认/现金前向平移/阶段边界；阶段e..t平均成交额除e前5日固定均额，CNY。ATR仅背景，不加入第二字段或深度/期限过滤。",
        "clock": "原T收盘15:05，在T首次确认center=T-2的左右各2根；只在当时已确认上涨段内实时开e，不追溯、不前填。下一合法开盘方可应用。",
        "different_function": "R76仅完整第九项数量准入失败、当前收益模型拟合0，裁决保持。本次只估一个周期内原预测残差系数，原八项固定、全部原成员保留、无源精确回退，不将缺字段填零或准入完整原卡。旧供给测试/力度/签名量/T01组合失败不修改。",
        "original_members": "原1507状态及142月115可用27未知、全部成熟训练成员/原标签/周期总权重1；最近20/至少10周期100行、原入场锁定与核心尺度/截距保持。",
        "fit": "已知原训练单列权重标准化、clip±5再中心化；未知原值NULL，设计修正分支0。全原训练行的设计和原预测残差各在周期内中心化，beta=sum(w*Dx*Dy)/(sum(w*Dx^2)+1)，alpha1、新全局截距0。",
        "missing": "非回调/段失效/初始或坏源仍各自原NO_VIEW；不把不活动修正当成交额值0。原模型未知双方未知，真实0原比值仍有限。",
        "prediction_gate": "完整原配对、周期等权原收益MSE两期均严格改善，入场年块5000次seed51030099改进95%下界均>0。",
        "periods": PERIODS, "economic_gate": "预测门过后另冻结20万元252日BASE/STRESS双期完整账户：净CAGR/净夏普均高于A、实际净pB>1、标准净期望>0、最大回撤<=10%、次数软目标。",
        "no_rescue": "不改R76确认/阶段/价格坐标/前5日/单位/原值缺失，或新增深度/方向/交互；不改单系数/alpha/当前缺失分支/原目标/成员/训练/时期/双期门，不挑有字段子样本评价、不拼失败分支。",
        "old_R76_direct_ninth_field_support_gate_passed": False, "complete_C04_all_member_field_admission": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "physical_first_vintage_verified": False,
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "new_return_labels": 0, "new_market_requests": 0, "overfit_removed": False,
        "goal_achieved": False, "orders_authorized": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources]}, exclusive=True)
    print("C04单系数可选残差协议已冻结，原R76直接第九项拒绝保持。")


def run() -> None:
    require(not (OUT / "RUN_STARTED.json").exists(), "唯一C04可选实验已开始，不重复")
    count = check()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "frozen_sources": count}, exclusive=True)
    field, daily, source_summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "ridge_alpha", "feature_clip"]]
        == [20, 10, 100, 1., 5.], "C04原训练口径改变")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    table("原C04日历阶段字段", daily)
    table("全部原状态C04可选字段", field)
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
                "C04原预测成熟钟改变")
            require(new and new["fit_index"] == old["fit_index"] and new["model"] == old["model"], "C04原模型或入场锁定改变")
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
        and paired.baseline_prediction.isna().sum() == 497, "C04原完整比较成员改变")
    passed = all(row["prediction_gate_passed"] for row in periods)
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R125",
        "status": "PASS_FIXED_C04_OPTIONAL_PREDICTION_GATE" if passed else "REJECTED_FIXED_C04_OPTIONAL_PREDICTION_GATE_FAILED",
        "periods": periods, "prediction_gate_passed": passed,
        "accounting": {"candidate_configurations": 1, "auxiliary_design_columns": 1, "original_monthly_records": 142,
            "available_monthly_records": 115, "original_unknown_months_preserved": 27,
            "distinct_auxiliary_training_inputs": len(cache), "auxiliary_coefficient_estimations": len(cache),
            "monthly_cache_reuses": len(receipts)-len(cache), "core_model_reestimations": 0,
            "new_return_labels": 0, "new_accounts": 0, "new_market_requests": 0},
        "all_natural_rows": len(paired), "paired_available_predictions": len(known), "original_unknown_predictions_preserved": 497,
        "exact_fallback_available_predictions": int(known.status.eq("EXACT_CORE_FALLBACK").sum()),
        "prediction_sign_changes": int(((known.baseline_prediction < 0) != (known.candidate_prediction < 0)).sum()),
        "economic_stage": "READY_FOR_SEPARATE_FIXED_ACCOUNT_REGISTRATION" if passed else "SKIPPED_PREDICTION_GATE_FAILED",
        "net_sharpe": None, "net_cagr": None, "old_R76_direct_ninth_field_support_gate_passed": False,
        "complete_C04_all_member_field_admission": False, "physical_first_vintage_verified": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    for name, frame in [("全部原状态C04固定预测", paired), ("原周期等权误差", losses), ("原成熟训练与复用", pd.DataFrame(receipts))]:
        table(name, frame)
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))


def verify() -> None:
    count = check()
    field, daily, source_summary = load_fields()
    pd.testing.assert_frame_equal(daily, pd.read_parquet(OUT / "results/原C04日历阶段字段.parquet"), check_exact=True)
    pd.testing.assert_frame_equal(field, pd.read_parquet(OUT / "results/全部原状态C04可选字段.parquet"), check_exact=True)
    require(source_summary == read(OUT / "source_summary.json"), "C04来源摘要不符")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    candidates = read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    verified, gradients, means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "C04改变原模型或成员")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "C04补原未知月份")
            continue
        rows = original_training(states, old, cfg)
        key = identity(rows, old["model"])
        require(key == new["input_identity"], "C04训练指纹不符")
        if key not in verified:
            model = new["auxiliary_model"]
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["training_rows"] == len(rows) and model["unknown_training_rows_retained"] == int((~known).sum()),
                "C04丢原未知训练行")
            if known.any():
                raw, w = rows.loc[known, FIELDS].to_numpy(float), rows.loc[known, "sample_weight"].to_numpy(float)
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw-mu)**2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw-mu)/sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0)
                    and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                    and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "C04尺度不能还原")
            dx, dy, w = system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = float(np.max(np.abs(dx.T@(w*(dx@beta-dy))+beta)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=w))))
            require(gradient < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "C04方程或截距不符")
            gradients.append(gradient)
            means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/全部原状态C04固定预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "C04原配对身份变化")
    error, fallbacks = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "C04原标签变化")
        old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "C04补原未知预测")
            continue
        a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES],
            [getattr(state, k) for k in FIELDS])
        require(status == row.status, "C04预测分支不符")
        error = max(error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
        if status == "EXACT_CORE_FALLBACK":
            require(a == b == row.baseline_prediction == row.candidate_prediction, "C04非阶段不是精确原预测")
            fallbacks += 1
    periods, losses = period_results(paired)
    require(periods == read(OUT / "prediction_summary.json")["periods"] and error == 0., "C04预测或区间不符")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权误差.parquet"), check_exact=True)
    result = {"at": now(), "status": "PASS_SAVED_C04_SAME_R76_FIELDS_FIXED_CORE_ALL_MEMBER_SINGLE_EQUATION_PREDICTION_RECOMPUTATION",
        "frozen_sources": count, "daily_rows": len(daily), "natural_field_rows": len(field), "monthly_records": len(candidates),
        "distinct_normal_equations": len(verified), "predictions_checked": len(paired), "cycle_losses_checked": len(losses),
        "exact_fallback_predictions": fallbacks, "maximum_prediction_error": error,
        "maximum_normal_equation_gradient": max(gradients), "maximum_global_design_mean": max(means),
        "new_model_fits": 0, "new_accounts": 0, "old_R76_direct_ninth_field_support_gate_passed": False,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "verification.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="C04阶段内独立单系数可选残差，原直接第九项拒绝保持")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
