"""隔离J01可选周期内残差；严格原源准入失败保持，不认证历史首版。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.point_j01_local_yield_intake_v1 import build_field, LONG, BASE, LEDGER, FIELDS
from research.point_optional_residual_model_v1 import fit, predict, identity, design, system
from research.point_macro_optional_correction_v1 import original_training, period_results
from research.point_p02_exit_prediction_v1 import PERIODS
from research.point_volatility_unit_exit_v1 import model_at_entry


CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OLD = ROOT / "reports/research/510300_point_j01_local_yield_intake_v1"
OUT = ROOT / "reports/research/510300_point_j01_optional_correction_v1"
PROPOSAL = ROOT / "reports/research/510300_point_h03_optional_correction_v1/next_J01_joint_equity_bond_source_proposal.json"
STUDY = "510300_POINT_J01_OPTIONAL_CORRECTION_V1"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name: str, frame: pd.DataFrame) -> None:
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def align(states: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "J01原状态身份重复")
    indices = states.origin_index.to_numpy(int)
    require((indices >= 0).all() and (indices < len(daily)).all(), "J01原状态越界")
    field = daily.iloc[indices].reset_index(drop=True).copy()
    require(np.array_equal(pd.to_datetime(states.origin).to_numpy(dtype="datetime64[ns]"),
                           pd.to_datetime(field.date).to_numpy(dtype="datetime64[ns]")), "J01原状态日期变化")
    field.insert(0, "cycle_id", states.cycle_id.to_numpy(int))
    field.insert(1, "origin_index", indices)
    field["auxiliary_available"] = np.isfinite(field[FIELDS].to_numpy(float)).all(axis=1)
    field["optional_status"] = np.where(field.auxiliary_available,
        "KNOWN_OPTIONAL_ALGORITHMIC_J01_INPUT_FIRST_VINTAGE_UNPROVED",
        "NO_VIEW_OPTIONAL_INPUT_EXACT_CORE_FALLBACK")
    return field


def load_fields() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """只按R104原源/原字段重建；不重跑其准入器或修改旧源门。"""
    stock = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=["date", "symbol", "mom20"])
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=["cycle_id", "origin_index", "origin"])
    rates = pd.read_parquet(LONG)
    daily = build_field(stock, rates)
    old_daily = pd.read_parquet(OLD / "results/日线J01固定利率来源支持.parquet")
    pd.testing.assert_frame_equal(daily, old_daily, check_exact=True)
    field = align(states, daily)
    summary = {"status": "PASS_SAME_R104_FIELDS_OPTIONAL_DEVELOPMENT_FUNCTION_ONLY",
        "raw_yield_rows": len(rates), "first_yield_date": str(rates.date.min().date()),
        "last_yield_date": str(rates.date.max().date()), "daily_rows": len(daily),
        "known_daily_rows": int(np.isfinite(daily[FIELDS].to_numpy(float)).all(axis=1).sum()),
        "original_state_rows": len(field), "optional_known_rows": int(field.auxiliary_available.sum()),
        "raw_missing_rows_preserved": int((~field.auxiliary_available).sum()),
        "exact_same_as_R104_saved_fields": True, "old_R104_strict_source_gate_passed": False,
        "all_original_mature_months_numerically_supported_in_R104": 115,
        "source_clock": "OBSERVATION_DAY_235959_ASSUMED_UPPER_BOUND_THEN_NEXT_STOCK_DAY_1505",
        "physical_first_vintage_verified": False, "complete_J01_all_member_field_admission": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "new_market_requests": 0}
    require([len(stock), len(states), len(rates), summary["optional_known_rows"]] == [3488, 1507, 3660, 1486],
            "J01原来源和状态数量改变")
    return field, daily, summary


def paths() -> list[Path]:
    own = [Path(__file__), ROOT / "tests/test_point_j01_optional_correction_v1.py", PROPOSAL,
        OUT / "prior_definition_and_source_addendum.json", OUT / "tests_receipt.json",
        ROOT / "research/point_j01_local_yield_intake_v1.py", ROOT / "research/point_optional_residual_model_v1.py",
        ROOT / "research/point_macro_optional_correction_v1.py", ROOT / "research/point_account_cashflow_state_v1.py",
        ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
        ROOT / "research/point_volatility_unit_exit_v1.py", ROOT / "research/point_p02_exit_prediction_v1.py",
        OLD / "freeze.json", OLD / "protocol.json", OLD / "summary.json", OLD / "prior_and_source_review.json",
        OLD / "saved_output_recomputation_receipt.json", OLD / "results/日线J01固定利率来源支持.parquet",
        CURRENT / "inputs/candidate_features.parquet", CURRENT / "inputs/within_models.json",
        CURRENT / "inputs/config/within_cycle_exit.json", CURRENT / "results/training_reference/samples.parquet",
        CURRENT / "results/training_reference/cycles.parquet", LONG, BASE, LEDGER]
    old_freeze = read(OLD / "freeze.json")
    require(digest(OLD / "protocol.json") == old_freeze["protocol_sha256"], "R104原协议变化")
    for row in old_freeze["sources"]:
        path = ROOT / row["path"]
        require(digest(path) == row["sha256"], "R104旧冻结来源变化：" + row["path"])
        own.append(path)
    return sorted(set(own))


def check() -> int:
    frozen = read(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "J01可选协议变化")
    for row in frozen["sources"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "J01可选冻结对象变化：" + row["path"])
    return len(frozen["sources"])


def freeze() -> None:
    require(not (OUT / "freeze.json").exists(), "唯一J01可选协议已冻结，不覆盖")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] >= 6 and tests["module_sha256"] == digest(Path(__file__))
        and tests["test_sha256"] == digest(ROOT / "tests/test_point_j01_optional_correction_v1.py"), "J01必要测试或版本不符")
    prior = read(OUT / "prior_definition_and_source_addendum.json")
    require(prior["old_R104_source_rejection_preserved"] and prior["separate_optional_development_function_bound"]
        and not prior["old_strict_source_gate_promoted"], "R104裁决或本次不同函数合同不符")
    sources = paths()
    protocol = {"at": now(), "study": STUDY, "protocol_decision": "TECH.R122", "result_decision": "TECH.R123",
        "candidate_configurations": 1, "fields": FIELDS,
        "hypothesis": "固定十年国债20原股票区间收益率变化及其与原mom20交互，是否增加固定原模型周期内继续残差信息。",
        "source_and_formula": "完全复用R104唯一3660行中债长历史源及原函数：100*(yield[T-1]-yield[T-21])为bp；第二项乘原mom20[T]。不换期限/交互/源，不重复增加mom20列。",
        "clock": "R104观察日23:59:59算法上界，T收盘15:05只用T-1与T-21精确股票日期。未证明每日物理首次发布或历史首版；不拿同日17:30曲线提前到15:05。",
        "distinct_function": "R104仅运行完整字段/首版准入而未拟合模型，失败保持；本次按后续独立可选残差框架，原八项完全固定、所有原成员/成熟钟/锁定版本保留。缺源函数回原预测，完整原字段/首版资格不因此通过。",
        "old_financial_failure": "旧LPR日变动与股价交互的独立20日收益/风险模型失败不变；当前原自然继续目标与周期内无新截距残差不是其窗口/方向/阈值救援。",
        "core_and_training": "原142月115可用27未知、1507状态、原最近20完整成熟周期/至少10周期100行、每周期总权重1；不重新拟合核心。",
        "fit": "仅两个辅助系数；已知训练值权重标准化、clip±5再中心化；未知设计修正分支0但原源字段仍NULL。全部原行的设计及原预测残差分别周期内中心化、alpha1、新全局截距0。",
        "prediction_gate": "完整原配对、周期等权原收益MSE在两固定时期均严格改善，入场年块5000次seed51030099改进95%下界均>0。",
        "periods": PERIODS, "economic_gate": "预测门过后另冻结20万元252日BASE/STRESS双期账户：净CAGR/净夏普同时高于A，实际净pB>1、标准净期望>0、回撤<=10%，次数软目标。",
        "no_rescue": "不改R104源/10年/20区间/日期/单位/mom20/交互、当前缺失分支、alpha、目标、训练、评价年代或门；不挑近期、补缺或拼R117/R119/R121。",
        "old_R104_strict_source_gate_passed": False, "complete_J01_all_member_field_admission": False,
        "physical_first_vintage_verified": False, "history_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "new_return_labels": 0, "new_market_requests": 0, "overfit_removed": False,
        "goal_achieved": False, "orders_authorized": False, "bond_trading_authorized": False}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources]}, exclusive=True)
    print("J01唯一可选历史残差协议已冻结；原R104未准入裁决保持。")


def run() -> None:
    require(not (OUT / "RUN_STARTED.json").exists(), "唯一J01可选实验已开始，不重复")
    count = check()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "frozen_sources": count}, exclusive=True)
    field, daily, source_summary = load_fields()
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    cycles = pd.read_parquet(CURRENT / "results/training_reference/cycles.parquet").set_index("cycle_id")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "ridge_alpha", "feature_clip"]]
        == [20, 10, 100, 1., 5.], "J01原训练口径变化")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    table("原J01日历字段与时钟", daily)
    table("全部原状态J01可选字段", field)
    write_json(OUT / "source_summary.json", source_summary, exclusive=True)
    cache, candidates, receipts = {}, [], []
    for old in originals:
        record = copy.deepcopy(old)
        record["auxiliary_model"] = None
        if old["status"] == "FIT_COMPLETE":
            rows = original_training(states, old, cfg)
            key = identity(rows, old["model"], FIELDS)
            first = key not in cache
            if first:
                cache[key] = {"model": fit(rows, old["model"], FIELDS), "fit_origin": old["fit_origin"]}
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
                "J01原预测成熟钟变化")
            require(new and new["fit_index"] == old["fit_index"] and new["model"] == old["model"], "J01原入场锁定或模型变化")
            a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(row, k) for k in BASE_FEATURES],
                [getattr(row, k) for k in FIELDS], FIELDS)
        predictions.append({"cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index), "origin": row.origin,
            "entry_index": entry, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
            "target": float(row.target), "baseline_prediction": a, "candidate_prediction": b,
            "auxiliary_available": bool(row.auxiliary_available), "status": status})
    paired = pd.DataFrame(predictions)
    periods, losses = period_results(paired)
    known = paired.loc[paired.baseline_prediction.notna()]
    require(len(originals) == 142 and len(receipts) == 115 and len(states) == 1507 and len(known) == 1010
        and paired.baseline_prediction.isna().sum() == 497, "J01原完整比较成员变化")
    passed = all(row["prediction_gate_passed"] for row in periods)
    summary = {"at": now(), "study": STUDY, "technical_decision": "TECH.R123",
        "status": "PASS_FIXED_J01_OPTIONAL_PREDICTION_GATE" if passed else "REJECTED_FIXED_J01_OPTIONAL_PREDICTION_GATE_FAILED",
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
        "net_sharpe": None, "net_cagr": None, "old_R104_strict_source_gate_passed": False,
        "complete_J01_all_member_field_admission": False, "physical_first_vintage_verified": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    for name, frame in [("全部原状态J01固定预测", paired), ("原周期等权误差", losses), ("原成熟训练与复用", pd.DataFrame(receipts))]:
        table(name, frame)
    write_json(OUT / "prediction_summary.json", summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False))


def verify() -> None:
    count = check()
    field, daily, source_summary = load_fields()
    pd.testing.assert_frame_equal(daily, pd.read_parquet(OUT / "results/原J01日历字段与时钟.parquet"), check_exact=True)
    pd.testing.assert_frame_equal(field, pd.read_parquet(OUT / "results/全部原状态J01可选字段.parquet"), check_exact=True)
    require(source_summary == read(OUT / "source_summary.json"), "J01来源摘要未还原")
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    candidates = read(OUT / "candidate_models.json")["models"]
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet")
    cfg = read(CURRENT / "inputs/config/within_cycle_exit.json")
    for key in FIELDS + ["auxiliary_available"]:
        states[key] = field[key].to_numpy()
    verified, gradients, means = set(), [], []
    for old, new in zip(originals, candidates, strict=True):
        require(all(old[k] == new[k] for k in old), "J01改变原模型或成员")
        if old["model"] is None:
            require(new["auxiliary_model"] is None, "J01填补原未知月份")
            continue
        rows = original_training(states, old, cfg)
        key = identity(rows, old["model"], FIELDS)
        require(key == new["input_identity"], "J01训练指纹不符")
        if key not in verified:
            model = new["auxiliary_model"]
            known = rows.auxiliary_available.to_numpy(bool)
            require(model["training_rows"] == len(rows) and model["unknown_training_rows_retained"] == int((~known).sum()),
                "J01丢原未知训练行")
            if known.any():
                raw, w = rows.loc[known, FIELDS].to_numpy(float), rows.loc[known, "sample_weight"].to_numpy(float)
                mu = np.average(raw, axis=0, weights=w)
                sd = np.sqrt(np.average((raw-mu)**2, axis=0, weights=w))
                sd = np.where(sd > 1e-12, sd, 1.)
                center = np.average(np.clip((raw-mu)/sd, -5., 5.), axis=0, weights=w)
                require(np.allclose(mu, model["mean"], atol=1e-13, rtol=0)
                    and np.allclose(sd, model["scale"], atol=1e-13, rtol=0)
                    and np.allclose(center, model["clip_center"], atol=1e-13, rtol=0), "J01尺度不能还原")
            dx, dy, w = system(rows, old["model"], model)
            beta = np.asarray(model["coefficients"])
            gradient = float(np.max(np.abs(dx.T@(w*(dx@beta-dy))+beta)))
            mean = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=w))))
            require(gradient < 1e-12 and mean < 1e-12 and model["global_intercept"] == 0., "J01正规方程或新截距不符")
            gradients.append(gradient)
            means.append(mean)
            verified.add(key)
    paired = pd.read_parquet(OUT / "results/全部原状态J01固定预测.parquet")
    require(np.array_equal(states[["cycle_id", "origin_index"]], paired[["cycle_id", "origin_index"]]), "J01原配对身份变化")
    error, fallbacks = 0., 0
    for state, row in zip(states.itertuples(index=False), paired.itertuples(index=False), strict=True):
        require(float(state.target) == float(row.target), "J01原标签变化")
        old, new = model_at_entry(originals, row.entry_index), model_at_entry(candidates, row.entry_index)
        if not old or old["model"] is None:
            require(np.isnan(row.baseline_prediction) and np.isnan(row.candidate_prediction), "J01填补原未知预测")
            continue
        a, b, status = predict(old["model"], new["auxiliary_model"], [getattr(state, k) for k in BASE_FEATURES],
            [getattr(state, k) for k in FIELDS], FIELDS)
        require(status == row.status, "J01预测分支变化")
        error = max(error, abs(a-row.baseline_prediction), abs(b-row.candidate_prediction))
        if status == "EXACT_CORE_FALLBACK":
            require(a == b == row.baseline_prediction == row.candidate_prediction, "J01缺源不是精确原预测")
            fallbacks += 1
    periods, losses = period_results(paired)
    require(periods == read(OUT / "prediction_summary.json")["periods"] and error == 0., "J01预测或区间不符")
    pd.testing.assert_frame_equal(losses, pd.read_parquet(OUT / "results/原周期等权误差.parquet"), check_exact=True)
    result = {"at": now(), "status": "PASS_SAVED_J01_SAME_R104_FIELDS_FIXED_CORE_ALL_MEMBER_NORMAL_EQUATION_PREDICTION_RECOMPUTATION",
        "frozen_sources": count, "daily_rows": len(daily), "natural_field_rows": len(field), "monthly_records": len(candidates),
        "distinct_normal_equations": len(verified), "predictions_checked": len(paired), "cycle_losses_checked": len(losses),
        "exact_fallback_predictions": fallbacks, "maximum_prediction_error": error,
        "maximum_normal_equation_gradient": max(gradients), "maximum_global_design_mean": max(means),
        "new_model_fits": 0, "new_accounts": 0, "old_R104_strict_source_gate_passed": False,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT / "verification.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="J01独立可选历史开发残差；不改R104源资格")
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
