"""仅核对B04原字段、成员元数据和设计资格，不读收益目标或拟合。"""
import json
import numpy as np
import pandas as pd
from research.point_account_cashflow_state_v1 import ROOT, now, digest, require, write_json
from research.point_b04_information_intake_v1 import check, validate_source_clock, build_field, FIELDS, METADATA, PRICE_COLUMNS
from research.point_b03_optional_correction_v1 import check as check_parent

old = ROOT / "reports/research/510300_point_b04_information_intake_v1"
out = ROOT / "reports/research/510300_point_b03_optional_correction_v1"
current = ROOT / "reports/research/510300_point_current_observation_20261001"
old_source_count, parent_source_count = check(), check_parent()
data = pd.read_parquet(current / "inputs/candidate_features.parquet", columns=PRICE_COLUMNS)
validate_source_clock(data)
daily = build_field(data)
pd.testing.assert_frame_equal(daily, pd.read_parquet(old / "results/日线B04事前字段.parquet"), check_exact=True)
states = pd.read_parquet(current / "results/training_reference/samples.parquet", columns=METADATA)
idx = states.origin_index.to_numpy(int)
field = daily.iloc[idx].reset_index(drop=True).copy()
require(np.array_equal(pd.to_datetime(field.date).to_numpy(), pd.to_datetime(states.origin).to_numpy()), "B04原状态日期变化")
field.insert(0, "cycle_id", states.cycle_id.to_numpy(int))
field.insert(1, "origin_index", idx)
raw = field[FIELDS].to_numpy(float)
known = np.isfinite(raw).all(axis=1)
require(np.array_equal(known, field.field_status.eq("B04_POST_SHOCK_VOLATILITY_AND_LOW_AVAILABLE")), "B04原值状态不一致")
require(len(data) == 3488 and len(states) == 1507 and int(known.sum()) == 285, "B04原字段数量变化")
records = json.loads((current / "inputs/within_models.json").read_bytes().decode("utf-8-sig"))["models"]
months = []
for r in records:
    if not isinstance(r["model"], dict):
        continue
    mask = states.cycle_id.isin(r["training_cycles"]).to_numpy()
    selected = states.loc[mask]
    require(len(selected) == r["training_rows"] and selected.cycle_id.nunique() == len(r["training_cycles"]), "B04原成熟行变动")
    require((selected.exit_index <= r["fit_index"]).all() and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(r["fit_origin"])).all(), "B04目标尚未成熟")
    ids = selected.cycle_id.to_numpy(int)
    unique, counts = np.unique(ids, return_counts=True)
    count_map = dict(zip(unique, counts))
    w = np.array([1 / count_map[i] for i in ids], float)
    x = raw[mask]
    k = np.isfinite(x).all(axis=1)
    phi = np.zeros_like(x)
    if k.any():
        mean = np.average(x[k], axis=0, weights=w[k])
        sd = np.sqrt(np.average((x[k] - mean) ** 2, axis=0, weights=w[k]))
        scale = np.where(sd > 1e-12, sd, 1.)
        z = np.clip((x[k] - mean) / scale, -5., 5.)
        phi[k] = z - np.average(z, axis=0, weights=w[k])
    dx = np.empty_like(phi)
    for i in unique:
        m = ids == i
        dx[m] = phi[m] - np.average(phi[m], axis=0, weights=w[m])
    eig = np.linalg.eigvalsh(dx.T @ (w[:, None] * dx))
    months.append({"fit_index": r["fit_index"], "fit_origin": r["fit_origin"], "original_training_rows": len(selected),
                   "known_rows": int(k.sum()), "unknown_original_rows_retained": int((~k).sum()),
                   "within_design_eigenvalues": eig.tolist(), "both_columns_identifiable": bool((eig > 0).all())})
require(len(months) == 115, "B04原可用月变化")
saved = pd.read_parquet(out / "results/全部原状态B03固定预测.parquet", columns=["cycle_id", "origin_index", "origin", "status"])
require(np.array_equal(saved[["cycle_id", "origin_index"]].to_numpy(int), states[["cycle_id", "origin_index"]].to_numpy(int)), "B04与原预测身份不一致")
available = ~saved.status.eq("NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY").to_numpy()
require(int(available.sum()) == 1010, "B04原可用预测状态变化")
periods = []
for name, start, end in (("2015_2019", "2015-01-05", "2019-12-31"), ("2020_2026", "2020-01-02", "2026-09-30")):
    m = available & pd.to_datetime(states.origin).between(start, end).to_numpy()
    periods.append({"period": name, "original_available_predictions": int(m.sum()), "joint_fields_known": int((m & known).sum()),
                    "exact_fallback_if_implemented": int((m & ~known).sum())})
design_passed = all(x["both_columns_identifiable"] for x in months)
pre = {"at": now(), "status": "PASS_SAME_B04_FIELDS_TWO_COLUMN_DESIGN_ONLY_NO_FIT" if design_passed else "REJECTED_B04_OPTIONAL_TWO_COLUMN_DESIGN_NOT_IDENTIFIABLE_ALL_ORIGINAL_MONTHS",
       "old_R100_source_count": old_source_count, "actual_parent_R133_frozen_source_count": parent_source_count,
       "same_original_daily_fields": True, "daily_rows": len(daily), "natural_rows": len(states), "joint_known_rows": int(known.sum()),
       "joint_unknown_rows_preserved": int((~known).sum()), "ready_months": len(months),
       "identifiable_months": sum(x["both_columns_identifiable"] for x in months),
       "minimum_within_design_eigenvalue": min(x["within_design_eigenvalues"][0] for x in months),
       "periods": periods, "monthly_design_support": months, "target_column_read": False, "future_target_values_read": False,
       "new_model_fits": 0, "new_accounts": 0, "new_network_requests": 0, "old_R100_direct_support_gate_passed": False,
       "complete_B04_all_member_field_admission": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
write_json(out / "next_B04_saved_field_design_preflight.json", pre, exclusive=True)
if design_passed:
    prior = json.loads((old / "prior_and_source_review.json").read_bytes().decode("utf-8-sig"))
    oldsummary = json.loads((old / "summary.json").read_bytes().decode("utf-8-sig"))
    refs = [ROOT / "research/point_b04_information_intake_v1.py"] + [old / n for n in (
        "protocol.json", "freeze.json", "summary.json", "prior_and_source_review.json", "saved_output_recomputation_receipt.json")]
    proposal = {"at": now(), "status": "PROPOSED_NOT_IMPLEMENTED_NOT_FROZEN_NOT_RUN",
        "proposed_study": "510300_POINT_B04_OPTIONAL_CORRECTION_V1", "parent_result_decision": "TECH.R133",
        "hypothesis": "原可知负收益冲击后1—10日，当前三日波动相对冲击前波动及是否跌破冻结冲击低点，能否在已知时补充原八项周期内继续价值。",
        "purpose_and_function_difference": "原TECH.R100仅裁决两列直接追加的全成员支持，0收益模型拟合；另可选两系数函数保留全部原成员及阶段外NaN并精确沿用核心。不重开T03/T14价格或融资入场、不改事件或挑首次收涨，不是新物理信息或独立证据。",
        "registered_definition": prior["registered_definition"], "field_binding": prior["field_binding"],
        "source_clock": prior["source_clock"], "missing_rule": prior["missing_rule"],
        "fixed_function_to_freeze": "仅原波动比及原0/1新低两列，不log/截断原比例、不加年龄/交互/未知指示。已知原训练行按原周期权重标准化clip±5再中心化；未知原值NaN且修正不活动。设计与原目标减固定核心预测分别周期内中心化，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1、新截距0。全部原行/标签/每周期总权重1、原八项和入场锁定保持；联合未知精确返回原核心、原核心未知双方未知。",
        "qualification_sequence": ["原25冻结源、3488原字段、1507身份及115原月两列设计预检完成，未读收益目标。", "下一核旧用途与现行函数去重，完整隔离实现与必要测试，真实拟合前另冻结唯一协议。", "原142月115可用27未知及全部成熟成员/目标保持，只运行一次固定两系数。", "完整原双期配对周期等权MSE均严格下降，5000次seed51030099原入场年块改进95%下界均>0才另冻结完整20万元252日BASE/STRESS账户。", "账户须两期净CAGR/夏普均高于A、实际净pB>1、标准净期望pB-q>0、回撤<=10%，次数软目标；失败拒绝不调参。"],
        "old_R100_facts": {"status": oldsummary["status"], "joint_known_rows": 285, "unknown_rows": 1222,
            "fully_supported_mature_months": 0, "ready_months": 115, "old_model_fits": 0},
        "old_T03_use": prior["old_T03_use"], "old_T14_use": prior["old_T14_use"],
        "old_result_scope": prior["old_result_scope"], "old_saved_account_metrics_reused": prior["old_saved_account_metrics"],
        "targeted_prior_use": prior["other_old_risk_use"],
        "saved_preflight": "reports/research/510300_point_b03_optional_correction_v1/next_B04_saved_field_design_preflight.json",
        "old_source_paths": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in refs],
        "preflight_program": "reports/research/510300_point_b03_optional_correction_v1/next_B04_preflight_and_proposal.py",
        "no_rescue": "不改严格负2倍/20与3日样本std/10日去重和1—10寿命/现金低点/源坐标/原0和比例>1，不加波动已回落筛选/首次收涨确认/丢成员/未知指示/新交互，不log/截断原比例或填0，不改标签/方向/训练/alpha/时期/门，不重跑T03/T14。",
        "old_R100_direct_support_gate_passed": False, "complete_B04_all_member_field_admission": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "physical_first_vintage_verified": False,
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "new_model_fits": 0, "new_return_labels": 0, "new_accounts": 0, "new_market_requests": 0,
        "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    write_json(out / "next_B04_two_optional_residual_proposal.json", proposal, exclusive=True)
print(json.dumps({k: v for k, v in pre.items() if k != "monthly_design_support"}, ensure_ascii=False))
