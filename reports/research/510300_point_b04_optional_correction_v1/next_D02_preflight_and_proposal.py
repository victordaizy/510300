"""仅核对D02原字段、成员元数据和设计资格，不读收益目标或拟合。"""
import json
import numpy as np
import pandas as pd
from research.point_account_cashflow_state_v1 import ROOT, now, digest, require, write_json
from research.point_d02_information_intake_v1 import check, validate_source_clock, build_field, FIELDS, METADATA, PRICE_COLUMNS
from research.point_b04_optional_correction_v1 import check as check_parent

old = ROOT / "reports/research/510300_point_d02_information_intake_v1"
out = ROOT / "reports/research/510300_point_b04_optional_correction_v1"
current = ROOT / "reports/research/510300_point_current_observation_20261001"
old_source_count, parent_source_count = check(), check_parent()
data = pd.read_parquet(current / "inputs/candidate_features.parquet", columns=PRICE_COLUMNS)
validate_source_clock(data)
daily = build_field(data)
pd.testing.assert_frame_equal(daily, pd.read_parquet(old / "results/日线D02事前字段.parquet"), check_exact=True)
states = pd.read_parquet(current / "results/training_reference/samples.parquet", columns=METADATA)
idx = states.origin_index.to_numpy(int)
field = daily.iloc[idx].reset_index(drop=True).copy()
require(np.array_equal(pd.to_datetime(field.date).to_numpy(), pd.to_datetime(states.origin).to_numpy()), "D02原状态日期变化")
field.insert(0, "cycle_id", states.cycle_id.to_numpy(int))
field.insert(1, "origin_index", idx)
raw = field[FIELDS].to_numpy(float)
known = np.isfinite(raw).all(axis=1)
require(np.array_equal(known, field.field_status.eq("D02_NEGATIVE_GAP_ABSORPTION_AVAILABLE")), "D02原值状态不一致")
require(len(data) == 3488 and len(states) == 1507 and int(known.sum()) == 762, "D02原字段数量变化")
records = json.loads((current / "inputs/within_models.json").read_bytes().decode("utf-8-sig"))["models"]
months = []
for r in records:
    if not isinstance(r["model"], dict):
        continue
    mask = states.cycle_id.isin(r["training_cycles"]).to_numpy()
    selected = states.loc[mask]
    require(len(selected) == r["training_rows"] and selected.cycle_id.nunique() == len(r["training_cycles"]), "D02原成熟行变动")
    require((selected.exit_index <= r["fit_index"]).all() and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(r["fit_origin"])).all(), "D02目标尚未成熟")
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
require(len(months) == 115, "D02原可用月变化")
saved = pd.read_parquet(out / "results/全部原状态B04固定预测.parquet", columns=["cycle_id", "origin_index", "origin", "status"])
require(np.array_equal(saved[["cycle_id", "origin_index"]].to_numpy(int), states[["cycle_id", "origin_index"]].to_numpy(int)), "D02与原预测身份不一致")
available = ~saved.status.eq("NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY").to_numpy()
require(int(available.sum()) == 1010, "D02原可用预测状态变化")
periods = []
for name, start, end in (("2015_2019", "2015-01-05", "2019-12-31"), ("2020_2026", "2020-01-02", "2026-09-30")):
    m = available & pd.to_datetime(states.origin).between(start, end).to_numpy()
    periods.append({"period": name, "original_available_predictions": int(m.sum()), "joint_fields_known": int((m & known).sum()),
                    "exact_fallback_if_implemented": int((m & ~known).sum())})
design_passed = all(x["both_columns_identifiable"] for x in months)
pre = {"at": now(), "status": "PASS_SAME_D02_FIELDS_TWO_COLUMN_DESIGN_ONLY_NO_FIT" if design_passed else "REJECTED_D02_OPTIONAL_TWO_COLUMN_DESIGN_NOT_IDENTIFIABLE_ALL_ORIGINAL_MONTHS",
       "old_R94_source_count": old_source_count, "actual_parent_R135_frozen_source_count": parent_source_count,
       "same_original_daily_fields": True, "daily_rows": len(daily), "natural_rows": len(states), "joint_known_rows": int(known.sum()),
       "joint_unknown_rows_preserved": int((~known).sum()), "ready_months": len(months),
       "identifiable_months": sum(x["both_columns_identifiable"] for x in months),
       "minimum_within_design_eigenvalue": min(x["within_design_eigenvalues"][0] for x in months),
       "periods": periods, "monthly_design_support": months, "target_column_read": False, "future_target_values_read": False,
       "new_model_fits": 0, "new_accounts": 0, "new_network_requests": 0, "old_R94_direct_support_gate_passed": False,
       "complete_D02_all_member_field_admission": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
write_json(out / "next_D02_saved_field_design_preflight.json", pre, exclusive=True)
if design_passed:
    prior = json.loads((old / "prior_and_source_review.json").read_bytes().decode("utf-8-sig"))
    oldsummary = json.loads((old / "summary.json").read_bytes().decode("utf-8-sig"))
    refs = [ROOT / "research/point_d02_information_intake_v1.py"] + [old / n for n in (
        "protocol.json", "freeze.json", "summary.json", "prior_and_source_review.json", "saved_output_recomputation_receipt.json")]
    proposal = {"at": now(), "status": "PROPOSED_NOT_IMPLEMENTED_NOT_FROZEN_NOT_RUN",
        "proposed_study": "510300_POINT_D02_OPTIONAL_CORRECTION_V1", "parent_result_decision": "TECH.R135",
        "hypothesis": "原严格负隔夜缺口被当日日内正经济log收益吸收的比例及原有符号缺口，在两项均有定义时能否补充原八项周期内继续价值。",
        "purpose_and_function_difference": "原TECH.R94仅裁决两列直接追加的全原成员支持，0收益模型拟合；另可选两系数函数保留全部原成员和原缺口值/吸收NaN，完整输入才活动、联合不完整精确沿用核心。不重开HIGH/LOW、gap_recovery、DAILY_01或PCA吸收率旧用途；不是新物理信息或独立证据。",
        "registered_definition": prior["registered_definition"], "field_binding": prior["field_binding"],
        "boundary_binding": prior["boundary_binding_before_quantity"], "source_clock": prior["source_clock"],
        "missing_rule": prior["missing_rule"],
        "fixed_function_to_freeze": "仅原negative_gap_absorption与signed_overnight_gap两列。原gap有符号：联合已知时严格负；吸收比例非负、真0及大于1均合法，不先log/截断原比例或给分母epsilon。正/零隔夜的吸收NaN及仍已知原gap各自保持，不将两列强制同时NaN或填0。两项完整原训练行按原周期权重标准化clip±5再中心化，其余行修正不活动。全部原行/目标/每周期总权重1、原八项和入场锁定保持。设计与原目标减固定核心预测分别周期内去均值，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1、新截距0；联合不完整精确返回原核心，原核心未知双方未知。不加缺失指示或条件子样本。",
        "qualification_sequence": ["原26冻结源、3488原字段、1507身份及115月两列设计预检完成，未读收益目标。", "下一核原日内/隔夜/缺口用途与现行可选函数去重，建立完整隔离代码和必要测试，真实收益拟合前另冻结唯一协议。", "原142月115可用27未知/全部原成熟行和目标保持，只运行一次两系数模型。", "完整原双期配对周期等权MSE均严格下降、5000次seed51030099原入场年块改进95%下界皆>0才另冻结完整20万元252日BASE/STRESS账户。", "账户要求两期净CAGR/夏普皆高于A、实际净pB>1、标准净期望pB-q>0、回撤<=10%，次数软目标；失败不调参救援。"],
        "old_R94_facts": {"status": oldsummary["status"], "joint_known_rows": 762, "joint_unknown_rows": 745,
            "fully_supported_mature_months": 0, "ready_months": 115, "old_model_fits": 0},
        "old_rapid_use": prior["old_rapid_use"], "old_D02_saved_account_rows_reused": prior["old_D02_saved_account_rows"],
        "old_gap_recovery_use": prior["old_gap_recovery_use"], "old_daily01_use": prior["old_daily01_use"],
        "old_session_related_use": prior["old_session_related_use"], "homonym_correction": prior["homonym_correction"],
        "old_metrics_scope": "只复用原R94抽取的8条旧D02保存指标，2万/20万元及两成本不混排；不重跑旧账户，也不将旧时期/年化口径替代当前252日双期。",
        "saved_preflight": "reports/research/510300_point_b04_optional_correction_v1/next_D02_saved_field_design_preflight.json",
        "old_source_paths": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in refs],
        "preflight_program": "reports/research/510300_point_b04_optional_correction_v1/next_D02_preflight_and_proposal.py",
        "no_rescue": "不改严格负缺口整数边界/当日现金经济log分解/原比例/有符号gap/真0与大于1/原部分未知；不改窗口、维度、成员/目标/方向/alpha/时期/成本/双期门，不加缺失指示/条件子样本，不重跑HIGH/LOW、gap_recovery、DAILY_01或移植PCA失败。",
        "old_R94_direct_support_gate_passed": False, "complete_D02_all_member_field_admission": False,
        "history_role": "DEVELOPMENT_CALIBRATION", "physical_first_vintage_verified": False,
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "new_model_fits": 0, "new_return_labels": 0, "new_accounts": 0, "new_market_requests": 0,
        "overfit_removed": False, "goal_achieved": False, "orders_authorized": False}
    write_json(out / "next_D02_two_optional_residual_proposal.json", proposal, exclusive=True)
print(json.dumps({k: v for k, v in pre.items() if k != "monthly_design_support"}, ensure_ascii=False))
