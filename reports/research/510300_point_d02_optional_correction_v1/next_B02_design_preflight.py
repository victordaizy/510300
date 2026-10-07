"""事前登记B02原五列的设计资格；不读取收益目标、不估计金融模型。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, now, digest, require, write_json
from research.point_b02_information_intake_v1 import (
    FIELDS, METADATA, PRICE_COLUMNS, build_field, check as check_old,
    validate_source_clock,
)
from research.point_d02_optional_correction_v1 import check as check_parent


OUT = ROOT / "reports/research/510300_point_d02_optional_correction_v1"
OLD = ROOT / "reports/research/510300_point_b02_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
PROTOCOL = OUT / "next_B02_design_protocol.json"
RESULT = OUT / "next_B02_saved_field_design_preflight.json"


def main():
    require(not PROTOCOL.exists() and not RESULT.exists(), "B02资格已登记或运行，不重复执行。")
    old_count, parent_count = check_old(), check_parent()
    require(old_count == 22 and parent_count == 49, "B02原源或D02父结果冻结数量变化。")
    paths = [
        OUT / "next_B02_design_preflight.py", ROOT / "research/point_b02_information_intake_v1.py",
        OLD / "protocol.json", OLD / "freeze.json", OLD / "summary.json",
        OLD / "prior_and_source_review.json", OLD / "saved_output_recomputation_receipt.json",
        OLD / "results/日线B02事前字段.parquet",
        CURRENT / "inputs/candidate_features.parquet",
        CURRENT / "results/training_reference/samples.parquet",
        CURRENT / "inputs/within_models.json",
        OUT / "prediction_summary.json", OUT / "verification.json",
        OUT / "results/全部原状态D02固定预测.parquet",
    ]
    protocol = {
        "at": now(), "study": "510300_POINT_B02_FIVE_COLUMN_DESIGN_QUALIFICATION_V1",
        "technical_decision": "TECH.R138", "parent_actual_model_decision": "TECH.R137",
        "hypothesis": "原二次试低五项记录是否在全部115原可用成熟月提供可识别的五列周期内设计。",
        "fields": FIELDS, "old_R102_frozen_sources": old_count,
        "parent_R137_frozen_sources": parent_count,
        "fixed_source_definition": "原20日低点、a后两完整日确认、a+3至a+10认领、距离严格小于0.5冻结ATR；本次金额/原首次前后完整五日均额、a及b原经济log收益、两次原CLV。原NaN/真实0与1/事件认领不变，不延窗、不加衰减成功筛选、不删字段。",
        "training_identity": "只读原样本身份/成熟时钟和原142模型元数据；115原可用月全部成熟原成员，每周期总权重1，27原未知月不新建。",
        "design": "完整五项已知的原行按原周期权重标准化、clip正负5并再次权重中心化；未知原行仍保留且修正设计不活动。然后按全部原行作周期内中心化。",
        "rank_rule_frozen_before_measurement": "对sqrt(w)*Dx作SVD；tol=max(矩阵形状)*float64机器精度*最大奇异值，rank=count(singular_values>tol)。所有115原可用月必须rank=5；岭惩罚矩阵可逆不等于数据识别。没有额外最低交易数门槛。",
        "failure_action": "任何原可用月rank<5则拒绝本固定五列可选函数资格，金融拟合/账户0，不删列、换窗、填未知、跳月或选择有利后半期救回。",
        "success_action": "若115月全部rank5，只接受设计资格；另做已有用途核对和必要测试，收益模型须另冻结，当前仍NOT_RUN。",
        "target_column_read": False, "future_target_values_parsed": False,
        "new_model_fits": 0, "new_accounts": 0, "new_market_requests": 0,
        "pre_protocol_technical_attempt": "首次运行在生成协议前因reports目录联接解析成E盘路径、无法生成工作区相对标签而退出；尚未读取字段/目标或计算设计。修正为既定工作区逻辑路径，不修改源、公式或资格规则。",
        "old_R102_direct_support_gate_promoted": False,
        "physical_first_vintage_verified": False, "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }
    write_json(PROTOCOL, protocol, exclusive=True)

    data = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet", columns=PRICE_COLUMNS)
    validate_source_clock(data)
    daily = build_field(data)
    pd.testing.assert_frame_equal(
        daily, pd.read_parquet(OLD / "results/日线B02事前字段.parquet"), check_exact=True,
    )
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    indices = states.origin_index.to_numpy(int)
    field = daily.iloc[indices].reset_index(drop=True)
    require(np.array_equal(pd.to_datetime(field.date).to_numpy(), pd.to_datetime(states.origin).to_numpy()),
            "B02原自然状态的日期身份变化。")
    raw = field[FIELDS].to_numpy(float)
    known = np.isfinite(raw).all(axis=1)
    require(np.array_equal(known, field.field_status.eq("B02_CURRENT_SECOND_TEST_FIVE_PRICE_AMOUNT_RECORDS_AVAILABLE")),
            "B02原五列与状态不一致。")
    require(len(data) == 3488 and len(states) == 1507 and known.sum() == 15, "B02原源或成员数变化。")
    records = json.loads((CURRENT / "inputs/within_models.json").read_bytes().decode("utf-8-sig"))["models"]
    months = []
    for record in records:
        if not isinstance(record["model"], dict):
            continue
        mask = states.cycle_id.isin(record["training_cycles"]).to_numpy()
        selected = states.loc[mask]
        require(len(selected) == record["training_rows"] and selected.cycle_id.nunique() == len(record["training_cycles"]),
                "B02原成熟训练成员变化。")
        require((selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "B02成员在原拟合时尚未成熟。")
        ids = selected.cycle_id.to_numpy(int)
        unique, counts = np.unique(ids, return_counts=True)
        count_map = dict(zip(unique, counts))
        weights = np.array([1. / count_map[i] for i in ids], dtype=float)
        x = raw[mask]
        active = np.isfinite(x).all(axis=1)
        phi = np.zeros_like(x)
        if active.any():
            mean = np.average(x[active], axis=0, weights=weights[active])
            sd = np.sqrt(np.average((x[active] - mean) ** 2, axis=0, weights=weights[active]))
            scale = np.where(sd > 1e-12, sd, 1.)
            z = np.clip((x[active] - mean) / scale, -5., 5.)
            phi[active] = z - np.average(z, axis=0, weights=weights[active])
        dx = np.empty_like(phi)
        for cycle_id in unique:
            m = ids == cycle_id
            dx[m] = phi[m] - np.average(phi[m], axis=0, weights=weights[m])
        weighted_design = np.sqrt(weights[:, None]) * dx
        singular = np.linalg.svd(weighted_design, compute_uv=False)
        tolerance = float(max(weighted_design.shape) * np.finfo(np.float64).eps * singular[0])
        rank = int((singular > tolerance).sum())
        months.append({
            "fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_training_rows": len(selected), "known_rows": int(active.sum()),
            "unknown_original_rows_retained": int((~active).sum()),
            "original_training_cycles": len(unique),
            "known_training_cycles": int(selected.loc[active, "cycle_id"].nunique()),
            "weighted_within_design_singular_values": singular.tolist(),
            "standard_svd_tolerance": tolerance, "rank": rank,
            "all_five_columns_identifiable": rank == 5,
        })
    require(len(records) == 142 and len(months) == 115, "B02原月度状态变化。")
    saved = pd.read_parquet(OUT / "results/全部原状态D02固定预测.parquet",
                            columns=["cycle_id", "origin_index", "origin", "status"])
    require(np.array_equal(saved[["cycle_id", "origin_index"]].to_numpy(int),
                           states[["cycle_id", "origin_index"]].to_numpy(int)), "B02原预测身份变化。")
    available = ~saved.status.eq("NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY").to_numpy()
    require(available.sum() == 1010, "B02原可用预测状态变化。")
    periods = []
    for name, start, end in (("2015_2019", "2015-01-05", "2019-12-31"),
                             ("2020_2026", "2020-01-02", "2026-09-30")):
        m = available & pd.to_datetime(states.origin).between(start, end).to_numpy()
        periods.append({"period": name, "original_available_predictions": int(m.sum()),
                        "joint_fields_known": int((m & known).sum()),
                        "exact_core_fallback_if_implemented": int((m & ~known).sum())})
    passed = all(m["all_five_columns_identifiable"] for m in months)
    rank_counts = {str(rank): sum(m["rank"] == rank for m in months) for rank in range(6)}
    result = {
        "at": now(), "technical_decision": "TECH.R138", "protocol_sha256": digest(PROTOCOL),
        "status": "PASS_FIXED_B02_FIVE_COLUMN_DESIGN_ONLY_NO_FINANCIAL_FIT" if passed
                  else "REJECTED_FIXED_B02_OPTIONAL_FIVE_COLUMN_DESIGN_NOT_IDENTIFIABLE_ALL_ORIGINAL_MONTHS",
        "same_original_daily_fields": True, "old_R102_frozen_sources_checked": old_count,
        "parent_R137_frozen_sources_checked": parent_count, "daily_rows": len(daily),
        "known_daily_rows": int(np.isfinite(daily[FIELDS].to_numpy(float)).all(axis=1).sum()),
        "natural_rows": len(states), "joint_known_rows": int(known.sum()),
        "joint_unknown_rows_preserved": int((~known).sum()), "ready_months": len(months),
        "original_unknown_months_preserved": len(records) - len(months),
        "identifiable_months": sum(m["all_five_columns_identifiable"] for m in months),
        "rank_counts": rank_counts, "minimum_known_training_rows": min(m["known_rows"] for m in months),
        "maximum_known_training_rows": max(m["known_rows"] for m in months),
        "design_gate_passed": passed, "monthly_design_support": months, "periods": periods,
        "target_column_read": False, "future_target_values_parsed": False,
        "new_model_fits": 0, "new_return_labels": 0, "new_accounts": 0,
        "new_market_requests": 0, "new_network_requests": 0,
        "old_R102_direct_support_gate_promoted": False,
        "physical_first_vintage_verified": False, "history_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "overfit_removed": False, "goal_achieved": False, "orders_authorized": False,
    }
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "B02资格源在执行中变化。")
    write_json(RESULT, result, exclusive=True)
    pd.DataFrame(months).to_csv(OUT / "next_B02_monthly_design_support.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in result.items() if k != "monthly_design_support"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
