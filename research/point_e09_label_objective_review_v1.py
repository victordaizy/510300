"""只读核对原退出目标、成熟时钟与旧合同，不生成新收益标签或模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_e09_label_objective_review_v1"
CURRENT = "reports/research/510300_point_current_observation_20261001"
NEXT = "reports/research/510300_point_next_information_intake_20261002"
FEATURES = ["log_holding_days", "cycle_return", "cycle_drawdown", "entry_mode", "mom5", "mom20", "sma120", "vol20"]
META = ["cycle_id", "origin_index", "origin", "early_exit_index", "early_exit_date", "exit_index", "mature_date", "reference_quantity"]
OLD = [
    ("510300_short_horizon_cycle_exit_v1", "至多五区间标签；保留原自然周期成熟。", "旧44已失败并停止相邻期限搜索。"),
    ("510300_directional_continuation_v1", "原继续收益严格大于零的方向分类。", "旧79已关闭，不改概率门或正则营救。"),
    ("510300_finite_horizon_exit_v1", "同持仓年龄倒推未来退出价值及自然目标配对。", "旧118已完成；不改年龄、支持、目标或惩罚营救。"),
    ("510300_probability_payoff_exit_v1", "原目标的条件方向概率与正负收益幅度分离。", "旧158已关闭，不作为新目标表达重试。"),
    ("510300_entry_payoff_gate_v1", "完整单次进入收益；新增权益延后确认时延后成熟。", "旧95进入筛选失败；不是当前调整库存账户的标签。"),
]
SOURCES = [
    "research/point_e09_label_objective_review_v1.py",
    "research/learned_cycle_exit_v1.py", "research/point_monthly_model_inputs_v1.py",
    "research/training_reference_observation_v1.py", "research/within_cycle_exit_inputs_v1.py",
    "research/intraday_overnight_increment_v1.py", "research/probability_payoff_exit_inputs_v1.py",
    "research/entry_payoff_gate_inputs_v1.py", "research/finite_horizon_exit_inputs_v1.py",
    "research/directional_continuation_inputs_v1.py", "research/point_core_observation_inputs_v1.py",
    "research/registered_factors.py", "research/build_registered_factor_dataset.py",
    "research/short_horizon_direction_volatility.py", "scripts/run_pattern_daily_state_learning_v3.py",
    "config/510300_learned_cycle_exit_v1.json", "config/510300_within_cycle_exit_v1.json",
    "config/return_tail_hypothesis_registry.yaml", "config/return_tail_gating_manifest.json",
    "docs/RETURN_TAIL_GATING_RESEARCH_SPEC.md",
    f"{NEXT}/E09_label_objective_intake_plan.json", f"{NEXT}/G1_prior_definition_routing_20261002.json",
    f"{CURRENT}/inputs/candidate_features.parquet", f"{CURRENT}/inputs/dividends.csv",
    f"{CURRENT}/inputs/within_models.json", f"{CURRENT}/results/training_reference/samples.parquet",
    f"{CURRENT}/results/training_reference/cycles.parquet", f"{CURRENT}/results/training_reference/decisions.parquet",
    f"{CURRENT}/results/training_reference/unfinished_states.parquet",
    "reports/research/510300_point_exit_sign_calibration_v1/error_components/summary.json",
    "reports/research/510300_point_c02_exit_prediction_v1/prediction_summary.json",
    "reports/research/510300_pattern_daily_state_learning_v3/protocol.json",
    "reports/research/510300_pattern_daily_state_learning_v3/summary.json",
]
for folder, _, _ in OLD:
    SOURCES.extend([f"research/{folder.removeprefix('510300_')}.py", f"config/{folder}.json",
                    f"reports/research/{folder}/result.json", f"reports/research/{folder}/acceptance_outcome.json"])


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def source_checks():
    frozen = read(str((OUT / "freeze.json").relative_to(ROOT)))
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "只读诊断协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def freeze():
    require(not (OUT / "protocol.json").exists(), "E09已经登记，不重复冻结。")
    sources = [{"path": path, "sha256": digest(ROOT / path)} for path in sorted(set(SOURCES))]
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "study": "510300_POINT_E09_LABEL_OBJECTIVE_REVIEW_V1", "frozen_at": now(),
        "purpose": "原目标及旧定义的只读准入复核；元数据不使用target或任何新目标收益。",
        "scope": "510300.SH日线与此前完成周线，多头优先；原策略及所有冻结失败保留。",
        "metadata_columns": META, "forbidden_columns": ["target", "net_profit_cny", "net_return"],
        "original_endpoint": "NEXT_RAW_OPEN_VS_NATURAL_EXIT_RAW_OPEN_BASE_FRICTION_AND_INCREMENTAL_RIGHTS",
        "clock": "原标签整周期自然退出成熟；模型仍用最近20完整周期、至少10周期100行。",
        "quantiles": [0., .25, .5, .75, .95, 1.],
        "metadata_groups": "按原点日期：全体、2015至2019、2020至2026-09-30；仅描述，不是新预测比较。",
        "checks": ["日期和市场索引一致", "T+1及严格不同退出原点", "原状态覆盖和明确排除原因",
                   "新增权益除息不晚于原成熟日期", "全部原月度成员及成熟时钟一致"],
        "old_contract_roles": "只抽取旧合同和保存终态/主指标，原242日账户不混入当前252日比较。",
        "next_target_admission": "只有不同经济问题及完整因果样本成立才另冻结；本诊断不选择新时域。",
        "new_labels": 0, "new_model_fits": 0, "new_strategy_accounts": 0,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    }
    write(OUT / "protocol.json", protocol)
    write(OUT / "freeze.json", {"frozen_at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    print(f"E09只读合同及元数据诊断已冻结，来源{len(sources)}份，不计算新标签或拟合。", flush=True)


def prior_review():
    studies = []
    for folder, expression, disposition in OLD:
        cfg = read(f"config/{folder}.json")
        result = read(f"reports/research/{folder}/result.json")
        outcome = read(f"reports/research/{folder}/acceptance_outcome.json")
        metrics = []
        for column in ("all_metrics", "earlier_diagnostics"):
            for row in result.get(column, []):
                if row.get("model") == cfg["primary"]:
                    metrics.append({"saved_group": column, **{key: row[key] for key in
                        ("model", "cost", "annualized_return", "net_sharpe", "max_drawdown", "trading_days") if key in row}})
        studies.append({"old_study": cfg["study_id"], "round": cfg["round"], "expression": expression,
                        "contract": {key: cfg[key] for key in ("initial_capital", "annual_days", "evaluation_start", "data_cutoff",
                            "earlier_start", "earlier_terminal", "label_max_intervals", "model_loss", "maximum_age") if key in cfg},
                        "old_terminal": outcome["status"], "old_decision": outcome["decision"],
                        "review_disposition": disposition, "saved_primary_metrics_not_recomputed": metrics})
    return {
        "status": "COMPLETED_BOUNDED_PRIOR_LABEL_AND_CLOCK_REVIEW_NO_DISTINCT_TARGET_ADMITTED",
        "studies": studies,
        "original_expression": "[q*(late_fill-early_fill)-late_commission+early_commission+q*新增权益]/(q*early_raw_open)",
        "rights_expression": "record_date>=early_exit_date 且 record_date<natural_exit_date；早已取得的权益两边抵消。",
        "unit_limit": "参考份额的单位继续收益；不等于完整20万元调整库存、现金再部署和核心/辅助混合账户收益。",
        "other_related_contracts": [
            {"name": "通用5/20日Target", "source": "research/registered_factors.py", "meaning": "t+1开盘至t+h收盘；不同于原两个开盘退出，代码存在不是策略通过。"},
            {"name": "20日方向/实现波动", "source": "research/short_horizon_direction_volatility.py", "meaning": "方向来自注册净收益；实现波动不同于方向，不据此提升策略。"},
            {"name": "五日普通日收益均值V3", "source": "reports/research/510300_pattern_daily_state_learning_v3/protocol.json", "meaning": "另一训练池/执行合同；已运行且FROZEN_V3_NUMERICAL_FAILURE，不重试。"},
            {"name": "BAD20/TAIL20与60日", "source": "config/return_tail_hypothesis_registry.yaml", "meaning": "原尾部注册合同，1.5%现金与242日；只核对表达，不认证来源通过或变更原分支状态。"},
        ],
        "P04_routing": "原完整卡未绑定H且NOT_RUN保留；至多五区间及按年龄倒推已有旧表达，不推断所有H都已完整检验。",
        "alternative_maturity": "原未完成周期不能直接从只含完成周期的samples构造；异步短期标签会改变训练单位和用途，本轮未接纳。",
        "original_target_error_established": False, "new_horizon_chosen": False,
        "new_target_definition_frozen": False, "new_candidate_model_registered": False,
        "source_clock_limit": "本次核对缓存/重建时钟，未建立每条历史来源的真实首版或当年部署存档。",
        "new_return_labels": 0, "new_model_fits": 0, "new_strategy_accounts": 0,
    }


def metadata():
    dates = pd.DatetimeIndex(pd.read_parquet(ROOT / CURRENT / "inputs/candidate_features.parquet", columns=["date"]).date)
    samples = pd.read_parquet(ROOT / CURRENT / "results/training_reference/samples.parquet", columns=META)
    cycles = pd.read_parquet(ROOT / CURRENT / "results/training_reference/cycles.parquet",
                            columns=["cycle_id", "entry_index", "entry_date", "entry_quantity", "exit_date", "exit_reasons"])
    require(dates.is_unique and dates.is_monotonic_increasing and cycles.cycle_id.is_unique, "原日历或周期身份不唯一。")
    require(not cycles.exit_reasons.fillna("").str.contains("研究终点", regex=False).any(), "原参考包含人工终点退出。")
    cycles["natural_exit_index"] = dates.get_indexer(pd.to_datetime(cycles.exit_date))
    indexed = cycles.set_index("cycle_id")
    out = samples.merge(cycles[["cycle_id", "entry_index", "entry_date", "entry_quantity", "natural_exit_index"]],
                        on="cycle_id", how="left", validate="many_to_one", sort=False)
    require(len(out) == len(samples) and out.entry_index.notna().all(), "原样本周期覆盖改变。")
    for index, date in (("origin_index", "origin"), ("early_exit_index", "early_exit_date"), ("exit_index", "mature_date"), ("entry_index", "entry_date")):
        require(np.array_equal(dates[out[index].to_numpy(int)].to_numpy(), pd.to_datetime(out[date]).to_numpy()), "日期与索引不一致：" + date)
    require(out.early_exit_index.eq(out.origin_index+1).all() and out.early_exit_index.lt(out.exit_index).all(), "原退出时钟改变。")
    require(out.origin_index.ge(out.entry_index).all() and out.early_exit_index.ge(out.entry_index+1).all(), "原参考不满足T+1年龄边界。")
    require(out.exit_index.eq(out.natural_exit_index).all() and out.reference_quantity.eq(out.entry_quantity).all(), "原自然终点或份额不同。")
    out["holding_age_sessions"] = out.origin_index-out.entry_index+1
    out["remaining_open_intervals"] = out.exit_index-out.early_exit_index
    out["label_wait_sessions_after_origin"] = out.exit_index-out.origin_index
    events = pd.read_csv(ROOT / CURRENT / "inputs/dividends.csv", usecols=["symbol", "record_date", "ex_date"])
    require(events.symbol.eq("510300.SH").all(), "分红标的不同。")
    for key in ("record_date", "ex_date"):
        events[key] = pd.to_datetime(events[key])
    out["incremental_rights_ex_date_after_maturity"] = [int((events.record_date.ge(row.early_exit_date) &
        events.record_date.lt(row.mature_date) & events.ex_date.gt(row.mature_date)).sum()) for row in out.itertuples()]
    require(out.incremental_rights_ex_date_after_maturity.eq(0).all(), "原成熟日尚不足以确认新增权益，不重写旧结果。")
    decisions = pd.read_parquet(ROOT / CURRENT / "results/training_reference/decisions.parquet",
                               columns=["origin", "origin_index", "learning_cycle_id", "requested_quantity", *FEATURES])
    held = decisions[decisions.learning_cycle_id.notna() & decisions.requested_quantity.eq(0)].copy()
    keys = set(zip(out.cycle_id.astype(int), out.origin_index.astype(int)))
    support = []
    for row in held.itertuples():
        cycle_id, t = int(row.learning_cycle_id), int(row.origin_index)
        cycle = indexed.loc[cycle_id]
        present = (cycle_id, t) in keys
        if pd.isna(cycle.exit_date):
            status = "UNMATURED_NATURAL_CYCLE"
        elif not np.isfinite([getattr(row, key) for key in FEATURES]).all():
            status = "NO_VIEW_INCOMPLETE_ORIGINAL_EIGHT"
        elif t+1 >= cycle.natural_exit_index:
            status = "NO_SEPARATE_CONTINUATION_INTERVAL"
        else:
            status = "ORIGINAL_MATURE_SAMPLE"
        require(present == (status == "ORIGINAL_MATURE_SAMPLE"), "原状态排除原因未完整解释。")
        support.append({"cycle_id": cycle_id, "origin_index": t, "origin": row.origin,
                        "metadata_status": status, "original_sample_present": present})
    support = pd.DataFrame(support)
    require(int(support.original_sample_present.sum()) == len(out), "原样本未覆盖全部支持状态。")
    cfg = read("config/510300_learned_cycle_exit_v1.json")
    require((cfg["recent_cycles"], cfg["minimum_cycles"], cfg["minimum_rows"]) == (20, 10, 100), "原成熟制度不同。")
    monthly = []
    for model in read(f"{CURRENT}/inputs/within_models.json")["models"]:
        fit = int(model["fit_index"])
        mature = out[out.exit_index.le(fit)]
        ids = mature[["cycle_id", "exit_index"]].drop_duplicates().sort_values(["exit_index", "cycle_id"]).tail(20).cycle_id.to_list()
        selected = mature[mature.cycle_id.isin(ids)]
        eligible = len(ids) >= 10 and len(selected) >= 100
        require(ids == model["training_cycles"] and len(selected) == model["training_rows"], "原月度训练成员不同。")
        require(eligible == (model["status"] == "FIT_COMPLETE"), "原月度可用状态改变。")
        monthly.append({"fit_index": fit, "fit_origin": model["fit_origin"], "original_status": model["status"],
                        "original_training_cycles": len(ids), "original_training_rows": len(selected),
                        "latest_mature_exit_index": int(selected.exit_index.max()) if len(selected) else None,
                        "future_training_rows": int(selected.exit_index.gt(fit).sum())})
    return out, support, pd.DataFrame(monthly), cycles


def describe(frame, cycles):
    rows = []
    for name, selected in (("ALL", frame), ("ORIGIN_2015_2019", frame[frame.origin.ge("2015-01-01") & frame.origin.lt("2020-01-01")]),
                           ("ORIGIN_2020_20260930", frame[frame.origin.ge("2020-01-01")])):
        row = {"metadata_group": name, "rows": len(selected), "represented_cycles": selected.cycle_id.nunique()}
        row.update({f"remaining_open_intervals_q{int(q*100):02d}": float(selected.remaining_open_intervals.quantile(q)) for q in (0., .25, .5, .75, .95, 1.)})
        rows.append(row)
    return {"metadata_groups": rows, "reference_cycles": len(cycles), "complete_reference_cycles": int(cycles.exit_date.notna().sum()),
            "cycles_without_original_continuation_rows": len(set(cycles.cycle_id)-set(frame.cycle_id)),
            "original_sample_rows": len(frame), "original_sample_cycles": frame.cycle_id.nunique(),
            "incremental_rights_clock_violations": int(frame.incremental_rights_ex_date_after_maturity.sum()),
            "row_count_is_independent_units": False}


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "E09已开始，不重复运行。")
    count = source_checks()
    write(OUT / "RUN_STARTED.json", {"at": now(), "frozen_sources": count, "new_labels": 0, "new_fits": 0})
    frame, support, monthly, cycles = metadata()
    (OUT / "results").mkdir(exist_ok=True)
    for name, table in (("原标签时距与成熟元数据", frame), ("原持仓状态排除原因", support), ("原月度成熟成员时钟", monthly)):
        table.to_parquet(OUT / "results" / (name+".parquet"), index=False)
        table.to_csv(OUT / "results" / (name+".csv"), index=False, encoding="utf-8-sig")
    review = prior_review()
    write(OUT / "prior_definition_review.json", review)
    summary = {"study": "510300_POINT_E09_LABEL_OBJECTIVE_REVIEW_V1", "at": now(),
               "status": "COMPLETED_ORIGINAL_LABEL_CLOCK_AND_PRIOR_REVIEW_NO_DISTINCT_NEW_TARGET_ADMITTED",
               **describe(frame, cycles), "held_original_states": len(support),
               "state_exclusion_counts": support.metadata_status.value_counts().to_dict(),
               "monthly_records": len(monthly), "ready_months": int(monthly.original_status.eq("FIT_COMPLETE").sum()),
               "future_training_rows": int(monthly.future_training_rows.sum()),
               "frozen_sources": source_checks(), "original_target_error_established": False,
               "new_horizon_chosen": False, "new_return_labels": 0, "new_model_fits": 0, "new_strategy_accounts": 0,
               "account_return_sharpe": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def verify():
    count = source_checks()
    frame, support, monthly, cycles = metadata()
    for name, table in (("原标签时距与成熟元数据", frame), ("原持仓状态排除原因", support), ("原月度成熟成员时钟", monthly)):
        pd.testing.assert_frame_equal(pd.read_parquet(OUT / "results" / (name+".parquet")), table, check_exact=True)
    require(read(str((OUT / "prior_definition_review.json").relative_to(ROOT))) == prior_review(), "旧合同抽取不一致。")
    saved = read(str((OUT / "summary.json").relative_to(ROOT)))
    for key, value in describe(frame, cycles).items():
        require(saved[key] == value, "保存的时距描述不一致：" + key)
    receipt = {"at": now(), "status": "PASS_SAVED_E09_ORIGINAL_LABEL_METADATA_AND_PRIOR_CONTRACT_RECOMPUTATION",
               "frozen_sources_unchanged": count, "sample_metadata_rows": len(frame), "held_state_rows": len(support),
               "monthly_clock_rows": len(monthly), "tables_exactly_equal": True, "prior_contracts_exactly_equal": True,
               "new_return_labels": 0, "new_model_fits": 0, "new_strategy_accounts": 0,
               "scope": "原标签元数据和旧合同的重算，不是目标收益重算、新目标接纳或独立策略验证。"}
    write(OUT / "saved_output_recomputation_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="E09只读标签经济含义、原时钟与旧合同复核。")
    parser.add_argument("action", choices=["freeze", "run", "verify"])
    arguments = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[arguments.action]()


if __name__ == "__main__":
    main()
