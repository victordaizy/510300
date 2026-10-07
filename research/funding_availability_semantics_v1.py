"""资金源时钟、旧分位数支持与原142训练池的隔离语义核对。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research import macro_technical_first_passage_inputs_v1 as model

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_funding_availability_semantics_v1"
JOINT = ROOT / "reports/research/510300_macro_technical_first_passage_v1_clock_adapter"
BASE = ROOT / "reports/research/510300_point_first_passage_study_v1"
FUND = ROOT / "reports/research/510300_factor96_funding_relief_v1/daily_features_lag1.parquet"
ORDERS = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/pmi_new_orders.parquet"
MARGIN = ROOT / "reports/research/510300_multidim_financing_composition_v1/data_correction/margin_corrected_20240808.parquet"
RAW = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet"
CARD = ROOT / "docs/510300_FUNDING_AVAILABILITY_SEMANTICS_V1.md"
SOURCES = [CARD, Path(__file__), Path(model.__file__), ROOT / "research/factor96_funding_relief_v1.py",
           BASE / "results/原点全部技术特征.parquet", BASE / "results/原点首次边界参考结果.parquet",
           FUND, ORDERS, MARGIN, RAW, JOINT / "saved_models.json", JOINT / "summary.json", JOINT / "protocol.json",
           JOINT / "results/全部3488当时已知技术与宏观_未知保留.parquet",
           JOINT / "results/四原上涨反弹案例_全部同钟宏观与技术.parquet",
           ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/上涨段全集.parquet"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(name, frame):
    frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)


def source_mask(funding):
    f = funding.copy()
    dates = pd.to_datetime(f.date).astype("datetime64[ns]")
    stat = pd.to_datetime(f.fund_stat_date).astype("datetime64[ns]")
    decision = dates.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    source_clock = pd.to_datetime(f.available_at, utc=True)
    policy_clock = pd.to_datetime(f.policy_known_at, utc=True)
    age = (dates - stat).dt.days
    finite = np.isfinite(f[["dr007", "rate", "gap_pp"]].to_numpy(float)).all(axis=1)
    return (stat.lt(dates) & age.between(1, 10) & source_clock.le(decision.dt.tz_convert("UTC"))
            & policy_clock.le(decision.dt.tz_convert("UTC")) & finite)


def freeze():
    if OUT.exists():
        raise FileExistsError("已有语义核对登记，不覆盖。")
    for path in SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)
    (OUT / "results").mkdir(parents=True)
    (OUT / "code").mkdir()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "protocol.json", {"at": now(), "study": "510300_FUNDING_AVAILABILITY_SEMANTICS_V1",
        "decision": "TECH.R195", "scope": "SOURCE_AVAILABILITY_VS_OLD_STATISTICAL_SUPPORT_NOT_A_NEW_FINANCIAL_CANDIDATE",
        "identities": {relative(p): digest(p) for p in SOURCES}, "discovery_before_registration": True,
        "fixed_funding_calendar_end": "2025-12-31", "change": "解释副本fund_known仅按原源时钟/源龄/有限原值判断",
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_downloads": 0,
        "financial_admission": "近期训练合同与预测输入全相同则本改正不能提高原近期金融结果，不重跑账户。",
        "original_protocol_execution_status": "PRESERVED_CORRECT_EXECUTION_OF_ITS_FROZEN_FUND_KNOWN_RULE",
        "historical_first_vintage": "NOT_CERTIFIED", "independent_validation": False, "goal_achieved": False})
    print("TECH.R195源语义核对已登记；不登记金融候选。")


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise FileExistsError("本轮已经开始，不重复。")
    protocol = load(OUT / "protocol.json")
    for path, expected in protocol["identities"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("登记来源改变：" + path)
    save(OUT / "RUN_STARTED.json", {"at": now(), "new_fits": 0, "new_accounts": 0})
    data = pd.read_parquet(BASE / "results/原点全部技术特征.parquet")
    funding = pd.read_parquet(FUND)
    orders, margin = pd.read_parquet(ORDERS), pd.read_parquet(MARGIN)
    old = model.views(data, orders, funding, margin)
    saved = pd.read_parquet(JOINT / "results/全部3488当时已知技术与宏观_未知保留.parquet")
    pd.testing.assert_frame_equal(old, saved, check_exact=True)
    source = source_mask(funding)
    revised_funding = funding.copy()
    revised_funding["fund_known"] = source
    new = model.views(data, orders, revised_funding, margin)
    assert len(old) == len(new) == 3488 and old.date.equals(new.date)
    assert not (old.joint_features_known & ~new.joint_features_known).any()
    source_rows = funding.copy()
    source_rows["source_known_under_original_lag_contract"] = source
    source_rows["old_quantile_signal_supported"] = funding.fund_known
    source_rows["available_source_without_old_quantile_support"] = source & ~funding.fund_known
    export("全部3307资金缓存_源可用与旧分位支持分开", source_rows)
    known = new[["date", "dr007", "rate", "funding_gap_pp", "funding_gap_change5", "orders_known", "margin_known"]].copy()
    known["old_funding_known_with_quantile_gate"] = old.funding_known
    known["source_funding_known_without_quantile_gate"] = new.funding_known
    known["old_joint_features_known"] = old.joint_features_known
    known["source_joint_features_known"] = new.joint_features_known
    known["additional_joint_origin"] = new.joint_features_known & ~old.joint_features_known
    export("全部3488原点_源与统计训练支持分开", known)
    outcomes = pd.read_parquet(BASE / "results/原点首次边界参考结果.parquet")
    records = load(JOINT / "saved_models.json")
    comparisons = []
    for record in records:
        index = int(record["fit_index"])
        original_pool = model.common_pool(old, outcomes, index)
        revised_pool = model.common_pool(new, outcomes, index)
        assert original_pool.origin_index.astype(int).tolist() == record["training_origins"]
        pool_same = original_pool.equals(revised_pool)
        rows = original_pool.origin_index.to_numpy(int)
        revised_rows = revised_pool.origin_index.to_numpy(int)
        feature_same = pool_same and all(np.array_equal(old.iloc[rows][columns].to_numpy(float),
                                                       new.iloc[revised_rows][columns].to_numpy(float))
                                        for columns in model.FEATURES.values())
        counts = revised_pool.event_class.value_counts()
        support = len(revised_pool) >= model.original.MINIMUM_ROWS and all(counts.get(name, 0) >= 10 for name in model.original.CLASSES)
        comparisons.append({"fit_index": index, "fit_date": old.date.iloc[index],
                            "old_training_rows": len(original_pool), "source_training_rows": len(revised_pool),
                            "additional_training_rows": len(revised_pool) - len(original_pool),
                            "pool_members_labels_weights_exactly_same": pool_same,
                            "both_models_training_features_exactly_same": feature_same,
                            "source_training_support": bool(support), "old_status": record["status"],
                            "source_available_does_not_mean_model_supported": not support})
    audit = pd.DataFrame(comparisons)
    assert len(audit) == 142
    export("全部142月训练合同精确比较_未拟合", audit)
    fields = ["date", "joint_features_known", *model.TECH, *model.MACRO]
    recent = old.date.ge(pd.Timestamp("2020-01-01"))
    recent_origins_same = old.loc[recent, fields].equals(new.loc[recent, fields])
    recent_months = audit.loc[audit.fit_date.ge(pd.Timestamp("2020-01-01"))]
    recent_training_same = recent_months.both_models_training_features_exactly_same.all()
    if recent_origins_same:
        pd.testing.assert_frame_equal(old.loc[recent, fields], new.loc[recent, fields], check_exact=True)
    export("近期原输入完整日历_未修改", old.loc[recent, fields])
    export("近期训练合同精确相同核对", recent_months)
    case_ids = pd.read_parquet(JOINT / "results/四原上涨反弹案例_全部同钟宏观与技术.parquet")[["date", "original_episode_id"]]
    case_ids["date"] = pd.to_datetime(case_ids.date).astype("datetime64[ns]")
    cases = case_ids.merge(known, on="date", how="left", validate="many_to_one")
    assert len(cases) == 240
    export("四原案例240行_不把源可用称作入场支持", cases)
    waves = pd.read_parquet(ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/上涨段全集.parquet")
    waves["confirm_up_date"] = pd.to_datetime(waves.confirm_up_date).astype("datetime64[ns]")
    waves = waves.merge(known, left_on="confirm_up_date", right_on="date", how="left", validate="many_to_one")
    assert len(waves) == 61 and int(waves.admitted.sum()) == 49
    export("原61分段49上涨确认_两种可用定义", waves)
    raw = pd.read_parquet(RAW)
    changed = audit.loc[~audit.both_models_training_features_exactly_same]
    extra = known.loc[known.additional_joint_origin]
    summary = {"at": now(), "study": protocol["study"], "registration_decision": "TECH.R195", "decision": "TECH.R196",
        "status": "COMPLETED_FUNDING_SOURCE_SEMANTICS_DIAGNOSTIC_NO_RECENT_MODEL_INPUT_CHANGE",
        "source_available_without_quantile_signal_support": int((source & ~funding.fund_known).sum()),
        "old_joint_origins": int(old.joint_features_known.sum()), "source_joint_origins": int(new.joint_features_known.sum()),
        "additional_joint_origins": len(extra), "additional_first": extra.date.min().isoformat() if len(extra) else None,
        "additional_last": extra.date.max().isoformat() if len(extra) else None,
        "monthly_records": len(audit), "changed_training_contracts": len(changed),
        "last_changed_month": changed.fit_date.max().isoformat() if len(changed) else None,
        "recent_months": len(recent_months), "recent_training_contracts_exactly_same": bool(recent_training_same),
        "recent_forecast_input_origins_exactly_same": bool(recent_origins_same),
        "original_model_fit_and_execution_contract_unchanged": True,
        "same_recent_account_result_implied": bool(recent_training_same and recent_origins_same),
        "financial_admission": "NOT_ADMITTED_THIS_MASK_CORRECTION_ALONE_CANNOT_CHANGE_REQUIRED_RECENT_FAILED_ACCOUNT" if recent_training_same and recent_origins_same else "REQUIRES_SEPARATE_FINANCIAL_PROTOCOL_NOT_REGISTERED",
        "raw_dr_first": pd.Timestamp(raw.date.min()).isoformat(), "raw_dr_last": pd.Timestamp(raw.date.max()).isoformat(),
        "funding_cache_last": pd.Timestamp(funding.date.max()).isoformat(),
        "source_semantics_misclassification_identified": True, "original_frozen_protocol_violation": False,
        "new_model_fits": 0, "new_accounts": 0, "new_stock_labels": 0, "new_network_requests": 0,
        "net_sharpe": None, "net_cagr": None, "new_financial_metrics": "NOT_COMPUTED",
        "independent_validation": False, "overfitting_removed": False, "goal_achieved": False,
        "next_scope": "先核对2026各原统计源与旧派生缓存的真实覆盖，区分未纳入缓存与源缺失；不自动延用旧值、改窗口或新跑旧失败。"}
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="资金来源与旧分位数支持的有限语义核对")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
