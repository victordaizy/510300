"""J01固定中国国债收益率来源、历史时钟及原成员支持检查，不拟合或回测。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_j01_local_yield_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
LONG = ROOT / "data/raw/regime_transition_router_v1/cgb_curve/cgb_1y_10y_daily_2012_2026.parquet"
BASE = ROOT / "data/raw/macro/china_government_bond_yields_daily.parquet"
LEDGER = ROOT / "data/curated/510300_stress_transmission_hazard_v2_mft_feature_execution_v1/china_10y_yield_release_ledger.parquet"
FIELDS = ["J01_yield20_change_bp", "J01_stock_yield20_interaction"]
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
TABLES = ("日线J01固定利率来源支持", "原自然成员J01来源支持", "原月度成熟成员J01来源支持")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def dates(values, label):
    result = pd.DatetimeIndex(values)
    require(result.tz is None, label+"必须为无时区的观察日期。")
    result = result.astype("datetime64[ns]")
    require(not result.hasnans and result.is_unique and result.is_monotonic_increasing,
            label+"必须唯一递增且不缺失。")
    require(result.equals(result.normalize()), label+"含非零时刻，不得静默截断。")
    return result


def validate_rates(frame):
    require({"date", "cgb_10y", "source", "retrieved_at"}.issubset(frame.columns), "国债原来源列不足。")
    index = dates(frame.date, "国债观察日期")
    require(frame.source.astype(str).str.contains("chinabond", case=False).all(), "国债来源身份变化。")
    received = pd.to_datetime(frame.retrieved_at, utc=True, errors="raise")
    require(received.notna().all(), "国债实际回取时间缺失。")
    return index


def build_field(current, rates):
    """T收盘只看T-1及T-21实际股票日期，保留历史回取与算法可用的区别。"""
    require({"date", "symbol", "mom20"}.issubset(current.columns), "当前股票来源列不足。")
    require(current.symbol.eq("510300.SH").all(), "股票标的必须为510300.SH。")
    stock_dates = dates(current.date, "股票日期")
    rate_dates = validate_rates(rates)
    position = {day: i for i, day in enumerate(rate_dates)}
    field = pd.DataFrame({"date": stock_dates, "decision_deadline": stock_dates.tz_localize("Asia/Shanghai")
                          + pd.Timedelta(hours=15, minutes=5)})
    for name in ["latest_observation", "base_observation"]:
        field[name] = pd.NaT
    for name in ["latest_algorithm_available_at", "base_algorithm_available_at", "latest_retrieved_at", "base_retrieved_at"]:
        field[name] = pd.Series(pd.NaT, index=field.index, dtype="datetime64[ns, UTC]")
    field["latest_source_row"] = -1
    field["base_source_row"] = -1
    field[FIELDS] = np.nan
    field["field_status"] = "NO_VIEW_PREVIOUS_20_STOCK_INTERVALS_NOT_READY"
    field["first_publication_evidence_proved"] = False
    field["saved_mom20_reused"] = current.mom20.to_numpy(float)
    received = pd.to_datetime(rates.retrieved_at, utc=True)
    for t in range(21, len(current)):
        recent_day, base_day = stock_dates[t-1], stock_dates[t-21]
        field.loc[t, ["latest_observation", "base_observation"]] = [recent_day, base_day]
        recent_row, base_row = position.get(recent_day, -1), position.get(base_day, -1)
        field.loc[t, ["latest_source_row", "base_source_row"]] = [recent_row, base_row]
        if recent_row < 0 or base_row < 0:
            field.loc[t, "field_status"] = "NO_VIEW_EXACT_REQUIRED_YIELD_DATE_MISSING"
            continue
        latest_known = (recent_day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_convert("UTC")
        base_known = (base_day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_convert("UTC")
        field.loc[t, ["latest_algorithm_available_at", "base_algorithm_available_at"]] = [latest_known, base_known]
        field.loc[t, ["latest_retrieved_at", "base_retrieved_at"]] = [received.iloc[recent_row], received.iloc[base_row]]
        require(latest_known < field.decision_deadline.iloc[t] and base_known < field.decision_deadline.iloc[t],
                "国债算法可用时间晚于当前判断。")
        recent_yield, base_yield = float(rates.cgb_10y.iloc[recent_row]), float(rates.cgb_10y.iloc[base_row])
        price_state = float(current.mom20.iloc[t])
        if not np.isfinite([recent_yield, base_yield, price_state]).all():
            field.loc[t, "field_status"] = "NO_VIEW_YIELD_OR_ORIGINAL_MOM20_NOT_FINITE"
            continue
        change_bp = (recent_yield-base_yield)*100.0
        field.loc[t, FIELDS] = [change_bp, change_bp*price_state]
        field.loc[t, "field_status"] = "ALGORITHMIC_HISTORY_CLOCK_ONLY_FIRST_VINTAGE_NOT_PROVED"
    return field


def member_support(samples, records, field):
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "原自然成员索引越界。")
    require(np.array_equal(pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]"),
                           field.date.iloc[indexes].to_numpy(dtype="datetime64[ns]")), "原自然成员日期变化。")
    values = field[FIELDS].to_numpy(float)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"]
                and len(selected) == record["training_rows"] and selected.cycle_id.nunique() == len(ids),
                "原成熟训练成员数量变化。")
        require((selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "原训练周期尚未成熟。")
        selected_indexes = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[selected_indexes]).all(axis=1)
        missing = field.field_status.iloc[selected_indexes[~finite]].value_counts().to_dict()
        rows.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "algorithmic_source_available_rows": int(finite.sum()), "algorithmic_source_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()), "first_publication_evidence_proved": False,
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in missing.items()}, ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)


def crosscheck_sources(primary, base, ledger):
    primary_dates, base_dates = validate_rates(primary), validate_rates(base)
    ledger_dates = dates(ledger.observation_date, "旧衍生账簿观察日")
    require(ledger.availability_rule.eq("NEXT_PIT_MARKET_SESSION_OPEN_AFTER_OBSERVATION_DATE").all(), "旧账簿时钟规则改变。")
    views = [("BASE", base_dates, base.cgb_10y.to_numpy(float)),
             ("LEGACY_DERIVED_LEDGER", ledger_dates, ledger.china_10y_yield.to_numpy(float))]
    rows = []
    primary_values = pd.Series(primary.cgb_10y.to_numpy(float), index=primary_dates)
    for name, index, values in views:
        other = pd.Series(values, index=index)
        common = primary_dates.intersection(index)
        equal = np.equal(primary_values.loc[common].to_numpy(), other.loc[common].to_numpy())
        rows.append({"source": name, "source_rows": len(index), "common_dates": len(common),
                     "exact_same_10y_values": int(equal.sum()), "different_10y_values": int((~equal).sum()),
                     "equal_values_do_not_prove_first_publication": True})
    return rows


def freeze():
    prior, tests = read_json(OUT/"prior_and_source_review.json"), read_json(OUT/"tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 7, "七项时钟与成员必要测试未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT/"tests/test_point_j01_local_yield_intake_v1.py"), "测试后来源代码变化。")
    require(prior["fixed_source_and_fields_bound_before_current_quantities"], "当前数量前必须绑定唯一来源与字段。")
    protocol = {"study": "510300_POINT_J01_LOCAL_YIELD_INTAKE_V1", "at": now(),
        "purpose": "来源及原成员准入，不拟合/回测；算法历史数值不能冒充真实首版。",
        "binding": prior["binding"], "support_gate": prior["support_gate"], "no_rescue": prior["no_rescue"],
        "fixed_source_operationalizations": 1, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "goal_achieved": False}
    paths = {Path(__file__), ROOT/"tests/test_point_j01_local_yield_intake_v1.py", OUT/"tests_receipt.json",
             OUT/"prior_and_source_review.json", DATA, SAMPLES, MODELS, LONG, BASE, LEDGER}
    for item in prior["direct_sources"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "事前来源变化："+item["path"])
        paths.add(ROOT/item["path"])
    write_json(OUT/"protocol.json", protocol)
    write_json(OUT/"freeze.json", {"at": now(), "protocol_sha256": digest(OUT/"protocol.json"),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sorted(paths)]})
    return {"状态": "J01唯一来源及事前历史字段已冻结", "来源数": len(paths), "新增拟合": 0}


def check():
    frozen = read_json(OUT/"freeze.json")
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "来源协议变化。")
    for item in frozen["sources"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "冻结来源变化："+item["path"])
    return len(frozen["sources"])


def derive():
    current = pd.read_parquet(DATA, columns=["date", "symbol", "mom20"])
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(current) == 3488 and len(samples) == 1507 and len(records) == 142, "原快照数量变化。")
    primary, base, ledger = (pd.read_parquet(p) for p in [LONG, BASE, LEDGER])
    comparisons = crosscheck_sources(primary, base, ledger)
    field = build_field(current, primary)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "原可用成熟月份数量变化。")
    natural = field.iloc[samples.origin_index.to_numpy(int)].reset_index(drop=True)
    natural.insert(1, "cycle_id", samples.cycle_id.to_numpy(int))
    natural.insert(2, "origin_index", samples.origin_index.to_numpy(int))
    periods = []
    for period, start, end in [("2015_2019", "2015-01-05", "2019-12-31"), ("2020_2026", "2020-01-02", "2026-09-30")]:
        part = natural.loc[natural.date.between(start, end)]
        months = ready.loc[pd.to_datetime(ready.fit_origin).between(start, end)]
        periods.append({"period": period, "original_natural_rows": len(part),
            "algorithmic_available_natural_rows": int(np.isfinite(part[FIELDS].to_numpy(float)).all(axis=1).sum()),
            "original_mature_months": len(months), "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
            "missing_original_training_rows_range": [int(months.algorithmic_source_missing_rows.min()), int(months.algorithmic_source_missing_rows.max())]})
    finite = np.isfinite(natural[FIELDS].to_numpy(float)).all(axis=1)
    facts = {"primary_source_rows": len(primary), "primary_source_first_date": pd.Timestamp(primary.date.iloc[0]).isoformat(),
        "primary_source_last_date": pd.Timestamp(primary.date.iloc[-1]).isoformat(), "source_value_comparisons": comparisons,
        "daily_rows": len(field), "finite_algorithmic_daily_rows": int(np.isfinite(field[FIELDS].to_numpy(float)).all(axis=1).sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "original_natural_rows": len(natural), "algorithmic_supported_natural_rows": int(finite.sum()),
        "natural_status_counts": {str(k): int(v) for k, v in natural.field_status.value_counts().items()},
        "original_monthly_records": len(support), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(support)-len(ready), "periods": periods,
        "algorithmic_source_member_support_passed": bool(finite.all() and ready.all_original_members_supported.all()),
        "historical_first_publication_evidence": "NOT_ESTABLISHED_HISTORICALLY_RETRIEVED_CURVE_AND_DERIVED_CLOCK",
        "first_publication_supported_natural_rows": 0, "source_gate_passed": False}
    return (field, natural, support), facts


def run():
    count = check()
    write_json(OUT/"ADMISSION_STARTED.json", {"at": now(), "new_model_fits": 0})
    tables, facts = derive()
    results = OUT/"results"
    results.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(results/(name+".parquet"), index=False)
        table.to_csv(results/(name+".csv"), index=False, encoding="utf-8-sig")
    summary = {"study": "510300_POINT_J01_LOCAL_YIELD_INTAKE_V1", "at": now(),
        "status": "NOT_ADMITTED_FIXED_J01_LOCAL_YIELD_SOURCE_PUBLICATION_OR_MEMBER_GATE_FAILED",
        "frozen_sources_unchanged": count, **facts, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0, "new_market_collection": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "仅拒绝这一固定本地国债源的当前模型准入，不证明所有股债机制失效，不复活旧联合预测失败。"}
    write_json(OUT/"summary.json", summary)
    require(check() == count, "运行后冻结来源变化。")
    return {"状态": summary["status"], "算法覆盖原成员": facts["algorithmic_supported_natural_rows"],
            "算法完整成熟月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        pd.testing.assert_frame_equal(pd.read_parquet(OUT/"results"/(name+".parquet")), rebuilt, check_exact=True)
    summary = read_json(OUT/"summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "保存的来源支持结果不同。")
    receipt = {"at": now(), "status": "PASS_SAVED_J01_SOURCE_CLOCK_AND_ORIGINAL_MEMBER_RECOMPUTATION",
        "frozen_sources_unchanged": count, "tables_exactly_equal": True, "daily_rows": len(tables[0]),
        "natural_member_rows": len(tables[1]), "monthly_member_rows": len(tables[2]),
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "三表与来源日期/数值身份复算，非首版/预测/账户验证。"}
    write_json(OUT/"saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="J01本地国债固定来源与原成员准入，不拟合或回测")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
