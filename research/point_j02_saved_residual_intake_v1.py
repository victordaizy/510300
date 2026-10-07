"""J02已有人民币残差的保存时钟及原成员准入，不重拟合外汇模型或账户。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_j02_saved_residual_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OLD = ROOT / "reports/research/510300_rmb_residual_state_v1"
FIELDS = ["J02_saved_fx_change5_residual"]
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
TABLES = ("日线J02保存残差来源支持", "原自然成员J02来源支持", "原月度成熟成员J02来源支持")


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
    index = pd.DatetimeIndex(values)
    require(index.tz is None, label+"必须是不含时区的观察日期。")
    index = index.astype("datetime64[ns]")
    require(not index.hasnans and index.is_unique and index.is_monotonic_increasing
            and index.equals(index.normalize()), label+"必须为唯一递增的完整日期。")
    return index


def validate_common(source):
    index = dates(source.date, "旧共同来源")
    availability = pd.DatetimeIndex(pd.to_datetime(source.available_at, utc=True)).astype("datetime64[ns, UTC]")
    expected = (index.tz_localize("Asia/Shanghai")+pd.Timedelta(days=2, hours=23, minutes=59)).tz_convert("UTC")
    require(availability.equals(expected), "旧共同来源第二自然日23:59规则变化。")
    require((pd.to_datetime(source.bar_end_utc, utc=True) <= pd.Series(availability)).all(), "离岸日柱结束时间晚于可用代理。")
    require(np.isfinite(source[["close", "dxy", "us_10y", "cn_10y"]].to_numpy(float)).all()
            and (source[["close", "dxy"]].to_numpy(float) > 0).all(), "旧真实报价或收益率来源无效。")
    gaps = source.window_crosses_missing_source_year.eq(True).to_numpy()
    require(source.loc[gaps, ["fx_change5", "usd_change5", "spread_change5"]].isna().all().all(),
            "跨缺失年份变化窗口不能当作连续五日。")
    return index, availability


def validate_saved_object(saved, market):
    require(market.symbol.eq("510300.SH").all(), "旧来源股票标的变化。")
    projected = market.loc[pd.to_datetime(market.date).between("2019-01-01", "2025-12-31")]
    require(np.array_equal(dates(projected.date, "旧市场投影").to_numpy(), dates(saved.date, "保存残差").to_numpy()),
            "旧保存模型与固定2019—2025市场投影不一致。")


def check_saved_clocks(saved, source, training):
    """核保存名单、当前来源及15:05可用代理；不重算OLS或预测公式。"""
    source_dates, availability = validate_common(source)
    saved_dates = dates(saved.date, "保存残差")
    known = saved.source_admitted.eq(True).to_numpy()
    require(np.array_equal(known, saved.model_status.eq("AVAILABLE").to_numpy()), "保存可用标识不一致。")
    valid = np.isfinite(source[["fx_change5", "usd_change5", "spread_change5"]].to_numpy(float)).all(axis=1)
    lookup = {pd.Timestamp(row["decision_date"]): row for row in training}
    require(len(lookup) == len(training) and set(lookup) == set(saved_dates[known]), "保存训练名单重复或与可用日不一致。")
    checked = 0
    for t in np.flatnonzero(known):
        day = saved_dates[t]
        row = saved.iloc[t]
        deadline = (day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15, minutes=5)).tz_convert("UTC")
        original_deadline = day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)
        require(pd.Timestamp(row.decision_at) == original_deadline, "旧16:00决策元数据变化。")
        latest = int(availability.searchsorted(deadline, side="right")-1)
        require(latest >= 0 and latest == int(row.source_idx) and valid[latest], "15:05最新已可用源与旧保存源不同。")
        require(pd.Timestamp(row.source_date) == source_dates[latest]
                and pd.Timestamp(row.source_available_at).tz_convert("UTC") == availability[latest], "保存来源日期/可用时间不同。")
        age = (day-source_dates[latest]).days
        require(age == int(row.source_age_days) and age <= 7, "保存来源超过原七自然日陈旧界。")
        record = lookup[day]
        require(int(record["current_source_idx"]) == latest, "训练记录当前来源不同。")
        ids = np.asarray(record["training_indices"], dtype=np.int64)
        lower = day-pd.DateOffset(years=2)
        expected = np.flatnonzero(valid & (np.arange(len(source)) < latest) & (source_dates >= lower))
        require(len(ids) >= 252 and len(ids) == len(set(ids)) and np.array_equal(ids, expected),
                "保存训练名单不足、重复、未来或超出原两年范围。")
        require(len(ids) == int(row.training_n) and int(row.training_max_idx) == int(ids[-1])
                and pd.Timestamp(row.training_start) == source_dates[ids[0]]
                and pd.Timestamp(row.training_end) == source_dates[ids[-1]], "保存训练范围/数量不同。")
        require((availability[ids] < deadline).all(), "保存训练输入尚不可用。")
        require(np.isfinite(row[["intercept", "beta_usd", "beta_spread", "residual_fx_change5"]].to_numpy(float)).all(),
                "保存模型参数/残差不是有限值。")
        require(float(row.observed_fx_change5) == float(source.fx_change5.iloc[latest])
                and float(row.usd_change5) == float(source.usd_change5.iloc[latest])
                and float(row.spread_change5) == float(source.spread_change5.iloc[latest]), "保存解释变量与原源不同。")
        checked += 1
    return {"saved_available_model_metadata_checked": checked, "saved_training_lists_checked": len(training),
            "algorithm_input_sets_same_at_1505_and_original_1600": True,
            "saved_original_calculation_at_1600_not_proof_of_publication_at_1505": True,
            "ols_rank_parameters_residual_formula_not_recomputed": True}


def build_field(current, saved, source, training):
    clocks = check_saved_clocks(saved, source, training)
    require(current.symbol.eq("510300.SH").all(), "当前股票标的变化。")
    current_dates = dates(current.date, "当前股票日历")
    saved_dates = dates(saved.date, "保存残差")
    positions = {day: i for i, day in enumerate(saved_dates)}
    field = pd.DataFrame({"date": current_dates, "saved_source_row": [positions.get(day, -1) for day in current_dates],
        "field_status": "NO_VIEW_SAVED_SOURCE_DATE_NOT_REGISTERED", "first_publication_evidence_proved": False})
    field[FIELDS] = np.nan
    field["saved_original_decision_at"] = pd.Series(pd.NaT, index=field.index, dtype="datetime64[ns, UTC]")
    field["saved_source_available_at"] = pd.Series(pd.NaT, index=field.index, dtype="datetime64[ns, UTC]")
    field["current_decision_deadline"] = (current_dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15, minutes=5)).tz_convert("UTC")
    for t, j in enumerate(field.saved_source_row.to_numpy(int)):
        if j < 0:
            continue
        row = saved.iloc[j]
        field.loc[t, "saved_original_decision_at"] = pd.Timestamp(row.decision_at).tz_convert("UTC")
        if not bool(row.source_admitted):
            field.loc[t, "field_status"] = "NO_VIEW_ORIGINAL_"+str(row.model_status)
            continue
        field.loc[t, "saved_source_available_at"] = pd.Timestamp(row.source_available_at).tz_convert("UTC")
        field.loc[t, FIELDS[0]] = float(row.residual_fx_change5)
        field.loc[t, "field_status"] = "SAVED_ALGORITHM_1505_INPUT_CLOCK_AVAILABLE_FIRST_PUBLICATION_NOT_PROVED"
    return field, clocks


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
                and len(selected) == record["training_rows"] and selected.cycle_id.nunique() == len(ids), "原成熟成员数量变化。")
        require((selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(), "原训练周期尚未成熟。")
        positions = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[positions]).all(axis=1)
        missing = field.field_status.iloc[positions[~finite]].value_counts().to_dict()
        rows.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict), "original_training_cycles": len(ids),
            "original_training_rows": len(selected), "saved_source_available_rows": int(finite.sum()),
            "saved_source_missing_rows": int((~finite).sum()), "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in missing.items()}, ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)


def freeze():
    prior, tests = read_json(OUT/"prior_and_source_review.json"), read_json(OUT/"tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 7, "七项保存来源必要测试未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT/"tests/test_point_j02_saved_residual_intake_v1.py"), "测试后代码变化。")
    require(prior["saved_source_and_missing_rule_bound_before_current_member_quantities"], "当前成员数量前必须绑定保存源。")
    protocol = {"study": "510300_POINT_J02_SAVED_RESIDUAL_INTAKE_V1", "at": now(), "binding": prior["binding"],
        "purpose": "既有两年保存残差的来源/算法时钟/原成员核对，不建立新252日模型或改变原毛收益初筛。",
        "support_gate": prior["support_gate"], "no_rescue": prior["no_rescue"], "fixed_source_operationalizations": 1,
        "empirical_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "history_role": "DEVELOPMENT_CALIBRATION", "goal_achieved": False}
    paths = {Path(__file__), ROOT/"tests/test_point_j02_saved_residual_intake_v1.py", OUT/"tests_receipt.json",
             OUT/"prior_and_source_review.json", CURRENT/"inputs/candidate_features.parquet",
             CURRENT/"inputs/within_models.json", CURRENT/"results/training_reference/samples.parquet"}
    for item in prior["direct_sources"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "事前来源变化："+item["path"])
        paths.add(ROOT/item["path"])
    write_json(OUT/"protocol.json", protocol)
    write_json(OUT/"freeze.json", {"at": now(), "protocol_sha256": digest(OUT/"protocol.json"),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sorted(paths)]})
    return {"状态": "J02既有两年残差源及未知规则已冻结", "来源数": len(paths), "新增拟合": 0}


def check():
    frozen = read_json(OUT/"freeze.json")
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "来源协议变化。")
    for item in frozen["sources"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "冻结来源变化："+item["path"])
    return len(frozen["sources"])


def derive():
    current = pd.read_parquet(CURRENT/"inputs/candidate_features.parquet", columns=["date", "symbol"])
    samples = pd.read_parquet(CURRENT/"results/training_reference/samples.parquet", columns=METADATA)
    records = read_json(CURRENT/"inputs/within_models.json")["models"]
    require(len(current) == 3488 and len(samples) == 1507 and len(records) == 142, "原快照数量变化。")
    saved = pd.read_parquet(OLD/"daily_models.parquet")
    source = pd.read_parquet(OLD/"inputs/common_sources.parquet")
    validate_saved_object(saved, pd.read_parquet(OLD/"inputs/market.parquet", columns=["date", "symbol"]))
    field, clocks = build_field(current, saved, source, read_json(OLD/"training_indices.json"))
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "原可用成熟月份数量变化。")
    natural = field.iloc[samples.origin_index.to_numpy(int)].reset_index(drop=True)
    natural.insert(1, "cycle_id", samples.cycle_id.to_numpy(int))
    natural.insert(2, "origin_index", samples.origin_index.to_numpy(int))
    periods = []
    for name, start, end in [("2015_2019", "2015-01-05", "2019-12-31"), ("2020_2026", "2020-01-02", "2026-09-30")]:
        part = natural.loc[natural.date.between(start, end)]
        months = ready.loc[pd.to_datetime(ready.fit_origin).between(start, end)]
        periods.append({"period": name, "original_natural_rows": len(part),
            "available_natural_rows": int(np.isfinite(part[FIELDS].to_numpy(float)).all(axis=1).sum()),
            "original_mature_months": len(months), "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
            "missing_original_training_rows_range": [int(months.saved_source_missing_rows.min()), int(months.saved_source_missing_rows.max())]})
    finite = np.isfinite(natural[FIELDS].to_numpy(float)).all(axis=1)
    facts = {**clocks, "daily_rows": len(field), "saved_model_rows": len(saved), "common_source_rows": len(source),
        "saved_model_start": pd.Timestamp(saved.date.iloc[0]).isoformat(), "saved_model_end": pd.Timestamp(saved.date.iloc[-1]).isoformat(),
        "finite_current_daily_rows": int(np.isfinite(field[FIELDS].to_numpy(float)).all(axis=1).sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "original_natural_rows": len(natural), "supported_natural_rows": int(finite.sum()),
        "natural_status_counts": {str(k): int(v) for k, v in natural.field_status.value_counts().items()},
        "original_monthly_records": len(support), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(support)-len(ready), "periods": periods,
        "algorithmic_source_member_support_passed": bool(finite.all() and ready.all_original_members_supported.all()),
        "historical_first_publication_evidence": "NOT_ESTABLISHED_OLD_SOURCE_REVISION_AND_PROXY_CLOCK_BOUNDARY_PRESERVED",
        "source_gate_passed": False, "old_numeric_gross_screen": "PASSED_ON_SOURCE_SUBSAMPLE_ONLY",
        "old_source_gate": "NOT_PASSED", "old_account_status": "NOT_RUN"}
    return (field, natural, support), facts


def run():
    count = check()
    write_json(OUT/"ADMISSION_STARTED.json", {"at": now(), "new_model_fits": 0})
    tables, facts = derive()
    result = OUT/"results"
    result.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(result/(name+".parquet"), index=False)
        table.to_csv(result/(name+".csv"), index=False, encoding="utf-8-sig")
    summary = {"study": "510300_POINT_J02_SAVED_RESIDUAL_INTAKE_V1", "at": now(),
        "status": "NOT_ADMITTED_J02_EXISTING_TWO_YEAR_SAVED_SOURCE_PUBLICATION_AND_MEMBER_GATES_FAILED",
        "frozen_sources_unchanged": count, **facts, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0, "new_market_collection": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "仅裁决既有两年保存残差的当前用途；旧有源数值初筛通过与来源门失败分开，不宣称金融失败或重建252日版本。"}
    write_json(OUT/"summary.json", summary)
    require(check() == count, "运行后冻结来源变化。")
    return {"状态": summary["status"], "支持原成员": facts["supported_natural_rows"],
            "完整原成熟月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        pd.testing.assert_frame_equal(pd.read_parquet(OUT/"results"/(name+".parquet")), rebuilt, check_exact=True)
    summary = read_json(OUT/"summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "保存来源支持结果不同。")
    receipt = {"at": now(), "status": "PASS_SAVED_J02_RESIDUAL_SOURCE_CLOCK_AND_MEMBER_RECOMPUTATION",
        "frozen_sources_unchanged": count, "tables_exactly_equal": True, "daily_rows": len(tables[0]),
        "natural_member_rows": len(tables[1]), "monthly_member_rows": len(tables[2]), "new_model_fits": 0,
        "new_return_labels": 0, "new_strategy_accounts": 0, "scope": "三表及保存元数据复核，非OLS/首版/收益或账户复算。"}
    write_json(OUT/"saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="J02已有保存人民币残差来源准入，不重拟合/回测")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
