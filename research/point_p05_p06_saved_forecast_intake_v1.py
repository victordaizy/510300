"""P05/P06既有预测的成熟时钟、原成员覆盖及发布证据准入，不训练或回测。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_p05_p06_saved_forecast_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OLD = ROOT / "reports/research/510300_factor96_daily_state_shrink_v1"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FORECAST = OLD / "daily_forecasts_lag1.parquet"
OUTER = OLD / "outer_models_lag1.json"
INNER = OLD / "inner_models_lag1.json"
MARKET = OLD / "inputs/market.parquet"
SOURCE_FIELDS = ["positive_agreement", "disagreement", "cusum", "monitor_paused"]
FIELDS = ["P05_positive_agreement", "P05_normalized_disagreement",
          "P06_mature_error_cusum", "P06_monitor_paused"]
SOURCE_COLUMNS = ["date", *SOURCE_FIELDS, "model_known", "inner_count", "training_count",
                  "external_stat_idx", "external_stat_date"]
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
TABLES = ("日线P05P06保存预测来源支持", "原自然成员P05P06来源支持", "原月度成熟成员P05P06来源支持")


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
        stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def checked_dates(values, name):
    dates = pd.DatetimeIndex(values).astype("datetime64[ns]")
    require(not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing,
            name + "日期必须唯一递增且不缺失。")
    return dates


def validate_saved_object(saved, market):
    """源标的及原2025截止投影身份校验，不把旧输入延伸年份补入保存预测。"""
    require(market.symbol.eq("510300.SH").all(), "旧预测源标的必须为510300.SH。")
    projected = market.loc[pd.to_datetime(market.date) <= pd.Timestamp("2025-12-31")]
    require(np.array_equal(checked_dates(projected.date, "旧市场投影").to_numpy(),
                           checked_dates(saved.date, "保存预测").to_numpy()),
            "保存预测与原截止市场投影日期不同。")


def check_saved_clocks(saved, outer, inner):
    """只检查保存的训练范围/成熟上限及主一日外部时钟，不复算拟合或标签。"""
    require(set(SOURCE_COLUMNS).issubset(saved.columns), "保存预测源字段不足。")
    dates = checked_dates(saved.date, "保存预测")
    known = saved.model_known.eq(True).to_numpy()
    outer_ids = [int(record["decision_idx"]) for record in outer]
    outer_set = set(outer_ids)
    require(len(outer_ids) == len(outer_set) and outer_set == set(np.flatnonzero(known)),
            "保存已知模型与外层记录不一致。")
    grouped = {}
    for record in inner:
        t = int(record["decision_idx"])
        require(t in outer_set, "内层记录缺少对应外层模型。")
        grouped.setdefault(t, []).append(record)
    count = 0
    for record in outer:
        t = int(record["decision_idx"])
        require(0 <= t < len(saved) and pd.Timestamp(record["date"]) == dates[t], "保存外层模型日期不一致。")
        lower = dates[t] - pd.DateOffset(years=2)
        require(pd.Timestamp(record["lower"]) == lower, "保存外层两年下限变化。")
        ids = np.asarray(record["training_indices"], dtype=np.int64)
        require(len(ids) >= 252 and len(ids) == len(set(ids))
                and (ids >= 0).all() and (ids+6 < t).all(), "外层训练成员不足、重复或尚未成熟。")
        require((dates[ids] >= lower).all(), "外层训练超出当前两年下限。")
        require(int(record["latest_training_exit_idx"]) == int(ids.max())+6,
                "保存外层最新成熟索引不一致。")
        require(int(saved.training_count.iloc[t]) == len(ids), "保存外层训练数量不一致。")
        require(int(saved.external_stat_idx.iloc[t]) == t-1
                and pd.Timestamp(saved.external_stat_date.iloc[t]) == dates[t-1],
                "保存源不是原主一日外部时钟。")
        records = grouped.get(t, [])
        validation_ids = [int(item["validation_idx"]) for item in records]
        require(len(records) >= 20 and len(validation_ids) == len(set(validation_ids))
                and validation_ids == record["inner_validation_indices"]
                and int(saved.inner_count.iloc[t]) == len(records), "保存内层成熟成员不足或清单变化。")
        for item in records:
            s = int(item["validation_idx"])
            first, last = int(item["training_first_idx"]), int(item["training_last_idx"])
            require(s % 5 == 0 and s+6 < t and int(item["validation_exit_idx"]) == s+6,
                    "内层验证相位错误或当前尚未成熟。")
            require(0 <= first <= last and last+6 < s and int(item["training_count"]) >= 60,
                    "内层训练上限尚未成熟或成员不足。")
            require(pd.Timestamp(item["lower"]) == lower and dates[first] >= lower
                    and dates[s] >= dates[t]-pd.DateOffset(years=1), "内层训练或验证超出固定时钟。")
            count += 1
    return {"outer_saved_clock_records_checked": len(outer), "inner_saved_clock_records_checked": count,
            "checks_are_saved_metadata_only_no_fit_or_label_recomputation": True}


def build_field(current, saved, outer, inner):
    """按日期精确连接原主预测；缺日期、原无模型及不完整记录保持未知。"""
    require(current.symbol.eq("510300.SH").all(), "当前标的必须为510300.SH。")
    dates = checked_dates(current.date, "当前市场")
    source_dates = checked_dates(saved.date, "保存预测")
    clocks = check_saved_clocks(saved, outer, inner)
    source_values = saved[SOURCE_FIELDS].to_numpy(float)
    known = saved.model_known.eq(True).to_numpy()
    finite = np.isfinite(source_values).all(axis=1)
    good = known & finite
    require(((source_values[good, 0] >= 0) & (source_values[good, 0] <= 1)).all()
            and (source_values[good, 1:3] >= 0).all()
            and np.isin(source_values[good, 3], [0., 1.]).all(), "保存四列的合法范围变化。")
    indexes = source_dates.get_indexer(dates)
    values = np.full((len(current), 4), np.nan)
    states = np.full(len(current), "NO_VIEW_SOURCE_DATE_NOT_SAVED", dtype=object)
    for i, source in enumerate(indexes):
        if source < 0:
            continue
        if not known[source]:
            states[i] = "NO_VIEW_SOURCE_ORIGINAL_MODEL_NOT_READY"
        elif not finite[source]:
            states[i] = "NO_VIEW_INCOMPLETE_SAVED_FORECAST_FIELDS"
        else:
            values[i] = source_values[source]
            states[i] = "SAVED_FORECAST_ALGORITHM_CLOCK_AVAILABLE_PUBLICATION_NOT_PROVED"
    result = pd.DataFrame(values, columns=FIELDS)
    result.insert(0, "date", dates)
    result["source_index"] = indexes
    result["field_status"] = states
    # 算法成员成熟不证明外部原始资料历史首次发布版本；本轮没有新证据补齐。
    result["first_publication_evidence_proved"] = False
    return result, clocks


def member_support(samples, records, field):
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "原自然成员索引越界。")
    require(np.array_equal(pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]"),
                           pd.to_datetime(field.date.iloc[indexes]).to_numpy(dtype="datetime64[ns]")),
            "原自然成员日期变化。")
    values = field[FIELDS].to_numpy(float)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"]
                and len(selected) == record["training_rows"]
                and selected.cycle_id.nunique() == len(ids), "原成熟训练成员数量变化。")
        require((selected.exit_index <= record["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "原训练周期尚未成熟。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx]).all(axis=1)
        states = field.field_status.iloc[idx[~finite]].value_counts().to_dict()
        rows.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "saved_source_available_rows": int(finite.sum()), "saved_source_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in states.items()},
                                               ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)


def freeze():
    tests, prior = read_json(OUT/"tests_receipt.json"), read_json(OUT/"prior_and_source_review.json")
    require(tests["passed"] and tests["tests"] == 7, "七项来源/时钟必要测试未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT/"tests/test_point_p05_p06_saved_forecast_intake_v1.py"),
            "来源检查代码在测试后变化。")
    require(prior["saved_source_fields_primary_clock_and_missing_rule_bound"], "保存四列/主时钟/未知未绑定。")
    protocol = {"study": "510300_POINT_P05_P06_SAVED_FORECAST_INTAKE_V1", "at": now(),
        "purpose": "只检查既有主时钟保存预测的算法成熟、原成员覆盖和发布证据，不训练或回测。",
        "registered_definitions": prior["registered_definitions"], "field_binding": prior["field_binding"],
        "support_gate": prior["support_gate"], "missing_rule": prior["missing_rule"],
        "no_rescue": prior["no_rescue"], "fixed_source_operationalizations": 1,
        "empirical_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "history_role": "DEVELOPMENT_CALIBRATION", "goal_achieved": False}
    paths = {Path(__file__), ROOT/"tests/test_point_p05_p06_saved_forecast_intake_v1.py",
             OUT/"tests_receipt.json", OUT/"prior_and_source_review.json", DATA, SAMPLES, MODELS, FORECAST, OUTER, INNER, MARKET}
    for item in prior["direct_sources"]:
        path = ROOT/item["path"]
        require(digest(path) == item["sha256"], "事前来源改变："+item["path"])
        paths.add(path)
    write_json(OUT/"protocol.json", protocol)
    write_json(OUT/"freeze.json", {"at": now(), "protocol_sha256": digest(OUT/"protocol.json"),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sorted(paths)]})
    return {"状态": "P05/P06保存四列及来源口径已冻结", "来源数": len(paths), "新增拟合": 0}


def check():
    frozen = read_json(OUT/"freeze.json")
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "来源协议改变。")
    for item in frozen["sources"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "冻结来源改变："+item["path"])
    return len(frozen["sources"])


def derive():
    current = pd.read_parquet(DATA, columns=["date", "symbol"])
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(current) == 3488 and len(samples) == 1507 and len(records) == 142, "原快照数量变化。")
    saved = pd.read_parquet(FORECAST, columns=SOURCE_COLUMNS)
    validate_saved_object(saved, pd.read_parquet(MARKET, columns=["date", "symbol"]))
    field, clocks = build_field(current, saved, read_json(OUTER), read_json(INNER))
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "原可用月数变化。")
    natural = field.iloc[samples.origin_index.to_numpy(int)].reset_index(drop=True)
    natural.insert(1, "cycle_id", samples.cycle_id.to_numpy(int))
    natural.insert(2, "origin_index", samples.origin_index.to_numpy(int))
    periods = []
    for period, start, end in (("2015_2019", "2015-01-05", "2019-12-31"),
                               ("2020_2026", "2020-01-02", "2026-09-30")):
        part = natural.loc[natural.date.between(start, end)]
        months = ready.loc[pd.to_datetime(ready.fit_origin).between(start, end)]
        periods.append({"period": period, "original_natural_rows": len(part),
            "available_natural_rows": int(np.isfinite(part[FIELDS].to_numpy(float)).all(axis=1).sum()),
            "original_mature_months": len(months), "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
            "minimum_missing_original_training_rows": int(months.saved_source_missing_rows.min()) if len(months) else None})
    natural_finite = np.isfinite(natural[FIELDS].to_numpy(float)).all(axis=1)
    facts = {**clocks, "daily_rows": len(field), "saved_forecast_rows": len(saved),
        "saved_source_start": pd.Timestamp(saved.date.iloc[0]).isoformat(),
        "saved_source_end": pd.Timestamp(saved.date.iloc[-1]).isoformat(),
        "finite_current_daily_rows": int(np.isfinite(field[FIELDS].to_numpy(float)).all(axis=1).sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "original_natural_rows": len(natural), "supported_natural_rows": int(natural_finite.sum()),
        "natural_status_counts": {str(k): int(v) for k, v in natural.field_status.value_counts().items()},
        "original_monthly_records": len(support), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(support)-len(ready), "periods": periods,
        "algorithmic_source_member_support_passed": bool(natural_finite.all() and ready.all_original_members_supported.all()),
        "historical_first_publication_evidence": "NOT_ESTABLISHED_OLD_SOURCE_LIMIT_PRESERVED",
        "source_gate_passed": False,
        "saved_formula_identity": "NOT_RECOMPUTED_SAVED_METADATA_AND_COVERAGE_ONLY"}
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
    summary = {"study": "510300_POINT_P05_P06_SAVED_FORECAST_INTAKE_V1", "at": now(),
        "status": "NOT_ADMITTED_P05_P06_SAVED_FORECAST_PUBLICATION_AND_MEMBER_SOURCE_GATES_FAILED",
        "frozen_sources_unchanged": count, **facts, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0, "new_market_bars": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "只拒绝已有保存预测的当前用途准入，不证明P05/P06所有机制失效；成熟上限检查不是首次发布证明。"}
    write_json(OUT/"summary.json", summary)
    require(check() == count, "运行后冻结来源改变。")
    return {"状态": summary["status"], "可用原成员": facts["supported_natural_rows"],
            "完整原可用月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        pd.testing.assert_frame_equal(pd.read_parquet(OUT/"results"/(name+".parquet")), rebuilt, check_exact=True)
    summary = read_json(OUT/"summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "保存来源支持结果不同。")
    receipt = {"at": now(), "status": "PASS_SAVED_P05_P06_SOURCE_CLOCK_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "tables_exactly_equal": True,
        "daily_rows": len(tables[0]), "natural_member_rows": len(tables[1]), "monthly_member_rows": len(tables[2]),
        "saved_metadata_clock_checks_repeated": True, "new_model_fits": 0,
        "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "三份来源支持表及保存成熟上限元数据复算，不重拟合/复算旧标签或账户，不是发布/收益验证。"}
    write_json(OUT/"saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="P05/P06保存预测来源及原成员准入，不训练/回测")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
