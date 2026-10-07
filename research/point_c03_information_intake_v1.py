"""C03固定十日量价效率差的时钟和原成员数量准入，不拟合模型。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_c03_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FIELD = "up_down_amount_efficiency_difference10"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
PRICE_COLUMNS = ["date", "symbol", "close", "previous_close", "dividend", "total_simple",
                 "total_log", "wealth", "amount", "amount_unit"]
TABLES = ("日线C03事前字段", "原自然成员C03支持", "原月度成熟成员C03支持")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def validate_source_clock(data):
    """原价格和当日已除息现金前向财富，绑定经济log收益，不回填初始区间。"""
    require(data.symbol.eq("510300.SH").all(), "C03标的身份不同。")
    close, previous, cash, simple, logs, wealth = data[[
        "close", "previous_close", "dividend", "total_simple", "total_log", "wealth"]].to_numpy(float).T
    require(np.isfinite(close).all() and (close > 0).all(), "C03收盘缺失或非正。")
    require(np.isfinite(cash).all() and (cash >= 0).all(), "C03现金除息事实缺失或为负。")
    require(pd.isna(previous[0]) and pd.isna(simple[0]) and pd.isna(logs[0]),
            "C03首个市场点没有前一区间，不用初始归一值填收益。")
    np.testing.assert_allclose(previous[1:], close[:-1], atol=1e-12, rtol=0)
    expected = (close[1:]+cash[1:])/previous[1:]-1
    require(np.isfinite(simple[1:]).all() and np.isfinite(logs[1:]).all(), "C03非初始经济收益缺失。")
    np.testing.assert_allclose(simple[1:], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(logs[1:], np.log1p(expected), atol=1e-12, rtol=0)
    np.testing.assert_allclose(wealth, np.r_[1., np.cumprod(1+expected)], atol=1e-12, rtol=1e-12)


def build_field(data):
    """固定十个完整实际市场日，正负日分别归组，任一分母为零则未知。"""
    require(set(PRICE_COLUMNS).issubset(data.columns), "C03源字段缺失。")
    require(data.symbol.eq("510300.SH").all() and data.amount_unit.eq("CNY").all(),
            "C03标的或成交额单位必须为510300.SH/CNY。")
    dates = pd.DatetimeIndex(data.date).astype("datetime64[ns]")
    require(len(data) > 0 and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
            and (dates.dayofweek <= 4).all(), "C03市场日期必须唯一递增且为工作日。")
    r = pd.Series(data.total_log.to_numpy(float))
    amount = pd.Series(data.amount.to_numpy(float))
    r.iloc[0] = np.nan
    valid = np.isfinite(r) & np.isfinite(amount) & amount.gt(0)
    returns, amounts = r.where(valid), amount.where(valid)
    positive, negative = returns.gt(0), returns.lt(0)
    up_return = returns.where(positive, 0.).where(valid).rolling(10, min_periods=10).sum()
    down_return = (-returns).where(negative, 0.).where(valid).rolling(10, min_periods=10).sum()
    up_amount = amounts.where(positive, 0.).where(valid).rolling(10, min_periods=10).sum()
    down_amount = amounts.where(negative, 0.).where(valid).rolling(10, min_periods=10).sum()
    average_amount = amounts.rolling(10, min_periods=10).mean()
    count = valid.astype(int).rolling(10, min_periods=10).sum()
    complete = count.eq(10)
    available = complete & up_amount.gt(0) & down_amount.gt(0)
    up_efficiency = average_amount*up_return/up_amount.where(up_amount.gt(0))
    down_efficiency = average_amount*down_return/down_amount.where(down_amount.gt(0))
    value = (up_efficiency-down_efficiency).where(available)
    require(np.isfinite(value.dropna().to_numpy()).all(), "C03效率差非有限，不替代异常值。")
    status = np.full(len(data), "NO_VIEW_INCOMPLETE_TEN_ACTUAL_DAYS", dtype=object)
    status[(complete & ~up_amount.gt(0)).to_numpy()] = "NO_VIEW_ZERO_POSITIVE_GROUP_AMOUNT"
    status[(complete & ~down_amount.gt(0)).to_numpy()] = "NO_VIEW_ZERO_NEGATIVE_GROUP_AMOUNT"
    status[(complete & ~up_amount.gt(0) & ~down_amount.gt(0)).to_numpy()] = "NO_VIEW_BOTH_GROUP_AMOUNTS_ZERO"
    status[available.to_numpy()] = "C03_TEN_COMPLETE_DAYS_AND_BOTH_GROUPS_AVAILABLE"
    index = np.arange(len(data))
    return pd.DataFrame({"date": dates, "economic_log_return": r.to_numpy(), "amount_cny": amount.to_numpy(),
        "positive_return_sum10": up_return.to_numpy(), "negative_abs_return_sum10": down_return.to_numpy(),
        "positive_amount_sum10": up_amount.to_numpy(), "negative_amount_sum10": down_amount.to_numpy(),
        "mean_amount10": average_amount.to_numpy(), "up_efficiency10": up_efficiency.to_numpy(),
        "down_efficiency10": down_efficiency.to_numpy(), FIELD: value.to_numpy(), "field_status": status,
        "valid_daily_source": valid.to_numpy(), "ten_day_start_index": np.where(index >= 9, index-9, -1),
        "latest_source_index": index})


def member_support(samples, records, field):
    """保留每个原训练成员；真实比值零有限，分组缺失和窗口未知不补零。"""
    require(set(METADATA).issubset(samples.columns), "C03原样本元数据缺失。")
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "C03原点越界。")
    left = pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]")
    right = pd.to_datetime(field.date.iloc[indexes]).to_numpy(dtype="datetime64[ns]")
    require(np.array_equal(left, right), "C03原点日期与冻结日线不一致。")
    values, states = field[FIELD].to_numpy(float), field.field_status.to_numpy(str)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"], "C03原月周期清单重复或变化。")
        require(len(selected) == record["training_rows"], "C03原训练成员数量变化。")
        require(selected.cycle_id.nunique() == len(ids), "C03原训练周期有缺失成员。")
        require((selected.exit_index <= record["fit_index"]).all(), "C03读取尚未成熟周期。")
        require((pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "C03原成员在拟合日期尚未成熟。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx])
        counts = pd.Series(states[idx][~finite]).value_counts().to_dict()
        rows.append({
            "fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "c03_available_rows": int(finite.sum()), "c03_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in counts.items()},
                                               ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)


def source_paths():
    prior = read_json(OUT / "prior_and_source_review.json")
    paths = [Path(__file__), ROOT / "tests/test_point_c03_information_intake_v1.py",
             OUT / "prior_and_source_review.json", OUT / "tests_receipt.json", DATA, SAMPLES, MODELS]
    paths += [ROOT / item["path"] for item in prior["direct_sources"]]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    tests = read_json(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 7, "C03七项必要测试尚未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_c03_information_intake_v1.py"),
            "C03代码在测试后变化。")
    prior = read_json(OUT / "prior_and_source_review.json")
    require(prior["conditional_purpose_and_source_clock_bound"], "C03用途/经济收益及金额时钟未绑定。")
    protocol = {"study": "510300_POINT_C03_INFORMATION_INTAKE_V1", "frozen_at": now(),
        "purpose": "收益拟合前检验固定C03直接第九项的原完整成员支持，不读取未来收益标签。",
        "registered_definition": prior["registered_definition"], "field_binding": prior["field_binding"],
        "source_clock": prior["source_clock"], "missing_rule": prior["missing_rule"],
        "support_gate": prior["support_gate"], "no_rescue": prior["no_rescue"],
        "fixed_operationalizations": 1, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False}
    paths = source_paths()
    sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in paths]
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    return {"状态": "C03固定十日操作定义及来源已冻结", "来源数": len(sources), "新增拟合": 0}


def check():
    frozen = read_json(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "C03协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "C03冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def derive():
    data = pd.read_parquet(DATA, columns=PRICE_COLUMNS)
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "C03原快照数量改变。")
    validate_source_clock(data)
    field = build_field(data)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "C03原可用模型数改变。")
    natural = field.iloc[samples.origin_index.to_numpy(int)].reset_index(drop=True)
    natural.insert(1, "cycle_id", samples.cycle_id.to_numpy(int))
    natural.insert(2, "origin_index", samples.origin_index.to_numpy(int))
    periods = []
    for period, start, end in (("2015_2019", "2015-01-05", "2019-12-31"),
                               ("2020_2026", "2020-01-02", "2026-09-30")):
        part = natural.loc[natural.date.between(start, end)]
        months = ready.loc[pd.to_datetime(ready.fit_origin).between(start, end)]
        periods.append({
            "period": period, "original_natural_rows": len(part),
            "available_natural_rows": int(part[FIELD].notna().sum()),
            "original_mature_months": len(months),
            "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
            "minimum_missing_original_training_rows": int(months.c03_missing_rows.min()) if len(months) else None,
        })
    accepted = bool(natural[FIELD].notna().all() and ready.all_original_members_supported.all())
    facts = {
        "support_gate_passed": accepted, "daily_rows": len(field),
        "finite_daily_field_rows": int(field[FIELD].notna().sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "zero_positive_group_daily_rows": int(field.field_status.eq("NO_VIEW_ZERO_POSITIVE_GROUP_AMOUNT").sum()),
        "zero_negative_group_daily_rows": int(field.field_status.eq("NO_VIEW_ZERO_NEGATIVE_GROUP_AMOUNT").sum()),
        "genuine_zero_available_daily_rows": int(field[FIELD].eq(0).sum()),
        "original_natural_rows": len(natural), "supported_natural_rows": int(natural[FIELD].notna().sum()),
        "genuine_zero_supported_natural_rows": int(natural[FIELD].eq(0).sum()),
        "original_monthly_records": len(support), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(support) - len(ready), "periods": periods,
    }
    return (field, natural, support), facts


def run():
    count = check()
    write_json(OUT / "ADMISSION_STARTED.json", {"at": now(), "new_model_fits": 0})
    tables, facts = derive()
    status = ("ADMITTED_FIXED_C03_MEMBER_SUPPORT_ONLY_NO_MODEL" if facts["support_gate_passed"]
              else "REJECTED_FIXED_C03_NINTH_FIELD_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT")
    results = OUT / "results"
    results.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(results / (name + ".parquet"), index=False)
        table.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {
        "study": "510300_POINT_C03_INFORMATION_INTAKE_V1", "completed_at": now(), "status": status,
        "fixed_operationalizations": 1, "frozen_sources_unchanged": count, **facts,
        "empirical_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "new_market_bars": 0, "financial_metrics": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "只裁决固定C03直接第九项的原成员支持，不检验整个量价非对称机制收益；分母缺失不补零或删行。",
    }
    write_json(OUT / "summary.json", summary)
    require(check() == count, "C03运行后冻结来源改变。")
    return {"状态": status, "原自然成员": len(tables[1]), "可用成员": facts["supported_natural_rows"],
            "完整原可用月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        saved = pd.read_parquet(OUT / "results" / (name + ".parquet"))
        pd.testing.assert_frame_equal(saved, rebuilt, check_exact=True)
    summary = read_json(OUT / "summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "C03保存支持指标复算不同。")
    receipt = {
        "at": now(), "status": "PASS_SAVED_C03_RETURN_AMOUNT_FIELDS_AND_ORIGINAL_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "daily_rows": len(tables[0]),
        "natural_member_rows": len(tables[1]), "monthly_member_rows": len(tables[2]),
        "tables_and_support_exactly_equal": True, "new_model_fits": 0,
        "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "固定十日量价字段和原成员数量复算，不是收益验证或独立验证。",
    }
    write_json(OUT / "saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="C03固定量价效率差数量准入，不运行策略")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
