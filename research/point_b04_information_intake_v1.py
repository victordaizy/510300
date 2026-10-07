"""B04固定冲击后波动比及新低标记的原成员准入；不读取收益标签或拟合。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_b04_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FIELD = "post_shock_vol3_ratio"
LOW = "below_frozen_shock_low"
FIELDS = [FIELD, LOW]
TICK = 0.001
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "previous_close",
                 "dividend", "total_simple", "total_log", "wealth"]
TABLES = ("日线B04事前字段", "原自然成员B04支持", "原月度成熟成员B04支持")


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
    """校验原OHLC、当日已除息现金与前向财富；首个市场点没有收益区间。"""
    require(len(data) > 0 and data.symbol.eq("510300.SH").all(), "B04标的或市场快照不同。")
    prices = data[["open", "high", "low", "close"]].to_numpy(float)
    require(np.isfinite(prices).all() and (prices > 0).all(), "B04价格缺失或非正。")
    opens, highs, lows, close = prices.T
    require(((lows <= opens) & (lows <= close) & (highs >= opens) & (highs >= close)).all(),
            "B04原OHLC关系不合法。")
    previous, cash, simple, logs, wealth = data[[
        "previous_close", "dividend", "total_simple", "total_log", "wealth"]].to_numpy(float).T
    require(np.isfinite(cash).all() and (cash >= 0).all(), "B04当日已除息现金缺失或为负。")
    require(pd.isna(previous[0]) and pd.isna(simple[0]) and pd.isna(logs[0]),
            "B04首个市场点没有前一区间，不填初始收益。")
    np.testing.assert_allclose(previous[1:], close[:-1], atol=1e-12, rtol=0)
    expected = (close[1:] + cash[1:]) / close[:-1] - 1
    require(np.isfinite(simple[1:]).all() and np.isfinite(logs[1:]).all(), "B04非初始收益缺失。")
    np.testing.assert_allclose(simple[1:], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(logs[1:], np.log1p(expected), atol=1e-12, rtol=0)
    np.testing.assert_allclose(wealth, np.r_[1., np.cumprod(1 + expected)], atol=1e-12, rtol=1e-12)


def quote_units(values, name, positive):
    """只核验既有0.001报价/现金单位，不修复离格数值。"""
    raw = np.asarray(values, dtype=float)
    valid = np.isfinite(raw) & ((raw > 0) if positive else (raw >= 0))
    scaled = raw[valid] / TICK
    require((np.abs(scaled) <= 1e15).all(), "B04" + name + "超出整数报价安全范围。")
    nearest = np.rint(scaled)
    require((np.abs(scaled - nearest) <= 1e-7).all(), "B04" + name + "不符合冻结0.001单位。")
    units = np.zeros(len(raw), dtype=np.int64)
    units[valid] = nearest.astype(np.int64)
    return units, valid


def sample_std(block):
    """等价样本标准差；先中心平移，保持真正恒定区间的零方差。"""
    return float(np.std(block - block[0], ddof=1))


def build_field(data):
    """冲击去重10日；后1—10日逐日记录，不等待未来确认、不延长阶段。"""
    require(set(PRICE_COLUMNS).issubset(data.columns), "B04源字段缺失。")
    require(data.symbol.eq("510300.SH").all(), "B04标的必须为510300.SH。")
    dates = pd.DatetimeIndex(data.date).astype("datetime64[ns]")
    n = len(data)
    require(n > 0 and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
            and (dates.dayofweek <= 4).all(), "B04市场日期必须唯一递增且为工作日。")
    close, cv = quote_units(data.close, "收盘", True)
    low, lv = quote_units(data.low, "最低价", True)
    cash, dv = quote_units(data.dividend, "当日已除息现金", False)
    returns = data.total_log.to_numpy(float)
    previous_valid = np.r_[False, cv[:-1]]
    interval_valid = cv & dv & previous_valid & np.isfinite(returns)
    values = np.full(n, np.nan)
    new_low = np.full(n, np.nan)
    references = np.full(n, np.nan)
    origins = np.full(n, -1, dtype=np.int64)
    ages = np.full(n, np.nan)
    earliest = np.full(n, -1, dtype=np.int64)
    raw_shocks = np.zeros(n, dtype=bool)
    accepted_shocks = np.zeros(n, dtype=bool)
    ignored_shocks = np.zeros(n, dtype=bool)
    states = np.full(n, "NO_VIEW_NO_ACTIVE_POST_SHOCK_STAGE", dtype=object)
    active, last = None, -1000
    for i in range(n):
        if i < 21:
            states[i] = "NO_VIEW_PRE_SHOCK_20_REAL_INTERVALS_NOT_READY"
            continue
        if not interval_valid[i-20:i+1].all():
            active, last = None, -1000
            states[i] = "NO_VIEW_MISSING_RAW_RETURN_OR_CASH_SOURCE"
            continue
        denominator = sample_std(returns[i-20:i])
        if not np.isfinite(denominator) or denominator <= 0:
            active, last = None, -1000
            states[i] = "NO_VIEW_NONPOSITIVE_PRE_SHOCK_VARIANCE"
            continue
        raw_shocks[i] = bool(returns[i] < -2 * denominator)
        if raw_shocks[i] and i - last >= 10:
            active, last = (i, denominator), i
            accepted_shocks[i] = True
        elif raw_shocks[i]:
            ignored_shocks[i] = True
        if active is None:
            continue
        e, pre_vol = active
        if i - e > 10:
            active = None
            continue
        origins[i], ages[i], references[i], earliest[i] = e, i-e, pre_vol, e-20
        if i == e:
            states[i] = "NO_VIEW_SHOCK_DAY_NOT_POST_SHOCK"
            continue
        if not lv[e]:
            states[i] = "NO_VIEW_FROZEN_SHOCK_LOW_SOURCE_MISSING"
            continue
        if not lv[i]:
            states[i] = "NO_VIEW_CURRENT_LOW_SOURCE_MISSING"
            continue
        require((cv[e:i+1] & dv[e:i+1]).all(), "B04事件内价格现金源异常，不跨未知延续。")
        # 比较L[t]/L[e]，L[j]=W[j]*(low[j]+cash[j])/(close[j]+cash[j])。
        # Python整数积严格比较；除息后的相等低点不受浮点舍入影响。
        gross_numerator, gross_denominator = 1, 1
        for j in range(e+1, i+1):
            gross_numerator *= int(close[j]) + int(cash[j])
            gross_denominator *= int(close[j-1])
        numerator = gross_numerator * (int(low[i])+int(cash[i])) * (int(close[e])+int(cash[e]))
        divisor = gross_denominator * (int(close[i])+int(cash[i])) * (int(low[e])+int(cash[e]))
        values[i] = sample_std(returns[i-2:i+1]) / pre_vol
        new_low[i] = float(numerator < divisor)
        require(np.isfinite(values[i]), "B04三日波动比非有限，不替代异常值。")
        states[i] = "B04_POST_SHOCK_VOLATILITY_AND_LOW_AVAILABLE"
    return pd.DataFrame({"date": dates, FIELD: values, LOW: new_low, "field_status": states,
        "raw_shock": raw_shocks, "accepted_shock": accepted_shocks,
        "deduplicated_shock": ignored_shocks, "shock_origin_index": origins,
        "post_shock_age": ages, "frozen_pre_shock_rv20": references,
        "earliest_source_index": earliest, "latest_source_index": np.arange(n)})


def member_support(samples, records, field):
    """完整保留原成员；阶段外未知不填零、不删除、不另开条件样本。"""
    require(set(METADATA).issubset(samples.columns), "B04原样本元数据缺失。")
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "B04原点越界。")
    left = pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]")
    right = pd.to_datetime(field.date.iloc[indexes]).to_numpy(dtype="datetime64[ns]")
    require(np.array_equal(left, right), "B04原点日期与冻结日线不一致。")
    values, states = field[FIELDS].to_numpy(float), field.field_status.to_numpy(str)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"], "B04原月周期清单重复或变化。")
        require(len(selected) == record["training_rows"], "B04原训练成员数量变化。")
        require(selected.cycle_id.nunique() == len(ids), "B04原训练周期有缺失成员。")
        require((selected.exit_index <= record["fit_index"]).all(), "B04读取尚未成熟周期。")
        require((pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "B04原成员在拟合日期尚未成熟。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx]).all(axis=1)
        counts = pd.Series(states[idx][~finite]).value_counts().to_dict()
        rows.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "b04_available_rows": int(finite.sum()), "b04_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in counts.items()},
                                               ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)


def source_paths():
    prior = read_json(OUT / "prior_and_source_review.json")
    paths = [Path(__file__), ROOT / "tests/test_point_b04_information_intake_v1.py",
             OUT / "prior_and_source_review.json", OUT / "tests_receipt.json", DATA, SAMPLES, MODELS]
    paths += [ROOT / item["path"] for item in prior["direct_sources"]]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    tests = read_json(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 7, "B04七项必要测试尚未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_b04_information_intake_v1.py"),
            "B04代码在测试后变化。")
    prior = read_json(OUT / "prior_and_source_review.json")
    require(prior["conditional_purpose_and_source_clock_bound"], "B04事件/用途/现金低点时钟未绑定。")
    protocol = {"study": "510300_POINT_B04_INFORMATION_INTAKE_V1", "frozen_at": now(),
        "purpose": "拟合前检验固定B04两列直接追加原八项的完整成员支持，不读取未来收益标签。",
        "registered_definition": prior["registered_definition"], "field_binding": prior["field_binding"],
        "source_clock": prior["source_clock"], "missing_rule": prior["missing_rule"],
        "support_gate": prior["support_gate"], "no_rescue": prior["no_rescue"],
        "fixed_operationalizations": 1, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False}
    for source in prior["direct_sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "B04事前来源改变：" + source["path"])
    sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in source_paths()]
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    return {"状态": "B04冲击后固定两列操作定义和来源已冻结", "来源数": len(sources), "新增拟合": 0}


def check():
    frozen = read_json(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "B04协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "B04冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def derive():
    data = pd.read_parquet(DATA, columns=PRICE_COLUMNS)
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "B04原快照数量改变。")
    validate_source_clock(data)
    field = build_field(data)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "B04原可用模型数改变。")
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
            "original_mature_months": len(months),
            "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
            "minimum_missing_original_training_rows": int(months.b04_missing_rows.min()) if len(months) else None})
    finite = np.isfinite(field[FIELDS].to_numpy(float)).all(axis=1)
    natural_finite = np.isfinite(natural[FIELDS].to_numpy(float)).all(axis=1)
    accepted = bool(natural_finite.all() and ready.all_original_members_supported.all())
    facts = {"support_gate_passed": accepted, "daily_rows": len(field),
        "finite_daily_field_rows": int(finite.sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "raw_shock_days": int(field.raw_shock.sum()), "accepted_shock_days": int(field.accepted_shock.sum()),
        "deduplicated_shock_days": int(field.deduplicated_shock.sum()),
        "volatility_ratio_above_one_daily_rows": int(field[FIELD].gt(1).sum()),
        "genuine_zero_volatility_daily_rows": int(field[FIELD].eq(0).sum()),
        "new_low_daily_rows": int(field[LOW].eq(1).sum()),
        "known_no_new_low_daily_rows": int(field[LOW].eq(0).sum()),
        "original_natural_rows": len(natural), "supported_natural_rows": int(natural_finite.sum()),
        "natural_status_counts": {str(k): int(v) for k, v in natural.field_status.value_counts().items()},
        "original_monthly_records": len(support), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(support)-len(ready), "periods": periods}
    return (field, natural, support), facts


def run():
    count = check()
    write_json(OUT / "ADMISSION_STARTED.json", {"at": now(), "new_model_fits": 0})
    tables, facts = derive()
    status = ("ADMITTED_FIXED_B04_MEMBER_SUPPORT_ONLY_NO_MODEL" if facts["support_gate_passed"]
              else "REJECTED_FIXED_B04_TWO_FIELD_BLOCK_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT")
    results = OUT / "results"
    results.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(results / (name + ".parquet"), index=False)
        table.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    write_json(OUT / "summary.json", {"study": "510300_POINT_B04_INFORMATION_INTAKE_V1",
        "completed_at": now(), "status": status, "fixed_operationalizations": 1,
        "frozen_sources_unchanged": count, **facts, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0, "new_market_bars": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "只裁决B04固定两列直接追加原八项的完整成员支持，阶段外未知不填零；未检验整个冲击修复机制的收益。"})
    require(check() == count, "B04运行后冻结来源改变。")
    return {"状态": status, "原自然成员": len(tables[1]), "可用成员": facts["supported_natural_rows"],
            "完整原可用月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        saved = pd.read_parquet(OUT / "results" / (name + ".parquet"))
        pd.testing.assert_frame_equal(saved, rebuilt, check_exact=True)
    summary = read_json(OUT / "summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "B04保存支持指标复算不同。")
    receipt = {"at": now(), "status": "PASS_SAVED_B04_SHOCK_FIELDS_AND_ORIGINAL_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "daily_rows": len(tables[0]), "natural_member_rows": len(tables[1]),
        "monthly_member_rows": len(tables[2]), "tables_and_support_exactly_equal": True,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "冲击/阶段内波动比/现金低点和原成员数量复算，不是收益验证或独立验证。"}
    write_json(OUT / "saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="B04冲击后固定两列成员支持准入，不运行策略")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
