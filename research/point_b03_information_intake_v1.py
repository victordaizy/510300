"""B03冻结前低收复用时及再跌破计数的原成员准入；不读取收益标签或拟合。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
from fractions import Fraction

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_b03_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FIELD = "first_reclaim_delay"
COUNT = "post_reclaim_rebreak_count"
FIELDS = [FIELD, COUNT]
TICK = 0.001
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "previous_close",
                 "dividend", "total_simple", "total_log", "wealth"]
TABLES = ("日线B03事前字段", "原自然成员B03支持", "原月度成熟成员B03支持")


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
    require(len(data) > 0 and data.symbol.eq("510300.SH").all(), "B03标的或市场快照不同。")
    prices = data[["open", "high", "low", "close"]].to_numpy(float)
    require(np.isfinite(prices).all() and (prices > 0).all(), "B03价格缺失或非正。")
    opens, highs, lows, close = prices.T
    require(((lows <= opens) & (lows <= close) & (highs >= opens) & (highs >= close)).all(),
            "B03原OHLC关系不合法。")
    previous, cash, simple, logs, wealth = data[[
        "previous_close", "dividend", "total_simple", "total_log", "wealth"]].to_numpy(float).T
    require(np.isfinite(cash).all() and (cash >= 0).all(), "B03当日已除息现金缺失或为负。")
    require(pd.isna(previous[0]) and pd.isna(simple[0]) and pd.isna(logs[0]),
            "B03首个市场点没有前一区间，不填初始收益。")
    np.testing.assert_allclose(previous[1:], close[:-1], atol=1e-12, rtol=0)
    expected = (close[1:] + cash[1:]) / close[:-1] - 1
    require(np.isfinite(simple[1:]).all() and np.isfinite(logs[1:]).all(), "B03非初始收益缺失。")
    np.testing.assert_allclose(simple[1:], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(logs[1:], np.log1p(expected), atol=1e-12, rtol=0)
    np.testing.assert_allclose(wealth, np.r_[1., np.cumprod(1 + expected)], atol=1e-12, rtol=1e-12)


def quote_units(values, name, positive):
    """只核验既有0.001报价/现金单位，不修复离格数值。"""
    raw = np.asarray(values, dtype=float)
    valid = np.isfinite(raw) & ((raw > 0) if positive else (raw >= 0))
    scaled = raw[valid] / TICK
    require((np.abs(scaled) <= 1e15).all(), "B03" + name + "超出整数报价安全范围。")
    nearest = np.rint(scaled)
    require((np.abs(scaled - nearest) <= 1e-7).all(), "B03" + name + "不符合冻结0.001单位。")
    units = np.zeros(len(raw), dtype=np.int64)
    units[valid] = nearest.astype(np.int64)
    return units, valid


def build_field(data):
    """每个严格前20日新低冻结事件；收复前未知，新破位重置，未成功事件不删。"""
    require(set(PRICE_COLUMNS).issubset(data.columns), "B03源字段缺失。")
    require(data.symbol.eq("510300.SH").all(), "B03标的必须为510300.SH。")
    dates = pd.DatetimeIndex(data.date).astype("datetime64[ns]")
    n = len(data)
    require(n > 0 and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
            and (dates.dayofweek <= 4).all(), "B03市场日期必须唯一递增且为工作日。")
    close, cv = quote_units(data.close, "收盘", True)
    low, lv = quote_units(data.low, "最低价", True)
    cash, dv = quote_units(data.dividend, "当日已除息现金", False)
    delay, counts = np.full(n, np.nan), np.full(n, np.nan)
    states = np.full(n, "NO_VIEW_NO_KNOWN_FROZEN_BREAK", dtype=object)
    raw_break = np.zeros(n, dtype=bool)
    origins = np.full(n, -1, dtype=np.int64)
    low_origins = np.full(n, -1, dtype=np.int64)
    reclaim_origins = np.full(n, -1, dtype=np.int64)
    event_ids = np.full(n, -1, dtype=np.int64)
    earliest = np.full(n, -1, dtype=np.int64)
    ages = np.full(n, np.nan)
    mapped_lows = [None] * n
    wealth, segment_start, active, next_event = None, -1, None, 0
    for i in range(n):
        if not (cv[i] and dv[i]):
            wealth, active, segment_start = None, None, -1
            states[i] = "NO_VIEW_RAW_CLOSE_OR_CASH_MISSING"
            continue
        if wealth is None:
            wealth, segment_start = Fraction(1), i
        else:
            wealth *= Fraction(int(close[i])+int(cash[i]), int(close[i-1]))
        if not lv[i]:
            active = None
            states[i] = "NO_VIEW_RAW_LOW_MISSING"
            continue
        mapped_lows[i] = wealth * Fraction(int(low[i])+int(cash[i]), int(close[i])+int(cash[i]))
        if i-20 < segment_start or any(value is None for value in mapped_lows[i-20:i]):
            active = None
            states[i] = "NO_VIEW_TWENTY_PRIOR_LOW_SOURCE_NOT_READY"
            continue
        # Fraction财富/低点只作相对比较；合法0.001源的相等边界无需epsilon。
        k = min(range(i-20, i), key=lambda j: mapped_lows[j])
        raw_break[i] = mapped_lows[i] < mapped_lows[k]
        if raw_break[i]:
            active = {"event": next_event, "e": i, "k": k,
                      "threshold": mapped_lows[k], "reclaim": None,
                      "count": 0, "was_below": False}
            next_event += 1
        if active is None:
            continue
        e = active["e"]
        origins[i], low_origins[i], event_ids[i] = e, active["k"], active["event"]
        ages[i], earliest[i] = i-e, e-20
        below = wealth < active["threshold"]
        above = wealth > active["threshold"]
        if active["reclaim"] is None:
            if above:
                active["reclaim"] = i
        elif below and not active["was_below"]:
            active["count"] += 1
        active["was_below"] = below
        if active["reclaim"] is None:
            states[i] = "NO_VIEW_FIRST_RECLAIM_NOT_YET_KNOWN"
            continue
        reclaim_origins[i] = active["reclaim"]
        delay[i], counts[i] = active["reclaim"]-e, active["count"]
        states[i] = "B03_FIRST_RECLAIM_DELAY_AND_REBREAK_COUNT_AVAILABLE"
    return pd.DataFrame({"date": dates, FIELD: delay, COUNT: counts, "field_status": states,
        "raw_break20": raw_break, "event_id": event_ids, "break_origin_index": origins,
        "frozen_low_origin_index": low_origins, "first_reclaim_index": reclaim_origins,
        "event_age": ages, "earliest_source_index": earliest,
        "latest_source_index": np.arange(n)})


def member_support(samples, records, field):
    """完整保留原成员；阶段外未知不填零、不删除、不另开条件样本。"""
    require(set(METADATA).issubset(samples.columns), "B03原样本元数据缺失。")
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "B03原点越界。")
    left = pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]")
    right = pd.to_datetime(field.date.iloc[indexes]).to_numpy(dtype="datetime64[ns]")
    require(np.array_equal(left, right), "B03原点日期与冻结日线不一致。")
    values, states = field[FIELDS].to_numpy(float), field.field_status.to_numpy(str)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"], "B03原月周期清单重复或变化。")
        require(len(selected) == record["training_rows"], "B03原训练成员数量变化。")
        require(selected.cycle_id.nunique() == len(ids), "B03原训练周期有缺失成员。")
        require((selected.exit_index <= record["fit_index"]).all(), "B03读取尚未成熟周期。")
        require((pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "B03原成员在拟合日期尚未成熟。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx]).all(axis=1)
        counts = pd.Series(states[idx][~finite]).value_counts().to_dict()
        rows.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "b03_available_rows": int(finite.sum()), "b03_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in counts.items()},
                                               ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)


def source_paths():
    prior = read_json(OUT / "prior_and_source_review.json")
    paths = [Path(__file__), ROOT / "tests/test_point_b03_information_intake_v1.py",
             OUT / "prior_and_source_review.json", OUT / "tests_receipt.json", DATA, SAMPLES, MODELS]
    paths += [ROOT / item["path"] for item in prior["direct_sources"]]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    tests = read_json(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 7, "B03七项必要测试尚未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_b03_information_intake_v1.py"),
            "B03代码在测试后变化。")
    prior = read_json(OUT / "prior_and_source_review.json")
    require(prior["conditional_purpose_and_source_clock_bound"], "B03用途/新低事件/首次收复和计数时钟未绑定。")
    protocol = {"study": "510300_POINT_B03_INFORMATION_INTAKE_V1", "frozen_at": now(),
        "purpose": "拟合前检验固定B03两列直接追加原八项的完整成员支持，不读取未来收益标签。",
        "registered_definition": prior["registered_definition"], "field_binding": prior["field_binding"],
        "source_clock": prior["source_clock"], "missing_rule": prior["missing_rule"],
        "support_gate": prior["support_gate"], "no_rescue": prior["no_rescue"],
        "fixed_operationalizations": 1, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False}
    for source in prior["direct_sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "B03事前来源改变：" + source["path"])
    sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in source_paths()]
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    return {"状态": "B03冻结低点收复/计数两列操作定义和来源已冻结", "来源数": len(sources), "新增拟合": 0}


def check():
    frozen = read_json(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "B03协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "B03冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def derive():
    data = pd.read_parquet(DATA, columns=PRICE_COLUMNS)
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "B03原快照数量改变。")
    validate_source_clock(data)
    field = build_field(data)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "B03原可用模型数改变。")
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
            "minimum_missing_original_training_rows": int(months.b03_missing_rows.min()) if len(months) else None})
    finite = np.isfinite(field[FIELDS].to_numpy(float)).all(axis=1)
    natural_finite = np.isfinite(natural[FIELDS].to_numpy(float)).all(axis=1)
    accepted = bool(natural_finite.all() and ready.all_original_members_supported.all())
    facts = {"support_gate_passed": accepted, "daily_rows": len(field),
        "finite_daily_field_rows": int(finite.sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "raw_break_events": int(field.raw_break20.sum()),
        "same_day_reclaim_event_rows": int((field.raw_break20 & field[FIELD].eq(0)).sum()),
        "genuine_zero_reclaim_delay_daily_rows": int(field[FIELD].eq(0).sum()),
        "genuine_zero_rebreak_count_daily_rows": int(field[COUNT].eq(0).sum()),
        "positive_rebreak_count_daily_rows": int(field[COUNT].gt(0).sum()),
        "unreclaimed_daily_rows": int(field.field_status.eq("NO_VIEW_FIRST_RECLAIM_NOT_YET_KNOWN").sum()),
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
    status = ("ADMITTED_FIXED_B03_MEMBER_SUPPORT_ONLY_NO_MODEL" if facts["support_gate_passed"]
              else "REJECTED_FIXED_B03_TWO_FIELD_BLOCK_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT")
    results = OUT / "results"
    results.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(results / (name + ".parquet"), index=False)
        table.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    write_json(OUT / "summary.json", {"study": "510300_POINT_B03_INFORMATION_INTAKE_V1",
        "completed_at": now(), "status": status, "fixed_operationalizations": 1,
        "frozen_sources_unchanged": count, **facts, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0, "new_market_bars": 0,
        "financial_metrics": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "只裁决B03固定首次收复用时/再破低计数两列直接追加原八项的完整成员支持，未收复未知不填零；未检验整个破位修复机制的收益。"})
    require(check() == count, "B03运行后冻结来源改变。")
    return {"状态": status, "原自然成员": len(tables[1]), "可用成员": facts["supported_natural_rows"],
            "完整原可用月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        saved = pd.read_parquet(OUT / "results" / (name + ".parquet"))
        pd.testing.assert_frame_equal(saved, rebuilt, check_exact=True)
    summary = read_json(OUT / "summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "B03保存支持指标复算不同。")
    receipt = {"at": now(), "status": "PASS_SAVED_B03_FROZEN_LOW_RECLAIM_FIELDS_AND_ORIGINAL_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "daily_rows": len(tables[0]), "natural_member_rows": len(tables[1]),
        "monthly_member_rows": len(tables[2]), "tables_and_support_exactly_equal": True,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "严格前低/现金首次收复及计数和原成员数量复算，不是收益验证或独立验证。"}
    write_json(OUT / "saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="B03冻结前低收复/计数两列成员支持准入，不运行策略")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
