"""D02固定负隔夜缺口吸收比例及缺口大小的原成员准入，不拟合模型。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_d02_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FIELD = "negative_gap_absorption"
GAP = "signed_overnight_gap"
FIELDS = [FIELD, GAP]
TICK = 0.001
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
PRICE_COLUMNS = ["date", "symbol", "open", "close", "previous_close", "dividend",
                 "total_simple", "total_log", "wealth"]
TABLES = ("日线D02事前字段", "原自然成员D02支持", "原月度成熟成员D02支持")


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
    require(data.symbol.eq("510300.SH").all(), "D02标的身份不同。")
    require(len(data) > 0, "D02市场快照为空。")
    opens = data.open.to_numpy(float)
    require(np.isfinite(opens).all() and (opens > 0).all(), "D02开盘缺失或非正。")
    close, previous, cash, simple, logs, wealth = data[[
        "close", "previous_close", "dividend", "total_simple", "total_log", "wealth"]].to_numpy(float).T
    require(np.isfinite(close).all() and (close > 0).all(), "D02收盘缺失或非正。")
    require(np.isfinite(cash).all() and (cash >= 0).all(), "D02现金除息事实缺失或为负。")
    require(pd.isna(previous[0]) and pd.isna(simple[0]) and pd.isna(logs[0]),
            "D02首个市场点没有前一区间，不用初始归一值填收益。")
    np.testing.assert_allclose(previous[1:], close[:-1], atol=1e-12, rtol=0)
    expected = (close[1:]+cash[1:])/previous[1:]-1
    require(np.isfinite(simple[1:]).all() and np.isfinite(logs[1:]).all(), "D02非初始经济收益缺失。")
    np.testing.assert_allclose(simple[1:], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(logs[1:], np.log1p(expected), atol=1e-12, rtol=0)
    np.testing.assert_allclose(wealth, np.r_[1., np.cumprod(1+expected)], atol=1e-12, rtol=1e-12)


def quote_units(values, name, positive):
    """按既有0.001价格/现金契约核验整数单位；不修复不合格数值。"""
    raw = np.asarray(values, dtype=float)
    valid = np.isfinite(raw) & ((raw > 0) if positive else (raw >= 0))
    scaled = raw[valid] / TICK
    require((np.abs(scaled) <= 1e15).all(), "D02" + name + "超出整数报价安全范围。")
    nearest = np.rint(scaled)
    require((np.abs(scaled-nearest) <= 1e-7).all(), "D02" + name + "不符合冻结0.001单位。")
    units = np.zeros(len(raw), dtype=np.int64)
    units[valid] = nearest.astype(np.int64)
    return units, valid


def build_field(data):
    """严格负隔夜才定义吸收比；当日现金先分配隔夜，日内不重复分红。"""
    require(set(PRICE_COLUMNS).issubset(data.columns), "D02源字段缺失。")
    require(data.symbol.eq("510300.SH").all(), "D02标的必须为510300.SH。")
    dates = pd.DatetimeIndex(data.date).astype("datetime64[ns]")
    require(len(data) > 0 and not dates.hasnans and dates.is_unique and dates.is_monotonic_increasing
            and (dates.dayofweek <= 4).all(), "D02市场日期必须唯一递增且为工作日。")
    opens, ov = quote_units(data.open.to_numpy(), "开盘", True)
    closes, cv = quote_units(data.close.to_numpy(), "收盘", True)
    previous, pv = quote_units(data.previous_close.to_numpy(), "前收盘", True)
    cash, dv = quote_units(data.dividend.to_numpy(), "当日已除息现金", False)
    valid = ov & cv & pv & dv
    valid[0] = False
    night_delta = opens + cash - previous
    day_delta = closes - opens
    night = np.full(len(data), np.nan)
    intra = np.full(len(data), np.nan)
    total = np.full(len(data), np.nan)
    idx = np.flatnonzero(valid)
    night[idx] = np.log1p(night_delta[idx] / previous[idx])
    intra[idx] = np.log1p(day_delta[idx] / (opens[idx] + cash[idx]))
    total[idx] = np.log1p((closes[idx] + cash[idx] - previous[idx]) / previous[idx])
    np.testing.assert_allclose(night[idx]+intra[idx], total[idx], atol=1e-12, rtol=0)
    available = valid & (night_delta < 0)
    value = np.full(len(data), np.nan)
    value[available] = np.maximum(intra[available], 0.) / np.abs(night[available])
    require(np.isfinite(value[available]).all(), "D02吸收比非有限，不替代异常值。")
    status = np.full(len(data), "NO_VIEW_MISSING_RAW_PRICE_OR_CASH_SOURCE", dtype=object)
    status[valid & (night_delta >= 0)] = "NO_VIEW_NONNEGATIVE_OVERNIGHT_GAP"
    status[available] = "D02_NEGATIVE_GAP_ABSORPTION_AVAILABLE"
    status[0] = "NO_VIEW_INITIAL_MARKET_POINT_HAS_NO_OVERNIGHT"
    return pd.DataFrame({"date": dates, GAP: night, "cash_inclusive_intraday_log_return": intra,
        "reconstructed_total_log_return": total, FIELD: value, "field_status": status,
        "valid_daily_source": valid, "overnight_delta_units": np.where(valid, night_delta, np.nan),
        "intraday_delta_units": np.where(valid, day_delta, np.nan),
        "earliest_source_index": np.arange(len(data))-1, "latest_source_index": np.arange(len(data))})


def member_support(samples, records, field):
    """保留每个原训练成员；负隔夜内真零有限，正/零隔夜阶段外未知不补零。"""
    require(set(METADATA).issubset(samples.columns), "D02原样本元数据缺失。")
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "D02原点越界。")
    left = pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]")
    right = pd.to_datetime(field.date.iloc[indexes]).to_numpy(dtype="datetime64[ns]")
    require(np.array_equal(left, right), "D02原点日期与冻结日线不一致。")
    values, states = field[FIELDS].to_numpy(float), field.field_status.to_numpy(str)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"], "D02原月周期清单重复或变化。")
        require(len(selected) == record["training_rows"], "D02原训练成员数量变化。")
        require(selected.cycle_id.nunique() == len(ids), "D02原训练周期有缺失成员。")
        require((selected.exit_index <= record["fit_index"]).all(), "D02读取尚未成熟周期。")
        require((pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "D02原成员在拟合日期尚未成熟。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx]).all(axis=1)
        counts = pd.Series(states[idx][~finite]).value_counts().to_dict()
        rows.append({
            "fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "d02_available_rows": int(finite.sum()), "d02_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in counts.items()},
                                               ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)


def source_paths():
    prior = read_json(OUT / "prior_and_source_review.json")
    paths = [Path(__file__), ROOT / "tests/test_point_d02_information_intake_v1.py",
             OUT / "prior_and_source_review.json", OUT / "tests_receipt.json", DATA, SAMPLES, MODELS]
    paths += [ROOT / item["path"] for item in prior["direct_sources"]]
    return sorted(set(paths), key=lambda p: p.as_posix())


def freeze():
    tests = read_json(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 7, "D02七项必要测试尚未通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_d02_information_intake_v1.py"),
            "D02代码在测试后变化。")
    prior = read_json(OUT / "prior_and_source_review.json")
    require(prior["conditional_purpose_and_source_clock_bound"], "D02用途/负缺口/现金时钟未绑定。")
    protocol = {"study": "510300_POINT_D02_INFORMATION_INTAKE_V1", "frozen_at": now(),
        "purpose": "收益拟合前检验固定D02两列直接追加原八项的完整成员支持，不读取未来收益标签。",
        "registered_definition": prior["registered_definition"], "field_binding": prior["field_binding"],
        "source_clock": prior["source_clock"], "missing_rule": prior["missing_rule"],
        "support_gate": prior["support_gate"], "no_rescue": prior["no_rescue"],
        "fixed_operationalizations": 1, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False}
    paths = source_paths()
    for source in prior["direct_sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "D02事前来源改变：" + source["path"])
    sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in paths]
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    return {"状态": "D02负缺口两列操作定义及来源已冻结", "来源数": len(sources), "新增拟合": 0}


def check():
    frozen = read_json(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "D02协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "D02冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def derive():
    data = pd.read_parquet(DATA, columns=PRICE_COLUMNS)
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "D02原快照数量改变。")
    validate_source_clock(data)
    field = build_field(data)
    np.testing.assert_allclose(field.reconstructed_total_log_return.iloc[1:], data.total_log.iloc[1:], atol=1e-12, rtol=0)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "D02原可用模型数改变。")
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
            "available_natural_rows": int(np.isfinite(part[FIELDS].to_numpy(float)).all(axis=1).sum()),
            "original_mature_months": len(months),
            "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
            "minimum_missing_original_training_rows": int(months.d02_missing_rows.min()) if len(months) else None,
        })
    accepted = bool(np.isfinite(natural[FIELDS].to_numpy(float)).all() and ready.all_original_members_supported.all())
    facts = {
        "support_gate_passed": accepted, "daily_rows": len(field),
        "finite_daily_field_rows": int(field[FIELD].notna().sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "nonnegative_gap_daily_rows": int(field.field_status.eq("NO_VIEW_NONNEGATIVE_OVERNIGHT_GAP").sum()),
        "exact_zero_gap_daily_rows": int(field[GAP].eq(0).sum()),
        "positive_gap_daily_rows": int(field[GAP].gt(0).sum()),
        "negative_gap_daily_rows": int(field[GAP].lt(0).sum()),
        "absorption_above_one_daily_rows": int(field[FIELD].gt(1).sum()),
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
    status = ("ADMITTED_FIXED_D02_MEMBER_SUPPORT_ONLY_NO_MODEL" if facts["support_gate_passed"]
              else "REJECTED_FIXED_D02_TWO_FIELD_BLOCK_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT")
    results = OUT / "results"
    results.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(results / (name + ".parquet"), index=False)
        table.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {
        "study": "510300_POINT_D02_INFORMATION_INTAKE_V1", "completed_at": now(), "status": status,
        "fixed_operationalizations": 1, "frozen_sources_unchanged": count, **facts,
        "empirical_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "new_market_bars": 0, "financial_metrics": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "只裁决固定D02吸收与缺口两列的原成员支持，不检验整个缺口修复机制收益；阶段外未知不补零或删行。",
    }
    write_json(OUT / "summary.json", summary)
    require(check() == count, "D02运行后冻结来源改变。")
    return {"状态": status, "原自然成员": len(tables[1]), "可用成员": facts["supported_natural_rows"],
            "完整原可用月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        saved = pd.read_parquet(OUT / "results" / (name + ".parquet"))
        pd.testing.assert_frame_equal(saved, rebuilt, check_exact=True)
    summary = read_json(OUT / "summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "D02保存支持指标复算不同。")
    receipt = {
        "at": now(), "status": "PASS_SAVED_D02_GAP_ABSORPTION_FIELDS_AND_ORIGINAL_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "daily_rows": len(tables[0]),
        "natural_member_rows": len(tables[1]), "monthly_member_rows": len(tables[2]),
        "tables_and_support_exactly_equal": True, "new_model_fits": 0,
        "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "固定负缺口现金分解及原成员数量复算，不是收益验证或独立验证。",
    }
    write_json(OUT / "saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="D02固定负隔夜缺口吸收两列数量准入，不运行策略")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
