"""A02突破后保留率的固定时钟和原成员数量准入，不拟合模型。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_a02_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FIELD = "breakout_three_day_retention_ratio"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
PRICE_COLUMNS = ["date", "symbol", "close", "high", "low", "dividend"]
TABLES = ("日线A02事前字段", "原自然成员A02支持", "原月度成熟成员A02支持")


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


def build_field(data):
    """先完成严格过去突破的当日观察，再登记今天的新突破。"""
    require(set(PRICE_COLUMNS).issubset(data.columns), "A02价格字段缺失。")
    dates = pd.to_datetime(data.date).astype("datetime64[ns]")
    require(dates.notna().all() and dates.is_monotonic_increasing and dates.is_unique,
            "A02日期缺失、重复或乱序，禁止删日期。")
    require(data.symbol.eq("510300.SH").all(), "A02观察对象改变。")
    close, high, low, dividend = [data[name].to_numpy(float) for name in PRICE_COLUMNS[2:]]
    require(np.isfinite(close).all() and np.isfinite(high).all() and np.isfinite(low).all()
            and (low > 0).all() and (high >= close).all() and (close >= low).all(),
            "A02原OHLC无效，禁止用有限行替代原日线。")
    require(np.isfinite(dividend).all() and (dividend >= 0).all(), "A02现金分红缺失或为负。")
    cash = np.cumsum(dividend)
    price, adjusted_high, adjusted_low = close + cash, high + cash, low + cash
    previous = np.r_[np.nan, price[:-1]]
    tr = np.maximum.reduce([adjusted_high - adjusted_low,
                            np.abs(adjusted_high - previous), np.abs(adjusted_low - previous)])
    atr = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    upper = pd.Series(adjusted_high).shift(1).rolling(20, min_periods=20).max().to_numpy()
    breakout = np.isfinite(upper) & (price > upper)
    latest = None
    rows = []
    for t in range(len(data)):
        row = {
            "date": dates.iloc[t], "economic_close": price[t], "economic_high": adjusted_high[t],
            "economic_low": adjusted_low[t], "prior20_high_today": upper[t],
            "previous_atr20_today": atr[t - 1] if t else np.nan,
            "breakout_confirmed_today": bool(breakout[t]), "latest_prior_breakout_index": latest,
            "latest_prior_breakout_date": dates.iloc[latest] if latest is not None else pd.NaT,
            "phase_age": t - latest if latest is not None else None,
            "frozen_breakout_upper": upper[latest] if latest is not None else np.nan,
            "frozen_previous_atr20": atr[latest - 1] if latest is not None and latest else np.nan,
            "observed_day_count": None, "passing_day_count": None,
            FIELD: np.nan, "field_status": "NO_VIEW_NO_PRIOR_BREAKOUT",
        }
        if latest is not None:
            age = t - latest
            if age > 3:
                row["field_status"] = "NO_VIEW_THREE_DAY_PHASE_EXPIRED"
            elif not np.isfinite(atr[latest - 1]) or atr[latest - 1] <= 0:
                row["field_status"] = "NO_VIEW_INCOMPLETE_OR_NONPOSITIVE_FROZEN_ATR"
            else:
                require(1 <= age <= 3, "A02使用了尚未完成的观察日。")
                floor = upper[latest] - 0.5 * atr[latest - 1]
                passed = ((adjusted_low[latest + 1:t + 1] >= floor)
                          & (price[latest + 1:t + 1] >= upper[latest]))
                require(len(passed) == age, "A02分母不是全部已完成观察日。")
                row.update(observed_day_count=age, passing_day_count=int(passed.sum()),
                           field_status="AVAILABLE_COMPLETED_POST_BREAKOUT_PHASE")
                row[FIELD] = float(passed.sum() / age)
        rows.append(row)
        if breakout[t]:
            latest = t
    return pd.DataFrame(rows)


def member_support(samples, records, field):
    """保留每个原训练成员；真实比值零有限，未知和过期不补零。"""
    require(set(METADATA).issubset(samples.columns), "A02原样本元数据缺失。")
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "A02原点越界。")
    left = pd.to_datetime(samples.origin).to_numpy(dtype="datetime64[ns]")
    right = pd.to_datetime(field.date.iloc[indexes]).to_numpy(dtype="datetime64[ns]")
    require(np.array_equal(left, right), "A02原点日期与冻结日线不一致。")
    values, states = field[FIELD].to_numpy(float), field.field_status.to_numpy(str)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"], "A02原月周期清单重复或变化。")
        require(len(selected) == record["training_rows"], "A02原训练成员数量变化。")
        require(selected.cycle_id.nunique() == len(ids), "A02原训练周期有缺失成员。")
        require((selected.exit_index <= record["fit_index"]).all(), "A02读取尚未成熟周期。")
        require((pd.to_datetime(selected.mature_date) <= pd.Timestamp(record["fit_origin"])).all(),
                "A02原成员在拟合日期尚未成熟。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx])
        counts = pd.Series(states[idx][~finite]).value_counts().to_dict()
        rows.append({
            "fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "a02_available_rows": int(finite.sum()), "a02_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in counts.items()},
                                               ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)


def source_paths():
    names = [
        "research/point_a02_information_intake_v1.py", "tests/test_point_a02_information_intake_v1.py",
        "reports/research/510300_point_a02_information_intake_v1/prior_and_source_review.json",
        "reports/research/510300_point_a02_information_intake_v1/tests_receipt.json",
        "reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json",
        "reports/research/510300_factor96_program_v1/factor_progress.json",
        "reports/research/510300_factor96_program_v1/strategy_progress.json",
        "reports/research/510300_point_next_information_intake_20261002/A02_candidate_intake_plan.json",
        "reports/research/510300_point_next_information_intake_20261002/G1_prior_definition_routing_20261002.json",
        "research/breakout_retest_entry_v1.py", "config/510300_breakout_retest_entry_v1.json",
        "docs/510300_BREAKOUT_RETEST_ENTRY_V1.md", "reports/research/510300_breakout_retest_entry_v1/result.json",
        "research/simple_price_entry_exit_v1.py", "docs/510300_SIMPLE_PRICE_ENTRY_EXIT_V1.md",
        "research/point_a04_exit_prediction_v1.py", "reports/research/510300_point_a04_exit_prediction_v1/protocol.json",
        "reports/research/510300_point_a04_exit_prediction_v1/prediction_summary.json",
        "research/point_c04_information_intake_v1.py", "reports/research/510300_point_c04_information_intake_v1/protocol.json",
        "reports/research/510300_point_c04_information_intake_v1/summary.json",
        "reports/research/510300_point_n04_source_intake_v1_0_1/summary.json",
        "reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json",
        "reports/research/510300_point_current_observation_20261001/protocol.json",
        "reports/research/510300_point_current_observation_20261001/freeze.json",
    ]
    return [ROOT / name for name in names] + [DATA, SAMPLES, MODELS]


def freeze():
    tests = read_json(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 5, "A02必要测试未全部通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_a02_information_intake_v1.py"),
            "A02代码在测试后变化。")
    prior = read_json(OUT / "prior_and_source_review.json")
    protocol = {
        "study": "510300_POINT_A02_INFORMATION_INTAKE_V1", "frozen_at": now(),
        "purpose": prior["purpose"], "registered_definition": prior["registered_card"]["definition"],
        "price_clock": prior["price_clock"], "ATR_clock": prior["ATR_clock"],
        "phase_binding": prior["phase_binding"], "passing_rule": prior["passing_rule"],
        "missing_rule": prior["missing_rule"], "support_gate": prior["support_gate"],
        "fixed_operationalizations": 1, "empirical_strategy_configurations": 0,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "source_scope": "仅原冻结日线和原成员元数据，历史全部DEVELOPMENT_CALIBRATION。",
        "no_rescue": "不改20/3/0.5、重叠更新、价格、缺失赋值、原成员或成熟门营救；原T01及旧回踩终态保留。",
        "orders_authorized": False, "goal_achieved": False,
    }
    paths = source_paths()
    require(all(path.is_file() for path in paths), "A02冻结来源缺失。")
    sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in paths]
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    return {"状态": "A02操作定义及来源已冻结", "来源数": len(sources), "新增拟合": 0}


def check():
    frozen = read_json(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "A02协议改变。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "A02冻结来源改变：" + source["path"])
    return len(frozen["sources"])


def derive():
    data = pd.read_parquet(DATA, columns=PRICE_COLUMNS)
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "A02原快照数量改变。")
    field = build_field(data)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "A02原可用模型数改变。")
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
            "minimum_missing_original_training_rows": int(months.a02_missing_rows.min()) if len(months) else None,
        })
    accepted = bool(natural[FIELD].notna().all() and ready.all_original_members_supported.all())
    facts = {
        "support_gate_passed": accepted, "daily_rows": len(field),
        "finite_daily_field_rows": int(field[FIELD].notna().sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "breakout_confirmations": int(field.breakout_confirmed_today.sum()),
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
    status = ("ADMITTED_FIXED_A02_MEMBER_SUPPORT_ONLY_NO_MODEL" if facts["support_gate_passed"]
              else "REJECTED_FIXED_A02_NINTH_FIELD_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT")
    results = OUT / "results"
    results.mkdir(exist_ok=False)
    for name, table in zip(TABLES, tables):
        table.to_parquet(results / (name + ".parquet"), index=False)
        table.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {
        "study": "510300_POINT_A02_INFORMATION_INTAKE_V1", "completed_at": now(), "status": status,
        "fixed_operationalizations": 1, "frozen_sources_unchanged": count, **facts,
        "empirical_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
        "new_strategy_accounts": 0, "new_market_bars": 0, "financial_metrics": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "只裁决固定A02直接第九项的原成员支持，不检验整个突破机制收益；缺失不补零或删行。",
    }
    write_json(OUT / "summary.json", summary)
    require(check() == count, "A02运行后冻结来源改变。")
    return {"状态": status, "原自然成员": len(tables[1]), "可用成员": facts["supported_natural_rows"],
            "完整原可用月": facts["fully_supported_original_mature_months"], "新增拟合": 0}


def verify():
    count = check()
    tables, facts = derive()
    for name, rebuilt in zip(TABLES, tables):
        saved = pd.read_parquet(OUT / "results" / (name + ".parquet"))
        pd.testing.assert_frame_equal(saved, rebuilt, check_exact=True)
    summary = read_json(OUT / "summary.json")
    require(all(summary[key] == value for key, value in facts.items()), "A02保存支持指标复算不同。")
    receipt = {
        "at": now(), "status": "PASS_SAVED_A02_CLOCK_FIELDS_AND_ORIGINAL_MEMBER_SUPPORT_RECOMPUTATION",
        "frozen_sources_unchanged": count, "daily_rows": len(tables[0]),
        "natural_member_rows": len(tables[1]), "monthly_member_rows": len(tables[2]),
        "tables_and_support_exactly_equal": True, "new_model_fits": 0,
        "new_return_labels": 0, "new_strategy_accounts": 0,
        "scope": "原阶段字段和原成员数量复算，不是收益验证或独立验证。",
    }
    write_json(OUT / "saved_output_recomputation_receipt.json", receipt)
    return {"状态": receipt["status"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="A02固定阶段数量准入，不运行策略")
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run, "verify": verify}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
