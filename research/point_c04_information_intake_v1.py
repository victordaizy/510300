"""C04回调期成交额收缩的固定定义与原成员数量准入，不拟合模型。"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_c04_information_intake_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
DATA = CURRENT / "inputs/candidate_features.parquet"
SAMPLES = CURRENT / "results/training_reference/samples.parquet"
MODELS = CURRENT / "inputs/within_models.json"
FIELD = "confirmed_pullback_amount_ratio"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]


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
    """只向前确认阶段；未定义、窗口不足和坏输入均保留未知。"""
    required = ["date", "close", "high", "low", "dividend", "amount", "volume", "symbol", "amount_unit", "volume_unit"]
    require(set(required).issubset(data.columns), "C04日线缺少必要字段。")
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 0 and not dates.hasnans and dates.is_monotonic_increasing and not dates.has_duplicates,
            "C04日期必须完整、唯一并递增。")
    require(data.symbol.eq("510300.SH").all(), "C04仅接纳510300.SH。")
    require(data.amount_unit.eq("CNY").all() and data.volume_unit.eq("share").all(),
            "C04成交额须为人民币元，成交量须为份。")
    close, high, low, dividend, amount, volume = [data[c].to_numpy(float) for c in required[1:7]]
    require(np.isfinite(close).all() and np.isfinite(high).all() and np.isfinite(low).all()
            and (close > 0).all() and (low > 0).all() and (high >= low).all(), "C04价格无效，禁止跳过日期。")
    require(np.isfinite(dividend).all() and (dividend >= 0).all(), "C04分红无效，禁止使用未来或缺失分红。")
    cash = np.cumsum(dividend)
    price, adjusted_high, adjusted_low = close + cash, high + cash, low + cash
    previous = np.r_[np.nan, price[:-1]]
    tr = np.maximum.reduce([adjusted_high - adjusted_low, np.abs(adjusted_high - previous), np.abs(adjusted_low - previous)])
    atr = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    valid_amount = np.isfinite(amount) & (amount >= 0) & np.isfinite(volume) & (volume >= 0)
    valid_amount &= (volume > 0) | (amount == 0)
    latest_low, pair, phase_start, broken = None, None, None, False
    rows = []
    for t in range(len(data)):
        low_today, high_today, pair_changed, started_today = False, False, False, False
        if t >= 4:
            center = t - 2
            neighbors = price[[center - 2, center - 1, center + 1, center + 2]]
            low_today, high_today = bool((price[center] < neighbors).all()), bool((price[center] > neighbors).all())
            if low_today:
                latest_low = center
            if high_today and latest_low is not None and latest_low < center and price[center] > price[latest_low]:
                pair = (latest_low, center)
                phase_start, broken, pair_changed = None, False, True
        value, baseline, phase_mean, depth_atr = np.nan, np.nan, np.nan, np.nan
        if pair is None:
            status = "NO_VIEW_NO_CONFIRMED_UP_SWING"
        else:
            a, b = pair
            if broken:
                status = "NO_VIEW_UP_SWING_INVALIDATED"
            elif price[t] <= price[a]:
                broken, phase_start = True, None
                status = "NO_VIEW_UP_SWING_INVALIDATED"
            elif price[t] >= price[b]:
                phase_start = None
                status = "NO_VIEW_NOT_IN_PULLBACK"
            else:
                if phase_start is None:
                    phase_start, started_today = t, True
                e = phase_start
                if e < 5:
                    status = "NO_VIEW_INCOMPLETE_PREVIOUS_FIVE_DAYS"
                elif not valid_amount[e - 5:e].all():
                    status = "NO_VIEW_INVALID_BASELINE_AMOUNT"
                else:
                    baseline = float(np.mean(amount[e - 5:e]))
                    if baseline <= 0:
                        status = "NO_VIEW_NONPOSITIVE_BASELINE_AMOUNT"
                    elif not valid_amount[e:t + 1].all():
                        status = "NO_VIEW_INVALID_PULLBACK_AMOUNT"
                    else:
                        phase_mean = float(np.mean(amount[e:t + 1]))
                        value = phase_mean / baseline
                        require(np.isfinite(value), "C04成交额比值非有限，禁止数值替代。")
                        status = "FIELD_AVAILABLE"
                if np.isfinite(atr[t]) and atr[t] > 0:
                    depth_atr = float((price[b] - price[t]) / atr[t])
        rows.append({
            "date": dates[t], "origin_index": t, "known_cash_adjusted_close": float(price[t]),
            FIELD: value, "field_status": status, "confirmed_low_today": low_today,
            "confirmed_high_today": high_today, "pair_changed_today": pair_changed,
            "phase_started_today": started_today, "latest_source_index": t,
            "low_index": pair[0] if pair else None, "high_index": pair[1] if pair else None,
            "low_confirmation_index": pair[0] + 2 if pair else None,
            "high_confirmation_index": pair[1] + 2 if pair else None,
            "phase_start_index": phase_start, "baseline_amount_cny": baseline,
            "pullback_mean_amount_cny": phase_mean, "pullback_depth_atr20_diagnostic_only": depth_atr,
        })
    return pd.DataFrame(rows)


def member_support(samples, records, field):
    """依据原月度成员清单计数，不读取目标、不删行或重新选周期。"""
    require(set(METADATA).issubset(samples.columns), "C04原样本元数据缺失。")
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(field)).all(), "C04原点索引越界。")
    require(np.array_equal(pd.to_datetime(samples.origin).to_numpy(), pd.to_datetime(field.date.iloc[indexes]).to_numpy()),
            "C04原点日期与冻结日线不一致。")
    values = field[FIELD].to_numpy(float)
    states = field.field_status.to_numpy(str)
    rows = []
    for record in records:
        ids = record["training_cycles"]
        selected = samples.loc[samples.cycle_id.isin(ids)]
        require(len(ids) == len(set(ids)) == record["training_cycle_count"], "C04原月度周期清单有重复或数量错误。")
        require(len(selected) == record["training_rows"], "C04原训练成员数量变化，禁止删行。")
        require(selected.cycle_id.nunique() == len(ids), "C04原训练周期存在缺失成员。")
        require((selected.exit_index <= record["fit_index"]).all(), "C04出现尚未成熟的训练周期。")
        idx = selected.origin_index.to_numpy(int)
        finite = np.isfinite(values[idx])
        counts = pd.Series(states[idx][~finite]).value_counts().to_dict()
        rows.append({
            "fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "original_model_available": isinstance(record["model"], dict),
            "original_training_cycles": len(ids), "original_training_rows": len(selected),
            "c04_available_rows": int(finite.sum()), "c04_missing_rows": int((~finite).sum()),
            "all_original_members_supported": bool(finite.all()),
            "missing_status_counts": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)


def source_paths():
    relative = [
        "research/point_c04_information_intake_v1.py", "tests/test_point_c04_information_intake_v1.py",
        "reports/research/510300_point_c04_information_intake_v1/prior_and_source_review.json",
        "reports/research/510300_point_c04_information_intake_v1/tests_receipt.json",
        "reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json",
        "reports/research/510300_factor96_program_v1/factor_progress.json",
        "reports/research/510300_factor96_program_v1/strategy_progress.json",
        "reports/research/510300_point_next_information_intake_20261002/C04_candidate_intake_plan.json",
        "reports/research/510300_point_a04_exit_prediction_v1/protocol.json",
        "reports/research/510300_point_a04_exit_prediction_v1/prediction_summary.json",
        "reports/research/510300_point_c06_exit_prediction_v1/protocol.json",
        "reports/research/510300_point_c06_exit_prediction_v1/prediction_summary.json",
        "research/daily_supply_test_v1.py", "research/force_pullback_inputs_v1.py",
        "config/510300_force_pullback_v1.json", "research/signed_volume_balance_inputs_v1.py",
        "research/signed_volume_within_inputs_v1.py", "config/510300_signed_volume_within_v1.json",
        "docs/510300_SIGNED_VOLUME_WITHIN_V1.md", "research/price_volume_coherence_inputs_v1.py",
        "config/510300_price_volume_coherence_v1.json",
    ]
    return [ROOT / name for name in relative] + [DATA, SAMPLES, MODELS]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    tests = read_json(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 6, "C04必要测试尚未全部通过。")
    require(tests["module_sha256"] == digest(Path(__file__))
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_c04_information_intake_v1.py"),
            "C04代码在测试后改变。")
    protocol = {
        "study": "510300_POINT_C04_INFORMATION_INTAKE_V1", "frozen_at": now(),
        "purpose": "单一C04操作定义的事前时钟及原模型第九项数量准入；不拟合、不读收益结果。",
        "price_clock": "原收盘加各自截至当日已除息现金累计；严格左右各2根日收盘，center=t-2，在t首次确认。",
        "up_swing": "确认高点b时配此前最新已确认低点a，a<b且正幅度才更新；不使用A04比例或T01阈值。",
        "pullback_phase": "已确认合法段的价格严格处于P[a]与P[b]之间，首次在当时可知的日t开启e=t，不追溯到b+1。",
        "phase_end": "当日价格>=P[b]结束；再进入区间重新e=t。价格<=P[a]使该段失效，须新的合法高点确认才能重新开始。新合法段在确认当日替换旧阶段。无期限或深度阈值。",
        "amount_definition": "mean(amount[e..t])/mean(amount[e-5..e-1])；原始成交额CNY，前五日窗口固定在e之前，不随t滚动；真实零额可进入均值，基准须正。",
        "missing": "非回调、无已确认段、段失效、缺失/负成交额或窗口不足均NO_VIEW；不填0，不删日期或原成员。",
        "background": "回撤/ATR20仅诊断列，不加入第二字段、不作为入场或深度过滤；ATR为20个完整TR简单均值。",
        "support_gate": "原1507自然状态字段全部有限，原142月度中的115成熟模型全部原有序训练成员有限；保留27个原无模型月份。任何不足均拒绝该直接第九项路线。",
        "source_scope": "仅已冻结510300日线，历史DEVELOPMENT_CALIBRATION，不新增行情或独立期间。",
        "no_rescue": "不换上涨确认、阶段起止、基准5日、缺失赋值、原成员或用途以营救本次固定定义；不恢复T01旧组合。",
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "orders_authorized": False, "goal_achieved": False,
    }
    paths = source_paths()
    require(all(p.is_file() for p in paths), "C04冻结来源缺失。")
    sources = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})
    return {"状态": "FROZEN_FIXED_C04_SUPPORT_INTAKE_ONLY", "来源数": len(sources)}


def check():
    frozen = read_json(OUT / "freeze.json")
    require(digest(OUT / "protocol.json") == frozen["protocol_sha256"], "C04协议冻结后变化。")
    for source in frozen["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "C04冻结来源变化：" + source["path"])
    return len(frozen["sources"])


def run():
    count = check()
    write_json(OUT / "ADMISSION_STARTED.json", {"at": now(), "new_model_fits": 0})
    data = pd.read_parquet(DATA)
    samples = pd.read_parquet(SAMPLES, columns=METADATA)
    records = read_json(MODELS)["models"]
    require(len(data) == 3488 and len(samples) == 1507 and len(records) == 142, "C04来源快照数量与冻结计划不同。")
    field = build_field(data)
    support = member_support(samples, records, field)
    ready = support.loc[support.original_model_available]
    require(len(ready) == 115, "C04原成熟模型数量改变。")
    natural = field.iloc[samples.origin_index.to_numpy(int)].reset_index(drop=True)
    natural.insert(1, "cycle_id", samples.cycle_id.to_numpy(int))
    field_full, monthly_full = bool(np.isfinite(natural[FIELD]).all()), bool(ready.all_original_members_supported.all())
    periods = []
    for period, start, end in [("2015_2019", "2015-01-05", "2019-12-31"), ("2020_2026", "2020-01-02", "2026-09-30")]:
        part = natural.loc[natural.date.between(start, end)]
        months = ready.loc[pd.to_datetime(ready.fit_origin).between(start, end)]
        periods.append({"period": period, "original_natural_rows": len(part), "available_natural_rows": int(part[FIELD].notna().sum()),
                        "original_mature_months": len(months), "fully_supported_mature_months": int(months.all_original_members_supported.sum()),
                        "minimum_missing_original_training_rows": int(months.c04_missing_rows.min()) if len(months) else None})
    accepted = field_full and monthly_full
    status = "ADMITTED_FIXED_C04_MEMBER_SUPPORT_ONLY_NO_MODEL" if accepted else "REJECTED_FIXED_C04_NINTH_FIELD_INCOMPLETE_ORIGINAL_MEMBER_SUPPORT"
    summary = {
        "study": "510300_POINT_C04_INFORMATION_INTAKE_V1", "completed_at": now(), "status": status,
        "fixed_operationalizations": 1, "support_gate_passed": accepted, "frozen_sources_unchanged": count,
        "daily_rows": len(field), "finite_daily_field_rows": int(field[FIELD].notna().sum()),
        "daily_status_counts": {str(k): int(v) for k, v in field.field_status.value_counts().items()},
        "confirmed_positive_pair_updates": int(field.pair_changed_today.sum()), "pullback_phase_starts": int(field.phase_started_today.sum()),
        "original_natural_rows": len(natural), "supported_natural_rows": int(natural[FIELD].notna().sum()),
        "original_monthly_records": len(support), "original_mature_months": len(ready),
        "fully_supported_original_mature_months": int(ready.all_original_members_supported.sum()),
        "original_no_model_months_preserved": len(support) - len(ready), "periods": periods,
        "label_columns_used": [], "training_metadata_columns_used": METADATA,
        "new_model_fits": 0, "new_return_labels": 0, "new_strategy_accounts": 0,
        "prediction_mse": "NOT_COMPUTED", "account_return_sharpe": "NOT_COMPUTED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "interpretation": "拒绝这一固定阶段字段直接作为原模型第九项，不等于整个成交收缩机制已做收益检验。",
    }
    results = OUT / "results"
    results.mkdir(exist_ok=True)
    for name, frame in [("日线C04阶段及成交额比", field), ("原自然成员支持", natural), ("原月度完整成员支持", support)]:
        frame.to_parquet(results / (name + ".parquet"), index=False)
        frame.to_csv(results / (name + ".csv"), index=False, encoding="utf-8-sig")
    check()
    write_json(OUT / "summary.json", summary)
    return {"状态": status, "原自然成员": len(natural), "可用自然成员": summary["supported_natural_rows"],
            "原成熟月份": len(ready), "完整支持月份": summary["fully_supported_original_mature_months"], "新增拟合": 0}


def main():
    parser = ArgumentParser(description="C04固定数量准入，不拟合模型或收益账户。")
    parser.add_argument("action", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    result = freeze() if args.action == "freeze" else run() if args.action == "run" else {"冻结来源未变": check()}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
