"""只核验分钟时间、成交总量和 VWAP；不构造跨窗口收益。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.nbs_v2_common import (ContractError, committed, csv_bytes, git,
                                   identity, load_json, now, sha, verify, write_once)

CONFIG = "config/510300_stk_mins_source_admission_v2.yaml"
MANIFEST = "config/510300_stk_mins_source_admission_v2_manifest.json"
OUT = "reports/data_quality/510300_stk_mins_source_admission_v2"
CURATED = "data/curated/510300_nbs_1000_negative_information_drift_v2"
SCOPE = [CONFIG, "research/nbs_v2_common.py", "research/stk_mins_source_admission_v2.py",
         "scripts/run_510300_nbs_v2_source.py", "tests/test_510300_nbs_v2_source.py"]


def config(root: Path) -> dict:
    return yaml.safe_load((root / CONFIG).read_text("utf-8"))


def freeze(root: Path) -> dict:
    """冻结所有既有来源身份；仅读文件字节与元数据。"""
    cfg = config(root)
    paths = set(SCOPE + ["docs/510300_NBS_V2_USER_REQUEST_20260905.md"])
    paths.update(cfg["inputs"].values())
    parent = load_json(root / cfg["inputs"]["parent_adjudication"])
    if parent["event_return_reads"] != 0 or parent["model_training_run"]:
        raise ContractError("V1已消费事件结果，V2不得按无收益父版本登记")
    acquisition = load_json(root / cfg["inputs"]["acquisition"])
    for rec in acquisition["records"]:
        paths.add(rec["receipt_relative_path"])
        receipt = load_json(root / rec["receipt_relative_path"])
        paths.update([receipt["raw_relative_path"], receipt["normalized_relative_path"]])
        if sha(root / receipt["raw_relative_path"]) != rec["response_sha256"]:
            raise ContractError("原始分钟采集字节与V1回执不一致")
        if sha(root / receipt["normalized_relative_path"]) != rec["normalized_sha256"]:
            raise ContractError("分钟规范化分区与V1回执不一致")
    inventory = load_json(root / cfg["inputs"]["nbs_inventory"])
    for rec in inventory["records"]:
        paths.add(rec["raw_relative_path"])
        if sha(root / rec["raw_relative_path"]) != rec["sha256"]:
            raise ContractError("NBS官方日程原件哈希不一致")
    evidence = load_json(root / cfg["inputs"]["timestamp_evidence"])
    for path, expected in evidence["evidence_file_sha256"].items():
        paths.add(path)
        if sha(root / path) != expected:
            raise ContractError("BAR_END既有证据版本不一致")
    m = load_json(root / cfg["inputs"]["parent_ledger_manifest"])
    if sha(root / cfg["inputs"]["parent_ledger"]) != m["pre_return_ledger_sha256"]:
        raise ContractError("V1无收益账本哈希不一致")
    report = {"model_id": cfg["data_model_id"], "state": "FROZEN_BEFORE_V2_SOURCE_MEASUREMENT",
              "frozen_at": now(), "event_return_values_read": False,
              "scope": SCOPE, "identities": [identity(root, p) for p in sorted(paths)],
              "position_impact": 0}
    write_once(root / MANIFEST, report)
    return {k: v for k, v in report.items() if k != "identities"}


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["ts_code", "trade_time", "vol", "amount", "low", "high"]
    if not set(required).issubset(frame.columns):
        raise ContractError("分钟源缺少时间、成交量额或VWAP边界列")
    out = frame.copy()
    out["trade_time"] = pd.to_datetime(out["trade_time"], errors="raise")
    out["date"] = out.trade_time.dt.normalize()
    out["clock"] = out.trade_time.dt.strftime("%H:%M:%S")
    for name in ["vol", "amount", "low", "high"]:
        out[name] = pd.to_numeric(out[name], errors="raise")
    return out.sort_values("trade_time").reset_index(drop=True)


def standard_clocks() -> set[str]:
    return set(pd.date_range("2000-01-01 09:30", "2000-01-01 11:30", freq="min").strftime("%H:%M:%S")) | set(
        pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="min").strftime("%H:%M:%S"))


def window_table(frame: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """产生单个窗口测量，不进行窗口之间的价格比值运算。"""
    rows = []
    for name, labels in windows.items():
        part = frame.loc[frame.clock.isin(labels)]
        for day, group in part.groupby("date", sort=True):
            complete = (len(group) == len(labels) and set(group.clock) == set(labels)
                        and np.isfinite(group[["vol", "amount"]]).all().all()
                        and (group.vol >= 0).all() and (group.amount >= 0).all()
                        and group.vol.sum() > 0 and group.amount.sum() > 0)
            rows.append({"date": day, "window": name, "complete": complete,
                         "vwap": float(group.amount.sum() / group.vol.sum())
                         if complete else np.nan})
    return pd.DataFrame(rows)


def measure(minute: pd.DataFrame, daily: pd.DataFrame, open_dates: pd.DatetimeIndex,
            independent: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    minute, independent = normalize(minute), normalize(independent)
    q = cfg["quality_gates"]
    clocks = standard_clocks()
    dates = pd.DatetimeIndex(open_dates).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ContractError("官方开放日日历重复或不递增")
    grouped = minute.groupby("date", sort=True)
    standard = grouped.clock.apply(lambda x: len(x) == 241 and set(x) == clocks)
    day = pd.DataFrame(index=dates)
    day.index.name = "date"
    day["standard_241"] = standard.reindex(dates, fill_value=False)
    totals = grouped[["vol", "amount"]].sum()
    ref = daily.copy()
    ref["date"] = pd.to_datetime(ref.trade_date.astype(str))
    if ref.duplicated("date").any() or not ref.ts_code.eq(cfg["symbol"]).all():
        raise ContractError("fund_daily主键或证券代码不合格")
    ref = ref.set_index("date").reindex(dates)
    day["volume_relative_error"] = abs(totals.vol.reindex(dates) / 100 - ref.vol) / ref.vol.abs()
    day["amount_relative_error"] = abs(totals.amount.reindex(dates) / 1000 - ref.amount) / ref.amount.abs()
    table = window_table(minute, cfg["windows"])
    complete = table.pivot(index="date", columns="window", values="complete").reindex(dates).fillna(False)
    for name in cfg["windows"]:
        day[f"{name}_complete"] = complete[name]
    eligible = pd.to_datetime(events.loc[events.final_event_eligibility, "scheduled_date"])
    event_ok = complete.all(axis=1).reindex(eligible, fill_value=False)
    finite = np.isfinite(minute[["vol", "amount", "low", "high"]]).all(axis=1)
    defined = finite & (minute.vol > 0) & (minute.amount > 0)
    per_bar = minute.amount.div(minute.vol.where(defined))
    tolerance = q["bar_vwap_price_tolerance_cny"] + q["binary_float_tolerance_cny"]
    outside = defined & ((per_bar < minute.low - tolerance) | (per_bar > minute.high + tolerance))
    illegal = (~minute.clock.isin(clocks) | ~minute.date.isin(dates)
               | (minute.trade_time.dt.microsecond != 0) | (minute.trade_time.dt.nanosecond != 0))
    a, b = pd.Timestamp(q["independent_overlap_start"]), pd.Timestamp(q["independent_overlap_end"])
    expected_overlap = dates[(dates >= a) & (dates <= b)]
    index = pd.MultiIndex.from_product([expected_overlap, list(cfg["windows"])], names=["date", "window"])
    left = table.set_index(["date", "window"]).reindex(index)
    right = window_table(independent, cfg["windows"]).set_index(["date", "window"]).reindex(index)
    matches = (left.complete.fillna(False) & right.complete.fillna(False)
               & np.isclose(left.vwap, right.vwap, atol=q["independent_vwap_absolute_tolerance_cny"],
                             rtol=q["independent_vwap_relative_tolerance"]))
    checks = {
        "standard_241_coverage": day.standard_241.mean() >= q["standard_241_bar_day_coverage_minimum"],
        "all_event_windows": len(event_ok) > 0 and event_ok.mean() == 1,
        "duplicate_keys_zero": int(minute.duplicated(["ts_code", "trade_time"]).sum()) == 0,
        "illegal_timestamps_zero": int(illegal.sum()) == 0,
        "only_510300": minute.ts_code.eq(cfg["symbol"]).all(),
        "daily_volume_error": np.isfinite(day.volume_relative_error).all()
                              and (day.volume_relative_error <= q["daily_volume_relative_error_maximum"]).all(),
        "daily_amount_error": np.isfinite(day.amount_relative_error).all()
                              and (day.amount_relative_error <= q["daily_amount_relative_error_maximum"]).all(),
        "all_bar_vwap_defined": defined.all(),
        "all_bar_vwap_inside_low_high": not outside.any(),
        "finite_positive_low_high": finite.all() and (minute.low > 0).all() and (minute.high >= minute.low).all(),
        "independent_keys_unique": not independent.duplicated(["ts_code", "trade_time"]).any(),
        "independent_only_510300": independent.ts_code.eq(cfg["symbol"]).all(),
        "independent_overlap_vwap": len(matches) > 0 and matches.mean() >= q["independent_four_window_vwap_match_minimum"],
        "bar_end_contract": cfg["timestamp_semantics"] == "BAR_END",
    }
    diagnostic = {}
    for field, reducer in [("high", "max"), ("low", "min"), ("close", "last")]:
        values = getattr(grouped[field], reducer)().reindex(dates)
        difference = abs(values - ref[field])
        diagnostic[field] = {"above_0_001_days": int((difference > .001 + 1e-12).sum()),
                             "maximum_absolute_difference": difference.max(), "hard_gate": False}
    report = {"checks": checks, "source_pass": all(checks.values()),
              "minute_rows": len(minute), "open_days": len(dates),
              "standard_241_days": day.standard_241.sum(), "eligible_events": len(event_ok),
              "complete_event_windows": event_ok.sum(), "undefined_bar_vwap_rows": (~defined).sum(),
              "bar_vwap_out_of_range_rows": outside.sum(),
              "illegal_timestamp_rows": illegal.sum(),
              "maximum_volume_relative_error": day.volume_relative_error.max(),
              "maximum_amount_relative_error": day.amount_relative_error.max(),
              "independent_window_count": len(matches), "independent_matching_windows": matches.sum(),
              "independent_vwap_match_rate": matches.mean(), "ohlc_diagnostic_only": diagnostic}
    defects = minute.loc[outside | ~defined, ["ts_code", "trade_time", "vol", "amount", "low", "high"]].copy()
    defects["bar_vwap"] = per_bar[outside | ~defined]
    return report, day.reset_index(), defects


def rebuild_ledger(parent: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    ledger = parent.copy()
    ordinal = pd.to_numeric(ledger.model_event_ordinal)
    ledger["era"] = "NOT_MODEL_ELIGIBLE"
    for name, start, end in [("TRAINING_ORIGIN", 1, 36), ("ERA_1", 37, 53),
                             ("ERA_2", 54, 70), ("ERA_3", 71, 86)]:
        ledger.loc[ordinal.between(start, end).fillna(False), "era"] = name
    day = quality.set_index("date")
    complete = day.filter(like="_complete").all(axis=1) & day.standard_241
    ledger["v2_four_windows_complete"] = pd.to_datetime(ledger.scheduled_date).map(complete).fillna(False)
    if not ledger.loc[ledger.final_event_eligibility, "v2_four_windows_complete"].all():
        raise ContractError("既有合格事件出现窗口缺失，禁止静默删样本")
    return ledger


def run(root: Path) -> dict:
    manifest = load_json(root / MANIFEST)
    verify(root, manifest["identities"])
    committed(root, SCOPE + [MANIFEST])
    if (root / OUT / "source_adjudication.json").exists():
        raise ContractError("V2来源检验已记录；请读取既有回执")
    cfg = config(root)
    acq = load_json(root / cfg["inputs"]["acquisition"])
    refs = {}
    for rec in acq["records"][:2]:
        receipt = load_json(root / rec["receipt_relative_path"])
        refs[rec["api"]] = pd.read_parquet(root / receipt["normalized_relative_path"])
    cal = refs["trade_cal"]
    opens = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1), "cal_date"].astype(str))).sort_values()
    events = pd.read_parquet(root / cfg["inputs"]["parent_ledger"])
    result, daily, defects = measure(pd.read_parquet(root / cfg["inputs"]["minute"]), refs["fund_daily"],
                                     opens, pd.read_parquet(root / cfg["inputs"]["independent_minute"]), events, cfg)
    evidence = load_json(root / cfg["inputs"]["timestamp_evidence"])
    result["checks"]["bar_end_evidence"] = evidence["pass"]
    result["source_pass"] = all(result["checks"].values())
    ledger = rebuild_ledger(events, daily)
    counts = ledger.loc[ledger.model_pre_return_eligibility, "era"].value_counts().to_dict()
    g1 = (counts == {"TRAINING_ORIGIN": 36, "ERA_1": 17, "ERA_2": 17, "ERA_3": 16}
          and result["eligible_events"] == 88)
    result.update({"model_id": cfg["data_model_id"], "generated_at": now(),
                   "state": "PASS_STK_MINS_SOURCE_ADMISSION_V2" if result["source_pass"] else "BLOCKED_STK_MINS_SOURCE_ADMISSION_V2",
                   "g1_pass": g1, "era_counts": counts, "freeze_commit": git(root, "rev-parse", "HEAD"),
                   "event_return_reads": 0, "event_labels_created": 0, "model_training_run": False,
                   "portfolio_evaluation_run": False, "position_impact": 0})
    write_once(root / OUT / "daily_measurement_quality.csv", csv_bytes(daily))
    write_once(root / OUT / "bar_vwap_defects.csv", csv_bytes(defects))
    write_once(root / CURATED / "event_ledger_pre_return.csv", csv_bytes(ledger))
    result["outputs"] = [identity(root, p) for p in [f"{OUT}/daily_measurement_quality.csv",
                         f"{OUT}/bar_vwap_defects.csv", f"{CURATED}/event_ledger_pre_return.csv"]]
    write_once(root / OUT / "source_adjudication.json", result)
    return result
