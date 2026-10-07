"""每日收盘本地观察：无网络、无订单、无账户重跑，缺资料时保持无观点。"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
STUDY_ROOT = PROJECT / "reports/research/510300_sequential_patterns_regime_v1"
sys.path.insert(0, str(STUDY_ROOT / "code"))
from sequential_patterns_regime_v1 import (
    build_labels, clean, control_returns, decisions, detect, digest, features, now, save_json,
)

FORWARD = PROJECT / "data/forward/510300_sequential_patterns_regime_v1"
V2_ROOT = PROJECT / "reports/research/510300_sequential_patterns_2021_2026_v2"


def resolve_local(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def write_status(value):
    FORWARD.mkdir(parents=True, exist_ok=True)
    path = FORWARD / "latest_status.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    value.update(checked_at=now(), source_mode="LOCAL_FILES_ONLY_NO_DOWNLOAD", orders_authorized=False,
                 trading_position="UNSET", current_action="ABSTAIN")
    value["configured_variants"] = ["V1_MIN6_STATE4"] + (["V2_MIN3_STATE2"] if (V2_ROOT / "relaxation_freeze.json").exists() else [])
    attention_key = (value.get("status"), value.get("data_end"), value.get("state"), value.get("decision_digest"))
    previous_key = (old.get("status"), old.get("data_end"), old.get("state"), old.get("decision_digest"))
    if attention_key != previous_key and value.get("status") not in ("NO_CHANGE", "NON_TRADING_DAY"):
        with (FORWARD / "attention_changes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False) + "\n")
    save_json(path, value)
    return value


def check():
    current = datetime.now(ZoneInfo("Asia/Shanghai"))
    today = current.strftime("%Y-%m-%d")
    baseline = json.loads((STUDY_ROOT / "results/summary.json").read_text(encoding="utf-8"))
    calendar = pd.read_csv(STUDY_ROOT / "inputs/calendar.csv", dtype=str)
    dates = set(calendar.trade_date)
    if today not in dates:
        state = "NON_TRADING_DAY" if today <= max(dates) else "NO_VIEW_CALENDAR_EXPIRED"
        return write_status({"status": state, "date": today, "data_end": baseline["data_end"], "forward_observation_created": False})
    inbox = FORWARD / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    manifest_path = inbox / f"{today}.json"
    if not manifest_path.exists():
        return write_status({"status": "NO_VIEW_NO_CURRENT_LOCAL_RECEIPT", "date": today,
                             "data_end": baseline["data_end"], "forward_observation_created": False,
                             "reason": "没有当天本地完整行情、股息覆盖与接收时间清单；不恢复采集。"})
    if not time(16, 0) <= current.time().replace(tzinfo=None) <= time(18, 0):
        return write_status({"status": "NO_VIEW_OUTSIDE_DECISION_WINDOW", "date": today,
                             "data_end": baseline["data_end"], "forward_observation_created": False})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    received = datetime.fromisoformat(manifest["source_received_at"])
    if received.tzinfo is None or received > current or received.astimezone(ZoneInfo("Asia/Shanghai")).date() != current.date():
        raise ValueError("接收时点须是当天、带时区且不晚于实际观察时间。")
    files = {}
    for key in ("prices", "dividends", "dividend_coverage"):
        files[key] = resolve_local(manifest[key]["path"])
        if digest(files[key]) != manifest[key]["sha256"]:
            raise ValueError(f"输入身份不一致：{key}")
    freeze = json.loads((STUDY_ROOT / "freeze.json").read_text(encoding="utf-8"))
    code_key = "code/sequential_patterns_regime_v1.py"
    if digest(STUDY_ROOT / code_key) != freeze["hashes"][code_key]:
        raise ValueError("研究代码与冻结版本不同。")
    raw = pd.read_parquet(files["prices"])
    div = pd.read_csv(files["dividends"], dtype={"record_date": str, "ex_date": str, "payment_date": str})
    coverage = json.loads(files["dividend_coverage"].read_text(encoding="utf-8"))
    if coverage["coverage_end"] < today or not coverage["complete_history_confirmed"]:
        raise ValueError("股息覆盖尚未到当天。")
    if coverage["distribution_file_sha256"] != digest(files["dividends"]):
        raise ValueError("股息清单与覆盖回执不一致。")
    df = features(raw, div)
    if df.date.iloc[-1] != today:
        raise ValueError("当天清单的实际最后行情日不是当天。")
    fixed_raw = pd.read_parquet(STUDY_ROOT / "inputs/prices.parquet")
    cols = ["open", "high", "low", "close", "volume", "amount"]
    pd.testing.assert_frame_equal(raw.iloc[:len(fixed_raw)][cols].reset_index(drop=True), fixed_raw[cols].reset_index(drop=True), check_dtype=False)
    assert pd.to_datetime(raw.date.iloc[:len(fixed_raw)]).reset_index(drop=True).equals(pd.to_datetime(fixed_raw.date).reset_index(drop=True)), "冻结历史日期发生变化。"
    expected = set(calendar.loc[(calendar.trade_date > baseline["data_end"]) & (calendar.trade_date <= today), "trade_date"])
    if not expected.issubset(set(df.date)):
        raise ValueError("新增本地行情包含交易日缺口。")
    ep, steps, signals = detect(df)
    labels = build_labels(df, signals)
    controls = control_returns(df)
    dec, train = decisions(df, signals, labels, controls)
    today_decision = dec[dec.date == today]
    v2_decision = None
    if (V2_ROOT / "relaxation_freeze.json").exists():
        v2_freeze = json.loads((V2_ROOT / "relaxation_freeze.json").read_text(encoding="utf-8"))
        v2_code = "code/sequential_patterns_sample_relaxation_v2.py"
        if digest(V2_ROOT / v2_code) != v2_freeze["hashes"][v2_code]:
            raise ValueError("V2样本放宽代码与冻结文件不一致。")
        sys.path.insert(0, str(V2_ROOT / "code"))
        from sequential_patterns_sample_relaxation_v2 import relax_saved_decisions
        v2_decision = relax_saved_decisions(today_decision, train)
    snapshot = FORWARD / "snapshots" / today
    if snapshot.exists():
        return write_status({"status": "NO_CHANGE", "date": today, "data_end": today,
                             "forward_observation_created": False, "reason": "当天已有不可覆盖的观察记录。"})
    snapshot.mkdir(parents=True, exist_ok=False)
    for key, path in files.items():
        shutil.copy2(path, snapshot / (key + path.suffix))
    for name, frame in (("state", df.tail(1)), ("steps", steps[steps.date == today]),
                        ("signals", signals[signals.signal_date == today]), ("decision", today_decision),
                        ("mature_today", labels[(labels.label_status == "MATURE") & (labels.exit_date == today)])):
        frame.to_csv(snapshot / f"{name}.csv", index=False, encoding="utf-8-sig")
    if v2_decision is not None:
        v2_decision.to_csv(snapshot / "decision_v2_min3_state2.csv", index=False, encoding="utf-8-sig")
        save_json(snapshot / "v2_identity.json", {"protocol_sha256": digest(V2_ROOT / "protocol.json"),
                                                  "code_sha256": digest(V2_ROOT / "code/sequential_patterns_sample_relaxation_v2.py")})
    save_json(snapshot / "input_receipt.json", {"manifest": manifest, "manifest_sha256": digest(manifest_path),
                                                "observed_at": now(), "protocol_sha256": digest(STUDY_ROOT / "protocol.json"),
                                                "code_sha256": digest(STUDY_ROOT / code_key),
                                                "historical_gap_days_are_not_forward_observations": True})
    save_json(snapshot / "snapshot_index.json", {p.name: digest(p) for p in snapshot.iterdir() if p.is_file()})
    return write_status({"status": "OBSERVATION_SAVED_RESEARCH_ONLY", "date": today, "data_end": today,
                         "state": df.state.iloc[-1], "decision_digest": digest(snapshot / "decision.csv"),
                         "forward_observation_created": True, "snapshot": str(snapshot),
                         "new_fits": 0, "new_accounts": 0, "historical_protocol_modified": False,
                         "signal_count": int((signals.signal_date == today).sum())})


if __name__ == "__main__":
    try:
        outcome = check()
    except Exception as exc:
        outcome = write_status({"status": "OBSERVATION_FAILED", "forward_observation_created": False,
                                "error": str(exc), "traceback": traceback.format_exc()})
    if sys.stdout is not None:
        print(json.dumps(clean(outcome), ensure_ascii=False, indent=2))
    if outcome["status"] == "OBSERVATION_FAILED":
        sys.exit(1)
