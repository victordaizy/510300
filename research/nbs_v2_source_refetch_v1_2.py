"""同供应商教程域名上的可续传重取，保留每次传输尝试和原数据门槛。"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

from research.nbs_v2_common import ContractError, committed, csv_bytes, git, identity, load_json, now, verify, write_once
from research.nbs_v2_source_refetch_v1_1 import COLUMNS, RAW as PARENT_RAW, client, fetch
from research.stk_mins_source_admission_v2 import measure, normalize

CONFIG = "config/510300_nbs_v2_source_refetch_v1_2_20260905.json"
OUT = "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905"
CURATED = "data/curated/510300_nbs_v2_source_refetch_v1_2_20260905"
MANIFEST = f"{OUT}/manifest.json"
SCOPE = [CONFIG, "research/nbs_v2_source_refetch_v1_2.py",
         "scripts/run_510300_nbs_v2_source_refetch_v1_2.py",
         "tests/test_510300_nbs_v2_source_refetch_v1_2.py"]


def retryable(receipt: dict) -> bool:
    return (receipt.get("http_status") in [429, 500, 502, 503, 504]
            or receipt.get("business_code") in ["CONNECTIONERROR", "TIMEOUT", "READTIMEOUT", "CONNECTTIMEOUT"])


def freeze(root: Path) -> dict:
    cfg = load_json(root / CONFIG)
    parent = load_json(root / cfg["parent_manifest"])
    verify(root, parent["identities"])
    verify(root, load_json(root / "config/510300_stk_mins_source_admission_v2_manifest.json")["identities"])
    failed = load_json(root / cfg["trigger_receipt"])
    if failed["success"] or not retryable(failed):
        raise ContractError("只允许对已确认的传输失败登记本次续取")
    tutorial = (root / cfg["tutorial_capture"]).read_text("utf-8")
    if cfg["endpoint"] not in tutorial or cfg["endpoint"] != "https://fast.xiaodefa.cn":
        raise ContractError("续取域名必须是用户教程明确列出的同供应商域名")
    paths = SCOPE + [cfg["parent_manifest"], cfg["trigger_receipt"], cfg["tutorial_capture"],
                     "research/nbs_v2_source_refetch_v1_1.py",
                     "config/510300_nbs_v2_source_refetch_v1_1_20260905.json"]
    for month in cfg["reuse_completed_months"]:
        p = f"{PARENT_RAW}/months/{month}/receipt.json"
        rec = load_json(root / p)
        verify(root, [rec["response"], rec["normalized"]])
        paths += [p, rec["response"]["path"], rec["normalized"]["path"]]
    result = {"state": "FROZEN_TRANSPORT_RESUMPTION_ORIGINAL_QUALITY_GATES",
              "generated_at": now(), "identities": [identity(root, p) for p in sorted(set(paths))],
              "event_return_values_read": False, "source_hard_gates_changed": False}
    write_once(root / MANIFEST, result)
    return {"state": result["state"], "frozen_files": len(result["identities"])}


def validate(root: Path) -> tuple[dict, dict]:
    cfg = load_json(root / CONFIG)
    verify(root, load_json(root / MANIFEST)["identities"])
    verify(root, load_json(root / cfg["parent_manifest"])["identities"])
    verify(root, load_json(root / "config/510300_stk_mins_source_admission_v2_manifest.json")["identities"])
    committed(root, SCOPE + [MANIFEST])
    source = yaml.safe_load((root / "config/510300_stk_mins_source_admission_v2.yaml").read_text("utf-8"))
    return cfg, source


def request_partition(root: Path, api, cfg: dict, key: str, parameters: dict, old: pd.DataFrame) -> dict:
    for attempt in range(1, cfg["maximum_transport_attempts"] + 1):
        namespace = f"resume_v1_2/{key}/attempt_{attempt}"
        path = root / PARENT_RAW / namespace / "receipt.json"
        previous = load_json(path) if path.exists() else None
        if previous and not previous.get("success"):
            if not retryable(previous):
                raise ContractError("认证、权限或参数错误不重试")
            continue
        try:
            return fetch(root, api, namespace, parameters, old)
        except ContractError:
            failed = load_json(path) if path.exists() else None
            if not failed or not retryable(failed):
                raise
            print(f"传输失败已留存：{key}，尝试 {attempt}/{cfg['maximum_transport_attempts']}", flush=True)
            if attempt < cfg["maximum_transport_attempts"]:
                time.sleep(cfg["retry_delay_seconds"][attempt - 1])
    raise ContractError("固定传输重试次数已用完，等待外部网络恢复")


def acquire(root: Path) -> dict:
    cfg, source = validate(root)
    if (root / OUT / "acquisition_receipt.json").exists():
        raise ContractError("来源重取已完成，请读取既有回执")
    api = client(root, cfg)
    original = normalize(pd.read_parquet(root / source["inputs"]["minute"]))
    source_month = original.trade_time.dt.strftime("%Y%m")
    alias_probes = []
    for day in cfg["alias_probe_dates"]:
        params = {"ts_code": "510300.SH", "freq": "1min", "start_date": f"{day} 09:00:00", "end_date": f"{day} 16:00:00"}
        rec = request_partition(root, api, cfg, f"alias_probe/{day}", params,
                                original.loc[original.date.eq(pd.Timestamp(day)), COLUMNS])
        if not rec["comparison"]["all_native_fields_identical"]:
            raise ContractError("教程域名返回内容与既有探测不一致，禁止按传输等价继续")
        alias_probes.append(rec)
        print(f"教程域名等价探测通过：{day}", flush=True)
    pieces = [original.loc[~source_month.isin(cfg["months"]), COLUMNS]]
    records = []
    for i, month in enumerate(cfg["months"], 1):
        if month in cfg["reuse_completed_months"]:
            rec = load_json(root / PARENT_RAW / "months" / month / "receipt.json")
            verify(root, [rec["response"], rec["normalized"]])
        else:
            left = pd.Timestamp(month + "01")
            right = min(left + pd.offsets.MonthEnd(0), pd.Timestamp(source["end_date"]))
            params = {"ts_code": "510300.SH", "freq": "1min", "start_date": f"{left:%Y-%m-%d} 00:00:00",
                      "end_date": f"{right:%Y-%m-%d} 23:59:59"}
            rec = request_partition(root, api, cfg, f"months/{month}", params,
                                    original.loc[source_month.eq(month), COLUMNS])
        pieces.append(pd.read_parquet(root / rec["normalized"]["path"]))
        records.append(rec)
        print(f"异常月份完成：{i}/{len(cfg['months'])}，月份={month}，字段变化={not rec['comparison']['all_native_fields_identical']}", flush=True)
    merged = pd.concat(pieces, ignore_index=True).sort_values("trade_time").reset_index(drop=True)
    if len(merged) != len(original) or merged.duplicated(["ts_code", "trade_time"]).any():
        raise ContractError("完整候选数据主键或行数不一致")
    path = root / CURATED / "510300_1min_candidate.parquet"
    if path.exists():
        raise ContractError("候选数据已存在，禁止覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    result = {"state": "REFETCH_COMPLETE_CANDIDATE_NOT_YET_ADMITTED", "generated_at": now(),
              "months_completed": len(records), "reused_months": len(cfg["reuse_completed_months"]),
              "fresh_rows": sum(r["comparison"]["rows"] for r in records), "candidate_rows": len(merged),
              "unchanged_months": sum(r["comparison"]["all_native_fields_identical"] for r in records),
              "changed_rows_by_field": {c: sum(r["comparison"]["changed_rows_by_field"][c] for r in records) for c in COLUMNS[2:]},
              "alias_probes": alias_probes, "records": records, "candidate": identity(root, path),
              "event_return_values_read": False, "source_hard_gates_changed": False, "position_impact": 0}
    write_once(root / OUT / "acquisition_receipt.json", result)
    return {k: v for k, v in result.items() if k not in ["records", "alias_probes"]}


def audit(root: Path) -> dict:
    cfg, source = validate(root)
    if (root / OUT / "source_adjudication.json").exists():
        raise ContractError("重取来源已审核，请读取既有回执")
    acq = load_json(root / OUT / "acquisition_receipt.json")
    verify(root, [acq["candidate"]])
    for rec in acq["records"]:
        verify(root, [rec["response"], rec["normalized"]])
    initial = load_json(root / source["inputs"]["acquisition"])
    refs = {}
    for r in initial["records"][:2]:
        rec = load_json(root / r["receipt_relative_path"])
        refs[r["api"]] = pd.read_parquet(root / rec["normalized_relative_path"])
    cal = refs["trade_cal"]
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1), "cal_date"].astype(str))).sort_values()
    result, daily, defects = measure(pd.read_parquet(root / acq["candidate"]["path"]), refs["fund_daily"], dates,
        pd.read_parquet(root / source["inputs"]["independent_minute"]),
        pd.read_parquet(root / source["inputs"]["parent_ledger"]), source)
    result["checks"]["bar_end_evidence"] = load_json(root / source["inputs"]["timestamp_evidence"])["pass"]
    result["source_pass"] = all(result["checks"].values())
    result.update({"state": "PASS_REFETCHED_DATA_ORIGINAL_V2_CONTRACT" if result["source_pass"]
                   else "BLOCKED_REFETCHED_DATA_ORIGINAL_V2_CONTRACT_FAILED",
                   "generated_at": now(), "freeze_commit": git(root, "rev-parse", "HEAD"),
                   "active_source_contract": "config/510300_stk_mins_source_admission_v2.yaml",
                   "source_hard_gates_changed": False, "candidate": acq["candidate"],
                   "all_77_months_identical_to_original": acq["unchanged_months"] == 77,
                   "event_return_values_read": False, "event_labels_created": 0,
                   "G2": "READY_FOR_FROZEN_PREDICTION" if result["source_pass"] else "NOT_RUN_BLOCKED_BY_G0",
                   "G3": "NOT_RUN", "G4": "NOT_RUN", "position_impact": 0})
    write_once(root / OUT / "daily_quality.csv", csv_bytes(daily))
    write_once(root / OUT / "bar_vwap_defects.csv", csv_bytes(defects))
    result["outputs"] = [identity(root, f"{OUT}/{p}") for p in ["daily_quality.csv", "bar_vwap_defects.csv"]]
    write_once(root / OUT / "source_adjudication.json", result)
    return result
