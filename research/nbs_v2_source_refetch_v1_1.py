"""使用新授权凭据重取所有异常月份，沿用原 V2 来源硬门。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import dotenv_values

from research.nbs_v2_common import (
    ContractError, clean, committed, csv_bytes, git, identity, load_json, now, sha, verify, write_once,
)
from research.stk_mins_source_admission_v1 import TushareProxyClient
from research.stk_mins_source_admission_v2 import measure, normalize, standard_clocks

CONFIG = "config/510300_nbs_v2_source_refetch_v1_1_20260905.json"
OUT = "reports/data_quality/510300_nbs_v2_source_refetch_v1_1_20260905"
RAW = "data/raw/tushare/510300_nbs_v2_source_refetch_v1_1_20260905"
CURATED = "data/curated/510300_nbs_v2_source_refetch_v1_1_20260905"
MANIFEST = f"{OUT}/refetch_manifest.json"
SCOPE = [CONFIG, "research/nbs_v2_source_refetch_v1_1.py",
         "scripts/run_510300_nbs_v2_source_refetch_v1_1.py",
         "tests/test_510300_nbs_v2_source_refetch_v1_1.py"]
COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]


def compare_partition(old: pd.DataFrame, fresh: pd.DataFrame) -> dict:
    """拒绝主键、证券、完整性和量额格式变化；报告所有字段变化，不择优拼柱。"""
    a, b = normalize(old), normalize(fresh)
    if b.empty or len(b) >= 8000:
        raise ContractError("重取响应为空或触及 8000 行截断界限")
    for f in [a, b]:
        if f.duplicated(["ts_code", "trade_time"]).any() or not f.ts_code.eq("510300.SH").all():
            raise ContractError("重取分区证券或主键不合格")
    fields = COLUMNS[2:]
    if not set(fields).issubset(b.columns) or not np.isfinite(b[fields]).all().all():
        raise ContractError("重取分区存在缺列或非有限值")
    if (b.vol < 0).any() or (b.amount < 0).any() or (b.low <= 0).any() or (b.high < b.low).any():
        raise ContractError("重取分区量额或价格边界非法")
    if not b.groupby("date").clock.apply(lambda x: len(x) == 241 and set(x) == standard_clocks()).all():
        raise ContractError("重取分区不是完整的 241 根开放日")
    p = a.merge(b, on=["ts_code", "trade_time"], how="outer", indicator=True,
                suffixes=("_old", "_fresh"), validate="one_to_one")
    if len(p) != len(a) or not p._merge.eq("both").all():
        raise ContractError("重取分区与冻结分区时间键不一致，禁止静默删样本")
    changed = {c: int(p[f"{c}_old"].ne(p[f"{c}_fresh"]).sum()) for c in fields}
    defined = b.vol.gt(0) & b.amount.gt(0)
    vwap = b.amount / b.vol.where(defined)
    outside = defined & (vwap.lt(b.low - .001 - 1e-12) | vwap.gt(b.high + .001 + 1e-12))
    return {"rows": len(b), "days": b.date.nunique(), "changed_rows_by_field": changed,
            "all_native_fields_identical": not any(changed.values()),
            "positive_bar_vwap_outside_rows": int(outside.sum()),
            "undefined_bar_vwap_rows": int((~defined).sum())}


def freeze(root: Path) -> dict:
    cfg = load_json(root / CONFIG)
    source = yaml.safe_load((root / cfg["source_contract"]).read_text("utf-8"))
    verify(root, load_json(root / cfg["source_manifest"])["identities"])
    defects = pd.read_csv(root / cfg["defects"])
    months = sorted(pd.to_datetime(defects.trade_time).dt.strftime("%Y%m").unique().tolist())
    if cfg["months"] != months or len(months) != 77:
        raise ContractError("必须事前冻结全部 77 个异常月份，禁止挑选月份")
    status = load_json(root / cfg["credential_status_receipt"])
    if not status.get("found") or status.get("expired") or status.get("disabled"):
        raise ContractError("供应商凭据状态尚未通过")
    paths = SCOPE + [cfg["source_contract"], cfg["source_manifest"], cfg["defects"],
                     cfg["credential_status_receipt"], cfg["tutorial_capture"],
                     "research/nbs_v2_common.py", "research/stk_mins_source_admission_v1.py",
                     "research/stk_mins_source_admission_v2.py",
                     "reports/data_quality/510300_nbs_v2_data_remediation_20260905/completion_receipt.json",
                     "config/510300_nbs_1000_negative_information_drift_v2.yaml"]
    paths += list(source["inputs"].values())
    report = {"state": "FROZEN_BEFORE_NEW_CREDENTIAL_MINUTE_REFETCH", "generated_at": now(),
              "identities": [identity(root, p) for p in sorted(set(paths))],
              "months": months, "probe_dates": cfg["probe_dates"], "event_return_values_read": False,
              "source_hard_gates_modified": False, "position_impact": 0}
    write_once(root / MANIFEST, report)
    return {"state": report["state"], "months": len(months), "frozen_files": len(report["identities"])}


def validate(root: Path) -> tuple[dict, dict]:
    cfg = load_json(root / CONFIG)
    verify(root, load_json(root / MANIFEST)["identities"])
    verify(root, load_json(root / cfg["source_manifest"])["identities"])
    committed(root, SCOPE + [MANIFEST])
    source = yaml.safe_load((root / cfg["source_contract"]).read_text("utf-8"))
    return cfg, source


def client(root: Path, cfg: dict) -> TushareProxyClient:
    env = dotenv_values(root / ".env")
    token = os.getenv("TUSHARE_PROXY_TOKEN") or env.get("TUSHARE_PROXY_TOKEN") or ""
    expiry = os.getenv("TUSHARE_PROXY_TOKEN_EXPIRES_AT") or env.get("TUSHARE_PROXY_TOKEN_EXPIRES_AT")
    if len(token) != 56 or not expiry or pd.Timestamp(expiry).tzinfo is None:
        raise ContractError("临时凭据或带时区的到期时间缺失")
    if pd.Timestamp(expiry) <= pd.Timestamp(now()):
        raise ContractError("临时凭据已过期，停止请求")
    return TushareProxyClient(cfg["endpoint"], token, cfg["minimum_request_interval_seconds"],
                             15, cfg["read_timeout_seconds"], 1, [])


def fetch(root: Path, api: TushareProxyClient, key: str, parameters: dict, old: pd.DataFrame) -> dict:
    """每个请求领取一次；中断后只复用完整且字节已验证的成功回执。"""
    path = root / RAW / key
    receipt_path = path / "receipt.json"
    if receipt_path.exists():
        rec = load_json(receipt_path)
        if not rec.get("success") or rec["parameters"] != parameters:
            raise ContractError("已有请求失败或参数不同，禁止自动重试")
        verify(root, [rec["response"], rec["normalized"]])
        return rec
    if (path / "claim.json").exists():
        raise ContractError("请求有 claim 而无完整回执，停止以避免重复请求")
    write_once(path / "claim.json", {"claimed_at": now(), "parameters": parameters})
    result = api.call("stk_mins", parameters)
    rec = {"api": "stk_mins", "parameters": parameters, "request_started_at": result.request_started_at,
           "response_received_at": result.response_received_at, "http_status": result.http_status,
           "business_code": result.business_code, "success": result.success,
           "credential_persisted": False, "event_return_values_read": False}
    if not result.success:
        rec["error_message"] = result.error_message
        write_once(receipt_path, rec)
        raise ContractError("供应商重取失败，详情见已去除凭据的请求回执")
    write_once(path / "response.json", result.raw_bytes)
    rec["response"] = identity(root, path / "response.json")
    fresh = result.frame[COLUMNS].copy()
    fresh.trade_time = pd.to_datetime(fresh.trade_time)
    fresh = fresh.sort_values("trade_time").reset_index(drop=True)
    rec["comparison"] = compare_partition(old, fresh)
    normalized_path = path / "normalized.parquet"
    if normalized_path.exists():
        raise ContractError("重取规范化分区已存在但尚无成功回执")
    fresh.to_parquet(normalized_path, index=False)
    rec["normalized"] = identity(root, normalized_path)
    write_once(receipt_path, rec)
    return rec


def probe(root: Path) -> dict:
    cfg, source = validate(root)
    if (root / OUT / "probe_receipt.json").exists():
        raise ContractError("三个异常日探测已有回执")
    api = client(root, cfg)
    minute = normalize(pd.read_parquet(root / source["inputs"]["minute"]))
    records = []
    for day in cfg["probe_dates"]:
        parameters = {"ts_code": "510300.SH", "freq": "1min", "start_date": f"{day} 09:00:00",
                      "end_date": f"{day} 16:00:00"}
        rec = fetch(root, api, f"probe/{day}", parameters,
                    minute.loc[minute.date.eq(pd.Timestamp(day)), COLUMNS])
        records.append(rec)
        print(f"异常日重取完成：{day}，逐字段变化={not rec['comparison']['all_native_fields_identical']}", flush=True)
    result = {"state": "PASS_REFETCH_TRANSPORT_AND_KEYS_ONLY", "generated_at": now(),
              "records": records, "source_admitted": False, "event_return_values_read": False}
    write_once(root / OUT / "probe_receipt.json", result)
    return result


def acquire(root: Path) -> dict:
    cfg, source = validate(root)
    if (root / OUT / "acquisition_receipt.json").exists():
        raise ContractError("77 个月份重取已完成")
    probe_result = load_json(root / OUT / "probe_receipt.json")
    if probe_result["state"] != "PASS_REFETCH_TRANSPORT_AND_KEYS_ONLY":
        raise ContractError("必须先通过三个异常日的权限、字段与主键检查")
    api = client(root, cfg)
    original = normalize(pd.read_parquet(root / source["inputs"]["minute"]))
    source_month = original.trade_time.dt.strftime("%Y%m")
    pieces = [original.loc[~source_month.isin(cfg["months"]), COLUMNS]]
    records = []
    for i, month in enumerate(cfg["months"], 1):
        left = pd.Timestamp(month + "01")
        right = min(left + pd.offsets.MonthEnd(0), pd.Timestamp(source["end_date"]))
        parameters = {"ts_code": "510300.SH", "freq": "1min",
                      "start_date": f"{left:%Y-%m-%d} 00:00:00", "end_date": f"{right:%Y-%m-%d} 23:59:59"}
        rec = fetch(root, api, f"months/{month}", parameters, original.loc[source_month.eq(month), COLUMNS])
        records.append(rec)
        pieces.append(pd.read_parquet(root / rec["normalized"]["path"]))
        print(f"异常月份重取：{i}/{len(cfg['months'])}，月份={month}，字段变化={not rec['comparison']['all_native_fields_identical']}", flush=True)
    merged = pd.concat(pieces, ignore_index=True).sort_values("trade_time").reset_index(drop=True)
    if len(merged) != len(original) or merged.duplicated(["ts_code", "trade_time"]).any():
        raise ContractError("重取月分区替换后主键或总行数变化，停止")
    path = root / CURATED / "510300_1min_candidate.parquet"
    if path.exists():
        raise ContractError("重取候选数据已存在，禁止覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    result = {"state": "REFETCH_COMPLETE_CANDIDATE_NOT_YET_ADMITTED", "generated_at": now(),
              "months_requested": len(records), "fresh_rows": sum(r["comparison"]["rows"] for r in records),
              "candidate_rows": len(merged), "unchanged_months": sum(r["comparison"]["all_native_fields_identical"] for r in records),
              "changed_rows_by_field": {c: sum(r["comparison"]["changed_rows_by_field"][c] for r in records) for c in COLUMNS[2:]},
              "records": records, "candidate": identity(root, path),
              "event_return_values_read": False, "position_impact": 0}
    write_once(root / OUT / "acquisition_receipt.json", result)
    return {k: v for k, v in result.items() if k != "records"}


def audit(root: Path) -> dict:
    cfg, source = validate(root)
    if (root / OUT / "source_adjudication.json").exists():
        raise ContractError("重取候选来源审核已经完成")
    acq = load_json(root / OUT / "acquisition_receipt.json")
    verify(root, [acq["candidate"]])
    for r in acq["records"]:
        verify(root, [r["response"], r["normalized"]])
    initial = load_json(root / source["inputs"]["acquisition"])
    refs = {}
    for r in initial["records"][:2]:
        receipt = load_json(root / r["receipt_relative_path"])
        refs[r["api"]] = pd.read_parquet(root / receipt["normalized_relative_path"])
    cal = refs["trade_cal"]
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1), "cal_date"].astype(str))).sort_values()
    result, daily, defects = measure(
        pd.read_parquet(root / acq["candidate"]["path"]), refs["fund_daily"], dates,
        pd.read_parquet(root / source["inputs"]["independent_minute"]),
        pd.read_parquet(root / source["inputs"]["parent_ledger"]), source,
    )
    result["checks"]["bar_end_evidence"] = load_json(root / source["inputs"]["timestamp_evidence"])["pass"]
    result["source_pass"] = all(result["checks"].values())
    result.update({"state": "PASS_REFETCHED_DATA_ORIGINAL_V2_CONTRACT" if result["source_pass"]
                   else "BLOCKED_REFETCHED_DATA_ORIGINAL_V2_CONTRACT_FAILED",
                   "generated_at": now(), "freeze_commit": git(root, "rev-parse", "HEAD"),
                   "active_source_contract": cfg["source_contract"], "source_hard_gates_changed": False,
                   "candidate": acq["candidate"], "all_77_months_identical_to_original": acq["unchanged_months"] == 77,
                   "event_return_values_read": False, "event_labels_created": 0,
                   "G2": "READY_FOR_PRE_RETURN_MODEL_IMPLEMENTATION_FREEZE" if result["source_pass"] else "NOT_RUN_BLOCKED_BY_G0",
                   "G3": "NOT_RUN", "G4": "NOT_RUN", "position_impact": 0})
    write_once(root / OUT / "daily_quality.csv", csv_bytes(daily))
    write_once(root / OUT / "bar_vwap_defects.csv", csv_bytes(defects))
    result["outputs"] = [identity(root, f"{OUT}/{p}") for p in ["daily_quality.csv", "bar_vwap_defects.csv"]]
    write_once(root / OUT / "source_adjudication.json", result)
    return result
