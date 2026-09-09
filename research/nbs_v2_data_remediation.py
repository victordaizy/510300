"""复核 NBS 数据缺陷；额外五分钟诊断不得替代冻结的逐分钟硬门。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import dotenv_values

from research.nbs_v2_common import (
    ContractError, committed, csv_bytes, git, identity, load_json, now, sha, verify, write_once,
)
from research.stk_mins_source_admission_v1 import TushareProxyClient
from research.stk_mins_source_admission_v2 import measure, normalize

CONFIG = "config/510300_nbs_v2_data_remediation_20260905.json"
OUT = "reports/data_quality/510300_nbs_v2_data_remediation_20260905"
RAW = "data/raw/tushare/510300_nbs_v2_data_remediation_20260905"
SCOPE = [CONFIG, "research/nbs_v2_data_remediation.py",
         "scripts/run_510300_nbs_v2_data_remediation.py",
         "tests/test_510300_nbs_v2_data_remediation.py"]
MANIFEST = f"{OUT}/precheck_manifest.json"


def windows(source: dict, model: dict) -> dict:
    result = dict(source["windows"])
    result.update({f"shift_{name}": model["placebos"]["B"][f"{name}_labels"]
                   for name in ["pre", "reaction", "entry", "exit"]})
    if len(result) != 8 or any(len(labels) != 5 or len(set(labels)) != 5
                              for labels in result.values()):
        raise ContractError("仅接受既有主实验与已授权安慰剂 B 的八个固定五分钟窗口")
    return result


def aggregate_measurement(frame: pd.DataFrame, dates: pd.DatetimeIndex,
                          fixed: dict, tolerance: float) -> pd.DataFrame:
    """只计算各窗口的量额和边界，不计算任意两个价格的比值或收益。"""
    f = normalize(frame)
    rows = []
    for name, labels in fixed.items():
        part = f.loc[f.clock.isin(labels)]
        grouped = part.groupby("date")
        totals = grouped.agg(volume_shares=("vol", "sum"), amount_cny=("amount", "sum"),
                             minimum_low=("low", "min"), maximum_high=("high", "max"),
                             rows=("clock", "size"))
        unique = grouped.clock.nunique()
        valid_fields = grouped.apply(
            lambda g: bool(np.isfinite(g[["vol", "amount", "low", "high"]]).all().all()
                           and (g.vol >= 0).all() and (g.amount >= 0).all()
                           and ((g.vol == 0) == (g.amount == 0)).all()
                           and (g.low > 0).all() and (g.high >= g.low).all()),
            include_groups=False,
        )
        totals = totals.reindex(dates)
        totals["window"] = name
        totals["complete"] = (totals.rows.eq(5) & unique.reindex(dates).eq(5)
                              & valid_fields.reindex(dates, fill_value=False))
        totals["positive_totals"] = (totals.volume_shares > 0) & (totals.amount_cny > 0)
        totals["vwap"] = totals.amount_cny / totals.volume_shares.where(totals.positive_totals)
        totals["inside_window_range"] = (totals.vwap.ge(totals.minimum_low - tolerance)
                                         & totals.vwap.le(totals.maximum_high + tolerance))
        totals["diagnostic_pass"] = (totals.complete & totals.positive_totals
                                      & totals.inside_window_range)
        totals.index.name = "date"
        rows.append(totals.reset_index())
    return pd.concat(rows, ignore_index=True)


def freeze(root: Path) -> dict:
    cfg = load_json(root / CONFIG)
    original = load_json(root / cfg["active_source_manifest"])
    verify(root, original["identities"])
    paths = SCOPE + [cfg["active_source_manifest"], cfg["active_model_contract"],
                     "research/stk_mins_source_admission_v1.py",
                     "reports/data_quality/510300_stk_mins_source_admission_v2/source_adjudication.json",
                     "reports/research/510300_nbs_1000_negative_information_drift_v2/execution_receipt.json"]
    receipt = {"state": "FROZEN_DATA_REMEDIATION_NO_RETURN_AUTHORITY", "generated_at": now(),
               "identities": [identity(root, p) for p in paths],
               "event_return_values_read": False, "position_impact": 0}
    write_once(root / MANIFEST, receipt)
    return {"state": receipt["state"], "files": len(paths)}


def validate(root: Path) -> tuple[dict, dict, dict]:
    verify(root, load_json(root / MANIFEST)["identities"])
    committed(root, SCOPE + [MANIFEST])
    cfg = load_json(root / CONFIG)
    verify(root, load_json(root / cfg["active_source_manifest"])["identities"])
    source = yaml.safe_load((root / cfg["active_source_contract"]).read_text("utf-8"))
    model = yaml.safe_load((root / cfg["active_model_contract"]).read_text("utf-8"))
    return cfg, source, model


def audit(root: Path) -> dict:
    _, cfg, model = validate(root)
    output = root / OUT / "measurement_recheck.json"
    if output.exists():
        raise ContractError("数据复核已有不可变回执，请直接读取")
    acq = load_json(root / cfg["inputs"]["acquisition"])
    refs = {}
    for rec in acq["records"][:2]:
        receipt = load_json(root / rec["receipt_relative_path"])
        refs[rec["api"]] = pd.read_parquet(root / receipt["normalized_relative_path"])
    cal = refs["trade_cal"]
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1),
                                                 "cal_date"].astype(str))).sort_values()
    minute = pd.read_parquet(root / cfg["inputs"]["minute"])
    independent = pd.read_parquet(root / cfg["inputs"]["independent_minute"])
    events = pd.read_parquet(root / cfg["inputs"]["parent_ledger"])
    strict, _, defects = measure(minute, refs["fund_daily"], dates, independent, events, cfg)
    strict["checks"]["bar_end_evidence"] = load_json(root / cfg["inputs"]["timestamp_evidence"])["pass"]
    strict["source_pass"] = all(strict["checks"].values())
    q = cfg["quality_gates"]
    fixed = windows(cfg, model)
    table = aggregate_measurement(minute, dates, fixed,
                                  q["bar_vwap_price_tolerance_cny"] + q["binary_float_tolerance_cny"])
    overlap = dates[(dates >= q["independent_overlap_start"]) & (dates <= q["independent_overlap_end"])]
    old = aggregate_measurement(independent, overlap, fixed,
                                q["bar_vwap_price_tolerance_cny"] + q["binary_float_tolerance_cny"])
    pair = table.merge(old, on=["date", "window"], suffixes=("_new", "_old"))
    match = (pair.complete_new & pair.complete_old
             & np.isclose(pair.vwap_new, pair.vwap_old,
                           atol=q["independent_vwap_absolute_tolerance_cny"],
                           rtol=q["independent_vwap_relative_tolerance"]))
    m, legacy = normalize(minute), normalize(independent)
    positive_defects = defects.loc[defects.vol.gt(0) & defects.amount.gt(0)].copy()
    positive_defects["nominal_excess_cny"] = np.maximum(
        positive_defects.amount - positive_defects.high * positive_defects.vol,
        positive_defects.low * positive_defects.vol - positive_defects.amount,
    )
    positive_defects["tick_plus_one_yuan_still_outside"] = (
        positive_defects.nominal_excess_cny > .001 * positive_defects.vol + 1 + 1e-8)
    p = positive_defects.merge(legacy, on=["ts_code", "trade_time"], suffixes=("_new", "_old"))
    raw_rows = []
    by_month = positive_defects.groupby(positive_defects.trade_time.dt.strftime("%Y%m"))
    for month, group in by_month:
        rec = next(r for r in acq["records"] if r["api"] == "stk_mins" and r["key"] == f"month={month}")
        receipt = load_json(root / rec["receipt_relative_path"])
        response = load_json(root / receipt["raw_relative_path"])
        raw = pd.DataFrame(response["data"]["items"], columns=response["data"]["fields"])
        raw.trade_time = pd.to_datetime(raw.trade_time)
        matched = group.merge(raw, on=["ts_code", "trade_time"], suffixes=("_local", "_raw"), validate="one_to_one")
        if len(matched) != len(group):
            raise ContractError("异常分钟缺失对应原始响应行")
        for field in ["vol", "amount", "low", "high"]:
            matched[f"{field}_unchanged_from_raw"] = matched[f"{field}_local"].eq(matched[f"{field}_raw"])
        matched["receipt_path"] = rec["receipt_relative_path"]
        raw_rows.append(matched[["ts_code", "trade_time", "receipt_path"]
                                + [f"{c}_unchanged_from_raw" for c in ["vol", "amount", "low", "high"]]])
    raw_proof = pd.concat(raw_rows, ignore_index=True)
    zero_pair = m.vol.eq(0) & m.amount.eq(0)
    result = {"state": "PASS_ACTIVE_V2_SOURCE_CONTRACT" if strict["source_pass"]
                       else "BLOCKED_ACTIVE_V2_BAR_GATE_UNRESOLVED",
              "generated_at": now(), "freeze_commit": git(root, "rev-parse", "HEAD"),
              "active_contract_result": strict,
              "additional_fixed_window_diagnostic": {
                  "active_gate_substitution_authorized": False,
                  "window_count": len(table), "passing_windows": int(table.diagnostic_pass.sum()),
                  "per_window": table.groupby("window").diagnostic_pass.agg(["count", "sum"]).to_dict("index"),
                  "same_provider_independent_capture_windows": len(pair),
                  "same_provider_independent_capture_matching_windows": int(match.sum()),
                  "independent_provider_claimed": False,
              },
              "positive_volume_outside_bar_rows": len(positive_defects),
              "zero_volume_zero_amount_rows": int(zero_pair.sum()),
              "inconsistent_zero_volume_amount_rows": int((m.vol.eq(0) ^ m.amount.eq(0)).sum()),
              "raw_response_defect_rows_verified": len(raw_proof),
              "all_defects_present_unchanged_in_raw_response": bool(raw_proof.filter(like="unchanged_from_raw").all().all()),
              "legacy_capture_defect_rows": len(p),
              "legacy_capture_defect_fields_identical": {c: bool(p[f"{c}_new"].eq(p[f"{c}_old"]).all())
                                                          for c in ["vol", "amount", "low", "high"]},
              "one_tick_plus_one_yuan_not_explained_rows": int(positive_defects.tick_plus_one_yuan_still_outside.sum()),
              "maximum_nominal_range_excess_cny": float(positive_defects.nominal_excess_cny.max()),
              "source_data_modified": False, "event_return_values_read": False,
              "event_labels_created": 0, "model_training_run": False, "position_impact": 0}
    outputs = {"fixed_window_measurements.csv": table, "positive_bar_defects.csv": positive_defects,
               "raw_response_defect_proof.csv": raw_proof}
    for filename, frame in outputs.items():
        write_once(root / OUT / filename, csv_bytes(frame))
    result["outputs"] = [identity(root, f"{OUT}/{name}") for name in outputs]
    write_once(output, result)
    return result


def probe(root: Path) -> dict:
    cfg, source, _ = validate(root)
    output = root / OUT / "supplier_refetch_receipt.json"
    if output.exists():
        raise ContractError("供应商重取已有不可变回执，请直接读取")
    claim = root / OUT / "supplier_refetch_claim.json"
    if claim.exists():
        raise ContractError("供应商重取已领取，不得静默重复请求")
    write_once(claim, {"started_at": now(), "dates": cfg["probe_dates"], "attempts_per_date": 1})
    env = dotenv_values(root / ".env")
    token = os.getenv("TUSHARE_PROXY_TOKEN") or env.get("TUSHARE_PROXY_TOKEN") or ""
    expiry = os.getenv("TUSHARE_PROXY_TOKEN_EXPIRES_AT") or env.get("TUSHARE_PROXY_TOKEN_EXPIRES_AT")
    result = {"generated_at": now(), "records": [], "event_return_values_read": False,
              "credential_persisted": False, "source_data_replaced": False}
    if not token:
        result["state"] = "BLOCKED_NO_EXISTING_EPHEMERAL_CREDENTIAL"
    elif not expiry or pd.Timestamp(expiry).tzinfo is None or pd.Timestamp(expiry) <= pd.Timestamp(now()):
        result["state"] = "BLOCKED_EXPIRED_OR_UNVERIFIABLE_CREDENTIAL_NO_NETWORK_REQUEST"
    else:
        oldcfg = yaml.safe_load((root / "config/510300_stk_mins_source_admission_v1.yaml").read_text("utf-8"))
        client = TushareProxyClient(oldcfg["source_contract"]["endpoint"], token, .65, 15,
                                   cfg["probe_read_timeout_seconds"], 1, [])
        minute = normalize(pd.read_parquet(root / source["inputs"]["minute"]))
        for day in cfg["probe_dates"]:
            params = {"ts_code": "510300.SH", "freq": "1min", "start_date": f"{day} 09:00:00",
                      "end_date": f"{day} 16:00:00"}
            response = client.call(cfg["probe_api"], params)
            rec = {"date": day, "parameters": params, "request_started_at": response.request_started_at,
                   "response_received_at": response.response_received_at,
                   "http_status": response.http_status, "business_code": response.business_code,
                   "success": response.success, "response_sha256": response.response_sha256}
            if not response.success:
                rec["error_message"] = response.error_message
                result["records"].append(rec)
                result["state"] = "BLOCKED_SUPPLIER_PROBE_FAILED_STOPPED"
                break
            raw_path = f"{RAW}/{day}/response.json"
            write_once(root / raw_path, response.raw_bytes)
            fresh = normalize(response.frame)
            old = minute.loc[minute.date.eq(pd.Timestamp(day))]
            pairs = old.merge(fresh, on=["ts_code", "trade_time"], suffixes=("_old", "_fresh"),
                              how="outer", indicator=True, validate="one_to_one")
            rec.update({"raw_response": identity(root, raw_path), "fresh_rows": len(fresh),
                        "old_rows": len(old), "matching_keys": int(pairs._merge.eq("both").sum()),
                        "fields_all_identical": {c: bool(pairs[f"{c}_old"].eq(pairs[f"{c}_fresh"]).all())
                                                 for c in ["vol", "amount", "low", "high"]}})
            defined = fresh.vol.gt(0) & fresh.amount.gt(0)
            vwap = fresh.amount / fresh.vol.where(defined)
            outside = defined & (vwap.lt(fresh.low - .001 - 1e-12) | vwap.gt(fresh.high + .001 + 1e-12))
            rec["fresh_outside_bar_rows"] = int(outside.sum())
            result["records"].append(rec)
        else:
            result["state"] = "SUPPLIER_PROBES_COMPLETE_CORRECTION_PRESENT" if any(
                not all(r["fields_all_identical"].values()) for r in result["records"]
            ) else "SUPPLIER_PROBES_COMPLETE_NO_CORRECTIONS"
    write_once(output, result)
    return result
