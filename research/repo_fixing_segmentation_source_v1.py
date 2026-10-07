"""从中国货币网同一原始响应提取FR007与FDR007，保留明确的口径和时钟。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_repo_fixing_segmentation_source_v1"
OLD = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/fdr007_daily_2015_2026.parquet"
URL = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/FrrHis"
METHOD = "https://www.chinamoney.com.cn/chinese/bkfrr/"


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ["raw", "code"]:
        (OUT / folder).mkdir(exist_ok=True)
    previous = pd.read_parquet(OLD)
    paths = [OLD, Path(__file__), *[ROOT / name for name in previous.raw_path.unique()]]
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_REPO_FIXING_SEGMENTATION_SOURCE_V1",
         "fields": ["FR007", "FDR007"], "definition": "同日frValueMap中的FR007减FDR007，单位百分点。两者为上午成交定盘，不是R007减DR007全天加权差，也不是纯非银利率。",
         "archive_rule": "沿用已保存官方JSON，缺失---保留未知，FDR推出以前不回填。",
         "availability": "官方现行方法11:30起发布；历史重建使用当日12:00的保守计划时点，研究下单只在下一个A股开盘。历史首次实际送达未经认证。",
         "network": "仅普通官方HTTP请求：方法页面GET和2026-08-26至2026-09-24的FrrHis只读POST；15秒超时，无自动重试。失败保留且不填值。",
         "planned_requests": 2, "existing_research_unchanged": True, "returns_read": False,
         "accounts": 0, "orders_authorized": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}}, True)
    print("FR007与FDR007同日差值的来源定义已固定，未计算候选收益。", flush=True)


def retrieve(name, method, url, params=None):
    receipt = {"at": now(), "method": method, "url": url, "params": params}
    try:
        response = requests.request(method, url, params=params, timeout=15,
                                    headers={"User-Agent": "Mozilla/5.0", "Referer": METHOD})
        path = OUT / "raw" / name
        path.write_bytes(response.content)
        receipt.update(status_code=response.status_code, completed_at=now(),
                       response_url=response.url, path=path.relative_to(ROOT).as_posix(), sha256=digest(path))
        response.raise_for_status()
        receipt["success"] = True
    except Exception as exc:
        receipt.update(success=False, error=f"{type(exc).__name__}: {exc}")
    save(OUT / "raw" / (name + ".receipt.json"), receipt, True)
    return receipt


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    method = retrieve("official_method.html", "GET", METHOD)
    extension = retrieve("fixings_20260826_20260924.json", "POST", URL,
                         {"lang": "CN", "startDate": "2026-08-26", "endDate": "2026-09-24"})
    old = pd.read_parquet(OLD)
    sources = [(ROOT / path, False) for path in old.raw_path.unique()]
    if extension["success"]:
        sources.append((ROOT / extension["path"], True))
    rows, excluded = [], []
    for path, is_new in sources:
        payload = read(path)
        if str(payload.get("head", {}).get("rep_code")) != "200":
            excluded.append({"path": path.relative_to(ROOT).as_posix(), "reason": "官方响应未成功"})
            continue
        for record in payload.get("records", []):
            day = pd.Timestamp(record["lfiProducDate"])
            if day > pd.Timestamp("2026-09-24"):
                continue
            values = record.get("frValueMap", {})
            a, b = values.get("FR007"), values.get("FDR007")
            if a in [None, "", "---"] or b in [None, "", "---"]:
                excluded.append({"date": day, "reason": "定盘数值缺失，未插值"})
                continue
            a, b = float(a), float(b)
            assert np.isfinite([a, b]).all()
            rows.append({"date": day, "fr007_percent": a, "fdr007_percent": b,
                         "segmentation_pp": a-b, "available_at": day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=12),
                         "raw_path": path.relative_to(ROOT).as_posix(), "raw_sha256": digest(path),
                         "source_url": URL, "new_request": is_new})
    frame = pd.DataFrame(rows).sort_values("date", kind="stable")
    if frame.date.duplicated().any():
        for _, part in frame[frame.date.duplicated(False)].groupby("date"):
            assert part[["fr007_percent", "fdr007_percent"]].drop_duplicates().shape[0] == 1
        frame = frame.drop_duplicates("date", keep="first")
    frame = frame.reset_index(drop=True)
    shared = old[["date", "first_release_value"]].merge(frame, on="date", how="left", validate="one_to_one")
    assert len(shared) == len(old) and shared.fdr007_percent.notna().all()
    np.testing.assert_allclose(shared.first_release_value, shared.fdr007_percent, atol=0, rtol=0)
    frame.to_parquet(OUT / "fixing_segmentation.parquet", index=False)
    save(OUT / "excluded_rows.json", excluded, True)
    save(OUT / "result.json", {"at": now(), "study_id": "510300_REPO_FIXING_SEGMENTATION_SOURCE_V1",
         "status": "SOURCE_READY_HISTORICAL_DELIVERY_UNVERIFIED", "rows": len(frame),
         "first_date": frame.date.min(), "last_date": frame.date.max(), "old_fdr_values_exact": len(shared),
         "additional_rows": int(frame.new_request.sum()), "new_requests": 2,
         "method_request_success": method["success"], "extension_request_success": extension["success"],
         "historical_first_vintage_verified": False, "field_substitution": False,
         "new_models": 0, "new_accounts": 0, "independent_forward_observations": 0,
         "goal_achieved": False}, True)
    print(f"定盘差值来源已提取{len(frame)}日；原{len(shared)}个FDR值精确一致，新增{int(frame.new_request.sum())}日。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FR/FDR同上午定盘差值来源")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
