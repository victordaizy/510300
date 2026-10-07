"""有界补取近期510300单位净值，保留原历史文件及采集时钟。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, digest, now
import research.etf_discount_compensation_probe_v1 as probe

OUT = ROOT / "reports/research/510300_etf_nav_recent_source_v1"
STUDY = "510300_ETF_NAV_RECENT_SOURCE_V1"
START, END = "2026-08-10", "2026-09-25"
REQUESTS = {
    "eastmoney": {"url": "https://api.fund.eastmoney.com/f10/lsjz",
        "params": {"fundCode": "510300", "pageIndex": "1", "pageSize": "100", "startDate": START, "endDate": END},
        "headers": {"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/jjjz_510300.html"}},
    "sina": {"url": "https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/CaihuiFundInfoService.getNav",
        "params": {"symbol": "510300", "datefrom": START, "dateto": END, "page": "1", "num": "100"},
        "headers": {"User-Agent": "Mozilla/5.0"}},
}


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("近期净值请求已经固定，不能重复覆盖。")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "range": [START, END],
        "requests": REQUESTS, "maximum_requests": 2, "retries": 0, "timeout_seconds_each": 30,
        "purpose": "补足最近已核对报价之后缺少的NAV日期，核对ETF折价补偿结论；只采集单位净值，不恢复PCF/IOPV计划任务。",
        "original_inputs": {str(probe.CURRENT.relative_to(ROOT)): digest(probe.CURRENT),
                            str((probe.OUT / "result.json").relative_to(ROOT)): digest(probe.OUT / "result.json")},
        "source_hierarchy": "本次基金官网页面经网页工具未能取得内容；使用原历史已有的东财/新浪公开接口，两者匹配不称独立官方首版认证。",
        "admission": "日期去重且两来源单位净值相差不超过1e-8，重叠日期同旧文件一致；缺失或差异分别保留，不填补。",
        "publication_time": "只认证本次请求和收取时间；未认证每个历史日期首次公告/送达时钟。",
        "outside_market_end": "9月25日若有净值但没有已核对价格，只保存净值，不扩展报价或预测标签。",
        "parameter_changes": 0, "new_accounts": 0, "new_fits": 0, "goal_achieved": False, "orders_authorized": False}, True)
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
                               "protocol_sha256": digest(OUT / "protocol.json")}, True)
    print("近期净值两项公开请求已固定，范围与上限明确。", flush=True)


def fetch(name):
    config = REQUESTS[name]
    receipt = {"provider": name, "requested_at": now(), **config}
    try:
        response = requests.get(config["url"], params=config["params"], headers=config["headers"], timeout=30)
        path = OUT / "raw" / (name + ".response")
        path.write_bytes(response.content)
        receipt.update(received_at=now(), status_code=response.status_code, response_sha256=digest(path),
                       response_bytes=len(response.content), final_url=response.url)
        response.raise_for_status()
        body = response.json()
        if name == "eastmoney":
            records = (body.get("Data") or {}).get("LSJZList") or []
            total = int(body.get("TotalCount") or 0)
            date_col, nav_col = "FSRQ", "DWJZ"
        else:
            result = body.get("result") or {}
            if int((result.get("status") or {}).get("code", -1)) != 0:
                raise ValueError("新浪净值返回非成功状态。")
            data = result.get("data") or {}
            records, total = data.get("data") or [], int(data.get("total_num") or 0)
            date_col, nav_col = "fbrq", "jjjz"
        if total > 100 or len(records) != total or not records:
            raise ValueError(f"单页范围不完整或为空：total={total}, rows={len(records)}。")
        frame = pd.DataFrame(records)[[date_col, nav_col]].rename(columns={date_col: "date", nav_col: "unit_nav"})
        frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
        frame["unit_nav"] = pd.to_numeric(frame.unit_nav, errors="raise")
        assert frame.date.between(START, END).all() and frame.date.is_unique
        assert frame.unit_nav.gt(0).all() and np.isfinite(frame.unit_nav).all()
        frame = frame.sort_values("date").reset_index(drop=True)
        frame.to_parquet(OUT / (name + ".parquet"), index=False)
        receipt.update(status="RECEIVED_COMPLETE_REQUESTED_RANGE", rows=len(frame), first_date=frame.date.min(), last_date=frame.date.max())
        save(OUT / "raw" / (name + ".receipt.json"), receipt, True)
        print(f"{name}近期单位净值收到{len(frame)}行。", flush=True)
        return frame, receipt
    except Exception as exc:
        receipt.update(status="REQUEST_OR_PARSE_FAILED", ended_at=now(), error_type=type(exc).__name__, error=str(exc))
        save(OUT / "raw" / (name + ".receipt.json"), receipt, True)
        print(f"{name}近期净值未完成：{type(exc).__name__}。", flush=True)
        return None, receipt


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = dict(zip(REQUESTS, pool.map(fetch, REQUESTS)))
    receipts = [item[1] for item in responses.values()]
    result = {"at": now(), "study_id": STUDY, "requests_made": 2, "provider_receipts": receipts,
        "source_first_vintage_verified": False, "new_accounts": 0, "new_fits": 0,
        "independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    if any(item[0] is None for item in responses.values()):
        result.update(status="INCOMPLETE_NO_COMMON_NEW_NAV_ADMITTED", admitted_rows=0)
        save(OUT / "result.json", result, True)
        return
    left, right = responses["eastmoney"][0], responses["sina"][0]
    joined = left.merge(right, on="date", how="outer", suffixes=("_eastmoney", "_sina"), indicator=True, validate="one_to_one")
    joined["difference"] = joined.unit_nav_eastmoney - joined.unit_nav_sina
    joined["agreed"] = joined._merge.eq("both") & joined.difference.abs().le(1e-8)
    old = pd.read_parquet(probe.CURRENT, columns=["date", "unit_nav"])
    joined = joined.merge(old.rename(columns={"unit_nav": "old_unit_nav"}), on="date", how="left", validate="one_to_one")
    joined["old_overlap_equal"] = joined.old_unit_nav.isna() | (joined.unit_nav_eastmoney - joined.old_unit_nav).abs().le(1e-8)
    joined["admitted"] = joined.agreed & joined.old_overlap_equal
    joined["unit_nav"] = joined.unit_nav_eastmoney.where(joined.admitted)
    joined.to_parquet(OUT / "new_nav_evidence.parquet", index=False)
    market = pd.read_parquet(probe.market_source.OUT / "inputs/market.parquet", columns=["date", "close"])
    admitted = joined[joined.admitted].copy()
    matched = admitted.merge(market, on="date", how="left", validate="one_to_one")
    matched["new_nav_date"] = matched.date.gt(old.date.max())
    matched.to_parquet(OUT / "matched_new_nav.parquet", index=False)
    new = matched[matched.new_nav_date & matched.close.notna()].copy()
    summaries = {}
    for name, cost in probe.market_source.distribution.COSTS.items():
        rows = pd.DataFrame([{"date": row.date, "close": row.close, "unit_nav": row.unit_nav,
                              **probe.space(row.close, row.unit_nav, cost)} for row in new.itertuples()])
        rows.to_parquet(OUT / (name + "_new_space.parquet"), index=False)
        summaries[name] = {"new_matched_dates": len(rows), "static_positive_days": int(rows.static_positive.sum()) if len(rows) else 0}
    result.update(status="NEW_NAV_EXTENSION_COMPLETED" if joined.admitted.all() else "PARTIAL_PROVIDER_AGREEMENT",
        admitted_rows=len(admitted), overlap_rows=int(joined.old_unit_nav.notna().sum()),
        overlap_difference_rows=int((joined.old_unit_nav.notna() & ~joined.old_overlap_equal).sum()),
        provider_difference_or_missing_rows=int((~joined.agreed).sum()),
        new_matched_nav_dates=len(new), new_nav_without_market=int((matched.new_nav_date & matched.close.isna()).sum()),
        first_new_matched_date=new.date.min(), last_new_matched_date=new.date.max(),
        new_discount_days=int(new.close.lt(new.unit_nav).sum()), new_static_scenarios=summaries)
    for path, sha in read(OUT / "protocol.json")["original_inputs"].items():
        assert digest(ROOT / path) == sha
    save(OUT / "result.json", result, True)
    print(f"近期净值补充完成：新增{len(new)}个与已核对报价匹配的日期，旧净值文件未改动。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            save(OUT / "RUN_FAILURE.json", {"at": now(), "type": type(exc).__name__, "message": str(exc)}, True)
            raise
