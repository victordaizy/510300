"""按接口实际20行分页补齐唯一缺页，复用已收到的新浪和东财首页。"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, digest, now
import research.etf_nav_recent_source_v1 as original
import research.etf_discount_compensation_probe_v1 as probe

OUT = ROOT / "reports/research/510300_etf_nav_recent_source_completion_v1"
STUDY = "510300_ETF_NAV_RECENT_SOURCE_COMPLETION_V1"


def run():
    original_result = read(original.OUT / "result.json")
    assert original_result["status"] == "INCOMPLETE_NO_COMMON_NEW_NAV_ADMITTED"
    first = json.loads((original.OUT / "raw/eastmoney.response").read_bytes())
    rows = (first.get("Data") or {}).get("LSJZList") or []
    total, page_size = int(first["TotalCount"]), int(first["PageSize"])
    assert total == 34 and page_size == 20 and len(rows) == page_size
    paths = [Path(__file__), original.OUT / "result.json", original.OUT / "sina.parquet",
             original.OUT / "raw/eastmoney.response", probe.CURRENT]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    config = original.REQUESTS["eastmoney"]
    params = {**config["params"], "pageSize": "20", "pageIndex": "2"}
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    save(OUT / "completion_plan.json", {"at": now(), "study_id": STUDY,
        "reason": "东财把请求pageSize100改为实际20行，总数34；补第二页14行。原两请求、未完成状态与响应保留。",
        "additional_requests": 1, "retries": 0, "url": config["url"], "params": params,
        "saved_inputs": hashes, "strategy_or_parameter_changes": 0, "orders_authorized": False}, True)
    receipt = {"requested_at": now(), "url": config["url"], "params": params}
    response = requests.get(config["url"], params=params, headers=config["headers"], timeout=30)
    raw_path = OUT / "raw/eastmoney_page2.response"
    raw_path.write_bytes(response.content)
    receipt.update(received_at=now(), status_code=response.status_code, response_sha256=digest(raw_path), response_bytes=len(response.content))
    save(OUT / "raw/eastmoney_page2.receipt.json", receipt, True)
    response.raise_for_status()
    body = response.json()
    second = (body.get("Data") or {}).get("LSJZList") or []
    assert int(body["TotalCount"]) == total and int(body["PageIndex"]) == 2 and len(second) == total - page_size
    east = pd.DataFrame(rows + second)[["FSRQ", "DWJZ"]].rename(columns={"FSRQ": "date", "DWJZ": "unit_nav_eastmoney"})
    east["date"] = pd.to_datetime(east.date).astype("datetime64[ns]")
    east["unit_nav_eastmoney"] = pd.to_numeric(east.unit_nav_eastmoney, errors="raise")
    assert east.date.is_unique and east.date.between(original.START, original.END).all()
    assert east.unit_nav_eastmoney.gt(0).all() and np.isfinite(east.unit_nav_eastmoney).all()
    sina = pd.read_parquet(original.OUT / "sina.parquet").rename(columns={"unit_nav": "unit_nav_sina"})
    joined = east.merge(sina, on="date", how="outer", indicator=True, validate="one_to_one").sort_values("date")
    joined["difference"] = joined.unit_nav_eastmoney - joined.unit_nav_sina
    joined["agreed"] = joined._merge.eq("both") & joined.difference.abs().le(1e-8)
    old = pd.read_parquet(probe.CURRENT, columns=["date", "unit_nav"])
    joined = joined.merge(old.rename(columns={"unit_nav": "old_unit_nav"}), on="date", how="left", validate="one_to_one")
    joined["old_overlap_equal"] = joined.old_unit_nav.isna() | (joined.unit_nav_eastmoney - joined.old_unit_nav).abs().le(1e-8)
    joined["admitted"] = joined.agreed & joined.old_overlap_equal
    joined["unit_nav"] = joined.unit_nav_eastmoney.where(joined.admitted)
    joined.to_parquet(OUT / "new_nav_evidence.parquet", index=False)
    market = pd.read_parquet(probe.market_source.OUT / "inputs/market.parquet", columns=["date", "close"])
    admitted = joined[joined.admitted]
    matched = admitted.merge(market, on="date", how="left", validate="one_to_one")
    matched["new_nav_date"] = matched.date.gt(old.date.max())
    matched.to_parquet(OUT / "matched_new_nav.parquet", index=False)
    new = matched[matched.new_nav_date & matched.close.notna()].copy()
    summaries = {}
    for name, cost in probe.market_source.distribution.COSTS.items():
        scenarios = pd.DataFrame([{"date": row.date, "close": row.close, "unit_nav": row.unit_nav,
                                  **probe.space(row.close, row.unit_nav, cost)} for row in new.itertuples()])
        scenarios.to_parquet(OUT / (name + "_new_space.parquet"), index=False)
        summaries[name] = {"new_matched_dates": len(scenarios), "static_positive_days": int(scenarios.static_positive.sum()) if len(scenarios) else 0}
    for path, sha in hashes.items():
        assert digest(ROOT / path) == sha, path
    result = {"at": now(), "study_id": STUDY, "original_study_id": original.STUDY,
        "status": "NEW_NAV_EXTENSION_COMPLETED" if joined.admitted.all() else "PARTIAL_PROVIDER_AGREEMENT",
        "new_requests_this_completion": 1, "total_public_requests_including_original": 3,
        "admitted_rows": len(admitted), "overlap_rows": int(joined.old_unit_nav.notna().sum()),
        "overlap_difference_rows": int((joined.old_unit_nav.notna() & ~joined.old_overlap_equal).sum()),
        "provider_difference_or_missing_rows": int((~joined.agreed).sum()),
        "new_matched_nav_dates": len(new), "new_nav_without_market": int((matched.new_nav_date & matched.close.isna()).sum()),
        "first_new_matched_date": new.date.min(), "last_new_matched_date": new.date.max(),
        "new_discount_days": int(new.close.lt(new.unit_nav).sum()), "new_static_scenarios": summaries,
        "requested_end_date": original.END, "latest_returned_nav_date": admitted.date.max(),
        "original_files_unchanged": True, "source_first_vintage_verified": False,
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(f"唯一缺页补齐：双源同意{len(admitted)}行，新增可匹配报价{len(new)}日；压力静态空间为正{summaries['STRESS']['static_positive_days']}日。", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        save(OUT / "RUN_FAILURE.json", {"at": now(), "type": type(exc).__name__, "message": str(exc)}, True)
        raise
