"""用固定公司年份探查免费多机构原件目录，不使用接口滚动EPS字段。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import save, now
from research.forward_eps_guosen_history_v1 import identity

OUT = ROOT / "reports/research/510300_forward_eps_multi_institution_probe_v1"
RAW = ROOT / "data/raw/510300_forward_eps_multi_institution_probe_v1"


def one(year: int) -> dict:
    params = {"code": "600519", "orgCode": "*", "qType": 0, "pageSize": 50, "pageNo": 1,
              "beginTime": f"{year}-01-01", "endTime": f"{year}-12-31", "industryCode": "*",
              "industry": "*", "rating": "*", "ratingChange": "*", "fields": ""}
    r = requests.get("https://reportapi.eastmoney.com/report/list", params=params,
                     headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/report/"},
                     timeout=(12, 30))
    if r.status_code in (401, 403, 429):
        raise PermissionError(f"公开来源访问限制：{r.status_code}")
    r.raise_for_status()
    d = r.json()
    assert isinstance(d.get("data"), list)
    path = RAW / f"{year}_first_page.json"
    path.write_bytes(r.content)
    receipt = {"retrieved_at": now(), "url": r.url, "params": params,
               "path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(r.content).hexdigest(),
               "bytes": len(r.content), "historical_eps_fields_not_used": True}
    save(OUT / f"{year}_receipt.json", receipt, exclusive=True)
    rows = d["data"]
    assert all(x["stockCode"] == "600519" and str(x["publishDate"]).startswith(str(year)) for x in rows)
    columns = ["stockCode", "stockName", "infoCode", "publishDate", "orgCode", "orgName", "orgSName", "title", "attachPages"]
    summary = {"year": year, "hits": int(d["hits"]), "total_pages": int(d["TotalPage"]),
               "sampled_first_page_reports": len(rows),
               "institutions_in_first_page": sorted({(x["orgCode"], x.get("orgSName") or x.get("orgName")) for x in rows}),
               "reports": [{k: x.get(k) for k in columns} for x in rows]}
    save(OUT / f"{year}_directory_sample.json", summary, exclusive=True)
    print("多机构免费目录样例完成", year, "全年目录条数", summary["hits"], "首页机构数", len(summary["institutions_in_first_page"]), flush=True)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    RAW.mkdir(parents=True, exist_ok=False)
    save(OUT / "protocol.json", {"registered_at": now(), "fixed_security": "600519.SH", "years": [2018, 2021, 2024],
         "selection_reason": "现有公开原件例子的贵州茅台，检查2022年前及近期的多机构公开资料是否存在；不是代表全指数的策略样本。",
         "first_page_only": True, "page_size": 50, "no_returns_read": True,
         "directory_eps_fields_used_as_historical_facts": False, "budget_cny": 0,
         "script": identity(Path(__file__))}, exclusive=True)
    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(one, year): year for year in [2018, 2021, 2024]}
        for f in as_completed(pending):
            try:
                completed.append(f.result())
            except Exception as exc:
                errors.append({"year": pending[f], "error_type": type(exc).__name__, "error": str(exc)})
    result = {"study_id": "510300_FORWARD_EPS_MULTI_INSTITUTION_PROBE_V1", "completed_at": now(),
              "status": "FREE_DIRECTORY_PROBE_COMPLETE" if not errors else "FREE_DIRECTORY_PROBE_PARTIAL",
              "years": [{k: v for k, v in x.items() if k != "reports"} for x in sorted(completed, key=lambda x: x["year"])],
              "errors": errors, "new_eps_facts": 0, "new_accounts_generated": 0, "goal_achieved": False}
    save(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
