"""核查二手住宅价格的原表、发布时钟与五年换基说明。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.industrial_receivable_monthly_source_v1 as transport
from research.exchange_bank_funding_gap_daily_v1 import adapt
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_housing_collateral_source_probe_v1"
STUDY = "510300_HOUSING_COLLATERAL_SOURCE_PROBE_V1"
SOURCES = {
    "2018-01": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1899851.html",
    "2021-01": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900997.html",
    "2025-12": "https://www.stats.gov.cn/sj/zxfb/202601/t20260119_1962319.html",
    "2026-01": "https://www.stats.gov.cn/sj/zxfb/202602/t20260213_1962617.html",
    "2026-08": "https://www.stats.gov.cn/sj/zxfb/202609/t20260915_1965304.html",
}
FETCH = adapt(transport.fetch, {"OUT": OUT, "_lock": Lock(), "_request_count": len(list((OUT / "requests").glob("*.json")))})
REQUEST = adapt(transport.request_job, {"fetch": FETCH})


def describe(month, receipt):
    soup = transport.soup_for(receipt)
    compact = re.sub(r"\s+", "", soup.get_text())
    at = compact.find("附注")
    tables = []
    for number, table in enumerate(soup.find_all("table")):
        rows = []
        for row in table.find_all("tr"):
            cells = [re.sub(r"\s+", "", c.get_text()) for c in row.find_all(["td", "th"], recursive=False)]
            if cells:
                rows.append(cells)
        tables.append({"table_idx": number, "rows": len(rows), "first_rows": rows[:4],
                       "preceding_text": [re.sub(r"\s+", "", p.get_text()) for p in table.find_all_previous("p", limit=3)]})
    return {"stat_month": month, "receipt": receipt,
            "title": soup.title.get_text(strip=True) if soup.title else None,
            "metadata": {m.get("name", m.get("property", "")): m.get("content", "") for m in soup.find_all("meta") if m.get("content")},
            "notes": compact[at:at + 2500] if at >= 0 else None, "tables": tables}


def run():
    for folder in ["raw", "requests", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "sources": SOURCES,
        "purpose": "区分二手与新建住宅、环比与同比、城市个数与市值；检查换基是否调整基本分类权数。",
        "candidate": "公布的一位小数环比指数小于100的二手住宅城市占70城比例，先查可测性，不计算策略收益。",
        "clock": "原公布日与目录日核对，无法证实更早可用时统一日末；迁移URL日期不充当公布时间。",
        "network": "五个官方公开URL，同时最多2次请求；超时普通重试一次，沿用200次全局上限与403/429停止规则。",
        "new_accounts": 0, "new_strategy_returns": 0, "orders_authorized": False}, True)
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"), "catalogue_lead_sha256": digest(OUT / "catalogue_lead.json"),
        "transport_code_sha256": digest(Path(transport.__file__))}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records, failed = [], []
    items = list(SOURCES.items())
    with ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(items), 2):
            batch = items[start:start + 2]
            jobs = [(url, "month_" + month) for month, url in batch]
            for (month, url), receipt in zip(batch, executor.map(REQUEST, jobs)):
                if receipt["status"] == "HTTP_OK":
                    records.append(describe(month, receipt))
                else:
                    failed.append({"stat_month": month, "url": url, "receipt": receipt})
            print(f"住宅价格口径探查已保存{len(records)}/5份原文。", flush=True)
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
    save(OUT / "source_measurements.json", records, True)
    save(OUT / "unresolved_sources.json", failed, True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "METHOD_NOTES_AND_TABLE_LAYOUT_SAVED" if len(records) == 5 else "PARTIAL_METHOD_SOURCE_PROBE",
        "saved_reports": len(records), "failed_reports": len(failed), "new_accounts": 0,
        "new_strategy_returns": 0, "current_market_view": "NO_VIEW", "goal_achieved": False}, True)


if __name__ == "__main__":
    run()
