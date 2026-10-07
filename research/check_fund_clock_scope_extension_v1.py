"""保存新增的交易所发布时间说明，并区分它们各自对应的数据字段。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from collect_bualuang_money_consensus_v1 import now, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_money_consensus_source_extension_v1/fund_clock"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sources = [
        ("sse_fund_home", "https://etf.sse.com.cn/", "全市场基金总规模", "总规模数据每交易日23:00更新",
         "当前页面关于全市场总规模的刷新说明；没有证明历史510300每日总份额字段的首次公开时点。"),
        ("sse_pcf_clock_notice", "https://www.sse.com.cn/assortment/fund/etf/rules/c/c_20150911_3985183.shtml",
         "每日ETF申购赎回清单PCF", "8：00",
         "通知正文署2010-10-27并规定2010-11-08起生效；网页标注2012-07-14。适用于PCF，不适用于ETF总份额或净申赎流量；最小申购赎回单位资产净值不是基金总资产净值。"),
    ]
    records = []
    for key, url, field, token, limit in sources:
        path = OUT / (key + ".html")
        if path.exists():
            body = path.read_bytes()
            status = "CACHED_CURRENT_RETRIEVAL"
        else:
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            body = response.content
            path.write_bytes(body)
            status = "FETCHED_CURRENT_PAGE"
        text = BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
        records.append({"source": key, "url": url, "retrieved_at": now(), "status": status,
            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "declared_scope": field,
            "clock_text_present": token in text, "historical_single_fund_share_clock_proven": False,
            "adjudication": limit})
    save(OUT / "result.json", {"status": "NEW_CLOCK_SOURCES_DO_NOT_ESTABLISH_HISTORICAL_SHARE_AVAILABILITY",
        "sources": records, "fund_variable": "510300最近五交易日真实份额净变化/期初份额",
        "C_D": "NOT_RUN_FUND_PUBLICATION_CLOCK", "volume_or_pcf_substitution_used": False,
        "historical_share_receipts_not_refetched": True})
    print(json.dumps({"新增说明数量": len(records), "适用范围": [r["declared_scope"] for r in records],
        "原文时点字段核对": [r["clock_text_present"] for r in records], "资金组": "NOT_RUN"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
