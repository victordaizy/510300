"""保存9月23日美国增长与利率事件的有限原件，不搜索交易参数。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import company_revision_driver_sources_v1 as source


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_september_rate_shock_v1"


def main():
    if (OUT / "scope.json").exists():
        raise SystemExit("本轮来源已经启动，请使用保存的原件。")
    source.OUT = OUT
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    source.save(OUT / "scope.json", {
        "recorded_at": source.now(),
        "question": "9月23日需求与供给约束的新信息，是否伴随实际收益率或通胀补偿变化；A股何时可以反应？",
        "selection": "定向复盘已知事件，不属于盲测或事前收益预测。",
        "primary_event": "2026-09-23美国PMI初值",
        "event_window": "9月22至25日美国利率与9月23至28日A股日线；按事件时点而非最优收益挑选。",
        "request_limit": 6,
        "reuse_prices": "510300_index_repricing_odds_v1及510300_weight_company_driver_bridge_20260929既存原件",
        "expectation_rule": "事后读取的带日期预告只能重建公开预期代理；网页更新与原始调查未取到的限制保留。",
        "new_accounts": 0,
        "new_return_tests": 0,
        "automatic_retry": False,
    })
    items = [
        ("spglobal_us_flash_commentary", "pdf", "https://www.spglobal.com/content/dam/spglobal/mi/en/documents/news-insights/research/2026/09/6293416_6293414_0.1.pdf", None, None),
        ("fed_h15_current", "html", "https://www.federalreserve.gov/releases/h15/", None, None),
        ("fed_barr_20260923", "html", "https://www.federalreserve.gov/newsevents/speech/barr20260923a.htm", None, None),
        ("squawk_pmi_preview", "html", "https://squawknews.com/news/349835-us-preview-flash-pmis-seen-moderating-but-scope-for-renewed-price-pressures", None, None),
        ("econoday_preview_date_conflict", "html", "https://fidelity.econoday.com/byshoweventarticle", {"cust": "fidelity", "fid": "705856", "lid": "0", "year": "2026"}, None),
        ("fed_september_calendar", "html", "https://www.federalreserve.gov/newsevents/2026-september.htm", None, None),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(source.fetch, items))
    source.save(OUT / "source_result.json", {
        "recorded_at": source.now(),
        "receipts": [r[0] for r in rows],
        "note": "主来源是美联储及调查发布机构；预告网站只用于有边界的历史公开预期重建。",
    })
    print("六个定向请求结束，原始响应与接收时间已保存。")


if __name__ == "__main__":
    main()
