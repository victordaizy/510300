"""补齐既定十大权重中的消费与医药经营原因，定向保存六份原始材料。"""

from concurrent.futures import ThreadPoolExecutor

import company_revision_driver_sources_v1 as source


OUT = source.ROOT / "reports/research/510300_consumer_health_driver_bridge_v1"


def main():
    if (OUT / "scope.json").exists():
        raise SystemExit("本轮资料已启动，直接使用已存原件，不重复请求。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    source.OUT = OUT
    source.save(OUT / "scope.json", {
        "recorded_at": source.now(),
        "selection": "补齐既定2026年8月末十大权重中的贵州茅台、美的集团、药明康德；没有按本轮收益重新选公司。",
        "question": "销售渠道、汇率对冲和订单结构怎样改变表面因子，以及未来兑现受什么约束？",
        "cohort": ["600519", "000333", "603259"],
        "request_count": 6,
        "index_scope_note": "美的是消费组公司，经营中的制造业务不能等同于沪深300工业行业权重。",
        "new_accounts": 0,
        "new_return_tests": 0,
        "no_historical_strategy_rewrite": True,
    })
    items = [
        ("moutai_h1", "pdf", "https://static.cninfo.com.cn/finalpage/2026-08-15/1225475868.PDF", None, None),
        ("midea_h1", "pdf", "https://disc.static.szse.cn/disc/disk03/finalpage/2026-08-29/df25443f-d67a-4cc5-8bd6-6c3e681a9575.PDF", None, None),
        ("wuxi_h1", "pdf", "https://static.cninfo.com.cn/finalpage/2026-08-04/1225455027.PDF", None, None),
        ("moutai_h1_meeting", "html", "https://wwwhy.moutai.com.cn/mtjt/2026-08/21/article_2026082123484650788.html", None, None),
        ("wuxi_h1_release", "html", "https://www.wuxiapptec.cn/en/news/wuxi-news/wuxi-apptec-delivers-strong-h1-2026-results-and-raises-full-year-guidance-driven-by-greater-success-of-multiple-customers-products-and-crdmo-model", None, None),
        ("wuxi_q1_presentation", "pdf", "https://officialsite-static.wuxiapptec.com/upload/2026_FIRST_QUARTERLY_RESULTS_PRESENTATION_a5a7717323.pdf", None, None),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(source.fetch, items))
    source.save(OUT / "source_result.json", {
        "recorded_at": source.now(),
        "receipts": [r[0] for r in responses],
    })
    for receipt, value in responses:
        if receipt["key"] == "wuxi_h1_release" and isinstance(value, str):
            print(value[-18500:], flush=True)


if __name__ == "__main__":
    main()
