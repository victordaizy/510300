"""在9月PMI公布前保存预期，并核对跨季资金的真实续作规模。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import company_revision_driver_sources_v1 as source


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_domestic_event_baseline_v1"
PBC = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/"


def main():
    if (OUT / "scope.json").exists():
        raise SystemExit("本轮采集已启动，复用已保存响应。")
    source.OUT = OUT
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    source.save(OUT / "scope.json", {
        "recorded_at": source.now(),
        "question": "即将公布的PMI怎样超出已有预期；逆回购增量和续作各是多少？",
        "pmi_event": "2026-09-30 09:30 Asia/Shanghai，实际发布以官方为准",
        "pmi_capture_type": "发布前当次公开日历快照，未取得调查全体样本及分歧分布",
        "source_limit": 10,
        "existing_forecasts": "原F1/F2/F3保留，不将标题PMI预期替换新订单预测。",
        "funding_rule": "实际操作、到期本金、操作上限和股票新增购买分开；未知渠道不填零。",
        "new_accounts": 0, "new_return_tests": 0, "automatic_retries": False,
    })
    items = [
        ("te_nbs_manufacturing_calendar", "html", "https://tradingeconomics.com/china/business-confidence", None, None),
        ("investing_nbs_manufacturing_calendar", "html", "https://www.investing.com/economic-calendar/chinese-manufacturing-pmi-594", None, None),
        ("investing_nonmanufacturing_calendar", "html", "https://www.investing.com/economic-calendar/chinese-non-manufacturing-pmi-831", None, None),
        ("pbc_20260929_operations", "html", PBC + "125475/2026092908461628271/index.html", None, None),
        ("pbc_20260928_operations", "html", PBC + "125475/2026092808454683233/index.html", None, None),
        ("pbc_20260922_operations", "html", PBC + "125475/2026092208451519789/index.html", None, None),
        ("pbc_20260923_operations", "html", PBC + "125475/2026092308513432696/index.html", None, None),
        ("pbc_overnight_plan_september", "html", PBC + "125469/2026092314094223341/index.html", None, None),
        ("pbc_overnight_plan_previous", "html", PBC + "125469/2026091016190551668/index.html", None, None),
        ("chinamoney_daily_bulletin", "json", "https://www.chinamoney.com.cn/ags/ms/cm-u-dlrp/PrDlyBltn", None, None),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(source.fetch, items))
    source.save(OUT / "source_result.json", {"recorded_at": source.now(), "receipts": [r[0] for r in rows]})
    for receipt, value in rows:
        if receipt["key"] == "chinamoney_daily_bulletin":
            print("货币网响应", str(value)[:1800])
    print("预期与操作来源保存完毕；没有启动后台采集。")


if __name__ == "__main__":
    main()
