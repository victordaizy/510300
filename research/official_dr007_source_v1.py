"""沿官方页面公开组件查找DR007日行情，保留业务错误和盘中状态。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup

import company_revision_driver_sources_v1 as source


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_official_dr007_source_v1"
BASE = "https://www.chinamoney.com.cn"


def fetch_component(key, path, kind="html", params=None, payload=None):
    source.OUT = OUT
    if (OUT / "receipts" / (key + ".json")).exists():
        raise ValueError("已保存该请求，不自动重试：" + key)
    return source.fetch((key, kind, BASE + path, params, payload))


def main():
    if (OUT / "scope.json").exists():
        raise SystemExit("已启动官方组件核对，后续只沿已读页面给出的链接继续。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    source.save(OUT / "scope.json", {
        "recorded_at": source.now(), "question": "官方日行情的indexType和DR007加权字段是什么，能否得到明确日期的最终值？",
        "basis": "上一轮官方接口返回indexType is null；本轮先读取官方组件，不盲猜参数。",
        "initial_requests": 3, "additional_limit": 8,
        "do_not_substitute": ["FDR007", "FR007", "R007", "Shibor", "盘中DR007"],
        "existing_forecasts_unchanged": True, "new_accounts": 0, "new_return_tests": 0,
    })
    items = [
        ("rmb_home_component", "/r/cms/chinese/chinamoney/html/newhome/rmb-market-h.html"),
        ("benchmarks_home_component", "/r/cms/chinese/chinamoney/html/newhome/benchmarks-h.html"),
        ("official_navigation", "/chinese/mainhead/"),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda item: fetch_component(*item), items))
    source.save(OUT / "component_receipts.json", {"recorded_at": source.now(), "receipts": [r[0] for r in rows]})
    for receipt, value in rows:
        print("组件", receipt["key"])
        path = OUT / "sources" / (receipt["key"] + ".html")
        if not path.exists():
            continue
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup.find_all(["a", "script", "div"]):
            if tag.name == "script":
                body = tag.get("src") or tag.get_text()
                if any(x in body for x in [".js", "PrDly", "indexType", "DR007", "Pledge"]):
                    print(body[:2600])
            elif tag.name == "a" and any(x in tag.get_text() for x in ["回购", "本币", "货币市场", "日行情"]):
                print(tag.get_text(" ", strip=True), tag.get("href"), tag.get("data-url"))
            elif tag.get("data-url"):
                print(tag.get("id"), tag.get("data-url"))


if __name__ == "__main__":
    main()
