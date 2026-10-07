"""定向取得经营指引与盈利预测，只比较同一对象的真实修正。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_company_revision_driver_v1"
IVAN_BASE = "https://www.ivanhoemines.com/news-stories/news-release/"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(item):
    key, kind, url, params, payload = item
    receipt = {"key": key, "url": url, "received_at": now(), "params": params, "payload": payload, "attempts": 1}
    value = None
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/" if "cninfo" in url else "https://data.eastmoney.com/"}
        if payload is None:
            response = requests.get(url, params=params, headers=headers, timeout=(8, 28))
        else:
            response = requests.post(url, data=payload, headers=headers, timeout=(8, 28))
        content = response.content
        receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        extension = ".pdf" if content.startswith(b"%PDF") else ".json" if kind == "json" else ".html"
        path = OUT / "sources" / (key + extension)
        path.write_bytes(content)
        response.raise_for_status()
        if kind == "json":
            value = response.json()
            receipt["status"] = "目录响应已保存"
        elif kind == "pdf":
            if not content.startswith(b"%PDF"):
                raise ValueError("没有取得PDF原件。")
            with pdfplumber.open(path) as doc:
                value = [{"page": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(doc.pages[:25])]
                receipt["total_pages"] = len(doc.pages)
            save(OUT / "sources" / (key + "_pages.json"), value)
            receipt["status"] = "原件及前25页以内文本已保存"
        else:
            response.encoding = "gb18030" if b"gb" in content[:3000].lower() else "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            value = soup.get_text("\n", strip=True)
            (OUT / "sources" / (key + ".txt")).write_text(value, encoding="utf-8")
            receipt.update(status="页面已保存", characters=len(value))
    except (requests.RequestException, ValueError, OSError) as exc:
        receipt.update(status="失败，不自动重试", error=str(exc))
    save(OUT / "receipts" / (key + ".json"), receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k in ["key", "status", "http_status", "characters", "error", "total_pages"]}, ensure_ascii=False), flush=True)
    return receipt, value


def main():
    if (OUT / "scope.json").exists():
        raise SystemExit("本轮来源已启动，复用已有响应，不重跑请求。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    save(OUT / "scope.json", {
        "recorded_at": now(), "previous_turn_classification": "PROGRESS_BROAD_INDEX_PRICE_PRESSURE_AND_COMPANY_UNIT_ECONOMICS",
        "cohort": ["300750", "601899"], "selection": "沿用上轮工业与原材料经营对象；已知阶段跌幅，不当作盲测。",
        "question": "经营约束与盈利预测是否发生同口径修正；既有预期是否已经包含税负和复产变化？",
        "first_stage_requests": 8, "broker_window": "2026-03-01至2026-09-29，每股一页100条；缺页不称不存在。",
        "broker_pair_rule": "先取得目录，再按同机构相邻报告确定有限原件；不挑预测上调或下调最显著的机构。",
        "known_issue": "紫金H1引用29至33万吨；艾芬豪7月29日指引29至31万吨且产品阶段范围有说明，需保留双方披露。",
        "excluded_search_hit": "标题为第三季度10.4万吨的紫金新闻页面没有读到正文日期，不认定为2026年Q3已公布。",
        "new_accounts": 0, "new_return_tests": 0,
    })
    items = [("ths_" + code, "html", f"https://basic.10jqka.com.cn/{code}/worth.html", None, None) for code in ["300750", "601899"]]
    items += [("ivanhoe_q1", "html", IVAN_BASE + "ivanhoe-mines-issues-2026-first-quarter-financial-results-overview-of-operations-and-exploration-activities/", None, None),
              ("ivanhoe_q2", "html", IVAN_BASE + "ivanhoe-mines-issues-2026-second-quarter-financial-results-overview-of-operations-and-exploration-activities/", None, None),
              ("ivanhoe_january", "html", IVAN_BASE + "ivanhoe-mines-provides-2025-production-results-2026-production-guidance/", None, None)]
    for code in ["300750", "601899"]:
        params = {"industryCode": "*", "pageSize": 100, "industry": "*", "rating": "*", "ratingChange": "*", "beginTime": "2026-03-01", "endTime": "2026-09-29", "pageNo": 1, "qType": 0, "fields": "", "stockCode": code}
        items.append(("broker_directory_" + code, "json", "https://reportapi.eastmoney.com/report/list", params, None))
    directory = json.loads((ROOT / "reports/research/510300_factor96_issuance_catalogue_v1/inputs/company_directory.json").read_text(encoding="utf-8-sig"))
    company = next(x for x in directory["stockList"] if x["code"] == "300750")
    payload = {"pageNum": "1", "pageSize": "100", "column": "szse", "tabName": "fulltext", "stock": "300750," + company["orgId"], "searchkey": "", "secid": "", "category": "", "trade": "", "seDate": "2026-07-15~2026-09-29", "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    items.append(("catl_company_announcements", "json", "https://www.cninfo.com.cn/new/hisAnnouncement/query", None, payload))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, items))
    save(OUT / "source_result.json", {"recorded_at": now(), "receipts": [x[0] for x in results], "responses": {x[0]["key"]: x[1] for x in results}})
    for key, value in [(x[0]["key"], x[1]) for x in results if "directory" in x[0]["key"]]:
        if isinstance(value, dict):
            rows = value.get("data") or []
            brief = [{k: r.get(k) for k in ["title", "orgSName", "publishDate", "infoCode", "predictThisYearNetProfit"]} for r in rows[:12]]
            print(json.dumps({"目录": key, "总数": value.get("TotalCount"), "本页": len(rows), "前12条": brief}, ensure_ascii=False), flush=True)
    value = next((x[1] for x in results if x[0]["key"] == "catl_company_announcements"), None)
    if isinstance(value, dict):
        rows = value.get("announcements") or []
        brief = [{k: r.get(k) for k in ["announcementTitle", "announcementTime", "adjunctUrl"]} for r in rows if any(w in r.get("announcementTitle", "") for w in ["投资者", "调研", "业绩", "澄清"])]
        print(json.dumps({"宁德目录条数": len(rows), "总数": value.get("totalAnnouncement"), "相关原件": brief}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
