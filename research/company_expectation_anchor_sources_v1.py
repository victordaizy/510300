"""有限取得三家既定公司的事前披露，不推断一致预期或生成账户。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_company_expectation_anchor_v1"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def request(key, url, params=None, payload=None):
    receipt = {"at": now(), "url": url, "params": params, "payload": payload, "attempts": 1}
    try:
        if payload is None:
            response = requests.get(url, params=params, headers=HEADERS, timeout=(8, 20))
        else:
            response = requests.post(url, data=payload, headers=HEADERS, timeout=(8, 20))
        receipt.update(http_status=response.status_code, final_url=response.url)
        response.raise_for_status()
        content = response.content
        (OUT / "sources" / (key + ".raw")).write_bytes(content)
        receipt.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), status="RAW_SAVED")
        data = response.json()
        save("sources/" + key + ".json", data)
    except (requests.RequestException, ValueError) as exc:
        data = None
        receipt.update(status="FAILED_NO_RETRY", error=str(exc))
    save("receipts/" + key + ".json", receipt)
    return data


def lookup_company(code):
    data = request("company_lookup_" + code, "https://www.cninfo.com.cn/new/information/topSearch/query", {"keyWord": code, "maxNum": 10})
    print(json.dumps({"股票": code, "组织查询": data}, ensure_ascii=False), flush=True)
    if not isinstance(data, list):
        return
    selected = next((x for x in data if str(x.get("code", "")) == code), None)
    if not selected:
        return
    org = selected.get("orgId")
    if not org:
        return
    payload = {"pageNum": "1", "pageSize": "100", "column": "sse" if code.startswith("6") else "szse", "tabName": "fulltext", "stock": code + "," + org, "searchkey": "", "secid": "", "category": "", "trade": "", "seDate": "2026-07-01~2026-08-07", "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    result = request("prior_announcements_" + code, "https://www.cninfo.com.cn/new/hisAnnouncement/query", payload=payload)
    if not isinstance(result, dict):
        return
    rows = result.get("announcements") or []
    brief = [{k: row.get(k) for k in ("secCode", "announcementTitle", "announcementTime", "adjunctUrl")} for row in rows if any(word in row.get("announcementTitle", "") for word in ("业绩", "招股", "激励计划", "投资者关系", "全球发售"))]
    print(json.dumps({"股票": code, "目录总数": result.get("totalAnnouncement"), "本次取得": len(rows), "相关公告": brief}, ensure_ascii=False), flush=True)


def download_preview():
    url = "https://static.cninfo.com.cn/finalpage/2026-07-20/1225431444.pdf"
    receipt = {"at": now(), "url": url, "attempts": 1}
    try:
        response = requests.get(url, headers=HEADERS, timeout=(8, 20))
        receipt["http_status"] = response.status_code
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError("未返回PDF文件")
        path = OUT / "sources/eoptolink_h1_preview.pdf"
        path.write_bytes(response.content)
        with pdfplumber.open(path) as doc:
            pages = [{"pdf_page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(doc.pages)]
        save("sources/eoptolink_h1_preview_pages.json", pages)
        receipt.update(status="ORIGINAL_PDF_SAVED", sha256=hashlib.sha256(response.content).hexdigest(), pages=len(pages))
        print(json.dumps({"新易盛预告": pages}, ensure_ascii=False), flush=True)
    except (requests.RequestException, ValueError) as exc:
        receipt.update(status="FAILED_NO_RETRY", error=str(exc))
    save("receipts/eoptolink_h1_preview.json", receipt)


def main():
    if (OUT / "scope.json").exists():
        raise RuntimeError("本轮已开始，复用已取得响应，不重跑下载。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    save("scope.json", {"created_at": now(), "previous_turn_classification": "PROGRESS_REAL_DRIVERS_COMPANY_TRANSMISSION_AND_CURRENT_PRICE_ANCHORS", "question": "三家既定公司在半年报前提供过什么可比预期，实际业绩相对这些基准究竟增加了哪些信息？", "cohort": ["300308", "300502", "688256"], "selection": "沿用上轮指数权重与经营传导对象，不按事后收益选样。", "already_seen": "已知三家公司H1实际业绩及8月至9月价格；已搜索到新易盛70至80亿元预告、寒武纪激励目标和中际旭创7月澄清线索。本轮是明确知情后的归因诊断，不冒充盲测。", "maximum_new_pdf_documents": 6, "query_window": "2026-07-01至2026-08-07，最多每家公司一页100条，不以缺页声称不存在公告。", "new_return_tests": 0, "new_accounts": 0, "orders_authorized": False, "goal_achieved": False})
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(lookup_company, code) for code in ("300308", "300502", "688256")]
        jobs.append(pool.submit(download_preview))
        for job in jobs:
            job.result()
    print("有限来源请求完成，待逐项区分预告、指引和考核目标。", flush=True)


if __name__ == "__main__":
    main()
