"""复用本地组织代码，一次取得前期目录、激励文件和当期预测页面。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
import requests
from bs4 import BeautifulSoup

from company_expectation_anchor_sources_v1 import HEADERS, OUT, ROOT, now, request, save


def catalogue(code):
    directory = json.loads((ROOT / "reports/research/510300_factor96_issuance_catalogue_v1/inputs/company_directory.json").read_text(encoding="utf-8-sig"))
    company = next(x for x in directory["stockList"] if x["code"] == code)
    payload = {"pageNum": "1", "pageSize": "100", "column": "sse" if code.startswith("6") else "szse", "tabName": "fulltext", "stock": code + "," + company["orgId"], "searchkey": "", "secid": "", "category": "", "trade": "", "seDate": "2026-07-01~2026-08-07", "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
    result = request("prior_announcements_cached_org_" + code, "https://www.cninfo.com.cn/new/hisAnnouncement/query", payload=payload)
    if isinstance(result, dict):
        rows = result.get("announcements") or []
        brief = [{k: r.get(k) for k in ("secCode", "announcementTitle", "announcementTime", "adjunctUrl")} for r in rows if any(w in r.get("announcementTitle", "") for w in ("业绩", "招股", "激励计划", "投资者关系", "全球发售"))]
        print(json.dumps({"股票": code, "目录总数": result.get("totalAnnouncement"), "返回条数": len(rows), "相关公告": brief}, ensure_ascii=False), flush=True)


def document(item):
    key, url, kind = item
    receipt = {"at": now(), "url": url, "attempts": 1, "source_kind": kind}
    try:
        response = requests.get(url, headers=HEADERS, timeout=(8, 25))
        receipt["http_status"] = response.status_code
        response.raise_for_status()
        path = OUT / "sources" / (key + "." + kind)
        path.write_bytes(response.content)
        receipt.update(sha256=hashlib.sha256(response.content).hexdigest(), bytes=len(response.content))
        if kind == "pdf":
            if not response.content.startswith(b"%PDF"):
                raise ValueError("返回内容不是PDF")
            with pdfplumber.open(path) as doc:
                pages = [{"pdf_page": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(doc.pages)]
            save("sources/" + key + "_pages.json", pages)
            receipt.update(status="COMPANY_DOCUMENT_MIRROR_PDF_SAVED", pages=len(pages))
            print(json.dumps({"文档": key, "页数": len(pages), "目标相关页": [x["pdf_page"] for x in pages if "135" in x["text"] or "不构成" in x["text"]]}, ensure_ascii=False), flush=True)
        else:
            response.encoding = "gb18030" if b"gb" in response.content[:3000].lower() else "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            content = soup.get_text("\n", strip=True)
            (OUT / "sources" / (key + ".txt")).write_text(content, encoding="utf-8")
            good = "2026" in content and "预测" in content and "净利润" in content
            receipt.update(status="VENDOR_FORECAST_HTML_SAVED" if good else "HTML_WITHOUT_VERIFIED_FORECAST_CONTENT", characters=len(content))
            print(json.dumps({"文档": key, "状态": receipt["status"], "文字长度": len(content)}, ensure_ascii=False), flush=True)
    except (requests.RequestException, ValueError) as exc:
        receipt.update(status="FAILED_NO_RETRY", error=str(exc))
    save("receipts/" + key + ".json", receipt)


def main():
    if (OUT / "enrichment_scope.json").exists():
        raise RuntimeError("补充请求已执行，不能自动重跑。")
    save("enrichment_scope.json", {"at": now(), "reason": "组织查询接口500；改用既有本地组织表发起不同的公告目录请求，不重试原查询。预测页面仅作本轮起的供应商参考快照。", "company_codes": ["300308", "300502", "688256"], "snapshot_revision_warning": "web打开与搜索展示的新易盛机构数和均值不同，两者均保留，不把差异记为真实上调。", "no_historical_consensus_backfill": True})
    items = [("cambricon_incentive_summary", "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2026/2026-7/2026-07-29/12466849.PDF", "pdf")]
    items += [("ths_" + code, "https://basic.10jqka.com.cn/" + code + "/worth.html", "html") for code in ("300308", "300502", "688256")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(catalogue, code) for code in ("300308", "300502", "688256")]
        tasks += [pool.submit(document, item) for item in items]
        for task in tasks:
            task.result()
    print("本轮补充来源已保存；不把目录失败、转载摘要或覆盖变化转换为盈利修正。", flush=True)


if __name__ == "__main__":
    main()
