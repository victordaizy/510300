"""读取两家首批参与机构的历史报告，分开融资、持仓和对冲口径。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_sfisf_institution_exposure_v1"
SOURCES = [
    {"key": "中信证券_2024年报", "report_end": "2024-12-31",
     "url": "https://www.citics.com/newsite/tzzgx/ggyth/gg/aggg/202503/P020250326744015972903.pdf",
     "expected": ["2024", "中信证券"]},
    {"key": "中金公司_2024年报", "report_end": "2024-12-31",
     "url": "https://static.cninfo.com.cn/finalpage/2025-03-29/1222948339.PDF",
     "source_note": "公司官网521；采用巨潮公司正式披露原报告。目录公告日期2025-03-29，不是已核实的最早公开时刻。",
     "expected": ["2024", "中国国际金融"]},
    {"key": "中金公司_2025中报", "report_end": "2025-06-30",
     "url": "https://static.cninfo.com.cn/finalpage/2025-08-30/1224627802.PDF",
     "source_note": "公司官网521；采用巨潮公司正式披露原报告。目录公告日期2025-08-30，不是已核实的最早公开时刻。",
     "expected": ["2025", "中国国际金融"]},
    {"key": "中信证券_2025中报", "report_end": "2025-06-30",
     "url": "https://static.cninfo.com.cn/finalpage/2025-08-29/1224609003.PDF",
     "source_note": "公司官网索引当日已迁移，采用巨潮正式披露。目录公告日期2025-08-29，不是已核实的最早公开时刻。",
     "expected": ["2025", "中信证券"]},
    {"key": "中信银行_20241025融资成交", "suffix": ".html", "report_end": None,
     "url": "https://www.citicbank.com/about/companynews/banknew/message/202410/t20241029_117040.html",
     "expected": ["10月25日", "中信证券"]},
]


def download(item):
    """已存文件原样复用，每个新来源只作一次有超时的请求。"""
    path = OUT / "sources" / (item["key"] + item.get("suffix", ".pdf"))
    receipt_path = path.with_suffix(".receipt.json")
    if not path.exists():
        response = requests.get(item["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=35)
        response.raise_for_status()
        if path.suffix == ".pdf" and not response.content.startswith(b"%PDF"):
            raise ValueError(f"下载内容不是PDF：{item['key']}，响应类型{response.headers.get('Content-Type')}")
        path.write_bytes(response.content)
        receipt = {**item, "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   "resolved_url": response.url, "path": path.relative_to(ROOT).as_posix(),
                   "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(),
                   "historical_first_vintage": False}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return item, path


def extract(item, path):
    if path.suffix == ".html":
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        content = soup.get_text(" ", strip=True)
        if not all(term in content for term in item["expected"]):
            raise ValueError(f"网页未覆盖指定正文：{item['key']}")
        path.with_suffix(".txt").write_text(content, encoding="utf-8")
        return {"source": item["key"], "kind": "网页", "bytes": path.stat().st_size}
    pages_path = path.with_suffix(".pages.json")
    if pages_path.exists():
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
    else:
        pages = []
        document = pdfium.PdfDocument(str(path))
        for index in range(len(document)):
            page = document[index]
            text_page = page.get_textpage()
            pages.append({"pdf_page": index + 1, "text": text_page.get_text_range()})
            text_page.close()
            page.close()
        document.close()
        pages_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    combined = "\n\n".join(f"PDF第{page['pdf_page']}页\n{page['text']}" for page in pages)
    if not all(term in combined for term in item["expected"]):
        raise ValueError(f"报告未覆盖指定公司与年度：{item['key']}")
    path.with_suffix(".txt").write_text(combined, encoding="utf-8")
    matches = []
    for page in pages:
        compact = "".join(page["text"].split())
        if any(term in compact for term in ["互换便利", "互換便利", "SFISF"]):
            matches.append(page)
    (OUT / (item["key"] + "_互换工具原文.json")).write_text(
        json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
    contexts = []
    for page in matches:
        compact = "".join(page["text"].split())
        for match in re.finditer("互换便利|互換便利|SFISF", compact):
            contexts.append({"pdf_page": page["pdf_page"],
                             "context": compact[max(0, match.start()-160):match.end()+650]})
    return {"source": item["key"], "pages": len(pages), "contexts": contexts}


def main():
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    prior = OUT / "来源读取结果.json"
    initial = OUT / "初次来源读取结果.json"
    if prior.exists() and not initial.exists():
        initial.write_bytes(prior.read_bytes())
    downloads = []
    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(item, pool.submit(download, item)) for item in SOURCES]
        for item, future in futures:
            try:
                downloads.append(future.result())
            except Exception as error:
                failures.append({"source": item["key"], "error": str(error)})
    index = []
    for item, path in downloads:
        try:
            result = extract(item, path)
            index.append(result)
            print(json.dumps(result, ensure_ascii=False))
        except Exception as error:
            failures.append({"source": item["key"], "error": str(error)})
    result = {"sources": index, "failures": failures}
    (OUT / "来源读取结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"成功来源": len(index), "未完成来源": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
