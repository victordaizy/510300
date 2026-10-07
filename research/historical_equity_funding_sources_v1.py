"""保存本轮新增的两份央行历史报告，并定位资本市场工具段落。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_equity_funding_realization_v1/sources"
SOURCES = [
    {"key": "2025Q1_货币政策执行报告", "report_date": "2025-05-09",
     "url": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/f5c4690f2cbd40918bf24c2d39ac58af/2025091218344076234/2025081517321679368.pdf",
     "expected": ["820", "1050"], "publisher": "中国人民银行"},
    {"key": "2025Q2_货币政策执行报告", "report_date": "2025-08-15",
     "url": "https://jrj.sh.gov.cn/cmsres/5f/5f21edf1707d4151897f0accc780b9b8/2ae040986f43b76ccc09111cbd06f12b.pdf",
     "expected": ["3100", "900"], "publisher": "中国人民银行；上海市委金融办网站保存原报告"},
    {"key": "20241010_央行原公告新华社转载", "report_date": "2024-10-10", "suffix": ".html",
     "url": "https://www1.xinhuanet.com/fortune/20241010/f87e7e4aea5a4fb2a08c4d2fe84086eb/c.html",
     "expected": ["08:27:38", "5000"], "publisher": "中国人民银行原公告；新华社当日转载"},
    {"key": "2025前三季度央行大事记", "report_date": "2025-11-11", "suffix": ".html",
     "url": "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/5896228/index.html",
     "expected": ["5月9日", "8月15日"], "publisher": "中国人民银行；只用于核对历史发布日期，不倒填新政策信息"},
]


def download(item):
    path = OUT / (item["key"] + item.get("suffix", ".pdf"))
    receipt_path = path.with_suffix(".receipt.json")
    if not path.exists():
        response = requests.get(item["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        if path.suffix == ".pdf" and not response.content.startswith(b"%PDF"):
            raise ValueError(f"下载内容不是PDF：{item['key']}")
        path.write_bytes(response.content)
        receipt = {**item, "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   "path": path.relative_to(ROOT).as_posix(), "bytes": len(response.content),
                   "sha256": hashlib.sha256(response.content).hexdigest(),
                   "historical_first_vintage": False}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return item, path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloads = list(pool.map(download, SOURCES))
    for item, path in downloads:
        if path.suffix == ".html":
            content = BeautifulSoup(path.read_bytes(), "html.parser").get_text(" ", strip=True)
            if not all(term in content for term in item["expected"]):
                raise ValueError(f"历史公告文本未覆盖所需事实：{item['key']}")
            path.with_suffix(".txt").write_text(content, encoding="utf-8")
            print(json.dumps({"公告": item["key"], "指定内容已定位": True}, ensure_ascii=False))
            continue
        pages = []
        document = pdfium.PdfDocument(str(path))
        for index in range(len(document)):
            page = document[index]
            text_page = page.get_textpage()
            pages.append({"pdf_page": index + 1, "text": text_page.get_text_range()})
            text_page.close()
            page.close()
        document.close()
        combined = "\n\n".join(f"PDF第{p['pdf_page']}页\n{p['text']}" for p in pages)
        if not all(term in combined for term in item["expected"]):
            raise ValueError(f"报告未覆盖所需数值：{item['key']}")
        path.with_suffix(".pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        path.with_suffix(".txt").write_text(combined, encoding="utf-8")
        matches = []
        for page in pages:
            compact = "".join(page["text"].split())
            if all(term in compact for term in item["expected"]):
                matches.append(page)
        print(json.dumps({"报告": item["key"], "页数": len(pages), "相关页面": matches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
