"""补充必要的利率公布值及已保存报告中的资管原因页。"""

import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_financial_driver_bridge_v1"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (OUT / "context_result.json").exists():
        raise SystemExit("本轮补充来源已完成，不重复采集。")
    receipts = []
    for name, url in [
        ("lpr_government_reprint", "https://jrj.sh.gov.cn/SCGK194/20260920/736cd13499494746bebda910dd12cc65.html"),
        ("lpr_bank_history", "https://www.psbc.com/cn/common/bjfw/dkscbjllcx/202507/t20250721_352102.html"),
    ]:
        receipt = {"at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "url": url, "attempts": 1}
        try:
            response = requests.get(url, timeout=(8, 25), headers={"User-Agent": "Mozilla/5.0"})
            content = response.content
            receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            (OUT / "sources" / f"{name}.html").write_bytes(content)
            if response.status_code == 200:
                soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
                (OUT / "sources" / f"{name}.txt").write_text(text, encoding="utf-8")
                receipt.update(status="网页已保存", expected_numbers_present="3.0" in text and "3.5" in text)
            else:
                receipt["status"] = "请求失败保留响应"
        except requests.RequestException as exc:
            receipt.update(status="请求失败", error=str(exc))
        receipts.append(receipt)
    with pdfplumber.open(OUT / "sources/pingan_h1.pdf") as pdf:
        pages = [{"pdf_page": i+1, "text": pdf.pages[i].extract_text() or ""} for i in range(65, 69)]
    save(OUT / "sources/pingan_asset_management_pages.json", pages)
    save(OUT / "context_result.json", {"receipts": receipts, "additional_existing_pdf_pages": [66, 67, 68, 69]})
    for page in pages:
        print("平安PDF页", page["pdf_page"], page["text"], flush=True)
    print(json.dumps(receipts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
