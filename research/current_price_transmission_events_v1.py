"""保存集中调整前的政策与回购信息，不改变既定价格窗口。"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_current_price_transmission_v1"
PRESS = "https://www.mccormick.senate.gov/news/press-releases/senators-mccormick-gallego-cornyn-fetterman-introduce-bill-to-keep-chinese-transceivers-out-of-u-s-national-security-systems/"
LEGISLATION = "https://www.mccormick.senate.gov/about/legislation/"
ANNOUNCEMENTS = "https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllBulletin/stockid/300308.phtml"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if (OUT / "event_source_result.json").exists():
        raise SystemExit("本轮信息来源已保存，不覆盖或重复采集。")
    save(OUT / "event_followup_scope.json", {
        "recorded_at": now(),
        "reason": "固定观察组最后完整交易日集中调整，沿已发现的政策提案和已完成回购线索查原件。",
        "prior_web_findings": "已看到9月25日参议员官方说明与中财网转载回购结果；本轮非盲测。",
        "unchanged": "公司集合、8月31日至9月28日窗口、收益规则、账户数、既有预测均不变。",
        "source_limit": "最多5个定向请求；Congress.gov正文此前无法打开，保留正文未取得。",
        "new_accounts": 0, "new_strategy_tests": 0,
    })
    receipts = []
    outputs = {}

    def fetch(key, url, kind="html"):
        receipt = {"key": key, "url": url, "received_at": now(), "attempts": 1}
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(8, 25))
            content = response.content
            receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            is_pdf = content.startswith(b"%PDF")
            path = OUT / "sources" / f"{key}{'.pdf' if is_pdf else '.html'}"
            path.write_bytes(content)
            response.raise_for_status()
            receipt["local_file"] = str(path.relative_to(ROOT))
            if kind == "pdf":
                if not is_pdf:
                    raise ValueError("来源未返回PDF原件")
                with pdfplumber.open(path) as pdf:
                    pages = [{"pdf_page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
                save(OUT / "sources" / f"{key}_pages.json", pages)
                outputs[key] = {"pages": pages, "url": url}
                receipt.update(status="公司原件已保存", pages=len(pages))
                result = None
            else:
                encoding = "gb18030" if "sina.com.cn" in url else "utf-8"
                soup = BeautifulSoup(content.decode(encoding, errors="replace"), "html.parser")
                for node in soup(["script", "style"]):
                    node.decompose()
                text = soup.get_text("\n", strip=True)
                (OUT / "sources" / f"{key}.txt").write_text(text, encoding="utf-8")
                outputs[key] = {"url": url, "text": text}
                receipt["status"] = "页面已保存"
                result = soup
        except (requests.RequestException, ValueError, OSError) as exc:
            receipt.update(status="来源失败，不自动重试", error=str(exc))
            result = None
        receipts.append(receipt)
        save(OUT / "receipts" / f"{key}.json", receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k in ["key", "status", "http_status", "pages", "error"]}, ensure_ascii=False), flush=True)
        return result

    fetch("optical_bill_sponsor", PRESS)
    fetch("optical_bill_status", LEGISLATION)
    listing = fetch("innolight_announcement_list", ANNOUNCEMENTS)
    if listing is not None:
        candidates = [a for a in listing.select("a[href]") if "关于股份回购结果暨股份变动公告" in a.get_text()]
        if candidates:
            target = urljoin(ANNOUNCEMENTS, candidates[0]["href"])
            detail = fetch("innolight_buyback_detail", target)
            if detail is not None:
                links = [urljoin(target, a["href"]) for a in detail.select("a[href]") if ".pdf" in a["href"].lower()]
                if links:
                    fetch("innolight_buyback", links[0], "pdf")
    save(OUT / "event_source_result.json", {
        "recorded_at": now(), "receipts": receipts, "responses": outputs,
        "bill_text_status": "Congress.gov网页读取失败，条款范围依据发起人官方摘要；未完成逐条法案解释。",
        "source_method": "单次请求保留原响应，动态公告表只用于定位，不以搜索摘要代替原件。",
    })


if __name__ == "__main__":
    main()
