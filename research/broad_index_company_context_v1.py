"""追踪主要行业压力的经营与政策约束，不将旧消息变成新冲击。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_broad_index_driver_bridge_v1"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(item):
    key, kind, url = item
    receipt = {"key": key, "url": url, "received_at": now(), "attempts": 1}
    value = None
    try:
        response = requests.get(url, timeout=(8, 30), headers={"User-Agent": "Mozilla/5.0"})
        content = response.content
        receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        path = OUT / "sources" / f"{key}{'.pdf' if content.startswith(b'%PDF') else '.html'}"
        path.write_bytes(content)
        response.raise_for_status()
        if kind == "pdf":
            if not content.startswith(b"%PDF"):
                raise ValueError("返回内容不是PDF")
            with pdfplumber.open(path) as pdf:
                value = [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages[:40])]
                receipt["total_pages"] = len(pdf.pages)
            save(OUT / "sources" / f"{key}_pages.json", value)
            receipt.update(status="PDF已保存并读取前40页以内", extracted_pages=len(value))
        else:
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            value = soup.get_text("\n", strip=True)
            (OUT / "sources" / f"{key}.txt").write_text(value, encoding="utf-8")
            receipt["status"] = "原页面已保存"
    except (requests.RequestException, OSError, ValueError) as exc:
        receipt.update(status="来源失败，不自动重试", error=str(exc))
    save(OUT / "receipts" / f"{key}.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt, value


def main():
    if (OUT / "company_source_result.json").exists():
        raise SystemExit("公司与税负背景已保存，不覆盖。")
    save(OUT / "company_followup_scope.json", {
        "recorded_at": now(), "reason": "全行业固定窗口显示工业及原材料为主要月度压力，追踪既定前十大权重中的宁德与紫金。",
        "selection_is_post_price_diagnostic": True,
        "not_new_strategy_test": True, "maximum_requests": 4,
        "excluded_search_hits": ["1225439717属于豪鹏科技，不能当宁德回应", "1225579080属于安克创新，不能当宁德分红"],
        "next_question": "销量、售价、税负、成本与矿山产能如何分别改变盈利路径？",
    })
    items = [
        ("catl_h1", "pdf", "https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-07-25/8e6750da-b178-4be9-a74a-4b1d8fc42c58.PDF"),
        ("battery_consumption_tax", "html", "https://szs.mof.gov.cn/zhengcefabu/202607/t20260717_3993743.htm"),
        ("battery_consumption_tax_qa", "html", "https://www.chinatax.gov.cn/chinatax/c102414/c5252006/content.html"),
        ("zijin_h1_company", "html", "https://www.zjky.cn/investor/2026yeji.htm"),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(fetch, items))
    save(OUT / "company_source_result.json", {"recorded_at": now(), "receipts": [x[0] for x in values], "responses": {x[0]["key"]: x[1] for x in values}})


if __name__ == "__main__":
    main()
