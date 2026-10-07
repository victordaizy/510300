"""按既定指数权重入口读取公司与上游原因，保存本轮有限资料。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_weight_company_driver_bridge_20260929"
SOURCES = [
    ("innolight_h1", "pdf", "https://static.cninfo.com.cn/finalpage/2026-08-22/1225491753.PDF"),
    ("eoptolink_h1", "pdf", "https://static.cninfo.com.cn/finalpage/2026-08-25/1225499406.PDF"),
    ("cambricon_h1", "pdf", "https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF"),
    ("eoptolink_september_ir", "pdf", "https://static.cninfo.com.cn/finalpage/2026-09-12/1225562013.PDF"),
    ("microsoft_fy26q3_call", "html", "https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3"),
    ("microsoft_fy26q4_call", "html", "https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q4"),
]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch(item):
    name, kind, url = item
    started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt = {"name": name, "kind": kind, "url": url, "started_at": started, "attempts": 1}
    path = OUT / "sources" / f"{name}.{kind}"
    try:
        response = requests.get(url, timeout=(8, 25), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update({"http_status": response.status_code, "final_url": response.url})
        response.raise_for_status()
        if kind == "pdf" and not response.content.startswith(b"%PDF"):
            raise ValueError("响应不包含PDF原件")
        path.write_bytes(response.content)
        receipt.update({"sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content)})
        if kind == "pdf":
            with pdfplumber.open(path) as doc:
                examined = min(50, len(doc.pages))
                pages = [{"pdf_page": i + 1, "text": doc.pages[i].extract_text() or ""} for i in range(examined)]
                receipt.update({"total_pages": len(doc.pages), "text_extracted_pages": examined})
            (OUT / "sources" / f"{name}_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            response.encoding = "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            content = soup.get_text("\n", strip=True)
            (OUT / "sources" / f"{name}.txt").write_text(content, encoding="utf-8")
            receipt["text_characters"] = len(content)
        receipt["status"] = "SAVED_ORIGINAL_REQUIRES_RELEVANT_CONTENT_REVIEW"
    except (requests.RequestException, ValueError) as exc:
        receipt.update({"status": "FAILED_NO_AUTOMATIC_RETRY", "error": str(exc)})
    return receipt


def main():
    if (OUT / "source_receipts.json").exists():
        raise RuntimeError("本轮来源已经保存，请复用原件；后续取证另存新记录。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    save("scope.json", {
        "study_id": "510300_WEIGHT_COMPANY_DRIVER_BRIDGE_20260929",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "previous_goal_turn_classification": "PROGRESS_CURRENT_MACRO_AND_INDEX_WEIGHT_BRIDGE",
        "selection": "沿上一轮已选定的中际旭创、新易盛、寒武纪，依据2026年8月末官方权重和电子需求传导确定，不按未来股价选样。",
        "question": "上游资本支出调整的真实原因，如何经过交付和公司经营转成未来盈利，并与价格已经反映的部分比较？",
        "upstream_case": "微软FY26Q3与FY26Q4交流用于区分需求、元器件价格与租赁会计口径；不预设三家公司的具体客户关系。",
        "already_seen": "已见公司中报标题及部分业绩摘要、微软1900亿至1750亿美元口径解释、8月末指数权重；未读取本案例价格响应。",
        "maximum_new_documents": 6,
        "pdf_initial_text_page_limit": 50,
        "forecast_horizon": "未来1至4周的判断更新；全年投资预算不能直接等于同期采购额。",
        "new_strategy_return_tests": 0,
        "new_accounts": 0,
        "orders_authorized": False,
        "goal_achieved": False,
    })
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, SOURCES))
    save("source_receipts.json", receipts)
    print(json.dumps(receipts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
