"""定向保存两家既定金融权重公司的原因分析原件，不运行收益筛选。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_financial_driver_bridge_v1"
DOCUMENTS = [
    ("cmb_h1", "https://static.cninfo.com.cn/finalpage/2026-08-29/1225530237.PDF", 65),
    ("pingan_h1", "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2026/2026-8/2026-08-21/12512669.PDF", 65),
    ("pingan_presentation", "https://static.cninfo.com.cn/finalpage/2026-08-21/1225487472.PDF", 56),
]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(item):
    name, url, cap = item
    receipt = {"at": now(), "url": url, "attempts": 1}
    try:
        response = requests.get(url, timeout=(8, 35), headers={"User-Agent": "Mozilla/5.0"})
        receipt["http_status"] = response.status_code
        data = response.content
        receipt.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if response.status_code != 200 or not data.startswith(b"%PDF"):
            (OUT / "sources" / f"{name}.response").write_bytes(data)
            receipt["status"] = "原件未取得"
        else:
            pdf_path = OUT / "sources" / f"{name}.pdf"
            pdf_path.write_bytes(data)
            with pdfplumber.open(pdf_path) as pdf:
                receipt["total_pages"] = len(pdf.pages)
                pages = [{"pdf_page": i+1, "text": page.extract_text() or ""} for i, page in enumerate(pdf.pages[:cap])]
            save(OUT / "sources" / f"{name}_pages.json", pages)
            receipt.update(status="公司原件已保存", extracted_pages=len(pages))
            words = ["净利息收益率", "短期投资波动", "一次性重大", "净利息收入", "关于净利息", "综合投资收益率", "分红险", "信贷成本"]
            receipt["relevant_pages"] = [p["pdf_page"] for p in pages if any(w in p["text"] for w in words)]
    except (requests.RequestException, OSError, ValueError) as exc:
        receipt.update(status="取得或读取失败", error=str(exc))
    save(OUT / "receipts" / f"{name}.json", receipt)
    print(json.dumps({"文档": name, **receipt}, ensure_ascii=False), flush=True)
    return receipt


def main():
    if (OUT / "source_result.json").exists():
        raise SystemExit("本轮来源工作已完成；不重复请求原接口。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    (OUT / "receipts").mkdir(exist_ok=True)
    save(OUT / "scope.json", {
        "created_at": now(),
        "previous_goal_turn": "PROGRESS_PRIOR_EXPECTATION_AND_CONDITIONAL_EARNINGS_HURDLES",
        "question": "利率、需求与资本市场变动如何通过金融权重公司的盈利形成放大或抵消？",
        "cohort": ["600036", "601318"],
        "selection": "按已保存8月末官方前十大权重中的金融公司选择，不按后续股价或业绩好坏选择。",
        "already_seen": "搜索已显示招行息差下降、平安净利润与营运利润增速不同及综合投资收益率下降；本轮明确为知情归因诊断。",
        "maximum_new_pdf_documents": 3,
        "new_accounts": 0, "new_return_tests": 0,
        "orders_authorized": False, "goal_achieved": False,
    })
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, DOCUMENTS))
    save(OUT / "source_result.json", {"at": now(), "documents": results})


if __name__ == "__main__":
    main()
