"""保存有限的当期官方材料，供原因与条件预测研究；不运行策略或交易。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_current_driver_outlook_20260929"
SOURCES = [
    ("fed_september_statement", "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm"),
    ("nbs_august_economy", "https://www.stats.gov.cn/sj/zxfb/202609/t20260915_1965307.html"),
    ("nbs_august_explanation", "https://www.stats.gov.cn/sj/zxfbhjd/202609/t20260915_1965332.html"),
    ("nbs_august_price", "https://www.stats.gov.cn/sj/zxfbhjd/202609/t20260909_1965262.html"),
    ("nbs_august_price_explanation", "https://www.stats.gov.cn/xxgk/jd/sjjd2020/202609/t20260909_1965261.html"),
    ("nbs_august_pmi", "https://www.stats.gov.cn/sj/zxfb/202608/t20260831_1965154.html"),
    ("chinamoney_home", "https://www.chinamoney.com.cn/chinese/index.html"),
    ("sse_holiday", "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml"),
]


def fetch(item):
    name, url = item
    receipt = {"name": name, "url": url, "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "attempts": 1}
    try:
        response = requests.get(url, timeout=(8, 20), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update({"http_status": response.status_code, "final_url": response.url, "bytes": len(response.content)})
        if response.status_code != 200:
            receipt["status"] = "HTTP_FAILURE_NO_SOURCE_ADMISSION"
            return receipt
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        source_text = soup.get_text("\n", strip=True)
        (OUT / "sources" / f"{name}.html").write_bytes(response.content)
        (OUT / "sources" / f"{name}.txt").write_text(source_text, encoding="utf-8")
        receipt.update({"status": "SAVED_HTTP200_CONTENT_REVIEW_REQUIRED", "sha256": hashlib.sha256(response.content).hexdigest(), "text_characters": len(source_text)})
    except requests.RequestException as exc:
        receipt.update({"status": "TRANSPORT_FAILURE_NO_RETRY", "error": str(exc)})
    return receipt


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("本日期快照已完成；后续资料请使用新的日期目录，保留原始判断。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    scope = {
        "study_id": "510300_CURRENT_DRIVER_OUTLOOK_20260929",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "question": "最新可获得信息中，哪些上游变化主导未来1—4周的资金、盈利与估值约束？",
        "already_seen": "已看到9月美联储声明、8月中国PPI和PMI、9月28日中国货币网盘中摘要及休市安排；此处不声称事前盲选或独立验证。",
        "horizon": "研究记录生成后的1—4周；国庆休市按官方安排单列",
        "factor_selection_basis": "可观察的供给需求冲击、政策反应和主体约束；不按未来收益挑选。",
        "new_strategy_backtests": 0,
        "new_accounts": 0,
        "current_price_claim_requires_dated_source": True,
        "no_market_consensus_does_not_imply_no_expectations": True,
        "prior_failures_unchanged": True,
        "target_net_sharpe": 1.2,
        "goal_achieved": False,
    }
    (OUT / "scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(fetch, SOURCES))
    (OUT / "source_receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
