"""固定行业和前十大权重观察组，补充当前阶段的传导覆盖。"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_broad_index_driver_bridge_v1"
SECTORS = {"000908": "能源", "000909": "原材料", "000910": "工业", "000911": "可选消费",
           "000912": "主要消费", "000913": "医药卫生", "000914": "金融和房地产", "000915": "信息技术",
           "000916": "通信服务", "000917": "公用事业"}
STOCKS = {"sz300750": "宁德时代", "sh600519": "贵州茅台", "sh601899": "紫金矿业", "sz000333": "美的集团", "sh603259": "药明康德"}
BASE, END = "2026-08-31", "2026-09-28"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(item):
    key, kind, url, params = item
    receipt = {"key": key, "kind": kind, "url": url, "params": params, "received_at": now(), "attempts": 1}
    value = None
    try:
        response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/" if kind == "quote" else "https://gu.qq.com/"}, timeout=(8, 25))
        content = response.content
        receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        path = OUT / "sources" / (key + (".pdf" if content.startswith(b"%PDF") else ".raw"))
        path.write_bytes(content)
        response.raise_for_status()
        if kind == "price":
            symbol, mode = key.rsplit("_", 1)
            text = response.text.strip()
            if not text.startswith("{"):
                text = text.split("=", 1)[1]
            block = json.loads(text.rstrip(";"))["data"][symbol]
            field = "qfqday" if mode == "qfq" and "qfqday" in block else "day"
            rows = [{"date": x[0], "open": float(x[1]), "close": float(x[2]), "high": float(x[3]), "low": float(x[4])} for x in block.get(field, []) if BASE <= x[0] <= END]
            value = {"symbol": symbol, "field": field, "requested_mode": mode, "rows": rows}
            receipt.update(status="价格序列已保存", rows=len(rows))
        elif kind == "quote":
            value = {}
            for symbol, payload in re.findall(r'var hq_str_([a-z]{2}\d{6})="([^"]*)";', content.decode("gb18030")):
                parts = payload.split(",")
                if len(parts) >= 32:
                    value[symbol] = {"date": parts[30], "time": parts[31], "previous_close": float(parts[2])}
            receipt.update(status="昨收报价已保存", symbols=len(value))
        elif kind == "pdf":
            if not content.startswith(b"%PDF"):
                raise ValueError("未返回PDF")
            with pdfplumber.open(path) as pdf:
                value = [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
            save(OUT / "sources" / f"{key}_pages.json", value)
            receipt.update(status="指数资料原件已保存", pages=len(value))
        else:
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            value = soup.get_text("\n", strip=True)
            (OUT / "sources" / f"{key}.txt").write_text(value, encoding="utf-8")
            receipt["status"] = "政策页面已保存"
    except (requests.RequestException, ValueError, KeyError, IndexError, OSError) as exc:
        receipt.update(status="失败，不自动重试", error=str(exc))
    save(OUT / "receipts" / f"{key}.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k in ["key", "status", "rows", "pages", "error"]}, ensure_ascii=False), flush=True)
    return receipt, value


def main():
    if (OUT / "source_result.json").exists():
        raise SystemExit("本轮来源已完成，不覆盖或重复请求。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    (OUT / "receipts").mkdir(exist_ok=True)
    save(OUT / "scope.json", {
        "recorded_at": now(), "previous_goal_turn": "PROGRESS_PRICE_RESPONSE_POLICY_SCOPE_AND_COMPLETED_BUYBACK",
        "selection": "完整10个不重复行业组；沿8月末前十大权重补齐此前未覆盖的5家公司，不按本轮表现择优。",
        "base_close": BASE, "end_close": END, "sectors": SECTORS, "new_stocks": STOCKS,
        "already_seen": "已看到宽基及5家公司阶段价格，9月28日行业下跌媒体摘要和电池规划；本轮是知情诊断。",
        "sector_mapping": "000914同时覆盖金融和房地产；不额外叠加000952。当前行业权重取8月末官方四舍五入值，权重乘行业价格变化仅是近似桥接。",
        "new_accounts": 0, "new_strategy_return_tests": 0, "maximum_requests": 25,
        "research_question": "月度与最新一日调整分别集中于哪些行业，下一阶段哪种上游原因最值得追踪？",
    })
    quote_symbols = ["sh" + code for code in SECTORS] + list(STOCKS)
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    items = [(s + "_raw", "price", url, {"param": f"{s},day,{BASE},{END},80,"}) for s in quote_symbols]
    items += [(s + "_qfq", "price", url, {"param": f"{s},day,{BASE},{END},80,qfq"}) for s in STOCKS]
    items += [("sina_previous_close", "quote", "https://hq.sinajs.cn/list=" + ",".join(quote_symbols), None)]
    for code in ["000914", "000916"]:
        items.append((code + "_factsheet", "pdf", f"https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/{code}factsheet.pdf", None))
    items += [
        ("battery_policy", "html", "https://www.miit.gov.cn/gyhxxhb/jgsj/dzxxsnew/zcwj/art/2026/art_6c2161b398414389a1bcc655d2113fa0.html", None),
        ("battery_policy_explanation", "html", "https://www.miit.gov.cn/zwgk/zcjd/art/2026/art_2d166bf3c90e45dfbdcd5fbaae9c9b17.html", None),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, items))
    save(OUT / "source_result.json", {"recorded_at": now(), "receipts": [x[0] for x in results], "responses": {x[0]["key"]: x[1] for x in results}})


if __name__ == "__main__":
    main()
