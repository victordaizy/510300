"""取得固定观察组短阶段价格及必要除息原件，保留原接收时点。"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_current_price_transmission_v1"
BASE = "2026-08-31"
END = "2026-09-28"
STOCKS = ["sz300308", "sz300502", "sh688256", "sh600036", "sh601318"]
SYMBOLS = ["sh510300", "sh000300"] + STOCKS


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(item):
    key, kind, url, params = item
    receipt = {"key": key, "kind": kind, "url": url, "params": params, "at": now(), "attempts": 1}
    data = None
    try:
        response = requests.get(url, params=params, timeout=(8, 25), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/" if kind == "quote" else "https://gu.qq.com/"})
        content = response.content
        receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        suffix = ".pdf" if kind == "pdf" and content.startswith(b"%PDF") else ".raw"
        path = OUT / "sources" / (key + suffix)
        path.write_bytes(content)
        response.raise_for_status()
        if kind == "kline":
            symbol, mode = key.rsplit("_", 1)
            text = response.text.strip()
            if not text.startswith("{"):
                text = text.split("=", 1)[1]
            block = json.loads(text.rstrip(";"))["data"][symbol]
            preferred = "qfqday" if mode == "qfq" else "day"
            field = preferred if preferred in block else "day"
            raw_rows = block.get(field, [])
            rows = [{"date": x[0], "open": float(x[1]), "close": float(x[2]), "high": float(x[3]), "low": float(x[4])} for x in raw_rows if BASE <= x[0] <= END]
            data = {"symbol": symbol, "requested_mode": mode, "series_field": field, "rows": rows}
            receipt.update(status="行情响应已保存", series_field=field, rows=len(rows))
        elif kind == "quote":
            data = {}
            for symbol, payload in re.findall(r'var hq_str_([a-z]{2}\d{6})="([^"]*)";', content.decode("gb18030")):
                fields = payload.split(",")
                if len(fields) >= 32:
                    data[symbol] = {"quote_date": fields[30], "quote_time": fields[31], "previous_close": float(fields[2])}
            receipt["status"] = "昨收报价已保存"
        elif kind == "pdf":
            if not content.startswith(b"%PDF"):
                raise ValueError("返回内容不是PDF")
            with pdfplumber.open(path) as pdf:
                data = [{"pdf_page": i+1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
            save(OUT / "sources" / f"{key}_pages.json", data)
            receipt.update(status="公司除息原件已保存", pages=len(data))
        else:
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            data = soup.get_text("\n", strip=True)
            (OUT / "sources" / f"{key}.txt").write_text(data, encoding="utf-8")
            receipt["status"] = "供应商分红页已保存"
    except (requests.RequestException, ValueError, KeyError, IndexError, OSError) as exc:
        receipt.update(status="请求或字段读取失败，不自动重试", error=str(exc))
    save(OUT / "receipts" / f"{key}.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k in ["key", "status", "http_status", "rows", "pages", "error"]}, ensure_ascii=False), flush=True)
    return receipt, data


def main():
    if (OUT / "source_result.json").exists():
        raise SystemExit("本轮价格来源已完成，不覆盖快照或重复请求。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    (OUT / "receipts").mkdir(exist_ok=True)
    save(OUT / "scope.json", {
        "created_at": now(), "previous_goal_turn": "PROGRESS_FINANCIAL_EARNINGS_CAUSES_AND_PRE_RELEASE_JUDGMENT",
        "question": "已研究的公司在全部半年报公开后的当前月度阶段，价格怎样变化，能否解释指数表现？",
        "base_close": BASE, "end_close": END, "symbols": SYMBOLS,
        "selection": "沿用既定五家公司，按全部H1公开后的月末基准至最新完整交易日；不按本轮收益更换公司或起点。",
        "already_seen": "已知三家科技相关公司及ETF的月末、9月28日报价；本轮为知情价格诊断，不冒充盲测或当时交易。已知平安9月10日派息0.98元。",
        "price_kind": "供应商未复权与前复权字段分别保留；前复权价格比不冒充完整账户收益。",
        "new_return_strategy_tests": 0, "new_accounts": 0, "orders_authorized": False,
        "maximum_new_http_requests": 12,
    })
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    items = [(s + "_qfq", "kline", url, {"param": f"{s},day,{BASE},{END},80,qfq"}) for s in ["sh510300"] + STOCKS]
    items += [(s + "_raw", "kline", url, {"param": f"{s},day,{BASE},{END},80,"}) for s in ["sh600036", "sh601318", "sh000300"]]
    items += [
        ("sina_previous_close", "quote", "https://hq.sinajs.cn/list=" + ",".join(SYMBOLS), None),
        ("pingan_dividend", "pdf", "https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2026/2026-9/2026-09-03/12580396.PDF", None),
        ("etf_dividend_vendor", "html", "https://q.fund.sohu.com/q/fh.php?code=510300", None),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(fetch, items))
    save(OUT / "source_result.json", {"recorded_at": now(), "receipts": [x[0] for x in responses], "responses": {x[0]["key"]: x[1] for x in responses}, "reused_raw_prices": "reports/research/510300_weight_company_driver_bridge_20260929/prices/result.json"})


if __name__ == "__main__":
    main()
