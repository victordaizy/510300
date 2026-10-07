"""取得指数同口径估值和带时点的ETF报价，用于价格与经营变化的桥接。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_index_repricing_odds_v1"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(item):
    key, kind, url, params = item
    receipt = {"key": key, "url": url, "params": params, "received_at": now(), "attempts": 1}
    value = None
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/" if kind == "quote" else "https://www.csindex.com.cn/"}
        response = requests.get(url, params=params, headers=headers, timeout=(8, 25))
        content = response.content
        receipt.update(http_status=response.status_code, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        (OUT / "sources" / (key + (".xls" if kind == "xls" else ".raw"))).write_bytes(content)
        response.raise_for_status()
        if kind == "xls":
            frame = pd.read_excel(BytesIO(content))
            value = json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))
            receipt.update(status="官方估值表已保存", columns=list(frame.columns), rows=len(frame))
        elif kind == "json":
            value = response.json()
            receipt.update(status="官方接口响应已保存", root_keys=list(value) if isinstance(value, dict) else None)
        elif kind == "csv":
            frame = pd.read_csv(BytesIO(content))
            value = json.loads(frame.to_json(orient="records", force_ascii=False))
            receipt.update(status="公开利率序列已保存", rows=len(frame), columns=list(frame.columns))
        elif kind == "html":
            response.encoding = "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            value = soup.get_text("\n", strip=True)
            (OUT / "sources" / (key + ".txt")).write_text(value, encoding="utf-8")
            receipt.update(status="公开预期页面已保存", characters=len(value))
        else:
            value = content.decode("gb18030")
            receipt["status"] = "供应商报价原文已保存，盘中时间单列"
        save(OUT / "sources" / (key + ".json"), value)
    except (requests.RequestException, ValueError, OSError, ImportError) as exc:
        receipt.update(status="取得失败，不自动重试", error=str(exc))
    save(OUT / "receipts" / (key + ".json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if key == "csi300_perf" and value is not None:
        print(json.dumps(value, ensure_ascii=False)[:1200], flush=True)
    if key == "csi300_indicator" and isinstance(value, list):
        print(json.dumps(value[:3], ensure_ascii=False), flush=True)
    return receipt, value


def main():
    if (OUT / "scope.json").exists():
        raise SystemExit("本轮来源已启动，复用已有响应，不重复请求。")
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    save(OUT / "scope.json", {
        "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS_FACTOR_MEANING_CAUSES_AND_TOP10_OPERATING_COVERAGE",
        "question": "近期指数价格变化是否可拆到盈利分母与估值；新经营信息需超过怎样的既有门槛才有净收益空间？",
        "base_date": "2026-08-31", "completed_price_date": "2026-09-28",
        "requests_first_stage": 3, "intraday_quote_is_historical_close": False,
        "new_companies": 0, "new_accounts": 0, "new_strategy_return_tests": 0,
        "not_a_parameter_search": True,
    })
    items = [
        ("csi300_indicator", "xls", "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/indicator/000300indicator.xls", None),
        ("csi300_perf", "json", "https://www.csindex.com.cn/csindex-home/perf/index-perf", {"indexCode": "000300", "startDate": "20260831", "endDate": "20260928"}),
        ("etf_intraday", "quote", "https://hq.sinajs.cn/list=sh510300", None),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, items))
    save(OUT / "source_result.json", {"recorded_at": now(), "receipts": [r[0] for r in rows]})


if __name__ == "__main__":
    main()
