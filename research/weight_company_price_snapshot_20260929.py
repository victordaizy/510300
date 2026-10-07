"""保存四个观察标的截至9月28日收盘的报价；不计算策略收益。"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_weight_company_driver_bridge_20260929/prices"
SYMBOLS = ["sh510300", "sz300308", "sz300502", "sh688256"]
CUTOFF = "2026-09-28"


def get(item):
    key, url, params, referer = item
    receipt = {"key": key, "url": url, "params": params, "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "attempts": 1}
    try:
        r = requests.get(url, params=params, headers={"Referer": referer, "User-Agent": "Mozilla/5.0"}, timeout=(8, 20))
        receipt["http_status"] = r.status_code
        r.raise_for_status()
        (OUT / f"{key}.raw").write_bytes(r.content)
        receipt["status"] = "RAW_SAVED"
        if key == "sina_previous_close":
            text = r.content.decode("gb18030")
            rows = {}
            for symbol, payload in re.findall(r'var hq_str_([a-z]{2}\d{6})="([^"]*)";', text):
                fields = payload.split(",")
                if len(fields) >= 32 and fields[30] == "2026-09-29":
                    rows[symbol] = {"quote_date": fields[30], "quote_time": fields[31], "previous_close": float(fields[2]), "previous_trading_date": CUTOFF}
            return receipt, rows
        text = r.text.strip()
        if not text.startswith("{"):
            text = text.split("=", 1)[1]
        data = json.loads(text.rstrip(";"))
        block = data["data"][key]
        rows = block.get("day", [])
        kept = [{"date": x[0], "open": float(x[1]), "close": float(x[2]), "high": float(x[3]), "low": float(x[4])} for x in rows if "2026-08-01" <= x[0] <= CUTOFF]
        return receipt, kept
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        receipt.update({"status": "FAILED_OR_UNUSABLE_NO_RETRY", "error": str(exc)})
        return receipt, None


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("价格快照已保存，不能覆盖其原接收时间。")
    OUT.mkdir(parents=True, exist_ok=True)
    items = [(symbol, "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get", {"param": f"{symbol},day,2026-08-01,{CUTOFF},80,"}, "https://gu.qq.com/") for symbol in SYMBOLS]
    items.append(("sina_previous_close", "https://hq.sinajs.cn/list=" + ",".join(SYMBOLS), None, "https://finance.sina.com.cn/"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(get, items))
    sources = {r[0]["key"]: r[1] for r in responses}
    second = sources.get("sina_previous_close") or {}
    comparisons = []
    for symbol in SYMBOLS:
        rows = sources.get(symbol) or []
        last = next((row for row in reversed(rows) if row["date"] == CUTOFF), None)
        reference = second.get(symbol)
        close = last["close"] if last else None
        same = bool(close is not None and reference and abs(close - reference["previous_close"]) < 1e-8)
        comparisons.append({"symbol": symbol, "date": CUTOFF, "tencent_unadjusted_close": close, "sina_previous_close": reference["previous_close"] if reference else None, "same_date_crosscheck": same, "status": "TWO_VENDOR_CLOSE_AGREE" if same else "NOT_CROSSCHECKED_OR_CONFLICT"})
    output = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "cutoff": CUTOFF, "source_type": "直接取得行情供应商报价，不称为交易所官方行情", "comparisons": comparisons, "raw_daily_rows": {s: sources.get(s) for s in SYMBOLS}, "receipts": [r[0] for r in responses], "strategy_return_computed": False, "corporate_action_adjusted": False, "usage": "最新收盘价定位；未复权区间价格不能直接当作持有总回报，盘中价格不加入收盘序列。"}
    (OUT / "result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
