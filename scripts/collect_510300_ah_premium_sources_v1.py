"""免费公开 A/H 溢价日线的有限来源探测；保留原始响应，不读取策略收益。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_0800")
OUT = ROOT / "data/raw/market/510300_ah_premium_source_v1" / STAMP


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = {"study_id": "510300_AH_PREMIUM_INCREMENT_V1", "collected_at": STAMP,
               "strategy_returns_read": False, "paid_sources_used": False,
               "historical_first_delivery_timestamp_proven": False, "sources": []}
    sources = [
        ("sina_hsahp", "https://finance.sina.com.cn/stock/hkstock/HSAHP/klc2_kl.js", {}, "js"),
        ("eastmoney_hsahp", "https://push2his.eastmoney.com/api/qt/stock/kline/get",
         {"secid": "100.HSAHP", "klt": "101", "fqt": "0", "lmt": "10000", "end": "20260913",
          "fields1": "f1,f2,f3,f4,f5,f6,f7,f8", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
          "ut": "f057cbcbce2a86e2866ab8877db1d059"}, "json"),
        ("hsi_factsheet", "https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/factsheets/ahpremiume.pdf", {}, "pdf"),
        ("hsi_history_pricing", "https://www.hsi.com.hk/static/uploads/contents/en/products/data_subscription/pdf_eng.pdf", {}, "pdf"),
    ]
    for name, url, params, ext in sources:
        entry = {"name": name, "url": url, "params": params,
                 "requested_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        try:
            response = requests.get(url, params=params, timeout=(8, 20),
                                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
            path = OUT / f"{name}.{ext}"
            path.write_bytes(response.content)
            entry.update({"http_status": response.status_code, "response_url": response.url,
                          "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(),
                          "raw_path": path.relative_to(ROOT).as_posix(),
                          "received_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                          "http_date": response.headers.get("Date"), "last_modified": response.headers.get("Last-Modified")})
            response.raise_for_status()
            if name == "sina_hsahp":
                from akshare.stock.cons import hk_js_decode
                import py_mini_racer
                context = py_mini_racer.MiniRacer()
                context.eval(hk_js_decode)
                encoded = response.text.split("=", 1)[1].split(";", 1)[0].replace('"', "")
                frame = pd.DataFrame(context.call("d", encoded))
                frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
            elif name == "eastmoney_hsahp":
                payload = response.json()
                frame = pd.DataFrame([row.split(",") for row in payload["data"]["klines"]],
                                     columns=["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "change_pct", "change", "turnover"])
                frame["date"] = pd.to_datetime(frame["date"])
                entry["vendor_symbol"] = payload["data"].get("code")
                entry["vendor_name"] = payload["data"].get("name")
            else:
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("官方文档响应不是 PDF")
                entry["status"] = "PASS_RAW_DOCUMENT_SAVED"
                receipt["sources"].append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                continue
            for column in ["open", "close", "high", "low"]:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            frame = frame.sort_values("date").reset_index(drop=True)
            if frame.empty or frame.date.duplicated().any() or frame.close.isna().any() or frame.close.le(0).any():
                raise ValueError("日线为空、日期重复或收盘值无效")
            parsed = OUT / f"{name}.parquet"
            frame.to_parquet(parsed, index=False)
            entry.update({"status": "PASS_PARSED_DAILY_SOURCE", "parsed_path": parsed.relative_to(ROOT).as_posix(),
                          "parsed_sha256": hashlib.sha256(parsed.read_bytes()).hexdigest(),
                          "rows": len(frame), "first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date()),
                          "columns": list(frame.columns)})
        except Exception as exc:
            entry.update({"status": "EXTERNAL_FREE_SOURCE_OR_PARSE_FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1200]})
        receipt["sources"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    receipt["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    (OUT / "source_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("来源回执：", OUT / "source_receipt.json", flush=True)


if __name__ == "__main__":
    main()
