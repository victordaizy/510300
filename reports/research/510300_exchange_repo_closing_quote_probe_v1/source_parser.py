"""只读解析已保存的GC007公开行情响应；不把报价历史等同已认证信息。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import py_mini_racer
from akshare.stock.cons import hk_js_decode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_exchange_repo_closing_quote_probe_v1"


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("报价探查已有终态，禁止重复解析覆盖。")
    receipt = read(OUT / "request_result.json")
    raw = ROOT / receipt["raw_path"]
    assert receipt["status"] == "HTTP_OK_UNPARSED" and digest(raw) == receipt["sha256"]
    text = raw.read_text(encoding="utf-8")
    matched = re.fullmatch(r'\s*var\s+KLC_KL_sh204007\s*=\s*("[^"]*")\s*;?\s*(?:/\*[\s\S]*?\*/\s*)?', text)
    assert matched, "未找到预期证券代码的压缩行情变量。"
    encoded = json.loads(matched.group(1))
    with py_mini_racer.MiniRacer() as runtime:
        runtime.eval(hk_js_decode)
        records = runtime.call("d", encoded)
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame.date, utc=True).dt.tz_convert(None).dt.normalize()
    assert frame.date.is_unique and frame.date.is_monotonic_increasing
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        assert np.isfinite(frame[column]).all()
    invalid = frame[(frame.high < frame[["open", "close"]].max(axis=1))
                    | (frame.low > frame[["open", "close"]].min(axis=1)) | (frame.low > frame.high)]
    frame.to_parquet(OUT / "gc007_quote_history.parquet", index=False)
    recent = frame[frame.date.ge("2026-09-14")][["date", "open", "high", "low", "close"]]
    save(OUT / "result.json", {"at": now(), "study_id": "510300_EXCHANGE_REPO_CLOSING_QUOTE_PROBE_V1",
         "status": "SECONDARY_QUOTE_SERIES_REQUIRES_FIELD_AND_INDEPENDENT_CROSSCHECK",
         "rows": len(frame), "first_date": frame.date.min(), "last_date": frame.date.max(),
         "columns": frame.columns.tolist(), "invalid_ohlc_rows": invalid.to_dict("records"),
         "recent_quotes": recent.to_dict("records"), "secondary_source": True,
         "historical_first_publication_verified": False, "new_accounts": 0, "new_strategy_returns": 0,
         "next_required_evidence": "确认日线日期和报价单位，并用独立行情来源核对收盘，不与全天定盘硬性要求相等。",
         "goal_achieved": False}, True)
    (OUT / "source_parser.py").write_bytes(Path(__file__).read_bytes())
    print(f"取得次级报价历史{len(frame)}行，{frame.date.min().date()}至{frame.date.max().date()}；仍需独立核对，未计算新收益。", flush=True)


if __name__ == "__main__":
    run()
