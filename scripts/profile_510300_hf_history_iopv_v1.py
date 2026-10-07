"""核对新增历史资料中的正值IOPV，仅报告原字段范围与变化，不准入交易。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from profile_510300_hf_history_v1 import BASE, OUT, digest


def main() -> None:
    target = OUT / "02_IOPV正值日期字段核对.json"
    if target.exists():
        raise SystemExit("字段核对结果已存在，拒绝覆盖。")
    manifest_path = BASE / "coverage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    days = []
    for item in manifest["files"]:
        if item["stream"] != "行情" or item.get("positive_iopv_rows", 0) == 0:
            continue
        frame = pq.read_table(item["path"], columns=["time", "price", "iopv"]).to_pandas()
        positive = frame.loc[frame.iopv.gt(0)].copy()
        ratios = positive.loc[positive.price.gt(0), "iopv"] / positive.loc[positive.price.gt(0), "price"]
        after = positive.loc[positive.time.ge(150500000)]
        around = frame.loc[frame.time.ge(150500000) & frame.time.le(150700000)]
        day = {
            "date": item["date"], "source_path": item["path"], "source_sha256": item["sha256"],
            "source_rows": len(frame), "positive_rows": len(positive),
            "null_rows": int(frame.iopv.isna().sum()), "zero_rows": int(frame.iopv.eq(0).sum()),
            "negative_rows": int(frame.iopv.lt(0).sum()),
            "first_positive_time": int(positive.time.min()), "last_positive_time": int(positive.time.max()),
            "raw_iopv_min": float(positive.iopv.min()), "raw_iopv_max": float(positive.iopv.max()),
            "raw_iopv_distinct_values": int(positive.iopv.nunique()),
            "median_raw_iopv_divided_by_raw_price": float(ratios.median()) if len(ratios) else None,
            "positive_rows_at_or_after_1505": len(after),
            "raw_iopv_distinct_values_at_or_after_1505": int(after.iopv.nunique()),
            "first_source_rows_1505_to_1507": around.head(5).to_dict(orient="records"),
            "last_source_rows_1505_to_1507": around.tail(5).to_dict(orient="records"),
        }
        days.append(day)
    payload = {
        "status": "POSITIVE_IOPV_SOURCE_FIELDS_PROFILED_NOT_ADMITTED",
        "days": days,
        "positive_rows_total": sum(day["positive_rows"] for day in days),
        "scope": "只核对原字段；正值与数值接近现价不证明IOPV计算时点、单位合同、独立性或实际接收时刻。",
        "strategy_returns_computed": False,
        "provenance": [
            {"path": str(manifest_path), "sha256": digest(manifest_path)},
            {"path": str(Path(__file__)), "sha256": digest(Path(__file__))},
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    index = [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "FILE_INDEX.csv"]
    pd.DataFrame(index).to_csv(OUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
