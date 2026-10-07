"""只补充明确失败的同一版本请求，合并保存档案，保留原失败结果。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.nfci_graph_vintages_source_v1_2 import parse, request_batch, save, now

OUT = ROOT / "reports/research/510300_nfci_graph_vintages_source_v1_3"
PARENT = ROOT / "reports/research/510300_nfci_graph_vintages_source_v1_2"


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("本次保存档案完成结果已存在，不能重复取得。")
    previous = json.loads((PARENT / "result.json").read_text(encoding="utf-8"))
    source = ROOT / previous["snapshot"]
    original_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    snapshot = ROOT / "data/raw/macro/510300_nfci_graph_vintages_source_v1_3" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f_0800")
    (snapshot / "raw").mkdir(parents=True, exist_ok=False)
    (snapshot / "receipts").mkdir()
    save(snapshot / "protocol.json", {"at": now(), "study_id": "510300_NFCI_VINTAGES_SAVED_COMPLETION_V1",
        "parent": previous, "parent_result_sha256": hashlib.sha256((PARENT / "result.json").read_bytes()).hexdigest(),
        "requested_vintages": original_protocol["requested_vintages"],
        "rule": "原成功批次逐文件复用；只对明确失败批次的相同系列和版本日补充一次，来源定义与时点不变。",
        "failed_original_receipts_preserved": True, "new_models": 0, "new_accounts": 0})
    rows, inherited, failed = [], [], []
    for receipt_file in sorted((source / "receipts").glob("*.json")):
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if receipt["status"] == "SAVED_AND_VERSION_COLUMNS_CHECKED":
            path = ROOT / receipt["raw_path"]
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
            rows.extend(parse(raw, receipt["pairs"], receipt["batch"]))
            inherited.append({"receipt": receipt_file.relative_to(ROOT).as_posix(), "raw_path": receipt["raw_path"], "sha256": receipt["sha256"]})
        else:
            dates = sorted({p[1] for p in receipt["pairs"]})
            part = request_batch(snapshot, receipt["batch"], dates)
            rows.extend(part)
            if not part:
                failed.append(receipt["batch"])
            print(f"失败批次 {receipt['batch']} 同请求补充：取得 {len(part)} 条版本状态。", flush=True)
    save(snapshot / "inherited_sources.json", inherited)
    frame = pd.DataFrame(rows).sort_values(["vintage_date", "series"]).reset_index(drop=True)
    assert not frame.duplicated(["series", "vintage_date"]).any()
    dates = original_protocol["requested_vintages"]
    expected = {(s, pd.Timestamp(d)) for d in dates for s in ["NFCI", "NFCICREDIT", "NFCIRISK"]}
    actual = set(zip(frame.series, frame.vintage_date))
    complete = not failed and actual == expected
    frame.to_parquet(snapshot / "vintage_features.parquet", index=False)
    if complete:
        obs = frame.pivot(index="vintage_date", columns="series", values="observation_date")
        assert obs.nunique(axis=1).eq(1).all()
    result = {"at": now(), "status": "COMPLETE_OFFICIAL_ARCHIVAL_VINTAGE_FEATURES" if complete else "PARTIAL_ARCHIVE_NOT_ADMITTED",
              "snapshot": snapshot.relative_to(ROOT).as_posix(), "series": ["NFCI", "NFCICREDIT", "NFCIRISK"],
              "requested_vintages": len(dates), "version_rows": len(frame), "first_vintage": dates[0], "last_vintage": dates[-1],
              "latest_observation_date": frame.observation_date.max().date().isoformat(),
              "reused_successful_requests": len(inherited), "supplementary_requests": len(previous["failed_batches"]),
              "failed_batches": failed, "missing_series_vintage_pairs": len(expected - actual),
              "new_model_fits": 0, "new_accounts": 0, "independent_market_observations": 0,
              "saved_raw_recomputation": "PASS_ALL_ADMITTED_VERSION_COLUMNS", "goal_achieved": False}
    save(snapshot / "result.json", result)
    save(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
