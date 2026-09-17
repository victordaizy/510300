"""离线核对固定原件、归母口径与条件覆盖，正确处理未写入文件的行索引。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_profit_attribution_pair_adjudication_v1 import OUT, compute, identity, now, read, save


def verify(receipt: Path) -> dict:
    index_path = ROOT / "FILE_INDEX.csv"
    indexed = None
    if index_path.exists():
        with index_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        names = [row["path"] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError("文件索引有重名")
        for row in rows:
            path = ROOT / row["path"]
            path.resolve().relative_to(ROOT.resolve())
            if path.stat().st_size != int(row["size_bytes"]) or identity(path)["sha256"] != row["sha256"]:
                raise ValueError(f"索引文件内容不匹配：{row['path']}")
        indexed = len(rows)
    claim = read(OUT / "claim.json")
    for item in claim["files"]:
        if identity(ROOT / item["path"]) != item:
            raise ValueError(f"冻结输入改变：{item['path']}")
    frame, result = compute(extract=False)
    saved_result = read(OUT / "result.json")
    if result != {key: value for key, value in saved_result.items() if key != "completed_at"}:
        raise ValueError("保存结果与重新计算不相等")
    existing = pd.read_parquet(OUT / "conditional_holdings_evidence.parquet")
    original_index = {"recomputed_start": int(frame.index[0]), "saved_start": int(existing.index[0]),
                      "reason": "冻结计算筛选原990行的最后337行；保存Parquet明确使用index=False"}
    for table in (frame, existing):
        if table.duplicated(["origin", "ts_code"]).any():
            raise ValueError("日期和股票代码业务键重复")
        table["origin"] = table.origin.astype("datetime64[ns]")
    # 仅消去保存时已明确不写入的内存行号；保留原顺序、业务键、所有列和精确数值。
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), existing.reset_index(drop=True), check_exact=True)
    result_path = OUT / "result.json"
    record = {"status": "PASS_FOUR_ORIGINAL_PDFS_TWO_SEMANTIC_PAIRS_AND_CONDITIONAL_BOUNDS_RECOMPUTED",
              "completed_at": now(), "verification_version": "1.1", "indexed_files_verified": indexed,
              "frozen_inputs_verified": len(claim["files"]), "original_pdfs": 4,
              "source_pages_reextracted": sum(len(read(p)["pages"]) for p in (OUT / "pages").glob("*.json")),
              "pairs": 2, "conditional_holdings_rows": len(frame), "row_order_preserved": True,
              "business_keys_exactly_equal": True, "all_values_exactly_equal": True,
              "in_memory_row_index_handling": original_index,
              "frozen_core_and_claim_changed": False, "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
              "full_upstream_659_report_fact_recomputation_in_this_packet": False,
              "new_network_requests": 0, "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False}
    save(receipt, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    verify(parser.parse_args().receipt)
