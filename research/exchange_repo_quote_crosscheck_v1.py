"""比较两个公开GC007报价源；只保存一致收盘，冲突保持未知。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.exchange_repo_closing_quote_probe_v1 import OUT
from research.selected_mix_reappraisal_v1 import read, save, now, digest


def run():
    if (OUT / "crosscheck_result.json").exists():
        raise RuntimeError("报价交叉核对已有终态。")
    receipt = read(OUT / "independent_quote_history_receipt.json")
    raw = ROOT / receipt["raw_path"]
    assert digest(raw) == receipt["sha256"] and receipt["status"] == "HTTP_OK_UNPARSED"
    payload = read(raw)
    assert payload["rc"] == 0
    data = payload["data"]
    assert data["code"] == "204007" and data["market"] == 1 and data["name"] == "GC007" and data["decimal"] == 3
    em = pd.DataFrame([row.split(",") for row in data["klines"]]).iloc[:, :7]
    em.columns = ["date", "open", "close", "high", "low", "volume", "amount"]
    em["date"] = pd.to_datetime(em.date)
    for column in em.columns[1:]:
        em[column] = pd.to_numeric(em[column], errors="raise")
        assert np.isfinite(em[column]).all()
    sina = pd.read_parquet(OUT / "gc007_quote_history.parquet")
    merged = em.merge(sina, on="date", how="outer", suffixes=("_em", "_sina"), validate="one_to_one")
    merged = merged[merged.date.between("2017-05-22", "2026-09-24")].sort_values("date").reset_index(drop=True)
    for column in ["open", "close", "high", "low"]:
        merged[column + "_agree"] = (merged[column + "_em"] - merged[column + "_sina"]).abs().le(1e-9)
    merged["gc_close_percent"] = merged.close_em.where(merged.close_agree)
    merged["source_status"] = np.where(merged.close_agree, "TWO_PUBLIC_PROVIDERS_CLOSE_AGREE", "CONFLICT_OR_MISSING_CLOSE_UNKNOWN")
    merged["historical_first_publication_verified"] = False
    merged.to_parquet(OUT / "crosschecked_closing_quotes.parquet", index=False)
    conflict = merged.loc[~merged.close_agree, ["date", "close_em", "close_sina", "source_status"]]
    save(OUT / "crosscheck_result.json", {"at": now(), "study_id": "510300_EXCHANGE_REPO_QUOTE_CROSSCHECK_V1",
         "status": "MATCHED_SECONDARY_CLOSES_READY_FOR_CONDITIONAL_DEVELOPMENT",
         "source_rows": len(merged), "agreed_closing_dates": int(merged.close_agree.sum()),
         "conflict_or_missing_closing_dates": len(conflict), "conflicts": conflict.to_dict("records"),
         "other_field_disagreements": {k: int((~merged[k + "_agree"]).sum()) for k in ["open", "high", "low"]},
         "admitted_field": "仅两个来源完全一致的原始收盘报价百分数；不使用成交额/量或开高低。冲突不择源、不填补。",
         "qualification": "这是两个公开行情查询的事后对账，仍不能认证历史初版或送达时刻；研究只能条件性使用。",
         "new_accounts": 0, "new_strategy_returns": 0, "goal_achieved": False}, True)
    (OUT / "crosscheck_code.py").write_bytes(Path(__file__).read_bytes())
    print(f"报价覆盖{len(merged)}日：收盘一致{int(merged.close_agree.sum())}日、冲突或缺失{len(conflict)}日保持未知。", flush=True)


if __name__ == "__main__":
    run()
