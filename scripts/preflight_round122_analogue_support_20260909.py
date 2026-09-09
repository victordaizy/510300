"""只读八项输入和成熟身份，检查相似周期方法能否直接复用旧训练支持。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.market_path_exit_inputs_v1 import FEATURES
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_cycle_analogue_support_20260909"


def main():
    source = ROOT / "reports/research/510300_market_path_state_preflight_20260908/augmented_reference_samples.parquet"
    clocks_path = ROOT / "reports/research/510300_learned_cycle_exit_v1/saved_models.json"
    samples = pd.read_parquet(source, columns=["cycle_id", "origin_index", "exit_index", *FEATURES])
    clocks = json.loads(clocks_path.read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    rows = []
    for clock in clocks:
        t = int(clock["fit_index"])
        mature = samples[samples.exit_index.le(t)]
        ordering = mature[["cycle_id", "exit_index"]].drop_duplicates().sort_values(["exit_index", "cycle_id"])
        ids = ordering.tail(20).cycle_id.tolist()
        selected = mature[mature.cycle_id.isin(ids)]
        require(ids == clock["training_cycles"] and len(selected) == clock["training_rows"], "旧成熟成员不一致")
        complete = bool(np.isfinite(selected[FEATURES].to_numpy(float)).all())
        support = len(ids) >= 10 and len(selected) >= 100
        require(support == (clock["status"] == "FIT_COMPLETE"), "原支持月份不一致")
        rows.append({"fit_index": t, "fit_origin": clock["fit_origin"], "mature_cycles": len(ids), "mature_rows": len(selected),
            "eight_inputs_complete": complete, "original_month_supported": support,
            "five_distinct_cycle_analogues_available": support and complete and len(ids) >= 5,
            "status": "INPUT_SUPPORT_AVAILABLE" if support and complete else "NO_VIEW_ORIGINAL_MONTH_SUPPORT"})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "逐月输入与完整周期支持.csv", index=False, encoding="utf-8-sig")
    receipt = {"completed_at": now(), "status": "INPUTS_AND_DISTINCT_MATURE_CYCLE_SUPPORT_COMPLETE_NO_REWARD_OR_ACCOUNT",
        "source_samples": len(samples), "source_cycles": int(samples.cycle_id.nunique()), "monthly_clocks": len(rows),
        "supported_months": sum(r["five_distinct_cycle_analogues_available"] for r in rows),
        "preserved_original_unsupported_months": sum(not r["original_month_supported"] for r in rows),
        "targets_read": False, "new_rewards": 0, "new_value_models": 0, "new_accounts": 0,
        "source_sha256": digest(source), "clock_sha256": digest(clocks_path), "preflight_source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
