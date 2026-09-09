"""仅核对每日观察状态的日期、单日开盘标签成熟时点和样本数量。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_observed_return_state_support_20260908"


def main():
    require(not (OUT / "result.json").exists(), "每日观察状态支持已经核对")
    feature_path = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    schedule_path = ROOT / "reports/research/510300_learned_cycle_exit_v1/saved_models.json"
    data = pd.read_parquet(feature_path, columns=["date", "total_simple"])
    require(data.date.is_monotonic_increasing and not data.date.duplicated().any(), "观察状态完整日历必须唯一递增")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    checks = []
    for record in schedule:
        t = int(record["fit_index"])
        # 原点收盘观察状态；下一开盘进入、再下一开盘退出，后者必须已发生。
        end = t-2
        left = max(1, end-242+1)
        rows = data.total_simple.iloc[left:end+1]
        known = np.isfinite(rows.to_numpy(float))
        positive = int((rows >= 0).sum())
        negative = int((rows < 0).sum())
        complete = len(rows) == 242 and known.all()
        checks.append({"fit_index": t, "fit_origin": record["fit_origin"], "first_state_index": left, "last_state_index": end,
                       "last_hypothetical_exit_index": end+2, "state_rows": len(rows), "nonnegative_state_rows": positive,
                       "negative_state_rows": negative, "exact_zero_state_rows": int(rows.eq(0).sum()), "complete_window": bool(complete),
                       "supports_60_rows_each_state": bool(complete and min(positive, negative) >= 60)})
    frame = pd.DataFrame(checks)
    ready = frame[frame.supports_60_rows_each_state]
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / "原月度日程的两观察状态支持.csv", index=False, encoding="utf-8-sig")
    result = {"checked_at": now(), "status": "OBSERVED_RETURN_STATE_COUNTS_AND_MATURITY_ONLY", "original_month_origins": len(checks),
              "supported_origins": len(ready), "first_supported_origin": ready.fit_origin.iloc[0] if len(ready) else None,
              "minimum_negative_count": int(frame.negative_state_rows.min()), "minimum_nonnegative_count": int(frame.nonnegative_state_rows.min()),
              "new_target_values_computed": False, "new_model_fits": 0, "new_accounts": 0, "method_uniqueness": "NOT_CONCLUSIVELY_ESTABLISHED",
              "goal_achieved": False, "position_impact": 0, "source_sha256": digest(Path(__file__)),
              "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [feature_path, schedule_path]]}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
