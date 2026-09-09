"""只核对两套历史模型的同期可用性，不计算误差或选择器收益。"""
import json
from bisect import bisect_right
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from research.learned_cycle_exit_v1 import FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_exit_forecast_pair_availability_20260908"


def main():
    require(not (OUT / "result.json").exists(), "同期模型可用性核对已完成")
    sample_path = ROOT / "reports/research/510300_learned_cycle_exit_v1/all_reference_samples.parquet"
    old_path = ROOT / "reports/research/510300_learned_cycle_exit_v1/saved_models.json"
    new_path = ROOT / "reports/research/510300_profit_drawdown_interaction_v1/saved_models.json"
    samples = pd.read_parquet(sample_path)
    samples = samples[samples.signal.eq("D60_INTRA")].sort_values(["cycle_id", "origin_index"])
    models = [json.loads(old_path.read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"], json.loads(new_path.read_text(encoding="utf-8"))["models"]]
    indexes = [[m["fit_index"] for m in series] for series in models]
    require(indexes[0] == indexes[1], "两模型原日程不同")
    rows = []
    for row in samples.itertuples():
        available, fit_dates = [], []
        values = np.array([getattr(row, col) for col in FEATURES])
        require(row.origin_index < row.early_exit_index < row.exit_index, "自然持仓状态的评价时钟错误")
        for series, fits in zip(models, indexes):
            k = bisect_right(fits, row.origin_index)-1
            record = series[k] if k >= 0 else None
            flag = record is not None and record["status"] == "FIT_COMPLETE" and np.isfinite(values).all()
            if flag:
                require(record["latest_exit_index"] <= record["fit_index"] <= row.origin_index and row.cycle_id not in record["training_cycles"], "同期模型含当前尚未成熟周期或未来资料")
            available.append(bool(flag))
            fit_dates.append(record["fit_origin"] if record else None)
        rows.append({"cycle_id": row.cycle_id, "origin_index": row.origin_index, "origin": row.origin,
            "exit_index": row.exit_index, "mature_date": row.mature_date, "old_model_available": available[0], "new_model_available": available[1],
            "both_available": all(available), "old_fit_origin": fit_dates[0], "new_fit_origin": fit_dates[1]})
    frame = pd.DataFrame(rows)
    groups = []
    for cycle_id, group in frame.groupby("cycle_id", sort=False):
        require(group.exit_index.nunique() == 1, "同一自然周期成熟日期不唯一")
        groups.append({"cycle_id": cycle_id, "states": len(group), "paired_available_states": int(group.both_available.sum()),
            "all_states_paired": bool(group.both_available.all()), "exit_index": int(group.exit_index.iloc[0]), "mature_date": group.mature_date.iloc[0]})
    cycles = pd.DataFrame(groups)
    schedule = []
    for record in models[0]:
        eligible = cycles[cycles.all_states_paired & cycles.exit_index.le(record["fit_index"])].sort_values(["exit_index", "cycle_id"]).tail(20)
        schedule.append({"fit_index": record["fit_index"], "fit_origin": record["fit_origin"], "complete_paired_cycles": len(eligible), "complete_paired_states": int(eligible.states.sum()),
            "supports_original_minimum": len(eligible) >= 10 and eligible.states.sum() >= 100})
    schedule = pd.DataFrame(schedule)
    ready = schedule[schedule.supports_original_minimum]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / "paired_model_availability.parquet", index=False)
    cycles.to_csv(OUT / "按完整自然周期的同期预测可用性.csv", index=False, encoding="utf-8-sig")
    schedule.to_csv(OUT / "月度完整成熟周期支持.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": now(), "status": "PAIRED_PAST_MODEL_AVAILABILITY_ONLY_NO_ERRORS_OR_ACCOUNT", "source_states": len(frame), "source_cycles": len(cycles),
        "paired_states": int(frame.both_available.sum()), "complete_paired_cycles": int(cycles.all_states_paired.sum()), "complete_paired_cycle_states": int(cycles.loc[cycles.all_states_paired, "states"].sum()),
        "partially_paired_cycles": int((cycles.paired_available_states.gt(0) & ~cycles.all_states_paired).sum()), "model_schedule_origins": len(schedule),
        "origins_supporting_10_cycles_100_states": len(ready), "first_supporting_origin": ready.fit_origin.iloc[0] if len(ready) else None,
        "new_model_fits": 0, "errors_or_selector_returns_computed": False, "new_accounts": 0, "position_impact": 0, "goal_achieved": False,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in [sample_path, old_path, new_path]], "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
