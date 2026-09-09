"""只检查三类现有参考在共同持仓时段内的成熟样本支持，不拟合或算新账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, require, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
OUT = ROOT / "reports/research/510300_shared_reference_episode_support_20260908"
SIGNALS = ["D60_INTRA", "S1_TREND_REBOUND", "R2_Z_CONFIRM"]


def main():
    require(not (OUT / "result.json").exists(), "共同参考时段支持已检查")
    data_path = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    data = pd.read_parquet(data_path)
    samples = pd.read_parquet(P31 / "all_reference_samples.parquet")
    ledgers = [pd.read_parquet(P31 / "reference" / f"{s}_ledger.parquet") for s in SIGNALS]
    dates = pd.DatetimeIndex(ledgers[0].date)
    for ledger in ledgers:
        require(pd.DatetimeIndex(ledger.date).equals(dates), "三参考没有相同完整日历")
    union = np.column_stack([l.shares.gt(0).to_numpy() for l in ledgers]).any(axis=1)
    group_id, active_id, groups, mapped = 0, None, [], []
    for i, (date, holding) in enumerate(zip(dates, union)):
        if holding and active_id is None:
            group_id += 1
            active_id = group_id
            groups.append({"group_id": group_id, "start_date": date, "mature_date": pd.NaT})
        if not holding and active_id is not None:
            groups[-1]["mature_date"] = date
            active_id = None
        mapped.append({"origin": date, "group_id": active_id})
    groups = pd.DataFrame(groups)
    # 统一研究终点强制平仓不能视为自然成熟。
    groups.loc[groups.mature_date.eq(dates[-1]), "mature_date"] = pd.NaT
    assigned = samples.merge(pd.DataFrame(mapped), on="origin", how="left", validate="many_to_one")
    require(assigned.group_id.notna().all(), "原参考状态未归入当日共同持仓时段")
    require(assigned.groupby(["signal", "cycle_id"]).group_id.nunique().eq(1).all(), "一个自然周期跨越共同空仓边界")
    group_sizes = assigned.groupby("group_id").agg(states=("signal", "size"), signal_count=("signal", "nunique"))
    groups = groups.merge(group_sizes, left_on="group_id", right_index=True, how="left")
    history = json.loads((P31 / "saved_models.json").read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    support = []
    for model in history:
        origin = data.date.iloc[model["fit_index"]]
        eligible = groups[groups.mature_date.notna() & groups.mature_date.le(origin) & groups.states.fillna(0).gt(0)].tail(20)
        selected = assigned[assigned.group_id.isin(eligible.group_id)]
        known_signals = set(selected.signal)
        support.append({"fit_index": model["fit_index"], "fit_origin": model["fit_origin"], "complete_nonoverlapping_groups": len(eligible),
                        "states": len(selected), "all_three_tasks_present": known_signals == set(SIGNALS),
                        "supports_10_groups_100_states": len(eligible) >= 10 and len(selected) >= 100 and known_signals == set(SIGNALS)})
    support = pd.DataFrame(support)
    ready = support[support.supports_10_groups_100_states]
    OUT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUT / "三参考共同持仓时段.csv", index=False, encoding="utf-8-sig")
    assigned[["signal", "cycle_id", "origin_index", "origin", "exit_index", "mature_date", "group_id"]].to_parquet(OUT / "original_state_groups.parquet", index=False)
    support.to_csv(OUT / "原月度日程的共同样本支持.csv", index=False, encoding="utf-8-sig")
    paths = [data_path, P31 / "all_reference_samples.parquet", P31 / "saved_models.json"]+[P31 / "reference" / f"{s}_ledger.parquet" for s in SIGNALS]
    result = {"checked_at": now(), "status": "SHARED_REFERENCE_CALENDAR_EPISODE_SUPPORT_ONLY", "original_states": len(samples),
              "natural_cycles_by_signal": samples.groupby("signal").cycle_id.nunique().to_dict(), "calendar_episodes": len(groups),
              "naturally_mature_episodes": int(groups.mature_date.notna().sum()), "supported_month_origins": len(ready),
              "first_supported_origin": ready.fit_origin.iloc[0] if len(ready) else None, "new_models_or_accounts": 0,
              "new_predictions_or_strategy_returns": False, "goal_achieved": False, "position_impact": 0,
              "source_sha256": digest(Path(__file__)), "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
