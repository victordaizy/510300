"""只核最近504个成熟日变化的四状态支持，不生成动作收益或拟合账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.joint_state_action_inputs_v1 import market_state_frame, STATE_NAMES
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_joint_state_action_support_20260908"


def main():
    source = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    schedule = ROOT / "reports/research/510300_learned_cycle_exit_v1/saved_models.json"
    data = pd.read_parquet(source)
    frame = market_state_frame(data)
    originals = json.loads(schedule.read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    records, transitions = [], []
    for record in originals:
        t = int(record["fit_index"])
        origins = np.arange(max(0, t-504), t)
        current = frame.market_state.iloc[origins].to_numpy(float)
        following = frame.market_state.iloc[origins+1].to_numpy(float)
        valid = np.isfinite(current) & np.isfinite(following)
        complete = len(origins) == 504 and bool(valid.all())
        counts = [int((current[valid] == s).sum()) for s in range(4)]
        eligible = complete and min(counts) >= 20
        records.append({"拟合收盘": record["fit_origin"], "拟合位置": t, "请求日变化行数": len(origins), "完整成熟日变化行数": int(valid.sum()),
                        "窗口完整": complete, "四状态均至少二十次": eligible, "最低状态出现次数": min(counts),
                        **{STATE_NAMES[s]+"次数": counts[s] for s in range(4)}})
        require(not len(origins) or origins[-1]+1 <= t, "支持统计纳入拟合之后的状态变化")
        if complete:
            for s in range(4):
                for dest in range(4):
                    transitions.append({"拟合收盘": record["fit_origin"], "当前状态": s, "下一日状态": dest,
                                        "发生次数": int(((current == s) & (following == dest)).sum()), "当前状态总次数": counts[s]})
    OUT.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(OUT / "market_states.parquet", index=False)
    pd.DataFrame(records).to_csv(OUT / "逐月四状态成熟支持.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(transitions).to_csv(OUT / "逐月市场状态变化次数.csv", index=False, encoding="utf-8-sig")
    result = {"recorded_at": now(), "status": "FOUR_STATE_MATURE_CALENDAR_SUPPORT_CHECKED_NO_REWARDS_MODELS_OR_ACCOUNTS", "market_rows": len(frame),
              "market_state_unknown_rows": int(frame.market_state.isna().sum()), "monthly_schedule_rows": len(records), "window_mature_transitions": 504,
              "minimum_required_origin_state_count": 20, "eligible_months": sum(r["四状态均至少二十次"] for r in records),
              "incomplete_window_months": sum(not r["窗口完整"] for r in records), "full_window_but_sparse_state_months": sum(r["窗口完整"] and not r["四状态均至少二十次"] for r in records),
              "minimum_observed_state_count": min(r["最低状态出现次数"] for r in records), "first_month": records[0], "last_month": records[-1],
              "new_models_or_accounts": 0, "new_reference_accounts": 0, "new_action_rewards_computed": 0, "strategy_returns_read": False,
              "source_sha256": digest(source), "schedule_sha256": digest(schedule), "implementation_sha256": digest(ROOT / "research/joint_state_action_inputs_v1.py"),
              "goal_achieved": False}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
