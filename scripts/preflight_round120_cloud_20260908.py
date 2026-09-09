"""只核云图因果定义及可用日历，不计算策略账户或收益。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.ichimoku_cloud_inputs_v1 import cloud_frame
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_cloud_definition_preflight_20260908"


def main():
    source = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    data = pd.read_parquet(source)
    frame = cloud_frame(data)
    require(len(frame) == len(data) and frame.date.equals(data.date), "云图改变原完整日历")
    np.testing.assert_allclose(frame.wealth_close.iloc[1:], data.wealth.iloc[1:], atol=1e-12, rtol=0)
    require(frame.entry_signal.iloc[:78].isna().all() and frame.entry_signal.iloc[78:].notna().all(), "当前真实云图的首次有效边界不同")
    counts = []
    for name, start, end in [("主历史", "2020-01-02", "2026-08-14"), ("较早历史", "2015-01-05", "2019-12-31")]:
        first = int(np.flatnonzero(data.date.ge(start))[0])
        last = int(np.flatnonzero(data.date.le(end))[-1])
        decisions = frame.iloc[first-1:last]
        counts.append({"历史": name, "决策收盘行数": len(decisions), "进入信息未知行数": int(decisions.entry_signal.isna().sum()),
                       "退出信息未知行数": int(decisions.exit_signal.isna().sum()), "原三进入条件成立行数": int(decisions.entry_signal.eq(1).sum()),
                       "已知退出成立行数": int(decisions.exit_signal.eq(1).sum()),
                       "进入退出同时成立行数": int((decisions.entry_signal.eq(1) & decisions.exit_signal.eq(1)).sum())})
    OUT.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(OUT / "cloud_factors.parquet", index=False)
    pd.DataFrame(counts).to_csv(OUT / "云图可用日历.csv", index=False, encoding="utf-8-sig")
    result = {"recorded_at": now(), "status": "CAUSAL_CLOUD_SOURCE_CALENDAR_AVAILABLE_NO_NEW_ACCOUNTS", "rows": len(frame),
              "entry_unknown_rows": int(frame.entry_signal.isna().sum()), "first_complete_cloud": str(frame.loc[frame.entry_signal.notna(), "date"].iloc[0].date()),
              "current_cloud_displacement_trading_rows": 26, "chronology": "CURRENT_CLOUD_USES_CALCULATIONS_FROM_T_MINUS_26_ONLY",
              "calendar": counts, "new_models_or_accounts": 0, "new_reference_accounts": 0, "strategy_returns_read": False,
              "source_sha256": digest(source), "implementation_sha256": digest(ROOT / "research/ichimoku_cloud_inputs_v1.py"), "goal_achieved": False}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
