"""V2仅降低已成熟样本数量，不改变任何价格定义、成本或收益证据门槛。"""
from __future__ import annotations

import pandas as pd


def relax_saved_decisions(base: pd.DataFrame, training_events: pd.DataFrame) -> pd.DataFrame:
    """复用冻结的前序统计量，案例6→3、同状态4→2，其余逐项不变。"""
    out = base.copy()
    pool = training_events.set_index("signal_id")
    for ix, row in out.iterrows():
        ids = row.train_signal_ids.split("|") if row.train_signal_ids else []
        complete = bool(ids) and pool.loc[ids, "excess_return"].notna().all()
        if ids:
            assert (pool.loc[ids, "exit_idx"] <= row.idx).all(), "不得读取未来标签。"
            assert (pool.loc[ids, "signal_idx"] >= row.idx - 504).all(), "不得放宽回看窗口。"
        enough = row.n_train >= 3 and complete
        recent = enough and row.normal90_lower > 0 and row.mean_excess > 0
        state_ok = bool(row.STATE_ONLY)
        full = recent and state_ok and row.state_n >= 2 and row.state_mean_net > 0 and row.state_mean_excess > 0
        reason = "ENABLED" if full else "STATE_INELIGIBLE" if not state_ok else "INSUFFICIENT_RECENT_EVENTS" if not enough else "RECENT_EDGE_UNCONFIRMED" if not recent else "INSUFFICIENT_STATE_EVENTS" if row.state_n < 2 else "STATE_EDGE_NONPOSITIVE"
        out.loc[ix, "RECENT_ONLY"] = bool(recent)
        out.loc[ix, "FULL"] = bool(full)
        out.loc[ix, "reason"] = reason
    unchanged = [c for c in base.columns if c not in ("RECENT_ONLY", "FULL", "reason")]
    pd.testing.assert_frame_equal(base[unchanged], out[unchanged])
    assert not (base.RECENT_ONLY & ~out.RECENT_ONLY).any()
    assert not (base.FULL & ~out.FULL).any()
    return out
