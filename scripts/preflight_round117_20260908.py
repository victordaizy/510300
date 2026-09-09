"""只恢复原参考每份市场路径，尚不训练候选或计算策略收益。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.market_path_state_v1 import reference_market_states, attach_reference_market_states
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/research/510300_market_path_state_preflight_20260908"
    p31 = root / "reports/research/510300_learned_cycle_exit_v1"
    paths = {"features": root / "reports/research/510300_adaptive_allocation_v1/features.parquet", "ledger": p31 / "reference/D60_INTRA_ledger.parquet",
             "decisions": p31 / "reference/D60_INTRA_decisions.parquet", "cycles": p31 / "reference/D60_INTRA_cycles.csv", "samples": p31 / "all_reference_samples.parquet"}
    data, ledger, decisions, cycles = pd.read_parquet(paths["features"]), pd.read_parquet(paths["ledger"]), pd.read_parquet(paths["decisions"]), pd.read_csv(paths["cycles"])
    states = reference_market_states(data, ledger, decisions, cycles)
    raw = pd.read_parquet(paths["samples"]); raw = raw[raw.signal.eq("D60_INTRA")].reset_index(drop=True)
    samples = attach_reference_market_states(raw, states)
    require(np.isfinite(samples[["market_cycle_return", "market_cycle_drawdown"]].to_numpy(float)).all(), "原训练状态无法全部恢复无买入费用路径")
    pd.testing.assert_frame_equal(raw, samples[raw.columns])
    for cycle in cycles.itertuples():
        s = states[states.cycle_id.eq(cycle.cycle_id)]
        if s.empty:
            continue
        origin = float(data.open.iloc[int(cycle.entry_index)])
        unit = data.close.iloc[s.origin_index.to_numpy(int)].to_numpy()+s.unit_confirmed_dividend.to_numpy()
        peak = np.maximum.accumulate(np.r_[origin, unit])[1:]
        np.testing.assert_allclose(s.market_unit_value, unit, atol=1e-12, rtol=0)
        np.testing.assert_allclose(s.market_unit_peak, peak, atol=1e-12, rtol=0)
        np.testing.assert_allclose(s.market_cycle_return, unit/origin-1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(s.market_cycle_drawdown, unit/peak-1, atol=1e-12, rtol=0)
    out.mkdir(parents=True, exist_ok=True)
    states.to_parquet(out / "reference_market_states.parquet", index=False)
    samples.to_parquet(out / "augmented_reference_samples.parquet", index=False)
    record = {"checked_at": now(), "status": "FULL_REFERENCE_MARKET_PATH_RECONSTRUCTED_NO_NEW_FITS_OR_ACCOUNTS", "full_holding_states": len(states),
              "original_training_states": len(raw), "complete_market_training_states": len(samples), "new_models": 0, "new_accounts": 0, "new_reference_accounts": 0,
              "changed_training_return_states": int((samples.market_cycle_return-samples.cycle_return).abs().gt(1e-12).sum()),
              "changed_training_drawdown_states": int((samples.market_cycle_drawdown-samples.cycle_drawdown).abs().gt(1e-12).sum()),
              "boundary": "只改变学习用的两个市场路径状态；真实净成本、自然参考退出与标签、成熟时间、实际账户风险保护保持。完整参考日逐步恢复峰值，不用筛过的训练行恢复。",
              "next_state": "ROUND117_MODEL_TESTS_AND_FREEZE_PENDING", "sources": [{"name": name, "path": str(path.relative_to(root)), "sha256": digest(path)} for name, path in paths.items()]}
    write_json(out / "result.json", record, exclusive=True)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
