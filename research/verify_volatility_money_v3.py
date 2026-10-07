"""复核保存的第三轮结果，并按先登记的日期规则去除未来窗口重叠。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_volatility_money_decomposition_v3"


def run() -> None:
    spec = importlib.util.spec_from_file_location("v3", ROOT / "research/volatility_money_decomposition_v3.py")
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    v3.reuse_frozen_evidence()
    daily = pd.read_csv(STUDY / "results/每日波动精确分解.csv")
    w = pd.read_csv(STUDY / "results/波动分解_固定周度_完整观察.csv")
    m = pd.read_csv(STUDY / "results/波动分解_104个月_完整观察.csv")
    assert len(w) == 445 and len(m) == 104
    components = daily[["v_variance_change_up2", "v_variance_change_down2", "v_variance_change_mean_correction"]].sum(axis=1, min_count=3)
    assert np.allclose(components, daily.v_variance_change_total, atol=1e-12, rtol=0, equal_nan=True)
    rolloff = daily[daily.downside_window_state == v3.ROLLOFF]
    assert (rolloff.v_recent5_down_sum > rolloff.v_previous5_down_sum).all()
    assert (rolloff.v_recent5_down_sum < rolloff.v_exited5_down_sum).all()
    original_label_columns = []
    for current, name in [(w, "固定周度_完整观察.csv"), (m, "104个月_完整观察.csv")]:
        original = pd.read_csv(STUDY / "inputs" / name)
        for col in original:
            if col.startswith(("E0_", "E1_")):
                original_label_columns.append(col)
                if pd.api.types.is_numeric_dtype(original[col]):
                    assert np.allclose(current[col], original[col], atol=1e-12, rtol=0, equal_nan=True)
                else:
                    assert current[col].fillna("缺失").equals(original[col].fillna("缺失"))
    associ = pd.read_csv(STUDY / "results/波动与后续风险收益关联.csv")
    for row in associ.to_dict("records"):
        part = w[w.training_regime == row["regime"]]
        if row["period"] != "全部":
            part = part[part.analysis_period == row["period"]]
        target = row["entry"] + "_20_" + row["outcome"]
        part = part.dropna(subset=[row["feature"], target])
        value = v3.weighted_corr(part[row["feature"]].to_numpy(), part[target].to_numpy(), v3.cycle_weights(part))
        assert abs(value - row["weighted_spearman"]) < 1e-12
    draws = np.load(STUDY / "results/固定时间块抽样.npz")
    for regime in v3.REGIMES:
        indices = draws[regime + "_indices"]
        counts = draws[regime + "_counts"]
        for index, count in zip(indices, counts, strict=True):
            assert np.array_equal(np.bincount(index, minlength=len(count)), count)
            for start in range(0, len(index), 6):
                assert np.all(np.diff(index[start:start + 6]) == 1)
    kept, correlations = [], []
    assert (STUDY / "evidence/nonoverlap_check_registered.json").exists()
    for regime, part in w.groupby("training_regime"):
        last_exit = "0000-00-00"
        for row in part.sort_values("observation_date").to_dict("records"):
            if pd.isna(row["E0_20_return"]):
                continue
            if row["E0_20_entry_date"] > last_exit:
                kept.append(row)
                last_exit = row["E0_20_exit_date"]
    chosen = pd.DataFrame(kept)
    v3.csv(chosen, "不重叠20日窗口_全部保留点.csv")
    for regime, part in chosen.groupby("training_regime"):
        assert (part.E0_20_entry_date.iloc[1:].reset_index(drop=True) > part.E0_20_exit_date.iloc[:-1].reset_index(drop=True)).all()
        for period, sample in v3.periods(part):
            for target in ["future_rv", "future_downside", "return", "worst_path"]:
                value = float(spearmanr(np.round(sample.v_rv20, 12), np.round(sample["E0_20_" + target], 12)).statistic)
                correlations.append({"regime": regime, "period": period, "n_nonoverlapping_windows": len(sample), "macro_cycles": sample.stat_month.nunique(), "outcome": target, "spearman": value, "role": "OVERLAP_SENSITIVITY_NOT_INDEPENDENT_NEW_SAMPLE"})
    result = pd.DataFrame(correlations)
    v3.csv(result, "不重叠20日窗口_关联复核.csv")
    v3.jsave("verification.json", {"status": "PASS_SAVED_RESULTS_AND_FIXED_NONOVERLAP_CHECK", "daily_rows": len(daily), "weekly_rows": len(w), "monthly_rows": len(m), "original_label_fields_checked_per_panel": len(set(original_label_columns)), "weighted_association_rows_recomputed": len(associ), "block_draws_verified": 4000, "nonoverlapping_windows": len(chosen), "old_forecasts_reused_without_refit": True, "active_goal_achieved": False, "verification_scope": "恒等式、状态定义、旧标签、关联复算、抽样索引及固定日期去重；不能证明预测因果或未来盈利。"})
    print(result.to_string(index=False))


if __name__ == "__main__":
    run()
