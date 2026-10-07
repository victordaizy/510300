"""选择诊断的可手算边界：稳定优势、反转、并列、缺日和重复身份。"""
import numpy as np
import pandas as pd
import pytest

from research.point_comparable_selection_v1 import align_returns, average_ranks, cscv, unique_candidates


def panel(a, b):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(a)), "A": a, "B": b})


def test_stable_advantage_has_no_rank_reversal():
    noise = np.tile([-.001, .001, -.001, .001], 2)
    summary, rows, blocks, selected = cscv(panel(noise + .003, noise - .003), blocks=2)
    assert summary["pbo_le_zero"] == 0
    assert summary["selected_oos_negative_fraction"] == 0
    assert len(blocks) == 2 and len(rows) == 2
    assert selected.winner.tolist() == ["A"]


def test_exact_half_sample_reversal_has_pbo_one():
    noise = np.tile([-.001, .001, -.001, .001], 2)
    shift = np.r_[np.repeat(.003, 4), np.repeat(-.003, 4)]
    summary, rows, _, _ = cscv(panel(noise + shift, noise - shift), blocks=2)
    assert summary["pbo_le_zero"] == 1
    assert summary["strictly_below_median_fraction"] == 1
    assert summary["selected_oos_negative_fraction"] == 1
    np.testing.assert_allclose(rows.oos_logit, -np.log(2), atol=1e-14)


def test_ties_have_fractional_selection_and_explicit_median_mass():
    noise = np.tile([-.001, .001, -.001, .001], 2)
    summary, rows, _, selected = cscv(panel(noise, noise), blocks=2)
    assert summary["pbo_le_zero"] == 1
    assert summary["strictly_below_median_fraction"] == 0
    assert summary["median_mass_fraction"] == 1
    np.testing.assert_allclose(rows.groupby("split").selection_weight.sum(), 1)
    np.testing.assert_allclose(selected.selection_fraction, .5)
    np.testing.assert_array_equal(average_ranks([1, 1, 2]), [1.5, 1.5, 3])


def test_missing_calendar_is_rejected_without_fill():
    a = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "net_return": [0., .001, -.001]})
    with pytest.raises(ValueError, match="日历不一致"):
        align_returns({"A": a, "B": a.drop(index=1)})
    broken = a.copy()
    broken.loc[1, "net_return"] = np.nan
    with pytest.raises(ValueError, match="未知"):
        align_returns({"A": broken})


def test_duplicate_requires_identity_under_both_costs():
    basic = panel([.001, -.001, .002, 0.], [.001, -.001, .002, 0.])
    stress = basic.copy()
    stress.loc[0, "B"] -= .0001
    for p in [basic, stress]:
        p["CASH"] = 0.
        p["A_COPY"] = p.A
    kept, records = unique_candidates({"BASE": basic, "STRESS": stress})
    assert kept == ["A", "B"]
    statuses = {r["candidate"]: r for r in records}
    assert statuses["CASH"]["status"] == "INACTIVE_RETAINED_RAW_EXCLUDED_RANKING"
    assert statuses["A_COPY"]["same_as"] == "A"


def test_uneven_blocks_preserve_every_calendar_row():
    noise = np.tile([-.001, .001], 5)
    summary, rows, blocks, _ = cscv(panel(noise + .003, noise - .003), blocks=4)
    assert blocks.rows.sum() == 10
    assert summary["combinations"] == 6
    assert (rows.is_rows + rows.oos_rows).eq(10).all()
