"""J01可选原字段、时钟、完整成员、精确回退及冻结原模型保护。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.point_j01_optional_correction_v1 import align
from research.point_j01_local_yield_intake_v1 import build_field, FIELDS
from research.point_optional_residual_model_v1 import fit, predict, design
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.learned_cycle_exit_v1 import FEATURES


def inputs():
    days = pd.bdate_range("2018-01-01", periods=45)
    stock = pd.DataFrame({"date": days, "symbol": "510300.SH", "mom20": .1})
    rates = pd.DataFrame({"date": days, "cgb_10y": 3.+np.arange(45)*.01,
        "source": "chinabond.via_akshare.bond_china_yield",
        "retrieved_at": pd.Timestamp("2026-09-01", tz="Asia/Shanghai")})
    return stock, rates


def training():
    rows = []
    for cycle in range(1, 4):
        for i in range(6):
            row = {"cycle_id": cycle, "origin_index": cycle*100+i, "target": .02*i+.1*cycle,
                "sample_weight": 1./6, "auxiliary_available": i != 0,
                FIELDS[0]: float(i+cycle) if i else np.nan,
                FIELDS[1]: float(i*i+cycle) if i else np.nan}
            row.update({field: .001*i for field in FEATURES})
            rows.append(row)
    core = {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES,
        "mean": [0.]*8, "scale": [1.]*8,
        "coefficients": [.01]*8, "intercept": .02, "feature_clip": 5.}
    return pd.DataFrame(rows), core


def test_same_fixed_dates_units_interaction_and_no_same_day_final_yield():
    stock, rates = inputs()
    rates.loc[21, "cgb_10y"] = 999.
    field = build_field(stock, rates)
    samples = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [20, 21], "origin": stock.date.iloc[[20, 21]].to_numpy()})
    result = align(samples, field)
    assert result.auxiliary_available.tolist() == [False, True]
    assert result.loc[1, FIELDS[0]] == pytest.approx(20.)
    assert result.loc[1, FIELDS[1]] == pytest.approx(2.)
    assert not result.first_publication_evidence_proved.any()
    assert result.loc[1, "latest_retrieved_at"] > result.loc[1, "decision_deadline"]


def test_missing_source_tail_never_carries_and_original_rows_remain():
    stock, rates = inputs()
    field = build_field(stock, rates.iloc[:32].copy())
    ids = [32, 33, 34, 44]
    samples = pd.DataFrame({"cycle_id": [1]*4, "origin_index": ids, "origin": stock.date.iloc[ids].to_numpy()})
    result = align(samples, field)
    assert len(result) == 4 and result.auxiliary_available.tolist() == [True, False, False, False]
    assert result.loc[1:, FIELDS].isna().all().all()


def test_future_source_prefix_invariant_and_true_zero_remains_known():
    stock, rates = inputs()
    rates.cgb_10y = 3.
    before = build_field(stock, rates)
    altered = rates.copy()
    altered.loc[30:, "cgb_10y"] = 8.
    after = build_field(stock, altered)
    pd.testing.assert_frame_equal(before.iloc[:31], after.iloc[:31], check_exact=True)
    samples = pd.DataFrame({"cycle_id": [1], "origin_index": [21], "origin": stock.date.iloc[[21]].to_numpy()})
    result = align(samples, before)
    assert result.auxiliary_available.iloc[0]
    np.testing.assert_array_equal(result[FIELDS].to_numpy(float), [[0., 0.]])


def test_original_identity_mismatch_is_rejected():
    stock, rates = inputs()
    daily = build_field(stock, rates)
    samples = pd.DataFrame({"cycle_id": [1], "origin_index": [21], "origin": stock.date.iloc[[22]].to_numpy()})
    with pytest.raises(ValueError, match="日期变化"):
        align(samples, daily)


def test_fixed_core_missing_exact_fallback_and_all_training_members_retained():
    rows, core = training()
    before = copy.deepcopy(core)
    model = fit(rows, core, FIELDS)
    assert core == before and model["global_intercept"] == 0.
    assert model["training_rows"] == 18 and model["unknown_training_rows_retained"] == 3
    base = [.01]*8
    a, b, status = predict(core, model, base, [np.nan, 1.], FIELDS)
    assert a == b == within_cycle_prediction(core, base) and status == "EXACT_CORE_FALLBACK"
    np.testing.assert_allclose(np.average(design(rows, model), axis=0, weights=rows.sample_weight), 0., atol=1e-12)
    assert rows.loc[~rows.auxiliary_available, FIELDS].isna().all().all()


def test_cycle_common_target_shift_does_not_enter_new_coefficients():
    rows, core = training()
    original = fit(rows, core, FIELDS)
    changed = rows.copy()
    changed.target += changed.cycle_id.map({1: 2., 2: -1., 3: 3.})
    shifted = fit(changed, core, FIELDS)
    np.testing.assert_allclose(original["coefficients"], shifted["coefficients"], atol=1e-12, rtol=0)


def test_all_unknown_training_and_unknown_origin_do_not_create_new_intercept():
    rows, core = training()
    rows[FIELDS] = np.nan
    rows.auxiliary_available = False
    model = fit(rows, core, FIELDS)
    np.testing.assert_array_equal(model["coefficients"], [0., 0.])
    assert not model["identified"] and model["unknown_training_rows_retained"] == len(rows)
    a, b, status = predict(core, model, [.01]*8, [0., 0.], FIELDS)
    assert a == b and status == "EXACT_CORE_FALLBACK"
