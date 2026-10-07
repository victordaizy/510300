"""H03合约母集、源钟、分类及固定原模型残差的必要测试。"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.point_h03_optional_correction_v1 import FIELDS, RAW_FIELDS, align, calculate, encode, fit, predict
from research.learned_cycle_exit_v1 import FEATURES
from research.point_optional_residual_model_v1 import design


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2016-01-04", periods=24)
    market = pd.DataFrame({"date": dates, "symbol": "510300.SH"})
    spot = pd.DataFrame({"date": dates, "close": 3000.*np.exp(np.arange(len(dates))*.002), "symbol": "000300.SH"})
    rows = [{"date": day, "symbol": symbol, "close": float(spot.close.iloc[i]*np.exp(.01+i*.001)),
             "open_interest": float(100+i)} for i, day in enumerate(dates) for symbol in ["IF1601", "IF1602", "IF1603", "IF1606"]]
    return market, pd.DataFrame(rows), spot


def states(market: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"cycle_id": 1, "origin_index": np.arange(len(market)), "origin": market.date})


def core() -> dict:
    return {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES, "mean": [0.]*8,
            "scale": [1.]*8, "coefficients": [.01]+[0.]*7, "intercept": .02, "feature_clip": 5.}


def training() -> pd.DataFrame:
    result = []
    for cycle in [1, 2, 3]:
        for k in range(6):
            row = {name: k*.1 for name in FEATURES}
            quadrant, flags = encode(1. if k%2 else -1., 1. if k%3 else -1.)
            row.update(dict(zip(FIELDS[:5], flags)))
            # 周期之间需有不同的已知设计均值，才能检验未知行在周期中心化中的贡献。
            row[FIELDS[5]] = .001*k+.002*cycle
            row.update(cycle_id=cycle, origin_index=cycle*10+k, target=.03*k-.02*cycle,
                       sample_weight=1./6, auxiliary_available=k!=0)
            if k==0:
                for field in FIELDS:
                    row[field] = np.nan
            result.append(row)
    return pd.DataFrame(result)


def test_quadrants_observed_flat_and_unknown_are_distinct():
    names = [encode(a,b)[0] for a,b in [(1.,1.),(1.,-1.),(-1.,1.),(-1.,-1.),(0.,1.),(1.,0.)]]
    assert names == ["OI_UP_PRICE_UP","OI_UP_PRICE_DOWN","OI_DOWN_PRICE_UP","OI_DOWN_PRICE_DOWN","OBSERVED_FLAT","OBSERVED_FLAT"]
    assert sum(encode(1.,1.)[1]) == 1.
    name, flags = encode(np.nan, 1.)
    assert name == "UNKNOWN" and np.isnan(flags).all()


def test_exact_five_day_same_contract_basis_and_following_close_clock():
    market, daily, spot = fixture()
    source = calculate(market,daily,spot)
    field = align(states(market),source)
    assert not field.auxiliary_available.iloc[:6].any()
    assert field.auxiliary_available.iloc[6:].all()
    assert field.source_date.iloc[10] == market.date.iloc[9]
    assert field.iloc[10][RAW_FIELDS[0]] == pytest.approx(np.log(109./104.))
    assert field.iloc[10][RAW_FIELDS[1]] == pytest.approx(.01)
    assert field.iloc[10][RAW_FIELDS[2]] == pytest.approx(.005)
    assert field.available_at_assumed.iloc[10] == pd.Timestamp(market.date.iloc[10]).tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15,minutes=5)


def test_future_quotes_and_new_contract_members_do_not_change_prefix():
    market,daily,spot = fixture()
    baseline = calculate(market,daily,spot)
    changed = daily.copy(deep=True)
    changed.loc[changed.date.ge(market.date.iloc[15]),"open_interest"] += 1000.
    changed.loc[changed.date.ge(market.date.iloc[15]) & changed.symbol.eq("IF1606"),"symbol"] = "IF1609"
    after = calculate(market,changed,spot)
    a,b = align(states(market),baseline),align(states(market),after)
    pd.testing.assert_frame_equal(a.iloc[:16],b.iloc[:16])


def test_incomplete_contract_day_is_not_partial_aggregate():
    market,daily,spot = fixture()
    drop = daily.date.eq(market.date.iloc[11]) & daily.symbol.eq("IF1606")
    source = calculate(market,daily.loc[~drop],spot)
    assert not source.complete_four_contracts.iloc[11]
    assert np.isnan(source.aggregate_OI.iloc[11])
    assert not source.auxiliary_available.iloc[11]
    assert not source.auxiliary_available.iloc[16]


def test_missing_whole_source_day_never_changes_calendar_window():
    market,daily,spot = fixture()
    source = calculate(market,daily.loc[~daily.date.eq(market.date.iloc[11])],spot)
    assert not source.auxiliary_available.iloc[11:17].any()
    assert source.auxiliary_available.iloc[17]
    assert source.window_start_date.iloc[16] == market.date.iloc[11]


def test_expiry_zero_oi_is_retained_and_basis_uses_current_observed_contracts():
    market,daily,spot = fixture()
    modified = daily.copy(deep=True)
    modified.loc[modified.date.eq(market.date.iloc[10]) & modified.symbol.eq("IF1601"),"open_interest"] = 0.
    source = calculate(market,modified,spot)
    assert source.reported_contract_rows.iloc[10] == 4
    assert source.observed_zero_OI_rows.iloc[10] == 1
    assert source.aggregate_OI.iloc[10] == 330.
    assert source.same_price_contract_count.iloc[10] == 3
    assert source.iloc[10][RAW_FIELDS[2]] == pytest.approx(.005)


def test_late_source_and_spot_missing_are_not_filled():
    market,daily,spot = fixture()
    limited = daily.loc[daily.date.le(market.date.iloc[15])]
    field = align(states(market),calculate(market,limited,spot))
    assert field.auxiliary_available.iloc[16]
    assert not field.auxiliary_available.iloc[17:].any()
    cut_spot = spot.loc[~spot.date.eq(market.date.iloc[12])]
    field2 = align(states(market),calculate(market,daily,cut_spot))
    assert np.isnan(field2.iloc[13][RAW_FIELDS[1]])
    assert field2.quadrant.iloc[13] == "UNKNOWN"


def test_price_comparable_boundary_does_not_admit_earlier_basis():
    market,daily,spot = fixture()
    source = calculate(market,daily,spot,str(market.date.iloc[8].date()))
    assert source[RAW_FIELDS[2]].iloc[:13].isna().all()
    assert source[RAW_FIELDS[2]].iloc[13:].notna().all()


def test_original_core_and_raw_missing_are_preserved_with_exact_fallback():
    rows, original = training(),core()
    before, snapshot = rows.copy(deep=True),copy.deepcopy(original)
    model = fit(rows,original)
    a,b,status = predict(original,model,[.1]*8,[np.nan]*6)
    assert a==b and status=="EXACT_CORE_FALLBACK"
    assert original==snapshot
    pd.testing.assert_frame_equal(rows,before)
    np.testing.assert_allclose(np.average(design(rows,model),axis=0,weights=rows.sample_weight),np.zeros(6),atol=1e-14,rtol=0)


def test_cycle_common_shift_is_removed_and_unknown_rows_still_contribute():
    rows=training()
    first=fit(rows,core())
    shift=rows.copy(deep=True)
    shift["target"] += shift.cycle_id*.4
    second=fit(shift,core())
    np.testing.assert_allclose(first["coefficients"],second["coefficients"],atol=1e-14,rtol=0)
    altered=rows.copy(deep=True)
    altered.loc[0,"target"] += .5
    third=fit(altered,core())
    assert not np.allclose(first["coefficients"],third["coefficients"],atol=1e-8,rtol=0)


def test_all_unknown_training_cannot_create_an_auxiliary_forecast():
    rows=training()
    rows[FIELDS]=np.nan
    rows["auxiliary_available"]=False
    model=fit(rows,core())
    a,b,status=predict(core(),model,[.1]*8,[0.,0.,0.,0.,1.,.01])
    assert a==b and status=="EXACT_CORE_FALLBACK"
    assert model["coefficients"] == [0.]*6
