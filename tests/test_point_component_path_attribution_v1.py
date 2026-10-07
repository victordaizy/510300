"""验证持有区间、事前状态及分红收益拆解。"""
import numpy as np
import pandas as pd
import pytest

from research import point_component_path_attribution_v1 as study


def test_open_boundary_and_censored_position():
    points = pd.DataFrame([
        {"entry_date": pd.Timestamp("2020-01-02"), "exit_date": pd.Timestamp("2020-01-06"), "status": "COMPLETE", "last_observation_date": pd.NaT},
        {"entry_date": pd.Timestamp("2020-01-07"), "exit_date": pd.NaT, "status": "RIGHT_CENSORED", "last_observation_date": pd.Timestamp("2020-01-10")},
    ])
    assert study.held_at(points, pd.Timestamp("2020-01-02")) is not None
    assert study.held_at(points, pd.Timestamp("2020-01-06")) is None
    assert study.held_at(points, pd.Timestamp("2020-01-10")) is not None
    assert study.held_at(points, pd.Timestamp("2020-01-13")) is None


def test_price_dividend_and_state_components_telescope():
    dates = pd.bdate_range("2020-01-02", periods=5)
    prices = pd.DataFrame({"date": dates, "open": [10., 11., 10., 12., 13.]})
    dividends = pd.DataFrame({"record_date": [dates[0], dates[2]], "ex_date": [dates[1], dates[3]], "cash_dividend_per_share": [.2, .4]})
    signals = pd.DataFrame({"execution_date": dates, "origin": dates-pd.Timedelta(days=1),
                            "holding_cause": ["CORE_ONLY", "AUXILIARY_ONLY", "CORE_AND_AUXILIARY", "CORE_ONLY", "CORE_ONLY"],
                            "component_flags": ["VINTAGE", "AUXILIARY", "VINTAGE+AUXILIARY", "VINTAGE", "VINTAGE"],
                            "core_target": [.2, 0., .2, .2, .2], "effective_auxiliary": [0., .3, .3, 0., 0.],
                            "vintage_contribution": [.2, 0., .2, .2, .2], "router_contribution": 0.}).set_index("execution_date")
    reference = study.old.point_reference(11., 12., .4)
    point = {"point_id": "测试", "model": "模型", "period": "时期", "entry_idx": 1, "exit_idx": 3, "entry_date": dates[1],
             "exit_date": dates[3], "entry_raw": 11., "entry_cause": "AUXILIARY_ONLY", "status": "COMPLETE", "point_net_return": reference["point_net_return"]}
    path, total = study.interval_path(point, prices, dividends, signals)
    assert len(path) == 2
    assert path.dividend_per_share_increment.sum() == pytest.approx(.4)
    assert total["gross_return"] == pytest.approx(1.4/11)
    assert total["AUXILIARY_ONLY_gross_return"] + total["CORE_AND_AUXILIARY_gross_return"] == pytest.approx(total["gross_return"])
    assert total["recomputed_net_return"] == pytest.approx(reference["point_net_return"])
    assert total["later_core_after_auxiliary_entry"]
    prices.loc[4, "open"] = 900.
    changed, again = study.interval_path(point, prices, dividends, signals)
    pd.testing.assert_frame_equal(path, changed)
    assert again["recomputed_net_return"] == total["recomputed_net_return"]


def test_unknown_zero_and_future_signal_rejection():
    assert study.current_cause(.2, np.nan, np.nan) == "UNKNOWN_HOLD"
    assert study.current_cause(0., 0., 0.) == "ZERO_WAIT"
    assert study.current_cause(0., .1, .1) == "AUXILIARY_ONLY"
    source = pd.DataFrame({"origin": [pd.Timestamp("2020-01-02")], "execution_date": [pd.Timestamp("2020-01-02")],
                           "budget131": [.5], "core_target": [.2], "effective_auxiliary": [0.], "target": [.2]})
    parents = source[["origin", "execution_date"]].copy()
    parents["VINTAGE_REFERENCE_RISK_parent_target"] = .2
    parents["MODEL_SUPPORT_REFERENCE_ROUTER_parent_target"] = .2
    with pytest.raises(ValueError, match="执行日"):
        study.add_components(source, parents)
