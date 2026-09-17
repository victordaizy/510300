"""检验风险到仓位的经济边界，以及不足一手时的真实退出规则。"""
import numpy as np
from research.intraday_overnight_increment_v1 import Account
from research.monthly_downside_forecast_v1 import CONFIG, read
from research.monthly_downside_accounts_v1 import risk_target, risk_request


def test_risk_target_is_capped_nonincreasing_and_unknown_is_preserved():
    cfg = read(CONFIG)
    np.testing.assert_allclose(risk_target(.02, cfg), .5)
    assert risk_target(0, cfg) == 1
    assert risk_target(.0001, cfg) == 1
    assert np.isnan(risk_target(np.nan, cfg)) and np.isnan(risk_target(-1, cfg))
    assert all(np.diff([risk_target(v, cfg) for v in [.001, .005, .02, .2]]) <= 0)


def test_target_below_one_lot_exits_even_inside_band_and_initial_entry_is_not_suppressed():
    cfg = read(CONFIG)
    held = Account(cash=199600, shares=100)
    result = risk_request(held, 4, .0001, cfg)
    assert result['requested_quantity'] == -100 and result['reference_weight'] == 0
    cash = Account(cash=200000)
    assert risk_request(cash, 4, .05, cfg)['requested_quantity'] == 2500
    held = Account(cash=100000, shares=25000)
    assert risk_request(held, 4, .55, cfg)['requested_quantity'] == 0
