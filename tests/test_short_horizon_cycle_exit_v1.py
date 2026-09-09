"""短期标签检验期限、分红权益及原周期成熟约束。"""
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import FEATURES, continuation_label, training_rows
from research.short_horizon_cycle_exit_v1 import relabel_samples


def setup():
    dates = pd.bdate_range("2020-01-01", periods=14)
    data = pd.DataFrame({"date": dates, "open": np.arange(10., 24.)})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    rows = []
    for cycle, origin, end in [(1, 0, 10), (2, 1, 4)]:
        rows.append({"cycle_id": cycle, "origin_index": origin, "early_exit_index": origin + 1, "exit_index": end,
                     "mature_date": dates[end], "reference_quantity": 100, "target": .25, "extra_dividend_cny": 0.,
                     **{key: .01 for key in FEATURES}})
    config = {"label_max_intervals": 5, "costs": {"BASE": {"commission": 0., "minimum": 0., "slippage": 0.}}, "tick": .001, "recent_cycles": 20}
    return data, dividends, pd.DataFrame(rows), config


def test_label_ends_after_five_open_intervals_or_earlier_natural_exit():
    data, dividends, samples, config = setup()
    result = relabel_samples(data, dividends, samples, config)
    assert result.label_exit_index.to_list() == [6, 4]
    assert result.exit_index.to_list() == [10, 4]
    assert np.isclose(result.target.iloc[0], (16 - 11) / 11)
    assert np.isclose(result.target.iloc[1], (14 - 12) / 12)


def test_dividend_uses_record_day_and_excludes_late_and_already_owned_rights():
    data, dividends, samples, config = setup()
    dates = data.date
    dividends = pd.DataFrame([{"record_date": dates[day], "ex_date": dates[day + 1], "payment_date": dates[12], "cash_dividend_per_share": .4}
                              for day in [0, 1, 5, 6, 9]])
    result = relabel_samples(data, dividends, samples, config)
    assert result.extra_dividend_cny.iloc[0] == 80.
    assert np.isclose(result.target.iloc[0], (500 + 80) / 1100)


def test_mature_short_label_does_not_admit_an_unfinished_reference_cycle():
    data, dividends, samples, config = setup()
    result = relabel_samples(data, dividends, samples, config)
    chosen, ids = training_rows(result, 6, config)
    assert ids == [2] and chosen.cycle_id.to_list() == [2]
    assert chosen.sample_weight.sum() == 1.
    chosen, ids = training_rows(result, 10, config)
    assert set(ids) == {1, 2} and chosen.sample_weight.sum() == 2.


def test_prices_after_the_short_label_exit_do_not_change_the_target():
    data, dividends, samples, config = setup()
    original = relabel_samples(data, dividends, samples, config)
    data.loc[7:, "open"] = 99999.
    changed = relabel_samples(data, dividends, samples, config)
    np.testing.assert_array_equal(original.target, changed.target)


def test_original_short_cycle_target_is_unchanged_including_existing_costs():
    data, dividends, samples, config = setup()
    config["costs"]["BASE"] = {"commission": .0002, "minimum": 5., "slippage": .0005}
    expected, extra = continuation_label(data, dividends, 100, 2, 4, config["costs"]["BASE"], config["tick"])
    result = relabel_samples(data, dividends, samples, config)
    assert result.target.iloc[1] == expected and result.extra_dividend_cny.iloc[1] == extra
