"""验证不会选未来行、午间跨窗或把条件范围及当前日当严格资格。"""

import numpy as np
import pandas as pd
import pytest

from research.pressure_source_admission_v1 import (
    add_previous_day_counts, conditional_prefix_windows, observe_grid, observed_book_features,
)
from research.quote_state_envelope_v1 import clock_ms


def quote(time, cumulative=100):
    row = {"time": time, "price": 40000., "iopv": 0., "cum_volume": cumulative}
    for i in range(1, 11):
        row[f"bid_px{i}"] = 40000 - i * 10
        row[f"ask_px{i}"] = 39990 + i * 10
        row[f"bid_vol{i}"] = 100.
        row[f"ask_vol{i}"] = 100.
    return row


def test_future_trade_in_prefix_does_not_prove_quote_available_at_nominal_time():
    out = conditional_prefix_windows([93000000], [100], [93000210, 93000300], [100, 100]).iloc[0]
    assert out.last_trade_after_nominal
    assert out.conditional_lower_ms == clock_ms([93000210]).item()
    assert out.conditional_clock_compatible
    assert not out.clock_mapping_verified and not out.receipt_time_verified


def test_zero_prefix_and_unknown_prefix_are_distinct():
    out = conditional_prefix_windows([93000000] * 4, [0, 50, np.nan, -100], [93000300], [100])
    assert out.exact_cumulative_prefix.tolist() == [True, False, False, False]
    assert out.last_included_trade_ms.iloc[0] == -1
    assert out.conditional_upper_exclusive_ms.iloc[0] == clock_ms([93000300]).item() + 10


def test_same_centisecond_trade_prefix_does_not_invent_intracentisecond_order():
    out = conditional_prefix_windows([93000000], [100], [93000210, 93000210], [100, 100]).iloc[0]
    assert out.conditional_lower_ms == clock_ms([93000210]).item()
    assert out.conditional_width_ms == 10


@pytest.mark.parametrize("times,volumes", [([93000211], [100]), ([93000210], [100.5]), ([93000210], [-100])])
def test_nonconforming_ticks_or_quantities_cannot_get_conditional_clock(times, volumes):
    out = conditional_prefix_windows([93000000], [100], times, volumes).iloc[0]
    assert not out.conditional_clock_compatible


def test_cumulative_prefix_later_than_floor_second_is_incompatible_without_shift():
    out = conditional_prefix_windows([93000000], [100], [93002000], [100]).iloc[0]
    assert out.exact_cumulative_prefix
    assert not out.conditional_clock_compatible


def test_best_quote_does_not_imply_fixed_band_covered():
    frame = pd.DataFrame([quote(93000000)])
    frame.loc[0, "ask_px10"] = 40030
    for i in range(1, 11):
        frame.loc[0, f"ask_px{i}"] = 40000 + i * 3
    out = observed_book_features(frame).iloc[0]
    assert out.best_two_sided_valid and out.ten_level_book_valid
    assert not out.fixed_10bp_band_visible


def test_endpoint_selection_ignores_later_quote_and_keeps_strict_false():
    quotes = pd.DataFrame([quote(93000000), quote(93101000, 200)])
    witnesses = conditional_prefix_windows(quotes.time, quotes.cum_volume, [92959200, 93101100], [100, 100])
    grid = pd.DataFrame({"minute": [571], "session": ["AM"], "grid_kind": ["M2_CONTINUOUS"], "hhmm": ["09:31"], "decision_ms": [571 * 60000]})
    out = observe_grid(quotes, witnesses, grid).iloc[0]
    assert out.source_quote_row == 0
    assert out.source_time == 93000000
    assert not out.source_nominal_fresh and not out.strict_point_qualified


def test_missing_prior_quote_is_missing_and_has_no_depth_zero_imputation():
    quotes = pd.DataFrame([quote(93001000)])
    witnesses = conditional_prefix_windows(quotes.time, quotes.cum_volume, [93001000], [100])
    grid = pd.DataFrame({"minute": [570], "session": ["AM"], "grid_kind": ["M2_CONTINUOUS"], "hhmm": ["09:30"], "decision_ms": [570 * 60000]})
    out = observe_grid(quotes, witnesses, grid).iloc[0]
    assert out.source_quote_row == -1 and not out.source_nominal_band_observed
    assert pd.isna(out.source_band_bid_depth_cny)


def test_exact_endpoint_second_cannot_be_treated_as_entirely_available_before_it():
    quotes = pd.DataFrame([quote(93000000)])
    witnesses = conditional_prefix_windows(quotes.time, quotes.cum_volume, [92959000], [100])
    grid = pd.DataFrame({"minute": [570], "session": ["AM"], "grid_kind": ["M2_CONTINUOUS"], "hhmm": ["09:30"], "decision_ms": [570 * 60000]})
    out = observe_grid(quotes, witnesses, grid).iloc[0]
    assert out.source_nominal_band_observed
    assert out.conditional_clock_compatible
    assert not out.conditional_range_before_endpoint
    assert not out.source_conditional_band_observed and not out.strict_point_qualified


def baseline_rows():
    rows = []
    for day, regime in [("20260702", "PRE_20260706"), ("20260703", "PRE_20260706"),
                        ("20260706", "POST_20260706"), ("20260707", "POST_20260706")]:
        for minute, sess in [(685, "AM"), (690, "OTHER"), (780, "PM"), (785, "PM")]:
            rows.append({"date": day, "regime": regime, "minute": minute, "hhmm": f"{minute // 60:02d}:{minute % 60:02d}",
                         "session": sess, "grid_kind": "M2_CONTINUOUS", "source_nominal_book_observed": True,
                         "source_nominal_band_observed": True, "source_conditional_band_observed": True,
                         "source_conditional_book_observed": True})
    return pd.DataFrame(rows)


def test_prior_baseline_excludes_current_day_and_regime_predecessor():
    frame = add_previous_day_counts(baseline_rows(), baseline_days=1)
    at = frame[frame.minute.eq(785)].set_index("date")
    assert at.prior_source_m2_pair_days.tolist() == [0, 1, 0, 1]
    assert at.source_60day_m2_candidate.tolist() == [False, True, False, True]
    assert not at.strict_60day_m2_candidate.any()


def test_five_minute_lag_cannot_cross_session_or_use_missing_endpoint():
    frame = add_previous_day_counts(baseline_rows(), baseline_days=1)
    assert not frame[frame.minute.isin([690, 780])].m2_source_field_pair.any()
    assert frame[frame.minute.eq(785)].m2_source_field_pair.all()
