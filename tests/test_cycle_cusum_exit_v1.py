"""核对固定基准、累积报警、缺失状态，以及额外退出后的实际账户行为。"""
import numpy as np
import pandas as pd
import pytest

from research.additional_cycle_exit_account_v1 import simulate_additional_exit
from research.cycle_cusum_controller_v1 import CycleCUSUMController
from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def market_fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=75), "total_log": np.r_[np.tile([-.01, .01], 30), np.zeros(15)]})
    cycle = {"cycle_id": 1, "entry_index": 60}
    return data, cycle


def test_baseline_ends_before_actual_buy_and_ignores_future_and_buy_day_return():
    data, cycle = market_fixture()
    data.loc[60:, "total_log"] = -.5
    controller = CycleCUSUMController(data)
    first = controller(60, cycle, 100., 100.)
    assert first["cusum_baseline_end"] == data.date.iloc[59]
    assert first["cusum_baseline_start"] == data.date.iloc[0]
    assert first["cusum_baseline_mean"] == pytest.approx(0.)
    assert first["cusum_baseline_sigma"] == pytest.approx(np.std(np.tile([-.01, .01], 30), ddof=1))
    assert first["cusum_value"] == 0. and first["cusum_daily_return"] is None and not first["cusum_alarm"]
    second = controller(61, cycle, 100., 100.)
    assert second["cusum_alarm"] and second["cusum_first_alarm_origin"] == data.date.iloc[61]


def test_small_weakness_accumulates_and_good_day_resets_to_zero():
    data, cycle = market_fixture()
    sigma = np.std(data.total_log.iloc[:60], ddof=1)
    data.loc[61:64, "total_log"] = [-sigma, sigma, -3.1 * sigma, -3.1 * sigma]
    controller = CycleCUSUMController(data)
    controller(60, cycle, 100., 100.)
    assert controller(61, cycle, 100., 100.)["cusum_value"] == pytest.approx(.5)
    assert controller(62, cycle, 100., 100.)["cusum_value"] == 0.
    assert not controller(63, cycle, 100., 100.)["cusum_alarm"]
    alarm = controller(64, cycle, 100., 100.)
    assert alarm["cusum_value"] == pytest.approx(5.2) and alarm["cusum_alarm"]


def test_new_cycle_resets_alarm_and_relocks_prior_baseline():
    data, cycle = market_fixture()
    data.loc[61, "total_log"] = -.2
    controller = CycleCUSUMController(data)
    controller(60, cycle, 100., 100.)
    assert controller(61, cycle, 100., 100.)["cusum_alarm"]
    new = controller(65, {"cycle_id": 2, "entry_index": 65}, 100., 100.)
    assert not new["cusum_alarm"] and new["cusum_value"] == 0.
    assert new["cusum_baseline_end"] == data.date.iloc[64]


@pytest.mark.parametrize("kind", ["short", "zero", "missing"])
def test_missing_or_zero_baseline_stays_no_view(kind):
    data, cycle = market_fixture()
    if kind == "short":
        cycle["entry_index"] = 5
    elif kind == "zero":
        data.loc[:59, "total_log"] = 0.
    else:
        data.loc[40, "total_log"] = np.nan
    controller = CycleCUSUMController(data)
    t = cycle["entry_index"]
    for day in [t, t + 1]:
        row = controller(day, cycle, 100., 100.)
        assert row["cusum_status"].startswith("NO_VIEW") and row["cusum_value"] is None and not row["additional_exit_requested"]


def test_missing_holding_day_is_not_skipped_and_existing_alarm_survives():
    data, cycle = market_fixture()
    data.loc[61, "total_log"] = -.2
    data.loc[62, "total_log"] = np.nan
    controller = CycleCUSUMController(data)
    controller(60, cycle, 100., 100.)
    assert controller(61, cycle, 100., 100.)["cusum_alarm"]
    missing = controller(62, cycle, 100., 100.)
    assert missing["cusum_value"] is None and missing["additional_exit_requested"]
    later = controller(63, cycle, 100., 100.)
    assert later["cusum_status"].startswith("NO_VIEW") and later["cusum_value"] is None and later["cusum_alarm"]


def account_fixture():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    rule = {"entry": np.ones(10, int), "exit": {1: np.zeros(10, bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, div, cfg, cost, str(dates[1].date()), rule, spec]


@pytest.mark.parametrize("learned", [False, True])
def test_disabled_additional_hook_exactly_matches_old_account_with_costs_and_dividend(learned):
    args = account_fixture()
    args[0].loc[2:, ["open", "close"]] = 9.8
    args[0].loc[3:, "previous_close"] = 9.8
    args[0].loc[2, "dividend"] = .2
    args[1] = pd.DataFrame([{"record_date": args[0].date.iloc[1], "ex_date": args[0].date.iloc[2],
                            "payment_date": args[0].date.iloc[5], "cash_dividend_per_share": .2}])
    model = {"fit_index": 0, "latest_exit_index": 0, "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": -.02}}
    original = simulate_rearmed_exit(*args, ExitController(args[0], [model]) if learned else None)
    actual = simulate_additional_exit(*args, ExitController(args[0], [model]) if learned else None)
    for a, b in zip(actual, original):
        pd.testing.assert_frame_equal(a, b)


def test_extra_alarm_keeps_correct_reason_and_pending_exit_when_limit_blocks():
    args = account_fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][4] = 0

    def extra(t, cycle, current, peak):
        return {"additional_exit_requested": t == 2, "additional_exit_reason": "标准化累积转弱值达到5，请求全部退出"}

    ledger, decisions, cycles = simulate_additional_exit(*args, extra)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.iloc[3].filled_quantity < 0
    assert cycles.iloc[0].exit_date == args[0].date.iloc[4]
    assert "累积转弱" in cycles.iloc[0].exit_reasons and "负" not in cycles.iloc[0].exit_reasons
    assert ledger[ledger.filled_quantity > 0].date.to_list() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert ledger.accounting_error.abs().max() < 1e-7


def test_additional_exit_requires_its_actual_reason():
    args = account_fixture()
    with pytest.raises(ValueError, match="原因"):
        simulate_additional_exit(*args, lambda *unused: {"additional_exit_requested": True})


def test_real_cusum_callback_exits_next_open_with_positive_old_model():
    data, _ = market_fixture()
    sigma = np.std(data.total_log.iloc[:60], ddof=1)
    data.loc[61:64, "total_log"] = [-sigma, sigma, -3.1 * sigma, -3.1 * sigma]
    data["close"] = 10. * np.exp(data.total_log.cumsum())
    data["open"] = data.close.shift(1).fillna(10.)
    data["previous_close"], data["dividend"] = data.open, 0.
    data["mom5"], data["mom20"], data["sma120"], data["vol20"] = .01, .02, .03, .15
    args = account_fixture()
    args[0], args[4] = data, str(data.date.iloc[60].date())
    args[5] = {"entry": np.ones(len(data), int), "exit": {1: np.zeros(len(data), bool)}}
    model = {"fit_index": 0, "latest_exit_index": 0, "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": .02}}
    controller = CycleCUSUMController(data, ExitController(data, [model]))
    ledger, decisions, cycles = simulate_additional_exit(*args, controller)
    assert cycles.iloc[0].exit_date == data.date.iloc[65]
    assert "累积转弱" in cycles.iloc[0].exit_reasons and "学习条件" not in cycles.iloc[0].exit_reasons
    assert (ledger.filled_quantity > 0).sum() == 1
    assert ledger.accounting_error.abs().max() < 1e-7
