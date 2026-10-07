"""前瞻账户研究的时钟、独立起点、不可提前验收及旧记录不变检查。"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import point_weight_forward_inputs_v1 as logic
from research.point_account_nr7_complement_v1 import verify_account


def market(n=12):
    dates = pd.bdate_range("2026-10-08", periods=n)
    price = 10 + .03 * np.arange(n)
    data = pd.DataFrame({"date": dates, "open": price, "close": price + .01, "dividend": 0.})
    data["ao"], data["ac"] = data.open, data.close
    target = np.resize([.1, .4, .4, 0., 0., .2, .4, 0., 0.], n)
    parents = pd.DataFrame({"origin": dates, logic.binary.PARENT_A: target})
    risks = pd.DataFrame({"idx": np.arange(n), "es95": .01})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return data, div, parents, risks


def protocol(n=1008):
    return {"frozen_at": "2026-10-01T23:00:00+08:00", "design_last_close": "2026-09-30",
            "terminal_sessions": n, "minimum_candidate_cycles": 30, "minimum_wins_and_losses": 5,
            "paired_blocks": [20, 252], "paired_replications": 2000, "trade_replications": 5000}


def registered(origin, execution):
    return {"origin": origin, "execution_date": execution,
            "generated_at": origin + "T16:00:00+08:00", "snapshot_verified": True,
            "same_protocol": True, "matches_current_causal_prefix": True}


def test_no_inherited_history_or_fake_cash_days_before_first_execution():
    d, div, p, r = market()
    assert logic.accounts(d.iloc[:1], div, p.iloc[:1], r.iloc[:1], d.date.iloc[1]) == {}
    result = logic.accounts(d, div, p, r, d.date.iloc[5])
    assert len(result) == 4
    for account in result.values():
        assert account["daily"].date.iloc[0] == d.date.iloc[5]
        first = account["daily"].iloc[0]
        assert first.equity == pytest.approx(200000 * (1 + first.net_return))
        assert account["trades"].empty or account["trades"].entry_date.ge(d.date.iloc[5]).all()
        verify_account(account)


@pytest.mark.parametrize("generated,expected", [
    ("2026-10-08T15:04:59+08:00", "LATE_OR_BEFORE_COMPLETED_CLOSE"),
    ("2026-10-08T15:05:00+08:00", "TIMELY_FROZEN_ACCOUNT_INPUTS"),
    ("2026-10-09T09:30:00+08:00", "LATE_OR_BEFORE_COMPLETED_CLOSE"),
])
def test_actual_registration_boundary(generated, expected):
    p = protocol()
    assert logic.registration_status("2026-10-08", "2026-10-09", generated, p["frozen_at"], p["design_last_close"]) == expected


def test_old_history_or_naive_clock_cannot_become_independent():
    p = protocol()
    result = logic.registration_status("2026-09-30", "2026-10-08", "2026-10-01T23:30:00+08:00", p["frozen_at"], p["design_last_close"])
    assert result == "INVALID_INDEPENDENCE_BOUNDARY"
    with pytest.raises(ValueError, match="时区"):
        logic.registration_status("2026-10-08", "2026-10-09", "2026-10-08T16:00:00", p["frozen_at"], p["design_last_close"])


def test_missing_flat_day_and_changed_snapshot_invalidate_whole_window():
    dates = pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"])
    records = [registered("2026-10-08", "2026-10-09"), registered("2026-10-12", "2026-10-13")]
    coverage = logic.account_registration_coverage(dates, "2026-10-09", "2026-10-13", records, protocol())
    assert len(coverage) == 3 and coverage.timely.sum() == 2
    assert logic.phase_status(3, coverage, protocol()) == "INELIGIBLE_INCOMPLETE_REGISTRATION"
    records.insert(1, registered("2026-10-09", "2026-10-12"))
    records[1]["matches_current_causal_prefix"] = False
    coverage = logic.account_registration_coverage(dates, "2026-10-09", "2026-10-13", records, protocol())
    assert not coverage.timely.all()


def test_good_partial_performance_cannot_trigger_early_acceptance():
    coverage = pd.DataFrame({"timely": [True] * 32})
    assert logic.phase_status(32, coverage, protocol()) == "ACCUMULATING_NO_EARLY_ACCEPTANCE"
    assert logic.phase_status(0, coverage.iloc[:0], protocol()) == "WAITING_FOR_FIRST_NEW_CLOSE"
    assert logic.phase_status(1008, coverage, protocol(), terminal_exists=True) == "TERMINAL_RESULT_ALREADY_FROZEN"
    with pytest.raises(ValueError, match="终点"):
        logic.phase_status(1009, pd.DataFrame({"timely": [True] * 1009}), protocol())


@pytest.mark.parametrize("length", [9, 12])
def test_append_market_keeps_funded_prefix_and_mature_trades_unchanged(length):
    d, div, p, r = market()
    old = logic.accounts(d.iloc[:8], div, p.iloc[:8], r.iloc[:8], d.date.iloc[1])
    new = logic.accounts(d.iloc[:length], div, p.iloc[:length], r.iloc[:length], d.date.iloc[1])
    for key in old:
        logic.prefix_check(old[key], new[key])
        changed = copy.deepcopy(new[key])
        changed["daily"].loc[0, "equity"] += 1
        with pytest.raises(AssertionError):
            logic.prefix_check(old[key], changed)


def test_initial_partial_year_is_never_counted_as_full_year():
    d, div, p, r = market()
    result = logic.accounts(d, div, p, r, d.date.iloc[1])[("STRESS", "SAVED_WEIGHT")]
    official = pd.bdate_range("2026-01-01", "2026-12-31")
    metrics = logic.measured_metrics(result, official)
    assert metrics["complete_official_years"] == 0
    assert np.isnan(metrics["average_full_year_cycles"])


def test_no_trade_window_cannot_pass_or_be_called_zero_risk_success():
    d, div, p, r = market()
    p[logic.binary.PARENT_A] = 0.
    accounts = logic.accounts(d, div, p, r, d.date.iloc[1])
    coverage = pd.DataFrame({"timely": [True] * (len(d) - 1)})
    result = logic.evaluate_terminal(accounts, coverage, True, protocol(len(d) - 1), pd.DatetimeIndex(d.date))
    assert result["status"] == "INCONCLUSIVE_INFORMATION_FLOOR_NOT_MET"
    assert not result["fixed_window_supported"]
    assert not result["goal_achieved"]


def test_synthetic_end_to_end_registration_and_first_account_day(tmp_path, monkeypatch):
    """全部时钟及行情属于临时测试夹具，绝不写入真实研究登记目录。"""
    from research import point_weight_forward_v1 as driver

    actual_root = driver.ROOT
    source = tmp_path / "source"
    (source / "inputs/config").mkdir(parents=True)
    (source / "results").mkdir()
    dates = pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09", "2026-10-12"])
    prices = pd.DataFrame({"date": dates, "symbol": "510300.SH", "open": 4.1 + .01 * np.arange(len(dates)),
                           "high": 4.3, "low": 3.9, "close": 4.11 + .01 * np.arange(len(dates)), "volume": 100000.})
    control = {"length": 2, "now": "2026-10-01T23:00:00+08:00"}

    def save_source():
        n = control["length"]
        prices.iloc[:n].to_parquet(source / "inputs/candidate_prices.parquet", index=False)
        signals = pd.DataFrame({"candidate": logic.binary.PARENT_A, "origin": dates[:n],
                                "execution_date": dates[1:n + 1], "target": .2})
        signals.to_parquet(source / "results/完整候选意向.parquet", index=False)
        registered = signals.rename(columns={"execution_date": "planned_execution_date"}).copy()
        registered["source_data_max_date"] = registered.origin
        registered["actual_generated_at"] = [(x.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)).isoformat()
                                              for x in registered.origin]
        registered["signal_status"] = "AVAILABLE"
        registered.to_parquet(source / "results/完整实际登记.parquet", index=False)

    save_source()
    for kind in ("ordinary", "within"):
        (source / f"inputs/{kind}_models.json").write_text(json.dumps({"models": []}), encoding="utf-8")
    pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"]).to_csv(
        source / "inputs/dividends.csv", index=False)
    module = Path("research/point_weight_forward_inputs_v1.py")
    (tmp_path / module).parent.mkdir()
    (tmp_path / module).write_bytes((actual_root / module).read_bytes())
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "OUT", tmp_path / "study")
    monkeypatch.setattr(driver, "calendar_dates", lambda: pd.DatetimeIndex(dates))
    monkeypatch.setattr(driver, "local_imports", lambda entries: [module.as_posix()])
    monkeypatch.setattr(driver.common, "now", lambda: control["now"])
    monkeypatch.setattr(driver, "source_state", lambda: ({"last_known_close": str(dates[control["length"] - 1].date()),
                                                         "version": "source"}, source))
    driver.initialize()
    first = driver.inspect_or_record()
    assert first["observed_account_sessions"] == first["registered_origins"] == 0
    control.update(length=3, now="2026-10-08T16:00:00+08:00")
    save_source()
    second = driver.inspect_or_record(record=True)
    assert second["registered_origins"] == 1 and second["new_account_evaluations"] == 0
    receipt = driver.OUT / "registrations/2026-10-08/receipt.json"
    before = receipt.read_bytes()
    control["now"] = "2026-10-08T16:01:00+08:00"
    driver.inspect_or_record(record=True)
    assert receipt.read_bytes() == before
    control.update(length=4, now="2026-10-09T16:00:00+08:00")
    save_source()
    third = driver.inspect_or_record(record=True)
    assert third["observed_account_sessions"] == 1 and third["new_account_evaluations"] == 4
    assert third["source_calendar_coverage"]["full_calendar_coverage"]
    assert third["status"] == "ACCUMULATING_NO_EARLY_ACCEPTANCE"
    assert not third["goal_achieved"]
