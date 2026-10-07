"""验证连续形态的时序、股息现金约束、T+1与停用恢复边界。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.sequential_patterns_regime_v1 import (
    FAMILIES, account, decisions, detect, features, trade_outcome,
)


def market(n=330):
    dates = pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "open": 10.0, "high": 10.1,
                         "low": 9.9, "close": 10.0, "volume": 1000000.0})


def shaped(n=330):
    d = market(n)
    for col, val in {"ao": 10.0, "ah": 10.1, "al": 9.9, "ac": 10.0,
                     "atr20": .2, "high10": 10.1, "low10": 9.9, "prior_low20": 9.0,
                     "shock_z": 0.0, "compression": False, "state": "RANGE",
                     "high_vol": False, "trend_z": 0.0, "volume_ratio": 1.0,
                     "downside_per_volume": 0.0, "dividend": 0.0}.items():
        d[col] = val
    return d


def empty_dividends():
    return pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])


def fixed_decisions(d):
    return pd.DataFrame([{"idx": i, "family": fam, "PATTERN_ONLY": True, "STATE_ONLY": True,
                          "RECENT_ONLY": True, "FULL": True} for i in range(252, len(d)) for fam in FAMILIES])


def test_dividend_neutral_features_and_future_append_invariance():
    prices = market()
    prices.loc[280:, ["open", "high", "low", "close"]] -= .3
    div = pd.DataFrame([{"record_date": prices.date.iloc[279], "ex_date": prices.date.iloc[280],
                         "payment_date": prices.date.iloc[283], "cash_dividend_per_share": .3}])
    full = features(prices, div)
    prefix = features(prices.iloc[:275], div)
    assert_frame_equal(full.iloc[:275].reset_index(drop=True), prefix)
    assert abs(full.ac.iloc[280] - full.ac.iloc[279]) < 1e-10
    assert abs(full.r.iloc[280]) < 1e-10


def test_expired_cases_survive_and_no_same_day_confirmation():
    d = shaped()
    d.loc[252, "compression"] = True
    d.loc[252, "ac"] = 12.0
    ep, steps, sig = detect(d)
    assert len(sig) == 0
    first = ep[ep.family == "BREAKOUT"].iloc[0]
    assert first.status == "EXPIRED_WITHOUT_CONFIRMATION"
    assert first.terminal_idx == 257
    assert len(steps[steps.episode_id == first.episode_id]) == 6


def test_breakout_cannot_claim_intrabar_low_probe_came_first():
    d = shaped()
    d.loc[252, "compression"] = True
    d.loc[253, ["ac", "al", "ah"]] = [10.5, 9.8, 10.6]
    _, _, sig = detect(d)
    s = sig[sig.family == "BREAKOUT"].iloc[0]
    assert s.path_class != "PRIOR_DAY_LOW_PROBED_BEFORE_BREAKOUT"
    assert s.signal_idx == 253


def test_detect_prefix_invariance_and_failed_reclaim_record():
    d = shaped()
    d.loc[252, ["ac", "al"]] = [8.8, 8.7]
    d.loc[253, ["ac", "al"]] = [8.3, 8.2]
    a, astep, asig = detect(d.iloc[:280])
    b, bstep, bsig = detect(d)
    assert_frame_equal(astep, bstep[bstep.idx < 280].reset_index(drop=True))
    first = a[a.family == "RECLAIM"].iloc[0]
    assert first.status == "FAILED_CONTINUED_DOWN"


def test_t_plus_one_and_limit_down_delay():
    d = shaped()
    sig = {"signal_id": "TEST", "family": "BREAKOUT", "state": "RANGE", "signal_idx": 252,
           "signal_date": d.date.iloc[252], "stop_index": 9.95}
    d.loc[253, "ac"] = 9.8
    d.loc[254, "open"] = 9.0
    result = trade_outcome(d, sig, "STRESS")
    assert result["entry_idx"] == 253
    assert result["exit_idx"] == 255
    assert result["exit_reason"] == "INVALIDATED"
    assert result["holding_sessions"] == 2


def test_receivable_is_not_available_cash_and_ledger_matches_label():
    d = shaped(266)
    sig = {"signal_id": "TEST", "family": "BREAKOUT", "state": "RANGE", "signal_idx": 252,
           "signal_date": d.date.iloc[252], "stop_index": 9.5}
    d.loc[254:, ["open", "high", "low", "close"]] -= .2
    d.loc[254, "dividend"] = .2
    div = pd.DataFrame([{"record_date": d.date.iloc[253], "ex_date": d.date.iloc[254],
                         "payment_date": d.date.iloc[257], "cash_dividend_per_share": .2}])
    ledger, trades, _, _ = account(d, div, pd.DataFrame([sig]), fixed_decisions(d), 252, 265, "PATTERN_ONLY", "STRESS")
    l = ledger.set_index("idx")
    assert l.loc[254, "receivable_cny"] > 0
    assert l.loc[254, "cash_cny"] == l.loc[253, "cash_cny"]
    assert l.loc[257, "receivable_cny"] == 0
    assert l.loc[257, "cash_cny"] > l.loc[256, "cash_cny"]
    assert len(trades) == 1
    event = trade_outcome(d, sig, "STRESS")
    capital = event["quantity"] * event["entry_price"]
    fee = max(5, capital * .0004)
    assert abs(event["net_return"] * (capital + fee) - trades.net_pnl.iloc[0]) < 1e-7
    assert abs(ledger.equity_cny.iloc[-1] - 200000 - trades.net_pnl.iloc[0]) < 1e-7


def test_recent_gate_never_reads_future_labels_and_can_observe_while_disabled():
    d = shaped(420)
    labels = pd.DataFrame([{"signal_id": f"B{i}", "family": "RECLAIM", "state": "RANGE",
                            "signal_idx": i, "signal_date": d.date.iloc[i], "entry_idx": i + 1,
                            "exit_idx": i + 6, "cost": "STRESS", "label_status": "MATURE", "net_return": .03}
                           for i in range(275, 405, 10)])
    controls = pd.DataFrame([{"signal_id": f"C{i}", "state": "RANGE", "signal_idx": i,
                              "entry_idx": i + 1, "exit_idx": i + 6, "net_return": .001}
                             for i in range(150, 410)])
    full, _ = decisions(d, pd.DataFrame(), labels, controls)
    truncated, _ = decisions(d.iloc[:360], pd.DataFrame(), labels[labels.exit_idx < 360], controls[controls.exit_idx < 360])
    assert_frame_equal(full[full.idx < 360].reset_index(drop=True), truncated)
    fam = full[full.family == "RECLAIM"]
    assert not fam.iloc[0].FULL
    assert fam.iloc[-1].FULL
    assert (fam.latest_mature_exit_idx.dropna() <= fam.loc[fam.latest_mature_exit_idx.notna(), "idx"]).all()
