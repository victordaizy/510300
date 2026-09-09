"""检查下行凸目标的真实抵消、平坦解、时点和实际进出。"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from research.joint_downside_reference_pair_inputs_v1 import minimum_joint_downside, joint_downside_reference_frames, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"risk_window": 242, "downside_benchmark": 0., "annual_days": 242, "weight_band": .1, "initial_capital": 200000., "lot": 100,
    "tick": .001, "limit_fraction": .1,
    "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}, "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}}


def fixture(n=540):
    data = pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=n), "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .0004})
    patterns = [np.array([-.02, -.01, .01, .02]), np.array([-.01, .02, -.02, .01])*.4]
    ledgers = {model: pd.DataFrame({"date": data.date.iloc[2:].to_numpy(), "net_return": patterns[j][np.arange(n-2) % 4]+.001,
        "mark_clock": ["CLOSE"]*(n-3)+["OPEN_TERMINAL"], "source_cost": "BASE", "source_model": model}) for j, model in enumerate(MODELS)}
    indices = np.arange(1, n-1)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": (.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    return data, ledgers, parents, data.date.iloc[2]


def test_known_optimum_and_independent_convex_objective_comparison():
    chosen = minimum_joint_downside([[-.2, 0.], [0., -.1]], .5)
    assert np.isclose(chosen["downside_budget"], .2) and np.isclose(chosen["joint_downside_second_moment"], .004)
    rng = np.random.default_rng(137)
    for _ in range(20):
        values = rng.normal(0, .02, (242, 2))
        previous = float(rng.random())
        answer = minimum_joint_downside(values, previous)
        def objective(weight):
            return np.mean(np.minimum(values[:, 0]*weight+values[:, 1]*(1-weight), 0.)**2)
        independent = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
        assert answer["joint_downside_second_moment"] <= independent.fun+1e-14
        assert answer["joint_downside_second_moment"] <= objective(previous)+1e-14


def test_flat_minimum_chooses_nearest_previous_and_boundaries_are_valid():
    values = np.array([[-.1, .1], [-.02, -.02]])
    assert minimum_joint_downside(values, .3)["downside_budget"] == .3
    assert np.isclose(minimum_joint_downside(values, .8)["downside_budget"], .5)
    assert minimum_joint_downside([[.1, .1], [0., 0.]], .7)["downside_budget"] == .7
    assert minimum_joint_downside([[-.1, -.2], [-.2, -.3]], .5)["downside_budget"] == 1.
    assert minimum_joint_downside([[-.2, -.1], [-.3, -.2]], .5)["downside_budget"] == 0.


def test_positive_and_negative_returns_offset_before_loss_is_measured():
    values = np.array([[-.1, .1], [.1, -.1]])
    chosen = minimum_joint_downside(values, .2)
    assert np.isclose(chosen["downside_budget"], .5) and chosen["joint_downside_second_moment"] < 1e-28
    incorrect_separate_losses = np.mean((np.minimum(values, 0.) @ np.array([.5, .5]))**2)
    assert incorrect_separate_losses > .002
    doubled = np.vstack([values, np.zeros_like(values)])
    assert minimum_joint_downside(doubled, .2)["joint_downside_second_moment"] < 1e-28


def test_base_risk_is_shared_targets_keep_cost_and_missing_states_remain_distinct():
    data, ledgers, parents, start = fixture()
    frames, summaries = joint_downside_reference_frames(data, ledgers, parents, CFG, start)
    base, stress = frames["BASE"], frames["STRESS"]
    pd.testing.assert_series_equal(base.downside_budget, stress.downside_budget)
    assert base.target.iloc[1] == .5 and stress.target.iloc[1] == .35
    assert summaries[0]["budget_changes"] > 0
    assert np.allclose(base.target.iloc[1:-1], base.downside_budget.iloc[1:-1]*.8+base.continuous_budget.iloc[1:-1]*.2)
    for parent in parents["BASE"].values():
        parent.loc[parent.origin_index.eq(400), "reference_weight"] = np.nan
    missing_target, _ = joint_downside_reference_frames(data, ledgers, parents, CFG, start)
    assert pd.isna(missing_target["BASE"].target.iloc[400])
    for ledger in ledgers.values():
        ledger.loc[ledger.date.ge(data.date.iloc[300]), "net_return"] = np.nan
    missing, _ = joint_downside_reference_frames(data, ledgers, parents, CFG, start)
    assert "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET" in set(missing["BASE"].risk_status)
    assert missing["BASE"].downside_budget.iloc[300:-1].nunique() == 1


def test_future_prefix_and_terminal_open_cannot_change_earlier_budget():
    data, ledgers, parents, start = fixture()
    original, _ = joint_downside_reference_frames(data, ledgers, parents, CFG, start)
    altered = {k: v.copy() for k, v in ledgers.items()}
    for ledger in altered.values():
        ledger.loc[ledger.date.ge(data.date.iloc[430]), "net_return"] = -.9
    changed, _ = joint_downside_reference_frames(data, altered, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:430], changed["BASE"].iloc[:430])
    altered = {k: v.copy() for k, v in ledgers.items()}
    for ledger in altered.values():
        ledger.loc[len(ledger)-1, "net_return"] = 100.
    terminal, _ = joint_downside_reference_frames(data, altered, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"], terminal["BASE"])
    short_ledgers = {k: v[v.date.le(data.date.iloc[399])].copy().reset_index(drop=True) for k, v in ledgers.items()}
    for ledger in short_ledgers.values():
        ledger.loc[len(ledger)-1, "mark_clock"] = "OPEN_TERMINAL"
    short_parents = {cost: {k: v[v.origin_index.lt(399)].copy() for k, v in group.items()} for cost, group in parents.items()}
    shorter, _ = joint_downside_reference_frames(data.iloc[:400], short_ledgers, short_parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:399], shorter["BASE"].iloc[:399])


def test_actual_account_partial_exit_full_exit_reentry_and_dividend():
    data, ledgers, parents, start = fixture()
    for group in parents.values():
        for parent in group.values():
            parent.loc[parent.origin_index.between(350, 370), "reference_weight"] = 0.
    data.loc[20, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[19]], "ex_date": [data.date.iloc[20]],
        "payment_date": [data.date.iloc[22]], "cash_dividend_per_share": [.1]})
    frames, _ = joint_downside_reference_frames(data, ledgers, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    assert ((ledger.filled_quantity < 0) & (ledger.shares > 0)).any()
    assert ledger[ledger.date.eq(data.date.iloc[351])].shares.iloc[0] == 0
    assert ledger[ledger.date.eq(data.date.iloc[372])].filled_quantity.iloc[0] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[19])].shares.iloc[0]*.1)
    assert np.isclose(ledger.dividend_recognized.sum(), ledger.dividend_paid.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
