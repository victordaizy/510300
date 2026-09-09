"""验证状态组合权重守恒、已结算回报时钟和过去预测不被未来改写。"""
import json
import numpy as np
import pandas as pd

from research.contextual_expert_tracking_v1 import FAMILIES, META, fixed_share, online_tracking


def sample(n=25):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "sma120": np.tile([-.1, .1], (n+1)//2)[:n],
                         "vol_ratio": np.ones(n), "variance60": np.full(n, .0001)})
    reward = np.tile(np.array([0, .002, .003, -.002, .001, -.001]), (n, 1))
    target = np.tile(np.array([0, 1, .8, .2, .5, .6]), (n, 1))
    config = {"learning_rate": .17, "share_fraction": 1/126, "minimum_daily_risk_scale": .005, "context_prior_observations": 20}
    return data, reward, target, config


def test_share_preserves_probability_and_can_restore_a_weak_expert():
    old = np.array([1 - 5e-12] + [1e-12] * 5)
    updated = fixed_share(old, np.array([-3, 3, 3, 3, 3, 3]), .17, 1/126)
    assert np.isfinite(updated).all() and (updated > 0).all()
    assert abs(updated.sum() - 1) < 1e-12
    assert updated[1] > old[1]


def test_reward_is_assigned_to_previous_decision_state():
    d, r, x, c = sample(5)
    _, _, updates = online_tracking(d, r, x, c, 0)
    row = updates.iloc[1]
    assert row.updated_previous_state4 == 0 and row.current_state4 == 2
    weights = np.array(json.loads(row.all_local_state_weights))
    np.testing.assert_allclose(weights[2], np.ones(6)/6)
    assert weights[0, 2] > weights[0, 3]
    assert row.latest_observed_return_date == d.date.iloc[1]


def test_future_rewards_and_targets_do_not_change_past_weights():
    d, r, x, c = sample(25)
    p, w, _ = online_tracking(d, r, x, c, 0)
    r[16:, 1:] = -.5
    x[16:, 1:] = 0
    q, v, _ = online_tracking(d, r, x, c, 0)
    for key in META:
        np.testing.assert_array_equal(p[key][:16], q[key][:16])
    pd.testing.assert_frame_equal(w[w.date < d.date.iloc[16]], v[v.date < d.date.iloc[16]])


def test_future_append_preserves_existing_nonterminal_origins():
    d, r, x, c = sample(25)
    p, _, _ = online_tracking(d.iloc[:20].copy(), r[:20], x[:20], c, 0)
    q, w, _ = online_tracking(d, r, x, c, 0)
    for key in META:
        np.testing.assert_array_equal(p[key][:-1], q[key][:19])
    np.testing.assert_allclose(w[FAMILIES].sum(axis=1), 1)
    assert w.target.between(0, 1).all()


def test_reward_scale_uses_previous_day_risk():
    d, r, x, c = sample(8)
    p, w, updates = online_tracking(d, r, x, c, 0)
    changed = d.copy()
    changed.loc[4, "variance60"] = .25
    q, v, other = online_tracking(changed, r, x, c, 0)
    for key in META:
        np.testing.assert_array_equal(p[key][:5], q[key][:5])
    assert updates.iloc[4].previous_day_risk_scale == .01
    assert other.iloc[5].previous_day_risk_scale == .5
