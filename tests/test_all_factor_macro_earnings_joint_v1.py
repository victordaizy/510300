"""共同来源时钟、原覆盖门及成熟池边界的必要检验。"""
import numpy as np
import pandas as pd
import pytest

from research.all_factor_macro_earnings_joint_v1 import EARNINGS, TECH, bind, common_pool


def frames(n=8):
    dates = pd.bdate_range("2023-05-01", periods=n)
    daily = pd.DataFrame({"date": dates, **{k: np.ones(n) for k in TECH},
        "first_passage_feature_known": True,
        "weekly_last_date": dates - pd.Timedelta(days=7),
        "weekly_available_date": dates - pd.Timedelta(days=1),
        "orders_known": True, "funding_known": True, "margin_known": True,
        "orders_available_at": "2023-04-28T09:30:00+08:00",
        "funding_available_at": "2023-04-28T09:30:00+08:00",
        "funding_policy_known_at": "2023-04-28T09:30:00+08:00",
        "margin_available_at": "2023-04-28T09:30:00+08:00",
        "fund_stat_date": dates - pd.Timedelta(days=1),
        "margin_stat_date": dates - pd.Timedelta(days=1),
        "pmi_orders_level": 1., "pmi_orders_change": .1,
        "funding_gap_pp": np.arange(n, dtype=float) / 10,
        "funding_gap_change5": 999.,
        "financing_net_change5": .01, "financing_buy_activity": .02})
    earnings = pd.DataFrame({"date": dates, **{k: np.full(n, .6) for k in EARNINGS},
        "member_count": 300, "valid_company_count": 250, "fundamental_valid": True,
        "latest_publication_date": pd.Timestamp("2023-04-28"),
        "latest_available_date": pd.Timestamp("2023-05-01")})
    return daily, earnings


def test_cutoff_recomputes_five_day_change_without_late_prior_value():
    d, e = frames()
    d.loc[0, "funding_available_at"] = "2023-05-01T15:06:00+08:00"
    result = bind(d, e)
    assert pd.isna(result.loc[0, "funding_gap_pp"])
    assert pd.isna(result.loc[5, "funding_gap_change5"])
    assert result.loc[6, "funding_gap_change5"] == pytest.approx(.5)
    assert result.loc[6, "joint_source_known"]


def test_earnings_coverage_and_missing_end_never_become_qualified():
    d, e = frames()
    e.loc[6, "valid_company_count"] = 239
    e.loc[6, "fundamental_valid"] = False
    e = e.iloc[:-1]
    result = bind(d, e)
    assert not result.loc[6, "source_earnings_known"]
    assert result.loc[6, "raw_earnings_context_only"]
    assert not result.loc[7, "source_earnings_known"]
    assert pd.isna(result.loc[7, "profit_improve_share"])


def test_naive_macro_time_and_duplicate_earnings_are_not_admitted():
    d, e = frames()
    d.loc[6, "margin_available_at"] = "2023-05-09 09:30:00"
    result = bind(d, e)
    assert not result.loc[6, "source_macro_known"]
    with pytest.raises(ValueError, match="日期重复"):
        bind(d, pd.concat([e, e.iloc[[0]]], ignore_index=True))


def test_common_pool_excludes_future_maturity_and_unqualified_source():
    d, e = frames()
    result = bind(d, e)
    result["joint_features_known"] = True
    result.loc[2, "joint_features_known"] = False
    labels = pd.DataFrame({"origin_index": range(8), "origin": result.date,
        "status": "MATURE_REFERENCE", "entry_idx": np.arange(8) + 1,
        "exit_idx": np.arange(8) + 2, "mature_idx": np.arange(8) + 2,
        "event_class": "PROFIT", "reference_net_return": .02})
    pool = common_pool(result, labels, 6)
    assert set(pool.origin_index) == {0, 1, 3, 4}
    assert pool.mature_idx.max() <= 6
    assert np.isfinite(pool.fit_weight).all()
    assert pool.fit_weight.mean() == pytest.approx(1.)
