"""核对解释评分的时点、缺失与同源边界，不检验未来交易收益。"""
import pandas as pd

from research.all_factor_joint_scorecard_v1 import available, core_score, credit_at, observations


def test_naive_or_future_publication_cannot_be_known():
    cutoff = pd.Timestamp("2023-06-19 15:05", tz="Asia/Shanghai")
    assert not available("2023-06-19 15:19:23+08:00", cutoff)
    assert not available("2023-06-19 14:00", cutoff)
    assert available("2023-06-19 14:00+08:00", cutoff)


def test_credit_announcement_and_modified_prior_both_required():
    row = {"month": "2023-06", "announcement_at": "2023-06-20 09:15+08:00",
           "survey_published_at": "2023-06-19 15:19:23+08:00",
           "survey_modified_at": "2023-06-19 15:19:23+08:00",
           "same_instrument_prior_version_clock_valid": True,
           "joint_credit_state": "DUAL_CREDIT_UNDERDELIVERY"}
    ledger = pd.DataFrame([row])
    assert credit_at(ledger, pd.Timestamp("2023-06-15 15:05", tz="Asia/Shanghai"))[0] is None
    assert credit_at(ledger, pd.Timestamp("2023-06-20 15:05", tz="Asia/Shanghai"))[0] == "DUAL_CREDIT_UNDERDELIVERY"
    ledger.loc[0, "joint_credit_state"] = "MATCHED_BOTH_TENORS"
    assert credit_at(ledger, pd.Timestamp("2023-06-20 15:05", tz="Asia/Shanghai"))[0] == "MATCHED_BOTH_TENORS"
    ledger.loc[0, "survey_modified_at"] = "2023-06-20 09:16+08:00"
    assert credit_at(ledger, pd.Timestamp("2023-06-20 15:05", tz="Asia/Shanghai"))[0] is None


def test_missing_core_input_is_unknown_not_bearish_zero():
    score = core_score({"daily_hist_atr": None, "momentum5_scaled": 1.})
    assert score["core_explanatory_score"] is None


def test_duplicate_common_source_count_never_increases_grade():
    values = {"daily_hist_atr": .2, "momentum5_scaled": 1., "ac": 4., "setup_high": 3.9,
              "support_information": True, "support_common_count": 1, "funding_gap_pp": .1,
              "financing_net_change5": -.01}
    a = core_score(values)
    b = core_score({**values, "support_common_count": 20})
    assert a["core_explanatory_score"] == b["core_explanatory_score"] == 60


def test_lpr_under_delivery_only_changes_existing_position_review():
    values = {"daily_hist_atr": .2, "momentum5_scaled": 1., "ac": 4., "setup_high": 3.9,
              "support_information": True, "credit_state": "DUAL_CREDIT_UNDERDELIVERY"}
    assert core_score(values, False)["core_explanatory_score"] == 60
    assert core_score(values, True)["core_explanatory_score"] == 20


def test_original_sixteen_hour_known_flags_do_not_bypass_fifteen_hour_clock():
    cutoff = pd.Timestamp("2023-06-15 15:05", tz="Asia/Shanghai")
    row = pd.Series({"orders_known": True, "orders_available_at": "2023-06-15 15:30+08:00",
                     "pmi_orders_level": -1., "pmi_orders_change": -.5,
                     "funding_known": True, "funding_available_at": "2023-06-15 15:30+08:00",
                     "funding_policy_known_at": "2023-06-13 09:20+08:00", "funding_gap_pp": -.1,
                     "funding_gap_change5": -.2, "margin_known": True,
                     "margin_available_at": "2023-06-15 15:30+08:00", "financing_net_change5": .01,
                     "financing_buy_activity": .1, "weekly_last_date": pd.Timestamp("2023-06-16"),
                     "weekly_available_date": pd.Timestamp("2023-06-17"), "weekly_hist_atr": .2})
    ledger = pd.DataFrame(columns=["month"])
    values, _ = observations(row, None, ledger, cutoff)
    assert values["pmi_orders"] is None
    assert values["funding_gap_pp"] is None
    assert values["financing_net_change5"] is None
    assert values["weekly_hist_atr"] is None
