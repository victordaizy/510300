"""检查财务事件篮子的过去信息约束和缺项处理。"""
import numpy as np
import pandas as pd

from research.factor96_t11_financial_cohorts_v1 import disclosure_frontier, measure_financial_cohorts


def row(identifier, company, period, date, quality, surprise=2., industry="A", known=True, member=True):
    return {"announcement_id": identifier, "ts_code": company, "report_period": pd.Timestamp(period),
            "available_date": pd.Timestamp(date), "industry_l1_code": industry, "industry_match_count": 1,
            "is_financial": False, "member_at_available": member, "L02": surprise, "L04_change": quality,
            "L02_known": known, "L04_known": known, "joint_known": known}


def baseline():
    return [row("old1", "oldA", "2020-03-31", "2020-04-28", -.1),
            row("old2", "oldB", "2021-03-31", "2021-04-26", .1),
            row("new", "newC", "2022-03-31", "2022-04-25", .2)]


def target(frame):
    return frame.loc[frame.date.eq(pd.Timestamp("2022-04-25"))].iloc[0]


def test_reference_uses_prior_same_quarter_and_preserves_nonzero_signal():
    daily, companies, dependencies = measure_financial_cohorts(pd.DataFrame(baseline()))
    observed = target(daily)
    assert observed.cohort_known and observed.positive_financial_measurement
    assert np.isclose(observed.L04_industry_cohort, np.sqrt(2))
    refs = dependencies.loc[dependencies.target_announcement_id.eq("new")]
    assert set(refs.source_announcement_id) == {"old1", "old2"}


def test_future_records_cannot_change_past_cohort():
    original, _, _ = measure_financial_cohorts(pd.DataFrame(baseline()))
    extended = baseline() + [row("future", "futureD", "2023-03-31", "2023-04-25", 1000000.)]
    changed, _, _ = measure_financial_cohorts(pd.DataFrame(extended))
    pd.testing.assert_series_equal(target(original), target(changed))


def test_missing_latest_report_does_not_fall_back_to_known_annual_report():
    rows = baseline()
    rows[-1] = row("new", "newC", "2022-03-31", "2022-04-25", np.nan, known=False)
    rows.append(row("annual", "newC", "2021-12-31", "2022-04-25", .5))
    frontier = disclosure_frontier(pd.DataFrame(rows))
    assert "annual" not in set(frontier.announcement_id)
    daily, _, _ = measure_financial_cohorts(pd.DataFrame(rows))
    assert not target(daily).cohort_known
    assert target(daily).status == "NO_VIEW_INCOMPLETE_DISCLOSURE_COHORT"


def test_late_old_report_cannot_reopen_a_new_event():
    rows = baseline() + [row("q3", "newC", "2022-09-30", "2022-10-25", .1),
                         row("lateh1", "newC", "2022-06-30", "2022-11-01", .2)]
    assert "lateh1" not in set(disclosure_frontier(pd.DataFrame(rows)).announcement_id)


def test_zero_industry_dispersion_stays_unknown():
    rows = baseline()
    rows[0]["L04_change"] = rows[1]["L04_change"] = 0.
    daily, _, _ = measure_financial_cohorts(pd.DataFrame(rows))
    assert not target(daily).cohort_known
    assert pd.isna(target(daily).L04_industry_cohort)


def test_reference_window_is_two_calendar_years():
    rows = baseline()
    rows[0]["available_date"] = pd.Timestamp("2020-04-24")
    _, companies, _ = measure_financial_cohorts(pd.DataFrame(rows))
    observed = companies.loc[companies.announcement_id.eq("new")].iloc[0]
    assert observed.industry_reference_count == 1 and not observed.known


def test_one_missing_current_company_makes_whole_cohort_unknown():
    rows = baseline() + [row("missing", "missingD", "2022-03-31", "2022-04-25", np.nan, known=False)]
    daily, _, _ = measure_financial_cohorts(pd.DataFrame(rows))
    assert target(daily).nonfinancial_reports == 2
    assert not target(daily).cohort_known and not target(daily).positive_financial_measurement


def test_industry_medians_do_not_allow_one_extreme_company_to_dominate():
    rows = baseline()[:2] + [row("b1", "oldB1", "2020-03-31", "2020-04-28", -.1, industry="B"),
                             row("b2", "oldB2", "2021-03-31", "2021-04-26", .1, industry="B")]
    for identifier, surprise, industry in [("a1", 0., "A"), ("a2", 0., "A"), ("a3", 100., "A"), ("b3", 2., "B")]:
        rows.append(row(identifier, identifier, "2022-03-31", "2022-04-25", .2, surprise, industry))
    daily, _, _ = measure_financial_cohorts(pd.DataFrame(rows))
    assert target(daily).L02_cohort == 1.
    assert not target(daily).positive_financial_measurement


def test_positive_relative_cash_score_does_not_hide_absolute_deterioration():
    rows = baseline()
    rows[0]["L04_change"], rows[1]["L04_change"], rows[2]["L04_change"] = -.5, -.3, -.1
    daily, _, _ = measure_financial_cohorts(pd.DataFrame(rows))
    observed = target(daily)
    assert observed.cohort_known and observed.L04_industry_cohort > 0
    assert not observed.positive_financial_measurement
