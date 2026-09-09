from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_510300_structural_equity_risk_premium_engine_v1 import (
    derive_independent_share_snapshot,
    merge_asof_state,
    merge_present_value_with_risk_free,
    normalize_statement,
    parse_financial_sector_document,
    prepare_independent_share_facts,
    resolve_effective_company_types,
)
from research.structural_equity_risk_premium_engine_v1 import (
    aggregate_cashflow_snapshot,
    build_component_snapshot,
    build_fixed_present_value_portfolios,
    derive_balance_snapshot,
    derive_ttm_pair_snapshot,
    detect_drawdown_episodes,
    first_principal_component,
    normalized_hhi,
    prepare_statement_vintages,
)


def test_ttm_snapshot_uses_only_versions_available_at_origin() -> None:
    facts = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"] * 6,
            "report_end": pd.to_datetime(
                [
                    "2021-09-30",
                    "2021-12-31",
                    "2022-09-30",
                    "2022-12-31",
                    "2023-09-30",
                    "2023-09-30",
                ]
            ),
            "available_at": pd.to_datetime(
                [
                    "2021-10-30",
                    "2022-03-30",
                    "2022-10-30",
                    "2023-03-30",
                    "2023-10-30",
                    "2024-03-30",
                ]
            ),
            "operating_profit": [80.0, 100.0, 90.0, 120.0, 105.0, 999.0],
        }
    )
    prepared = prepare_statement_vintages(facts, ["operating_profit"])
    snapshot = derive_ttm_pair_snapshot(
        prepared,
        ["000001.SZ"],
        origin=pd.Timestamp("2023-11-30"),
        value_columns=["operating_profit"],
        prefix="income",
    )
    assert len(snapshot) == 1
    assert snapshot.loc[0, "income_report_end"] == pd.Timestamp("2023-09-30")
    assert snapshot.loc[0, "operating_profit_ttm"] == 135.0
    assert snapshot.loc[0, "operating_profit_ttm_prior_year"] == 110.0


def test_balance_snapshot_requires_three_point_in_time_equity_observations() -> None:
    facts = pd.DataFrame(
        {
            "stock_code": ["600000.SH"] * 3,
            "report_end": pd.to_datetime(
                ["2021-09-30", "2022-09-30", "2023-09-30"]
            ),
            "available_at": pd.to_datetime(
                ["2021-10-30", "2022-10-30", "2023-10-30"]
            ),
            "total_parent_equity": [100.0, 110.0, 120.0],
            "total_assets": [500.0, 520.0, 550.0],
            "share_capital": [10.0, 10.0, 10.0],
        }
    )
    prepared = prepare_statement_vintages(
        facts, ["total_parent_equity", "total_assets", "share_capital"]
    )
    snapshot = derive_balance_snapshot(
        prepared, ["600000.SH"], origin=pd.Timestamp("2023-11-30")
    )
    assert len(snapshot) == 1
    assert snapshot.loc[0, "parent_equity_current"] == 120.0
    assert snapshot.loc[0, "parent_equity_prior_year"] == 110.0
    assert snapshot.loc[0, "parent_equity_two_years_prior"] == 100.0


def test_normalized_hhi_has_expected_boundaries() -> None:
    assert normalized_hhi(pd.Series([1.0])) == 1.0
    assert normalized_hhi(pd.Series([1.0, 1.0, 1.0])) == 0.0
    assert 0.0 < normalized_hhi(pd.Series([8.0, 1.0, 1.0])) < 1.0
    assert normalized_hhi(pd.Series([0.0, np.nan])) == 0.0


def test_first_principal_component_is_anchored_to_positive_direction() -> None:
    index = pd.RangeIndex(30)
    anchor = pd.Series(np.linspace(-2.0, 2.0, len(index)), index=index)
    frame = pd.DataFrame(
        {
            "portfolio_a": anchor * 2.0 + 0.1,
            "portfolio_b": anchor * 0.5 - 0.2,
            "portfolio_c": anchor * 1.2 + 0.3,
        },
        index=index,
    )
    result = first_principal_component(
        frame, sign_anchor=anchor, minimum_rows=24
    )
    assert result.status.startswith("PASS_")
    aligned = pd.concat([result.factor, anchor], axis=1).dropna()
    assert aligned.corr().iloc[0, 1] > 0.99
    assert result.explained_variance_ratio is not None
    assert result.explained_variance_ratio > 0.99


def test_drawdown_detector_uses_complete_underwater_episode() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    wealth = [1.0, 1.10, 1.00, 0.80, 0.90, 1.05, 1.11, 1.12]
    episodes = detect_drawdown_episodes(
        dates, wealth, activation_threshold=-0.10
    )
    assert len(episodes) == 1
    row = episodes.iloc[0]
    assert row["peak_date"] == dates[1]
    assert row["trough_date"] == dates[3]
    assert row["recovery_date"] == dates[6]
    assert np.isclose(row["peak_to_trough_drawdown"], 0.80 / 1.10 - 1.0)
    assert not bool(row["right_censored"])


def test_component_market_cap_uses_frozen_close_and_reported_shares_fallback() -> None:
    market = pd.DataFrame(
        {
            "stock_code": ["600000.SH"],
            "industry_code": ["BANK"],
            "industry_name": ["银行"],
            "total_market_cap": [np.nan],
            "market_asof_date": pd.to_datetime(["2015-01-30"]),
            "total_shares": [10.0],
            "report_end_close": [np.nan],
            "fallback_raw_close": [12.0],
        }
    )
    income = pd.DataFrame(
        {
            "stock_code": ["600000.SH"],
            "income_report_end": pd.to_datetime(["2014-09-30"]),
            "total_operating_revenue_ttm": [200.0],
            "total_operating_revenue_ttm_prior_year": [180.0],
            "operating_profit_ttm": [40.0],
            "operating_profit_ttm_prior_year": [35.0],
            "parent_net_profit_ttm": [30.0],
            "parent_net_profit_ttm_prior_year": [25.0],
        }
    )
    cashflow = pd.DataFrame(
        {
            "stock_code": ["600000.SH"],
            "operating_cashflow_ttm": [45.0],
            "operating_cashflow_ttm_prior_year": [40.0],
        }
    )
    balance = pd.DataFrame(
        {
            "stock_code": ["600000.SH"],
            "balance_report_end": pd.to_datetime(["2014-09-30"]),
            "parent_equity_current": [120.0],
            "parent_equity_prior_year": [110.0],
            "parent_equity_two_years_prior": [100.0],
            "share_capital_current": [10.0],
        }
    )
    component = build_component_snapshot(
        market,
        income,
        cashflow,
        balance,
        origin=pd.Timestamp("2015-01-30"),
        growth_clip=(-1.0, 3.0),
        roe_clip=(-1.0, 1.0),
        roe_change_clip=(-1.0, 1.0),
    )
    assert component.loc[0, "total_market_cap"] == 120.0
    assert component.loc[0, "state_weight"] == 1.0
    assert (
        component.loc[0, "state_market_cap_source"]
        == "POINT_IN_TIME_CLOSE_TIMES_REPORTED_TOTAL_SHARES_FALLBACK"
    )


def test_present_value_portfolio_family_is_exactly_the_frozen_sixteen() -> None:
    rank = pd.Series(np.linspace(0.05, 0.95, 9))
    frame = pd.DataFrame(
        {
            "origin": pd.Timestamp("2021-01-29"),
            "state_weight": np.repeat(1.0 / 9.0, 9),
            "earnings_to_price": np.linspace(0.01, 0.09, 9),
            "book_to_price": np.linspace(0.10, 0.90, 9),
            "sales_to_price": np.linspace(0.20, 1.00, 9),
            "operating_cashflow_to_price": np.linspace(0.03, 0.11, 9),
            "earnings_to_price_winsor": np.linspace(0.01, 0.09, 9),
            "book_to_price_winsor": np.linspace(0.10, 0.90, 9),
            "sales_to_price_winsor": np.linspace(0.20, 1.00, 9),
            "operating_cashflow_to_price_winsor": np.linspace(0.03, 0.11, 9),
            "earnings_to_price_industry_rank": rank,
            "book_to_price_industry_rank": rank,
            "sales_to_price_industry_rank": rank,
            "operating_cashflow_to_price_industry_rank": rank,
            "quality_rank": rank.iloc[::-1].reset_index(drop=True),
            "duration_proxy": rank,
        }
    )
    portfolios = build_fixed_present_value_portfolios(frame)
    expected = {
        f"{prefix}_{bucket}"
        for prefix in ["EP", "BP", "SP", "CFP"]
        for bucket in ["LOW", "MID", "HIGH"]
    } | {
        "HIGH_EP_HIGH_QUALITY",
        "HIGH_EP_LOW_QUALITY",
        "LONG_DURATION",
        "SHORT_DURATION",
    }
    assert len(portfolios) == 16
    assert set(portfolios["portfolio_id"]) == expected


def test_financial_availability_uses_notice_date_and_discloses_revision_risk() -> None:
    row = {
        "SECUCODE": "600000.SH",
        "SECURITY_CODE": "600000",
        "SECURITY_TYPE_CODE": "058001001",
        "ORG_CODE": "10000001",
        "REPORT_DATE": "2022-12-31",
        "REPORT_TYPE": "年报",
        "REPORT_DATE_NAME": "2022年年报",
        "NOTICE_DATE": "2023-03-31",
        "UPDATE_DATE": "2024-04-30",
        "CURRENCY": "CNY",
        "source_page": 1,
        "source_sha256": "0" * 64,
        "TOTAL_OPERATE_INCOME": 100.0,
        "OPERATE_PROFIT": 20.0,
        "PARENT_NETPROFIT": 15.0,
    }
    frame, conflicts, stats = normalize_statement([row], "income")
    assert conflicts.empty
    assert frame.loc[0, "available_at"] == pd.Timestamp("2023-03-31")
    assert (
        frame.loc[0, "revision_risk_status"]
        == "CURRENT_VALUE_MAY_INCLUDE_POST_NOTICE_REVISION"
    )
    assert stats["revision_delayed_rows"] == 1


def test_asof_state_merge_normalizes_mixed_datetime_units() -> None:
    base = pd.DataFrame(
        {
            "date": pd.Series(
                np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[ms]")
            )
        }
    )
    source = pd.DataFrame(
        {
            "available_at": pd.Series(
                np.array(["2021-01-01"], dtype="datetime64[us]")
            ),
            "value": [3.0],
        }
    )
    merged = merge_asof_state(
        base,
        source,
        base_date="date",
        source_date="available_at",
        columns=["value"],
    )
    assert merged["value"].tolist() == [3.0, 3.0]
    assert str(merged["date"].dtype) == "datetime64[ns]"


def test_independent_share_fallback_is_point_in_time() -> None:
    source = pd.DataFrame(
        {
            "con_code": ["600000.SH", "600000.SH"],
            "report_period": ["2021-12-31", "2022-03-31"],
            "available_at": ["2022-03-30", "2022-04-30"],
            "total_shares": [10.0, 11.0],
        }
    )
    prepared = prepare_independent_share_facts(source)
    snapshot = derive_independent_share_snapshot(
        prepared, ["600000.SH"], origin=pd.Timestamp("2022-04-15")
    )
    assert snapshot.loc[0, "independent_total_shares"] == 10.0
    assert snapshot.loc[0, "independent_share_report_end"] == pd.Timestamp(
        "2021-12-31"
    )


def test_cashflow_gate_cannot_pass_when_market_cap_members_are_below_gate() -> None:
    component = pd.DataFrame(
        {
            "origin": pd.Timestamp("2021-01-29"),
            "stock_code": ["600000.SH"],
            "state_weight": [1.0],
            "revenue_yoy": [0.1],
            "operating_profit_yoy": [0.1],
            "operating_cashflow_yoy": [0.1],
            "roe_change": [0.01],
        }
    )
    result = aggregate_cashflow_snapshot(
        component,
        minimum_coverages={
            "revenue": 0.8,
            "operating_profit": 0.8,
            "operating_cashflow": 0.7,
            "roe_change": 0.8,
        },
        minimum_market_cap_members=2,
    )
    assert result["cf_data_status"] == "NO_VIEW_MARKET_CAP_MEMBER_COVERAGE_BELOW_GATE"
    assert not result["market_cap_gate_pass"]


def test_financial_sector_serializer_empty_object_is_audited_as_empty_batch() -> None:
    rows, response_status = parse_financial_sector_document(
        {
            "$types": {
                "System.Object, mscorlib, Version=4.0.0.0, Culture=neutral, "
                "PublicKeyToken=b77a5c561934e089": "1"
            },
            "$type": "1",
        },
        stock_code="000562.SZ",
        statement="balance",
    )
    assert rows == []
    assert response_status == "SERIALIZER_EMPTY_OBJECT_NO_ROWS"


def test_financial_sector_unknown_missing_data_response_remains_a_hard_failure() -> None:
    with np.testing.assert_raises_regex(ValueError, "缺少 data 列表"):
        parse_financial_sector_document(
            {"success": False, "message": "temporary error"},
            stock_code="600000.SH",
            statement="balance",
        )


def test_historical_company_type_override_replaces_current_profile_type() -> None:
    effective = resolve_effective_company_types(
        profile_types={"000627.SZ": "1", "600000.SH": "3"},
        historical_overrides={"000627.SZ": "2"},
        target_codes={"000627.SZ", "600000.SH"},
        allowed_company_types={"1": "SECURITIES", "2": "INSURANCE", "3": "BANK"},
    )
    assert effective == {"000627.SZ": "2", "600000.SH": "3"}


def test_present_value_risk_free_merge_normalizes_mixed_datetime_units() -> None:
    present_value = pd.DataFrame(
        {
            "origin": pd.Series(
                np.array(["2021-01-29", "2021-02-26"], dtype="datetime64[s]")
            ),
            "factor": [1.0, 2.0],
        }
    )
    risk_free = pd.DataFrame(
        {
            "date": pd.Series(
                np.array(["2021-01-28", "2021-02-25"], dtype="datetime64[ms]")
            ),
            "cgb_10y": [3.1, 3.2],
        }
    )
    merged = merge_present_value_with_risk_free(present_value, risk_free)
    assert merged["cgb_10y"].tolist() == [3.1, 3.2]
    assert str(merged["origin"].dtype) == "datetime64[ns]"
    assert str(merged["cgb_date"].dtype) == "datetime64[ns]"
