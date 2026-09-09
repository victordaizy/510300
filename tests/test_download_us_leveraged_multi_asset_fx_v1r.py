"""V1R官方USD/CNY中间价采集器的合成测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.download_us_leveraged_multi_asset_fx_v1r import (
    FxCorrectionDataError,
    load_override,
    parse_usd_cny_pages,
    strict_lag_coverage,
)


def _page(records: list[tuple[str, str]], currency: str = "USD/CNY") -> dict:
    return {
        "head": {"rep_code": "200"},
        "data": {"currency": currency},
        "records": [
            {"date": date, "values": [value]} for date, value in records
        ],
    }


def test_override_only_allows_official_fx_source_and_14_day_holiday_window() -> None:
    payload = load_override()
    correction = payload["correction"]
    assert correction["source_to"] == "CHINAMONEY_CFETS_OFFICIAL_USD_CNY_CENTRAL_PARITY"
    assert correction["maximum_fx_staleness_calendar_days_to"] == 14
    assert correction["strategy_return_or_rank_may_be_computed_during_acquisition"] is False


def test_parse_official_usd_cny_pages_is_unique_positive_and_sorted() -> None:
    frame = parse_usd_cny_pages(
        [
            _page([("2020-01-03", "6.9681"), ("2020-01-02", "6.9614")]),
            _page([("2019-12-31", "6.9762")]),
        ]
    )
    assert frame["date"].tolist() == [
        pd.Timestamp("2019-12-31"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    assert frame["cny_per_usd"].tolist() == pytest.approx([6.9762, 6.9614, 6.9681])


def test_parse_rejects_duplicate_dates_and_currency_drift() -> None:
    with pytest.raises(FxCorrectionDataError, match="日期重复"):
        parse_usd_cny_pages(
            [_page([("2020-01-02", "6.9")]), _page([("2020-01-02", "6.9")])]
        )
    with pytest.raises(FxCorrectionDataError, match="币种漂移"):
        parse_usd_cny_pages([_page([("2020-01-02", "6.9")], currency="EUR/CNY")])


def test_strict_lag_coverage_uses_prior_observation_and_reports_holiday_age() -> None:
    dates = pd.to_datetime(["2020-09-30", "2020-10-09", "2020-10-12"])
    panel = pd.concat(
        [pd.DataFrame({"ticker": ticker, "date": dates}) for ticker in ["UPRO", "TQQQ", "TMF", "UGL"]],
        ignore_index=True,
    )
    fx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-09-29", "2020-09-30", "2020-10-09"]),
            "cny_per_usd": [6.8, 6.79, 6.78],
        }
    )
    coverage = strict_lag_coverage(
        panel,
        fx,
        start=pd.Timestamp("2020-09-30"),
        end=pd.Timestamp("2020-10-12"),
    )
    assert coverage["strictly_lagged"] is True
    assert coverage["maximum_fx_age_calendar_days"] == 9
