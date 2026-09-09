from __future__ import annotations

import pandas as pd

from scripts.acquire_510300_asymmetric_stress_hazard_v1_constituent_history import (
    select_required_symbols,
    validate_downloaded_panel,
)


def test_select_required_symbols_uses_only_frozen_index_and_dates() -> None:
    membership = pd.DataFrame(
        {
            "membership_date": ["2015-01-05", "2015-01-05", "2021-01-04"],
            "index_code": ["000300", "000905", "000300"],
            "symbol": ["000001.sz", "000002.SZ", "600000.SH"],
        }
    )
    assert select_required_symbols(membership) == ["000001.SZ"]


def test_validate_panel_allows_suspension_forward_fill() -> None:
    membership = pd.DataFrame(
        {
            "membership_date": ["2015-01-05", "2015-01-06"],
            "index_code": ["000300", "000300"],
            "symbol": ["000001.SZ", "000001.SZ"],
        }
    )
    panel = pd.DataFrame(
        {
            "date": ["2015-01-05"],
            "con_code": ["000001.SZ"],
            "pre_close": [10.0],
            "raw_close": [10.1],
            "total_return_close": [10.1],
        }
    )
    metrics = validate_downloaded_panel(panel, membership)
    assert metrics["validated_member_day_count"] == 2
    assert metrics["missing_member_day_count_after_suspension_forward_fill"] == 0


def test_validate_panel_keeps_pre_first_trade_member_day_as_no_view() -> None:
    membership = pd.DataFrame(
        {
            "membership_date": ["2015-01-05", "2015-01-06"],
            "index_code": ["000300", "000300"],
            "symbol": ["000001.SZ", "000001.SZ"],
        }
    )
    panel = pd.DataFrame(
        {
            "date": ["2015-01-06"],
            "con_code": ["000001.SZ"],
            "pre_close": [10.0],
            "raw_close": [10.1],
            "total_return_close": [10.1],
        }
    )
    metrics = validate_downloaded_panel(panel, membership)
    assert metrics["missing_member_day_count_after_suspension_forward_fill"] == 1
    assert metrics["date_level_no_view_required"] is True
