from __future__ import annotations

import pandas as pd

from scripts.quality_check_daily import (
    build_daily_metadata,
    check_daily_data,
    check_metadata_contract,
)


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-17"),
                "open": 4.70,
                "high": 4.82,
                "low": 4.69,
                "close": 4.80,
                "volume": 1000.0,
                "amount": 4760.0,
                "symbol": "510300.SH",
                "source": "akshare.fund_etf_hist_sina",
                "volume_unit": "share",
                "amount_unit": "CNY",
                "retrieved_at": "2026-08-18T20:00:00+08:00",
            },
            {
                "date": pd.Timestamp("2026-08-18"),
                "open": 4.80,
                "high": 4.90,
                "low": 4.75,
                "close": 4.85,
                "volume": 2000.0,
                "amount": 9650.0,
                "symbol": "510300.SH",
                "source": "sina.hq.batch_quote",
                "volume_unit": "share",
                "amount_unit": "CNY",
                "retrieved_at": "2026-08-18T20:00:00+08:00",
            },
        ]
    )


def _config() -> dict:
    return {
        "project": {
            "start_date": "2021-08-12",
            "end_date": "2026-08-12",
            "feature_warmup_start": "2026-08-17",
        },
        "symbols": {"etf": "510300.SH"},
        "quality": {"price_tick": 0.001},
    }


def test_metadata_records_each_row_level_source(tmp_path) -> None:
    data = _daily_frame()
    data_file = tmp_path / "510300_daily_raw.parquet"
    data.to_parquet(data_file, index=False)
    metadata = build_daily_metadata(
        data,
        data_file,
        requested_start="2026-08-17",
        requested_end="2026-08-18",
        evaluation_start="2021-08-12",
        historical_evaluation_end="2026-08-12",
    )

    assert metadata["row_count"] == 2
    assert metadata["actual_first_date"] == "2026-08-17"
    assert metadata["actual_last_date"] == "2026-08-18"
    assert metadata["source_segments"] == [
        {
            "source": "akshare.fund_etf_hist_sina",
            "row_count": 1,
            "first_date": "2026-08-17",
            "last_date": "2026-08-17",
        },
        {
            "source": "sina.hq.batch_quote",
            "row_count": 1,
            "first_date": "2026-08-18",
            "last_date": "2026-08-18",
        },
    ]


def test_daily_check_accepts_declared_current_quote_fallback(tmp_path) -> None:
    data = _daily_frame()
    data_file = tmp_path / "510300_daily_raw.parquet"
    data.to_parquet(data_file, index=False)
    metadata = build_daily_metadata(
        data,
        data_file,
        requested_start="2026-08-17",
        requested_end="2026-08-18",
        evaluation_start="2021-08-12",
        historical_evaluation_end="2026-08-12",
    )

    errors, warnings = check_daily_data(data, _config(), metadata)

    assert errors == []
    assert warnings == []


def test_metadata_contract_rejects_stale_rows_and_sources(tmp_path) -> None:
    data = _daily_frame()
    data_file = tmp_path / "510300_daily_raw.parquet"
    data.to_parquet(data_file, index=False)
    metadata = build_daily_metadata(
        data,
        data_file,
        requested_start="2026-08-17",
        requested_end="2026-08-18",
        evaluation_start="2021-08-12",
        historical_evaluation_end="2026-08-12",
    )
    metadata["row_count"] = 1
    metadata["source_segments"] = []

    errors = check_metadata_contract(data, metadata, metadata["sha256"])
    codes = {item["code"] for item in errors}

    assert "METADATA_ROW_COUNT_MISMATCH" in codes
    assert "METADATA_SOURCE_SEGMENTS_MISMATCH" in codes
