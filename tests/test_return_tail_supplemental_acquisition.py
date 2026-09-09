from __future__ import annotations

import pandas as pd

from research.return_tail_supplemental_acquisition import (
    normalize_cffex_proxy_daily,
    normalize_sina_five_day_minutes,
    normalize_sina_option_quote,
    normalize_sse_daily_statistics,
    parse_sina_batch_quotes,
)


def test_normalize_cffex_proxy_daily_parses_contract() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["IO2608-P-4200.CFX"],
            "trade_date": ["20260814"],
            "exchange": ["CFFEX"],
            "pre_settle": [10.0],
            "pre_close": [9.0],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "settle": [10.2],
            "vol": [100],
            "amount": [10.0],
            "oi": [200],
        }
    )
    result = normalize_cffex_proxy_daily(raw, "2026-08-16")
    assert result.loc[0, "contract_code"] == "IO2608-P-4200"
    assert result.loc[0, "option_type"] == "P"
    assert result.loc[0, "strike"] == 4200


def test_normalize_sse_statistics_removes_commas_and_hyphens() -> None:
    records = [
        {
            "CONTRACT_VOLUME": "-",
            "CALL_VOLUME": "940,706",
            "LEAVES_QTY": "869,256",
            "CP_RATE": "58.63",
            "PUT_VOLUME": "551,573",
            "TRADE_DATE": "2020-01-02",
            "TOTAL_MONEY": "-",
            "TOTAL_VOLUME": "1,492,279",
            "SECURITY_CODE": "510300",
            "LEAVES_CALL_QTY": "459,561",
            "LEAVES_PUT_QTY": "409,695",
            "SECURITY_ABBR": "300ETF",
        }
    ]
    result = normalize_sse_daily_statistics(records, "2026-08-16")
    assert pd.isna(result.loc[0, "contract_count"])
    assert result.loc[0, "total_volume"] == 1492279
    assert result.loc[0, "open_interest"] == 869256


def test_normalize_sina_quote_preserves_five_levels_and_timestamp() -> None:
    values = ["1"] * 43
    values[20] = "0.12"
    values[22] = "0.11"
    values[32] = "2026-08-14 14:56:00"
    values[36] = "510300"
    values[37] = "300ETF沽8月4200"
    result = normalize_sina_option_quote("10010001", values, "2026-08-16")
    assert result.loc[0, "bid1"] == 0.11
    assert result.loc[0, "ask1"] == 0.12
    assert result.loc[0, "underlying_code"] == "510300.SH"
    assert result.loc[0, "quote_timestamp"] == pd.Timestamp("2026-08-14 14:56:00")


def test_parse_sina_batch_quotes_ignores_expired_empty_records() -> None:
    text = (
        'var hq_str_CON_OP_10000001="1,2,3";\n'
        'var hq_str_CON_OP_10000002="";\n'
    )
    assert parse_sina_batch_quotes(text) == {"10000001": ["1", "2", "3"]}


def test_normalize_sina_five_day_minutes_builds_auditable_timestamp() -> None:
    records = [
        [
            {
                "d": "2026-08-14",
                "i": "09:30:00",
                "p": "0.1200",
                "a": "0.1200",
                "v": "3",
                "t": "0",
            }
        ]
    ]
    result = normalize_sina_five_day_minutes("10000001", records, "2026-08-16")
    assert result.loc[0, "contract_code"] == "10000001.SH"
    assert result.loc[0, "minute_timestamp"] == pd.Timestamp("2026-08-14 09:30:00")
    assert result.loc[0, "price"] == 0.12
    assert result.loc[0, "volume"] == 3
