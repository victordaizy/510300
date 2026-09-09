from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from research.option_research_data_acquisition import (
    merge_underlying_reference,
    normalize_contract_master,
    normalize_option_daily,
    normalize_sse_risk_indicators,
    parse_cffex_month_zip,
    trading_date_chunks,
)


def test_trading_date_chunks_sorts_and_deduplicates() -> None:
    chunks = trading_date_chunks(
        ["2026-08-14", "2026-08-12", "2026-08-13", "2026-08-12"], 2
    )
    assert chunks == [
        [pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-13")],
        [pd.Timestamp("2026-08-14")],
    ]


def test_contract_master_preserves_adjusted_contract() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["10000001.SH", "10000002.SH"],
            "name": ["标准合约", "调整合约A"],
            "opt_code": ["OP510300.SH", "OP510300.SH"],
            "call_put": ["C", "P"],
            "exercise_price": [4.2, 4.288],
            "maturity_date": ["20260826", "20260826"],
            "list_date": ["20260701", "20260701"],
            "delist_date": ["20260826", "20260826"],
            "per_unit": [10000, 10260],
            "opt_multiplier": [10000, 10260],
        }
    )
    result = normalize_contract_master(raw, "2026-08-16 12:00:00+08:00")
    assert result["is_adjusted"].tolist() == [False, True]
    assert result["contract_unit"].tolist() == [10000, 10260]
    assert result["strike"].tolist() == [4.2, 4.288]


def test_option_daily_does_not_invent_bid_ask() -> None:
    master_raw = pd.DataFrame(
        {
            "ts_code": ["10000001.SH"],
            "name": ["测试合约"],
            "opt_code": ["OP510300.SH"],
            "call_put": ["P"],
            "exercise_price": [4.2],
            "maturity_date": ["20260826"],
            "list_date": ["20260701"],
            "delist_date": ["20260826"],
            "per_unit": [10000],
            "opt_multiplier": [10000],
        }
    )
    master = normalize_contract_master(master_raw, "2026-08-16 12:00:00+08:00")
    raw = pd.DataFrame(
        {
            "ts_code": ["10000001.SH"],
            "trade_date": ["20260814"],
            "exchange": ["SSE"],
            "pre_settle": [0.1],
            "pre_close": [0.1],
            "open": [0.11],
            "high": [0.12],
            "low": [0.09],
            "close": [0.10],
            "settle": [0.10],
            "vol": [100],
            "amount": [10.0],
            "oi": [200],
        }
    )
    underlying = pd.DataFrame({"date": ["2026-08-14"], "close": [4.726]})
    result = normalize_option_daily(
        raw, master, underlying, "2026-08-16 12:00:00+08:00"
    )
    assert "bid1" not in result.columns
    assert "ask1" not in result.columns
    assert "quote_timestamp" not in result.columns
    assert result.loc[0, "underlying_close"] == pytest.approx(4.726)


def test_underlying_merge_only_adds_missing_dates_after_cross_check() -> None:
    existing = pd.DataFrame(
        {
            "date": ["2026-08-13"],
            "close": [4.729],
            "source": ["已有来源"],
            "retrieved_at": ["2026-08-13T22:00:00+08:00"],
        }
    )
    downloaded = pd.DataFrame(
        {
            "date": ["2026-08-13", "2026-08-14"],
            "close": [4.729, 4.726],
            "source": ["对照来源", "对照来源"],
            "retrieved_at": [
                pd.Timestamp("2026-08-16T12:00:00+08:00"),
                pd.Timestamp("2026-08-16T12:00:00+08:00"),
            ],
        }
    )
    merged, evidence = merge_underlying_reference(existing, downloaded)
    assert len(merged) == 2
    assert merged.loc[merged["date"].eq(pd.Timestamp("2026-08-13")), "source"].item() == "已有来源"
    assert evidence["added_day_count"] == 1
    assert merged["retrieved_at"].map(type).eq(str).all()


def test_underlying_merge_rejects_price_mismatch() -> None:
    existing = pd.DataFrame({"date": ["2026-08-13"], "close": [4.729]})
    downloaded = pd.DataFrame({"date": ["2026-08-13"], "close": [4.700]})
    with pytest.raises(ValueError, match="重叠收盘价不一致"):
        merge_underlying_reference(existing, downloaded)


def test_sse_risk_filters_510300_and_preserves_zero_iv() -> None:
    records = [
        {
            "TRADE_DATE": "2026-08-14",
            "SECURITY_ID": "10010001",
            "CONTRACT_ID": "510300P2608M04200",
            "CONTRACT_SYMBOL": "300ETF沽8月4200",
            "CONTRACT_TYPE": "认沽",
            "DELTA_VALUE": "-0.300",
            "THETA_VALUE": "-0.010",
            "GAMMA_VALUE": "0.100",
            "VEGA_VALUE": "0.050",
            "RHO_VALUE": "-0.010",
            "IMPLC_VOLATLTY": "0.000",
        },
        {
            "TRADE_DATE": "2026-08-14",
            "SECURITY_ID": "10010002",
            "CONTRACT_ID": "510050C2608M03000",
            "CONTRACT_SYMBOL": "50ETF购8月3000",
            "CONTRACT_TYPE": "认购",
            "DELTA_VALUE": "0.500",
            "THETA_VALUE": "-0.010",
            "GAMMA_VALUE": "0.100",
            "VEGA_VALUE": "0.050",
            "RHO_VALUE": "0.010",
            "IMPLC_VOLATLTY": "0.200",
        },
    ]
    result = normalize_sse_risk_indicators(records, "2026-08-16 12:00:00+08:00")
    assert result["contract_code"].tolist() == ["10010001.SH"]
    assert result.loc[0, "option_type"] == "P"
    assert result.loc[0, "implied_volatility"] == 0.0


def test_parse_cffex_month_zip_filters_date_and_io_contracts() -> None:
    csv_text = (
        "合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,"
        "今收盘,今结算,前结算,涨跌1,涨跌2,Delta\n"
        "IO2608-P-4200,10,11,9,100,10.5,200,5,10.2,10.1,10.0,0.2,0.1,-0.25\n"
        "IF2608,4200,4210,4190,1000,100,2000,10,4205,4202,4200,5,2,--\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("20260814_1.csv", csv_text.encode("gb18030"))
        archive.writestr("20260815_1.csv", csv_text.encode("gb18030"))
    result = parse_cffex_month_zip(
        buffer.getvalue(),
        start="2026-08-14",
        end="2026-08-14",
        retrieved_at="2026-08-16 12:00:00+08:00",
    )
    assert len(result) == 1
    assert result.loc[0, "contract_code"] == "IO2608-P-4200"
    assert result.loc[0, "option_type"] == "P"
    assert result.loc[0, "strike"] == 4200
    assert result.loc[0, "turnover_cny"] == pytest.approx(105000.0)
