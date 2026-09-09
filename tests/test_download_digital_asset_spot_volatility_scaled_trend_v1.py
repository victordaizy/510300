"""数字资产Coinbase日线采集器的合成测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.download_digital_asset_spot_volatility_scaled_trend_v1 import (
    DigitalAssetDataError,
    load_contract,
    normalize_candles,
)


def _timestamp(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def test_contract_freezes_two_spot_products_and_disables_strategy_acquisition() -> None:
    contract = load_contract()
    assert contract["universe"]["fixed_products"] == ["BTC-USD", "ETH-USD"]
    assert contract["information_timing"]["candle_granularity_seconds"] == 86400
    assert contract["data_sources"]["strategy_return_or_rank_may_be_computed_during_acquisition"] is False


def test_normalize_coinbase_candles_sorts_schema_and_dollar_turnover() -> None:
    rows = [
        [_timestamp("2020-01-02"), 90.0, 110.0, 100.0, 105.0, 1000.0],
        [_timestamp("2020-01-01"), 80.0, 105.0, 90.0, 100.0, 2000.0],
    ]
    frame = normalize_candles(
        "BTC-USD",
        rows,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-02"),
    )
    assert frame["date"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    assert frame["dollar_turnover_usd"].tolist() == pytest.approx([200000.0, 105000.0])


def test_missing_utc_daily_bucket_is_failure_not_forward_fill() -> None:
    rows = [
        [_timestamp("2020-01-01"), 80.0, 105.0, 90.0, 100.0, 2000.0],
        [_timestamp("2020-01-03"), 90.0, 110.0, 100.0, 105.0, 1000.0],
    ]
    with pytest.raises(DigitalAssetDataError, match="缺少1个UTC日线桶"):
        normalize_candles(
            "BTC-USD",
            rows,
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-01-03"),
        )


def test_nonpositive_volume_is_rejected() -> None:
    rows = [[_timestamp("2020-01-01"), 80.0, 105.0, 90.0, 100.0, 0.0]]
    with pytest.raises(DigitalAssetDataError, match="非正成交量"):
        normalize_candles(
            "ETH-USD",
            rows,
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-01-01"),
        )
