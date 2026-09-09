"""点时估值价格补采及V1.1审计测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.point_in_time_valuation_price_acquisition import (
    ROOT,
    cache_covers_request,
    load_config,
    normalize_daily,
    required_missing_symbols,
    sha256_file,
    validate_snapshot_prices,
)
from research.point_in_time_valuation_v2_post_acquisition import (
    load_config as load_v1_1_config,
    verify_hashes,
)


def test_供应商日线保持未复权价格并转换成交单位() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20160701"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "vol": [123.0],
            "amount": [456.0],
        }
    )
    result = normalize_daily(
        raw,
        "000001.SZ",
        pd.Timestamp("2016-07-01"),
        pd.Timestamp("2016-07-31"),
        pd.Timestamp("2026-08-18 12:00:00+08:00"),
    )
    assert result.loc[0, "raw_close"] == 10.2
    assert result.loc[0, "volume"] == 12_300.0
    assert result.loc[0, "amount_cny"] == 456_000.0
    assert result.loc[0, "source"] == "tushare_proxy.daily"


def test_缓存必须明确覆盖请求起止区间() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-07-01"]),
            "con_code": ["A"],
            "raw_close": [1.0],
            "request_start": pd.to_datetime(["2016-07-01"]),
            "request_end": pd.to_datetime(["2019-12-31"]),
        }
    )
    assert cache_covers_request(data, "A", pd.Timestamp("2016-07-01"), pd.Timestamp("2019-12-31"))
    assert not cache_covers_request(data, "A", pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31"))


def test_补采范围固定为437只和40个月() -> None:
    config = load_config()
    weights = pd.read_parquet(ROOT / config["data_contracts"]["historical_weights"]["file"])
    prior = pd.read_parquet(ROOT / config["data_contracts"]["prior_snapshot_audit"]["file"])
    symbols, dates = required_missing_symbols(weights, prior, config)
    assert len(symbols) == 437
    assert len(dates) == 40
    assert dates[0] == pd.Timestamp("2016-08-31")
    assert dates[-1] == pd.Timestamp("2019-11-29")


def test_正式快照价格表通过覆盖和无未来价格闸门() -> None:
    config = load_config()
    path = ROOT / config["artifacts"]["snapshot_prices"]
    data = pd.read_parquet(path)
    evidence = validate_snapshot_prices(data, config)
    assert evidence["status"] == "PASS"
    assert evidence["snapshot_count"] == 60
    assert evidence["row_count"] == 18_000
    assert evidence["minimum_constituents"] == 300
    assert evidence["minimum_price_weight_coverage"] >= 0.99
    assert evidence["missing_price_row_count"] == 0
    assert evidence["future_price_row_count"] == 0
    assert (pd.to_datetime(data["price_trade_date"]) <= pd.to_datetime(data["date"])).all()


def test_价格补采报告记录独立核验和6只跨起点停牌证券() -> None:
    config = load_config()
    report = json.loads(
        (ROOT / config["artifacts"]["report_json"]).read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS_FIVE_YEAR_SNAPSHOT_PRICES"
    assert report["supplemental_prehistory_symbol_count"] == 6
    assert report["cross_source_overlap"]["overlap_row_count"] == 1998
    assert report["cross_source_overlap"]["exact_match_ratio"] == 1.0
    assert report["cross_source_overlap"]["maximum_absolute_close_difference"] == 0.0
    assert report["token_persisted_in_outputs"] is False
    assert report["output_sha256"] == sha256_file(
        ROOT / config["artifacts"]["snapshot_prices"]
    )


def test_V1_1输入哈希和分支状态精确() -> None:
    config = load_v1_1_config()
    assert verify_hashes(config)["status"] == "PASS"
    report = json.loads(
        (ROOT / config["artifacts"]["report_json"]).read_text(encoding="utf-8")
    )
    five = report["history_gates"]["five_year"]
    seven = report["history_gates"]["seven_year"]
    assert report["branch_status"]["VAL01_RAW_EY_5Y"] == "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY"
    assert five["price_coverage_failed_months"] == []
    assert five["ttm_coverage_failed_months"] == []
    assert five["minimum_local_coverages"]["price_weight_coverage"] >= 0.99
    assert five["minimum_local_coverages"]["ttm_metric_weight_coverage"] == pytest.approx(0.98573)
    assert five["normalized_coverage_failed_months"] == [
        str(period) for period in pd.period_range("2016-08", "2017-04", freq="M")
    ]
    assert five["cgb_10y_exact_date_failed_months"] == []
    assert len(seven["missing_weight_months"]) == 24
    assert report["governance"]["return_calculation_performed"] is False
    assert report["governance"]["position_mapping_performed"] is False

