"""510300分钟行情接口与质量审计测试。"""

from __future__ import annotations

import json
import unittest

import pandas as pd

from market_data.minute import (
    MinuteDataRequest,
    SinaMinuteProvider,
    TencentIntradaySnapshotProvider,
    TencentMinuteProvider,
    aggregate_one_minute_to_15m,
)
from scripts.quality_check_510300_15m import check_minute_data, cross_check_daily


EXPECTED_TIMES = [
    "09:45:00", "10:00:00", "10:15:00", "10:30:00",
    "10:45:00", "11:00:00", "11:15:00", "11:30:00",
    "13:15:00", "13:30:00", "13:45:00", "14:00:00",
    "14:15:00", "14:30:00", "14:45:00", "15:00:00",
]


class _FakeResponse:
    def __init__(self, records: list[dict]) -> None:
        self.text = f"/*prefix*/=({json.dumps(records)});"

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.last_params: dict | None = None

    def get(self, url: str, params: dict, timeout: float) -> _FakeResponse:
        self.last_params = params
        return _FakeResponse(self.records)


class _FakeTencentResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeTencentSession:
    def get(self, url: str, params: dict, headers: dict, timeout: float) -> _FakeTencentResponse:
        payload = {
            "code": 0,
            "msg": "",
            "data": {
                "sh510300": {
                    "m15": [
                        ["202608120945", "4.727", "4.738", "4.739", "4.720", "642843.0", {}, "1.23"],
                        ["202608121500", "4.755", "4.748", "4.756", "4.748", "1757647.0", {}, "2.34"],
                    ]
                }
            },
        }
        return _FakeTencentResponse(payload)


class _FakeTencentIntradaySession:
    def get(self, url: str, params: dict, headers: dict, timeout: float) -> _FakeTencentResponse:
        payload = {
            "code": 0,
            "msg": "",
            "data": {
                "sh510300": {
                    "data": {
                        "date": "20260812",
                        "data": [
                            "1500 4.748 9010137 4275919420.94",
                            "1506 4.748 9011593 4276610729.74",
                            "1530 4.748 9014177 4277837612.94",
                        ],
                    }
                }
            },
        }
        return _FakeTencentResponse(payload)


def _records_for_day(date: str) -> list[dict]:
    output = []
    for index, time in enumerate(EXPECTED_TIMES):
        price = 4.000 + index * 0.001
        output.append(
            {
                "day": f"{date} {time}",
                "open": f"{price:.3f}",
                "high": f"{price + 0.002:.3f}",
                "low": f"{price - 0.002:.3f}",
                "close": f"{price + 0.001:.3f}",
                "volume": "1000",
                "amount": f"{1000 * (price + 0.0005):.4f}",
            }
        )
    return output


def _config() -> dict:
    return {
        "symbols": {"etf": "510300.SH"},
        "session": {
            "expected_bar_end_times": EXPECTED_TIMES,
            "expected_bars_per_full_day": 16,
        },
        "quality": {
            "price_tick": 0.001,
            "price_tolerance_ticks": 1,
            "volume_relative_tolerance": 0.01,
            "amount_relative_tolerance": 0.01,
        },
        "market_regime": {"close_auction_and_after_hours_start": "2026-07-06"},
    }


class MinuteProviderTests(unittest.TestCase):
    def test_新浪接口标准化为bar_end语义(self) -> None:
        session = _FakeSession(_records_for_day("2026-08-11"))
        provider = SinaMinuteProvider(session=session)
        data = provider.fetch(MinuteDataRequest(symbol="510300.SH", frequency_minutes=15))
        self.assertEqual(len(data), 16)
        self.assertEqual(data.iloc[0]["bar_end"], pd.Timestamp("2026-08-11 09:45:00"))
        self.assertEqual(data.iloc[0]["bar_start"], pd.Timestamp("2026-08-11 09:30:00"))
        self.assertEqual(data.iloc[-1]["bar_end"], pd.Timestamp("2026-08-11 15:00:00"))
        self.assertEqual(data["timestamp_meaning"].unique().tolist(), ["bar_end"])
        self.assertEqual(session.last_params["datalen"], "1970")
        self.assertEqual(data.iloc[0]["session_phase"], "opening_auction_mixed")
        self.assertEqual(data.iloc[-1]["session_phase"], "closing_auction_mixed")
        self.assertFalse(bool(data["post_close_included"].any()))

    def test_复权分钟请求被拒绝(self) -> None:
        request = MinuteDataRequest(symbol="510300.SH", adjustment="qfq")
        with self.assertRaises(ValueError):
            request.validate()

    def test_腾讯第二来源将手转换为份且不伪造成交额(self) -> None:
        provider = TencentMinuteProvider(session=_FakeTencentSession())
        data = provider.fetch(
            MinuteDataRequest(symbol="510300.SH", frequency_minutes=15, maximum_bars=320)
        )
        self.assertEqual(data.iloc[0]["volume"], 64_284_300.0)
        self.assertTrue(pd.isna(data.iloc[0]["amount"]))
        self.assertEqual(data.iloc[0]["amount_unit"], "UNAVAILABLE")
        self.assertEqual(data.iloc[-1]["bar_end"], pd.Timestamp("2026-08-12 15:00:00"))

    def test_腾讯当日快照能分离盘后固定价格成交(self) -> None:
        provider = TencentIntradaySnapshotProvider(session=_FakeTencentIntradaySession())
        data = provider.fetch(
            MinuteDataRequest(symbol="510300.SH", frequency_minutes=1, maximum_bars=1)
        )
        self.assertEqual(data.iloc[0]["session_phase"], "closing_auction")
        self.assertEqual(data.iloc[-1]["session_phase"], "post_close_fixed_price")
        self.assertEqual(data.iloc[-1]["cumulative_volume"], 901_417_700.0)
        self.assertEqual(data.iloc[-1]["incremental_volume"], 258_400.0)

    def test_一分反聚合采用bar_end且最后一根纳入集合竞价记录(self) -> None:
        times = list(pd.date_range("2026-08-12 09:31", "2026-08-12 10:00", freq="1min"))
        times += list(pd.date_range("2026-08-12 14:46", "2026-08-12 14:57", freq="1min"))
        times += [pd.Timestamp("2026-08-12 15:00")]
        values = pd.Series(range(len(times)), dtype=float)
        one_minute = pd.DataFrame(
            {
                "bar_end": times,
                "open": values,
                "high": values + 0.5,
                "low": values - 0.5,
                "close": values + 0.25,
                "volume": 100.0,
                "amount": 400.0,
            }
        )
        aggregated = aggregate_one_minute_to_15m(one_minute, EXPECTED_TIMES)
        ten = aggregated.loc[aggregated["bar_end"] == pd.Timestamp("2026-08-12 10:00")].iloc[0]
        close = aggregated.loc[aggregated["bar_end"] == pd.Timestamp("2026-08-12 15:00")].iloc[0]
        self.assertEqual(ten["source_minute_count"], 15)
        self.assertEqual(ten["open"], 15.0)
        self.assertEqual(close["source_minute_count"], 13)
        self.assertEqual(close["close"], values.iloc[-1] + 0.25)


class MinuteQualityTests(unittest.TestCase):
    def _standard_data(self, dates: list[str]) -> pd.DataFrame:
        session = _FakeSession(sum((_records_for_day(date) for date in dates), []))
        return SinaMinuteProvider(session=session).fetch(MinuteDataRequest(symbol="510300.SH"))

    def test_完整交易日通过基础质量检查(self) -> None:
        data = self._standard_data(["2026-08-10", "2026-08-11"])
        errors, warnings, evidence = check_minute_data(data, _config())
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(evidence["full_trading_days"], 2)

    def test_内部交易日缺一根必须失败(self) -> None:
        data = self._standard_data(["2026-08-07", "2026-08-10", "2026-08-11"])
        missing_time = pd.Timestamp("2026-08-10 10:15:00")
        data = data.loc[data["bar_end"] != missing_time].reset_index(drop=True)
        errors, _, _ = check_minute_data(data, _config())
        self.assertIn("INTERNAL_PARTIAL_TRADING_DAY", {item["code"] for item in errors})

    def test_分钟聚合与日线逐字段交叉检查(self) -> None:
        data = self._standard_data(["2026-08-11"])
        daily = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-08-11")],
                "open": [data.iloc[0]["open"]],
                "high": [data["high"].max()],
                "low": [data["low"].min()],
                "close": [data.iloc[-1]["close"]],
                "volume": [data["volume"].sum()],
                "amount": [data["amount"].sum()],
            }
        )
        errors, warnings, evidence = cross_check_daily(data, daily, _config())
        self.assertEqual(errors, [])
        self.assertEqual({item["code"] for item in warnings}, {"POST_CLOSE_DATA_MISSING"})
        self.assertEqual(evidence["matched_full_days"], 1)

    def test_旧制日线收盘与末笔价差异只警告(self) -> None:
        data = self._standard_data(["2026-06-30"])
        daily = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-06-30")],
                "open": [data.iloc[0]["open"]],
                "high": [data["high"].max()],
                "low": [data["low"].min()],
                "close": [data.iloc[-1]["close"] + 0.003],
                "volume": [data["volume"].sum()],
                "amount": [data["amount"].sum()],
            }
        )
        errors, warnings, _ = cross_check_daily(data, daily, _config())
        self.assertEqual(errors, [])
        self.assertIn("LEGACY_CLOSE_DEFINITION_DIFFERENCE", {item["code"] for item in warnings})

    def test_新制日线总量高于十五分钟时标记盘后代理缺口(self) -> None:
        data = self._standard_data(["2026-08-11"])
        daily = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-08-11")],
                "open": [data.iloc[0]["open"]],
                "high": [data["high"].max()],
                "low": [data["low"].min()],
                "close": [data.iloc[-1]["close"]],
                "volume": [data["volume"].sum() + 100],
                "amount": [data["amount"].sum() + 400],
            }
        )
        errors, warnings, evidence = cross_check_daily(data, daily, _config())
        warning_codes = {item["code"] for item in warnings}
        post_stats = evidence["daily_minus_15m_gap_by_market_regime"]["from_2026_07_06"]
        self.assertEqual(errors, [])
        self.assertIn("POST_CLOSE_VOLUME_PROXY_POSITIVE", warning_codes)
        self.assertEqual(post_stats["volume_gap_shares"]["positive_days"], 1)
        self.assertEqual(post_stats["volume_gap_shares"]["total"], 100.0)


if __name__ == "__main__":
    unittest.main()
