"""十五分钟技术因子时点与数据质量分层测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.intraday_technical_factors import build_intraday_technical_factors


def _bars(days: int = 12) -> pd.DataFrame:
    times = [
        "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30",
        "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00",
    ]
    records = []
    dates = pd.bdate_range("2026-07-20", periods=days)
    for day_index, day in enumerate(dates):
        for slot, time in enumerate(times, start=1):
            close = 4.0 + day_index * 0.01 + slot * 0.001
            records.append(
                {
                    "bar_end": pd.Timestamp(f"{day.date()} {time}"),
                    "trade_date": day,
                    "open": close - 0.001,
                    "high": close + 0.002,
                    "low": close - 0.003,
                    "close": close,
                    "volume": 1_000_000 + slot * 10_000 + day_index * 1_000,
                    "amount": (1_000_000 + slot * 10_000 + day_index * 1_000) * close,
                    "construction": "exchange_15m_kline",
                    "session_phase": "intraday",
                    "bar_position": "regular",
                }
            )
    return pd.DataFrame(records)


class IntradayTechnicalFactorTests(unittest.TestCase):
    def test_修改未来bar不改变过去因子(self) -> None:
        bars = _bars()
        before = build_intraday_technical_factors(bars)
        changed = bars.copy()
        changed.loc[changed.index[-1], ["close", "high", "amount", "volume"]] = [
            8.0, 8.1, 8_000_000.0, 1_000_000.0,
        ]
        after = build_intraday_technical_factors(changed)
        columns = [
            "momentum_4bar", "realized_vol_16bar", "relative_volume_z_20d",
            "session_vwap", "session_vwap_slope_3bar_bps",
            "time_of_day_rvol_median_20d", "ema_8bar", "ema_21bar", "rsi_7bar",
            "adx_14bar", "vwap_deviation_atr",
        ]
        pd.testing.assert_frame_equal(before.iloc[:-1][columns], after.iloc[:-1][columns])

    def test_fvg不跨交易日且只允许原生ohlcv(self) -> None:
        bars = _bars()
        data = build_intraday_technical_factors(bars)
        self.assertFalse(bool(data.loc[data["bar_slot"].isin([1, 2]), "fvg_factor_eligible"].any()))
        reconstructed = bars.copy()
        reconstructed.loc[reconstructed["trade_date"] == reconstructed["trade_date"].min(), "construction"] = (
            "transaction_reconstructed_15m"
        )
        result = build_intraday_technical_factors(reconstructed)
        first_day = result["trade_date"].min()
        self.assertFalse(bool(result.loc[result["trade_date"] == first_day, "fvg_factor_eligible"].any()))
        self.assertTrue(result.loc[result["trade_date"] == first_day, "candle_range"].isna().all())
        self.assertTrue(result.loc[result["trade_date"] == first_day, "close_volume_factor_eligible"].all())

    def test_同一时点成交量基准严格排除当日(self) -> None:
        bars = _bars(days=25)
        data = build_intraday_technical_factors(bars)
        target = data.loc[data["bar_slot"] == 1].iloc[20]
        historical = data.loc[data["bar_slot"] == 1, "log_volume"].iloc[:20]
        self.assertAlmostEqual(target["log_volume_slot_mean_20d"], float(historical.mean()))
        historical_volume = data.loc[data["bar_slot"] == 1, "volume"].iloc[:20]
        self.assertAlmostEqual(
            target["volume_slot_median_20d"],
            float(historical_volume.median()),
        )
        self.assertAlmostEqual(
            target["time_of_day_rvol_median_20d"],
            float(target["volume"] / historical_volume.median()),
        )
        self.assertTrue(np.isfinite(target["relative_volume_z_20d"]))

    def test_当日vwap重置且开盘区间在前两根后冻结(self) -> None:
        bars = _bars(days=25)
        data = build_intraday_technical_factors(bars)
        target_day = data.loc[data["trade_date"] == data["trade_date"].max()].copy()
        first = target_day.iloc[0]
        expected_first_vwap = first["amount"] / first["volume"]
        self.assertAlmostEqual(first["session_vwap"], expected_first_vwap)
        self.assertTrue(np.isnan(first["session_vwap_slope_3bar_bps"]))
        expected_high = float(target_day.iloc[:2]["high"].max())
        expected_low = float(target_day.iloc[:2]["low"].min())
        self.assertTrue((target_day.iloc[1:]["opening_range_high"] == expected_high).all())
        self.assertTrue((target_day.iloc[1:]["opening_range_low"] == expected_low).all())
        self.assertTrue(target_day.iloc[:2]["orb_breakout_up_bps"].isna().all())


if __name__ == "__main__":
    unittest.main()
