"""IF 与 Breadth 信息传播 Alpha 的无未来函数测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.information_propagation_alpha import (
    LeadLagParameters,
    build_no_if_information_propagation_features,
    build_information_propagation_features,
    build_intraday_breadth,
    fit_quantile_edges,
    select_causal_futures_contract,
    summarize_signal_deciles,
)
from research.information_propagation_timestamp_audit import audit_timestamp_alignment


class InformationPropagationFeatureTests(unittest.TestCase):
    def test_无IF实验以指数领先解释510300收益(self) -> None:
        times = pd.date_range("2026-01-05 09:30", periods=4, freq="min")
        etf = pd.DataFrame(
            {"trade_time": times, "open": [10.0, 10.1, 10.2, 10.3], "close": [10.0, 10.1, 10.2, 10.3]}
        )
        index = pd.DataFrame(
            {"trade_time": times, "close": [100.0, 102.0, 103.0, 104.0]}
        )
        breadth = pd.DataFrame(
            {
                "trade_time": times,
                "leader_return": [np.nan, 0.015, 0.01, 0.01],
                "top50_breadth": [0.5, 0.6, 0.7, 0.8],
                "top50_breadth_impulse_1m": [np.nan, 0.1, 0.1, 0.1],
            }
        )
        parameters = LeadLagParameters(
            lookback_minutes=1,
            breadth_impulse_minutes=1,
            horizons_minutes=(1,),
            normalization_sessions=2,
            normalization_minimum_sessions=2,
        )

        result = build_no_if_information_propagation_features(
            etf, index, breadth, parameters
        )
        row = result.loc[result["trade_time"] == times[1]].iloc[0]

        expected = 102 / 100 - 10.1 / 10.0
        self.assertAlmostEqual(float(row["index_lead_etf_1m"]), expected)
        self.assertIn("etf_execution_next_open_1m", result.columns)

    def test_精确分钟收益不跨午休(self) -> None:
        times = list(pd.date_range("2026-01-05 09:30", periods=8, freq="min"))
        times += list(pd.date_range("2026-01-05 13:00", periods=8, freq="min"))
        etf = pd.DataFrame({"trade_time": times, "close": np.linspace(4.0, 4.15, len(times))})
        futures = pd.DataFrame({"trade_time": times, "close": np.linspace(4000, 4015, len(times))})
        component_rows = []
        for symbol, scale in [("A.SH", 1.0), ("B.SH", 1.1)]:
            for index, timestamp in enumerate(times):
                component_rows.append(
                    {"trade_time": timestamp, "con_code": symbol, "close": scale + index * 0.001}
                )
        components = pd.DataFrame(component_rows)
        weights = pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2025-12-31")] * 2,
                "con_code": ["A.SH", "B.SH"],
                "weight": [60.0, 40.0],
            }
        )
        parameters = LeadLagParameters(
            lookback_minutes=3,
            horizons_minutes=(3,),
            leader_count=2,
            normalization_sessions=2,
            normalization_minimum_sessions=2,
        )

        result = build_information_propagation_features(etf, futures, components, weights, parameters)

        afternoon_open = result.loc[result["trade_time"] == pd.Timestamp("2026-01-05 13:00")].iloc[0]
        morning_near_close = result.loc[result["trade_time"] == pd.Timestamp("2026-01-05 09:36")].iloc[0]
        self.assertTrue(pd.isna(afternoon_open["etf_return"]))
        self.assertTrue(pd.isna(morning_near_close["etf_forward_3m"]))

    def test_历史权重严格使用交易日前快照(self) -> None:
        dates = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
        rows = []
        for day in dates:
            for minute in pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=5, freq="min"):
                rows.extend(
                    [
                        {"trade_time": minute, "con_code": "A.SH", "close": 10.0 + minute.minute * 0.01},
                        {"trade_time": minute, "con_code": "B.SH", "close": 20.0 + minute.minute * 0.01},
                    ]
                )
        weights = pd.DataFrame(
            {
                "trade_date": ["2025-12-31", "2025-12-31", "2026-01-05", "2026-01-05"],
                "con_code": ["A.SH", "B.SH", "A.SH", "B.SH"],
                "weight": [90.0, 10.0, 10.0, 90.0],
            }
        )
        parameters = LeadLagParameters(
            lookback_minutes=1,
            horizons_minutes=(1,),
            leader_count=2,
            normalization_sessions=2,
            normalization_minimum_sessions=2,
        )

        breadth = build_intraday_breadth(pd.DataFrame(rows), weights, parameters)

        day_one_snapshot = breadth.loc[breadth["session_date"] == dates[0], "weight_snapshot_date"].dropna().unique()
        day_two_snapshot = breadth.loc[breadth["session_date"] == dates[1], "weight_snapshot_date"].dropna().unique()
        self.assertEqual(day_one_snapshot.tolist(), [np.datetime64("2025-12-31")])
        self.assertEqual(day_two_snapshot.tolist(), [np.datetime64("2026-01-05")])

    def test_IF合约由前一交易日成交量选择(self) -> None:
        rows = []
        for day, volumes in [
            ("2026-01-05", {"IF2601": 100, "IF2602": 10}),
            ("2026-01-06", {"IF2601": 20, "IF2602": 200}),
            ("2026-01-07", {"IF2601": 5, "IF2602": 300}),
        ]:
            for contract, volume in volumes.items():
                rows.append(
                    {
                        "trade_time": pd.Timestamp(f"{day} 09:30"),
                        "contract": contract,
                        "close": 4000.0,
                        "volume": volume,
                    }
                )

        selected = select_causal_futures_contract(pd.DataFrame(rows))

        selection = selected.set_index("session_date")["contract"].to_dict()
        self.assertNotIn(pd.Timestamp("2026-01-05"), selection)
        self.assertEqual(selection[pd.Timestamp("2026-01-06")], "IF2601")
        self.assertEqual(selection[pd.Timestamp("2026-01-07")], "IF2602")

    def test_Top50等权宽度与权重宽度严格区分(self) -> None:
        times = pd.date_range("2026-01-05 09:30", periods=2, freq="min")
        components = pd.DataFrame(
            [
                {"trade_time": times[0], "con_code": "A.SH", "close": 10.0},
                {"trade_time": times[1], "con_code": "A.SH", "close": 9.0},
                {"trade_time": times[0], "con_code": "B.SH", "close": 10.0},
                {"trade_time": times[1], "con_code": "B.SH", "close": 11.0},
            ]
        )
        weights = pd.DataFrame(
            {
                "trade_date": ["2025-12-31", "2025-12-31"],
                "con_code": ["A.SH", "B.SH"],
                "weight": [90.0, 10.0],
            }
        )
        parameters = LeadLagParameters(
            lookback_minutes=1,
            leader_count=2,
            normalization_sessions=2,
            normalization_minimum_sessions=2,
        )

        breadth = build_intraday_breadth(components, weights, parameters).iloc[-1]

        self.assertAlmostEqual(float(breadth["top50_breadth"]), 0.5)
        self.assertAlmostEqual(float(breadth["top50_weighted_breadth"]), 0.1)

    def test_Top50宽度冲量使用精确五分钟(self) -> None:
        times = pd.date_range("2026-01-05 09:30", periods=7, freq="min")
        prices = [10, 11, 12, 13, 14, 15, 14]
        components = pd.DataFrame(
            [
                {"trade_time": timestamp, "con_code": symbol, "close": price}
                for symbol in ["A.SH", "B.SH"]
                for timestamp, price in zip(times, prices)
            ]
        )
        weights = pd.DataFrame(
            {
                "trade_date": ["2025-12-31", "2025-12-31"],
                "con_code": ["A.SH", "B.SH"],
                "weight": [50.0, 50.0],
            }
        )
        parameters = LeadLagParameters(
            lookback_minutes=1,
            breadth_impulse_minutes=5,
            leader_count=2,
            normalization_sessions=2,
            normalization_minimum_sessions=2,
        )

        breadth = build_intraday_breadth(components, weights, parameters)
        last = breadth.loc[breadth["trade_time"] == times[-1]].iloc[0]

        self.assertAlmostEqual(float(last["top50_breadth_impulse_5m"]), -1.0)

    def test_执行收益从下一分钟开盘开始(self) -> None:
        times = pd.date_range("2026-01-05 09:30", periods=8, freq="min")
        etf = pd.DataFrame(
            {
                "trade_time": times,
                "open": [10, 11, 12, 13, 20, 30, 40, 50],
                "close": [10, 11, 12, 13, 21, 33, 44, 55],
            }
        )
        futures = pd.DataFrame({"trade_time": times, "close": np.arange(100, 108)})
        components = pd.DataFrame(
            [
                {"trade_time": timestamp, "con_code": symbol, "close": 10 + index}
                for symbol in ["A.SH", "B.SH"]
                for index, timestamp in enumerate(times)
            ]
        )
        weights = pd.DataFrame(
            {
                "trade_date": ["2025-12-31", "2025-12-31"],
                "con_code": ["A.SH", "B.SH"],
                "weight": [50.0, 50.0],
            }
        )
        parameters = LeadLagParameters(
            lookback_minutes=1,
            horizons_minutes=(1,),
            leader_count=2,
            normalization_sessions=2,
            normalization_minimum_sessions=2,
        )

        result = build_information_propagation_features(
            etf, futures, components, weights, parameters
        )
        row = result.loc[result["trade_time"] == times[3]].iloc[0]

        self.assertAlmostEqual(float(row["etf_execution_next_open_1m"]), 33 / 20 - 1)


class InformationPropagationSummaryTests(unittest.TestCase):
    def test_分位边界只由开发期拟合且能识别单调收益(self) -> None:
        timestamps = pd.date_range("2022-01-01", periods=120, freq="D")
        signal = np.linspace(-3.0, 3.0, len(timestamps))
        features = pd.DataFrame(
            {
                "trade_time": timestamps,
                "alpha_score": signal,
                "etf_forward_5m": signal * 0.0001,
            }
        )
        evaluation_periods = {"development": ("2022-01-01", "2022-04-30")}

        deciles, yearly = summarize_signal_deciles(
            features,
            signal_column="alpha_score",
            horizons_minutes=(5,),
            development_start="2022-01-01",
            development_end="2022-04-30",
            evaluation_periods=evaluation_periods,
            quantile_count=10,
        )

        means = deciles.sort_values("signal_bucket")["mean_return_bps"].to_numpy()
        self.assertTrue(np.all(np.diff(means) > 0))
        self.assertGreater(float(yearly.iloc[0]["high_minus_low_bps"]), 0.0)

    def test_重复值过多时拒绝伪造十组(self) -> None:
        signal = pd.Series([0.0] * 100)
        with self.assertRaises(ValueError):
            fit_quantile_edges(signal, 10)


class TimestampAuditTests(unittest.TestCase):
    @staticmethod
    def _frames(shift_index_minutes: int = 0) -> tuple[pd.DataFrame, ...]:
        times = list(pd.date_range("2026-01-05 09:30", "2026-01-05 11:30", freq="min"))
        times += list(pd.date_range("2026-01-05 13:01", "2026-01-05 15:00", freq="min"))
        etf = pd.DataFrame({"trade_time": times})
        index = pd.DataFrame(
            {"trade_time": [value + pd.Timedelta(minutes=shift_index_minutes) for value in times]}
        )
        futures = pd.DataFrame({"trade_time": times})
        components = pd.DataFrame(
            [
                {"trade_time": timestamp, "con_code": symbol}
                for symbol in ["A.SH", "B.SH"]
                for timestamp in times
            ]
        )
        return etf, index, futures, components

    def test_时间戳完全对齐且语义确认后通过(self) -> None:
        etf, index, futures, components = self._frames()
        result = audit_timestamp_alignment(
            etf, index, futures, components,
            sample_trading_days=1,
            manual_semantics_status="CONFIRMED",
        )
        self.assertEqual(result["status"], "PASS")

    def test_指数错位一分钟时硬失败(self) -> None:
        etf, index, futures, components = self._frames(shift_index_minutes=1)
        result = audit_timestamp_alignment(
            etf, index, futures, components,
            sample_trading_days=1,
            manual_semantics_status="CONFIRMED",
        )
        self.assertEqual(result["status"], "FAIL_TIMESTAMP_ALIGNMENT")


if __name__ == "__main__":
    unittest.main()
