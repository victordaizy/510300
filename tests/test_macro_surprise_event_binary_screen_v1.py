"""宏观预期意外事件二元筛选V1的尺度、时钟与状态回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.macro_surprise_event_binary_screen_v1 import (  # noqa: E402
    CANDIDATE_CONTRACT_SHA256,
    EXPECTED_CANDIDATES,
    SERIES_IDS,
    build_candidate_states,
    build_event_features,
    load_candidate_contract,
    sha256_file,
)


def _synthetic_raw_events() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=13, freq="MS")
    prior_surprises = [-1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -0.5, 0.5, -3.0, 3.0, -2.5, 2.5]
    rows: list[dict[str, object]] = []
    for series_id in SERIES_IDS:
        current = -2.0 if series_id in {"PMI", "M2"} else 1.0
        surprises = [*prior_surprises, current]
        for date, surprise in zip(dates, surprises, strict=True):
            rows.append(
                {
                    "series_id": series_id,
                    "product": f"合成-{series_id}",
                    "release_date": date,
                    "actual": surprise,
                    "forecast": 0.0,
                    "previous": 0.0,
                    "source_function": f"synthetic_{series_id}",
                    "source": "合成测试",
                    "retrieved_at": "2026-08-28T00:00:00+08:00",
                }
            )
    return pd.DataFrame(rows)


def test_standardization_scale_uses_only_prior_events() -> None:
    """当前宏观意外不得进入自身尺度估计。"""

    raw = _synthetic_raw_events()
    events, _daily = build_event_features(
        raw,
        standardization_window=36,
        minimum_prior_events=12,
    )
    pmi = events.loc[events["series_id"].eq("PMI")].sort_values("release_date")
    expected_scale = float(
        np.std(
            [-1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -0.5, 0.5, -3.0, 3.0, -2.5, 2.5],
            ddof=0,
        )
    )

    assert pmi["prior_surprise_scale"].iloc[:-1].isna().all()
    assert np.isclose(pmi.iloc[-1]["prior_surprise_scale"], expected_scale)
    assert np.isclose(
        pmi.iloc[-1]["standardized_surprise"],
        -2.0 / expected_scale,
    )


def test_same_day_breadth_and_series_specific_triggers_are_exact() -> None:
    """同日两项负面意外应触发广度、PMI和M2规则。"""

    _events, daily = build_event_features(
        _synthetic_raw_events(),
        standardization_window=36,
        minimum_prior_events=12,
    )
    last = daily.sort_values("release_date").iloc[-1]

    assert last["negative_count"] == 2
    assert last["negative_series"] == "M2|PMI"
    assert bool(last["trigger_any_negative"])
    assert bool(last["trigger_two_negative"])
    assert bool(last["trigger_pmi_negative"])
    assert bool(last["trigger_m2_negative"])


def test_event_state_starts_strictly_after_release_date_and_holds_china_days() -> None:
    """周五公布事件不得在周五成交，应从下一个中国交易日起按交易日持有。"""

    market_dates = pd.Series(
        pd.to_datetime(
            ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
        )
    )
    daily_events = pd.DataFrame(
        {
            "release_date": pd.to_datetime(["2024-01-05"]),
            "trigger_any_negative": [True],
            "trigger_severe_negative": [False],
            "trigger_two_negative": [False],
            "trigger_composite_negative": [False],
            "trigger_pmi_negative": [False],
            "trigger_m2_negative": [False],
        }
    )

    states = build_candidate_states(market_dates, daily_events)

    assert states["MACRO_ANY_NEG_HOLD1"].tolist() == [1, 1, 0, 1, 1]
    assert states["MACRO_ANY_NEG_HOLD3"].tolist() == [1, 1, 0, 0, 0]
    assert states["MACRO_ANY_NEG_HOLD5"].tolist() == [1, 1, 0, 0, 0]
    signal_column = "MACRO_ANY_NEG_HOLD3__signal_asof_event_date"
    assert states.loc[2:, signal_column].eq(pd.Timestamp("2024-01-05")).all()
    assert bool(
        (
            states.loc[states[signal_column].notna(), signal_column]
            < states.loc[states[signal_column].notna(), "date"]
        ).all()
    )
    for candidate_id in EXPECTED_CANDIDATES:
        assert states[candidate_id].dtype == np.int8
        assert set(states[candidate_id].unique()).issubset({0, 1})


def test_candidate_contract_hash_series_and_double_twenty_gate_are_immutable() -> None:
    """事件下载前合同、序列、八规则与双20门必须保持不变。"""

    contract = load_candidate_contract()

    assert sha256_file(ROOT / "config" / "510300_macro_surprise_event_binary_screen_v1_candidates.yaml") == CANDIDATE_CONTRACT_SHA256
    assert [item["series_id"] for item in contract["event_contract"]["series"]] == SERIES_IDS
    assert [item["candidate_id"] for item in contract["candidates"]] == EXPECTED_CANDIDATES
    assert contract["objective"]["minimum_annualized_excess"] == 0.20
    assert contract["objective"]["minimum_rolling_excess_median"] == 0.20
    assert contract["scope"]["allowed_target_states"] == [0, 1]
