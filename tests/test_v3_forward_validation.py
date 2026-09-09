from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from research.v3_forward_validation import (
    HORIZONS,
    ZERO_HASH,
    _monitor_status,
    append_signal_record,
    record_hash,
    resolve_forward_outcomes,
    shadow_position_for_edge,
    verify_signal_hash_chain,
)


def _signal_record(date: str, edge: float) -> dict[str, object]:
    record: dict[str, object] = {
        "signal_date": pd.Timestamp(date),
        "model_version": "V3_FORWARD_1",
        "data_available_time": f"{date}T16:30:00+08:00",
        "earnings_state": "盈利上行",
        "market_state": "熊市",
        "vol_state": "低波动",
        "shadow_target_position": 0.75,
    }
    for horizon in HORIZONS:
        record[f"edge_{horizon}"] = edge
        record[f"cash_return_{horizon}d"] = 0.01 * horizon / 242
    return record


def test_signal_log_is_append_only_and_hash_chained() -> None:
    data, action = append_signal_record(pd.DataFrame(), _signal_record("2026-08-13", 0.06))
    assert action == "APPENDED"
    assert data.iloc[0]["previous_record_hash"] == ZERO_HASH
    assert data.iloc[0]["record_hash"] == record_hash(data.iloc[0].to_dict())

    data, action = append_signal_record(data, _signal_record("2026-08-14", 0.07))
    assert action == "APPENDED"
    assert data.iloc[1]["previous_record_hash"] == data.iloc[0]["record_hash"]
    verify_signal_hash_chain(data)

    with pytest.raises(ValueError, match="回填"):
        append_signal_record(data, _signal_record("2026-08-13", 0.06))


def test_signal_log_rejects_mutation_and_accepts_identical_retry() -> None:
    record = _signal_record("2026-08-13", 0.06)
    data, _ = append_signal_record(pd.DataFrame(), record)
    retried, action = append_signal_record(data, record)
    assert action == "IDEMPOTENT"
    pd.testing.assert_frame_equal(data, retried)

    tampered = data.copy()
    tampered.loc[0, "edge_60"] = 0.99
    with pytest.raises(ValueError, match="内容哈希"):
        verify_signal_hash_chain(tampered)

    changed = dict(record)
    changed["edge_60"] = 0.08
    with pytest.raises(ValueError, match="禁止覆盖"):
        append_signal_record(data, changed)


@pytest.mark.parametrize(
    ("edge", "expected"),
    [
        (-0.10, 0.00),
        (-0.099, 0.25),
        (-0.05, 0.25),
        (0.00, 0.50),
        (0.05, 0.50),
        (0.051, 0.75),
        (0.10, 0.75),
        (0.101, 1.00),
    ],
)
def test_shadow_position_boundaries(edge: float, expected: float) -> None:
    mapping = [
        {"maximum_edge": -0.10, "target_position": 0.00},
        {"maximum_edge": -0.05, "target_position": 0.25},
        {"maximum_edge": 0.05, "target_position": 0.50},
        {"maximum_edge": 0.10, "target_position": 0.75},
        {"maximum_edge": None, "target_position": 1.00},
    ]
    assert shadow_position_for_edge(edge, mapping) == expected


def test_outcome_only_resolves_after_horizon_and_never_overwrites() -> None:
    signals, _ = append_signal_record(pd.DataFrame(), _signal_record("2026-08-13", 0.06))
    dates = pd.bdate_range("2026-08-13", periods=243)
    total = pd.DataFrame({"date": dates, "close": 100.0 * np.power(1.001, np.arange(243))})
    now = datetime(2026, 9, 1, 18, tzinfo=ZoneInfo("Asia/Shanghai"))

    too_early, count = resolve_forward_outcomes(
        signals, pd.DataFrame(), total.iloc[:20], now
    )
    assert count == 0
    assert pd.isna(too_early.loc[0, "future_return_20"])

    mature, count = resolve_forward_outcomes(signals, too_early, total.iloc[:21], now)
    assert count == 1
    locked_return = mature.loc[0, "future_return_20"]
    assert locked_return == pytest.approx(1.001**20 - 1)
    assert pd.notna(mature.loc[0, "resolution_hash_20"])

    revised = total.iloc[:21].copy()
    revised.loc[20, "close"] = 1.0
    rerun, count = resolve_forward_outcomes(signals, mature, revised, now)
    assert count == 0
    assert rerun.loc[0, "future_return_20"] == locked_return


def test_monitor_is_insufficient_before_preregistered_minimum() -> None:
    evaluation = {
        "realized_observations": 119,
        "exact_non_overlapping": {"queue_observations": [4]},
    }
    config = {
        "monitoring": {
            "minimum_realized_60d_signals_for_decision": 120,
            "minimum_exact_non_overlapping_60d_observations": 4,
        }
    }
    status, detail = _monitor_status(evaluation, config)
    assert status == "INSUFFICIENT_DATA"
    assert detail["sample_sufficient"] is False
