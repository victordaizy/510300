from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.val01_raw_ey_5y_v1_forward_evaluation import (
    SIGNAL_VALUE_COLUMNS_FORBIDDEN_IN_LABELS,
    _etf_horizon_return,
    frozen_bucket,
    hac_rank_slope,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_bucket_boundaries() -> None:
    assert frozen_bucket(0.0) == "B1_EXPENSIVE"
    assert frozen_bucket(0.199999) == "B1_EXPENSIVE"
    assert frozen_bucket(0.20) == "B2"
    assert frozen_bucket(0.40) == "B3"
    assert frozen_bucket(0.60) == "B4_CHEAP"
    assert frozen_bucket(1.0) == "B4_CHEAP"


def test_etf_horizon_return_excludes_entry_day_dividend_and_includes_later_event() -> None:
    holding = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-19", "2026-01-20"]),
            "etf_open": [10.0, 10.0],
            "etf_close": [10.0, 10.0],
        }
    )
    result = _etf_horizon_return(
        holding,
        {pd.Timestamp("2026-01-19"): 1.0, pd.Timestamp("2026-01-20"): 1.0},
    )
    assert result["entry_day_dividend_excluded"] == 1.0
    assert result["dividend_event_count"] == 1
    assert result["etf_total_return"] == pytest.approx(0.10, abs=1e-15)
    assert result["etf_price_return"] == 0.0


def test_hac_rank_slope_detects_strict_positive_relation() -> None:
    x = pd.Series(np.arange(30, dtype=float))
    y = pd.Series(np.arange(30, dtype=float))
    result = hac_rank_slope(x, y, max_lags=2)
    assert result["slope"] > 0
    assert result["t_statistic"] > 0
    assert result["one_sided_p_value"] < 0.01


def test_config_disables_strategy_and_execution_layers() -> None:
    config = load_config()
    protocol = config["protocol"]
    assert protocol["forward_label_generation_enabled"] is True
    assert protocol["predictive_ic_calculation_enabled"] is True
    for name in (
        "historical_strategy_return_calculation_enabled",
        "historical_position_mapping_enabled",
        "current_position_mapping_enabled",
        "target_share_generation_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    ):
        assert protocol[name] is False


def test_generated_label_table_is_physically_separate_and_censored() -> None:
    config = load_config()
    path = ROOT / config["artifacts"]["label_table"]
    if not path.exists():
        return
    labels = pd.read_parquet(path)
    assert len(labels) == 183
    assert not SIGNAL_VALUE_COLUMNS_FORBIDDEN_IN_LABELS.intersection(labels.columns)
    expected = {60: (58, 3), 120: (55, 6), 242: (49, 12)}
    for horizon, (matured, censored) in expected.items():
        frame = labels.loc[labels["horizon_trading_days"].eq(horizon)]
        assert frame["label_status"].eq("MATURED").sum() == matured
        assert frame["label_status"].eq("RIGHT_CENSORED_DATA_CUTOFF").sum() == censored
        censored_rows = frame.loc[frame["label_status"].ne("MATURED")]
        assert censored_rows[
            ["etf_total_return", "h00300_total_return", "etf_price_return"]
        ].isna().all().all()


def test_generated_report_never_claims_alpha_or_strategy_run() -> None:
    config = load_config()
    path = ROOT / config["artifacts"]["report_json"]
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    governance = report["governance"]
    assert governance["alpha_pass"] is False
    assert governance["historical_strategy_return_calculated"] is False
    assert governance["historical_position_mapping_performed"] is False
    assert governance["current_position_mapping_performed"] is False
    assert governance["orders_generated"] is False
    assert report["status"] == "REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST"
    assert report["primary_predictive_gate"]["failed_checks"] == [
        "minimum_observations_per_bucket"
    ]
    primary_buckets = report["diagnostics"]["242"]["etf_total_return"]["buckets"]
    assert [row["observations"] for row in primary_buckets] == [1, 11, 12, 25]
