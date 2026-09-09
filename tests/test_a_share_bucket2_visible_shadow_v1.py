"""桶2固定低波公式可见Shadow测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.a_share_bucket2_visible_shadow_v1 import (
    extend_total_return_history,
    is_scheduled_signal_date,
    select_signal,
    target_frame,
)
from scripts.run_a_share_bucket2_visible_shadow_v1 import (
    TushareFailoverApi,
    _retained_state_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "a_share_bucket2_visible_shadow_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "a_share_bucket2_visible_shadow_v1_manifest.json"


def _config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_visible_shadow_is_paper_only_and_permanently_not_blind() -> None:
    config = _config()
    assert config["protocol"]["evidence_grade"] == "VISIBLE_SHADOW_NOT_BLIND"
    assert config["governance"]["strict_blind_claim_allowed"] is False
    assert config["governance"]["visible_shadow_target_generation_enabled"] is True
    assert config["governance"]["simulated_account_mapping_enabled"] is True
    assert config["governance"]["real_position_mapping_enabled"] is False
    assert config["governance"]["order_file_generation_enabled"] is False
    assert config["governance"]["broker_connection_enabled"] is False
    assert config["governance"]["live_trading_enabled"] is False
    assert config["data"]["exchanges"] == ["SSE", "SZSE"]


def test_visible_shadow_implementation_matches_post_result_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["manifest_type"] == (
        "VISIBLE_SHADOW_IMPLEMENTATION_MANIFEST_AFTER_FIRST_VISIBLE_RESULT"
    )
    assert manifest["strict_blind_claim_allowed"] is False
    for relative, expected_hash in manifest["frozen_implementation_files"].items():
        assert _sha256(ROOT / relative) == expected_hash


def test_total_return_extension_preserves_corporate_action_identity() -> None:
    prior = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-14"),
                "con_code": "A.SZ",
                "raw_open": 10.0,
                "raw_high": 10.2,
                "raw_low": 9.9,
                "raw_close": 10.0,
                "pre_close": 9.8,
                "total_return_open": 20.0,
                "total_return_high": 20.4,
                "total_return_low": 19.8,
                "total_return_close": 20.0,
                "volume": 1000000.0,
                "amount": 1000000000.0,
            }
        ]
    )
    increment = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-17"),
                "con_code": "A.SZ",
                "pre_close": 5.0,
                "raw_open": 5.1,
                "raw_high": 5.3,
                "raw_low": 5.0,
                "raw_close": 5.2,
                "volume": 1000000.0,
                "amount": 1000000000.0,
            }
        ]
    )
    extended = extend_total_return_history(prior, increment, 1e-8)
    latest = extended.sort_values("date").iloc[-1]
    assert np.isclose(latest["linked_adjustment_factor"], 4.0)
    assert np.isclose(latest["total_return_close"], 20.8)
    assert latest["return_identity_difference"] <= 1e-8


def test_signal_schedule_is_fixed_every_60_trading_days() -> None:
    calendar = pd.bdate_range("2026-08-19", periods=121)
    assert is_scheduled_signal_date(calendar, calendar[0], calendar[0], 60)
    assert not is_scheduled_signal_date(calendar, calendar[0], calendar[59], 60)
    assert is_scheduled_signal_date(calendar, calendar[0], calendar[60], 60)
    assert is_scheduled_signal_date(calendar, calendar[0], calendar[120], 60)


def test_lowest_twenty_day_volatility_is_selected_with_legacy_eligibility() -> None:
    dates = pd.bdate_range("2026-01-02", periods=130)
    rows = []
    for code, amplitude in (("LOW.SZ", 0.001), ("HIGH.SZ", 0.02)):
        log_returns = 0.0003 + amplitude * np.sin(np.arange(len(dates)) * 0.7)
        total_close = 20.0 * np.exp(np.cumsum(log_returns))
        for index, date in enumerate(dates):
            raw_close = float(total_close[index])
            rows.append(
                {
                    "date": date,
                    "con_code": code,
                    "raw_high": raw_close * 1.01,
                    "raw_low": raw_close * 0.99,
                    "raw_close": raw_close,
                    "total_return_close": raw_close,
                    "volume": 1000000.0,
                    "amount": 1500000000.0,
                }
            )
    panel = pd.DataFrame(rows)
    master = pd.DataFrame(
        [
            {"ts_code": "LOW.SZ", "list_date": dates[0], "delist_date": pd.NaT, "split_bucket": 2},
            {"ts_code": "HIGH.SZ", "list_date": dates[0], "delist_date": pd.NaT, "split_bucket": 2},
        ]
    )
    benchmark = pd.DataFrame({"date": dates, "close": np.linspace(1000.0, 1100.0, len(dates))})
    selection, features, _ = select_signal(panel, master, benchmark, dates[-1], _config())
    assert selection.con_code == "LOW.SZ"
    assert selection.eligible_count == 2
    assert features.loc[features["signal_output"].eq("SIGNAL_READY")].shape[0] == 2
    target = target_frame(selection)
    assert target.iloc[0]["selection_rank"] == 1
    assert target.iloc[0]["eligible_count"] == 2
    assert target.iloc[0]["regime"] == "BUCKET2_VISIBLE_SHADOW_LOWVOL20"


def test_failed_current_report_never_presents_retained_state_as_new_signal() -> None:
    payload = {
        "run_status": "FAILED",
        "collection_status": "NO_VIEW",
        "as_of_date": "2026-08-19",
        "error_type": "ReadTimeout",
        "error": "供应商超时",
    }
    prior_state = {
        "data_trade_date": "2026-08-18",
        "status": "PENDING_T1_OPEN",
        "current_target": {"con_code": "601688.SH", "name": "华泰证券"},
    }
    report = _retained_state_report(
        payload,
        prior_state,
        "这不是本次新信号，必须等待下一次成功刷新。",
    )
    assert "`FAILED`" in report
    assert "`NO_VIEW`" in report
    assert "旧状态冒充本次成功：`false`" in report
    assert "仅作留痕的上次成功状态" in report
    assert "这不是本次新信号" in report


def test_tushare_client_fails_over_and_records_endpoint_evidence() -> None:
    class FailingClient:
        def daily(self, **_: object) -> pd.DataFrame:
            raise TimeoutError("节点一超时")

    class WorkingClient:
        def daily(self, **_: object) -> pd.DataFrame:
            return pd.DataFrame([{"ts_code": "601688.SH"}])

    api = TushareFailoverApi(
        [("https://node-one", FailingClient()), ("https://node-two", WorkingClient())]
    )
    result = api.daily(trade_date="20260819")
    assert result.iloc[0]["ts_code"] == "601688.SH"
    assert api.endpoint == "https://node-two"
    assert api.successful_endpoints == ["https://node-two"]
    assert api.failure_events == [
        {
            "method": "daily",
            "endpoint": "https://node-one",
            "error_type": "TimeoutError",
            "error": "节点一超时",
        }
    ]
