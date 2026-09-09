"""510300期权前向盘口采集器的冻结安全门测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

import scripts.audit_510300_option_forward_orderbook_v1 as audit_module
from scripts.collect_510300_option_forward_orderbook_v1 import (
    active_contracts,
    atomic_parquet,
    capture_window_valid,
    validate_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")


@pytest.fixture
def contract() -> dict:
    path = ROOT / "config" / "510300_option_forward_orderbook_v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_code": ["10000001.SH", "10000002.SH", "10000003.SH"],
            "option_type": ["C", "P", "C"],
            "list_date": pd.to_datetime(["2026-08-01", "2026-08-19", "2026-07-01"]),
            "expiry_date": pd.to_datetime(["2026-09-23", "2026-09-23", "2026-08-18"]),
            "delist_date": pd.to_datetime(["2026-09-23", "2026-09-23", "2026-08-18"]),
            "strike": [4.8, 4.8, 4.7],
            "contract_unit": [10000, 10000, 10000],
            "is_adjusted": [False, False, False],
        }
    )


def valid_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_code": ["10000001.SH", "10000002.SH"],
            "bid1": [0.10, 0.11],
            "ask1": [0.11, 0.12],
            "quote_timestamp": pd.to_datetime(
                ["2026-08-19 15:00:00", "2026-08-19 14:59:59"]
            ),
        }
    )


def test_capture_window_requires_same_date_and_exact_window(contract: dict) -> None:
    expected = pd.Timestamp("2026-08-19")
    assert capture_window_valid(
        datetime(2026, 8, 19, 14, 58, 30, tzinfo=TIMEZONE), expected, contract
    )
    assert capture_window_valid(
        datetime(2026, 8, 19, 15, 5, 30, tzinfo=TIMEZONE), expected, contract
    )
    assert not capture_window_valid(
        datetime(2026, 8, 19, 14, 58, 29, tzinfo=TIMEZONE), expected, contract
    )
    assert not capture_window_valid(
        datetime(2026, 8, 20, 15, 0, 0, tzinfo=TIMEZONE), expected, contract
    )


def test_active_contracts_uses_point_in_time_listing_and_delisting(master: pd.DataFrame) -> None:
    active = active_contracts(master, pd.Timestamp("2026-08-19"))
    assert active["contract_code"].tolist() == ["10000001.SH", "10000002.SH"]


def test_valid_snapshot_passes_all_frozen_gates(contract: dict, master: pd.DataFrame) -> None:
    active = active_contracts(master, pd.Timestamp("2026-08-19"))
    started = datetime(2026, 8, 19, 14, 59, 0, tzinfo=TIMEZONE)
    finished = datetime(2026, 8, 19, 15, 1, 0, tzinfo=TIMEZONE)
    audit = validate_snapshot(
        valid_snapshot(), active, pd.Timestamp("2026-08-19"), started, finished, contract
    )
    assert audit["status"] == "PASS"
    assert all(audit["gates"].values())


@pytest.mark.parametrize("failure", ["missing", "inverted", "duplicate", "old_quote_date"])
def test_invalid_snapshot_fails_closed(
    failure: str, contract: dict, master: pd.DataFrame
) -> None:
    active = active_contracts(master, pd.Timestamp("2026-08-19"))
    snapshot = valid_snapshot()
    if failure == "missing":
        snapshot = snapshot.iloc[:1].copy()
    elif failure == "inverted":
        snapshot.loc[0, "bid1"] = 0.20
    elif failure == "duplicate":
        snapshot = pd.concat([snapshot, snapshot.iloc[[0]]], ignore_index=True)
    elif failure == "old_quote_date":
        snapshot["quote_timestamp"] = pd.Timestamp("2026-08-18 15:00:00")
    started = datetime(2026, 8, 19, 14, 59, 0, tzinfo=TIMEZONE)
    finished = datetime(2026, 8, 19, 15, 1, 0, tzinfo=TIMEZONE)
    audit = validate_snapshot(
        snapshot, active, pd.Timestamp("2026-08-19"), started, finished, contract
    )
    assert audit["status"] == "NO_VIEW"
    assert not all(audit["gates"].values())


def test_capture_finishing_after_window_fails(contract: dict, master: pd.DataFrame) -> None:
    active = active_contracts(master, pd.Timestamp("2026-08-19"))
    started = datetime(2026, 8, 19, 15, 4, 0, tzinfo=TIMEZONE)
    finished = datetime(2026, 8, 19, 15, 5, 31, tzinfo=TIMEZONE)
    audit = validate_snapshot(
        valid_snapshot(), active, pd.Timestamp("2026-08-19"), started, finished, contract
    )
    assert audit["status"] == "NO_VIEW"
    assert audit["gates"]["capture_started_in_window"]
    assert not audit["gates"]["capture_finished_in_window"]


def test_forward_parquet_never_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "20260819.parquet"
    original = pd.DataFrame({"value": [1]})
    atomic_parquet(original, path)
    original_hash = path.read_bytes()
    with pytest.raises(FileExistsError, match="不可覆盖"):
        atomic_parquet(pd.DataFrame({"value": [2]}), path)
    assert path.read_bytes() == original_hash


def test_forward_outcome_maturity_and_dividend_are_auditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = pd.bdate_range("2026-08-19", periods=65)
    market_path = tmp_path / "market.parquet"
    dividend_path = tmp_path / "dividends.csv"
    pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "close": 100.0,
        }
    ).to_parquet(market_path, index=False)
    pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "ex_date": [dates[5]],
            "cash_dividend_per_share": [1.0],
        }
    ).to_csv(dividend_path, index=False)
    monkeypatch.setattr(audit_module, "MARKET_FILE", market_path)
    monkeypatch.setattr(audit_module, "DIVIDEND_FILE", dividend_path)
    registry = {
        "targets": {
            "cash_annual_rate": 0.015,
            "trading_days_per_year": 242,
            "primary": {"threshold": -0.05},
        }
    }
    outcomes = audit_module.build_forward_outcomes([dates[0], dates[10]], registry)
    assert outcomes.loc[0, "maturity_date20"] == dates[20]
    assert outcomes.loc[0, "maturity_date60"] == dates[60]
    assert outcomes.loc[0, "relative_return20"] > 0
    assert outcomes.loc[0, "bad20"] == False  # noqa: E712
    assert pd.isna(outcomes.loc[1, "maturity_date60"])


def test_bad_state_episode_merging_uses_frozen_trading_day_gap() -> None:
    outcomes = pd.DataFrame(
        {
            "signal_market_position": [1, 5, 30, 55, 80],
            "bad20": [True, True, True, True, False],
        }
    )
    assert audit_module.count_independent_bad_episodes(outcomes, merge_gap=20) == 3


def test_windows_powershell_runner_is_ascii_safe() -> None:
    """Windows PowerShell 5.1 会按本地代码页误读无 BOM 的 UTF-8 中文脚本。"""
    runner = ROOT / "scripts" / "run_510300_option_forward_orderbook_v1_once.ps1"
    raw = runner.read_bytes()
    assert raw.isascii(), "启动脚本必须保持纯 ASCII，避免计划任务下出现解析错误"
