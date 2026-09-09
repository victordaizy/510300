from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

from market_data.etf_daily_crosscheck_v1_2 import SseEtfDailyCrosscheckProvider
from market_data.etf_primary_market import EtfPrimaryMarketRequest, ProviderSnapshot
from research.primary_market_forward_readiness_v1_2 import (
    evaluate_forward_readiness_v1_2,
)
from scripts.free_source_storage_v1_2 import write_content_addressed_raw


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG = ROOT / "config" / "primary_market_forward_v1_2.yaml"
RUNNER = ROOT / "scripts" / "run_510300_primary_market_forward_v1_2.ps1"
TASK_WRAPPER = (
    ROOT / "scripts" / "run_510300_primary_market_collection_task_v1_2.ps1"
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeDailySession:
    def __init__(self) -> None:
        self.last_params: dict[str, str] = {}
        self.last_headers: dict[str, str] = {}

    def get(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.last_params = params
        self.last_headers = headers
        return _FakeResponse(
            {
                "result": [
                    {
                        "SEC_CODE": "510300",
                        "SEC_NAME": "300ETF",
                        "TX_DATE": "20260826",
                        "OPEN_PRICE": "4.601",
                        "HIGH_PRICE": "4.640",
                        "LOW_PRICE": "4.590",
                        "CLOSE_PRICE": "4.616",
                        "TRADE_VOL": "74569.59",
                        "TRADE_AMT": "343801.62",
                    }
                ]
            }
        )


def test_official_daily_crosscheck_provider_parses_frozen_fields() -> None:
    session = _FakeDailySession()
    snapshot = SseEtfDailyCrosscheckProvider(session=session).fetch(
        EtfPrimaryMarketRequest(fund_code="510300"), date(2026, 8, 26)
    )
    assert snapshot.record["trade_date"] == date(2026, 8, 26)
    assert snapshot.record["open"] == 4.601
    assert snapshot.record["high"] == 4.64
    assert snapshot.record["low"] == 4.59
    assert snapshot.record["close"] == 4.616
    assert session.last_params["sqlId"] == (
        "COMMON_SSE_CP_GPJCTPZ_GPLB_CJGK_MRGK_C"
    )
    assert session.last_params["TX_DATE"] == "2026-08-26"
    assert "sse.com.cn" in session.last_headers["Referer"]


def test_raw_response_path_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    snapshot = ProviderSnapshot(
        record={"fund_code": "510300", "trade_date": "2026-08-26"},
        raw_payload={"result": [{"SEC_CODE": "510300"}]},
    )
    observed_at = datetime(2026, 8, 26, 15, 1, tzinfo=TIMEZONE)
    metadata = write_content_addressed_raw(
        snapshot,
        tmp_path / "raw",
        "daily_crosscheck",
        observed_at,
        root=tmp_path,
    )
    path = tmp_path / metadata["path"]
    assert path.stem == metadata["sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    second = write_content_addressed_raw(
        snapshot,
        tmp_path / "raw",
        "daily_crosscheck",
        observed_at,
        root=tmp_path,
    )
    assert second["path"] == metadata["path"]
    assert second["existing_identical"] is True


def _forward_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_date = pd.Timestamp("2026-08-26")
    pcf = pd.DataFrame(
        {
            "trading_day": [trade_date],
            "retrieved_at": ["2026-08-26T09:30:01+08:00"],
        }
    )
    rows: list[dict[str, object]] = []
    for minute in range(120):
        exchange_timestamp = trade_date + pd.Timedelta(hours=9, minutes=30 + minute)
        rows.append(
            {
                "trade_date": trade_date,
                "exchange_timestamp": exchange_timestamp,
                "retrieved_at": exchange_timestamp.tz_localize(
                    "Asia/Shanghai"
                ).isoformat(),
                "iopv": 4.615,
                "last_price": 4.616,
                "open": 4.601,
                "high": 4.640,
                "low": 4.590,
            }
        )
    final_timestamp = trade_date + pd.Timedelta(hours=14, minutes=59, seconds=30)
    rows.append(
        {
            "trade_date": trade_date,
            "exchange_timestamp": final_timestamp,
            "retrieved_at": final_timestamp.tz_localize("Asia/Shanghai").isoformat(),
            "iopv": 4.615,
            "last_price": 4.616,
            "open": 4.601,
            "high": 4.640,
            "low": 4.590,
        }
    )
    return pcf, pd.DataFrame(rows)


def _crosscheck(*, close: float = 4.616) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-08-26"],
            "fund_code": ["510300"],
            "open": [4.601],
            "high": [4.640],
            "low": [4.590],
            "close": [close],
            "raw_hash_verified": [True],
            "receipt_verified": [True],
        }
    )


def _evaluate(crosscheck: pd.DataFrame) -> dict[str, object]:
    pcf, iopv = _forward_inputs()
    return evaluate_forward_readiness_v1_2(
        pcf,
        iopv,
        crosscheck,
        minimum_full_coverage_days=20,
        recommended_full_coverage_days=40,
        minimum_snapshots_per_day=120,
        first_unseen_evaluation_full_coverage_days=80,
        replication_full_coverage_days=120,
        crosscheck_required_from=date(2026, 8, 26),
        minimum_final_snapshot_time="14:59:00",
        maximum_ohl_price_difference_cny=0.005,
        maximum_close_price_difference_cny=0.01,
    )


def test_new_quality_day_fails_closed_without_official_daily_crosscheck() -> None:
    result = _evaluate(pd.DataFrame())
    quality = result["daily_quality"][0]
    assert result["full_coverage_days"] == 0
    assert quality["checks"]["official_daily_crosscheck_present_once"] is False
    assert quality["complete_quality_day"] is False


def test_new_quality_day_passes_with_replayable_official_crosscheck() -> None:
    result = _evaluate(_crosscheck())
    quality = result["daily_quality"][0]
    assert result["full_coverage_days"] == 1
    assert quality["checks"]["official_daily_crosscheck_present_once"] is True
    assert quality["checks"]["official_daily_crosscheck_raw_hash_verified"] is True
    assert quality["checks"]["official_daily_crosscheck_receipt_verified"] is True
    assert quality["checks"]["official_daily_ohl_match"] is True
    assert quality["checks"]["official_daily_close_match"] is True
    assert quality["complete_quality_day"] is True


def test_crosscheck_price_mismatch_keeps_day_incomplete() -> None:
    result = _evaluate(_crosscheck(close=4.70))
    quality = result["daily_quality"][0]
    assert result["full_coverage_days"] == 0
    assert quality["checks"]["official_daily_close_match"] is False
    assert "official_daily_close_match" in quality["failure_reasons"]


def test_v1_2_config_and_supervisor_use_versioned_outputs() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["collector"]["collector_id"].endswith("V1_2")
    assert config["outputs"]["pcf_daily_file"].endswith("_v1_2.parquet")
    assert config["outputs"]["iopv_snapshot_file"].endswith("_v1_2.parquet")
    assert config["outputs"]["daily_crosscheck_file"].endswith("_v1_2.parquet")
    assert config["quality"]["crosscheck_required_from"] == "2026-08-26"
    source = config["source_contract"]["sources"]["daily_crosscheck"]
    assert source["source_id"] == "SSE_ETF_DAILY_TURNOVER_OFFICIAL"
    assert source["access_cost_cny"] == 0
    supervisor = yaml.safe_load(
        (ROOT / "config" / "priority_forward_supervisor_v1_2.yaml").read_text(
            encoding="utf-8"
        )
    )
    primary = next(
        task for task in supervisor["tasks"] if task["id"] == "PRIMARY_MARKET_PCF_IOPV"
    )
    assert primary["launcher"].endswith("_v1_2.ps1")
    assert supervisor["outputs"]["codex_task_claim_directory"].endswith("_v1_6")


def _powershell_5_parse(path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CODEX_POWERSHELL_PARSE_TARGET"] = str(path)
    command = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:CODEX_POWERSHELL_PARSE_TARGET, [ref]$tokens, [ref]$errors) "
        "| Out-Null; if ($errors.Count -gt 0) { exit 1 }"
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@pytest.mark.parametrize("path", [RUNNER, TASK_WRAPPER])
def test_v1_2_powershell_entrypoints_are_bom_and_parse(path: Path) -> None:
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    result = _powershell_5_parse(path)
    assert result.returncode == 0, result.stderr
