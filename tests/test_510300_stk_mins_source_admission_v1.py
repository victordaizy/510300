"""510300 STK_MINS来源准入V1的离线回归测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from research.stk_mins_source_admission_v1 import (
    ApiResult,
    TushareProxyClient,
    audit_minute_frames,
    expected_standard_times,
    find_latest_valid_capture,
    load_protocol,
    normalize_api_frame,
    save_immutable_capture,
    sha256_bytes,
    verify_protocol_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    status_code = 200
    content = json.dumps(
        {
            "code": 0,
            "msg": "",
            "data": {
                "fields": [
                    "ts_code",
                    "trade_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                ],
                "items": [
                    ["510300.SH", "2017-01-03 09:30:00", 3, 3, 3, 3, 100, 300]
                ],
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    text = content.decode("utf-8")

    @staticmethod
    def json() -> dict[str, object]:
        return json.loads(_FakeResponse.text)


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, endpoint: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return _FakeResponse()


def _minute_day(day: str, *, price: float = 10.0) -> pd.DataFrame:
    timestamps = [pd.Timestamp(f"{day} {label}") for label in expected_standard_times()]
    return pd.DataFrame(
        {
            "ts_code": "510300.SH",
            "trade_time": timestamps,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "vol": 100.0,
            "amount": price * 100.0,
        }
    )


def _daily_row(day: str, *, price: float = 10.0, minute_rows: int = 241) -> dict[str, object]:
    return {
        "ts_code": "510300.SH",
        "trade_date": day.replace("-", ""),
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "pre_close": price,
        "vol": float(minute_rows),
        "amount": float(price * minute_rows / 10.0),
    }


def test_protocol_manifest_and_241_bar_clock_are_frozen() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    manifest = verify_protocol_manifest(PROJECT_ROOT)

    labels = expected_standard_times()
    assert manifest["freeze_state"] == "SOURCE_PROTOCOL_FROZEN_BEFORE_BULK_ACQUISITION"
    assert len(labels) == 241
    assert labels[0] == "09:30:00"
    assert labels[120] == "11:30:00"
    assert labels[121] == "13:01:00"
    assert labels[-1] == "15:00:00"
    assert "13:00:00" not in labels
    assert protocol["timestamp_contract"]["research_windows"]["reaction"][
        "bar_labels"
    ] == ["10:01:00", "10:02:00", "10:03:00", "10:04:00", "10:05:00"]


def test_normalization_sorts_descending_provider_rows_ascending() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    frame = pd.DataFrame(
        [
            ["510300.SH", "2017-01-03 09:31:00", 3, 3.1, 2.9, 3, 100, 300],
            ["510300.SH", "2017-01-03 09:30:00", 3, 3.1, 2.9, 3, 100, 300],
        ],
        columns=protocol["api_contracts"]["stk_mins"]["required_columns"],
    )

    normalized = normalize_api_frame("stk_mins", frame, protocol)

    assert normalized["trade_time"].tolist() == [
        pd.Timestamp("2017-01-03 09:30:00"),
        pd.Timestamp("2017-01-03 09:31:00"),
    ]


def test_client_uses_header_and_capture_never_persists_credential(tmp_path: Path) -> None:
    token = "A" * 56
    session = _FakeSession()
    client = TushareProxyClient(
        endpoint="https://example.invalid",
        token=token,
        minimum_interval_seconds=0,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        maximum_attempts=1,
        retry_delays_seconds=[0],
        session=session,
    )
    parameters = {
        "ts_code": "510300.SH",
        "freq": "1min",
        "start_date": "2017-01-03 09:00:00",
        "end_date": "2017-01-03 15:30:00",
    }

    result = client.call("stk_mins", parameters)
    receipt = save_immutable_capture(
        tmp_path,
        Path("raw/stk_mins/probe"),
        result,
        client.endpoint,
    )

    assert result.success is True
    assert session.calls[0]["headers"]["x-api-key"] == token  # type: ignore[index]
    assert token not in json.dumps(session.calls[0]["json"], ensure_ascii=False)
    assert receipt["credential_persisted"] is False
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert token.encode("utf-8") not in path.read_bytes()
    match = find_latest_valid_capture(
        tmp_path,
        Path("raw/stk_mins/probe"),
        "stk_mins",
        parameters,
    )
    assert match is not None
    assert len(match[1]) == 1


def test_capture_detects_normalized_file_hash_drift(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["510300.SH"],
            "trade_time": [pd.Timestamp("2017-01-03 09:30:00")],
            "open": [3.0],
            "high": [3.0],
            "low": [3.0],
            "close": [3.0],
            "vol": [100.0],
            "amount": [300.0],
        }
    )
    raw = b'{"code":0}'
    result = ApiResult(
        api_name="stk_mins",
        request_parameters={"k": "v"},
        request_started_at="2026-09-04T22:00:00+08:00",
        response_received_at="2026-09-04T22:00:01+08:00",
        http_status=200,
        success=True,
        business_code="0",
        error_message="",
        raw_bytes=raw,
        response_sha256=sha256_bytes(raw),
        frame=frame,
    )
    receipt = save_immutable_capture(
        tmp_path,
        Path("raw/stk_mins/drift"),
        result,
        "https://example.invalid",
    )
    normalized = tmp_path / receipt["normalized_relative_path"]
    with normalized.open("ab") as handle:
        handle.write(b"drift")

    assert (
        find_latest_valid_capture(
            tmp_path,
            Path("raw/stk_mins/drift"),
            "stk_mins",
            {"k": "v"},
        )
        is None
    )


def test_daily_reconciliation_and_event_windows_pass_on_complete_data() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    days = ["2026-07-02", "2026-07-03"]
    calendar = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE"],
            "cal_date": [day.replace("-", "") for day in days],
            "is_open": [1, 1],
            "pretrade_date": ["20260701", "20260702"],
        }
    )
    minute = pd.concat([_minute_day(day) for day in days], ignore_index=True)
    daily = pd.DataFrame([_daily_row(day) for day in days])

    ledger, report = audit_minute_frames(
        calendar,
        daily,
        minute,
        protocol,
        eligible_event_dates=[pd.Timestamp(days[0]).date()],
        timestamp_evidence_pass=True,
    )

    assert len(ledger) == 2
    assert report["state"] == "PASS_STK_MINS_SOURCE_ADMISSION"
    assert report["general_source_pass"] is True
    assert report["event_window_gate_pass"] is True
    assert report["coverage"]["standard_241_bar_day_coverage"] == 1.0
    assert report["reconciliation"]["maximum_volume_relative_error"] == 0.0
    assert report["reconciliation"]["maximum_amount_relative_error"] == 0.0


def test_missing_event_window_blocks_even_when_relaxed_general_gate_passes() -> None:
    protocol = copy.deepcopy(load_protocol(PROJECT_ROOT))
    protocol["quality_gates"]["standard_241_bar_day_coverage_minimum"] = 0.5
    complete_day = "2026-07-01"
    event_day = "2026-07-02"
    calendar = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE"],
            "cal_date": ["20260701", "20260702"],
            "is_open": [1, 1],
            "pretrade_date": ["20260630", "20260701"],
        }
    )
    incomplete_event = _minute_day(event_day)
    incomplete_event = incomplete_event.loc[
        incomplete_event["trade_time"].dt.strftime("%H:%M:%S").ne("10:03:00")
    ]
    minute = pd.concat(
        [_minute_day(complete_day), incomplete_event], ignore_index=True
    )
    daily = pd.DataFrame([_daily_row(complete_day), _daily_row(event_day)])

    _, report = audit_minute_frames(
        calendar,
        daily,
        minute,
        protocol,
        eligible_event_dates=[pd.Timestamp(event_day).date()],
        timestamp_evidence_pass=True,
    )

    assert report["general_source_pass"] is True
    assert report["event_window_gate_pass"] is False
    assert report["state"] == "BLOCKED_STK_MINS_SOURCE_ADMISSION"
