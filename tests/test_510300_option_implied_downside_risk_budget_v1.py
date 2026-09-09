"""510300 期权隐含下行风险预算 V1 数据准入测试。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

from research.option_implied_downside_risk_budget_v1 import (
    ApiResult,
    TushareProxyClient,
    black76_price,
    build_contract_map,
    build_daily_surface,
    canonical_json_bytes,
    find_latest_valid_capture,
    implied_volatility_black76,
    interpolate_rate,
    interpolate_total_variance,
    load_protocol,
    sanitize_text,
    save_immutable_capture,
    sha256_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingSession:
    """记录请求并返回预设响应。"""

    def __init__(self, response: requests.Response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs: object) -> requests.Response:
        self.calls.append({"url": url, **kwargs})
        return self.response


def make_response(payload: dict, status_code: int = 200) -> requests.Response:
    """构造 requests.Response。"""

    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


def make_api_result(token_free_value: str = "测试") -> ApiResult:
    """构造成功的内存 API 结果。"""

    body = {
        "code": 0,
        "msg": "",
        "data": {"fields": ["value"], "items": [[token_free_value]]},
    }
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return ApiResult(
        api_name="demo",
        request_parameters={"key": "公开参数"},
        fields_requested="",
        request_started_at="2026-09-04T20:20:00+08:00",
        response_received_at="2026-09-04T20:20:01+08:00",
        http_status=200,
        success=True,
        error_code="0",
        error_message="",
        raw_bytes=raw,
        response_sha256=sha256_bytes(raw),
        frame=pd.DataFrame({"value": [token_free_value]}),
    )


def test_canonical_json_is_stable_and_sanitizer_hides_credentials() -> None:
    token = "a" * 56
    left = canonical_json_bytes({"b": 2, "a": 1})
    right = canonical_json_bytes({"a": 1, "b": 2})
    assert left == right
    text = sanitize_text(f"token={token} x-api-key:{token}", token)
    assert token not in text
    assert "凭据已隐藏" in text


def test_proxy_client_uses_header_and_never_places_token_in_body() -> None:
    token = "b" * 56
    response = make_response(
        {
            "code": 0,
            "msg": "成功",
            "data": {"fields": ["trade_date", "close"], "items": [["20200102", 4.0]]},
        }
    )
    session = RecordingSession(response)
    client = TushareProxyClient(
        endpoint="https://tt.xiaodefa.cn",
        token=token,
        minimum_interval_seconds=0,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        maximum_attempts=1,
        retry_delays_seconds=[0],
        session=session,
    )
    result = client.call("fund_daily", {"ts_code": "510300.SH"})
    assert result.success
    assert result.frame.to_dict("records") == [{"trade_date": "20200102", "close": 4.0}]
    assert session.calls[0]["headers"]["x-api-key"] == token
    assert token not in json.dumps(session.calls[0]["json"], ensure_ascii=False)
    assert token not in result.raw_bytes.decode("utf-8")


def test_immutable_capture_is_resumable_and_contains_no_credential(tmp_path: Path) -> None:
    token = "c" * 56
    result = make_api_result()
    partition = Path("data/raw/test/demo")
    first = save_immutable_capture(
        tmp_path, partition, result, "https://tt.xiaodefa.cn"
    )
    second = save_immutable_capture(
        tmp_path, partition, result, "https://tt.xiaodefa.cn"
    )
    assert first["raw_relative_path"] != second["raw_relative_path"]
    found = find_latest_valid_capture(
        tmp_path, partition, "demo", {"key": "公开参数"}
    )
    assert found is not None
    receipt, frame = found
    assert receipt["credential_persisted"] is False
    assert frame.to_dict("records") == [{"value": "测试"}]
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert token.encode("utf-8") not in path.read_bytes()


@pytest.mark.parametrize("call_put", ["C", "P"])
def test_black76_implied_volatility_round_trip(call_put: str) -> None:
    expected = 0.28
    price = black76_price(4.0, 3.9, 45 / 365, 0.02, expected, call_put)
    actual = implied_volatility_black76(
        price,
        4.0,
        3.9,
        45 / 365,
        0.02,
        call_put,
        0.01,
        2.0,
    )
    assert actual == pytest.approx(expected, abs=1e-9)


def test_shibor_interpolation_refuses_extrapolation() -> None:
    row = {"1w": 1.7, "2w": 1.8, "1m": 2.0, "3m": 2.6}
    assert interpolate_rate(row, 7) == pytest.approx(0.017)
    assert interpolate_rate(row, 22) == pytest.approx(0.019)
    assert interpolate_rate(row, 6) is None
    assert interpolate_rate(row, 91) is None


def test_total_variance_interpolation_uses_no_extrapolation() -> None:
    value = interpolate_total_variance(
        [-0.1, 0.0],
        [0.3, 0.2],
        [30 / 365, 30 / 365],
        -0.05,
    )
    expected = 0.5 * (0.3**2 * 30 / 365) + 0.5 * (0.2**2 * 30 / 365)
    assert value == pytest.approx(expected)
    assert (
        interpolate_total_variance([-0.1, 0.0], [0.3, 0.2], [0.1, 0.1], 0.01)
        is None
    )


def test_contract_map_uses_exact_opt_code_and_preserves_adjusted_rows() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    columns = protocol["api_contracts"]["opt_basic"]["required_columns"]
    defaults = {column: "" for column in columns}
    rows = []
    for ts_code, opt_code, multiplier in (
        ("10000001.SH", "OP510300.SH", 10000),
        ("10000002.SH", "OP510300.SH", 9800),
        ("10000003.SH", "OP510050.SH", 10000),
    ):
        row = dict(defaults)
        row.update(
            {
                "ts_code": ts_code,
                "exchange": "SSE",
                "name": "测试合约",
                "opt_code": opt_code,
                "call_put": "C",
                "exercise_price": 4.0,
                "opt_multiplier": multiplier,
                "maturity_date": "20201223",
                "list_date": "20200101",
                "delist_date": "20201223",
            }
        )
        rows.append(row)
    result = build_contract_map(pd.DataFrame(rows), protocol)
    assert result["ts_code"].tolist() == ["10000001.SH", "10000002.SH"]
    assert result["is_standard_multiplier"].tolist() == [True, False]


def synthetic_day_options(trade_date: pd.Timestamp) -> pd.DataFrame:
    """生成可同时包围 30 日和 60 日的无噪声期权链。"""

    forward = 4.0
    rate = 0.02
    volatility = 0.25
    rows: list[dict] = []
    serial = 0
    for dte in (20, 40, 80):
        maturity = trade_date + pd.Timedelta(days=dte)
        maturity_years = dte / 365
        for k in (-0.08, -0.05, -0.02, 0.02, 0.05):
            strike = forward * math.exp(k)
            for call_put in ("C", "P"):
                serial += 1
                rows.append(
                    {
                        "ts_code": f"{serial:08d}.SH",
                        "trade_date": trade_date,
                        "settle": black76_price(
                            forward,
                            strike,
                            maturity_years,
                            rate,
                            volatility,
                            call_put,
                        ),
                        "vol": 100.0,
                        "oi": 100.0,
                        "exercise_price": strike,
                        "opt_multiplier": 10000.0,
                        "call_put_normalized": call_put,
                        "maturity_date": maturity,
                        "list_date": trade_date - pd.Timedelta(days=100),
                        "delist_date": maturity,
                        "is_standard_multiplier": True,
                    }
                )
    return pd.DataFrame(rows)


def test_daily_surface_builds_only_from_legal_brackets() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    trade_date = pd.Timestamp("2024-01-02")
    options = synthetic_day_options(trade_date)
    shibor_row = {"1w": 2.0, "2w": 2.0, "1m": 2.0, "3m": 2.0}
    ledger, surface = build_daily_surface(
        trade_date,
        options,
        underlying_close=4.0,
        shibor_row=shibor_row,
        protocol=protocol,
    )
    assert ledger["state"] == "PASS_VALID_SURFACE"
    assert surface is not None
    assert surface["iv30_atm"] == pytest.approx(0.25, abs=1e-8)
    assert surface["iv30_put5"] == pytest.approx(0.25, abs=1e-8)
    assert surface["iv60_atm"] == pytest.approx(0.25, abs=1e-8)


def test_daily_surface_preserves_missing_shibor_as_no_view() -> None:
    protocol = load_protocol(PROJECT_ROOT)
    trade_date = pd.Timestamp(datetime(2024, 1, 2))
    ledger, surface = build_daily_surface(
        trade_date,
        synthetic_day_options(trade_date),
        underlying_close=4.0,
        shibor_row=None,
        protocol=protocol,
    )
    assert ledger["state"] == "NO_VIEW_MISSING_SAME_DAY_SHIBOR"
    assert surface is None
