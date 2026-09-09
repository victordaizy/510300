from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.asymmetric_stress_hazard_dr007_tushare_v1_0_2 import (
    AVAILABILITY_RULE,
    CredentialExpiredError,
    REPO_MATURITY,
    TS_CODE,
    VALUE_FIELD,
    VALUE_SEMANTICS,
    build_request,
    calendar_year_chunks,
    combine_and_validate_chunks,
    parse_repo_daily_payload,
    public_transport_metadata,
    resolve_transport,
)
from research.asymmetric_stress_hazard_source_remediation_v1_0_1 import (
    sha256_file,
)
from scripts.acquire_510300_asymmetric_stress_hazard_dr007_tushare_v1_0_2 import (
    RAW_ROOT,
    relative,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "config"
    / "510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2.yaml"
)


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _payload(
    rows: list[list[object]],
    *,
    fields: list[str] | None = None,
    code: int = 0,
    message: str = "",
) -> bytes:
    body = {
        "code": code,
        "msg": message,
        "data": {
            "fields": fields
            or ["ts_code", "trade_date", "repo_maturity", "weight"],
            "items": rows,
        },
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def test_contract_freezes_exact_dr007_ib_weight_without_series_proxy(
    contract: dict[str, object],
) -> None:
    source = contract["dr007_source"]
    admission = contract["admission"]
    assert source["api_name"] == "repo_daily"
    assert source["ts_code"] == TS_CODE == "DR007.IB"
    assert source["repo_maturity"] == REPO_MATURITY == "DR007"
    assert source["selected_field"] == VALUE_FIELD == "weight"
    assert source["value_semantics"] == VALUE_SEMANTICS
    assert source["availability_rule"] == AVAILABILITY_RULE
    assert source["forbidden_substitutes"] == [
        "FDR007",
        "R007",
        "FR007",
        "EXCHANGE_REPO_R_007",
    ]
    assert admission["series_proxy_rescue_allowed"] is False
    assert admission["feature_construction_allowed_before_all_sources_admitted"] is False
    assert admission["return_evaluation_allowed_in_this_stage"] is False


def test_calendar_year_chunks_cover_frozen_window_once() -> None:
    chunks = calendar_year_chunks(date(2015, 1, 5), date(2026, 8, 14))
    assert len(chunks) == 12
    assert chunks[0] == (date(2015, 1, 5), date(2015, 12, 31))
    assert chunks[-1] == (date(2026, 1, 1), date(2026, 8, 14))
    for previous, current in zip(chunks[:-1], chunks[1:], strict=True):
        assert previous[1] + pd.Timedelta(days=1) == current[0]


def test_parse_repo_daily_accepts_only_dr007_ib_weight() -> None:
    frame = parse_repo_daily_payload(
        _payload(
            [
                ["DR007.IB", "20200804", "DR007", 2.1144],
                ["DR007.IB", "20200805", "DR007", 2.0123],
            ]
        ),
        chunk_start=date(2020, 8, 4),
        chunk_end=date(2020, 8, 5),
    )
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-08-04",
        "2020-08-05",
    ]
    assert frame["dr007"].tolist() == [2.1144, 2.0123]
    assert frame["provider_value_field"].unique().tolist() == ["weight"]
    assert frame["value_semantics"].unique().tolist() == [
        "WEIGHTED_AVERAGE_RATE_PERCENT"
    ]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (["R007.IB", "20200804", "R007", 2.1785], "非 DR007.IB"),
        (["DR007.IB", "20200804", "FDR007", 2.1144], "非 DR007 期限"),
    ],
)
def test_parse_repo_daily_rejects_semantic_substitutes(
    row: list[object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_repo_daily_payload(
            _payload([row]),
            chunk_start=date(2020, 8, 4),
            chunk_end=date(2020, 8, 4),
        )


def test_missing_market_date_remains_date_level_no_view() -> None:
    first = parse_repo_daily_payload(
        _payload([["DR007.IB", "20240102", "DR007", 1.80]]),
        chunk_start=date(2024, 1, 2),
        chunk_end=date(2024, 1, 2),
    )
    last = parse_repo_daily_payload(
        _payload([["DR007.IB", "20240104", "DR007", 1.75]]),
        chunk_start=date(2024, 1, 4),
        chunk_end=date(2024, 1, 4),
    )
    combined, metrics = combine_and_validate_chunks(
        [first, last],
        market_dates=[date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        observation_start=date(2024, 1, 2),
        observation_cutoff=date(2024, 1, 4),
    )
    assert len(combined) == 2
    assert metrics["missing_market_session_count"] == 1
    assert metrics["missing_market_session_examples"] == ["2024-01-03"]
    assert metrics["missing_value_rule"] == "NO_VIEW_NO_INTERPOLATION"
    assert metrics["interpolation_performed"] is False
    assert metrics["substitute_used"] is False


def test_transport_keeps_secret_out_of_public_metadata_and_enforces_expiry(
    contract: dict[str, object],
) -> None:
    standard = resolve_transport(
        contract,
        {"TUSHARE_TOKEN": "standard-secret"},
        now=pd.Timestamp("2026-09-02T12:00:00+08:00"),
    )
    url, payload, headers = build_request(
        standard,
        date(2020, 8, 4),
        date(2020, 8, 4),
    )
    assert url == "https://api.tushare.pro/repo_daily"
    assert payload["token"] == "standard-secret"
    assert headers == {}
    assert "standard-secret" not in json.dumps(
        public_transport_metadata(standard), ensure_ascii=False
    )

    with pytest.raises(CredentialExpiredError, match="已经过期|已于"):
        resolve_transport(
            contract,
            {
                "TUSHARE_PROXY_TOKEN": "proxy-secret",
                "TUSHARE_PROXY_TOKEN_EXPIRES_AT": "2026-08-20T00:12:08+08:00",
            },
            now=pd.Timestamp("2026-09-02T12:00:00+08:00"),
        )
    proxy = resolve_transport(
        contract,
        {
            "TUSHARE_PROXY_TOKEN": "proxy-secret",
            "TUSHARE_PROXY_TOKEN_EXPIRES_AT": "2026-08-20T00:12:08+08:00",
        },
        now=pd.Timestamp("2026-09-02T12:00:00+08:00"),
        allow_expired_proxy_for_probe=True,
    )
    proxy_url, proxy_payload, proxy_headers = build_request(
        proxy,
        date(2020, 8, 4),
        date(2020, 8, 4),
    )
    assert proxy_url == "https://fast.xiaodefa.cn"
    assert "token" not in proxy_payload
    assert proxy_headers == {"x-api-key": "proxy-secret"}
    assert "proxy-secret" not in json.dumps(
        public_transport_metadata(proxy), ensure_ascii=False
    )


def test_v1_0_2_parent_pins_match_immutable_v1_0_1_files(
    contract: dict[str, object],
) -> None:
    parent = contract["immutable_parent"]
    for path_key, hash_key in (
        ("status_path", "status_sha256"),
        ("receipt_path", "receipt_sha256"),
        ("contract_path", "contract_sha256"),
        ("sw_manifest_path", "sw_manifest_sha256"),
        ("pboc_manifest_path", "pboc_manifest_sha256"),
    ):
        assert sha256_file(ROOT / parent[path_key]) == parent[hash_key]
    calendar = contract["market_calendar"]
    assert sha256_file(ROOT / calendar["path"]) == calendar["sha256"]


def test_workspace_relative_path_does_not_follow_data_junction() -> None:
    logical = relative(RAW_ROOT / "credential_preflight" / "example.json")
    assert logical == (
        "data/raw/510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2/"
        "credential_preflight/example.json"
    )
