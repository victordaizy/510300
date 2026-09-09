from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.asymmetric_stress_hazard_source_remediation_v1_0_1 import (
    OBSERVATION_CUTOFF,
    OBSERVATION_START,
    build_csi300_member_day_sw_industry,
    parse_pboc_open_market_notice,
    sha256_file,
    summarize_pboc_policy_rate_ledger,
    validate_dr007_daily,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "config/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1.yaml"
)
CURATED_ROOT = (
    ROOT
    / "data"
    / "curated"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
)
STATUS_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1.json"
)
RECEIPT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1_receipt.json"
)


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _pboc_html(
    *,
    notice_date: str = "2024-07-24",
    title: str = "公开市场业务交易公告 [2024]第144号",
    cells: tuple[str, ...] = ("7 天", "661 亿元", "1. 7 0 %"),
) -> bytes:
    row = "".join(f"<td>{cell}</td>" for cell in cells)
    return (
        "<html><head>"
        f'<meta name="ArticleTitle" content="{title}">'
        f'<meta name="PubDate" content="{notice_date}">'
        "</head><body>"
        f"文章来源： {notice_date} 09:20:30"
        f"<table><tr>{row}</tr></table>"
        "</body></html>"
    ).encode("utf-8")


def test_contract_keeps_dr007_exact_and_blocks_proxy_rescue(
    config: dict[str, object],
) -> None:
    dr007 = config["remediated_sources"]["dr007_daily"]
    assert dr007["selected_series"] == "DR007"
    assert dr007["selected_field"] == "WEIGHTED_AVERAGE_RATE_PERCENT"
    assert dr007["required_availability_rule"] == (
        "NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE"
    )
    assert dr007["forbidden_substitutes"] == [
        "FDR007",
        "R007",
        "FR007",
        "EXCHANGE_REPO_R_007",
    ]
    assert config["admission"]["proxy_rescue_allowed"] is False
    assert config["admission"][
        "feature_construction_allowed_before_all_sources_admitted"
    ] is False


def test_pboc_parser_normalizes_only_numeric_layout_spaces() -> None:
    result = parse_pboc_open_market_notice(
        _pboc_html(),
        source_url="https://www.pbc.gov.cn/example/20240724/index.html",
    )
    assert result.notice_date.isoformat() == "2024-07-24"
    assert result.seven_day_operation_amount_100m == 661.0
    assert result.seven_day_rate_percent == 1.7
    assert result.parse_status == "PUBLISHED_7D_OPERATION_RATE"


def test_pboc_zero_operation_without_rate_stays_missing() -> None:
    result = parse_pboc_open_market_notice(
        _pboc_html(
            notice_date="2026-08-14",
            title="公开市场业务交易公告 [2026]第157号",
            cells=("7 天", "0 亿元", "0 亿元"),
        ),
        source_url="https://www.pbc.gov.cn/example/20260814/index.html",
    )
    assert result.seven_day_operation_amount_100m == 0.0
    assert result.seven_day_rate_percent is None
    assert result.parse_status == "ZERO_7D_OPERATION_RATE_NOT_PUBLISHED"


def test_pboc_summary_uses_actual_change_dates_without_interpolation() -> None:
    ledger = pd.DataFrame(
        {
            "notice_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "published_at": [
                "2024-01-02 09:00:00",
                "2024-01-03 09:00:00",
                "2024-01-04 09:00:00",
            ],
            "source_url": ["https://pbc/1", "https://pbc/2", "https://pbc/3"],
            "raw_path": ["raw/1", "raw/2", "raw/3"],
            "raw_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "parse_status": [
                "PUBLISHED_7D_OPERATION_RATE",
                "ZERO_7D_OPERATION_RATE_NOT_PUBLISHED",
                "PUBLISHED_7D_OPERATION_RATE",
            ],
            "seven_day_operation_amount_100m": [100.0, 0.0, 200.0],
            "seven_day_rate_percent": [1.8, None, 1.7],
        }
    )
    summary = summarize_pboc_policy_rate_ledger(ledger)
    assert summary["published_7d_rate_date_count"] == 2
    assert summary["zero_operation_without_published_rate_count"] == 1
    assert summary["rate_change_count"] == 2
    assert summary["interpolation_performed"] is False
    assert summary["inferred_change_dates_used"] is False


def _synthetic_membership() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rows = []
    for session in dates:
        for value in range(300):
            rows.append(
                {
                    "membership_date": session,
                    "symbol": f"{value:06d}.SZ",
                }
            )
    return pd.DataFrame(rows)


def _synthetic_sw_history() -> pd.DataFrame:
    rows = []
    for value in range(300):
        rows.append(
            {
                "symbol": f"{value:06d}.SZ",
                "industry_code": "110101",
                "industry_l1_code": "11",
                "effective_date": pd.Timestamp("2020-01-01"),
                "record_updated_at": pd.Timestamp(
                    "2024-01-03 09:00:00" if value == 299 else "2023-12-31 09:00:00"
                ),
                "classification_standard": "SW_2014",
                "source_sha256": "a" * 64,
            }
        )
    return pd.DataFrame(rows)


def test_sw_bitemporal_clock_turns_unavailable_member_into_no_view() -> None:
    output, metrics = build_csi300_member_day_sw_industry(
        _synthetic_membership(),
        _synthetic_sw_history(),
        observation_start=pd.Timestamp("2024-01-02").date(),
        observation_cutoff=pd.Timestamp("2024-01-03").date(),
    )
    counts = (
        output.assign(
            available=output["mapping_status"].eq("PIT_AVAILABLE_BY_MARKET_CLOSE")
        )
        .groupby("membership_date")["available"]
        .sum()
        .tolist()
    )
    assert counts == [299, 300]
    assert metrics["pit_no_view_member_days"] == 1
    assert metrics["full_300_member_session_count"] == 1
    assert metrics["missing_value_rule"] == "NO_VIEW_NO_INTERPOLATION"


def test_dr007_validation_rejects_fdr007_and_preserves_missing_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": [OBSERVATION_START, OBSERVATION_CUTOFF],
            "dr007": [2.50, 1.40],
        }
    )
    market_dates = [
        OBSERVATION_START,
        pd.Timestamp("2020-01-02").date(),
        OBSERVATION_CUTOFF,
    ]
    with pytest.raises(ValueError, match="禁止把 FDR007 作为 DR007"):
        validate_dr007_daily(
            frame,
            market_dates=market_dates,
            source_identity="FDR007",
        )
    metrics = validate_dr007_daily(
        frame,
        market_dates=market_dates,
        source_identity="AUTHORIZED_CFETS_DR007_EXPORT",
    )
    assert metrics["selected_series"] == "DR007"
    assert metrics["missing_market_session_count"] == 1
    assert metrics["missing_value_rule"] == "NO_VIEW_NO_INTERPOLATION"


def test_frozen_acquisition_artifacts_match_contract_hashes(
    config: dict[str, object],
) -> None:
    sources = config["remediated_sources"]
    for source_name in ("sw_pit_industry", "pboc_7d_reverse_repo_policy_rate"):
        contract = sources[source_name]
        path = ROOT / contract["acquisition_manifest"]
        assert sha256_file(path) == contract["acquisition_manifest_sha256"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"].startswith("PASS_SOURCE_ACQUIRED_")
    probe = sources["dr007_daily"]
    probe_path = ROOT / probe["public_retention_probe_manifest"]
    assert sha256_file(probe_path) == probe["public_retention_probe_manifest_sha256"]


def test_current_remediation_receipt_blocks_only_m2_and_unlocks_nothing() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    assert status["status"] == "NO_VIEW_SOURCE_REMEDIATION_PARTIAL_DR007_BLOCKED"
    assert status["blocked_channels"] == ["M2"]
    assert status["channel_results"]["F3"]["admitted"] is True
    assert status["channel_results"]["T2"]["admitted"] is True
    assert status["channel_results"]["T3"]["admitted"] is True
    assert status["channel_results"]["M2"]["admitted"] is False
    assert status["feature_construction_allowed"] is False
    assert status["g2_allowed"] is False
    assert status["return_evaluation"] == "NOT_ALLOWED"
    assert status["bad10_census_rerun"] is False
    assert status["substitute_or_proxy_used"] is False

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    declared = receipt.pop("receipt_payload_sha256")
    payload = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == declared
    assert receipt["outputs"]["status"]["sha256"] == sha256_file(STATUS_PATH)
    assert receipt["all_required_sources_admitted"] is False
