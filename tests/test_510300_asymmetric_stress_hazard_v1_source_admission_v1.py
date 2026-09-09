from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.asymmetric_stress_hazard_source_admission_v1 import (
    adjudicate_channel_sources,
    evaluate_cgb10y,
    evaluate_constituent_total_return_panel,
    evaluate_missing_required_sources,
    evaluate_official_pe,
    evaluate_pit_industry_intervals,
    evaluate_pit_membership,
)
from research.project_evidence_contract_v1 import EvidenceContractError


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "config/510300_asymmetric_stress_hazard_v1_source_contracts_v1.yaml"
)


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _pe_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "index_code": "000300",
            "pe_official": [10.0 + index for index in range(len(dates))],
            "provider_original_field": "peg",
            "source": "csindex.indexCsiDsPe",
            "retrieved_at": "2026-08-13T00:00:00+08:00",
        }
    )


def _pe_raw(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "payloads": {
            "pe": {
                "data": [
                    {
                        "tradeDate": row.date.strftime("%Y%m%d"),
                        "peg": float(row.pe_official),
                    }
                    for row in frame.itertuples(index=False)
                ]
            }
        }
    }


def test_source_contract_locks_dr007_clocks_and_no_trade(
    config: dict[str, object],
) -> None:
    missing = config["missing_required_sources"]
    assert missing["dr007_daily"]["selected_series"] == "DR007"
    assert missing["dr007_daily"]["forbidden_substitutes"] == [
        "FDR007",
        "R007",
        "FR007",
        "EXCHANGE_REPO_R_007",
    ]
    assert missing["reverse_repo_policy_rate_7d"]["interpolation_allowed"] is False
    assert config["availability_clock"]["missing_value_rule"] == (
        "NO_VIEW_NO_INTERPOLATION"
    )
    program = config["program"]
    assert program["research_state"] == "DISCOVERY_ONLY"
    assert program["position_impact"] == 0
    assert program["model_training_allowed"] is False
    assert program["portfolio_evaluation_allowed"] is False
    assert program["order_generation_allowed"] is False
    assert program["live_trading_allowed"] is False


def test_official_pe_requires_exact_raw_payload_match(
    config: dict[str, object],
) -> None:
    dates = pd.bdate_range("2021-01-04", periods=4)
    frame = _pe_frame(dates)
    result = evaluate_official_pe(
        frame,
        _pe_raw(frame),
        dates,
        contract=config["source_contract"]["csi300_official_pe"],
        raw_contract=config["source_contract"]["csi300_official_raw_snapshot"],
    )
    assert result["admitted"] is True
    assert result["metrics"]["raw_payload_exact_match"] is True
    assert result["feature_values_constructed"] is False


def test_official_pe_rejects_raw_payload_value_mismatch(
    config: dict[str, object],
) -> None:
    dates = pd.bdate_range("2021-01-04", periods=3)
    frame = _pe_frame(dates)
    raw = _pe_raw(frame)
    raw["payloads"]["pe"]["data"][1]["peg"] = 99.0
    with pytest.raises(EvidenceContractError, match="数值不一致"):
        evaluate_official_pe(
            frame,
            raw,
            dates,
            contract=config["source_contract"]["csi300_official_pe"],
            raw_contract=config["source_contract"][
                "csi300_official_raw_snapshot"
            ],
        )


def test_cgb10y_preserves_leading_no_view_and_next_open_clock(
    config: dict[str, object],
) -> None:
    market_dates = pd.bdate_range("2021-01-04", periods=5)
    frame = pd.DataFrame(
        {
            "date": market_dates[1:],
            "cgb_10y": np.linspace(2.5, 2.8, 4),
            "source": "chinabond.via_akshare.bond_china_yield",
            "retrieved_at": pd.Timestamp("2026-08-13", tz="Asia/Shanghai"),
        }
    )
    result = evaluate_cgb10y(
        frame,
        market_dates,
        contract=config["source_contract"][
            "china_10y_government_bond_yield"
        ],
    )
    assert result["admitted"] is True
    assert result["date_level_no_view_required"] is True
    assert result["metrics"]["leading_no_view_market_sessions"] == 1
    assert result["metrics"]["missing_market_sessions_in_source_span"] == 0
    assert result["availability_clock"].startswith("NEXT_TRADING_DAY_OPEN")


def _membership(dates: pd.DatetimeIndex, count: int = 300) -> pd.DataFrame:
    rows = []
    for date in dates:
        for index in range(count):
            rows.append(
                {
                    "membership_date": date,
                    "index_code": "000300",
                    "symbol": f"{index:06d}.SZ",
                    "state_transition_session": date,
                    "state_cycle_id": "TEST_OFFICIAL_CYCLE",
                    "source_contract": "TEST_OFFICIAL_REPLAY",
                }
            )
    return pd.DataFrame(rows)


def _membership_receipt(dates: pd.DatetimeIndex, row_count: int) -> dict[str, object]:
    return {
        "membership_admission": {
            "status": "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION",
            "row_count": row_count,
            "open_session_count": len(dates),
            "active_constituent_count_each_open_session": 300,
        }
    }


def test_membership_requires_exactly_300_members_each_market_session(
    config: dict[str, object],
) -> None:
    dates = pd.bdate_range("2021-01-04", periods=2)
    membership = _membership(dates)
    result = evaluate_pit_membership(
        membership,
        _membership_receipt(dates, len(membership)),
        dates,
        contract=config["source_contract"]["pit_csi300_membership"],
        admission_contract=config["source_contract"][
            "pit_csi300_membership_admission"
        ],
    )
    assert result["admitted"] is True
    assert result["metrics"]["constituent_count_each_session"] == 300

    incomplete = membership.iloc[:-1].copy()
    with pytest.raises(EvidenceContractError, match="恰好 300"):
        evaluate_pit_membership(
            incomplete,
            _membership_receipt(dates, len(incomplete)),
            dates,
            contract=config["source_contract"]["pit_csi300_membership"],
            admission_contract=config["source_contract"][
                "pit_csi300_membership_admission"
            ],
        )


def test_constituent_gap_becomes_date_level_no_view_without_fill(
    config: dict[str, object],
) -> None:
    dates = pd.bdate_range("2021-01-04", periods=2)
    membership = _membership(dates)
    rows = []
    for date in dates:
        upper = 300 if date == dates[0] else 299
        for index in range(upper):
            rows.append(
                {
                    "date": date,
                    "con_code": f"{index:06d}.SZ",
                    "total_return_close": 100.0 + index,
                    "price_source": "tushare_proxy.daily",
                    "adjustment_source": "tushare_proxy.adj_factor",
                    "is_suspended": False,
                }
            )
    result, window = evaluate_constituent_total_return_panel(
        pd.DataFrame(rows),
        membership,
        contract=config["source_contract"]["constituent_total_return_panel"],
        observation_cutoff=dates[-1].date().isoformat(),
    )
    assert window == (dates[0], dates[-1])
    assert result["admitted"] is True
    assert result["date_level_no_view_required"] is True
    assert result["metrics"]["full_300_member_session_count"] == 1
    assert result["metrics"]["no_view_session_count"] == 1
    assert result["metrics"]["missing_member_day_count"] == 1


def test_industry_structure_cannot_override_unproven_version_provenance(
    config: dict[str, object],
) -> None:
    dates = pd.bdate_range("2021-01-04", periods=2)
    membership = _membership(dates)
    intervals = pd.DataFrame(
        {
            "con_code": [f"{index:06d}.SZ" for index in range(300)],
            "industry_l1_code": "801010.SI",
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.NaT,
            "classification_usage": "POINT_IN_TIME_INTERVAL",
            "source": "tushare_proxy.index_member_all",
        }
    )
    raw = pd.DataFrame(
        {
            "l1_code": "801010.SI",
            "ts_code": [f"{index:06d}.SZ" for index in range(300)],
            "in_date": "20200101",
            "out_date": "",
            "source": "tushare_proxy.index_member_all",
        }
    )
    result = evaluate_pit_industry_intervals(
        intervals,
        raw,
        membership,
        contract=config["source_contract"]["pit_industry_intervals"],
        raw_contract=config["source_contract"]["pit_industry_intervals_raw"],
        coverage_window=(dates[0], dates[-1]),
    )
    assert result["metrics"]["full_300_member_session_count"] == 2
    assert result["metrics"]["independent_version_provenance_proven"] is False
    assert result["admitted"] is False
    assert result["status"] == (
        "BLOCKED_NO_INDEPENDENT_VERSION_PROVEN_PIT_INDUSTRY_PROVENANCE"
    )


def test_missing_m2_sources_are_terminal_blockers_not_substitutable(
    config: dict[str, object],
) -> None:
    results = evaluate_missing_required_sources(config["missing_required_sources"])
    assert results["dr007_daily"]["admitted"] is False
    assert results["dr007_daily"]["status"] == (
        "BLOCKED_NO_FROZEN_DR007_DAILY_SERIES"
    )
    assert results["reverse_repo_policy_rate_7d"]["admitted"] is False
    assert "FDR007" in results["dr007_daily"]["limitations"][1]


def test_any_missing_required_source_blocks_features_g2_model_and_portfolio(
    config: dict[str, object],
) -> None:
    dependencies = {
        source_id
        for channel in ("M1", "M2", "M3", "F1", "F2", "F3", "T1", "T2", "T3")
        for source_id in config["admission_gates"][f"{channel}_requires"]
    }
    source_results = {
        source_id: {
            "admitted": True,
            "status": "PASS_SOURCE_ADMISSION",
            "date_level_no_view_required": False,
        }
        for source_id in dependencies
    }
    source_results.update(
        evaluate_missing_required_sources(config["missing_required_sources"])
    )
    channels, decision = adjudicate_channel_sources(
        source_results, config["admission_gates"]
    )
    assert channels["M2"]["admitted"] is False
    assert decision["status"] == "NO_VIEW_SOURCE_ADMISSION_FAILED"
    assert decision["feature_construction_allowed"] is False
    assert decision["feature_values_constructed"] is False
    assert decision["g2_allowed"] is False
    assert decision["model_training_allowed"] is False
    assert decision["portfolio_evaluation_allowed"] is False
    assert decision["position_impact"] == 0

