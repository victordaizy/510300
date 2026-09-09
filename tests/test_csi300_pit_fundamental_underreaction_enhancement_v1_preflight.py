from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from research.csi300_pit_fundamental_underreaction_enhancement_v1_preflight import (
    FORBIDDEN_PRE_RETURN_MARKERS,
    PRE_RETURN_READ_PATHS,
    build_target_event_inventory,
    evaluate_fact_gate,
    first_open_strictly_after,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "csi300_pit_fundamental_underreaction_enhancement_v1.yaml"


def test_first_open_is_strictly_after_publication_date() -> None:
    events = pd.Series(pd.to_datetime(["2026-08-07", "2026-08-08", "2026-08-14"]))
    opens = pd.to_datetime(["2026-08-07", "2026-08-10", "2026-08-14"])
    result = first_open_strictly_after(events, opens)
    assert result.iloc[0] == pd.Timestamp("2026-08-10")
    assert result.iloc[1] == pd.Timestamp("2026-08-10")
    assert pd.isna(result.iloc[2])


def test_target_inventory_uses_next_open_membership_and_pit_industry() -> None:
    events = pd.DataFrame(
        {
            "announcement_id": ["A", "B", "C"],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "report_period": ["2025-12-31"] * 3,
            "period_type": ["FY"] * 3,
            "event_publication_date": ["2026-04-24", "2026-04-25", "2026-04-24"],
            "official_timestamp_at": ["2026-04-24T00:00:00+08:00"] * 3,
            "official_pdf_url": [
                "https://static.cninfo.com.cn/finalpage/2026-04-24/A.PDF",
                "https://static.cninfo.com.cn/finalpage/2026-04-25/B.PDF",
                "https://static.cninfo.com.cn/finalpage/2026-04-24/C.PDF",
            ],
        }
    )
    membership = pd.DataFrame(
        {
            "membership_date": ["2026-04-27", "2026-04-27"],
            "symbol": ["000001.SZ", "000002.SZ"],
        }
    )
    industries = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "industry_l1": ["制造", "银行"],
            "industry_l1_code": ["801120.SI", "801780.SI"],
            "valid_from": ["2020-01-01", "2020-01-01"],
            "valid_to": [None, None],
            "available_at": ["2020-01-01 15:00:00", "2020-01-01 15:00:00"],
        }
    )
    inventory, metrics = build_target_event_inventory(
        events,
        membership,
        industries,
        event_start="2018-01-01",
        event_end="2026-08-14",
        allowed_period_types=["Q1", "H1", "Q3", "FY"],
        excluded_industry_codes=["801780.SI", "801790.SI"],
    )
    assert inventory["announcement_id"].tolist() == ["A", "B"]
    assert inventory["membership_date"].nunique() == 1
    assert inventory["membership_date"].iloc[0] == pd.Timestamp("2026-04-27")
    assert metrics["pit_csi300_member_event_count"] == 2
    assert metrics["official_fact_target_event_count"] == 1
    assert inventory.loc[inventory["announcement_id"] == "B", "inventory_status"].item() == (
        "PRE_REGISTERED_FINANCIAL_INDUSTRY_EXCLUSION"
    )


def test_missing_official_fact_artifacts_block_return_evaluation() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inventory = pd.DataFrame(
        {
            "announcement_id": ["A"],
            "included_in_phase1_fact_target": [True],
            "publication_year": [2025],
            "industry_l1_code": ["801120.SI"],
        }
    )
    gate = evaluate_fact_gate(config, inventory)
    assert gate["gate_id"] == "G6_OFFICIAL_PDF_VERIFIED_FACT_ARCHIVE_PASS"
    assert gate["passed"] is False
    assert gate["observed"] == "MISSING_OFFICIAL_FACT_ARTIFACTS"


def test_preflight_allowlist_contains_no_market_or_future_return_path() -> None:
    for path in PRE_RETURN_READ_PATHS:
        assert not any(marker.lower() in path.lower() for marker in FORBIDDEN_PRE_RETURN_MARKERS)


def test_protocol_forbids_phase_1_portfolio_and_strategy_sharpe() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["governance"]["portfolio_construction_allowed"] is False
    assert config["governance"]["strategy_sharpe_allowed"] is False
    assert config["governance"]["official_fact_data_gate_must_pass_before_price_read"] is True
    assert config["phase_1_evaluation"]["portfolio_return"] == "NOT_CALCULATED"
    assert config["phase_1_evaluation"]["strategy_sharpe"] == "NOT_CALCULATED"
