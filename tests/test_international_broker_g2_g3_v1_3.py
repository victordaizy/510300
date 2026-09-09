from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.international_broker_g2_g3_reconstruction_v1_3 import (
    _validate_disclosures,
    build_entity_boundary,
    build_profit_bridge,
    load_config,
    verify_official_sources,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_tables() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(ROOT / "config" / "international_broker_g2_g3_v1_3.yaml")
    sources, lookup = verify_official_sources(ROOT, config)
    disclosures = _validate_disclosures(config, lookup)
    entity = build_entity_boundary(config, lookup, disclosures)
    bridge = build_profit_bridge(entity, config)
    return config, sources, entity, bridge


def test_all_30_official_sources_pass_hash_and_authority(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, sources, _, _ = built_tables
    assert len(sources) == 30
    assert sources["source_status"].eq("PASS").all()
    assert sources["authority_level"].eq("PRIMARY").all()
    assert sources["file_sha256"].eq(sources["actual_sha256"]).all()


def test_profit_bridge_is_exact_six_by_five_matrix(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    config, _, entity, bridge = built_tables
    expected = {
        (ticker, int(year))
        for ticker in config["protocol"]["matrix_tickers"]
        for year in config["protocol"]["matrix_years"]
    }
    assert set(zip(bridge["ticker"], bridge["year"])) == expected
    assert len(bridge) == 30
    assert len(entity) == 35
    assert not bridge[["ticker", "year"]].duplicated().any()


def test_g3_has_no_completed_company_year_and_no_inferred_split(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, _, bridge = built_tables
    assert bridge["g3_complete"].eq(False).all()
    for field in (
        "recurring_client_driven_profit",
        "market_sensitive_profit",
        "one_off_profit",
    ):
        assert bridge[field].isna().all()
    assert bridge["view_status"].eq("NO_VIEW").all()
    assert bridge["authorization"].eq("RESEARCH_ONLY_NO_POSITION_CHANGE").all()


def test_guotai_haitong_2025_keeps_two_entities_without_aggregation(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, entity, bridge = built_tables
    entity_rows = entity.loc[
        entity["ticker"].eq("601211.SH") & entity["year"].eq(2025)
    ]
    assert set(entity_rows["entity_name"]) == {
        "国泰海通金融控股有限公司",
        "海通国际控股有限公司",
    }
    haitong = entity_rows.loc[entity_rows["entity_name"].eq("海通国际控股有限公司")]
    assert haitong["partial_consolidation_period"].eq(True).all()
    company_year = bridge.loc[
        bridge["ticker"].eq("601211.SH") & bridge["year"].eq(2025)
    ].iloc[0]
    assert pd.isna(company_year["international_entity_net_profit"])
    assert company_year["stop_status"] == (
        "BLOCKED_G3_NO_UNDISCLOSED_ENTITY_AGGREGATION"
    )


def test_guotai_haitong_historical_candidates_remain_separate_and_unverified(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, entity, bridge = built_tables
    historical = entity.loc[
        entity["ticker"].eq("601211.SH") & entity["year"].between(2021, 2024)
    ]
    assert len(historical) == 8
    for year in range(2021, 2025):
        rows = historical.loc[historical["year"].eq(year)]
        assert set(rows["entity_name"]) == {
            "国泰君安金融控股有限公司",
            "海通国际控股有限公司",
        }
    assert historical["consolidated_flag"].isna().all()
    assert historical["direct_ownership_pct"].isna().all()
    bridge_rows = bridge.loc[
        bridge["ticker"].eq("601211.SH") & bridge["year"].between(2021, 2024)
    ]
    assert bridge_rows["stop_status"].eq(
        "BLOCKED_G2_DUAL_PLATFORM_BOUNDARY_UNVERIFIED"
    ).all()


def test_boci_is_negative_control_for_all_five_years(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, entity, _ = built_tables
    rows = entity.loc[entity["ticker"].eq("601696.SH")]
    assert len(rows) == 5
    assert rows["consolidated_flag"].eq(False).all()
    assert rows["direct_ownership_pct"].isna().all()
    assert rows["indirect_ownership_pct"].isna().all()
    assert rows["shareholder_not_subsidiary_pct"].eq(33.42).all()
    assert rows["international_entity_net_profit"].isna().all()


def test_protocol_forbids_g6_returns_and_510300_input(
    built_tables: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    config, _, _, _ = built_tables
    governance = config["governance"]
    assert governance["allow_g6_valuation"] is False
    assert governance["allow_return_conclusion"] is False
    assert governance["allow_510300_input"] is False
    assert governance["allow_position_change"] is False
    assert governance["allow_order_generation"] is False


def test_v13_inputs_and_outputs_contain_no_replacement_character() -> None:
    paths = [
        ROOT / "config" / "international_broker_g2_g3_v1_3.yaml",
        ROOT / "data" / "processed" / "international_broker" / "v1_3" / "entity_consolidation_boundary.csv",
        ROOT / "data" / "processed" / "international_broker" / "v1_3" / "profit_bridge_2021_2025.csv",
        ROOT / "reports" / "research" / "international_broker_g2_g3_v1_3_status.json",
        ROOT / "reports" / "research" / "international_broker_g2_g3_v1_3_status.md",
    ]
    for path in paths:
        assert "\ufffd" not in path.read_text(encoding="utf-8"), path
