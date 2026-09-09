from __future__ import annotations

from pathlib import Path

from research.international_broker_subsidiary_map import (
    build_subsidiary_map,
    load_subsidiary_map_config,
    verify_document_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_subsidiary_map_config()


def test_subsidiary_map_preserves_research_only_boundary() -> None:
    assert CONFIG["governance"]["allow_return_test"] is False
    assert CONFIG["governance"]["allow_510300_input"] is False
    assert CONFIG["governance"]["allow_position_change"] is False


def test_g2_document_prerequisite_passes() -> None:
    result = verify_document_prerequisite(ROOT, CONFIG)
    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_initial_map_has_five_platform_parents_and_one_negative_control() -> None:
    table, failures = build_subsidiary_map(ROOT, CONFIG)
    assert failures == []
    confirmed = table["platform_status"].eq(
        "CONFIRMED_CONSOLIDATED_OVERSEAS_PLATFORM"
    )
    no_platform = table["platform_status"].str.startswith("NO_CONSOLIDATED")
    assert table.loc[confirmed, "parent_ticker"].nunique() == 5
    assert table.loc[no_platform, "parent_ticker"].tolist() == ["601696.SH"]


def test_haitong_platform_negative_net_assets_are_not_lost() -> None:
    table, _ = build_subsidiary_map(ROOT, CONFIG)
    row = table.loc[table["subsidiary_name"].eq("海通国际控股有限公司")].iloc[0]
    assert float(row["net_assets"]) == -155.32
    assert float(row["net_profit"]) == -32.68
    assert bool(row["acquisition_flag"]) is True
