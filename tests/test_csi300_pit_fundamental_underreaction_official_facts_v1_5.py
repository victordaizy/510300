from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import yaml

from research import csi300_pit_fundamental_underreaction_official_facts_v1_5 as parser


ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_4_replay_sources"
)


def test_quarterly_contamination_contract_is_hash_bound_and_exact() -> None:
    documents = parser.quarterly_contamination_corrections()
    assert len(documents) == 2
    assert sum(len(row["corrected_metrics"]) for row in documents.values()) == 5
    for pdf_sha256, document in documents.items():
        pdf_path = REPLAY_ROOT / (
            f"{document['announcement_id']}__{document['announcement_id']}.pdf"
        )
        assert pdf_path.exists()
        assert len(pdf_sha256) == 64
        assert pdf_path.stat().st_size == int(document["official_pdf_size_bytes"])


def test_signalway_fy_uses_annual_ytd_instead_of_first_quarter() -> None:
    content = (REPLAY_ROOT / "1202063757__1202063757.pdf").read_bytes()
    result = parser.extract_official_pdf_facts(content, period_type="FY")
    values = {
        row["metric_id"]: Decimal(str(row["metric_value_cny"]))
        for row in result["metrics"]
    }
    assert result["document_complete"] is True
    assert values["OPERATING_REVENUE_YTD"] == Decimal("3477692868.18")
    assert values["PARENT_NET_PROFIT_YTD"] == Decimal("1265920025.79")
    assert values["OPERATING_CASH_FLOW_YTD"] == Decimal("1056028826.67")
    assert result["v1_5_deferred_expensive_paths_receipt"][
        "full_v1_4_path_executed"
    ] is False
    assert result["v1_5_quarterly_contamination_guard_receipt"]["status"].startswith(
        "PASS_"
    )


def test_full_v1_4_path_runs_only_after_incomplete_primary(monkeypatch) -> None:
    primary = {
        "metrics": [],
        "missing_metrics": list(parser.REQUIRED_METRICS),
        "document_complete": False,
        "document_status": "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE",
        "parser_version": parser.PARSER_VERSION,
    }
    full_metrics = [
        {
            "metric_id": metric_id,
            "metric_value_cny": float(index + 1),
            "source_locator": "{}",
        }
        for index, metric_id in enumerate(parser.REQUIRED_METRICS)
    ]
    full = {
        "metrics": full_metrics,
        "missing_metrics": [],
        "document_complete": True,
        "document_status": "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED",
        "parser_version": "V1_4_TEST",
    }
    monkeypatch.setattr(parser, "_primary_semantic_result", lambda *args, **kwargs: primary)
    monkeypatch.setattr(
        parser._v1_4,
        "extract_official_pdf_facts",
        lambda *args, **kwargs: full,
    )
    result = parser.extract_official_pdf_facts(b"%PDF-test", period_type="FY")
    receipt = result["v1_5_deferred_expensive_paths_receipt"]
    assert result["document_complete"] is True
    assert receipt["full_v1_4_path_executed"] is True
    assert receipt["primary_metric_count"] == 0
    assert result["parser_version"] == parser.PARSER_VERSION


def test_replay_receipt_passes_authoritative_gate_with_only_registered_differences() -> None:
    receipt = json.loads(
        (REPLAY_ROOT / "replay_results_v1_5/receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["target_count"] == 22
    assert receipt["complete_count"] == 22
    assert receipt["authoritative_metric_value_match_count"] == 22
    assert receipt["exact_v1_4_metric_value_match_count"] == 20
    assert receipt["quarterly_contamination_corrected_document_count"] == 2
    assert receipt["quarterly_contamination_corrected_metric_count"] == 5
    assert receipt["all_nine_metric_replay_and_authoritative_value_match_passed"] is True
    for row in receipt["results"]:
        for difference in row["metric_differences_from_v1_4"].values():
            assert difference["registered_quarterly_contamination_correction"] is True


def test_v1_5_config_and_collector_use_isolated_checkpoint_root() -> None:
    config_path = (
        ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_5.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["extraction"]["parser_version"] == parser.PARSER_VERSION
    assert config["artifacts"]["checkpoint_root"].endswith("/checkpoints_v1_5")
    assert config["artifacts"]["checkpoint_root"] != config["supersedes"][
        "legacy_checkpoint_root"
    ]
    from scripts import (
        collect_csi300_pit_fundamental_underreaction_official_facts_v1_5 as collector,
    )

    collector._patch_base_collector()
    assert collector._base.PARSER_VERSION == parser.PARSER_VERSION
    assert collector._base.CONFIG_PATH == collector.CONFIG_PATH
    assert collector._base.process_document is collector.process_document
