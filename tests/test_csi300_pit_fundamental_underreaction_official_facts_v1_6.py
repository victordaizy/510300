from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import yaml

from research import csi300_pit_fundamental_underreaction_official_facts_v1_6 as parser


ROOT = Path(__file__).resolve().parents[1]


def _empty_result() -> dict[str, object]:
    return {
        "metrics": [],
        "missing_metrics": list(parser.REQUIRED_METRICS),
        "document_complete": False,
        "document_status": "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE",
        "parser_version": parser.PARSER_VERSION,
    }


def test_ocr_number_normalization_repairs_decimal_and_separator_glyphs() -> None:
    assert parser._normalize_ocr_number("663，524，805，12") == Decimal(
        "663524805.12"
    )
    assert parser._normalize_ocr_number("1，949》548》273．48") == Decimal(
        "1949548273.48"
    )
    assert parser._normalize_ocr_number("1320，87乙456．53") == Decimal(
        "1320872456.53"
    )


def test_low_text_image_statement_run_requires_audit_and_note_context() -> None:
    page_texts = [
        "年度报告",
        "审计报告 我们审计了后附财务报表，管理层负责按照企业会计准则编制并公允列报财务报表。",
        "审计意见 我们认为财务报表在所有重大方面公允反映了公司财务状况和经营成果。",
        "页眉",
        "页眉",
        "页眉",
        "页眉",
        "页眉",
        "财务报表附注 一、公司基本情况 本公司依法设立并持续经营，记账本位币为人民币。",
    ]
    assert parser.find_image_statement_pages(page_texts) == (3, 4, 5, 6, 7)


def test_inverse_core_profit_label_is_admitted_from_exact_summary_row() -> None:
    result = _empty_result()
    pages = [
        (
            "二、会计数据和财务指标摘要（千元）\n"
            "项目 2016年1-6月 2015年1-6月\n"
            "扣除非经常性损益后归属于上市公司股东的净利润 "
            "2,992,606 2,815,116 6.30%"
        )
    ]
    receipt = parser._add_inverse_core_parent_profit(
        result,
        pages,
        period_type="H1",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert receipt["status"] == "PASS_EXACT_INVERSE_CORE_LABEL_ADMITTED"
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 2992606000.0


def test_net_receivable_requires_current_and_prior_reconciliation() -> None:
    result = _empty_result()
    page_texts = [
        (
            "合并资产负债表 单位：千元 流动资产 货币资金 1 18,972,333 "
            "33,407,879 应收票据及应收账款 4 22,181,067 26,398,228 "
            "存货 26,316,928 26,234,139 资产总计 120,709,368 143,962,023"
        ),
        (
            "财务报表附注 单位：千元 五、合并财务报表主要项目注释 "
            "4.应收票据及应收账款 2018年6月30日 2017年12月31日 "
            "应收票据 1,389,766 2,052,945 应收账款 31,518,729 33,488,333 "
            "减：坏账准备 10,727,428 9,143,050"
        ),
    ]
    receipt = parser._add_net_receivable_reconciliation(result, page_texts)
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert receipt["status"] == (
        "PASS_NET_ACCOUNTS_RECEIVABLE_RECONCILED_AND_ADMITTED"
    )
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 20791301000.0


def test_enhanced_ocr_observation_keeps_exact_page_and_value() -> None:
    records = [
        {
            "page_index": 8,
            "render_scale": 3.0,
            "pixel_width": 2400,
            "pixel_height": 3400,
            "lines": [
                {
                    "words": [
                        {
                            "text": "应收账款",
                            "x": 100,
                            "y": 200,
                            "width": 180,
                            "height": 40,
                        },
                        {
                            "text": "663，524，805，12",
                            "x": 900,
                            "y": 200,
                            "width": 360,
                            "height": 40,
                        },
                    ]
                }
            ],
        }
    ]
    observed = parser._ocr_observations(records)["ACCOUNTS_RECEIVABLE_END"]
    assert len(observed) == 1
    assert observed[0].page_index == 8
    assert observed[0].value == Decimal("663524805.12")


def test_v1_6_replay_receipt_passes_nonregression_and_three_gap_gates() -> None:
    receipt_path = ROOT / (
        "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
        "v1_6_gap_replay_sources/replay_results_v1_6/receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["parser_version"] == parser.PARSER_VERSION
    assert receipt["target_count"] == 25
    assert receipt["v1_5_nonregression_target_count"] == 22
    assert receipt["v1_6_gap_target_count"] == 3
    assert receipt["complete_count"] == 25
    assert receipt["authoritative_metric_value_match_count"] == 25
    assert receipt["targeted_receipt_pass_count"] == 3
    assert receipt["all_nine_metric_replay_and_authoritative_value_match_passed"] is True


def test_v1_6_config_and_collector_use_isolated_artifacts() -> None:
    config_path = (
        ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_6.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["extraction"]["parser_version"] == parser.PARSER_VERSION
    assert config["artifacts"]["checkpoint_root"].endswith("/checkpoints_v1_6")
    assert config["artifacts"]["checkpoint_root"] != config["supersedes"][
        "legacy_checkpoint_root"
    ]
    assert config["artifacts"]["receipt"].endswith("/receipt_v1_6.json")
    assert config["artifacts"]["facts"].endswith(
        "/official_financial_facts_v1_6.parquet"
    )

    from scripts import (
        collect_csi300_pit_fundamental_underreaction_official_facts_v1_6 as collector,
    )

    collector._patch_base_collector()
    assert collector._base.PARSER_VERSION == parser.PARSER_VERSION
    assert collector._base.CONFIG_PATH == collector.CONFIG_PATH
    assert collector._base.process_document is collector.process_document


def test_complete_checkpoint_migration_restamps_without_changing_values() -> None:
    from scripts import (
        migrate_csi300_pit_official_facts_v1_5_complete_checkpoints_to_v1_6 as migration,
    )

    metrics = [
        {
            "metric_id": metric_id,
            "metric_value_cny": float(index + 1),
            "source_locator": json.dumps(
                {"parser_version": "V1_5_TEST", "page": index + 1},
                ensure_ascii=False,
            ),
        }
        for index, metric_id in enumerate(parser.REQUIRED_METRICS)
    ]
    source = {
        "parser_version": "V1_5_TEST",
        "checkpoint_status": "PARSED_COMPLETE",
        "document_complete": True,
        "missing_metrics": [],
        "metrics": metrics,
    }
    migrated = migration._restamp_complete_checkpoint(
        source,
        source_path=ROOT / "tmp/checkpoints_v1_5/2018/123.json.gz",
        source_sha256="a" * 64,
    )
    assert migrated["parser_version"] == parser.PARSER_VERSION
    assert [row["metric_value_cny"] for row in migrated["metrics"]] == [
        row["metric_value_cny"] for row in metrics
    ]
    assert all(
        json.loads(row["source_locator"])["parser_version"] == parser.PARSER_VERSION
        for row in migrated["metrics"]
    )
    receipt = migrated["v1_6_checkpoint_migration_receipt"]
    assert receipt["source_checkpoint_sha256"] == "a" * 64
    assert receipt["metric_count"] == 9
