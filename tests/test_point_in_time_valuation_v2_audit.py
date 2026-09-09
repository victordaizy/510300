"""沪深300点时估值V2可重建性审计测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.point_in_time_valuation_v2_audit import (
    ROOT,
    _weighted_coverage,
    audit_hashes,
    audit_r5_replay,
    canonical_directory_hash,
    load_config,
    summarize_history_gate,
)


REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_point_in_time_valuation_v2_reconstructibility.json"
COVERAGE_FILE = ROOT / "data" / "audit" / "000300_point_in_time_valuation_v2_snapshot_coverage.parquet"


def test_审计配置强制关闭收益_ic_仓位和订单(tmp_path: Path) -> None:
    config = load_config()
    for field in (
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    ):
        assert config["protocol"][field] is False
    config["protocol"]["return_calculation_enabled"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="禁止项"):
        load_config(invalid)


def test_权重覆盖率保留供应商权重和而非强制归一化() -> None:
    weights = pd.DataFrame(
        {"con_code": ["A", "B", "C"], "weight": [50.0, 30.0, 20.01]}
    )
    assert _weighted_coverage(weights, {"A", "B"}) == pytest.approx(0.80)
    assert _weighted_coverage(weights, {"A", "B", "C"}) == pytest.approx(1.0001)


def _synthetic_coverage() -> pd.DataFrame:
    dates = pd.date_range("2016-08-31", periods=60, freq="ME")
    data = pd.DataFrame(
        {
            "date": dates,
            "calendar_month": dates.to_period("M").astype(str),
            "price_weight_coverage": 1.0,
            "ttm_metric_weight_coverage": 0.99,
            "normalized_metric_weight_coverage": 0.98,
            "raw_ey_ready_weight_coverage": 0.99,
            "normalized_ey_ready_weight_coverage": 0.98,
            "cgb_10y_available_exact_date": True,
            "before_first_evaluation": True,
        }
    )
    data.loc[0, "price_weight_coverage"] = 0.0
    data.loc[0, "raw_ey_ready_weight_coverage"] = 0.0
    data.loc[0, "normalized_metric_weight_coverage"] = 0.0
    data.loc[0, "normalized_ey_ready_weight_coverage"] = 0.0
    return data


def test_历史闸门不会把权重完整误判为估值可运行() -> None:
    result = summarize_history_gate(_synthetic_coverage(), load_config())
    five = result["five_year"]
    seven = result["seven_year"]
    assert five["local_weight_snapshot_count"] == 60
    assert five["missing_weight_months"] == []
    assert five["raw_ey_status"] == "PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED"
    assert five["normalized_ey_status"] == "PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED"
    assert seven["missing_weight_months"][:2] == ["2014-08", "2014-09"]
    assert len(seven["missing_weight_months"]) == 24
    assert seven["raw_ey_status"] == "BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED"


def test_冻结输入哈希全部一致() -> None:
    result = audit_hashes(load_config())
    assert result["status"] == "PASS"
    assert result["checked_file_count"] >= 10


def test_132个财务检查点目录指纹与报告一致() -> None:
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    digest, files = canonical_directory_hash(
        ROOT / "data" / "raw" / "fundamentals" / "vip_checkpoints"
    )
    assert len(files) == 132
    assert digest == report["datasets"]["financials"]["checkpoint_content_sha256"]
    assert all(
        report["datasets"]["financials"]["checkpoint_by_api"][api]["file_count"] == 44
        for api in ("income_vip", "balancesheet_vip", "fina_indicator_vip")
    )


def test_R5信号只做工程重放且逐字段精确匹配() -> None:
    config = load_config()
    contracts = config["data_contracts"]
    read = lambda name: pd.read_parquet(ROOT / contracts[name]["file"])
    result = audit_r5_replay(
        read("vendor_valuation"),
        read("pre_tushare_constituent_backup"),
        read("current_constituent_daily"),
        read("index_daily"),
        read("preserved_r5_enhanced_signals"),
        config,
    )
    assert result["engineering_replay_status"] == "PASS_ENGINEERING_REPLAY_ONLY"
    assert result["comparison_rows"] == 1211
    assert result["calendar_exact_match"] is True
    assert all(
        item["maximum_absolute_difference"] <= 1e-12
        and item["missingness_mismatch_count"] == 0
        for item in result["numeric_field_errors"].values()
    )
    assert all(value == 0 for value in result["boolean_field_mismatch_counts"].values())
    assert result["algebraic_market_cap_cancellation"]["status"] == "PROVEN"
    assert result["algebraic_market_cap_cancellation"]["maximum_absolute_error"] <= 1e-12
    assert result["economic_evidence_status"].startswith("NO_VIEW")
    assert result["return_or_position_evaluation_performed"] is False


def test_产物明确记录40个月价格缺口和9个月标准化缺口() -> None:
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    coverage = pd.read_parquet(COVERAGE_FILE)
    five = report["history_gates"]["five_year"]
    assert report["overall_status"] == "PARTIALLY_RECONSTRUCTIBLE_LOCAL_ACQUISITION_REQUIRED"
    assert report["formal_valuation_run_status"] == "NO_VIEW"
    assert len(coverage) == 120
    assert len(five["price_coverage_failed_months"]) == 40
    assert five["price_coverage_failed_months"] == [
        str(period) for period in pd.period_range("2016-08", "2019-11", freq="M")
    ]
    assert len(five["normalized_coverage_failed_months"]) == 9
    assert five["normalized_coverage_failed_months"][-1] == "2017-04"
    assert five["ttm_coverage_failed_months"] == []
    assert five["cgb_10y_exact_date_failed_months"] == []
    assert report["governance"] == {
        "return_calculation_performed": False,
        "ic_calculation_performed": False,
        "position_mapping_performed": False,
        "order_generation_performed": False,
        "broker_connection_performed": False,
        "frozen_files_mutated": False,
    }

