from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.donchian_discovery_data_audit import (
    audit_all,
    audit_market,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load(
    (ROOT / "config" / "donchian_discovery_v1.yaml").read_text(encoding="utf-8")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_complete_fixture(tmp_path: Path) -> dict:
    registry = copy.deepcopy(REGISTRY)
    dates = pd.bdate_range("2012-05-28", periods=30)
    market_path = tmp_path / "market.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    distribution_path = tmp_path / "dividends.csv"
    attestation_path = tmp_path / "dividends_coverage.json"
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [4.0 + index * 0.01 for index in range(len(dates))],
            "high": [4.1 + index * 0.01 for index in range(len(dates))],
            "low": [3.9 + index * 0.01 for index in range(len(dates))],
            "close": [4.05 + index * 0.01 for index in range(len(dates))],
            "volume": [1_000_000] * len(dates),
            "amount": [4_000_000] * len(dates),
            "symbol": ["510300.SH"] * len(dates),
            "source": ["测试固定源"] * len(dates),
            "retrieved_at": ["2026-08-18T00:00:00+08:00"] * len(dates),
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": [4000.0 + index for index in range(len(dates))],
            "symbol": ["H00300"] * len(dates),
            "retrieved_at": ["2026-08-18T00:00:00+08:00"] * len(dates),
        }
    )
    distribution = pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": ["2012-12-17"],
            "ex_date": ["2012-12-18"],
            "payment_date": ["2012-12-24"],
            "cash_dividend_per_share": [0.033],
            "source": [
                "https://www.sse.com.cn/disclosure/fund/announcement/c/2013-03-27/510300_2012_n.pdf"
            ],
        }
    )
    market.to_parquet(market_path, index=False)
    benchmark.to_parquet(benchmark_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    attestation_path.write_text(
        json.dumps(
            {
                "complete_history_confirmed": True,
                "distribution_file_sha256": _sha256(distribution_path),
                "coverage_start": str(dates.min().date()),
                "coverage_end": str(dates.max().date()),
                "retrieved_at": "2026-08-18T00:00:00+08:00",
                "official_sources": ["https://www.sse.com.cn/"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    market_contract = registry["data_contracts"]["market"]
    market_contract.update(
        {
            "file": market_path.relative_to(tmp_path).as_posix(),
            "frozen_snapshot_sha256": _sha256(market_path),
            "required_first_date": str(dates.min().date()),
            "required_last_date": str(dates.max().date()),
            "minimum_rows": len(dates),
        }
    )
    benchmark_contract = registry["data_contracts"]["benchmark"]
    benchmark_contract.update(
        {
            "file": benchmark_path.relative_to(tmp_path).as_posix(),
            "frozen_snapshot_sha256": _sha256(benchmark_path),
            "required_first_date": str(dates.min().date()),
            "required_last_date": str(dates.max().date()),
            "minimum_rows": len(dates),
        }
    )
    distribution_contract = registry["data_contracts"]["distributions"]
    distribution_contract.update(
        {
            "file": distribution_path.relative_to(tmp_path).as_posix(),
            "frozen_snapshot_sha256": _sha256(distribution_path),
            "coverage_attestation_file": attestation_path.relative_to(
                tmp_path
            ).as_posix(),
            "coverage_start_required": str(dates.min().date()),
            "coverage_end_required": str(dates.max().date()),
        }
    )
    return registry


def test_registry_freezes_exactly_one_20_10_candidate() -> None:
    evidence = validate_registry(REGISTRY)
    assert evidence["candidate_count"] == 1
    assert evidence["candidate_ids"] == ["T0_DONCHIAN_20_10"]
    assert REGISTRY["candidate"]["entry_lookback_trading_days"] == 20
    assert REGISTRY["candidate"]["exit_lookback_trading_days"] == 10
    assert REGISTRY["candidate"]["parameter_search_allowed"] is False
    assert REGISTRY["candidate"]["parameter_neighbors_allowed"] is False


def test_governance_disables_all_live_and_rescue_paths() -> None:
    governance = REGISTRY["governance"]
    assert governance["modifies_r5"] is False
    assert governance["reopens_r6"] is False
    assert governance["creates_r7"] is False
    assert governance["article_macd_candidate_enabled"] is False
    assert governance["martin_candidate_enabled"] is False
    assert governance["divergence_candidate_enabled"] is False
    assert governance["position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert REGISTRY["protocol"]["true_forward_start"] is None


def test_complete_synthetic_data_contract_passes_without_return_calculation(
    tmp_path: Path,
) -> None:
    registry = _write_complete_fixture(tmp_path)
    result = audit_all(tmp_path, registry)
    assert result["status"] == "PASS"
    assert result["return_calculation_allowed"] is True
    assert result["position_mapping_enabled"] is False
    assert result["order_generation_enabled"] is False


def test_missing_known_2012_distribution_blocks_returns(tmp_path: Path) -> None:
    registry = _write_complete_fixture(tmp_path)
    distribution_contract = registry["data_contracts"]["distributions"]
    distribution_path = tmp_path / distribution_contract["file"]
    frame = pd.read_csv(distribution_path)
    frame.loc[:, "ex_date"] = "2013-01-18"
    frame.to_csv(distribution_path, index=False)
    distribution_contract["frozen_snapshot_sha256"] = _sha256(distribution_path)
    result = audit_all(tmp_path, registry)
    assert result["status"] == "NO_VIEW"
    assert result["return_calculation_allowed"] is False
    assert "BLOCKED_INCOMPLETE_DISTRIBUTION_HISTORY" in result["failure_categories"]


def test_invalid_ohlc_blocks_market_branch(tmp_path: Path) -> None:
    registry = _write_complete_fixture(tmp_path)
    market_contract = registry["data_contracts"]["market"]
    market_path = tmp_path / market_contract["file"]
    frame = pd.read_parquet(market_path)
    frame.loc[0, "high"] = frame.loc[0, "low"] - 0.01
    frame.to_parquet(market_path, index=False)
    market_contract["frozen_snapshot_sha256"] = _sha256(market_path)
    result = audit_market(tmp_path, market_contract)
    assert result["status"] == "BLOCKED_MARKET_DATA_CONTRACT"
    assert result["return_test_allowed"] is False
    assert result["invalid_ohlc_rows"] == 1


def test_enabling_parameter_search_invalidates_frozen_registry() -> None:
    registry = copy.deepcopy(REGISTRY)
    registry["candidate"]["parameter_search_allowed"] = True
    with pytest.raises(ValueError, match="参数搜索"):
        validate_registry(registry)
