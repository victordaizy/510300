from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.industry_expectation_gap_frozen_input_recovery import (
    FrozenInputRecoveryError,
    reconstruct_json_bytes_from_log,
    validate_recovered_readiness_semantics,
    validate_recovery_config,
    verify_manifest_with_declared_recoveries,
)


def _snapshot() -> dict:
    return {
        "forward_readiness": {
            "status": "COLLECTING",
            "observed_trading_days": 4,
            "full_coverage_days": 2,
            "minimum_required_days": 20,
            "recommended_days": 40,
            "eligible_for_research_evaluation": False,
            "eligible_for_position_mapping": False,
        }
    }


def _readiness() -> dict:
    return {
        "generated_at": "2026-08-18T15:01:00.111590+08:00",
        "status": "COLLECTING",
        "observed_trading_days": 4,
        "full_coverage_days": 2,
        "minimum_full_coverage_days": 20,
        "recommended_full_coverage_days": 40,
        "eligible_for_research_evaluation": False,
        "eligible_for_position_mapping": False,
    }


def test_reconstructs_last_marked_json_with_crlf(tmp_path: Path) -> None:
    payload = _readiness()
    log = tmp_path / "collector.log"
    log.write_text(
        "前一条\n"
        + json.dumps({"generated_at": "早期"}, ensure_ascii=False, indent=2)
        + "\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    result = reconstruct_json_bytes_from_log(
        log,
        marker="2026-08-18T15:01:00.111590+08:00",
        newline="CRLF",
    )
    assert b"\r\n" in result
    assert json.loads(result.decode("utf-8")) == payload


def test_semantics_must_match_frozen_snapshot() -> None:
    result = validate_recovered_readiness_semantics(_readiness(), _snapshot())
    assert result["full_coverage_days"] == 2
    broken = _readiness()
    broken["full_coverage_days"] = 3
    with pytest.raises(FrozenInputRecoveryError, match="语义"):
        validate_recovered_readiness_semantics(broken, _snapshot())


def test_manifest_accepts_only_declared_exact_recovery(tmp_path: Path) -> None:
    target = tmp_path / "mutable.json"
    target.write_text("当前版本", encoding="utf-8")
    recovered_payload = _readiness()
    recovered_bytes = json.dumps(
        recovered_payload, ensure_ascii=False, indent=2
    ).replace("\n", "\r\n").encode("utf-8")
    recovered = tmp_path / "recovered.json"
    recovered.write_bytes(recovered_bytes)
    log = tmp_path / "collector.log"
    log.write_text(
        json.dumps(recovered_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(_snapshot(), ensure_ascii=False), encoding="utf-8")
    expected = hashlib.sha256(recovered_bytes).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frozen_files": [
                    {
                        "path": "mutable.json",
                        "sha256": expected,
                        "bytes": len(recovered_bytes),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    recovery = {
        "target_path": "mutable.json",
        "expected_sha256": expected,
        "expected_bytes": len(recovered_bytes),
        "recovered_path": "recovered.json",
        "source_log_path": "collector.log",
        "source_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "source_marker": "2026-08-18T15:01:00.111590+08:00",
        "newline": "CRLF",
        "frozen_semantic_snapshot_path": "snapshot.json",
        "frozen_semantic_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    result = verify_manifest_with_declared_recoveries(
        tmp_path,
        manifest_relative_path="manifest.json",
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        recoveries=[recovery],
    )
    assert result["recovered_target_count"] == 1

    recovery["expected_sha256"] = "0" * 64
    with pytest.raises(FrozenInputRecoveryError, match="原清单"):
        verify_manifest_with_declared_recoveries(
            tmp_path,
            manifest_relative_path="manifest.json",
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            recoveries=[recovery],
        )


def test_production_recovery_config_stays_non_trading() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config" / "industry_expectation_gap_forward_evaluation_v1_1_recovery.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_recovery_config(config)
    assert config["constraints"]["partial_forward_return_read"] is False
    assert config["safety"]["position_mapping_enabled"] is False
    assert config["safety"]["order_generation_enabled"] is False


def test_production_recovered_file_matches_original_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    recovered = (
        root
        / "paper"
        / "industry_expectation_gap_v1"
        / "frozen_inputs"
        / "2026-08-18_primary_market_readiness.json"
    )
    assert recovered.stat().st_size == 967
    assert hashlib.sha256(recovered.read_bytes()).hexdigest() == (
        "1d4432ce581cf63f449315ef4932230fccc7e15a09affcac9e2662121fa5d2d8"
    )
