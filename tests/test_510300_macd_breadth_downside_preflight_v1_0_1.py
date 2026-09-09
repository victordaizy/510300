from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import macd_breadth_downside_preflight_v1 as base
import macd_breadth_downside_preflight_v1_0_1 as corrected


def test_correction_changes_only_input_warmup_and_output_paths() -> None:
    original = base.load_config()
    config = corrected.load_config()
    assert config["dates"] == original["dates"]
    assert config["features"] == original["features"]
    assert config["outcomes"] == original["outcomes"]
    assert config["event_sampling"] == original["event_sampling"]
    assert config["bootstrap"] == original["bootstrap"]
    assert config["gates"] == original["gates"]
    assert config["scope"] == original["scope"]
    assert config["governance"] == original["governance"]
    assert (
        config["inputs"]["official_weighted_breadth"]["path"]
        == "data/features/000300_official_weighted_breadth_daily_v1_0_1_warmup.parquet"
    )


def test_repaired_breadth_preserves_evaluation_dates_and_fills_only_warmup_gap() -> None:
    original_path = ROOT / "data" / "features" / "000300_official_weighted_breadth_daily.parquet"
    repaired_path = (
        ROOT
        / "data"
        / "features"
        / "000300_official_weighted_breadth_daily_v1_0_1_warmup.parquet"
    )
    original = pd.read_parquet(original_path).sort_values("date").reset_index(drop=True)
    repaired = pd.read_parquet(repaired_path).sort_values("date").reset_index(drop=True)
    assert len(original) == len(repaired) == 1211
    assert pd.DatetimeIndex(original["date"]).equals(pd.DatetimeIndex(repaired["date"]))
    change = "official_weighted_above_ma20_share_change_5d"
    assert original[change].isna().sum() == 5
    assert repaired[change].isna().sum() == 0
    overlap = original[change].notna()
    assert (original.loc[overlap, change] == repaired.loc[overlap, change]).all()
    assert (repaired["weight_snapshot_date"] <= repaired["date"]).all()
    assert repaired["ma20_weight_coverage"].min() >= 0.99
    assert repaired["ma60_weight_coverage"].min() >= 0.99


def test_remediation_report_confirms_no_candidate_outcomes_read() -> None:
    path = (
        ROOT
        / "reports"
        / "data_quality"
        / "000300_official_weighted_breadth_warmup_v1_0_1.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS_DATA_REMEDIATION_ONLY_NO_CANDIDATE_RETURN_READ"
    assert report["candidate_outcomes_computed"] is False
    assert report["evaluation_row_count"] == 1211
    assert report["evaluation_b20_change_missing_rows"] == 0
    assert report["maximum_existing_change_difference"] == 0.0


def test_manifest_hashes_validate_after_freeze() -> None:
    if not corrected.MANIFEST_PATH.exists():
        return
    config = corrected.load_config()
    manifest = json.loads(corrected.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == corrected.sha256_file(corrected.CONFIG_PATH)
    assert corrected.validate_manifest(config)["hash_mismatches"] == {}
