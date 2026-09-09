"""上交所官方股票期权历史快照导入器测试。"""

from __future__ import annotations

import json
import copy
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.sse_option_historical_snapshot_import_v1 import (
    extract_levels,
    normalize_sse_snapshot,
    validate_license_evidence,
    validate_license_evidence_at_root,
    validate_input_path,
    validate_master_path,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(
        (ROOT / "config" / "sse_option_historical_snapshot_import_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def synthetic_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_code": ["1001.SH", "1002.SH"],
            "underlying_code": ["510300.SH", "510300.SH"],
            "option_type": ["C", "P"],
            "list_date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "expiry_date": pd.to_datetime(["2026-09-23", "2026-09-23"]),
            "delist_date": pd.to_datetime(["2026-09-23", "2026-09-23"]),
            "strike": [4.8, 4.8],
            "contract_unit": [10000, 10000],
            "is_adjusted": [False, False],
        }
    )


def synthetic_raw() -> pd.DataFrame:
    rows = []
    for code, bid, ask in (("1001", 0.10, 0.11), ("1002", 0.12, 0.13)):
        for timestamp, offset in (("20260819145910", -0.01), ("20260819145959", 0.0)):
            rows.append(
                {
                    "SecurityID": code,
                    "DateTime": timestamp,
                    "PreClosePx": 0.10,
                    "OpenPx": 0.10,
                    "HighPx": 0.15,
                    "LowPx": 0.08,
                    "LastPx": 0.105,
                    "TotalLongPosition": 1000,
                    "TotalVolumeTrade": 200,
                    "TotalValueTrade": 20000,
                    "BidPrice[5]": f"{bid + offset},0.09,0.08,0.07,0.06",
                    "BidOrderQty[5]": "10,9,8,7,6",
                    "OfferPx[5]": f"{ask + offset},0.14,0.15,0.16,0.17",
                    "OfferQty[5]": "11,12,13,14,15",
                    "PhaseCode": "U  1",
                    "AvgPx": 0.105,
                    "PreSettlePx": 0.10,
                    "SettlePx": 0.105,
                }
            )
    return pd.DataFrame(rows)


def test_official_array_columns_are_expanded_without_position_guessing() -> None:
    raw = synthetic_raw()
    result = extract_levels(raw, "BidPrice", 5)
    assert result.columns.tolist() == [
        "BidPrice1",
        "BidPrice2",
        "BidPrice3",
        "BidPrice4",
        "BidPrice5",
    ]
    assert result.iloc[-1, 0] == pytest.approx(0.12)


def test_explicit_five_level_columns_are_supported() -> None:
    raw = pd.DataFrame({f"OfferPx_{level}": [level / 100] for level in range(1, 6)})
    result = extract_levels(raw, "OfferPx", 5)
    assert result.iloc[0].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])


def test_incomplete_expanded_columns_fail_closed() -> None:
    raw = pd.DataFrame({"BidPrice1": [0.1], "BidPrice2": [0.09]})
    with pytest.raises(ValueError, match="展开列不完整"):
        extract_levels(raw, "BidPrice", 5)


def test_normalizer_selects_latest_close_window_snapshot_and_full_universe(
    config: dict,
) -> None:
    normalized, audit = normalize_sse_snapshot(
        synthetic_raw(), synthetic_master(), pd.Timestamp("2026-08-19"), config
    )
    assert audit["status"] == "PASS"
    assert audit["active_contract_coverage"] == 1.0
    assert len(normalized) == 2
    assert normalized["snapshot_timestamp"].dt.strftime("%H:%M:%S").eq("14:59:59").all()
    assert normalized.set_index("contract_code").loc["1001.SH", "bid1"] == pytest.approx(0.10)
    assert normalized.set_index("contract_code").loc["1002.SH", "ask1"] == pytest.approx(0.13)


def test_same_second_uses_last_source_row_deterministically(config: dict) -> None:
    raw = synthetic_raw()
    duplicate = raw.loc[
        raw["SecurityID"].eq("1001") & raw["DateTime"].eq("20260819145959")
    ].copy()
    duplicate["BidPrice[5]"] = "0.109,0.09,0.08,0.07,0.06"
    duplicate["OfferPx[5]"] = "0.119,0.14,0.15,0.16,0.17"
    raw = pd.concat([raw, duplicate], ignore_index=True)
    normalized, audit = normalize_sse_snapshot(
        raw, synthetic_master(), pd.Timestamp("2026-08-19"), config
    )
    assert audit["status"] == "PASS"
    selected = normalized.set_index("contract_code").loc["1001.SH"]
    assert selected["bid1"] == pytest.approx(0.109)
    assert selected["source_row_number"] == len(raw) + 1


def test_missing_active_contract_is_no_view(config: dict) -> None:
    raw = synthetic_raw().loc[lambda frame: frame["SecurityID"].eq("1001")].copy()
    _, audit = normalize_sse_snapshot(
        raw, synthetic_master(), pd.Timestamp("2026-08-19"), config
    )
    assert audit["status"] == "NO_VIEW"
    assert audit["active_contract_coverage"] == 0.5
    assert not audit["gates"]["exact_active_contract_set"]


def test_license_evidence_is_mandatory(tmp_path: Path, config: dict) -> None:
    test_config = copy.deepcopy(config)
    test_config["paths"]["licensed_input_root"] = "licensed"
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    with pytest.raises(PermissionError, match="缺少合法数据许可"):
        validate_license_evidence_at_root(
            licensed_root / "missing.json",
            pd.Timestamp("2026-08-19"),
            test_config,
            tmp_path,
        )
    document_path = licensed_root / "license.pdf"
    document_path.write_bytes(b"licensed-test-document")
    document_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
    evidence_path = licensed_root / "license.json"
    evidence_path.write_text(
        json.dumps(
            {
                "provider": "上证所信息网络有限公司",
                "license_holder": "测试机构",
                "authorized_product": "股票期权历史行情",
                "valid_start": "2026-01-01",
                "valid_end": "2026-12-31",
                "research_use_permitted": True,
                "evidence_document_path": "license.pdf",
                "evidence_document_sha256": document_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = validate_license_evidence_at_root(
        evidence_path, pd.Timestamp("2026-08-19"), test_config, tmp_path
    )
    assert result["license_holder"] == "测试机构"


def test_license_product_and_document_hash_fail_closed(tmp_path: Path, config: dict) -> None:
    test_config = copy.deepcopy(config)
    test_config["paths"]["licensed_input_root"] = "licensed"
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    document_path = licensed_root / "license.pdf"
    document_path.write_bytes(b"licensed-test-document")
    evidence_path = licensed_root / "license.json"
    evidence = {
        "provider": "上证所信息网络有限公司",
        "license_holder": "测试机构",
        "authorized_product": "股票历史行情",
        "valid_start": "2026-01-01",
        "valid_end": "2026-12-31",
        "research_use_permitted": True,
        "evidence_document_path": "license.pdf",
        "evidence_document_sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="授权产品"):
        validate_license_evidence_at_root(
            evidence_path, pd.Timestamp("2026-08-19"), test_config, tmp_path
        )
    evidence["authorized_product"] = "股票期权历史行情"
    evidence["evidence_document_sha256"] = "a" * 64
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="SHA-256不匹配"):
        validate_license_evidence_at_root(
            evidence_path, pd.Timestamp("2026-08-19"), test_config, tmp_path
        )


def test_input_path_must_match_official_licensed_layout(tmp_path: Path, config: dict) -> None:
    test_config = copy.deepcopy(config)
    test_config["paths"]["licensed_input_root"] = "licensed"
    expected = tmp_path / "licensed" / "sho" / "20260819" / "Snapshot.csv"
    expected.parent.mkdir(parents=True)
    expected.write_text("test", encoding="utf-8")
    resolved = validate_input_path(
        expected, pd.Timestamp("2026-08-19"), test_config, tmp_path
    )
    assert resolved == expected.resolve()
    outside = tmp_path / "Snapshot.csv"
    outside.write_text("test", encoding="utf-8")
    with pytest.raises(PermissionError, match="许可目录"):
        validate_input_path(
            outside, pd.Timestamp("2026-08-19"), test_config, tmp_path
        )


def test_contract_master_path_is_frozen(tmp_path: Path, config: dict) -> None:
    test_config = copy.deepcopy(config)
    test_config["paths"]["contract_master"] = "master.parquet"
    expected = tmp_path / "master.parquet"
    expected.write_bytes(b"test")
    assert validate_master_path(expected, test_config, tmp_path) == expected.resolve()
    other = tmp_path / "other.parquet"
    other.write_bytes(b"test")
    with pytest.raises(PermissionError, match="冻结值"):
        validate_master_path(other, test_config, tmp_path)
