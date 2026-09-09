"""Round5联合输入日期闸门与旧信号失效测试。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.refresh_r5_daily_signal import (
    Round5InputDateMismatchError,
    _write_non_consumable_signal,
    load_round5_freshness_contract,
    validate_round5_input_asof_alignment,
)


def _dates(*values: str) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(values))})


def test_20260818_misaligned_inputs_fail_with_required_status() -> None:
    """复现510300到8月18日、其余核心输入仍停在8月17日的事故。"""

    frames = {
        "510300": _dates("2026-08-15", "2026-08-18"),
        "000300": _dates("2026-08-15", "2026-08-17"),
        "H00300": _dates("2026-08-15", "2026-08-17"),
        "valuation": _dates("2026-08-15", "2026-08-17"),
    }

    with pytest.raises(Round5InputDateMismatchError) as raised:
        validate_round5_input_asof_alignment(
            signal_date=pd.Timestamp("2026-08-18"),
            frames=frames,
            contract=load_round5_freshness_contract(),
        )

    error = raised.value
    assert error.status == "FAILED_INPUT_DATE_MISMATCH"
    assert error.input_asof_dates == {
        "510300": "2026-08-18",
        "000300": "2026-08-17",
        "H00300": "2026-08-17",
        "valuation": "2026-08-17",
    }
    assert any("核心行情as_of_date不一致" in reason for reason in error.reasons)


def test_failed_alignment_replaces_old_signal_with_non_consumable_tombstone(
    tmp_path,
) -> None:
    """失败时消费者路径不再保留可误认成今日目标的旧信号。"""

    output = tmp_path / "round5_latest_signal.json"
    output.write_text(
        json.dumps(
            {
                "signal_date": "2026-08-12",
                "target_position": 0.3,
                "automatic_ordering_authorized": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _write_non_consumable_signal(
        target_date=pd.Timestamp("2026-08-18"),
        failure_status="FAILED_INPUT_DATE_MISMATCH",
        error="核心行情as_of_date不一致",
        input_asof_dates={
            "510300": "2026-08-18",
            "000300": "2026-08-17",
            "H00300": "2026-08-17",
            "valuation": "2026-08-17",
        },
        output_file=output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED_INPUT_DATE_MISMATCH"
    assert payload["target_date"] == "2026-08-18"
    assert payload["consumable"] is False
    assert payload["signal"] is None
    assert "target_position" not in payload
    assert payload["prior_signal_reference"]["signal_date"] == "2026-08-12"


def test_common_market_date_accepts_contract_allowed_valuation_lag() -> None:
    frames = {
        "510300": _dates("2026-08-18"),
        "000300": _dates("2026-08-18"),
        "H00300": _dates("2026-08-18"),
        "valuation": _dates("2026-08-17"),
    }

    result = validate_round5_input_asof_alignment(
        signal_date=pd.Timestamp("2026-08-18"),
        frames=frames,
        contract=load_round5_freshness_contract(),
    )

    assert result["status"] == "PASS"
    assert result["input_asof_dates"]["valuation"] == "2026-08-17"

