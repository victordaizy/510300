"""验证时间隔离、未知状态、统计恒等式与有限采集失败路径。"""
import json

import numpy as np
import pandas as pd
import pytest

from research.roadmap_execution_v1 import ROOT, admission_state, available_asof, bootstrap_indices, net_payoff
from research.roadmap_public_vintages_v1 import collect_source, payload_rows, source_requests


def test_paired_blocks_are_contiguous_and_reproducible():
    indices = bootstrap_indices(51, 20, 7, 42)
    assert indices.shape == (20, 51)
    assert np.array_equal(indices, bootstrap_indices(51, 20, 7, 42))
    for start in range(0, 49, 7):
        assert np.all(np.diff(indices[:, start:start + 7], axis=1) % 51 == 1)


def test_net_payoff_keeps_flat_and_does_not_charge_twice():
    result = net_payoff(np.array([0.1, -0.05, 0.0, -0.03]))
    assert result["profit_probability"] == 0.25
    assert result["loss_probability"] == 0.5
    assert result["flat_probability"] == 0.25
    assert result["pG_minus_qL_net"] == pytest.approx(0.005)
    assert result["cost_deducted_again"] is False


def test_all_losing_cycles_are_not_missing_or_zero():
    result = net_payoff(np.array([-0.02, -0.04]))
    assert result["mean_gain_net"] == 0
    assert result["pG_minus_qL_net"] == pytest.approx(-0.03)


def test_late_receipt_and_future_data_do_not_enter_past():
    original = pd.DataFrame({"economic_at": ["2026-09-29T07:00:00Z", "2026-09-29T07:00:00Z"],
                             "received_at": ["2026-09-29T08:00:00Z", "2026-09-30T08:00:00Z"], "value": [1, 2]})
    future = pd.DataFrame({"economic_at": ["2026-10-02T07:00:00Z"], "received_at": ["2026-10-02T08:00:00Z"], "value": [3]})
    cutoff = pd.Timestamp("2026-09-29T09:00:00Z")
    assert available_asof(original, cutoff).value.tolist() == [1]
    assert available_asof(pd.concat([original, future], ignore_index=True), cutoff).value.tolist() == [1]


def test_unknown_receipt_is_not_inferred_from_economic_date():
    frame = pd.DataFrame({"economic_at": ["2026-01-01T00:00:00Z"], "received_at": [None]})
    assert available_asof(frame, pd.Timestamp("2026-10-02T00:00:00Z")).empty


def test_old_use_or_missing_source_never_becomes_portfolio_candidate():
    assert admission_state(False, True, True).startswith("NOT_ADMITTED_OLD")
    assert admission_state(True, False, True).startswith("NOT_ADMITTED_SOURCE")
    assert admission_state(True, True, False).startswith("NOT_ADMITTED_ECONOMIC")
    assert admission_state(True, True, True) == "RESEARCH_CANDIDATE_ONLY"


def test_request_error_produces_receipt_without_fake_data(tmp_path):
    def failed_request(*args, **kwargs):
        raise ConnectionError("测试网络失败")
    result = collect_source(source_requests("2026-09-30")[0], tmp_path,
                            {"request_timeout_seconds": 1, "max_bytes_per_response": 100}, request=failed_request)
    assert result["received_at"] is None
    assert result["rows"] == 0
    assert result["status"] == "REQUEST_FAILED"
    assert json.loads((tmp_path / "SSE_MARGIN.receipt.json").read_text(encoding="utf-8"))["prediction_generated"] is False


def test_unknown_source_schema_does_not_fabricate_rows():
    assert payload_rows({"message": "请求失败"}) == []
    assert len(source_requests("2026-09-30")) == 3


def test_official_calendar_does_not_collect_weekend_as_makeup_trading():
    frame = pd.read_csv(ROOT / "data/reference/sse_trade_calendar_2026.csv")
    dates = pd.to_datetime(frame.trade_date)
    assert dates.loc[dates.lt("2026-10-02")].max() == pd.Timestamp("2026-09-30")
    assert dates.loc[dates.ge("2026-10-02")].min() == pd.Timestamp("2026-10-08")
    assert not dates.eq("2026-10-10").any()
