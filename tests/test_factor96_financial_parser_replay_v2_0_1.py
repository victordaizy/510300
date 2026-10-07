"""锁住已确认错误必须进入字段修复集合的实际失败模式。"""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from research.factor96_financial_parser_replay_v2_0_1 import extend_targets_with_confirmed_cases


def test_adds_missing_metric_without_changing_frozen_input():
    targets = [{"announcement_id": "1", "sha256": "a", "target_metrics": ["PROFIT"]}]
    original = deepcopy(targets)
    expanded, additions = extend_targets_with_confirmed_cases(targets, [
        {"announcement_id": "1", "pdf_sha256": "a", "metric_id": "ASSETS"}])
    assert targets == original
    assert expanded[0]["target_metrics"] == ["PROFIT", "ASSETS"]
    assert len(additions) == 1


def test_already_included_cases_are_not_added_twice():
    targets = [{"announcement_id": "1", "sha256": "a", "target_metrics": ["ASSETS"]}]
    expanded, additions = extend_targets_with_confirmed_cases(targets, [
        {"announcement_id": "1", "pdf_sha256": "a", "metric_id": "ASSETS"}])
    assert expanded == targets and additions == []


def test_missing_document_does_not_implicitly_expand_download_scope():
    with pytest.raises(AssertionError, match="不能自动新增请求"):
        extend_targets_with_confirmed_cases([], [{"announcement_id": "1", "pdf_sha256": "a", "metric_id": "ASSETS"}])


def test_hash_conflict_stops_extension():
    with pytest.raises(AssertionError, match="哈希不符"):
        extend_targets_with_confirmed_cases([{"announcement_id": "1", "sha256": "a", "target_metrics": []}],
            [{"announcement_id": "1", "pdf_sha256": "b", "metric_id": "ASSETS"}])


def test_real_failed_case_is_outside_old_scope_and_inside_union():
    base = Path(__file__).resolve().parents[1] / "reports/research/510300_factor96_financial_parser_scope_v2"
    targets = json.loads((base / "batch_targets.json").read_text(encoding="utf-8"))
    cases = json.loads((base / "inputs/confirmed_contradictions.json").read_text(encoding="utf-8"))
    before = (base / "batch_targets.json").read_bytes()
    old_pairs = {(t["announcement_id"], metric) for t in targets for metric in t["target_metrics"]}
    old_ledger = pd.DataFrame(sorted(old_pairs), columns=["announcement_id", "metric_id"])
    failed = old_ledger.loc[old_ledger.announcement_id.eq("1214959878") & old_ledger.metric_id.eq("TOTAL_ASSETS_END")]
    with pytest.raises(IndexError):
        failed.iloc[0]
    expanded, additions = extend_targets_with_confirmed_cases(targets, cases)
    new_pairs = {(t["announcement_id"], metric) for t in expanded for metric in t["target_metrics"]}
    assert len(old_pairs) == 2529 and len(new_pairs) == 2530
    assert len(expanded) == len(targets) == 2112
    assert new_pairs - old_pairs == {("1214959878", "TOTAL_ASSETS_END")}
    assert {(c["announcement_id"], c["metric_id"]) for c in cases}.issubset(new_pairs)
    assert len(additions) == 1 and (base / "batch_targets.json").read_bytes() == before
