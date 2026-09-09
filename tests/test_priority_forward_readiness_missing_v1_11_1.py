"""缺少资料与确实零个质量日必须保持不同状态。"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import run_priority_forward_codex_automation_v1_11_1 as entry
from scripts.priority_forward_data_paths_v1 import WorkspacePaths


@pytest.mark.parametrize("payload, expected", [
    (None, None),
    ({"generated_at": "2026-09-06"}, None),
    ({"daily_quality": [], "legacy_complete_quality_day_count": 0}, 0),
])
def test_missing_readiness_is_not_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload, expected) -> None:
    monkeypatch.setattr(entry.parent, "PATHS", WorkspacePaths(tmp_path, tmp_path / "unused_data"))
    path = tmp_path / "reports/data_quality/510300_primary_market_readiness_v1_2.json"
    if payload is not None:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = entry.collection_status(datetime(2026, 9, 6, tzinfo=ZoneInfo("Asia/Shanghai")))
    assert result["strict_route_quality_days_in_readiness"] == expected
    if expected is None:
        assert result["status"] == "NO_VIEW_MISSING_OR_INCOMPLETE_READINESS"
