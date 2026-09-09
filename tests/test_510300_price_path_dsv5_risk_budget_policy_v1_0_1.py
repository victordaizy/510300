"""历史日历截断与官方未来日历的回归用例。"""
from pathlib import Path

import pandas as pd
import pytest

from research import price_path_dsv5_risk_budget_policy_v1 as base
from research.price_path_dsv5_risk_budget_policy_v1_0_1 import forward_calendar


def test_extend_truncated_history_from_official_calendar(monkeypatch):
    old = pd.bdate_range("2025-12-29", "2026-01-08")
    new = pd.bdate_range("2026-01-01", "2026-01-20")
    monkeypatch.setattr(pd, "read_csv", lambda path, **kw: pd.DataFrame({"trade_date": old if str(path) == "old" else new}))
    out = forward_calendar(Path(), {"inputs": {"calendar": "old", "calendar_2026": "new"}})
    assert out.equals(old.union(new))


def test_overlap_mismatch_still_blocks(monkeypatch):
    old = pd.bdate_range("2025-12-29", "2026-01-08").delete(4)
    new = pd.bdate_range("2026-01-01", "2026-01-20")
    monkeypatch.setattr(pd, "read_csv", lambda path, **kw: pd.DataFrame({"trade_date": old if str(path) == "old" else new}))
    with pytest.raises(base.ContractError, match="重叠区间"):
        forward_calendar(Path(), {"inputs": {"calendar": "old", "calendar_2026": "new"}})


def test_current_date_only_sources_can_register_future_grid():
    root = Path(__file__).resolve().parents[1]
    cfg = base.load_config(root)
    calendar = forward_calendar(root, cfg)
    assert calendar.max() == pd.Timestamp("2026-12-31")
    anchor = calendar.get_indexer([pd.Timestamp("2018-11-21")])[0]
    grid = calendar[anchor::5]
    assert (grid > pd.Timestamp("2026-09-05")).any()
