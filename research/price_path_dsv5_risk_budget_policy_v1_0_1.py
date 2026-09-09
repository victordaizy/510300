"""V1.0.1仅修复历史截断日历与未来官方日历的拼接；不改政策。"""
from pathlib import Path

import pandas as pd

from research import price_path_dsv5_risk_budget_policy_v1 as base

CORRECTION_MANIFEST = "config/510300_price_path_dsv5_risk_budget_policy_v1_0_1_manifest.json"


def forward_calendar(root: Path, cfg: dict) -> pd.DatetimeIndex:
    historical = pd.read_csv(root / cfg["inputs"]["calendar"], usecols=["trade_date"])
    official = pd.read_csv(root / cfg["inputs"]["calendar_2026"], usecols=["trade_date"])
    old = pd.DatetimeIndex(pd.to_datetime(historical.trade_date))
    new = pd.DatetimeIndex(pd.to_datetime(official.trade_date))
    if old.has_duplicates or new.has_duplicates or not old.is_monotonic_increasing or not new.is_monotonic_increasing:
        raise base.ContractError("交易日历重复或不递增")
    overlap = old[(old >= new.min()) & (old <= min(old.max(), new.max()))]
    expected = new[new <= min(old.max(), new.max())]
    if not overlap.equals(expected):
        raise base.ContractError("历史重叠区间交易日历与官方日历不一致")
    if old.max() < new.min():
        raise base.ContractError("历史与官方日历之间存在未覆盖区间")
    return old.union(new).sort_values()


def activate() -> None:
    base.MANIFEST = CORRECTION_MANIFEST
    base.forward_calendar = forward_calendar
