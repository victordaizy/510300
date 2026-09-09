"""历史沪深300成员代码哈希留出的数据拼接与点时特征。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, ModelRules, build_features


TOTAL_RETURN_COLUMNS = (
    "total_return_open",
    "total_return_high",
    "total_return_low",
    "total_return_close",
)

RANK_SPECS = (
    ("raw_medium", True),
    ("raw_momentum60", True),
    ("raw_momentum20", True),
    ("raw_reversal5", True),
    ("raw_vol20", False),
    ("raw_vol60", False),
    ("raw_drawdown60", True),
    ("raw_liquidity_shock", True),
    ("raw_range_compression", True),
    ("raw_trend_efficiency", True),
)


def hash_bucket(code: str) -> int:
    """用证券代码的SHA256首字节生成稳定的五桶划分。"""

    return hashlib.sha256(str(code).encode("utf-8")).digest()[0] % 5


def read_codes(paths: list[Path]) -> list[str]:
    """只读取代码列，避免训练阶段把留出组收益载入内存。"""

    codes: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=["con_code"])
        codes.update(frame["con_code"].dropna().astype(str).unique())
    return sorted(codes)


def build_master(codes: list[str]) -> pd.DataFrame:
    """形成确定性的训练/留出证券清单。"""

    master = pd.DataFrame({"ts_code": sorted(set(map(str, codes)))})
    master["hash_bucket"] = master["ts_code"].map(hash_bucket)
    master["split_group"] = np.where(master["hash_bucket"].eq(0), "HOLDOUT", "TRAIN")
    master["list_date"] = pd.Timestamp("1900-01-01")
    master["delist_date"] = pd.NaT
    return master


def _read_filtered(path: Path, codes: list[str]) -> pd.DataFrame:
    if not codes:
        raise ValueError("证券代码筛选为空")
    frame = pd.read_parquet(path, filters=[("con_code", "in", list(map(str, codes)))])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["con_code"] = frame["con_code"].astype(str)
    unexpected = set(frame["con_code"].unique()).difference(codes)
    if unexpected:
        raise RuntimeError(f"行情筛选混入非目标证券：{sorted(unexpected)[:5]}")
    return frame


def load_continuous_panel(early_path: Path, recent_path: Path, codes: list[str]) -> tuple[pd.DataFrame, dict]:
    """合并两段行情，并按重叠期比例统一复权价格基准。"""

    early = _read_filtered(early_path, codes)
    recent = _read_filtered(recent_path, codes)
    overlap = early[["date", "con_code", "total_return_close"]].merge(
        recent[["date", "con_code", "total_return_close"]],
        on=["date", "con_code"],
        suffixes=("_early", "_recent"),
        validate="one_to_one",
    )
    overlap["scale"] = overlap["total_return_close_recent"] / overlap["total_return_close_early"]
    scales = overlap.groupby("con_code", as_index=False)["scale"].median()
    scale_map = dict(zip(scales["con_code"], scales["scale"], strict=True))
    early["seam_scale"] = early["con_code"].map(scale_map).fillna(1.0)
    for column in TOTAL_RETURN_COLUMNS:
        early[column] = pd.to_numeric(early[column], errors="coerce") * early["seam_scale"]
    early.drop(columns="seam_scale", inplace=True)
    early["source_priority"] = 0
    recent["source_priority"] = 1
    panel = (
        pd.concat([early, recent], ignore_index=True, sort=False)
        .sort_values(["date", "con_code", "source_priority"])
        .drop_duplicates(["date", "con_code"], keep="last")
        .drop(columns="source_priority")
        .sort_values(["con_code", "date"])
        .reset_index(drop=True)
    )
    if panel[["date", "con_code"]].duplicated().any():
        raise RuntimeError("拼接行情仍存在证券日期重复")
    total_close = pd.to_numeric(panel["total_return_close"], errors="coerce")
    absolute_returns = total_close.groupby(panel["con_code"]).pct_change().abs()
    active_mask = panel["is_index_member"].fillna(False).astype(bool)
    audit = {
        "rows": int(len(panel)),
        "codes": int(panel["con_code"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "overlap_rows": int(len(overlap)),
        "scaled_code_count": int(len(scales)),
        "maximum_absolute_total_return_change": float(absolute_returns.max(skipna=True)),
        "absolute_changes_above_30pct": int(absolute_returns.gt(0.30).sum()),
        "active_member_absolute_changes_above_30pct": int((absolute_returns.gt(0.30) & active_mask).sum()),
    }
    return panel, audit


def load_continuous_benchmark(early_path: Path, recent_path: Path) -> tuple[pd.DataFrame, dict]:
    """拼接H00300全收益指数并消除可能的基准缩放差异。"""

    early = pd.read_parquet(early_path, columns=["date", "close"]).copy()
    recent = pd.read_parquet(recent_path, columns=["date", "close"]).copy()
    for frame in (early, recent):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    overlap = early.merge(recent, on="date", suffixes=("_early", "_recent"), validate="one_to_one")
    scale = float((overlap["close_recent"] / overlap["close_early"]).median())
    early["close"] *= scale
    early["source_priority"] = 0
    recent["source_priority"] = 1
    benchmark = (
        pd.concat([early, recent], ignore_index=True)
        .sort_values(["date", "source_priority"])
        .drop_duplicates("date", keep="last")
        .drop(columns="source_priority")
        .sort_values("date")
        .reset_index(drop=True)
    )
    changes = benchmark["close"].pct_change().abs()
    if changes.max(skipna=True) > 0.15:
        raise RuntimeError("H00300拼接出现超过15%的单日变化")
    return benchmark, {
        "rows": int(len(benchmark)),
        "date_min": str(benchmark["date"].min().date()),
        "date_max": str(benchmark["date"].max().date()),
        "overlap_rows": int(len(overlap)),
        "early_scale": scale,
        "maximum_absolute_change": float(changes.max(skipna=True)),
    }


def build_point_in_time_features(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark: pd.DataFrame,
    rules: ModelRules,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """只在当日指数成员中重新计算截面排名。"""

    features, calendar = build_features(panel, master, benchmark, rules)
    membership = panel[["date", "con_code", "is_index_member"]].copy()
    membership["date"] = pd.to_datetime(membership["date"])
    if membership[["date", "con_code"]].duplicated().any():
        raise RuntimeError("成员标记存在证券日期重复")
    positions = pd.Series(np.arange(len(calendar), dtype=int), index=calendar)
    continuity = panel[["date", "con_code"]].drop_duplicates().copy()
    continuity["date"] = pd.to_datetime(continuity["date"])
    continuity["calendar_position"] = continuity["date"].map(positions)
    continuity.dropna(subset=["calendar_position"], inplace=True)
    continuity.sort_values(["con_code", "date"], inplace=True)
    discontinuity = continuity.groupby("con_code")["calendar_position"].diff().ne(1)
    continuity["segment"] = discontinuity.groupby(continuity["con_code"]).cumsum()
    continuity["continuous_history_rows"] = continuity.groupby(["con_code", "segment"]).cumcount() + 1
    continuity = continuity[["date", "con_code", "continuous_history_rows"]]
    features = features.drop(columns="is_index_member", errors="ignore").merge(
        membership,
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    ).merge(continuity, on=["date", "con_code"], how="left", validate="one_to_one")
    features["is_index_member"] = features["is_index_member"].fillna(False).astype(bool)
    features["eligible"] = (
        features["eligible"].astype(bool)
        & features["is_index_member"]
        & features["continuous_history_rows"].fillna(0).ge(120)
    )
    for output, (raw_name, ascending) in zip(FEATURE_COLUMNS, RANK_SPECS, strict=True):
        features[output] = np.nan
        features.loc[features["eligible"], output] = (
            features.loc[features["eligible"]]
            .groupby("date")[raw_name]
            .rank(pct=True, ascending=ascending)
        )
    features["signal_output"] = np.where(features["eligible"], "SIGNAL_READY", "NO_VIEW")
    return features.sort_values(["date", "con_code"]).reset_index(drop=True), calendar
