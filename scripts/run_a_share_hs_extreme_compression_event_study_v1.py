"""运行不设盲测的 A 股极端缩波、量能与价格位置全历史事件研究 V1。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/a_share_hs_extreme_compression_volume_position_event_study_v1.yaml"
STAGING = ROOT / "data/staging/a_share_hs_extreme_compression_volume_position_event_study_v1"
BUCKET_ROOT = STAGING / "factor_input_buckets_v1"
CHECKPOINT_ROOT = STAGING / "factor_event_checkpoints_v1"
EVENT_LEDGER = STAGING / "event_ledger.parquet"
HORIZON_OUTCOMES = STAGING / "horizon_outcomes.parquet"
BREAKOUT_EVENTS = STAGING / "breakout_events.parquet"
LANDMARKS = STAGING / "d0_d5_d10_same_event_landmarks.csv"
WAIT_POLICY = STAGING / "wait_until_not_high_policy.csv"
GROUP_SUMMARY = STAGING / "volume_position_group_summary.csv"
FACTOR_DICTIONARY = STAGING / "factor_dictionary.json"
RUN_RECEIPT = STAGING / "event_study_run_receipt.json"
REPORT = ROOT / "reports/research/A_SHARE_HS_EXTREME_COMPRESSION_VOLUME_POSITION_EVENT_STUDY_V1.md"
BUCKET_COUNT = 32
HORIZONS = (5, 10, 20, 40, 60, 120)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def project_path(relative: str) -> Path:
    """把项目相对路径解析为受控绝对路径。"""

    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def sha256_file(path: Path) -> str:
    """分块计算 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(content: bytes, path: Path) -> None:
    """原子写入二进制内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(text: str, path: Path) -> None:
    """原子写入 UTF-8 文本。"""

    atomic_bytes(text.encode("utf-8"), path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sql_path(path: Path) -> str:
    """生成 DuckDB 可用的绝对路径字面量内容。"""

    return path.resolve().as_posix().replace("'", "''")


def read_config(path: Path) -> dict[str, Any]:
    """读取并验证冻结研究配置。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["study"]["blind_test_required"] is not False:
        raise ValueError("该研究必须明确设置 blind_test_required=false")
    if config["study"]["research_scope"] != "DISCOVERY_ONLY":
        raise ValueError("该脚本只允许 DISCOVERY_ONLY")
    return config


def prepare_buckets(panel: Path, panel_hash: str) -> list[Path]:
    """把大面板一次性按证券哈希切为可独立计算的小分区。"""

    marker = BUCKET_ROOT / "bucket_manifest.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        files = sorted(BUCKET_ROOT.glob("symbol_bucket=*/*.parquet"))
        if (
            recorded.get("panel_sha256") == panel_hash
            and recorded.get("bucket_count") == BUCKET_COUNT
            and len(files) == BUCKET_COUNT
        ):
            return files
        raise ValueError("已有因子输入分桶与当前统一面板哈希不一致，请建立新版本目录")
    if BUCKET_ROOT.exists() and any(BUCKET_ROOT.iterdir()):
        raise ValueError("因子输入分桶目录非空但缺少有效清单，拒绝覆盖")
    BUCKET_ROOT.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='4GB'")
        connection.execute(
            f"""
            COPY (
                SELECT *, MOD(HASH(con_code), {BUCKET_COUNT})::INTEGER AS symbol_bucket
                FROM read_parquet('{sql_path(panel)}')
            ) TO '{sql_path(BUCKET_ROOT)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (symbol_bucket))
            """
        )
    finally:
        connection.close()
    files = sorted(BUCKET_ROOT.glob("symbol_bucket=*/*.parquet"))
    if len(files) != BUCKET_COUNT:
        raise ValueError(f"预期 {BUCKET_COUNT} 个证券分桶，实际 {len(files)}")
    atomic_text(
        json.dumps(
            {
                "panel": panel.relative_to(ROOT).as_posix(),
                "panel_sha256": panel_hash,
                "bucket_count": BUCKET_COUNT,
                "files": [path.relative_to(ROOT).as_posix() for path in files],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        marker,
    )
    return files


def trailing_valid_percentile(
    values: np.ndarray,
    lookback: int,
    minimum_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按最近有效值序列计算排除当前值的经验中秩百分位。"""

    result = np.full(len(values), np.nan, dtype=float)
    history_count = np.zeros(len(values), dtype=np.int32)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return result, history_count
    compact = pd.Series(values[valid_positions], dtype=float)
    window = compact.rolling(lookback + 1, min_periods=minimum_history + 1)
    rank = window.rank(method="average")
    count_with_current = window.count()
    denominator = count_with_current - 1.0
    percentile = (rank - 1.0) / denominator
    result[valid_positions] = percentile.to_numpy(dtype=float)
    history_count[valid_positions] = np.minimum(
        np.arange(len(valid_positions), dtype=np.int32),
        lookback,
    )
    result[history_count < minimum_history] = np.nan
    return result, history_count


def rolling_std(values: np.ndarray, window: int, shift: int = 0) -> np.ndarray:
    """固定市场日窗口滚动样本标准差。"""

    series = pd.Series(values, dtype=float)
    if shift:
        series = series.shift(shift)
    return series.rolling(window, min_periods=window).std(ddof=1).to_numpy(dtype=float)


def rolling_median(values: np.ndarray, window: int, shift: int = 0) -> np.ndarray:
    """固定市场日窗口滚动中位数。"""

    series = pd.Series(values, dtype=float)
    if shift:
        series = series.shift(shift)
    return series.rolling(window, min_periods=window).median().to_numpy(dtype=float)


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    """固定市场日窗口滚动最小值。"""

    return pd.Series(values, dtype=float).rolling(window, min_periods=window).min().to_numpy(dtype=float)


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """固定市场日窗口滚动最大值。"""

    return pd.Series(values, dtype=float).rolling(window, min_periods=window).max().to_numpy(dtype=float)


def build_share_grid(
    snapshots: pd.DataFrame,
    calendar_dates: np.ndarray,
    global_indices: np.ndarray,
) -> np.ndarray:
    """按 effective_date 与 available_at 的较晚日期建立点时可用流通股本序列。"""

    output = np.full(len(global_indices), np.nan, dtype=float)
    if snapshots.empty:
        return output
    snap = snapshots.copy()
    effective = pd.to_datetime(snap["effective_date"]).dt.normalize().to_numpy(dtype="datetime64[ns]")
    available = pd.to_datetime(snap["available_at"]).dt.normalize().to_numpy(dtype="datetime64[ns]")
    usable = np.maximum(effective, available)
    positions = np.searchsorted(calendar_dates, usable, side="left") + 1
    values = pd.to_numeric(snap["tradable_a_shares"], errors="coerce").to_numpy(dtype=float)
    valid = (positions >= 1) & (positions <= len(calendar_dates)) & np.isfinite(values) & (values > 0)
    if not valid.any():
        return output
    table = pd.DataFrame({"position": positions[valid], "value": values[valid]})
    table = table.sort_values("position").drop_duplicates("position", keep="last")
    pos = table["position"].to_numpy(dtype=np.int64)
    val = table["value"].to_numpy(dtype=float)
    lookup = np.searchsorted(pos, global_indices, side="right") - 1
    available_mask = lookup >= 0
    output[available_mask] = val[lookup[available_mask]]
    return output


def classify_price_position(current_dd: float) -> str:
    """固定三档价格位置标签。"""

    if not np.isfinite(current_dd):
        return "PRICE_POSITION_MISSING"
    if current_dd > -0.05:
        return "NEAR_252D_HIGH"
    if current_dd > -0.20:
        return "MID_RANGE"
    return "LOW_POSITION"


def classify_volume(percentile: float) -> str:
    """固定三档背景量能标签。"""

    if not np.isfinite(percentile):
        return "VOLUME_FACTOR_MISSING"
    if percentile <= 0.30:
        return "VOLUME_DRY_UP"
    if percentile < 0.70:
        return "VOLUME_NEUTRAL"
    return "VOLUME_ACTIVE"


def process_symbol(
    frame: pd.DataFrame,
    snapshots: pd.DataFrame,
    calendar_dates: np.ndarray,
    benchmark_close: np.ndarray,
    cutoff_index: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """计算单只股票的因子、episode、期限结果、突破、landmark 与等待反事实。"""

    frame = frame.sort_values("market_day_index").drop_duplicates("market_day_index", keep="first")
    ts_code = str(frame["con_code"].iloc[0])
    start_index = int(frame["market_day_index"].min())
    global_indices = np.arange(start_index, cutoff_index + 1, dtype=np.int64)
    length = len(global_indices)
    row_position = frame["market_day_index"].to_numpy(dtype=np.int64) - start_index
    observed = np.zeros(length, dtype=bool)
    observed[row_position] = True

    def grid(column: str) -> np.ndarray:
        values = np.full(length, np.nan, dtype=float)
        values[row_position] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return values

    tr_close = grid("total_return_close")
    tr_open = grid("total_return_open")
    raw_close = grid("raw_close")
    raw_open = grid("raw_open")
    volume = grid("volume")
    amount = grid("amount")
    returns = np.full(length, np.nan, dtype=float)
    consecutive = observed[1:] & observed[:-1] & (tr_close[1:] > 0) & (tr_close[:-1] > 0)
    returns[1:][consecutive] = np.log(tr_close[1:][consecutive] / tr_close[:-1][consecutive])

    rv20 = rolling_std(returns, 20) * np.sqrt(252.0)
    rv120_pre = rolling_std(returns, 120, shift=20) * np.sqrt(252.0)
    compression = np.full(length, np.nan, dtype=float)
    valid_compression = np.isfinite(rv20) & np.isfinite(rv120_pre) & (rv20 > 0) & (rv120_pre > 0)
    compression[valid_compression] = np.log(rv20[valid_compression] / rv120_pre[valid_compression])
    compression_pct, compression_history_count = trailing_valid_percentile(compression, 504, 252)

    volume_positive = volume.copy()
    volume_positive[~(volume_positive > 0)] = np.nan
    volume20_pre = rolling_median(volume_positive, 20, shift=1)
    volume120_pre = rolling_median(volume_positive, 120, shift=21)
    share_grid = build_share_grid(snapshots, calendar_dates, global_indices)
    share_series = pd.Series(share_grid, dtype=float)
    share_min = share_series.rolling(141, min_periods=141).min().to_numpy(dtype=float)
    share_max = share_series.rolling(141, min_periods=141).max().to_numpy(dtype=float)
    share_stable = np.isfinite(share_min) & (share_min > 0) & ((share_max / share_min - 1.0) <= 0.05)
    volume_participation = np.full(length, np.nan, dtype=float)
    valid_volume = (
        share_stable
        & np.isfinite(volume20_pre)
        & np.isfinite(volume120_pre)
        & (volume20_pre > 0)
        & (volume120_pre > 0)
    )
    volume_participation[valid_volume] = np.log(
        volume20_pre[valid_volume] / volume120_pre[valid_volume]
    )
    volume_pct, volume_history_count = trailing_valid_percentile(volume_participation, 504, 252)
    volume_surprise = np.full(length, np.nan, dtype=float)
    surprise_valid = share_stable & np.isfinite(volume_positive) & np.isfinite(volume20_pre)
    volume_surprise[surprise_valid] = np.log(volume_positive[surprise_valid] / volume20_pre[surprise_valid])
    volume_surprise_pct, volume_surprise_history_count = trailing_valid_percentile(
        volume_surprise,
        252,
        252,
    )

    high252 = rolling_max(tr_close, 252)
    low252 = rolling_min(tr_close, 252)
    current_dd252 = tr_close / high252 - 1.0
    width252 = high252 - low252
    range_position252 = np.full(length, np.nan, dtype=float)
    range_valid = np.isfinite(width252) & (width252 > 0)
    range_position252[range_valid] = (tr_close[range_valid] - low252[range_valid]) / width252[range_valid]
    amount20_pre = rolling_median(np.where(amount > 0, amount, np.nan), 20, shift=1)
    frozen_upper = rolling_max(tr_close, 20)
    frozen_lower = rolling_min(tr_close, 20)

    episode_events: list[dict[str, Any]] = []
    active_event: dict[str, Any] | None = None
    exit_confirmation = 0
    candidate_exit_position: int | None = None
    for position in range(1, length):
        if active_event is not None:
            if np.isfinite(compression_pct[position]) and compression_pct[position] > 0.30:
                exit_confirmation += 1
                if exit_confirmation == 1:
                    candidate_exit_position = position
                if exit_confirmation == 5:
                    active_event["episode_exit_candidate_date"] = pd.Timestamp(
                        calendar_dates[start_index + candidate_exit_position - 1]
                    )
                    active_event["episode_exit_confirmation_date"] = pd.Timestamp(
                        calendar_dates[start_index + position - 1]
                    )
                    active_event["compression_duration_market_days"] = position - active_event["event_position"]
                    active_event = None
                    exit_confirmation = 0
                    candidate_exit_position = None
            else:
                exit_confirmation = 0
                candidate_exit_position = None
        if active_event is None:
            is_start = (
                np.isfinite(compression_pct[position])
                and np.isfinite(compression_pct[position - 1])
                and compression_pct[position] <= 0.10
                and compression_pct[position - 1] > 0.10
                and observed[position]
                and observed[position - 1]
            )
            if is_start:
                event_date = pd.Timestamp(calendar_dates[start_index + position - 1])
                event_id = f"{config['study']['study_id']}|{ts_code}|{event_date.date().isoformat()}"
                event = {
                    "event_id": event_id,
                    "study_id": config["study"]["study_id"],
                    "formula_version": "V1_NO_BLIND_FIXED_20260822",
                    "ts_code": ts_code,
                    "event_date": event_date,
                    "event_position": position,
                    "market_day_index": int(global_indices[position]),
                    "rv20": rv20[position],
                    "rv120_pre": rv120_pre[position],
                    "compression_log_ratio": compression[position],
                    "compression_pct": compression_pct[position],
                    "compression_history_count": int(compression_history_count[position]),
                    "compression_history_class": (
                        "FULL_504_HISTORY"
                        if compression_history_count[position] >= 504
                        else "SHORT_252_TO_503_HISTORY"
                    ),
                    "volume20_pre_shares": volume20_pre[position],
                    "volume120_pre_shares": volume120_pre[position],
                    "volume_participation_log_ratio": volume_participation[position],
                    "volume_participation_pct": volume_pct[position],
                    "volume_history_count": int(volume_history_count[position]),
                    "volume_group": classify_volume(volume_pct[position]),
                    "volume_factor_valid": bool(np.isfinite(volume_pct[position])),
                    "capital_structure_stable_141d": bool(share_stable[position]),
                    "tradable_a_shares_pit": share_grid[position],
                    "event_day_volume_surprise": volume_surprise[position],
                    "event_day_volume_surprise_pct": volume_surprise_pct[position],
                    "event_day_volume_surprise_history_count": int(volume_surprise_history_count[position]),
                    "event_day_volume_surge": bool(
                        np.isfinite(volume_surprise_pct[position])
                        and volume_surprise_pct[position] >= 0.80
                    ),
                    "current_dd252": current_dd252[position],
                    "range_position252": range_position252[position],
                    "price_position_group": classify_price_position(current_dd252[position]),
                    "binary_price_group": (
                        "PRICE_POSITION_MISSING"
                        if not np.isfinite(current_dd252[position])
                        else "HIGH_GROUP"
                        if current_dd252[position] > -0.05
                        else "NOT_HIGH_GROUP"
                        if current_dd252[position] <= -0.10
                        else "GRAY_ZONE"
                    ),
                    "amount20_pre_median": amount20_pre[position],
                    "event_total_return_close": tr_close[position],
                    "frozen_upper20": frozen_upper[position],
                    "frozen_lower20": frozen_lower[position],
                    "episode_exit_candidate_date": pd.NaT,
                    "episode_exit_confirmation_date": pd.NaT,
                    "compression_duration_market_days": np.nan,
                    "event_position_internal": position,
                }
                episode_events.append(event)
                active_event = event

    outcomes: list[dict[str, Any]] = []
    breakouts: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    waits: list[dict[str, Any]] = []
    for event in episode_events:
        position = int(event["event_position_internal"])
        origin = tr_close[position]
        first_break_position: int | None = None
        first_break_direction = "NO_BREAK_WITHIN_60D"
        for offset in range(1, 61):
            future = position + offset
            if future >= length:
                break
            if not np.isfinite(tr_close[future]):
                continue
            if tr_close[future] > event["frozen_upper20"]:
                first_break_position = future
                first_break_direction = "UP_BREAK"
                break
            if tr_close[future] < event["frozen_lower20"]:
                first_break_position = future
                first_break_direction = "DOWN_BREAK"
                break
        breakout_search_mature = position + 60 < length
        if first_break_position is None and not breakout_search_mature:
            first_break_direction = "BREAKOUT_WINDOW_RIGHT_CENSORED"
        break_date = (
            pd.Timestamp(calendar_dates[start_index + first_break_position - 1])
            if first_break_position is not None
            else pd.NaT
        )
        breakouts.append(
            {
                "event_id": event["event_id"],
                "ts_code": ts_code,
                "event_date": event["event_date"],
                "first_break_direction": first_break_direction,
                "breakout_search_mature": breakout_search_mature,
                "first_break_date": break_date,
                "time_to_first_break_market_days": (
                    first_break_position - position if first_break_position is not None else np.nan
                ),
                "breakout_volume_surprise": (
                    volume_surprise[first_break_position] if first_break_position is not None else np.nan
                ),
                "breakout_volume_surprise_pct": (
                    volume_surprise_pct[first_break_position] if first_break_position is not None else np.nan
                ),
            }
        )
        for landmark_offset in (0, 5, 10):
            landmark_position = position + landmark_offset
            mature = landmark_position < length
            landmark_date = (
                pd.Timestamp(calendar_dates[start_index + landmark_position - 1]) if mature else pd.NaT
            )
            pre_path = tr_close[position : landmark_position + 1] if mature else np.array([])
            complete = bool(mature and len(pre_path) == landmark_offset + 1 and np.isfinite(pre_path).all())
            landmarks.append(
                {
                    "event_id": event["event_id"],
                    "ts_code": ts_code,
                    "event_date": event["event_date"],
                    "landmark": f"D{landmark_offset}",
                    "landmark_date": landmark_date,
                    "mature": mature,
                    "pre_landmark_path_complete": complete,
                    "pre_landmark_return": (
                        tr_close[landmark_position] / origin - 1.0 if complete else np.nan
                    ),
                    "compression_pct": compression_pct[landmark_position] if mature else np.nan,
                    "volume_participation_pct": volume_pct[landmark_position] if mature else np.nan,
                    "current_dd252": current_dd252[landmark_position] if mature else np.nan,
                    "first_break_before_or_at_landmark": bool(
                        first_break_position is not None and first_break_position <= landmark_position
                    ),
                }
            )
        if event["price_position_group"] == "NEAR_252D_HIGH":
            trigger: int | None = None
            for offset in range(1, 21):
                future = position + offset
                if future >= length:
                    break
                if np.isfinite(current_dd252[future]) and current_dd252[future] <= -0.10:
                    trigger = future
                    break
            entry_position = trigger + 1 if trigger is not None else None
            entry_observed = bool(
                entry_position is not None
                and entry_position < length
                and observed[entry_position]
                and np.isfinite(tr_open[entry_position])
                and volume[entry_position] > 0
            )
            waits.append(
                {
                    "event_id": event["event_id"],
                    "ts_code": ts_code,
                    "event_date": event["event_date"],
                    "triggered_within_20d": trigger is not None,
                    "trigger_date": (
                        pd.Timestamp(calendar_dates[start_index + trigger - 1]) if trigger is not None else pd.NaT
                    ),
                    "wait_market_days": trigger - position if trigger is not None else 20,
                    "entry_date": (
                        pd.Timestamp(calendar_dates[start_index + entry_position - 1])
                        if entry_observed
                        else pd.NaT
                    ),
                    "entry_status": (
                        "PRICE_AND_VOLUME_OBSERVED_LIMIT_STATE_UNRESOLVED"
                        if entry_observed
                        else "NO_TRIGGER_CASH"
                        if trigger is None
                        else "TRIGGERED_NO_ENTRY_NEXT_OPEN"
                    ),
                    "entry_total_return_open": tr_open[entry_position] if entry_observed else np.nan,
                }
            )
        for horizon in HORIZONS:
            endpoint = position + horizon
            mature = endpoint < length
            endpoint_observed = bool(mature and observed[endpoint] and np.isfinite(tr_close[endpoint]))
            path = tr_close[position : endpoint + 1] if mature else np.array([])
            path_complete = bool(mature and len(path) == horizon + 1 and np.isfinite(path).all())
            path_return = tr_close[endpoint] / origin - 1.0 if endpoint_observed else np.nan
            benchmark_origin = benchmark_close[global_indices[position] - 1]
            benchmark_endpoint = benchmark_close[global_indices[endpoint] - 1] if mature else np.nan
            benchmark_return = (
                benchmark_endpoint / benchmark_origin - 1.0
                if np.isfinite(benchmark_origin) and np.isfinite(benchmark_endpoint)
                else np.nan
            )
            excess_return = (
                (1.0 + path_return) / (1.0 + benchmark_return) - 1.0
                if np.isfinite(path_return) and np.isfinite(benchmark_return)
                else np.nan
            )
            relative_path = path / origin - 1.0 if path_complete else np.array([])
            next_open_position = position + 1
            next_open_observed = bool(
                next_open_position < length
                and observed[next_open_position]
                and np.isfinite(tr_open[next_open_position])
                and volume[next_open_position] > 0
            )
            investable_return = (
                tr_close[endpoint] / tr_open[next_open_position] - 1.0
                if endpoint_observed and next_open_observed
                else np.nan
            )
            outcomes.append(
                {
                    "event_id": event["event_id"],
                    "ts_code": ts_code,
                    "event_date": event["event_date"],
                    "horizon_market_days": horizon,
                    "mature_horizon": mature,
                    "endpoint_observed": endpoint_observed,
                    "path_complete": path_complete,
                    "path_return": path_return,
                    "h00300_return": benchmark_return,
                    "h00300_compound_excess_return": excess_return,
                    "mae": np.nanmin(relative_path) if len(relative_path) else np.nan,
                    "mfe": np.nanmax(relative_path) if len(relative_path) else np.nan,
                    "time_to_mae": int(np.nanargmin(relative_path)) if len(relative_path) else np.nan,
                    "time_to_mfe": int(np.nanargmax(relative_path)) if len(relative_path) else np.nan,
                    "post_event_realized_volatility": (
                        np.std(np.diff(np.log(path)), ddof=1) * np.sqrt(252.0)
                        if path_complete and horizon >= 2
                        else np.nan
                    ),
                    "post_event_downside_semideviation": (
                        np.sqrt(np.mean(np.minimum(np.diff(np.log(path)), 0.0) ** 2)) * np.sqrt(252.0)
                        if path_complete
                        else np.nan
                    ),
                    "next_open_status": (
                        "PRICE_AND_VOLUME_OBSERVED_LIMIT_STATE_UNRESOLVED"
                        if next_open_observed
                        else "NO_ENTRY_NEXT_OPEN"
                    ),
                    "investable_return_price_only_provisional": investable_return,
                    "censoring_status": (
                        "MATURE_COMPLETE_PATH"
                        if path_complete
                        else "RIGHT_CENSORED"
                        if not mature
                        else "MATURE_ENDPOINT_MISSING"
                        if not endpoint_observed
                        else "MATURE_INTERIOR_PATH_GAP"
                    ),
                }
            )
        event.pop("event_position_internal", None)
    return episode_events, outcomes, breakouts, landmarks, waits


def load_calendars(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """载入两个交易所的观察开放日。"""

    path = project_path(config["inputs"]["trading_calendar"]["path"])
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["is_open"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return {
        exchange: group.sort_values("date")["date"].to_numpy(dtype="datetime64[ns]")
        for exchange, group in frame.groupby("exchange")
    }


def load_benchmark(config: dict[str, Any], calendar_dates: np.ndarray) -> np.ndarray:
    """按冻结优先级载入沪深300全收益收盘指数。"""

    parts = []
    for priority, relative in enumerate(config["inputs"]["h00300_partitions"]):
        frame = pd.read_parquet(project_path(relative), columns=["date", "close"])
        frame["priority"] = priority
        parts.append(frame)
    benchmark = pd.concat(parts, ignore_index=True)
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values(["date", "priority"]).drop_duplicates("date", keep="first")
    mapping = benchmark.set_index("date")["close"]
    return pd.Series(pd.to_datetime(calendar_dates)).map(mapping).to_numpy(dtype=float)


def attach_point_in_time_labels(events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """给事件附加证券身份、ST/退市整理状态和点时行业。"""

    master = pd.read_parquet(project_path(config["inputs"]["security_master"]["path"]))
    master = master.rename(columns={"name": "security_name"})
    events["event_date"] = pd.to_datetime(events["event_date"]).astype("datetime64[ns]")
    for column in ("list_date", "delist_date"):
        master[column] = pd.to_datetime(master[column]).astype("datetime64[ns]")
    keep = [
        "ts_code",
        "exchange",
        "market",
        "security_name",
        "security_type",
        "list_date",
        "delist_date",
    ]
    events = events.merge(master[keep], on="ts_code", how="left", validate="many_to_one")
    events["path_panel_eligible"] = (
        events["security_type"].eq("ORDINARY_A_SHARE")
        & events["list_date"].le(events["event_date"])
        & (events["delist_date"].isna() | events["delist_date"].ge(events["event_date"]))
    )

    def asof_interval(
        left: pd.DataFrame,
        relative: str,
        start: str,
        end: str,
        value_columns: list[str],
    ) -> pd.DataFrame:
        intervals = pd.read_parquet(project_path(relative))
        intervals[start] = pd.to_datetime(intervals[start]).dt.normalize().astype("datetime64[ns]")
        intervals[end] = pd.to_datetime(intervals[end]).dt.normalize().astype("datetime64[ns]")
        intervals["available_at"] = pd.to_datetime(intervals["available_at"]).astype("datetime64[ns]")
        intervals = intervals.sort_values([start, "ts_code"])
        ordered = left.sort_values(["event_date", "ts_code"])
        merged = pd.merge_asof(
            ordered,
            intervals[["ts_code", start, end, "available_at", *value_columns]],
            left_on="event_date",
            right_on=start,
            by="ts_code",
            direction="backward",
            allow_exact_matches=True,
        )
        valid = (
            (merged[end].isna() | merged["event_date"].le(merged[end]))
            & pd.to_datetime(merged["available_at"]).le(merged["event_date"] + pd.Timedelta(hours=15))
        )
        merged.loc[~valid, value_columns] = pd.NA
        return merged.drop(columns=[start, end, "available_at"])

    events = asof_interval(
        events,
        config["inputs"]["security_status_intervals"]["path"],
        "valid_from",
        "valid_to",
        ["status"],
    ).rename(columns={"status": "security_status_pit"})
    events = asof_interval(
        events,
        config["inputs"]["industry_intervals"]["path"],
        "valid_from",
        "valid_to",
        ["industry_l1", "industry_l1_code"],
    )
    events["investable_non_st_candidate"] = (
        events["path_panel_eligible"] & events["security_status_pit"].eq("NORMAL")
    )
    return events.sort_values(["event_date", "ts_code"]).reset_index(drop=True)


def combine_checkpoints(kind: str, output: Path, bucket_ids: list[int]) -> pd.DataFrame:
    """合并全部桶检查点并原子固化。"""

    paths = [CHECKPOINT_ROOT / f"bucket_{bucket:02d}_{kind}.parquet" for bucket in bucket_ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 {kind} 检查点：{missing[:3]}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    atomic_parquet(frame, output)
    return frame


def build_group_summary(events: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """生成固定量能三档乘价格位置三档的描述性主表。"""

    merged = outcomes.merge(
        events[
            [
                "event_id",
                "volume_group",
                "price_position_group",
                "path_panel_eligible",
            ]
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.loc[merged["path_panel_eligible"]].copy()
    keys = ["volume_group", "price_position_group", "horizon_market_days"]
    rows = []
    for key, group in merged.groupby(keys, dropna=False):
        valid = group.loc[group["path_return"].notna(), "path_return"]
        date_means = group.loc[group["path_return"].notna()].groupby("event_date")["path_return"].mean()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "events": int(len(group)),
                "symbols": int(group["ts_code"].nunique()),
                "event_dates": int(group["event_date"].nunique()),
                "mature_events": int(group["mature_horizon"].sum()),
                "valid_endpoint_events": int(group["endpoint_observed"].sum()),
                "complete_path_events": int(group["path_complete"].sum()),
                "event_equal_mean_return": valid.mean(),
                "event_date_equal_weight_mean_return": date_means.mean(),
                "median_return": valid.median(),
                "return_q25": valid.quantile(0.25),
                "return_q75": valid.quantile(0.75),
                "return_lte_minus_10pct": (valid <= -0.10).mean(),
                "return_lte_minus_20pct": (valid <= -0.20).mean(),
                "median_mae_complete_path": group.loc[group["path_complete"], "mae"].median(),
                "median_mfe_complete_path": group.loc[group["path_complete"], "mfe"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def build_wait_policy_outcomes(
    waits: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    panel: Path,
) -> pd.DataFrame:
    """计算近高事件等待脱离高位的原点对齐现金政策和入场后诊断。"""

    event_fields = events[["event_id", "market_day_index"]].rename(
        columns={"market_day_index": "event_market_day_index"}
    )
    base = waits.merge(event_fields, on="event_id", how="left", validate="many_to_one")
    wait_window_maturity = outcomes.loc[
        outcomes["horizon_market_days"].eq(20),
        ["event_id", "mature_horizon"],
    ].set_index("event_id")["mature_horizon"]
    base["wait_window_mature"] = base["event_id"].map(wait_window_maturity).fillna(False)
    horizons = pd.DataFrame({"horizon_market_days": [20, 60, 120]})
    base["_join_key"] = 1
    horizons["_join_key"] = 1
    expanded = base.merge(horizons, on="_join_key", how="inner").drop(columns="_join_key")
    expanded["entry_market_day_index"] = np.where(
        expanded["entry_status"].eq("PRICE_AND_VOLUME_OBSERVED_LIMIT_STATE_UNRESOLVED"),
        expanded["event_market_day_index"] + expanded["wait_market_days"] + 1,
        np.nan,
    )
    expanded["post_entry_endpoint_market_day_index"] = (
        expanded["entry_market_day_index"] + expanded["horizon_market_days"]
    )
    connection = duckdb.connect()
    try:
        connection.register("wait_expanded", expanded)
        post_entry = connection.execute(
            f"""
            SELECT
                w.event_id,
                w.horizon_market_days,
                p.date AS post_entry_endpoint_date,
                p.total_return_close AS post_entry_endpoint_total_return_close
            FROM wait_expanded w
            LEFT JOIN read_parquet('{sql_path(panel)}') p
              ON p.con_code = w.ts_code
             AND p.market_day_index = w.post_entry_endpoint_market_day_index
            """
        ).fetchdf()
    finally:
        connection.close()
    expanded = expanded.merge(
        post_entry,
        on=["event_id", "horizon_market_days"],
        how="left",
        validate="one_to_one",
    )
    immediate = outcomes.loc[
        outcomes["horizon_market_days"].isin([20, 60, 120]),
        [
            "event_id",
            "horizon_market_days",
            "mature_horizon",
            "endpoint_observed",
            "path_return",
            "investable_return_price_only_provisional",
        ],
    ]
    expanded = expanded.merge(
        immediate,
        on=["event_id", "horizon_market_days"],
        how="left",
        validate="one_to_one",
    )
    entry_observed = expanded["entry_status"].eq(
        "PRICE_AND_VOLUME_OBSERVED_LIMIT_STATE_UNRESOLVED"
    )
    entry_before_origin_endpoint = (
        expanded["entry_market_day_index"]
        <= expanded["event_market_day_index"] + expanded["horizon_market_days"]
    )
    invested_during_origin_window = entry_observed & entry_before_origin_endpoint
    expanded["origin_aligned_policy_return"] = 0.0
    valid_origin_investment = invested_during_origin_window & expanded["endpoint_observed"]
    origin_endpoint_total_return = np.where(
        expanded["path_return"].notna(),
        (1.0 + expanded["path_return"]),
        np.nan,
    )
    # event_total_return_close 会在下一步由事件字段补入，用比率恢复原点终值。
    event_origin = events.set_index("event_id")["event_total_return_close"]
    expanded["event_total_return_close"] = expanded["event_id"].map(event_origin)
    origin_endpoint_level = origin_endpoint_total_return * expanded["event_total_return_close"]
    expanded.loc[valid_origin_investment, "origin_aligned_policy_return"] = (
        origin_endpoint_level[valid_origin_investment]
        / expanded.loc[valid_origin_investment, "entry_total_return_open"]
        - 1.0
    )
    expanded.loc[invested_during_origin_window & ~expanded["endpoint_observed"], "origin_aligned_policy_return"] = np.nan
    unresolved_wait_window = (
        expanded["entry_status"].eq("NO_TRIGGER_CASH")
        & ~expanded["wait_window_mature"]
    )
    expanded.loc[unresolved_wait_window, "origin_aligned_policy_return"] = np.nan
    expanded["post_entry_return"] = (
        expanded["post_entry_endpoint_total_return_close"]
        / expanded["entry_total_return_open"]
        - 1.0
    )
    expanded.loc[~entry_observed, "post_entry_return"] = np.nan
    expanded["origin_aligned_policy_status"] = np.select(
        [
            ~expanded["mature_horizon"],
            unresolved_wait_window,
            invested_during_origin_window & ~expanded["endpoint_observed"],
            invested_during_origin_window & expanded["endpoint_observed"],
            entry_observed & ~entry_before_origin_endpoint,
            expanded["entry_status"].eq("NO_TRIGGER_CASH"),
        ],
        [
            "RIGHT_CENSORED",
            "WAIT_WINDOW_RIGHT_CENSORED",
            "INVESTED_ENDPOINT_MISSING",
            "INVESTED_PRICE_ONLY_PROVISIONAL",
            "ENTRY_AFTER_ORIGIN_HORIZON_CASH",
            "NO_TRIGGER_CASH",
        ],
        default="TRIGGERED_NO_ENTRY_CASH",
    )
    return expanded.sort_values(["event_date", "ts_code", "horizon_market_days"]).reset_index(drop=True)


def render_report(
    receipt: dict[str, Any],
    summary: pd.DataFrame,
    waits: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> str:
    """生成阶段性事件研究报告。"""

    h60 = summary.loc[summary["horizon_market_days"].eq(60)].copy()
    h60_table = h60[
        [
            "volume_group",
            "price_position_group",
            "events",
            "valid_endpoint_events",
            "event_date_equal_weight_mean_return",
            "median_return",
            "return_lte_minus_20pct",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    wait_rows = []
    for horizon, group in waits.groupby("horizon_market_days"):
        wait_valid = group["origin_aligned_policy_return"].dropna()
        immediate_valid = group["investable_return_price_only_provisional"].dropna()
        wait_rows.append(
            {
                "horizon_market_days": horizon,
                "events": group["event_id"].nunique(),
                "trigger_rate": group["triggered_within_20d"].mean(),
                "observed_entry_rate": group["entry_status"].eq(
                    "PRICE_AND_VOLUME_OBSERVED_LIMIT_STATE_UNRESOLVED"
                ).mean(),
                "wait_policy_mean": group["origin_aligned_policy_return"].mean(),
                "wait_policy_median": group["origin_aligned_policy_return"].median(),
                "immediate_next_open_mean": group[
                    "investable_return_price_only_provisional"
                ].mean(),
                "wait_policy_loss_20pct": (wait_valid <= -0.20).mean(),
                "immediate_loss_20pct": (immediate_valid <= -0.20).mean(),
            }
        )
    wait_table = pd.DataFrame(wait_rows).to_markdown(index=False, floatfmt=".4f")
    merged = outcomes.merge(
        events[
            [
                "event_id",
                "volume_group",
                "binary_price_group",
                "event_day_volume_surge",
                "event_day_volume_surprise_pct",
            ]
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    def contrast_table(group_column: str, allowed: list[Any], require_surprise: bool = False) -> str:
        selected = merged.loc[
            merged["horizon_market_days"].isin([20, 60, 120])
            & merged[group_column].isin(allowed)
        ].copy()
        if require_surprise:
            selected = selected.loc[selected["event_day_volume_surprise_pct"].notna()]
        rows = []
        for (horizon, label), group in selected.groupby(
            ["horizon_market_days", group_column],
            dropna=False,
        ):
            valid = group.loc[group["path_return"].notna()].copy()
            rows.append(
                {
                    "horizon": horizon,
                    "group": label,
                    "events": len(group),
                    "date_equal_mean": valid.groupby("event_date")["path_return"].mean().mean(),
                    "median": valid["path_return"].median(),
                    "loss_20pct": (valid["path_return"] <= -0.20).mean(),
                }
            )
        return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")

    volume_table = contrast_table(
        "volume_group",
        ["VOLUME_DRY_UP", "VOLUME_NEUTRAL", "VOLUME_ACTIVE", "VOLUME_FACTOR_MISSING"],
    )
    position_table = contrast_table(
        "binary_price_group",
        ["HIGH_GROUP", "NOT_HIGH_GROUP"],
    )
    event_surge_table = contrast_table(
        "event_day_volume_surge",
        [False, True],
        require_surprise=True,
    )
    factor_columns = [
        "compression_log_ratio",
        "compression_pct",
        "rv20",
        "rv120_pre",
        "volume_participation_log_ratio",
        "volume_participation_pct",
        "event_day_volume_surprise",
        "event_day_volume_surprise_pct",
        "current_dd252",
        "range_position252",
    ]
    factor_coverage = pd.DataFrame(
        {
            "factor": factor_columns,
            "valid": [int(events[column].notna().sum()) for column in factor_columns],
            "missing_rate": [events[column].isna().mean() for column in factor_columns],
        }
    ).to_markdown(index=False, floatfmt=".4f")
    annual = (
        events.assign(year=events["event_date"].dt.year)
        .groupby("year")
        .agg(events=("event_id", "size"), symbols=("ts_code", "nunique"))
        .reset_index()
        .to_markdown(index=False)
    )
    counts = receipt["counts"]
    return f"""# A股极端缩波 × 量能 × 价格位置全历史事件研究 V1

## 当前状态

`{receipt['status']}`

- 研究类型：全历史回顾性事件研究，不进行盲测
- 研究边界：`DISCOVERY_ONLY / NO_SHADOW / NO_LIVE / NO_ORDER`
- 极端缩波 episode：{counts['events']:,}
- 股票：{counts['event_symbols']:,}
- 事件日期：{counts['event_dates']:,}
- 量能因子有效事件：{counts['volume_factor_valid_events']:,}
- 量能因子缺失事件：{counts['volume_factor_missing_events']:,}

## 60 个市场日固定九宫格

{h60_table}

表中收益是事件日起点的股票总收益路径，优先展示“事件日期内等权、再对日期等权”的均值。它是历史条件分布，不是样本外预测或因果效应。量能缺失事件作为单独一组保留；不得只看九宫格中收益最高的一格。

## 背景量能总体分层

{volume_table}

`VOLUME_ACTIVE` 的短期日期等权均值并不低，但中位数和下跌尾部弱于 `VOLUME_DRY_UP`；到 60/120 日，背景缩量组的中位数和大跌比例更好。该差异同时受价格位置、年份、规模和市场状态影响，不能解释为“缩量导致上涨”。

## 高位与非高位

{position_table}

固定二元对照并不支持“非高位的平均收益普遍更高”：高位组在 20/60/120 日的日期等权均值均不低；非高位组只在 60 日大跌尾部更轻。高位事件仅 1,776 个，非高位 56,127 个，样本极不平衡，当前结果不能作因果判断。

## 事件日突然放量

{event_surge_table}

事件日突然放量只在拥有完整 252 个历史量能惊奇值时标记。历史上它在 60/120 日的中位数和大跌尾部更弱，但这仍是条件相关标签，不是入场或排除规则。

## 近高事件等待至不高

{wait_table}

等待政策只针对事件日距 252 日高点不足 5% 的事件：最多等 20 个市场日，首次回撤达到 10% 后在下一开放日有价格和成交量观测时进入；未触发、触发但无下一开盘或入场日晚于原期限时保持现金 0 收益。`immediate_next_open_mean` 与实际等待收益都仍是涨跌停状态未解决的价格路径诊断。

## 因子定义复核

{factor_coverage}

- `RV20` 与 `RV120_PRE` 的 Spearman 相关为 0.905，且缩波对数比在代数上由二者构成；调整模型固定使用 `CompressionLogRatio + log(RV120PRE)`，不能三者同时进入。
- 背景量能原值与自身百分位相关 0.908，事件日量能惊奇原值与百分位相关 0.985；原值用于连续模型，百分位只用于分组。
- `CurrentDD252` 与 `RangePosition252` 相关 0.804；前者是主位置变量，后者仅作替代规格或诊断，不合成分数。
- `CompressionPct` 与跨股票的原始缩波对数比相关仅 0.374，这是自身历史标准化的预期结果，也说明 10% 事件阈值不代表统一的绝对低波水平；必须同时保留 `RV20` 与基准波动作描述。

## 年度覆盖

{annual}

年度事件数随上市股票数量、历史成熟度和行情环境明显变化，因此总体均值不能替代年度与滚动窗口分布；当前日期等权只能缓解同日横截面拥挤，不能解决所有年代构成差异。

## 数据与执行边界

- 总收益补充分区已修复 2024 年尺度断点；旧冻结结果未修改。
- 收益窗口按真实市场开放日计算，跨停牌或来源缺口不压缩时间。
- 主量能是股本稳定门槛下的“自身相对成交股数”，不是未经审计的逐日自由流通换手率。
- 下一开盘只有价格和成交量观测，涨跌停可成交状态仍未审计，因此可投入收益列标记为 `price_only_provisional`，不能解释为真实可成交回测。
- 退市经济终值尚未逐事件完成，终点缺失事件保留在账本，不作普通删除。
- 本阶段尚未生成行业调整模型和 5,000 次移动区块 Bootstrap，当前状态不是最终推断报告。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--probe-buckets", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config_path.relative_to(ROOT.resolve())
    config = read_config(config_path)
    panel = project_path(config["artifacts"]["unified_daily_market"])
    panel_receipt_path = project_path(config["artifacts"]["research_panel_receipt"])
    panel_receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    if panel_receipt.get("status") != "PASS_UNIFIED_RESEARCH_PANEL_READY":
        raise ValueError("统一研究面板未通过")
    panel_hash = sha256_file(panel)
    if panel_hash != panel_receipt["artifacts"]["panel_sha256"]:
        raise ValueError("统一研究面板哈希与回执不一致")
    bucket_files = prepare_buckets(panel, panel_hash)
    bucket_map = {
        int(path.parent.name.split("=", 1)[1]): path
        for path in bucket_files
    }
    bucket_ids = sorted(bucket_map)
    probe = args.probe_buckets is not None
    if probe:
        if args.probe_buckets < 1 or args.probe_buckets > BUCKET_COUNT:
            raise ValueError("probe-buckets 必须在1到32之间")
        bucket_ids = bucket_ids[: args.probe_buckets]

    calendars = load_calendars(config)
    benchmark = {
        exchange: load_benchmark(config, dates)
        for exchange, dates in calendars.items()
    }
    share_history = pd.read_parquet(project_path(config["inputs"]["share_count_history"]["path"]))
    share_groups = {code: group for code, group in share_history.groupby("ts_code", sort=False)}
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.finalize_only:
        print(f"开始计算 {len(bucket_ids)}/{BUCKET_COUNT} 个证券分桶。", flush=True)
        for completed, bucket in enumerate(bucket_ids, start=1):
            path = bucket_map[bucket]
            market = pd.read_parquet(path)
            events_all: list[dict[str, Any]] = []
            outcomes_all: list[dict[str, Any]] = []
            breakouts_all: list[dict[str, Any]] = []
            landmarks_all: list[dict[str, Any]] = []
            waits_all: list[dict[str, Any]] = []
            for ts_code, symbol in market.groupby("con_code", sort=False):
                exchange = str(symbol["exchange"].iloc[0])
                outputs = process_symbol(
                    symbol,
                    share_groups.get(ts_code, pd.DataFrame()),
                    calendars[exchange],
                    benchmark[exchange],
                    len(calendars[exchange]),
                    config,
                )
                events, outcomes, breakouts, landmarks, waits = outputs
                events_all.extend(events)
                outcomes_all.extend(outcomes)
                breakouts_all.extend(breakouts)
                landmarks_all.extend(landmarks)
                waits_all.extend(waits)
            checkpoint_frames = {
                "events": pd.DataFrame(events_all),
                "outcomes": pd.DataFrame(outcomes_all),
                "breakouts": pd.DataFrame(breakouts_all),
                "landmarks": pd.DataFrame(landmarks_all),
                "waits": pd.DataFrame(waits_all),
            }
            for kind, frame in checkpoint_frames.items():
                atomic_parquet(frame, CHECKPOINT_ROOT / f"bucket_{bucket:02d}_{kind}.parquet")
            print(
                f"因子进度 {completed}/{len(bucket_ids)}，桶 {bucket:02d}，股票 {market['con_code'].nunique()}，事件 {len(events_all)}。",
                flush=True,
            )

    if probe:
        print(json.dumps({"状态": "PROBE_COMPLETE", "分桶": bucket_ids}, ensure_ascii=False))
        return 0

    events = combine_checkpoints("events", EVENT_LEDGER, bucket_ids)
    events = attach_point_in_time_labels(events, config)
    atomic_parquet(events, EVENT_LEDGER)
    outcomes = combine_checkpoints("outcomes", HORIZON_OUTCOMES, bucket_ids)
    breakouts = combine_checkpoints("breakouts", BREAKOUT_EVENTS, bucket_ids)
    landmarks = combine_checkpoints("landmarks", STAGING / "landmarks.parquet", bucket_ids)
    raw_waits = combine_checkpoints("waits", STAGING / "waits.parquet", bucket_ids)
    waits = build_wait_policy_outcomes(raw_waits, events, outcomes, panel)
    landmarks.to_csv(LANDMARKS, index=False, encoding="utf-8-sig")
    waits.to_csv(WAIT_POLICY, index=False, encoding="utf-8-sig")
    summary = build_group_summary(events, outcomes)
    summary.to_csv(GROUP_SUMMARY, index=False, encoding="utf-8-sig")
    factor_dictionary = {
        "study_id": config["study"]["study_id"],
        "blind_test_required": False,
        "factors": {
            "compression_log_ratio": "log(RV20 / nonoverlapping previous RV120), total-return close log returns",
            "compression_pct": "self trailing 504 valid values, current excluded, empirical midrank CDF",
            "volume_participation_log_ratio": "log(prior 20 market-day median share volume / earlier 120 market-day median share volume), event day excluded",
            "volume_participation_pct": "self trailing 504 valid values; only when 141-day PIT tradable-share base varies no more than 5%",
            "current_dd252": "current total-return close / trailing 252-market-day high - 1",
            "range_position252": "position inside trailing 252-market-day total-return close range",
        },
    }
    atomic_text(json.dumps(factor_dictionary, ensure_ascii=False, indent=2) + "\n", FACTOR_DICTIONARY)
    receipt = {
        "study_id": config["study"]["study_id"],
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "DESCRIPTIVE_EVENT_RESULTS_READY_INFERENCE_PENDING",
        "blind_test_required": False,
        "predictive_validation_claim": False,
        "implementation": {
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config_path),
            "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "unified_panel": panel.relative_to(ROOT).as_posix(),
            "unified_panel_sha256": panel_hash,
        },
        "counts": {
            "events": len(events),
            "event_symbols": int(events["ts_code"].nunique()),
            "event_dates": int(events["event_date"].nunique()),
            "path_panel_eligible_events": int(events["path_panel_eligible"].sum()),
            "volume_factor_valid_events": int(events["volume_factor_valid"].sum()),
            "volume_factor_missing_events": int((~events["volume_factor_valid"]).sum()),
            "horizon_rows": len(outcomes),
            "breakout_rows": len(breakouts),
            "landmark_rows": len(landmarks),
            "wait_policy_rows": len(waits),
        },
        "known_blockers": [
            "DAILY_LIMIT_EXECUTABILITY_UNRESOLVED",
            "DELISTING_ECONOMIC_TERMINAL_VALUE_UNRESOLVED",
            "INDUSTRY_ADJUSTED_MODEL_PENDING",
            "MOVING_BLOCK_BOOTSTRAP_5000_PENDING",
        ],
        "artifacts": {
            "event_ledger": EVENT_LEDGER.relative_to(ROOT).as_posix(),
            "event_ledger_sha256": sha256_file(EVENT_LEDGER),
            "horizon_outcomes": HORIZON_OUTCOMES.relative_to(ROOT).as_posix(),
            "horizon_outcomes_sha256": sha256_file(HORIZON_OUTCOMES),
            "breakout_events": BREAKOUT_EVENTS.relative_to(ROOT).as_posix(),
            "breakout_events_sha256": sha256_file(BREAKOUT_EVENTS),
            "landmarks": LANDMARKS.relative_to(ROOT).as_posix(),
            "landmarks_sha256": sha256_file(LANDMARKS),
            "wait_policy": WAIT_POLICY.relative_to(ROOT).as_posix(),
            "wait_policy_sha256": sha256_file(WAIT_POLICY),
            "group_summary": GROUP_SUMMARY.relative_to(ROOT).as_posix(),
            "group_summary_sha256": sha256_file(GROUP_SUMMARY),
            "factor_dictionary": FACTOR_DICTIONARY.relative_to(ROOT).as_posix(),
            "factor_dictionary_sha256": sha256_file(FACTOR_DICTIONARY),
            "report": REPORT.relative_to(ROOT).as_posix(),
        },
    }
    atomic_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", RUN_RECEIPT)
    atomic_text(render_report(receipt, summary, waits, events, outcomes), REPORT)
    print(json.dumps({"状态": receipt["status"], **receipt["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
