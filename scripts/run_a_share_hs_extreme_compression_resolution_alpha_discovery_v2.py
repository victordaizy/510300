"""运行极端缩波后超额收益与第一次升波状态研究 V2。

本脚本严格复用 V1.2.1 的 64,729 个 episode 和冻结边界，不修改10%缩波
阈值。V2 新增市场波动水平/方向、全市场低波广度、个股第一次升波、P2/P3
预测原点以及沪深300和点时行业超额。全部结果仍为回顾性 DISCOVERY_ONLY。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.a_share_hs_extreme_compression_resolution_alpha_v2 import (  # noqa: E402
    classify_expansion_lead_lag,
    classify_expansion_price_state,
    compound_excess,
    compute_market_volatility_state,
    cumulative_quintile_membership,
    event_date_equal_mean,
    first_cross_after_origins,
    holm_adjust,
    realized_volatility,
    trailing_valid_midrank_percentile,
)


CONFIG = ROOT / "config/a_share_hs_extreme_compression_resolution_alpha_discovery_v2.yaml"
STUDY_ID = "A_SHARE_HS_EXTREME_COMPRESSION_RESOLUTION_ALPHA_DISCOVERY_V2"
TIME_ZONE = ZoneInfo("Asia/Shanghai")
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20260823


def project_path(relative: str) -> Path:
    """解析并约束项目内路径。"""

    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    """载入根协议和七份正式计算前合同，并校验冻结边界。"""

    root_contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if root_contract["study"]["study_id"] != STUDY_ID:
        raise RuntimeError("V2研究ID不匹配")
    if root_contract["study"]["protocol_status"] != "PROTOCOL_FROZEN_BEFORE_FIRST_V2_CALCULATION":
        raise RuntimeError("V2协议尚未冻结")
    if root_contract["study"]["data_cutoff"] != "2026-08-14":
        raise RuntimeError("数据截止日必须固定为2026-08-14")
    if not root_contract["governance"]["preserve_parent_events"]:
        raise RuntimeError("禁止重建或修改父事件")
    if root_contract["governance"]["best_quintile_selection_allowed"]:
        raise RuntimeError("禁止从市场波动五等分中挑选最佳箱体")
    contracts: dict[str, Any] = {}
    for name, relative in root_contract["contracts"].items():
        path = project_path(relative)
        if path.suffix.lower() == ".json":
            contracts[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            contracts[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
    family = contracts["primary_family"]
    if len(family["hypotheses"]) != 6 or family["multiple_testing"] != "HOLM_6":
        raise RuntimeError("主要检验家族必须固定为6项并使用Holm校正")
    sequence = contracts["nested_sequence"]
    if [item["id"] for item in sequence["models"]] != [f"M{i}" for i in range(7)]:
        raise RuntimeError("嵌套加法模型必须固定为M0至M6")
    return root_contract, contracts


def load_calendar(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    dates = (
        frame.loc[frame["is_open"] & frame["exchange"].eq("SSE"), "date"]
        .drop_duplicates()
        .sort_values()
    )
    dates = dates.loc[dates.le(cutoff)].reset_index(drop=True)
    return pd.DataFrame(
        {
            "market_day_index": np.arange(1, len(dates) + 1, dtype=np.int32),
            "market_date": dates,
        }
    )


def build_market_state(
    contract: dict[str, Any],
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    benchmark = pd.read_parquet(
        project_path(contract["inputs"]["h00300_total_return"]),
        columns=["date", "close"],
    )
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    aligned = calendar.merge(
        benchmark,
        left_on="market_date",
        right_on="date",
        how="left",
        validate="one_to_one",
    )[["market_day_index", "market_date", "close"]].rename(
        columns={"market_date": "date", "close": "h00300_close"}
    )
    state = compute_market_volatility_state(
        aligned[["date", "h00300_close"]].rename(columns={"h00300_close": "close"})
    ).rename(columns={"close": "h00300_close"})
    state.insert(0, "market_day_index", aligned["market_day_index"].to_numpy(dtype=np.int32))
    if state["h00300_close"].isna().any():
        missing = state.loc[state["h00300_close"].isna(), "date"]
        raise RuntimeError(f"H00300在统一交易日历缺失：{len(missing)}日")
    return state


def build_share_grid(
    snapshots: pd.DataFrame,
    calendar_dates: np.ndarray,
    global_indices: np.ndarray,
) -> np.ndarray:
    """按 effective_date 与 available_at 的较晚日期生成点时流通股本。"""

    output = np.full(global_indices.size, np.nan, dtype=float)
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
    available_positions = table["position"].to_numpy(dtype=np.int64)
    available_values = table["value"].to_numpy(dtype=float)
    lookup = np.searchsorted(available_positions, global_indices, side="right") - 1
    mask = lookup >= 0
    output[mask] = available_values[lookup[mask]]
    return output


def _stock_state_for_symbol(
    frame: pd.DataFrame,
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    calendar_dates: np.ndarray,
    cutoff_index: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """计算单只股票的绝对低波、5/20升波和事件窗口状态。"""

    frame = frame.sort_values("market_day_index").drop_duplicates("market_day_index", keep="last")
    start_index = int(frame["market_day_index"].min())
    global_indices = np.arange(start_index, cutoff_index + 1, dtype=np.int64)
    length = global_indices.size
    row_position = frame["market_day_index"].to_numpy(dtype=np.int64) - start_index
    observed = np.zeros(length, dtype=bool)
    observed[row_position] = (
        frame["observed_traded_row"].fillna(False).astype(bool).to_numpy()
        & pd.to_numeric(frame["total_return_close"], errors="coerce").notna().to_numpy()
    )

    def grid(column: str) -> np.ndarray:
        values = np.full(length, np.nan, dtype=float)
        values[row_position] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return values

    total_return_close = grid("total_return_close")
    raw_close = grid("raw_close")
    log_return = np.full(length, np.nan, dtype=float)
    consecutive = (
        observed[1:]
        & observed[:-1]
        & (total_return_close[1:] > 0)
        & (total_return_close[:-1] > 0)
    )
    log_return[1:][consecutive] = np.log(
        total_return_close[1:][consecutive] / total_return_close[:-1][consecutive]
    )
    rv5 = realized_volatility(log_return, window=5)
    rv20 = realized_volatility(log_return, window=20)
    ratio = np.full(length, np.nan, dtype=float)
    valid_ratio = np.isfinite(rv5) & np.isfinite(rv20) & (rv5 > 0) & (rv20 > 0)
    ratio[valid_ratio] = np.log(rv5[valid_ratio] / rv20[valid_ratio])
    rv20_percentile, rv20_history_count = trailing_valid_midrank_percentile(
        rv20,
        lookback_valid_values=756,
        minimum_valid_history=252,
    )
    previous_ratio = np.roll(ratio, 1)
    previous_ratio[0] = np.nan
    expansion_cross = (
        np.isfinite(previous_ratio)
        & np.isfinite(ratio)
        & (previous_ratio <= 0.0)
        & (ratio > 0.0)
    )
    share_grid = build_share_grid(snapshots, calendar_dates, global_indices)
    float_market_cap = raw_close * share_grid

    valid_breadth = observed & np.isfinite(rv20_percentile)
    lowvol = valid_breadth & (rv20_percentile <= 0.20)
    valid_float = valid_breadth & np.isfinite(float_market_cap) & (float_market_cap > 0)
    lowvol_float = valid_float & (rv20_percentile <= 0.20)
    aggregate = {
        "global_indices": global_indices,
        "valid_breadth": valid_breadth,
        "lowvol": lowvol,
        "valid_float": valid_float,
        "lowvol_float": lowvol_float,
        "float_market_cap": float_market_cap,
    }

    event_frames: list[pd.DataFrame] = []
    for event in events.itertuples(index=False):
        origin_global_index = int(event.market_day_index)
        origin_position = origin_global_index - start_index
        maximum_offset = min(60, cutoff_index - origin_global_index)
        if origin_position < 0 or maximum_offset < 0:
            continue
        positions = origin_position + np.arange(maximum_offset + 1, dtype=np.int64)
        target_indices = origin_global_index + np.arange(maximum_offset + 1, dtype=np.int64)
        target_dates = pd.to_datetime(calendar_dates[target_indices - 1])
        event_frames.append(
            pd.DataFrame(
                {
                    "event_id": event.event_id,
                    "ts_code": event.ts_code,
                    "event_date": pd.Timestamp(event.event_date).normalize(),
                    "day_offset": np.arange(maximum_offset + 1, dtype=np.int16),
                    "target_market_day_index": target_indices.astype(np.int32),
                    "target_date": target_dates,
                    "stock_observed": observed[positions],
                    "total_return_close": total_return_close[positions],
                    "stock_rv5": rv5[positions],
                    "stock_rv20": rv20[positions],
                    "stock_rv20_percentile_756": rv20_percentile[positions],
                    "stock_rv20_history_count": rv20_history_count[positions],
                    "stock_log_rv5_div_rv20": ratio[positions],
                    "stock_vol_falling": np.isfinite(ratio[positions]) & (ratio[positions] <= 0.0),
                    "stock_first_expansion_cross": expansion_cross[positions],
                    "point_in_time_float_market_cap": float_market_cap[positions],
                }
            )
        )
    if event_frames:
        event_path = pd.concat(event_frames, ignore_index=True)
    else:
        event_path = pd.DataFrame(
            columns=[
                "event_id",
                "ts_code",
                "event_date",
                "day_offset",
                "target_market_day_index",
                "target_date",
                "stock_observed",
                "total_return_close",
                "stock_rv5",
                "stock_rv20",
                "stock_rv20_percentile_756",
                "stock_rv20_history_count",
                "stock_log_rv5_div_rv20",
                "stock_vol_falling",
                "stock_first_expansion_cross",
                "point_in_time_float_market_cap",
            ]
        )
    return event_path, aggregate


def build_stock_state_and_breadth(
    contract: dict[str, Any],
    calendar: pd.DataFrame,
    market_state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """分桶计算全部股票的低波广度，并只保存episode的0至60日状态。"""

    events = pd.read_parquet(
        project_path(contract["inputs"]["parent_event_ledger"]),
        columns=[
            "event_id",
            "ts_code",
            "event_date",
            "market_day_index",
            "frozen_upper20",
            "frozen_lower20",
        ],
    )
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    event_groups = {str(code): group.copy() for code, group in events.groupby("ts_code", sort=False)}
    shares = pd.read_parquet(
        project_path(contract["inputs"]["share_count_history"]),
        columns=["ts_code", "effective_date", "available_at", "tradable_a_shares"],
    )
    share_groups = {str(code): group.copy() for code, group in shares.groupby("ts_code", sort=False)}
    del shares

    calendar_dates = calendar["market_date"].to_numpy(dtype="datetime64[ns]")
    cutoff_index = int(calendar["market_day_index"].max())
    valid_count = np.zeros(cutoff_index, dtype=np.int64)
    lowvol_count = np.zeros(cutoff_index, dtype=np.int64)
    float_valid_count = np.zeros(cutoff_index, dtype=np.int64)
    float_total = np.zeros(cutoff_index, dtype=float)
    float_lowvol = np.zeros(cutoff_index, dtype=float)

    output_root = project_path(contract["artifacts"]["root"])
    checkpoints = output_root / "stock_state_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    bucket_root = project_path(contract["inputs"]["factor_input_buckets"])
    bucket_paths = sorted(
        bucket_root.glob("symbol_bucket=*/*.parquet"),
        key=lambda path: int(path.parent.name.split("=")[-1]),
    )
    if not bucket_paths:
        raise RuntimeError("未找到V1冻结股票分桶")
    checkpoint_receipts: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for number, bucket_path in enumerate(bucket_paths, start=1):
        bucket_id = int(bucket_path.parent.name.split("=")[-1])
        print(f"计算股票低波与升波状态：分桶 {number}/{len(bucket_paths)}（{bucket_id}）", flush=True)
        frame = pd.read_parquet(
            bucket_path,
            columns=[
                "con_code",
                "market_day_index",
                "raw_close",
                "total_return_close",
                "observed_traded_row",
            ],
        )
        bucket_event_frames: list[pd.DataFrame] = []
        for code, symbol_frame in frame.groupby("con_code", sort=False):
            code = str(code)
            symbol_events = event_groups.get(code)
            if symbol_events is None:
                symbol_events = events.iloc[0:0]
            event_path, aggregate = _stock_state_for_symbol(
                symbol_frame,
                share_groups.get(code, pd.DataFrame()),
                symbol_events,
                calendar_dates,
                cutoff_index,
            )
            indices0 = aggregate["global_indices"] - 1
            valid = aggregate["valid_breadth"]
            low = aggregate["lowvol"]
            valid_float = aggregate["valid_float"]
            low_float = aggregate["lowvol_float"]
            np.add.at(valid_count, indices0[valid], 1)
            np.add.at(lowvol_count, indices0[low], 1)
            np.add.at(float_valid_count, indices0[valid_float], 1)
            np.add.at(float_total, indices0[valid_float], aggregate["float_market_cap"][valid_float])
            np.add.at(float_lowvol, indices0[low_float], aggregate["float_market_cap"][low_float])
            if not event_path.empty:
                bucket_event_frames.append(event_path)
                seen_event_ids.update(event_path["event_id"].unique().tolist())
        bucket_output = (
            pd.concat(bucket_event_frames, ignore_index=True)
            if bucket_event_frames
            else pd.DataFrame(columns=["event_id"])
        )
        checkpoint = checkpoints / f"stock_state_bucket_{bucket_id:02d}.parquet"
        atomic_parquet(bucket_output, checkpoint)
        checkpoint_receipts.append(
            {
                "bucket_id": bucket_id,
                "input": str(bucket_path.relative_to(ROOT)).replace("\\", "/"),
                "input_sha256": sha256_file(bucket_path),
                "output": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "output_rows": int(len(bucket_output)),
                "output_sha256": sha256_file(checkpoint),
            }
        )
        del frame, bucket_event_frames, bucket_output

    missing_events = sorted(set(events["event_id"]) - seen_event_ids)
    if missing_events:
        raise RuntimeError(f"股票波动状态未覆盖全部父事件：{len(missing_events)}")

    consolidated = project_path(contract["artifacts"]["stock_volatility_event_path"])
    temporary = consolidated.with_suffix(".tmp.parquet")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='4GB'")
        pattern = sql_path(checkpoints / "stock_state_bucket_*.parquet")
        connection.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{pattern}')
                ORDER BY event_date, ts_code, day_offset
            ) TO '{sql_path(temporary)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """
        )
    finally:
        connection.close()
    os.replace(temporary, consolidated)

    result = market_state.copy()
    result["lowvol_breadth_eligible_count"] = valid_count
    result["lowvol_breadth_low_count"] = lowvol_count
    result["lowvol_breadth_equal_weight"] = np.divide(
        lowvol_count,
        valid_count,
        out=np.full(cutoff_index, np.nan, dtype=float),
        where=valid_count > 0,
    )
    result["lowvol_breadth_float_coverage"] = np.divide(
        float_valid_count,
        valid_count,
        out=np.full(cutoff_index, np.nan, dtype=float),
        where=valid_count > 0,
    )
    result["lowvol_breadth_float_weight"] = np.divide(
        float_lowvol,
        float_total,
        out=np.full(cutoff_index, np.nan, dtype=float),
        where=float_total > 0,
    )
    receipt = {
        "bucket_count": len(bucket_paths),
        "event_count": int(len(events)),
        "stock_event_path_rows": int(
            duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{sql_path(consolidated)}')").fetchone()[0]
        ),
        "market_dates_with_equal_weight_breadth": int(result["lowvol_breadth_equal_weight"].notna().sum()),
        "market_dates_with_float_weight_breadth": int(result["lowvol_breadth_float_weight"].notna().sum()),
        "minimum_float_coverage": float(result["lowvol_breadth_float_coverage"].dropna().min()),
        "checkpoints": checkpoint_receipts,
    }
    return result, receipt


def _breakout_base_status(direction: str, identification: str, required: str) -> str:
    if identification == "BREAKOUT_WINDOW_RIGHT_CENSORED":
        return "RIGHT_CENSORED_BREAKOUT_WINDOW"
    if identification in {
        "FIRST_BREAK_TIME_POTENTIALLY_UNOBSERVED",
        "BREAKOUT_UNRESOLVED_INCOMPLETE_PATH",
    }:
        return "NO_VIEW_BREAKOUT_PATH"
    if identification == "NO_BREAK_WITHIN_60D_COMPLETE_PATH":
        return "NO_ENTRY_NO_BREAK_WITHIN_60D"
    if identification != "FIRST_BREAK_FULLY_OBSERVED_TO_BREAK":
        return "NO_VIEW_UNKNOWN_BREAKOUT_STATUS"
    if direction != required:
        return f"NO_ENTRY_FIRST_BREAK_{direction}"
    return "ELIGIBLE_DIRECTION_PATH"


def build_policy_origin_ledger(
    contract: dict[str, Any],
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从冻结episode生成P0/P1/P2/P3各自唯一预测原点。"""

    event_columns = [
        "event_id",
        "ts_code",
        "event_date",
        "market_day_index",
        "event_total_return_close",
        "frozen_upper20",
        "frozen_lower20",
        "industry_l1",
        "industry_l1_code",
    ]
    events = pd.read_parquet(
        project_path(contract["inputs"]["parent_event_ledger"]),
        columns=event_columns,
    )
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    breakouts = pd.read_parquet(
        project_path(contract["inputs"]["parent_breakouts"]),
        columns=[
            "event_id",
            "first_break_date",
            "time_to_first_break_market_days",
            "breakout_identification_status",
            "first_break_direction_v1_2",
        ],
    )
    merged = events.merge(breakouts, on="event_id", how="left", validate="one_to_one")
    merged["first_break_date"] = pd.to_datetime(merged["first_break_date"]).dt.normalize()
    path_file = project_path(contract["inputs"]["parent_episode_path"])
    requested = merged.loc[
        merged["time_to_first_break_market_days"].notna(),
        ["event_id", "time_to_first_break_market_days"],
    ].copy()
    requested["break_offset"] = requested["time_to_first_break_market_days"].astype(int)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='4GB'")
        connection.register("break_requests", requested)
        break_and_confirm = connection.execute(
            f"""
            SELECT
                r.event_id,
                r.break_offset,
                b.target_date AS break_path_date,
                b.target_market_day_index AS break_market_day_index,
                b.total_return_close AS break_total_return_close,
                b.stock_observed AS break_stock_observed,
                b.market_day_mature AS break_market_day_mature,
                c.target_date AS confirm_date,
                c.target_market_day_index AS confirm_market_day_index,
                c.total_return_close AS confirm_total_return_close,
                c.stock_observed AS confirm_stock_observed,
                c.market_day_mature AS confirm_market_day_mature
            FROM break_requests r
            LEFT JOIN read_parquet('{sql_path(path_file)}') b
              ON b.event_id = r.event_id AND b.day_offset = r.break_offset
            LEFT JOIN read_parquet('{sql_path(path_file)}') c
              ON c.event_id = r.event_id AND c.day_offset = r.break_offset + 1
            """
        ).fetchdf()
        down_requests = merged.loc[
            merged["first_break_direction_v1_2"].eq("DOWN_BREAK")
            & merged["time_to_first_break_market_days"].notna(),
            ["event_id", "time_to_first_break_market_days", "frozen_lower20"],
        ].copy()
        down_requests["break_offset"] = down_requests["time_to_first_break_market_days"].astype(int)
        connection.unregister("break_requests")
        connection.register("down_requests", down_requests)
        reclaim_window = connection.execute(
            f"""
            SELECT
                r.event_id,
                r.break_offset,
                p.day_offset,
                p.target_date,
                p.target_market_day_index,
                p.total_return_close,
                p.stock_observed,
                p.market_day_mature,
                r.frozen_lower20
            FROM down_requests r
            LEFT JOIN read_parquet('{sql_path(path_file)}') p
              ON p.event_id = r.event_id
             AND p.day_offset BETWEEN r.break_offset + 1 AND r.break_offset + 5
            ORDER BY r.event_id, p.day_offset
            """
        ).fetchdf()
    finally:
        connection.close()
    merged = merged.merge(break_and_confirm, on="event_id", how="left", validate="one_to_one")

    common_columns = [
        "event_id",
        "ts_code",
        "event_date",
        "market_day_index",
        "frozen_upper20",
        "frozen_lower20",
        "industry_l1",
        "industry_l1_code",
    ]
    policy_frames: list[pd.DataFrame] = []

    p0 = merged[common_columns + ["event_total_return_close"]].copy()
    p0["policy_id"] = "P0_COMPRESSION_D0_BASELINE"
    p0["policy_status"] = np.where(
        p0["event_total_return_close"].notna(),
        "ENTRY_TRIGGERED_PRICE_ORIGIN_OBSERVED",
        "NO_VIEW_EVENT_CLOSE_MISSING",
    )
    p0["entry_triggered"] = p0["event_total_return_close"].notna()
    p0["forecast_origin_date"] = p0["event_date"]
    p0["forecast_origin_market_day_index"] = p0["market_day_index"]
    p0["forecast_origin_total_return_close"] = p0["event_total_return_close"]
    p0["wait_market_days"] = 0.0
    p0["first_break_direction"] = merged["first_break_direction_v1_2"].to_numpy()
    p0["first_break_date"] = merged["first_break_date"].to_numpy()
    p0 = p0.drop(columns=["event_total_return_close"])
    policy_frames.append(p0)

    for policy_id, required_direction in (
        ("P1_FIRST_UP_BREAK_BASELINE", "UP_BREAK"),
        ("P2_UP_BREAK_PERSISTENCE_2C", "UP_BREAK"),
    ):
        frame = merged[common_columns].copy()
        base_status = [
            _breakout_base_status(direction, status, required_direction)
            for direction, status in zip(
                merged["first_break_direction_v1_2"].astype(str),
                merged["breakout_identification_status"].astype(str),
            )
        ]
        frame["policy_id"] = policy_id
        frame["first_break_direction"] = merged["first_break_direction_v1_2"].to_numpy()
        frame["first_break_date"] = merged["first_break_date"].to_numpy()
        if policy_id.startswith("P1_"):
            status = np.asarray(base_status, dtype=object)
            eligible = status == "ELIGIBLE_DIRECTION_PATH"
            observed_origin = (
                merged["break_stock_observed"].fillna(False).astype(bool).to_numpy()
                & merged["break_market_day_mature"].fillna(False).astype(bool).to_numpy()
                & merged["break_total_return_close"].notna().to_numpy()
            )
            status[eligible & observed_origin] = "ENTRY_TRIGGERED_PRICE_ORIGIN_OBSERVED"
            status[eligible & ~observed_origin] = "NO_VIEW_BREAK_CLOSE_MISSING"
            frame["entry_triggered"] = status == "ENTRY_TRIGGERED_PRICE_ORIGIN_OBSERVED"
            frame["forecast_origin_date"] = merged["break_path_date"]
            frame["forecast_origin_market_day_index"] = merged["break_market_day_index"]
            frame["forecast_origin_total_return_close"] = merged["break_total_return_close"]
            frame["wait_market_days"] = merged["time_to_first_break_market_days"]
        else:
            status = np.asarray(base_status, dtype=object)
            eligible = status == "ELIGIBLE_DIRECTION_PATH"
            mature = merged["confirm_market_day_mature"].fillna(False).astype(bool).to_numpy()
            observed = (
                merged["confirm_stock_observed"].fillna(False).astype(bool).to_numpy()
                & merged["confirm_total_return_close"].notna().to_numpy()
            )
            persistent = observed & (
                pd.to_numeric(merged["confirm_total_return_close"], errors="coerce").to_numpy(dtype=float)
                > pd.to_numeric(merged["frozen_upper20"], errors="coerce").to_numpy(dtype=float)
            )
            status[eligible & ~mature] = "RIGHT_CENSORED_CONFIRMATION_DAY"
            status[eligible & mature & ~observed] = "NO_VIEW_CONFIRMATION_CLOSE_MISSING"
            status[eligible & mature & observed & ~persistent] = "NO_ENTRY_UP_BREAK_NOT_PERSISTENT"
            status[eligible & mature & persistent] = "ENTRY_TRIGGERED_PRICE_ORIGIN_OBSERVED"
            frame["entry_triggered"] = status == "ENTRY_TRIGGERED_PRICE_ORIGIN_OBSERVED"
            frame["forecast_origin_date"] = merged["confirm_date"]
            frame["forecast_origin_market_day_index"] = merged["confirm_market_day_index"]
            frame["forecast_origin_total_return_close"] = merged["confirm_total_return_close"]
            frame["wait_market_days"] = merged["time_to_first_break_market_days"] + 1.0
        frame["policy_status"] = status
        not_triggered = ~frame["entry_triggered"]
        frame.loc[not_triggered, [
            "forecast_origin_date",
            "forecast_origin_market_day_index",
            "forecast_origin_total_return_close",
            "wait_market_days",
        ]] = [pd.NaT, np.nan, np.nan, np.nan]
        policy_frames.append(frame)

    reclaim_by_event = {event_id: group.sort_values("day_offset") for event_id, group in reclaim_window.groupby("event_id", sort=False)}
    p3 = merged[common_columns].copy()
    p3["policy_id"] = "P3_DOWN_BREAK_RECLAIM_5D"
    p3["first_break_direction"] = merged["first_break_direction_v1_2"].to_numpy()
    p3["first_break_date"] = merged["first_break_date"].to_numpy()
    p3_status: list[str] = []
    p3_triggered: list[bool] = []
    p3_origin_date: list[pd.Timestamp | pd.NaT] = []
    p3_origin_index: list[float] = []
    p3_origin_close: list[float] = []
    p3_wait: list[float] = []
    for row in merged.itertuples(index=False):
        base = _breakout_base_status(
            str(row.first_break_direction_v1_2),
            str(row.breakout_identification_status),
            "DOWN_BREAK",
        )
        if base != "ELIGIBLE_DIRECTION_PATH":
            p3_status.append(base)
            p3_triggered.append(False)
            p3_origin_date.append(pd.NaT)
            p3_origin_index.append(np.nan)
            p3_origin_close.append(np.nan)
            p3_wait.append(np.nan)
            continue
        window = reclaim_by_event.get(row.event_id)
        if window is None or window.empty:
            p3_status.append("NO_VIEW_RECLAIM_WINDOW_MISSING")
            p3_triggered.append(False)
            p3_origin_date.append(pd.NaT)
            p3_origin_index.append(np.nan)
            p3_origin_close.append(np.nan)
            p3_wait.append(np.nan)
            continue
        window = window.sort_values("day_offset").reset_index(drop=True)
        mature = window["market_day_mature"].fillna(False).astype(bool)
        observed = window["stock_observed"].fillna(False).astype(bool) & window["total_return_close"].notna()
        reclaim = observed & (
            pd.to_numeric(window["total_return_close"], errors="coerce")
            > pd.to_numeric(window["frozen_lower20"], errors="coerce")
        )
        reclaim_positions = np.flatnonzero(reclaim.to_numpy())
        if reclaim_positions.size:
            candidate_position = int(reclaim_positions[0])
            required = window.iloc[: candidate_position + 1]
            if not required["market_day_mature"].fillna(False).astype(bool).all():
                status = "RIGHT_CENSORED_BEFORE_RECLAIM"
                triggered = False
            elif not (
                required["stock_observed"].fillna(False).astype(bool)
                & required["total_return_close"].notna()
            ).all():
                status = "NO_VIEW_GAP_BEFORE_RECLAIM"
                triggered = False
            else:
                status = "ENTRY_TRIGGERED_PRICE_ORIGIN_OBSERVED"
                triggered = True
                candidate = window.iloc[candidate_position]
        elif not mature.all():
            status = "RIGHT_CENSORED_RECLAIM_WINDOW"
            triggered = False
        elif not observed.all():
            status = "NO_VIEW_RECLAIM_WINDOW_GAP"
            triggered = False
        else:
            status = "NO_ENTRY_NO_RECLAIM_WITHIN_5D"
            triggered = False
        p3_status.append(status)
        p3_triggered.append(triggered)
        if triggered:
            p3_origin_date.append(pd.Timestamp(candidate["target_date"]).normalize())
            p3_origin_index.append(float(candidate["target_market_day_index"]))
            p3_origin_close.append(float(candidate["total_return_close"]))
            p3_wait.append(float(candidate["day_offset"]))
        else:
            p3_origin_date.append(pd.NaT)
            p3_origin_index.append(np.nan)
            p3_origin_close.append(np.nan)
            p3_wait.append(np.nan)
    p3["policy_status"] = p3_status
    p3["entry_triggered"] = p3_triggered
    p3["forecast_origin_date"] = p3_origin_date
    p3["forecast_origin_market_day_index"] = p3_origin_index
    p3["forecast_origin_total_return_close"] = p3_origin_close
    p3["wait_market_days"] = p3_wait
    policy_frames.append(p3)

    output = pd.concat(policy_frames, ignore_index=True)
    output["forecast_origin_date"] = pd.to_datetime(output["forecast_origin_date"]).dt.normalize()
    output["forecast_origin_market_day_index"] = pd.to_numeric(
        output["forecast_origin_market_day_index"], errors="coerce"
    ).astype("Int64")
    if output.duplicated(["event_id", "policy_id"]).any():
        raise RuntimeError("政策预测原点主键重复")
    if len(output) != len(events) * 4:
        raise RuntimeError("政策预测原点未覆盖全部episode×政策")

    triggered = output.loc[output["entry_triggered"], ["event_id", "ts_code", "forecast_origin_date"]].copy()
    connection = duckdb.connect()
    try:
        connection.register("triggered_origins", triggered)
        industry = connection.execute(
            f"""
            SELECT r.event_id, r.ts_code, r.forecast_origin_date,
                   p.industry_l1 AS origin_industry_l1,
                   p.industry_l1_code AS origin_industry_l1_code
            FROM triggered_origins r
            LEFT JOIN read_parquet('{sql_path(project_path(contract['inputs']['point_in_time_industry_daily']))}') p
              ON p.con_code = r.ts_code AND p.date = r.forecast_origin_date
            """
        ).fetchdf()
    finally:
        connection.close()
    output = output.merge(
        industry,
        on=["event_id", "ts_code", "forecast_origin_date"],
        how="left",
        validate="many_to_one",
    )
    output["origin_industry_l1"] = output["origin_industry_l1"].fillna(output["industry_l1"])
    output["origin_industry_l1_code"] = output["origin_industry_l1_code"].fillna(output["industry_l1_code"])
    output = output.sort_values(["event_date", "ts_code", "policy_id"]).reset_index(drop=True)
    receipt = {
        "rows": int(len(output)),
        "events": int(output["event_id"].nunique()),
        "policies": {
            policy: {
                "triggered": int(group["entry_triggered"].sum()),
                "no_entry": int(group["policy_status"].str.startswith("NO_ENTRY").sum()),
                "no_view": int(group["policy_status"].str.startswith("NO_VIEW").sum()),
                "right_censored": int(group["policy_status"].str.startswith("RIGHT_CENSORED").sum()),
            }
            for policy, group in output.groupby("policy_id", sort=True)
        },
    }
    return output, receipt


def build_first_expansion_ledger(
    contract: dict[str, Any],
    calendar: pd.DataFrame,
    market_state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """识别事件后60日内市场和个股第一次从降波穿越到升波。"""

    events = pd.read_parquet(
        project_path(contract["inputs"]["parent_event_ledger"]),
        columns=[
            "event_id",
            "ts_code",
            "event_date",
            "market_day_index",
            "frozen_upper20",
            "frozen_lower20",
        ],
    )
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    stock_path = project_path(contract["artifacts"]["stock_volatility_event_path"])
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='4GB'")
        connection.register("event_origins", events)
        stock_summary = connection.execute(
            f"""
            WITH first_cross AS (
                SELECT event_id,
                       MIN(day_offset) FILTER (
                           WHERE day_offset > 0 AND stock_first_expansion_cross
                       ) AS stock_first_expansion_offset
                FROM read_parquet('{sql_path(stock_path)}')
                GROUP BY event_id
            ), required_path AS (
                SELECT
                    e.event_id,
                    c.stock_first_expansion_offset,
                    SUM(
                        CASE WHEN p.day_offset > 0
                                  AND p.day_offset <= COALESCE(c.stock_first_expansion_offset, 60)
                                  AND NOT p.stock_observed
                             THEN 1 ELSE 0 END
                    ) AS missing_before_transition_or_window
                FROM event_origins e
                LEFT JOIN first_cross c USING (event_id)
                LEFT JOIN read_parquet('{sql_path(stock_path)}') p USING (event_id)
                GROUP BY e.event_id, c.stock_first_expansion_offset
            )
            SELECT
                e.event_id,
                r.stock_first_expansion_offset,
                r.missing_before_transition_or_window,
                p.target_date AS stock_first_expansion_date,
                p.target_market_day_index AS stock_first_expansion_market_day_index,
                p.total_return_close AS stock_first_expansion_close,
                p.stock_log_rv5_div_rv20 AS stock_ratio_at_first_expansion,
                p.stock_observed AS stock_transition_observed,
                n.stock_observed AS next_open_proxy_row_observed
            FROM event_origins e
            LEFT JOIN required_path r USING (event_id)
            LEFT JOIN read_parquet('{sql_path(stock_path)}') p
              ON p.event_id = e.event_id
             AND p.day_offset = r.stock_first_expansion_offset
            LEFT JOIN read_parquet('{sql_path(stock_path)}') n
              ON n.event_id = e.event_id
             AND n.day_offset = r.stock_first_expansion_offset + 1
            """
        ).fetchdf()
    finally:
        connection.close()
    output = events.merge(stock_summary, on="event_id", how="left", validate="one_to_one")

    cross_positions = market_state.loc[
        market_state["market_first_expansion_cross"], "market_day_index"
    ].to_numpy(dtype=np.int64)
    market_position, market_offset = first_cross_after_origins(
        cross_positions,
        output["market_day_index"].to_numpy(dtype=np.int64),
        maximum_offset=60,
    )
    cutoff_index = int(calendar["market_day_index"].max())
    calendar_date_lookup = calendar.set_index("market_day_index")["market_date"]
    output["market_first_expansion_market_day_index"] = pd.Series(
        np.where(market_position >= 0, market_position, np.nan), dtype="Float64"
    ).astype("Int64")
    output["market_first_expansion_offset"] = np.where(market_offset >= 0, market_offset, np.nan)
    output["market_first_expansion_date"] = pd.to_datetime(
        output["market_first_expansion_market_day_index"].map(calendar_date_lookup)
    ).dt.normalize()
    output["market_search_mature_60d"] = output["market_day_index"] + 60 <= cutoff_index
    output["stock_search_mature_60d"] = output["market_day_index"] + 60 <= cutoff_index

    event_market_columns = [
        "market_day_index",
        "market_vol_percentile_756",
        "market_vol_quintile",
        "market_log_rv5_div_rv20",
        "market_vol_falling",
        "market_low_and_falling",
        "lowvol_breadth_equal_weight",
        "lowvol_breadth_float_weight",
    ]
    event_market = market_state[event_market_columns].rename(
        columns={column: f"event_{column}" for column in event_market_columns if column != "market_day_index"}
    )
    output = output.merge(event_market, on="market_day_index", how="left", validate="many_to_one")

    stock_status: list[str] = []
    stock_no_view_reason: list[str] = []
    price_states: list[str] = []
    lead_lag_states: list[str] = []
    for row in output.itertuples(index=False):
        has_stock_cross = pd.notna(row.stock_first_expansion_offset)
        missing = int(row.missing_before_transition_or_window or 0)
        if has_stock_cross and missing == 0 and bool(row.stock_transition_observed):
            status = "FIRST_EXPANSION_OBSERVED"
            reason = "NONE"
            price_state = classify_expansion_price_state(
                float(row.stock_first_expansion_close),
                float(row.frozen_lower20),
                float(row.frozen_upper20),
            )
            stock_offset = float(row.stock_first_expansion_offset)
        elif has_stock_cross and missing > 0:
            status = "NO_VIEW"
            reason = "MISSING_STOCK_PATH_BEFORE_FIRST_OBSERVED_CROSS"
            price_state = "EXPANSION_PATH_UNRESOLVED"
            stock_offset = np.nan
        elif not bool(row.stock_search_mature_60d):
            status = "RIGHT_CENSORED"
            reason = "SEARCH_WINDOW_NOT_MATURE_AT_DATA_CUTOFF"
            price_state = "EXPANSION_PATH_UNRESOLVED"
            stock_offset = np.nan
        elif missing > 0:
            status = "NO_VIEW"
            reason = "MISSING_STOCK_PATH_WITHOUT_OBSERVED_CROSS"
            price_state = "EXPANSION_PATH_UNRESOLVED"
            stock_offset = np.nan
        else:
            status = "NO_EXPANSION_WITHIN_60D"
            reason = "NONE"
            price_state = "EXPANSION_NOT_OBSERVED"
            stock_offset = np.nan
        market_value = float(row.market_first_expansion_offset) if pd.notna(row.market_first_expansion_offset) else np.nan
        stock_status.append(status)
        stock_no_view_reason.append(reason)
        price_states.append(price_state)
        lead_lag_states.append(classify_expansion_lead_lag(stock_offset, market_value))
    output["stock_first_expansion_status"] = stock_status
    output["no_view_reason"] = stock_no_view_reason
    output["first_expansion_price_state"] = price_states
    output["first_expansion_direction"] = price_states
    output["stock_minus_market_expansion_lag"] = (
        pd.to_numeric(output["stock_first_expansion_offset"], errors="coerce")
        - pd.to_numeric(output["market_first_expansion_offset"], errors="coerce")
    )
    output["expansion_lead_lag_state"] = lead_lag_states
    output["first_expansion_path_complete"] = output["stock_first_expansion_status"].eq(
        "FIRST_EXPANSION_OBSERVED"
    )
    output["next_open_observed"] = (
        output["first_expansion_path_complete"]
        & output["next_open_proxy_row_observed"].fillna(False).astype(bool)
    )
    output["right_censored"] = output["stock_first_expansion_status"].eq("RIGHT_CENSORED")
    output = output.rename(columns={"event_market_day_index": "event_market_day_index_unused"})
    keep = [
        "event_id",
        "ts_code",
        "event_date",
        "market_day_index",
        "event_market_vol_percentile_756",
        "event_market_vol_quintile",
        "event_market_log_rv5_div_rv20",
        "event_market_vol_falling",
        "event_market_low_and_falling",
        "event_lowvol_breadth_equal_weight",
        "event_lowvol_breadth_float_weight",
        "market_first_expansion_date",
        "stock_first_expansion_date",
        "market_first_expansion_offset",
        "stock_first_expansion_offset",
        "stock_minus_market_expansion_lag",
        "expansion_lead_lag_state",
        "stock_first_expansion_status",
        "stock_first_expansion_close",
        "stock_ratio_at_first_expansion",
        "first_expansion_price_state",
        "first_expansion_direction",
        "first_expansion_path_complete",
        "next_open_observed",
        "right_censored",
        "no_view_reason",
    ]
    output = output[keep].sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    receipt = {
        "rows": int(len(output)),
        "stock_transition_status": output["stock_first_expansion_status"].value_counts().to_dict(),
        "price_states": output["first_expansion_price_state"].value_counts().to_dict(),
        "lead_lag_states": output["expansion_lead_lag_state"].value_counts().to_dict(),
    }
    return output, receipt


def build_event_market_origin_ledger(
    policy_origins: pd.DataFrame,
    market_state: pd.DataFrame,
) -> pd.DataFrame:
    """同时保存episode原点和各政策预测原点的市场状态。"""

    fields = [
        "date",
        "market_day_index",
        "market_rv5",
        "market_rv20",
        "market_rv120",
        "market_vol_percentile_756",
        "market_vol_quintile",
        "market_log_rv5_div_rv20",
        "market_vol_falling",
        "market_vol_expanding",
        "market_first_expansion_cross",
        "market_low_and_falling",
        "lowvol_breadth_equal_weight",
        "lowvol_breadth_float_weight",
        "lowvol_breadth_float_coverage",
        "data_available_at",
        "data_status",
    ]
    state = market_state[fields].copy()
    event_state = state.add_prefix("event_").rename(
        columns={"event_date": "event_date", "event_market_day_index": "market_day_index"}
    )
    output = policy_origins.merge(
        event_state,
        on=["event_date", "market_day_index"],
        how="left",
        validate="many_to_one",
    )
    origin_state = state.add_prefix("origin_").rename(
        columns={
            "origin_date": "forecast_origin_date",
            "origin_market_day_index": "forecast_origin_market_day_index",
        }
    )
    output = output.merge(
        origin_state,
        on=["forecast_origin_date", "forecast_origin_market_day_index"],
        how="left",
        validate="many_to_one",
    )
    return output.sort_values(["event_date", "ts_code", "policy_id"]).reset_index(drop=True)


def compute_point_in_time_industry_peer_return(
    requests: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    """按政策原点的点时行业固定同行，剔除自身并计算端点收益。"""

    keys = ["event_id", "policy_id", "horizon_market_days"]
    valid = requests.dropna(
        subset=[
            "ts_code",
            "forecast_origin_date",
            "target_date",
            "origin_industry_l1_code",
            "stock_close_return",
        ]
    ).copy()
    if valid.empty:
        return requests[keys].assign(
            industry_peer_count=0,
            industry_peer_equal_weight_return=np.nan,
            industry_peer_compound_excess_return=np.nan,
        )
    valid["forecast_origin_date"] = pd.to_datetime(valid["forecast_origin_date"]).dt.normalize()
    valid["target_date"] = pd.to_datetime(valid["target_date"]).dt.normalize()
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='4GB'")
        connection.register("peer_requests", valid)
        results = connection.execute(
            f"""
            SELECT
                r.event_id,
                r.policy_id,
                r.horizon_market_days,
                COUNT(*) AS industry_peer_count,
                AVG(p1.total_return_close / p0.total_return_close - 1.0)
                    AS industry_peer_equal_weight_return
            FROM peer_requests r
            INNER JOIN read_parquet('{sql_path(project_path(contract['inputs']['point_in_time_industry_daily']))}') p0
              ON p0.date = r.forecast_origin_date
             AND p0.industry_l1_code = r.origin_industry_l1_code
             AND p0.con_code <> r.ts_code
             AND p0.total_return_close > 0
            INNER JOIN read_parquet('{sql_path(project_path(contract['inputs']['unified_daily_market']))}') p1
              ON p1.con_code = p0.con_code
             AND p1.date = r.target_date
             AND p1.total_return_close > 0
            GROUP BY r.event_id, r.policy_id, r.horizon_market_days
            """
        ).fetchdf()
    finally:
        connection.close()
    output = requests[keys + ["stock_close_return"]].merge(
        results,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    output["industry_peer_count"] = output["industry_peer_count"].fillna(0).astype(int)
    valid_peer = output["industry_peer_count"].ge(5)
    output["industry_peer_equal_weight_return"] = output[
        "industry_peer_equal_weight_return"
    ].where(valid_peer)
    output["industry_peer_compound_excess_return"] = compound_excess(
        output["stock_close_return"].to_numpy(dtype=float),
        output["industry_peer_equal_weight_return"].to_numpy(dtype=float),
    )
    output.loc[~valid_peer, "industry_peer_compound_excess_return"] = np.nan
    return output.drop(columns=["stock_close_return"])


def build_policy_outcomes(
    contract: dict[str, Any],
    calendar: pd.DataFrame,
    market_state: pd.DataFrame,
    policy_origins: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """计算各政策预测原点之后20/60日的股票、指数和行业超额。"""

    horizons = pd.DataFrame({"horizon_market_days": [20, 60]})
    origins = policy_origins.assign(_join_key=1).merge(
        horizons.assign(_join_key=1), on="_join_key", how="inner"
    ).drop(columns="_join_key")
    cutoff_index = int(calendar["market_day_index"].max())
    maximum_wait = {
        "P0_COMPRESSION_D0_BASELINE": 0,
        "P1_FIRST_UP_BREAK_BASELINE": 60,
        "P2_UP_BREAK_PERSISTENCE_2C": 61,
        "P3_DOWN_BREAK_RECLAIM_5D": 65,
    }
    origins["maximum_policy_wait_market_days"] = origins["policy_id"].map(maximum_wait).astype(int)
    origins["opportunity_mature"] = (
        origins["market_day_index"]
        + origins["maximum_policy_wait_market_days"]
        + origins["horizon_market_days"]
        <= cutoff_index
    )
    origin_index = pd.to_numeric(origins["forecast_origin_market_day_index"], errors="coerce")
    origins["target_market_day_index"] = origin_index + origins["horizon_market_days"]
    origins["triggered_horizon_mature"] = (
        origins["entry_triggered"]
        & origin_index.notna()
        & origins["target_market_day_index"].le(cutoff_index)
    )
    date_lookup = calendar.set_index("market_day_index")["market_date"]
    origins["target_date"] = pd.to_datetime(origins["target_market_day_index"].map(date_lookup)).dt.normalize()
    requests = origins.loc[
        origins["triggered_horizon_mature"],
        [
            "event_id",
            "policy_id",
            "horizon_market_days",
            "ts_code",
            "forecast_origin_market_day_index",
            "target_market_day_index",
            "forecast_origin_total_return_close",
        ],
    ].copy()
    requests["forecast_origin_market_day_index"] = requests[
        "forecast_origin_market_day_index"
    ].astype(int)
    requests["target_market_day_index"] = requests["target_market_day_index"].astype(int)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='4GB'")
        connection.register("outcome_requests", requests)
        path_results = connection.execute(
            f"""
            WITH path_values AS (
                SELECT
                    r.event_id,
                    r.policy_id,
                    r.horizon_market_days,
                    p.market_day_index - r.forecast_origin_market_day_index AS day_offset,
                    p.total_return_close / r.forecast_origin_total_return_close - 1.0 AS path_return,
                    p.market_day_index,
                    p.total_return_close
                FROM outcome_requests r
                LEFT JOIN read_parquet('{sql_path(project_path(contract['inputs']['unified_daily_market']))}') p
                  ON p.con_code = r.ts_code
                 AND p.market_day_index BETWEEN r.forecast_origin_market_day_index
                                            AND r.target_market_day_index
                 AND p.total_return_close > 0
            ), extrema AS (
                SELECT
                    event_id,
                    policy_id,
                    horizon_market_days,
                    COUNT(path_return) AS observed_close_rows,
                    MIN(path_return) AS close_path_mae,
                    MAX(path_return) AS close_path_mfe,
                    MAX(CASE WHEN day_offset = horizon_market_days THEN path_return END)
                        AS stock_close_return
                FROM path_values
                GROUP BY event_id, policy_id, horizon_market_days
            )
            SELECT
                e.*,
                MIN(v.day_offset) FILTER (WHERE v.path_return = e.close_path_mae)
                    AS time_to_close_path_mae,
                MIN(v.day_offset) FILTER (WHERE v.path_return = e.close_path_mfe)
                    AS time_to_close_path_mfe
            FROM extrema e
            LEFT JOIN path_values v
              ON v.event_id = e.event_id
             AND v.policy_id = e.policy_id
             AND v.horizon_market_days = e.horizon_market_days
            GROUP BY ALL
            """
        ).fetchdf()
    finally:
        connection.close()
    keys = ["event_id", "policy_id", "horizon_market_days"]
    output = origins.merge(path_results, on=keys, how="left", validate="one_to_one")
    output["close_path_complete"] = (
        output["triggered_horizon_mature"]
        & output["observed_close_rows"].eq(output["horizon_market_days"] + 1)
        & output["stock_close_return"].notna()
    )
    output.loc[~output["close_path_complete"], [
        "stock_close_return",
        "close_path_mae",
        "close_path_mfe",
        "time_to_close_path_mae",
        "time_to_close_path_mfe",
    ]] = np.nan

    market_lookup = market_state.set_index("market_day_index")["h00300_close"]
    output["h00300_origin_close"] = pd.to_numeric(
        output["forecast_origin_market_day_index"], errors="coerce"
    ).map(market_lookup)
    output["h00300_target_close"] = pd.to_numeric(
        output["target_market_day_index"], errors="coerce"
    ).map(market_lookup)
    output["h00300_close_return"] = (
        output["h00300_target_close"] / output["h00300_origin_close"] - 1.0
    )
    output["h00300_compound_excess_return"] = compound_excess(
        output["stock_close_return"].to_numpy(dtype=float),
        output["h00300_close_return"].to_numpy(dtype=float),
    )
    output.loc[~output["close_path_complete"], "h00300_compound_excess_return"] = np.nan

    industry_requests = output.loc[
        output["close_path_complete"],
        keys
        + [
            "ts_code",
            "forecast_origin_date",
            "target_date",
            "origin_industry_l1_code",
            "stock_close_return",
        ],
    ].copy()
    industry = compute_point_in_time_industry_peer_return(industry_requests, contract)
    output = output.merge(industry, on=keys, how="left", validate="one_to_one")

    policy_no_entry = output["policy_status"].str.startswith("NO_ENTRY")
    policy_no_view = output["policy_status"].str.startswith("NO_VIEW")
    policy_right_censored = output["policy_status"].str.startswith("RIGHT_CENSORED")
    triggered_complete = output["entry_triggered"] & output["close_path_complete"]
    output["outcome_status"] = np.select(
        [
            ~output["opportunity_mature"],
            policy_no_view,
            policy_right_censored,
            policy_no_entry,
            output["entry_triggered"] & ~output["triggered_horizon_mature"],
            output["entry_triggered"] & output["triggered_horizon_mature"] & ~output["close_path_complete"],
            triggered_complete,
        ],
        [
            "OPPORTUNITY_NOT_MATURE",
            "NO_VIEW_POLICY_PATH",
            "RIGHT_CENSORED_POLICY_PATH",
            "NO_ENTRY_CASH_ZERO",
            "TRIGGERED_HORIZON_RIGHT_CENSORED",
            "TRIGGERED_CLOSE_PATH_INCOMPLETE",
            "TRIGGERED_MATURE_COMPLETE_CLOSE_PATH",
        ],
        default="NO_VIEW_UNCLASSIFIED",
    )
    output["conditional_industry_excess_return"] = np.where(
        triggered_complete,
        output["industry_peer_compound_excess_return"],
        np.nan,
    )
    output["conditional_h00300_excess_return"] = np.where(
        triggered_complete,
        output["h00300_compound_excess_return"],
        np.nan,
    )
    valid_policy_value = output["opportunity_mature"] & ~policy_no_view & ~policy_right_censored
    output["policy_value_industry_excess_return"] = np.where(
        valid_policy_value & policy_no_entry,
        0.0,
        np.where(
            valid_policy_value & triggered_complete,
            output["industry_peer_compound_excess_return"],
            np.nan,
        ),
    )
    output["policy_value_h00300_excess_return"] = np.where(
        valid_policy_value & policy_no_entry,
        0.0,
        np.where(
            valid_policy_value & triggered_complete,
            output["h00300_compound_excess_return"],
            np.nan,
        ),
    )
    output["mae_mfe_tie_rule"] = "EARLIEST_MARKET_DAY_OFFSET"
    output = output.sort_values(
        ["event_date", "ts_code", "policy_id", "horizon_market_days"]
    ).reset_index(drop=True)
    receipt = {
        "rows": int(len(output)),
        "triggered_complete": int(triggered_complete.sum()),
        "industry_excess_observed": int(output["industry_peer_compound_excess_return"].notna().sum()),
        "h00300_excess_observed": int(output["h00300_compound_excess_return"].notna().sum()),
        "outcome_status": output["outcome_status"].value_counts().to_dict(),
    }
    return output, receipt


def expected_shortfall(values: pd.Series, probability: float = 0.10) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(clean.size * probability)))
    return float(np.sort(clean)[:count].mean())


def summarize_return_subset(
    frame: pd.DataFrame,
    *,
    label: str,
    horizon: int,
    membership: str,
) -> dict[str, Any]:
    industry = pd.to_numeric(frame["conditional_industry_excess_return"], errors="coerce")
    market = pd.to_numeric(frame["conditional_h00300_excess_return"], errors="coerce")
    stock = pd.to_numeric(frame["stock_close_return"], errors="coerce")
    return {
        "sample_type": label,
        "membership": membership,
        "horizon_market_days": int(horizon),
        "events": int(len(frame)),
        "event_dates": int(pd.to_datetime(frame["event_date"]).nunique()),
        "industry_excess_observed": int(industry.notna().sum()),
        "mean_industry_excess_event_equal": float(industry.mean()),
        "mean_industry_excess_date_equal": event_date_equal_mean(
            frame.assign(_value=industry), "_value"
        ),
        "median_industry_excess": float(industry.median()),
        "positive_industry_excess_rate": float(industry.gt(0).mean()) if industry.notna().any() else np.nan,
        "industry_excess_es10": expected_shortfall(industry),
        "mean_h00300_excess_event_equal": float(market.mean()),
        "mean_h00300_excess_date_equal": event_date_equal_mean(
            frame.assign(_value=market), "_value"
        ),
        "mean_stock_return": float(stock.mean()),
        "stock_return_es10": expected_shortfall(stock),
        "mean_close_path_mae": float(pd.to_numeric(frame["close_path_mae"], errors="coerce").mean()),
        "mean_close_path_mfe": float(pd.to_numeric(frame["close_path_mfe"], errors="coerce").mean()),
    }


def build_market_state_summaries(
    policy_outcomes: pd.DataFrame,
    market_origin_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成五等分、累计加入、低高波×升降波和政策状态描述。"""

    market_columns = [
        "event_id",
        "policy_id",
        "event_market_vol_quintile",
        "event_market_vol_percentile_756",
        "event_market_log_rv5_div_rv20",
        "event_market_vol_falling",
        "event_market_low_and_falling",
        "event_lowvol_breadth_equal_weight",
        "event_lowvol_breadth_float_weight",
        "origin_market_vol_quintile",
        "origin_market_vol_percentile_756",
        "origin_market_log_rv5_div_rv20",
        "origin_market_vol_falling",
        "origin_market_low_and_falling",
    ]
    merged = policy_outcomes.merge(
        market_origin_ledger[market_columns],
        on=["event_id", "policy_id"],
        how="left",
        validate="many_to_one",
    )
    p0 = merged.loc[
        merged["policy_id"].eq("P0_COMPRESSION_D0_BASELINE")
        & merged["close_path_complete"]
    ].copy()
    quintile_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for horizon in (20, 60):
        horizon_data = p0.loc[p0["horizon_market_days"].eq(horizon)]
        for quintile in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            subset = horizon_data.loc[horizon_data["event_market_vol_quintile"].eq(quintile)]
            quintile_rows.append(
                summarize_return_subset(
                    subset,
                    label="INDEPENDENT_MARKET_VOL_QUINTILE",
                    horizon=horizon,
                    membership=quintile,
                )
            )
            members = cumulative_quintile_membership(quintile)
            cumulative = horizon_data.loc[horizon_data["event_market_vol_quintile"].isin(members)]
            cumulative_rows.append(
                summarize_return_subset(
                    cumulative,
                    label="CUMULATIVE_LOW_TO_HIGH_MARKET_VOL",
                    horizon=horizon,
                    membership="_".join(members),
                )
            )
        level = np.where(
            pd.to_numeric(horizon_data["event_market_vol_percentile_756"], errors="coerce") <= 0.40,
            "LOW_Q1_Q2",
            "NORMAL_HIGH_Q3_Q5",
        )
        direction = np.where(
            horizon_data["event_market_vol_falling"].fillna(False),
            "FALLING_RV5_LTE_RV20",
            "EXPANDING_RV5_GT_RV20",
        )
        horizon_data = horizon_data.assign(_four_state=pd.Series(level, index=horizon_data.index) + "|" + pd.Series(direction, index=horizon_data.index))
        for state, subset in horizon_data.groupby("_four_state", sort=True):
            direction_rows.append(
                summarize_return_subset(
                    subset,
                    label="MARKET_LEVEL_BY_DIRECTION",
                    horizon=horizon,
                    membership=str(state),
                )
            )

    policy_rows: list[dict[str, Any]] = []
    for (policy, horizon), data in merged.groupby(["policy_id", "horizon_market_days"], sort=True):
        event_low_and_falling = data["event_market_low_and_falling"].fillna(False).astype(bool)
        origin_low_and_falling = data["origin_market_low_and_falling"].fillna(False).astype(bool)
        for state_name, mask in {
            "ALL": pd.Series(True, index=data.index),
            "EVENT_LOW_AND_FALLING": event_low_and_falling,
            "EVENT_OTHER_MARKET_STATE": ~event_low_and_falling,
            "ORIGIN_LOW_AND_FALLING": origin_low_and_falling,
            "ORIGIN_OTHER_OR_NO_ORIGIN": ~origin_low_and_falling,
        }.items():
            subset = data.loc[mask]
            conditional = pd.to_numeric(subset["conditional_industry_excess_return"], errors="coerce")
            policy_value = pd.to_numeric(subset["policy_value_industry_excess_return"], errors="coerce")
            policy_rows.append(
                {
                    "policy_id": policy,
                    "horizon_market_days": int(horizon),
                    "market_state": state_name,
                    "episodes": int(len(subset)),
                    "event_dates": int(pd.to_datetime(subset["event_date"]).nunique()),
                    "triggered": int(subset["entry_triggered"].sum()),
                    "trigger_rate": float(subset["entry_triggered"].mean()) if len(subset) else np.nan,
                    "conditional_observed": int(conditional.notna().sum()),
                    "conditional_industry_excess_date_equal": event_date_equal_mean(
                        subset.assign(_value=conditional), "_value"
                    ),
                    "conditional_industry_excess_es10": expected_shortfall(conditional),
                    "policy_value_observed": int(policy_value.notna().sum()),
                    "policy_value_industry_excess_date_equal": event_date_equal_mean(
                        subset.assign(_value=policy_value), "_value"
                    ),
                    "policy_value_industry_excess_es10": expected_shortfall(policy_value),
                    "no_view_rate": float(subset["outcome_status"].str.contains("NO_VIEW").mean())
                    if len(subset)
                    else np.nan,
                }
            )
    return (
        pd.DataFrame(quintile_rows),
        pd.DataFrame(cumulative_rows),
        pd.DataFrame(direction_rows),
        pd.DataFrame(policy_rows),
    )


def moving_block_bootstrap_means(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """对完整交易日序列执行固定长度非循环移动区块抽样。"""

    if values.size < block_length:
        return np.full(repetitions, np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(values.size / block_length))
    output = np.full(repetitions, np.nan, dtype=float)
    batch_size = 50
    offsets = np.arange(block_length, dtype=np.int64)
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        starts = rng.integers(0, values.size - block_length + 1, size=(size, blocks))
        indices = (starts[:, :, None] + offsets).reshape(size, -1)[:, : values.size]
        with np.errstate(invalid="ignore"):
            output[start : start + size] = np.nanmean(values[indices], axis=1)
    return output


def build_primary_hypothesis_results(
    policy_outcomes: pd.DataFrame,
    calendar: pd.DataFrame,
    family: dict[str, Any],
) -> pd.DataFrame:
    """运行预冻结P1/P2/P3×20/60主要检验并统一Holm校正。"""

    all_dates = pd.DatetimeIndex(pd.to_datetime(calendar["market_date"]).dt.normalize())
    rows: list[dict[str, Any]] = []
    for number, hypothesis in enumerate(family["hypotheses"]):
        subset = policy_outcomes.loc[
            policy_outcomes["policy_id"].eq(hypothesis["policy"])
            & policy_outcomes["horizon_market_days"].eq(hypothesis["horizon_market_days"])
        ].copy()
        daily = (
            subset.groupby("event_date")["policy_value_industry_excess_return"]
            .mean()
            .reindex(all_dates)
        )
        values = daily.to_numpy(dtype=float)
        estimate = float(np.nanmean(values))
        samples = moving_block_bootstrap_means(
            values,
            block_length=int(hypothesis["horizon_market_days"]),
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED + number,
        )
        centered_values = values.copy()
        centered_values[np.isfinite(centered_values)] -= estimate
        null_samples = moving_block_bootstrap_means(
            centered_values,
            block_length=int(hypothesis["horizon_market_days"]),
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED + 100 + number,
        )
        p_value = float(
            (1 + np.sum(null_samples[np.isfinite(null_samples)] >= estimate))
            / (1 + np.isfinite(null_samples).sum())
        )
        conditional = pd.to_numeric(subset["conditional_industry_excess_return"], errors="coerce")
        rows.append(
            {
                "hypothesis_id": hypothesis["id"],
                "policy_id": hypothesis["policy"],
                "horizon_market_days": int(hypothesis["horizon_market_days"]),
                "episodes": int(subset["event_id"].nunique()),
                "event_dates_with_policy_value": int(daily.notna().sum()),
                "triggered": int(subset["entry_triggered"].sum()),
                "trigger_rate": float(subset["entry_triggered"].mean()),
                "conditional_industry_excess_date_equal": event_date_equal_mean(
                    subset.assign(_value=conditional), "_value"
                ),
                "policy_value_industry_excess_date_equal": estimate,
                "bootstrap_ci_lower_2_5pct": float(np.nanquantile(samples, 0.025)),
                "bootstrap_ci_upper_97_5pct": float(np.nanquantile(samples, 0.975)),
                "p_value_one_sided_unadjusted": p_value,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "block_length_market_days": int(hypothesis["horizon_market_days"]),
            }
        )
    output = pd.DataFrame(rows)
    output["p_value_holm_6"] = holm_adjust(output["p_value_one_sided_unadjusted"])
    output["holm_5pct_pass"] = output["p_value_holm_6"] <= 0.05
    output["ci_lower_above_zero"] = output["bootstrap_ci_lower_2_5pct"] > 0.0
    output["historical_primary_gate_pass"] = output["holm_5pct_pass"] & output[
        "ci_lower_above_zero"
    ]
    return output


def validate_p0_reconciliation(
    contract: dict[str, Any],
    policy_outcomes: pd.DataFrame,
) -> dict[str, Any]:
    """确认P0严格复现父研究事件日收盘口径的20/60日结果。"""

    parent = pd.read_parquet(
        project_path(contract["inputs"]["parent_horizon_outcomes"]),
        columns=[
            "event_id",
            "horizon_market_days",
            "path_complete",
            "path_return",
            "h00300_compound_excess_return",
            "industry_peer_compound_excess_return",
        ],
    )
    parent = parent.loc[parent["horizon_market_days"].isin([20, 60])].copy()
    child = policy_outcomes.loc[
        policy_outcomes["policy_id"].eq("P0_COMPRESSION_D0_BASELINE"),
        [
            "event_id",
            "horizon_market_days",
            "close_path_complete",
            "stock_close_return",
            "h00300_compound_excess_return",
            "industry_peer_compound_excess_return",
        ],
    ].copy()
    joined = parent.merge(
        child,
        on=["event_id", "horizon_market_days"],
        how="outer",
        suffixes=("_parent", "_child"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("P0与父期限结果主键不一致")
    complete_match = (
        joined["path_complete"].fillna(False).astype(bool)
        == joined["close_path_complete"].fillna(False).astype(bool)
    )
    if not complete_match.all():
        raise RuntimeError(f"P0路径完整标记与父研究不一致：{int((~complete_match).sum())}行")
    complete = joined["path_complete"].fillna(False).astype(bool)
    differences: dict[str, float] = {}
    for parent_column, child_column, label in (
        ("path_return", "stock_close_return", "stock_return"),
        (
            "h00300_compound_excess_return_parent",
            "h00300_compound_excess_return_child",
            "h00300_excess",
        ),
        (
            "industry_peer_compound_excess_return_parent",
            "industry_peer_compound_excess_return_child",
            "industry_excess",
        ),
    ):
        left = pd.to_numeric(joined.loc[complete, parent_column], errors="coerce")
        right = pd.to_numeric(joined.loc[complete, child_column], errors="coerce")
        common = left.notna() & right.notna()
        missing_mismatch = left.notna() != right.notna()
        if missing_mismatch.any():
            raise RuntimeError(f"P0 {label} 缺失模式不一致：{int(missing_mismatch.sum())}行")
        maximum = float((left[common] - right[common]).abs().max()) if common.any() else 0.0
        differences[label] = maximum
        if maximum > 1e-12:
            raise RuntimeError(f"P0 {label} 未在1e-12内复现父研究：{maximum}")
    return {
        "rows": int(len(joined)),
        "complete_rows": int(complete.sum()),
        "parent_endpoint_return_retained_on_incomplete_path_rows": int(
            (joined["path_return"].notna() & ~complete).sum()
        ),
        "maximum_absolute_differences": differences,
        "pass_within_1e_12": True,
    }


def collect_input_records(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """冻结V2实际直接读取的文件路径、字节和SHA-256。"""

    paths: list[Path] = [CONFIG]
    paths.extend(project_path(relative) for relative in contract["contracts"].values())
    for name, relative in contract["inputs"].items():
        path = project_path(relative)
        if path.is_dir():
            paths.extend(sorted(path.glob("symbol_bucket=*/*.parquet")))
            manifest = path / "bucket_manifest.json"
            if manifest.exists():
                paths.append(manifest)
        else:
            paths.append(path)
    unique = sorted(set(path.resolve() for path in paths))
    records = []
    for path in unique:
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _format_percent(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.2%}"


def render_report(
    receipt: dict[str, Any],
    quintiles: pd.DataFrame,
    direction: pd.DataFrame,
    primary: pd.DataFrame,
    policy_state: pd.DataFrame,
) -> str:
    p0_quintiles = quintiles[
        [
            "membership",
            "horizon_market_days",
            "events",
            "event_dates",
            "mean_industry_excess_date_equal",
            "mean_h00300_excess_date_equal",
            "industry_excess_es10",
        ]
    ].copy()
    for column in (
        "mean_industry_excess_date_equal",
        "mean_h00300_excess_date_equal",
        "industry_excess_es10",
    ):
        p0_quintiles[column] = p0_quintiles[column].map(_format_percent)
    direction_table = direction[
        [
            "membership",
            "horizon_market_days",
            "events",
            "mean_industry_excess_date_equal",
            "industry_excess_es10",
        ]
    ].copy()
    for column in ("mean_industry_excess_date_equal", "industry_excess_es10"):
        direction_table[column] = direction_table[column].map(_format_percent)
    primary_table = primary[
        [
            "hypothesis_id",
            "policy_id",
            "horizon_market_days",
            "trigger_rate",
            "conditional_industry_excess_date_equal",
            "policy_value_industry_excess_date_equal",
            "bootstrap_ci_lower_2_5pct",
            "bootstrap_ci_upper_97_5pct",
            "p_value_holm_6",
            "historical_primary_gate_pass",
        ]
    ].copy()
    for column in (
        "trigger_rate",
        "conditional_industry_excess_date_equal",
        "policy_value_industry_excess_date_equal",
        "bootstrap_ci_lower_2_5pct",
        "bootstrap_ci_upper_97_5pct",
    ):
        primary_table[column] = primary_table[column].map(_format_percent)
    focus_policy = policy_state.loc[
        policy_state["policy_id"].isin(
            ["P2_UP_BREAK_PERSISTENCE_2C", "P3_DOWN_BREAK_RECLAIM_5D"]
        )
        & policy_state["market_state"].isin(["EVENT_LOW_AND_FALLING", "EVENT_OTHER_MARKET_STATE"]),
        [
            "policy_id",
            "horizon_market_days",
            "market_state",
            "episodes",
            "trigger_rate",
            "conditional_industry_excess_date_equal",
            "policy_value_industry_excess_date_equal",
            "conditional_industry_excess_es10",
        ],
    ].copy()
    for column in (
        "trigger_rate",
        "conditional_industry_excess_date_equal",
        "policy_value_industry_excess_date_equal",
        "conditional_industry_excess_es10",
    ):
        focus_policy[column] = focus_policy[column].map(_format_percent)
    primary_passes = int(primary["historical_primary_gate_pass"].sum())
    return f"""# 沪深A股极端缩波后超额收益与第一次升波研究 V2

## 研究身份

- 研究ID：`{STUDY_ID}`
- 数据截止：2026-08-14
- 状态：`HISTORICAL_ALPHA_HYPOTHESIS_DISCOVERY_COMPLETED_NO_EXECUTION`
- 父事件：V1.2.1冻结的64,729个episode，事件、10%门槛、U0/L0和60日方向窗均未修改。
- 主要目标：识别个股极端缩波后，市场低波/降波、第一次升波及P1/P2/P3状态是否对应点时行业和沪深300超额。
- 所有收益仍为收盘到收盘的价格研究口径；没有交易成本、真实涨跌停可执行性或实盘授权。

## P0：缩波事件后，市场波动五等分

五等分在唯一市场日期上按点时H00300 RV20历史百分位固定映射；不能从表中选择最佳箱体作为新策略。

{p0_quintiles.to_markdown(index=False)}

## P0：市场波动水平 × 波动方向

`FALLING`表示 `RV5 <= RV20`，`EXPANDING`表示 `RV5 > RV20`。第一次升波是时点，不是方向。

{direction_table.to_markdown(index=False)}

## 预冻结六项政策—期限主要检验

主要估计量为全episode政策价值：未入场按现金零收益，NO_VIEW排除并报告；先按事件日等权，再做20/60日移动区块Bootstrap及Holm六项校正。

{primary_table.to_markdown(index=False)}

- 同时满足Holm 5%与Bootstrap 95%下界大于0：{primary_passes}/6。
- 即使历史门槛通过，也只表示回顾性条件相关，不是未见数据支持的alpha。

## P2/P3在事件日市场低波且仍降波时的描述

{focus_policy.to_markdown(index=False)}

## 第一次升波账本

- 个股第一次升波状态：`{json.dumps(receipt['first_expansion']['stock_transition_status'], ensure_ascii=False)}`
- 个股升波时价格状态：`{json.dumps(receipt['first_expansion']['price_states'], ensure_ascii=False)}`
- 市场/个股升波先后：`{json.dumps(receipt['first_expansion']['lead_lag_states'], ensure_ascii=False)}`

## 对账与数据完整性

- P0与父研究20/60日股票、H00300及点时行业超额在1e-12内逐行复现：{receipt['p0_reconciliation']['pass_within_1e_12']}。
- 股票绝对低波广度主字段为等权；流通市值加权仅在点时股本可得样本上作为敏感性。
- V2直接输入在运行前后SHA-256一致：{receipt['input_integrity']['post_run_match']}。

## 解释边界

1. 市场低波五等分和累计加入曲线只用于剂量—反应、单调性和稀释诊断，不产生五条策略。
2. 第一次升波主要预测未来波动幅度；只有与P2/P3方向状态结合并改善超额及左尾，才可能成为alpha候选。
3. 条件入场均值和全episode政策价值必须同时阅读；罕见高收益状态可能对所有机会的政策价值接近零。
4. 当前未完成可执行成本、涨跌停、退市真实经济终值和forward shadow验证，禁止Shadow、实盘和订单。
5. 嵌套M0—M6的状态账本已形成，但特质缩波M4与条件模型的完整purged walk-forward仍属于下一道验证门槛；不得用本报告挑赢家绕过该门槛。
"""


def main() -> int:
    contract, contracts = load_protocol()
    cutoff = pd.Timestamp(contract["study"]["data_cutoff"]).normalize()
    output_paths = {name: project_path(relative) for name, relative in contract["artifacts"].items()}
    output_paths["root"].mkdir(parents=True, exist_ok=True)

    print("冻结并哈希V2直接输入。", flush=True)
    input_records_before = collect_input_records(contract)
    input_fingerprint = canonical_sha256(input_records_before)
    calendar = load_calendar(project_path(contract["inputs"]["trading_calendar"]), cutoff)
    market_state = build_market_state(contract, calendar)

    print("计算全市场股票低波广度和episode个股升波路径。", flush=True)
    market_state, stock_receipt = build_stock_state_and_breadth(contract, calendar, market_state)
    atomic_parquet(market_state, output_paths["market_volatility_state_daily"])

    print("生成P0/P1/P2/P3预测原点。", flush=True)
    policy_origins, policy_origin_receipt = build_policy_origin_ledger(contract, calendar)
    atomic_parquet(policy_origins, output_paths["policy_origin_ledger"])

    print("生成市场与个股第一次升波账本。", flush=True)
    first_expansion, expansion_receipt = build_first_expansion_ledger(
        contract, calendar, market_state
    )
    atomic_parquet(first_expansion, output_paths["first_vol_expansion_transition_ledger"])

    market_origin = build_event_market_origin_ledger(policy_origins, market_state)
    atomic_parquet(market_origin, output_paths["event_market_volatility_origin_ledger"])

    print("计算政策预测原点后的20/60日超额收益。", flush=True)
    policy_outcomes, outcome_receipt = build_policy_outcomes(
        contract, calendar, market_state, policy_origins
    )
    atomic_parquet(policy_outcomes, output_paths["policy_outcomes"])
    p0_reconciliation = validate_p0_reconciliation(contract, policy_outcomes)

    print("生成市场状态描述和六项预冻结检验。", flush=True)
    quintiles, cumulative, direction, policy_state = build_market_state_summaries(
        policy_outcomes, market_origin
    )
    primary = build_primary_hypothesis_results(
        policy_outcomes, calendar, contracts["primary_family"]
    )
    atomic_csv(quintiles, output_paths["market_state_quintile_summary"])
    atomic_csv(cumulative, output_paths["market_state_cumulative_summary"])
    atomic_csv(direction, output_paths["market_direction_state_summary"])
    atomic_csv(policy_state, output_paths["policy_market_state_summary"])
    atomic_csv(primary, output_paths["primary_hypothesis_results"])

    input_records_after = collect_input_records(contract)
    post_run_match = input_records_after == input_records_before
    if not post_run_match:
        raise RuntimeError("V2运行期间直接输入发生变化")
    artifacts = {}
    for name, path in output_paths.items():
        if name == "root" or not path.exists():
            continue
        artifacts[name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    receipt: dict[str, Any] = {
        "study_id": STUDY_ID,
        "generated_at": datetime.now(TIME_ZONE).isoformat(),
        "data_cutoff": str(cutoff.date()),
        "status": "HISTORICAL_ALPHA_HYPOTHESIS_DISCOVERY_COMPLETED_NO_EXECUTION",
        "parent_event_count": 64729,
        "contracts": {
            name: {
                "path": relative,
                "sha256": sha256_file(project_path(relative)),
            }
            for name, relative in contract["contracts"].items()
        },
        "input_integrity": {
            "direct_input_count": len(input_records_before),
            "aggregate_fingerprint": input_fingerprint,
            "post_run_match": post_run_match,
            "records": input_records_before,
        },
        "stock_state": stock_receipt,
        "policy_origins": policy_origin_receipt,
        "first_expansion": expansion_receipt,
        "policy_outcomes": outcome_receipt,
        "p0_reconciliation": p0_reconciliation,
        "primary_hypotheses": primary.to_dict(orient="records"),
        "artifacts": artifacts,
        "governance": {
            "historical_only": True,
            "predictive_alpha_claim": False,
            "executable_backtest": False,
            "shadow_authorized": False,
            "live_authorized": False,
            "orders_generated": False,
        },
    }
    atomic_json(receipt, output_paths["run_receipt"])
    artifacts["run_receipt"] = {
        "path": str(output_paths["run_receipt"].relative_to(ROOT)).replace("\\", "/"),
        "bytes": output_paths["run_receipt"].stat().st_size,
        "sha256": sha256_file(output_paths["run_receipt"]),
    }
    report = render_report(receipt, quintiles, direction, primary, policy_state)
    atomic_text(report, output_paths["report_markdown"])
    atomic_json(receipt, output_paths["report_json"])
    print(f"V2完成：{output_paths['report_markdown']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
