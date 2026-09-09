"""在组合收益读取前冻结沪深主动上市普通A股信号；不计算组合收益。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import gc
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1_4_signal_freeze_active_universe.yaml"
ENGINE_PATH = ROOT / "src/a_share_hs_concentrated_low_risk_trend_v1.py"
ENGINE_SPEC = importlib.util.spec_from_file_location("a_share_hs_engine", ENGINE_PATH)
assert ENGINE_SPEC and ENGINE_SPEC.loader
ENGINE = importlib.util.module_from_spec(ENGINE_SPEC)
sys.modules[ENGINE_SPEC.name] = ENGINE
ENGINE_SPEC.loader.exec_module(ENGINE)


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board_for_code(ts_code: str) -> str:
    code = str(ts_code).split(".")[0].zfill(6)
    if code.endswith(".SH"):
        return "SSE_STAR" if code.startswith("688") else "SSE_MAIN"
    return "SZSE_CHINEXT" if code.startswith(("300", "301")) else "SZSE_MAIN"


def _load_daily_market(manifest_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("daily_market_datasets", [])
    frames_by_role: dict[str, list[pd.DataFrame]] = {}
    receipts: list[dict[str, Any]] = []
    required = [
        "con_code", "date", "raw_open", "raw_high", "raw_low", "raw_close",
        "total_return_high", "total_return_low", "total_return_close", "volume", "amount", "is_suspended",
    ]
    for item in datasets:
        path = project_path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"日线数据集不存在：{item['path']}")
        frame = pd.read_parquet(path, columns=required)
        frame = frame.rename(columns={"con_code": "ts_code", "volume": "volume_shares", "amount": "amount_cny"})
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["dataset_role"] = item.get("dataset_role", "UNSPECIFIED")
        frames_by_role.setdefault(item.get("dataset_role", "UNSPECIFIED"), []).append(frame)
        receipts.append({"path": item["path"], "rows": len(frame), "sha256": sha256_file(path), "dataset_role": item.get("dataset_role")})
    if not frames_by_role:
        raise ValueError("联合日线清单为空")
    key = ["ts_code", "date"]
    frozen_panel = pd.concat(frames_by_role.get("FROZEN_V1_1", []), ignore_index=True)
    supplement_panel = pd.concat(frames_by_role.get("DISJOINT_COVERAGE_SUPPLEMENT", []), ignore_index=True)
    other_frames = [frame for role, frames in frames_by_role.items() if role not in {"FROZEN_V1_1", "DISJOINT_COVERAGE_SUPPLEMENT"} for frame in frames]
    del frames_by_role
    frozen_index = pd.MultiIndex.from_frame(frozen_panel[key].drop_duplicates())
    supplement_index = pd.MultiIndex.from_frame(supplement_panel[key])
    supplement_overlap_mask = supplement_index.isin(frozen_index)
    overlap_rows = int(supplement_overlap_mask.sum())
    supplement_panel = supplement_panel.loc[~supplement_overlap_mask].copy()
    for receipt in receipts:
        if receipt["dataset_role"] == "DISJOINT_COVERAGE_SUPPLEMENT":
            receipt["overlap_rows_discarded_in_favor_of_frozen_v1_1"] = overlap_rows
    panel_parts = [frozen_panel, supplement_panel, *other_frames]
    panel = pd.concat(panel_parts, ignore_index=True)
    del frozen_panel, supplement_panel, other_frames, frozen_index, supplement_index, supplement_overlap_mask, panel_parts
    duplicate_mask = panel.duplicated(key, keep=False)
    if duplicate_mask.any():
        value_columns = [
            "raw_open", "raw_high", "raw_low", "raw_close", "total_return_high",
            "total_return_low", "total_return_close", "volume_shares", "amount_cny", "is_suspended",
        ]
        duplicate_values = panel.loc[duplicate_mask, key + value_columns]
        conflict = duplicate_values.groupby(key, sort=False, dropna=False)[value_columns].nunique(dropna=False).gt(1).any(axis=1)
        if conflict.any():
            examples = conflict[conflict].head(5).index.tolist()
            raise ValueError(f"重叠日线数据存在数值冲突：{examples}")
        panel = panel.drop_duplicates(key, keep="first")
    return panel.sort_values(["ts_code", "date"]).reset_index(drop=True), receipts


def _asof_join(panel: pd.DataFrame, events: pd.DataFrame, *, left_date: str, right_date: str, columns: list[str]) -> pd.DataFrame:
    left = panel.sort_values([left_date, "ts_code"]).copy()
    right = events.sort_values([right_date, "ts_code"])[["ts_code", right_date, *columns]].copy()
    result = pd.merge_asof(
        left,
        right,
        left_on=left_date,
        right_on=right_date,
        by="ts_code",
        direction="backward",
        allow_exact_matches=True,
    )
    return result.sort_values(["ts_code", left_date]).reset_index(drop=True)


def _fast_rolling_max_drawdown(values: np.ndarray, window: int = 120) -> np.ndarray:
    """逐窗口计算与冻结内核完全相同的最大回撤，避免逐行 rolling.apply。"""

    output = np.full(values.shape[0], np.nan, dtype=float)
    if values.shape[0] < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    if valid.any():
        valid_windows = windows[valid]
        peaks = np.maximum.accumulate(valid_windows, axis=1)
        output[window - 1:][valid] = np.abs(np.min(valid_windows / peaks - 1.0, axis=1))
    return output


def _fast_ts_vol_percentile(values: np.ndarray, *, lookback: int = 504, min_history: int = 252) -> np.ndarray:
    """计算严格排除当期的历史中位秩百分位；窗口规则与冻结内核一致。"""

    output = np.full(values.shape[0], np.nan, dtype=float)
    finite = np.isfinite(values)
    short_end = min(values.shape[0], lookback)
    for index in range(min_history, short_end):
        current = values[index]
        if not np.isfinite(current):
            continue
        history = values[:index]
        history = history[np.isfinite(history)]
        if history.size >= min_history:
            output[index] = ((history < current).sum() + 0.5 * (history == current).sum()) / history.size
    if values.shape[0] <= lookback:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values[:-1], window_shape=lookback)
    current = values[lookback:]
    valid_count = finite[:-1].astype(np.int16)
    valid_windows = np.lib.stride_tricks.sliding_window_view(valid_count, window_shape=lookback).sum(axis=1)
    valid_current = np.isfinite(current) & (valid_windows >= min_history)
    if valid_current.any():
        history_windows = windows[valid_current]
        current_values = current[valid_current]
        less = np.sum(history_windows < current_values[:, None], axis=1)
        equal = np.sum(history_windows == current_values[:, None], axis=1)
        output[lookback:][valid_current] = (less + 0.5 * equal) / valid_windows[valid_current]
    return output


def _symbol_features_fast(frame: pd.DataFrame) -> pd.DataFrame:
    """冻结内核 _symbol_features 的等价向量化实现。"""

    result = frame.sort_values("date").copy()
    close = result["total_return_close"].astype(float)
    high = result["total_return_high"].astype(float)
    low = result["total_return_low"].astype(float)
    log_return = np.log(close / close.shift(1))
    result["log_return"] = log_return
    result["rv20"] = log_return.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    result["rv60"] = log_return.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(252)
    result["rv120"] = log_return.rolling(120, min_periods=115).std(ddof=1) * np.sqrt(252)
    negative_squared = np.minimum(log_return.to_numpy(dtype=float), 0.0) ** 2
    result["downside_vol60"] = (
        pd.Series(negative_squared, index=result.index).rolling(60, min_periods=60).mean().pow(0.5) * np.sqrt(252)
    )
    result["max_drawdown120"] = _fast_rolling_max_drawdown(close.to_numpy(dtype=float), 120)
    range_term = np.log(high / low) ** 2
    result["parkinson_range_vol60"] = (
        range_term.rolling(60, min_periods=60).mean() * (252 / (4 * np.log(2)))
    ).pow(0.5)
    result["ma120"] = close.rolling(120, min_periods=120).mean()
    result["momentum_60_5"] = close.shift(5) / close.shift(60) - 1.0
    result["rv20_div_rv120"] = result["rv20"] / result["rv120"]
    result["vol_ratio"] = np.log(result["rv20_div_rv120"])
    result["ts_vol_percentile"] = _fast_ts_vol_percentile(
        result["vol_ratio"].to_numpy(dtype=float), lookback=504, min_history=252
    )
    result["total_mcap"] = result["raw_close"].astype(float) * result["total_a_shares"].astype(float)
    result["float_mcap"] = result["raw_close"].astype(float) * result["tradable_a_shares"].astype(float)
    result["total_mcap_20d_median"] = result["total_mcap"].rolling(20, min_periods=20).median()
    result["float_mcap_20d_median"] = result["float_mcap"].rolling(20, min_periods=20).median()
    result["amount_20d_median"] = result["amount_cny"].astype(float).rolling(20, min_periods=20).median()
    suspended = result["is_suspended"].fillna(True).astype(bool)
    zero_volume = result["volume_shares"].fillna(0).astype(float).le(0)
    result["suspension_days_20"] = suspended.astype(int).rolling(20, min_periods=20).sum()
    result["zero_volume_days_20"] = zero_volume.astype(int).rolling(20, min_periods=20).sum()
    result["valid_returns_120"] = log_return.notna().astype(int).rolling(120, min_periods=120).sum()
    return result


def calculate_panel_features_fast(panel: pd.DataFrame) -> pd.DataFrame:
    """按冻结内核公式计算特征，但用向量化窗口替代逐行慢路径。"""

    required = {
        "ts_code", "date", "raw_close", "total_return_high", "total_return_low", "total_return_close",
        "volume_shares", "amount_cny", "is_suspended", "total_a_shares", "tradable_a_shares",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"行情面板缺少字段：{', '.join(missing)}")
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if work.duplicated(["ts_code", "date"]).any():
        raise ValueError("行情面板存在重复的证券-交易日")
    groups = [_symbol_features_fast(group) for _, group in work.groupby("ts_code", sort=False)]
    # 输入已由联合日线加载器和点时连接保持 ts_code、date 顺序；避免再做一次全表排序复制。
    return pd.concat(groups, ignore_index=True).reset_index(drop=True)


def _normalized_rank_by_group(values: pd.Series, groups: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """返回平均并列秩归一值和每组有效样本数。"""

    ranks = values.groupby(groups, sort=False, dropna=False).rank(method="average", ascending=True)
    counts = values.groupby(groups, sort=False, dropna=False).transform("count")
    normalized = (ranks - 1.0) / (counts - 1.0)
    normalized = normalized.mask(counts.eq(1), 0.0)
    return normalized, counts


def calculate_cross_sectional_scores_fast(panel: pd.DataFrame) -> pd.DataFrame:
    """冻结内核截面评分的等价向量化实现。"""

    required = {"date", "ts_code", "industry_l1", "base_eligible", "float_mcap_20d_median", *ENGINE.RISK_METRICS}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"截面面板缺少字段：{', '.join(missing)}")
    output = panel
    eligible_mask = output["base_eligible"].fillna(False).astype(bool)
    eligible_columns = ["date", "industry_l1", "float_mcap_20d_median", *ENGINE.RISK_METRICS]
    eligible = output.loc[eligible_mask, eligible_columns].copy()
    row_count = len(output)
    eligible_positions = np.flatnonzero(eligible_mask.to_numpy())
    float_mcap_percentile = np.full(row_count, np.nan, dtype=float)
    risk_sum = np.zeros(row_count, dtype=float)
    worst_component = np.full(row_count, -np.inf, dtype=float)
    risk_valid = np.zeros(row_count, dtype=bool)
    risk_valid[eligible_positions] = True
    fallback = np.zeros(row_count, dtype=bool)
    if eligible.empty:
        output["float_mcap_percentile"] = float_mcap_percentile
        output["risk_score"] = np.nan
        output["worst_component_rank"] = np.nan
        output["industry_rank_fallback_global"] = fallback
        return output

    global_mcap, _ = _normalized_rank_by_group(eligible["float_mcap_20d_median"], [eligible["date"]])
    float_mcap_percentile[eligible_positions] = global_mcap.to_numpy(dtype=float)
    for metric in ENGINE.RISK_METRICS:
        global_rank, _ = _normalized_rank_by_group(eligible[metric], [eligible["date"]])
        industry_rank, industry_count = _normalized_rank_by_group(
            eligible[metric], [eligible["date"], eligible["industry_l1"]]
        )
        use_industry = eligible["industry_l1"].notna() & industry_count.ge(10)
        chosen = global_rank.where(~use_industry, industry_rank)
        chosen_values = chosen.to_numpy(dtype=float)
        finite = np.isfinite(chosen_values)
        risk_sum[eligible_positions] += np.where(finite, chosen_values, 0.0)
        worst_component[eligible_positions] = np.maximum(
            worst_component[eligible_positions], np.where(finite, chosen_values, -np.inf)
        )
        risk_valid[eligible_positions] &= finite
        fallback[eligible_positions[~use_industry.to_numpy()]] = True
        del global_rank, industry_rank, industry_count, chosen, chosen_values, finite
    risk_score = np.full(row_count, np.nan, dtype=float)
    risk_score[risk_valid] = risk_sum[risk_valid] / len(ENGINE.RISK_METRICS)
    worst_component[~risk_valid] = np.nan
    output["float_mcap_percentile"] = float_mcap_percentile
    output["risk_score"] = risk_score
    output["worst_component_rank"] = worst_component
    output["industry_rank_fallback_global"] = fallback
    output.drop(columns=list(ENGINE.RISK_METRICS), inplace=True)
    del eligible, eligible_mask, eligible_positions, risk_sum, risk_valid, global_mcap
    gc.collect()
    return output


def attach_stable_risk_scores_fast(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """用唯一整数键连接 t-20/t-40 截面分数，避免两次整表 merge。"""

    result = panel
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    calendar_index = pd.Series(np.arange(len(dates), dtype=np.int32), index=dates)
    day_index = result["date"].map(calendar_index)
    if day_index.isna().any():
        raise ValueError("面板日期不完全属于传入的市场日历")
    day_values = day_index.to_numpy(dtype=np.int64)
    code_values, _ = pd.factorize(result["ts_code"], sort=True)
    key_base = len(dates) + 1
    keys = code_values.astype(np.int64) * key_base + day_values
    key_index = pd.Index(keys)
    if key_index.has_duplicates:
        raise ValueError("稳定风险评分连接键存在重复")
    risk_values = result["risk_score"].to_numpy(dtype=float)
    for lag in (20, 40):
        lagged = np.full(len(result), np.nan, dtype=float)
        valid = day_values >= lag
        locations = key_index.get_indexer(keys[valid] - lag)
        found = locations >= 0
        valid_positions = np.flatnonzero(valid)
        lagged[valid_positions[found]] = risk_values[locations[found]]
        result[f"risk_score_t_minus_{lag}"] = lagged
    result["stable_risk_score"] = (
        0.50 * result["risk_score"]
        + 0.30 * result["risk_score_t_minus_20"]
        + 0.20 * result["risk_score_t_minus_40"]
    )
    del day_index, day_values, code_values, keys, key_index, risk_values
    gc.collect()
    return result


def mark_entry_and_exit_conditions_fast(panel: pd.DataFrame) -> pd.DataFrame:
    """原位冻结进入与退出条件，定义与冻结内核一致。"""

    result = panel
    close = result["total_return_close"].astype(float)
    result["entry_eligible"] = (
        result["base_eligible"].fillna(False).astype(bool)
        & result["total_mcap_20d_median"].ge(10_000_000_000)
        & result["float_mcap_20d_median"].ge(5_000_000_000)
        & result["float_mcap_percentile"].ge(0.30)
        & result["amount_20d_median"].ge(100_000_000)
        & result["suspension_days_20"].eq(0)
        & result["zero_volume_days_20"].eq(0)
        & result["valid_returns_120"].ge(115)
        & result["risk_score"].le(0.12)
        & result["risk_score_t_minus_20"].le(0.20)
        & result["risk_score_t_minus_40"].le(0.25)
        & result["stable_risk_score"].le(0.18)
        & result["worst_component_rank"].le(0.35)
        & result["ts_vol_percentile"].le(0.10)
        & close.ge(result["ma120"])
        & result["momentum_60_5"].gt(0)
        & result["signal_date_tradable"].fillna(False)
    )
    confirmed_downtrend = close.lt(result["ma120"]) & result["momentum_60_5"].lt(0)
    result["confirmed_downtrend"] = confirmed_downtrend
    result["normal_exit"] = (
        result["risk_score"].gt(0.35)
        | result["ts_vol_percentile"].ge(0.70)
        | confirmed_downtrend
    )
    result["emergency_exit"] = (
        result["ts_vol_percentile"].ge(0.90)
        | result["rv20_div_rv120"].ge(1.60)
    )
    result["forced_exit"] = False
    return result


def build_signal_panel(contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[dict[str, Any]]]:
    sources = contract["sources"]
    audit = json.loads(project_path(sources["combined_data_audit_report"]).read_text(encoding="utf-8"))
    if audit.get("status") != "READY_FOR_SIGNAL_CONSTRUCTION":
        raise RuntimeError(f"联合数据审计未通过：{audit.get('status')}")
    panel, market_receipts = _load_daily_market(project_path(sources["daily_market_manifest"]))
    master = pd.read_parquet(project_path(sources["security_master"]))
    master = master.loc[master["security_type"].eq("ORDINARY_A_SHARE")].copy()
    master["ts_code"] = master["ts_code"].astype(str)
    panel = panel.loc[panel["ts_code"].isin(set(master["ts_code"]))].copy()
    panel = panel.merge(
        master[["ts_code", "exchange", "list_date", "delist_date", "security_type"]],
        on="ts_code", how="left", validate="many_to_one",
    )
    shares = pd.read_parquet(project_path(sources["share_counts"]))
    shares["ts_code"] = shares["ts_code"].astype(str)
    shares["effective_date"] = pd.to_datetime(shares["effective_date"], errors="raise").dt.normalize()
    panel = _asof_join(panel, shares, left_date="date", right_date="effective_date", columns=["total_a_shares", "tradable_a_shares", "available_at"])
    panel = panel.rename(columns={"available_at": "share_available_at"})
    industry = pd.read_parquet(project_path(sources["industry_intervals"]))
    industry["ts_code"] = industry["ts_code"].astype(str)
    industry["valid_from"] = pd.to_datetime(industry["valid_from"], errors="raise").dt.normalize()
    panel = _asof_join(panel, industry, left_date="date", right_date="valid_from", columns=["industry_l1", "industry_l1_code", "valid_to", "available_at"])
    panel = panel.rename(columns={"available_at": "industry_available_at"})
    status = pd.read_parquet(project_path(sources["status_intervals"]))
    status["ts_code"] = status["ts_code"].astype(str)
    status["valid_from"] = pd.to_datetime(status["valid_from"], errors="raise").dt.normalize()
    panel = _asof_join(panel, status, left_date="date", right_date="valid_from", columns=["status", "valid_to", "available_at"])
    panel = panel.rename(columns={"available_at": "status_available_at"})

    calendar_frame = pd.read_parquet(project_path(sources["trading_calendar"]))
    calendar_dates = pd.DatetimeIndex(
        pd.to_datetime(calendar_frame.loc[calendar_frame["is_open"].astype(bool), "date"], errors="raise").dt.normalize().unique()
    ).sort_values()
    calendar_index = pd.Series(np.arange(len(calendar_dates), dtype=int), index=calendar_dates)
    panel["calendar_index"] = panel["date"].map(calendar_index)
    panel["list_date"] = pd.to_datetime(panel["list_date"], errors="coerce").dt.normalize()
    panel["delist_date"] = pd.to_datetime(panel["delist_date"], errors="coerce").dt.normalize()
    first_list_index = panel["list_date"].map(lambda value: calendar_dates.searchsorted(value) if pd.notna(value) else np.nan)
    panel["listed_trading_days"] = panel["calendar_index"] - first_list_index + 1
    panel["signal_date_tradable"] = (
        panel["raw_open"].notna()
        & panel["raw_close"].notna()
        & ~panel["is_suspended"].fillna(True).astype(bool)
        & panel["volume_shares"].fillna(0).gt(0)
    )
    panel["base_eligible"] = (
        panel["security_type"].eq("ORDINARY_A_SHARE")
        & panel["exchange"].isin(["SSE", "SZSE"])
        & panel["delist_date"].isna()
        & panel["status"].eq("NORMAL")
        & panel["listed_trading_days"].ge(int(contract["universe"]["minimum_listed_trading_days"]))
        & panel["industry_l1"].notna()
        & panel["total_a_shares"].gt(0)
        & panel["tradable_a_shares"].gt(0)
        & panel["share_available_at"].le(panel["date"] + pd.Timedelta(hours=15))
        & panel["industry_available_at"].le(panel["date"] + pd.Timedelta(hours=15))
        & panel["status_available_at"].le(panel["date"] + pd.Timedelta(hours=15))
    )
    feature_input = panel[[
        "ts_code", "date", "raw_open", "raw_high", "raw_low", "raw_close",
        "total_return_high", "total_return_low", "total_return_close", "volume_shares", "amount_cny",
        "is_suspended", "total_a_shares", "tradable_a_shares", "industry_l1", "base_eligible", "signal_date_tradable",
    ]].copy()
    features = calculate_panel_features_fast(feature_input)
    required_after_features = {
        "ts_code", "date", "total_return_close", "log_return", "rv60", "rv120", "downside_vol60",
        "max_drawdown120", "parkinson_range_vol60", "ma120", "momentum_60_5", "rv20_div_rv120",
        "ts_vol_percentile", "total_mcap_20d_median", "float_mcap_20d_median", "amount_20d_median",
        "suspension_days_20", "zero_volume_days_20", "valid_returns_120", "industry_l1", "base_eligible",
        "signal_date_tradable",
    }
    features.drop(columns=[column for column in features.columns if column not in required_after_features], inplace=True)
    del panel, feature_input
    gc.collect()
    features = calculate_cross_sectional_scores_fast(features)
    features = attach_stable_risk_scores_fast(features, calendar_dates)
    features = mark_entry_and_exit_conditions_fast(features)
    return features, calendar_dates, market_receipts


def freeze_signals(contract: dict[str, Any], features: pd.DataFrame, calendar_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_dates = calendar_dates[calendar_dates.isin(features.loc[features["entry_eligible"].astype(bool), "date"].unique())]
    if len(eligible_dates) == 0:
        raise RuntimeError("没有任何满足冻结入场条件的信号日")
    frequency = int(contract["selection"]["frequency_trading_days"])
    signal_dates = eligible_dates[::frequency]
    return_panel = features.pivot_table(index="date", columns="ts_code", values="log_return", aggfunc="last").sort_index()
    features.drop(columns=["log_return"], inplace=True)
    events: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        snapshot = features.loc[features["date"].eq(signal_date)].copy()
        return_history = return_panel.loc[:signal_date].tail(int(contract["selection"]["correlation_window_days"]))
        selected20 = ENGINE.select_low_risk_names(
            snapshot, return_history,
            maximum_names=int(contract["selection"]["maximum_names"]),
            max_names_per_industry=int(contract["selection"]["max_names_per_industry"]),
            max_pairwise_correlation=float(contract["selection"]["max_pairwise_correlation"]),
            min_pair_observations=int(contract["selection"]["correlation_min_valid_observations"]),
        )
        candidate_count = int(snapshot["entry_eligible"].fillna(False).sum())
        top3 = selected20[:3]
        for rank, code in enumerate(selected20, start=1):
            row = snapshot.loc[snapshot["ts_code"].eq(code)].sort_values("ts_code").iloc[0]
            selected_rows.append({
                "signal_date": signal_date, "ts_code": code, "rank": rank,
                "stable_risk_score": float(row["stable_risk_score"]),
                "ts_vol_percentile": float(row["ts_vol_percentile"]),
                "momentum_60_5": float(row["momentum_60_5"]),
                "industry_l1": str(row["industry_l1"]), "candidate_count": candidate_count,
            })
        events.append({
            "signal_date": signal_date, "candidate_count": candidate_count,
            "top1": json.dumps(selected20[:1], ensure_ascii=False),
            "top3": json.dumps(top3, ensure_ascii=False),
            "top5": json.dumps(selected20[:5], ensure_ascii=False),
            "top20": json.dumps(selected20[:20], ensure_ascii=False),
            "top3_count": len(top3), "cash_weight": 1.0 - sum(ENGINE.target_weights(top3, variant=3).values()),
        })
    return pd.DataFrame(events), pd.DataFrame(selected_rows)


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    features, calendar_dates, market_receipts = build_signal_panel(contract)
    events, selected = freeze_signals(contract, features, calendar_dates)
    root = project_path(contract["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    events_path = project_path(contract["outputs"]["signal_events"])
    selected_path = project_path(contract["outputs"]["selected_symbols"])
    events.to_parquet(events_path, index=False)
    selected.to_parquet(selected_path, index=False)
    receipt = {
        "contract_id": contract["contract_id"], "model_id": contract["model_id"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "SIGNAL_FROZEN_BEFORE_RETURN_READ", "view_status": "NO_VIEW",
        "portfolio_return_values_read": False, "positions_generated": False, "orders_generated": False,
        "signal_dates": len(events), "selected_symbol_count": int(selected["ts_code"].nunique()) if not selected.empty else 0,
        "candidate_count_min": int(events["candidate_count"].min()), "candidate_count_max": int(events["candidate_count"].max()),
        "market_inputs": market_receipts,
        "artifacts": {
            "signal_events": {"path": events_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(events_path)},
            "selected_symbols": {"path": selected_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(selected_path)},
        },
    }
    receipt_path = project_path(contract["outputs"]["receipt"])
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "A_SHARE_HS_SIGNAL_FREEZE_V1",
        "contract_id": contract["contract_id"], "model_id": contract["model_id"],
        "audited_at": receipt["frozen_at"], "status": receipt["status"], "view_status": "NO_VIEW",
        "portfolio_return_values_read": False, "signals_generated": True, "positions_generated": False, "orders_generated": False,
        "signal_dates": len(events), "selected_symbol_count": receipt["selected_symbol_count"],
        "candidate_count_min": receipt["candidate_count_min"], "candidate_count_max": receipt["candidate_count_max"],
        "artifacts": receipt["artifacts"],
        "conclusion": "主动上市普通沪深A股信号已按冻结规则生成；下一步只允许收集实际入选证券公司行动，然后才可读取组合收益。",
    }
    report_path = project_path(contract["outputs"]["report_json"])
    report_md = project_path(contract["outputs"]["report_markdown"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        "# 沪深A股主动上市普通股信号冻结\n\n"
        f"- 状态：`{report['status']}`\n- 研究视图：`NO_VIEW`\n- 组合收益读取：否\n"
        f"- 信号日：{report['signal_dates']}\n- 入选证券全集：{report['selected_symbol_count']}只\n\n"
        "信号使用总收益价格构造风险与趋势特征，但未读取组合收益、未生成仓位和订单。\n",
        encoding="utf-8",
    )
    print(json.dumps({"状态": report["status"], "信号日": report["signal_dates"], "入选证券": report["selected_symbol_count"], "组合收益读取": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
