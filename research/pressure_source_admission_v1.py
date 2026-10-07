"""510300来源资格诊断；条件时钟和本地字段覆盖均不替代原协议准入。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.quote_state_envelope_v1 import clock_ms


def session_labels(times: np.ndarray) -> np.ndarray:
    """沿用原研究的14:57截止；这不是重新解释历史交易制度。"""
    t = np.asarray(times, dtype=np.int64)
    return np.select(
        [t < 93000000, t < 113000000, t < 130000000,
         t < 145700000, t < 150500000, t < 153000000],
        ["PREOPEN", "AM", "MIDDAY", "PM", "CLOSE_WINDOW", "AFTER_HOURS"],
        default="POST_1530",
    )


def make_grid(measurement: dict) -> pd.DataFrame:
    rows = []
    for start, end, name in measurement["sessions_minutes"]:
        for minute in range(start, end):
            rows.append((minute, name, "M2_CONTINUOUS"))
    for minute in measurement["closing_probe_minutes"]:
        rows.append((minute, "CLOSE_WINDOW", "CLOSE_PROBE"))
    start, end = measurement["after_hours_probe_minutes"]
    for minute in range(start, end + 1):
        rows.append((minute, "AFTER_HOURS" if minute < 930 else "POST_1530", "AFTER_HOURS_PROBE"))
    grid = pd.DataFrame(rows, columns=["minute", "session", "grid_kind"])
    grid["hhmm"] = grid.minute.map(lambda m: f"{m // 60:02d}:{m % 60:02d}")
    grid["decision_ms"] = grid.minute * 60000
    return grid


def conditional_prefix_windows(quote_times, quote_cumulative, trade_times, trade_volumes,
                               quote_precision_ms: int = 1000, tick_precision_ms: int = 10) -> pd.DataFrame:
    """累计量前缀与旧截断时间假设的相容性，绝非已认证的快照生成时间。

    同一逐笔时间内不推断消息次序；零累计量只有零前缀的含义。
    不匹配量、负量、非整数份数、网格不符都保留失败，不能移时间补救。
    """
    qt = np.asarray(quote_times, dtype=np.int64)
    qc = np.asarray(quote_cumulative, dtype=np.float64)
    tt = np.asarray(trade_times, dtype=np.int64)
    tv = np.asarray(trade_volumes, dtype=np.float64)
    if len(qt) != len(qc) or len(tt) != len(tv):
        raise ValueError("时间与数量数组长度不一致。")
    nominal = clock_ms(qt)
    clock_ms(tt)
    quantities_valid = (np.all(np.isfinite(tv)) and np.all(tv > 0)
                        and np.all(tv == np.floor(tv)) and float(tv.sum()) < 2**53)
    sorted_indices = np.argsort(tt, kind="stable")
    tt, tv = tt[sorted_indices], tv[sorted_indices]
    ticks_ms = clock_ms(tt)
    cumulative = np.cumsum(tv, dtype=np.float64)
    valid_cum = np.isfinite(qc) & (qc >= 0) & (qc == np.floor(qc))
    safe_cum = np.where(valid_cum, qc, -1)
    positions = np.searchsorted(cumulative, safe_cum, side="left")
    exact = valid_cum & (qc == 0)
    last = np.full(len(qt), -1, dtype=np.int64)
    next_ceiling = np.full(len(qt), 86400000, dtype=np.int64)
    if len(tt):
        safe = np.minimum(positions, len(tt) - 1)
        positive_exact = valid_cum & (qc > 0) & (positions < len(tt)) & (cumulative[safe] == qc)
        exact |= positive_exact
        last[positive_exact] = ticks_ms[safe[positive_exact]]
        next_indices = np.where(qc == 0, 0, positions + 1)
        has_next = exact & (next_indices < len(tt))
        next_ceiling[has_next] = ticks_ms[next_indices[has_next]] + tick_precision_ms
    exact &= quantities_valid
    lower = np.maximum(nominal, last)
    upper = np.minimum(nominal + quote_precision_ms, next_ceiling)
    quote_grid = qt % quote_precision_ms == 0
    tick_grid = bool(np.all(tt % tick_precision_ms == 0))
    compatible = exact & quote_grid & tick_grid & (lower < upper)
    return pd.DataFrame({
        "quote_nominal_ms": nominal, "exact_cumulative_prefix": exact,
        "last_included_trade_ms": np.where(exact, last, -1),
        "conditional_lower_ms": lower, "conditional_upper_exclusive_ms": upper,
        "conditional_width_ms": upper - lower, "conditional_clock_compatible": compatible,
        "quote_second_grid": quote_grid, "trade_centisecond_grid": tick_grid,
        "last_trade_after_nominal": exact & (last > nominal),
        "clock_mapping_verified": False, "receipt_time_verified": False,
    })


def observed_book_features(quotes: pd.DataFrame, band_bps: float = 10) -> pd.DataFrame:
    bid = quotes[[f"bid_px{i}" for i in range(1, 11)]].to_numpy(dtype=float)
    ask = quotes[[f"ask_px{i}" for i in range(1, 11)]].to_numpy(dtype=float)
    bv = quotes[[f"bid_vol{i}" for i in range(1, 11)]].to_numpy(dtype=float)
    av = quotes[[f"ask_vol{i}" for i in range(1, 11)]].to_numpy(dtype=float)
    best = np.isfinite(bid[:, 0]) & np.isfinite(ask[:, 0]) & (bid[:, 0] > 0) & (ask[:, 0] > bid[:, 0])
    valid = (np.all(np.isfinite(bid) & (bid > 0), axis=1)
             & np.all(np.isfinite(ask) & (ask > 0), axis=1)
             & np.all(np.isfinite(bv) & (bv >= 0), axis=1)
             & np.all(np.isfinite(av) & (av >= 0), axis=1)
             & np.all(np.diff(bid, axis=1) < 0, axis=1)
             & np.all(np.diff(ask, axis=1) > 0, axis=1) & best)
    mid_raw = (bid[:, 0] + ask[:, 0]) / 2
    lower, upper = mid_raw * (1 - band_bps / 10000), mid_raw * (1 + band_bps / 10000)
    visible = valid & (bid[:, -1] <= lower) & (ask[:, -1] >= upper)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = 10000 * (ask[:, 0] - bid[:, 0]) / mid_raw
    depth = np.sum(np.where(bid >= lower[:, None], bid * bv, 0), axis=1) / 10000
    return pd.DataFrame({"best_two_sided_valid": best, "ten_level_book_valid": valid,
                         "fixed_10bp_band_visible": visible,
                         "source_mid_cny": np.where(best, mid_raw / 10000, np.nan),
                         "source_spread_bps": np.where(best, spread, np.nan),
                         "source_band_bid_depth_cny": np.where(valid, depth, np.nan)})


def observe_grid(quotes: pd.DataFrame, witnesses: pd.DataFrame, grid: pd.DataFrame,
                 maximum_age_ms: int = 5000, band_bps: float = 10) -> pd.DataFrame:
    """仅选名义时间不晚于端点的源行；不能将其提升为端点前已收到。"""
    quotes = quotes.reset_index(drop=True)
    times = clock_ms(quotes.time)
    order = np.argsort(times, kind="stable")
    times = times[order]
    positions = np.searchsorted(times, grid.decision_ms.to_numpy(), side="right") - 1
    available = positions >= 0
    indices = np.where(available, order[np.maximum(positions, 0)], -1)
    selected = quotes.reindex(indices).reset_index(drop=True)
    book = observed_book_features(selected, band_bps)
    witness = witnesses.reindex(indices).reset_index(drop=True)
    out = grid.reset_index(drop=True).copy()
    out["source_quote_row"] = indices
    out["source_time"] = selected.time.to_numpy()
    out["source_price_raw"] = selected.price.to_numpy()
    out["source_iopv_raw"] = selected.iopv.to_numpy()
    out["source_cum_volume"] = selected.cum_volume.to_numpy()
    out["quote_age_nominal_ms"] = out.decision_ms - witness.quote_nominal_ms
    same_session = session_labels(np.nan_to_num(selected.time.to_numpy(), nan=0).astype(np.int64)) == out.session
    out["source_same_session"] = available & same_session
    out["source_nominal_fresh"] = available & same_session & out.quote_age_nominal_ms.between(0, maximum_age_ms)
    out = pd.concat([out, book, witness], axis=1)
    # 缺行的布尔值必须恢复为False，不能由NaN的真值性产生资格。
    for name in witnesses.columns:
        if witnesses[name].dtype == bool:
            out[name] = out[name].fillna(False).astype(bool)
    out["source_nominal_book_observed"] = out.source_nominal_fresh & out.ten_level_book_valid
    out["source_nominal_band_observed"] = out.source_nominal_book_observed & out.fixed_10bp_band_visible
    out["conditional_range_before_endpoint"] = (out.conditional_clock_compatible
                                                  & out.conditional_upper_exclusive_ms.le(out.decision_ms))
    out["source_conditional_book_observed"] = out.source_nominal_book_observed & out.conditional_range_before_endpoint
    out["source_conditional_band_observed"] = out.source_nominal_band_observed & out.conditional_range_before_endpoint
    out["source_positive_iopv"] = out.source_iopv_raw.gt(0)
    out["source_duplicate_time"] = selected.time.isin(quotes.time[quotes.time.duplicated(keep=False)])
    out["m1_probe"] = out.hhmm.isin(["14:51", "14:56", "15:00", "15:06"])
    reasons = []
    for row in out.itertuples():
        local = []
        if row.source_quote_row < 0:
            local.append("NO_PRIOR_SOURCE_QUOTE")
        elif not row.source_same_session:
            local.append("SOURCE_OUTSIDE_FROZEN_SESSION")
        elif not row.source_nominal_fresh:
            local.append("SOURCE_NOMINAL_AGE_GT_5S")
        if not row.ten_level_book_valid:
            local.append("INVALID_OR_INCOMPLETE_TEN_LEVEL_BOOK")
        elif not row.fixed_10bp_band_visible:
            local.append("TEN_LEVELS_DO_NOT_COVER_FIXED_10BP_BAND")
        if row.grid_kind == "M2_CONTINUOUS":
            if not row.exact_cumulative_prefix:
                local.append("CUMULATIVE_VOLUME_PREFIX_NOT_EXACT")
            elif not row.conditional_clock_compatible:
                local.append("LEGACY_CONDITIONAL_CLOCK_INCOMPATIBLE")
            elif not row.conditional_range_before_endpoint:
                local.append("LEGACY_RANGE_NOT_ENTIRELY_BEFORE_ENDPOINT")
        reasons.append("|".join(local) if local else "LOCAL_FIELDS_OBSERVED_ONLY")
    out["local_diagnostic_reasons"] = reasons
    out["strict_point_qualified"] = False
    out["strict_reasons"] = "NO_RECEIVE_CLOCK|UNVERIFIED_VENDOR_TIME_MAPPING|NO_SYNC_REFERENCE_CONTRACT|NO_REFERENCE_ERROR_BOUND"
    out.loc[out.grid_kind.ne("M2_CONTINUOUS"), "strict_reasons"] += "|NO_AFTER_HOURS_QUEUE_REPLAY|IOPV_MAPPING_UNVERIFIED|CLOSE_COMPONENTS_UNVERIFIED"
    return out


def add_previous_day_counts(matrix: pd.DataFrame, baseline_days: int = 60, lag_minutes: int = 5) -> pd.DataFrame:
    """不读取收益；按旧M2逻辑核对当前固定价带、同会话五分钟前端点。"""
    out = matrix.sort_values(["date", "minute"], kind="stable").reset_index(drop=True).copy()
    continuous = out.grid_kind.eq("M2_CONTINUOUS")
    index = pd.MultiIndex.from_arrays([out.date, out.minute])
    lag_index = pd.MultiIndex.from_arrays([out.date, out.minute - lag_minutes])
    lag_book = pd.Series(out.source_nominal_book_observed.to_numpy(), index=index).reindex(lag_index).fillna(False).to_numpy(dtype=bool)
    lag_conditional = pd.Series(out.source_conditional_book_observed.to_numpy(), index=index).reindex(lag_index).fillna(False).to_numpy(dtype=bool)
    lag_session = pd.Series(out.session.to_numpy(), index=index).reindex(lag_index).to_numpy()
    same_session = out.session.to_numpy() == lag_session
    out["m2_same_session_lag_exists"] = continuous & same_session & lag_book
    out["m2_source_field_pair"] = continuous & same_session & lag_book & out.source_nominal_band_observed
    out["m2_source_conditional_pair"] = out.m2_source_field_pair & lag_conditional & out.source_conditional_band_observed
    out["m2_strict_field_pair"] = False
    groups = ["regime", "hhmm", "grid_kind"]
    out["calendar_prior_days"] = out.groupby(groups, sort=False).cumcount()
    for flag, suffix in [("source_nominal_book_observed", "nominal_book"),
                         ("source_nominal_band_observed", "nominal_band"),
                         ("m2_source_field_pair", "source_m2_pair"),
                         ("m2_source_conditional_pair", "conditional_m2_pair"),
                         ("m2_strict_field_pair", "strict_m2_pair")]:
        total = out.groupby(groups, sort=False)[flag].cumsum()
        out[f"prior_{suffix}_days"] = total.astype(int) - out[flag].astype(int)
    out["source_60day_m2_candidate"] = out.m2_source_field_pair & out.prior_source_m2_pair_days.ge(baseline_days)
    out["conditional_60day_m2_candidate"] = out.m2_source_conditional_pair & out.prior_conditional_m2_pair_days.ge(baseline_days)
    out["strict_60day_m2_candidate"] = out.m2_strict_field_pair & out.prior_strict_m2_pair_days.ge(baseline_days)
    return out
