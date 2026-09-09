"""比较延后重新计算与震荡缩波入场的近五年持股期收益。

这是原冻结模型终局淘汰后的事后历史诊断。脚本不覆盖原信号、不连接券商、
不生成订单，也不把表现较好的变体自动升级为 Shadow 或实盘模型。
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/a_share_hs_low_risk_vol_compression_delay_sideways_v1.yaml"
BACKTEST_PATH = ROOT / "scripts/run_a_share_hs_concentrated_low_risk_trend_v1_6_backtest.py"
ACTION_PATH = ROOT / "scripts/audit_a_share_hs_concentrated_low_risk_trend_v1_5_corporate_actions.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BACKTEST = _load_module("a_share_hs_delay_sideways_backtest_base", BACKTEST_PATH)
ACTION = _load_module("a_share_hs_delay_sideways_action_base", ACTION_PATH)
ENGINE = BACKTEST.ENGINE


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_codes(value: str) -> list[str]:
    return [str(item) for item in json.loads(value)]


def variant_name(trend_mode: str, delay_days: int) -> str:
    prefix = "POSITIVE" if trend_mode == "POSITIVE_TREND" else "SIDEWAYS"
    return f"{prefix}_RECALC_D{delay_days}"


def common_entry_mask(features: pd.DataFrame) -> pd.Series:
    """复现冻结条件中除趋势以外的全部硬门槛。"""

    return (
        features["base_eligible"].fillna(False).astype(bool)
        & features["total_mcap_20d_median"].ge(10_000_000_000)
        & features["float_mcap_20d_median"].ge(5_000_000_000)
        & features["float_mcap_percentile"].ge(0.30)
        & features["amount_20d_median"].ge(100_000_000)
        & features["suspension_days_20"].eq(0)
        & features["zero_volume_days_20"].eq(0)
        & features["valid_returns_120"].ge(115)
        & features["risk_score"].le(0.12)
        & features["risk_score_t_minus_20"].le(0.20)
        & features["risk_score_t_minus_40"].le(0.25)
        & features["stable_risk_score"].le(0.18)
        & features["worst_component_rank"].le(0.35)
        & features["ts_vol_percentile"].le(0.10)
        & features["signal_date_tradable"].fillna(False).astype(bool)
    )


def attach_diagnostic_entry_flags(features: pd.DataFrame) -> pd.DataFrame:
    """增加正趋势和非确认阴跌两种进入标记，并核对原冻结标记。"""

    result = features
    common = common_entry_mask(result)
    close = pd.to_numeric(result["total_return_close"], errors="coerce")
    ma120 = pd.to_numeric(result["ma120"], errors="coerce")
    momentum = pd.to_numeric(result["momentum_60_5"], errors="coerce")
    positive = common & close.ge(ma120) & momentum.gt(0)
    confirmed_downtrend = close.lt(ma120) & momentum.lt(0)
    sideways = common & ~confirmed_downtrend
    frozen = result["entry_eligible"].fillna(False).astype(bool)
    if not frozen.equals(positive.fillna(False).astype(bool)):
        mismatch = int((frozen != positive.fillna(False).astype(bool)).sum())
        raise RuntimeError(f"重建的正趋势进入条件与冻结标记不一致：{mismatch} 行")
    result["entry_positive"] = positive.fillna(False).astype(bool)
    result["entry_sideways"] = sideways.fillna(False).astype(bool)
    result["confirmed_downtrend_diagnostic"] = confirmed_downtrend.fillna(False).astype(bool)
    return result


def select_names(
    snapshot: pd.DataFrame,
    return_history: pd.DataFrame,
    *,
    trend_mode: str,
    maximum_names: int,
    selection: dict[str, Any],
) -> list[str]:
    """沿用冻结行业和相关性约束；震荡模式只改变第三排序键。"""

    ranked = snapshot.copy()
    if trend_mode == "NO_CONFIRMED_DOWNTREND_SIDEWAYS_PREFERRED":
        ranked["momentum_60_5"] = -pd.to_numeric(
            ranked["momentum_60_5"], errors="coerce"
        ).abs()
    return ENGINE.select_low_risk_names(
        ranked,
        return_history,
        maximum_names=maximum_names,
        max_names_per_industry=int(selection["max_names_per_industry"]),
        max_pairwise_correlation=float(selection["max_pairwise_correlation"]),
        min_pair_observations=int(selection["correlation_min_valid_observations"]),
    )


def build_recalculation_events(
    features: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    reference_events: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在原定信号日后第N个交易日重新计算全市场候选与排序。"""

    cutoff = pd.Timestamp(contract["as_of_date"]).normalize()
    delays = [int(value) for value in contract["recalculation_delay_trading_days"]]
    trend_modes = list(contract["trend_modes"])
    reference_dates = pd.DatetimeIndex(reference_events["signal_date"].unique()).sort_values()
    calendar_position = {date: index for index, date in enumerate(calendar_dates)}
    needed_dates: set[pd.Timestamp] = set(reference_dates)
    for origin_date in reference_dates:
        origin_position = calendar_position.get(origin_date)
        if origin_position is None:
            raise RuntimeError(f"参考信号日不在交易日历：{origin_date.date()}")
        for delay in delays:
            target_position = origin_position + delay
            if target_position < len(calendar_dates) and calendar_dates[target_position] <= cutoff:
                needed_dates.add(calendar_dates[target_position])
    snapshot_source = features.loc[features["date"].isin(needed_dates)].copy()
    snapshots = {
        pd.Timestamp(date).normalize(): frame.copy()
        for date, frame in snapshot_source.groupby("date", sort=False)
    }
    return_panel = features.pivot_table(
        index="date", columns="ts_code", values="log_return", aggfunc="last"
    ).sort_index()
    events: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selection = contract["selection"]
    screened_maximum = int(selection["maximum_names_screened"])
    correlation_window = int(selection["correlation_window_days"])
    flag_by_mode = {
        "POSITIVE_TREND": "entry_positive",
        "NO_CONFIRMED_DOWNTREND_SIDEWAYS_PREFERRED": "entry_sideways",
    }
    for trend_mode in trend_modes:
        flag = flag_by_mode[trend_mode]
        for delay in delays:
            variant = variant_name(trend_mode, delay)
            for origin_date in reference_dates:
                origin_position = calendar_position[origin_date]
                recalculation_position = origin_position + delay
                if recalculation_position >= len(calendar_dates):
                    continue
                recalculation_date = calendar_dates[recalculation_position]
                if recalculation_date > cutoff:
                    continue
                origin_snapshot = snapshots[origin_date]
                recalculation_snapshot = snapshots[recalculation_date].copy()
                origin_codes = set(
                    origin_snapshot.loc[origin_snapshot[flag].fillna(False), "ts_code"].astype(str)
                )
                recalculated_mask = recalculation_snapshot[flag].fillna(False).astype(bool)
                recalculation_snapshot["entry_eligible"] = recalculated_mask
                return_history = return_panel.loc[:recalculation_date].tail(correlation_window)
                selected = select_names(
                    recalculation_snapshot,
                    return_history,
                    trend_mode=trend_mode,
                    maximum_names=screened_maximum,
                    selection=selection,
                )
                top3 = selected[: int(selection["maximum_names_held"])]
                events.append(
                    {
                        "variant": variant,
                        "trend_mode": trend_mode,
                        "delay_trading_days": delay,
                        "origin_signal_date": origin_date,
                        "signal_date": recalculation_date,
                        "origin_candidate_count": len(origin_codes),
                        "candidate_count": int(recalculated_mask.sum()),
                        "top1": json.dumps(selected[:1], ensure_ascii=False),
                        "top3": json.dumps(top3, ensure_ascii=False),
                        "top5": json.dumps(selected[:5], ensure_ascii=False),
                        "top20": json.dumps(selected[:20], ensure_ascii=False),
                        "top3_count": len(top3),
                    }
                )
                by_code = recalculation_snapshot.drop_duplicates("ts_code").set_index("ts_code")
                for rank, code in enumerate(selected, start=1):
                    row = by_code.loc[code]
                    selected_rows.append(
                        {
                            "variant": variant,
                            "trend_mode": trend_mode,
                            "delay_trading_days": delay,
                            "origin_signal_date": origin_date,
                            "recalculation_date": recalculation_date,
                            "ts_code": code,
                            "rank": rank,
                            "stable_risk_score": float(row["stable_risk_score"]),
                            "ts_vol_percentile": float(row["ts_vol_percentile"]),
                            "momentum_60_5": float(row["momentum_60_5"]),
                            "absolute_momentum_60_5": abs(float(row["momentum_60_5"])),
                            "total_return_close_minus_ma120": float(
                                row["total_return_close"] - row["ma120"]
                            ),
                            "industry_l1": str(row["industry_l1"]),
                        }
                    )
    del return_panel, snapshots, snapshot_source
    gc.collect()
    event_frame = pd.DataFrame(events).sort_values(
        ["variant", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)
    selected_frame = pd.DataFrame(selected_rows)
    if not selected_frame.empty:
        selected_frame = selected_frame.sort_values(
            ["variant", "recalculation_date", "rank"], kind="mergesort"
        ).reset_index(drop=True)
    return event_frame, selected_frame


def validate_positive_delay_zero(
    generated_events: pd.DataFrame,
    reference_events: pd.DataFrame,
) -> None:
    """未延后的正趋势变体必须逐日复现冻结信号。"""

    generated = generated_events.loc[
        generated_events["variant"].eq("POSITIVE_RECALC_D0"),
        ["signal_date", "candidate_count", "top20"],
    ].copy()
    reference = reference_events[["signal_date", "candidate_count", "top20"]].copy()
    merged = reference.merge(
        generated,
        on="signal_date",
        how="outer",
        suffixes=("_reference", "_generated"),
        indicator=True,
    )
    count_equal = merged["candidate_count_reference"].eq(merged["candidate_count_generated"])
    top_equal = merged["top20_reference"].eq(merged["top20_generated"])
    if not merged["_merge"].eq("both").all() or not count_equal.all() or not top_equal.all():
        bad = merged.loc[
            ~merged["_merge"].eq("both") | ~count_equal | ~top_equal,
            ["signal_date", "_merge", "candidate_count_reference", "candidate_count_generated"],
        ]
        raise RuntimeError(f"正趋势D0未复现冻结信号：{bad.head(5).to_dict('records')}")


def audit_zero_action_fallback(
    ts_code: str,
    cutoff: pd.Timestamp,
    primary_failure: dict[str, Any],
) -> dict[str, Any]:
    """当CNINFO解析失败时，以两套分红历史和本地复权因子证明零公司行动。"""

    symbol = ts_code.split(".", 1)[0]
    try:
        sina = ACTION.ak.stock_history_dividend_detail(symbol, indicator="分红")
        ths = ACTION.ak.stock_fhps_detail_ths(symbol)
    except Exception as error:
        return {
            "ts_code": ts_code,
            "status": "FAILED",
            "primary_failure": primary_failure,
            "fallback_error": f"{type(error).__name__}: {error}",
        }
    required_sina = {
        "公告日期", "送股", "转增", "派息", "进度", "除权除息日", "股权登记日", "红股上市日",
    }
    required_ths = {
        "报告期", "董事会日期", "分红方案说明", "A股股权登记日", "A股除权除息日",
    }
    if not required_sina.issubset(sina.columns) or not required_ths.issubset(ths.columns):
        return {
            "ts_code": ts_code,
            "status": "FAILED",
            "primary_failure": primary_failure,
            "fallback_error": "替代分红源字段不完整",
            "sina_columns": [str(value) for value in sina.columns],
            "ths_columns": [str(value) for value in ths.columns],
        }
    sina = sina.copy()
    sina["公告日期"] = pd.to_datetime(sina["公告日期"], errors="coerce").dt.normalize()
    sina = sina.loc[sina["公告日期"].le(cutoff)].copy()
    sina_numeric = sina[["送股", "转增", "派息"]].apply(pd.to_numeric, errors="coerce")
    sina_zero = bool(
        not sina.empty
        and sina_numeric.notna().all().all()
        and np.isclose(sina_numeric.to_numpy(dtype=float), 0.0).all()
        and sina[["除权除息日", "股权登记日", "红股上市日"]].isna().all().all()
    )
    ths = ths.copy()
    ths_board_dates = pd.to_datetime(ths["董事会日期"], errors="coerce").dt.normalize()
    ths = ths.loc[ths_board_dates.le(cutoff)].copy()
    ths_zero = bool(
        not ths.empty
        and ths["分红方案说明"].astype(str).str.contains("不分配不转增", regex=False).all()
        and ths[["A股股权登记日", "A股除权除息日"]].isna().all().all()
    )
    checkpoint = ROOT / "data/raw/constituents/.tushare_stock_history_cache" / f"{ts_code.replace('.', '_')}.parquet"
    if not checkpoint.exists():
        return {
            "ts_code": ts_code,
            "status": "FAILED",
            "primary_failure": primary_failure,
            "fallback_error": f"缺少本地复权因子检查点：{checkpoint.relative_to(ROOT).as_posix()}",
        }
    prices = pd.read_parquet(checkpoint)
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    prices = prices.loc[prices["date"].le(cutoff)].copy()
    factors = pd.to_numeric(prices["adj_factor"], errors="coerce")
    price_equal = np.isclose(
        pd.to_numeric(prices["raw_close"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(prices["total_return_close"], errors="coerce").to_numpy(dtype=float),
        rtol=0,
        atol=1e-10,
        equal_nan=False,
    )
    factor_zero_action = bool(
        not prices.empty
        and factors.notna().all()
        and factors.nunique(dropna=True) == 1
        and np.isclose(float(factors.iloc[0]), 1.0, rtol=0, atol=1e-12)
        and price_equal.all()
    )
    if not (sina_zero and ths_zero and factor_zero_action):
        return {
            "ts_code": ts_code,
            "status": "FAILED",
            "primary_failure": primary_failure,
            "fallback_error": "替代分红源或复权因子未能共同证明零公司行动",
            "sina_zero_action": sina_zero,
            "ths_zero_action": ths_zero,
            "factor_zero_action": factor_zero_action,
        }
    return {
        "ts_code": ts_code,
        "status": "SUCCESS_ZERO_ACTION_CROSS_SOURCE_FALLBACK",
        "rows": 0,
        "primary_failure": primary_failure,
        "sina_history_rows_through_cutoff": len(sina),
        "ths_history_rows_through_cutoff": len(ths),
        "local_price_rows_through_cutoff": len(prices),
        "local_price_start": prices["date"].min().date().isoformat(),
        "local_price_end": prices["date"].max().date().isoformat(),
        "adj_factor_min": float(factors.min()),
        "adj_factor_max": float(factors.max()),
        "raw_close_equals_total_return_close_all_rows": True,
        "source_urls": [
            "https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/688981.phtml",
            "https://basic.10jqka.com.cn/new/688981/bonus.html",
            "https://www.sse.com.cn/market/stockdata/dividends/dividend/index_his.shtml",
            "https://www.smics.com/uploads/67f6423d/e00981.pdf",
        ],
        "interpretation": "截至截止日，两套分红历史均为不分配不转增，且本地逐日复权因子恒为1、复权收盘与原始收盘完全一致，因此公司行动事件为0行。",
    }


def load_corporate_actions(
    execution_codes: set[str],
    contract: dict[str, Any],
    coverage_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """复用已审计快照，并对新增入选证券补抓公司行动；任何失败均停止。"""

    prior_raw = pd.read_parquet(project_path(contract["sources"]["prior_corporate_action_raw"]))
    prior_receipt = load_json(
        project_path(contract["sources"]["prior_corporate_action_fetch_receipt"])
    )
    covered_codes = {
        str(row["ts_code"])
        for row in prior_receipt.get("fetch", [])
        if row.get("status") == "SUCCESS"
    }
    missing_codes = sorted(execution_codes - covered_codes)
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    fetched_frames: list[pd.DataFrame] = []
    fetch_receipts: list[dict[str, Any]] = []
    for code in missing_codes:
        frame, receipt = ACTION._fetch_one(code, retrieved_at)
        if receipt.get("status") == "SUCCESS":
            fetched_frames.append(frame)
            fetch_receipts.append(receipt)
        else:
            fallback = audit_zero_action_fallback(code, cutoff, receipt)
            fetch_receipts.append(fallback)
    failures = [
        row
        for row in fetch_receipts
        if row.get("status") not in {"SUCCESS", "SUCCESS_ZERO_ACTION_CROSS_SOURCE_FALLBACK"}
    ]
    if failures:
        raise RuntimeError(f"新增证券公司行动抓取失败，停止回测：{failures}")
    frames = [prior_raw.loc[prior_raw["ts_code"].astype(str).isin(execution_codes)].copy()]
    frames.extend(frame for frame in fetched_frames if not frame.empty)
    raw = pd.concat(frames, ignore_index=True) if frames else prior_raw.iloc[0:0].copy()
    raw = raw.loc[raw["ts_code"].astype(str).isin(execution_codes)].copy()
    events, all_events = ACTION._normalize_events(raw, coverage_start, cutoff)
    receipt = {
        "status": "CORPORATE_ACTIONS_READY_FOR_DIAGNOSTIC",
        "retrieved_at": retrieved_at,
        "execution_symbol_count": len(execution_codes),
        "reused_prior_success_symbol_count": len(execution_codes & covered_codes),
        "new_fetch_symbol_count": len(missing_codes),
        "new_fetch": fetch_receipts,
        "failed_fetch_count": 0,
        "raw_rows": len(raw),
        "evaluation_window_event_rows": len(events),
        "all_date_event_rows": len(all_events),
        "coverage_start": coverage_start.date().isoformat(),
        "cutoff": cutoff.date().isoformat(),
    }
    return raw, events, receipt


def trailing_metrics(
    nav: pd.DataFrame,
    benchmark_nav: pd.Series,
    trades: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    window_start: pd.Timestamp,
) -> dict[str, Any]:
    ordered_dates = pd.DatetimeIndex(
        pd.to_datetime(nav["date"], errors="raise").dt.normalize().unique()
    ).sort_values()
    first = int(ordered_dates.searchsorted(window_start, side="left"))
    predecessor = max(0, first - 1)
    start_with_predecessor = ordered_dates[predecessor]
    sub_nav = nav.loc[
        pd.to_datetime(nav["date"], errors="raise").dt.normalize().ge(start_with_predecessor)
    ].copy()
    sub_trades = [
        row for row in trades if pd.Timestamp(row["date"]).normalize() >= window_start
    ]
    sub_actions = [
        row for row in actions
        if pd.Timestamp(row["date"]).normalize() >= start_with_predecessor
    ]
    metrics = BACKTEST._holding_period_metrics(
        sub_nav, benchmark_nav, sub_trades, sub_actions
    )
    metrics["calendar_window_start"] = window_start.date().isoformat()
    metrics["calendar_window_end"] = ordered_dates[-1].date().isoformat()
    return metrics


def trade_costs(trades: list[dict[str, Any]], window_start: pd.Timestamp | None = None) -> dict[str, Any]:
    filtered = [
        row for row in trades
        if window_start is None or pd.Timestamp(row["date"]).normalize() >= window_start
    ]
    commission = float(sum(float(row["commission"]) for row in filtered))
    stamp = float(sum(float(row["stamp_duty"]) for row in filtered))
    slippage = float(
        sum(
            abs(float(row["execution_price"]) - float(row["market_open"]))
            * int(row["shares"])
            for row in filtered
        )
    )
    return {
        "trade_legs": len(filtered),
        "commission_cny": commission,
        "stamp_duty_cny": stamp,
        "slippage_cny": slippage,
        "total_direct_cost_cny": commission + stamp + slippage,
    }


def holding_return_ledger(
    nav: pd.DataFrame,
    trades: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    benchmark_nav: pd.Series,
    variant: str,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    """导出逐持股风险区间收益，纯现金区间不进入输出。"""

    strategy = nav.sort_values("date").drop_duplicates("date").copy()
    strategy["date"] = pd.to_datetime(strategy["date"], errors="raise").dt.normalize()
    strategy = strategy.set_index("date")
    benchmark = benchmark_nav.reindex(strategy.index).ffill().bfill().astype(float)
    strategy_return = strategy["equity"].astype(float).pct_change().fillna(0.0)
    benchmark_return = benchmark.pct_change().fillna(0.0)
    current_position = strategy["positions_count"].astype(int).gt(0)
    previous_position = current_position.shift(1, fill_value=False)
    trade_dates = pd.DatetimeIndex(
        pd.to_datetime([row["date"] for row in trades], errors="raise")
    ).normalize() if trades else pd.DatetimeIndex([])
    action_dates = pd.DatetimeIndex(
        pd.to_datetime([row["date"] for row in actions], errors="raise")
    ).normalize() if actions else pd.DatetimeIndex([])
    date_positions = {date: index for index, date in enumerate(strategy.index)}
    action_recognition_dates: set[pd.Timestamp] = set()
    for action_date in action_dates:
        position = date_positions.get(action_date)
        if position is not None and position + 1 < len(strategy.index):
            action_recognition_dates.add(strategy.index[position + 1])
    trade_flag = pd.Series(strategy.index.isin(trade_dates), index=strategy.index)
    action_flag = pd.Series(strategy.index.isin(action_recognition_dates), index=strategy.index)
    active = current_position | previous_position | trade_flag | action_flag
    active.iloc[0] = False
    reasons: list[str] = []
    for date in strategy.index:
        parts: list[str] = []
        if bool(current_position.loc[date]):
            parts.append("CURRENT_CLOSE_POSITION")
        if bool(previous_position.loc[date]):
            parts.append("PREVIOUS_CLOSE_POSITION")
        if bool(trade_flag.loc[date]):
            parts.append("TRADE_DATE")
        if bool(action_flag.loc[date]):
            parts.append("CORPORATE_ACTION_CASH_RECOGNITION")
        reasons.append("|".join(parts))
    ledger = pd.DataFrame(
        {
            "variant": variant,
            "date": strategy.index,
            "strategy_return": strategy_return,
            "h00300_return": benchmark_return,
            "excess_return": strategy_return - benchmark_return,
            "equity": strategy["equity"].astype(float),
            "cash": strategy["cash"].astype(float),
            "stock_exposure": (
                (strategy["equity"] - strategy["cash"]) / strategy["equity"]
            ).clip(0.0, 1.0),
            "positions_count_close": strategy["positions_count"].astype(int),
            "active_reason": reasons,
            "is_holding_risk_interval": active,
        }
    )
    ledger = ledger.loc[ledger["is_holding_risk_interval"]].copy().reset_index(drop=True)
    ledger["strategy_holding_nav"] = (1.0 + ledger["strategy_return"]).cumprod()
    ledger["h00300_same_dates_nav"] = (1.0 + ledger["h00300_return"]).cumprod()
    ledger["trailing_five_year_window"] = ledger["date"].ge(window_start)
    return ledger


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def markdown_report(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for name, result in report["variants"].items():
        focus = result["trailing_five_years"]
        rows.append(
            "| {name} | {mode} | {delay} | {active}/{events} | {intervals} | {total:.2%} | "
            "{annual:.2%} | {vol:.2%} | {drawdown:.2%} | {benchmark:.2%} | {ir:.3f} | {cost:,.2f} |".format(
                name=name,
                mode="正趋势" if result["trend_mode"] == "POSITIVE_TREND" else "非确认阴跌、偏好震荡",
                delay=result["delay_trading_days"],
                active=result["trailing_recalculation_dates_with_candidates"],
                events=result["trailing_recalculation_dates"],
                intervals=focus["holding_intervals"],
                total=focus["total_return"],
                annual=focus["annualized_return_on_holding_clock"],
                vol=focus["annualized_volatility_on_holding_clock"],
                drawdown=focus["max_drawdown_on_holding_clock"],
                benchmark=focus["h00300_total_return_same_holding_intervals"],
                ir=focus["information_ratio_on_holding_clock"],
                cost=result["trailing_costs"]["total_direct_cost_cny"],
            )
        )
    best = report["descriptive_comparison"]["highest_trailing_total_return_variant"]
    return "\n".join(
        [
            "# 沪深A股低风险缩波：延后重算与震荡偏好诊断",
            "",
            f"- 数据截止：{report['date_end']}",
            f"- 状态：`{report['status']}`",
            "- 性质：原模型终局淘汰后的事后历史诊断；不得据此挑选最佳变体进入Shadow或实盘。",
            "- 延后定义：原定信号日后0/5/10个交易日，对全市场重新计算全部条件、候选和排序，再于下一交易日开盘成交；允许延后日出现全新候选。",
            "- 震荡定义：排除“复权收盘价低于MA120且MOM60_5为负”；其余候选按绝对动量接近0优先。",
            "- 缩波硬门槛：TSVolPct ≤ 10%。",
            "- 收益口径：20万元、可执行候选等权并重新分配、整手、含佣金/印花税/滑点，仅计算持股经济风险区间，纯现金区间剔除。",
            "- 停牌估值：持仓在不可交易且无当日收盘时，按上一有效收盘结转并写入独立估值审计表；不得成交。",
            "- 停牌风险检查：不可交易且无点时特征时记为延期，不沿用旧指标冒充当日指标，待后续可计算的计划检查日再判断。",
            "",
            "## 最近五年结果",
            "",
            "| 变体 | 趋势定义 | 延后交易日 | 有候选重算日/重算日 | 持股区间 | 累计收益 | 持股时钟年化 | 年化波动 | 最大回撤 | 沪深300同日收益 | IR | 直接成本(元) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"历史描述上累计收益最高的是 `{best}`，但这是同一历史样本上的事后比较，不构成模型选择结论。",
            "",
            "## 对账与特殊状态",
            "",
            f"- 正趋势D0信号复现：{report['baseline_positive_d0_signal_reconciled']}；收益复现：{report['baseline_positive_d0_return_reconciled']}。",
            f"- 公司行动覆盖：{report['corporate_action_coverage']['execution_symbol_count']}只；复用{report['corporate_action_coverage']['reused_prior_success_symbol_count']}只，新补{report['corporate_action_coverage']['new_fetch_symbol_count']}只，失败{report['corporate_action_coverage']['failed_fetch_count']}只。",
            f"- 停牌上一收盘结转：{report['suspension_mark_carry_forward_rows']}行、{report['suspension_mark_carry_forward_unique_dates']}个唯一日期。",
            f"- 停牌风险检查延期：{report['deferred_untradable_risk_review_rows']}行、{report['deferred_untradable_risk_review_unique_dates']}个唯一日期。",
            f"- 持股收益输出中的纯现金行：{report['pure_cash_rows_in_holding_return_output']}。",
            "",
            "## 治理边界",
            "",
            "- 原V1.7淘汰结论保持不变。",
            "- 不生成订单，不连接券商，不授权Shadow或实盘。",
            "- 若要继续，应另行预注册单一规则并使用未见数据验证，不能从本表挑最高收益者直接采用。",
            "",
        ]
    )


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(contract["as_of_date"]).normalize()
    window_start = pd.Timestamp(contract["simulation"]["trailing_window_start"]).normalize()
    reference_events = pd.read_parquet(
        project_path(contract["sources"]["reference_signal_events"])
    )
    reference_events["signal_date"] = pd.to_datetime(
        reference_events["signal_date"], errors="raise"
    ).dt.normalize()
    signal_config = yaml.safe_load(
        project_path(contract["signal_config"]).read_text(encoding="utf-8")
    )
    features, calendar_dates, market_receipts = BACKTEST.V141.V14.build_signal_panel(
        signal_config
    )
    features = attach_diagnostic_entry_flags(features)
    events, selected = build_recalculation_events(
        features, calendar_dates, reference_events, contract
    )
    validate_positive_delay_zero(events, reference_events)
    execution_codes: set[str] = set()
    for value in events["top3"]:
        execution_codes.update(parse_codes(value))
    if not execution_codes:
        raise RuntimeError("六个诊断变体均无可执行候选，无法计算持股期收益")

    sim_start = pd.Timestamp(reference_events["signal_date"].min()).normalize()
    sim_dates = calendar_dates[(calendar_dates >= sim_start) & (calendar_dates <= cutoff)]
    capital = float(contract["simulation"]["initial_capital_cny"])
    benchmark_navs, benchmark_files = BACKTEST._benchmark_navs(
        features, sim_dates, capital, contract
    )
    features = features.loc[features["ts_code"].astype(str).isin(execution_codes)].copy()
    execution_market, market_files = BACKTEST._load_execution_market(
        project_path(contract["sources"]["daily_market_manifest"]), execution_codes
    )
    action_raw, corporate_actions, action_receipt = load_corporate_actions(
        execution_codes, contract, sim_start, cutoff
    )

    nav_frames: list[pd.DataFrame] = []
    trade_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    valuation_rows: list[dict[str, Any]] = []
    deferred_risk_rows: list[dict[str, Any]] = []
    holding_ledgers: list[pd.DataFrame] = []
    variant_results: dict[str, Any] = {}
    for trend_mode in contract["trend_modes"]:
        for delay in [int(value) for value in contract["recalculation_delay_trading_days"]]:
            variant = variant_name(trend_mode, delay)
            variant_events = events.loc[events["variant"].eq(variant)].copy()
            nav, trades, ledger = BACKTEST._simulate_variant(
                variant_name=variant,
                maximum_names=int(contract["selection"]["maximum_names_held"]),
                events=variant_events,
                features=features,
                market=execution_market,
                corporate_actions=corporate_actions,
                calendar_dates=calendar_dates,
                capital=capital,
                review_frequency=int(
                    contract["simulation"]["risk_review_frequency_trading_days"]
                ),
                allocation_mode=str(contract["simulation"]["allocation_mode"]),
                carry_forward_untradable_marks=True,
            )
            applied_actions = [row for row in ledger if "entitled_shares" in row]
            applied_valuations = [row for row in ledger if "valuation_method" in row]
            deferred_reviews = [row for row in ledger if "risk_review_method" in row]
            full_metrics = BACKTEST._holding_period_metrics(
                nav, benchmark_navs["H00300_TOTAL_RETURN"], trades, applied_actions
            )
            focus_metrics = trailing_metrics(
                nav,
                benchmark_navs["H00300_TOTAL_RETURN"],
                trades,
                applied_actions,
                window_start,
            )
            holding = holding_return_ledger(
                nav,
                trades,
                applied_actions,
                benchmark_navs["H00300_TOTAL_RETURN"],
                variant,
                window_start,
            )
            if len(holding) != int(full_metrics["holding_intervals"]):
                raise RuntimeError(f"{variant} 持股收益明细行数与全样本指标不一致")
            focus_holding = holding.loc[holding["trailing_five_year_window"]]
            if len(focus_holding) != int(focus_metrics["holding_intervals"]):
                raise RuntimeError(f"{variant} 最近五年持股收益明细行数不一致")
            focus_compound = float((1.0 + focus_holding["strategy_return"]).prod() - 1.0)
            if not np.isclose(
                focus_compound, float(focus_metrics["total_return"]), rtol=0, atol=1e-12
            ):
                raise RuntimeError(f"{variant} 最近五年逐日复合收益不一致")
            nav_frames.append(nav)
            trade_rows.extend(trades)
            action_rows.extend(applied_actions)
            valuation_rows.extend(applied_valuations)
            deferred_risk_rows.extend(deferred_reviews)
            holding_ledgers.append(holding)
            recent_events = variant_events.loc[variant_events["signal_date"].ge(window_start)]
            latest = variant_events.sort_values("signal_date").iloc[-1]
            variant_results[variant] = {
                "trend_mode": trend_mode,
                "delay_trading_days": delay,
                "recalculation_dates": len(variant_events),
                "recalculation_dates_with_candidates": int(
                    variant_events["candidate_count"].gt(0).sum()
                ),
                "trailing_recalculation_dates": len(recent_events),
                "trailing_recalculation_dates_with_candidates": int(
                    recent_events["candidate_count"].gt(0).sum()
                ),
                "latest_recalculation_date": pd.Timestamp(latest["signal_date"]).date().isoformat(),
                "latest_origin_signal_date": pd.Timestamp(
                    latest["origin_signal_date"]
                ).date().isoformat(),
                "latest_top3": parse_codes(str(latest["top3"])),
                "full_sample": full_metrics,
                "trailing_five_years": focus_metrics,
                "full_costs": trade_costs(trades),
                "trailing_costs": trade_costs(trades, window_start),
            }

    nav_frame = pd.concat(nav_frames, ignore_index=True)
    trades_frame = pd.DataFrame(trade_rows)
    actions_frame = pd.DataFrame(action_rows)
    valuations_frame = pd.DataFrame(valuation_rows)
    deferred_risk_frame = pd.DataFrame(deferred_risk_rows)
    holding_frame = pd.concat(holding_ledgers, ignore_index=True)
    if int((~holding_frame["is_holding_risk_interval"]).sum()) != 0:
        raise RuntimeError("持股收益输出中仍含纯现金区间")

    prior_report = load_json(
        ROOT
        / "reports/backtest/a_share_hs_concentrated_low_risk_trend_v1_8_1_full_invested_200k_holding_period_corrected.json"
    )
    prior_base = prior_report["variants"]["TOP3"]
    generated_base = variant_results["POSITIVE_RECALC_D0"]["full_sample"]
    generated_focus = variant_results["POSITIVE_RECALC_D0"]["trailing_five_years"]
    baseline_reconciled = bool(
        np.isclose(
            float(prior_base["total_return"]),
            float(generated_base["total_return"]),
            rtol=0,
            atol=1e-12,
        )
        and np.isclose(
            float(prior_base["trailing_year_focus"]["total_return"]),
            float(generated_focus["total_return"]),
            rtol=0,
            atol=1e-12,
        )
    )
    if not baseline_reconciled:
        raise RuntimeError("正趋势D0收益未复现V1.8.1基线")

    output_paths = {
        key: project_path(value)
        for key, value in contract["outputs"].items()
        if key != "root"
    }
    project_path(contract["outputs"]["root"]).mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_paths["recalculation_events"], index=False)
    selected.to_parquet(output_paths["selected_symbols"], index=False)
    nav_frame.to_parquet(output_paths["nav_daily"], index=False)
    trades_frame.to_parquet(output_paths["trades"], index=False)
    action_raw.to_parquet(output_paths["corporate_action_raw"], index=False)
    corporate_actions.to_parquet(output_paths["corporate_action_events"], index=False)
    output_paths["corporate_action_fetch_receipt"].write_text(
        json.dumps(json_ready(action_receipt), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    actions_frame.to_parquet(output_paths["corporate_action_ledger"], index=False)
    valuations_frame.to_parquet(output_paths["suspension_valuation_ledger"], index=False)
    deferred_risk_frame.to_parquet(output_paths["deferred_risk_review_ledger"], index=False)
    holding_frame.to_parquet(output_paths["holding_interval_returns"], index=False)
    holding_frame.to_csv(
        output_paths["holding_interval_returns_csv"],
        index=False,
        encoding="utf-8-sig",
        float_format="%.12f",
    )

    best_total = max(
        variant_results,
        key=lambda name: variant_results[name]["trailing_five_years"]["total_return"],
    )
    best_ir = max(
        variant_results,
        key=lambda name: variant_results[name]["trailing_five_years"][
            "information_ratio_on_holding_clock"
        ],
    )
    report = {
        "schema_version": "A_SHARE_HS_DELAY_SIDEWAYS_DIAGNOSTIC_V1",
        "contract_id": contract["contract_id"],
        "parent_diagnostic_contract_id": contract["parent_diagnostic_contract_id"],
        "parent_terminal_contract_id": contract["parent_terminal_contract_id"],
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "POST_RESULT_DIAGNOSTIC_COMPLETED_NO_SHADOW_NO_LIVE",
        "view_status": "POST_RESULT_HISTORICAL_DIAGNOSTIC",
        "date_start": sim_dates[0].date().isoformat(),
        "date_end": sim_dates[-1].date().isoformat(),
        "trailing_window_start": window_start.date().isoformat(),
        "trailing_window_end": cutoff.date().isoformat(),
        "delay_semantics": contract["research_question"]["delay_semantics"],
        "trend_definitions": contract["trend_modes"],
        "baseline_positive_d0_signal_reconciled": True,
        "baseline_positive_d0_return_reconciled": baseline_reconciled,
        "execution_symbol_count": len(execution_codes),
        "variants": variant_results,
        "descriptive_comparison": {
            "highest_trailing_total_return_variant": best_total,
            "highest_trailing_information_ratio_variant": best_ir,
            "selection_authorized": False,
        },
        "corporate_action_coverage": action_receipt,
        "input_market_receipts": market_receipts,
        "return_files_opened": sorted(set(market_files + benchmark_files)),
        "pure_cash_rows_in_holding_return_output": int(
            (~holding_frame["is_holding_risk_interval"]).sum()
        ),
        "suspension_mark_carry_forward_rows": len(valuations_frame),
        "suspension_mark_carry_forward_unique_dates": int(
            valuations_frame["date"].nunique() if not valuations_frame.empty else 0
        ),
        "deferred_untradable_risk_review_rows": len(deferred_risk_frame),
        "deferred_untradable_risk_review_unique_dates": int(
            deferred_risk_frame["date"].nunique() if not deferred_risk_frame.empty else 0
        ),
        "signals_generated": True,
        "simulated_positions_generated": True,
        "simulated_trades_generated": True,
        "orders_generated": False,
        "broker_connection": False,
        "shadow_authorized": False,
        "live_trading_authorized": False,
        "governance": contract["governance"],
        "artifacts": {
            key: path.relative_to(ROOT).as_posix() for key, path in output_paths.items()
        },
    }
    report = json_ready(report)
    output_paths["report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_paths["report_markdown"].write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "状态": report["status"],
                "基线信号复现": report["baseline_positive_d0_signal_reconciled"],
                "基线收益复现": report["baseline_positive_d0_return_reconciled"],
                "最近五年结果": {
                    name: {
                        "累计收益": result["trailing_five_years"]["total_return"],
                        "持股时钟年化": result["trailing_five_years"][
                            "annualized_return_on_holding_clock"
                        ],
                        "最大回撤": result["trailing_five_years"][
                            "max_drawdown_on_holding_clock"
                        ],
                        "持股区间": result["trailing_five_years"]["holding_intervals"],
                    }
                    for name, result in report["variants"].items()
                },
                "纯现金输出行": report["pure_cash_rows_in_holding_return_output"],
                "订单生成": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
