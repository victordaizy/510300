"""510300方向切换V1：宏观事后归因、收益解剖、Oracle与预测能力前沿。"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.direction_switch_common_v1 import (
    ROOT,
    atomic_json,
    atomic_parquet,
    atomic_text,
    execution_targets_from_intervals,
    generated_at,
    index_cagr,
    maximum_round_trips_per_year,
    normalize_dividends,
    normalize_market,
    prepare_total_return_market,
    sha256_file,
    simulate_binary_execution_targets,
)


def load_diagnostic_inputs(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    market = normalize_market(
        pd.read_parquet(ROOT / inputs["diagnostic_etf"]["file"]), "510300.SH"
    )
    dividends = normalize_dividends(
        pd.read_csv(ROOT / inputs["dividends"]["file"])
    )
    if len(dividends) != int(inputs["dividends"]["required_events"]):
        raise ValueError("510300分红事件数不匹配")
    benchmark = pd.read_parquet(ROOT / inputs["diagnostic_h00300"]["file"])
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="raise")
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    if benchmark["date"].duplicated().any():
        raise ValueError("H00300存在重复日期")
    if set(benchmark["symbol"].dropna().astype(str).unique()) != {"H00300"}:
        raise ValueError("全收益基准代码不是H00300")
    if (benchmark["close"] <= 0.0).any():
        raise ValueError("H00300存在非正点位")
    market_audit = json.loads(
        (ROOT / inputs["diagnostic_market_audit"]["file"]).read_text(
            encoding="utf-8"
        )
    )
    if market_audit.get("status") != inputs["diagnostic_market_audit"][
        "required_status"
    ]:
        raise ValueError("2015市场输入审计未通过")
    prepared = prepare_total_return_market(market, dividends)
    start = pd.Timestamp(config["data_scope"]["diagnostic_start"])
    end = pd.Timestamp(config["data_scope"]["diagnostic_end"])
    warmup_source = normalize_market(
        pd.read_parquet(ROOT / inputs["if_execution_etf"]["file"]), "510300.SH"
    )
    overlap = prepared[["date", "open", "high", "low", "close"]].merge(
        warmup_source[["date", "open", "high", "low", "close"]],
        on="date",
        how="inner",
        suffixes=("_diagnostic", "_warmup"),
        validate="one_to_one",
    )
    if overlap.empty:
        raise ValueError("诊断行情与冻结R6行情没有重叠日期")
    for column in ["open", "high", "low", "close"]:
        maximum_difference = float(
            (
                overlap[f"{column}_diagnostic"]
                - overlap[f"{column}_warmup"]
            )
            .abs()
            .max()
        )
        if maximum_difference > 1e-12:
            raise ValueError(f"诊断行情与冻结R6行情重叠{column}不一致")
    prior = warmup_source.loc[warmup_source["date"] < start].tail(1)
    if len(prior) != 1:
        raise ValueError("冻结R6行情无法提供诊断起点前一个交易日")
    prepared_with_warmup = prepare_total_return_market(
        pd.concat([prior, market], ignore_index=True), dividends
    )
    selected = prepared_with_warmup.loc[
        prepared_with_warmup["date"].between(start, end)
    ].copy()
    if selected.empty or selected["date"].iloc[0] != start:
        raise ValueError("诊断行情未覆盖冻结起点")
    if selected["date"].iloc[-1] != end:
        raise ValueError("诊断行情未覆盖冻结终点")
    benchmark_dates = set(benchmark["date"])
    missing_benchmark = [
        date.date().isoformat()
        for date in selected["date"]
        if date not in benchmark_dates
    ]
    if missing_benchmark:
        raise ValueError(f"H00300缺少诊断交易日：{missing_benchmark[:5]}")
    return {
        "market": prepared_with_warmup,
        "selected_market": selected.reset_index(drop=True),
        "dividends": dividends,
        "benchmark": benchmark.reset_index(drop=True),
        "market_audit": market_audit,
        "warmup_audit": {
            "source_file": inputs["if_execution_etf"]["file"],
            "source_sha256": inputs["if_execution_etf"]["sha256"],
            "warmup_date": prior["date"].iloc[0].date().isoformat(),
            "overlap_rows": int(len(overlap)),
            "maximum_ohlc_difference": 0.0,
        },
    }


def _compound_return(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(np.expm1(np.log1p(values.to_numpy(dtype=float)).sum()))


def _return_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "observations": 0,
            "cumulative_return": None,
            "annualized_mean": None,
            "annualized_volatility": None,
            "sharpe_zero_cash_rate": None,
            "maximum_drawdown": None,
            "positive_fraction": None,
        }
    wealth = (1.0 + values).cumprod()
    peak = wealth.cummax()
    volatility = float(values.std(ddof=1) * math.sqrt(242.0)) if len(values) > 1 else 0.0
    annualized_mean = float(values.mean() * 242.0)
    return {
        "observations": int(len(values)),
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "annualized_mean": annualized_mean,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": annualized_mean / volatility if volatility > 0.0 else None,
        "maximum_drawdown": float((wealth / peak - 1.0).min()),
        "positive_fraction": float((values > 0.0).mean()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def _open_to_close_total_return(
    market: pd.DataFrame, start_index: int, end_index: int
) -> float | None:
    if start_index < 0 or end_index >= len(market) or start_index > end_index:
        return None
    entry = float(market.iloc[start_index]["open"])
    exit_close = float(market.iloc[end_index]["close"])
    dividends = float(
        market.iloc[start_index : end_index + 1]["cash_dividend_per_share"].sum()
    )
    return float((exit_close + dividends) / entry - 1.0)


def _close_to_close_total_return(
    market: pd.DataFrame, start_index: int, end_index: int
) -> float | None:
    if start_index < 0 or end_index >= len(market) or start_index >= end_index:
        return None
    entry = float(market.iloc[start_index]["close"])
    exit_close = float(market.iloc[end_index]["close"])
    dividends = float(
        market.iloc[start_index + 1 : end_index + 1]["cash_dividend_per_share"].sum()
    )
    return float((exit_close + dividends) / entry - 1.0)


def build_macro_post_mortem(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = config["macro_post_mortem"]
    paths = config["inputs"]
    source_result = json.loads(
        (ROOT / paths["macro_result"]["file"]).read_text(encoding="utf-8")
    )
    if source_result.get("decision") != paths["macro_result"]["required_decision"]:
        raise ValueError("宏观前序拒绝状态不匹配")
    events = pd.read_parquet(ROOT / paths["macro_events"]["file"])
    strategy = pd.read_parquet(ROOT / paths["macro_strategy_ledger"]["file"])
    buy_hold = pd.read_parquet(ROOT / paths["macro_buy_hold_ledger"]["file"])
    panel = pd.read_parquet(ROOT / paths["macro_point_in_time_panel"]["file"])
    market = inputs["market"].copy().reset_index(drop=True)
    for frame in [events, strategy, buy_hold, panel]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    events.sort_values("date", kind="mergesort", inplace=True)
    strategy.sort_values("date", kind="mergesort", inplace=True)
    buy_hold.sort_values("date", kind="mergesort", inplace=True)
    panel.sort_values("date", kind="mergesort", inplace=True)
    if not strategy["date"].equals(buy_hold["date"]):
        raise ValueError("宏观策略与买入持有账本日期不一致")
    if strategy["date"].duplicated().any() or panel["date"].duplicated().any():
        raise ValueError("宏观归因输入日期重复")
    market_index = {date: index for index, date in enumerate(market["date"])}
    panel_target = panel.set_index("date")["target_position_5y"]
    ledger = strategy[["date", "daily_return", "actual_position"]].merge(
        buy_hold[["date", "daily_return"]].rename(
            columns={"daily_return": "buy_hold_daily_return"}
        ),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    ledger["strategy_log_return"] = np.log1p(ledger["daily_return"])
    ledger["buy_hold_log_return"] = np.log1p(ledger["buy_hold_daily_return"])
    ledger["daily_log_excess"] = (
        ledger["strategy_log_return"] - ledger["buy_hold_log_return"]
    )
    ledger_by_date = ledger.set_index("date")
    market_dates = market["date"].tolist()
    rows: list[dict[str, Any]] = []

    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.date)
        if event_date not in market_index:
            raise ValueError(f"宏观事件不在510300日历：{event_date.date()}")
        event_index = market_index[event_date]
        execution_index = event_index + 1
        if execution_index >= len(market):
            execution_date = None
        else:
            execution_date = pd.Timestamp(market.iloc[execution_index]["date"])
        later_panel = panel.loc[panel["date"] > event_date]
        reentry_signals = later_panel.loc[later_panel["target_position_5y"].eq(1.0), "date"]
        reentry_signal_date = (
            pd.Timestamp(reentry_signals.iloc[0]) if not reentry_signals.empty else None
        )
        if reentry_signal_date is not None and reentry_signal_date in market_index:
            candidate_index = market_index[reentry_signal_date] + 1
            reentry_index = candidate_index if candidate_index < len(market) else None
        else:
            reentry_index = None
        reentry_date = (
            pd.Timestamp(market.iloc[reentry_index]["date"])
            if reentry_index is not None
            else None
        )
        episode_end_index = reentry_index if reentry_index is not None else len(market) - 1
        pre_start = max(0, event_index - int(contract["pre_signal_trading_days"]))
        pre_signal_return = _close_to_close_total_return(
            market, pre_start, event_index
        )
        if execution_date is not None:
            low_slice = market.iloc[execution_index : episode_end_index + 1]
            lowest_row = low_slice.loc[low_slice["low"].idxmin()]
            next_open_to_low = float(
                lowest_row["low"] / market.iloc[execution_index]["open"] - 1.0
            )
            lowest_date = pd.Timestamp(lowest_row["date"])
        else:
            next_open_to_low = None
            lowest_date = None
        if execution_date is not None:
            attribution_end = reentry_date or pd.Timestamp(ledger["date"].iloc[-1])
            episode_ledger = ledger_by_date.loc[
                (ledger_by_date.index >= execution_date)
                & (ledger_by_date.index <= attribution_end)
            ].copy()
        else:
            episode_ledger = ledger_by_date.iloc[0:0].copy()
        downside_excess = episode_ledger.loc[
            episode_ledger["buy_hold_daily_return"] < 0.0, "daily_log_excess"
        ]
        upside_excess = episode_ledger.loc[
            episode_ledger["buy_hold_daily_return"] > 0.0, "daily_log_excess"
        ]
        avoided_loss_log = float(downside_excess.clip(lower=0.0).sum())
        failed_avoidance_log = float((-downside_excess.clip(upper=0.0)).sum())
        missed_gain_log = float((-upside_excess.clip(upper=0.0)).sum())
        favorable_underexposure_log = float(upside_excess.clip(lower=0.0).sum())
        episode_log_excess = float(episode_ledger["daily_log_excess"].sum())
        recovery: dict[str, Any] = {}
        for horizon in contract["recovery_gap_horizons"]:
            if reentry_index is None:
                recovery[str(horizon)] = {
                    "date": None,
                    "overnight_gap_return": None,
                    "day_total_return": None,
                    "cumulative_from_reentry_open": None,
                }
                continue
            day_index = reentry_index + int(horizon) - 1
            if day_index >= len(market):
                recovery[str(horizon)] = {
                    "date": None,
                    "overnight_gap_return": None,
                    "day_total_return": None,
                    "cumulative_from_reentry_open": None,
                }
                continue
            recovery[str(horizon)] = {
                "date": pd.Timestamp(market.iloc[day_index]["date"]),
                "overnight_gap_return": float(
                    market.iloc[day_index]["overnight_standalone_return"]
                ),
                "day_total_return": float(market.iloc[day_index]["total_return"]),
                "cumulative_from_reentry_open": _open_to_close_total_return(
                    market, reentry_index, day_index
                ),
            }
        forward_20 = getattr(event, "etf_forward_20d")
        if pd.isna(forward_20):
            classification = "CENSORED_NO_20D"
        elif float(forward_20) < 0.0:
            classification = "CORRECT_NEGATIVE_20D"
        else:
            classification = "WRONG_NONNEGATIVE_20D"
        rows.append(
            {
                "event_id": str(event.event_id),
                "signal_date": event_date,
                "signal_target_position": float(panel_target.loc[event_date]),
                "execution_date": execution_date,
                "reentry_signal_date": reentry_signal_date,
                "reentry_execution_date": reentry_date,
                "episode_censored": reentry_date is None,
                "pre_signal_10d_total_return": pre_signal_return,
                "next_open_to_episode_low_price_return": next_open_to_low,
                "episode_low_date": lowest_date,
                "cash_or_underweight_trading_days": int(len(episode_ledger)),
                "average_missing_exposure": (
                    float((1.0 - episode_ledger["actual_position"]).mean())
                    if not episode_ledger.empty
                    else None
                ),
                "avoided_loss_log": avoided_loss_log,
                "avoided_loss_equivalent_return": float(np.expm1(avoided_loss_log)),
                "failed_avoidance_log": failed_avoidance_log,
                "missed_gain_log": missed_gain_log,
                "missed_gain_equivalent_return": float(np.expm1(missed_gain_log)),
                "favorable_underexposure_log": favorable_underexposure_log,
                "episode_log_excess_vs_buy_hold": episode_log_excess,
                "episode_relative_return_vs_buy_hold": float(np.expm1(episode_log_excess)),
                "etf_forward_20d": None if pd.isna(forward_20) else float(forward_20),
                "classification": classification,
                "recovery_gap_detail_json": json.dumps(
                    recovery, ensure_ascii=False, default=str
                ),
            }
        )

    event_frame = pd.DataFrame(rows)
    event_dates = [market_index[pd.Timestamp(value)] for value in events["date"]]
    primary_days = int(contract["key_miss_forward_days"])
    key_miss_candidates: list[dict[str, Any]] = []
    for index in range(len(market) - primary_days):
        date = pd.Timestamp(market.iloc[index]["date"])
        target = panel_target.get(date, np.nan)
        if pd.isna(target) or float(target) != 1.0:
            continue
        if any(abs(index - event_index) <= primary_days for event_index in event_dates):
            continue
        forward = _close_to_close_total_return(market, index, index + primary_days)
        if forward is None or forward > float(contract["key_miss_loss_threshold"]):
            continue
        key_miss_candidates.append(
            {
                "date": date,
                "forward_20d_total_return": float(forward),
                "target_position": float(target),
                "market_index": index,
            }
        )
    selected_misses: list[dict[str, Any]] = []
    for candidate in sorted(key_miss_candidates, key=lambda item: item["forward_20d_total_return"]):
        if any(
            abs(candidate["market_index"] - existing["market_index"])
            < int(contract["key_miss_independence_days"])
            for existing in selected_misses
        ):
            continue
        selected_misses.append(candidate)
        if len(selected_misses) >= int(contract["key_miss_count"]):
            break
    for item in selected_misses:
        item.pop("market_index", None)

    source_metrics = source_result["portfolio_backtest"]["five_year"]
    report = {
        "model_id": contract["model_id"],
        "status": contract["status"],
        "generated_at": generated_at(),
        "source_model": {
            "study_id": source_result["study_id"],
            "decision": source_result["decision"],
            "model_status": contract["source_model_status"],
            "mechanism_evidence": contract["mechanism_evidence"],
            "portfolio_eligible": contract["portfolio_eligible"],
            "paper_eligible": contract["paper_eligible"],
            "live_eligible": contract["live_eligible"],
            "rescue_allowed": contract["rescue_allowed"],
        },
        "frozen_result_recap": {
            "event_20d_mean": source_result["event_study"]["primary_event_summary"]["mean"],
            "event_20d_negative_fraction": source_result["event_study"][
                "primary_event_summary"
            ]["negative_fraction"],
            "net_sharpe": source_metrics["strategy"]["sharpe_zero_cash_rate"],
            "annualized_excess_vs_buy_hold": source_metrics[
                "annualized_excess_vs_buy_hold"
            ],
            "annualized_excess_vs_h00300": source_metrics[
                "annualized_excess_vs_h00300"
            ],
            "strategy_max_drawdown": source_metrics["strategy"]["max_drawdown"],
            "buy_hold_max_drawdown": source_metrics["buy_hold"]["max_drawdown"],
        },
        "event_count": int(len(event_frame)),
        "complete_event_count": int(
            event_frame["classification"].ne("CENSORED_NO_20D").sum()
        ),
        "correct_event_count": int(
            event_frame["classification"].eq("CORRECT_NEGATIVE_20D").sum()
        ),
        "wrong_event_count": int(
            event_frame["classification"].eq("WRONG_NONNEGATIVE_20D").sum()
        ),
        "censored_event_count": int(
            event_frame["classification"].eq("CENSORED_NO_20D").sum()
        ),
        "total_episode_log_excess_vs_buy_hold": float(
            event_frame["episode_log_excess_vs_buy_hold"].sum()
        ),
        "total_avoided_loss_equivalent_return": float(
            np.expm1(event_frame["avoided_loss_log"].sum())
        ),
        "total_missed_gain_equivalent_return": float(
            np.expm1(event_frame["missed_gain_log"].sum())
        ),
        "key_missed_drawdowns": selected_misses,
        "interpretation": "事件方向关联存在，但错过上涨、错误退出和恢复滞后超过避损贡献；仅作归因，不修改旧模型。",
        "model_change_allowed": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "live_trading_authorized": False,
    }
    return report, event_frame


def render_macro_post_mortem(report: dict[str, Any]) -> str:
    recap = report["frozen_result_recap"]
    lines = [
        "# 510300宏观压力模型事后归因",
        "",
        f"状态：`{report['status']}`",
        "",
        "旧模型保持永久拒绝；本报告只解释损益来源，不允许改变窗口、阈值、恢复规则或加入过滤器。",
        "",
        "## 冻结结论复核",
        "",
        f"- 事件后20日平均收益：{recap['event_20d_mean']:.4%}",
        f"- 事件后20日负收益比例：{recap['event_20d_negative_fraction']:.2%}",
        f"- 净夏普率：{recap['net_sharpe']:.6f}",
        f"- 相对510300含分红买入持有年化超额：{recap['annualized_excess_vs_buy_hold']:.4%}",
        f"- 相对H00300年化超额：{recap['annualized_excess_vs_h00300']:.4%}",
        "",
        "## 逐事件归因摘要",
        "",
        f"- 事件：{report['event_count']}个；完整{report['complete_event_count']}个，方向正确{report['correct_event_count']}个，错误{report['wrong_event_count']}个，删失{report['censored_event_count']}个。",
        f"- 避损等价累计：{report['total_avoided_loss_equivalent_return']:.4%}",
        f"- 错过上涨等价累计：{report['total_missed_gain_equivalent_return']:.4%}",
        f"- 事件期合计相对买入持有：{np.expm1(report['total_episode_log_excess_vs_buy_hold']):.4%}",
        "",
        "## 关键漏报",
        "",
        "| 起点 | 后20日总收益 | 当时目标仓位 |",
        "|---|---:|---:|",
    ]
    for item in report["key_missed_drawdowns"]:
        lines.append(
            f"| {pd.Timestamp(item['date']).date()} | {item['forward_20d_total_return']:.4%} | {item['target_position']:.0%} |"
        )
    lines.extend(
        [
            "",
            "逐事件明细另存为Parquet，含信号前10日、下一开盘至低点、避损、错过上涨、恢复后第1/3/5日跳空和每次相对贡献。",
            "",
            "`MODEL_CHANGE_ALLOWED=false`，`PAPER_ELIGIBLE=false`，`LIVE_ELIGIBLE=false`。",
            "",
        ]
    )
    return "\n".join(lines)


def build_return_anatomy(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = config["return_anatomy"]
    market = inputs["selected_market"].copy().reset_index(drop=True)
    valid = market.loc[market["total_return"].notna()].copy()
    total_log = float(np.log1p(valid["total_return"]).sum())
    tail_results: dict[str, Any] = {}
    for count in contract["tail_day_counts"]:
        count = int(count)
        best = valid.nlargest(count, "total_return")
        worst = valid.nsmallest(count, "total_return")
        best_log = float(np.log1p(best["total_return"]).sum())
        worst_log = float(np.log1p(worst["total_return"]).sum())
        tail_results[str(count)] = {
            "best_days_log_contribution": best_log,
            "best_days_equivalent_return": float(np.expm1(best_log)),
            "worst_days_log_contribution": worst_log,
            "worst_days_equivalent_return": float(np.expm1(worst_log)),
            "wealth_return_without_best_days": float(np.expm1(total_log - best_log)),
            "wealth_return_without_worst_days": float(np.expm1(total_log - worst_log)),
            "best_day_dates": [value.date().isoformat() for value in best["date"]],
            "worst_day_dates": [value.date().isoformat() for value in worst["date"]],
        }

    large_drop_threshold = float(
        valid["total_return"].quantile(float(contract["large_drop_quantile"]))
    )
    large_drop_indices = valid.index[valid["total_return"] <= large_drop_threshold].tolist()
    recovery_summary: dict[str, Any] = {}
    recovery_rows: list[dict[str, Any]] = []
    for horizon in contract["recovery_horizons"]:
        values: list[float] = []
        first_gap: list[float] = []
        capturable_from_next_open: list[float] = []
        for index in large_drop_indices:
            end_index = index + int(horizon)
            if end_index >= len(market):
                continue
            close_recovery = _close_to_close_total_return(market, index, end_index)
            next_open_capture = _open_to_close_total_return(market, index + 1, end_index)
            if close_recovery is None or next_open_capture is None:
                continue
            gap = float(market.iloc[index + 1]["overnight_standalone_return"])
            values.append(close_recovery)
            first_gap.append(gap)
            capturable_from_next_open.append(next_open_capture)
            recovery_rows.append(
                {
                    "drop_date": pd.Timestamp(market.iloc[index]["date"]),
                    "horizon": int(horizon),
                    "drop_day_total_return": float(market.iloc[index]["total_return"]),
                    "close_to_close_recovery": close_recovery,
                    "first_recovery_overnight_gap": gap,
                    "next_open_to_horizon_close_return": next_open_capture,
                }
            )
        recovery_summary[str(horizon)] = {
            "events": len(values),
            "mean_close_to_close_recovery": float(np.mean(values)) if values else None,
            "median_close_to_close_recovery": float(np.median(values)) if values else None,
            "positive_recovery_fraction": float(np.mean(np.asarray(values) > 0.0)) if values else None,
            "mean_first_overnight_gap": float(np.mean(first_gap)) if first_gap else None,
            "mean_next_open_capturable_return": (
                float(np.mean(capturable_from_next_open))
                if capturable_from_next_open
                else None
            ),
        }

    execution_capture: dict[str, Any] = {}
    for count in contract["tail_day_counts"]:
        count = int(count)
        worst = valid.nsmallest(count, "total_return")
        full_loss = float((-worst["total_return"].clip(upper=0.0)).sum())
        intraday_loss = float(
            (-worst["intraday_standalone_return"].clip(upper=0.0)).sum()
        )
        overnight_loss = float(
            (-worst["overnight_standalone_return"].clip(upper=0.0)).sum()
        )
        execution_capture[str(count)] = {
            "sum_full_day_losses": full_loss,
            "sum_loss_after_same_day_open": intraday_loss,
            "sum_loss_already_realized_by_open": overnight_loss,
            "perfect_prior_close_signal_avoidable_fraction_after_open": (
                intraday_loss / full_loss if full_loss > 0.0 else None
            ),
        }

    best_40 = valid.nlargest(40, "total_return")
    positive_total_contribution = float(best_40["total_return"].sum())
    positive_overnight_contribution = float(best_40["overnight_contribution"].sum())
    report = {
        "model_id": contract["model_id"],
        "status": "COMPLETED_POST_MORTEM_DIAGNOSTIC_ONLY",
        "generated_at": generated_at(),
        "scope": {
            "start": market["date"].iloc[0].date().isoformat(),
            "end": market["date"].iloc[-1].date().isoformat(),
            "observations": int(len(market)),
            "dividend_events_in_scope": int(
                market["cash_dividend_per_share"].gt(0.0).sum()
            ),
        },
        "identity_max_absolute_error": float(
            (
                market["total_return"]
                - market["overnight_contribution"]
                - market["intraday_contribution"]
            )
            .abs()
            .max()
        ),
        "return_summaries": {
            "total_return": _return_summary(market["total_return"]),
            "price_return": _return_summary(market["price_return"]),
            "overnight_standalone_return": _return_summary(
                market["overnight_standalone_return"]
            ),
            "intraday_standalone_return": _return_summary(
                market["intraday_standalone_return"]
            ),
        },
        "tail_day_contributions": tail_results,
        "large_drop_definition": {
            "quantile": float(contract["large_drop_quantile"]),
            "threshold": large_drop_threshold,
            "event_count": len(large_drop_indices),
        },
        "post_large_drop_recovery": recovery_summary,
        "best_40_day_gap_concentration": {
            "sum_total_return": positive_total_contribution,
            "sum_overnight_contribution": positive_overnight_contribution,
            "overnight_share_of_arithmetic_contribution": (
                positive_overnight_contribution / positive_total_contribution
                if positive_total_contribution != 0.0
                else None
            ),
        },
        "next_open_execution_capture": execution_capture,
        "interpretation_boundary": "隔夜与日内的单独复合路径不是两个可相加策略；只有贡献恒等式可用于逐日归因。",
        "strategy_created": False,
        "paper_eligible": False,
        "live_trading_authorized": False,
    }
    daily = market[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "cash_dividend_per_share",
            "price_return",
            "total_return",
            "overnight_contribution",
            "intraday_contribution",
            "overnight_standalone_return",
            "intraday_standalone_return",
            "total_wealth_index",
        ]
    ].copy()
    recovery_frame = pd.DataFrame(recovery_rows)
    report["recovery_event_rows"] = int(len(recovery_frame))
    return report, daily


def render_return_anatomy(report: dict[str, Any]) -> str:
    total = report["return_summaries"]["total_return"]
    overnight = report["return_summaries"]["overnight_standalone_return"]
    intraday = report["return_summaries"]["intraday_standalone_return"]
    lines = [
        "# 510300收益来源解剖 V1",
        "",
        f"状态：`{report['status']}`",
        "",
        f"区间：{report['scope']['start']}至{report['scope']['end']}，{report['scope']['observations']}个交易日，含{report['scope']['dividend_events_in_scope']}次除息。",
        "",
        "## 总体结构",
        "",
        f"- 含分红累计收益：{total['cumulative_return']:.4%}",
        f"- 含分红年化均值/波动率：{total['annualized_mean']:.4%} / {total['annualized_volatility']:.4%}",
        f"- 隔夜单独复合路径：{overnight['cumulative_return']:.4%}",
        f"- 日内单独复合路径：{intraday['cumulative_return']:.4%}",
        f"- 逐日贡献恒等式最大误差：{report['identity_max_absolute_error']:.3e}",
        "",
        "隔夜和日内的单独复合路径不能直接相加；逐日总收益只等于隔夜贡献加日内贡献。",
        "",
        "## 最好/最差交易日",
        "",
        "| 日数 | 最好日等价贡献 | 去掉最好日后的累计收益 | 最差日等价贡献 | 去掉最差日后的累计收益 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for count, item in report["tail_day_contributions"].items():
        lines.append(
            f"| {count} | {item['best_days_equivalent_return']:.4%} | {item['wealth_return_without_best_days']:.4%} | {item['worst_days_equivalent_return']:.4%} | {item['wealth_return_without_worst_days']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## 大跌后的修复与下一开盘约束",
            "",
            f"大跌冻结定义为日总收益不高于样本5%分位（{report['large_drop_definition']['threshold']:.4%}），共{report['large_drop_definition']['event_count']}次。",
            "",
            "| 修复期限 | 平均收盘到收盘修复 | 首日平均隔夜跳空 | 下一开盘后平均可捕获 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon, item in report["post_large_drop_recovery"].items():
        lines.append(
            f"| {horizon}日 | {item['mean_close_to_close_recovery']:.4%} | {item['mean_first_overnight_gap']:.4%} | {item['mean_next_open_capturable_return']:.4%} |"
        )
    lines.extend(
        [
            "",
            "本报告只回答收益结构和可执行性，不产生信号、仓位或订单。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_decision_blocks(
    market: pd.DataFrame, metric_start_index: int, horizon: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dividend_values = market["cash_dividend_per_share"].to_numpy(dtype=float)
    for year, group in market.iloc[metric_start_index:].groupby(
        market.iloc[metric_start_index:]["date"].dt.year, sort=True
    ):
        indices = group.index.to_list()
        if len(indices) <= horizon:
            continue
        starts = list(range(indices[0], indices[-1] + 1, int(horizon)))
        for sequence, start_index in enumerate(starts[:-1], start=1):
            end_index = starts[sequence]
            if end_index > indices[-1]:
                continue
            entry_open = float(market.iloc[start_index]["open"])
            exit_open = float(market.iloc[end_index]["open"])
            cash_dividend = float(dividend_values[start_index + 1 : end_index + 1].sum())
            gross_return = float((exit_open + cash_dividend) / entry_open - 1.0)
            rows.append(
                {
                    "block_id": f"H{horizon}_{int(year)}_{sequence:03d}",
                    "horizon": int(horizon),
                    "year": int(year),
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "start_date": pd.Timestamp(market.iloc[start_index]["date"]),
                    "end_date": pd.Timestamp(market.iloc[end_index]["date"]),
                    "gross_open_to_open_total_return": gross_return,
                    "negative_block": bool(gross_return < 0.0),
                }
            )
    return pd.DataFrame(rows)


def _negative_episodes(blocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in blocks.groupby("year", sort=True):
        negative = group.loc[group["negative_block"]].sort_values(
            "start_index", kind="mergesort"
        )
        current: list[pd.Series] = []
        for _, row in negative.iterrows():
            if current and int(current[-1]["end_index"]) != int(row["start_index"]):
                rows.append(_episode_row(int(year), current))
                current = []
            current.append(row)
        if current:
            rows.append(_episode_row(int(year), current))
    return pd.DataFrame(rows)


def _episode_row(year: int, members: list[pd.Series]) -> dict[str, Any]:
    log_return = float(
        sum(math.log1p(float(member["gross_open_to_open_total_return"])) for member in members)
    )
    start = members[0]
    end = members[-1]
    return {
        "episode_id": f"E_{year}_{pd.Timestamp(start['start_date']).strftime('%Y%m%d')}",
        "year": year,
        "start_index": int(start["start_index"]),
        "end_index": int(end["end_index"]),
        "start_date": pd.Timestamp(start["start_date"]),
        "end_date": pd.Timestamp(end["end_date"]),
        "block_count": len(members),
        "gross_log_return": log_return,
        "gross_total_return": float(np.expm1(log_return)),
        "avoided_gross_log_loss_score": float(-log_return),
    }


def _selected_episode_ids_for_threshold(
    episodes: pd.DataFrame, cap: int, threshold: float
) -> tuple[str, ...]:
    selected: list[str] = []
    for _, group in episodes.groupby("year", sort=True):
        candidates = group.loc[
            group["avoided_gross_log_loss_score"] >= threshold
        ].sort_values(
            ["avoided_gross_log_loss_score", "start_date"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected.extend(candidates.head(int(cap))["episode_id"].astype(str).tolist())
    return tuple(sorted(selected))


def _intervals_for_episode_ids(
    episodes: pd.DataFrame, episode_ids: tuple[str, ...]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not episode_ids:
        return []
    selected = episodes.loc[episodes["episode_id"].isin(episode_ids)].sort_values(
        "start_date", kind="mergesort"
    )
    return [
        (pd.Timestamp(row.start_date), pd.Timestamp(row.end_date))
        for row in selected.itertuples(index=False)
    ]


def _add_relative_metrics(
    metrics: pd.DataFrame,
    *,
    buy_hold: pd.Series,
    h00300_cagr: float,
) -> pd.DataFrame:
    output = metrics.copy()
    output["annualized_excess_vs_510300_tr"] = output["cagr"] - float(
        buy_hold["cagr"]
    )
    output["annualized_excess_vs_h00300"] = output["cagr"] - h00300_cagr
    denominator = abs(float(buy_hold["max_drawdown"]))
    output["max_drawdown_ratio_vs_510300"] = (
        output["max_drawdown"].abs() / denominator if denominator > 0.0 else np.nan
    )
    return output


def _simulate_target_batches(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: list[np.ndarray],
    execution: dict[str, Any],
    metric_start_index: int,
    *,
    double_cost: bool = False,
    batch_size: int = 256,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for start in range(0, len(targets), batch_size):
        batch = np.stack(targets[start : start + batch_size], axis=0)
        metrics, _ = simulate_binary_execution_targets(
            market,
            dividends,
            batch,
            execution,
            metric_start_index=metric_start_index,
            commission_rate=(
                float(execution["double_cost_commission_rate_per_leg"])
                if double_cost
                else None
            ),
            slippage_bps=(
                float(execution["double_cost_slippage_bps_per_leg"])
                if double_cost
                else None
            ),
        )
        metrics.index = range(start, start + len(metrics))
        rows.append(metrics)
    return pd.concat(rows).sort_index() if rows else pd.DataFrame()


def _diagnostic_simulation_frame(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    market = inputs["market"].copy()
    start = pd.Timestamp(config["data_scope"]["diagnostic_start"])
    end = pd.Timestamp(config["data_scope"]["diagnostic_end"])
    start_matches = market.index[market["date"].eq(start)].tolist()
    if len(start_matches) != 1 or start_matches[0] == 0:
        raise ValueError("诊断模拟需要冻结起点前一个交易日")
    warmup_index = start_matches[0] - 1
    simulation = market.iloc[warmup_index:].loc[lambda frame: frame["date"] <= end].copy()
    simulation.reset_index(drop=True, inplace=True)
    return simulation, 1


def build_constrained_oracle(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    contract = config["constrained_oracle"]
    execution = config["execution"]
    market, metric_start_index = _diagnostic_simulation_frame(config, inputs)
    buy_hold_target = np.ones(len(market), dtype=float)
    buy_hold_target[:metric_start_index] = 0.0
    buy_hold_metrics, _ = simulate_binary_execution_targets(
        market,
        inputs["dividends"],
        buy_hold_target,
        execution,
        metric_start_index=metric_start_index,
    )
    buy_hold = buy_hold_metrics.iloc[0]
    h00300_cagr = index_cagr(
        inputs["benchmark"]["date"],
        inputs["benchmark"]["close"],
        pd.Timestamp(market.iloc[metric_start_index]["date"]),
        pd.Timestamp(market.iloc[-1]["date"]),
    )
    all_grid_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    policy_targets: dict[str, np.ndarray] = {}
    block_sets: dict[int, dict[str, pd.DataFrame]] = {}

    for horizon in contract["decision_block_trading_days"]:
        horizon = int(horizon)
        blocks = _build_decision_blocks(market, metric_start_index, horizon)
        episodes = _negative_episodes(blocks)
        block_sets[horizon] = {"blocks": blocks, "episodes": episodes}
        thresholds = [float("inf")]
        thresholds.extend(
            sorted(
                set(episodes["avoided_gross_log_loss_score"].astype(float).tolist()),
                reverse=True,
            )
        )
        thresholds.append(float("-inf"))
        for cap in contract["annual_maximum_round_trips"]:
            cap = int(cap)
            definitions: list[dict[str, Any]] = []
            targets: list[np.ndarray] = []
            seen: set[tuple[str, ...]] = set()
            for threshold in thresholds:
                episode_ids = _selected_episode_ids_for_threshold(episodes, cap, threshold)
                if episode_ids in seen:
                    continue
                seen.add(episode_ids)
                intervals = _intervals_for_episode_ids(episodes, episode_ids)
                target = execution_targets_from_intervals(
                    market["date"], intervals, metric_start_index=metric_start_index
                )
                annual_counts = maximum_round_trips_per_year(market["date"], target)
                if annual_counts and max(annual_counts.values()) > cap:
                    raise ValueError("Oracle策略超过冻结的年度往返上限")
                definitions.append(
                    {
                        "horizon": horizon,
                        "annual_round_trip_cap": cap,
                        "threshold": threshold,
                        "selected_episode_ids": episode_ids,
                        "selected_episode_count": len(episode_ids),
                        "maximum_realized_annual_round_trips": (
                            max(annual_counts.values()) if annual_counts else 0
                        ),
                    }
                )
                targets.append(target)
            metrics = _simulate_target_batches(
                market,
                inputs["dividends"],
                targets,
                execution,
                metric_start_index,
            )
            metrics = _add_relative_metrics(
                metrics, buy_hold=buy_hold, h00300_cagr=h00300_cagr
            )
            definition_frame = pd.DataFrame(
                [
                    {
                        **item,
                        "selected_episode_ids_json": json.dumps(
                            item["selected_episode_ids"], ensure_ascii=False
                        ),
                    }
                    for item in definitions
                ]
            ).drop(columns=["selected_episode_ids"])
            grid = pd.concat(
                [definition_frame.reset_index(drop=True), metrics.reset_index(drop=True)],
                axis=1,
            )
            all_grid_rows.append(grid)
            best_sharpe_index = int(grid["net_sharpe"].idxmax())
            best_return_index = int(grid["total_return"].idxmax())
            best_sharpe_definition = definitions[best_sharpe_index]
            best_return_definition = definitions[best_return_index]
            best_sharpe_target = targets[best_sharpe_index]
            best_return_target = targets[best_return_index]
            double_metrics = _simulate_target_batches(
                market,
                inputs["dividends"],
                [best_sharpe_target, best_return_target],
                execution,
                metric_start_index,
                double_cost=True,
            )
            double_metrics = _add_relative_metrics(
                double_metrics, buy_hold=buy_hold, h00300_cagr=h00300_cagr
            )
            key_prefix = f"h{horizon}_cap{cap}"
            policy_targets[f"{key_prefix}_best_sharpe"] = best_sharpe_target
            policy_targets[f"{key_prefix}_best_return"] = best_return_target
            best_sharpe_row = grid.loc[best_sharpe_index].to_dict()
            best_return_row = grid.loc[best_return_index].to_dict()
            best_sharpe_row["double_cost_net_sharpe"] = float(
                double_metrics.iloc[0]["net_sharpe"]
            )
            best_return_row["double_cost_net_sharpe"] = float(
                double_metrics.iloc[1]["net_sharpe"]
            )
            best_family_sharpe = float(best_sharpe_row["net_sharpe"])
            if best_family_sharpe < 1.20:
                reachability = "ORACLE_FAMILY_BELOW_1_20_NO_SPACE_AT_THIS_FREQUENCY_CAP"
            elif best_family_sharpe <= 1.50:
                reachability = "ORACLE_FAMILY_1_20_TO_1_50_MINIMAL_ERROR_MARGIN"
            else:
                reachability = "ORACLE_FAMILY_ABOVE_1_50_PREDICTION_MARGIN_EXISTS"
            summaries.append(
                {
                    "horizon": horizon,
                    "annual_round_trip_cap": cap,
                    "candidate_negative_episodes": int(len(episodes)),
                    "finite_policy_count": int(len(grid)),
                    "best_sharpe_policy": best_sharpe_row,
                    "best_return_policy": best_return_row,
                    "reachability": reachability,
                    "best_sharpe_selected_episode_ids": list(
                        best_sharpe_definition["selected_episode_ids"]
                    ),
                    "best_return_selected_episode_ids": list(
                        best_return_definition["selected_episode_ids"]
                    ),
                }
            )

    grid_frame = pd.concat(all_grid_rows, ignore_index=True)
    best_overall = max(
        summaries, key=lambda item: item["best_sharpe_policy"]["net_sharpe"]
    )
    report = {
        "model_id": contract["model_id"],
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "generated_at": generated_at(),
        "scope": {
            "start": market.iloc[metric_start_index]["date"].date().isoformat(),
            "end": market.iloc[-1]["date"].date().isoformat(),
            "execution_asset": "510300.SH_OR_CASH",
            "initial_capital_cny": execution["initial_capital_cny"],
        },
        "search_family": contract["label"],
        "mathematical_upper_bound_claimed": contract["mathematical_upper_bound_claimed"],
        "buy_hold_metrics": buy_hold.to_dict(),
        "h00300_cagr": h00300_cagr,
        "constraint_summaries": summaries,
        "best_overall_search_family_sharpe": best_overall[
            "best_sharpe_policy"
        ]["net_sharpe"],
        "best_overall_constraint": {
            "horizon": best_overall["horizon"],
            "annual_round_trip_cap": best_overall["annual_round_trip_cap"],
            "reachability": best_overall["reachability"],
        },
        "strategy_created": False,
        "paper_eligible": False,
        "live_trading_authorized": False,
    }
    auxiliary = {
        "market": market,
        "metric_start_index": metric_start_index,
        "buy_hold": buy_hold,
        "h00300_cagr": h00300_cagr,
        "block_sets": block_sets,
        "policy_targets": policy_targets,
    }
    return report, grid_frame, auxiliary


def render_oracle(report: dict[str, Any]) -> str:
    lines = [
        "# 510300/现金受约束 Oracle V1",
        "",
        f"状态：`{report['status']}`",
        "",
        "这是使用事后未来信息的有限搜索族诊断，不是可交易策略，也不声称覆盖所有可能仓位路径的严格数学上界。",
        "",
        "| 决策块 | 年往返上限 | 搜索族最高净夏普 | 搜索族最高净收益 | 双倍成本夏普 | 可达性判断 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["constraint_summaries"]:
        sharpe = item["best_sharpe_policy"]
        returns = item["best_return_policy"]
        lines.append(
            f"| {item['horizon']}日 | {item['annual_round_trip_cap']} | {sharpe['net_sharpe']:.4f} | {returns['total_return']:.4%} | {sharpe['double_cost_net_sharpe']:.4f} | `{item['reachability']}` |"
        )
    lines.extend(
        [
            "",
            f"全表最高搜索族净夏普率：{report['best_overall_search_family_sharpe']:.4f}。",
            "",
            "Oracle结果只用于衡量预测误差容忍度；不能作为真实收益、候选信号或交易授权。",
            "",
        ]
    )
    return "\n".join(lines)


def _uniform_hash(seed: int, label: str) -> float:
    digest = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def _merge_intervals(
    intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _skill_target(
    market: pd.DataFrame,
    metric_start_index: int,
    true_episodes: pd.DataFrame,
    false_blocks: pd.DataFrame,
    *,
    recall: float,
    false_positive_rate: float,
    lead_days: int,
    recovery_lag_days: int,
    annual_cap: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions: list[dict[str, Any]] = []
    for row in true_episodes.itertuples(index=False):
        if _uniform_hash(seed, f"TP|{row.episode_id}") < recall:
            predictions.append(
                {
                    "kind": "TP",
                    "id": str(row.episode_id),
                    "start": max(metric_start_index, int(row.start_index) - lead_days),
                    "end": min(len(market), int(row.end_index) + recovery_lag_days),
                }
            )
    for row in false_blocks.itertuples(index=False):
        if _uniform_hash(seed, f"FP|{row.block_id}") < false_positive_rate:
            predictions.append(
                {
                    "kind": "FP",
                    "id": str(row.block_id),
                    "start": max(metric_start_index, int(row.start_index) - lead_days),
                    "end": min(len(market), int(row.end_index) + recovery_lag_days),
                }
            )
    predictions.sort(key=lambda item: (item["start"], item["kind"], item["id"]))
    accepted: list[dict[str, Any]] = []
    year_counts: dict[int, int] = {}
    for prediction in predictions:
        year = int(pd.Timestamp(market.iloc[prediction["start"]]["date"]).year)
        if year_counts.get(year, 0) >= annual_cap:
            continue
        year_counts[year] = year_counts.get(year, 0) + 1
        accepted.append(prediction)
    merged = _merge_intervals(
        [(int(item["start"]), int(item["end"])) for item in accepted]
    )
    target = np.ones(len(market), dtype=float)
    target[:metric_start_index] = 0.0
    for start, end in merged:
        target[start:end] = 0.0
    accepted_true = sum(item["kind"] == "TP" for item in accepted)
    accepted_false = sum(item["kind"] == "FP" for item in accepted)
    realized_recall = (
        accepted_true / len(true_episodes) if len(true_episodes) else 0.0
    )
    realized_fpr = accepted_false / len(false_blocks) if len(false_blocks) else 0.0
    precision = (
        accepted_true / (accepted_true + accepted_false)
        if accepted_true + accepted_false
        else 1.0
    )
    diagnostics = {
        "realized_recall": float(realized_recall),
        "realized_false_positive_rate": float(realized_fpr),
        "realized_precision": float(precision),
        "accepted_true_events": int(accepted_true),
        "accepted_false_events": int(accepted_false),
        "merged_cash_episodes": int(len(merged)),
        "maximum_realized_annual_round_trips": int(max(year_counts.values(), default=0)),
    }
    return target, diagnostics


def build_skill_frontier(
    config: dict[str, Any],
    inputs: dict[str, Any],
    oracle_auxiliary: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = config["skill_frontier"]
    execution = config["execution"]
    market: pd.DataFrame = oracle_auxiliary["market"]
    metric_start_index = int(oracle_auxiliary["metric_start_index"])
    buy_hold: pd.Series = oracle_auxiliary["buy_hold"]
    h00300_cagr = float(oracle_auxiliary["h00300_cagr"])
    parameter_rows: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []

    for horizon, recall, fpr, lead, lag, cap, seed in itertools.product(
        contract["downside_horizons"],
        contract["recall_grid"],
        contract["false_positive_rate_grid"],
        contract["lead_trading_days_grid"],
        contract["recovery_lag_trading_days_grid"],
        contract["annual_maximum_round_trips"],
        contract["deterministic_hash_seeds"],
    ):
        horizon = int(horizon)
        blocks = oracle_auxiliary["block_sets"][horizon]["blocks"]
        true_episodes = oracle_auxiliary["block_sets"][horizon]["episodes"]
        false_blocks = blocks.loc[~blocks["negative_block"]].copy()
        target, diagnostics = _skill_target(
            market,
            metric_start_index,
            true_episodes,
            false_blocks,
            recall=float(recall),
            false_positive_rate=float(fpr),
            lead_days=int(lead),
            recovery_lag_days=int(lag),
            annual_cap=int(cap),
            seed=int(seed),
        )
        parameter_rows.append(
            {
                "downside_horizon": horizon,
                "requested_recall": float(recall),
                "requested_false_positive_rate": float(fpr),
                "lead_trading_days": int(lead),
                "recovery_lag_trading_days": int(lag),
                "annual_round_trip_cap": int(cap),
                "seed": int(seed),
                **diagnostics,
            }
        )
        targets.append(target)

    base_metrics = _simulate_target_batches(
        market,
        inputs["dividends"],
        targets,
        execution,
        metric_start_index,
    )
    base_metrics = _add_relative_metrics(
        base_metrics, buy_hold=buy_hold, h00300_cagr=h00300_cagr
    )
    double_metrics = _simulate_target_batches(
        market,
        inputs["dividends"],
        targets,
        execution,
        metric_start_index,
        double_cost=True,
    )
    detail = pd.concat(
        [
            pd.DataFrame(parameter_rows).reset_index(drop=True),
            base_metrics.reset_index(drop=True),
        ],
        axis=1,
    )
    detail["double_cost_net_sharpe"] = double_metrics["net_sharpe"].to_numpy()
    group_columns = [
        "downside_horizon",
        "requested_recall",
        "requested_false_positive_rate",
        "lead_trading_days",
        "recovery_lag_trading_days",
        "annual_round_trip_cap",
    ]
    numeric_columns = [
        "realized_recall",
        "realized_false_positive_rate",
        "realized_precision",
        "accepted_true_events",
        "accepted_false_events",
        "merged_cash_episodes",
        "maximum_realized_annual_round_trips",
        "total_return",
        "cagr",
        "annualized_volatility",
        "net_sharpe",
        "max_drawdown",
        "average_exposure",
        "trade_count",
        "cash_entry_count",
        "turnover_over_initial_capital",
        "explicit_cost_cny",
        "slippage_cost_cny",
        "ending_equity_cny",
        "annualized_excess_vs_510300_tr",
        "annualized_excess_vs_h00300",
        "max_drawdown_ratio_vs_510300",
        "double_cost_net_sharpe",
    ]
    grouped = (
        detail.groupby(group_columns, as_index=False, sort=True)[numeric_columns]
        .median()
        .rename(columns={column: f"median_{column}" for column in numeric_columns})
    )
    gates = contract["gates"]
    grouped["hard_pass"] = (
        grouped["median_net_sharpe"].ge(float(gates["net_sharpe_min"]))
        & grouped["median_annualized_excess_vs_510300_tr"].gt(0.0)
        & grouped["median_annualized_excess_vs_h00300"].gt(0.0)
        & grouped["median_max_drawdown_ratio_vs_510300"].le(
            float(gates["max_drawdown_ratio_vs_510300_max"])
        )
        & grouped["median_double_cost_net_sharpe"].ge(
            float(gates["double_cost_net_sharpe_min"])
        )
    )
    passers = grouped.loc[grouped["hard_pass"]].copy()
    if passers.empty:
        least_demanding = None
    else:
        least = passers.sort_values(
            [
                "requested_recall",
                "requested_false_positive_rate",
                "lead_trading_days",
                "recovery_lag_trading_days",
                "annual_round_trip_cap",
            ],
            ascending=[True, False, True, False, True],
            kind="mergesort",
        ).iloc[0]
        least_demanding = least.to_dict()
    by_horizon: list[dict[str, Any]] = []
    for horizon, frame in grouped.groupby("downside_horizon", sort=True):
        passing = frame.loc[frame["hard_pass"]]
        by_horizon.append(
            {
                "horizon": int(horizon),
                "grid_cells": int(len(frame)),
                "passing_cells": int(len(passing)),
                "minimum_requested_recall_among_passers": (
                    float(passing["requested_recall"].min())
                    if not passing.empty
                    else None
                ),
                "maximum_requested_fpr_among_passers": (
                    float(passing["requested_false_positive_rate"].max())
                    if not passing.empty
                    else None
                ),
                "maximum_median_net_sharpe": float(frame["median_net_sharpe"].max()),
            }
        )
    report = {
        "model_id": contract["model_id"],
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "generated_at": generated_at(),
        "grid_cells": int(len(grouped)),
        "deterministic_seeds_per_cell": int(
            len(contract["deterministic_hash_seeds"])
        ),
        "passing_cells": int(grouped["hard_pass"].sum()),
        "least_demanding_passing_cell": least_demanding,
        "by_horizon": by_horizon,
        "hard_gates": gates,
        "interpretation": "该曲线量化冻结标签与误差模型下所需分类能力，不证明任何现实信号能达到该能力。",
        "strategy_created": False,
        "paper_eligible": False,
        "live_trading_authorized": False,
    }
    return report, grouped


def render_skill_frontier(report: dict[str, Any]) -> str:
    lines = [
        "# 510300夏普目标预测能力前沿 V1",
        "",
        f"状态：`{report['status']}`",
        "",
        f"冻结网格共{report['grid_cells']}格，每格取{report['deterministic_seeds_per_cell']}个确定性哈希样本的中位数；通过全部硬门的格子为{report['passing_cells']}个。",
        "",
        "| 负收益标签期限 | 网格数 | 通过数 | 通过格最低请求召回率 | 通过格最高请求假阳性率 | 网格最高净夏普 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["by_horizon"]:
        recall = (
            "无" if item["minimum_requested_recall_among_passers"] is None else f"{item['minimum_requested_recall_among_passers']:.0%}"
        )
        fpr = (
            "无" if item["maximum_requested_fpr_among_passers"] is None else f"{item['maximum_requested_fpr_among_passers']:.0%}"
        )
        lines.append(
            f"| {item['horizon']}日 | {item['grid_cells']} | {item['passing_cells']} | {recall} | {fpr} | {item['maximum_median_net_sharpe']:.4f} |"
        )
    if report["least_demanding_passing_cell"] is None:
        lines.extend(
            [
                "",
                "冻结误差网格内没有任何分类能力组合同时满足夏普、双基准超额、回撤和双倍成本门。",
            ]
        )
    else:
        item = report["least_demanding_passing_cell"]
        lines.extend(
            [
                "",
                "按冻结排序得到的最低要求通过格：",
                "",
                f"- 标签期限：{int(item['downside_horizon'])}日",
                f"- 请求召回率/假阳性率：{item['requested_recall']:.0%} / {item['requested_false_positive_rate']:.0%}",
                f"- 提前量/恢复滞后：{int(item['lead_trading_days'])} / {int(item['recovery_lag_trading_days'])}个交易日",
                f"- 年往返上限：{int(item['annual_round_trip_cap'])}",
                f"- 中位净夏普：{item['median_net_sharpe']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "该结果是目标可达性诊断，不是现实预测模型，也不授权Paper、订单或实盘。",
            "",
        ]
    )
    return "\n".join(lines)


def write_diagnostic_artifacts(
    config: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    post_mortem, post_mortem_events = build_macro_post_mortem(config, inputs)
    atomic_parquet(
        post_mortem_events, ROOT / artifacts["macro_post_mortem_events"]
    )
    atomic_json(post_mortem, ROOT / artifacts["macro_post_mortem_json"])
    atomic_text(
        render_macro_post_mortem(post_mortem),
        ROOT / artifacts["macro_post_mortem_markdown"],
    )

    anatomy, anatomy_daily = build_return_anatomy(config, inputs)
    atomic_parquet(anatomy_daily, ROOT / artifacts["return_anatomy_daily"])
    atomic_json(anatomy, ROOT / artifacts["return_anatomy_json"])
    atomic_text(
        render_return_anatomy(anatomy),
        ROOT / artifacts["return_anatomy_markdown"],
    )

    oracle, oracle_grid, oracle_auxiliary = build_constrained_oracle(config, inputs)
    atomic_parquet(oracle_grid, ROOT / artifacts["oracle_grid"])
    atomic_json(oracle, ROOT / artifacts["oracle_json"])
    atomic_text(render_oracle(oracle), ROOT / artifacts["oracle_markdown"])

    frontier, frontier_grid = build_skill_frontier(
        config, inputs, oracle_auxiliary
    )
    atomic_parquet(frontier_grid, ROOT / artifacts["skill_frontier_grid"])
    atomic_json(frontier, ROOT / artifacts["skill_frontier_json"])
    atomic_text(
        render_skill_frontier(frontier),
        ROOT / artifacts["skill_frontier_markdown"],
    )
    return {
        "macro_post_mortem": post_mortem,
        "return_anatomy": anatomy,
        "oracle": oracle,
        "skill_frontier": frontier,
        "artifact_hashes": {
            relative: sha256_file(ROOT / relative)
            for relative in [
                artifacts["macro_post_mortem_json"],
                artifacts["macro_post_mortem_markdown"],
                artifacts["macro_post_mortem_events"],
                artifacts["return_anatomy_json"],
                artifacts["return_anatomy_markdown"],
                artifacts["return_anatomy_daily"],
                artifacts["oracle_json"],
                artifacts["oracle_markdown"],
                artifacts["oracle_grid"],
                artifacts["skill_frontier_json"],
                artifacts["skill_frontier_markdown"],
                artifacts["skill_frontier_grid"],
            ]
        },
    }
