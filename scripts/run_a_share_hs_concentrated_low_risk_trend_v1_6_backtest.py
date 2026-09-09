"""读取已冻结信号和公司行动，执行沪深 A 股历史小账户回测。"""

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
CONFIG = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1_6_backtest.yaml"
V141_PATH = ROOT / "scripts/freeze_a_share_hs_concentrated_low_risk_trend_v1_4_1_signal.py"
V141_SPEC = importlib.util.spec_from_file_location("a_share_hs_signal_v141_for_backtest", V141_PATH)
assert V141_SPEC and V141_SPEC.loader
V141 = importlib.util.module_from_spec(V141_SPEC)
sys.modules[V141_SPEC.name] = V141
V141_SPEC.loader.exec_module(V141)
ENGINE = V141.V14.ENGINE


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_top(value: str) -> list[str]:
    parsed = json.loads(value)
    return [str(item) for item in parsed]


def _nav_from_close(close: pd.Series, dates: pd.DatetimeIndex, capital: float) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").sort_index()
    aligned = values.reindex(dates).ffill().bfill()
    if aligned.isna().any() or (aligned <= 0).any():
        raise ValueError("基准价格在回测日期上存在缺失或非正值")
    return aligned / aligned.iloc[0] * capital


def _benchmark_navs(
    features: pd.DataFrame,
    dates: pd.DatetimeIndex,
    capital: float,
    contract: dict[str, Any],
) -> tuple[dict[str, pd.Series], list[str]]:
    opened_files: list[str] = []
    h_path = project_path(contract["sources"]["h00300_total_return"])
    etf_path = project_path(contract["sources"]["etf_510300_checkpoint"])
    h00300 = pd.read_parquet(h_path, columns=["date", "close"])
    etf = pd.read_parquet(etf_path, columns=["date", "total_return_close"])
    for frame in (h00300, etf):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    opened_files.extend([str(h_path.relative_to(ROOT)), str(etf_path.relative_to(ROOT))])
    eligible = features.loc[
        features["base_eligible"].fillna(False).astype(bool)
        & features["date"].between(dates[0], dates[-1], inclusive="both")
    ].copy()
    eligible["simple_return"] = np.expm1(eligible["log_return"].astype(float))
    equal_return = eligible.groupby("date", sort=True)["simple_return"].mean()
    weighted = eligible.dropna(subset=["simple_return", "float_mcap_20d_median"])
    weighted["weighted_return"] = weighted["simple_return"] * weighted["float_mcap_20d_median"]
    float_return = weighted.groupby("date", sort=True)["weighted_return"].sum() / weighted.groupby("date", sort=True)["float_mcap_20d_median"].sum()
    def compound(returns: pd.Series) -> pd.Series:
        aligned = returns.reindex(dates).fillna(0.0)
        aligned.iloc[0] = 0.0
        return capital * (1.0 + aligned).cumprod()
    navs = {
        "H00300_TOTAL_RETURN": _nav_from_close(
            h00300.set_index("date")["close"], dates, capital
        ),
        "ETF_510300_BUY_AND_HOLD": _nav_from_close(
            etf.set_index("date")["total_return_close"], dates, capital
        ),
        "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT": compound(equal_return),
        "ELIGIBLE_UNIVERSE_FLOAT_MCAP_WEIGHTED": compound(float_return),
    }
    del eligible, weighted, h00300, etf
    gc.collect()
    return navs, opened_files


def _load_execution_market(manifest_path: Path, codes: set[str]) -> tuple[pd.DataFrame, list[str]]:
    market, receipts = V141.V14._load_daily_market(manifest_path)
    market = market.loc[market["ts_code"].isin(codes), [
        "ts_code", "date", "raw_open", "raw_close", "volume_shares", "is_suspended",
    ]].copy()
    return market, [str(item["path"]) for item in receipts]


def _fully_invested_executable_targets(
    selected: list[str],
    *,
    account: Any,
    opens: dict[str, float],
    marks: dict[str, float],
    tradable: dict[str, bool],
) -> dict[str, float]:
    """剔除等权预算下不可整手执行的证券，并把权重重新分配给剩余证券。"""

    remaining = list(dict.fromkeys(map(str, selected)))
    if not remaining:
        return {}
    equity = float(account.equity(marks))
    while remaining:
        weight = 1.0 / len(remaining)
        executable: list[str] = []
        for code in remaining:
            open_price = float(opens.get(code, np.nan))
            if not np.isfinite(open_price) or open_price <= 0 or not tradable.get(code, False):
                continue
            raw_target = int(np.floor(equity * weight / open_price))
            shares = max(0, raw_target // 100 * 100)
            if shares >= 100 and shares * open_price >= 5_000:
                executable.append(code)
        if executable == remaining:
            return {code: weight for code in remaining}
        remaining = executable
    return {}


def _simulate_variant(
    *,
    variant_name: str,
    maximum_names: int,
    events: pd.DataFrame,
    features: pd.DataFrame,
    market: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    capital: float,
    review_frequency: int,
    allocation_mode: str = "FROZEN_CASH_LADDER",
    carry_forward_untradable_marks: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    start_date = pd.Timestamp(events["signal_date"].min()).normalize()
    end_date = pd.Timestamp(market["date"].max()).normalize()
    dates = calendar_dates[(calendar_dates >= start_date) & (calendar_dates <= end_date)]
    anchor_index = int(calendar_dates.searchsorted(start_date))
    full_map = {
        pd.Timestamp(row.signal_date).normalize(): _parse_top(row.top20)[:maximum_names]
        for row in events.itertuples(index=False)
    }
    full_dates = set(full_map)
    risk_dates = {
        date for date in dates
        if (int(calendar_dates.searchsorted(date)) - anchor_index) % review_frequency == 0
        and date not in full_dates
    }
    feature_codes = set(features["ts_code"].astype(str))
    market = market.loc[market["ts_code"].isin(feature_codes)].copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market_days = {date: day.set_index("ts_code") for date, day in market.groupby("date", sort=False)}
    feature_days = {date: day.set_index("ts_code") for date, day in features.groupby("date", sort=False)}
    action_records = corporate_actions.copy()
    action_records["effective_date"] = pd.to_datetime(action_records["effective_date"], errors="raise").dt.normalize()
    action_records["record_date"] = pd.to_datetime(action_records["record_date"], errors="coerce").dt.normalize()
    record_map = {date: frame for date, frame in action_records.dropna(subset=["record_date"]).groupby("record_date", sort=False)}
    effective_map = {date: frame for date, frame in action_records.groupby("effective_date", sort=False)}
    account = ENGINE.SmallAccount(initial_cash=capital)
    pending_target: list[str] | None = None
    pending_risk_exits: set[str] = set()
    entitlements: dict[int, int] = {}
    previous_marks: dict[str, float] = {}
    nav_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    action_ledger: list[dict[str, Any]] = []
    valuation_ledger: list[dict[str, Any]] = []
    risk_review_ledger: list[dict[str, Any]] = []
    missing_mark_dates: list[str] = []
    for date in dates:
        day = market_days.get(date, pd.DataFrame())
        opens = {} if day.empty else pd.to_numeric(day["raw_open"], errors="coerce").dropna().to_dict()
        closes = {} if day.empty else pd.to_numeric(day["raw_close"], errors="coerce").dropna().to_dict()
        tradable = {}
        if not day.empty:
            for code, row in day.iterrows():
                tradable[str(code)] = bool(
                    pd.notna(row.get("raw_open"))
                    and pd.notna(row.get("raw_close"))
                    and not bool(row.get("is_suspended", True))
                    and float(row.get("volume_shares", 0) or 0) > 0
                )
        if pending_target is not None:
            if allocation_mode == "FULLY_INVESTED_EQUAL_WEIGHT_REDISTRIBUTE_EXECUTABLE":
                targets = _fully_invested_executable_targets(
                    pending_target,
                    account=account,
                    opens=opens,
                    marks=previous_marks,
                    tradable=tradable,
                )
            elif allocation_mode == "FULLY_INVESTED_EQUAL_WEIGHT":
                targets = (
                    {code: 1.0 / len(pending_target) for code in pending_target}
                    if pending_target
                    else {}
                )
            elif allocation_mode == "FROZEN_CASH_LADDER":
                targets = ENGINE.target_weights(
                    pending_target,
                    variant=maximum_names if maximum_names in {1, 3, 5, 20} else 3,
                )
            else:
                raise ValueError(f"未知仓位分配模式：{allocation_mode}")
            account.rebalance(
                date=date,
                targets=targets,
                opens=opens,
                marks=previous_marks,
                tradable=tradable,
                maximum_one_lot_equity_weight=(
                    None
                    if allocation_mode == "FULLY_INVESTED_EQUAL_WEIGHT_REDISTRIBUTE_EXECUTABLE"
                    else 0.40
                ),
                reason="FULL_SELECTION",
            )
            pending_target = None
        for code in sorted(pending_risk_exits):
            if code in account.positions and tradable.get(code, False):
                account._execute(date, code, "SELL", account.positions[code], float(opens[code]), "EMERGENCY_EXIT")
                pending_risk_exits.discard(code)
        marks: dict[str, float] = {}
        for code in account.positions:
            if code not in closes or not np.isfinite(float(closes[code])):
                if (
                    carry_forward_untradable_marks
                    and not tradable.get(code, False)
                    and code in previous_marks
                    and np.isfinite(float(previous_marks[code]))
                ):
                    marks[code] = float(previous_marks[code])
                    valuation_ledger.append({
                        "date": date,
                        "variant": variant_name,
                        "ts_code": code,
                        "valuation_method": "PREVIOUS_VALID_CLOSE_CARRY_FORWARD_UNTRADABLE",
                        "mark_price": float(previous_marks[code]),
                        "tradable": False,
                    })
                else:
                    missing_mark_dates.append(f"{date.date()}:{code}")
            else:
                marks[code] = float(closes[code])
        if missing_mark_dates:
            raise RuntimeError(f"持仓缺少有效收盘估值：{missing_mark_dates[:5]}")
        equity = account.equity(marks)
        nav_rows.append({
            "date": date,
            "variant": variant_name,
            "cash": account.cash,
            "equity": equity,
            "positions_count": len(account.positions),
        })
        for code, shares in sorted(account.positions.items()):
            position_rows.append({"date": date, "variant": variant_name, "ts_code": code, "shares": int(shares), "equity": equity})
        for event_index, row in record_map.get(date, pd.DataFrame()).iterrows():
            entitlements[int(event_index)] = int(account.positions.get(str(row["ts_code"]), 0))
        for event_index, row in effective_map.get(date, pd.DataFrame()).iterrows():
            if pd.isna(row["record_date"]):
                entitlements[int(event_index)] = int(account.positions.get(str(row["ts_code"]), 0))
            entitled_shares = int(entitlements.pop(int(event_index), 0))
            if entitled_shares <= 0:
                continue
            cash = entitled_shares * float(row["cash_dividend_per_share"])
            bonus_exact = entitled_shares * (
                float(row["bonus_ratio_per_share"]) + float(row["transfer_ratio_per_share"])
            )
            bonus_shares = int(round(bonus_exact))
            if not np.isclose(bonus_exact, bonus_shares, atol=1e-8):
                raise RuntimeError(f"公司行动产生非整数股份：{row['ts_code']} {date.date()}")
            if cash:
                account.receive_dividend(cash)
            if bonus_shares:
                code = str(row["ts_code"])
                account.positions[code] = account.positions.get(code, 0) + bonus_shares
            action_ledger.append({
                "date": date,
                "variant": variant_name,
                "ts_code": str(row["ts_code"]),
                "source_row_index": int(row["source_row_index"]),
                "entitled_shares": entitled_shares,
                "cash_dividend": cash,
                "bonus_shares": bonus_shares,
            })
        if date in full_map:
            pending_target = full_map[date]
            pending_risk_exits.clear()
        elif date in risk_dates:
            day_features = feature_days.get(date, pd.DataFrame())
            for code in list(account.positions):
                if code not in day_features.index:
                    if carry_forward_untradable_marks and not tradable.get(code, False):
                        risk_review_ledger.append({
                            "date": date,
                            "variant": variant_name,
                            "ts_code": code,
                            "risk_review_method": "DEFERRED_UNTRADABLE_NO_POINT_IN_TIME_FEATURES",
                            "tradable": False,
                        })
                        continue
                    raise RuntimeError(f"风险检查缺少持仓点时特征：{date.date()} {code}")
                row = day_features.loc[code]
                if bool(row.get("emergency_exit", False)) or not bool(row.get("base_eligible", False)):
                    pending_risk_exits.add(code)
        previous_marks = marks
    trades = account.trade_records()
    for row in trades:
        row["variant"] = variant_name
    return (
        pd.DataFrame(nav_rows),
        trades,
        action_ledger + position_rows + valuation_ledger + risk_review_ledger,
    )


def _holding_period_metrics(
    nav: pd.DataFrame,
    benchmark_nav: pd.Series,
    trades: list[dict[str, Any]],
    action_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    """仅在组合承担持股经济风险的区间推进收益时钟。"""

    ordered = nav.sort_values("date").copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise").dt.normalize()
    ordered = ordered.drop_duplicates("date", keep="last").set_index("date")
    equity = ordered["equity"].astype(float)
    if len(equity) < 2 or (equity <= 0).any() or not np.isfinite(equity).all():
        raise ValueError("持股期收益计算所需净值无效")
    benchmark = benchmark_nav.reindex(equity.index).ffill().bfill().astype(float)
    if (benchmark <= 0).any() or not np.isfinite(benchmark).all():
        raise ValueError("持股期收益计算所需基准净值无效")

    strategy_returns = equity.pct_change().fillna(0.0)
    benchmark_returns = benchmark.pct_change().fillna(0.0)
    current_position = ordered["positions_count"].astype(int).gt(0)
    previous_position = current_position.shift(1, fill_value=False)
    trade_dates = pd.DatetimeIndex(
        pd.to_datetime([row["date"] for row in trades], errors="raise")
    ).normalize() if trades else pd.DatetimeIndex([])

    action_effect_dates = pd.DatetimeIndex(
        pd.to_datetime([row["date"] for row in action_ledger], errors="raise")
    ).normalize() if action_ledger else pd.DatetimeIndex([])
    action_cash_recognition_dates: set[pd.Timestamp] = set()
    date_positions = {date: index for index, date in enumerate(equity.index)}
    for action_date in action_effect_dates:
        index = date_positions.get(action_date)
        if index is not None and index + 1 < len(equity.index):
            action_cash_recognition_dates.add(equity.index[index + 1])

    economic_event = pd.Series(
        equity.index.isin(trade_dates) | equity.index.isin(action_cash_recognition_dates),
        index=equity.index,
    )
    active = current_position | previous_position | economic_event
    active.iloc[0] = False
    active_strategy = strategy_returns.loc[active]
    active_benchmark = benchmark_returns.loc[active]
    if active_strategy.empty:
        raise ValueError("没有可计算的持股风险区间")

    holding_strategy_nav = (1.0 + active_strategy).cumprod()
    holding_benchmark_nav = (1.0 + active_benchmark).cumprod()
    excess = active_strategy - active_benchmark
    holding_intervals = int(active.sum())
    strategy_total = float(holding_strategy_nav.iloc[-1] - 1.0)
    benchmark_total = float(holding_benchmark_nav.iloc[-1] - 1.0)
    years = holding_intervals / 252.0
    strategy_annualized = float((1.0 + strategy_total) ** (1.0 / years) - 1.0)
    benchmark_annualized = float((1.0 + benchmark_total) ** (1.0 / years) - 1.0)
    information_ratio = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
        if len(excess) > 1 and excess.std(ddof=1) > 0
        else np.nan
    )
    stock_exposure = ((ordered["equity"] - ordered["cash"]) / ordered["equity"]).clip(0.0, 1.0)
    return {
        "performance_clock": "HOLDING_INTERVALS_ONLY",
        "holding_interval_definition": "当日或前一日收盘有持仓，或当日发生交易，或当日确认此前持仓的公司行动现金",
        "holding_intervals": holding_intervals,
        "cash_intervals_excluded": int(len(equity) - 1 - holding_intervals),
        "holding_date_start": active.index[active][0].date().isoformat(),
        "holding_date_end": active.index[active][-1].date().isoformat(),
        "total_return": strategy_total,
        "annualized_return_on_holding_clock": strategy_annualized,
        "annualized_volatility_on_holding_clock": float(active_strategy.std(ddof=1) * np.sqrt(252)),
        "max_drawdown_on_holding_clock": float((holding_strategy_nav / holding_strategy_nav.cummax() - 1.0).min()),
        "h00300_total_return_same_holding_intervals": benchmark_total,
        "h00300_annualized_return_same_holding_clock": benchmark_annualized,
        "h00300_max_drawdown_same_holding_clock": float((holding_benchmark_nav / holding_benchmark_nav.cummax() - 1.0).min()),
        "annualized_arithmetic_excess_on_holding_clock": float(excess.mean() * 252),
        "information_ratio_on_holding_clock": information_ratio,
        "ending_equity": float(equity.iloc[-1]),
        "average_stock_exposure_all_calendar_days": float(stock_exposure.iloc[1:].mean()),
        "average_stock_exposure_holding_intervals": float(stock_exposure.loc[active].mean()),
    }


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    signal_receipt = _load_json(project_path(contract["sources"]["signal_receipt"]))
    action_report = _load_json(project_path(contract["sources"]["corporate_action_report"]))
    if signal_receipt.get("portfolio_return_values_read") is not False:
        raise RuntimeError("信号冻结回执已提前读取组合收益")
    if action_report.get("status") != "READY_FOR_RETURN_READ" or action_report.get("portfolio_return_values_read") is not False:
        raise RuntimeError(f"公司行动门禁未通过：{action_report.get('status')}")
    events = pd.read_parquet(project_path(contract["sources"]["signal_events"]))
    selected = pd.read_parquet(project_path(contract["sources"]["selected_symbols"]))
    corporate_actions = pd.read_parquet(project_path(contract["sources"]["corporate_action_events"]))
    events["signal_date"] = pd.to_datetime(events["signal_date"], errors="raise").dt.normalize()
    selected["signal_date"] = pd.to_datetime(selected["signal_date"], errors="raise").dt.normalize()
    calendar_frame = pd.read_parquet(project_path(contract["sources"]["trading_calendar"]))
    calendar_dates = pd.DatetimeIndex(
        pd.to_datetime(calendar_frame.loc[calendar_frame["is_open"].astype(bool), "date"], errors="raise").dt.normalize().unique()
    ).sort_values()
    signal_config_path = project_path(
        contract.get(
            "signal_config",
            "config/a_share_hs_concentrated_low_risk_trend_v1_4_1_signal_freeze_calendar_corrected.yaml",
        )
    )
    features, feature_calendar, market_receipts = V141.V14.build_signal_panel(
        yaml.safe_load(signal_config_path.read_text(encoding="utf-8"))
    )
    if not feature_calendar.equals(calendar_dates):
        raise RuntimeError("回测使用的交易日历与信号冻结日历不一致")
    top3_codes = set()
    for value in events["top20"]:
        top3_codes.update(_parse_top(value))
    signal_codes = set(selected["ts_code"].astype(str))
    execution_codes = top3_codes | signal_codes
    benchmark_start = events["signal_date"].min()
    benchmark_end = pd.Timestamp("2026-08-14")
    sim_dates = calendar_dates[(calendar_dates >= benchmark_start) & (calendar_dates <= benchmark_end)]
    benchmark_navs, benchmark_files = _benchmark_navs(features, sim_dates, float(contract["simulation"]["initial_capital_cny"]), contract)
    features = features.loc[features["ts_code"].isin(execution_codes)].copy()
    del calendar_frame
    gc.collect()
    execution_market, market_files = _load_execution_market(project_path(contract["sources"]["daily_market_manifest"]), execution_codes)
    all_nav: list[pd.DataFrame] = []
    all_trades: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    variant_metrics: dict[str, Any] = {}
    for variant_name, maximum_names in contract["simulation"]["variants"].items():
        nav, trades, ledger = _simulate_variant(
            variant_name=variant_name,
            maximum_names=int(maximum_names),
            events=events,
            features=features,
            market=execution_market,
            corporate_actions=corporate_actions,
            calendar_dates=calendar_dates,
            capital=float(contract["simulation"]["initial_capital_cny"]),
            review_frequency=int(contract["simulation"]["risk_review_frequency_trading_days"]),
            allocation_mode=str(contract["simulation"].get("allocation_mode", "FROZEN_CASH_LADDER")),
        )
        all_nav.append(nav)
        all_trades.extend(trades)
        all_actions.extend([row for row in ledger if "entitled_shares" in row])
        series = nav.set_index("date")["equity"]
        performance_clock = str(contract["simulation"].get("primary_performance_clock", "FULL_CALENDAR"))
        applied_actions = [row for row in ledger if "entitled_shares" in row]
        if performance_clock == "HOLDING_INTERVALS_ONLY":
            metrics = _holding_period_metrics(
                nav,
                benchmark_navs["H00300_TOTAL_RETURN"],
                trades,
                applied_actions,
            )
            trailing_years = contract["simulation"].get("focus_trailing_years")
            if trailing_years is not None:
                trailing_years = int(trailing_years)
                if trailing_years < 1:
                    raise ValueError("重点回看年数必须为正整数")
                ordered_dates = pd.DatetimeIndex(pd.to_datetime(nav["date"], errors="raise")).normalize().sort_values()
                cutoff = ordered_dates[-1] - pd.DateOffset(years=trailing_years)
                first_in_window = int(ordered_dates.searchsorted(cutoff, side="left"))
                predecessor = max(0, first_in_window - 1)
                subperiod_nav = nav.loc[pd.to_datetime(nav["date"]).dt.normalize() >= ordered_dates[predecessor]].copy()
                subperiod_trades = [
                    row for row in trades if pd.Timestamp(row["date"]).normalize() >= cutoff
                ]
                subperiod_actions = [
                    row for row in applied_actions if pd.Timestamp(row["date"]).normalize() >= ordered_dates[predecessor]
                ]
                subperiod = _holding_period_metrics(
                    subperiod_nav,
                    benchmark_navs["H00300_TOTAL_RETURN"],
                    subperiod_trades,
                    subperiod_actions,
                )
                subperiod["requested_trailing_years"] = trailing_years
                subperiod["calendar_window_start"] = cutoff.date().isoformat()
                subperiod["calendar_window_end"] = ordered_dates[-1].date().isoformat()
                metrics["trailing_year_focus"] = subperiod
        elif performance_clock == "FULL_CALENDAR":
            metrics = ENGINE.annualized_metrics(series, benchmark_navs["H00300_TOTAL_RETURN"])
            metrics["ending_equity"] = float(series.iloc[-1])
            metrics["total_return"] = float(series.iloc[-1] / series.iloc[0] - 1.0)
        else:
            raise ValueError(f"未知收益时钟：{performance_clock}")
        metrics["trade_count"] = int(len(trades))
        metrics["corporate_action_count"] = int(sum(row.get("variant") == variant_name for row in all_actions))
        variant_metrics[variant_name] = metrics
    nav_frame = pd.concat(all_nav, ignore_index=True)
    trades_frame = pd.DataFrame(all_trades)
    actions_frame = pd.DataFrame(all_actions)
    root = project_path(contract["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    nav_path = project_path(contract["outputs"]["nav_daily"])
    trades_path = project_path(contract["outputs"]["trades"])
    actions_path = project_path(contract["outputs"]["corporate_action_ledger"])
    nav_frame.to_parquet(nav_path, index=False)
    trades_frame.to_parquet(trades_path, index=False)
    actions_frame.to_parquet(actions_path, index=False)
    report = {
        "schema_version": "A_SHARE_HS_BACKTEST_V1",
        "contract_id": contract["contract_id"],
        "signal_contract_id": contract["signal_contract_id"],
        "corporate_action_contract_id": contract["corporate_action_contract_id"],
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": str(contract.get("result_status", "BACKTEST_COMPLETED_NO_GATE_DECISION")),
        "view_status": "POST_RESULT_HISTORICAL_DIAGNOSTIC" if contract.get("post_result_diagnostic", False) else "HISTORICAL_VIEW",
        "post_result_diagnostic": bool(contract.get("post_result_diagnostic", False)),
        "parent_terminal_contract_id": contract.get("parent_terminal_contract_id"),
        "portfolio_return_values_read": True,
        "return_files_opened": sorted(set(market_files + benchmark_files)),
        "signals_generated": False,
        "simulated_positions_generated": True,
        "simulated_trades_generated": True,
        "orders_generated": False,
        "broker_connection": False,
        "date_start": sim_dates[0].date().isoformat(),
        "date_end": sim_dates[-1].date().isoformat(),
        "signal_dates": int(len(events)),
        "zero_candidate_signal_dates": int(events["candidate_count"].eq(0).sum()),
        "selected_symbol_count": int(selected["ts_code"].nunique()),
        "variants": variant_metrics,
        "benchmarks": {
            name: {"start": float(series.iloc[0]), "end": float(series.iloc[-1])}
            for name, series in benchmark_navs.items()
        },
        "simulation": {
            "initial_capital_cny": float(contract["simulation"]["initial_capital_cny"]),
            "allocation_mode": str(contract["simulation"].get("allocation_mode", "FROZEN_CASH_LADDER")),
            "primary_performance_clock": str(contract["simulation"].get("primary_performance_clock", "FULL_CALENDAR")),
        },
        "gate_status": str(contract.get("gate_status", "NOT_EVALUATED_REQUIRED_OFFSET_AND_ROBUSTNESS_DIAGNOSTICS_PENDING")),
        "artifacts": {
            "nav_daily": {"path": nav_path.relative_to(ROOT).as_posix()},
            "trades": {"path": trades_path.relative_to(ROOT).as_posix()},
            "corporate_action_ledger": {"path": actions_path.relative_to(ROOT).as_posix()},
        },
        "conclusion": str(contract.get("conclusion", "已完成固定信号的历史成本回测；不根据收益选择持股数，正式 Shadow 门禁仍待预注册偏移和稳健性诊断。")),
    }
    report_path = project_path(contract["outputs"]["report_json"])
    report_md_path = project_path(contract["outputs"]["report_markdown"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# 沪深 A 股 {contract['contract_id']} 历史回测", "",
        f"- 状态：`{report['status']}`", f"- 回测区间：{report['date_start']} 至 {report['date_end']}",
        f"- 信号日：{report['signal_dates']}（零候选现金日 {report['zero_candidate_signal_dates']}）",
        f"- 主模型：TOP3；收益时钟：{report['simulation']['primary_performance_clock']}", "- 组合收益读取：是（仅历史回测）", "- 实盘订单：否", "",
        "## 结果摘要", "",
    ]
    for name, metrics in variant_metrics.items():
        drawdown = metrics.get("max_drawdown", metrics.get("max_drawdown_on_holding_clock"))
        lines.append(f"- {name}：期末权益 {metrics['ending_equity']:.2f} 元，收益 {metrics['total_return']:.4%}，收益时钟最大回撤 {drawdown:.4%}，交易腿 {metrics['trade_count']} 条")
        focus = metrics.get("trailing_year_focus")
        if focus:
            lines.extend([
                "",
                f"### 最近{focus['requested_trailing_years']}年重点结果",
                "",
                f"- 日历窗口：{focus['calendar_window_start']} 至 {focus['calendar_window_end']}",
                f"- 实际持股风险区间：{focus['holding_intervals']}个；排除纯现金区间：{focus['cash_intervals_excluded']}个",
                f"- 持股期累计收益：{focus['total_return']:.4%}；持股时钟年化：{focus['annualized_return_on_holding_clock']:.4%}",
                f"- 同期同持股日H00300：{focus['h00300_total_return_same_holding_intervals']:.4%}；年化：{focus['h00300_annualized_return_same_holding_clock']:.4%}",
                f"- 持股时钟信息比率：{focus['information_ratio_on_holding_clock']:.4f}",
                f"- 最大回撤：策略 {focus['max_drawdown_on_holding_clock']:.4%}，H00300 {focus['h00300_max_drawdown_same_holding_clock']:.4%}",
                f"- 持股区间平均股票暴露：{focus['average_stock_exposure_holding_intervals']:.4%}",
            ])
    lines.extend(["", report["conclusion"], ""])
    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"状态": report["status"], "区间": [report["date_start"], report["date_end"]], "主模型期末权益": variant_metrics["TOP3"]["ending_equity"], "组合收益读取": True, "实盘订单": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
