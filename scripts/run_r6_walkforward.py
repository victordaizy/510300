"""运行R6预注册候选、统一历史伪样本外评价和小账户压力测试。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis, norm, skew


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.defensive_valuation_timing_engine import (
    build_model_positions,
    build_timing_features,
    schedule_asymmetric_execution,
)
from backtest.engine import BacktestCosts, run_long_cash_backtest
from backtest.valuation_fvg_engine import build_valuation_signals
from src.strategies.r6_common import quantize_targets
from src.strategies.r6_r5_overlay import build_r5_overlay_positions
from src.strategies.r6_trend_vol_regime import build_trend_vol_positions
from src.strategies.r6_vol_target import build_vol_target_positions


MATRIX_FILE = ROOT / "config" / "r6_experiment_matrix.yaml"
R5_CONFIG_FILE = ROOT / "config" / "round5_defensive_valuation_timing.yaml"
REPORT_JSON = ROOT / "reports" / "backtest" / "r6_model_selection.json"
REPORT_MD = ROOT / "reports" / "backtest" / "r6_model_selection.md"
OUTPUT_DIR = ROOT / "data" / "processed" / "r6_walkforward"
FORWARD_STATUS = ROOT / "paper" / "r6_forward" / "status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(primary: str, fallback: str | None = None) -> Path:
    path = ROOT / primary
    if path.exists():
        return path
    if fallback is not None and (ROOT / fallback).exists():
        return ROOT / fallback
    raise FileNotFoundError(f"缺少R6输入：{primary}")


def _load_inputs(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    long_panel = config["data"]["long_panel"]
    paths = {
        "etf": _resolve_path(long_panel["etf"], long_panel["fallback_etf"]),
        "index": _resolve_path(long_panel["index"], long_panel["fallback_index"]),
        "h00300": _resolve_path(
            long_panel["total_return_benchmark"],
            long_panel["fallback_total_return_benchmark"],
        ),
        "valuation": _resolve_path(config["data"]["common_panel"]["valuation"]),
        "dividends": _resolve_path(config["data"]["dividends"]),
    }
    frames = {
        "etf": pd.read_parquet(paths["etf"]),
        "index": pd.read_parquet(paths["index"]),
        "h00300": pd.read_parquet(paths["h00300"]),
        "valuation": pd.read_parquet(paths["valuation"]),
        "dividends": pd.read_csv(paths["dividends"]),
    }
    for name in ("etf", "index", "h00300", "valuation"):
        frames[name]["date"] = pd.to_datetime(frames[name]["date"])
        frames[name] = frames[name].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    required_etf = {"date", "open", "high", "low", "close"}
    if missing := required_etf - set(frames["etf"].columns):
        raise ValueError(f"510300长面板缺少字段：{sorted(missing)}")
    return frames, paths


def _build_r5_positions(
    index: pd.DataFrame,
    valuation_raw: pd.DataFrame,
    r5_config: dict[str, Any],
) -> pd.DataFrame:
    timing = build_timing_features(index, r5_config)
    valuation = build_valuation_signals(valuation_raw, r5_config["valuation"])
    width = (valuation["fair_high_12m"] - valuation["fair_low_12m"]).replace(0.0, np.nan)
    valuation["raw_continuous_position"] = (
        (valuation["fair_high_12m"] - valuation["index_close"]) / width
    ).clip(0.0, 1.0)
    model = build_model_positions(timing, r5_config, valuation)
    scheduled = schedule_asymmetric_execution(model, r5_config)
    return scheduled[["date", "target_position"]].rename(
        columns={"target_position": "r5_target_position"}
    )


def _candidate_targets(
    candidate: dict[str, Any],
    etf: pd.DataFrame,
    r5_positions: pd.DataFrame,
    execution: dict[str, Any],
) -> pd.DataFrame:
    common = {
        "review_every": int(execution["review_every_trading_days"]),
        "maximum_increase": float(execution["maximum_position_increase_per_review"]),
    }
    family = candidate["family"]
    if family == "A_VOL_TARGET":
        return build_vol_target_positions(
            etf,
            target_volatility=float(candidate["target_volatility"]),
            estimator=str(candidate["estimator"]),
            **common,
        )
    if family == "B_TREND_VOL":
        return build_trend_vol_positions(
            etf,
            mapping=str(candidate["mapping"]),
            normalization_window=int(candidate["normalization_window"]),
            **common,
        )
    if family == "C_R5_OVERLAY":
        return build_r5_overlay_positions(
            etf,
            r5_positions,
            valuation_mode=str(candidate["valuation_mode"]),
            multiplier_variant=str(candidate["multiplier_variant"]),
            normalization_window=756,
            **common,
        )
    raise ValueError(f"未知R6候选族：{family}")


def _grid_targets(targets: pd.DataFrame, step: float) -> pd.DataFrame:
    result = targets.copy()
    result["continuous_target_position"] = result["target_position"]
    result["target_position"] = quantize_targets(result["target_position"], step)
    return result


def _buy_hold_targets(calendar: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame({"date": pd.to_datetime(calendar), "target_position": 1.0})
    result["trade_allowed"] = False
    result.loc[result.index[0], "trade_allowed"] = True
    result["risk_off_override"] = False
    result["signal_reason"] = "买入持有首次建仓"
    return result


def _costs(execution: dict[str, Any], multiplier: float = 1.0) -> BacktestCosts:
    return BacktestCosts(
        commission_rate=float(execution["commission_rate"]) * multiplier,
        minimum_commission_cny=float(execution["minimum_commission_cny"]) * multiplier,
        stamp_duty_rate=0.0,
        slippage_bps=float(execution["slippage_bps_per_leg"]) * multiplier,
        lot_size=int(execution["lot_size"]),
        cash_annual_rate=float(execution["cash_annual_rate"]),
    )


def _run_variant(
    *,
    etf: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    initial_cash: float,
    costs: BacktestCosts,
    minimum_trade_shares: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = targets.loc[targets["target_position"].notna()].copy()
    if ready.empty:
        raise ValueError("候选没有可执行目标")
    start = pd.Timestamp(ready["date"].min())
    end = min(pd.Timestamp(ready["date"].max()), pd.Timestamp(etf["date"].max()))
    market_calendar = etf.loc[pd.to_datetime(etf["date"]).between(start, end), "date"]
    aligned = pd.DataFrame({"date": pd.to_datetime(market_calendar)}).merge(
        ready,
        on="date",
        how="left",
        validate="one_to_one",
    )
    required = ["target_position", "trade_allowed", "risk_off_override"]
    if aligned[required].isna().any().any():
        raise ValueError("候选目标未完整覆盖ETF日历")
    return run_long_cash_backtest(
        etf,
        dividends,
        aligned,
        initial_cash,
        costs,
        start,
        end,
        minimum_trade_shares=minimum_trade_shares,
    )


def _benchmark_returns(
    h00300: pd.DataFrame,
    calendar: pd.Series,
) -> pd.Series:
    close = h00300.set_index("date")["close"].astype(float)
    aligned = close.reindex(pd.DatetimeIndex(calendar)).ffill(limit=2)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"H00300无法覆盖策略日历：{missing[:5]}")
    return aligned.pct_change(fill_method=None).fillna(0.0)


def _longest_underwater(returns: pd.Series) -> int:
    equity = (1.0 + returns).cumprod()
    underwater = equity.lt(equity.cummax())
    longest = 0
    current = 0
    for value in underwater:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _deflated_sharpe_probability(returns: pd.Series, trials: int) -> float | None:
    clean = returns.dropna()
    if len(clean) < 60 or clean.std(ddof=1) <= 0:
        return None
    daily_sr = clean.mean() / clean.std(ddof=1)
    annual_sr = daily_sr * math.sqrt(242)
    sr_std = math.sqrt((1.0 + 0.5 * annual_sr**2) / max(len(clean) - 1, 1))
    gamma = 0.5772156649
    expected_max = sr_std * (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / max(trials, 2))
        + gamma * norm.ppf(1.0 - 1.0 / (max(trials, 2) * math.e))
    )
    denominator = math.sqrt(
        max(
            1.0
            - skew(clean, bias=False) * daily_sr
            + ((kurtosis(clean, fisher=False, bias=False) - 1.0) / 4.0) * daily_sr**2,
            1e-12,
        )
    )
    statistic = (daily_sr - expected_max / math.sqrt(242)) * math.sqrt(len(clean) - 1) / denominator
    return float(norm.cdf(statistic))


def _period_metrics(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    etf_benchmark: pd.DataFrame,
    h00300_returns: pd.Series,
    start: str,
    end: str | None,
    trials: int,
) -> dict[str, Any]:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end else pd.Timestamp(ledger["date"].max())
    strategy = ledger.loc[pd.to_datetime(ledger["date"]).between(start_date, end_date)].copy()
    etf = etf_benchmark.loc[
        pd.to_datetime(etf_benchmark["date"]).between(start_date, end_date)
    ].copy()
    if len(strategy) < 20 or len(etf) != len(strategy):
        return {"status": "INSUFFICIENT_DATA", "observations": int(len(strategy))}
    dates = pd.DatetimeIndex(strategy["date"])
    strategy_returns = strategy.set_index("date")["daily_return"].reindex(dates)
    etf_returns = etf.set_index("date")["daily_return"].reindex(dates)
    total_returns = h00300_returns.reindex(dates)
    if total_returns.isna().any():
        raise ValueError("H00300区间收益存在缺口")
    years = len(strategy_returns) / 242.0

    def annualized(values: pd.Series) -> tuple[float, float, float, float, float]:
        total = float((1.0 + values).prod() - 1.0)
        cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0
        volatility = float(values.std(ddof=1) * math.sqrt(242))
        sharpe = float(values.mean() * 242 / volatility) if volatility > 0 else math.nan
        downside = float(values.where(values.lt(0.0), 0.0).std(ddof=1) * math.sqrt(242))
        sortino = float(values.mean() * 242 / downside) if downside > 0 else math.nan
        return total, cagr, volatility, sharpe, sortino

    total_return, cagr, volatility, sharpe, sortino = annualized(strategy_returns)
    _, etf_cagr, _, etf_sharpe, _ = annualized(etf_returns)
    _, total_cagr, _, total_sharpe, _ = annualized(total_returns)
    cumulative = (1.0 + strategy_returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    excess_etf = strategy_returns - etf_returns
    excess_total = strategy_returns - total_returns
    tracking_error = float(excess_etf.std(ddof=1) * math.sqrt(242))
    information_ratio = (
        float(excess_etf.mean() * 242 / tracking_error) if tracking_error > 0 else math.nan
    )
    rolling_strategy = (1.0 + strategy_returns).rolling(242).apply(np.prod, raw=True) - 1.0
    rolling_etf = (1.0 + etf_returns).rolling(242).apply(np.prod, raw=True) - 1.0
    rolling_excess = (rolling_strategy - rolling_etf).dropna()
    trade_slice = trades.loc[
        pd.to_datetime(trades["date"]).between(start_date, end_date)
    ] if not trades.empty else trades
    notional = (
        float((trade_slice["quantity"] * trade_slice["execution_price"]).sum())
        if not trade_slice.empty
        else 0.0
    )
    explicit_cost = (
        float((trade_slice["commission"] + trade_slice["stamp_duty"]).sum())
        if not trade_slice.empty
        else 0.0
    )
    slippage_cost = (
        float(
            (
                (trade_slice["execution_price"] - trade_slice["open_price"]).abs()
                * trade_slice["quantity"]
            ).sum()
        )
        if not trade_slice.empty
        else 0.0
    )
    yearly = pd.DataFrame(
        {"strategy": strategy_returns, "etf": etf_returns, "h00300": total_returns}
    ).groupby(lambda value: value.year).apply(
        lambda group: pd.Series(
            {
                "strategy": (1.0 + group["strategy"]).prod() - 1.0,
                "etf": (1.0 + group["etf"]).prod() - 1.0,
                "h00300": (1.0 + group["h00300"]).prod() - 1.0,
            }
        ),
        include_groups=False,
    )
    yearly["excess_vs_etf"] = yearly["strategy"] - yearly["etf"]
    positive_years = yearly["excess_vs_etf"].clip(lower=0.0)
    concentration = (
        float(positive_years.max() / positive_years.sum())
        if positive_years.sum() > 0
        else math.nan
    )
    subperiod_excess: list[float] = []
    for positions in np.array_split(np.arange(len(strategy_returns)), 3):
        if len(positions) == 0:
            continue
        strategy_part = float((1.0 + strategy_returns.iloc[positions]).prod() - 1.0)
        etf_part = float((1.0 + etf_returns.iloc[positions]).prod() - 1.0)
        subperiod_excess.append(strategy_part - etf_part)
    return {
        "status": "SUCCESS",
        "start_date": str(dates.min().date()),
        "end_date": str(dates.max().date()),
        "observations": int(len(strategy_returns)),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else math.nan,
        "annualized_excess_vs_etf": cagr - etf_cagr,
        "annualized_excess_vs_h00300": cagr - total_cagr,
        "etf_buy_hold_sharpe": etf_sharpe,
        "h00300_sharpe": total_sharpe,
        "information_ratio_vs_etf": information_ratio,
        "average_position": float(strategy["actual_position"].mean()),
        "annual_turnover": notional / float(strategy["equity"].mean()) / years,
        "trade_count": int(len(trade_slice)),
        "explicit_cost_cny": explicit_cost,
        "slippage_cost_cny": slippage_cost,
        "longest_underwater_trading_days": _longest_underwater(strategy_returns),
        "rolling_242d_excess": {
            "observations": int(len(rolling_excess)),
            "median": float(rolling_excess.median()) if len(rolling_excess) else math.nan,
            "minimum": float(rolling_excess.min()) if len(rolling_excess) else math.nan,
            "maximum": float(rolling_excess.max()) if len(rolling_excess) else math.nan,
            "positive_ratio": float(rolling_excess.gt(0.0).mean()) if len(rolling_excess) else math.nan,
        },
        "positive_subperiod_ratio": float(np.mean(np.array(subperiod_excess) > 0.0)),
        "single_year_positive_excess_concentration": concentration,
        "yearly": {
            str(int(year)): {key: float(value) for key, value in row.items()}
            for year, row in yearly.iterrows()
        },
        "deflated_sharpe_probability": _deflated_sharpe_probability(
            strategy_returns, trials
        ),
    }


def _gate_result(
    metrics: dict[str, Any],
    small: dict[str, Any],
    stress: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, bool]:
    if metrics.get("status") != "SUCCESS":
        return {"sufficient_data": False}
    rolling = metrics["rolling_242d_excess"]
    drawdown_limit = abs(metrics["max_drawdown"]) <= abs(metrics.get("etf_max_drawdown", math.nan)) * float(
        gates["drawdown_ratio_to_buy_hold_maximum"]
    )
    return {
        "sufficient_data": True,
        "positive_excess_vs_etf": metrics["annualized_excess_vs_etf"]
        > float(gates["excess_vs_etf_buy_hold_minimum"]),
        "positive_excess_vs_h00300": metrics["annualized_excess_vs_h00300"]
        > float(gates["excess_vs_h00300_minimum"]),
        "sharpe_minimum": metrics["sharpe"] >= float(gates["sharpe_minimum"]),
        "sharpe_improvement": metrics["sharpe"] - metrics["etf_buy_hold_sharpe"]
        >= float(gates["sharpe_improvement_minimum"]),
        "drawdown_reduction": bool(drawdown_limit),
        "rolling_median": rolling["median"] > float(
            gates["rolling_242d_excess_median_minimum"]
        ),
        "rolling_positive_ratio": rolling["positive_ratio"]
        >= float(gates["rolling_242d_positive_ratio_minimum"]),
        "subperiod_stability": metrics["positive_subperiod_ratio"]
        >= float(gates["positive_subperiod_ratio_minimum"]),
        "year_concentration": metrics["single_year_positive_excess_concentration"]
        <= float(gates["single_year_positive_excess_concentration_maximum"]),
        "small_account_positive": small["annualized_excess_vs_etf"] >= 0.0,
        "double_cost_positive": stress["annualized_excess_vs_etf"]
        >= float(gates["stress_cost_excess_minimum"]),
        "deflated_sharpe": (metrics["deflated_sharpe_probability"] or 0.0)
        >= float(gates["deflated_sharpe_probability_minimum"]),
    }


def _attach_benchmark_drawdown(metrics: dict[str, Any], benchmark: dict[str, Any]) -> None:
    if metrics.get("status") == "SUCCESS" and benchmark.get("status") == "SUCCESS":
        metrics["etf_max_drawdown"] = benchmark["max_drawdown"]


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# R6预注册模型统一选择报告",
        "",
        f"> 证据标签：`{payload['evidence_label']}`。当前没有严格锁定样本外，结果不得解释为可实盘冠军。",
        "",
        "| 排名 | 候选 | 族 | 连续验证Sharpe | 连续伪OOS Sharpe | 10%档对ETF超额 | 连续最大回撤 | 25%小账户超额 | 历史门槛 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(payload["ranking"], start=1):
        validation = row["continuous_periods"]["validation"]
        continuous_pseudo = row["continuous_periods"]["historical_pseudo_oos"]
        pseudo = row["periods"]["historical_pseudo_oos"]
        small = row["small_account_historical_pseudo_oos"]
        pct = lambda value: "—" if value is None else f"{value:.2%}"
        num = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {rank} | {row['id']} | {row['family']} | {num(validation.get('sharpe'))} | "
            f"{num(continuous_pseudo.get('sharpe'))} | {pct(pseudo.get('annualized_excess_vs_etf'))} | "
            f"{pct(continuous_pseudo.get('max_drawdown'))} | {pct(small.get('annualized_excess_vs_etf'))} | "
            f"{'通过' if row['historical_gate_pass'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            f"- 历史验证排序领先者：`{payload['historical_validation_leader']}`。",
            f"- 完整历史门槛通过数：{payload['historical_gate_pass_count']}。",
            "- `R6_FORWARD_1`：未激活。原因是当前数据均已被研究查看，`true_locked_oos_start`尚未设置。",
            "- 即使历史门槛通过，也只能形成待冻结候选，不能自动下单或映射真实仓位。",
            "",
            "## 可复现性",
            "",
            f"- 实验矩阵SHA-256：`{payload['hashes']['r6_experiment_matrix.yaml']}`",
            f"- 候选数：{payload['candidate_count']}（冻结后禁止追加）。",
            "- 所有信号在收盘形成，最早于下一交易日开盘执行。",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(MATRIX_FILE.read_text(encoding="utf-8"))
    if len(config["candidates"]) != int(config["research"]["candidate_count"]):
        raise ValueError("实验矩阵候选数与冻结声明不一致")
    frames, paths = _load_inputs(config)
    etf = frames["etf"]
    execution = config["execution"]
    r5_config = yaml.safe_load(R5_CONFIG_FILE.read_text(encoding="utf-8"))
    r5_positions = _build_r5_positions(frames["index"], frames["valuation"], r5_config)
    base_costs = _costs(execution)
    stress_costs = _costs(execution, float(execution["stress_cost_multiplier"]))
    research_cash = float(execution["continuous_research_account_cny"])
    small_cash = float(execution["small_account_cny"])
    candidates_payload: list[dict[str, Any]] = []
    split_config = config["splits"]
    split_names = ("development", "validation", "historical_pseudo_oos")

    for candidate in config["candidates"]:
        targets = _candidate_targets(candidate, etf, r5_positions, execution)
        grid10 = _grid_targets(targets, float(execution["research_grid_step"]))
        grid25 = _grid_targets(targets, float(execution["small_account_grid_step"]))
        continuous_ledger, continuous_trades = _run_variant(
            etf=etf,
            dividends=frames["dividends"],
            targets=targets,
            initial_cash=research_cash,
            costs=base_costs,
        )
        ledger10, trades10 = _run_variant(
            etf=etf,
            dividends=frames["dividends"],
            targets=grid10,
            initial_cash=research_cash,
            costs=base_costs,
        )
        small_ledger, small_trades = _run_variant(
            etf=etf,
            dividends=frames["dividends"],
            targets=grid25,
            initial_cash=small_cash,
            costs=base_costs,
            minimum_trade_shares=int(execution["small_account_minimum_normal_trade_shares"]),
        )
        stress_ledger, stress_trades = _run_variant(
            etf=etf,
            dividends=frames["dividends"],
            targets=grid25,
            initial_cash=small_cash,
            costs=stress_costs,
            minimum_trade_shares=int(execution["small_account_minimum_normal_trade_shares"]),
        )
        start = str(pd.Timestamp(ledger10["date"].min()).date())
        end = str(pd.Timestamp(ledger10["date"].max()).date())
        calendar = ledger10["date"]
        benchmark_ledger, benchmark_trades = run_long_cash_backtest(
            etf,
            frames["dividends"],
            _buy_hold_targets(calendar),
            research_cash,
            base_costs,
            start,
            end,
        )
        small_benchmark, small_benchmark_trades = run_long_cash_backtest(
            etf,
            frames["dividends"],
            _buy_hold_targets(small_ledger["date"]),
            small_cash,
            base_costs,
            str(pd.Timestamp(small_ledger["date"].min()).date()),
            str(pd.Timestamp(small_ledger["date"].max()).date()),
        )
        stress_benchmark, stress_benchmark_trades = run_long_cash_backtest(
            etf,
            frames["dividends"],
            _buy_hold_targets(stress_ledger["date"]),
            small_cash,
            stress_costs,
            str(pd.Timestamp(stress_ledger["date"].min()).date()),
            str(pd.Timestamp(stress_ledger["date"].max()).date()),
        )
        h00300_returns = _benchmark_returns(frames["h00300"], calendar)
        small_h00300 = _benchmark_returns(frames["h00300"], small_ledger["date"])
        periods: dict[str, Any] = {}
        continuous_periods: dict[str, Any] = {}
        for split_name in split_names:
            split_start, split_end = split_config[split_name]
            period = _period_metrics(
                ledger10,
                trades10,
                benchmark_ledger,
                h00300_returns,
                split_start,
                split_end,
                int(config["research"]["candidate_count"]),
            )
            benchmark_period = _period_metrics(
                benchmark_ledger,
                benchmark_trades,
                benchmark_ledger,
                h00300_returns,
                split_start,
                split_end,
                1,
            )
            _attach_benchmark_drawdown(period, benchmark_period)
            periods[split_name] = period
            continuous_period = _period_metrics(
                continuous_ledger,
                continuous_trades,
                benchmark_ledger,
                h00300_returns,
                split_start,
                split_end,
                int(config["research"]["candidate_count"]),
            )
            _attach_benchmark_drawdown(continuous_period, benchmark_period)
            continuous_periods[split_name] = continuous_period
        anchored_walkforward: dict[str, Any] = {}
        first_walkforward_year = pd.Timestamp(split_config["validation"][0]).year
        last_walkforward_year = pd.Timestamp(ledger10["date"].max()).year
        for year in range(first_walkforward_year, last_walkforward_year + 1):
            year_metrics = _period_metrics(
                ledger10,
                trades10,
                benchmark_ledger,
                h00300_returns,
                f"{year}-01-01",
                f"{year}-12-31",
                int(config["research"]["candidate_count"]),
            )
            year_benchmark = _period_metrics(
                benchmark_ledger,
                benchmark_trades,
                benchmark_ledger,
                h00300_returns,
                f"{year}-01-01",
                f"{year}-12-31",
                1,
            )
            _attach_benchmark_drawdown(year_metrics, year_benchmark)
            anchored_walkforward[str(year)] = year_metrics
        pseudo_start, pseudo_end = split_config["historical_pseudo_oos"]
        small_metrics = _period_metrics(
            small_ledger,
            small_trades,
            small_benchmark,
            small_h00300,
            pseudo_start,
            pseudo_end,
            int(config["research"]["candidate_count"]),
        )
        small_benchmark_metrics = _period_metrics(
            small_benchmark,
            small_benchmark_trades,
            small_benchmark,
            small_h00300,
            pseudo_start,
            pseudo_end,
            1,
        )
        _attach_benchmark_drawdown(small_metrics, small_benchmark_metrics)
        stress_h00300 = _benchmark_returns(frames["h00300"], stress_ledger["date"])
        stress_metrics = _period_metrics(
            stress_ledger,
            stress_trades,
            stress_benchmark,
            stress_h00300,
            pseudo_start,
            pseudo_end,
            int(config["research"]["candidate_count"]),
        )
        stress_benchmark_metrics = _period_metrics(
            stress_benchmark,
            stress_benchmark_trades,
            stress_benchmark,
            stress_h00300,
            pseudo_start,
            pseudo_end,
            1,
        )
        _attach_benchmark_drawdown(stress_metrics, stress_benchmark_metrics)
        gate_checks = _gate_result(
            continuous_periods["historical_pseudo_oos"],
            small_metrics,
            stress_metrics,
            config["selection"]["gates"],
        )
        gate_checks["research_grid_positive"] = (
            periods["historical_pseudo_oos"].get("annualized_excess_vs_etf", -math.inf)
            >= 0.0
            and periods["historical_pseudo_oos"].get(
                "annualized_excess_vs_h00300", -math.inf
            )
            >= 0.0
        )
        candidate_dir = OUTPUT_DIR / candidate["id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        targets.to_parquet(candidate_dir / "continuous_targets.parquet", index=False)
        grid10.to_parquet(candidate_dir / "research_grid_targets.parquet", index=False)
        grid25.to_parquet(candidate_dir / "small_account_targets.parquet", index=False)
        candidates_payload.append(
            {
                **candidate,
                "continuous_periods": continuous_periods,
                "periods": periods,
                "anchored_walkforward": anchored_walkforward,
                "small_account_historical_pseudo_oos": small_metrics,
                "double_cost_historical_pseudo_oos": stress_metrics,
                "gate_checks": gate_checks,
                "historical_gate_pass": all(gate_checks.values()),
            }
        )

    group_positive: dict[str, float] = {}
    for group in sorted({item["neighbor_group"] for item in candidates_payload}):
        members = [item for item in candidates_payload if item["neighbor_group"] == group]
        group_positive[group] = float(
            np.mean(
                [
                    item["periods"]["historical_pseudo_oos"].get(
                        "annualized_excess_vs_etf", -math.inf
                    )
                    > 0.0
                    for item in members
                ]
            )
        )
    for item in candidates_payload:
        stability = group_positive[item["neighbor_group"]] >= 0.50
        item["gate_checks"]["neighbor_stability"] = stability
        item["historical_gate_pass"] = all(item["gate_checks"].values())

    def ranking_key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = item["continuous_periods"]["validation"]
        if metrics.get("status") != "SUCCESS":
            return (math.inf, math.inf, math.inf, math.inf, math.inf)
        return (
            -float(metrics["sharpe"]),
            abs(float(metrics["max_drawdown"])),
            -float(metrics["rolling_242d_excess"]["median"]),
            -min(
                float(metrics["annualized_excess_vs_etf"]),
                float(metrics["annualized_excess_vs_h00300"]),
            ),
            float(metrics["annual_turnover"]),
        )

    ranking = sorted(candidates_payload, key=ranking_key)
    hashes = {
        MATRIX_FILE.name: _sha256(MATRIX_FILE),
        R5_CONFIG_FILE.name: _sha256(R5_CONFIG_FILE),
        **{path.name: _sha256(path) for path in paths.values()},
    }
    payload = {
        "strategy_id": config["research"]["strategy_id"],
        "evidence_label": config["research"]["current_evidence_label"],
        "candidate_count": len(ranking),
        "historical_validation_leader": ranking[0]["id"],
        "historical_gate_pass_count": int(sum(item["historical_gate_pass"] for item in ranking)),
        "true_locked_oos_start": config["research"]["true_locked_oos_start"],
        "r6_forward_1_status": "NOT_ACTIVATED_NO_TRUE_LOCKED_OOS",
        "neighbor_group_positive_excess_ratio": group_positive,
        "ranking": ranking,
        "hashes": hashes,
        "governance": config["governance"],
    }
    clean_payload = _clean(payload)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(clean_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    REPORT_MD.write_text(_render_report(clean_payload), encoding="utf-8")
    FORWARD_STATUS.parent.mkdir(parents=True, exist_ok=True)
    FORWARD_STATUS.write_text(
        json.dumps(
            {
                "strategy": "R6_FORWARD_1",
                "status": "NOT_ACTIVATED_NO_TRUE_LOCKED_OOS",
                "historical_validation_leader": clean_payload["historical_validation_leader"],
                "historical_gate_pass_count": clean_payload["historical_gate_pass_count"],
                "automatic_ordering_authorized": False,
                "reason": "历史已被查看，尚无严格锁定样本外；不得冻结前向冠军。",
                "experiment_matrix_sha256": hashes[MATRIX_FILE.name],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "historical_validation_leader": clean_payload["historical_validation_leader"],
                "historical_gate_pass_count": clean_payload["historical_gate_pass_count"],
                "r6_forward_1_status": clean_payload["r6_forward_1_status"],
                "report": REPORT_MD.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
