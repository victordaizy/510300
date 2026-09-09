"""生成、回测并固化沪深300双袖带Alpha终稿策略。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.cross_sectional_engine import CrossSectionalCosts, run_weighted_open_backtest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "final_alpha_strategy.yaml"
PANEL_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
INDEX_FILE = ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
BENCHMARK_FILE = ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
BREADTH_FILE = ROOT / "data" / "features" / "000300_equal_weight_breadth_daily.parquet"
REPORT_DIR = ROOT / "reports" / "backtest"
OUTPUT_DIR = ROOT / "data" / "processed" / "final_alpha_strategy"
PAPER_DIR = ROOT / "paper"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_factor_panel(
    panel: pd.DataFrame,
    index: pd.DataFrame,
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    """只用信号日及更早数据构造点时成员截面因子。"""

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["con_code", "date"]).reset_index(drop=True)
    grouped = data.groupby("con_code", sort=False)
    data["return_1d"] = grouped["total_return_close"].pct_change(fill_method=None)
    for days in (20, 60, 120):
        data[f"momentum_{days}d"] = grouped["total_return_close"].pct_change(
            days, fill_method=None
        )
    data["volatility_20d"] = (
        grouped["return_1d"].rolling(20, min_periods=15).std().reset_index(level=0, drop=True)
    )
    data["volatility_60d"] = (
        grouped["return_1d"].rolling(60, min_periods=40).std().reset_index(level=0, drop=True)
    )
    rolling_high = (
        grouped["total_return_close"]
        .rolling(60, min_periods=40)
        .max()
        .reset_index(level=0, drop=True)
    )
    data["drawdown_60d"] = data["total_return_close"] / rolling_high - 1.0
    data["average_amount_20d"] = (
        grouped["amount"].rolling(20, min_periods=15).mean().reset_index(level=0, drop=True)
    )

    factor_columns = [
        "momentum_20d",
        "momentum_60d",
        "momentum_120d",
        "volatility_20d",
        "volatility_60d",
        "drawdown_60d",
    ]
    members = data["is_index_member"].astype(bool)
    for column in factor_columns:
        rank_column = f"member_rank_{column}"
        data[rank_column] = np.nan
        data.loc[members, rank_column] = data.loc[members].groupby("date")[column].rank(
            pct=True
        )

    rank_m20 = data["member_rank_momentum_20d"]
    rank_m60 = data["member_rank_momentum_60d"]
    rank_m120 = data["member_rank_momentum_120d"]
    rank_v20 = data["member_rank_volatility_20d"]
    rank_v60 = data["member_rank_volatility_60d"]
    rank_dd60 = data["member_rank_drawdown_60d"]
    data["score_concentrated_bull"] = 0.50 * rank_m60 + 0.30 * rank_m20 + 0.20 * rank_dd60
    data["score_diversified_bull"] = (
        0.45 * rank_m120 + 0.30 * rank_m60 + 0.15 * (1.0 - rank_v60) + 0.10 * rank_dd60
    )
    data["score_defensive"] = (
        0.60 * (1.0 - rank_v60) + 0.20 * (1.0 - rank_v20) + 0.20 * rank_dd60
    )

    index_state = index[["date", "close"]].copy().sort_values("date")
    index_state["date"] = pd.to_datetime(index_state["date"])
    index_state["index_above_ma120"] = index_state["close"] > index_state["close"].rolling(120).mean()
    index_state["index_return_20d"] = index_state["close"].pct_change(20)
    breadth_state = breadth[["date", "equal_above_ma60_share"]].copy()
    breadth_state["date"] = pd.to_datetime(breadth_state["date"])
    index_state = index_state.merge(breadth_state, on="date", how="left", validate="one_to_one")
    index_state["regime"] = np.where(
        index_state["index_above_ma120"]
        & index_state["index_return_20d"].gt(0.0)
        & index_state["equal_above_ma60_share"].gt(0.50),
        "BULL",
        "DEFENSIVE",
    )
    return data.merge(
        index_state[["date", "regime"]],
        on="date",
        how="left",
        validate="many_to_one",
    )


def build_targets(factors: pd.DataFrame, config: dict) -> pd.DataFrame:
    """按固定十交易日节奏生成双袖带合并目标权重。"""

    start = pd.Timestamp(config["schedule"]["start_date"])
    end = pd.Timestamp(config["schedule"]["end_date"])
    calendar = pd.DatetimeIndex(
        sorted(factors.loc[factors["date"].between(start, end), "date"].unique())
    )
    step = int(config["schedule"]["rebalance_every_trading_days"])
    signal_dates = set(calendar[::step])
    minimum_amount = float(config["universe"]["minimum_20d_average_amount_cny"])
    eligible = factors.loc[
        factors["date"].isin(signal_dates)
        & factors["is_index_member"].astype(bool)
        & ~factors["is_suspended"].astype(bool)
        & factors["average_amount_20d"].ge(minimum_amount)
    ].copy()

    sleeve_definitions = {
        "concentrated": {
            "capital_weight": float(config["sleeves"]["concentrated"]["capital_weight"]),
            "holdings": int(config["sleeves"]["concentrated"]["holdings"]),
            "bull_score": "score_concentrated_bull",
        },
        "diversified": {
            "capital_weight": float(config["sleeves"]["diversified"]["capital_weight"]),
            "holdings": int(config["sleeves"]["diversified"]["holdings"]),
            "bull_score": "score_diversified_bull",
        },
    }
    rows: list[dict] = []
    for signal_date in sorted(signal_dates):
        day = eligible.loc[eligible["date"].eq(signal_date)].copy()
        if day.empty:
            continue
        regime = str(day["regime"].iloc[0])
        for sleeve, definition in sleeve_definitions.items():
            score_column = definition["bull_score"] if regime == "BULL" else "score_defensive"
            candidates = day.dropna(subset=[score_column]).sort_values(
                [score_column, "con_code"], ascending=[False, True]
            )
            holdings = int(definition["holdings"])
            selected = candidates.head(holdings)
            if selected.empty:
                continue
            per_name_weight = float(definition["capital_weight"]) / holdings
            for row in selected.itertuples(index=False):
                rows.append(
                    {
                        "signal_date": signal_date,
                        "con_code": str(row.con_code),
                        "sleeve": sleeve,
                        "sleeve_weight": per_name_weight,
                        "score": float(getattr(row, score_column)),
                        "regime": regime,
                    }
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("没有生成任何策略目标")
    merged = (
        raw.groupby(["signal_date", "con_code", "regime"], as_index=False)
        .agg(
            target_weight=("sleeve_weight", "sum"),
            sleeves=("sleeve", lambda values: "+".join(sorted(values))),
            score=("score", "max"),
        )
        .sort_values(["signal_date", "target_weight", "score", "con_code"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
    maximum_weight = float(config["risk"]["maximum_merged_single_name_weight"])
    if merged["target_weight"].max() > maximum_weight + 1e-9:
        raise ValueError(
            f"合并后单名权重{merged['target_weight'].max():.2%}超过上限{maximum_weight:.2%}"
        )
    return merged


def _benchmark_series(benchmark: pd.DataFrame, calendar: pd.Series, initial_cash: float) -> pd.DataFrame:
    data = benchmark[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").reindex(pd.DatetimeIndex(calendar)).reset_index()
    data.columns = ["date", "close"]
    if data["close"].isna().any():
        raise ValueError("基准无法覆盖策略交易日历")
    data["daily_return"] = data["close"].pct_change().fillna(0.0)
    data["equity"] = initial_cash * (1.0 + data["daily_return"]).cumprod()
    return data


def _return_metrics(returns: pd.Series, dates: pd.Series) -> dict:
    if returns.empty:
        return {}
    total_return = float((1.0 + returns).prod() - 1.0)
    elapsed_days = max((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days, 1)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(242))
    sharpe = float(returns.mean() * 242 / volatility) if volatility > 0 else None
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "calmar": cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else None,
        "observations": int(len(returns)),
    }


def summarize(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    initial_cash: float,
) -> dict:
    strategy = _return_metrics(ledger["daily_return"], ledger["date"])
    benchmark_metrics = _return_metrics(benchmark["daily_return"], benchmark["date"])
    rolling_strategy = ledger["equity"] / ledger["equity"].shift(242) - 1.0
    rolling_benchmark = benchmark["equity"] / benchmark["equity"].shift(242) - 1.0
    rolling_excess = rolling_strategy - rolling_benchmark
    valid_rolling = rolling_excess.dropna()
    benchmark_variance = float(benchmark["daily_return"].var(ddof=1))
    beta = (
        float(ledger["daily_return"].cov(benchmark["daily_return"]) / benchmark_variance)
        if benchmark_variance > 0
        else None
    )
    if trades.empty:
        total_cost = 0.0
        traded_notional = 0.0
    else:
        total_cost = float(trades[["commission", "stamp_duty", "slippage"]].sum().sum())
        traded_notional = float(trades["notional"].sum())
    annual_rows: dict[str, dict] = {}
    annual = pd.DataFrame(
        {
            "date": ledger["date"],
            "strategy": ledger["daily_return"],
            "benchmark": benchmark["daily_return"],
        }
    )
    annual["year"] = annual["date"].dt.year
    for year, frame in annual.groupby("year", sort=True):
        strategy_return = float((1.0 + frame["strategy"]).prod() - 1.0)
        benchmark_return = float((1.0 + frame["benchmark"]).prod() - 1.0)
        annual_rows[str(int(year))] = {
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "excess": strategy_return - benchmark_return,
            "observations": int(len(frame)),
            "complete_year": bool(len(frame) >= 230),
        }
    return {
        "strategy": strategy,
        "benchmark": benchmark_metrics,
        "annualized_excess": strategy["cagr"] - benchmark_metrics["cagr"],
        "rolling_242d_excess": {
            "median": float(valid_rolling.median()),
            "minimum": float(valid_rolling.min()),
            "maximum": float(valid_rolling.max()),
            "positive_ratio": float((valid_rolling > 0).mean()),
            "at_least_20pct_ratio": float((valid_rolling >= 0.20).mean()),
            "observations": int(len(valid_rolling)),
        },
        "actual_beta": beta,
        "average_exposure": float(ledger["actual_exposure"].mean()),
        "average_position_count": float(ledger["position_count"].mean()),
        "trade_rows": int(len(trades)),
        "traded_notional_over_initial_cash": traded_notional / initial_cash,
        "total_cost_cny": total_cost,
        "calendar_years": annual_rows,
    }


def _period_summaries(ledger: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    periods = {
        "development": ("2021-08-12", "2024-06-28", "模型开发历史"),
        "pseudo_oos": ("2024-08-01", "2025-06-30", "已参与后续研究，只能称伪样本外"),
        "contaminated_recent": ("2025-08-01", "2026-08-11", "受污染近期回顾"),
    }
    output: dict[str, dict] = {}
    for name, (start, end, governance) in periods.items():
        mask = ledger["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        strategy_metrics = _return_metrics(ledger.loc[mask, "daily_return"], ledger.loc[mask, "date"])
        benchmark_metrics = _return_metrics(
            benchmark.loc[mask, "daily_return"], benchmark.loc[mask, "date"]
        )
        output[name] = {
            "start_date": start,
            "end_date": end,
            "governance": governance,
            "strategy": strategy_metrics,
            "benchmark": benchmark_metrics,
            "total_return_excess": strategy_metrics["total_return"] - benchmark_metrics["total_return"],
        }
    return output


def _render_markdown(report: dict) -> str:
    base = report["base_cost"]
    stress = report["stress_cost"]
    gate = report["retrospective_gate"]
    lines = [
        "# 沪深300双袖带Alpha终稿策略", "",
        f"> 历史回测门槛：`{gate['status']}`。全部历史均已用于研究，这不是未来收益承诺，也不是真正样本外证明。", "",
        "## 核心结果", "",
        "|情景|策略年化|基准年化|年化超额|滚动242日超额中位|滚动达到20%比例|最大回撤|Sharpe|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("基础成本", base), ("15bp滑点压力", stress)):
        lines.append(
            f"|{label}|{item['strategy']['cagr']:.2%}|{item['benchmark']['cagr']:.2%}|"
            f"{item['annualized_excess']:.2%}|{item['rolling_242d_excess']['median']:.2%}|"
            f"{item['rolling_242d_excess']['at_least_20pct_ratio']:.2%}|"
            f"{item['strategy']['maximum_drawdown']:.2%}|{item['strategy']['sharpe_zero_cash_rate']:.2f}|"
        )
    lines += [
        "", "## 完整规则", "",
        "- 股票池：信号日当时的沪深300成分股；20日平均成交额不低于5000万元；停牌股不入选。",
        "- 节奏：每10个交易日生成一次信号，收盘后计算，下一交易日开盘执行。",
        "- 市场状态：指数高于MA120、20日收益为正且等权成分股站上MA60比例高于50%时为强市，否则为防御状态。",
        "- 集中袖带40%：强市按60日动量50%+20日动量30%+60日回撤位置20%，持有前三；防御期改用低波评分。",
        "- 分散袖带60%：强市按120日动量45%+60日动量30%+低波15%+回撤位置10%，持有前八；防御期同样使用低波评分。",
        "- 两袖带权重相加；目标买卖按100股整手向下取整，未分配资金留作现金。",
        "- 成本：佣金单边3bp且最低5元；卖出印花税在2023-08-28前按10bp、此后按5bp；基础滑点单边5bp，压力情景15bp。",
        "- 停牌、近似涨停买入或近似跌停卖出不假定成交；旧仓无法卖出时保留到可交易。",
        "", "## 自然年度", "",
        "|年份|策略|基准|超额|完整年度|", "|---|---:|---:|---:|---|",
    ]
    for year, item in base["calendar_years"].items():
        lines.append(
            f"|{year}|{item['strategy_return']:.2%}|{item['benchmark_return']:.2%}|"
            f"{item['excess']:.2%}|{item['complete_year']}|"
        )
    lines += ["", "## 分段回顾", "", "|区间|治理标签|策略收益|基准收益|超额|", "|---|---|---:|---:|---:|"]
    for name, item in report["periods"].items():
        lines.append(
            f"|{name}|{item['governance']}|{item['strategy']['total_return']:.2%}|"
            f"{item['benchmark']['total_return']:.2%}|{item['total_return_excess']:.2%}|"
        )
    lines += [
        "", "## 治理与使用边界", "",
        "- 该策略是在已查看的2021-08-12至2026-08-12历史上形成，`PASS_RETROSPECTIVE`只表示指定历史口径达标。",
        "- 真正前向记录从2026-08-13开始；冻结后不得继续用相同历史修改权重、窗口、持股数或状态阈值。",
        "- 上实盘前至少完成纸面交易、券商成交对账、公司行动核对和容量检查；当前配置未授权自动下单。",
        "- 组合集中度高，基础情景也可能出现显著回撤；任何20%数字都不是保本或收益保证。", "",
    ]
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    panel = pd.read_parquet(PANEL_FILE)
    index = pd.read_parquet(INDEX_FILE)
    breadth = pd.read_parquet(BREADTH_FILE)
    benchmark_raw = pd.read_parquet(BENCHMARK_FILE)
    factors = build_factor_panel(panel, index, breadth)
    targets = build_targets(factors, config)

    account = config["account"]
    cost_config = config["costs"]
    initial_cash = float(account["initial_cash_cny"])
    base_costs = CrossSectionalCosts(
        commission_rate=float(cost_config["commission_rate"]),
        minimum_commission_cny=float(cost_config["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(cost_config["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(
            cost_config["stamp_duty_sell_rate_before_reduction"]
        ),
        stamp_duty_reduction_effective_date=str(
            cost_config["stamp_duty_reduction_effective_date"]
        ),
        slippage_bps_per_leg=float(cost_config["base_slippage_bps_per_leg"]),
        cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]),
    )
    stress_costs = CrossSectionalCosts(
        commission_rate=base_costs.commission_rate,
        minimum_commission_cny=base_costs.minimum_commission_cny,
        stamp_duty_sell_rate=base_costs.stamp_duty_sell_rate,
        stamp_duty_sell_rate_before_reduction=(
            base_costs.stamp_duty_sell_rate_before_reduction
        ),
        stamp_duty_reduction_effective_date=(
            base_costs.stamp_duty_reduction_effective_date
        ),
        slippage_bps_per_leg=float(cost_config["stress_slippage_bps_per_leg"]),
        cash_annual_rate=base_costs.cash_annual_rate,
        lot_size=base_costs.lot_size,
    )
    start = config["schedule"]["start_date"]
    end = config["schedule"]["end_date"]
    maximum_gap = float(config["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_weighted_open_backtest(
        panel, targets, initial_cash, base_costs, start, end, maximum_gap
    )
    stress_ledger, stress_trades = run_weighted_open_backtest(
        panel, targets, initial_cash, stress_costs, start, end, maximum_gap
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base_summary = summarize(base_ledger, base_trades, benchmark, initial_cash)
    stress_summary = summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    annual_target = 0.20
    gate_pass = (
        base_summary["annualized_excess"] >= annual_target
        and base_summary["rolling_242d_excess"]["median"] >= annual_target
    )
    gate = {
        "status": "PASS_RETROSPECTIVE" if gate_pass else "FAIL_RETROSPECTIVE",
        "annualized_excess_minimum": annual_target,
        "rolling_242d_excess_median_minimum": annual_target,
        "warning": "历史门槛不是未来收益承诺",
    }
    latest_date = pd.Timestamp(targets["signal_date"].max())
    latest_targets = targets.loc[targets["signal_date"].eq(latest_date)].copy()
    report = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "strategy": config["strategy"],
        "cost_assumptions": config["costs"],
        "retrospective_gate": gate,
        "base_cost": base_summary,
        "stress_cost": stress_summary,
        "periods": _period_summaries(base_ledger, benchmark),
        "latest_signal_date": str(latest_date.date()),
        "latest_targets": latest_targets.to_dict("records"),
        "data_governance": {
            "point_in_time_membership": True,
            "signal_to_execution": "NEXT_TRADING_DAY_OPEN",
            "strict_holdout_available": False,
            "true_forward_start": config["governance"]["true_forward_start"],
            "hashes": {
                "config": _sha256(CONFIG_FILE),
                "constituent_panel": _sha256(PANEL_FILE),
                "index": _sha256(INDEX_FILE),
                "benchmark": _sha256(BENCHMARK_FILE),
                "breadth": _sha256(BREADTH_FILE),
            },
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(OUTPUT_DIR / "targets.parquet", index=False)
    base_ledger.to_parquet(OUTPUT_DIR / "base_ledger.parquet", index=False)
    base_trades.to_parquet(OUTPUT_DIR / "base_trades.parquet", index=False)
    stress_ledger.to_parquet(OUTPUT_DIR / "stress_ledger.parquet", index=False)
    stress_trades.to_parquet(OUTPUT_DIR / "stress_trades.parquet", index=False)
    (REPORT_DIR / "final_alpha_strategy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (REPORT_DIR / "final_alpha_strategy.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    (PAPER_DIR / "final_alpha_latest_signal.json").write_text(
        json.dumps(
            {
                "strategy_id": config["strategy"]["strategy_id"],
                "signal_date": str(latest_date.date()),
                "execution": "下一交易日开盘",
                "targets": latest_targets.to_dict("records"),
                "live_order_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(base_ledger["date"], base_ledger["equity"] / initial_cash, label="双袖带策略（基础成本）")
    axis.plot(benchmark["date"], benchmark["equity"] / initial_cash, label="沪深300全收益指数")
    axis.set_title("沪深300双袖带Alpha策略历史净值")
    axis.set_ylabel("累计净值")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "final_alpha_strategy_equity_curve.png", dpi=160)
    plt.close(figure)

    print(f"历史门槛：{gate['status']}")
    print(f"基础成本年化超额：{base_summary['annualized_excess']:.2%}")
    print(f"滚动242日超额中位：{base_summary['rolling_242d_excess']['median']:.2%}")
    print(f"基础成本最大回撤：{base_summary['strategy']['maximum_drawdown']:.2%}")
    print(f"压力成本年化超额：{stress_summary['annualized_excess']:.2%}")
    print(f"报告：{REPORT_DIR / 'final_alpha_strategy.md'}")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
