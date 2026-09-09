"""510300微观结构风险开关的2015年以来连续长周期复制。

本研究只把已经冻结的候选规则应用到更早历史，并按新的20万元账户合同
计算净收益、夏普率、回撤和压力事件归因。它不搜索参数、不生成订单，也
不在微观结构规则未通过第一阶段门槛时运行估值组合。
"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import validate_etf_microstructure_dual_shadow_v1_frozen as frozen_candidate
from binary_state_feasibility_v1 import CostModel, simulate_binary_path


PROJECT_ID = "510300_LIQUIDITY_STRESS_LONG_CYCLE_REPLICATION_V1"
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "510300_liquidity_stress_long_cycle_replication_v1.yaml"
)


def sha256_file(path: Path) -> str:
    """返回文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    """把有限数转换为JSON可安全保存的浮点数。"""

    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """规范日期、排序并拒绝重复日期。"""

    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, kind="mergesort", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        duplicates = result.loc[result[column].duplicated(keep=False), column]
        raise ValueError(
            f"{column}存在重复日期：{duplicates.dt.strftime('%Y-%m-%d').tolist()[:20]}"
        )
    return result


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入JSON研究结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet研究结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """原子写入UTF-8 BOM CSV，方便Windows直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_text(content: str, path: Path) -> None:
    """原子写入可读文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_and_validate_config() -> tuple[dict[str, Any], dict[str, str]]:
    """读取冻结协议，并只检查协议列明的文件哈希与账户合同。"""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ValueError("长周期复制项目编号不匹配")
    if config["protocol"]["state"] != "FROZEN_BEFORE_LONG_CYCLE_RESULT_COMPUTATION":
        raise ValueError("协议不是结果计算前冻结状态")

    tracked: dict[str, str] = {}
    candidate = config["frozen_candidate"]
    records = [candidate["config"], candidate["implementation"]]
    records.extend(config["inputs"].values())
    for record in records:
        path_value = record.get("path")
        expected = record.get("sha256")
        if not path_value or not expected:
            continue
        path = PROJECT_ROOT / str(path_value)
        if not path.exists():
            raise FileNotFoundError(f"冻结输入不存在：{path_value}")
        actual = sha256_file(path)
        tracked[str(path_value)] = actual
        if actual != str(expected):
            raise ValueError(
                f"冻结输入哈希漂移：{path_value}，expected={expected}，actual={actual}"
            )

    account_source = yaml.safe_load(
        (PROJECT_ROOT / config["inputs"]["account_contract_source"]["path"]).read_text(
            encoding="utf-8"
        )
    )["execution"]
    execution = config["execution"]
    exact_account_checks = {
        "initial_capital_cny": execution["initial_capital_cny"],
        "commission_rate_per_leg": execution["base_cost"][
            "commission_rate_per_leg"
        ],
        "minimum_commission_cny_per_leg": execution["base_cost"][
            "minimum_commission_cny_per_leg"
        ],
        "slippage_bps_per_leg": execution["base_cost"]["slippage_bps_per_leg"],
        "double_cost_commission_rate_per_leg": execution["stress_cost"][
            "commission_rate_per_leg"
        ],
        "double_cost_slippage_bps_per_leg": execution["stress_cost"][
            "slippage_bps_per_leg"
        ],
        "lot_size_shares": execution["lot_size_shares"],
        "stamp_duty_rate": execution["stamp_duty_rate"],
        "cash_annual_rate": execution["cash_annual_rate"],
    }
    mismatches = {
        key: {"contract": account_source.get(key), "replication": value}
        for key, value in exact_account_checks.items()
        if float(account_source.get(key)) != float(value)
    }
    if mismatches:
        raise ValueError(f"20万元账户合同不一致：{mismatches}")
    if not bool(account_source["t_plus_one"]):
        raise ValueError("账户合同必须使用T+1")
    if account_source["signal_clock"] != "AFTER_CLOSE":
        raise ValueError("账户合同信号时钟不是收盘后")
    if account_source["execution_clock"] != "NEXT_TRADING_DAY_OPEN":
        raise ValueError("账户合同执行时钟不是下一交易日开盘")
    return config, tracked


def load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """拼接同口径历史信号，并形成连续市场与分红输入。"""

    inputs = config["inputs"]
    dates = config["dates"]
    warmup_start = pd.Timestamp(dates["feature_warmup_start"])
    signal_cutoff = pd.Timestamp(dates["signal_cutoff"])
    execution_end = pd.Timestamp(dates["last_execution_mark_date"])
    seam_date = pd.Timestamp(dates["margin_seam_overlap_date"])

    share_premium = normalize_date(
        pd.read_parquet(PROJECT_ROOT / inputs["share_and_close_premium"]["path"])
    )
    required_share = {"date", "fund_shares", "close_premium_to_nav"}
    if missing := required_share.difference(share_premium.columns):
        raise ValueError(f"份额与折溢价缺少字段：{sorted(missing)}")
    prohibited = [
        column
        for column in share_premium.columns
        if any(token in column.lower() for token in ("forward_return", "future_return", "target_return"))
    ]
    if prohibited:
        raise ValueError(f"信号输入含未来收益字段：{prohibited}")

    historical_margin = normalize_date(
        pd.read_parquet(PROJECT_ROOT / inputs["historical_margin"]["path"])
    )
    current_margin = normalize_date(
        pd.read_parquet(PROJECT_ROOT / inputs["current_margin"]["path"])
    )
    margin_fields = ["rzye", "rqye", "rqyl", "rzrqye"]
    for label, frame in [
        ("historical_margin", historical_margin),
        ("current_margin", current_margin),
    ]:
        if missing := {"date", *margin_fields}.difference(frame.columns):
            raise ValueError(f"{label}缺少字段：{sorted(missing)}")

    historical_seam = historical_margin.loc[
        historical_margin["date"].eq(seam_date)
    ]
    current_seam = current_margin.loc[current_margin["date"].eq(seam_date)]
    if len(historical_seam) != 1 or len(current_seam) != 1:
        raise ValueError("两融拼接日记录不是各自唯一一行")
    direct_tolerance = float(
        config["data_admission"]["seam_requirements"][
            "direct_fields_absolute_tolerance"
        ]
    )
    derived_tolerance = float(
        config["data_admission"]["seam_requirements"][
            "derived_fields_absolute_tolerance"
        ]
    )
    seam_differences: dict[str, float] = {}
    for column in margin_fields:
        difference = abs(
            float(historical_seam[column].iloc[0])
            - float(current_seam[column].iloc[0])
        )
        seam_differences[column] = difference
        tolerance = derived_tolerance if column in {"rqye", "rzrqye"} else direct_tolerance
        if difference > tolerance:
            raise ValueError(
                f"两融拼接日字段不一致：{column}差值{difference}大于容差{tolerance}"
            )

    margin = pd.concat(
        [
            historical_margin.loc[historical_margin["date"].lt(seam_date)],
            current_margin.loc[current_margin["date"].ge(seam_date)],
        ],
        ignore_index=True,
    )
    margin = normalize_date(margin)

    etf = normalize_date(pd.read_parquet(PROJECT_ROOT / inputs["etf_market"]["path"]))
    benchmark = normalize_date(
        pd.read_parquet(PROJECT_ROOT / inputs["benchmark_total_return"]["path"])
    )
    for label, frame, required in [
        ("510300行情", etf, {"date", "open", "close"}),
        ("H00300总回报", benchmark, {"date", "close"}),
    ]:
        if missing := required.difference(frame.columns):
            raise ValueError(f"{label}缺少字段：{sorted(missing)}")

    signal_calendar = etf.loc[
        etf["date"].between(warmup_start, signal_cutoff), ["date"]
    ].copy()
    signal = (
        signal_calendar.merge(
            share_premium[["date", "close_premium_to_nav", "fund_shares"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            margin[["date", *margin_fields]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    required_signal = list(config["data_admission"]["required_signal_columns"])
    missing_mask = signal[required_signal].isna().any(axis=1)
    if missing_mask.any():
        dates_missing = signal.loc[missing_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"连续信号输入存在缺失且禁止填充：{dates_missing[:50]}")
    nonpositive = {
        column: int((pd.to_numeric(signal[column], errors="coerce") <= 0.0).sum())
        for column in ["fund_shares", *margin_fields]
    }
    if any(nonpositive.values()):
        raise ValueError(f"对数特征输入存在非正值：{nonpositive}")

    market = etf.loc[
        etf["date"].between(warmup_start, execution_end), ["date", "open", "close"]
    ].rename(columns={"open": "etf_open", "close": "etf_close"})
    market = market.merge(
        benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("连续执行行情或H00300总回报存在缺失")

    dividends = pd.read_csv(PROJECT_ROOT / inputs["cash_dividends"]["path"])
    required_dividend = {
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
    }
    if missing := required_dividend.difference(dividends.columns):
        raise ValueError(f"分红数据缺少字段：{sorted(missing)}")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)

    audit = {
        "status": "PASS_SAME_DEFINITION_CONTINUOUS_HISTORY",
        "signal_rows": int(len(signal)),
        "signal_first_date": signal["date"].min().date().isoformat(),
        "signal_last_date": signal["date"].max().date().isoformat(),
        "market_rows": int(len(market)),
        "market_first_date": market["date"].min().date().isoformat(),
        "market_last_date": market["date"].max().date().isoformat(),
        "historical_margin_rows": int(len(historical_margin)),
        "current_margin_rows": int(len(current_margin)),
        "combined_margin_rows": int(len(margin)),
        "margin_seam_date": seam_date.date().isoformat(),
        "margin_seam_absolute_differences": seam_differences,
        "historical_margin_source": sorted(
            str(value) for value in historical_margin.get("source", pd.Series(dtype=str)).dropna().unique()
        ),
        "missing_signal_rows": 0,
        "duplicate_signal_dates": 0,
        "future_return_columns_read": 0,
        "static_fill_rows": 0,
        "proxy_substitution_rows": 0,
        "dividend_event_count": int(len(dividends)),
        "dividend_first_ex_date": dividends["ex_date"].min().date().isoformat(),
        "dividend_last_ex_date": dividends["ex_date"].max().date().isoformat(),
    }
    return signal, market.reset_index(drop=True), dividends, audit


def cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    """根据冻结新账户合同构建基础或压力成本。"""

    if scenario not in {"BASE", "STRESS"}:
        raise ValueError("成本情景只能是BASE或STRESS")
    spec = config["execution"]["base_cost" if scenario == "BASE" else "stress_cost"]
    return CostModel(
        commission_rate=float(spec["commission_rate_per_leg"]),
        minimum_commission=float(spec["minimum_commission_cny_per_leg"]),
        slippage_bps=float(spec["slippage_bps_per_leg"]),
        cash_annual_rate=float(config["execution"]["cash_annual_rate"]),
        trading_days_per_year=int(config["metrics"]["annualization_trading_days"]),
        lot_size=int(config["execution"]["lot_size_shares"]),
    )


def sharpe_ratio(returns: pd.Series, annualization: int) -> float | None:
    """计算零无风险利率的年化净夏普率。"""

    values = pd.to_numeric(returns, errors="coerce").dropna()
    if len(values) < 2:
        return None
    volatility = float(values.std(ddof=1))
    if not np.isfinite(volatility) or volatility <= 0.0:
        return None
    return float(values.mean() / volatility * math.sqrt(annualization))


def path_metrics(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
    annualization: int,
) -> dict[str, Any]:
    """汇总一条已扣成本净值。"""

    returns = pd.to_numeric(ledger["daily_return"].iloc[1:], errors="raise")
    total_return = float(ledger["equity"].iloc[-1] / initial_capital - 1.0)
    cagr = float(np.expm1(np.log1p(returns).mean() * annualization))
    executed = trades if not trades.empty else pd.DataFrame()
    return {
        "anchor_date": ledger["date"].iloc[0].date().isoformat(),
        "evaluation_start_date": ledger["date"].iloc[1].date().isoformat(),
        "end_date": ledger["date"].iloc[-1].date().isoformat(),
        "trading_day_count": int(len(returns)),
        "initial_capital_cny": initial_capital,
        "final_equity_cny": float(ledger["equity"].iloc[-1]),
        "cumulative_net_return": total_return,
        "cagr": cagr,
        "net_sharpe": sharpe_ratio(returns, annualization),
        "maximum_drawdown": float(ledger["drawdown"].min()),
        "cash_day_count": int((ledger["policy_state"].iloc[1:] == 0).sum()),
        "cash_day_share": float((ledger["policy_state"].iloc[1:] == 0).mean()),
        "trade_leg_count": int(len(executed)),
        "total_commission_cny": (
            0.0 if executed.empty else float(executed["commission"].sum())
        ),
        "total_slippage_cost_cny": (
            0.0 if executed.empty else float(executed["slippage_cost"].sum())
        ),
    }


def period_path_metrics(
    ledger: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    annualization: int,
) -> dict[str, Any]:
    """从连续账户净值中提取一个固定期间，不重置账户。"""

    selected = np.flatnonzero(ledger["date"].between(start, end).to_numpy())
    if len(selected) == 0:
        return {
            "first_date": None,
            "last_date": None,
            "trading_day_count": 0,
            "return": None,
            "sharpe": None,
            "maximum_drawdown": None,
            "cash_day_share": None,
        }
    first = int(selected[0])
    last = int(selected[-1])
    base_index = max(first - 1, 0)
    base_equity = float(ledger["equity"].iloc[base_index])
    path = ledger.iloc[base_index : last + 1].copy()
    path_return = float(ledger["equity"].iloc[last] / base_equity - 1.0)
    relative = path["equity"] / base_equity
    drawdown = relative / relative.cummax() - 1.0
    returns = pd.to_numeric(ledger["daily_return"].iloc[first : last + 1], errors="raise")
    return {
        "first_date": ledger["date"].iloc[first].date().isoformat(),
        "last_date": ledger["date"].iloc[last].date().isoformat(),
        "trading_day_count": int(last - first + 1),
        "return": path_return,
        "sharpe": sharpe_ratio(returns, annualization),
        "maximum_drawdown": float(drawdown.min()),
        "cash_day_share": float((ledger["policy_state"].iloc[first : last + 1] == 0).mean()),
    }


def build_annual_results(
    ledgers: dict[str, dict[str, pd.DataFrame]],
    annualization: int,
) -> pd.DataFrame:
    """生成逐年收益、夏普、回撤和超额。"""

    rows: list[dict[str, Any]] = []
    years = sorted(
        int(value)
        for value in ledgers["BASE"]["strategy"]["date"].iloc[1:].dt.year.unique()
    )
    for scenario, paths in ledgers.items():
        for year in years:
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)
            strategy = period_path_metrics(
                paths["strategy"], start=start, end=end, annualization=annualization
            )
            buy_hold = period_path_metrics(
                paths["buy_hold"], start=start, end=end, annualization=annualization
            )
            rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "first_date": strategy["first_date"],
                    "last_date": strategy["last_date"],
                    "trading_day_count": strategy["trading_day_count"],
                    "strategy_net_return": strategy["return"],
                    "buy_hold_net_return": buy_hold["return"],
                    "net_excess_return": (
                        None
                        if strategy["return"] is None or buy_hold["return"] is None
                        else float(strategy["return"] - buy_hold["return"])
                    ),
                    "strategy_net_sharpe": strategy["sharpe"],
                    "buy_hold_net_sharpe": buy_hold["sharpe"],
                    "strategy_maximum_drawdown": strategy["maximum_drawdown"],
                    "buy_hold_maximum_drawdown": buy_hold["maximum_drawdown"],
                    "strategy_cash_day_share": strategy["cash_day_share"],
                }
            )
    return pd.DataFrame(rows)


def build_pressure_cycle_results(
    config: dict[str, Any],
    ledgers: dict[str, dict[str, pd.DataFrame]],
    annualization: int,
) -> pd.DataFrame:
    """按预先冻结的独立压力周期归因。"""

    rows: list[dict[str, Any]] = []
    for scenario, paths in ledgers.items():
        for cycle_id, cycle in config["pressure_cycles"].items():
            start = pd.Timestamp(cycle["start"])
            end = pd.Timestamp(cycle["end"])
            strategy = period_path_metrics(
                paths["strategy"], start=start, end=end, annualization=annualization
            )
            buy_hold = period_path_metrics(
                paths["buy_hold"], start=start, end=end, annualization=annualization
            )
            rows.append(
                {
                    "scenario": scenario,
                    "cycle_id": cycle_id,
                    "cycle_label": cycle["label"],
                    "declared_start": cycle["start"],
                    "declared_end": cycle["end"],
                    "first_date": strategy["first_date"],
                    "last_date": strategy["last_date"],
                    "trading_day_count": strategy["trading_day_count"],
                    "strategy_net_return": strategy["return"],
                    "buy_hold_net_return": buy_hold["return"],
                    "net_excess_return": (
                        None
                        if strategy["return"] is None or buy_hold["return"] is None
                        else float(strategy["return"] - buy_hold["return"])
                    ),
                    "strategy_net_sharpe": strategy["sharpe"],
                    "buy_hold_net_sharpe": buy_hold["sharpe"],
                    "strategy_maximum_drawdown": strategy["maximum_drawdown"],
                    "buy_hold_maximum_drawdown": buy_hold["maximum_drawdown"],
                    "strategy_cash_day_share": strategy["cash_day_share"],
                }
            )
    return pd.DataFrame(rows)


def find_cash_episodes(states: pd.Series) -> list[dict[str, Any]]:
    """识别连续现金区间，并保留样本边界删失状态。"""

    values = pd.to_numeric(states, errors="raise").astype(int).to_numpy()
    episodes: list[dict[str, Any]] = []
    active_start: int | None = None
    left_censored = False
    for index in range(1, len(values)):
        if values[index] == 0 and active_start is None:
            if values[index - 1] == 1:
                active_start = index
                left_censored = False
            elif index == 1:
                active_start = index
                left_censored = True
        elif values[index] == 1 and active_start is not None:
            episodes.append(
                {
                    "start_index": active_start,
                    "end_index": index,
                    "left_censored": left_censored,
                    "right_censored": False,
                }
            )
            active_start = None
            left_censored = False
    if active_start is not None:
        episodes.append(
            {
                "start_index": active_start,
                "end_index": len(values) - 1,
                "left_censored": left_censored,
                "right_censored": True,
            }
        )
    return episodes


def episode_path_values(
    ledger: pd.DataFrame, start_index: int, end_index: int
) -> dict[str, float]:
    """计算一个现金区间相对区间前一日的收益和路径回撤。"""

    prior = max(start_index - 1, 0)
    base = float(ledger["equity"].iloc[prior])
    path = ledger["equity"].iloc[prior : end_index + 1].astype(float) / base
    return {
        "return": float(ledger["equity"].iloc[end_index] / base - 1.0),
        "minimum_path_return": float(path.min() - 1.0),
        "maximum_path_return": float(path.max() - 1.0),
    }


def build_cash_episode_results(
    sample: pd.DataFrame,
    ledgers: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """生成全部现金区间和触发后5/10/20/60日结果。"""

    strategy_base = ledgers["BASE"]["strategy"]
    episodes = find_cash_episodes(strategy_base["policy_state"])
    episode_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    windows = [5, 10, 20, 60]

    for episode_number, episode in enumerate(episodes, start=1):
        start_index = int(episode["start_index"])
        end_index = int(episode["end_index"])
        prior_index = max(start_index - 1, 0)
        base_strategy = episode_path_values(
            ledgers["BASE"]["strategy"], start_index, end_index
        )
        base_buy_hold = episode_path_values(
            ledgers["BASE"]["buy_hold"], start_index, end_index
        )
        stress_strategy = episode_path_values(
            ledgers["STRESS"]["strategy"], start_index, end_index
        )
        stress_buy_hold = episode_path_values(
            ledgers["STRESS"]["buy_hold"], start_index, end_index
        )
        episode_rows.append(
            {
                "episode_number": episode_number,
                "signal_date": sample["date"].iloc[prior_index].date().isoformat(),
                "cash_execution_start": sample["date"].iloc[start_index].date().isoformat(),
                "reentry_or_sample_end": sample["date"].iloc[end_index].date().isoformat(),
                "cash_trading_days": int(
                    (strategy_base["policy_state"].iloc[start_index:end_index] == 0).sum()
                ),
                "left_censored": bool(episode["left_censored"]),
                "right_censored": bool(episode["right_censored"]),
                "base_strategy_return": base_strategy["return"],
                "base_buy_hold_return": base_buy_hold["return"],
                "base_net_excess_return": float(
                    base_strategy["return"] - base_buy_hold["return"]
                ),
                "stress_strategy_return": stress_strategy["return"],
                "stress_buy_hold_return": stress_buy_hold["return"],
                "stress_net_excess_return": float(
                    stress_strategy["return"] - stress_buy_hold["return"]
                ),
                "stress_strategy_minimum_path_return": stress_strategy[
                    "minimum_path_return"
                ],
                "stress_buy_hold_minimum_path_return": stress_buy_hold[
                    "minimum_path_return"
                ],
                "stress_avoided_path_loss": float(
                    stress_strategy["minimum_path_return"]
                    - stress_buy_hold["minimum_path_return"]
                ),
                "stress_missed_upside": float(
                    max(stress_buy_hold["return"] - stress_strategy["return"], 0.0)
                ),
            }
        )

        if bool(episode["left_censored"]):
            continue
        for window in windows:
            window_end = start_index + window - 1
            complete = window_end < len(sample)
            if not complete:
                window_end = len(sample) - 1
            base_strategy_window = episode_path_values(
                ledgers["BASE"]["strategy"], start_index, window_end
            )
            base_buy_hold_window = episode_path_values(
                ledgers["BASE"]["buy_hold"], start_index, window_end
            )
            stress_strategy_window = episode_path_values(
                ledgers["STRESS"]["strategy"], start_index, window_end
            )
            stress_buy_hold_window = episode_path_values(
                ledgers["STRESS"]["buy_hold"], start_index, window_end
            )
            forward_rows.append(
                {
                    "episode_number": episode_number,
                    "cash_execution_start": sample["date"].iloc[start_index].date().isoformat(),
                    "window_trading_days": window,
                    "window_complete": complete,
                    "window_end": sample["date"].iloc[window_end].date().isoformat(),
                    "base_strategy_return": base_strategy_window["return"],
                    "base_buy_hold_return": base_buy_hold_window["return"],
                    "base_net_excess_return": float(
                        base_strategy_window["return"] - base_buy_hold_window["return"]
                    ),
                    "stress_strategy_return": stress_strategy_window["return"],
                    "stress_buy_hold_return": stress_buy_hold_window["return"],
                    "stress_net_excess_return": float(
                        stress_strategy_window["return"]
                        - stress_buy_hold_window["return"]
                    ),
                    "stress_strategy_minimum_path_return": stress_strategy_window[
                        "minimum_path_return"
                    ],
                    "stress_buy_hold_minimum_path_return": stress_buy_hold_window[
                        "minimum_path_return"
                    ],
                    "stress_avoided_path_loss": float(
                        stress_strategy_window["minimum_path_return"]
                        - stress_buy_hold_window["minimum_path_return"]
                    ),
                }
            )

    episode_frame = pd.DataFrame(episode_rows)
    forward_frame = pd.DataFrame(forward_rows)
    complete_episodes = episode_frame.loc[
        ~episode_frame.get("left_censored", pd.Series(dtype=bool)).astype(bool)
        & ~episode_frame.get("right_censored", pd.Series(dtype=bool)).astype(bool)
    ].copy()
    positive = complete_episodes.loc[
        complete_episodes.get("stress_net_excess_return", pd.Series(dtype=float)).gt(0.0),
        "stress_net_excess_return",
    ]
    dependence = {
        "episode_count_including_censored": int(len(episode_frame)),
        "complete_episode_count": int(len(complete_episodes)),
        "positive_complete_episode_count_under_stress": int(len(positive)),
        "largest_positive_episode_share_under_stress": (
            None if positive.empty else float(positive.max() / positive.sum())
        ),
        "total_complete_episode_net_excess_under_stress": (
            0.0
            if complete_episodes.empty
            else float(complete_episodes["stress_net_excess_return"].sum())
        ),
    }
    return episode_frame, forward_frame, dependence


def leave_one_cycle_out_results(
    config: dict[str, Any],
    ledgers: dict[str, dict[str, pd.DataFrame]],
    annualization: int,
) -> list[dict[str, Any]]:
    """删除每个预声明压力周期的交易日后，检查全样本方向是否反转。"""

    rows: list[dict[str, Any]] = []
    strategy = ledgers["STRESS"]["strategy"]
    buy_hold = ledgers["STRESS"]["buy_hold"]
    for cycle_id, cycle in config["pressure_cycles"].items():
        start = pd.Timestamp(cycle["start"])
        end = pd.Timestamp(cycle["end"])
        keep = ~strategy["date"].between(start, end)
        keep.iloc[0] = False
        strategy_returns = strategy.loc[keep, "daily_return"].astype(float)
        buy_hold_returns = buy_hold.loc[keep, "daily_return"].astype(float)
        strategy_total = float(np.prod(1.0 + strategy_returns) - 1.0)
        buy_hold_total = float(np.prod(1.0 + buy_hold_returns) - 1.0)
        rows.append(
            {
                "deleted_cycle_id": cycle_id,
                "deleted_cycle_label": cycle["label"],
                "remaining_trading_days": int(keep.sum()),
                "strategy_cumulative_return": strategy_total,
                "buy_hold_cumulative_return": buy_hold_total,
                "cumulative_net_excess": float(strategy_total - buy_hold_total),
                "strategy_sharpe": sharpe_ratio(strategy_returns, annualization),
            }
        )
    return rows


def h00300_reference_metrics(
    sample: pd.DataFrame, initial_capital: float, annualization: int
) -> dict[str, Any]:
    """形成不参与主门槛的H00300总回报参考。"""

    equity = initial_capital * sample["benchmark_close"] / float(
        sample["benchmark_close"].iloc[0]
    )
    returns = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    evaluation_returns = returns.iloc[1:]
    return {
        "cumulative_return": float(equity.iloc[-1] / initial_capital - 1.0),
        "cagr": float(
            np.expm1(np.log1p(evaluation_returns).mean() * annualization)
        ),
        "sharpe": sharpe_ratio(evaluation_returns, annualization),
        "maximum_drawdown": float(drawdown.min()),
    }


def evaluate_gates(
    config: dict[str, Any],
    scenario_summaries: dict[str, dict[str, Any]],
    cycle_results: pd.DataFrame,
) -> dict[str, Any]:
    """按结果计算前冻结的四项第一阶段门槛裁决。"""

    gates = config["decision_gates"]["phase_1_microstructure"]
    minimum_sharpe = float(gates["gate_1_net_sharpe"]["minimum"])
    sharpe_values = {
        scenario: scenario_summaries[scenario]["strategy"]["net_sharpe"]
        for scenario in ["BASE", "STRESS"]
    }
    gate_1 = all(
        value is not None and float(value) >= minimum_sharpe
        for value in sharpe_values.values()
    )

    excess_values = {
        scenario: scenario_summaries[scenario]["comparison"][
            "cumulative_net_excess_return"
        ]
        for scenario in ["BASE", "STRESS"]
    }
    gate_2 = all(float(value) > 0.0 for value in excess_values.values())

    maximum_ratio = float(
        gates["gate_3_drawdown_reduction"][
            "maximum_drawdown_magnitude_ratio_vs_buy_hold"
        ]
    )
    drawdown_ratios: dict[str, float | None] = {}
    for scenario in ["BASE", "STRESS"]:
        strategy_drawdown = abs(
            float(scenario_summaries[scenario]["strategy"]["maximum_drawdown"])
        )
        buy_hold_drawdown = abs(
            float(scenario_summaries[scenario]["buy_hold"]["maximum_drawdown"])
        )
        drawdown_ratios[scenario] = (
            None if buy_hold_drawdown <= 0.0 else strategy_drawdown / buy_hold_drawdown
        )
    gate_3 = all(
        ratio is not None and ratio <= maximum_ratio
        for ratio in drawdown_ratios.values()
    )

    stress_cycles = cycle_results.loc[
        cycle_results["scenario"].eq("STRESS")
    ].copy()
    positive = stress_cycles.loc[
        pd.to_numeric(stress_cycles["net_excess_return"], errors="coerce").gt(0.0),
        "net_excess_return",
    ].astype(float)
    positive_count = int(len(positive))
    largest_positive_share = (
        None if positive.empty else float(positive.max() / positive.sum())
    )
    minimum_positive = int(
        gates["gate_4_multiple_independent_cycles"][
            "minimum_positive_pressure_cycles_under_stress"
        ]
    )
    maximum_largest_share = float(
        gates["gate_4_multiple_independent_cycles"][
            "maximum_largest_positive_cycle_share"
        ]
    )
    gate_4 = bool(
        positive_count >= minimum_positive
        and largest_positive_share is not None
        and largest_positive_share <= maximum_largest_share
    )

    phase_1_pass = bool(gate_1 and gate_2 and gate_3 and gate_4)
    return {
        "gate_1_net_sharpe_at_least_1_2_in_base_and_stress": {
            "passed": gate_1,
            "threshold": minimum_sharpe,
            "values": sharpe_values,
        },
        "gate_2_positive_cumulative_net_excess_in_base_and_stress": {
            "passed": gate_2,
            "threshold_exclusive": 0.0,
            "values": excess_values,
        },
        "gate_3_maximum_drawdown_magnitude_at_most_80pct_of_buy_hold": {
            "passed": gate_3,
            "maximum_ratio": maximum_ratio,
            "values": drawdown_ratios,
        },
        "gate_4_multiple_independent_pressure_cycles": {
            "passed": gate_4,
            "minimum_positive_cycles": minimum_positive,
            "positive_cycle_count": positive_count,
            "maximum_largest_positive_cycle_share": maximum_largest_share,
            "largest_positive_cycle_share": largest_positive_share,
        },
        "phase_1_passed": phase_1_pass,
        "phase_2_status": (
            "REQUIRED_BY_FROZEN_PROTOCOL"
            if phase_1_pass
            else "SKIPPED_PHASE_1_FAILED"
        ),
        "gate_5_combination_adds_value": (
            "NOT_EVALUATED_PHASE_2_REQUIRED"
            if phase_1_pass
            else "NOT_EVALUATED_BY_STOP_RULE"
        ),
    }


def pct(value: Any, digits: int = 2) -> str:
    """Markdown百分比格式。"""

    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def number(value: Any, digits: int = 3) -> str:
    """Markdown数值格式。"""

    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_report(
    payload: dict[str, Any],
    annual: pd.DataFrame,
    cycles: pd.DataFrame,
    episodes: pd.DataFrame,
) -> str:
    """生成面向普通读者的中文结果文档。"""

    status = payload["status"]
    summaries = payload["scenario_summaries"]
    gates = payload["decision_gates"]
    lines = [
        "# 510300流动性压力风险开关：2015年以来连续历史复制",
        "",
        f"**最终状态：{status}**",
        "",
        "本报告只检验已经冻结的微观结构风险开关。规则、阈值、252日分位、迟滞、三票触发和双影子门均未修改；信号在T日收盘后形成，T+1开盘执行。没有加入估值、MACD、分钟择时、做T或新因子。",
        "",
        "## 一、主结果",
        "",
        "| 成本情景 | 策略累计净收益 | 买入持有累计净收益 | 累计净超额 | 策略CAGR | 策略净夏普 | 策略最大回撤 | 买入持有最大回撤 | 现金日占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in ["BASE", "STRESS"]:
        summary = summaries[scenario]
        lines.append(
            "| {scenario} | {strategy_return} | {buy_return} | {excess} | {cagr} | {sharpe} | {mdd} | {buy_mdd} | {cash} |".format(
                scenario="现实成本" if scenario == "BASE" else "压力成本",
                strategy_return=pct(summary["strategy"]["cumulative_net_return"]),
                buy_return=pct(summary["buy_hold"]["cumulative_net_return"]),
                excess=pct(summary["comparison"]["cumulative_net_excess_return"]),
                cagr=pct(summary["strategy"]["cagr"]),
                sharpe=number(summary["strategy"]["net_sharpe"]),
                mdd=pct(summary["strategy"]["maximum_drawdown"]),
                buy_mdd=pct(summary["buy_hold"]["maximum_drawdown"]),
                cash=pct(summary["strategy"]["cash_day_share"]),
            )
        )

    lines.extend(
        [
            "",
            "现实成本为20万元、佣金万分之二、每笔最低5元、单边5BP滑点；压力成本为佣金万分之四、单边10BP滑点，最低佣金仍按合同5元。策略和买入持有使用相同的首次建仓成本。",
            "",
            "## 二、五项决策门槛",
            "",
            "| 门槛 | 结果 | 关键数值 |",
            "|---|---|---|",
        ]
    )
    gate_1 = gates["gate_1_net_sharpe_at_least_1_2_in_base_and_stress"]
    gate_2 = gates["gate_2_positive_cumulative_net_excess_in_base_and_stress"]
    gate_3 = gates["gate_3_maximum_drawdown_magnitude_at_most_80pct_of_buy_hold"]
    gate_4 = gates["gate_4_multiple_independent_pressure_cycles"]
    lines.extend(
        [
            f"| 1. 基础与压力成本净夏普均不低于1.2 | {'通过' if gate_1['passed'] else '不通过'} | BASE {number(gate_1['values']['BASE'])}；STRESS {number(gate_1['values']['STRESS'])} |",
            f"| 2. 两种成本下累计净超额均为正 | {'通过' if gate_2['passed'] else '不通过'} | BASE {pct(gate_2['values']['BASE'])}；STRESS {pct(gate_2['values']['STRESS'])} |",
            f"| 3. 最大回撤幅度不超过买入持有的80% | {'通过' if gate_3['passed'] else '不通过'} | BASE {pct(gate_3['values']['BASE'])}；STRESS {pct(gate_3['values']['STRESS'])}（这里是回撤幅度比例） |",
            f"| 4. 至少两个独立压力周期贡献正超额，且最大单周期不超过正贡献80% | {'通过' if gate_4['passed'] else '不通过'} | 正贡献周期 {gate_4['positive_cycle_count']} 个；最大占比 {pct(gate_4['largest_positive_cycle_share'])} |",
            f"| 5. 估值基线加微观结构否决必须增加价值 | {'待第二阶段' if gates['phase_1_passed'] else '未运行'} | {gates['gate_5_combination_adds_value']} |",
            "",
            f"第一阶段总裁决：**{'通过' if gates['phase_1_passed'] else '不通过'}**。",
            "",
            "## 三、逐年结果",
            "",
            "以下列出压力成本下每个自然年；2015与2026是部分年度。",
            "",
            "| 年份 | 策略净收益 | 买入持有净收益 | 净超额 | 策略净夏普 | 策略最大回撤 | 买入持有最大回撤 | 现金日占比 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    stress_annual = annual.loc[annual["scenario"].eq("STRESS")]
    for row in stress_annual.itertuples(index=False):
        lines.append(
            f"| {row.year} | {pct(row.strategy_net_return)} | {pct(row.buy_hold_net_return)} | {pct(row.net_excess_return)} | {number(row.strategy_net_sharpe)} | {pct(row.strategy_maximum_drawdown)} | {pct(row.buy_hold_maximum_drawdown)} | {pct(row.strategy_cash_day_share)} |"
        )

    lines.extend(
        [
            "",
            "## 四、预声明压力周期归因",
            "",
            "| 压力周期 | 策略净收益 | 买入持有净收益 | 净超额 | 策略最大回撤 | 买入持有最大回撤 | 现金日占比 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    stress_cycles = cycles.loc[cycles["scenario"].eq("STRESS")]
    for row in stress_cycles.itertuples(index=False):
        lines.append(
            f"| {row.cycle_label} | {pct(row.strategy_net_return)} | {pct(row.buy_hold_net_return)} | {pct(row.net_excess_return)} | {pct(row.strategy_maximum_drawdown)} | {pct(row.buy_hold_maximum_drawdown)} | {pct(row.strategy_cash_day_share)} |"
        )

    lines.extend(
        [
            "",
            "## 五、全部现金区间",
            "",
            "区间收益从退出前一交易日收盘算到重新进入日（或样本末日）收盘，已经包含实际交易成本和分红处理。",
            "",
            "| 编号 | 退出执行日 | 重入/样本末日 | 现金交易日 | 压力成本策略收益 | 买入持有收益 | 净超额 | 避免的路径损失 | 错失上涨 | 删失 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in episodes.itertuples(index=False):
        censored = "左删失" if row.left_censored else ""
        if row.right_censored:
            censored = (censored + "、" if censored else "") + "右删失"
        lines.append(
            f"| {row.episode_number} | {row.cash_execution_start} | {row.reentry_or_sample_end} | {row.cash_trading_days} | {pct(row.stress_strategy_return)} | {pct(row.stress_buy_hold_return)} | {pct(row.stress_net_excess_return)} | {pct(row.stress_avoided_path_loss)} | {pct(row.stress_missed_upside)} | {censored or '完整'} |"
        )

    boundary = payload["boundaries"]
    data_audit = payload["data_admission"]
    lines.extend(
        [
            "",
            "## 六、数据口径与边界",
            "",
            f"- 连续信号历史：{data_audit['signal_first_date']}至{data_audit['signal_last_date']}，共{data_audit['signal_rows']}个510300交易日；正式评价为{payload['evaluation_scope']['evaluation_start']}至{payload['evaluation_scope']['last_execution_mark_date']}。",
            f"- 两融拼接日：{data_audit['margin_seam_date']}。直接字段零差；重建融券余额绝对差{data_audit['margin_seam_absolute_differences']['rqye']:.12f}元。",
            "- 历史融券余额不是替代指标：使用上交所融券余量乘同日510300收盘价，和当前同名字段在拼接日达到浮点舍入级一致。没有静态填充，也没有把市场级两融数据替代510300自身数据。",
            "- H00300只作为指数总回报参考；所有通过门槛均以真实可交易的510300含分红买入持有为主基准。",
            f"- 估值组合状态：{boundary['valuation_combination_status']}。",
            "- 本研究仅是历史反向复制，不是未见数据上的前瞻证明，也不授权Paper、下单或实盘。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    """执行一次冻结长周期复制并写出全部研究结果。"""

    config, input_hashes = load_and_validate_config()
    signal, market, dividends, data_audit = load_inputs(config)
    candidate_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_candidate"]["config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    features = frozen_candidate._rebuild_features(
        signal, market, dividends, candidate_config
    )
    if not features["date"].equals(signal["date"]):
        raise ValueError("冻结规则输出日期与连续信号日期不一致")

    execution = market.merge(
        features[
            [
                "date",
                "cash_raw",
                "cash_final",
                "state_final",
                "micro_risk_vote_count",
                "shadow_gate_active",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
    execution["cash_execution"] = execution["cash_final"].shift(1)
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    evaluation_end = pd.Timestamp(config["dates"]["last_execution_mark_date"])
    selected = np.flatnonzero(
        execution["date"].between(evaluation_start, evaluation_end).to_numpy()
    )
    if len(selected) == 0:
        raise ValueError("评价区间没有交易日")
    first = int(selected[0])
    last = int(selected[-1])
    if first <= 0:
        raise ValueError("评价起点缺少前一交易日现金锚点")
    sample = execution.iloc[first - 1 : last + 1][
        ["date", "etf_open", "etf_close", "benchmark_close", "cash_execution"]
    ].reset_index(drop=True)
    if sample["cash_execution"].iloc[1:].isna().any():
        missing_dates = sample.loc[
            sample["cash_execution"].isna(), "date"
        ].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"评价区间缺少T-1信号状态：{missing_dates}")

    strategy_states = np.zeros(len(sample), dtype=np.int8)
    strategy_states[1:] = np.where(
        sample["cash_execution"].iloc[1:].astype(bool).to_numpy(), 0, 1
    ).astype(np.int8)
    buy_hold_states = np.ones(len(sample), dtype=np.int8)
    buy_hold_states[0] = 0

    initial_capital = float(config["execution"]["initial_capital_cny"])
    annualization = int(config["metrics"]["annualization_trading_days"])
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    trades: dict[str, pd.DataFrame] = {}
    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario in ["BASE", "STRESS"]:
        costs = cost_model(config, scenario)
        strategy_ledger, strategy_trades = simulate_binary_path(
            sample[["date", "etf_open", "etf_close", "benchmark_close"]],
            dividends,
            strategy_states,
            costs=costs,
            initial_capital=initial_capital,
            reinvest_paid_dividends=True,
        )
        buy_hold_ledger, buy_hold_trades = simulate_binary_path(
            sample[["date", "etf_open", "etf_close", "benchmark_close"]],
            dividends,
            buy_hold_states,
            costs=costs,
            initial_capital=initial_capital,
            reinvest_paid_dividends=True,
        )
        strategy_summary = path_metrics(
            strategy_ledger,
            strategy_trades,
            initial_capital=initial_capital,
            annualization=annualization,
        )
        buy_hold_summary = path_metrics(
            buy_hold_ledger,
            buy_hold_trades,
            initial_capital=initial_capital,
            annualization=annualization,
        )
        comparison = {
            "cumulative_net_excess_return": float(
                strategy_summary["cumulative_net_return"]
                - buy_hold_summary["cumulative_net_return"]
            ),
            "relative_terminal_wealth_advantage": float(
                strategy_summary["final_equity_cny"]
                / buy_hold_summary["final_equity_cny"]
                - 1.0
            ),
            "cagr_excess": float(strategy_summary["cagr"] - buy_hold_summary["cagr"]),
            "maximum_drawdown_magnitude_ratio": float(
                abs(strategy_summary["maximum_drawdown"])
                / abs(buy_hold_summary["maximum_drawdown"])
            ),
            "maximum_drawdown_improvement_percentage_points": float(
                strategy_summary["maximum_drawdown"]
                - buy_hold_summary["maximum_drawdown"]
            ),
        }
        ledgers[scenario] = {
            "strategy": strategy_ledger,
            "buy_hold": buy_hold_ledger,
        }
        trades[scenario] = strategy_trades
        scenario_summaries[scenario] = {
            "strategy": strategy_summary,
            "buy_hold": buy_hold_summary,
            "comparison": comparison,
        }

    annual_results = build_annual_results(ledgers, annualization)
    cycle_results = build_pressure_cycle_results(config, ledgers, annualization)
    episode_results, forward_results, event_dependence = build_cash_episode_results(
        sample, ledgers
    )
    leave_one_out = leave_one_cycle_out_results(config, ledgers, annualization)
    gates = evaluate_gates(config, scenario_summaries, cycle_results)
    h00300 = h00300_reference_metrics(sample, initial_capital, annualization)
    status = (
        "PHASE_1_PASSED_PHASE_2_REQUIRED"
        if gates["phase_1_passed"]
        else config["stop_rules"]["if_phase_1_fails"]
    )

    output_paths = {
        key: PROJECT_ROOT / value
        for key, value in config["outputs"].items()
        if key != "root"
    }
    payload: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "status": status,
        "phase_1_passed": gates["phase_1_passed"],
        "candidate_policy": config["frozen_candidate"]["candidate_name"],
        "evidence_class": config["protocol"]["evidence_class"],
        "config_file": str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_hashes": input_hashes,
        "data_admission": data_audit,
        "evaluation_scope": {
            "feature_warmup_start": config["dates"]["feature_warmup_start"],
            "evaluation_start": config["dates"]["evaluation_start"],
            "signal_cutoff": config["dates"]["signal_cutoff"],
            "last_execution_mark_date": config["dates"]["last_execution_mark_date"],
            "signal_time": config["scope"]["signal_clock"],
            "execution_time": config["scope"]["execution_clock"],
        },
        "account_contract": config["execution"],
        "frozen_rule": config["frozen_candidate"],
        "scenario_summaries": scenario_summaries,
        "h00300_total_return_reference": h00300,
        "decision_gates": gates,
        "event_dependence": event_dependence,
        "leave_one_pressure_cycle_out": leave_one_out,
        "annual_results": json.loads(
            annual_results.to_json(orient="records", date_format="iso")
        ),
        "pressure_cycle_results": json.loads(
            cycle_results.to_json(orient="records", date_format="iso")
        ),
        "cash_episode_results": json.loads(
            episode_results.to_json(orient="records", date_format="iso")
        ),
        "event_forward_window_results": json.loads(
            forward_results.to_json(orient="records", date_format="iso")
        ),
        "boundaries": {
            "candidate_formula_changed": False,
            "thresholds_changed": False,
            "dual_shadow_internal_cost_changed": False,
            "new_account_contract_used_only_for_realized_wealth_paths": True,
            "future_or_forward_signal_data_used": False,
            "valuation_combination_status": gates["phase_2_status"],
            "parameter_rescue_allowed": False,
            "research_only": True,
            "paper_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
        "artifacts": {
            key: str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for key, path in output_paths.items()
        },
    }

    atomic_parquet(features, output_paths["features"])
    atomic_parquet(ledgers["BASE"]["strategy"], output_paths["strategy_base_ledger"])
    atomic_parquet(
        ledgers["STRESS"]["strategy"], output_paths["strategy_stress_ledger"]
    )
    atomic_parquet(ledgers["BASE"]["buy_hold"], output_paths["buy_hold_base_ledger"])
    atomic_parquet(
        ledgers["STRESS"]["buy_hold"], output_paths["buy_hold_stress_ledger"]
    )
    atomic_csv(trades["BASE"], output_paths["trades_base"])
    atomic_csv(trades["STRESS"], output_paths["trades_stress"])
    atomic_csv(annual_results, output_paths["annual_results"])
    atomic_csv(cycle_results, output_paths["pressure_cycle_results"])
    atomic_csv(episode_results, output_paths["cash_episode_results"])
    atomic_csv(forward_results, output_paths["forward_window_results"])
    atomic_json(payload, output_paths["report_json"])
    atomic_text(
        markdown_report(payload, annual_results, cycle_results, episode_results),
        output_paths["report_markdown"],
    )
    return payload


def main() -> int:
    """命令行入口。"""

    payload = run()
    print(
        json.dumps(
            {
                "project_id": payload["project_id"],
                "status": payload["status"],
                "phase_1_passed": payload["phase_1_passed"],
                "base": payload["scenario_summaries"]["BASE"],
                "stress": payload["scenario_summaries"]["STRESS"],
                "decision_gates": payload["decision_gates"],
                "report_json": payload["artifacts"]["report_json"],
                "report_markdown": payload["artifacts"]["report_markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
