"""510300原油趋势月度择时V1的冻结研究实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import NormalDist
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)


CONFIG_PATH = ROOT / "config" / "510300_oil_trend_monthly_timing_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_oil_trend_monthly_timing_v1_manifest.json"


class ContractError(RuntimeError):
    """冻结协议或输入契约不成立。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        ("protocol", "project_id"): "510300_OIL_TREND_MONTHLY_TIMING_V1",
        ("scope", "execution_asset"): "510300.SH",
        ("dates", "evaluation_start"): "2015-01-05",
        ("dates", "evaluation_end"): "2026-08-12",
        ("data_contract", "oil_series"): "RBRTE",
        ("oil_factor", "lag_months"): 12,
        ("oil_factor", "availability_lag_us_business_days"): 5,
        ("oil_factor", "robustness_availability_lag_us_business_days"): 10,
        ("portfolio_gates", "base_net_sharpe_minimum"): 1.20,
    }
    for keys, value in expected.items():
        observed: Any = config
        for key in keys:
            observed = observed[key]
        if observed != value:
            raise ContractError(f"冻结字段{'.'.join(keys)}异常：{observed!r}")
    if config["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("只允许持有510300.SH或人民币现金")
    forbidden_true = [
        config["scope"]["leverage_allowed"],
        config["scope"]["short_selling_allowed"],
        config["scope"]["derivatives_execution_allowed"],
        config["scope"]["live_trading_authorized"],
        config["governance"]["order_generation"],
        config["governance"]["broker_connection"],
        config["governance"]["position_change"],
        config["governance"]["live_trading_authorized"],
    ]
    if any(forbidden_true):
        raise ContractError("研究边界不得授权杠杆、卖空、衍生品或实盘执行")
    if config["protocol"]["one_shot"] is not True:
        raise ContractError("候选必须是一次性冻结检验")
    if config["protocol"]["parameter_rescue_after_result"] != "forbidden":
        raise ContractError("结果后参数救援必须被禁止")
    if config["signal"]["positive_factor_target_position"] != 0.0:
        raise ContractError("正因子方向必须按文献预注册为风险规避")
    if config["signal"]["nonpositive_factor_target_position"] != 1.0:
        raise ContractError("非正因子方向必须按文献预注册为持有510300")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError(f"冻结清单不存在：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("冻结清单状态不是FROZEN_BEFORE_FIRST_RESULT")
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("冻结清单项目标识不一致")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后配置哈希发生变化")
    for relative, expected_hash in manifest.get("tracked_files", {}).items():
        path = _project_path(relative)
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ContractError(f"冻结实现文件发生变化：{relative}")
    for relative, expected_hash in manifest.get("input_files", {}).items():
        path = _project_path(relative)
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ContractError(f"冻结输入文件发生变化：{relative}")
    if manifest.get("candidate_outcomes_read_before_freeze") is not False:
        raise ContractError("冻结清单未证明结果在冻结前不可见")
    if manifest.get("portfolio_returns_read_before_freeze") is not False:
        raise ContractError("冻结清单未证明组合收益在冻结前不可见")
    return manifest


def _load_market(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    market = pd.read_parquet(path).copy()
    required = set(config["inputs"]["etf_daily"]["required_columns"])
    if missing := required.difference(market.columns):
        raise ContractError(f"510300行情缺少字段：{sorted(missing)}")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    market[numeric_columns] = market[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    market = market.sort_values("date").reset_index(drop=True)
    if market["date"].isna().any() or market["date"].duplicated().any():
        raise ContractError("510300行情日期缺失或重复")
    if market[["open", "high", "low", "close"]].isna().any().any():
        raise ContractError("510300行情OHLC存在空值")
    if (market[["open", "high", "low", "close"]] <= 0).any().any():
        raise ContractError("510300行情OHLC存在非正值")
    invalid_ohlc = (
        (market["high"] < market[["open", "close"]].max(axis=1))
        | (market["low"] > market[["open", "close"]].min(axis=1))
        | (market["high"] < market["low"])
    )
    if invalid_ohlc.any():
        raise ContractError("510300行情存在OHLC约束错误")
    contract = config["data_contract"]
    if len(market) != int(contract["expected_etf_rows"]):
        raise ContractError(f"510300行情行数异常：{len(market)}")
    if market["date"].min() != pd.Timestamp(contract["expected_etf_first_date"]):
        raise ContractError("510300行情首日异常")
    if market["date"].max() != pd.Timestamp(contract["expected_etf_last_date"]):
        raise ContractError("510300行情末日异常")
    evaluation = market.loc[
        market["date"].between(
            pd.Timestamp(config["dates"]["evaluation_start"]),
            pd.Timestamp(config["dates"]["evaluation_end"]),
        )
    ]
    if len(evaluation) != int(contract["expected_evaluation_rows"]):
        raise ContractError(f"510300评价期行数异常：{len(evaluation)}")
    return market


def _load_dividends(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    dividends = pd.read_csv(path).copy()
    required = set(config["inputs"]["dividends"]["required_columns"])
    if missing := required.difference(dividends.columns):
        raise ContractError(f"分红表缺少字段：{sorted(missing)}")
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="coerce")
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="coerce"
    )
    if dividends[["record_date", "ex_date", "payment_date"]].isna().any().any():
        raise ContractError("分红日期存在空值")
    if dividends["cash_dividend_per_share"].isna().any():
        raise ContractError("分红金额存在空值")
    if (dividends["cash_dividend_per_share"] < 0).any():
        raise ContractError("分红金额不能为负")
    if not (
        (dividends["record_date"] < dividends["ex_date"])
        & (dividends["ex_date"] <= dividends["payment_date"])
    ).all():
        raise ContractError("分红日期顺序异常")
    return dividends.sort_values("record_date").reset_index(drop=True)


def _load_benchmark(path: Path) -> pd.DataFrame:
    benchmark = pd.read_parquet(path).copy()
    if not {"date", "close"}.issubset(benchmark.columns):
        raise ContractError("H00300基准缺少date或close")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    if benchmark["date"].isna().any() or benchmark["date"].duplicated().any():
        raise ContractError("H00300基准日期缺失或重复")
    if benchmark["close"].isna().any() or (benchmark["close"] <= 0).any():
        raise ContractError("H00300基准价格缺失或非正")
    return benchmark


def _load_oil(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    oil = pd.read_parquet(path).copy()
    required = set(config["inputs"]["oil_daily"]["required_columns"])
    required.add("robust_availability_date")
    if missing := required.difference(oil.columns):
        raise ContractError(f"原油表缺少字段：{sorted(missing)}")
    for column in ("date", "availability_date", "robust_availability_date"):
        oil[column] = pd.to_datetime(oil[column], errors="coerce")
    oil["value"] = pd.to_numeric(oil["value"], errors="coerce")
    oil = oil.sort_values("date").reset_index(drop=True)
    if oil["date"].isna().any() or oil["date"].duplicated().any():
        raise ContractError("原油日期缺失或重复")
    if oil["value"].isna().any() or (oil["value"] <= 0).any():
        raise ContractError("原油价格缺失或非正，预注册对数因子不可计算")
    if sorted(oil["series"].astype(str).unique().tolist()) != [
        config["data_contract"]["oil_series"]
    ]:
        raise ContractError("原油序列标识异常")
    if not (oil["availability_date"] > oil["date"]).all():
        raise ContractError("原油可得日没有严格晚于观测日")
    if not (oil["robust_availability_date"] >= oil["availability_date"]).all():
        raise ContractError("稳健可得日早于主可得日")
    return oil


def load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    audit_path = _project_path(config["inputs"]["input_audit"]["path"])
    if not audit_path.exists():
        raise ContractError("原油趋势输入审计不存在")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    specification = config["inputs"]["input_audit"]
    if audit.get("status") != specification["required_status"]:
        raise ContractError("原油趋势输入审计不是PASS")
    if audit.get("candidate_outcomes_computed") is not False:
        raise ContractError("输入审计曾计算候选结果")
    if audit.get("portfolio_returns_computed") is not False:
        raise ContractError("输入审计曾计算组合收益")
    for relative, details in audit.get("output_files", {}).items():
        path = _project_path(relative)
        if not path.exists() or sha256_file(path) != details.get("sha256"):
            raise ContractError(f"输入审计输出哈希失配：{relative}")

    market = _load_market(_project_path(config["inputs"]["etf_daily"]["path"]), config)
    dividends = _load_dividends(
        _project_path(config["inputs"]["dividends"]["path"]), config
    )
    benchmark = _load_benchmark(
        _project_path(config["inputs"]["benchmark_total_return"]["path"])
    )
    oil = _load_oil(_project_path(config["inputs"]["oil_daily"]["path"]), config)
    return market, dividends, benchmark, oil, audit


def build_monthly_oil_factor(
    oil: pd.DataFrame,
    lag_months: int,
    *,
    availability_column: str = "availability_date",
) -> pd.DataFrame:
    """按冻结公式构造月末标准化移动平均的对数变化。"""

    if lag_months < 2:
        raise ValueError("原油趋势窗口至少为两个月")
    if availability_column not in oil.columns:
        raise ValueError(f"原油表缺少可得日字段：{availability_column}")
    frame = oil[["date", "value", availability_column, "series", "units"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame[availability_column] = pd.to_datetime(frame[availability_column])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["value"].isna().any() or (frame["value"] <= 0).any():
        raise ValueError("原油价格必须完整且为正")
    frame["oil_month"] = frame["date"].dt.to_period("M")
    monthly = (
        frame.sort_values("date")
        .groupby("oil_month", sort=True, as_index=False)
        .tail(1)
        .sort_values("date")
        .reset_index(drop=True)
    )
    monthly = monthly.rename(
        columns={
            "date": "oil_observation_date",
            "value": "oil_month_end_price",
            availability_column: "factor_available_date",
        }
    )
    monthly["normalized_moving_average"] = (
        monthly["oil_month_end_price"].rolling(lag_months, min_periods=lag_months).mean()
        / monthly["oil_month_end_price"]
    )
    monthly["oil_factor"] = np.log(monthly["normalized_moving_average"]).diff()
    monthly["lag_months"] = int(lag_months)
    monthly["availability_source_column"] = availability_column
    output = monthly.loc[monthly["oil_factor"].notna()].copy().reset_index(drop=True)
    if output["factor_available_date"].duplicated().any():
        raise ValueError("月度原油因子可得日重复")
    if not output["factor_available_date"].is_monotonic_increasing:
        raise ValueError("月度原油因子可得日不是递增序列")
    return output[
        [
            "oil_month",
            "oil_observation_date",
            "factor_available_date",
            "oil_month_end_price",
            "normalized_moving_average",
            "oil_factor",
            "lag_months",
            "availability_source_column",
            "series",
            "units",
        ]
    ]


def _confirmed_month_end_decisions(market: pd.DataFrame) -> pd.DataFrame:
    frame = market[["date"]].copy().sort_values("date").reset_index(drop=True)
    frame["market_month"] = frame["date"].dt.to_period("M")
    final_observed_month = frame["market_month"].max()
    confirmed = frame.loc[frame["market_month"] < final_observed_month]
    decisions = (
        confirmed.groupby("market_month", sort=True, as_index=False)
        .tail(1)
        .sort_values("date")
        .reset_index(drop=True)
        .rename(columns={"date": "decision_date"})
    )
    trading_dates = frame["date"].tolist()
    position = {date: index for index, date in enumerate(trading_dates)}
    decisions["execution_date"] = [
        trading_dates[position[date] + 1]
        if position[date] + 1 < len(trading_dates)
        else pd.NaT
        for date in decisions["decision_date"]
    ]
    return decisions.loc[decisions["execution_date"].notna()].reset_index(drop=True)


def build_signal_schedule(
    market: pd.DataFrame,
    monthly_factor: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    decisions = _confirmed_month_end_decisions(market)
    right = monthly_factor.sort_values("factor_available_date").copy()
    decisions["decision_date"] = pd.to_datetime(decisions["decision_date"]).astype(
        "datetime64[ns]"
    )
    right["factor_available_date"] = pd.to_datetime(
        right["factor_available_date"]
    ).astype("datetime64[ns]")
    signals = pd.merge_asof(
        decisions.sort_values("decision_date"),
        right,
        left_on="decision_date",
        right_on="factor_available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    risk_off_target = float(config["signal"]["positive_factor_target_position"])
    risk_on_target = float(config["signal"]["nonpositive_factor_target_position"])
    signals["target_position"] = np.where(
        signals["oil_factor"].isna(),
        np.nan,
        np.where(signals["oil_factor"] > 0.0, risk_off_target, risk_on_target),
    )
    signals["risk_off_signal"] = signals["oil_factor"] > 0.0
    signals["signal_reason"] = np.where(
        signals["oil_factor"].isna(),
        "因子不可得，维持前一目标或现金",
        np.where(
            signals["oil_factor"] > 0.0,
            "原油趋势因子为正，按预注册负向关系持有现金",
            "原油趋势因子非正，持有510300",
        ),
    )
    if (
        signals.loc[signals["oil_factor"].notna(), "factor_available_date"]
        > signals.loc[signals["oil_factor"].notna(), "decision_date"]
    ).any():
        raise ContractError("信号使用了决策日之后才可得的原油数据")
    return signals.reset_index(drop=True)


def _structural_period(date: pd.Timestamp, config: dict[str, Any]) -> str | None:
    for name, bounds in config["dates"]["structural_periods"].items():
        if pd.Timestamp(bounds["start"]) <= date <= pd.Timestamp(bounds["end"]):
            return str(name)
    return None


def add_mechanism_targets(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """构造相邻月度执行开盘之间的分红含权收益。"""

    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    complete = signals.loc[signals["target_position"].notna()].copy()
    complete = complete.sort_values("execution_date").reset_index(drop=True)
    complete["exit_execution_date"] = complete["execution_date"].shift(-1)
    complete = complete.loc[
        complete["execution_date"].between(start, end)
        & complete["exit_execution_date"].notna()
        & (complete["exit_execution_date"] <= end)
    ].copy()
    open_by_date = market.set_index("date")["open"]
    missing_entry = complete.loc[~complete["execution_date"].isin(open_by_date.index)]
    missing_exit = complete.loc[~complete["exit_execution_date"].isin(open_by_date.index)]
    if not missing_entry.empty or not missing_exit.empty:
        raise ContractError("机制目标的进入日或退出日缺少开盘价")
    future_returns: list[float] = []
    entitled_dividends: list[float] = []
    for row in complete.itertuples(index=False):
        entry_date = pd.Timestamp(row.execution_date)
        exit_date = pd.Timestamp(row.exit_execution_date)
        entry_open = float(open_by_date.loc[entry_date])
        exit_open = float(open_by_date.loc[exit_date])
        entitled = float(
            dividends.loc[
                (dividends["record_date"] >= entry_date)
                & (dividends["record_date"] < exit_date),
                "cash_dividend_per_share",
            ].sum()
        )
        entitled_dividends.append(entitled)
        future_returns.append((exit_open + entitled) / entry_open - 1.0)
    complete["entitled_cash_dividend_per_share"] = entitled_dividends
    complete["future_interval_total_return"] = future_returns
    complete["structural_period"] = [
        _structural_period(pd.Timestamp(date), config)
        for date in complete["execution_date"]
    ]
    complete["calendar_year"] = complete["execution_date"].dt.year
    complete["risk_group"] = np.where(
        complete["oil_factor"] > 0.0, "RISK_OFF_POSITIVE", "RISK_ON_NONPOSITIVE"
    )
    return complete.reset_index(drop=True)


def _optional_float(value: Any) -> float | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def newey_west_slope(
    factor: pd.Series,
    future_return: pd.Series,
    *,
    max_lag: int = 3,
) -> dict[str, float | int | None]:
    frame = pd.DataFrame({"x": factor, "y": future_return}).dropna()
    if len(frame) < max_lag + 4:
        return {"observations": int(len(frame)), "slope": None, "standard_error": None, "t_stat": None, "max_lag": int(max_lag)}
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ y
    residual = y - design @ coefficients
    score = design * residual[:, None]
    meat = score.T @ score
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    variance = float(covariance[1, 1])
    standard_error = math.sqrt(variance) if variance > 0 else None
    slope = float(coefficients[1])
    t_stat = slope / standard_error if standard_error else None
    return {
        "observations": int(len(frame)),
        "slope": slope,
        "standard_error": _optional_float(standard_error),
        "t_stat": _optional_float(t_stat),
        "max_lag": int(max_lag),
    }


def circular_block_bootstrap_group_difference(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    repetitions = int(config["bootstrap"]["repetitions"])
    block_length = int(config["bootstrap"]["block_length_months"])
    confidence = float(config["bootstrap"]["confidence_level"])
    seed = int(config["bootstrap"]["random_seed"])
    valid = frame[["oil_factor", "future_interval_total_return"]].dropna().reset_index(drop=True)
    n = len(valid)
    if n < block_length or repetitions < 1:
        return {
            "repetitions": repetitions,
            "block_length_months": block_length,
            "valid_draws": 0,
            "confidence_interval": [None, None],
        }
    factors = valid["oil_factor"].to_numpy(dtype=float)
    returns = valid["future_interval_total_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    block_count = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sampled_factor = factors[indices]
        sampled_return = returns[indices]
        risk_off = sampled_return[sampled_factor > 0.0]
        risk_on = sampled_return[sampled_factor <= 0.0]
        if len(risk_off) and len(risk_on):
            differences.append(float(risk_off.mean() - risk_on.mean()))
    alpha = (1.0 - confidence) / 2.0
    interval = (
        [float(np.quantile(differences, alpha)), float(np.quantile(differences, 1.0 - alpha))]
        if differences
        else [None, None]
    )
    return {
        "repetitions": repetitions,
        "block_length_months": block_length,
        "valid_draws": int(len(differences)),
        "confidence_level": confidence,
        "confidence_interval": interval,
        "seed": seed,
    }


def _group_difference(frame: pd.DataFrame, statistic: str = "mean") -> float | None:
    valid = frame[["oil_factor", "future_interval_total_return"]].dropna()
    risk_off = valid.loc[valid["oil_factor"] > 0.0, "future_interval_total_return"]
    risk_on = valid.loc[valid["oil_factor"] <= 0.0, "future_interval_total_return"]
    if risk_off.empty or risk_on.empty:
        return None
    if statistic == "mean":
        return float(risk_off.mean() - risk_on.mean())
    if statistic == "median":
        return float(risk_off.median() - risk_on.median())
    raise ValueError(f"不支持的统计量：{statistic}")


def evaluate_mechanism_gate(
    mechanism: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    valid = mechanism[["execution_date", "calendar_year", "structural_period", "oil_factor", "future_interval_total_return"]].dropna().copy()
    risk_off_count = int((valid["oil_factor"] > 0.0).sum())
    risk_on_count = int((valid["oil_factor"] <= 0.0).sum())
    mean_difference = _group_difference(valid, "mean")
    median_difference = _group_difference(valid, "median")
    regression = newey_west_slope(valid["oil_factor"], valid["future_interval_total_return"])
    bootstrap = circular_block_bootstrap_group_difference(valid, config)
    bootstrap_upper = bootstrap["confidence_interval"][1]

    structural: dict[str, dict[str, Any]] = {}
    for name in config["dates"]["structural_periods"]:
        subset = valid.loc[valid["structural_period"] == name]
        structural[name] = {
            "observations": int(len(subset)),
            "risk_off_observations": int((subset["oil_factor"] > 0.0).sum()),
            "risk_on_observations": int((subset["oil_factor"] <= 0.0).sum()),
            "mean_difference": _group_difference(subset, "mean"),
        }
    leave_one_year_out: dict[str, float | None] = {}
    for year in sorted(valid["calendar_year"].unique()):
        leave_one_year_out[str(int(year))] = _group_difference(
            valid.loc[valid["calendar_year"] != int(year)], "mean"
        )

    gate_config = config["mechanism_gates"]
    gates = {
        "minimum_complete_monthly_targets": len(valid)
        >= int(gate_config["minimum_complete_monthly_targets"]),
        "minimum_observations_each_group": min(risk_off_count, risk_on_count)
        >= int(gate_config["minimum_observations_each_group"]),
        "full_sample_mean_difference_negative": mean_difference is not None
        and mean_difference < 0.0,
        "full_sample_median_difference_negative": median_difference is not None
        and median_difference < 0.0,
        "bootstrap_90pct_upper_negative": bootstrap_upper is not None
        and float(bootstrap_upper) < 0.0,
        "newey_west_slope_negative": regression["slope"] is not None
        and float(regression["slope"]) < 0.0,
        "newey_west_one_sided_t": regression["t_stat"] is not None
        and float(regression["t_stat"])
        <= float(gate_config["newey_west_one_sided_t_maximum"]),
        "every_structural_period_mean_difference_negative": bool(structural)
        and all(
            details["mean_difference"] is not None
            and float(details["mean_difference"]) < 0.0
            for details in structural.values()
        ),
        "every_leave_one_calendar_year_out_mean_difference_negative": bool(
            leave_one_year_out
        )
        and all(value is not None and float(value) < 0.0 for value in leave_one_year_out.values()),
    }
    return {
        "passed": bool(all(gates.values())),
        "observations": int(len(valid)),
        "risk_off_observations": risk_off_count,
        "risk_on_observations": risk_on_count,
        "risk_off_mean_return": _optional_float(
            valid.loc[valid["oil_factor"] > 0.0, "future_interval_total_return"].mean()
        ),
        "risk_on_mean_return": _optional_float(
            valid.loc[valid["oil_factor"] <= 0.0, "future_interval_total_return"].mean()
        ),
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "newey_west_regression": regression,
        "bootstrap": bootstrap,
        "structural_periods": structural,
        "leave_one_calendar_year_out": leave_one_year_out,
        "gates": gates,
    }


def _engine_targets(signals: pd.DataFrame) -> pd.DataFrame:
    targets = signals.loc[
        signals["target_position"].notna(),
        ["decision_date", "execution_date", "target_position", "signal_reason"],
    ].copy()
    return targets.rename(columns={"decision_date": "date"}).sort_values("date").reset_index(drop=True)


def _costs(config: dict[str, Any], name: str) -> BacktestCosts:
    specification = config["portfolio"][name]
    return BacktestCosts(
        commission_rate=float(specification["commission_rate_per_leg"]),
        minimum_commission_cny=float(specification["minimum_commission_cny_per_leg"]),
        stamp_duty_rate=float(specification["stamp_duty_rate"]),
        slippage_bps=float(specification["slippage_bps_per_leg"]),
        lot_size=int(config["portfolio"]["lot_size_shares"]),
        cash_annual_rate=float(config["portfolio"]["cash_annual_rate"]),
    )


def _engine_start_date(market: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.Timestamp:
    earlier = market.loc[market["date"] < evaluation_start, "date"]
    if earlier.empty:
        raise ContractError("评价期前缺少一个交易日用于执行前置信号")
    return pd.Timestamp(earlier.max())


def _run_account(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    costs: BacktestCosts,
    *,
    delayed_start_months: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    engine_start = _engine_start_date(market, start)
    prepared = targets.copy()
    if delayed_start_months > 0:
        cutoff = start + pd.DateOffset(months=int(delayed_start_months))
        prepared.loc[prepared["execution_date"] < cutoff, "target_position"] = 0.0
        prepared.loc[prepared["execution_date"] < cutoff, "signal_reason"] = (
            f"延迟起点稳健性：前{int(delayed_start_months)}个月持有现金"
        )
    ledger, trades = run_long_cash_backtest(
        prices=market,
        dividends=dividends,
        targets=prepared,
        initial_cash=float(config["portfolio"]["initial_capital_cny"]),
        costs=costs,
        start_date=engine_start,
        end_date=end,
    )
    ledger = ledger.loc[ledger["date"].between(start, end)].reset_index(drop=True)
    if not trades.empty:
        trades = trades.loc[trades["date"].between(start, end)].reset_index(drop=True)
    return ledger, trades


def _run_buy_hold(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    engine_start = _engine_start_date(market, start)
    targets = pd.DataFrame(
        {
            "date": [engine_start],
            "execution_date": [start],
            "target_position": [1.0],
            "signal_reason": ["510300买入持有基准"],
        }
    )
    return _run_account(
        market,
        dividends,
        targets,
        config,
        _costs(config, "base_costs"),
    )


def _benchmark_cagr(
    benchmark: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    subset = benchmark.loc[benchmark["date"].between(start_date, end_date)].copy()
    if len(subset) < 2:
        raise ContractError("H00300基准区间不足两行")
    elapsed_days = max((subset["date"].iloc[-1] - subset["date"].iloc[0]).days, 1)
    total_return = float(subset["close"].iloc[-1] / subset["close"].iloc[0] - 1.0)
    return float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


def _return_series_summary(returns: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"observations": 0, "total_return": None, "annualized_return": None, "annualized_volatility": None, "sharpe_zero_cash_rate": None}
    total = float(np.prod(1.0 + values) - 1.0)
    annualized_return = float((1.0 + total) ** (242.0 / len(values)) - 1.0) if total > -1.0 else -1.0
    volatility = float(np.std(values, ddof=1) * math.sqrt(242.0)) if len(values) > 1 else None
    mean_return = float(np.mean(values) * 242.0)
    sharpe = mean_return / volatility if volatility and volatility > 0 else None
    return {
        "observations": int(len(values)),
        "total_return": total,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": _optional_float(sharpe),
    }


def probabilistic_sharpe_probability(
    returns: pd.Series,
    benchmark_annual_sharpe: float,
) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return None
    daily_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    benchmark_daily = float(benchmark_annual_sharpe) / math.sqrt(242.0)
    series = pd.Series(values)
    skewness = float(series.skew())
    raw_kurtosis = float(series.kurt()) + 3.0
    denominator_squared = 1.0 - skewness * daily_sharpe + ((raw_kurtosis - 1.0) / 4.0) * daily_sharpe**2
    if denominator_squared <= 0:
        return None
    z_score = (daily_sharpe - benchmark_daily) * math.sqrt(len(values) - 1) / math.sqrt(denominator_squared)
    return float(NormalDist().cdf(z_score))


def deflated_sharpe_probability(
    returns: pd.Series,
    total_trial_count: int,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return {"total_trial_count": int(total_trial_count), "expected_maximum_null_sharpe_annualized": None, "probability": None}
    trial_count = max(int(total_trial_count), 1)
    if trial_count == 1:
        expected_maximum_daily = 0.0
    else:
        gamma = 0.5772156649015329
        normal = NormalDist()
        trial_std_daily = 1.0 / math.sqrt(len(values) - 1)
        expected_maximum_daily = trial_std_daily * (
            (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + gamma * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        )
    probability = probabilistic_sharpe_probability(
        pd.Series(values), expected_maximum_daily * math.sqrt(242.0)
    )
    return {
        "total_trial_count": trial_count,
        "expected_maximum_null_sharpe_annualized": float(expected_maximum_daily * math.sqrt(242.0)),
        "probability": probability,
    }


def _annual_contribution_analysis(
    strategy_ledger: pd.DataFrame,
    buy_hold_ledger: pd.DataFrame,
) -> dict[str, Any]:
    merged = strategy_ledger[["date", "daily_return"]].merge(
        buy_hold_ledger[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    merged["year"] = merged["date"].dt.year
    annual_rows: list[dict[str, Any]] = []
    for year, subset in merged.groupby("year", sort=True):
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        buy_hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        annual_rows.append(
            {
                "year": int(year),
                "strategy_return": strategy_return,
                "buy_hold_return": buy_hold_return,
                "excess_return": strategy_return - buy_hold_return,
            }
        )
    positives = [max(float(row["excess_return"]), 0.0) for row in annual_rows]
    positive_sum = float(sum(positives))
    maximum_share = max(positives) / positive_sum if positive_sum > 0 else None
    delete_year: dict[str, float] = {}
    for year in sorted(merged["year"].unique()):
        subset = merged.loc[merged["year"] != int(year)]
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        buy_hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        delete_year[str(int(year))] = strategy_return - buy_hold_return
    return {
        "annual_rows": annual_rows,
        "maximum_single_positive_year_share_of_positive_excess": maximum_share,
        "delete_calendar_year_total_return_excess_vs_buy_hold": delete_year,
    }


def evaluate_portfolio(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    primary_signals: pd.DataFrame,
    robust_signals: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    primary_targets = _engine_targets(primary_signals)
    robust_targets = _engine_targets(robust_signals)
    base_ledger, base_trades = _run_account(
        market, dividends, primary_targets, config, _costs(config, "base_costs")
    )
    double_ledger, double_trades = _run_account(
        market, dividends, primary_targets, config, _costs(config, "double_costs")
    )
    buy_hold_ledger, buy_hold_trades = _run_buy_hold(market, dividends, config)
    robust_ledger, robust_trades = _run_account(
        market, dividends, robust_targets, config, _costs(config, "base_costs")
    )
    initial_cash = float(config["portfolio"]["initial_capital_cny"])
    base_summary = summarize_backtest(base_ledger, base_trades, initial_cash)
    double_summary = summarize_backtest(double_ledger, double_trades, initial_cash)
    buy_hold_summary = summarize_backtest(buy_hold_ledger, buy_hold_trades, initial_cash)
    robust_summary = summarize_backtest(robust_ledger, robust_trades, initial_cash)
    benchmark_cagr = _benchmark_cagr(
        benchmark,
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    base_excess_buy_hold = float(base_summary["cagr"] - buy_hold_summary["cagr"])
    base_excess_h00300 = float(base_summary["cagr"] - benchmark_cagr)
    robust_excess_buy_hold = float(robust_summary["cagr"] - buy_hold_summary["cagr"])
    drawdown_ratio = (
        abs(float(base_summary["max_drawdown"])) / abs(float(buy_hold_summary["max_drawdown"]))
        if float(buy_hold_summary["max_drawdown"]) < 0.0
        else None
    )

    structural: dict[str, Any] = {}
    merged_returns = base_ledger[["date", "daily_return"]].merge(
        buy_hold_ledger[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    for name, bounds in config["dates"]["structural_periods"].items():
        subset = merged_returns.loc[
            merged_returns["date"].between(
                pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
            )
        ]
        strategy = _return_series_summary(subset["daily_return_strategy"])
        buy_hold = _return_series_summary(subset["daily_return_buy_hold"])
        structural[name] = {
            "strategy": strategy,
            "buy_hold": buy_hold,
            "annualized_excess_vs_buy_hold": (
                float(strategy["annualized_return"] - buy_hold["annualized_return"])
                if strategy["annualized_return"] is not None
                and buy_hold["annualized_return"] is not None
                else None
            ),
        }

    delayed_starts: dict[str, Any] = {}
    for months in config["portfolio"]["delayed_start_months"]:
        delayed_ledger, delayed_trades = _run_account(
            market,
            dividends,
            primary_targets,
            config,
            _costs(config, "base_costs"),
            delayed_start_months=int(months),
        )
        summary = summarize_backtest(delayed_ledger, delayed_trades, initial_cash)
        delayed_starts[str(int(months))] = {
            "summary": summary,
            "annualized_excess_vs_buy_hold": float(summary["cagr"] - buy_hold_summary["cagr"]),
        }

    annual = _annual_contribution_analysis(base_ledger, buy_hold_ledger)
    trial_count = int(
        manifest["selection_bias_control"]["total_trial_count_including_current"]
    )
    psr = probabilistic_sharpe_probability(base_ledger["daily_return"], 1.0)
    dsr = deflated_sharpe_probability(base_ledger["daily_return"], trial_count)
    gates_config = config["portfolio_gates"]
    gates = {
        "base_net_sharpe": base_summary["sharpe_zero_cash_rate"] is not None
        and float(base_summary["sharpe_zero_cash_rate"])
        >= float(gates_config["base_net_sharpe_minimum"]),
        "double_cost_net_sharpe": double_summary["sharpe_zero_cash_rate"] is not None
        and float(double_summary["sharpe_zero_cash_rate"])
        >= float(gates_config["double_cost_net_sharpe_minimum"]),
        "annualized_excess_vs_510300_buy_hold_positive": base_excess_buy_hold > 0.0,
        "annualized_excess_vs_h00300_total_return_positive": base_excess_h00300 > 0.0,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio is not None
        and drawdown_ratio <= float(gates_config["maximum_drawdown_ratio_vs_buy_hold_maximum"]),
        "every_structural_period_net_sharpe": all(
            details["strategy"]["sharpe_zero_cash_rate"] is not None
            and float(details["strategy"]["sharpe_zero_cash_rate"])
            >= float(gates_config["every_structural_period_net_sharpe_minimum"])
            for details in structural.values()
        ),
        "every_structural_period_excess_vs_buy_hold": all(
            details["annualized_excess_vs_buy_hold"] is not None
            and float(details["annualized_excess_vs_buy_hold"]) > 0.0
            for details in structural.values()
        ),
        "every_delayed_start_net_sharpe": all(
            details["summary"]["sharpe_zero_cash_rate"] is not None
            and float(details["summary"]["sharpe_zero_cash_rate"])
            >= float(gates_config["every_delayed_start_net_sharpe_minimum"])
            for details in delayed_starts.values()
        ),
        "every_delayed_start_excess_vs_buy_hold": all(
            float(details["annualized_excess_vs_buy_hold"]) > 0.0
            for details in delayed_starts.values()
        ),
        "ten_business_day_clock_net_sharpe": robust_summary["sharpe_zero_cash_rate"] is not None
        and float(robust_summary["sharpe_zero_cash_rate"])
        >= float(gates_config["ten_business_day_clock_net_sharpe_minimum"]),
        "ten_business_day_clock_excess_vs_buy_hold": robust_excess_buy_hold > 0.0,
        "delete_any_calendar_year_excess_vs_buy_hold": bool(
            annual["delete_calendar_year_total_return_excess_vs_buy_hold"]
        )
        and all(
            float(value) > 0.0
            for value in annual["delete_calendar_year_total_return_excess_vs_buy_hold"].values()
        ),
        "maximum_single_positive_year_share": annual["maximum_single_positive_year_share_of_positive_excess"] is not None
        and float(annual["maximum_single_positive_year_share_of_positive_excess"])
        <= float(gates_config["maximum_single_positive_year_share_of_positive_excess"]),
        "deflated_sharpe_probability": dsr["probability"] is not None
        and float(dsr["probability"])
        >= float(gates_config["deflated_sharpe_probability_minimum"]),
        "probabilistic_sharpe_above_1p0": psr is not None
        and float(psr) >= float(gates_config["probabilistic_sharpe_above_1p0_minimum"]),
    }
    report = {
        "passed": bool(all(gates.values())),
        "base": base_summary,
        "double_cost": double_summary,
        "buy_hold_510300": buy_hold_summary,
        "h00300_total_return": {"cagr": benchmark_cagr},
        "robust_ten_business_day_clock": robust_summary,
        "annualized_excess_vs_510300_buy_hold": base_excess_buy_hold,
        "annualized_excess_vs_h00300_total_return": base_excess_h00300,
        "robust_clock_annualized_excess_vs_buy_hold": robust_excess_buy_hold,
        "maximum_drawdown_ratio_vs_buy_hold": _optional_float(drawdown_ratio),
        "structural_periods": structural,
        "delayed_starts": delayed_starts,
        "annual_contribution": annual,
        "probabilistic_sharpe_above_1p0": psr,
        "deflated_sharpe": dsr,
        "gates": gates,
    }
    artifacts = {
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "double_cost_ledger": double_ledger,
        "double_cost_trades": double_trades,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
        "robust_clock_ledger": robust_ledger,
        "robust_clock_trades": robust_trades,
    }
    return report, artifacts


def _input_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = _project_path(specification["path"])
        snapshot[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return snapshot


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    audit: dict[str, Any],
    monthly_factor: pd.DataFrame,
    signals: pd.DataFrame,
    mechanism: pd.DataFrame,
    mechanism_gate: dict[str, Any],
    portfolio: dict[str, Any] | None,
) -> dict[str, Any]:
    mechanism_passed = bool(mechanism_gate["passed"])
    portfolio_passed = bool(portfolio and portfolio["passed"])
    if not mechanism_passed:
        status = config["adjudication"]["mechanism_fail_status"]
        return_evaluation = "NOT_ALLOWED"
        net_sharpe: float | str = "NOT_COMPUTED"
    elif not portfolio_passed:
        status = config["adjudication"]["portfolio_fail_status"]
        return_evaluation = "ALLOWED_AND_COMPLETED"
        net_sharpe = float(portfolio["base"]["sharpe_zero_cash_rate"]) if portfolio and portfolio["base"]["sharpe_zero_cash_rate"] is not None else "NOT_COMPUTED"
    else:
        status = config["adjudication"]["portfolio_pass_status"]
        return_evaluation = "ALLOWED_AND_COMPLETED"
        net_sharpe = float(portfolio["base"]["sharpe_zero_cash_rate"])
    valid_signals = signals.loc[signals["oil_factor"].notna()]
    factor_summary = {
        "rows": int(len(monthly_factor)),
        "first_oil_month": str(monthly_factor["oil_month"].min()),
        "last_oil_month": str(monthly_factor["oil_month"].max()),
        "minimum": float(monthly_factor["oil_factor"].min()),
        "median": float(monthly_factor["oil_factor"].median()),
        "maximum": float(monthly_factor["oil_factor"].max()),
        "positive_share": float((monthly_factor["oil_factor"] > 0.0).mean()),
    }
    report = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
        },
        "pre_freeze_data_decision": config["protocol"]["pre_freeze_data_decision"],
        "literature_transfer": config["literature"],
        "input_audit": {
            "status": audit["status"],
            "candidate_outcomes_computed": audit["candidate_outcomes_computed"],
            "portfolio_returns_computed": audit["portfolio_returns_computed"],
            "oil_rows": audit["oil_daily"]["rows"],
            "raw_response_sha256": audit["source"]["raw_response_sha256"],
        },
        "factor": factor_summary,
        "signals": {
            "total_confirmed_month_end_rows": int(len(signals)),
            "valid_factor_rows": int(len(valid_signals)),
            "risk_off_rows": int((valid_signals["oil_factor"] > 0.0).sum()),
            "risk_on_rows": int((valid_signals["oil_factor"] <= 0.0).sum()),
            "lookahead_violations": int((valid_signals["factor_available_date"] > valid_signals["decision_date"]).sum()),
        },
        "mechanism_gate_passed": mechanism_passed,
        "mechanism": mechanism_gate,
        "portfolio_evaluated": mechanism_passed,
        "portfolio": portfolio,
        "adjudication": {
            "return_evaluation": return_evaluation,
            "net_sharpe": net_sharpe,
            "historical_target_achieved": portfolio_passed,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "parameter_rescue_allowed": False,
            "reverse_direction_allowed": False,
            "alternate_window_allowed": False,
            "historical_pass_authorizes_trading": False,
        },
        "selection_bias_control": manifest["selection_bias_control"],
        "input_snapshot": _input_snapshot(config),
        "boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    if len(mechanism) != mechanism_gate["observations"]:
        raise ContractError("机制表行数与机制报告不一致")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    mechanism = report["mechanism"]
    lines = [
        "# 510300 原油趋势月度择时 V1 正式裁决",
        "",
        f"- 项目标识：`{report['project_id']}`",
        f"- 状态：`{report['status']}`",
        f"- 评价区间：{report['evaluation_window']['start']} 至 {report['evaluation_window']['end']}",
        "- 可执行资产：仅 `510300.SH` 或人民币现金",
        "- 实盘授权：否",
        "",
        "## 冻结前数据决定",
        "",
        f"WTI 因 `{report['pre_freeze_data_decision']['rejection_reason']}` 被排除；改用 EIA Brent `RBRTE`。这一决定发生在任何 510300 候选收益读取之前。",
        "",
        "## 机制门",
        "",
        f"- 是否通过：`{mechanism['passed']}`",
        f"- 完整月度目标：{mechanism['observations']}",
        f"- 风险规避组 / 持有组：{mechanism['risk_off_observations']} / {mechanism['risk_on_observations']}",
        f"- 风险规避组减持有组平均收益：{mechanism['mean_difference']}",
        f"- 风险规避组减持有组中位收益：{mechanism['median_difference']}",
        f"- Newey-West 斜率 / t 值：{mechanism['newey_west_regression']['slope']} / {mechanism['newey_west_regression']['t_stat']}",
        f"- 90% 区块自助置信区间：{mechanism['bootstrap']['confidence_interval']}",
        "",
        "### 逐门槛结果",
        "",
    ]
    for name, value in mechanism["gates"].items():
        lines.append(f"- `{name}`：`{value}`")
    lines.extend(["", "## 组合与夏普率", ""])
    if not report["portfolio_evaluated"]:
        lines.extend(
            [
                "机制门失败，因此组合收益读取被协议禁止。",
                "",
                "- `RETURN_EVALUATION=NOT_ALLOWED`",
                "- `NET_SHARPE=NOT_COMPUTED`",
            ]
        )
    else:
        portfolio = report["portfolio"]
        lines.extend(
            [
                f"- 基准成本净夏普率：{portfolio['base']['sharpe_zero_cash_rate']}",
                f"- 双倍成本净夏普率：{portfolio['double_cost']['sharpe_zero_cash_rate']}",
                f"- 相对510300买入持有年化超额：{portfolio['annualized_excess_vs_510300_buy_hold']}",
                f"- 相对H00300全收益年化超额：{portfolio['annualized_excess_vs_h00300_total_return']}",
                f"- 组合门是否通过：`{portfolio['passed']}`",
                "",
                "### 逐组合门槛结果",
                "",
            ]
        )
        for name, value in portfolio["gates"].items():
            lines.append(f"- `{name}`：`{value}`")
    lines.extend(
        [
            "",
            "## 最终边界",
            "",
            f"历史目标是否达到：`{report['adjudication']['historical_target_achieved']}`。前瞻目标是否达到：`False`。本研究不生成订单、不连接券商、不改变任何持仓。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def run_study(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    market, dividends, benchmark, oil, audit = load_inputs(config)
    factor = build_monthly_oil_factor(
        oil,
        int(config["oil_factor"]["lag_months"]),
        availability_column="availability_date",
    )
    robust_factor = build_monthly_oil_factor(
        oil,
        int(config["oil_factor"]["lag_months"]),
        availability_column="robust_availability_date",
    )
    signals = build_signal_schedule(market, factor, config)
    robust_signals = build_signal_schedule(market, robust_factor, config)
    mechanism = add_mechanism_targets(signals, market, dividends, config)
    mechanism_gate = evaluate_mechanism_gate(mechanism, config)
    portfolio: dict[str, Any] | None = None
    portfolio_artifacts: dict[str, pd.DataFrame] = {}
    if mechanism_gate["passed"]:
        portfolio, portfolio_artifacts = evaluate_portfolio(
            market,
            dividends,
            benchmark,
            signals,
            robust_signals,
            config,
            manifest,
        )
    report = build_report(
        config,
        manifest,
        audit,
        factor,
        signals,
        mechanism,
        mechanism_gate,
        portfolio,
    )
    if not write:
        return report
    result_path = _project_path(config["paths"]["result_json"])
    if result_path.exists():
        raise FileExistsError(f"正式结果已经存在，禁止覆盖：{result_path}")
    _atomic_parquet(_project_path(config["paths"]["monthly_factor_table"]), factor)
    _atomic_parquet(_project_path(config["paths"]["signal_table"]), signals)
    _atomic_parquet(_project_path(config["paths"]["mechanism_table"]), mechanism)
    for name, frame in portfolio_artifacts.items():
        _atomic_parquet(_project_path(config["paths"][name]), frame)
    _atomic_json(result_path, report)
    _atomic_text(_project_path(config["paths"]["result_markdown"]), render_markdown(report))
    return report
