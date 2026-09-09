"""交易所聚合 IVS 到 510300/现金二元配置的固定文献期限筛选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.intraday_binary_livermore_screen_v1 import (
    CostModel,
    build_benchmark_returns,
    simulate_candidate,
    summarize_period,
    trading_date_with_offset,
)
from research.option_cash_basis_feature_audit_v1 import construct_pair_basis


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_exchange_ivs_etf_binary_screen_v1.yaml"


def sha256_file(path: Path) -> str:
    """流式计算 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """加载并核对不可变的核心研究合同。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["study_id"] != "510300_EXCHANGE_IVS_ETF_BINARY_SCREEN_V1":
        raise ValueError("研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("协议不是冻结前实现完成状态")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("协议必须禁止参数营救")
    scope = config["scope"]
    if list(scope["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("持仓范围必须严格为510300和现金")
    if list(scope["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if bool(scope["derivatives_execution_allowed"]):
        raise ValueError("本筛选不得执行期权")
    objective = config["objective"]
    if objective["benchmark_id"] != "H00300":
        raise ValueError("主基准必须为H00300")
    if float(objective["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化超额硬门必须为20%")
    if float(objective["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动超额中位数硬门必须为20%")
    candidates = config["candidate_family"]["candidates"]
    if len(candidates) != int(config["candidate_family"]["candidate_count"]):
        raise ValueError("固定候选数量与列表不一致")
    expected_ids = {
        "IVS_WEEKLY_OOS_1W",
        "IVS_WEEKLY_OOS_2W",
        "IVS_WEEKLY_OOS_4W",
        "IVS_WEEKLY_OOS_8W",
        "IVS_MONTHLY_OOS_1M",
    }
    if {candidate["id"] for candidate in candidates} != expected_ids:
        raise ValueError("候选必须严格等于论文五个期限")
    if config["ivs_construction"]["forbidden_signal"] != "option_cash_basis":
        raise ValueError("必须明确禁止使用已失败的期权现货基差")
    return config


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """验证协议、程序、依赖和输入均与冻结时一致。"""

    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError("冻结清单不存在，禁止运行")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["study_id"] != config["protocol"]["study_id"]:
        raise ValueError("冻结清单研究编号不匹配")
    if sha256_file(CONFIG_PATH) != manifest["protocol_sha256"]:
        raise ValueError("冻结后协议发生漂移")
    for relative_path, expected in manifest["source_artifacts"].items():
        path = ROOT / relative_path
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"冻结后程序或测试漂移：{relative_path}")
    for relative_path, expected in manifest["input_artifacts"].items():
        path = ROOT / relative_path
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"冻结输入漂移：{relative_path}")
    return manifest


def _required_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}缺少字段：{sorted(missing)}")


def _pair_builder_config(config: dict[str, Any]) -> dict[str, Any]:
    """为锁定的认购认沽匹配函数构造只含特征规则的配置。"""

    feature = config["ivs_construction"]
    return {
        "dates": {
            "start": feature["feature_start"],
            "end": feature["feature_end"],
        },
        "pair_construction": {
            "option_types": ["C", "P"],
            "minimum_calendar_days_to_expiry": int(
                feature["minimum_calendar_days_to_expiry"]
            ),
            "adjusted_contracts_allowed": bool(
                feature["adjusted_contracts_allowed"]
            ),
            "pair_keys": [
                "trade_date",
                "expiry_date",
                "strike",
                "contract_unit",
                "is_adjusted",
            ],
            "require_positive_combined_open_interest": True,
            "no_arbitrage_bounds": {
                "enabled": True,
                "absolute_price_tolerance_cny": 0.00011,
            },
        },
        "risk_free_curve": {
            "maximum_backward_staleness_calendar_days": 7,
        },
        "independent_direction_check": {
            "implied_volatility_minimum": 0.005,
            "implied_volatility_maximum": 5.0,
        },
    }


def build_exchange_ivs_feature(
    option_eod: pd.DataFrame,
    risk_indicators: pd.DataFrame,
    rate_curve: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只从交易所IV差构造日频特征；已失败的基差列不会进入结果。"""

    pairs, pair_audit = construct_pair_basis(
        option_eod,
        risk_indicators,
        rate_curve,
        _pair_builder_config(config),
    )
    valid = pairs.loc[pairs["ivs_pair_valid"]].copy()
    if valid.empty:
        raise ValueError("没有可用于聚合的交易所IVS期权对")
    valid["weighted_ivs"] = valid["ivs_pair"] * valid["combined_open_interest"]
    feature = valid.groupby("trade_date", sort=True).agg(
        valid_pair_count=("ivs_pair", "size"),
        total_combined_open_interest=("combined_open_interest", "sum"),
        weighted_ivs_sum=("weighted_ivs", "sum"),
        minimum_dte_calendar_days=("dte_calendar_days", "min"),
        maximum_dte_calendar_days=("dte_calendar_days", "max"),
    )
    feature["exchange_ivs"] = (
        feature["weighted_ivs_sum"] / feature["total_combined_open_interest"]
    )
    feature = feature.reset_index()[
        [
            "trade_date",
            "exchange_ivs",
            "valid_pair_count",
            "total_combined_open_interest",
            "minimum_dte_calendar_days",
            "maximum_dte_calendar_days",
        ]
    ]
    feature = feature.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if not np.isfinite(feature["exchange_ivs"]).all():
        raise ValueError("日频交易所IVS含非有限值")
    if len(feature) < int(config["ivs_construction"]["required_feature_days"]):
        raise ValueError("日频交易所IVS覆盖不足")
    audit = {
        "pair_construction": pair_audit,
        "feature_days": int(len(feature)),
        "first_feature_date": str(feature["trade_date"].min().date()),
        "last_feature_date": str(feature["trade_date"].max().date()),
        "median_valid_pairs_per_day": float(feature["valid_pair_count"].median()),
        "ivs_minimum": float(feature["exchange_ivs"].min()),
        "ivs_median": float(feature["exchange_ivs"].median()),
        "ivs_maximum": float(feature["exchange_ivs"].max()),
        "basis_signal_columns_exported": 0,
    }
    return feature, audit


def _schedule_endpoints(
    feature: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_dates: pd.Series,
    schedule: str,
) -> pd.DataFrame:
    """生成周末或月末实际交易日端点。"""

    frame = feature[["trade_date", "exchange_ivs"]].merge(
        benchmark[["date", "close"]].rename(
            columns={"date": "trade_date", "close": "benchmark_close"}
        ),
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )
    positions = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(market_dates).dt.normalize(),
            "market_position": np.arange(len(market_dates), dtype=int),
        }
    )
    frame = frame.merge(positions, on="trade_date", how="inner", validate="one_to_one")
    if schedule == "WEEKLY_LAST_TRADING_DAY":
        frame["period_key"] = frame["trade_date"].dt.to_period("W-FRI").astype(str)
    elif schedule == "MONTHLY_LAST_TRADING_DAY":
        frame["period_key"] = frame["trade_date"].dt.to_period("M").astype(str)
    else:
        raise ValueError(f"未知预测日程：{schedule}")
    endpoints = (
        frame.sort_values("trade_date", kind="mergesort")
        .groupby("period_key", sort=True, as_index=False)
        .tail(1)
        .sort_values("trade_date", kind="mergesort")
        .reset_index(drop=True)
    )
    return endpoints


def expanding_ols_forecasts(
    endpoints: pd.DataFrame,
    *,
    candidate_id: str,
    schedule: str,
    horizon_periods: int,
    minimum_training_observations: int,
    cash_annual_rate: float,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """逐端点递归拟合，只允许已完全成熟的未来标签进入训练。"""

    required = {
        "trade_date",
        "exchange_ivs",
        "benchmark_close",
        "market_position",
    }
    if missing := required - set(endpoints.columns):
        raise ValueError(f"预测端点缺少字段：{sorted(missing)}")
    if horizon_periods <= 0 or minimum_training_observations <= 1:
        raise ValueError("预测期限或最小训练样本非法")
    frame = endpoints.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    x = frame["exchange_ivs"].to_numpy(float)
    closes = frame["benchmark_close"].to_numpy(float)
    market_positions = frame["market_position"].to_numpy(int)
    if not np.isfinite(x).all() or not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("预测端点的IVS或基准收盘价非法")
    label = np.full(len(frame), np.nan, dtype=float)
    label_end_index = np.full(len(frame), -1, dtype=int)
    cash_log_daily = np.log1p(cash_annual_rate / trading_days_per_year)
    for start_index in range(0, len(frame) - horizon_periods):
        end_index = start_index + horizon_periods
        elapsed_trading_days = int(
            market_positions[end_index] - market_positions[start_index]
        )
        if elapsed_trading_days <= 0:
            raise ValueError("预测标签交易日跨度必须为正")
        excess_log_return = (
            np.log(closes[end_index] / closes[start_index])
            - elapsed_trading_days * cash_log_daily
        )
        if schedule == "WEEKLY_LAST_TRADING_DAY":
            excess_log_return /= horizon_periods
        label[start_index] = excess_log_return
        label_end_index[start_index] = end_index

    rows: list[dict[str, Any]] = []
    for signal_index in range(len(frame)):
        eligible = np.flatnonzero(
            np.isfinite(label) & (label_end_index <= signal_index)
        )
        training_count = int(len(eligible))
        base = {
            "candidate_id": candidate_id,
            "schedule": schedule,
            "horizon_periods": int(horizon_periods),
            "signal_date": pd.Timestamp(frame.loc[signal_index, "trade_date"]),
            "exchange_ivs": float(x[signal_index]),
            "training_observations": training_count,
        }
        if training_count < minimum_training_observations:
            rows.append(
                {
                    **base,
                    "status": "NO_VIEW_INSUFFICIENT_MATURE_LABELS",
                    "intercept": np.nan,
                    "slope": np.nan,
                    "forecast_excess_log_return": np.nan,
                    "target_state": 0,
                    "maximum_training_label_end_date": pd.NaT,
                }
            )
            continue
        design = np.column_stack(
            [np.ones(training_count, dtype=float), x[eligible]]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, label[eligible], rcond=None)
        intercept = float(coefficients[0])
        slope = float(coefficients[1])
        forecast = float(intercept + slope * x[signal_index])
        maximum_label_end_index = int(label_end_index[eligible].max())
        maximum_label_end_date = pd.Timestamp(
            frame.loc[maximum_label_end_index, "trade_date"]
        )
        signal_date = pd.Timestamp(frame.loc[signal_index, "trade_date"])
        if maximum_label_end_date > signal_date:
            raise AssertionError("递归回归使用了信号日之后的标签")
        rows.append(
            {
                **base,
                "status": "MODEL_READY",
                "intercept": intercept,
                "slope": slope,
                "forecast_excess_log_return": forecast,
                "target_state": int(forecast > 0.0),
                "maximum_training_label_end_date": maximum_label_end_date,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_states(
    market: pd.DataFrame,
    forecasts: pd.DataFrame,
    candidate_ids: list[str],
) -> dict[str, np.ndarray]:
    """将端点预测扩展为每日收盘目标状态，成交仍发生在下一交易日开盘。"""

    dates = pd.to_datetime(market["trade_date"]).dt.normalize()
    states: dict[str, np.ndarray] = {}
    for candidate_id in candidate_ids:
        ready = forecasts.loc[
            forecasts["candidate_id"].eq(candidate_id)
            & forecasts["status"].eq("MODEL_READY")
        ].copy()
        ready_map = {
            pd.Timestamp(record["signal_date"]): int(record["target_state"])
            for record in ready.to_dict("records")
        }
        current = 0
        values = np.zeros(len(market), dtype=np.int8)
        for index, date in enumerate(dates):
            normalized = pd.Timestamp(date)
            if normalized in ready_map:
                current = ready_map[normalized]
            values[index] = current
        if not set(np.unique(values)).issubset({0, 1}):
            raise AssertionError("候选状态不是严格二元")
        states[candidate_id] = values
    return states


def _cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    costs = config["costs"]
    slippage_key = (
        "base_slippage_bps_per_leg"
        if scenario == "BASE"
        else "stress_slippage_bps_per_leg"
    )
    if scenario not in {"BASE", "STRESS"}:
        raise ValueError(f"未知成本情景：{scenario}")
    return CostModel(
        scenario=scenario,
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(costs[slippage_key]),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """哈希核对并加载冻结输入。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, contract in contracts.items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定输入不存在：{path}")
        actual = sha256_file(path)
        if actual != contract["required_sha256"]:
            raise ValueError(f"固定输入哈希漂移：{contract['file']}")
        paths[name] = path
        hashes[contract["file"]] = actual

    option_eod = pd.read_parquet(paths["option_eod"])
    risk = pd.read_parquet(paths["option_risk_indicators"])
    rates = pd.read_parquet(paths["government_bond_short_curve"])
    execution = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])

    _required_columns(
        execution,
        contracts["execution_daily"]["required_columns"],
        "510300日线",
    )
    _required_columns(
        benchmark,
        contracts["benchmark_total_return"]["required_columns"],
        "H00300全收益",
    )
    if set(execution["symbol"].dropna().unique()) != {
        contracts["execution_daily"]["required_symbol"]
    }:
        raise ValueError("510300日线标的代码不匹配")
    if set(benchmark["symbol"].dropna().unique()) != {
        contracts["benchmark_total_return"]["required_symbol"]
    }:
        raise ValueError("H00300基准代码不匹配")
    execution = execution.rename(columns={"date": "trade_date"}).copy()
    execution["trade_date"] = pd.to_datetime(execution["trade_date"]).dt.normalize()
    execution["bar_end"] = execution["trade_date"] + pd.Timedelta(hours=15)
    execution = execution.sort_values("trade_date", kind="mergesort").drop_duplicates(
        "trade_date", keep="last"
    )
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date", kind="mergesort").drop_duplicates(
        "date", keep="last"
    )
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column]).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("现金分红事件数量不匹配")
    if execution["trade_date"].duplicated().any() or benchmark["date"].duplicated().any():
        raise ValueError("执行或基准日期重复")
    if (execution[["open", "close"]] <= 0).any().any() or (benchmark["close"] <= 0).any():
        raise ValueError("执行或基准价格非法")

    feature_start = pd.Timestamp(config["ivs_construction"]["feature_start"])
    feature_end = pd.Timestamp(config["ivs_construction"]["feature_end"])
    execution = execution.loc[
        execution["trade_date"].between(feature_start, feature_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(feature_start, feature_end, inclusive="both")
    ].reset_index(drop=True)
    missing_benchmark = sorted(set(execution["trade_date"]) - set(benchmark["date"]))
    if missing_benchmark:
        raise ValueError(f"H00300缺少执行交易日：{missing_benchmark[:5]}")
    audit = {
        "hashes": hashes,
        "option_rows": int(len(option_eod)),
        "risk_rows": int(len(risk)),
        "rate_rows": int(len(rates)),
        "execution_rows": int(len(execution)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_events": int(len(dividends)),
        "first_execution_date": str(execution["trade_date"].min().date()),
        "last_execution_date": str(execution["trade_date"].max().date()),
    }
    return {
        "option_eod": option_eod,
        "risk": risk,
        "rates": rates,
        "execution": execution,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def build_all_forecasts(
    config: dict[str, Any],
    feature: pd.DataFrame,
    benchmark: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """一次性生成五个冻结期限的递归预测。"""

    pieces: list[pd.DataFrame] = []
    for candidate in config["candidate_family"]["candidates"]:
        endpoints = _schedule_endpoints(
            feature,
            benchmark,
            market["trade_date"],
            candidate["schedule"],
        )
        forecasts = expanding_ols_forecasts(
            endpoints,
            candidate_id=candidate["id"],
            schedule=candidate["schedule"],
            horizon_periods=int(candidate["horizon_periods"]),
            minimum_training_observations=int(
                candidate["minimum_training_observations"]
            ),
            cash_annual_rate=float(config["costs"]["cash_annual_rate"]),
            trading_days_per_year=int(
                config["objective"]["annualization_trading_days"]
            ),
        )
        pieces.append(forecasts)
    result = pd.concat(pieces, ignore_index=True)
    ready = result.loc[result["status"].eq("MODEL_READY")]
    if ready.empty:
        raise ValueError("五个期限均没有就绪预测")
    if (
        ready["maximum_training_label_end_date"] > ready["signal_date"]
    ).any():
        raise AssertionError("存在前视训练标签")
    return result.sort_values(
        ["candidate_id", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)


def evaluate_candidates(
    config: dict[str, Any],
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
    states: dict[str, np.ndarray],
) -> pd.DataFrame:
    """以统一账本评价成本、时期和起点扰动。"""

    evaluation = config["evaluation"]
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    benchmark_returns = build_benchmark_returns(benchmark)
    rows: list[dict[str, Any]] = []
    for candidate in config["candidate_family"]["candidates"]:
        candidate_id = candidate["id"]
        for scenario in evaluation["cost_scenarios"]:
            costs = _cost_model(config, scenario)
            for period in evaluation["periods"]:
                period_base_start = pd.Timestamp(period["start_date"])
                period_end = pd.Timestamp(period["end_date"])
                for offset in evaluation["start_date_perturbations_trading_days"]:
                    start_date = trading_date_with_offset(
                        market_dates, period_base_start, int(offset)
                    )
                    ledger, trades, diagnostics = simulate_candidate(
                        market,
                        dividends,
                        states[candidate_id],
                        start_date=start_date,
                        end_date=period_end,
                        costs=costs,
                        initial_capital=float(config["scope"]["initial_capital_cny"]),
                    )
                    summary = summarize_period(
                        ledger,
                        trades,
                        benchmark_returns,
                        period_id=period["id"],
                        start_date=start_date,
                        end_date=period_end,
                        objective=config["objective"],
                    )
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "schedule": candidate["schedule"],
                            "horizon_periods": int(candidate["horizon_periods"]),
                            "scenario": scenario,
                            "start_offset": int(offset),
                            **summary,
                            "t_plus_one_violations": int(
                                diagnostics["t_plus_one_violations"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """按预先声明的压力最差双门排序。"""

    rows: list[dict[str, Any]] = []
    for candidate_id, group in results.groupby("candidate_id", sort=True):
        stress = group.loc[group["scenario"].eq("STRESS")]
        base = group.loc[group["scenario"].eq("BASE")]
        if stress["rolling_242d_excess_median"].isna().any():
            rank_score = -np.inf
            stress_min_rolling = np.nan
        else:
            stress_min_rolling = float(stress["rolling_242d_excess_median"].min())
            stress_min_annual = float(stress["annualized_excess"].min())
            rank_score = min(stress_min_annual, stress_min_rolling)
        stress_min_annual = float(stress["annualized_excess"].min())
        hard_pass = bool(stress["both_20pct_gates"].all())
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            and (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        stress_combined = stress.loc[
            stress["period_id"].eq("COMBINED")
            & stress["start_offset"].eq(0)
        ].iloc[0]
        base_combined = base.loc[
            base["period_id"].eq("COMBINED")
            & base["start_offset"].eq(0)
        ].iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "schedule": str(stress.iloc[0]["schedule"]),
                "horizon_periods": int(stress.iloc[0]["horizon_periods"]),
                "rank_score": float(rank_score),
                "hard_pass": hard_pass,
                "positive_shadow": positive_shadow,
                "stress_min_annualized_excess": stress_min_annual,
                "stress_min_rolling_242d_excess_median": stress_min_rolling,
                "stress_combined_annualized_excess_offset0": float(
                    stress_combined["annualized_excess"]
                ),
                "stress_combined_rolling_median_offset0": float(
                    stress_combined["rolling_242d_excess_median"]
                ),
                "base_combined_annualized_excess_offset0": float(
                    base_combined["annualized_excess"]
                ),
                "base_combined_rolling_median_offset0": float(
                    base_combined["rolling_242d_excess_median"]
                ),
                "stress_trade_leg_count_total": int(stress["trade_leg_count"].sum()),
                "stress_execution_cost_cny_total": float(
                    stress["execution_cost_cny"].sum()
                ),
                "maximum_t_plus_one_violations": int(
                    group["t_plus_one_violations"].max()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "rank_score",
            "stress_trade_leg_count_total",
            "stress_execution_cost_cny_total",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"无法JSON序列化：{type(value).__name__}")


def _percent(value: float | None) -> str:
    return "不可计算" if value is None or not np.isfinite(value) else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    """渲染决策导向的中文报告。"""

    rows = []
    for item in report["candidate_summary"]:
        rows.append(
            "| {candidate} | {annual} | {rolling} | {combined} | {passed} |".format(
                candidate=item["candidate_id"],
                annual=_percent(item["stress_min_annualized_excess"]),
                rolling=_percent(item["stress_min_rolling_242d_excess_median"]),
                combined=_percent(item["stress_combined_annualized_excess_offset0"]),
                passed="是" if item["hard_pass"] else "否",
            )
        )
    table = "\n".join(rows)
    top = report["top_candidate"]
    return f"""# 510300 交易所聚合 IVS 二元筛选 V1

状态：`{report['status']}`

本筛选只执行 510300 与现金；期权只提供信息。已失败的期权—现货基差没有进入信号。

## 固定候选结果

| 候选 | 压力最差年化净超额 | 压力最差滚动242日中位数 | 压力综合期年化净超额 | 全门通过 |
|---|---:|---:|---:|---:|
{table}

## 结论

- 排名第一：`{top['candidate_id']}`。
- 压力最差年化净超额：{_percent(top['stress_min_annualized_excess'])}。
- 压力最差滚动242日超额中位数：{_percent(top['stress_min_rolling_242d_excess_median'])}。
- 硬门通过数：{report['hard_pass_candidate_count']} / {report['candidate_count']}。
- 决策：{report['decision']}
"""


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """执行冻结后唯一一次历史筛选。"""

    if config_path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许使用固定协议入口")
    config = load_config(config_path)
    manifest = verify_freeze_manifest(config)
    artifacts = config["artifacts"]
    report_path = ROOT / artifacts["report_json"]
    if report_path.exists():
        raise FileExistsError("固定筛选报告已存在，禁止重复运行")
    inputs, input_audit = load_inputs(config)
    feature, feature_audit = build_exchange_ivs_feature(
        inputs["option_eod"], inputs["risk"], inputs["rates"], config
    )
    forecasts = build_all_forecasts(
        config, feature, inputs["benchmark"], inputs["execution"]
    )
    candidate_ids = [
        candidate["id"] for candidate in config["candidate_family"]["candidates"]
    ]
    states = build_candidate_states(inputs["execution"], forecasts, candidate_ids)
    period_results = evaluate_candidates(
        config,
        inputs["execution"],
        inputs["benchmark"],
        inputs["dividends"],
        states,
    )
    summary = summarize_candidates(period_results)
    hard_pass_count = int(summary["hard_pass"].sum())
    positive_count = int(summary["positive_shadow"].sum())
    top = summary.iloc[0].to_dict()
    status = (
        config["governance"]["result_label_if_passed"]
        if hard_pass_count > 0
        else config["governance"]["result_label_if_rejected"]
    )

    output_dir = ROOT / artifacts["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = ROOT / artifacts["ivs_feature_parquet"]
    forecasts_path = ROOT / artifacts["forecasts_parquet"]
    summary_path = ROOT / artifacts["candidate_summary_parquet"]
    results_path = ROOT / artifacts["period_results_parquet"]
    feature.to_parquet(feature_path, index=False)
    forecasts.to_parquet(forecasts_path, index=False)
    summary.to_parquet(summary_path, index=False)
    period_results.to_parquet(results_path, index=False)

    top_candidate_id = str(top["candidate_id"])
    combined = next(
        period for period in config["evaluation"]["periods"] if period["id"] == "COMBINED"
    )
    top_cost = _cost_model(config, "STRESS")
    top_ledger, top_trades, top_diagnostics = simulate_candidate(
        inputs["execution"],
        inputs["dividends"],
        states[top_candidate_id],
        start_date=pd.Timestamp(combined["start_date"]),
        end_date=pd.Timestamp(combined["end_date"]),
        costs=top_cost,
        initial_capital=float(config["scope"]["initial_capital_cny"]),
    )
    top_ledger_path = ROOT / artifacts["top_candidate_daily_ledger_parquet"]
    top_trades_path = ROOT / artifacts["top_candidate_trades_parquet"]
    top_ledger.to_parquet(top_ledger_path, index=False)
    top_trades.to_parquet(top_trades_path, index=False)

    forecast_audit: dict[str, Any] = {}
    for candidate_id, group in forecasts.groupby("candidate_id", sort=True):
        ready = group.loc[group["status"].eq("MODEL_READY")]
        forecast_audit[candidate_id] = {
            "schedule_rows": int(len(group)),
            "model_ready_rows": int(len(ready)),
            "no_view_rows": int((group["status"] != "MODEL_READY").sum()),
            "first_ready_signal_date": None
            if ready.empty
            else str(ready["signal_date"].min().date()),
            "last_ready_signal_date": None
            if ready.empty
            else str(ready["signal_date"].max().date()),
            "negative_slope_share": None
            if ready.empty
            else float((ready["slope"] < 0.0).mean()),
            "full_state_forecast_share": None
            if ready.empty
            else float(ready["target_state"].mean()),
            "maximum_training_label_end_le_signal": bool(
                ready.empty
                or (
                    ready["maximum_training_label_end_date"] <= ready["signal_date"]
                ).all()
            ),
        }

    output_files = [
        feature_path,
        forecasts_path,
        summary_path,
        results_path,
        top_ledger_path,
        top_trades_path,
    ]
    report = {
        "status": status,
        "study_id": config["protocol"]["study_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "freeze_manifest_sha256": sha256_file(
            ROOT / artifacts["freeze_manifest"]
        ),
        "protocol_sha256": manifest["protocol_sha256"],
        "scope": {
            "execution_asset": "510300.SH",
            "information_asset": "510300 ETF options",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "historical_evidence_label": config["protocol"][
                "historical_evidence_label"
            ],
            "basis_signal_used": False,
        },
        "input_audit": input_audit,
        "feature_audit": feature_audit,
        "forecast_audit": forecast_audit,
        "candidate_count": int(len(summary)),
        "hard_pass_candidate_count": hard_pass_count,
        "positive_shadow_candidate_count": positive_count,
        "top_candidate": top,
        "candidate_summary": summary.to_dict("records"),
        "period_result_rows": int(len(period_results)),
        "maximum_t_plus_one_violations": int(
            period_results["t_plus_one_violations"].max()
        ),
        "top_candidate_diagnostics": top_diagnostics,
        "artifacts": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in output_files
        },
        "decision": (
            "存在回溯全门候选；只允许另行冻结246交易日前瞻验证。"
            if hard_pass_count > 0
            else "五个论文期限均未通过；整族永久拒绝且不得结果后改期限或门槛。"
        ),
        "governance": config["governance"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    markdown_path = ROOT / artifacts["report_markdown"]
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "hard_pass_candidate_count": hard_pass_count,
                "positive_shadow_candidate_count": positive_count,
                "top_candidate": top,
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="固定协议路径",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
