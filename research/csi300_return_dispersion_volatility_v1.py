"""510300 沪深300横截面收益离散度波动预算 V1。

本模块严格先检验横截面收益离散度能否增量预测下一自然月实现方差。
只有机制门全部通过，才允许构建510300/现金组合并计算净费后夏普率。
历史结果从不授权 Paper、Shadow、订单、券商连接或实盘。
"""

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

from backtest.engine import BacktestCosts, run_long_cash_backtest, summarize_backtest  # noqa: E402


CONFIG_PATH = ROOT / "config" / "510300_csi300_return_dispersion_volatility_v1.yaml"
MANIFEST_PATH = (
    ROOT / "config" / "510300_csi300_return_dispersion_volatility_v1r1_manifest.json"
)


class ContractError(RuntimeError):
    """冻结协议、清单或输入数据不满足要求。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"缺少协议文件：{path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ContractError("协议文件必须是YAML对象")
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    dates = config["dates"]
    contract = config["data_contract"]
    factor = config["factor"]
    model = config["mechanism_model"]
    allocation = config["allocation"]
    portfolio = config["portfolio"]
    governance = config["governance"]

    if protocol["project_id"] != "510300_CSI300_RETURN_DISPERSION_VOLATILITY_V1R1":
        raise ContractError("project_id不符合冻结候选")
    if protocol["candidate_model_id"] != "510300_CSI300_RETURN_DISPERSION_VOLATILITY_V1R1":
        raise ContractError("candidate_model_id不符合预结果修正版")
    if protocol["revision_kind"] != "PRE_RESULT_INPUT_ASSERTION_CORRECTION_ONLY":
        raise ContractError("预结果修正类型漂移")
    if protocol["supersedes_project_id"] != "510300_CSI300_RETURN_DISPERSION_VOLATILITY_V1":
        raise ContractError("被替代候选标识漂移")
    if protocol["superseded_attempt_future_outcomes_computed"] is not False:
        raise ContractError("被替代尝试不得已计算未来波动结果")
    if protocol["superseded_attempt_portfolio_returns_computed"] is not False:
        raise ContractError("被替代尝试不得已计算组合收益")
    if protocol["one_shot"] is not True:
        raise ContractError("研究必须声明one_shot=true")
    if protocol["historical_510300_data_already_contaminated"] is not True:
        raise ContractError("必须承认510300历史价格已经在既有研究中被观察")
    if protocol["candidate_factor_values_read_before_freeze"] is not True:
        raise ContractError("修正版必须承认冻结前已由失败尝试读取因子文件")
    for key in (
        "candidate_future_volatility_outcomes_read_before_freeze",
        "candidate_portfolio_returns_read_before_freeze",
        "pristine_blind_holdout_claim_allowed",
    ):
        if protocol[key] is not False:
            raise ContractError(f"协议字段{key}必须固定为false")
    if scope["execution_asset"] != "510300.SH":
        raise ContractError("执行资产必须固定为510300.SH")
    if scope["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("允许持有范围必须仅为510300与现金")
    if any(
        [
            scope["leverage_allowed"],
            scope["short_selling_allowed"],
            scope["derivatives_execution_allowed"],
            scope["live_trading_authorized"],
        ]
    ):
        raise ContractError("禁止杠杆、卖空、衍生品执行和实盘")
    if dates["evaluation_start"] != "2015-01-05":
        raise ContractError("评价起点必须固定为2015-01-05")
    if dates["evaluation_end"] != "2026-08-14":
        raise ContractError("评价终点必须固定为2026-08-14")
    if int(contract["expected_factor_months"]) != 255:
        raise ContractError("因子月份数必须固定为255")
    if (
        int(contract["early_active_member_count_minimum"]) != 300
        or int(contract["early_active_member_count_maximum"]) != 301
    ):
        raise ContractError("早期月末点时成分数量边界必须固定为300至301")
    if int(contract["post_2015_panel_active_members_per_month_end"]) != 300:
        raise ContractError("2015年后面板月末点时成分数必须固定为300")
    if contract["historical_membership_null_opt_in_policy"] != (
        "EXCLUDE_UNRESOLVED_SOURCE_ROWS_NO_IMPUTATION"
    ):
        raise ContractError("历史成员空纳入日期必须固定为排除且不插值")
    expected_unresolved = [
        {"symbol": "600312.SH", "opt_out": "2012-01-01"},
        {"symbol": "600501.SH", "opt_out": "2008-06-14"},
        {"symbol": "600549.SH", "opt_out": "2019-06-17"},
        {"symbol": "600786.SH", "opt_out": "2008-06-14"},
    ]
    if contract["expected_unresolved_null_opt_in_intervals"] != expected_unresolved:
        raise ContractError("历史成员4条未决空纳入日期记录锁定值漂移")
    if int(contract["minimum_valid_member_count"]) != 285:
        raise ContractError("有效成分最低数量必须固定为285")
    if not math.isclose(float(contract["minimum_valid_member_coverage"]), 0.95):
        raise ContractError("有效成分最低覆盖率必须固定为95%")
    if factor["active_factor_count"] != 1:
        raise ContractError("活跃因子数量必须固定为1")
    if factor["id"] != "CSI300_POINT_IN_TIME_MONTHLY_EQUAL_WEIGHT_RETURN_DISPERSION":
        raise ContractError("横截面离散度因子标识不得改变")
    if factor["threshold_selection"] != "none" or factor["winsorization"] != "none":
        raise ContractError("第一版不得选择阈值或缩尾")
    if model["estimation"] != "expanding_ordinary_least_squares":
        raise ContractError("机制模型必须固定为扩展窗OLS")
    if int(model["minimum_training_pairs"]) != 96:
        raise ContractError("初始训练配对数必须固定为96")
    if allocation["baseline_only_allocation"] != "forbidden":
        raise ContractError("离散度失败后禁止基线单独进入组合")
    if not math.isclose(float(allocation["risk_aversion_gamma"]), 5.0):
        raise ContractError("风险厌恶系数必须固定为5")
    if float(portfolio["initial_capital_cny"]) != 20000.0:
        raise ContractError("初始资金必须固定为20000元")
    if int(portfolio["lot_size_shares"]) != 100:
        raise ContractError("交易单位必须固定为100份")
    if float(portfolio["base_costs"]["minimum_commission_cny_per_leg"]) != 5.0:
        raise ContractError("最低单边佣金必须固定为5元")
    if float(portfolio["base_costs"]["slippage_bps_per_leg"]) != 5.0:
        raise ContractError("基准单边滑点必须固定为5bp")
    if float(portfolio["stress_costs"]["slippage_bps_per_leg"]) != 10.0:
        raise ContractError("压力单边滑点必须固定为10bp")
    if governance["order_generation"] or governance["broker_connection"]:
        raise ContractError("研究协议禁止订单生成和券商连接")
    if governance["position_change"] or governance["live_trading_authorized"]:
        raise ContractError("研究协议禁止真实仓位变化和实盘")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("缺少冻结清单；禁止在冻结前读取候选结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("冻结清单状态无效")
    if manifest.get("result_preexisted_at_freeze") is not False:
        raise ContractError("冻结时不得已经存在结果")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后协议哈希发生变化")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件或输入发生漂移：{mismatches}")
    selection = manifest.get("selection_bias_control", {})
    if int(selection.get("total_trial_count_including_current", 0)) < 1:
        raise ContractError("冻结清单缺少选择偏差试验计数")
    return manifest


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ContractError(f"{label}缺少字段：{missing}")


def _load_audit(specification: dict[str, Any], label: str) -> dict[str, Any]:
    path = project_path(specification["path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    required = specification["required_status"]
    if document.get("status") != required:
        raise ContractError(f"{label}状态不是{required}：{document.get('status')}")
    return document


def _validate_factor_active_member_counts(
    factor: pd.DataFrame, config: dict[str, Any]
) -> None:
    required = {"factor_source_segment", "active_member_count"}
    missing = sorted(required.difference(factor.columns))
    if missing:
        raise ContractError(f"月度离散度因子缺少成员数量校验字段：{missing}")
    early_segment = factor["factor_source_segment"].eq("EARLY_SINA_RECONSTRUCTION")
    allowed_segments = {
        "EARLY_SINA_RECONSTRUCTION",
        "EXISTING_SINA_EXTERNAL_PANEL",
        "EXISTING_SINA_BACKUP_PANEL",
    }
    observed_segments = set(factor["factor_source_segment"].astype(str).unique())
    if observed_segments != allowed_segments:
        raise ContractError("月度离散度因子来源区段不符合冻结契约")
    contract = config["data_contract"]
    early_minimum = int(contract["early_active_member_count_minimum"])
    early_maximum = int(contract["early_active_member_count_maximum"])
    if not factor.loc[early_segment, "active_member_count"].between(
        early_minimum, early_maximum
    ).all():
        raise ContractError("早期月末点时成分数不在300至301的冻结边界")
    post_2015_members = int(contract["post_2015_panel_active_members_per_month_end"])
    if not factor.loc[~early_segment, "active_member_count"].eq(post_2015_members).all():
        raise ContractError("2015年后月末点时成分数不是300")


def load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specifications = config["inputs"]
    for name, specification in specifications.items():
        path = project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"缺少输入{name}：{path}")

    _load_audit(specifications["historical_membership_audit"], "历史成员审计")
    _load_audit(specifications["existing_2014_2021_panel_audit"], "2014至2021成分面板审计")
    _load_audit(specifications["etf_input_audit"], "510300行情审计")
    input_audit = _load_audit(specifications["input_audit"], "离散度输入审计")
    if input_audit.get("candidate_future_volatility_outcomes_computed") is not False:
        raise ContractError("输入构建阶段不得计算候选未来波动结果")
    if input_audit.get("candidate_portfolio_returns_computed") is not False:
        raise ContractError("输入构建阶段不得计算候选组合收益")

    factor_spec = specifications["monthly_dispersion_factor"]
    factor = pd.read_parquet(project_path(factor_spec["path"]))
    _require_columns(factor, list(factor_spec["required_columns"]), "月度离散度因子")
    factor["month_end_date"] = pd.to_datetime(factor["month_end_date"], errors="coerce")
    factor["previous_month_end_date"] = pd.to_datetime(
        factor["previous_month_end_date"], errors="coerce"
    )
    numeric_factor = [
        "active_member_count",
        "valid_return_count",
        "valid_return_coverage",
        "equal_weight_mean_monthly_return",
        "cross_sectional_return_dispersion",
        "cross_sectional_return_dispersion_squared",
    ]
    factor[numeric_factor] = factor[numeric_factor].apply(pd.to_numeric, errors="coerce")
    if factor[["factor_month", "month_end_date", *numeric_factor]].isna().any().any():
        raise ContractError("月度离散度因子存在空值")
    factor = factor.sort_values("factor_month").reset_index(drop=True)
    if factor["factor_month"].duplicated().any():
        raise ContractError("月度离散度因子月份重复")
    expected_months = pd.period_range(
        config["data_contract"]["expected_first_factor_month"],
        config["data_contract"]["expected_last_factor_month"],
        freq="M",
    ).astype(str)
    if factor["factor_month"].tolist() != expected_months.tolist():
        raise ContractError("月度离散度因子月份不连续或端点不符")
    if len(factor) != int(config["data_contract"]["expected_factor_months"]):
        raise ContractError("月度离散度因子行数不符合冻结契约")
    _validate_factor_active_member_counts(factor, config)
    if (factor["valid_return_count"] < int(config["data_contract"]["minimum_valid_member_count"])).any():
        raise ContractError("存在月份有效成分收益数量不足")
    if (
        factor["valid_return_coverage"]
        < float(config["data_contract"]["minimum_valid_member_coverage"])
    ).any():
        raise ContractError("存在月份有效成分收益覆盖率不足")
    if (factor["cross_sectional_return_dispersion"] <= 0).any():
        raise ContractError("横截面收益离散度必须为正数")
    if not np.allclose(
        factor["cross_sectional_return_dispersion_squared"],
        factor["cross_sectional_return_dispersion"] ** 2,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ContractError("离散度平方不能由离散度精确重建")

    index_spec = specifications["csi300_price_index"]
    index_daily = pd.read_parquet(project_path(index_spec["path"]))
    _require_columns(index_daily, list(index_spec["required_columns"]), "沪深300价格指数")
    index_daily["date"] = pd.to_datetime(index_daily["date"], errors="coerce")
    index_daily[["open", "high", "low", "close", "volume"]] = index_daily[
        ["open", "high", "low", "close", "volume"]
    ].apply(pd.to_numeric, errors="coerce")
    index_daily = index_daily.sort_values("date").reset_index(drop=True)
    if index_daily[["date", "open", "high", "low", "close"]].isna().any().any():
        raise ContractError("沪深300价格指数存在空值")
    if index_daily["date"].duplicated().any():
        raise ContractError("沪深300价格指数日期重复")
    if (index_daily[["open", "high", "low", "close"]] <= 0).any().any():
        raise ContractError("沪深300价格指数价格必须为正数")
    if (index_daily["high"] < index_daily[["open", "close", "low"]].max(axis=1)).any():
        raise ContractError("沪深300价格指数最高价不满足OHLC约束")
    if (index_daily["low"] > index_daily[["open", "close", "high"]].min(axis=1)).any():
        raise ContractError("沪深300价格指数最低价不满足OHLC约束")
    if index_daily["date"].min() > pd.Timestamp(
        config["data_contract"]["csi300_index"]["required_first_date_on_or_before"]
    ):
        raise ContractError("沪深300价格指数起点过晚")
    if index_daily["date"].max() < pd.Timestamp(
        config["data_contract"]["csi300_index"]["required_last_date_on_or_after"]
    ):
        raise ContractError("沪深300价格指数终点过早")

    etf_spec = specifications["etf_daily"]
    market = pd.read_parquet(project_path(etf_spec["path"]))
    _require_columns(market, list(etf_spec["required_columns"]), "510300行情")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market[["open", "high", "low", "close", "volume", "amount"]] = market[
        ["open", "high", "low", "close", "volume", "amount"]
    ].apply(pd.to_numeric, errors="coerce")
    market = market.sort_values("date").reset_index(drop=True)
    if market[["date", "open", "high", "low", "close"]].isna().any().any():
        raise ContractError("510300行情存在空值")
    if market["date"].duplicated().any():
        raise ContractError("510300行情日期重复")
    evaluation_dates = market.loc[
        market["date"].between(
            pd.Timestamp(config["dates"]["evaluation_start"]),
            pd.Timestamp(config["dates"]["evaluation_end"]),
        ),
        "date",
    ]
    if evaluation_dates.empty:
        raise ContractError("510300评价区间为空")
    if evaluation_dates.iloc[0] != pd.Timestamp(config["dates"]["evaluation_start"]):
        raise ContractError("510300评价起点不精确")
    if evaluation_dates.iloc[-1] != pd.Timestamp(config["dates"]["evaluation_end"]):
        raise ContractError("510300评价终点不精确")

    dividend_spec = specifications["dividends"]
    dividends = pd.read_csv(project_path(dividend_spec["path"]))
    _require_columns(dividends, list(dividend_spec["required_columns"]), "510300分红")
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="coerce")
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="coerce"
    )
    if dividends[
        ["record_date", "ex_date", "payment_date", "cash_dividend_per_share"]
    ].isna().any().any():
        raise ContractError("510300分红存在空值")
    if (dividends["cash_dividend_per_share"] < 0).any():
        raise ContractError("510300分红金额不能为负数")
    if not (
        (dividends["record_date"] <= dividends["ex_date"])
        & (dividends["ex_date"] <= dividends["payment_date"])
    ).all():
        raise ContractError("510300分红日期顺序异常")

    benchmark_spec = specifications["benchmark_total_return"]
    benchmark = pd.read_parquet(project_path(benchmark_spec["path"]))
    _require_columns(benchmark, list(benchmark_spec["required_columns"]), "H00300基准")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "close"]).sort_values("date")
    benchmark = benchmark.drop_duplicates("date", keep="last").reset_index(drop=True)
    if (benchmark["close"] <= 0).any():
        raise ContractError("H00300基准价格必须为正数")
    benchmark_eval = benchmark.loc[
        benchmark["date"].between(
            pd.Timestamp(config["dates"]["evaluation_start"]),
            pd.Timestamp(config["dates"]["evaluation_end"]),
        )
    ]
    if benchmark_eval.empty:
        raise ContractError("H00300评价区间为空")
    if benchmark_eval["date"].iloc[0] != pd.Timestamp(config["dates"]["evaluation_start"]):
        raise ContractError("H00300评价起点不精确")
    if benchmark_eval["date"].iloc[-1] != pd.Timestamp(config["dates"]["evaluation_end"]):
        raise ContractError("H00300评价终点不精确")
    return factor, index_daily, market, dividends, benchmark, input_audit


def monthly_realized_variance(index_daily: pd.DataFrame) -> pd.DataFrame:
    frame = index_daily[["date", "close"]].copy().sort_values("date")
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for month, subset in frame.groupby("month", sort=True):
        valid = subset["log_return"].dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "month": str(month),
                "month_end_date": pd.Timestamp(subset["date"].max()),
                "trading_return_count": int(len(valid)),
                "realized_variance": float(np.square(valid.to_numpy(dtype=float)).sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or (result["realized_variance"] <= 0).any():
        raise ContractError("沪深300月度实现方差不可用")
    return result


def build_mechanism_panel(
    factor: pd.DataFrame,
    index_daily: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    variance = monthly_realized_variance(index_daily)
    variance_by_month = variance.set_index("month")
    rows: list[dict[str, Any]] = []
    last_complete_target = pd.Period(
        config["dates"]["mechanism_evaluation_last_target_month"], freq="M"
    )
    for observation in factor.itertuples(index=False):
        current_period = pd.Period(str(observation.factor_month), freq="M")
        target_period = current_period + 1
        current_key = str(current_period)
        target_key = str(target_period)
        if current_key not in variance_by_month.index:
            raise ContractError(f"因子月{current_key}缺少当前实现方差")
        current = variance_by_month.loc[current_key]
        target_variance = np.nan
        target_end = pd.NaT
        target_count = 0
        if target_period <= last_complete_target:
            if target_key not in variance_by_month.index:
                raise ContractError(f"完整目标月{target_key}缺少实现方差")
            target = variance_by_month.loc[target_key]
            target_variance = float(target["realized_variance"])
            target_end = pd.Timestamp(target["month_end_date"])
            target_count = int(target["trading_return_count"])
        rows.append(
            {
                "factor_month": current_key,
                "decision_date": pd.Timestamp(observation.month_end_date),
                "target_month": target_key,
                "target_month_end_date": target_end,
                "current_realized_variance": float(current["realized_variance"]),
                "current_trading_return_count": int(current["trading_return_count"]),
                "cross_sectional_return_dispersion": float(
                    observation.cross_sectional_return_dispersion
                ),
                "cross_sectional_return_dispersion_squared": float(
                    observation.cross_sectional_return_dispersion_squared
                ),
                "next_month_realized_variance": target_variance,
                "next_month_trading_return_count": target_count,
            }
        )
    panel = pd.DataFrame(rows)
    panel["log_current_realized_variance"] = np.log(panel["current_realized_variance"])
    panel["log_dispersion_squared"] = np.log(
        panel["cross_sectional_return_dispersion_squared"]
    )
    panel["log_next_month_realized_variance"] = np.log(
        panel["next_month_realized_variance"]
    )
    if not np.isfinite(
        panel[["log_current_realized_variance", "log_dispersion_squared"]].to_numpy()
    ).all():
        raise ContractError("机制预测变量的对数不可用")
    return panel


def _ols_fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ContractError("OLS输入形状无效")
    if len(y) <= x.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContractError("OLS输入观测不足或存在非有限值")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    return coefficients, residuals


def _newey_west_regression(
    target: np.ndarray,
    regressors: np.ndarray,
    *,
    max_lag: int,
    coefficient_names: list[str],
) -> dict[str, Any]:
    y = np.asarray(target, dtype=float)
    x = np.asarray(regressors, dtype=float)
    coefficients, residuals = _ols_fit(x, y)
    inverse = np.linalg.pinv(x.T @ x)
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    estimates: dict[str, dict[str, float | None]] = {}
    for index, name in enumerate(coefficient_names):
        estimate = float(coefficients[index])
        standard_error = float(standard_errors[index])
        estimates[name] = {
            "coefficient": estimate,
            "standard_error": standard_error if standard_error > 0 else None,
            "t_stat": estimate / standard_error if standard_error > 0 else None,
        }
    return {
        "observations": int(len(y)),
        "max_lag": int(max_lag),
        "coefficients": estimates,
        "r_squared": _optional_float(
            1.0 - float(np.dot(residuals, residuals)) / float(np.dot(y - y.mean(), y - y.mean()))
            if float(np.dot(y - y.mean(), y - y.mean())) > 0
            else None
        ),
    }


def expanding_forecasts(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    minimum = int(config["mechanism_model"]["minimum_training_pairs"])
    first_decision = pd.Period("2014-12", freq="M")
    last_decision = pd.Period(config["dates"]["last_complete_factor_month"], freq="M")
    rows: list[dict[str, Any]] = []
    indexed = panel.set_index("factor_month")
    panel_period = panel.copy()
    panel_period["target_period"] = pd.PeriodIndex(panel_period["target_month"], freq="M")
    for decision_period in pd.period_range(first_decision, last_decision, freq="M"):
        key = str(decision_period)
        if key not in indexed.index:
            raise ContractError(f"决策月{key}缺少因子")
        current = indexed.loc[key]
        training = panel_period.loc[
            (panel_period["target_period"] <= decision_period)
            & panel_period["log_next_month_realized_variance"].notna()
        ].copy()
        if len(training) < minimum:
            raise ContractError(f"决策月{key}训练配对不足{minimum}")
        y = training["log_next_month_realized_variance"].to_numpy(dtype=float)
        baseline_x = np.column_stack(
            [
                np.ones(len(training)),
                training["log_current_realized_variance"].to_numpy(dtype=float),
            ]
        )
        augmented_x = np.column_stack(
            [
                np.ones(len(training)),
                training["log_current_realized_variance"].to_numpy(dtype=float),
                training["log_dispersion_squared"].to_numpy(dtype=float),
            ]
        )
        baseline_coefficients, _ = _ols_fit(baseline_x, y)
        augmented_coefficients, _ = _ols_fit(augmented_x, y)
        current_baseline = np.array(
            [1.0, float(current["log_current_realized_variance"])], dtype=float
        )
        current_augmented = np.array(
            [
                1.0,
                float(current["log_current_realized_variance"]),
                float(current["log_dispersion_squared"]),
            ],
            dtype=float,
        )
        baseline_log_forecast = float(current_baseline @ baseline_coefficients)
        augmented_log_forecast = float(current_augmented @ augmented_coefficients)
        floor = float(config["mechanism_model"]["forecast_floor"])
        baseline_forecast = max(math.exp(baseline_log_forecast), floor)
        augmented_forecast = max(math.exp(augmented_log_forecast), floor)
        rows.append(
            {
                "factor_month": key,
                "decision_date": pd.Timestamp(current["decision_date"]),
                "target_month": str(current["target_month"]),
                "target_month_end_date": current["target_month_end_date"],
                "training_pairs": int(len(training)),
                "log_current_realized_variance": float(
                    current["log_current_realized_variance"]
                ),
                "log_dispersion_squared": float(current["log_dispersion_squared"]),
                "next_month_realized_variance": _optional_float(
                    current["next_month_realized_variance"]
                ),
                "log_next_month_realized_variance": _optional_float(
                    current["log_next_month_realized_variance"]
                ),
                "baseline_log_variance_forecast": baseline_log_forecast,
                "augmented_log_variance_forecast": augmented_log_forecast,
                "baseline_variance_forecast": baseline_forecast,
                "augmented_variance_forecast": augmented_forecast,
                "baseline_intercept": float(baseline_coefficients[0]),
                "baseline_beta_current_rv": float(baseline_coefficients[1]),
                "augmented_intercept": float(augmented_coefficients[0]),
                "augmented_beta_current_rv": float(augmented_coefficients[1]),
                "augmented_beta_dispersion": float(augmented_coefficients[2]),
            }
        )
    forecasts = pd.DataFrame(rows)
    if forecasts["factor_month"].duplicated().any():
        raise ContractError("扩展窗预测月份重复")
    return forecasts


def _dm_t_stat(loss_difference: np.ndarray, max_lag: int) -> float | None:
    values = np.asarray(loss_difference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= max_lag + 2:
        return None
    centered = values - values.mean()
    n = len(values)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return None
    return float(values.mean() / math.sqrt(long_run_variance / n))


def _circular_block_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block_count = math.ceil(length / block_length)
    starts = rng.integers(0, length, size=block_count)
    offsets = np.arange(block_length, dtype=int)
    return ((starts[:, None] + offsets[None, :]) % length).ravel()[:length]


def _bootstrap_beta_dispersion(
    evaluation: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["mechanism_evaluation"]
    repetitions = int(specification["bootstrap_repetitions"])
    block_length = int(specification["bootstrap_block_length_months"])
    confidence = float(specification["bootstrap_confidence_level"])
    seed = int(specification["random_seed"])
    y = evaluation["log_next_month_realized_variance"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(evaluation)),
            evaluation["log_current_realized_variance"].to_numpy(dtype=float),
            evaluation["log_dispersion_squared"].to_numpy(dtype=float),
        ]
    )
    rng = np.random.default_rng(seed)
    coefficients: list[float] = []
    for _ in range(repetitions):
        indices = _circular_block_indices(len(evaluation), block_length, rng)
        try:
            estimate, _ = _ols_fit(x[indices], y[indices])
        except ContractError:
            continue
        if math.isfinite(float(estimate[2])):
            coefficients.append(float(estimate[2]))
    alpha = (1.0 - confidence) / 2.0
    interval = (
        [
            float(np.quantile(coefficients, alpha)),
            float(np.quantile(coefficients, 1.0 - alpha)),
        ]
        if coefficients
        else [None, None]
    )
    return {
        "method": specification["bootstrap_method"],
        "repetitions": repetitions,
        "valid_draws": int(len(coefficients)),
        "block_length_months": block_length,
        "confidence_level": confidence,
        "confidence_interval": interval,
        "seed": seed,
    }


def _structural_period(date: pd.Timestamp, config: dict[str, Any]) -> str | None:
    for name, bounds in config["dates"]["structural_periods"].items():
        if pd.Timestamp(bounds["start"]) <= date <= pd.Timestamp(bounds["end"]):
            return str(name)
    return None


def _mechanism_subset_summary(
    subset: pd.DataFrame,
    *,
    max_lag: int,
) -> dict[str, Any]:
    y = subset["log_next_month_realized_variance"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(len(subset)),
            subset["log_current_realized_variance"].to_numpy(dtype=float),
            subset["log_dispersion_squared"].to_numpy(dtype=float),
        ]
    )
    regression = _newey_west_regression(
        y,
        x,
        max_lag=max_lag,
        coefficient_names=["intercept", "log_current_rv", "log_dispersion_squared"],
    )
    improvement = float(subset["qlike_loss_difference"].mean())
    return {
        "observations": int(len(subset)),
        "newey_west_augmented_regression": regression,
        "beta_dispersion": regression["coefficients"]["log_dispersion_squared"],
        "mean_qlike_baseline": float(subset["qlike_baseline"].mean()),
        "mean_qlike_augmented": float(subset["qlike_augmented"].mean()),
        "mean_qlike_improvement": improvement,
        "qlike_dm_t_stat": _dm_t_stat(
            subset["qlike_loss_difference"].to_numpy(dtype=float), max_lag
        ),
        "mean_log_variance_squared_error_baseline": float(
            subset["log_variance_squared_error_baseline"].mean()
        ),
        "mean_log_variance_squared_error_augmented": float(
            subset["log_variance_squared_error_augmented"].mean()
        ),
    }


def evaluate_mechanism_gate(
    forecasts: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    first = pd.Period(config["dates"]["mechanism_evaluation_first_target_month"], freq="M")
    last = pd.Period(config["dates"]["mechanism_evaluation_last_target_month"], freq="M")
    evaluation = forecasts.copy()
    evaluation["target_period"] = pd.PeriodIndex(evaluation["target_month"], freq="M")
    evaluation = evaluation.loc[
        evaluation["target_period"].between(first, last)
        & evaluation["next_month_realized_variance"].notna()
    ].copy()
    if evaluation.empty:
        raise ContractError("机制样本外评价区间为空")
    actual = evaluation["next_month_realized_variance"].to_numpy(dtype=float)
    baseline = evaluation["baseline_variance_forecast"].to_numpy(dtype=float)
    augmented = evaluation["augmented_variance_forecast"].to_numpy(dtype=float)
    evaluation["qlike_baseline"] = actual / baseline + np.log(baseline)
    evaluation["qlike_augmented"] = actual / augmented + np.log(augmented)
    evaluation["qlike_loss_difference"] = (
        evaluation["qlike_baseline"] - evaluation["qlike_augmented"]
    )
    evaluation["log_variance_squared_error_baseline"] = np.square(
        evaluation["log_next_month_realized_variance"]
        - evaluation["baseline_log_variance_forecast"]
    )
    evaluation["log_variance_squared_error_augmented"] = np.square(
        evaluation["log_next_month_realized_variance"]
        - evaluation["augmented_log_variance_forecast"]
    )
    evaluation["structural_period"] = [
        _structural_period(pd.Timestamp(value), config)
        for value in evaluation["target_month_end_date"]
    ]
    if evaluation["structural_period"].isna().any():
        raise ContractError("机制评价存在未归类结构时期")
    max_lag = int(config["mechanism_evaluation"]["newey_west_max_lag_months"])
    full = _mechanism_subset_summary(evaluation, max_lag=max_lag)
    structural = {
        name: _mechanism_subset_summary(
            evaluation.loc[evaluation["structural_period"] == name],
            max_lag=max_lag,
        )
        for name in config["dates"]["structural_periods"]
    }
    bootstrap = _bootstrap_beta_dispersion(evaluation, config)
    beta = full["beta_dispersion"]
    lower = bootstrap["confidence_interval"][0]
    gate_spec = config["mechanism_gates"]
    gates = {
        "minimum_complete_oos_target_months": len(evaluation)
        >= int(gate_spec["minimum_complete_oos_target_months"]),
        "full_sample_beta_rd_positive": beta["coefficient"] is not None
        and float(beta["coefficient"]) > 0.0,
        "full_sample_beta_rd_one_sided_t": beta["t_stat"] is not None
        and float(beta["t_stat"])
        >= float(gate_spec["full_sample_beta_rd_one_sided_t_minimum"]),
        "bootstrap_90pct_lower_beta_rd_positive": lower is not None and float(lower) > 0.0,
        "full_sample_qlike_improvement_positive": float(full["mean_qlike_improvement"]) > 0.0,
        "full_sample_qlike_dm_one_sided_t": full["qlike_dm_t_stat"] is not None
        and float(full["qlike_dm_t_stat"])
        >= float(gate_spec["full_sample_qlike_dm_one_sided_t_minimum"]),
        "every_structural_period_beta_rd_positive": all(
            details["beta_dispersion"]["coefficient"] is not None
            and float(details["beta_dispersion"]["coefficient"]) > 0.0
            for details in structural.values()
        ),
        "every_structural_period_qlike_improvement_positive": all(
            float(details["mean_qlike_improvement"]) > 0.0
            for details in structural.values()
        ),
    }
    result = {
        "passed": bool(all(gates.values())),
        "observations": int(len(evaluation)),
        "first_target_month": str(evaluation["target_month"].iloc[0]),
        "last_target_month": str(evaluation["target_month"].iloc[-1]),
        "full_sample": full,
        "bootstrap": bootstrap,
        "structural_periods": structural,
        "gates": gates,
    }
    return result, evaluation.drop(columns=["target_period"])


def build_portfolio_signals(
    forecasts: pd.DataFrame,
    factor: pd.DataFrame,
    benchmark: pd.DataFrame,
    market: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    monthly_benchmark = (
        benchmark.assign(month=benchmark["date"].dt.to_period("M").astype(str))
        .groupby("month", sort=True)
        .agg(month_end_date=("date", "max"), close=("close", "last"))
        .reset_index()
    )
    monthly_benchmark["monthly_total_return"] = monthly_benchmark["close"].pct_change()
    factor_dates = factor.set_index("factor_month")["month_end_date"]
    market_dates = pd.DatetimeIndex(market["date"].sort_values().unique())
    minimum_mean_observations = int(
        config["allocation"]["expected_return_minimum_observations"]
    )
    gamma = float(config["allocation"]["risk_aversion_gamma"])
    rows: list[dict[str, Any]] = []
    for forecast in forecasts.itertuples(index=False):
        decision_period = pd.Period(str(forecast.factor_month), freq="M")
        available = monthly_benchmark.loc[
            pd.PeriodIndex(monthly_benchmark["month"], freq="M") <= decision_period,
            "monthly_total_return",
        ].dropna()
        if len(available) < minimum_mean_observations:
            raise ContractError(
                f"决策月{forecast.factor_month}的历史均值观测不足{minimum_mean_observations}"
            )
        historical_mean = float(available.mean())
        forecast_variance = float(forecast.augmented_variance_forecast)
        if not math.isfinite(forecast_variance) or forecast_variance <= 0:
            raise ContractError("扩展模型方差预测无效")
        raw_weight = max(historical_mean, 0.0) / (gamma * forecast_variance)
        target_position = float(np.clip(raw_weight, 0.0, 1.0))
        decision_date = pd.Timestamp(factor_dates.loc[str(forecast.factor_month)])
        future_dates = market_dates[market_dates > decision_date]
        if len(future_dates) == 0:
            raise ContractError(f"决策日{decision_date.date()}之后没有510300交易日")
        execution_date = pd.Timestamp(future_dates[0])
        rows.append(
            {
                "factor_month": str(forecast.factor_month),
                "decision_date": decision_date,
                "execution_date": execution_date,
                "target_month": str(forecast.target_month),
                "training_pairs": int(forecast.training_pairs),
                "historical_mean_monthly_total_return": historical_mean,
                "augmented_forecast_monthly_variance": forecast_variance,
                "raw_target_weight": raw_weight,
                "target_position": target_position,
                "signal_reason": (
                    "历史平均月收益非正，目标为现金"
                    if target_position == 0.0
                    else "横截面离散度扩展波动预测的固定均值方差目标权重"
                ),
            }
        )
    signals = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    if signals["decision_date"].duplicated().any():
        raise ContractError("组合信号决策日重复")
    if not signals["target_position"].between(0.0, 1.0).all():
        raise ContractError("组合目标仓位超出0到1")
    if signals["execution_date"].iloc[0] != pd.Timestamp(config["dates"]["evaluation_start"]):
        raise ContractError("首个组合目标没有在冻结评价起点执行")
    return signals


def _engine_targets(signals: pd.DataFrame) -> pd.DataFrame:
    return (
        signals[["decision_date", "execution_date", "target_position", "signal_reason"]]
        .rename(columns={"decision_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )


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
        raise ContractError("评价期前缺少一个交易日用于执行2014年12月信号")
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
    return _run_account(market, dividends, targets, config, _costs(config, "base_costs"))


def _benchmark_cagr(
    benchmark: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    subset = benchmark.loc[benchmark["date"].between(start_date, end_date)].copy()
    if len(subset) < 2:
        raise ContractError("H00300基准区间不足两行")
    if subset["date"].iloc[0] != start_date or subset["date"].iloc[-1] != end_date:
        raise ContractError("H00300基准区间没有精确落在冻结评价起止日")
    elapsed_days = max((subset["date"].iloc[-1] - subset["date"].iloc[0]).days, 1)
    total_return = float(subset["close"].iloc[-1] / subset["close"].iloc[0] - 1.0)
    return float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


def _return_series_summary(returns: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "observations": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_zero_cash_rate": None,
        }
    total = float(np.prod(1.0 + values) - 1.0)
    annualized_return = (
        float((1.0 + total) ** (242.0 / len(values)) - 1.0) if total > -1.0 else -1.0
    )
    volatility = (
        float(np.std(values, ddof=1) * math.sqrt(242.0)) if len(values) > 1 else None
    )
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
    denominator_squared = (
        1.0
        - skewness * daily_sharpe
        + ((raw_kurtosis - 1.0) / 4.0) * daily_sharpe**2
    )
    if denominator_squared <= 0:
        return None
    z_score = (
        (daily_sharpe - benchmark_daily)
        * math.sqrt(len(values) - 1)
        / math.sqrt(denominator_squared)
    )
    return float(NormalDist().cdf(z_score))


def deflated_sharpe_probability(
    returns: pd.Series,
    total_trial_count: int,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return {
            "total_trial_count": int(total_trial_count),
            "expected_maximum_null_sharpe_annualized": None,
            "probability": None,
        }
    trial_count = max(int(total_trial_count), 1)
    if trial_count == 1:
        expected_maximum_daily = 0.0
    else:
        euler_gamma = 0.5772156649015329
        normal = NormalDist()
        trial_std_daily = 1.0 / math.sqrt(len(values) - 1)
        expected_maximum_daily = trial_std_daily * (
            (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        )
    probability = probabilistic_sharpe_probability(
        pd.Series(values), expected_maximum_daily * math.sqrt(242.0)
    )
    return {
        "total_trial_count": trial_count,
        "expected_maximum_null_sharpe_annualized": float(
            expected_maximum_daily * math.sqrt(242.0)
        ),
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
    signals: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    targets = _engine_targets(signals)
    base_ledger, base_trades = _run_account(
        market, dividends, targets, config, _costs(config, "base_costs")
    )
    stress_ledger, stress_trades = _run_account(
        market, dividends, targets, config, _costs(config, "stress_costs")
    )
    buy_hold_ledger, buy_hold_trades = _run_buy_hold(market, dividends, config)
    initial_cash = float(config["portfolio"]["initial_capital_cny"])
    base = summarize_backtest(base_ledger, base_trades, initial_cash)
    stress = summarize_backtest(stress_ledger, stress_trades, initial_cash)
    buy_hold = summarize_backtest(buy_hold_ledger, buy_hold_trades, initial_cash)
    h00300_cagr = _benchmark_cagr(
        benchmark,
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    structural: dict[str, dict[str, Any]] = {}
    for name, bounds in config["dates"]["structural_periods"].items():
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"])
        strategy_summary = _return_series_summary(
            base_ledger.loc[base_ledger["date"].between(start, end), "daily_return"]
        )
        buy_hold_summary = _return_series_summary(
            buy_hold_ledger.loc[
                buy_hold_ledger["date"].between(start, end), "daily_return"
            ]
        )
        structural[name] = {
            "strategy": strategy_summary,
            "buy_hold": buy_hold_summary,
            "annualized_excess_vs_buy_hold": (
                float(strategy_summary["annualized_return"])
                - float(buy_hold_summary["annualized_return"])
                if strategy_summary["annualized_return"] is not None
                and buy_hold_summary["annualized_return"] is not None
                else None
            ),
        }
    delayed: dict[str, dict[str, Any]] = {}
    for months in config["portfolio"]["delayed_start_months"]:
        ledger, trades = _run_account(
            market,
            dividends,
            targets,
            config,
            _costs(config, "base_costs"),
            delayed_start_months=int(months),
        )
        details = summarize_backtest(ledger, trades, initial_cash)
        details["annualized_excess_vs_buy_hold"] = float(details["cagr"]) - float(
            buy_hold["cagr"]
        )
        delayed[str(int(months))] = details
    annual = _annual_contribution_analysis(base_ledger, buy_hold_ledger)
    total_trials = int(
        manifest["selection_bias_control"]["total_trial_count_including_current"]
    )
    dsr = deflated_sharpe_probability(base_ledger["daily_return"], total_trials)
    psr_above_1 = probabilistic_sharpe_probability(base_ledger["daily_return"], 1.0)
    drawdown_ratio = (
        abs(float(base["max_drawdown"])) / abs(float(buy_hold["max_drawdown"]))
        if float(buy_hold["max_drawdown"]) < 0
        else None
    )
    base_excess_buy_hold = float(base["cagr"]) - float(buy_hold["cagr"])
    base_excess_h00300 = float(base["cagr"]) - h00300_cagr
    gate_spec = config["portfolio_gates"]
    gates = {
        "base_net_sharpe": base["sharpe_zero_cash_rate"] is not None
        and float(base["sharpe_zero_cash_rate"])
        >= float(gate_spec["base_net_sharpe_minimum"]),
        "stress_net_sharpe": stress["sharpe_zero_cash_rate"] is not None
        and float(stress["sharpe_zero_cash_rate"])
        >= float(gate_spec["stress_net_sharpe_minimum"]),
        "annualized_excess_vs_510300_buy_hold_positive": base_excess_buy_hold > 0.0,
        "annualized_excess_vs_h00300_total_return_positive": base_excess_h00300 > 0.0,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio is not None
        and drawdown_ratio
        <= float(gate_spec["maximum_drawdown_ratio_vs_buy_hold_maximum"]),
        "every_structural_period_net_sharpe": all(
            details["strategy"]["sharpe_zero_cash_rate"] is not None
            and float(details["strategy"]["sharpe_zero_cash_rate"])
            >= float(gate_spec["every_structural_period_net_sharpe_minimum"])
            for details in structural.values()
        ),
        "every_structural_period_excess_vs_buy_hold_positive": all(
            details["annualized_excess_vs_buy_hold"] is not None
            and float(details["annualized_excess_vs_buy_hold"]) > 0.0
            for details in structural.values()
        ),
        "every_delayed_start_net_sharpe": all(
            details["sharpe_zero_cash_rate"] is not None
            and float(details["sharpe_zero_cash_rate"])
            >= float(gate_spec["every_delayed_start_net_sharpe_minimum"])
            for details in delayed.values()
        ),
        "every_delayed_start_excess_vs_buy_hold_positive": all(
            float(details["annualized_excess_vs_buy_hold"]) > 0.0
            for details in delayed.values()
        ),
        "delete_any_calendar_year_excess_vs_buy_hold_positive": all(
            float(value) > 0.0
            for value in annual[
                "delete_calendar_year_total_return_excess_vs_buy_hold"
            ].values()
        ),
        "maximum_single_positive_year_share": annual[
            "maximum_single_positive_year_share_of_positive_excess"
        ]
        is not None
        and float(annual["maximum_single_positive_year_share_of_positive_excess"])
        <= float(gate_spec["maximum_single_positive_year_share_of_positive_excess"]),
        "deflated_sharpe_probability": dsr["probability"] is not None
        and float(dsr["probability"])
        >= float(gate_spec["deflated_sharpe_probability_minimum"]),
        "probabilistic_sharpe_above_1p0": psr_above_1 is not None
        and float(psr_above_1)
        >= float(gate_spec["probabilistic_sharpe_above_1p0_minimum"]),
    }
    result = {
        "passed": bool(all(gates.values())),
        "base": base,
        "stress": stress,
        "buy_hold_510300": buy_hold,
        "h00300_total_return_cagr": h00300_cagr,
        "base_annualized_excess_vs_510300_buy_hold": base_excess_buy_hold,
        "base_annualized_excess_vs_h00300": base_excess_h00300,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio,
        "structural_periods": structural,
        "delayed_starts": delayed,
        "annual_contribution": annual,
        "deflated_sharpe": dsr,
        "probabilistic_sharpe_above_1p0": psr_above_1,
        "gates": gates,
    }
    tables = {
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "stress_ledger": stress_ledger,
        "stress_trades": stress_trades,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
    }
    return result, tables


def _input_snapshot(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for specification in config["inputs"].values():
        path = project_path(specification["path"])
        relative = path.relative_to(ROOT).as_posix()
        output[relative] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return output


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    factor: pd.DataFrame,
    forecasts: pd.DataFrame,
    mechanism_evaluation: pd.DataFrame,
    mechanism_result: dict[str, Any],
    signals: pd.DataFrame | None,
    portfolio_result: dict[str, Any] | None,
) -> dict[str, Any]:
    mechanism_passed = bool(mechanism_result["passed"])
    portfolio_evaluated = portfolio_result is not None
    if not mechanism_passed:
        status = config["adjudication"]["mechanism_fail_status"]
        return_evaluation = "NOT_ALLOWED"
        net_sharpe: float | str = "NOT_COMPUTED"
        historical_target_achieved = False
    else:
        if portfolio_result is None or signals is None:
            raise ContractError("机制通过后缺少组合评价")
        if portfolio_result["base"]["sharpe_zero_cash_rate"] is None:
            raise ContractError("组合评价缺少净夏普率")
        net_sharpe = float(portfolio_result["base"]["sharpe_zero_cash_rate"])
        historical_target_achieved = bool(portfolio_result["passed"])
        status = (
            config["adjudication"]["portfolio_pass_status"]
            if historical_target_achieved
            else config["adjudication"]["portfolio_fail_status"]
        )
        return_evaluation = "ALLOWED_AND_COMPLETED"
    signal_summary: dict[str, Any] | None = None
    if signals is not None:
        signal_summary = {
            "rows": int(len(signals)),
            "first_decision_date": signals["decision_date"].min().date().isoformat(),
            "last_decision_date": signals["decision_date"].max().date().isoformat(),
            "minimum_target_weight": float(signals["target_position"].min()),
            "maximum_target_weight": float(signals["target_position"].max()),
            "mean_target_weight": float(signals["target_position"].mean()),
            "cash_target_months": int((signals["target_position"] == 0.0).sum()),
            "full_weight_months": int((signals["target_position"] == 1.0).sum()),
        }
    return {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "paper_transfer": {
            "primary_title": config["literature"]["primary_mechanism_paper"]["title"],
            "primary_sample_end": config["literature"]["primary_mechanism_paper"][
                "sample_end"
            ],
            "evaluation_start": config["dates"]["evaluation_start"],
            "expected_direction": config["factor"]["expected_predictive_direction"],
            "operational_model": config["mechanism_model"]["augmented_regression"],
            "paper_replication_claimed": False,
        },
        "factor_summary": {
            "rows": int(len(factor)),
            "first_month": str(factor["factor_month"].iloc[0]),
            "last_month": str(factor["factor_month"].iloc[-1]),
            "minimum_valid_member_coverage": float(factor["valid_return_coverage"].min()),
        },
        "forecast_rows": int(len(forecasts)),
        "mechanism_evaluation_rows": int(len(mechanism_evaluation)),
        "mechanism_gate_passed": mechanism_passed,
        "mechanism": mechanism_result,
        "portfolio_evaluated": portfolio_evaluated,
        "signal_summary": signal_summary,
        "portfolio": portfolio_result,
        "selection_bias_control": manifest["selection_bias_control"],
        "input_snapshot": _input_snapshot(config),
        "adjudication": {
            "return_evaluation": return_evaluation,
            "net_sharpe": net_sharpe,
            "target_net_sharpe": float(config["adjudication"]["target_net_sharpe"]),
            "historical_target_achieved": historical_target_achieved,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "no_rescue": True,
        },
        "execution_boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    full = report["mechanism"]["full_sample"]
    beta = full["beta_dispersion"]
    bootstrap = report["mechanism"]["bootstrap"]
    lines = [
        "# 510300 沪深300横截面收益离散度波动预算 V1",
        "",
        f"- 项目标识：`{report['project_id']}`",
        f"- 最终状态：`{report['status']}`",
        f"- 机制门：`{'PASS' if report['mechanism_gate_passed'] else 'FAIL'}`",
        f"- 组合收益是否获准计算：`{report['portfolio_evaluated']}`",
        f"- RETURN_EVALUATION：`{report['adjudication']['return_evaluation']}`",
        f"- NET_SHARPE：`{report['adjudication']['net_sharpe']}`",
        "- 实盘授权：`false`",
        "",
        "## 冻结候选",
        "",
        "唯一活跃因子是沪深300点时成分股月度总收益的等权横截面样本标准差。模型以本月实现方差为基线控制项，扩展模型只增加本月离散度平方的对数；不设阈值、不反转方向、不组合既有被拒候选。",
        "",
        "2015年起逐月扩展窗样本外预测下一自然月000300价格指数实现方差。只有扩展模型相对基线通过系数方向、HAC显著性、移动块自助区间、QLIKE和结构分段全部门槛，才允许读取510300组合收益。",
        "",
        "## 机制门结果",
        "",
        f"- 完整样本外目标月：{report['mechanism']['observations']}",
        f"- 离散度系数：{beta['coefficient']}",
        f"- Newey-West t值：{beta['t_stat']}",
        f"- 90%移动块自助区间：{bootstrap['confidence_interval']}",
        f"- 平均QLIKE改善：{full['mean_qlike_improvement']}",
        f"- QLIKE DM t值：{full['qlike_dm_t_stat']}",
        "",
        "机制门明细：",
        "",
    ]
    for name, passed in report["mechanism"]["gates"].items():
        lines.append(f"- `{name}`：`{passed}`")
    lines.extend(["", "## 组合评价", ""])
    portfolio = report["portfolio"]
    if portfolio is None:
        lines.append("机制门失败，按冻结协议禁止计算策略收益、夏普率或仅用波动基线生成替代仓位。")
    else:
        base = portfolio["base"]
        stress = portfolio["stress"]
        buy_hold = portfolio["buy_hold_510300"]
        lines.extend(
            [
                f"- 基准成本净夏普率：{base['sharpe_zero_cash_rate']}",
                f"- 10bp滑点压力净夏普率：{stress['sharpe_zero_cash_rate']}",
                f"- 510300买入持有夏普率：{buy_hold['sharpe_zero_cash_rate']}",
                f"- 基准成本CAGR：{base['cagr']}",
                f"- 最大回撤：{base['max_drawdown']}",
                "",
                "组合门明细：",
                "",
            ]
        )
        for name, passed in portfolio["gates"].items():
            lines.append(f"- `{name}`：`{passed}`")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "历史成员与行情为回溯重建，不能充当历史月末可得性时间戳证明；历史通过也只能进入独立前瞻确认，不能生成订单、连接券商或改变仓位。",
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


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def run_study(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    result_path = project_path(config["paths"]["result_json"])
    if write and result_path.exists():
        raise FileExistsError(f"一次性结果已经存在，禁止覆盖：{result_path}")
    factor, index_daily, market, dividends, benchmark, _ = load_inputs(config)
    mechanism_panel = build_mechanism_panel(factor, index_daily, config)
    forecasts = expanding_forecasts(mechanism_panel, config)
    mechanism_result, mechanism_evaluation = evaluate_mechanism_gate(forecasts, config)

    signals: pd.DataFrame | None = None
    portfolio_result: dict[str, Any] | None = None
    portfolio_tables: dict[str, pd.DataFrame] = {}
    if mechanism_result["passed"]:
        signals = build_portfolio_signals(forecasts, factor, benchmark, market, config)
        portfolio_result, portfolio_tables = evaluate_portfolio(
            market, dividends, benchmark, signals, config, manifest
        )

    report = build_report(
        config,
        manifest,
        factor,
        forecasts,
        mechanism_evaluation,
        mechanism_result,
        signals,
        portfolio_result,
    )
    if write:
        _atomic_json(result_path, report)
        _atomic_text(
            project_path(config["paths"]["result_markdown"]),
            render_markdown(report),
        )
        _atomic_parquet(
            project_path(config["paths"]["mechanism_table"]), mechanism_evaluation
        )
        _atomic_parquet(project_path(config["paths"]["forecast_table"]), forecasts)
        if signals is not None:
            _atomic_parquet(project_path(config["paths"]["portfolio_targets"]), signals)
            _atomic_parquet(
                project_path(config["paths"]["base_ledger"]),
                portfolio_tables["base_ledger"],
            )
            _atomic_csv(
                project_path(config["paths"]["base_trades"]),
                portfolio_tables["base_trades"],
            )
            _atomic_parquet(
                project_path(config["paths"]["stress_ledger"]),
                portfolio_tables["stress_ledger"],
            )
            _atomic_csv(
                project_path(config["paths"]["stress_trades"]),
                portfolio_tables["stress_trades"],
            )
            _atomic_parquet(
                project_path(config["paths"]["buy_hold_ledger"]),
                portfolio_tables["buy_hold_ledger"],
            )
            _atomic_csv(
                project_path(config["paths"]["buy_hold_trades"]),
                portfolio_tables["buy_hold_trades"],
            )
    return report
