"""510300 沪深300 PCA 吸收率择时 V1。

本模块把因子机制检验与组合收益评价严格分层。只有冻结后的机制门全部
通过，程序才会读取510300、分红和H00300并计算组合表现。历史结果从不
授权 Paper、Shadow、订单、券商连接或实盘。
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


CONFIG_PATH = ROOT / "config" / "510300_csi300_absorption_ratio_timing_v1.yaml"
MANIFEST_PATH = (
    ROOT / "config" / "510300_csi300_absorption_ratio_timing_v1_manifest.json"
)


class ContractError(RuntimeError):
    """冻结协议、清单或输入不满足要求。"""


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


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ContractError(f"{label}缺少字段：{missing}")


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
    selection = config["selection_bias"]
    governance = config["governance"]

    expected_id = "510300_CSI300_ABSORPTION_RATIO_TIMING_V1"
    if protocol["project_id"] != expected_id or protocol["candidate_model_id"] != expected_id:
        raise ContractError("吸收率候选标识漂移")
    if protocol["one_shot"] is not True:
        raise ContractError("候选必须固定为一次性检验")
    if protocol["pre_factor_data_contract_correction"] is not True:
        raise ContractError("必须如实登记冻结前成员区间纠错")
    if protocol["pre_factor_failure_receipt"] != (
        "reports/frozen/510300_csi300_absorption_ratio_timing_v1_pre_factor_membership_failure.json"
    ):
        raise ContractError("冻结前成员失败收据路径漂移")
    if protocol["historical_510300_data_already_contaminated"] is not True:
        raise ContractError("必须承认既有510300历史已经被观察")
    if protocol["candidate_factor_values_read_before_freeze"] is not True:
        raise ContractError("输入构建会读取因子值，协议必须如实声明")
    for key in (
        "candidate_market_return_outcomes_read_before_freeze",
        "candidate_portfolio_returns_read_before_freeze",
        "pristine_blind_holdout_claim_allowed",
    ):
        if protocol[key] is not False:
            raise ContractError(f"协议字段{key}必须固定为false")
    if scope["execution_asset"] != "510300.SH":
        raise ContractError("执行资产必须固定为510300.SH")
    if scope["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("允许持有范围必须仅为510300与现金")
    if not scope["long_only"]:
        raise ContractError("必须固定为只做多")
    if any(
        [
            scope["leverage_allowed"],
            scope["short_selling_allowed"],
            scope["derivatives_execution_allowed"],
            scope["live_trading_authorized"],
        ]
    ):
        raise ContractError("禁止杠杆、卖空、衍生品执行和实盘")
    if dates["evaluation_start"] != "2015-01-05" or dates["evaluation_end"] != "2026-08-14":
        raise ContractError("冻结评价区间必须是2015-01-05至2026-08-14")
    if dates["pre_2015_returns_allowed_for_evaluation"] is not False:
        raise ContractError("2015年前收益不得进入评价")
    if contract["calendar_source"] != "csi300_price_index_date_column_only_before_freeze":
        raise ContractError("冻结前只能读取000300日期列作为交易日历")
    if contract["factor_builder_forbidden_columns_before_freeze"] != [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        raise ContractError("冻结前市场结果禁读字段漂移")
    if int(contract["minimum_valid_member_count"]) != 285:
        raise ContractError("有效成分最低数量必须固定为285")
    if not math.isclose(float(contract["minimum_valid_member_coverage"]), 0.95):
        raise ContractError("有效成分最低覆盖率必须固定为95%")
    expected_corrections = [
        {
            "symbol": "600357.SH",
            "original_opt_in": "2005-04-08",
            "original_opt_out": "2014-07-10",
            "corrected_opt_out": "2009-12-29",
            "correction_reason": "上交所资料确认承德钒钛于2009-12-29终止上市，且指数补充公告规定若在2010-01-04前退市则自退市日起剔除",
            "evidence_urls": [
                "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170572/files/36d0635dee474838943e931bcc04c3df.pdf",
                "https://www.sse.com.cn/market/sseindex/diclosure/c/c_20150911_3984987.shtml",
            ],
            "outcome_independent": True,
        }
    ]
    if contract["pre_factor_membership_interval_corrections"] != expected_corrections:
        raise ContractError("冻结前成员区间纠错内容漂移")
    if int(contract["expected_relevant_symbol_count_after_corrections"]) != 714:
        raise ContractError("纠错后首个因子日至结束日的相关证券数量必须固定为714")
    if contract["membership_is_required_only_from_first_possible_factor_date"] is not True:
        raise ContractError("成员完整性必须从500日价格暖机后的首个可能因子日开始")
    if factor["active_factor_count"] != 1:
        raise ContractError("活跃因子数量必须固定为1")
    if factor["id"] != "CSI300_POINT_IN_TIME_PCA_ABSORPTION_RATIO_STANDARDIZED_SHIFT":
        raise ContractError("吸收率因子标识漂移")
    if int(factor["covariance_window_trading_days"]) != 500:
        raise ContractError("协方差窗口必须固定为500个交易日")
    if int(factor["exponential_half_life_trading_days"]) != 250:
        raise ContractError("指数权重半衰期必须固定为250个交易日")
    if int(factor["short_average_trading_days"]) != 15:
        raise ContractError("短均值必须固定为15个交易日")
    if int(factor["long_average_trading_days"]) != 250:
        raise ContractError("长均值必须固定为250个交易日")
    if factor["eigenvector_count_formula"] != "floor(valid_member_count/5)":
        raise ContractError("特征向量数量必须固定为有效成分数的五分之一向下取整")
    if not math.isclose(float(factor["upper_risk_off_threshold"]), 1.0):
        raise ContractError("风险关闭阈值必须固定为+1")
    if not math.isclose(float(factor["lower_risk_on_threshold"]), -1.0):
        raise ContractError("风险开启阈值必须固定为-1")
    if not math.isclose(float(factor["initial_target_position"]), 0.5):
        raise ContractError("初始目标仓位必须固定为50%")
    if factor["threshold_mapping"] != "HYSTERESIS_CARRY_PREVIOUS_STATE":
        raise ContractError("阈值之间必须维持上一状态")
    if model["target"] != "next_execution_open_to_following_open_000300_price_index_log_return":
        raise ContractError("机制结果时钟漂移")
    if int(model["signal_lag_trading_days"]) != 1:
        raise ContractError("信号必须滞后一个交易日执行")
    if int(model["newey_west_max_lag_trading_days"]) != 20:
        raise ContractError("Newey-West最大滞后必须固定为20日")
    if allocation["alternate_continuous_mapping"] != "forbidden":
        raise ContractError("禁止结果后改为连续仓位映射")
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
    if int(selection["expected_prior_manifest_count"]) != 333:
        raise ContractError("冻结前历史manifest保守计数必须固定为333")
    if int(selection["expected_total_trial_count_including_current"]) != 334:
        raise ContractError("含当前候选的试验次数必须固定为334")
    if governance["order_generation"] or governance["broker_connection"]:
        raise ContractError("研究协议禁止订单生成和券商连接")
    if governance["position_change"] or governance["live_trading_authorized"]:
        raise ContractError("研究协议禁止真实仓位变化和实盘")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("缺少冻结清单；禁止读取候选市场结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_MARKET_RESULT":
        raise ContractError("冻结清单状态无效")
    if manifest.get("result_preexisted_at_freeze") is not False:
        raise ContractError("冻结时不得已存在结果")
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
    if int(selection.get("prior_manifest_count", -1)) != 333:
        raise ContractError("冻结清单的历史试验计数不是333")
    if int(selection.get("total_trial_count_including_current", -1)) != 334:
        raise ContractError("冻结清单的总试验计数不是334")
    return manifest


def _load_audit(specification: dict[str, Any], label: str) -> dict[str, Any]:
    path = project_path(specification["path"])
    if not path.exists():
        raise ContractError(f"缺少{label}：{path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    required = specification["required_status"]
    if document.get("status") != required:
        raise ContractError(f"{label}状态不是{required}：{document.get('status')}")
    return document


def load_mechanism_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specifications = config["inputs"]
    factor_spec = specifications["daily_absorption_ratio_factor"]
    index_spec = specifications["csi300_price_index"]
    for specification in (factor_spec, index_spec):
        if not project_path(specification["path"]).exists():
            raise ContractError(f"缺少机制输入：{specification['path']}")
    input_audit = _load_audit(specifications["input_audit"], "吸收率输入审计")
    _load_audit(specifications["source_inventory"], "吸收率源文件清单")
    if input_audit.get("candidate_market_return_outcomes_read_or_computed_before_freeze") is not False:
        raise ContractError("输入构建阶段已经读取或计算候选市场结果")
    if input_audit.get("candidate_portfolio_returns_read_or_computed_before_freeze") is not False:
        raise ContractError("输入构建阶段已经读取或计算组合收益")

    factor = pd.read_parquet(project_path(factor_spec["path"]))
    _require_columns(factor, factor_spec["required_columns"], "日度吸收率因子")
    factor["date"] = pd.to_datetime(factor["date"], errors="coerce")
    if factor["date"].isna().any() or factor["date"].duplicated().any():
        raise ContractError("日度吸收率因子日期无效或重复")
    factor = factor.sort_values("date").reset_index(drop=True)
    numeric = [
        "active_member_count",
        "valid_member_count",
        "valid_member_coverage",
        "eigenvector_count",
        "absorption_ratio",
        "absorption_ratio_mean_15",
        "absorption_ratio_mean_250",
        "absorption_ratio_std_250",
        "standardized_absorption_ratio_shift",
    ]
    factor[numeric] = factor[numeric].apply(pd.to_numeric, errors="coerce")
    required_from = pd.Timestamp(config["data_contract"]["required_complete_signal_on_every_trading_day_from"])
    end = pd.Timestamp(config["dates"]["factor_calculation_end"])
    evaluation_factor = factor.loc[factor["date"].between(required_from, end)]
    if evaluation_factor.empty or evaluation_factor[numeric].isna().any().any():
        raise ContractError("冻结信号区间存在缺失或非数值因子")
    if not evaluation_factor["valid_member_count"].ge(
        int(config["data_contract"]["minimum_valid_member_count"])
    ).all():
        raise ContractError("冻结信号区间有效成分数不足285")
    if not evaluation_factor["valid_member_coverage"].ge(
        float(config["data_contract"]["minimum_valid_member_coverage"])
    ).all():
        raise ContractError("冻结信号区间有效成分覆盖率不足95%")
    if not evaluation_factor["absorption_ratio"].between(0.0, 1.0).all():
        raise ContractError("吸收率不在0到1之间")

    index_daily = pd.read_parquet(project_path(index_spec["path"]))
    _require_columns(index_daily, index_spec["required_columns"], "000300价格指数")
    index_daily["date"] = pd.to_datetime(index_daily["date"], errors="coerce")
    index_daily = index_daily.sort_values("date").drop_duplicates("date", keep="last")
    index_daily[["open", "high", "low", "close"]] = index_daily[
        ["open", "high", "low", "close"]
    ].apply(pd.to_numeric, errors="coerce")
    if index_daily["date"].isna().any() or index_daily[["open", "high", "low", "close"]].isna().any().any():
        raise ContractError("000300价格指数日期或价格无效")
    if (index_daily[["open", "high", "low", "close"]] <= 0).any().any():
        raise ContractError("000300价格指数包含非正价格")
    if index_daily["date"].min() > pd.Timestamp(config["dates"]["constituent_price_history_start"]):
        raise ContractError("000300价格指数起点过晚")
    if index_daily["date"].max() < end:
        raise ContractError("000300价格指数终点过早")
    return factor, index_daily.reset_index(drop=True), input_audit


def build_hysteresis_states(
    factor: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    upper = float(config["factor"]["upper_risk_off_threshold"])
    lower = float(config["factor"]["lower_risk_on_threshold"])
    state = float(config["factor"]["initial_target_position"])
    rows: list[dict[str, Any]] = []
    for row in factor.sort_values("date").itertuples(index=False):
        shift = _optional_float(row.standardized_absorption_ratio_shift)
        prior = state
        if shift is None:
            event = "NO_VIEW_CARRY_PREVIOUS_TARGET"
        elif shift >= upper:
            state = 0.0
            event = "RISK_OFF_TRIGGER" if prior != state else "RISK_OFF_STATE_CONFIRMED"
        elif shift <= lower:
            state = 1.0
            event = "RISK_ON_TRIGGER" if prior != state else "RISK_ON_STATE_CONFIRMED"
        else:
            event = "BETWEEN_THRESHOLDS_CARRY_PREVIOUS_TARGET"
        rows.append(
            {
                "decision_date": pd.Timestamp(row.date),
                "standardized_absorption_ratio_shift": shift,
                "target_position": float(state),
                "previous_target_position": float(prior),
                "state_changed": bool(state != prior),
                "signal_reason": event,
                "risk_off_override": bool(state == 0.0 and prior > 0.0),
            }
        )
    states = pd.DataFrame(rows)
    if states.empty or states["decision_date"].duplicated().any():
        raise ContractError("吸收率状态表为空或日期重复")
    return states


def _next_two_market_dates(
    decision: pd.Timestamp,
    market_dates: pd.DatetimeIndex,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    position = int(market_dates.searchsorted(pd.Timestamp(decision), side="right"))
    if position + 1 >= len(market_dates):
        return None
    return pd.Timestamp(market_dates[position]), pd.Timestamp(market_dates[position + 1])


def build_mechanism_panel(
    states: pd.DataFrame,
    index_daily: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    market = index_daily.set_index("date").sort_index()
    dates = pd.DatetimeIndex(market.index.unique()).sort_values()
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    rows: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        mapped = _next_two_market_dates(pd.Timestamp(row.decision_date), dates)
        if mapped is None:
            continue
        execution_date, target_end_date = mapped
        if execution_date < start or target_end_date > end:
            continue
        start_open = float(market.loc[execution_date, "open"])
        end_open = float(market.loc[target_end_date, "open"])
        target_return = float(math.log(end_open / start_open))
        rows.append(
            {
                "decision_date": pd.Timestamp(row.decision_date),
                "execution_date": execution_date,
                "target_end_date": target_end_date,
                "standardized_absorption_ratio_shift": _optional_float(
                    row.standardized_absorption_ratio_shift
                ),
                "target_position": float(row.target_position),
                "signal_reason": str(row.signal_reason),
                "target_open_to_open_log_return": target_return,
            }
        )
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ContractError("机制评价面板为空")
    if panel["execution_date"].duplicated().any():
        raise ContractError("机制评价执行日重复")
    panel["structural_period"] = [
        _structural_period(pd.Timestamp(value), config)
        for value in panel["execution_date"]
    ]
    if panel["structural_period"].isna().any():
        raise ContractError("机制评价存在未归类结构时期")
    return panel


def _ols_fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ContractError("OLS输入形状无效")
    if len(y) <= x.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContractError("OLS输入观测不足或包含非有限值")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ContractError("OLS设计矩阵秩不足")
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
    effective_lag = min(int(max_lag), len(y) - 2)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / (effective_lag + 1.0)
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
    centered = y - y.mean()
    denominator = float(np.dot(centered, centered))
    return {
        "observations": int(len(y)),
        "max_lag": int(effective_lag),
        "coefficients": estimates,
        "r_squared": _optional_float(
            1.0 - float(np.dot(residuals, residuals)) / denominator
            if denominator > 0
            else None
        ),
    }


def _circular_block_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block_count = math.ceil(length / block_length)
    starts = rng.integers(0, length, size=block_count)
    offsets = np.arange(block_length, dtype=int)
    return ((starts[:, None] + offsets[None, :]) % length).ravel()[:length]


def _bootstrap_exposure_beta(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["mechanism_model"]
    repetitions = int(specification["bootstrap_repetitions"])
    block_length = int(specification["bootstrap_block_length_trading_days"])
    confidence = float(specification["bootstrap_confidence_level"])
    rng = np.random.default_rng(int(specification["random_seed"]))
    y = panel["target_open_to_open_log_return"].to_numpy(dtype=float)
    exposure = panel["target_position"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(panel)), exposure])
    estimates: list[float] = []
    for _ in range(repetitions):
        indices = _circular_block_indices(len(panel), block_length, rng)
        try:
            coefficients, _ = _ols_fit(x[indices], y[indices])
        except ContractError:
            continue
        if math.isfinite(float(coefficients[1])):
            estimates.append(float(coefficients[1]))
    alpha = (1.0 - confidence) / 2.0
    interval = (
        [
            float(np.quantile(estimates, alpha)),
            float(np.quantile(estimates, 1.0 - alpha)),
        ]
        if estimates
        else [None, None]
    )
    return {
        "method": specification["bootstrap_method"],
        "repetitions": repetitions,
        "valid_draws": int(len(estimates)),
        "block_length_trading_days": block_length,
        "confidence_level": confidence,
        "confidence_interval": interval,
        "seed": int(specification["random_seed"]),
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
    counts = {
        "risk_off_days": int(subset["target_position"].eq(0.0).sum()),
        "initial_neutral_days": int(subset["target_position"].eq(0.5).sum()),
        "risk_on_days": int(subset["target_position"].eq(1.0).sum()),
    }
    y = subset["target_open_to_open_log_return"].to_numpy(dtype=float)
    exposure = subset["target_position"].to_numpy(dtype=float)
    try:
        regression = _newey_west_regression(
            y,
            np.column_stack([np.ones(len(subset)), exposure]),
            max_lag=max_lag,
            coefficient_names=["intercept", "target_position"],
        )
        beta = regression["coefficients"]["target_position"]
    except ContractError:
        regression = None
        beta = {"coefficient": None, "standard_error": None, "t_stat": None}
    state_returns: dict[str, dict[str, float | int | None]] = {}
    for label, value in (("RISK_OFF", 0.0), ("INITIAL_NEUTRAL", 0.5), ("RISK_ON", 1.0)):
        values = subset.loc[
            subset["target_position"].eq(value), "target_open_to_open_log_return"
        ]
        state_returns[label] = {
            "observations": int(len(values)),
            "mean_daily_log_return": float(values.mean()) if len(values) else None,
            "annualized_mean_log_return": float(values.mean() * 242.0) if len(values) else None,
        }
    return {
        "observations": int(len(subset)),
        "state_counts": counts,
        "newey_west_regression": regression,
        "beta_exposure": beta,
        "state_returns": state_returns,
    }


def evaluate_mechanism_gate(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_lag = int(config["mechanism_model"]["newey_west_max_lag_trading_days"])
    full = _mechanism_subset_summary(panel, max_lag=max_lag)
    structural = {
        name: _mechanism_subset_summary(
            panel.loc[panel["structural_period"].eq(name)].copy(), max_lag=max_lag
        )
        for name in config["dates"]["structural_periods"]
    }
    bootstrap = _bootstrap_exposure_beta(panel, config)
    beta = full["beta_exposure"]
    lower = bootstrap["confidence_interval"][0]
    gate_spec = config["mechanism_gates"]
    minimum_structural_extreme = int(
        gate_spec["minimum_each_extreme_state_days_per_structural_period"]
    )
    gates = {
        "minimum_complete_oos_trading_returns": len(panel)
        >= int(gate_spec["minimum_complete_oos_trading_returns"]),
        "minimum_full_risk_on_days": full["state_counts"]["risk_on_days"]
        >= int(gate_spec["minimum_full_risk_on_days"]),
        "minimum_full_risk_off_days": full["state_counts"]["risk_off_days"]
        >= int(gate_spec["minimum_full_risk_off_days"]),
        "full_sample_beta_exposure_positive": beta["coefficient"] is not None
        and float(beta["coefficient"]) > 0.0,
        "full_sample_beta_exposure_one_sided_t": beta["t_stat"] is not None
        and float(beta["t_stat"])
        >= float(gate_spec["full_sample_beta_exposure_one_sided_t_minimum"]),
        "bootstrap_90pct_lower_beta_exposure_positive": lower is not None
        and float(lower) > 0.0,
        "every_structural_period_beta_exposure_positive": all(
            details["beta_exposure"]["coefficient"] is not None
            and float(details["beta_exposure"]["coefficient"]) > 0.0
            for details in structural.values()
        ),
        "minimum_each_extreme_state_days_per_structural_period": all(
            details["state_counts"]["risk_on_days"] >= minimum_structural_extreme
            and details["state_counts"]["risk_off_days"] >= minimum_structural_extreme
            for details in structural.values()
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "observations": int(len(panel)),
        "first_execution_date": panel["execution_date"].min().date().isoformat(),
        "last_target_end_date": panel["target_end_date"].max().date().isoformat(),
        "full_sample": full,
        "bootstrap": bootstrap,
        "structural_periods": structural,
        "gates": gates,
    }


def load_portfolio_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specifications = config["inputs"]
    _load_audit(specifications["etf_input_audit"], "510300行情审计")
    market_spec = specifications["etf_daily"]
    dividend_spec = specifications["dividends"]
    benchmark_spec = specifications["benchmark_total_return"]
    market = pd.read_parquet(project_path(market_spec["path"]))
    dividends = pd.read_csv(project_path(dividend_spec["path"]))
    benchmark = pd.read_parquet(project_path(benchmark_spec["path"]))
    _require_columns(market, market_spec["required_columns"], "510300行情")
    _require_columns(dividends, dividend_spec["required_columns"], "510300分红")
    _require_columns(benchmark, benchmark_spec["required_columns"], "H00300基准")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="coerce")
    if market["date"].isna().any() or benchmark["date"].isna().any():
        raise ContractError("组合行情或基准日期无效")
    if dividends[["ex_date", "payment_date"]].isna().any().any():
        raise ContractError("分红关键日期无效")
    return market.reset_index(drop=True), dividends, benchmark.reset_index(drop=True)


def build_portfolio_signals(
    states: pd.DataFrame,
    market: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    market_dates = pd.DatetimeIndex(market["date"].sort_values().unique())
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    evaluation_end = pd.Timestamp(config["dates"]["evaluation_end"])
    earlier = market_dates[market_dates < evaluation_start]
    if len(earlier) == 0:
        raise ContractError("评价期前缺少用于首日执行的交易日")
    engine_start = pd.Timestamp(earlier.max())
    rows: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        decision = pd.Timestamp(row.decision_date)
        if decision not in market_dates:
            if engine_start <= decision < evaluation_end:
                raise ContractError(f"吸收率信号日不是510300交易日：{decision.date()}")
            continue
        position = int(market_dates.searchsorted(decision, side="right"))
        if position >= len(market_dates):
            continue
        execution = pd.Timestamp(market_dates[position])
        if decision < engine_start or execution > evaluation_end:
            continue
        rows.append(
            {
                "decision_date": decision,
                "execution_date": execution,
                "standardized_absorption_ratio_shift": _optional_float(
                    row.standardized_absorption_ratio_shift
                ),
                "target_position": float(row.target_position),
                "signal_reason": str(row.signal_reason),
                "risk_off_override": bool(row.risk_off_override),
            }
        )
    signals = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    if signals.empty or signals["decision_date"].duplicated().any():
        raise ContractError("组合目标为空或决策日重复")
    first = signals.loc[signals["execution_date"] >= evaluation_start].iloc[0]
    if pd.Timestamp(first["decision_date"]) != engine_start:
        raise ContractError("缺少评价首日前一个交易日的冻结目标")
    if pd.Timestamp(first["execution_date"]) != evaluation_start:
        raise ContractError("首个目标没有在2015-01-05开盘执行")
    return signals


def _engine_targets(signals: pd.DataFrame) -> pd.DataFrame:
    return (
        signals[
            [
                "decision_date",
                "execution_date",
                "target_position",
                "signal_reason",
                "risk_off_override",
            ]
        ]
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
        raise ContractError("评价期前缺少一个交易日")
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
        mask = prepared["execution_date"] < cutoff
        prepared.loc[mask, "target_position"] = 0.0
        prepared.loc[mask, "signal_reason"] = f"延迟起点稳健性：前{int(delayed_start_months)}个月持有现金"
        prepared.loc[mask, "risk_off_override"] = True
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
            "risk_off_override": [False],
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
        raise ContractError("H00300基准没有精确覆盖冻结评价起止日")
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
    signals: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    targets = _engine_targets(signals)
    base_ledger, base_trades = _run_account(market, dividends, targets, config, _costs(config, "base_costs"))
    stress_ledger, stress_trades = _run_account(market, dividends, targets, config, _costs(config, "stress_costs"))
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
            buy_hold_ledger.loc[buy_hold_ledger["date"].between(start, end), "daily_return"]
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
        details["annualized_excess_vs_buy_hold"] = float(details["cagr"]) - float(buy_hold["cagr"])
        delayed[str(int(months))] = details
    annual = _annual_contribution_analysis(base_ledger, buy_hold_ledger)
    total_trials = int(manifest["selection_bias_control"]["total_trial_count_including_current"])
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
        and float(base["sharpe_zero_cash_rate"]) >= float(gate_spec["base_net_sharpe_minimum"]),
        "stress_net_sharpe": stress["sharpe_zero_cash_rate"] is not None
        and float(stress["sharpe_zero_cash_rate"]) >= float(gate_spec["stress_net_sharpe_minimum"]),
        "annualized_excess_vs_510300_buy_hold_positive": base_excess_buy_hold > 0.0,
        "annualized_excess_vs_h00300_total_return_positive": base_excess_h00300 > 0.0,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio is not None
        and drawdown_ratio <= float(gate_spec["maximum_drawdown_ratio_vs_buy_hold_maximum"]),
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
            for value in annual["delete_calendar_year_total_return_excess_vs_buy_hold"].values()
        ),
        "maximum_single_positive_year_share": annual[
            "maximum_single_positive_year_share_of_positive_excess"
        ]
        is not None
        and float(annual["maximum_single_positive_year_share_of_positive_excess"])
        <= float(gate_spec["maximum_single_positive_year_share_of_positive_excess"]),
        "deflated_sharpe_probability": dsr["probability"] is not None
        and float(dsr["probability"]) >= float(gate_spec["deflated_sharpe_probability_minimum"]),
        "probabilistic_sharpe_above_1p0": psr_above_1 is not None
        and float(psr_above_1) >= float(gate_spec["probabilistic_sharpe_above_1p0_minimum"]),
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
    states: pd.DataFrame,
    mechanism_panel: pd.DataFrame,
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
    complete_shift = factor["standardized_absorption_ratio_shift"].notna()
    transitions = int(states["state_changed"].sum())
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
            "primary_sample_end": config["literature"]["primary_mechanism_paper"]["sample_end"],
            "evaluation_start": config["dates"]["evaluation_start"],
            "paper_replication_claimed": False,
            "paper_bond_return_borrowed": False,
        },
        "factor_summary": {
            "rows": int(len(factor)),
            "first_date": factor["date"].min().date().isoformat(),
            "last_date": factor["date"].max().date().isoformat(),
            "first_complete_standardized_shift_date": factor.loc[
                complete_shift, "date"
            ].min().date().isoformat(),
            "minimum_valid_member_coverage": float(factor["valid_member_coverage"].min()),
            "state_transition_count": transitions,
        },
        "mechanism_evaluation_rows": int(len(mechanism_panel)),
        "mechanism_gate_passed": mechanism_passed,
        "mechanism": mechanism_result,
        "portfolio_evaluated": portfolio_evaluated,
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
    beta = full["beta_exposure"]
    bootstrap = report["mechanism"]["bootstrap"]
    lines = [
        "# 510300 沪深300 PCA 吸收率择时 V1",
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
        "唯一因子是沪深300点时成分股日总收益的PCA吸收率。使用500日指数加权协方差、250日半衰期、最大floor(N/5)个特征值；标准化变化为15日均值减250日均值，再除以250日样本标准差。变化不高于-1时目标为100%，不低于+1时目标为0%，阈值之间维持上一状态。",
        "",
        "机制结果以信号次日开盘至再下一交易日开盘的000300价格指数收益为目标。只有冻结仓位对该收益的Newey-West系数、移动块自助区间、状态样本量和两个结构时期全部通过，才允许读取510300组合收益。",
        "",
        "## 机制门结果",
        "",
        f"- 完整样本外交易收益：{report['mechanism']['observations']}",
        f"- 仓位系数：{beta['coefficient']}",
        f"- Newey-West t值：{beta['t_stat']}",
        f"- 90%移动块自助区间：{bootstrap['confidence_interval']}",
        f"- 风险关闭日：{full['state_counts']['risk_off_days']}",
        f"- 风险开启日：{full['state_counts']['risk_on_days']}",
        "",
        "机制门明细：",
        "",
    ]
    for name, passed in report["mechanism"]["gates"].items():
        lines.append(f"- `{name}`：`{passed}`")
    lines.extend(["", "## 组合评价", ""])
    portfolio = report["portfolio"]
    if portfolio is None:
        lines.append("机制门失败，按冻结协议禁止读取或计算510300策略收益、夏普率及替代仓位。")
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
            "历史成员与行情为回溯重建，不是历史日终可得性证明。原论文的空仓资产是美国国债，本研究固定为零收益现金，未借用论文收益。即便历史通过，也只能进入独立前瞻确认，不能生成订单、连接券商或改变仓位。",
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
    factor, index_daily, _ = load_mechanism_inputs(config)
    states = build_hysteresis_states(factor, config)
    mechanism_panel = build_mechanism_panel(states, index_daily, config)
    mechanism_result = evaluate_mechanism_gate(mechanism_panel, config)

    signals: pd.DataFrame | None = None
    portfolio_result: dict[str, Any] | None = None
    portfolio_tables: dict[str, pd.DataFrame] = {}
    if mechanism_result["passed"]:
        market, dividends, benchmark = load_portfolio_inputs(config)
        signals = build_portfolio_signals(states, market, config)
        portfolio_result, portfolio_tables = evaluate_portfolio(
            market, dividends, benchmark, signals, config, manifest
        )

    report = build_report(
        config,
        manifest,
        factor,
        states,
        mechanism_panel,
        mechanism_result,
        signals,
        portfolio_result,
    )
    if write:
        _atomic_json(result_path, report)
        _atomic_text(project_path(config["paths"]["result_markdown"]), render_markdown(report))
        _atomic_parquet(project_path(config["paths"]["mechanism_table"]), mechanism_panel)
        if signals is not None:
            _atomic_parquet(project_path(config["paths"]["portfolio_targets"]), signals)
            _atomic_parquet(project_path(config["paths"]["base_ledger"]), portfolio_tables["base_ledger"])
            _atomic_csv(project_path(config["paths"]["base_trades"]), portfolio_tables["base_trades"])
            _atomic_parquet(project_path(config["paths"]["stress_ledger"]), portfolio_tables["stress_ledger"])
            _atomic_csv(project_path(config["paths"]["stress_trades"]), portfolio_tables["stress_trades"])
            _atomic_parquet(project_path(config["paths"]["buy_hold_ledger"]), portfolio_tables["buy_hold_ledger"])
            _atomic_csv(project_path(config["paths"]["buy_hold_trades"]), portfolio_tables["buy_hold_trades"])
    return report
