"""510300因果高斯HMM三阶段路由V1。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from three_state_trend_router_v1_0_1 import (  # noqa: E402
    ALL_STATES,
    STATE_BEAR,
    STATE_BULL,
    STATE_RANGE,
    ContractError,
    evaluate_portfolio,
    evaluate_state_mechanism,
    load_inputs as load_daily_inputs,
)


CONFIG_PATH = ROOT / "config" / "510300_causal_gaussian_hmm_regime_router_v1.yaml"
MODEL_PATH = ROOT / "config" / "510300_causal_gaussian_hmm_regime_router_v1_model.json"
MANIFEST_PATH = ROOT / "config" / "510300_causal_gaussian_hmm_regime_router_v1_manifest.json"

FEATURE_NAMES = (
    "LOG_RETURN_5D_PER_SQRT_DAY",
    "LOG_RETURN_20D_PER_SQRT_DAY",
    "LOG_ANNUALIZED_REALIZED_VOLATILITY_20D",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _nested(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in keys:
        value = value[key]
    return value


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        ("protocol", "project_id"): "510300_CAUSAL_GAUSSIAN_HMM_REGIME_ROUTER_V1",
        ("protocol", "one_shot"): True,
        ("scope", "execution_asset"): "510300.SH",
        ("scope", "signal_asset"): "000300.SH_PRICE_INDEX",
        ("dates", "training_start"): "2012-06-26",
        ("dates", "training_end"): "2014-12-31",
        ("dates", "evaluation_start"): "2015-01-05",
        ("dates", "evaluation_end"): "2026-08-14",
        ("hidden_markov_model", "model_family"): "DIAGONAL_GAUSSIAN_HMM",
        ("hidden_markov_model", "latent_state_count"): 3,
        ("hidden_markov_model", "random_state"): 20260831,
        ("hidden_markov_model", "kmeans_n_init"): 20,
        ("hidden_markov_model", "model_selection_or_hyperparameter_search"): "forbidden",
        ("decision_rule", "majority_threshold"): 0.5,
        ("states", STATE_BULL, "target_position"): 1.0,
        ("states", STATE_BEAR, "target_position"): 0.0,
        ("states", STATE_RANGE, "target_position"): 0.0,
        ("range_policy", "historical_policy"): "NO_TRADE",
        ("range_policy", "range_t_backtest_allowed"): False,
        ("portfolio", "initial_capital_cny"): 20000.0,
        ("portfolio", "lot_size_shares"): 100,
        ("portfolio", "base_costs", "slippage_bps_per_leg"): 5.0,
        ("portfolio", "stress_costs", "slippage_bps_per_leg"): 10.0,
        ("portfolio_gates", "base_net_sharpe_minimum"): 1.2,
        ("portfolio_gates", "stress_net_sharpe_minimum"): 1.2,
        ("selection_bias", "prior_manifest_count"): 347,
        ("selection_bias", "non_manifest_prefreeze_failed_attempt_count"): 1,
        ("selection_bias", "expected_total_trial_count_including_current"): 349,
    }
    for keys, wanted in expected.items():
        observed = _nested(config, keys)
        if observed != wanted:
            raise ContractError(
                f"冻结字段{'.'.join(keys)}异常：期望{wanted!r}，得到{observed!r}"
            )
    if config["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("持仓集合必须严格等于510300.SH与人民币现金")
    forbidden_scope = (
        "leverage_allowed",
        "short_selling_allowed",
        "derivatives_execution_allowed",
        "live_trading_authorized",
    )
    if any(bool(config["scope"][key]) for key in forbidden_scope):
        raise ContractError("作用域不得授权杠杆、卖空、衍生品或实盘")
    rescue_fields = (
        "parameter_rescue_after_result",
        "threshold_rescue_after_result",
        "window_rescue_after_result",
        "feature_rescue_after_result",
        "state_label_rescue_after_result",
        "range_policy_rescue_after_result",
        "direction_reversal_after_result",
        "combination_with_rejected_candidates_after_result",
    )
    if any(config["protocol"][key] != "forbidden" for key in rescue_fields):
        raise ContractError("结果后救援开关没有全部冻结为forbidden")
    governance_switches = (
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_enabled",
        "order_generation",
        "broker_connection",
        "position_change",
        "live_trading_authorized",
    )
    if any(bool(config["governance"][key]) for key in governance_switches):
        raise ContractError("研究协议不得产生信号、仓位、订单、券商连接或实盘")
    if config["mechanism_evaluation"]["forward_horizons_trading_days"] != [5, 20, 60]:
        raise ContractError("机制评价期限必须固定为5、20、60日")
    if config["mechanism_evaluation"]["primary_horizon_trading_days"] != 20:
        raise ContractError("主要状态期限必须固定为20日")
    if len(config["features"]["definitions"]) != len(FEATURE_NAMES):
        raise ContractError("技术特征数量必须严格等于三个")
    observed_names = tuple(item["name"] for item in config["features"]["definitions"])
    if observed_names != FEATURE_NAMES:
        raise ContractError("技术特征名称或顺序发生漂移")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("冻结清单不存在，禁止读取2015年后的候选结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("冻结清单项目编号不匹配")
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_2015_PLUS_CANDIDATE_EVALUATION":
        raise ContractError("冻结清单状态不允许候选评价")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后配置文件发生漂移")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件发生漂移：{mismatches}")
    visibility_fields = (
        "candidate_2015_plus_states_read_before_freeze",
        "candidate_2015_plus_outcomes_read_before_freeze",
        "candidate_portfolio_returns_read_before_freeze",
    )
    if any(manifest.get(field) is not False for field in visibility_fields):
        raise ContractError("冻结前候选可见性证明失败")
    selection = manifest.get("selection_bias_control", {})
    if int(selection.get("prior_manifest_count", -1)) != int(
        config["selection_bias"]["prior_manifest_count"]
    ):
        raise ContractError("冻结前清单计数不符合协议")
    if int(selection.get("total_trial_count_including_current", -1)) != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        raise ContractError("累计试验计数不符合协议")
    return manifest


def compute_feature_panel(index_daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "close"}
    missing = sorted(required.difference(index_daily.columns))
    if missing:
        raise ContractError(f"000300日线缺少特征字段：{missing}")
    frame = index_daily[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if frame["date"].duplicated().any():
        raise ContractError("000300特征输入存在重复日期")
    if frame["close"].isna().any() or (frame["close"] <= 0).any():
        raise ContractError("000300特征输入存在无效收盘价")
    log_close = np.log(frame["close"].to_numpy(dtype=float))
    daily_log_return = pd.Series(log_close).diff()
    frame[FEATURE_NAMES[0]] = (
        pd.Series(log_close).diff(5).to_numpy(dtype=float) / math.sqrt(5.0)
    )
    frame[FEATURE_NAMES[1]] = (
        pd.Series(log_close).diff(20).to_numpy(dtype=float) / math.sqrt(20.0)
    )
    realized_volatility = (
        daily_log_return.pow(2).rolling(20, min_periods=20).mean().pow(0.5)
        * math.sqrt(242.0)
    )
    frame[FEATURE_NAMES[2]] = np.log(realized_volatility.to_numpy(dtype=float))
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.dropna(subset=list(FEATURE_NAMES), inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if not np.isfinite(frame[list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
        raise ContractError("技术特征仍含非有限数")
    return frame


def _log_emission_density(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if observations.ndim != 2 or means.ndim != 2 or variances.shape != means.shape:
        raise ValueError("HMM发射参数维度不合法")
    if observations.shape[1] != means.shape[1]:
        raise ValueError("观察维数与发射均值维数不一致")
    if not np.isfinite(observations).all() or not np.isfinite(means).all():
        raise ValueError("HMM观察或均值包含非有限数")
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("HMM方差必须全部为有限正数")
    residual = observations[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        np.log(2.0 * math.pi * variances)[None, :, :]
        + residual * residual / variances[None, :, :],
        axis=2,
    )


def forward_backward(
    observations: np.ndarray,
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    probability_floor: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    observations = np.asarray(observations, dtype=float)
    initial_probabilities = np.asarray(initial_probabilities, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    state_count = means.shape[0]
    if observations.ndim != 2 or len(observations) < 2:
        raise ValueError("HMM训练至少需要两行二维观察")
    if initial_probabilities.shape != (state_count,):
        raise ValueError("HMM初始概率维度不合法")
    if transition_matrix.shape != (state_count, state_count):
        raise ValueError("HMM转移矩阵维度不合法")
    if probability_floor <= 0:
        raise ValueError("概率下限必须为正数")
    initial = np.maximum(initial_probabilities, probability_floor)
    initial /= initial.sum()
    transition = np.maximum(transition_matrix, probability_floor)
    transition /= transition.sum(axis=1, keepdims=True)
    log_initial = np.log(initial)
    log_transition = np.log(transition)
    log_emission = _log_emission_density(observations, means, variances)
    row_count = len(observations)
    log_alpha = np.empty((row_count, state_count), dtype=float)
    log_alpha[0] = log_initial + log_emission[0]
    for row in range(1, row_count):
        log_alpha[row] = log_emission[row] + logsumexp(
            log_alpha[row - 1][:, None] + log_transition,
            axis=0,
        )
    log_likelihood = float(logsumexp(log_alpha[-1]))
    log_beta = np.zeros((row_count, state_count), dtype=float)
    for row in range(row_count - 2, -1, -1):
        log_beta[row] = logsumexp(
            log_transition
            + log_emission[row + 1][None, :]
            + log_beta[row + 1][None, :],
            axis=1,
        )
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)
    expected_transitions = np.zeros((state_count, state_count), dtype=float)
    for row in range(row_count - 1):
        log_xi = (
            log_alpha[row][:, None]
            + log_transition
            + log_emission[row + 1][None, :]
            + log_beta[row + 1][None, :]
            - log_likelihood
        )
        expected_transitions += np.exp(log_xi)
    return log_likelihood, gamma, expected_transitions


def causal_filter(
    observations: np.ndarray,
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    probability_floor: float,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    transition = np.asarray(transition_matrix, dtype=float)
    posterior = np.asarray(initial_probabilities, dtype=float)
    posterior = np.maximum(posterior, probability_floor)
    posterior /= posterior.sum()
    emissions = _log_emission_density(observations, means, variances)
    filtered = np.empty((len(observations), len(posterior)), dtype=float)
    for row in range(len(observations)):
        prior = posterior if row == 0 else posterior @ transition
        log_posterior = np.log(np.maximum(prior, probability_floor)) + emissions[row]
        log_posterior -= float(logsumexp(log_posterior))
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()
        filtered[row] = posterior
    return filtered


def update_posterior(
    previous_posterior: np.ndarray,
    observation: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    probability_floor: float,
) -> np.ndarray:
    prior = np.asarray(previous_posterior, dtype=float) @ np.asarray(
        transition_matrix, dtype=float
    )
    log_emission = _log_emission_density(
        np.asarray(observation, dtype=float).reshape(1, -1), means, variances
    )[0]
    log_posterior = np.log(np.maximum(prior, probability_floor)) + log_emission
    log_posterior -= float(logsumexp(log_posterior))
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()
    return posterior


def fit_diagonal_gaussian_hmm(
    observations: np.ndarray,
    state_count: int,
    random_state: int,
    kmeans_n_init: int,
    max_iterations: int,
    tolerance: float,
    variance_floor: float,
    transition_pseudocount: float,
    probability_floor: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or len(observations) <= state_count:
        raise ValueError("HMM训练观察数量不足")
    if not np.isfinite(observations).all():
        raise ValueError("HMM训练观察包含非有限数")
    kmeans = KMeans(
        n_clusters=state_count,
        n_init=kmeans_n_init,
        random_state=random_state,
        algorithm="lloyd",
    )
    labels = kmeans.fit_predict(observations)
    means = np.asarray(kmeans.cluster_centers_, dtype=float)
    global_variance = np.var(observations, axis=0, ddof=0)
    variances = np.empty_like(means)
    for state in range(state_count):
        cluster = observations[labels == state]
        variances[state] = (
            np.var(cluster, axis=0, ddof=0) if len(cluster) >= 2 else global_variance
        )
    variances = np.maximum(variances, variance_floor)
    initial_probabilities = np.bincount(labels, minlength=state_count).astype(float) + 1.0
    initial_probabilities /= initial_probabilities.sum()
    transition_counts = np.full(
        (state_count, state_count), transition_pseudocount, dtype=float
    )
    for previous, current in zip(labels[:-1], labels[1:]):
        transition_counts[int(previous), int(current)] += 1.0
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    converged = False
    iteration_count = 0
    first_log_likelihood: float | None = None
    last_improvement: float | None = None
    minimum_improvement = math.inf
    for iteration in range(1, max_iterations + 1):
        log_likelihood, gamma, expected_transitions = forward_backward(
            observations,
            initial_probabilities,
            transition_matrix,
            means,
            variances,
            probability_floor,
        )
        if first_log_likelihood is None:
            first_log_likelihood = log_likelihood
        weights = gamma.sum(axis=0)
        if (weights <= probability_floor).any():
            raise ContractError("HMM训练出现空潜在状态")
        updated_initial = np.maximum(gamma[0], probability_floor)
        updated_initial /= updated_initial.sum()
        updated_transition = expected_transitions + transition_pseudocount
        updated_transition /= updated_transition.sum(axis=1, keepdims=True)
        updated_means = gamma.T @ observations / weights[:, None]
        residual = observations[:, None, :] - updated_means[None, :, :]
        updated_variances = np.einsum(
            "nk,nkd->kd", gamma, residual * residual
        ) / weights[:, None]
        updated_variances = np.maximum(updated_variances, variance_floor)
        updated_log_likelihood, updated_gamma, _ = forward_backward(
            observations,
            updated_initial,
            updated_transition,
            updated_means,
            updated_variances,
            probability_floor,
        )
        improvement = float(updated_log_likelihood - log_likelihood)
        minimum_improvement = min(minimum_improvement, improvement)
        if improvement < -1e-6:
            raise ContractError(f"HMM EM对数似然下降：{improvement}")
        initial_probabilities = updated_initial
        transition_matrix = updated_transition
        means = updated_means
        variances = updated_variances
        gamma = updated_gamma
        iteration_count = iteration
        last_improvement = improvement
        if improvement <= tolerance:
            converged = True
            break
    final_log_likelihood, final_gamma, _ = forward_backward(
        observations,
        initial_probabilities,
        transition_matrix,
        means,
        variances,
        probability_floor,
    )
    parameters = {
        "initial_probabilities": initial_probabilities,
        "transition_matrix": transition_matrix,
        "means": means,
        "variances": variances,
    }
    diagnostics = {
        "converged": bool(converged),
        "iterations": int(iteration_count),
        "maximum_iterations": int(max_iterations),
        "initial_log_likelihood": float(first_log_likelihood),
        "final_log_likelihood": float(final_log_likelihood),
        "last_em_improvement": float(last_improvement),
        "minimum_em_improvement": float(minimum_improvement),
        "variance_floor_bindings": int(np.isclose(variances, variance_floor).sum()),
    }
    return parameters, diagnostics, final_gamma


def fit_prefreeze_model(
    index_daily: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    feature_panel = compute_feature_panel(index_daily)
    training_start = pd.Timestamp(config["dates"]["training_start"])
    training_end = pd.Timestamp(config["dates"]["training_end"])
    training = feature_panel.loc[
        feature_panel["date"].between(training_start, training_end)
    ].copy()
    expected_rows = int(config["data_contract"]["expected_training_feature_rows"])
    if len(training) != expected_rows:
        raise ContractError(
            f"训练特征行数异常：期望{expected_rows}，得到{len(training)}"
        )
    if str(training["date"].min().date()) != config["dates"]["training_start"]:
        raise ContractError("训练特征首日不符合冻结合同")
    if str(training["date"].max().date()) != config["dates"]["training_end"]:
        raise ContractError("训练特征末日不符合冻结合同")
    raw = training[list(FEATURE_NAMES)].to_numpy(dtype=float)
    feature_mean = raw.mean(axis=0)
    feature_std = raw.std(axis=0, ddof=0)
    if (feature_std <= 0).any() or not np.isfinite(feature_std).all():
        raise ContractError("训练期特征标准差无效")
    standardized = (raw - feature_mean) / feature_std
    model_config = config["hidden_markov_model"]
    parameters, diagnostics, gamma = fit_diagonal_gaussian_hmm(
        standardized,
        int(model_config["latent_state_count"]),
        int(model_config["random_state"]),
        int(model_config["kmeans_n_init"]),
        int(model_config["expectation_maximization_max_iterations"]),
        float(model_config["convergence_tolerance"]),
        float(model_config["variance_floor"]),
        float(model_config["transition_pseudocount"]),
        float(model_config["probability_floor"]),
    )
    raw_emission_means = (
        parameters["means"] * feature_std[None, :] + feature_mean[None, :]
    )
    return20_order = np.argsort(raw_emission_means[:, 1], kind="stable")
    latent_to_state = {
        str(int(return20_order[0])): STATE_BEAR,
        str(int(return20_order[1])): STATE_RANGE,
        str(int(return20_order[2])): STATE_BULL,
    }
    semantic_to_latent = {
        state: int(latent) for latent, state in latent_to_state.items()
    }
    filtered_training = causal_filter(
        standardized,
        parameters["initial_probabilities"],
        parameters["transition_matrix"],
        parameters["means"],
        parameters["variances"],
        float(model_config["probability_floor"]),
    )
    effective_mass = gamma.mean(axis=0)
    minimum_mass = float(model_config["minimum_effective_training_mass_per_latent_state"])
    training_gate = bool(
        diagnostics["converged"]
        and np.isfinite(effective_mass).all()
        and float(effective_mass.min()) >= minimum_mass
        and diagnostics["variance_floor_bindings"] == 0
    )
    status = (
        "PASS_PREFIT_TRAINING_ONLY_MODEL_IDENTIFIABILITY"
        if training_gate
        else "FAIL_PREFIT_TRAINING_ONLY_MODEL_IDENTIFIABILITY"
    )
    return {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "status": status,
        "created_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256_at_prefit": sha256_file(CONFIG_PATH),
        "training": {
            "start": str(training["date"].min().date()),
            "end": str(training["date"].max().date()),
            "rows": int(len(training)),
            "candidate_2015_plus_states_computed": False,
            "candidate_2015_plus_outcomes_read_or_computed": False,
            "candidate_portfolio_returns_read_or_computed": False,
        },
        "feature_names": list(FEATURE_NAMES),
        "standardization": {
            "mean": feature_mean.tolist(),
            "population_std": feature_std.tolist(),
            "fit_period": "2012-06-26_TO_2014-12-31_ONLY",
        },
        "parameters": {
            key: value.tolist() for key, value in parameters.items()
        },
        "raw_emission_feature_means": raw_emission_means.tolist(),
        "state_mapping": {
            "rule": model_config["state_label_rule"],
            "latent_to_semantic": latent_to_state,
            "semantic_to_latent": semantic_to_latent,
        },
        "training_diagnostics": {
            **diagnostics,
            "effective_state_mass": effective_mass.tolist(),
            "minimum_effective_state_mass": float(effective_mass.min()),
            "required_minimum_effective_state_mass": minimum_mass,
            "training_gate_passed": training_gate,
        },
        "filtered_posterior_at_training_end": filtered_training[-1].tolist(),
        "boundaries": {
            "model_selection_or_hyperparameter_search": "FORBIDDEN",
            "fit_data_end_must_not_exceed": config["dates"]["training_end"],
            "post_2014_candidate_information_used": False,
        },
    }


def _file_summary(path: Path, date_column: str = "date") -> dict[str, Any]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    summary: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
    }
    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="raise")
        summary["first_date"] = str(dates.min().date())
        summary["last_date"] = str(dates.max().date())
    return summary


def build_prefreeze_audit(
    config: dict[str, Any], model_payload: dict[str, Any], model_path: Path
) -> dict[str, Any]:
    if model_payload["status"] != "PASS_PREFIT_TRAINING_ONLY_MODEL_IDENTIFIABILITY":
        raise ContractError("训练期模型可识别性门失败，不得生成通过型输入审计")
    for name, specification in config["inputs"].items():
        if name == "input_audit":
            continue
        path = _project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"预冻结输入缺失：{name}")
        expected_hash = specification.get("sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise ContractError(f"预冻结输入哈希不匹配：{name}")
    range_config = config["range_policy"]
    intraday_path = _project_path(range_config["observed_intraday_data"]["path"])
    quality_path = _project_path(range_config["source_quality_receipt"]["path"])
    intraday_result_path = _project_path(
        range_config["existing_intraday_family_result"]["path"]
    )
    for path in (intraday_path, quality_path, intraday_result_path):
        if not path.exists():
            raise ContractError(f"震荡做T边界证据缺失：{path}")
    if sha256_file(intraday_path) != range_config["observed_intraday_data"]["sha256"]:
        raise ContractError("分钟数据哈希不符合冻结合同")
    if sha256_file(quality_path) != range_config["source_quality_receipt"]["sha256"]:
        raise ContractError("分钟数据质量回执哈希不符合冻结合同")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    existing_intraday = json.loads(intraday_result_path.read_text(encoding="utf-8"))
    if quality.get("status") != range_config["source_quality_receipt"]["required_status"]:
        raise ContractError("分钟数据质量回执状态不符合冻结合同")
    if (
        existing_intraday.get("status")
        != range_config["existing_intraday_family_result"]["required_status"]
    ):
        raise ContractError("既有分钟候选状态不符合冻结合同")
    daily = {
        "etf_daily": _file_summary(_project_path(config["inputs"]["etf_daily"]["path"])),
        "index_daily": _file_summary(
            _project_path(config["inputs"]["index_daily"]["path"])
        ),
        "benchmark_total_return": _file_summary(
            _project_path(config["inputs"]["benchmark_total_return"]["path"])
        ),
        "dividends": _file_summary(
            _project_path(config["inputs"]["dividends"]["path"]), "ex_date"
        ),
    }
    observed_intraday = _file_summary(intraday_path, "trade_date")
    intraday_dates = pd.to_datetime(
        pd.read_parquet(intraday_path, columns=["trade_date"])["trade_date"],
        errors="raise",
    )
    observed_intraday["trading_days"] = int(intraday_dates.dt.normalize().nunique())
    coverage_complete = bool(
        observed_intraday["first_date"]
        <= range_config["required_intraday_start_for_historical_t"]
        and observed_intraday["last_date"]
        >= range_config["required_intraday_end_for_historical_t"]
    )
    if coverage_complete:
        raise ContractError("分钟覆盖事实与冻结的震荡不交易边界冲突")
    return {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "generated_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "status": "PASS_PREFREEZE_CAUSAL_HMM_MODEL_AND_DAILY_INPUTS_RANGE_T_NOT_ALLOWED",
        "candidate_2015_plus_states_read_or_computed": False,
        "candidate_2015_plus_outcomes_read_or_computed": False,
        "candidate_portfolio_returns_read_or_computed": False,
        "daily_inputs": daily,
        "prefreeze_model": {
            "path": model_path.relative_to(ROOT).as_posix(),
            "bytes": int(model_path.stat().st_size),
            "sha256": sha256_file(model_path),
            "status": model_payload["status"],
            "training": model_payload["training"],
            "training_diagnostics": model_payload["training_diagnostics"],
            "state_mapping": model_payload["state_mapping"],
            "post_2014_candidate_information_used": False,
        },
        "range_t_data_gate": {
            "status": "FAIL_COMPLETE_2015_TO_2026_INTRADAY_COVERAGE",
            "range_t_return_evaluation": "NOT_ALLOWED",
            "fallback_range_policy": "NO_TRADE",
            "coverage_complete": False,
            "required_first_date": range_config[
                "required_intraday_start_for_historical_t"
            ],
            "required_last_date": range_config["required_intraday_end_for_historical_t"],
            "observed_intraday": observed_intraday,
            "source_quality_receipt": {
                "path": quality_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(quality_path),
                "status": quality["status"],
            },
            "existing_intraday_family_status": existing_intraday["status"],
        },
        "official_trading_rule": {
            "stock_etf_t_plus_one": True,
            "same_day_new_buy_then_sell_allowed": False,
            "inventory_sell_then_buy_back_requires_separate_inventory_model": True,
            "sources": range_config["official_rule_sources"],
        },
        "admission": {
            "daily_probabilistic_three_state_strategy_allowed": True,
            "historical_range_t_allowed": False,
            "reason": "训练期模型可识别且日线合同完整；完整分钟合同不足，震荡阶段固定不交易。",
        },
    }


def load_frozen_model(config: dict[str, Any]) -> dict[str, Any]:
    path = _project_path(config["hidden_markov_model"]["frozen_model_path"])
    if not path.exists():
        raise ContractError("冻结HMM模型不存在")
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("冻结HMM模型项目编号不匹配")
    if model.get("status") != "PASS_PREFIT_TRAINING_ONLY_MODEL_IDENTIFIABILITY":
        raise ContractError("冻结HMM模型训练期门未通过")
    if model.get("config_sha256_at_prefit") != sha256_file(CONFIG_PATH):
        raise ContractError("模型拟合后配置发生漂移")
    training = model.get("training", {})
    if training.get("end") != config["dates"]["training_end"]:
        raise ContractError("冻结HMM模型训练截止日异常")
    if any(
        training.get(field) is not False
        for field in (
            "candidate_2015_plus_states_computed",
            "candidate_2015_plus_outcomes_read_or_computed",
            "candidate_portfolio_returns_read_or_computed",
        )
    ):
        raise ContractError("冻结模型使用了2015年后的候选信息")
    if tuple(model.get("feature_names", [])) != FEATURE_NAMES:
        raise ContractError("冻结HMM模型特征顺序异常")
    mapping = model.get("state_mapping", {})
    if set(mapping.get("semantic_to_latent", {})) != set(ALL_STATES):
        raise ContractError("冻结HMM模型缺少三态映射")
    return model


def _model_arrays(model: dict[str, Any]) -> tuple[np.ndarray, ...]:
    parameters = model["parameters"]
    initial = np.asarray(parameters["initial_probabilities"], dtype=float)
    transition = np.asarray(parameters["transition_matrix"], dtype=float)
    means = np.asarray(parameters["means"], dtype=float)
    variances = np.asarray(parameters["variances"], dtype=float)
    feature_mean = np.asarray(model["standardization"]["mean"], dtype=float)
    feature_std = np.asarray(model["standardization"]["population_std"], dtype=float)
    if initial.shape != (3,) or transition.shape != (3, 3):
        raise ContractError("冻结HMM概率参数维度异常")
    if means.shape != (3, 3) or variances.shape != (3, 3):
        raise ContractError("冻结HMM发射参数维度异常")
    if feature_mean.shape != (3,) or feature_std.shape != (3,):
        raise ContractError("冻结HMM标准化参数维度异常")
    if not np.allclose(initial.sum(), 1.0, atol=1e-10):
        raise ContractError("冻结HMM初始概率不归一")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise ContractError("冻结HMM转移概率不归一")
    if (variances <= 0).any() or (feature_std <= 0).any():
        raise ContractError("冻结HMM方差或标准差无效")
    return initial, transition, means, variances, feature_mean, feature_std


def classify_next_state_probabilities(
    next_probabilities: np.ndarray,
    semantic_to_latent: dict[str, int],
    threshold: float,
) -> tuple[str, str]:
    probabilities = np.asarray(next_probabilities, dtype=float)
    bull = float(probabilities[semantic_to_latent[STATE_BULL]])
    bear = float(probabilities[semantic_to_latent[STATE_BEAR]])
    range_probability = float(probabilities[semantic_to_latent[STATE_RANGE]])
    if bull >= threshold and bull > bear and bull > range_probability:
        return STATE_BULL, "NEXT_BULL_MAJORITY"
    if bear >= threshold and bear > bull and bear > range_probability:
        return STATE_BEAR, "NEXT_BEAR_MAJORITY"
    return STATE_RANGE, "NO_DIRECTIONAL_MAJORITY"


def build_state_panel(
    index_daily: pd.DataFrame, model: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    features = compute_feature_panel(index_daily)
    initial, transition, means, variances, feature_mean, feature_std = _model_arrays(
        model
    )
    standardized = (
        features[list(FEATURE_NAMES)].to_numpy(dtype=float) - feature_mean[None, :]
    ) / feature_std[None, :]
    training_end = pd.Timestamp(config["dates"]["training_end"])
    training_mask = features["date"].le(training_end).to_numpy()
    training_observations = standardized[training_mask]
    if len(training_observations) != int(
        config["data_contract"]["expected_training_feature_rows"]
    ):
        raise ContractError("冻结模型复核时训练特征行数异常")
    probability_floor = float(config["hidden_markov_model"]["probability_floor"])
    training_filtered = causal_filter(
        training_observations,
        initial,
        transition,
        means,
        variances,
        probability_floor,
    )
    frozen_final = np.asarray(model["filtered_posterior_at_training_end"], dtype=float)
    if not np.allclose(training_filtered[-1], frozen_final, atol=1e-10, rtol=1e-10):
        raise ContractError("训练期最终过滤概率不能精确复现冻结模型")
    post_training = features.loc[features["date"].gt(training_end)].copy()
    post_training_observations = standardized[~training_mask]
    semantic_to_latent = {
        state: int(latent)
        for state, latent in model["state_mapping"]["semantic_to_latent"].items()
    }
    threshold = float(config["decision_rule"]["majority_threshold"])
    previous_posterior = frozen_final.copy()
    filtered_rows: list[np.ndarray] = []
    next_rows: list[np.ndarray] = []
    states: list[str] = []
    reasons: list[str] = []
    for observation in post_training_observations:
        filtered = update_posterior(
            previous_posterior,
            observation,
            transition,
            means,
            variances,
            probability_floor,
        )
        next_probabilities = filtered @ transition
        next_probabilities /= next_probabilities.sum()
        state, reason = classify_next_state_probabilities(
            next_probabilities, semantic_to_latent, threshold
        )
        filtered_rows.append(filtered)
        next_rows.append(next_probabilities)
        states.append(state)
        reasons.append(reason)
        previous_posterior = filtered
    filtered_matrix = np.vstack(filtered_rows)
    next_matrix = np.vstack(next_rows)
    if not np.allclose(filtered_matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ContractError("样本外过滤概率不归一")
    if not np.allclose(next_matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ContractError("样本外下一状态概率不归一")
    for state in ALL_STATES:
        latent = semantic_to_latent[state]
        post_training[f"filtered_probability::{state}"] = filtered_matrix[:, latent]
        post_training[f"next_probability::{state}"] = next_matrix[:, latent]
    post_training["state"] = states
    post_training["decision_reason"] = reasons
    post_training["maximum_next_probability"] = next_matrix.max(axis=1)
    post_training["target_position"] = np.where(
        post_training["state"].eq(STATE_BULL), 1.0, 0.0
    )
    post_training["state_changed"] = post_training["state"].ne(
        post_training["state"].shift()
    )
    post_training["episode_id"] = post_training["state_changed"].cumsum().astype(int)
    post_training["episode_row"] = (
        post_training.groupby("episode_id").cumcount().add(1).astype(int)
    )
    post_training.reset_index(drop=True, inplace=True)
    return post_training


def build_targets(state_panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    targets = state_panel.loc[
        state_panel["date"].between(start, end), ["date", "state", "target_position"]
    ].copy()
    targets["risk_off_override"] = targets["state"].ne(STATE_BULL)
    targets["trade_allowed"] = True
    targets["signal_reason"] = "因果HMM下一状态路由::" + targets["state"].astype(str)
    return targets


def probability_diagnostics(
    state_panel: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    evaluation = state_panel.loc[state_panel["date"].between(start, end)].copy()
    by_state: dict[str, Any] = {}
    for state in ALL_STATES:
        values = evaluation.loc[
            evaluation["state"].eq(state), "maximum_next_probability"
        ].to_numpy(dtype=float)
        by_state[state] = {
            "days": int(len(values)),
            "mean_maximum_next_probability": float(values.mean()) if len(values) else None,
            "median_maximum_next_probability": (
                float(np.median(values)) if len(values) else None
            ),
        }
    return {
        "evaluation_rows": int(len(evaluation)),
        "majority_threshold": float(config["decision_rule"]["majority_threshold"]),
        "overall_mean_maximum_next_probability": float(
            evaluation["maximum_next_probability"].mean()
        ),
        "overall_median_maximum_next_probability": float(
            evaluation["maximum_next_probability"].median()
        ),
        "by_decision_state": by_state,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}%}"


def render_markdown(report: dict[str, Any]) -> str:
    mechanism = report["mechanism_evaluation"]
    portfolio = report["portfolio_evaluation"]
    base = portfolio["base"]
    stress = portfolio["stress"]
    full = mechanism["periods"]["FULL"]
    lines = [
        "# 510300 因果高斯 HMM 三阶段路由 V1 结果",
        "",
        f"- 状态：`{report['status']}`",
        f"- 训练区间：{report['model']['training']['start']}—{report['model']['training']['end']}，仅用于拟合。",
        f"- 一次性评价区间：{report['evaluation_window']['start']}—{report['evaluation_window']['end']}。",
        "- 动作：下一阶段上涨概率过半且最高时100%；下一阶段下跌或无方向多数时0%。",
        "- 震荡做T：`NOT_ALLOWED_DATA_AND_T_PLUS_ONE_GATE_FAILED`。",
        "",
        "## 状态识别",
        "",
        "| 决策状态 | 交易日 | 独立区间 | 未来20日均值 | 正收益率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for state in (STATE_BULL, STATE_RANGE, STATE_BEAR):
        lines.append(
            f"| `{state}` | {mechanism['state_days'][state]} | "
            f"{mechanism['episode_counts'][state]} | "
            f"{_format_percent(full[state]['mean'])} | "
            f"{_format_percent(full[state]['positive_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"- 状态机制门：`{'PASS' if mechanism['passed'] else 'FAIL'}`。",
            f"- 下一状态最大概率中位数：{_format_percent(report['probability_diagnostics']['overall_median_maximum_next_probability'])}。",
            "",
            "## 组合结果",
            "",
            "| 成本口径 | 净夏普 | 年化收益 | 最大回撤 | 总收益 | 交易次数 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 基准成本 | {_format_number(base['sharpe_zero_cash_rate'])} | "
            f"{_format_percent(base['cagr'])} | {_format_percent(base['max_drawdown'])} | "
            f"{_format_percent(base['total_return'])} | {base['trade_count']} |",
            f"| 压力成本 | {_format_number(stress['sharpe_zero_cash_rate'])} | "
            f"{_format_percent(stress['cagr'])} | {_format_percent(stress['max_drawdown'])} | "
            f"{_format_percent(stress['total_return'])} | {stress['trade_count']} |",
            "",
            f"- 买入持有基准成本夏普：{_format_number(portfolio['buy_hold_base']['sharpe_zero_cash_rate'])}。",
            f"- 最大回撤比例（策略/买入持有）：{_format_number(portfolio['maximum_drawdown_ratio_vs_buy_hold'])}。",
            f"- 历史净夏普1.2目标：`{'PASS' if report['adjudication']['historical_target_achieved'] else 'FAIL'}`。",
            "- 独立前向目标：`NOT_ACHIEVED`。",
            "",
            "## 边界",
            "",
            "结果出现后不得调整特征、HMM参数、训练窗口、状态映射、50%阈值、震荡动作或拼接既有失败候选。",
            "本结果不生成Paper、Shadow、仓位、订单、券商连接或实盘授权。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    model = load_frozen_model(config)
    etf, index, total_return_index, dividends, input_audit = load_daily_inputs(config)
    state_panel = build_state_panel(index, model, config)
    mechanism, episodes = evaluate_state_mechanism(
        state_panel, total_return_index, config
    )
    targets = build_targets(state_panel, config)
    portfolio, portfolio_frames = evaluate_portfolio(
        etf, dividends, targets, bool(mechanism["passed"]), config
    )
    historical_pass = bool(portfolio["passed"])
    status = (
        "PASS_HISTORICAL_CAUSAL_HMM_REGIME_ROUTER_FORWARD_REQUIRED"
        if historical_pass
        else "REJECTED_FROZEN_CAUSAL_HMM_REGIME_ROUTER_TARGET_OR_STATE_GATE_FAILED_NO_RESCUE"
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "scope": {
            "execution_asset": config["scope"]["execution_asset"],
            "signal_asset": config["scope"]["signal_asset"],
            "allowed_holdings": config["scope"]["allowed_holdings"],
            "long_only": True,
            "leverage_allowed": False,
        },
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
            "trading_days": int(config["data_contract"]["expected_evaluation_rows"]),
        },
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "candidate_2015_plus_states_read_before_freeze": False,
            "candidate_2015_plus_outcomes_read_before_freeze": False,
            "candidate_portfolio_returns_read_before_freeze": False,
            "selection_bias_control": manifest["selection_bias_control"],
        },
        "model": {
            "family": config["hidden_markov_model"]["model_family"],
            "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
            "model_sha256": sha256_file(MODEL_PATH),
            "training": model["training"],
            "training_diagnostics": model["training_diagnostics"],
            "feature_names": model["feature_names"],
            "state_mapping": model["state_mapping"],
            "transition_matrix": model["parameters"]["transition_matrix"],
            "decision_probability": "ONE_STEP_AHEAD_NEXT_STATE",
            "majority_threshold": config["decision_rule"]["majority_threshold"],
        },
        "state_definition": {
            "position_mapping": {
                state: config["states"][state]["target_position"] for state in ALL_STATES
            },
            "tie_or_no_majority": STATE_RANGE,
        },
        "range_policy": {
            "historical_policy": "NO_TRADE",
            "range_t_return_evaluation": "NOT_ALLOWED",
            "data_gate_status": input_audit["range_t_data_gate"]["status"],
            "stock_etf_t_plus_one": True,
            "observed_intraday_start": input_audit["range_t_data_gate"][
                "observed_intraday"
            ]["first_date"],
            "required_intraday_start": input_audit["range_t_data_gate"][
                "required_first_date"
            ],
        },
        "probability_diagnostics": probability_diagnostics(state_panel, config),
        "mechanism_evaluation": mechanism,
        "portfolio_evaluation": portfolio,
        "adjudication": {
            "return_evaluation": "COMPLETED_FIXED_CAUSAL_HMM_THREE_STATE_MAPPING",
            "base_net_sharpe": portfolio["base"]["sharpe_zero_cash_rate"],
            "stress_net_sharpe": portfolio["stress"]["sharpe_zero_cash_rate"],
            "target_net_sharpe": config["objective"]["target_net_sharpe"],
            "historical_target_achieved": historical_pass,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "reason": (
                "历史状态门与组合硬门全部通过，但仍需独立前向验证"
                if historical_pass
                else "状态机制门或净组合硬门至少一项失败，冻结协议禁止结果后救援"
            ),
        },
        "boundaries": {
            "parameter_feature_threshold_or_window_rescue": "FORBIDDEN",
            "state_mapping_or_range_policy_rescue": "FORBIDDEN",
            "combination_with_rejected_candidates": "FORBIDDEN",
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "position_mapping_enabled": False,
            "order_generation": False,
            "broker_connection": False,
            "position_change": False,
            "live_trading_authorized": False,
        },
        "artifacts": {},
    }
    if write:
        paths = {key: _project_path(value) for key, value in config["paths"].items()}
        for key in (
            "state_panel",
            "targets",
            "base_ledger",
            "base_trades",
            "stress_ledger",
            "stress_trades",
            "result_json",
            "result_markdown",
        ):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        evaluation_output = state_panel.loc[
            state_panel["date"].between(
                pd.Timestamp(config["dates"]["evaluation_start"]),
                pd.Timestamp(config["dates"]["evaluation_end"]),
            )
        ].copy()
        evaluation_output.to_parquet(paths["state_panel"], index=False)
        targets.to_parquet(paths["targets"], index=False)
        portfolio_frames["base_ledger"].to_parquet(paths["base_ledger"], index=False)
        portfolio_frames["base_trades"].to_parquet(paths["base_trades"], index=False)
        portfolio_frames["stress_ledger"].to_parquet(paths["stress_ledger"], index=False)
        portfolio_frames["stress_trades"].to_parquet(paths["stress_trades"], index=False)
        artifact_keys = (
            "state_panel",
            "targets",
            "base_ledger",
            "base_trades",
            "stress_ledger",
            "stress_trades",
        )
        report["artifacts"] = {
            key: _artifact_record(paths[key]) for key in artifact_keys
        }
        report["artifacts"]["episodes_not_persisted"] = {
            "rows": int(len(episodes)),
            "reason": "区间汇总已进入结果JSON，不增加重复文件",
        }
        paths["result_json"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths["result_markdown"].write_text(render_markdown(report), encoding="utf-8")
    return report
