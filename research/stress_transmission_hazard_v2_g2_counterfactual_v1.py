"""V2 的反事实 G2：历史 prequential B2 对 B1 结构增量检验。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


class G2CounterfactualError(RuntimeError):
    """反事实 G2 契约不满足。"""


B1_FEATURES = [
    "b1_realized_vol20_risk_percentile",
    "b1_drawdown20_risk_percentile",
    "b1_downside_return5_risk_percentile",
]
B2_FEATURES = ["T", "T_x_F"]
FORBIDDEN_INPUT_COLUMNS = {
    "minimum_path_return",
    "first_breach_date",
    "cash_dividend_per_share_in_horizon",
    "forward_return",
    "portfolio_return",
    "nav",
    "sharpe",
    "drawdown",
}


@dataclass(frozen=True)
class Era:
    era_id: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class LogisticFit:
    intercept: float
    slopes: tuple[float, ...]
    objective: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class G2CounterfactualArtifacts:
    predictions: pd.DataFrame
    model_vintages: pd.DataFrame
    era_metrics: pd.DataFrame
    bootstrap_distribution: pd.DataFrame
    result: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise G2CounterfactualError(f"{label}缺少列：{missing}")


def _reject_forbidden_columns(frame: pd.DataFrame, label: str) -> None:
    present = sorted(FORBIDDEN_INPUT_COLUMNS.intersection(map(str, frame.columns)))
    if present:
        raise G2CounterfactualError(f"{label}包含禁止读取列：{present}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce").dt.normalize()
    if result.isna().any():
        raise G2CounterfactualError(f"{label}存在非法日期")
    return result


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if values.isna().any() or not values.isin([True, False]).all():
        raise G2CounterfactualError(f"{label}必须是非空布尔值")
    return values.astype(bool)


def parse_eras(raw: Sequence[Mapping[str, Any]]) -> list[Era]:
    if len(raw) != 4:
        raise G2CounterfactualError("G2 必须使用恰好四个冻结时代")
    eras: list[Era] = []
    for item in raw:
        era_id = str(item.get("id", "")).strip()
        start = pd.to_datetime(item.get("start"), errors="coerce")
        end = pd.to_datetime(item.get("end"), errors="coerce")
        if not era_id or pd.isna(start) or pd.isna(end) or start > end:
            raise G2CounterfactualError(f"非法时代：{item}")
        eras.append(
            Era(
                era_id=era_id,
                start=pd.Timestamp(start).normalize(),
                end=pd.Timestamp(end).normalize(),
            )
        )
    for previous, current in zip(eras, eras[1:]):
        if current.start <= previous.end:
            raise G2CounterfactualError("四个时代必须递增且不重叠")
    return eras


def _assign_era(date: pd.Timestamp, eras: Sequence[Era]) -> str:
    matches = [era.era_id for era in eras if era.start <= date <= era.end]
    if len(matches) != 1:
        raise G2CounterfactualError(f"日期无法唯一分配时代：{date.date()}")
    return matches[0]


def _weighted_log_loss(
    outcome: np.ndarray, probability: np.ndarray, weight: np.ndarray, clip: float
) -> float:
    p = np.clip(probability.astype(float), clip, 1.0 - clip)
    y = outcome.astype(float)
    w = weight.astype(float)
    if len(y) == 0 or not np.isfinite(w).all() or w.sum() <= 0.0:
        raise G2CounterfactualError("Log Loss 输入为空或权重非法")
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.average(loss, weights=w))


def fit_nonnegative_logistic(
    features: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    *,
    l2_penalty: float = 1.0,
    max_iterations: int = 2000,
    ftol: float = 1.0e-12,
    gtol: float = 1.0e-8,
) -> LogisticFit:
    """拟合截距不惩罚、所有斜率非负的固定 L2 逻辑回归。"""

    x = np.asarray(features, dtype=float)
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(weight, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(w) or len(y) == 0:
        raise G2CounterfactualError("逻辑回归输入形状非法")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise G2CounterfactualError("逻辑回归输入含非有限值")
    if not np.isin(y, [0.0, 1.0]).all() or len(np.unique(y)) != 2:
        raise G2CounterfactualError("逻辑回归训练集必须同时含正负类")
    if (w <= 0.0).any() or w.sum() <= 0.0:
        raise G2CounterfactualError("逻辑回归权重必须为有限正数")
    if not math.isclose(l2_penalty, 1.0, abs_tol=1e-12):
        raise G2CounterfactualError("L2 惩罚必须固定为 1.0")

    weighted_rate = float(np.average(y, weights=w))
    weighted_rate = float(np.clip(weighted_rate, 1.0e-8, 1.0 - 1.0e-8))
    initial = np.zeros(x.shape[1] + 1, dtype=float)
    initial[0] = math.log(weighted_rate / (1.0 - weighted_rate))
    weight_sum = float(w.sum())

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        slopes = parameters[1:]
        linear = intercept + x @ slopes
        probability = expit(linear)
        nll = np.logaddexp(0.0, linear) - y * linear
        objective = float(np.dot(w, nll) / weight_sum)
        objective += 0.5 * l2_penalty * float(np.dot(slopes, slopes))
        residual = w * (probability - y) / weight_sum
        gradient = np.empty_like(parameters)
        gradient[0] = residual.sum()
        gradient[1:] = x.T @ residual + l2_penalty * slopes
        return objective, gradient

    fitted = minimize(
        objective_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None), *[(0.0, None) for _ in range(x.shape[1])]],
        options={"maxiter": max_iterations, "ftol": ftol, "gtol": gtol},
    )
    if not fitted.success or not np.isfinite(fitted.fun):
        raise G2CounterfactualError(
            f"固定逻辑回归未收敛：status={fitted.status}, message={fitted.message}"
        )
    parameters = np.asarray(fitted.x, dtype=float)
    if (parameters[1:] < -1.0e-10).any():
        raise G2CounterfactualError("固定逻辑回归斜率违反非负约束")
    parameters[1:] = np.maximum(parameters[1:], 0.0)
    return LogisticFit(
        intercept=float(parameters[0]),
        slopes=tuple(float(value) for value in parameters[1:]),
        objective=float(fitted.fun),
        iterations=int(fitted.nit),
        converged=True,
    )


def _predict(fit: LogisticFit, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    return expit(fit.intercept + x @ np.asarray(fit.slopes, dtype=float))


def _prepare_common_sample(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    eras: Sequence[Era],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _reject_forbidden_columns(samples, "G1 样本准入")
    _reject_forbidden_columns(events, "G1 事件准入")
    required = [
        "origin_date",
        "horizon_end_date",
        "bad10",
        "event_id",
        "sample_group_id",
        "bootstrap_event_block_id",
        "calendar_year_block_id",
        *B1_FEATURES,
        "F",
        "T",
        "T_x_F",
        "b2_vs_b1_eligible",
        "b2_vs_b1_model_weight",
    ]
    _require_columns(samples, required, "G1 样本准入")
    _require_columns(
        events,
        [
            "event_id",
            "total_positive_origin_count",
            "b2_eligible_positive_origin_count",
            "b2_identifiable_event",
        ],
        "G1 事件准入",
    )
    work = samples.copy()
    work["origin_date"] = _normalize_dates(work["origin_date"], "原点日期")
    work["horizon_end_date"] = _normalize_dates(
        work["horizon_end_date"], "标签到期日"
    )
    if work["origin_date"].gt(cutoff).any() or work["horizon_end_date"].gt(cutoff).any():
        raise G2CounterfactualError("样本包含历史截止日之后观察")
    if work["horizon_end_date"].lt(work["origin_date"]).any():
        raise G2CounterfactualError("标签到期日早于原点")
    work["bad10"] = pd.to_numeric(work["bad10"], errors="coerce")
    if work["bad10"].isna().any() or not work["bad10"].isin([0, 1]).all():
        raise G2CounterfactualError("BAD10 必须且只能为 0/1")
    work["b2_vs_b1_eligible"] = _strict_bool(
        work["b2_vs_b1_eligible"], "B2/B1 共同样本状态"
    )
    work["event_id"] = work["event_id"].astype("string").fillna("").str.strip()
    work["sample_group_id"] = work["sample_group_id"].astype("string").str.strip()
    work["bootstrap_event_block_id"] = (
        work["bootstrap_event_block_id"].astype("string").str.strip()
    )
    work["calendar_year_block_id"] = (
        work["calendar_year_block_id"].astype("string").str.strip()
    )
    positives_all = work.loc[work["bad10"].eq(1)].copy()
    if positives_all.empty or positives_all["event_id"].eq("").any():
        raise G2CounterfactualError("BAD10 正样本事件标识非法")
    if work.loc[work["bad10"].eq(0), "event_id"].ne("").any():
        raise G2CounterfactualError("非事件风险日不得带正事件标识")

    event_summary = positives_all.groupby("event_id", sort=True).agg(
        event_min_origin_date=("origin_date", "min"),
        event_max_horizon_end_date=("horizon_end_date", "max"),
    )
    event_frame = events.copy()
    event_frame["event_id"] = event_frame["event_id"].astype("string").str.strip()
    if event_frame["event_id"].duplicated().any():
        raise G2CounterfactualError("G1 事件准入 event_id 不唯一")
    if set(event_summary.index.astype(str)) != set(event_frame["event_id"].astype(str)):
        raise G2CounterfactualError("样本事件集合与事件准入账本不一致")

    common = work.loc[work["b2_vs_b1_eligible"]].copy()
    if common.empty:
        raise G2CounterfactualError("B2/B1 共同样本为空")
    feature_columns = [*B1_FEATURES, *B2_FEATURES, "F"]
    for column in feature_columns:
        common[column] = pd.to_numeric(common[column], errors="coerce")
    if common[feature_columns].isna().any().any() or not np.isfinite(
        common[feature_columns].to_numpy(dtype=float)
    ).all():
        raise G2CounterfactualError("共同样本特征含空值或非有限值")
    if not np.allclose(
        common["T_x_F"].to_numpy(dtype=float),
        (common["T"] * common["F"]).to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise G2CounterfactualError("T_x_F 与冻结乘积不一致")
    common["sample_weight"] = pd.to_numeric(
        common["b2_vs_b1_model_weight"], errors="coerce"
    )
    if common["sample_weight"].isna().any() or (
        common["sample_weight"] <= 0.0
    ).any():
        raise G2CounterfactualError("共同样本权重必须为有限正数")
    positive_weight = (
        common.loc[common["bad10"].eq(1)]
        .groupby("event_id", sort=True)["sample_weight"]
        .sum()
    )
    if not np.allclose(positive_weight.to_numpy(dtype=float), 1.0, atol=1.0e-12):
        raise G2CounterfactualError("共同样本正事件权重和必须逐事件等于 1")
    if not common.loc[common["bad10"].eq(0), "sample_weight"].eq(1.0).all():
        raise G2CounterfactualError("共同样本非事件风险日权重必须等于 1")

    common = common.merge(
        event_summary,
        left_on="event_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    positive = common["bad10"].eq(1)
    common.loc[~positive, "event_min_origin_date"] = pd.NaT
    common.loc[~positive, "event_max_horizon_end_date"] = pd.NaT
    evaluation_date = common["origin_date"].where(
        ~positive, common["event_min_origin_date"]
    )
    common["evaluation_vintage_year"] = evaluation_date.dt.year.astype(int)
    common["era_id"] = [
        _assign_era(pd.Timestamp(date), eras) for date in evaluation_date
    ]
    common = common.sort_values(
        ["evaluation_vintage_year", "origin_date", "sample_group_id"],
        kind="stable",
    ).reset_index(drop=True)
    metadata = {
        "total_sample_count": int(len(work)),
        "total_positive_origin_count": int(work["bad10"].eq(1).sum()),
        "total_non_event_risk_day_count": int(work["bad10"].eq(0).sum()),
        "total_independent_event_count": int(len(event_summary)),
        "common_sample_day_count": int(len(common)),
        "common_positive_origin_count": int(common["bad10"].eq(1).sum()),
        "common_non_event_risk_day_count": int(common["bad10"].eq(0).sum()),
        "common_identifiable_event_count": int(
            common.loc[common["bad10"].eq(1), "event_id"].nunique()
        ),
    }
    return common, metadata


def _training_rows_for_vintage(
    common: pd.DataFrame, vintage_start: pd.Timestamp
) -> pd.DataFrame:
    negative = common["bad10"].eq(0) & common["horizon_end_date"].lt(vintage_start)
    positive = common["bad10"].eq(1) & common[
        "event_max_horizon_end_date"
    ].lt(vintage_start)
    training = common.loc[negative | positive].copy()
    if not training.empty and training["horizon_end_date"].ge(vintage_start).any():
        raise G2CounterfactualError("训练集含未到期标签")
    for _, group in training.loc[training["bad10"].eq(1)].groupby("event_id"):
        full = common.loc[common["event_id"].eq(group["event_id"].iloc[0])]
        if len(group) != len(full):
            raise G2CounterfactualError("训练集拆分了正事件")
    return training


def build_prequential_predictions(
    common: pd.DataFrame,
    *,
    first_vintage_year: int,
    last_vintage_year: int,
    l2_penalty: float,
    max_iterations: int,
    ftol: float,
    gtol: float,
    probability_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_pieces: list[pd.DataFrame] = []
    vintage_rows: list[dict[str, Any]] = []
    for year in range(first_vintage_year, last_vintage_year + 1):
        vintage_start = pd.Timestamp(year=year, month=1, day=1)
        training = _training_rows_for_vintage(common, vintage_start)
        train_events = int(
            training.loc[training["bad10"].eq(1), "event_id"].nunique()
        )
        train_negatives = int(training["bad10"].eq(0).sum())
        evaluation = common.loc[common["evaluation_vintage_year"].eq(year)].copy()
        base_row: dict[str, Any] = {
            "model_vintage_year": year,
            "model_vintage_start": vintage_start,
            "training_max_horizon_end_date": (
                training["horizon_end_date"].max() if not training.empty else pd.NaT
            ),
            "training_row_count": int(len(training)),
            "training_positive_origin_count": int(training["bad10"].eq(1).sum()),
            "training_positive_event_count": train_events,
            "training_negative_risk_day_count": train_negatives,
            "evaluation_row_count": int(len(evaluation)),
            "evaluation_positive_event_count": int(
                evaluation.loc[evaluation["bad10"].eq(1), "event_id"].nunique()
            ),
            "evaluation_negative_risk_day_count": int(
                evaluation["bad10"].eq(0).sum()
            ),
        }
        if training.empty or train_events < 1 or train_negatives < 1:
            vintage_rows.append(
                {
                    **base_row,
                    "model_state": "NO_VIEW_INSUFFICIENT_LABEL_MATURED_TRAINING_CLASSES",
                    "b0_probability": np.nan,
                    "b1_intercept": np.nan,
                    **{f"b1_slope_{column}": np.nan for column in B1_FEATURES},
                    "b1_objective": np.nan,
                    "b1_iterations": 0,
                    "b2_intercept": np.nan,
                    **{f"b2_slope_{column}": np.nan for column in B2_FEATURES},
                    "b2_objective": np.nan,
                    "b2_iterations": 0,
                }
            )
            continue

        y_train = training["bad10"].to_numpy(dtype=int)
        w_train = training["sample_weight"].to_numpy(dtype=float)
        b0_probability = float(
            np.clip(
                np.average(y_train.astype(float), weights=w_train),
                probability_clip,
                1.0 - probability_clip,
            )
        )
        b1 = fit_nonnegative_logistic(
            training[B1_FEATURES].to_numpy(dtype=float),
            y_train,
            w_train,
            l2_penalty=l2_penalty,
            max_iterations=max_iterations,
            ftol=ftol,
            gtol=gtol,
        )
        b2 = fit_nonnegative_logistic(
            training[B2_FEATURES].to_numpy(dtype=float),
            y_train,
            w_train,
            l2_penalty=l2_penalty,
            max_iterations=max_iterations,
            ftol=ftol,
            gtol=gtol,
        )
        vintage_rows.append(
            {
                **base_row,
                "model_state": "VIEW_ALLOWED_LABEL_MATURED_PREQUENTIAL",
                "b0_probability": b0_probability,
                "b1_intercept": b1.intercept,
                **{
                    f"b1_slope_{column}": value
                    for column, value in zip(B1_FEATURES, b1.slopes)
                },
                "b1_objective": b1.objective,
                "b1_iterations": b1.iterations,
                "b2_intercept": b2.intercept,
                **{
                    f"b2_slope_{column}": value
                    for column, value in zip(B2_FEATURES, b2.slopes)
                },
                "b2_objective": b2.objective,
                "b2_iterations": b2.iterations,
            }
        )
        if evaluation.empty:
            continue
        scored = evaluation[
            [
                "origin_date",
                "horizon_end_date",
                "bad10",
                "event_id",
                "sample_group_id",
                "bootstrap_event_block_id",
                "calendar_year_block_id",
                "sample_weight",
                "era_id",
                "evaluation_vintage_year",
            ]
        ].copy()
        scored["model_vintage_start"] = vintage_start
        scored["probability_B0"] = b0_probability
        scored["probability_B1"] = _predict(
            b1, evaluation[B1_FEATURES].to_numpy(dtype=float)
        )
        scored["probability_B2"] = _predict(
            b2, evaluation[B2_FEATURES].to_numpy(dtype=float)
        )
        y_eval = scored["bad10"].to_numpy(dtype=float)
        for model in ("B0", "B1", "B2"):
            probability = np.clip(
                scored[f"probability_{model}"].to_numpy(dtype=float),
                probability_clip,
                1.0 - probability_clip,
            )
            scored[f"row_log_loss_{model}"] = -(
                y_eval * np.log(probability)
                + (1.0 - y_eval) * np.log1p(-probability)
            )
        scored["row_log_loss_improvement_B1_minus_B2"] = (
            scored["row_log_loss_B1"] - scored["row_log_loss_B2"]
        )
        prediction_pieces.append(scored)

    vintage_columns = [
        "model_vintage_year",
        "model_vintage_start",
        "training_max_horizon_end_date",
        "training_row_count",
        "training_positive_origin_count",
        "training_positive_event_count",
        "training_negative_risk_day_count",
        "evaluation_row_count",
        "evaluation_positive_event_count",
        "evaluation_negative_risk_day_count",
        "model_state",
        "b0_probability",
        "b1_intercept",
        *[f"b1_slope_{column}" for column in B1_FEATURES],
        "b1_objective",
        "b1_iterations",
        "b2_intercept",
        *[f"b2_slope_{column}" for column in B2_FEATURES],
        "b2_objective",
        "b2_iterations",
    ]
    vintages = pd.DataFrame(vintage_rows, columns=vintage_columns)
    if not prediction_pieces:
        raise G2CounterfactualError("没有任何可用的历史 prequential 预测")
    predictions = pd.concat(prediction_pieces, ignore_index=True).sort_values(
        ["origin_date", "sample_group_id"], kind="stable", ignore_index=True
    )
    if not bool(
        predictions["horizon_end_date"]
        .ge(predictions["model_vintage_start"])
        .all()
    ):
        raise G2CounterfactualError("预测标签时钟断言失败")
    for event_id, group in predictions.loc[predictions["bad10"].eq(1)].groupby(
        "event_id", sort=True
    ):
        if group["evaluation_vintage_year"].nunique() != 1:
            raise G2CounterfactualError(f"正事件跨预测版本：{event_id}")
    return predictions, vintages


def _calibration_parameters(
    outcome: np.ndarray, probability: np.ndarray, weight: np.ndarray, clip: float
) -> tuple[float | None, float | None]:
    if len(np.unique(outcome)) != 2:
        return None, None
    p = np.clip(probability.astype(float), clip, 1.0 - clip)
    x = np.log(p / (1.0 - p))
    y = outcome.astype(float)
    w = weight.astype(float)
    weight_sum = float(w.sum())

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * x
        fitted = expit(linear)
        nll = np.logaddexp(0.0, linear) - y * linear
        residual = w * (fitted - y) / weight_sum
        return (
            float(np.dot(w, nll) / weight_sum),
            np.asarray([residual.sum(), np.dot(residual, x)], dtype=float),
        )

    fitted = minimize(
        objective_and_gradient,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None, None
    return float(fitted.x[0]), float(fitted.x[1])


def classification_metrics(
    frame: pd.DataFrame, probability_column: str, *, probability_clip: float
) -> dict[str, float | int | None]:
    y = frame["bad10"].to_numpy(dtype=int)
    p = frame[probability_column].to_numpy(dtype=float)
    w = frame["sample_weight"].to_numpy(dtype=float)
    intercept, slope = _calibration_parameters(y, p, w, probability_clip)
    both_classes = len(np.unique(y)) == 2
    return {
        "row_count": int(len(frame)),
        "positive_event_count": int(
            frame.loc[frame["bad10"].eq(1), "event_id"].nunique()
        ),
        "negative_risk_day_count": int(frame["bad10"].eq(0).sum()),
        "event_weighted_log_loss": _weighted_log_loss(y, p, w, probability_clip),
        "event_weighted_brier_score": float(
            np.average(np.square(p - y), weights=w)
        ),
        "event_weighted_pr_auc": (
            float(average_precision_score(y, p, sample_weight=w))
            if both_classes
            else None
        ),
        "event_weighted_roc_auc_description_only": (
            float(roc_auc_score(y, p, sample_weight=w)) if both_classes else None
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def build_era_metrics(
    predictions: pd.DataFrame,
    eras: Sequence[Era],
    *,
    probability_clip: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for era in eras:
        frame = predictions.loc[predictions["era_id"].eq(era.era_id)]
        if frame.empty:
            rows.append(
                {
                    "era_id": era.era_id,
                    "start": era.start,
                    "end": era.end,
                    "prediction_state": "NO_VIEW_NO_PREQUENTIAL_PREDICTIONS",
                    "row_count": 0,
                    "positive_event_count": 0,
                    "negative_risk_day_count": 0,
                    "B1_event_weighted_log_loss": np.nan,
                    "B2_event_weighted_log_loss": np.nan,
                    "log_loss_improvement_B1_minus_B2": np.nan,
                    "same_direction_positive": False,
                }
            )
            continue
        y = frame["bad10"].to_numpy(dtype=int)
        w = frame["sample_weight"].to_numpy(dtype=float)
        b1 = _weighted_log_loss(
            y, frame["probability_B1"].to_numpy(dtype=float), w, probability_clip
        )
        b2 = _weighted_log_loss(
            y, frame["probability_B2"].to_numpy(dtype=float), w, probability_clip
        )
        improvement = b1 - b2
        rows.append(
            {
                "era_id": era.era_id,
                "start": era.start,
                "end": era.end,
                "prediction_state": "VIEW_ALLOWED_PREQUENTIAL",
                "row_count": int(len(frame)),
                "positive_event_count": int(
                    frame.loc[frame["bad10"].eq(1), "event_id"].nunique()
                ),
                "negative_risk_day_count": int(frame["bad10"].eq(0).sum()),
                "B1_event_weighted_log_loss": b1,
                "B2_event_weighted_log_loss": b2,
                "log_loss_improvement_B1_minus_B2": improvement,
                "same_direction_positive": bool(improvement > 0.0),
            }
        )
    return pd.DataFrame(rows)


def event_year_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    if repetitions != 5000 or seed != 20260903:
        raise G2CounterfactualError("Bootstrap 次数和种子必须保持 5000/20260903")
    work = predictions.copy()
    work["weighted_improvement"] = (
        work["sample_weight"]
        * work["row_log_loss_improvement_B1_minus_B2"]
    )
    positive = work.loc[work["bad10"].eq(1)]
    negative = work.loc[work["bad10"].eq(0)]
    event_blocks = positive.groupby("bootstrap_event_block_id", sort=True).agg(
        numerator=("weighted_improvement", "sum"),
        denominator=("sample_weight", "sum"),
    )
    year_blocks = negative.groupby("calendar_year_block_id", sort=True).agg(
        numerator=("weighted_improvement", "sum"),
        denominator=("sample_weight", "sum"),
    )
    if event_blocks.empty or year_blocks.empty:
        raise G2CounterfactualError("Bootstrap 必须同时有事件块和日历年块")
    if not np.allclose(
        event_blocks["denominator"].to_numpy(dtype=float), 1.0, atol=1.0e-12
    ):
        raise G2CounterfactualError("预测样本的事件块权重和必须为 1")
    rng = np.random.default_rng(seed)
    event_indices = rng.integers(
        0, len(event_blocks), size=(repetitions, len(event_blocks))
    )
    year_indices = rng.integers(
        0, len(year_blocks), size=(repetitions, len(year_blocks))
    )
    event_num = event_blocks["numerator"].to_numpy(dtype=float)
    event_den = event_blocks["denominator"].to_numpy(dtype=float)
    year_num = year_blocks["numerator"].to_numpy(dtype=float)
    year_den = year_blocks["denominator"].to_numpy(dtype=float)
    numerator = event_num[event_indices].sum(axis=1) + year_num[year_indices].sum(axis=1)
    denominator = event_den[event_indices].sum(axis=1) + year_den[year_indices].sum(axis=1)
    if (denominator <= 0.0).any():
        raise G2CounterfactualError("Bootstrap 权重分母非法")
    return pd.DataFrame(
        {
            "replicate": np.arange(repetitions, dtype=int),
            "event_and_calendar_year_block_log_loss_improvement_B1_minus_B2": numerator
            / denominator,
        }
    )


def build_g2_counterfactual_artifacts(
    *,
    samples: pd.DataFrame,
    events: pd.DataFrame,
    historical_cutoff: str | pd.Timestamp,
    era_contracts: Sequence[Mapping[str, Any]],
    first_vintage_year: int = 2016,
    last_vintage_year: int = 2026,
    l2_penalty: float = 1.0,
    max_iterations: int = 2000,
    ftol: float = 1.0e-12,
    gtol: float = 1.0e-8,
    probability_clip: float = 1.0e-12,
    bootstrap_repetitions: int = 5000,
    bootstrap_seed: int = 20260903,
    bootstrap_lower_quantile: float = 0.10,
    required_positive_eras: int = 3,
) -> G2CounterfactualArtifacts:
    """执行固定反事实 G2；只绕过 G1 门，不补造数据。"""

    cutoff = pd.Timestamp(historical_cutoff).normalize()
    if pd.isna(cutoff) or cutoff != pd.Timestamp("2026-08-14"):
        raise G2CounterfactualError("历史截止日必须保持 2026-08-14")
    if first_vintage_year != 2016 or last_vintage_year != 2026:
        raise G2CounterfactualError("模型版本年份必须保持 2016—2026")
    if not math.isclose(probability_clip, 1.0e-12, rel_tol=0.0, abs_tol=1.0e-18):
        raise G2CounterfactualError("概率裁剪必须保持 1e-12")
    if not math.isclose(bootstrap_lower_quantile, 0.10, abs_tol=1.0e-12):
        raise G2CounterfactualError("单侧 90% 下界必须使用 10% 分位数")
    if required_positive_eras != 3:
        raise G2CounterfactualError("时代同方向门必须保持 3/4")
    eras = parse_eras(era_contracts)
    common, sample_metadata = _prepare_common_sample(
        samples,
        events,
        cutoff=cutoff,
        eras=eras,
    )
    predictions, vintages = build_prequential_predictions(
        common,
        first_vintage_year=first_vintage_year,
        last_vintage_year=last_vintage_year,
        l2_penalty=l2_penalty,
        max_iterations=max_iterations,
        ftol=ftol,
        gtol=gtol,
        probability_clip=probability_clip,
    )
    era_metrics = build_era_metrics(
        predictions, eras, probability_clip=probability_clip
    )
    bootstrap = event_year_block_bootstrap(
        predictions,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    model_metrics = {
        model: classification_metrics(
            predictions, f"probability_{model}", probability_clip=probability_clip
        )
        for model in ("B0", "B1", "B2")
    }
    overall_improvement = float(
        model_metrics["B1"]["event_weighted_log_loss"]
        - model_metrics["B2"]["event_weighted_log_loss"]
    )
    positive_eras = int(era_metrics["same_direction_positive"].sum())
    bootstrap_values = bootstrap[
        "event_and_calendar_year_block_log_loss_improvement_B1_minus_B2"
    ].to_numpy(dtype=float)
    bootstrap_lower_bound = float(
        np.quantile(bootstrap_values, bootstrap_lower_quantile, method="linear")
    )
    gate_checks = {
        "overall_event_weighted_log_loss_improvement_strictly_positive": bool(
            overall_improvement > 0.0
        ),
        "at_least_three_of_four_eras_same_direction": bool(
            positive_eras >= required_positive_eras
        ),
        "event_and_calendar_year_block_bootstrap_one_sided_90pct_lower_bound_strictly_positive": bool(
            bootstrap_lower_bound > 0.0
        ),
    }
    counterfactual_passed = bool(all(gate_checks.values()))
    era_records: list[dict[str, Any]] = []
    for row in era_metrics.itertuples(index=False):
        era_records.append(
            {
                "era_id": str(row.era_id),
                "prediction_state": str(row.prediction_state),
                "row_count": int(row.row_count),
                "positive_event_count": int(row.positive_event_count),
                "negative_risk_day_count": int(row.negative_risk_day_count),
                "log_loss_improvement_B1_minus_B2": (
                    float(row.log_loss_improvement_B1_minus_B2)
                    if pd.notna(row.log_loss_improvement_B1_minus_B2)
                    else None
                ),
                "same_direction_positive": bool(row.same_direction_positive),
            }
        )
    result: dict[str, Any] = {
        "program_id": "510300_STRESS_TRANSMISSION_HAZARD_V2",
        "stage_id": "G2_STRUCTURAL_INCREMENT_COUNTERFACTUAL",
        "counterfactual_assumption": "ASSUME_G1_MECHANISM_COUNT_GATE_PASS_FOR_G2_ONLY",
        "actual_G1_DATA_AND_EVENTS": "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY",
        "assumed_G1_DATA_AND_EVENTS": "PASS_BY_USER_COUNTERFACTUAL_ASSUMPTION",
        "formal_g2_status": "NOT_ADMISSIBLE_ACTUAL_G1_REMAINS_NO_VIEW",
        "counterfactual_g2_status": (
            "PASS_G2_STRUCTURAL_INCREMENT_COUNTERFACTUAL_ONLY"
            if counterfactual_passed
            else "REJECTED_G2_STRUCTURAL_INCREMENT_COUNTERFACTUAL_ONLY"
        ),
        "counterfactual_g2_passed": counterfactual_passed,
        "gate_checks": gate_checks,
        "sample": {
            **sample_metadata,
            "scored_prediction_row_count": int(len(predictions)),
            "scored_positive_event_count": int(
                predictions.loc[predictions["bad10"].eq(1), "event_id"].nunique()
            ),
            "scored_negative_risk_day_count": int(
                predictions["bad10"].eq(0).sum()
            ),
            "view_allowed_model_vintage_count": int(
                vintages["model_state"].eq(
                    "VIEW_ALLOWED_LABEL_MATURED_PREQUENTIAL"
                ).sum()
            ),
        },
        "model_metrics": model_metrics,
        "event_weighted_log_loss_improvement_B1_minus_B2": overall_improvement,
        "positive_direction_era_count": positive_eras,
        "required_positive_direction_era_count": required_positive_eras,
        "era_results": era_records,
        "bootstrap": {
            "units": ["EVENT", "CALENDAR_YEAR"],
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "one_sided_confidence": 0.90,
            "lower_quantile": bootstrap_lower_quantile,
            "lower_bound": bootstrap_lower_bound,
            "median": float(np.quantile(bootstrap_values, 0.50, method="linear")),
            "upper_90pct_descriptive": float(
                np.quantile(bootstrap_values, 0.90, method="linear")
            ),
        },
        "coefficient_constraints_satisfied": bool(
            (
                vintages.filter(regex=r"^(b1|b2)_slope_")
                .dropna()
                .to_numpy(dtype=float)
                >= -1.0e-12
            ).all()
        ),
        "future_calendar_year_training_used": False,
        "unmatured_label_training_used": False,
        "observation_after_historical_cutoff_read": False,
        "actual_future_path_value_read": False,
        "portfolio_or_performance_artifact_read": False,
        "probability_threshold_selected": False,
        "counterfactual_models_trained": True,
        "formal_model_admitted": False,
        "g3_formally_authorized": False,
        "position_impact": 0,
        "next_allowed_step": (
            "MAY_FREEZE_G3_COUNTERFACTUAL_ONLY_IF_USER_DIRECTS"
            if counterfactual_passed
            else "STOP_COUNTERFACTUAL_V2_NO_FEATURE_OR_MODEL_RESCUE"
        ),
    }
    return G2CounterfactualArtifacts(
        predictions=predictions.reset_index(drop=True),
        model_vintages=vintages.reset_index(drop=True),
        era_metrics=era_metrics.reset_index(drop=True),
        bootstrap_distribution=bootstrap.reset_index(drop=True),
        result=result,
    )
