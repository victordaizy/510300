"""510300 压力传导危险率 V2 的 G1B 前序年代可识别性门。

本模块只使用日期、BAD10 事件身份、B2/B1 共同样本资格和既有切分元数据，
计算季度扩展时钟下理论上可被过去模型预测的独立事件数。它不读取任何特征值，
不拟合模型，也不计算收益或预测表现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype


PROGRAM_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2"
EXECUTION_ID = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_"
    "G1B_PREQUENTIAL_ERA_IDENTIFIABILITY_V1"
)

INPUT_READ_COLUMNS = (
    "origin_date",
    "horizon_end_date",
    "bad10",
    "event_id",
    "b2_vs_b1_eligible",
    "split_assignment",
    "calendar_year_block_id",
)

DERIVED_ALLOWED_FIELDS = (
    "era_id",
    "quarter_end_date",
    "model_vintage_date",
    "event_min_origin_date",
    "whole_event_max_horizon_end_date",
    "training_positive_event_count",
    "training_negative_risk_day_count",
    "prequential_predicted",
)

FORBIDDEN_SOURCE_COLUMNS = (
    "b1_realized_vol20_risk_percentile",
    "b1_drawdown20_risk_percentile",
    "b1_downside_return5_risk_percentile",
    "F",
    "M",
    "T",
    "T_x_F",
    "T_x_M",
    "minimum_path_return",
    "training_weight",
    "b2_vs_b1_model_weight",
    "b3_vs_b2_model_weight",
    "portfolio_return",
    "nav",
    "sharpe",
    "position",
)

UNASSIGNED_SPLIT = "UNASSIGNED_PRE_MODEL_FREEZE"


class G1BIdentifiabilityError(RuntimeError):
    """G1B 输入、时钟、整事件或冻结门契约被违反。"""


@dataclass(frozen=True)
class Era:
    """一个冻结年代的闭区间。"""

    era_id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    g1a_identifiable_event_count: int
    logical_prequential_upper_bound: int


@dataclass(frozen=True)
class G1BArtifacts:
    """G1B 的非模型审计产物。"""

    event_audit: pd.DataFrame
    vintage_audit: pd.DataFrame
    era_audit: pd.DataFrame
    result: dict[str, Any]


def _require_exact_columns(frame: pd.DataFrame) -> None:
    actual = list(frame.columns)
    expected = list(INPUT_READ_COLUMNS)
    if actual != expected:
        raise G1BIdentifiabilityError(
            f"G1B 实际读取列必须精确等于冻结顺序：actual={actual}, expected={expected}"
        )
    forbidden = sorted(set(actual).intersection(FORBIDDEN_SOURCE_COLUMNS))
    if forbidden:
        raise G1BIdentifiabilityError(f"G1B 读入了禁止字段：{forbidden}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    if converted.isna().any():
        raise G1BIdentifiabilityError(f"{label} 含空值或非法日期")
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted.dt.normalize()


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if is_bool_dtype(values.dtype):
        if values.isna().any():
            raise G1BIdentifiabilityError(f"{label} 含空值")
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise G1BIdentifiabilityError(f"{label} 必须且只能为布尔值或 0/1")
    return numeric.eq(1)


def prepare_allowed_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """规范化精确七列输入，并拒绝任何额外列。"""

    _require_exact_columns(frame)
    work = frame.copy()
    work["origin_date"] = _normalize_dates(work["origin_date"], "origin_date")
    work["horizon_end_date"] = _normalize_dates(
        work["horizon_end_date"], "horizon_end_date"
    )
    if work["origin_date"].duplicated().any():
        raise G1BIdentifiabilityError("origin_date 必须逐交易日唯一")
    if not work["origin_date"].lt(work["horizon_end_date"]).all():
        raise G1BIdentifiabilityError("每个标签期限必须严格晚于信息原点")

    labels = pd.to_numeric(work["bad10"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise G1BIdentifiabilityError("bad10 必须且只能为 0/1")
    work["bad10"] = labels.astype("int64")
    work["b2_vs_b1_eligible"] = _strict_bool(
        work["b2_vs_b1_eligible"], "b2_vs_b1_eligible"
    )
    work["event_id"] = work["event_id"].astype("string").fillna("").str.strip()
    work["split_assignment"] = (
        work["split_assignment"].astype("string").fillna("").str.strip()
    )
    work["calendar_year_block_id"] = (
        work["calendar_year_block_id"].astype("string").fillna("").str.strip()
    )
    if not work["split_assignment"].eq(UNASSIGNED_SPLIT).all():
        raise G1BIdentifiabilityError("父样本切分必须全部保持未分配")
    if work["calendar_year_block_id"].eq("").any():
        raise G1BIdentifiabilityError("calendar_year_block_id 不得为空")

    positive = work["bad10"].eq(1)
    if work.loc[positive, "event_id"].eq("").any():
        raise G1BIdentifiabilityError("BAD10 正样本必须绑定事件 ID")
    if work.loc[~positive, "event_id"].ne("").any():
        raise G1BIdentifiabilityError("非事件风险日不得绑定正事件 ID")
    return work.sort_values("origin_date", kind="stable").reset_index(drop=True)


def parse_eras(raw: Sequence[Mapping[str, Any]]) -> list[Era]:
    """解析冻结四时代并确认边界和顺序未变化。"""

    expected = (
        ("ERA_1_2015_2017", "2015-2017", "2015-01-05", "2017-12-29"),
        ("ERA_2_2018_2020", "2018-2020", "2018-01-02", "2020-12-31"),
        ("ERA_3_2021_2023", "2021-2023", "2021-01-04", "2023-12-29"),
        ("ERA_4_2024_CUTOFF", "2024-2026", "2024-01-02", "2026-08-14"),
    )
    if len(raw) != len(expected):
        raise G1BIdentifiabilityError("G1B 必须保留精确四个冻结时代")
    eras: list[Era] = []
    for item, frozen in zip(raw, expected):
        era_id, label, start, end = frozen
        actual_identity = (
            str(item["id"]),
            str(item["label"]),
            str(item["start"]),
            str(item["end"]),
        )
        if actual_identity != frozen:
            raise G1BIdentifiabilityError(
                f"冻结时代发生变化：actual={actual_identity}, expected={frozen}"
            )
        eras.append(
            Era(
                era_id=era_id,
                label=label,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                g1a_identifiable_event_count=int(
                    item["g1a_identifiable_event_count"]
                ),
                logical_prequential_upper_bound=int(
                    item["logical_prequential_upper_bound"]
                ),
            )
        )
    return eras


def _assign_era(date: pd.Timestamp, eras: Sequence[Era]) -> str:
    matches = [era.era_id for era in eras if era.start <= date <= era.end]
    if len(matches) != 1:
        raise G1BIdentifiabilityError(
            f"日期未唯一落入冻结时代：{date.date().isoformat()}"
        )
    return matches[0]


def build_quarterly_vintages(trading_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """按每个季度结束后的首个样本交易日生成模型版本时钟。"""

    normalized = pd.Series(list(trading_dates), dtype="datetime64[ns]").dropna()
    normalized = normalized.dt.normalize().drop_duplicates().sort_values()
    if normalized.empty:
        raise G1BIdentifiabilityError("无法从空交易日序列生成季度时钟")
    first_date = pd.Timestamp(normalized.iloc[0])
    last_date = pd.Timestamp(normalized.iloc[-1])
    period = first_date.to_period("Q") - 1
    rows: list[dict[str, Any]] = []
    seen_vintages: set[pd.Timestamp] = set()
    while True:
        quarter_end = period.end_time.normalize()
        if quarter_end >= last_date:
            break
        candidates = normalized.loc[normalized.gt(quarter_end)]
        if candidates.empty:
            break
        vintage_date = pd.Timestamp(candidates.iloc[0])
        if vintage_date not in seen_vintages:
            rows.append(
                {
                    "quarter_id": f"{quarter_end.year}Q{quarter_end.quarter}",
                    "quarter_end_date": quarter_end,
                    "model_vintage_date": vintage_date,
                }
            )
            seen_vintages.add(vintage_date)
        period += 1
    vintages = pd.DataFrame(rows)
    if vintages.empty or not vintages["quarter_end_date"].lt(
        vintages["model_vintage_date"]
    ).all():
        raise G1BIdentifiabilityError("季度结束后首个交易日时钟非法")
    return vintages


def _positive_event_summary(samples: pd.DataFrame, eras: Sequence[Era]) -> pd.DataFrame:
    positive = samples.loc[samples["bad10"].eq(1)].copy()
    if positive.empty:
        raise G1BIdentifiabilityError("BAD10 正事件集合为空")
    positive["eligible_int"] = positive["b2_vs_b1_eligible"].astype("int64")
    events = (
        positive.groupby("event_id", sort=True)
        .agg(
            event_min_origin_date=("origin_date", "min"),
            whole_event_max_horizon_end_date=("horizon_end_date", "max"),
            total_positive_origin_count=("origin_date", "size"),
            b2_eligible_positive_origin_count=("eligible_int", "sum"),
            calendar_year_block_count=("calendar_year_block_id", "nunique"),
        )
        .reset_index()
    )
    events["b2_identifiable_event"] = events[
        "b2_eligible_positive_origin_count"
    ].gt(0)
    events["era_id"] = [
        _assign_era(pd.Timestamp(value), eras)
        for value in events["event_min_origin_date"]
    ]
    return events


def _validate_g1a_counts(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    eras: Sequence[Era],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = samples["b2_vs_b1_eligible"]
    negative = samples["bad10"].eq(0)
    identifiable = events["b2_identifiable_event"]
    actual = {
        "total_sample_day_count": int(len(samples)),
        "total_positive_origin_count": int((~negative).sum()),
        "total_non_event_risk_day_count": int(negative.sum()),
        "total_independent_event_count": int(len(events)),
        "b2_common_sample_day_count": int(eligible.sum()),
        "b2_eligible_non_event_risk_day_count": int((eligible & negative).sum()),
        "b2_identifiable_event_count": int(identifiable.sum()),
        "event_era_distribution": {
            era.era_id: int(
                (
                    identifiable
                    & events["era_id"].eq(era.era_id)
                ).sum()
            )
            for era in eras
        },
    }
    expected_normalized = {
        "total_sample_day_count": int(expected["total_sample_day_count"]),
        "total_positive_origin_count": int(expected["total_positive_origin_count"]),
        "total_non_event_risk_day_count": int(
            expected["total_non_event_risk_day_count"]
        ),
        "total_independent_event_count": int(expected["total_independent_event_count"]),
        "b2_common_sample_day_count": int(expected["b2_common_sample_day_count"]),
        "b2_eligible_non_event_risk_day_count": int(
            expected["b2_eligible_non_event_risk_day_count"]
        ),
        "b2_identifiable_event_count": int(expected["b2_identifiable_event_count"]),
        "event_era_distribution": {
            str(key): int(value)
            for key, value in expected["event_era_distribution"].items()
        },
    }
    if actual != expected_normalized:
        raise G1BIdentifiabilityError(
            f"G1A 冻结计数漂移：actual={actual}, expected={expected_normalized}"
        )
    for era in eras:
        if actual["event_era_distribution"][era.era_id] != (
            era.g1a_identifiable_event_count
        ):
            raise G1BIdentifiabilityError(f"{era.era_id} 的 G1A 事件数与时代契约冲突")
    return actual


def audit_prequential_identifiability(
    frame: pd.DataFrame,
    *,
    fixed_eras: Sequence[Mapping[str, Any]],
    expected_g1a: Mapping[str, Any],
    minimum_training_positive_events: int,
    minimum_training_negative_risk_days: int,
    minimum_prequential_events_per_era: int,
    minimum_qualified_eras: int,
) -> G1BArtifacts:
    """执行只计数、不拟合模型的季度前序年代可识别性审计。"""

    if minimum_training_positive_events != 1:
        raise G1BIdentifiabilityError("最低训练正事件数必须保持 1")
    if minimum_training_negative_risk_days != 1:
        raise G1BIdentifiabilityError("最低训练负风险日数必须保持 1")
    if minimum_prequential_events_per_era != 5:
        raise G1BIdentifiabilityError("每时代前序事件门必须保持 5")
    if minimum_qualified_eras != 3:
        raise G1BIdentifiabilityError("正式 G2 的合格时代门必须保持 3")

    samples = prepare_allowed_sample(frame)
    eras = parse_eras(fixed_eras)
    all_events = _positive_event_summary(samples, eras)
    g1a = _validate_g1a_counts(samples, all_events, eras, expected_g1a)
    events = all_events.loc[all_events["b2_identifiable_event"]].copy()
    events = events.sort_values(
        ["event_min_origin_date", "event_id"], kind="stable"
    ).reset_index(drop=True)
    negative_eligible = samples.loc[
        samples["bad10"].eq(0) & samples["b2_vs_b1_eligible"]
    ].copy()

    vintages = build_quarterly_vintages(samples["origin_date"])
    vintage_rows: list[dict[str, Any]] = []
    training_event_ids_by_vintage: dict[pd.Timestamp, set[str]] = {}
    for row in vintages.itertuples(index=False):
        vintage_date = pd.Timestamp(row.model_vintage_date)
        matured_events = events.loc[
            events["whole_event_max_horizon_end_date"].lt(vintage_date)
        ]
        matured_negatives = negative_eligible.loc[
            negative_eligible["horizon_end_date"].lt(vintage_date)
        ]
        training_event_ids = set(matured_events["event_id"].astype(str))
        training_event_ids_by_vintage[vintage_date] = training_event_ids
        positive_count = int(len(training_event_ids))
        negative_count = int(len(matured_negatives))
        identifiable = (
            positive_count >= minimum_training_positive_events
            and negative_count >= minimum_training_negative_risk_days
        )
        training_dates = [
            *matured_events["whole_event_max_horizon_end_date"].tolist(),
            *matured_negatives["horizon_end_date"].tolist(),
        ]
        training_max = max(training_dates) if training_dates else pd.NaT
        no_future_training = bool(
            pd.isna(training_max) or pd.Timestamp(training_max) < vintage_date
        )
        if not no_future_training:
            raise G1BIdentifiabilityError("季度训练集包含未成熟或未来标签")
        vintage_rows.append(
            {
                "quarter_id": str(row.quarter_id),
                "quarter_end_date": pd.Timestamp(row.quarter_end_date),
                "model_vintage_date": vintage_date,
                "training_positive_event_count": positive_count,
                "training_negative_risk_day_count": negative_count,
                "training_max_matured_horizon_end_date": training_max,
                "b1_and_b2_training_identifiable": bool(identifiable),
                "no_future_training": no_future_training,
            }
        )
    vintage_audit = pd.DataFrame(vintage_rows)

    event_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        event_start = pd.Timestamp(event.event_min_origin_date)
        available = vintage_audit.loc[
            vintage_audit["model_vintage_date"].le(event_start)
        ]
        if available.empty:
            vintage_date = pd.NaT
            quarter_id = ""
            training_positive = 0
            training_negative = 0
            training_identifiable = False
            no_future_training = True
            event_in_training = False
        else:
            vintage = available.iloc[-1]
            vintage_date = pd.Timestamp(vintage["model_vintage_date"])
            quarter_id = str(vintage["quarter_id"])
            training_positive = int(vintage["training_positive_event_count"])
            training_negative = int(vintage["training_negative_risk_day_count"])
            training_identifiable = bool(
                vintage["b1_and_b2_training_identifiable"]
            )
            no_future_training = bool(vintage["no_future_training"])
            event_in_training = str(event.event_id) in training_event_ids_by_vintage[
                vintage_date
            ]
        if event_in_training:
            raise G1BIdentifiabilityError(
                f"评价事件被用于自身训练：{event.event_id}"
            )
        no_event_split = bool(pd.isna(vintage_date) or vintage_date <= event_start)
        if not no_event_split:
            raise G1BIdentifiabilityError(f"事件预测版本晚于事件起点：{event.event_id}")
        prequential = bool(
            not pd.isna(vintage_date)
            and training_identifiable
            and no_event_split
            and no_future_training
            and not event_in_training
        )
        if pd.isna(vintage_date):
            reason = "NO_VIEW_NO_PRIOR_QUARTERLY_VINTAGE"
        elif not training_identifiable:
            reason = "NO_VIEW_INSUFFICIENT_LABEL_MATURED_TRAINING_CLASSES"
        else:
            reason = "PREQUENTIAL_PREDICTION_IDENTIFIABLE"
        event_rows.append(
            {
                "event_id": str(event.event_id),
                "event_min_origin_date": event_start,
                "whole_event_max_horizon_end_date": pd.Timestamp(
                    event.whole_event_max_horizon_end_date
                ),
                "era_id": str(event.era_id),
                "total_positive_origin_count": int(event.total_positive_origin_count),
                "b2_eligible_positive_origin_count": int(
                    event.b2_eligible_positive_origin_count
                ),
                "calendar_year_block_count": int(event.calendar_year_block_count),
                "evaluation_quarter_id": quarter_id,
                "model_vintage_date": vintage_date,
                "training_positive_event_count": training_positive,
                "training_negative_risk_day_count": training_negative,
                "b1_and_b2_training_identifiable": training_identifiable,
                "event_present_in_own_training": event_in_training,
                "no_event_split": no_event_split,
                "no_future_training": no_future_training,
                "prequential_predicted": prequential,
                "prediction_state": reason,
            }
        )
    event_audit = pd.DataFrame(event_rows)

    evaluation_counts = (
        event_audit.groupby("model_vintage_date", dropna=False).size().to_dict()
    )
    predicted_counts = (
        event_audit.loc[event_audit["prequential_predicted"]]
        .groupby("model_vintage_date")
        .size()
        .to_dict()
    )
    vintage_audit["evaluation_identifiable_event_count"] = [
        int(evaluation_counts.get(value, 0))
        for value in vintage_audit["model_vintage_date"]
    ]
    vintage_audit["prequential_predicted_event_count"] = [
        int(predicted_counts.get(value, 0))
        for value in vintage_audit["model_vintage_date"]
    ]

    era_rows: list[dict[str, Any]] = []
    for era in eras:
        subset = event_audit.loc[event_audit["era_id"].eq(era.era_id)]
        predicted = subset.loc[subset["prequential_predicted"]]
        predicted_count = int(len(predicted))
        training_identifiable = bool(
            predicted_count > 0
            and predicted["b1_and_b2_training_identifiable"].all()
        )
        no_event_split = bool(subset["no_event_split"].all())
        no_future_training = bool(subset["no_future_training"].all())
        if predicted_count > era.logical_prequential_upper_bound:
            raise G1BIdentifiabilityError(
                f"{era.era_id} 超过冻结逻辑上界："
                f"{predicted_count}>{era.logical_prequential_upper_bound}"
            )
        qualified = bool(
            predicted_count >= minimum_prequential_events_per_era
            and training_identifiable
            and no_event_split
            and no_future_training
        )
        era_rows.append(
            {
                "era_id": era.era_id,
                "era_label": era.label,
                "era_start": era.start,
                "era_end": era.end,
                "g1a_identifiable_event_count": int(len(subset)),
                "logical_prequential_upper_bound": (
                    era.logical_prequential_upper_bound
                ),
                "prequential_predicted_independent_event_count": predicted_count,
                "minimum_prequential_events_required": (
                    minimum_prequential_events_per_era
                ),
                "b1_and_b2_training_identifiable": training_identifiable,
                "no_event_split": no_event_split,
                "no_future_training": no_future_training,
                "qualified_era": qualified,
                "era_state": (
                    "QUALIFIED_PREQUENTIAL_ERA"
                    if qualified
                    else "NO_VIEW_INSUFFICIENT_PREQUENTIAL_EVENTS_OR_TRAINING"
                ),
            }
        )
    era_audit = pd.DataFrame(era_rows)
    qualified_era_count = int(era_audit["qualified_era"].sum())
    logical_max_qualified_eras = int(
        sum(
            era.logical_prequential_upper_bound
            >= minimum_prequential_events_per_era
            for era in eras
        )
    )
    if logical_max_qualified_eras != 2:
        raise G1BIdentifiabilityError("冻结逻辑上界必须保持最多两个合格时代")
    if qualified_era_count > logical_max_qualified_eras:
        raise G1BIdentifiabilityError("实际合格时代数超过冻结逻辑上界")

    passed = qualified_era_count >= minimum_qualified_eras
    result: dict[str, Any] = {
        "G1B_PREQUENTIAL_ERA_IDENTIFIABILITY": (
            "PASS_PREQUENTIAL_ERA_IDENTIFIABILITY"
            if passed
            else "NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS"
        ),
        "g1a": g1a,
        "quarterly_model_vintage_count": int(len(vintage_audit)),
        "training_identifiable_vintage_count": int(
            vintage_audit["b1_and_b2_training_identifiable"].sum()
        ),
        "prequential_predicted_independent_event_count": int(
            event_audit["prequential_predicted"].sum()
        ),
        "prequential_events_by_era": {
            str(row.era_id): int(row.prequential_predicted_independent_event_count)
            for row in era_audit.itertuples(index=False)
        },
        "qualified_eras": [
            str(value)
            for value in era_audit.loc[era_audit["qualified_era"], "era_id"]
        ],
        "qualified_era_count": qualified_era_count,
        "minimum_qualified_eras_required": minimum_qualified_eras,
        "logical_max_qualified_eras": logical_max_qualified_eras,
        "formal_g2_allowed": bool(passed),
        "model_trained": False,
        "return_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
    }
    return G1BArtifacts(
        event_audit=event_audit,
        vintage_audit=vintage_audit,
        era_audit=era_audit,
        result=result,
    )

