"""510300未来20日稀有上涨机会预测V1。

本模块有两个严格分离的阶段：

1. 冻结前只构造不含未来收益或标签的因果技术特征；
2. 冻结后才构造标签、拟合预序列模型并运行完整账户回测。

研究仅允许510300多头或现金，不产生Paper、Shadow、订单或实盘动作。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)


PROJECT_ID = "510300_UP20_RARE_EVENT_FORECAST_V1"
CONFIG_PATH = ROOT / "config" / "510300_up20_rare_event_forecast_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_up20_rare_event_forecast_v1_manifest.json"

STATE_UP = "UP20"
STATE_RANGE = "RANGE20"
STATE_DOWN = "DOWN20"
CLASS_ORDER = (STATE_UP, STATE_RANGE, STATE_DOWN)
FEATURE_COLUMNS = (
    "setup_prior",
    "breadth_ignition",
    "price_acceptance",
    "stress_risk",
)


class ContractError(RuntimeError):
    """数据、时钟或冻结合同不满足。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"预测协议配置不存在：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ContractError("预测协议配置不是对象")
    if config.get("protocol", {}).get("project_id") != PROJECT_ID:
        raise ContractError("预测协议项目编号不一致")
    if config["research_question"]["target_label"] != STATE_UP:
        raise ContractError("预测目标必须固定为UP20")
    if config["model"]["features"] != list(FEATURE_COLUMNS):
        raise ContractError("模型特征必须严格固定为四个模块")
    if config["model"]["classes"] != list(CLASS_ORDER):
        raise ContractError("三分类顺序不符合冻结合同")
    if config["model"]["objective_scaling"] != (
        "SUM_NEGATIVE_LOG_LIKELIHOOD_PLUS_HALF_LAMBDA_SLOPE_SQUARED"
    ):
        raise ContractError("模型目标函数缩放口径不符合冻结合同")
    if config["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("持仓范围超出510300和现金")
    for key in (
        "leverage_allowed",
        "short_selling_allowed",
        "derivatives_execution_allowed",
        "paper_or_live_signal_allowed",
    ):
        if config["scope"][key] is not False:
            raise ContractError(f"治理开关必须关闭：scope.{key}")
    for key in (
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_enabled",
        "order_generation",
        "broker_connection",
        "position_change",
        "live_trading_authorized",
    ):
        if config["governance"][key] is not False:
            raise ContractError(f"治理开关必须关闭：governance.{key}")
    if config["inputs"]["pit_membership"]["historical_weights_used"] is not False:
        raise ContractError("禁止使用尚未获准的历史成分权重")
    return config


def _validate_receipt(path_value: str, expected_sha256: str) -> Path:
    path = _project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"冻结输入不存在：{path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ContractError(
            f"冻结输入哈希漂移：{path.relative_to(ROOT).as_posix()}，"
            f"期望{expected_sha256}，实际{actual}"
        )
    return path


def validate_declared_inputs(config: dict[str, Any]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for group_name in ("inputs", "source_results"):
        for key, receipt in config[group_name].items():
            if not isinstance(receipt, dict) or "path" not in receipt or "sha256" not in receipt:
                continue
            resolved[f"{group_name}.{key}"] = _validate_receipt(
                receipt["path"], receipt["sha256"]
            )
    audit = _read_json(resolved["inputs.membership_audit"])
    required_status = config["inputs"]["membership_audit"]["required_status"]
    if audit.get("overall_status") != required_status:
        raise ContractError("官方PIT成员审计状态不符合预测合同")
    if audit.get("weight_admission", {}).get("status") != (
        "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS"
    ):
        raise ContractError("上游历史权重阻断状态发生未解释变化")
    budget = _read_json(resolved["source_results.information_budget"])
    if budget.get("adjudication", {}).get(
        "conservative_up20_forecast_research_advancement_gate_passed"
    ) is not True:
        raise ContractError("信息预算没有授权启动固定UP20预测研究")
    return resolved


def _normalize_date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column]).dt.normalize()
    return result


def _load_market_calendar_and_total_return(
    config: dict[str, Any], resolved: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = config["feature_data_contract"]
    etf = _normalize_date_column(
        pd.read_parquet(resolved["inputs.etf_daily"]), "date"
    ).sort_values("date", ignore_index=True)
    tri_source = _normalize_date_column(
        pd.read_parquet(resolved["inputs.benchmark_total_return"]), "date"
    ).sort_values("date", ignore_index=True)
    if len(etf) != int(contract["expected_etf_execution_calendar_rows"]):
        raise ContractError("510300执行日历行数不符合冻结合同")
    if len(tri_source) != int(contract["expected_total_return_source_rows"]):
        raise ContractError("H00300源表行数不符合冻结合同")
    if etf["date"].duplicated().any() or tri_source["date"].duplicated().any():
        raise ContractError("ETF或全收益指数日期重复")
    if str(etf["date"].iloc[0].date()) != contract["expected_total_return_first_date"]:
        raise ContractError("ETF执行日历起点不符合冻结合同")
    if str(etf["date"].iloc[-1].date()) != contract["expected_total_return_last_date"]:
        raise ContractError("ETF执行日历终点不符合冻结合同")
    calendar = etf[["date"]].copy()
    tri = calendar.merge(tri_source[["date", "close"]], on="date", how="left")
    tri["close"] = pd.to_numeric(tri["close"], errors="coerce")
    if tri["close"].isna().any() or (tri["close"] <= 0).any():
        raise ContractError("ETF可执行日历上的H00300点位不完整")
    if len(tri) != int(contract["expected_feature_rows"]):
        raise ContractError("H00300重索引后的特征日历行数异常")
    evaluation = calendar["date"].between(
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    if int(evaluation.sum()) != int(contract["expected_evaluation_rows"]):
        raise ContractError("评价交易日数量不符合冻结合同")
    return etf, tri


def compute_price_features(tri: pd.DataFrame) -> pd.DataFrame:
    """仅用T日及以前H00300点位计算三个价格型模块。"""
    frame = tri[["date", "close"]].copy().sort_values("date", ignore_index=True)
    close = frame["close"].astype(float)
    one_day_return = close.pct_change(fill_method=None)

    high_252 = close.rolling(252, min_periods=252).max()
    low_5 = close.rolling(5, min_periods=5).min()
    drawdown_component = ((high_252 / close - 1.0) / 0.20).clip(0.0, 1.0)
    recovery_component = ((close / low_5 - 1.0) / 0.05).clip(0.0, 1.0)
    setup_prior = drawdown_component * recovery_component

    net_20 = close / close.shift(20) - 1.0
    path_20 = one_day_return.abs().rolling(20, min_periods=20).sum()
    direction_efficiency = (net_20 / path_20.replace(0.0, np.nan)).clip(0.0, 1.0)
    low_20 = close.rolling(20, min_periods=20).min()
    high_20 = close.rolling(20, min_periods=20).max()
    range_width = (high_20 - low_20).replace(0.0, np.nan)
    range_position = ((close - low_20) / range_width).clip(0.0, 1.0)
    price_acceptance = 0.5 * direction_efficiency + 0.5 * range_position

    downside = one_day_return.clip(upper=0.0)
    downside_volatility = downside.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(242.0)
    downside_stress = (downside_volatility / 0.30).clip(0.0, 1.0)
    five_day_loss = (-(close / close.shift(5) - 1.0) / 0.08).clip(0.0, 1.0)

    price_available = setup_prior.notna() & price_acceptance.notna() & downside_stress.notna()
    result = pd.DataFrame(
        {
            "date": frame["date"],
            "tri_close": close,
            "setup_prior": setup_prior.fillna(0.0),
            "price_acceptance": price_acceptance.fillna(0.0),
            "price_downside_stress": downside_stress.fillna(0.0),
            "price_five_day_loss_stress": five_day_loss.fillna(0.0),
            "price_features_available": price_available,
        }
    )
    return result


@dataclass(frozen=True)
class BreadthBuildResult:
    daily: pd.DataFrame
    diagnostics: dict[str, Any]


def compute_equal_weight_breadth(
    membership: pd.DataFrame,
    constituent: pd.DataFrame,
    config: dict[str, Any],
) -> BreadthBuildResult:
    """用官方PIT成员集合计算等权广度，不读取历史权重。"""
    cutoff = pd.Timestamp(config["dates"]["evaluation_end"])
    members = membership.copy()
    members["date"] = pd.to_datetime(members["membership_date"]).dt.normalize()
    members["symbol"] = members["symbol"].astype(str)
    members = members.loc[members["date"] <= cutoff, ["date", "symbol"]]
    members = members.sort_values(["date", "symbol"], ignore_index=True)
    if members.duplicated(["date", "symbol"]).any():
        raise ContractError("官方PIT成员表存在日期股票重复")
    member_counts = members.groupby("date", sort=True).size()
    expected_members = int(
        config["feature_data_contract"]["expected_members_per_evaluation_day"]
    )
    if not member_counts.eq(expected_members).all():
        raise ContractError("官方PIT成员表并非每个评价日恰好300只")

    panel = constituent.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["symbol"] = panel["con_code"].astype(str)
    panel["member_total_return_close"] = pd.to_numeric(
        panel["total_return_close"], errors="coerce"
    )
    panel = panel.loc[
        panel["date"] <= cutoff,
        ["date", "symbol", "member_total_return_close"],
    ].sort_values(["symbol", "date"], ignore_index=True)
    if panel.duplicated(["date", "symbol"]).any():
        raise ContractError("成分价格面板存在日期股票重复")
    if panel["member_total_return_close"].isna().any() or (
        panel["member_total_return_close"] <= 0
    ).any():
        raise ContractError("成分全收益价格存在空值或非正数")

    expected_rows = int(
        config["feature_data_contract"]["expected_constituent_rows_through_cutoff"]
    )
    if len(panel) != expected_rows:
        raise ContractError("截止日内成分价格行数不符合冻结合同")
    expected_dates = int(
        config["feature_data_contract"]["expected_constituent_dates_through_cutoff"]
    )
    if panel["date"].nunique() != expected_dates:
        raise ContractError("截止日内成分价格日期数不符合冻结合同")

    panel_dates = pd.Index(sorted(panel["date"].unique()))
    ordinal_map = pd.Series(np.arange(len(panel_dates)), index=panel_dates)
    panel["date_ordinal"] = panel["date"].map(ordinal_map).astype(int)
    ordinal_gap = panel.groupby("symbol", sort=False)["date_ordinal"].diff()
    panel["continuous_segment"] = ordinal_gap.ne(1).groupby(panel["symbol"]).cumsum()
    segment_keys = [panel["symbol"], panel["continuous_segment"]]
    panel["member_return_1d"] = panel.groupby(segment_keys, sort=False)[
        "member_total_return_close"
    ].pct_change(fill_method=None)
    panel["member_ma20"] = panel.groupby(segment_keys, sort=False)[
        "member_total_return_close"
    ].transform(lambda values: values.rolling(20, min_periods=20).mean())

    overlap_start = panel["date"].min()
    overlap_members = members.loc[members["date"] >= overlap_start]
    overlap_keys = overlap_members.merge(
        panel[["date", "symbol"]], on=["date", "symbol"], how="outer", indicator=True
    )
    member_only_rows = int(overlap_keys["_merge"].eq("left_only").sum())
    price_only_rows = int(overlap_keys["_merge"].eq("right_only").sum())
    maximum_mismatch = int(
        config["feature_data_contract"]["maximum_membership_price_set_mismatch_rows"]
    )
    if member_only_rows > maximum_mismatch or price_only_rows > maximum_mismatch:
        raise ContractError("PIT成员集合与价格集合的单侧错位行数超过冻结上限")

    joined = members.merge(
        panel[
            [
                "date",
                "symbol",
                "member_total_return_close",
                "member_return_1d",
                "member_ma20",
            ]
        ],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    overlap_joined = joined.loc[joined["date"] >= overlap_start]
    aggregate_matched_coverage = float(
        overlap_joined["member_total_return_close"].notna().mean()
    )
    if aggregate_matched_coverage < float(
        config["feature_data_contract"]["minimum_matched_member_coverage"]
    ):
        raise ContractError("成分价格与官方成员的重叠期总体匹配率不足")

    joined["return_valid"] = joined["member_return_1d"].notna()
    joined["ma20_valid"] = joined["member_ma20"].notna()
    joined["is_advancer"] = joined["member_return_1d"].gt(0.0)
    joined["is_above_ma20"] = joined["member_total_return_close"].gt(
        joined["member_ma20"]
    ) & joined["ma20_valid"]
    joined["is_extreme_down"] = joined["member_return_1d"].le(-0.03)
    daily = joined.groupby("date", sort=True).agg(
        member_count=("symbol", "size"),
        matched_price_count=("member_total_return_close", "count"),
        return_valid_count=("member_return_1d", "count"),
        ma20_valid_count=("member_ma20", "count"),
        advancer_count=("is_advancer", "sum"),
        above_ma20_count=("is_above_ma20", "sum"),
        extreme_down_count=("is_extreme_down", "sum"),
    ).reset_index()
    daily["return_coverage"] = daily["return_valid_count"] / daily["member_count"]
    daily["ma20_coverage"] = daily["ma20_valid_count"] / daily["member_count"]
    daily["advancer_share"] = daily["advancer_count"] / daily[
        "return_valid_count"
    ].replace(0, np.nan)
    daily["above_ma20_share"] = daily["above_ma20_count"] / daily[
        "ma20_valid_count"
    ].replace(0, np.nan)
    daily["extreme_down_share"] = daily["extreme_down_count"] / daily[
        "return_valid_count"
    ].replace(0, np.nan)

    module = config["feature_modules"]["BREADTH_IGNITION"]
    return_min = float(module["minimum_return_coverage"])
    ma_min = float(module["minimum_ma20_coverage"])
    lag = int(module["change_lookback_trading_days"])
    scale = float(module["positive_change_full_scale"])
    daily["breadth_available"] = daily["return_coverage"].ge(return_min) & daily[
        "ma20_coverage"
    ].ge(ma_min)
    lag_available = daily["breadth_available"].shift(lag, fill_value=False)
    usable_change = daily["breadth_available"] & lag_available
    advancer_change = (
        (daily["advancer_share"] - daily["advancer_share"].shift(lag)) / scale
    ).clip(0.0, 1.0)
    ma20_change = (
        (daily["above_ma20_share"] - daily["above_ma20_share"].shift(lag)) / scale
    ).clip(0.0, 1.0)
    daily["breadth_ignition"] = (
        0.5 * advancer_change + 0.5 * ma20_change
    ).where(usable_change, 0.0)
    daily["breadth_ignition"] = daily["breadth_ignition"].fillna(0.0)
    daily["breadth_extreme_down_stress"] = (
        daily["extreme_down_share"] / 0.15
    ).clip(0.0, 1.0).where(daily["return_coverage"].ge(return_min), 0.0)
    daily["breadth_extreme_down_stress"] = daily[
        "breadth_extreme_down_stress"
    ].fillna(0.0)

    diagnostics = {
        "membership_rows": int(len(members)),
        "membership_dates": int(members["date"].nunique()),
        "constituent_rows_through_cutoff": int(len(panel)),
        "constituent_dates_through_cutoff": int(panel["date"].nunique()),
        "constituent_first_date": str(panel["date"].min().date()),
        "constituent_last_date": str(panel["date"].max().date()),
        "overlap_member_only_rows": member_only_rows,
        "overlap_price_only_rows": price_only_rows,
        "overlap_matched_member_coverage": aggregate_matched_coverage,
        "minimum_daily_return_coverage_in_overlap": float(
            daily.loc[daily["date"] >= overlap_start, "return_coverage"].min()
        ),
        "minimum_daily_ma20_coverage_after_first_available": float(
            daily.loc[daily["breadth_available"], "ma20_coverage"].min()
        ),
        "first_breadth_available_date": str(
            daily.loc[daily["breadth_available"], "date"].min().date()
        ),
        "historical_weights_used": False,
    }
    return BreadthBuildResult(daily=daily, diagnostics=diagnostics)


def assemble_outcome_free_features(
    price_features: pd.DataFrame,
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "date",
        "return_coverage",
        "ma20_coverage",
        "advancer_share",
        "above_ma20_share",
        "extreme_down_share",
        "breadth_available",
        "breadth_ignition",
        "breadth_extreme_down_stress",
    ]
    frame = price_features.merge(breadth[columns], on="date", how="left")
    frame["breadth_available"] = frame["breadth_available"].fillna(False).astype(bool)
    for column in (
        "return_coverage",
        "ma20_coverage",
        "advancer_share",
        "above_ma20_share",
        "extreme_down_share",
        "breadth_ignition",
        "breadth_extreme_down_stress",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["stress_risk"] = frame[
        [
            "price_downside_stress",
            "price_five_day_loss_stress",
            "breadth_extreme_down_stress",
        ]
    ].max(axis=1)
    frame["model_features_available"] = frame["price_features_available"].astype(bool)
    ordered = [
        "date",
        "tri_close",
        *FEATURE_COLUMNS,
        "price_features_available",
        "model_features_available",
        "breadth_available",
        "return_coverage",
        "ma20_coverage",
        "advancer_share",
        "above_ma20_share",
        "extreme_down_share",
        "price_downside_stress",
        "price_five_day_loss_stress",
        "breadth_extreme_down_stress",
    ]
    return frame[ordered].sort_values("date", ignore_index=True)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def build_outcome_free_features(write: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config()
    if config["protocol"]["state"] not in (
        "READY_TO_BUILD_OUTCOME_FREE_FEATURES",
        "READY_TO_FREEZE_BEFORE_MODEL_OUTCOME_CALCULATION",
    ):
        raise ContractError("协议状态不允许构造冻结前特征")
    if MANIFEST_PATH.exists() and write:
        raise ContractError("预测协议已经冻结，拒绝覆盖冻结前特征")
    resolved = validate_declared_inputs(config)
    _etf, tri = _load_market_calendar_and_total_return(config, resolved)
    membership = pd.read_parquet(resolved["inputs.pit_membership"])
    constituent = pd.read_parquet(resolved["inputs.constituent_daily"])
    if len(membership) != int(
        config["feature_data_contract"]["expected_membership_rows"]
    ):
        raise ContractError("官方PIT成员行数不符合冻结合同")
    price_features = compute_price_features(tri)
    breadth_result = compute_equal_weight_breadth(membership, constituent, config)
    features = assemble_outcome_free_features(price_features, breadth_result.daily)

    contract = config["feature_data_contract"]
    if len(features) != int(contract["expected_feature_rows"]):
        raise ContractError("特征输出行数不符合冻结合同")
    if features["date"].duplicated().any():
        raise ContractError("特征输出日期重复")
    forbidden_tokens = ("future", "label", "state", "target", "p_up", "p_down")
    forbidden_columns = [
        column
        for column in features.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden_columns:
        raise ContractError(f"冻结前特征含未来或结果字段：{forbidden_columns}")
    values = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ContractError("四个模块必须全部为0到1之间的有限值")
    evaluation = features["date"].between(
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    if int(evaluation.sum()) != int(contract["expected_evaluation_rows"]):
        raise ContractError("特征输出中的评价日数量异常")
    if not features.loc[evaluation, "model_features_available"].all():
        raise ContractError("2015年以来价格型模型特征存在不可用日")

    audit: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "status": "PASS_OUTCOME_FREE_CAUSAL_FEATURES_READY_TO_FREEZE",
        "feature_stage": "BEFORE_MODEL_OUTCOME_CALCULATION",
        "input_receipts": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for key, path in resolved.items()
        },
        "calendar": {
            "source_total_return_rows": int(
                contract["expected_total_return_source_rows"]
            ),
            "execution_calendar_rows": int(len(features)),
            "first_date": str(features["date"].iloc[0].date()),
            "last_date": str(features["date"].iloc[-1].date()),
            "evaluation_rows": int(evaluation.sum()),
        },
        "breadth": breadth_result.diagnostics,
        "feature_contract": {
            "feature_columns": list(FEATURE_COLUMNS),
            "minimum": {
                column: float(features[column].min()) for column in FEATURE_COLUMNS
            },
            "maximum": {
                column: float(features[column].max()) for column in FEATURE_COLUMNS
            },
            "future_columns_present": False,
            "labels_present": False,
            "historical_weights_used": False,
            "all_evaluation_price_features_available": True,
        },
        "model_outcomes_read": False,
        "future_return_created": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "paper_or_shadow_signal_allowed": False,
        "live_trading_authorized": False,
    }
    if write:
        feature_path = _project_path(config["paths"]["features_prefreeze"])
        audit_path = _project_path(config["paths"]["feature_audit"])
        existing = [path for path in (feature_path, audit_path) if path.exists()]
        if existing:
            raise FileExistsError(f"冻结前特征产物已存在，拒绝覆盖：{existing}")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feature_path, index=False)
        audit["feature_artifact"] = _artifact_record(feature_path)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return features, audit


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("UP20预测冻结清单不存在")
    manifest = _read_json(MANIFEST_PATH)
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("UP20预测冻结清单项目编号错误")
    if manifest.get("state") != "FROZEN_BEFORE_MODEL_OUTCOME_CALCULATION":
        raise ContractError("UP20预测冻结清单状态错误")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("UP20预测配置在冻结后发生漂移")
    for relative, expected in manifest.get("tracked_files", {}).items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ContractError(f"冻结跟踪文件漂移：{relative}")
    for relative, expected in manifest.get("input_files", {}).items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ContractError(f"冻结输入文件漂移：{relative}")
    if manifest.get("model_outcomes_computed_before_freeze") is not False:
        raise ContractError("冻结清单没有证明模型结果在冻结前不可见")
    return manifest


def build_forward_labels(features: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """在冻结后构造标签；成熟日是锚点之后第20个ETF交易日收盘。"""
    frame = features[["date", "tri_close"]].copy().sort_values("date", ignore_index=True)
    frame["future_20d_total_return"] = frame["tri_close"].shift(-horizon) / frame[
        "tri_close"
    ] - 1.0
    maturity = frame["date"].shift(-horizon)
    frame["label_maturity_date"] = maturity
    label = np.full(len(frame), None, dtype=object)
    valid = frame["future_20d_total_return"].notna().to_numpy()
    returns = frame["future_20d_total_return"].to_numpy(dtype=float)
    label[valid & (returns >= 0.05)] = STATE_UP
    label[valid & (returns <= -0.05)] = STATE_DOWN
    label[valid & (returns > -0.05) & (returns < 0.05)] = STATE_RANGE
    frame["true_class"] = label
    return frame[[
        "date",
        "future_20d_total_return",
        "label_maturity_date",
        "true_class",
    ]]


def _softmax_range_reference(
    parameters: np.ndarray, features: np.ndarray
) -> np.ndarray:
    up_linear = parameters[0] + features @ parameters[1:5]
    down_linear = parameters[5] + features @ parameters[6:10]
    logits = np.column_stack(
        [up_linear, np.zeros(len(features), dtype=float), down_linear]
    )
    normalizer = logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits - normalizer)


def _objective_and_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    l2_lambda: float,
) -> tuple[float, np.ndarray]:
    probabilities = _softmax_range_reference(parameters, features)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0)
    slopes = np.concatenate([parameters[1:5], parameters[6:10]])
    objective = float(-np.log(chosen).sum() + 0.5 * l2_lambda * np.dot(slopes, slopes))
    up_error = probabilities[:, 0] - (labels == 0).astype(float)
    down_error = probabilities[:, 2] - (labels == 2).astype(float)
    gradient = np.empty(10, dtype=float)
    gradient[0] = up_error.sum()
    gradient[1:5] = features.T @ up_error + l2_lambda * parameters[1:5]
    gradient[5] = down_error.sum()
    gradient[6:10] = features.T @ down_error + l2_lambda * parameters[6:10]
    return objective, gradient


def fit_sign_constrained_multinomial(
    features: np.ndarray,
    labels: np.ndarray,
    l2_lambda: float,
    max_iterations: int,
    ftol: float,
    initial_parameters: np.ndarray | None = None,
) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("符号约束模型必须恰好接收四个特征")
    if len(features) != len(labels) or len(features) == 0:
        raise ValueError("模型特征与标签长度不一致或为空")
    if not np.isfinite(features).all():
        raise ValueError("模型特征存在非有限值")
    if initial_parameters is None:
        counts = np.bincount(labels, minlength=3).astype(float)
        initial_parameters = np.zeros(10, dtype=float)
        initial_parameters[0] = math.log((counts[0] + 0.5) / (counts[1] + 0.5))
        initial_parameters[5] = math.log((counts[2] + 0.5) / (counts[1] + 0.5))
    initial = np.asarray(initial_parameters, dtype=float).copy()
    if initial.shape != (10,):
        raise ValueError("模型初值必须为10维")
    bounds = [
        (None, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (None, 0.0),
        (None, None),
        (None, 0.0),
        (None, 0.0),
        (None, 0.0),
        (0.0, None),
    ]
    for index, (lower, upper) in enumerate(bounds):
        if lower is not None:
            initial[index] = max(initial[index], lower)
        if upper is not None:
            initial[index] = min(initial[index], upper)
    result = minimize(
        fun=lambda values: _objective_and_gradient(
            values, features, labels, float(l2_lambda)
        ),
        x0=initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(max_iterations), "ftol": float(ftol)},
    )
    parameters = np.asarray(result.x, dtype=float)
    constraints_hold = bool(
        np.all(parameters[1:4] >= -1e-10)
        and parameters[4] <= 1e-10
        and np.all(parameters[6:9] <= 1e-10)
        and parameters[9] >= -1e-10
    )
    success = bool(result.success and np.isfinite(parameters).all() and constraints_hold)
    return {
        "success": success,
        "parameters": parameters,
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "constraints_hold": constraints_hold,
    }


def _class_codes(values: Iterable[Any]) -> np.ndarray:
    mapping = {STATE_UP: 0, STATE_RANGE: 1, STATE_DOWN: 2}
    codes = np.array([mapping.get(value, -1) for value in values], dtype=int)
    return codes


def generate_prequential_predictions(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
    progress: bool = True,
) -> pd.DataFrame:
    """逐评价日拟合，只使用成熟日不晚于当日的标签。"""
    combined = features.merge(labels, on="date", how="left", validate="one_to_one")
    combined = combined.sort_values("date", ignore_index=True)
    evaluation_mask = combined["date"].between(
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    evaluation_indices = np.flatnonzero(evaluation_mask.to_numpy())
    if len(evaluation_indices) != int(
        config["feature_data_contract"]["expected_evaluation_rows"]
    ):
        raise ContractError("预序列模型评价日数量异常")

    x_all = combined[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_all = _class_codes(combined["true_class"])
    maturity = pd.to_datetime(combined["label_maturity_date"])
    feature_valid = combined["model_features_available"].to_numpy(dtype=bool)
    model_config = config["model"]
    minimum_n = int(model_config["minimum_mature_training_observations"])
    minimum_up = int(model_config["minimum_mature_up_events"])
    minimum_down = int(model_config["minimum_mature_down_events"])
    l2_lambda = float(model_config["l2_lambda_on_slopes"])
    max_iterations = int(model_config["optimizer_max_iterations"])
    ftol = float(model_config["optimizer_ftol"])

    rows: list[dict[str, Any]] = []
    warm_start: np.ndarray | None = None
    last_progress = 0
    for ordinal, index in enumerate(evaluation_indices, start=1):
        decision_date = pd.Timestamp(combined.at[index, "date"])
        train_mask = (
            feature_valid
            & (y_all >= 0)
            & maturity.notna().to_numpy()
            & (maturity.to_numpy(dtype="datetime64[ns]") <= decision_date.to_datetime64())
        )
        train_indices = np.flatnonzero(train_mask)
        train_y = y_all[train_indices]
        counts = np.bincount(train_y, minlength=3) if len(train_y) else np.zeros(3, dtype=int)
        enough = bool(
            len(train_indices) >= minimum_n
            and counts[0] >= minimum_up
            and counts[2] >= minimum_down
        )
        baseline = (
            counts / counts.sum()
            if counts.sum() > 0
            else np.array([np.nan, np.nan, np.nan], dtype=float)
        )
        model_valid = False
        probabilities = np.array([np.nan, np.nan, np.nan], dtype=float)
        fit_result: dict[str, Any] = {
            "success": False,
            "parameters": np.full(10, np.nan),
            "objective": np.nan,
            "iterations": 0,
            "optimizer_status": -1,
            "optimizer_message": "成熟样本或类别数量不足",
            "constraints_hold": False,
        }
        if enough and feature_valid[index]:
            fit_result = fit_sign_constrained_multinomial(
                x_all[train_indices],
                train_y,
                l2_lambda,
                max_iterations,
                ftol,
                warm_start,
            )
            if fit_result["success"]:
                warm_start = fit_result["parameters"].copy()
                probabilities = _softmax_range_reference(
                    fit_result["parameters"], x_all[index : index + 1]
                )[0]
                model_valid = bool(np.isfinite(probabilities).all())
        maximum_maturity = (
            pd.Timestamp(maturity.iloc[train_indices].max()) if len(train_indices) else pd.NaT
        )
        parameters = np.asarray(fit_result["parameters"], dtype=float)
        row = {
            "date": decision_date,
            "setup_prior": float(combined.at[index, "setup_prior"]),
            "breadth_ignition": float(combined.at[index, "breadth_ignition"]),
            "price_acceptance": float(combined.at[index, "price_acceptance"]),
            "stress_risk": float(combined.at[index, "stress_risk"]),
            "breadth_available": bool(combined.at[index, "breadth_available"]),
            "model_features_available": bool(feature_valid[index]),
            "mature_training_observations": int(len(train_indices)),
            "mature_up_events": int(counts[0]),
            "mature_range_events": int(counts[1]),
            "mature_down_events": int(counts[2]),
            "maximum_training_label_maturity_date": maximum_maturity,
            "model_valid": model_valid,
            "optimizer_success": bool(fit_result["success"]),
            "optimizer_iterations": int(fit_result["iterations"]),
            "optimizer_status": int(fit_result["optimizer_status"]),
            "optimizer_message": str(fit_result["optimizer_message"]),
            "sign_constraints_hold": bool(fit_result["constraints_hold"]),
            "objective": float(fit_result["objective"]),
            "p_up": float(probabilities[0]),
            "p_range": float(probabilities[1]),
            "p_down": float(probabilities[2]),
            "base_p_up": float(baseline[0]),
            "base_p_range": float(baseline[1]),
            "base_p_down": float(baseline[2]),
            "up_intercept": float(parameters[0]),
            "up_setup_coefficient": float(parameters[1]),
            "up_breadth_coefficient": float(parameters[2]),
            "up_acceptance_coefficient": float(parameters[3]),
            "up_stress_coefficient": float(parameters[4]),
            "down_intercept": float(parameters[5]),
            "down_setup_coefficient": float(parameters[6]),
            "down_breadth_coefficient": float(parameters[7]),
            "down_acceptance_coefficient": float(parameters[8]),
            "down_stress_coefficient": float(parameters[9]),
        }
        rows.append(row)
        if progress:
            completed_percent = int(100 * ordinal / len(evaluation_indices))
            if completed_percent >= last_progress + 10:
                last_progress = completed_percent
                print(f"预序列模型进度：{ordinal}/{len(evaluation_indices)}")

    predictions = pd.DataFrame(rows)
    if predictions["date"].duplicated().any():
        raise ContractError("预序列预测日期重复")
    valid = predictions["model_valid"]
    if valid.any():
        probability_sums = predictions.loc[valid, ["p_up", "p_range", "p_down"]].sum(axis=1)
        if not np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
            raise ContractError("模型概率之和不为1")
        if not (
            predictions.loc[valid, "maximum_training_label_maturity_date"]
            <= predictions.loc[valid, "date"]
        ).all():
            raise ContractError("训练集使用了尚未成熟的标签")
    return predictions


def apply_frozen_decision_rule(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    result = predictions.copy()
    means = config["source_results"]["fixed_class_return_means"]
    cost_hurdle = float(config["decision_rule"]["fixed_round_trip_cost_hurdle"])
    result["expected_net_utility"] = (
        result["p_up"] * float(means[STATE_UP])
        + result["p_range"] * float(means[STATE_RANGE])
        + result["p_down"] * float(means[STATE_DOWN])
        - cost_hurdle
    )
    result["up_probability_gate"] = result["p_up"].ge(
        float(config["decision_rule"]["minimum_up_probability"])
    )
    result["down_probability_veto_pass"] = result["p_down"].le(
        float(config["decision_rule"]["maximum_down_probability"])
    )
    result["expected_utility_gate"] = result["expected_net_utility"].ge(
        float(config["decision_rule"]["expected_utility_minimum"])
    )
    result["stress_hard_veto_pass"] = result["stress_risk"].le(
        float(config["feature_modules"]["STRESS_VETO"]["hard_veto_threshold"])
    )
    common = result["model_valid"] & result["up_probability_gate"] & result[
        "expected_utility_gate"
    ]
    result["primary_long_signal"] = (
        common
        & result["down_probability_veto_pass"]
        & result["stress_hard_veto_pass"]
    )
    result["control_without_stress_veto_long_signal"] = (
        common & result["down_probability_veto_pass"]
    )
    result["control_without_down_probability_veto_long_signal"] = (
        common & result["stress_hard_veto_pass"]
    )
    return result


def _load_account_inputs(
    config: dict[str, Any], resolved: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    etf = _normalize_date_column(
        pd.read_parquet(resolved["inputs.etf_daily"]), "date"
    ).sort_values("date", ignore_index=True)
    dividends = pd.read_csv(resolved["inputs.dividends"])
    for column in ("record_date", "ex_date", "payment_date"):
        if column in dividends:
            dividends[column] = pd.to_datetime(dividends[column]).dt.normalize()
    return etf, dividends


def _costs(config: dict[str, Any], cost_name: str) -> BacktestCosts:
    values = config["evaluation"][cost_name]
    return BacktestCosts(
        commission_rate=float(values["commission_rate_per_leg"]),
        minimum_commission_cny=float(values["minimum_commission_cny_per_leg"]),
        stamp_duty_rate=float(values["stamp_duty_rate"]),
        slippage_bps=float(values["slippage_bps_per_leg"]),
        lot_size=int(config["evaluation"]["lot_size_shares"]),
        cash_annual_rate=float(config["evaluation"]["cash_annual_rate"]),
    )


def _summary(
    ledger: pd.DataFrame, trades: pd.DataFrame, initial_capital: float
) -> dict[str, Any]:
    result = dict(summarize_backtest(ledger, trades, initial_capital))
    result["total_slippage_cost_cny"] = float(ledger["daily_slippage_cost_cny"].sum())
    result["total_execution_cost_cny"] = float(
        ledger["daily_total_execution_cost_cny"].sum()
    )
    return result


def _period_metrics(
    ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    frame = ledger.loc[ledger["date"].between(start, end)].copy()
    returns = frame["daily_return"].to_numpy(dtype=float)
    if len(returns) == 0:
        return {
            "observations": 0,
            "total_return": None,
            "annualized_mean_return": None,
            "annualized_volatility": None,
            "net_sharpe": None,
            "maximum_drawdown": None,
        }
    volatility = float(np.std(returns, ddof=1) * math.sqrt(242.0))
    annualized_mean = float(np.mean(returns) * 242.0)
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "observations": int(len(returns)),
        "total_return": float(np.prod(1.0 + returns) - 1.0),
        "annualized_mean_return": annualized_mean,
        "annualized_volatility": volatility,
        "net_sharpe": annualized_mean / volatility if volatility > 0 else None,
        "maximum_drawdown": float(drawdown.min()),
    }


def build_phase_blocks(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    evaluation_dates: pd.Series,
    offset: int,
    horizon: int,
    signal_column: str,
) -> pd.DataFrame:
    if offset < 0 or offset >= horizon:
        raise ValueError("相位偏移必须位于0到持有期减1")
    prediction_map = predictions.set_index("date")
    label_map = labels.set_index("date")
    dates = pd.Series(pd.to_datetime(evaluation_dates).to_numpy()).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for block_id, anchor_position in enumerate(range(offset, len(dates), horizon)):
        signal_date = pd.Timestamp(dates.iloc[anchor_position])
        complete = anchor_position + horizon < len(dates)
        end_position = min(anchor_position + horizon, len(dates) - 1)
        end_date = pd.Timestamp(dates.iloc[end_position])
        prediction = prediction_map.loc[signal_date]
        label = label_map.loc[signal_date]
        predicted_long = bool(prediction[signal_column]) if complete else False
        rows.append(
            {
                "phase_offset": int(offset),
                "block_id": int(block_id),
                "anchor_position": int(anchor_position),
                "signal_date": signal_date,
                "execution_start": (
                    pd.Timestamp(dates.iloc[anchor_position + 1])
                    if anchor_position + 1 < len(dates)
                    else pd.NaT
                ),
                "execution_end": end_date,
                "complete": bool(complete),
                "future_20d_total_return": (
                    float(label["future_20d_total_return"]) if complete else np.nan
                ),
                "true_class": str(label["true_class"]) if complete else "CENSORED",
                "predicted_long": predicted_long,
                "target_position": float(predicted_long),
                "model_valid": bool(prediction["model_valid"]),
                "p_up": float(prediction["p_up"]),
                "p_range": float(prediction["p_range"]),
                "p_down": float(prediction["p_down"]),
                "base_p_up": float(prediction["base_p_up"]),
                "base_p_range": float(prediction["base_p_range"]),
                "base_p_down": float(prediction["base_p_down"]),
                "expected_net_utility": float(prediction["expected_net_utility"]),
                "stress_risk": float(prediction["stress_risk"]),
                "up_probability_gate": bool(prediction["up_probability_gate"]),
                "down_probability_veto_pass": bool(
                    prediction["down_probability_veto_pass"]
                ),
                "expected_utility_gate": bool(prediction["expected_utility_gate"]),
                "stress_hard_veto_pass": bool(prediction["stress_hard_veto_pass"]),
            }
        )
    return pd.DataFrame(rows)


def _classification_metrics(blocks: pd.DataFrame) -> dict[str, Any]:
    complete = blocks.loc[blocks["complete"]].copy()
    predicted = complete["predicted_long"]
    true_up = complete["true_class"].eq(STATE_UP)
    true_range = complete["true_class"].eq(STATE_RANGE)
    true_down = complete["true_class"].eq(STATE_DOWN)
    captured_up = int((predicted & true_up).sum())
    false_range = int((predicted & true_range).sum())
    false_down = int((predicted & true_down).sum())
    predicted_count = int(predicted.sum())
    up_count = int(true_up.sum())
    range_count = int(true_range.sum())
    down_count = int(true_down.sum())
    return {
        "complete_blocks": int(len(complete)),
        "censored_blocks": int((~blocks["complete"]).sum()),
        "true_up_blocks": up_count,
        "true_range_blocks": range_count,
        "true_down_blocks": down_count,
        "predicted_long_blocks": predicted_count,
        "captured_up_blocks": captured_up,
        "false_range_blocks": false_range,
        "false_down_blocks": false_down,
        "up_precision": captured_up / predicted_count if predicted_count else None,
        "up_recall": captured_up / up_count if up_count else None,
        "range_false_long_rate": false_range / range_count if range_count else None,
        "down_false_long_rate": false_down / down_count if down_count else None,
    }


def _probability_skill(blocks: pd.DataFrame) -> dict[str, Any]:
    evaluation = blocks.loc[blocks["complete"] & blocks["model_valid"]].copy()
    if evaluation.empty:
        return {
            "observations": 0,
            "model_brier": None,
            "causal_base_brier": None,
            "brier_skill": None,
            "model_log_loss": None,
            "causal_base_log_loss": None,
            "log_loss_skill": None,
        }
    codes = _class_codes(evaluation["true_class"])
    one_hot = np.eye(3)[codes]
    model = evaluation[["p_up", "p_range", "p_down"]].to_numpy(dtype=float)
    baseline = evaluation[
        ["base_p_up", "base_p_range", "base_p_down"]
    ].to_numpy(dtype=float)
    model_brier = float(np.mean(np.sum((model - one_hot) ** 2, axis=1)))
    base_brier = float(np.mean(np.sum((baseline - one_hot) ** 2, axis=1)))
    model_log_loss = float(
        -np.mean(np.log(np.clip(model[np.arange(len(codes)), codes], 1e-15, 1.0)))
    )
    base_log_loss = float(
        -np.mean(np.log(np.clip(baseline[np.arange(len(codes)), codes], 1e-15, 1.0)))
    )
    return {
        "observations": int(len(evaluation)),
        "model_brier": model_brier,
        "causal_base_brier": base_brier,
        "brier_skill": 1.0 - model_brier / base_brier if base_brier > 0 else None,
        "model_log_loss": model_log_loss,
        "causal_base_log_loss": base_log_loss,
        "log_loss_skill": (
            1.0 - model_log_loss / base_log_loss if base_log_loss > 0 else None
        ),
        "brier_improvement": base_brier - model_brier,
        "log_loss_improvement": base_log_loss - model_log_loss,
    }


def evaluate_phase(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    evaluation_dates: pd.Series,
    etf: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    offset: int,
    signal_column: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    horizon = int(config["evaluation"]["horizon_trading_days"])
    blocks = build_phase_blocks(
        predictions,
        labels,
        evaluation_dates,
        offset,
        horizon,
        signal_column,
    )
    targets = blocks[["signal_date", "target_position"]].rename(
        columns={"signal_date": "date"}
    )
    targets["trade_allowed"] = True
    targets["risk_off_override"] = targets["target_position"].eq(0.0)
    targets["signal_reason"] = np.where(
        targets["target_position"].eq(1.0),
        f"冻结UP20预测::{signal_column}::满仓",
        f"冻结UP20预测::{signal_column}::现金",
    )
    initial = float(config["evaluation"]["initial_capital_cny"])
    start = config["dates"]["evaluation_start"]
    end = config["dates"]["evaluation_end"]
    base_ledger, base_trades = run_long_cash_backtest(
        etf,
        dividends,
        targets,
        initial,
        _costs(config, "base_costs"),
        start,
        end,
    )
    stress_ledger, stress_trades = run_long_cash_backtest(
        etf,
        dividends,
        targets,
        initial,
        _costs(config, "stress_costs"),
        start,
        end,
    )
    recent_start, recent_end = (
        pd.Timestamp(value) for value in config["dates"]["recent_period"]
    )
    report = {
        "phase_offset": int(offset),
        "signal_column": signal_column,
        "classification": _classification_metrics(blocks),
        "probability_skill": _probability_skill(blocks),
        "base": _summary(base_ledger, base_trades, initial),
        "stress": _summary(stress_ledger, stress_trades, initial),
        "recent_period": {
            "start": str(recent_start.date()),
            "end": str(recent_end.date()),
            "base": _period_metrics(base_ledger, recent_start, recent_end),
            "stress": _period_metrics(stress_ledger, recent_start, recent_end),
        },
    }
    frames = {
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "stress_ledger": stress_ledger,
        "stress_trades": stress_trades,
        "targets": targets,
    }
    return report, blocks, frames


def _passes_minimum(value: Any, threshold: float) -> bool:
    return bool(value is not None and np.isfinite(float(value)) and float(value) >= threshold)


def _flatten_phase_report(report: dict[str, Any]) -> dict[str, Any]:
    classification = report["classification"]
    return {
        "phase_offset": report["phase_offset"],
        "complete_blocks": classification["complete_blocks"],
        "predicted_long_blocks": classification["predicted_long_blocks"],
        "captured_up_blocks": classification["captured_up_blocks"],
        "false_range_blocks": classification["false_range_blocks"],
        "false_down_blocks": classification["false_down_blocks"],
        "up_precision": classification["up_precision"],
        "up_recall": classification["up_recall"],
        "base_net_sharpe": report["base"]["sharpe_zero_cash_rate"],
        "stress_net_sharpe": report["stress"]["sharpe_zero_cash_rate"],
        "stress_total_return": report["stress"]["total_return"],
        "stress_max_drawdown": report["stress"]["max_drawdown"],
        "recent_stress_net_sharpe": report["recent_period"]["stress"]["net_sharpe"],
        "brier_skill": report["probability_skill"]["brier_skill"],
        "log_loss_skill": report["probability_skill"]["log_loss_skill"],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}%}"


def render_markdown(report: dict[str, Any]) -> str:
    primary = report["primary_result"]
    classification = primary["classification"]
    skill = primary["probability_skill"]
    lines = [
        "# 510300 未来20日稀有上涨机会预测 V1 结果",
        "",
        f"- 正式状态：`{report['status']}`",
        "- 资产边界：仅510300或现金；无杠杆、无做空、无衍生品。",
        "- 证据边界：历史因果预序列检验；独立前向样本仍为0。",
        "- 规则边界：四模块、符号约束模型、概率阈值、相位和成本均在结果前冻结。",
        "",
        "## 主相位账户结果",
        "",
        "| 口径 | 净夏普 | 总收益 | 最大回撤 | 交易次数 |",
        "|---|---:|---:|---:|---:|",
        f"| 基准成本 | {_fmt(primary['base']['sharpe_zero_cash_rate'])} | "
        f"{_pct(primary['base']['total_return'])} | {_pct(primary['base']['max_drawdown'])} | "
        f"{primary['base']['trade_count']} |",
        f"| 压力成本 | {_fmt(primary['stress']['sharpe_zero_cash_rate'])} | "
        f"{_pct(primary['stress']['total_return'])} | {_pct(primary['stress']['max_drawdown'])} | "
        f"{primary['stress']['trade_count']} |",
        f"| 2021年以来压力 | {_fmt(primary['recent_period']['stress']['net_sharpe'])} | "
        f"{_pct(primary['recent_period']['stress']['total_return'])} | "
        f"{_pct(primary['recent_period']['stress']['maximum_drawdown'])} | NA |",
        "",
        "## 稀有事件信息质量",
        "",
        "| 指标 | 结果 | 冻结预算/门 |",
        "|---|---:|---:|",
        f"| 捕获UP20 | {classification['captured_up_blocks']} | ≥14 |",
        f"| 误入DOWN20 | {classification['false_down_blocks']} | ≤1 |",
        f"| 误入RANGE20 | {classification['false_range_blocks']} | ≤10 |",
        f"| 预测满仓区间 | {classification['predicted_long_blocks']} | ≥14 |",
        f"| UP20精确率 | {_pct(classification['up_precision'])} | 诊断项 |",
        f"| UP20召回率 | {_pct(classification['up_recall'])} | 信息预算参考 |",
        f"| Brier Skill | {_fmt(skill['brier_skill'])} | >0 |",
        f"| Log Loss Skill | {_fmt(skill['log_loss_skill'])} | >0 |",
        "",
        "## 20种相位压力净夏普",
        "",
        "| offset | 压力净夏普 | 捕获UP | 误入DOWN | 误入RANGE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for phase in report["phase_results"]:
        metrics = phase["classification"]
        lines.append(
            f"| {phase['phase_offset']} | {_fmt(phase['stress']['sharpe_zero_cash_rate'])} | "
            f"{metrics['captured_up_blocks']} | {metrics['false_down_blocks']} | "
            f"{metrics['false_range_blocks']} |"
        )
    lines.extend(
        [
            "",
            "## 冻结门判定",
            "",
        ]
    )
    for name, passed in report["adjudication"]["gates"].items():
        lines.append(f"- `{name}`：`{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            report["adjudication"]["interpretation"],
            "",
            "控制项仅用于解释否决器贡献，不能替代主规则。无论历史是否通过，Paper、Shadow、持仓映射、订单、券商连接与实盘授权均保持关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(write: bool = True, progress: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    resolved = validate_declared_inputs(config)
    feature_path = _project_path(config["paths"]["features_prefreeze"])
    feature_audit_path = _project_path(config["paths"]["feature_audit"])
    if not feature_path.exists() or not feature_audit_path.exists():
        raise FileNotFoundError("冻结前特征或审计不存在")
    frozen_inputs = manifest.get("input_files", {})
    for path in (feature_path, feature_audit_path):
        relative = path.relative_to(ROOT).as_posix()
        if frozen_inputs.get(relative) != sha256_file(path):
            raise ContractError(f"冻结前特征产物与清单不一致：{relative}")
    feature_audit = _read_json(feature_audit_path)
    if feature_audit.get("status") != "PASS_OUTCOME_FREE_CAUSAL_FEATURES_READY_TO_FREEZE":
        raise ContractError("冻结前特征审计没有通过")
    if feature_audit.get("model_outcomes_read") is not False:
        raise ContractError("冻结前特征阶段已经读取模型结果")

    formal_output_keys = (
        "daily_predictions",
        "primary_blocks",
        "phase_results",
        "primary_base_ledger",
        "primary_base_trades",
        "primary_stress_ledger",
        "primary_stress_trades",
        "result_json",
        "result_markdown",
    )
    output_paths = {
        key: _project_path(config["paths"][key]) for key in formal_output_keys
    }
    if write:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"正式预测产物已存在，拒绝覆盖：{existing}")

    features = _normalize_date_column(pd.read_parquet(feature_path), "date")
    if tuple(column for column in FEATURE_COLUMNS if column not in features.columns):
        raise ContractError("冻结特征缺少模型列")
    labels = build_forward_labels(
        features, int(config["evaluation"]["horizon_trading_days"])
    )
    predictions = generate_prequential_predictions(features, labels, config, progress)
    predictions = apply_frozen_decision_rule(predictions, config)
    prediction_output = predictions.merge(
        labels, on="date", how="left", validate="one_to_one"
    )

    etf, dividends = _load_account_inputs(config, resolved)
    evaluation_dates = etf.loc[
        etf["date"].between(
            pd.Timestamp(config["dates"]["evaluation_start"]),
            pd.Timestamp(config["dates"]["evaluation_end"]),
        ),
        "date",
    ].reset_index(drop=True)
    phase_reports: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    primary_blocks: pd.DataFrame | None = None
    primary_frames: dict[str, pd.DataFrame] | None = None
    for offset in config["evaluation"]["robustness_phase_offsets"]:
        phase_report, blocks, frames = evaluate_phase(
            predictions,
            labels,
            evaluation_dates,
            etf,
            dividends,
            config,
            int(offset),
            "primary_long_signal",
        )
        phase_reports.append(phase_report)
        phase_rows.append(_flatten_phase_report(phase_report))
        if int(offset) == int(config["evaluation"]["primary_phase_offset"]):
            primary_blocks = blocks
            primary_frames = frames
    if primary_blocks is None or primary_frames is None:
        raise ContractError("主相位没有生成账户结果")
    primary_report = phase_reports[int(config["evaluation"]["primary_phase_offset"])]

    control_reports: dict[str, Any] = {}
    for control_name, signal_column in (
        (
            "WITHOUT_HARD_STRESS_VETO",
            "control_without_stress_veto_long_signal",
        ),
        (
            "WITHOUT_DOWN_PROBABILITY_VETO",
            "control_without_down_probability_veto_long_signal",
        ),
    ):
        control_report, _control_blocks, _control_frames = evaluate_phase(
            predictions,
            labels,
            evaluation_dates,
            etf,
            dividends,
            config,
            int(config["evaluation"]["primary_phase_offset"]),
            signal_column,
        )
        control_reports[control_name] = control_report

    gates_config = config["historical_acceptance_gates"]
    classification = primary_report["classification"]
    probability_skill = primary_report["probability_skill"]
    gates = {
        "primary_base_net_sharpe": _passes_minimum(
            primary_report["base"]["sharpe_zero_cash_rate"],
            float(gates_config["primary_base_net_sharpe_minimum"]),
        ),
        "primary_stress_net_sharpe": _passes_minimum(
            primary_report["stress"]["sharpe_zero_cash_rate"],
            float(gates_config["primary_stress_net_sharpe_minimum"]),
        ),
        "primary_recent_stress_net_sharpe": _passes_minimum(
            primary_report["recent_period"]["stress"]["net_sharpe"],
            float(gates_config["primary_recent_stress_net_sharpe_minimum"]),
        ),
        "primary_stress_total_return_positive": bool(
            float(primary_report["stress"]["total_return"]) > 0.0
        ),
        "minimum_captured_up_blocks": bool(
            classification["captured_up_blocks"]
            >= int(gates_config["minimum_captured_up_blocks"])
        ),
        "maximum_false_down_blocks": bool(
            classification["false_down_blocks"]
            <= int(gates_config["maximum_false_down_blocks"])
        ),
        "maximum_false_range_blocks": bool(
            classification["false_range_blocks"]
            <= int(gates_config["maximum_false_range_blocks"])
        ),
        "every_phase_stress_net_sharpe": all(
            _passes_minimum(
                phase["stress"]["sharpe_zero_cash_rate"],
                float(gates_config["every_phase_stress_net_sharpe_minimum"]),
            )
            for phase in phase_reports
        ),
        "primary_brier_skill_vs_causal_base_positive": bool(
            probability_skill["brier_skill"] is not None
            and float(probability_skill["brier_skill"]) > 0.0
        ),
        "primary_log_loss_skill_vs_causal_base_positive": bool(
            probability_skill["log_loss_skill"] is not None
            and float(probability_skill["log_loss_skill"]) > 0.0
        ),
        "minimum_primary_predicted_long_blocks": bool(
            classification["predicted_long_blocks"]
            >= int(gates_config["minimum_primary_predicted_long_blocks"])
        ),
    }
    historical_pass = bool(all(gates.values()))
    status = (
        "PASS_FROZEN_UP20_RARE_EVENT_FORECAST_V1_HISTORICAL_ONLY_FORWARD_REQUIRED_NOT_TRADABLE"
        if historical_pass
        else "REJECTED_FROZEN_UP20_RARE_EVENT_FORECAST_V1_TARGET_OR_INFORMATION_BUDGET_GATE_FAILED_NO_RESCUE"
    )
    interpretation = (
        "固定四模块UP20预测器在完整历史账户、压力成本、近期窗口、信息预算和20种相位门下全部通过；这只建立历史候选，必须等待独立前向证据，尚不能交易。"
        if historical_pass
        else "固定四模块UP20预测器没有同时达到历史夏普1.2、稀有事件信息预算与相位稳健性门；V1按预注册规则停止，不以控制项或调参救援。"
    )

    finite_phase_sharpes = [
        float(phase["stress"]["sharpe_zero_cash_rate"])
        for phase in phase_reports
        if phase["stress"]["sharpe_zero_cash_rate"] is not None
        and np.isfinite(float(phase["stress"]["sharpe_zero_cash_rate"]))
    ]
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "research_question": config["research_question"],
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "model_outcomes_computed_before_freeze": False,
            "source_oracle_and_information_budget_known": True,
        },
        "data_and_clock_boundary": {
            "execution_calendar": "510300.SH",
            "label_asset": "H00300_TOTAL_RETURN_INDEX",
            "signal_time": "T_CLOSE",
            "execution_time": "T_PLUS_1_OPEN",
            "label_maturity_clock": config["model"]["label_maturity_clock"],
            "overlapping_training_labels_allowed": True,
            "independent_evaluation_unit": "NON_OVERLAPPING_20D_BLOCK",
            "historical_weights_used": False,
            "feature_audit_path": feature_audit_path.relative_to(ROOT).as_posix(),
            "feature_audit_sha256": sha256_file(feature_audit_path),
        },
        "model_diagnostics": {
            "evaluation_predictions": int(len(predictions)),
            "valid_predictions": int(predictions["model_valid"].sum()),
            "invalid_predictions": int((~predictions["model_valid"]).sum()),
            "optimizer_failures": int((~predictions["optimizer_success"]).sum()),
            "sign_constraint_failures": int(
                (predictions["model_valid"] & ~predictions["sign_constraints_hold"]).sum()
            ),
            "first_valid_prediction_date": (
                str(predictions.loc[predictions["model_valid"], "date"].min().date())
                if predictions["model_valid"].any()
                else None
            ),
            "last_valid_prediction_date": (
                str(predictions.loc[predictions["model_valid"], "date"].max().date())
                if predictions["model_valid"].any()
                else None
            ),
            "fixed_features": list(FEATURE_COLUMNS),
            "fixed_l2_lambda": float(config["model"]["l2_lambda_on_slopes"]),
        },
        "primary_result": primary_report,
        "phase_results": phase_reports,
        "phase_robustness_summary": {
            "minimum_stress_net_sharpe": (
                min(finite_phase_sharpes) if finite_phase_sharpes else None
            ),
            "median_stress_net_sharpe": (
                float(np.median(finite_phase_sharpes)) if finite_phase_sharpes else None
            ),
            "maximum_stress_net_sharpe": (
                max(finite_phase_sharpes) if finite_phase_sharpes else None
            ),
            "phase_count": int(len(phase_reports)),
            "all_phases_at_least_1_2": gates["every_phase_stress_net_sharpe"],
        },
        "controls_not_eligible_for_selection": control_reports,
        "information_budget_comparison": {
            "required_captured_up_blocks": int(
                config["source_results"]["fixed_information_budget"][
                    "minimum_captured_bull_blocks"
                ]
            ),
            "maximum_false_down_blocks": int(
                config["source_results"]["fixed_information_budget"][
                    "maximum_false_bear_blocks"
                ]
            ),
            "maximum_false_range_blocks": int(
                config["source_results"]["fixed_information_budget"][
                    "maximum_false_range_blocks"
                ]
            ),
            "observed": classification,
        },
        "adjudication": {
            "historical_acceptance_passed": historical_pass,
            "gates": gates,
            "interpretation": interpretation,
            "realistic_up20_forecast": (
                "PASS_HISTORICAL_ONLY_FORWARD_REQUIRED"
                if historical_pass
                else "REJECTED_FROZEN_V1_NO_RESCUE"
            ),
            "historical_account_simulation_target_achieved": historical_pass,
            "verified_forward_observations": 0,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "live_trading_authorized": False,
        },
        "selection_bias_control": manifest["selection_bias_control"],
        "governance": {
            "historical_research_only": True,
            "forward_validation_started": False,
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
        for path in output_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        prediction_output.to_parquet(output_paths["daily_predictions"], index=False)
        primary_blocks.to_parquet(output_paths["primary_blocks"], index=False)
        pd.DataFrame(phase_rows).to_parquet(output_paths["phase_results"], index=False)
        primary_frames["base_ledger"].to_parquet(
            output_paths["primary_base_ledger"], index=False
        )
        primary_frames["base_trades"].to_parquet(
            output_paths["primary_base_trades"], index=False
        )
        primary_frames["stress_ledger"].to_parquet(
            output_paths["primary_stress_ledger"], index=False
        )
        primary_frames["stress_trades"].to_parquet(
            output_paths["primary_stress_trades"], index=False
        )
        artifact_keys = (
            "daily_predictions",
            "primary_blocks",
            "phase_results",
            "primary_base_ledger",
            "primary_base_trades",
            "primary_stress_ledger",
            "primary_stress_trades",
        )
        report["artifacts"] = {
            key: _artifact_record(output_paths[key]) for key in artifact_keys
        }
        output_paths["result_json"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_paths["result_markdown"].write_text(
            render_markdown(report), encoding="utf-8"
        )
    return report


__all__ = [
    "CLASS_ORDER",
    "CONFIG_PATH",
    "ContractError",
    "FEATURE_COLUMNS",
    "MANIFEST_PATH",
    "PROJECT_ID",
    "STATE_DOWN",
    "STATE_RANGE",
    "STATE_UP",
    "apply_frozen_decision_rule",
    "assemble_outcome_free_features",
    "build_forward_labels",
    "build_outcome_free_features",
    "build_phase_blocks",
    "compute_equal_weight_breadth",
    "compute_price_features",
    "fit_sign_constrained_multinomial",
    "generate_prequential_predictions",
    "load_config",
    "run_study",
    "sha256_file",
    "validate_manifest",
]
