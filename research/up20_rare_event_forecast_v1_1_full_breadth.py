"""510300 UP20预测V1.1：只扩展官方PIT等权广度的历史覆盖。"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from up20_rare_event_forecast_v1 import (  # noqa: E402
    ContractError,
    FEATURE_COLUMNS,
    _artifact_record,
    _flatten_phase_report,
    _load_account_inputs,
    _load_market_calendar_and_total_return,
    _normalize_date_column,
    _passes_minimum,
    apply_frozen_decision_rule,
    assemble_outcome_free_features,
    build_forward_labels,
    compute_price_features,
    evaluate_phase,
    generate_prequential_predictions,
    load_config as load_parent_config,
    sha256_file,
    validate_declared_inputs,
)


PROJECT_ID = "510300_UP20_RARE_EVENT_FORECAST_V1_1_FULL_BREADTH"
CONFIG_PATH = ROOT / "config" / "510300_up20_rare_event_forecast_v1_1_full_breadth.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_up20_rare_event_forecast_v1_1_full_breadth_manifest.json"
PARENT_PROJECT_ID = "510300_UP20_RARE_EVENT_FORECAST_V1"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_receipt(receipt: dict[str, Any], label: str) -> Path:
    path = _project_path(receipt["path"])
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")
    actual = sha256_file(path)
    if actual != receipt["sha256"]:
        raise ContractError(f"{label}哈希漂移：期望{receipt['sha256']}，实际{actual}")
    return path


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"V1.1配置不存在：{path}")
    child = yaml.safe_load(path.read_text(encoding="utf-8"))
    if child.get("protocol", {}).get("project_id") != PROJECT_ID:
        raise ContractError("V1.1项目编号错误")
    if child["protocol"]["parent_v1_result_known"] is not True:
        raise ContractError("必须如实声明父版本负结果已经可见")
    if child["protocol"]["parent_v1_rejection_preserved"] is not True:
        raise ContractError("父版本拒绝状态必须永久保留")
    if child["protocol"]["only_permitted_change"] != (
        "FULL_PERIOD_OFFICIAL_PIT_EQUAL_WEIGHT_BREADTH_COVERAGE"
    ):
        raise ContractError("V1.1允许变化的范围不正确")

    parent_paths = {
        key: _validate_receipt(receipt, f"父版本{key}")
        for key, receipt in child["parent_protocol"].items()
        if isinstance(receipt, dict) and "path" in receipt and "sha256" in receipt
    }
    parent = load_parent_config(parent_paths["config"])
    parent_result = _read_json(parent_paths["result"])
    if parent_result.get("project_id") != PARENT_PROJECT_ID:
        raise ContractError("父版本结果项目编号错误")
    if parent_result.get("status") != child["parent_protocol"]["result"][
        "required_status"
    ]:
        raise ContractError("父版本正式拒绝状态发生变化")

    runtime = deepcopy(child)
    inherited = child["parent_protocol"]["inherited_without_change"]
    for section in inherited:
        if section not in parent:
            raise ContractError(f"父版本缺少继承段：{section}")
        runtime[section] = deepcopy(parent[section])
    runtime["inputs"] = deepcopy(parent["inputs"])
    for key, receipt in child["additional_inputs"].items():
        runtime["inputs"][key] = deepcopy(receipt)
    runtime["feature_data_contract"] = {
        **deepcopy(parent["feature_data_contract"]),
        **deepcopy(child["feature_data_contract"]),
    }
    runtime["paths"] = deepcopy(child["paths"])
    runtime["selection_bias"] = deepcopy(child["selection_bias"])
    runtime["protocol"] = deepcopy(child["protocol"])
    runtime["parent_protocol"] = deepcopy(child["parent_protocol"])
    runtime["additional_inputs"] = deepcopy(child["additional_inputs"])
    runtime["price_splice_contract"] = deepcopy(child["price_splice_contract"])
    runtime["coverage_hypothesis"] = deepcopy(child["coverage_hypothesis"])

    if runtime["model"] != parent["model"]:
        raise ContractError("V1.1模型与父版本不一致")
    if runtime["decision_rule"] != parent["decision_rule"]:
        raise ContractError("V1.1决策门与父版本不一致")
    if runtime["evaluation"] != parent["evaluation"]:
        raise ContractError("V1.1执行与成本合同和父版本不一致")
    if runtime["historical_acceptance_gates"] != parent[
        "historical_acceptance_gates"
    ]:
        raise ContractError("V1.1验收门与父版本不一致")
    if runtime["feature_modules"] != parent["feature_modules"]:
        raise ContractError("V1.1四模块定义与父版本不一致")
    if runtime["coverage_hypothesis"]["model_or_threshold_change_allowed"] is not False:
        raise ContractError("V1.1禁止模型或阈值变化")
    return runtime


def validate_all_inputs(config: dict[str, Any]) -> dict[str, Path]:
    resolved = validate_declared_inputs(config)
    for key, receipt in config["parent_protocol"].items():
        if isinstance(receipt, dict) and "path" in receipt and "sha256" in receipt:
            resolved[f"parent_protocol.{key}"] = _validate_receipt(
                receipt, f"父版本{key}"
            )
    for key, receipt in config["additional_inputs"].items():
        resolved[f"additional_inputs.{key}"] = _validate_receipt(
            receipt, f"扩展输入{key}"
        )
    external_audit = _read_json(
        resolved["additional_inputs.external_panel_audit"]
    )
    if external_audit.get("status") != config["additional_inputs"][
        "external_panel_audit"
    ]["required_status"]:
        raise ContractError("早期外部价格面板审计状态未通过")
    if config["price_splice_contract"]["external_membership_used"] is not False:
        raise ContractError("禁止使用外部面板成员口径")
    if config["price_splice_contract"]["current_panel_membership_used"] is not False:
        raise ContractError("禁止使用当前面板成员标志代替官方PIT成员")
    return resolved


def _prepare_native_returns(
    frame: pd.DataFrame,
    market_dates: pd.Series,
    source_name: str,
    cutoff: pd.Timestamp,
    valid_price_required_from: pd.Timestamp,
) -> pd.DataFrame:
    required = {"date", "con_code", "total_return_close"}
    if missing := required.difference(frame.columns):
        raise ContractError(f"{source_name}缺少字段：{sorted(missing)}")
    panel = frame[["date", "con_code", "total_return_close"]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["symbol"] = panel["con_code"].astype(str)
    panel["source_close"] = pd.to_numeric(panel["total_return_close"], errors="coerce")
    panel = panel.loc[panel["date"] <= cutoff, ["date", "symbol", "source_close"]]
    if panel.duplicated(["date", "symbol"]).any():
        raise ContractError(f"{source_name}存在日期股票重复")
    raw_rows = int(len(panel))
    raw_dates = int(panel["date"].nunique())
    raw_first_date = str(panel["date"].min().date())
    raw_last_date = str(panel["date"].max().date())
    invalid_price = panel["source_close"].isna() | panel["source_close"].le(0)
    invalid_on_or_after_required_date = invalid_price & panel["date"].ge(
        valid_price_required_from
    )
    if invalid_on_or_after_required_date.any():
        raise ContractError(
            f"{source_name}在价格有效性强制起点后存在空值或非正数"
        )
    invalid_rows_before_required_date = int(invalid_price.sum())
    panel = panel.loc[~invalid_price].copy()
    ordinal = pd.Series(
        np.arange(len(market_dates), dtype=int),
        index=pd.DatetimeIndex(pd.to_datetime(market_dates)),
    )
    panel["date_ordinal"] = panel["date"].map(ordinal)
    outside = int(panel["date_ordinal"].isna().sum())
    if outside:
        raise ContractError(f"{source_name}有{outside}行不在510300执行日历")
    panel["date_ordinal"] = panel["date_ordinal"].astype(int)
    panel = panel.sort_values(["symbol", "date"], ignore_index=True)
    gaps = panel.groupby("symbol", sort=False)["date_ordinal"].diff().ne(1)
    panel["source_segment"] = gaps.groupby(panel["symbol"]).cumsum()
    panel["native_return_1d"] = panel.groupby(
        [panel["symbol"], panel["source_segment"]], sort=False
    )["source_close"].pct_change(fill_method=None)
    panel["price_source_segment"] = source_name
    panel.attrs.update(
        {
            "raw_rows_through_cutoff": raw_rows,
            "raw_dates_through_cutoff": raw_dates,
            "raw_first_date": raw_first_date,
            "raw_last_date": raw_last_date,
            "valid_rows_through_cutoff": int(len(panel)),
            "invalid_price_rows_before_required_date": invalid_rows_before_required_date,
            "valid_price_required_from": str(valid_price_required_from.date()),
        }
    )
    return panel


def _overlap_return_audit(
    external: pd.DataFrame,
    current: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    start = max(external["date"].min(), current["date"].min())
    end = min(external["date"].max(), current["date"].max())
    left = external.loc[
        external["date"].between(start, end),
        ["date", "symbol", "native_return_1d"],
    ].rename(columns={"native_return_1d": "external_return"})
    right = current.loc[
        current["date"].between(start, end),
        ["date", "symbol", "native_return_1d"],
    ].rename(columns={"native_return_1d": "current_return"})
    pairs = left.merge(right, on=["date", "symbol"], how="inner").dropna()
    if pairs.empty:
        raise ContractError("两个成分价格来源没有可比原生日收益")
    difference = (pairs["external_return"] - pairs["current_return"]).abs()
    correlation = float(pairs["external_return"].corr(pairs["current_return"]))
    if math.isnan(correlation) and bool(difference.le(1e-15).all()):
        # 两个零方差序列逐点完全一致时，相关系数在数学上未定义，
        # 但对价格源一致性审计应视为完全一致；其他 NaN 情形仍然失败。
        correlation = 1.0
    median_difference = float(difference.median())
    p99_difference = float(difference.quantile(0.99))
    within_one_bp = float(difference.le(0.0001).mean())
    contract = config["feature_data_contract"]
    gates = {
        "correlation": correlation
        >= float(contract["minimum_overlap_native_return_correlation"]),
        "median_absolute_difference": median_difference
        <= float(contract["maximum_overlap_median_absolute_return_difference"]),
        "p99_absolute_difference": p99_difference
        <= float(contract["maximum_overlap_p99_absolute_return_difference"]),
        "share_within_one_basis_point": within_one_bp
        >= float(contract["minimum_overlap_share_within_one_basis_point"]),
    }
    if not all(gates.values()):
        raise ContractError(f"两个价格来源的原生日收益兼容性未通过：{gates}")
    return {
        "start_date": str(pd.Timestamp(start).date()),
        "end_date": str(pd.Timestamp(end).date()),
        "paired_returns": int(len(pairs)),
        "correlation": correlation,
        "median_absolute_return_difference": median_difference,
        "p99_absolute_return_difference": p99_difference,
        "share_within_one_basis_point": within_one_bp,
        "gates": gates,
    }


def reconstruct_spliced_price_panel(
    external: pd.DataFrame,
    current: pd.DataFrame,
    market_dates: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff = pd.Timestamp(config["dates"]["evaluation_end"])
    contract = config["feature_data_contract"]
    external_native = _prepare_native_returns(
        external,
        market_dates,
        "EXTERNAL_SINA_HFQ",
        cutoff,
        pd.Timestamp(contract["external_price_validity_required_from"]),
    )
    current_native = _prepare_native_returns(
        current,
        market_dates,
        "CURRENT_CONSTITUENT_PANEL",
        cutoff,
        pd.Timestamp(contract["current_price_validity_required_from"]),
    )
    if external_native.attrs["raw_rows_through_cutoff"] != int(
        contract["expected_external_panel_rows"]
    ):
        raise ContractError("外部价格面板行数不符合冻结合同")
    if external_native.attrs["raw_dates_through_cutoff"] != int(
        contract["expected_external_panel_dates"]
    ):
        raise ContractError("外部价格面板日期数不符合冻结合同")
    if (
        external_native.attrs["raw_first_date"]
        != contract["expected_external_first_date"]
        or external_native.attrs["raw_last_date"]
        != contract["expected_external_last_date"]
    ):
        raise ContractError("外部价格面板日期边界不符合冻结合同")
    if len(external_native) != int(contract["expected_external_valid_panel_rows"]):
        raise ContractError("外部价格面板有效行数不符合冻结合同")
    if external_native.attrs["invalid_price_rows_before_required_date"] != int(
        contract["expected_external_invalid_price_rows_before_required_from"]
    ):
        raise ContractError("外部价格面板评价前无效价格行数不符合冻结合同")
    if current_native.attrs["raw_rows_through_cutoff"] != int(
        contract["expected_current_rows_through_cutoff"]
    ):
        raise ContractError("当前成分价格面板截止日行数异常")
    if current_native.attrs["raw_dates_through_cutoff"] != int(
        contract["expected_current_dates_through_cutoff"]
    ):
        raise ContractError("当前成分价格面板截止日日期数异常")
    if len(current_native) != int(contract["expected_current_valid_rows_through_cutoff"]):
        raise ContractError("当前成分价格面板截止日有效行数异常")
    if current_native.attrs["invalid_price_rows_before_required_date"] != int(
        contract["expected_current_invalid_price_rows_before_required_from"]
    ):
        raise ContractError("当前成分价格面板评价前无效价格行数不符合冻结合同")

    overlap = _overlap_return_audit(external_native, current_native, config)
    external_last = pd.Timestamp(
        config["price_splice_contract"]["external_return_last_date"]
    )
    current_first = pd.Timestamp(
        config["price_splice_contract"]["current_return_first_date"]
    )
    if current_first <= external_last:
        raise ContractError("价格来源选择区间重叠")
    selected = pd.concat(
        [
            external_native.loc[external_native["date"] <= external_last],
            current_native.loc[current_native["date"] >= current_first],
        ],
        ignore_index=True,
    )
    if selected.duplicated(["date", "symbol"]).any():
        raise ContractError("收益拼接后存在日期股票重复")
    selected = selected.sort_values(["symbol", "date"], ignore_index=True)
    ordinal = pd.Series(
        np.arange(len(market_dates), dtype=int),
        index=pd.DatetimeIndex(pd.to_datetime(market_dates)),
    )
    selected["date_ordinal"] = selected["date"].map(ordinal).astype(int)
    ordinal_gap = selected.groupby("symbol", sort=False)["date_ordinal"].diff()
    new_segment = ordinal_gap.ne(1) | selected["native_return_1d"].isna()
    selected["continuous_segment"] = new_segment.groupby(selected["symbol"]).cumsum()
    selected["normalized_total_return_close"] = selected.groupby(
        [selected["symbol"], selected["continuous_segment"]], sort=False
    )["native_return_1d"].transform(
        lambda values: (1.0 + values.fillna(0.0)).cumprod()
    )
    selected["member_ma20"] = selected.groupby(
        [selected["symbol"], selected["continuous_segment"]], sort=False
    )["normalized_total_return_close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    diagnostics = {
        "source_price_quality": {
            "external": dict(external_native.attrs),
            "current": dict(current_native.attrs),
        },
        "overlap_native_return_audit": overlap,
        "selected_rows": int(len(selected)),
        "selected_external_rows": int(
            selected["price_source_segment"].eq("EXTERNAL_SINA_HFQ").sum()
        ),
        "selected_current_rows": int(
            selected["price_source_segment"].eq("CURRENT_CONSTITUENT_PANEL").sum()
        ),
        "selected_symbols": int(selected["symbol"].nunique()),
        "selected_first_date": str(selected["date"].min().date()),
        "selected_last_date": str(selected["date"].max().date()),
        "raw_price_level_splice_used": False,
        "native_returns_computed_before_source_selection": True,
        "external_membership_used": False,
        "external_weights_used": False,
    }
    return selected, diagnostics


def compute_full_period_official_breadth(
    official_membership: pd.DataFrame,
    reconstructed_prices: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    members = official_membership.copy()
    members["date"] = pd.to_datetime(members["membership_date"]).dt.normalize()
    members["symbol"] = members["symbol"].astype(str)
    members = members[["date", "symbol"]].sort_values(
        ["date", "symbol"], ignore_index=True
    )
    contract = config["feature_data_contract"]
    if len(members) != int(contract["expected_official_membership_rows"]):
        raise ContractError("官方PIT成员行数异常")
    counts = members.groupby("date").size()
    if not counts.eq(int(contract["expected_official_members_per_day"])).all():
        raise ContractError("官方PIT成员不是每日300只")
    prices = reconstructed_prices[
        [
            "date",
            "symbol",
            "native_return_1d",
            "normalized_total_return_close",
            "member_ma20",
            "price_source_segment",
        ]
    ]
    joined = members.merge(
        prices, on=["date", "symbol"], how="left", validate="one_to_one"
    )
    joined["return_valid"] = joined["native_return_1d"].notna()
    joined["ma20_valid"] = joined["member_ma20"].notna()
    joined["is_advancer"] = joined["native_return_1d"].gt(0.0)
    joined["is_above_ma20"] = joined["normalized_total_return_close"].gt(
        joined["member_ma20"]
    ) & joined["ma20_valid"]
    joined["is_extreme_down"] = joined["native_return_1d"].le(-0.03)
    daily = joined.groupby("date", sort=True).agg(
        member_count=("symbol", "size"),
        matched_price_count=("normalized_total_return_close", "count"),
        return_valid_count=("native_return_1d", "count"),
        ma20_valid_count=("member_ma20", "count"),
        advancer_count=("is_advancer", "sum"),
        above_ma20_count=("is_above_ma20", "sum"),
        extreme_down_count=("is_extreme_down", "sum"),
    ).reset_index()
    daily["price_coverage"] = daily["matched_price_count"] / daily["member_count"]
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
    ).where(usable_change, 0.0).fillna(0.0)
    daily["breadth_extreme_down_stress"] = (
        daily["extreme_down_share"] / 0.15
    ).clip(0.0, 1.0).where(daily["return_coverage"].ge(return_min), 0.0).fillna(0.0)

    early_end = pd.Timestamp("2019-12-20")
    post_splice = pd.Timestamp(config["price_splice_contract"]["current_return_first_date"])
    early_coverage = float(
        daily.loc[daily["date"] <= early_end, "price_coverage"].mean()
    )
    post_coverage = float(
        daily.loc[daily["date"] >= post_splice, "price_coverage"].mean()
    )
    if early_coverage < float(contract["minimum_early_official_member_price_coverage"]):
        raise ContractError("2015—2019官方成员价格覆盖率不足")
    if post_coverage < float(
        contract["minimum_post_splice_official_member_price_coverage"]
    ):
        raise ContractError("拼接后官方成员价格覆盖率不足")
    available_dates = daily.loc[daily["breadth_available"], "date"]
    first_available = (
        str(available_dates.min().date()) if not available_dates.empty else None
    )
    if first_available != contract["expected_first_breadth_available_date"]:
        raise ContractError(
            f"全期广度首个可用日异常：期望{contract['expected_first_breadth_available_date']}，实际{first_available}"
        )
    diagnostics = {
        "official_membership_rows": int(len(members)),
        "official_membership_dates": int(members["date"].nunique()),
        "early_2015_2019_mean_price_coverage": early_coverage,
        "post_splice_mean_price_coverage": post_coverage,
        "full_period_mean_price_coverage": float(daily["price_coverage"].mean()),
        "minimum_daily_price_coverage": float(daily["price_coverage"].min()),
        "minimum_daily_return_coverage": float(daily["return_coverage"].min()),
        "minimum_daily_ma20_coverage": float(daily["ma20_coverage"].min()),
        "first_breadth_available_date": first_available,
        "breadth_available_days": int(daily["breadth_available"].sum()),
        "official_membership_used": True,
        "external_membership_used": False,
        "external_weights_used": False,
        "current_panel_membership_used": False,
    }
    return daily, diagnostics


def build_outcome_free_features(
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config()
    if config["protocol"]["state"] not in (
        "READY_TO_BUILD_OUTCOME_FREE_FEATURES",
        "READY_TO_FREEZE_BEFORE_MODEL_OUTCOME_CALCULATION",
    ):
        raise ContractError("V1.1协议状态不允许构造冻结前特征")
    if MANIFEST_PATH.exists() and write:
        raise ContractError("V1.1已经冻结，拒绝覆盖冻结前特征")
    resolved = validate_all_inputs(config)
    etf, tri = _load_market_calendar_and_total_return(config, resolved)
    external = pd.read_parquet(
        resolved["additional_inputs.external_constituent_price_panel"]
    )
    current = pd.read_parquet(
        resolved["additional_inputs.current_constituent_price_panel"]
    )
    official = pd.read_parquet(
        resolved["additional_inputs.official_pit_membership"]
    )
    reconstructed, splice_diagnostics = reconstruct_spliced_price_panel(
        external, current, etf["date"], config
    )
    breadth, breadth_diagnostics = compute_full_period_official_breadth(
        official, reconstructed, config
    )
    price_features = compute_price_features(tri)
    features = assemble_outcome_free_features(price_features, breadth)
    contract = config["feature_data_contract"]
    if len(features) != int(contract["expected_feature_rows"]):
        raise ContractError("V1.1特征输出行数异常")
    if str(features["date"].iloc[0].date()) != contract[
        "expected_feature_first_date"
    ] or str(features["date"].iloc[-1].date()) != contract[
        "expected_feature_last_date"
    ]:
        raise ContractError("V1.1特征日期边界异常")
    evaluation = features["date"].between(
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    if int(evaluation.sum()) != int(contract["expected_evaluation_rows"]):
        raise ContractError("V1.1评价日数量异常")
    forbidden_tokens = ("future", "label", "state", "target", "p_up", "p_down")
    forbidden = [
        column
        for column in features.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ContractError(f"冻结前特征含结果字段：{forbidden}")
    values = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ContractError("V1.1四模块不是0到1之间有限值")
    if not features.loc[evaluation, "model_features_available"].all():
        raise ContractError("V1.1评价期价格特征存在不可用日")

    audit: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "status": "PASS_OUTCOME_FREE_FULL_PERIOD_OFFICIAL_BREADTH_READY_TO_FREEZE",
        "parent_v1_result_known": True,
        "parent_v1_rejection_preserved": True,
        "feature_stage": "BEFORE_NEW_MODEL_OUTCOME_CALCULATION",
        "input_receipts": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for key, path in resolved.items()
        },
        "calendar": {
            "feature_rows": int(len(features)),
            "evaluation_rows": int(evaluation.sum()),
            "first_date": str(features["date"].iloc[0].date()),
            "last_date": str(features["date"].iloc[-1].date()),
        },
        "price_splice": splice_diagnostics,
        "breadth": breadth_diagnostics,
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
            "external_membership_used": False,
            "historical_weights_used": False,
            "model_or_threshold_changed_from_parent": False,
        },
        "new_model_outcomes_read": False,
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
            raise FileExistsError(f"V1.1冻结前特征产物已存在：{existing}")
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
        raise FileNotFoundError("V1.1冻结清单不存在")
    manifest = _read_json(MANIFEST_PATH)
    if manifest.get("project_id") != PROJECT_ID:
        raise ContractError("V1.1冻结清单项目编号错误")
    if manifest.get("state") != "FROZEN_BEFORE_NEW_MODEL_OUTCOME_CALCULATION":
        raise ContractError("V1.1冻结状态错误")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("V1.1配置在冻结后发生漂移")
    for relative, expected in manifest.get("tracked_files", {}).items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ContractError(f"V1.1冻结跟踪文件漂移：{relative}")
    for relative, expected in manifest.get("input_files", {}).items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ContractError(f"V1.1冻结输入漂移：{relative}")
    if manifest.get("new_model_outcomes_computed_before_freeze") is not False:
        raise ContractError("V1.1没有证明新结果在冻结前不可见")
    return manifest


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
    parent = report["parent_v1_comparison"]
    lines = [
        "# 510300 UP20预测 V1.1 全期广度覆盖结果",
        "",
        f"- 正式状态：`{report['status']}`",
        "- 唯一变化：官方PIT等权广度从2015-01-05开始可用。",
        "- 未变化：四模块、模型、惩罚、概率门、压力线、20日持有期、相位、成本与验收门。",
        "- 父版本负结果已知且保留；本版本不产生Paper、订单或实盘授权。",
        "",
        "## 父版本与V1.1",
        "",
        "| 版本 | 基准净夏普 | 压力净夏普 | 捕获UP20 | 误入DOWN20 | 预测满仓区间 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| V1 | {_fmt(parent['base_net_sharpe'])} | {_fmt(parent['stress_net_sharpe'])} | "
        f"{parent['captured_up_blocks']} | {parent['false_down_blocks']} | {parent['predicted_long_blocks']} |",
        f"| V1.1 | {_fmt(primary['base']['sharpe_zero_cash_rate'])} | "
        f"{_fmt(primary['stress']['sharpe_zero_cash_rate'])} | {classification['captured_up_blocks']} | "
        f"{classification['false_down_blocks']} | {classification['predicted_long_blocks']} |",
        "",
        "## V1.1完整账户结果",
        "",
        "| 指标 | 结果 | 冻结门 |",
        "|---|---:|---:|",
        f"| 基准净夏普 | {_fmt(primary['base']['sharpe_zero_cash_rate'])} | ≥1.2 |",
        f"| 压力净夏普 | {_fmt(primary['stress']['sharpe_zero_cash_rate'])} | ≥1.2 |",
        f"| 2021年以来压力净夏普 | {_fmt(primary['recent_period']['stress']['net_sharpe'])} | ≥1.2 |",
        f"| 压力总收益 | {_pct(primary['stress']['total_return'])} | >0 |",
        f"| 压力最大回撤 | {_pct(primary['stress']['max_drawdown'])} | 诊断 |",
        f"| 捕获UP20 | {classification['captured_up_blocks']} | ≥14 |",
        f"| 误入DOWN20 | {classification['false_down_blocks']} | ≤1 |",
        f"| 误入RANGE20 | {classification['false_range_blocks']} | ≤10 |",
        f"| 预测满仓区间 | {classification['predicted_long_blocks']} | ≥14 |",
        f"| Brier Skill | {_fmt(skill['brier_skill'])} | >0 |",
        f"| Log Loss Skill | {_fmt(skill['log_loss_skill'])} | >0 |",
        "",
        "## 20种相位压力净夏普",
        "",
        "| offset | 压力净夏普 | 满仓区间 | 捕获UP | 误入DOWN | 误入RANGE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in report["phase_results"]:
        metrics = phase["classification"]
        lines.append(
            f"| {phase['phase_offset']} | {_fmt(phase['stress']['sharpe_zero_cash_rate'])} | "
            f"{metrics['predicted_long_blocks']} | {metrics['captured_up_blocks']} | "
            f"{metrics['false_down_blocks']} | {metrics['false_range_blocks']} |"
        )
    lines.extend(["", "## 冻结门", ""])
    for name, passed in report["adjudication"]["gates"].items():
        lines.append(f"- `{name}`：`{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## 正确解释",
            "",
            report["adjudication"]["interpretation"],
            "",
            "独立前向观察数仍为0；Paper、Shadow、持仓映射、订单、券商连接与实盘授权全部关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(write: bool = True, progress: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    resolved = validate_all_inputs(config)
    feature_path = _project_path(config["paths"]["features_prefreeze"])
    audit_path = _project_path(config["paths"]["feature_audit"])
    if not feature_path.exists() or not audit_path.exists():
        raise FileNotFoundError("V1.1冻结前特征或审计不存在")
    for path in (feature_path, audit_path):
        relative = path.relative_to(ROOT).as_posix()
        if manifest["input_files"].get(relative) != sha256_file(path):
            raise ContractError(f"V1.1冻结特征与清单不一致：{relative}")
    audit = _read_json(audit_path)
    if audit.get("status") != (
        "PASS_OUTCOME_FREE_FULL_PERIOD_OFFICIAL_BREADTH_READY_TO_FREEZE"
    ):
        raise ContractError("V1.1冻结前特征审计未通过")
    if audit.get("new_model_outcomes_read") is not False:
        raise ContractError("V1.1冻结前已经读取新模型结果")

    formal_keys = (
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
    output_paths = {key: _project_path(config["paths"][key]) for key in formal_keys}
    if write:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"V1.1正式结果已存在，拒绝覆盖：{existing}")

    features = _normalize_date_column(pd.read_parquet(feature_path), "date")
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
    primary_offset = int(config["evaluation"]["primary_phase_offset"])
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
        if int(offset) == primary_offset:
            primary_blocks = blocks
            primary_frames = frames
    if primary_blocks is None or primary_frames is None:
        raise ContractError("V1.1主相位账户结果缺失")
    primary_report = next(
        phase for phase in phase_reports if phase["phase_offset"] == primary_offset
    )

    control_reports: dict[str, Any] = {}
    for control_name, signal_column in (
        ("WITHOUT_HARD_STRESS_VETO", "control_without_stress_veto_long_signal"),
        (
            "WITHOUT_DOWN_PROBABILITY_VETO",
            "control_without_down_probability_veto_long_signal",
        ),
    ):
        control, _blocks, _frames = evaluate_phase(
            predictions,
            labels,
            evaluation_dates,
            etf,
            dividends,
            config,
            primary_offset,
            signal_column,
        )
        control_reports[control_name] = control

    gates_config = config["historical_acceptance_gates"]
    classification = primary_report["classification"]
    skill = primary_report["probability_skill"]
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
            skill["brier_skill"] is not None and float(skill["brier_skill"]) > 0.0
        ),
        "primary_log_loss_skill_vs_causal_base_positive": bool(
            skill["log_loss_skill"] is not None
            and float(skill["log_loss_skill"]) > 0.0
        ),
        "minimum_primary_predicted_long_blocks": bool(
            classification["predicted_long_blocks"]
            >= int(gates_config["minimum_primary_predicted_long_blocks"])
        ),
    }
    historical_pass = bool(all(gates.values()))
    status = (
        "PASS_FROZEN_UP20_RARE_EVENT_FORECAST_V1_1_FULL_BREADTH_HISTORICAL_ONLY_FORWARD_REQUIRED_NOT_TRADABLE"
        if historical_pass
        else "REJECTED_FROZEN_UP20_RARE_EVENT_FORECAST_V1_1_FULL_BREADTH_NO_RESCUE"
    )
    interpretation = (
        "仅补齐官方PIT全期广度后，同一V1模型和全部冻结门均通过；这只建立历史候选，必须转入独立前向阶段。"
        if historical_pass
        else "补齐2015年以来官方PIT等权广度仍未使同一V1模型通过；广度历史缺失不能解释父版本失败，V1家族停止且不得调参救援。"
    )
    parent_result = _read_json(
        resolved["parent_protocol.result"]
    )
    parent_primary = parent_result["primary_result"]
    parent_comparison = {
        "project_id": parent_result["project_id"],
        "status": parent_result["status"],
        "base_net_sharpe": parent_primary["base"]["sharpe_zero_cash_rate"],
        "stress_net_sharpe": parent_primary["stress"]["sharpe_zero_cash_rate"],
        "captured_up_blocks": parent_primary["classification"]["captured_up_blocks"],
        "false_down_blocks": parent_primary["classification"]["false_down_blocks"],
        "false_range_blocks": parent_primary["classification"]["false_range_blocks"],
        "predicted_long_blocks": parent_primary["classification"][
            "predicted_long_blocks"
        ],
        "result_unchanged": True,
    }
    finite_sharpes = [
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
        "coverage_hypothesis": config["coverage_hypothesis"],
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "parent_v1_result_known": True,
            "new_model_outcomes_computed_before_freeze": False,
        },
        "only_change_vs_parent": {
            "full_period_official_pit_equal_weight_breadth": True,
            "model_changed": False,
            "decision_thresholds_changed": False,
            "costs_changed": False,
            "holding_period_changed": False,
            "phase_changed": False,
            "acceptance_gates_changed": False,
            "external_membership_used": False,
            "historical_weights_used": False,
        },
        "feature_audit": {
            "path": audit_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(audit_path),
            "status": audit["status"],
            "first_breadth_available_date": audit["breadth"][
                "first_breadth_available_date"
            ],
        },
        "model_diagnostics": {
            "evaluation_predictions": int(len(predictions)),
            "valid_predictions": int(predictions["model_valid"].sum()),
            "invalid_predictions": int((~predictions["model_valid"]).sum()),
            "optimizer_failures": int((~predictions["optimizer_success"]).sum()),
            "sign_constraint_failures": int(
                (predictions["model_valid"] & ~predictions["sign_constraints_hold"]).sum()
            ),
        },
        "parent_v1_comparison": parent_comparison,
        "primary_result": primary_report,
        "phase_results": phase_reports,
        "phase_robustness_summary": {
            "minimum_stress_net_sharpe": min(finite_sharpes) if finite_sharpes else None,
            "median_stress_net_sharpe": (
                float(np.median(finite_sharpes)) if finite_sharpes else None
            ),
            "maximum_stress_net_sharpe": max(finite_sharpes) if finite_sharpes else None,
            "phase_count": int(len(phase_reports)),
            "all_phases_at_least_1_2": gates["every_phase_stress_net_sharpe"],
        },
        "controls_not_eligible_for_selection": control_reports,
        "adjudication": {
            "historical_acceptance_passed": historical_pass,
            "coverage_hypothesis_supported": historical_pass,
            "gates": gates,
            "interpretation": interpretation,
            "historical_account_simulation_target_achieved": historical_pass,
            "verified_forward_observations": 0,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "live_trading_authorized": False,
        },
        "selection_bias_control": manifest["selection_bias_control"],
        "governance": {
            "parent_rejection_preserved": True,
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
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "PROJECT_ID",
    "build_outcome_free_features",
    "compute_full_period_official_breadth",
    "load_config",
    "reconstruct_spliced_price_panel",
    "run_study",
    "validate_manifest",
]
