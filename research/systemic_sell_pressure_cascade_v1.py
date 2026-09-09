"""全A股系统性卖压扩散驱动的510300次日二元退出候选V1。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.binary_state_feasibility_v1 import (
    CostModel,
    build_benchmark_ledger,
    build_perfect_block_states,
    load_and_audit_inputs as load_v1_inputs,
    load_config as load_v1_config,
    simulate_binary_path,
    summarize_path,
    validate_manifest as validate_v1_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_systemic_sell_pressure_cascade_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_systemic_sell_pressure_cascade_v1_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    scope = config["scope"]
    objective = config["objective"]
    signal = config["signal"]
    partitions = config["partitions"]
    if protocol["candidate_id"] != "510300_SYSTEMIC_SELL_PRESSURE_CASCADE_V1":
        raise ValueError("候选编号不匹配")
    if protocol["prior_failed_formula_reused"] or protocol["prior_failed_family_parameter_rescue"]:
        raise ValueError("本候选不得救回既有失败公式族")
    if protocol["outcome_used_to_choose_parameters"]:
        raise ValueError("参数不得由本候选结果选择")
    if protocol["live_trading_authorized"]:
        raise ValueError("历史候选不得授权实盘")
    if scope["execution_asset"] != "510300.SH":
        raise ValueError("唯一执行资产必须是510300.SH")
    if list(scope["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("允许持有资产必须严格为510300与现金")
    if list(scope["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if int(scope["cash_exit_holding_days"]) != 1:
        raise ValueError("本候选每次空仓必须只持续一个交易日")
    if scope["leverage_allowed"] or scope["short_selling_allowed"] or scope["derivatives_allowed"]:
        raise ValueError("禁止杠杆、卖空和衍生品")
    if objective["benchmark_id"] != "H00300":
        raise ValueError("基准必须是H00300全收益指数")
    if float(objective["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额目标必须固定为20个百分点")
    if float(objective["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动超额中位数目标必须固定为20个百分点")
    expected_features = [
        "down_2_share",
        "down_5_share",
        "lower_limit_close_share",
        "close_low_down_share",
        "amount_weighted_down_share",
        "amount_weighted_down_5_share",
        "negative_cross_section_median_return",
        "cross_section_return_dispersion",
    ]
    if list(signal["risk_features_equal_weighted"]) != expected_features:
        raise ValueError("系统性卖压特征集合或顺序偏离冻结定义")
    if int(signal["raw_feature_prior_window_trading_days"]) != 242:
        raise ValueError("点时百分位窗口必须固定为242日")
    if int(config["universe"]["pre_panel_listing_age_calendar_days_for_mature_history"]) != 180:
        raise ValueError("面板前成熟股票的上市年龄必须固定为180个日历日")
    if float(signal["score_cutoff_quantile"]) != 0.95:
        raise ValueError("风险分数阈值必须固定为校准期95%分位")
    if signal["score_cutoff_quantile_method"] != "linear":
        raise ValueError("风险分数分位算法必须固定为linear")
    if signal["parameter_grid_search_allowed"]:
        raise ValueError("禁止参数网格搜索")
    if partitions["visible_validation_execution_start"] != "2019-01-02":
        raise ValueError("可见验证执行起点偏离冻结定义")
    if partitions["visible_validation_execution_end"] != "2022-12-30":
        raise ValueError("可见验证执行终点偏离冻结定义")
    if partitions["replication_execution_start"] != "2025-01-02":
        raise ValueError("条件复验执行起点偏离冻结定义")
    if not partitions["replication_panel_may_be_loaded_only_after_visible_gates_pass"]:
        raise ValueError("复验面板必须在可见门通过后才允许载入")


def validate_manifest(root: Path = ROOT, path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("候选冻结清单不存在，禁止读取历史结果")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("candidate_id") != "510300_SYSTEMIC_SELL_PRESSURE_CASCADE_V1":
        raise ValueError("候选冻结清单编号不匹配")
    drift: list[dict[str, str | None]] = []
    for section in ("tracked_files", "data_files"):
        for relative, expected in manifest.get(section, {}).items():
            target = root / relative
            actual = sha256_file(target) if target.exists() else None
            if actual != expected:
                drift.append({"文件": relative, "预期": expected, "实际": actual})
    if drift:
        raise ValueError(f"候选冻结清单漂移：{json.dumps(drift, ensure_ascii=False)}")
    return manifest


def load_master(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    master = pd.read_parquet(root / config["inputs"]["stock_master"]).copy()
    required = {
        "ts_code",
        "list_status",
        "list_date",
        "delist_date",
        "split_group",
    }
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"股票主表缺少列：{missing}")
    master["ts_code"] = master["ts_code"].astype(str)
    master["list_date"] = pd.to_datetime(master["list_date"], errors="raise").dt.normalize()
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce").dt.normalize()
    if master["ts_code"].duplicated().any():
        raise ValueError("股票主表代码重复")
    return master.reset_index(drop=True)


def _strict_prior_rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    """当前值相对严格此前固定窗口的右侧经验CDF。"""

    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    if window <= 0:
        raise ValueError("百分位窗口必须为正整数")
    for index in range(window, len(numeric)):
        current = numeric[index]
        prior = numeric[index - window : index]
        if np.isfinite(current) and np.isfinite(prior).all():
            result[index] = float(np.searchsorted(np.sort(prior), current, side="right") / window)
    return pd.Series(result, index=values.index, dtype=float)


def build_daily_features_from_frame(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    *,
    required_split_group: str,
    feature_start: str,
    feature_end: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_panel = {
        "con_code",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "pct_chg",
        "amount",
        "is_suspended",
    }
    missing = sorted(required_panel.difference(panel.columns))
    if missing:
        raise ValueError(f"股票日线面板缺少列：{missing}")
    data = panel[list(required_panel)].copy()
    data["con_code"] = data["con_code"].astype(str)
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    start = pd.Timestamp(feature_start)
    end = pd.Timestamp(feature_end)
    data = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    if data.empty:
        raise ValueError("特征区间内没有股票日线")
    if data.duplicated(["con_code", "date"]).any():
        raise ValueError("股票日线存在代码日期重复")

    universe = config["universe"]
    allowed_master = master.loc[
        (master["split_group"] == required_split_group)
        & master["list_status"].isin(universe["master_statuses"])
    ].copy()
    allowed_codes = set(allowed_master["ts_code"])
    unexpected_codes = sorted(set(data["con_code"]).difference(allowed_codes))
    if unexpected_codes:
        raise ValueError(f"面板含不属于{required_split_group}的股票：{unexpected_codes[:10]}")
    master_index = allowed_master.set_index("ts_code")
    data["list_date"] = data["con_code"].map(master_index["list_date"])
    data["delist_date"] = data["con_code"].map(master_index["delist_date"])
    listing_violation = (
        data["list_date"].isna()
        | (data["date"] < data["list_date"])
        | (data["delist_date"].notna() & (data["date"] > data["delist_date"]))
    )
    if listing_violation.any():
        raise ValueError(f"发现{int(listing_violation.sum())}行违反点时上市退市区间")

    numeric_columns = ["raw_open", "raw_high", "raw_low", "raw_close", "pct_chg", "amount"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data.sort_values(["con_code", "date"], inplace=True, kind="mergesort")
    observed_history = data.groupby("con_code", sort=False).cumcount() + 1
    pre_panel_mature = data["list_date"] <= (
        start
        - pd.Timedelta(
            days=int(universe["pre_panel_listing_age_calendar_days_for_mature_history"])
        )
    )
    history_mature = (
        observed_history >= int(universe["minimum_observed_history_days"])
    ) | pre_panel_mature
    valid_prices = (
        data[["raw_open", "raw_high", "raw_low", "raw_close"]].notna().all(axis=1)
        & (data[["raw_open", "raw_high", "raw_low", "raw_close"]] > 0.0).all(axis=1)
        & (data["raw_high"] >= data[["raw_open", "raw_close"]].max(axis=1))
        & (data["raw_low"] <= data[["raw_open", "raw_close"]].min(axis=1))
    )
    valid_amount = data["amount"].notna() & (data["amount"] > 0.0)
    not_suspended = ~data["is_suspended"].fillna(True).astype(bool)
    eligible = (
        history_mature
        & valid_prices
        & valid_amount
        & not_suspended
        & data["pct_chg"].notna()
    )
    eligible_data = data.loc[eligible].copy()
    if eligible_data.empty:
        raise ValueError("没有形成合格股票日线")
    returns = eligible_data["pct_chg"].to_numpy(dtype=float) / 100.0
    eligible_data["return_1d"] = returns
    day_range = eligible_data["raw_high"] - eligible_data["raw_low"]
    close_location = np.where(
        day_range > 0.0,
        (eligible_data["raw_close"] - eligible_data["raw_low"]) / day_range,
        np.where(eligible_data["return_1d"] < 0.0, 0.0, 0.5),
    )
    eligible_data["close_location"] = np.clip(close_location, 0.0, 1.0)
    signal = config["signal"]
    eligible_data["down_2"] = eligible_data["return_1d"] <= float(signal["down_2_threshold"])
    eligible_data["down_5"] = eligible_data["return_1d"] <= float(signal["down_5_threshold"])
    eligible_data["lower_limit_close"] = (
        (eligible_data["return_1d"] <= float(signal["lower_limit_return_threshold"]))
        & (
            (eligible_data["raw_close"] - eligible_data["raw_low"]).abs()
            <= float(signal["lower_limit_close_absolute_tolerance_cny"])
        )
    )
    eligible_data["close_low_down"] = (
        (eligible_data["return_1d"] < 0.0)
        & (eligible_data["close_location"] <= float(signal["close_location_upper_bound"]))
    )
    eligible_data["down"] = eligible_data["return_1d"] < 0.0
    eligible_data["amount_down"] = eligible_data["amount"] * eligible_data["down"].astype(float)
    eligible_data["amount_down_5"] = eligible_data["amount"] * eligible_data["down_5"].astype(float)

    grouped = eligible_data.groupby("date", sort=True)
    daily = grouped.agg(
        eligible_stock_count=("con_code", "size"),
        down_2_count=("down_2", "sum"),
        down_5_count=("down_5", "sum"),
        lower_limit_close_count=("lower_limit_close", "sum"),
        close_low_down_count=("close_low_down", "sum"),
        total_amount=("amount", "sum"),
        down_amount=("amount_down", "sum"),
        down_5_amount=("amount_down_5", "sum"),
        cross_section_median_return=("return_1d", "median"),
        cross_section_return_dispersion=("return_1d", "std"),
    ).reset_index()
    if (daily["eligible_stock_count"] <= 0).any() or (daily["total_amount"] <= 0.0).any():
        raise ValueError("日度聚合出现空股票池或非正成交额")
    daily["down_2_share"] = daily["down_2_count"] / daily["eligible_stock_count"]
    daily["down_5_share"] = daily["down_5_count"] / daily["eligible_stock_count"]
    daily["lower_limit_close_share"] = (
        daily["lower_limit_close_count"] / daily["eligible_stock_count"]
    )
    daily["close_low_down_share"] = daily["close_low_down_count"] / daily["eligible_stock_count"]
    daily["amount_weighted_down_share"] = daily["down_amount"] / daily["total_amount"]
    daily["amount_weighted_down_5_share"] = daily["down_5_amount"] / daily["total_amount"]
    daily["negative_cross_section_median_return"] = -daily["cross_section_median_return"]
    risk_features = list(signal["risk_features_equal_weighted"])
    window = int(signal["raw_feature_prior_window_trading_days"])
    percentile_columns: list[str] = []
    for feature in risk_features:
        percentile_column = f"{feature}_risk_percentile"
        daily[percentile_column] = _strict_prior_rolling_percentile(daily[feature], window)
        percentile_columns.append(percentile_column)
    daily["systemic_sell_pressure_risk_score"] = daily[percentile_columns].mean(
        axis=1,
        skipna=False,
    )
    daily["risk_feature_percentiles_available"] = daily[percentile_columns].notna().all(axis=1)
    daily.sort_values("date", inplace=True)
    daily.reset_index(drop=True, inplace=True)

    audit = {
        "required_split_group": required_split_group,
        "raw_rows_in_feature_interval": int(len(data)),
        "raw_unique_stocks": int(data["con_code"].nunique()),
        "eligible_rows": int(len(eligible_data)),
        "eligible_unique_stocks": int(eligible_data["con_code"].nunique()),
        "daily_rows": int(len(daily)),
        "first_date": daily["date"].iloc[0].date().isoformat(),
        "last_date": daily["date"].iloc[-1].date().isoformat(),
        "first_risk_score_date": (
            None
            if not daily["risk_feature_percentiles_available"].any()
            else daily.loc[daily["risk_feature_percentiles_available"], "date"].iloc[0].date().isoformat()
        ),
        "duplicate_code_dates": 0,
        "listing_interval_violations": 0,
        "invalid_price_rows": int((~valid_prices).sum()),
        "invalid_amount_rows": int((~valid_amount).sum()),
        "suspended_rows": int((~not_suspended).sum()),
        "minimum_eligible_stock_count": int(daily["eligible_stock_count"].min()),
        "median_eligible_stock_count": float(daily["eligible_stock_count"].median()),
        "status": "PASS",
    }
    return daily, audit


def load_and_build_daily_features(
    root: Path,
    panel_relative_path: str,
    master: pd.DataFrame,
    *,
    required_split_group: str,
    feature_start: str,
    feature_end: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "con_code",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "pct_chg",
        "amount",
        "is_suspended",
    ]
    panel = pd.read_parquet(root / panel_relative_path, columns=columns)
    features, audit = build_daily_features_from_frame(
        panel,
        master,
        required_split_group=required_split_group,
        feature_start=feature_start,
        feature_end=feature_end,
        config=config,
    )
    audit["panel_path"] = panel_relative_path
    audit["panel_sha256"] = sha256_file(root / panel_relative_path)
    return features, audit


def calibrate_and_apply_signal(
    features: pd.DataFrame,
    *,
    calibration_start: str,
    calibration_end: str,
    signal_start: str,
    signal_end: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = features.copy()
    calibration_mask = (
        (data["date"] >= pd.Timestamp(calibration_start))
        & (data["date"] <= pd.Timestamp(calibration_end))
        & data["risk_feature_percentiles_available"]
    )
    calibration_scores = data.loc[calibration_mask, "systemic_sell_pressure_risk_score"].dropna()
    if len(calibration_scores) < 242:
        raise ValueError("风险分数校准样本不足242日")
    signal_contract = config["signal"]
    cutoff = float(
        np.quantile(
            calibration_scores.to_numpy(dtype=float),
            float(signal_contract["score_cutoff_quantile"]),
            method=signal_contract["score_cutoff_quantile_method"],
        )
    )
    in_signal_partition = (
        (data["date"] >= pd.Timestamp(signal_start))
        & (data["date"] <= pd.Timestamp(signal_end))
    )
    direction_condition = data["cross_section_median_return"] < 0.0
    amount_condition = data["amount_weighted_down_share"] >= float(
        signal_contract["require_amount_weighted_down_share_at_least"]
    )
    data["cash_next_trading_day"] = (
        in_signal_partition
        & data["risk_feature_percentiles_available"]
        & (data["systemic_sell_pressure_risk_score"] >= cutoff)
        & direction_condition
        & amount_condition
    )
    data["score_cutoff"] = cutoff
    signal_rows = data.loc[in_signal_partition].copy()
    if signal_rows.empty:
        raise ValueError("信号分区没有日度特征")
    calibration = {
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "calibration_observations": int(len(calibration_scores)),
        "score_cutoff_quantile": float(signal_contract["score_cutoff_quantile"]),
        "score_cutoff": cutoff,
        "signal_start": signal_start,
        "signal_end": signal_end,
        "signal_partition_observations": int(len(signal_rows)),
        "risk_score_available_signal_days": int(
            signal_rows["risk_feature_percentiles_available"].sum()
        ),
        "cash_signal_days": int(signal_rows["cash_next_trading_day"].sum()),
        "cash_signal_share": float(signal_rows["cash_next_trading_day"].mean()),
    }
    return data, calibration


def cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    if scenario not in {"BASE", "STRESS"}:
        raise ValueError("成本情景必须为BASE或STRESS")
    costs = config["costs"]
    slippage = (
        costs["base_slippage_bps_per_leg"]
        if scenario == "BASE"
        else costs["stress_slippage_bps_per_leg"]
    )
    return CostModel(
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(slippage),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def build_policy_states(
    sample_market: pd.DataFrame,
    signal_frame: pd.DataFrame,
) -> np.ndarray:
    """第i日开盘状态严格由第i-1个交易日收盘信号决定。"""

    sample = sample_market.reset_index(drop=True)
    if len(sample) < 2:
        raise ValueError("组合样本至少需要信号日和执行日")
    signal_map = {
        pd.Timestamp(row.date): bool(row.cash_next_trading_day)
        for row in signal_frame[["date", "cash_next_trading_day"]].itertuples(index=False)
    }
    states = np.ones(len(sample), dtype=np.int8)
    states[0] = 0
    for index in range(1, len(sample)):
        prior_date = pd.Timestamp(sample.loc[index - 1, "date"])
        if prior_date not in signal_map:
            raise ValueError(f"执行日前一交易日缺少信号：{prior_date.date().isoformat()}")
        states[index] = 0 if signal_map[prior_date] else 1
    return states


def _tail_capture_diagnostics(
    sample: pd.DataFrame,
    dividends: pd.DataFrame,
    states: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    _, blocks, last_end = build_perfect_block_states(
        sample,
        dividends,
        horizon=1,
        cash_annual_rate=float(config["costs"]["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
    )
    if last_end != len(sample) - 1 or len(blocks) != len(sample) - 1:
        raise ValueError("1日神谕块没有覆盖全部执行日")
    true_state = blocks["oracle_state"].to_numpy(dtype=np.int8)
    exits = states[1:] == 0
    bad = true_state == 0
    good = true_state == 1
    loss = (
        blocks["cash_gross_factor"].to_numpy(dtype=float)
        - blocks["full_gross_factor"].to_numpy(dtype=float)
    ).clip(min=0.0)
    total_loss = float(loss[bad].sum())
    captured_loss = float(loss[bad & exits].sum())
    return {
        "execution_days": int(len(blocks)),
        "cash_days": int(exits.sum()),
        "cash_day_share": float(exits.mean()),
        "bad_day_count": int(bad.sum()),
        "good_day_count": int(good.sum()),
        "bad_day_recall": float((exits & bad).sum() / bad.sum()) if bad.any() else None,
        "false_exit_rate_on_good_days": float((exits & good).sum() / good.sum()) if good.any() else None,
        "bad_opportunity_loss_capture": (
            0.0 if total_loss <= 0.0 else captured_loss / total_loss
        ),
        "cash_day_precision_bad": float((exits & bad).sum() / exits.sum()) if exits.any() else None,
    }


def evaluate_partition(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    signal_frame: pd.DataFrame,
    *,
    signal_start: str,
    execution_start: str,
    execution_end: str,
    scenarios: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    sample = market.loc[
        (market["date"] >= pd.Timestamp(signal_start))
        & (market["date"] <= pd.Timestamp(execution_end))
    ].reset_index(drop=True)
    if sample.empty:
        raise ValueError("组合评价区间没有市场数据")
    if sample["date"].iloc[0] != pd.Timestamp(signal_start):
        raise ValueError("组合样本首日不是冻结信号起点")
    if sample["date"].iloc[1] != pd.Timestamp(execution_start):
        raise ValueError("组合样本第二日不是冻结执行起点")
    if sample["date"].iloc[-1] != pd.Timestamp(execution_end):
        raise ValueError("组合样本终日不是冻结执行终点")
    states = build_policy_states(sample, signal_frame)
    benchmark = build_benchmark_ledger(
        sample,
        float(config["scope"]["initial_capital_cny"]),
    )
    paths: dict[str, Any] = {}
    for scenario in scenarios:
        ledger, trades = simulate_binary_path(
            sample,
            dividends,
            states,
            costs=cost_model(config, scenario),
            initial_capital=float(config["scope"]["initial_capital_cny"]),
            reinvest_paid_dividends=bool(
                config["scope"]["full_state_reinvests_paid_cash_dividends"]
            ),
        )
        summary = summarize_path(
            ledger,
            trades,
            benchmark,
            objective=config["objective"],
        )
        paths[scenario] = {
            "summary": summary,
            "ledger": ledger,
            "trades": trades,
        }
    return {
        "sample": sample,
        "states": states,
        "tail_capture": _tail_capture_diagnostics(sample, dividends, states, config),
        "paths": paths,
    }


def visible_gate_checks(
    evaluation: dict[str, Any],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    target = float(config["objective"]["minimum_annualized_excess"])
    rolling_target = float(config["objective"]["minimum_rolling_excess_median"])
    base = evaluation["paths"]["BASE"]["summary"]
    stress = evaluation["paths"]["STRESS"]["summary"]
    checks = {
        "base_annualized_excess_at_least_20pct": bool(base["annualized_excess"] >= target),
        "stress_annualized_excess_at_least_20pct": bool(stress["annualized_excess"] >= target),
        "base_rolling_excess_median_at_least_20pct": bool(
            base["rolling_242d_excess_median"] is not None
            and base["rolling_242d_excess_median"] >= rolling_target
        ),
        "stress_rolling_excess_median_at_least_20pct": bool(
            stress["rolling_242d_excess_median"] is not None
            and stress["rolling_242d_excess_median"] >= rolling_target
        ),
        "minimum_signal_days": bool(
            calibration["cash_signal_days"]
            >= int(config["visible_acceptance"]["minimum_signal_days"])
        ),
    }
    checks["all_visible_gates_pass"] = bool(all(checks.values()))
    return checks


def replication_gate_checks(
    evaluation: dict[str, Any],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    target = float(config["objective"]["minimum_annualized_excess"])
    rolling_target = float(config["objective"]["minimum_rolling_excess_median"])
    stress = evaluation["paths"]["STRESS"]["summary"]
    checks = {
        "stress_annualized_excess_at_least_20pct": bool(stress["annualized_excess"] >= target),
        "stress_rolling_excess_median_at_least_20pct": bool(
            stress["rolling_242d_excess_median"] is not None
            and stress["rolling_242d_excess_median"] >= rolling_target
        ),
        "minimum_signal_days": bool(
            calibration["cash_signal_days"]
            >= int(config["replication_acceptance"]["minimum_signal_days"])
        ),
    }
    checks["all_replication_gates_pass"] = bool(all(checks.values()))
    return checks


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    v1_manifest: dict[str, Any],
    market_audit: dict[str, Any],
    visible_feature_audit: dict[str, Any],
    visible_calibration: dict[str, Any],
    visible_evaluation: dict[str, Any],
    visible_gates: dict[str, bool],
    replication_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    visible_pass = bool(visible_gates["all_visible_gates_pass"])
    if not visible_pass:
        status = "VISIBLE_REJECTED_FROZEN_REPLICATION_UNREAD"
    elif replication_payload is None:
        status = "VISIBLE_PASS_REPLICATION_NOT_COMPLETED"
    elif replication_payload["gates"]["all_replication_gates_pass"]:
        status = "VISIBLE_AND_REPLICATION_HISTORICAL_PASS_PAPER_SHADOW_ONLY"
    else:
        status = "VISIBLE_PASS_REPLICATION_REJECTED_FROZEN"
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "candidate_id": config["protocol"]["candidate_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": status,
        "goal_achieved": False,
        "reason_goal_not_achieved": (
            "候选未通过冻结历史门槛"
            if not visible_pass
            else "历史结果即使通过也不能替代真正前向Paper或Shadow证据"
        ),
        "historical_evidence_label": config["protocol"]["historical_evidence_label"],
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(MANIFEST_PATH),
            "frozen_at": manifest["frozen_at"],
        },
        "upstream_v1_manifest_frozen_at": v1_manifest["frozen_at"],
        "scope": config["scope"],
        "objective": config["objective"],
        "costs": config["costs"],
        "market_data_audit": market_audit,
        "visible": {
            "feature_audit": visible_feature_audit,
            "calibration": visible_calibration,
            "base_summary": visible_evaluation["paths"]["BASE"]["summary"],
            "stress_summary": visible_evaluation["paths"]["STRESS"]["summary"],
            "tail_capture": visible_evaluation["tail_capture"],
            "gates": visible_gates,
        },
        "replication_panel_read": replication_payload is not None,
        "replication": replication_payload,
        "decision": {
            "visible_pass": visible_pass,
            "replication_opened": replication_payload is not None,
            "parameter_rescue_allowed": False,
            "current_state_signal_created": False,
            "paper_or_shadow_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    visible = report["visible"]
    base = visible["base_summary"]
    stress = visible["stress_summary"]
    tail = visible["tail_capture"]
    lines = [
        "# 510300 全A股系统性卖压扩散次日退出 V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 目标达成：`{str(report['goal_achieved']).lower()}`",
        "- 仓位：仅满仓510300或空仓现金；信号后只退出一个交易日。",
        "",
        "## 2019—2022可见验证",
        "",
        f"- 风险分数校准样本：{visible['calibration']['calibration_observations']}日；95%阈值：{visible['calibration']['score_cutoff']:.6f}。",
        f"- 空仓信号：{visible['calibration']['cash_signal_days']}日，占验证信号日{visible['calibration']['cash_signal_share']:.2%}。",
        f"- 基础/压力年化净超额：{base['annualized_excess']:.2%} / {stress['annualized_excess']:.2%}。",
        f"- 基础/压力242日滚动超额中位数：{base['rolling_242d_excess_median']:.2%} / {stress['rolling_242d_excess_median']:.2%}。",
        f"- 压力策略年化：{stress['strategy_cagr']:.2%}；H00300年化：{stress['benchmark_cagr']:.2%}。",
        f"- 压力最大回撤：{stress['maximum_drawdown']:.2%}；成交腿数：{stress['trade_leg_count']}。",
        f"- 坏日召回：{tail['bad_day_recall']:.2%}；好日误退出：{tail['false_exit_rate_on_good_days']:.2%}；坏日机会损失捕获：{tail['bad_opportunity_loss_capture']:.2%}。",
        "",
        "## 硬门",
        "",
    ]
    for name, passed in visible["gates"].items():
        lines.append(f"- `{name}`：{'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## 条件复验", ""])
    if not report["replication_panel_read"]:
        lines.append("可见门未全部通过，2023—2026独立哈希股票面板未载入。")
    else:
        replication = report["replication"]
        replication_stress = replication["stress_summary"]
        lines.extend(
            [
                f"- 复验空仓信号：{replication['calibration']['cash_signal_days']}日。",
                f"- 压力年化净超额：{replication_stress['annualized_excess']:.2%}。",
                f"- 压力242日滚动超额中位数：{replication_stress['rolling_242d_excess_median']:.2%}。",
            ]
        )
        for name, passed in replication["gates"].items():
            lines.append(f"- `{name}`：{'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 决策",
            "",
            "失败后不改变特征、窗口、阈值、空仓期限或验证区间救回；本报告未生成当前状态、Paper/Shadow、订单或实盘授权。",
        ]
    )
    return "\n".join(lines) + "\n"
