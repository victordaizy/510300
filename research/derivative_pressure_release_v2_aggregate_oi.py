from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from research.episodic_alpha_library_v1 import (
    _build_option_pressure_panel,
    _entry_open,
    _load_etf_daily,
    _load_minute_bars,
    _minute_day_lookup,
    _stress_net_return,
    confirmation_from_minutes,
    event_metrics,
    historical_screen_decision,
    prior_rolling_quantile,
    prior_rolling_z,
)
from research.frozen_protocol_support_v1 import (
    ProtocolError,
    acquire_immutable_claim,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_json_sha256,
    frozen_file_inventory,
    relative_path,
    resolve_path,
    sha256_file,
    strict_json_dumps,
    validate_frozen_manifest,
)


PROJECT_ID = "510300_DERIVATIVE_PRESSURE_RELEASE_V2_AGGREGATE_OI"
BUNDLE_STATUS = "FROZEN_PRE_OUTCOME_INDEPENDENT_ONE_SHOT"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ProtocolError(f"配置不是对象：{path}")
    if config.get("protocol", {}).get("project_id") != PROJECT_ID:
        raise ProtocolError("项目标识不匹配")
    return config


def _configured_outputs(root: Path, config: Mapping[str, Any]) -> list[Path]:
    artifacts = config["artifacts"]
    return [
        resolve_path(root, artifacts[key])
        for key in (
            "one_shot_claim",
            "one_shot_receipt",
            "data_admission_json",
            "result_json",
            "result_markdown",
            "feature_table",
            "event_table",
        )
    ]


def freeze_bundle(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(config_path.resolve())
    protocol = config["protocol"]
    if protocol.get("state") != "PRE_OUTCOME_INDEPENDENT_FREEZE":
        raise ProtocolError("协议不是独立冻结前状态")
    if int(protocol.get("max_research_runs", 0)) != 1:
        raise ProtocolError("协议没有把研究运行次数锁定为1")
    if protocol.get("relation_to_v1") != "INDEPENDENT_PREFROZEN_DATA_CONSTRUCTION_VERSION":
        raise ProtocolError("V2与V1的关系没有冻结为独立数据构造版本")
    if protocol.get("rescue_v1") is not False:
        raise ProtocolError("V2不得被标记为V1救援")
    if protocol.get("post_result_parameter_change") != "FORBIDDEN":
        raise ProtocolError("协议没有禁止结果后改参")
    if any(bool(value) for value in config["boundaries"].values()):
        raise ProtocolError("交易隔离开关中存在被打开的项目")

    manifest_path = resolve_path(root, config["artifacts"]["freeze_manifest"])
    if manifest_path.exists():
        raise ProtocolError(f"冻结清单已存在，禁止覆盖：{manifest_path}")
    existing_outputs = [relative_path(root, path) for path in _configured_outputs(root, config) if path.exists()]
    if existing_outputs:
        raise ProtocolError(f"冻结前已经存在V2结果或运行占位：{existing_outputs}")

    parent_manifest_path = root / "config/510300_episodic_alpha_library_v1_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("manifest_sha256") != protocol["parent_v1_manifest_content_sha256"]:
        raise ProtocolError("V1冻结清单内容摘要与V2协议记录不一致")
    parent_adjudication_path = root / "reports/research/510300_episodic_alpha_library_v1_post_run_adjudication.json"
    if sha256_file(parent_adjudication_path) != protocol["parent_v1_adjudication_sha256"]:
        raise ProtocolError("V1运行后裁决摘要与V2协议记录不一致")

    entries: list[tuple[str | Path, str]] = [
        (path, "PROTOCOL_IMPLEMENTATION") for path in config["freeze"]["files"]
    ]
    entries.extend(
        (dependency["path"], dependency["role"])
        for dependency in config["immutable_dependencies"]
    )
    if config["freeze"].get("include_all_data_contract_files", False):
        entries.extend(
            (contract["path"], f"DATA_CONTRACT::{contract_id}")
            for contract_id, contract in config["data_contracts"].items()
        )

    manifest: dict[str, Any] = {
        "schema_version": "510300_DERIVATIVE_PRESSURE_RELEASE_V2_FREEZE_MANIFEST_V1",
        "project_id": PROJECT_ID,
        "protocol_version": protocol["version"],
        "bundle_status": BUNDLE_STATUS,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_observed_before_freeze": False,
        "pre_freeze_output_absence_verified": True,
        "max_research_runs": 1,
        "v1_unchanged": True,
        "rescue_v1": False,
        "position_impact": 0,
        "live_trading_authorized": False,
        "frozen_files": frozen_file_inventory(root, entries),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest,
        excluded_keys=("manifest_sha256",),
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "status": BUNDLE_STATUS,
        "manifest_path": relative_path(root, manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_file_count": len(manifest["frozen_files"]),
        "results_observed_before_freeze": False,
        "max_research_runs": 1,
    }


def validate_bundle(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path.resolve())
    manifest_path = resolve_path(root, config["artifacts"]["freeze_manifest"])
    manifest = validate_frozen_manifest(
        root,
        manifest_path,
        expected_manifest_sha256,
        project_id=PROJECT_ID,
        bundle_status=BUNDLE_STATUS,
    )
    if int(manifest.get("max_research_runs", 0)) != 1:
        raise ProtocolError("冻结清单中的最大运行次数不是1")
    return config, manifest


def _normalize_if_inputs(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contracts = config["data_contracts"]
    daily_path = resolve_path(root, contracts["if_contract_daily"]["path"])
    expiry_path = resolve_path(root, contracts["if_contract_expiry"]["path"])
    daily = pd.read_parquet(daily_path)
    expiry = pd.read_parquet(expiry_path)
    required_daily = {"symbol", "date", "close", "open_interest"}
    required_expiry = {"symbol", "expiry_date"}
    if not required_daily.issubset(daily.columns):
        raise ProtocolError(f"IF日度数据缺少字段：{sorted(required_daily - set(daily.columns))}")
    if not required_expiry.issubset(expiry.columns):
        raise ProtocolError(f"IF到期日数据缺少字段：{sorted(required_expiry - set(expiry.columns))}")

    daily = daily.copy()
    expiry = expiry.copy()
    daily["symbol"] = daily["symbol"].astype(str).str.upper().str.strip()
    expiry["symbol"] = expiry["symbol"].astype(str).str.upper().str.strip()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    expiry["expiry_date"] = pd.to_datetime(expiry["expiry_date"], errors="coerce").dt.normalize()
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["open_interest"] = pd.to_numeric(daily["open_interest"], errors="coerce")
    duplicate_rows = int(daily.duplicated(["symbol", "date"], keep=False).sum())
    duplicate_expiry = int(expiry.duplicated("symbol", keep=False).sum())
    if duplicate_rows or duplicate_expiry:
        raise ProtocolError(
            f"IF输入键不唯一：daily_duplicate_rows={duplicate_rows}, expiry_duplicate_rows={duplicate_expiry}"
        )
    daily = daily.dropna(subset=["symbol", "date"])
    expiry = expiry.dropna(subset=["symbol", "expiry_date"])
    merged = daily.merge(
        expiry[["symbol", "expiry_date"]],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    missing_expiry = int(merged["expiry_date"].isna().sum())
    if missing_expiry:
        raise ProtocolError(f"IF日度数据存在{missing_expiry}行缺失合约到期日")
    merged["dte"] = (merged["expiry_date"] - merged["date"]).dt.days
    admission = {
        "if_daily_rows": int(len(merged)),
        "if_contract_count": int(merged["symbol"].nunique()),
        "if_trade_date_count": int(merged["date"].nunique()),
        "if_date_min": merged["date"].min(),
        "if_date_max": merged["date"].max(),
        "daily_duplicate_key_rows": duplicate_rows,
        "expiry_duplicate_key_rows": duplicate_expiry,
        "missing_expiry_rows": missing_expiry,
    }
    return merged, expiry, admission


def aggregate_if_open_interest(if_daily: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "expiry_date", "open_interest"}
    if not required.issubset(if_daily.columns):
        raise ProtocolError(f"聚合持仓输入缺少字段：{sorted(required - set(if_daily.columns))}")
    eligible = if_daily[
        if_daily["expiry_date"].gt(if_daily["date"])
        & if_daily["open_interest"].gt(0)
    ].copy()
    return (
        eligible.groupby("date", as_index=False, observed=True)
        .agg(
            aggregate_open_interest=("open_interest", "sum"),
            aggregate_contract_count=("symbol", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_component_features(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contracts = config["data_contracts"]
    if_daily, expiry, admission = _normalize_if_inputs(root, config)
    spot = pd.read_parquet(resolve_path(root, contracts["csi300_price_index_daily"]["path"]))
    etf = _load_etf_daily(resolve_path(root, contracts["etf_daily"]["path"]))

    aggregate = aggregate_if_open_interest(if_daily)
    aggregate["aggregate_open_interest"] = pd.to_numeric(
        aggregate["aggregate_open_interest"], errors="coerce"
    )

    branch = config["branches"]["derivative_pressure_release"]
    minimum_dte = int(branch["if_front_minimum_calendar_days_to_expiry"])
    front_eligible = if_daily[
        if_daily["dte"].ge(minimum_dte)
        & if_daily["close"].gt(0)
        & if_daily["open_interest"].gt(0)
    ].sort_values(["date", "expiry_date", "symbol"])
    front = front_eligible.groupby("date", as_index=False, observed=True).first()
    front = front.rename(
        columns={
            "symbol": "front_symbol",
            "close": "if_close",
            "open_interest": "front_open_interest",
        }
    )

    required_spot = {"date", "close"}
    if not required_spot.issubset(spot.columns):
        raise ProtocolError(f"沪深300价格指数缺少字段：{sorted(required_spot - set(spot.columns))}")
    spot = spot.copy()
    spot["date"] = pd.to_datetime(spot["date"], errors="coerce").dt.normalize()
    spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
    spot = (
        spot.dropna(subset=["date", "close"])
        .drop_duplicates("date", keep="last")[["date", "close"]]
        .rename(columns={"close": "csi300_close"})
    )

    feature = etf[["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    feature = feature.merge(
        front[["date", "front_symbol", "expiry_date", "dte", "if_close", "front_open_interest"]],
        on="date",
        how="inner",
    )
    feature = feature.merge(spot, on="date", how="inner", validate="one_to_one")
    feature = feature.merge(aggregate, on="date", how="left", validate="one_to_one")
    feature = feature.sort_values("date").reset_index(drop=True)
    feature["aggregate_oi_missing"] = (
        feature["aggregate_open_interest"].isna()
        | feature["aggregate_open_interest"].le(0)
        | feature["aggregate_contract_count"].fillna(0).le(0)
    )
    feature["delta_aggregate_oi"] = np.where(
        ~feature["aggregate_oi_missing"]
        & ~feature["aggregate_oi_missing"].shift(1, fill_value=True),
        np.log(feature["aggregate_open_interest"] / feature["aggregate_open_interest"].shift(1)),
        np.nan,
    )
    feature["if_annualized_basis"] = (
        (feature["if_close"] / feature["csi300_close"] - 1.0) * 365.0 / feature["dte"]
    )
    feature["same_front_contract"] = feature["front_symbol"].eq(feature["front_symbol"].shift(1))
    feature["roll_day"] = ~feature["same_front_contract"]
    expiry_days = set(pd.to_datetime(expiry["expiry_date"], errors="coerce").dropna().dt.normalize())
    feature["expiry_day"] = feature["date"].isin(expiry_days)

    option_panel = _build_option_pressure_panel(root, config, etf)
    feature = feature.merge(option_panel, left_on="date", right_on="trade_date", how="left")
    feature = feature.drop(columns=["trade_date"])
    rolling = config["rolling_rules"]
    window = int(rolling["standard_z_window_days"])
    minimum = int(rolling["standard_z_minimum_history_days"])
    feature["basis_z"] = prior_rolling_z(feature["if_annualized_basis"], window, minimum)
    feature["delta_oi_z"] = prior_rolling_z(feature["delta_aggregate_oi"], window, minimum)
    feature["put_skew_z"] = prior_rolling_z(feature["put_skew"], window, minimum)
    feature["vrp_z"] = prior_rolling_z(feature["vrp"], window, minimum)
    feature["pressure"] = (
        -feature["basis_z"]
        + feature["delta_oi_z"]
        + feature["put_skew_z"]
        + feature["vrp_z"]
    )
    feature["pressure_q95_prior252"] = prior_rolling_quantile(
        feature["pressure"],
        0.95,
        int(rolling["percentile_window_days"]),
        int(rolling["percentile_minimum_history_days"]),
    )
    feature["candidate_preconfirmation"] = pd.Series(pd.NA, index=feature.index, dtype="boolean")

    admission.update(
        {
            "schema_version": "510300_DERIVATIVE_PRESSURE_RELEASE_V2_DATA_ADMISSION_V1",
            "project_id": PROJECT_ID,
            "aggregate_contract_universe": "EXPIRY_DATE_STRICTLY_AFTER_TRADE_DATE_AND_OPEN_INTEREST_GT_ZERO",
            "aggregate_oi_date_count": int(len(aggregate)),
            "aggregate_oi_date_min": aggregate["date"].min() if len(aggregate) else None,
            "aggregate_oi_date_max": aggregate["date"].max() if len(aggregate) else None,
            "aggregate_oi_missing_feature_rows": int(feature["aggregate_oi_missing"].sum()),
            "expiry_day_feature_rows": int(feature["expiry_day"].sum()),
            "feature_rows": int(len(feature)),
            "basis_z_nonnull_rows": int(feature["basis_z"].notna().sum()),
            "delta_aggregate_oi_nonnull_rows": int(feature["delta_aggregate_oi"].notna().sum()),
            "delta_oi_z_nonnull_rows": int(feature["delta_oi_z"].notna().sum()),
            "put_skew_z_nonnull_rows": int(feature["put_skew_z"].notna().sum()),
            "vrp_z_nonnull_rows": int(feature["vrp_z"].notna().sum()),
            "joint_z_component_nonnull_rows": int(
                feature[["basis_z", "delta_oi_z", "put_skew_z", "vrp_z"]].notna().all(axis=1).sum()
            ),
            "pressure_nonnull_rows": int(feature["pressure"].notna().sum()),
            "pressure_q95_nonnull_rows": int(feature["pressure_q95_prior252"].notna().sum()),
            "pressure_percentile_constructed": bool(feature["pressure_q95_prior252"].notna().any()),
            "input_hashes": {
                contract_id: sha256_file(resolve_path(root, contract["path"]))
                for contract_id, contract in contracts.items()
            },
        }
    )
    return feature, admission


EVENT_COLUMNS = [
    "model_id",
    "signal_date",
    "confirmation_date",
    "exit_date",
    "front_symbol",
    "aggregate_open_interest",
    "aggregate_contract_count",
    "delta_aggregate_oi",
    "if_annualized_basis",
    "put_skew",
    "vrp",
    "basis_z",
    "delta_oi_z",
    "put_skew_z",
    "vrp_z",
    "pressure",
    "return_q05_prior252",
    "pressure_q95_prior252",
    "etf_daily_return",
    "expiry_day",
    "basis_weakens",
    "aggregate_oi_increases",
    "candidate_preconfirmation",
    "confirmation_complete",
    "no_continued_new_low",
    "above_first15_vwap",
    "price_impact_declines",
    "opening_repair_ratio",
    "opening_repair_pass",
    "signal",
    "mature_event",
    "entry_price",
    "exit_price",
    "gross_return",
    "stress_net_return",
    "double_stress_net_return",
    "historical_evidence_label",
    "position_impact",
]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def build_candidates_and_events(
    root: Path,
    config: Mapping[str, Any],
    feature: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature = feature.copy()
    rolling = config["rolling_rules"]
    feature["etf_daily_return"] = feature["etf_close"] / feature["etf_close"].shift(1) - 1.0
    feature["return_q05_prior252"] = prior_rolling_quantile(
        feature["etf_daily_return"],
        0.05,
        int(rolling["percentile_window_days"]),
        int(rolling["percentile_minimum_history_days"]),
    )
    feature["basis_weakens"] = (
        feature["same_front_contract"]
        & feature["if_annualized_basis"].lt(feature["if_annualized_basis"].shift(1))
    )
    feature["aggregate_oi_increases"] = feature["delta_aggregate_oi"].gt(0)
    complete_components = feature[["basis_z", "delta_oi_z", "put_skew_z", "vrp_z"]].notna().all(axis=1)
    feature["candidate_preconfirmation"] = (
        ~feature["aggregate_oi_missing"]
        & ~feature["expiry_day"]
        & feature["etf_daily_return"].le(feature["return_q05_prior252"])
        & feature["pressure"].ge(feature["pressure_q95_prior252"])
        & feature["basis_weakens"]
        & feature["aggregate_oi_increases"]
        & complete_components
    ).astype(bool)
    candidates = feature[feature["candidate_preconfirmation"]].copy()
    if candidates.empty:
        return feature, _empty_events()

    bars = _load_minute_bars(resolve_path(root, config["data_contracts"]["etf_minute"]["path"]))
    day_lookup = _minute_day_lookup(bars)
    date_sequence = feature["date"].tolist()
    date_position = {pd.Timestamp(day): index for index, day in enumerate(date_sequence)}
    costs = config["capital_and_costs"]
    repair_minimum = float(config["shared_clocks"]["repair_ratio_minimum"])
    records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        signal_day = pd.Timestamp(row["date"])
        position = date_position[signal_day]
        confirmation_day = pd.Timestamp(date_sequence[position + 1]) if position + 1 < len(date_sequence) else pd.NaT
        exit_day = pd.Timestamp(date_sequence[position + 2]) if position + 2 < len(date_sequence) else pd.NaT
        confirmation = (
            confirmation_from_minutes(day_lookup[confirmation_day])
            if pd.notna(confirmation_day) and confirmation_day in day_lookup
            else confirmation_from_minutes(bars.iloc[0:0].copy())
        )
        next_daily = feature.iloc[position + 1] if position + 1 < len(feature) else None
        next_open = float(next_daily["etf_open"]) if next_daily is not None else np.nan
        prior_close = float(row["etf_close"])
        entry_price = float(confirmation["entry_price_0945"])
        denominator = prior_close - next_open if np.isfinite(next_open) else np.nan
        if np.isfinite(denominator) and denominator > 0 and np.isfinite(entry_price):
            repair_ratio = (entry_price - next_open) / denominator
            repair_pass = repair_ratio >= repair_minimum
        elif np.isfinite(denominator) and denominator <= 0:
            repair_ratio = 1.0
            repair_pass = True
        else:
            repair_ratio = np.nan
            repair_pass = False
        exit_price = _entry_open(day_lookup, exit_day, "09:45:00") if pd.notna(exit_day) else np.nan
        confirmed = bool(
            confirmation["confirmation_complete"]
            and confirmation["no_continued_new_low"]
            and confirmation["above_first15_vwap"]
            and repair_pass
            and confirmation["price_impact_declines"]
        )
        mature = bool(
            confirmed
            and np.isfinite(entry_price)
            and np.isfinite(exit_price)
            and entry_price > 0
            and exit_price > 0
        )
        gross = exit_price / entry_price - 1.0 if mature else np.nan
        stress_net = _stress_net_return(
            gross,
            float(costs["commission_rate_per_leg"]),
            float(costs["stress_slippage_bps_per_leg"]),
        )
        double_stress_net = _stress_net_return(
            gross,
            float(costs["commission_rate_per_leg"]),
            float(costs["double_stress_slippage_bps_per_leg"]),
        )
        record = {
            "model_id": PROJECT_ID,
            "signal_date": signal_day,
            "confirmation_date": confirmation_day,
            "exit_date": exit_day,
            "front_symbol": row["front_symbol"],
            "aggregate_open_interest": row["aggregate_open_interest"],
            "aggregate_contract_count": row["aggregate_contract_count"],
            "delta_aggregate_oi": row["delta_aggregate_oi"],
            "if_annualized_basis": row["if_annualized_basis"],
            "put_skew": row["put_skew"],
            "vrp": row["vrp"],
            "basis_z": row["basis_z"],
            "delta_oi_z": row["delta_oi_z"],
            "put_skew_z": row["put_skew_z"],
            "vrp_z": row["vrp_z"],
            "pressure": row["pressure"],
            "return_q05_prior252": row["return_q05_prior252"],
            "pressure_q95_prior252": row["pressure_q95_prior252"],
            "etf_daily_return": row["etf_daily_return"],
            "expiry_day": bool(row["expiry_day"]),
            "basis_weakens": bool(row["basis_weakens"]),
            "aggregate_oi_increases": bool(row["aggregate_oi_increases"]),
            "candidate_preconfirmation": True,
            **confirmation,
            "opening_repair_ratio": repair_ratio,
            "opening_repair_pass": bool(repair_pass),
            "signal": confirmed,
            "mature_event": mature,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross,
            "stress_net_return": stress_net,
            "double_stress_net_return": double_stress_net,
            "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
            "position_impact": 0.0,
        }
        records.append({column: record.get(column) for column in EVENT_COLUMNS})
    return feature, pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)


def _render_markdown(result: Mapping[str, Any]) -> str:
    counts = result["feature_counts"]
    metrics = result["metrics"]
    lines = [
        "# 510300 衍生品压力释放 V2 聚合 OI：一次性权威结果",
        "",
        f"- 科学状态：`{result['scientific_status']}`",
        f"- 资源状态：`{result['resource_allocation_status']}`",
        f"- 收益评价：`{result['return_evaluation']}`",
        f"- 运行次数：`{result['research_run_number']}/1`",
        f"- V1 状态：`{result['v1_status']}`（未修改、未救援）",
        f"- 当前仓位影响：`{result['position_impact']}`",
        "",
        "## 特征构造",
        "",
        f"- 聚合 OI 非空行：{counts['aggregate_oi_nonnull_rows']}",
        f"- 聚合 OI 变化非空行：{counts['delta_aggregate_oi_nonnull_rows']}",
        f"- 聚合 OI z 非空行：{counts['delta_oi_z_nonnull_rows']}",
        f"- 四分量联合 z 非空行：{counts['joint_z_component_nonnull_rows']}",
        f"- Pressure 非空行：{counts['pressure_nonnull_rows']}",
        f"- Pressure 历史 95% 门非空行：{counts['pressure_q95_nonnull_rows']}",
        f"- 到期日硬禁新信号行：{counts['expiry_day_rows']}",
        "",
        "## 事件结果",
        "",
        f"- 候选事件：{metrics.get('candidate_preconfirmation_count', 0)}",
        f"- 确认信号：{metrics.get('confirmed_signal_count', 0)}",
        f"- 成熟事件：{metrics.get('mature_event_count', 0)}",
        f"- 压力成本后均值（bp）：{metrics.get('mean_stress_net_bps')}",
        f"- 单侧 95% 净均值下界（bp）：{metrics.get('stress_net_mean_lcb_bps')}",
        f"- 保守净 Sharpe：{metrics.get('conservative_net_sharpe')}",
        "",
        "## 最终边界",
        "",
        result["decision_reason"],
        "",
        "本结果不产生信号、仓位、订单或券商连接。若资源状态已关闭，则不得以近月+次月、成交量权重或更短窗口继续建立后续版本。",
        "",
    ]
    return "\n".join(lines)


def run_one_shot(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_bundle(root, config_path, expected_manifest_sha256)
    artifacts = config["artifacts"]
    claim_path = resolve_path(root, artifacts["one_shot_claim"])
    receipt_path = resolve_path(root, artifacts["one_shot_receipt"])
    if receipt_path.exists():
        raise ProtocolError("一次性V2运行回执已经存在，禁止再次运行")
    claim = {
        "schema_version": "510300_DERIVATIVE_PRESSURE_RELEASE_V2_RUN_CLAIM_V1",
        "project_id": PROJECT_ID,
        "status": "ONE_SHOT_RESEARCH_RUN_CLAIMED",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_run_number": 1,
        "maximum_research_runs": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "position_impact": 0,
    }
    acquire_immutable_claim(claim_path, claim)

    try:
        feature, admission = build_component_features(root, config)
        pressure_percentile_constructed = bool(admission["pressure_percentile_constructed"])
        if pressure_percentile_constructed:
            feature, events = build_candidates_and_events(root, config, feature)
            metrics = event_metrics(events, config)
            scientific_status = historical_screen_decision(metrics, config)
            return_evaluation = (
                "READ_FOR_MATURE_CONFIRMED_EVENTS_ONLY"
                if int(metrics["mature_event_count"]) > 0
                else "NOT_READ_NO_MATURE_CONFIRMED_EVENTS"
            )
            if scientific_status == config["evaluation"]["result_if_all_historical_gates_pass"]:
                resource_status = config["evaluation"]["resource_status_if_historical_pass"]
                decision_reason = "历史发现筛查通过全部冻结门，但只允许零仓位前向 Shadow，不构成交易授权。"
            else:
                resource_status = config["evaluation"]["resource_status_if_not_historical_pass"]
                decision_reason = "一次性 V2 未通过全部冻结门，公开日频衍生品压力代理资源族关闭，不得继续创建 V3 或事后改参。"
        else:
            events = _empty_events()
            scientific_status = config["evaluation"]["result_if_feature_construction_fails"]
            resource_status = config["evaluation"]["resource_status_if_not_historical_pass"]
            return_evaluation = "NOT_ALLOWED_PRESSURE_PERCENTILE_NOT_CONSTRUCTED"
            metrics = {
                "candidate_preconfirmation_count": 0,
                "confirmed_signal_count": 0,
                "mature_event_count": 0,
                "mean_gross_bps": None,
                "mean_stress_net_bps": None,
                "stress_net_mean_lcb_bps": None,
                "conservative_net_sharpe": None,
                "historical_active_gate_diagnostics_pass": False,
                "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
                "forward_mature_event_count": 0,
            }
            decision_reason = (
                "聚合 OI 修复后仍不能形成冻结要求的 prior-252 Pressure 分位门；未读取策略入场/退出收益。"
                "按一次性 V2 资源规则，公开日频衍生品压力代理资源族关闭。"
            )

        feature_counts = {
            "feature_rows": int(len(feature)),
            "aggregate_oi_nonnull_rows": int(feature["aggregate_open_interest"].notna().sum()),
            "aggregate_oi_missing_rows": int(feature["aggregate_oi_missing"].sum()),
            "delta_aggregate_oi_nonnull_rows": int(feature["delta_aggregate_oi"].notna().sum()),
            "basis_z_nonnull_rows": int(feature["basis_z"].notna().sum()),
            "delta_oi_z_nonnull_rows": int(feature["delta_oi_z"].notna().sum()),
            "put_skew_z_nonnull_rows": int(feature["put_skew_z"].notna().sum()),
            "vrp_z_nonnull_rows": int(feature["vrp_z"].notna().sum()),
            "joint_z_component_nonnull_rows": int(
                feature[["basis_z", "delta_oi_z", "put_skew_z", "vrp_z"]].notna().all(axis=1).sum()
            ),
            "pressure_nonnull_rows": int(feature["pressure"].notna().sum()),
            "pressure_q95_nonnull_rows": int(feature["pressure_q95_prior252"].notna().sum()),
            "expiry_day_rows": int(feature["expiry_day"].sum()),
        }
        result: dict[str, Any] = {
            "schema_version": "510300_DERIVATIVE_PRESSURE_RELEASE_V2_RESULT_V1",
            "project_id": PROJECT_ID,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": scientific_status,
            "resource_allocation_status": resource_status,
            "return_evaluation": return_evaluation,
            "research_run_number": 1,
            "maximum_research_runs": 1,
            "post_result_parameter_change": "FORBIDDEN",
            "v3_or_later": "FORBIDDEN_IF_NOT_HISTORICAL_PASS",
            "v1_status": config["protocol"]["v1_status"],
            "v1_unchanged": True,
            "rescue_v1": False,
            "manifest_sha256": manifest["manifest_sha256"],
            "feature_counts": feature_counts,
            "metrics": metrics,
            "decision_reason": decision_reason,
            "current_validated_high_sharpe_strategy": "NONE",
            "current_holding_route": "CASH_CNY",
            "position_impact": 0,
            "boundaries": config["boundaries"],
        }

        admission_path = resolve_path(root, artifacts["data_admission_json"])
        feature_path = resolve_path(root, artifacts["feature_table"])
        event_path = resolve_path(root, artifacts["event_table"])
        result_path = resolve_path(root, artifacts["result_json"])
        markdown_path = resolve_path(root, artifacts["result_markdown"])
        atomic_write_json(admission_path, admission)
        atomic_write_parquet(feature_path, feature)
        atomic_write_parquet(event_path, events)
        atomic_write_json(result_path, result)
        atomic_write_text(markdown_path, _render_markdown(result))

        output_hashes = {
            relative_path(root, path): sha256_file(path)
            for path in (admission_path, feature_path, event_path, result_path, markdown_path)
        }
        receipt = {
            "schema_version": "510300_DERIVATIVE_PRESSURE_RELEASE_V2_RUN_RECEIPT_V1",
            "project_id": PROJECT_ID,
            "status": "ONE_SHOT_RESEARCH_RUN_COMPLETED_AND_CONSUMED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_run_number": 1,
            "maximum_research_runs": 1,
            "manifest_sha256": manifest["manifest_sha256"],
            "scientific_status": scientific_status,
            "resource_allocation_status": resource_status,
            "return_evaluation": return_evaluation,
            "output_hashes": output_hashes,
            "position_impact": 0,
            "live_trading_authorized": False,
        }
        atomic_write_json(receipt_path, receipt)
        return {
            "status_summary": {
                "scientific_status": scientific_status,
                "resource_allocation_status": resource_status,
                "return_evaluation": return_evaluation,
                "research_runs_consumed": 1,
                "maximum_research_runs": 1,
                "pressure_nonnull_rows": feature_counts["pressure_nonnull_rows"],
                "pressure_q95_nonnull_rows": feature_counts["pressure_q95_nonnull_rows"],
                "candidate_preconfirmation_count": metrics.get("candidate_preconfirmation_count", 0),
                "mature_event_count": metrics.get("mature_event_count", 0),
                "position_impact": 0,
            },
            "result": result,
            "receipt": receipt,
        }
    except Exception as exc:
        if not receipt_path.exists():
            failure_receipt = {
                "schema_version": "510300_DERIVATIVE_PRESSURE_RELEASE_V2_RUN_RECEIPT_V1",
                "project_id": PROJECT_ID,
                "status": "ONE_SHOT_RESEARCH_RUN_FAILED_AND_CONSUMED",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "research_run_number": 1,
                "maximum_research_runs": 1,
                "manifest_sha256": manifest["manifest_sha256"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "rerun_allowed": False,
                "position_impact": 0,
            }
            atomic_write_json(receipt_path, failure_receipt)
        raise
