"""沪深300点时估值 V2 可重建性审计。

本模块只审计输入、点时语义、历史覆盖与 R5 信号工程重放；
不计算未来收益、IC、策略收益，不映射任何真实或纸面仓位。
"""

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

from backtest.defensive_valuation_timing_engine import (
    build_model_positions,
    build_timing_features,
    schedule_asymmetric_execution,
)
from backtest.valuation_fvg_engine import attach_market_cap_context, build_valuation_signals
from research.build_point_in_time_fundamental_panel import (
    derive_company_metrics,
    select_latest_vintages,
)
from research.diagnose_valuation_negative_ic import derive_company_normalization


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "point_in_time_valuation_v2_audit.yaml"

CORE_FINANCIAL_COLUMNS = (
    "revenue_cny",
    "net_profit_parent_cny",
    "equity_parent_cny",
    "total_shares",
    "bps",
)

R5_NUMERIC_FIELDS = (
    "raw_continuous_position",
    "strategic_position",
    "trend_score",
    "risk_penalty",
    "continuous_model_position",
    "discrete_model_position",
    "target_position",
)

R5_BOOLEAN_FIELDS = ("trade_allowed", "risk_off_override")


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_directory_hash(directory: Path) -> tuple[str, list[dict[str, Any]]]:
    """用相对路径和逐文件哈希生成目录内容指纹。"""

    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.parquet"), key=lambda item: item.as_posix()):
        records.append(
            {
                "file": path.relative_to(directory).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), records


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """读取审计配置并验证禁止项。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    forbidden = (
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    )
    enabled = [field for field in forbidden if bool(protocol.get(field))]
    if enabled:
        raise ValueError(f"审计配置错误，以下禁止项被启用：{enabled}")
    return config


def _safe_date(value: Any) -> str | None:
    timestamp = pd.Timestamp(value) if pd.notna(value) else pd.NaT
    return str(timestamp.date()) if pd.notna(timestamp) else None


def _safe_number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _sanitize(value: Any) -> Any:
    """把 pandas/numpy 类型转换为严格 JSON 可序列化值。"""

    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def audit_hashes(config: dict[str, Any]) -> dict[str, Any]:
    """逐项校验配置冻结的输入哈希。"""

    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        if "file" not in contract:
            continue
        relative = contract["file"]
        path = ROOT / relative
        actual = sha256_file(path) if path.exists() else None
        expected = contract.get("sha256")
        rows.append(
            {
                "dataset": name,
                "file": relative,
                "exists": path.exists(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": bool(actual == expected) if expected else path.exists(),
            }
        )
    return {
        "status": "PASS" if rows and all(row["matches"] for row in rows) else "BLOCKED_HASH_DRIFT",
        "checked_file_count": len(rows),
        "rows": rows,
    }


def audit_weights(weights: pd.DataFrame) -> dict[str, Any]:
    """审计历史月度权重的模式、唯一键和范围。"""

    required = {"index_code", "con_code", "trade_date", "weight", "source", "retrieved_at"}
    if missing := required.difference(weights.columns):
        raise ValueError(f"历史权重缺少字段：{sorted(missing)}")
    data = weights.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    counts = data.groupby("trade_date")["con_code"].nunique()
    sums = data.groupby("trade_date")["weight"].sum(min_count=1)
    duplicate_count = int(data.duplicated(["trade_date", "con_code"]).sum())
    status = "PASS" if all(
        [
            duplicate_count == 0,
            not data[list(required)].isna().any().any(),
            counts.eq(300).all(),
            sums.between(99.0, 101.0).all(),
        ]
    ) else "BLOCKED"
    return {
        "status": status,
        "rows": int(len(data)),
        "snapshot_count": int(data["trade_date"].nunique()),
        "first_date": _safe_date(data["trade_date"].min()),
        "last_date": _safe_date(data["trade_date"].max()),
        "unique_constituent_count": int(data["con_code"].nunique()),
        "minimum_constituents": int(counts.min()),
        "maximum_constituents": int(counts.max()),
        "minimum_weight_sum_pct": float(sums.min()),
        "maximum_weight_sum_pct": float(sums.max()),
        "duplicate_key_count": duplicate_count,
        "sources": sorted(data["source"].dropna().astype(str).unique().tolist()),
        "retrieved_at_min": str(data["retrieved_at"].min()),
        "retrieved_at_max": str(data["retrieved_at"].max()),
        "unit": "percent",
        "point_in_time_semantics": "权重仅从trade_date收盘数据可得后使用；retrieved_at不等于生效日",
    }


def audit_financials(
    financials: pd.DataFrame,
    checkpoint_contract: dict[str, Any],
) -> dict[str, Any]:
    """审计财务事件表的可得日、版本和原始季度检查点。"""

    required = {
        "con_code",
        "report_period",
        "available_at",
        "announcement_date",
        "source",
        "retrieved_at",
        *CORE_FINANCIAL_COLUMNS,
    }
    if missing := required.difference(financials.columns):
        raise ValueError(f"财务事件表缺少字段：{sorted(missing)}")
    data = financials.copy()
    for column in ("report_period", "available_at", "announcement_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    keys = ["con_code", "report_period", "available_at"]
    duplicate_count = int(data.duplicated(keys).sum())
    version_counts = data.groupby(["con_code", "report_period"])["available_at"].nunique()
    multi_version_pairs = version_counts[version_counts.gt(1)].index
    multi = data.set_index(["con_code", "report_period"]).loc[multi_version_pairs].reset_index()
    changed_pair_count = 0
    for _, group in multi.groupby(["con_code", "report_period"], sort=False):
        if any(group[column].nunique(dropna=True) > 1 for column in CORE_FINANCIAL_COLUMNS):
            changed_pair_count += 1

    checkpoint_dir = ROOT / checkpoint_contract["directory"]
    directory_hash, checkpoint_files = canonical_directory_hash(checkpoint_dir)
    per_api: dict[str, Any] = {}
    for api_name in checkpoint_contract["expected_api_names"]:
        files = [row for row in checkpoint_files if row["file"].startswith(f"{api_name}/")]
        periods = [Path(row["file"]).stem for row in files]
        per_api[api_name] = {
            "file_count": len(files),
            "first_period": min(periods) if periods else None,
            "last_period": max(periods) if periods else None,
            "expected_file_count": int(checkpoint_contract["expected_quarter_count_per_api"]),
            "count_matches": len(files) == int(checkpoint_contract["expected_quarter_count_per_api"]),
        }
    core_coverage = {
        column: float(data[column].notna().mean()) for column in CORE_FINANCIAL_COLUMNS
    }
    announcement_after_available = int((data["announcement_date"] > data["available_at"]).sum())
    available_before_period = int((data["available_at"] < data["report_period"]).sum())
    delay_days = (data["available_at"] - data["report_period"]).dt.days
    checkpoint_pass = (
        len(checkpoint_files) == int(checkpoint_contract["expected_file_count"])
        and all(item["count_matches"] for item in per_api.values())
    )
    structural_pass = all(
        [
            duplicate_count == 0,
            not data[["con_code", "report_period", "available_at", "announcement_date"]].isna().any().any(),
            announcement_after_available == 0,
            available_before_period == 0,
            checkpoint_pass,
        ]
    )
    return {
        "status": "PASS_PIT_RECONSTRUCTION_WITH_CURRENT_ARCHIVE_CAVEAT" if structural_pass else "BLOCKED",
        "rows": int(len(data)),
        "symbol_count": int(data["con_code"].nunique()),
        "report_period_count": int(data["report_period"].nunique()),
        "first_report_period": _safe_date(data["report_period"].min()),
        "last_report_period": _safe_date(data["report_period"].max()),
        "first_available_at": _safe_date(data["available_at"].min()),
        "last_available_at": _safe_date(data["available_at"].max()),
        "duplicate_event_key_count": duplicate_count,
        "multi_version_company_report_pairs": int(len(multi_version_pairs)),
        "multi_version_pairs_with_value_changes": int(changed_pair_count),
        "announcement_after_available_count": announcement_after_available,
        "available_before_report_period_count": available_before_period,
        "availability_delay_days": {
            "median": _safe_number(delay_days.median()),
            "p95": _safe_number(delay_days.quantile(0.95)),
            "maximum": _safe_number(delay_days.max()),
        },
        "field_non_null_ratio": core_coverage,
        "sources": sorted(data["source"].dropna().astype(str).unique().tolist()),
        "retrieved_at_min": str(data["retrieved_at"].min()),
        "retrieved_at_max": str(data["retrieved_at"].max()),
        "checkpoint_directory": checkpoint_contract["directory"],
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_expected_file_count": int(checkpoint_contract["expected_file_count"]),
        "checkpoint_content_sha256": directory_hash,
        "checkpoint_by_api": per_api,
        "units": {
            "revenue_cny": "CNY",
            "net_profit_parent_cny": "CNY",
            "equity_parent_cny": "CNY",
            "total_shares": "shares",
            "bps": "CNY_per_share",
        },
        "archive_caveat": "原始季度表于2026-08-13统一回取；available_at可重建当时可得信息，但未取得逐日不可变供应商快照来独立证明历史值从未被供应商修订。",
    }


def audit_constituent_panel(data: pd.DataFrame, label: str) -> dict[str, Any]:
    """审计成分股日线价格与名为总市值字段的实际覆盖。"""

    required = {
        "date",
        "con_code",
        "raw_close",
        "is_index_member",
        "total_market_cap_cny",
    }
    if missing := required.difference(data.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")
    panel = data.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    members = panel.loc[panel["is_index_member"].fillna(False).astype(bool)].copy()
    counts = members.groupby("date")["con_code"].nunique()
    cap = pd.to_numeric(members["total_market_cap_cny"], errors="coerce")
    raw_close = pd.to_numeric(members["raw_close"], errors="coerce")
    return {
        "label": label,
        "rows": int(len(members)),
        "trading_day_count": int(members["date"].nunique()),
        "symbol_count": int(members["con_code"].nunique()),
        "first_date": _safe_date(members["date"].min()),
        "last_date": _safe_date(members["date"].max()),
        "minimum_members_per_day": int(counts.min()),
        "maximum_members_per_day": int(counts.max()),
        "raw_close_non_null_ratio": float(raw_close.notna().mean()),
        "market_cap_non_null_ratio": float(cap.notna().mean()),
        "market_cap_positive_ratio": float(cap.gt(0).mean()),
        "market_cap_all_null_day_count": int(
            members.assign(_cap=cap).groupby("date")["_cap"].apply(lambda values: values.isna().all()).sum()
        ),
    }


def _weighted_coverage(
    snapshot_weights: pd.DataFrame,
    valid_codes: set[str],
) -> float:
    weights = snapshot_weights[["con_code", "weight"]].copy()
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce") / 100.0
    return float(weights.loc[weights["con_code"].astype(str).isin(valid_codes), "weight"].sum())


def build_snapshot_coverage(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    index_daily: pd.DataFrame,
    bonds: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """逐官方月度快照审计价格、TTM、标准化盈利和国债覆盖。"""

    weight_data = weights.copy()
    weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"], errors="coerce")
    financial_data = financials.copy()
    financial_data["report_period"] = pd.to_datetime(financial_data["report_period"], errors="coerce")
    financial_data["available_at"] = pd.to_datetime(financial_data["available_at"], errors="coerce")
    price_data = prices.copy()
    price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce")
    price_data["raw_close"] = pd.to_numeric(price_data["raw_close"], errors="coerce")
    index_data = index_daily.copy()
    index_data["date"] = pd.to_datetime(index_data["date"], errors="coerce")
    bond_data = bonds.copy()
    bond_data["date"] = pd.to_datetime(bond_data["date"], errors="coerce")
    bond_value_column = "cgb_10y"
    if bond_value_column not in bond_data:
        raise ValueError("国债收益率数据缺少cgb_10y")

    index_dates = set(index_data.loc[index_data["close"].notna(), "date"])
    bond_dates = set(bond_data.loc[pd.to_numeric(bond_data[bond_value_column], errors="coerce").notna(), "date"])
    evaluation = pd.Timestamp(config["protocol"]["evaluation_start"])
    five = config["history_requirements"]["five_year"]
    seven = config["history_requirements"]["seven_year"]
    five_start = pd.Period(five["required_first_month"], freq="M")
    five_end = pd.Period(five["required_last_month_before_evaluation"], freq="M")
    seven_start = pd.Period(seven["required_first_month"], freq="M")
    seven_end = pd.Period(seven["required_last_month_before_evaluation"], freq="M")
    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weight_data.groupby("trade_date", sort=True):
        snapshot_date = pd.Timestamp(snapshot_date)
        symbols = set(snapshot_weights["con_code"].astype(str))
        known = select_latest_vintages(
            financial_data.loc[financial_data["con_code"].astype(str).isin(symbols)],
            snapshot_date,
        )
        ttm = derive_company_metrics(known)
        normalized = derive_company_normalization(known)
        exact_prices = price_data.loc[
            price_data["date"].eq(snapshot_date)
            & price_data["con_code"].astype(str).isin(symbols)
            & price_data["raw_close"].gt(0),
            ["con_code", "raw_close"],
        ].drop_duplicates("con_code", keep="last")
        price_codes = set(exact_prices["con_code"].astype(str))
        ttm_codes = set(ttm.loc[ttm["ttm_eps"].notna(), "con_code"].astype(str)) if not ttm.empty else set()
        normalized_codes = (
            set(normalized.loc[normalized["normalized_eps"].notna(), "con_code"].astype(str))
            if not normalized.empty
            else set()
        )
        period = snapshot_date.to_period("M")
        rows.append(
            {
                "date": snapshot_date,
                "calendar_month": str(period),
                "constituent_count": int(snapshot_weights["con_code"].nunique()),
                "weight_sum": float(pd.to_numeric(snapshot_weights["weight"], errors="coerce").sum() / 100.0),
                "price_weight_coverage": _weighted_coverage(snapshot_weights, price_codes),
                "ttm_metric_weight_coverage": _weighted_coverage(snapshot_weights, ttm_codes),
                "normalized_metric_weight_coverage": _weighted_coverage(snapshot_weights, normalized_codes),
                "raw_ey_ready_weight_coverage": _weighted_coverage(snapshot_weights, price_codes & ttm_codes),
                "normalized_ey_ready_weight_coverage": _weighted_coverage(
                    snapshot_weights, price_codes & normalized_codes
                ),
                "financial_known_event_count": int(len(known)),
                "ttm_company_count": int(len(ttm_codes)),
                "normalization_company_count": int(len(normalized_codes)),
                "index_close_available": snapshot_date in index_dates,
                "cgb_10y_available_exact_date": snapshot_date in bond_dates,
                "before_first_evaluation": snapshot_date < evaluation,
                "in_five_year_preevaluation_window": five_start <= period <= five_end,
                "in_seven_year_preevaluation_window": seven_start <= period <= seven_end,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_history_gate(
    coverage: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """形成原始EY、标准化EY和利差模型的历史分支闸门。"""

    requirements = config["history_requirements"]
    price_min = float(requirements["minimum_price_weight_coverage"])
    ttm_min = float(requirements["minimum_ttm_earnings_weight_coverage"])
    normalized_min = float(requirements["minimum_normalized_earnings_weight_coverage"])

    def window(label: str) -> dict[str, Any]:
        item = requirements[label]
        start = pd.Period(item["required_first_month"], freq="M")
        end = pd.Period(item["required_last_month_before_evaluation"], freq="M")
        required_months = [str(period) for period in pd.period_range(start, end, freq="M")]
        frame = coverage.loc[coverage["calendar_month"].isin(required_months)].copy()
        present = set(frame["calendar_month"])
        missing_weight_months = [month for month in required_months if month not in present]
        price_failed = frame.loc[frame["price_weight_coverage"].lt(price_min), "calendar_month"].tolist()
        ttm_failed = frame.loc[frame["ttm_metric_weight_coverage"].lt(ttm_min), "calendar_month"].tolist()
        normalized_failed = frame.loc[
            frame["normalized_metric_weight_coverage"].lt(normalized_min), "calendar_month"
        ].tolist()
        bond_failed = frame.loc[~frame["cgb_10y_available_exact_date"], "calendar_month"].tolist()
        raw_ready = not missing_weight_months and not price_failed and not ttm_failed
        normalized_ready = raw_ready and not normalized_failed
        spread_ready = normalized_ready and not bond_failed
        return {
            "required_snapshot_count": int(item["required_monthly_snapshots_before_evaluation"]),
            "local_weight_snapshot_count": int(len(frame)),
            "missing_weight_months": missing_weight_months,
            "price_coverage_failed_months": price_failed,
            "ttm_coverage_failed_months": ttm_failed,
            "normalized_coverage_failed_months": normalized_failed,
            "cgb_10y_exact_date_failed_months": bond_failed,
            "raw_ey_status": "PASS_LOCAL_RECONSTRUCTIBLE" if raw_ready else "PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED",
            "normalized_ey_status": "PASS_LOCAL_RECONSTRUCTIBLE" if normalized_ready else "PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED",
            "normalized_ey_spread_status": "PASS_LOCAL_RECONSTRUCTIBLE" if spread_ready else "PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED",
            "minimum_local_coverages": {
                column: _safe_number(frame[column].min())
                for column in (
                    "price_weight_coverage",
                    "ttm_metric_weight_coverage",
                    "normalized_metric_weight_coverage",
                    "raw_ey_ready_weight_coverage",
                    "normalized_ey_ready_weight_coverage",
                )
            },
        }

    five = window("five_year")
    seven = window("seven_year")
    five["raw_ey_status"] = (
        five["raw_ey_status"]
        if not five["missing_weight_months"]
        else "BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED"
    )
    for field in ("raw_ey_status", "normalized_ey_status", "normalized_ey_spread_status"):
        if seven["missing_weight_months"]:
            seven[field] = "BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED"
    local_expanding = coverage.loc[coverage["before_first_evaluation"]].copy()
    expanding = {
        "local_snapshot_count_before_evaluation": int(len(local_expanding)),
        "first_local_snapshot": _safe_date(local_expanding["date"].min()),
        "last_local_snapshot_before_evaluation": _safe_date(local_expanding["date"].max()),
        "raw_ey_ready_snapshot_count": int(
            (
                local_expanding["raw_ey_ready_weight_coverage"].ge(min(price_min, ttm_min))
            ).sum()
        ),
        "normalized_ey_ready_snapshot_count": int(
            local_expanding["normalized_ey_ready_weight_coverage"].ge(min(price_min, normalized_min)).sum()
        ),
        "status": "PARTIALLY_RECONSTRUCTIBLE_ACQUISITION_REQUIRED",
        "reason": "扩展窗口起点已由最早官方权重固定，但早期价格与标准化盈利覆盖尚未通过，禁止从较晚的有利日期重新选择起点。",
    }
    return {"five_year": five, "seven_year": seven, "expanding": expanding}


def audit_cross_checks(
    vendor: pd.DataFrame,
    official: pd.DataFrame,
    existing_panel: pd.DataFrame,
) -> dict[str, Any]:
    """核对官方PE、第三方PE与既有成分重建的同向性。"""

    value = vendor[["date", "pe_ttm", "pb"]].copy()
    value["date"] = pd.to_datetime(value["date"], errors="coerce")
    official_pe = official[["date", "pe_official"]].copy()
    official_pe["date"] = pd.to_datetime(official_pe["date"], errors="coerce")
    daily = official_pe.merge(value[["date", "pe_ttm"]], on="date", how="inner")
    daily["relative_error_vendor_vs_official"] = daily["pe_ttm"] / daily["pe_official"] - 1.0
    panel = existing_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    component_column = "total_market_cap_pe_ttm_profitable_only"
    monthly = panel[["date", component_column]].merge(official_pe, on="date", how="inner")
    monthly["relative_error_component_vs_official"] = monthly[component_column] / monthly["pe_official"] - 1.0
    return {
        "status": "PASS_CROSS_CHECK_ONLY_NON_VINTAGE",
        "official_vs_vendor": {
            "overlap_rows": int(len(daily)),
            "first_date": _safe_date(daily["date"].min()),
            "last_date": _safe_date(daily["date"].max()),
            "pearson_correlation": _safe_number(daily[["pe_official", "pe_ttm"]].corr().iloc[0, 1]),
            "median_absolute_relative_error": _safe_number(
                daily["relative_error_vendor_vs_official"].abs().median()
            ),
        },
        "official_vs_existing_component_panel": {
            "overlap_rows": int(len(monthly)),
            "pearson_correlation": _safe_number(
                monthly[["pe_official", component_column]].corr().iloc[0, 1]
            ),
            "median_absolute_relative_error": _safe_number(
                monthly["relative_error_component_vs_official"].abs().median()
            ),
        },
        "governance": "三条序列均为2026年事后回取或事后重建，只能证明口径大体一致，不能把交叉相关性当作历史点时版本证明。",
    }


def audit_stale_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """识别仍引用旧输入哈希或旧行数的状态文件。"""

    contracts = config["data_contracts"]
    current_hash = sha256_file(ROOT / contracts["current_constituent_daily"]["file"])
    vendor_hash = sha256_file(ROOT / contracts["vendor_valuation"]["file"])
    constituent_status = json.loads(
        (ROOT / contracts["stale_constituent_status"]["file"]).read_text(encoding="utf-8")
    )
    monthly_status = json.loads(
        (ROOT / contracts["existing_monthly_status"]["file"]).read_text(encoding="utf-8")
    )
    vendor_metadata = json.loads(
        (ROOT / contracts["stale_vendor_metadata"]["file"]).read_text(encoding="utf-8")
    )
    monthly_constituent_hash = monthly_status.get("inputs", {}).get(
        "data/raw/constituents/000300_constituent_daily.parquet"
    )
    rows = [
        {
            "artifact": contracts["stale_constituent_status"]["file"],
            "recorded_input_or_output_hash": constituent_status.get("output_sha256"),
            "current_target_hash": current_hash,
            "stale": constituent_status.get("output_sha256") != current_hash,
            "meaning": "状态文件描述的是覆盖了流通市值的旧新浪版本；当前同路径已被Tushare价格面板替换。",
        },
        {
            "artifact": contracts["existing_monthly_status"]["file"],
            "recorded_input_or_output_hash": monthly_constituent_hash,
            "current_target_hash": current_hash,
            "stale": monthly_constituent_hash != current_hash,
            "meaning": "既有60月面板仍可由保留备份解释，但无法用当前主路径原样复算。",
        },
        {
            "artifact": contracts["stale_vendor_metadata"]["file"],
            "recorded_input_or_output_hash": vendor_metadata.get("sha256"),
            "current_target_hash": vendor_hash,
            "stale": vendor_metadata.get("sha256") != vendor_hash,
            "meaning": "估值Parquet已扩展历史，但metadata仍记录旧行数、旧日期范围和旧哈希。",
        },
    ]
    return {
        "status": "STALE_METADATA_DETECTED" if any(row["stale"] for row in rows) else "PASS",
        "rows": rows,
        "mutation_performed": False,
    }


def audit_r5_replay(
    vendor: pd.DataFrame,
    backup_constituents: pd.DataFrame,
    current_constituents: pd.DataFrame,
    index_daily: pd.DataFrame,
    preserved: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """重放R5信号并证明聚合市值的代数抵消，不计算策略收益。"""

    r5_config_path = ROOT / config["data_contracts"]["r5_config"]["file"]
    r5_config = yaml.safe_load(r5_config_path.read_text(encoding="utf-8"))
    valuation = attach_market_cap_context(
        build_valuation_signals(vendor, r5_config["valuation"]), backup_constituents
    )
    timing = build_timing_features(index_daily, r5_config)
    model = build_model_positions(timing, r5_config, valuation)
    replay = schedule_asymmetric_execution(model, r5_config)
    start = pd.Timestamp(r5_config["backtest"]["start_date"])
    end = pd.Timestamp(r5_config["backtest"]["end_date"])
    replay = replay.loc[replay["date"].between(start, end)].copy()
    saved = preserved.copy()
    saved["date"] = pd.to_datetime(saved["date"], errors="coerce")
    saved = saved.loc[saved["date"].between(start, end)].copy()
    compare = saved[["date", *R5_NUMERIC_FIELDS, *R5_BOOLEAN_FIELDS]].merge(
        replay[["date", *R5_NUMERIC_FIELDS, *R5_BOOLEAN_FIELDS]],
        on="date",
        how="outer",
        suffixes=("_saved", "_replay"),
        indicator=True,
        validate="one_to_one",
    )
    numeric_errors: dict[str, Any] = {}
    for field in R5_NUMERIC_FIELDS:
        left = pd.to_numeric(compare[f"{field}_saved"], errors="coerce")
        right = pd.to_numeric(compare[f"{field}_replay"], errors="coerce")
        both_missing = left.isna() & right.isna()
        one_missing = left.isna() ^ right.isna()
        difference = (left - right).abs().mask(both_missing, 0.0)
        numeric_errors[field] = {
            "maximum_absolute_difference": _safe_number(difference.max()),
            "missingness_mismatch_count": int(one_missing.sum()),
        }
    boolean_errors = {
        field: int(
            (
                compare[f"{field}_saved"].astype("boolean")
                != compare[f"{field}_replay"].astype("boolean")
            ).fillna(True).sum()
        )
        for field in R5_BOOLEAN_FIELDS
    }
    exact = (
        compare["_merge"].eq("both").all()
        and all(
            item["missingness_mismatch_count"] == 0
            and (item["maximum_absolute_difference"] or 0.0) <= 1e-12
            for item in numeric_errors.values()
        )
        and all(value == 0 for value in boolean_errors.values())
    )
    width = (valuation["fair_high_12m"] - valuation["fair_low_12m"]).replace(0.0, np.nan)
    reduced = ((valuation["fair_high_12m"] - valuation["index_close"]) / width).clip(0.0, 1.0)
    cancellation_error = (valuation["raw_continuous_position"] - reduced).abs()

    backup = backup_constituents.copy()
    backup["date"] = pd.to_datetime(backup["date"], errors="coerce")
    backup_eval = backup.loc[backup["date"].between(start, end)]
    current = current_constituents.copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce")
    current_eval = current.loc[current["date"].between(start, end)]
    current_daily = current_eval.groupby("date")["total_market_cap_cny"].agg(
        all_null=lambda values: values.isna().all(),
        aggregate_sum="sum",
    )
    return {
        "engineering_replay_status": "PASS_ENGINEERING_REPLAY_ONLY" if exact else "BLOCKED_REPLAY_MISMATCH",
        "comparison_rows": int(len(compare)),
        "calendar_exact_match": bool(compare["_merge"].eq("both").all()),
        "numeric_field_errors": numeric_errors,
        "boolean_field_mismatch_counts": boolean_errors,
        "backup_market_cap_non_null_ratio_evaluation": float(
            pd.to_numeric(backup_eval["total_market_cap_cny"], errors="coerce").notna().mean()
        ),
        "current_market_cap_non_null_ratio_evaluation": float(
            pd.to_numeric(current_eval["total_market_cap_cny"], errors="coerce").notna().mean()
        ),
        "current_all_null_market_cap_day_count": int(current_daily["all_null"].sum()),
        "current_zero_aggregate_from_all_null_day_count": int(
            (current_daily["all_null"] & current_daily["aggregate_sum"].eq(0.0)).sum()
        ),
        "algebraic_market_cap_cancellation": {
            "status": "PROVEN" if _safe_number(cancellation_error.max()) is not None and cancellation_error.max() <= 1e-12 else "NOT_PROVEN",
            "maximum_absolute_error": _safe_number(cancellation_error.max()),
            "reduced_formula": "clip((fair_high_12m-index_close)/(fair_high_12m-fair_low_12m),0,1)",
            "interpretation": "聚合市值在R5连续估值仓位公式中完全抵消，只承担非空闸门作用，没有提供独立的估值尺度。",
        },
        "economic_evidence_status": "NO_VIEW_NON_VINTAGE_VENDOR_PE_PB_AND_CAP_CANCELLATION",
        "return_or_position_evaluation_performed": False,
    }


def build_acquisition_requirements(
    coverage: pd.DataFrame,
    history: dict[str, Any],
    weights: pd.DataFrame,
) -> list[dict[str, Any]]:
    """把审计失败项转换为最小补采清单。"""

    five = history["five_year"]
    seven = history["seven_year"]
    five_frame = coverage.loc[coverage["in_five_year_preevaluation_window"]].copy()
    missing_price_months = five["price_coverage_failed_months"]
    missing_price_symbols: set[str] = set()
    if missing_price_months:
        missing_dates = set(
            five_frame.loc[five_frame["calendar_month"].isin(missing_price_months), "date"]
        )
        weight_data = weights.copy()
        weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"], errors="coerce")
        missing_price_symbols = set(
            weight_data.loc[weight_data["trade_date"].isin(missing_dates), "con_code"].astype(str)
        )
    first_normalized_pass = five_frame.loc[
        five_frame["normalized_metric_weight_coverage"].ge(0.90), "date"
    ]
    return [
        {
            "priority": 1,
            "gate": "VAL01_RAW_EY_5Y",
            "action": "补采官方权重快照成分在早期月份的未复权日线，并按停牌规则仅向前填充。",
            "required_months": missing_price_months,
            "required_symbol_count": len(missing_price_symbols),
            "recommended_raw_date_range": "2016-07-01至2019-12-20",
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
            "source": "tushare.daily或更高等级可审计源",
            "acceptance": "每个2016-08至2021-07官方权重快照的价格权重覆盖>=99%，停牌填充值不得来自未来日期。",
        },
        {
            "priority": 2,
            "gate": "VAL01_NORM_EY_5Y_AND_VAL02_5Y",
            "action": "补采2015-09以前的季度利润表、资产负债表和财务指标，消除标准化盈利在历史起点的截断偏差。",
            "current_first_report_period": "2015-09-30",
            "first_local_snapshot_with_normalized_coverage_at_least_90pct": (
                _safe_date(first_normalized_pass.min()) if not first_normalized_pass.empty else None
            ),
            "minimum_target_start_for_four_ttm_observations": "2014-06-30",
            "preferred_target_start_for_full_three_year_normalization_at_2016-08": "2012-06-30",
            "fields": [
                "ann_date",
                "f_ann_date",
                "end_date",
                "revenue",
                "n_income_attr_p",
                "total_hldr_eqy_exc_min_int",
                "total_share",
                "bps",
            ],
            "acceptance": "逐快照available_at过滤后，标准化盈利权重覆盖>=90%，并报告使用完整3年还是最少4个TTM观察的固定规则。",
        },
        {
            "priority": 3,
            "gate": "VAL01_7Y_AND_VAL02_7Y",
            "action": "补采2014-08至2016-07共24个月官方历史权重及相应价格；VAL-02另补同区间10年国债收益率。",
            "missing_weight_months": seven["missing_weight_months"],
            "financial_lookback_note": "若7年标准化盈利从2014-08开始且要求完整三年公司历史，财务原始期至少需追溯至2010-06。",
            "acceptance": "首个评价日前恰有84个经审计月度状态，所有月份同时通过成分、权重、价格、财务和利率闸门。",
        },
        {
            "priority": 4,
            "gate": "INDEPENDENT_SOURCE_VALIDATION",
            "action": "保留中证官方滚动PE为只读交叉核验，并新增原始响应/取得时刻/接口语义快照；不得将其替代成分股点时主链。",
            "acceptance": "交叉核验差异阈值在正式模型卡中预先冻结，且任何不一致均先归因到口径而非策略表现。",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    """生成不含收益结论的中文审计报告。"""

    datasets = report["datasets"]
    history = report["history_gates"]
    five = history["five_year"]
    seven = history["seven_year"]
    r5 = report["r5_replay"]
    pct = lambda value: "—" if value is None else f"{float(value):.2%}"
    lines = [
        "# 沪深300点时估值数据 V2 可重建性审计",
        "",
        "> 结论先行：本地数据不是完全缺失，而是“可部分重建但必须补采”。原始EY五年版主要缺2016-08至2019-11的成分股价格；标准化EY五年版还缺足够早的财务历史；七年版同时缺24个月权重、价格、早期财务和国债历史。正式VAL-01/02继续维持NO_VIEW。",
        "",
        "## 1. 总闸门",
        "",
        f"- 总状态：`{report['overall_status']}`",
        f"- 正式估值运行：`{report['formal_valuation_run_status']}`",
        f"- 输入哈希：`{report['hash_audit']['status']}`",
        "- 本轮未计算未来收益、IC、策略收益，未映射仓位，未生成订单。",
        "",
        "## 2. 核心数据证据",
        "",
        "| 数据 | 日期范围 | 行/快照 | 关键结论 |",
        "|---|---|---:|---|",
        f"| 官方月度权重 | {datasets['weights']['first_date']} 至 {datasets['weights']['last_date']} | {datasets['weights']['snapshot_count']}快照 | 每期{datasets['weights']['minimum_constituents']}只；权重和{datasets['weights']['minimum_weight_sum_pct']:.3f}%—{datasets['weights']['maximum_weight_sum_pct']:.3f}% |",
        f"| 点时财务事件 | {datasets['financials']['first_report_period']} 至 {datasets['financials']['last_report_period']} | {datasets['financials']['rows']} | {datasets['financials']['multi_version_company_report_pairs']}组多版本，{datasets['financials']['multi_version_pairs_with_value_changes']}组数值确有变化 |",
        f"| 当前成分日线 | {datasets['current_constituents']['first_date']} 至 {datasets['current_constituents']['last_date']} | {datasets['current_constituents']['rows']} | 未复权收盘覆盖{pct(datasets['current_constituents']['raw_close_non_null_ratio'])}，市值字段覆盖{pct(datasets['current_constituents']['market_cap_non_null_ratio'])} |",
        f"| 旧成分备份 | {datasets['backup_constituents']['first_date']} 至 {datasets['backup_constituents']['last_date']} | {datasets['backup_constituents']['rows']} | 市值字段覆盖{pct(datasets['backup_constituents']['market_cap_non_null_ratio'])}，但实为流通市值近似 |",
        "",
        "财务金额单位为人民币元，总股本单位为股，BPS单位为元/股；权重单位为百分比。旧备份的`total_market_cap_cny`实际由未复权收盘价乘新浪流通股本得到，不是中证指数计算用调整市值，字段名不能按字面解释。",
        "",
        "## 3. 五年窗口（2016-08至2021-07）",
        "",
        f"- 本地权重快照：{five['local_weight_snapshot_count']}/{five['required_snapshot_count']}。",
        f"- 原始EY：`{five['raw_ey_status']}`。价格未通过月份{len(five['price_coverage_failed_months'])}个，TTM未通过月份{len(five['ttm_coverage_failed_months'])}个。",
        f"- 标准化EY：`{five['normalized_ey_status']}`。标准化盈利未通过月份{len(five['normalized_coverage_failed_months'])}个。",
        f"- 标准化EY利差：`{five['normalized_ey_spread_status']}`。10年国债快照日缺失月份{len(five['cgb_10y_exact_date_failed_months'])}个。",
        f"- 最低覆盖：价格{pct(five['minimum_local_coverages']['price_weight_coverage'])}，TTM{pct(five['minimum_local_coverages']['ttm_metric_weight_coverage'])}，标准化盈利{pct(five['minimum_local_coverages']['normalized_metric_weight_coverage'])}。",
        "",
        "五年权重历史本身完整，但不能把‘60个权重快照存在’写成估值历史已就绪。当前成分日线从2019-12-23开始，因此更早40个月价格为硬缺口；财务事件从2015Q3开始，导致早期标准化盈利观察不足。",
        "",
        "## 4. 七年窗口（2014-08至2021-07）",
        "",
        f"- 本地权重快照：{seven['local_weight_snapshot_count']}/{seven['required_snapshot_count']}。",
        f"- 缺失权重月份：{len(seven['missing_weight_months'])}个（2014-08至2016-07）。",
        f"- 原始EY：`{seven['raw_ey_status']}`；标准化EY：`{seven['normalized_ey_status']}`；利差：`{seven['normalized_ey_spread_status']}`。",
        "- 七年标准化盈利若要求2014-08即有完整三年公司历史，财务原始期至少需追溯至2010-06附近；只补2014年财报仍不充分。",
        "",
        "## 5. R5 精确重放",
        "",
        f"- 工程状态：`{r5['engineering_replay_status']}`，逐字段比较{r5['comparison_rows']}个交易日。",
        f"- 最大代数误差：{r5['algebraic_market_cap_cancellation']['maximum_absolute_error']:.3e}。",
        f"- 当前主成分文件市值非空率：{pct(r5['current_market_cap_non_null_ratio_evaluation'])}；旧备份：{pct(r5['backup_market_cap_non_null_ratio_evaluation'])}。",
        "- 关键结论：R5的聚合市值在连续仓位公式中完全抵消，只是非空闸门。旧信号可工程复现，不证明其估值经济机制成立。",
        f"- 经济证据状态：`{r5['economic_evidence_status']}`。",
        "",
        "## 6. 元数据漂移",
        "",
    ]
    for item in report["stale_metadata"]["rows"]:
        lines.append(f"- `{item['artifact']}`：{'已过期' if item['stale'] else '一致'}。{item['meaning']}")
    lines.extend(
        [
            "",
            "这些旧文件没有被修改；新报告只记录漂移，避免破坏既有审计链。",
            "",
            "## 7. 补采顺序与验收",
            "",
        ]
    )
    for item in report["acquisition_requirements"]:
        lines.append(f"{item['priority']}. **{item['gate']}**：{item['action']}")
        lines.append(f"   - 验收：{item['acceptance']}")
    lines.extend(
        [
            "",
            "## 8. 下一道允许闸门",
            "",
            "只有补采完成并重新运行本审计后，才允许冻结VAL-01/02模型卡。冻结前仍需明确：月度信号使用当月权重快照的最早可用时点、国债缺报日对齐规则、标准化盈利是否要求完整三年历史，以及扩展窗口的固定起点。上述定义未冻结前，不运行收益检验。",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(config: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """执行完整审计并返回报告和月度覆盖表。"""

    config = config or load_config()
    contracts = config["data_contracts"]
    read = lambda name: pd.read_parquet(ROOT / contracts[name]["file"])
    weights = read("historical_weights")
    financials = read("point_in_time_financials")
    current_constituents = read("current_constituent_daily")
    backup_constituents = read("pre_tushare_constituent_backup")
    index_daily = read("index_daily")
    vendor = read("vendor_valuation")
    official = read("official_pe")
    bonds = read("government_bond_yields")
    existing_panel = read("existing_monthly_fundamental_panel")
    preserved = read("preserved_r5_enhanced_signals")

    hash_audit = audit_hashes(config)
    coverage = build_snapshot_coverage(
        weights, financials, current_constituents, index_daily, bonds, config
    )
    history = summarize_history_gate(coverage, config)
    datasets = {
        "weights": audit_weights(weights),
        "financials": audit_financials(financials, contracts["financial_checkpoints"]),
        "current_constituents": audit_constituent_panel(current_constituents, "current_tushare_panel"),
        "backup_constituents": audit_constituent_panel(backup_constituents, "pre_tushare_sina_backup"),
        "vendor_valuation": {
            "rows": int(len(vendor)),
            "first_date": _safe_date(pd.to_datetime(vendor["date"]).min()),
            "last_date": _safe_date(pd.to_datetime(vendor["date"]).max()),
            "point_in_time_vintage_proven": False,
        },
        "official_pe": {
            "rows": int(len(official)),
            "first_date": _safe_date(pd.to_datetime(official["date"]).min()),
            "last_date": _safe_date(pd.to_datetime(official["date"]).max()),
            "point_in_time_vintage_proven": False,
        },
        "government_bonds": {
            "rows": int(len(bonds)),
            "first_date": _safe_date(pd.to_datetime(bonds["date"]).min()),
            "last_date": _safe_date(pd.to_datetime(bonds["date"]).max()),
        },
    }
    r5 = audit_r5_replay(
        vendor,
        backup_constituents,
        current_constituents,
        index_daily,
        preserved,
        config,
    )
    cross_checks = audit_cross_checks(vendor, official, existing_panel)
    stale = audit_stale_metadata(config)
    acquisition = build_acquisition_requirements(coverage, history, weights)
    formal_ready = all(
        history["five_year"][field] == "PASS_LOCAL_RECONSTRUCTIBLE"
        for field in ("raw_ey_status", "normalized_ey_status", "normalized_ey_spread_status")
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(ZoneInfo(config["protocol"]["timezone"])).isoformat(),
        "overall_status": "PASS_LOCAL_RECONSTRUCTIBLE" if formal_ready else "PARTIALLY_RECONSTRUCTIBLE_LOCAL_ACQUISITION_REQUIRED",
        "formal_valuation_run_status": "ALLOWED" if formal_ready else "NO_VIEW",
        "governance": {
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "frozen_files_mutated": False,
        },
        "hash_audit": hash_audit,
        "datasets": datasets,
        "history_gates": history,
        "cross_checks": cross_checks,
        "r5_replay": r5,
        "stale_metadata": stale,
        "acquisition_requirements": acquisition,
        "source_documents": config["source_documents"],
        "artifacts": config["artifacts"],
    }
    return _sanitize(report), coverage


def write_artifacts(
    report: dict[str, Any],
    coverage: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> None:
    """原子式写出覆盖表、JSON和Markdown报告。"""

    config = config or load_config()
    artifacts = config["artifacts"]
    coverage_path = ROOT / artifacts["snapshot_coverage"]
    json_path = ROOT / artifacts["report_json"]
    markdown_path = ROOT / artifacts["report_markdown"]
    for path in (coverage_path, json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    coverage_temp = coverage_path.with_suffix(coverage_path.suffix + ".tmp")
    coverage.to_parquet(coverage_temp, index=False)
    coverage_temp.replace(coverage_path)
    report["artifacts"]["snapshot_coverage_sha256"] = sha256_file(coverage_path)
    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    json_temp.replace(json_path)
    markdown_temp = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_temp.write_text(render_markdown(report), encoding="utf-8")
    markdown_temp.replace(markdown_path)

