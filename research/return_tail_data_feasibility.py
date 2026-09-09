"""510300 稀疏尾部风险门控的数据可行性审计。

本模块只检查输入存在性、字段、覆盖和点时语义，不读取未来收益标签，
也不计算信号效果、仓位或订单。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "config" / "return_tail_hypothesis_registry.yaml"
MASTER_REPORT_FILE = ROOT / "reports" / "data_quality" / "return_tail_feasibility.json"
OPTIONS_REPORT_FILE = ROOT / "reports" / "data_quality" / "options_surface_feasibility.md"
CONSTITUENT_REPORT_FILE = (
    ROOT / "reports" / "data_quality" / "constituent_tail_feasibility.md"
)
EARNINGS_REPORT_FILE = (
    ROOT / "reports" / "data_quality" / "earnings_revision_vintage_feasibility.md"
)
UNIVARIATE_REPORT_FILE = ROOT / "reports" / "discovery" / "univariate_signal_oos.md"
INCREMENTAL_REPORT_FILE = (
    ROOT / "reports" / "discovery" / "incremental_information_audit.md"
)
TIMING_REPORT_FILE = ROOT / "reports" / "backtest" / "unit_timing_contribution.md"
STATUS_FILE = ROOT / "paper" / "return_tail_gate_forward" / "status.json"
MANIFEST_FILE = ROOT / "config" / "return_tail_gating_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text_write(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _resolve(root: Path, relative: str) -> Path:
    return root / Path(relative)


def _file_evidence(root: Path, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "file": _relative(root, path),
            "exists": False,
            "bytes": 0,
            "sha256": None,
        }
    return {
        "file": _relative(root, path),
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def _required_columns(data: pd.DataFrame, required: list[str]) -> list[str]:
    return sorted(set(required).difference(data.columns))


def _normalize_option_type(values: pd.Series) -> pd.Series:
    mapping = {
        "P": "P",
        "PUT": "P",
        "认沽": "P",
        "C": "C",
        "CALL": "C",
        "认购": "C",
    }
    return values.astype(str).str.strip().str.upper().map(mapping)


def audit_options(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    quote_path = _resolve(root, contract["quote_file"])
    master_path = _resolve(root, contract["contract_master_file"])
    cffex_path = _resolve(root, contract["cffex_cross_validation_file"])
    underlying_path = _resolve(root, contract["underlying_file"])
    dividend_path = _resolve(root, contract["dividend_file"])
    risk_free_path = _resolve(root, contract["risk_free_file"])
    files = {
        "quote": _file_evidence(root, quote_path),
        "contract_master": _file_evidence(root, master_path),
        "cffex_cross_validation": _file_evidence(root, cffex_path),
        "underlying": _file_evidence(root, underlying_path),
        "dividend": _file_evidence(root, dividend_path),
        "risk_free": _file_evidence(root, risk_free_path),
    }
    primary_missing = [
        name
        for name in ("quote", "contract_master", "underlying", "dividend", "risk_free")
        if not files[name]["exists"]
    ]
    base: dict[str, Any] = {
        "branch": "OPTIONS",
        "status": "BLOCKED_MISSING_LOCAL_OPTION_HISTORY",
        "return_test_allowed": False,
        "files": files,
        "missing_primary_inputs": primary_missing,
        "official_references": contract.get("official_references", []),
        "public_data_boundary": (
            "交易所规则证明实时行情含买卖报价、成交量和持仓量；这不等于本地已经拥有"
            "可回放的历史收盘快照。510300期权历史盘口需要上交所授权行情产品或具备"
            "相应许可的数据商交付；中金所快照只用于IO交叉验证。"
        ),
        "checks": {},
        "blockers": [],
    }
    if primary_missing:
        if "quote" in primary_missing:
            base["blockers"].append(
                "本地已有合约主表、逐合约日线、风险指标和最后五日分钟成交，但没有"
                "2019-12-23至2026-08-14逐日收盘bid1、ask1及市场时间戳，无法验证历史曲面。"
            )
        if "contract_master" in primary_missing:
            base["blockers"].append("本地缺少可识别调整合约的510300期权合约主表。")
        for name in ("underlying", "dividend", "risk_free"):
            if name in primary_missing:
                base["blockers"].append(f"本地缺少期权分支必要输入：{name}。")
        base["blockers"].append(
            "不得用网页当前行情、仅成交价日线、最后一笔残留盘口或估计价差替代逐日历史盘口。"
        )
        if not files["cffex_cross_validation"]["exists"]:
            base["blockers"].append("本地也没有IO股指期权同概念交叉验证链。")
        return base

    quote = pd.read_parquet(quote_path)
    master = pd.read_parquet(master_path)
    missing_quote_columns = _required_columns(
        quote, list(contract["quote_required_columns"])
    )
    missing_master_columns = _required_columns(
        master, list(contract["master_required_columns"])
    )
    base["checks"]["missing_quote_columns"] = missing_quote_columns
    base["checks"]["missing_master_columns"] = missing_master_columns
    if missing_quote_columns or missing_master_columns:
        base["status"] = "BLOCKED_INVALID_OPTION_SCHEMA"
        base["blockers"] = [
            f"期权行情缺少字段：{missing_quote_columns}",
            f"合约主表缺少字段：{missing_master_columns}",
        ]
        return base

    quote = quote.copy()
    master = master.copy()
    quote["trade_date"] = pd.to_datetime(quote["trade_date"], errors="coerce").dt.normalize()
    quote["expiry_date"] = pd.to_datetime(quote["expiry_date"], errors="coerce").dt.normalize()
    quote["quote_timestamp"] = pd.to_datetime(quote["quote_timestamp"], errors="coerce")
    quote["normalized_option_type"] = _normalize_option_type(quote["option_type"])
    numeric_columns = [
        "strike",
        "contract_unit",
        "bid1",
        "ask1",
        "close",
        "volume",
        "open_interest",
        "underlying_close",
    ]
    for column in numeric_columns:
        quote[column] = pd.to_numeric(quote[column], errors="coerce")

    required = list(contract["quote_required_columns"])
    null_ratio = float(quote[required].isna().mean().max()) if len(quote) else 1.0
    inverted = quote["bid1"].gt(quote["ask1"]) & quote["ask1"].gt(0)
    inverted_ratio = float(inverted.mean()) if len(quote) else 1.0
    timestamp_same_date = quote["quote_timestamp"].dt.normalize().eq(quote["trade_date"])
    quote_minutes = (
        quote["quote_timestamp"].dt.hour * 60
        + quote["quote_timestamp"].dt.minute
    )
    timestamp_in_close_window = quote_minutes.between(14 * 60 + 55, 16 * 60)
    timestamp_valid_ratio = float((timestamp_same_date & timestamp_in_close_window).mean())

    underlying = pd.read_parquet(underlying_path, columns=["date"])
    underlying_dates = pd.to_datetime(underlying["date"], errors="coerce").dt.normalize()
    listing = pd.Timestamp(contract["listing_date"])
    cutoff = pd.Timestamp(contract["cutoff_date"])
    expected_dates = set(underlying_dates[underlying_dates.between(listing, cutoff)].dropna())
    available_dates = set(quote["trade_date"].dropna())
    date_coverage = len(expected_dates & available_dates) / max(len(expected_dates), 1)

    midpoint = (quote["bid1"] + quote["ask1"]) / 2.0
    relative_spread = (quote["ask1"] - quote["bid1"]) / midpoint.replace(0, np.nan)
    normalized_status = quote["contract_status"].astype(str).str.strip().str.upper()
    allowed_status = {
        str(value).strip().upper() for value in contract["allowed_contract_status"]
    }
    valid_quote = (
        quote["normalized_option_type"].notna()
        & normalized_status.isin(allowed_status)
        & quote["bid1"].gt(0)
        & quote["ask1"].ge(quote["bid1"])
        & relative_spread.le(float(contract["maximum_relative_bid_ask_spread"]))
        & quote["volume"].gt(0)
        & quote["open_interest"].gt(0)
        & quote["strike"].gt(0)
        & quote["contract_unit"].gt(0)
        & quote["underlying_close"].gt(0)
    )
    usable = quote.loc[valid_quote].copy()
    usable["days_to_expiry"] = (usable["expiry_date"] - usable["trade_date"]).dt.days
    usable_puts = usable.loc[
        usable["normalized_option_type"].eq("P")
        & usable["strike"].lt(usable["underlying_close"])
        & usable["days_to_expiry"].between(7, 90)
    ]
    strike_counts = (
        usable_puts.groupby(["trade_date", "expiry_date"])["strike"].nunique()
        if len(usable_puts)
        else pd.Series(dtype="int64")
    )
    minimum_strikes = int(contract["minimum_usable_otm_put_strikes_per_expiry"])
    qualifying = strike_counts.loc[strike_counts.ge(minimum_strikes)]
    surface_dates: set[pd.Timestamp] = set()
    for trade_date in sorted(set(usable_puts["trade_date"].dropna())):
        expiries = [
            pd.Timestamp(expiry)
            for date, expiry in qualifying.index
            if pd.Timestamp(date) == pd.Timestamp(trade_date)
        ]
        tenors = sorted((expiry - pd.Timestamp(trade_date)).days for expiry in expiries)
        if any(tenor <= 25 for tenor in tenors) and any(tenor >= 25 for tenor in tenors):
            surface_dates.add(pd.Timestamp(trade_date))
    surface_usable_date_ratio = len(surface_dates) / max(len(expected_dates), 1)

    master_codes = set(master["contract_code"].dropna().astype(str))
    quote_codes = set(quote["contract_code"].dropna().astype(str))
    master_coverage = len(master_codes & quote_codes) / max(len(quote_codes), 1)
    adjusted_identifiable = bool(master["is_adjusted"].notna().all())
    adjusted_contract_count = int(master["is_adjusted"].fillna(False).astype(bool).sum())

    checks = {
        "row_count": int(len(quote)),
        "contract_count": int(quote["contract_code"].nunique()),
        "first_trade_date": str(quote["trade_date"].min().date()),
        "last_trade_date": str(quote["trade_date"].max().date()),
        "expected_trading_date_count": int(len(expected_dates)),
        "available_trading_date_count": int(len(available_dates & expected_dates)),
        "date_coverage": float(date_coverage),
        "maximum_required_field_null_ratio": null_ratio,
        "inverted_quote_ratio": inverted_ratio,
        "median_relative_bid_ask_spread": float(relative_spread.median()),
        "quotes_with_allowed_contract_status_ratio": float(
            normalized_status.isin(allowed_status).mean()
        ),
        "quote_timestamp_valid_ratio": timestamp_valid_ratio,
        "surface_usable_date_ratio": float(surface_usable_date_ratio),
        "contract_master_coverage": float(master_coverage),
        "adjusted_contract_identifiable": adjusted_identifiable,
        "adjusted_contract_count": adjusted_contract_count,
        "cffex_cross_validation_available": files["cffex_cross_validation"]["exists"],
    }
    base["checks"].update(checks)
    gates = {
        "date_coverage": date_coverage
        >= float(contract["minimum_expected_date_coverage"]),
        "required_field_null_ratio": null_ratio
        <= float(contract["maximum_required_field_null_ratio"]),
        "inverted_quote_ratio": inverted_ratio
        <= float(contract["maximum_inverted_quote_ratio"]),
        "quote_timestamp": timestamp_valid_ratio >= 0.99,
        "surface_coverage": surface_usable_date_ratio
        >= float(contract["minimum_surface_usable_date_ratio"]),
        "contract_master_coverage": master_coverage >= 0.999,
        "adjusted_contracts": adjusted_identifiable and adjusted_contract_count > 0,
    }
    base["gates"] = gates
    if all(gates.values()):
        base["status"] = "PASS"
        base["return_test_allowed"] = True
        base["blockers"] = []
    else:
        base["status"] = "BLOCKED_OPTION_QUALITY_GATES"
        base["blockers"] = [name for name, passed in gates.items() if not passed]
    return base


def _active_weight_age_days(
    daily_dates: pd.Series, weight_dates: pd.Series
) -> pd.Series:
    normalized_daily = pd.to_datetime(daily_dates, errors="coerce").astype(
        "datetime64[ns]"
    )
    normalized_weights = pd.to_datetime(weight_dates, errors="coerce").astype(
        "datetime64[ns]"
    )
    left = pd.DataFrame({"date": sorted(normalized_daily.dropna().unique())})
    right = pd.DataFrame(
        {"weight_date": sorted(normalized_weights.dropna().unique())}
    )
    if left.empty or right.empty:
        return pd.Series(dtype="float64")
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("weight_date"),
        left_on="date",
        right_on="weight_date",
        direction="backward",
    )
    return (merged["date"] - merged["weight_date"]).dt.days


def audit_constituents(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    paths = {
        name: _resolve(root, contract[key])
        for name, key in (
            ("membership", "membership_file"),
            ("weights", "weights_file"),
            ("daily", "daily_file"),
            ("industry", "industry_file"),
            ("market", "market_file"),
        )
    }
    files = {name: _file_evidence(root, path) for name, path in paths.items()}
    missing = [name for name, evidence in files.items() if not evidence["exists"]]
    result: dict[str, Any] = {
        "branch": "CONSTITUENTS",
        "status": "BLOCKED_MISSING_CONSTITUENT_INPUTS",
        "return_test_allowed": False,
        "files": files,
        "missing_primary_inputs": missing,
        "checks": {},
        "gates": {},
        "warnings": [],
        "blockers": [],
    }
    if missing:
        result["blockers"] = [f"缺少成分股输入：{', '.join(missing)}"]
        return result

    membership = pd.read_parquet(paths["membership"])
    weights = pd.read_parquet(paths["weights"])
    daily = pd.read_parquet(paths["daily"])
    industry = pd.read_parquet(paths["industry"])
    required = {
        "membership": ["symbol", "opt_in", "opt_out", "source"],
        "weights": ["con_code", "trade_date", "weight", "source"],
        "daily": [
            "date",
            "con_code",
            "is_index_member",
            "is_suspended",
            "total_return_open",
            "total_return_high",
            "total_return_low",
            "total_return_close",
        ],
        "industry": ["con_code", "industry_l1", "classification_usage", "source"],
    }
    frames = {
        "membership": membership,
        "weights": weights,
        "daily": daily,
        "industry": industry,
    }
    schema_missing = {
        name: _required_columns(frames[name], columns)
        for name, columns in required.items()
    }
    schema_missing = {name: value for name, value in schema_missing.items() if value}
    result["checks"]["missing_columns"] = schema_missing
    if schema_missing:
        result["status"] = "BLOCKED_INVALID_CONSTITUENT_SCHEMA"
        result["blockers"] = [f"字段缺失：{schema_missing}"]
        return result

    daily = daily.copy()
    weights = weights.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    weights["trade_date"] = pd.to_datetime(
        weights["trade_date"], errors="coerce"
    ).dt.normalize()
    members = daily.loc[daily["is_index_member"].fillna(False).astype(bool)].copy()
    member_counts = members.groupby("date")["con_code"].nunique()
    valid_close_counts = members.groupby("date")["total_return_close"].count()
    valid_close_ratio = valid_close_counts / member_counts.replace(0, np.nan)
    first_date = pd.Timestamp(daily["date"].min())
    last_date = pd.Timestamp(daily["date"].max())
    trading_days = int(daily["date"].nunique())
    weight_age = _active_weight_age_days(daily["date"], weights["trade_date"])
    maximum_weight_age = (
        int(weight_age.max()) if len(weight_age) and weight_age.notna().any() else None
    )
    missing_weight_age_days = int(weight_age.isna().sum()) if len(weight_age) else trading_days
    industry_usage = sorted(
        industry["classification_usage"].dropna().astype(str).unique().tolist()
    )
    industry_is_point_in_time = any(
        token in value.upper() for value in industry_usage for token in ("POINT_IN_TIME", "PIT")
    )
    sources = {
        "membership": sorted(membership["source"].dropna().astype(str).unique().tolist()),
        "weights": sorted(weights["source"].dropna().astype(str).unique().tolist()),
        "daily_price": sorted(daily["price_source"].dropna().astype(str).unique().tolist())
        if "price_source" in daily
        else [],
        "industry": sorted(industry["source"].dropna().astype(str).unique().tolist()),
    }
    result["checks"].update(
        {
            "row_count": int(len(daily)),
            "trading_days": trading_days,
            "first_date": str(first_date.date()),
            "last_date": str(last_date.date()),
            "unique_constituents": int(daily["con_code"].nunique()),
            "minimum_members_per_day": int(member_counts.min()),
            "maximum_members_per_day": int(member_counts.max()),
            "days_member_count_not_300": int((member_counts != 300).sum()),
            "minimum_valid_close_ratio": float(valid_close_ratio.min()),
            "median_valid_close_ratio": float(valid_close_ratio.median()),
            "weight_snapshot_count": int(weights["trade_date"].nunique()),
            "first_weight_date": str(weights["trade_date"].min().date()),
            "last_weight_date": str(weights["trade_date"].max().date()),
            "maximum_weight_age_calendar_days": maximum_weight_age,
            "days_without_prior_weight_snapshot": missing_weight_age_days,
            "industry_classification_usage": industry_usage,
            "industry_is_point_in_time": industry_is_point_in_time,
            "sources": sources,
        }
    )
    gates = {
        "member_count": int(member_counts.min())
        >= int(contract["minimum_member_count_per_day"])
        and int(member_counts.max()) <= int(contract["maximum_member_count_per_day"]),
        "valid_close_coverage": float(valid_close_ratio.min())
        >= float(contract["minimum_valid_close_coverage"]),
        "minimum_history": trading_days >= int(contract["minimum_history_trading_days"]),
        "required_start": first_date <= pd.Timestamp(contract["required_start_date"]),
        "required_end": last_date >= pd.Timestamp(contract["required_end_date"]),
        "weight_age": maximum_weight_age is not None
        and maximum_weight_age <= int(contract["maximum_weight_age_calendar_days"])
        and missing_weight_age_days == 0,
        "point_in_time_industry": industry_is_point_in_time,
    }
    result["gates"] = gates
    result["warnings"].append(
        "历史成员区间来自第三方开源整理并与当前官方快照交叉验证；尚未逐期保存官方调样公告哈希。"
    )
    if not industry_is_point_in_time:
        result["warnings"].append(
            "行业表是静态映射，不能支持X1的行业扩散次统计量，也不能回填为历史行业。"
        )
    failed = [name for name, passed in gates.items() if not passed]
    if not failed:
        result["status"] = "PASS"
        result["return_test_allowed"] = True
    else:
        result["status"] = "PARTIAL_CONSTITUENT_HISTORY_NOT_RESEARCH_READY"
        result["blockers"] = failed
    return result


def audit_earnings_consensus(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    snapshot_path = _resolve(root, contract["snapshot_file"])
    membership_path = _resolve(root, contract["membership_file"])
    weights_path = _resolve(root, contract["weights_file"])
    actual_financials = (
        root / "data" / "raw" / "fundamentals" / "csi300_financials_point_in_time.parquet"
    )
    files = {
        "consensus_vintages": _file_evidence(root, snapshot_path),
        "membership": _file_evidence(root, membership_path),
        "weights": _file_evidence(root, weights_path),
        "actual_company_financials_non_substitute": _file_evidence(root, actual_financials),
    }
    result: dict[str, Any] = {
        "branch": "EARNINGS_CONSENSUS",
        "status": "TERMINATED_NO_POINT_IN_TIME_CONSENSUS_HISTORY",
        "return_test_allowed": False,
        "files": files,
        "checks": {},
        "blockers": [],
        "termination_is_protocol_required": True,
    }
    if not snapshot_path.exists():
        result["blockers"] = [
            "没有分析师一致预期历史快照文件。",
            "现有点时财务表记录公司已公告财报，不含FTM EPS一致预期、分析师数、分歧或vendor vintage。",
            "协议禁止用当前数据库回填过去一致预期，因此E1在首轮直接终止。",
        ]
        return result

    consensus = pd.read_parquet(snapshot_path)
    missing_columns = _required_columns(consensus, list(contract["required_columns"]))
    result["checks"]["missing_columns"] = missing_columns
    if missing_columns:
        result["status"] = "TERMINATED_INVALID_CONSENSUS_VINTAGE_SCHEMA"
        result["blockers"] = [f"一致预期快照缺少字段：{missing_columns}"]
        return result
    consensus = consensus.copy()
    consensus["asof_timestamp"] = pd.to_datetime(
        consensus["asof_timestamp"], errors="coerce"
    )
    consensus["asof_date"] = consensus["asof_timestamp"].dt.normalize()
    required = list(contract["required_columns"])
    null_ratio = float(consensus[required].isna().mean().max()) if len(consensus) else 1.0
    days = int(consensus["asof_date"].nunique())
    duplicate_vintages = int(
        consensus.duplicated(
            ["asof_timestamp", "con_code", "forecast_fiscal_year", "source_vintage_id"]
        ).sum()
    )
    result["checks"].update(
        {
            "row_count": int(len(consensus)),
            "snapshot_days": days,
            "first_asof_timestamp": consensus["asof_timestamp"].min().isoformat(),
            "last_asof_timestamp": consensus["asof_timestamp"].max().isoformat(),
            "maximum_required_field_null_ratio": null_ratio,
            "duplicate_vintage_rows": duplicate_vintages,
            "source_vintage_count": int(consensus["source_vintage_id"].nunique()),
        }
    )
    gates = {
        "minimum_history": days >= int(contract["minimum_history_trading_days"]),
        "required_fields_complete": null_ratio <= 0.01,
        "unique_vintage_rows": duplicate_vintages == 0,
        "membership_available": membership_path.exists(),
        "weights_available": weights_path.exists(),
    }
    result["gates"] = gates
    if all(gates.values()):
        result["status"] = "PASS"
        result["return_test_allowed"] = True
        result["termination_is_protocol_required"] = False
        result["blockers"] = []
    else:
        result["status"] = "TERMINATED_CONSENSUS_VINTAGE_QUALITY_GATES"
        result["blockers"] = [name for name, passed in gates.items() if not passed]
    return result


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    protocol = registry["protocol"]
    hypotheses = registry["hypotheses"]
    combinations = registry["combination_models"]
    policy = registry["position_policy"]
    ids = [item["id"] for item in hypotheses] + [
        item["id"] for item in combinations
    ] + [policy["id"]]
    expected_count = int(protocol["candidate_budget"])
    if len(ids) != expected_count:
        raise ValueError(f"候选预算应为{expected_count}，实际为{len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("候选ID重复")
    if int(protocol["univariate_hypothesis_count"]) != len(hypotheses):
        raise ValueError("单变量候选计数与注册表不一致")
    if int(protocol["combination_model_count"]) != len(combinations):
        raise ValueError("组合模型计数与注册表不一致")
    governance = registry["governance"]
    forbidden_true = [
        key
        for key in (
            "position_mapping_enabled",
            "order_generation_enabled",
            "broker_connection_enabled",
            "live_trading_authorized",
        )
        if bool(governance[key])
    ]
    if forbidden_true:
        raise ValueError(f"研究治理边界异常：{forbidden_true}不得启用")
    return {"candidate_ids": ids, "candidate_count": len(ids)}


def _render_header(title: str, checked_at: str, status: str) -> list[str]:
    return [
        f"# {title}",
        "",
        f"审计时间：`{checked_at}`  ",
        f"状态：`{status}`  ",
        "收益回测许可：`DISABLED`" if status != "PASS" else "收益回测许可：`BRANCH_ONLY`",
        "",
    ]


def _render_options_report(result: dict[str, Any], checked_at: str) -> str:
    lines = _render_header("期权曲面数据可行性审计", checked_at, result["status"])
    lines.extend(
        [
            "## 结论",
            "",
            (
                "O1—O4 当前不得运行收益检验。"
                if not result["return_test_allowed"]
                else "期权分支的数据门槛已通过，只允许进入原始预测检验。"
            ),
            "",
            "## 本地输入",
            "",
            "| 输入 | 存在 | 字节数 | 文件 |",
            "|---|---:|---:|---|",
        ]
    )
    for name, evidence in result["files"].items():
        lines.append(
            f"| {name} | {str(evidence['exists']).lower()} | {evidence['bytes']} | "
            f"`{evidence['file']}` |"
        )
    lines.extend(["", "## 已执行检查", "", "```json"])
    lines.append(json.dumps(result.get("checks", {}), ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## 阻断项", ""])
    blockers = result.get("blockers", [])
    lines.extend([f"- {item}" for item in blockers] or ["- 无。"])
    lines.extend(
        [
            "",
            "## 官方数据边界",
            "",
            f"{result['public_data_boundary']}",
            "",
        ]
    )
    for url in result.get("official_references", []):
        lines.append(f"- {url}")
    lines.extend(
        [
            "",
            "只有行情链、合约主表、标准/调整标志、合约单位、收盘买卖报价、成交量、持仓量、"
            "市场时间戳和曲面覆盖全部通过后，才可把本报告状态改为 PASS。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_constituent_report(result: dict[str, Any], checked_at: str) -> str:
    lines = _render_header("成分股共同尾部数据可行性审计", checked_at, result["status"])
    checks = result.get("checks", {})
    lines.extend(
        [
            "## 结论",
            "",
            (
                "X1 当前不得运行收益检验。现有长表可用于继续做数据治理，但不满足完整协议。"
                if not result["return_test_allowed"]
                else "X1 数据门槛已通过，只允许进入原始预测检验。"
            ),
            "",
            "## 可审计覆盖",
            "",
            "```json",
            json.dumps(checks, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 门槛",
            "",
            "| 检查 | 通过 |",
            "|---|---:|",
        ]
    )
    for name, passed in result.get("gates", {}).items():
        lines.append(f"| {name} | {str(bool(passed)).lower()} |")
    lines.extend(["", "## 警告", ""])
    lines.extend([f"- {item}" for item in result.get("warnings", [])] or ["- 无。"])
    lines.extend(["", "## 阻断项", ""])
    lines.extend([f"- `{item}`" for item in result.get("blockers", [])] or ["- 无。"])
    lines.extend(
        [
            "",
            "静态行业映射不得冒充点时行业分类。缺失的 2019—2021 成分股日线也不得通过"
            "只保留有数据股票来隐性缩短样本。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_earnings_report(result: dict[str, Any], checked_at: str) -> str:
    lines = _render_header("盈利预测历史快照可行性审计", checked_at, result["status"])
    lines.extend(
        [
            "## 结论",
            "",
            (
                "E1 按预注册停止条件终止。"
                if result["status"].startswith("TERMINATED")
                else "E1 数据门槛已通过，只允许进入原始预测检验。"
            ),
            "",
            "公司公告财报、当前一致预期或事后整理的 FY1/FY2 数值都不能替代真实历史 vintage。",
            "",
            "## 本地输入",
            "",
            "| 输入 | 存在 | 文件 |",
            "|---|---:|---|",
        ]
    )
    for name, evidence in result["files"].items():
        lines.append(
            f"| {name} | {str(evidence['exists']).lower()} | `{evidence['file']}` |"
        )
    lines.extend(["", "## 已执行检查", "", "```json"])
    lines.append(json.dumps(result.get("checks", {}), ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## 停止原因", ""])
    lines.extend([f"- {item}" for item in result.get("blockers", [])] or ["- 无。"])
    lines.extend(
        [
            "",
            "若未来采购到包含 as-of 时间、预测财年、EPS、分析师数、分歧和不可变 vintage ID 的"
            "历史快照，必须建立下一份独立数据接入协议；不得在当前首轮静默恢复 E1。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_not_run_report(
    title: str,
    checked_at: str,
    reason: str,
    prohibited_outputs: list[str],
) -> str:
    lines = [
        f"# {title}",
        "",
        "状态：`NOT_RUN_DATA_FEASIBILITY_GATE`  ",
        f"记录时间：`{checked_at}`",
        "",
        "## 结论",
        "",
        reason,
        "",
        "本文件不是空白占位，而是数据门控的正式否决记录。本次没有读取未来收益标签。",
        "",
        "## 未生成内容",
        "",
    ]
    lines.extend([f"- {item}" for item in prohibited_outputs])
    lines.extend(
        [
            "",
            "只有 `reports/data_quality/return_tail_feasibility.json` 明确列出可运行的候选 ID 后，"
            "才允许由独立研究脚本替换本记录。",
            "",
        ]
    )
    return "\n".join(lines)


def run_feasibility_audit(
    root: Path = ROOT,
    registry_file: Path | None = None,
) -> dict[str, Any]:
    registry_path = registry_file or (root / "config" / "return_tail_hypothesis_registry.yaml")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry_evidence = validate_registry(registry)
    checked_at = datetime.now(ZoneInfo(registry["timing"]["timezone"])).isoformat()
    options = audit_options(root, registry["data_contracts"]["options"])
    constituents = audit_constituents(root, registry["data_contracts"]["constituents"])
    earnings = audit_earnings_consensus(
        root, registry["data_contracts"]["earnings_consensus"]
    )
    branches = {
        "OPTIONS": options,
        "CONSTITUENTS": constituents,
        "EARNINGS_CONSENSUS": earnings,
    }
    family_hypotheses: dict[str, list[str]] = {}
    for item in registry["hypotheses"]:
        family_hypotheses.setdefault(item["family"], []).append(item["id"])
    allowed_hypotheses = sorted(
        hypothesis
        for family, result in branches.items()
        if result["status"] == "PASS"
        for hypothesis in family_hypotheses.get(family, [])
    )
    terminated_hypotheses = sorted(
        hypothesis
        for family, result in branches.items()
        if result["status"].startswith("TERMINATED")
        for hypothesis in family_hypotheses.get(family, [])
    )
    if len(allowed_hypotheses) == 6:
        overall_status = "ALL_BRANCHES_READY"
    elif allowed_hypotheses:
        overall_status = "PARTIAL_BRANCH_READY"
    else:
        overall_status = "DATA_FEASIBILITY_BLOCKED"
    manifest = _file_evidence(root, root / "config" / "return_tail_gating_manifest.json")
    master = {
        "project_id": registry["protocol"]["project_id"],
        "checked_at": checked_at,
        "status": overall_status,
        "evidence_label": "DISCOVERY_ONLY_NO_RETURN_LABELS_READ",
        "registry": {
            **registry_evidence,
            "file": _relative(root, registry_path),
            "sha256": sha256(registry_path),
        },
        "manifest": manifest,
        "branch_status": {
            family: result["status"] for family, result in branches.items()
        },
        "hypotheses_allowed_for_return_test": allowed_hypotheses,
        "hypotheses_terminated": terminated_hypotheses,
        "study_wide_return_backtest_allowed": len(allowed_hypotheses) > 0,
        "combination_model_build_allowed": False,
        "position_policy_build_allowed": False,
        "true_forward_start": registry["protocol"]["true_forward_start"],
        "branches": branches,
        "governance": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    _atomic_json_write(root / MASTER_REPORT_FILE.relative_to(ROOT), master)
    _atomic_text_write(
        root / OPTIONS_REPORT_FILE.relative_to(ROOT),
        _render_options_report(options, checked_at),
    )
    _atomic_text_write(
        root / CONSTITUENT_REPORT_FILE.relative_to(ROOT),
        _render_constituent_report(constituents, checked_at),
    )
    _atomic_text_write(
        root / EARNINGS_REPORT_FILE.relative_to(ROOT),
        _render_earnings_report(earnings, checked_at),
    )
    reason = (
        "当前没有任何单信号分支通过数据可行性门槛；禁止通过部分日期、幸存合约或"
        "当前数据库回填来生成看似完整的历史结果。"
        if not allowed_hypotheses
        else f"仅候选 {allowed_hypotheses} 获得分支级许可；其他候选仍被阻断。"
    )
    _atomic_text_write(
        root / UNIVARIATE_REPORT_FILE.relative_to(ROOT),
        _render_not_run_report(
            "单变量信号历史伪样本外检验",
            checked_at,
            reason,
            ["BAD20/TAIL20发生率", "Lift与Precision", "Brier、Log Loss和校准曲线"],
        ),
    )
    _atomic_text_write(
        root / INCREMENTAL_REPORT_FILE.relative_to(ROOT),
        _render_not_run_report(
            "相对冻结R6信息集的增量信息审计",
            checked_at,
            "单变量原始预测能力尚未获得数据许可，增量模型没有合法输入。",
            ["R6基准与增量模型比较", "增量Brier/Log Loss", "组合模型与候选排名"],
        ),
    )
    _atomic_text_write(
        root / TIMING_REPORT_FILE.relative_to(ROOT),
        _render_not_run_report(
            "单位动态择时贡献审计",
            checked_at,
            "发现层和增量信息层尚未通过，按协议不得提前映射防御状态。",
            ["单位成本后择时贡献", "同平均仓位和同波动静态基准", "P1仓位曲线与交易记录"],
        ),
    )
    status = {
        "project_id": registry["protocol"]["project_id"],
        "status": "AWAITING_DATA_FEASIBILITY",
        "checked_at": checked_at,
        "historical_contamination_cutoff": registry["protocol"][
            "historical_contamination_cutoff"
        ],
        "true_forward_start": None,
        "eligible_signal_days": 0,
        "matured_20d_outcomes": 0,
        "matured_60d_outcomes": 0,
        "independent_bad_state_episodes": 0,
        "branch_status": master["branch_status"],
        "hypotheses_allowed_for_return_test": allowed_hypotheses,
        "hypotheses_terminated": terminated_hypotheses,
        "output": "NO_VIEW",
        "failure_category": "DATA_FEASIBILITY_NOT_PASSED",
        "message": "未运行收益回测，未生成仓位或订单；等待至少一个分支完整通过。",
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
    }
    _atomic_json_write(root / STATUS_FILE.relative_to(ROOT), status)
    return master


__all__ = [
    "audit_constituents",
    "audit_earnings_consensus",
    "audit_options",
    "run_feasibility_audit",
    "sha256",
    "validate_registry",
]
