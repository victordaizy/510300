"""VAL-04 PB—ROE 残余收益候选的无收益数据可行性审计。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.build_point_in_time_fundamental_panel import (
    derive_company_metrics,
    select_latest_vintages,
)
from research.index_driver_attribution_v1_2 import (
    truncate_superseded_l1_intervals,
)
from research.index_driver_attribution_v1_3 import (
    restrict_industry_intervals_to_research_window,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "val04_pb_roe_residual_data_feasibility_v1.yaml"


@dataclass(frozen=True)
class FeasibilityResult:
    """完整公司截面、逐月审计和总状态。"""

    constituent_panel: pd.DataFrame
    snapshot_audit: pd.DataFrame
    report: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("VAL-04 数据协议不是映射")
    return config


def verify_input_hashes(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / str(contract["file"])
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "dataset": name,
                "file": str(contract["file"]),
                "expected_sha256": str(contract["sha256"]),
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "FAILED",
        "files": rows,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}缺少字段：{missing}")


def prepare_snapshot_prices(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """只保留快照日价格，重叠时以专用历史快照为准。"""

    required = {"date", "con_code", "raw_close"}
    _require_columns(historical, required, "历史快照价格")
    _require_columns(current, required, "当前成分日线")
    frames: list[pd.DataFrame] = []
    for source_rank, source in ((0, current), (1, historical)):
        data = source[["date", "con_code", "raw_close"]].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        data = data.loc[data["date"].isin(snapshot_dates)].copy()
        data["source_rank"] = source_rank
        frames.append(data)
    combined = pd.concat(frames, ignore_index=True)
    combined["raw_close"] = pd.to_numeric(combined["raw_close"], errors="coerce")
    if combined[["date", "con_code"]].isna().any().any():
        raise ValueError("快照价格存在空日期或证券代码")
    return (
        combined.sort_values(["date", "con_code", "source_rank"])
        .drop_duplicates(["date", "con_code"], keep="last")
        .drop(columns="source_rank")
        .reset_index(drop=True)
    )


def prepare_industry_intervals(
    intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """限定研究窗口并按既有冻结规则消除跨一级行业重叠。"""

    required = {
        "con_code",
        "industry_l1",
        "in_date",
        "out_date",
        "classification_usage",
        "source",
    }
    _require_columns(intervals, required, "中信行业区间")
    usage = intervals["classification_usage"].astype(str)
    if not usage.str.contains("POINT_IN_TIME", case=False, na=False).all():
        raise ValueError("行业区间包含非点时分类")
    relevant = restrict_industry_intervals_to_research_window(intervals, start, end)
    resolved = truncate_superseded_l1_intervals(relevant)
    resolved["in_date"] = pd.to_datetime(resolved["in_date"], errors="coerce")
    resolved["out_date"] = pd.to_datetime(resolved["out_date"], errors="coerce")
    return resolved


def active_industry_for_date(
    intervals: pd.DataFrame, date: pd.Timestamp
) -> pd.DataFrame:
    """选择信号日有效行业；重复有效分类为硬错误。"""

    active = intervals.loc[
        intervals["in_date"].le(date)
        & (intervals["out_date"].isna() | intervals["out_date"].gt(date)),
        ["con_code", "industry_l1"],
    ].copy()
    duplicates = active.loc[active["con_code"].duplicated(False), "con_code"]
    if not duplicates.empty:
        raise ValueError(
            f"{date.date()}存在重复有效行业：{sorted(duplicates.astype(str).unique())[:10]}"
        )
    return active


def design_matrix_audit(complete: pd.DataFrame) -> dict[str, int]:
    """只审计未来模型设计矩阵的秩，不拟合系数或残差。"""

    if complete.empty:
        return {
            "design_row_count": 0,
            "design_column_count": 0,
            "design_rank": 0,
            "design_residual_degrees_of_freedom": 0,
        }
    sector = pd.get_dummies(
        complete["industry_l1"].astype(str), dtype=float
    ).sort_index(axis=1)
    numeric = pd.DataFrame(
        {
            "roe_clipped": complete["ttm_roe"].clip(-0.5, 0.8).astype(float),
            "revenue_growth_clipped": complete["ttm_revenue_growth_yoy"]
            .clip(-1.0, 1.0)
            .astype(float),
        },
        index=complete.index,
    )
    matrix = pd.concat([sector, numeric], axis=1).to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "design_row_count": int(matrix.shape[0]),
        "design_column_count": int(matrix.shape[1]),
        "design_rank": rank,
        "design_residual_degrees_of_freedom": int(matrix.shape[0] - rank),
    }


def _weight_coverage(panel: pd.DataFrame, mask: pd.Series) -> float:
    return float(panel.loc[mask.fillna(False), "snapshot_weight"].sum())


def audit_snapshot(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """计算单月覆盖和设计矩阵闸门。"""

    gates = config["quality_gates"]
    data = panel.copy()
    finite_columns = ["book_to_price", "ttm_roe", "ttm_revenue_growth_yoy"]
    for column in finite_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    flags = {
        "price": data["raw_close"].gt(0),
        "industry": data["industry_l1"].notna(),
        "book_to_price": data["book_to_price"].gt(0),
        "roe": data["ttm_roe"].notna(),
        "revenue_growth": data["ttm_revenue_growth_yoy"].notna(),
    }
    complete_mask = pd.concat(flags, axis=1).all(axis=1)
    data["complete_case"] = complete_mask
    complete = data.loc[complete_mask].copy()
    design = design_matrix_audit(complete)
    coverages = {
        f"{name}_weight_coverage": _weight_coverage(data, mask)
        for name, mask in flags.items()
    }
    complete_coverage = _weight_coverage(data, complete_mask)
    member_count = int(data["con_code"].nunique())
    weight_sum = float(data["snapshot_weight"].sum())
    active_industry_count = int(complete["industry_l1"].nunique())
    duplicate_weight_count = int(data["con_code"].duplicated().sum())
    future_financial_count = int(
        (
            data["selected_available_at"].notna()
            & data["selected_available_at"].gt(date)
        ).sum()
    )
    failures: list[str] = []
    member_range = gates["member_count_range"]
    weight_range = gates["weight_sum_range"]
    checks = {
        "MEMBER_COUNT": int(member_range[0]) <= member_count <= int(member_range[1]),
        "WEIGHT_SUM": float(weight_range[0]) <= weight_sum <= float(weight_range[1]),
        "PRICE_COVERAGE": coverages["price_weight_coverage"]
        >= float(gates["minimum_price_weight_coverage"]),
        "INDUSTRY_COVERAGE": coverages["industry_weight_coverage"]
        >= float(gates["minimum_industry_weight_coverage"]),
        "BOOK_TO_PRICE_COVERAGE": coverages["book_to_price_weight_coverage"]
        >= float(gates["minimum_book_to_price_weight_coverage"]),
        "ROE_COVERAGE": coverages["roe_weight_coverage"]
        >= float(gates["minimum_roe_weight_coverage"]),
        "REVENUE_GROWTH_COVERAGE": coverages["revenue_growth_weight_coverage"]
        >= float(gates["minimum_revenue_growth_weight_coverage"]),
        "COMPLETE_CASE_COVERAGE": complete_coverage
        >= float(gates["minimum_complete_case_weight_coverage"]),
        "COMPLETE_COMPANY_COUNT": len(complete)
        >= int(gates["minimum_complete_company_count"]),
        "ACTIVE_INDUSTRY_COUNT": active_industry_count
        >= int(gates["minimum_active_industry_count"]),
        "DESIGN_RESIDUAL_DOF": design["design_residual_degrees_of_freedom"]
        >= int(gates["minimum_design_residual_degrees_of_freedom"]),
        "DUPLICATE_WEIGHT_SECURITY": duplicate_weight_count
        == int(gates["duplicate_weight_security_count"]),
        "FUTURE_FINANCIAL_EVENT": future_financial_count
        == int(gates["future_financial_event_count"]),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "date": date,
        "member_count": member_count,
        "weight_sum": weight_sum,
        **coverages,
        "complete_case_weight_coverage": complete_coverage,
        "complete_company_count": int(len(complete)),
        "active_industry_count": active_industry_count,
        **design,
        "duplicate_weight_security_count": duplicate_weight_count,
        "duplicate_active_industry_count": 0,
        "future_financial_event_count": future_financial_count,
        "output": "DATA_READY" if not failures else "NO_VIEW",
        "failure_category": "PASS" if not failures else "|".join(failures),
    }


def _selected_available_at(vintages: pd.DataFrame) -> pd.DataFrame:
    if vintages.empty:
        return pd.DataFrame(columns=["con_code", "selected_available_at"])
    latest = (
        vintages.sort_values(["con_code", "report_period", "available_at"])
        .groupby("con_code", as_index=False)
        .tail(1)[["con_code", "available_at"]]
        .rename(columns={"available_at": "selected_available_at"})
    )
    return latest


def build_feasibility(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    historical_prices: pd.DataFrame,
    current_prices: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    config: Mapping[str, Any],
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> FeasibilityResult:
    """构造 120 个月无未来收益的公司级可行性面板。"""

    _require_columns(weights, {"con_code", "trade_date", "weight"}, "官方权重")
    weight_data = weights.copy()
    weight_data["trade_date"] = pd.to_datetime(
        weight_data["trade_date"], errors="coerce"
    ).dt.normalize()
    weight_data["weight"] = pd.to_numeric(weight_data["weight"], errors="coerce")
    start = pd.Timestamp(config["research_window"]["snapshot_start"])
    end = pd.Timestamp(config["research_window"]["snapshot_end"])
    weight_data = weight_data.loc[weight_data["trade_date"].between(start, end)].copy()
    if weight_data[["trade_date", "con_code", "weight"]].isna().any().any():
        raise ValueError("官方权重存在空日期、证券或权重")
    snapshot_dates = pd.DatetimeIndex(sorted(weight_data["trade_date"].unique()))
    prices = prepare_snapshot_prices(historical_prices, current_prices, snapshot_dates)
    price_by_date = {
        pd.Timestamp(date): frame[["con_code", "raw_close"]].copy()
        for date, frame in prices.groupby("date", sort=False)
    }
    intervals = prepare_industry_intervals(industry_intervals, start, end)
    events = financials.copy()
    events["report_period"] = pd.to_datetime(events["report_period"], errors="coerce")
    events["available_at"] = pd.to_datetime(events["available_at"], errors="coerce")
    if events[["con_code", "report_period", "available_at"]].isna().any().any():
        raise ValueError("财务档案存在空证券、报告期或可得日")

    cross_sections: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    total = len(snapshot_dates)
    for number, (snapshot_date, snapshot) in enumerate(
        weight_data.groupby("trade_date", sort=True), start=1
    ):
        date = pd.Timestamp(snapshot_date)
        snapshot = snapshot[["con_code", "weight"]].copy()
        snapshot["snapshot_weight"] = snapshot["weight"] / 100.0
        symbols = set(snapshot["con_code"].astype(str))
        vintages = select_latest_vintages(
            events.loc[events["con_code"].astype(str).isin(symbols)], date
        )
        metrics = derive_company_metrics(vintages)
        available = _selected_available_at(vintages)
        active = active_industry_for_date(intervals, date)
        price = price_by_date.get(
            date, pd.DataFrame(columns=["con_code", "raw_close"])
        )
        cross = snapshot.merge(price, on="con_code", how="left", validate="one_to_one")
        cross = cross.merge(metrics, on="con_code", how="left", validate="one_to_one")
        cross = cross.merge(available, on="con_code", how="left", validate="one_to_one")
        cross = cross.merge(active, on="con_code", how="left", validate="one_to_one")
        cross["date"] = date
        cross["book_to_price"] = cross["book_value_per_share"] / cross["raw_close"]
        cross["price_to_book"] = cross["raw_close"] / cross["book_value_per_share"]
        cross["log_price_to_book"] = np.log(
            cross["price_to_book"].where(cross["price_to_book"].gt(0))
        )
        ordered = [
            "date",
            "con_code",
            "snapshot_weight",
            "raw_close",
            "industry_l1",
            "latest_report_period",
            "selected_available_at",
            "book_value_per_share",
            "book_to_price",
            "price_to_book",
            "log_price_to_book",
            "ttm_roe",
            "ttm_revenue_growth_yoy",
            "ttm_profit_growth_yoy",
        ]
        cross = cross[ordered].copy()
        flags = pd.DataFrame(
            {
                "price": cross["raw_close"].gt(0),
                "industry": cross["industry_l1"].notna(),
                "book_to_price": cross["book_to_price"].gt(0),
                "roe": cross["ttm_roe"].replace([np.inf, -np.inf], np.nan).notna(),
                "revenue_growth": cross["ttm_revenue_growth_yoy"]
                .replace([np.inf, -np.inf], np.nan)
                .notna(),
            }
        )
        cross["complete_case"] = flags.all(axis=1)
        cross_sections.append(cross)
        audit_rows.append(audit_snapshot(cross, date, config))
        if progress is not None:
            progress(number, total, date)

    constituent_panel = pd.concat(cross_sections, ignore_index=True)
    snapshot_audit = pd.DataFrame(audit_rows).sort_values("date").reset_index(drop=True)
    expected = int(config["research_window"]["expected_snapshot_count"])
    all_pass = bool(
        len(snapshot_audit) == expected
        and snapshot_audit["output"].eq("DATA_READY").all()
    )
    hash_audit = verify_input_hashes(config)
    all_pass &= hash_audit["status"] == "PASS"
    status = (
        config["next_stage_gate"]["pass_status"]
        if all_pass
        else config["next_stage_gate"]["fail_status"]
    )
    numeric_summary_columns = [
        "price_weight_coverage",
        "industry_weight_coverage",
        "book_to_price_weight_coverage",
        "roe_weight_coverage",
        "revenue_growth_weight_coverage",
        "complete_case_weight_coverage",
        "complete_company_count",
        "active_industry_count",
        "design_residual_degrees_of_freedom",
    ]
    summary = {
        column: {
            "minimum": float(snapshot_audit[column].min()),
            "median": float(snapshot_audit[column].median()),
            "maximum": float(snapshot_audit[column].max()),
        }
        for column in numeric_summary_columns
    }
    failed_rows = snapshot_audit.loc[snapshot_audit["output"].ne("DATA_READY")]
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "status": status,
        "hash_audit": hash_audit,
        "window": {
            "first_snapshot": str(snapshot_audit["date"].min().date()),
            "last_snapshot": str(snapshot_audit["date"].max().date()),
            "snapshot_count": int(len(snapshot_audit)),
            "passing_snapshot_count": int(snapshot_audit["output"].eq("DATA_READY").sum()),
            "failed_snapshot_count": int(len(failed_rows)),
        },
        "coverage_summary": summary,
        "failure_categories": {
            str(key): int(value)
            for key, value in failed_rows["failure_category"].value_counts().items()
        },
        "failed_snapshots": [
            {
                "date": str(row.date.date()),
                "failure_category": str(row.failure_category),
            }
            for row in failed_rows.itertuples(index=False)
        ],
        "candidate_definition": config["frozen_candidate_definition"],
        "governance": {
            "future_returns_read": False,
            "model_fitted": False,
            "residual_signal_calculated": False,
            "ic_calculated": False,
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
        },
        "meaning": "仅为数据可行性结论，不是预测、仓位或买卖指令。",
    }
    return FeasibilityResult(constituent_panel, snapshot_audit, report)


def render_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage_summary"]
    window = report["window"]
    lines = [
        "# VAL-04 PB—ROE 残余收益数据可行性报告 V1",
        "",
        f"> 状态：`{report['status']}`。本报告没有读取未来收益、拟合模型或生成仓位。",
        "",
        "## 窗口结论",
        "",
        f"- 快照：{window['first_snapshot']} 至 {window['last_snapshot']}，共 {window['snapshot_count']} 个月。",
        f"- 通过：{window['passing_snapshot_count']}；失败：{window['failed_snapshot_count']}。",
        f"- 输入哈希：`{report['hash_audit']['status']}`。",
        "",
        "## 全窗口覆盖",
        "",
        "|指标|最低|中位数|最高|",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "price_weight_coverage": "价格权重覆盖",
        "industry_weight_coverage": "点时行业权重覆盖",
        "book_to_price_weight_coverage": "正B/P权重覆盖",
        "roe_weight_coverage": "ROE权重覆盖",
        "revenue_growth_weight_coverage": "营收同比权重覆盖",
        "complete_case_weight_coverage": "完整样本权重覆盖",
        "complete_company_count": "完整样本公司数",
        "active_industry_count": "完整样本行业数",
        "design_residual_degrees_of_freedom": "设计矩阵残差自由度",
    }
    percentage = {
        "price_weight_coverage",
        "industry_weight_coverage",
        "book_to_price_weight_coverage",
        "roe_weight_coverage",
        "revenue_growth_weight_coverage",
        "complete_case_weight_coverage",
    }
    for key, label in labels.items():
        row = coverage[key]
        if key in percentage:
            values = [f"{100 * row[name]:.4f}%" for name in ("minimum", "median", "maximum")]
        else:
            values = [f"{row[name]:.0f}" for name in ("minimum", "median", "maximum")]
        lines.append(f"|{label}|{values[0]}|{values[1]}|{values[2]}|")
    lines.extend(["", "## 失败快照", ""])
    if report["failed_snapshots"]:
        lines.extend(
            f"- {row['date']}：`{row['failure_category']}`"
            for row in report["failed_snapshots"]
        )
    else:
        lines.append("- 无。120/120 个快照全部通过预冻结数据门槛。")
    lines.extend(
        [
            "",
            "## 治理边界",
            "",
            "- 未读取未来收益，未计算 IC。",
            "- 未拟合横截面回归，未计算残差或估值分位。",
            "- 未计算策略收益，未生成仓位、目标股数或订单。",
            "- 本结论只决定是否允许另行冻结模型卡。",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_artifacts(
    result: FeasibilityResult, config: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    constituent_path = ROOT / artifacts["constituent_panel"]
    snapshot_path = ROOT / artifacts["snapshot_audit"]
    json_path = ROOT / artifacts["status_json"]
    markdown_path = ROOT / artifacts["status_markdown"]
    for path in (constituent_path, snapshot_path, json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.constituent_panel.to_parquet(constituent_path, index=False)
    result.snapshot_audit.to_parquet(snapshot_path, index=False)
    _atomic_text(
        json_path, json.dumps(result.report, ensure_ascii=False, indent=2) + "\n"
    )
    _atomic_text(markdown_path, render_markdown(result.report))
    inventory = {}
    for name, path in (
        ("constituent_panel", constituent_path),
        ("snapshot_audit", snapshot_path),
        ("status_json", json_path),
        ("status_markdown", markdown_path),
    ):
        inventory[name] = {
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": result.report["status"],
        "generated_at": result.report["generated_at"],
        "artifacts": inventory,
        "input_hash_audit": result.report["hash_audit"],
        "window": result.report["window"],
        "governance": result.report["governance"],
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    manifest_path = ROOT / artifacts["result_manifest"]
    _atomic_text(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "artifacts": inventory,
        "result_manifest": {
            "file": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(manifest_path),
            "content_sha256": manifest["manifest_content_sha256"],
        },
    }


__all__ = [
    "FeasibilityResult",
    "active_industry_for_date",
    "audit_snapshot",
    "build_feasibility",
    "canonical_hash",
    "design_matrix_audit",
    "load_config",
    "prepare_industry_intervals",
    "prepare_snapshot_prices",
    "render_markdown",
    "sha256_file",
    "verify_input_hashes",
    "write_artifacts",
]
