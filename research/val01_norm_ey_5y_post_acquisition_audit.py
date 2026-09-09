"""在财务历史扩展成功后重跑 VAL01_NORM_EY_5Y 数据闸门。

本模块只构建正常化盈利审计面板并核对覆盖与独立估值序列，不读取未来收益，
不计算 IC，不生成仓位或订单。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.build_point_in_time_fundamental_panel import select_latest_vintages
from research.diagnose_valuation_negative_ic import (
    aggregate_normalized_snapshot,
    derive_company_normalization,
)
from research.point_in_time_valuation_v2_audit import (
    build_snapshot_coverage,
    load_config as load_base_audit_config,
    summarize_history_gate,
)
from research.point_in_time_valuation_v2_post_acquisition import (
    combined_price_input,
    load_config as load_v1_1_config,
    verify_hashes as verify_v1_1_hashes,
)
from research.val01_norm_ey_5y_financial_extension import (
    CONFIG_FILE,
    ROOT,
    _atomic_parquet,
    _atomic_text,
    load_config,
    sha256_file,
)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def build_normalized_metric_audit_panel(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    index_daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """按固定官方权重快照构建不含收益的正常化盈利数值面板。"""

    weight_data = weights.copy()
    weight_data["trade_date"] = pd.to_datetime(weight_data["trade_date"], errors="coerce")
    weight_data = weight_data.loc[weight_data["trade_date"].between(start, end)].copy()
    financial_data = financials.copy()
    financial_data["report_period"] = pd.to_datetime(
        financial_data["report_period"], errors="coerce"
    )
    financial_data["available_at"] = pd.to_datetime(
        financial_data["available_at"], errors="coerce"
    )
    price_data = prices.copy()
    price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce")
    index_data = index_daily.copy()
    index_data["date"] = pd.to_datetime(index_data["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot_weights in weight_data.groupby("trade_date", sort=True):
        snapshot_date = pd.Timestamp(snapshot_date)
        symbols = set(snapshot_weights["con_code"].dropna().astype(str))
        vintages = select_latest_vintages(
            financial_data.loc[
                financial_data["con_code"].astype(str).isin(symbols)
            ],
            snapshot_date,
        )
        metrics = derive_company_normalization(
            vintages, lookback_years=3, minimum_observations=4
        )
        snapshot_prices = price_data.loc[
            price_data["date"].eq(snapshot_date), ["con_code", "raw_close"]
        ].drop_duplicates("con_code", keep="last")
        index_close = index_data.loc[
            index_data["date"].eq(snapshot_date), "close"
        ]
        if index_close.empty:
            raise ValueError(f"{snapshot_date.date()} 缺少指数收盘价")
        row = aggregate_normalized_snapshot(
            snapshot_weights,
            snapshot_prices,
            metrics,
            snapshot_date,
            float(index_close.iloc[-1]),
        )
        row["known_financial_event_count"] = int(len(vintages))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if len(result) != weight_data["trade_date"].nunique():
        raise ValueError("正常化盈利审计面板快照数与权重快照数不一致")
    return result


def independent_valuation_cross_check(
    normalized_panel: pd.DataFrame,
    official_pe: pd.DataFrame,
    vendor_valuation: pd.DataFrame,
) -> dict[str, Any]:
    """用官方和第三方 PE 做方向交叉核验，不把它们作为模型输入。"""

    panel = normalized_panel[
        ["date", "weighted_normalized_earnings_yield", "component_normalized_pe"]
    ].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    official = official_pe[["date", "pe_official"]].copy()
    official["date"] = pd.to_datetime(official["date"], errors="coerce")
    vendor = vendor_valuation[["date", "pe_ttm"]].copy()
    vendor["date"] = pd.to_datetime(vendor["date"], errors="coerce")
    merged = panel.merge(official, on="date", how="left").merge(
        vendor, on="date", how="left"
    )
    merged["official_earnings_yield"] = np.where(
        merged["pe_official"].gt(0), 1.0 / merged["pe_official"], np.nan
    )
    merged["vendor_earnings_yield"] = np.where(
        merged["pe_ttm"].gt(0), 1.0 / merged["pe_ttm"], np.nan
    )

    def correlation(other: str) -> dict[str, Any]:
        valid = merged[
            ["weighted_normalized_earnings_yield", other]
        ].dropna()
        if len(valid) < 3:
            return {"overlap_rows": len(valid), "spearman_correlation": None}
        rho = spearmanr(
            valid["weighted_normalized_earnings_yield"], valid[other]
        ).statistic
        return {
            "overlap_rows": len(valid),
            "spearman_correlation": _safe_float(rho),
            "first_date": str(
                merged.loc[valid.index, "date"].min().date()
            ),
            "last_date": str(
                merged.loc[valid.index, "date"].max().date()
            ),
        }

    positive = merged["weighted_normalized_earnings_yield"].gt(0)
    return {
        "status": "CROSS_CHECK_ONLY_NON_VINTAGE_NOT_A_SIGNAL_INPUT",
        "snapshot_count": len(merged),
        "positive_normalized_earnings_yield_count": int(positive.sum()),
        "official_inverse_pe": correlation("official_earnings_yield"),
        "vendor_inverse_pe": correlation("vendor_earnings_yield"),
        "governance": "官方与第三方PE均为事后回取，且口径是TTM而非正常化盈利；仅核对尺度和同向性，不设收益含义或放行阈值。",
    }


def run_post_acquisition_audit(
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """核验扩展档案并重跑五年正常化盈利覆盖。"""

    config = config or load_config(CONFIG_FILE)
    acquisition_path = ROOT / config["artifacts"]["acquisition_report_json"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if not acquisition["status"].startswith("PASS_TARGET_SCOPE_"):
        raise RuntimeError(
            f"财务扩展采集尚未通过：{acquisition['status']}"
        )
    extension_path = ROOT / acquisition["extension_archive"]["file"]
    extended_path = ROOT / acquisition["extended_archive"]["file"]
    if sha256_file(extension_path) != acquisition["extension_archive"]["sha256"]:
        raise RuntimeError("扩展事件文件哈希与采集报告不一致")
    if sha256_file(extended_path) != acquisition["extended_archive"]["sha256"]:
        raise RuntimeError("合并财务档案哈希与采集报告不一致")
    v1_1_config = load_v1_1_config()
    if verify_v1_1_hashes(v1_1_config)["status"] != "PASS":
        raise RuntimeError("V1.1 冻结输入哈希发生漂移")
    contracts = v1_1_config["data_contracts"]
    read = lambda name: pd.read_parquet(ROOT / contracts[name]["file"])
    weights = read("historical_weights")
    current_prices = read("current_constituent_daily")
    snapshot_prices = read("snapshot_prices")
    index_daily = read("index_daily")
    bonds = read("government_bond_yields")
    prices = combined_price_input(snapshot_prices, current_prices)
    extended_financials = pd.read_parquet(extended_path)
    base_config = load_base_audit_config()
    coverage = build_snapshot_coverage(
        weights,
        extended_financials,
        prices,
        index_daily,
        bonds,
        base_config,
    )
    history = summarize_history_gate(coverage, base_config)
    first = pd.Timestamp(
        f"{config['normalization_contract']['first_required_signal_month']}-01"
    )
    last_period = pd.Period(
        config["normalization_contract"]["last_required_signal_month"], freq="M"
    )
    last = weights.loc[
        pd.to_datetime(weights["trade_date"]).dt.to_period("M").eq(last_period),
        "trade_date",
    ].max()
    normalized_panel = build_normalized_metric_audit_panel(
        weights,
        extended_financials,
        prices,
        index_daily,
        first,
        pd.Timestamp(last),
    )
    base_contracts = base_config["data_contracts"]
    official_pe = pd.read_parquet(ROOT / base_contracts["official_pe"]["file"])
    vendor = pd.read_parquet(ROOT / base_contracts["vendor_valuation"]["file"])
    cross_check = independent_valuation_cross_check(
        normalized_panel, official_pe, vendor
    )
    five = history["five_year"]
    norm_pass = five["normalized_ey_status"] == "PASS_LOCAL_RECONSTRUCTIBLE"
    spread_pass = five["normalized_ey_spread_status"] == "PASS_LOCAL_RECONSTRUCTIBLE"
    status = (
        "PASS_VAL01_NORM_EY_5Y_AND_VAL02_SPREAD_DATA_ONLY"
        if norm_pass and spread_pass
        else "BLOCKED_NORMALIZED_EARNINGS_COVERAGE"
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": "1.2.0",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "overall_status": status,
        "formal_return_evaluation_status": (
            "MODEL_CARD_FREEZE_ALLOWED_NO_RETURN_TEST_YET" if norm_pass else "NO_VIEW"
        ),
        "branch_status": {
            "VAL01_RAW_EY_5Y": "REJECTED_PREDICTIVE_SCREEN_NOT_REOPENED",
            "VAL01_NORM_EY_5Y": (
                "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY"
                if norm_pass else five["normalized_ey_status"]
            ),
            "VAL02_NORM_EY_SPREAD_5Y": (
                "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY"
                if spread_pass else five["normalized_ey_spread_status"]
            ),
            "VAL01_NORM_EY_7Y": history["seven_year"]["normalized_ey_status"],
            "VAL02_NORM_EY_SPREAD_7Y": history["seven_year"]["normalized_ey_spread_status"],
        },
        "history_gates": history,
        "financial_extension": acquisition,
        "normalized_metric_panel": {
            "snapshot_count": len(normalized_panel),
            "first_date": str(normalized_panel["date"].min().date()),
            "last_date": str(normalized_panel["date"].max().date()),
            "minimum_normalized_earnings_weight_coverage": _safe_float(
                normalized_panel["normalized_earnings_weight_coverage"].min()
            ),
            "maximum_normalized_earnings_weight_coverage": _safe_float(
                normalized_panel["normalized_earnings_weight_coverage"].max()
            ),
        },
        "independent_cross_check": cross_check,
        "archive_caveat": acquisition["archive_caveat"],
        "governance": {
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
            "broker_connection_performed": False,
            "raw_ey_rejected_branch_reopened": False,
        },
        "artifacts": {
            "coverage": config["artifacts"]["post_acquisition_coverage"],
            "normalized_metric_audit_panel": config["artifacts"]["normalized_metric_audit_panel"],
            "report_json": config["artifacts"]["post_acquisition_report_json"],
            "report_markdown": config["artifacts"]["post_acquisition_report_markdown"],
        },
    }
    return report, coverage, normalized_panel


def render_markdown(report: dict[str, Any]) -> str:
    five = report["history_gates"]["five_year"]
    panel = report["normalized_metric_panel"]
    cross = report["independent_cross_check"]
    return "\n".join(
        [
            "# 沪深300点时估值 V2 可重建性审计 V1.2",
            "",
            f"> 状态：`{report['overall_status']}`。本报告只审计正常化盈利数据，不包含未来收益、IC、仓位或订单。",
            "",
            "## 五年覆盖闸门",
            "",
            f"- 标准化 EY：`{report['branch_status']['VAL01_NORM_EY_5Y']}`。",
            f"- 标准化 EY 利差：`{report['branch_status']['VAL02_NORM_EY_SPREAD_5Y']}`。",
            f"- 覆盖失败月份：{five['normalized_coverage_failed_months']}。",
            f"- 正常化盈利最低/最高权重覆盖：{panel['minimum_normalized_earnings_weight_coverage']:.4%}/{panel['maximum_normalized_earnings_weight_coverage']:.4%}。",
            "",
            "## 独立交叉核验",
            "",
            f"- 官方 PE 倒数与正常化 EY 的 Spearman 相关：{cross['official_inverse_pe']['spearman_correlation']}。",
            f"- 第三方 PE 倒数与正常化 EY 的 Spearman 相关：{cross['vendor_inverse_pe']['spearman_correlation']}。",
            f"- 边界：{cross['governance']}",
            "",
            "## 治理",
            "",
            "- 原始 EY 分支已在预测筛选中拒绝，本次补数不会重新打开该分支。",
            "- 若五年覆盖通过，下一步只允许冻结 VAL01_NORM_EY_5Y 模型卡；模型卡冻结前仍不看未来收益。",
            f"- {report['archive_caveat']}",
            "",
        ]
    )


def write_artifacts(
    report: dict[str, Any],
    coverage: pd.DataFrame,
    normalized_panel: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> None:
    config = config or load_config(CONFIG_FILE)
    artifacts = config["artifacts"]
    coverage_path = ROOT / artifacts["post_acquisition_coverage"]
    panel_path = ROOT / artifacts["normalized_metric_audit_panel"]
    json_path = ROOT / artifacts["post_acquisition_report_json"]
    markdown_path = ROOT / artifacts["post_acquisition_report_markdown"]
    _atomic_parquet(coverage, coverage_path)
    _atomic_parquet(normalized_panel, panel_path)
    report["artifacts"]["coverage_sha256"] = sha256_file(coverage_path)
    report["artifacts"]["normalized_metric_audit_panel_sha256"] = sha256_file(panel_path)
    _atomic_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), json_path
    )
    _atomic_text(render_markdown(report), markdown_path)
