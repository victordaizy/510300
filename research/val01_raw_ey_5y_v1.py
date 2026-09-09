"""构建VAL01原始盈利收益率五年分位的无收益标签月度输入。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.build_point_in_time_fundamental_panel import (
    derive_company_metrics,
    select_latest_vintages,
)
from research.point_in_time_valuation_price_acquisition import normalize_current_history


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "val01_raw_ey_5y_v1.yaml"

FORBIDDEN_OUTPUT_TOKENS = (
    "future_return",
    "forward_return",
    "information_coefficient",
    "target_position",
    "position_size",
    "target_shares",
    "order_quantity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    forbidden_flags = (
        "future_return_label_generation_enabled",
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    )
    enabled = [name for name in forbidden_flags if config["protocol"].get(name)]
    if enabled:
        raise ValueError(f"模型卡配置错误，禁止项被启用：{enabled}")
    if config["percentile_definition"]["algorithm"] != "EXACT_EMPIRICAL_MIDRANK":
        raise ValueError("只允许已冻结的精确midrank算法")
    return config


def verify_input_hashes(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "dataset": name,
                "file": contract["file"],
                "expected_sha256": contract["sha256"],
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "BLOCKED_HASH_DRIFT",
        "rows": rows,
    }


def _normalize_weights(weights: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {"trade_date", "con_code", "weight"}
    if missing := required.difference(weights.columns):
        raise ValueError(f"权重表缺少字段：{sorted(missing)}")
    data = weights.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    definition = config["signal_definition"]
    data = data.loc[
        data["trade_date"].between(
            pd.Timestamp(definition["snapshot_start"]),
            pd.Timestamp(definition["snapshot_end"]),
        )
    ].copy()
    if data[["trade_date", "con_code", "weight"]].isna().any().any():
        raise ValueError("权重表存在无效日期、证券代码或权重")
    if data.duplicated(["trade_date", "con_code"]).any():
        raise ValueError("权重表存在重复证券快照")
    counts = data.groupby("trade_date")["con_code"].nunique()
    expected_count = int(definition["expected_snapshot_count"])
    expected_members = int(definition["expected_constituents_per_snapshot"])
    if len(counts) != expected_count or not counts.eq(expected_members).all():
        raise ValueError(
            f"权重快照结构不符：月份数{len(counts)}，成分数范围{counts.min()}—{counts.max()}"
        )
    periods = pd.PeriodIndex(counts.index, freq="M")
    expected_periods = pd.period_range(periods.min(), periods.max(), freq="M")
    if not periods.equals(expected_periods):
        raise ValueError("权重快照月份不连续")
    return data.sort_values(["trade_date", "con_code"]).reset_index(drop=True)


def build_full_snapshot_prices(
    weights: pd.DataFrame,
    snapshot_prices: pd.DataFrame,
    current_daily: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """合并早期专用价格快照与后期成分日线，且保留真实成交日年龄。"""

    weight_data = _normalize_weights(weights, config)
    required_early = {
        "date",
        "con_code",
        "weight_pct",
        "raw_close",
        "price_trade_date",
        "price_age_calendar_days",
        "is_suspended_or_stale",
        "source",
    }
    if missing := required_early.difference(snapshot_prices.columns):
        raise ValueError(f"早期价格快照缺少字段：{sorted(missing)}")
    early = snapshot_prices.copy()
    early["date"] = pd.to_datetime(early["date"], errors="coerce").dt.normalize()
    early["price_trade_date"] = pd.to_datetime(
        early["price_trade_date"], errors="coerce"
    ).dt.normalize()
    early_end = early["date"].max()
    if pd.isna(early_end):
        raise ValueError("早期价格快照为空")
    early = early[
        [
            "date",
            "con_code",
            "weight_pct",
            "raw_close",
            "price_trade_date",
            "price_age_calendar_days",
            "is_suspended_or_stale",
            "source",
        ]
    ].copy()
    current = normalize_current_history(current_daily)
    current["date"] = pd.to_datetime(current["date"], errors="coerce").dt.normalize()
    current["price_trade_date"] = pd.to_datetime(
        current["price_trade_date"], errors="coerce"
    ).dt.normalize()
    current = current.loc[current["date"].gt(early_end)].copy()
    current["price_age_calendar_days"] = (
        current["date"] - current["price_trade_date"]
    ).dt.days
    current = current.rename(columns={"is_suspended": "is_suspended_or_stale"})
    current["weight_pct"] = np.nan
    current = current[
        [
            "date",
            "con_code",
            "weight_pct",
            "raw_close",
            "price_trade_date",
            "price_age_calendar_days",
            "is_suspended_or_stale",
            "source",
        ]
    ]
    price_rows = pd.concat([early, current], ignore_index=True)
    price_rows = price_rows.loc[
        price_rows["date"].isin(weight_data["trade_date"].unique())
    ].drop_duplicates(["date", "con_code"], keep="last")
    panel = weight_data.rename(columns={"trade_date": "date", "weight": "official_weight_pct"}).merge(
        price_rows,
        on=["date", "con_code"],
        how="left",
        validate="one_to_one",
    )
    early_compare = panel.loc[panel["date"].le(early_end) & panel["weight_pct"].notna()]
    if not np.allclose(
        early_compare["official_weight_pct"],
        early_compare["weight_pct"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("早期价格快照内嵌权重与冻结官方权重不一致")
    panel["raw_close"] = pd.to_numeric(panel["raw_close"], errors="coerce")
    if (panel["price_trade_date"] > panel["date"]).any():
        raise ValueError("价格面板使用了快照日之后的成交价")
    panel["is_suspended_or_stale"] = panel["is_suspended_or_stale"].fillna(False).astype(bool)
    return panel.sort_values(["date", "con_code"]).reset_index(drop=True)


def empirical_midrank_percentile(window: pd.Series) -> float:
    """按冻结公式计算窗口最后一个值的精确经验midrank分位。"""

    values = pd.to_numeric(window, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        return np.nan
    current = values[-1]
    return float((np.count_nonzero(values < current) + 0.5 * np.count_nonzero(values == current)) / len(values))


def apply_frozen_percentile(
    signals: pd.DataFrame,
    window_months: int,
) -> pd.DataFrame:
    result = signals.copy()
    source = result["weighted_raw_earnings_yield"].where(result["input_status"].eq("PASS"))
    observation_counts: list[int] = []
    percentiles: list[float] = []
    for index in range(len(result)):
        start = max(0, index - window_months + 1)
        window = source.iloc[start : index + 1]
        observation_counts.append(int(window.notna().sum()))
        if len(window) == window_months and window.notna().all():
            percentiles.append(empirical_midrank_percentile(window))
        else:
            percentiles.append(np.nan)
    result["percentile_window_observations"] = observation_counts
    result["raw_ey_percentile_60m"] = percentiles
    result["percentile_ready"] = result["raw_ey_percentile_60m"].notna()
    result["percentile_status"] = np.where(result["percentile_ready"], "READY", "WARMUP")
    return result


def _next_trading_day_map(snapshot_dates: pd.Series, index_daily: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    if "date" not in index_daily:
        raise ValueError("指数交易日历缺少date")
    calendar = np.sort(
        pd.to_datetime(index_daily["date"], errors="coerce").dropna().dt.normalize().unique()
    )
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for value in pd.to_datetime(snapshot_dates, errors="coerce"):
        date = pd.Timestamp(value).normalize()
        location = int(np.searchsorted(calendar, np.datetime64(date), side="right"))
        mapping[date] = pd.Timestamp(calendar[location]) if location < len(calendar) else pd.NaT
    return mapping


def build_signal_inputs(
    weights: pd.DataFrame,
    financials: pd.DataFrame,
    snapshot_prices: pd.DataFrame,
    current_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    config: dict[str, Any],
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> pd.DataFrame:
    """构造120个月聚合EY及60月分位，不读取或生成任何未来收益。"""

    price_panel = build_full_snapshot_prices(weights, snapshot_prices, current_daily, config)
    financial_data = financials.copy()
    financial_data["report_period"] = pd.to_datetime(
        financial_data["report_period"], errors="coerce"
    ).dt.normalize()
    financial_data["available_at"] = pd.to_datetime(
        financial_data["available_at"], errors="coerce"
    ).dt.normalize()
    definition = config["signal_definition"]
    dates = sorted(price_panel["date"].unique())
    rows: list[dict[str, Any]] = []
    for number, snapshot_value in enumerate(dates, start=1):
        snapshot_date = pd.Timestamp(snapshot_value)
        snapshot = price_panel.loc[price_panel["date"].eq(snapshot_date)].copy()
        symbols = set(snapshot["con_code"].astype(str))
        known = select_latest_vintages(
            financial_data.loc[financial_data["con_code"].astype(str).isin(symbols)],
            snapshot_date,
        )
        if not known.empty and (known["available_at"] > snapshot_date).any():
            raise ValueError(f"{snapshot_date.date()}财务版本发生前视")
        metrics = derive_company_metrics(known)
        panel = snapshot.merge(metrics, on="con_code", how="left", validate="one_to_one")
        panel["weight"] = panel["official_weight_pct"] / 100.0
        valid_price = panel["raw_close"].notna() & panel["raw_close"].gt(0)
        valid_ttm = panel["ttm_eps"].notna()
        valid_joint = valid_price & valid_ttm & panel["weight"].gt(0)
        panel["company_raw_earnings_yield"] = panel["ttm_eps"] / panel["raw_close"]
        price_coverage = float(panel.loc[valid_price, "weight"].sum())
        ttm_coverage = float(panel.loc[valid_ttm, "weight"].sum())
        joint_coverage = float(panel.loc[valid_joint, "weight"].sum())
        weighted_raw_ey = (
            float(
                np.average(
                    panel.loc[valid_joint, "company_raw_earnings_yield"],
                    weights=panel.loc[valid_joint, "weight"],
                )
            )
            if valid_joint.any() and joint_coverage > 0
            else np.nan
        )
        status = "PASS"
        if price_coverage < float(definition["minimum_price_weight_coverage"]):
            status = "BLOCKED_PRICE_COVERAGE"
        elif ttm_coverage < float(definition["minimum_ttm_earnings_weight_coverage"]):
            status = "BLOCKED_TTM_COVERAGE"
        elif joint_coverage < float(definition["minimum_joint_weight_coverage"]):
            status = "BLOCKED_JOINT_COVERAGE"
        if status != "PASS":
            weighted_raw_ey = np.nan
        stale = valid_price & panel["is_suspended_or_stale"]
        negative = valid_joint & panel["ttm_net_profit_parent_cny"].lt(0)
        rows.append(
            {
                "date": snapshot_date,
                "signal_observation_date": snapshot_date,
                "constituent_count": int(panel["con_code"].nunique()),
                "official_weight_sum": float(panel["weight"].sum()),
                "price_weight_coverage": price_coverage,
                "ttm_earnings_weight_coverage": ttm_coverage,
                "joint_valid_weight_coverage": joint_coverage,
                "valid_price_constituent_count": int(valid_price.sum()),
                "valid_ttm_constituent_count": int(valid_ttm.sum()),
                "valid_joint_constituent_count": int(valid_joint.sum()),
                "negative_ttm_profit_constituent_count": int(negative.sum()),
                "negative_ttm_profit_weight_share": float(panel.loc[negative, "weight"].sum()),
                "stale_price_constituent_count": int(stale.sum()),
                "stale_price_weight_share": float(panel.loc[stale, "weight"].sum()),
                "maximum_price_age_calendar_days": (
                    int(panel.loc[valid_price, "price_age_calendar_days"].max())
                    if valid_price.any()
                    else np.nan
                ),
                "latest_price_trade_date": panel.loc[valid_price, "price_trade_date"].max(),
                "latest_financial_available_at": known["available_at"].max() if not known.empty else pd.NaT,
                "weighted_raw_earnings_yield": weighted_raw_ey,
                "input_status": status,
                "historical_evidence_label": "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS",
            }
        )
        if progress is not None:
            progress(number, len(dates), snapshot_date)
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    next_days = _next_trading_day_map(result["date"], index_daily)
    result["earliest_execution_date"] = result["date"].map(next_days)
    result["signal_available_after"] = "SNAPSHOT_DATE_CLOSE"
    result["earliest_execution_time"] = "NEXT_INDEX_TRADING_DAY_OPEN"
    result = apply_frozen_percentile(
        result,
        int(config["percentile_definition"]["window_months"]),
    )
    if (result["latest_price_trade_date"] > result["date"]).any():
        raise ValueError("聚合输出含未来成交价")
    if (result["latest_financial_available_at"] > result["date"]).any():
        raise ValueError("聚合输出含未来财务版本")
    lower_columns = [str(column).lower() for column in result.columns]
    leaked = [
        column
        for column in lower_columns
        if any(token in column for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if leaked:
        raise ValueError(f"输出意外包含禁止字段：{leaked}")
    return result


def _cross_check_source(
    signals: pd.DataFrame,
    comparator: pd.DataFrame,
    pe_column: str,
    window_end: pd.Timestamp,
    minimum_aligned: int,
    minimum_spearman: float,
) -> dict[str, Any]:
    data = comparator[["date", pe_column]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data[pe_column] = pd.to_numeric(data[pe_column], errors="coerce")
    data = data.loc[data[pe_column].gt(0)].drop_duplicates("date", keep="last")
    merged = signals[["date", "weighted_raw_earnings_yield"]].merge(
        data, on="date", how="inner", validate="one_to_one"
    )
    merged["inverse_pe"] = 1.0 / merged[pe_column]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        valid = frame[["weighted_raw_earnings_yield", "inverse_pe"]].dropna()
        relative = valid["weighted_raw_earnings_yield"] / valid["inverse_pe"] - 1.0
        return {
            "aligned_months": int(len(valid)),
            "spearman_correlation": (
                float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) >= 2 else None
            ),
            "pearson_correlation": (
                float(valid.corr(method="pearson").iloc[0, 1]) if len(valid) >= 2 else None
            ),
            "median_relative_difference_vs_inverse_pe": (
                float(relative.median()) if not relative.empty else None
            ),
            "median_absolute_relative_difference_vs_inverse_pe": (
                float(relative.abs().median()) if not relative.empty else None
            ),
        }

    warmup = summarize(merged.loc[merged["date"].le(window_end)])
    full = summarize(merged)
    warmup_spearman = warmup["spearman_correlation"]
    passed = bool(
        warmup["aligned_months"] >= minimum_aligned
        and warmup_spearman is not None
        and warmup_spearman >= minimum_spearman
    )
    return {
        "status": "PASS_DIAGNOSTIC" if passed else "BLOCKED_DIAGNOSTIC",
        "warmup_window": warmup,
        "full_history": full,
    }


def build_report(
    signals: pd.DataFrame,
    vendor_valuation: pd.DataFrame,
    official_pe: pd.DataFrame,
    config: dict[str, Any],
    hash_audit: dict[str, Any],
) -> dict[str, Any]:
    percentile = config["percentile_definition"]
    cross_config = config["independent_cross_check"]
    window_end = pd.Timestamp(cross_config["required_window_end"])
    minimum_aligned = int(cross_config["minimum_aligned_months_per_source"])
    minimum_spearman = float(cross_config["minimum_spearman_correlation_with_inverse_pe"])
    cross_checks = {
        "official_csindex_pe": _cross_check_source(
            signals, official_pe, "pe_official", window_end, minimum_aligned, minimum_spearman
        ),
        "vendor_pe": _cross_check_source(
            signals, vendor_valuation, "pe_ttm", window_end, minimum_aligned, minimum_spearman
        ),
    }
    ready = signals.loc[signals["percentile_ready"]].copy()
    expected_first = pd.Timestamp(percentile["expected_first_ready_observation_date"])
    expected_execution = pd.Timestamp(percentile["expected_first_earliest_execution_date"])
    first_ready = ready.iloc[0] if not ready.empty else None
    structure_pass = bool(
        len(signals) == int(config["signal_definition"]["expected_snapshot_count"])
        and signals["constituent_count"].eq(
            int(config["signal_definition"]["expected_constituents_per_snapshot"])
        ).all()
        and signals["input_status"].eq("PASS").all()
        and first_ready is not None
        and pd.Timestamp(first_ready["date"]) == expected_first
        and pd.Timestamp(first_ready["earliest_execution_date"]) == expected_execution
        and int(signals["percentile_ready"].sum()) == len(signals) - int(percentile["window_months"]) + 1
        and signals["earliest_execution_date"].gt(signals["date"]).all()
    )
    cross_pass = all(row["status"] == "PASS_DIAGNOSTIC" for row in cross_checks.values())
    status = (
        "PASS_SIGNAL_INPUTS_NO_RETURN_LABELS"
        if hash_audit["status"] == "PASS" and structure_pass and cross_pass
        else "BLOCKED_SIGNAL_INPUT_GATE"
    )
    governance = {
        "future_return_labels_generated": False,
        "return_calculation_performed": False,
        "ic_calculation_performed": False,
        "position_mapping_performed": False,
        "order_generation_performed": False,
        "broker_connection_performed": False,
        "upstream_v1_1_files_mutated": False,
    }
    price_min_index = signals["price_weight_coverage"].idxmin()
    ttm_min_index = signals["ttm_earnings_weight_coverage"].idxmin()
    joint_min_index = signals["joint_valid_weight_coverage"].idxmin()
    stale_max_index = signals["stale_price_constituent_count"].idxmax()
    age_max_index = signals["maximum_price_age_calendar_days"].idxmax()
    return {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": status,
        "research_state": config["protocol"]["state"],
        "hash_audit": hash_audit,
        "signal_input_summary": {
            "row_count": int(len(signals)),
            "first_observation_date": str(signals["date"].min().date()),
            "last_observation_date": str(signals["date"].max().date()),
            "input_pass_month_count": int(signals["input_status"].eq("PASS").sum()),
            "warmup_month_count": int((~signals["percentile_ready"]).sum()),
            "percentile_ready_month_count": int(signals["percentile_ready"].sum()),
            "first_percentile_ready_date": str(ready["date"].min().date()) if not ready.empty else None,
            "first_earliest_execution_date": (
                str(ready["earliest_execution_date"].min().date()) if not ready.empty else None
            ),
            "minimum_price_weight_coverage": float(signals["price_weight_coverage"].min()),
            "minimum_price_weight_coverage_date": str(
                signals.loc[price_min_index, "date"].date()
            ),
            "minimum_ttm_earnings_weight_coverage": float(
                signals["ttm_earnings_weight_coverage"].min()
            ),
            "minimum_ttm_earnings_weight_coverage_date": str(
                signals.loc[ttm_min_index, "date"].date()
            ),
            "minimum_joint_valid_weight_coverage": float(
                signals["joint_valid_weight_coverage"].min()
            ),
            "minimum_joint_valid_weight_coverage_date": str(
                signals.loc[joint_min_index, "date"].date()
            ),
            "missing_price_constituent_row_count": int(
                (signals["constituent_count"] - signals["valid_price_constituent_count"]).sum()
            ),
            "missing_ttm_constituent_row_count": int(
                (signals["constituent_count"] - signals["valid_ttm_constituent_count"]).sum()
            ),
            "minimum_valid_price_constituent_count": int(
                signals["valid_price_constituent_count"].min()
            ),
            "minimum_valid_ttm_constituent_count": int(
                signals["valid_ttm_constituent_count"].min()
            ),
            "minimum_valid_joint_constituent_count": int(
                signals["valid_joint_constituent_count"].min()
            ),
            "maximum_stale_price_constituent_count": int(
                signals["stale_price_constituent_count"].max()
            ),
            "maximum_stale_price_constituent_count_date": str(
                signals.loc[stale_max_index, "date"].date()
            ),
            "maximum_price_age_calendar_days": int(
                signals["maximum_price_age_calendar_days"].max()
            ),
            "maximum_price_age_calendar_days_date": str(
                signals.loc[age_max_index, "date"].date()
            ),
            "minimum_percentile": float(ready["raw_ey_percentile_60m"].min()),
            "maximum_percentile": float(ready["raw_ey_percentile_60m"].max()),
        },
        "structure_gate_passed": structure_pass,
        "independent_cross_checks": cross_checks,
        "cross_check_gate_passed": cross_pass,
        "semantic_caveat": cross_config["semantic_caveat"],
        "trial_registration": config["trial_registration"],
        "true_forward": config["true_forward"],
        "governance": governance,
        "artifacts": config["artifacts"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["signal_input_summary"]
    official = report["independent_cross_checks"]["official_csindex_pe"]["warmup_window"]
    vendor = report["independent_cross_checks"]["vendor_pe"]["warmup_window"]
    return "\n".join(
        [
            "# VAL01_RAW_EY_5Y_V1 信号输入闸门",
            "",
            f"> 状态：`{report['status']}`。这里只验证无收益标签的月度估值输入，不构成Alpha、仓位或交易结论。",
            "",
            "## 月度输入",
            "",
            f"- 范围：{summary['first_observation_date']}至{summary['last_observation_date']}，共{summary['row_count']}个月。",
            f"- 当月数据闸门通过：{summary['input_pass_month_count']}个月。",
            f"- 暖机/分位就绪：{summary['warmup_month_count']}/{summary['percentile_ready_month_count']}个月。",
            f"- 首个有效观察/最早执行：{summary['first_percentile_ready_date']}/{summary['first_earliest_execution_date']}。",
            f"- 最低价格/TTM/联合覆盖：{summary['minimum_price_weight_coverage']:.4%}/{summary['minimum_ttm_earnings_weight_coverage']:.4%}/{summary['minimum_joint_valid_weight_coverage']:.4%}。",
            f"- 最差覆盖日期：价格{summary['minimum_price_weight_coverage_date']}，TTM{summary['minimum_ttm_earnings_weight_coverage_date']}，联合{summary['minimum_joint_valid_weight_coverage_date']}。",
            f"- 36,000个成分月度行中缺价格{summary['missing_price_constituent_row_count']}行、缺TTM{summary['missing_ttm_constituent_row_count']}行。",
            f"- 单月最多陈旧价格公司：{summary['maximum_stale_price_constituent_count']}只（{summary['maximum_stale_price_constituent_count_date']}）；最大价格年龄：{summary['maximum_price_age_calendar_days']}个自然日（{summary['maximum_price_age_calendar_days_date']}）。",
            "",
            "## 独立交叉核验（仅诊断）",
            "",
            f"- 中证官方PE：暖机窗同日{official['aligned_months']}个月，EY与1/PE的Spearman={official['spearman_correlation']:.4f}。",
            f"- 供应商PE：暖机窗同日{vendor['aligned_months']}个月，EY与1/PE的Spearman={vendor['spearman_correlation']:.4f}。",
            f"- 口径限制：{report['semantic_caveat']}。",
            "",
            "## 治理结论",
            "",
            "- 未生成未来收益标签、IC、仓位、份额、订单或券商动作。",
            "- 历史数据为事后归档重建，仍标记`HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS`。",
            "- 本闸门通过只允许保存模型输入；进入收益验证前仍需单独授权。",
            "",
        ]
    )


def write_artifacts(
    signals: pd.DataFrame,
    report: dict[str, Any],
    config: dict[str, Any],
) -> None:
    artifacts = config["artifacts"]
    signal_path = ROOT / artifacts["signal_inputs"]
    report_path = ROOT / artifacts["report_json"]
    markdown_path = ROOT / artifacts["report_markdown"]
    for path in (signal_path, report_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_signal = signal_path.with_suffix(signal_path.suffix + ".tmp")
    signals.to_parquet(temp_signal, index=False)
    temp_signal.replace(signal_path)
    report["artifacts"]["signal_inputs_sha256"] = sha256_file(signal_path)
    temp_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_report.replace(report_path)
    temp_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temp_markdown.write_text(render_markdown(report), encoding="utf-8")
    temp_markdown.replace(markdown_path)
