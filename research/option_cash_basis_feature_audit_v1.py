"""510300 期权—现货基差特征构造审计；本程序禁止读取收益数据。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "510300_option_cash_basis_feature_audit_v1.yaml"


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_date(series: pd.Series) -> pd.Series:
    """统一为无时区、按日归一化的时间戳。"""

    normalized = (
        pd.to_datetime(series, errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return normalized.astype("datetime64[ns]")


def verify_input(root: Path, specification: dict[str, Any]) -> Path:
    """核对固定输入路径和哈希。"""

    path = root / str(specification["path"])
    if not path.exists():
        raise FileNotFoundError(f"固定输入不存在：{path}")
    actual = sha256_file(path)
    expected = str(specification["sha256"])
    if actual != expected:
        raise ValueError(
            f"固定输入哈希漂移：{path}，期望 {expected}，实际 {actual}"
        )
    return path


def interpolate_maturity_rate(
    days: np.ndarray,
    rate_3m_percent: np.ndarray,
    rate_6m_percent: np.ndarray,
    rate_1y_percent: np.ndarray,
) -> np.ndarray:
    """按自然日线性插值短端国债曲线，并在两端钳制。"""

    x0, x1, x2 = 91.3125, 182.625, 365.25
    d = np.asarray(days, dtype=float)
    r0 = np.asarray(rate_3m_percent, dtype=float)
    r1 = np.asarray(rate_6m_percent, dtype=float)
    r2 = np.asarray(rate_1y_percent, dtype=float)
    between_3m_6m = r0 + (r1 - r0) * ((d - x0) / (x1 - x0))
    between_6m_1y = r1 + (r2 - r1) * ((d - x1) / (x2 - x1))
    percent = np.where(
        d <= x0,
        r0,
        np.where(d <= x1, between_3m_6m, np.where(d <= x2, between_6m_1y, r2)),
    )
    return percent / 100.0


def _prepare_option_rows(
    option_eod: pd.DataFrame,
    risk_indicators: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """筛选合法记录、合并交易所隐含波动率并消除重复合约。"""

    pair_rule = config["pair_construction"]
    start = pd.Timestamp(config["dates"]["start"])
    end = pd.Timestamp(config["dates"]["end"])
    required_option = {
        "trade_date",
        "contract_code",
        "option_type",
        "expiry_date",
        "strike",
        "contract_unit",
        "is_adjusted",
        "close",
        "volume",
        "open_interest",
        "underlying_close",
    }
    if missing := required_option - set(option_eod.columns):
        raise ValueError(f"期权日线缺少字段：{sorted(missing)}")
    required_risk = {"trade_date", "contract_code", "implied_volatility"}
    if missing := required_risk - set(risk_indicators.columns):
        raise ValueError(f"交易所风险指标缺少字段：{sorted(missing)}")

    rows = option_eod[list(required_option)].copy()
    rows["trade_date"] = normalize_date(rows["trade_date"])
    rows["expiry_date"] = normalize_date(rows["expiry_date"])
    raw_rows = int(len(rows))
    rows = rows.loc[rows["trade_date"].between(start, end, inclusive="both")].copy()
    in_date_rows = int(len(rows))
    rows["dte_calendar_days"] = (rows["expiry_date"] - rows["trade_date"]).dt.days
    numeric_columns = [
        "strike",
        "contract_unit",
        "close",
        "volume",
        "open_interest",
        "underlying_close",
        "dte_calendar_days",
    ]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.loc[
        rows["option_type"].isin(pair_rule["option_types"])
        & rows["trade_date"].notna()
        & rows["expiry_date"].notna()
        & rows[numeric_columns].notna().all(axis=1)
        & rows["strike"].gt(0)
        & rows["contract_unit"].gt(0)
        & rows["close"].gt(0)
        & rows["underlying_close"].gt(0)
        & rows["dte_calendar_days"].ge(
            int(pair_rule["minimum_calendar_days_to_expiry"])
        )
    ].copy()
    if not bool(pair_rule["adjusted_contracts_allowed"]):
        rows = rows.loc[~rows["is_adjusted"].astype(bool)].copy()
    base_valid_rows = int(len(rows))

    risk = risk_indicators[list(required_risk)].copy()
    risk["trade_date"] = normalize_date(risk["trade_date"])
    risk["implied_volatility"] = pd.to_numeric(
        risk["implied_volatility"], errors="coerce"
    )
    risk = (
        risk.sort_values(["trade_date", "contract_code"], kind="mergesort")
        .drop_duplicates(["trade_date", "contract_code"], keep="last")
    )
    rows = rows.merge(
        risk,
        on=["trade_date", "contract_code"],
        how="left",
        validate="many_to_one",
    )

    pair_keys = list(pair_rule["pair_keys"])
    rows = rows.sort_values(
        [*pair_keys, "option_type", "open_interest", "volume", "contract_code"],
        ascending=[*[True] * len(pair_keys), True, False, False, True],
        kind="mergesort",
    )
    rows = rows.drop_duplicates([*pair_keys, "option_type"], keep="first")
    unique_contract_rows = int(len(rows))
    audit = {
        "raw_option_rows": raw_rows,
        "rows_inside_date_contract": in_date_rows,
        "base_valid_rows": base_valid_rows,
        "unique_contract_rows": unique_contract_rows,
    }
    return rows.reset_index(drop=True), audit


def _pair_calls_and_puts(
    rows: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """以完全相同的交割经济条款匹配认购与认沽。"""

    pair_keys = list(config["pair_construction"]["pair_keys"])
    value_columns = [
        "contract_code",
        "close",
        "volume",
        "open_interest",
        "underlying_close",
        "dte_calendar_days",
        "implied_volatility",
    ]
    calls = rows.loc[rows["option_type"].eq("C"), [*pair_keys, *value_columns]].copy()
    puts = rows.loc[rows["option_type"].eq("P"), [*pair_keys, *value_columns]].copy()
    calls = calls.rename(columns={column: f"call_{column}" for column in value_columns})
    puts = puts.rename(columns={column: f"put_{column}" for column in value_columns})
    pairs = calls.merge(puts, on=pair_keys, how="inner", validate="one_to_one")
    potential_pairs = int(len(pairs))
    same_spot = np.isclose(
        pairs["call_underlying_close"].to_numpy(float),
        pairs["put_underlying_close"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )
    same_dte = pairs["call_dte_calendar_days"].eq(pairs["put_dte_calendar_days"])
    mismatch_count = int((~same_spot | ~same_dte).sum())
    pairs = pairs.loc[same_spot & same_dte].copy()
    pairs["underlying_close"] = pairs["call_underlying_close"]
    pairs["dte_calendar_days"] = pairs["call_dte_calendar_days"].astype(int)
    audit = {
        "potential_matched_pairs": potential_pairs,
        "underlying_or_dte_mismatch_pairs": mismatch_count,
        "economically_identical_pairs": int(len(pairs)),
    }
    return pairs.reset_index(drop=True), audit


def _attach_rates(
    pairs: pd.DataFrame, rate_curve: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """仅向过去匹配曲线日期，然后按每个合约剩余期限插值。"""

    required = {"date", "cgb_3m", "cgb_6m", "cgb_1y"}
    if missing := required - set(rate_curve.columns):
        raise ValueError(f"短端国债曲线缺少字段：{sorted(missing)}")
    rates = rate_curve[list(required)].copy().rename(columns={"date": "rate_date"})
    rates["rate_date"] = normalize_date(rates["rate_date"])
    for column in ["cgb_3m", "cgb_6m", "cgb_1y"]:
        rates[column] = pd.to_numeric(rates[column], errors="coerce")
    rates = (
        rates.dropna()
        .drop_duplicates("rate_date", keep="last")
        .sort_values("rate_date", kind="mergesort")
    )
    tolerance = pd.Timedelta(
        days=int(config["risk_free_curve"]["maximum_backward_staleness_calendar_days"])
    )
    attached = pd.merge_asof(
        pairs.sort_values("trade_date", kind="mergesort"),
        rates,
        left_on="trade_date",
        right_on="rate_date",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    missing_rate = int(attached["rate_date"].isna().sum())
    attached = attached.dropna(subset=["rate_date", "cgb_3m", "cgb_6m", "cgb_1y"]).copy()
    attached["rate_staleness_calendar_days"] = (
        attached["trade_date"] - attached["rate_date"]
    ).dt.days
    attached["risk_free_rate"] = interpolate_maturity_rate(
        attached["dte_calendar_days"].to_numpy(float),
        attached["cgb_3m"].to_numpy(float),
        attached["cgb_6m"].to_numpy(float),
        attached["cgb_1y"].to_numpy(float),
    )
    return attached.reset_index(drop=True), {
        "pairs_without_backward_rate": missing_rate,
        "pairs_with_rate": int(len(attached)),
    }


def construct_pair_basis(
    option_eod: pd.DataFrame,
    risk_indicators: pd.DataFrame,
    rate_curve: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """构造通过过滤的逐认购—认沽对基差。"""

    rows, row_audit = _prepare_option_rows(option_eod, risk_indicators, config)
    pairs, pair_audit = _pair_calls_and_puts(rows, config)
    pairs, rate_audit = _attach_rates(pairs, rate_curve, config)
    tau = pairs["dte_calendar_days"].to_numpy(float) / 365.25
    strike = pairs["strike"].to_numpy(float)
    spot = pairs["underlying_close"].to_numpy(float)
    rate = pairs["risk_free_rate"].to_numpy(float)
    present_value_strike = strike * np.exp(-rate * tau)
    call_close = pairs["call_close"].to_numpy(float)
    put_close = pairs["put_close"].to_numpy(float)
    tolerance = float(
        config["pair_construction"]["no_arbitrage_bounds"][
            "absolute_price_tolerance_cny"
        ]
    )
    call_lower = np.maximum(spot - present_value_strike, 0.0)
    call_upper = spot
    put_lower = np.maximum(present_value_strike - spot, 0.0)
    put_upper = present_value_strike
    call_pass = (call_close >= call_lower - tolerance) & (
        call_close <= call_upper + tolerance
    )
    put_pass = (put_close >= put_lower - tolerance) & (
        put_close <= put_upper + tolerance
    )
    no_arbitrage_pass = call_pass & put_pass
    no_arbitrage_rejections = int((~no_arbitrage_pass).sum())
    if bool(config["pair_construction"]["no_arbitrage_bounds"]["enabled"]):
        pairs = pairs.loc[no_arbitrage_pass].copy()
        tau = tau[no_arbitrage_pass]
        present_value_strike = present_value_strike[no_arbitrage_pass]

    pairs["present_value_strike"] = present_value_strike
    pairs["option_implied_spot"] = (
        pairs["call_close"] + pairs["present_value_strike"] - pairs["put_close"]
    )
    pairs["pair_basis"] = (
        (pairs["option_implied_spot"] - pairs["underlying_close"])
        / (pairs["underlying_close"] * np.sqrt(tau))
    )
    pairs["combined_open_interest"] = (
        pairs["call_open_interest"] + pairs["put_open_interest"]
    )
    positive_oi = pairs["combined_open_interest"].gt(0)
    zero_oi_pairs = int((~positive_oi).sum())
    if bool(config["pair_construction"]["require_positive_combined_open_interest"]):
        pairs = pairs.loc[positive_oi].copy()

    iv_rule = config["independent_direction_check"]
    iv_min = float(iv_rule["implied_volatility_minimum"])
    iv_max = float(iv_rule["implied_volatility_maximum"])
    pairs["ivs_pair"] = pairs["call_implied_volatility"] - pairs["put_implied_volatility"]
    pairs["ivs_pair_valid"] = (
        pairs["call_implied_volatility"].between(iv_min, iv_max, inclusive="both")
        & pairs["put_implied_volatility"].between(iv_min, iv_max, inclusive="both")
        & np.isfinite(pairs["ivs_pair"])
    )
    if not np.isfinite(pairs["pair_basis"]).all():
        raise ValueError("逐对基差出现非有限值")
    pairs = pairs.sort_values(
        ["trade_date", "expiry_date", "strike", "contract_unit", "is_adjusted"],
        kind="mergesort",
    ).reset_index(drop=True)
    audit = {
        **row_audit,
        **pair_audit,
        **rate_audit,
        "no_arbitrage_rejected_pairs": no_arbitrage_rejections,
        "zero_combined_open_interest_pairs": zero_oi_pairs,
        "final_valid_pairs": int(len(pairs)),
        "pairs_with_valid_exchange_ivs": int(pairs["ivs_pair_valid"].sum()),
    }
    return pairs, audit


def aggregate_daily_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """按认购认沽合计持仓量聚合日频 OBASIS 与独立 IVS。"""

    frame = pairs.copy()
    weight = frame["combined_open_interest"].to_numpy(float)
    frame["weighted_basis"] = frame["pair_basis"] * weight
    frame["weighted_dte"] = frame["dte_calendar_days"] * weight
    frame["weighted_rate"] = frame["risk_free_rate"] * weight
    daily = frame.groupby("trade_date", sort=True).agg(
        valid_pair_count=("pair_basis", "size"),
        total_combined_open_interest=("combined_open_interest", "sum"),
        weighted_basis_sum=("weighted_basis", "sum"),
        weighted_dte_sum=("weighted_dte", "sum"),
        weighted_rate_sum=("weighted_rate", "sum"),
        minimum_dte_calendar_days=("dte_calendar_days", "min"),
        maximum_dte_calendar_days=("dte_calendar_days", "max"),
        maximum_rate_staleness_calendar_days=("rate_staleness_calendar_days", "max"),
    )
    daily["obasis"] = (
        daily["weighted_basis_sum"] / daily["total_combined_open_interest"]
    )
    daily["oi_weighted_dte_calendar_days"] = (
        daily["weighted_dte_sum"] / daily["total_combined_open_interest"]
    )
    daily["oi_weighted_risk_free_rate"] = (
        daily["weighted_rate_sum"] / daily["total_combined_open_interest"]
    )
    iv_rows = frame.loc[frame["ivs_pair_valid"]].copy()
    iv_rows["weighted_ivs"] = (
        iv_rows["ivs_pair"] * iv_rows["combined_open_interest"]
    )
    iv_daily = iv_rows.groupby("trade_date", sort=True).agg(
        ivs_pair_count=("ivs_pair", "size"),
        ivs_weight=("combined_open_interest", "sum"),
        weighted_ivs_sum=("weighted_ivs", "sum"),
    )
    iv_daily["exchange_ivs"] = iv_daily["weighted_ivs_sum"] / iv_daily["ivs_weight"]
    daily = daily.join(iv_daily[["ivs_pair_count", "exchange_ivs"]], how="left")
    daily["ivs_pair_count"] = daily["ivs_pair_count"].fillna(0).astype(int)
    daily = daily.reset_index()
    keep = [
        "trade_date",
        "obasis",
        "exchange_ivs",
        "valid_pair_count",
        "ivs_pair_count",
        "total_combined_open_interest",
        "oi_weighted_dte_calendar_days",
        "oi_weighted_risk_free_rate",
        "minimum_dte_calendar_days",
        "maximum_dte_calendar_days",
        "maximum_rate_staleness_calendar_days",
    ]
    return daily[keep].sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    """在共同有限样本上计算皮尔逊相关系数。"""

    frame = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 2 or frame["left"].std() == 0 or frame["right"].std() == 0:
        return None
    return float(frame["left"].corr(frame["right"]))


def quantiles(series: pd.Series) -> dict[str, float]:
    """生成固定分位数摘要。"""

    return {
        "minimum": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
        "standard_deviation": float(series.std(ddof=1)),
    }


def build_report(
    config: dict[str, Any],
    input_paths: dict[str, Path],
    pairs: pd.DataFrame,
    daily: pd.DataFrame,
    audit_counts: dict[str, int],
    output_paths: dict[str, Path],
    checked_at: datetime,
) -> dict[str, Any]:
    """生成机器可读质量报告与硬门结果。"""

    common = daily.dropna(subset=["obasis", "exchange_ivs"]).copy()
    correlation = safe_correlation(common["obasis"], common["exchange_ivs"])
    year_correlations: dict[str, float | None] = {}
    for year, group in common.groupby(common["trade_date"].dt.year, sort=True):
        year_correlations[str(int(year))] = safe_correlation(
            group["obasis"], group["exchange_ivs"]
        )
    all_option_dates = normalize_date(
        pd.read_parquet(input_paths["option_eod"], columns=["trade_date"])["trade_date"]
    )
    start = pd.Timestamp(config["dates"]["start"])
    end = pd.Timestamp(config["dates"]["end"])
    all_option_dates = set(
        all_option_dates.loc[all_option_dates.between(start, end, inclusive="both")]
        .dropna()
        .tolist()
    )
    feature_dates = set(daily["trade_date"].tolist())
    missing_dates = sorted(all_option_dates - feature_dates)
    quality = config["quality_gates"]
    direction = config["independent_direction_check"]
    gates = {
        "minimum_feature_days": {
            "actual": int(len(daily)),
            "required": int(quality["minimum_feature_days"]),
            "pass": int(len(daily)) >= int(quality["minimum_feature_days"]),
        },
        "minimum_median_valid_pairs_per_day": {
            "actual": float(daily["valid_pair_count"].median()),
            "required": int(quality["minimum_median_valid_pairs_per_day"]),
            "pass": float(daily["valid_pair_count"].median())
            >= int(quality["minimum_median_valid_pairs_per_day"]),
        },
        "maximum_missing_option_trading_days": {
            "actual": int(len(missing_dates)),
            "required_maximum": int(quality["maximum_missing_option_trading_days"]),
            "pass": int(len(missing_dates))
            <= int(quality["maximum_missing_option_trading_days"]),
        },
        "minimum_common_ivs_days": {
            "actual": int(len(common)),
            "required": int(direction["minimum_common_days"]),
            "pass": int(len(common)) >= int(direction["minimum_common_days"]),
        },
        "minimum_obasis_ivs_correlation": {
            "actual": correlation,
            "required": float(direction["hard_minimum_daily_pearson_correlation"]),
            "pass": correlation is not None
            and correlation
            >= float(direction["hard_minimum_daily_pearson_correlation"]),
        },
    }
    all_pass = all(bool(item["pass"]) for item in gates.values())
    return {
        "status": (
            "PASS_FEATURE_CONSTRUCTION_ELIGIBLE_FOR_PRE_OUTCOME_STRATEGY_FREEZE"
            if all_pass
            else "FAILED_FEATURE_CONSTRUCTION_STOP"
        ),
        "project_id": config["protocol"]["project_id"],
        "checked_at": checked_at.isoformat(),
        "scope": {
            "underlying": "510300.SH",
            "feature_only": True,
            "future_return_files_read": False,
            "strategy_defined": False,
        },
        "date_coverage": {
            "first_feature_date": str(daily["trade_date"].min().date()),
            "last_feature_date": str(daily["trade_date"].max().date()),
            "feature_days": int(len(daily)),
            "option_trading_days": int(len(all_option_dates)),
            "missing_feature_dates": [str(date.date()) for date in missing_dates],
        },
        "input_artifacts": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "construction_counts": audit_counts,
        "daily_summary": {
            "obasis": quantiles(daily["obasis"]),
            "exchange_ivs": quantiles(common["exchange_ivs"]),
            "valid_pairs_per_day": quantiles(daily["valid_pair_count"].astype(float)),
            "combined_open_interest": quantiles(
                daily["total_combined_open_interest"].astype(float)
            ),
            "oi_weighted_dte_calendar_days": quantiles(
                daily["oi_weighted_dte_calendar_days"]
            ),
        },
        "independent_direction_check": {
            "common_days": int(len(common)),
            "daily_pearson_correlation": correlation,
            "yearly_correlations": year_correlations,
        },
        "quality_gates": gates,
        "output_artifacts": {
            "feature_parquet": {
                "path": output_paths["feature_parquet"].relative_to(ROOT).as_posix(),
                "sha256": sha256_file(output_paths["feature_parquet"]),
                "rows": int(len(daily)),
            },
            "pair_parquet": {
                "path": output_paths["pair_parquet"].relative_to(ROOT).as_posix(),
                "sha256": sha256_file(output_paths["pair_parquet"]),
                "rows": int(len(pairs)),
            },
        },
        "decision": (
            "特征构造通过；下一步才可在看收益前冻结满仓/空仓规则。"
            if all_pass
            else "特征构造未通过；禁止把该变量送入策略筛选。"
        ),
        "governance": config["governance"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """渲染简洁、可审计的中文报告。"""

    direction = report["independent_direction_check"]
    gates = report["quality_gates"]
    gate_lines = "\n".join(
        f"- {name}：{'通过' if item['pass'] else '失败'}；实际={item['actual']}"
        for name, item in gates.items()
    )
    yearly = "\n".join(
        f"- {year}：{value:.6f}" if value is not None else f"- {year}：不可计算"
        for year, value in direction["yearly_correlations"].items()
    )
    return f"""# 510300 期权—现货基差特征审计 V1

状态：`{report['status']}`

本审计只构造特征，没有读取 510300 或 H00300 的未来收益，也没有定义交易规则。

## 覆盖

- 特征区间：{report['date_coverage']['first_feature_date']} 至 {report['date_coverage']['last_feature_date']}
- 特征交易日：{report['date_coverage']['feature_days']}
- 有效认购认沽对：{report['construction_counts']['final_valid_pairs']}
- 每日有效对中位数：{report['daily_summary']['valid_pairs_per_day']['median']:.1f}

## 独立方向核验

- 与交易所认购减认沽隐含波动率差的共同日：{direction['common_days']}
- 日频皮尔逊相关：{direction['daily_pearson_correlation']}

分年度相关：

{yearly}

## 硬门

{gate_lines}

## 决策

{report['decision']}
"""


def main() -> int:
    """执行固定特征审计并输出哈希化证据。"""

    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    input_paths = {
        name: verify_input(ROOT, specification)
        for name, specification in config["inputs"].items()
    }
    option_eod = pd.read_parquet(input_paths["option_eod"])
    risk_indicators = pd.read_parquet(input_paths["option_risk_indicators"])
    rate_curve = pd.read_parquet(input_paths["government_bond_short_curve"])
    pairs, audit_counts = construct_pair_basis(
        option_eod, risk_indicators, rate_curve, config
    )
    daily = aggregate_daily_features(pairs)
    if pairs.empty or daily.empty:
        raise ValueError("特征构造结果为空")

    output_paths = {
        name: ROOT / path
        for name, path in config["outputs"].items()
    }
    output_paths["feature_parquet"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["pair_parquet"].parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_paths["feature_parquet"], index=False)
    pair_columns = [
        "trade_date",
        "expiry_date",
        "strike",
        "contract_unit",
        "is_adjusted",
        "call_contract_code",
        "put_contract_code",
        "call_close",
        "put_close",
        "underlying_close",
        "dte_calendar_days",
        "risk_free_rate",
        "rate_date",
        "rate_staleness_calendar_days",
        "present_value_strike",
        "option_implied_spot",
        "pair_basis",
        "combined_open_interest",
        "call_implied_volatility",
        "put_implied_volatility",
        "ivs_pair",
        "ivs_pair_valid",
    ]
    pairs[pair_columns].to_parquet(output_paths["pair_parquet"], index=False)
    checked_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    report = build_report(
        config,
        input_paths,
        pairs,
        daily,
        audit_counts,
        output_paths,
        checked_at,
    )
    output_paths["report_json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_paths["report_markdown"].write_text(
        render_markdown(report), encoding="utf-8"
    )
    summary = {
        "status": report["status"],
        "feature_days": report["date_coverage"]["feature_days"],
        "valid_pairs": report["construction_counts"]["final_valid_pairs"],
        "obasis_ivs_correlation": report["independent_direction_check"][
            "daily_pearson_correlation"
        ],
        "feature_sha256": report["output_artifacts"]["feature_parquet"]["sha256"],
        "pair_sha256": report["output_artifacts"]["pair_parquet"]["sha256"],
        "report_sha256": sha256_file(output_paths["report_json"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
