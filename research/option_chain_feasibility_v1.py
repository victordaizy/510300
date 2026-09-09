"""510300期权链的结果盲数据可行性审计。

本模块只读取逐合约当日数据、合约主表、交易日历、分红与来源证据。
它不读取未来收益、标签、策略净值或夏普率，也不生成预测模型和仓位。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import yaml


RISK_NUMERIC_COLUMNS = (
    "delta",
    "theta",
    "gamma",
    "vega",
    "rho",
    "implied_volatility",
)


def sha256(path: Path) -> str:
    """计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    """读取并执行最小协议结构校验。"""

    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("期权链可行性协议不是对象")
    if protocol.get("protocol", {}).get("study_id") != "510300_OPTION_CHAIN_FEASIBILITY_V1":
        raise ValueError("期权链可行性研究ID不匹配")
    if protocol["protocol"].get("future_outcome_reads_allowed") is not False:
        raise ValueError("数据可行性阶段不得允许读取未来结果")
    if protocol["protocol"].get("portfolio_evaluation_allowed") is not False:
        raise ValueError("数据可行性阶段不得允许组合评价")
    return protocol


def assert_outcome_blind_columns(
    frames: dict[str, pd.DataFrame], forbidden_tokens: Iterable[str]
) -> dict[str, Any]:
    """拒绝任何疑似未来标签、策略收益或夏普字段。"""

    tokens = tuple(str(value).lower() for value in forbidden_tokens)
    inspected: dict[str, list[str]] = {}
    violations: list[dict[str, str]] = []
    for name, frame in frames.items():
        columns = [str(value) for value in frame.columns]
        inspected[name] = columns
        for column in columns:
            lowered = column.lower()
            for token in tokens:
                if token in lowered:
                    violations.append({"input": name, "column": column, "token": token})
    if violations:
        raise ValueError(
            "结果盲输入检查失败："
            + json.dumps(violations, ensure_ascii=False, separators=(",", ":"))
        )
    return {
        "future_data_reads": 0,
        "inspected_inputs": inspected,
        "violations": [],
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quantile_spread_ratio(values: pd.Series, scale: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3 or not np.isfinite(scale) or scale <= 0:
        return math.inf
    return float((clean.quantile(0.90) - clean.quantile(0.10)) / scale)


def _shape_violation_counts(
    strikes: pd.Series,
    prices: pd.Series,
    option_type: str,
    price_tolerance: float,
    slope_tolerance: float,
) -> tuple[int, int]:
    table = pd.DataFrame(
        {
            "strike": pd.to_numeric(strikes, errors="coerce"),
            "price": pd.to_numeric(prices, errors="coerce"),
        }
    ).dropna()
    table = table.sort_values("strike").drop_duplicates("strike", keep="last")
    if len(table) < 2:
        return 0, 0
    strike_diff = np.diff(table["strike"].to_numpy(dtype=float))
    price_diff = np.diff(table["price"].to_numpy(dtype=float))
    valid = strike_diff > 0
    slopes = price_diff[valid] / strike_diff[valid]
    if option_type == "C":
        monotonic = int(np.sum(price_diff > price_tolerance))
    elif option_type == "P":
        monotonic = int(np.sum(price_diff < -price_tolerance))
    else:
        raise ValueError(f"未知期权类型：{option_type}")
    convexity = int(np.sum(np.diff(slopes) < -slope_tolerance)) if len(slopes) >= 2 else 0
    comparisons = int(len(price_diff) + max(len(slopes) - 1, 0))
    return monotonic + convexity, comparisons


def audit_maturity_surface(
    rows: pd.DataFrame, geometry: dict[str, Any]
) -> dict[str, Any]:
    """审计单个交易日、单个到期日的可计算性和明显套利一致性。"""

    if rows.empty:
        raise ValueError("期限审计收到空数据")
    table = rows.copy()
    price_field = str(geometry["price_field"])
    for column in (
        "strike",
        "underlying_close",
        price_field,
        "implied_volatility",
        "delta",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    spot = float(table["underlying_close"].median())
    dte = int(table["days_to_expiry"].iloc[0])
    positive_price = table[price_field].gt(0)
    positive_iv = table["implied_volatility"].gt(0)
    valid_numeric = (
        table["strike"].gt(0)
        & table["underlying_close"].gt(0)
        & positive_price
        & positive_iv
        & table["delta"].notna()
        & np.isfinite(table["delta"])
    )
    usable = table.loc[valid_numeric].copy()
    priced_count = int(positive_price.sum())
    positive_iv_ratio = _safe_ratio(int((positive_price & positive_iv).sum()), priced_count)

    calls = usable.loc[usable["option_type"].eq("C")].copy()
    puts = usable.loc[usable["option_type"].eq("P")].copy()
    call_by_strike = calls.sort_values("contract_code").drop_duplicates("strike", keep="last")
    put_by_strike = puts.sort_values("contract_code").drop_duplicates("strike", keep="last")
    pairs = call_by_strike[["strike", price_field, "delta", "implied_volatility"]].merge(
        put_by_strike[["strike", price_field, "delta", "implied_volatility"]],
        on="strike",
        how="inner",
        suffixes=("_call", "_put"),
        validate="one_to_one",
    )
    pair_count = int(len(pairs))
    if pair_count:
        atm_gap = float((pairs["strike"] / spot - 1.0).abs().min())
        parity_forward = pairs["strike"] + (
            pairs[f"{price_field}_call"] - pairs[f"{price_field}_put"]
        )
        parity_dispersion = _quantile_spread_ratio(parity_forward, spot)
    else:
        atm_gap = math.inf
        parity_dispersion = math.inf

    otm_puts = puts.loc[puts["strike"].lt(spot)]
    otm_calls = calls.loc[calls["strike"].gt(spot)]
    put_absolute_delta = otm_puts["delta"].abs().dropna()
    target_delta = float(geometry["target_put_absolute_delta"])
    put_delta_bracket = bool(
        len(put_absolute_delta)
        and put_absolute_delta.min() <= target_delta <= put_absolute_delta.max()
    )

    price_tolerance = float(geometry["monotonicity_price_tolerance"])
    slope_tolerance = float(geometry["convexity_slope_tolerance"])
    call_violations, call_comparisons = _shape_violation_counts(
        call_by_strike["strike"],
        call_by_strike[price_field],
        "C",
        price_tolerance,
        slope_tolerance,
    )
    put_violations, put_comparisons = _shape_violation_counts(
        put_by_strike["strike"],
        put_by_strike[price_field],
        "P",
        price_tolerance,
        slope_tolerance,
    )
    upper_tolerance = float(geometry["price_upper_bound_tolerance_ratio"])
    call_upper = int((calls[price_field] > spot * (1.0 + upper_tolerance)).sum())
    put_upper = int(
        (puts[price_field] > puts["strike"] * (1.0 + upper_tolerance)).sum()
    )
    violation_count = call_violations + put_violations + call_upper + put_upper
    comparison_count = (
        call_comparisons + put_comparisons + int(len(calls)) + int(len(puts))
    )
    violation_ratio = _safe_ratio(violation_count, comparison_count)

    gates = {
        "minimum_matched_call_put_strikes": pair_count
        >= int(geometry["minimum_matched_call_put_strikes_per_maturity"]),
        "minimum_otm_puts": len(otm_puts)
        >= int(geometry["minimum_otm_puts_per_maturity"]),
        "minimum_otm_calls": len(otm_calls)
        >= int(geometry["minimum_otm_calls_per_maturity"]),
        "atm_pair_available": atm_gap <= float(geometry["maximum_atm_moneyness_gap"]),
        "put_25delta_bracket_available": put_delta_bracket,
        "positive_iv_ratio": positive_iv_ratio
        >= float(geometry["minimum_positive_iv_ratio"]),
        "put_call_parity_dispersion": parity_dispersion
        <= float(geometry["maximum_put_call_parity_forward_dispersion_ratio"]),
        "obvious_static_arbitrage_ratio": violation_ratio
        <= float(geometry["maximum_obvious_static_arbitrage_violation_ratio"]),
    }
    return {
        "trade_date": pd.Timestamp(table["trade_date"].iloc[0]).normalize(),
        "expiry_date": pd.Timestamp(table["expiry_date"].iloc[0]).normalize(),
        "days_to_expiry": dte,
        "underlying_close": spot,
        "contract_rows": int(len(table)),
        "usable_rows": int(len(usable)),
        "matched_call_put_strikes": pair_count,
        "otm_put_count": int(len(otm_puts)),
        "otm_call_count": int(len(otm_calls)),
        "positive_iv_ratio": positive_iv_ratio,
        "atm_moneyness_gap": atm_gap,
        "put_25delta_bracket_available": put_delta_bracket,
        "parity_forward_dispersion_ratio": parity_dispersion,
        "obvious_static_arbitrage_violation_count": int(violation_count),
        "obvious_static_arbitrage_comparison_count": int(comparison_count),
        "obvious_static_arbitrage_violation_ratio": violation_ratio,
        "near_30d_bucket": int(geometry["near_30d_minimum_dte"])
        <= dte
        <= int(geometry["near_30d_maximum_dte"]),
        "near_60d_bucket": int(geometry["near_60d_minimum_dte"])
        <= dte
        <= int(geometry["near_60d_maximum_dte"]),
        "valid_maturity": bool(all(gates.values())),
        **{f"gate_{key}": bool(value) for key, value in gates.items()},
    }


def build_surface_audits(
    option_eod: pd.DataFrame,
    risk: pd.DataFrame,
    calendar: pd.DataFrame,
    geometry: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """构造逐期限与逐日审计，不计算未来收益。"""

    eod = option_eod.copy()
    risk_frame = risk.copy()
    eod["trade_date"] = pd.to_datetime(eod["trade_date"], errors="coerce").dt.normalize()
    eod["expiry_date"] = pd.to_datetime(eod["expiry_date"], errors="coerce").dt.normalize()
    risk_frame["trade_date"] = pd.to_datetime(
        risk_frame["trade_date"], errors="coerce"
    ).dt.normalize()
    merged = eod.merge(
        risk_frame[["trade_date", "contract_code", *RISK_NUMERIC_COLUMNS]],
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
        indicator="risk_merge",
    )
    merged["days_to_expiry"] = (
        merged["expiry_date"] - merged["trade_date"]
    ).dt.days
    in_dte = merged["days_to_expiry"].between(
        int(geometry["minimum_days_to_expiry"]),
        int(geometry["maximum_days_to_expiry"]),
        inclusive="both",
    )
    if not bool(geometry["adjusted_contracts_allowed_in_surface_geometry"]):
        in_dte &= ~merged["is_adjusted"].fillna(False).astype(bool)
    eligible = merged.loc[in_dte].copy()
    maturity_rows = [
        audit_maturity_surface(group, geometry)
        for _, group in eligible.groupby(["trade_date", "expiry_date"], sort=True)
    ]
    maturity_audit = pd.DataFrame(maturity_rows).sort_values(
        ["trade_date", "expiry_date"]
    ).reset_index(drop=True)

    calendar_dates = (
        pd.to_datetime(calendar["trade_date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    expected = pd.DataFrame({"trade_date": calendar_dates})
    if maturity_audit.empty:
        aggregated = expected.iloc[0:0].copy()
    else:
        aggregated = (
            maturity_audit.groupby("trade_date", as_index=False)
            .agg(
                maturity_count=("expiry_date", "nunique"),
                valid_maturity_count=("valid_maturity", "sum"),
                minimum_dte=("days_to_expiry", "min"),
                maximum_dte=("days_to_expiry", "max"),
                any_valid_30d=(
                    "near_30d_bucket",
                    lambda values: bool(
                        maturity_audit.loc[values.index, "valid_maturity"].astype(bool).mul(
                            pd.Series(values, index=values.index).astype(bool)
                        ).any()
                    ),
                ),
                any_valid_60d=(
                    "near_60d_bucket",
                    lambda values: bool(
                        maturity_audit.loc[values.index, "valid_maturity"].astype(bool).mul(
                            pd.Series(values, index=values.index).astype(bool)
                        ).any()
                    ),
                ),
            )
        )
    daily = expected.merge(aggregated, on="trade_date", how="left", validate="one_to_one")
    for column in ("maturity_count", "valid_maturity_count"):
        daily[column] = daily[column].fillna(0).astype(int)
    for column in ("any_valid_30d", "any_valid_60d"):
        daily[column] = daily[column].fillna(False).astype(bool)

    contract_counts = (
        merged.groupby("trade_date", as_index=False)
        .agg(
            raw_contract_rows=("contract_code", "size"),
            standard_contract_rows=(
                "is_adjusted",
                lambda values: int((~pd.Series(values).fillna(False).astype(bool)).sum()),
            ),
            adjusted_contract_rows=(
                "is_adjusted",
                lambda values: int(pd.Series(values).fillna(False).astype(bool).sum()),
            ),
            risk_matched_rows=(
                "risk_merge",
                lambda values: int(pd.Series(values).eq("both").sum()),
            ),
        )
    )
    daily = daily.merge(contract_counts, on="trade_date", how="left", validate="one_to_one")
    for column in (
        "raw_contract_rows",
        "standard_contract_rows",
        "adjusted_contract_rows",
        "risk_matched_rows",
    ):
        daily[column] = daily[column].fillna(0).astype(int)
    daily["risk_key_match_ratio"] = np.where(
        daily["raw_contract_rows"].gt(0),
        daily["risk_matched_rows"] / daily["raw_contract_rows"],
        0.0,
    )
    daily["valid_surface_day"] = (
        daily["valid_maturity_count"].ge(
            int(geometry["minimum_valid_maturities_per_day"])
        )
        & daily["any_valid_30d"]
        & daily["any_valid_60d"]
    )
    summary = {
        "calendar_day_count": int(len(daily)),
        "valid_surface_day_count": int(daily["valid_surface_day"].sum()),
        "valid_surface_day_ratio": float(daily["valid_surface_day"].mean())
        if len(daily)
        else 0.0,
        "minimum_valid_maturities_observed": int(daily["valid_maturity_count"].min())
        if len(daily)
        else 0,
        "median_valid_maturities": float(daily["valid_maturity_count"].median())
        if len(daily)
        else 0.0,
        "maximum_valid_maturities": int(daily["valid_maturity_count"].max())
        if len(daily)
        else 0,
        "maturity_rows": int(len(maturity_audit)),
        "valid_maturity_rows": int(maturity_audit["valid_maturity"].sum())
        if len(maturity_audit)
        else 0,
        "risk_key_match_ratio": float(
            _safe_ratio(
                int((merged["risk_merge"] == "both").sum()),
                int(len(merged)),
            )
        ),
    }
    return maturity_audit, daily, summary


def audit_adjustment_ledger(
    root: Path, protocol: dict[str, Any], master: pd.DataFrame
) -> dict[str, Any]:
    """验证调整合约是否由独立公告账本和证据文件完整覆盖。"""

    inputs = protocol["inputs"]
    ledger_path = root / inputs["contract_adjustment_ledger"]
    evidence_path = root / inputs["contract_adjustment_evidence"]
    adjusted_codes = set(
        master.loc[master["is_adjusted"].fillna(False).astype(bool), "contract_code"].astype(str)
    )
    result: dict[str, Any] = {
        "ledger_path": inputs["contract_adjustment_ledger"],
        "evidence_path": inputs["contract_adjustment_evidence"],
        "ledger_exists": ledger_path.exists(),
        "evidence_exists": evidence_path.exists(),
        "adjusted_contract_count": int(len(adjusted_codes)),
        "covered_adjusted_contract_count": 0,
        "coverage_ratio": 0.0 if adjusted_codes else 1.0,
        "complete": False,
    }
    if not ledger_path.exists() or not evidence_path.exists():
        return result
    ledger = pd.read_parquet(ledger_path)
    required = {
        "contract_code",
        "announcement_date",
        "effective_date",
        "old_contract_unit",
        "new_contract_unit",
        "old_strike",
        "new_strike",
        "official_notice_url",
        "official_notice_sha256",
    }
    missing = sorted(required.difference(ledger.columns))
    result["missing_columns"] = missing
    if missing:
        return result
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    covered = set(ledger["contract_code"].dropna().astype(str)) & adjusted_codes
    result.update(
        {
            "covered_adjusted_contract_count": int(len(covered)),
            "coverage_ratio": _safe_ratio(len(covered), len(adjusted_codes)),
            "duplicate_contract_rows": int(ledger["contract_code"].duplicated().sum()),
            "official_url_missing_rows": int(
                ledger["official_notice_url"].fillna("").astype(str).str.strip().eq("").sum()
            ),
            "official_hash_missing_rows": int(
                ledger["official_notice_sha256"].fillna("").astype(str).str.len().ne(64).sum()
            ),
            "evidence_status": evidence.get("status"),
        }
    )
    result["complete"] = bool(
        result["coverage_ratio"] == 1.0
        and result["duplicate_contract_rows"] == 0
        and result["official_url_missing_rows"] == 0
        and result["official_hash_missing_rows"] == 0
        and evidence.get("status") == "PASS"
    )
    return result


def _normalize_remote_risk(records: list[dict[str, Any]]) -> pd.DataFrame:
    raw = pd.DataFrame(records)
    if raw.empty:
        return pd.DataFrame(columns=["trade_date", "contract_code", *RISK_NUMERIC_COLUMNS])
    selected = raw.loc[
        raw["CONTRACT_ID"].astype(str).str.startswith("510300", na=False)
    ].copy()
    result = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(selected["TRADE_DATE"], errors="coerce").dt.normalize(),
            "contract_code": selected["SECURITY_ID"].astype(str).str.strip() + ".SH",
            "delta": pd.to_numeric(selected["DELTA_VALUE"], errors="coerce"),
            "theta": pd.to_numeric(selected["THETA_VALUE"], errors="coerce"),
            "gamma": pd.to_numeric(selected["GAMMA_VALUE"], errors="coerce"),
            "vega": pd.to_numeric(selected["VEGA_VALUE"], errors="coerce"),
            "rho": pd.to_numeric(selected["RHO_VALUE"], errors="coerce"),
            "implied_volatility": pd.to_numeric(
                selected["IMPLC_VOLATLTY"], errors="coerce"
            ),
        }
    )
    return result.sort_values(["trade_date", "contract_code"]).reset_index(drop=True)


def validate_official_sources(
    protocol: dict[str, Any], calendar: pd.DataFrame, local_risk: pd.DataFrame
) -> dict[str, Any]:
    """在线核验官方页面和三个确定性历史风险指标切片。"""

    source = protocol["source_contract"]
    headers = {
        "Referer": "https://www.sse.com.cn/",
        "User-Agent": "Mozilla/5.0 510300-option-feasibility-audit/1.0",
    }
    session = requests.Session()
    session.headers.update(headers)
    page_specs = {
        "listing": (source["official_listing_url"], "2019年12月23日"),
        "market": (source["official_market_page"], "上海证券交易所"),
        "adjustment": (source["official_adjustment_page"], "上海证券交易所"),
        "historical_product": (source["official_historical_product_page"], "行情历史数据"),
        "product_price": (source["official_product_price_page"], "产品服务价格"),
    }
    pages: dict[str, Any] = {}
    for name, (url, marker) in page_specs.items():
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            pages[name] = {
                "url": url,
                "status_code": int(response.status_code),
                "content_sha256": hashlib.sha256(response.content).hexdigest(),
                "content_bytes": int(len(response.content)),
                "required_marker_present": marker.encode("utf-8") in response.content,
                "passed": bool(
                    response.status_code == 200 and marker.encode("utf-8") in response.content
                ),
            }
        except Exception as exc:
            pages[name] = {
                "url": url,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }

    dates = (
        pd.to_datetime(calendar["trade_date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    positions = [float(value) for value in source["live_validation_sample_positions"]]
    indices = sorted({int(round(value * (len(dates) - 1))) for value in positions})
    sample_dates = [pd.Timestamp(dates.iloc[index]).normalize() for index in indices]
    local = local_risk.copy()
    local["trade_date"] = pd.to_datetime(local["trade_date"], errors="coerce").dt.normalize()
    samples: list[dict[str, Any]] = []
    tolerance = float(source["live_validation_numeric_tolerance"])
    for trade_date in sample_dates:
        params = {
            "isPagination": "false",
            "trade_date": trade_date.strftime("%Y%m%d"),
            "sqlId": source["official_risk_sql_id"],
            "contractSymbol": "",
        }
        try:
            response = session.get(
                source["official_risk_endpoint"], params=params, timeout=45
            )
            response.raise_for_status()
            payload = response.json()
            records = payload.get("result", [])
            remote = _normalize_remote_risk(records)
            cached = local.loc[local["trade_date"].eq(trade_date), [
                "trade_date",
                "contract_code",
                *RISK_NUMERIC_COLUMNS,
            ]].copy()
            remote_duplicates = int(remote.duplicated(["trade_date", "contract_code"]).sum())
            cached_duplicates = int(cached.duplicated(["trade_date", "contract_code"]).sum())
            joined = remote.merge(
                cached,
                on=["trade_date", "contract_code"],
                how="outer",
                suffixes=("_remote", "_cached"),
                indicator=True,
            )
            key_match = bool(len(joined) and joined["_merge"].eq("both").all())
            maximum_difference = 0.0
            for column in RISK_NUMERIC_COLUMNS:
                left = pd.to_numeric(joined[f"{column}_remote"], errors="coerce")
                right = pd.to_numeric(joined[f"{column}_cached"], errors="coerce")
                both_nan = left.isna() & right.isna()
                difference = (left - right).abs().where(~both_nan, 0.0)
                if difference.notna().any():
                    maximum_difference = max(maximum_difference, float(difference.max()))
            passed = bool(
                key_match
                and remote_duplicates == 0
                and cached_duplicates == 0
                and maximum_difference <= tolerance
            )
            samples.append(
                {
                    "trade_date": str(trade_date.date()),
                    "remote_rows": int(len(remote)),
                    "cached_rows": int(len(cached)),
                    "remote_response_sha256": hashlib.sha256(response.content).hexdigest(),
                    "key_match": key_match,
                    "maximum_numeric_difference": maximum_difference,
                    "passed": passed,
                }
            )
        except Exception as exc:
            samples.append(
                {
                    "trade_date": str(trade_date.date()),
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
    return {
        "provider": "上海证券交易所",
        "pages": pages,
        "historical_risk_samples": samples,
        "official_pages_passed": bool(all(value.get("passed") for value in pages.values())),
        "historical_risk_revalidation_passed": bool(
            samples and all(value.get("passed") for value in samples)
        ),
        "same_day_publication_time_proven": bool(
            source["same_day_publication_time_proven"]
        ),
        "conservative_availability_lag_trading_days": int(
            source["conservative_availability_lag_trading_days"]
        ),
    }


def decide_terminal_status(hard_gates: dict[str, bool], terminal_states: dict[str, str]) -> str:
    """全部硬门通过才返回数据可行性PASS。"""

    return terminal_states["pass"] if all(hard_gates.values()) else terminal_states["blocked"]


def run_feasibility_audit(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """运行完整结果盲数据审计并返回机器结果和明细。"""

    inputs = protocol["inputs"]
    required_paths = {
        name: root / path
        for name, path in inputs.items()
        if name
        in {
            "contract_master",
            "option_eod",
            "official_risk_indicators",
            "trading_calendar",
            "dividends",
        }
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"数据可行性必需输入缺失：{missing}")
    master = pd.read_parquet(required_paths["contract_master"])
    option_eod = pd.read_parquet(required_paths["option_eod"])
    risk = pd.read_parquet(required_paths["official_risk_indicators"])
    calendar = pd.read_parquet(required_paths["trading_calendar"])
    dividends = pd.read_csv(required_paths["dividends"])
    outcome_blind = assert_outcome_blind_columns(
        {
            "contract_master": master,
            "option_eod": option_eod,
            "official_risk_indicators": risk,
            "trading_calendar": calendar,
            "dividends": dividends,
        },
        protocol["outcome_blindness"]["forbidden_column_tokens"],
    )

    for frame, date_columns in (
        (master, ("list_date", "expiry_date", "delist_date")),
        (option_eod, ("trade_date", "expiry_date")),
        (risk, ("trade_date",)),
        (calendar, ("trade_date",)),
    ):
        for column in date_columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    start = pd.Timestamp(protocol["protocol"]["evaluation_start"]).normalize()
    end = pd.Timestamp(protocol["protocol"]["evaluation_end"]).normalize()
    option_eod = option_eod.loc[option_eod["trade_date"].between(start, end)].copy()
    risk = risk.loc[risk["trade_date"].between(start, end)].copy()
    calendar = calendar.loc[calendar["trade_date"].between(start, end)].copy()

    duplicate_eod = int(option_eod.duplicated(["trade_date", "contract_code"]).sum())
    duplicate_risk = int(risk.duplicated(["trade_date", "contract_code"]).sum())
    duplicate_master = int(master["contract_code"].duplicated().sum())
    maturity_audit, daily_audit, surface_summary = build_surface_audits(
        option_eod,
        risk,
        calendar,
        protocol["surface_geometry"],
    )
    adjustment = audit_adjustment_ledger(root, protocol, master)
    entitlement_path = root / inputs["vendor_entitlement_evidence"]
    entitlement_present = entitlement_path.exists()
    entitlement_status = None
    if entitlement_present:
        entitlement_payload = json.loads(entitlement_path.read_text(encoding="utf-8"))
        entitlement_status = entitlement_payload.get("status")
    official_receipt = validate_official_sources(protocol, calendar, risk)

    dividend_dates = pd.to_datetime(dividends["ex_date"], errors="coerce").dt.normalize()
    in_scope_dividends = dividends.loc[dividend_dates.between(start, end)].copy()
    dividend_ledger_complete = bool(
        len(in_scope_dividends)
        and in_scope_dividends["source"].fillna("").astype(str).str.startswith(
            "https://www.sse.com.cn/"
        ).all()
    )
    actual_start = option_eod["trade_date"].min()
    actual_end = option_eod["trade_date"].max()
    calendar_dates = set(calendar["trade_date"].dropna())
    eod_dates = set(option_eod["trade_date"].dropna())
    date_coverage_ratio = _safe_ratio(len(calendar_dates & eod_dates), len(calendar_dates))
    complete_point_in_time_chain = bool(
        entitlement_present
        and entitlement_status == "PASS"
        and adjustment["complete"]
        and official_receipt["historical_risk_revalidation_passed"]
        and dividend_ledger_complete
    )
    gates = {
        "start_date_at_listing": bool(
            pd.notna(actual_start)
            and actual_start == pd.Timestamp(protocol["protocol"]["option_listing_date"])
        ),
        "end_date_matches_frozen_cutoff": bool(pd.notna(actual_end) and actual_end == end),
        "full_trading_date_coverage": date_coverage_ratio == 1.0,
        "valid_surface_day_ratio": surface_summary["valid_surface_day_ratio"]
        >= float(protocol["hard_gates"]["minimum_valid_surface_day_ratio"]),
        "contract_adjustment_ledger_complete": bool(adjustment["complete"]),
        "no_duplicate_date_contract_rows": max(duplicate_eod, duplicate_risk)
        <= int(protocol["hard_gates"]["maximum_duplicate_date_contract_rows"]),
        "unique_contract_master": duplicate_master == 0,
        "official_risk_history_revalidation": bool(
            official_receipt["historical_risk_revalidation_passed"]
        ),
        "complete_point_in_time_option_chain": complete_point_in_time_chain,
        "future_data_reads_zero": outcome_blind["future_data_reads"]
        == int(protocol["hard_gates"]["future_data_reads"]),
    }
    status = decide_terminal_status(gates, protocol["terminal_states"])
    failure_reasons = [name for name, passed in gates.items() if not passed]
    input_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in required_paths.values()
    }
    report = {
        "study_id": protocol["protocol"]["study_id"],
        "version": protocol["protocol"]["version"],
        "status": status,
        "research_stage": "DATA_FEASIBILITY_ONLY",
        "scope": {
            "underlying": "510300.SH",
            "start": str(start.date()),
            "end": str(end.date()),
            "calendar_days": int(len(calendar_dates)),
            "option_eod_rows": int(len(option_eod)),
            "official_risk_rows": int(len(risk)),
            "contract_master_rows": int(len(master)),
            "adjusted_contract_count": int(
                master["is_adjusted"].fillna(False).astype(bool).sum()
            ),
            "in_scope_dividend_events": int(len(in_scope_dividends)),
        },
        "price_contract": {
            "field": protocol["surface_geometry"]["price_field"],
            "close_and_settlement_mixed": False,
            "surface_geometry_uses_adjusted_contracts": bool(
                protocol["surface_geometry"][
                    "adjusted_contracts_allowed_in_surface_geometry"
                ]
            ),
            "static_arbitrage_repair_method": protocol["surface_geometry"][
                "static_arbitrage_repair_method"
            ],
        },
        "coverage": {
            "actual_first_trade_date": str(actual_start.date()),
            "actual_last_trade_date": str(actual_end.date()),
            "trading_date_coverage_ratio": date_coverage_ratio,
            "duplicate_option_eod_rows": duplicate_eod,
            "duplicate_official_risk_rows": duplicate_risk,
            "duplicate_contract_master_rows": duplicate_master,
            **surface_summary,
        },
        "adjustment_ledger": adjustment,
        "dividend_ledger": {
            "in_scope_event_count": int(len(in_scope_dividends)),
            "all_sources_are_sse_urls": dividend_ledger_complete,
            "complete_for_data_gate": dividend_ledger_complete,
        },
        "point_in_time_evidence": {
            "vendor_entitlement_evidence_path": inputs["vendor_entitlement_evidence"],
            "vendor_entitlement_evidence_present": entitlement_present,
            "vendor_entitlement_status": entitlement_status,
            "official_risk_history_revalidated": official_receipt[
                "historical_risk_revalidation_passed"
            ],
            "same_day_publication_time_proven": official_receipt[
                "same_day_publication_time_proven"
            ],
            "conservative_availability_lag_trading_days": official_receipt[
                "conservative_availability_lag_trading_days"
            ],
            "complete_point_in_time_option_chain": complete_point_in_time_chain,
        },
        "outcome_blindness": outcome_blind,
        "hard_gates": gates,
        "failed_hard_gates": failure_reasons,
        "input_sha256": input_hashes,
        "governance": {
            "prediction_model_created": False,
            "future_10d_label_read": False,
            "portfolio_evaluation_run": False,
            "position_mapping_enabled": False,
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "order_generation_enabled": False,
            "live_trading_authorized": False,
            "model_action": "ABSTAIN",
            "model_position_target": "UNSET",
        },
        "next_stage": (
            "FREEZE_OPTION_IMPLIED_10D_LEFT_TAIL_PROTOCOL"
            if status == protocol["terminal_states"]["pass"]
            else "OPTION_MODEL_NOT_ALLOWED_BACKUP_CANDIDATE_MAY_START_SEPARATELY"
        ),
    }
    return report, maturity_audit, daily_audit, official_receipt


def render_markdown(report: dict[str, Any]) -> str:
    """生成简洁且不把BLOCKED误写成策略失败的中文报告。"""

    coverage = report["coverage"]
    adjustment = report["adjustment_ledger"]
    evidence = report["point_in_time_evidence"]
    gates = report["hard_gates"]
    gate_lines = "\n".join(
        f"| `{name}` | {'通过' if passed else '失败'} |"
        for name, passed in gates.items()
    )
    failed = "、".join(report["failed_hard_gates"]) or "无"
    return (
        "# 510300期权链数据可行性 V1\n\n"
        f"最终状态：`{report['status']}`\n\n"
        "本阶段只审计2019-12-23以来的逐合约数据、期限结构和来源证据；"
        "未读取未来10日收益，未创建预测模型，未运行组合回测。\n\n"
        "## 数据范围\n\n"
        f"- 逐合约日行情：{report['scope']['option_eod_rows']}行\n"
        f"- 上交所风险指标：{report['scope']['official_risk_rows']}行\n"
        f"- 合约主表：{report['scope']['contract_master_rows']}行\n"
        f"- 调整合约：{report['scope']['adjusted_contract_count']}个\n"
        f"- 交易日：{report['scope']['calendar_days']}日\n"
        f"- 日期覆盖率：{coverage['trading_date_coverage_ratio']:.2%}\n"
        f"- 有效曲面日：{coverage['valid_surface_day_count']}/{coverage['calendar_day_count']}"
        f"（{coverage['valid_surface_day_ratio']:.2%}）\n"
        f"- 风险指标键匹配率：{coverage['risk_key_match_ratio']:.2%}\n\n"
        "## 硬门\n\n"
        "| 硬门 | 结果 |\n|---|---|\n"
        f"{gate_lines}\n\n"
        f"失败硬门：{failed}。\n\n"
        "## 点时与调整证据\n\n"
        f"- 调整公告账本存在：{str(adjustment['ledger_exists']).lower()}\n"
        f"- 调整合约公告覆盖率：{adjustment['coverage_ratio']:.2%}\n"
        f"- 历史日行情许可证据存在：{str(evidence['vendor_entitlement_evidence_present']).lower()}\n"
        f"- 上交所历史风险指标抽样复核：{str(evidence['official_risk_history_revalidated']).lower()}\n"
        f"- 同日正式发布时间已证明：{str(evidence['same_day_publication_time_proven']).lower()}\n"
        f"- 保守可用性滞后：{evidence['conservative_availability_lag_trading_days']}个交易日\n"
        f"- 完整point-in-time期权链：{str(evidence['complete_point_in_time_option_chain']).lower()}\n\n"
        "## 授权边界\n\n"
        "`FUTURE_DATA_READS=0`，`MODEL_ACTION=ABSTAIN`，"
        "`MODEL_POSITION_TARGET=UNSET`。本报告不产生现金目标、已有持仓覆盖、"
        "Paper/Shadow信号、订单或实盘授权。\n"
    )
