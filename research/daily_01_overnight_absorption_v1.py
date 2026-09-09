"""DAILY_01隔夜冲击—日内吸收：冻结特征公式与只读数据闸门。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.quality_check_daily import check_daily_data, check_metadata_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "daily_01_overnight_absorption_v1.yaml"
SETTINGS_FILE = ROOT / "config" / "settings.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point_in_time_percentile(values: pd.Series, history: int) -> pd.Series:
    """用严格截至前一行的固定历史计算经验百分位。"""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    for index in range(history, len(numeric)):
        current = numeric[index]
        previous = numeric[index - history : index]
        if np.isfinite(current) and np.isfinite(previous).all():
            result[index] = float(np.mean(previous <= current))
    return pd.Series(result, index=values.index, dtype=float)


def build_daily_decomposition(
    market: pd.DataFrame, dividends: pd.DataFrame
) -> pd.DataFrame:
    """构造分红修正的总收益、隔夜贡献和日内贡献。"""

    result = market.copy().sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    cash = dividends[["ex_date", "cash_dividend_per_share"]].copy()
    cash["ex_date"] = pd.to_datetime(cash["ex_date"], errors="raise")
    cash = cash.rename(columns={"ex_date": "date", "cash_dividend_per_share": "dividend"})
    result = result.merge(cash, on="date", how="left", validate="one_to_one")
    result["dividend"] = result["dividend"].fillna(0.0)
    result["previous_close"] = result["close"].shift(1)
    denominator = result["previous_close"]
    result["total_return"] = (result["close"] + result["dividend"]) / denominator - 1.0
    result["overnight_contribution"] = (
        result["open"] + result["dividend"] - denominator
    ) / denominator
    result["intraday_contribution"] = (result["close"] - result["open"]) / denominator
    result["decomposition_error"] = result["total_return"] - (
        result["overnight_contribution"] + result["intraday_contribution"]
    )
    return result


def build_signal_states(
    features: pd.DataFrame,
    *,
    entry_percentile: float,
    exit_percentile: float,
    maximum_holding_days: int,
) -> pd.DataFrame:
    """生成研究状态；不生成账户仓位、份额或订单。"""

    result = features.copy().sort_values("date").reset_index(drop=True)
    state = 0
    holding_days = 0
    targets: list[float] = []
    actions: list[str] = []
    holding_history: list[int] = []
    for row in result.itertuples(index=False):
        action = "HOLD"
        percentile = getattr(row, "absorption_percentile")
        entry_eligible = bool(getattr(row, "entry_sign_eligible"))
        if state == 0:
            if entry_eligible and pd.notna(percentile) and percentile >= entry_percentile:
                state = 1
                holding_days = 0
                action = "ENTER"
        else:
            holding_days += 1
            percentile_exit = pd.notna(percentile) and percentile < exit_percentile
            time_exit = holding_days >= maximum_holding_days
            if percentile_exit or time_exit:
                state = 0
                holding_days = 0
                action = "EXIT"
        targets.append(float(state))
        actions.append(action)
        holding_history.append(holding_days)
    result["research_state_target"] = targets
    result["signal_action"] = actions
    result["holding_signal_days"] = holding_history
    return result


def build_absorption_features(
    market: pd.DataFrame, dividends: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """按冻结公式构造特征；函数不读取或生成任何未来收益。"""

    contract = config["feature_contract"]
    result = build_daily_decomposition(market, dividends)
    aggregation = int(contract["aggregation_window_trading_days"])
    volatility = int(contract["volatility_window_trading_days"])
    history = int(contract["percentile_history_trading_days"])
    result["overnight_contribution_5d"] = result["overnight_contribution"].rolling(
        aggregation, min_periods=aggregation
    ).sum()
    result["intraday_contribution_5d"] = result["intraday_contribution"].rolling(
        aggregation, min_periods=aggregation
    ).sum()
    result["realized_volatility_20d"] = result["total_return"].rolling(
        volatility, min_periods=volatility
    ).std(ddof=1)
    denominator = result["realized_volatility_20d"] * np.sqrt(aggregation)
    result["absorption_raw"] = (
        result["intraday_contribution_5d"] - result["overnight_contribution_5d"]
    ) / denominator.where(denominator > 0)
    result["absorption_percentile"] = point_in_time_percentile(
        result["absorption_raw"], history
    )
    result["entry_sign_eligible"] = (
        result["overnight_contribution_5d"].lt(0)
        & result["intraday_contribution_5d"].gt(0)
    )
    result["past_total_return_5d"] = (
        (1.0 + result["total_return"])
        .rolling(aggregation, min_periods=aggregation)
        .apply(np.prod, raw=True)
        - 1.0
    )
    return build_signal_states(
        result,
        entry_percentile=float(contract["entry_percentile"]),
        exit_percentile=float(contract["exit_percentile_strictly_below"]),
        maximum_holding_days=int(contract["maximum_holding_trading_days"]),
    )


def audit_and_load_inputs(
    root: Path = ROOT, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """核验冻结输入；只计算同期贡献恒等式，不计算信号未来结果。"""

    config = config or load_config()
    errors: list[dict] = []
    contracts = config["data_contracts"]
    for name, contract in contracts.items():
        path = root / contract["file"]
        if not path.exists():
            errors.append({"code": "MISSING_DATA_FILE", "contract": name, "file": contract["file"]})
            continue
        actual_hash = sha256_file(path)
        if actual_hash != contract["sha256"]:
            errors.append(
                {
                    "code": "DATA_HASH_MISMATCH",
                    "contract": name,
                    "expected": contract["sha256"],
                    "actual": actual_hash,
                }
            )

    market_contract = contracts["etf_market"]
    metadata_contract = contracts["etf_market_metadata"]
    cross_contract = contracts["etf_market_cross_check"]
    distribution_contract = contracts["distributions"]
    coverage_contract = contracts["distribution_coverage"]
    market = pd.read_parquet(root / market_contract["file"])
    metadata = _load_json(root / metadata_contract["file"])
    settings = yaml.safe_load((root / "config" / "settings.yaml").read_text(encoding="utf-8"))
    data_errors, data_warnings = check_daily_data(market, settings, metadata)
    errors.extend(data_errors)
    errors.extend(check_metadata_contract(market, metadata, sha256_file(root / market_contract["file"])))

    dates = pd.to_datetime(market["date"], errors="coerce")
    required_columns = set(market_contract["required_columns"])
    missing_columns = sorted(required_columns.difference(market.columns))
    if missing_columns:
        errors.append({"code": "MISSING_MARKET_COLUMNS", "columns": missing_columns})
    if len(market) != int(market_contract["required_rows"]):
        errors.append({"code": "MARKET_ROW_COUNT_MISMATCH", "actual": int(len(market))})
    if dates.min().date().isoformat() != market_contract["required_first_date"]:
        errors.append({"code": "MARKET_FIRST_DATE_MISMATCH"})
    if dates.max().date().isoformat() != market_contract["required_last_date"]:
        errors.append({"code": "MARKET_LAST_DATE_MISMATCH"})
    if int(metadata.get("schema_version", -1)) != int(metadata_contract["schema_version"]):
        errors.append({"code": "METADATA_SCHEMA_VERSION_MISMATCH"})

    cross = _load_json(root / cross_contract["file"])
    if cross.get("status") != cross_contract["required_status"]:
        errors.append({"code": "CROSS_SOURCE_STATUS_FAILED", "actual": cross.get("status")})
    if int(cross.get("overlap_rows", -1)) != int(cross_contract["required_overlap_rows"]):
        errors.append({"code": "CROSS_SOURCE_ROW_COUNT_MISMATCH"})
    if cross.get("primary_sha256") != market_contract["sha256"]:
        errors.append({"code": "CROSS_SOURCE_PRIMARY_HASH_MISMATCH"})

    dividends = pd.read_csv(root / distribution_contract["file"])
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise")
    if len(dividends) != int(distribution_contract["required_event_count"]):
        errors.append({"code": "DIVIDEND_EVENT_COUNT_MISMATCH", "actual": int(len(dividends))})
    if dividends["ex_date"].min().date().isoformat() != distribution_contract["required_first_ex_date"]:
        errors.append({"code": "DIVIDEND_FIRST_DATE_MISMATCH"})
    if dividends["ex_date"].max().date().isoformat() != distribution_contract["required_last_ex_date"]:
        errors.append({"code": "DIVIDEND_LAST_DATE_MISMATCH"})

    coverage = _load_json(root / coverage_contract["file"])
    if bool(coverage.get("complete_history_confirmed")) is not bool(coverage_contract["required_complete_history_confirmed"]):
        errors.append({"code": "DIVIDEND_COMPLETE_HISTORY_NOT_CONFIRMED"})
    if coverage.get("coverage_start") != coverage_contract["required_coverage_start"]:
        errors.append({"code": "DIVIDEND_COVERAGE_START_MISMATCH"})
    if coverage.get("coverage_end") != coverage_contract["required_coverage_end"]:
        errors.append({"code": "DIVIDEND_COVERAGE_END_MISMATCH"})

    cutoff = pd.Timestamp(config["protocol"]["historical_evaluation_cutoff"])
    evaluation_market = market.loc[dates.le(cutoff)].copy().reset_index(drop=True)
    if len(evaluation_market) != int(market_contract["evaluation_rows_through_cutoff"]):
        errors.append({"code": "EVALUATION_ROW_COUNT_MISMATCH", "actual": int(len(evaluation_market))})
    if pd.Timestamp(coverage["coverage_end"]) < cutoff:
        errors.append({"code": "DIVIDEND_COVERAGE_BEFORE_EVALUATION_CUTOFF"})
    in_range_dividends = dividends.loc[
        dividends["ex_date"].between(evaluation_market["date"].min(), cutoff)
    ]
    missing_ex_dates = in_range_dividends.loc[
        ~in_range_dividends["ex_date"].isin(pd.to_datetime(evaluation_market["date"])),
        "ex_date",
    ]
    if not missing_ex_dates.empty:
        errors.append(
            {
                "code": "DIVIDEND_EX_DATE_MISSING_FROM_MARKET",
                "dates": missing_ex_dates.dt.strftime("%Y-%m-%d").tolist(),
            }
        )

    decomposition = build_daily_decomposition(evaluation_market, dividends)
    identity_error = float(decomposition["decomposition_error"].abs().dropna().max())
    if not np.isfinite(identity_error) or identity_error > 1e-12:
        errors.append({"code": "RETURN_DECOMPOSITION_IDENTITY_FAILED", "maximum_error": identity_error})

    audit = {
        "status": "PASS" if not errors else "NO_VIEW",
        "project_id": config["protocol"]["project_id"],
        "data_cutoff": str(cutoff.date()),
        "raw_market_rows": int(len(market)),
        "evaluation_market_rows": int(len(evaluation_market)),
        "evaluation_first_date": str(pd.to_datetime(evaluation_market["date"]).min().date()),
        "evaluation_last_date": str(pd.to_datetime(evaluation_market["date"]).max().date()),
        "dividend_events_in_evaluation_range": int(len(in_range_dividends)),
        "maximum_decomposition_identity_error": identity_error,
        "errors": errors,
        "warnings": data_warnings,
        "feature_formula_frozen": True,
        "feature_values_computed_on_real_data": False,
        "signal_state_computed_on_real_data": False,
        "future_return_columns_loaded": False,
        "predictive_outcomes_computed": False,
        "strategy_backtest_computed": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    return evaluation_market, dividends, audit
