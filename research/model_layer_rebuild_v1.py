"""510300模型层重建：R5拆解、第一批独立技术模型和估值数据闸门。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis, norm, skew

from backtest.defensive_valuation_timing_engine import schedule_asymmetric_execution
from backtest.engine import BacktestCosts, run_long_cash_backtest
from backtest.small_account_execution import quantize_position


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "model_layer_rebuild_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "model_layer_rebuild_v1_manifest.json"


@dataclass(frozen=True)
class FractionalCosts:
    commission_rate: float
    slippage_bps: float
    cash_annual_rate: float
    trading_days_per_year: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol["project_id"] != "510300_MODEL_LAYER_REBUILD_V1":
        raise ValueError("统一协议项目编号不匹配")
    if protocol["true_forward_start"] is not None:
        raise ValueError("受污染历史不能伪造前向起点")
    if protocol.get("reporting_revision") != 1 or not protocol.get("reporting_fix_only"):
        raise ValueError("报告层修订记录缺失")
    for switch in ("position_mapping_enabled", "order_generation_enabled", "broker_connection_enabled"):
        if protocol[switch]:
            raise ValueError(f"安全开关不得启用：{switch}")
    r5 = config["r5_decomposition"]["variants"]
    expected_masks = {
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    }
    actual_masks = {(item["valuation"], item["trend"], item["risk"]) for item in r5}
    if actual_masks != expected_masks or len(r5) != 8:
        raise ValueError("R5拆解必须恰好覆盖三个组件的八个组合")
    technical_ids = [item["id"] for item in config["technical_models"]]
    expected_technical = [
        "TECH_01_TSMOM_63_126_252",
        "TECH_03_DONCHIAN_55_20",
        "TECH_03_DONCHIAN_55_20_25_FLOOR",
        "TECH_03_DONCHIAN_100_50_ROBUSTNESS",
        "TECH_05_TREND_PULLBACK",
        "TECH_08_BREADTH_TREND",
    ]
    if technical_ids != expected_technical:
        raise ValueError("技术候选集合偏离冻结清单")
    if config["valuation_models"]["formal_status"] != "NO_VIEW_BLOCKED_NON_VINTAGE_HISTORY":
        raise ValueError("当前估值数据不得解除点时历史闸门")


def validate_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_FILE.exists():
        raise ValueError("实现清单不存在，禁止收益运行")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ValueError("实现清单项目编号不匹配")
    content_hash = manifest.get("manifest_content_sha256")
    hash_payload = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    canonical = json.dumps(
        hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != content_hash:
        raise ValueError("实现清单内容哈希校验失败")
    mismatches: list[str] = []
    for group in ("frozen_files", "data_files"):
        for relative, expected in manifest[group].items():
            path = root / relative
            actual = sha256_file(path) if path.exists() else None
            if actual != expected:
                mismatches.append(relative)
    if mismatches:
        raise ValueError(f"冻结哈希不匹配：{mismatches}")
    return manifest


def _read_parquet(root: Path, contract: dict[str, Any]) -> pd.DataFrame:
    path = root / contract["file"]
    if sha256_file(path) != contract["sha256"]:
        raise ValueError(f"数据哈希不匹配：{contract['file']}")
    return pd.read_parquet(path)


def _normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result.sort_values(column).reset_index(drop=True)


def _market_audit(frame: pd.DataFrame, symbol: str, require_ohlc: bool) -> dict[str, Any]:
    required = {"date", "close", "symbol"}
    if require_ohlc:
        required |= {"open", "high", "low"}
    missing = sorted(required.difference(frame.columns))
    numeric_columns = ["close"] if not require_ohlc else ["open", "high", "low", "close"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce") if not missing else pd.DataFrame()
    invalid_ohlc = 0
    if require_ohlc and not numeric.empty:
        invalid_ohlc = int(
            (
                (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
                | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
            ).sum()
        )
    result = {
        "rows": int(len(frame)),
        "first_date": str(frame["date"].min().date()),
        "last_date": str(frame["date"].max().date()),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "missing_columns": missing,
        "symbols": sorted(frame["symbol"].dropna().astype(str).unique().tolist()),
        "missing_or_nonpositive_prices": int((numeric.isna() | (numeric <= 0)).sum().sum()) if not numeric.empty else None,
        "invalid_ohlc_rows": invalid_ohlc,
    }
    result["status"] = "PASS" if all(
        [
            not missing,
            result["duplicate_dates"] == 0,
            result["symbols"] == [symbol],
            result["missing_or_nonpositive_prices"] == 0,
            invalid_ohlc == 0,
        ]
    ) else "BLOCKED"
    return result


def audit_and_load_inputs(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    contracts = config["data_contracts"]
    hash_checks = {}
    for name, contract in contracts.items():
        path = root / contract["file"]
        actual = sha256_file(path) if path.exists() else None
        hash_checks[name] = {
            "file": contract["file"],
            "exists": path.exists(),
            "expected_sha256": contract["sha256"],
            "actual_sha256": actual,
            "matches": actual == contract["sha256"],
        }
    if not all(item["matches"] for item in hash_checks.values()):
        failed = [name for name, item in hash_checks.items() if not item["matches"]]
        raise ValueError(f"输入哈希失败：{failed}")

    etf = _normalize_date(_read_parquet(root, contracts["etf_market"]))
    price_index = _normalize_date(_read_parquet(root, contracts["price_index"]))
    total_return = _normalize_date(_read_parquet(root, contracts["total_return_index"]))
    breadth = _normalize_date(_read_parquet(root, contracts["official_breadth"]))
    vendor_valuation = _normalize_date(_read_parquet(root, contracts["vendor_valuation"]))
    point_fundamental = _normalize_date(
        _read_parquet(root, contracts["point_in_time_fundamental_monthly"])
    )
    financials = _read_parquet(root, contracts["point_in_time_financial_records"])
    bonds = _normalize_date(_read_parquet(root, contracts["government_bond_yields"]))
    constituents = _normalize_date(_read_parquet(root, contracts["constituents"]))
    preserved_r5 = _normalize_date(_read_parquet(root, contracts["preserved_r5_enhanced_signals"]))
    preserved_r5_baseline = _normalize_date(_read_parquet(root, contracts["preserved_r5_baseline_signals"]))
    dividends = pd.read_csv(root / contracts["dividends"]["file"])
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    coverage = json.loads(
        (root / contracts["dividend_coverage"]["file"]).read_text(encoding="utf-8")
    )

    index_part = price_index[["date", "open", "high", "low", "close"]].rename(
        columns={column: f"price_index_{column}" for column in ("open", "high", "low", "close")}
    )
    tri_part = total_return[["date", "close"]].rename(columns={"close": "h00300_close"})
    etf_part = etf[["date", "open", "high", "low", "close"]].rename(
        columns={column: f"etf_{column}" for column in ("open", "high", "low", "close")}
    )
    market = etf_part.merge(index_part, on="date", validate="one_to_one")
    market = market.merge(tri_part, on="date", validate="one_to_one")
    market["tr_scale_factor"] = market["h00300_close"] / market["price_index_close"]
    for column in ("open", "high", "low", "close"):
        market[f"signal_{column}"] = market[f"price_index_{column}"] * market["tr_scale_factor"]
    close_error = float((market["signal_close"] - market["h00300_close"]).abs().max())

    start = pd.Timestamp(config["protocol"]["evaluation_start"])
    end = pd.Timestamp(config["protocol"]["evaluation_end"])
    evaluation_market = market.loc[market["date"].between(start, end)].copy()
    evaluation_breadth = breadth.loc[breadth["date"].between(start, end)].copy()
    breadth_checks = {
        "rows": int(len(evaluation_breadth)),
        "first_date": str(evaluation_breadth["date"].min().date()),
        "last_date": str(evaluation_breadth["date"].max().date()),
        "component_count_all_300": bool(evaluation_breadth["component_count"].eq(300).all()),
        "valid_for_direction_model_all": bool(evaluation_breadth["valid_for_direction_model"].all()),
        "minimum_ma60_weight_coverage": float(evaluation_breadth["ma60_weight_coverage"].min()),
        "maximum_weight_age_calendar_days": int(evaluation_breadth["weight_snapshot_age_calendar_days"].max()),
        "calendar_matches_etf": set(evaluation_breadth["date"]) == set(evaluation_market["date"]),
    }
    breadth_checks["status"] = "PASS" if all(
        [
            breadth_checks["rows"] == 1211,
            breadth_checks["component_count_all_300"],
            breadth_checks["valid_for_direction_model_all"],
            breadth_checks["minimum_ma60_weight_coverage"] >= 0.99,
            breadth_checks["maximum_weight_age_calendar_days"] <= 35,
            breadth_checks["calendar_matches_etf"],
        ]
    ) else "BLOCKED"

    point_months = int(point_fundamental["date"].nunique())
    vendor_point_in_time = bool(contracts["vendor_valuation"]["point_in_time_vintage_proven"])
    valuation_gate = {
        "status": "NO_VIEW_BLOCKED_NON_VINTAGE_HISTORY",
        "return_calculation_allowed": False,
        "vendor_daily_history_rows": int(len(vendor_valuation)),
        "vendor_history_start": str(vendor_valuation["date"].min().date()),
        "vendor_history_end": str(vendor_valuation["date"].max().date()),
        "vendor_point_in_time_vintage_proven": vendor_point_in_time,
        "point_in_time_months": point_months,
        "point_in_time_start": str(point_fundamental["date"].min().date()),
        "point_in_time_end": str(point_fundamental["date"].max().date()),
        "minimum_required_months": int(config["valuation_models"]["point_in_time_minimum_history_months"]),
        "history_length_pass": point_months >= int(config["valuation_models"]["point_in_time_minimum_history_months"]),
        "failure_categories": [
            "BLOCKED_VENDOR_VALUATION_NOT_PROVEN_POINT_IN_TIME_VINTAGE",
            "BLOCKED_POINT_IN_TIME_FUNDAMENTAL_HISTORY_SHORTER_THAN_7_YEARS",
            "BLOCKED_NO_5Y_OR_7Y_WARMUP_AT_2021_EVALUATION_START",
        ],
    }
    r5_cap = constituents.loc[
        constituents["date"].between(start, end), "total_market_cap_cny"
    ]
    r5_gate = {
        "preserved_signal_rows": int(len(preserved_r5)),
        "preserved_signal_calendar_matches": set(preserved_r5["date"]) == set(evaluation_market["date"]),
        "preserved_component_fields_complete": bool(
            preserved_r5[["raw_continuous_position", "strategic_position", "trend_score", "risk_penalty", "crisis_event"]]
            .notna()
            .all()
            .all()
        ),
        "current_constituent_market_cap_non_null_ratio": float(r5_cap.notna().mean()),
        "raw_replay_status": "BLOCKED_SOURCE_DRIFT_CURRENT_MARKET_CAP_MISSING",
        "decomposition_input_status": "PASS_PRESERVED_DERIVED_SIGNAL_SNAPSHOT",
    }
    distribution_checks = {
        "event_count": int(len(dividends)),
        "complete_history_confirmed": bool(coverage.get("complete_history_confirmed")),
        "coverage_hash_matches": coverage.get("distribution_file_sha256") == contracts["dividends"]["sha256"],
        "status": "PASS" if len(dividends) == 14 and coverage.get("complete_history_confirmed") else "BLOCKED",
    }
    technical_gate_pass = all(
        [
            _market_audit(etf, "510300.SH", True)["status"] == "PASS",
            _market_audit(price_index, "000300.SH", True)["status"] == "PASS",
            _market_audit(total_return, "H00300", False)["status"] == "PASS",
            len(evaluation_market) == 1211,
            close_error <= 1e-10,
            breadth_checks["status"] == "PASS",
            distribution_checks["status"] == "PASS",
        ]
    )
    audit = {
        "project_id": config["protocol"]["project_id"],
        "data_cutoff": config["protocol"]["evaluation_end"],
        "hash_checks": hash_checks,
        "market": {
            "etf": _market_audit(etf, "510300.SH", True),
            "price_index": _market_audit(price_index, "000300.SH", True),
            "total_return_index": _market_audit(total_return, "H00300", False),
            "evaluation_rows": int(len(evaluation_market)),
            "synthetic_total_return_close_maximum_error": close_error,
        },
        "technical_gate": {
            "status": "PASS" if technical_gate_pass else "NO_VIEW",
            "return_calculation_allowed": bool(technical_gate_pass),
            "breadth": breadth_checks,
        },
        "valuation_gate": valuation_gate,
        "r5_replay_gate": r5_gate,
        "distributions": distribution_checks,
        "government_bond_rows": int(len(bonds)),
        "point_in_time_financial_record_rows": int(len(financials)),
    }
    datasets = {
        "market": market,
        "evaluation_market": evaluation_market.reset_index(drop=True),
        "breadth": breadth,
        "vendor_valuation": vendor_valuation,
        "point_fundamental": point_fundamental,
        "financials": financials,
        "bonds": bonds,
        "constituents": constituents,
        "preserved_r5": preserved_r5,
        "preserved_r5_baseline": preserved_r5_baseline,
        "dividends": dividends,
    }
    return datasets, audit


def _evaluation_slice(
    frame: pd.DataFrame, config: dict[str, Any], force_first_trade: bool = True
) -> pd.DataFrame:
    start = pd.Timestamp(config["protocol"]["evaluation_start"])
    end = pd.Timestamp(config["protocol"]["evaluation_end"])
    result = frame.loc[frame["date"].between(start, end)].copy().reset_index(drop=True)
    if len(result) != 1211:
        raise ValueError("模型目标未完整覆盖统一1211日评价日历")
    if force_first_trade:
        result.loc[0, "trade_allowed"] = True
        result.loc[0, "signal_reason"] = "统一评价首日建立当时状态"
    result["risk_off_override"] = False
    return result


def build_tech01_targets(
    market: pd.DataFrame, model: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    data = market.copy().sort_values("date").reset_index(drop=True)
    cash_daily = float(config["execution"]["cash_annual_rate"]) / int(
        config["execution"]["trading_days_per_year"]
    )
    signs = []
    for days in model["lookbacks"]:
        excess = data["signal_close"] / data["signal_close"].shift(days) - 1.0 - ((1.0 + cash_daily) ** days - 1.0)
        column = f"momentum_excess_{days}d"
        data[column] = excess
        signs.append(np.sign(excess))
    data["momentum_sign_sum"] = sum(signs)
    data["raw_target"] = np.select(
        [
            data["momentum_sign_sum"] <= -2,
            data["momentum_sign_sum"] <= 0,
            data["momentum_sign_sum"] <= 2,
        ],
        [0.0, 0.25, 0.75],
        default=1.0,
    )
    ready = data[[f"momentum_excess_{days}d" for days in model["lookbacks"]]].notna().all(axis=1)
    review = np.zeros(len(data), dtype=bool)
    ready_indices = np.flatnonzero(ready.to_numpy())
    review[ready_indices[:: int(model["review_every_trading_days"])]] = True
    scheduled = data["raw_target"].where(review).ffill()
    data["target_position"] = scheduled
    data["trade_allowed"] = review
    data["signal_reason"] = np.where(review, "五日复核多期限时间序列动量", "复核间隔维持")
    data["variant_id"] = model["id"]
    return _evaluation_slice(data, config)


def build_donchian_targets(
    market: pd.DataFrame, model: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    data = market.copy().sort_values("date").reset_index(drop=True)
    entry_days = int(model["entry_days"])
    exit_days = int(model["exit_days"])
    data["entry_channel"] = data["signal_high"].shift(1).rolling(entry_days).max()
    data["exit_channel"] = data["signal_low"].shift(1).rolling(exit_days).min()
    cash_state = float(model.get("cash_state_position", 0.0))
    if cash_state not in {0.0, 0.25}:
        raise ValueError("唐奇安现金状态只允许0%或已登记的25%底仓")
    targets = np.full(len(data), cash_state, dtype=float)
    actions = np.full(len(data), "HOLD", dtype=object)
    state = 0.0
    for index in range(len(data)):
        if state == 0.0 and np.isfinite(data.at[index, "entry_channel"]):
            if data.at[index, "signal_close"] > data.at[index, "entry_channel"]:
                state = 1.0
                actions[index] = "ENTER"
        elif state == 1.0 and np.isfinite(data.at[index, "exit_channel"]):
            if data.at[index, "signal_close"] < data.at[index, "exit_channel"]:
                state = 0.0
                actions[index] = "EXIT"
        targets[index] = 1.0 if state == 1.0 else cash_state
    data["target_position"] = targets
    data["donchian_action"] = actions
    data["trade_allowed"] = data["donchian_action"].ne("HOLD")
    data["signal_reason"] = data["donchian_action"].map(
        {"ENTER": "唐奇安突破进入", "EXIT": "唐奇安通道退出", "HOLD": "通道状态维持"}
    )
    data["variant_id"] = model["id"]
    return _evaluation_slice(data, config)


def _rolling_ols_slope(values: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    array = values.to_numpy(float)
    result = np.full(len(array), np.nan)
    for index in range(window - 1, len(array)):
        y = array[index - window + 1 : index + 1]
        if np.isfinite(y).all():
            result[index] = float(np.sum(x_centered * (y - y.mean())) / denominator)
    return pd.Series(result, index=values.index)


def build_tech05_targets(
    market: pd.DataFrame, model: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    data = market.copy().sort_values("date").reset_index(drop=True)
    close = data["signal_close"]
    data["ma120"] = close.rolling(int(model["ma_long_days"])).mean()
    data["slope60"] = _rolling_ols_slope(np.log(close), int(model["slope_days"]))
    data["ma20"] = close.rolling(int(model["z_days"])).mean()
    data["std20"] = close.rolling(int(model["z_days"])).std(ddof=1)
    data["z20"] = (close - data["ma20"]) / data["std20"].replace(0.0, np.nan)
    data["long_trend"] = close.gt(data["ma120"]) & data["slope60"].gt(0.0)
    targets = np.zeros(len(data), dtype=float)
    actions = np.full(len(data), "HOLD", dtype=object)
    state = 0.0
    holding_days = 0
    for index, row in data.iterrows():
        ready = pd.notna(row["z20"]) and pd.notna(row["ma120"]) and pd.notna(row["slope60"])
        if not ready:
            targets[index] = state
            continue
        if state > 0:
            holding_days += 1
            if (
                not bool(row["long_trend"])
                or float(row["z20"]) >= float(model["exit_z"])
                or holding_days >= int(model["maximum_holding_days"])
            ):
                state = 0.0
                holding_days = 0
                actions[index] = "EXIT"
            elif state == float(model["first_layer"]) and float(row["z20"]) <= float(model["second_zone_upper"]):
                state = float(model["second_layer"])
                actions[index] = "ADD_SECOND_LAYER"
        elif bool(row["long_trend"]) and float(row["z20"]) < float(model["first_zone_upper"]):
            state = float(model["first_layer"])
            holding_days = 0
            actions[index] = "ENTER_FIRST_LAYER"
        targets[index] = state
    data["target_position"] = targets
    data["pullback_action"] = actions
    data["trade_allowed"] = data["pullback_action"].ne("HOLD")
    data["signal_reason"] = data["pullback_action"].map(
        {
            "ENTER_FIRST_LAYER": "上涨趋势回撤首层",
            "ADD_SECOND_LAYER": "深回撤第二层",
            "EXIT": "趋势破坏、回归均线或期限退出",
            "HOLD": "回撤状态维持",
        }
    )
    data["variant_id"] = model["id"]
    return _evaluation_slice(data, config)


def build_tech08_targets(
    market: pd.DataFrame,
    breadth: pd.DataFrame,
    model: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    data = market.copy().sort_values("date").reset_index(drop=True)
    days = int(model["trend_ma_days"])
    data["trend_ma60"] = data["signal_close"].rolling(days).mean()
    data["trend_ma60_change20"] = data["trend_ma60"] / data["trend_ma60"].shift(
        int(model["trend_ma_slope_days"])
    ) - 1.0
    data = data.merge(
        breadth[["date", "official_weighted_above_ma60_share", "weight_snapshot_date", "weight_snapshot_age_calendar_days", "ma60_weight_coverage", "valid_for_direction_model"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    data["trend_up"] = data["signal_close"].gt(data["trend_ma60"]) & data[
        "trend_ma60_change20"
    ].gt(0.0)
    breadth_value = data["official_weighted_above_ma60_share"]
    data["target_position"] = np.select(
        [
            data["trend_up"] & breadth_value.ge(float(model["breadth_confirm"])),
            data["trend_up"] & breadth_value.ge(float(model["breadth_weak"])),
            data["trend_up"],
        ],
        [1.0, 0.75, 0.50],
        default=0.0,
    )
    data["trade_allowed"] = data["target_position"].ne(data["target_position"].shift(1))
    data["signal_reason"] = np.select(
        [
            ~data["trend_up"],
            breadth_value.ge(float(model["breadth_confirm"])),
            breadth_value.ge(float(model["breadth_weak"])),
        ],
        ["指数趋势未确认", "趋势与宽度共同确认", "趋势成立但宽度中性"],
        default="趋势成立但宽度弱",
    )
    data["variant_id"] = model["id"]
    result = _evaluation_slice(data, config)
    if result["official_weighted_above_ma60_share"].isna().any():
        raise ValueError("TECH-08评价期存在宽度缺口")
    return result


def build_all_technical_targets(
    datasets: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    market = datasets["market"]
    result: dict[str, pd.DataFrame] = {}
    for model in config["technical_models"]:
        if model["family"] == "TIME_SERIES_MOMENTUM":
            targets = build_tech01_targets(market, model, config)
        elif model["family"] in {
            "DONCHIAN",
            "DONCHIAN_FIXED_FLOOR",
            "DONCHIAN_FIXED_ROBUSTNESS",
        }:
            targets = build_donchian_targets(market, model, config)
        elif model["family"] == "TREND_PULLBACK":
            targets = build_tech05_targets(market, model, config)
        elif model["family"] == "POINT_IN_TIME_BREADTH_TREND":
            targets = build_tech08_targets(market, datasets["breadth"], model, config)
        else:
            raise ValueError(f"未知技术模型家族：{model['family']}")
        result[model["id"]] = targets
    return result


def build_valuation_engineering_audit(
    datasets: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valuation = datasets["vendor_valuation"].copy()
    settings = config["valuation_models"]
    valuation["implied_eps"] = valuation["index_close_pe_source"] / valuation["pe_ttm"]
    growth = valuation["implied_eps"].pct_change(
        int(settings["normalized_growth_lag_days"]), fill_method=None
    ).clip(*[float(value) for value in settings["normalized_growth_clip"]])
    valuation["normalized_growth"] = growth.rolling(
        int(settings["normalized_growth_median_days"]),
        min_periods=int(settings["normalized_growth_minimum_observations"]),
    ).median()
    valuation["raw_ey"] = 1.0 / valuation["pe_ttm"]
    valuation["normalized_ey"] = valuation["raw_ey"] * (1.0 + valuation["normalized_growth"])
    bonds = datasets["bonds"][["date", "cgb_10y"]].sort_values("date")
    valuation = pd.merge_asof(
        valuation.sort_values("date"),
        bonds,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    valuation["normalized_ey_spread"] = valuation["normalized_ey"] - valuation["cgb_10y"] / 100.0
    start = pd.Timestamp(config["protocol"]["evaluation_start"])
    end = pd.Timestamp(config["protocol"]["evaluation_end"])
    evaluation = valuation.loc[valuation["date"].between(start, end)]
    audit = {
        "formal_status": settings["formal_status"],
        "return_calculation_allowed": False,
        "formulas_implemented": ["raw_ey", "normalized_ey", "normalized_ey_spread"],
        "vendor_rows": int(len(valuation)),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_formula_coverage": {
            column: float(evaluation[column].notna().mean())
            for column in ("raw_ey", "normalized_ey", "cgb_10y", "normalized_ey_spread")
        },
        "registered_val01_trial_count": len(settings["val01_registered_trials"]),
        "registered_val02_trial_count": len(settings["val02_registered_trials"]),
        "signals_or_returns_emitted": False,
        "reason": "公式可计算不等于历史点时性通过；禁止用当前抓取的供应商PE历史替代历史版本",
    }
    return valuation, audit


def build_r5_component_targets(
    datasets: dict[str, pd.DataFrame], config: dict[str, Any], root: Path = ROOT
) -> dict[str, pd.DataFrame]:
    snapshot = datasets["preserved_r5"].copy()
    r5_config = yaml.safe_load(
        (root / config["data_contracts"]["r5_config"]["file"]).read_text(encoding="utf-8")
    )
    results: dict[str, pd.DataFrame] = {}
    for variant in config["r5_decomposition"]["variants"]:
        identifier = variant["id"]
        if not any((variant["valuation"], variant["trend"], variant["risk"])):
            targets = pd.DataFrame({"date": snapshot["date"], "target_position": 1.0})
            targets["trade_allowed"] = False
            targets.loc[0, "trade_allowed"] = True
            targets["risk_off_override"] = False
            targets["signal_reason"] = "买入持有首次建仓"
            results[identifier] = targets
            continue
        if variant["valuation"]:
            position = snapshot["strategic_position"].copy()
            if variant["trend"]:
                position = position + float(r5_config["valuation"]["trend_confirmation_amplitude"]) * snapshot["trend_score"]
        elif variant["trend"]:
            position = float(r5_config["trend"]["baseline_mid_position"]) + float(
                r5_config["trend"]["baseline_trend_amplitude"]
            ) * snapshot["trend_score"]
        else:
            position = pd.Series(1.0, index=snapshot.index)
        if variant["risk"]:
            position = position - snapshot["risk_penalty"]
        continuous = position.clip(0.0, 1.0)
        grid = float(r5_config["execution"]["position_grid_step"])
        model = snapshot[["date", "crisis_event"]].copy()
        if not variant["risk"]:
            model["crisis_event"] = False
        model["continuous_model_position"] = continuous
        model["discrete_model_position"] = (
            np.floor(continuous / grid + 0.5) * grid
        ).clip(0.0, 1.0)
        scheduled = schedule_asymmetric_execution(model, r5_config)
        outer_step = float(config["execution"]["allowed_position_grid"][1])
        scheduled["inner_target_position"] = scheduled["target_position"]
        scheduled["target_position"] = scheduled["target_position"].map(
            lambda value: quantize_position(value, outer_step)
        )
        scheduled["variant_id"] = identifier
        results[identifier] = scheduled
    return results


def make_execution_costs(config: dict[str, Any], stress: bool) -> BacktestCosts:
    execution = config["execution"]
    multiplier = float(execution["stress_multiplier"]) if stress else 1.0
    return BacktestCosts(
        commission_rate=float(execution["commission_rate"]) * multiplier,
        minimum_commission_cny=float(execution["minimum_commission_cny"]) * multiplier,
        stamp_duty_rate=0.0,
        slippage_bps=float(execution["slippage_bps_per_leg"]) * multiplier,
        lot_size=int(execution["lot_size_shares"]),
        cash_annual_rate=float(execution["cash_annual_rate"]),
    )


def build_buy_hold_targets(calendar: pd.Series, identifier: str = "BUY_HOLD") -> pd.DataFrame:
    targets = pd.DataFrame({"date": pd.to_datetime(calendar), "target_position": 1.0})
    targets["trade_allowed"] = False
    targets.loc[0, "trade_allowed"] = True
    targets["risk_off_override"] = False
    targets["signal_reason"] = "统一首日买入持有"
    targets["variant_id"] = identifier
    return targets


def build_h00300_benchmark(
    market: pd.DataFrame,
    initial_wealth: float,
    cash_annual_rate: float,
    trading_days: int,
) -> pd.DataFrame:
    """构造与T+1首次入场口径一致、无交易成本的H00300基准。"""

    data = market.copy().sort_values("date").reset_index(drop=True)
    equity = np.full(len(data), float(initial_wealth), dtype=float)
    if len(data) >= 2:
        equity[1] = initial_wealth * (1.0 + cash_annual_rate / trading_days) * float(
            data.loc[1, "signal_close"] / data.loc[1, "signal_open"]
        )
        for index in range(2, len(data)):
            equity[index] = equity[index - 1] * float(
                data.loc[index, "signal_close"] / data.loc[index - 1, "signal_close"]
            )
    result = pd.DataFrame({"date": data["date"], "equity": equity})
    result["daily_return"] = result["equity"].pct_change().fillna(0.0)
    result["actual_position"] = np.where(result.index == 0, 0.0, 1.0)
    result["equity_peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["equity_peak"] - 1.0
    return result


def build_fixed_mix_benchmark(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    exposure: float,
    initial_wealth: float,
    cash_annual_rate: float,
    trading_days: int,
) -> pd.DataFrame:
    """构造每日恒定权重的理论510300含分红/现金基准，不计调仓成本。"""

    if not 0.0 <= exposure <= 1.0:
        raise ValueError("固定仓位必须位于0到1之间")
    data = market.copy().sort_values("date").reset_index(drop=True)
    distributions = (
        dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    )
    etf_returns = np.zeros(len(data), dtype=float)
    cash_daily = cash_annual_rate / trading_days
    if len(data) >= 2:
        etf_returns[1] = (1.0 + cash_daily) * float(
            data.loc[1, "etf_close"] / data.loc[1, "etf_open"]
        ) - 1.0
    for index in range(2, len(data)):
        distribution = float(distributions.get(pd.Timestamp(data.loc[index, "date"]), 0.0))
        etf_returns[index] = float(
            (data.loc[index, "etf_close"] + distribution)
            / data.loc[index - 1, "etf_close"]
            - 1.0
        )
    returns = exposure * etf_returns + (1.0 - exposure) * cash_daily
    returns[0] = 0.0
    equity = initial_wealth * np.cumprod(1.0 + returns)
    result = pd.DataFrame(
        {
            "date": data["date"],
            "equity": equity,
            "daily_return": returns,
            "actual_position": exposure,
        }
    )
    result["equity_peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["equity_peak"] - 1.0
    return result


def simulate_fractional_targets(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    initial_wealth: float,
    costs: FractionalCosts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = market.copy().sort_values("date").reset_index(drop=True)
    signals = targets.copy().sort_values("date").reset_index(drop=True)
    signal_map = signals.set_index("date").to_dict("index")
    events = dividends.copy()
    ex_map = {date: frame.to_dict("records") for date, frame in events.groupby("ex_date")}
    cash = float(initial_wealth)
    shares = 0.0
    receivable = 0.0
    scheduled: dict[pd.Timestamp, float] = {}
    previous_date: pd.Timestamp | None = None
    previous_target = 0.0
    ledger_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for row in prices.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if previous_date is not None:
            cash *= 1.0 + costs.cash_annual_rate / costs.trading_days_per_year
        shares_at_start = shares
        entitlement_today = 0.0
        for event in ex_map.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            entitlement_today += entitlement
            payment_date = pd.Timestamp(event["payment_date"])
            scheduled[payment_date] = scheduled.get(payment_date, 0.0) + entitlement
        receivable += entitlement_today
        signal = signal_map.get(previous_date, {}) if previous_date is not None else {}
        if signal:
            previous_target = float(signal["target_position"])
        trade_allowed = bool(signal.get("trade_allowed", False))
        raw_open = float(row.etf_open)
        open_equity = cash + shares * raw_open + receivable
        current_value = shares * raw_open
        desired_direction = previous_target * open_equity - current_value
        reference_price = raw_open * (
            1.0 + costs.slippage_bps / 10000.0
            if desired_direction >= 0
            else 1.0 - costs.slippage_bps / 10000.0
        )
        desired_shares = previous_target * open_equity / reference_price
        quantity = desired_shares - shares if trade_allowed else 0.0
        side = "NONE"
        commission = 0.0
        slippage_cost = 0.0
        execution_notional = 0.0
        cash_before = cash
        if quantity > 1e-12:
            maximum = cash / (reference_price * (1.0 + costs.commission_rate))
            quantity = min(quantity, maximum)
            if quantity > 1e-12:
                side = "BUY"
                execution_notional = quantity * reference_price
                commission = execution_notional * costs.commission_rate
                cash -= execution_notional + commission
                shares += quantity
                slippage_cost = quantity * (reference_price - raw_open)
        elif quantity < -1e-12:
            sell_quantity = min(-quantity, shares_at_start)
            if sell_quantity > 1e-12:
                side = "SELL"
                execution_notional = sell_quantity * reference_price
                commission = execution_notional * costs.commission_rate
                cash += execution_notional - commission
                shares -= sell_quantity
                quantity = -sell_quantity
                slippage_cost = sell_quantity * (raw_open - reference_price)
        if side != "NONE":
            trade_rows.append(
                {
                    "date": date,
                    "signal_date": previous_date,
                    "side": side,
                    "quantity": float(abs(quantity)),
                    "signed_quantity": float(quantity),
                    "open_price": raw_open,
                    "execution_price": reference_price,
                    "execution_notional": execution_notional,
                    "commission": commission,
                    "slippage_cost": slippage_cost,
                    "cash_before_trade": cash_before,
                    "cash_after_trade": cash,
                    "target_position": previous_target,
                }
            )
        payment_today = float(scheduled.pop(date, 0.0))
        if payment_today:
            cash += payment_today
            receivable -= payment_today
            if abs(receivable) < 1e-12:
                receivable = 0.0
        equity = cash + shares * float(row.etf_close) + receivable
        ledger_rows.append(
            {
                "date": date,
                "shares": shares,
                "cash": cash,
                "dividend_receivable": receivable,
                "dividend_entitlement_today": entitlement_today,
                "dividend_payment_today": payment_today,
                "close": float(row.etf_close),
                "equity": equity,
                "actual_position": shares * float(row.etf_close) / equity if equity else 0.0,
                "daily_commission": commission,
                "daily_slippage_cost": slippage_cost,
            }
        )
        previous_date = date
    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    return ledger, pd.DataFrame(trade_rows)


def _longest_drawdown(equity: pd.Series) -> int:
    underwater = equity < equity.cummax() - 1e-12
    current = longest = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def summarize(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_wealth: float,
    config: dict[str, Any],
    minimum_commission: float = 0.0,
) -> dict[str, Any]:
    trading_days = int(config["execution"]["trading_days_per_year"])
    cash_rate = float(config["execution"]["cash_annual_rate"])
    returns = ledger["daily_return"].astype(float)
    observations = max(len(ledger) - 1, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_wealth - 1.0)
    cagr = float((1.0 + total_return) ** (trading_days / observations) - 1.0)
    sample = returns.iloc[1:]
    excess = sample - cash_rate / trading_days
    volatility = float(sample.std(ddof=1) * np.sqrt(trading_days))
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(trading_days)) if excess.std(ddof=1) > 0 else None
    downside = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt((downside**2).mean()) * np.sqrt(trading_days))
    sortino = float(excess.mean() * trading_days / downside_deviation) if downside_deviation > 0 else None
    maximum_drawdown = float(ledger["drawdown"].min())
    if trades.empty:
        trade_count = 0
        turnover = 0.0
        commission = 0.0
        slippage = 0.0
        minimum_ratio = 0.0
    else:
        trade_count = int(len(trades))
        quantity = trades["quantity"].astype(float)
        execution_price = trades["execution_price"].astype(float)
        turnover = float((quantity * execution_price).sum() / ledger["equity"].mean())
        commission = float(trades["commission"].sum())
        if "slippage_cost" in trades:
            slippage = float(trades["slippage_cost"].sum())
        else:
            slippage = float((abs(trades["execution_price"] - trades["open_price"]) * quantity).sum())
        minimum_ratio = (
            float(np.isclose(trades["commission"], minimum_commission, atol=1e-9).mean())
            if minimum_commission > 0
            else 0.0
        )
    positive = ledger["actual_position"].astype(float).gt(1e-10).to_numpy()
    holding_spells: list[int] = []
    current_spell = 0
    for is_invested in positive:
        if is_invested:
            current_spell += 1
        elif current_spell:
            holding_spells.append(current_spell)
            current_spell = 0
    if current_spell:
        holding_spells.append(current_spell)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_excess_cash": sharpe,
        "sortino_excess_cash": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar": float(cagr / abs(maximum_drawdown)) if maximum_drawdown < 0 else None,
        "longest_drawdown_recovery_trading_days": _longest_drawdown(ledger["equity"]),
        "average_exposure": float(ledger["actual_position"].mean()),
        "average_positive_exposure_spell_trading_days": (
            float(np.mean(holding_spells)) if holding_spells else 0.0
        ),
        "trade_count": trade_count,
        "turnover_over_average_equity": turnover,
        "commission": commission,
        "slippage_cost": slippage,
        "total_execution_cost": commission + slippage,
        "minimum_commission_trade_ratio": minimum_ratio,
        "ending_equity": float(ledger["equity"].iloc[-1]),
    }


def relative_metrics(
    strategy: pd.DataFrame, benchmark: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = strategy[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_benchmark"),
        validate="one_to_one",
    )
    trading_days = int(config["execution"]["trading_days_per_year"])
    strategy_return = merged["daily_return_strategy"].astype(float)
    benchmark_return = merged["daily_return_benchmark"].astype(float)
    active = strategy_return - benchmark_return
    n = max(len(merged) - 1, 1)
    cagr_strategy = float(np.prod(1.0 + strategy_return) ** (trading_days / n) - 1.0)
    cagr_benchmark = float(np.prod(1.0 + benchmark_return) ** (trading_days / n) - 1.0)
    active_std = active.iloc[1:].std(ddof=1)
    relative_wealth = ((1.0 + strategy_return) / (1.0 + benchmark_return)).cumprod()
    active_drawdown = relative_wealth / relative_wealth.cummax() - 1.0
    up = benchmark_return > 0
    down = benchmark_return < 0
    rolling = {}
    for window in config["evaluation"]["rolling_windows_trading_days"]:
        values = (
            (1.0 + strategy_return).rolling(window).apply(np.prod, raw=True)
            / (1.0 + benchmark_return).rolling(window).apply(np.prod, raw=True)
            - 1.0
        ).dropna()
        rolling[str(window)] = {
            "observations": int(len(values)),
            "median": float(values.median()) if len(values) else None,
            "positive_ratio": float((values > 0).mean()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
        }
    result = {
        "annualized_active_return": cagr_strategy - cagr_benchmark,
        "information_ratio": float(active.iloc[1:].mean() / active_std * np.sqrt(trading_days)) if active_std > 0 else None,
        "upside_capture": float(strategy_return[up].sum() / benchmark_return[up].sum()) if benchmark_return[up].sum() else None,
        "downside_capture": float(strategy_return[down].sum() / benchmark_return[down].sum()) if benchmark_return[down].sum() else None,
        "active_maximum_drawdown": float(active_drawdown.min()),
        "daily_relative_win_rate": float((strategy_return.iloc[1:] > benchmark_return.iloc[1:]).mean()),
        "rolling_active_returns": rolling,
    }
    active_frame = merged[["date"]].copy()
    active_frame["active_return"] = active
    return result, active_frame


def stability_diagnostics(
    strategy: pd.DataFrame, benchmark: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    merged = strategy[["date", "daily_return"]].merge(
        benchmark[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_benchmark"),
        validate="one_to_one",
    )
    midpoint = len(merged) // 2
    halves = []
    for label, frame in (("H1", merged.iloc[:midpoint]), ("H2", merged.iloc[midpoint:])):
        relative = float(
            np.prod((1.0 + frame["daily_return_strategy"]) / (1.0 + frame["daily_return_benchmark"])) - 1.0
        )
        halves.append({"period": label, "active_return": relative})
    merged["year"] = merged["date"].dt.year
    annual_rows = []
    for year, frame in merged.groupby("year"):
        relative = float(
            np.prod((1.0 + frame["daily_return_strategy"]) / (1.0 + frame["daily_return_benchmark"])) - 1.0
        )
        annual_rows.append({"year": int(year), "active_return": relative})
    annual = pd.DataFrame(annual_rows)
    best_year = int(annual.loc[annual["active_return"].idxmax(), "year"])
    without = merged.loc[merged["year"].ne(best_year)]
    n = max(len(without) - 1, 1)
    annualized_without = float(
        np.prod((1.0 + without["daily_return_strategy"]) / (1.0 + without["daily_return_benchmark"]))
        ** (int(config["execution"]["trading_days_per_year"]) / n)
        - 1.0
    )
    positive_logs = np.log1p(annual.loc[annual["active_return"] > 0, "active_return"])
    concentration = float(positive_logs.max() / positive_logs.sum()) if positive_logs.sum() > 0 else None
    return {
        "chronological_halves": halves,
        "both_halves_positive": all(item["active_return"] > 0 for item in halves),
        "annual": annual.to_dict("records"),
        "positive_year_ratio": float((annual["active_return"] > 0).mean()),
        "best_year_removed": best_year,
        "annualized_active_return_without_best_year": annualized_without,
        "maximum_positive_year_contribution_share": concentration,
    }


def classify_model(
    base: dict[str, Any],
    benchmark: dict[str, Any],
    relative: dict[str, Any],
    stress_relative: dict[str, Any],
    stability: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["evaluation"]["historical_gates"]
    checks = {
        "annualized_active_return_at_least_1_5pct": relative["annualized_active_return"] >= float(gates["annualized_active_return_minimum"]),
        "information_ratio_at_least_0_35": relative["information_ratio"] is not None and relative["information_ratio"] >= float(gates["information_ratio_minimum"]),
        "both_chronological_halves_positive": stability["both_halves_positive"],
        "double_cost_active_return_positive": stress_relative["annualized_active_return"] > 0,
        "remove_best_year_active_return_positive": stability["annualized_active_return_without_best_year"] > 0,
        "positive_year_ratio_majority": stability["positive_year_ratio"] > float(gates["positive_year_ratio_minimum"]),
        "single_positive_year_contribution_not_over_50pct": stability["maximum_positive_year_contribution_share"] is not None and stability["maximum_positive_year_contribution_share"] <= float(gates["maximum_positive_year_contribution_share"]),
    }
    risk_gates = config["evaluation"]["risk_overlay_gates"]
    sharpe_improvement = (
        base["sharpe_excess_cash"] - benchmark["sharpe_excess_cash"]
        if base["sharpe_excess_cash"] is not None and benchmark["sharpe_excess_cash"] is not None
        else None
    )
    drawdown_reduction = 1.0 - abs(base["maximum_drawdown"]) / abs(benchmark["maximum_drawdown"])
    risk_checks = {
        "sharpe_improvement_at_least_0_15": sharpe_improvement is not None and sharpe_improvement >= float(risk_gates["sharpe_improvement_minimum"]),
        "drawdown_reduction_at_least_20pct": drawdown_reduction >= float(risk_gates["drawdown_reduction_minimum"]),
        "upside_capture_at_least_70pct": relative["upside_capture"] is not None and relative["upside_capture"] >= float(risk_gates["upside_capture_minimum"]),
    }
    if all(checks.values()):
        status = "HISTORICAL_SCREEN_PASS_FORWARD_REQUIRED"
    elif all(risk_checks.values()) and not checks["annualized_active_return_at_least_1_5pct"]:
        status = "RISK_OVERLAY_CANDIDATE_HISTORICAL_ONLY"
    else:
        status = "REJECT_HISTORICAL"
    return {
        "status": status,
        "alpha_pass": False,
        "alpha_pass_blocker": "TRUE_OOS_NOT_STARTED",
        "historical_checks": checks,
        "risk_overlay_checks": risk_checks,
        "sharpe_improvement": sharpe_improvement,
        "drawdown_reduction": drawdown_reduction,
    }


def multiple_testing_diagnostics(
    active_frames: dict[str, pd.DataFrame], config: dict[str, Any]
) -> dict[str, Any]:
    identifiers = list(active_frames)
    merged: pd.DataFrame | None = None
    for identifier, frame in active_frames.items():
        current = frame.rename(columns={"active_return": identifier})
        merged = current if merged is None else merged.merge(current, on="date", validate="one_to_one")
    assert merged is not None
    matrix = merged[identifiers].to_numpy(float)[1:]
    settings = config["evaluation"]["multiple_testing"]
    n, candidate_count = matrix.shape
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1)
    observed_rc = float(np.sqrt(n) * means.max())
    observed_spa = float(np.max(np.sqrt(n) * means / np.where(stds > 0, stds, np.inf)))
    centered = matrix - means
    repetitions = int(settings["repetitions"])
    block = int(settings["circular_block_length_trading_days"])
    rng = np.random.default_rng(int(settings["random_seed"]))
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    rc_boot = np.empty(repetitions)
    spa_boot = np.empty(repetitions)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sample = centered[indices]
        sample_means = sample.mean(axis=0)
        sample_stds = sample.std(axis=0, ddof=1)
        rc_boot[repetition] = np.sqrt(n) * sample_means.max()
        spa_boot[repetition] = np.max(
            np.sqrt(n) * sample_means / np.where(sample_stds > 0, sample_stds, np.inf)
        )
    registered = int(settings["registered_technical_trial_count"])
    expected_max_z = float(norm.ppf((registered - 0.375) / (registered + 0.25)))
    dsr = {}
    annualization = int(config["execution"]["trading_days_per_year"])
    for index, identifier in enumerate(identifiers):
        values = matrix[:, index]
        daily_sr = float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) > 0 else 0.0
        null_sr = expected_max_z / np.sqrt(max(n - 1, 1))
        sample_skew = float(skew(values, bias=False))
        sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
        denominator = np.sqrt(
            max(1e-12, 1.0 - sample_skew * daily_sr + ((sample_kurtosis - 1.0) / 4.0) * daily_sr**2)
        )
        z_value = (daily_sr - null_sr) * np.sqrt(max(n - 1, 1)) / denominator
        dsr[identifier] = {
            "annualized_active_sharpe": daily_sr * np.sqrt(annualization),
            "probability": float(norm.cdf(z_value)),
        }

    slice_count = int(settings["cscv_slices"])
    slices = [np.asarray(item, dtype=int) for item in np.array_split(np.arange(n), slice_count)]
    pbo_records = []
    for train_slices in combinations(range(slice_count), slice_count // 2):
        train_set = set(train_slices)
        train_indices = np.concatenate([slices[index] for index in train_slices])
        test_indices = np.concatenate([slices[index] for index in range(slice_count) if index not in train_set])
        train = matrix[train_indices]
        test = matrix[test_indices]
        train_scores = train.mean(axis=0) / np.where(train.std(axis=0, ddof=1) > 0, train.std(axis=0, ddof=1), np.inf)
        test_scores = test.mean(axis=0) / np.where(test.std(axis=0, ddof=1) > 0, test.std(axis=0, ddof=1), np.inf)
        selected = int(np.argmax(train_scores))
        rank_from_worst = int(np.argsort(np.argsort(test_scores))[selected]) + 1
        relative_rank = rank_from_worst / (candidate_count + 1.0)
        logit = float(np.log(relative_rank / (1.0 - relative_rank)))
        pbo_records.append(
            {
                "selected_candidate": identifiers[selected],
                "oos_relative_rank": relative_rank,
                "logit": logit,
                "overfit": relative_rank < 0.5,
            }
        )
    pbo = float(np.mean([item["overfit"] for item in pbo_records]))
    return {
        "registered_technical_trial_count": registered,
        "return_tested_candidate_count": candidate_count,
        "candidate_ids": identifiers,
        "common_observations": n,
        "white_reality_check": {
            "p_value": float((1 + np.sum(rc_boot >= observed_rc)) / (repetitions + 1)),
            "observed_statistic": observed_rc,
            "method": "20日循环区块居中主动收益自助法",
        },
        "hansen_spa": {
            "p_value": float((1 + np.sum(spa_boot >= observed_spa)) / (repetitions + 1)),
            "observed_statistic": observed_spa,
            "method": "学生化20日循环区块自助近似",
        },
        "deflated_sharpe_ratio": dsr,
        "pbo": {
            "method": "8折CSCV",
            "split_count": len(pbo_records),
            "value": pbo,
            "pass": pbo <= float(settings["pbo_maximum"]),
            "records": pbo_records,
        },
    }


def shapley_r5(active_values: dict[tuple[bool, bool, bool], float]) -> dict[str, float]:
    components = ["V", "T", "R"]
    result = {component: 0.0 for component in components}
    n = len(components)
    for index, component in enumerate(components):
        other_indices = [value for value in range(n) if value != index]
        for size in range(len(other_indices) + 1):
            for subset in combinations(other_indices, size):
                mask = [False] * n
                for item in subset:
                    mask[item] = True
                with_component = mask.copy()
                with_component[index] = True
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                result[component] += weight * (
                    active_values[tuple(with_component)] - active_values[tuple(mask)]
                )
    return result


def build_event_audit(targets: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    events = targets.loc[targets["trade_allowed"].astype(bool)].copy()
    if trades.empty:
        events["execution_status"] = "NO_TRADE"
        return events
    executed = trades.copy()
    executed["signal_date"] = pd.to_datetime(executed["signal_date"]).dt.normalize()
    selected_columns = [
        "signal_date",
        "date",
        "side",
        "quantity",
        "open_price",
        "execution_price",
        "commission",
        "target_position",
    ]
    executed = executed[selected_columns].rename(columns={"date": "execution_date"})
    result = events.merge(
        executed,
        left_on="date",
        right_on="signal_date",
        how="left",
        suffixes=("_signal", "_execution"),
    )
    result["execution_status"] = np.where(result["execution_date"].notna(), "EXECUTED", "NO_TRADE_TARGET_UNCHANGED_OR_BELOW_THRESHOLD")
    return result
