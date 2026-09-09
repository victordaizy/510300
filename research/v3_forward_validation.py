"""V3_FORWARD_1前瞻信号、到期结果和固定监控协议。"""

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
from scipy.stats import spearmanr

from research.build_point_in_time_fundamental_panel import select_latest_vintages
from research.valuation_v2_expected_return import (
    BOND_FILE,
    CONSTITUENT_FILE,
    FAIR_PE_FEATURES,
    FINANCIALS_FILE,
    INDEX_FILE,
    INDUSTRY_FILE,
    TOTAL_RETURN_FILE,
    VALUATION_FILE,
    WEIGHTS_FILE,
    aggregate_v2_snapshot,
    build_market_monthly,
    derive_industry_aware_normalization,
    expanding_conditioned_fair_pe_v2,
)
from research.valuation_v3_state_aware_return import (
    exact_non_overlapping_queues,
    moving_block_bootstrap_ic,
    newey_west_slope,
    v3_expected_return,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "v3_forward_1.yaml"
MANIFEST_FILE = ROOT / "config" / "v3_forward_1_manifest.json"
FORWARD_DIR = ROOT / "data" / "forward" / "v3_forward_1"
SIGNAL_LOG = FORWARD_DIR / "valuation_forward_signal_log.parquet"
OUTCOME_LOG = FORWARD_DIR / "valuation_forward_outcome_log.parquet"
SHADOW_LOG = FORWARD_DIR / "valuation_shadow_position_log.parquet"
STATUS_FILE = ROOT / "paper" / "v3_forward_1_latest_status.json"
REPORT_DIR = ROOT / "reports" / "forward" / "v3_forward_1"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
FORWARD_CONSTITUENT_CLOSE_FILE = (
    ROOT / "data" / "raw" / "forward" / "v3_forward_1_constituent_close.parquet"
)
V3_HISTORY_FILE = ROOT / "data" / "features" / "000300_valuation_v3_state_aware_return.parquet"

HORIZONS = (20, 60, 120, 242)
ZERO_HASH = "0" * 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def record_hash(record: dict[str, Any]) -> str:
    payload = {
        key: _canonical(value)
        for key, value in record.items()
        if key != "record_hash"
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_signal_hash_chain(signals: pd.DataFrame) -> None:
    """验证信号日志顺序、前序哈希和逐行内容哈希。"""

    if signals.empty:
        return
    data = signals.sort_values("signal_date").reset_index(drop=True)
    if data["signal_date"].duplicated().any():
        raise ValueError("前瞻信号日志存在重复signal_date")
    expected_previous = ZERO_HASH
    for row in data.to_dict("records"):
        if row["previous_record_hash"] != expected_previous:
            raise ValueError(f"{row['signal_date']}的前序哈希不一致")
        expected = record_hash(row)
        if row["record_hash"] != expected:
            raise ValueError(f"{row['signal_date']}的内容哈希不一致")
        expected_previous = expected


def append_signal_record(
    existing: pd.DataFrame,
    record: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """只允许日期递增追加；同日完全相同为幂等，不允许覆盖。"""

    data = existing.copy()
    if not data.empty:
        data["signal_date"] = pd.to_datetime(data["signal_date"])
        verify_signal_hash_chain(data)
    signal_date = pd.Timestamp(record["signal_date"])
    if not data.empty and signal_date < data["signal_date"].max():
        raise ValueError("禁止向前瞻信号日志回填更早日期")
    same = data.loc[data["signal_date"].eq(signal_date)] if not data.empty else data
    if not same.empty:
        old = same.iloc[-1].to_dict()
        comparable = {key: old.get(key) for key in record}
        if all(_canonical(comparable[key]) == _canonical(record[key]) for key in record):
            return data.sort_values("signal_date").reset_index(drop=True), "IDEMPOTENT"
        raise ValueError("同一signal_date已有不同内容，禁止覆盖历史信号")
    previous = ZERO_HASH if data.empty else str(data.sort_values("signal_date").iloc[-1]["record_hash"])
    row = dict(record)
    row["previous_record_hash"] = previous
    row["record_hash"] = record_hash(row)
    combined = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    combined = combined.sort_values("signal_date").reset_index(drop=True)
    verify_signal_hash_chain(combined)
    return combined, "APPENDED"


def shadow_position_for_edge(edge_60d: float, mapping: list[dict[str, Any]]) -> float:
    for item in mapping:
        maximum = item.get("maximum_edge")
        if maximum is None or edge_60d <= float(maximum):
            return float(item["target_position"])
    raise ValueError("Shadow仓位映射没有覆盖全部Edge")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def validate_frozen_manifest(manifest_file: Path = MANIFEST_FILE) -> dict[str, Any]:
    if not manifest_file.exists():
        raise FileNotFoundError("缺少V3_FORWARD_1冻结清单")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for relative, expected in manifest["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"冻结文件指纹不一致：{relative}；必须创建新模型版本")
    return manifest


def _latest_at_or_before(
    frame: pd.DataFrame,
    date_column: str,
    signal_date: pd.Timestamp,
    maximum_age_days: int | None = None,
) -> pd.Series:
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    eligible = data.loc[data[date_column].le(signal_date)].sort_values(date_column)
    if eligible.empty:
        raise ValueError(f"{date_column}在信号日前没有可用记录")
    row = eligible.iloc[-1]
    if maximum_age_days is not None:
        age = (signal_date - pd.Timestamp(row[date_column])).days
        if age > maximum_age_days:
            raise ValueError(f"{date_column}最新记录已过期{age}天")
    return row


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def _current_market_states(
    signal_date: pd.Timestamp,
    norm_eps: float,
    market_monthly: pd.DataFrame,
    existing_signals: pd.DataFrame,
) -> tuple[str, str, str]:
    """严格按冻结定义生成H1/H2/H3状态，不增加其他分类。"""

    history = pd.read_parquet(V3_HISTORY_FILE)[
        ["date", "component_normalized_index_eps"]
    ].copy()
    history["date"] = pd.to_datetime(history["date"])
    if not existing_signals.empty:
        forward_eps = existing_signals[["signal_date", "norm_eps"]].rename(
            columns={"signal_date": "date", "norm_eps": "component_normalized_index_eps"}
        )
        forward_eps["date"] = pd.to_datetime(forward_eps["date"])
        history = pd.concat([history, forward_eps], ignore_index=True)
    anchor_date = signal_date - pd.DateOffset(years=1)
    anchor = history.loc[history["date"].le(anchor_date)].sort_values("date")
    if anchor.empty:
        raise ValueError("缺少12个月前正常化EPS状态锚")
    earnings_state = (
        "盈利上行"
        if norm_eps >= float(anchor.iloc[-1]["component_normalized_index_eps"])
        else "盈利下行"
    )

    monthly = market_monthly.sort_values("date").reset_index(drop=True)
    if len(monthly) < 10:
        raise ValueError("不足10个月市场历史，无法生成趋势状态")
    current_close = float(monthly.iloc[-1]["index_close"])
    ma_10m = float(monthly["index_close"].tail(10).mean())
    prior_3m_close = float(monthly.iloc[-4]["index_close"])
    momentum_3m = current_close / prior_3m_close - 1.0
    if current_close > ma_10m and momentum_3m > 0:
        market_state = "牛市"
    elif current_close < ma_10m and momentum_3m < 0:
        market_state = "熊市"
    else:
        market_state = "震荡"

    prior_volatility = monthly.loc[
        monthly["date"].lt(signal_date), "realized_volatility_3m"
    ].dropna()
    if len(prior_volatility) < 12:
        raise ValueError("不足12个月历史波动率，无法生成波动状态")
    current_volatility = float(monthly.iloc[-1]["realized_volatility_3m"])
    volatility_state = (
        "高波动" if current_volatility >= float(prior_volatility.median()) else "低波动"
    )
    return earnings_state, market_state, volatility_state


def build_forward_signal_snapshot(
    signal_date: pd.Timestamp,
    generated_at: datetime,
    enforce_same_local_date: bool = True,
) -> dict[str, Any]:
    """仅从信号日已可得数据生成一次不可回写的V3_FORWARD_1快照。"""

    config = _load_config()
    signal_date = pd.Timestamp(signal_date).normalize()
    local_date = pd.Timestamp(generated_at.astimezone(ZoneInfo("Asia/Shanghai"))).normalize().tz_localize(None)
    if signal_date < pd.Timestamp(config["model"]["true_oos_start"]):
        raise ValueError("signal_date早于真正OOS起点")
    if enforce_same_local_date and signal_date != local_date:
        raise ValueError("禁止事后回填：signal_date必须等于本地生成日期")

    index_daily = pd.read_parquet(INDEX_FILE)
    total_return = pd.read_parquet(TOTAL_RETURN_FILE)
    etf_daily = pd.read_parquet(ETF_FILE)
    constituent_daily = pd.read_parquet(CONSTITUENT_FILE)
    weights = pd.read_parquet(WEIGHTS_FILE)
    financials = pd.read_parquet(FINANCIALS_FILE)
    valuation = pd.read_parquet(VALUATION_FILE)
    bonds = pd.read_parquet(BOND_FILE)
    industry = pd.read_parquet(INDUSTRY_FILE)
    for frame in (index_daily, total_return, etf_daily, constituent_daily, valuation, bonds):
        frame["date"] = pd.to_datetime(frame["date"])
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    financials["available_at"] = pd.to_datetime(financials["available_at"])
    financials["report_period"] = pd.to_datetime(financials["report_period"])

    for name, frame in (
        ("000300", index_daily),
        ("H00300", total_return),
        ("510300", etf_daily),
    ):
        if not frame["date"].eq(signal_date).any():
            raise ValueError(f"{name}没有{signal_date.date()}收盘数据，不生成信号")

    snapshot_weights = weights.loc[weights["trade_date"].le(signal_date)]
    if snapshot_weights.empty:
        raise ValueError("信号日前没有官方权重")
    weight_date = pd.Timestamp(snapshot_weights["trade_date"].max())
    snapshot_weights = snapshot_weights.loc[
        snapshot_weights["trade_date"].eq(weight_date)
    ].copy()
    symbols = set(snapshot_weights["con_code"].astype(str))
    if FORWARD_CONSTITUENT_CLOSE_FILE.exists():
        forward_prices = pd.read_parquet(FORWARD_CONSTITUENT_CLOSE_FILE)
        forward_prices["date"] = pd.to_datetime(forward_prices["date"])
        current_prices = forward_prices.loc[
            forward_prices["date"].eq(signal_date)
            & forward_prices["con_code"].astype(str).isin(symbols),
            ["con_code", "raw_close"],
        ].drop_duplicates("con_code", keep="last")
    else:
        current_prices = pd.DataFrame(columns=["con_code", "raw_close"])
    if current_prices["con_code"].nunique() != len(symbols):
        historical_prices = constituent_daily.loc[
            constituent_daily["date"].eq(signal_date)
            & constituent_daily["con_code"].astype(str).isin(symbols),
            ["con_code", "raw_close"],
        ].drop_duplicates("con_code", keep="last")
        current_prices = pd.concat([current_prices, historical_prices], ignore_index=True)
        current_prices = current_prices.drop_duplicates("con_code", keep="first")
    if current_prices["con_code"].nunique() != len(symbols):
        raise ValueError(
            f"成分股当日价格覆盖不足：{current_prices['con_code'].nunique()}/{len(symbols)}"
        )
    vintages = select_latest_vintages(
        financials.loc[
            financials["con_code"].astype(str).isin(symbols)
            & financials["available_at"].le(signal_date)
        ],
        signal_date,
    )
    metrics = derive_industry_aware_normalization(vintages, industry)
    index_close = float(
        index_daily.loc[index_daily["date"].eq(signal_date), "close"].iloc[-1]
    )
    aggregate = aggregate_v2_snapshot(
        snapshot_weights,
        current_prices,
        metrics,
        signal_date,
        index_close,
    )
    if aggregate["normalized_earnings_weight_coverage"] < 0.98:
        raise ValueError("正常化盈利权重覆盖低于98%")

    maximum_age = int(config["data_governance"]["valuation_forward_fill_max_calendar_days"])
    _latest_at_or_before(valuation, "date", signal_date, maximum_age)
    maximum_bond_age = int(config["data_governance"]["macro_forward_fill_max_calendar_days"])
    _latest_at_or_before(bonds, "date", signal_date, maximum_bond_age)
    market_monthly = build_market_monthly(
        index_daily.loc[index_daily["date"].le(signal_date)],
        total_return.loc[total_return["date"].le(signal_date)],
        valuation.loc[valuation["date"].le(signal_date)],
        bonds.loc[bonds["date"].le(signal_date)],
    )
    current_fundamental = pd.DataFrame([aggregate])
    fair = expanding_conditioned_fair_pe_v2(market_monthly, current_fundamental)
    fair = fair.loc[fair["date"].eq(signal_date)]
    if fair.empty:
        raise ValueError("FairPE扩展窗口没有生成当前信号")
    fair_pe = float(fair.iloc[-1]["conditioned_fair_pe_v2"])
    current_market = market_monthly.loc[market_monthly["date"].eq(signal_date)].iloc[-1]
    existing = pd.read_parquet(SIGNAL_LOG) if SIGNAL_LOG.exists() else pd.DataFrame()
    earnings_state, market_state, vol_state = _current_market_states(
        signal_date,
        float(aggregate["component_normalized_index_eps"]),
        market_monthly,
        existing,
    )
    annual_lambda = float(config["frozen_parameters"]["annual_mean_reversion_lambda"])
    results = {
        horizon: v3_expected_return(
            normalized_eps=float(aggregate["component_normalized_index_eps"]),
            current_index=index_close,
            current_normalized_pe=float(aggregate["component_normalized_pe"]),
            annual_growth=float(aggregate["expected_earnings_growth_annual"]),
            fair_pe=fair_pe,
            annual_dividend_yield=float(current_market["trailing_dividend_yield_annual"]),
            annual_cash_rate=float(current_market["cgb_1y"] / 100.0),
            horizon_days=horizon,
            annual_reversion_speed=annual_lambda,
        )
        for horizon in HORIZONS
    }
    edge_60 = results[60]["expected_excess_total_return"]
    shadow_position = shadow_position_for_edge(
        edge_60, config["shadow_position"]["mapping"]
    )
    input_paths = (
        INDEX_FILE,
        TOTAL_RETURN_FILE,
        ETF_FILE,
        CONSTITUENT_FILE,
        WEIGHTS_FILE,
        FINANCIALS_FILE,
        VALUATION_FILE,
        BOND_FILE,
        INDUSTRY_FILE,
    )
    fingerprint_paths = [*input_paths]
    if FORWARD_CONSTITUENT_CLOSE_FILE.exists():
        fingerprint_paths.append(FORWARD_CONSTITUENT_CLOSE_FILE)
    record: dict[str, Any] = {
        "signal_date": signal_date,
        "data_available_time": generated_at.isoformat(),
        "norm_eps": float(aggregate["component_normalized_index_eps"]),
        "growth": float(aggregate["expected_earnings_growth_annual"]),
        "current_pe": float(aggregate["component_normalized_pe"]),
        "fair_pe": fair_pe,
        "lambda": annual_lambda,
        "target_pe": float(results[242]["target_pe"]),
        "dividend": float(current_market["trailing_dividend_yield_annual"]),
        "cash": float(current_market["cgb_1y"] / 100.0),
        "earnings_state": earnings_state,
        "market_state": market_state,
        "vol_state": vol_state,
        "close_510300": float(etf_daily.loc[etf_daily["date"].eq(signal_date), "close"].iloc[-1]),
        "close_000300": index_close,
        "model_version": str(config["model"]["model_version"]),
        "official_weight_date": weight_date,
        "normalized_earnings_weight_coverage": float(
            aggregate["normalized_earnings_weight_coverage"]
        ),
        "shadow_target_position": float(shadow_position),
        "input_fingerprint": hashlib.sha256(
            json.dumps(
                {path.relative_to(ROOT).as_posix(): _sha256(path) for path in fingerprint_paths},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    for horizon, values in results.items():
        record[f"expected_return_{horizon}d"] = float(values["expected_total_return"])
        record[f"edge_{horizon}"] = float(values["expected_excess_total_return"])
        record[f"cash_return_{horizon}d"] = float(values["cash_return"])
        record[f"target_pe_{horizon}d"] = float(values["target_pe"])
    return record


def _new_outcome_row(signal: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "signal_date": pd.Timestamp(signal["signal_date"]),
        "model_version": signal["model_version"],
        "signal_record_hash": signal["record_hash"],
        "state_at_signal": json.dumps(
            {
                "earnings_state": signal["earnings_state"],
                "market_state": signal["market_state"],
                "vol_state": signal["vol_state"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    for horizon in HORIZONS:
        row[f"edge_{horizon}"] = float(signal[f"edge_{horizon}"])
        row[f"future_return_{horizon}"] = np.nan
        row[f"future_excess_{horizon}"] = np.nan
        row[f"target_date_{horizon}"] = pd.NaT
        row[f"resolved_at_{horizon}"] = None
        row[f"resolution_hash_{horizon}"] = None
    return row


def resolve_forward_outcomes(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    total_return_daily: pd.DataFrame,
    resolved_at: datetime,
) -> tuple[pd.DataFrame, int]:
    """只在目标交易日真实出现后填入结果；已有结果永不重算覆盖。"""

    signal_data = signals.copy()
    signal_data["signal_date"] = pd.to_datetime(signal_data["signal_date"])
    result = outcomes.copy()
    if not result.empty:
        result["signal_date"] = pd.to_datetime(result["signal_date"])
    known_dates = set(result["signal_date"]) if not result.empty else set()
    new_rows = [
        _new_outcome_row(row)
        for row in signal_data.to_dict("records")
        if pd.Timestamp(row["signal_date"]) not in known_dates
    ]
    if new_rows:
        result = pd.concat([result, pd.DataFrame(new_rows)], ignore_index=True)
    total = total_return_daily[["date", "close"]].copy()
    total["date"] = pd.to_datetime(total["date"])
    total = total.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    positions = {date: index for index, date in enumerate(total["date"])}
    signal_lookup = signal_data.set_index("signal_date")
    resolved_count = 0
    for index, row in result.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        if signal_date not in positions or signal_date not in signal_lookup.index:
            continue
        start = positions[signal_date]
        signal = signal_lookup.loc[signal_date]
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[-1]
        for horizon in HORIZONS:
            column = f"future_return_{horizon}"
            if pd.notna(result.at[index, column]):
                continue
            target_index = start + horizon
            if target_index >= len(total):
                continue
            target_date = pd.Timestamp(total.at[target_index, "date"])
            future_return = float(total.at[target_index, "close"] / total.at[start, "close"] - 1.0)
            future_excess = future_return - float(signal[f"cash_return_{horizon}d"])
            resolution_payload = {
                "signal_date": str(signal_date.date()),
                "horizon": horizon,
                "target_date": str(target_date.date()),
                "future_return": future_return,
                "future_excess": future_excess,
                "resolved_at": resolved_at.isoformat(),
            }
            result.at[index, column] = future_return
            result.at[index, f"future_excess_{horizon}"] = future_excess
            result.at[index, f"target_date_{horizon}"] = target_date
            result.at[index, f"resolved_at_{horizon}"] = resolved_at.isoformat()
            result.at[index, f"resolution_hash_{horizon}"] = hashlib.sha256(
                json.dumps(resolution_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            resolved_count += 1
    return result.sort_values("signal_date").reset_index(drop=True), resolved_count


def _edge_quintile_table(sample: pd.DataFrame, signal: str, target: str) -> list[dict[str, Any]]:
    if len(sample) < 5:
        return []
    labels = ["最低20%", "次低", "中性", "次高", "最高20%"]
    data = sample.copy()
    data["group"] = pd.qcut(data[signal].rank(method="first"), 5, labels=labels)
    grouped = data.groupby("group", observed=False)[target].agg(["count", "mean", "median"])
    return [
        {
            "group": label,
            "observations": int(grouped.loc[label, "count"]),
            "mean_future_excess": float(grouped.loc[label, "mean"]),
            "median_future_excess": float(grouped.loc[label, "median"]),
        }
        for label in labels
    ]


def _state_hypothesis_results(sample: pd.DataFrame, signal: str, target: str) -> dict[str, Any]:
    specifications = {
        "H1": ("earnings_state", "盈利上行", "盈利下行"),
        "H2": ("market_state", "熊市", "牛市"),
        "H3": ("vol_state", "低波动", "高波动"),
    }
    output: dict[str, Any] = {}
    for hypothesis, (column, preferred, comparison) in specifications.items():
        values: dict[str, Any] = {}
        for state in (preferred, comparison):
            group = sample.loc[sample[column].eq(state)]
            values[state] = {
                "observations": int(len(group)),
                "spearman_ic": (
                    float(spearmanr(group[signal], group[target]).statistic)
                    if len(group) >= 4
                    and group[signal].nunique() > 1
                    and group[target].nunique() > 1
                    else None
                ),
            }
        left = values[preferred]["spearman_ic"]
        right = values[comparison]["spearman_ic"]
        output[hypothesis] = {
            "preferred_state": preferred,
            "comparison_state": comparison,
            "states": values,
            "direction_supported": (
                bool(left > right) if left is not None and right is not None else None
            ),
        }
    return output


def _leave_one_block_out_positive_ratio(
    sample: pd.DataFrame,
    signal: str,
    target: str,
    block_length: int = 20,
) -> float | None:
    if len(sample) < block_length * 2:
        return None
    values: list[float] = []
    for start in range(0, len(sample), block_length):
        keep = sample.drop(sample.index[start : start + block_length])
        if len(keep) >= 4:
            values.append(float(spearmanr(keep[signal], keep[target]).statistic))
    return float(np.mean(np.asarray(values) > 0)) if values else None


def evaluate_forward_horizon(
    merged: pd.DataFrame,
    trading_dates: pd.Series,
    horizon: int,
) -> dict[str, Any]:
    signal = f"edge_{horizon}"
    target = f"future_excess_{horizon}"
    sample = merged[
        ["signal_date", signal, target, "earnings_state", "market_state", "vol_state"]
    ].dropna(subset=[signal, target]).sort_values("signal_date").copy()
    sample = sample.rename(columns={"signal_date": "date"})
    if len(sample) < 4:
        return {
            "horizon_days": horizon,
            "realized_observations": int(len(sample)),
            "status": "INSUFFICIENT_DATA",
        }
    pearson = float(sample[signal].corr(sample[target]))
    rank_ic = float(spearmanr(sample[signal], sample[target]).statistic)
    lag = min(horizon - 1, len(sample) - 1)
    bootstrap = moving_block_bootstrap_ic(
        sample[signal], sample[target], min(horizon, len(sample))
    )
    queues = exact_non_overlapping_queues(
        sample, trading_dates, horizon, signal, target
    )
    split = len(sample) // 2
    halves = []
    for label, group in (("前半段", sample.iloc[:split]), ("后半段", sample.iloc[split:])):
        halves.append(
            {
                "period": label,
                "observations": int(len(group)),
                "spearman_ic": (
                    float(spearmanr(group[signal], group[target]).statistic)
                    if len(group) >= 4 else None
                ),
            }
        )
    rolling_ics: list[float] = []
    if len(sample) >= 60:
        for end in range(60, len(sample) + 1):
            window = sample.iloc[end - 60 : end]
            rolling_ics.append(float(spearmanr(window[signal], window[target]).statistic))
    calibration_error = sample[signal] - sample[target]
    return {
        "horizon_days": horizon,
        "realized_observations": int(len(sample)),
        "status": "MONITORING",
        "pearson_ic": pearson,
        "rank_ic": rank_ic,
        "hac": newey_west_slope(sample[signal], sample[target], lag),
        "block_bootstrap": bootstrap,
        "exact_non_overlapping": {
            "queue_observations": [row["observations"] for row in queues],
            "queue_ics": [row["spearman_ic"] for row in queues],
        },
        "top_bottom_mean_return": (
            float(sample.nlargest(max(1, len(sample) // 5), signal)[target].mean())
            - float(sample.nsmallest(max(1, len(sample) // 5), signal)[target].mean())
        ),
        "edge_groups": _edge_quintile_table(sample, signal, target),
        "calibration": {
            "mean_error": float(calibration_error.mean()),
            "mean_absolute_error": float(calibration_error.abs().mean()),
            "positive_edge_precision": (
                float(sample.loc[sample[signal].gt(0), target].gt(0).mean())
                if sample[signal].gt(0).any() else None
            ),
        },
        "chronological_halves": halves,
        "rolling_60_signal_ic": {
            "window_count": len(rolling_ics),
            "positive_ratio": float(np.mean(np.asarray(rolling_ics) > 0)) if rolling_ics else None,
            "latest": rolling_ics[-1] if rolling_ics else None,
        },
        "leave_one_20d_block_out_positive_ratio": _leave_one_block_out_positive_ratio(
            sample, signal, target
        ),
        "state_hypotheses": _state_hypothesis_results(sample, signal, target),
    }


def _monitor_status(
    evaluation_60: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    monitor = config["monitoring"]
    minimum = int(monitor["minimum_realized_60d_signals_for_decision"])
    queue_counts = evaluation_60.get("exact_non_overlapping", {}).get(
        "queue_observations", []
    )
    sufficient = bool(
        evaluation_60.get("realized_observations", 0) >= minimum
        and max(queue_counts, default=0)
        >= int(monitor["minimum_exact_non_overlapping_60d_observations"])
    )
    if not sufficient:
        return "INSUFFICIENT_DATA", {
            "sample_sufficient": False,
            "minimum_realized_60d_signals": minimum,
            "minimum_exact_non_overlapping_60d_observations": int(
                monitor["minimum_exact_non_overlapping_60d_observations"]
            ),
        }
    gate = monitor["gate"]
    halves = evaluation_60["chronological_halves"]
    checks = {
        "rank_ic": evaluation_60["rank_ic"] > float(gate["spearman_ic_minimum"]),
        "hac_p_value": evaluation_60["hac"]["two_sided_p_value_normal_approximation"]
        < float(gate["hac_p_value_maximum"]),
        "bootstrap_lower_bound": evaluation_60["block_bootstrap"]["ic_95pct_interval"][0]
        >= float(gate["bootstrap_lower_bound_minimum"]),
        "top_bottom": evaluation_60["top_bottom_mean_return"]
        > float(gate["top_minus_bottom_mean_return_minimum"]),
        "rolling_positive_ratio": evaluation_60["rolling_60_signal_ic"]["positive_ratio"]
        is not None
        and evaluation_60["rolling_60_signal_ic"]["positive_ratio"]
        >= float(gate["rolling_positive_ratio_minimum"]),
        "chronological_halves": all(
            row["spearman_ic"] is not None
            and row["spearman_ic"] >= float(gate["both_chronological_half_ic_minimum"])
            for row in halves
        ),
        "not_single_extreme_block": evaluation_60[
            "leave_one_20d_block_out_positive_ratio"
        ]
        is not None
        and evaluation_60["leave_one_20d_block_out_positive_ratio"]
        >= float(gate["leave_one_20d_block_out_positive_ratio_minimum"]),
    }
    state_directions = [
        item["direction_supported"]
        for item in evaluation_60["state_hypotheses"].values()
        if item["direction_supported"] is not None
    ]
    detail = {
        "sample_sufficient": True,
        "gate_checks": checks,
        "state_hypothesis_direction_supported_count": int(sum(state_directions)),
        "state_hypothesis_evaluable_count": len(state_directions),
        "state_explanation_status": (
            "FAIL" if len(state_directions) == 3 and not any(state_directions) else "MONITORING"
        ),
    }
    if all(checks.values()):
        return "PASS", detail
    if evaluation_60["rank_ic"] <= 0 or evaluation_60["top_bottom_mean_return"] <= 0:
        return "FAIL", detail
    return "MONITORING", detail


def build_forward_monitor_report(
    signals: pd.DataFrame,
    outcomes: pd.DataFrame,
    total_return_daily: pd.DataFrame,
    generated_at: datetime,
) -> dict[str, Any]:
    config = _load_config()
    signal_columns = [
        "signal_date",
        "earnings_state",
        "market_state",
        "vol_state",
        "shadow_target_position",
    ]
    merged = outcomes.merge(
        signals[signal_columns], on="signal_date", how="left", validate="one_to_one"
    )
    trading_dates = pd.to_datetime(total_return_daily["date"]).drop_duplicates().sort_values()
    evaluations = [
        evaluate_forward_horizon(merged, trading_dates, horizon) for horizon in HORIZONS
    ]
    evaluation_60 = next(item for item in evaluations if item["horizon_days"] == 60)
    status, gate_detail = _monitor_status(evaluation_60, config)
    return {
        "model_version": config["model"]["model_version"],
        "generated_at": generated_at.isoformat(),
        "status": status,
        "signal_count": int(len(signals)),
        "latest_signal_date": (
            str(pd.to_datetime(signals["signal_date"]).max().date()) if not signals.empty else None
        ),
        "evaluations": evaluations,
        "primary_60d_gate": gate_detail,
        "shadow": {
            "latest_target_position": (
                float(signals.sort_values("signal_date").iloc[-1]["shadow_target_position"])
                if not signals.empty else None
            ),
            "real_orders_authorized": False,
        },
        "governance": {
            "review_interval_signals": int(
                config["monitoring"]["review_every_new_trading_signals"]
            ),
            "model_changes_allowed_from_report": False,
            "only_state_hypotheses": ["H1", "H2", "H3"],
        },
    }


def render_forward_monitor_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['model_version']}前瞻验证监控",
        "",
        f"- 状态：`{report['status']}`。",
        f"- 信号数：{report['signal_count']}；最新信号日：{report['latest_signal_date'] or '—'}。",
        "- 本报告只监控，不允许反向修改V3_FORWARD_1。",
        "",
        "|期限|已兑现样本|Pearson IC|Rank IC|HAC p值|Bootstrap 95%区间|Top-Bottom|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["evaluations"]:
        if item["status"] == "INSUFFICIENT_DATA":
            lines.append(
                f"|{item['horizon_days']}日|{item['realized_observations']}|—|—|—|—|—|"
            )
            continue
        interval = item["block_bootstrap"]["ic_95pct_interval"]
        lines.append(
            f"|{item['horizon_days']}日|{item['realized_observations']}|"
            f"{item['pearson_ic']:.3f}|{item['rank_ic']:.3f}|"
            f"{item['hac']['two_sided_p_value_normal_approximation']:.3f}|"
            f"[{interval[0]:.3f}, {interval[1]:.3f}]|"
            f"{item['top_bottom_mean_return']:.2%}|"
        )
    lines.extend(
        [
            "",
            "## 60日预注册状态假设",
            "",
            "|假设|优先状态|对照状态|方向是否支持|",
            "|---|---|---|---|",
        ]
    )
    evaluation_60 = next(item for item in report["evaluations"] if item["horizon_days"] == 60)
    if evaluation_60["status"] != "INSUFFICIENT_DATA":
        for name, item in evaluation_60["state_hypotheses"].items():
            support = item["direction_supported"]
            lines.append(
                f"|{name}|{item['preferred_state']}|{item['comparison_state']}|"
                f"{'—' if support is None else str(support)}|"
            )
    else:
        lines.append("|H1/H2/H3|—|—|样本不足|")
    return "\n".join(lines) + "\n"


def _write_monitor_if_due(
    report: dict[str, Any],
    force: bool = False,
) -> bool:
    config = _load_config()
    interval = int(config["monitoring"]["review_every_new_trading_signals"])
    index_file = REPORT_DIR / "monitor_index.json"
    prior = (
        json.loads(index_file.read_text(encoding="utf-8"))
        if index_file.exists()
        else {"last_report_signal_count": 0}
    )
    due = report["signal_count"] - int(prior["last_report_signal_count"]) >= interval
    if not force and not due:
        return False
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(report["generated_at"]).replace(":", "").replace("+", "_")
    json_file = REPORT_DIR / f"monitor_{report['signal_count']:05d}_{stamp}.json"
    md_file = json_file.with_suffix(".md")
    json_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    md_file.write_text(render_forward_monitor_markdown(report), encoding="utf-8")
    index_file.write_text(
        json.dumps(
            {
                "last_report_signal_count": report["signal_count"],
                "last_report_file": json_file.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _append_shadow_record(signals: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """将最新信号映射为独立的Shadow仓位审计记录，不产生真实订单。"""

    latest = signals.sort_values("signal_date").iloc[-1]
    record = {
        "signal_date": pd.Timestamp(latest["signal_date"]),
        "effective_rule": "信号日后下一交易日开盘生效",
        "instrument": "510300.SH",
        "edge_60": float(latest["edge_60"]),
        "shadow_target_position": float(latest["shadow_target_position"]),
        "model_version": str(latest["model_version"]),
        "signal_record_hash": str(latest["record_hash"]),
        "real_order_submitted": False,
    }
    existing = _read_parquet_or_empty(SHADOW_LOG)
    if not existing.empty:
        existing["signal_date"] = pd.to_datetime(existing["signal_date"])
        same = existing.loc[existing["signal_date"].eq(record["signal_date"])]
        if not same.empty:
            old = same.iloc[-1]
            keys = list(record)
            if all(_canonical(old.get(key)) == _canonical(record[key]) for key in keys):
                return existing.sort_values("signal_date").reset_index(drop=True), "IDEMPOTENT"
            raise ValueError("同一信号日已有不同Shadow仓位记录，禁止覆盖")
        if record["signal_date"] < existing["signal_date"].max():
            raise ValueError("禁止向前瞻Shadow日志回填更早日期")
    combined = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    return combined.sort_values("signal_date").reset_index(drop=True), "APPENDED"


def _write_status(payload: dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(STATUS_FILE)


def _common_close_available(signal_date: pd.Timestamp) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for label, path in (
        ("000300", INDEX_FILE),
        ("H00300", TOTAL_RETURN_FILE),
        ("510300", ETF_FILE),
    ):
        if not path.exists():
            missing.append(f"{label}:文件不存在")
            continue
        frame = pd.read_parquet(path, columns=["date"])
        dates = pd.to_datetime(frame["date"]).dt.normalize()
        if not dates.eq(signal_date).any():
            latest = dates.max()
            latest_text = "无数据" if pd.isna(latest) else str(latest.date())
            missing.append(f"{label}:最新{latest_text}")
    return not missing, missing


def run_forward_cycle(
    signal_date: pd.Timestamp | None = None,
    generated_at: datetime | None = None,
    enforce_same_local_date: bool = True,
    force_report: bool = False,
) -> dict[str, Any]:
    """运行一次V3_FORWARD_1日终周期；缺数据时跳过，绝不回填历史信号。"""

    now = generated_at or datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    local_date = pd.Timestamp(now.astimezone(ZoneInfo("Asia/Shanghai"))).normalize().tz_localize(None)
    target_date = pd.Timestamp(signal_date or local_date).normalize()
    manifest = validate_frozen_manifest()
    status: dict[str, Any] = {
        "model_version": _load_config()["model"]["model_version"],
        "attempted_signal_date": str(target_date.date()),
        "run_at": now.isoformat(),
        "manifest_sha256": _sha256(MANIFEST_FILE),
        "frozen_file_count": len(manifest["frozen_files"]),
        "real_orders_authorized": False,
    }

    available, missing = _common_close_available(target_date)
    if not available:
        status.update(
            {
                "cycle_status": "SKIPPED_NO_CURRENT_CLOSE",
                "missing_inputs": missing,
                "message": "等待真实收盘数据；未生成、未回填信号。",
            }
        )
        _write_status(status)
        return status

    signals = _read_parquet_or_empty(SIGNAL_LOG)
    if not signals.empty:
        signals["signal_date"] = pd.to_datetime(signals["signal_date"])
        verify_signal_hash_chain(signals)
    existing_today = (
        not signals.empty and signals["signal_date"].dt.normalize().eq(target_date).any()
    )
    if existing_today:
        signal_action = "IDEMPOTENT"
    else:
        record = build_forward_signal_snapshot(
            target_date,
            now,
            enforce_same_local_date=enforce_same_local_date,
        )
        signals, signal_action = append_signal_record(signals, record)
        _atomic_parquet(signals, SIGNAL_LOG)

    shadow, shadow_action = _append_shadow_record(signals)
    if shadow_action == "APPENDED" or not SHADOW_LOG.exists():
        _atomic_parquet(shadow, SHADOW_LOG)

    total_return = pd.read_parquet(TOTAL_RETURN_FILE)
    outcomes = _read_parquet_or_empty(OUTCOME_LOG)
    outcomes, resolved_count = resolve_forward_outcomes(
        signals, outcomes, total_return, now
    )
    _atomic_parquet(outcomes, OUTCOME_LOG)
    report = build_forward_monitor_report(signals, outcomes, total_return, now)
    report_written = _write_monitor_if_due(report, force=force_report)
    status.update(
        {
            "cycle_status": "SUCCESS",
            "signal_action": signal_action,
            "shadow_action": shadow_action,
            "resolved_outcome_cells": resolved_count,
            "monitor_status": report["status"],
            "monitor_report_written": report_written,
            "signal_count": int(len(signals)),
            "latest_signal_record_hash": str(
                signals.sort_values("signal_date").iloc[-1]["record_hash"]
            ),
        }
    )
    _write_status(status)
    return status
