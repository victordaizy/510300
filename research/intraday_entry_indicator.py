"""510300十五分钟状态切换进场指标：趋势突破与震荡低吸分轨输出。"""

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

from research.intraday_technical_factors import build_intraday_technical_factors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "intraday_entry_indicator.yaml"


def load_indicator_config(path: Path | None = None) -> dict[str, Any]:
    config_file = path or CONFIG_FILE
    with config_file.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _validate_parameters(parameters: dict[str, Any]) -> None:
    if int(parameters["first_entry_bar_slot"]) < 3:
        raise ValueError("ORB完成前不能生成进场信号")
    if int(parameters["last_entry_bar_slot"]) > 15:
        raise ValueError("15:00收盘K线之后没有同日下一根可交易K线")
    if float(parameters["range_adx_maximum"]) >= float(parameters["trend_adx_minimum"]):
        raise ValueError("震荡ADX上限必须低于趋势ADX下限")
    breadth_minimum = float(parameters["breadth_confirmation_minimum"])
    if not 0.0 <= breadth_minimum <= 1.0:
        raise ValueError("广度确认阈值必须位于0到1之间")
    if parameters["breadth_policy"] not in {
        "DISABLED_510300_ONLY", "OPTIONAL_CONFIRMATION", "REQUIRED"
    }:
        raise ValueError(
            "breadth_policy只支持DISABLED_510300_ONLY、OPTIONAL_CONFIRMATION或REQUIRED"
        )


def _attach_intraday_breadth(
    data: pd.DataFrame,
    breadth: pd.DataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    output = data.copy()
    if breadth is None:
        output["breadth_fraction"] = np.nan
        return output
    schema = config["data"]["breadth_schema"]
    timestamp_column = str(schema["timestamp_column"])
    value_column = str(schema["value_column"])
    missing = {timestamp_column, value_column}.difference(breadth.columns)
    if missing:
        raise ValueError(f"盘中广度数据缺少字段：{sorted(missing)}")
    right = breadth[[timestamp_column, value_column]].copy()
    right[timestamp_column] = pd.to_datetime(right[timestamp_column])
    if right[timestamp_column].duplicated().any():
        raise ValueError("盘中广度signal_asof存在重复")
    values = pd.to_numeric(right[value_column], errors="coerce")
    invalid = values.notna() & ~values.between(0.0, 1.0)
    if invalid.any():
        raise ValueError("breadth_fraction必须位于0到1之间")
    right = right.rename(
        columns={timestamp_column: "signal_asof", value_column: "breadth_fraction"}
    )
    output["signal_asof"] = pd.to_datetime(output["signal_asof"])
    return output.merge(right, on="signal_asof", how="left", validate="one_to_one")


def _finite_rows(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = data[columns].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=data.index)


def _signal_reason(row: pd.Series) -> str:
    code = row["entry_signal"]
    if code == "TREND_BREAKOUT_CONFIRMED":
        return "价格、VWAP斜率、EMA、ADX、前30分钟突破、分时放量及盘中广度共同确认"
    if code == "TREND_BREAKOUT_ETF_ONLY":
        return "510300自身价格、VWAP斜率、EMA、ADX、前30分钟突破与分时放量共同确认"
    if code == "TREND_BREAKOUT_CORE":
        return "趋势突破核心条件成立，但缺少沪深300成分股盘中VWAP广度"
    if code == "TREND_BREAKOUT_BREADTH_REJECTED":
        return "趋势突破核心条件成立，但盘中广度未达到确认阈值"
    if code == "MEAN_REVERSION_BUY":
        return "ADX与VWAP斜率显示震荡，价格低于VWAP至少0.8ATR且RSI进入低位"
    if code == "ENTRY_WINDOW_CLOSED":
        return "技术指标可用，但当前K线不在允许新进场的时段"
    if code == "INSUFFICIENT_DATA":
        return "指标预热不足或OHLCV质量不合格"
    return "趋势突破与震荡低吸条件均未同时成立"


def build_intraday_entry_indicator(
    technical_factors: pd.DataFrame,
    config: dict[str, Any] | None = None,
    intraday_breadth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """根据每根15分钟K线结束时已知的信息生成进场候选，不执行订单。"""

    indicator_config = config or load_indicator_config()
    parameters = indicator_config["parameters"]
    _validate_parameters(parameters)
    required = {
        "signal_asof", "bar_slot", "close", "session_vwap",
        "session_vwap_slope_3bar_bps", "ema_8bar", "ema_21bar", "adx_14bar",
        "opening_range_direct_eligible", "orb_breakout_up_bps", "orb_breakout_down_bps",
        "time_of_day_rvol_median_20d", "atr_14bar", "rsi_7bar",
        "vwap_deviation_atr",
    }
    missing = required.difference(technical_factors.columns)
    if missing:
        raise ValueError(f"十五分钟技术因子缺少字段：{sorted(missing)}")
    data = technical_factors.copy().sort_values("signal_asof").reset_index(drop=True)
    data = _attach_intraday_breadth(data, intraday_breadth, indicator_config)

    first_slot = int(parameters["first_entry_bar_slot"])
    last_slot = int(parameters["last_entry_bar_slot"])
    in_entry_window = data["bar_slot"].between(first_slot, last_slot)
    common_columns = [
        "close", "session_vwap", "session_vwap_slope_3bar_bps", "ema_8bar",
        "ema_21bar", "adx_14bar", "time_of_day_rvol_median_20d", "atr_14bar",
        "rsi_7bar", "vwap_deviation_atr",
    ]
    common_feature_ready = _finite_rows(data, common_columns)
    common_ready = common_feature_ready & in_entry_window
    trend_ready = (
        common_ready
        & _finite_rows(data, ["orb_breakout_up_bps"])
        & data["opening_range_direct_eligible"].fillna(False).astype(bool)
    )

    data["criterion_price_above_vwap"] = data["close"] > data["session_vwap"]
    data["criterion_vwap_slope_up"] = data["session_vwap_slope_3bar_bps"] > float(
        parameters["trend_vwap_slope_3bar_bps_minimum"]
    )
    data["criterion_ema_bullish"] = data["ema_8bar"] > data["ema_21bar"]
    data["criterion_adx_trend"] = data["adx_14bar"] > float(
        parameters["trend_adx_minimum"]
    )
    data["criterion_orb_breakout"] = data["orb_breakout_up_bps"] > float(
        parameters["orb_breakout_bps_minimum"]
    )
    data["criterion_rvol_breakout"] = data["time_of_day_rvol_median_20d"] > float(
        parameters["time_of_day_rvol_minimum"]
    )
    trend_core = trend_ready & data[
        [
            "criterion_price_above_vwap", "criterion_vwap_slope_up",
            "criterion_ema_bullish", "criterion_adx_trend", "criterion_orb_breakout",
            "criterion_rvol_breakout",
        ]
    ].all(axis=1)

    data["criterion_adx_range"] = data["adx_14bar"] < float(
        parameters["range_adx_maximum"]
    )
    data["criterion_vwap_slope_flat"] = data["session_vwap_slope_3bar_bps"].abs() <= float(
        parameters["range_vwap_slope_3bar_abs_bps_maximum"]
    )
    data["criterion_no_abnormal_volume"] = data["time_of_day_rvol_median_20d"] < float(
        parameters["time_of_day_rvol_minimum"]
    )
    data["criterion_vwap_oversold"] = data["vwap_deviation_atr"] < float(
        parameters["mean_reversion_vwap_deviation_atr_maximum"]
    )
    data["criterion_rsi_oversold"] = data["rsi_7bar"] < float(
        parameters["mean_reversion_rsi_7bar_maximum"]
    )
    mean_reversion = common_ready & data[
        [
            "criterion_adx_range", "criterion_vwap_slope_flat",
            "criterion_no_abnormal_volume", "criterion_vwap_oversold",
            "criterion_rsi_oversold",
        ]
    ].all(axis=1)

    data["criterion_price_below_vwap"] = data["close"] < data["session_vwap"]
    data["criterion_vwap_slope_down"] = data["session_vwap_slope_3bar_bps"] < -float(
        parameters["trend_vwap_slope_3bar_bps_minimum"]
    )
    data["criterion_ema_bearish"] = data["ema_8bar"] < data["ema_21bar"]
    data["criterion_orb_breakdown"] = data["orb_breakout_down_bps"] < -float(
        parameters["orb_breakout_bps_minimum"]
    )
    data["criterion_rvol_breakdown"] = data["time_of_day_rvol_median_20d"] > float(
        parameters["time_of_day_rvol_minimum"]
    )
    trend_breakdown = trend_ready & data[
        [
            "criterion_price_below_vwap", "criterion_vwap_slope_down",
            "criterion_ema_bearish", "criterion_adx_trend", "criterion_orb_breakdown",
            "criterion_rvol_breakdown",
        ]
    ].all(axis=1)
    data["criterion_vwap_overbought"] = data["vwap_deviation_atr"] > float(
        parameters["mean_reversion_vwap_deviation_atr_minimum_for_sell"]
    )
    data["criterion_rsi_overbought"] = data["rsi_7bar"] > float(
        parameters["mean_reversion_rsi_7bar_minimum_for_sell"]
    )
    mean_reversion_sell = common_ready & data[
        [
            "criterion_adx_range", "criterion_vwap_slope_flat",
            "criterion_no_abnormal_volume", "criterion_vwap_overbought",
            "criterion_rsi_overbought",
        ]
    ].all(axis=1)

    breadth_available = data["breadth_fraction"].notna()
    breadth_passed = data["breadth_fraction"] > float(
        parameters["breadth_confirmation_minimum"]
    )
    data["trend_core_candidate"] = trend_core
    data["breadth_available"] = breadth_available
    data["breadth_confirmed"] = breadth_available & breadth_passed
    data["entry_signal"] = "NO_ENTRY"
    data.loc[~common_feature_ready, "entry_signal"] = "INSUFFICIENT_DATA"
    data.loc[common_feature_ready & ~in_entry_window, "entry_signal"] = "ENTRY_WINDOW_CLOSED"
    data.loc[mean_reversion, "entry_signal"] = "MEAN_REVERSION_BUY"
    breadth_disabled = parameters["breadth_policy"] == "DISABLED_510300_ONLY"
    data.loc[trend_core & breadth_disabled, "entry_signal"] = "TREND_BREAKOUT_ETF_ONLY"
    data.loc[trend_core & ~breadth_available, "entry_signal"] = "TREND_BREAKOUT_CORE"
    data.loc[
        trend_core & breadth_available & ~breadth_passed,
        "entry_signal",
    ] = "TREND_BREAKOUT_BREADTH_REJECTED"
    data.loc[
        trend_core & breadth_available & breadth_passed,
        "entry_signal",
    ] = "TREND_BREAKOUT_CONFIRMED"
    data.loc[trend_core & breadth_disabled, "entry_signal"] = "TREND_BREAKOUT_ETF_ONLY"

    signal_names = {
        "TREND_BREAKOUT_CONFIRMED": "趋势突破（广度确认）",
        "TREND_BREAKOUT_ETF_ONLY": "趋势突破（ETF自身确认）",
        "TREND_BREAKOUT_CORE": "趋势突破（待广度确认）",
        "TREND_BREAKOUT_BREADTH_REJECTED": "趋势突破（广度否决）",
        "MEAN_REVERSION_BUY": "震荡低吸",
        "ENTRY_WINDOW_CLOSED": "进场窗口关闭",
        "NO_ENTRY": "观望",
        "INSUFFICIENT_DATA": "数据不足",
    }
    data["entry_signal_cn"] = data["entry_signal"].map(signal_names)
    optional_breadth = parameters["breadth_policy"] == "OPTIONAL_CONFIRMATION"
    data["is_entry_candidate"] = data["entry_signal"].isin(
        ["TREND_BREAKOUT_CONFIRMED", "TREND_BREAKOUT_ETF_ONLY", "MEAN_REVERSION_BUY"]
        + (["TREND_BREAKOUT_CORE"] if optional_breadth else [])
    )
    data["is_fully_confirmed"] = data["entry_signal"].isin(
        ["TREND_BREAKOUT_CONFIRMED", "TREND_BREAKOUT_ETF_ONLY", "MEAN_REVERSION_BUY"]
    )
    candidate_number_today = data["is_entry_candidate"].astype(int).groupby(
        pd.to_datetime(data["signal_asof"]).dt.normalize(),
        sort=False,
    ).cumsum()
    data["is_entry_event"] = data["is_entry_candidate"] & candidate_number_today.eq(1)
    data["is_fully_confirmed_event"] = data["is_entry_event"] & data["is_fully_confirmed"]
    data["exit_signal"] = "NO_EXIT"
    data.loc[~common_feature_ready, "exit_signal"] = "INSUFFICIENT_DATA"
    data.loc[common_feature_ready & ~in_entry_window, "exit_signal"] = "EXIT_WINDOW_CLOSED"
    data.loc[mean_reversion_sell, "exit_signal"] = "MEAN_REVERSION_SELL"
    data.loc[trend_breakdown, "exit_signal"] = "TREND_BREAKDOWN_CORE"
    exit_names = {
        "TREND_BREAKDOWN_CORE": "趋势跌破减仓",
        "MEAN_REVERSION_SELL": "震荡高位减仓",
        "EXIT_WINDOW_CLOSED": "减仓窗口关闭",
        "NO_EXIT": "无减仓信号",
        "INSUFFICIENT_DATA": "数据不足",
    }
    data["exit_signal_cn"] = data["exit_signal"].map(exit_names)
    data["is_exit_candidate"] = data["exit_signal"].isin(
        ["TREND_BREAKDOWN_CORE", "MEAN_REVERSION_SELL"]
    )
    exit_number_today = data["is_exit_candidate"].astype(int).groupby(
        pd.to_datetime(data["signal_asof"]).dt.normalize(), sort=False
    ).cumsum()
    data["is_exit_event"] = data["is_exit_candidate"] & exit_number_today.eq(1)
    data["market_regime"] = "TRANSITION"
    data.loc[
        data["criterion_adx_range"] & data["criterion_vwap_slope_flat"],
        "market_regime",
    ] = "RANGE"
    data.loc[
        data["criterion_adx_trend"] & data["criterion_vwap_slope_up"],
        "market_regime",
    ] = "UPTREND"
    data.loc[~common_feature_ready, "market_regime"] = "UNKNOWN"
    data["signal_reason_cn"] = data.apply(_signal_reason, axis=1)
    data["execution_rule"] = "估值目标先确定方向；技术信号后下一根可交易15分钟K线开盘；禁止使用信号K线收盘价成交"
    data["t_plus_one_note"] = "当天新买份额不可当天卖出；日内差价只能使用可卖旧底仓"
    data["indicator_id"] = indicator_config["indicator"]["indicator_id"]
    data["indicator_version"] = indicator_config["indicator"]["version"]
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value


def _write_parquet_atomic(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def main() -> int:
    config = load_indicator_config()
    bars_file = PROJECT_ROOT / config["data"]["bars_file"]
    quality_file = PROJECT_ROOT / config["data"]["quality_file"]
    if not bars_file.exists() or not quality_file.exists():
        raise FileNotFoundError("缺少510300十五分钟数据或质量报告")
    quality = json.loads(quality_file.read_text(encoding="utf-8"))
    quality_accepted = (
        quality.get("coverage", {}).get("status") == "PASS"
        or (
            quality.get("status") == "PASS_WITH_SOURCE_CAVEATS"
            and quality.get("research_use", {}).get("preferred_five_year_15m_ohlcv") is True
        )
    )
    if not quality_accepted:
        raise RuntimeError("十五分钟覆盖审计未通过，禁止生成进场指标")

    breadth_file_value = config["data"].get("intraday_breadth_file")
    breadth = None
    if breadth_file_value:
        breadth_file = PROJECT_ROOT / str(breadth_file_value)
        if not breadth_file.exists():
            raise FileNotFoundError(f"配置的盘中广度文件不存在：{breadth_file}")
        breadth = pd.read_parquet(breadth_file)

    factors = build_intraday_technical_factors(pd.read_parquet(bars_file))
    indicator = build_intraday_entry_indicator(factors, config, breadth)
    output = config["output"]
    indicator_file = PROJECT_ROOT / output["indicator_file"]
    latest_file = PROJECT_ROOT / output["latest_signal_file"]
    report_file = PROJECT_ROOT / output["report_file"]
    _write_parquet_atomic(indicator, indicator_file)

    latest = indicator.iloc[-1]
    candidates = indicator.loc[indicator["is_entry_event"]]
    latest_candidate = candidates.iloc[-1] if not candidates.empty else None
    selected_columns = [
        "signal_asof", "trade_date", "bar_slot", "close", "session_vwap",
        "session_vwap_slope_3bar_bps", "orb_breakout_up_bps",
        "time_of_day_rvol_median_20d", "adx_14bar", "atr_14bar",
        "vwap_deviation_atr", "rsi_7bar", "breadth_fraction", "market_regime",
        "entry_signal", "entry_signal_cn", "is_entry_candidate", "is_fully_confirmed",
        "is_entry_event", "is_fully_confirmed_event", "exit_signal", "exit_signal_cn",
        "is_exit_candidate", "is_exit_event", "signal_reason_cn",
        "execution_rule", "t_plus_one_note",
    ]
    latest_payload = {
        "status": "SIGNAL_GENERATED_NOT_ORDER",
        "indicator_id": config["indicator"]["indicator_id"],
        "indicator_version": config["indicator"]["version"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "latest_bar": latest[selected_columns].to_dict(),
        "latest_historical_entry_candidate": (
            latest_candidate[selected_columns].to_dict() if latest_candidate is not None else None
        ),
        "breadth_status": (
            "DISABLED_510300_ONLY_SCOPE"
            if config["parameters"]["breadth_policy"] == "DISABLED_510300_ONLY"
            else "INTRADAY_BREADTH_ATTACHED"
            if breadth is not None
            else "MISSING_OPTIONAL_CONFIRMATION"
        ),
        "automatic_order_authorized": False,
    }
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(
        json.dumps(_json_ready(latest_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "status": "PASS_EXPLORATORY_INDICATOR",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "indicator_id": config["indicator"]["indicator_id"],
        "indicator_version": config["indicator"]["version"],
        "research_status": config["indicator"]["research_status"],
        "input": {
            "bars_file": bars_file.relative_to(PROJECT_ROOT).as_posix(),
            "bars_sha256": _sha256_file(bars_file),
            "quality_file": quality_file.relative_to(PROJECT_ROOT).as_posix(),
        },
        "output": indicator_file.relative_to(PROJECT_ROOT).as_posix(),
        "row_count": len(indicator),
        "first_signal_asof": indicator["signal_asof"].min(),
        "last_signal_asof": indicator["signal_asof"].max(),
        "signal_counts": indicator["entry_signal"].value_counts().to_dict(),
        "entry_condition_bar_count": int(indicator["is_entry_candidate"].sum()),
        "entry_event_count": int(indicator["is_entry_event"].sum()),
        "fully_confirmed_event_count": int(indicator["is_fully_confirmed_event"].sum()),
        "exit_condition_bar_count": int(indicator["is_exit_candidate"].sum()),
        "exit_event_count": int(indicator["is_exit_event"].sum()),
        "parameters": config["parameters"],
        "breadth_status": latest_payload["breadth_status"],
        "limitations": [
            "当前范围只研究510300 ETF，成分股盘中广度已禁用且不参与信号",
            "依赖精确high/low的ORB、ATR和ADX仅在原生15分钟OHLCV及完成预热后生成正式候选",
            "本产物是进场指标而非收益回测，不包含止盈止损、仓位优化或自动下单授权",
            "510300实行T+1，当天新买份额不可当天卖出",
        ],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(latest_payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
