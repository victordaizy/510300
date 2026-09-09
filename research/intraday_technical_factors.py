"""构建510300十五分钟技术因子，不包含目标标签、策略规则或参数搜索。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_from_1m_raw.parquet"
QUALITY_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_15m_from_1m_quality.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "510300_15m_technical_factors.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "research" / "510300_15m_technical_factors.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _past_slot_statistics(
    data: pd.DataFrame,
    column: str,
    window_days: int,
) -> tuple[pd.Series, pd.Series]:
    """同一日内时点仅用以前交易日计算历史均值和标准差。"""

    past = data.groupby("bar_slot", sort=False)[column].shift(1)
    keys = data["bar_slot"]
    mean = past.groupby(keys, sort=False).transform(
        lambda values: values.rolling(window_days, min_periods=window_days).mean()
    )
    std = past.groupby(keys, sort=False).transform(
        lambda values: values.rolling(window_days, min_periods=window_days).std(ddof=1)
    )
    return mean, std


def _past_slot_median(
    data: pd.DataFrame,
    column: str,
    window_days: int,
) -> pd.Series:
    """同一日内时点仅用以前交易日计算滚动中位数。"""

    past = data.groupby("bar_slot", sort=False)[column].shift(1)
    return past.groupby(data["bar_slot"], sort=False).transform(
        lambda values: values.rolling(window_days, min_periods=window_days).median()
    )


def _wilder_average(
    values: pd.Series,
    eligible: pd.Series,
    period: int,
) -> pd.Series:
    """对合格的连续OHLCV区间计算Wilder平滑，前置不合格数据不参与初始化。"""

    return values.where(eligible).ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
        ignore_na=False,
    ).mean()


def _masked_rolling_mean(
    values: pd.Series,
    eligible: pd.Series,
    window: int,
) -> pd.Series:
    count = eligible.astype(int).rolling(window, min_periods=window).sum()
    result = values.rolling(window, min_periods=window).mean()
    return result.where(count == window)


def build_intraday_technical_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """构建在每根K线结束时可得的技术因子。"""

    normalized = bars.copy()
    if "construction" not in normalized.columns:
        if "source_grade" not in normalized.columns:
            raise ValueError("十五分钟数据缺少construction或source_grade，无法判定质量层级")
        normalized["construction"] = np.where(
            normalized["source_grade"].eq("cross_validated_unofficial_proxy"),
            "tushare_1m_aggregated_15m",
            "unknown_15m_construction",
        )
    normalized["bar_end"] = pd.to_datetime(normalized["bar_end"])
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.normalize()
    normalized = normalized.sort_values("bar_end").reset_index(drop=True)
    inferred_slot = normalized.groupby("trade_date").cumcount() + 1
    if "bar_position" not in normalized.columns:
        normalized["bar_position"] = np.select(
            [inferred_slot.eq(1), inferred_slot.eq(9), inferred_slot.eq(16)],
            ["first", "afternoon_first", "last"],
            default="regular",
        )
    if "session_phase" not in normalized.columns:
        normalized["session_phase"] = np.select(
            [inferred_slot.eq(1), inferred_slot.between(2, 8), inferred_slot.eq(9), inferred_slot.between(10, 15), inferred_slot.eq(16)],
            ["opening_auction_mixed", "continuous_am", "afternoon_open", "continuous_pm", "closing_auction_mixed"],
            default="unknown",
        )
    required = {
        "bar_end", "trade_date", "open", "high", "low", "close", "volume",
        "amount", "construction", "session_phase", "bar_position",
    }
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"十五分钟数据缺少字段：{sorted(missing)}")
    data = normalized
    if data.empty or data["bar_end"].duplicated().any():
        raise ValueError("十五分钟数据为空或bar_end重复")
    counts = data.groupby("trade_date").size()
    if not (counts == 16).all():
        raise ValueError("技术因子要求每个交易日完整16根K线")

    data["bar_slot"] = data.groupby("trade_date").cumcount() + 1
    data["signal_asof"] = data["bar_end"]
    trusted_constructions = {"exchange_15m_kline", "tushare_1m_aggregated_15m"}
    data["direct_ohlcv_eligible"] = data["construction"].isin(trusted_constructions)
    data["close_volume_factor_eligible"] = True
    data["factor_quality_tier"] = np.select(
        [
            data["construction"].eq("exchange_15m_kline"),
            data["construction"].eq("tushare_1m_aggregated_15m"),
        ],
        ["A_DIRECT_OHLCV", "A_CROSS_VALIDATED_1M_AGGREGATED"],
        default="B_RECONSTRUCTED_CLOSE_VOLUME",
    )

    close = data["close"].astype(float)
    log_close = np.log(close)
    data["log_return_1bar"] = log_close.diff()
    data["return_1bar"] = close.pct_change(fill_method=None)
    for window in [4, 16, 32, 80, 160]:
        data[f"momentum_{window}bar"] = close / close.shift(window) - 1.0
        data[f"realized_vol_{window}bar"] = data["log_return_1bar"].rolling(
            window,
            min_periods=window,
        ).std(ddof=1)
    downside = data["log_return_1bar"].clip(upper=0.0)
    for window in [16, 80]:
        data[f"downside_vol_{window}bar"] = downside.rolling(
            window,
            min_periods=window,
        ).std(ddof=1)
    for span in [16, 64, 160]:
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        data[f"close_over_ema_{span}"] = close / ema - 1.0
    for span in [8, 21]:
        data[f"ema_{span}bar"] = close.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()

    price_change = close.diff()
    gain = price_change.clip(lower=0.0)
    loss = -price_change.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / 7.0, adjust=False, min_periods=7).mean()
    average_loss = loss.ewm(alpha=1.0 / 7.0, adjust=False, min_periods=7).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    data["rsi_7bar"] = 100.0 - 100.0 / (1.0 + relative_strength)
    data.loc[(average_loss == 0.0) & (average_gain > 0.0), "rsi_7bar"] = 100.0
    data.loc[(average_loss == 0.0) & (average_gain == 0.0), "rsi_7bar"] = 50.0

    data["log_volume"] = np.log1p(data["volume"].astype(float))
    for window_days in [20, 60]:
        mean, std = _past_slot_statistics(data, "log_volume", window_days)
        data[f"log_volume_slot_mean_{window_days}d"] = mean
        data[f"relative_volume_z_{window_days}d"] = (data["log_volume"] - mean) / std.replace(
            0.0, np.nan
        )
    data["volume_slot_median_20d"] = _past_slot_median(data, "volume", 20)
    data["time_of_day_rvol_median_20d"] = (
        data["volume"] / data["volume_slot_median_20d"].replace(0.0, np.nan)
    )
    data["bar_vwap"] = data["amount"] / data["volume"]
    data["close_over_bar_vwap"] = close / data["bar_vwap"] - 1.0
    data["bar_vwap_is_estimated"] = ~data["direct_ohlcv_eligible"]

    grouped = data.groupby("trade_date", sort=False)
    cumulative_volume = grouped["volume"].cumsum()
    cumulative_amount = grouped["amount"].cumsum()
    data["session_vwap"] = cumulative_amount / cumulative_volume.replace(0.0, np.nan)
    prior_session_vwap = grouped["session_vwap"].shift(3)
    data["session_vwap_slope_3bar_bps"] = (
        data["session_vwap"] / prior_session_vwap - 1.0
    ) * 10_000.0
    data["vwap_deviation_fraction"] = close / data["session_vwap"] - 1.0

    opening_high_observation = data["high"].where(data["bar_slot"].le(2))
    opening_low_observation = data["low"].where(data["bar_slot"].le(2))
    data["opening_range_high"] = opening_high_observation.groupby(
        data["trade_date"], sort=False
    ).cummax()
    data["opening_range_low"] = opening_low_observation.groupby(
        data["trade_date"], sort=False
    ).cummin()
    data["opening_range_high"] = grouped["opening_range_high"].ffill()
    data["opening_range_low"] = grouped["opening_range_low"].ffill()
    opening_direct_count = data["direct_ohlcv_eligible"].where(
        data["bar_slot"].le(2),
        False,
    ).astype(int).groupby(data["trade_date"], sort=False).cumsum()
    data["opening_range_complete"] = data["bar_slot"].ge(2)
    data["opening_range_direct_eligible"] = (
        data["opening_range_complete"] & opening_direct_count.eq(2)
    )
    after_opening_range = data["bar_slot"].gt(2)
    data["orb_breakout_up_bps"] = (
        (close / data["opening_range_high"] - 1.0) * 10_000.0
    ).where(after_opening_range & data["opening_range_direct_eligible"])
    data["orb_breakout_down_bps"] = (
        (close / data["opening_range_low"] - 1.0) * 10_000.0
    ).where(after_opening_range & data["opening_range_direct_eligible"])

    phase = 2.0 * np.pi * (data["bar_slot"] - 1) / 16.0
    data["time_of_day_sin"] = np.sin(phase)
    data["time_of_day_cos"] = np.cos(phase)
    data["is_first_bar"] = data["bar_slot"].eq(1)
    data["is_afternoon_first_bar"] = data["bar_slot"].eq(9)
    data["is_last_bar"] = data["bar_slot"].eq(16)

    same_day_three = (
        data["trade_date"].eq(data["trade_date"].shift(1))
        & data["trade_date"].eq(data["trade_date"].shift(2))
    )
    direct_three = (
        data["direct_ohlcv_eligible"]
        & data["direct_ohlcv_eligible"].shift(1, fill_value=False)
        & data["direct_ohlcv_eligible"].shift(2, fill_value=False)
    )
    data["fvg_factor_eligible"] = same_day_three & direct_three
    bullish_gap = data["low"] - data["high"].shift(2)
    bearish_gap = data["low"].shift(2) - data["high"]
    data["bullish_fvg"] = (bullish_gap > 0.0).where(data["fvg_factor_eligible"])
    data["bearish_fvg"] = (bearish_gap > 0.0).where(data["fvg_factor_eligible"])
    data["bullish_fvg_size"] = bullish_gap.clip(lower=0.0).where(data["fvg_factor_eligible"])
    data["bearish_fvg_size"] = bearish_gap.clip(lower=0.0).where(data["fvg_factor_eligible"])
    data["bullish_fvg_size_bps"] = (
        data["bullish_fvg_size"] / data["high"].shift(2) * 10_000.0
    ).where(data["fvg_factor_eligible"])
    data["bearish_fvg_size_bps"] = (
        data["bearish_fvg_size"] / data["low"].shift(2) * 10_000.0
    ).where(data["fvg_factor_eligible"])

    direct = data["direct_ohlcv_eligible"]
    price_range = data["high"] - data["low"]
    data["candle_body"] = (data["close"] - data["open"]).where(direct)
    data["candle_range"] = price_range.where(direct)
    data["upper_wick"] = (
        data["high"] - data[["open", "close"]].max(axis=1)
    ).where(direct)
    data["lower_wick"] = (
        data[["open", "close"]].min(axis=1) - data["low"]
    ).where(direct)
    data["close_location_in_range"] = (
        (data["close"] - data["low"]) / price_range.replace(0.0, np.nan)
    ).where(direct)

    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_eligible = direct & direct.shift(1, fill_value=False)
    data["true_range"] = true_range.where(true_range_eligible)
    data["atr_14bar"] = _masked_rolling_mean(data["true_range"], true_range_eligible, 14)
    data["atr_14bar_bps"] = data["atr_14bar"] / close * 10_000.0

    high_change = data["high"].diff()
    low_change = -data["low"].diff()
    plus_dm = high_change.where((high_change > low_change) & (high_change > 0.0), 0.0)
    minus_dm = low_change.where((low_change > high_change) & (low_change > 0.0), 0.0)
    smoothed_true_range = _wilder_average(true_range, true_range_eligible, 14)
    smoothed_plus_dm = _wilder_average(plus_dm, true_range_eligible, 14)
    smoothed_minus_dm = _wilder_average(minus_dm, true_range_eligible, 14)
    plus_di = 100.0 * smoothed_plus_dm / smoothed_true_range.replace(0.0, np.nan)
    minus_di = 100.0 * smoothed_minus_dm / smoothed_true_range.replace(0.0, np.nan)
    directional_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / directional_sum.replace(0.0, np.nan)
    data["plus_di_14bar"] = plus_di
    data["minus_di_14bar"] = minus_di
    data["adx_14bar"] = _wilder_average(dx, dx.notna(), 14)
    data["vwap_deviation_atr"] = (
        (close - data["session_vwap"]) / data["atr_14bar"].replace(0.0, np.nan)
    )
    return data


def main() -> int:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到完整十五分钟数据：{INPUT_FILE}")
    if not QUALITY_FILE.exists():
        raise FileNotFoundError(f"找不到十五分钟质量报告：{QUALITY_FILE}")
    quality = json.loads(QUALITY_FILE.read_text(encoding="utf-8"))
    quality_status = quality.get("status")
    preferred_five_year = quality.get("research_use", {}).get(
        "preferred_five_year_15m_ohlcv",
        False,
    )
    if quality_status != "PASS_WITH_SOURCE_CAVEATS" or not preferred_five_year:
        raise RuntimeError("五年十五分钟数据未通过研究用途审计，禁止构建因子")
    data = build_intraday_technical_factors(pd.read_parquet(INPUT_FILE))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".parquet.tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OUTPUT_FILE)
    factor_columns = [
        column for column in data.columns
        if column.startswith(
            (
                "momentum_", "realized_vol_", "downside_vol_", "close_over_ema_",
                "relative_volume_", "bullish_fvg", "bearish_fvg", "session_vwap",
                "orb_", "time_of_day_rvol_", "ema_", "rsi_", "adx_", "atr_",
                "plus_di_", "minus_di_", "vwap_deviation_",
            )
        )
    ]
    report = {
        "status": "PASS_WITH_QUALITY_TIERS",
        "scope": "510300十五分钟技术因子计算，不含目标标签和策略选择",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "input_file": INPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "input_sha256": _sha256_file(INPUT_FILE),
        "output_file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "output_sha256": _sha256_file(OUTPUT_FILE),
        "row_count": int(len(data)),
        "first_signal_asof": data["signal_asof"].min().isoformat(),
        "last_signal_asof": data["signal_asof"].max().isoformat(),
        "quality_tier_counts": {
            str(key): int(value) for key, value in data["factor_quality_tier"].value_counts().items()
        },
        "fvg_eligible_rows": int(data["fvg_factor_eligible"].sum()),
        "bullish_fvg_count": int(data["bullish_fvg"].fillna(False).sum()),
        "bearish_fvg_count": int(data["bearish_fvg"].fillna(False).sum()),
        "factor_columns": factor_columns,
        "timing": "所有因子仅使用当前bar结束时及以前的信息；真实交易只能从信号后的实际成交时点起算",
        "restrictions": [
            "五年Tushare代理1分钟聚合OHLCVA已逐日与独立日线完全对账；来源仍属于第三方代理而非官方直连",
            "任何策略仍不得假设能够在15分钟柱内high或low成交",
            "本文件不含未来收益标签、入场阈值、退出规则或回测结论",
            "PV、LC和Level-2订单流尚未接入，不能由OHLCV伪造",
        ],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
