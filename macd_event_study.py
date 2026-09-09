from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SECID = "1.510300"
RANDOM_SEED = 20260815


@dataclass(frozen=True)
class TimeframeConfig:
    name: str
    klt: int
    start: str
    pivot_order: int
    min_pivot_gap: int
    max_pivot_gap: int
    horizons: dict[int, str]


CONFIGS = (
    TimeframeConfig(
        name="5分钟",
        klt=5,
        start="2021-08-15",
        pivot_order=3,
        min_pivot_gap=6,
        max_pivot_gap=48,
        horizons={1: "5分钟", 3: "15分钟", 12: "60分钟", 48: "1交易日"},
    ),
    TimeframeConfig(
        name="日K",
        klt=101,
        start="2021-08-15",
        pivot_order=3,
        min_pivot_gap=5,
        max_pivot_gap=60,
        horizons={1: "1日", 3: "3日", 5: "5日", 20: "20日"},
    ),
    TimeframeConfig(
        name="周K",
        klt=102,
        start="2021-08-15",
        pivot_order=2,
        min_pivot_gap=3,
        max_pivot_gap=26,
        horizons={1: "1周", 4: "4周", 12: "12周"},
    ),
)


def fetch_klines(klt: int, retries: int = 3) -> pd.DataFrame:
    params = {
        "secid": SECID,
        "klt": klt,
        "fqt": 1,
        "beg": "19900101",
        "end": "20500101",
        "lmt": 100000,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": "Mozilla/5.0 MACD event study"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not payload.get("data") or not payload["data"].get("klines"):
                raise RuntimeError(f"行情接口未返回 K 线：klt={klt}")
            rows = [line.split(",") for line in payload["data"]["klines"]]
            columns = [
                "datetime",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude_pct",
                "change_pct",
                "change",
                "turnover_pct",
            ]
            frame = pd.DataFrame(rows, columns=columns)
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            for column in columns[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        except Exception as exc:  # 网络瞬时失败时重试
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"下载行情失败：klt={klt}") from last_error


def add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ema_fast = result["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = result["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    result["dif"] = ema_fast - ema_slow
    result["dea"] = result["dif"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["histogram"] = 2 * (result["dif"] - result["dea"])
    return result


def is_pivot(values: np.ndarray, index: int, order: int, mode: str) -> bool:
    center = values[index]
    left = values[index - order : index]
    right = values[index + 1 : index + order + 1]
    if not np.isfinite(center) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return False
    if mode == "low":
        return center <= left.min() and center <= right.min() and (
            center < left.min() or center < right.min()
        )
    return center >= left.max() and center >= right.max() and (
        center > left.max() or center > right.max()
    )


def divergence_events(frame: pd.DataFrame, config: TimeframeConfig) -> list[dict[str, object]]:
    lows = frame["low"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    dif = frame["dif"].to_numpy(dtype=float)
    order = config.pivot_order
    low_pivots = [
        i for i in range(order, len(frame) - order) if is_pivot(lows, i, order, "low")
    ]
    high_pivots = [
        i for i in range(order, len(frame) - order) if is_pivot(highs, i, order, "high")
    ]
    events: list[dict[str, object]] = []

    for pivots, signal_name, direction in (
        (low_pivots, "底背离", 1),
        (high_pivots, "顶背离", -1),
    ):
        for previous, current in zip(pivots, pivots[1:]):
            gap = current - previous
            if gap < config.min_pivot_gap or gap > config.max_pivot_gap:
                continue
            if not np.isfinite(dif[previous]) or not np.isfinite(dif[current]):
                continue
            divergent = (
                lows[current] < lows[previous] and dif[current] > dif[previous]
                if direction == 1
                else highs[current] > highs[previous] and dif[current] < dif[previous]
            )
            if not divergent:
                continue
            confirmation = current + order
            if confirmation >= len(frame):
                continue
            events.append(
                {
                    "event_index": confirmation,
                    "pivot_index": current,
                    "previous_pivot_index": previous,
                    "signal": signal_name,
                    "direction": direction,
                }
            )
    return events


def crossover_events(frame: pd.DataFrame) -> list[dict[str, object]]:
    spread = frame["dif"] - frame["dea"]
    golden = (spread > 0) & (spread.shift(1) <= 0)
    death = (spread < 0) & (spread.shift(1) >= 0)
    events: list[dict[str, object]] = []
    for index in frame.index[golden.fillna(False)]:
        events.append(
            {"event_index": int(index), "pivot_index": None, "previous_pivot_index": None,
             "signal": "金叉", "direction": 1}
        )
    for index in frame.index[death.fillna(False)]:
        events.append(
            {"event_index": int(index), "pivot_index": None, "previous_pivot_index": None,
             "signal": "死叉", "direction": -1}
        )
    return events


def bootstrap_ci(values: np.ndarray, samples: int = 20000) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(RANDOM_SEED + len(values))
    chunk_size = max(1, min(samples, 2_000_000 // len(values)))
    means: list[np.ndarray] = []
    remaining = samples
    while remaining > 0:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def normal_p_value(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    standard_error = values.std(ddof=1) / math.sqrt(len(values))
    if standard_error == 0:
        return 0.0 if values.mean() != 0 else 1.0
    z_score = abs(values.mean() / standard_error)
    return 2 * (1 - NormalDist().cdf(z_score))


def analyze_events(
    frame: pd.DataFrame,
    events: Iterable[dict[str, object]],
    config: TimeframeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, object]] = []
    for event in events:
        index = int(event["event_index"])
        if frame.loc[index, "datetime"] < pd.Timestamp(config.start):
            continue
        row = {
            "周期": config.name,
            "信号": event["signal"],
            "方向": "看多" if event["direction"] == 1 else "看空",
            "信号时间": frame.loc[index, "datetime"],
            "信号收盘价": frame.loc[index, "close"],
        }
        if event["pivot_index"] is not None:
            pivot_index = int(event["pivot_index"])
            previous_index = int(event["previous_pivot_index"])
            row["摆动点时间"] = frame.loc[pivot_index, "datetime"]
            row["前摆动点时间"] = frame.loc[previous_index, "datetime"]
        else:
            row["摆动点时间"] = pd.NaT
            row["前摆动点时间"] = pd.NaT
        for bars, label in config.horizons.items():
            future_index = index + bars
            row[f"{label}收益"] = (
                frame.loc[future_index, "close"] / frame.loc[index, "close"] - 1
                if future_index < len(frame)
                else math.nan
            )
        event_rows.append(row)

    event_frame = pd.DataFrame(event_rows)
    summary_rows: list[dict[str, object]] = []
    if event_frame.empty:
        return event_frame, pd.DataFrame()

    signal_order = ["底背离", "顶背离", "金叉", "死叉"]
    sample_frame = frame[frame["datetime"] >= pd.Timestamp(config.start)].copy()
    for signal in signal_order:
        signal_events = event_frame[event_frame["信号"] == signal]
        if signal_events.empty:
            continue
        direction = 1 if signal in ("底背离", "金叉") else -1
        for bars, label in config.horizons.items():
            raw = signal_events[f"{label}收益"].dropna().to_numpy(dtype=float)
            if len(raw) == 0:
                continue
            directional = direction * raw
            baseline_raw = (
                sample_frame["close"].shift(-bars) / sample_frame["close"] - 1
            ).dropna().to_numpy(dtype=float)
            baseline_directional = direction * baseline_raw
            ci_low, ci_high = bootstrap_ci(directional)
            summary_rows.append(
                {
                    "周期": config.name,
                    "信号": signal,
                    "观察窗": label,
                    "样本数": len(raw),
                    "平均原始收益": raw.mean(),
                    "中位原始收益": np.median(raw),
                    "方向胜率": (directional > 0).mean(),
                    "平均方向收益": directional.mean(),
                    "无条件平均方向收益": baseline_directional.mean(),
                    "相对基准方向优势": directional.mean() - baseline_directional.mean(),
                    "均值95%CI下限": ci_low,
                    "均值95%CI上限": ci_high,
                    "均值为零近似p值": normal_p_value(directional),
                }
            )
    return event_frame, pd.DataFrame(summary_rows)


def percentage(value: float) -> str:
    return "—" if pd.isna(value) else f"{value * 100:.3f}%"


def build_report(
    metadata: list[dict[str, object]],
    summary: pd.DataFrame,
) -> str:
    lines = [
        "# 510300 MACD 事件研究",
        "",
        "生成日期：2026-08-15",
        "",
        "## 口径",
        "",
        "- 标的：华泰柏瑞沪深300ETF（510300），前复权行情。",
        "- MACD：DIF=EMA(12)-EMA(26)，DEA=EMA(DIF,9)。",
        "- 金叉/死叉：DIF 当根上穿/下穿 DEA，按该根收盘价作为信号价。",
        "- 底背离：后一个价格摆动低点更低、DIF 低点更高；顶背离反之。",
        "- 背离仅在右侧摆动确认后入场，避免使用尚未发生的数据。",
        "- 方向胜率：看多信号未来收益>0；看空信号未来收益<0。",
        "- 相对基准方向优势：信号平均方向收益减去同期所有 K 线的无条件平均方向收益。",
        "- 事件研究允许观察窗重叠；收益未扣交易成本，不能直接视为可交易策略净收益。",
        "",
        "## 数据覆盖",
        "",
        "| 周期 | 原始数据起点 | 统计起点 | 数据终点 | K线数 | 背离确认参数 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in metadata:
        lines.append(
            f"| {item['周期']} | {item['原始数据起点']} | {item['统计起点']} | "
            f"{item['数据终点']} | {item['K线数']} | {item['背离确认参数']} |"
        )

    for timeframe in [config.name for config in CONFIGS]:
        lines.extend(["", f"## {timeframe}", ""])
        subset = summary[summary["周期"] == timeframe]
        lines.extend(
            [
                "| 信号 | 观察窗 | N | 平均原始收益 | 中位收益 | 方向胜率 | 相对基准优势 | 方向收益95%CI | p值 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['信号']} | {row['观察窗']} | {int(row['样本数'])} | "
                f"{percentage(row['平均原始收益'])} | {percentage(row['中位原始收益'])} | "
                f"{percentage(row['方向胜率'])} | {percentage(row['相对基准方向优势'])} | "
                f"[{percentage(row['均值95%CI下限'])}, {percentage(row['均值95%CI上限'])}] | "
                f"{row['均值为零近似p值']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "分钟数据的公开接口仅返回最近约六周，因此日内结论只可视为近期样本描述。"
            "日K和周K按过去五年统计。背离结果依赖摆动点参数，后续应做参数敏感性和样本外检验。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []

    for config in CONFIGS:
        raw = fetch_klines(config.klt)
        frame = add_macd(raw)
        events = divergence_events(frame, config) + crossover_events(frame)
        event_frame, summary_frame = analyze_events(frame, events, config)
        all_events.append(event_frame)
        all_summaries.append(summary_frame)
        raw.to_csv(OUTPUT_DIR / f"510300_{config.klt}_raw.csv", index=False, encoding="utf-8-sig")
        metadata.append(
            {
                "周期": config.name,
                "原始数据起点": raw["datetime"].min().strftime("%Y-%m-%d %H:%M"),
                "统计起点": max(pd.Timestamp(config.start), raw["datetime"].min()).strftime("%Y-%m-%d %H:%M"),
                "数据终点": raw["datetime"].max().strftime("%Y-%m-%d %H:%M"),
                "K线数": len(raw),
                "背离确认参数": (
                    f"左右各{config.pivot_order}根，间隔"
                    f"{config.min_pivot_gap}-{config.max_pivot_gap}根"
                ),
            }
        )

    events_output = pd.concat(all_events, ignore_index=True)
    summary_output = pd.concat(all_summaries, ignore_index=True)
    events_output.to_csv(OUTPUT_DIR / "macd_events.csv", index=False, encoding="utf-8-sig")
    summary_output.to_csv(OUTPUT_DIR / "macd_summary.csv", index=False, encoding="utf-8-sig")
    report = build_report(metadata, summary_output)
    (OUTPUT_DIR / "macd_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
