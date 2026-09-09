"""研究510300十五分钟三柱价格不平衡事件，不赋予SMC或订单流叙事。"""

from __future__ import annotations

import json
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "market" / "510300_15m_from_1m_raw.parquet"
QUALITY_FILE = ROOT / "reports" / "data_quality" / "510300_15m_from_1m_quality.json"
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry.yaml"
JSON_FILE = ROOT / "reports" / "research" / "510300_intraday_imbalance_research.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "510300_intraday_imbalance_research.md"


def _roundtrip_net_return(entry: float, exit_price: float, settings: dict) -> float:
    initial_cash = float(settings["initial_cash"])
    lot_size = int(settings["lot_size"])
    slippage = float(settings["slippage_bps_base"]) / 10_000.0
    commission_rate = float(settings["commission_rate"])
    minimum_commission = float(settings["minimum_commission_cny"])
    stamp_duty_rate = float(settings["stamp_duty_rate"])
    buy_price = entry * (1.0 + slippage)
    sell_price = exit_price * (1.0 - slippage)

    def commission(notional: float) -> float:
        return max(minimum_commission, notional * commission_rate)

    lots = int(initial_cash // (buy_price * lot_size))
    while lots > 0:
        shares = lots * lot_size
        buy_notional = shares * buy_price
        if buy_notional + commission(buy_notional) <= initial_cash + 1e-9:
            break
        lots -= 1
    if lots <= 0:
        return np.nan
    shares = lots * lot_size
    buy_notional = shares * buy_price
    cash = initial_cash - buy_notional - commission(buy_notional)
    sell_notional = shares * sell_price
    ending = cash + sell_notional - commission(sell_notional) - sell_notional * stamp_duty_rate
    return float(ending / initial_cash - 1.0)


def build_event_frame(bars: pd.DataFrame, settings: dict, horizons: tuple[int, ...]) -> pd.DataFrame:
    data = bars.copy()
    data["bar_end"] = pd.to_datetime(data["bar_end"])
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    data = data.sort_values("bar_end").reset_index(drop=True)
    counts = data.groupby("trade_date").size()
    if data.empty or not (counts == 16).all():
        raise ValueError("三柱价格不平衡研究要求每个交易日完整16根15分钟K线")
    data["bar_slot"] = data.groupby("trade_date").cumcount() + 1
    same_day_three = (
        data["trade_date"].eq(data["trade_date"].shift(1))
        & data["trade_date"].eq(data["trade_date"].shift(2))
    )
    # 第9、10根的三柱结构跨越午间休市，不作为连续三柱事件。
    consecutive_session = ~data["bar_slot"].isin([1, 2, 9, 10])
    eligible = same_day_three & consecutive_session
    bullish_lower = data["high"].shift(2)
    bullish_upper = data["low"]
    bearish_lower = data["high"]
    bearish_upper = data["low"].shift(2)
    data["bullish_3bar_imbalance"] = (bullish_upper > bullish_lower) & eligible
    data["bearish_3bar_imbalance"] = (bearish_lower < bearish_upper) & eligible
    data["bullish_gap_lower"] = bullish_lower.where(data["bullish_3bar_imbalance"])
    data["bullish_gap_upper"] = bullish_upper.where(data["bullish_3bar_imbalance"])
    data["bearish_gap_lower"] = bearish_lower.where(data["bearish_3bar_imbalance"])
    data["bearish_gap_upper"] = bearish_upper.where(data["bearish_3bar_imbalance"])

    for horizon in horizons:
        gross: list[float] = []
        net: list[float] = []
        mae: list[float] = []
        mfe: list[float] = []
        valid: list[bool] = []
        for index_value in range(len(data)):
            entry_index = index_value + 1
            exit_index = index_value + horizon
            same_day = (
                exit_index < len(data)
                and data.loc[index_value, "trade_date"] == data.loc[exit_index, "trade_date"]
            )
            if not same_day:
                gross.append(np.nan)
                net.append(np.nan)
                mae.append(np.nan)
                mfe.append(np.nan)
                valid.append(False)
                continue
            entry = float(data.loc[entry_index, "open"])
            exit_price = float(data.loc[exit_index, "close"])
            path = data.loc[entry_index:exit_index]
            gross.append(float(exit_price / entry - 1.0))
            net.append(_roundtrip_net_return(entry, exit_price, settings))
            mae.append(float(path["low"].min() / entry - 1.0))
            mfe.append(float(path["high"].max() / entry - 1.0))
            valid.append(True)
        data[f"future_{horizon}bar_gross_return"] = gross
        data[f"future_{horizon}bar_hypothetical_net_return"] = net
        data[f"future_{horizon}bar_mae"] = mae
        data[f"future_{horizon}bar_mfe"] = mfe
        data[f"future_{horizon}bar_valid"] = valid

    for side in ("bullish", "bearish"):
        filled: list[bool | None] = []
        bars_to_fill: list[float] = []
        event_column = f"{side}_3bar_imbalance"
        for index_value, row in data.iterrows():
            if not bool(row[event_column]):
                filled.append(None)
                bars_to_fill.append(np.nan)
                continue
            day = row["trade_date"]
            future = data.loc[(data.index > index_value) & data["trade_date"].eq(day)]
            if side == "bullish":
                condition = future["low"] <= float(row["bullish_gap_lower"])
            else:
                condition = future["high"] >= float(row["bearish_gap_upper"])
            matches = future.index[condition]
            if len(matches):
                filled.append(True)
                bars_to_fill.append(float(matches[0] - index_value))
            else:
                filled.append(False)
                bars_to_fill.append(np.nan)
        data[f"{side}_same_day_full_fill"] = filled
        data[f"{side}_bars_to_full_fill"] = bars_to_fill
    return data


def _day_block_bootstrap(values: pd.DataFrame, iterations: int = 1000) -> dict:
    day_means = values.groupby("trade_date")["effect_vs_slot_baseline"].mean().dropna()
    if len(day_means) < 20:
        return {"days": int(len(day_means)), "iterations": 0, "median": None, "ci_2_5pct": None, "ci_97_5pct": None}
    generator = np.random.default_rng(zlib.crc32(f"{len(day_means)}|{day_means.mean()}".encode("utf-8")))
    array = day_means.to_numpy(dtype=float)
    boot = [float(generator.choice(array, size=len(array), replace=True).mean()) for _ in range(iterations)]
    return {
        "days": int(len(day_means)), "iterations": iterations,
        "median": float(np.median(boot)),
        "ci_2_5pct": float(np.quantile(boot, 0.025)),
        "ci_97_5pct": float(np.quantile(boot, 0.975)),
    }


def evaluate_event(data: pd.DataFrame, hypothesis: dict) -> dict:
    event = hypothesis["event"]
    horizon = int(hypothesis["horizon_bars"])
    target = f"future_{horizon}bar_hypothetical_net_return"
    valid = data[f"future_{horizon}bar_valid"]
    baseline_source = data.loc[valid, ["bar_slot", target]].dropna()
    baseline_by_slot = baseline_source.groupby("bar_slot")[target].mean()
    events = data.loc[data[event] & valid].copy()
    events["slot_baseline"] = events["bar_slot"].map(baseline_by_slot)
    events["effect_vs_slot_baseline"] = events[target] - events["slot_baseline"]
    expected_positive = hypothesis["expected_sign"] == "positive"
    yearly = events.groupby(events["trade_date"].dt.year)["effect_vs_slot_baseline"].mean().dropna()
    direction_ratio = None if yearly.empty else float(np.mean(yearly > 0) if expected_positive else np.mean(yearly < 0))
    bootstrap = _day_block_bootstrap(events)
    mean_return = None if events.empty else float(events[target].mean())
    mean_effect = None if events.empty else float(events["effect_vs_slot_baseline"].mean())
    fill_prefix = "bullish" if event.startswith("bullish") else "bearish"
    fill = events[f"{fill_prefix}_same_day_full_fill"].dropna()
    checks = {
        "event_return_expected_direction": mean_return is not None and ((mean_return > 0) if expected_positive else (mean_return < 0)),
        "slot_adjusted_effect_expected_direction": mean_effect is not None and ((mean_effect > 0) if expected_positive else (mean_effect < 0)),
        "year_direction_ratio_ge_60pct": direction_ratio is not None and direction_ratio >= 0.60,
        "bootstrap_median_expected_direction": (
            bootstrap["median"] is not None
            and ((bootstrap["median"] > 0) if expected_positive else (bootstrap["median"] < 0))
        ),
        "at_least_100_events": len(events) >= 100,
    }
    score = int(sum(checks.values()))
    rating = "STRONG" if score == 5 else "WEAK" if score >= 3 else "INCONCLUSIVE" if score >= 2 else "REJECTED"
    return {
        **hypothesis,
        "event_count": int(len(events)),
        "event_day_count": int(events["trade_date"].nunique()),
        "mean_hypothetical_net_return": mean_return,
        "median_hypothetical_net_return": None if events.empty else float(events[target].median()),
        "positive_ratio": None if events.empty else float((events[target] > 0).mean()),
        "mean_effect_vs_same_slot_baseline": mean_effect,
        "mean_mae": None if events.empty else float(events[f"future_{horizon}bar_mae"].mean()),
        "mean_mfe": None if events.empty else float(events[f"future_{horizon}bar_mfe"].mean()),
        "same_day_full_fill_rate": None if fill.empty else float(fill.astype(bool).mean()),
        "median_bars_to_full_fill": None if events[f"{fill_prefix}_bars_to_full_fill"].dropna().empty else float(events[f"{fill_prefix}_bars_to_full_fill"].median()),
        "year_effects": {str(year): float(value) for year, value in yearly.items()},
        "year_expected_direction_ratio": direction_ratio,
        "day_block_bootstrap_effect": bootstrap,
        "evidence": {"score": score, "rating": rating, "checks": checks},
        "implementation_status": "EVENT_EVIDENCE_ONLY_NOT_DIRECTLY_TRADABLE_FOR_NEW_T1_SHARES",
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# 510300十五分钟三柱价格不平衡研究", "",
        "## 口径", "",
        "- 使用五年1分钟数据自行聚合的15分钟主序列。",
        "- 事件只定义为三柱OHLC价格不平衡，不解释为主力、机构扫单或真实订单流。",
        "- 第9、10根柱会跨越午间休市，因此不用于连续三柱事件。",
        "- 收益以下一根柱开盘为起点，并报告同一bar-slot基准差。",
        "- 成本后收益是假设性往返估计；510300新买份额受T+1限制，不能直接据此做当日买卖。", "",
        "## 结果", "",
        "|ID|事件|前瞻柱数|事件数|假设成本后收益|相对同槽位效应|同日完整回补率|得分|评级|",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        fill = item["same_day_full_fill_rate"]
        lines.append(
            f"|{item['hypothesis_id']}|{item['event']}|{item['horizon_bars']}|{item['event_count']}|"
            f"{item['mean_hypothetical_net_return'] or 0:.4%}|{item['mean_effect_vs_same_slot_baseline'] or 0:.4%}|"
            f"{'' if fill is None else f'{fill:.2%}'}|{item['evidence']['score']}/5|{item['evidence']['rating']}|"
        )
    lines += ["", "FVG结果只决定该事件是否值得进入后续条件研究，不自动生成ETF日内策略。", ""]
    return "\n".join(lines)


def main() -> int:
    quality = json.loads(QUALITY_FILE.read_text(encoding="utf-8"))
    if quality.get("status") != "PASS_WITH_SOURCE_CAVEATS" or not quality.get("research_use", {}).get("preferred_five_year_15m_ohlcv"):
        raise RuntimeError("五年15分钟主序列没有通过预期质量门")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    hypotheses = registry["intraday_hypotheses"]
    horizons = tuple(sorted({int(item["horizon_bars"]) for item in hypotheses}))
    data = build_event_frame(pd.read_parquet(INPUT_FILE), settings["backtest"], horizons)
    results = [evaluate_event(data, item) for item in hypotheses]
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "source": INPUT_FILE.relative_to(ROOT).as_posix(),
        "source_quality": quality["status"],
        "hypothesis_count": len(results),
        "results": results,
        "pv_lc_status": "BLOCKED_NO_LEVEL2_NOT_APPROXIMATED",
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"日内事件报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
