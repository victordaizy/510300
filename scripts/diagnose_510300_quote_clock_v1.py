"""用快照累计成交量定位逐笔前缀，记录跨流时钟下界；不调整时间或重选事件。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.pressure_recovery_local_exploration_v1 import with_clock

OUT = ROOT / "reports/research/510300_pressure_recovery_v1/order_lineage_20261001"
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo_selected"
GRID = ROOT / "reports/research/510300_pressure_recovery_v1/measurement_correction_20261001/01_五分钟消息测量修正版.parquet"
EVENTS = ROOT / "reports/research/510300_pressure_recovery_v1/local_exploration_20261001/02_固定观察事件表.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_volume_prefix(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """只用正成交量定位，不能由零量记录或插值制造已包含消息的下界。"""
    positive = trades.loc[trades.volume.gt(0) & trades.time.lt(150500000)].copy()
    positive = with_clock(positive)
    clocks = positive._at.dt.as_unit("ns").astype("int64").to_numpy()
    cumulative = positive.volume.cumsum().to_numpy(dtype=float)
    if len(cumulative) == 0 or np.any(np.diff(cumulative) <= 0):
        raise ValueError("正成交量前缀非严格递增。")
    viewed = with_clock(quotes)
    positions = np.searchsorted(cumulative, viewed.cum_volume.to_numpy(dtype=float), side="left")
    within = positions < len(cumulative)
    exact = within & (cumulative[np.minimum(positions, len(cumulative) - 1)] == viewed.cum_volume.to_numpy(dtype=float))
    nominal = viewed._at.dt.as_unit("ns").astype("int64").to_numpy()
    last_included = pd.to_datetime(np.where(exact, clocks[np.minimum(positions, len(clocks) - 1)], np.iinfo("int64").min), utc=True).tz_convert("Asia/Shanghai")
    result = pd.DataFrame({"date": viewed.date.astype(str), "source_quote_time": viewed._at,
                           "source_raw_time": viewed.time, "source_cumulative_volume": viewed.cum_volume,
                           "positive_trade_prefix_exact": exact, "latest_included_trade_time_lower_bound": last_included,
                           "prefix_last_positive_trade_row": np.where(exact, positions, -1)})
    differences = (clocks[np.minimum(positions, len(clocks) - 1)] - nominal) / 1e9
    result["included_trade_minus_nominal_quote_seconds"] = np.where(exact, differences, np.nan)
    result["contains_trade_later_than_nominal_quote"] = exact & (differences > 0)
    result["source_receive_time_established"] = False
    return result


def main() -> None:
    if (OUT / "quote_clock_summary.json").exists():
        raise SystemExit("时钟定位结果已存在，拒绝覆盖。")
    folders = {day: next(BASE.glob(day + "_*")) for day in ["20260709", "20260812", "20260904"]}
    inputs = [Path(__file__), GRID, EVENTS,
              ROOT / "reports/research/510300_pressure_recovery_v1/source_followup_20261001/ldds_level2_2_0_10.pages.json"]
    inputs += [folder / (name + ".parquet") for folder in folders.values() for name in ["行情", "逐笔成交"]]
    registration = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "method": "以逐笔正成交量累积和精确定位快照累计成交量，不尝试平移时间，不优化匹配率或策略收益。",
                    "source_document": "本地LDDS 2.0.10物理页10的DataTimeStamp为最新订单时间（秒）；与逐笔百分之一秒字段精度不同。",
                    "source_field_mapping_is_independently_proven": False,
                    "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in inputs]}
    (OUT / "quote_clock_registration.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frames, day_summaries = [], []
    for day, folder in folders.items():
        quotes = pd.read_parquet(folder / "行情.parquet")
        trades = pd.read_parquet(folder / "逐笔成交.parquet")
        quotes = quotes.loc[((quotes.time >= 93000000) & (quotes.time < 113000000)) | ((quotes.time >= 130000000) & (quotes.time < 145700000))]
        frame = locate_volume_prefix(quotes, trades)
        frames.append(frame)
        day_summaries.append({"date": day, "snapshots": len(frame), "all_quote_millisecond_digits_zero": bool(frame.source_raw_time.mod(1000).eq(0).all()),
                              "exact_cumulative_volume_prefixes": int(frame.positive_trade_prefix_exact.sum()),
                              "quotes_containing_later_source_trade": int(frame.contains_trade_later_than_nominal_quote.sum()),
                              "maximum_proven_nominal_time_difference_seconds": float(frame.included_trade_minus_nominal_quote_seconds.max())})
    located = pd.concat(frames, ignore_index=True)
    located.to_csv(OUT / "04_快照累计量时钟定位.csv", index=False, encoding="utf-8-sig")
    grid = pd.read_parquet(GRID)
    joined = grid[["decision_at", "trade_date", "source_quote_at", "measurement_status"]].merge(
        located, how="left", left_on="source_quote_at", right_on="source_quote_time", validate="many_to_one")
    joined["included_trade_after_measurement_deadline"] = joined.latest_included_trade_time_lower_bound.gt(joined.decision_at)
    joined["absence_of_violation_is_receipt_proof"] = False
    joined.to_csv(OUT / "05_原测量时点的时钟下界.csv", index=False, encoding="utf-8-sig")
    events = pd.read_csv(EVENTS)
    event_rows = []
    lookup = joined.set_index("decision_at")
    for event in events.itertuples():
        record = {"event_id": event.event_id, "trade_date": event.trade_date}
        for name, at in [("shock", event.shock_at), ("observation", event.observation_end_at)]:
            point = lookup.loc[pd.Timestamp(at)]
            record[name + "_at"] = at
            record[name + "_nominal_quote_at"] = str(point.source_quote_at)
            record[name + "_latest_included_trade_at"] = str(point.latest_included_trade_time_lower_bound)
            record[name + "_contains_later_trade"] = bool(point.included_trade_after_measurement_deadline)
        event_rows.append(record)
    event_clock = pd.DataFrame(event_rows)
    event_clock.to_csv(OUT / "06_原事件的时钟限制.csv", index=False, encoding="utf-8-sig")
    summary = {"status": "COARSE_QUOTE_TIME_CANNOT_BE_USED_AS_EXACT_AVAILABILITY_TIME", "daily": day_summaries,
               "snapshots": len(located), "exact_prefixes": int(located.positive_trade_prefix_exact.sum()),
               "quotes_containing_later_source_trade": int(located.contains_trade_later_than_nominal_quote.sum()),
               "original_grid_rows": len(joined), "original_grid_contains_later_source_trade": int(joined.included_trade_after_measurement_deadline.sum()),
               "original_events_with_later_trade_at_shock": int(event_clock.shock_contains_later_trade.sum()),
               "original_events_with_later_trade_at_observation_end": int(event_clock.observation_contains_later_trade.sum()),
               "original_events_with_either_clock_violation": int((event_clock.shock_contains_later_trade | event_clock.observation_contains_later_trade).sum()),
               "effect": "原17个事件及原价格路径仍作为名义源时钟描述；不能据此声称在标注截止时点已知盘口。未重新选择事件或计算策略收益。",
               "source_receive_time_proven": False, "subsecond_timestamp_shift_fitted": False,
               "original_m1_m2_admission": "NO_VIEW", "strategy_net_expectancy": "NOT_COMPUTED", "goal_achieved": False}
    (OUT / "quote_clock_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(sha256(ROOT / item["path"]) == item["sha256"] for item in registration["files"]):
        raise AssertionError("输入发生变化。")
    report = f"""# 快照时间的精度限制：累计成交量提供的直接证据

**源快照末尾的“.000”不能视为精确毫秒的可用时间。14,220个连续时段快照都能以累计成交量定位到逐笔成交前缀；其中{summary['quotes_containing_later_source_trade']:,}个已经包含名义快照时刻之后的源成交。**

这是同源数据内的时钟口径不一致，不能直接认定真实交易发生了时间倒流，也不能把某个拟合偏移当作已知接收时间。

已保存的上交所LDDS Level-2接口说明书2.0.10，物理页10把竞价快照DataTimeStamp定义为‘最新订单时间（秒）’，物理页21把逐笔TickTime定义到百分之一秒。这与本次快照全部毫秒尾数为零、逐笔仍保留秒内变化的情况一致。来源导出字段与原始协议的映射尚未独立验证。

三个日期中，累计量证明已包含成交晚于名义快照时刻的最大差值分别为0.74秒、0.56秒、0.44秒。它们是从已包含成交得到的下界证据，不是完整接收延迟，也不是源端最大时钟误差保证。

## 对此前本地探索的具体影响

在原711个分钟测量时点中，有{summary['original_grid_contains_later_source_trade']}个使用的快照已经包含测量截止时点之后的源成交。原17个价格事件中，冲击起点有{summary['original_events_with_later_trade_at_shock']}个、固定观察终点有{summary['original_events_with_later_trade_at_observation_end']}个存在这项证据；两者任一受影响的事件共{summary['original_events_with_either_clock_violation']}个。

因此，原事件与价格路径保留为按名义源时间形成的描述，不能声称对应盘口在精确截止时点已经可知。此前它们本就未准入原M1/M2成交收益，本轮没有将其升格，也没有重新选事件以改善表现。五分钟成交/申报窗口的微秒/纳秒修正仍正确，但它本身无法修复快照字段的精度和接收时钟缺口。

未发现已包含未来成交的时点也不等于通过了可得性检验：没有成交的间隙仍可能发生订单变化，且本地收到数据的时间仍未知。给所有快照机械加1秒或寻找最优偏移，都不能代替来源定义和接收记录。

## 与订单重建的关系

按名义时刻只累积此前消息时，十档价量全一致比例只有2.24%。订单数量生命周期可以自洽，同时快照与逐笔的时钟口径仍不一致；不能据前者宣称已经可以可信重放自身排队成交。当前证据为时钟限制提供了具体解释，未证明所有剩余不一致都仅由时间精度造成。

本轮不平移时间、不优化快照匹配比例、不改变交易规则或计算新策略收益。

- [所有快照的累计量定位](04_快照累计量时钟定位.csv)
- [原711个测量时点的时钟下界](05_原测量时点的时钟下界.csv)
- [原17个事件的逐项限制](06_原事件的时钟限制.csv)
"""
    (OUT / "07_快照时钟限制与原结果适用范围.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
