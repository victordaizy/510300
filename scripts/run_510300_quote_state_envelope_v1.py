"""原事件的秒内报价范围检验；全部范围只作条件性描述，不生成成交收益。"""

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
from research.l2_order_lineage_v1 import inventory_events, trade_lineage
from research.quote_state_envelope_v1 import event_comparison, quote_envelopes

CONFIG = ROOT / "config/510300_quote_state_envelope_v1.json"
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/quote_envelope_20261001"
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo_selected"
EVENTS = ROOT / "reports/research/510300_pressure_recovery_v1/local_exploration_20261001/02_固定观察事件表.csv"
GRID = ROOT / "reports/research/510300_pressure_recovery_v1/measurement_correction_20261001/01_五分钟消息测量修正版.parquet"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def raw_clock(timestamp: pd.Timestamp) -> int:
    return int(timestamp.strftime("%H%M%S") + f"{timestamp.microsecond // 1000:03d}")


def chart(events: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(11, 8))
    positions = np.arange(len(events))
    names = []
    for i, row in enumerate(events.itertuples()):
        names.append(row.event_id.removeprefix("P0_").replace("_", " "))
        if row.comparison_status == "CONDITIONAL_DESCRIPTIVE_BOUNDS_ONLY":
            color = "#b64e43" if row.observation_price_change_upper_bps < 0 else "#1d7895" if row.observation_price_change_lower_bps > 0 else "#949a9f"
            axis.plot([row.observation_price_change_lower_bps, row.observation_price_change_upper_bps], [i, i], color=color, linewidth=3)
            axis.scatter(row.original_observation_price_change_bps, i, color="#252a31", s=24, zorder=3)
        else:
            axis.text(0, i, "  来源不在条件范围内，保留未知", fontsize=9)
    axis.axvline(0, color="#555555", linewidth=.8, linestyle="--")
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlabel("固定五分钟观察期间的价格变化（基点）")
    axis.set_title("510300 原17个事件：秒内时点不确定下的价格变化范围\n线段：条件范围；黑点：原来源快照描述。不是成交后收益。", loc="left", fontsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=.15)
    figure.text(.08, .025, "依赖整秒快照、10毫秒逐笔、同一市场时钟及快照字段同步等假设；实际接收时间仍未知。", fontsize=9, color="#555555")
    figure.tight_layout(rect=[0, .055, 1, 1])
    figure.savefig(OUT / "原事件价格变化条件范围.png", dpi=150)
    plt.close(figure)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("本批范围结果非空，拒绝覆盖或修改范围规则。")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folders = {}
    for day in config["source_dates"]:
        matches = [p for p in BASE.glob(day + "_*") if all((p / (name + ".parquet")).exists() for name in ["行情", "逐笔委托", "逐笔成交"])]
        if len(matches) != 1:
            raise ValueError("本地三流子集不唯一：" + day)
        folders[day] = matches[0]
    inputs = [CONFIG, Path(__file__), ROOT / "research/quote_state_envelope_v1.py", ROOT / "research/l2_order_lineage_v1.py", EVENTS, GRID,
              ROOT / "reports/research/510300_pressure_recovery_v1/order_lineage_20261001/quote_clock_summary.json",
              ROOT / "reports/research/510300_pressure_recovery_v1/source_followup_20261001/ldds_level2_2_0_10.pages.json"]
    inputs += [folder / (name + ".parquet") for folder in folders.values() for name in ["行情", "逐笔委托", "逐笔成交"]]
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = {"registered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "protocol": config,
               "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in inputs]}
    save_json("registration_receipt.json", receipt)
    snapshots, daily = [], []
    for day, folder in folders.items():
        quotes, orders, trades = [pd.read_parquet(folder / (name + ".parquet")) for name in ["行情", "逐笔委托", "逐笔成交"]]
        lineage, _ = trade_lineage(orders, trades)
        inventory, checks = inventory_events(orders, lineage)
        if checks["missing_order_reference_events"] or checks["negative_remaining_timestamp_batches"] or checks["event_before_add_time"]:
            raise ValueError("订单库存前提与先前结果不一致。")
        frame = quote_envelopes(quotes, trades, inventory, config["features"]["depth_band_bps"])
        snapshots.append(frame)
        daily.append({"date": day, "snapshots": len(frame), "clock_intervals_valid": int(frame.clock_interval_valid.sum()),
                      "snapshots_inside_conditional_outer_box": int(frame.source_snapshot_inside_outer_box.sum()),
                      "scalar_envelopes_known": int(frame.feature_status.eq("CONDITIONAL_OUTER_ENVELOPE").sum()),
                      "median_interval_width_ms": float(frame.loc[frame.clock_interval_valid, "conditional_interval_width_ms"].median())})
        print(json.dumps(daily[-1], ensure_ascii=False), flush=True)
    snapshots = pd.concat(snapshots, ignore_index=True)
    snapshots.to_parquet(OUT / "01_全部快照条件范围.parquet", index=False)
    pd.DataFrame(daily).to_csv(OUT / "02_逐日范围覆盖.csv", index=False, encoding="utf-8-sig")
    fixed = pd.read_csv(EVENTS)
    grid = pd.read_parquet(GRID).set_index("decision_at")
    lookup = snapshots.set_index(["date", "time"])
    event_rows = []
    for event in fixed.itertuples():
        start_quote_time = grid.loc[pd.Timestamp(event.shock_at), "source_quote_at"]
        end_quote_time = grid.loc[pd.Timestamp(event.observation_end_at), "source_quote_at"]
        date = int(event.trade_date.replace("-", ""))
        start = lookup.loc[(date, raw_clock(start_quote_time))]
        end = lookup.loc[(date, raw_clock(end_quote_time))]
        row = {"event_id": event.event_id, "trade_date": event.trade_date, "shock_at": event.shock_at,
               "observation_end_at": event.observation_end_at, "original_observation_price_change_bps": event.observation_price_change_bps,
               "original_moving_bid_depth_ratio": event.bid_depth_ratio, "original_spread_label": event.spread_label,
               "start_snapshot_inside_box": start.source_snapshot_inside_outer_box,
               "endpoint_snapshot_inside_box": end.source_snapshot_inside_outer_box,
               **event_comparison(start, end), "point_in_time_admission": "NO_VIEW", "actual_order": False}
        event_rows.append(row)
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT / "03_原事件判断的条件范围.csv", index=False, encoding="utf-8-sig")
    measured = events.loc[events.comparison_status.eq("CONDITIONAL_DESCRIPTIVE_BOUNDS_ONLY")]
    price_contains = measured.original_observation_price_change_bps.between(measured.observation_price_change_lower_bps - 1e-7, measured.observation_price_change_upper_bps + 1e-7)
    depth_contains = (measured.original_moving_bid_depth_ratio >= measured.moving_bid_depth_ratio_lower - 1e-7) & (measured.moving_bid_depth_ratio_upper.isna() | measured.original_moving_bid_depth_ratio.le(measured.moving_bid_depth_ratio_upper + 1e-7))
    if not price_contains.all() or not depth_contains.all() or set(events.event_id) != set(fixed.event_id):
        raise AssertionError("范围不包含原观察或事件集合改变。")
    summary = {"status": "CONDITIONAL_OUTER_ENVELOPES_COMPLETED_NOT_RECEIPT_OR_FILL_VALIDATION",
               "snapshots": len(snapshots), "clock_interval_valid": int(snapshots.clock_interval_valid.sum()),
               "source_snapshots_inside_outer_box": int(snapshots.source_snapshot_inside_outer_box.sum()),
               "source_snapshots_outside_outer_box": int((~snapshots.source_snapshot_inside_outer_box).sum()),
               "events_retained": len(events), "events_with_source_contained_bounds": len(measured),
               "price_direction_counts": measured.price_direction.value_counts().to_dict() if len(measured) else {},
               "moving_depth_direction_counts": measured.moving_depth_direction.value_counts().to_dict() if len(measured) else {},
               "spread_direction_counts": measured.spread_direction.value_counts().to_dict() if len(measured) else {},
               "joint_state_reconstruction_proven": False, "source_field_mapping_independently_validated": False,
               "received_at_upper_bound": None, "actual_orders": 0, "new_network_requests": 0,
               "qualified_m1_m2_signals": 0, "strategy_net_expectancy": "NOT_COMPUTED", "full_account_sharpe": "NOT_COMPUTED",
               "goal_achieved": False, "position_impact": 0}
    save_json("summary.json", summary)
    if not all(digest(ROOT / item["path"]) == item["sha256"] for item in receipt["files"]):
        raise AssertionError("登记输入发生变化。")
    save_json("verification.json", {"original_event_set_unchanged": True, "source_contained_events_inside_price_and_depth_bounds": True,
                                    "source_files_and_old_results_unchanged": True, "outside_box_cases_not_promoted": True,
                                    "clock_offsets_or_returns_optimized": False, "not_a_strategy_validation": True})
    save_json("goal_progress.json", {"previous_goal_turn_classification": "PROGRESS", "current_goal_turn_classification": "PROGRESS",
                                     "evidence": ["把所有候选源更新时间与批内未知顺序纳入条件数量外界", "逐个原事件记录判断能否在全部范围内成立，未选时间偏移"],
                                     "goal_achieved": False, "goal_should_remain_active": True,
                                     "remaining": ["条件范围不能建立源字段映射或接收时钟", "仅三个盘口日期且同步篮子估值不足", "实际/独立验证成交与全账户目标仍未建立"]})
    report = ["# 秒内时点不确定时，原观察还能确定什么", "",
              "**本轮计算条件性行情范围，保留原17个事件；没有移动时钟寻找最优匹配，也没有重算策略收益。范围不是实际接收时间，更不是成交价格保证。**", "",
              "## 范围依赖的前提", "",
              "假设导出快照时刻对应截断整秒、逐笔对应截断10毫秒，二者使用同一市场时钟，累计成交量与盘口属于同一状态，且数据转换没有额外影响。这些前提与已保存交易所字段定义及源数据精度相符，但尚未独立验证。", "",
              "先由累计成交量确定已包含的最后一笔成交及下一笔未包含成交，再与整个名义秒相交。范围内包含全部源消息时刻；同一时间戳内部不能确定先后时，对每个价位保留全部增加或减少可能造成的数量外界。不选某个具体偏移，不用后续收益决定范围。", "",
              "不同价位的上下界可能不能同时实现，所以这是包含可能状态的外包络；即使包住所见快照，也不代表重建出了一个唯一、真实、可下单的盘口。", "",
              "## 快照核对", "", "| 日期 | 快照数 | 时间范围非空 | 来源十档与总量落在条件范围内 |", "|---|---:|---:|---:|"]
    for item in daily:
        report.append(f"| {item['date']} | {item['snapshots']:,} | {item['clock_intervals_valid']:,} | {item['snapshots_inside_conditional_outer_box']:,} |")
    report += ["", f"共{len(snapshots):,}个快照，{summary['source_snapshots_outside_outer_box']:,}个不在条件范围内。它们不能被强行纳入，原事件只要任一端出现该问题，就不作方向稳定性结论。", "",
               "## 原17个事件", "",
               f"保留全部17个事件，其中{len(measured)}个的两端来源状态均落在条件范围内。观察期间价格变化的方向计数：{summary['price_direction_counts']}；移动买盘深度比的方向计数：{summary['moving_depth_direction_counts']}；价差变化计数：{summary['spread_direction_counts']}。", "",
               "正/负方向须在整个外界上成立；区间接触零或跨越零时保留不确定。深度比下界大于1才能称为该条件范围内始终增加；分母下界为零时，上界保留未知。移动价格带深度增加依旧不能证明原绝对价位补单。", "",
               "这些是固定观察期内的描述，没有把确认期间的反弹计为成交后收益。即使某一方向在整个条件范围内成立，原名义截止时点是否已收到数据依然未知，不能恢复为M2信号。", "",
               "## 实际可用范围仍未知", "",
               "行情状态的条件范围与数据到达时间是两个问题。现有材料不能为实际收到快照的时刻提供上界，因此本轮无法产生可准入的实际入场、成交率、净期望或全账户夏普。同步参考估值、60个同刻合格交易日及独立成交验证仍缺失。原目标尚未完成。", "",
               "- [逐日范围覆盖](02_逐日范围覆盖.csv)", "- [17个原事件的完整范围](03_原事件判断的条件范围.csv)",
               "- [观察期间价格变化图](原事件价格变化条件范围.png)", ""]
    (OUT / "00_条件范围与原观察稳定性.md").write_text("\n".join(report), encoding="utf-8")
    chart(events)
    index = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    pd.DataFrame(index).to_csv(OUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    size = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    if size > config["reporting"]["new_result_storage_limit_bytes"]:
        raise ValueError("新增结果超过5MiB预算。")
    print(json.dumps(clean({"完成": summary, "新增结果字节": size}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
