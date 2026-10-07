"""利用已存三日逐笔数据核对订单生命周期、十档快照和固定事件压力区间。"""

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
from research.l2_order_lineage_v1 import compare_snapshot_book, inventory_events, pressure_bounds, trade_lineage

CONFIG = ROOT / "config/510300_l2_order_lineage_v1.json"
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/order_lineage_20261001"
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo_selected"
EVENTS = ROOT / "reports/research/510300_pressure_recovery_v1/local_exploration_20261001/02_固定观察事件表.csv"


def sha256(path: Path) -> str:
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


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("本次订单关联结果已存在，拒绝覆盖。")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources = {}
    for day in config["source_dates"]:
        folders = [p for p in BASE.glob(day + "_*") if all((p / (name + ".parquet")).exists() for name in ("行情", "逐笔委托", "逐笔成交"))]
        if len(folders) != 1:
            raise ValueError("日期没有唯一的完整本地三流子集：" + day)
        sources[day] = folders[0]
    files = [CONFIG, Path(__file__), ROOT / "research/l2_order_lineage_v1.py",
             ROOT / "research/pressure_recovery_measurement_correction_v1_1.py", EVENTS,
             ROOT / config["source_semantics"]["local_exchange_document"],
             ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo/README.md",
             ROOT / "reports/research/510300_pressure_recovery_v1/measurement_correction_20261001/summary.json",
             ROOT / "reports/research/510300_pressure_recovery_v1/volatility_conditioned_proxy_20261001/acceptance_outcome.json"]
    files += [folder / (stream + ".parquet") for folder in sources.values() for stream in ("行情", "逐笔委托", "逐笔成交")]
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = {"registered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "protocol": config,
               "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]}
    save_json("registration_receipt.json", receipt)
    fixed_events = pd.read_csv(EVENTS)
    daily, pressure, lifecycle_rows = [], [], []
    lineage_columns = ["date", "time", "trade_id", "passive_order_id", "aggressive_order_id", "phase", "passive_link_found",
                       "passive_side_consistent", "passive_price_consistent", "passive_strictly_prior",
                       "same_source_direction_supported", "supported_direction_sign"]
    for compact, folder in sources.items():
        day = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        quotes, orders, trades = [pd.read_parquet(folder / (name + ".parquet")) for name in ("行情", "逐笔委托", "逐笔成交")]
        lineage, linkage = trade_lineage(orders, trades)
        inventory, lifecycle = inventory_events(orders, lineage)
        comparison = compare_snapshot_book(quotes, inventory)
        lineage[lineage_columns].reset_index(names="source_trade_row_index").to_parquet(OUT / f"{compact}_逐笔关联结论.parquet", index=False)
        comparison.to_parquet(OUT / f"{compact}_逐档快照核对.parquet", index=False)
        snapshot = {"snapshots_compared": len(comparison), "both_best_prices_match": int(comparison.both_best_prices_match.sum()),
                    "all_ten_prices_match": int(comparison.all_ten_prices_match.sum()),
                    "all_ten_prices_and_quantities_match": int(comparison.all_ten_prices_and_quantities_match.sum()),
                    "both_total_quantities_match": int(comparison.both_total_quantities_match.sum()),
                    "snapshots_with_negative_inventory": int(comparison.negative_price_level_count.gt(0).sum()),
                    "clock_shift_seconds": 0, "full_feed_or_own_fill_established": False}
        daily.append({"trade_date": day, **linkage, **lifecycle, **snapshot})
        lifecycle_rows.append({"trade_date": day, **lifecycle})
        current = fixed_events.loc[fixed_events.trade_date.eq(day)].copy()
        ends = pd.DatetimeIndex(pd.concat([current.shock_at, current.observation_end_at], ignore_index=True))
        bounded = pressure_bounds(lineage, ends).set_index("endpoint")
        for event in current.itertuples():
            record = {"event_id": event.event_id, "trade_date": day, "shock_at": event.shock_at,
                      "observation_end_at": event.observation_end_at, "observation_price_change_bps": event.observation_price_change_bps,
                      "source_bs_independently_validated": False, "formal_m2_status": "NO_VIEW", "actual_order": False}
            for name, at in [("shock", event.shock_at), ("observation", event.observation_end_at)]:
                for key, value in bounded.loc[pd.Timestamp(at)].to_dict().items():
                    record[name + "_" + key] = value
            pressure.append(record)
        print(json.dumps({"日期": day, "被动单关联": f"{linkage['passive_links_found']}/{linkage['continuous_source_rows']}",
                          "生命周期负余额批次": lifecycle["negative_remaining_timestamp_batches"],
                          "十档价量全一致快照": f"{snapshot['all_ten_prices_and_quantities_match']}/{snapshot['snapshots_compared']}"}, ensure_ascii=False), flush=True)
    daily = pd.DataFrame(daily)
    pressure = pd.DataFrame(pressure)
    daily.to_csv(OUT / "01_逐日关联与库存结论.csv", index=False, encoding="utf-8-sig")
    pressure.to_csv(OUT / "02_固定事件卖压不确定区间.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lifecycle_rows).to_csv(OUT / "03_订单生命周期汇总.csv", index=False, encoding="utf-8-sig")
    total_snapshots = int(daily.snapshots_compared.sum())
    all_matches = int(daily.all_ten_prices_and_quantities_match.sum())
    counts = {
        "all_trade_rows": int(daily.trade_rows.sum()), "source_bs_matches_id_comparison": int(daily.source_bs_matches_order_id_comparison.sum()),
        "continuous_source_trade_rows": int(daily.continuous_source_rows.sum()), "passive_links_found": int(daily.passive_links_found.sum()),
        "missing_order_reference_events": int(daily.missing_order_reference_events.sum()),
        "negative_lifecycle_timestamp_batches": int(daily.negative_remaining_timestamp_batches.sum()),
        "aggressive_add_absent": int(daily.aggressive_add_absent.sum()),
        "direction_supported_by_strict_source_chronology": int(daily.internally_supported_direction_rows.sum()),
        "snapshots_compared": total_snapshots, "ten_level_price_quantity_exact_matches": all_matches,
        "ten_level_exact_match_fraction": all_matches / total_snapshots,
        "observation_windows_robust_net_selling": int(pressure.observation_net_selling_under_both_extremes.sum()),
        "observation_windows_robust_net_buying": int(pressure.observation_net_buying_under_both_extremes.sum()),
        "observation_windows_direction_uncertain": int((~pressure.observation_net_selling_under_both_extremes & ~pressure.observation_net_buying_under_both_extremes).sum()),
        "robust_net_selling_and_price_rebounded": int((pressure.observation_net_selling_under_both_extremes & pressure.observation_price_change_bps.gt(0)).sum()),
    }
    summary = {"status": "COMPLETED_SAME_SOURCE_LINEAGE_AND_BOOK_CHECK_ONLY", **counts,
               "independent_aggressor_validation": False, "snapshot_timing_shift_tuned": False,
               "complete_exchange_feed_proven": False, "own_queue_or_fill_probability_established": False,
               "three_days_are_independent_strategy_validation": False, "strategy_net_expectancy": "NOT_COMPUTED",
               "full_account_sharpe": "NOT_COMPUTED", "new_network_requests": 0, "actual_orders": 0,
               "goal_achieved": False, "position_impact": 0}
    save_json("summary.json", summary)
    save_json("goal_progress.json", {"previous_goal_turn_classification": "PROGRESS", "current_goal_turn_classification": "PROGRESS",
                                     "evidence": ["按交易所剩余委托口径完成被动单关联及数量生命周期检查", "完成零时间平移的逐档快照核对", "原17个事件保留方向不确定量并计算极端边界"],
                                     "goal_achieved": False, "goal_should_remain_active": True,
                                     "remaining": ["同源一致性不证明源转换与接收时钟正确", "缺同步篮子估值及60日同刻基线", "缺自身成交或独立验证的执行模型", "原净期望与全账户夏普目标尚未建立"]})
    if not all(sha256(ROOT / item["path"]) == item["sha256"] for item in receipt["files"]):
        raise AssertionError("登记输入或既有结果发生变化。")
    if len(pressure) != len(fixed_events) or set(pressure.event_id) != set(fixed_events.event_id):
        raise AssertionError("原事件集合改变。")
    if not ((pressure.shock_imbalance_lower <= pressure.shock_imbalance_upper).all() and (pressure.observation_imbalance_lower <= pressure.observation_imbalance_upper).all()):
        raise AssertionError("压力区间方向错误或分母缺失。")
    save_json("verification.json", {"source_hashes_and_existing_results_unchanged": True, "all_17_events_retained": True,
                                    "pressure_interval_ordering_passed": True, "clock_shift_search_performed": False,
                                    "lifecycle_has_missing_or_negative": bool(counts["missing_order_reference_events"] or counts["negative_lifecycle_timestamp_batches"]),
                                    "snapshot_mismatches_retained": total_snapshots - all_matches, "not_a_strategy_validation": True})
    lines = ["# 订单关联、库存与盘口快照：同源证据到哪一步", "",
             "**本轮只使用已有三个日期，纠正了将‘找不到主动订单新增记录’等同于漏数据的误解，并进一步核对已挂订单的数量与十档快照。以下均不构成自身订单成交验证。**", "",
             "## 为什么只按成交两侧找新增会低估覆盖", "",
             "已保存的上交所LDDS Level-2接口说明书2.0.10物理页21—22规定：连续竞价先发布主动成交，再发布剩余新增委托。完全被吃掉的主动订单不会留下新增余量。集合竞价则先发布订单，再发布集中成交。来源README也明确提示上海剩余委托口径。", "",
             "据此，连续成交的已挂一侧应扣减对应订单；不能又从主动订单随后才发布的余量中扣一次。订单关联使用ex_order_id，8月12日全零的order_id不能代替它。", "",
             "## 关联与库存结果", "",
             "| 日期 | 连续源成交 | 被动单新增关联 | 生命周期负余额批次 | 十档价量逐项一致快照 |", "|---|---:|---:|---:|---:|"]
    for row in daily.itertuples():
        lines.append(f"| {row.trade_date} | {row.continuous_source_rows:,} | {row.passive_links_found:,} | {row.negative_remaining_timestamp_batches:,} | {row.all_ten_prices_and_quantities_match:,}/{row.snapshots_compared:,} |")
    lines += ["", f"全部{counts['all_trade_rows']:,}条成交的来源B/S标记均与买卖订单号大小关系一致。该一致性本身可能来自同一个生成规则，不能当成独立验证。连续源成交中，{counts['direction_supported_by_strict_source_chronology']:,}条还获得严格早于成交的被动订单时间、方向、限价以及另一侧未早于成交的同源时序支持。无法区分同时间先后的记录不强行认定。", "",
              f"全日订单生命周期按相同时间戳一批处理，缺失订单关联事件{counts['missing_order_reference_events']:,}条，出现负剩余数量的批次{counts['negative_lifecycle_timestamp_batches']:,}个。没有假定一个时间戳内部的个人队列次序。盘后交易未重建自身队列。", "",
              f"快照核对只用不晚于快照源时间的订单与成交，不调整时钟偏移。共比较{total_snapshots:,}个连续时段有效十档，全部价位和数量都一致的有{all_matches:,}个，占{all_matches/total_snapshots:.2%}。不一致快照完整保留，不能用调时间偏移挑一个最好看的比例。", "",
              "同源两流或三流相互印证，能增强对内部字段和余量口径的理解；它们仍共享来源，未证明来源转换、源端接收时钟、全量报文与个人执行条件。即使数量重建通过，也不能默认限价单会成交。", "",
              "## 原17个事件的方向边界", "",
              "沿用修正后的(t−5分钟,t]窗口。仅使用获得上述严格同源时序支持的成交方向；把所有其余成交量分别按买入和卖出处理，得到量差上下界。其含义是‘在当前订单号和剩余委托解释成立的条件下’，不是外部验证的主动卖压。", "",
              f"固定五分钟观察期内，{counts['observation_windows_robust_net_selling']}个事件的量差上下界都为净卖出，{counts['observation_windows_robust_net_buying']}个都为净买入，{counts['observation_windows_direction_uncertain']}个仍无法确定。其中净卖出两端都成立、同时价格回升的有{counts['robust_net_selling_and_price_rebounded']}个。", "",
              "这些只是描述原样本，不删除失败事件，也没有依据这些标签另选策略。没有同步篮子及合格历史对照，仍不能认定是暂时卖压消退，或证明确认之后存在可取得的额外收益。", "",
              "## 原目标的剩余缺口", "",
              "M1仍需同步参考估值及误差边界、盘后自身排队或真实成交证据；M2仍需过去60个合格同刻交易日以及同价格/波动对照。本次只有三个盘口日期，不构成独立策略验证。原成交后净期望、胜率及全账户夏普仍为NOT_COMPUTED，原盈利目标尚未完成。", "",
              "本轮没有下载、付费、发单或改变旧失败策略。", "",
              "- [逐日关联与库存结果](01_逐日关联与库存结论.csv)",
              "- [17个固定事件的压力边界](02_固定事件卖压不确定区间.csv)",
              "- [订单生命周期汇总](03_订单生命周期汇总.csv)",
              "- 每日Parquet保存逐笔关联结论、源行号和所有重建十档快照，可与已存原件逐项对应。", ""]
    (OUT / "00_订单关联与盘口一致性结论.md").write_text("\n".join(lines), encoding="utf-8")
    index = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    pd.DataFrame(index).to_csv(OUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    size = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    if size > config["reporting"]["new_result_storage_limit_bytes"]:
        raise ValueError("本次派生结果超过预定15MiB空间预算。")
    print(json.dumps(clean({"完成": summary, "新增结果字节": size}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
