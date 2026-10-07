"""保存五分钟流量修正与同价位深度分解，保留原事件及策略结论。"""

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

from research.pressure_recovery_local_exploration_v1 import detect_price_episodes, quote_at, with_clock
from research.pressure_recovery_measurement_correction_v1_1 import FLOW_COLUMNS, correct_flow_columns, depth_decomposition

PARENT = ROOT / "reports/research/510300_pressure_recovery_v1/local_exploration_20261001"
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/measurement_correction_20261001"
CONFIG = ROOT / "config/510300_pressure_recovery_measurement_correction_v1_1.json"
SOURCE = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo_selected"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def direct_window_check(grid: pd.DataFrame, orders: pd.DataFrame, trades: pd.DataFrame) -> int:
    """逐个时点直接筛选源消息，与累积和算法独立核对六个字段。"""
    tested = 0
    for row in grid.itertuples():
        start, end = row.decision_at - pd.Timedelta(minutes=5), row.decision_at
        selected_trades = trades.loc[trades._at.gt(start) & trades._at.le(end)]
        selected_orders = orders.loc[orders._at.gt(start) & orders._at.le(end)]
        total = float(selected_trades.volume.sum())
        signed = float(selected_trades.loc[selected_trades.bs_flag.eq("B"), "volume"].sum()) - float(selected_trades.loc[selected_trades.bs_flag.eq("S"), "volume"].sum())
        expected = {"trade_volume_5m_source_units": total, "reported_bs_volume_imbalance_5m": signed / total if total > 0 else np.nan}
        for kind in ("A", "D"):
            for side in ("B", "S"):
                expected[f"reported_order_{kind}_{side}_volume_5m"] = float(selected_orders.loc[selected_orders.order_type.eq(kind) & selected_orders.order_code.eq(side), "volume"].sum())
        for field, value in expected.items():
            if not np.isclose(getattr(row, field), value, atol=1e-10, equal_nan=True):
                raise AssertionError(f"显式时间筛选不一致：{row.decision_at}，{field}")
            tested += 1
    return tested


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit("修正目录非空，拒绝覆盖本批证据。")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_folders = {}
    for day in config["source_dates"]:
        folders = [p for p in SOURCE.glob(day + "_*") if all((p / (name + ".parquet")).exists() for name in ("行情", "逐笔委托", "逐笔成交"))]
        if len(folders) != 1:
            raise ValueError("未找到唯一的已完成来源日期：" + day)
        source_folders[day] = folders[0]
    inputs = [CONFIG, Path(__file__), ROOT / "research/pressure_recovery_measurement_correction_v1_1.py",
              ROOT / "research/pressure_recovery_local_exploration_v1.py", ROOT / "research/pressure_recovery_v1.py",
              PARENT / "01_盘口与消息测量表.parquet", PARENT / "02_固定观察事件表.csv",
              ROOT / "reports/research/510300_pressure_recovery_v1/volatility_conditioned_proxy_20261001/acceptance_outcome.json"]
    inputs += [p / (name + ".parquet") for p in source_folders.values() for name in ("行情", "逐笔委托", "逐笔成交")]
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = {"registered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "protocol": config,
               "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in inputs]}
    save_json("registration_receipt.json", receipt)
    original = pd.read_parquet(PARENT / "01_盘口与消息测量表.parquet")
    events = pd.read_csv(PARENT / "02_固定观察事件表.csv")
    corrections, changes, decompositions, explicit_checks = [], [], [], 0
    for compact, folder in source_folders.items():
        day = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        quotes, orders, trades = [with_clock(pd.read_parquet(folder / (name + ".parquet"))) for name in ("行情", "逐笔委托", "逐笔成交")]
        old = original.loc[original.trade_date.eq(day)].copy()
        new = correct_flow_columns(old, orders, trades)
        explicit_checks += direct_window_check(new, orders, trades)
        unaffected = [name for name in old.columns if name not in FLOW_COLUMNS]
        pd.testing.assert_frame_equal(old[unaffected], new[unaffected])
        if detect_price_episodes(old) != detect_price_episodes(new):
            raise AssertionError("流量修正意外改变了价格事件。")
        corrections.append(new)
        for column in FLOW_COLUMNS:
            for at, before, after in zip(old.decision_at, old[column], new[column]):
                changes.append({"decision_at": at, "trade_date": day, "field": column,
                                "original_value": before, "corrected_value": after,
                                "changed": not np.isclose(before, after, atol=1e-10, equal_nan=True)})
        for event in events.loc[events.trade_date.eq(day)].itertuples():
            start = quote_at(quotes, event.shock_at)
            end = quote_at(quotes, event.observation_end_at)
            if start is None or end is None:
                raise AssertionError("原固定事件时点没有可追溯报价。")
            detail = depth_decomposition(start, end, config["depth_diagnostic"]["band_bps"])
            if not np.isclose(detail["moving_depth_ratio"], event.bid_depth_ratio, rtol=1e-9):
                raise AssertionError("盘口原值与原深度比不一致。")
            if not np.isclose(detail["moving_depth_change_cny"], detail["same_absolute_band_net_change_cny"] + detail["band_shift_net_component_cny"], atol=1e-6):
                raise AssertionError("价格带深度归因不闭合。")
            before_flow = new.loc[new.decision_at.eq(pd.Timestamp(event.shock_at))].iloc[0]
            after_flow = new.loc[new.decision_at.eq(pd.Timestamp(event.observation_end_at))].iloc[0]
            decompositions.append({"event_id": event.event_id, "trade_date": day, "shock_at": event.shock_at,
                                   "observation_end_at": event.observation_end_at, "start_quote_at": start._at,
                                   "endpoint_quote_at": end._at, "observation_price_change_bps": event.observation_price_change_bps,
                                   "spread_label": event.spread_label, **detail,
                                   "shock_5m_trade_volume": before_flow.trade_volume_5m_source_units,
                                   "observation_5m_trade_volume": after_flow.trade_volume_5m_source_units,
                                   "shock_reported_bs_imbalance": before_flow.reported_bs_volume_imbalance_5m,
                                   "observation_reported_bs_imbalance": after_flow.reported_bs_volume_imbalance_5m,
                                   "reported_bs_direction_verified": False,
                                   "formal_m2_status": "NO_VIEW", "actual_order": False})
        print(f"{day}：237个时点已用直接时间比较核对，价格事件保持不变。", flush=True)
    corrected = pd.concat(corrections).sort_index()
    changes = pd.DataFrame(changes)
    depth = pd.DataFrame(decompositions)
    corrected.to_parquet(OUT / "01_五分钟消息测量修正版.parquet", index=False)
    corrected.to_csv(OUT / "01_五分钟消息测量修正版.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(OUT / "02_新旧字段逐项对照.csv", index=False, encoding="utf-8-sig")
    depth.to_csv(OUT / "03_固定事件深度分解.csv", index=False, encoding="utf-8-sig")
    grouped = depth.assign(price_rebounded=depth.observation_price_change_bps.gt(0), moving_depth_increased=depth.moving_depth_ratio.gt(1))
    grouped.groupby(["price_rebounded", "moving_depth_increased"]).size().rename("events").reset_index().to_csv(OUT / "04_价格与深度状态计数.csv", index=False, encoding="utf-8-sig")
    increase = depth.loc[depth.moving_depth_ratio.gt(1)]
    all_changed = changes.groupby("field").changed.sum().astype(int).to_dict()
    original_signed = original.reported_bs_volume_imbalance_5m.to_numpy()
    corrected_signed = corrected.reported_bs_volume_imbalance_5m.to_numpy()
    sign_flips = int(((original_signed * corrected_signed) < 0).sum())
    summary = {"status": "COMPLETED_MEASUREMENT_CORRECTION_NOT_ALPHA_VALIDATION", "measurement_rows": len(corrected),
               "explicit_timestamp_checks": explicit_checks, "corrected_field_changed_rows": all_changed,
               "source_bs_imbalance_sign_flips": sign_flips, "original_price_events": len(events),
               "all_nonflow_fields_and_price_events_unchanged": True, "depth_increase_events": len(increase),
               "depth_increase_but_original_band_not_increased": int(increase.initial_band_depth_ratio.le(1).sum()),
               "depth_increase_but_no_common_band_positive_change": int(increase.same_absolute_band_net_change_cny.le(0).sum()),
               "depth_increase_with_nonpositive_observation_price": int(increase.observation_price_change_bps.le(0).sum()),
               "depth_windows_fully_visible": int(depth.both_moving_bands_fully_visible.sum()),
               "original_band_endpoint_unknown": int((~depth.initial_band_at_endpoint_fully_visible).sum()),
               "source_days": 3, "original_minimum_independent_days_for_interval": 30,
               "matched_price_only_control_status": "NOT_ESTABLISHED_60_QUALIFIED_SAME_SLOT_DAYS_AND_SYNCHRONIZED_REFERENCE_MISSING",
               "source_bs_is_verified_aggressor": False, "source_orders_are_complete_exchange_feed": False,
               "strategy_net_expectancy": "NOT_COMPUTED", "full_account_sharpe": "NOT_COMPUTED",
               "new_network_requests": 0, "actual_orders": 0, "goal_achieved": False, "position_impact": 0}
    save_json("summary.json", summary)
    if not all(sha256(ROOT / item["path"]) == item["sha256"] for item in receipt["files"]):
        raise AssertionError("登记输入或旧研究文件发生变化。")
    save_json("verification.json", {"direct_source_timestamp_comparisons": explicit_checks, "all_passed": True,
                                    "original_price_events_unchanged": True, "unchanged_nonflow_fields": unaffected,
                                    "source_and_parent_hashes_unchanged": True, "not_a_strategy_validation": True})
    example = depth.loc[depth.event_id.eq("P0_20260904_1326")].iloc[0]
    report = f"""# 五分钟消息测量修正与深度变化解释

**本轮修正了真实的时间单位错误，并确认‘跟随中间价移动的买盘金额增加’不能直接解释为原价位承接恢复。没有增加策略、改变事件或重跑已拒绝的波动切换。**

## 已修正的错误

本机pandas 3.0.5将消息时间和测量端点保存为微秒。原函数直接取整数，却减去纳秒表示的五分钟长度；原‘五分钟’因此覆盖约3.47天，在逐日输入中主要表现为盘中累计。两侧时间现均显式统一为纳秒，窗口严格为(t−5分钟,t]。

三个日期711个时点的六个字段重新计算，并逐项用Timestamp显式筛选原始消息核对，共{explicit_checks:,}项通过。修正字段为成交量、来源B/S标记量不平衡，以及买卖新增/撤销量。来源B/S标记和订单消息完整性仍未经独立验证，不称为已验证主动买卖。

7月9日10:00的原‘五分钟成交量’为171,740,800份，修正后为18,805,200份。全部测量时点中，来源B/S不平衡符号发生反转的有{sign_flips}个。后续应读取本目录修正版，不使用旧表六个字段做压力判断。

旧表的价格、价差、深度、17个事件及其固定观察终点逐项保持不变。原事件只由价格定义；上一轮波动切换也只使用独立分钟价格，因此没有受这些错误流量字段驱动。旧文件作为历史证据保留。

## 深度增加具体发生在哪里

原17个事件中，移动价格带买盘金额增加的有{len(increase)}个。保持同一10bp宽度时，价格带会随中间价移动；比较的绝对价格位置可能已经变了。

本轮将变化精确拆为：共同绝对价格区间内金额净变化，加上新进入区间金额，减去退出区间金额。另将冲击时的绝对价格带固定到观察结束。后者用于解释口径，不作为新交易信号。

在{len(increase)}个移动深度增加事件中：

- {summary['depth_increase_but_original_band_not_increased']}个在原绝对价格带的买盘金额没有增加。
- {summary['depth_increase_but_no_common_band_positive_change']}个没有正的共同价位区间金额净变化。
- {summary['depth_increase_with_nonpositive_observation_price']}个在观察期内价格没有回升。

所有{summary['depth_windows_fully_visible']}个事件的两端移动价格带均在可见十档范围内；固定原价格带到终点有{summary['original_band_endpoint_unknown']}个无法完整看见，保留为未知。

例如9月4日13:26事件：五分钟后，随价格移动的买盘金额为起点的{example.moving_depth_ratio:.3f}倍，价格却下跌{abs(example.observation_price_change_bps):.2f}基点；两个价格带重叠比例为{example.band_overlap_fraction:.1%}，原绝对价格带的终点买盘金额为{example.initial_band_endpoint_depth_cny:,.2f}元。更多低价位买单不能直接解释成原价位已被承接。

共同价位金额变化也只是净快照变化，不能从两个端点判断其中多少属于新增、撤单或成交。没有据此将某个事件升格为M2信号。

## 对原研究的影响

原文要求与相同时段、相近跌幅、相近波动的价格规则比较，再证明盘口信息有额外价值。当前仍只有三个盘口日期，缺少过去60个合格同刻交易日、同步参考估值和已验证成交；不足以完成该项检验，亦低于原合同用于区间估计的30个独立日期门槛。

本轮解决了可修复的测量错误，并给出‘深度增加但未恢复’的具体反例。原M1/M2成交后净期望、胜率和全账户夏普仍为NOT_COMPUTED。没有补采，没有实际订单，没有新参数搜索。

## 可查看文件

- [五分钟消息测量修正版](01_五分钟消息测量修正版.csv)
- [新旧字段逐项对照](02_新旧字段逐项对照.csv)
- [17个固定事件的深度分解](03_固定事件深度分解.csv)
- [价格与深度状态计数](04_价格与深度状态计数.csv)
"""
    (OUT / "00_测量修正与微观增量结论.md").write_text(report, encoding="utf-8")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    pd.DataFrame(files).to_csv(OUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    size = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    if size > config["reporting"]["result_size_limit_bytes"]:
        raise AssertionError("本批结果超过5MiB预算。")
    print(json.dumps({"完成": summary, "新增结果字节": size, "案例": example.to_dict()}, default=str, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
