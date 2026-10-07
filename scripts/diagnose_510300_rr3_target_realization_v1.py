"""对冻结的全部3:1候选检查目标兑现；事件路径不能合成为新账户。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
PARENT = WORKSPACE / "reports/research/510300_state_mechanism_rr3_v1"
OUT = WORKSPACE / "reports/research/510300_rr3_target_realization_diagnostic_v1"
MAIN = WORKSPACE / "reports/research/510300_integrated_research_continuation_20260924"
STUDY = "510300_RR3_TARGET_REALIZATION_DIAGNOSTIC_V1"
START, END = "2021-01-04", "2026-08-14"
TARGET = "FROZEN_TARGET_CLOSE_NEXT_OPEN"
STOP = "STRUCTURAL_STOP_CLOSE_NEXT_OPEN"
TIME = "STRATEGY_TIME_LIMIT"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_parent():
    frozen = read(PARENT / "freeze.json")
    for item in frozen["files"]:
        assert sha(PARENT / item["path"]) == item["sha256"], item["path"]
    return frozen


def frozen_engine():
    spec = importlib.util.spec_from_file_location("frozen_rr3_event_engine", PARENT / "code/state_mechanism_rr3_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("目标兑现诊断已经冻结，不能覆盖。")
    parent_freeze = verify_parent()
    candidates = pd.read_parquet(PARENT / "results/all_candidate_facts.parquet")
    ready = candidates[candidates.reference_ticket_status.eq("READY")]
    primary = ready[ready.entry_date.between(START, END)]
    protocol = {
        "study_id": STUDY,
        "frozen_at": now(),
        "purpose": "补齐原账户受持仓占用而没有执行的候选，检查固定目标在固定失效和期限前是否兑现。",
        "diagnostic_after_parent_results_known": True,
        "source": "原冻结候选母集，不重造或改写任何触发、目标、失效、期限、状态或3:1门槛。",
        "population": "全部reference_ticket_status=READY；主样本2021-01-04至2026-08-14，之前只作历史诊断，不称留出。",
        "primary_candidates": len(primary),
        "all_history_candidates": len(ready),
        "event_path": "逐信号独立重置20万元，仅尝试原次日开盘一次。使用原冻结UNION成交与退出引擎，BASE和STRESS均保留。取消其他信号的资金占用只用于诊断，不构成可交易组合。",
        "execution": "原压力3:1票据、1%计划风险、整手、最低费、跳高不成交、T+1、分红应收、收盘触发次可卖开盘。不能以未来最高价假设目标成交。",
        "endpoint": "最多20或5个持有交易日，期限到达后等第一个可卖开盘；资料不足则保留未完成。失效后的上涨不计成功。",
        "independent_check": "用向量化首个收盘穿越目标或失效的位置复核触发，再检查下一可卖开盘；与原已成交四笔核对价格、日期和退出原因。",
        "event_groups": "以入场日至原最迟退出日的固定区间相交作传递合并；标签只说明时间重叠，不宣称组间独立。",
        "mechanism_assessment": "逐候选保留原时点PMI新订单、DR007利差与变化、成分股成交分类强度及变化、共同覆盖、节前与公开冲击。缺资料=NO_VIEW，不能把缺失当经济不支持。",
        "mechanism_comparison": "只描述原策略经济条件支持/不支持/NO_VIEW的目标兑现数；一侧无样本则不可识别，不拟合或重选条件。",
        "inference": "样本少且重叠，不计算独立二项置信区间，不计算事件拼接夏普，不把两成本或同行情重复信号当独立事件。",
        "new_model_fits": 0,
        "new_continuous_strategy_accounts": 0,
        "new_market_collection": False,
        "orders_authorized": False,
        "review_package": False,
        "frozen_parent_status_unchanged": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", protocol)
    (OUT / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), OUT / "code/diagnose_510300_rr3_target_realization_v1.py")
    sources = [PARENT / item["path"] for item in parent_freeze["files"]]
    sources.extend(PARENT / name for name in [
        "freeze.json", "result.json", "saved_diagnostic.json", "results/all_candidate_facts.parquet", "results/known_states.parquet",
        "results/accounts/BASE/UNION/trades.parquet", "results/accounts/STRESS/UNION/trades.parquet",
        "results/accounts/STRESS/UNION/ledger.parquet",
    ])
    sources.extend([OUT / "protocol.json", OUT / "code/diagnose_510300_rr3_target_realization_v1.py"])
    save(OUT / "freeze.json", {
        "frozen_at": now(), "before_new_event_paths": True,
        "files": [{"path": str(path.relative_to(WORKSPACE)), "sha256": sha(path)} for path in sources],
    })
    print(f"已冻结目标兑现诊断：全部历史{len(ready)}个，主样本{len(primary)}个。")


def first_close_trigger(d, dividends, candidate, entry_idx):
    """先以向量化首穿越检查固定经济路径，不调用账户退出实现。"""
    last = min(len(d) - 1, entry_idx + int(candidate.max_holding_sessions) - 1)
    segment = d.iloc[entry_idx:last + 1]
    entitlement = np.zeros(len(segment), dtype=float)
    entitled = dividends[(dividends.ex_date > candidate.entry_date) & (dividends.ex_date <= str(segment.date.iloc[-1]))]
    for event in entitled.itertuples():
        entitlement += (segment.date.to_numpy() >= event.ex_date) * float(event.cash_dividend_per_share)
    economic_close = segment.close.to_numpy(dtype=float) + entitlement
    hits_stop = np.flatnonzero(economic_close <= float(candidate.stop_raw))
    hits_target = np.flatnonzero(economic_close >= float(candidate.target_raw))
    stop_offset = int(hits_stop[0]) if len(hits_stop) else len(segment) + 1
    target_offset = int(hits_target[0]) if len(hits_target) else len(segment) + 1
    if min(stop_offset, target_offset) < len(segment):
        offset = min(stop_offset, target_offset)
        return entry_idx + offset, STOP if stop_offset <= target_offset else TARGET
    if last == entry_idx + int(candidate.max_holding_sessions) - 1:
        return last, TIME
    return None, "RIGHT_CENSORED"


def fixed_clusters(candidates, last_idx):
    ordered = candidates.sort_values(["entry_idx", "signal_id"])
    labels, cluster, right = {}, 0, -1
    for row in ordered.itertuples():
        start = int(row.entry_idx)
        end = min(last_idx, start + int(row.max_holding_sessions))
        if start > right:
            cluster += 1
        right = max(right, end)
        labels[row.signal_id] = f"重叠组{cluster:02d}"
    return labels


def describe(frame):
    filled = frame[frame.filled]
    completed = filled[filled.completed]
    return {
        "candidates": len(frame), "filled_events": len(filled), "completed_events": len(completed),
        "unfilled_events": int((~frame.filled).sum()),
        "right_censored_filled_events": int((frame.filled & ~frame.completed).sum()),
        "target_exits": int(completed.exit_reason.eq(TARGET).sum()),
        "stop_exits": int(completed.exit_reason.eq(STOP).sum()),
        "time_exits": int(completed.exit_reason.eq(TIME).sum()),
        "positive_events": int(completed.net_pnl.gt(0).sum()),
        "fixed_overlap_groups": int(frame.fixed_overlap_group.nunique()),
        "mean_event_R_descriptive_only": float(completed.realized_R.mean()) if len(completed) else None,
        "median_event_R_descriptive_only": float(completed.realized_R.median()) if len(completed) else None,
        "worst_event_R": float(completed.realized_R.min()) if len(completed) else None,
        "mean_event_R_is_not_account_return": True,
    }


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("诊断已经开始，禁止改口径重跑。")
    for item in read(OUT / "freeze.json")["files"]:
        assert sha(WORKSPACE / item["path"]) == item["sha256"], item["path"]
    assert sha(Path(__file__)) == sha(OUT / "code/diagnose_510300_rr3_target_realization_v1.py")
    verify_parent()
    save(OUT / "RUN_STARTED.json", {"started_at": now()})
    engine = frozen_engine()
    d, dividends, information = engine.load(PARENT)
    x = pd.read_parquet(PARENT / "results/known_states.parquet")
    assert d.date.tolist() == x.date.tolist()
    all_candidates = pd.read_parquet(PARENT / "results/all_candidate_facts.parquet")
    candidates = all_candidates[all_candidates.reference_ticket_status.eq("READY")].copy()
    labels = fixed_clusters(candidates, len(d) - 1)
    accounts = PARENT / "results/accounts"
    saved_trades = {cost: pd.read_parquet(accounts / cost / "UNION/trades.parquet") for cost in engine.COSTS}
    union_ledger = pd.read_parquet(accounts / "STRESS/UNION/ledger.parquet").set_index("idx")
    rows, all_ledgers, all_decisions, verification_rows, matched_saved_cycles = [], [], [], 0, 0
    for cost in engine.COSTS:
        for candidate in candidates.itertuples():
            i = int(candidate.entry_idx)
            fact = x.iloc[i]
            end = min(len(d) - 1, i + int(candidate.max_holding_sessions))
            # 只延长等待执行，不延长入场信号或原持有期限。
            while end < len(d) - 1 and not engine.can_fill(d, end, "SELL"):
                end += 1
            selected = candidates[candidates.signal_id.eq(candidate.signal_id)]
            ledger, trades, decisions, terminal = engine.simulate(d, dividends, x, selected, "UNION", cost, i, end)
            np.testing.assert_allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7, rtol=0)
            verification_rows += len(ledger)
            assert len(trades) <= 1
            buys = decisions[decisions.action.eq("BUY_FILLED")]
            filled, completed = len(buys) == 1, len(trades) == 1
            plan, status = engine.ticket(200000, 200000, candidate.reference_raw, candidate.target_raw, candidate.stop_raw)
            assert status == "READY"
            known = bool(fact.state_information_known)
            support = bool(fact.trend_mechanism_support if candidate.kind == "TREND_PULLBACK" else fact.repair_mechanism_support)
            econ = "NO_VIEW" if not known else ("SUPPORTED" if support else "NOT_SUPPORTED")
            primary = START <= candidate.entry_date <= END
            busy = bool(primary and i - 1 in union_ledger.index and int(union_ledger.loc[i - 1, "shares"]) > 0)
            row = {
                "signal_id": candidate.signal_id, "kind": candidate.kind, "cost": cost,
                "entry_idx": i, "entry_date": candidate.entry_date,
                "scope": "PRIMARY" if primary else "EARLIER_DIAGNOSTIC",
                "fixed_overlap_group": labels[candidate.signal_id],
                "fixed_window_end": str(d.date.iloc[min(len(d) - 1, i + int(candidate.max_holding_sessions))]),
                "max_holding_sessions": int(candidate.max_holding_sessions),
                "reference_raw": candidate.reference_raw, "target_raw": candidate.target_raw, "stop_raw": candidate.stop_raw,
                "reference_rr": candidate.reference_planned_net_rr, "planned_net_rr_at_limit": plan["planned_net_rr_at_limit"],
                "buy_limit": plan["buy_limit"], "planned_loss_cny": plan["planned_loss_cny"],
                "quantity": int(plan["quantity"]), "open_raw": float(d.open.iloc[i]),
                "filled": filled, "completed": completed,
                "entry_decision_reason": str(decisions.reason.iloc[0]),
                "original_union_occupied_at_day_start": busy,
                "original_union_executed_same_signal": bool(saved_trades[cost].signal_id.eq(candidate.signal_id).any()),
                "market_state": fact.market_state,
                "macro_state": fact.macro_state if bool(fact.pmi_known) else "NO_VIEW",
                "funding_state": fact.funding_state if bool(fact.funding_known) else "NO_VIEW",
                "economic_assessment": econ,
                "state_information_known": known,
                "known_new_orders": float(fact.pmi_level + 50) if bool(fact.pmi_known) else None,
                "new_orders_change3": float(fact.pmi_change3) if bool(fact.pmi_known) else None,
                "funding_gap_pp": float(fact.funding_gap_pp) if bool(fact.funding_known) else None,
                "funding_change5_pp": float(fact.funding_change5_pp) if bool(fact.funding_known) else None,
                "flow_known": bool(fact.flow_known),
                "flow5": float(fact.flow5) if bool(fact.flow_known) else None,
                "flow_breadth5": float(fact.flow_breadth5) if bool(fact.flow_known) else None,
                "flow_change3": float(fact.flow_change3) if pd.notna(fact.flow_change3) else None,
                "breadth_change3": float(fact.breadth_change3) if pd.notna(fact.breadth_change3) else None,
                "preholiday3": bool(fact.preholiday3), "known_scheduled_event": bool(fact.known_scheduled_event),
                "global_shock": bool(fact.global_shock),
                "trigger_idx": None, "trigger_date": None, "trigger_reason": None,
                "exit_reason": "UNFILLED" if not filled else "RIGHT_CENSORED",
                "exit_date": None, "exit_idx": None, "net_pnl": None, "realized_R": None,
                "actual_open_stress_net_rr": None,
            }
            if filled:
                trigger_idx, trigger_reason = first_close_trigger(d, dividends, candidate, i)
                row.update(trigger_idx=trigger_idx, trigger_date=str(d.date.iloc[trigger_idx]) if trigger_idx is not None else None, trigger_reason=trigger_reason)
                expected_exit = None
                if trigger_idx is not None:
                    expected_exit = next((j for j in range(max(i + 1, trigger_idx + 1), end + 1) if engine.can_fill(d, j, "SELL")), None)
                assert completed == (expected_exit is not None)
                trade = trades.iloc[0] if completed else terminal["active"]
                assert float(trade["actual_open_stress_net_rr"]) >= 3 - 1e-10
                assert float(trade["planned_loss_cny"]) <= 2000 + 1e-7
                row["actual_open_stress_net_rr"] = float(trade["actual_open_stress_net_rr"])
                if completed:
                    assert int(trade.exit_idx) == expected_exit and int(trade.exit_idx) > i
                    assert trade.exit_reason == trigger_reason
                    for field in ["exit_reason", "exit_date", "exit_idx", "net_pnl", "realized_R", "entry_price", "exit_price", "dividend_cny", "entry_fee", "exit_fee"]:
                        row[field] = trade[field]
                saved = saved_trades[cost][saved_trades[cost].signal_id.eq(candidate.signal_id)]
                if len(saved):
                    assert completed and len(saved) == 1
                    original = saved.iloc[0]
                    for field in ["entry_idx", "exit_idx", "entry_price", "exit_price", "exit_reason"]:
                        assert trade[field] == original[field], (candidate.signal_id, field)
                    matched_saved_cycles += 1
            for frame, target in [(ledger, all_ledgers), (decisions, all_decisions)]:
                frame = frame.copy()
                frame["event_signal_id"] = candidate.signal_id
                frame["cost"] = cost
                target.append(frame)
            rows.append(row)
    events = pd.DataFrame(rows)
    results = OUT / "results"
    results.mkdir(exist_ok=True)
    events.to_parquet(results / "event_outcomes.parquet", index=False)
    pd.concat(all_ledgers, ignore_index=True).to_parquet(results / "event_ledgers.parquet", index=False)
    pd.concat(all_decisions, ignore_index=True).to_parquet(results / "event_decisions.parquet", index=False)
    aggregates = {}
    for cost in engine.COSTS:
        part = events[events.cost.eq(cost)]
        aggregates[cost] = {"ALL_HISTORY": describe(part)}
        for scope in ["PRIMARY", "EARLIER_DIAGNOSTIC"]:
            aggregates[cost][scope] = describe(part[part.scope.eq(scope)])
    groups = []
    for keys, group in events.groupby(["scope", "cost", "kind", "economic_assessment"], dropna=False):
        groups.append({"scope": keys[0], "cost": keys[1], "kind": keys[2], "economic_assessment": keys[3], **describe(group)})
    primary_stress = events[events.cost.eq("STRESS") & events.scope.eq("PRIMARY")]
    has_supported = primary_stress.economic_assessment.eq("SUPPORTED").any()
    has_unsupported = primary_stress.economic_assessment.eq("NOT_SUPPORTED").any()
    mechanism_status = "DESCRIPTIVE_ONLY_SMALL_OVERLAPPING_SAMPLE" if has_supported and has_unsupported else "NOT_IDENTIFIABLE_MISSING_COMPARISON_SIDE"
    result = {
        "study_id": STUDY, "completed_at": now(),
        "status": "FROZEN_TARGET_REALIZATION_DIAGNOSTIC_ONLY",
        "event_path_count": len(events), "unique_candidates": int(events.signal_id.nunique()),
        "continuous_strategy_account_count": 0, "new_model_fits": 0,
        "aggregates": aggregates, "mechanism_groups": groups, "mechanism_increment_status": mechanism_status,
        "economic_support_is_causal_proof": False,
        "overlap_groups_are_independent_observations": False,
        "event_profits_can_be_added_as_account_profit": False,
        "prior_rejections_changed": False, "goal_achieved": False,
        "new_market_collection": False, "orders_authorized": False,
        "next_question": "历史目标位置不足以证明兑现概率；现有合格候选太少且缺少经济支持的一侧，不能从同批结果反推新的宏观状态映射。",
    }
    save(OUT / "result.json", result)
    verify_parent()
    save(OUT / "verification.json", {
        "checked_at": now(), "status": "PASS_EVENT_MECHANICS_NOT_STRATEGY_VALIDATION",
        "parent_frozen_bytes_unchanged": True, "ledger_rows": verification_rows,
        "actual_saved_cycles_matched_across_two_costs": matched_saved_cycles,
        "independent_first_close_and_next_open_checks": int(events.filled.sum()),
        "all_filled_stress_rr_at_least_3": True,
        "T_plus_1_and_fixed_exit_preserved": True,
        "new_information_collected": False,
    })
    write_report(events, result)
    print(json.dumps(clean({"主样本压力结果": aggregates["STRESS"]["PRIMARY"], "全部历史压力结果": aggregates["STRESS"]["ALL_HISTORY"], "经济增量可识别性": mechanism_status}), ensure_ascii=False, indent=2))


def write_report(events, result):
    a = result["aggregates"]["STRESS"]["PRIMARY"]
    older = result["aggregates"]["STRESS"]["EARLIER_DIAGNOSTIC"]
    p = events[events.cost.eq("STRESS") & events.scope.eq("PRIMARY")]
    reason_zh = {TARGET: "达到冻结目标后退出", STOP: "先触发结构失效退出", TIME: "持有期限到期退出", "UNFILLED": "开盘未成交", "RIGHT_CENSORED": "截至数据终点仍未完成"}
    lines = [
        f"已逐一检查冻结3:1母集中的全部{result['unique_candidates']}个历史候选，两档成本共{result['event_path_count']}条单信号路径。主样本9个候选，在压力成本下{a['filled_events']}个可成交、{a['unfilled_events']}个未成交；{a['target_exits']}个目标退出、{a['stop_exits']}个结构失效退出、{a['time_exits']}个期限退出。每次均从独立20万元开始，这些重叠事件的金额不能相加成策略收益。",
        "这一步补齐了原连续账户因持仓占用而没有执行的信号。它是看到原研究结果后的原因诊断，不是新的留出验证，也没有修改原状态路由、目标、失效、20/5日期限、1%计划风险或压力成本后3:1要求。",
        f"主样本的9个候选按固定持有区间归并为{a['fixed_overlap_groups']}个时间重叠组；这个数只是重叠说明，不是独立事件数。此前历史还有{older['candidates']}个合格候选，压力成本下{older['filled_events']}个成交、{older['target_exits']}个目标退出；该段同样被反复研究，不能作为新的独立证据。",
        "逐候选的压力结果如下。金额只对应独立事件参考本金，不等于原连续账户金额；盈利也不等于到达原目标。",
    ]
    for row in p.itertuples():
        state = "共同宏观资金资料不齐，经济支持为NO_VIEW" if row.economic_assessment == "NO_VIEW" else ("原经济支持条件满足" if row.economic_assessment == "SUPPORTED" else "资料齐备，但原经济支持条件未满足")
        macro = f"当时已知新订单{row.known_new_orders:.1f}，三个月变化{row.new_orders_change3:+.1f}个百分点" if pd.notna(row.known_new_orders) else "没有当时可用的新订单判断"
        outcome = f"{reason_zh[row.exit_reason]}，净损益{row.net_pnl:+.2f}元、{row.realized_R:+.3f}R" if row.completed else reason_zh[row.exit_reason]
        occupied = "；原账户当日开始已有持仓" if row.original_union_occupied_at_day_start else ""
        lines.append(f"- {row.entry_date}拟入场，{'趋势回调' if row.kind == 'TREND_PULLBACK' else '急跌修复'}：{outcome}。{macro}；{state}{occupied}。")
    lines.extend([
        f"经济条件比较的正式状态为{result['mechanism_increment_status']}。不能把宏观扩张、资金利率回落单独当成完整经济支持；还需要股票参与事实及其时点。缺失资料保留为无观点，不能算反证。当前样本不足以可靠估计这些状态下达到目标的概率，更不能根据这几笔的成败重新选择状态定义。",
        "3:1只是在固定目标和失效价下的事前空间比例。如果只有恰好+3R与-1R两种结果，且成本已经计入，盈亏平衡胜率才是25%；本研究还存在期限退出、跳空越过失效和未成交，不能套用这一二点分布当作真实胜率门槛。真实目标必须在失效之前达到，并按下一可卖开盘扣费计算结果。",
        "节前三日、计划发布和已公开外部冲击标签继续保留在事件事实中。没有用这批结果调整节前窗口或把原失败避险规则反向操作；跨休市风险和T+1损失也没有从路径中删除。",
        "本次保持原冻结研究全部字节不变，独立首个收盘穿越及下一可卖开盘检查与原引擎一致，原实际成交的四笔在两档成本下价格、日期和退出原因均一致。实现一致性不构成经济机制的因果证明。",
        "高夏普目标仍未完成。现在缺少的不是把目标画得更远，而是足够的事前事实和独立事件，证明该目标在特定市场背景、资金条件和期限内值得承担风险。当前结果不支持将本轮候选直接作为操作方案。",
    ])
    report = "\n\n".join(lines) + "\n"
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    status_path = MAIN / "current_status.json"
    status = read(status_path)
    status.update({
        "updated_at": now(), "goal_achieved": False,
        "latest_goal_turn_classification": "PROGRESS_FROZEN_RR3_ALL_CANDIDATE_REALIZATION_DIAGNOSTIC",
        "same_condition_consecutive_no_progress_goal_turns": 0,
        "latest_rr3_realization_diagnostic": str(OUT.relative_to(WORKSPACE)),
        "rr3_realization_primary_stress": a,
        "rr3_realization_mechanism_increment_status": result["mechanism_increment_status"],
        "remaining_research_question": result["next_question"],
    })
    diagnostics = status.setdefault("additional_diagnostics", [])
    if STUDY not in diagnostics:
        diagnostics.append(STUDY)
    save(status_path, status)
    main_report = MAIN / "最新研究结论.md"
    prefix = (
        f"本续轮补齐了全部冻结高比例候选的目标兑现诊断。主样本9个候选，压力成本下{a['filled_events']}个单独可成交、{a['unfilled_events']}个未成交；其中{a['target_exits']}个达到冻结目标、{a['stop_exits']}个先结构失效、{a['time_exits']}个期限退出。包括原账户因已有持仓而未执行的候选，未按结果挑选。\n\n"
        f"这些是独立重置本金的事件路径，存在{a['fixed_overlap_groups']}个固定时间重叠组，不能拼接为账户利润或夏普。宏观资金支持比较为{result['mechanism_increment_status']}，不能由缺少对照一侧推出宏观无效。原3:1、目标、止损、期限、状态和先前裁决保持不变。\n\n"
        f"详见[全部3:1候选的目标兑现](<{(OUT / '研究结论.md').as_posix()}>)。高夏普目标仍未完成。以下保留既有综合记录。\n\n"
    )
    main_report.write_text(prefix + main_report.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结并执行3:1全部候选的目标兑现诊断")
    parser.add_argument("command", choices=["freeze", "run"])
    command = parser.parse_args().command
    freeze() if command == "freeze" else run()
