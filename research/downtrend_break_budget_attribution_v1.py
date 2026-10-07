"""解释全部首次突破的原资金预算及实际减仓，不改策略或运行账户。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import daily_supply_test_v1 as risk_base
from research.point_fresh_repair_order_study_v1 import load
from research.point_account_nr7_inputs_v1 import fill, fee, open_blocked
from research.downtrend_break_study_v1 import OUT as PRIOR, EXPLANATION, PRIMARY, PERIODS, COSTS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

OUT = ROOT / "reports/research/510300_downtrend_break_budget_attribution_v1"
FORWARD_KEYS = (
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation",
)
DOCS = ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md")
MASK_LABELS = {0: "未由风险预算触发", 1: "仓位", 2: "ES", 3: "仓位+ES", 4: "跳空", 5: "仓位+跳空", 6: "ES+跳空", 7: "仓位+ES+跳空"}


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def mask(q, raw, budgets, buying=False, existing=0):
    """完整列出原risk_ok失败项，不将多个同时约束硬选为一个主因。"""
    if not q:
        return 0
    friction = q * (fill(raw, 1, "STRESS") - raw) + fee(q * fill(raw, 1, "STRESS"), "STRESS") if buying else 0.
    if existing > q:
        sold = existing - q
        friction += sold * (raw - fill(raw, -1, "STRESS")) + fee(sold * fill(raw, -1, "STRESS"), "STRESS")
    exit_friction = q * (raw - fill(raw, -1, "STRESS")) + fee(q * fill(raw, -1, "STRESS"), "STRESS")
    notional = q * raw
    position = notional > budgets["position_budget"] - .5 * friction + 1e-8
    es = notional * budgets["es95"] + exit_friction + friction > budgets["es_budget"] + 1e-8
    gap = .1 * notional + exit_friction + friction > budgets["gap_budget"] + 1e-8
    value = int(position) + 2 * int(es) + 4 * int(gap)
    require(bool(value) == (not risk_base.risk_ok(q, raw, budgets, buying=buying, existing=existing)),
            "预算失败项与原风险函数不同。")
    return value


def budget_fields(nav, peak, es, raw, desired, maximum, existing=0):
    budget = risk_base.limits(nav, peak, es)
    base_gap, dd_gap = .05 * nav, .5 * max(0., nav - .9 * peak)
    source = ("REMAINING_DD_MARGIN" if dd_gap < base_gap - 1e-8 else
              "BASE_AND_DD_TIED" if abs(dd_gap - base_gap) <= 1e-8 else "BASE_GAP_ONLY")
    binding = mask(desired + 100, raw, budget, existing=existing) if desired < maximum or not existing else 0
    return {
        "known_nav": nav, "known_peak": peak, "known_drawdown": 1 - nav / peak, "known_es95": es,
        "known_position_budget": budget["position_budget"], "known_es_budget": budget["es_budget"],
        "known_gap_budget": budget["gap_budget"], "known_gap_budget_source": source,
        "known_budget_binding_mask": binding, "known_budget_binding": MASK_LABELS[binding],
        "known_desired_quantity": desired, "known_maximum_quantity_before_risk": maximum,
        "known_desired_notional_ratio": desired * raw / nav,
    }


def freeze():
    require(not (OUT / "protocol.json").exists(), "全体原资金预算归因已经固定。")
    result, delivery = read(PRIOR / "summary.json"), read(PRIOR / "delivery_receipt.json")
    require(result["technical_decision"] == "TECH.R173" and delivery["all_qualification_rows"] == 132,
            "R173全体金融结果及点位交付尚未完成。")
    paths = {}
    for source in read(PRIOR / "protocol.json")["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "R173冻结来源改变。")
        paths[source["path"]] = source["sha256"]
    extras = [Path(__file__), PRIOR / "summary.json", PRIOR / "delivery_receipt.json",
              PRIOR / "next_all_signal_budget_attribution_proposal.json", PRIOR / "saved_result_verification.json",
              PRIOR / "results/全部首次突破资格_当时指标成交取消及A覆盖.parquet",
              PRIOR / "results/完整账户共同口径比较.parquet",
              EXPLANATION / "results/全部原点_上一确认下降结构与首次突破.parquet",
              EXPLANATION / "results/四案例逐日完整量价与上一确认下降结构.parquet",
              EXPLANATION / "results/原61分段及49正式波段_完整首次突破覆盖.parquet",
              ROOT / "research/joint_onset_budget_attribution_v1.py",
              ROOT / "reports/research/510300_joint_onset_budget_attribution_v1/summary.json"]
    for period in PERIODS:
        for cost in COSTS:
            folder = PRIOR / f"results/accounts/{period}/{cost}/{PRIMARY}"
            extras.extend(folder / f"{name}.parquet" for name in ("daily", "orders", "trades", "decisions", "rejections"))
    for path in extras:
        paths[path.relative_to(ROOT).as_posix()] = digest(path)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_DOWNTREND_BREAK_SAVED_BUDGET_ATTRIBUTION_V1", "technical_decision": "TECH.R174",
        "hypothesis": "R173四场景实际交易质量正而近期完整账户弱，可能涉及原已知ES/剩余回撤跳空预算压缩入场、持有风险减仓及费用；需要完整原账本归因。",
        "scope": "132已知资格/106原周期/全部原订单；两时期两费用，11次开取消及2已有持仓每费用分别保留；完成104/开放2不混合。",
        "role": "POST_RESULT_DIAGNOSTIC_OF_SAVED_RESULTS_NOT_NEW_FINANCIAL_CANDIDATE",
        "origin_budget": "从原收盘现金/应收/份额/NAV/当时峰值与当时ES精确复算50%意向及原cap_quantity，不用未来结果决定份额。已有持仓原点不冒作新入场。",
        "opening": "复算原真实次开和已发生分红平移、限价/旧H2取消、原开盘预算及现金、STRESS风险检验和实际费用；全部原买单份额/成交价/费用匹配。",
        "holding": "每个实际持有原点复算仅向上确认低点floor、已知跌破退出或原cap_quantity(existing=q)减仓，全部实际卖单数量/成交价/费用精确匹配。",
        "groups": "两时期×两费用×七非空风险失败项集合共28个入场单元、同28个持有减仓单元，空组保留未知均值；四场景×BUY/风险减仓/结构退出共12费用单元；不挑赢家组。",
        "case_chart": "只使用原四固定案例窗，收盘价格及原A/R173真实资金覆盖；四例的所有出生/失败保留，不据金融结果选窗。",
        "no_rescue": "不改原预算、确认、失效、时期/成本，不新增量价/MACD/RV过滤、不混A、不运行等额/零费/放大账户，R173及R166/R167和旧配仓终态保持。",
        "necessary_new_tests": 0, "required_saved_checks": "全体原点预定份额、持有决定与全部买卖订单匹配；七集合分解与原risk_ok一致；原周期与各费用合计保持。",
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
        "sources": [{"path": path, "sha256": value} for path, value in sorted(paths.items())],
    }, exclusive=True)
    print("R174固定全部信号原预算、全持有决定及原买卖订单归因；不改政策或风险预算。", flush=True)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本固定预算归因已经开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结预算解释来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, risks, _ = load()
    quote = data.set_index("date")
    indices = dict(zip(data.date, data.index))
    cash_shift = np.cumsum(data.dividend.to_numpy(float))
    saved_risk = dict(zip(risks.idx.astype(int), risks.es95))
    signals = pd.read_parquet(EXPLANATION / "results/全部原点_上一确认下降结构与首次突破.parquet").set_index("date")
    qualifications = pd.read_parquet(PRIOR / "results/全部首次突破资格_当时指标成交取消及A覆盖.parquet")
    metrics = pd.read_parquet(PRIOR / "results/完整账户共同口径比较.parquet")
    origin_rows, holding_rows, order_rows, cycle_rows, scenario_rows, checks = [], [], [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = PRIOR / f"results/accounts/{period}/{cost}/{PRIMARY}"
            daily = pd.read_parquet(folder / "daily.parquet").set_index("date")
            daily["known_peak"] = np.maximum.accumulate(np.r_[200000., daily.equity.to_numpy(float)])[1:]
            decisions = pd.read_parquet(folder / "decisions.parquet").set_index("origin")
            orders = pd.read_parquet(folder / "orders.parquet")
            trades = pd.read_parquet(folder / "trades.parquet")
            local = qualifications.loc[qualifications.period.eq(period) & qualifications.cost.eq(cost)]
            native_count = 0
            for event in local.itertuples(index=False):
                origin, decision, state = event.date, decisions.loc[event.date], daily.loc[event.date]
                raw, nav, peak, es = float(quote.loc[origin, "close"]), float(state.equity), float(state.known_peak), float(decision.known_es95)
                require(int(state.shares) == int(decision.shares_before), "原点实际库存与原决定不符。")
                require(es == saved_risk[indices[origin]] and event.A_known_target == decision.source_weight,
                        "事前风险估计或A已知目标不符。")
                row = event._asdict()
                row.update(origin=origin, native_flat_request=int(state.shares) == 0,
                           execution_open=np.nan, execution_cash_adjusted_open=np.nan,
                           execution_pre_buy_cost_cap_quantity=np.nan, execution_risk_reduced_lots=np.nan,
                           execution_cash_was_binding=False, expected_actual_buy_quantity=0)
                if int(state.shares):
                    require(event.execution_status == "ALREADY_HOLDING_NO_NEW_CYCLE", "已有持仓被误记为新买入。")
                    row.update(known_nav=nav, known_peak=peak, known_drawdown=1-nav/peak,
                               known_desired_quantity=int(decision.desired_shares), known_budget_binding_mask=np.nan,
                               known_budget_binding="已有持仓，无新的入场预算", known_gap_budget_source=None,
                               actual_origin_to_buy_notional_ratio=np.nan)
                else:
                    require(not bool(state.risk_stopped) and np.isfinite(es), "合格空仓原点缺少已知可用风险状态。")
                    maximum = int(.5 * nav / raw // 100) * 100
                    budget = risk_base.limits(nav, peak, es)
                    capped = risk_base.cap_quantity(raw, budget, maximum)
                    require(capped == int(decision.desired_shares), "原入场计划份额未精确复现。")
                    row.update(budget_fields(nav, peak, es, raw, capped, maximum))
                    require(row["known_budget_binding_mask"] > 0, "最大入场份额的风险来源未解释。")
                    native_count += 1
                    next_date = decision.execution_date
                    i = indices[next_date]
                    require(i == indices[origin] + 1 and origin < next_date, "入场原点未使用真实次开时钟。")
                    opening = float(quote.loc[next_date, "open"])
                    opening_adjusted = opening + float(cash_shift[i])
                    expected, cash_binding = 0, False
                    if not open_blocked(data, i, 1) and np.isfinite(opening_adjusted) and opening_adjusted > decision.stop_index:
                        open_nav = nav + float(daily.loc[next_date, "dividend_accrual"])
                        open_budget = risk_base.limits(open_nav, peak, es)
                        open_cap = risk_base.cap_quantity(opening, open_budget, capped)
                        expected = open_cap
                        px = fill(opening, 1, cost)
                        while expected:
                            debit = expected * px + fee(expected * px, cost)
                            cash_binding = cash_binding or debit > float(state.cash) + 1e-8
                            if debit <= float(state.cash) + 1e-8 and risk_base.risk_ok(expected, opening, open_budget, buying=True):
                                break
                            expected -= 100
                        row.update(execution_pre_buy_cost_cap_quantity=open_cap,
                                   execution_risk_reduced_lots=(open_cap-expected)//100)
                    buys = orders.loc[orders.origin.eq(origin) & orders.side.eq("BUY")]
                    require(len(buys) <= 1, "新首次突破包含多个买入或加仓。")
                    actual = int(buys.quantity.iloc[0]) if len(buys) else 0
                    require(expected == actual == int(event.entry_quantity), "原次开实际买入份额未精确复现。")
                    row.update(execution_open=opening, execution_cash_adjusted_open=opening_adjusted,
                               execution_cash_was_binding=cash_binding, expected_actual_buy_quantity=expected,
                               actual_origin_to_buy_notional_ratio=actual * opening / nav)
                origin_rows.append(row)
            floors = {}
            local_holding = 0
            for origin, decision in decisions.loc[decisions.shares_before.gt(0)].iterrows():
                state, signal = daily.loc[origin], signals.loc[origin]
                q = int(state.shares)
                require(q == int(decision.shares_before), "持有预算原点与实际库存不符。")
                active = trades.loc[trades.entry_date.le(origin) & (trades.exit_date.isna() | trades.exit_date.gt(origin))]
                require(len(active) == 1, "持有原点缺少唯一原周期。")
                cycle = active.iloc[0]
                floor = floors.get(int(cycle.cycle_id), float(cycle.entry_stop_index))
                if bool(signal.structure_known) and np.isfinite(signal.latest_confirmed_low_price):
                    require(signal.L2_confirmation_index <= indices[origin], "风险线使用了未来低点确认。")
                    floor = max(floor, float(signal.latest_confirmed_low_price))
                floors[int(cycle.cycle_id)] = floor
                require(floor == float(decision.known_current_structural_stop), "实际只上移的失效线未精确复现。")
                nav, peak, raw, es = float(state.equity), float(state.known_peak), float(quote.loc[origin, "close"]), float(decision.known_es95)
                invalid = bool(signal.structure_known and signal.known_cash_adjusted_close <= floor)
                if bool(state.risk_stopped):
                    desired, reason = 0, "ACCOUNT_DRAWDOWN_STOP"
                elif invalid:
                    desired, reason = 0, "DOWNTREND_BREAK_STRUCTURAL_STOP_FAILED_CLOSE"
                elif np.isfinite(es):
                    desired = risk_base.cap_quantity(raw, risk_base.limits(nav, peak, es), q, existing=q)
                    reason = "REPAIR_ORIGINAL_RISK_REDUCTION" if desired < q else "REPAIR_KEEP_CONFIRMED_PRICE"
                else:
                    desired, reason = q, "UNKNOWN_KEEP_EXISTING"
                require(desired == int(decision.desired_shares) and reason == decision.reason,
                        "原持有风险预算或失效决定未精确复现。")
                row = {"period": period, "cost": cost, "origin": origin, "execution_date": decision.execution_date,
                       "cycle_id": int(cycle.cycle_id), "shares_before": q, "desired_shares": desired,
                       "reason": reason, "known_stop": floor, "known_cash_adjusted_close": signal.known_cash_adjusted_close,
                       "known_structure_available": bool(signal.structure_known), "known_nav": nav, "known_peak": peak,
                       "known_es95": es, "known_drawdown": 1-nav/peak, "known_budget_binding_mask": 0,
                       "known_budget_binding": MASK_LABELS[0], "actual_cycle_status": cycle.status}
                if 0 < desired < q:
                    row.update(budget_fields(nav, peak, es, raw, desired, q, existing=q))
                    require(row["known_budget_binding_mask"] > 0, "风险减仓缺少真实绑定预算。")
                holding_rows.append(row)
                local_holding += 1
            for order in orders.itertuples(index=False):
                decision, state = decisions.loc[order.origin], daily.loc[order.origin]
                require(order.date == decision.execution_date and indices[order.date] == indices[order.origin]+1,
                        "原订单未按次真实开执行。")
                require(not open_blocked(data, indices[order.date], 1 if order.side == "BUY" else -1), "原已成交订单对应开盘不可成交。")
                raw, px = float(quote.loc[order.date, "open"]), fill(float(quote.loc[order.date, "open"]), 1 if order.side == "BUY" else -1, cost)
                quantity = int(order.quantity)
                if order.side == "SELL":
                    require(quantity == int(state.shares) - int(decision.desired_shares), "原卖单份额与已知决定不符。")
                    cycle = trades.loc[trades.cycle_id.eq(order.cycle_id)].iloc[0]
                    require(order.date > cycle.entry_date, "原卖单违反T+1。")
                require(order.raw_open == raw and order.fill_price == px and abs(order.commission-fee(quantity*px, cost)) < 1e-10,
                        "原订单价格或费用未精确复现。")
                expected_slip = quantity * abs(px - raw)
                require(abs(order.slippage - expected_slip) < 1e-9, "原订单滑点金额不符。")
                kind = "BUY" if order.side == "BUY" else "RISK_REDUCTION" if order.reason == "REPAIR_ORIGINAL_RISK_REDUCTION" else "STRUCTURAL_EXIT"
                order_rows.append({**order._asdict(), "period": period, "cost": cost, "order_kind": kind,
                                   "original_quantity_verified": quantity, "friction": order.commission+order.slippage,
                                   "minimum_fee_bound": bool(order.commission == 5.)})
            for trade in trades.itertuples(index=False):
                own = orders.loc[orders.cycle_id.eq(trade.cycle_id)]
                require(int(own.loc[own.side.eq("BUY"), "quantity"].sum()) == int(trade.entry_quantity), "原周期买入数量未保持。")
                reductions = own.loc[own.reason.eq("REPAIR_ORIGINAL_RISK_REDUCTION")]
                cycle_rows.append({**trade._asdict(), "period": period, "cost": cost,
                                   "risk_reduction_orders": len(reductions),
                                   "risk_reduction_quantity": int(reductions.quantity.sum()),
                                   "risk_reduction_friction": float((reductions.commission+reductions.slippage).sum()),
                                   "all_order_friction": float((own.commission+own.slippage).sum()),
                                   "entry_actual_notional_ratio": trade.entry_quantity*trade.entry_raw/trade.entry_equity})
            primary = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost) & metrics.policy.eq(PRIMARY)].iloc[0]
            a = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost) & metrics.policy.eq("A_SAVED_WEIGHT")].iloc[0]
            friction = float((orders.commission+orders.slippage).sum())
            require(abs(friction-primary.total_commission-primary.total_slippage) < 1e-8, "原费用合计未保持。")
            scenario_rows.append({"period": period, "cost": cost, "native_flat_requests": native_count,
                                  "holding_decisions_verified": local_holding, "orders_verified": len(orders),
                                  "primary_completed_pnl": primary.completed_cycle_net_pnl,
                                  "primary_terminal_gain": primary.ending_equity-200000,
                                  "original_A_terminal_gain": a.ending_equity-200000,
                                  "all_actual_friction": friction,
                                  "primary_actual_quantity_gross_gain": primary.gross_at_actual_quantities_pnl,
                                  "gross_gain_minus_A_net_gain": primary.gross_at_actual_quantities_pnl-(a.ending_equity-200000),
                                  "primary_all_calendar_exposure": primary.mean_exposure,
                                  "A_all_calendar_exposure": a.mean_exposure,
                                  "primary_holding_days": primary.holding_days, "A_holding_days": a.holding_days,
                                  "original_financial_result_unchanged": True})
            checks.append({"period": period, "cost": cost, "qualification_rows": len(local),
                           "native_flat_origin_decisions_verified": native_count,
                           "holding_decisions_verified": local_holding, "actual_buys_verified": int(orders.side.eq("BUY").sum()),
                           "actual_sells_verified": int(orders.side.eq("SELL").sum()), "orders_verified": len(orders)})
    origins, holdings, order_frame, cycles, scenarios = map(pd.DataFrame, (origin_rows, holding_rows, order_rows, cycle_rows, scenario_rows))
    require(len(origins) == 132 and len(cycles) == 106 and origins.expected_actual_buy_quantity.gt(0).sum() == 106,
            "全体132资格或106原实际周期缺失。")
    require(origins.loc[~origins.native_flat_request, "actual_net_return"].isna().all(), "已有持仓资格被填入新的收益。")
    table("全部132资格_原预算开盘执行与真实资金", origins)
    table("全部持有原点_只上移失效及原风险减仓", holdings)
    table("全部真实买卖订单_份额费用及风险用途", order_frame)
    table("全部106原周期_投入及风险减仓费用", cycles)
    table("四场景_原A资金覆盖及真实费用差距", scenarios)
    entry_groups, reduction_groups, cost_groups = [], [], []
    for period in PERIODS:
        for cost in COSTS:
            native = origins.loc[origins.period.eq(period) & origins.cost.eq(cost) & origins.native_flat_request]
            reduced = holdings.loc[holdings.period.eq(period) & holdings.cost.eq(cost) & holdings.reason.eq("REPAIR_ORIGINAL_RISK_REDUCTION")]
            for bit in range(1, 8):
                group = native.loc[native.known_budget_binding_mask.eq(bit)]
                complete = group.loc[group.execution_status.eq("COMPLETE")]
                entry_groups.append({"period": period, "cost": cost, "binding_mask": bit, "binding": MASK_LABELS[bit],
                                     "native_origins": len(group), "actual_buys": int(group.expected_actual_buy_quantity.gt(0).sum()),
                                     "complete_cycles": len(complete), "open_cycles": int(group.execution_status.eq("RIGHT_CENSORED").sum()),
                                     "cancelled_origins": int(group.execution_status.eq("CANCELLED_NEXT_OPEN_AT_OR_BELOW_INITIAL_STOP").sum()),
                                     "mean_known_drawdown": float(group.known_drawdown.mean()),
                                     "mean_actual_entry_notional_ratio": float(group.actual_origin_to_buy_notional_ratio.mean()),
                                     "original_completed_pnl": float(complete.actual_net_pnl.sum()), "role": "DESCRIPTIVE_NOT_A_FILTER"})
                group = reduced.loc[reduced.known_budget_binding_mask.eq(bit)]
                reduction_groups.append({"period": period, "cost": cost, "binding_mask": bit, "binding": MASK_LABELS[bit],
                                         "reduction_origins": len(group), "planned_reduced_quantity": int((group.shares_before-group.desired_shares).sum()),
                                         "mean_known_drawdown": float(group.known_drawdown.mean()), "role": "KNOWN_HOLDING_CONSTRAINT_NOT_NEW_POLICY"})
            own = order_frame.loc[order_frame.period.eq(period) & order_frame.cost.eq(cost)]
            for kind in ("BUY", "RISK_REDUCTION", "STRUCTURAL_EXIT"):
                group = own.loc[own.order_kind.eq(kind)]
                cost_groups.append({"period": period, "cost": cost, "order_kind": kind, "orders": len(group),
                                    "quantity": int(group.quantity.sum()), "commission": float(group.commission.sum()),
                                    "slippage": float(group.slippage.sum()), "friction": float(group.friction.sum()),
                                    "minimum_fee_orders": int(group.minimum_fee_bound.sum())})
    table("入场原预算七集合_全部28单元", pd.DataFrame(entry_groups))
    table("持有风险减仓七集合_全部28单元", pd.DataFrame(reduction_groups))
    table("原买入风险减仓结构退出_全部12费用单元", pd.DataFrame(cost_groups))
    make_chart(data, origins)
    summary = {
        "at": now(), "study": protocol["study"], "technical_decision": "TECH.R174",
        "status": "COMPLETED_ALL_SAVED_BREAK_BUDGETS_HOLDING_AND_ORDERS_NO_POLICY_CHANGE",
        "qualification_rows": len(origins), "unique_known_origins": 66, "native_flat_origin_decisions_verified": int(origins.native_flat_request.sum()),
        "holding_decisions_verified": len(holdings), "actual_orders_verified": len(order_frame),
        "actual_buys_verified": int(order_frame.side.eq("BUY").sum()), "actual_sells_verified": int(order_frame.side.eq("SELL").sum()),
        "original_cycles": len(cycles), "completed_cycles": int(cycles.status.eq("COMPLETE").sum()),
        "open_cycles": int(cycles.status.ne("COMPLETE").sum()), "entry_group_cells": 28,
        "holding_group_cells": 28, "order_cost_cells": 12, "checks": checks,
        "stress_origin_binding_counts": origins.loc[origins.cost.eq("STRESS") & origins.native_flat_request].known_budget_binding.value_counts().to_dict(),
        "stress_gap_budget_sources": origins.loc[origins.cost.eq("STRESS") & origins.native_flat_request].known_gap_budget_source.value_counts().to_dict(),
        "frozen_sources": len(protocol["sources"]), "charts": ["原四案例_价格与真实资金覆盖.png"],
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "old_risk_budgets_preserved": True, "latest_financial_strategy_decision_preserved": "TECH.R173",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }
    write_json(OUT / "summary.json", summary, exclusive=True)
    print(f"全部132资格、{len(holdings)}原持有决定及{len(order_frame)}真实订单核对完成，28+28预算/12费用单元保持；0新账户。", flush=True)


def make_chart(data, origins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else FontProperties(family="SimHei")
    cases = pd.read_parquet(EXPLANATION / "results/四案例逐日完整量价与上一确认下降结构.parquet")
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    titles = {18: "2015失败反弹", 37: "2019慢涨及失败突破", 42: "2020修复与加速", 55: "2024急涨"}
    for row, (episode, case) in enumerate(cases.groupby("original_episode_id", sort=True)):
        period = "2015_2019" if case.date.max() < pd.Timestamp("2020-01-01") else "2020_2026"
        primary = pd.read_parquet(PRIOR / f"results/accounts/{period}/STRESS/{PRIMARY}/daily.parquet")
        from research.point_first_passage_study_v1 import CONTROL
        a = pd.read_parquet(CONTROL / period / "STRESS/A_SAVED_WEIGHT/daily.parquet")
        lo, hi = case.date.min(), case.date.max()
        primary = primary.loc[primary.date.between(lo, hi)]
        a = a.loc[a.date.between(lo, hi)]
        axes[row, 0].plot(case.date, case.close, color="#183851", linewidth=1.8, label="真实日收盘")
        events = origins.loc[origins.period.eq(period) & origins.cost.eq("STRESS") & origins.date.between(lo, hi)]
        axes[row, 0].scatter(events.date, events.close, color="#df6b24", marker="^", s=36, zorder=5, label="当时首次突破资格")
        axes[row, 1].plot(primary.date, primary.exposure, color="#df6b24", linewidth=1.6, label="突破账户实际收盘暴露")
        axes[row, 1].plot(a.date, a.exposure, color="#275776", linewidth=1.6, label="原A实际收盘暴露")
        axes[row, 1].set_ylim(-.01, .56)
        axes[row, 1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[row, 0].set_title(titles[int(episode)] + "：价格与资格", fontproperties=font)
        axes[row, 1].set_title("原固定风险预算下的真实资金覆盖", fontproperties=font)
        for axis in axes[row]:
            axis.grid(alpha=.18)
            axis.legend(prop=font, fontsize=8, loc="upper left")
            axis.tick_params(axis="x", labelrotation=25)
    fig.suptitle("原四固定案例，压力成本；全部真实资金覆盖，事后案例窗仅用于解释", fontproperties=font, fontsize=15)
    fig.savefig(OUT / "原四案例_价格与真实资金覆盖.png", dpi=150)
    plt.close(fig)


def deliver():
    require(not (OUT / "delivery_receipt.json").exists(), "原预算归因报告已经交付。")
    summary = read(OUT / "summary.json")
    origins = pd.read_parquet(OUT / "results/全部132资格_原预算开盘执行与真实资金.parquet")
    scenarios = pd.read_parquet(OUT / "results/四场景_原A资金覆盖及真实费用差距.parquet")
    costs = pd.read_parquet(OUT / "results/原买入风险减仓结构退出_全部12费用单元.parquet")
    entries = pd.read_parquet(OUT / "results/入场原预算七集合_全部28单元.parquet")
    holding = pd.read_parquet(OUT / "results/持有风险减仓七集合_全部28单元.parquet")
    cases = origins.loc[origins.cost.eq("STRESS") & origins.origin.isin(pd.to_datetime(["2019-01-18", "2019-03-18", "2020-05-29", "2024-09-24"]))]
    table("固定案例突破_事前风险预算与实际份额", cases)
    lines = ["# 正交易期望为何没有转化为更高账户收益与夏普", "",
             "本轮只解释R173已保存的全部信号、实际资金与订单。四场景交易质量为正的事实和完整账户拒绝保持；没有改策略、风险预算、费用或运行新账户。", "",
             f"132资格、{summary['native_flat_origin_decisions_verified']}空仓原预算、{summary['holding_decisions_verified']}持有决定及{summary['actual_orders_verified']}买卖订单精确复算。106原周期（104完成/2开放）、所有次开取消/已有持仓保留。0新增账户/拟合/训练标签/行情采集；28入场、28减仓及12费用单元全部报告，空组不填收益。", "",
             "## 入场资金受什么约束", "",
             "初始意向50%，最终按原ES预算2.5%NAV、负10%跳空预算min(5%NAV,半剩余10%回撤余量)及平仓摩擦逐手减少。跳空预算随自身历史回撤而收缩，金额不能依据以后涨跌调整。约束可同时存在，表中保留全部失败项集合。", "",
             "| 时期/成本 | 仓位绑定原点 | ES绑定原点 | 跳空绑定原点 | ES+跳空等多项绑定 |", "|---|---:|---:|---:|---:|"]
    for period in PERIODS:
        for cost in COSTS:
            own = entries.loc[entries.period.eq(period) & entries.cost.eq(cost)].set_index("binding_mask")
            lines.append(f"| {period}/{cost} | {int(own.loc[1,'native_origins'])} | {int(own.loc[2,'native_origins'])} | {int(own.loc[4,'native_origins'])} | {int(own.loc[[3,5,6,7],'native_origins'].sum())} |")
    lines.extend(["", "| 固定突破信号 | 当时账户回撤 | 已知5日ES | 原计划份额 | 实际买入份额 | 入场实际金额/NAV | 原点全部绑定项 |", "|---|---:|---:|---:|---:|---:|---|"])
    for r in cases.itertuples(index=False):
        lines.append(f"| {r.origin:%Y-%m-%d} | {r.known_drawdown:.2%} | {r.known_es95:.2%} | {r.known_desired_quantity} | {r.entry_quantity} | {r.actual_origin_to_buy_notional_ratio:.2%} | {r.known_budget_binding} |")
    lines.extend(["", "2019及2024点位的投入差异必须看各自当时NAV/峰值/回撤余量和ES；不把后来赢家统一放大，不把绑定类别事后变成入场过滤。原A库存与当时目标仍分开保留。", "",
                  "## 持有减仓和交易摩擦", "",
                  "所有持有原点的只上移确认低点和原风险份额已核对。价格上行或ES/回撤预算变化都可能要求减仓；已确认失效才是结构退出。只有实际风险减仓费用归入这一类，没有假设保留份额直到终点。", "",
                  "| 时期/成本 | 买入摩擦/元 | 风险减仓次数/摩擦元 | 结构退出摩擦/元 | 全部摩擦/元 |", "|---|---:|---:|---:|---:|"])
    for r in scenarios.itertuples(index=False):
        own = costs.loc[costs.period.eq(r.period) & costs.cost.eq(r.cost)].set_index("order_kind")
        lines.append(f"| {r.period}/{r.cost} | {own.loc['BUY','friction']:.2f} | {int(own.loc['RISK_REDUCTION','orders'])}/{own.loc['RISK_REDUCTION','friction']:.2f} | {own.loc['STRUCTURAL_EXIT','friction']:.2f} | {r.all_actual_friction:.2f} |")
    lines.extend(["", "风险减仓摩擦不是全部手续费。近期压力18次减仓合计118.50元，占全部5227.61元约2.27%；单看这些小额减仓的最低佣金，无法解释原A与突破账户的大幅收益差距。是否少减仓还涉及违反原风险预算和之后价格路径，本轮没有计算另一账户。", "",
                  "## 资金覆盖和利润缺口", "",
                  "下表全部来自原保存账本：毛损益采用实际发生份额和价格，并保留股息/期末持仓。将它和A净增额并列只判断手续费是否足以解释差额，不产生零费策略或零费夏普。", "",
                  "| 时期/成本 | 突破账户净增/元 | 实际份额毛增/元 | 原A净增/元 | 毛增−A净增/元 | 全日历暴露：突破/A | 持有日：突破/A |", "|---|---:|---:|---:|---:|---:|---:|"])
    for r in scenarios.itertuples(index=False):
        lines.append(f"| {r.period}/{r.cost} | {r.primary_terminal_gain:.2f} | {r.primary_actual_quantity_gross_gain:.2f} | {r.original_A_terminal_gain:.2f} | {r.gross_gain_minus_A_net_gain:.2f} | {r.primary_all_calendar_exposure:.2%}/{r.A_all_calendar_exposure:.2%} | {r.primary_holding_days}/{r.A_holding_days} |")
    lines.extend(["", "近期实际份额毛盈利已经远小于原A净盈利，交易成本不能单独解释差距；资金覆盖、入场选择和整个持有路径也不同。现金日造成低暴露，但机械放大仓位会同时放大亏损和风险约束，不能从这张表推出更高夏普。R173已完成周期资金协方差恒等式保持，不把平均回报项包装成等额账户。", "",
                  "![原四案例价格与真实资金覆盖](原四案例_价格与真实资金覆盖.png)", "",
                  "## 接受、拒绝及后续入口", "",
                  "接受全部原预算来源、真实份额和费用拆解事实。排除将R173近期不足简单归因为最低佣金小额减仓，或认为只要省手续费就可追平A的解释。入场资本受已知ES和剩余回撤约束的事实保留，但不因此放宽预算；本轮没有改善收益/夏普，最新金融结果仍R173。", "",
                  "R173固定完整政策、原A、R166/R167及全部旧波动配仓/半仓/自身盈利过滤终态保持。日成交均价偏离的有限旧核对也找到了ETF_CLOSE_VWAP同表达以及20/60滚动量加权、新低锚定的旧用途；不能把它们当作新字段直接重跑。此处只定位旧事实，不从源码冒称新的准入或当前收益。", "",
                  "下一入口须是实质不同的点位信息/完整用途、真实新样本，或真实来源/实现错误。当前没有已准入待跑新金融策略，不追加本失败规则的过滤/窗口/退出/成本/预算搜索。已登记真实A/POINT前瞻十二值及1008实际交易日口径保持，不能以历史复用或尚未到来的新日期代替独立验证。", "",
                  "全部历史DEVELOPMENT_CALIBRATION，first-vintage未认证，独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，去过拟合与完整收益夏普目标未达，目标继续active。本轮是新的保存账本诊断证据，不能称策略晋升或全项目停止。", "",
                  "- [全部资格及原预算/真实执行](results/全部132资格_原预算开盘执行与真实资金.csv)",
                  "- [全部持有决定](results/全部持有原点_只上移失效及原风险减仓.csv)、[全部真实订单](results/全部真实买卖订单_份额费用及风险用途.csv)",
                  "- [全部原周期](results/全部106原周期_投入及风险减仓费用.csv)、[四场景原A覆盖和费用](results/四场景_原A资金覆盖及真实费用差距.csv)",
                  "- [28入场单元](results/入场原预算七集合_全部28单元.csv)、[28减仓单元](results/持有风险减仓七集合_全部28单元.csv)、[12费用单元](results/原买入风险减仓结构退出_全部12费用单元.csv)",
                  "- [固定分解定义](protocol.json)、[实际核对结果](summary.json)、[原R173完整金融结果](../510300_downtrend_break_study_v1/summary.json)", ""])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_SAVED_BUDGETS_ORDERS_AND_FOUR_CASE_COVERAGE_DELIVERED",
        "technical_decision": "TECH.R174", "new_accounts": 0, "report": report.relative_to(ROOT).as_posix(),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                    [Path(__file__), OUT / "summary.json", report, OUT / "原四案例_价格与真实资金覆盖.png"]],
        "goal_achieved": False,
    }, exclusive=True)
    print("全部原预算/订单、四案例资金图及诊断报告已交付；没有改变金融结果。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结并一次解释全部原预算与真实订单，不运行账户。")
    parser.add_argument("command", choices=("freeze", "run", "deliver"))
    command = parser.parse_args().command
    {"freeze": freeze, "run": run, "deliver": deliver}[command]()
