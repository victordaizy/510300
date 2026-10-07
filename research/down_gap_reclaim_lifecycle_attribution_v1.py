"""只读R185保存周期，解释最新缺口收复的覆盖、固定失效和真实资金路径。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.down_gap_reclaim_study_v1 import OUT as FINANCIAL, EXPLANATION, PRIMARY, PERIODS, COSTS
from research.down_gap_reclaim_explanation_v1 import KNOWN_TABLE, MAP_TABLE, CASE_TABLE
from research.full_range_reversal_lifecycle_attribution_v1 import dividend_entitlements, classify
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_first_passage_study_v1 import read, write_json, digest, now, require, WEIGHT

OUT = ROOT / "reports/research/510300_down_gap_reclaim_lifecycle_attribution_v1"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
CASE_IDS = (18, 37, 42, 55)
LEDGERS = ("daily", "orders", "trades", "decisions", "rejections")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "收复固定生命周期归因已登记。")
    require(read(STATE)["latest_technical_decision"] == "TECH.R185", "当前不再是R185，不覆盖其他工作。")
    prior = read(FINANCIAL / "summary.json")
    delivery = read(FINANCIAL / "delivery_receipt.json")
    require(prior["technical_decision"] == "TECH.R185" and prior["new_primary_accounts"] == 4
            and delivery["all_actual_cycle_rows"] == 102, "全体四保存账户尚未完成交付。")
    for parent in (FINANCIAL, EXPLANATION):
        for source in read(parent / "protocol.json")["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "原冻结来源改变。")
    paths = [Path(__file__), ROOT / "research/down_gap_reclaim_account_v1.py", ROOT / "research/down_gap_reclaim_inputs_v1.py",
             ROOT / "research/full_range_reversal_lifecycle_attribution_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/point_first_passage_study_v1.py", WEIGHT / "inputs/dividends.csv",
             FINANCIAL / "protocol.json", FINANCIAL / "summary.json", FINANCIAL / "delivery_receipt.json",
             FINANCIAL / "saved_result_verification.json", FINANCIAL / "next_all_reclaim_lifecycle_diagnostic_proposal.json",
             FINANCIAL / "results/完整账户共同口径比较.parquet",
             FINANCIAL / "results/全部收复资格_原锚当时量价及执行状态.parquet",
             FINANCIAL / "results/全部实际进出点位_已知收复量价及真实资金.parquet",
             EXPLANATION / "protocol.json", EXPLANATION / "summary.json",
             EXPLANATION / f"results/{KNOWN_TABLE}.parquet", EXPLANATION / f"results/{MAP_TABLE}.parquet",
             EXPLANATION / f"results/{CASE_TABLE}.parquet",
             EXPLANATION / "results/全部向下缺口原锚_收复替换及开放不筛选.parquet",
             EXPLANATION / "results/全部首次收复至固定线失败或新缺口_仅路径.parquet"]
    for period in PERIODS:
        for cost in COSTS:
            folder = FINANCIAL / f"results/accounts/{period}/{cost}/{PRIMARY}"
            paths.extend(folder / f"{name}.parquet" for name in LEDGERS)
            paths.append(folder / "terminal.json")
    paths = list(dict.fromkeys(paths))
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_DOWN_GAP_RECLAIM_SAVED_LIFECYCLE_ATTRIBUTION_V1", "technical_decision": "TECH.R186",
        "hypothesis": "具体上涨中收复的早晚、固定下沿/新向下缺口失效、实际资金和退出时钟，可能解释近期点位期望正但全账户仍低A、早期负的结果。",
        "population": "全部60当期首次收复、120资格、四保存主账户/102周期（100完成/2开放）、全部9每费用开盘取消、112原锚和原61/49段；失败和开放不删。",
        "known_clock": "按真实持有原点匹配固定原锚下沿、库存、已知CLOSE<=线或新完整向下缺口、原风险只减及锁定退出；固定线不能上移。首个已知退出请求与最终卖出前原点分别记录。",
        "retrospective_alignment": "全部首次收复与所有原底峰区间相交关系逐一保存，无匹配保留未知；两原区间底至峰/确认至峰同时报告，未来阶段和峰值不可做交易门。",
        "wealth_identity": "周期财富=真实SELL净现金−真实BUY支出+登记库存已发生股息应收+真实剩余份额×日收盘，逐日匹配库存；最大持有收盘财富仅事后解释，不是可实现止盈。",
        "exit_identity": "完成净利润−最终卖出前原点财富=该原点剩余份额×(真实卖出原开盘−原点收盘)+其后已发生周期股息应收−最终佣金−滑点；全100退出核对，不生成收盘退出账户或任意新退出上界。",
        "concentration_and_groups": "四场景全完成净利润、正利润、最大赢家及占正利润/正净利润比；净利润非正时净占比未知。最高收盘财富正/非正×最终赢/零/亏六组×四场景全报告，开放单列，不按组优化。",
        "source_clock": "仅R185保存结果及原正常化分红，不采集、不拟合、不运行新账户或新标签；R185及所有旧终态不营救。",
        "charts": "四原案例全部相交压力周期，窗扩至自身入出/观察末日，真实买卖及已知固定线、失效、财富和当时量/日MACD/上一完整周/RV；案例不改，开放不得越过自身账户末日。",
        "next_limit": "本固定全体归因一次完成，之后不同信息+完整用途/真实新样本/来源实现错误才可准入新金融；不改最新锚/消耗/退出/量MACD/RV/费用预算或混A。",
        "necessary_new_tests": 0, "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False, "goal_achieved": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print("R186全体收复固定失效和真实资金归因已冻结；0新账户。", flush=True)


def alignments_and_coverage(known, episodes):
    population = known.loc[known.date.ge("2015-01-01")]
    aligned, covered = [], []
    for event in population.loc[population.reclaim_event].itertuples(index=False):
        matched = episodes.loc[episodes.bottom_idx.le(event.origin_index) & episodes.peak_idx.ge(event.origin_index)]
        base = {"reclaim_date": event.date, "reclaim_index": int(event.origin_index), "anchor_id": str(event.reclaim_anchor_id),
                "gap_birth_index": int(event.reclaim_reference_index), "gap_age_sessions": int(event.reclaim_age_sessions),
                "retrospective_only_not_a_decision_field": True}
        if not len(matched):
            aligned.append({**base, "episode_id": -1, "alignment_status": "NO_RETROSPECTIVE_ASCENT_AT_ORIGIN",
                            "sessions_since_bottom": np.nan, "sessions_until_peak": np.nan,
                            "fraction_ascent_elapsed": np.nan, "episode_admitted": None, "after_original_confirmation": None})
        for ep in matched.itertuples(index=False):
            length = int(ep.peak_idx-ep.bottom_idx)
            aligned.append({**base, "episode_id": int(ep.episode_id), "alignment_status": "ALL_MATCHING_ORIGINAL_ASCENTS",
                            "sessions_since_bottom": int(event.origin_index-ep.bottom_idx),
                            "sessions_until_peak": int(ep.peak_idx-event.origin_index),
                            "fraction_ascent_elapsed": (event.origin_index-ep.bottom_idx)/length if length else np.nan,
                            "episode_admitted": bool(ep.admitted), "after_original_confirmation": bool(event.date >= ep.confirm_up_date)})
    for ep in episodes.itertuples(index=False):
        bottom = population.loc[population.origin_index.between(ep.bottom_idx, ep.peak_idx)]
        confirmation = population.loc[population.origin_index.between(ep.confirm_up_idx, ep.peak_idx)]
        require(int(bottom.reclaim_event.sum()) == int(ep.reclaims_bottom_to_peak)
                and int(confirmation.reclaim_event.sum()) == int(ep.reclaims_confirm_to_peak), "原两区间覆盖未还原R183。")
        events = bottom.loc[bottom.reclaim_event]
        covered.append({**ep._asdict(), "bottom_to_peak_reclaims": len(events) if len(bottom) else np.nan,
                        "confirmation_to_peak_reclaims": int(confirmation.reclaim_event.sum()) if len(confirmation) else np.nan,
                        "first_reclaim_date": events.date.iloc[0] if len(events) else pd.NaT,
                        "first_reclaim_sessions_since_bottom": int(events.origin_index.iloc[0]-ep.bottom_idx) if len(events) else np.nan,
                        "bottom_peak_and_confirmation_only_retrospective": True})
    return pd.DataFrame(aligned), pd.DataFrame(covered)


def saved_cycle(cycle, account, quotes, dividends, period, cost):
    local_orders = account["orders"].loc[account["orders"].cycle_id.eq(cycle.cycle_id)].sort_values("date")
    daily, decisions = account["daily"], account["decisions"]
    observed_end = cycle.exit_date if cycle.status == "COMPLETE" else daily.date.max()
    selected = daily.loc[daily.date.between(cycle.entry_date, observed_end)]
    entitlements = dividend_entitlements(local_orders, dividends, daily.date.max())
    require(abs(sum(row["dividend_cny"] for row in entitlements)-float(cycle.dividend_cny)) <= 1e-6, "周期股息与原登记份额不符。")
    require(int(local_orders.loc[local_orders.side.eq("BUY"), "quantity"].sum()) == int(cycle.entry_quantity), "出现加仓或原入场份额不同。")
    floor = int(cycle.entry_reclaim_floor_ticks)
    require(int(cycle.fixed_reclaim_floor_ticks) == floor, "原固定线改变。")
    entry = quotes.loc[cycle.entry_origin]
    parent = quotes.iloc[int(entry.reclaim_reference_index)]
    require(bool(entry.reclaim_event) and bool(parent.new_down_gap)
            and str(parent.observed_anchor_id) == str(entry.reclaim_anchor_id), "真实入点没有此前最新缺口首次收复。")
    point_rows, cash_rows, first_request, reductions = [], [], None, 0
    for day in selected.itertuples(index=False):
        until = local_orders.loc[local_orders.date.le(day.date)]
        buys, sells = until.loc[until.side.eq("BUY")], until.loc[until.side.eq("SELL")]
        quantity = int(buys.quantity.sum()-sells.quantity.sum())
        cash_delta = float((sells.quantity*sells.fill_price-sells.commission).sum()-(buys.quantity*buys.fill_price+buys.commission).sum())
        accrued = float(sum(row["dividend_cny"] for row in entitlements if row["ex_date"] <= day.date))
        marked = cash_delta+accrued+quantity*float(day.close)
        require(quantity == int(day.shares), "保存周期库存与日账不同。")
        cash_rows.append({"period": period, "cost": cost, "cycle_id": int(cycle.cycle_id), "date": day.date,
                          "actual_shares": quantity, "actual_close": float(day.close), "cash_flow_cny": cash_delta,
                          "accrued_cycle_dividend_cny": accrued, "cycle_marked_pnl_cny": marked,
                          "holding_at_close": quantity > 0, "posthoc_wealth_not_executable_exit": True})
        if quantity == 0:
            continue
        event = quotes.loc[day.date]
        original = decisions.loc[decisions.origin.eq(day.date)]
        require(len(original) == 1, "持有原点没有唯一决定。")
        decision = original.iloc[0]
        require(int(decision.shares_before) == quantity and int(decision.known_current_reclaim_floor_ticks) == floor,
                "保存固定线或库存不同。")
        close_failed = bool(event.current_quote_known and int(event.known_cash_close_ticks) <= floor)
        new_gap = bool(event.current_quote_known and event.new_down_gap)
        if decision.reason == "DOWN_GAP_RECLAIM_FAILED_KNOWN_CLOSE":
            require(close_failed and int(decision.desired_shares) == 0, "固定线失效或优先顺序不同。")
        elif decision.reason == "DOWN_GAP_RECLAIM_NEW_DOWN_GAP":
            require(new_gap and not close_failed and int(decision.desired_shares) == 0, "新缺口退出或优先顺序不同。")
        require(int(decision.desired_shares) <= quantity, "收复账户持有中加仓。")
        reductions += int(0 < int(decision.desired_shares) < quantity)
        if int(decision.desired_shares) == 0 and first_request is None:
            first_request = day.date
        point_rows.append({"period": period, "cost": cost, "cycle_id": int(cycle.cycle_id), "origin": day.date,
                           "entry_origin": cycle.entry_origin, "shares_before": quantity, "fixed_floor_ticks": floor,
                           "quote_known": bool(event.current_quote_known), "known_close_ticks": int(event.known_cash_close_ticks),
                           "known_new_down_gap": new_gap, "close_failed_fixed_floor": close_failed,
                           "decision_reason": decision.reason, "desired_shares": int(decision.desired_shares),
                           "planned_next_open": decision.execution_date, "actual_cycle_marked_pnl_cny": marked})
    marks = pd.DataFrame(cash_rows)
    held = marks.loc[marks.holding_at_close]
    require(len(held) > 0, "真实周期没有持有收盘。")
    peak = held.loc[held.cycle_marked_pnl_cny.idxmax()]
    row = {"period": period, "cost": cost, "cycle_id": int(cycle.cycle_id), "entry_origin": cycle.entry_origin,
           "entry_date": cycle.entry_date, "exit_date": cycle.exit_date, "status": cycle.status,
           "cycle_observed_until": observed_end, "entry_raw": float(cycle.entry_raw), "entry_fill": float(cycle.entry_price),
           "entry_quantity": int(cycle.entry_quantity), "actual_buy_debit": float(cycle.buy_debit), "fixed_floor_ticks": floor,
           "gap_birth_date": parent.date, "gap_birth_index": int(entry.reclaim_reference_index),
           "reclaim_age_sessions": int(entry.reclaim_age_sessions), "anchor_id": str(entry.reclaim_anchor_id),
           "holding_close_rows": len(held), "risk_reduction_origins": reductions, "first_known_exit_request_origin": first_request,
           "best_observed_holding_close_date": peak.date, "best_observed_holding_close_pnl_cny": float(peak.cycle_marked_pnl_cny),
           "best_observed_holding_close_shares": int(peak.actual_shares), "actual_net_pnl": cycle.net_pnl,
           "actual_net_return": cycle.net_return, "final_result_group": "RIGHT_CENSORED_NOT_A_COMPLETED_GROUP",
           "final_exit_reason": None, "final_exit_origin": pd.NaT, "exit_origin_marked_pnl_cny": np.nan,
           "final_open_gap_gross_cny": np.nan, "dividend_accrual_after_final_origin_cny": np.nan,
           "final_sell_commission": np.nan, "final_sell_slippage": np.nan, "exit_clock_identity_error_cny": np.nan,
           "decline_from_best_holding_close_to_actual_net_pnl_cny": np.nan, "loss_with_previously_positive_holding_close": None,
           "exit_on_new_gap_above_fixed_floor": None, "best_close_is_posthoc_not_a_decision_field": True}
    if cycle.status == "COMPLETE":
        require(first_request is not None, "完成周期没有原已知退出请求。")
        final = local_orders.loc[local_orders.side.eq("SELL")].iloc[-1]
        before = marks.loc[marks.date.eq(final.origin)]
        require(len(before) == 1, "最后卖出原点没有财富记录。")
        before = before.iloc[0]
        require(int(before.actual_shares) == int(final.quantity), "最后卖出不是原真实余量。")
        opening_delta = int(final.quantity)*(float(final.raw_open)-float(before.actual_close))
        later_dividend = float(cycle.dividend_cny)-float(before.accrued_cycle_dividend_cny)
        residual = float(cycle.net_pnl)-float(before.cycle_marked_pnl_cny)-opening_delta-later_dividend+float(final.commission)+float(final.slippage)
        require(abs(residual) <= 1e-6, "最后次开身份不能还原实际净利润。")
        row.update(final_result_group=classify(float(peak.cycle_marked_pnl_cny), float(cycle.net_pnl)),
                   final_exit_reason=final.reason, final_exit_origin=final.origin,
                   exit_origin_marked_pnl_cny=float(before.cycle_marked_pnl_cny), final_open_gap_gross_cny=opening_delta,
                   dividend_accrual_after_final_origin_cny=later_dividend, final_sell_commission=float(final.commission),
                   final_sell_slippage=float(final.slippage), exit_clock_identity_error_cny=residual,
                   decline_from_best_holding_close_to_actual_net_pnl_cny=float(peak.cycle_marked_pnl_cny)-float(cycle.net_pnl),
                   loss_with_previously_positive_holding_close=bool(peak.cycle_marked_pnl_cny > 0 and cycle.net_pnl < 0),
                   exit_on_new_gap_above_fixed_floor=bool(final.reason == "DOWN_GAP_RECLAIM_NEW_DOWN_GAP"))
        require(abs(float(marks.cycle_marked_pnl_cny.iloc[-1])-float(cycle.net_pnl)) <= 1e-6, "退出日周期财富与净利润不符。")
    return row, point_rows, cash_rows


def groups(cycles):
    rows = []
    for period in PERIODS:
        for cost in COSTS:
            for peak_group in ("POSITIVE_HOLDING_CLOSE", "NONPOSITIVE_HOLDING_CLOSE"):
                for result in ("WIN", "ZERO", "LOSS"):
                    group = peak_group+"_FINAL_"+result
                    selected = cycles.loc[cycles.period.eq(period) & cycles.cost.eq(cost) & cycles.final_result_group.eq(group)]
                    rows.append({"period": period, "cost": cost, "retrospective_group": group, "completed_cycles": len(selected),
                                 "actual_net_pnl_cny": float(selected.actual_net_pnl.sum()),
                                 "mean_actual_net_return": float(selected.actual_net_return.mean()),
                                 "final_open_gap_gross_cny": float(selected.final_open_gap_gross_cny.sum()),
                                 "final_sell_friction_cny": float((selected.final_sell_commission+selected.final_sell_slippage).sum()),
                                 "posthoc_only_not_a_filter_or_new_strategy": True})
    return pd.DataFrame(rows)


def charts(known, cases, cycles, holding, wealth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths, case_rows = [], []
    for case_id in CASE_IDS:
        case = cases.loc[cases.original_episode_id.eq(case_id)]
        lo, hi = case.date.min(), case.date.max()
        stress = cycles.loc[cycles.cost.eq("STRESS") & cycles.entry_date.le(hi) & cycles.cycle_observed_until.ge(lo)]
        lower = min(lo, stress.entry_date.min()) if len(stress) else lo
        upper = max(hi, stress.cycle_observed_until.max()) if len(stress) else hi
        frame = known.loc[known.date.between(lower, upper)]
        fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1, 1]})
        price, pnl, macd, volume = axes
        price.plot(frame.date, frame.close, color="#29475c", label="原价收盘")
        price.fill_between(frame.date, frame.low, frame.high, color="#b0c1ce", alpha=.3, label="整日高低区间")
        price.axvspan(lo, hi, color="#cfd5d9", alpha=.15, label="原固定案例窗")
        for cycle in stress.itertuples(index=False):
            h = holding.loc[holding.period.eq(cycle.period) & holding.cost.eq("STRESS") & holding.cycle_id.eq(cycle.cycle_id)]
            w = wealth.loc[wealth.period.eq(cycle.period) & wealth.cost.eq("STRESS") & wealth.cycle_id.eq(cycle.cycle_id)]
            shift = known.set_index("date").cash_shift.reindex(h.origin).to_numpy(float)
            price.step(h.origin, h.fixed_floor_ticks*.001-shift, where="post", color="#be6b38", label="当时原锚固定线（原价映射）")
            price.scatter([cycle.entry_date], [cycle.entry_raw], marker="^", color="#1f8b78", s=55)
            if cycle.status == "COMPLETE":
                price.scatter([cycle.exit_date], [known.loc[known.date.eq(cycle.exit_date)].open.iloc[0]], marker="v", color="#af3555", s=55)
                at_request = known.loc[known.date.eq(cycle.first_known_exit_request_origin)].iloc[0]
                price.scatter([at_request.date], [at_request.close], marker="x", color="#8e3f6d", s=55)
            pnl.plot(w.date, w.cycle_marked_pnl_cny, label=f"{cycle.entry_origin:%Y-%m-%d}实际周期")
            case_rows.append({"original_episode_id": case_id, **cycle._asdict(), "original_case_window_start": lo,
                              "original_case_window_end": hi, "figure_observed_start": lower, "figure_observed_end": upper,
                              "retrospective_intersection_only": True})
        for ax in (price, pnl):
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
        pnl.axhline(0, color="#666", linewidth=.7)
        macd.plot(frame.date, frame.daily_dif, label="日DIF", color="#bf8629")
        macd.bar(frame.date, frame.daily_hist, color=np.where(frame.daily_hist.ge(0), "#ae5471", "#488f7f"), alpha=.35, label="日MACD柱")
        macd.plot(frame.date, frame.weekly_hist, label="上一完整周柱", color="#29475c")
        macd.axhline(0, color="#777", linewidth=.6)
        macd.legend(fontsize=8)
        volume.bar(frame.date, frame.relative_volume, color="#7894a5", alpha=.65, label="相对量")
        volume.plot(frame.date, frame.rv_ratio, color="#bd8129", label="RV比")
        volume.legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.16)
        price.set_ylabel("价格/元")
        pnl.set_ylabel("实际周期财富/元")
        macd.set_ylabel("动量原值")
        volume.set_ylabel("量及波动率比")
        fig.suptitle(f"原案例{case_id}：向下缺口收复、固定线与真实资金\n三角为真实开盘买卖，叉为首个已知退出请求；最高财富仅事后观察", fontsize=15)
        fig.autofmt_xdate()
        fig.tight_layout(rect=(0, 0, 1, .94))
        path = OUT / f"案例{case_id}_真实收复生命周期与资金.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path.relative_to(ROOT).as_posix())
    table("四原案例全部相交压力周期_真实时钟与资金", pd.DataFrame(case_rows))
    return paths


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "固定归因已开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "归因冻结来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    known = pd.read_parquet(EXPLANATION / f"results/{KNOWN_TABLE}.parquet")
    quotes = known.set_index("date")
    require(quotes.index.is_unique, "保存原点日期不唯一。")
    episodes = pd.read_parquet(EXPLANATION / f"results/{MAP_TABLE}.parquet")
    cases = pd.read_parquet(EXPLANATION / f"results/{CASE_TABLE}.parquet")
    dividends = normalize_dividends(pd.read_csv(WEIGHT / "inputs/dividends.csv"))
    aligned, coverage = alignments_and_coverage(known, episodes)
    require(aligned.reclaim_index.nunique() == 60 and len(coverage) == 61 and int(coverage.admitted.sum()) == 49, "原事件或分段总体不同。")
    cycle_rows, holding_rows, wealth_rows, scenarios = [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = FINANCIAL / f"results/accounts/{period}/{cost}/{PRIMARY}"
            account = {name: pd.read_parquet(folder / f"{name}.parquet") for name in LEDGERS}
            for cycle in account["trades"].itertuples(index=False):
                row, holding, wealth = saved_cycle(cycle, account, quotes, dividends, period, cost)
                cycle_rows.append(row)
                holding_rows.extend(holding)
                wealth_rows.extend(wealth)
            scenarios.append({"period": period, "cost": cost, "cycles": len(account["trades"]),
                              "completed": int(account["trades"].status.eq("COMPLETE").sum()),
                              "open": int(account["trades"].status.ne("COMPLETE").sum()), "original_orders": len(account["orders"]),
                              "original_decisions": len(account["decisions"]), "all_original_rejections": len(account["rejections"])})
    cycles, holding, wealth = pd.DataFrame(cycle_rows), pd.DataFrame(holding_rows), pd.DataFrame(wealth_rows)
    require(len(cycles) == 102 and int(cycles.status.eq("COMPLETE").sum()) == 100, "全部保存周期数量不同。")
    require(cycles.loc[cycles.status.ne("COMPLETE"), "actual_net_return"].isna().all(), "开放周期被赋予完成收益。")
    table("全部60首次收复_原上涨位置仅事后对齐", aligned)
    table("原61分段及49上涨_两种原区间覆盖不筛选", coverage)
    table("全部102周期_原锚固定线失效及真实资金时钟", cycles)
    table("全部持有原点_固定线新缺口及原决定核对", holding)
    table("全部周期逐日库存现金股息及收盘财富", wealth)
    grouped = groups(cycles)
    table("全部六组两时期两费用_最高财富与最终结果仅解释", grouped)
    figures = charts(known, cases, cycles, holding, wealth)
    complete = cycles.loc[cycles.status.eq("COMPLETE")]
    pressure = complete.loc[complete.cost.eq("STRESS")]
    diagnostic_rows = []
    for scenario in scenarios:
        selected = complete.loc[complete.period.eq(scenario["period"]) & complete.cost.eq(scenario["cost"])]
        winners = selected.loc[selected.actual_net_pnl.gt(0)]
        largest = winners.loc[winners.actual_net_pnl.idxmax()] if len(winners) else None
        net, positive = float(selected.actual_net_pnl.sum()), float(winners.actual_net_pnl.sum())
        diagnostic_rows.append({**scenario, "completed_positive_close_then_loss": int(selected.loss_with_previously_positive_holding_close.astype(bool).sum()),
                                "risk_reduction_origins": int(selected.risk_reduction_origins.sum()),
                                "final_open_gap_gross_cny": float(selected.final_open_gap_gross_cny.sum()),
                                "post_final_origin_dividend_cny": float(selected.dividend_accrual_after_final_origin_cny.sum()),
                                "final_exit_friction_cny": float((selected.final_sell_commission+selected.final_sell_slippage).sum()),
                                "completed_pnl_cny": net, "close_floor_exits": int(selected.final_exit_reason.eq("DOWN_GAP_RECLAIM_FAILED_KNOWN_CLOSE").sum()),
                                "new_down_gap_exits": int(selected.final_exit_reason.eq("DOWN_GAP_RECLAIM_NEW_DOWN_GAP").sum()),
                                "all_final_exit_reason_counts": selected.final_exit_reason.value_counts().to_dict(),
                                "positive_close_final_win": int(selected.final_result_group.eq("POSITIVE_HOLDING_CLOSE_FINAL_WIN").sum()),
                                "never_positive_close_final_loss": int(selected.final_result_group.eq("NONPOSITIVE_HOLDING_CLOSE_FINAL_LOSS").sum()),
                                "largest_positive_cycle_origin": largest.entry_origin if largest is not None else None,
                                "largest_positive_cycle_pnl_cny": float(largest.actual_net_pnl) if largest is not None else np.nan,
                                "largest_positive_share_of_positive_pnl": float(largest.actual_net_pnl)/positive if largest is not None else np.nan,
                                "largest_positive_share_of_net_pnl": float(largest.actual_net_pnl)/net if largest is not None and net > 0 else np.nan,
                                "exit_identity_max_abs_error": float(selected.exit_clock_identity_error_cny.abs().max())})
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "诊断运行期间来源改变。")
    write_json(OUT / "summary.json", {
        "at": now(), "technical_decision": "TECH.R186", "status": "COMPLETED_ALL_SAVED_RECLAIM_COVERAGE_FIXED_FAILURE_AND_WEALTH_IDENTITIES",
        "all_first_reclaims": 60, "episode_alignment_rows": len(aligned), "unmatched_reclaims_preserved": int(aligned.episode_id.eq(-1).sum()),
        "original_episode_rows": len(coverage), "original_admitted_waves": int(coverage.admitted.sum()),
        "admitted_without_bottom_to_peak_reclaim": int((coverage.admitted & coverage.bottom_to_peak_reclaims.eq(0)).sum()),
        "admitted_without_confirmation_to_peak_reclaim": int((coverage.admitted & coverage.confirmation_to_peak_reclaims.eq(0)).sum()),
        "all_actual_cycle_rows": len(cycles), "completed_cycles": len(complete), "open_cycles": int(cycles.status.ne("COMPLETE").sum()),
        "holding_origins_verified": len(holding), "cycle_close_wealth_rows": len(wealth), "complete_exit_identities_verified": len(complete),
        "retrospective_group_cells": len(grouped), "all_scenario_diagnostics": diagnostic_rows, "charts": figures,
        "pressure_complete_cycles": len(pressure), "pressure_positive_holding_close_then_loss": int(pressure.loss_with_previously_positive_holding_close.astype(bool).sum()),
        "new_accounts": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "necessary_new_tests": 0,
        "frozen_sources": len(protocol["sources"]), "latest_financial_strategy_decision_preserved": "TECH.R185",
        "historical_sample_role": "DEVELOPMENT_CALIBRATION", "source_first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED", "goal_achieved": False,
    }, exclusive=True)
    print(f"R186全60收复、102周期、{len(holding)}持有原点及100退出财富身份完成；0新账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全部保存收复周期及原上涨覆盖的固定归因。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
