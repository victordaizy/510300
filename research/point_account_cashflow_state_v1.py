"""隔离研究：按已发生成交重建账户周期状态，不改写原策略或生成订单。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_nr7_inputs_v1 import fee, fill


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_account_cashflow_state_v1"
BASE = Path("reports/research/510300_point_second_weight_comparison_v1/inputs")
E01 = Path("reports/research/510300_point_exit_target_transmission_diagnostic_v1")
STUDY = "510300_POINT_ACCOUNT_CASHFLOW_STATE_V1"
PERIODS = ("2015_2019", "2020_2026")
COSTS = ("BASE", "STRESS")
REFERENCES = {
    "2015_2019": Path("reports/research/510300_point_core_observation_v1/results/earlier_diagnostic/references/ENTRY_VINTAGE_STRESS_decisions.parquet"),
    "2020_2026": Path("reports/research/510300_point_current_observation_20261001/results/references/ENTRY_VINTAGE_STRESS_decisions.parquet"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    """保存标准 JSON，缺失值不写成非标准 NaN。"""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat() if not pd.isna(value) else None
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(data + b"\n")


def reconstruct_states(prices, dividends, orders, cost, initial_cash=200000., cutoff=None):
    """逐日推进，只读取截止当日的成交和除息；不接收最终交易结果表。

    周期净标记 = 累计卖出净收入 + 已确认股息 + 现有库存×收盘价 - 累计买入支出。
    累计买入支出随实际加仓增加；这不是剩余库存成本收益率或资金年化收益率。
    """
    require(cost in COSTS, "费用情景无效。")
    p = prices[["date", "close"]].copy()
    p["date"] = pd.to_datetime(p.date)
    require(len(p) and p.date.is_unique and p.date.is_monotonic_increasing, "价格日期须唯一递增。")
    end = pd.Timestamp(cutoff) if cutoff is not None else p.date.iloc[-1]
    p = p.loc[p.date.le(end)].reset_index(drop=True)
    require(len(p) > 0 and np.isfinite(p.close).all() and p.close.gt(0).all(), "缺少有效已知收盘价。")
    o = orders.copy()
    o["date"] = pd.to_datetime(o.date)
    o["origin"] = pd.to_datetime(o.origin)
    o = o.loc[o.date.le(end)].sort_values("date", kind="stable")
    require(o.date.isin(p.date).all(), "观察日历没有覆盖已发生成交。")
    require(o.origin.lt(o.date).all(), "实际成交缺少此前决策原点。")
    require(o.side.isin(["BUY", "SELL"]).all(), "成交方向无效。")
    require(np.isfinite(o[["quantity", "fill_price", "commission", "cycle_id"]]).all().all(), "成交字段缺失。")
    require(o.quantity.gt(0).all() and o.quantity.eq(np.floor(o.quantity)).all(), "成交数量须为正整数。")
    require(o.cycle_id.ge(1).all() and o.cycle_id.eq(np.floor(o.cycle_id)).all(), "周期身份须为正整数。")
    require(o.fill_price.gt(0).all() and o.commission.ge(0).all(), "成交价格或费用无效。")
    div = dividends.copy()
    for field in ("record_date", "ex_date", "payment_date"):
        div[field] = pd.to_datetime(div[field])
    div = div.loc[div.record_date.le(end)].reset_index(drop=True)
    require(div.record_date.le(div.ex_date).all() and div.ex_date.le(div.payment_date).all(), "分红时点顺序无效。")
    require(np.isfinite(div.cash_dividend_per_share).all() and div.cash_dividend_per_share.ge(0).all(), "分红金额无效。")
    calendar = set(p.date)
    order_events = {day: group for day, group in o.groupby("date", sort=False)}
    record_events, ex_events, pay_events = {}, {}, {}
    for event in div.itertuples():
        for mapping, date in ((record_events, event.record_date), (ex_events, event.ex_date), (pay_events, event.payment_date)):
            mapping.setdefault(date, []).append(event)
    states, entitlement, holding, daily = {}, {}, [], []
    active = None
    cash, receivable = float(initial_cash), 0.
    require(np.isfinite(cash) and cash > 0, "初始现金无效。")
    for day_index, price in enumerate(p.itertuples()):
        day, close = price.date, float(price.close)
        commission, accrual, paid = 0., 0., 0.
        if day in order_events:
            for order in order_events[day].itertuples():
                cid, quantity = int(order.cycle_id), int(order.quantity)
                amount = quantity * float(order.fill_price)
                commission += float(order.commission)
                if order.side == "BUY":
                    if active is None:
                        require(cid not in states, "已经结束的周期身份不可复用。")
                        states[cid] = {"actual_cycle_id": cid, "actual_entry_date": day,
                                       "first_day_index": day_index, "shares": 0,
                                       "known_buy_debit_cny": 0., "known_sell_net_cny": 0.,
                                       "known_dividend_cny": 0., "known_buy_orders": 0,
                                       "known_sell_orders": 0.}
                        active = cid
                    require(active == cid, "加仓归属于另一活跃周期。")
                    s = states[cid]
                    debit = amount + float(order.commission)
                    s["shares"] += quantity
                    s["known_buy_debit_cny"] += debit
                    s["known_buy_orders"] += 1
                    cash -= debit
                else:
                    require(active == cid and cid in states, "卖出缺少对应活跃周期。")
                    s = states[cid]
                    require(quantity <= s["shares"], "周期出现负库存。")
                    proceeds = amount - float(order.commission)
                    s["shares"] -= quantity
                    s["known_sell_net_cny"] += proceeds
                    s["known_sell_orders"] += 1
                    cash += proceeds
                    if s["shares"] == 0:
                        active = None
        # 登记日采用当天实际成交后的库存；不在登记日提前计入未来股息。
        for event in record_events.get(day, []):
            if active is not None:
                amount = states[active]["shares"] * float(event.cash_dividend_per_share)
                entitlement[event.Index] = (active, amount)
                for date in (event.ex_date, event.payment_date):
                    require(date > end or date in calendar, "权益处理日期缺少观察记录。")
        for event in ex_events.get(day, []):
            if event.Index in entitlement:
                cid, amount = entitlement[event.Index]
                states[cid]["known_dividend_cny"] += amount
                accrual += amount
                receivable += amount
        for event in pay_events.get(day, []):
            if event.Index in entitlement:
                _, amount = entitlement[event.Index]
                paid += amount
                cash += amount
                receivable -= amount
        require(receivable >= -1e-7, "分红支付早于权益确认。")
        shares = states[active]["shares"] if active is not None else 0
        if active is not None:
            s = states[active]
            mark = s["known_sell_net_cny"] + s["known_dividend_cny"] + shares * close - s["known_buy_debit_cny"]
            exit_px = fill(close, -1, cost)
            exit_reserve = shares * (close - exit_px) + fee(shares * exit_px, cost)
            row = {key: value for key, value in s.items() if key != "first_day_index"}
            row.update(origin=day, known_holding_sessions=day_index - s["first_day_index"] + 1,
                       known_extra_buy_orders=s["known_buy_orders"] - 1,
                       actual_close_mark_pnl_cny=mark,
                       actual_close_mark_return=mark / s["known_buy_debit_cny"],
                       hypothetical_close_exit_cost_cny=exit_reserve,
                       hypothetical_close_liquidation_return=(mark - exit_reserve) / s["known_buy_debit_cny"])
            holding.append(row)
        equity = cash + receivable + shares * close
        cycle_profit = sum(s["known_sell_net_cny"] + s["known_dividend_cny"] + s["shares"] * close - s["known_buy_debit_cny"] for s in states.values())
        require(abs(equity - initial_cash - cycle_profit) < 1e-6, "周期现金流无法还原账户净值。")
        daily.append({"date": day, "cash": cash, "receivable": receivable, "shares": shares,
                      "equity": equity, "dividend_accrual": accrual, "dividend_paid": paid,
                      "commission": commission, "known_cycle_profit_cny": cycle_profit})
    final = []
    close = float(p.close.iloc[-1])
    for s in states.values():
        row = {key: value for key, value in s.items() if key != "first_day_index"}
        pnl = s["known_sell_net_cny"] + s["known_dividend_cny"] + s["shares"] * close - s["known_buy_debit_cny"]
        row.update(known_terminal_mark_pnl_cny=pnl, known_terminal_return=pnl / s["known_buy_debit_cny"])
        final.append(row)
    return pd.DataFrame(holding), pd.DataFrame(daily), pd.DataFrame(final)


def source_paths():
    paths = [BASE / "prices.parquet", BASE / "dividends.csv", E01 / "results/全部决策与退出传递.parquet",
             E01 / "summary.json", Path("reports/research/510300_saved_reference_own_exit_states_20260908/result.json")]
    paths.extend(REFERENCES.values())
    for period in PERIODS:
        for cost in COSTS:
            folder = BASE / f"controls/{period}/{cost}/A_SAVED_WEIGHT"
            paths.extend(folder / f"{name}.parquet" for name in ("daily", "orders", "trades"))
    paths.extend(Path(f"research/{name}.py") for name in (
        "point_account_cashflow_state_v1", "point_account_nr7_inputs_v1", "daily_supply_test_v1",
        "point_weight_path_inputs_v1", "learned_cycle_exit_v1", "rearmed_cycle_exit_account_v1",
        "within_cycle_exit_inputs_v1", "entry_vintage_exit_inputs_v1"))
    paths.append(Path("tests/test_point_account_cashflow_state_v1.py"))
    return paths


def freeze():
    require(not OUT.exists(), "本实验目录已存在，不覆盖冻结记录。")
    sources = [{"path": path.as_posix(), "sha256": digest(ROOT / path)} for path in source_paths()]
    protocol = {
        "study": STUDY, "frozen_at": now(), "role": "CAUSAL_STATE_DEFINITION_AND_SAVED_ACCOUNT_QUALIFICATION_ONLY",
        "user_authorization": "允许新增隔离实验代码，保留原冻结策略",
        "hypothesis": "实际加减仓及参考周期身份差异，可能使参考持仓收益状态不代表真实账户；先核对定义和覆盖，不假设可预测收益。",
        "definition": "累计卖出净收入+已确认股息+剩余库存乘已知收盘价-累计实际买入支出；收益率以截至当日累计买入支出为分母。",
        "clocks": "origin收盘状态；订单须origin早于成交日；股息登记日只登记权益，除息日确认收益，到账日仅转移现金；不使用最终周期结果构造历史状态。",
        "price_and_dividend_limit": "使用原账户已接纳原价/分红快照；无新行情采集。分红表缺独立逐事件首次取得时钟，因此不作为预期股息预测源。",
        "comparison": "固定版本参考原cycle_return为参考库存市值加股息除以原入场支出减一；实际现金流状态为另一经济口径，不能直接替换原模型输入。BASE账户也比较同一STRESS参考来源。",
        "strata": ["参考收益状态可用/不可用", "实际与参考入场日期相同/不同", "截至当日仅首次买入/已有加仓或减仓"],
        "preknown_sparse_phase": {"condition": "holding_at_origin且ordinary_requested_quantity>0且ordinary_learning_cycle_id为空且core_ordinary_contribution>0", "2015_2019_each_cost_cycles": 0, "2020_2026_each_cost_cycles": 5, "interpretation": "该阶段单独改动不能影响较早区间；不再单独设计交易覆盖规则。"},
        "prior_duplicate_boundary": "20260908诊断的单次实际入场可比状态已无预测符号/确认分歧；多次成交或其他入场明确不可比。本实验仅补真实现金流定义，旧三状态退出、半退出、单成分退出等失败不重开。",
        "tests_and_gate": "手算多次成交、权益跨周期归属、未来变化不改历史前缀、负库存和错周期拒绝；4账户逐日现金/应收/库存/净值/股息/佣金及最终自然周期复算误差<=1e-6元。",
        "statistical_role": "只报告覆盖、结构和同日状态差；同一周期多日及两个费用情景不当独立样本；不读取未来回报作参数选择。",
        "new_models": 0, "new_strategy_accounts": 0, "new_parameter_grid": 0,
        "new_execution_assets": [], "orders_authorized": False,
        "failure_exit": "来源变化、归属或恒等式不一致则失败并保留；若只有账面状态差异没有新可验证机制，E02保持未注册。",
        "history_role": "DEVELOPMENT_CALIBRATION_NOT_INDEPENDENT_VALIDATION",
        "sources": sources,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    OUT.mkdir(parents=True)
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"at": protocol["frozen_at"], "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources}, exclusive=True)
    print("已冻结账户现金流状态资格研究；尚未计算新状态差异。", flush=True)


def check_sources():
    freeze_record = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT / "protocol.json") == freeze_record["protocol_sha256"], "冻结协议发生变化。")
    for item in freeze_record["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结来源发生变化：" + item["path"])
    return len(freeze_record["sources"])


def save_table(name, frame):
    folder = OUT / "results"
    folder.mkdir(exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")


def verify_account(reconstructed, terminal, daily, trades, period, cost):
    require(np.array_equal(reconstructed.date.to_numpy(), daily.date.to_numpy()), "账户日历不一致。")
    errors = {field: float(np.abs(reconstructed[field].to_numpy() - daily[field].to_numpy()).max()) for field in
              ("cash", "receivable", "shares", "equity", "dividend_accrual", "dividend_paid", "commission")}
    require(max(errors.values()) < 1e-6, "逐日账户保存值无法复算。")
    merged = terminal.merge(trades, left_on="actual_cycle_id", right_on="cycle_id", validate="one_to_one", suffixes=("", "_saved"))
    require(len(merged) == len(terminal) == len(trades), "实际周期身份未完整匹配。")
    errors["cycle_buy_debit_cny"] = float((merged.known_buy_debit_cny - merged.buy_debit).abs().max())
    errors["cycle_sell_net_cny"] = float((merged.known_sell_net_cny - merged.sell_net_cny).abs().max())
    errors["cycle_dividend_cny"] = float((merged.known_dividend_cny - merged.dividend_cny).abs().max())
    completed = merged.status.eq("COMPLETE")
    require(merged.loc[completed, "shares"].eq(0).all(), "自然完成周期未清仓。")
    errors["completed_cycle_pnl_cny"] = float((merged.loc[completed, "known_terminal_mark_pnl_cny"] - merged.loc[completed, "net_pnl"]).abs().max())
    errors["completed_cycle_return"] = float((merged.loc[completed, "known_terminal_return"] - merged.loc[completed, "net_return"]).abs().max())
    require(max(errors.values()) < 1e-6, "最终自然周期保存值无法复算。")
    return {"period": period, "cost": cost, "days": len(daily), "cycles": len(trades),
            "complete_cycles": int(completed.sum()), "max_errors": errors}, merged


def run():
    require(not (OUT / "summary.json").exists() and not (OUT / "failure.json").exists(), "本冻结实验已有裁决，不重复运行。")
    source_count = check_sources()
    prices = pd.read_parquet(ROOT / BASE / "prices.parquet")
    dividends = pd.read_csv(ROOT / BASE / "dividends.csv")
    e01 = pd.read_parquet(ROOT / E01 / "results/全部决策与退出传递.parquet")
    all_rows, all_terminal, verification = [], [], []
    for period in PERIODS:
        reference = pd.read_parquet(ROOT / REFERENCES[period])
        ref_cols = ["origin", "model_selection_origin", "learning_cycle_id", "cycle_return", "cycle_drawdown", "log_holding_days"]
        reference = reference[ref_cols].rename(columns={field: "reference_" + field for field in ref_cols if field != "origin"})
        for cost in COSTS:
            folder = ROOT / BASE / f"controls/{period}/{cost}/A_SAVED_WEIGHT"
            daily, orders, trades = [pd.read_parquet(folder / (name + ".parquet")) for name in ("daily", "orders", "trades")]
            calendar = prices.loc[prices.date.isin(daily.date)]
            held, reconstructed, terminal = reconstruct_states(calendar, dividends, orders, cost)
            verified, final = verify_account(reconstructed, terminal, daily, trades, period, cost)
            verification.append(verified)
            key = e01.loc[e01.period.eq(period) & e01.cost.eq(cost) & e01.holding_at_origin,
                          ["origin", "actual_cycle_id", "shares_before", "vintage_model_selection_origin",
                           "ordinary_requested_quantity", "ordinary_learning_cycle_id", "core_ordinary_contribution",
                           "vintage_learned_exit_requested", "vintage_continuation_prediction"]]
            require(len(key) == len(held), "E01持仓原点与实际重建数量不一致。")
            key = key.rename(columns={"actual_cycle_id": "e01_actual_cycle_id"})
            paired = held.merge(key, on="origin", validate="one_to_one").merge(reference, on="origin", how="left", validate="one_to_one")
            require(paired.actual_cycle_id.eq(paired.e01_actual_cycle_id).all() and paired.shares.eq(paired.shares_before).all(), "E01周期或库存归属不一致。")
            available = paired.reference_cycle_return.notna()
            require(paired.loc[available, "reference_model_selection_origin"].eq(paired.loc[available, "vintage_model_selection_origin"]).all(), "固定参考入场身份不一致。")
            paired["reference_state_available"] = available
            paired["same_entry_date"] = paired.actual_entry_date.eq(paired.reference_model_selection_origin).where(available)
            paired["known_inventory_changed"] = paired.known_extra_buy_orders.gt(0) | paired.known_sell_orders.gt(0)
            paired["state_stratum"] = np.where(~available, "NO_REFERENCE_HOLDING_STATE", np.where(paired.same_entry_date.eq(True), "SAME_ENTRY", "DIFFERENT_ENTRY"))
            paired["cashflow_stratum"] = np.where(paired.known_inventory_changed, "PRIOR_ACTUAL_ADD_OR_REDUCTION", "INITIAL_BUY_ONLY")
            paired["return_difference"] = paired.actual_close_mark_return - paired.reference_cycle_return
            paired["return_sign_disagreement"] = np.sign(paired.actual_close_mark_return).ne(np.sign(paired.reference_cycle_return)) & available
            paired["ordinary_new_entry_continuation"] = paired.ordinary_requested_quantity.gt(0) & paired.ordinary_learning_cycle_id.isna() & paired.core_ordinary_contribution.gt(0)
            paired["period"], paired["cost"] = period, cost
            final["period"], final["cost"] = period, cost
            all_rows.append(paired)
            all_terminal.append(final)
            reconstructed["period"], reconstructed["cost"] = period, cost
            save_table(f"逐日复算_{period}_{cost}", reconstructed)
    rows = pd.concat(all_rows, ignore_index=True)
    terminal = pd.concat(all_terminal, ignore_index=True)
    strata = []
    for keys, group in rows.groupby(["period", "cost", "state_stratum", "cashflow_stratum"], sort=True):
        diff = group.return_difference.dropna()
        strata.append(dict(zip(["period", "cost", "state_stratum", "cashflow_stratum"], keys),
                           holding_origins=len(group), actual_cycles=group.actual_cycle_id.nunique(),
                           return_sign_disagreements=int(group.return_sign_disagreement.sum()),
                           mean_absolute_return_difference=float(diff.abs().mean()) if len(diff) else None,
                           max_absolute_return_difference=float(diff.abs().max()) if len(diff) else None))
    strata = pd.DataFrame(strata)
    sparse = rows.loc[rows.ordinary_new_entry_continuation].copy()
    counts = [{"period": p, "cost": c, "holding_origins": int(((sparse.period == p) & (sparse.cost == c)).sum()),
               "actual_cycles": int(sparse.loc[sparse.period.eq(p) & sparse.cost.eq(c), "actual_cycle_id"].nunique())} for p in PERIODS for c in COSTS]
    require(all(item["actual_cycles"] == (0 if item["period"] == "2015_2019" else 5) for item in counts), "已知稀疏阶段覆盖未能还原。")
    save_table("持仓现金流状态与参考身份", rows)
    save_table("固定状态分层", strata)
    save_table("自然周期终点复算", terminal)
    save_table("普通参考新入场接续旧持仓", sparse)
    final_source_count = check_sources()
    summary = {
        "study": STUDY, "at": now(), "status": "COMPLETED_CAUSAL_CASHFLOW_STATE_QUALIFICATION",
        "source_files_checked": final_source_count, "accounts_recomputed": len(verification),
        "holding_rows": len(rows), "unique_holding_origins_each_period": rows.loc[rows.cost.eq("STRESS")].groupby("period").size().to_dict(),
        "strata": strata.to_dict("records"), "sparse_phase_counts": counts,
        "verification": verification, "field_definition_gate": "PASS_ACTUAL_CASHFLOW_ACCOUNT_AND_PREFIX_DEFINITION",
        "predictive_information_gate": "NOT_TESTED_NOT_ACCEPTED", "strategy_increment_status": "E02_NOT_REGISTERED_NOT_RUN",
        "economic_comparison": "NOT_RUN", "new_model_fits": 0, "new_strategy_accounts": 0,
        "new_bars": 0, "new_prospective_observations": 0, "goal_achieved": False,
        "interpretation": "真实账户累计加减仓现金流可正确重建；与参考收益状态的差异同时包含入场日期、库存调整、费用和分母差异，不证明参考预测错误，也不证明替换输入会提高收益。",
        "failure_family_preserved": ["POSITION_STATE_ONLY_EXIT", "PARTIAL_LEARNED_EXIT", "SINGLE_COMPONENT_EXIT", "REMAINING_HOLDING_VALUE_TRANSFER"],
    }
    require(source_count == final_source_count, "来源清点数量变化。")
    write_json(OUT / "summary.json", summary, exclusive=True)
    artifacts = [{"path": path.relative_to(OUT).as_posix(), "sha256": digest(path)} for path in sorted((OUT / "results").iterdir())]
    write_json(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_SAVED_FOUR_ACCOUNT_CASHFLOW_RECOMPUTATION",
               "protocol_sha256": digest(OUT / "protocol.json"), "summary_sha256": digest(OUT / "summary.json"),
               "source_files_unchanged": final_source_count, "artifacts": artifacts}, exclusive=True)
    print(f"账户现金流状态资格研究完成：{len(verification)}账户、{len(rows)}持仓原点；没有新策略收益或独立验证。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="隔离账户现金流状态资格研究")
    parser.add_argument("action", choices=("freeze", "run", "check"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "check":
        print(f"冻结原件未变：{check_sources()}份。", flush=True)
    else:
        try:
            run()
        except Exception as error:
            if OUT.exists() and not (OUT / "failure.json").exists() and not (OUT / "summary.json").exists():
                write_json(OUT / "failure.json", {"at": now(), "status": "FAILED_PRESERVED_NO_RULE_REPAIR",
                           "error": str(error), "goal_achieved": False}, exclusive=True)
            raise


if __name__ == "__main__":
    main()
