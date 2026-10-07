"""固定已有五日预测的持仓期限，不重新选择信息、样本或预测参数。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.funding_afternoon_response_daily_v1 as prior
import research.repo_segmentation_daily_v1 as parent
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, digest, now
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_funding_forecast_maturity_account_v1"
DIAGNOSTIC = ROOT / "reports/research/510300_funding_transmission_diagnostic_v1_2"
STUDY = "510300_FUNDING_FORECAST_MATURITY_ACCOUNT_V1"
MODELS = ["PRICE", prior.FUNDING, prior.RESPONSE, prior.PRIMARY]
VARIANTS = ["ROLL_NO_ADD", "FIXED5_NO_ADD"]
PRIMARY = prior.PRIMARY + "__FIXED5_NO_ADD"
COSTS = parent.distribution.COSTS


def risk_only_plan(account, ref, peak, prediction):
    """按当日既有尾部预算最多保留原份额；不能增加库存。"""
    nav, cost = account.value(ref), COSTS["STRESS"]
    context = {"decision_equity": nav, "peak_equity": peak,
               "shares_before_decision": account.shares, "es95": max(float(prediction.es95), 1e-8)}
    for target in range(account.shares, -1, -100):
        if not parent.risk_valid(engine, target, ref, context, cost):
            continue
        limits, adjustment = parent.adjusted_limits(engine, target, ref, context, cost)
        exit_price = engine.fill_price(ref, -1, cost, .001)
        exit_cost = target * (ref - exit_price) + engine.commission(target, exit_price, cost)
        weight = target * ref / nav
        return {**context, **limits, "target_shares": target, "requested_quantity": target - account.shares,
                "planned_weight": weight, "estimated_adjustment_cost": adjustment,
                "reserved_exit_cost": exit_cost, "reference_price": ref,
                "mu5": float(prediction.mu5), "variance5": float(prediction.variance5),
                "score": weight * prediction.mu5 - 2 * weight * weight * prediction.variance5 - (adjustment + exit_cost) / nav}
    raise AssertionError("零库存必须是可行目标。")


def simulate(market, x, dividends, forecasts, model, variant, cost_name, start=parent.START):
    assert variant in [*VARIANTS, "ORIGINAL"]
    first = int(np.flatnonzero(market.date.ge(start))[0])
    prediction_map = {int(r.idx): r for r in forecasts[forecasts.model.eq(model)].itertuples()}
    account = engine.Account(200000.)
    cfg = {"lot": 100, "tick": .001, "limit_fraction": .1}
    cost = COSTS[cost_name]
    previous_equity, previous_mark, previous_reserve, peak = 200000., float(market.close.iloc[first - 1]), 0., 200000.
    stopped, last_view, entry_idx = False, first - 6, None
    records, decisions = [], []
    events = dividends.to_dict("records")
    policy = model + "__" + variant
    for i in range(first, len(market)):
        row = market.iloc[i]
        day, op, close = row.date, float(row.open), float(row.close)
        old_shares, recognized, paid = account.shares, 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                account.receivables[k] = value
                recognized += value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        ref = float(row.previous_close - row.dividend)
        prediction = prediction_map.get(i)
        if prediction is not None:
            last_view = i
        deadline = entry_idx + 5 if entry_idx is not None else None
        plan = {"requested_quantity": 0, "target_shares": account.shares, "reason": "NO_VIEW_HOLD_OLD_INVENTORY"}
        if stopped:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="DRAWDOWN_STOP")
        elif variant == "FIXED5_NO_ADD" and account.shares and i >= deadline:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="FIVE_DAY_MATURITY")
        elif prediction is not None:
            pressure = float(x.pressure5.iloc[i])
            if account.shares and variant == "FIXED5_NO_ADD":
                plan = {**risk_only_plan(account, ref, peak, prediction), "reason": "FIXED_MATURITY_DAILY_RISK"}
            else:
                if account.shares and variant == "ROLL_NO_ADD":
                    pressure = 0.
                plan = {**parent.plan(engine, account, ref, peak, prediction, pressure), "reason": "DAILY_STATE_DISTRIBUTION"}
        elif i - last_view >= 5:
            plan.update(requested_quantity=-account.shares, target_shares=0, reason="NO_VIEW_INVENTORY_EXPIRED")
        requested = int(plan["requested_quantity"])
        before_gap = requested
        if requested > 0:
            while requested > 0 and not parent.risk_valid(engine, account.shares + requested, op, plan, COSTS["STRESS"]):
                requested -= 100
        sellable = account.sellable(i)
        execution = engine.execute_order(account, requested, op, float(row.previous_close), float(row.dividend), i, cost, cfg)
        if execution["filled_quantity"] > 0 and old_shares == 0:
            entry_idx = i
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == len(market) - 1 and account.shares:
            exit_price = engine.fill_price(close, -1, COSTS["STRESS"], .001)
            reserve = account.shares * (close - exit_price) + engine.commission(account.shares, exit_price, COSTS["STRESS"])
        equity = account.value(close) - reserve
        price_pnl = old_shares * (op - previous_mark) + account.shares * (close - op)
        error = equity - previous_equity - price_pnl - recognized + execution["commission"] + execution["slippage_cost"] + reserve - previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        if drawdown >= .1:
            stopped = True
        records.append({"date": day, "idx": i, "policy": policy, "cost": cost_name, "open": op, "mark": close,
            "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
            "terminal_exit_reserve": reserve, "equity": equity, "net_return": equity / previous_equity - 1,
            "pnl": equity - previous_equity, "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
            "exposure": account.shares * close / equity, "accounting_error": error, "drawdown": drawdown,
            "risk_stopped": stopped, "sellable_before": sellable,
            "terminal_unliquidated": bool(i == len(market) - 1 and account.shares), **execution})
        decisions.append({"date": day, "idx": i, "policy": policy, "cost": cost_name, "prediction_available": prediction is not None,
            "pressure5": float(x.pressure5.iloc[i]), "pre_open_requested_quantity": before_gap,
            "gap_checked_request": requested, "filled_quantity": execution["filled_quantity"], "actual_open": op,
            "cycle_entry_idx": entry_idx, "deadline_idx": entry_idx + 5 if entry_idx is not None else None, **plan})
        if account.shares == 0:
            entry_idx = None
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def implementation_checks():
    dates = pd.bdate_range("2020-01-01", periods=20)
    market = pd.DataFrame({"date": dates, "open": 4., "close": 4., "previous_close": 4., "dividend": 0.})
    x = pd.DataFrame({"pressure5": np.ones(20)})
    forecasts = pd.DataFrame({"idx": np.arange(1, 20), "model": "PRICE", "mu5": .04, "variance5": .001, "es95": .025})
    dividends = pd.DataFrame(columns=["ex_date", "record_date", "payment_date", "cash_dividend_per_share"])
    proxy = SimpleNamespace(START="2020-01-02", END="2020-12-31", COSTS=COSTS, plan=parent.plan, risk_valid=parent.risk_valid)
    for cost in COSTS:
        original, _ = simulate(market, x, dividends, forecasts, "PRICE", "ORIGINAL", cost)
        control, _ = inventory.simulate(proxy, engine, market, x, dividends, forecasts, "PRICE_DOWN_ONLY", cost)
        for column in ["shares", "cash", "equity", "filled_quantity", "commission", "slippage_cost"]:
            np.testing.assert_allclose(original[column], control[column], atol=1e-9, rtol=0)
        for variant in VARIANTS:
            ledger, decisions = simulate(market, x, dividends, forecasts, "PRICE", variant, cost)
            parent.account_verification(ledger, decisions)
            assert not ((ledger.shares.shift(1, fill_value=0) > 0) & ledger.filled_quantity.gt(0)).any()
            if variant == "FIXED5_NO_ADD":
                assert decisions.loc[decisions.reason.eq("FIVE_DAY_MATURITY"), "idx"].to_list() == [6, 12, 18]
                assert ledger.loc[ledger.idx.isin([6, 12, 18]), "shares"].eq(0).all()
    sparse = forecasts.iloc[:1]
    ledger, decisions = simulate(market, x, dividends, sparse, "PRICE", "FIXED5_NO_ADD", "STRESS")
    assert decisions.loc[decisions.idx.eq(6), "reason"].iloc[0] == "FIVE_DAY_MATURITY"
    assert ledger.loc[ledger.idx.ge(6), "shares"].eq(0).all()
    return {"synthetic_original_matches_engine": True, "no_add_during_cycle": True,
            "fifth_following_session_exit": True, "maturity_without_new_forecast": True,
            "cash_T1_risk_account_identities": True}


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("五日期限检验已固定，不能覆盖。")
    tests = implementation_checks()
    for folder in ["code", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "已有模型预测五日收益，滚动更新却能无限延长同一笔仓位；匹配固定五日到期是否改善账户？",
        "motivation": "保存账户隔夜含息损益为负，五日预测和实际持有路径不同；这是事后提出的新开发检验，不是确认性证据。",
        "models": MODELS, "variants": VARIANTS,
        "forecast_reuse": "原四种每日两日历年训练预测逐值复用；不重新拟合、不增特征、不改近邻、入场均值或成本门槛。",
        "entry": "空仓时沿用原效用、弱势入场及风险上限；所有新账户在持仓周期内禁止加仓，避免混合不同期限的预测。",
        "ROLL_NO_ADD": "原每日五日前景决定持有或退出，唯一变化为周期内不加仓；用于识别加仓变化。",
        "FIXED5_NO_ADD": "首笔实际买入的第五个后续交易日开盘必须卖出；中途只因当日尾部预算或10%回撤止损减仓，不因新五日均值续期。到期当天只卖，下一交易日才可再入。",
        "daily_risk": "有每日预测时，最大可保留份额受原压力费用后尾部预算约束，不能加仓。缺预测时仍执行已固定到期；ROLL沿用缺信息满五日退出。",
        "maturity_is_not_guaranteed_fill": "到期未成交继续要求退出；T+1、涨跌停、现金与100份整手照常。未到期末端计清算储备，不偷看数据终点退出。",
        "account_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "assets": ["510300.SH", "CASH_CNY"], "capital_cny": 200000,
        "period": [parent.START, parent.END], "annual_days": 242, "cash_and_risk_free": 0,
        "costs": COSTS, "targets": {"sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "new_full_accounts": 16, "reused_original_accounts": 8,
        "primary_comparisons": ["联合FIXED5相对联合ROLL_NO_ADD", "联合FIXED5相对相同期限的下午价格对照"],
        "additional_comparisons": "四种信息分别比较FIXED5与ROLL_NO_ADD；ROLL_NO_ADD与原有可加仓账户；联合FIXED5与另外两个同期限信息对照。全部披露，不事后换主候选。",
        "intervals": "20交易日循环区块2000次，seed2026092603；固定2020—2023/2024—末端；所有滚动两年。",
        "development_gate": "主压力三目标同时通过，两个主增量区间下界为正，且两个固定时期增量均正；之后仍需独立验证。",
        "duplicate_boundary": "旧固定五日剩余价值研究改变训练标签、使用旧形态退出；本轮复用新资金意外预测，只检验实际库存期限。旧期限扫描不恢复，不搜索1/3/10/20日。",
        "new_fits": 0, "new_market_downloads": 0, "new_parameter_grid": 0,
        "independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "protocol.json", protocol, True)
    save(OUT / "implementation_checks.json", tests, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    paths = [Path(__file__), Path(parent.__file__), Path(engine.__file__), Path(inventory.__file__), parent.DIVIDENDS,
        parent.OUT / "inputs/market.parquet", prior.OUT / "inputs/decision_information.parquet",
        prior.OUT / "results/predictions.parquet", DIAGNOSTIC / "result.json",
        ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py"]
    for model in MODELS:
        for cost in COSTS:
            paths.append(prior.OUT / "accounts" / cost / model / "ledger.parquet")
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_account_results": True}, True)
    print("五日到期与不加仓对照已固定，定向账户检查通过；下一步运行16个新账户。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["files"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    market = pd.read_parquet(parent.OUT / "inputs/market.parquet")
    x = pd.read_parquet(prior.OUT / "inputs/decision_information.parquet")
    forecasts = pd.read_parquet(prior.OUT / "results/predictions.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    accounts, checks, cycle_rows = {}, [], []
    for model in MODELS:
        for variant in VARIANTS:
            for cost in COSTS:
                policy = model + "__" + variant
                ledger, decisions = simulate(market, x, dividends, forecasts, model, variant, cost)
                check = parent.account_verification(ledger, decisions)
                assert not ((ledger.shares.shift(1, fill_value=0) > 0) & ledger.filled_quantity.gt(0)).any()
                mature = decisions[decisions.reason.eq("FIVE_DAY_MATURITY")]
                if variant == "FIXED5_NO_ADD":
                    assert mature.idx.ge(mature.deadline_idx).all()
                folder = OUT / "accounts" / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                closed, pending = cycles(ledger)
                closed.to_parquet(folder / "cycles.parquet", index=False)
                save(folder / "pending_cycle.json", pending, True)
                accounts[policy, cost] = ledger
                checks.append({"policy": policy, "cost": cost, **check, "no_cycle_add": True})
                cycle_rows.append({"policy": policy, "cost": cost, "closed": len(closed), "open": pending is not None,
                    "maturity_exit_requests": len(mature), "maturity_unfilled_requests": int(mature.filled_quantity.eq(0).sum())})
                m = metrics(ledger)
                sh = "未定义" if m["sharpe"] is None else f"{m['sharpe']:.6f}"
                print(f"{policy}/{cost}：夏普{sh}，年化{m['annual_return']:.3%}，回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, windows = summarize_accounts(OUT, market, accounts)
    rng = np.random.default_rng(2026092603)
    comparisons = []
    for cost in COSTS:
        pairs = []
        for model in MODELS:
            original = pd.read_parquet(prior.OUT / "accounts" / cost / model / "ledger.parquet")
            pairs.extend([(model + "__FIXED5_NO_ADD", model + "__ROLL_NO_ADD", accounts[model + "__ROLL_NO_ADD", cost]),
                          (model + "__ROLL_NO_ADD", model + "__ORIGINAL", original)])
        for control in [prior.RESPONSE, prior.FUNDING, "PRICE"]:
            pairs.append((PRIMARY, control + "__FIXED5_NO_ADD", accounts[control + "__FIXED5_NO_ADD", cost]))
        for name, control, right in pairs:
            left = accounts[name, cost]
            assert left.date.equals(right.date)
            periods = []
            for lo, hi in [(parent.START, "2023-12-31"), ("2024-01-01", parent.END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi,
                    "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"policy": name, "cost": cost, "control": control,
                **paired_interval(left, right, rng), "fixed_periods": periods})
    primary = next(row for row in measures if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    main_controls = [prior.PRIMARY + "__ROLL_NO_ADD", prior.RESPONSE + "__FIXED5_NO_ADD"]
    primary_pairs = [row for row in comparisons if row["policy"] == PRIMARY and row["cost"] == "STRESS" and row["control"] in main_controls]
    increment = all(row["lower_95"] > 0 and all(p["annual_arithmetic_difference"] > 0 for p in row["fixed_periods"]) for row in primary_pairs)
    assert len(primary_pairs) == 2
    for path, sha in frozen["files"].items():
        assert digest(ROOT / path) == sha, path
    result = {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_FUNDING_MATURITY_STRATEGY",
        "primary": primary, "all_accounts": measures, "comparisons": comparisons,
        "primary_increment_gate": increment, "account_checks": checks, "cycles": cycle_rows,
        "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in windows if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
        "new_full_accounts": 16, "reused_original_accounts": 8, "reused_prediction_days": 920,
        "new_fits": 0, "new_market_downloads": 0, "new_independent_forward_observations": 0,
        "original_files_unchanged": True, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("16个期限对照账户完成，完整结果已经保存。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "freeze", "run"])
    action = parser.parse_args().action
    if action == "check":
        print(implementation_checks())
    elif action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            save(OUT / "RUN_FAILURE.json", {"at": now(), "type": type(exc).__name__, "message": str(exc)}, True)
            raise
