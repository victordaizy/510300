"""固定既有信号，分离费用、账户风控与仓位限制的历史影响。"""
import argparse
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research import factor96_margin_repair_v1 as core
from research import factor96_rapid_feasibility_v1 as rapid
from scripts.record_factor96_remaining_changes_v1 import update

ROOT = rapid.ROOT
OUT = ROOT / "reports/research/510300_factor96_bottleneck_diagnostic_v1"
STUDY = "510300_FACTOR96_BOTTLENECK_DIAGNOSTIC_V1"
REQUEST = "先定位夏普1.2的主要瓶颈：信号、交易成本还是账户约束"
ZERO = {"commission": 0., "minimum": 0., "slippage": 0.}
SOURCES = {
    "feasibility": ("screen_features.parquet", 10),
    "capital_proxy": ("proxy_features.parquet", 10),
    "structure": (None, 10),
    "orders_inventory": ("daily_signals.parquet", 20),
    "external_resilience": ("features_and_signals.parquet", 5),
    "breadth_speed": ("daily_signals.parquet", 10),
}
read, save = rapid.read, rapid.save


def source_inventory():
    tables, cases = [], []
    for name, (feature_name, hold) in SOURCES.items():
        folder = ROOT / f"reports/research/510300_factor96_rapid_{name}_v1"
        table = pd.read_csv(folder / "all_account_metrics.csv")
        table = table.loc[table.cost.eq("STRESS")].copy()
        table["study"] = name
        if "group" not in table:
            table["group"] = "MAIN"
        table["case_id"] = table.study + "__" + table.group + "__" + table.policy
        table["source_metrics"] = str((folder / "all_account_metrics.csv").relative_to(ROOT))
        tables.append(table)
        for row in table.loc[table.capital.eq(200000) & ~table.policy.isin(["BUY_HOLD_50", "CASH"])].to_dict("records"):
            feature = folder / (feature_name or f"{row['group']}_signals.parquet")
            cases.append({key: row[key] for key in ["study", "group", "policy", "case_id", "ledger_path", "source_metrics"]} | {
                "features": str(feature.relative_to(ROOT)), "hold": hold,
                "start": "2022-02-07" if row["group"] == "ETF_MIGRATION_PROXY" else "2017-01-03",
                "end": "2025-12-31", "capital": 200000})
    return pd.concat(tables, ignore_index=True), cases


def freeze():
    table, cases = source_inventory()
    assert len(cases) == 49 and len(table) == 116
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    assert not mandate["orders_authorized"] and not mandate["delivery_package_required"]
    save(OUT / "protocol.json", {
        "at": core.now(), "study_id": STUDY, "user_instruction": REQUEST,
        "classification": "FIXED_SIGNAL_COUNTERFACTUAL_DIAGNOSTIC_NOT_NEW_STRATEGIES",
        "scope": "最近六轮全部49组非买入持有、非现金的信号与对照；保留重复规则，不当作49个独立机制。",
        "cases": cases, "primary_capital": 200000, "annual_days": 242, "cash_return": 0,
        "fixed_holdings_addback": "116份原压力账本，含2万元、20万元及基准：权益加回累计佣金、累计滑点和当天期末清算准备。原持仓不变、返费不再投资，仅作费用归因，不是新可交易策略。",
        "factorial": {"risk_modes": ["ORIGINAL", "CAP50_ONLY", "CAP100_ONLY"], "costs": ["STRESS", "ZERO"],
            "ORIGINAL": "沿用ES预算、回撤剩余空间、半仓入场和10%回撤永久退出。ZERO同时移除执行和规划费用。",
            "CAP50_ONLY": "仅保留入场50%仓位限制、整手、现金、T+1和涨跌停；去掉ES、回撤储备、期间减仓及永久退出。仓位可随持有期价格漂移。",
            "CAP100_ONLY": "同上，入场上限改为100%，不加杠杆、不做空。",
            "fixed": "所有已保存布尔信号、判断时钟、评价区间、持有间隔保持不变；原信号本身已包含的资料和价格条件均保留。",
            "position_sizing": "入场分别按前收盘除息参考价与当日开盘价计算上限，取较小值；按当时费用和可用现金约束。",
            "terminal_cost": "各模拟按自身成本情形扣期末清算准备。"},
        "reproduction_gate": "先复现全部49份ORIGINAL/STRESS账本，逐日核对权益、持仓、现金、分红、费用和成交；失败则不运行反事实。",
        "planned_reproductions": 49, "planned_new_counterfactual_accounts": 245,
        "report": "全体同表和逐规则配对差值；事后最佳只解释上限线索，不作新候选。2017-2025与ETF迁移2022-2025不合并为同区间竞赛。",
        "limits": "固定信号失败不证明510300不可能达到目标；去掉这些约束也不是所有可能账户结构的数学上界。费用与风控有相互作用，不能把各项差值简单相加。历史重建样本不是独立验证。",
        "mandate_unchanged": True, "goal_achieved": False, "orders_authorized": False, "delivery_package_required": False})
    table.to_csv(OUT / "source_accounts.csv", index=False, encoding="utf-8-sig")
    paths = {Path(__file__), rapid.ENGINE_PATH, Path(core.__file__), OUT / "protocol.json", OUT / "source_accounts.csv",
             rapid.PRICE_BASE / "market.parquet", rapid.PRICE_BASE / "dividends.csv"}
    paths |= {ROOT / c[k] for c in cases for k in ["features", "source_metrics"]}
    paths |= {ROOT / p for p in table.ledger_path}
    save(OUT / "freeze.json", {"at": core.now(), "files": [{"path": str(p), "sha256": core.digest(p)} for p in sorted(paths)]})
    status_path = ROOT / "reports/research/510300_factor96_program_v1/status.json"
    status = read(status_path)
    save(OUT / "authority_change.json", {"at": core.now(), "user_instruction": REQUEST, "previous_round": status["latest_round"],
        "new_direction": "BOTTLENECK_DIAGNOSTIC_FIRST", "mandate_relaxation_authorized": False, "diagnostic_counterfactuals_only": True})
    status.update(research_priority="BOTTLENECK_DIAGNOSTIC_FIRST", next_independent_source_action="RUN_FIXED_SIGNAL_FACTORIAL_DIAGNOSTIC",
                  current_research_phase="BOTTLENECK_DIAGNOSTIC_FROZEN")
    update(status_path, status)
    mandate.update(latest_research_direction_instruction=REQUEST, research_priority="BOTTLENECK_DIAGNOSTIC_FIRST",
        latest_priority_receipt=str((OUT / "authority_change.json").relative_to(ROOT)))
    update(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("瓶颈诊断已固定：116份持仓返费归因、49份原账本复现、245个新反事实账户。", flush=True)


def original_target(engine, account, reference, peak, es, cost):
    """与既有算法相同，仅把规划费用作为显式反事实参数。"""
    nav = account.value(reference)
    if not np.isfinite(es) or nav <= 0:
        return 0
    remaining = max(0., nav - .9 * peak)
    cap = min(.5, .025 / max(es, 1e-12), .5 * remaining / (.1 * nav))
    q = math.floor(max(0., cap) * nav / reference / 100) * 100
    while q > 0:
        change = q - account.shares
        px = engine.fill_price(reference, 1 if change > 0 else -1, cost, .001)
        friction = abs(change) * abs(px-reference) + engine.commission(change, px, cost)
        exit_px = engine.fill_price(reference, -1, cost, .001)
        reserve = q * (reference-exit_px) + engine.commission(q, exit_px, cost)
        if (q * reference * es + friction + reserve <= .025 * nav + 1e-8 and
            q * reference * .1 + friction + reserve <= min(.05 * nav, .5 * remaining) + 1e-8 and
            q * reference <= .5 * (nav-friction) + 1e-8):
            return q
        q -= 100
    return 0


def capped_target(engine, account, price, cap, cost):
    nav = account.value(price)
    px = engine.fill_price(price, 1, cost, .001)
    q = math.floor(cap * nav / px / 100) * 100
    while q > 0:
        fee = engine.commission(q, px, cost)
        if q * px <= cap * (nav-fee) + 1e-8 and q * px + fee <= account.cash + 1e-8:
            return q
        q -= 100
    return 0


def simulate(m, f, dividends, case, mode, cost_name, engine):
    capital, policy, hold = case["capital"], case["policy"], case["hold"]
    cost = core.COSTS["STRESS"] if cost_name == "STRESS" else ZERO
    account = engine.Account(float(capital))
    config = {"lot": 100, "tick": .001, "limit_fraction": .1}
    indices = np.flatnonzero(m.date.between(case["start"], case["end"]))
    first, last = int(indices[0]), int(indices[-1])
    previous_equity, previous_mark, reserve_before, peak = float(capital), float(m.close.iloc[first-1]), 0., float(capital)
    entry, stopped, pending_exit = None, False, False
    ledger, events = [], dividends.to_dict("records")
    rows, es, signals = list(m.itertuples(index=False)), f.es95.to_numpy(float), f[policy].to_numpy(bool)
    for i in indices:
        row = rows[i]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                amount = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[k] = amount
                    recognized += amount
            if event["payment_date"] < day and k in account.receivables:
                paid_now = account.receivables.pop(k)
                account.cash += paid_now
                paid += paid_now
        quantity, reason = 0, "现金或持有"
        reference = float(rows[i-1].close - row.dividend)
        if old:
            pending_exit |= stopped or i >= entry + hold
            if pending_exit:
                quantity, reason = -old, "回撤退出" if stopped else "固定持有期到期"
            elif mode == "ORIGINAL":
                target = min(old, original_target(engine, account, reference, peak, es[i-1], cost))
                quantity, reason = target-old, "风险预算减仓"
        elif not stopped and signals[i-1]:
            if mode == "ORIGINAL":
                quantity = min(original_target(engine, account, reference, peak, es[i-1], cost),
                               original_target(engine, account, op, peak, es[i-1], cost))
            else:
                cap = .5 if mode == "CAP50_ONLY" else 1.
                quantity = min(capped_target(engine, account, reference, cap, cost), capped_target(engine, account, op, cap, cost))
            reason = "前收盘信号入场"
        execution = engine.execute_order(account, quantity, op, float(row.previous_close), float(row.dividend), int(i), cost, config)
        if execution["filled_quantity"] > 0 and old == 0:
            entry, pending_exit = int(i), False
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                paid_now = account.receivables.pop(k)
                account.cash += paid_now
                paid += paid_now
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            exit_price = engine.fill_price(close, -1, cost, .001)
            reserve = account.shares * (close-exit_price) + engine.commission(account.shares, exit_price, cost)
        equity = account.value(close)-reserve
        price_pnl = old * (op-previous_mark) + account.shares * (close-op)
        error = equity-previous_equity-price_pnl-recognized+execution["commission"]+execution["slippage_cost"]+reserve-reserve_before
        assert abs(error) < 1e-6, (case["case_id"], mode, cost_name, day, error)
        account.assert_valid()
        peak = max(peak, equity)
        dd = 1-equity/peak
        if mode == "ORIGINAL":
            stopped |= dd >= .1
        ledger.append({"date": day, "idx": int(i), "equity": equity, "net_return": equity/previous_equity-1,
            "shares": account.shares, "cash": account.cash, "dividend_receivable": account.receivable(),
            "dividend_recognized": recognized, "dividend_paid": paid, "mark": close,
            "exposure": account.shares*close/equity, "drawdown": dd, "risk_stopped": stopped,
            "terminal_exit_reserve": reserve, "accounting_error": error, "reason": reason, **execution})
        if not account.shares:
            entry, pending_exit = None, False
        previous_equity, previous_mark, reserve_before = equity, close, reserve
    return pd.DataFrame(ledger)


def details(ledger, capital):
    stopped = ledger.loc[ledger.risk_stopped, "date"]
    result = core.metrics(ledger, capital)
    result.update(entries=int((ledger.filled_quantity.gt(0) & ledger.shares_before.eq(0)).sum()),
        risk_reductions=int((ledger.filled_quantity.lt(0) & ledger.reason.eq("风险预算减仓")).sum()),
        stopped_at=stopped.iloc[0].isoformat() if len(stopped) else None,
        holding_days=int(ledger.shares.gt(0).sum()), max_exposure=float(ledger.exposure.max()),
        entry_mean_exposure=ledger.loc[ledger.filled_quantity.gt(0) & ledger.shares_before.eq(0), "exposure"].mean())
    for prefix, start, end in [("early", "2017-01-03", "2020-12-31"), ("late", "2021-01-01", "2025-12-31")]:
        result.update({prefix+"_"+key: value for key, value in rapid.period_metrics(ledger, start, end).items()})
    return result


def run():
    for item in read(OUT / "freeze.json")["files"]:
        assert core.digest(Path(item["path"])) == item["sha256"], item["path"]
    protocol = read(OUT / "protocol.json")
    cases = protocol["cases"]
    save(OUT / "run_started.json", {"at": core.now(), "reproductions": 49, "new_counterfactual_accounts": 245})
    m = pd.read_parquet(rapid.PRICE_BASE / "market.parquet")
    m.date = pd.to_datetime(m.date)
    m = m.loc[m.date.le("2025-12-31")].reset_index(drop=True)
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    features = {p: pd.read_parquet(ROOT / p) for p in {c["features"] for c in cases}}
    for f in features.values():
        f.date = pd.to_datetime(f.date)
        assert m.date.equals(f.date)
    spec = importlib.util.spec_from_file_location("bottleneck_frozen_account_engine", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    rows, reproduction = [], []
    for n, case in enumerate(cases, 1):
        actual = pd.read_parquet(ROOT / case["ledger_path"])
        replay = simulate(m, features[case["features"]], dividends, case, "ORIGINAL", "STRESS", engine)
        columns = [c for c in actual if c != "reason"]
        pd.testing.assert_frame_equal(actual[columns], replay[columns], check_dtype=False, check_exact=False, rtol=0, atol=1e-7)
        rows.append({**case, "mode": "ORIGINAL", "cost": "STRESS", "reproduction": True, **details(actual, case["capital"])})
        reproduction.append({"case_id": case["case_id"], "max_equity_error": float((actual.equity-replay.equity).abs().max()), "all_numeric_and_status_columns_match": True})
        if n % 10 == 0 or n == len(cases):
            print(f"已逐日复现原账户 {n}/{len(cases)}。", flush=True)
    save(OUT / "reproduction_receipt.json", {"at": core.now(), "status": "PASS", "accounts": reproduction})
    source_accounts = pd.read_csv(OUT / "source_accounts.csv")
    addbacks = []
    for row in source_accounts.to_dict("records"):
        ledger = pd.read_parquet(ROOT / row["ledger_path"])
        base = core.metrics(ledger, row["capital"])
        rebated = ledger.copy()
        refund = ledger.commission.cumsum()+ledger.slippage_cost.cumsum()+ledger.terminal_exit_reserve
        rebated["equity"] = ledger.equity+refund
        rebated["cash"] = ledger.cash+ledger.commission.cumsum()+ledger.slippage_cost.cumsum()
        nav = np.r_[row["capital"], rebated.equity.to_numpy(float)]
        rebated["net_return"] = nav[1:]/nav[:-1]-1
        rebated["exposure"] = rebated.shares*rebated.mark/rebated.equity
        for column in ["commission", "slippage_cost", "terminal_exit_reserve"]:
            rebated[column] = 0.
        metrics = core.metrics(rebated, row["capital"])
        addbacks.append({key: row[key] for key in ["case_id", "study", "group", "policy", "capital", "ledger_path"]} | {
            "original_sharpe": base["net_sharpe"], "rebated_sharpe": metrics["net_sharpe"],
            "original_cagr": base["cagr"], "rebated_cagr": metrics["cagr"],
            "original_max_drawdown": base["max_drawdown"], "rebated_max_drawdown": metrics["max_drawdown"],
            "commission_CNY": base["commission"], "slippage_CNY": base["slippage"],
            "terminal_reserve_CNY": base["terminal_exit_reserve"], "total_refund_CNY": float(refund.iloc[-1]),
            "sharpe_change": (metrics["net_sharpe"]-base["net_sharpe"]) if base["net_sharpe"] is not None else None,
            "cagr_change": metrics["cagr"]-base["cagr"]})
    pd.DataFrame(addbacks).to_csv(OUT / "fixed_holdings_cost_attribution.csv", index=False, encoding="utf-8-sig")
    count = 0
    for case in cases:
        for mode, cost in [("ORIGINAL", "ZERO"), ("CAP50_ONLY", "STRESS"), ("CAP50_ONLY", "ZERO"), ("CAP100_ONLY", "STRESS"), ("CAP100_ONLY", "ZERO")]:
            ledger = simulate(m, features[case["features"]], dividends, case, mode, cost, engine)
            folder = OUT / "accounts" / case["case_id"] / f"{mode}_{cost}"
            folder.mkdir(parents=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            rows.append({**case, "mode": mode, "cost": cost, "reproduction": False,
                "ledger_path": str((folder / "ledger.parquet").relative_to(ROOT)), **details(ledger, case["capital"])})
            count += 1
        if count % 25 == 0 or count == 245:
            print(f"已完成反事实账户 {count}/245。", flush=True)
    table = pd.DataFrame(rows)
    table["sharpe_at_least_1_2"] = table.net_sharpe.ge(1.2)
    table["joint_sharpe_and_cagr"] = table.net_sharpe.ge(1.2) & table.cagr.ge(.1)
    table["joint_with_drawdown"] = table.joint_sharpe_and_cagr & table.max_drawdown.le(.1)
    table.to_csv(OUT / "all_counterfactual_metrics.csv", index=False, encoding="utf-8-sig")
    records = []
    for mode, cost in [("ORIGINAL", "ZERO"), ("CAP50_ONLY", "STRESS"), ("CAP50_ONLY", "ZERO"), ("CAP100_ONLY", "STRESS"), ("CAP100_ONLY", "ZERO")]:
        original = table.loc[table["mode"].eq("ORIGINAL") & table.cost.eq("STRESS")].set_index("case_id")
        scenario = table.loc[table["mode"].eq(mode) & table.cost.eq(cost)].set_index("case_id")
        for key, row in scenario.iterrows():
            before = original.loc[key]
            records.append({"case_id": key, "mode": mode, "cost": cost,
                "sharpe_change": row.net_sharpe-before.net_sharpe, "cagr_change": row.cagr-before.cagr,
                "drawdown_change": row.max_drawdown-before.max_drawdown,
                "exposure_change": row.mean_exposure-before.mean_exposure,
                "entries_change": int(row.entries-before.entries)})
    pd.DataFrame(records).to_csv(OUT / "paired_changes_from_original.csv", index=False, encoding="utf-8-sig")
    save(OUT / "run_completed.json", {"at": core.now(), "reproductions": 49, "counterfactual_accounts": count,
        "source_cost_attributions": len(addbacks), "goal_achieved": False, "orders_authorized": False})
    print("瓶颈诊断计算完成；等待读取结果并登记解释。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run"])
    stage = parser.parse_args().stage
    {"freeze": freeze, "run": run}[stage]()
