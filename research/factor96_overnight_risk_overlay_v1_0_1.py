"""T16：隔夜下行占比或方差预测误差升高时，削减固定观察基准的已有份额。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from research import factor96_margin_repair_v1 as core

ROOT = core.ROOT
OUT = ROOT/"reports/research/510300_factor96_overnight_risk_overlay_v1_0_1"
PREVIOUS = ROOT/"reports/research/510300_factor96_crowding_overlay_v1"
STUDY = "510300_FACTOR96_OVERNIGHT_RISK_OVERLAY_V1_0_1"
POLICIES = ["PRICE_ALL", "PRICE_COMMON", "D06_ONLY", "P02_ONLY", "FULL"]
save, now, digest = core.save, core.now, core.digest


def predictable_variance(returns, decay=.94, seed_days=60):
    """用此前收益平方预测下一日方差；中断后重新等待完整的种子窗口。"""
    values = np.asarray(returns, dtype=float)
    forecast = np.full(len(values), np.nan)
    for i in range(seed_days, len(values)):
        if i > 0 and np.isfinite(forecast[i-1]) and np.isfinite(values[i-1]):
            forecast[i] = decay*forecast[i-1]+(1-decay)*values[i-1]**2
        else:
            seed = values[i-seed_days:i]
            if np.isfinite(seed).all():
                forecast[i] = np.mean(seed**2)
    return forecast


def risk_features(market):
    d = market.reset_index(drop=True)
    previous = d.close.shift()
    overnight = (d.open+d.dividend)/previous-1
    intraday = (d.close+d.dividend)/(d.open+d.dividend)-1
    total = (d.close+d.dividend)/previous-1
    valid = np.isfinite(overnight) & np.isfinite(intraday) & np.isfinite(total)
    overnight, intraday, total = overnight.where(valid), intraday.where(valid), total.where(valid)
    denominator = (overnight**2+intraday**2).rolling(60, min_periods=60).sum()
    numerator = overnight.clip(upper=0).pow(2).rolling(60, min_periods=60).sum()
    result = pd.DataFrame({"date": d.date, "overnight_return": overnight, "intraday_return": intraday,
        "session_total_return": total, "D06": numerator/denominator.where(denominator.gt(0))})
    result["overnight_tail_mean"] = overnight.rolling(60, min_periods=60).apply(lambda a: np.sort(a)[:3].mean(), raw=True)
    result["forecast_variance"] = predictable_variance(total)
    result["P02"] = total.pow(2)/result.forecast_variance.where(result.forecast_variance.gt(0))
    for label, prefix in [("D06", "d06"), ("P02", "p02")]:
        history = result[label].shift().rolling(252, min_periods=120)
        result[label+"_q90"] = history.quantile(.9)
        result[label+"_q70"] = history.quantile(.7)
        result[prefix+"_known"] = np.isfinite(result[[label, label+"_q90", label+"_q70"]]).all(axis=1)
        result[prefix+"_high"] = result[prefix+"_known"] & result[label].gt(result[label+"_q90"])
        result[prefix+"_low"] = result[prefix+"_known"] & result[label].lt(result[label+"_q70"])
    return result


def lag_features(market, price, risk, lag):
    assert lag in (0, 1)
    f = price[["date", "wealth", "es95", *core.STATE]].copy()
    f["ma20"] = f.wealth.rolling(20, min_periods=20).mean()
    f["price_long"] = f.wealth.gt(f.ma20)
    observed = risk.drop(columns="date").copy()
    observed["stat_date"] = market.date
    observed["stat_idx"] = np.arange(len(market))
    observed = observed.shift(lag)
    for col in ["d06_known", "p02_known", "d06_high", "p02_high", "d06_low", "p02_low"]:
        observed[col] = observed[col].fillna(False).astype(bool)
    observed["stat_idx"] = observed.stat_idx.fillna(-1).astype(int)
    f = pd.concat([f, observed], axis=1)
    f["common_known"] = f.d06_known & f.p02_known
    return f


def overlay_transition(active, clear_streak, known, d06_high, p02_high, d06_low, p02_low, policy, base_live):
    """任一高风险触发；必须回到各自较低阈值两日才恢复，区间内继续保留削减。"""
    if not base_live or policy in ["PRICE_ALL", "PRICE_COMMON"]:
        return False, 0
    if not known:
        return active, 0
    if policy == "FULL":
        trigger, clear = d06_high or p02_high, d06_low and p02_low
    elif policy == "D06_ONLY":
        trigger, clear = d06_high, d06_low
    elif policy == "P02_ONLY":
        trigger, clear = p02_high, p02_low
    else:
        raise ValueError(f"未登记覆盖层：{policy}")
    if not active:
        return bool(trigger), 0
    clear_streak = clear_streak+1 if clear else 0
    return (False, 0) if clear_streak >= 2 else (True, clear_streak)


def reduced_quantity(baseline_quantity, active):
    """在风险限制后的基准份额上减半，100份基准可降到零但不清除削减状态。"""
    assert baseline_quantity >= 0 and baseline_quantity % 100 == 0
    return (baseline_quantity//200)*100 if active else baseline_quantity


def load_engine():
    spec = importlib.util.spec_from_file_location("factor96_overnight_engine", OUT/"code/account_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simulate(d, x, dividends, policy, capital, cost_name, start, end, engine):
    ids = np.flatnonzero(d.date.between(start, end))
    first, last = int(ids[0]), int(ids[-1])
    account = engine.Account(float(capital))
    cost, cfg = core.COSTS[cost_name], {"lot": 100, "tick": .001, "limit_fraction": .1}
    previous_equity, previous_mark, previous_reserve, peak = capital, float(d.close.iloc[first-1]), 0., capital
    stopped, exit_pending, base_live, active = False, False, False, False
    base_ceiling, clear_streak, cycle_no, base_entry_idx = 0, 0, 0, -1
    events, records, decisions = dividends.to_dict("records"), [], []
    for i in ids:
        row, f = d.iloc[i], x.iloc[i-1]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        old_base_live, old_active, old_streak, old_ceiling = base_live, active, clear_streak, base_ceiling
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0)*event["cash_dividend_per_share"]
                account.receivables[k], recognized = value, recognized+value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash, paid = account.cash+value, paid+value
        ref = float(d.close.iloc[i-1]-row.dividend)
        risk_target = core.target_quantity(engine, account, ref, peak, float(f.es95))
        can_enter = bool(f.price_long and (policy == "PRICE_ALL" or f.common_known))
        q, reason, initial_entry = 0, "无新的可进入价格基准", False
        if base_live:
            exit_pending |= bool(stopped or not f.price_long or i >= base_entry_idx+5)
            if exit_pending:
                q, reason = -old, "价格基准转弱、五日到期或风险停机，继续退出直到完成"
            else:
                base_ceiling = min(base_ceiling, risk_target)
                active, clear_streak = overlay_transition(active, clear_streak, bool(f.common_known),
                    bool(f.d06_high), bool(f.p02_high), bool(f.d06_low), bool(f.p02_low), policy, True)
                target = reduced_quantity(base_ceiling, active)
                q = target-old
                reason = "覆盖层减至风险限制后原份额的一半" if active else "基准持有或两个风险变量低于各自70分位两日后恢复"
        elif not stopped and can_enter and i < last:
            q, reason, initial_entry = risk_target, "价格基准下一开盘首次进入，削减层只管理已有周期", True
        requested_before_open = q
        if q > 0:
            open_risk = core.target_quantity(engine, account, op, peak, float(f.es95))
            if base_live:
                base_ceiling = min(base_ceiling, open_risk)
                q = max(0, min(q, reduced_quantity(base_ceiling, active)-old))
            else:
                q = min(q, open_risk)
        sellable = account.sellable(i)
        trade = engine.execute_order(account, q, op, float(row.previous_close), float(row.dividend), int(i), cost, cfg)
        if initial_entry and trade["filled_quantity"] > 0:
            base_live, base_ceiling, active, clear_streak = True, account.shares, False, 0
            base_entry_idx = int(i)
            cycle_no += 1
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash, paid = account.cash+value, paid+value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = engine.fill_price(close, -1, core.COSTS["STRESS"], .001)
            reserve = account.shares*(close-px)+engine.commission(account.shares, px, core.COSTS["STRESS"])
        equity = account.value(close)-reserve
        price_pnl = old*(op-previous_mark)+account.shares*(close-op)
        error = equity-previous_equity-price_pnl-recognized+trade["commission"]+trade["slippage_cost"]+reserve-previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1-equity/peak
        stopped |= drawdown >= .1
        records.append({"date": day, "idx": i, "policy": policy, "open": op, "mark": close, "cash": account.cash,
            "shares": account.shares, "dividend_receivable": account.receivable(), "terminal_exit_reserve": reserve,
            "equity": equity, "net_return": equity/previous_equity-1, "price_pnl": price_pnl, "dividend_recognized": recognized,
            "dividend_paid": paid, "exposure": account.shares*close/equity, "accounting_error": error, "drawdown": drawdown,
            "risk_stopped": stopped, "sellable_before": sellable, "terminal_unliquidated": bool(i == last and account.shares), **trade})
        decisions.append({"date": day, "origin": d.date.iloc[i-1], "origin_idx": i-1, "policy": policy,
            **{col: f[col] for col in ["stat_date", "stat_idx", "price_long", "common_known", "d06_known", "p02_known",
                "d06_high", "p02_high", "d06_low", "p02_low", "D06", "D06_q90", "D06_q70", "P02", "P02_q90", "P02_q70", "forecast_variance", "overnight_tail_mean"]},
            "can_enter": can_enter, "es95_5d": f.es95, "base_live_before": old_base_live, "base_live": base_live,
            "base_ceiling_before": old_ceiling, "base_ceiling": base_ceiling, "overlay_before": old_active,
            "overlay_active": active, "clear_streak_before": old_streak, "clear_streak": clear_streak,
            "base_exit_pending": exit_pending, "cycle_no": cycle_no, "base_entry_idx": base_entry_idx, "initial_entry": initial_entry,
            "risk_target": risk_target, "reason": reason, "pre_open_request": requested_before_open,
            "requested_quantity": q, "filled_quantity": trade["filled_quantity"]})
        # 半仓因整手取整变成零，不等于价格基准已经结束，削减状态必须继续保存。
        if exit_pending and account.shares == 0:
            base_live, exit_pending, active, clear_streak, base_ceiling = False, False, False, 0, 0
            base_entry_idx = -1
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def prepare():
    assert not (OUT/"source_receipt.json").exists()
    for folder in ["inputs", "code", "source_evidence", "program_before"]:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    sources = {f"inputs/{name}": PREVIOUS/f"inputs/{name}" for name in ["market.parquet", "dividends.csv", "price_features.parquet", "risk_training_records.json", "mature_risk_labels.parquet"]}
    sources.update({"source_evidence/original_v1_launch_failure.json": ROOT/"reports/research/510300_factor96_overnight_risk_overlay_v1/launch_failure_01.json",
        "source_evidence/original_v1_protocol.json": ROOT/"reports/research/510300_factor96_overnight_risk_overlay_v1/protocol.json",
        "source_evidence/current_mandate.json": ROOT/"config/510300_existing_data_training_mandate_v1.json",
        "source_evidence/baseline_protocol.json": PREVIOUS/"protocol.json", "source_evidence/baseline_result.json": PREVIOUS/"result.json",
        "source_evidence/baseline_saved_verification.json": PREVIOUS/"saved_verification_receipt.json",
        "source_evidence/old_session_protocol.json": ROOT/"config/510300_intraday_overnight_increment_v1.json",
        "source_evidence/old_session_report.md": ROOT/"reports/research/510300_intraday_overnight_increment_v1/REPORT.md",
        "source_evidence/old_account_volatility_protocol.json": ROOT/"config/510300_account_volatility_exposure_v1.json",
        "source_evidence/old_account_volatility_outcome.json": ROOT/"reports/research/510300_account_volatility_exposure_v1/acceptance_outcome.json",
        "source_evidence/T13_version_result.json": ROOT/"reports/research/510300_factor96_supply_version_ledger_v1/result.json"})
    previous_freeze = {x["path"]: x["sha256"] for x in json.loads((PREVIOUS/"freeze.json").read_text(encoding="utf-8"))["files"]}
    receipts = []
    for name, source in sources.items():
        shutil.copy2(source, OUT/name)
        checksum = digest(OUT/name)
        if name.startswith("inputs/"):
            assert checksum == previous_freeze[name]
        receipts.append({"source": source.relative_to(ROOT).as_posix(), "snapshot": name, "sha256": checksum})
    for path in (ROOT/"reports/research/510300_factor96_program_v1").iterdir():
        if path.is_file():
            shutil.copy2(path, OUT/"program_before"/path.name)
    save(OUT/"source_evidence/interpretation_and_overlap.json", {"at": now(),
        "baseline_interpretation": "T16草案的已验证基准，本轮明确采用T06已经完成账本复算的同一MA20/五日观察基准；其盈利能力未通过，不称合格盈利基准。结果只回答覆盖层增量和完整账户指标，不预设可部署。",
        "baseline_not_selected_by_return": "不选择旧账户波动模型或其他历史高夏普基准；延续最近T06同一价格基准和风险/成本约束，T06失败不改写。",
        "difference_from_old_session": "旧日内隔夜研究以D20作为收益模型输入；本轮D06为60日隔夜负收益平方占两时段风险比例，P02是当日收益平方相对日前方差预测，用于已有周期的减半/两日恢复。",
        "difference_from_old_volatility": "旧第181轮依据完整账户已实现60日波动连续分配股票预算且联合目标失败；本轮不改该结构，只检验固定价格基准上的分位触发覆盖层。",
        "five_day_resolution": "和T06一致，五日字段作为所有臂基准持有上限，到期优先于恢复。",
        "ewma_choice": "lambda=0.94为本轮事先固定研究参数，不从本轮历史绩效选择，也不声称最优。",
        "T13_remaining": "32份标题候选角色与10条引用已处理；全部事件去重、字段冲突、M06发行缴款日历及其自由流通市值分母仍未齐。"}, True)
    save(OUT/"source_receipt.json", {"at": now(), "sources": receipts, "new_prices_collected": 0, "new_returns_evaluated": False,
        "execution_reference": "https://www.sse.com.cn/assortment/fund/etf/question/", "execution_reference_checked": "2026-09-27",
        "scope": "复用冻结市场/分红与成熟风险输入，T16新的风险变量尚未计算。"}, True)
    print("T16已准备同一价格基准、完整账户和成熟风险输入。", flush=True)


def freeze():
    assert (OUT/"source_receipt.json").exists() and not (OUT/"freeze.json").exists()
    assert json.loads((OUT/"prefreeze_test_receipt.json").read_text(encoding="utf-8"))["exit_code"] == 0
    protocol = {"at": now(), "study_id": STUDY, "primary": "FULL", "new_primary_overlays": 1, "new_entry_strategies": 0,
        "periods": core.PERIODS, "capital": [200000, 20000], "costs": core.COSTS, "annual_days": 242,
        "policies": POLICIES, "primary_lag": 0, "one_day_extra_lag_policies": ["PRICE_COMMON", "FULL"], "expected_accounts": 56,
        "baseline": "与T06完全相同MA20财富收盘条件、最多5个开盘间隔、风险后实际原份额上限；仅已核对账本而非已验证盈利，不能把它称合格入场策略。",
        "returns": "rCO=(开盘+当日每份分红)/前收盘-1；rOC=(收盘+当日分红)/(开盘+当日分红)-1，精确复合为当日持有总收益。分红为应收财富归属，不视作当日可交易现金。",
        "D06": "完整最近60日sum(min(rCO,0)^2)/sum(rCO^2+rOC^2)；分母为零或缺失则未知。另报最差3个rCO的均值，非额外入场条件。",
        "P02": "收益平方/事前预测方差；预测以此前60个完整日收益平方均值初始化，之后v[t]=0.94*v[t-1]+0.06*r[t-1]^2。当前平方收益从不进入当天分母。缺口后重建60日种子。",
        "thresholds": "两个变量各自严格此前252日、至少120有效点的90和70分位；当前值不进入当日阈值。",
        "clock": "主版本当日收盘得到D06/P02及阈值，下一合法开盘仅调整旧份额；额外延迟一交易日仅敏感性，不延迟当前基准和成熟风险预算。",
        "cut": "FULL任一变量严格超过自身P90则减半风险限制后的基准份额，向下取整100；只管理已建立周期，不用风险层另开入场。",
        "restore": "FULL两变量都严格低于各自P70连续2日才恢复风险允许的原份额；单项臂按自身低阈值两日恢复。未知重置恢复计数、保留削减；70至90分位区间不视作恢复。",
        "entry_coverage": "PRICE_COMMON和三个覆盖臂首次进入都要求两个风险变量可知；PRICE_ALL不加该源条件。",
        "risk": "沿用两年成熟5日ES95、2.5%预算、50%最大名义比例、10%跳空5%预算和回撤剩余空间折半；10%回撤后合法开盘退出且不重启。",
        "account": "100份、0.001元、T+1、方向涨跌停、最低佣金5元、分红应收及现金、期末压力退出成本准备、保留全部现金日。",
        "acceptance": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1, "both_main_capitals_stress_required": True},
        "bootstrap": {"block": 20, "draws": 4000, "seed": 20261002, "comparisons": ["PRICE_COMMON", "D06_ONLY", "P02_ONLY"]},
        "stop": "固定OR覆盖层不达标则结束；不调60日、252日、0.94、90/70分位、两日恢复或基准来挽救，不晋升表现最好单项。",
        "independent_forward_validation": False, "goal_achieved": False, "goal_status": "active", "orders_authorized": False}
    for name, source in {"factor96_overnight_risk_overlay_v1_0_1.py": Path(__file__), "factor96_margin_repair_v1.py": Path(core.__file__),
        "account_engine.py": ROOT/"research/intraday_overnight_increment_v1.py", "test_factor96_overnight_risk_overlay_v1_0_1.py": ROOT/"tests/test_factor96_overnight_risk_overlay_v1_0_1.py"}.items():
        shutil.copy2(source, OUT/"code"/name)
    save(OUT/"protocol.json", protocol, True)
    save(OUT/"freeze.json", {"at": now(), "before_new_features_and_strategy_returns": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file()]}, True)
    print("T16风险覆盖层已冻结，主版本与延迟对照共56个账户。", flush=True)


def run():
    frozen = json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    for r in frozen["files"]:
        assert digest(OUT/r["path"]) == r["sha256"], r["path"]
    assert digest(Path(__file__)) == digest(OUT/"code/factor96_overnight_risk_overlay_v1_0_1.py")
    assert digest(Path(core.__file__)) == digest(OUT/"code/factor96_margin_repair_v1.py")
    save(OUT/"run_started.json", {"at": now(), "freeze_sha256": digest(OUT/"freeze.json")}, True)
    market = pd.read_parquet(OUT/"inputs/market.parquet")
    market = market[market.date.le("2025-12-31")].reset_index(drop=True)
    price = pd.read_parquet(OUT/"inputs/price_features.parquet")
    assert market.date.equals(price.date)
    risk = risk_features(market)
    risk.to_parquet(OUT/"risk_features.parquet", index=False)
    div, engine = core.normalize_dividends(pd.read_csv(OUT/"inputs/dividends.csv")), load_engine()
    metrics, annual, stages, accounts, overlays = [], [], [], {}, []
    for lag in [0, 1]:
        x = lag_features(market, price, risk, lag)
        x.to_parquet(OUT/f"daily_features_lag{lag}.parquet", index=False)
        for period, (start, end) in core.PERIODS.items():
            v = x[x.date.between(start, end)]
            stages.append({"period": period, "lag": lag, "days": len(v), "common_known": int(v.common_known.sum()),
                "D06_high": int((v.common_known & v.d06_high).sum()), "P02_high": int((v.common_known & v.p02_high).sum()),
                "either_high": int((v.common_known & (v.d06_high | v.p02_high)).sum()),
                "both_low": int((v.common_known & v.d06_low & v.p02_low).sum())})
            for capital in [200000, 20000]:
                for cost in core.COSTS:
                    for policy in (POLICIES if lag == 0 else ["PRICE_COMMON", "FULL"]):
                        ledger, decisions = simulate(market, x, div, policy, capital, cost, start, end, engine)
                        folder = OUT/f"accounts/{period}/{capital}/{cost}/LAG{lag}/{policy}"
                        folder.mkdir(parents=True, exist_ok=True)
                        ledger.to_parquet(folder/"ledger.parquet", index=False)
                        decisions.to_parquet(folder/"decisions.parquet", index=False)
                        cycles = core.cycle_records(ledger, capital)
                        cycles.to_csv(folder/"cycles.csv", index=False, encoding="utf-8-sig")
                        closed = cycles[cycles.closed.astype(bool)]
                        key = {"period": period, "capital": capital, "cost": cost, "lag": lag, "policy": policy}
                        m = core.metrics(ledger, capital)
                        metrics.append({**key, **m, "closed_cycles": len(closed), "win_rate": closed.profit.gt(0).mean() if len(closed) else None,
                            "point_pass": m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["cagr"] >= .1 and m["max_drawdown"] <= .1})
                        cut = decisions.overlay_active & ~decisions.overlay_before
                        restored = ~decisions.overlay_active & decisions.overlay_before & ~decisions.base_exit_pending
                        overlays.append({**key, "cut_episodes": int(cut.sum()), "restore_episodes": int(restored.sum()),
                            "reduced_decision_days": int(decisions.overlay_active.sum()), "initial_base_cycles": int(decisions.cycle_no.max()),
                            "reduced_zero_share_days": int((decisions.overlay_active & ledger.shares.eq(0)).sum())})
                        prior = capital
                        for year, part in ledger.groupby(ledger.date.dt.year):
                            annual.append({**key, "year": int(year), **core.metrics(part, prior)})
                            prior = float(part.equity.iloc[-1])
                        accounts[period, capital, cost, lag, policy] = ledger
                        if period == "MAIN" and capital == 200000 and cost == "STRESS":
                            print(f"T16 {policy}/额外延迟{lag}：夏普{m['net_sharpe']}，年化{m['cagr']:.3%}，触发削减{int(cut.sum())}次。", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT/"metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT/"annual_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlays).to_csv(OUT/"overlay_counts.csv", index=False, encoding="utf-8-sig")
    save(OUT/"signal_stage_counts.json", stages)
    selected = frame[(frame.period == "MAIN") & (frame.cost == "STRESS") & (frame.lag == 0) & (frame.policy == "FULL")]
    a = accounts["MAIN", 200000, "STRESS", 0, "FULL"]
    n = len(a)
    starts = np.random.default_rng(20261002).integers(0, n, size=(4000, int(np.ceil(n/20))))
    indices = ((starts[:, :, None]+np.arange(20)) % n).reshape(4000, -1)[:, :n]
    np.savez_compressed(OUT/"bootstrap_indices.npz", indices=indices)
    increments = []
    for policy in ["PRICE_COMMON", "D06_ONLY", "P02_ONLY"]:
        diff = a.net_return.to_numpy()-accounts["MAIN", 200000, "STRESS", 0, policy].net_return.to_numpy()
        boot = diff[indices].mean(axis=1)*242
        increments.append({"comparison": "FULL_MINUS_"+policy, "annual_arithmetic_increment": diff.mean()*242,
                           "ci95_low": np.quantile(boot, .025), "ci95_high": np.quantile(boot, .975), "global_selection_adjusted": False})
    save(OUT/"paired_increment.json", increments)
    save(OUT/"result.json", {"study_id": STUDY, "at": now(), "new_accounts": len(frame), "new_primary_overlays": 1,
        "new_entry_strategies": 0, "primary_rows": selected.to_dict("records"), "historical_joint_point_pass": bool(selected.point_pass.all()),
        "increments": increments, "goal_achieved": False, "goal_status": "active", "independent_forward_observations": 0,
        "external_review": "NOT_PERFORMED", "orders_authorized": False})
    print(f"T16固定削减层完成{len(frame)}账户，共同目标通过：{bool(selected.point_pass.all())}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "run": run}[args.action]()
