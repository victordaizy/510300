"""510300形态锚定的内部参与、日内隔夜补偿有限增量实验。

本文件只使用归档输入，支持冻结、研究运行和只读保存结果复核。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = Path(__file__).resolve().parents[1] / "reports/research/510300_all_research_abcd_increment_v1"
POLICIES = ("A", "B", "C", "D", "A_COVERAGE", "BUY_HOLD", "CASH")
PERIODS = {"PRIMARY": ("2021-01-04", "2026-08-14"), "RECENT_DIAGNOSTIC": ("2024-08-20", "2026-08-14")}


def parent(root):
    spec = importlib.util.spec_from_file_location("abcd_frozen_parent", root / "code/parent_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(root):
    d = pd.read_parquet(root / "inputs/features.parquet").reset_index(drop=True)
    s = pd.read_parquet(root / "inputs/signals.parquet")
    dv = pd.read_csv(root / "inputs/dividends.csv")
    return d, s, dv


def clocks(root):
    d, signals, _ = load(root)
    raw = pd.read_parquet(root / "inputs/cross_section.parquet")
    quality = pd.read_parquet(root / "inputs/attribution_quality.parquet")
    industries = pd.read_parquet(root / "inputs/industry_attribution.parquet")
    for frame in (raw, quality, industries):
        frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    raw["weight_snapshot_date"] = pd.to_datetime(raw.weight_snapshot_date).dt.strftime("%Y-%m-%d")
    quality["weight_snapshot_date"] = pd.to_datetime(quality.weight_snapshot_date).dt.strftime("%Y-%m-%d")
    # 月末当日快照不用于当日判断；次日可用仍是历史重建假设，不冒充首发时间认证。
    quality["usable"] = quality.valid_for_attribution & (quality.weight_snapshot_date < quality.date)
    good = quality.set_index("date").usable.reindex(d.date).fillna(False).to_numpy(bool)
    returns = raw.pivot(index="date", columns="con_code", values="constituent_return_1d").reindex(d.date)
    returns.index = np.arange(len(d))
    industry_daily = industries.pivot(index="date", columns="industry_l1", values="weighted_return_contribution_1d").reindex(d.date)
    industry_daily.index = np.arange(len(d))
    raw_by_day = dict(tuple(raw.groupby("date")))
    prev = d.close.shift(1)
    intra = (d.close - d.open) / prev
    overnight = (d.open - prev + d.dividend) / prev
    compensation = (intra - (-overnight).clip(lower=0)).rolling(3, min_periods=3).mean()
    session = pd.DataFrame({"idx": np.arange(len(d)), "date": d.date, "intraday_return": intra,
                            "overnight_with_entitled_dividend_return": overnight,
                            "compensation3": compensation, "attribution_usable": good})
    assert np.allclose((intra + overnight).iloc[1:], ((d.close + d.dividend) / prev - 1).iloc[1:], atol=1e-12)
    last = int(d.index[d.date <= PERIODS["PRIMARY"][1]][-1])
    rows, anchors, members = [], [], []
    for sig in signals.to_dict("records"):
        setup, confirm = int(sig["setup_idx"]), int(sig["signal_idx"])
        if confirm > last:
            continue
        reason, group = "READY", []
        cohort = pd.DataFrame()
        if setup < 4 or not good[max(0, setup - 4):setup + 1].all() or sig["setup_date"] not in raw_by_day:
            reason = "NO_VIEW_ANCHOR_FIVE_CONSECUTIVE_DAYS"
        else:
            aggregate = industry_daily.iloc[setup - 4:setup + 1].sum(min_count=5)
            ranked = sorted(((str(k), float(v)) for k, v in aggregate.items() if pd.notna(v) and v > 0), key=lambda x: (-x[1], x[0]))
            group = [k for k, _ in ranked[:3]]
            if not group:
                reason = "NO_VIEW_NO_POSITIVE_ANCHOR_INDUSTRY"
            else:
                at = raw_by_day[sig["setup_date"]]
                cohort = at.loc[at.industry_l1.isin(group) & at.industry_available, ["con_code", "snapshot_weight", "industry_l1", "weight_snapshot_date"]].copy()
                if not len(cohort) or float(cohort.snapshot_weight.sum()) <= 0:
                    reason = "NO_VIEW_EMPTY_ANCHOR_COHORT"
        breadth = np.full(len(d), np.nan)
        coverage = np.full(len(d), np.nan)
        if reason == "READY":
            codes = cohort.con_code.tolist()
            w = cohort.snapshot_weight.to_numpy(float)
            values = returns.reindex(columns=codes).to_numpy(float)
            valid = np.isfinite(values)
            coverage = valid @ w / w.sum()
            positive = (valid & (values > 0)) @ w / w.sum()
            positive[(coverage < 0.98 - 1e-12) | ~good] = np.nan
            breadth = pd.Series(positive).rolling(3, min_periods=3).mean().to_numpy()
            if not np.isfinite(breadth[setup]):
                reason = "NO_VIEW_ANCHOR_MEMBER_COVERAGE"
            for m in cohort.to_dict("records"):
                members.append({"signal_id": sig["signal_id"], "anchor_date": sig["setup_date"], **m})
        initial_b, initial_c = breadth[setup], float(compensation.iloc[setup])
        anchors.append({"signal_id": sig["signal_id"], "family": sig["family"], "setup_idx": setup,
                        "setup_date": sig["setup_date"], "signal_idx": confirm, "signal_date": sig["signal_date"],
                        "core_industries": "|".join(group), "core_member_count": len(cohort),
                        "core_weight": float(cohort.snapshot_weight.sum()) if len(cohort) else None,
                        "initial_breadth3": initial_b, "initial_compensation3": initial_c, "anchor_status": reason})
        for i in range(confirm, last + 1):
            b = float(breadth[i] - initial_b) if reason == "READY" else np.nan
            c = float(compensation.iloc[i] - initial_c)
            available = bool(np.isfinite(b) and np.isfinite(c))
            rows.append({"signal_id": sig["signal_id"], "family": sig["family"], "idx": i, "date": d.date.iloc[i],
                         "B_participation_change": b, "C_compensation_change": c,
                         "B_member_weight_coverage": coverage[i], "joint_available": available,
                         "B": bool(b >= 0) if np.isfinite(b) else False,
                         "C": bool(c >= 0) if np.isfinite(c) else False,
                         "D": bool(b >= 0 and c >= 0) if available else False})
    return pd.DataFrame(rows), pd.DataFrame(anchors), pd.DataFrame(members), session


def simulate(p, d, dividends, signals, context, start, end, policy, cost):
    """沿用母版现金/整手/T+1/股息/失效/最长五日，只改变两个明确决策。"""
    cash, shares, receivable = 200000.0, 0, 0.0
    active, exit_due, pending = None, None, None
    peak = previous = 200000.0
    ledger, trades, decisions, book = [], [], [], []
    by_ex = dict(tuple(dividends.groupby("ex_date")))
    by_signal = {int(k): g.sort_values("family", key=lambda x: x.map({v: j for j, v in enumerate(p.FAMILIES)})).to_dict("records") for k, g in signals.groupby("signal_idx")}
    lookup = {(r["signal_id"], int(r["idx"])): r for r in context.to_dict("records")}
    cycle = 0
    for i in range(start, end + 1):
        r = d.iloc[i]
        accrual = paid = fees = notional = 0.0
        exited_id = ""
        if shares and r.date in by_ex:
            for item in by_ex[r.date].itertuples():
                amount = shares * float(item.cash_dividend_per_share)
                book.append({"payment_date": item.payment_date, "amount": amount})
                receivable += amount
                accrual += amount
                active["dividend_cny"] += amount
        unpaid = []
        for item in book:
            if item["payment_date"] <= r.date:
                cash += item["amount"]
                receivable -= item["amount"]
                paid += item["amount"]
            else:
                unpaid.append(item)
        book = unpaid
        if shares and exit_due is not None and i >= exit_due and i > active["entry_idx"]:
            if p.tradable(d, i, "SELL"):
                px = p.fill_price(float(r.open), "SELL", cost)
                fee = p.commission(shares * px, cost)
                cash += shares * px - fee
                fees += fee
                notional += shares * px
                active.update(exit_idx=i, exit_date=r.date, exit_price=px, exit_fee=fee, quantity=shares,
                              net_pnl=shares * (px - active["entry_price"]) + active["dividend_cny"] - active["entry_fee"] - fee,
                              holding_sessions=i - active["entry_idx"])
                trades.append(active.copy())
                exited_id = active["signal_id"]
                shares, active, exit_due = 0, None, None
            else:
                decisions.append({"date": r.date, "idx": i, "signal_id": active["signal_id"], "decision_type": "EXECUTION", "reason": "SELL_DEFERRED_LIMIT_OR_NO_VOLUME"})
        if not shares and pending is not None:
            sig = pending
            invalid_open = r.ao <= sig["stop_index"]
            if p.tradable(d, i, "BUY") and not invalid_open:
                px = p.fill_price(float(r.open), "BUY", cost)
                shares = p.buy_quantity(cash, px, cost)
                if shares:
                    fee = p.commission(shares * px, cost)
                    cash -= shares * px + fee
                    fees += fee
                    notional += shares * px
                    cycle += 1
                    active = dict(sig, cycle=cycle, entry_idx=i, entry_date=r.date, entry_price=px, entry_fee=fee, dividend_cny=0.0, entry_equity=previous)
                    decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "decision_type": "EXECUTION", "reason": "BUY_FILLED"})
                else:
                    decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "decision_type": "EXECUTION", "reason": "INSUFFICIENT_CASH"})
            else:
                decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "decision_type": "EXECUTION", "reason": "ENTRY_GAP_INVALIDATED" if invalid_open else "BUY_UNFILLED_LIMIT_OR_NO_VOLUME"})
            pending = None
        if shares and exit_due is None:
            view = lookup.get((active["signal_id"], i), {})
            invalid = r.ac < active["stop_index"]
            timed = i - active["entry_idx"] + 1 >= 5
            # 缺值不等于看空。只有当相应变量已知为负，才产生本组额外退出。
            b_bad = bool(np.isfinite(view.get("B_participation_change", np.nan)) and view["B_participation_change"] < 0)
            c_bad = bool(np.isfinite(view.get("C_compensation_change", np.nan)) and view["C_compensation_change"] < 0)
            disabled = (policy == "B" and b_bad) or (policy == "C" and c_bad) or (policy == "D" and (b_bad or c_bad))
            if invalid or timed or disabled:
                exit_due = i + 1
                active["exit_reason"] = "INVALIDATED" if invalid else "INCREMENT_DISABLED" if disabled else "TIME_EXIT"
                active["exit_requested_idx"] = i
                decisions.append({"date": r.date, "idx": i, "signal_id": active["signal_id"], "decision_type": "HOLD", "reason": active["exit_reason"], "B_bad": b_bad, "C_bad": c_bad, **{k: view.get(k) for k in ["B_participation_change", "C_compensation_change", "joint_available"]}})
        equity = cash + shares * float(r.close) + receivable
        peak = max(peak, equity)
        ledger.append({"idx": i, "date": r.date, "cash_cny": cash, "shares": shares, "close": r.close,
                       "receivable_cny": receivable, "dividend_accrual_cny": accrual, "dividend_paid_cny": paid,
                       "fees_cny": fees, "notional_cny": notional, "equity_cny": equity,
                       "daily_return": equity / previous - 1, "drawdown": equity / peak - 1,
                       "exposure": shares * float(r.close) / equity, "active_signal_id": active["signal_id"] if active else "",
                       "exited_signal_id": exited_id, "exit_pending": exit_due is not None})
        assert cash >= -1e-6 and shares % 100 == 0 and receivable >= -1e-6
        previous = equity
        if i < end and i in by_signal:
            for sig in by_signal[i]:
                view = lookup.get((sig["signal_id"], i), {})
                common = bool(view.get("joint_available", False))
                qualify = policy == "A" or (common and (policy == "A_COVERAGE" or bool(view.get(policy, False))))
                reason = "ENTRY_ACCEPTED" if qualify else "NO_VIEW_ENTRY" if not common else "INCREMENT_ENTRY_REJECTED"
                if qualify and (shares or pending is not None):
                    reason = "CAPITAL_OCCUPIED_OR_PRIORITY"
                elif qualify:
                    pending = sig
                decisions.append({"date": r.date, "idx": i, "signal_id": sig["signal_id"], "decision_type": "ENTRY", "reason": reason,
                                  **{k: view.get(k) for k in ["B_participation_change", "C_compensation_change", "joint_available"]}})
    terminal = {"open_position": bool(shares), "shares": shares, "last_signal_id": active["signal_id"] if active else None,
                "pending_sell": exit_due is not None, "terminal_haircut_cny": 0.0,
                "valuation_policy": "与母版相同：期末收盘标记并另计卖出费用储备，不伪造期末已卖出。"}
    if shares:
        px = p.fill_price(float(d.close.iloc[end]), "SELL", cost)
        terminal["terminal_haircut_cny"] = shares * (float(d.close.iloc[end]) - px) + p.commission(shares * px, cost)
    return pd.DataFrame(ledger), pd.DataFrame(trades), pd.DataFrame(decisions), terminal


def mechanism_checks(p):
    """在人工构造路径上检验账户同一性、次日执行、缺值及固定交互。"""
    dates = pd.bdate_range("2020-01-01", periods=12).strftime("%Y-%m-%d")
    close = np.array([4.0, 4.02, 4.04, 4.01, 4.06, 4.1, 4.12, 4.09, 4.13, 4.15, 4.17, 4.2])
    d = pd.DataFrame({"date": dates, "open": close - .01, "high": close + .02, "low": close - .03, "close": close, "ao": close - .01, "ac": close, "dividend": 0., "volume": 100000.})
    d.loc[4, "dividend"] = .03
    dv = pd.DataFrame([{"ex_date": dates[4], "payment_date": dates[6], "cash_dividend_per_share": .03}])
    sig = pd.DataFrame([{"signal_id": "fixture", "family": "BREAKOUT", "signal_idx": 1, "signal_date": dates[1], "setup_idx": 0, "setup_date": dates[0], "stop_index": 3.0}])
    ctx = pd.DataFrame([{"signal_id": "fixture", "idx": i, "joint_available": True, "B": True, "C": True, "D": True, "B_participation_change": .1, "C_compensation_change": .1} for i in range(1, 12)])
    decision = pd.DataFrame([{"idx": i, "family": "BREAKOUT", "PATTERN_ONLY": True} for i in range(12)])
    checks = []
    for cost in p.COSTS:
        old = p.account(d, dv, sig, decision, 0, 11, "PATTERN_ONLY", cost)
        new = simulate(p, d, dv, sig, ctx, 0, 11, "A", cost)
        pd.testing.assert_frame_equal(old[0], new[0][old[0].columns])
        assert math.isclose(old[1].net_pnl.sum(), new[1].net_pnl.sum(), abs_tol=1e-8)
        checks.append(f"{cost}_母版账户逐行一致含分红应收及到账")
    changed = ctx.copy()
    changed.loc[changed.idx >= 2, ["B_participation_change", "B"]] = [-.1, False]
    b = simulate(p, d, dv, sig, changed, 0, 11, "B", "STRESS")
    assert int(b[1].entry_idx.iloc[0]) == 2 and int(b[1].exit_idx.iloc[0]) == 3
    checks.append("买入日已知恶化也只能次日开盘退出且承担第一晚")
    missing = changed.copy()
    missing.loc[missing.idx >= 2, "B_participation_change"] = np.nan
    m = simulate(p, d, dv, sig, missing, 0, 11, "B", "STRESS")
    assert int(m[1].exit_idx.iloc[0]) == 7
    checks.append("持有期间缺值不制造额外卖出")
    missing.loc[missing.idx == 1, "joint_available"] = False
    m = simulate(p, d, dv, sig, missing, 0, 11, "A_COVERAGE", "STRESS")
    assert not len(m[1]) and m[0].shares.eq(0).all()
    checks.append("入场缺值对照保持无交易")
    ctx.loc[ctx.idx == 1, ["C", "D", "C_compensation_change"]] = [False, False, -.1]
    m = simulate(p, d, dv, sig, ctx, 0, 11, "D", "STRESS")
    assert not len(m[1])
    checks.append("D只使用B与C同时满足的一个逻辑交互")
    return checks


def freeze(root):
    p = parent(root)
    if (root / "freeze.json").exists():
        raise RuntimeError("协议已冻结，禁止覆盖。")
    checks = mechanism_checks(p)
    protocol = {
        "study_id": "510300_ALL_RESEARCH_ABCD_INCREMENT_V1", "frozen_at": p.now(),
        "user_authorization": "综合后继续执行附件中的 A/B/C/D 增量实验",
        "primary_period": PERIODS["PRIMARY"], "diagnostic_period": PERIODS["RECENT_DIAGNOSTIC"],
        "common_end_reason": "指数内部输入只到2026-08-14；不得把8月之后的价格单边延长用于四组比较。",
        "account": {"capital_cny": 200000, "annual_days": 252, "cash_yield": 0, "assets": ["510300.SH", "CASH_CNY"], "lot": 100, "tick": .001, "T_plus_1": True, "execution": "当日收盘判断，次一可执行开盘；原形态失效和五日退出优先保留", "costs": p.COSTS, "minimum_commission": 5},
        "A": "V1/V2已冻结PATTERN_ONLY，三个形态定义及其优先顺序全部沿用。",
        "B": {"variable": "形态准备日固定核心群体的三日加权上涨参与率相对准备日的变化", "anchor": "原形态setup_date，不回填交易。此前连续五日归因合格且快照日期严格早于归因日；将五日正贡献合计最高的至多三个行业及准备日成员、权重固定。", "daily": "同一固定成员、固定权重；逐日上涨者权重除以准备日群体总权重，再取最近三个连续交易日均值。", "coverage": "每日固定群体已知收益权重至少98%，归因质量合格；不随当日赢家换成员、不剔除缺值重归一化。", "decision": "确认日变化>=0可入场；持仓后已知变化<0于下一可卖开盘退出。"},
        "C": {"variable": "三日日内补偿相对同一形态准备日的变化", "daily": "I=(close-open)/previous_close；O=(open-previous_close+entitled_cash_dividend)/previous_close；K=mean_3(I-max(-O,0))；C=K_t-K_setup。", "decision": "确认日C>=0可入场；持仓后已知C<0于下一可卖开盘退出。", "scope": "不删除隔夜、不假设当日新买份额可收盘卖出，不是D20/D60方向比较或隔夜下行平方占比。"},
        "D": "仅一个预定交互：B>=0且C>=0时准入；任一已知转负就退出。无加权优化、无新阈值。",
        "missing": "B/C/D入场统一要求两变量可计算；A保持原样；增设A_COVERAGE只使用相同可用性，不使用变量符号，分离缺失筛选贡献。持仓缺值本身不构成卖出。",
        "additional_controls": ["A_COVERAGE", "BUY_HOLD", "CASH"],
        "account_budget": 28, "training_fits": 0, "parameter_grids": 0,
        "novelty": "新问题为既定短形态内的固定起点群体变化和补偿变化；旧固定驱动周期M1/T1、静态广度、D20/D60以及隔夜下行风险的原失败全部继承，未重新运行旧研究。",
        "numerical_acceptance": {"net_cagr": .10, "net_sharpe": 1.2, "max_drawdown": .10, "both_costs": True, "annual_trade_minimum": None},
        "increment_evidence": "主要时期压力账户B-A/C-A/D-A，净日均收益差与夏普差的共同20日区块单侧下界均>0；三候选Bonferroni，单项alpha=.05/3。同时列A_COVERAGE比较，不以较近期点值替代主时期。",
        "uncertainty": {"block_length": 20, "repetitions": 2000, "seed": 20260924, "zero_volatility_sharpe": "UNDEFINED，不用0或无穷代替"},
        "diagnostics": ["每年收益、完整周期、空仓比例和平均暴露", "最大一笔和前三笔利润集中；不删掉原交易后改报策略", "同平均暴露及同波动线性缩放A仅作事后归因", "在相同形态事件内控制trend_z、log(rv20)、family的固定OLS残差关系，属于样本内解释，不作预测资格或交易门"],
        "historical_source_boundary": "月度权重与行业区间为既有历史重建，snapshot_date不是已认证first_public时间；旧质量门只能支持快照归因。该局限禁止将本轮结果升级为实时或独立验证。",
        "independence": "2026-09-24创建，全部市场历史此前已见，独立前向=0；多重比较校正不消除全部既往研究选择。",
        "stop": "一次运行全部固定比较；不因失败改变核心数量、3/5日窗、0阈值、方向、形态、配比和费用；新数值通过也不能自动恢复旧终止策略。",
        "new_market_collection": False, "orders_authorized": False,
    }
    p.save_json(root / "protocol.json", protocol)
    p.save_json(root / "mechanism_checks.json", {"passed": len(checks), "checks": checks, "real_account_evaluations": 0})
    code = root / "code/all_research_abcd_increment_v1.py"
    if Path(__file__).resolve() != code.resolve():
        shutil.copy2(Path(__file__), code)
    label = DEFAULT.parent / "510300_sequential_patterns_regime_v1/results/event_labels.parquet"
    if label.exists():
        shutil.copy2(label, root / "inputs/parent_event_labels.parquet")
    files = [root / "protocol.json", root / "mechanism_checks.json", *sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*.py"))]
    p.save_json(root / "freeze.json", {"frozen_at": p.now(), "status": "FROZEN_BEFORE_NEW_ACCOUNT_RESULTS", "files": [{"path": f.relative_to(root).as_posix(), "sha256": p.digest(f)} for f in files if f.is_file()]})
    print(json.dumps({"冻结": str(root / "freeze.json"), "机制检查": len(checks), "账户预算": 28}, ensure_ascii=False), flush=True)


def check_freeze(root, p):
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for f in frozen["files"]:
        if p.digest(root / f["path"]) != f["sha256"]:
            raise RuntimeError(f"冻结文件变化：{f['path']}")


def conditional_diagnostic(root, p, d, signals, ctx):
    labels = pd.read_parquet(root / "inputs/parent_event_labels.parquet")
    labels = labels.loc[(labels.cost == "STRESS") & (labels.label_status == "MATURE") & (labels.signal_date >= PERIODS["PRIMARY"][0]) & (labels.exit_date <= PERIODS["PRIMARY"][1])].copy()
    c = ctx.merge(signals[["signal_id", "signal_idx"]], on="signal_id")
    c = c.loc[c.idx == c.signal_idx]
    frame = labels.merge(c[["signal_id", "B_participation_change", "C_compensation_change", "joint_available"]], on="signal_id")
    frame = frame.loc[frame.joint_available].copy()
    frame["trend_z"] = d.trend_z.iloc[frame.signal_idx.to_numpy(int)].to_numpy()
    frame["log_rv20"] = np.log(d.rv20.iloc[frame.signal_idx.to_numpy(int)].to_numpy())
    frame = frame.dropna(subset=["trend_z", "log_rv20", "net_return"])
    control = np.column_stack([np.ones(len(frame)), frame.trend_z, frame.log_rv20, (frame.family == "RECLAIM").astype(float), (frame.family == "REPAIR").astype(float)])
    result = []
    if len(frame) > control.shape[1] + 2:
        y = frame.net_return.to_numpy()
        ry = y - control @ np.linalg.lstsq(control, y, rcond=None)[0]
        for name in ["B_participation_change", "C_compensation_change"]:
            x = frame[name].to_numpy()
            rx = x - control @ np.linalg.lstsq(control, x, rcond=None)[0]
            value = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 1e-12 and ry.std() > 1e-12 else None
            result.append({"variable": name, "n": len(frame), "residual_correlation": value, "use": "IN_SAMPLE_DIAGNOSTIC_ONLY_NO_PREDICTION_OR_ACCOUNT_FIT"})
    p.save_csv(root / "results/conditional_event_inputs.csv", frame)
    p.save_json(root / "results/conditional_diagnostic.json", {"results": result, "controls": ["intercept", "trend_z", "log_rv20", "family"], "limitation": "这是同一已见小样本的残差描述，事件重叠、标签期限和选择偏差仍在；不能代替独立增量。"})


def statistics(root, p, all_paths, all_metrics):
    comparisons, attribution = [], []
    for period in PERIODS:
        paths = all_paths[period, "STRESS"]
        names = ["A", "B", "C", "D", "A_COVERAGE"]
        matrix = np.column_stack([p.reserved_returns(paths[k][0], paths[k][3]) for k in names])
        n = len(matrix)
        rng = np.random.default_rng(20260924)
        starts = rng.integers(0, n - 20 + 1, size=(2000, math.ceil(n / 20)))
        indices = (starts[:, :, None] + np.arange(20)).reshape(2000, -1)[:, :n]
        np.savez_compressed(root / f"results/bootstrap_{period}.npz", indices=indices)
        draws = matrix[indices]
        means = draws.mean(axis=1)
        sd = draws.std(axis=1, ddof=1)
        sharpes = np.divide(means * math.sqrt(252), sd, out=np.full_like(means, np.nan), where=sd > 1e-14)
        for cand, base in [("B", "A"), ("C", "A"), ("D", "A"), ("B", "A_COVERAGE"), ("C", "A_COVERAGE"), ("D", "A_COVERAGE"), ("D", "B"), ("D", "C")]:
            j, k = names.index(cand), names.index(base)
            diff = (means[:, j] - means[:, k]) * 252
            ds = sharpes[:, j] - sharpes[:, k]
            valid = ds[np.isfinite(ds)]
            a = .05 / 3 if base == "A" else .05
            comparisons.append({"period": period, "cost": "STRESS", "candidate": cand, "reference": base,
                                "annual_mean_return_difference": float((matrix[:, j].mean() - matrix[:, k].mean()) * 252),
                                "annual_mean_difference_lower": float(np.quantile(diff, a)),
                                "annual_mean_difference_upper": float(np.quantile(diff, 1 - a)),
                                "sharpe_difference_lower": float(np.quantile(valid, a)) if len(valid) else None,
                                "sharpe_difference_upper": float(np.quantile(valid, 1 - a)) if len(valid) else None,
                                "valid_sharpe_bootstraps": len(valid), "total_bootstraps": 2000, "one_sided_alpha": a,
                                "historical_increment_supported": bool(np.quantile(diff, a) > 0 and len(valid) == 2000 and np.quantile(valid, a) > 0),
                                "scope": "已见历史联合区块不确定性，非独立验证"})
        for cand in ["B", "C", "D"]:
            ar = matrix[:, 0]
            cr = matrix[:, names.index(cand)]
            aexp = float(paths["A"][0].exposure.mean())
            cexp = float(paths[cand][0].exposure.mean())
            for kind, scale in [("SAME_AVERAGE_EXPOSURE", cexp / aexp if aexp else np.nan), ("SAME_VOLATILITY", cr.std(ddof=1) / ar.std(ddof=1) if ar.std(ddof=1) else np.nan)]:
                linear = ar * scale
                attribution.append({"period": period, "candidate": cand, "kind": kind, "A_scale": scale,
                                    "candidate_minus_scaled_A_annual_mean": float((cr.mean() - linear.mean()) * 252),
                                    "scope": "事后线性缩放含费用收益，仅解释暴露，不是可执行账户"})
    p.save_csv(root / "results/paired_increment.csv", comparisons)
    p.save_csv(root / "results/exposure_attribution.csv", attribution)
    return comparisons


def run(root):
    p = parent(root)
    check_freeze(root, p)
    if (root / "RUN_STARTED.json").exists():
        raise RuntimeError("本轮一次研究机会已启动；禁止无说明重复运行。")
    p.save_json(root / "RUN_STARTED.json", {"started_at": p.now(), "new_market_collection": False})
    d, signals, dividends = load(root)
    context, anchors, members, session = clocks(root)
    results = root / "results"
    results.mkdir(exist_ok=True)
    context.to_parquet(results / "signal_daily_context.parquet", index=False)
    p.save_csv(results / "signal_anchors.csv", anchors)
    p.save_csv(results / "fixed_core_members.csv", members)
    session.to_parquet(results / "session_components.parquet", index=False)
    trigger = context.merge(signals[["signal_id", "signal_idx", "signal_date"]], on="signal_id")
    trigger = trigger.loc[trigger.idx == trigger.signal_idx]
    p.save_csv(results / "all_signal_inputs.csv", trigger)
    all_paths, comparison, annual, concentration, proof = {}, [], [], [], []
    blank = pd.DataFrame([{"idx": i, "family": fam, "PATTERN_ONLY": True} for i in range(len(d)) for fam in p.FAMILIES])
    for period, (first, last) in PERIODS.items():
        start, end = int(d.index[d.date >= first][0]), int(d.index[d.date <= last][-1])
        for cost in p.COSTS:
            paths = {}
            for policy in POLICIES:
                if policy in ("BUY_HOLD", "CASH"):
                    result = p.account(d, dividends, signals, blank, start, end, policy, cost)
                else:
                    result = simulate(p, d, dividends, signals, context, start, end, policy, cost)
                ledger, trades, decisions, terminal = result
                paths[policy] = result
                m = dict(period=period, cost=cost, policy=policy, **p.metrics(ledger, trades, terminal))
                m["cash_day_fraction"] = float(ledger.shares.eq(0).mean())
                comparison.append(m)
                dst = results / "accounts" / period / cost / policy
                dst.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(dst / "ledger.parquet", index=False)
                trades.to_parquet(dst / "trades.parquet", index=False)
                decisions.to_parquet(dst / "decisions.parquet", index=False)
                p.save_json(dst / "terminal.json", terminal)
                p.save_json(dst / "metrics.json", m)
                adjusted = p.reserved_returns(ledger, terminal)
                temp = ledger.assign(adjusted_return=adjusted, year=ledger.date.str[:4])
                for year, g in temp.groupby("year"):
                    annual.append({"period": period, "cost": cost, "policy": policy, "year": year, "days": len(g),
                                   "net_return": float(np.prod(1 + g.adjusted_return) - 1), "average_exposure": float(g.exposure.mean()),
                                   "completed_cycles_by_entry_year": int(trades.entry_date.str.startswith(year).sum()) if len(trades) else 0})
                if len(trades):
                    pnl = trades.net_pnl.sort_values(ascending=False)
                    total = float(pnl.sum())
                    concentration.append({"period": period, "cost": cost, "policy": policy, "completed_pnl": total,
                                          "largest_cycle_pnl": float(pnl.iloc[0]), "largest_cycle_share": float(pnl.iloc[0] / total) if total > 0 else None,
                                          "top3_cycle_share": float(pnl.head(3).sum() / total) if total > 0 else None,
                                          "pnl_without_largest_cycle_arithmetic_only": float(total - pnl.iloc[0]), "note": "利润归因，不是删除交易后的新账户。"})
                if policy == "A":
                    original = p.account(d, dividends, signals, blank, start, end, "PATTERN_ONLY", cost)
                    pd.testing.assert_frame_equal(original[0], ledger[original[0].columns])
                    assert np.allclose(original[1].net_pnl, trades.net_pnl)
                    proof.append({"period": period, "cost": cost, "same_as_parent": True, "days": len(ledger)})
            all_paths[period, cost] = paths
            print(json.dumps({"已完成": period, "费用": cost, "账户": len(paths)}, ensure_ascii=False), flush=True)
    p.save_csv(results / "account_comparison.csv", comparison)
    p.save_csv(results / "annual.csv", annual)
    p.save_csv(results / "profit_concentration.csv", concentration)
    p.save_json(results / "A_parent_equivalence.json", proof)
    conditional_diagnostic(root, p, d, signals, context)
    paired = statistics(root, p, all_paths, comparison)
    primary_signals = trigger.loc[trigger.signal_date.between(*PERIODS["PRIMARY"])]
    p.save_json(results / "summary.json", {"completed_at": p.now(), "new_policy_accounts": 28,
        "baseline_equivalence_replays": 4, "new_strategy_fits": 0, "in_sample_diagnostic_projection_fits": 3,
        "market_downloads": 0, "strict_forward_observations": 0, "primary_signals": len(primary_signals),
        "common_available_signals": int(primary_signals.joint_available.sum()),
        "primary_two_cost_numerical_pass": [k for k in ["A", "B", "C", "D"] if all(r["numerical_target_pass"] for r in comparison if r["period"] == "PRIMARY" and r["policy"] == k)],
        "primary_increment_supported": [r["candidate"] for r in paired if r["period"] == "PRIMARY" and r["reference"] == "A" and r["historical_increment_supported"]],
        "status": "COMPLETED_FROZEN_EXPLORATORY_COMPARISON", "independent_validation": "NOT_ESTABLISHED", "orders_authorized": False})
    verify(root)


def verify(root):
    p = parent(root)
    check_freeze(root, p)
    count, max_error, metric_error = 0, 0.0, 0.0
    stored = pd.read_csv(root / "results/account_comparison.csv")
    for path in sorted((root / "results/accounts").glob("*/*/*")):
        if not path.is_dir():
            continue
        ledger = pd.read_parquet(path / "ledger.parquet")
        trades = pd.read_parquet(path / "trades.parquet")
        terminal = json.loads((path / "terminal.json").read_text(encoding="utf-8"))
        ident = ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny
        error = float((ident - ledger.equity_cny).abs().max())
        max_error = max(error, max_error)
        assert error < 1e-6 and ledger.shares.mod(100).eq(0).all()
        previous = ledger.equity_cny.shift(1).fillna(200000.)
        assert np.allclose(ledger.equity_cny / previous - 1, ledger.daily_return, atol=1e-12)
        if len(trades):
            assert (trades.exit_idx > trades.entry_idx).all()
            expected = trades.quantity * (trades.exit_price - trades.entry_price) + trades.dividend_cny - trades.entry_fee - trades.exit_fee
            assert np.allclose(expected, trades.net_pnl, atol=1e-6)
        m = p.metrics(ledger, trades, terminal)
        period, cost, policy = path.relative_to(root / "results/accounts").parts
        row = stored.loc[(stored.period == period) & (stored.cost == cost) & (stored.policy == policy)].iloc[0]
        for k in ["net_cagr", "net_sharpe", "max_drawdown", "net_profit_cny", "completed_cycles", "average_exposure"]:
            if m[k] is None:
                assert pd.isna(row[k])
            else:
                delta = abs(float(m[k]) - float(row[k]))
                metric_error = max(metric_error, delta)
                assert delta < 1e-6
        count += len(ledger)
    # 实际截断输入后的变量与确认信号必须不受截断日后数据影响。
    d, signals, dv = load(root)
    cut_checks = []
    for date in ["2021-12-31", "2024-09-24", "2025-12-31"]:
        sub = d.loc[d.date <= date]
        n = len(sub)
        original = p.features(pd.read_parquet(root / "inputs/prices.parquet"), dv)
        truncated = p.features(pd.read_parquet(root / "inputs/prices.parquet").iloc[:n].copy(), dv)
        pd.testing.assert_frame_equal(original.iloc[:n].reset_index(drop=True), truncated.reset_index(drop=True))
        generated = p.detect(truncated)[2]
        expected = signals.loc[signals.signal_idx < n].reset_index(drop=True)
        pd.testing.assert_frame_equal(generated.reset_index(drop=True), expected, check_dtype=False)
        cut_checks.append({"cutoff": date, "rows": n, "signals": len(expected), "pass": True})
    p.save_json(root / "saved_verification.json", {"verified_at": p.now(), "status": "PASS_SAVED_ACCOUNT_METRICS_FREEZE_AND_PRICE_CAUSALITY", "accounts": len(stored), "ledger_rows": count,
        "maximum_accounting_error_cny": max_error, "maximum_metric_error": metric_error, "price_truncation_checks": cut_checks,
        "new_accounts_during_verification": 0, "new_fits": 0, "new_downloads": 0,
        "limitations": "此处重算保存账目和价格状态，不认证历史首发时间或独立投资优势。"})
    print(json.dumps({"保存结果复核": "通过", "账户": len(stored), "逐日记录": count}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="510300全研究综合与固定增量实验")
    parser.add_argument("action", choices=["freeze", "run", "verify"])
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.action](args.root.resolve())
