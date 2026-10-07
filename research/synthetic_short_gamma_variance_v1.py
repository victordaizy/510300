"""仅以ETF及现金检验负Gamma库存、前瞻波动定价与宏观增量。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import ndtr


WORK = Path(__file__).resolve().parents[1]
PARENT = WORK / "reports/research/510300_daily_liquidity_insurance_tail_v1"
OUT = WORK / "reports/research/510300_synthetic_short_gamma_variance_v1"
STUDY = "510300_SYNTHETIC_SHORT_GAMMA_VARIANCE_V1"
PRICE = ["log_var1", "log_var5", "log_var22"]
MACRO = PRICE + ["funding_gap_pp", "credit_acceleration3_pp", "GSPC_z", "VIX_z"]
MODELS = {"PRICE": PRICE, "MACRO": MACRO}
POLICIES = ["FIXED25", "UNCONDITIONAL", "PRICE", "MACRO", "BUY_HOLD"]
COSTS = {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005},
         "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}
START, END = "2021-01-04", "2026-08-14"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def account_engine(root):
    name = "synthetic_short_gamma_frozen_cash_engine"
    spec = importlib.util.spec_from_file_location(name, root / "code/account_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def option_shape(spot, strike, sigma, years):
    """零融资利率、含分红财富坐标下的虚拟看跌价格与卖方正Delta。"""
    assert min(spot, strike, sigma, years) > 0
    scale = sigma * np.sqrt(years)
    d1 = (np.log(spot / strike) + .5 * sigma * sigma * years) / scale
    d2 = d1 - scale
    return float(strike * ndtr(-d2) - spot * ndtr(-d1)), float(ndtr(-d1))


def specification():
    return {
        "study_id": STUDY, "registered_at": now(), "period": [START, END],
        "objective": "完整账户压力净夏普>=1.2、年化>=10%、最大回撤<=10%；未独立验证不宣称目标实现",
        "primary": "MACRO", "primary_comparison": "MACRO_MINUS_PRICE",
        "hypothesis": "若观察到的隐含波动定价高于条件未来实际波动与交易成本，负Gamma形状的现货库存能否获得足够补偿；宏观首先预测波动而非直接预测反弹方向。",
        "critical_limit": "仅ETF与现金，没有交易期权、没有收到权利金。离散、风险截断的Delta形状并非真实卖期权，也不保证复制期权损益或获得期权市场溢价。",
        "assets": ["510300.SH", "CASH_CNY"], "capital_cny": 200000, "cash_rate": 0,
        "prices_and_information": "复用上一轮冻结市场、社融和DR007时钟；美国收益和VIX变化使用已冻结执行日09:00视图。期权IV源日i-2。期权首次发布时钟未认证，仅开发回放。",
        "price_variance_features": PRICE, "macro_additional_features": MACRO[len(PRICE):],
        "target": "未来20个开盘到开盘含分红区间的年化已实现平方对数收益均值；exit_idx=i+20，退出日期必须严格早于拟合日期",
        "training": {"lookback_calendar_years": 2, "frequency": "每交易日", "minimum_common_mature_rows": 252,
                     "estimator": "标准化后截断[-5,5]，带未惩罚截距Ridge(lambda=10)拟合log方差。标准化、残差均只用当时共同训练池。",
                     "mean_prediction": "exp(对数预测)乘训练残差exp的均值，避免把对数尺度中位数当均值",
                     "risk_prediction": "exp(对数预测加训练残差90%分位数)；为开发预测上侧界，不宣称正式90%置信界",
                     "persistence_control": "前一收盘20日年化平方收益均值",
                     "overlap": "20日目标重叠，不把日频行当独立事件；评价另列20个固定非重叠相位"},
        "inventory_shape": {
            "cohort": "从2021-01-04起每20交易日固定重置一次，全部时间边界不依据结果选择。起点前一收盘的含分红财富为虚拟行权价。",
            "initial_units": "每个账户每个固定区间起点09:00权益的50%除当时ETF参考价，作为该区间冻结最大形状份额；无入场权利金现金流。",
            "delta": "虚拟看跌卖方Delta=N(-d1)，每个交易日更新价格、剩余20日区间期限和已知IV。无预测时保留零新增资格。",
            "dividends": "Delta使用含分红财富相对锚点，避免除息机械跳空；真实现金账户按登记、除息、支付分账。",
        },
        "pricing_gate": "PRICE/MACRO使用观察IV和各自条件预测波动，在同一20日等效期限的虚拟平值看跌模型中计算价格差/现价；超过压力双边滑点佣金、双边tick与最低佣金尺度才可持有。它是信号代理，不是可成交报价、到账保费或剩余实际合约的精确定价差。",
        "holding_gate_updates": "门槛每天更新，失去资格即请求清仓；不因看到持仓盈亏临时改变规则。",
        "tail_budget": {"five_day_ES95_equity_fraction": .025, "maximum_inventory_fraction": .5,
                        "gap_stress": -.1, "gap_budget_equity_fraction": .05,
                        "drawdown_reserve": "同上一轮，10%压力损失至多消耗距90%权益峰值剩余空间的一半",
                        "tail_estimate": "最近两年已成熟五日实际下侧5%平均损失，与2.062713乘条件风险波动率乘sqrt(5/252)两者取较大。未满足预测时禁止增加份额。",
                        "drawdown_stop": "收盘回撤>=10%后在下一可卖开盘退出，本轮不重启；不是保证损失上限"},
        "execution": "09:00确定数量；开盘仅复核已确定风险预算和现金。100份、tick0.001、T+1、涨跌停方向拒单。目标与现仓差不足账户10个百分点时保持份额，退出和风险超限不受此带宽限制；沿用既有10个百分点带宽，不搜索。",
        "costs": COSTS, "annual_days": 252,
        "controls": {"FIXED25": "相同信息可见期及风险规则下25%库存", "UNCONDITIONAL": "同一Delta形状，不使用隐含减预测波动门槛",
                     "PRICE": "价格方差预测及同一形状", "BUY_HOLD": "实际现金账户买入持有，非相同风险预算"},
        "missing": "模型或当前IV不可见则不新增，已有份额最多保留五日后退出；所有缺失日计入账户，不用未来标签完整度筛选交易",
        "evaluation": "全部账户、固定2021-2023/2024-终点、全部滚动两年；预测QLIKE和MSE、20个非重叠相位、风险上界突破率；成熟错误单独记录，不追加新的启停优化",
        "inference": "主要配对完整账户增量；20日循环区块2000次，seed=2026092501；开发区间未校正全项目历史选择",
        "advancement": "压力账户三项目标及宏观相对价格年化增量95%下界>0、两个固定子期均正；即使通过仍需随后数据独立检验",
        "deduplication": [
            "上一轮研究预测五日方向分布并承接价格回撤，本轮转为预测未来方差和预先固定的负Gamma库存形状。",
            "旧HAR五多尺度特征、扩展训练、每五日更新未通过其相关性门；保持其拒绝，本轮不导入旧预测或称其已合格。",
            "第109轮GARCH只调节两个旧策略的风险，不检验本轮库存形状；不复活旧组合。",
            "原O4为25日无模型隐含方差减20日HAR的指标研究，本轮使用不同、明确有限的平值价格代理，不称O4已通过。",
            "旧锚定分钟网格因事件资格交集为零而关闭；本轮无000300分钟锚、无该网格事件或成本参数变更。",
        ],
        "new_market_data_collection": False, "orders_authorized": False,
        "parameter_search": False, "independent_validation": False, "review_package": False,
    }


def freeze(root):
    assert not (root / "freeze.json").exists(), "本轮已经冻结，不能覆盖。"
    for name in ["inputs", "code", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    sources = {"market.parquet": PARENT / "results/market.parquet", "information.parquet": PARENT / "results/decision_information.parquet",
               "capital_views.parquet": PARENT / "inputs/capital_views.parquet", "dividends.csv": PARENT / "inputs/dividends.csv",
               "mandate.json": WORK / "config/510300_existing_data_training_mandate_v1.json",
               "parent_result.json": PARENT / "result.json", "old_har_result.json": WORK / "reports/research/510300_har_volatility_20d_report.json"}
    for name, source in sources.items():
        shutil.copy2(source, root / "inputs" / name)
    shutil.copy2(__file__, root / "code/synthetic_short_gamma_variance_v1.py")
    shutil.copy2(PARENT / "code/account_engine.py", root / "code/account_engine.py")
    save(root / "protocol.json", specification(), True)
    save(root / "authority_update.json", {"at": now(), "user_instruction": "请继续，直到完成高夏普目标停止",
        "retained": "两年训练、每日更新、账户尾部风险、20万元、510300与现金；不恢复采集或实盘",
        "old_failures_preserved": True, "objective_complete": False}, True)
    files = [*sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*")), root / "protocol.json", root / "authority_update.json"]
    save(root / "freeze.json", {"at": now(), "before_new_labels_models_accounts": True,
        "files": {p.relative_to(root).as_posix(): digest(p) for p in files},
        "sources": {k: str(v.relative_to(WORK)) for k, v in sources.items()}}, True)
    print("已冻结：未来方差预测、负Gamma库存与宏观增量；尚无新收益结果。", flush=True)


def verify_freeze(root):
    for name, value in json.loads((root / "freeze.json").read_text(encoding="utf-8"))["files"].items():
        assert digest(root / name) == value, f"冻结文件改变：{name}"


def inputs(root, e):
    d = pd.read_parquet(root / "inputs/market.parquet")
    x = pd.read_parquet(root / "inputs/information.parquet")
    v = pd.read_parquet(root / "inputs/capital_views.parquet")
    for frame in [d, x, v]:
        frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
    d = d[d.date.le(END)].reset_index(drop=True)
    x = x[x.date.le(END)].reset_index(drop=True)
    x = x.merge(v[["date", "GSPC_z", "VIX_z", "global_known", "GSPC_available_at", "VIX_available_at"]], on="date", validate="one_to_one")
    assert d.date.tolist() == x.date.tolist()
    for lag in [1, 5, 22]:
        x[f"log_var{lag}"] = np.log((252 * d.total_log.pow(2).rolling(lag).mean()).clip(lower=1e-10)).shift(1)
    x["rv20_persistence"] = (252 * d.total_log.pow(2).rolling(20).mean()).shift(1)
    x["variance_features_known"] = x.funding_known & x.credit_known & x.global_known
    x["variance_features_known"] &= np.isfinite(x[MACRO].to_numpy(float)).all(axis=1)
    known = x.variance_features_known
    for key in ["GSPC_available_at", "VIX_available_at"]:
        assert (pd.to_datetime(x.loc[known, key]) <= pd.to_datetime(x.loc[known, "decision_time"])).all()
    d["wealth"] = np.exp(d.total_log.fillna(0).cumsum())
    div = e.normalize_dividends(pd.read_csv(root / "inputs/dividends.csv"))
    return d, x, div


def make_labels(d, div):
    one = np.full(len(d), np.nan)
    for i in range(len(d) - 1):
        cash = div.loc[div.record_date.ge(d.date.iloc[i]) & div.record_date.lt(d.date.iloc[i + 1]), "cash_dividend_per_share"].sum()
        one[i] = np.log((d.open.iloc[i + 1] + cash) / d.open.iloc[i])
    rows = []
    for i in range(len(d) - 5):
        cash = div.loc[div.record_date.ge(d.date.iloc[i]) & div.record_date.lt(d.date.iloc[i + 5]), "cash_dividend_per_share"].sum()
        row = {"idx": i, "date": d.date.iloc[i], "exit5_idx": i + 5,
               "return5": (d.open.iloc[i + 5] + cash) / d.open.iloc[i] - 1}
        if i + 20 < len(d):
            row.update(exit20_idx=i + 20, exit20_date=d.date.iloc[i + 20], variance20=float(252 * np.square(one[i:i + 20]).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_at(i, x, labels):
    date = x.date.iloc[i]
    lower = date - pd.DateOffset(years=2)
    hist = labels[labels.date.ge(lower) & labels.exit20_idx.lt(i)]
    hist = hist[x.variance_features_known.iloc[hist.idx.to_numpy(int)].to_numpy(bool)]
    receipt = {"idx": i, "date": date, "lower_bound": lower, "training_count": len(hist),
               "latest_training_exit": hist.exit20_idx.max() if len(hist) else None}
    if not x.variance_features_known.iloc[i] or len(hist) < 252:
        return [], [], {**receipt, "status": "NO_VIEW"}
    assert (hist.variance20 > 0).all()
    indices, y = hist.idx.to_numpy(int), np.log(hist.variance20.to_numpy(float))
    recent5 = labels[labels.date.ge(lower) & labels.exit5_idx.lt(i)].return5.to_numpy(float)
    empirical_es = max(0., -float(np.sort(recent5)[:max(1, int(np.ceil(.05 * len(recent5))))].mean()))
    predictions, models = [], []
    for name, columns in MODELS.items():
        raw = x.loc[indices, columns].to_numpy(float)
        mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
        scale[scale < 1e-12] = 1
        z = np.clip((raw - mean) / scale, -5, 5)
        center, ymean = z.mean(axis=0), y.mean()
        beta = np.linalg.solve((z - center).T @ (z - center) + 10 * np.eye(len(columns)), (z - center).T @ (y - ymean))
        intercept = float(ymean - center @ beta)
        residual = y - (intercept + z @ beta)
        smear = float(np.exp(residual).mean())
        upper_residual = float(np.quantile(residual, .9))
        current = np.clip((x.loc[i, columns].to_numpy(float) - mean) / scale, -5, 5)
        log_pred = float(intercept + current @ beta)
        variance = float(np.exp(log_pred) * smear)
        upper = float(max(variance, np.exp(log_pred + upper_residual)))
        tail = max(empirical_es, 2.062713 * np.sqrt(upper * 5 / 252))
        predictions.append({**receipt, "model": name, "variance_prediction": variance,
                            "risk_variance_upper": upper, "variance_persistence": x.rv20_persistence.iloc[i],
                            "empirical_es5": empirical_es, "es95_5": tail})
        models.append({**receipt, "model": name, "features": columns, "training_indices": indices.tolist(),
                       "mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist(), "intercept": intercept,
                       "residual_smear": smear, "residual_upper90": upper_residual})
    return predictions, models, {**receipt, "status": "UPDATED"}


def learn(x, labels, root):
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = fit_at(int(i), x, labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 350 == 0:
            print(f"方差分布已更新至{x.date.iloc[i].date()}。", flush=True)
    result = pd.DataFrame(predictions)
    result.to_parquet(root / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/update_receipts.parquet", index=False)
    save(root / "results/saved_models.json", models)
    return result, models


def budget(nav, peak, es):
    return {"es_budget": .025 * nav, "gap_budget": min(.05 * nav, .5 * max(0., nav - .9 * peak)),
            "notional_budget": .5 * nav, "es95": max(es, 1e-10)}


def valid_quantity(e, target, price, limits):
    px = e.fill_price(price, -1, COSTS["STRESS"], .001)
    exit_cost = target * (price - px) + e.commission(target, px, COSTS["STRESS"])
    notion = target * price
    return (notion <= limits["notional_budget"] + 1e-8 and notion * limits["es95"] + exit_cost <= limits["es_budget"] + 1e-8
            and notion * .1 + exit_cost <= limits["gap_budget"] + 1e-8)


def simulate(e, d, x, div, pred, policy, cost_name):
    first = int(np.flatnonzero(d.date.ge(START))[0])
    last = len(d) - 1
    lookup = {(r.idx, r.model): r for r in pred.itertuples()}
    account = e.Account(200000.)
    previous_nav, previous_mark, peak = 200000., float(d.close.iloc[first - 1]), 200000.
    stopped, last_view = False, first - 6
    shape_units, anchor_wealth = 0., 1.
    events, rows, decisions = div.to_dict("records"), [], []
    cfg, cost = {"lot": 100, "tick": .001, "limit_fraction": .1}, COSTS[cost_name]
    for i in range(first, last + 1):
        row, info = d.iloc[i], x.iloc[i]
        date, op, close = row.date, float(row.open), float(row.close)
        old_shares, recognized, paid = account.shares, 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == date:
                value = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                account.receivables[k] = value
                recognized += value
            if event["payment_date"] < date and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        ref = float(row.previous_close - row.dividend)
        nav = account.value(ref)
        phase = (i - first) % 20
        if phase == 0:
            shape_units = .5 * nav / ref
            anchor_wealth = float(d.wealth.iloc[i - 1])
        model = policy if policy in MODELS else "PRICE"
        prediction = lookup.get((i, model))
        known = prediction is not None and bool(info.option_known)
        plan = {"target_shares": account.shares, "reason": "NO_VIEW_KEEP_OLD", "iv_gate": None, "carry_proxy": None,
                "cost_hurdle": None, "shape_delta": None, "variance_prediction": None, "risk_variance_upper": None,
                "es95": None, "es_budget": None, "gap_budget": None, "notional_budget": None}
        if i == last or (stopped and policy != "BUY_HOLD"):
            plan.update(target_shares=0, reason="TERMINAL_EXIT" if i == last else "DRAWDOWN_STOP")
        elif policy == "BUY_HOLD":
            if i == first:
                qty = e.affordable_quantity(account.cash, e.fill_price(ref, 1, cost, .001), cost, 100)
                plan.update(target_shares=qty, reason="INITIAL_BUY_HOLD")
        elif known:
            last_view = i
            sigma = float(info.iv_atm)
            remaining = (20 - phase) / 252
            relative_spot = float(d.wealth.iloc[i - 1] / anchor_wealth)
            _, delta = option_shape(relative_spot, 1., sigma, remaining)
            # 所有定价判断使用同一个20日期限；不把不同期限期权报价差当可成交套利。
            quote_iv, _ = option_shape(1., 1., sigma, 20 / 252)
            quote_physical, _ = option_shape(1., 1., np.sqrt(prediction.variance_prediction), 20 / 252)
            carry = quote_iv - quote_physical
            hurdle = 2 * (.001 + .0004) + 2 * .001 / ref + 2 * 5 / (.5 * nav)
            gate = carry > hurdle
            es = prediction.es95_5
            if policy in ["FIXED25", "UNCONDITIONAL"]:
                es = max(prediction.empirical_es5, 2.062713 * np.sqrt(prediction.variance_persistence * 5 / 252))
            limits = budget(nav, peak, es)
            raw_target = (.25 * nav / ref) if policy == "FIXED25" else shape_units * delta
            if policy in MODELS and not gate:
                raw_target = 0.
            target = int(raw_target / 100) * 100
            target = min(target, int(limits["notional_budget"] / ref / 100) * 100)
            while target > 0 and not valid_quantity(e, target, ref, limits):
                target -= 100
            # 同一固定带宽仅减少细碎调仓；不阻止风险减仓或失去资格后的退出。
            hold_valid = valid_quantity(e, account.shares, ref, limits)
            if target > 0 and hold_valid and abs(target - account.shares) * ref / nav < .1:
                target = account.shares
            plan.update(target_shares=target, reason="DAILY_SHAPE_AND_RISK", iv_gate=gate, carry_proxy=carry,
                        cost_hurdle=hurdle, shape_delta=delta, variance_prediction=prediction.variance_prediction,
                        risk_variance_upper=prediction.risk_variance_upper, **limits)
        elif i - last_view >= 5:
            plan.update(target_shares=0, reason="NO_VIEW_EXPIRED")
        request = int(plan["target_shares"] - account.shares)
        preopen_request = request
        if request > 0 and policy != "BUY_HOLD":
            while request > 0 and not valid_quantity(e, account.shares + request, op, plan):
                request -= 100
        execution = e.execute_order(account, request, op, float(row.previous_close), float(row.dividend), i, cost, cfg)
        mark = op if i == last else close
        if i != last:
            for k, event in enumerate(events):
                if event["payment_date"] == date and k in account.receivables:
                    value = account.receivables.pop(k)
                    account.cash += value
                    paid += value
                if event["record_date"] == date:
                    account.entitlements[k] = account.shares
        nav_end = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav_end - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, nav_end)
        dd = 1 - nav_end / peak
        if policy != "BUY_HOLD" and dd >= .1:
            stopped = True
        rows.append({"date": date, "idx": i, "policy": policy, "cost": cost_name,
                     "equity": nav_end, "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                     "dividend_recognized": recognized, "dividend_paid": paid, "price_pnl": price_pnl,
                     "mark": mark, "net_return": nav_end / previous_nav - 1, "pnl": nav_end - previous_nav,
                     "exposure": account.shares * mark / nav_end, "drawdown": dd, "risk_stopped": stopped,
                     "virtual_option_cashflow": 0., "accounting_error": error,
                     "terminal_unliquidated": bool(i == last and account.shares), **execution})
        decisions.append({"date": date, "idx": i, "policy": policy, "cost": cost_name, "phase": phase,
                          "shape_max_units": shape_units, "known": known, "preopen_requested": preopen_request,
                          "gap_checked_request": request, "filled_quantity": execution["filled_quantity"],
                          "reference_price": ref, "nav_before": nav, **plan})
        previous_nav, previous_mark = nav_end, mark
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def metrics(e, ledger):
    r = ledger.net_return.to_numpy(float)
    value = e.return_metrics(r, 252)
    sd = r.std(ddof=0)
    value.update(ending_equity=ledger.equity.iloc[-1], mean_exposure=ledger.exposure.mean(),
                 cost_cny=(ledger.commission + ledger.slippage_cost).sum(),
                 cash_days=int(ledger.shares.eq(0).sum()), buy_adjustments=int(ledger.filled_quantity.gt(0).sum()),
                 sell_adjustments=int(ledger.filled_quantity.lt(0).sum()), worst_day=r.min(),
                 daily_skew=float(((r - r.mean()) ** 3).mean() / sd ** 3) if sd > 0 else None,
                 stopped=bool(ledger.risk_stopped.any()),
                 same_share_path_gross_pnl=(ledger.price_pnl + ledger.dividend_recognized).sum())
    return value


def prediction_evaluation(pred, lab):
    f = pred.merge(lab[["idx", "exit20_idx", "variance20"]], on="idx", how="left", validate="many_to_one")
    f["qlike"] = f.variance20 / f.variance_prediction - np.log(f.variance20 / f.variance_prediction) - 1
    f["qlike_persistence"] = f.variance20 / f.variance_persistence - np.log(f.variance20 / f.variance_persistence) - 1
    f["mse"] = (f.variance20 - f.variance_prediction) ** 2
    f["upper_breach"] = (f.variance20 > f.risk_variance_upper).where(f.variance20.notna())
    rows, phases, monitor = {}, [], []
    for model, all_rows in f.groupby("model"):
        g = all_rows.dropna(subset=["variance20"])
        rows[model] = {"mature_predictions": len(g), "qlike": g.qlike.mean(), "qlike_persistence": g.qlike_persistence.mean(),
                       "mse": g.mse.mean(), "upper90_breach_rate": g.upper_breach.astype(float).mean()}
        for phase in range(20):
            h = g[g.idx.mod(20).eq(phase)]
            phases.append({"model": model, "phase": phase, "observations": len(h), "qlike": h.qlike.mean(),
                           "qlike_persistence": h.qlike_persistence.mean()})
        for row in all_rows.itertuples():
            history = all_rows[all_rows.exit20_idx.lt(row.idx)].tail(126)
            monitor.append({"date": row.date, "idx": row.idx, "model": model, "mature_prior": len(history),
                            "max_mature_exit": history.exit20_idx.max() if len(history) else None,
                            "mean_qlike": history.qlike.mean(), "upper_breach_rate": history.upper_breach.astype(float).mean(),
                            "used_for_new_switch_rule": False})
    return f, rows, pd.DataFrame(phases), pd.DataFrame(monitor)


def tests(e):
    p_low, delta_low = option_shape(.9, 1, .2, 20 / 252)
    p_high, delta_high = option_shape(1.1, 1, .2, 20 / 252)
    assert delta_low > delta_high and p_low > p_high
    price_small, _ = option_shape(1, 1, .1, 20 / 252)
    price_big, _ = option_shape(1, 1, .3, 20 / 252)
    assert price_big > price_small
    a = e.Account(200000.)
    c = {"lot": 100, "tick": .001, "limit_fraction": .1}
    e.execute_order(a, 1000, 4., 4., 0., 2, COSTS["STRESS"], c)
    assert e.execute_order(a, -1000, 4., 4., 0., 2, COSTS["STRESS"], c)["filled_quantity"] == 0
    assert a.cash < 200000.
    assert budget(185000., 200000., .04)["gap_budget"] < budget(200000., 200000., .04)["gap_budget"]
    return {"falling_price_increases_shape_delta": True, "higher_volatility_raises_virtual_put_price": True,
            "no_premium_credited": True, "T_plus_one": True, "drawdown_reserve_contracts": True}


def verify_saved(root):
    verify_freeze(root)
    e = account_engine(root)
    x = pd.read_parquet(root / "results/information.parquet")
    labels = pd.read_parquet(root / "results/labels.parquet").set_index("idx")
    predictions = pd.read_parquet(root / "results/predictions.parquet")
    lookup = {(r.idx, r.model): r for r in predictions.itertuples()}
    models = json.loads((root / "results/saved_models.json").read_text(encoding="utf-8"))
    for record in models:
        i = record["idx"]
        sample = labels.loc[record["training_indices"]]
        assert sample.exit20_idx.max() < i
        assert (sample.date >= x.date.iloc[i] - pd.DateOffset(years=2)).all()
        z = np.clip((x.loc[i, record["features"]].to_numpy(float) - record["mean"]) / record["scale"], -5, 5)
        logvalue = record["intercept"] + z @ record["beta"]
        mean = np.exp(logvalue) * record["residual_smear"]
        risk = max(mean, np.exp(logvalue + record["residual_upper90"]))
        actual = lookup[(i, record["model"])]
        np.testing.assert_allclose([mean, risk], [actual.variance_prediction, actual.risk_variance_upper], rtol=1e-12, atol=1e-12)
    prefix = []
    for cutoff in ["2023-12-29", "2025-12-31"]:
        eligible = x[x.date.le(cutoff) & x.idx.isin(predictions.idx)]
        if not len(eligible):
            continue
        i = int(eligible.idx.iloc[-1])
        ps, _, _ = fit_at(i, x.iloc[:i + 1], labels.reset_index()[labels.exit5_idx.to_numpy() < i].copy())
        for p in ps:
            np.testing.assert_allclose(p["variance_prediction"], lookup[(i, p["model"])].variance_prediction, rtol=0, atol=1e-12)
        prefix.append(str(x.date.iloc[i].date()))
    max_error, accounts = 0., 0
    for path in (root / "accounts").glob("*/*_ledger.parquet"):
        f = pd.read_parquet(path)
        expected_cash = 200000. - (f.filled_quantity * f.fill_price.fillna(0)).cumsum() - f.commission.cumsum() + f.dividend_paid.cumsum()
        np.testing.assert_allclose(f.cash, expected_cash, rtol=0, atol=1e-7)
        np.testing.assert_allclose(f.filled_quantity.cumsum(), f.shares, rtol=0, atol=0)
        np.testing.assert_allclose(f.cash + f.shares * f.mark + f.dividend_receivable, f.equity, rtol=0, atol=1e-7)
        assert f.virtual_option_cashflow.eq(0).all()
        assert not f.terminal_unliquidated.iloc[-1]
        max_error = max(max_error, f.accounting_error.abs().max())
        accounts += 1
    receipt = {"at": now(), "saved_predictions_recomputed": len(models), "prefix_checks": prefix,
               "strict_calendar_two_years_and_maturity": True, "cash_accounts_reconciled": accounts,
               "all_virtual_option_cashflows_zero": True, "max_accounting_error": max_error, "mechanism_checks": tests(e)}
    save(root / "verification.json", receipt)
    return receipt


def run(root):
    verify_freeze(root)
    save(root / "RUN_STARTED.json", {"started_at": now()}, True)
    e = account_engine(root)
    save(root / "mechanism_checks.json", tests(e))
    d, x, div = inputs(root, e)
    labels = make_labels(d, div)
    x.to_parquet(root / "results/information.parquet", index=False)
    labels.to_parquet(root / "results/labels.parquet", index=False)
    pred, models = learn(x, labels, root)
    scored, scores, phases, monitor = prediction_evaluation(pred, labels)
    scored.to_parquet(root / "results/prediction_evaluation.parquet", index=False)
    phases.to_parquet(root / "results/nonoverlap_prediction_phases.parquet", index=False)
    monitor.to_parquet(root / "results/mature_error_monitor.parquet", index=False)
    save(root / "results/prediction_metrics.json", scores)
    accounts, all_metrics, yearly, windows = {}, [], [], []
    for cost in COSTS:
        folder = root / "accounts" / cost
        folder.mkdir(exist_ok=True)
        for policy in POLICIES:
            ledger, decisions = simulate(e, d, x, div, pred, policy, cost)
            ledger.to_parquet(folder / f"{policy}_ledger.parquet", index=False)
            decisions.to_parquet(folder / f"{policy}_decisions.parquet", index=False)
            accounts[(cost, policy)] = ledger
            all_metrics.append({"cost": cost, "policy": policy, **metrics(e, ledger)})
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost, "policy": policy, "year": year, **metrics(e, group)})
            if cost == "STRESS":
                dates = pd.DatetimeIndex(ledger.date)
                for i, end in enumerate(dates):
                    lower = end - pd.DateOffset(years=2)
                    if lower < dates[0]:
                        continue
                    subset = ledger.iloc[dates.searchsorted(lower):i + 1]
                    windows.append({"policy": policy, "start": subset.date.iloc[0], "end": end,
                                    **e.return_metrics(subset.net_return.to_numpy(float), 252)})
            print(f"已完成{cost}／{policy}实际ETF现金账户。", flush=True)
    window = pd.DataFrame(windows)
    window.to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    save(root / "results/yearly_metrics.json", yearly)
    save(root / "results/account_metrics.json", all_metrics)
    macro, price = accounts[("STRESS", "MACRO")], accounts[("STRESS", "PRICE")]
    a, b = macro.net_return.to_numpy(float), price.net_return.to_numpy(float)
    rng = np.random.default_rng(2026092501)
    differences = [float((a[ix] - b[ix]).mean() * 252) for ix in [e.block_indices(rng, len(a), 20) for _ in range(2000)]]
    halves = {}
    for name, low, high in [("2021_2023", START, "2023-12-31"), ("2024_END", "2024-01-01", END)]:
        mask = macro.date.between(low, high).to_numpy()
        halves[name] = {"annual_arithmetic_increment": float((a[mask] - b[mask]).mean() * 252),
                       "MACRO": e.return_metrics(a[mask], 252), "PRICE": e.return_metrics(b[mask], 252)}
    compare = {"annual_arithmetic_increment": float((a - b).mean() * 252), "ci95": np.quantile(differences, [.025, .975]),
               "fixed_halves": halves, "development_only": True}
    primary = next(m for m in all_metrics if m["cost"] == "STRESS" and m["policy"] == "MACRO")
    point = primary["net_sharpe"] is not None and primary["net_sharpe"] >= 1.2 and primary["annualized_return"] >= .1 and primary["max_drawdown"] >= -.1
    incremental = compare["ci95"][0] > 0 and all(v["annual_arithmetic_increment"] > 0 for v in halves.values())
    recent = window[window.policy.eq("MACRO")]
    result = {"study_id": STUDY, "completed_at": now(), "status": "DISCOVERY_CANDIDATE_REQUIRES_FORWARD" if point and incremental else "FROZEN_NO_QUALIFIED_SYNTHETIC_SHORT_GAMMA",
              "primary": primary, "all_metrics": all_metrics, "prediction_metrics": scores, "comparison": compare,
              "model_updates": len(models), "updated_days": int(pred.idx.nunique()),
              "first_prediction_date": pred.date.min(), "last_prediction_date": pred.date.max(),
              "rolling_two_years": {"windows": len(recent), "median_sharpe": recent.net_sharpe.median(),
                 "max_sharpe": recent.net_sharpe.max(), "min_sharpe": recent.net_sharpe.min(),
                 "max_cagr": recent.annualized_return.max(), "min_cagr": recent.annualized_return.min(),
                 "simultaneous_target_windows": int((recent.net_sharpe.ge(1.2) & recent.annualized_return.ge(.1) & recent.max_drawdown.ge(-.1)).sum())},
              "point_target_met": point, "increment_gate_met": incremental, "independent_validation": False,
              "goal_achieved": False, "new_collection": False, "orders_authorized": False}
    save(root / "result.json", result, True)
    receipt = verify_saved(root)
    print(json.dumps(clean({"状态": result["status"], "主账户": primary, "方差预测": scores, "账户增量": compare, "验证": receipt}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify_saved}[args.command](args.root)
