"""三类信息的日更成熟校准：遵守两年窗口，不择优轮换旧失败策略。"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_FLOOR
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from research import factor96_margin_repair_v1 as core

ROOT = core.ROOT
OUT = ROOT / "reports/research/510300_factor96_daily_state_shrink_v1"
SOURCE = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1"
STUDY = "510300_FACTOR96_DAILY_STATE_SHRINK_V1"
FEATURES = ["A01", "E01", "F01"]
POLICIES = ["NO_STATE", "STATIC_EQUAL", "CALIBRATED", "DISAGREEMENT", "MONITORED", "FULL"]
save, now, digest = core.save, core.now, core.digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_features(market, price, cumulative, members, coverage, margin, lag):
    """日线价格当日收盘形成，外部广度及融资整体滞后一或二交易日。"""
    assert lag in (1, 2)
    dates = pd.DatetimeIndex(market.date)
    assert cumulative.index.equals(dates) and members.index.equals(dates)
    assert cumulative.columns.equals(members.columns)
    valid = members & cumulative.notna()
    counts = valid.sum(axis=1)
    allowed = coverage.set_index("date").aggregation_state.reindex(dates).eq("VIEW_ALLOWED")
    admitted = counts.ge(294) & members.sum(axis=1).eq(300) & allowed
    breadth = ((cumulative.gt(0) & valid).sum(axis=1) / counts.where(counts.gt(0))).where(admitted)
    balance = market[["date"]].merge(margin[["date", "market_rzye"]], how="left", on="date", validate="one_to_one").market_rzye
    financing = (balance / balance.shift(5) - 1).where(balance.rolling(6).count().eq(6))
    wealth = price.wealth.reset_index(drop=True)
    denominator = wealth.diff().abs().rolling(20).sum()
    f = price[["date", "wealth", "es95"]].copy().reset_index(drop=True)
    f["A01"] = (wealth - wealth.shift(20)) / denominator.where(denominator.gt(0))
    f["E01"] = pd.Series(breadth.to_numpy()).shift(lag)
    f["E01_change5_diagnostic"] = pd.Series(breadth.to_numpy()).diff(5).shift(lag)
    f["F01"] = financing.shift(lag)
    f["external_stat_idx"] = pd.Series(np.arange(len(f))).shift(lag).fillna(-1).astype(int)
    f["external_stat_date"] = market.date.shift(lag)
    f["breadth_usable_members"] = pd.Series(counts.to_numpy()).shift(lag)
    f["features_known"] = np.isfinite(f[FEATURES]).all(axis=1)
    f["feature_status"] = np.where(f.features_known, "VIEW_ALLOWED", "NO_VIEW")
    return f


def fit_record(raw, target, ids):
    """固定强度1的单变量岭回归，三个家族分别拟合；截距为训练均值。"""
    x, y = raw[ids], target[ids]
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x - mean) / scale
    intercept = float(y.mean())
    slope = np.mean(z * (y-intercept)[:, None], axis=0) / (np.mean(z*z, axis=0) + 1.)
    variance = float(np.var(y, ddof=1))
    return {"mean": mean.tolist(), "scale": scale.tolist(), "slope": slope.tolist(),
            "intercept": intercept, "variance": variance}


def predict_record(model, row):
    return model["intercept"] + (row-np.array(model["mean"])) / np.array(model["scale"]) * np.array(model["slope"])


def monitor(errors):
    """固定最近20个非重叠成熟残差的单侧CUSUM；过5停用，低于2.5连续5次恢复。"""
    score, paused, clear, stops, restores = 0., False, 0, 0, 0
    for error in errors[-20:]:
        score = max(0., score + float(error) - .25)
        if not paused and score >= 5.:
            paused, clear, stops = True, 0, stops+1
        elif paused:
            clear = clear+1 if score <= 2.5 else 0
            if clear >= 5:
                paused, clear, restores = False, 0, restores+1
    return {"cusum": score, "monitor_paused": paused, "monitor_clear": clear,
            "monitor_stop_count": stops, "monitor_restore_count": restores}


def fraction(mu, variance):
    """固定风险厌恶4的均值方差映射，向下取0/12.5/25/37.5/50五档。"""
    if not np.isfinite([mu, variance]).all() or mu <= 0 or variance <= 0:
        return 0.
    numerator, denominator = Decimal(str(mu)), Decimal(4)*Decimal(str(variance))
    units = int((numerator/denominator*8).to_integral_value(rounding=ROUND_FLOOR))
    return min(4, max(0, units))/8


def forecast(frame, labels, emit_progress=False):
    """外层和全部内层训练均截断在当前日减两日历年，不能引用旧模型的更老标签。"""
    n = len(frame)
    dates, raw = pd.DatetimeIndex(frame.date), frame[FEATURES].to_numpy(float)
    target = labels.gross_return5.to_numpy(float)
    valid = np.isfinite(raw).all(axis=1) & np.isfinite(target)
    index = np.arange(n)
    output = frame.copy()
    for col in ["raw_forecast", "calibrated_forecast", "no_state_forecast", "variance", "disagreement", "positive_agreement", "cusum"]:
        output[col] = np.nan
    output["model_known"], output["monitor_paused"] = False, False
    output["inner_count"], output["training_count"] = 0, 0
    for name in POLICIES:
        output["fraction_"+name] = 0.
    outer, inner, attempts = [], [], {"outer": 0, "inner": 0}
    for t in range(n):
        if not np.isfinite(raw[t]).all():
            continue
        lower = dates[t]-pd.DateOffset(years=2)
        eligible = valid & (dates >= lower) & (index+6 < t)
        ids = np.flatnonzero(eligible)
        if len(ids) < 252:
            continue
        # 全局原点索引模5固定相位；内层结果必须在当前判断日之前成熟。
        validation = np.flatnonzero(eligible & (index % 5 == 0) & (dates >= dates[t]-pd.DateOffset(years=1)))
        current_inner, errors = [], []
        for s in validation:
            training = np.flatnonzero(valid & (dates >= lower) & (index+6 < s))
            if len(training) < 60:
                continue
            model = fit_record(raw, target, training)
            attempts["inner"] += 1
            if model["variance"] <= 1e-12:
                continue
            predicted = predict_record(model, raw[s])
            residual = target[s]-predicted
            sigma = np.sqrt(model["variance"])
            standardized_error = (float(predicted.mean())-float(target[s]))/sigma
            errors.append(standardized_error)
            current_inner.append({"decision_idx": t, "validation_idx": int(s), "validation_exit_idx": int(s+6),
                "training_count": len(training), "training_first_idx": int(training.min()), "training_last_idx": int(training.max()),
                "training_ids_sha256": __import__("hashlib").sha256(training.astype("<i8").tobytes()).hexdigest(),
                "lower": lower, "target": float(target[s]), "predictions": predicted.tolist(),
                "residuals": residual.tolist(), "standardized_error": standardized_error, **model})
        inner.extend(current_inner)
        if len(current_inner) < 20:
            continue
        model = fit_record(raw, target, ids)
        attempts["outer"] += 1
        if model["variance"] <= 1e-12:
            continue
        predictions = predict_record(model, raw[t])
        corrections = np.array([r["residuals"] for r in current_inner]).mean(axis=0)
        shrink = len(current_inner)/(len(current_inner)+60.)
        calibrated = predictions + shrink*corrections
        # 0.28%是两边压力比例成本代理；整手、最小佣金和价位误差在实际账户另计。
        friction = .0028
        no_state_net = model["intercept"]-friction
        raw_net, calibrated_net = float(predictions.mean()-friction), float(calibrated.mean()-friction)
        sigma = np.sqrt(model["variance"])
        disagreement = float(np.std(calibrated, ddof=0)/sigma)
        agreement = float(np.mean(calibrated > friction))
        scaled = calibrated_net*agreement/(1+disagreement) if calibrated_net > 0 else calibrated_net
        state = monitor(errors)
        means = {"NO_STATE": no_state_net, "STATIC_EQUAL": raw_net, "CALIBRATED": calibrated_net,
                 "DISAGREEMENT": scaled, "MONITORED": 0. if state["monitor_paused"] else calibrated_net,
                 "FULL": 0. if state["monitor_paused"] else scaled}
        values = {name: fraction(mu, model["variance"]) for name, mu in means.items()}
        output.loc[t, ["raw_forecast", "calibrated_forecast", "no_state_forecast", "variance", "disagreement", "positive_agreement", "cusum"]] = [
            raw_net, calibrated_net, no_state_net, model["variance"], disagreement, agreement, state["cusum"]]
        output.loc[t, ["model_known", "monitor_paused", "inner_count", "training_count"]] = [True, state["monitor_paused"], len(current_inner), len(ids)]
        for name, value in values.items():
            output.loc[t, "fraction_"+name] = value
        outer.append({"decision_idx": t, "date": dates[t], "lower": lower, "training_indices": ids.tolist(),
            "latest_training_exit_idx": int(ids.max()+6), "inner_validation_indices": [r["validation_idx"] for r in current_inner],
            "raw_predictions": predictions.tolist(), "corrections": corrections.tolist(), "shrink": shrink,
            "calibrated_predictions": calibrated.tolist(), "policy_net_expectations": means, "fractions": values,
            **model, **state})
        if emit_progress and t % 300 == 0:
            print(f"日更校准已处理至{dates[t].date()}，有效判断{len(outer)}日。", flush=True)
    output["model_status"] = np.where(output.model_known, "VIEW_ALLOWED", "NO_VIEW")
    output.attrs["fit_attempt_counts"] = attempts
    return output, outer, inner


def load_engine():
    spec = importlib.util.spec_from_file_location("factor96_daily_shrink_account", OUT/"code/account_engine.py")
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
    entry_idx, entry_wealth, stopped, pending = -1, np.nan, False, False
    records, decisions, events = [], [], dividends.to_dict("records")
    for i in ids:
        row, f = d.iloc[i], x.iloc[i-1]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0)*event["cash_dividend_per_share"]
                account.receivables[k], recognized = value, recognized+value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash, paid = account.cash+value, paid+value
        reference = float(d.close.iloc[i-1]-row.dividend)
        desired = float(f["fraction_"+policy]) if f.model_known else 0.
        target = core.target_quantity(engine, account, reference, peak, float(f.es95), desired)
        entry_before, wealth_before = entry_idx, entry_wealth
        if old:
            pending |= bool(stopped or not f.model_known or desired == 0 or i >= entry_idx+20)
        reason = "依当时成熟预测申请五档仓位"
        if pending or stopped:
            q, reason = -old, "无可用模型、零仓位、20日到期或回撤停机，继续退出"
        elif old:
            q = target-old
            if float(f.wealth) < entry_wealth:
                q, reason = min(q, 0), "低于本周期买入财富价，不增加份额"
        else:
            q = target if i < last else 0
        before = q
        if q > 0:
            q = max(0, min(q, core.target_quantity(engine, account, op, peak, float(f.es95), desired)-old))
            open_wealth = float(f.wealth)*(op+float(row.dividend))/float(d.close.iloc[i-1])
            if old and open_wealth < entry_wealth:
                q, reason = 0, "开盘财富价低于本周期首次进入价，撤去新增请求"
        sellable = account.sellable(i)
        trade = engine.execute_order(account, q, op, float(row.previous_close), float(row.dividend), int(i), cost, cfg)
        if old == 0 and trade["filled_quantity"] > 0:
            entry_idx = int(i)
            entry_wealth = float(f.wealth)*(op+float(row.dividend))/float(d.close.iloc[i-1])
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
        decisions.append({"date": day, "origin_idx": int(i-1), "origin": f.date, "policy": policy,
            "model_known": bool(f.model_known), "feature_status": f.feature_status, "model_status": f.model_status,
            "external_stat_idx": int(f.external_stat_idx), "external_stat_date": f.external_stat_date,
            "desired_fraction": desired, "es95": f.es95, "entry_idx_before": entry_before,
            "entry_wealth_before": wealth_before, "origin_wealth": f.wealth, "entry_idx": entry_idx,
            "reason": reason, "exit_pending": pending, "pre_open_request": before,
            "requested_quantity": q, "filled_quantity": trade["filled_quantity"]})
        if not account.shares:
            entry_idx, entry_wealth, pending = -1, np.nan, False
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def prepare():
    assert not (OUT/"source_receipt.json").exists()
    paths = {
        "inputs/market.parquet": SOURCE/"inputs/market.parquet", "inputs/dividends.csv": SOURCE/"inputs/dividends.csv",
        "inputs/margin.parquet": SOURCE/"inputs/margin.parquet", "inputs/price_features.parquet": SOURCE/"inputs/price_features.parquet",
        "inputs/labels.parquet": SOURCE/"inputs/mature_risk_labels.parquet",
        "inputs/risk_training_records.json": SOURCE/"inputs/risk_training_records.json",
        "inputs/constituent_return20.parquet": SOURCE/"constituent_return20.parquet",
        "inputs/constituent_returns.parquet": SOURCE/"constituent_returns.parquet",
        "inputs/membership_mask.parquet": SOURCE/"membership_mask.parquet",
        "inputs/classified.parquet": SOURCE/"inputs/classified.parquet",
        "inputs/membership.parquet": SOURCE/"inputs/membership.parquet",
        "inputs/daily_coverage.parquet": SOURCE/"inputs/daily_coverage.parquet",
        "source_evidence/previous_source_receipt.json": SOURCE/"source_receipt.json",
        "source_evidence/previous_verification.json": SOURCE/"saved_verification_receipt.json",
        "source_evidence/current_mandate.json": ROOT/"config/510300_existing_data_training_mandate_v1.json",
        "source_evidence/daily_training_authority.json": ROOT/"reports/research/510300_daily_liquidity_insurance_tail_v1/authority_update.json",
        "source_evidence/old_adaptive_allocation.json": ROOT/"config/510300_adaptive_allocation_v1.json",
        "source_evidence/old_regret_experts.json": ROOT/"config/510300_adaptive_regret_experts_v1.json",
        "source_evidence/old_conditional_score.json": ROOT/"config/510300_conditional_score_policy_v1.json",
        "source_evidence/factor_registry.json": core.OUT/"factor_registry.json",
        "source_evidence/strategy_registry.json": core.OUT/"strategy_registry.json",
        "code/account_engine.py": SOURCE/"code/account_engine.py",
    }
    receipts = []
    for name, source in paths.items():
        target = OUT/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        receipts.append({"original": source.relative_to(ROOT).as_posix(), "snapshot": name, "sha256": digest(source)})
    for path in (ROOT/"reports/research/510300_factor96_program_v1").iterdir():
        if path.is_file():
            target = OUT/"program_before"/path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    for path in (SOURCE/"source_evidence").glob("*.json"):
        if any(word in path.name for word in ["margin", "constituent", "return_build", "membership"]):
            shutil.copy2(path, OUT/"source_evidence"/path.name)
    save(OUT/"source_evidence/source_gate_and_priority.json", {"at": now(), "previous_turn_classification": "PROGRESS",
        "T13": "发行检索完成，事件全链和自由流通分母仍未满足。保留NOT_RUN，本轮不继续扩张全文采集。",
        "weight_source_check": "官方规则页仍只证明方法；已检索的接口文档按symbol返回权重快照，没有准入本轮所需全历史权重或自由流通序列；未声称公开历史数据绝对不存在。",
        "reviewed_urls": ["https://github.com/akfamily/akshare/blob/main/docs/data/index/index.md", "https://www.csindex.com.cn/#/services/dataServices"],
        "T18_original_quarterly_seed": "NOT_RUN_CONFLICTS_WITH_CURRENT_DAILY_TRAINING_REQUIREMENT",
        "active_question": "依现行每天更新要求研究三家族日更成熟校准变体，不能声称原季度草案已逐字实现。",
        "distinction": "不使用旧PANIC/REARM/SELECTED_MIX账户或按净值择优。新预测从价格路径效率、点时总回报广度、融资净变化产生；固定岭强度、等权及成熟误差校准。旧价格优化及后悔加权失败不变。",
        "selection_limit": "历史被反复研究，属于开发证据；没有新的独立前向样本。"}, True)
    save(OUT/"source_receipt.json", {"at": now(), "sources": receipts, "new_network_data_requests": 0,
        "historical_first_publication_receipts": "NOT_ESTABLISHED", "new_strategy_returns_evaluated": False}, True)
    print("已冻结来源副本；未计算本轮策略收益。", flush=True)


def freeze():
    assert (OUT/"source_receipt.json").exists() and not (OUT/"freeze.json").exists()
    protocol = {"at": now(), "study_id": STUDY, "primary": "FULL", "reference_strategy": "T18_DAILY_AUTHORITY_VARIANT",
        "original_T18_quarterly_seed": "NOT_RUN；附件为参考，现行用户明确要求两日历年训练及每天更新，故事前记录时钟差异。",
        "question": "固定三家族日更预测，以已成熟误差校准并按分歧及失效监测调整，是否改善无状态及固定等权预测的完整账户。",
        "inputs": FEATURES, "feature_definitions": {"A01": "含分红财富20日有向路径效率，分母0未知",
            "E01": "当前历史成员中有完整20日总回报的上涨比例，至少294/300及VIEW_ALLOWED；缺失不填零；5日变化另存不作为第四输入",
            "F01": "六个连续统计日余额完整时，五日融资余额净变化/起点余额；沿用既有隐含净融资口径"},
        "clock": "A01取当前收盘；E01/F01统计资料主方案滞后一完整交易日，对照多等一日；下一开盘执行",
        "outer_training": "每天仅原点日期>=当前日减2日历年，且j+6<当前索引，三个特征及5日开盘含分红标签完整；至少252行",
        "models": "三个单变量岭回归，训练内标准化ddof0，lambda=1，截距等于训练标签均值；同一当前有效样本、等权组合，不筛选专家",
        "inner_validation": "当前两年下限不变；最近一年固定全局原点索引mod5=0且s+6<t；各s仅用j+6<s且原点>=当前两年下限的行，至少60；至少20个内层预测",
        "calibration": "每个专家加mean(内层实际减预测)*n/(n+60)；所有校准仅成熟原点，不读取当前或未来收益",
        "P05": "校准预测扣0.0028后为正的比例a，校准三预测标准差/训练标签标准差d；正净期望乘a/(1+d)",
        "P06": "最近20个固定五日相位内层预测，以(三预测均值-实际)/内层训练标准差作过高预测损失；g=max(0,g+error-0.25)，g>=5停用；g<=2.5连续5个成熟观察恢复。每日只在当前两年窗口内重算此有界监测，不是无限记忆CUSUM。账户回撤停机永不由此恢复。",
        "controls": POLICIES, "cost_proxy": "净期望扣0.0028压力双边比例成本代理，未含最小佣金及价位；实际账户另按完整成本记账",
        "position": "净期望/(4*训练5日收益方差)，向下取0/0.125/0.25/0.375/0.5，再受既有ES/缺口/回撤预算限制；非正净期望0；五档为当前50%仓位上限内的份额",
        "cycle": "最长20开盘间隔；无模型或零目标后下个合法开盘退出，退出完成前不取消；允许模型增仓但财富价低于本周期首次成交开盘对应财富价时不加仓，不摊平；到期退出当天不重入",
        "risk": "沿用日更两年成熟5日ES95预算2.5%、负10%缺口预算5%、剩余10%回撤空间一半、仓位不超过50%；回撤10%后下一可卖开盘退出且不重启",
        "periods": core.PERIODS, "capital_cny": [200000, 20000], "costs": core.COSTS,
        "planned_accounts": 112, "counting": "6变体*2外部时钟*2资金*2成本*2区间=96；2基准*2资金*2成本*2区间=16",
        "benchmark": "期初50%买入持有及零息现金，基准不受候选风险退出；所有账户保留现金日、未平仓损益及末日压力退出成本准备",
        "annual_days": 242, "annual_252": "仅换算诊断，不据此改判定", "T_plus_1": True, "lot": 100, "tick": .001,
        "statistics": "主期20万元压力FULL对STATIC_EQUAL及NO_STATE配对日收益增量，20日循环区块4000次，种子20260927；保留两个对照，不作全球反复研究已校正声明",
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1,
            "point_pass": "MAIN主时钟压力成本的20万及2万均达到三项目标；历史点估计通过仍不等于独立验证"},
        "stop_condition": "固定经济规则失败后不改窗口、lambda、监测阈值或仓位映射救援；实现错误另留原版并修正版",
        "parameter_search": False, "independent_forward_observations": 0, "external_review": "NOT_PERFORMED",
        "goal_achieved": False, "orders_authorized": False}
    for path in [Path(__file__), ROOT/"research/factor96_margin_repair_v1.py", ROOT/"research/factor96_library_intake_v1.py", ROOT/"tests/test_factor96_daily_state_shrink_v1.py"]:
        shutil.copy2(path, OUT/"code"/path.name)
    save(OUT/"protocol.json", protocol, True)
    frozen = [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    save(OUT/"freeze.json", {"at": now(), "files": frozen}, True)
    print("日更三家族校准协议已冻结，112账户预算，经济参数不扫描。", flush=True)


def run():
    for row in read(OUT/"freeze.json")["files"]:
        assert digest(OUT/row["path"]) == row["sha256"], row
    assert digest(Path(__file__)) == digest(OUT/"code"/Path(__file__).name)
    save(OUT/"run_started.json", {"at": now(), "freeze_sha256": digest(OUT/"freeze.json")}, True)
    d = pd.read_parquet(OUT/"inputs/market.parquet")
    d = d[d.date <= "2025-12-31"].reset_index(drop=True)
    price = pd.read_parquet(OUT/"inputs/price_features.parquet")
    assert d.date.equals(price.date)
    labels = pd.read_parquet(OUT/"inputs/labels.parquet")
    assert len(labels) == len(d)
    cumulative, members = (pd.read_parquet(OUT/"inputs"/name) for name in ["constituent_return20.parquet", "membership_mask.parquet"])
    coverage, margin = (pd.read_parquet(OUT/"inputs"/name) for name in ["daily_coverage.parquet", "margin.parquet"])
    dividends = core.normalize_dividends(pd.read_csv(OUT/"inputs/dividends.csv"))
    frames, fits = {}, {}
    for lag in [1, 2]:
        feature = build_features(d, price, cumulative, members, coverage, margin, lag)
        frame, outer, inner = forecast(feature, labels, True)
        frames[lag] = frame
        frame.to_parquet(OUT/f"daily_forecasts_lag{lag}.parquet", index=False)
        save(OUT/f"outer_models_lag{lag}.json", outer, True)
        save(OUT/f"inner_models_lag{lag}.json", inner, True)
        fits[lag] = {"outer_model_days": len(outer), "inner_validation_rows": len(inner),
            "fit_attempt_counts": frame.attrs["fit_attempt_counts"],
            "univariate_fit_count": 3*sum(frame.attrs["fit_attempt_counts"].values())}
        print(f"时钟{lag}完成{len(outer)}个判断日、{len(inner)}个内层验证记录。", flush=True)
    engine = load_engine()
    metrics, annual, count = [], [], 0
    for period, (start, end) in core.PERIODS.items():
        for capital in [200000, 20000]:
            for cost_name in core.COSTS:
                for lag in [1, 2]:
                    for policy in POLICIES:
                        ledger, decisions = simulate(d, frames[lag], dividends, policy, capital, cost_name, start, end, engine)
                        key = f"{period}_{capital}_{cost_name}_lag{lag}_{policy}"
                        folder = OUT/"accounts"/key
                        folder.mkdir(parents=True, exist_ok=False)
                        ledger.to_parquet(folder/"ledger.parquet", index=False)
                        decisions.to_parquet(folder/"decisions.parquet", index=False)
                        cycles = core.cycle_records(ledger, capital)
                        cycles.to_csv(folder/"cycles.csv", index=False, encoding="utf-8-sig")
                        metric = {"account_id": key, "period": period, "capital": capital, "cost": cost_name, "lag": lag, "policy": policy,
                            "completed_cycles": int(cycles.closed.sum()), **core.metrics(ledger, capital)}
                        metric["point_targets_met"] = bool(metric["net_sharpe"] is not None and metric["net_sharpe"] >= 1.2 and metric["cagr"] >= .1 and metric["max_drawdown"] <= .1)
                        metrics.append(metric)
                        for year, block in ledger.groupby(ledger.date.dt.year):
                            initial = float(block.equity.iloc[0]/(1+block.net_return.iloc[0]))
                            annual.append({"account_id": key, "year": int(year), **core.metrics(block, initial)})
                        count += 1
                for policy in ["BUY_HOLD_50", "CASH"]:
                    ledger, decisions = core.simulate(d, price, dividends, policy, capital, cost_name, start, end, engine)
                    key = f"{period}_{capital}_{cost_name}_benchmark_{policy}"
                    folder = OUT/"accounts"/key
                    folder.mkdir(parents=True, exist_ok=False)
                    ledger.to_parquet(folder/"ledger.parquet", index=False)
                    decisions.to_parquet(folder/"decisions.parquet", index=False)
                    cycles = core.cycle_records(ledger, capital)
                    cycles.to_csv(folder/"cycles.csv", index=False, encoding="utf-8-sig")
                    metrics.append({"account_id": key, "period": period, "capital": capital, "cost": cost_name, "lag": None,
                        "policy": policy, "completed_cycles": int(cycles.closed.sum()), "point_targets_met": False, **core.metrics(ledger, capital)})
                    for year, block in ledger.groupby(ledger.date.dt.year):
                        initial = float(block.equity.iloc[0]/(1+block.net_return.iloc[0]))
                        annual.append({"account_id": key, "year": int(year), **core.metrics(block, initial)})
                    count += 1
                print(f"完整账户已完成{count}/112。", flush=True)
    assert count == 112
    table = pd.DataFrame(metrics)
    table.to_csv(OUT/"metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT/"annual_metrics.csv", index=False, encoding="utf-8-sig")
    main = table[(table.period == "MAIN") & (table.cost == "STRESS") & table.lag.eq(1) & table.policy.eq("FULL")]
    eligible = bool(len(main) == 2 and main.point_targets_met.all())
    rng = np.random.default_rng(20260927)
    primary_ledger = pd.read_parquet(OUT/"accounts/MAIN_200000_STRESS_lag1_FULL/ledger.parquet")
    n = len(primary_ledger)
    indices = ((rng.integers(0, n, size=(4000, int(np.ceil(n/20))))[:, :, None]+np.arange(20)) % n).reshape(4000, -1)[:, :n]
    np.savez_compressed(OUT/"bootstrap_indices.npz", indices=indices)
    pairs = []
    for control in ["STATIC_EQUAL", "NO_STATE"]:
        other = pd.read_parquet(OUT/f"accounts/MAIN_200000_STRESS_lag1_{control}/ledger.parquet")
        assert primary_ledger.date.equals(other.date)
        diff = primary_ledger.net_return.to_numpy()-other.net_return.to_numpy()
        boot = diff[indices].mean(axis=1)*242
        pairs.append({"control": control, "annual_mean_return_difference": float(diff.mean()*242),
            "interval95": np.quantile(boot, [.025, .975]).tolist(), "is_cagr_difference": False})
    save(OUT/"paired_increment.json", pairs, True)
    result = {"at": now(), "study_id": STUDY, "status": "FIXED_DAILY_VARIANT_POINT_PASS_REQUIRES_VALIDATION" if eligible else "FIXED_DAILY_VARIANT_TARGET_FAILED",
        "original_T18_quarterly_seed": "NOT_RUN", "primary_accounts": main.to_dict("records"), "primary_historical_point_pass": eligible,
        "new_accounts": count, "parameter_configurations": 1, "fit_counts": fits,
        "model_available_days": {str(lag): int(frame.model_known.sum()) for lag, frame in frames.items()},
        "primary_increment": pairs, "source_limit": "2026回取历史及保守一/二日时钟，没有逐日首次发布版本；历史曾被研究，不能称独立样本外",
        "external_review": "NOT_PERFORMED", "independent_forward_observations": 0, "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "new_network_data_requests": 0}
    save(OUT/"result.json", result, True)
    print(json.dumps(core.clean(result), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    args = parser.parse_args()
    globals()[args.action]()
