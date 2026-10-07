"""T10固定检验：海外定价之后，A股完整首日是否仍反应不足。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import requests

from research import factor96_margin_repair_v1 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_crossborder_absorption_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1/completed_run"
STUDY = "510300_FACTOR96_CROSSBORDER_ABSORPTION_V1"
POLICIES = ["PRICE_COMMON", "POSITIVE", "FULL", "OVERREACTION"]
PREDICTORS = ["ashr_log_return", "fx_log_return", "prior_a_log_return"]
save, now, digest = core.save, core.now, core.digest
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/ASHR?period1=1356998400&period2=1767225600&interval=1d&events=div%2Csplits%2CcapitalGains"
REFERENCES = {
    "ashr_prospectus.pdf": "https://etf.dws.com/en-us/AssetDownload/Inline/ce51b065-fc18-496f-9b88-8996a37d16b3/CHINA-1-Prospectus.pdf",
    "nyse_core_hours.html": "https://www.nyse.com/markets/hours-calendars",
    "cfets_fx_clock.pdf": "https://www.chinamoney.com.cn/dqs/cm-s-notice-query/fileDownLoad.do?contentId=384571&mode=open&priority=0",
}


def parse_ashr(payload: dict) -> pd.DataFrame:
    """只用原始收盘与已发生现金分配重建收益；不使用调整收盘。"""
    raw = payload["chart"]["result"][0]
    meta = raw["meta"]
    assert meta["symbol"] == "ASHR" and meta["currency"] == "USD"
    assert meta["exchangeTimezoneName"] == "America/New_York"
    assert not raw.get("events", {}).get("splits"), "存在拆股，须重新通过来源门"
    assert not raw.get("events", {}).get("capitalGains"), "存在单列资本分配，不能忽略或重复计入"
    dates = pd.to_datetime(raw["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    quote = raw["indicators"]["quote"][0]
    frame = pd.DataFrame({"date": dates, **{k: quote[k] for k in ["open", "high", "low", "close", "volume"]}})
    assert frame.date.is_unique and frame.date.is_monotonic_increasing
    assert np.isfinite(frame[["open", "high", "low", "close"]]).all().all()
    assert frame[["open", "high", "low", "close"]].gt(0).all().all()
    cash = {}
    for item in raw.get("events", {}).get("dividends", {}).values():
        date = pd.to_datetime(item["date"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        assert date in set(frame.date) and date not in cash
        cash[date] = float(item["amount"])
    frame["cash_distribution"] = frame.date.map(cash).fillna(0.)
    frame["log_return"] = np.log((frame.close + frame.cash_distribution) / frame.close.shift())
    # 统一等待至纽约17:00，兼容夏令时；提前收盘日也不提前使用。
    frame["available_at"] = (pd.DatetimeIndex(frame.date).tz_localize("America/New_York") + pd.Timedelta(hours=17)).tz_convert("Asia/Shanghai")
    return frame


def align_external(market: pd.DataFrame, us: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """每个A股开盘前09:20，仅取已完成的美国时段和当日已公布中间价。"""
    dates = pd.DatetimeIndex(market.date)
    clock = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=20)
    us_clock = pd.DatetimeIndex(us.available_at)
    rows = []
    for t, day in enumerate(dates):
        latest = int(us_clock.searchsorted(clock[t], side="right") - 1)
        baseline = int(us_clock.searchsorted(clock[t-1], side="right") - 1) if t else -1
        valid = baseline >= 0 and latest > baseline
        if valid:
            valid = (day - us.date.iloc[latest]).days <= 7 and (dates[t-1] - us.date.iloc[baseline]).days <= 7
        values = us.log_return.iloc[baseline+1:latest+1] if valid else pd.Series(dtype=float)
        valid = bool(valid and values.notna().all())
        rows.append({"date": day, "forecast_at": clock[t], "baseline_us_date": us.date.iloc[baseline] if baseline >= 0 else pd.NaT,
                     "latest_us_date": us.date.iloc[latest] if latest >= 0 else pd.NaT,
                     "latest_us_available_at": us.available_at.iloc[latest] if latest >= 0 else pd.NaT,
                     "us_session_count": max(0, latest-baseline) if baseline >= 0 else 0,
                     "ashr_log_return": float(values.sum()) if valid else np.nan})
    aligned = pd.DataFrame(rows)
    fix = fx.copy().sort_values("available_at")
    fix["available_at"] = pd.to_datetime(fix.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    fix = fix.rename(columns={"date": "fx_date", "first_release_value": "fx_rate", "available_at": "fx_available_at"})
    joined = pd.merge_asof(aligned, fix[["fx_date", "fx_rate", "fx_available_at"]],
                           left_on="forecast_at", right_on="fx_available_at", direction="backward")
    same_day = joined.fx_date.eq(joined.date)
    joined["fx_log_return"] = np.log(joined.fx_rate / joined.fx_rate.shift()).where(same_day & same_day.shift(fill_value=False))
    joined["a_log_return"] = np.log((market.close + market.dividend) / market.close.shift())
    joined["a_gap_log_return"] = np.log((market.open + market.dividend) / market.close.shift())
    joined["prior_a_log_return"] = joined.a_log_return.shift()
    joined["external_known"] = np.isfinite(joined[PREDICTORS]).all(axis=1)
    return joined


def rolling_response_model(aligned: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """两自然年OLS逐日更新，当前完整首日和未来持有期均不参与估计。"""
    frame = aligned.copy()
    matrix = np.column_stack([np.ones(len(frame)), frame[PREDICTORS].to_numpy(float)])
    target = frame.a_log_return.to_numpy(float)
    gap = frame.a_gap_log_return.to_numpy(float)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(target) & np.isfinite(gap)
    for col in ["predicted_response", "response_sigma", "predicted_gap", "gap_sigma"]:
        frame[col] = np.nan
    frame["training_n"], frame["training_max_idx"] = 0, -1
    frame["model_known"] = False
    records = []
    for t in range(len(frame)):
        # 当前首日结果未可得也不影响开盘前模型输出。
        if not np.isfinite(matrix[t]).all():
            continue
        lower = frame.date.iloc[t] - pd.DateOffset(years=2)
        ids = np.flatnonzero(valid & (np.arange(len(frame)) < t) & frame.date.ge(lower).to_numpy())
        if len(ids) < 252:
            continue
        x = matrix[ids]
        beta, _, rank, _ = np.linalg.lstsq(x, target[ids], rcond=None)
        beta_gap, _, rank_gap, _ = np.linalg.lstsq(x, gap[ids], rcond=None)
        if rank != 4 or rank_gap != 4:
            continue
        sigma = float(np.sqrt(np.sum((target[ids] - x @ beta)**2) / (len(ids)-4)))
        sigma_gap = float(np.sqrt(np.sum((gap[ids] - x @ beta_gap)**2) / (len(ids)-4)))
        if sigma <= 1e-12:
            continue
        frame.loc[t, ["predicted_response", "response_sigma", "predicted_gap", "gap_sigma"]] = [matrix[t] @ beta, sigma, matrix[t] @ beta_gap, sigma_gap]
        frame.loc[t, ["training_n", "training_max_idx", "model_known"]] = [len(ids), int(ids.max()), True]
        records.append({"idx": t, "date": frame.date.iloc[t], "forecast_at": frame.forecast_at.iloc[t],
                        "training_indices": ids.tolist(), "beta": beta.tolist(), "beta_gap": beta_gap.tolist(),
                        "sigma": sigma, "sigma_gap": sigma_gap})
    frame["response_residual"] = frame.a_log_return - frame.predicted_response
    frame["response_z"] = frame.response_residual / frame.response_sigma
    frame["price_signal"] = frame.model_known & frame.a_log_return.gt(0)
    frame["external_positive"] = frame.ashr_log_return.gt(0)
    frame["underreaction"] = frame.response_z.lt(-1)
    frame["overreaction"] = frame.response_z.gt(1)
    return frame, records


def lag_features(model: pd.DataFrame, price: pd.DataFrame, lag: int) -> pd.DataFrame:
    """敏感性等待一日不移动风险时钟，并要求等待后仍未收盘破首日低点。"""
    assert lag in (0, 1)
    frame = price[["date", "wealth", "low_w", "es95", *core.STATE]].copy()
    event = model.drop(columns="date").copy()
    event["source_day"] = model.date
    event["source_idx"] = np.arange(len(model))
    event["frozen_stop"] = price.low_w.to_numpy()
    if lag:
        event = event.shift(lag)
    for col in ["external_known", "model_known", "price_signal", "external_positive", "underreaction", "overreaction"]:
        event[col] = event[col].fillna(False).astype(bool)
    event["source_idx"] = event.source_idx.fillna(-1).astype(int)
    frame = pd.concat([frame, event], axis=1)
    frame["price_signal"] &= frame.wealth.ge(frame.frozen_stop)
    return frame


def signal_for(frame: pd.Series, policy: str) -> bool:
    if policy == "PRICE_COMMON":
        return bool(frame.price_signal)
    if policy == "POSITIVE":
        return bool(frame.price_signal and frame.external_positive)
    if policy == "FULL":
        return bool(frame.price_signal and frame.external_positive and frame.underreaction)
    if policy == "OVERREACTION":
        return bool(frame.price_signal and frame.external_positive and frame.overreaction)
    raise ValueError(f"未知预先固定账户：{policy}")


def load_engine():
    spec = importlib.util.spec_from_file_location("factor96_crossborder_engine", OUT / "code/account_engine.py")
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
    entry_idx, stop_level = None, np.nan
    stopped, exit_pending = False, False
    events = dividends.to_dict("records")
    records, decisions = [], []
    for i in ids:
        row, f = d.iloc[i], x.iloc[i-1]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                account.receivables[k] = value
                recognized += value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        ref = float(d.close.iloc[i-1] - row.dividend)
        q, reason = 0, "无新的合格首日信息吸收事件"
        signal = signal_for(f, policy)
        trigger = ""
        if old:
            if stopped:
                trigger = "回撤停机"
            elif float(f.wealth) < stop_level:
                trigger = "收盘跌破冻结首日低点"
            elif i >= entry_idx + 5:
                trigger = "五日到期"
            exit_pending |= bool(trigger)
            if exit_pending:
                q, reason = -old, trigger or "先前退出继续等待可成交"
            else:
                q = min(old, core.target_quantity(engine, account, ref, peak, float(f.es95))) - old
                reason = "已有份额只按风险预算削减"
        elif not stopped and signal and i < last:
            q = core.target_quantity(engine, account, ref, peak, float(f.es95))
            reason = "完整首日观察结束后下一开盘申请"
        before = q
        if q > 0:
            q = min(q, core.target_quantity(engine, account, op, peak, float(f.es95)))
        sellable = account.sellable(i)
        trade = engine.execute_order(account, q, op, float(row.previous_close), float(row.dividend), int(i), cost, cfg)
        if trade["filled_quantity"] > 0 and old == 0:
            entry_idx, stop_level = int(i), float(f.frozen_stop)
            exit_pending = False
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = engine.fill_price(close, -1, core.COSTS["STRESS"], .001)
            reserve = account.shares * (close-px) + engine.commission(account.shares, px, core.COSTS["STRESS"])
        equity = account.value(close) - reserve
        price_pnl = old * (op-previous_mark) + account.shares * (close-op)
        error = equity - previous_equity - price_pnl - recognized + trade["commission"] + trade["slippage_cost"] + reserve - previous_reserve
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        stopped |= drawdown >= .1
        records.append({"date": day, "idx": i, "policy": policy, "open": op, "mark": close, "cash": account.cash,
                        "shares": account.shares, "dividend_receivable": account.receivable(), "terminal_exit_reserve": reserve,
                        "equity": equity, "net_return": equity/previous_equity-1, "price_pnl": price_pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid, "exposure": account.shares*close/equity,
                        "accounting_error": error, "drawdown": drawdown, "risk_stopped": stopped, "sellable_before": sellable,
                        "terminal_unliquidated": bool(i == last and account.shares), **trade})
        decisions.append({"date": day, "origin": d.date.iloc[i-1], "origin_idx": i-1, "policy": policy, "active_signal": signal,
                          **{col: f[col] for col in ["source_day", "source_idx", "forecast_at", "latest_us_date", "fx_date",
                                                     "price_signal", "model_known", "external_positive", "underreaction", "overreaction",
                                                     "ashr_log_return", "fx_log_return", "predicted_response", "response_sigma", "response_z", "training_max_idx"]},
                          "es95_5d": f.es95, "reason": reason, "pre_open_request": before, "requested_quantity": q,
                          "filled_quantity": trade["filled_quantity"], "entry_idx": entry_idx, "frozen_stop": stop_level})
        if account.shares == 0:
            entry_idx, exit_pending = None, False
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    return pd.DataFrame(records), pd.DataFrame(decisions)


def prepare():
    assert not (OUT / "source_receipt.json").exists(), "来源已准备，不能覆盖"
    for name in ["inputs", "raw/fx", "code", "source_evidence", "program_before"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    paths = {
        "inputs/market.parquet": FIRST / "inputs/market.parquet",
        "inputs/dividends.csv": FIRST / "inputs/dividends.csv",
        "inputs/price_features.parquet": FIRST / "daily_features_lag1.parquet",
        "inputs/risk_training_records.json": FIRST / "risk_training_records.json",
        "inputs/mature_risk_labels.parquet": FIRST / "mature_risk_labels.parquet",
        "inputs/ashr_cached.parquet": ROOT / "data/raw/us_china_overnight_v1/ASHR_daily.parquet",
        "inputs/fx_original.parquet": ROOT / "data/raw/macro/510300_macro_stress_2015_v2/usdcny_midpoint_daily_2015_2026.parquet",
        "source_evidence/cached_us_source_receipt.json": ROOT / "reports/data_quality/us_china_overnight_inputs_v1.json",
        "source_evidence/current_mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
        "source_evidence/old_us_binary_contract.yaml": ROOT / "config/510300_us_china_overnight_binary_screen_v1_candidates.yaml",
        "source_evidence/old_us_binary_result.json": ROOT / "reports/research/510300_us_china_overnight_binary_screen_v1.json",
        "source_evidence/old_global_information_result.json": ROOT / "reports/research/510300_overnight_global_information_v1/result.json",
        "source_evidence/old_breakout_retest_contract.md": ROOT / "docs/510300_BREAKOUT_RETEST_ENTRY_V1.md",
        "source_evidence/old_breakout_retest_result.json": ROOT / "reports/research/510300_breakout_retest_entry_v1/acceptance_outcome.json",
        "source_evidence/if_source_gate.json": ROOT / "reports/research/510300_if_open_interest_increment_v1/data_preflight.json",
    }
    receipts = []
    for name, path in paths.items():
        shutil.copy2(path, OUT / name)
        receipts.append({"original": path.relative_to(ROOT).as_posix(), "snapshot": name, "sha256": digest(path)})
    for p in (ROOT / "reports/research/510300_factor96_program_v1").iterdir():
        if p.is_file():
            shutil.copy2(p, OUT / "program_before" / p.name)
    fx = pd.read_parquet(OUT / "inputs/fx_original.parquet")
    fx = fx[fx.date.le("2025-12-31")].reset_index(drop=True)
    for path, group in fx.groupby("raw_path"):
        source = ROOT / path
        assert group.source_hash.nunique() == 1 and digest(source) == group.source_hash.iloc[0]
        dest = OUT / "raw/fx" / source.name
        assert not dest.exists(), "同名来源冲突"
        shutil.copy2(source, dest)
        receipts.append({"original": path, "snapshot": dest.relative_to(OUT).as_posix(), "sha256": digest(source)})
    fx.to_parquet(OUT / "inputs/fx.parquet", index=False)
    source_path = OUT / "raw/ASHR_chart_2013_2025.json"
    assert source_path.exists(), "先保存公开ASHR图表响应，不能临时换源"
    us = parse_ashr(json.loads(source_path.read_text(encoding="utf-8")))
    cached = pd.read_parquet(OUT / "inputs/ashr_cached.parquet")
    match = us.merge(cached, on="date", suffixes=("_new", "_cached"), validate="one_to_one")
    assert len(match) == len(us)
    for col in ["open", "high", "low", "close"]:
        np.testing.assert_allclose(match[col+"_new"], match[col+"_cached"], rtol=0, atol=1e-8)
    us.to_parquet(OUT / "inputs/ashr.parquet", index=False)
    references = []
    for name, url in REFERENCES.items():
        path = OUT / "source_evidence" / name
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
        references.append({"url": url, "path": path.relative_to(OUT).as_posix(), "sha256": digest(path), "retrieved_at": now()})
    save(OUT / "source_evidence/overlap_and_source_decisions.json", {
        "at": now(), "T01": {"status": "NOT_RUN_REJECTED_NEAR_DUPLICATE_NO_PARAMETER_RESCUE", "reason": "旧突破回踩研究已停止窗口、容忍区间、确认与退出细调；现草案添加ER及改变保留率阈值不足以证明独立机制，不新增账户。"},
        "T09": {"status": "NOT_RUN_CARRY_ADJUSTMENT_SOURCE_GATE", "reason": "逐合约IF行情可用；截止到期日前当时已公告分红的指数点数需历史权重及公告版本对应。原基差不能冒充扣持有成本基差，现未准入。"},
        "T10": {"status": "ADMITTED_NEW_RESIDUAL_QUESTION_DISCOVERY_ONLY", "reason": "旧八规则在A股开盘前按外盘符号或分位做全仓二元择时；本次按固定两年模型观察完整首日残差，次日开盘才入场，单独比较正外盘条件与额外残差条件。不是旧八规则改阈值。"},
        "J03_operationalization": "因策略卡要求完整首日，本轮主回归标签为完整首日总收益；原因子卡次日开盘预测另存描述性列，不据此更换主规则。",
        "J06_boundary": "负向外部冲击的相对抗跌定义与T10正信息低反应不同，本轮不把J06记作已检验。"
    }, True)
    save(OUT / "source_receipt.json", {"at": now(), "sources": receipts, "references": references,
        "ashr_url": YAHOO_URL, "ashr_raw_sha256": digest(source_path), "ashr_rows": len(us),
        "ashr_cash_distributions": int(us.cash_distribution.gt(0).sum()), "ashr_quote_match_cached": True,
        "ashr_adjusted_close_used": False, "fx_rows": len(fx), "fx_raw_files": fx.raw_path.nunique(),
        "cash_distribution_values_source": "Yahoo公开历史响应；尚未逐笔认证发行方公告首版，不能声称严格PIT版本已证实。",
        "historical_publication_clock": "纽约17:00保守使用完成日线；中间价09:15发布，09:20估计首日；中国16:00观察完整首日。",
        "first_public_version_authenticated": False, "new_strategy_returns_evaluated": False}, True)
    print("T10来源已固定：ASHR原价加现金分配、官方中间价原文、历史重叠与时钟分别保存。", flush=True)


def freeze():
    assert (OUT / "source_receipt.json").exists() and not (OUT / "freeze.json").exists()
    tests = json.loads((OUT / "prefreeze_test_receipt.json").read_text(encoding="utf-8"))
    assert tests["exit_code"] == 0
    protocol = {
        "study_id": STUDY, "at": now(), "primary": "FULL", "new_candidates": 1, "periods": core.PERIODS,
        "capital": [200000, 20000], "costs": core.COSTS, "annual_days": 242, "policies": POLICIES,
        "extra_delay_policies": ["POSITIVE", "FULL"], "expected_accounts": 48,
        "hypothesis": "海外正信息已出现，但A股完整首日正涨幅仍比此前模型预计低一个残差标准差，余下吸收可能延续。",
        "counterevidence": "ASHR溢价、汇率与非同步时钟可造成回归残差；低反应不必延续，亦可能当地信息更弱。",
        "foreign_asset_choice": "ASHR同样跟踪沪深300；选择基于底层对应，不在FXI/MCHI/KWEB中比较绩效择优。",
        "us_return": "之前A股开盘前至当日开盘前新完成的所有美国日线之现金分配总收益连乘，转对数；无新增美国时段则缺失，不填零。",
        "us_clock": "每个美国交易日纽约17:00视为可得，按America/New_York处理夏令时；只读取当日中国09:20以前完成的值。",
        "fx_clock": "官方当日09:15中间价，09:20模型可用；两相邻A股日均须有当日发布，取对数比；它是中间价控制量而非可交易外汇收盘价。",
        "model": {"method": "OLS_WITH_INTERCEPT", "predictors": PREDICTORS, "window_calendar_years": 2,
                  "minimum_training_rows": 252, "update": "EVERY_TRADING_DAY_BEFORE_OPEN", "training_last": "严格早于当日，完整首日标签已成熟",
                  "response": "log((510300收盘+当日除息现金)/前收盘)", "sigma": "训练残差平方和除以n-4后开方", "regularization_or_search": False},
        "entry": "FULL=model_known且首日总收益>0且ASHR区间总收益>0且实际首日收益<预测-1sigma；16:00确认，下一A股日开盘。",
        "controls": {"PRICE_COMMON": "相同来源和模型覆盖，只要求A股首日总收益>0", "POSITIVE": "再要求ASHR区间总收益>0，不要求残差",
                     "OVERREACTION": "同正信息、正A股首日，但反应>预测+1sigma；竞争解释对照，不能替换失败主方案"},
        "exit": "收盘财富价严格低于冻结首日财富最低价，或入场后5个开盘间隔，或共同账户风险触发，下一可成交开盘退出。",
        "event_overlap": "每次完整首日代表一个新观察；持仓期不加仓，忽略同期新入场观察。所有来源合格和不合格日均保留。",
        "extra_delay": "整体事件和残差额外等待1个A股日，最新收盘仍不低于首日低点；当天风险估计不回退。不能晋升延迟对照。",
        "account": "510300/人民币现金；现金利率0，100份整手，T+1，方向涨跌停、最低5元佣金、滑点向不利方向取整，分红权益与期末退出成本准备。",
        "risk": "复用冻结两年成熟5日ES95每日估计、50%最高目标、2.5%ES预算、10%跳空下5%损失预算、剩余回撤空间折半及10%回撤停机。",
        "acceptance": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1, "both_capitals_stress_required": True},
        "bootstrap": {"primary": "FULL_MINUS_POSITIVE", "secondary": "FULL_MINUS_PRICE_COMMON", "period": "MAIN", "capital": 200000,
                      "cost": "STRESS", "block": 20, "draws": 4000, "seed": 20260930},
        "selection": "新候选库第5个固定主问题，历史在旧研究中已观察；对照与延迟不是独立新候选。",
        "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "forward_observations": 0,
        "stop": "固定主方案失败则保存并停止；不改1sigma、方向、模型窗口、观察时钟或退出，不将表现较好对照晋升。",
        "orders_authorized": False, "goal_achieved": False, "goal_status": "active",
    }
    for name, path in {"factor96_crossborder_absorption_v1.py": Path(__file__), "factor96_margin_repair_v1.py": Path(core.__file__),
                       "account_engine.py": ROOT / "research/intraday_overnight_increment_v1.py",
                       "test_factor96_crossborder_absorption_v1.py": ROOT / "tests/test_factor96_crossborder_absorption_v1.py"}.items():
        shutil.copy2(path, OUT / "code" / name)
    save(OUT / "protocol.json", protocol, True)
    files = [p for folder in ["inputs", "raw", "code", "source_evidence", "program_before"] for p in (OUT / folder).rglob("*") if p.is_file()]
    files += [OUT / "protocol.json", OUT / "source_receipt.json", OUT / "prefreeze_test_receipt.json"]
    save(OUT / "freeze.json", {"at": now(), "before_new_features_and_strategy_returns": True,
                              "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in files]}, True)
    print("T10已冻结：一个主问题、三个对照和一日等待敏感性，共48个账户情景。", flush=True)


def run():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        assert digest(OUT / row["path"]) == row["sha256"], row["path"]
    assert digest(Path(__file__)) == digest(OUT / "code/factor96_crossborder_absorption_v1.py")
    assert digest(Path(core.__file__)) == digest(OUT / "code/factor96_margin_repair_v1.py")
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")}, True)
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market = market[market.date.le("2025-12-31")].reset_index(drop=True)
    price = pd.read_parquet(OUT / "inputs/price_features.parquet")
    assert market.date.equals(price.date)
    aligned = align_external(market, pd.read_parquet(OUT / "inputs/ashr.parquet"), pd.read_parquet(OUT / "inputs/fx.parquet"))
    aligned.to_parquet(OUT / "aligned_external.parquet", index=False)
    model, training = rolling_response_model(aligned)
    model.to_parquet(OUT / "response_models.parquet", index=False)
    save(OUT / "response_training_records.json", training)
    div = core.normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    engine = load_engine()
    metrics, annual, stages, accounts = [], [], [], {}
    for lag in [0, 1]:
        x = lag_features(model, price, lag)
        x.to_parquet(OUT / f"daily_features_lag{lag}.parquet", index=False)
        for period, (start, end) in core.PERIODS.items():
            v = x[x.date.between(start, end)]
            stages.append({"period": period, "extra_delay": lag, "days": len(v), "model_known": int(v.model_known.sum()),
                           "first_day_up": int(v.price_signal.sum()), "positive_information": int((v.price_signal & v.external_positive).sum()),
                           "full_signals": int((v.price_signal & v.external_positive & v.underreaction).sum()),
                           "overreaction_signals": int((v.price_signal & v.external_positive & v.overreaction).sum())})
            for capital in [200000, 20000]:
                for cost in core.COSTS:
                    for policy in (POLICIES if lag == 0 else ["POSITIVE", "FULL"]):
                        ledger, decisions = simulate(market, x, div, policy, capital, cost, start, end, engine)
                        folder = OUT / f"accounts/{period}/{capital}/{cost}/LAG{lag}/{policy}"
                        folder.mkdir(parents=True, exist_ok=True)
                        ledger.to_parquet(folder / "ledger.parquet", index=False)
                        decisions.to_parquet(folder / "decisions.parquet", index=False)
                        cycles = core.cycle_records(ledger, capital)
                        cycles.to_csv(folder / "cycles.csv", index=False, encoding="utf-8-sig")
                        closed = cycles[cycles.closed.astype(bool)]
                        key = {"period": period, "capital": capital, "cost": cost, "lag": lag, "policy": policy}
                        m = core.metrics(ledger, capital)
                        metrics.append({**key, **m, "closed_cycles": len(closed), "win_rate": closed.profit.gt(0).mean() if len(closed) else None,
                                        "point_pass": m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2 and m["cagr"] >= .1 and m["max_drawdown"] <= .1})
                        previous = capital
                        for year, part in ledger.groupby(ledger.date.dt.year):
                            annual.append({**key, "year": int(year), **core.metrics(part, previous)})
                            previous = float(part.equity.iloc[-1])
                        accounts[period, capital, cost, lag, policy] = ledger
                        if period == "MAIN" and capital == 200000 and cost == "STRESS":
                            print(f"T10 {policy}/额外等待{lag}：夏普{m['net_sharpe']}，年化{m['cagr']:.3%}，完整周期{len(closed)}。", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    save(OUT / "signal_stage_counts.json", stages)
    primary = frame[(frame.period == "MAIN") & (frame.cost == "STRESS") & (frame.lag == 0) & (frame.policy == "FULL")]
    a = accounts["MAIN", 200000, "STRESS", 0, "FULL"]
    n = len(a)
    starts = np.random.default_rng(20260930).integers(0, n, size=(4000, int(np.ceil(n/20))))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(4000, -1)[:, :n]
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=indices)
    increments = []
    for other in ["POSITIVE", "PRICE_COMMON"]:
        b = accounts["MAIN", 200000, "STRESS", 0, other]
        diff = a.net_return.to_numpy() - b.net_return.to_numpy()
        draws = diff[indices].mean(axis=1) * 242
        increments.append({"comparison": "FULL_MINUS_" + other, "annual_arithmetic_increment": diff.mean()*242,
                           "ci95_low": np.quantile(draws, .025), "ci95_high": np.quantile(draws, .975), "global_selection_adjusted": False})
    save(OUT / "paired_increment.json", increments)
    save(OUT / "result.json", {"study_id": STUDY, "at": now(), "new_accounts": len(frame), "new_candidates": 1,
        "primary_rows": primary.to_dict("records"), "historical_joint_point_pass": bool(primary.point_pass.all()),
        "increments": increments, "response_models": len(training), "goal_achieved": False, "goal_status": "active",
        "independent_forward_observations": 0, "external_review": "NOT_PERFORMED", "orders_authorized": False})
    print(f"T10固定检验完成{len(frame)}个账户，共同目标通过：{bool(primary.point_pass.all())}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "run": run}[args.action]()
