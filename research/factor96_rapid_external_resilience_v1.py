"""J06：美国大盘负向冲击后，检验510300完整首日的相对抗跌信息。"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research import factor96_rapid_feasibility_v1 as rapid
from scripts.record_factor96_remaining_changes_v1 import update

ROOT, core = rapid.ROOT, rapid.core
OUT = ROOT / "reports/research/510300_factor96_rapid_external_resilience_v1"
STUDY = "510300_FACTOR96_RAPID_EXTERNAL_RESILIENCE_V1"
SOURCE = ROOT / "data/raw/cross_market_chart_ml_v1/GSPC_daily.parquet"
START, END, HOLD = "2017-01-03", "2025-12-31", 5
CANDIDATES = ["SHOCK_RELATIVE", "SHOCK_ABSOLUTE", "SHOCK_BOTH"]
POLICIES = CANDIDATES + ["SHOCK_WEAK_CONTROL", "SHOCK_ALL", "PRICE_CONTROL", "BUY_HOLD_50", "CASH"]
read, save = rapid.read, rapid.save


def freeze():
    assert not OUT.exists(), "本研究已经存在，不覆盖"
    save(OUT / "protocol.json", {
        "at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "previous_turn_classification": "PROGRESS_32_ORDERS_INVENTORY_ACCOUNTS_AND_NEGATIVE_CORRELATION_STABILITY_EVIDENCE",
        "question": "美国大盘预先定义的负向冲击后，510300的完整首日比正常联动预测更强，能否改善随后五日的账户收益。相对抗跌不自动代表绝对获利。",
        "source": "只使用现存GSPC价格指数原始收盘，不把adj_close或总收益指数混入。美国资产仅用于信息，执行始终510300/人民币现金。",
        "source_clock": "纽约当日17:00后保守视作日线已完成，按America/New_York转换夏令时。中国当日09:20只能读取此前已结束美国时段；无新增美国时段则未知，不填零。历史供应商实际首次交付未证实。",
        "interval": "相邻A股开盘前09:20之间新增的所有已完成美国交易日对数收益求和；最新源记录及基准记录各距对应A股日最多7自然日。",
        "shock": "本次美国区间对数收益 <= -1.5 * 区间开始前已知最近20个美国日收益标准差 * sqrt(本次美国时段数)。波动不包含本次冲击。",
        "normal_response_model": "每天开盘前以此前两日历年、至少252个合格A股日OLS，因变量为A股当日含分红对数收益，自变量为截距、本次美国区间对数收益、此前一日A股收益。训练严格早于当日。",
        "resilience": "完整首日A股实际含分红对数收益减开盘前模型预测；另除以训练残差标准差存储z值。入场不再做残差分位或幅度搜索。",
        "candidates": {"SHOCK_RELATIVE": "负向冲击且完整首日残差>0。",
                       "SHOCK_ABSOLUTE": "负向冲击且完整首日A股收益>=0。",
                       "SHOCK_BOTH": "同时满足相对抗跌和绝对不跌；预先指定的联合问题。"},
        "controls": {"SHOCK_WEAK_CONTROL": "负向冲击且残差<=0，竞争解释对照，不能在主问题失败后晋升。",
                     "SHOCK_ALL": "合格负向冲击后不按A股反应筛选。",
                     "PRICE_CONTROL": "相同模型覆盖下，只有A股首日收益>=0的价格对照。",
                     "BUY_HOLD_50": "初始半仓买入持有背景，未使用候选风险停止。", "CASH": "零利息现金。"},
        "execution": "完整首日收盘后确认，次开盘入场，固定5个开盘间隔后退出；持有中忽略新信号。复用既有50%最大仓位、ES预算、回撤储备与10%永久停止、整手/T+1/涨跌停/现金分红/期末退出准备。",
        "capital": [200000, 20000], "costs": core.COSTS, "annual_days": 242,
        "sample": [START, END], "period_diagnostics": [[START, "2020-12-31"], ["2021-01-01", END]],
        "planned_accounts": 32, "candidate_rules": 3, "fixed_event_rules_including_weak_control": 4,
        "followup_filter": "20万STRESS夏普>=0.8、CAGR>=5%、回撤<=10%、入场>=20、前后两段收益正，且夏普和CAGR均高于SHOCK_ALL。仅决定是否深入，不能代替最终夏普1.2/CAGR10%及独立验证。",
        "correlation_diagnostic": "仅在合格负向冲击中，残差z与下一开盘起五个开盘间隔的含分红固定份额收益计算Spearman，全期及前后两段；另按时间顺序保留相隔至少5交易日事件。重叠样本不作为独立显著性证据。",
        "novelty": {"T10": "旧ASHR正信息反应不足在首日正涨幅后入场；本次用美国大盘负向冲击及本地抗跌，不重跑旧T10或改变其方向。",
                    "US_CHINA_BINARY": "旧四只中国ETF隔夜负值/尾部/相对SPY的开盘前二元规则，没有观察本地完整首日的预期响应残差。",
                    "OVERNIGHT_GLOBAL": "旧65变量全球分类模型在当日开盘前配置；本次是单一全球冲击后的有无本地响应筛选对照。",
                    "scope": "只说明本次核对的三个旧实验有何不同，不宣称该经济主题在全部历史研究中从未出现。"},
        "no_rescue": "固定冲击阈值、模型、窗口、条件、持有期、费用和时期；保留全表及全部旧失败。",
        "official_references": ["https://www.nyse.com/markets/hours-calendars", "https://www.spglobal.com/spdji/en/indices/equity/sp-500/"],
        "orders_authorized": False, "delivery_package_required": False,
    })
    paths = [Path(__file__), Path(rapid.__file__), Path(core.__file__), rapid.ENGINE_PATH, SOURCE,
             rapid.PRICE_BASE / "market.parquet", rapid.PRICE_BASE / "price_features.parquet", rapid.PRICE_BASE / "dividends.csv",
             ROOT / "reports/research/510300_factor96_crossborder_absorption_v1/protocol.json",
             ROOT / "config/510300_us_china_overnight_binary_screen_v1_candidates.yaml",
             ROOT / "docs/510300_OVERNIGHT_GLOBAL_INFORMATION_V1_PROTOCOL.md", OUT / "protocol.json"]
    save(OUT / "freeze.json", {"at": core.now(), "files": [{"path": str(p), "sha256": core.digest(p)} for p in paths]})
    print("已冻结J06负冲击抗跌问题，3个候选、5个对照，合计32个账户。", flush=True)


def align(m, source):
    us = source.copy().sort_values("date").reset_index(drop=True)
    us.date = pd.to_datetime(us.date)
    assert us.date.is_unique and us.close.gt(0).all() and us.symbol.eq("^GSPC").all()
    us["log_return"] = np.log(us.close).diff()
    us["vol20"] = us.log_return.rolling(20).std(ddof=1)
    us["available_at"] = (pd.DatetimeIndex(us.date)+pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("Asia/Shanghai")
    clocks = pd.DatetimeIndex(us.available_at)
    dates = pd.DatetimeIndex(m.date)
    decisions = dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=20)
    records = []
    for i, day in enumerate(dates):
        latest = int(clocks.searchsorted(decisions[i], side="right")-1)
        base = int(clocks.searchsorted(decisions[i-1], side="right")-1) if i else -1
        valid = base >= 0 and latest > base
        if valid:
            valid = (day-us.date.iloc[latest]).days <= 7 and (dates[i-1]-us.date.iloc[base]).days <= 7
        values = us.log_return.iloc[base+1:latest+1] if valid else pd.Series(dtype=float)
        vol = float(us.vol20.iloc[base]) if valid else np.nan
        valid = bool(valid and values.notna().all() and np.isfinite(vol) and vol > 0)
        records.append({"date": day, "forecast_at": decisions[i],
                        "base_us_date": us.date.iloc[base] if base >= 0 else pd.NaT,
                        "last_us_date": us.date.iloc[latest] if latest >= 0 else pd.NaT,
                        "last_us_available_at": us.available_at.iloc[latest] if latest >= 0 else pd.NaT,
                        "us_sessions": latest-base if valid else 0,
                        "us_interval_return": float(values.sum()) if valid else np.nan,
                        "pre_interval_us_vol20": vol if valid else np.nan})
    f = pd.DataFrame(records)
    f["a_return"] = np.log((m.close+m.dividend)/m.close.shift())
    f["prior_a_return"] = f.a_return.shift()
    f["us_shock_z"] = f.us_interval_return/(f.pre_interval_us_vol20*np.sqrt(f.us_sessions.where(f.us_sessions.gt(0))))
    available = f.last_us_available_at.notna()
    assert (f.loc[available, "last_us_available_at"] <= f.loc[available, "forecast_at"]).all()
    return f


def response_features(m, source, risk):
    f = align(m, source)
    x = np.column_stack([np.ones(len(f)), f.us_interval_return, f.prior_a_return])
    y = f.a_return.to_numpy(float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    dates = pd.DatetimeIndex(f.date)
    prediction, sigma = np.full(len(f), np.nan), np.full(len(f), np.nan)
    counts, last = np.zeros(len(f), int), np.full(len(f), -1, int)
    betas = np.full((len(f), 3), np.nan)
    for i, day in enumerate(dates):
        if not np.isfinite(x[i]).all():
            continue
        lower = dates.searchsorted(day-pd.DateOffset(years=2))
        ids = np.flatnonzero(valid[lower:i])+lower
        counts[i] = len(ids)
        if len(ids) < 252:
            continue
        beta, _, rank, _ = np.linalg.lstsq(x[ids], y[ids], rcond=None)
        if rank != 3:
            continue
        error = y[ids]-x[ids]@beta
        s = float(np.sqrt(np.square(error).sum()/(len(ids)-3)))
        if s <= 1e-12:
            continue
        prediction[i], sigma[i], last[i], betas[i] = x[i]@beta, s, int(ids[-1]), beta
    f["predicted_a_return"], f["response_sigma"] = prediction, sigma
    f["training_count"], f["training_last_idx"] = counts, last
    for j, name in enumerate(["intercept", "us_beta", "prior_a_beta"]):
        f[name] = betas[:, j]
    f["response_residual"] = f.a_return-f.predicted_a_return
    f["response_z"] = f.response_residual/f.response_sigma
    f["model_known"] = f.predicted_a_return.notna() & f.response_z.notna()
    f["SHOCK_ALL"] = f.model_known & f.us_shock_z.le(-1.5)
    f["SHOCK_RELATIVE"] = f.SHOCK_ALL & f.response_residual.gt(0)
    f["SHOCK_ABSOLUTE"] = f.SHOCK_ALL & f.a_return.ge(0)
    f["SHOCK_BOTH"] = f.SHOCK_RELATIVE & f.SHOCK_ABSOLUTE
    f["SHOCK_WEAK_CONTROL"] = f.SHOCK_ALL & f.response_residual.le(0)
    f["PRICE_CONTROL"] = f.model_known & f.a_return.ge(0)
    f["es95"] = risk.es95.to_numpy()
    ids = np.flatnonzero(f.model_known)
    assert (f.training_last_idx.iloc[ids].to_numpy() < ids).all()
    return f


def diagnostics(f, m, dividends):
    rows, last_idx = [], -100
    for i in np.flatnonzero(f.SHOCK_ALL & f.date.between(START, END)):
        entry, exit_idx = i+1, i+1+HOLD
        if exit_idx >= len(m):
            continue
        a, b = m.date.iloc[entry], m.date.iloc[exit_idx]
        entitled = dividends.record_date.ge(a) & dividends.record_date.lt(b) & dividends.ex_date.le(b)
        y = (m.open.iloc[exit_idx]-m.open.iloc[entry]+dividends.loc[entitled, "cash_dividend_per_share"].sum())/m.open.iloc[entry]
        independent_spacing = i-last_idx >= HOLD
        if independent_spacing:
            last_idx = i
        rows.append({"date": f.date.iloc[i], "idx": int(i), "entry_date": a, "exit_date": b,
                     "response_z": f.response_z.iloc[i], "us_shock_z": f.us_shock_z.iloc[i], "forward_return5": float(y),
                     "nonoverlap_subset": independent_spacing, **{key: bool(f[key].iloc[i]) for key in POLICIES[:4]}})
    events = pd.DataFrame(rows)
    events.to_parquet(OUT / "event_labels.parquet", index=False)
    summary = []
    for kind, subset in [("all_events", events), ("nonoverlap_subset", events.loc[events.nonoverlap_subset])]:
        for label, start, end in [("full", START, END), ("early", START, "2020-12-31"), ("late", "2021-01-01", END)]:
            part = subset.loc[subset.date.between(start, end)]
            summary.append({"sample": kind, "period": label, "events": len(part),
                            "spearman": float(part.response_z.corr(part.forward_return5, method="spearman")) if len(part) > 2 else None})
    save(OUT / "correlation_diagnostic.json", {"at": core.now(), "rows": summary, "independent_validation": False})
    return summary


def run():
    for row in read(OUT / "freeze.json")["files"]:
        assert core.digest(Path(row["path"])) == row["sha256"]
    save(OUT / "run_started.json", {"at": core.now(), "planned_accounts": 32})
    m = pd.read_parquet(rapid.PRICE_BASE / "market.parquet")
    m.date = pd.to_datetime(m.date)
    m = m.loc[m.date.le(END)].reset_index(drop=True)
    risk = pd.read_parquet(rapid.PRICE_BASE / "price_features.parquet")
    risk.date = pd.to_datetime(risk.date)
    assert m.date.equals(risk.date) and len(m) == 3307
    us = pd.read_parquet(SOURCE)
    us.date = pd.to_datetime(us.date)
    us = us.loc[us.date.le(END)].copy()
    f = response_features(m, us, risk)
    f.to_parquet(OUT / "features_and_signals.parquet", index=False)
    cut = pd.Timestamp("2021-01-04")
    short = response_features(m.loc[m.date.le(cut)].copy(), us.loc[us.date.le(cut)].copy(), risk.loc[risk.date.le(cut)].copy())
    pd.testing.assert_frame_equal(f.loc[f.date.le(cut)].reset_index(drop=True), short)
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    spec = importlib.util.spec_from_file_location("external_resilience_account_engine", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    rapid.START, rapid.END, rapid.HOLD = START, END, HOLD
    rows, annual = [], []
    for capital in [200000, 20000]:
        for cost in ["BASE", "STRESS"]:
            for policy in POLICIES:
                ledger, orders = rapid.simulate(m, f, dividends, policy, capital, cost, engine)
                ledger["reason"] = ledger.reason.replace({"十个开盘间隔到期": "五个开盘间隔到期"})
                if not orders.empty:
                    orders["reason"] = orders.reason.replace({"十个开盘间隔到期": "五个开盘间隔到期"})
                directory = OUT / "accounts" / f"{capital}_{cost}_{policy}"
                directory.mkdir(parents=True)
                ledger.to_parquet(directory / "ledger.parquet", index=False)
                orders.to_parquet(directory / "orders.parquet", index=False)
                row = core.metrics(ledger, capital)
                row.update(policy=policy, capital=capital, cost=cost,
                           entries=int(((ledger.filled_quantity > 0) & ledger.shares_before.eq(0)).sum()),
                           stopped_at=ledger.loc[ledger.risk_stopped, "date"].min().isoformat() if ledger.risk_stopped.any() else None,
                           ledger_path=str((directory / "ledger.parquet").relative_to(ROOT)))
                for label, a, b in [("early", START, "2020-12-31"), ("late", "2021-01-01", END)]:
                    row.update({label+"_"+k: v for k, v in rapid.period_metrics(ledger, a, b).items()})
                rows.append(row)
                for year in range(2017, 2026):
                    annual.append({"policy": policy, "capital": capital, "cost": cost, "year": year,
                                   **rapid.period_metrics(ledger, f"{year}-01-01", f"{year}-12-31")})
            print(f"已完成{capital}元、{cost}成本的8个抗跌账户。", flush=True)
    rapid.HOLD = 10
    table = pd.DataFrame(rows)
    table["candidate"] = table.policy.isin(CANDIDATES)
    table["beats_shock_control"] = False
    for (capital, cost), part in table.groupby(["capital", "cost"]):
        baseline = part.loc[part.policy.eq("SHOCK_ALL")].iloc[0]
        table.loc[part.index, "beats_shock_control"] = part.net_sharpe.gt(baseline.net_sharpe) & part.cagr.gt(baseline.cagr)
    table["joint_historical_target"] = table.candidate & table.net_sharpe.ge(1.2) & table.cagr.ge(.1) & table.max_drawdown.le(.1)
    table["worth_followup"] = (table.candidate & table.net_sharpe.ge(.8) & table.cagr.ge(.05) & table.max_drawdown.le(.1)
                               & table.entries.ge(20) & table.early_net_profit_fraction.gt(0) & table.late_net_profit_fraction.gt(0)
                               & table.beats_shock_control)
    table.to_csv(OUT / "all_account_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    primary.to_csv(OUT / "20万元压力成本比较.csv", index=False, encoding="utf-8-sig")
    corr = diagnostics(f, m, dividends)
    evaluation = f.loc[f.date.between(START, END)]
    save(OUT / "result.json", {"at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "account_scenarios": len(table), "candidate_rules": 3, "fixed_event_rules_including_weak_control": 4,
        "source_known_days": int(evaluation.model_known.sum()), "shock_events": int(evaluation.SHOCK_ALL.sum()),
        "event_counts": {key: int(evaluation[key].sum()) for key in POLICIES[:4]},
        "primary_results": primary.to_dict("records"), "correlation_diagnostic": corr,
        "worth_followup": primary.loc[primary.worth_followup, "policy"].tolist(),
        "joint_historical_target_count": int(primary.joint_historical_target.sum()), "causal_prefix_check": "PASS",
        "historical_vendor_first_delivery_proven": False, "independent_forward_observations": 0, "qualified_candidates": 0,
        "goal_achieved": False, "new_source_downloads": 0, "orders_authorized": False, "delivery_package_created": False})
    print(primary[["policy", "net_sharpe", "cagr", "max_drawdown", "entries", "worth_followup"]].to_string(index=False), flush=True)
    print("条件相关性：", corr, flush=True)


def record():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    table = pd.read_csv(OUT / "all_account_metrics.csv")
    assert len(table) == 32
    for row in table.itertuples():
        actual = core.metrics(pd.read_parquet(ROOT / row.ledger_path), row.capital)
        for key in ["cagr", "max_drawdown", "end_equity", "commission", "slippage"]:
            assert np.isclose(actual[key], getattr(row, key), rtol=1e-10, atol=1e-8)
        if pd.notna(row.net_sharpe):
            assert np.isclose(actual["net_sharpe"], row.net_sharpe, atol=1e-10)
    save(OUT / "saved_verification_receipt.json", {"at": core.now(), "status": "PASS_32_SAVED_ACCOUNT_RECOMPUTATIONS_AND_CAUSAL_PREFIX",
                                                   "max_accounting_error": float(table.max_identity_error.max())})
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    promising = result["worth_followup"]
    names = {"SHOCK_RELATIVE": "冲击后相对抗跌", "SHOCK_ABSOLUTE": "冲击后当天不跌", "SHOCK_BOTH": "相对抗跌且当天不跌",
             "SHOCK_WEAK_CONTROL": "冲击后相对更弱对照", "SHOCK_ALL": "冲击后直接入场对照", "PRICE_CONTROL": "单纯当天不跌对照",
             "BUY_HOLD_50": "半仓买入持有背景", "CASH": "现金"}
    conclusion = "3个预先候选均未通过继续研究初筛，本轮固定抗跌规则结束。" if not promising else "值得深入的预先候选："+"、".join(promising)+"；尚未正式达标。"
    lines = ["只交易510300及现金、成本后夏普1.2目标不变。本轮以美国大盘急跌为事前事件，完整观察A股首日后才决定入场；3个候选及5个对照，共32个账户。",
             conclusion, "2017-2025年，20万元压力成本：", "| 规则 | 净夏普 | 年化收益 | 最大回撤 | 入场次数 |", "|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        value = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "无交易"
        lines.append(f"| {names[row.policy]} | {value} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.entries} |")
    lines += ["", f"共有{result['shock_events']}个合格负向冲击观察；各条件数量：{result['event_counts']}。相邻观察可能属于同一冲击，不视为独立样本。",
              "| 相关性样本 | 时段 | 事件数 | 残差z对随后5日收益的秩相关 |", "|---|---|---:|---:|"]
    for r in result["correlation_diagnostic"]:
        value = f"{r['spearman']:.3f}" if r['spearman'] is not None else "未知"
        lines.append(f"| {r['sample']} | {r['period']} | {r['events']} | {value} |")
    lines += ["", "模型只用此前两日历年的成熟首日收益，按美国纽约17:00和中国09:20对齐。交易必须等完整A股首日结束，再次开盘进入，持有五个开盘间隔。含所有现金日、分红、佣金、滑点、整手和T+1。",
              "美国17:00是保守市场完成假设，参考[NYSE时段](https://www.nyse.com/markets/hours-calendars)；并不证明免费数据供应商当时已交付。使用[S&P500价格指数](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)仅定义外部价格冲击，不把它当作可执行账户收益。",
              "保留旧T10正消息反应不足、四只中国ETF隔夜二元和全球分类模型的原结果。本次完整首日负冲击抗跌是独立固定问题，不能推广为全部跨市场机制无效。",
              "32份保存账本复算一致；截断未来价格重新计算历史模型、信号和来源时钟一致。历史已经用于多轮探索，独立前向观察仍为0。没有交付包、没有订单。"]
    (OUT / "快速检验结论.md").write_text("\n\n".join(lines[:3])+"\n\n"+"\n".join(lines[3:]), encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / "status.json", program / "factor_progress.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    objects = [read(p) for p in paths]
    status, factors, mandate = objects
    assert status["latest_round"] == "510300_FACTOR96_RAPID_ORDERS_INVENTORY_V1"
    before = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    rel = OUT.relative_to(ROOT).as_posix()
    note = f"J06外部负向冲击的完整首日抗跌：3个候选及5个对照、32个账户完成，待深入候选{len(promising)}，正式达标0。"
    status.update(at=core.now(), latest_round=STUDY, latest_result=rel+"/result.json", last_research_result=note,
        last_completed_account_experiment=STUDY, last_completed_account_result=rel+"/result.json", latest_progress_receipt=rel+"/saved_verification_receipt.json",
        latest_continuation_classification="PROGRESS_32_EXTERNAL_RESILIENCE_ACCOUNTS_WITH_EVENT_CONTROLS",
        admitted_account_scenarios_this_round=32, invalid_implementation_accounts_this_round=0,
        cumulative_admitted_account_scenarios=status["cumulative_admitted_account_scenarios"]+32,
        cumulative_executed_account_scenarios=status["cumulative_executed_account_scenarios"]+32,
        rapid_screen_direction_tests=status["rapid_screen_direction_tests"]+4,
        current_research_phase="DEEPEN_PROMISING_EXTERNAL_RESILIENCE" if promising else "EXTERNAL_RESILIENCE_FIXED_RULES_ENDED",
        next_independent_source_action="DEEPEN_FROZEN_EXTERNAL_RESILIENCE" if promising else "SELECT_A_DISTINCT_UNTESTED_MECHANISM_USING_READY_DATA",
        goal_status="active", goal_achieved=False, qualified_candidates=[], source_detail_work_deferred_by_user=True)
    if not promising:
        status["stopped_simple_research_branches"].append("3_EXTERNAL_RESILIENCE_CANDIDATES_AND_1_WEAK_CONTROL")
    for row in factors:
        if row["id"] == "J06":
            row["rapid_negative_shock_screen"] = {"study_id": STUDY, "result": rel+"/result.json", "candidate_count": 3,
                                                  "status": "PROMISING_REQUIRES_REVIEW" if promising else "NO_PROMISING_FIXED_SIMPLE_RULE",
                                                  "historical_vendor_first_delivery_proven": False}
    mandate.update(current_round=STUDY, current_protocol=rel+"/protocol.json", last_research_result=note,
        latest_progress_receipt=rel+"/saved_verification_receipt.json", latest_continuation_report=rel+"/快速检验结论.md",
        latest_continuation_classification=status["latest_continuation_classification"], research_execution_state=status["current_research_phase"],
        goal_status="active", goal_achieved=False)
    for path, obj in zip(paths, objects):
        update(path, core.clean(obj))
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": core.now(), "before": before,
        "after": [{"path": str(p), "sha256": core.digest(p)} for p in paths], "new_accounts": 32, "goal_achieved": False})
    print(note, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run", "record"])
    stage = parser.parse_args().stage
    {"freeze": freeze, "run": run, "record": record}[stage]()
