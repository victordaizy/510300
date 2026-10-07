"""以既有资料初筛产品迁移代理和到期日条件下的IF持仓集中度。"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research import factor96_rapid_feasibility_v1 as rapid
from scripts.record_factor96_remaining_changes_v1 import update

ROOT, core = rapid.ROOT, rapid.core
OUT = ROOT / "reports/research/510300_factor96_rapid_structure_v1"
STUDY = "510300_FACTOR96_RAPID_STRUCTURE_V1"
IF_PATH = ROOT / "data/raw/futures/cffex_if_contract_history_v1_0_1/cffex_if_contract_daily.parquet"
SHARE_PATH = ROOT / "data/raw/flow/510300_cross_etf_forced_flow_sse_weekly_v1.parquet"
PRICE_PATH = ROOT / "data/raw/all_etf_momentum_v1r/etf_total_return_panel_tushare_adj.parquet"
POOL = ["510300.SH", "510310.SH", "510330.SH"]
GROUPS = {"IF_HHI": ("2017-01-03", "2020-12-31"), "ETF_MIGRATION_PROXY": ("2022-02-07", "2023-12-31")}
END = "2025-12-31"
read, save = rapid.read, rapid.save


def freeze():
    assert not OUT.exists(), "本研究已经存在，不重复冻结或覆盖"
    status = read(ROOT / "reports/research/510300_factor96_program_v1/status.json")
    protocol = {
        "at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "previous_goal_turn": "PROGRESS_104_ACCOUNTS_COMPLETED_AND_NEGATIVE_DIRECTIONS_STOPPED",
        "previous_round": status["latest_round"],
        "user_priority": "仍只交易510300，换收益机制；先简单检验可行性，有效果才深入；无需交付包。",
        "hypotheses": {
            "IF_HHI": "各当日在市IF合约OI占比平方和。按近月名义到期剩余日数分5日桶，只与此前两日历年同桶比较，至少40日，检验HIGH>=70%和LOW<=30%两个方向。另存近月份额，未预设集中一定是多头或空头。",
            "ETF_MIGRATION_PROXY": "固定三只现成上海沪深300ETF：510300周度份额增幅，减去三只份额差按当周原始收盘价估值后的资金量/前周份额按当周价估值的规模。仅为部分产品周度迁移代理；不冒充完整同指数日频NAV体系。此前两日历年至少52个周观测，HIGH>=70%与LOW<=30%。",
        },
        "novelty_and_exclusions": {
            "IF": "研究OI在合约间的分布，非旧总OI四象限、基差/期限曲线水平或强制抛售耗竭状态；不重跑或救援那些冻结失败。H02/H04的严格分红与持有成本资料不足，本轮不运行。",
            "ETF": "研究单产品相对同篮子部分产品的份额差，非旧五条总申赎二元规则；G04原卡完整体系仍未验证。固定三只用于现成资料代理，不依结果更换ETF池。",
        },
        "expiry_rule": "仅由合约代码年月按第三个周五计算名义到期日，不使用未来最后出现日期。若当日在市任一合约名义到期已过，或非4只合约，则本日缺失。名义日仅用于季节性分组，遇假期不猜实际顺延日期。",
        "official_references": ["https://www.cffex.com.cn/cn/hs300.html", "https://www.sse.com.cn/market/funddata/volumn/etfvolumn/"],
        "source_clock": "无法证明历史逐条首次发布时刻；统一假设T统计在T+1收盘可用，T+2开盘执行。该保守时滞是探索假设，不证明历史真实可得。ETF仅在新周观测的可用日产生一次信号，不前填成每日申赎。",
        "signals": "两个方向均要求信号当日ETF过去5日总收益>0。PRICE_CONTROL在相同可用日和历史数量门下仅使用该价格条件；BUY_HOLD_50为背景基准，不沿用候选风险停止。",
        "same_day_information": "分位阈值不包含本次观测；ETF周度差及IF字段均整体延迟一个交易日后才能触发。",
        "groups": GROUPS, "end": END,
        "sample_reason": "IF采用与上一批一致的2017-2025；周度ETF只从2021开始，至少一年预热，固定2022-02-07至2025年，保留匹配区间基准与两段结果。不得跨区间直接比较两类夏普。",
        "execution": "复用上一批固定10个开盘间隔、20万/2万、BASE/STRESS成本、现金日、分红、整手与T+1账本；最大仓位50%，ES和10%回撤终止合同不变；现金收益0、年化242日。",
        "planned_account_scenarios": 32, "new_direction_candidates": 4,
        "next_stage_filter": "20万STRESS夏普>=0.8、CAGR>=5%、回撤<=10%、入场>=20、两固定子期收益为正，且夏普和CAGR都高于匹配PRICE_CONTROL。只筛选值得深化者，不是最终验收。",
        "final_goal": "成本后夏普>=1.2、同时报告并检验CAGR>=10%及回撤<=10%，正式达标还需独立验证。",
        "no_rescue": "不依本轮结果改方向、窗口、阈值、持有期、成本或期间；失败立即结束本次简单用法。",
        "orders_authorized": False, "delivery_package_required": False,
    }
    save(OUT / "protocol.json", protocol)
    files = [Path(__file__), Path(rapid.__file__), Path(core.__file__), rapid.ENGINE_PATH,
             rapid.PRICE_BASE / "market.parquet", rapid.PRICE_BASE / "price_features.parquet",
             rapid.PRICE_BASE / "dividends.csv", IF_PATH, SHARE_PATH, PRICE_PATH, OUT / "protocol.json"]
    save(OUT / "freeze.json", {"at": core.now(), "files": [{"path": str(p), "sha256": core.digest(p)} for p in files]})
    print("已冻结两个新问题、四个方向及匹配价格对照，共32个账户；尚未读取本轮绩效。", flush=True)


def nominal_expiry(symbol):
    year, month = 2000 + int(symbol[2:4]), int(symbol[4:6])
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)


def prior_quantiles(frame, key, minimum, buckets=None):
    dates = pd.DatetimeIndex(frame.date)
    values = frame[key].to_numpy(float)
    low, high = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    counts = np.zeros(len(frame), dtype=int)
    for i, day in enumerate(dates):
        start = dates.searchsorted(day - pd.DateOffset(years=2))
        history = values[start:i]
        valid = np.isfinite(history)
        if buckets is not None:
            valid &= buckets[start:i] == buckets[i]
        history = history[valid]
        counts[i] = len(history)
        if len(history) >= minimum:
            low[i], high[i] = np.quantile(history, [.3, .7])
    frame["q30"], frame["q70"], frame["history_count"] = low, high, counts
    frame["eligible"] = np.isfinite(values) & np.isfinite(low) & (high > low)
    frame["HIGH"] = frame.eligible & (values >= high)
    frame["LOW"] = frame.eligible & (values <= low)
    return frame


def if_features(daily):
    d = daily.copy()
    d.date = pd.to_datetime(d.date)
    assert not d.duplicated(["date", "symbol"]).any()
    assert d.symbol.str.fullmatch(r"IF\d{4}").all()
    expiry = {symbol: nominal_expiry(symbol) for symbol in d.symbol.unique()}
    d["nominal_expiry"] = d.symbol.map(expiry)
    d["dte"] = (d.nominal_expiry - d.date).dt.days
    rows = []
    for day, g in d.groupby("date", sort=True):
        total = g.open_interest.sum()
        valid = len(g) == 4 and g.dte.ge(0).all() and g.open_interest.ge(0).all() and total > 0
        nearest = g.sort_values("nominal_expiry").iloc[0]
        shares = g.open_interest / total if total > 0 else g.open_interest * np.nan
        rows.append({"date": day, "IF_HHI": float(np.square(shares).sum()) if valid else np.nan,
                     "near_share": float(nearest.open_interest / total) if valid else np.nan,
                     "near_nominal_dte": int(nearest.dte), "dte_bucket": int(nearest.dte) // 5,
                     "contract_count": len(g), "aggregate_oi": float(total), "input_valid": valid})
    f = pd.DataFrame(rows)
    return prior_quantiles(f, "IF_HHI", 40, f.dte_bucket.to_numpy())


def etf_features(shares, prices):
    d = shares.loc[shares.ts_code.isin(POOL)].copy()
    d.date = pd.to_datetime(d.date)
    px = prices.loc[prices.con_code.isin(POOL), ["date", "con_code", "raw_close"]].copy()
    px.date = pd.to_datetime(px.date)
    assert not d.duplicated(["date", "ts_code"]).any()
    assert not px.duplicated(["date", "con_code"]).any()
    d = d.merge(px, left_on=["date", "ts_code"], right_on=["date", "con_code"], how="left", validate="one_to_one")
    s = d.pivot(index="date", columns="ts_code", values="fund_shares").reindex(columns=POOL)
    p = d.pivot(index="date", columns="ts_code", values="raw_close").reindex(columns=POOL)
    valid = s.gt(0).all(axis=1) & s.shift().gt(0).all(axis=1) & p.gt(0).all(axis=1)
    pool_flow = (s.diff() * p).sum(axis=1, min_count=3) / (s.shift() * p).sum(axis=1, min_count=3)
    single = s["510300.SH"].pct_change(fill_method=None)
    f = pd.DataFrame({"date": s.index, "ETF_MIGRATION_PROXY": (single-pool_flow).where(valid).to_numpy(),
                      "single_share_change": single.to_numpy(), "pool_flow_proxy": pool_flow.to_numpy(),
                      "input_valid": valid.to_numpy()})
    return prior_quantiles(f, "ETF_MIGRATION_PROXY", 52)


def signals(m, risk, observations, key):
    f = pd.DataFrame({"date": m.date, "es95": risk.es95})
    calendar = pd.DatetimeIndex(m.date)
    obs = observations.copy()
    pos = calendar.get_indexer(pd.DatetimeIndex(obs.date))
    keep = (pos >= 0) & (pos+1 < len(calendar))
    obs = obs.loc[keep].copy()
    obs["source_date"] = obs.date
    obs["date"] = calendar[pos[keep]+1]
    cols = ["date", "source_date", key, "q30", "q70", "history_count", "eligible", "HIGH", "LOW"]
    f = f.merge(obs[cols], on="date", how="left", validate="one_to_one")
    r = np.log((m.close+m.dividend)/m.close.shift())
    f["price_condition"] = r.rolling(5).sum().gt(0).to_numpy()
    for side in ["HIGH", "LOW"]:
        f[key+"_"+side] = f[side].eq(True) & f.price_condition
    f["PRICE_CONTROL"] = f.eligible.eq(True) & f.price_condition
    return f


def run():
    for row in read(OUT / "freeze.json")["files"]:
        assert core.digest(Path(row["path"])) == row["sha256"]
    save(OUT / "run_started.json", {"at": core.now(), "expected_account_scenarios": 32})
    m = pd.read_parquet(rapid.PRICE_BASE / "market.parquet")
    m.date = pd.to_datetime(m.date)
    m = m.loc[m.date.le(END)].reset_index(drop=True)
    risk = pd.read_parquet(rapid.PRICE_BASE / "price_features.parquet")
    risk.date = pd.to_datetime(risk.date)
    assert m.date.equals(risk.date) and len(m) == 3307
    assert m.symbol.eq("510300.SH").all()
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    futures = pd.read_parquet(IF_PATH)
    futures.date = pd.to_datetime(futures.date)
    futures = futures.loc[futures.date.le(END)].copy()
    shares = pd.read_parquet(SHARE_PATH)
    shares.date = pd.to_datetime(shares.date)
    shares = shares.loc[shares.date.le(END)].copy()
    prices = pd.read_parquet(PRICE_PATH, columns=["date", "con_code", "raw_close"], filters=[("con_code", "in", POOL)])
    prices.date = pd.to_datetime(prices.date)
    prices = prices.loc[prices.date.le(END)].copy()
    observations = {"IF_HHI": if_features(futures), "ETF_MIGRATION_PROXY": etf_features(shares, prices)}
    cut = pd.Timestamp("2023-12-29")
    prefixes = {"IF_HHI": if_features(futures.loc[futures.date.le(cut)]),
                "ETF_MIGRATION_PROXY": etf_features(shares.loc[shares.date.le(cut)], prices.loc[prices.date.le(cut)])}
    for key, frame in observations.items():
        pd.testing.assert_frame_equal(frame.loc[frame.date.le(cut)].reset_index(drop=True), prefixes[key].reset_index(drop=True))
        frame.to_parquet(OUT / f"{key}_observations.parquet", index=False)
    spec = importlib.util.spec_from_file_location("structure_account_engine", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    records, annual, source_coverage = [], [], {}
    for key, (start, split) in GROUPS.items():
        rapid.START, rapid.END = start, END
        f = signals(m, risk, observations[key], key)
        f.to_parquet(OUT / f"{key}_signals.parquet", index=False)
        short_m = m.loc[m.date.le(cut)].copy()
        short_risk = risk.loc[risk.date.le(cut)].copy()
        short_signals = signals(short_m, short_risk, prefixes[key], key)
        pd.testing.assert_frame_equal(f.loc[f.date.le(cut)].reset_index(drop=True), short_signals.reset_index(drop=True))
        evaluation = f.loc[f.date.between(start, END)]
        source_coverage[key] = {"first_observation": observations[key].date.min(),
                                "last_observation": observations[key].date.max(),
                                "evaluation_start": start, "evaluation_end": END,
                                "observation_count": len(observations[key]),
                                "invalid_input_observations": int((~observations[key].input_valid).sum()),
                                "eligible_decision_days": int(evaluation.eligible.eq(True).sum()),
                                "high_decision_days": int(evaluation[key+"_HIGH"].sum()),
                                "low_decision_days": int(evaluation[key+"_LOW"].sum()),
                                "historical_publication_timestamp_verified": False}
        for capital in [200000, 20000]:
            for cost in ["BASE", "STRESS"]:
                for policy in [key+"_HIGH", key+"_LOW", "PRICE_CONTROL", "BUY_HOLD_50"]:
                    ledger, orders = rapid.simulate(m, f, dividends, policy, capital, cost, engine)
                    folder = OUT / "accounts" / f"{key}_{capital}_{cost}_{policy}"
                    folder.mkdir(parents=True)
                    ledger.to_parquet(folder / "ledger.parquet", index=False)
                    orders.to_parquet(folder / "orders.parquet", index=False)
                    metric = core.metrics(ledger, capital)
                    metric.update(group=key, policy=policy, capital=capital, cost=cost,
                                  entries=int(((ledger.filled_quantity > 0) & ledger.shares_before.eq(0)).sum()),
                                  start=start, end=END,
                                  stopped_at=ledger.loc[ledger.risk_stopped, "date"].min().isoformat() if ledger.risk_stopped.any() else None,
                                  ledger_path=str((folder / "ledger.parquet").relative_to(ROOT)))
                    for label, a, b in [("early", start, split), ("late", (pd.Timestamp(split)+pd.Timedelta(days=1)).isoformat(), END)]:
                        metric.update({label+"_"+k: v for k, v in rapid.period_metrics(ledger, a, b).items()})
                    records.append(metric)
                    for year in range(pd.Timestamp(start).year, 2026):
                        annual.append({"group": key, "policy": policy, "capital": capital, "cost": cost, "year": year,
                                       **rapid.period_metrics(ledger, f"{year}-01-01", f"{year}-12-31")})
        print(f"已完成{key}的16个账户及匹配价格对照。", flush=True)
    rapid.START, rapid.END = "2017-01-03", END
    table = pd.DataFrame(records)
    table["new_candidate"] = ~table.policy.isin(["PRICE_CONTROL", "BUY_HOLD_50"])
    table["joint_historical_target"] = table.net_sharpe.ge(1.2) & table.cagr.ge(.1) & table.max_drawdown.le(.1) & table.new_candidate
    table["beats_matched_price_control"] = False
    for (key, capital, cost), part in table.groupby(["group", "capital", "cost"]):
        control = part.loc[part.policy.eq("PRICE_CONTROL")].iloc[0]
        table.loc[part.index, "beats_matched_price_control"] = part.net_sharpe.gt(control.net_sharpe) & part.cagr.gt(control.cagr)
    table["worth_followup"] = (table.new_candidate & table.net_sharpe.ge(.8) & table.cagr.ge(.05) & table.max_drawdown.le(.1)
                               & table.entries.ge(20) & table.early_net_profit_fraction.gt(0) & table.late_net_profit_fraction.gt(0)
                               & table.beats_matched_price_control)
    table.to_csv(OUT / "all_account_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    primary.to_csv(OUT / "20万元压力成本比较.csv", index=False, encoding="utf-8-sig")
    save(OUT / "coverage.json", source_coverage)
    result = {"at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
              "new_directions": 4, "account_scenarios": len(table), "primary_results": primary.to_dict("records"),
              "worth_followup": primary.loc[primary.worth_followup, "policy"].tolist(),
              "joint_historical_target_count": int(primary.joint_historical_target.sum()),
              "publication_clock": "ASSUMED_T_PLUS_1_CLOSE_NOT_HISTORICALLY_VERIFIED", "causal_prefix_check": "PASS",
              "original_G04_T07_validated": False, "qualified_candidates": 0, "independent_forward_observations": 0,
              "goal_achieved": False, "orders_authorized": False, "delivery_package_created": False}
    save(OUT / "result.json", result)
    print(primary[["group", "policy", "net_sharpe", "cagr", "max_drawdown", "entries", "worth_followup"]].to_string(index=False), flush=True)


def record():
    result = read(OUT / "result.json")
    table = pd.read_csv(OUT / "all_account_metrics.csv")
    assert len(table) == 32 and not (OUT / "program_update_receipt.json").exists()
    for row in table.itertuples():
        ledger = pd.read_parquet(ROOT / row.ledger_path)
        metric = core.metrics(ledger, row.capital)
        for key in ["cagr", "max_drawdown", "end_equity", "commission", "slippage"]:
            assert np.isclose(metric[key], getattr(row, key), rtol=1e-10, atol=1e-8)
        if pd.notna(row.net_sharpe):
            assert np.isclose(metric["net_sharpe"], row.net_sharpe, atol=1e-10)
    save(OUT / "saved_verification_receipt.json", {"at": core.now(), "status": "PASS_32_SAVED_ACCOUNTS_AND_CAUSAL_PREFIX",
                                                   "max_accounting_error": float(table.max_identity_error.max())})
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    names = {"IF_HHI_HIGH": "IF异常高集中", "IF_HHI_LOW": "IF异常低集中",
             "ETF_MIGRATION_PROXY_HIGH": "510300相对份额强", "ETF_MIGRATION_PROXY_LOW": "510300相对份额弱",
             "PRICE_CONTROL": "同频价格对照", "BUY_HOLD_50": "半仓买入持有背景"}
    lines = ["本轮保留只交易510300及现金、成本后夏普1.2目标。先用现成资料完成4个新方向与匹配对照，共32个账户。",
             "IF是持仓在合约间的分布，并按同剩余期限比较；ETF是510300相对另外两只沪深300ETF的周度份额差。不是重跑此前失败的总申赎或基差规则。",
             "| 数据及区间 | 方向 | 压力夏普 | 年化收益 | 最大回撤 | 入场次数 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        period = "IF 2017-2025" if row.group == "IF_HHI" else "ETF 2022-2025"
        val = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "无交易"
        lines.append(f"| {period} | {names[row.policy]} | {val} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.entries} |")
    promising = result["worth_followup"]
    conclusion = "本轮没有通过预先初筛条件的新方向，停止这四条简单规则的深化。" if not promising else "值得进一步研究的候选："+"、".join(promising)+"；尚未正式达标。"
    lines += ["", conclusion,
              "数据和因果边界：按统计日后一个交易日收盘可用、再下一开盘成交；历史首次发布时间没有逐条证实。ETF仅三只上海产品，用原始价格估值份额差，非完整日频NAV体系。IF名义到期日由合约代码和第三个周五规则计算，避免用未来最后出现日；假期导致名义日期已过及合约数异常的观测不使用。",
              "两种机制区间不同，只与各自匹配对照比较。保留全部现金日、分红、费用、T+1、整手和风险退出；保存账本复算32账户，截断未来数据重算特征及信号一致。所有历史已受多轮研究，不能称独立验证。",
              "相关官方字段口径：[IF合约及持仓量](https://www.cffex.com.cn/cn/hs300.html)，[上交所ETF总份额](https://www.sse.com.cn/market/funddata/volumn/etfvolumn/)。这些页面不补足历史首次发布时间。",
              "下一动作由账户效果决定；本轮未创建交付包，未改变交易资产，也未恢复任何采集或执行任务。"]
    (OUT / "快速检验结论.md").write_text("\n\n".join(lines[:2])+"\n\n"+"\n".join(lines[2:]), encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / "status.json", program / "factor_progress.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    objects = [read(p) for p in paths]
    status, factors, mandate = objects
    assert status["latest_round"] == "510300_FACTOR96_RAPID_CAPITAL_PROXY_V1"
    before = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    rel = OUT.relative_to(ROOT).as_posix()
    note = f"已完成产品迁移代理和IF期限条件持仓集中度4个方向、32账户，初筛待深化候选{len(promising)}，正式达标0；旧失败保留。"
    status.update(at=core.now(), latest_round=STUDY, latest_result=rel+"/result.json", last_research_result=note,
                  last_completed_account_experiment=STUDY, last_completed_account_result=rel+"/result.json",
                  latest_progress_receipt=rel+"/saved_verification_receipt.json",
                  latest_continuation_classification="PROGRESS_32_STRUCTURE_ACCOUNTS_WITH_MATCHED_CONTROLS",
                  admitted_account_scenarios_this_round=32, invalid_implementation_accounts_this_round=0,
                  cumulative_admitted_account_scenarios=status["cumulative_admitted_account_scenarios"]+32,
                  cumulative_executed_account_scenarios=status["cumulative_executed_account_scenarios"]+32,
                  rapid_screen_direction_tests=status["rapid_screen_direction_tests"]+4,
                  current_research_phase="DEEPEN_PROMISING_STRUCTURE_ONLY" if promising else "STRUCTURE_SIMPLE_SCREENS_ENDED_NO_PROMISING_CANDIDATE",
                  next_independent_source_action="DEEPEN_FROZEN_PROMISING_STRUCTURE" if promising else "SELECT_DIFFERENT_MECHANISM_FROM_READY_DATA_OR_DISCUSS_DIRECTION",
                  goal_status="active", goal_achieved=False, qualified_candidates=[], source_detail_work_deferred_by_user=True)
    if not promising:
        status["stopped_simple_research_branches"].append("4_ETF_MIGRATION_AND_IF_CONCENTRATION_DIRECTIONS")
    for factor in factors:
        if factor["id"] in ["G04", "H05"]:
            factor["rapid_structure_screen"] = {"study_id": STUDY, "result": rel+"/result.json",
                                                 "status": "PROMISING_CANDIDATE_REQUIRES_REVIEW" if promising else "NO_PROMISING_FIXED_SIMPLE_RULE",
                                                 "strict_original_definition_validated": False}
    mandate.update(current_round=STUDY, current_protocol=rel+"/protocol.json", last_research_result=note,
                   latest_progress_receipt=rel+"/saved_verification_receipt.json", latest_continuation_report=rel+"/快速检验结论.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    for path, obj in zip(paths, objects):
        update(path, core.clean(obj))
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": core.now(), "before": before,
                                              "after": [{"path": str(p), "sha256": core.digest(p)} for p in paths],
                                              "new_accounts": 32, "goal_achieved": False})
    print(note, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run", "record"])
    stage = parser.parse_args().stage
    {"freeze": freeze, "run": run, "record": record}[stage]()
