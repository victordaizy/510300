"""E03：同成员均线参与和收盘新高新低的恢复速度，比较旧广度及价格。"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research import factor96_rapid_feasibility_v1 as rapid
from scripts.record_factor96_remaining_changes_v1 import update

ROOT, core = rapid.ROOT, rapid.core
OUT = ROOT / "reports/research/510300_factor96_rapid_breadth_speed_v1"
STUDY = "510300_FACTOR96_RAPID_BREADTH_SPEED_V1"
INPUT = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs"
START, END = "2017-01-03", "2025-12-31"
KEYS = ["MA20_CHANGE5", "HIGH_LOW_CHANGE5", "E01_LEVEL", "E01_CHANGE5"]
CANDIDATES = [key+"_"+side for key in KEYS[:2] for side in ["HIGH", "LOW"]]
POLICIES = CANDIDATES + ["E01_LEVEL_HIGH", "E01_CHANGE5_HIGH", "PRICE_CONTROL", "BUY_HOLD_50"]
read, save = rapid.read, rapid.save


def freeze():
    assert not OUT.exists(), "不覆盖既有实验"
    save(OUT / "protocol.json", {
        "at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "previous_turn": "PROGRESS_32_EXTERNAL_RESILIENCE_ACCOUNTS_AND_ZERO_INCREMENT_FINDING",
        "question": "同样ETF价格改善与旧20日上涨广度下，内部站上均线和创出收盘新高的参与速度是否提供额外收益信息。",
        "data": "复用点时沪深300成员、四态总股东收益及逐日可用覆盖，不新增下载或历史权重建设。公司行为不可解释和缺失保持未知，官方停牌仅沿用已核验收益。",
        "wealth": "按每只股票连续有效收益区间重建任意基点财富收盘；缺失处切断区间，不用零填未知。所有成员必须在当前及5日前各有21个连续有效财富收盘。",
        "same_members": "取当前和5日前的历史成员交集，并要求上述两端窗口完整；同一组名单分别计算两天比例后相减。两端均300成员、VIEW_ALLOWED，同组可用数至少294；不足不产生信号。",
        "features": {"MA20_CHANGE5": "同组成员中财富收盘高于含当日MA20的比例，减5日前同组比例。",
                     "HIGH_LOW_CHANGE5": "当前收盘严格高于此前20日财富收盘最高的比例，减严格低于此前20日财富收盘最低的比例；再减5日前同组的这个差。只用收盘极值，不冒充日内高低。",
                     "E01_LEVEL": "当前同组成员过去20日复合总回报为正的比例，作为旧广度水平对照。",
                     "E01_CHANGE5": "当前同组20日上涨比例减5日前同组比例，作为广度速度对照。"},
        "ties": "财富均线及高低关系相对容差1e-12内视为相等，零对数收益容差1e-12内不计上涨；阈值不据结果调整。",
        "source_clock": "所有成分统计整体滞后一交易日进入收盘判断，随后次开盘执行；历史实际首次供应商交付未证实。缺失不前填、不因缺失自动认定弱势。",
        "signals": "四个变量都用此前两日历年、至少120个有效值的30/70分位；当前值不进入阈值。四者资料均合格时才可比较。两个E03变量各HIGH/LOW四候选，E01水平与变化各HIGH作固定对照。所有入场均另要求当日ETF过去5日总回报>0，价格对照仅使用该条件。",
        "account": "与前期统一使用固定10个开盘间隔、20万和2万、BASE/STRESS、50%最大仓位、ES及回撤储备和10%永久停止、整手、T+1、涨跌停、分红应收与到账和期末退出准备。现金利息0。",
        "sample": [START, END], "periods": [[START, "2020-12-31"], ["2021-01-01", END]],
        "planned_accounts": 32, "new_direction_candidates": 4, "annual_days": 242, "costs": core.COSTS,
        "followup_filter": "20万STRESS夏普>=0.8、CAGR>=5%、回撤<=10%、入场>=20、前后两期收益正，且夏普及CAGR均高于E01_CHANGE5_HIGH和PRICE_CONTROL。仅用于是否深化，正式目标仍夏普1.2/CAGR10%和独立验证。",
        "correlation": "记录两项E03对旧广度变化及ETF过去5日收益的Spearman。对未来10个开盘间隔含分红固定份额收益，全样本及两子期均报告；重叠标签不视为独立显著性。",
        "overlap_review": "旧多数广度是20日复合收益>0的水平门；T18只将该水平作为预测输入，其5日变化仅作诊断。T02是两次指数试低时成分日内新低收缩；T06是同成员中位收益走弱。本次均线及收盘新高新低速度不是上述定义，但同属价格派生，不宣称独立经济证据。",
        "no_rescue": "保留全部旧失败；本次失败不改变均线、前窗、分位、价格条件、覆盖或持有期。",
        "orders_authorized": False, "delivery_package_required": False,
    })
    paths = [Path(__file__), Path(rapid.__file__), Path(core.__file__), rapid.ENGINE_PATH,
             rapid.PRICE_BASE / "market.parquet", rapid.PRICE_BASE / "price_features.parquet", rapid.PRICE_BASE / "dividends.csv",
             *[INPUT / name for name in ["classified.parquet", "membership.parquet", "daily_coverage.parquet"]],
             ROOT / "research/factor96_daily_state_shrink_v1.py", ROOT / "docs/510300_BREADTH_MAJORITY_TREND_V1.md", OUT / "protocol.json"]
    save(OUT / "freeze.json", {"at": core.now(), "files": [{"path": str(p), "sha256": core.digest(p)} for p in paths]})
    print("已冻结E03同成员参与速度的4个方向、3个匹配对照和半仓背景，共32账户。", flush=True)


def member_features(m, classified, membership, coverage):
    dates = pd.DatetimeIndex(m.date)
    symbols = pd.Index(sorted(membership.symbol.unique()))
    usable = (classified.return_is_usable & classified.constituent_return_state.isin(["TRADED_VALID", "OFFICIAL_SUSPENSION"])
              & np.isfinite(classified.daily_total_shareholder_return) & classified.daily_total_shareholder_return.gt(-1))
    returns = classified.assign(value=classified.daily_total_shareholder_return.where(usable)).pivot(index="date", columns="symbol", values="value")
    returns = returns.reindex(index=dates, columns=symbols)
    member = membership.rename(columns={"membership_date": "date"}).assign(present=True)
    member = member.pivot(index="date", columns="symbol", values="present").reindex(index=dates, columns=symbols).eq(True)
    allowed = coverage.set_index("date").aggregation_state.reindex(dates).eq("VIEW_ALLOWED")
    logs = np.log1p(returns)
    wealth = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for symbol in symbols:
        values = logs[symbol]
        wealth[symbol] = np.exp(values.groupby(values.isna().cumsum()).cumsum())
    complete = wealth.notna().rolling(21, min_periods=21).sum().eq(21)
    common = member & member.shift(5, fill_value=False) & complete & complete.shift(5, fill_value=False)
    count = common.sum(axis=1)
    known = count.ge(294) & member.sum(axis=1).eq(300) & member.shift(5, fill_value=False).sum(axis=1).eq(300)
    known &= allowed & allowed.shift(5, fill_value=False)
    above = wealth.gt(wealth.rolling(20, min_periods=20).mean()*(1+1e-12))
    high = wealth.gt(wealth.shift().rolling(20, min_periods=20).max()*(1+1e-12))
    low = wealth.lt(wealth.shift().rolling(20, min_periods=20).min()*(1-1e-12))
    positive = logs.rolling(20, min_periods=20).sum().gt(1e-12)
    denominator = count.where(count.gt(0))
    def fraction(flags):
        return (flags & common).sum(axis=1)/denominator
    current_above, old_above = fraction(above), fraction(above.shift(5, fill_value=False))
    current_hilo, old_hilo = fraction(high)-fraction(low), fraction(high.shift(5, fill_value=False))-fraction(low.shift(5, fill_value=False))
    current_e01, old_e01 = fraction(positive), fraction(positive.shift(5, fill_value=False))
    f = pd.DataFrame({"date": dates, "common_members": count.to_numpy(), "source_known": known.to_numpy(),
                      "current_ma20_fraction": current_above.to_numpy(), "prior_ma20_same_members": old_above.to_numpy(),
                      "current_high_low_fraction": current_hilo.to_numpy(), "prior_high_low_same_members": old_hilo.to_numpy(),
                      "MA20_CHANGE5": (current_above-old_above).where(known).to_numpy(),
                      "HIGH_LOW_CHANGE5": (current_hilo-old_hilo).where(known).to_numpy(),
                      "E01_LEVEL": current_e01.where(known).to_numpy(), "E01_CHANGE5": (current_e01-old_e01).where(known).to_numpy()})
    return f


def signals(m, risk, source):
    f = source.copy()
    dates = pd.DatetimeIndex(f.date)
    for key in KEYS:
        raw = f[key].to_numpy(float)
        low, high = np.full(len(f), np.nan), np.full(len(f), np.nan)
        count = np.zeros(len(f), int)
        for i, day in enumerate(dates):
            first = dates.searchsorted(day-pd.DateOffset(years=2))
            history = raw[first:i]
            history = history[np.isfinite(history)]
            count[i] = len(history)
            if len(history) >= 120:
                low[i], high[i] = np.quantile(history, [.3, .7])
        f[key+"_q30"], f[key+"_q70"], f[key+"_history_count"] = low, high, count
        f[key+"_known"] = np.isfinite(raw) & np.isfinite(low) & (high > low)
        f[key+"_HIGH"] = f[key+"_known"] & (raw >= high)
        f[key+"_LOW"] = f[key+"_known"] & (raw <= low)
    f["common_known"] = f[[key+"_known" for key in KEYS]].all(axis=1)
    f = f.rename(columns={"date": "source_date"}).shift(1)
    f.insert(0, "date", m.date.to_numpy())
    f["common_known"] = f.common_known.eq(True)
    f["price_return5"] = np.log((m.close+m.dividend)/m.close.shift()).rolling(5).sum().to_numpy()
    f["PRICE_CONTROL"] = f.common_known & f.price_return5.gt(0)
    for key in POLICIES[:6]:
        f[key] = f[key].eq(True) & f.PRICE_CONTROL
    f["es95"] = risk.es95.to_numpy()
    return f


def diagnostics(f, m, dividends):
    rows = []
    for i in np.flatnonzero(f.common_known & f.date.between(START, END)):
        a, b = i+1, i+11
        if b >= len(m):
            continue
        entitled = dividends.record_date.ge(m.date.iloc[a]) & dividends.record_date.lt(m.date.iloc[b]) & dividends.ex_date.le(m.date.iloc[b])
        y = (m.open.iloc[b]-m.open.iloc[a]+dividends.loc[entitled, "cash_dividend_per_share"].sum())/m.open.iloc[a]
        rows.append({"date": f.date.iloc[i], **{k: float(f[k].iloc[i]) for k in KEYS},
                     "price_return5": float(f.price_return5.iloc[i]), "future_return10": float(y)})
    labels = pd.DataFrame(rows)
    labels.to_parquet(OUT / "diagnostic_labels.parquet", index=False)
    results = []
    for key in KEYS[:2]:
        for period, a, b in [("full", START, END), ("early", START, "2020-12-31"), ("late", "2021-01-01", END)]:
            sample = labels.loc[labels.date.between(a, b)]
            results.append({"factor": key, "period": period, "observations": len(sample),
                            "future_spearman": float(sample[key].corr(sample.future_return10, method="spearman")),
                            "old_breadth_change_spearman": float(sample[key].corr(sample.E01_CHANGE5, method="spearman")),
                            "price_return5_spearman": float(sample[key].corr(sample.price_return5, method="spearman"))})
    save(OUT / "correlation_diagnostic.json", {"at": core.now(), "rows": results, "independent_validation": False,
                                               "overlapping_labels": True, "significance_claim": False})
    return results


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
    data = [pd.read_parquet(INPUT / name) for name in ["classified.parquet", "membership.parquet", "daily_coverage.parquet"]]
    for d, col in zip(data, ["date", "membership_date", "date"]):
        d[col] = pd.to_datetime(d[col])
        d.drop(d.index[d[col].gt(END)], inplace=True)
    source = member_features(m, *data)
    source.to_parquet(OUT / "same_member_features.parquet", index=False)
    f = signals(m, risk, source)
    f.to_parquet(OUT / "daily_signals.parquet", index=False)
    cut = pd.Timestamp("2021-01-04")
    earlier = [d.loc[d[col].le(cut)].copy() for d, col in zip(data, ["date", "membership_date", "date"])]
    short_m, short_risk = m.loc[m.date.le(cut)].copy(), risk.loc[risk.date.le(cut)].copy()
    short_source = member_features(short_m, *earlier)
    pd.testing.assert_frame_equal(source.loc[source.date.le(cut)].reset_index(drop=True), short_source)
    pd.testing.assert_frame_equal(f.loc[f.date.le(cut)].reset_index(drop=True), signals(short_m, short_risk, short_source))
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    spec = importlib.util.spec_from_file_location("breadth_speed_account_engine", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    rapid.START, rapid.END, rapid.HOLD = START, END, 10
    records = []
    for capital in [200000, 20000]:
        for cost in ["BASE", "STRESS"]:
            for policy in POLICIES:
                ledger, orders = rapid.simulate(m, f, dividends, policy, capital, cost, engine)
                folder = OUT / "accounts" / f"{capital}_{cost}_{policy}"
                folder.mkdir(parents=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                orders.to_parquet(folder / "orders.parquet", index=False)
                row = core.metrics(ledger, capital)
                row.update(policy=policy, capital=capital, cost=cost,
                           entries=int(((ledger.filled_quantity > 0) & ledger.shares_before.eq(0)).sum()),
                           stopped_at=ledger.loc[ledger.risk_stopped, "date"].min().isoformat() if ledger.risk_stopped.any() else None,
                           ledger_path=str((folder / "ledger.parquet").relative_to(ROOT)))
                for label, a, b in [("early", START, "2020-12-31"), ("late", "2021-01-01", END)]:
                    row.update({label+"_"+k: v for k, v in rapid.period_metrics(ledger, a, b).items()})
                records.append(row)
            print(f"已完成{capital}元、{cost}成本的8个广度速度账户。", flush=True)
    table = pd.DataFrame(records)
    table["candidate"] = table.policy.isin(CANDIDATES)
    table["beats_controls"] = True
    for (capital, cost), part in table.groupby(["capital", "cost"]):
        for control in ["E01_CHANGE5_HIGH", "PRICE_CONTROL"]:
            base = part.loc[part.policy.eq(control)].iloc[0]
            table.loc[part.index, "beats_controls"] &= part.net_sharpe.gt(base.net_sharpe) & part.cagr.gt(base.cagr)
    table["joint_historical_target"] = table.candidate & table.net_sharpe.ge(1.2) & table.cagr.ge(.1) & table.max_drawdown.le(.1)
    table["worth_followup"] = (table.candidate & table.net_sharpe.ge(.8) & table.cagr.ge(.05) & table.max_drawdown.le(.1)
                               & table.entries.ge(20) & table.early_net_profit_fraction.gt(0) & table.late_net_profit_fraction.gt(0)
                               & table.beats_controls)
    table.to_csv(OUT / "all_account_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS")]
    primary.to_csv(OUT / "20万元压力成本比较.csv", index=False, encoding="utf-8-sig")
    corr = diagnostics(f, m, dividends)
    evaluation = f.loc[f.date.between(START, END)]
    save(OUT / "result.json", {"at": core.now(), "study_id": STUDY, "classification": "EXPLORATORY_RAPID_SCREEN_NOT_ACCEPTANCE",
        "account_scenarios": len(table), "new_directions": 4, "eligible_decision_days": int(evaluation.common_known.sum()),
        "source_known_days": int(source.source_known[source.date.between(START, END)].sum()),
        "primary_results": primary.to_dict("records"), "correlation_diagnostic": corr,
        "worth_followup": primary.loc[primary.worth_followup, "policy"].tolist(), "joint_historical_target_count": int(primary.joint_historical_target.sum()),
        "causal_prefix_check": "PASS", "independent_forward_observations": 0, "qualified_candidates": 0,
        "goal_achieved": False, "new_source_downloads": 0, "orders_authorized": False, "delivery_package_created": False})
    print(primary[["policy", "net_sharpe", "cagr", "max_drawdown", "entries", "worth_followup"]].to_string(index=False), flush=True)
    print("相关性诊断：", corr, flush=True)


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
    decision = "四个新方向均未通过初筛，本轮简单规则结束。" if not promising else "待深化候选："+"、".join(promising)+"，尚未正式达标。"
    names = {"MA20_CHANGE5_HIGH": "均线参与恢复快", "MA20_CHANGE5_LOW": "均线参与恢复慢",
             "HIGH_LOW_CHANGE5_HIGH": "新高减新低扩散快", "HIGH_LOW_CHANGE5_LOW": "新高减新低扩散慢",
             "E01_LEVEL_HIGH": "旧20日上涨广度水平对照", "E01_CHANGE5_HIGH": "旧广度变化对照", "PRICE_CONTROL": "ETF价格对照", "BUY_HOLD_50": "半仓背景"}
    lines = ["继续只交易510300和现金，以完整账户成本后夏普1.2为目标。本轮固定同成员比较内部参与速度，4个方向加4个对照，共32账户。",
             decision, "2017-2025年，20万元压力成本：", "| 规则 | 净夏普 | 年化收益 | 最大回撤 | 入场次数 |", "|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        value = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "无交易"
        lines.append(f"| {names[row.policy]} | {value} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.entries} |")
    lines += ["", f"相同成员和阈值资料门允许{result['eligible_decision_days']}个判断日；所有现金日仍保留在完整账户。", "| 指标 | 时段 | 对未来收益的秩相关 | 对旧广度变化的秩相关 |", "|---|---|---:|---:|"]
    for row in result["correlation_diagnostic"]:
        lines.append(f"| {row['factor']} | {row['period']} | {row['future_spearman']:.3f} | {row['old_breadth_change_spearman']:.3f} |")
    lines += ["", "当前与5日前使用同组至少294名历史有效成员；缺失收益切断财富区间，不填零。成员变化、公司行为未知、覆盖不足不能制造参与改善；所有成分统计延后一交易日使用。均线参与和收盘新高新低速度与旧20日正收益广度定义不同，但仍属于同一家族的价格派生信息。",
              "32份账本复算与截断未来输入检查通过。源资料为历史重建，不证明实际首次交付时点；重叠未来标签不当作独立观察或显著性。旧多数广度、T02、T06、T18失败保持不变。没有交付包、没有下单。"]
    (OUT / "快速检验结论.md").write_text("\n\n".join(lines[:3])+"\n\n"+"\n".join(lines[3:]), encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program / "status.json", program / "factor_progress.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    objects = [read(p) for p in paths]
    status, factors, mandate = objects
    assert status["latest_round"] == "510300_FACTOR96_RAPID_EXTERNAL_RESILIENCE_V1"
    before = [{"path": str(p), "sha256": core.digest(p)} for p in paths]
    rel = OUT.relative_to(ROOT).as_posix()
    note = f"E03同成员参与恢复速度4方向、32账户完成，待深化{len(promising)}，正式达标0；与旧广度变化及价格对照同台比较。"
    status.update(at=core.now(), latest_round=STUDY, latest_result=rel+"/result.json", last_research_result=note,
        last_completed_account_experiment=STUDY, last_completed_account_result=rel+"/result.json", latest_progress_receipt=rel+"/saved_verification_receipt.json",
        latest_continuation_classification="PROGRESS_32_SAME_MEMBER_BREADTH_SPEED_ACCOUNTS",
        admitted_account_scenarios_this_round=32, invalid_implementation_accounts_this_round=0,
        cumulative_admitted_account_scenarios=status["cumulative_admitted_account_scenarios"]+32,
        cumulative_executed_account_scenarios=status["cumulative_executed_account_scenarios"]+32,
        rapid_screen_direction_tests=status["rapid_screen_direction_tests"]+4,
        current_research_phase="DEEPEN_PROMISING_BREADTH_SPEED" if promising else "BREADTH_SPEED_FIXED_RULES_ENDED",
        next_independent_source_action="DEEPEN_FROZEN_BREADTH_SPEED" if promising else "REASSESS_READY_DISTINCT_MECHANISMS_WITHOUT_REPEATING_PRICE_DERIVATIVES",
        goal_status="active", goal_achieved=False, qualified_candidates=[], source_detail_work_deferred_by_user=True)
    if not promising:
        status["stopped_simple_research_branches"].append("4_SAME_MEMBER_BREADTH_SPEED_DIRECTIONS")
    for row in factors:
        if row["id"] == "E03":
            row["rapid_same_member_screen"] = {"study_id": STUDY, "result": rel+"/result.json", "status": "PROMISING_REQUIRES_REVIEW" if promising else "NO_PROMISING_FIXED_SIMPLE_RULE", "price_extreme_definition": "TOTAL_RETURN_CLOSE_NOT_INTRADAY_HIGH_LOW"}
    mandate.update(current_round=STUDY, current_protocol=rel+"/protocol.json", last_research_result=note,
        latest_progress_receipt=rel+"/saved_verification_receipt.json", latest_continuation_report=rel+"/快速检验结论.md",
        latest_continuation_classification=status["latest_continuation_classification"], research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
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
