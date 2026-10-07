"""一次固定的账户测量与窄幅突破增量比较，不按结果继续改参。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import daily_supply_test_v1 as risk_base
from research import point_account_nr7_inputs_v1 as engine
from research import upward_episode_anatomy_v1 as common
from research.adaptive_allocation_v1 import normalize_dividends

SOURCE = ROOT / "reports/research/510300_point_current_observation_20261001"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
OUT = ROOT / "reports/research/510300_point_account_nr7_complement_v1"
COSTS = ("BASE", "STRESS")


def save_table(name, frame):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p.with_suffix(".parquet"), index=False)
    frame.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("该研究已有记录，不覆盖或重选规则。")
    OUT.mkdir(parents=True)
    files = {}
    source_files = [
        (SOURCE / "inputs/candidate_prices.parquet", "inputs/prices.parquet"),
        (SOURCE / "inputs/dividends.csv", "inputs/dividends.csv"),
        (SOURCE / "inputs/candidate_dividend_coverage.json", "inputs/dividend_coverage.json"),
        (SOURCE / "results/完整候选意向.parquet", "inputs/parent_signals.parquet"),
        (SOURCE / "results/全部自然点位.parquet", "inputs/original_points.parquet"),
        (CONTEXT / "active_goal_effective_requirements.json", "inputs/effective_requirements.json"),
        (CONTEXT / "account_return_sharpe_scope_addendum_20261001.json", "inputs/user_scope.json"),
    ]
    for file in [Path(__file__), Path(engine.__file__), Path(risk_base.__file__), Path(common.__file__),
                 ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_point_account_nr7_v1.py"]:
        source_files.append((file, "code/" + file.name))
    for src, relative in source_files:
        dest = OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        files[relative] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(dest)}
    protocol = {
        "study": "510300_POINT_ACCOUNT_NR7_COMPLEMENT_V1", "at": common.now(),
        "user_instruction": "意思是，我们的交易还不够，还要继续提高我们的收益夏普率",
        "question": "当前两个固定点位映射到共同现金账户后表现如何；一条不同时间尺度的日线窄幅突破能否补空仓并同时提高净收益、夏普和有效次数？",
        "known_before_freeze": "此前已知道两点位2020年以来pB约1.0951/1.0289，2020—2025年均约4.17/4次、机会重合。旧模型加减仓账户、常规压缩后20日突破和RSI失败摆动均有冻结结果，保留原结论。所有历史已用于其他研究。",
        "baseline_semantics": "只读取STRESS父状态的正/零，入场按共同风险预算确定份额，以后只减仓不补仓，父状态零退出。它是对当前点位建立的新资金测量，不是复活旧失败加减仓账户；旧1.53夏普不可继承。两档执行成本使用同一组固定父状态。",
        "capital": 200000, "assets": ["510300.SH", "CASH_CNY"], "long_only": True,
        "bars": ["DAILY", "PREVIOUS_COMPLETED_WEEK_CONTEXT_ONLY"],
        "primary_period": ["2020-01-02", "2026-09-30"],
        "earlier_nr7_diagnostic": ["2015-01-05", "2019-12-31"],
        "policies": list(engine.POLICIES), "primary_increment": "POINT_A_PLUS_NR7 minus POINT_A",
        "nr7": {
            "setup": "当日原始最高减最低的报价刻度宽度，严格小于此前六日每一日的宽度，且大于零。",
            "confirmation": "只接受准备日紧接的下一交易日收盘严格超过准备日最高价，价格以截至各日已除息现金前向平移。没有三日等待、20日新高或MACD过滤。",
            "execution": "确认后的下一交易日开盘仅尝试一次；开盘已低于结构止损或涨停不可买则取消，持仓或当天刚卖出时忽略新事件。",
            "stop": "准备日最低价，按前向现金平移保存；不是当日最低价触及即成交。",
            "target": "入场开盘加两倍开盘至结构止损距离；计划2R不等于实际盈亏比。",
            "exit": "收盘触及止损、计划2R或持有满10个收盘，下一可卖开盘退出；退出受阻持续请求。",
            "parameters_searched": False,
            "combination": "组合空仓且旧A父目标为正时优先旧A；父目标明确为零且有新窄幅确认才进入NR7。持仓按入场来源自身退出，不中途转换身份，不同日卖出再买入。",
        },
        "distinction_from_old_compression": "旧第54轮为5/20/60日收益波动压缩及长期低分位，准备20日后等待20日区间突破；本轮为单日高低幅七日新低、紧接一日收盘突破。保留旧阈值和旧裁决，不从旧失败中搜索救援参数。",
        "costs": {k: {"commission_each_side": risk_base.COSTS[k][0], "slippage_each_side": risk_base.COSTS[k][1],
                      "minimum_fee_cny": 5, "tick": .001, "lot": 100} for k in COSTS},
        "risk": "沿用本任务完整账户：最高50%股票仓位、5日条件ES预算2.5%、10%跳空情景预算5%并受回撤余量约束；估计只用两年内已成熟5日标签，按已知RV20距离选126条、最差7条均值。10%账户收盘回撤触发下一可卖开盘退出并停止新开仓。此约束不保证真实最大回撤一定不超10%。",
        "clock": "次日开盘使用前一收盘的信号、风险和最大份额，开盘价格只允许缩减可买份额，不增加昨晚计划数量；T+1、不虚构日内先后。",
        "dividend": "按登记日收盘份额确认资格，除息形成应收，支付日现金到账；应收不能用于买入。",
        "cash_and_sharpe": "现金收益率及无风险比较率均明确假设0；252交易日年化，全部交易日日收益包含空仓和成本，标准差ddof=1。",
        "terminal": "最后真实收盘按市值和应收记账，不人为清仓；未来退出费用另列估计，不计为已发生成本或完整交易。",
        "trade_quality": "完整账户周期净损益除实际买入支出，含风险减仓；pB>1且均值>0。另保留原固定十万元点位，两个口径不混用。",
        "frequency": "按最终清仓年统计周期；部分减仓订单不增加次数，零交易年份保留，2026单列。",
        "increment_gate": "两成本下组合相对A年化收益和夏普都严格提高、年均完整周期增加、组合回撤不超过10%、组合及NR7单独账户实际净pB>1且均值为正。另分别报告2020—2023与2024—2026结果，不选择最有利时期。即使通过也仅是开发期线索。",
        "uncertainty": "固定20日循环区块、配对重采样2000次、种子20261001，报告收益和夏普增量描述区间；不声称校正长期多次选择或建立独立验证。",
        "sources": ["https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/narrow-range-day-nr7",
                    "https://web.stanford.edu/~wfsharpe/art/sr/sr.htm"],
        "source_limits": "StockCharts给出窄幅日及突破概念；本研究的次日收盘确认、2R、10日期限和账户组合是事前固定的本地改写，来源不提供510300盈利证明。",
        "orders_authorized": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "before_new_account_results": True, "files": files})
    print("新账户测量和唯一NR7补充假说已固定，尚未读取新账户结果。", flush=True)


def paired_interval(a, b):
    n = len(a)
    rng = np.random.default_rng(20261001)
    rows = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=(n + 19) // 20)
        indices = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
        x, y = a[indices], b[indices]
        sx = x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 1e-14 else np.nan
        sy = y.mean() / y.std(ddof=1) * np.sqrt(252) if y.std(ddof=1) > 1e-14 else np.nan
        cx, cy = np.expm1(np.log1p(x).mean() * 252), np.expm1(np.log1p(y).mean() * 252)
        rows.append((sy - sx, cy - cx))
    values = np.asarray(rows)
    return {"sharpe_delta_lower": float(np.nanquantile(values[:, 0], .025)),
            "sharpe_delta_upper": float(np.nanquantile(values[:, 0], .975)),
            "cagr_delta_lower": float(np.nanquantile(values[:, 1], .025)),
            "cagr_delta_upper": float(np.nanquantile(values[:, 1], .975))}


def annual_rows(result, period, policy, cost):
    rows = []
    complete = result["trades"].loc[result["trades"].status.eq("COMPLETE")]
    for year, g in result["daily"].groupby(result["daily"].date.dt.year):
        count = int(complete.exit_date.dt.year.eq(year).sum()) if len(complete) else 0
        rows.append({"period": period, "policy": policy, "cost": cost, "year": int(year),
                     "full_year": int(year) < 2026, "completed_cycles": count,
                     "calendar_year_return": float(np.prod(1 + g.net_return) - 1),
                     **engine.return_statistics(g.net_return)})
    return rows


def verify_account(result):
    d = result["daily"]
    expected = d.equity.to_numpy() / np.r_[200000., d.equity.to_numpy()[:-1]] - 1
    np.testing.assert_allclose(d.net_return, expected, atol=1e-13, rtol=0)
    np.testing.assert_allclose(d.equity, d.cash + d.shares * d.close + d.receivable, atol=1e-7, rtol=0)
    total = float((d.price_pnl + d.dividend_accrual - d.commission - d.slippage).sum())
    if abs(total - (float(d.equity.iloc[-1]) - 200000)) > 1e-6:
        raise AssertionError("账户毛损益、摩擦和净财富不一致。")
    orders = result["orders"]
    if len(orders):
        if not orders.origin.lt(orders.date).all():
            raise AssertionError("存在同时或未来信号成交。")
        signed = np.where(orders.side.eq("BUY"), orders.quantity, -orders.quantity)
        observed = pd.Series(signed, index=pd.DatetimeIndex(orders.date)).groupby(level=0).sum()
        shares = observed.reindex(pd.DatetimeIndex(d.date), fill_value=0).cumsum()
        np.testing.assert_array_equal(d.shares, shares)
    for t in result["trades"].loc[result["trades"].status.eq("COMPLETE")].itertuples():
        if not t.entry_origin < t.entry_date < t.exit_date:
            raise AssertionError("完成交易违反下一开盘或T+1。")
        value = (t.sell_net_cny + t.dividend_cny - t.buy_debit) / t.buy_debit
        if abs(value - t.net_return) > 1e-13:
            raise AssertionError("交易净回报复算不符。")
    return {"days": len(d), "orders": len(orders),
            "completed_cycles": int(result["trades"].status.eq("COMPLETE").sum()),
            "maximum_accounting_error": float(d.accounting_error.abs().max())}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("本固定实验已经开始，不覆盖运行结果。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for name, record in frozen["files"].items():
        if common.digest(OUT / name) != record["sha256"]:
            raise ValueError("固定输入改变：" + name)
        if name.startswith("code/") and common.digest(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("执行程序不再对应事前固定版本：" + name)
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "orders_authorized": False})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    data, _ = common.features(prices, dividends)
    signals = engine.nr7_signals(data)
    parents = pd.read_parquet(OUT / "inputs/parent_signals.parquet").pivot(index="origin", columns="candidate", values="target").reset_index()
    risks, membership = risk_base.risk_estimates(data)
    if not membership.label_exit_idx.le(membership.decision_idx).all():
        raise ValueError("风险估计使用了未成熟标签。")
    for name, frame in [("全部窄幅准备与确认", signals), ("事前风险", risks), ("成熟风险样本", membership)]:
        save_table(name, frame)
    accounts, records, annual, checks = {}, [], [], []
    for period in ("2020_2026", "2015_2019_NR7_DIAGNOSTIC"):
        if period == "2020_2026":
            d, start, policies = data, "2020-01-02", engine.POLICIES
        else:
            d = data.loc[data.date.le(pd.Timestamp("2019-12-31"))].reset_index(drop=True)
            start, policies = "2015-01-05", ("NR7_ONLY",)
        for cost in COSTS:
            for policy in policies:
                result = engine.account(d, dividends, signals.iloc[:len(d)], parents, risks, policy, cost, start)
                key = (period, cost, policy)
                accounts[key] = result
                check = verify_account(result)
                checks.append({"period": period, "cost": cost, "policy": policy, **check})
                for name in ("daily", "trades", "orders", "rejections", "decisions"):
                    save_table(f"accounts/{period}/{cost}/{policy}/{name}", result[name])
                common.save_json(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", result["terminal"])
                m = engine.metrics(result)
                yearly = annual_rows(result, period, policy, cost)
                full_counts = [x["completed_cycles"] for x in yearly if x["full_year"]]
                m["average_full_year_cycles"] = float(np.mean(full_counts))
                m["zero_trade_full_years"] = sum(n == 0 for n in full_counts)
                records.append({"period": period, "cost": cost, "policy": policy, "name": engine.NAMES[policy], **m})
                annual.extend(yearly)
                print(f"{period}/{cost}/{engine.NAMES[policy]}：夏普{m['net_sharpe']:.4f}，年化{m['net_cagr']:.2%}，周期{m['completed_cycles']}。", flush=True)

    metrics = pd.DataFrame(records)
    save_table("完整账户比较", metrics)
    save_table("逐年收益与完整周期", pd.DataFrame(annual))
    save_table("账户复算", pd.DataFrame(checks))
    increments, eras, sources = [], [], []
    for cost in COSTS:
        a = accounts[("2020_2026", cost, "POINT_A")]
        b = accounts[("2020_2026", cost, "POINT_A_PLUS_NR7")]
        am, bm = engine.metrics(a), engine.metrics(b)
        nm = engine.metrics(accounts[("2020_2026", cost, "NR7_ONLY")])
        row = {"cost": cost, "sharpe_delta": bm["net_sharpe"] - am["net_sharpe"],
               "cagr_delta": bm["net_cagr"] - am["net_cagr"],
               "ending_equity_delta": bm["ending_equity"] - am["ending_equity"],
               "cycle_delta": bm["completed_cycles"] - am["completed_cycles"],
               "average_full_year_cycle_delta": bm["average_full_year_cycles"] - am["average_full_year_cycles"],
               **paired_interval(a["daily"].net_return.to_numpy(float), b["daily"].net_return.to_numpy(float))}
        row["fixed_increment_gate_pass"] = bool(
            row["sharpe_delta"] > 0 and row["cagr_delta"] > 0 and row["average_full_year_cycle_delta"] > 0
            and bm["max_drawdown"] <= .1 and bm["p_times_b"] > 1 and bm["mean_cycle_net_return"] > 0
            and nm["p_times_b"] > 1 and nm["mean_cycle_net_return"] > 0)
        increments.append(row)
        for era, left, right in [("2020_2023", "2020-01-01", "2023-12-31"), ("2024_2026", "2024-01-01", "2026-09-30")]:
            for policy in engine.POLICIES:
                result = accounts[("2020_2026", cost, policy)]
                part = result["daily"].loc[result["daily"].date.between(left, right)]
                eras.append({"cost": cost, "era": era, "policy": policy, **engine.return_statistics(part.net_return)})
        for policy in engine.POLICIES:
            result = accounts[("2020_2026", cost, policy)]
            for source in ("CORE", "NR7"):
                t = result["trades"].loc[result["trades"].source.eq(source)]
                if len(t):
                    sources.append({"cost": cost, "policy": policy, "source": source, **engine.trade_statistics(t)})
    save_table("新增机会对账户的增量", pd.DataFrame(increments))
    save_table("分时期账户比较", pd.DataFrame(eras))
    save_table("按入场来源的交易质量", pd.DataFrame(sources))

    prefix = []
    for end in ("2021-12-31", "2023-12-29", "2025-12-31"):
        short = data.loc[data.date.le(pd.Timestamp(end))].reset_index(drop=True)
        sf = engine.nr7_signals(short)
        pd.testing.assert_frame_equal(signals.iloc[:len(short)].reset_index(drop=True), sf)
        sr, _ = risk_base.risk_estimates(short)
        pd.testing.assert_frame_equal(risks.iloc[:len(sr)].reset_index(drop=True), sr)
        result = engine.account(short, dividends, sf, parents, sr, "POINT_A_PLUS_NR7", "STRESS", "2020-01-02")
        full = accounts[("2020_2026", "STRESS", "POINT_A_PLUS_NR7")]["daily"]
        pd.testing.assert_frame_equal(full.iloc[:len(result["daily"])].reset_index(drop=True), result["daily"])
        prefix.append({"cutoff": end, "days": len(result["daily"]), "exact_daily_match": True})
    save_table("历史截断一致性", pd.DataFrame(prefix))
    summary = {"study": "510300_POINT_ACCOUNT_NR7_COMPLEMENT_V1", "at": common.now(),
               "status": "DEVELOPMENT_INCREMENT_ONLY" if all(x["fixed_increment_gate_pass"] for x in increments) else "REJECTED_FROZEN_NR7_INCREMENT",
               "data_cutoff": "2026-09-30", "new_account_evaluations": len(accounts),
               "new_candidate_mechanisms": 1, "parameter_search": False,
               "main_period_nr7_confirmations": int(signals.entry_event.loc[data.date.between("2019-12-31", "2026-09-29")].sum()),
               "all_history_nr7_confirmations_since_2015": int(signals.entry_event.loc[data.date.ge("2015-01-01")].sum()),
               "paired_increment": increments, "necessary_tests_passed": 6,
               "historical_prefix_checks": prefix, "max_daily_accounting_error": max(x["maximum_accounting_error"] for x in checks),
               "risk_sample_time_checks": len(membership), "prospective_completed_points": 0,
               "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    common.save_json(OUT / "acceptance_outcome.json", {"at": common.now(), "status": summary["status"],
        "goal_achieved": False, "old_candidate_statuses_unchanged": True,
        "decision": "固定规则未同时改善收益、夏普、有效次数和实际pB，保留结果，不改NR7窗口、目标或持有期救回。"
        if summary["status"].startswith("REJECTED") else "只保留开发期增量线索，仍需独立前瞻证据。"})
    report(metrics, pd.DataFrame(annual), pd.DataFrame(increments), pd.DataFrame(eras), pd.DataFrame(sources), summary, accounts)
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(), "summary_sha256": common.digest(OUT / "summary.json"),
        "report_sha256": common.digest(OUT / "研究结论.md"), "account_metrics_sha256": common.digest(OUT / "results/完整账户比较.parquet"),
        "saved_account_recomputed": True, "independent_validation": "NOT_ESTABLISHED"})
    state = json.loads((CONTEXT / "state.json").read_text(encoding="utf-8"))
    state.update(updated_at=common.now(), status="research_in_progress", latest_user_instruction="意思是，我们的交易还不够，还要继续提高我们的收益夏普率",
                 current_phase="LONG_ACCOUNT_RETURN_AND_SHARPE_IMPROVEMENT", current_priority="LONG_ACCOUNT_RETURN_AND_SHARPE_IMPROVEMENT",
                 latest_completed_study=summary["study"], latest_result=str((OUT / "summary.json").relative_to(ROOT)),
                 latest_report=str((OUT / "研究结论.md").relative_to(ROOT)), latest_research_status=summary["status"],
                 new_investment_account_evaluations_this_continuation=len(accounts), previous_goal_turn_classification="progress",
                 blocked_audit_count=0, goal_achieved=False, quality_gate_preserved="同口径净收益与净夏普改进、实际净pB>1、次数软目标；日周线、多头优先。",
                 latest_progress="恢复账户目标，完成两条点位共同风险账户基线及唯一NR7补充机制的两成本比较、较早时期诊断。",
                 account_scope_addendum="account_return_sharpe_scope_addendum_20261001.json")
    common.save_json(CONTEXT / "state.json", state)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def report(metrics, annual, increments, eras, sources, summary, accounts):
    rows = ["# 510300：从点位质量回到完整账户收益与夏普", "",
            "用户最新要求是继续增加有效交易并提高收益和夏普。本轮据此恢复完整账户研究，交易次数仍为软目标。数据截至2026年9月30日，历史全部属于重复使用的开发资料，前瞻完成点位仍为0。", "",
            "完成了10条账户测量：2020年以来四个方案各两档成本，另有窄幅突破2015—2019的两档成本诊断。只有一条新机制，没有参数网格。两个旧点位的固定规则和原失败策略的关闭结论都保留。", "",
            "共同口径为20万元、510300与现金、无杠杆、最高50%股票预算，并沿用五日ES和回撤余量约束。信号收盘已知、次日开盘执行；应收股息和现金分账，空仓日完整计入，现金及无风险比较率明确假设为0，252日年化。期末持仓按最后真实收盘计价，不人为清仓。", "",
            "**这里的基线是把当前点位放入共同风险账户后的新测量。** 原来每笔固定约十万元的点位胜率和pB不等于账户统计；旧源模型的动态加减仓账户也不能把夏普直接转给这两个点位。", "",
            "## 同期完整账户结果", "",
            "| 方案 | 成本 | 净夏普 | 净年化 | 最大回撤 | 2020—2025年均周期 | 完成周期 | 实际净pB | 期末权益 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.loc[metrics.period.eq("2020_2026")].itertuples():
        rows.append(f"| {r.name} | {r.cost} | {r.net_sharpe:.3f} | {r.net_cagr:.2%} | {r.max_drawdown:.2%} | {r.average_full_year_cycles:.2f} | {r.completed_cycles} | {r.p_times_b:.3f} | {r.ending_equity:,.2f}元 |")
    rows += ["", "BASE为单边万二佣金、万五滑点；STRESS为单边万四佣金、千一滑点，均含至少5元佣金、0.001元不利刻度与100份整手。风险预算统一按压力成本检查。周期从空仓到最终清仓，部分减仓订单不增加次数。", "",
             "## 新增机会是否值得加入", "",
             "窄幅突破的准备条件是当日高低幅严格小于此前六日；只承认下一交易日收盘超过准备日最高价，再下一开盘入场。准备日低点为结构失效，计划目标为入场至失效距离的两倍，最长持有10个收盘；全部退出由收盘触发后到下一可卖开盘执行。", "",
             "组合只在原A目标明确为零且账户空仓时接纳新机会，已有持仓始终按原入场来源退出；没有把持仓中间的另一个信号重复计为一笔交易。", "",
             "| 成本 | 夏普变化 | 年化变化 | 年均周期变化 | 期末资金变化 | 夏普变化95%描述区间 | 固定增量门槛 |",
             "|---|---:|---:|---:|---:|---|---|"]
    for r in increments.itertuples():
        rows.append(f"| {r.cost} | {r.sharpe_delta:+.3f} | {r.cagr_delta*100:+.2f}个百分点 | {r.average_full_year_cycle_delta:+.2f} | {r.ending_equity_delta:+,.2f}元 | [{r.sharpe_delta_lower:.3f}, {r.sharpe_delta_upper:.3f}] | {'通过开发期门槛' if r.fixed_increment_gate_pass else '未通过'} |")
    rows += ["", "固定门槛同时要求两成本下收益和夏普提高、次数增加、组合回撤不超过10%、组合及新增方法自身实际净pB>1且净均值为正。区间来自固定20日配对循环区块重采样，仅描述已有样本，不是独立验证或反复研究后的显著性证明。", "",
             "## 压力成本下的次数及分时期表现", "",
             "| 年份 | 原A周期 | 组合周期 | 原A当年收益 | 组合当年收益 |", "|---|---:|---:|---:|---:|"]
    sub = annual.loc[annual.period.eq("2020_2026") & annual.cost.eq("STRESS")]
    for year in sorted(sub.year.unique()):
        a = sub.loc[sub.year.eq(year) & sub.policy.eq("POINT_A")].iloc[0]
        b = sub.loc[sub.year.eq(year) & sub.policy.eq("POINT_A_PLUS_NR7")].iloc[0]
        rows.append(f"| {year}{'（至9月30日）' if year == 2026 else ''} | {int(a.completed_cycles)} | {int(b.completed_cycles)} | {a.calendar_year_return:.2%} | {b.calendar_year_return:.2%} |")
    rows += ["", "| 时期 | 方案 | 净夏普 | 净年化 | 最大回撤 |", "|---|---|---:|---:|---:|"]
    for r in eras.loc[eras.cost.eq("STRESS") & eras.policy.isin(["POINT_A", "POINT_A_PLUS_NR7"])].itertuples():
        rows.append(f"| {r.era} | {engine.NAMES[r.policy]} | {r.net_sharpe:.3f} | {r.net_cagr:.2%} | {r.max_drawdown:.2%} |")
    rows += ["", "分段只切开连续账户的实际日收益，不在段首重新启动账户或撤销此前风险状态。", "",
             "| 窄幅突破单独账户时期 | 成本 | 完成周期 | 净胜率 | 实际净B | 实际净pB | 净夏普 | 净年化 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.loc[metrics.policy.eq("NR7_ONLY")].itertuples():
        rows.append(f"| {r.period} | {r.cost} | {r.completed_cycles} | {r.win_rate:.2%} | {r.payoff:.3f} | {r.p_times_b:.3f} | {r.net_sharpe:.3f} | {r.net_cagr:.2%} |")
    rows += ["", "## 收益不足来自哪里", "",
             "| 压力成本方案 | 实际份额下毛损益 | 佣金 | 滑点与刻度 | 账户净损益 | 平均股票仓位 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in metrics.loc[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS")].itertuples():
        rows.append(f"| {r.name} | {r.gross_at_actual_quantities_pnl:+,.2f}元 | {r.total_commission:,.2f}元 | {r.total_slippage:,.2f}元 | {r.ending_equity-200000:+,.2f}元 | {r.mean_exposure:.2%} |")
    rows += ["", "毛损益使用各账户实际成交份额与时点加回摩擦，是解释性账簿分解，不是另一条无成本可执行账户。新增方法若自身缺少毛优势，降低费用也不能自动创造有效信号。", "",
             "| 组合中的入场来源（压力成本） | 完成周期 | 净胜率 | 实际净pB | 平均周期净回报 | 完成周期净损益 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in sources.loc[sources.cost.eq("STRESS") & sources.policy.eq("POINT_A_PLUS_NR7")].itertuples():
        rows.append(f"| {r.source} | {r.completed_cycles} | {r.win_rate:.2%} | {r.p_times_b:.3f} | {r.mean_cycle_net_return:.2%} | {r.completed_cycle_net_pnl:+,.2f}元 |")
    rows += ["", "各来源周期收益只说明实际组合内的贡献，不能视为彼此独立账户。新增持仓也可能占用原机会，所以完整账户差值优先于简单加总各方法盈利。", "",
             "## 本轮裁决与后续方向", "", f"固定新假说状态：**{summary['status']}**。整体目标仍未实现。",
             "", "两条已有点位保留为历史线索；新窄幅假说若没有同时提高收益、夏普和交易质量，就不加入候选。既有RSI失败摆动、二次供给测试、普通周支撑、指标拐点与旧高过拟合混合方案的原裁决保持。",
             "", "下一步首先依据本表区分信号毛优势、持仓期间波动、退出延迟、摩擦和仓位约束。目标是增加扣费后能贡献账户收益的不同机会，并提高已持有期间的单位风险收益；不能仅按交易次数、单笔高胜率或放大仓位宣布改进。",
             "", "要声称实现，需要同一固定规则的收益、夏普、实际pB、回撤和频率共同支持，并补充独立前瞻记录。旧候选的前瞻等待新交易日，新假说研究可继续；本轮没有新行情采集、期权回测、分钟线、委托或自动监控。",
             "", "六项针对性测试通过，三次历史截断下信号、已成熟风险和账户逐日结果完全一致；10条账户的资金、份额、股息、费用和收益已按保存数据复算。实现正确不等于策略有效。",
             "", "方法出处：[StockCharts窄幅日说明](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/narrow-range-day-nr7)仅用于提出这条固定假说；[Sharpe原文](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)用于明确按完整期差额收益计算夏普。", "",
             "复查入口：`protocol.json`、`summary.json`、`results/完整账户比较.csv`、`results/新增机会对账户的增量.csv`、`results/逐年收益与完整周期.csv`、各方案`daily/trades/orders`文件。", ""]
    (OUT / "研究结论.md").write_text("\n".join(rows), encoding="utf-8")
    plot(accounts, metrics)


def plot(accounts, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    colors = {"POINT_A": "#175778", "POINT_B": "#879aad", "NR7_ONLY": "#c08d40", "POINT_A_PLUS_NR7": "#b74646"}
    for policy in engine.POLICIES:
        d = accounts[("2020_2026", "STRESS", policy)]["daily"]
        axes[0].plot(d.date, d.equity / 200000, label=engine.NAMES[policy], color=colors[policy], lw=1.7)
    axes[0].set_ylabel("20万元账户净值")
    axes[0].set_title("增加机会后，收益和夏普是否一起提高？\n510300 · 日线 · 压力成本 · 截至2026年9月30日", loc="left", fontsize=14)
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    sub = metrics.loc[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS")]
    x = np.arange(len(sub))
    axes[1].bar(x - .18, sub.net_sharpe, width=.36, color=[colors[k] for k in sub.policy], label="净夏普")
    for i, r in enumerate(sub.itertuples()):
        axes[1].text(i - .18, r.net_sharpe, f"{r.net_sharpe:.2f}", ha="center", va="bottom" if r.net_sharpe >= 0 else "top", fontsize=10)
    axes[1].set_xticks(x, [f"{engine.NAMES[r.policy]}\n年化{r.net_cagr:.2%}｜年均{r.average_full_year_cycles:.2f}笔" for r in sub.itertuples()], fontsize=9)
    axes[1].set_ylabel("净夏普（全部交易日）")
    axes[1].axhline(0, color="#888", lw=.7)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.06, .012, "历史开发期结果；现金/无风险比较率假设0；252日年化；未建立独立验证。风险减仓不增加完整周期数。", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(OUT / "交易次数与账户收益夏普.png", dpi=150)
    fig.savefig(OUT / "交易次数与账户收益夏普.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点位账户基线与唯一窄幅补充假说。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
