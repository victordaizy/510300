"""复核既有广度口径及保存账户，解释阶段差异并交付中文规则。"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.breadth_majority_trend_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, VARIANTS
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300多数成分上涨_第62轮_20260907"
DOCUMENT = OUT / "多数成分上涨_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_BREADTH_MAJORITY_LOCAL_GATE_20260907.md"


def verify_source(cfg):
    original = json.loads((ROOT / "reports/audit/510300_stress_transmission_hazard_v2_g1_historical_remediation_build_v1.json").read_text(encoding="utf-8"))
    paths = original["outputs"]
    for key in ["internal_features", "remediated_classified_returns", "four_state_daily_coverage"]:
        require(digest(ROOT / paths[key]["path"]) == paths[key]["sha256"], "原广度来源与既有构建回执不一致")
    saved = pd.read_parquet(ROOT / cfg["breadth"])
    members = pd.read_parquet(ROOT / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet")
    members = members.rename(columns={"membership_date": "date"})
    members["date"] = pd.to_datetime(members.date)
    dates = pd.DatetimeIndex(saved.date)
    symbols = pd.Index(sorted(members.symbol.unique()))
    source = pd.read_parquet(ROOT / paths["remediated_classified_returns"]["path"],
        columns=["date", "symbol", "daily_total_shareholder_return", "return_is_usable", "constituent_return_state"])
    valid = source.return_is_usable.eq(True) & source.constituent_return_state.isin(["TRADED_VALID", "OFFICIAL_SUSPENSION"])
    valid &= np.isfinite(source.daily_total_shareholder_return)
    source["usable_return"] = source.daily_total_shareholder_return.where(valid)
    require(source.usable_return.dropna().gt(-1).all(), "可用成员回报低于负百分之百")
    wide = source.pivot(index="date", columns="symbol", values="usable_return").reindex(index=dates, columns=symbols)
    rolling = np.expm1(np.log1p(wide).rolling(20, min_periods=20).sum())
    mask = members.assign(member=True).pivot(index="date", columns="symbol", values="member").reindex(index=dates, columns=symbols).eq(True)
    current = rolling.where(mask)
    count, member_count = current.notna().sum(axis=1), mask.sum(axis=1)
    ratio = count / member_count
    breadth = (current.gt(0).sum(axis=1) / count.replace(0, np.nan)).where(ratio.ge(.98))
    require(np.array_equal(count, saved.return20_scoreable_member_count), "二十日可计量成员数量复算不符")
    require(np.allclose(ratio, saved.return20_coverage_ratio, rtol=0, atol=1e-14), "二十日成员覆盖比例复算不符")
    require(np.allclose(breadth, saved.breadth20, rtol=0, atol=1e-14, equal_nan=True), "独立广度复算不符")
    output = pd.DataFrame({"date": dates, "members": member_count.to_numpy(), "scoreable_members": count.to_numpy(),
        "coverage": ratio.to_numpy(), "saved_breadth": saved.breadth20.to_numpy(), "recomputed_breadth": breadth.to_numpy()})
    output.to_csv(RESEARCH / "saved_component_breadth_recomputation.csv", index=False, encoding="utf-8-sig")
    return {"source_member_return_rows": len(source), "source_calendar_days": len(output), "finite_breadth_days": int(breadth.notna().sum()),
            "missing_breadth_days": int(breadth.isna().sum()), "maximum_breadth_difference": float(np.nanmax(abs(breadth.to_numpy() - saved.breadth20.to_numpy()))),
            "source_identity_checked_against_existing_build_receipt": True, "new_source_downloads": 0}


def verify_accounts(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "广度登记文件发生变化")
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    data = pd.read_parquet(ROOT / cfg["features"])
    metric_rows, differences, profit_rows, groups, cycle_rows, decision_rows, economics = [], [], [], [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {m["model"]: pd.read_parquet(folder / f"{m['model']}_ledger.parquet") for m in result[key] if m["cost"] == cost}
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = accounts[m["model"]]
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "保存净值与完整账户构成不符")
                prior = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / prior - 1, ledger.net_return, atol=1e-13, rtol=0), "保存每日净收益复算不符")
                reproduced = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "mean_exposure"]:
                    require(abs(reproduced[field] - m[field]) < 1e-10, "保存历史指标复算不符")
                metric_rows.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": reproduced["net_sharpe"]})
            new = accounts[PRIMARY]
            for control in ["PRICE20_ONLY", "REARM_RIDGE", "PANIC_LEARNED_HALF"]:
                old = accounts[control]
                diff = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(new.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_pnl_difference": float(new.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(new.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(new.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(new.slippage_cost.sum() - old.slippage_cost.sum())}
                diff["reconciliation_error"] = diff["terminal_nav_difference"] - diff["price_pnl_difference"] - diff["dividend_difference"] + diff["commission_difference"] + diff["slippage_difference"]
                require(abs(diff["reconciliation_error"]) < 1e-6, "账户差额未由价格分红和费用解释")
                differences.append(diff)
            for model, (use_breadth, _) in VARIANTS.items():
                ledger = accounts[model]
                cycles = pd.read_csv(folder / f"{model}_cycles.csv")
                decisions = pd.read_parquet(folder / f"{model}_decisions.parquet")
                ledger_by_day = ledger.set_index("date")
                rearmed, last_exit = True, -1000000
                for row in decisions.itertuples():
                    t = int(row.origin_index)
                    factor = factors.iloc[t]
                    price_mean = float(data.wealth.iloc[t - 19:t + 1].mean()) if t >= 19 else np.nan
                    require(abs(price_mean - factor.wealth_mean20) < 1e-12, "实际原点价格均线复算不符")
                    state = ledger_by_day.loc[row.origin] if row.origin in ledger_by_day.index else None
                    held = state is not None and state.shares > 0
                    if state is not None and state.filled_quantity > 0:
                        rearmed = False
                    if state is not None and state.filled_quantity < 0:
                        last_exit = t
                    known_false = bool(factor.price_weak or (use_breadth and factor.breadth_weak))
                    if not held and known_false:
                        rearmed = True
                    require(bool(row.entry_rearmed) == rearmed and bool(row.rearm_condition_known_false) == known_false, "缺失或持仓错误改变再次进入资格")
                    view = bool(factor.combined_entry_view if use_breadth else factor.price_entry_view)
                    require((row.entry_observation_state == "VIEW_ALLOWED") == view, "保存进入来源状态不符")
                    if not view and not held:
                        require(row.requested_quantity == 0 and pd.isna(row.reference_weight), "空仓无观点被转成新买入或零模型观点")
                    if row.requested_quantity > 0:
                        require(not held and view and rearmed and t - last_exit >= cfg["specification"]["cooldown"], "实际买入请求不满足来源或等待条件")
                        require(factor.price_above20 and (not use_breadth or factor.breadth_majority), "实际买入请求不满足完整原点条件")
                    if held:
                        require(bool(row.price_exit_requested) == bool(factor.price_weak), "持仓价格退出记录不符")
                        require(bool(row.breadth_exit_requested) == bool(use_breadth and factor.breadth_weak), "缺失广度被错误当成退出观点")
                decision_rows.append({"period": period, "cost": cost, "model": model, "decision_origins": len(decisions),
                    "entry_no_view_origins": int(decisions.entry_observation_state.eq("NO_VIEW").sum()),
                    "no_view_with_zero_requested_quantity_and_null_cash_reference_verified": True, "rearm_path_replayed": True})
                require(abs(cycles.net_profit_cny.sum() - (ledger.equity.iloc[-1] - cfg["initial_capital"])) < 1e-6, "周期利润不等于完整账户总利润")
                for cycle in cycles.itertuples():
                    buy = ledger_by_day.loc[pd.Timestamp(cycle.entry_date)]
                    sell = ledger_by_day.loc[pd.Timestamp(cycle.exit_date)]
                    profit = sell.notional - sell.commission + cycle.dividend_cny - buy.notional - buy.commission
                    require(abs(profit - cycle.net_profit_cny) < 1e-6, "单次完整周期净利润无法从真实成交和分红重建")
                    require(buy.filled_quantity == cycle.entry_quantity and sell.filled_quantity == -cycle.entry_quantity, "单次持仓份额出现非预定变化")
                    origin = factors.iloc[int(cycle.entry_index) - 1]
                    group = "广度缺失" if not origin.breadth_valid else ("多数上涨" if origin.breadth_majority else "未过半")
                    cycle_rows.append({"period": period, "cost": cost, "model": model, "cycle_id": cycle.cycle_id,
                        "entry_date": cycle.entry_date, "exit_date": cycle.exit_date, "entry_origin_breadth_group": group,
                        "entry_origin_breadth": origin.breadth20 if origin.breadth_valid else None,
                        "holding_intervals": cycle.holding_intervals, "net_profit_cny": cycle.net_profit_cny,
                        "cycle_profit_recomputation_error": profit - cycle.net_profit_cny})
                winning = float(cycles.loc[cycles.net_profit_cny.gt(0), "net_profit_cny"].sum())
                profit_rows.append({"period": period, "cost": cost, "model": model, "cycles": len(cycles), "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()),
                    "positive_profit_sum": winning, "nonpositive_profit_sum": float(cycles.loc[cycles.net_profit_cny.le(0), "net_profit_cny"].sum()),
                    "total_profit": float(cycles.net_profit_cny.sum()), "largest_cycle_profit": float(cycles.net_profit_cny.max()),
                    "largest_two_cycle_profits": float(cycles.net_profit_cny.nlargest(2).sum())})
                economics.append({"period": period, "cost": cost, "model": model, "price_pnl": float(ledger.price_pnl.sum()),
                    "dividends": float(ledger.dividend_recognized.sum()), "commission": float(ledger.commission.sum()),
                    "slippage": float(ledger.slippage_cost.sum()), "net_profit": float(ledger.equity.iloc[-1] - cfg["initial_capital"])})
    cycles_frame = pd.DataFrame(cycle_rows)
    for keys, group in cycles_frame.groupby(["period", "cost", "model", "entry_origin_breadth_group"], sort=False):
        groups.append(dict(zip(["period", "cost", "model", "entry_origin_breadth_group"], keys),
            cycles=len(group), positive_cycles=int(group.net_profit_cny.gt(0).sum()), total_cycle_profit=float(group.net_profit_cny.sum())))
    outputs = [("saved_metrics_recomputation.csv", metric_rows), ("saved_account_differences.csv", differences),
        ("saved_cycle_profit_summary.csv", profit_rows), ("saved_entry_breadth_groups.csv", groups),
        ("saved_cycle_entry_breadth.csv", cycle_rows), ("saved_decision_state_replay.csv", decision_rows), ("saved_account_economics.csv", economics)]
    for filename, values in outputs:
        pd.DataFrame(values).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    return {"recomputed_account_records": len(metric_rows), "account_differences": len(differences), "cycle_profit_sums": len(profit_rows),
            "recomputed_cycle_records": len(cycle_rows), "replayed_decision_origins": sum(r["decision_origins"] for r in decision_rows),
            "saved_entry_breadth_groups": len(groups), "new_diagnostic_accounts": 0, "new_models": 0}, profit_rows, groups, economics


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 62)), "索引不是截至61轮，不能重复覆盖")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    verification, profits, groups, economics = verify_accounts(cfg, result)
    source_verification = verify_source(cfg)
    receipt = {"verified_at": now(), "status": "SOURCE_BREADTH_AND_ACCOUNT_REPLAY_COMPLETE", **verification,
               **source_verification, "security_audit_performed": False, "goal_achieved": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_LOCAL_IMPROVEMENT_OVER_WEAK_PRICE_CONTROL_NO_STABLE_HIGH_SHARPE"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status,
        "decision": "广度组合在两段两费用优于同口径价格对照，但主评价仍亏损，较早改善受到来源缺失及少数上涨行情影响。结束这个独立二十日趋势组合，保留广度可作有限附加筛选的线索。",
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300多数成分上涨：第62轮结果", "", "## 老板先看结论", "",
        "**加入成分股上涨广度，比只看二十日价格均线有改善，但仍没有得到稳定高夏普。** 主评价净夏普负0.100、复合年化负1.36%；较早历史净夏普0.832、年化7.28%。主评价仍低于原学习策略；较早的改善又受到大量广度缺失和少数行情影响，不能选较早结果当成功。结束这个固定的独立趋势组合，不调整窗口或门槛补救。", "",
        "主方案只用两个市场条件：多数成分股近二十日上涨，同时510300高于二十日均线时进入；任一已知条件转弱时退出，另保留6%亏损、8%追踪退出和60日最长持有。另做删除广度的同规则对照，比较广度自身有没有帮助。没有补任何新数据，没有训练模型。", "",
        "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "PRICE20_ONLY", "REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "主评价2020年1月2日至2026年8月14日开盘，共1604个账户日；较早2015年1月5日至2019年12月31日开盘，共1219日。完整账户各20万元，所有空仓日、费用、分红及终点保留。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "## 为什么还不够", "",
        "主方案基础主评价62个完整周期，只有15个盈利，平均持有6.15个交易日，实际成交124笔；压力费用后只13个盈利。删除广度对照为82个周期、164笔成交。广度减少了一些反复进入，但频繁的小亏仍消耗了上涨利润。", "",
        "主方案基础主评价的价格损益为负1,423.80元，加分红2,469.60元，合计只增加1,045.80元；再减佣金4,518.96元、滑点13,883.80元，净亏17,356.96元。这是保存实际份额路径的损益分解，不能把加回费用的数额称作另一个零费用可交易账户。", "",
        "较早30个周期有15个盈利，平均9.20个交易日。2015年只有一次完整交易，3月5日至5月7日赚57,220.24元；2019年1月21日至3月26日赚40,824.50元，两次合计98,044.74元，高于整个较早账户84,908.00元净利润，其余交易合计为负。这说明较早高点估计集中于少数上涨行情。", "",
        "### 来源缺失和策略效果需要分清", "",
        "原缓存日期内，较早1219日有758日广度有值；但实际决策使用前一收盘，首个原点是2014年12月31日、最后原点是2019年12月30日，因此实际1219个决策原点只有757个有效，462个无观点。主评价1604个原点1592个有效，12个无观点。两种计数并不冲突。", "",
        "主方案实际持仓收盘中，主评价10日、较早33日广度缺失；这些日子没有因缺失自动清仓，价格及原保护退出照常独立检查。也没有在广度缺失日重新给予进入资格。所有缺失和空仓日都仍计入净夏普。", "",
        "以下只把原价格对照的已完成周期，按进入前一日的已知广度分组。各笔费用、份额和退出沿用原保存路径，属于事后归因，不能把分组利润当作重新执行筛选后的账户收益。", "",
        "|时期及费用|原价格对照进入时广度|周期数|盈利周期|原周期净利润合计|", "|---|---|---:|---:|---:|"]
    for row in groups:
        if row["model"] == "PRICE20_ONLY":
            label = "主评价" if row["period"] == "evaluation" else "较早历史"
            cost_label = "基础" if row["cost"] == "BASE" else "压力"
            lines.append(f"|{label}／{cost_label}|{row['entry_origin_breadth_group']}|{row['cycles']}|{row['positive_cycles']}|{row['total_cycle_profit']:,.2f}元|")
    lines += ["", "较早原价格对照有26个进入原点广度缺失，基础合计亏13,494.91元；其中也包括2015年2月16日至5月7日赚54,307.47元的交易，所以缺失既挡住亏损也可能错过盈利。不能把广度组合相对价格对照多赚的全部金额归因于多数上涨。主评价原价格对照的82次进入全部有广度值，说明主评价的局部改善并非由进入时缺失单独造成；但原周期分组仍不证明因果或稳定性。", "",
        "## 各年基础费用结果", "", "年度表包含所有年份；首末年度可能是部分期间，复合年化按原242日口径，另列实际期间收益。年度夏普不替代完整历史夏普。", "",
        "|时期|年份|主方案净夏普|实际期间收益|复合年化|成交笔数|", "|---|---:|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        label = "主评价" if row.period == "evaluation" else "较早历史"
        lines.append(f"|{label}|{row.year}|{row.net_sharpe:.3f}|{row.cumulative_return:.2%}|{row.annualized_return:.2%}|{row.trade_count}|")
    lines += ["", "## 全部中文因子与进出场规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8")
    lines += [protocol.split("\n", 1)[1], "", "## 核对与交付", "",
        "15项必要测试通过后冻结。首次最后一项测试尝试修改Pandas生成的只读数组，冻结前复制测试数组后完成非法输入注入；正式策略逻辑没有因该测试改变，初次失败记录保留。", "",
        f"已从原1,347,235条四态成员收益和点时成员重新核对2823日广度，2350个有值日与原缓存一致，473日缺失保持一致；最大差为{source_verification['maximum_breadth_difference']:.3g}。这只是核对已缓存因子，没有补数据、重建F/T或生成新策略。", "",
        f"20条完整账户指标、12条账户差额、8组周期利润、{verification['recomputed_cycle_records']}个周期成交利润及{verification['replayed_decision_origins']}个决策原点的进入许可、缺失状态和已知退出均核对完成。直接数据、全部账户、周期和逐日因子决定以普通文件保存。", "",
        "下一步只考虑把已有有效广度作为原学习策略进入时的附加筛选，保留原学习退出；广度缺失则沿用原学习策略本来的决定，明确记录来源无观点和基线回退。这可以避免把缺失当成择时收益。仍需另行冻结并运行，当前没有第63轮账户，也没有证据表明这个下一方案会达到1.2。", "",
        "目标尚未实现，不制作GPT数值包，不启动真实交易。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "entry_exit_coverage.csv", "signal_counts.csv",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_cycle_profit_summary.csv", "saved_entry_breadth_groups.csv",
        "saved_cycle_entry_breadth.csv", "saved_decision_state_replay.csv", "saved_account_economics.csv", "saved_component_breadth_recomputation.csv",
        "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json", "tests_initial_fixture_failure.json", "result.json"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    factors.to_csv(OUT / "全部逐日因子.csv", index=False, encoding="utf-8-sig")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for model in VARIANTS:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
                shutil.copy2(RESEARCH / period / cost / f"{model}_cycles.csv", OUT / f"{period}_{cost}_{model}_cycles.csv")
    NEXT_NOTE.write_text("""# 第62轮之后：仅将可用广度作为原学习进入的附加筛选

第62轮两个设置和20条两段两费用记录已经完成。主方案净夏普主评价负0.099805/负0.237576，较早0.831769/0.730108；删除广度的20日价格对照主负0.262818/负0.409612，较早0.025395/负0.083688。广度相对弱价格规则有两段两费用局部改善，但主评价仍亏损。主62周期基础15盈利、压力13盈利、124成交；较早30周期15盈利60成交。主价格损益加分红1045.80元，费用18402.76216元，净亏17356.96216元。较早2015和2019各一次最大交易合计98044.73824元，超过完整账户84907.9996元。结束这个独立20日价格加多数广度进出场组合，不改窗口、阈值、缺失处理、确认或保护参数。

广度独立字段已核实：原INTERNAL_GATED_COLUMNS不含breadth20。二十日总股东回报来自可用四态，当前点时300成员至少294完整才计算，缺失不填。所有2350个广度有值日的四态逐日源状态允许使用，组合F/T可能因其他窗口缺失而无观点。已从原1347235条四态回报和点时成员复算2823日期，2350有值、473缺失及可计量数量、覆盖比例都一致；没有新下载、补齐或F/T重算。官方成员收盘生效和2015扩展时钟沿用原修正。只是历史重建，不是实时来源证明。

实际决策原点主1592/1604有广度，较早757/1219有广度（462缺失）；与原缓存较早758/1219的差异来自前一日决策原点。持仓时主10收盘、较早33收盘广度缺失，独立价格保护继续检查，未以缺失强制卖出或恢复进入资格。原价格对照主82进入全有广度，较早56中26缺失，基础这26笔合计负13494.91182元，但含54307.47224元最大盈利。saved_entry_breadth_groups.csv保留全部两费用分组，只是旧份额旧退出路径事后诊断，不是可交易筛选，也不证明全额差额来自广度。

下一项最多一个主设置：在原R32 REARM_RIDGE准备进入时，若已有广度有效，就额外要求breadth20严格超过一半；有效但未过半则暂缓进入，保留未消耗的原进入资格。广度缺失时沿用原R32本来已有的进入决定，记录广度NO_VIEW和基线回退，不把缺失当看空，也不假造广度预测。进入后完全保留原R32学习退出、6%亏损、8%追踪及60收盘退出，不再用广度触发退出；没有新模型拟合。

这个下一项测试广度是否帮助一个已有较好基线，区别于第62轮独立二十日价格主方案。阈值及源覆盖均沿用，不选择年份，不用结果决定缺失处置。实际买入才消耗原D60资格；全部退出后必须原D60条件消失才能再给资格，单纯广度下降或缺失不能重新给资格。使用第62轮source_aware_cycle_account_v1明确rearm_allowed输入及旧ExitController，先测试与原R32无附加筛选时各字段经济路径一致，测试未知广度回退不是填值，然后冻结来源和旧模型、两段两费用一次执行。当前未建立第63轮配置、代码或账户，不宣称新增改进。

第62轮15必要测试先通过；首次只读数组测试夹具改为copy后注入，无正式逻辑变更。保存完整中文说明和普通结果，无GPT数值包、无安全审计。索引截至62：336设置、348已评价来源版本、353登记来源版本含5旧未运行、1044主评价记录。EPS和所有慢数据补齐暂停，执行只510300与现金，账户规则和夏普口径保持，目标未完成。
""", encoding="utf-8")
    record = {"round": 62, "study": result["study_id"], "title": "多数成分上涨与二十日价格趋势", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated",
        "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 2, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"),
        "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays",
        "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND62_COMPLETE_BREADTH_LOCAL_IMPROVEMENT_NOT_STABLE", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计62轮、336个不同设置或范围、348个已评价来源版本、1044个主评价记录；登记353含5个旧未运行绑定。较早、参考与模型另计。",
        checks=f"第62轮15必要测试、20指标、12账户差额、8周期利润组、{verification['recomputed_cycle_records']}周期成本和{verification['replayed_decision_origins']}决定复算；2823日独立广度源复核一致。",
        process_state_note="第62轮两段两费用账户、来源与缺失归因、普通中文交付完成；第63轮只有原学习进入附加广度的方向说明，没有登记或运行。",
        next_work={"status": "NEW_LOCAL_ENTRY_GATE_DIRECTION_NOT_REGISTERED", "focus": "只在广度有效时筛选原学习进入，缺失沿用原基线决定，原学习退出不变",
                   "source": str(NEXT_NOTE.relative_to(ROOT))}, latest_saved_breadth_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "BREADTH_MAJORITY_TREND_ROUND62_CHINESE_RESULTS", "rounds": [62],
        "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "BREADTH_CHINESE_RULES_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
