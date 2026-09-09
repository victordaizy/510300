"""交付完整预算并集的中文规则，解释仓位、费用和夏普变化。"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.panic_learned_signal_union_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P46
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300信号完整预算合并_第64轮_20260907"
DOCUMENT = OUT / "完整预算合并_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_UNION_VOLATILITY_ROUTER_20260907.md"


def verify(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "并集冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    metrics, differences, states, trades = [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        source = pd.read_parquet(RESEARCH / f"{period}_states.parquet")
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {}
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                accounts[m["model"]] = ledger
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "保存净值与现金份额不符")
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0), "保存净收益不能由全日历净值还原")
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "mean_exposure"]:
                    require(abs(actual[field] - m[field]) < 1e-10, "保存评价指标不能复算")
                metrics.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": actual["net_sharpe"]})
            current = accounts[PRIMARY]
            for control in ["PANIC_LEARNED_HALF", "REARM_RIDGE"]:
                old = accounts[control]
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(old.date)), "对照日历不同")
                d = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                d["reconciliation_error"] = d["terminal_nav_difference"] - d["price_difference"] - d["dividend_difference"] + d["commission_difference"] + d["slippage_difference"]
                require(abs(d["reconciliation_error"]) < 1e-6, "账户差额没有解释完全")
                differences.append(d)
            old = accounts["PANIC_LEARNED_HALF"]
            require(current.shares.gt(0).equals(old.shares.gt(0)), "原各半与并集持仓日期不一致")
            changed = current.filled_quantity.ne(0) != old.filled_quantity.ne(0)
            for i in np.flatnonzero(changed):
                trades.append({"period": period, "cost": cost, "date": current.date.iloc[i],
                    "old_filled_quantity": int(old.filled_quantity.iloc[i]), "new_filled_quantity": int(current.filled_quantity.iloc[i])})
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                s = source.iloc[t]
                require(row.origin == s.date and row.execution_date > row.origin, "来源原点或下一开盘时钟错误")
                expected = max(s.panic_state, s.learned_state)
                require(expected == row.reference_weight and .5 * (s.panic_state + s.learned_state) == row.original_half_target, "保存组合目标与专家输入不符")
                held = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(held.shares) if held is not None else 0
                nav = float(held.equity) if held is not None else cfg["initial_capital"]
                close = float(data.close.iloc[t])
                target_shares = math.floor(expected * nav / close / cfg["lot"]) * cfg["lot"]
                if expected == 1 and shares > 0 and abs(expected - shares * close / nav) < cfg["weight_band"]:
                    target_shares = shares
                require(row.requested_quantity == target_shares - shares, "保存请求没有使用实际账户和整手规则")
            states.append({"period": period, "cost": cost, "replayed_decision_origins": len(decisions),
                "holding_dates_identical_to_half": True, "held_closes": int(current.shares.gt(0).sum()),
                "old_trades": int(old.filled_quantity.ne(0).sum()), "new_trades": int(current.filled_quantity.ne(0).sum()),
                "changed_trade_dates": int(changed.sum())})
    for name, rows in [("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_state_and_request_replay.csv", states), ("saved_changed_trade_dates.csv", trades)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "SAVED_ACCOUNTS_STATES_AND_ACTUAL_REQUESTS_RECONCILED",
        "recomputed_account_records": len(metrics), "account_differences": len(differences),
        "replayed_decision_origins": sum(x["replayed_decision_origins"] for x in states),
        "changed_trade_date_records": len(trades), "holding_dates_identical_to_half_all_four_accounts": True,
        "new_diagnostic_accounts": 0, "new_models": 0, "security_audit_performed": False}, differences


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 64)), "索引不是截至63轮，不能重复更新")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, differences = verify(cfg, result)
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_REJECTED_FULL_BUDGET_UNION_ALL_FOUR_SHARPES_LOWER_THAN_HALF"
    decision = "主评价和较早历史、基础和压力费用的夏普都低于原各半。收益增加但风险、成交金额和费用同步增加。结束完整预算并集，不扫描权重或改阈值挽救。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300信号合并使用完整预算：第64轮结果", "", "## 先看结论", "",
        "**这个方案增加了收益，但没有提高夏普，结束本方案。** 主评价基础净夏普从原各半的0.944降至0.825，较早从0.604降至0.591；两档费用、两段历史都低于原各半，未达到1.2。", "",
        "方案很简单：原急跌反弹或原学习策略中，只要任一要求持有，就使用账户可负担的完整预算；两个都退出才清仓。所有旧信号、模型、数据和费用标准保持原样。", "",
        "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "PANIC_LEARNED_HALF", "REARM_RIDGE", "PANIC_ONLY", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "主评价为2020年1月2日至2026年8月14日开盘，共1604日；较早为2015年1月5日至2019年12月31日开盘，共1219日。各20万元，现金日也计入，终点统一开盘退出。原急跌单独的主评价高点仅来自3个周期，较早8个周期，不能把其中一个时期的高夏普当作稳定证据。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "## 为什么仓位加大后夏普反而下降", "",
        "四个账户与原各半的持仓日期完全相同，改变主要来自每次投入的金额。主评价基础平均股票仓位从8.74%增至16.30%，年化波动从原约4.26%升至8.03%；复合年化由4.01%升至6.50%，回撤由5.39%扩大到10.46%。较早平均仓位由14.36%升至27.72%，复合年化由4.19%升至7.57%，回撤由11.07%扩大到21.17%。收益增长没有抵消风险增长。", "",
        "主评价只有18个原点两条信号同时持有，学习单独持有235个、急跌单独持有10个、两者空仓1341个；较早没有重叠持有，学习单独299个、急跌单独40个、两者空仓880个。因此大部分有效信号日都提高了原来半仓的目标，而非获得新的择时信息。这些是决策日期数，不是独立交易数。", "",
        "主评价成交从52笔减少到50笔，只免去了2020年2月18日的中途减仓和2025年4月10日的中途加仓；较早仍为32笔。虽然笔数稍少，主评价基础总佣金及滑点由5,120.05元增至10,342.21元，较早由2,928.05元增至6,326.26元，因为成交份额更大。不能用少两笔交易推断费用必然减少。", "",
        "|历史及费用|相对原各半的终值增加|价格损益增加|分红增加|佣金增加|滑点增加|", "|---|---:|---:|---:|---:|---:|"]
    for d in differences:
        if d["control"] != "PANIC_LEARNED_HALF":
            continue
        label = ("主评价" if d["period"] == "evaluation" else "较早") + ("／基础" if d["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{d['terminal_nav_difference']:,.2f}元|{d['price_difference']:,.2f}元|{d['dividend_difference']:,.2f}元|{d['commission_difference']:,.2f}元|{d['slippage_difference']:,.2f}元|")
    lines += ["", "终值差等于价格损益差加分红差，再扣佣金和滑点差。资金复利、整手和部分重叠也会改变结果，本账户不是旧收益直接乘二。", "",
        "## 全部因子及完整进出场规则", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1],
        "", "八个学习因子的每月均值、尺度、系数及27次无模型更新，全部以中文保存在同目录《原学习模型逐月中文规则.md》。本轮复用原114次成熟拟合，无新增拟合。", "",
        "## 各年基础费用表现", "", "首末年可能为部分年份；年度拆分用来呈现差异，不用于重新选交易年份。", "",
        "|时期|年份|净夏普|实际期间收益|复合年化|成交笔数|", "|---|---:|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for r in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        lines.append(f"|{'主评价' if r.period == 'evaluation' else '较早'}|{r.year}|{r.net_sharpe:.3f}|{r.cumulative_return:.2%}|{r.annualized_return:.2%}|{r.trade_count}|")
    lines += ["", "## 实际运行和下一步", "",
        f"13项必要测试在冻结前通过，含4个原各半完整账户核对；正式运行4个新账户，复用16个对照。本轮只有1个策略设置、1个来源版本，没有运行中断。已复算20个账户指标、8条净值差额和{receipt['replayed_decision_origins']}个原点的状态及实际份额请求，没有新增归因账户。", "",
        "下一步用同一批现有资料测试一个固定的波动状态选择：5日波动严格大于60日波动的1.5倍时选急跌策略，其余已知状态选原学习策略，每日收盘确定、下一开盘执行。比例沿用急跌信号现有定义。第9、39、40、53轮已有其他状态切换研究，这不是声称策略切换从未做过；新方向只检查这两个已有专家与这一个波动定义，不扫描阈值、权重或多套窗口。此份交付时第65轮尚未登记或计算。", "",
        "EPS和慢来源补齐继续暂停。目标1.2未完成，原各半与原学习都只是历史比较候选，没有独立验证为稳定超额。持续研究保持进行，无GPT数值包，无真实交易。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_state_and_request_replay.csv", "saved_changed_trade_dates.csv",
        "tests_receipt.json", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(ROOT / cfg["chinese_saved_models"], OUT / "原学习模型逐月中文规则.md")
    for period in ["evaluation", "earlier_diagnostic"]:
        pd.read_parquet(RESEARCH / f"{period}_states.parquet").to_csv(OUT / f"{period}_全部逐日状态.csv", index=False, encoding="utf-8-sig")
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_trades.csv", OUT / f"{period}_{cost}_trades.csv")
    NEXT_NOTE.write_text("# 第64轮后：固定波动状态选择已有专家\n\n" + decision + "\n\n"
        "第64轮主基础/压力夏普0.824738/0.759944，较早0.591316/0.557988；原各半0.944391/0.879003及0.604421/0.572113。四个账户持仓日期与各半完全相同，主只减少2020-02-18、2025-04-10两笔成交，早期没有减少。提高仓位使金额和费用上升。全部结果、5646个原点请求与20指标在reports/research/510300_panic_learned_signal_union_v1。不能再做并集仓位扫描。\n\n"
        "下一项最多一个设置：复用R46原两专家BASE状态，短期5日收益样本波动严格大于60日波动1.5倍时选择V6急跌状态，否则选择R32原学习状态。仅已知有效正长期波动及已选专家0/1状态才能出目标，缺失NO_VIEW，保留份额。每天收盘重新选择，下一开盘完整账户，不因换专家而机械先卖再买；目标相同不变，目标0全退出，未成交按次日最新目标重算，终点统一开盘退出。全预算上限100%、10个百分点份额带宽、原两费用和窗口、20万元、242年化、整手分红T+1不变。不额外训练或新参考账户。\n\n"
        "查重已核R9四/两状态专家学习、R39趋势SMA120/效率20选S1或R32、R40同选择但实际进入后锁定、R53方差比5/60选突破或反弹；这些失败保留。本轮是固定波动状态选R32或V6，不宣称在线选择未曾做过；未登记也未计算。先写中文全部规则和必要测试，然后冻结一次，失败不改窗口、1.5阈值、持有时切换或方向挽救。\n\n"
        "索引截至64轮：338不同设置、351已评价来源版本、356登记含5旧未运行、1062主评价记录。EPS/公募/报表慢源暂停；无GPT数值包，仅510300与现金。目标未完成。\n", encoding="utf-8")
    record = {"round": 64, "study": result["study_id"], "title": "急跌与学习信号合并使用完整预算", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND64_COMPLETE_FULL_BUDGET_UNION_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计64轮，338不同设置，351已评价来源版本，356登记含5旧未运行，1062主评价记录；对照复用不新增候选设置。",
        checks="第64轮13测试含4个旧各半账户核对，20指标、8差额、5646原点目标与实际请求复算；四账户持仓日期与原各半相同。",
        process_state_note="第64轮已完成账户、归因与中文交付；第65轮只有固定波动状态选择原急跌或学习的方向，未登记或运行。",
        next_work={"status": "VOLATILITY_EXPERT_ROUTER_DIRECTION_NOT_REGISTERED", "focus": "短期波动明显放大时选择急跌策略，其余已知状态选择原学习策略", "source": str(NEXT_NOTE.relative_to(ROOT))},
        latest_saved_union_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "PANIC_LEARNED_UNION_ROUND64_CHINESE_RESULTS", "rounds": [64], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "UNION_RULES_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
