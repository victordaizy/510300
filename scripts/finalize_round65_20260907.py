"""保存固定波动选择的阶段差异和全部中文进出场规则。"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.volatility_expert_router_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300波动状态选择_第65轮_20260907"
DOCUMENT = OUT / "波动状态选择_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_VOLATILITY_ROUTER_ATR_BANDS_20260907.md"


def verify(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "固定波动研究来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    for w in [5, 60]:
        reconstructed = data.total_simple.rolling(w).std(ddof=1) * np.sqrt(cfg["annual_days"])
        require(np.allclose(reconstructed, data[f"vol{w}"], rtol=0, atol=1e-12, equal_nan=True), "短长波动不能由原收益复算")
    metrics, differences, replay, exits = [], [], [], []
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
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, rtol=0, atol=1e-6), "完整净值恒等式不成立")
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, rtol=0, atol=1e-13), "净值与净收益不符")
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost"]:
                    require(abs(actual[field] - m[field]) < 1e-10, "保存指标不能复算")
                metrics.append({"period": period, "cost": cost, "model": m["model"], "net_sharpe": actual["net_sharpe"], "days": len(ledger)})
            current = accounts[PRIMARY]
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF"]:
                old = accounts[control]
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(old.date)), "对照日历不一致")
                d = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                d["reconciliation_error"] = d["terminal_nav_difference"] - d["price_difference"] - d["dividend_difference"] + d["commission_difference"] + d["slippage_difference"]
                require(abs(d["reconciliation_error"]) < 1e-6, "保存账户差额没有解释完全")
                differences.append(d)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                s = source.iloc[t]
                f = data.iloc[t]
                expert = "PANIC" if f.vol5 > cfg["volatility_multiplier"] * f.vol60 else "LEARNED"
                expected = s.panic_state if expert == "PANIC" else s.learned_state
                require(row.origin == s.date and row.execution_date > row.origin, "来源与成交时钟不符")
                require(row.selected_expert == expert and row.reference_weight == expected, "选择或目标与冻结规则不符")
                old = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(old.shares) if old is not None else 0
                nav = float(old.equity) if old is not None else cfg["initial_capital"]
                target_shares = math.floor(expected * nav / f.close / cfg["lot"]) * cfg["lot"]
                if expected == 1 and shares > 0 and abs(expected - shares * f.close / nav) < cfg["weight_band"]:
                    target_shares = shares
                require(row.requested_quantity == target_shares - shares, "请求份额没有沿用实际净值与整手")
            indexed = decisions.set_index("origin")
            previous_expert = indexed.selected_expert.shift()
            for row in current[current.filled_quantity.lt(0)].itertuples():
                decision = indexed.loc[row.origin]
                previous = previous_expert.loc[row.origin]
                previous_state_now = decision.panic_state if previous == "PANIC" else decision.learned_state
                reason = "所选专家状态退出"
                if previous != decision.selected_expert and previous_state_now == 1:
                    reason = "市场状态切换且原专家仍持有"
                if row.mark_clock == "OPEN_TERMINAL":
                    reason = "终点退出"
                exits.append({"period": period, "cost": cost, "origin": row.origin, "execution_date": row.date,
                    "previous_expert": previous, "selected_expert": decision.selected_expert,
                    "previous_expert_current_state": previous_state_now, "selected_target": decision.reference_weight,
                    "filled_quantity": row.filled_quantity, "reason": reason})
            replay.append({"period": period, "cost": cost, "replayed_decision_origins": len(decisions),
                "held_closes": int(current.shares.gt(0).sum()), "actual_sells": int(current.filled_quantity.lt(0).sum())})
    for name, rows in [("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_state_and_request_replay.csv", replay), ("saved_exit_reason_rows.csv", exits)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "SAVED_VOLATILITY_STATES_ACCOUNTS_AND_REQUESTS_RECONCILED",
        "recomputed_account_records": len(metrics), "account_differences": len(differences), "volatility_columns_rebuilt": 2,
        "replayed_decision_origins": sum(x["replayed_decision_origins"] for x in replay), "exit_records_classified": len(exits),
        "new_diagnostic_accounts": 0, "new_models": 0, "security_audit_performed": False}, differences, exits


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 65)), "索引不是截至64轮，不能重复更新")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, differences, exits = verify(cfg, result)
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_REJECTED_VOLATILITY_ROUTER_MAIN_LOCAL_GAIN_EARLIER_WORSE"
    decision = "主评价夏普相对原学习略高，但较早明显变差；两段两费用均低于原各半，未达到1.2。结束固定波动专家选择，不反向、改阈值或改切换时机。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300按短期波动选择策略：第65轮结果", "", "## 先看结论", "",
        "**这个切换只在主评价略有改善，较早历史变差，结束本方案。** 主评价基础净夏普0.738，原学习为0.705；较早0.563，低于原学习0.748。两段两费用也都低于原各半，目标1.2没有实现。", "",
        "规则只多一个选择：最近5日波动大于60日波动的1.5倍时，使用原急跌反弹策略；否则使用原学习策略。每天收盘决定，下一开盘按所选策略的持有或退出状态执行。不是事后为每个年份挑收益最好的策略。", "",
        "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "PANIC_ONLY", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "主评价2020年1月2日至2026年8月14日开盘1604日；较早2015年1月5日至2019年12月31日开盘1219日。各20万元，全部空仓日、分红和实际费用保留。原急跌单独主评价较高仅有3个周期，较早8个周期；不能据此宣称稳定高夏普。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "## 切换怎样影响交易", "",
        "主评价1604个决策原点中，高短期波动166个，选择急跌；其余1438个选择学习。急跌被选且要求持有的原点只有12个，学习被选且要求持有223个。较早1219个原点中107个选择急跌、1112个选择学习，其中分别19和277个要求持有。以上是信号日期，不是独立交易次数。", "",
        "主评价实际27次买入、27次卖出，较早23次买入、23次卖出。原学习分别24个周期和9个周期。新策略主评价234个收盘持仓、较早296个；原输入在评价原点没有缺失，两档费用也没有未成交请求。", "",
        "主评价27次卖出中，5次因为市场状态切换、原来采用的专家当时仍要求持有而新专家要求空仓，21次是所选专家退出，另1次是统一终点退出。较早23次卖出中，10次属于前述切换退出、13次所选专家退出。两费用的分类一致。切换让原持仓被拆开并产生重新进入，尤其较早成交从原学习18笔增加到46笔。", "",
        "主评价的夏普小幅上升伴随风险下降，但基础复合年化从原学习5.38%降至4.98%；较早复合年化从8.64%降至6.85%，最大回撤从13.79%扩大到22.80%。这不是两个时期共同变好的策略。", "",
        "|历史及费用|相对原学习的终值差|价格损益差|分红差|佣金增加|滑点增加|", "|---|---:|---:|---:|---:|---:|"]
    for d in differences:
        if d["control"] != "REARM_RIDGE":
            continue
        label = ("主评价" if d["period"] == "evaluation" else "较早") + ("／基础" if d["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{d['terminal_nav_difference']:,.2f}元|{d['price_difference']:,.2f}元|{d['dividend_difference']:,.2f}元|{d['commission_difference']:,.2f}元|{d['slippage_difference']:,.2f}元|")
    lines += ["", "基础费用下，主评价终值比原学习少7,083.61元，较早少24,298.85元。价格损益减少和新增费用共同造成差额，不能把全部亏损都归因于切换手续费；也不能只保留主评价较高的夏普。", "",
        "## 全部因子及进入、持有、退出、再进入", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1],
        "", "补充原价格因子的确定口径：当最高价等于最低价时，旧收盘区间位置取0.5，低于急跌进入所需0.6，因此不满足该条件。没有因此生成一个新的进入信号。原模型全部逐月标准化数值和系数见同目录《原学习模型逐月中文规则.md》。", "",
        "## 各年基础费用表现", "", "年度拆分用于展示差异，首末年可能为部分期间，不重新筛选交易年份。", "",
        "|时期|年份|净夏普|实际期间收益|复合年化|成交笔数|", "|---|---:|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for r in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        lines.append(f"|{'主评价' if r.period == 'evaluation' else '较早'}|{r.year}|{r.net_sharpe:.3f}|{r.cumulative_return:.2%}|{r.annualized_return:.2%}|{r.trade_count}|")
    lines += ["", "## 运行、核对与下一步", "",
        "10项必要测试通过后冻结一个设置，再一次计算4个新账户，复用16个对照；没有重训模型或补行情。测试覆盖边界、缺失、未来数据不影响过去、相同目标免去多余买卖、改变目标的下一开盘退出与再进入。账户运行无中断。", "",
        f"保存结果核对已完成：20个指标、8条经济差额、{receipt['replayed_decision_origins']}个决策原点及请求份额、{receipt['exit_records_classified']}条实际卖出原因；5日和60日波动均从保存的含分红收益复算。此前组装运行文件时一次文本替换检查在写文件前中止，未产生研究配置、账户或收益，修正文件生成过程后才测试和冻结；它不增加策略设置或来源评价次数。", "",
        "下一步停止围绕这两个专家调整配比与切换条件，改检查价格波动通道本身能否给出完整进出场。拟只用一个平均真实波幅趋势通道设置：穿越上轨进入，跌破随行情移动的下轨退出；不需要EPS、预测模型或新增资料。指标计算机制参考[TradingView官方Supertrend说明](https://www.tradingview.com/support/solutions/43000634738-supertrend/)。下一轮尚未登记或产生账户。", "",
        "本轮查重发现14日方向运动与25趋势门槛在旧三状态路由已经研究，未再次计算该路线；在研究、配置和文档的有限关键词检索中没有找到超级趋势通道实现，这只是查重范围内的结果，不表示所有可能的历史别名都不存在。", "",
        "EPS和慢来源补齐保持暂停。净夏普1.2和稳定超额仍未实现，原学习和原各半只作为比较候选保留；持续研究保持进行，无GPT数值包，无真实交易。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_state_and_request_replay.csv", "saved_exit_reason_rows.csv",
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
    NEXT_NOTE.write_text("""# 第65轮后：平均真实波幅趋势通道方向

第65轮固定5日波动大于60日1.5倍选急跌V6，否则选R32；主BASE/STRESS夏普0.737793/0.656625，较早0.563230/0.512354。主相对R32只局部改善，早明显变差；两段两费用均低于R46各半。主基础年化4.9822%、回撤9.4727%；早6.8521%、回撤22.8029%。主27周期54成交、早23周期46成交；主5和早10次在原专家仍为1时因市场切换转为0退出，两费用相同。不要靠反向选择、改1.5倍、改窗口、进入锁定或重扫权重救活。

主基础相对R32终值少7083.6076元，价格少2631.1、分红少3326.4、佣金增加295.0076、滑点增加831.1；较早终值少24298.85398，价格少18934.8、分红同、佣金增1326.95398、滑点增4037.1。完整逐笔原因在saved_exit_reason_rows.csv。新报告目录reports/research/510300_volatility_expert_router_v1，10测试3.48秒通过、20指标8差额5646原点100退出原因、2波动列复算。正式运行无中断；之前生成文件时替换检查写文件前失败，无账户或配置，不计新来源版本。索引累计65轮339不同设置、352已评价来源版本、357登记含5旧未运行、1072主记录。不要再跑冻结run或finalizer。

下一项尚未登记或计算，最多一个平均真实波幅跟踪通道设置（Supertrend），候选窗口10交易日、带宽3倍是预定研究值，不声称官方唯一默认或最优。只用已有510300价格及分红，新的价格机制，不接着调两个旧专家。

官方机制来源：https://www.tradingview.com/support/solutions/43000634738-supertrend/ 。中间线取最高最低均值，上下基础轨为中间线加减倍数乘平均真实波幅。延续上轨时只在新上轨更低或前收盘已突破旧上轨时更新；下轨相反。在原下行状态，当前收盘严格高于更新上轨才转上行；在上行状态，严格低于更新下轨才转下行；相等保持。上行目标1，下行目标0，收盘后下一开盘执行，不称盘中已成交。初始化与缺失规则必须在冻结前写清楚；未成熟保持NO_VIEW，不能从未来方向回填早期。

先核旧既有日线策略名是否有同义通道。已有限检索research/config/docs中supertrend、super_trend、超级趋势、keltner、chandelier、parabolic、sar、kalman、dual thrust、aroon等无命中；但ADX/方向运动14日25阈值有旧510300_THREE_STATE_TREND_ROUTER_V1及修正版本，因此放弃新ADX候选，不重开。无需展开全部旧研究。

波幅应使用因果分红财富尺度的高低收盘，明确最高最低如何转换、真实波幅包含与前收盘的缺口、首次均值和后续Wilder递推；实际账户继续原现金分红登记、应收、到账，不能用信号财富代替账户收益。当前财富收盘乘当日最高加分红与收盘加分红的比值可得到同尺度最高，最低同理；先验证与因果前日财富推导一致，不前复权回填。最后统一开盘退出。无额外止盈止损门槛扫描、无杠杆、无新模型/参考/数据。保留20万元、两档费用、两历史窗口、242年化、全现金日、整手T+1限制。若失败结束，不换ATR窗口与倍率救活。EPS慢源暂停，无GPT数值包；只510300与现金，目标未完成。
""", encoding="utf-8")
    record = {"round": 65, "study": result["study_id"], "title": "按短期波动选择急跌或原学习策略", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND65_COMPLETE_VOLATILITY_ROUTER_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计65轮，339不同设置，352已评价来源版本，357登记含5旧未运行，1072主评价记录；对照复用不新增设置。",
        checks="第65轮10测试，20指标8差额5646原点及请求100卖出原因、2波动列复算。",
        process_state_note="第65轮账户、归因、中文交付完成。第66轮为平均真实波幅趋势通道方向，未登记未运行。",
        next_work={"status": "ATR_TREND_BANDS_DIRECTION_NOT_REGISTERED", "focus": "已有价格的平均真实波幅趋势通道，完整独立进出场", "source": str(NEXT_NOTE.relative_to(ROOT))},
        latest_saved_volatility_router_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "VOLATILITY_ROUTER_ROUND65_CHINESE_RESULTS", "rounds": [65], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "VOLATILITY_ROUTER_RULES_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
