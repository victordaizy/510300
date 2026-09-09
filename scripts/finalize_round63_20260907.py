"""保存广度附加进入的完整失败归因，保留输出修正前的原账户。"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.breadth_learned_entry_gate_v1_output_fix import ROOT, OUT as RESEARCH, PARENT_OUT, CONFIG, CORRECTION, PRIMARY, P32
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300广度筛选学习进入_第63轮_20260907"
DOCUMENT = OUT / "广度筛选学习进入_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_BREADTH_GATE_SIGNAL_UNION_20260907.md"


def verify(cfg, result):
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"] + correction["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "原设置、输出修正或首个账户身份改变")
    for suffix in ["ledger.parquet", "decisions.parquet", "trades.csv", "cycles.csv"]:
        require(digest(PARENT_OUT / "evaluation/BASE" / f"{PRIMARY}_{suffix}") == digest(RESEARCH / "evaluation/BASE" / f"{PRIMARY}_{suffix}"), "首次计算的基础账户没有原样保留")
    data = pd.read_parquet(ROOT / cfg["features"])
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    stored = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"][cfg["saved_model_key"]]
    fits = np.array([x["fit_index"] for x in stored])
    metrics, differences, matched, profits, model_checks, states = [], [], [], [], [], []
    cycle_count = 0
    for period, result_key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {m["model"]: pd.read_parquet(folder / f"{m['model']}_ledger.parquet") for m in result[result_key] if m["cost"] == cost}
            for m in result[result_key]:
                if m["cost"] != cost:
                    continue
                ledger = accounts[m["model"]]
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "现金份额及分红应收不等于净值")
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0), "保存净收益与完整净值不符")
                reproduced = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "mean_exposure"]:
                    require(abs(reproduced[field] - m[field]) < 1e-10, "保存指标无法复算")
                metrics.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": reproduced["net_sharpe"]})
            current = accounts[PRIMARY]
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF"]:
                old = accounts[control]
                diff = {"period": period, "cost": cost, "control": control, "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()), "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()), "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                diff["reconciliation_error"] = diff["terminal_nav_difference"] - diff["price_difference"] - diff["dividend_difference"] + diff["commission_difference"] + diff["slippage_difference"]
                require(abs(diff["reconciliation_error"]) < 1e-6, "新旧完整账户差额未解释")
                differences.append(diff)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            old_cycles = pd.read_csv(P32 / period / cost / "REARM_RIDGE_cycles.csv")
            joined = old_cycles.merge(cycles, on="entry_origin", how="outer", suffixes=("_old", "_new"), indicator=True)
            joined.insert(0, "cost", cost)
            joined.insert(0, "period", period)
            matched.append(joined)
            ledger_by_date = current.set_index("date")
            rearmed, last_exit, last_cycle, negative = True, -1000000, None, 0
            prediction_count, max_error, fallback_requests = 0, 0., 0
            for row in decisions.itertuples():
                t = int(row.origin_index)
                f = factors.iloc[t]
                state = ledger_by_date.loc[row.origin] if row.origin in ledger_by_date.index else None
                held = state is not None and state.shares > 0
                if state is not None and state.filled_quantity > 0:
                    rearmed = False
                if state is not None and state.filled_quantity < 0:
                    last_exit = t
                if not held and f.base_raw_entry == 0:
                    rearmed = True
                require(bool(row.entry_rearmed) == rearmed and bool(row.rearm_condition_known_false) == bool(f.base_raw_entry == 0), "附加广度错误恢复原进入资格")
                expected = int(f.base_raw_entry) if not f.breadth_valid or f.breadth_majority else 0
                require(int(row.effective_raw_entry) == expected, "有效筛选或无观点回退不符")
                if row.requested_quantity > 0:
                    require(not held and rearmed and expected == 1 and t - last_exit >= cfg["specification"]["cooldown"], "实际请求不满足原条件、广度规则或等待")
                    fallback_requests += int(not f.breadth_valid)
                if held:
                    if row.learning_cycle_id != last_cycle:
                        negative, last_cycle = 0, row.learning_cycle_id
                    position = int(np.searchsorted(fits, t, side="right") - 1)
                    model = stored[position] if position >= 0 else None
                    values = np.array([getattr(row, name) for name in FEATURES], float)
                    estimate = None
                    if model is not None and model["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
                        require(model["fit_index"] <= t and model["latest_exit_index"] <= model["fit_index"], "持仓预测使用了尚未成熟周期")
                        coefficients = model["model"]
                        x = np.clip((values - np.array(coefficients["mean"])) / np.array(coefficients["scale"]), -coefficients["feature_clip"], coefficients["feature_clip"])
                        estimate = float(coefficients["intercept"] + np.dot(x, coefficients["coefficients"]))
                        error = abs(estimate - row.continuation_prediction)
                        require(error < 1e-12, "保存预测与原模型系数不符")
                        max_error = max(max_error, error)
                        prediction_count += 1
                    else:
                        require(pd.isna(row.continuation_prediction), "无可用模型时产生预测")
                    negative = negative + 1 if estimate is not None and estimate < 0 else 0
                    require(row.negative_confirmation_count == negative and bool(row.learned_exit_requested) == bool(negative >= 2), "原学习退出计数发生变化")
                    require("广度" not in str(row.exit_reasons), "广度被用于本轮持仓退出")
            model_checks.append({"period": period, "cost": cost, "replayed_predictions": prediction_count, "maximum_prediction_error": max_error})
            states.append({"period": period, "cost": cost, "decision_origins": len(decisions), "missing_breadth_baseline_buy_requests": fallback_requests,
                           "original_rearm_and_exit_state_replayed": True})
            require(abs(cycles.net_profit_cny.sum() - (current.equity.iloc[-1] - cfg["initial_capital"])) < 1e-6, "周期利润与账户总利润不符")
            for cycle in cycles.itertuples():
                buy, sell = ledger_by_date.loc[pd.Timestamp(cycle.entry_date)], ledger_by_date.loc[pd.Timestamp(cycle.exit_date)]
                actual = sell.notional - sell.commission + cycle.dividend_cny - buy.notional - buy.commission
                require(abs(actual - cycle.net_profit_cny) < 1e-6, "实际成交分红无法重建周期利润")
                require(buy.filled_quantity == cycle.entry_quantity and sell.filled_quantity == -cycle.entry_quantity, "周期出现意外追加或部分退出")
                origin = factors.iloc[int(cycle.entry_index) - 1]
                require(origin.effective_raw_entry == 1, "实际买入缺少合格原点")
                require(bool(cycle.entry_used_missing_breadth_baseline_fallback) == bool(not origin.breadth_valid), "实际周期回退标记不符")
                cycle_count += 1
            for fallback, group in cycles.groupby("entry_used_missing_breadth_baseline_fallback"):
                profits.append({"period": period, "cost": cost, "entry_used_missing_breadth_baseline_fallback": bool(fallback),
                    "cycles": len(group), "positive_cycles": int(group.net_profit_cny.gt(0).sum()), "net_profit": float(group.net_profit_cny.sum())})
    for name, rows in [("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_entry_fallback_profit_groups.csv", profits), ("saved_prediction_replay.csv", model_checks), ("saved_decision_state_replay.csv", states)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    pd.concat(matched, ignore_index=True).to_csv(RESEARCH / "saved_entry_origin_cycle_comparison.csv", index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "PRESERVED_ACCOUNT_AND_FROZEN_GATE_REPLAY_COMPLETE", "recomputed_account_records": len(metrics),
        "account_differences": len(differences), "recomputed_cycle_records": cycle_count, "replayed_decision_origins": sum(x["decision_origins"] for x in states),
        "replayed_predictions": sum(x["replayed_predictions"] for x in model_checks), "maximum_prediction_error": max(x["maximum_prediction_error"] for x in model_checks),
        "first_saved_account_files_preserved_identically": 4, "entry_origin_comparison_rows": sum(len(x) for x in matched),
        "new_diagnostic_accounts": 0, "new_models": 0, "security_audit_performed": False, "goal_achieved": False}, profits


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 63)), "索引不是截至62轮，不能重复覆盖")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, profits = verify(cfg, result)
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_REJECTED_BREADTH_ENTRY_GATE_BOTH_PERIODS_AND_COSTS_WORSE"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status,
        "decision": "两段两档费用均低于原学习策略，附加广度使部分进入延后并丢失原盈利机会。结束该筛选，不反向使用、改阈值或改回退方式。",
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300广度筛选原学习进入：第63轮结果", "", "## 老板先看结论", "",
        "**这道附加广度筛选没有改善原学习策略，两段历史、两档费用都变差，结束该方法。** 主评价基础净夏普从原0.705降至0.316，较早从0.748降至0.723，仍未达到1.2。没有补数据，没有训练新模型。", "",
        "只在原策略准备进入时检查广度：有效且多数成分股近20日上涨才进入；有效但未过半则等待；广度缺失时沿用原策略判断。实际持仓后继续使用原学习退出和保护规则，广度不触发退出。", "",
        "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "主评价2020年1月2日至2026年8月14日开盘1604日，较早2015年1月5日至2019年12月31日开盘1219日，各20万元，全部空仓、费用、分红及终点保留。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "## 失败原因与实际交易", "",
        "主评价实际19个完整周期、9个盈利、38笔成交，原策略24周期48笔成交；较早7周期4盈利14笔成交，原策略9周期18笔成交。减少交易并没有提高完整账户净夏普。", "",
        "主评价基础账户相对原策略少55,250.74元：价格损益少56,888.80元，分红少1,453.60元，节省佣金779.66元和滑点2,312.00元，费用节省不足以弥补失去的收益。较早少8,334.44元：价格损益少9,226.40元，分红相同，节省佣金228.66元、滑点663.30元。", "",
        "附加条件可能让进入过晚。例如原策略2020年2月5日买入、3月2日退出，周期净赚8,366.35元；新账户等到3月3日才进入，3月13日退出，周期亏14,093.62元。原策略2021年1月5日进入、1月21日退出赚9,688.50元，新账户1月8日才进入、同日退出亏2,228.34元。这些是各自实际账户的成交与金额，并不是固定本金下只改变某一天的隔离实验。", "",
        "也有避免亏损的例子：原2024年8月28日至9月12日亏8,177.70元的进入，在新账户没有同一原点交易。所有有利和不利的对应均保存，不能只选这些避免亏损的案例。主评价原24与新19周期只有13个相同进入原点，旧独有11、新独有6；较早原9与新7只有5个相同原点，旧独有4、新独有2。", "",
        "原主评价546个合格进入原点中，有260个被有效但未过半的广度暂缓，剩286个；較早574中144个暂缓，剩430个。这些是原信号日期，不是独立交易次数。实际空仓且原资格有效时，被广度暂缓的原点主105个、较早10个，统计未额外要求等待期结束，因此不能全部称为当日可成交机会。", "",
        "### 缺失回退的贡献", "",
        "主评价12个原点广度缺失，其中11个原进入条件合格，但最终没有实际周期通过缺失回退进入。较早462个原点缺失，其中281个原条件合格；实际7个周期中6个通过缺失回退进入，只有1个由有效多数广度放行。较早表现大部分沿用了原策略，不能归功于广度。", "",
        "|时期及费用|实际进入方式|周期数|盈利周期|净利润合计|", "|---|---|---:|---:|---:|"]
    for row in profits:
        label = "主评价" if row["period"] == "evaluation" else "较早历史"
        cost_label = "基础" if row["cost"] == "BASE" else "压力"
        method = "广度缺失，原基线回退" if row["entry_used_missing_breadth_baseline_fallback"] else "有效多数广度放行"
        lines.append(f"|{label}／{cost_label}|{method}|{row['cycles']}|{row['positive_cycles']}|{row['net_profit']:,.2f}元|")
    lines += ["", "主评价160个持仓收盘都有原学习预测，其中18个周期通过学习条件退出；较早271个持仓收盘只有67个有预测、204个无成熟模型，只有1个周期通过学习退出。无模型没有被填成预测或自动清仓，原保护规则继续执行。", "",
        "## 全部中文因子、进入与退出规则", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1],
        "", "原学习训练的固定细节：每月首个收盘只使用已自然结束的最近20个参考周期，至少10个周期、100个状态行才拟合；每周期总权重相同，岭惩罚为1。基线原参考有34个自然结束周期；本轮复用全部原模型，不增加样本。完整模型逐月均值、尺度、系数和无模型月份见同目录《原学习模型逐月中文规则.md》。", "",
        "## 各年基础费用结果", "", "年度指标只用于展示阶段差异；首末年可为部分期间，不替代完整历史结果。", "",
        "|时期|年份|新策略夏普|实际期间收益|复合年化|成交笔数|", "|---|---:|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        label = "主评价" if row.period == "evaluation" else "较早历史"
        lines.append(f"|{label}|{row.year}|{row.net_sharpe:.3f}|{row.cumulative_return:.2%}|{row.annualized_return:.2%}|{row.trade_count}|")
    lines += ["", "## 实际运行与核对", "",
        "10项必要测试通过后冻结，包含4次原完整账户验证重放。首次比较时，未保存决策中的缺失与Parquet读回的空值表示不同；冻结前统一文件格式后，原账户和全部原决策字段相同。没有填零或改变规则。", "",
        "正式运行首个主评价基础账户完成后，汇总误用了不存在的预测字段名而中断。另行登记输出修正，改为读取实际的预测字段，原进出场、参数、模型和原冻结文件均保留；首个账户的净值、决定、交易及周期四份文件原样复制，未重新计算。随后只补算其余三个账户。首次中断记录和修正记录均附带。", "",
        f"已复算16个账户指标、8条账户差额、{receipt['recomputed_cycle_records']}个完整周期成交利润、{receipt['replayed_decision_origins']}个决策原点及{receipt['replayed_predictions']}个原模型预测，最大预测差{receipt['maximum_prediction_error']:.3g}。一个策略设置对应两个来源执行版本，不能算成两个不同策略；最终16个评价记录不重复计算首次已保存账户。", "",
        "下一步停止继续给原学习进入增加广度门槛，改查两条已有信号合并持仓的简单机制：急跌回升或原学习状态任一有效持有时，使用账户可负担的完整预算；两者都退出时清仓。它有可能减少半仓、满仓间的分批买卖，但会提高资金暴露，不保证提高夏普。当前只完成有限查重和方向说明，尚无第64轮配置或账户。", "",
        "目标1.2尚未实现，持续研究保持进行，无GPT数值包及真实交易。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "entry_exit_coverage.csv", "signal_counts.csv",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_entry_fallback_profit_groups.csv", "saved_prediction_replay.csv",
        "saved_decision_state_replay.csv", "saved_entry_origin_cycle_comparison.csv", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    for name in ["tests_receipt.json", "tests_initial_null_representation_failure.json", "OUTPUT_FAILURE.json"]:
        shutil.copy2(PARENT_OUT / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "原冻结设置.json")
    shutil.copy2(CORRECTION, OUT / "输出字段修正登记.json")
    shutil.copy2(ROOT / cfg["chinese_saved_models"], OUT / "原学习模型逐月中文规则.md")
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    columns = [c for c in factors.columns if c not in ["wealth_mean20", "price_valid", "price_above20", "price_weak", "breadth_weak",
        "combined_entry_view", "combined_entry", "combined_rearm_allowed", "price_entry_view", "price_entry"]]
    factors[columns].to_csv(OUT / "全部逐日进入因子.csv", index=False, encoding="utf-8-sig")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_cycles.csv", OUT / f"{period}_{cost}_cycles.csv")
    NEXT_NOTE.write_text("""# 第63轮之后：已有急跌与学习信号合并持仓

第63轮广度仅筛选R32进入失败，两段两费用均低于原基线。主0.315538/0.259300、年化1.9886%、回撤11.4003%，较早0.723260/0.705640、年化8.0378%。主19周期9盈利38成交，较早7周期4盈利14成交，其中6周期广度缺失时回退原基线；主没有实际缺失回退进入。主160持仓收盘全部原预测，较早271中67有预测204无模型；18/1学习退出。基础终值比原R32主少55250.7372元、较早少8334.4395元。停止一半广度阈值、方向、等待、缺失回退或退出条件邻近调整。

主原546合格原点有260被广度有效未过半筛掉，较早574有144筛掉；空仓原资格有效但未另要求等待结束的拒绝日期105/10。主24旧与19新只有13共同进入原点，旧独有11新独有6；较早9与7只有5共同、旧4新2。2020旧02-05至03-02赚8366.35008、新03-03至03-13亏14093.62470；2021旧01-05至01-21赚9688.50168、新01-08至01-21亏2228.34248。另也避免旧2024-08-28亏8177.70374的原点，但整体不成立。全部对应在saved_entry_origin_cycle_comparison.csv，不依据特定日期反转或新选门槛。

原 main research/breadth_learned_entry_gate_v1.py 的首个主BASE账户完成后，统计覆盖误用了holding.learned_prediction；实际列为continuation_prediction。原配置和原文件不改。config/510300_breadth_learned_entry_gate_v1_output_fix.json 冻结仅输出修正，research/breadth_learned_entry_gate_v1_output_fix.py 原样保留首次四个文件，再完成另三账户。最终结果在 reports/research/510300_breadth_learned_entry_gate_v1_output_fix，原部分账户和OUTPUT_FAILURE.json在原文件夹。不能重新跑原脚本或修正脚本覆盖。10测试通过含4原账户重放，首次NaN/None表示问题按相同Parquet格式比较后相同，规则未变；最终16指标、8差额、52周期成本、5646原点、454原预测核对，最大预测差0，首个账户4文件逐字节保留。原D60中文逐月模型114拟合27无模型，本轮0拟合。

下一机制最多1设置，利用现有R46两个专家状态：PANIC V6与原R32学习。只要两个状态都有效且任一为1，下一开盘把510300目标设为100%；两者都0时目标0。借用空闲预算并把重叠信号净额合并，两个都1也不能超过100%。当一条退出、另一条仍为1时保持目标100%，不因专家交接先减半再加回；当都0才全退出。使用完整账户模拟，不能把两个保存收益直接相加，不能声称这是原固定各半账户简单乘2。会增加资金暴露，不保证提高夏普。若信号缺失保持NO_VIEW，不将缺失填零。

优先直接复用 reports/research/510300_panic_learned_equal_blend_v1/evaluation_states.parquet 与 earlier_diagnostic_states.parquet 的基础费用原点专家状态，或用 research/simple_signal_blend_v1.py decision_state 及原冻结决定复核。原研究 research/panic_learned_equal_blend_v1.py 已读，原设置是固定各半和每日收盘决定、下一开盘执行，使用 research/event_clock_account_v1.py simulate_event_account、10个百分点带宽。新方法沿用该实际现金、整手、费用和目标更新接口，明确目标连续100%时是否因分红和余额追加，未成交目标如何逐日更新。组合目标0会请求卖出，但原专家独立退出并不保证新组合直接触发6%损失止损；文件需要按真实整体规则说明。先必要的状态并集、未知状态、只留一个信号、费用与时钟测试，再冻结账户运行。0新模型、0新来源。

限定查重已读R28、R29、R9、R24、R46配置。R28三种旧原始价格信号里有多数及一种半仓两种满仓，没有本R32学习与V6全预算并集；R29是原价格信号固定25/50/75配比与波动缩减；R9已经做过市场状态专家跟踪，R24做过季度选择，不能称在线专家或策略切换从未研究。R46仅固定50/50，一直保留局部线索，未宣称独立稳定；本下一项改变持仓合并和资金占用，不再泛扫权重。目前没有第64轮配置、代码或收益。

索引截至63：337个不同设置、350已评价来源执行版本、355登记版本含5旧未运行、1052主评价记录；本轮1设置、2来源版本、8主记录和8较早记录，首次部分账户并入最终记录只计一次。EPS和慢来源补齐暂停，无GPT数值包，无新安全审计。只510300与现金，原两个历史窗口、20万元、242年化、两费用、分红、T+1和限价约束保持，目标未完成。
""", encoding="utf-8")
    record = {"round": 63, "study": result["study_id"], "title": "有效广度筛选原学习进入，缺失沿用原基线", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 2,
        "reporting_only_source_correction": str(CORRECTION.relative_to(ROOT)), "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"),
        "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption"]:
        index[key] += 1
    for key in ["evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += 8
    index.update(updated_at=now(), status="ROUND63_COMPLETE_BREADTH_LEARNED_GATE_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计63轮，337不同设置，350已评价来源版本，355登记含5旧未运行，1052主评价记录；本轮输出修正多一个来源版本，首个部分账户并入最终只计一次。",
        checks="第63轮10测试含4旧基线验证，16指标、8差额、52周期、5646决定、454原模型预测核对；首个账户四文件原样保留。",
        process_state_note="第63轮已完成全部账户、仅输出字段修正、归因及中文交付；第64轮只有两个既有信号全预算合并方向，没有登记或运行。",
        next_work={"status": "EXISTING_SIGNAL_UNION_DIRECTION_NOT_REGISTERED", "focus": "已有急跌回升与原学习状态任一持有时使用完整预算，两者都退出时清仓",
                   "source": str(NEXT_NOTE.relative_to(ROOT))}, latest_saved_breadth_entry_gate_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "BREADTH_LEARNED_ENTRY_GATE_ROUND63_CHINESE_RESULTS", "rounds": [63], "directory": str(OUT),
        "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "BREADTH_LEARNED_GATE_CHINESE_RULES_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
