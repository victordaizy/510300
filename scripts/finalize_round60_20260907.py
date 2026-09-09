"""核对累积转弱规则没有改变原账户，交付完整中文因子和历史结果。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.cycle_cusum_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P32
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, CN, chinese_formula

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300持仓累积转弱_第60轮_20260907"
DOCUMENT = OUT / "持仓累积转弱_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_CUSUM_NEW_ENTRY_20260907.md"


def metric(result, model, period="evaluation", cost="BASE"):
    return next(m for m in result["all_metrics" if period == "evaluation" else "earlier_diagnostics"] if m["model"] == model and m["cost"] == cost)


def table(rows):
    lines = ["|方案|费用|净夏普|复合年化|最大回撤幅度|较买入持有年化差|平均股票仓位|成交笔数|佣金（元）|滑点（元）|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        lines.append(f"|{m['name']}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{-m['max_drawdown']:.2%}|{100*m['annualized_return_excess_vs_buy_hold']:.2f}个百分点|{m['mean_exposure']:.2%}|{m['trade_count']}|{m['commission']:.2f}|{m['slippage_cost']:.2f}|")
    return lines + [""]


def verify_saved(cfg, result, data):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "累积转弱登记文件变化")
    comparisons, checks, replay, alarms = [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            old = pd.read_parquet(P32 / period / cost / "REARM_RIDGE_ledger.parquet")
            current = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            economic = [c for c in old.columns if c != "execution_reasons"]
            pd.testing.assert_frame_equal(current[economic], old[economic])
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            old_decisions = pd.read_parquet(P32 / period / cost / "REARM_RIDGE_decisions.parquet")
            decision_fields = [c for c in old_decisions.columns if c != "exit_reasons"]
            pd.testing.assert_frame_equal(decisions[decision_fields], old_decisions[decision_fields])
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            old_cycles = pd.read_csv(P32 / period / cost / "REARM_RIDGE_cycles.csv")
            cycle_fields = [c for c in old_cycles.columns if c != "exit_reasons"]
            pd.testing.assert_frame_equal(cycles[cycle_fields], old_cycles[cycle_fields])
            comparisons.append({"period": period, "cost": cost, "account_days": len(current), "account_fields_exactly_equal_except_reasons": len(economic),
                                "decision_fields_exactly_equal_except_reasons": len(decision_fields), "cycle_fields_exactly_equal_except_reasons": len(cycle_fields),
                                "changed_quantity_decisions": 0, "changed_filled_quantities": 0, "maximum_equity_difference": 0.,
                                "changed_exit_reason_days": int(current.execution_reasons.ne(old.execution_reasons).sum())})
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                nav = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(nav, ledger.equity, atol=1e-6, rtol=0), "账户现金份额应收合计不一致")
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0), "每日收益与净值不一致")
                recomputed = summarize(ledger, cfg)
                for name in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost"]:
                    require(abs(recomputed[name] - m[name]) < 1e-10, "保存指标无法复算")
                checks.append({"period": period, "cost": cost, "model": m["model"], "net_sharpe": recomputed["net_sharpe"], "days": len(ledger)})
            for cycle in cycles.itertuples():
                group = decisions[decisions.cusum_cycle_id.eq(cycle.cycle_id)]
                anchor = int(cycle.entry_index) - 1
                history = data.total_log.iloc[anchor - cfg["baseline_trading_days"] + 1:anchor + 1].to_numpy(float)
                require(len(history) == 60 and np.isfinite(history).all(), "实际持仓前基准并不完整")
                mean, scale = float(np.mean(history)), float(np.std(history, ddof=1))
                require(np.allclose(group.cusum_baseline_mean, mean, atol=1e-14, rtol=0) and np.allclose(group.cusum_baseline_sigma, scale, atol=1e-14, rtol=0), "固定基准无法由入场前60日复算")
                statistic, alarmed, maximum_error = 0., False, 0.
                for row in group.itertuples():
                    if row.origin_index > cycle.entry_index:
                        statistic = max(0., statistic + (mean - data.total_log.iloc[int(row.origin_index)]) / scale - cfg["allowance_sigma"])
                    alarmed = alarmed or statistic >= cfg["alarm_threshold"]
                    maximum_error = max(maximum_error, abs(statistic - row.cusum_value))
                    require(maximum_error < 1e-11 and alarmed == bool(row.cusum_alarm), "逐日累积量或报警无法复算")
                replay.append({"period": period, "cost": cost, "cycle_id": int(cycle.cycle_id), "replayed_closes": len(group),
                               "maximum_statistic_error": maximum_error, "maximum_cusum": float(group.cusum_value.max()), "alarmed": alarmed})
                fired = group[group.cusum_alarm.eq(True)]
                if len(fired):
                    first = fired.iloc[0]
                    old_row = old_decisions[old_decisions.origin.eq(first.origin)].iloc[0]
                    require(old_row.requested_quantity < 0 and first.requested_quantity == old_row.requested_quantity, "报警存在不同的新退出动作，不能判为无增量")
                    alarms.append({"period": period, "cost": cost, "cycle_id": int(cycle.cycle_id), "first_alarm_origin": first.origin,
                                   "new_exit_reasons": first.exit_reasons, "original_exit_reasons": old_row.exit_reasons,
                                   "new_requested_quantity": int(first.requested_quantity), "old_requested_quantity": int(old_row.requested_quantity),
                                   "new_actual_exit_date": cycle.exit_date, "old_actual_exit_date": old_cycles.loc[old_cycles.cycle_id.eq(cycle.cycle_id), "exit_date"].iloc[0]})
    for filename, rows in [("saved_account_equality.csv", comparisons), ("saved_metrics_recomputation.csv", checks),
                           ("saved_cusum_replay.csv", replay), ("saved_alarm_overlap.csv", alarms)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "ALL_FOUR_ACCOUNT_PATHS_AND_ACTIONS_IDENTICAL_TO_ORIGINAL_EXCEPT_ADDITIONAL_REASONS",
               "recomputed_account_records": len(checks), "identical_candidate_account_paths": len(comparisons),
               "replayed_cycle_cost_records": len(replay), "replayed_holding_closes": sum(r["replayed_closes"] for r in replay),
               "maximum_cusum_replay_error": max(r["maximum_statistic_error"] for r in replay),
               "alarm_cost_records_already_exiting": len(alarms), "new_account_actions": 0,
               "new_diagnostic_accounts": 0, "new_model_fits": 0, "security_audit_performed": False, "goal_achieved": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    return receipt


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 60)), "索引不是截至59轮，不重复覆盖")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "不能关闭达到1.2的候选")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    verification = verify_saved(cfg, result, data)
    status = "COMPLETED_NO_INCREMENT_ALL_ACCOUNT_PATHS_IDENTICAL"
    decision = "主评价没有报警，较早两次报警与原固定止损和追踪退出同时成立；两段两费用所有账户数值、实际成交和原决策均相同。结束累积转弱附加退出，不降低阈值、不改容忍或窗口、不替换原退出继续搜索。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False,
               "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "model_coverage.csv", "cycle_fixed_baselines.csv",
                     "first_alarm_records.csv", "saved_account_equality.csv", "saved_metrics_recomputation.csv", "saved_cusum_replay.csv", "saved_alarm_overlap.csv",
                     "saved_verification_receipt.json", "acceptance_outcome.json", "result.json", "tests_receipt.json"]:
        (OUT / filename).write_bytes((RESEARCH / filename).read_bytes())
    cn = {"origin": "收盘决策日", "execution_date": "计划成交日", "action": "动作", "requested_quantity": "请求份额", "exit_reasons": "退出原因",
          "entry_rearmed": "再次进入资格", "learning_status": "学习模型状态", "learning_fit_origin": "原模型训练日",
          "continuation_prediction": "原模型继续持有优势预测", "negative_confirmation_count": "连续负预测次数", "learned_exit_requested": "原学习退出请求",
          "cusum_baseline_start": "固定基准起日", "cusum_baseline_end": "固定基准末日", "cusum_baseline_mean": "固定日平均收益",
          "cusum_baseline_sigma": "固定日波动", "cusum_daily_return": "本日完整持有区间收益", "cusum_standardized_shortfall": "本日标准化转弱程度",
          "cusum_value": "累积转弱值", "cusum_status": "累积检测状态", "cusum_information_available": "累积检测信息可用",
          "cusum_alarm": "累积转弱报警", "cusum_first_alarm_origin": "首次报警日", "additional_exit_requested": "额外退出请求",
          "additional_exit_reason": "额外退出原因", **dict(zip(FEATURES, CN))}
    ledger_cn = {"date": "日期", "cash": "现金", "shares": "份额", "equity": "账户净值", "net_return": "当日净收益率", "mark": "估值价格",
                 "mark_clock": "估值时点", "dividend_receivable": "分红应收", "dividend_recognized": "确认分红", "dividend_paid": "到账分红",
                 "commission": "佣金", "slippage_cost": "滑点成本", "filled_quantity": "实际成交份额", "requested_quantity": "请求份额",
                 "exposure": "股票仓位", "execution_reasons": "执行原因", "status": "成交状态"}
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        factors = pd.read_parquet(RESEARCH / f"{period}_factors.parquet").rename(columns={"date": "origin"})
        for cost, cost_name in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet").merge(factors, on="origin", how="left", validate="one_to_one")
            decisions.rename(columns={**cn, "d60_factor": "日内相对隔夜强弱", "raw_entry": "原进入条件", "raw_exit": "原价格退出条件",
                                      "total_log": "市场当日含分红对数收益", "daily_information_complete": "市场日收益完整"}).to_csv(
                OUT / f"{label}_{cost_name}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            ledger.rename(columns=ledger_cn).to_csv(OUT / f"{label}_{cost_name}_完整账户.csv", index=False, encoding="utf-8-sig")
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            cycles.rename(columns={"entry_origin": "进入决策日", "entry_date": "买入日", "exit_date": "卖出日", "entry_quantity": "买入份额",
                                    "entry_cost_cny": "含佣金买入成本", "net_profit_cny": "周期净利润", "holding_intervals": "持有交易区间数",
                                    "dividend_cny": "周期确认分红", "exit_reasons": "退出原因"}).to_csv(OUT / f"{label}_{cost_name}_全部持仓周期.csv", index=False, encoding="utf-8-sig")
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"][cfg["model_key"]]
    model_lines = ["# 第60轮沿用的原学习模型逐月中文规则", "", "以下为第31轮原日内强弱线性模型，114次成功拟合、27次无模型；本轮没有重新训练。没有使用第59轮扩展进入路径的模型。", ""]
    for m in models:
        model_lines += [f"## {m['fit_origin']}", ""]
        if m["status"] == "FIT_COMPLETE":
            model_lines += [f"使用{m['training_cycle_count']}个成熟周期、{m['training_rows']}条状态，最新退出日{m['latest_exit_date']}。", ""] + chinese_formula(m["model"]) + [""]
        else:
            model_lines += ["成熟周期或状态不足，原学习模型无观点；不产生零收益预测。", ""]
    (OUT / "原学习模型逐月中文规则.md").write_text("\n".join(model_lines) + "\n", encoding="utf-8")
    lines = ["# 510300持仓累积转弱：第60轮结果", "", "## 老板先看结论", "",
             "**新增退出条件没有改变任何一笔实际成交，也没有提高净夏普，结束这项改动。** 主评价没有出现累积转弱报警；较早历史的两次报警与原6%亏损和8%追踪退出同时成立，原账户在同一时点已经要求卖出。增加一条看起来合理的规则，并不必然增加策略信息。", "",
             "这轮实际完成一个新设置、两段历史及两档费用。没有新模型训练、没有新参考路径、没有补EPS或外部数据。", "",
             "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for key in [PRIMARY, "REARM_RIDGE", "REARM_NONE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
        lines.append(f"|{metric(result,key)['name']}|{metric(result,key)['net_sharpe']:.3f}|{metric(result,key,cost='STRESS')['net_sharpe']:.3f}|{metric(result,key,'earlier_diagnostic')['net_sharpe']:.3f}|{metric(result,key,'earlier_diagnostic','STRESS')['net_sharpe']:.3f}|")
    lines += ["", "主评价2020年1月2日至2026年8月14日开盘，1604个账户日；较早2015年1月5日至2019年12月31日开盘，1219个账户日。两段各20万元，保留全部空仓日、分红、佣金和滑点。净夏普1.2目标仍未完成，局部组合0.944也没有证明稳定高夏普。", "",
              "## 新条件怎样工作", "",
              "实际买入前一个收盘，用刚过去60日的含分红收益固定平均值和波动。买入后的每个完整持有交易日，把低于原平均水平的程度按原波动标准化并累积，每天扣除0.5的小幅容忍；累计低于0则归零，达到5时下一开盘请求全部退出。买入当天不使用其中尚未持有的隔夜收益。原自然退出和原学习退出都保留。", "",
              "## 为什么没有改善", "",
              "主评价24个持仓周期、252个持仓收盘的检测信息全部完整，累计量最高4.740011，未达到预先固定的5。因此不是数据缺失导致不报警，而是在这段原持仓路径上新条件没有先触发。", "",
              "较早9个持仓周期、299个持仓收盘信息也完整。两次报警都被原退出规则覆盖：", "",
              "|报警收盘日|累计值|原规则同日已经触发|原、新实际卖出日|", "|---|---:|---|---|",
              "|2015年1月19日|5.472922|固定止损和追踪退出|2015年1月20日|",
              "|2015年6月19日|6.547558|固定止损和追踪退出|2015年6月23日|", "",
              "以上是两个不同事件，基础与压力费用分别保存一次，共四条报警费用记录，不能算四次独立机会。日志增加了真实的累积转弱原因，原来的卖出请求、日期、份额和成交没有变化。", "",
              "已逐项核对四条候选账户：除新增退出原因文本以外，全部原账户字段、全部原决策字段和全部持仓周期字段与原学习策略一致。账户净值、成交、佣金、滑点差额均为零。不会因为最高值接近5就把阈值降到4.7，也不改窗口和容忍继续追结果。", "",
              "## 完整历史表现", "", "### 主评价", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早历史", ""] + table(result["earlier_diagnostics"])
    lines += ["主评价24个周期、14个盈利、48笔成交；较早9个周期、5个盈利、18笔成交。两档费用的周期和成交数量一致。主评价23个周期记有原学习退出原因，较早2个周期记有原学习退出原因；较早新增累积报警的2个周期与原固定止损重叠。", "",
              "## 全部中文因子、进入、持有、退出及再次进入", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n\n", 1)[1]
    lines += [protocol.replace("\n## ", "\n### "), "",
              "## 文件和必要核对", "",
              "“原学习模型逐月中文规则.md”逐月列出原模型的全部系数及无模型状态。四份“逐日全部因子和进出场.csv”包含原八因子、进入因子、新固定基准、每日标准化转弱、累计值、报警及实际请求。四份“完整账户.csv”保留所有现金日，四份“全部持仓周期.csv”保留全部盈利和亏损周期。其他结果表保留程序原始字段供复算。", "",
              f"26项必要测试通过后冻结。保存结果复算覆盖{verification['recomputed_account_records']}条账户比较记录，以及{verification['replayed_cycle_cost_records']}个周期费用记录、{verification['replayed_holding_closes']}个持仓收盘的累积值，四个候选账户的经济路径和原决策逐项相同。核对没有新增账户或模型，不代表收益目标通过。合成测试的浮点边界修正记录在配置和测试回执中，发生在正式收益读取前。没有额外安全审计或GPT数值包。", "",
              "## 下一步", "",
              "下一轮改为寻找不同的进入机会。准备检验价格在已发生的阶段低点后，重新站上从该低点起累计的成交量加权价格，是否能形成可用的回升进入条件；这是待检验的日线价格代理，不称作公募申购或投资者真实持仓成本。先把起算点在当时如何确定、分红口径及进入后何时退出写清，最多登记一个设置。目前尚未登记或运行该下一轮。"]
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    NEXT_NOTE.write_text("""# 第60轮之后：用事先可确定起点的成交量加权价格寻找新进入机会

第60轮累积转弱附加退出已完成：四个候选账户经济字段、成交和原决策完全等于R32，主0.704868/0.641136、较早0.747698/0.724697。主24周期最高累计4.740011无报警；较早2015-01-19和06-19两次5.472922/6.547558报警与原固定止损及追踪退出同日重叠。完整核对与中文交付、索引均已完成，停止该阈值/容忍/窗口/组合方式搜索。

下一项拟改进入机制，最多一个有限设置，不继续给D60叠加筛选或重训。使用现有日线，以当天已经观察到的过去252交易日收盘新低作为可因果确定的起算日，累计从起算日至当前的成交量加权价格；后来价格从其下方重新站上时尝试下一开盘进入。锚点必须是当时已知的新低，不可用全样本最低点、事后转折点或手工挑日期。当前只有方向与查重完成，尚未确定完整进出场、登记第61轮或生成收益结果。

方法出处已查：TradingView官方“Anchored VWAP drawing tool” https://www.tradingview.com/support/solutions/43000669764-anchored-vwap-drawing-tool/ 介绍从所选起点累计量价；其VWAP官方说明 https://www.tradingview.com/support/solutions/43000502018-volume-weighted-average-price-vwap/ 给出典型价格乘成交量的累计和除以累计量。它们是计算方法出处，不证明择时或超额收益。

事前需要定清：价格用含分红财富尺度；每日最高、最低和收盘的均价可用同日财富指数除收盘价比例转成共同尺度，再按成交量加权。日线典型价格只能叫累计量价代理，不是逐笔真实成交均价、公募净流入或投资者持仓成本。成交量统一乘常数不应改变该比值。遇到缺失或零累计量保留无观点。需确认252日新低是否含今日、至少多少历史、同价如何处理、起算当天不自触发和跨起算点不产生伪穿越。

进入后应锁定本次进入所对应的起算日，再继续纳入随后实际发生的量价；不能在下跌后重置到更低锚点而放松原退出线。退出规则要先明确：例如重新跌破该固定起点的累计量价代理，以及复用已有固定风险退出规则。不要在运行后依据结果改变起点、风险阈值或确认天数。具体唯一版本须先写中文协议和必要测试后冻结，再完整两段两费用比较。可以用现有通用额外退出引擎记录正确原因，或已有单次进出账户，保留原模型及买入持有作对照。

限定查重：research/config/docs 文件名及源码中的 anchored、avwap、锚定成交、锚定均价，只查到旧 research/anchored_sparse_mean_reversion_grid_v1.py 与 docs/510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_SPEC.md。已读其定义，它以510300和000300同步分钟价格构造指数隐含价格锚及基差事件，未使用由阶段低点起算的成交量加权价格；不得混淆或重启旧网格。第37轮 research/volume_weighted_trend_v1.py 与 docs/510300_VOLUME_WEIGHTED_TREND_V1.md 使用20/60固定滚动均线，不是固定低点起算窗口，其失败结果保留，不能说量价方法尚未试过。旧盈利加仓scope_check只描述6%浮盈条件主评价3个旧周期，没运行真实加仓账户，本轮不转去重做那项。

EPS及慢数据补齐继续暂停，只用510300与现金，不开其他证券权限，不做GPT数值包或安全审计。主2020-01-02至2026-08-14开盘，较早2015-01-05至2019-12-31开盘，各20万元、242年化、原两费用、现金和无风险收益零，保留全部空仓日及成交分红限制。目标仍未完成。
""", encoding="utf-8")
    record = {"round": 60, "study": result["study_id"], "title": "原学习退出加持仓累积转弱退出", "status": status,
              "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts",
              "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"),
              "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), status="ROUND60_COMPLETE_CUSUM_ADDITIONAL_EXIT_NO_INCREMENT", running_studies=[], goal_achieved=False, latest_completed_round=record,
                 count_warning="累计60轮、333个不同配置或范围、345个已评价来源版本、1024个主评价记录；登记350含5个旧未运行绑定。较早、参考、内部拟合另计。",
                 checks="第60轮26项必要测试、20条账户指标复算、四候选路径及原决定逐项相同、66个周期费用记录的1102个持仓收盘累计值复算完成。",
                 process_state_note="第60轮两段两费用账户完成，经济路径完全等于旧R32，新增退出没有独立动作。目标未完成，下一进入机制尚未登记或运行。",
                 next_work={"status": "DIRECTION_AND_PRIOR_METHOD_SEARCH_ONLY_NOT_FROZEN_NOT_RUNNING", "focus": "低点之后重回固定起点成交量加权价格的进入机制，最多一个新设置",
                            "source": str(NEXT_NOTE.relative_to(ROOT))},
                 latest_saved_cusum_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "CYCLE_CUSUM_EXIT_ROUND60_CHINESE_RESULTS", "rounds": [60], "directory": str(OUT),
                                "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "CHINESE_CUSUM_RULES_AND_SAVED_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
               "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
               "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", receipt)
    print(json.dumps({"交付": receipt, "核对": verification}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
