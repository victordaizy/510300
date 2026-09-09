"""保存固定低点量价进入的失败归因、全部中文规则和有限的互补诊断。"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.low_anchored_reclaim_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300低点量价进入_第61轮_20260907"
DOCUMENT = OUT / "低点量价进入_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_LOW_ANCHOR_BREADTH_NEXT_STEP_20260907.md"


def verify_and_diagnose(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "低点量价冻结输入变化")
    data = pd.read_parquet(ROOT / cfg["features"])
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    checks, deltas, profits, replay, interactions, ideal = [], [], [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {m["model"]: pd.read_parquet(folder / f"{m['model']}_ledger.parquet") for m in result[key] if m["cost"] == cost}
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = accounts[m["model"]]
                nav = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(nav, ledger.equity, atol=1e-6, rtol=0), "账户净值无法由现金份额应收重建")
                require(np.allclose(ledger.equity / previous - 1., ledger.net_return, atol=1e-13, rtol=0), "每日净收益与净值不符")
                reproduced = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "mean_exposure"]:
                    require(abs(reproduced[field] - m[field]) < 1e-10, "保存指标复算不符")
                checks.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": reproduced["net_sharpe"]})
            current = accounts[PRIMARY]
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF"]:
                old = accounts[control]
                difference = float(current.equity.iloc[-1] - old.equity.iloc[-1])
                price = float(current.price_pnl.sum() - old.price_pnl.sum())
                dividend = float(current.dividend_recognized.sum() - old.dividend_recognized.sum())
                commission = float(current.commission.sum() - old.commission.sum())
                slippage = float(current.slippage_cost.sum() - old.slippage_cost.sum())
                error = difference - price - dividend + commission + slippage
                require(abs(error) < 1e-6, "新旧账户差额未完整解释")
                deltas.append({"period": period, "cost": cost, "control": control, "terminal_nav_difference": difference,
                               "price_pnl_difference": price, "dividend_difference": dividend, "commission_difference": commission,
                               "slippage_difference": slippage, "reconciliation_error": error})
                interactions.append({"period": period, "cost": cost, "control": control, "full_day_net_return_correlation": float(current.net_return.corr(old.net_return)),
                                     "both_held_closes": int((current.shares.gt(0) & old.shares.gt(0)).sum()),
                                     "new_only_held_closes": int((current.shares.gt(0) & old.shares.eq(0)).sum()),
                                     "control_only_held_closes": int((current.shares.eq(0) & old.shares.gt(0)).sum())})
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            total = float(cycles.net_profit_cny.sum())
            require(abs(current.equity.iloc[-1] - cfg["initial_capital"] - total) < 1e-6, "所有持仓周期利润与完整账户终值不一致")
            winning = float(cycles.loc[cycles.net_profit_cny.gt(0), "net_profit_cny"].sum())
            profits.append({"period": period, "cost": cost, "cycles": len(cycles), "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()),
                            "positive_profit_sum": winning, "nonpositive_profit_sum": float(cycles.loc[cycles.net_profit_cny.le(0), "net_profit_cny"].sum()),
                            "total_profit": total, "largest_cycle_profit": float(cycles.net_profit_cny.max()),
                            "largest_share_of_positive_profits": float(cycles.net_profit_cny.max() / winning),
                            "two_largest_share_of_positive_profits": float(cycles.net_profit_cny.nlargest(2).sum() / winning)})
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            for cycle in cycles.itertuples():
                origin = int(cycle.entry_index) - 1
                anchor = int(factors.anchor_index.iloc[origin])
                require(factors.raw_entry.iloc[origin] == 1 and anchor < origin, "实际进入没有过去已知起点及穿越")
                require(data.wealth.iloc[anchor] < data.wealth.iloc[anchor - 251:anchor].min() and anchor >= 251, "实际进入的起点不是真正252日新低")
                require(factors.anchor_index.iloc[origin - 1] == anchor, "实际进入跨了两个不同起点")
                require(data.wealth.iloc[origin - 1] <= factors.weighted_price.iloc[origin - 1] and data.wealth.iloc[origin] > factors.weighted_price.iloc[origin], "实际进入不是同起点向上穿越")
                held = decisions[decisions.anchor_cycle_id.eq(cycle.cycle_id)]
                maximum_error = 0.
                for row in held.itertuples():
                    t = int(row.origin_index)
                    window = data.iloc[anchor:t + 1]
                    adjusted = window[["high", "low", "close"]].mean(axis=1) * window.wealth / window.close
                    value = float((adjusted * window.volume).sum() / window.volume.sum())
                    error = abs(value - row.locked_weighted_price)
                    maximum_error = max(maximum_error, error)
                    require(error < 1e-11 and row.locked_anchor_index == anchor, "持仓累计均价或锁定起点复算不符")
                    require(bool(data.wealth.iloc[t] < value) == bool(row.below_locked_price), "保存固定起点退出判断与输入不符")
                replay.append({"period": period, "cost": cost, "cycle_id": int(cycle.cycle_id), "entry_origin": cycle.entry_origin,
                               "locked_anchor_date": data.date.iloc[anchor], "holding_closes": len(held), "maximum_weighted_price_error": maximum_error})
            keys = [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF"]
            returns = np.column_stack([accounts[k].net_return.to_numpy(float) for k in keys])
            mean, covariance = returns.mean(axis=0), np.cov(returns, rowvar=False, ddof=1)
            faces = []
            for count in range(1, 4):
                for selected in combinations(range(3), count):
                    selected = list(selected)
                    weights = np.linalg.solve(covariance[np.ix_(selected, selected)], mean[selected])
                    feasible = bool(np.all(weights >= 0) and weights.sum() > 0)
                    full = np.zeros(3)
                    if feasible:
                        full[selected] = weights / weights.sum()
                        sharpe = float(np.sqrt(cfg["annual_days"]) * (mean @ full) / np.sqrt(full @ covariance @ full))
                    else:
                        sharpe = None
                    faces.append({"period": period, "cost": cost, "selected_series": "；".join(keys[i] for i in selected),
                                  "nonnegative_feasible": feasible, "hindsight_constant_mix_sharpe": sharpe,
                                  **{f"weight_{k}": float(full[i]) if feasible else None for i, k in enumerate(keys)}})
            best = max((r for r in faces if r["nonnegative_feasible"]), key=lambda r: r["hindsight_constant_mix_sharpe"])
            for row in faces:
                row["best_in_this_period_cost"] = row is best
                row["tradable_account_computed"] = False
            ideal.extend(faces)
    for filename, rows in [("saved_metrics_recomputation.csv", checks), ("saved_account_differences.csv", deltas),
                           ("saved_cycle_profit_summary.csv", profits), ("saved_fixed_anchor_replay.csv", replay),
                           ("saved_path_interactions.csv", interactions), ("saved_hindsight_constant_mix_diagnostic.csv", ideal)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "ACCOUNT_ANCHOR_AND_PROFIT_RECOMPUTATION_COMPLETE", "recomputed_account_records": len(checks),
               "account_differences": len(deltas), "cycle_profit_sums": len(profits), "replayed_cycle_cost_records": len(replay),
               "replayed_holding_closes": sum(r["holding_closes"] for r in replay),
               "maximum_fixed_price_error": max(r["maximum_weighted_price_error"] for r in replay),
               "hindsight_constant_mix_face_records": len(ideal), "new_diagnostic_accounts": 0, "new_models": 0,
               "hindsight_mixing_is_not_a_tradable_policy_or_global_sharpe_bound": True,
               "security_audit_performed": False, "goal_achieved": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    return receipt, profits, [r for r in ideal if r["best_in_this_period_cost"]]


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 61)), "索引不是截至60轮，不能重复覆盖")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "不能关闭达到1.2的候选")
    receipt, profits, ideal = verify_and_diagnose(cfg, result)
    status = "COMPLETED_NO_STABLE_IMPROVEMENT_PROFIT_CONCENTRATION"
    decision = "主评价有正收益但低于原学习退出，较早收益接近零且压力费用复合收益为负；收益集中在少数行情，保存净收益的事后固定组合诊断未显示接近1.2的稳定互补。结束固定低点量价设置及邻近参数搜索。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False,
               "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
    OUT.mkdir(parents=True, exist_ok=True)
    files = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "entry_exit_coverage.csv", "signal_counts.csv", "result.json",
             "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_cycle_profit_summary.csv", "saved_fixed_anchor_replay.csv",
             "saved_path_interactions.csv", "saved_hindsight_constant_mix_diagnostic.csv", "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json"]
    for filename in files:
        (OUT / filename).write_bytes((RESEARCH / filename).read_bytes())
    cn = {"date": "日期", "wealth": "含分红财富指数", "typical_wealth_price": "财富尺度典型价格", "volume": "成交量", "prior_window_low": "此前251日最低水平",
          "new_anchor": "新的252日低点", "anchor_index": "当前起算日位置", "anchor_date": "当前起算日", "weighted_price": "当前起点累计量价价格",
          "cumulative_volume": "当前起点累计量", "cumulative_price_volume": "当前起点累计量价乘积", "invalid_rows": "无效数据行数",
          "range_status": "当前累计量价状态", "entry_cross": "同起点向上穿越", "raw_entry": "原始合格进入信号", "daily_inputs_complete": "当天量价完整",
          "relative_to_weighted_price": "收盘相对累计价格位置", "origin": "收盘决策日", "execution_date": "计划成交日", "action": "动作",
          "requested_quantity": "请求份额", "exit_reasons": "退出原因", "entry_rearmed": "再次进入资格", "locked_anchor_date": "本持仓固定起算日",
          "locked_weighted_price": "本持仓固定起点累计价格", "locked_cumulative_volume": "本持仓起点累计量", "locked_cumulative_price_volume": "本持仓起点累计量价乘积",
          "locked_invalid_rows": "本持仓起点无效行数", "locked_range_status": "本持仓累计数据状态", "current_wealth": "本日财富指数",
          "distance_to_locked_price": "本日相对持仓累计价格位置", "below_locked_price": "低于本持仓累计价格", "current_market_anchor_differs": "当前市场起点与本持仓不同",
          "additional_exit_requested": "固定起点退出请求", "additional_exit_reason": "固定起点退出原因"}
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    factors.rename(columns=cn).to_csv(OUT / "全部日期的量价因子.csv", index=False, encoding="utf-8-sig")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost, cost_name in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet").merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            decisions.rename(columns=cn).to_csv(OUT / f"{label}_{cost_name}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            ledger.rename(columns={"date": "日期", "cash": "现金", "shares": "份额", "equity": "账户净值", "net_return": "当日净收益率", "mark": "估值价格",
                                    "mark_clock": "估值时点", "commission": "佣金", "slippage_cost": "滑点成本", "filled_quantity": "实际成交份额",
                                    "requested_quantity": "请求份额", "dividend_receivable": "分红应收", "dividend_recognized": "确认分红", "dividend_paid": "到账分红",
                                    "exposure": "股票仓位", "execution_reasons": "执行原因", "status": "成交状态"}).to_csv(OUT / f"{label}_{cost_name}_完整账户.csv", index=False, encoding="utf-8-sig")
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            cycles.rename(columns={"entry_origin": "进入决策日", "entry_date": "买入日", "exit_date": "卖出日", "entry_quantity": "买入份额", "entry_cost_cny": "含佣金买入成本",
                                    "net_profit_cny": "周期净利润", "holding_intervals": "持有交易区间数", "dividend_cny": "周期确认分红", "exit_reasons": "退出原因",
                                    "locked_anchor_date": "固定起算日"}).to_csv(OUT / f"{label}_{cost_name}_全部持仓周期.csv", index=False, encoding="utf-8-sig")
    lines = ["# 510300低点量价进入：第61轮结果", "", "## 老板先看结论", "",
             "**这套进入机制没有实现净夏普1.2，也没有表现出稳定改善，结束该设置。** 主评价基础净夏普0.519、年化4.60%；较早历史净夏普只有0.086、年化0.28%，压力费用后较早复合年化为负0.38%。它确实找到不同交易，但少数上涨行情承担了大部分利润。", "",
             "没有继续补每股盈利预测或其他数据，也没有新增模型训练。用已发生的阶段低点作为量价计算起点，价格重新站上其累计量价均值时进入，跌破进入时固定起点的累计价格时退出，同时保留6%亏损、8%追踪退出和60日最长持有。", "",
             "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for key in [PRIMARY, "VWMA_20_60", "REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
        lines.append(f"|{metric(result,key)['name']}|{metric(result,key)['net_sharpe']:.3f}|{metric(result,key,cost='STRESS')['net_sharpe']:.3f}|{metric(result,key,'earlier_diagnostic')['net_sharpe']:.3f}|{metric(result,key,'earlier_diagnostic','STRESS')['net_sharpe']:.3f}|")
    lines += ["", "主评价为2020年1月2日至2026年8月14日开盘，共1604个账户日；较早历史为2015年1月5日至2019年12月31日开盘，共1219个账户日。各20万元，全部空仓日、佣金、滑点、分红及终点保留。", "",
              "## 完整历史表现", "", "### 主评价", ""] + table(result["all_metrics"])
    lines += ["### 较早历史", ""] + table(result["earlier_diagnostics"])
    lines += ["## 交易与利润来自哪里", "",
              "主评价37次合格穿越，涉及16个不同信号起点，最终发生25个完整持仓周期，实际使用13个起算日；较早28次合格穿越、11个信号起点，形成21个周期及11个实际起算日。进入信号可能发生在已经持有或等待期，不能把每个穿越都当成实际交易。全部决定日的当前累计量价、全部持仓收盘的固定累计量价均完整。", "",
              "主评价25个周期只有5个盈利，较早21个周期也只有5个盈利。具体利润合计如下：", "",
              "|历史及费用|盈利周期利润合计|亏损周期利润合计|全部周期净利润|最大单次占全部盈利利润|", "|---|---:|---:|---:|---:|"]
    for p in profits:
        lines.append(f"|{'主评价' if p['period']=='evaluation' else '较早历史'}／{'基础' if p['cost']=='BASE' else '压力'}|{p['positive_profit_sum']:,.2f}元|{p['nonpositive_profit_sum']:,.2f}元|{p['total_profit']:,.2f}元|{p['largest_share_of_positive_profits']:.2%}|")
    lines += ["", "基础主评价最大盈利来自2020年4月15日至7月15日的持仓，净赚54,719.55元；2024年9月26日至10月10日又贡献36,384.60元。较早最大盈利来自2019年1月7日至4月9日，净赚49,968.40元，但整个较早账户合计只赚2,806.01元。这说明抓到少数大涨与形成稳定策略仍有距离。所有其他亏损交易和年份都已保留。", "",
              "主评价22个周期记有跌破固定累计价格退出，较早17个周期记有该原因。主评价273个、较早338个实际持仓收盘都保留固定起点；各有1个收盘出现新的市场起点，但持仓起点没有跟着移动。", "",
              "252日只是识别新低的窗口，累计价格起点不会到期自动重置；未出现新低时，可能沿用更早的起算日。例如部分2020年交易仍从2019年1月2日起算。这是事先确定的状态规则，没有事后挑选最低点。", "",
              "## 与原策略是否互补", "",
              "基础主评价与原学习策略的完整日收益相关系数约0.447，较早约0.234；两者都持有的收盘分别65日、97日。新方案独自持有的收盘分别208日、241日，说明不是完全重复的仓位路径，但不同并不自动产生高夏普。", "",
              "另外只对三条已保存的完整净收益序列作事后代数诊断：本轮新策略、原学习策略及原急跌回升各半组合。使用整段已发生的收益计算均值和协方差，检查所有非空子集，求非负固定权重能给出的最高历史夏普。结果如下：", "",
              "|情景|事后固定混合近似净夏普|本轮新策略的事后权重|", "|---|---:|---:|"]
    for row in ideal:
        lines.append(f"|{'主评价' if row['period']=='evaluation' else '较早历史'}／{'基础' if row['cost']=='BASE' else '压力'}|{row['hindsight_constant_mix_sharpe']:.3f}|{row['weight_'+PRIMARY]:.2%}|")
    lines += ["", "该诊断看过未来，不能执行，也不算新增策略或账户。它只是对已经扣费的三条收益序列做固定组合计算，没有模拟共享现金、整数份额、净额成交或调整费用，更不能当成所有动态策略的夏普上限。诊断中较早历史选择不给本轮策略任何权重，主评价最优近似也仅0.950，因此没有据此再开一轮固定组合权重搜索。", "",
              "## 全部中文因子与进出场规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n\n", 1)[1]
    lines += [protocol.replace("\n## ", "\n### "), "",
              "## 交付和必要核对", "",
              "“全部日期的量价因子.csv”保留起算点和每个因子。四份“逐日全部因子和进出场.csv”、四份“完整账户.csv”、四份“全部持仓周期.csv”列明两段两费用的完整决定、现金日及所有盈亏。其他结果表保留程序字段，便于复算。没有学习模型系数，因为本轮未使用预测模型。", "",
              f"10项必要新测试在冻结前通过。保存结果已核对20条账户记录、8项新旧差额、4项周期利润合计，并重算{receipt['replayed_cycle_cost_records']}个周期费用记录和{receipt['replayed_holding_closes']}个持仓收盘的固定量价。事后混合诊断只有28条非空子集记录，没有新增回测账户、模型或参数研究。未制作GPT数值包，也未追加安全审计。", "",
              "## 下一步", "",
              "下一项先检查已经缓存的沪深300成分股上涨广度能否直接复用，用多数成分股是否共同上涨来补充单一ETF价格的信息。现有广度文件的日期覆盖和有效数据覆盖不同，不能看到日期齐全就当成每日因子有效。只做必要的现有字段核对，保留无观点状态；若不能快速复用，就换方向，继续暂停数据补齐。当前只完成缓存定位、字段和覆盖检查，尚未登记或运行下一策略。"]
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    NEXT_NOTE.write_text("""# 第61轮之后：只检查现有成分股广度的直接复用

第61轮低点累计量价进入已完成并结束。主评价0.518750/0.462305、年化4.6027%、回撤17.0883%；较早0.086284/0.035244、基础年化0.2770%、压力负0.3801%、回撤28.0098%。主25周期5盈利，较早21周期5盈利。全部因子实际完整，利润集中。只看三条保存净收益的事后非负固定权重代数诊断：主最高近似0.950070/0.882257、较早0.747698/0.724697，较早给新策略0权重；这是使用未来的非交易诊断，28子集记录、0新账户，不能当全局上限或可交易组合。中文交付、保存账户/起点/利润核对和索引完成。停止该起点、窗口、穿越、确认和保护参数搜索。

下一方向尝试信息来源不同的成分价格广度，但只复用现有免费缓存，不新下载或补历史。不交易任何成分股，执行仍510300与现金。最多一个有限新设置，考虑“多数成分共同上涨的恢复或确认”与价格趋势的组合，具体进入和退出尚未确定，不先读新策略收益。

定位到 data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/internal_F_T_features.parquet，395102字节、2823行，日期2015-01-05至2026-08-14。字段 breadth20、return20_scoreable_member_count、return20_coverage_ratio、point_in_time_member_count、internal_feature_state、internal_formula_coverage_state、four_state_daily_coverage_state 等。日期齐全不表示数据齐全：breadth20共有473日缺失；较早1219日只有758日广度有值，主1604日1592日有值。原内部组合状态仅较早163日、主1117日为VIEW_ALLOWED，不能宣称2823日组合可用，也不能把NO_VIEW改成VIEW或取 before_no_view_gate 列替代。

已经读过 research/stress_transmission_hazard_v2.py 约第590至745行和 research/stress_transmission_hazard_v2_mft_features_v1.py 约第590至644行。breadth20的定义是当日点时成员中，最近20个交易日总股东回报为正的数量，除以20日回报可计算成员数量；覆盖达到98%才保留该字段，否则缺失。当前成员数每日300，门槛即至少294个可计量成员。收益仅来自原四态中可用的状态，不插值。原F、T组合另要求尾部和共振覆盖及四态逐日组合门；原内部组合有很多NO_VIEW。下一步先确认 breadth20 是否作为独立、已按98%门处理的原始字段可用，以及四态来源与快照时钟应保留的条件。不得把旧F/T失败当成被修复，不降低它们的门槛，不重新计算F/T或重启DSV5预测链。

旧研究是成分脆弱性F/T对下行风险的增量预测及受限策略映射；当前v6允许新的有限历史方法直接完整账户，但仍不允许未来数据或忽略源缺失。单独广度研究若可快速复用，要先明确独立字段状态及缺失日的处理，所有空仓或回退日仍在完整日历中；缺失不能当成50%广度或自动预测空仓。不能把较早大片无数据造成的少交易解释成策略稳健。若需要额外补齐则不开展，立即换方向，避免重回慢数据路线。

其他定位到的广度缓存不够长：data/features/000300_equal_weight_breadth_daily.parquet 与 data/features/000300_official_weighted_breadth_daily_v1_0_1_warmup.parquet 均仅1211行，2021-08-12至2026-08-12，无法覆盖原完整两段。data/raw/constituents/000300_constituent_daily.parquet 484200行、1614日期，2019-12-23至2026-08-19；不要据此声称2015已有全量原日线，也不启动旧构建器补齐。

限定查重没有找到“breadth20超过50%恢复”的相同完整策略。config/510300_macd_breadth_downside_preflight_v1.yaml 的 breadth_thrust_5d 使用此前官方权重广度五日变化符号，是另一项旧机制预检，不等于新的独立广度进入；旧结果保留。后续应只读必要配置及源合同，不展开全部旧报告。现在只有来源定位和定义阅读，没有第62轮配置、模型或账户。

EPS、估值、研报日期、财报、公募及慢来源补齐继续暂停。20万元、242年化、原主与较早窗口、两档费用、零现金及无风险收益、T+1、100份、分红与限价约束均保持。目标未完成，无GPT数值包，无额外安全审计。
""", encoding="utf-8")
    record = {"round": 61, "study": result["study_id"], "title": "阶段新低后重回固定起点量价价格", "status": status,
              "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated",
              "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"),
              "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays",
                "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND61_COMPLETE_LOW_ANCHOR_NO_STABLE_IMPROVEMENT", running_studies=[], goal_achieved=False, latest_completed_round=record,
                 count_warning="累计61轮、334个不同配置或范围、346个已评价来源版本、1034个主评价记录；登记351含5个旧未运行绑定。较早、参考、内部拟合和非交易代数诊断另计。",
                 checks="第61轮10项必要测试、20条账户复算、8项账户差额、4项利润合计及92个周期费用记录中1222个持仓收盘固定起点价格核对完成。",
                 process_state_note="第61轮两段两费用账户、失败归因和普通中文交付完成；下一成分广度方向只有来源定位与字段定义检查，没有新配置或运行进程。",
                 next_work={"status": "EXISTING_SOURCE_DEFINITION_AND_COVERAGE_CHECK_ONLY", "focus": "只检查现有breadth20能否作为独立广度字段快速复用，不补数据",
                            "source": str(NEXT_NOTE.relative_to(ROOT))},
                 latest_saved_low_anchor_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "LOW_ANCHORED_RECLAIM_ROUND61_CHINESE_RESULTS", "rounds": [61], "directory": str(OUT),
                                "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "LOW_ANCHOR_CHINESE_RULES_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
                "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
                "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
