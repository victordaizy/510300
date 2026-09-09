"""记录低开收复的真实失败原因、全部中文规则与下一信息方向。"""
from __future__ import annotations

import json
import shutil
from decimal import Decimal

import numpy as np
import pandas as pd

from research.gap_recovery_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.saved_policy_account_review import review_accounts
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300低开收复与日内退出_第69轮_20260908"
DOCUMENT = OUT / "低开收复与日内退出_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_GAP_RECOVERY_GLOBAL_VOLATILITY_TERM_20260908.md"
SOURCE_PROBE = ROOT / "data/source_probes/cboe_volatility_term_20260908"


def factor_review(cfg):
    data = pd.read_parquet(ROOT / cfg["features"])
    saved = pd.read_parquet(RESEARCH / "factors.parquet")
    first_valid, events, valid, checks = None, [], [], []
    quantum = Decimal(str(cfg["tick"]))
    for t, row in enumerate(data.itertuples()):
        values = [row.open, row.close, row.previous_close, row.dividend]
        complete = all(np.isfinite(x) for x in values)
        valid.append(complete)
        if not complete:
            events.append(np.nan)
            checks.append({"date": row.date, "entry_event": np.nan, "weak_exit_event": np.nan,
                "opening_gap_after_dividend": np.nan, "close_recovery_after_dividend": np.nan, "intraday_change": np.nan})
            continue
        op, cl, previous, cash = [Decimal(str(x)).quantize(quantum) for x in values]
        entry, weak = op + cash < previous and cl + cash > previous, cl <= op
        require(not (entry and weak), "进入和退出事件意外冲突")
        events.append(1. if entry else 0. if weak else np.nan)
        if first_valid is None:
            first_valid = t
            if np.isnan(events[-1]):
                events[-1] = 0.
        checks.append({"date": row.date, "entry_event": float(entry), "weak_exit_event": float(weak),
            "opening_gap_after_dividend": float(op + cash - previous), "close_recovery_after_dividend": float(cl + cash - previous),
            "intraday_change": float(cl - op)})
    reference = pd.DataFrame(checks)
    intent = pd.Series(events).ffill()
    reference["policy_intent"] = intent
    reference["target"] = intent.where(valid)
    errors = {}
    for field in ["entry_event", "weak_exit_event", "opening_gap_after_dividend", "close_recovery_after_dividend", "intraday_change", "policy_intent", "target"]:
        tolerance = 1e-12 if "change" in field or "dividend" in field else 0.
        np.testing.assert_allclose(reference[field], saved[field], atol=tolerance, rtol=0, equal_nan=True)
        errors[field] = float((reference[field] - saved[field]).abs().max())
    reference.to_csv(RESEARCH / "saved_decimal_event_replay.csv", index=False, encoding="utf-8-sig")
    return reference.target.to_numpy(float), {"rows": len(reference), "maximum_errors": errors,
        "method": "DECIMAL_MONEY_COMPARISONS_AND_FORWARD_FILLED_EVENT_STATES"}


def timing_review():
    factors = pd.read_parquet(RESEARCH / "factors.parquet").set_index("date")
    rows, groups = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            current = []
            for trade in ledger[ledger.filled_quantity.gt(0)].itertuples():
                origin = factors.loc[trade.origin]
                row = {"period": period, "cost": cost, "origin": trade.origin, "entry_date": trade.date,
                    "origin_has_entry_event": origin.entry_event == 1,
                    "signal_day_intraday_return_not_earned": origin.close / origin.open - 1,
                    "actual_entry_day_intraday_return_before_cost": trade.mark / trade.open - 1}
                current.append(row)
            rows.extend(current)
            groups.append({"period": period, "cost": cost, "buy_count": len(current),
                "all_buy_origins_have_entry_events": all(x["origin_has_entry_event"] for x in current),
                "mean_signal_day_intraday_return_not_earned": float(np.mean([x["signal_day_intraday_return_not_earned"] for x in current])),
                "mean_actual_entry_day_intraday_return_before_cost": float(np.mean([x["actual_entry_day_intraday_return_before_cost"] for x in current]))})
    pd.DataFrame(rows).to_csv(RESEARCH / "saved_signal_day_vs_entry_day.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(groups).to_csv(RESEARCH / "saved_signal_day_vs_entry_day_summary.csv", index=False, encoding="utf-8-sig")
    return groups


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 69)), "索引不是截至68轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "低开收复冻结文件改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    targets, factor_receipt = factor_review(cfg)
    receipt, groups = review_accounts(cfg, result, RESEARCH, PRIMARY, targets)
    timing = timing_review()
    receipt.update(independent_factor_replay=factor_receipt, entry_timing_records=sum(x["buy_count"] for x in timing),
        accounting_reviewer_source_sha256=digest(ROOT / "scripts/saved_policy_account_review.py"))
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_REJECTED_GAP_RECOVERY_NEGATIVE_AFTER_EXECUTION_AND_COSTS"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "两段两费用均亏损，主评价价格加分红在扣费前已亏损，短周期高换手进一步侵蚀；不能把信号当天上涨当作可赚收益。结束该事件规则，不反向或调强弱门槛挽救。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    source_probe = json.loads((SOURCE_PROBE / "source_probe_receipt.json").read_text(encoding="utf-8"))
    source_coverage = json.loads((SOURCE_PROBE / "source_pair_coverage_receipt.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300低开收复与日内退出：第69轮", "", "## 结果和决定", "",
        "**这条规则失败，直接结束，不继续调参。** 主评价基础净夏普负1.108、压力负1.602；较早基础负0.343、压力负0.785。主评价基础年化负11.73%、最大回撤60.22%；较早基础年化负4.39%、最大回撤33.18%。夏普1.2目标仍未实现。", "",
        "进入要求除息修正后低开、收盘严格收复前收盘；退出要求持仓期间收盘不高于当天开盘。信号在收盘才完整，实际买卖在下一开盘。本轮只用现有行情，10项关键测试后冻结一套规则，四个新账户一次完成。", "",
        "|方案|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "PANIC_ONLY", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "原急跌主评价只有3个完整周期，较早8个且亏损；原学习和原各半也尚未得到独立稳定超额证明。保留这些候选供比较，不称为已达到目标。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "主评价2020年1月2日至2026年8月14日开盘，共1604日；较早2015年1月5日至2019年12月31日开盘，共1219日。每段20万元，242日年化，现金收益为零，包含全部空仓日、分红、佣金和滑点。", "",
        "## 失败原因：看到上涨与赚到上涨不同", "",
        "首先，进入条件挑出了当日低开高走的行情，但当天上涨在下一开盘买入前已经发生。下面只看实际买入对应的日期，比较信号原点当天与实际买入当天的开盘至收盘涨幅。两档费用的交易日期相同，因此展示基础费用对应日期。", "",
        "|历史|实际买入次数|信号当天平均日内涨幅，策略没有赚到|实际买入当天平均日内涨幅，未扣费|", "|---|---:|---:|---:|"]
    for item in timing:
        if item["cost"] == "BASE":
            lines.append(f"|{'主评价' if item['period'] == 'evaluation' else '较早'}|{item['buy_count']}|{item['mean_signal_day_intraday_return_not_earned']:.4%}|{item['mean_actual_entry_day_intraday_return_before_cost']:.4%}|")
    lines += ["", "信号当天的正收益是进入条件挑选出来的已发生上涨，不是预测能力证据。实际买入当天的简单平均日内收益也不是完整策略收益：它没有包括全部持有日、退出前隔夜、费用和分红，不能用该平均数代替账户夏普。", "",
        "其次，持有很短、交易频繁。主评价258个完整周期516笔成交，持仓486个收盘，平均每周期1.88个收盘；较早167个周期334笔成交，持仓340个收盘，平均2.04个收盘。两个时期都没有追加、未成交或评价期内资料缺失。", "",
        "|历史及费用|完整周期|盈利周期|亏损周期|价格损益|分红|佣金|滑点|最终净利润|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        label = ("主评价" if g["period"] == "evaluation" else "较早") + ("／基础" if g["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{g['cycles']}|{g['positive_cycles']}|{g['negative_cycles']}|{g['gross_price_profit']:,.2f}元|{g['dividend_recognized']:,.2f}元|{g['commission']:,.2f}元|{g['slippage']:,.2f}元|{g['net_profit']:,.2f}元|")
    lines += ["", "主评价基础价格损益加分红已经亏57,303.70元，佣金和滑点再花55,245.21元，最终亏112,548.91元；所以不是只要降低费用就能修好。较早价格损益加分红为4,853.90元，但费用45,369.68元，最终亏40,515.78元。这是同一保存账户的经济分解，没有另外重跑免手续费策略。", "",
        "主评价初始目标为一，且首个买入原点确有进入事件；末端意图为零，486个满目标原点对应486个持仓收盘。较早初始目标为零，末端意图为一，341个满目标原点减去统一终点开盘退出的一日，对应340个持仓收盘。所有实际买入原点均满足进入事件，没有将重复信号数冒充交易周期数。", "",
        "## 全部因子与中文进出场规则", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1].replace("\n## ", "\n### "),
        "", "## 各年基础费用表现", "", "|时期|年份|净夏普|该年实际期间收益|成交笔数|", "|---|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        value = f"{row.net_sharpe:.3f}" if np.isfinite(row.net_sharpe) else "无波动，未定义"
        lines.append(f"|{'主评价' if row.period == 'evaluation' else '较早'}|{row.year}|{value}|{row.cumulative_return:.2%}|{row.trade_count}|")
    lines += ["", "年度仅展示阶段差异，不能据此删掉亏损年份或翻转交易方向。", "",
        "## 核对完成，转向新的信息来源", "",
        f"10项必要测试一次通过，测试本体3.60秒，正式冻结和四账户运行均无中断。信号四项原资料完整，只有最初一行缺前收盘，未影响两评价窗口。用十进制金额和事件前向延续独立还原3456行因子及目标；20指标12差额、{receipt['replayed_decision_origins']}个实际份额请求、{receipt['actual_cycles']}个含分红完整周期和{receipt['entry_timing_records']}个买入时点均已核对，没有重新模拟账户。", "",
        "下一步不继续修改日线形态，先检查海外不同期限的预期波动是否能给出风险状态。Cboe官方页面直接提供VIX和九天波动率指数的历史日线入口，这次已取得两个小文件；它们只作为观察因子，交易仍限510300和现金。[Cboe官方历史数据](https://www.cboe.com/tradable_products/vix/vix_historical_data)", "",
        "已取得VIX的9266行与九天指数的3941行，均到2026年9月4日，无重复日期或非正收盘。来源检查会按既定行情终点截断，不能把终点后的数据用于本轮或下一轮历史决定。两个文件在2011年1月4日至2026年8月14日合并日历有3958日，其中32个源日期缺少一项；以美国东部17时视为收盘完成、北京时间下一中国交易日9时作决定的试配口径，主评价1604个原点中1573个有完整且不过期的一对，31个无观点；较早1219个均有完整的一对。保留不完整源日期，不跨日期拼接比值，也不通过跳过最新不完整日来虚构完整覆盖。", "",
        "拟只检验一个规则：九天预期波动低于一个月预期波动时允许持有，达到或超过时退出；最新资料不完整或超过时效则保留实际份额。夏令时、共同源日期、9时可用边界、缺失以及实际份额换算必须先明确并测试，再冻结。当前只完成公开来源和潜在覆盖检查，第70轮尚未构造比值、登记或回测；历史首次送达时间也没有独立证明。此前第3轮用过VIX水平和海外价格模型，本次不是宣称从未用过海外信息。", "",
        "研究继续，目标仍未实现。EPS与慢来源补齐暂停，不制作GPT数值包，不扩大真实交易权限。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    names = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv", "source_receipt.json",
        "tests_receipt.json", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json", "saved_decimal_event_replay.csv",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv",
        "saved_no_view_requests.csv", "saved_signal_day_vs_entry_day.csv", "saved_signal_day_vs_entry_day_summary.csv"]
    for name in names:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    mapping = {"date": "日期", "open": "开盘", "close": "收盘", "previous_close": "前收盘", "dividend": "当天每份分红",
        "opening_gap_after_dividend": "除息修正后开盘缺口", "close_recovery_after_dividend": "除息修正后收盘收复额", "intraday_change": "日内涨跌额",
        "entry_event": "进入事件", "weak_exit_event": "转弱退出事件", "source_state": "资料状态", "policy_intent": "保留的策略意图", "target": "当日有效目标", "policy_reason": "原因"}
    pd.read_parquet(RESEARCH / "factors.parquet").rename(columns=mapping).to_csv(OUT / "全部逐日因子和状态.csv", index=False, encoding="utf-8-sig")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_trades.csv", OUT / f"{period}_{cost}_trades.csv")
    NEXT.write_text("""# 第69轮后：海外波动期限信息来源已经小范围取得

第69轮510300_GAP_RECOVERY_V1完整失败并交付。主BASE/STRESS夏普负1.107611/负1.601713、基础年化负11.7333%、回撤60.2206%、累计负56.2745%；早负0.342706/负0.784698、基础年化负4.3945%、回撤33.1757%、累计负20.2579%。主258周期516成交486持仓收盘，基础73盈185亏；早167周期334成交340持仓收盘，基础66盈101亏；各0追加、0未成交、0评价原点NO_VIEW。主初始1且该原点有进入事件，末端0；早初始0末端1，341目标1原点减终点1=340持仓。

主基础价格负64182加分红6878.3=负57303.7，佣金13580.81384加滑点41664.4=55245.21384，净负112548.91384。早价格1967.3加分红2886.6=4853.9，佣金11384.57912加滑点33985.1=45369.67912，净负40515.77912。信号原点平均日内上涨主0.972353%、早1.147668%，实际买入当天平均开收涨幅主负0.022341%、早0.212431%，均只是时点诊断，不是含全部持仓和费用的账户收益。所有买入原点有低开收复事件。不能把信号当天已经发生的涨幅算进利润，结束该形态，不翻向或调阈值挽救。

冻结helper research/gap_recovery_inputs_v1.py、runner research/gap_recovery_v1.py，config/510300_gap_recovery_v1.json、docs/510300_GAP_RECOVERY_V1.md；结果 reports/research/510300_gap_recovery_v1。四项输入除首日前收盘外完整，前收盘等于上一实际收盘、每日分红等于事件账本，金额均千分之一单位。整数比较避免浮点制造严格突破；10测试3.60秒一次通过，冻结和run均一次完成。保存十进制事件及前向延续独立复算3456行，20指标12差额5646实际请求850含分红周期和850买入时点核对完成。通用只读核对 scripts/saved_policy_account_review.py 重用旧saved_cycles，不产生新拟合或账户。finalize_round69_20260908.py完成后不要重跑。交付 deliverables/510300低开收复与日内退出_第69轮_20260908，规则全中文，附逐日因子与完整账户，无GPT包。

索引截至69应为343不同设置、356已评价来源版本、361登记含5旧未运行、1112主评价记录。前目标回合67/68完成是PROGRESS，本回合69及新源检查也属于实质进展，不标记目标完成。每轮仅1设置1来源、4新账户16旧对照、0拟合0新参考周期。

下一70只完成来源检查，未生成比值/策略目标、未登记或回测。初步想用Cboe九天波动率相对一个月波动率识别海外风险状态：VIX9D收盘低于VIX收盘时持有510300，达到或超过时退出；只是一个拟议条件，不是有效性结论。只观察海外指数，不交易期货或期权。第3轮已有海外信息研究，使用旧Yahoo的VIX水平、价格变化和其他全球指数做16模型，主夏普0.362994、事后最好0.421404，不能重新包装为海外信息首次研究。新变化是期限关系和直接官方小文件，没有额外模型。

官方入口 https://www.cboe.com/tradable_products/vix/vix_historical_data 已明确给VIX、VIX9D下载链接，两个文件于2026-09-08 01:30:47北京时间取得，HTTP200默认TLS校验，未使用凭证或付费。保存 data/source_probes/cboe_volatility_term_20260908/VIX_History.csv（472309字节、9266行、1990-01-02至2026-09-04，sha256 9258a25814fb936e7df32a96bcf3914fdbf342afb33cdf8759cdfbc8064ddb72），VIX9D_History.csv（200183字节、3941行、2011-01-04至2026-09-04，sha256 a4058453d315d6ddf8527d4b2f519a993b58027a865aef2239c769e4500db02c）。source_probe_receipt.json保存URL/日期/响应头/哈希；source_pair_coverage_receipt.json保存日期、正值和潜在时钟检查。不得重复下载覆盖；目前收盘均正、日期无重复。先看过VIX3M动态仪表盘，没有直接取得历史文件，转用官方已列出的9天文件，不继续调查3月源，也没有建立3月候选。

用两个源日期外连接并截2011-01-04至2026-08-14共3958源日期，32日期有一项缺失，具体在coverage_receipt。源日期不能内连接后悄悄跳过，也不把前一日9天指数与当日VIX拼接。潜在时钟：美国东部当天17点视为收盘完成，用时区库处理夏令时，换算北京时间；每个中国执行日09点选择不晚于该时刻的最新源日期行，年龄按可用时刻至决定时刻的自然日小数，≤7天有效。最新行有任一指数缺失则NO_VIEW，不能跳回先前完整的一对隐藏缺失。主1604原点1573完整不过期、31NO_VIEW；早1219全部完整。缺失日仍进完整账户，保留实际份额。新CSV有8/14以后的原始记录，构建时限定旧评价终点，不能用其参与更早决定。该来源仅历史可行性，最初历史实际送达未证明，17时是保守市场完成假设，不是真实时延证据。

9时决定时钟可沿用旧R3 docs/510300_OVERNIGHT_GLOBAL_INFORMATION_V1_PROTOCOL.md以及research/overnight_global_information_v1.py的available_times思路，但不调用其train/prepare（会重读训练收益）。中国份额预算只用前一收盘价格和账户余额，外部观察只用9时已知信息，下一9:30开盘成交。现有event_clock_account_v1.py将决策记在前收盘原点，若复用必须另存真实decision_time并核对每条available_at≤decision_time、与执行日同日09点，不能声称海外数据在前一中国收盘已知，也不能用本次即将发生的开盘定目标。缺失保留、入场出场、等于阈值、旧意图、未成交和终点开盘都先中文冻结，只有一个候选；必要时间/缺失和成交测试通过后再运行。没有借预测显著性阻止账户。

EPS、财报、股数、公募与慢来源补齐暂停，不扩大数据工程。继续只510300现金、两原历史窗口、20万元、242年化、两费用、分红、整手、T+1和方向涨跌停；不创建子任务，不生成GPT数值包或安全审计，不交易。目标未实现，继续有依据研究。
""", encoding="utf-8")
    record = {"round": 69, "study": result["study_id"], "title": "低开收复进入与日内转弱退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND69_COMPLETE_GAP_RECOVERY_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计69轮，343不同设置，356已评价来源版本，361登记含5旧未运行，1112主评价记录。",
        checks="第69轮10测试、3456独立金额事件、20指标12差额5646请求850含分红周期及850买入时点；已取得两项Cboe小文件且核对潜在时钟覆盖。",
        process_state_note="第69轮失败归因及中文交付完成；第70轮只完成VIX与VIX9D来源检查，尚未构建比值或登记。",
        next_work={"status": "GLOBAL_VOLATILITY_TERM_SOURCE_FILES_AVAILABLE_RULE_NOT_REGISTERED", "focus": "已取得VIX与九天指数，拟比较期限风险状态；先冻结09时信息和完整进出场规则", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_gap_recovery_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "GAP_RECOVERY_ROUND69_CHINESE_RESULTS", "rounds": [69], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "GAP_RECOVERY_RULES_AND_FULL_ACCOUNT_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt, "周期": groups, "买入时点": timing,
        "下一来源": {"文件数": len(source_probe["sources"]), "覆盖": source_coverage["coverage"]}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
