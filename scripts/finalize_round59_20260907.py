"""交付第59轮中文因子、进出场和全部结果，并记录下一项有限机制。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.entry_path_coverage_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P32
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, CN

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300退出训练覆盖_第59轮_20260907"
DOCUMENT = OUT / "退出训练覆盖_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_ENTRY_PATH_COVERAGE_NEXT_STEP_20260907.md"


def metric(result, model, period="evaluation", cost="BASE"):
    return next(m for m in result["all_metrics" if period == "evaluation" else "earlier_diagnostics"] if m["model"] == model and m["cost"] == cost)


def table(rows):
    lines = ["|方案|费用|净夏普|复合年化|最大回撤幅度|较买入持有年化差|平均股票仓位|成交笔数|佣金（元）|滑点（元）|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        lines.append(f"|{m['name']}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{-m['max_drawdown']:.2%}|{100*m['annualized_return_excess_vs_buy_hold']:.2f}个百分点|{m['mean_exposure']:.2%}|{m['trade_count']}|{m['commission']:.2f}|{m['slippage_cost']:.2f}|")
    return lines + [""]


def saved_cycle_comparison():
    records = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            a = pd.read_csv(RESEARCH / period / cost / f"{PRIMARY}_cycles.csv")
            b = pd.read_csv(P32 / period / cost / "REARM_RIDGE_cycles.csv")
            cols = ["entry_origin", "entry_date", "exit_date", "net_profit_cny", "entry_cost_cny", "exit_reasons"]
            joined = a[cols].merge(b[cols], on="entry_origin", how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
            joined["period"], joined["cost"] = period, cost
            joined["cycle_return_new"] = joined.net_profit_cny_new / joined.entry_cost_cny_new
            joined["cycle_return_old"] = joined.net_profit_cny_old / joined.entry_cost_cny_old
            joined["cycle_return_difference"] = joined.cycle_return_new - joined.cycle_return_old
            records.append(joined)
    compared = pd.concat(records, ignore_index=True)
    compared.to_csv(RESEARCH / "saved_entry_matched_cycle_comparison.csv", index=False, encoding="utf-8-sig")
    return compared


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 59)), set(range(1, 60))], "已有其他后续轮次，不覆盖索引")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "本脚本不能关闭出现1.2的候选")
    verification = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    reference = result["reference_summary"]
    comparisons = saved_cycle_comparison()
    decision = "扩大参考进入路径后，主评价两费用明显弱于原学习退出，较早两费用仅小幅改善；结束此项覆盖扩展，不调整组数、路径间隔、权重、岭惩罚、标签或确认天数继续搜索。"
    status = "COMPLETED_MAIN_PERIOD_DETERIORATION_NO_ACROSS_PERIOD_IMPROVEMENT"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
               "primary_base_sharpe": metric(result, PRIMARY)["net_sharpe"], "primary_stress_sharpe": metric(result, PRIMARY, cost="STRESS")["net_sharpe"],
               "earlier_base_sharpe": metric(result, PRIMARY, "earlier_diagnostic")["net_sharpe"],
               "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
    OUT.mkdir(parents=True, exist_ok=True)
    files = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "model_coverage.csv", "result.json",
             "reference_summary.json", "reference_paths.csv", "reference_episodes.csv", "training_receipts.csv", "每月模型中文规则.md",
             "saved_account_recomputation.csv", "saved_account_differences.csv", "saved_cycle_profit_summary.csv", "saved_model_coverage_comparison.csv",
             "saved_entry_matched_cycle_comparison.csv", "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json"]
    for filename in files:
        (OUT / filename).write_bytes((RESEARCH / filename).read_bytes())
    factors = pd.read_parquet(RESEARCH / "entry_factors.parquet")
    factors.rename(columns={"date": "收盘信息日期", "d60_factor": "六十日日内相对隔夜强弱", "raw_entry": "原进入条件成立", "raw_exit": "原价格退出条件成立",
                            "signal_available": "原信号信息完整"}).to_csv(OUT / "所有日期的入场因子.csv", index=False, encoding="utf-8-sig")
    ledger_cn = {"date": "日期", "cash": "现金", "shares": "持有份额", "dividend_receivable": "分红应收", "equity": "账户净值", "net_return": "当日净收益率",
                 "mark": "估值价格", "mark_clock": "估值时点", "commission": "佣金", "slippage_cost": "滑点成本", "filled_quantity": "实际成交份额",
                 "requested_quantity": "请求份额", "status": "成交状态", "price_pnl": "价格盈亏", "dividend_recognized": "确认分红", "dividend_paid": "到账分红",
                 "exposure": "股票仓位", "accounting_error": "记账差额", "execution_reasons": "执行原因"}
    decision_cn = {"origin": "收盘决策日", "execution_date": "计划执行日", "action": "动作", "requested_quantity": "请求份额",
                   "exit_reasons": "退出原因", "entry_rearmed": "已有再次进入资格", "learning_status": "学习信息状态",
                   "continuation_prediction": "继续持有优势预测", "learning_fit_origin": "所用模型训练日", "negative_confirmation_count": "连续负预测次数",
                   "learned_exit_requested": "学习退出请求", **dict(zip(FEATURES, CN))}
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost, cost_label in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            ledger.rename(columns=ledger_cn).to_csv(OUT / f"{label}_{cost_label}_完整账户.csv", index=False, encoding="utf-8-sig")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            decisions = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            decisions.rename(columns={**decision_cn, "d60_factor": "六十日日内相对隔夜强弱", "raw_entry": "原进入条件", "raw_exit": "原价格退出", "signal_available": "原信号完整"}).to_csv(
                OUT / f"{label}_{cost_label}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            cycles.rename(columns={"entry_origin": "进入决策日", "entry_date": "实际买入日", "exit_date": "实际卖出日", "entry_quantity": "买入份额",
                                    "entry_cost_cny": "含佣金买入成本", "net_profit_cny": "周期净利润", "holding_intervals": "持有交易区间数",
                                    "dividend_cny": "周期确认分红", "exit_reasons": "退出原因"}).to_csv(OUT / f"{label}_{cost_label}_全部持仓周期.csv", index=False, encoding="utf-8-sig")
    a, b = metric(result, PRIMARY), metric(result, PRIMARY, "earlier_diagnostic")
    old, early_old = metric(result, "REARM_RIDGE"), metric(result, "REARM_RIDGE", "earlier_diagnostic")
    delta = pd.read_csv(RESEARCH / "saved_account_differences.csv")
    d = delta[delta.period.eq("evaluation") & delta.cost.eq("BASE") & delta.control.eq("REARM_RIDGE")].iloc[0]
    old_coefficients = json.loads((ROOT / "reports/research/510300_learned_cycle_exit_v1/saved_models.json").read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    original_fits = sum(m["status"] == "FIT_COMPLETE" for m in old_coefficients)
    main_matched = comparisons[comparisons.period.eq("evaluation") & comparisons.cost.eq("BASE")]
    lines = ["# 510300退出训练覆盖：第59轮结果", "", "## 老板先看这一页", "",
             f"**这轮没有达到净夏普1.2，扩大退出训练覆盖的改动放弃。** 主评价净夏普由原方案的{old['net_sharpe']:.3f}降到{a['net_sharpe']:.3f}，最大回撤由{-old['max_drawdown']:.2%}扩大到{-a['max_drawdown']:.2%}；较早历史只由{early_old['net_sharpe']:.3f}微升到{b['net_sharpe']:.3f}。更多模拟路径没有形成更可靠的退出决策。", "",
             "本轮没有补每股盈利预测、估值或公募申购资料。只使用已经存在的510300日线，保持原进场条件、八个学习因子和线性模型复杂度，让模型看到不同日期买入后可能出现的持仓路径。", "",
             "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for key in [PRIMARY, "REARM_RIDGE", "REARM_NONE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
        m = metric(result, key)
        lines.append(f"|{m['name']}|{m['net_sharpe']:.3f}|{metric(result,key,cost='STRESS')['net_sharpe']:.3f}|{metric(result,key,'earlier_diagnostic')['net_sharpe']:.3f}|{metric(result,key,'earlier_diagnostic','STRESS')['net_sharpe']:.3f}|")
    lines += ["", "主评价：2020年1月2日至2026年8月14日开盘，共1604个账户日。较早历史：2015年1月5日至2019年12月31日开盘，共1219个账户日。每段20万元，包含全部空仓日及终点，不是只计算持仓日或盈利交易。", "",
              "原急跌回升及学习退出各半的方案只是在主评价有局部改善，较早历史较弱，不能称作已实现稳定高夏普。以前急跌回升单策略主评价1.118也未达到1.2，且只有3个不同持仓周期。当前没有通过独立验证的1.2策略。", "",
              "## 这轮究竟增加了多少信息", "",
              f"共登记{reference['attempted_hypothetical_paths']}个假想进入尝试，{reference['filled_hypothetical_paths']}个实际模拟买入成交，{reference['naturally_completed_paths']}个自然退出完成。1次首次买入因成交约束未完成；23条路径因资料终点或尚未自然完成而未成熟，其中包括1条下一开盘已经是资料终点、没有买入的尝试。三类处理结果1189、1、23合计1213，不把未成交或未成熟路径当零收益标签。", "",
              f"这些进入原点只对应{reference['reference_episodes']}个连续信号组；{reference['mature_label_bearing_episodes']}组完全成熟并含可用标签。共保存{reference['reference_state_rows']}条持仓状态，{reference['natural_label_rows']}条自然退出标签，其中{reference['group_mature_label_rows']}条所在整组已成熟。组内最早结束的路径不能提前训练，所以其余126条虽有自然退出标签，仍被整组成熟规则挡住。", "",
              f"每组总权重为1，组内路径和路径内状态分别等权。即使如此，这些都是同一历史价格上的重叠路径，不能声称获得1213次独立市场机会。旧模型有{original_fits}次实际月度拟合，本轮为{result['new_model_fits']}次；141次检查中有{result['no_view_fit_origins']}次样本不足，预测保留缺失。首次达到训练要求为2016年10月10日。", "",
              "每次成功训练采用10至20个已成熟组、227至749条路径、8615至32744条状态。八个因子全部保留；入场模式固定为1，没有额外信息。增加重复路径与增加独立行情不是同一回事。", "",
              "## 完整历史表现", "", "### 主评价", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早历史诊断", ""] + table(result["earlier_diagnostics"])
    lines += ["主评价新增方案24个完整持仓周期、13个盈利，较早9个周期、6个盈利；两档费用都为48笔和18笔实际成交。新模型触发学习退出的周期分别20个、2个。主评价254个持仓收盘全部有预测；较早299个持仓收盘中95个有预测，204个仍沿用原自然退出。这些日期没有删掉，也没有把缺失预测替换为零。", "",
              "基础费用每边佣金万分之2、最低5元，滑点万分之5；压力费用每边佣金万分之4、最低5元，滑点千分之1。年化242日，现金收益与无风险收益按零，100份整数单位、0.001元价格单位、方向涨跌停、次日可卖及分红权益均进入账户。", "",
              "## 失败来自哪里", "",
              f"基础主评价终值比原学习账户少{-d.terminal_nav_difference:,.2f}元。价格盈亏少{-d.price_pnl_difference:,.2f}元，分红少{-d.dividend_difference:,.2f}元；佣金和滑点合计反而少{-d.commission_difference-d.slippage_difference:,.2f}元。因此不能把明显退步归因于手续费增加。差额包含后续现金、份额和进入时点的连锁影响，不把全部差额说成某一次退出的因果损失。", "",
              f"主评价新旧方案各24个周期，有{int(main_matched['_merge'].eq('both').sum())}个相同进入原点，另外各3个原点不同。完整对照记录保留所有相同及不同原点。相同原点也出现了两类相反错误：", "",
              "|进入决策日|新方案卖出日|原方案卖出日|新周期净收益率|原周期净收益率|现象|", "|---|---|---|---:|---:|---|"]
    for origin, observation in [("2024-09-24", "退出更早，少拿到后续上涨"), ("2021-06-07", "退出更晚，承担更多下跌")]:
        row = main_matched[main_matched.entry_origin.eq(origin)].iloc[0]
        lines.append(f"|{origin}|{row.exit_date_new}|{row.exit_date_old}|{row.cycle_return_new:.2%}|{row.cycle_return_old:.2%}|{observation}|")
    lines += ["", "上表是对保存结果的事后解释，不据这些日期挑选交易、反转预测方向或调整退出阈值。周期净收益率以各周期含买入佣金的成本为分母，不能把比率差直接换算为整个账户的差额。较早历史的基础账户终值只增加535.70元，不能抵消主评价的大幅退步。", "",
              "## 全部因子与中文进出场规则", ""]
    protocol = (ROOT / "docs/510300_ENTRY_PATH_COVERAGE_V1.md").read_text(encoding="utf-8").split("## 每个因子的中文定义", 1)[1]
    lines += ["### 每个因子的中文定义" + protocol.replace("\n## ", "\n### "), "",
              "## 交付文件及核对范围", "",
              "- “每月模型中文规则.md”列出全部实际月度模型的截距、八因子均值、标准差和系数，也逐月说明未训练状态。",
              "- “所有日期的入场因子.csv”及四份“逐日全部因子和进出场.csv”保存实际用到的因子、预测、模型日期、请求和原因。",
              "- 四份“完整账户.csv”和四份“全部持仓周期.csv”保存两段两费用，包括空仓日和全部失败交易。",
              "- 英文文件名的结果表、分组表、训练记录和差额表保留程序原始字段，便于复算；上述策略文字没有用代码代替。", "",
              f"本轮26项必要测试通过；已复算{verification['recomputed_account_records']}条账户记录，核对{verification['reconciled_account_differences']}项新旧账户差额和{verification['reconciled_cycle_profit_sums']}项周期利润合计，检查{verification['verified_fit_group_weights']}个训练组的权重，重现{verification['recomputed_holding_predictions']}个实际持仓预测，预测差额为零。这些是实现与保存结果核对，不代表策略有效性通过。没有新增验证账户、重新训练或安全审计，没有制作GPT审阅数值包或压缩包。", "",
              "## 下一步", "",
              "下一项准备检验持仓后累积转弱是否能直接触发退出：入场前锁定历史均值和波动，持仓后只累积低于这一基准的偏离，达到事先固定的幅度后下一开盘退出。先核对与旧事件生命周期报警的区别，再冻结一个有限版本；不根据第59轮个别亏损日期设阈值。该下一项尚未登记、运行或产生收益结果。", "",
              "目标继续保持未完成；原学习退出及原组合保留作对照，第59轮失败结果保留。"]
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    next_text = """# 第59轮之后：持仓内累积转弱退出

第59轮已完成并失败：扩大进入路径后主评价基础/压力夏普0.341014/0.276071，较早0.751132/0.728159；原学习退出分别0.704868/0.641136和0.747698/0.724697。保留原模型，不继续调整参考组数、路径权重、岭惩罚或标签。普通中文交付与索引已有完整结果。

下一项先检查一种直接识别持仓转弱的简单机制，不依赖EPS、外部数据或新的收益预测拟合。候选思路：按实际入场前已知的60日含分红日对数收益，固定该持仓的基准均值和样本标准差；从买入后的下一个收盘开始，逐日累积“基准均值减实际日收益”的标准化偏离，每日扣除0.5，最低归零，达到5则下一开盘请求退出。0.5与5来自通用累积和设计惯例及项目旧报警定义，不由本轮收益选择；这不是对金融收益独立、正态或已具备超额的承诺。买入当天的隔夜收益不属于本账户，初步选择该天不计入检测；最终必须先把这一时点规则写清并测试。

参考原始方法说明：[美国国家标准与技术研究院累积和控制图](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)。该页给出单侧累计量，以及漂移容忍0.5、报警量4或5的一般惯例。本次只拟用5，不扫描4、5或其他邻近值。方法出处不证明市场收益有效。

已做限定检索：research与config的CUSUM、Page-Hinkley、累计和命中旧 research/episodic_alpha_library_v1.py:949 的 negative_page_hinkley_alarm。旧 docs/510300_EPISODIC_ALPHA_LIBRARY_V1_SPEC.md 第56行规定：激活时锁定40个已完成前向事件，用其均值和标准差，报警后禁止新进入，已有仓位按原退出规则走。拟检验机制直接用当前持仓逐日已观察到的收益请求退出，不重启旧事件策略的激活和生命周期研究，也不把旧权限或旧未达标证据转为认可。

还需要确定并冻结与原学习退出的结合方式。效率优先，最多一个新增设置，保留原R32、原自然退出和买入持有直接对照，不做两套阈值搜索。以现有下一开盘完整账户执行。若通过回调触发，必须保存“累积转弱退出”真实原因，不能滥用旧 learned_exit_requested 键而把报警记录成“连续两个负收益预测”。不改已冻结老引擎的历史版本；新增可测试的通用退出回调或受限包装即可。

必要测试先行：过去基准不读入场后收益、第一日处理、归零与累积、跨周期重置、缺失与零标准差保留无观点、受阻退出持续、费用分红和完整账户。冻结后只跑两段两档费用，失败即换机制，不围绕窗口和阈值继续扫。当前只有方向与同类研究检索完成，尚未登记第60轮或生成新账户。

继续仅用510300与现金，全部日历、20万元、242年化、两档原费用、零现金收益、次日可卖、100份单位及分红口径不变。EPS及慢来源暂停，不制作GPT数值包；总目标仍为未完成。
"""
    NEXT_NOTE.write_text(next_text, encoding="utf-8")
    record = {"round": 59, "study": result["study_id"], "title": "扩大进入路径的线性退出及等待新机会", "status": status,
              "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts",
              "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "new_reference_accounts_definition": result["new_reference_accounts_definition"],
              "primary_base": a, "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": a}
    if 59 not in completed:
        index["completed_rounds"].append(record)
        for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                    "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
            index[key] += 1
        index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    else:
        index["completed_rounds"][-1] = record
    index.update(updated_at=now(), status="ROUND59_COMPLETE_ENTRY_PATH_COVERAGE_FAILED", running_studies=[], goal_achieved=False,
                 latest_completed_round=record, count_warning="累计59轮、332个不同配置或范围、344个已评价来源版本、1014个主评价记录；登记349含5个旧未运行绑定。1213条新参考路径及119次月度拟合另计，不是独立策略。",
                 checks="第59轮26项必要测试，20条账户记录复算、8项账户差额、4项周期合计、2055项训练组权重及698条保存预测重现完成。",
                 process_state_note="第59轮参考路径、119次实际拟合及正式两段两费用账户均完成，进程正常退出；新策略未达1.2，目标仍未完成。",
                 next_work={"status": "DIRECTION_AND_PRIOR_METHOD_SEARCH_ONLY_NOT_FROZEN_NOT_RUNNING", "focus": "以持仓内累积转弱直接触发退出，最多一个新增设置",
                            "source": str(NEXT_NOTE.relative_to(ROOT)), "method_scope_note": "不同于旧40个完成前向事件的生命周期报警；必须先确定当前持仓的基准与正确退出原因"},
                 latest_saved_entry_path_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    delivery = {"created_at": now(), "type": "ENTRY_PATH_COVERAGE_ROUND59_CHINESE_RESULTS", "rounds": [59], "directory": str(OUT),
                "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False}
    if not any(x.get("type") == delivery["type"] for x in index["deliveries"]):
        index["deliveries"].append(delivery)
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "CHINESE_RULES_AND_SAVED_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
               "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
               "all_rules_written_in_chinese": True, "new_gpt_review_archive_created": False, "security_audit_performed": False,
               "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
