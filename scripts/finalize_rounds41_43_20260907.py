"""将仓位、连续下跌及下午退出三轮结果整理成完整中文普通交付。"""
import json
import re
from pathlib import Path
import pandas as pd
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300仓位与退出_第41至43轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
DIAG = ROOT / "reports/research/510300_sizing_exit_saved_diagnostic_v1"
STUDIES = [
    (41, "entry_volatility_sizing", "进入时按波动率或固定半仓控制预算", "ENTRY_VOL10",
     "波动率设置没有提高夏普；半仓只是大致同比缩小收益和风险，夏普微差不作改善证据，两者主评价年化低于买入持有。停止初始仓位和风险目标细调。"),
    (42, "price_streak_exit", "连续两个收盘下跌退出", "PRICE_STREAK_EXIT",
     "主评价弱于原候选且较早亏损，不采用；停止连续下跌天数和跌幅阈值搜索。"),
    (43, "afternoon_learned_exit", "原退出模型下午提前确认", "AFTERNOON_EXIT",
     "主评价基础与压力均弱于原候选，较早全为原规则回退而非验证；停止该时刻、容量和确认阈值细调。"),
]


def source(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 41)), set(range(1, 44))], "研究索引已有其他新轮次，停止覆盖")
    OUT.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    results = {n: json.loads((source(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, title, primary, decision in STUDIES}
    copies, records, increments, pm_cycles = [], [], [], []
    for n, slug, title, primary, decision in STUDIES:
        result = results[n]
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                folder = source(slug) / period / cost
                baseline = pd.read_parquet(folder / "REARM_RIDGE_ledger.parquet")
                keys = ["ENTRY_VOL10", "ENTRY_HALF"] if n == 41 else [primary]
                for key in keys:
                    ledger = pd.read_parquet(folder / f"{key}_ledger.parquet")
                    require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(baseline.date)), "保存账户比较缺少日期")
                    delta = {"round": n, "period": period, "cost": cost, "model": key,
                             "final_equity_difference": float(ledger.equity.iloc[-1] - baseline.equity.iloc[-1]),
                             "price_pnl_difference": float(ledger.price_pnl.sum() - baseline.price_pnl.sum()),
                             "dividend_difference": float(ledger.dividend_recognized.sum() - baseline.dividend_recognized.sum()),
                             "extra_commission_and_slippage": float(ledger.commission.sum() + ledger.slippage_cost.sum() - baseline.commission.sum() - baseline.slippage_cost.sum())}
                    require(abs(delta["final_equity_difference"] - delta["price_pnl_difference"] - delta["dividend_difference"] + delta["extra_commission_and_slippage"]) < 1e-6,
                            "保存账户价格、分红与费用的差额无法核对")
                    increments.append(delta)
                    if n == 43 and period == "earlier_diagnostic":
                        pd.testing.assert_frame_equal(ledger, baseline)
                if n == 43 and period == "evaluation":
                    current_cycles = pd.read_csv(folder / f"{primary}_cycles.csv")
                    old_cycles = pd.read_csv(ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / cost / "REARM_RIDGE_cycles.csv")
                    require(current_cycles.entry_date.to_list() == old_cycles.entry_date.to_list(), "下午退出与原候选的进入日期不一致，不能按相同进入匹配")
                    for a, b in zip(current_cycles.to_dict("records"), old_cycles.to_dict("records")):
                        pm_cycles.append({"cost": cost, "cycle_id": a["cycle_id"], "entry_date": a["entry_date"],
                                          "new_exit_date": a["exit_date"], "old_exit_date": b["exit_date"], "new_exit_origin": a["exit_origin"],
                                          "new_exit_reason": a["exit_reasons"], "new_entry_quantity": a["entry_quantity"], "old_entry_quantity": b["entry_quantity"],
                                          "new_cycle_profit": a["net_profit_cny"], "old_cycle_profit": b["net_profit_cny"]})
        record = {"round": n, "study": result["study_id"], "title": title, "status": "COMPLETED_TARGET_NOT_MET_NO_IMPROVEMENT",
                  "result": f"reports/research/510300_{slug}_v1/result.json", "candidate_configurations": result["candidate_configurations"],
                  "evaluated_candidate_source_runs": result["candidate_configurations"], "evaluation_accounts": result["evaluation_accounts"],
                  "new_accounts_generated": result["new_accounts_generated"], "reused_control_accounts": result["reused_control_accounts"],
                  "earlier_diagnostic_accounts": result["earlier_diagnostic_accounts"], "new_earlier_diagnostic_accounts": result["new_earlier_diagnostic_accounts"],
                  "new_model_fits": 0, "new_reference_accounts": 0,
                  "primary_base": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "BASE"),
                  "primary_stress": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "STRESS"),
                  "post_selected_best_base": result["post_selected_best_base"]}
        records.append(record)
        write_json(source(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": record["status"], "decision": decision,
                   "goal_achieved": False, "position_impact": 0, "earlier_minute_increment_tested": False if n == 43 else None})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((source(slug) / filename).read_bytes())
            copies.append(dest)
    for n, slug, filename in [(41, "entry_volatility_sizing", "entry_sizing_statistics.csv"), (42, "price_streak_exit", "price_exit_statistics.csv"),
                              (43, "afternoon_learned_exit", "existing_minute_inventory.json")]:
        dest = OUT / f"第{n}轮_{filename}"
        dest.write_bytes((source(slug) / filename).read_bytes())
        copies.append(dest)
    for cost in ["BASE", "STRESS"]:
        dest = OUT / f"第43轮_{cost}_全部下午判断与未成交.csv"
        dest.write_bytes((source("afternoon_learned_exit") / "evaluation" / cost / "afternoon_decisions.csv").read_bytes())
        copies.append(dest)
    pd.DataFrame(increments).to_csv(DIAG / "saved_increment_decomposition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pm_cycles).to_csv(DIAG / "afternoon_matched_cycles.csv", index=False, encoding="utf-8-sig")
    for old, name in [(DIAG / "saved_increment_decomposition.csv", "全部保存账户_价格分红费用差额.csv"),
                      (DIAG / "afternoon_matched_cycles.csv", "下午退出与原候选_按相同进入日期比较.csv"),
                      (ROOT / "deliverables/510300简单策略改动_第34至36轮_20260907/已有线性退出模型_每月实际中文规则.md", "沿用原线性模型_每月实际中文规则.md")]:
        dest = OUT / name
        dest.write_bytes(old.read_bytes())
        copies.append(dest)
    write_json(DIAG / "result.json", {"completed_at": now(), "status": "SAVED_ACCOUNT_DIFFERENCES_RECONCILED",
               "account_comparisons": len(increments), "matched_afternoon_cycle_records_including_two_costs": len(pm_cycles),
               "new_accounts_generated": 0, "new_model_fits": 0, "goal_achieved": False})
    p41, p42, p43 = results[41], results[42], results[43]
    content = ["# 510300：仓位和退出的三轮改动", "", "面向管理层的简明结论，以及全部因子、进出场规则和历史表现。更新日期：2026年9月7日。", "",
               "## 先看结果", "",
               "完整扣费净夏普至少1.2的目标仍未完成。本次完成四个新设置：按波动率决定进入预算、固定半仓进入、连续两个收盘下跌退出、下午提前检查原退出模型。没有一项形成明显局部改善，因此不采用这些改动，不继续细调它们的仓位比例、天数或下午时刻。", "",
               "原第32轮线性退出候选继续作为比较基线，主评价基础夏普0.705、压力0.641，复合年化5.38%，最大回撤10.59%；较早基础夏普0.748。但它也未完成目标，较早仅9次完整持仓、其中只有2次包含学习退出，不能称为已验证的稳定超额策略。保留它是事后比较选择，不改写第32轮预定主方案树模型失败的事实。", "",
               "|方案|主评价基础净夏普|主评价压力净夏普|基础复合年化|基础最大回撤|较早基础净夏普|结论|",
               "|---|---:|---:|---:|---:|---:|---|",
               "|原线性退出候选|0.705|0.641|5.38%|10.59%|0.748|比较基线，未达标|",
               "|进入时按波动率给预算|0.690|0.623|3.27%|6.02%|0.782|缩小风险，未提高主评价夏普|",
               "|每次固定半仓进入|0.707|0.644|2.79%|5.46%|0.752|收益风险约同比缩小，微小夏普差不当成改善|",
               "|连续两个收盘下跌退出|0.608|0.537|5.03%|14.49%|−0.044|主评价变弱，较早亏损|",
               "|下午提前确认原模型|0.632|0.566|4.63%|10.63%|0.748，纯回退|主评价变弱，较早没有分钟增量证据|", "",
               "主评价从2020年1月2日至2026年8月14日开盘，共1604日；较早诊断从2015年1月5日至2019年12月31日开盘，共1219日。各从20万元开始，只使用510300和现金。夏普按完整账户所有交易日日收益计算，包含空仓和费用；每年242日，现金及无风险收益假设均为零。", "",
               "主评价买入持有的基础复合年化为3.63%、净夏普0.288、最大回撤40.95%。两种缩减进入仓位的年化均低于买入持有，因此不能仅因回撤较小而宣称同时改善了超额收益。其他新退出规则虽仍有较高于买入持有的主评价年化，也没有达到目标或证明稳定。", "",
               "## 本次为什么没有改善", "",
               "第一，减少仓位主要改变承担多少风险。两种预算设置在两个时期、两档费用下，实际买卖日期和方向全部与原满仓候选相同，只改变份额。固定半仓的夏普0.707与原0.705几乎相同；预定波动率预算则为0.690。并未找到更好的进入或退出机会。", "",
               "波动率预算在主评价24次持仓中，中位预算比例约63.46%，最小27.96%，2次使用全额预算；较早9次持仓的中位比例约32.83%，最小21.41%，1次全额预算。主评价实际平均股票仓位为10.28%，因为大量日期空仓；预算比例并不等于全期间平均仓位。固定半仓的平均股票仓位为7.88%。", "",
               "第二，简单的连续下跌退出并没有更好地保留收益。主评价形成29次完整持仓、58笔买卖，较早13次、26笔买卖。入场时因为已连续下跌暂缓了主评价1次、较早3次原始机会；主评价27次普通退出和较早13次退出在触发时带有连续下跌条件。增加的交易未带来足够改善，较早年化为−0.36%。", "",
               "第三，把同一个模型的确认提前到下午，也没有提高它的判断质量。两档费用下都检查了252个实际持仓日，134次有可用下午模型判断，94次缺少当天分钟观察，24次为当天新买入而不能卖出。共形成17次下午退出请求，16次模拟成交、1次因分钟成交量容量不足而保留退出请求。没有把该次未成交从账户中删除。", "",
               "下午版本与原候选的24次进入日期全部相同，主评价基础费用终点少赚13,084.83元。其中价格损益少13,065.20元、分红少202.40元，佣金及滑点反而节省182.77元，三项能够核对。这次变差主要来自价格路径，并非额外费用。后续资金和整手份额也会随先前收益变化，因此这是完整账户差额，不能把全部差额归为一笔卖早交易的独立损失。", "",
               "上述比较都是已经看过的历史研究结果，不构成独立验证。没有根据某个较好年份换评价窗口，也没有把压力费用降低来接近1.2。", ""]
    for n, slug, title, primary, decision in STUDIES:
        content += [f"## 第{n}轮完整中文规则：{title}", ""]
        rule_file = ROOT / f"docs/510300_{slug.upper()}_V1.md"
        rule_lines = rule_file.read_text(encoding="utf-8").splitlines()
        for line in rule_lines[1:]:
            content.append("#" + line if line.startswith("##") else line)
        content += ["", f"### 第{n}轮所有主评价结果", ""] + table(results[n]["all_metrics"])
        content += [f"### 第{n}轮所有较早诊断结果", ""] + table(results[n]["earlier_diagnostics"])
        if n == 43:
            content += ["较早段候选两档账户逐列等于原收盘候选；这里的0.748不是下午方法在较早历史中获得的成绩。该时期没有任何分钟信号，全部按照预定回退处理。", ""]
        content += ["本轮结论：" + decision, ""]
    content += ["## 原逐月模型和全部结果文件", "",
                "第41和第43轮沿用同一组原月度线性退出模型，无新拟合。全部实际截距、因子均值、标准差和系数均在同目录的中文模型文件中，按月可查，不以代码代替规则。第42轮不使用退出预测模型。", ""]
    for path in copies:
        content.append(f"- [{path.name}]({path.name})")
    content += ["", "完整逐日账户、决策及逐笔成交仍保存在以下来源目录，未以汇总数字替代原始账户。", ""]
    for n, slug, title, primary, decision in STUDIES:
        folder = source(slug)
        content.append(f"- 第{n}轮：[直接结果文件](<{(folder / 'result.json').as_posix()}>)；[基础逐日账户](<{(folder / 'evaluation/BASE' / (primary + '_ledger.parquet')).as_posix()}>)。")
    content += ["", "## 已完成检查与下一步", "",
                "本次三轮共有18项必要测试通过：第41轮5项、第42轮4项、第43轮9项。下午事件重点验证了不能使用当天15时05分模型、一天不重复确认、今日最终收盘不能泄漏到下午、行情缺失完整回退、买入当天不能卖出、容量不足及价格缺失时保留退出、分红登记前后权益。全部新账户完成财富恒等式、实际份额和终点结算核对；16项保存账户差额已拆分为价格、分红和费用。", "",
                "本次新增4个配置、22个主评价记录，其中8条新账户、14条复用对照；另22个较早记录，其中6条新账户、16条复用，含2条下午候选纯回退。没有新模型或参考账户。累计完成43轮、311个不同配置或范围、323个已评价来源版本、858个主评价记录；登记来源版本328含5个旧未运行来源绑定。计数包含同一策略的费用对照，不能把858当成858个独立策略。", "",
                "不继续改变本次仓位、连续下跌退出和下午时刻。下一项先检查退出模型的学习目标：现有目标是“持有到原策略最后一次退出还能赚多少”，预测期限随周期而变，未直接回答固定短期继续持有的价值。拟在原八因子和参考状态上，检验固定五个开盘间隔、且不晚于原自然退出的继续持有标签，保留原进入和重新进入机制。先核对旧研究是否已有相同设置，固定一种标签和原模型容量，再计算完整账户，不重新补来源。此项尚未登记和运行，不能称已有改善。", "",
                "EPS、研报日期、股数、估值、财报及公募来源补齐继续暂停。持续研究保持活动；目标未完成。其他ETF是否可计入目标的问题仍待回复，当前只使用510300与现金。不制作GPT数值审阅包。", ""]
    document = OUT / "仓位与退出的改动_全部因子规则与历史表现.md"
    text = "\n".join(content)
    require("```" not in text, "中文规则出现了代码块")
    document.write_text(text, encoding="utf-8")
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS41_43_COMPLETE_SHORT_HORIZON_EXIT_LABEL_NEXT", latest_completed_round=records[-1],
                 running_studies=[], goal_achieved=False, process_state_note="第41至43轮均已实际完成，原运行进程已退出；下一项短期退出标签研究尚未登记。",
                 count_warning="累计43轮、311不同配置或范围、323已评价来源版本、858主评价记录；登记328含5旧未运行绑定。较早记录另计，不是独立策略数。",
                 next_work=["保留第32轮原线性退出及等待新机会候选作为比较基线，主评价0.705及较早0.748尚未达标。",
                            "第41轮按10%波动目标定进入预算主评价0.690，固定半仓0.707；同日期同方向，年化低于买入持有，停止风险目标及初始比例细调。",
                            "第42轮连续两个收盘下跌退出主评价0.608、较早-0.044，不采用，停止天数和跌幅阈值搜索。",
                            "第43轮下午原模型确认主评价0.631585、压力0.566415，两档均17请求16成交1容量受阻，基础少13084.83元，主要价格损益变弱，停止时刻和容量细调。",
                            "已有一分钟数据1211日可复用，第三方代理与时间标签语义边界仍保留。393个主评价日无分钟回退；较早全无分钟，原候选0.748不可称为下午机制验证。",
                            "下一项拟检查原学习退出可变期限标签，先查旧实现，若未覆盖则固定五个开盘间隔且不晚于原自然退出的一种新标签，复用八因子、原Ridge容量和参考状态，保留进入与再进入；尚未登记或运行。",
                            "EPS与来源补齐暂停，其他ETF问题待回复，不重复询问；不做GPT数值包。"],
                 checks="第41至43轮18项必要测试通过；全部新账户核对财富及结算；16项保存账户差額及48项按进入日期匹配的周期记录完成。",
                 latest_saved_sizing_exit_diagnostic=str((DIAG / "result.json").relative_to(ROOT)),
                 existing_minute_inventory=str((source("afternoon_learned_exit") / "existing_minute_inventory.json").relative_to(ROOT)))
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [41, 42, 43]]
    delivery = {"created_at": now(), "type": "SIZING_EXIT_ROUNDS41_43_CHINESE_RESULTS", "rounds": [41, 42, 43],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 43 and index["evaluation_accounts_in_this_resumption"] == 858 and index["evaluated_configurations_in_this_resumption"] == 311 and
            index["evaluated_candidate_source_runs_including_corrected_replays"] == 323 and index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 328,
            "累计研究计数不符")
    write_json(INDEX, index)
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "交付文档中的本地文件链接失效：" + target)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_SAVED_ACCOUNT_ATTRIBUTION", "rounds": [41, 42, 43],
               "new_configurations": 4, "main_records": 22, "new_main_accounts": 8, "reused_main_accounts": 14,
               "earlier_records": 22, "new_earlier_accounts": 6, "reused_earlier_accounts": 16, "earlier_afternoon_fallback_records": 2,
               "new_model_fits": 0, "new_reference_accounts": 0, "necessary_tests_passed": 18, "new_gpt_review_archive_created": False,
               "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "交付文件": len(receipt["files"]), "文档字符": len(text),
                      "已完成轮数": 43, "主评价记录": 858, "目标达到": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
