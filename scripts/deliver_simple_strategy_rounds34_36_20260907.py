"""交付三轮已完成结果、全部中文规则，并更新持续研究索引。"""
import json
from pathlib import Path
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import chinese_formula

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300简单策略改动_第34至36轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
STUDIES = [
    (34, "trend_learned_equal_blend", "趋势与学习退出各半", "COMPLETED_TARGET_NOT_MET_NO_IMPROVEMENT", "STOP_TWO_STATE_WEIGHT_TUNING", "TREND_LEARNED_HALF"),
    (35, "high_low_relation", "三个高低价关系因子", "COMPLETED_TARGET_NOT_MET_RECENT_PERIOD_FAILED", "STOP_HIGH_LOW_FACTOR_NEIGHBOR_TUNING", "H2_QUALITY"),
    (36, "learned_entry_opportunity", "从完整交易学习入场", "COMPLETED_TARGET_NOT_MET_IDENTICAL_ACCOUNT_PATHS", "REMOVE_ENTRY_MODEL_WITH_ZERO_ACCOUNT_INCREMENT", "CYCLE_ENTRY_RIDGE"),
]


def number(value):
    return "未计算" if value is None else f"{value:.3f}"


def percentage(value):
    return "未计算" if value is None else f"{value:.2%}"


def table(rows):
    lines = ["|费用|策略|净夏普|复合年化收益|最大回撤|买卖笔数|平均股票仓位|佣金及滑点合计（元）|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = ["基础" if row["cost"] == "BASE" else "压力", row["name"], number(row["net_sharpe"]),
                  percentage(row["annualized_return"]), percentage(row["max_drawdown"]), str(row["trade_count"]),
                  percentage(row["mean_exposure"]), f"{row['commission'] + row['slippage_cost']:.2f}"]
        lines.append("|" + "|".join(values) + "|")
    return lines + [""]


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed == set(range(1, 34)) or completed == set(range(1, 37)), "交付索引轮数出现未处理的新变化")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {n: json.loads((ROOT / f"reports/research/510300_{slug}_v1/result.json").read_text(encoding="utf-8"))
               for n, slug, title, status, decision, primary in STUDIES}
    records, copies = [], []
    r36 = ROOT / "reports/research/510300_learned_entry_opportunity_v1"
    identical = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            new = pd.read_parquet(r36 / period / cost / "CYCLE_ENTRY_RIDGE_ledger.parquet")
            old = pd.read_parquet(r36 / period / cost / "REARM_RIDGE_ledger.parquet")
            pd.testing.assert_frame_equal(new, old)
            identical.append({"period": period, "cost": cost, "rows": len(new), "all_saved_ledger_columns_identical": True})
    members = pd.read_csv(r36 / "training_memberships.csv")
    require((members.exit_index <= members.fit_index).all(), "入场模型使用了尚未完成的交易")
    coverage = results[36]["entry_model_coverage"]
    require(all(r["nonpositive_rejections"] == 0 for r in coverage), "入场规则实际产生了拒绝，原结论需重写")
    for n, slug, title, status, decision, primary in STUDIES:
        result = results[n]
        record = {"round": n, "study": result["study_id"], "title": title, "status": status,
                  "result": f"reports/research/510300_{slug}_v1/result.json",
                  "candidate_configurations": result["candidate_configurations"],
                  "evaluated_candidate_source_runs": result["candidate_configurations"],
                  "evaluation_accounts": result["evaluation_accounts"], "new_accounts_generated": result["new_accounts_generated"],
                  "reused_control_accounts": result["reused_control_accounts"], "earlier_diagnostic_accounts": result["earlier_diagnostic_accounts"],
                  "new_earlier_diagnostic_accounts": result["new_earlier_diagnostic_accounts"],
                  "new_model_fits": result.get("completed_fits", 0), "reference_accounts": result.get("reference_accounts", 0),
                  "primary_base": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "BASE"),
                  "primary_stress": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "STRESS"),
                  "post_selected_best_base": result["post_selected_best_base"]}
        records.append(record)
        outcome = {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}
        if n == 36:
            outcome.update(saved_identical_paths=identical, mature_memberships_checked=len(members),
                           no_account_increment=True, all_primary_entry_predictions_positive=True)
        write_json(ROOT / f"reports/research/510300_{slug}_v1/acceptance_outcome.json", outcome)
        for filename, label in [("metrics.csv", "全部主评价"), ("earlier_diagnostics.csv", "全部较早诊断"),
                                ("yearly_metrics.csv", "逐年结果"), ("era_metrics.csv", "分阶段结果")]:
            target = OUT / f"第{n}轮_{label}.csv"
            target.write_bytes((ROOT / f"reports/research/510300_{slug}_v1" / filename).read_bytes())
            copies.append(target)
    for source, filename in [(r36 / "entry_model_coverage.csv", "第36轮_实际入场检查.csv"),
                             (r36 / "training_receipts.csv", "第36轮_每月训练状态.csv"),
                             (r36 / "每月入场模型中文规则.md", "完整交易入场模型_每月实际中文规则.md")]:
        target = OUT / filename
        target.write_bytes(source.read_bytes())
        copies.append(target)
    saved_exits = json.loads((ROOT / "reports/research/510300_learned_cycle_exit_v1/saved_models.json").read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    exit_lines = ["# 本次复用的日内强弱线性退出模型：逐月实际中文规则", "",
                  "本文件只转写已有模型，没有重新训练。连续两个真实持仓收盘预测继续持有收益为负时，请求下一开盘退出。", ""]
    for item in saved_exits:
        exit_lines += [f"## {item['fit_origin']}", "", f"当时有{item['training_cycle_count']}个成熟周期、{item['training_rows']}条状态。", ""]
        exit_lines += chinese_formula(item["model"]) + [""] if item["status"] == "FIT_COMPLETE" else ["样本不足，未形成模型，沿用原价格与时间退出。", ""]
    exit_document = OUT / "已有线性退出模型_每月实际中文规则.md"
    exit_document.write_text("\n".join(exit_lines) + "\n", encoding="utf-8")
    copies.append(exit_document)

    content = ["# 510300 简单策略改动：第34至36轮完整说明", "",
               "数据终点：2026年8月14日开盘。文件生成：2026年9月7日。", "",
               "## 老板先看这一页", "",
               "目标仍是完整账户扣费净夏普至少1.2，目前没有实现。前瞻每股收益、研报日期、股数、估值、财报与公募资料的补齐已暂停。以下三轮只使用现有数据，直接检验交易规则。", "",
               "继续作为比较基线的是第32轮‘线性退出＋等待新机会’：主评价基础费用净夏普0.705、压力费用0.641；较早历史为0.748及0.725。主评价复合年化收益5.38%，最大回撤10.59%，实际24次完整买卖。它是已有历史中比较后保留的候选，尚无独立验证。", "",
               "这次新增五个设置，没有一个比该候选进一步改善。新增模型全部撤下，原规则及历史记录保留：", "",
               "|改动|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|决定|", "|---|---:|---:|---:|---|",
               "|趋势策略和已有学习退出状态各占一半|0.685|0.586|0.668|低于原候选，停止扫描组合权重。|",
               "|高低价斜率标准分|0.025|−0.014|0.309|近年不支持，停止细调。|",
               "|高低价斜率加拟合质量，预定主方案|−0.004|−0.036|0.557|近年不支持，停止细调。|",
               "|高低价斜率和拟合质量共同调整|0.034|0.015|0.915|较早好、近年差，不能只取较早历史。|",
               "|用完整已结束交易学习是否入场|0.705|0.641|0.748|与原账户逐日完全一致，移除额外模型。|", "",
               "第34轮组合在基础费用下把最大回撤从10.59%降到9.67%，但夏普和年化收益下降，买卖由48笔增加到145笔；压力费用下最大回撤反而为11.73%，原候选为11.01%。因此不能仅凭基础回撤较小就认为整体改善。", "",
               "第35轮三个新设置的主评价复合年化收益全部为负。夏普使用算术平均日收益，年化收益使用复合增长；在有波动时，夏普略为正与复合年化略为负可能同时出现。", "",
               "第36轮确实完成了学习：一个连续历史参考账户产生33笔自然结束的有效交易，完成85次月度拟合，另56个时点样本不足。首次拟合在2019年8月1日。主评价24次入场都有模型，预测均为正，区间约0.79%至17.32%，因此没有拒绝一次机会；较早9次机会均在成熟模型形成之前，全部沿用原规则。四条新账户与原候选的所有保存账户列完全一致。这是模型没有改变决策，不是新的策略成功。", "",
               "## 比较口径", "",
               "主评价：2020年1月2日至2026年8月14日开盘，保留全部1604个账户日。较早诊断：2015年1月5日至2019年12月31日开盘，保留全部1219日。两个时期各自从20万元开始，只持有510300和现金。较早历史已经被看过，不能称为全新的独立检验。", "",
               "基础费用为佣金万分之二、最低5元、滑点万分之五；压力费用为佣金万分之四、最低5元、滑点千分之一。现金和无风险收益均按零，年化242日。所有账户保留100份整手、0.001元价格间隔、方向性涨跌停、买入次日可卖、实际分红权益及未成交，统一在研究终点开盘请求退出。", "",
               "表中的买卖笔数把买入和卖出分别计数。平均仓位包含空仓日期；最大回撤以负号表示从先前净值高点的最大下降比例。结果全部扣除实际模拟费用，因子只用当时已知信息。", ""]
    for n, slug, title, status, decision, primary in STUDIES:
        content += [f"## 第{n}轮：{title}的全部历史结果", "", "### 主评价，2020年至2026年8月", ""]
        content += table(results[n]["all_metrics"])
        content += ["### 较早诊断，2015年至2019年", ""] + table(results[n]["earlier_diagnostics"])
    content += ["## 每项因子和进入、持有、退出、重新进入规则", "",
                "以下为每轮计算收益前登记的中文规则。‘预定主方案’是事先选择的地位，后续结果不会改写该地位。逐月变化的实际模型系数附在本文件夹两份中文模型说明中。", ""]
    for n, slug, title, status, decision, primary in STUDIES:
        protocol = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8")
        content += ["\n".join("#" + line if line.startswith("#") else line for line in protocol.splitlines()), ""]
    content += ["## 当前候选的证据边界", "",
               "保留第32轮候选不代表确认达到1.2。它的主评价相对原日内强弱策略少赚877.16元，主要把回撤和波动降低；相对‘仅等待新机会、没有学习退出’的对照，多赚33108.58元。此前已完成的二十日和六十日连续区块增量区间全部跨过零，不能确认稳定超额。", "",
               "較早时期只有两次实际退出包含学习条件，因此0.748不能理解为学习模型已经得到充分的跨时期验证。2020、2024及2025等单独年度出现较高夏普，也不能替代全期净夏普0.705。", "",
               "此前历史筛选最高1.118来自另一项只有三次近年完整交易的反弹设置，其较早历史亏损，仍未达标。本次不会把它重新定义为已验证策略。", "",
               "## 完成数量、必要核对和后续", "",
               "第34至36轮新增五个配置，合计26个主评价记录，其中10个新账户、16个原样对照；另26个较早诊断记录，其中10个新账户、16个原样对照。第36轮另有一个参考账户和85次模型拟合，独立计数。第35轮的3437次滚动高低价回归只是因子计算，不计为收益预测模型训练。", "",
               "第35轮六项因子测试通过，核对分红变换、已知回归结果、历史标准化、缺失和未来数据隔离；第36轮七项机制测试通过，核对成熟完整交易、分红权益、缺失状态、拒绝机会后的恢复、未成交和原账户一致性。运行时完整账户的日期、财富恒等式、次日可卖和终点结算均通过。本次又核对第36轮四条完整账户逐列相同及1804条训练成员记录的成熟时间。", "",
               "累计完成36轮、302个不同配置或范围、314个已评价来源版本、800个主评价记录。登记来源版本319，包含5个旧的未运行来源绑定。重复对照、较早诊断、参考账户和逐月拟合没有当成新的独立策略。", "",
               "后续停止本轮的组合权重、高低价参数、入场预测阈值和三因子调整。下一项拟检验现有成交量是否能使趋势均线提供额外信息：固定二十日与六十日成交量加权均线，同规则普通均线作为对照。先明确进入、退出与等待规则，再做有限账户比较；不采用新的补数工程。该方向目前仅完成名称检索，尚未登记或计算收益。", "",
               "EPS等补齐保持暂停。其他ETF是否计入目标的范围问题已提出但尚未收到回答，当前继续只研究510300和现金，不重复询问。持续研究保持运行，未把目标标记完成。本文件及普通结果文件用于直接查看，没有制作GPT审阅数值包。", "",
               "## 同目录的实际模型与原始结果", ""]
    for path in copies:
        content += [f"- [{path.name}]({path.name})"]
    content += ["", "上一份完整说明：[学习退出与重新进入：第31至33轮](../510300学习退出与重新进入_20260907/学习退出与重新进入_全部因子规则及历史表现.md)。", ""]
    document = OUT / "全部因子_进出场规则与历史表现.md"
    rendered = "\n".join(content).replace("較早", "较早")
    require("```" not in rendered and all(p.exists() for p in copies), "中文说明或结果文件不完整")
    document.write_text(rendered, encoding="utf-8")
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS34_36_COMPLETE_SIMPLE_VOLUME_TREND_RESEARCH_NEXT", latest_completed_round=records[-1],
                 running_studies=[], goal_achieved=False,
                 process_state_note="第34至36轮均已完成，没有这三轮的运行进程；持续研究目标与既有自动任务保持活动。",
                 count_warning="完成36轮、302个不同配置或范围、314个已评价来源版本、800个主评价记录；登记来源版本319含5个旧未运行绑定。本次另26个较早记录和1个参考账户不计入800。",
                 next_work=["保留第32轮原REARM_RIDGE作为比较基线，0.705及较早0.748均非确认达标。",
                            "第34轮等权组合下降至0.685，停止继续扫描两个状态的权重。第35轮三种高低价关系主评价失败，不调窗口、阈值及方向。",
                            "第36轮85次三因子入场模型训练没有拒绝任何机会，四条账户与原候选完全相同，撤下额外模型，不提高入场阈值救回。",
                            "下一项拟用现有成交量加权二十日及六十日均线与相同普通均线对照，单次明确进出场，检验成交量对趋势的增量；尚未登记和运行。名称检索只有config/minute_data.yaml的一般字段命中，未发现同名策略登记，仍须核对旧策略等价性。",
                            "EPS及来源补齐保持暂停，其他ETF范围问题待回复且不重复询问；继续中文规则和普通结果文件。"],
                 checks="第35轮6项、第36轮7项必要测试通过；第34至36轮完整账户结算核对通过；第36轮4条账户逐列相同，1804条训练成员成熟边界通过。")
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [34, 35, 36]]
    delivery = {"created_at": now(), "type": "SIMPLE_STRATEGY_ROUNDS34_36_CHINESE_RESULTS", "rounds": [34, 35, 36],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 36 and index["evaluation_accounts_in_this_resumption"] == 800 and
            index["evaluated_configurations_in_this_resumption"] == 302 and index["evaluated_candidate_source_runs_including_corrected_replays"] == 314 and
            index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 319, "累计研究计数不符")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_ALL_ROUND_RESULTS", "rounds": [34, 35, 36],
               "new_configurations": 5, "main_records": 26, "new_main_accounts": 10, "reused_main_accounts": 16,
               "earlier_records": 26, "new_earlier_accounts": 10, "reference_accounts": 1, "new_model_fits": 85,
               "necessary_tests_passed": 13, "new_gpt_review_archive_created": False, "security_audit_performed": False,
               "goal_achieved": False, "files": [{"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"报告": str(document), "交付文件数": len(receipt["files"]), "中文报告字符": len(rendered),
                      "完成轮数": 36, "主评价记录": 800, "目标达到": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
