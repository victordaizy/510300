"""交付短期标签、自身表现过滤和固定互补组合三轮中文结果。"""
import json
import re
from pathlib import Path
import pandas as pd
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300学习目标与互补组合_第44至46轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
DIAG = ROOT / "reports/research/510300_learned_panic_saved_diagnostic_v1"
STUDIES = [
    (44, "short_horizon_cycle_exit", "学习五日内继续持有价值", "SHORT_HORIZON_RIDGE", "COMPLETED_NO_ROBUST_IMPROVEMENT",
     "主评价夏普微升但最大回撤扩大，较早收益和夏普更弱，未形成稳定改善；停止相邻预测期限和模型参数搜索。"),
    (45, "self_performance_entry", "原策略过去一年赚钱才进入", "POSITIVE_SELF_YEAR", "COMPLETED_TARGET_NOT_MET_NO_IMPROVEMENT",
     "两个时期都弱于原候选，历史表现过滤延迟或错过恢复行情，不采用；停止年度窗口和盈利门槛细调。"),
    (46, "panic_learned_equal_blend", "急跌回升和原学习退出各半", "PANIC_LEARNED_HALF", "COMPLETED_MAIN_PERIOD_LOCAL_IMPROVEMENT_EARLIER_WEAKER",
     "主评价风险收益改善，净夏普仍低于1.2；较早夏普低于原候选且急跌样本少，仅保留局部研究线索，不确认稳定超额，不扫描这两个信号的相邻权重。"),
]


def folder(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 44)), set(range(1, 47))], "研究索引存在其他轮次变动，停止覆盖")
    results = {n: json.loads((folder(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, title, key, status, decision in STUDIES}
    diagnostic = json.loads((DIAG / "result.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    records, copies, increments, deferrals, entry_changes = [], [], [], [], []
    for n, slug, title, key, status, decision in STUDIES:
        result = results[n]
        record = {"round": n, "study": result["study_id"], "title": title, "status": status,
                  "result": f"reports/research/510300_{slug}_v1/result.json", "candidate_configurations": result["candidate_configurations"],
                  "evaluated_candidate_source_runs": result["candidate_configurations"], "evaluation_accounts": result["evaluation_accounts"],
                  "new_accounts_generated": result["new_accounts_generated"], "reused_control_accounts": result["reused_control_accounts"],
                  "earlier_diagnostic_accounts": result["earlier_diagnostic_accounts"], "new_earlier_diagnostic_accounts": result["new_earlier_diagnostic_accounts"],
                  "new_model_fits": result.get("completed_fits", 0), "new_reference_accounts": result.get("new_reference_accounts", 0),
                  "primary_base": next(m for m in result["all_metrics"] if m["model"] == key and m["cost"] == "BASE"),
                  "primary_stress": next(m for m in result["all_metrics"] if m["model"] == key and m["cost"] == "STRESS"),
                  "post_selected_best_base": result["post_selected_best_base"]}
        records.append(record)
        write_json(folder(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
                   "goal_achieved": False, "position_impact": 0})
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                a = pd.read_parquet(folder(slug) / period / cost / f"{key}_ledger.parquet")
                b = pd.read_parquet(folder(slug) / period / cost / "REARM_RIDGE_ledger.parquet")
                require(pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)), "保存账户日历不一致")
                d = {"round": n, "period": period, "cost": cost, "final_equity_difference": float(a.equity.iloc[-1] - b.equity.iloc[-1]),
                     "price_pnl_difference": float(a.price_pnl.sum() - b.price_pnl.sum()), "dividend_difference": float(a.dividend_recognized.sum() - b.dividend_recognized.sum()),
                     "extra_commission_and_slippage": float(a.commission.sum() + a.slippage_cost.sum() - b.commission.sum() - b.slippage_cost.sum())}
                require(abs(d["final_equity_difference"] - d["price_pnl_difference"] - d["dividend_difference"] + d["extra_commission_and_slippage"]) < 1e-6,
                        "账户差额无法由价格、分红和费用核对")
                increments.append(d)
                if n == 45:
                    decisions = pd.read_parquet(folder(slug) / period / cost / f"{key}_decisions.parquet")
                    for row in decisions[decisions.entry_deferred_for_nonpositive_performance.eq(True)].to_dict("records"):
                        deferrals.append({"period": period, "cost": cost, "decision_date": row["origin"], "reference_year_net_return": row["reference_year_net_return"]})
                    current = pd.read_csv(folder(slug) / period / cost / f"{key}_cycles.csv")
                    old = pd.read_csv(ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / cost / "REARM_RIDGE_cycles.csv")
                    for label, source, other in [("原进入日期被推迟或略过", old, current), ("新出现的进入日期", current, old)]:
                        for row in source[~source.entry_date.isin(other.entry_date)].to_dict("records"):
                            entry_changes.append({"period": period, "cost": cost, "comparison": label,
                                                  **{c: row[c] for c in ["entry_date", "exit_date", "entry_quantity", "net_profit_cny"]}})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((folder(slug) / filename).read_bytes())
            copies.append(dest)
    extra = [(folder("short_horizon_cycle_exit") / "每月短期退出模型中文规则.md", "第44轮_每月实际模型中文规则.md"),
             (folder("short_horizon_cycle_exit") / "training_receipts.csv", "第44轮_全部训练时点.csv"),
             (folder("short_horizon_cycle_exit") / "model_coverage.csv", "第44轮_实际持仓模型覆盖.csv"),
             (folder("self_performance_entry") / "entry_gate_statistics.csv", "第45轮_全部进入检查统计.csv"),
             (folder("self_performance_entry") / "self_performance_flags.csv", "第45轮_完整历史表现因子.csv"),
             (folder("panic_learned_equal_blend") / "state_counts.csv", "第46轮_共同持仓状态.csv"),
             (DIAG / "all_increment_intervals.csv", "第46轮_全部区块差额区间.csv"),
             (DIAG / "saved_account_differences.csv", "第46轮_相对原满仓与半仓的差额.csv"),
             (ROOT / "deliverables/510300仓位与退出_第41至43轮_20260907/沿用原线性模型_每月实际中文规则.md", "沿用原模型_每月实际中文规则.md")]
    for original, name in extra:
        dest = OUT / name
        dest.write_bytes(original.read_bytes())
        copies.append(dest)
    for name, rows in [("三轮全部账户_价格分红费用差额.csv", increments), ("第45轮_所有暂缓进入日期.csv", deferrals),
                       ("第45轮_原进入与新进入的不同日期.csv", entry_changes)]:
        dest = OUT / name
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        copies.append(dest)
    inventory_path = ROOT / "reports/research/510300_existing_share_premium_inventory_20260907.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory.update(local_reviewed_at=now(), next_action="PAUSE_THIS_SOURCE_DIRECTION_NO_BACKFILL",
                     available_at_evidence_not_proved_by_merge_report=True,
                     reviewed_existing_contract="config/510300_episodic_alpha_library_v1.yaml",
                     reviewed_existing_quality_report="reports/data_quality/510300_etf_share_premium_level_full_v1.json",
                     conclusion="已有资料支持历史数值拼接，未证明逐日真实披露时刻。旧逐日来源契约也要求记录available_at。当前不以自填15时字段替代发布证据，不恢复份额或净值补齐队列。")
    write_json(inventory_path, inventory)
    dest = OUT / "已有份额折溢价资料_本次未启动策略.json"
    dest.write_bytes(inventory_path.read_bytes())
    copies.append(dest)
    content = ["# 510300：学习目标、失效过滤和互补组合", "", "第44至46轮研究说明，2026年9月7日。全部规则以中文呈现，附完整历史结果及每月实际模型。", "",
               "## 管理层先看这一页", "",
               "本轮找到一项主评价中的局部改善：急跌回升和原线性退出各占一半预算，完整扣费净夏普从原候选0.705升至0.944，最大回撤降至5.39%。但组合年化收益为4.01%，低于原候选5.38%；较早历史净夏普只有0.604，低于原候选0.748。夏普1.2目标仍未完成，也未确认稳定超额。", "",
               "|方案|主评价基础夏普|主评价压力夏普|基础复合年化|基础最大回撤|较早基础夏普|较早压力夏普|",
               "|---|---:|---:|---:|---:|---:|---:|",
               "|原线性退出及等待新机会|0.705|0.641|5.38%|10.59%|0.748|0.725|",
               "|第44轮：最多五日的继续持有标签|0.712|0.663|6.24%|15.98%|0.694|0.670|",
               "|第45轮：自身过去一年赚钱才进入|0.532|0.470|3.85%|11.49%|0.695|0.676|",
               "|第46轮：急跌回升与原学习退出各半|**0.944**|**0.879**|**4.01%**|**5.39%**|**0.604**|**0.572**|", "",
               "主评价统一为2020年1月2日至2026年8月14日开盘，1604日；较早诊断为2015年1月5日至2019年12月31日开盘，1219日。两个时期分别从20万元开始，只有510300与现金，现金及无风险收益为零，242日年化。表中最大回撤以损失幅度显示，后附原始指标表保留负号。", "",
               "组合相对买入持有的基础复合年化超额仅约0.39个百分点，较早约0.29个百分点；压力费用下分别约0.11和0.08个百分点。两段都有一点正的年化超额，但幅度很小，不能据此声称已经稳定实现超额。", "",
               "## 失败原因现在能说清楚哪些", "",
               "第一，把模型预测期限缩短，未能带来一致改善。原1461条参考持仓状态中1295条标签被缩短，仍是原34个自然结束周期；完成114次月度拟合，27个时点无足够成熟样本。每次训练周期、行数和标准化均与原模型一致，隔离了目标变化。主评价有21次完整持仓，比原24次更少，收益提高但回撤显著扩大；较早九次持仓中三次包含学习退出，收益变弱。不能把预测期限不一致直接认定为根本原因。", "",
               "第44轮基础费用主评价终点比原候选多15,649.48元，可拆为价格损益多6,297.00元、分红多8,083.00元、费用节省1,269.48元。较早反而少13,497.71元，其中价格少13,488.40元、费用增加9.31元。该方向不作统一升级，不继续调整相邻预测期限。", "",
               "第二，知道过去一年亏钱，不等于知道下一次机会也差。第45轮连续参考账户从2013年6月3日起共3210日，中间不重置，过滤账户空仓时参考仍持续运行，避免“停止交易后永远没有新表现”的循环。主评价检查27次进入资格，23次允许、4次暂缓；较早14次检查，7次允许、7次暂缓，两个时期没有资料不足的进入检查。", "",
               "主评价四个暂缓信号日期为2021年10月28日、2022年5月12日、2024年2月8日和2024年9月24日。以2024年9月为例，参考策略过去一年收益为−2.14%，于是原本9月25日的进入推迟到9月30日；原账户该次持仓至9月30日净赚33,605.62元，新账户改为9月30日至10月9日净赚15,058.03元。两者实际份额和其他前序交易也不同，不能把单笔差额完全当成只有进入日期变化的独立因果效果。", "",
               "较早过滤避开了原2018年10月9日的一笔约10,059.93元亏损，但也略过了原2019年5月31日开始的一笔约19,618.60元盈利，并把2019年2月末进入推迟到3月。整体主评价终点少26,244.49元，较早少14,436.88元。历史表现过滤在这些恢复行情中存在滞后，不继续调整该窗口和盈利门槛。", "",
               "第三，固定互补组合比这些过滤更有研究价值，但它没有解决跨时期稳定性。主评价中，两个信号均不持有1341个决策日、仅学习退出持有235日、仅急跌回升持有10日、两个同时持有18日；较早分别为880、299、40和零日。过去两个策略的重叠结构也不同，不能假设组合长期保持相同风险收益关系。", "",
               "急跌回升原单策略主评价只有三次完整交易：2020年2月5日至2月18日、2021年7月29日至8月11日、2025年4月9日至4月23日。较早有八次完整交易且整体亏损。组合主评价52笔买卖、较早32笔，并不意味着出现了52个或32个独立急跌机会。", "",
               "## 固定组合的不确定性检查", "",
               "对已保存收益做二十日及六十日循环区块抽样，每项两千次；比较组合与原满仓候选、原每次固定半仓候选的夏普差和年化算术收益差。两时期、两档费用、两种对照、两种区块，共16项比较。它们是在点估计已看过之后登记的诊断，不能消除多轮历史筛选偏差，也不是重新运行可交易账户。", "",
               f"全部16项中，夏普差区间有{diagnostic['sharpe_intervals_crossing_zero']}项跨零，年化算术收益差区间有{diagnostic['mean_intervals_crossing_zero']}项跨零。主评价各比较的夏普差区间为正，较早仍未清楚区分改善；因此不能把主评价区间结果扩展成跨时期或独立验证。", "",
               "|主评价基础费用的比较|区块长度|夏普差95%区间|年化算术收益差95%区间|",
               "|---|---:|---:|---:|"]
    for r in diagnostic["intervals"]:
        if r["period"] == "evaluation" and r["cost"] == "BASE":
            name = "相对原满仓候选" if r["control"] == "ORIGINAL_FULL" else "相对每次固定半仓"
            content.append(f"|{name}|{r['block_days']}日|{r['sharpe_difference_low']:.3f} 至 {r['sharpe_difference_high']:.3f}|{r['annual_arithmetic_return_difference_low']:.2%} 至 {r['annual_arithmetic_return_difference_high']:.2%}|")
    content += ["", "相对固定半仓的对照还包含份额估算和再平衡差异。固定半仓只在进入时确定份额，组合使用每日目标和十个百分点调整限制，因此不能把全部增量严格归因于新增急跌信号。", ""]
    for n, slug, title, key, status, decision in STUDIES:
        content += [f"## 第{n}轮：完整因子及进出场规则", ""]
        lines = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]
        content += ["#" + line if line.startswith("##") else line for line in lines]
        content += ["", f"### 第{n}轮所有主评价结果", ""] + table(results[n]["all_metrics"])
        content += [f"### 第{n}轮所有较早诊断结果", ""] + table(results[n]["earlier_diagnostics"])
        content += ["本轮结论：" + decision, ""]
    content += ["## 交付文件与直接证据", "", "全部逐年和分阶段结果、每月实际模型、过滤日期以及区块区间均附在同目录。", ""]
    for p in copies:
        content.append(f"- [{p.name}]({p.name})")
    content += ["", "完整账户直接来源：", ""]
    for n, slug, title, key, status, decision in STUDIES:
        content.append(f"- 第{n}轮：[直接结果](<{(folder(slug) / 'result.json').as_posix()}>)；[基础费用逐日账户](<{(folder(slug) / 'evaluation/BASE' / (key + '_ledger.parquet')).as_posix()}>)。")
    content += ["", "## 已完成核对与后续边界", "",
               "第44和45轮共10项必要测试通过，验证标签期限、股权登记日边界、周期成熟、未来数据隔离、完整收益窗口及进入许可。第46轮复用已有状态映射和事件账户，直接核对两源状态完整、三档预算、全日历、实际财富恒等式和结算。三轮全部12项相对原候选的保存账户差额可核对；组合另完成16项循环区块比较，不生成新账户。", "",
               "本次3个新配置，22个主评价记录，其中6条新账户和16条复用对照；另22个较早记录，同样6条新账户和16条复用。第44轮新增114次拟合、27个更新时点无模型；第45轮新增一条3210日连续参考账户。累计46轮、314个不同配置或范围、326个已评价来源版本、880个主评价记录；登记来源版本331含5个旧未运行绑定。重复费用对照和较早诊断不算新的独立策略。", "",
               "原第32轮候选继续作为比较基线；第46轮作为主评价局部改善线索单独保存，不宣称替代它成为已验证策略。互补机会的组合值得继续研究，但下一项须有新的明确机制或不同来源的信号，不能继续按本轮结果搜索这两个信号的相邻权重、阈值或有利年份。优先复用已有可用数据，先核对是否已覆盖相同组合，再固定少量规则计算完整账户。", "",
               "本次顺带核对已有基金份额和折溢价文件，共3430日，2012年7月至2026年8月，数值没有重复或缺失。但其中自填当日15时的字段不能证明真实披露时间，既有来源契约也要求逐日可用时刻证据。因此未启动新的份额策略或数据补齐队列，也没有把这份盘后资料提前用来交易。", "",
               "EPS及研报日期、股数、估值、财报和公募来源补齐继续暂停。不制作GPT数值包。目标仍未完成，研究保持活动，其他ETF范围问题仍待用户明确，当前只使用510300与现金。", ""]
    document = OUT / "学习目标与互补组合_全部因子规则和历史表现.md"
    text = "\n".join(content)
    require("```" not in text, "完整中文规则出现代码块")
    document.write_text(text, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "交付文件链接缺失：" + target)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    m46 = results[46]
    local_candidate = {"study": m46["study_id"], "model": "PANIC_LEARNED_HALF", "status": "MAIN_PERIOD_LOCAL_IMPROVEMENT_EARLIER_WEAKER_NOT_VALIDATED",
                       "source_result": "reports/research/510300_panic_learned_equal_blend_v1/result.json",
                       "base_main_sharpe": records[-1]["primary_base"]["net_sharpe"], "stress_main_sharpe": records[-1]["primary_stress"]["net_sharpe"],
                       "base_earlier_sharpe": next(m["net_sharpe"] for m in m46["earlier_diagnostics"] if m["model"] == "PANIC_LEARNED_HALF" and m["cost"] == "BASE"),
                       "stress_earlier_sharpe": next(m["net_sharpe"] for m in m46["earlier_diagnostics"] if m["model"] == "PANIC_LEARNED_HALF" and m["cost"] == "STRESS"),
                       "distinct_panic_cycles_main": 3, "distinct_panic_cycles_earlier": 8,
                       "saved_uncertainty": str((DIAG / "result.json").relative_to(ROOT)), "goal_achieved": False}
    index.update(updated_at=now(), status="ROUNDS44_46_COMPLETE_COMPLEMENTARY_SIGNAL_RESEARCH_CONTINUES", latest_completed_round=records[-1], running_studies=[], goal_achieved=False,
                 process_state_note="第44至46轮与保存收益诊断均已完成，全部运行进程已退出。无下一项已登记研究。",
                 count_warning="累计46轮、314不同配置或范围、326已评价来源版本、880主评价记录；登记331含5旧未运行绑定。较早、拟合、参考另计。",
                 additional_main_period_candidate=local_candidate,
                 next_work=["原第32轮作为比较基线保留，第46轮固定急跌回升与原学习退出各半为局部线索，主评价0.944391、压力0.879003，较早0.604421和0.572113，仍未达到1.2。",
                            "组合主评价只有三个不同急跌持仓周期，较早八个且急跌单策略亏损。16项区块比较中主评价夏普差区间为正、较早区间跨零；不能当独立或跨时期稳定验证。",
                            "停止第44轮相邻标签期限、第45轮自身历史表现窗口和门槛，以及第46轮两个信号的相邻权重搜索。",
                            "后续优先查既有策略结果中是否有当前组合未覆盖、机制不同且有更多独立机会的信号。新组合须先说明来源、样本选择和完整进出规则，不因当前三次急跌盈利反推阈值；无新机制不重复运行。也可转向明确的新机制，不能只调原候选的参数。",
                            "已有份额折溢价3430日的数值合并不证明历史披露时刻，旧契约要求available_at；本次未启动新策略，保持相关补齐暂停。",
                            "EPS及所有原来源补齐暂停，其他ETF目标问题待回复，不重复询问；不做GPT数值包、不创建子任务。"],
                 checks="第44及45轮10项必要测试通过；第46轮复用事件账户并核对全日历、状态和结算；12项保存账户差额及16项循环区块比较完成。",
                 latest_saved_complementary_diagnostic=str((DIAG / "result.json").relative_to(ROOT)))
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [44, 45, 46]]
    delivery = {"created_at": now(), "type": "LABEL_PERFORMANCE_COMPLEMENT_ROUNDS44_46_CHINESE_RESULTS", "rounds": [44, 45, 46],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 46 and index["evaluation_accounts_in_this_resumption"] == 880 and index["evaluated_configurations_in_this_resumption"] == 314 and
            index["evaluated_candidate_source_runs_including_corrected_replays"] == 326 and index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 331,
            "累计研究计数不符")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_SAVED_DIAGNOSTICS", "rounds": [44, 45, 46],
               "new_configurations": 3, "main_records": 22, "new_main_accounts": 6, "reused_main_accounts": 16,
               "earlier_records": 22, "new_earlier_accounts": 6, "reused_earlier_accounts": 16, "new_model_fits": 114,
               "no_view_fit_origins": 27, "new_reference_accounts": 1, "necessary_tests_passed": 10,
               "interval_records": 16, "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "交付文件": len(receipt["files"]), "文档字符": len(text), "累计完成轮数": 46,
                      "主评价记录": 880, "新局部线索夏普": local_candidate["base_main_sharpe"], "目标达到": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
