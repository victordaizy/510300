"""将单一入场一致性检验加入同一中文报告，并关闭第33轮记录。"""
from pathlib import Path
import json
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300学习退出与重新进入_20260907"
SOURCE = ROOT / "reports/research/510300_consistent_entry_exit_v1"
DOCUMENT = OUT / "学习退出与重新进入_全部因子规则及历史表现.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"


def main():
    result = json.loads((SOURCE / "result.json").read_text(encoding="utf-8"))
    content = DOCUMENT.read_text(encoding="utf-8")
    old = "本次第31、32轮共9个新增设置、34个主评价记录，其中18个新账户、16个原样对照；另34个较早诊断记录，其中18个新账户、16个复用对照。"
    new = "本次第31至33轮共10个新增设置、40个主评价记录，其中20个新账户、20个原样对照；另40个较早诊断记录，其中20个新账户、20个复用对照。"
    content = content.replace(old, new)
    section = ["## 第33轮：入场与退出使用同一模型的检验结果", "",
               "只增加一项条件：原入场信号成立时，使用当时的收盘价和成本估计构造假设持仓状态；已有线性模型预测继续持有收益为负，就暂缓买入。模型、训练和原退出规则均不变。", "",
               "主评价34次入场检查中，有10次预测为负而暂缓，实际仍完成24次买入，主要改变了部分进入日期。基础夏普从0.705降到0.677，压力夏普从0.641降到0.613，基础年化收益从5.38%降到5.14%。较早时期9次检查中没有一次负预测被拒绝，其中5次没有成熟模型而沿用原规则，所以结果与原候选完全一致。", "",
               "该额外条件没有改善，停止继续调整这种假设持仓状态的入场过滤。当前保留的仍是第32轮原线性退出＋等待新机会，不增加本轮条件。", "",
               "|策略|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|主评价基础年化|主评价基础最大回撤|",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["all_metrics"]:
        if row["cost"] != "BASE":
            continue
        values = []
        for earlier, cost in [(False, "BASE"), (False, "STRESS"), (True, "BASE"), (True, "STRESS")]:
            m = next(x for x in result["earlier_diagnostics" if earlier else "all_metrics"] if x["model"] == row["model"] and x["cost"] == cost)
            values.append("未计算" if m["net_sharpe"] is None else f"{m['net_sharpe']:.3f}")
        section.append("|" + row["name"] + "|" + "|".join(values + [f"{row['annualized_return']:.2%}", f"{row['max_drawdown']:.2%}"]) + "|")
    section += ["", "第33轮新增4项必要测试通过，核对负预测暂缓、零预测保留、未来模型不可提前使用、当前成本构造以及暂缓不消耗入场许可。", ""]
    if "## 第33轮：入场与退出使用同一模型的检验结果" not in content:
        content = content.replace("## 保留候选的改善来自哪里", "\n".join(section) + "\n## 保留候选的改善来自哪里", 1)
    start = content.index("## 下一步只保留一个具体方向")
    end = content.index("## 全部中文因子及完整进出场规则", start)
    next_section = """## 下一步只保留一个具体方向

同一模型检查入场的第33轮已完成，没有改善。保留第32轮线性退出和等待新机会作为比较基线，停止两层树及入场假设状态过滤的细调。

下一步拟只检验一个固定等权组合：既有趋势与震荡反弹状态占一半，新的日内强弱线性退出及等待新机会状态占一半。两个状态都只能由当时已有信息形成，组合须重新计算整手、费用、分红和未成交，不能平均两条账户收益替代。先做这一项，不重新扫描权重和风险阈值。该组合尚未登记或运行，没有新的结果。

EPS、研报、估值和公募来源补齐继续暂停。其他ETF是否可计入目标仍待用户回复，当前资产范围保持510300与现金。完整账户净夏普1.2和稳定超额仍未确认。

"""
    content = content[:start] + next_section + content[end:]
    if "## 第33轮：入场前检查已有退出模型的判断" not in content:
        rules = "\n".join("#" + line if line.startswith("#") else line for line in (ROOT / "docs/510300_CONSISTENT_ENTRY_EXIT_V1.md").read_text(encoding="utf-8").splitlines())
        rules = rules.replace("](510300_LEARNED_CYCLE_EXIT_V1.md)", "](#全部中文因子及完整进出场规则)")
        content = content.replace("## 每月实际模型和必要验证", rules + "\n\n## 每月实际模型和必要验证", 1)
    copied = []
    for source, name in [("metrics.csv", "入场一致_全部主评价.csv"), ("earlier_diagnostics.csv", "入场一致_全部较早诊断.csv"),
                         ("yearly_metrics.csv", "入场一致_逐年结果.csv"), ("era_metrics.csv", "入场一致_分阶段结果.csv"),
                         ("entry_check_statistics.csv", "入场一致_暂缓及实际买入次数.csv")]:
        (OUT / name).write_bytes((SOURCE / source).read_bytes())
        copied.append(name)
        if f"]({name})" not in content:
            content += f"\n- [{name}]({name})\n"
    DOCUMENT.write_text(content, encoding="utf-8")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    record = {"round": 33, "study": result["study_id"], "title": "入场前检查同一线性退出模型", "status": "COMPLETED_TARGET_NOT_MET_NO_IMPROVEMENT",
              "result": "reports/research/510300_consistent_entry_exit_v1/result.json", "candidate_configurations": 1,
              "evaluated_candidate_source_runs": 1, "evaluation_accounts": 6, "new_accounts_generated": 2, "reused_control_accounts": 4,
              "earlier_diagnostic_accounts": 6, "new_earlier_diagnostic_accounts": 2, "new_model_fits": 0,
              "post_selected_best_base": result["post_selected_best_base"],
              "primary_base": next(x for x in result["primary"] if x["cost"] == "BASE"),
              "primary_stress": next(x for x in result["primary"] if x["cost"] == "STRESS")}
    if not any(r["round"] == 33 for r in index["completed_rounds"]):
        index["completed_rounds"].append(record)
        index["evaluation_accounts_in_this_resumption"] += 6
        for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                    "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
            index[key] += 1
    index.update({"updated_at": now(), "status": "ROUNDS31_33_COMPLETE_FIXED_BLEND_RESEARCH_CONTINUES", "latest_completed_round": record,
                  "running_studies": [], "goal_achieved": False,
                  "process_state_note": "第31至33轮及保存账户诊断全部完成，当前没有这些研究的运行进程。",
                  "count_warning": "完成33轮，297个不同配置或范围、309个已评价来源版本、774个主评价记录。登记来源版本314，含5个旧未运行绑定。第31至33轮另40个较早诊断和3个参考账户不计入774。",
                  "next_work": ["第33轮入场一致性条件失败，不改变假设持仓状态、模型符号或阈值继续修补。",
                                "保留第32轮REARM_RIDGE作为候选，预定下一轮只做一个固定等权组合：原S1趋势与震荡状态、新REARM_RIDGE状态各半，重新模拟完整账户，不扫描权重。",
                                "候选0.705与较早0.748尚不足以确认目标；较早仅两次学习退出，增量区间跨零。",
                                "EPS及来源补齐继续暂停，资产范围问题待回复，不重复询问。"],
                  "checks": "第31至33轮18项必要机制测试通过，全部新账户完成日期、财富、T+1及进入退出状态核对；两种连续区块增量诊断已完成。"})
    index["partial_rounds"] = [x for x in index.get("partial_rounds", []) if x["round"] != 33]
    for delivery in index["deliveries"]:
        if delivery.get("type") == "LEARNED_CYCLE_EXIT_AND_REARM_CHINESE_RESULTS":
            delivery.update(rounds=[31, 32, 33], updated_at=now())
    require(index["evaluated_configurations_in_this_resumption"] == 297 and index["evaluation_accounts_in_this_resumption"] == 774, "第33轮统计计数不符")
    write_json(INDEX, index)
    write_json(SOURCE / "acceptance_outcome.json", {"recorded_at": now(), "status": "COMPLETED_TARGET_NOT_MET_NO_IMPROVEMENT",
               "goal_achieved": False, "decision": "STOP_HYPOTHETICAL_ENTRY_FILTER_TUNING", "position_impact": 0})
    receipt_path = OUT / "delivery_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(updated_at=now(), new_candidate_configurations=10, main_evaluation_records=40, new_main_accounts=20,
                   earlier_diagnostic_records=40, new_earlier_accounts=20, necessary_tests_passed=18)
    paths = [OUT / f["path"] for f in receipt["files"]]
    paths += [OUT / name for name in copied if OUT / name not in paths]
    receipt["files"] = [{"path": p.name, "sha256": digest(p)} for p in paths]
    write_json(receipt_path, receipt)
    print(json.dumps({"中文报告已合并第31至33轮": str(DOCUMENT), "完成轮数": 33, "主评价记录": 774}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
