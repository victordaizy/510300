"""交付两轮学习退出及重新进入结果，保存完整中文模型规则。"""
from pathlib import Path
import json
import pandas as pd
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300学习退出与重新进入_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def metric(result, key, cost="BASE", early=False):
    return next(x for x in result["earlier_diagnostics" if early else "all_metrics"] if x["model"] == key and x["cost"] == cost)


def number(value):
    return "未计算" if value is None else f"{value:.3f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r31 = read("reports/research/510300_learned_cycle_exit_v1/result.json")
    r32 = read("reports/research/510300_rearmed_session_exit_v1/result.json")
    diagnostic = read("reports/research/510300_learned_rearm_saved_diagnostic_v1/result.json")
    copied = []
    for prefix, study in [("学习退出", "510300_learned_cycle_exit_v1"), ("重新进入", "510300_rearmed_session_exit_v1")]:
        for source, label in [("metrics.csv", "全部主评价"), ("earlier_diagnostics.csv", "全部较早诊断"),
                              ("yearly_metrics.csv", "逐年结果"), ("era_metrics.csv", "分阶段结果")]:
            dest = OUT / f"{prefix}_{label}.csv"
            dest.write_bytes((ROOT / "reports/research" / study / source).read_bytes())
            copied.append(dest.name)
    extra = [("510300_learned_cycle_exit_v1", "training_receipts.csv", "每月训练时间与成熟周期.csv"),
             ("510300_learned_cycle_exit_v1", "model_coverage.csv", "学习退出_模型覆盖.csv"),
             ("510300_learned_cycle_exit_v1", "saved_path_cost_and_reentry_diagnostic.csv", "学习退出_费用与重复进入诊断.csv"),
             ("510300_rearmed_session_exit_v1", "entry_state_statistics.csv", "重新进入_等待与实际交易统计.csv"),
             ("510300_learned_rearm_saved_diagnostic_v1", "account_difference_decomposition.csv", "保留候选_完整损益差额.csv"),
             ("510300_learned_rearm_saved_diagnostic_v1", "increment_intervals.csv", "保留候选_增量区间.csv"),
             ("510300_learned_rearm_saved_diagnostic_v1", "model_coverage.csv", "保留候选_模型覆盖.csv")]
    for study, source, label in extra:
        (OUT / label).write_bytes((ROOT / "reports/research" / study / source).read_bytes())
        copied.append(label)
    models = OUT / "全部历史月份的实际模型中文规则.md"
    models.write_bytes((ROOT / "reports/research/510300_learned_cycle_exit_v1/每月模型中文规则.md").read_bytes())
    lines = ["# 510300：学习退出与重新进入的完整结果", "", f"整理时间：{now()}。", "",
             "**目标仍未达到。当前值得继续检验的候选是“六十日日内强于隔夜＋线性学习退出＋卖出后等待新入场机会”。** 主评价基础夏普0.705、压力夏普0.641；较早诊断基础0.748、压力0.725。它改善了部分风险表现，仍未确认稳定超额，也未达到完整账户净夏普1.2。", "",
             "本次第31、32轮共9个新增设置、34个主评价记录，其中18个新账户、16个原样对照；另34个较早诊断记录，其中18个新账户、16个复用对照。第31轮建立三个历史参考账户、2458条持仓状态记录，完成706次月度模型训练；140个模型更新时点因成熟周期或样本不足没有模型。第32轮完全复用这些模型，没有新训练。", "",
             "## 这次发现的失败原因", "",
             "第31轮增加持仓状态学习后，六个设置的主评价夏普都低于各自原规则。提前卖出还会改变未来的重新进入日期，不能把退出优化视为只减少亏损的单向操作。趋势切换线性版本比原规则少约6.25万元，其中额外费用仅约976元，差距主要来自持仓时点和分红变化；不能仅归因于佣金。", "",
             "日内强弱信号有另一项具体冲突：旧入场条件持续有效时，学习模型先要求卖出，原入场规则等两天又允许买回。线性版本出现52次这样的再次买入，两层树出现70次；主评价交易次数分别从原规则36次升至144次和188次，额外费用约1.70万元及2.51万元。", "",
             "第32轮只增加“实际卖出后，旧入场条件至少在一个收盘消失过，才允许新机会再次进入”。三种设置都消除了这类重复买回。线性退出的主评价夏普从0.061提高到0.705、最大回撤从26.84%降至10.59%、买卖次数从144次降至48次。两层树虽然降至52次买卖，夏普仍只有0.164，因此不能认为减少交易本身足够解决问题。", "",
             "第32轮预定主方案是重复买回最多的两层树版本，它没有改善到可接受程度。当前保留的线性版本是本轮比较后的候选，不把它改称预定主方案，也不把局部改善作为已达标证明。", "",
             "## 固定评价口径", "",
             "主评价为2020年1月2日至2026年8月14日开盘，1604个账户日；较早诊断为2015年1月5日至2019年12月31日开盘，1219日。两段分别从20万元现金开始，较早历史也已研究过，不能称为独立验证。", "",
             "仅510300与现金，年化242日，现金及无风险收益零。保留100份整手、0.001元价格单位、方向性涨跌停、买入次日可卖、真实分红权益及到账、退出受阻和终点开盘退出。基础费用为单边佣金万分之二、最低5元、滑点万分之五；压力费用为佣金万分之四、最低5元、滑点千分之一。夏普由完整每日净收益计算；年化收益为复合年化收益。", ""]
    for title, result in [("第31轮：六个学习退出设置及原样对照", r31), ("第32轮：重新进入规则及原样对照", r32)]:
        lines += ["## " + title, "", "|策略|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|主评价基础年化收益|主评价基础最大回撤|主评价买卖次数|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in result["all_metrics"]:
            if row["cost"] != "BASE":
                continue
            key = row["model"]
            numbers = [number(metric(result, key, cost, early)["net_sharpe"]) for early, cost in [(False, "BASE"), (False, "STRESS"), (True, "BASE"), (True, "STRESS")]]
            lines.append("|" + row["name"] + "|" + "|".join(numbers + [f"{row['annualized_return']:.2%}", f"{row['max_drawdown']:.2%}", str(row["trade_count"])]) + "|")
        lines.append("")
    lines += ["## 保留候选的改善来自哪里", "",
              "相对原日内强弱策略，主评价基础年化收益从5.43%略降至5.38%，最终资产少877.16元；最大回撤从21.76%降至10.59%，夏普从0.465升至0.705。改善主要体现在承担更小的波动和回撤，主评价总盈利没有增加。", "",
              "相对“只等待新机会、不使用学习退出”的对照，线性学习版本在主评价多赚33108.58元，较早诊断多赚11822.12元。主评价差额由市场损益增加43897.50元、分红减少6349.20元、额外费用4439.72元构成。这是完整账户差额，包含后续仓位、现金、分红及交易路径变化，不是纯预测准确率的贡献。", "",
              "较早历史共有9次完整买卖，只有2次退出含学习条件。299个持仓收盘判断中，95次有成熟模型、204次没有模型而沿用原退出。主评价24次完整买卖，23次退出含学习条件，252个持仓判断均有模型。因此，两段较高夏普不能被解释为学习模型已经在两个时期获得同等验证。", "",
              "把每日账户差额按连续20日区间抽样，线性版本相对“仅等待新机会”的年化算术收益增量及95%区间如下。区间没有修正多次历史策略筛选，不是独立验证。", "",
              "|时期与费用|年化算术增量|95%区间下限|95%区间上限|", "|---|---:|---:|---:|"]
    for item in diagnostic["intervals"]:
        if item["block"] == 20 and item["comparator"] == "REARM_NONE":
            label = ("主评价" if item["period"] == "evaluation" else "较早诊断") + ("／基础费用" if item["cost"] == "BASE" else "／压力费用")
            lines.append(f"|{label}|{item['annualized_arithmetic_increment']:.2%}|{item['interval_95_low']:.2%}|{item['interval_95_high']:.2%}|")
    all_cross = all(x["interval_crosses_zero"] for x in diagnostic["intervals"])
    lines += ["", "20日和60日的全部比较区间均跨零，尚不能确认稳定的收益增量。" if all_cross else "全部20日和60日比较区间详见普通结果文件，不能仅凭正点估计确认稳定增量。", "",
              "## 下一步只保留一个具体方向", "",
              "先检查目前的入场和退出是否使用了相互矛盾的判断：原价格条件触发入场时，已有线性退出模型可能已经不看好继续持有。下一步拟只检验一个事先固定的入场与退出一致版本：使用当时已经训练好的线性模型判断是否暂缓进入，并保留等待新机会和原退出规则。入场前的持仓状态只能使用当时已知价格和预估成本构造，必须明确与真实成交后状态的差异。该下一步尚未登记或运行，没有新结果。", "",
              "两层树版本和第31轮其余失败设置停止附近参数细调；EPS、研报、估值和公募来源补齐继续暂停。其他ETF是否可计入目标仍待用户回复，当前资产范围保持510300与现金。", "",
              "## 全部中文因子及完整进出场规则", ""]
    for path in ["docs/510300_LEARNED_CYCLE_EXIT_V1.md", "docs/510300_REARMED_SESSION_EXIT_V1.md"]:
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                line = "#" + line
            line = line.replace("](510300_LEARNED_CYCLE_EXIT_V1.md)", "](#全部中文因子及完整进出场规则)")
            lines.append(line)
        lines.append("")
    lines += ["## 每月实际模型和必要验证", "",
              "[所有历史月份的实际线性系数、标准化参数与决策树中文条件](全部历史月份的实际模型中文规则.md)。此文件列出每次训练所用的成熟周期数量和最新结束日期，决策规则没有用代码替代。", "",
              "第31轮10项必要测试通过，覆盖未成熟周期排除、各周期等权、未来模型不可提前使用、分红登记权益、两边卖出费用、连续负预测、没有模型时沿用原规则，以及退出受阻后的持续请求。第32轮4项测试通过，覆盖持续旧信号不能重复进入、条件重现与原等待期同时满足、买入未成交不消耗许可和无学习模型的对照。全部新账户核对完整日期、财富恒等式、次日可卖和终点退出。", "",
              "没有新交易授权或真实券商成交。本次提供中文文档和普通结果文件，不制作GPT数值审阅包。", "", "## 普通结果文件", ""]
    lines += [f"- [{file}]({file})" for file in copied]
    document = OUT / "学习退出与重新进入_全部因子规则及历史表现.md"
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = read("reports/research/510300_sharpe_1_2_latest_research.json")
    for round_, result, folder, title in [(31, r31, "510300_learned_cycle_exit_v1", "根据成熟持仓周期学习退出"),
                                          (32, r32, "510300_rearmed_session_exit_v1", "退出后等待原条件重现")]:
        record = {"round": round_, "study": result["study_id"], "title": title, "status": "COMPLETED_TARGET_NOT_MET",
                  "result": f"reports/research/{folder}/result.json", "evaluated_candidate_source_runs": result["candidate_configurations"],
                  **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                             "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "post_selected_best_base"]},
                  "primary_base": next(x for x in result["primary"] if x["cost"] == "BASE"),
                  "primary_stress": next(x for x in result["primary"] if x["cost"] == "STRESS")}
        if round_ == 31:
            record.update({"reference_accounts": 3, "completed_fits": r31["completed_fits"], "training_sample_rows": r31["training_sample_rows"]})
        if not any(r["round"] == round_ for r in index["completed_rounds"]):
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += result["candidate_configurations"]
        write_json(ROOT / "reports/research" / folder / "acceptance_outcome.json", {"recorded_at": now(), "status": "COMPLETED_TARGET_NOT_MET",
                   "historical_point_target_met": result["historical_point_target_met"], "goal_achieved": False,
                   "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
    candidate = {"study": r32["study_id"], "model": "REARM_RIDGE", "status": "POST_SELECTED_LOCAL_IMPROVEMENT_NOT_VALIDATED",
                 "source_result": "reports/research/510300_rearmed_session_exit_v1/result.json",
                 "base_main_sharpe": metric(r32, "REARM_RIDGE")["net_sharpe"], "base_earlier_sharpe": metric(r32, "REARM_RIDGE", early=True)["net_sharpe"],
                 "stress_main_sharpe": metric(r32, "REARM_RIDGE", "STRESS")["net_sharpe"], "stress_earlier_sharpe": metric(r32, "REARM_RIDGE", "STRESS", True)["net_sharpe"],
                 "earlier_learned_exit_cycles": 2, "main_completed_round_trips": 24, "earlier_completed_round_trips": 9,
                 "saved_uncertainty": "reports/research/510300_learned_rearm_saved_diagnostic_v1/result.json", "goal_achieved": False}
    index.update({"updated_at": now(), "status": "ROUNDS31_32_COMPLETE_ENTRY_EXIT_CONSISTENCY_RESEARCH_CONTINUES", "goal_achieved": False,
                  "latest_completed_round": record, "running_studies": [], "current_followup_candidate": candidate,
                  "process_state_note": "第31、32轮和保存账户增量诊断完成，当前没有这些研究的运行进程。",
                  "count_warning": "完成32轮，296个不同配置或范围、308个已评价来源版本、768个主评价记录。登记来源版本313，包含5个旧未运行绑定。第31、32轮另34个较早诊断及3个参考账户不计入768。",
                  "next_work": ["停止第31轮其余失败设置和两层树退出的邻近模型参数细调。",
                                "仅保留REARM_RIDGE为候选；先检查入场条件与已有退出预测是否矛盾，下一轮只登记一个基于事前预估持仓状态的入场退出一致版本，不重训旧模型。",
                                "既有结果仍不足以确认稳定超额；较早时期仅两次学习退出，增量区间跨零，不把0.705或0.748当目标达成。",
                                "EPS及其他来源补齐保持暂停，资产范围问题待回复，不重复询问。"],
                  "checks": "两轮14项必要机制测试通过；全部新账户日期、财富、T+1、重新进入状态核对。保存账户增量及两种连续区块区间已完成。",
                  "prepare_gpt_numerical_review_package": False})
    index["partial_rounds"] = [x for x in index.get("partial_rounds", []) if x["round"] not in [31, 32]]
    delivery = {"created_at": now(), "type": "LEARNED_CYCLE_EXIT_AND_REARM_CHINESE_RESULTS", "rounds": [31, 32],
                "directory": str(OUT), "main_document": str(document), "monthly_chinese_models": str(models), "new_gpt_review_archive_created": False}
    if not any(d.get("type") == delivery["type"] for d in index["deliveries"]):
        index["deliveries"].append(delivery)
    require(index["evaluated_configurations_in_this_resumption"] == 296 and index["evaluation_accounts_in_this_resumption"] == 768, "本轮统计计数未对齐")
    write_json(INDEX, index)
    write_json(OUT / "delivery_receipt.json", {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_SAVED_RESULTS",
               "main_document": str(document), "monthly_rules": str(models), "new_candidate_configurations": 9,
               "main_evaluation_records": 34, "new_main_accounts": 18, "earlier_diagnostic_records": 34,
               "new_earlier_accounts": 18, "reference_accounts": 3, "completed_fits": r31["completed_fits"], "necessary_tests_passed": 14,
               "files": [{"path": p.name, "sha256": digest(p)} for p in [document, models] + [OUT / name for name in copied]],
               "goal_achieved": False, "new_gpt_review_archive_created": False, "security_audit_performed": False})
    print(json.dumps({"中文报告": str(document), "逐月模型中文规则": str(models), "完成轮数": len(index["completed_rounds"]), "主评价记录": 768}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
