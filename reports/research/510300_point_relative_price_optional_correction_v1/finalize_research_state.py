"""依据实际结果更新四份长期事实文件，保留其他分支和真实前瞻登记。"""
from pathlib import Path
import copy
import hashlib
import json
from datetime import datetime

ROOT = Path.cwd()
REL = "reports/research/510300_point_relative_price_optional_correction_v1"
OUT = ROOT / REL
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FACTS = ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
         "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, exclusive=True):
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def timestamp():
    return datetime.now().astimezone().isoformat()


def protected(state):
    exact = {"earliest_future_exchange_session", "next_new_close_eligible_at", "registered_candidate_intents",
             "accepted_independent_candidates", "current_validated_candidates"}
    return {key:value for key,value in state.items()
            if "forward" in key or "prospective" in key or key in exact}


def main():
    if (OUT / "facts_update_receipt.json").exists():
        raise RuntimeError("本次事实已登记，不重复覆盖。")
    result = read(OUT / "prediction_summary.json")
    qualification = read(OUT / "source_and_design_qualification.json")
    verification = read(OUT / "verification.json")
    if result["prediction_gate_passed"] or result["accounting"]["new_accounts"] != 0:
        raise RuntimeError("最终处置要求读取实际门失败和账户未运行事实。")
    if verification["maximum_prediction_error"] != 0. or verification["new_model_fits"] != 0:
        raise RuntimeError("保存结果未完整复算。")
    freeze = read(OUT / "freeze.json")
    if sha(OUT / "protocol.json") != freeze["protocol_sha256"]:
        raise RuntimeError("金融协议已变化。")
    for item in freeze["sources"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("冻结对象变化：" + item["path"])
    old_phase_rel = "reports/research/510300_point_d02_optional_correction_v1/finite_optional_phase_summary.json"
    old_phase = read(ROOT / old_phase_rel)
    employment_rel = "reports/research/510300_point_employment_delivery_optional_correction_v1/prediction_summary.json"
    employment = read(ROOT / employment_rel)
    if old_phase["function_configurations"] != 11 or old_phase["accepted_function_configurations"] != 0:
        raise RuntimeError("原有限阶段口径发生变化。")
    phase = {
        "at": timestamp(), "scope": "原十一固定可选函数及随后两项不同用途，非项目全局试验史。",
        "function_configurations": 13, "accepted_function_configurations": 0,
        "original_finite_phase": old_phase_rel,
        "additional_results": [employment_rel, REL + "/prediction_summary.json"],
        "total_auxiliary_coefficient_estimations": old_phase["total_auxiliary_coefficient_estimations"]
            + employment["accounting"]["auxiliary_coefficient_estimations"] + result["accounting"]["auxiliary_coefficient_estimations"],
        "total_monthly_cache_reuses": old_phase["total_monthly_cache_reuses"]
            + employment["accounting"]["monthly_cache_reuses"] + result["accounting"]["monthly_cache_reuses"],
        "core_model_reestimations": 0, "new_accounts": 0, "new_return_labels": 0,
        "original_natural_rows_shared_not_added": 1507, "paired_predictions_shared_not_added": 1010,
        "original_unknown_predictions_shared_not_added": 497, "cycles_by_period_shared_not_added": [6, 18],
        "trial_interpretation": "13个固定表达、325辅助估计、1170月复用，共用样本，不是325独立策略或312独立周期；不是全项目尝试总数。",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False,
        "new_fits_in_compilation": 0,
    }
    save(OUT / "extended_optional_phase_summary.json", phase)
    banner = (
        "510300相对510500的原5/20日价差可选两系数已一次完成并拒绝。TECH.R143来源/无标签设计通过，"
        "3488日/1507状态保持、1459完整48未知/12部分有值，115成熟月均秩2；原价至2026-08-14，"
        "收盘15:00在原15:05前，首版未认证。首次源校验误把E70的2025年底范围屏蔽当成全期间字段，"
        "原程序/协议/失败保留，另完整v1_0_1仅修正跨用途边界，实际字段/窗口/缺失/成员不变；15必要测试首次通过。"
        "TECH.R144唯一金融协议/R145结果，25实际辅助估计/90复用、核心0重拟合，142月115可用27未知，"
        "1010配对/497原未知、981活动29精确回退，21负与非负判断变化不是21交易。"
        "早期MSE下降5.765501%、近期上升0.923732%，原5000入场年块95%改善区间两期跨零，整体FAIL。"
        "经济SKIPPED，新账户/标签/采集0，收益夏普NOT_COMPUTED。45冻结对象、3488原字段/142版本/25方程/"
        "1507预测/24周期损失复算正常通过，预测差0、复算0拟合。原十三表达累计325辅助估计/1170复用、0接受，"
        "非全局尝试统计；E71/E72及所有旧拒绝保持。新实际源与金融比较打断此前首次输入阻塞，"
        "本轮progress、计数0；目标active未完成。下一原E03真实前瞻及2026-10-08 15:05原登记保持，"
        "1008实际日一次机制评价不等于完整策略验证。独立验证/去过拟合/完整收益夏普目标未达。"
    )
    rows = []
    for period in result["periods"]:
        interval = period["improvement_interval"]
        rows.append(f'| {period["period"]} | {period["paired_rows"]} / {period["cycles"]} | '
                    f'{period["baseline_raw_return_mse"]:.12f} | {period["candidate_raw_return_mse"]:.12f} | '
                    f'{period["relative_mse_change"]:+.6%} | [{interval["low"]:+.12f}, {interval["high"]:+.12f}] | FAIL |')
    table = "\n".join([
        "| 固定时期 | 配对状态 / 周期 | 原MSE | 候选MSE | 相对变化 | 原入场年块95%改善区间 | 门 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |", *rows])
    state_append = f'''

## TECH.R143—R145：原两ETF相对价格的持有退出可选函数已完成并拒绝

本轮有实际新来源入口和金融比较，不是再次状态核对。首次输入阻塞之后，其他分支E70/E71/E72新增了两ETF相对价格来源、固定五日评分失败及其全机会归因。技术线仅观察510500.SH，风险执行标的仍510300.SH/现金。新假设是在原八项技术状态相似时，原5/20日价格变化差能否增加原继续持有相对下一合法开盘退出的预测信息；不借用其他线评分、五日标签、80线机会或账户。

TECH.R143先绑定旧用途、两固定窗口和全原成员。原价格2013-12-02至2026-08-14、基金主表身份明确；3488日中3048有完整两项，1507状态1459完整/48未知，12状态只一项有值，各原值分别保持。原115成熟月标准SVD两列均可识别，未读目标/0拟合。全字段直接支持仍不成立，允许的是原缺失处精确回原核心的另一个完整函数。原价未复权/未含分红，不是纯规模因果、预期差或资金流；15:00公开价到15:05的钟只是历史重构，不认证第一次收件。

首次资格程序已退出1：误要求另一个仅检验至2025年底的E70主动屏蔽列等于新技术合同的2026原价派生值，149日×2列缺失位置不同。共同有限字段差0、原价格全重叠一致。原程序、原协议及失败保留，完整v1_0_1仅按E70既定范围核共同字段，范围外原屏蔽必须保持NaN；未改变新事前原价合同、窗口、原未知或经济门。初版12测试通过；修正版加3项边界回归，15测试首次通过。最终资格通过才登记TECH.R144，不把初版失败说成成功。

TECH.R144固定alpha1两系数/新截距0，原八项、原尺度与截距、1507行、原目标、周期总权重1、142月115可用27未知、最近20/至少10周期100行和入场锁定保持。两原价差联合已知才标准化clip正负5并中心化；设计与原目标减固定核心的残差各周期内中心化。缺项精确回核心、核心未知双方未知。预测门仍要求两个原时期完整配对、周期等权原MSE严格改善且5000原入场年块seed51030099的改善95%下界均>0，未通过则账户SKIPPED。

TECH.R145实际运行一次25辅助估计/90月复用，核心0重拟合。1010完整配对/497原未知，981活动/29精确回退，21符号变化不是交易次数。原双期结果：

{table}

两期门均FAIL、整体REJECTED。未计算新账户净CAGR/夏普/pB，不得称收益或夏普提高。45冻结对象和全部原字段/142版本/25方程/1507预测/24周期损失保存复算正常退出0；预测最大差0、方程梯度最大4.94396e-17、全局设计均值最大3.99680e-16，复算0新拟合。冻结前采用原fit同数组布局复算尺度，1e-13容差保持，不修改原D02/就业配送模块。

原十一函数汇总不改；加就业配送和本次相对价共13固定表达、325辅助估计/1170月复用、整体接受0。共用1507/1010/6与18，不相加成独立证据，也不是全项目搜索总数。E71/E72、原轮动/跨ETF流/压力图谱、TECH.R139/R142及B02资格裁决分别保持，不从早期优点选时期或反转近期救回。该限定结果不能证明所有相对价格信息永远无效。

上一轮为no_progress，本轮实际新来源资格、金融比较与保存复算为progress，首次输入阻塞连续计数归0，历史首次记录仍保留。目标active/未完成，独立验证NOT_ESTABLISHED、物理首版未认证、全局DSR/PBO NOT_COMPUTED、去过拟合未完成。没有自己的运行中进程。当前没有另一已准入待跑收益候选；新实质机制可按原授权另审，不迁移其他分支权限。

下一具体实验仍是原E03 SAVED_WEIGHT/POINT_BINARY真实前瞻：首个2026-10-08 15:05合格新收盘只启动，原两登记/0真实新账户日和闭合周期不改。终点1008实际交易日只评价一次，最低30周期/各5盈亏不是功效保证，资金分配机制比较不自动满足完整策略目标。不回填历史，不把未来日期或WAITING文件称为活进程。

直接证据：[完整报告](../{REL}/研究结果与下一步.md)、[来源资格](../{REL}/source_and_design_qualification.json)、[实际结果](../{REL}/prediction_summary.json)、[保存复算](../{REL}/verification.json)、[原范围失败](../{REL}/initial_source_scope_failure_and_repair_registration.json)、[十三表达有限累计](../{REL}/extended_optional_phase_summary.json)、[下一路线](../{REL}/next_research_route.json)。
'''
    decisions_append = f'''

## TECH.R143 — 两ETF相对原价的不同用途与无标签来源资格

- 假设：原技术八项相似时，510300相对510500的5/20日原价状态补充原持有退出价值，不是原E71五日联合评分营救。
- 验证方法：先绑定两个ETF身份、原未复权价/固定窗口、15:00到原15:05钟，原3488日/1507状态和所有缺价保持；只读元数据，原115成熟月标准SVD均须两列秩2。旧轮动、跨ETF周流、压力图谱及E71/E72用途分别核查。
- 结果：3048日完整、1459状态完整/48未知/12部分有值，115月均秩2，0收益拟合/账户/联网。初版错误要求E70范围外屏蔽值同新技术合同的2026字段而退出1，原件保留；另完整v1_0_1仅修范围校验，原价/字段/窗口/成员/未知不改；共同有限差0，15必要测试首次通过后最终资格PASS。
- 为什么接受/拒绝：接受新完整回退函数的来源和设计资格，不等于接受金融增量；原E70范围截至2025年底不移植为技术线已授权全期间的遮罩，也不改原E70/E71输入。缺原价仍未知，完整全成员字段直接资格未通过，首版未认证。
- 是否需要重新验证：本来源检查已完成，不按金融结果扩窗/换标的/补缺价。真实新来源或独立数据按另协议；下一唯一金融函数R144。
- 证据：[用途](../{REL}/prior_definition_and_function_addendum.json)、[修正登记](../{REL}/initial_source_scope_failure_and_repair_registration.json)、[资格](../{REL}/source_and_design_qualification.json)。

## TECH.R144 — 原价差唯一可选两系数金融协议

- 假设：固定两原价格变化差在原八项外形成稳健跨期周期内预测增量，再可能提高净收益和夏普。
- 验证方法：源资格和15测试通过后冻结一次，原全部行/目标/周期权重/月钟/入场锁定保持；alpha1/新截距0、联合缺失精确原核心。完整双期原MSE均下降及5000入场年块95%下界均>0后才另冻结原20万/252日/50%风险上限的BASE/STRESS完整账户。
- 结果：冻结45源对象，实际一次25辅助估计/90复用，核心0重拟合，见R145。无新收益标签或账户。
- 为什么接受/拒绝：接受事前可执行的隔离协议，不将两个原ETF价差解释为资金归属或确定风格因果。原十二表达及E71失败保持，尺度复算在金融冻结前沿用原fit同数组布局，容差不改。
- 是否需要重新验证：不改窗口、符号、字段、时期、费用或原成熟成员救回，结果失败后该固定表达结束。
- 证据：[协议](../{REL}/protocol.json)、[测试](../{REL}/tests_receipt_v1_0_1.json)、[冻结](../{REL}/freeze.json)。

## TECH.R145 — 相对原价实际预测拒绝，完整保存复算通过

- 假设：R144固定函数在原6/18自然周期和两个固定时期提高继续价值预测，且足够稳定进入账户比较。
- 验证方法：1010完整配对/497原未知、周期等权原MSE和固定原年块95%门；原源/模型/25方程/1507预测/24周期损失另保存复算，0重拟合。
- 结果：早期MSE0.006796098625414917→0.006404269496162215，下降5.765501%，改善95%[-0.000068580676,+0.001608449046]跨零；近期0.0055065266523754795→0.0055573922166346395，上升0.923732%，95%[-0.000144526556,+0.000032936376]跨零。两门FAIL、整体REJECTED，981活动/29精确回退，21符号变化非21交易。45冻结对象/142版本/25方程/1507预测/24周期损失正常复算通过、预测差0；账户SKIPPED/0新账户和标签/净收益夏普NOT_COMPUTED。
- 为什么接受/拒绝：早期点估计下降但不稳定，近期误差点估计变差，不符合事前双期门；拒绝金融增量，接受实现和保存复算。原十三固定表达累计接受0，E71/E72及其他旧失败不逆转，不选早期、删除缺价成员或换符号救回。
- 是否需要重新验证：该固定开发表达停止。实质新机制或真实独立数据另登记；原E03两登记及2026-10-08首钟/1008实际日机制评价保持。真实新来源和金融比较打断此前首次阻塞，本轮progress/阻塞计数0，完整目标active未完成。独立验证、去过拟合和全局尝试校正未建立。
- 证据：[结果](../{REL}/prediction_summary.json)、[复算](../{REL}/verification.json)、[报告](../{REL}/研究结果与下一步.md)、[运行回执](../{REL}/continuation_receipt.json)、[限定累计](../{REL}/extended_optional_phase_summary.json)、[下一路线](../{REL}/next_research_route.json)。
'''
    report = "# 原两ETF相对价格可选持有退出预测：实际结果与下一步\n\n" + banner + "\n\n" + state_append + (
        "\n## 复现入口\n\n完整源范围修正版为`research/point_relative_price_optional_correction_v1_0_1.py`。"
        "原已执行目录带唯一登记保护，勿原地重跑金融模型；在保留全部45冻结源的干净复现目录中按"
        "`qualify`、`freeze`、`run`、`verify`顺序执行。记录命令为"
        "`.venv\\Scripts\\python.exe -X utf8 -m research.point_relative_price_optional_correction_v1_0_1`加上述动作。"
        "原v1源比对程序及source_protocol.json只用于说明初次退出，不能替代修正版成功入口。\n"
    )
    with (OUT / "研究结果与下一步.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    before_state = STATE.read_bytes()
    state = json.loads(before_state.decode("utf-8-sig"))
    state_original = copy.deepcopy(state)
    immutable = protected(state)
    (OUT / "state_before_R145_fact_update.json").write_bytes(before_state)
    backups = OUT / "facts_before_R145"
    backups.mkdir(exist_ok=False)
    receipts = []
    for relative in FACTS:
        path = ROOT / relative
        before = path.read_bytes()
        (backups / path.name).write_bytes(before)
        text = before.decode("utf-8-sig")
        lines = text.splitlines(keepends=True)
        matches = [i for i,line in enumerate(lines[:65]) if line.startswith(">") and "TECH.R142" in line]
        if len(matches) != 1:
            raise RuntimeError("技术入口段不唯一：" + relative)
        index = matches[0]
        prefix = "> 技术线本轮最新TECH.R145："
        if relative.endswith("PROJECT_STATE_TECHNICAL_LINE.md"):
            prefix = "> 用户允许新增隔离实验，原冻结策略保持。最新TECH.R145："
        elif relative.endswith("RESEARCH_DECISIONS_TECHNICAL_LINE.md"):
            prefix = "> 依据当前协议、代码、结果及Git历史；最新TECH.R145："
        line_end = "\r\n" if lines[index].endswith("\r\n") else "\n"
        lines[index] = prefix + banner + line_end
        updated = "".join(lines) + (decisions_append if "DECISIONS" in relative else state_append)
        if path.read_bytes() != before:
            raise RuntimeError("写入前其他分支更新了事实文件，请按新版本重新整合：" + relative)
        bom = b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
        path.write_bytes(bom + updated.encode("utf-8"))
        receipts.append({"path": relative, "before_sha256": hashlib.sha256(before).hexdigest(), "after_sha256": sha(path),
                         "other_branch_sections_preserved": True, "latest_own_decision": "TECH.R145"})
    state.update({
        "updated_at": timestamp(), "status": "research_in_progress", "goal_status": "active",
        "goal_tool_status_confirmed": "active", "goal_achieved": False,
        "latest_actual_model_decision": "TECH.R145", "latest_model_decision": "TECH.R145", "latest_technical_decision": "TECH.R145",
        "latest_completed_study": result["study"], "current_study": result["study"],
        "latest_result": REL + "/prediction_summary.json", "latest_prediction_increment": REL + "/prediction_summary.json",
        "latest_report": REL + "/研究结果与下一步.md", "latest_research_status": result["status"],
        "latest_progress": banner, "latest_overall_summary": banner, "latest_continuation_outcome": banner,
        "current_direction": "原两ETF5/20相对价格可选函数已拒绝；下一原真实前瞻或另有实质新输入。",
        "current_priority": "保留十三固定表达失败；有实质新机制须先审原来源和成员，真实前瞻按原钟接续。",
        "current_phase": "FIXED_RELATIVE_PRICE_OPTIONAL_RESIDUAL_REJECTED_SAVED_OUTPUT_VERIFIED",
        "current_unmet_evidence": "相对价格函数早期误差点估计下降但区间跨零，近期误差点估计变差且区间跨零，两期均FAIL；未运行账户，完整收益夏普/独立验证/去过拟合目标未达。",
        "next_research_question": "原SAVED_WEIGHT相对POINT_BINARY在真实新期间同资金成本风险约束下是否提高净收益与净夏普；另有实质新信息才建立不同持有退出模型。",
        "next_information_intake_plan": REL + "/next_research_route.json", "next_research_action": REL + "/next_research_route.json",
        "next_experiment_status": "NO_CURRENT_ADMISSIBLE_NEW_FINANCIAL_EXPERIMENT_REAL_FORWARD_DATA_NOT_YET_OCCURRED",
        "next_strategy_increment_status": "NOT_REGISTERED_NO_OTHER_CURRENT_ADMISSIBLE_UNRUN_MODEL",
        "next_candidate_field_status": "FIXED_RELATIVE_PRICE_TWO_FIELD_FUNCTION_REJECTED",
        "latest_next_information_intake": REL + "/source_and_design_qualification.json",
        "latest_continuation_receipt": REL + "/continuation_receipt.json",
        "previous_goal_turn_classification": state_original.get("goal_turn_classification", "no_progress"),
        "goal_turn_classification": "progress", "current_goal_turn_classification": "PROGRESS_ACTUAL_FIXED_RELATIVE_PRICE_MODEL_REJECTED",
        "goal_turn_progress_classification": "ACTUAL_NEW_SOURCE_QUALIFICATION_FIXED_MODEL_AND_SAVED_OUTPUT_VERIFICATION",
        "prior_blocked_audit_count": state_original.get("blocked_audit_count", 1), "blocked_audit_count": 0,
        "previous_input_blocker_audit": state_original.get("latest_blocked_audit"), "latest_blocked_audit": None,
        "blocking_decision": None, "blocked_audit_key": None, "blocked_reason": None,
        "live_own_process_handle": None, "verified_wait": False,
        "new_model_fits_this_continuation": 25, "new_candidate_coefficient_estimations_this_continuation": 25,
        "total_estimation_calls_this_continuation": 25, "reused_monthly_model_records_this_continuation": 90,
        "original_strategy_source_files_changed_this_continuation": 0,
        "new_accounts_this_continuation": 0, "new_accounts_in_current_phase": 0,
        "new_investment_account_evaluations_this_continuation": 0, "new_return_labels_this_continuation": 0,
        "new_market_requests_this_continuation": 0, "new_bars_this_continuation": 0,
        "new_point_replays_this_continuation": 0, "internal_reference_replays_this_continuation": 0,
        "necessary_tests_passed_this_continuation": 15,
        "saved_prediction_rows_recomputed_this_continuation": 1507,
        "saved_table_verification_status": verification["status"],
        "economic_stage_status": result["economic_stage"], "whole_model_overfitting_removed": False,
        "actual_candidate_trials": result["accounting"], "current_candidate_trials": result["accounting"],
        "current_phase_trial_accounting": result["accounting"],
        "bounded_optional_phase_summary": REL + "/extended_optional_phase_summary.json",
    })
    if protected(state) != immutable:
        raise RuntimeError("原真实前瞻或独立候选状态被改变。")
    if STATE.read_bytes() != before_state:
        raise RuntimeError("其他分支更新了目标状态，不能覆盖新状态。")
    save(STATE, state, exclusive=False)
    receipt = {"at": timestamp(), "classification": "PROGRESS_ACTUAL_FIXED_FINANCIAL_COMPARISON_AND_RECOMPUTATION",
               "facts": receipts, "state_sha256": sha(STATE), "protected_forward_and_independence_keys": len(immutable),
               "protected_values_unchanged": True, "new_fits_in_fact_update": 0,
               "actual_financial_auxiliary_estimations_this_turn": 25, "goal_status": "active", "goal_achieved": False}
    save(OUT / "facts_update_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
