"""保存就业配送的实际拒绝与复算修复，保留原策略、前瞻及其他分支。"""
import json
import os
from research.point_account_cashflow_state_v1 import ROOT, now, digest, require, write_json

OUT = ROOT / "reports/research/510300_point_employment_delivery_optional_correction_v1"
REL = OUT.relative_to(ROOT).as_posix()
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FILES = ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
         "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]

def read(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))

def protected(key):
    return "forward" in key or "prospective" in key or key in {
        "earliest_future_exchange_session", "next_new_close_eligible_at", "registered_candidate_intents",
        "accepted_independent_candidates", "current_validated_candidates"}

def replace(path, before, after):
    require(path.read_bytes() == before, "共享事实已更新，停止覆盖：" + str(path))
    temporary = path.with_name(path.name + ".tech_R142.tmp")
    with temporary.open("xb") as stream:
        stream.write(after)
    os.replace(temporary, path)

def main():
    require(not (OUT / "facts_update_receipt.json").exists(), "事实更新已完成，不重复。")
    result = read(OUT / "prediction_summary.json")
    verify = read(OUT / "verification_v1_0_2.json")
    qualification = read(OUT / "source_and_design_qualification.json")
    tests = read(OUT / "tests_receipt.json")
    require(result["technical_decision"] == "TECH.R142" and not result["prediction_gate_passed"]
            and verify["maximum_prediction_error"] == 0 and verify["distinct_normal_equations"] == 25
            and verify["new_model_fits"] == 0 and qualification["identifiable_months"] == 115
            and tests["passed"] and tests["tests"] == 12, "实际完成凭据不一致。")
    require(digest(OUT / "saved_output_recomputation_v1_0_2.py") == verify["repair_program_sha256"], "复算修复代码变化。")
    overview = (
        "不同原制造业就业/配送水平的可选两系数已实际拒绝。TECH.R140先绑不同用途及无目标源/设计资格，"
        "原140月HTML、132旧原值/源一致及8本地2026原件、115成熟月两列可识别；"
        "1507状态1390有值117未知保持，公布日23:59:59/原点15:05，首版未认证。"
        "TECH.R141唯一金融协议/R142结果，原八项/142月115可用27未知固定，"
        "25辅助估计/90复用、核心0重拟合；1010完整配对/497原未知、1010活动/0可用配对回退，15负与非负判断改变非15交易。"
        "早期MSE下降2.212102%，改进95%区间跨零FAIL；近期上升0.592986%，改善区间全负FAIL，整体拒绝。"
        "账户SKIPPED、收益夏普NOT_COMPUTED、新标签/账户/行情/联网0。12必要测试最终通过，首轮11过1夹具结构不符，生产代码不改；"
        "原verify因C/F归约次序产生最大5.2994e-13尺度差而退出，原失败保留；独立v1_0_1按原fit同数组归约恢复25尺度精确0，"
        "全部数学检查回执落盘后末尾缺json导入退出，另v1_0_2完整复算正常通过，容差及原冻结模型/结果均不改。"
        "140原月源/142版本/25方程/1507预测/24周期损失复算通过、预测差0，验证0拟合。"
        "另一宏观E65五日失败、TECH.R139十一固定表达及B02资格终态分别保持，不迁移他线权限或收益。"
        "该固定就业配送路线结束，不按结果换差分、符号、组合或时期；下一仍原E03两登记真实前瞻，"
        "首个合法新收盘2026-10-08 15:05/0新账户日与闭合周期保持。独立验证、去过拟合和完整收益夏普目标未达。"
    )
    rows = []
    for p in result["periods"]:
        ci = p["improvement_interval"]
        rows.append(f"| {p['period']} | {p['paired_rows']} / {p['cycles']} | {p['baseline_raw_return_mse']:.12f} | {p['candidate_raw_return_mse']:.12f} | {p['relative_mse_change'] * 100:+.6f}% | [{ci['low']:.12f}, {ci['high']:.12f}] | FAIL |")
    table = "\n".join(rows)
    comparison = "| 固定时期 | 完整配对 / 周期 | 原MSE | 就业配送MSE | 相对变化 | 改进95%区间 | 门 |\n|---|---:|---:|---:|---:|---:|---|\n" + table
    report = f"""# 制造业就业与配送：不同持有退出用途的固定比较

实际完成：{result['at']}。源/设计TECH.R140，唯一金融协议R141，实际结果R142。

整体REJECTED_FIXED_EMPLOYMENT_DELIVERY_OPTIONAL_PREDICTION_GATE_FAILED。早期点估计改善2.212102%但区间跨零；近期误差增加0.592986%、改善区间全负。经济{result['economic_stage']}，新账户0，净CAGR/净夏普NOT_COMPUTED。完整目标尚未完成。

{comparison}

## 经济问题和旧失败

就业原扩散指数描述用工增减普遍程度，配送原指数高表示交付更快；不是就业人数、交货天数、市场预期差或唯一供需原因。不对两原值添加差分、相减、交互或50线。三种原分类/样本组完整保留，原DI水平同尺度是开发假设，未证明群体完全可比。

另一宏观E65是原八宏观变量加两问项、五日费用标签、504日滚动树/岭和80分自然机会，已经失败。本版本固定原技术八项，针对原1507自然持有成员的剩余价值继续/提前退出目标，只另解周期内残差两系数。用途与函数不同，不重开E65或迁移其胜率、收益、权限。TECH.R139十一固定表达及B02资格终态保持，本版本是之后单列的另一个开发问题；不是新物理信息或独立样本。

## 来源与事前资格

只用原技术线140份本地当月HTML，2015-01至2026-08连续；132旧就业/配送原值、原源、方法组与E64精确一致，8份2026已存原件提取两问项，重复表必须一致。原公布日期当日23:59:59才可用，原决定15:05，下一合法开盘应用，不借另一任务较早09:00/09:30钟。历史首版/首次收件未认证，不采集新源。

1507原状态1390有两项、117未知保留，115原可用成熟月两列标准SVD秩均2，原27未知模型不补建。资格只读原身份/成熟元数据，未读收益目标。原全部行/目标/周期等权和原八项固定；已知行标准化clip正负5后重新中心化，未知仅令修正不活动。设计及原目标减固定核心分别周期内中心化，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1、新截距0。缺输入精确回核心，核心未知双方未知。

## 实际工作量与金融限制

唯一配置，25不同成熟输入辅助估计、90月复用、核心0重拟合、新标签0。原1010完整配对/497原未知；本版本两项在全部1010原可用配对有值，所以活动1010/回退0，不代表117原源未知被补全。15个负与非负参考判断改变不等于15交易。原两期6/18周期，原入场年块5000次/seed51030099；相关历史仍DEVELOPMENT_CALIBRATION，不是独立验证。

加上R139原十一表达，本有限可选函数记录范围合计12配置/300辅助估计/1080月复用，整体通过0；这不是全项目所有试验数、独立策略数或独立周期数，不据此计算DSR/PBO。全局DSR/PBO仍NOT_COMPUTED。

## 测试与复算问题的处理

12必要测试最终通过，12 passed in 3.57s。首轮1 failed/11 passed，因合成夹具把旧月另放在无当月行的主表；修为实际同表多月行后通过，生产模块未改。测试覆盖当月与两表头结构、重复表矛盾、非法数值、0/100端点和部分未知、公布日延迟及微秒时钟、最新缺失不回退、未来前缀、身份/时钟唯一性和方法组原值保留。

原金融模块verify命令因尺度核对失败退出：原fit用完整二维raw[known]得到C布局，模板verify用DataFrame.loc得到F布局；原值相同，np.average归约顺序造成最大5.299433731348951e-13差，超过原绝对1e-13容差。25组按原fit同数组操作均精确恢复，未放宽容差、重估模型或改结果。原失败及诊断保存。

独立v1_0_1全部数学检查通过并写verification.json后，末尾输出缺json导入导致进程退出；该脚本和回执保留，不能称其命令正常完成。完整v1_0_2另加导入，只重复保存输出核对，正常完成；140月字段、142原版本、25不同方程、1507预测、24周期损失复算通过，预测差0、最大梯度{verify['maximum_normal_equation_gradient']:.16g}，复算0拟合/账户。原金融模块和冻结协议仍原样，金融拒绝终态保持。建议复算使用[完整v1_0_2脚本](saved_output_recomputation_v1_0_2.py)，结果见[最终复算凭据](verification_v1_0_2.json)。

## 下一步

本固定就业配送表达结束，不按结果改符号、差分、维度、alpha、缺失掩码、方法组、时期、费用或与旧单期优点混合。下一具体实验沿原E03 SAVED_WEIGHT/POINT_BINARY登记的真实前瞻比较，原登记、2026-10-08 15:05首个新收盘及0真实新账户日/闭合周期不重置。不同实质机制仍须另核旧用途、事前来源和原成员后冻结；当前没有另一新收益模型准入，其他分支授权独立保留。独立验证、去过拟合、完整收益夏普目标未达。
"""
    with (OUT / "研究结果与下一步.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    entries = f"""
## TECH.R140 — 原就业配送两问项的不同用途与无标签资格

- 假设：制造业用工和交付快慢能补充固定技术八项的持有退出价值；不是原E65五日联合评分重跑。
- 验证方法：先绑定两原DI水平/不同函数和原技术线23:59:59钟，核140本地HTML/132旧原值与源、原1507身份，原115成熟月标准SVD两列秩，仅读元数据不读目标。
- 结果：140月/132一致/8已存2026原件，1390有值117未知，115月全秩2；0金融拟合/账户/联网，原27未知模型保持。原首版未认证，完整全成员字段支持不因此成立。
- 接受/拒绝原因：接受另一个完整核心回退函数的开发来源与设计资格，不是金融增量；原E65、旧PMI/CPI与R139失败分别保持。三原方法组同尺度仅是开发假设。
- 是否需要重新验证：唯一金融表达另冻结R141；实际作用失败不靠改字段/时钟/方法组救回。历史首次发布和独立验证仍未建立。
- 直接证据：[用途](../{REL}/prior_definition_and_function_addendum.json)、[源协议](../{REL}/source_protocol.json)、[实际资格](../{REL}/source_and_design_qualification.json)。

## TECH.R141 — 就业配送唯一两系数事前金融协议

- 假设：上述两原水平在固定核心之外形成跨期稳健的周期内预测增量，再可能提升净收益/夏普。
- 验证方法：12必要测试最终通过后冻结唯一两系数alpha1/新截距0；原八项/所有原行/目标/权重/月钟/入场锁定不变。原双期完整配对MSE和5000年块95%下界均需改善，才另冻结原20万252日BASE/STRESS完整账户。
- 结果：实际金融运行一次，25辅助估计/90复用、核心0；见R142。冻结对象{verify['frozen_sources']}，未采集市场数据/新标签。
- 接受/拒绝原因：接受事前可执行的隔离协议与固定缺失函数，不接受来源通过等于收益有效；原十一函数停止记录保持。
- 是否需要重新验证：不得按实测结果修改符号、差分、组合、时期或费用；下一独立样本/新实质机制另登记。
- 直接证据：[金融协议](../{REL}/protocol.json)、[测试](../{REL}/tests_receipt.json)、[冻结](../{REL}/freeze.json)。

## TECH.R142 — 就业配送实际拒绝，保存复算另版修复通过

- 假设：R141固定表达能在原6/18周期的两个时期稳健提高继续价值预测。
- 验证方法：25真实辅助估计/90复用、原142月115可用27未知；1010完整配对/497原未知、原周期等权MSE及年块门。保存源/方程/预测/周期损失另复算，0重拟合。
- 结果：早期MSE0.006796098625414917→0.006645761966223465，下降2.212102%，改善95%[-0.000008373003928480,+0.000235935558853463]跨零FAIL；近期0.0055065266523754795→0.005539179583843024，上升0.592986%，95%[-0.000059404859539008,-0.000009830266731136]全负FAIL。整体REJECTED，账户SKIPPED/0新账户与标签/净收益夏普NOT_COMPUTED；1010活动/0可用配对回退，15判断变化不是15交易。
- 接受/拒绝原因：近期误差确实变差、早期区间跨零，不选早期或反转输出救回。原verify的C/F归约差5.2994e-13失败保留；按原fit同数组操作25尺度精确恢复、不放宽1e-13。独立v1_0_1数学回执已落盘但末尾json导入错误退出，完整v1_0_2正常通过，140月/142版本/25方程/1507预测/24周期损失差0。金融模型/结果和所有冻结对象未改。
- 是否需要重新验证：本固定表达结束，不重拟合救回。后续使用独立完整v1_0_2做保存复算，原错误原件保留；原E03两个登记真实前瞻及2026-10-08 15:05/0新日与闭合周期保持。开发历史非独立验证，DSR/PBO NOT_COMPUTED，去过拟合/完整目标未达。
- 直接证据：[实际结果](../{REL}/prediction_summary.json)、[报告](../{REL}/研究结果与下一步.md)、[最初失败诊断](../{REL}/initial_verification_failure_and_diagnosis.json)、[末尾输出错误](../{REL}/verification_v1_0_1_stdout_failure.json)、[最终复算](../{REL}/verification_v1_0_2.json)。
"""
    state_entry = f"\n## 技术线接续：TECH.R140源设计 / R141协议 / R142实际拒绝\n\n{overview}\n\n{comparison}\n\n直接证据：[报告](../{REL}/研究结果与下一步.md)、[实际结果](../{REL}/prediction_summary.json)、[最终复算](../{REL}/verification_v1_0_2.json)。\n"
    originals, replacements = {}, {}
    snapshots = OUT / "facts_before_R142"
    snapshots.mkdir(exist_ok=False)
    for filename in FILES:
        path = ROOT / filename
        before = path.read_bytes()
        originals[filename] = before
        with (snapshots / path.name).open("xb") as stream:
            stream.write(before)
        text = before.decode("utf-8-sig")
        require(not any("TECH.R" + str(n) in text for n in [140, 141, 142]), "新技术编号已经写入。")
        newline = "\r\n" if "\r\n" in text else "\n"
        lines, count = text.splitlines(keepends=True), 0
        for i, line in enumerate(lines):
            if line.startswith("> ") and "最新TECH.R139" in line:
                lines[i] = line.split("最新TECH.R139", 1)[0] + "最新TECH.R142：" + overview + newline
                count += 1
        require(count == 1, "旧技术入口匹配不唯一。")
        addition = entries if "RESEARCH_DECISIONS" in filename else state_entry
        text = "".join(lines).rstrip("\r\n") + newline + addition.replace("\n", newline)
        replacements[filename] = (b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
    old_bytes = STATE.read_bytes()
    state = json.loads(old_bytes.decode("utf-8-sig"))
    require(state["latest_actual_model_decision"] == "TECH.R137", "原目标模型状态已经变化。")
    original_forward = {k:v for k,v in state.items() if protected(k)}
    with (OUT / "state_before_R142.json").open("xb") as stream:
        stream.write(old_bytes)
    parent = read(ROOT / "reports/research/510300_point_d02_optional_correction_v1/prediction_summary.json")["accounting"]
    for key, value in list(state.items()):
        if "this_continuation" in key and not key.startswith("prior_") and not protected(key) and isinstance(value, (int, float, bool)):
            state[key] = False if isinstance(value, bool) else 0
    state.update({"updated_at": now(), "status": "research_in_progress", "goal_status": "active", "goal_achieved": False,
        "latest_actual_model_decision": "TECH.R142", "latest_model_decision": "TECH.R142", "latest_technical_decision": "TECH.R142",
        "current_phase": "FIXED_EMPLOYMENT_DELIVERY_TWO_FIELD_OPTIONAL_REJECTED_SAVED_VERIFICATION_V1_0_2_PASS",
        "latest_actual_prediction_result": REL + "/prediction_summary.json", "latest_research_report": REL + "/研究结果与下一步.md",
        "latest_employment_delivery_optional_verification": REL + "/verification_v1_0_2.json",
        "latest_employment_delivery_source_qualification": REL + "/source_and_design_qualification.json",
        "latest_employment_delivery_verification_initial_failure": REL + "/initial_verification_failure_and_diagnosis.json",
        "previous_actual_candidate_trials": parent, "previous_actual_candidate_trials_before_R142": parent,
        "actual_candidate_trials": result["accounting"], "current_candidate_trials": result["accounting"], "current_phase_trial_accounting": result["accounting"],
        "new_model_fits_this_continuation": 25, "auxiliary_coefficient_estimations_this_continuation": 25,
        "new_candidate_coefficient_estimations_this_continuation": 25, "total_estimation_calls_this_continuation": 25,
        "total_return_model_fit_calls_this_continuation": 25, "total_fit_calls_this_continuation": 25,
        "reused_monthly_model_records_this_continuation": 90, "necessary_tests_passed_this_continuation": 12,
        "necessary_tests_passed_in_current_phase": 12, "original_no_model_months_preserved_this_continuation": 27,
        "saved_prediction_rows_recomputed_this_continuation": 1507, "preserved_no_view_prediction_rows_this_continuation": 497,
        "source_freeze_count_this_continuation": verify["frozen_sources"], "current_phase_source_freeze_count": verify["frozen_sources"],
        "current_phase_field_source_freeze_count": verify["frozen_sources"], "current_phase_original_state_rows": 1507,
        "current_phase_optional_known_rows": 1390, "current_phase_optional_unknown_rows": 117,
        "current_phase_available_paired_predictions": 1010, "current_phase_original_unknown_predictions": 497,
        "current_phase_active_optional_predictions": 1010, "current_phase_exact_core_fallback_predictions": 0,
        "current_phase_prediction_sign_changes": 15, "current_phase_known_genuine_zero_fields": None,
        "current_phase_raw_gap_known_rows": None, "current_phase_raw_gap_preserved_in_incomplete_rows": None,
        "current_phase_above_one_absorption_rows": None,
        "current_phase_field_scope": "原140当月就业/配送DI水平、1390原状态有值117未知，公布日23:59:59延迟；两系数实际双期拒绝。",
        "current_phase_prior_source_coverage_role": "140已存源/132旧值精确一致/115成熟月设计可识别；非物理首版或全部1507成员字段支持。",
        "current_member_support_this_continuation": "1010原可用预测完整配对且两项已知；117源未知与497核心未知分别保留。",
        "validation_method_this_continuation": "12测试最终通过；原复算归约和末尾输出错误保留，独立v1_0_2同原fit归约/原容差完成保存输出复算，0金融拟合。",
        "source_freeze_count_interpretation": "本唯一金融函数冻结源数按实际回执；25辅助估计/90复用。140报告不是金融候选数，复算不加入拟合。",
        "historical_refit_counter_interpretation": "核心0重拟合；25为本次就业配送辅助系数，原verify/独立两版核对/归约诊断金融拟合0。",
        "account_return_sharpe_this_continuation": "NOT_COMPUTED", "account_return_sharpe_reason": "早期区间跨零、近期MSE变差且改善区间全负，整体拒绝，账户阶段SKIPPED。",
        "code_files_added_this_continuation": ["research/point_employment_delivery_optional_correction_v1.py", "tests/test_point_employment_delivery_optional_correction_v1.py"],
        "report_programs_added_this_continuation": [REL + "/build_isolated_module.py", REL + "/saved_output_recomputation_v1_0_1.py", REL + "/saved_output_recomputation_v1_0_2.py", REL + "/finalize_research_state.py"],
        "source_purpose_and_clock_bound_this_continuation": True,
        "algorithmic_source_supported_natural_rows_this_continuation": 1390,
        "original_monthly_membership_checks_this_continuation": 142,
        "control_monthly_parameter_checks_this_continuation": 142,
        "saved_natural_member_rows_recomputed_this_continuation": 1507,
        "recorded_optional_function_scope_after_R142": {"scope": "R139十一表达及之后单列就业配送，不是全项目试验总数", "configurations": 12, "auxiliary_estimations": 300, "monthly_reuses": 1080, "accepted": 0, "new_accounts": 0},
        "goal_turn_classification": "progress", "goal_turn_progress_classification": "ACTUAL_DIFFERENT_EMPLOYMENT_DELIVERY_MODEL_REJECTED_VERIFICATION_REPAIRED_AND_FACTS_SAVED",
        "blocked_audit_count": 0, "blocking_decision": None, "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False,
        "current_phase_definition_browsing": "NONE_LOCAL_ORIGINAL_HTML_ONLY", "complete_employment_delivery_all_member_field_admission": False,
        "original_frozen_models_changed": False,
    })
    require({k:v for k,v in state.items() if protected(k)} == original_forward, "前瞻或已验证候选状态改变。")
    updated = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for filename in FILES:
        require((ROOT / filename).read_bytes() == originals[filename], "其他分支在准备中更新，停止覆盖。")
    require(STATE.read_bytes() == old_bytes, "目标状态在准备中改变。")
    for filename in FILES:
        replace(ROOT / filename, originals[filename], replacements[filename])
    replace(STATE, old_bytes, updated)
    receipt = {"at":now(), "status":"PASS_R140_R141_R142_REPORT_AND_FOUR_FACTS_STATE_UPDATE",
        "updated_facts":[{"path":f,"sha256":digest(ROOT/f)} for f in FILES], "state_sha256":digest(STATE),
        "protected_forward_keys":len(original_forward), "forward_values_preserved":True,
        "new_model_fits_in_finalization":0, "new_accounts":0, "new_network_requests":0, "goal_achieved":False}
    write_json(OUT / "facts_update_receipt.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == "__main__":
    main()
