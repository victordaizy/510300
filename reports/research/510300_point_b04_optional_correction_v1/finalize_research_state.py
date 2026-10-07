"""保存B04实际结果与下一D02来源设计资格；不拟合、不开账户或修改原策略。"""
from pathlib import Path
import json
import os
import re
from research.point_account_cashflow_state_v1 import ROOT, now, require, write_json
from research.point_b04_optional_correction_v1 import check

out = ROOT / "reports/research/510300_point_b04_optional_correction_v1"
rel = out.relative_to(ROOT).as_posix()
read = lambda p: json.loads(p.read_bytes().decode("utf-8-sig"))
result = read(out / "prediction_summary.json")
verification = read(out / "verification.json")
source = read(out / "source_summary.json")
tests = read(out / "tests_receipt.json")
nextpre = read(out / "next_D02_saved_field_design_preflight.json")
proposal = read(out / "next_D02_two_optional_residual_proposal.json")
require(result["technical_decision"] == "TECH.R135" and not result["prediction_gate_passed"], "B04实际终态不一致")
require(result["periods"][0]["prediction_gate_passed"] and not result["periods"][1]["prediction_gate_passed"], "B04双期判定不一致")
require(check() == 48 and verification["maximum_prediction_error"] == 0. and tests["passed"] and tests["tests"] == 15, "B04完成凭据不一致")
require(nextpre["identifiable_months"] == 115 and not nextpre["target_column_read"] and proposal["parent_result_decision"] == "TECH.R135", "下一D02资格不一致")

overview = "B04原负收益冲击后波动比/冻结低点两列的可选周期内残差完成，整体预测门拒绝。原R100严格负2倍/20与3真实收益日样本std/10日去重/后1—10寿命/现金低点0或1/阶段外未知保持；3488日线769有值，1507状态285有值1222未知，146不创新低真0/139新低真1。原八项及142月115可用27未知保持，25辅助估计/90月复用、核心0重拟合；1010完整配对/497原未知、185活动/825精确回退，0负/非负判断改变。早期MSE降1.295546%，改进95%[+0.000000393634,+0.000401070853]全正PASS；近期降0.184601%，95%[-0.000021632578,+0.000048457005]跨零FAIL，整体FAIL。经济SKIPPED、0新账户/标签、收益夏普NOT_COMPUTED。15必要测试首次通过/48冻结对象，3488原字段/142版本/25方程/1507预测/24周期误差复算通过，预测差0。TECH.R134协议/R135结果，原R100/T03/T14及E03保持，0联网/行情。下一D02严格负隔夜缺口吸收比例及有符号缺口的可选两系数仅提案NOT_RUN；26旧源/3488字段/1507身份一致，762联合有值745联合不完整，原正零隔夜吸收NaN及已知gap分别保持，115月两列设计可识别最小特征值2.171160840982；原可用预测早期133/262、近期400/748有两项，预检未读目标/0拟合。原R94完整成员拒绝/HIGH-LOW及相关旧日内隔夜用途不改，不移植PCA吸收率。独立验证、去过拟合和完整目标未达。"
rows = []
for p in result["periods"]:
    ci = p["improvement_interval"]
    gate = "PASS" if p["prediction_gate_passed"] else "FAIL"
    rows.append(f"| {p['period']} | {p['paired_rows']} / {p['cycles']} | {p['optional_input_known_rows']} / {p['exact_fallback_rows']} | {p['baseline_raw_return_mse']:.12f} | {p['candidate_raw_return_mse']:.12f} | {p['relative_mse_change']*100:+.6f}% | [{ci['low']:.12f}, {ci['high']:.12f}] | {gate} |")
table = "\n".join(rows)
report = f"""# B04急跌后波动变化与冲击低点：固定两系数可选修正结果

实际完成：{result['at']}。协议TECH.R134，结果TECH.R135。上一轮B03实际拒绝及事实更新属于进展；本轮完成唯一B04可选函数的必要测试、冻结、真实比较与保存复算。原冻结策略保持。

## 结论

本版本{result['status']}。两期MSE点估计都下降，较早改进95%区间全正通过，近期区间跨零未通过，整体拒绝。不能只取较早期或把近期不显著改写为收益优势。经济阶段{result['economic_stage']}，新账户0，净CAGR/净夏普NOT_COMPUTED。完整目标、独立验证及去过拟合未达。

| 固定时期 | 完整配对状态 / 自然周期 | 字段已知 / 精确回退 | 原MSE | B04可选MSE | 相对变化 | 改进95%区间 | 门 |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

沿用原入场年块5000次/seed51030099，较早2017/2018/2019三块、近期2020/2021/2022/2024/2025/2026六块。这是开发历史敏感性，独立性未建立；1010状态不是1010独立交易，仅6/18个自然周期。原目标为自然终点继续退出相对下一合法开盘提前退出的BASE扣费增量，以当前剩余持有价值为分母，不是完整账户收益或A实际剩余现金流。

## 固定字段与不同可选用途

原经济log收益r[t]=log((raw_close[t]+当日已除息cash[t])/raw_close[t−1])，首区间未知。严格r[e]<−2*std(r[e−20:e−1])才触发，20个真实前序收益完整、样本std(ddof1)>0。接受起点与前接受起点相隔至少10实际日，较近冲击不重置。e当天未知，后1—10日逐日记录，11日到期；年龄10时新接受冲击先重置为年龄0。首次收涨不终止记录，不按未来修复确认挑成员。

第一列为当前最后3个真实收益区间样本std／冻结冲击前20区间std；早期年龄1时三日窗口含冲击前一天及冲击日，不改成仅冲击之后三日。原真0及大于1比例合法，不先log/截断比值，不补分母epsilon。第二列为当前现金财富低点严格低于冻结冲击日低点的0/1；相等或较高是真0，只在合法阶段完整源时有效。原0.001报价/现金单位验收及Python整数交叉乘积比较不改，未知、阶段外不填零、延续或压缩交易日。来源最晚当前完整收盘，原15:05可用、下一合法开盘应用。

原TECH.R100直接两列完整成员支持失败保持，原115可用训练月无一全部成员完整，旧收益拟合0。T03旧后1—3日收涨且不破冲击低点，另叠融资，原主无事件；T14旧后1—10日首次收涨记录波动比后清等待，比例并非过滤阈值，另有资金/低点/退出规则。只复用16旧保存指标，不重算其20万元/lag1/242年化账户，不与当前252日/两个固定时期混排，零事件夏普null不填0。

本次固定核心外另可选两系数函数，全部原成熟行/目标/周期总权重1、原八项/截距/尺度、最近20/至少10周期100行、入场锁定及两负确认保持。联合已知原行按原权重标准化、clip±5、再中心化；原联合NaN保持，只让修正不活动。设计与原目标减固定核心预测分别周期内去均值，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1、新截距0。缺输入精确返回核心浮点值，原核心未知双方未知。原通用两字段API和旧字段器均未改。

## 实际工作量及从预测到动作的限制

3488日线769已知，1507原自然状态285有两项/1222未知：1191无活跃阶段、31冲击当天。已知状态146不创新低真0、139新低真1，实际快照波动真0有0但合成规则允许。142月115可用27未知保持，1010完整配对/497原未知；较早38字段已知224精确回退，近期147/601，合计185活动/825回退。

所有1010条可用保存预测的负/非负判断变化0。原两负值确认使用这个判断，因此在这些参考状态上该判断输入没有变化。这不是已经复算全部实际账户或证明所有未来状态的收益恒等；本轮账户阶段未运行，也不能仅凭MSE下降宣称出现更好的退出点。

唯一配置、两列，25次不同成熟输入的真实辅助方程估计、90月版本复用、原核心0重拟合，新标签/账户/行情/联网均0。115原可用月两列设计可识别、最小特征值1.4261877542922679，资格预检未读目标。合成测试与区间模拟不计真实收益估计次数。

## 必要测试、冻结与复算

15必要测试首次通过（15 passed in 5.29s）：原七项严格负2倍/20日参考/当前3日、十日去重与新冲击重置、真0及比例>1、现金低点相等/整数单位、坏源与完整日窗、未来前缀和经济身份、全部原成员和成熟时钟；新八项原身份/15:05、冲击当天/阶段到期未知、合法已接受锚及冻结分母、两系数方程与固定核心/全行、联合NaN精确回退不拼值、全未知训练、周期共同误差消除、未知原行保留及原比例/0或1域。

新增research/point_b04_optional_correction_v1.py及tests/test_point_b04_optional_correction_v1.py。唯一金融模型冻结后运行一次，48冻结对象不变；3488字段与R100完全相同、142原版本/25不同方程/1507预测/24周期误差复算通过，预测差0，最大方程梯度1.973247953923618e−17，最大总体设计均值2.6645352591003756e−16。复算新增估计0/账户0，只证明实现一致，不证明独立性或盈利。不给失败版本改窗、阈值、分母、阶段、未知、方向、成员、标签、alpha、时期或门，不拼失败分支。

## 下一项有限实验

D02“严格负隔夜缺口的日内吸收比例及原有符号缺口”可选两系数仅NOT_RUN提案，旧准确编号TECH.R94。原rCO=log((open+当日cash)/previous_close)，rOC=log((close+cash)/(open+cash))，rCO+rOC=原total_log；以原0.001整数报价/现金确定严格负缺口，不加收益epsilon。负隔夜时吸收=max(rOC,0)/abs(rCO)，真0及大于1保留，有符号gap=rCO仍为负。

正/零隔夜时原吸收NaN、原gap仍已知，两个原值分别保持，不强制同时NaN、不补吸收0或删成员；只有两项完整可选修正才活动。该部分未知结构不同于B04联合NaN，下一实现须专门验证。旧HIGH/LOW、gap_recovery、DAILY_01及其他日内隔夜条件用途保持，不移植沪深300成分PCA吸收率或available-members失败。

已核26旧冻结源、3488原字段及1507原身份完全一致，762联合已知745联合不完整，115可用训练月两列设计可识别、最小特征值2.1711608409818477，预检未读目标/未来收益、0拟合/账户/联网。较早原262可用预测有133两项，若实施129精确回退；近期748有400、348回退。此资格不是原R94完整成员支持晋升或金融收益证据。

下一须核旧用途与现行可选函数去重，建立完整隔离实现、必要测试，收益拟合前另冻结唯一方案。原完整配对双期MSE严格下降及原入场年块95%改进下界皆>0保持；通过后另冻结20万元252日BASE/STRESS完整账户，要求双期净CAGR/净夏普均高于A、实际净pB>1、标准净期望pB−q>0、回撤<=10%，次数软目标。失败不调参救援。

## 当前边界

历史全部DEVELOPMENT_CALIBRATION，物理首版未证、独立验证未建立、全项目DSR/PBO未计算，去过拟合和完整目标未达。原E03 SAVED_WEIGHT/POINT_BINARY前瞻注册、首个合法新原点2026-10-08 15:05、0真实新账户日/闭合周期不改。其他分支权限和结果保持。

证据：[协议](protocol.json)、[实际结果](prediction_summary.json)、[来源一致](source_summary.json)、[测试](tests_receipt.json)、[复算](verification.json)、[下一D02设计资格](next_D02_saved_field_design_preflight.json)、[下一提案](next_D02_two_optional_residual_proposal.json)、[下一预检完整程序](next_D02_preflight_and_proposal.py)。
"""

protocol_entry = f"""### TECH.R134 / B04冲击后原波动比及冻结低点的固定可选两系数协议

- **假设**：可知负收益冲击后1—10日的当前短波动相对冻结前波动及是否创新低，可能增加原八项周期内继续价值。
- **验证方法**：原R100/T03/T14及风险路由有限去重，15测试首次通过、115月两列设计资格未读目标；48对象收益拟合前冻结。严格负2倍/20与3真实收益日样本std/10日去重与阶段/现金低点0或1/NaN保持；原全部成员/标签/八项/142月115可用27未知/入场锁定不改，仅固定两系数周期内岭alpha1、新截距0。原双期完整MSE和入场年块5000次/seed51030099改进95%下界皆>0为门。
- **结果**：协议先冻结；原R100直接完整成员支持失败保持，不恢复T03/T14价格或融资入场。
- **为什么接受/拒绝**：只接受另限定函数的开发比较资格，不接受收益或独立性结论，不把当前三日窗口改成全在冲击之后。
- **是否需要重新验证**：协议不调参，实际裁决TECH.R135接续。
- **直接证据**：[协议](../{rel}/protocol.json)、[旧用途区别](../{rel}/prior_definition_and_function_addendum.json)、[测试](../{rel}/tests_receipt.json)。

"""
result_entry = f"""### TECH.R135 / B04双期误差点估计下降但近期不稳定，0负值判断变化

- **假设**：上述固定可选函数可改善继续价值预测，过门后再检验完整账户收益夏普。
- **验证方法**：冻结后只运行一次；1010完整配对/497原未知、原核心/目标/全部成员/142月115可用27未知保持。25真实辅助估计、90月复用、核心0重拟合；较早38字段已知224回退、近期147/601，185活动/825精确回退。
- **结果**：REJECTED_FIXED_B04_OPTIONAL_PREDICTION_GATE_FAILED。较早262状态/6周期，MSE0.006796098625→0.006708052011降1.295546%，改进95%[+0.000000393634,+0.000401070853]全正PASS；近期748/18，0.005506526652→0.005496361531降0.184601%，95%[−0.000021632578,+0.000048457005]跨零FAIL，整体FAIL。负/非负判断改变0，只指保存参考状态，未复算全账户或证明未来收益恒等。经济SKIPPED、新标签/账户/行情/联网0、净CAGR/夏普NOT_COMPUTED。15测试/48对象、3488原字段/142版本/25方程/1507预测/24周期误差复算通过，预测差0、梯度最大1.973247953923618e−17。
- **为什么接受/拒绝**：接受实现一致和较早限定开发结果，拒绝整体稳定增量；不能从MSE下降直接推出更优退出点或净夏普。原285已知1222NaN、146真0/139真1、原R100/T03/T14和其他冻结失败保持。
- **是否需要重新验证**：不改冲击/波动/寿命/低点/未知/方向/成员/标签/alpha/时期或门营救。下一D02原负缺口吸收比例及有符号缺口可选两系数仅NOT_RUN提案；26旧源/3488字段/1507身份一致，762联合有值745联合不完整，115月两列设计可识别最小特征值2.171160840982；较早133/262、近期400/748原可用预测有两项。预检未读目标/0拟合，须去重、完整实现/测试/事前冻结。原正零隔夜吸收NaN及仍已知gap分别保持，不强制两列NaN或补0；R94完整支持失败及HIGH/LOW/其他日内隔夜/PCA归属保持。原E03/0真实新日周期保持，独立验证/DSR/PBO/去过拟合/完整目标未达。
- **直接证据**：[结果](../{rel}/prediction_summary.json)、[报告](../{rel}/研究结果与下一步.md)、[复算](../{rel}/verification.json)、[下一D02资格](../{rel}/next_D02_saved_field_design_preflight.json)、[下一提案](../{rel}/next_D02_two_optional_residual_proposal.json)。

"""
state_entry = f"""## 技术线接续：TECH.R134协议 / TECH.R135实际结果

{overview}

较早预测门PASS但近期FAIL，整体拒绝，0负/非负判断变化仅指保存参考状态，不是账户收益恒等证明。净收益夏普NOT_COMPUTED、0可执行策略晋升。下一D02仅设计资格与提案，未实现/冻结/拟合，原R94完整成员门不晋升。原E03 SAVED_WEIGHT/POINT_BINARY注册及2026-10-08 15:05、0真实新账户日/周期保持，其他分支授权与记录不改。

直接证据：[报告](../{rel}/研究结果与下一步.md)、[协议](../{rel}/protocol.json)、[结果](../{rel}/prediction_summary.json)、[复算](../{rel}/verification.json)、[下一D02提案](../{rel}/next_D02_two_optional_residual_proposal.json)。

"""

report_path = out / "研究结果与下一步.md"
with report_path.open("xb") as f:
    f.write(report.encode("utf-8"))
snapshot = out / "facts_before_R135"
snapshot.mkdir(exist_ok=False)
updates = []
for name in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"):
    p = ROOT / "docs" / name
    b = p.read_bytes()
    txt = b.decode("utf-8-sig")
    require("TECH.R134" not in txt and "TECH.R135" not in txt, "本轮条目已有，不重复：" + name)
    txt, count = re.subn(r"(?m)^(> [^\r\n]*最新)TECH\.R133：[^\r\n]*", lambda m: m.group(1) + "TECH.R135：" + overview, txt)
    require(count == 1, "自身前序横幅不唯一：" + name)
    addition = protocol_entry + result_entry if "RESEARCH_DECISIONS" in name else state_entry
    newline = "\r\n" if b"\r\n" in b else "\n"
    addition = addition.replace("\r\n", "\n").replace("\n", newline)
    newtxt = txt.rstrip("\r\n") + newline * 2 + addition.rstrip("\r\n") + newline
    newbytes = (b"\xef\xbb\xbf" if b.startswith(b"\xef\xbb\xbf") else b"") + newtxt.encode("utf-8")
    with (snapshot / name).open("xb") as f:
        f.write(b)
    updates.append((p, b, newbytes))

state_path = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
state_bytes = state_path.read_bytes()
state = json.loads(state_bytes.decode("utf-8-sig"))
require(state["latest_actual_model_decision"] == "TECH.R133", "当前实际结果指针变化")
protected = {k: v for k, v in state.items() if "forward" in k or "prospective" in k or k in (
    "earliest_future_exchange_session", "next_new_close_eligible_at", "registered_candidate_intents", "accepted_independent_candidates", "current_validated_candidates")}
with (out / "state_before_R135.json").open("xb") as f:
    f.write(state_bytes)
parent_result = read(ROOT / state["latest_actual_prediction_result"])
require(parent_result["technical_decision"] == "TECH.R133", "前序实际来源不符")
state["previous_actual_candidate_trials_before_R135"] = parent_result["accounting"]
pred, ver = rel + "/prediction_summary.json", rel + "/verification.json"
prop, pre = rel + "/next_D02_two_optional_residual_proposal.json", rel + "/next_D02_saved_field_design_preflight.json"
for k in ("latest_progress", "latest_overall_summary", "latest_continuation_outcome"):
    state[k] = overview
for k in ("latest_actual_model_decision", "latest_model_decision", "latest_technical_decision", "latest_prediction_technical_decision"):
    state[k] = "TECH.R135"
for k in ("latest_result", "latest_prediction_increment", "latest_actual_prediction_result"):
    state[k] = pred
for k in ("latest_research_status", "current_prediction_stage_status"):
    state[k] = result["status"]
for k in ("economic_stage_status", "latest_prediction_economic_stage_status", "current_economic_stage_status"):
    state[k] = result["economic_stage"]
state.update(updated_at=now(), status="research_in_progress", goal_achieved=False,
    latest_completed_study=result["study"], current_study=result["study"], latest_report=rel + "/研究结果与下一步.md",
    current_phase="FIXED_B04_TWO_OPTIONAL_RESIDUAL_REJECTED_NEXT_D02_FUNCTION_REVIEW",
    current_direction="原前低收复用时/再失守两列可选用途跨期拒绝；下一负收益冲击后波动及低点两列只提案。",
    current_priority="核D02旧用途及现行函数去重，完整实现/测试后冻结一次；不营救B04或旧T02。",
    next_research_question=proposal["hypothesis"], next_information_intake_plan=prop, latest_next_information_intake=pre,
    next_strategy_increment_status="NOT_RUN_D02_SEPARATE_TWO_COEFFICIENT_FUNCTION_PROPOSAL",
    next_candidate_field_status="SAME_D02_JOINT_RAW_FIELDS_DESIGN_PREFLIGHT_ONLY_NOT_PROMOTED",
    next_experiment_status=proposal["status"], current_goal_turn_classification="PROGRESS_ACTUAL_FIXED_B04_MODEL_REJECTED",
    previous_goal_turn_classification="PROGRESS_ACTUAL_FIXED_C01_MODEL_REJECTED", blocked_audit_count=0,
    blocked_audit_key=None, blocked_reason=None, blocking_decision=None, prior_blocked_audit_count=0,
    new_accounts_this_continuation=0, new_accounts_in_current_phase=0, new_investment_account_evaluations_this_continuation=0,
    new_strategy_accounts_this_continuation=0, new_point_replays_this_continuation=0, internal_reference_replays_this_continuation=0,
    new_model_fits_this_continuation=25, new_candidate_coefficient_estimations_this_continuation=25,
    auxiliary_coefficient_estimations_this_continuation=25, total_estimation_calls_this_continuation=25,
    total_return_model_fit_calls_this_continuation=25, original_core_coefficient_reestimations_this_continuation=0,
    historical_model_refits_this_continuation=0, baseline_model_recomputation_fits_this_continuation=0,
    new_control_fits_this_continuation=0, new_return_labels_this_continuation=0, new_network_requests_this_continuation=0,
    new_market_data_requests_this_continuation=0, new_market_bars_this_continuation=0, new_bars_this_continuation=0,
    original_monthly_membership_checks_this_continuation=142, reused_monthly_model_records_this_continuation=90,
    original_no_model_months_preserved_this_continuation=27, necessary_tests_passed_this_continuation=15,
    necessary_tests_passed_in_current_phase=15, latest_source_freeze_count=48,
    code_files_added_this_continuation=["research/point_b04_optional_correction_v1.py", "tests/test_point_b04_optional_correction_v1.py"],
    code_files_changed_this_continuation=0, original_strategy_source_files_changed_this_continuation=0,
    current_study_prediction_gate_status="FAIL", prediction_gate_status="FAIL", account_return_sharpe_this_continuation="NOT_COMPUTED",
    account_return_sharpe_reason="B04固定跨期预测门失败，无新账户。",
    new_method_admitted="ONE_FIXED_OPTIONAL_TWO_COEFFICIENT_DEVELOPMENT_FUNCTION_ONLY",
    new_market_information_admitted=False, new_externally_sourced_market_information_admitted=False,
    new_historical_field_admitted=False, new_historical_field_promoted=False, promoted_executable_strategy=False,
    overall_overfit_removed=False, whole_model_overfitting_removed=False, formal_global_DSR_PBO="NOT_COMPUTED",
    independent_validation="NOT_ESTABLISHED", current_phase_original_state_rows=1507,
    current_phase_optional_known_rows=1430, current_phase_optional_unknown_rows=77,
    current_phase_available_paired_predictions=1010, current_phase_original_unknown_predictions=497,
    current_phase_exact_core_fallback_predictions=61, current_phase_active_optional_predictions=949,
    current_phase_prediction_sign_changes=14, current_phase_known_genuine_zero_fields=None,
    current_phase_known_genuine_one_fields=None,
    current_phase_genuine_zero_volatility_rows=source["genuine_zero_volatility_known_rows"],
    current_phase_known_no_new_low_rows=source["known_no_new_low_rows"],
    current_phase_field_scope="原R100固定前20日低点收复用时及再破转换计数，1430联合已知77未收复NaN保持；另固定两系数，原完整支持门仍失败。",
    current_phase_definition_browsing="NONE_ORIGINAL_LOCAL_FIELDS_ONLY",
    current_unmet_evidence="B04近期MSE点估计下降但区间跨零，较早点估计恶化，双期门失败、经济阶段未跑；独立验证/物理历史首版/全项目多重尝试校正未建立，完整收益夏普目标未达。下一D02仅设计资格及提案。",
    latest_B04_optional_verification=ver, latest_b04_optional_source_summary=rel + "/source_summary.json",
    latest_B04_design_identifiability=rel + "/design_identifiability_preflight.json",
    latest_D02_optional_metadata_design_preflight=pre, complete_B04_all_member_field_admission=False,
    old_R100_direct_two_field_support_gate_passed=False, complete_D02_all_member_field_admission=False,
    old_R94_direct_two_field_support_gate_passed=False, current_phase_next_D02_old_sources_checked=25,
    historical_refit_counter_interpretation="原核心重拟合0，25次实际估计为新B04辅助两系数方程，均计入new_model_fits；不是50独立配置，复算0估计。",
    actual_candidate_trials=result["accounting"], current_phase_trial_accounting=result["accounting"], orders_authorized=False)
state.update(
    previous_goal_turn_classification="PROGRESS_ACTUAL_FIXED_B03_MODEL_REJECTED",
    current_phase="FIXED_B04_TWO_OPTIONAL_RESIDUAL_REJECTED_NEXT_D02_FUNCTION_REVIEW",
    current_direction="原冲击后波动比/冻结低点两列可选函数较早PASS但近期不稳定，整体拒绝；下一负隔夜缺口吸收两列仅提案。",
    current_priority="核D02旧日内隔夜用途及可选函数去重，完整实现/测试后冻结一次；不营救B04或旧HIGH/LOW。",
    current_phase_optional_known_rows=285, current_phase_optional_unknown_rows=1222,
    current_phase_exact_core_fallback_predictions=825, current_phase_active_optional_predictions=185,
    current_phase_prediction_sign_changes=0, current_phase_known_new_low_rows=source["known_new_low_rows"],
    current_phase_next_D02_old_sources_checked=26,
    account_return_sharpe_reason="B04较早预测门通过，近期区间跨零，整体门失败；无新账户。",
    current_phase_field_scope="原R100严格负2倍20日波动冲击，后1—10日原rv3/冻结rv20及现金低点0/1；285联合已知1222未知保持，另固定两系数，原完整成员门失败。",
    current_unmet_evidence="B04双期MSE点估计下降，较早门PASS但近期改善区间跨零，整体FAIL且参考状态负/非负判断0变化；经济阶段未跑，独立验证/物理首版/多重尝试校正未建立，完整收益夏普目标未达。下一D02只是设计资格及提案。",
    historical_refit_counter_interpretation="原核心重拟合0，25次实际估计为新B04辅助两系数方程，已计入new_model_fits；复算0估计。",
    current_phase_genuine_zero_delay_rows=None, current_phase_genuine_zero_rebreak_rows=None)
require({k: state[k] for k in protected} == protected, "前瞻保护字段改变")
receipt = {"at": now(), "study": result["study"], "actual_result_decision": "TECH.R135", "status": result["status"],
    "classification": "PROGRESS_ACTUAL_FIXED_B04_MODEL_REJECTED", "result": pred, "verification": ver,
    "accounting": result["accounting"], "next_model_status": "NOT_RUN", "next_proposal": prop,
    "next_design_preflight": pre, "protected_state_keys_unchanged": list(protected), "goal_achieved": False,
    "independent_validation": "NOT_ESTABLISHED", "whole_model_overfit_removed": False,
    "new_model_fits_during_verification": 0, "new_accounts": 0, "new_network_requests": 0,
    "original_strategy_source_changes": 0, "frozen_source_count": 48, "frozen_definition_after_run_changed": False,
    "next_source_only_preflight_failed_read_attempts": 0}
write_json(out / "continuation_receipt.json", receipt, exclusive=True)
state["latest_continuation_receipt"] = rel + "/continuation_receipt.json"
for p, b, newbytes in updates:
    require(p.read_bytes() == b, "事实文件并发变化，停止覆盖：" + str(p))
    temp = p.with_name(p.name + ".TECH_R135.tmp")
    with temp.open("xb") as f:
        f.write(newbytes)
    require(p.read_bytes() == b, "替换前事实文件并发变化")
    os.replace(temp, p)
require(state_path.read_bytes() == state_bytes, "状态并发变化，停止覆盖")
temp = state_path.with_name("state.TECH_R135.tmp")
with temp.open("xb") as f:
    f.write((json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
require(state_path.read_bytes() == state_bytes, "替换前状态并发变化")
os.replace(temp, state_path)
print(json.dumps({"状态": "B04实际拒绝、完整报告、四份长期事实及状态已接续R135；D02只NOT_RUN提案。",
    "报告": str(report_path), "前瞻保护字段": len(protected), "真实辅助估计": 25, "核心重拟合": 0, "新账户": 0}, ensure_ascii=False))
