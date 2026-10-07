"""保存本轮B03实际结果并接续长期事实；不拟合、不修改原策略或前瞻注册。"""
from pathlib import Path
import json
import os
import re

from research.point_account_cashflow_state_v1 import ROOT, now, require, write_json
from research.point_b03_optional_correction_v1 import check


out = ROOT / "reports/research/510300_point_b03_optional_correction_v1"
rel = out.relative_to(ROOT).as_posix()
read = lambda p: json.loads(p.read_bytes().decode("utf-8-sig"))
result = read(out / "prediction_summary.json")
verification = read(out / "verification.json")
source = read(out / "source_summary.json")
tests = read(out / "tests_receipt.json")
nextpre = read(out / "next_B04_saved_field_design_preflight.json")
proposal = read(out / "next_B04_two_optional_residual_proposal.json")
require(result["technical_decision"] == "TECH.R133" and not result["prediction_gate_passed"], "实际B03结果不一致")
require(check() == 48 and verification["maximum_prediction_error"] == 0. and tests["passed"] and tests["tests"] == 15, "B03完成凭据不一致")
require(nextpre["identifiable_months"] == 115 and not nextpre["target_column_read"] and proposal["parent_result_decision"] == "TECH.R133", "下一B04资格不一致")

overview = "B03原前20日低点首次严格收复用时/再失守计数的可选两系数周期内残差完成，整体预测门拒绝。原R101现金财富低点/严格破位与收复/新低重置/同日真0/再破转换及未收复未知保持；3488日线3212联合有值，1507状态1430有值77未收复NO_VIEW，原状态中用时真0有775、再破真0有1049。原八项及142月115可用27未知保持，25辅助估计/90月复用、核心0重拟合；1010完整配对/497原未知、949活动/61精确回退，14符号变化非交易。早期MSE增0.238503%，改进95%[-0.000122793298,+0.000016145786]跨零；近期MSE降1.490444%，95%[-0.000041272563,+0.000271041445]跨零，均FAIL。经济SKIPPED、0新账户/标签、净收益夏普NOT_COMPUTED。15必要测试首次通过/48冻结对象，原3488字段/142版本/25方程/1507预测/24周期误差复算通过，预测差0。TECH.R132协议/R133结果；原R101完整成员失败、旧T02失败及T13完整NOT_RUN来源门、E03保持，0联网/行情。下一B04负收益冲击后原波动比/冻结低点两列的可选函数仅提案NOT_RUN：25旧源/3488字段/1507身份一致，285有值1222阶段外未知，115月两列设计可识别最小特征值1.426187754292；原可用预测早期38/262、近期147/748有字段，预检未读目标/0拟合。原R100完整成员支持拒绝/T03/T14保持，不改负2倍/20与3日std/10日去重与阶段/未知。独立验证、去过拟合及完整目标未达。"
rows = []
for p in result["periods"]:
    ci = p["improvement_interval"]
    rows.append(f"| {p['period']} | {p['paired_rows']} / {p['cycles']} | {p['optional_input_known_rows']} / {p['exact_fallback_rows']} | {p['baseline_raw_return_mse']:.12f} | {p['candidate_raw_return_mse']:.12f} | {p['relative_mse_change']*100:+.6f}% | [{ci['low']:.12f}, {ci['high']:.12f}] | FAIL |")
table = "\n".join(rows)
report = f"""# B03前低收复用时与再次失守：固定两系数可选修正结果

实际完成：{result['at']}。协议TECH.R132，结果TECH.R133。上一轮C01实际拒绝并接续事实属于进展；本轮完成登记B03唯一函数的测试、冻结、真实比较与保存复算，原策略和其他分支权限保持。

## 结论

本版本{result['status']}。较早MSE上升，近期MSE下降但改进95%区间仍跨零；两期都不满足约定双期严格改善及95%下界>0，整体拒绝。经济阶段{result['economic_stage']}，新账户0，净CAGR/净夏普NOT_COMPUTED。不能只取近期下降晋升，不能拼接早期J01与近期B03的事后优点。完整目标、独立验证和去过拟合均未达。

| 固定时期 | 完整配对状态 / 自然周期 | 字段已知 / 精确回退 | 原MSE | B03可选MSE | 相对变化 | 改进95%区间 | 门 |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

沿用原入场年块5000次、seed51030099；较早2017/2018/2019三块，近期2020/2021/2022/2024/2025/2026六块，是开发历史敏感性，独立性未建立。1010状态不是1010笔独立交易，只有6/18个自然周期。原目标是自然终点继续退出相对下一合法开盘提前退出的BASE扣费增量，以当前剩余持有价值为分母，不是完整账户收益或A实际剩余现金流。

## 原字段、时钟和可选用途

经济财富W[0]=1，W[t]=W[t−1]*(raw_close[t]+当日已除息cash[t])/raw_close[t−1]，低点L[t]=W[t]*(raw_low[t]+cash[t])/(raw_close[t]+cash[t])。当前严格低于前20个实际日最低L时，接受新事件e、冻结此前低点L0，新严格低点先重置。相等不算新低。原0.001合法单位核验后用Fraction财富及低点严格比较，不加入信号epsilon，不跨坏源比较或压缩交易日。

当前事件首次W[t]>L0后才知道first_reclaim_delay=t−e，可同日真实0，之后冻结；尚未收复两列联合NaN保持，不用等待年龄、未来完成用时或上一成功事件值替代。首次严格收复后，由不低于L0到严格低于L0的转换才累加post_reclaim_rebreak_count；连续低于不多计，新低重置不带旧计数。无额外事件寿命或二次试低筛选。所有来源最迟为原当前收盘，15:05原点可用，下一合法开盘应用；不使用未来收复确认。

原TECH.R101直接追加两列的完整成员支持失败保持，全部115可用训练月没有一个支持全原成员，旧收益模型拟合0。T02是旧已完成的二次试低/内部背离收复实验，主固定失败保留；T13是可知供给退潮，完整策略NOT_RUN_FULL_ISSUANCE_COVERAGE_AND_DENOMINATOR_GATE，不是已回测失败。旧8行20万元、lag0、242年化EARLY/MAIN保存指标只复用，不重算或与当前252年化混排。

本次另限定可选两系数用途：原核心八项、截距/尺度、全部成熟训练行/目标、每周期总权重1、最近20/至少10周期100行、入场版本锁定及两负值确认不改。已知原行按原权重标准化、clip±5、再中心化；联合NaN保持，只让修正设计不活动。设计与原目标减固定核心预测的残差分别周期内去均值；beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1、新截距0。缺输入精确返回原核心浮点值，核心未知双方未知。不加等待年龄、缺失指示、log用时或计数。原通用两字段API未改，新增B03隔离入口和原联合整数/NaN/事件时钟验收。

## 实际工作量与验证

3488日线3212联合有值；1507原状态1430已知、77尚未收复未知，原用时真0有775、再破真0有1049，真实0没有当作缺失。142月115可用27未知保持；1010完整配对、497原未知，早期255已知/7精确回退，近期694/54，合计949活动/61回退。14个预测符号改变不是14笔交易。

唯一配置、两列、25次不同成熟输入的实际辅助方程估计、90个月版本复用、原核心0重拟合，新标签/账户/行情/联网均0。合成测试与区间模拟不计真实收益模型估计。115原可用训练月两列设计均可识别，最小特征值0.0551366588830288；资格预检未读收益目标。

15必要测试首次通过（15 passed in 7.67s）：原七项严格低点/现金相等边界、首次确认和真0、新低重置/再破转换、坏源清事件/完整20日、未来前缀不变/经济身份、原成熟成员与未收复未知；新八项原身份/来源时钟、冻结事件/用时/计数拒绝、两系数方程/固定核心及全行、联合NaN精确回退/不拼值、全未知训练、周期共同误差消除、未知原行的周期对比、真实0与负数/小数非法记录。

新增research/point_b03_optional_correction_v1.py及tests/test_point_b03_optional_correction_v1.py，原策略/旧字段器/通用API不改。金融模型冻结后仅运行一次，48冻结对象保持；3488日原字段与R101完全一致，142原版本/25不同方程/1507预测/24周期误差保存复算通过，预测差0，最大方程梯度5.7462715141731735e−18，最大总体设计均值1.963341769863435e−15。复算新增估计0/账户0，只证明实现一致，不证明独立性或盈利。

## 下一项有限实验

B04原“严格负收益冲击后1—10日的三日波动／冲击前20日波动、当前是否跌破冻结冲击低点”两列可选残差只提案NOT_RUN，旧准确编号TECH.R100。原R100完整成员支持失败与旧T03/T14状态保持，不重开其价格/融资入场或按收涨选日。保留经济log收益、严格负2倍前序20真实区间样本std、10日接受事件去重、后1—10日寿命、当前最后3实际收益样本std、现金低点严格比较；不改阈值、时窗、事件筛选或阶段外未知。

25旧冻结源及3488原字段、1507元数据身份已核对完全相同；285原状态联合有值、1222阶段外未知保持。115可用训练月两列设计可识别，最小特征值1.4261877542922679；源／设计预检未读目标列或未来收益，0拟合／账户／联网。早期原262可用预测中38有字段，若实施224精确回退；近期748中147有字段、601回退。低覆盖意味着大部分预测仍沿用核心，本预检不证明增量有用，也不提升旧全成员来源门。

下一须核旧用途和现行可选函数去重，完整隔离实现、必要测试并事前冻结唯一函数，再真实运行一次。原完整配对、双期周期等权MSE严格下降和原整年块改进95%下界>0保持。通过后另冻结20万元252日BASE/STRESS共同完整账户，要求双期净CAGR及净夏普均高于A、实际净pB>1、标准净期望pB−q>0、回撤<=10%，次数软目标；失败不调参救援。

下一源预检首次引用了上一实验C01表名而找不到文件，在读取任何收益目标或写出预检前终止；现完整程序改用实际B03表名及status列后完成，金融模型未重跑，字段／窗口／目标未改。

## 当前边界

历史全部DEVELOPMENT_CALIBRATION，物理首版未证、独立验证未建立、全项目DSR/PBO未计算；去过拟合与完整目标未达。原E03 SAVED_WEIGHT/POINT_BINARY前瞻注册、首个合法新原点2026-10-08 15:05和0真实新账户日/闭合周期不改，不倒填未形成的新观察。其他分支结果和授权保持。

证据：[协议](protocol.json)、[实际结果](prediction_summary.json)、[原字段核对](source_summary.json)、[必要测试](tests_receipt.json)、[复算](verification.json)、[下一B04设计资格](next_B04_saved_field_design_preflight.json)、[下一提案](next_B04_two_optional_residual_proposal.json)、[下一预检完整程序](next_B04_preflight_and_proposal.py)。
"""

protocol_entry = f"""### TECH.R132 / B03原收复用时与再失守的固定可选两系数协议

- **假设**：严格前低收复之后的原用时及再失守次数可能增加原八项周期内继续价值信息，联合未知精确保留核心。
- **验证方法**：原R101/T02/T13有限旧用途和现行字段检索，15必要测试首次通过、115月两列设计可识别且预检未读目标；48对象在真实拟合前冻结。仅原两列，全部原成熟行/目标/权重/142月115可用27未知/原八项/入场锁定保持。固定两系数周期内岭alpha1、新截距0，原双期完整MSE及入场年块5000次/seed51030099改进95%下界皆>0为门，失败不跑账户。
- **结果**：协议先冻结，原R101直接完整成员支持失败保持；不是新物理信息、独立数据或收益证据。
- **为什么接受/拒绝**：只接受另限定可选函数的开发资格，不重开T02入场或把T13完整NOT_RUN来源门当金融失败。
- **是否需要重新验证**：定义及方程不调参，实际裁决由TECH.R133接续。
- **直接证据**：[协议](../{rel}/protocol.json)、[旧用途和函数区别](../{rel}/prior_definition_and_function_addendum.json)、[测试](../{rel}/tests_receipt.json)。

"""
result_entry = f"""### TECH.R133 / B03近期点估计改善但跨期门拒绝，下一B04只设计预检

- **假设**：固定两系数可选用途可改善原继续价值预测，过门后才检验完整账户收益夏普。
- **验证方法**：冻结后只跑一次，完整1010配对/497原未知、原核心/标签/142月115可用27未知保持；25真实辅助方程估计、90复用、核心0重拟合。早期255字段已知7回退、近期694已知54回退，949活动/61精确回退。
- **结果**：REJECTED_FIXED_B03_OPTIONAL_PREDICTION_GATE_FAILED。早期262状态/6周期，MSE0.006796098625→0.006812307497增0.238503%，改进95%[−0.000122793298,+0.000016145786]跨零；近期748/18，0.005506526652→0.005424454982降1.490444%，95%[−0.000041272563,+0.000271041445]跨零。14符号改变非交易，经济SKIPPED，新标签/账户/行情/联网0、净CAGR/夏普NOT_COMPUTED。15测试/48冻结对象，3488字段与R101完全相同、142原版本/25方程/1507预测/24周期误差复算通过，预测差0、梯度最大5.7462715141731735e−18。
- **为什么接受/拒绝**：接受实现和保存一致，拒绝本固定用途稳定增量；不只取近期下降或事后拼J01较早优点。原1430有值77未收复NaN、真0和旧R101/T02/T13状态保持，未证明所有破低修复机制永远无效。
- **是否需要重新验证**：不改前20低点/严格收复/新低重置/用时/计数/未知，不加寿命/等待年龄/二测筛选，不变号或换标签/alpha/时期/门。下一B04原冲击后波动比/冻结低点两列可选函数仅NOT_RUN提案；25旧源/3488字段/1507身份一致，285有值1222未知，115月两列设计可识别最小特征值1.426187754292，早期38/262、近期147/748原可用预测有字段。预检未读目标/0拟合，须去重、完整实现、必要测试并另冻结；原R100/T03/T14失败不营救。原E03/0真实新日周期保持，独立验证/DSR/PBO/去过拟合/完整目标未达。
- **直接证据**：[实际结果](../{rel}/prediction_summary.json)、[完整报告](../{rel}/研究结果与下一步.md)、[复算](../{rel}/verification.json)、[下一B04设计资格](../{rel}/next_B04_saved_field_design_preflight.json)、[下一提案](../{rel}/next_B04_two_optional_residual_proposal.json)。

"""
state_entry = f"""## 技术线接续：TECH.R132协议 / TECH.R133实际结果

{overview}

本轮整体拒绝，净收益夏普NOT_COMPUTED，未晋升可执行策略。下一B04模型未实现/冻结/运行，设计资格不改变原R100完整成员拒绝；真实拟合前须另冻结，双期失败停止。原E03 SAVED_WEIGHT/POINT_BINARY注册及2026-10-08 15:05、0真实新账户日/闭合周期保持。其他分支记录和授权不改。

直接证据：[报告](../{rel}/研究结果与下一步.md)、[协议](../{rel}/protocol.json)、[实际结果](../{rel}/prediction_summary.json)、[复算](../{rel}/verification.json)、[下一B04提案](../{rel}/next_B04_two_optional_residual_proposal.json)。

"""

report_path = out / "研究结果与下一步.md"
with report_path.open("xb") as f:
    f.write(report.encode("utf-8"))
snapshot = out / "facts_before_R133"
snapshot.mkdir(exist_ok=False)
updates = []
for name in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"):
    p = ROOT / "docs" / name
    b = p.read_bytes()
    txt = b.decode("utf-8-sig")
    require("TECH.R132" not in txt and "TECH.R133" not in txt, "本轮条目已有，不重复：" + name)
    txt, count = re.subn(r"(?m)^(> [^\r\n]*最新)TECH\.R131：[^\r\n]*", lambda m: m.group(1) + "TECH.R133：" + overview, txt)
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
require(state["latest_actual_model_decision"] == "TECH.R131", "当前实际结果指针变化")
protected = {k: v for k, v in state.items() if "forward" in k or "prospective" in k or k in (
    "earliest_future_exchange_session", "next_new_close_eligible_at", "registered_candidate_intents", "accepted_independent_candidates", "current_validated_candidates")}
with (out / "state_before_R133.json").open("xb") as f:
    f.write(state_bytes)
parent_result = read(ROOT / state["latest_actual_prediction_result"])
require(parent_result["technical_decision"] == "TECH.R131", "前序实际来源不符")
state["previous_actual_candidate_trials_before_R133"] = parent_result["accounting"]
pred, ver = rel + "/prediction_summary.json", rel + "/verification.json"
prop, pre = rel + "/next_B04_two_optional_residual_proposal.json", rel + "/next_B04_saved_field_design_preflight.json"
for k in ("latest_progress", "latest_overall_summary", "latest_continuation_outcome"):
    state[k] = overview
for k in ("latest_actual_model_decision", "latest_model_decision", "latest_technical_decision", "latest_prediction_technical_decision"):
    state[k] = "TECH.R133"
for k in ("latest_result", "latest_prediction_increment", "latest_actual_prediction_result"):
    state[k] = pred
for k in ("latest_research_status", "current_prediction_stage_status"):
    state[k] = result["status"]
for k in ("economic_stage_status", "latest_prediction_economic_stage_status", "current_economic_stage_status"):
    state[k] = result["economic_stage"]
state.update(updated_at=now(), status="research_in_progress", goal_achieved=False,
    latest_completed_study=result["study"], current_study=result["study"], latest_report=rel + "/研究结果与下一步.md",
    current_phase="FIXED_B03_TWO_OPTIONAL_RESIDUAL_REJECTED_NEXT_B04_FUNCTION_REVIEW",
    current_direction="原前低收复用时/再失守两列可选用途跨期拒绝；下一负收益冲击后波动及低点两列只提案。",
    current_priority="核B04旧用途及现行函数去重，完整实现/测试后冻结一次；不营救B03或旧T02。",
    next_research_question=proposal["hypothesis"], next_information_intake_plan=prop, latest_next_information_intake=pre,
    next_strategy_increment_status="NOT_RUN_B04_SEPARATE_TWO_COEFFICIENT_FUNCTION_PROPOSAL",
    next_candidate_field_status="SAME_B04_JOINT_RAW_FIELDS_DESIGN_PREFLIGHT_ONLY_NOT_PROMOTED",
    next_experiment_status=proposal["status"], current_goal_turn_classification="PROGRESS_ACTUAL_FIXED_B03_MODEL_REJECTED",
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
    code_files_added_this_continuation=["research/point_b03_optional_correction_v1.py", "tests/test_point_b03_optional_correction_v1.py"],
    code_files_changed_this_continuation=0, original_strategy_source_files_changed_this_continuation=0,
    current_study_prediction_gate_status="FAIL", prediction_gate_status="FAIL", account_return_sharpe_this_continuation="NOT_COMPUTED",
    account_return_sharpe_reason="B03固定跨期预测门失败，无新账户。",
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
    current_phase_genuine_zero_delay_rows=source["genuine_zero_delay_known_rows"],
    current_phase_genuine_zero_rebreak_rows=source["genuine_zero_rebreak_known_rows"],
    current_phase_field_scope="原R101固定前20日低点收复用时及再破转换计数，1430联合已知77未收复NaN保持；另固定两系数，原完整支持门仍失败。",
    current_phase_definition_browsing="NONE_ORIGINAL_LOCAL_FIELDS_ONLY",
    current_unmet_evidence="B03近期MSE点估计下降但区间跨零，较早点估计恶化，双期门失败、经济阶段未跑；独立验证/物理历史首版/全项目多重尝试校正未建立，完整收益夏普目标未达。下一B04仅设计资格及提案。",
    latest_B03_optional_verification=ver, latest_b03_optional_source_summary=rel + "/source_summary.json",
    latest_B03_design_identifiability=rel + "/design_identifiability_preflight.json",
    latest_B04_optional_metadata_design_preflight=pre, complete_B03_all_member_field_admission=False,
    old_R101_direct_two_field_support_gate_passed=False, complete_B04_all_member_field_admission=False,
    old_R100_direct_two_field_support_gate_passed=False, current_phase_next_B04_old_sources_checked=25,
    historical_refit_counter_interpretation="原核心重拟合0，25次实际估计为新B03辅助两系数方程，均计入new_model_fits；不是50独立配置，复算0估计。",
    actual_candidate_trials=result["accounting"], current_phase_trial_accounting=result["accounting"], orders_authorized=False)
require({k: state[k] for k in protected} == protected, "前瞻保护字段改变")
receipt = {"at": now(), "study": result["study"], "actual_result_decision": "TECH.R133", "status": result["status"],
    "classification": "PROGRESS_ACTUAL_FIXED_B03_MODEL_REJECTED", "result": pred, "verification": ver,
    "accounting": result["accounting"], "next_model_status": "NOT_RUN", "next_proposal": prop,
    "next_design_preflight": pre, "protected_state_keys_unchanged": list(protected), "goal_achieved": False,
    "independent_validation": "NOT_ESTABLISHED", "whole_model_overfit_removed": False,
    "new_model_fits_during_verification": 0, "new_accounts": 0, "new_network_requests": 0,
    "original_strategy_source_changes": 0, "frozen_source_count": 48, "frozen_definition_after_run_changed": False,
    "next_source_only_preflight_failed_read_attempts": 1,
    "next_source_only_preflight_read_error": "预检首次引用C01表名不存在，收益目标读取或写入前终止；仅改用实际B03表名和status列，金融模型0重跑。"}
write_json(out / "continuation_receipt.json", receipt, exclusive=True)
state["latest_continuation_receipt"] = rel + "/continuation_receipt.json"
for p, b, newbytes in updates:
    require(p.read_bytes() == b, "事实文件并发变化，停止覆盖：" + str(p))
    temp = p.with_name(p.name + ".TECH_R133.tmp")
    with temp.open("xb") as f:
        f.write(newbytes)
    require(p.read_bytes() == b, "替换前事实文件并发变化")
    os.replace(temp, p)
require(state_path.read_bytes() == state_bytes, "状态并发变化，停止覆盖")
temp = state_path.with_name("state.TECH_R133.tmp")
with temp.open("xb") as f:
    f.write((json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
require(state_path.read_bytes() == state_bytes, "替换前状态并发变化")
os.replace(temp, state_path)
print(json.dumps({"状态": "B03实际拒绝、完整报告、四份长期事实及状态已接续R133；B04只NOT_RUN提案。",
    "报告": str(report_path), "前瞻保护字段": len(protected), "真实辅助估计": 25, "核心重拟合": 0, "新账户": 0}, ensure_ascii=False))
