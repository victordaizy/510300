"""保存真实来源进展、待运行第十九轮以及中文完整规则交付。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from scripts.continue_forward_eps_two_institution_pipeline_20260907 import ExistingProcess, SOURCE_PARENT_PID, SOURCE_PARENT_CREATION

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
ARCHIVE = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"
PIPELINE = ROOT / "reports/research/510300_forward_eps_two_institution_pipeline_v1"
DELIVERY = ROOT / "deliverables/510300前瞻EPS多机构覆盖与完整规则_20260907"


def main():
    original = INDEX.read_text(encoding="utf-8")
    index = json.loads(original)
    assert len(index["completed_rounds"]) == 18
    assert index["registered_configurations_in_this_resumption"] == 186
    assert index["evaluation_accounts_in_this_resumption"] == 438
    assert sum(x["candidate_configurations"] for x in index["completed_rounds"]) == 186
    assert not (ROOT / "reports/research/510300_forward_eps_two_institution_policy_v2/result.json").exists()
    observed = now()
    process = ExistingProcess(SOURCE_PARENT_PID, SOURCE_PARENT_CREATION)
    assert process.running(), "更新前原采集状态已经改变，需要重新确认"
    process.close()
    receipts = sum(1 for _ in (ARCHIVE / "document_records").glob("*.json"))
    errors = sum(1 for _ in (ARCHIVE / "errors").glob("*.json"))
    pipe = read(PIPELINE / "status.json")
    assert pipe["stage"] == "等待原采集及其自动第一版提取"
    directory = read(ROOT / "reports/research/510300_forward_eps_soochow_directory_v1/result.json")
    coverage = read(ROOT / "reports/research/510300_forward_eps_coverage_representativeness_v1/result.json")
    fixed = read(ROOT / "reports/research/510300_forward_eps_soochow_facts_v3/first_250_check/result.json")
    verified = read(ROOT / "reports/research/510300_forward_eps_soochow_facts_v3/first_250_check/saved_source_verification.json")
    assert verified["status"].startswith("PASS_")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    report = DELIVERY / "前瞻EPS多机构覆盖进度与三种策略完整规则.md"
    text = f"""# 前瞻EPS研究：扩大有效覆盖，并同时设定进场与退场

更新于：{observed}。研究对象：沪深300ETF（510300）与人民币现金。目标：完整账户扣费后夏普率至少1.2，并验证超额收益是否稳定。本轮使用免费公开资料，不准备GPT数值审阅包。

## 先看结论

**第十九轮三种策略的因子、买入、加仓、减仓、清仓和重新入场规则已写定，账户尚未运行。** 当前正在处理新来源，不能提前给出新夏普。原有研究最高历史观察值约0.687，仍未达到1.2；这个值本身也没有证明稳定超额。

当前已实际保存东吴原件{receipts}份，固定队列共6734份，记录的采集失败{errors}份。数量仅代表此刻保存进度。现有采集与顺序接续程序已确认在运行；全部原件阶段结束后，会自动完成第三版EPS提取、因子、三组账户和保存数值核对。

## 本次发现了什么问题

| 问题 | 实际检查结果 | 对研究的影响 |
| --- | --- | --- |
| 单机构早期历史太少 | 原国信2021年及以前仅108份原件，其中72份已识别EPS | 旧模型有用预测主要集中在较晚年份，早期长期持有现金不能当成完整周期预测能力 |
| 有预测的公司不等于整个指数 | 原有效月份只有77至163家公司有合格EPS增长预测 | 部分公司的平均盈利预期可能不能代表沪深300整体 |
| 权重资料的历史版本证据不足 | 回看样本权重覆盖约38.6%至70.0%，中位数57.2%；逐期版本仍未补齐 | 权重只用于说明样本代表性，本轮模型不使用这些权重 |
| 旧报告表格被漏读 | 相同最早250份原件，初版仅识别12份；修正后识别166份、492条年度预测 | 漏提取会让真实已有的盈利信息变成缺失，改变策略何时能产生进出场判断 |
| 部分原件仍有明确缺口 | 固定250份中84份尚未解析，包括身份、日期、表格未识别或EPS冲突 | 保留缺失；新增目录数量不能直接当作有效因子数量 |

新东吴完整目录中，有6734份属于历史沪深300成员的报告，涉及397家公司；2017至2021年共3072份。原国信队列为3352份。先在每个判断日还原当时真正的300家公司，再取当时已公开的信息，避免用今天的成员反推历史。

固定250份修正后的492条EPS，以及587条可选利润或市盈率事实，已经核对原表、年度列和较晚公开日期。第二版已识别的90份完整事实保持一致；相关31项测试通过。以上都是来源检查成果，**不等于新策略已获利**。

## 老板可以怎样理解这三种策略

三种策略共同回答一个问题：根据当时可以看到的未来盈利预测，下一段时间持有多少510300，比保持现状更合适。盈利看下一年度预测，修正看同一目标年度的利润预期变化，估值看报告发布时的前瞻市盈率。每个月重新判断一次。

| 策略 | 用到的全部预测因子 | 要验证的问题 |
| --- | --- | --- |
| 第一种：两机构盈利三因子，主方案 | 前瞻EPS增长、同年度利润预期修正、报告前瞻盈利收益率 | 合并国信和东吴观点，是否比单一机构提供更好的持仓判断 |
| 第二种：东吴单机构三因子 | 同样三项，但只用东吴；判断和训练月份与第一种完全相同 | 为增加第二家机构提供可比对照，避免把月份不同造成的差异当成优势 |
| 第三种：两机构盈利加信息质量 | 第一种三项，再加利润上修广度、双机构覆盖比例、报告平均年龄，共六项 | 预测覆盖范围及新鲜程度，是否能帮助判断什么时候更值得相信盈利信息 |

这里的两家券商公开研报只是部分机构观点。下一年度EPS也不等于精确未来十二个月EPS。报告中的市盈率对应报告参考价格，不能称为已经校正股数和公司行为的当前月末估值。

## 每项因子到底怎样计算

1. **前瞻EPS增长。** 同一份报告中，下一年度预测每股收益减当年度预测每股收益，把差额乘二，再除以两者绝对值之和。两个值都为零、缺一个年份或有歧义，就保留缺失。使用同份报告，避免把不同报告股本口径直接混在一起。
2. **同年度利润预期修正。** 对同一机构，把当前能看到的下一年度利润预测，与90个日历日前能看到的同一绝对年度利润预测比较；仍是差额乘二，除以两值绝对值之和。要求利润标签一致，不把归母净利润与其他净利润随意相减。跨报告采用明确利润预测，避免直接用未经股数校正的每股收益修正。
3. **报告前瞻盈利收益率。** 取同份报告下一年度市盈率的倒数。没有明确市盈率、等于零或无法确认时保留缺失；不从当前价格倒推出当年预测。
4. **利润上修广度。** 先合并每家公司的有效机构修正，再计算上修公司数减下修公司数，除以可比较公司总数。它反映上调预期是否普遍，而不只看中位数。
5. **双机构覆盖比例。** 在EPS增长有有效值的公司中，同时有国信和东吴两家有效观点的公司占比。零可以表示这些公司实际都只有一家有效机构，不能把没有预测填成零。
6. **报告平均年龄。** 先在每家公司内，平均有效EPS报告距离判断日有多少日历日，再对不同公司等权平均。报告越旧，年龄越大。

每家公司、每家机构只保留最新可用报告，信息日期严格早于判断日，最长180个日历日。同一家公司有两个有效机构时先取简单平均，再对不同公司取中位数；只有一个机构时使用它实际给出的观点。机构写报告多不增加该公司的权重。最新报告如果无法识别，保留无新观点，不能跳过它回用旧预测。

增长和盈利收益率至少分别覆盖30家公司，利润修正至少覆盖15家公司。东吴和合并机构都达标，且质量变量可用，才成为三种策略共同的合格月份。上述覆盖门槛和六项因子已经固定，不按新结果临时选择。

## 怎样从因子得出持仓

每月最后一个交易日收盘后，用此前至少十二个有效、且未来60交易日收益已经完整兑现的月末样本，估计当前未来60交易日含分红收益。只用当时已经知道的数据训练。模型采用固定收缩强度的线性回归，限制系数过大；系数随着新的已兑现历史更新。某一个因子变高不会机械触发买入，要看当时模型对全部因子的合并判断。

随后同时比较六个选择：保持现有份额、全部现金、25%股票、50%股票、75%股票、100%股票。每个目标先换算为实际能买到的100份整手数量，并考虑账户现金和费用。

每个选择的评分等于：**股票占比乘预计60日收益，减去两倍股票占比平方乘预计60日收益方差，再减预计交易成本占净值的比例。** 方差来自此前60个交易日含分红日收益的样本方差乘60。选择评分最高的实际份额，近似同分优先保持现状。

## 什么时候进入，什么时候退出

| 当前情况与判断 | 下一步动作 | 执行时间 |
| --- | --- | --- |
| 当前空仓，评分最高选择要求持有510300 | 买入该目标对应的实际份额 | 下一交易日开盘 |
| 当前持仓，评分最高选择要求更多份额 | 加仓 | 下一交易日开盘 |
| 当前持仓，评分最高选择要求较少但非零份额 | 卖出差额、降低仓位 | 下一交易日开盘 |
| 当前持仓，评分最高选择是全部现金 | 清仓退出 | 下一交易日开盘 |
| 最优选择与现有份额相同，或评分近似打平 | 保持现有份额 | 无新交易 |
| 缺少有效预测、样本不足，或者不是月末 | 保留无新观点或维持份额的记录，保持现有份额 | 等下一次有效月末 |
| 已经清仓，后续新的有效月末再次要求持有 | 按同样规则重新进入 | 新判断后的下一交易日开盘 |
| 涨跌停、现金或可卖份额导致不能全部成交 | 按实际可成交数量记账，保留未成交记录 | 普通月末方案等后续有效月末重新判断 |

**清仓的经济含义是：考虑未来盈利判断、波动和交易成本后，持有股票已经不如持有现金。** 本轮没有额外加入任意止盈百分比或止损百分比。统一评价终点的清算单独标记，不能算成模型自主找到了卖点。

## 历史表现怎样报告

三种新策略各运行基础与压力费用一条完整账户，共六条新账户；加上旧国信EPS和买入持有的四条保存对照，共十条评价账户。起始20万元，评价2020年1月2日至2026年8月14日开盘终点，保留所有早期现金月份，按242交易日年化。现金收益和计算夏普的参考收益都为零。

基础费用为佣金万分之二、每笔最低5元，单边滑点万分之五；压力费用为佣金万分之四、每笔最低5元，单边滑点千分之一。分红、应收现金、100份整手、下一开盘、涨跌停及股票隔日可卖都计入。结果会同时呈现净夏普、收益、最大回撤、逐年表现、实际买卖清单及不确定性区间。

截至本说明生成时，十八轮已经完成。第十七轮主方案基础夏普为0.687，但共同月份只用EPS的对照已经有0.679，新增估值贡献区间跨过零，且只有17次预测；不能把后续持有旧份额的收益当成持续预测成功。第十八轮保留36次EPS预测，基础夏普0.643、压力0.633，只有两次用到了估值修正。目标1.2仍未达到，稳定超额也尚未证实。

本轮下一步是完成新的来源与三种账户，并根据实际表现决定研究方向。前瞻EPS继续作为主线；股东回报、公募净申购和所有期限逆回购仍按其真实口径研究。本轮三种策略尚未把未完成校正的数据写成已使用因子。

## 对应记录

- 完整策略登记：`docs/510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V2.md`。
- 双机构因子说明：`docs/510300_FORWARD_EPS_TWO_INSTITUTION_FEATURES_V2.md`。
- 固定250份来源结果：`reports/research/510300_forward_eps_soochow_facts_v3/first_250_check/result.json`。
- 来源保存核对：同目录 `saved_source_verification.json`。
- 自动接续状态：`reports/research/510300_forward_eps_two_institution_pipeline_v1/status.json`。

本说明为中文规则及来源进度交付，没有生成或上传GPT数值审阅包。
"""
    report.write_text(text, encoding="utf-8")
    index.update({
        "updated_at": observed,
        "status": "CONTINUING_SOOCHOW_ORIGINALS_SOURCE_CORRECTION_AND_REGISTERED_ROUND_19",
        "goal_achieved": False, "registered_configurations_in_this_resumption": 189,
        "evaluated_configurations_in_this_resumption": 186,
        "registered_candidate_source_runs_including_unrun_legacy_bindings": 200,
        "prepare_gpt_numerical_review_package": False,
        "latest_continuation_note": "docs/510300_FORWARD_EPS_TWO_INSTITUTION_CONTINUATION_20260907.md",
        "count_warning": "十八轮已评价186不同方法、192候选来源版本、438评价账户。第十九轮另登记三方法，来源更正前后各三绑定，方法登记189、来源登记200；含两个更早未跑绑定、三个本轮旧绑定及三个待运行更正绑定。旧已评价配置汇总182按完成轮次修正为186，没有因此新增回测。复用对照、来源重放不算独立实验。",
        "running_studies": [
            {"study": "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_V1", "stage": "ORIGINAL_COLLECTION_CONFIRMED_RUNNING",
             "session_id": 47679, "source_parent_pid": SOURCE_PARENT_PID, "source_parent_creation": SOURCE_PARENT_CREATION,
             "observed_at": observed, "archived_receipts": receipts, "errors_saved": errors,
             "selected_reports": 6734, "next_original_command_stage": "510300_FORWARD_EPS_SOOCHOW_FACTS_V1"},
            {"study": "510300_FORWARD_EPS_TWO_INSTITUTION_PIPELINE_V1", "stage": "WAITING_EXISTING_ARCHIVE_AND_V1_FACTS",
             "session_id": 2240, "pid": pipe["pipeline_pid"], "status_file": "reports/research/510300_forward_eps_two_institution_pipeline_v1/status.json",
             "result_file": "reports/research/510300_forward_eps_two_institution_pipeline_v1/result.json",
             "auto_stages_after_original_completion": ["SOOCHOW_FACTS_V3", "SAVED_SOURCE_VERIFICATION", "TWO_INSTITUTION_FEATURES_V2", "TWO_INSTITUTION_POLICY_V2", "SAVED_ACCOUNT_VERIFICATION"]},
        ],
        "partial_rounds": [{"round": 19, "study": "510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V2",
            "status": "THREE_METHODS_REGISTERED_SOURCE_COLLECTION_IN_PROGRESS_ACCOUNTS_NOT_RUN",
            "candidate_configurations": 3, "planned_evaluation_accounts": 10, "planned_new_accounts": 6,
            "planned_reused_control_accounts": 4, "new_models_fit": 0, "new_accounts_generated": 0,
            "unrun_superseded_binding": "510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V1",
            "source_only_amendment": True, "entry_exit_rules": "docs/510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V2.md"}],
        "process_state_note": "原东吴归档47679和顺序接续2240在运行；已完成的18轮不重算。接续等待原归档及其自动V1事实，再运行V3事实、来源核对、双机构因子V2、策略V2和保存账户核对。以操作系统及pipeline/status.json为准，不重启已有采集，不称新模型已运行。",
    })
    index["new_evidence"].extend([
        "原国信2021年及以前只有108原件、72份EPS；50有效月末77至163家公司，权重回看覆盖38.63%至69.97%，中位数57.17%；权重版本证明仍未完成，不入模。",
        "东吴完整公开目录16253份，全历史成员固定队列6734份、397家公司，2017至2021年3072份；原件归档正在原会话执行。",
        "固定最早250份V1识别12份36条EPS，V2识别90份264条，V3识别166份492条81公司。旧版90份事实保持一致，全部原行、年份、时钟及587条可选事实核对通过，31项测试通过。",
        "第十九轮三方法与完整中文进出场已登记；来源更正仅更换东吴事实V3绑定，旧三绑定未跑，方法与成本参数逐项相同。新账户尚未运行，不产生新夏普。",
        "顺序接续2240已持有原采集父进程句柄等待，完整资料结束后自动完成更正事实、因子、三方法六新四复用十评价账户及保存数值核对。",
    ])
    for folder, result in [
        ("510300_forward_eps_coverage_representativeness_v1", coverage),
        ("510300_forward_eps_soochow_directory_v1", directory),
        ("510300_forward_eps_soochow_facts_v3/first_250_check", fixed),
    ]:
        index["completed_source_rebuilds"].append({"study": result.get("study_id", folder.upper()),
            "result": f"reports/research/{folder}/result.json", **result})
    index["pending_source_work"].append({"study": "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_AND_FACTS_V3",
        "status": "6734_FIXED_ORIGINALS_IN_PROGRESS_WITH_REGISTERED_LAYOUT_CORRECTION",
        "directory_result": "reports/research/510300_forward_eps_soochow_directory_v1/result.json",
        "selected_reports": 6734, "observed_archived_receipts": receipts, "observed_at": observed,
        "first_250_parsed": 166, "first_250_eps_facts": 492, "remaining_work": "等待完整原件与更正事实，补充有效月度覆盖并运行已登记共同月份三方法；局部来源检查不能当完整EPS数据。"})
    index["next_work"] = [
        "优先核对原归档47679与顺序接续2240真实运行状态；无变化不重启、不并行重复执行其五步。",
        "接续成功后读取已保存第十九轮实际覆盖、训练、进入退出及账户结果；必要核对通过才更新完成轮次与交付，不先宣称夏普改善。",
        "比较两机构与共同月份东吴、质量变量与无质量变量；区分早期数据扩展、预测更新和静态持仓贡献。保留所有失败，目标未达到继续下一轮。",
        "前瞻EPS为主，继续当前股数与价格、股东回报、公募净申购和全部期限逆回购的明确口径；无GPT数值审阅包。",
    ]
    index["checks"] = "本轮31项相关测试通过；固定250原件中166份492条EPS、587条可选预测事实、较晚日期、90份旧版成功事实核对通过；三组新来源附录目视检查，未重训或生成账户。已有18轮核对记录保留，无额外安全审计。"
    index["deliveries"].append({"created_at": observed, "type": "CHINESE_MD_SOURCE_PROGRESS_AND_FULL_ENTRY_EXIT_NO_GPT_PACKAGE",
        "directory": str(DELIVERY), "main_document": str(report), "new_accounts_generated": 0,
        "new_gpt_review_archive_created": False, "pending_round": 19})
    snapshot = ROOT / "reports/research/510300_sharpe_1_2_before_two_institution_update_20260907.json"
    with snapshot.open("x", encoding="utf-8") as handle:
        handle.write(original)
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已更新：18轮完成，R19三方法待运行；原件已保存", receipts, "/ 6734，两个现有进程继续。")
    print(str(report))


if __name__ == "__main__":
    main()
