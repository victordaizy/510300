"""交付中文口径解释与完整进出场，登记尚未执行的第二十轮。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

DIAG = ROOT / "reports/research/510300_forward_eps_report_pe_identity_v1"
OUT = ROOT / "deliverables/510300前瞻EPS市盈率口径与完整进出场_20260907"
NEW_PIPE = ROOT / "reports/research/510300_forward_eps_valuation_consistency_pipeline_v1"
OLD_PIPE = ROOT / "reports/research/510300_forward_eps_two_institution_pipeline_v1"


def link(label, path):
    return f"[{label}](<{path.as_posix()}>)"


def main():
    result = read(DIAG / "result.json")
    reviewed = read(DIAG / "original_page_adjudication.json")
    assert reviewed["reviewed_reports"] == 7 and reviewed["annual_rows"] == 21
    progress = read(OLD_PIPE / "status.json")
    next_progress = read(NEW_PIPE / "status.json")
    assert progress["dependency_process_confirmed_running"]
    assert next_progress["dependency_process_confirmed_running"]
    assert not (ROOT / "reports/research/510300_forward_eps_two_institution_policy_v2/result.json").exists()
    assert not (ROOT / "reports/research/510300_forward_eps_valuation_consistency_policy_v1/result.json").exists()
    OUT.mkdir(parents=True, exist_ok=False)
    table = []
    for source in result["source_summaries"]:
        table.append(f'| {source["source"]} | {source["reports"]} | {source["reports_with_any_comparable_year"]} | {source["reports_with_outside_reference_range_year"]} | {source["comparable_reports_with_no_common_implied_price_across_years"]} |')
    examples = []
    for case in reviewed["cases"]:
        pdf = ROOT / case["raw_pdf"]["path"]
        examples.append(f'| {link(case["sec_name"] + "原报告", pdf)} | {case["category"]} | {case["finding"]} |')
    text = f"""# 前瞻EPS市盈率问题与三种策略完整进出场

更新日期：2026年9月7日。本次已经确认，部分研报原件中的EPS、市盈率和参考价本身存在不一致，不能只把网页数字抄进因子后便认为含义正确。新一组对照同时比较原市盈率、关系相容的市盈率和完全不用市盈率，所有进入与退出规则已固定；尚未出现新账户成绩。

截至本次记录，原十八轮完整评价中的历史最高基础夏普率仍约0.687，未达到1.2；此前较高方案的增量区间跨零，尚未证明稳定超额。不能把发现数据问题、补足报告或登记新策略说成已经提高夏普率。

## 这次发现了什么

总共核对2960份报告中的8877条年度预测，按原文显示精度检查EPS乘市盈率是否可能还原参考价。例如EPS显示1.83、市盈率显示整数11，舍入空间与显示11.00时不同；本次保留这些原文精度，没有人为设置一个百分比容忍线。

| 来源范围 | 已提取报告数 | 有至少一个年度可比较 | 至少一个年度与参考价不相容 | 跨年度无法对应共同价格 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

国信的754份占可比较2661份的约28.34%；东吴固定早期样本37份占可比较108份的约34.26%。这是数值关系不相容比例，不能称为已经逐份确认的错误率。价格时点、股份口径、舍入以外的原文问题或其他原因，都可能造成差异。东吴早期固定样本不是完整6734份的代表性随机抽样，不能将这个比例外推到全部报告。

随后选取七份具体异常报告，查看八页原PDF并逐项核对21条预测年度。下面列出的数值都与原件相同，这些例子的差异并非把EPS行或年度列读错。

| 原件 | 已确认的情况 | 具体证据 |
| --- | --- | --- |
{chr(10).join(examples)}

中信证券例子不能把右栏11.07直接认作中信的正确股价；西部矿业例子不能悄悄选择正文那组数字覆盖表格。两个东吴例子尚未确认差异由哪个价格时点或股份口径造成，也不会从乘积反填“真实历史价格”。

## 如何处理这些因子

前瞻EPS继续作为主线。原EPS和利润预测、旧市盈率以及此前账户都保留，新方案只对市盈率另加来源条件。至少两个预测年度的EPS与非零PE，以及首页正参考价，必须在原文精度内具有共同价格区间。通过才进入新的“关系相容PE”因子；否则保留缺失，不填零、不猜正确值、不改取更早研报补齐。

通过数值关系条件仍不等于获得月末真实估值。报告参考价与当前月末价、股份定义、预测期限、货币单位和历史版本还需要各自对齐。原第十九轮作为原报告信息的基线继续执行，本新增第二十轮用相同月份检验PE是否有用；完整规则在下文。

## 当前进度与历史成绩状态

截至{progress["observed_at"]}，东吴固定6734份原件已记录{progress["archived_receipt_files"]}份归档凭证，原采集进程仍在运行。这个带时点的数字不是实时计数。第十九轮在等待原件与事实处理，第二十轮已经接在第十九轮账户及保存核对之后自动执行；不重启现有采集。

第二十轮登记三个方法、六条新成本账户和四条复用对照，共十条评价账户。此时已完成轮次仍为18，已评价186种不同方法或范围、438条评价账户；加上待运行的第十九与第二十轮，登记不同方法192种。已评价候选来源版本仍192，登记来源版本203；这些计数不能当独立验证次数。

来源条件九项测试通过；七份原件的目视读数核对完成。新三种方法尚未训练和产生回测表现，没有可报告的新夏普率、回撤或超额。研究目标保持未完成，后续继续读取真实账户结果。没有制作GPT审阅包，也没有新增安全审计。

"""
    policy = (ROOT / "docs/510300_FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V1.md").read_text(encoding="utf-8")
    text += policy.replace("# 第二十轮：前瞻EPS中的市盈率是否可靠、有用", "## 三种新策略的全部中文规则", 1)
    text += "\n\n原始数值关系明细：" + link("逐年度EPS、市盈率与参考价", DIAG / "逐年度EPS市盈率参考价关系.csv")
    text += "；" + link("逐份报告汇总与缺口", DIAG / "逐份报告估值关系与缺口.csv") + "。\n"
    text += "来源处理记录：" + link("七份原PDF目视核对", DIAG / "original_page_adjudication.json") + "。\n"
    path = OUT / "前瞻EPS市盈率问题与三种策略进出场.md"
    assert "```" not in text
    for phrase in ["首次进入", "加仓", "减仓", "全部退出", "重新进入", "无法成交", "尚未训练", "1604日"]:
        assert phrase in text, phrase
    path.write_text(text, encoding="utf-8")
    save(OUT / "delivery_receipt.json", {"created_at": now(), "document": identity(path),
         "source_result": identity(DIAG / "result.json"), "source_visual_review": identity(DIAG / "original_page_adjudication.json"),
         "chinese_rule_text_without_code": True, "source_count_and_case_values_verified": True,
         "new_models_fit": 0, "new_account_evaluations": 0, "gpt_review_package_created": False}, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    original = index_path.read_text(encoding="utf-8")
    index = json.loads(original)
    assert len(index["completed_rounds"]) == 18
    assert index["registered_configurations_in_this_resumption"] == 189
    assert index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 200
    assert index["evaluation_accounts_in_this_resumption"] == 438
    index["registered_configurations_in_this_resumption"] += 3
    index["registered_candidate_source_runs_including_unrun_legacy_bindings"] += 3
    index["updated_at"] = now()
    index["status"] = "CONTINUING_SOOCHOW_ORIGINALS_AND_REGISTERED_ROUNDS_19_20"
    index["partial_rounds"].append({"round": 20, "study": "510300_FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V1",
        "status": "THREE_METHODS_REGISTERED_WAITING_EXISTING_ROUND_19_ACCOUNTS_NOT_RUN",
        "candidate_configurations": 3, "planned_evaluation_accounts": 10, "planned_new_accounts": 6,
        "planned_reused_control_accounts": 4, "new_models_fit": 0, "new_accounts_generated": 0,
        "entry_exit_rules": "docs/510300_FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V1.md",
        "primary": "Q1_RELATION_QUALIFIED_PE", "common_month_control": "Q2_MATCHED_RAW_REPORTED_PE",
        "removal_control": "Q3_MATCHED_EPS_PROFIT_NO_PE"})
    for study in index["running_studies"]:
        if study["study"] == "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_V1":
            study.update({"observed_at": progress["observed_at"], "archived_receipts": progress["archived_receipt_files"]})
    index["running_studies"].append({"study": "510300_FORWARD_EPS_VALUATION_CONSISTENCY_PIPELINE_V1",
        "stage": "WAITING_EXISTING_ROUND_19_COMPLETION_AND_VERIFICATION", "session_id": 91733,
        "pid": next_progress["pipeline_pid"], "observed_at": next_progress["observed_at"],
        "status_file": "reports/research/510300_forward_eps_valuation_consistency_pipeline_v1/status.json",
        "result_file": "reports/research/510300_forward_eps_valuation_consistency_pipeline_v1/result.json",
        "auto_stages": ["VALUATION_CONSISTENCY_FEATURES_V1", "VALUATION_CONSISTENCY_POLICY_V1", "SAVED_RELATION_MODEL_ACCOUNT_VERIFICATION"]})
    diagnostic = {"study": result["study_id"], "result": "reports/research/510300_forward_eps_report_pe_identity_v1/result.json",
        "report_rows": 2960, "annual_rows": 8877, "visually_reviewed_reports": 7, "visually_reviewed_pages": 8,
        "source_summaries": result["source_summaries"], "new_account_evaluations": 0,
        "outside_relation_count_not_adjudicated_error_rate": True, "existing_source_values_preserved": True}
    index["completed_diagnostics"].append(diagnostic)
    index["latest_report_pe_relation_diagnostic"] = diagnostic
    index["count_warning"] = "十八轮已评价186种不同方法、192候选来源版本、438评价账户。第十九与第二十轮各登记3种尚未评价，故登记方法192、登记来源版本203。原来源重放、旧未跑绑定与复用对照不是独立实验。"
    index["process_state_note"] = "原采集47679持续；第十九轮接续2240等待原件和V1事实，之后按原五步执行。新增第二十轮接续91733等待2240完成及保存核对，再构建PE关系条件、运行三方法和核对。三条现有会话不重复启动，以操作系统、各自status/result/failure为准。"
    index["new_evidence"].extend([
        "2960份报告8877条年度EPS、PE和参考价关系已核对；国信754/2661可比较报告至少一年度不相容，早期东吴37/108。不是已确认错误率，不能外推全东吴。",
        "七份原PDF八页、21条年度值目视相符，确认侧栏证券身份混用、正文与表格PE冲突和跨年度关系问题；两份东吴的差异原因仍未知。",
        "第二十轮在第十九轮收益出现前固定：关系相容PE、原PE、去除PE，共同月份三模型；EPS增长和利润修正原值保持。来源条件9项测试通过，尚无新训练或账户。",
        "新接续91733已运行并持有原第十九轮PID3920及创建时间句柄，按顺序自动执行，未重启原采集；明确全部进入、增加、减少、退出、缺失和再入规则。",
    ])
    index["next_work"].append("按已运行的2240和91733顺序处理第十九及第二十轮真实结果。第二十轮比较PE关系条件、原PE与去除PE，区分共同月份、PE公司与机构构成及实际进出场；不要将来源相容自动等同精确月末估值。")
    index["deliveries"].append({"created_at": now(), "type": "CHINESE_PE_SOURCE_FINDINGS_AND_COMPLETE_RULES_MD_NO_GPT_PACKAGE",
        "directory": str(OUT), "main_document": str(path), "new_account_evaluations": 0,
        "new_gpt_review_archive_created": False})
    for task in index["pending_source_work"]:
        if task["study"] == "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_AND_FACTS_V3":
            task.update({"observed_archived_receipts": progress["archived_receipt_files"], "observed_at": progress["observed_at"]})
    backup = ROOT / "reports/research/510300_sharpe_1_2_before_pe_relation_update_20260907.json"
    with backup.open("x", encoding="utf-8") as handle:
        handle.write(original)
    temporary = index_path.with_suffix(".next.json")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    continuation = ROOT / index["latest_continuation_note"]
    addition = f"""

## 后续：原报告PE关系问题与第二十轮，2026年9月7日

已完成 `510300_forward_eps_report_pe_identity_v1`，2960份报告、8877年度行，国信754份至少一年度与参考价不相容、16份跨年度无共同价格；固定早期东吴37份不相容但无跨年共同区间冲突。数量不能称错误率。七原PDF八页、21年度值已经目视核对，原表与提取结果相同；中信侧栏出现国联，西部矿业正文与表格PE冲突，东方/中天/华泰跨年度PE与EPS关系不一，隆基与阳光原因未定。见 result.json 和 original_page_adjudication.json。

第十九轮仍作为原报告信息基线运行；其“报告PE倒数”不得宣称已验证的参考价或当前月末真实盈利收益率。未改动旧来源、预测或账户。第十九轮收益出现前已固定第二十轮 `config/510300_forward_eps_valuation_consistency_policy_v1.json`，来源 `510300_forward_eps_valuation_consistency_features_v1`：正参考价、至少两个非零PE预测年度，所有可比较年度EPS乘PE的原文舍入区间须与参考价有共同交集；显式零PE整份PE不使用；未列PE不补值，最新报告不回用旧报告。币种、股数、当前价格及不可变时点仍未知，不称完整来源证明。9项来源条件测试通过。

三方法Q1关系相容PE、Q2共同月份原PE、Q3共同月份EPS增长加利润修正且完全不用PE。EPS增长和利润修正原值保持；公司内机构平均、公司间中位数。合并EPS至少30公司、利润修正15、相容PE30，四项变量均可用；三模型相同月份，不另外要求东吴单机构达标。十二成熟月末、60日标签、岭强度10、完整20万1604日账户、两费用、每方法6新加4复用共10评价、20/60日区块各2000次保持。这里“每方法6新”应读作三方法合计6新账户。完整中文进出场见 `docs/510300_FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V1.md`：有效月末评分、下一开盘入/增/减/清仓、缺失与非月末保持、清仓后下个有效月末重判、未成交不持续挂单、终点清算另标。

新顺序接续会话91733，PID{next_progress["pipeline_pid"]}，路径 `reports/research/510300_forward_eps_valuation_consistency_pipeline_v1`。代码 `scripts/continue_forward_eps_valuation_consistency_pipeline_20260907.py`。持有旧接续3920的真实句柄及02:29:18.999112创建时间，等待其pipeline/result.json和policy_v2/saved_numerical_verification.json同时完成，接着三步：新PE关系因子、第二十轮三方法完整账户、保存原件关系/模型/账户核对。每步日志完成凭证独立；failure须解释处理，不能重启旧过程或同时手动执行相同步骤。新接续也不自动更新完成轮次、交付新账户成绩或完成目标。

截至{progress["observed_at"]}原件归档{progress["archived_receipt_files"]}/6734；以实时状态为准。已完成18轮、评价186方法/192来源版本/438账户不变；登记方法192，登记来源版本203，第十九与第二十各3待运行。每轮真实完成后分别增加3已评价方法、3已评价来源版本、10评价账户，不能提前计入。

普通中文新交付 `{path.relative_to(ROOT).as_posix()}` 包含来源问题、七原件链接和三方法全部中文进出场；没有新GPT包。后续读取19、20真实结果解释覆盖、PE是否有用、预测兑现和持仓/退出。不能将来源问题一概归为解析错误，也不能把通过数学关系视为真估值。目标active，未达到1.2，没有独立证据。
"""
    addition = addition.replace("每方法6新加4复用共10评价", "三方法合计6新加4复用共10评价").replace("这里“每方法6新”应读作三方法合计6新账户。", "")
    with continuation.open("a", encoding="utf-8") as handle:
        handle.write(addition)
    print(json.dumps({"document": str(path), "registered_methods": 192, "registered_source_versions": 203,
                      "completed_rounds": 18, "evaluated_accounts": 438, "new_account_results": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
