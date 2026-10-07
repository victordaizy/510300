"""保存字段并集修复的结果和局限，更新当前研究进度。"""
from pathlib import Path
import json
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_financial_parser_replay_v2_0_1 import BASE, OUT
from research.factor96_earnings_cashflow_measurement_v1 import digest, now, save


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    assert not (OUT / "round_status.json").exists()
    source = read(OUT / "source_repair_result.json")
    measured = read(OUT / "measurement_replay_result.json")
    batch = read(BASE / "batch_complete.json")
    download = read(BASE / "download_complete.json")
    assert source["all_confirmed_regressions_passed"]
    assert source["fields_in_repair_scope"] == 2530
    assert source["unchanged_legacy_fields_outside_scope"] == 33143
    assert batch["parsed_documents"] == 2085
    assert measured["new_accounts"] == 0 and not measured["returns_read"]
    assert not (BASE / "source_repair_result.json").exists()
    assert (BASE / "replay_scope_invalidation_addendum.json").exists()
    counts = source["field_status_counts"]
    study = "510300_FACTOR96_FINANCIAL_SCOPE_EXTENSION_V2_0_1"
    status = {"at": now(), "study_id": study,
        "this_turn_classification": "PROGRESS_SOURCE_SCOPE_UNION_AND_UNCHANGED_MEASUREMENT_COMPLETE",
        "status": "SOURCE_VERSION_COMPLETE_T11_NOT_RUN", "target_fields": 2530,
        "original_summary_fields": 2529, "extra_previously_confirmed_fields": 1,
        "target_documents": 2112, "same_hash_parsed_documents": 2085,
        "field_status_counts": counts, "all_confirmed_source_regressions_passed": True,
        "old_v1": "TERMINATED_CONFIRMED_STANDALONE_STATEMENT_CONTAMINATION",
        "old_v2_replay": "FAILED_KNOWN_NON_SUMMARY_FIELD_OMITTED_PRESERVED",
        "new_downloads_in_scope_extension": 0, "new_pdf_parses_in_scope_extension": 0,
        "new_accounts": 0, "new_returns": 0, "new_admitted_trading_features": 0,
        "cumulative_admitted_account_scenarios": 432,
        "cumulative_invalid_implementation_accounts": 152,
        "cumulative_executed_account_scenarios": 584,
        "qualified_candidates": [], "T11": "NOT_RUN", "O02": "NOT_COMPUTED",
        "T11_cohort_definition": "FROZEN_REAL_MEASUREMENT_NOT_RUN_SOURCE_ADAPTER_REQUIRED",
        "goal_status": "active", "goal_achieved": False,
        "current_market_view": "NO_VIEW", "external_review": "NOT_PERFORMED",
        "orders_authorized": False}
    save(OUT / "round_status.json", status)
    annual = "\n".join(
        f"| {r['year']} | {r['trading_days']} | {r['joint_median']:g} | {r['coverage_median']:.1%} |"
        for r in measured["yearly_daily_coverage"] if r["year"] >= 2018)
    cases = "\n".join(
        f"| {r['announcement_id']} | {r['metric_id']} | {r['observed_cny']:,.2f} | 通过 |"
        for r in [*source["known_regressions"], source["new_scope_regression"]])
    report = f"""只操作510300实现完整账户成本后夏普1.2的目标尚未实现。本轮完成财务来源修复范围的补漏及原公式重算，新增策略账户为0。2,530个目标字段中，{counts['REPARSED_UNCHANGED_VALUE']:,}个重解析金额与旧值一致，{counts['REPARSED_CHANGED_VALUE']:,}个金额不同，{counts['NO_VIEW_UNRESOLVED_SOURCE_FIELD']:,}个保持未知并清除旧数值。金额不同包含显示精度差异，不能全部称为已确认错误。来源修复使后续研究可以使用明确标记的新版本，但尚未证明T11具有收益能力。

原V2在全部2,085份同哈希PDF解析结束后，重算于已确认反例检查处停止：原2,529字段按摘要提取路径选取，遗漏了北方华创2022Q3总资产这一项已确认的非摘要路径错误。旧值40元，原文正确值40,895,706,832.37元；原V2已保存同一PDF的正确解析结果，却未把该字段列入替换范围。V2.0.1取原范围与全部7项已确认错误的并集，仅新增这一字段，共2,530项。文档范围仍是2,112份，PDF解析器、2,085份解析结果与财务公式均保持原样，没有新增下载或PDF解析。旧V2失败代码、日志和范围否定在相邻原目录中保留，不能称旧V2重算成功。

PDF来源批次使用1,161份本地同哈希原文和924份新取得的同哈希原文；另27份没有形成可用原文并保持未知。原V1与V2总逻辑请求不超过2,106次，V2新请求为{download['logical_http_requests']}次。本地解析进程曾在1,839份已保存结果后消失，终止原因未查明；随后在冻结恢复方案下仅解析缺少的246份，检查既有结果和来源记录字节不变。该恢复不增加网络请求。

更早的V1修复仍被否定：白云山2020年报的公司单体经营现金流778,464,955.25元曾被误标为合并值；正确合并值为585,185,023.09元。V2对无合并前缀的独立报表切断继承，识别本年列，仅显式续表允许继承表头。规则是在见到真实错误后制定。V1部分批次、终止记录和旧值没有覆盖。

全部8项已确认反例在最终字段版本中的金额如下；这不是全样本人工核验。

| 公告ID | 字段 | 本版金额（元） | 已确认原文检查 |
|---|---|---:|---|
{cases}

新版仍有35,673个三类财务字段；范围外33,143个字段沿用旧档案，未全部重解析。field_change_ledger列明2,530个目标的旧值、新值、范围和定位；repaired_verified_facts保留legacy列。未恢复的275个字段不能用旧值填缺，也不能解释为公司数值为零。

财务公式、历史依赖和时序保持原冻结定义：单季归母利润由累计值作季度差分，以前季资产缩放；L02以前两年同季度观测的样本标准差归一化，零标准差保持未知；L04按真正TTM计算现金流与利润之差相对资产的同比变化。报告使用名义披露后的首个交易日、最新报告前沿与200日报告年龄，金融业排除，历史成员和行业日期代理不变。12,483个成员公告事件中，L02可算{measured['L02_measurable_events']:,}个、L04可算{measured['L04_measurable_events']:,}个、两项均可算{measured['joint_measurable_events_after_source_repair']:,}个。旧版6,977个机械完整事件已随来源错误被否定，不能和本版拼接。

| 年份 | 交易日 | 每日两项可算公司数中位 | 相对非金融成员覆盖中位 |
|---|---:|---:|---:|
{annual}

本轮未读取市场收益、运行T11账户或计算O02。累计432个正式账户情景加152个否定实现情景，共584个，合格候选仍为空；独立前向观测仍为0。原库完成7个固定问题，加1个日更变体为8个问题。来源工程不能算作策略通过。目标继续为510300.SH与人民币现金、20万元主账户和2万元对照、成本后夏普至少1.2、年化至少10%、目标最大回撤不超过10%，其余已授权账户约束保持。

T11披露日行业财务篮子的定义已单独冻结并通过9项时序与缺失回归测试，实际聚合尚未运行；其旧来源接入脚本需要另存版本接入本次修复结果，聚合公式不随结果改变。O02观察窗、两年日更参考、入场和完整账户规则仍须在读取T11收益前冻结。未满足时序与来源条件的原始严格T11保持未准入；若研究名义日期代理变体，应明确区分其局限。

公告时钟与更正候选的完整本地包随source_evidence一并保留：173,449条规范公告中173,398条只有午夜时间，统一retrieved_at是重建时刻，不能当历史首次见到时刻。历史成员范围内发现567条同公司同报告期的更正或更新候选，尚未确认全部修订关系和金额方向。旧档案会排除带更正、取消等当前标题的记录，但标题状态实际变化时刻缺失。上述问题没有因同哈希原文而消失。

其他限制：归母净利润和合并现金流的权益口径不完全一致；两年前同季只有两个标准化基准；原文显示精度影响微小差分；重述和合并范围变化未统一；行业可用时刻仍为日期代理；范围外旧字段尚未全验；当前下载同字节PDF不能证明历史首次公开版本。资料覆盖截至2026年8月14日，当前市场NO_VIEW。核验回执只证明所列身份与固定结果可重算，不等同独立经济验证或外部审阅。

直接证据：本目录replay_protocol、replay_freeze、effective_targets、scope_extensions、source_repair_result、measurement_replay_result、逐字段及逐依赖表；原V2目录中的batch_complete、execution_recovery_01和replay_scope_invalidation_addendum。交付包的根索引与全新解压只读重算回执另行生成。外部审阅NOT_PERFORMED，未上传、无订单权限。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    prompt = """请先核对本包字段范围补漏和来源局限，再评价后续研究优先级。目标是只交易510300与现金，完整成本后夏普至少1.2、年化至少10%、目标回撤不超过10%；本轮新增账户0，目标未实现，T11未运行、O02未计算。

请检查V1合并/单体报表错误、V2遗漏北方华创2022Q3总资产的失败和V2.0.1并集补漏是否完整保留。核对全部8项已确认反例、2,530字段替换范围、275项未知清除旧值，以及33,143项范围外沿用旧值的边界。金额不同不能全部称错误更正。检查本地恢复是否只补缺少的246份、既有结果和请求是否不变。

请复核单季拆分、TTM、两年前同季标准化、日期代理、依赖时序、报告年龄和行业选择；批评首次公开版本、更正关系与标题状态时钟不完整的影响。区分机械可算、可交易信息与独立经济验证。已冻结的行业披露篮子尚未实际测量，不要将其当作有效策略。

请提出可落实的下一步与停止条件：来源接入、行业披露篮子测量、O02首轮反应、两年日更规则、完整账户检验、家族比较和独立验证。不得事后挑盈利年份、改方向或门槛救援冻结失败。文件核验、回归测试和大样本字段数不证明夏普1.2；本包未外部审阅，不授权订单。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    current = read(program / "status.json")
    assert current["cumulative_executed_account_scenarios"] == 584 and not current["qualified_candidates"]
    result_path = (OUT / "round_status.json").relative_to(ROOT).as_posix()
    source_text = "2085份原PDF解析完成；补入原V2遗漏的1个已确认错误字段，共2530字段；8项反例通过；原公式重算完成；T11账户仍未运行。"
    current.update({"at": now(), "latest_round": study, "latest_result": result_path,
        "latest_progress_receipt": result_path,
        "latest_continuation_classification": status["this_turn_classification"],
        "source_field_candidates_this_round": 2530,
        "new_source_documents_this_round": 924, "new_parsed_financial_documents_this_round": 2085,
        "source_field_candidate_kind": "原2529摘要字段与已确认错误并集补漏；未形成T11交易因子",
        "latest_financial_parser_scope_result": (OUT / "source_repair_result.json").relative_to(ROOT).as_posix(),
        "latest_financial_parser_scope_checkpoint": result_path,
        "latest_earnings_measurement": (OUT / "measurement_replay_result.json").relative_to(ROOT).as_posix(),
        "active_source_controller": None, "completion_controller_directory": None,
        "last_source_result": source_text})
    save(program / "status.json", current, exclusive=False)
    for name, ids in [("factor_progress.json", {"L02", "L04"}), ("strategy_progress.json", {"T11"})]:
        items = read(program / name)
        for item in items:
            if item["id"] in ids:
                item.update({"current_status": "SOURCE_SCOPE_UNION_AND_MEASUREMENT_COMPLETE_T11_NOT_RUN",
                    "current_evidence": source_text, "current_evidence_path": result_path,
                    "individual_performance": None})
        save(program / name, items, exclusive=False)
        frame = pd.DataFrame(items)
        for csv_name in [name.replace(".json", ".csv"),
                         "96因子当前进度.csv" if name.startswith("factor") else "18策略当前进度.csv"]:
            frame.to_csv(program / csv_name, index=False, encoding="utf-8-sig")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update({"current_round": study, "latest_progress_receipt": result_path,
        "current_protocol": (OUT / "replay_protocol.json").relative_to(ROOT).as_posix(),
        "latest_continuation_report": (OUT / "研究结论.md").relative_to(ROOT).as_posix(),
        "latest_continuation_classification": status["this_turn_classification"],
        "research_execution_state": "SOURCE_SCOPE_UNION_COMPLETE_DELIVERY_VERIFICATION_PENDING_T11_NOT_RUN",
        "active_source_controller": None, "last_source_result": source_text})
    save(mandate_path, mandate, exclusive=False)
    after = OUT / "program_after"
    after.mkdir(exist_ok=False)
    for path in program.iterdir():
        if path.is_file():
            shutil.copy2(path, after / path.name)
    shutil.copy2(mandate_path, after / "mandate.json")
    for name in ["report", "verify", "package"]:
        path = ROOT / "scripts" / f"{name}_factor96_financial_parser_scope_v2_0_1.py"
        shutil.copy2(path, OUT / "code" / path.name)
    frozen = [p for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    save(OUT / "result_freeze.json", {"at": now(), "files": [
        {"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in frozen]})
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
