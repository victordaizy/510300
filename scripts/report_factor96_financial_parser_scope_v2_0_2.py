"""完成财报来源范围与数字边界修复的结论和当前进度。"""
from pathlib import Path
import json
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_financial_parser_replay_v2_0_2 import BASE, OUT, PROBE, SOURCE
from research.factor96_earnings_cashflow_measurement_v1 import digest, now, save


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    assert not (OUT / "round_status.json").exists()
    source = read(OUT / "source_repair_result.json")
    measured = read(OUT / "measurement_replay_result.json")
    probe = read(PROBE / "result.json")
    assert source["all_confirmed_regressions_passed"] and source["all_boundary_regressions_passed"]
    assert source["fields_in_repair_scope"] == 2535 and source["unchanged_legacy_fields_outside_scope"] == 33138
    assert not measured["returns_read"] and measured["new_accounts"] == 0
    counts = source["field_status_counts"]
    study = "510300_FACTOR96_FINANCIAL_SCOPE_EXTENSION_V2_0_2"
    status = {"at": now(), "study_id": study,
        "this_turn_classification": "PROGRESS_NUMERIC_BOUNDARY_REPAIR_AND_UNCHANGED_FINANCIAL_REPLAY",
        "status": "SCOPED_SOURCE_VERSION_COMPLETE_T11_NOT_RUN", "target_fields": 2535,
        "original_summary_fields": 2529, "confirmed_scope_omission_fields": 1, "numeric_boundary_candidates": 5,
        "target_documents": 2117, "same_hash_parsed_documents": 2090,
        "field_status_counts": counts, "known_eight_regressions_passed": True,
        "additional_five_boundary_values_passed": True, "all_legacy_values_verified": False,
        "previous_v2_0_1_measurement": "PRESERVED_BUT_BLOCKED_FOR_T11_NEW_LEGACY_SOURCE_ERRORS",
        "new_accounts": 0, "new_returns": 0, "new_admitted_trading_features": 0,
        "cumulative_admitted_account_scenarios": 432, "cumulative_invalid_implementation_accounts": 152,
        "cumulative_executed_account_scenarios": 584, "qualified_candidates": [],
        "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "independent_forward_observations": 0,
        "T11_cohort_definition": "FROZEN_REAL_MEASUREMENT_NOT_RUN_SOURCE_ADAPTER_REQUIRED",
        "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "round_status.json", status)
    evidence = [*source["known_regressions"], source["new_scope_regression"], *source["boundary_regressions"]]
    cases = "\n".join(f"| {r['announcement_id']} | {r['metric_id']} | {r['observed_cny']:,.2f} | 通过 |" for r in evidence)
    annual = "\n".join(f"| {r['year']} | {r['trading_days']} | {r['joint_median']:g} | {r['coverage_median']:.1%} |"
                       for r in measured["yearly_daily_coverage"] if r["year"] >= 2018)
    report = f"""只操作510300实现完整账户成本后夏普1.2的目标尚未实现。本轮完成2,535个财务字段的范围修复和固定公式重算，新增策略账户为0。字段中{counts['REPARSED_UNCHANGED_VALUE']:,}项与旧值在1.1分容差内一致、{counts['REPARSED_CHANGED_VALUE']:,}项金额不同、{counts['NO_VIEW_UNRESOLVED_SOURCE_FIELD']:,}项保留未知并清除旧值。金额差异也包括显示精度，不能全部称已确认错误。范围外33,138项旧字段未全部重新核验。

本轮沿三次实质反证推进，原结果全部保留。V1解析器曾将白云山2020年报公司单体经营现金流778,464,955.25元误作合并值，正确合并值585,185,023.09元；V1已被否定。V2修正报表状态并完成2,085份同哈希PDF解析，但2,529个摘要路径目标漏掉已确认错误的北方华创2022Q3总资产，原重算停止。V2.0.1只补这一字段为2,530项并完成重算，随后在范围外旧字段的财务极值诊断中发现新错误，因此该版本测量已标记不能接入T11。

新增核查按旧来源定位中数值跨度后仍有千分组的统一模式扫描33,143项未修复字段，得到5个候选，再固定这5份原文的同哈希核查。5份均取得、每份一次HTTP。3项明显错误已逐页目视确认：中航电子2019Q3归母净利润266,339元应为266,339,389.64元；亨通光电2019FY总资产41,247,482元应为41,247,482,963.25元；东方明珠2020Q3归母净利润307,273,154.641元应为1,307,273,154.64元。另两项差异不足1分，仍在完整候选范围内统一用原文值重建。V2.0.2共处理2,535字段，解析器和经济公式不改，不用目视确认的常量替代通用解析结果。

原来源批次共有2,112个文档目标：1,161份本地同哈希复用、924份新同哈希原文，27份缺失或失败保持未知。原V1与V2总逻辑请求不超过2,106次。解析进程在1,839份已保存结果后消失，原因未确定；冻结恢复计划后仅补246份缺失解析，既有1,839份结果与4,227份来源IO记录字节不变。新增边界核查的5份为另行冻结的不同文档；连同原批次共2,117个目标、2,090份同哈希可用原文。没有重复请求原批次27份未知。

以下列示原8项已确认反例和新增5项边界候选。本表只证明这些具体字段与保存原文相符，不代表全档案人工复核。

| 公告ID | 字段 | 本版金额（元） | 核对 |
|---|---|---:|---|
{cases}

财务规则保持原冻结定义：单季归母利润按累计值作季度差分、以前季资产缩放；L02相对前两年同季的均值和样本标准差，零波动保持未知；L04使用真正TTM现金流与归母利润之差相对资产，并计算同比变化。成员、行业、披露名义日期代理、最新报告前沿、历史依赖及200日报告年龄都保持。{measured['target_member_report_events']:,}个成员公告事件中，L02可算{measured['L02_measurable_events']:,}个，L04可算{measured['L04_measurable_events']:,}个，两项同时可算{measured['joint_measurable_events_after_source_repair']:,}个。这是测量完整性；可算事件数相同也可能有重要金额变化，不能代替数值核对。

| 年份 | 交易日 | 每日两项可算公司数中位 | 相对非金融成员覆盖中位 |
|---|---:|---:|---:|
{annual}

本轮未读取市场收益、计算O02或运行T11账户。累计正式432个加否定实现152个，共584个账户情景；合格候选仍为空，独立前向观测0。原库已完成7个固定问题，加1个日更变体为8个，不能把来源修复算作新增策略通过。目标继续为510300.SH与人民币现金、20万元主账户和2万元对照、成本后夏普至少1.2、年化至少10%、目标回撤不超过10%，既有风险与执行约束保持。

T11的行业披露篮子定义已冻结并通过9项时序、缺失及聚合测试，实际测量尚未运行；接入本版来源需要独立保存来源适配版本。O02首轮反应、两年日更规则、入场、失效和完整账户仍须在读取T11收益前冻结。原每日240家存量广度模型及其失败保持；披露日新报告篮子不能被描述为补齐该旧覆盖要求。

公告时钟与更正候选包一并保留。173,449条规范公告中173,398条仅有午夜时刻，统一retrieved_at为重建时间；历史成员范围内567条同公司同报告期的更正候选尚不等同确认修订链；当前标题中的取消/更新状态未建立历史变更时刻。名义日期代理、当前同哈希原文和机械重算都不能证明历史首次可得。

其他限制：范围外33,138项旧字段未全部重新核对；本次数值边界扫描只覆盖具备精确定位的一种模式；归母利润与合并现金流权益口径不同；两年前同季只有两个标准化基准；报表重述、显示精度及合并范围变化未统一；行业时钟为供应商日期代理。资料截至2026年8月14日，不形成当前市场观点。回归和只读重算不能代替经济有效性或外部审阅。

证据入口为本目录protocol、replay_freeze、input_identity、source_repair_result、measurement_replay_result及完整字段/依赖/逐日表；旧V2、V2.0.1目录保存各自失败和原结果；legacy_token_boundary目录保存5份原文与原页核对。交付回执另记录文件索引、哈希、CRC和全新解压只读重算。外部审阅NOT_PERFORMED，未上传、无订单权限。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    prompt = """请评审本包财务来源修复、保留反证和固定公式测量。目标仍为只交易510300与现金、完整成本后夏普至少1.2、年化至少10%、目标回撤不超过10%；本轮新增账户0、T11未运行、O02未计算，目标未实现。

先核对V1合并口径错误、V2遗漏已确认字段的失败、V2.0.1范围外旧数字错误与V2.0.2完整5项候选处理。检查2,535字段变化、275项未知清除旧值、33,138项范围外沿用旧值，以及全部13项具体来源检查的适用边界。不得把896项金额变化都称为错误更正。5份新增核查各一次请求，原失败批次没有重试；旧结果不能覆盖成成功。

请批评单季拆分、TTM、两年前同季标准化、名义日期代理、行业和成员时序、依赖、年龄、重述和首次版本缺口。区分机械可算与当时可交易信息。判断仍需核对的最小关键证据，避免无目的扩大审计。

请给出可执行的下一步优先级、准入条件与停止条件：行业披露篮子、O02首轮反应、两年日更映射、完整账户比较、家族选择偏差处理和独立验证。不得挑盈利子期或调门槛救援冻结失败。哈希和固定结果重算通过不证明夏普目标，也不授权订单；本包未外部审阅。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    current = read(program / "status.json")
    assert current["cumulative_executed_account_scenarios"] == 584 and not current["qualified_candidates"]
    result_path = (OUT / "round_status.json").relative_to(ROOT).as_posix()
    text = "原8项反例及新增5项数字边界候选已在2535字段版本中核对；3项新明显金额错误修正；原公式重算完成，T11账户仍未运行。"
    current.update({"at": now(), "latest_round": study, "latest_result": result_path, "latest_progress_receipt": result_path,
        "latest_continuation_classification": status["this_turn_classification"],
        "source_field_candidates_this_round": 2535, "new_source_documents_this_round": 929,
        "new_parsed_financial_documents_this_round": 2090,
        "source_field_candidate_kind": "限定来源修复与财务测量；范围外旧字段并非全量核验；无交易因子准入",
        "latest_financial_parser_scope_result": (OUT / "source_repair_result.json").relative_to(ROOT).as_posix(),
        "latest_financial_parser_scope_checkpoint": result_path,
        "latest_earnings_measurement": (OUT / "measurement_replay_result.json").relative_to(ROOT).as_posix(),
        "active_source_controller": None, "completion_controller_directory": None, "last_source_result": text})
    save(program / "status.json", current, exclusive=False)
    for name, ids in [("factor_progress.json", {"L02", "L04"}), ("strategy_progress.json", {"T11"})]:
        items = read(program / name)
        for item in items:
            if item["id"] in ids:
                item.update({"current_status": "SCOPED_SOURCE_AND_MEASUREMENT_COMPLETE_T11_NOT_RUN",
                    "current_evidence": text, "current_evidence_path": result_path, "individual_performance": None})
        save(program / name, items, exclusive=False)
        for csv_name in [name.replace(".json", ".csv"), "96因子当前进度.csv" if name.startswith("factor") else "18策略当前进度.csv"]:
            pd.DataFrame(items).to_csv(program / csv_name, index=False, encoding="utf-8-sig")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update({"current_round": study, "latest_progress_receipt": result_path,
        "current_protocol": (OUT / "protocol.json").relative_to(ROOT).as_posix(),
        "latest_continuation_report": (OUT / "研究结论.md").relative_to(ROOT).as_posix(),
        "latest_continuation_classification": status["this_turn_classification"],
        "research_execution_state": "V2_0_2_SOURCE_COMPLETE_DELIVERY_VERIFICATION_PENDING_T11_NOT_RUN",
        "active_source_controller": None, "last_source_result": text})
    save(mandate_path, mandate, exclusive=False)
    after = OUT / "program_after"
    after.mkdir(exist_ok=False)
    for path in program.iterdir():
        if path.is_file():
            shutil.copy2(path, after / path.name)
    shutil.copy2(mandate_path, after / "mandate.json")
    for prefix in ["report", "verify", "package"]:
        path = ROOT / "scripts" / f"{prefix}_factor96_financial_parser_scope_v2_0_2.py"
        shutil.copy2(path, OUT / "code" / path.name)
    frozen = [p for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    save(OUT / "result_freeze.json", {"at": now(), "files": [
        {"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in frozen]})
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
