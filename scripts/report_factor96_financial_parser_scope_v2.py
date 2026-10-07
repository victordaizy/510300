"""V2来源批次与原公式重算完成后才生成结论，不将源修复当作策略通过。"""
from pathlib import Path
import json
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_financial_parser_batch_v2 import OUT, PRIOR, STUDY, digest, now, read, save


def main():
    assert not (OUT / "round_status.json").exists() and (OUT / "measurement_replay_result.json").exists()
    assert not (OUT / "parser_semantic_invalidation_addendum.json").exists()
    source = read(OUT / "source_repair_result.json")
    measurement = read(OUT / "measurement_replay_result.json")
    batch = read(OUT / "batch_complete.json")
    download = read(OUT / "download_complete.json")
    protocol = read(OUT / "batch_protocol.json")
    changes = pd.read_parquet(OUT / "field_change_ledger.parquet")
    assert len(changes) == 2529 and source["all_confirmed_regressions_passed"]
    assert batch["new_accounts"] == measurement["new_accounts"] == 0
    assert measurement["T11"] == "NOT_RUN" and not measurement["returns_read"]
    counts = changes.status.value_counts().to_dict()
    status = {"at": now(), "study_id": STUDY, "this_turn_classification": "PROGRESS_V2_SOURCE_REPAIR_AND_UNCHANGED_MEASUREMENT",
        "status": "SOURCE_VERSION_COMPLETE_T11_NOT_RUN", "target_fields": 2529, "target_documents": 2112,
        "field_status_counts": counts, "all_confirmed_source_regressions_passed": True,
        "prior_parser": "V1_TERMINATED_CONFIRMED_STANDALONE_STATEMENT_CONTAMINATION",
        "new_accounts": 0, "new_returns": 0, "cumulative_admitted_account_scenarios": 432,
        "cumulative_invalid_implementation_accounts": 152, "cumulative_executed_account_scenarios": 584,
        "qualified_candidates": [], "new_admitted_trading_features": 0, "T11": "NOT_RUN", "O02": "NOT_COMPUTED",
        "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "round_status.json", status)
    annual = measurement["yearly_daily_coverage"]
    table = "\n".join(f"| {r['year']} | {r['trading_days']} | {r['joint_median']:g} | {r['coverage_median']:.1%} |"
                      for r in annual if r["year"] >= 2018)
    known = "\n".join(f"| {c['announcement_id']} | {c['metric_id']} | {c['expected_cny']:,.2f} | {c['observed_cny']:,.2f} |"
                      for c in [*source["known_regressions"], source["new_scope_regression"]])
    report = f"""只操作510300实现成本后夏普1.2的目标尚未实现。本轮完成财报来源新版本和原公式覆盖重算，没有新增策略账户。冻结范围为2,112份公告、2,529个字段；其中{counts.get('REPARSED_CHANGED_VALUE', 0)}个字段金额与旧档案不同，{counts.get('REPARSED_UNCHANGED_VALUE', 0)}个重解析金额一致，{counts.get('NO_VIEW_UNRESOLVED_SOURCE_FIELD', 0)}个仍保持未知。金额变化不能全部称为错误更正，也包括显示精度差异。

原先的七条金额反例已由V2取得对应原文值。第一次修复V1在批量抽查中又暴露了合并口径错误：白云山2020年报中，第158页公司单体现金流778,464,955.25元被误标为合并值；第152页合并表实际为585,185,023.09元。V1已停止，旧代码、部分批次、原文及否定记录全部保留，原定V1覆盖重算没有执行。V2切断无合并前缀的独立报表状态，识别本年列，只有显式续表可以继承前表列头。规则是在发现该反例之后制定，不能称为对此反例事前未知。

V2接管1,175份已完成回执；4个终止时未完成的尝试不重试；尚未尝试的原计划最多932次HTTP请求，本次实际新请求{download['logical_http_requests']}次。原V1与V2的合并逻辑请求不超过原定2,106次。已有失败、版本不一致和中断都保持未知。成功字段要求当前PDF的原字节身份与旧档案完全一致；本次抓取时刻和历史名义披露日分开保存。

22项金额、表头和报表边界回归及4项接管行为测试通过；七份真实PDF覆盖原七条错误与新增口径反例。以下金额是对具体已确认反例的核对，不能据此宣称全档案均已人工复核。

| 公告ID | 字段 | 原文期望值（元） | V2取值（元） |
|---|---|---:|---:|
{known}

新版本中共有{source['fields_in_version']:,}个字段，冻结摘要来源范围外的{source['unchanged_legacy_fields_outside_scope']:,}个字段沿用旧档案，并单独标明。只替换2,529个目标字段；未取得或有冲突的字段清除旧数值，不以旧错值填缺。field_change_ledger.parquet记录全部目标的旧值、新值、状态和来源；repaired_verified_facts.parquet保留legacy_*列及新来源定位。

修正后按完全相同的单季利润、上一季资产、前两年同季标准化、真实TTM、名义披露后首个交易日、最新报告前沿及200日报告年龄重算。成员公告中，两项均可计算的事件为{measurement['joint_measurable_events_after_source_repair']:,}个，L02为{measurement['L02_measurable_events']:,}个，L04为{measurement['L04_measurable_events']:,}个。此处仍是测量完整性，不是交易有效率。旧版发现错误前的6,977个机械完整事件及其来源否定保留在历史包中，没有覆盖。

| 年份 | 交易日 | 每日两因子可计算公司数中位 | 相对非金融成员覆盖中位 |
|---|---:|---:|---:|
{table}

本轮不读市场收益、不计算T11入场、不训练新的状态、不形成账户。累计正式432个加否定实现152个，共584个账户情景不变，合格候选仍为空。原库已完成7个固定问题，加一个日更变体共8个问题，不能把来源修复算作第9个策略通过。只交易510300.SH与人民币现金，20万元完整账户、成本后夏普至少1.2、年化至少10%和目标回撤不超过10%的要求保持。

下一步须在读取T11收益前固定行业处理、ETF聚合、O02首个完整交易日反应、两年日更状态及完整账户规则，并检查实际信息可用时钟。O02当前NOT_COMPUTED，T11当前NOT_RUN。若来源覆盖或时序不能支持固定定义，应保持NO_VIEW或停止该路径；不得调收益阈值、挑年份或恢复已失败的策略。

保留的限制包括：范围外旧字段没有全部重解析；显示精度不同可能影响微小差分；归母利润与合并现金流不是完全相同权益口径；两年前同季只有两个基准观测；报告重述及合并范围变化未统一；行业可用时刻仍为供应商日期代理；当前取得的同哈希原文不证明历史首次公开版本。数据覆盖截至2026年8月14日，不能形成当前市场观点。

直接证据见batch_protocol.json、batch_freeze.json、replay_protocol.json、replay_freeze.json、source_repair_result.json、measurement_replay_result.json及全部逐字段/逐依赖/逐日结果。source_handoff保存V1失败和接管依据；V1完整阶段记录在相邻510300_factor96_financial_parser_repair_v1目录。只读重算和交付结构核查将由独立回执记录，不能代替外部审阅。外部审阅NOT_PERFORMED，无订单权限。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    prompt = """请评审本包财报来源修复及同公式测量，不要把数据工程当成夏普目标实现。目标仍为只交易510300与现金，20万元完整账户成本后夏普至少1.2、年化至少10%、目标回撤不超过10%。本轮新增账户0，T11未运行，O02未计算。

请先看研究结论和V1语义否定，核对七个原错误、白云山合并/单体现金流反例及V2表头/单位/时段状态。检查终止后是否重复已尝试URL、失败字段是否清除旧值、所有2529目标是否按同一规则处理，以及范围外33144旧字段的限制是否被诚实保留。金额变化可能包含显示精度差异，不应全部归类为已确认错误。

请批评季度差分、真正TTM、前两年同季标准化、名义披露日代理和历史首次版本边界；检查依赖、报告年龄、行业口径和缺失是否导致选择偏差。请提出具体下一步：T11行业归一化/ETF聚合、O02首日反应、两年日更规则、完整账户固定检验及独立验证。给出优先级、准入条件和停止条件。不得通过挑盈利子期、缩尾隐藏来源问题或调门槛救援冻结失败。文件哈希和机械重算通过不能证明策略有效；本包没有外部审阅，也不授权任何订单。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    current = read(program / "status.json")
    assert current["cumulative_executed_account_scenarios"] == 584 and not current["qualified_candidates"]
    current.update({"at": now(), "latest_round": STUDY, "latest_result": (OUT / "round_status.json").relative_to(ROOT).as_posix(),
        "new_source_documents_this_round": download["status_counts"].get("DOWNLOADED_SAME_HASH_PDF", 0),
        "new_parsed_financial_documents_this_round": batch["parsed_status_counts"].get("PARSED", 0),
        "new_searchable_text_documents_this_round": 0,
        "new_archived_source_http_responses_this_round": sum('http_status' in read(path) for path in (OUT / "fetch_receipts").glob("*.json")),
        "source_field_candidates_this_round": 2529,
        "source_field_candidate_kind": "同原PDF重解析来源版本；范围外沿用旧字段；未形成T11交易因子",
        "latest_financial_parser_scope_result": (OUT / "source_repair_result.json").relative_to(ROOT).as_posix(),
        "latest_earnings_measurement": (OUT / "measurement_replay_result.json").relative_to(ROOT).as_posix()})
    save(program / "status.json", current, exclusive=False)
    for filename, ids in [("factor_progress.json", {"L02", "L04"}), ("strategy_progress.json", {"T11"})]:
        items = read(program / filename)
        for item in items:
            if item["id"] in ids:
                item.update({"current_status": "SOURCE_VERSION_AND_MEASUREMENT_COMPLETE_T11_NOT_RUN",
                    "current_evidence": "V2来源范围重解析及原公式覆盖已完成；旧V1口径错误保留否定；ETF聚合/O02/交易账户尚未运行。",
                    "current_evidence_path": (OUT / "round_status.json").relative_to(ROOT).as_posix(), "individual_performance": None})
        save(program / filename, items, exclusive=False)
        pd.DataFrame(items).to_csv(program / ("96因子当前进度.csv" if filename.startswith("factor") else "18策略当前进度.csv"), index=False, encoding="utf-8-sig")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update({"current_round": STUDY, "latest_progress_receipt": (OUT / "round_status.json").relative_to(ROOT).as_posix(),
        "current_protocol": (OUT / "batch_protocol.json").relative_to(ROOT).as_posix(),
        "latest_continuation_report": (OUT / "研究结论.md").relative_to(ROOT).as_posix(),
        "latest_continuation_classification": status["this_turn_classification"],
        "research_execution_state": "V2_SOURCE_VERSION_COMPLETE_T11_NOT_RUN",
        "last_source_result": "V1合并口径反例保留；V2同范围重解析和原公式重算完成；无新策略账户。"})
    save(mandate_path, mandate, exclusive=False)
    after = OUT / "program_after"
    after.mkdir(exist_ok=False)
    for path in program.iterdir():
        if path.is_file():
            shutil.copy2(path, after / path.name)
    shutil.copy2(mandate_path, after / "mandate.json")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    frozen = [path for path in sorted(OUT.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
    save(OUT / "result_freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in frozen]})
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
