"""登记九方案的来源与批准背景，完整保留账户统计、同文档未知和原金额。"""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from research.factor96_repurchase_9_plan_context_v1 import OUT, read, save, now, digest
from scripts.record_factor96_remaining_changes_v1 import update


def main():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    paths = [program/name for name in ("status.json", "strategy_progress.json", "factor_progress.json")]
    paths.append(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in paths]
    status, strategies, factors, mandate = objects
    assert not mandate["delivery_package_required"] and not mandate["orders_authorized"]
    protected_keys = ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
                      "cumulative_invalid_implementation_account_scenarios", "last_completed_account_experiment",
                      "last_completed_account_result", "independent_forward_observations", "completed_total_fixed_questions",
                      "qualified_candidates", "orders_authorized", "repurchase_positive_batches_with_explicit_purpose",
                      "repurchase_positive_batches_with_unknown_purpose")
    protected = {k: status[k] for k in protected_keys}
    before = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]
    relative = OUT.relative_to(ROOT).as_posix()
    stat = (ROOT/".env").stat()
    credential = {"at": now(), "check": "FILE_METADATA_ONLY_NO_SECRET_READ_OR_API_REQUEST",
                  "bytes": stat.st_size, "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                  "changed_since_prior_check": stat.st_size != 136 or datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() != "2026-09-05T04:24:46.438171+00:00"}
    save(OUT/"credential_metadata_check.json", credential, True)
    note = ("21个用途未知正向批次的9个原方案已逐项核实：19批次有原方案批准用途背景，2批次确认当时处于用途变更待股东大会批准。"
            "新增包钢变更、会前通知和决议3份原文，确认2024-12-10公开后才显示用途变更获批；不倒填11月及12月初。"
            "同文档用途567明确/21未知的口径不变，跨文件背景单列。自由流通分母仍缺失，0新回测。")
    for row in strategies:
        if row["id"] == "T12":
            row.update(current_status="NOT_RUN_FREE_FLOAT_AND_FULL_PURPOSE_VERSION_GATE", current_evidence=note,
                       current_evidence_path=relative+"/result.json", source_gate_path=relative+"/result.json")
    for row in factors:
        if row["id"] in ("M01", "M02"):
            row.update(current_status="REVIEWED_PRIOR_PLAN_CONTEXTS_AVAILABLE_STRICT_FACTOR_NOT_RUN", current_note=note,
                       current_evidence_path=relative+"/result.json")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key] = 0
    status.update(at=now(), latest_round=result["study_id"], latest_result=relative+"/result.json",
                  latest_progress_receipt=relative+"/result.json", last_source_result=note,
                  latest_continuation_classification="PROGRESS_NINE_ORIGINAL_PLAN_CONTEXTS_AND_APPROVAL_CHAIN_RESOLVED",
                  current_research_phase="T12_REVIEWED_PLAN_CONTEXTS_AVAILABLE_FREE_FLOAT_PENDING",
                  new_source_documents_this_round=3, new_searchable_text_documents_this_round=3,
                  reused_repurchase_pdf_documents_this_round=16, new_reviewed_original_plans_this_round=9,
                  new_reviewed_plan_document_facts_this_round=19, new_archived_source_http_responses_this_round=result["source_requests"],
                  source_field_candidates_this_round=19, admitted_account_scenarios_this_round=0,
                  invalid_implementation_accounts_this_round=0,
                  latest_repurchase_publication_batches=relative+"/publication_batches_with_prior_context.json",
                  latest_repurchase_focused_plan_history=relative+"/plan_document_facts.json",
                  latest_repurchase_prior_plan_contexts=relative+"/target_batch_plan_contexts.json",
                  latest_repurchase_focused_approval_chain=relative+"/baogang_approval_chain.json",
                  repurchase_unknown_document_batches_with_prior_plan_context=21,
                  repurchase_target_batches_confirmed_pending_purpose_approval=2,
                  repurchase_actionable_original_plan_reference_gaps_in_this_scope=0,
                  repurchase_actionable_approval_identity_bridge_gaps_in_this_scope=0,
                  same_document_unknown_purpose_is_not_unreviewed_source=True,
                  next_independent_source_action="T13_EXISTING_RIGHTS_CALENDAR_EVENT_IDENTITY_CHAINS",
                  source_field_candidate_kind="19个原方案及变更批准文档事实，21批次背景关联；不增加用途同文档覆盖率或账户样本。",
                  goal_status="active", goal_achieved=False, delivery_package_required=False)
    for k, v in protected.items():
        assert status[k] == v
    mandate.update(current_round=result["study_id"], current_protocol=relative+"/protocol.json",
                   latest_progress_receipt=relative+"/result.json", last_research_result=note, last_source_result=note,
                   latest_continuation_report=relative+"/研究进展.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    paragraphs = [
        "成本后净夏普1.2目标仍未实现，本轮没有新增回测。已核实剩余21个正向回购披露批次对应的9个原方案及批准背景，复用16份本地原文、定向取得3份新原文。没有制作交付包。",
        "21批次全部有当时已公开的原方案批准用途作为背景：19批次在已读文件范围内没有待批用途变更，另外2批次确认正在等待用途变更的股东大会批准。它们的执行公告同文档用途字段仍保持未知；跨文件原方案背景单列，没有把同文档覆盖率从567/588强行改成全覆盖。",
        "原方案背景分类为员工持股或股权激励13个批次、注销5个批次、激励或可转债转换两类用途3个批次。13个激励背景批次中包含包钢2个待变更批次：旧批准用途为激励，拟改为注销；不得将待批新用途当成已生效。",
        "包钢原方案无需股东大会审议，批准用途为股权激励或员工持股。2024-08-28公告明确用途变更须经股东大会通过才生效；2024-11-23会议通知把第1项议案指向8月28日披露的该事项；2024-12-10决议公告显示12月9日表决通过。原文链分别为1221005006、1221816066、1221965881。查询在12月10日公告代理时钟前仍保留待批，包括11月7日和12月4日两批执行，未把批准日期或后续公告倒填到此前。",
        "京东方2024年激励方案与2025年新注销方案保持两个方案身份。2025-04-22同时公布的旧方案用途变更提议只连旧方案，新注销方案当时也仍待股东大会批准。在本次已存证据范围内，2025-06-10报告书才确认新方案已于5月23日获批；没有拿报告书追述的5月24日公告日期冒充已取得该日原件。目标中5个新方案执行批次均晚于6月10日。",
        "润泽科技原方案允许用于员工持股、股权激励或转换可转债，3个执行批次保留两类规范化用途，金额分配未知；同文档中1元名义对价的重组业绩补偿回购注销属于另一个事项，未混入集中竞价回购。顺丰2025-10-31公开的金额扩大及期限延长保持其独立时钟，不能用于9月4日的首次执行背景。",
        "保存了19个文档事实、21个批次背景和17组公开边界前一秒/当时的方案状态。检查包括初始待批、后续用途变更待批、实际批准日与公开日不同、同公司同日两个方案及未来预算变更。946批次的既有金额、时钟、身份、同文档用途及隔离状态均未改写。",
        "以上回放只覆盖本轮已读方案文件，尚未证明全部期间用途变更覆盖，也没有获得历史首次发布版本证据。保留日期末代理等级与交易因子不准入标记。严格M01自由流通分母仍为空，T12没有绩效结论。",
        "本轮只检查了凭据文件的大小及修改时间，仍是136字节、2026-09-05T04:24:46.438171+00:00；没有读取或打印密钥，也没有重新请求已报40101过期的接口。有效研究账户552、无效实现152、总计704、合格候选0、独立前向观测0保持不变。",
        "这21批次的原方案引用和包钢批准身份桥接缺口已经处理，不应再次当作未读资料重复开展。下一项可独立推进的现有资料工作为T13配股公告事件身份与认购日历链；原T12/T13的自由流通分母仍需取得可用数据。",
        "可直接查看21批次原方案与批准背景.csv；详细证据在plan_document_facts.json，按时钟查询在asof_plan_boundary_queries.json，包钢链在baogang_approval_chain.json。所有输出保存在本地研究目录。",
    ]
    (OUT/"研究进展.md").write_text("\n\n".join(paragraphs)+"\n", encoding="utf-8")
    for p, value in zip(paths, objects):
        update(p, value)
    pd.DataFrame(strategies).to_csv(program/"18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program/"96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT/"program_update_receipt.json", {"at": now(), "before": before,
         "after": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
         "protected_counts_and_authority": protected, "new_accounts": 0, "goal_achieved": False,
         "delivery_package_created": False}, True)
    print("九方案原文与批准背景已登记，21个原方案引用缺口关闭；同文档未知、账户统计和交易权限保持原值。")


if __name__ == "__main__":
    main()
