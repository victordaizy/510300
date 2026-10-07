"""登记正文用途复核的实际进度，保持账户结果和交易权限不变。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from research.factor96_repurchase_body_purpose_v1 import OUT, now, read, save, digest
from scripts.record_factor96_remaining_changes_v1 import update


def main():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    assert result["positive_batches_with_reported_purpose"] == 567
    # 更正抽取数量字段的描述，不改变逐份判断、原始记录和聚合结果。
    old_receipt = read(OUT / "source_review_receipt.json")
    if "candidate_contexts_read" in old_receipt:
        corrected = dict(old_receipt)
        corrected["candidate_contexts_extracted"] = corrected.pop("candidate_contexts_read")
        corrected["correction_at"] = now()
        corrected["superseded_field"] = "candidate_contexts_read"
        corrected["correction_note"] = "332为自动抽取片段总数，不声称逐段全文阅读332个片段；实际复核是195份用途条款、20份无明确片段全文追读及特殊案例追读。"
        save(OUT / "source_review_receipt_v1_0_1.json", corrected, True)
    program = ROOT / "reports/research/510300_factor96_program_v1"
    files = [program / name for name in ("status.json", "strategy_progress.json", "factor_progress.json")]
    files.append(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in files]
    status, strategies, factors, mandate = objects
    before = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files]
    protected_keys = ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
                      "cumulative_invalid_implementation_account_scenarios", "last_completed_account_experiment",
                      "last_completed_account_result", "independent_forward_observations", "completed_total_fixed_questions",
                      "qualified_candidates", "orders_authorized")
    protected = {k: status[k] for k in protected_keys}
    assert not mandate["delivery_package_required"] and not mandate["orders_authorized"]
    relative = OUT.relative_to(ROOT).as_posix()
    note = ("194个正向批次的195份同文档用途已复核，新增明确174份公告/173个批次；"
            "588正向批次中567个用途明确、21个未知（19同文档未说明、2变更批准链未确认）。"
            "22份条件性后备注销未误作当前注销；3份表列维护价值与股份注销安排分开保存。"
            "自由流通分母仍缺失，M01全空、T12未回测，0新账户。")
    for row in strategies:
        if row["id"] == "T12":
            row.update(current_status="NOT_RUN_FREE_FLOAT_AND_PURPOSE_CLOCK_SOURCE_GATE", current_evidence=note,
                       current_evidence_path=relative + "/result.json", source_gate_path=relative + "/result.json")
    for row in factors:
        if row["id"] in ("M01", "M02"):
            row.update(current_note=note, current_evidence_path=relative + "/result.json",
                       current_status="SAME_DOCUMENT_REPORTED_PURPOSE_IMPROVED_STRICT_FACTOR_NOT_RUN")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key] = 0
    status.update(at=now(), latest_round=result["study_id"], latest_result=relative + "/result.json",
                  latest_progress_receipt=relative + "/result.json", last_source_result=note,
                  latest_continuation_classification="PROGRESS_194_POSITIVE_BATCH_PURPOSES_REVIEWED",
                  current_research_phase="M01_FREE_FLOAT_MISSING_21_POSITIVE_PURPOSE_BATCHES_UNRESOLVED",
                  admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
                  reused_repurchase_pdf_documents_this_round=195, new_reviewed_execution_purpose_documents_this_round=195,
                  new_resolved_execution_purpose_documents_this_round=174, new_resolved_positive_purpose_batches_this_round=173,
                  repurchase_explicit_execution_purposes=743, repurchase_unknown_execution_purposes=205,
                  repurchase_positive_batches_with_explicit_purpose=567, repurchase_positive_batches_with_unknown_purpose=21,
                  latest_repurchase_execution_context=relative + "/execution_context_rows.json",
                  latest_repurchase_publication_batches=relative + "/publication_batches.json",
                  latest_repurchase_same_document_purpose_review=relative + "/reviewed_source_rows.json",
                  repurchase_positive_purpose_coverage_ratio=567/588,
                  repurchase_positive_purpose_unknown_reasons={"NO_SAME_DOCUMENT_STATEMENT": 19, "AMENDMENT_APPROVAL_NOT_CONFIRMED": 2},
                  source_field_candidate_kind="195份既有执行公告的同文档用途条款，无新增数据来源、自由流通分母或策略收益。",
                  goal_status="active", goal_achieved=False, delivery_package_required=False)
    for key, value in protected.items():
        assert status[key] == value
    mandate.update(current_round=result["study_id"], current_protocol=relative + "/protocol.json",
                   latest_progress_receipt=relative + "/result.json", last_research_result=note, last_source_result=note,
                   latest_continuation_report=relative + "/研究进展.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    paragraphs = [
        "本轮未达到成本后净夏普1.2，也没有新增回测。完成的是194个正向回购披露批次的同文档用途复核：588个正向批次中，用途明确数由394升至567，覆盖率由67.01%升至96.43%，仍有21个未知。按照用户要求，没有制作交付包。",
        "固定范围包含195份公告、79个方案和60家公司。原解析器对其中149份没有标准表头的公告、46份菜单不完整或符号未识别的公告保留了未知。本轮按原文补充174份公告，对应173个公开批次；科大讯飞同日两份公告仍合并为一个批次。",
        "保留未知的21份/批次中，19份执行公告全文没有写明股份用途，只引用其他方案公告；另2份包钢公告只在本文件报告董事会用途变更，股东大会批准链未由该文确认。没有借用较晚完成公告或其他方案填补未知，也没有把审批未明直接写成已生效。",
        "新补用途的174份公告包括：员工持股或股权激励91份、减少注册资本62份、转换可转债8份、注销加激励两类用途7份、维护公司价值6份。这是文档数量，不是独立交易样本数量；同一方案有多份月度进展。",
        "22份文档的条件性后备注销与当期用途分开。格力两份公告明确不少于70%的回购股份注销、其余不超过30%用于激励，这只是股份数量界限，未按70/30拆分已支出金额。海大五份公告同时列出注销与激励，未虚构分配比例。",
        "中国石化、中远海控、药明康德三份公告的用途表勾选维护公司价值，正文则明确股份全部注销；表列用途与股份处置分别保存。万泰生物、阿特斯公告中另一个历史方案的激励库存没有混入本次注销用途。新产业两次回购股份合并作为激励来源也没有被重记为新买入。",
        "对恒力石化、紫金矿业四份符号异常公告以及药明康德一份双语义公告查看了原PDF第一页。保留195份精确条款、页码和原文位置；自动抽取332段候选上下文，实际逐份判断记录见reviewed_source_rows.json。抽取数量的准确字段说明见source_review_receipt_v1_0_1.json。",
        "946个总批次目前742个有同文档报告用途、204个未知；948条执行公告中743条有报告用途、205条未知。未对非正向批次的其余未知做无依据填补。全库金额、差分、日期、方案身份和修订冲突隔离保持前轮值，12个重点文档做了公开时钟前一秒与当时的查询检查。",
        "这些字段仍是历史目录日期末代理下的同文档报告用途，不构成完整历史首次披露时钟或全部用途生效版本链。自由流通股本数据服务此前实际返回40101凭据过期；本轮没有重复请求。严格M01值仍全空，T12仍未运行，旧回购失败账户保持原结论。",
        "研究账户数量维持有效552、无效实现152、合计704；合格候选0、独立前向观测0。后续要推进原T12，仍需口径一致的历史自由流通股本，以及把剩余21项用途缺口与当时已公开的原方案、变更批准证据相连。只操作510300与现金的范围没有改变。",
    ]
    (OUT / "研究进展.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    for path, value in zip(files, objects):
        update(path, value)
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": now(), "before": before,
         "after": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files],
         "protected_account_and_authority_fields": protected, "goal_achieved": False,
         "new_accounts": 0, "new_network_requests": 0, "delivery_package_created": False}, True)
    print("用途复核进展已登记：567/588正向批次用途明确，21项保留未知；账户数量与权限不变，没有交付包。")


if __name__ == "__main__":
    main()
