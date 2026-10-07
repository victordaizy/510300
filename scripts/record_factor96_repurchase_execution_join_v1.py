"""记录执行批次和条款关联进展，不增加账户数，不把未归一化金额叫作完整M01。"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_repurchase_execution_join_v1 import OUT, digest, now, read, save
from scripts.record_factor96_remaining_changes_v1 import update


def main():
    assert not (OUT / "program_update_receipt.json").exists()
    result = read(OUT / "result.json")
    batches = read(OUT / "publication_batches.json")
    assert len(batches) == 946 and result["positive_increment_batches"] == 588
    known_positive = sum(b["positive_disclosed_increment"] and b["purpose_consensus"] is not None for b in batches)
    assert known_positive == 394
    program = ROOT / "reports/research/510300_factor96_program_v1"
    files = [program / name for name in ("status.json", "strategy_progress.json", "factor_progress.json")]
    files.append(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    objects = [read(p) for p in files]
    status, strategies, factors, mandate = objects
    before = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files]
    protected = {k: status[k] for k in ("cumulative_admitted_account_scenarios", "cumulative_executed_account_scenarios",
                                      "last_completed_account_experiment", "last_completed_account_result",
                                      "independent_forward_observations", "completed_total_fixed_questions")}
    assert not mandate["delivery_package_required"] and not mandate["orders_authorized"]
    relative = OUT.relative_to(ROOT).as_posix()
    note = ("948执行公告接成946个公开批次，588个有正向披露增量；其中394用途明确、194未知。"
            "新补32旧库存方案与执行根无交集；已读变更实际先于11条执行，均为天合光能延期。"
            "两组同日公告去重并保持金额不变；21条修订/身份冲突及隔离记录继续保留。"
            "完整M01仍因自由流通股本分母缺失而全空，0新账户，不救援旧成交额分母回购失败家族。")
    for row in strategies:
        if row["id"] == "T12":
            row.update(current_status="NOT_RUN_FREE_FLOAT_AND_PURPOSE_CLOCK_SOURCE_GATE", current_evidence=note,
                       current_evidence_path=relative + "/result.json", source_gate_path=relative + "/result.json")
    for row in factors:
        if row["id"] in ("M01", "M02"):
            row.update(current_note=note, current_evidence_path=relative + "/result.json",
                       current_status="DISCLOSURE_BATCHES_AND_PARTIAL_PURPOSE_CONTEXT_BUILT_STRICT_FACTOR_NOT_RUN")
    for key in list(status):
        if key.endswith("_this_round") and (key.startswith("new_") or key.startswith("source_field_candidates") or key.startswith("reused_")):
            status[key] = 0
    status.update(at=now(), latest_round=result["study_id"], latest_result=relative + "/result.json",
                  latest_progress_receipt=relative + "/result.json", last_source_result=note,
                  latest_continuation_classification="PROGRESS_EXECUTION_PUBLICATION_BATCHES_AND_TERM_CLOCK_JOINED",
                  current_research_phase="M01_NUMERATOR_BATCHES_READY_FREE_FLOAT_AND_PURPOSE_COVERAGE_PENDING",
                  admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
                  reused_repurchase_pdf_documents_this_round=7, new_execution_context_rows_this_round=948,
                  repurchase_publication_batches=946, repurchase_positive_disclosed_increment_batches=588,
                  repurchase_positive_batches_with_explicit_purpose=394, repurchase_positive_batches_with_unknown_purpose=194,
                  latest_repurchase_execution_context=relative + "/execution_context_rows.json",
                  latest_repurchase_publication_batches=relative + "/publication_batches.json",
                  source_field_candidate_kind="既有948执行条款关联与同公开日批次，无新增原文、自由流通数据或策略收益。",
                  goal_status="active", goal_achieved=False, delivery_package_required=False)
    for key, value in protected.items():
        assert status[key] == value
    mandate.update(current_round=result["study_id"], current_protocol=relative + "/protocol.json",
                   latest_progress_receipt=relative + "/result.json", last_research_result=note, last_source_result=note,
                   latest_continuation_report=relative + "/研究进展.md",
                   latest_continuation_classification=status["latest_continuation_classification"],
                   research_execution_state=status["current_research_phase"], goal_status="active", goal_achieved=False)
    lines = [
        "本轮没有实现成本后净夏普1.2，也没有新增回测。完成的是948条执行公告的条款关联和同公开时钟去重，仍只研究510300与人民币现金，没有制作交付包。",
        "948条公告归成946个方案/公开日末批次，588个有可确认的正向披露增量。全部批次中569个用途明确、377个未知；正向批次中394个用途明确、194个未知。多用途不分配不存在的金额比例，未知不由未来或旧原方案用途填补。",
        "两组同日公告分别是：润泽科技2025-08-06同时披露7月末未买入及8月5日首次回购10473100元，合并为一个10473100元新增批次；科大讯飞2026-09-01的首次与进展公告均报告30021592元，同一笔只记一次。合并前后披露金额不变，不能把同日两个文档视为两个独立历史样本。",
        "新补32个旧库存原方案与现有948执行的方案根没有交集。已读54份变更共有11个方案根与执行库重叠，但只有1份变更先于其后的执行：天合光能2025-05-24公开的延期，影响11条后续执行记录。其他变更没有被倒填到较早执行中。",
        "天合光能原方案2024-06-25董事会批准、期限12个月，旧截止日按该条款推算为2025-06-24，并明确标为日历推算。2025-05-24公开后才使用新的2026-03-24截止日和自有/自筹资金来源。2026-03-23实际完成这件事到3月25日才公开，在3月23日与24日的查询中均不显示已完成。",
        "旧表的47个起始基数未知、5个身份/修订冲突及16个后续隔离记录全部保留，不改为零、不跨方案求差。已披露正向增量是历史实施信息，不等于未来持续买盘。",
        "以前用成交额与权重归一化的回购需求日频家族及固定到期变体已经失败，本轮未重复或修改这些账户。原定义M01需要事件前自由流通市值；凭据仍过期，所有M01归一化值保持空白，T12仍未回测。",
        "有效研究账户情景552、无效实现情景152、合计704保持不变；合格候选0、独立前向观测0。下一项可独立进行的工作是核实194个正向批次中未知的用途及其来源口径；分母来源取得前不能开展原T12绩效检验。",
        "现行数据为execution_context_rows.json、publication_batches.json以及两张CSV；两个同日案例和期限变更边界查询分别保存在same_clock_examples.json、trina_term_clock_queries.json。",
    ]
    (OUT / "研究进展.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    for path, value in zip(files, objects):
        update(path, value)
    pd.DataFrame(strategies).to_csv(program / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(program / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    save(OUT / "program_update_receipt.json", {"at": now(), "before": before,
         "after": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files],
         "account_counts_unchanged": protected, "new_accounts": 0, "new_network_requests": 0,
         "goal_achieved": False, "delivery_package_created": False})
    print("执行批次与时钟关联已登记，588正向批次中194用途待核实；没有新增回测或交付包。")


if __name__ == "__main__":
    main()
