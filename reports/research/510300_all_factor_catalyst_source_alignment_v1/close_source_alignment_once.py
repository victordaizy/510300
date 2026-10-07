"""核对原评分/交易未改变，保存来源用途结论并一次更新长期事实。"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import all_factor_catalyst_source_alignment_v1 as study
from research import all_factor_joint_path_attribution_v1 as io

OUT = study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCE = ["latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
    "current_new_strategy_return_sharpe", "latest_actual_financial_primary_four_scene_metrics"]
FORWARD = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at", "current_validated_candidates",
    "forward_account_comparison_protocol", "latest_forward_account_check", "new_prospective_sessions_this_continuation",
    "new_prospective_cycles_this_continuation", "next_experiment"]
DOCS = ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]


def status():
    return subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT)


def unchanged_table(source, result, name):
    old = pd.read_parquet(source)
    new = pd.read_parquet(result)
    pd.testing.assert_frame_equal(old.reset_index(drop=True), new[list(old.columns)].reset_index(drop=True), check_dtype=False)
    return {"name": name, "source_rows": len(old), "original_columns_preserved_exact": True}


def verify():
    checks = [
        unchanged_table(io.SCORES, OUT / "results/全部13952共同评分_催化记录与未知.parquet", "全部四模型评分"),
        unchanged_table(study.CASES, OUT / "results/原全部30案例_催化钟与解释等级不变.parquet", "原30案例与解释等级"),
        unchanged_table(study.ROUTES, OUT / "results/全部22856实际决策_催化当时可知.parquet", "原22856实际决策"),
        unchanged_table(study.QUALIFICATION, OUT / "results/全部65原节点_15时05分首次可知与原动作.parquet", "原65节点与动作资格"),
        unchanged_table(study.CREDIT, OUT / "results/全部84月LPR_原预期与缺失不替代工具.parquet", "84月LPR实际/预期/缺失"),
    ]
    config = io.read(OUT / "protocol.json")
    unchanged = all(io.sha(ROOT / source["path"]) == source["sha256"] for source in config["sources"])
    io.require(unchanged, "来源匹配结束后固定输入改变。")
    data = pd.read_parquet(OUT / "results/全部3488原点_有记录与全覆盖未知分开.parquet")
    io.require(not data.all_policy_absence_known.any(), "缺少来源被误认作全样本无政策。")
    io.write(OUT / "verification_receipt.json", {
        "at": io.now(), "necessary_tests_passed": 5, "pytest_chunk": "6767f9", "source_hashes_unchanged": unchanged,
        "all_original_values_preserved": checks, "all_absence_states_remain_unestablished": True,
        "same_source_is_not_independent_votes": True, "financial_admission": "NOT_ADMITTED",
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "independent_validation": "NOT_ESTABLISHED",
    }, exclusive=True)


def close_once():
    io.write(OUT / "close_started.json", {"at": io.now(), "state_updates_planned": 1}, exclusive=True)
    original = io.read(STATE)
    original_git = status()
    financial = {key: copy.deepcopy(original.get(key)) for key in FINANCE}
    forward = {key: copy.deepcopy(original.get(key)) for key in FORWARD}
    observer = copy.deepcopy(original.get("independent_official_share_observer"))
    io.require(financial["latest_actual_financial_decision"] == "TECH.R256", "实际金融不是R256。")
    io.write(OUT / "state_before_TECH_R260.json", original, exclusive=True)
    io.require(io.read(OUT / "goal_service_status_after_delivery.json")["goal"]["status"] == "active", "目标服务状态不是active。")
    summary = io.read(OUT / "summary.json")
    updated, at = copy.deepcopy(original), io.now()
    updated.update({
        "updated_at": at, "status": "research_active", "goal_achieved": False,
        "latest_completed_study": OUT.absolute().relative_to(ROOT).as_posix(),
        "latest_result": "reports/research/510300_all_factor_catalyst_source_alignment_v1/summary.json",
        "latest_report": "reports/research/510300_all_factor_catalyst_source_alignment_v1/催化来源_完整共同原点与准入结论.md",
        "latest_research_status": summary["status"], "latest_registration_decision": "TECH.R259", "latest_technical_decision": "TECH.R260",
        "current_study": summary["study_id"], "current_phase": "CATALYST_SOURCE_ALL_JOINT_ORIGINS_COMPLETE_OLD_SOURCE_NUMERIC_USE_CLOSED",
        "new_accounts_in_current_phase": 0,
        "previous_goal_turn_classification": original.get("latest_goal_turn_classification"),
        "latest_goal_turn_classification": "PROGRESS_R259_R260_ACTUAL_SOURCE_ALIGNMENT_AND_COMPLETE_USE_CLOSURE",
        "latest_progress": "完成65原节点×3488日历与13952评分、22856实际请求的全量对应；可评分区间仅4宣布原点/2支持原点，旧资料不能开启可靠催化阶段数值用途。",
        "latest_continuation_outcome": "R259—R260有限来源实验终止：描述对应成立，完整政策分母/同工具新预期不具备，0新配置/拟合/账户；原R256金融拒绝保持。",
        "current_direction": "全因素讨论保留；旧65目录及八家族重复归档不再当新证据。后续需要真正不同且完整的信息用途或实际新独立点位。",
        "current_priority": "保留来源用途关闭；只有新增完整来源/不同完整机制或原协议真实新样本，才能重新登记比较；不反复扫描旧65节点。",
        "next_research_question": "是否实际出现具有不同依据的完整信息用途，或原前瞻协议所需的真实新完整收盘/来源版本？目前没有已准入待运行数值用途。",
        "latest_all_factor_catalyst_source_alignment": "reports/research/510300_all_factor_catalyst_source_alignment_v1/summary.json",
        "latest_catalyst_source_purpose_disposition": "TERMINAL_OLD_SOURCE_NUMERIC_USE_NOT_ADMITTED_R260",
        "next_all_factor_source_purpose_status": "COMPLETED_TERMINAL_NOT_ADMITTED_R260",
        "latest_continuation_receipt": "reports/research/510300_all_factor_catalyst_source_alignment_v1/project_state_update_receipt.json",
        "latest_continuation_audit": "reports/research/510300_all_factor_catalyst_source_alignment_v1/verification_receipt.json",
        "latest_goal_tool_status_receipt": "reports/research/510300_all_factor_catalyst_source_alignment_v1/goal_service_status_after_delivery.json",
        "goal_tool_status_confirmed": "active", "current_admitted_unrun_complete_uses": 0,
        "current_admitted_unrun_numeric_candidates": 0, "blocked_audit_count": 0, "goal_blocked_audit_count": 0,
        "blocked_reason": None, "blocked_audit_key": None, "current_goal_turn_blocker_id": None,
        "blocking_decision": "NONE_THIS_TURN_FINISHED_NEW_ALL_ORIGIN_SOURCE_EVIDENCE_AND_CLOSED_PURPOSE",
        "current_unmet_evidence": "R256固定联合金融四场景失败；R260旧催化源仅4可评分宣布原点/2支持原点，完整分母和新增同工具预期不具备。当前新独立完成点位0、已准入未运行配置0，提高收益/Sharpe未实现。",
    })
    block = """### TECH.R259—R260：催化来源与全因素评分全原点匹配（2026-10-06）

**最新实际金融仍为R256固定拒绝。R260完成R258声明的有限来源实验，0新拟合/账户/收益标签；接受全量描述对应，关闭旧催化资料直接开启数值评分的用途。目标未完成，goal实际active。**

假设→旧资料中的公开催化可能被共同模型遗漏，但是否足以提供新的完整可检验用途，须核对时间、分母、预期和既有失败。
验证方法→冻结原65节点与动作/共同来源/经济身份，按原3488日15:05实际日历映射首次可知；原16:00首次日期另列；全部13952模型评分、22856真实请求、30原案例与84月LPR完整绑定，八既有家族按实际结果保留。没有再下载或重训旧源。
结果→65节点/60共同来源/63经济身份，7宣布节点只来自6共同原文；原支持资格9节点。全日历56原点有目录新记录、6原点有宣布；每模型528可评分原点中，21原点有新目录记录、4原点有宣布、2原点有支持资格。15:05与原16:00首次日期差0，不能捏造时钟变化造成新机会。第一原点之前决策未知；所有3488日的完整无政策/无动作状态均未认证。
具体当时信息→2024/5/17 18:22公布上界只能映射5/20原点；2024/9/24两节点同一来源，模型均拒绝，不能当两票。2025/4/4日期上界映射4/7，仍保留非激活动作。2025/5/7 09:20:49宣布只在5/7原点可用，不能回填主模型5/6进入决定，即使5/7开盘前可知也不改变原前收盘决策。
接受/拒绝理由→接受全部当时来源与评分/真实路径对应为描述证据。拒绝使用该旧65目录训练新催化阶段评分：完整政策分母NOT_ESTABLISHED，新增同工具事前数量预期0，新独立来源0；84月LPR不是逆回购/降准预期。旧77月共识、非线性预期差、原支持/主线等失败不因新名称重开。
重新验证→本来源用途终止；不再重复盘点65节点、补无动作0、挑上涨案例、调原30字段树或逆转失败信号。只有真实新增且有完整采样规则的来源、具有不同依据的完整用途，或原协议实际到达的新独立点位，才重新登记比较。当前已准入未运行配置/用途均0，独立完成点位0；项目全因素研究和日周线多头目标继续，不是项目整体停止。
必要验证→5个时间/日历/共同来源/未知语义测试通过；原13952评分、22856决策、30解释案例、65资格节点、84月预期原值逐列保持；所有固定源哈希、金融五字段、前瞻十三字段、官方份额观察合同与已有Git状态保持。未创建采集自动化或新金融策略。
证据→`reports/research/510300_all_factor_catalyst_source_alignment_v1/summary.json`、`催化来源_完整共同原点与准入结论.md`、`results/全部65原节点_15时05分首次可知与原动作.parquet`、`results/全部13952共同评分_催化记录与未知.parquet`、`verification_receipt.json`。

"""
    documents = []
    for name in DOCS:
        path = ROOT / "docs" / name
        content = path.read_bytes()
        (OUT / (path.stem + "_before_TECH_R260.md")).write_bytes(content)
        title, old_body = content.split(b"\n", 1)
        insertion = ("\n" + block).encode("utf-8")
        final = title + b"\n" + insertion + old_body
        path.write_bytes(final)
        io.require(path.read_bytes()[len(title) + 1 + len(insertion):] == old_body, "项目文档原正文改变。")
        documents.append({"path": path.absolute().relative_to(ROOT).as_posix(), "old_body_preserved_exact": True, "sha256": io.sha(path)})
    io.write(STATE, updated)
    final_state = io.read(STATE)
    io.require(all(final_state.get(k) == v for k, v in financial.items()), "金融五字段改变。")
    io.require(all(final_state.get(k) == v for k, v in forward.items()), "前瞻十三字段改变。")
    io.require(final_state.get("independent_official_share_observer") == observer, "官方份额观察合同改变。")
    io.require(status() == original_git, "用户已有Git跟踪状态改变。")
    io.write(OUT / "project_state_update_receipt.json", {
        "at": at, "registration": "TECH.R259", "decision": "TECH.R260", "state_updates": 1,
        "classification": updated["latest_goal_turn_classification"], "financial_five_preserved_exact_R256": True,
        "forward_thirteen_preserved_exact": True, "official_share_contract_preserved_exact": True,
        "tracked_git_status_preserved_exact": True, "documents": documents, "new_fits": 0, "new_accounts": 0,
        "new_labels": 0, "new_independent_completed_points": 0, "goal_service_status": "active", "goal_achieved": False,
        "archive_writing_itself_is_not_research_progress": True}, exclusive=True)
    print("来源用途已终止归档，项目长期事实一次更新；金融仍R256拒绝，目标active未完成。")


if __name__ == "__main__":
    verify()
    close_once()
