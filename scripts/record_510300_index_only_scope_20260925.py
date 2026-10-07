"""登记用户撤回期权范围；保留已完成研究，未运行四腿实验停止。"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_index_only_resume_20260925"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    state_path = ROOT / "reports/research/510300_integrated_research_continuation_20260924/current_status.json"
    mandate, state = read(mandate_path), read(state_path)
    save(OUT / "previous_mandate.json", mandate, True)
    studies = []
    for folder in ["510300_protected_put_spread_daily_v1", "510300_put_spread_buyback_cost_daily_v1",
                   "510300_option_credit_spread_router_daily_v1", "510300_option_paired_state_value_daily_v1"]:
        result_path = ROOT / "reports/research" / folder / "result.json"
        result = read(result_path)
        studies.append({"study_id": result["study_id"], "path": result_path.relative_to(ROOT).as_posix(),
                        "status": result["status"], "primary": result["primary"],
                        "retained_as_history_only": True, "active_candidate": False})
    condor_code = ROOT / "research/option_protected_iron_condor_daily_v1.py"
    condor_root = ROOT / "reports/research/510300_option_protected_iron_condor_daily_v1"
    assert not (condor_root / "RUN_STARTED.json").exists()
    receipt = {"at": timestamp, "user_instruction": "不做期权了，我只做指数",
               "active_assets": ["510300.SH", "CASH_CNY"], "option_research_authorized": False,
               "high_sharpe_objective_continues": True, "training_window_calendar_years": 2,
               "update_frequency": "EVERY_TRADING_DAY", "single_trade_rr3_required": False,
               "account_tail_risk_retained": True, "new_market_collection_enabled": False,
               "orders_authorized": False, "completed_option_studies": studies,
               "unexecuted_condor": {"status": "NOT_RUN_CANCELLED_BY_USER_SCOPE_CHANGE",
                                     "source": condor_code.relative_to(ROOT).as_posix(),
                                     "sha256": hashlib.sha256(condor_code.read_bytes()).hexdigest(),
                                     "protocol_frozen": (condor_root / "freeze.json").exists(), "accounts": 0}}
    save(OUT / "authority_update.json", receipt, True)
    mandate.update(latest_user_instruction=receipt["user_instruction"], executable_assets=receipt["active_assets"],
                   option_research_authorized=False, latest_asset_scope_revision_at=timestamp,
                   latest_asset_scope_receipt=(OUT / "authority_update.json").relative_to(ROOT).as_posix(),
                   current_round="510300_INDEX_ONLY_RESUME_20260925", goal_achieved=False,
                   latest_progress_receipt=(OUT / "authority_update.json").relative_to(ROOT).as_posix(),
                   last_research_result="用户撤回期权研究；四项已完成期权实验保留为历史，铁鹰未冻结未运行。继续纯510300指数ETF与现金研究。")
    mandate["archived_option_account_tail_risk_contract"] = mandate.pop("option_account_tail_risk_contract", {})
    save(mandate_path, mandate)
    state.update(updated_at=timestamp, active_assets=receipt["active_assets"], option_research_authorized=False,
                 latest_research_workflow="INDEX_ONLY_RESEARCH_RESUMED_OPTIONS_CANCELLED",
                 latest_goal_turn_classification="PROGRESS_USER_NARROWED_SCOPE_TO_INDEX_ONLY",
                 same_condition_consecutive_no_progress_goal_turns=0,
                 active_blocker_id=None, active_blocker_description=None,
                 remaining_research_question="仅510300指数ETF和现金，核对已有状态研究后推进具有明确区别的账户检验。",
                 latest_asset_scope_receipt=(OUT / "authority_update.json").relative_to(ROOT).as_posix(),
                 completed_option_studies_archived=[row["study_id"] for row in studies],
                 option_condor_status="NOT_RUN_CANCELLED_BY_USER_SCOPE_CHANGE")
    state["pending_option_research_scope"] = {"status": "REVOKED_BY_USER", "receipt": mandate["latest_asset_scope_receipt"]}
    save(state_path, state)
    note = "用户最新要求只做指数。当前可研究与模拟资产已恢复为510300.SH和现金；期权后续研究停止。四项已完成期权实验没有合格策略，原结果保留且不列为当前候选。刚写好的四腿铁鹰程序没有冻结、没有训练、没有运行账户，状态为NOT_RUN_CANCELLED_BY_USER_SCOPE_CHANGE。\n\n高夏普目标、20万元本金、最近两年训练、每日更新和账户尾部风险仍然有效。既有行情采集暂停状态与无订单授权保持。\n"
    (OUT / "范围变更.md").write_text(note, encoding="utf-8")
    report = state_path.parent / "最新研究结论.md"
    report.write_text(f"2026-09-25最新范围：{note}\n---\n\n以下为按时间保留的旧研究记录，其期权范围说明已被最新用户指令覆盖。\n\n" + report.read_text(encoding="utf-8"), encoding="utf-8")
    print("已恢复510300指数ETF与现金范围；期权研究停止，四腿实验未运行。")


if __name__ == "__main__":
    main()
