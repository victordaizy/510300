"""仅同步目标服务已返回的blocked状态与恢复条件；不继续研究。"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import all_factor_joint_path_attribution_v1 as io

CONTEXT = Path(__file__).absolute().parent
STATE = CONTEXT / "state.json"
FINANCE = ["latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
    "current_new_strategy_return_sharpe", "latest_actual_financial_primary_four_scene_metrics"]
FORWARD = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at", "current_validated_candidates",
    "forward_account_comparison_protocol", "latest_forward_account_check", "new_prospective_sessions_this_continuation",
    "new_prospective_cycles_this_continuation", "next_experiment"]
DOCS = ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]


def main():
    old = io.read(STATE)
    folder = (ROOT / old["latest_no_progress_check"]).parent
    actual = io.read(folder / "goal_status_blocked.json")
    audit = io.read(folder / "record.json")
    io.require(actual["goal"]["status"] == "blocked", "目标服务尚未返回blocked。")
    io.require(audit["consecutive_no_progress_count"] == 3 and audit["blocked_status_eligible_now"], "三次相同阻碍条件未满足。")
    io.write(folder / "blocked_status_sync_started.json", {"at": io.now(), "goal_service_status": "blocked"}, exclusive=True)
    protected = {key: copy.deepcopy(old.get(key)) for key in FINANCE + FORWARD + ["independent_official_share_observer"]}
    git_before = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT)
    io.write(folder / "state_before_blocked_status_sync.json", old, exclusive=True)
    at = io.now()
    block = """### 当前目标状态：blocked，R260后连续三轮同一阻碍（2026-10-06）

**目标服务实际已将“请继续，提高夏普率和收益率”设为blocked；目标未完成。此状态表示当前无法在合格证据条件下继续推进，不是收益/Sharpe通过，也不是用户要求暂停。**

接续事实→R260完成来源匹配后，三次不同目标轮次均实际重新核对。第三次原观察器于2026-10-06 15:01:36+08确认新完整收盘0、最新可完成交易日2026-09-30、下一原交易日2026-10-08；实际新自然完成点位0、官方份额归一版本0，已准入未运行完整用途/数值候选0。最近R256/R258/R260均terminal，采集句柄不存在；不是运行中任务的等待。
研究结论→最新金融保持TECH.R256固定联合拒绝，四场景经济门未通过。近期STRESS净年化-0.7125%、净Sharpe -0.440314、实际净pB0.219372，独立验证未建立。旧65催化目录仅4可评分宣布原点/2支持原点；其完整分母和新增同工具预期未具备，R260数值用途关闭。原全因素12类83项目录、30案例和全部反例/未知保留。
受阻理由→缺少已具备完整来源依据的不同用途，且真实新收盘/官方份额版本/独立完成点位尚未到达。重复状态、重扫旧目录或调已冻结失败的阈值/权重/时期，不能满足原收益、全日历净Sharpe、实际净pB/净期望和去过拟合要求。
恢复条件→实际合格新收盘/来源版本或新独立样本到达，或获得具有不同依据且可完整检验的新用途后再恢复。原最早新收盘时点10月8日15:05、份额首采窗10月8日23:05；需要真实输入，不能把日期当作数据已到。未创建自动采集或自动恢复任务。
保留范围→20万元、510300与现金、日线/前已完成周、多头优先、次数软目标、相同费用/风险、完整收益和Sharpe及独立验证要求不变。金融五字段、前瞻十三字段、官方份额观察合同和旧文档正文保持；新策略金融结果不存在。
证据→`reports/research/510300_daily_weekly_goal_continuation_20261001/continuation_audits/post_R260_3_20261006_150211_313007/record.json`及`goal_status_blocked.json`。前三次分类均NO_PROGRESS、相同阻碍；此状态同步本身不算研究进展。

"""
    documents = []
    for name in DOCS:
        path = ROOT / "docs" / name
        previous = path.read_bytes()
        (folder / (path.stem + "_before_blocked_status_sync.md")).write_bytes(previous)
        title, body = previous.split(b"\n", 1)
        insert = ("\n" + block).encode("utf-8")
        new = title + b"\n" + insert + body
        path.write_bytes(new)
        io.require(path.read_bytes()[len(title) + 1 + len(insert):] == body, "旧项目文档正文改变。")
        documents.append({"path": str(path.absolute().relative_to(ROOT)), "old_body_preserved_exact": True, "sha256": io.sha(path)})
    state = copy.deepcopy(old)
    state.update({"updated_at": at, "status": "research_blocked", "goal_achieved": False,
        "goal_tool_status_confirmed": "blocked", "blocking_decision": "BLOCKED_AFTER_THREE_ACTUAL_CONSECUTIVE_SAME_CONDITION_CHECKS",
        "latest_goal_tool_status_receipt": str((folder / "goal_status_blocked.json").absolute().relative_to(ROOT)),
        "current_blocked_scope": "提高收益/完整净Sharpe及去过拟合独立验证的当前路径：缺少已准入不同完整用途或真实新证据。",
        "current_blocked_reentry_condition": "真实合格新收盘/官方份额版本/独立点位到达，或不同完整用途获得完整来源依据后恢复；不重扫旧65目录或救已冻结失败参数。",
        "latest_continuation_outcome": "R260后连续三轮NO_PROGRESS，实际目标服务已blocked；完整目标未完成，金融R256和所有旧结果/观察合同保持。",
        "latest_blocked_status_sync_receipt": str((folder / "blocked_status_sync_receipt.json").absolute().relative_to(ROOT)),
        "background_monitor_running": False})
    io.write(STATE, state)
    after = io.read(STATE)
    io.require(all(after.get(k) == v for k, v in protected.items()), "blocked同步改变金融/前瞻/份额合同。")
    io.require(subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT) == git_before, "已有Git跟踪状态改变。")
    io.write(folder / "blocked_status_sync_receipt.json", {"at": at, "actual_goal_service_status": "blocked",
        "goal_achieved": False, "consecutive_same_blocker_turns": 3, "financial_five_preserved_exact_R256": True,
        "forward_thirteen_preserved_exact": True, "official_share_contract_preserved_exact": True,
        "tracked_git_status_preserved_exact": True, "documents": documents,
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_source_requests": 0,
        "status_sync_is_not_research_progress": True}, exclusive=True)
    print("目标服务blocked已同步项目状态和四份长期文档；目标未完成，研究工作停止。")


if __name__ == "__main__":
    main()
