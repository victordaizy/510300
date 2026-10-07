"""归档券商来源与阶段观察事实，保留旧正文、失败和独立前瞻字段。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research.broker_cycle_sources_v1 import OUT, ROOT, stamp, write_json

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD = (
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation", "next_experiment",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepend(path, text):
    raw = path.read_bytes()
    first, separator, body = raw.partition(b"\n")
    if not separator:
        raise ValueError("事实文件没有可保留的标题与正文。")
    backup = OUT / f"{path.stem}_before_TECH_R210.md"
    with backup.open("xb") as stream:
        stream.write(raw)
    prefix = ("\n" + text.strip() + "\n\n").replace("\n", "\r\n").encode("utf-8")
    updated = first + separator + prefix + body
    path.write_bytes(updated)
    saved = path.read_bytes()
    if saved != updated or not saved.endswith(body):
        raise ValueError("旧事实正文未能逐字节保留。")
    return {"path": str(path.absolute().relative_to(ROOT)), "old_sha256": digest(raw),
            "new_sha256": digest(saved), "old_body_preserved_exact": True, "inserted_bytes": len(prefix)}


def main():
    receipt_path = OUT / "project_state_update_receipt.json"
    if receipt_path.exists() or (OUT / "state_before_TECH_R210.json").exists():
        raise RuntimeError("本阶段归档已执行或开始，不再次插入事实。")
    phase = read(OUT / "phase_observation_summary.json")
    sources = read(OUT / "source_collection_receipt.json")
    follow = read(OUT / "guojin_public_pdf_follow_receipt.json")
    author = read(OUT / "source_author_and_method_review.json")
    service = read(OUT / "goal_service_status_resumed.json")
    if service["goal"]["status"] != "active" or phase["prefix_checks"] != 17 or author["verified_distinct_report_contents"] != 7:
        raise ValueError("实际目标状态、观察或来源阅读尚未完成。")
    report = "reports/research/510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md"
    summary = "reports/research/510300_broker_cycle_inspiration_v1/phase_observation_summary.json"
    proof = "reports/research/510300_broker_cycle_inspiration_v1/goal_service_status_resumed.json"
    note = f"""> 最新日周线目标事实（2026-10-05，TECH.R209—R210，优先于下方旧blocked快照）：用户明确要求策略随市场改变，参考招商‘主线·资金·博弈·周期’及其他券商研究；目标服务实际active，完整收益夏普目标未达。本轮9次公开报告本机请求8成功/1失败，4家7份不同报告正文（5原PDF/2作者转载）；先登记后解释原3488日、61分段/49上涨、四原案例240行及17关键日，17截断复算精确通过、三原输入哈希保持、4图已查看。ETF活跃/波动象限既含2015下跌也含2024启动，2019早期量与宏观未全面转强；接受进一步研究阶段内背景和分阶段进出，拒绝直接把象限命名牛熊或把ETF量当指数换手。0新拟合/标签/账户，新收益夏普NOT_COMPUTED；最新实际金融仍TECH.R208四经济门0/4及稳定失败，独立验证未建立。旧三轮blocked归档为上一周期，本恢复周期受阻计数0；原策略、旧失败与13项独立前瞻逐值保持。

最新研究边界：用户已授权另立阶段识别、宏观角色、进场、持有和退出机制；旧冻结用途不得改写，但不再将‘必须沿用旧退出’或‘必须另换数据源’解释成全部新实验的永久限制。市场状态驱动的预定改变可以作为策略；研究者看收益后新增规则必须另登记，不能叫事前自适应。小资金容量较低是可核实优势，不等于收益方向优势。原E03仍独立保留。

下一步：定义早期修复与趋势延续的当时可知判别、信息未知分支及失效；核对完整反例与主线对510300的覆盖，再固定一个主政策、同入场固定退出及同退出无阶段的归因对照，按同资金风险四场景完整运行。当前为机制假设，不是已登记待跑金融候选，不宣称已有新胜率或夏普。此前旧blocked恢复条件不覆盖本次用户新授权。

依据：[券商原报告启发、具体上涨和下一实验](../{report})；[来源与观察卡](510300_BROKER_CYCLE_INSPIRATION_V1.md)；[全部观察与实际检查](../{summary})；[目标服务active返回](../{proof})。
"""
    decisions = """### TECH.R209—R210：券商周期框架与分阶段策略研究（2026-10-05）

用户最新指令允许策略随市场状态改变，并要求阅读招商与多家券商报告。下面‘接受’区分接受研究假设和接受交易策略；本轮没有交易策略晋升，也没有新金融结果。旧R208及旧在线专家/状态路由失败保留。

| 方向 | 假设 | 验证方法 | 实际结果 | 接受/拒绝理由 | 是否重新验证 |
|---|---|---|---|---|---|
| 阶段驱动的策略变化 | 起势、乘势、转势与退势应有不同进入、持有及失效规则 | 读取4家7份报告并登记逐时观察；下一用途以完整账户及归因对照检验 | 来源与观察完成；新金融NOT_COMPUTED | 接受新研究方向，旧失败不构成永久停止探索；未接受为赚钱策略 | 是，完整规则、全日历反例、同风险账户及独立证据 |
| 活跃与波动直接四象限择时 | 双增加可直接认定可买牛市 | 全3488日及固定反例；ETF20/60活跃和波动，前缀复算17次 | 2015下跌、2024启动同象限；评价日历中867双增加日有419过去五日跌；49上涨确认四类都有 | 拒绝直接牛熊命名/直接下单，接受描述用途；ETF量也不等于指数换手 | 要成为策略须验证价格/资金背景、假启动和全部成本 |
| 宏观统一正向加分 | PMI、融资、周线全部改善才适合进入 | 原17日已知钟与四案例顺序；未来分阶段同池比较 | 2019/2024早期慢变量仍弱，2020分阶段变化；非因果和非预测证明 | 不接受为通用硬门；接受宏观与阶段交互的假设 | 是，需要事前定义、完整反例、成熟标签/实际账户 |
| 主线匹配与轮动收敛 | 主线覆盖宽基及扩散决定510300是否适合参与 | 国金原PDF与兴业作者正文；当时行业/成分版本和日线合同先行 | 来源机制可明确；本地数值与账户NOT_COMPUTED | 接受作为第二研究方向；行业超额不可算作510300收益 | 是，先补合格输入，禁止事后选赢家 |
| 分阶段持有/退出 | 初期修复与趋势延续可有不同退出，固定2R/20日不是永久规则 | 新主政策配同入场固定退出及同退出无阶段对照 | 机制设计，尚无新账户 | 接受另立用途，保留旧R208固定退出与失败；不能边看收益边改 | 是，先完成新金融协议，再一次全场景运行 |
| 小资金容量与频率 | 低容量约束有利于选择机会，更多小额交易也许不划算 | 原份额/成交额单位核对；2855日10万元/日额、冻结费用计算 | 比例中位0.00548%/最大0.06435%；10万往返约0.14%/0.28%，3000约0.4333%/0.5333%，刻度另计 | 接受容量与最低费事实；拒绝由小资金直接推出正期望或高夏普 | 盈利及实际开盘冲击需要后续账户/独立验证 |
| 固定年历周期与事后自适应 | 逢某年/固定五年足以提前确定阶段 | 区分报告叙事与可执行变量，检查既有状态路由失败 | 未形成可检验的逐时证据 | 不准入固定年份信号；不把事后切换赢家包装为自适应 | 有实质事前可知机制和完整检验才再立用途 |

本轮实质工作：9本机来源请求8成功、1华创TLS失败完整保留；5原PDF/2转载，原3488日/61分段/49上涨/240案例/17关键日，0新模型/标签/账户，17前缀精确不变。日历未知和原宏观钟保留，3输入哈希保持，4图已查看。收益、夏普、实际胜率/盈亏比NOT_COMPUTED，独立验证未建立。最新金融仍R208拒绝。目标服务实际active，旧blocked周期归档、本周期0，13项前瞻保持。

依据：[完整报告](../reports/research/510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md)、[用途卡](510300_BROKER_CYCLE_INSPIRATION_V1.md)、[观察协议与结果](../reports/research/510300_broker_cycle_inspiration_v1/phase_observation_summary.json)。
"""
    old_state_raw = STATE.read_bytes()
    with (OUT / "state_before_TECH_R210.json").open("xb") as stream:
        stream.write(old_state_raw)
    state = json.loads(old_state_raw.decode("utf-8-sig"))
    if state["latest_actual_financial_decision"] != "TECH.R208":
        raise ValueError("其他金融研究已更新当前事实，先协调再归档。")
    forward_before = {key: state[key] for key in FORWARD}
    previous = {key: state.get(key) for key in (
        "status", "goal_status", "consecutive_blocked_goal_turns", "blocked_key", "blocked_reason", "blocked_scope",
        "goal_blocked_at", "goal_tool_blocked_receipt", "latest_goal_tool_status_receipt", "latest_goal_continuation_check",
        "current_phase", "latest_technical_decision", "latest_registration_decision", "latest_result", "latest_report")}
    state["historical_blocked_cycle_before_TECH_R209_user_adaptive_strategy_resumption"] = previous
    state.update({
        "updated_at": stamp(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": stamp(),
        "latest_goal_tool_status_receipt": proof, "goal_status_reconciliation": "用户最新恢复并授权阶段驱动的新用途；服务实际active；旧blocked周期归档，本周期从0审查。",
        "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0, "blocked_key": None, "blocked_reason": None,
        "blocked_audit_key": None, "blocked_scope": None, "goal_blocked_at": None,
        "goal_tool_blocked_receipt": None, "latest_goal_tool_update_result": None,
        "latest_technical_decision": "TECH.R210", "latest_registration_decision": "TECH.R209",
        "latest_actual_financial_decision": "TECH.R208", "latest_actual_financial_result": previous["latest_result"],
        "latest_result": summary, "latest_report": report,
        "latest_progress": "4家7份券商报告、9本机请求8成功；原3488日/四案例/全部分段的时序观察完成，接受分阶段进出新机制设计。0新金融，不晋升策略。",
        "current_phase": "BROKER_CYCLE_FRAMEWORK_AND_KNOWN_STATE_DESCRIPTION_COMPLETED_NEW_POLICY_DESIGN",
        "goal_turn_classification": "PROGRESS_USER_ADAPTIVE_STRATEGY_DIRECTION_NEW_REPORT_SOURCES_AND_PHASE_DESCRIPTION",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 0,
        "current_phase_trial_accounting": {"scope": "TECH_R209_R210_REPORT_INTAKE_AND_STATE_DESCRIPTION_ONLY",
            "native_report_requests": sources["requests"] + 1, "successful_responses": sources["success"] + int(follow["status"].startswith("SAVED")),
            "broker_count": 4, "distinct_report_contents": 7, "new_market_requests": 0, "new_market_bars": 0,
            "new_fits": 0, "new_labels": 0, "new_accounts": 0, "known_daily_rows": 3488, "case_rows": 240,
            "episodes": 61, "admitted_episodes": 49, "prefix_checks": 17, "figures_viewed": 4},
        "current_goal_turn_actual_work": {"new_source_report_contents": 7, "new_state_observation_rows": 3488,
            "native_report_requests": 9, "source_failures_retained": 1, "new_accounts": 0, "new_fits": 0, "new_labels": 0},
        "next_information_source_proposal": "主线与510300匹配及行业轮动收敛的当时日线/分类版本合同；已有宏观用于阶段背景，不要求为另立策略必须另换数据源。",
        "next_isolated_strategy_design": {"hypothesis": "早期修复与趋势延续采用不同点位及当时继续/失效退出",
            "status": "MECHANISM_HYPOTHESIS_NOT_FULL_FINANCIAL_REGISTRATION",
            "comparisons": ["同入场固定退出", "同退出无阶段区分"], "current_financial_run": "NOT_RUN",
            "old_results_and_strategies_preserved": True, "old_forward_next_experiment_preserved": True},
        "latest_user_strategy_adaptation_direction": "允许按事前可知阶段改变进入、持有与退出；旧配置失败只约束该配置，新用途另登记；小资金容量优势与收益证据分开。",
    })
    docs = []
    for name in ("PROJECT_STATE", "PROJECT_STATE_TECHNICAL_LINE"):
        docs.append(prepend(ROOT / f"docs/{name}.md", note))
    for name in ("RESEARCH_DECISIONS", "RESEARCH_DECISIONS_TECHNICAL_LINE"):
        docs.append(prepend(ROOT / f"docs/{name}.md", decisions))
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved_state = read(STATE)
    if {key: saved_state[key] for key in FORWARD} != forward_before:
        raise ValueError("原独立前瞻字段改变。")
    write_json(receipt_path, {"at": stamp(), "status": "SAVED_SOURCE_DESCRIPTION_AND_USER_DIRECTION_OLD_FACTS_PRESERVED",
        "docs": docs, "old_state_sha256": digest(old_state_raw), "new_state_sha256": digest(STATE.read_bytes()),
        "forward_fields_preserved_exact": list(FORWARD), "prior_blocked_cycle_archived": previous,
        "current_goal_status": saved_state["goal_status"], "current_blocked_count": 0,
        "latest_financial_decision": saved_state["latest_actual_financial_decision"], "new_financial_metrics": "NOT_COMPUTED"})
    print("四份事实文档和目标状态已归档；旧正文逐字节、13项独立前瞻逐值保持。当前active，最新金融仍R208。", flush=True)


if __name__ == "__main__":
    main()
