"""归档阶段账户的实际失败和局部事实，保留原事实正文及前瞻约定。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research import broker_stage_policy_study_v1 as study
from research import point_first_passage_study_v1 as original
from research.close_broker_cycle_inspiration_v1 import FORWARD

ROOT, OUT = study.ROOT, study.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
MAINLINE = ROOT / "reports/research/510300_broker_mainline_source_preflight_v1"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def prepared_prepend(path: Path, text: str) -> tuple[bytes, bytes, bytes]:
    old = path.read_bytes()
    title, separator, body = old.partition(b"\n")
    if not separator:
        raise ValueError("事实文件没有标题与原正文，停止归档。")
    prefix = ("\n" + text.strip() + "\n\n").replace("\n", "\r\n").encode("utf-8")
    new = title + separator + prefix + body
    if not new.endswith(body):
        raise ValueError("事实正文未能精确保留。")
    return old, new, body


def main() -> None:
    receipt_path = OUT / "project_state_update_receipt.json"
    if receipt_path.exists() or (OUT / "state_before_TECH_R212.json").exists():
        raise RuntimeError("阶段结果归档已执行或开始，不重复写入。")
    summary = study.read(OUT / "summary.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    preflight = study.read(OUT / "control_preflight.json")
    tests = study.read(OUT / "tests_receipt.json")
    protocol = study.read(OUT / "protocol.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    mainline = study.read(MAINLINE / "summary.json")
    if (summary["decision"] != "TECH.R212" or summary["new_accounts"] != 12
            or summary["saved_account_checks"] != 12 or delivery["saved_accounts_recomputed"] != 12
            or tests["passed"] != 9 or tests["exit_code"] != 0
            or not all(row["original_daily_orders_trades_exact"] for row in preflight["checks"])
            or service["goal"]["status"] != "active" or summary["goal_achieved"] is not False):
        raise ValueError("完整金融结果、保存复算、原A精确核对或实际目标状态不一致。")
    if summary["all_economic_gates_passed"] or summary["historical_stability_passed"]:
        raise ValueError("裁决与准备归档的拒绝事实不一致。")
    own = {"research/broker_stage_policy_inputs_v1.py", "research/broker_stage_policy_study_v1.py",
           "tests/test_broker_stage_policy_v1.py", "docs/510300_BROKER_STAGE_POLICY_V1.md"}
    frozen = []
    for item in protocol["sources"]:
        path = ROOT / item["path"]
        if relative(path) in own:
            actual = digest(path.read_bytes())
            if actual != item["sha256"]:
                raise ValueError("本轮冻结代码、测试或用途卡发生改变。")
            frozen.append({"path": relative(path), "sha256": actual})
    if len(frozen) != 4:
        raise ValueError("四份本轮冻结文件记录不全。")
    before_state = STATE.read_bytes()
    state = json.loads(before_state.decode("utf-8-sig"))
    if state["latest_actual_financial_decision"] != "TECH.R208" or state["latest_technical_decision"] != "TECH.R210":
        raise ValueError("当前目标事实已被其他研究更新，需先核实。")
    forward_before = {key: state[key] for key in FORWARD}
    report = relative(OUT / "阶段点位_完整结果与失败归因.md")
    result = relative(OUT / "summary.json")
    source_report = relative(MAINLINE / "主线匹配_来源覆盖与时钟检查.md")
    proof = relative(OUT / "goal_service_status_after_result.json")
    note = f"""> 最新日周线完整金融事实（2026-10-05，TECH.R211—R212，优先于下方R210未运行与旧blocked快照）：按券商阶段思路固定三路线进入、EARLY→TREND真实持仓确认及上一完整周结构失效，主政策4账户、同进入固定退出4账户、无阶段资格同退出4账户共12新账户一次完成；原A4账户净值/订单/周期精确复现，9必要测试、12保存账户复算通过，51关键日/720案例行及4图已查看。当前固定政策拒绝：四经济门0/4、稳定门失败，完整收益夏普目标未达。目标服务实际active，本轮PROGRESS、连续受阻0；原策略、旧失败及13项独立前瞻逐值保持。

压力费用的同口径结果：2015—2019主政策净年化2.1871%/夏普0.52919/DD6.3228%/27完成、12胜15负/实际B2.07181/pB0.92080/完整年均5.4次；原A1.8371%/0.43804/pB0.61094/年均4.4。2020—2026主政策−0.2400%/−0.06608/DD7.1595%/44完成、12胜32负/B2.60110/pB0.70939/年均6.667；原A3.9908%/1.21691/pB1.07290/年均4.167。基础费与全部归因控制、20/252日区块各2000次区间一并保存，不选有利时期或尺度。

具体事实与解释：主政策2024-09-24决定、09-25开盘进入、09-30确认持有阶段、10-17开盘退出，完成净回报12.1365%、盈利3647.23568元；不能由此单笔推出全期策略有效。阶段持有相对同进入固定退出四场景CAGR/Sharpe点值均改善，多数区间跨零，不构成稳定晋升。近期回踩8笔全亏，保留全部；不据结果删路线。近期实际数量毛损益3983.80元、摩擦7079.20元、净−3095.40元；加回摩擦的固定数量年化0.3043%仍远低于原A扣费3.9908%，主要须提高机会区分，不能只降低费用救援。

主政策风险与未完成：较早压力平均暴露12.17%/最坏日−4.1031%/最坏完成周期−15.3817%/最大盈利集中34.50%/最长空仓81日，另1未平仓净损益3784.16360元进入账户、不计完成胜率；近期平均暴露9.00%/最坏日−1.6303%/最坏周期−4.5759%/盈利集中34.41%/空仓121日，未平仓0。账户DD未触发10%停买，单笔亏损与NAV回撤分开；2026非完整年。原A的open_pnl_cny在本16行指标表仍UNKNOWN，完整净值已包含其自然持仓，未用未知值支持新方案。

下一步与实际来源检查：已只读检查9项本地表/构建程序。行业贡献、加权宽度两个现成日表不存在；7536行业区间的available_at全部由生效日15:00生成，非历史首发认证；成员和已修复成分回报直接日期各覆盖2823/2855评价日，末日至2026-08-14，缺32日，不倒填。月度行业字段缺独立来源钟，月度权重的逐日承接须另核。当前行业主线金融用途NOT_ADMITTED，尚无已准入待跑金融候选；不据此否定整个主线方向。优先明确主线对510300的匹配和扩散输入合同，或另立不依赖行业分类的成分扩散观察，固定对照后检验；不重开本配置改阈值。历史仅开发/校准，独立NOT_ESTABLISHED、首版NOT_CERTIFIED、全项目搜索选择校正NOT_COMPUTED。

依据：[完整结果、点位及费用归因](../{report})；[固定用途卡](510300_BROKER_STAGE_POLICY_V1.md)；[全部金融指标与区间](../{result})；[主线来源检查](../{source_report})；[目标服务active实际返回](../{proof})。
"""
    decisions = f"""### TECH.R211—R212：分阶段进场/持有的完整账户结果及下一来源检查（2026-10-05）

R211一次预登记、R212一次运行：12新账户/0拟合/0标签/0行情请求；原A4账户精确复现、9必要测试和12保存复算通过。这里接受的是具体事实或继续检验的假设，当前没有策略晋升；下方R210‘未运行’保留为历史记录。

| 方向 | 假设 | 验证方法 | 实际结果 | 为什么接受/拒绝 | 是否需要重新验证 |
|---|---|---|---|---|---|
| 三路线分阶段完整政策 | 低波动修复、放量重新定价和趋势回踩可提高完整收益/夏普及有效次数 | 相同20万元/风险/股息/T+1/费用，主政策与原A、两归因对照；两时期两费用全部运行 | 四经济门0/4、历史稳定失败；较早压力2.1871%/0.52919/pB0.92080，近期−0.2400%/−0.06608/pB0.70939 | 拒绝本固定配置；近期低于原A，两个时期实际pB均未过1，局部上涨不足以晋升 | 不调参重跑救援；实质不同用途另登记，并保留独立验证要求 |
| 阶段持有与结构失效 | 当时确认后延续持有可能比固定1ATR/2ATR/20日退出更合适 | 同进入规则固定退出控制，其余风险/费用一致；退出改变后续资金占用，不能称每笔配对 | CAGR/Sharpe四场景点值均高于固定退出，多数95%区间仍跨零 | 接受该有限历史点值；未接受稳定优势或新有效策略 | 是，不同独立机会集、完整账户及稳定检验 |
| 宏观按进入/确认/退出分工 | 资金改善用于修复、融资用于确认，慢变量不硬压重新定价 | R211已知钟与未知分支固定；全部来源和时期保留，原无阶段资格同退出作对照 | 较早修复10完成pB1.25301，近期23完成pB0.68023；近期回踩8完成0胜/8负 | 当前宏观与阶段资格没有跨时期稳定收益作用；不能据失败只删回踩或反向使用融资 | 本配置不营救；不同信息角色另登记 |
| 2024实际启动与退出 | 全日历的事前规则可识别该上涨并持有 | 9月24日决定、次开真实份额，已知周柱/融资晋级，结构失效次开退出 | 09-25至10-17完成净回报12.1365%、盈利3647.23568元 | 接受已完成周期事实，拒绝由单笔证明全期、因果或独立效果 | 个案事实已保存；泛化仍需全部反例和独立验证 |
| 增加交易次数 | 新机制增加机会同时也许保留质量 | 按真实完整周期，风险减仓不重复计数；2026部分年分列 | 较早/近期完整年均5.4/6.667，相对A4.4/4.167；近期胜率27.27% | 次数确实增加，但质量与账户收益不达标，不恢复逐年5次硬配额 | 是，须与净pB、净收益及夏普同时验收 |
| 费用是唯一失败来源 | 降低摩擦可能使策略成功 | 只按实际成交份额、时点、风险加回摩擦，作为解释而非新策略 | 近期毛3983.80元/摩擦7079.20元/净−3095.40元；无费固定数量年化0.3043%，低于A扣费3.9908% | 拒绝‘仅费用导致失败’，仍需更好的机会区分；不降低冻结费用救援 | 解释已经完成；不同信息用途以原费用再验证 |
| 行业主线与510300匹配 | 产业共识、行业扩散和宽基暴露可能解释不同启动的持续性 | 先读取9本地表、原构建逻辑与2855评价日；不构造新信号 | 两现成特征日表无；7536/7536行业钟为生效日生成；成员/回报2823日、缺32日；行业字段首发钟未建立 | 接受继续研究；当前数据不直接准入金融用途，不能事后行业选赢家或将未知填零 | 是，先明确时钟/分类版本、日线与权重覆盖，可另立无行业分类的成分扩散观察 |

未平仓和完整风险已进报告。无盈利来源的B/pB为UNKNOWN，不显示无限或伪高胜率。保存了登记前时间精度测试失败和交付阶段完整年计数错误；后者仅记者复算口径更正，金融运行原频率已经正确，冻结规则/账户没有改动。成员日期列最初误请求date，随后按真实membership_date更正，原说明和源表保留；不当作新增数据。

目标服务实际active，当前金融政策终态拒绝、目标未达，连续受阻0。本轮是实际金融进展，非只写状态；原策略/旧失败、13项独立前瞻逐值保持。独立验证NOT_ESTABLISHED、首版NOT_CERTIFIED、全项目历史选择校正NOT_COMPUTED。

依据：[完整结果](../{report})、[冻结用途](510300_BROKER_STAGE_POLICY_V1.md)、[全部裁决与区间](../{result})、[来源检查](../{source_report})。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                       ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        prepared.append((path, *prepared_prepend(path, text)))
        if (OUT / f"{name}_before_TECH_R212.md").exists():
            raise RuntimeError("事实备份已存在，不能重复插入。")
    state["previous_phase_before_TECH_R211_R212"] = {key: state.get(key) for key in (
        "current_phase", "latest_technical_decision", "latest_registration_decision", "latest_result", "latest_report",
        "latest_actual_financial_decision", "latest_actual_financial_result", "current_phase_trial_accounting")}
    pressure = [row for row in summary["metrics"] if row["cost"] == "STRESS" and row["policy"] == "STAGE_ENTRY_AND_EXIT"]
    state.update({"updated_at": original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "blocked_key": None, "blocked_reason": None, "blocked_audit_key": None, "blocked_scope": None,
        "previous_goal_turn_classification": "PROGRESS_R209_R210_REPORT_SOURCES_AND_PHASE_DESCRIPTION",
        "goal_turn_classification": "PROGRESS_R211_R212_TWELVE_NEW_FINANCIAL_ACCOUNTS_AND_SOURCE_DIAGNOSIS",
        "latest_technical_decision": "TECH.R212", "latest_registration_decision": "TECH.R211",
        "latest_actual_financial_decision": "TECH.R212", "latest_actual_financial_result": result,
        "latest_result": result, "latest_report": report, "latest_completed_study": relative(OUT),
        "latest_research_status": summary["status"], "current_study": "510300_BROKER_STAGE_POLICY_V1",
        "current_phase": "STAGE_POLICY_FINANCIAL_REJECTION_AND_MAINLINE_SOURCE_PREFLIGHT_COMPLETED",
        "latest_progress": "12新账户一次完成、原A4精确复现、9测试和12保存复算；2024实际捕捉，四完整门/稳定失败；9本地来源检查，行业金融未准入。",
        "new_accounts_in_current_phase": 12, "necessary_tests_passed_in_current_phase": 9,
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "latest_long_point_metrics": {"scope": "R212_PRIMARY_STRESS_COMPLETE_CYCLES_DEVELOPMENT_NOT_CURRENT_MARKET", "STRESS": pressure},
        "current_phase_trial_accounting": {"scope": "TECH_R211_R212_FIXED_POLICY_AND_TWO_ATTRIBUTION_CONTROLS",
            "new_accounts": 12, "original_A_exact_replays": 4, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
            "necessary_tests_passed": 9, "saved_accounts_recomputed": 12, "known_daily_rows": 3488,
            "evaluation_calendar_days": 2855, "seventeen_dates_three_policies_rows": 51,
            "four_cases_three_policies_rows": 720, "figures_viewed": 4, "local_mainline_source_files_inspected": 9},
        "current_goal_turn_actual_work": {"new_accounts": 12, "original_A_replays": 4, "new_fits": 0, "new_labels": 0,
            "new_market_requests": 0, "saved_accounts_recomputed": 12, "source_tables_inspected": 9},
        "next_research_question": "主线如何传导到510300：明确分类首发/生效、版本、行业与成分日线、成员权重覆盖及未知分支；另可预登记无行业分类的成分扩散观察。固定用途后与既有策略完整比较，不能据R212失败删路线。",
        "next_information_source_proposal": "现有行业输入NOT_ADMITTED；行业钟为生效日代理，成员/回报缺32评价日、两特征日表不存在。先明确主线匹配/扩散来源等级和完整用途，不把缺口填零。",
        "latest_mainline_source_preflight": relative(MAINLINE / "summary.json"),
        "next_isolated_strategy_design": {"hypothesis": "早期修复与趋势延续采用不同点位及继续/失效退出",
            "status": summary["status"], "registration_decision": "TECH.R211", "result_decision": "TECH.R212",
            "current_financial_run": "TERMINAL_REJECTED", "new_accounts": 12,
            "old_results_and_strategies_preserved": True, "old_forward_next_experiment_preserved": True},
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False,
        "historical_search_selection_correction": "NOT_COMPUTED"})
    if {key: state[key] for key in FORWARD} != forward_before:
        raise ValueError("拟写状态改变了原独立前瞻字段。")
    with (OUT / "state_before_TECH_R212.json").open("xb") as stream:
        stream.write(before_state)
    docs = []
    for path, old, new, body in prepared:
        with (OUT / f"{path.stem}_before_TECH_R212.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        saved = path.read_bytes()
        if saved != new or not saved.endswith(body):
            raise ValueError("事实文档旧正文没有逐字节保留。")
        docs.append({"path": relative(path), "old_sha256": digest(old), "new_sha256": digest(saved), "old_body_preserved_exact": True})
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if {key: study.read(STATE)[key] for key in FORWARD} != forward_before:
        raise ValueError("保存后原前瞻字段不一致。")
    report_path = OUT / "阶段点位_完整结果与失败归因.md"
    report_old = report_path.read_bytes()
    report_new = report_old.replace(b"| nan |", "| UNKNOWN（无盈利笔，B不可估） |".encode("utf-8"))
    if report_old.count(b"| nan |") != 1:
        raise ValueError("来源未知展示更正范围不是原唯一一行。")
    with (OUT / "report_before_unknown_display_correction.md").open("xb") as stream:
        stream.write(report_old)
    report_path.write_bytes(report_new)
    figures = [{"path": path, "sha256": digest((ROOT / path).read_bytes()), "viewed": True}
               for path in delivery["figures"]]
    study.write(OUT / "figure_view_receipt.json", {"at": original.now(), "figures": figures,
        "method": "本研究轮四张科学图已逐张通过view_image查看，中文标题/图例可读，无截断；未重新渲染。",
        "new_accounts": 0, "new_fits": 0})
    study.write(receipt_path, {"at": original.now(), "status": "R212_FINANCIAL_REJECTION_AND_NEXT_SOURCE_LIMITS_SAVED",
        "docs": docs, "forward_fields_preserved_exact": list(FORWARD), "frozen_own_code_test_card_preserved": frozen,
        "old_state_sha256": digest(before_state), "new_state_sha256": digest(STATE.read_bytes()),
        "unknown_display_change": {"old_sha256": digest(report_old), "new_sha256": digest(report_new),
            "meaning": "只把无盈利来源的nan显示为UNKNOWN；没有变动规则、账户或数值。"},
        "financial_status": summary["status"], "mainline_source_status": mainline["status"],
        "current_goal_status": "active", "goal_achieved": False, "current_blocked_count": 0,
        "new_accounts_completed_this_phase": 12, "new_accounts_created_by_this_archive": 0})
    print("R212完整金融拒绝与来源限制已归档：四事实文档旧正文逐字节、13前瞻逐值及4冻结文件保持；目标active，尚未实现。", flush=True)


if __name__ == "__main__":
    main()
