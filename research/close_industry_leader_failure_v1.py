"""一次归档完整行业失效账户拒绝，保留原事实与独立前瞻。"""
from __future__ import annotations

import json

from research import industry_leader_failure_study_v1 as study
from research.close_broker_cycle_inspiration_v1 import FORWARD
from research.close_broker_stage_policy_v1 import digest, prepared_prepend

ROOT, OUT, parent = study.ROOT, study.OUT, study.parent
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def main():
    if (OUT / "project_state_update_receipt.json").exists() or (OUT / "state_before_TECH_R224.json").exists():
        raise RuntimeError("行业失效金融已归档或开始，不重复更新。")
    summary = study.read(OUT / "summary.json")
    diag = study.read(OUT / "post_run_diagnosis.json")
    delivery = study.read(OUT / "delivery_receipt.json")
    views = study.read(OUT / "figure_view_receipt.json")
    service = study.read(OUT / "goal_service_status_after_result.json")
    if (summary["decision"] != "TECH.R224" or summary["new_accounts"] != 12
            or summary["original_R212_exact_replays"] != 4 or summary["necessary_tests_passed"] != 9
            or summary["saved_account_checks"] != 12 or len(summary["metrics"]) != 20
            or len(summary["comparisons"]) != 16 or summary["all_economic_gates_passed"]
            or summary["historical_stability_passed"] or any(x["economic_passed"] for x in summary["gates"])
            or diag["original_eight_actual_clock_states"].get("ACTUAL_EXTRA_EXIT") != 1
            or views["viewed"] != 4 or service["goal"]["status"] != "active"
            or study.digest(ROOT / delivery["report"]) != delivery["report_sha256"]):
        raise ValueError("完整账户、拒绝结果、时钟归因或交付状态不匹配。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    if state["latest_technical_decision"] != "TECH.R222" or state["latest_actual_financial_decision"] != "TECH.R216":
        raise ValueError("项目事实已由其他研究推进，先核实。")
    forward = {key: state[key] for key in FORWARD}
    previous = {key: state.get(key) for key in ("latest_technical_decision", "latest_registration_decision",
        "latest_actual_financial_decision", "latest_actual_financial_result", "latest_report", "latest_result", "current_phase",
        "next_entry_anchored_industry_failure_proposal")}
    report = delivery["report"]
    result = study.relative(OUT / "summary.json")
    proof = study.relative(OUT / "goal_service_status_after_result.json")
    date = parent.original.now()[:10]
    note = f"""> 最新实际金融（{date}，TECH.R223—R224）：真实进入锚点的固定领先行业失效唯一配置已完整运行，**拒绝**；四场景经济门0/4、稳定门失败，完整收益夏普目标未达。9必要测试、原R212四账户五表及终态精确复现后固定，12新账户一次运行/12账本与时序核验，20账户比较、32两尺度区间、4图已实际查看。0新拟合/训练目标/采集/参数搜索/金融重跑。目标服务实际active、本轮PROGRESS、连续受阻0；原13项独立前瞻保持。

主政策在实际进入前一源日固定前三行业，只在实际进入索引+5收盘看五个已形成源日，ETF同期含息累计正且原领先行业均值非正才次开退出；原风险退出优先，未知不重试。初始锚保持294/300、至少5成员/98%20报价/4行业；新退出用途只测三领先组各98%完整，跟随未知另存，未改R222全组门。分类公布钟须不晚于实际进入09:30，源龄和失败保留。对照为原A、原阶段、同领先覆盖ETF正退出和全日历ETF正退出；原资金费用风险与全部日历不改。

压力早期：新CAGR2.0172%/Sharpe0.493683/DD6.3248%，28完成12赢16输、胜率42.86%、B2.136698、pB0.915728、标准净期望+0.344299，完整年5.6次；低于阶段2.1871%/0.529186，完整财富少1780.55元。压力近期：新−0.1242%/−0.026868/DD7.0748%，45完成15赢30输、胜率33.33%、B1.992259、pB0.664086、标准期望−0.002580，完整年6.667次；比阶段−0.2400%/−0.066082小幅改善，却明显低于A3.9908%/1.216910及同覆盖价格−0.0487%/0.001210。基础费用也未通过，所有场景pB均未过1；不是每个场景标准期望都负。

实际归因：原8个信号锚说明性亏损，在真实买入第五日只有1次额外退出，3次此前/当日开盘已按原规则退出、4次ETF同期已非正。每费用口径早期1、近期4次实际额外退出；压力近期4笔对应改善2808.29元，释放资金又新增2026-05-11信号/05-12进入、净亏1753.77元，连同其他数量变化完整财富只改善1488.24元。早期2017-03-22周期基础小赢20元变−28.59，压力原亏−89.81变−132.54；新增04-05信号/04-06进入净亏1181.26元。没有按原亏损节约额宣传完整收益。

毛优势仍弱：近期压力毛5867.90元、摩擦7475.05元、净−1607.15元；固定实际数量路径无费用解释上界CAGR0.4464%/Sharpe0.1666仍低于A，非可执行策略。四场景无拒单、DD停买未触发，但压力早期34/近期46次风险减仓，不能说风险预算没有影响。2019-06-12至07-10、2025-06-26至08-04盈利持有及2024-09-25至10-17大上涨保持；原A开放损益字段未知不填0。新早期开放损益3746.88元、近期0。

下一优先：进入证据是否更新，突破是否伴随新的PMI/政策利率已知信息、资金/融资与行业参与，还是相同背景重复触发。使用全部143原事件、原四案例/所有假启动/本次两个新增亏损，先解释公布钟和信号顺序再立一个完整进入—持有机制；不因本次新增亏损直接删重新定价路线。当前仅提出下一描述，未登记/运行、无已准入待跑金融。R212/R216/R224配置拒绝及R222、R220事实均保持，历史开发/首版未认证/独立未建立。

依据：[完整金融及失败归因](../{report})、[实际账户结果](../{result})、[用途卡](510300_INDUSTRY_LEADER_FAILURE_V1.md)、[实际目标状态](../{proof})。
"""
    decisions = f"""### TECH.R223—R224：实际进入后的固定领先行业失效（{date}）

9测试、4原阶段精确复现、12新账户一次运行、20完整账户比较/32两尺度区间/4图实际查看。四经济门0/4、稳定失败，拒绝固定配置；目标active、PROGRESS、受阻0，收益夏普目标未达。0新拟合/标签/行情/参数搜索或金融重跑。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 真实进入锚点 | 信号后5槽能代表实际买入后观察 | 实际成交前一源日固定行业/成员，实际进入+5收盘、严格止于前一源日 | 原8亏损只有1实际触发、3原退出已发生、4ETF已非正 | 接受实际时钟实现，拒绝将原8描述当8可交易退出 | 新不同用途仍需真实成交钟，当前不改天数救规则 |
| 三领先行业用途 | ETF仍正而原主线失效足以改善完整账户 | 一主政策/同覆盖价格/全价格、A与阶段四对照，原资金费用风险两时期 | 压力近期−0.1242%/−0.026868，早期2.0172%/0.493683；0/4经济门 | 拒绝配置，近期只比弱阶段改善且早期变差，不满足pB/对照/稳定 | 不重复或调门救该配置；新不同进入机制另立 |
| 覆盖与公布钟 | 只测领先行业需完整观察全部跟随行业 | 金融用途事前定义三领先98%完整，跟随单列；分类公布<=实际买入09:30 | 新9测试、12账户源钟/未知检查通过；原R222门不改 | 接受用途对象一致的独立定义，不接受改写原全组已知门或按结果改覆盖 | 新用途事前定义角色和未知，不回填原失败 |
| 原8说明性赢家筛选 | 原8个后来亏损可证明新策略准确率 | 全8真实时钟及所有实际检查、未检查和不同期间保存 | 只有2023年3月实际触发，其余3真实触发来自原8之外 | 拒绝100%胜率与8笔可执行节约，保留完整事件集合 | 新独立样本才验证方向，当前全历史开发 |
| 缩短旧亏损等于账户收益提升 | 4旧亏损改善可直接累加到净值 | 完成净损益差+开放差复算全财富，并保留释放现金新进入 | 近期对应改善2808.29元，但新05-12交易亏1753.77，完整增量1488.24 | 拒绝单笔节约代替完整现金账户；接受账户路径反馈证据 | 后续进入和持有必须联合检验 |
| 早期退出无损 | 行业弱能避免所有反弹误判 | 全周期按原信号身份对齐，基础/压力反例同时保留 | 2017周期基础20盈利变−28.59、压力−89.81变−132.54；另新增亏1181.26 | 拒绝只报近期改善，费用下赢家身份也不同 | 原反例必须进入新完整用途，不改当前天数 |
| 更高胜率等于更好策略 | 新胜率提高足以补弱毛优势 | 实际净回报计算胜率/B/pB/标准期望，全账比较 | 近期压力胜率33.33%较阶段27.27%高，B2.601降1.992，pB0.664、EV−0.002580 | 拒绝仅提高胜率宣称目标达成；pB>1与标准期望分别检验 | 新完整账户同风险/成本验证 |
| 仅费用造成失败 | 小资金降低摩擦即可使模型优于A | 毛损益/费用/净财富分解，固定实际数量零费用解释边界 | 近期毛5867.90/费7475.05/净−1607.15；零费CAGR0.4464%/S0.1666仍低于A | 拒绝费用单因解释；零费边界不是可执行策略 | 新进入机制需改善毛优势，保持压力费用 |
| 账户约束造成全部失败 | 拒单或DD停止使好信号无法执行 | 保存全部实际拒单、停买和风险减仓 | 无拒单/无DD停买，压力早期34/近期46风险减仓 | 拒绝全部归因拒单，保留风险预算影响，不放开风险救配置 | 新机制维持相同风险，解释边界另列 |
| 历史稳定与去过拟合 | 小幅近期净值提升可当稳定增量 | 四对照20/252各2000全部32区间、原历史身份保留 | 近期对A两尺度全负，对阶段区间跨0；稳定失败 | 拒绝稳定/独立/过拟合去除声称 | 仅新不同机制或真实新样本，当前配置终局拒绝 |
| 下一进入信息顺序 | 价格事件需区分新的信息更新和旧背景重复触发 | 下一描述全部143事件及PMI/政策/融资公布钟、行业参与、原四案例和新增反例 | 尚未登记/运行，PROPOSED | 接受下一个问题，不按亏损路线表反选策略 | 是，先完整解释再登记进入—持有账户，历史不能充独立 |

来源：[完整结果](../{report})、[金融摘要](../{result})、[失败归因](../{study.relative(OUT / 'post_run_diagnosis.json')})、[固定用途](510300_INDUSTRY_LEADER_FAILURE_V1.md)。原13独立前瞻和旧事实保持；当前无已准入待跑金融。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
            ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        path = ROOT / f"docs/{name}.md"
        old, new, body = prepared_prepend(path, text)
        prepared.append((path, old, new, body))
    with (OUT / "state_before_TECH_R224.json").open("xb") as stream:
        stream.write(raw)
    doc_receipts = []
    for path, old, new, body in prepared:
        path.write_bytes(new)
        if path.read_bytes() != new or not path.read_bytes().endswith(body):
            raise ValueError("原事实正文未逐字节保持。")
        doc_receipts.append({"path": study.relative(path), "old_sha256": digest(old), "new_sha256": digest(new), "old_body_preserved_exact": True})
    state["previous_phase_before_TECH_R223_R224"] = previous
    state.update({"updated_at": parent.original.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": parent.original.now(),
        "latest_goal_tool_status_receipt": proof, "consecutive_blocked_goal_turns": 0, "blocked_audit_count": 0,
        "latest_technical_decision": "TECH.R224", "latest_registration_decision": "TECH.R223",
        "latest_actual_financial_decision": "TECH.R224", "latest_actual_financial_result": result,
        "latest_actual_financial_status": summary["status"], "latest_report": report, "latest_result": result,
        "latest_progress": "实际进入三领先行业失效12完整账户终局拒绝；真实时钟、新增亏损和弱毛优势归因完成，下一优先进入信息更新。",
        "current_phase": "ENTRY_ANCHORED_INDUSTRY_LEADER_FAILURE_REJECTED_NEXT_ENTRY_INFORMATION_SEQUENCE_DESCRIPTION",
        "goal_turn_classification": "PROGRESS_R223_R224_COMPLETE_ACCOUNTS_AND_REAL_CLOCK_CAPITAL_REUSE_FAILURE_ATTRIBUTION",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "new_accounts_in_current_phase": 12, "necessary_tests_passed_in_current_phase": 9,
        "current_financial_candidate_admission": "FIXED_INDUSTRY_LEADER_FAILURE_REJECTED_NO_ADMITTED_UNRUN_FINANCIAL_CANDIDATE",
        "current_phase_trial_accounting": {"scope": "TECH_R223_R224_ENTRY_ANCHORED_INDUSTRY_LEADER_FAILURE",
            "new_accounts": 12, "original_R212_exact_replays": 4, "original_A_new_replays": 0,
            "new_fits": 0, "new_labels": 0, "new_market_bars": 0, "parameter_grid": False,
            "necessary_tests": 9, "saved_account_checks": 12, "full_account_metric_rows": 20,
            "paired_interval_rows": 32, "economic_gates_passed": 0, "economic_gates_total": 4,
            "historical_stability_passed": False, "all_original_eight_clock_cases": 8, "original_eight_actual_extra_exit": 1,
            "stress_earlier_extra_exits": 1, "stress_recent_extra_exits": 4, "financial_reruns": 0, "figures_viewed": 4},
        "current_goal_turn_actual_work": {"new_financial_purposes_completed": 1, "new_accounts": 12,
            "original_account_exact_replays": 4, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
            "completed_capital_reuse_failure_diagnoses": 1, "new_admitted_unrun_financial_purposes": 0},
        "next_entry_anchored_industry_failure_proposal": {"status": "COMPLETED_REJECTED_FIXED_CONFIGURATION",
            "registration": "TECH.R223", "decision": "TECH.R224", "result": result, "economic_gates_passed": 0, "historical_stability_passed": False},
        "next_information_source_proposal": "复用已保存实际公布钟与背景，先区分新PMI/政策信息、资金融资变化、行业参与和价格突破顺序；全部事件/假启动保留，未登记新金融。",
        "next_mainline_entry_information_description": {"status": "PROPOSED_NOT_REGISTERED_OR_RUN",
            "question": "突破是否伴随新的已知宏观资金信息和行业参与，还是同一背景的反复触发。",
            "scope": "全部143原进入事件、原四案例/所有假启动和R224新增2017-04-05、2026-05-11信号；只解释已发生顺序。",
            "source_fields": ["orders_available_at", "orders_reference_period", "funding_available_at", "funding_policy_known_at", "margin_available_at", "原资金融资值", "R222行业参与与源龄", "原量价事件"],
            "no_result_selection": "不因原路线事后净损益直接删路线，不挑原盈利片段、期限或阈值；解释后新完整用途另登记。",
            "new_financial_run": "NOT_RUN", "independent_validation": "NOT_ESTABLISHED"},
    })
    with STATE.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    saved = study.read(STATE)
    if {key: saved[key] for key in FORWARD} != forward:
        raise ValueError("原13独立前瞻字段被改变。")
    study.write(OUT / "project_state_update_receipt.json", {"at": parent.original.now(), "status": "R224_FINANCIAL_REJECTION_AND_FAILURE_ATTRIBUTION_ARCHIVED",
        "docs": doc_receipts, "old_state_sha256": digest(raw), "new_state_sha256": study.digest(STATE),
        "forward_fields_preserved_exact": list(FORWARD), "latest_technical_decision": "TECH.R224", "latest_actual_financial_decision": "TECH.R224",
        "goal_status": saved["goal_status"], "goal_turn_classification": saved["goal_turn_classification"], "consecutive_blocked_goal_turns": 0,
        "new_accounts": 12, "economic_gates_passed": 0, "goal_achieved": False})
    print("R224实际金融拒绝和失败归因已一次归档；四长期事实原正文、原13前瞻保持，目标active。", flush=True)


if __name__ == "__main__":
    main()
