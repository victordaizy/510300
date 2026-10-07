"""一次交付当前官方份额来源与独立观察器，保留全部历史金融终态。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research import fund_share_current_official_sources_v1 as source
from research import fund_share_daily_snapshot_observer_v1 as observer
from research.source_expectation_transmission_review_v1 import FORWARD, FINANCE

ROOT, OUT = source.ROOT, source.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def deliver():
    protocol = source.read(OUT / "source_protocol.json")
    if (OUT / "summary.json").exists():
        raise RuntimeError("本轮实际交付已完成，不能覆盖。")
    original = source.read(source.PRIOR)
    tests = source.read(OUT / "observer_tests_receipt.json")
    fixed = source.read(OUT / "current_source_receipts.json")
    follow = source.read(OUT / "linked_script_follow_receipt.json")
    seed = source.read(OUT / "seed_snapshot/normalized.json")
    seed_receipt = source.read(OUT / "seed_snapshot/receipt.json")
    daily_protocol = source.read(OUT / "daily_observer_protocol.json")
    if source.sha(Path(source.__file__)) != protocol["collector_sha256"]:
        raise ValueError("登记后的原采集器发生变化。")
    for row in protocol["prior_sources"]:
        if source.sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("旧来源裁决发生变化。")
    if tests["exit_code"] != 0 or tests["passed"] != 10 or tests["observer_sha256"] != source.sha(Path(observer.__file__)):
        raise ValueError("必要测试/观察器身份不同。")
    if daily_protocol["observer_sha256"] != source.sha(Path(observer.__file__)) or seed_receipt["http_status"] != 200:
        raise ValueError("冻结观察器或真实官方响应不符。")
    if seed["status"] != "BASELINE_SEED_NOT_PROSPECTIVE" or seed["security"] != "510300.SH":
        raise ValueError("本轮未取得正确标的的官方基线。")
    if seed["statistics_date"] != "2026-09-30":
        raise ValueError("本轮当前官方统计日期与已经检查的范围不同。")
    rows = [*fixed["results"], *follow["results"]]
    for row in rows:
        path = OUT / "sources" / row["id"] / "response.body"
        if row["status"] != "SAVED_CURRENT_OFFICIAL_RESPONSE" or source.sha(path) != row["sha256"]:
            raise ValueError("当前来源响应没有完整保存。")
    if source.sha(OUT / "seed_snapshot/response.body") != seed_receipt["raw_response_sha256"]:
        raise ValueError("官方数据原响应发生变化。")
    semantics = {
        "at": source.now(), "home_total_scale": {"field": "SCALE", "unit": "亿元", "scope": "基金市场总览FUND_TYPE=fund00",
            "publication_statement": "当前页面总规模每交易日23:00更新", "html_receipt": "sources/FUND_HOME/receipt.json",
            "implementation": "sources/FUND_HOME_JS/text.txt与sources/FUND_API_JS/text.txt"},
        "single_etf_units": {"field": "TOT_VOL", "unit": "万份", "scope": "指定SEC_CODE统计日基金总份额",
            "publication_statement": "当前ETF规模页面称清算后数据；没有认证该字段全历史首次公开钟",
            "raw_field_identity_confirmed": True, "seed_response": "seed_snapshot/response.body"},
        "clock_adjudication": "总规模23:00不能直接移接到单ETF总份额；观察器选择23:05是采集窗口，available_at仅实际收到并保存时刻。",
        "old_history": {"closure_retained": original["status"], "closed_responses_not_rescanned": original["historical_raw_response_count"],
                        "historical_first_publication": "NOT_ESTABLISHED", "old_eight_rules": "REJECTED_FROZEN_NOT_REPLAYED"},
        "new_version": "来源直接query.sse.com.cn；取得时刻及原数值明确，但不是历史首次公开认证。",
        "split_and_flow": "尚未建立拆分处理合同；原份额变化仅原单位观察，不等于净申购、人民币现金流或投资者身份。",
    }
    source.write(OUT / "source_semantics_review.json", semantics)
    result = {"at": source.now(), "registration": "TECH.R243", "decision": "TECH.R244",
        "status": "CURRENT_OFFICIAL_SHARE_SEED_SAVED_FORWARD_VERSION_OBSERVER_READY_NOT_FINANCIAL_VALIDATION",
        "current_official_document_requests": 4, "current_linked_script_requests": 2, "current_official_data_requests": 1,
        "actual_requests": 7, "saved_raw_responses": 7, "official_seed_response_rows": seed_receipt["source_rows"],
        "target_seed_rows": 1, "seed_statistics_date": seed["statistics_date"], "seed_available_at": seed["available_at"],
        "seed_units": seed["total_units"], "seed_role": seed["status"], "new_prospective_share_dates": 0,
        "new_prospective_share_changes": 0, "new_model_admitted_inputs": 0, "necessary_tests": 10,
        "closed_response_rescans": 0, "old_rule_replays": 0, "new_fits": 0, "new_labels": 0,
        "new_accounts": 0, "new_financial_runs": 0, "minute_bars": 0, "trade_signals": 0,
        "forward_observer_first_eligible_source_date": "2026-10-08", "forward_collection_window_start": "2026-10-08T23:05:00+08:00",
        "earliest_two_new_source_dates": ["2026-10-08", "2026-10-09"], "calendar_end": daily_protocol["calendar_end"],
        "not_a_verified_wait": "没有确认存活的采集进程/任务；日历日期不计已观察，未创建自动采集。",
        "financial_admission": "NOT_ADMITTED", "split_adjustment": "NOT_ESTABLISHED",
        "historical_first_vintage": "NOT_CERTIFIED", "independent_strategy_validation": "NOT_ESTABLISHED",
        "current_source_direction_view": "NO_VIEW", "net_cagr": "NOT_COMPUTED_NO_NEW_ACCOUNT",
        "net_sharpe": "NOT_COMPUTED_NO_NEW_ACCOUNT", "goal_achieved": False}
    source.write(OUT / "summary.json", result)
    text = f"""# 510300当前官方份额来源与独立日频版本观察

2026-10-06；TECH.R243—R244。上一轮R241—R242完成来源与旧研究去重，属于实质进展。本轮针对份额时钟缺口，完成真实当前官方获取、字段核对、观察器、必要测试与一次基线；收益和夏普目标仍未达成。

## 新取得的实际证据

4个当前官方页面/脚本、2个当前原文链接脚本、1次直接官方份额API，共7次本机GET，7个响应均保存。没有历史日期批量查询，没有重扫2220个已关闭的旧响应，没有重跑八条份额规则。本地AKShare源码仅用于定位API；本轮数值直接来自query.sse.com.cn原响应。

| 字段 | 对象及单位 | 当前原文说明 | 本项目可用结论 |
|---|---|---|---|
| SCALE | 基金市场总览，亿元 | 当前首页写总规模每交易日23:00更新 | 说明总规模更新安排；不能直接认证单ETF历史总份额 |
| TOT_VOL | 单ETF统计日总份额，万份 | 当前ETF规模页写当日清算后数据 | 不能用于当天16:00决定；只有真实取得的版本才可知 |

官方原文：[基金网站首页](https://etf.sse.com.cn/home/)、[ETF规模](https://www.sse.com.cn/market/funddata/volumn/etfvolumn/)。实现链与字段核对见source_semantics_review.json及sources全部原响应。

实际基线：统计日**{seed['statistics_date']}**，510300总份额原值**{seed['reported_total_units_10k']}万份**，规范值**{seed['total_units']}份**，实际完成取得时刻**{seed['available_at']}**。官方响应920个产品行，本研究仅抽出唯一510300行；没有据此认证动态沪深300 ETF体系。该记录标为BASELINE_SEED_NOT_PROSPECTIVE，历史公开时刻NOT_ESTABLISHED，不计前瞻，不与未来新版本混算变化。

## 已实现的日频观察器

完整入口research/fund_share_daily_snapshot_observer_v1.py已实现并冻结。它直接保存官方原响应、请求参数、实际完成时刻、原数值/单位和规范值，原文件不覆盖。没有连接券商，也不生成交易信号。

只在实际新交易日23:05之后获取当日统计值；该时间是保守采集安排，不是移接总规模更新时间。请求日、实际统计日、官方日历和实际采集钟必须对应。日期不同/仍旧统计日、字段缺失、非有限值、重复目标行、非交易日或未来日保留NO_VIEW，不填零、不猜字段、不提前补采。

两份相邻官方交易日的新版本都在使用时点前实际取得，才计算原份额变化。基线、缺日和使用时点之后取得的版本不能混入；拆分处理尚未建立，因此该变化不解释成确定净创造、更不等于人民币现金流或国家队身份。合格的原版本观察也不等于已准入模型或已验证预测优势。

10项必要测试全部通过，覆盖单位换算、错误标的、规模冒充份额、旧日冒充新日、基线混用、跨缺失交易日、晚到版本、合格相邻版本、无效数值和不执行JSONP代码。测试验证实现与时间资格，没有验证收益、胜率或夏普。

## 下一次实际工作

当前独立份额前瞻日期0、变化0、交易信号0、新金融账户0；没有存活采集句柄，没有建立自动任务，不将日历作为verified wait。首个允许的新统计日2026-10-08，采集窗口从23:05开始。至少实际取得10月8日与9日两份新版本后，才可能在10月12日的后续使用时点分析第一份原日变化；缺任一份则保持NOT_COMPUTED。

研究下一步是支持出现后，新增份额观测与价格响应是否共同提供持续性信息。现在只有数据观察合同，没有新的金融主策略；不会将旧G01—G04/八二元规则改名重跑，也不会用先验未验证的份额变化给过去五笔盈亏加过滤。政策工具匹配的事前数量预期部分仍未取得、尚未登记为数值模型。

原E03独立前瞻13字段及R240实际金融身份保持。日频份额观察是独立来源合同；不会替代E03的2026-10-08 15:05资格。日历目前只到2026-12-31，超出日历须停，不猜工作日。

复算与证据：source_protocol.json、current_source_receipts.json、linked_script_follow_receipt.json、current_api_follow_protocol.json、daily_observer_protocol.json、observer_tests_receipt.json、seed_snapshot原响应/规范值/回执、summary.json。当前份额方向观点NO_VIEW，新CAGR/Sharpe没有计算，独立策略验证和去过拟合均未建立。
"""
    with (OUT / "当前官方份额_字段时钟与独立日频观察结果.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    card = """# 510300官方日频份额版本观察：固定使用合同

2026-10-06，TECH.R243—R244。用途为新官方数据版本观察，未准入金融模型，没有交易动作或收益验证。只观察510300.SH，日线信息；原E03独立。

已完成一次freeze、十项必要测试和一次seed，不重复这些动作。请求直接来自上交所当前官方API，原字段TOT_VOL单位万份；available_at只等于本机实际收到保存时刻，不能用STAT_DATE或网页总规模23:00冒充历史首次公布。

种子为2026-09-30统计值，实际2026-10-06取得，明确排除前瞻和后续份额变化。正式观察从2026-10-08开始；一个统计日一次，只有当前实际交易日23:05以后可调用。缺失/错误日/请求失败保留，不插值、不补旧日期。

📁 C:\\Users\\戴周阳\\Documents\\New project 8\\research\\fund_share_daily_snapshot_observer_v1.py（PowerShell运行入口）

```powershell
Set-Location -LiteralPath 'C:\\Users\\戴周阳\\Documents\\New project 8'
.\\.venv\\Scripts\\python.exe -X utf8 -m research.fund_share_daily_snapshot_observer_v1 observe --date 2026-10-08
```

上述命令只在实际2026-10-08 23:05以后执行，本轮没有提前调用。每个新统计日有独立目录、原响应、实际钟和规范值；已经创建的目录禁止覆盖/重跑。未创建自动采集任务。

两份相邻官方交易日的新版本均在使用时点前取得，才能计算原份额增量；一份基线与一份新版本不能配对。首次可能配对10月8日与9日，是否实际取得需看真实回执。10月12日等后续使用时点仍必须晚于双方available_at。

拆分/人民币资金流/投资者身份未建立，不把原份额变化解释为现金净流入或国家队。原G01—G04与八失败规则保持。新的适应性策略还需事前固定完整输入/动作和对照，再比较完整账户净收益/夏普/实际pB与风险，不能由观察器晋升为策略。

当前日历仅2026年，缺日历时停止。当前没有新前瞻份额变化、模型预测或交易信号；目标未达，独立验证未建立。

证据：[实际来源与基线结果](../reports/research/510300_current_official_share_source_v1/当前官方份额_字段时钟与独立日频观察结果.md)、[冻结观察协议](../reports/research/510300_current_official_share_source_v1/daily_observer_protocol.json)、[必要测试回执](../reports/research/510300_current_official_share_source_v1/observer_tests_receipt.json)。
"""
    with (ROOT / "docs/510300_DAILY_OFFICIAL_SHARE_VERSION_OBSERVER_V1.md").open("x", encoding="utf-8") as stream:
        stream.write(card)
    print("R244实际交付已保存：7官方响应、1基线、10测试、完整独立观察器；0新金融。", flush=True)


def close():
    result = source.read(OUT / "summary.json")
    protocol = source.read(OUT / "source_protocol.json")
    service = source.read(OUT / "goal_service_status_after_delivery.json")
    if service["goal"]["status"] != "active":
        raise ValueError("目标服务未确认active。")
    if (OUT / "project_state_update_receipt.json").exists():
        raise RuntimeError("项目事实已更新，不重复。")
    old_state = STATE.read_bytes()
    state = json.loads(old_state.decode("utf-8-sig"))
    for keys, name in [(FORWARD, "forward_before"), (FINANCE, "financial_before")]:
        if {k: state[k] for k in keys} != protocol[name]:
            raise ValueError("原前瞻或实际金融身份发生变化。")
    report = relative(OUT / "当前官方份额_字段时钟与独立日频观察结果.md")
    summary = relative(OUT / "summary.json")
    source_card = "510300_DAILY_OFFICIAL_SHARE_VERSION_OBSERVER_V1.md"
    note = f"""### TECH.R243—R244：真实当前官方份额与独立日频版本观察器（2026-10-06）

上一轮R241—R242实际来源与旧家族去重为PROGRESS。本轮完成4当前官方页/脚本、2当前原文链接脚本、1当前官方份额API共7真实GET/7保存响应；没有历史批量查询、2220旧响应重扫或8失败规则重放。本地库仅定位，数值直接query.sse.com.cn。真实920产品行只规范唯一510300，不认证动态全体系。

当前官网总规模SCALE单位亿元、23:00更新；单产品份额TOT_VOL单位万份且称清算后，两个字段的更新时间不能移接到历史。真实510300基线统计2026-09-30、原2411958.77万份/24119587700份，2026-10-06T08:02:05.265002+08:00实际取得；BASELINE_SEED_NOT_PROSPECTIVE，历史首发未知，不计前瞻、不与新日期混算。

完整独立日频观察器已实现/冻结，10必要测试通过；保存原响应与实际钟，严格官方日历、标的/单位/统计日、基线排除、连续新版本及使用时点。采集选择新实际交易日23:05以后，是保守安排而非历史公布推定。拆分/现金流/投资者身份未知，原份额增量只作描述。没有自动任务或存活采集句柄，不能称verified wait；当前新份额日期/变化/信号0，数值金融用途NOT_ADMITTED、0待跑金融。

原E03的13字段保持，独立来源首个允许2026-10-08 23:05；须实际取得8日和9日两版本，才可能在12日后续时点分析第一份原日变化，缺任一仍NOT_COMPUTED。日历只至2026年底。另同工具事前数量预期仍缺合格新输入，不将旧M1/M2、全样本冲击、八份额或G01—G04更名救援。

最新实际金融仍TECH.R240固定拒绝；收益/夏普目标未达，独立/首版/去过拟合未建立。本轮0新金融/账户/拟合/标签/分钟柱，目标服务active/PROGRESS、受阻0；旧正文和全部原终态保留。

依据：[实际来源、字段与基线结果](../{report})、[完整统计](../{summary})、[固定使用合同]({source_card})、[真实API回执](../{relative(OUT/'seed_snapshot/receipt.json')})、[十必要测试](../{relative(OUT/'observer_tests_receipt.json')})。
"""
    decisions = f"""### TECH.R243—R244：官方份额字段时钟与新版本来源合同（2026-10-06）

7当前官方响应、唯一510300基线、完整观察器与10必要测试均实际完成；0新金融，新策略收益夏普未计算。旧金融仍R240固定拒绝，原13E03保持，目标active未达/PROGRESS、受阻0。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 总规模23:00能补历史份额钟 | 官网固定时间可以认证TOT_VOL全历史 | 当前首页、实现链、份额页与API字段分别核对 | 总规模SCALE为亿元，TOT_VOL为单ETF万份；当前清算后说明，无历史首版 | 拒绝跨字段/跨版本移接；接受保守未来采集安排 | 只有对应字段/版本原披露证据才补；不扫2220旧响应 |
| 当前直接官方可取得 | 可用真实下载版本建立新日频观测 | 一次空日期当前API与代码定位、原响应保存和唯一标的规范 | 200成功，920产品行中唯一510300，9月30日2411958.77万份，10月6日实际取得 | 接受实际当前来源与明确单位；不计新前瞻或认证历史公开 | 后续只取得真实新统计日版本 |
| 基线可代第一个前瞻日 | 当前取得旧统计日能算独立新变化 | 严格SEED/OBSERVE身份与实际钟测试 | 基线明确排除，0新日期/0变化 | 拒绝将旧日期、已有结果或seed混算独立样本 | 实际连续新版本后才计算原变化 |
| 缺日/晚版本可承接计算 | 可插值或将晚到数值用于原收盘 | 官方日历、连续性与使用时点必要测试 | 缺日和未到available_at均NOT_COMPUTED，错误字段/标的NO_VIEW | 拒绝伪造可知信息；接受实现门 | 不改日期/缺失规则救数据覆盖 |
| 份额增量即现金需求 | 新份额能直接给出人民币流入或国家队 | 原单位/值与制度用途分开，保留拆分未知 | 原份额可以描述，拆分/现金流/身份未识别；不生成交易信号 | 拒绝自动资金归因和策略晋升 | 不同机制/完整来源和固定金融用途另登记 |
| 新观察器提高策略收益 | 数据可得即意味着收益夏普改善 | 实现、真实种子与10测试；没有模型/账户 | 新收益夏普NOT_COMPUTED，数值金融NOT_ADMITTED | 接受数据基础的实质进展，未接受预测或盈利优势 | 先实际新版本，再完整共同池策略与独立验证 |

旧G01—G04/八失败规则及2220源关闭保持；本轮只获取当前原文/最新值，未重扫历史。观察器不是原E03替代，未建立自动任务或verified wait。最早8日/9日实际取得后才可能配对，日历不计结果。

依据：[完整结果](../{report})、[固定合同]({source_card})、[实际统计](../{summary})、[十项测试回执](../{relative(OUT/'observer_tests_receipt.json')})。
"""
    prepared = []
    for name, content in [("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                          ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)]:
        p = ROOT / f"docs/{name}.md"
        old = p.read_bytes()
        title, separator, body = old.partition(b"\n")
        if not separator:
            raise ValueError("事实文件缺少标题分隔。")
        new = title + separator + ("\n" + content.strip() + "\n\n").replace("\n", "\r\n").encode("utf-8") + body
        prepared.append((p, old, new, body))
    state.update({"updated_at": source.now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
        "latest_technical_decision": "TECH.R244", "latest_registration_decision": "TECH.R243", "latest_result": summary,
        "latest_report": report, "latest_completed_study": relative(OUT), "latest_research_status": result["status"],
        "current_study": "510300_CURRENT_OFFICIAL_SHARE_SOURCE_V1",
        "current_phase": "CURRENT_OFFICIAL_SHARE_OBSERVER_IMPLEMENTED_SEED_ONLY_NO_FINANCIAL_ADMISSION",
        "latest_progress": "真实直接官方份额与字段核对、原版本实际钟、完整独立日频观察器和10测试已完成；基线不计前瞻。",
        "current_priority": "取得真实连续新官方份额版本；同时新同工具事前数量预期仍待合格来源。旧份额/预期家族不重跑，不按已知盈亏补门。",
        "next_research_question": "支持出现后的价格响应与新增真实可知份额/传导版本，能否区分持续与短反弹？数值用途未准入，先真实来源。",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "current_financial_candidate_admission": "R240_FIXED_REJECTION_R244_DATA_OBSERVER_ONLY_NOT_ADMITTED",
        "new_accounts_in_current_phase": 0, "necessary_tests_passed_in_current_phase": 10,
        "current_phase_trial_accounting": result,
        "goal_turn_classification": "PROGRESS_R243_R244_DIRECT_OFFICIAL_SOURCE_OBSERVER_IMPLEMENTED_AND_SEEDED",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "latest_goal_service_status": "active",
        "latest_goal_service_status_observed_at": service["observed_at_utc"],
        "latest_goal_tool_status_receipt": relative(OUT / "goal_service_status_after_delivery.json"),
        "current_goal_turn_actual_work": {"new_official_requests": 7, "direct_official_seed": 1, "new_source_observer_implemented": True,
                                          "necessary_tests": 10, "new_prospective_share_dates": 0, "new_financial_runs": 0},
        "next_source_matching_proposal": {**state["next_source_matching_proposal"],
            "status": "PARTIAL_CURRENT_OFFICIAL_SHARE_SOURCE_IMPLEMENTED_POLICY_EXPECTATION_NOT_ADMITTED",
            "current_fund_source_result": summary,
            "new_version_observer_protocol": relative(OUT / "daily_observer_protocol.json"),
            "policy_numeric_expectation_status": "PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN",
            "financial_admission": "NOT_ADMITTED", "new_financial_runs": 0},
        "independent_official_share_observer": {"protocol": relative(OUT / "daily_observer_protocol.json"),
            "code": "research/fund_share_daily_snapshot_observer_v1.py", "status": "READY_NEW_VERSION_OBSERVATION_ONLY",
            "seed": relative(OUT / "seed_snapshot/normalized.json"), "new_observed_statistics_dates": 0,
            "new_unit_changes": 0, "first_allowed_statistics_date": "2026-10-08", "first_capture_window_start": "2026-10-08T23:05:00+08:00",
            "automatic_collection_created": False, "live_process_handle": None, "financial_admission": "NOT_ADMITTED"}})
    with (OUT / "state_before_TECH_R244.json").open("xb") as stream:
        stream.write(old_state)
    documents = []
    for p, old, new, body in prepared:
        if p.read_bytes() != old:
            raise RuntimeError("准备之后事实文档改变。")
        with (OUT / f"{p.stem}_before_TECH_R244.md").open("xb") as stream:
            stream.write(old)
        p.write_bytes(new)
        if not p.read_bytes().endswith(body):
            raise AssertionError("原正文没有保持。")
        documents.append({"path": relative(p), "old_body_preserved_exact": True})
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    saved = source.read(STATE)
    for keys, name in [(FORWARD, "forward_before"), (FINANCE, "financial_before")]:
        if {k: saved[k] for k in keys} != protocol[name]:
            raise AssertionError("原前瞻或金融字段没有保持。")
    source.write(OUT / "project_state_update_receipt.json", {"at": source.now(), "decision": "TECH.R244",
        "documents": documents, "actual_state_updates": 1, "forward_preserved_exact": list(FORWARD), "finance_preserved_exact": list(FINANCE),
        "new_financial_runs": 0, "new_prospective_share_dates": 0, "goal_status": "active", "goal_achieved": False})
    print("R243—R244四份长期事实和状态已一次更新；E03与R240金融身份精确保持。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="当前官方份额来源与观察器交付")
    parser.add_argument("action", choices=("deliver", "close"))
    args = parser.parse_args()
    {"deliver": deliver, "close": close}[args.action]()


if __name__ == "__main__":
    main()
