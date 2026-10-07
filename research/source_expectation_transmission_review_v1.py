"""政策预期、券商方法与宽基传导的隔离来源审查；不拟合、不生成账户。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_source_expectation_transmission_review_v1"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
BROKERS = ROOT / "reports/research/510300_broker_cycle_inspiration_v1"
SHARED = ROOT / "reports/research/510300_shared_cash_source_ownership_v1"
CONTEXT = ROOT / "reports/research/510300_core_support_complement_description_v1/implementation_v1_0_1"
NODES = ROOT / "reports/research/510300_support_price_acceptance_v1/results/全部来源首次原ETF可知_宣布与确认分开.parquet"
POINTS = CONTEXT / "results/全部12支持接受_公布锚价格时钟与宏观原值.parquet"
TRADES = SHARED / "results/主机制全部真实支持补充_赢亏与费用复本完整.parquet"
REGISTER, DECISION = "TECH.R241", "TECH.R242"
FORWARD = (
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation", "next_experiment",
)
FINANCE = ("latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
           "current_new_strategy_return_sharpe", "latest_actual_financial_primary_four_scene_metrics")
SOURCES = (
    ("GOV_202501", "政府原文：2025年央行任务与此前政策指引", "https://www.gov.cn/lianbo/bumen/202501/content_6996406.htm"),
    ("GOV_20200327", "政府原文：2020年3月27日政策安排", "https://app.www.gov.cn/govdata/gov/202003/27/456604/article.html"),
    ("CMB_20230613", "招商银行研究院：降息对股债汇的影响", "https://pdf.dfcfw.com/pdf/H3_AP202306191591057214_1.pdf"),
    ("HT_20200329", "海通原PDF：2020年3月末固定收益研究", "https://www.haitong.com/jfimg/colimg/upload/20200330/53751585537266002.pdf"),
    ("DS_REPO", "Das和Song作者仓库：政策冲击数据说明", "https://raw.githubusercontent.com/wtsong/china_mpshocks/main/README.md"),
    ("DS_EVENTS", "Das和Song作者仓库：事件层政策冲击数据", "https://raw.githubusercontent.com/wtsong/china_mpshocks/main/china_mpshocks.csv"),
)
FAMILIES = {
    "早期6个月与446政策因子": ROOT / "reports/research/510300_expectations_policy_evidence_v1/results/summary.json",
    "77个月共识增量": ROOT / "reports/research/510300_money_consensus_increment_v2/status.json",
    "非线性预期差联合状态": ROOT / "reports/research/510300_multidim_money_surprise_score_v1/result.json",
    "CPI与FOMC历史预期锚": ROOT / "reports/research/510300_historical_index_expectation_anchor_v1/result.json",
    "历史信用预期差": ROOT / "reports/research/510300_historical_index_credit_expectation_gap_v1/result.json",
    "行业主线输入时钟": ROOT / "reports/research/510300_broker_mainline_source_preflight_v1/summary.json",
    "成分队列失效账户": ROOT / "reports/research/510300_broker_cohort_failure_v1/summary.json",
    "行业领涨失效账户": ROOT / "reports/research/510300_industry_leader_failure_v1/summary.json",
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def rel(path):
    return path.absolute().relative_to(ROOT).as_posix()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def table(name, frame):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists():
        raise RuntimeError("结果已存在，禁止覆盖：" + name)
    frame.to_parquet(p.with_suffix(".parquet"), index=False)
    frame.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def register():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("来源审查已经登记。")
    state = read(STATE)
    files = [NODES, POINTS, TRADES, SHARED / "summary.json", SHARED / "next_source_expectation_transmission_review_proposal.json",
             BROKERS / "source_author_and_method_review.json", *FAMILIES.values()]
    for row in read(BROKERS / "source_author_and_method_review.json")["checks"]:
        folder = BROKERS / "sources" / row["source_id"]
        files.extend([folder / "receipt.json", folder / "text.txt"])
    frozen = [{"path": rel(p), "sha256": digest(p)} for p in files]
    write(OUT / "protocol.json", {
        "at": now(), "registration": REGISTER, "decision": DECISION,
        "status": "DESCRIPTIVE_SOURCE_REVIEW_REGISTERED_NOT_FINANCIAL_ADMISSION",
        "question": "新增的事前预期、政策实际增量和主线到ETF的传导证据是否可用，旧家族是否已经否定相同用途？",
        "already_seen_before_registration": "R240全部盈亏、五真实补充、7券商报告、旧共识77月与非线性结果、当前网页检索和打开内容均已读；不声称盲测或首次探索。",
        "browser_exploration_before_registration": {"search_queries": 11, "open_targets": 8, "role": "查找与阅读；独立于下方6次本机原响应归档"},
        "new_archive_requests": [{"id": sid, "title": title, "url": url} for sid, title, url in SOURCES],
        "request_rule": "每地址一次本机GET，不重试，HTTP失败保留；这不是把此前web读取次数计为零。",
        "scope": {"verified_broker_reports": 7, "old_policy_nodes": 65, "saved_support_points": 12,
                  "actual_complements": 5, "report_case_pairs": 35, "saved_source_families": 8},
        "purpose": "原作者方法、历史方向指引、同变量事前数量预期、政策实际值、ETF传导和事后归因分开；报告内部日期不等于公开上界。",
        "expectation_match": ["政策工具/标的", "数量或利率及单位", "预期针对的公布时点", "事前原文与公开上界", "首版与修订身份"],
        "two_clocks": "预期必须在政策公布前；后续评论必须在策略决定前。二者不能互换。",
        "transmission": "行业相对收益不等于510300方向；成交量不等于净申购；成分与权重必须使用当时版本。",
        "minute_bars": 0, "event_dataset_rule": "只读作者事件层CSV与方法，不收集或交易分钟线；高频估计变量尚未认证为本项目可实时取得的日线因子。",
        "new_fits": 0, "new_labels": 0, "new_accounts": 0, "new_financial_runs": 0,
        "no_rescue": "R240/R232/R236及旧共识、旧联合评分和旧传导账户保持；不更改窗口、退出、时期或来源优先级重跑。",
        "frozen_sources": frozen, "review_code_sha256": digest(Path(__file__)),
        "forward_before": {k: state[k] for k in FORWARD}, "financial_before": {k: state[k] for k in FINANCE},
        "independent_validation": "NOT_ESTABLISHED", "financial_admission": "NOT_ADMITTED", "goal_achieved": False,
    })
    print("R241描述来源审查已登记，6个固定归档地址，0新增金融/拟合。", flush=True)


def fetch(source):
    sid, title, url = source
    folder = OUT / "sources" / sid
    folder.mkdir(parents=True, exist_ok=False)
    result = {"id": sid, "title": title, "url": url, "started_at": now(), "retries": 0,
              "first_vintage": "NOT_CERTIFIED", "historical_publication_clock": "NOT_ESTABLISHED"}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        response = session.get(url, timeout=(10, 30))
        result.update(http_status=response.status_code, final_url=response.url,
                      response_headers={k: response.headers.get(k) for k in ("Content-Type", "Date", "Last-Modified")})
        raw = response.content
        if len(raw) > 20_000_000:
            raise ValueError("响应超过20MB本次上限。")
        raw_path = folder / ("original.pdf" if raw.startswith(b"%PDF") else "response.body")
        with raw_path.open("xb") as stream:
            stream.write(raw)
        result.update(raw_path=rel(raw_path), bytes=len(raw), sha256=digest(raw_path))
        response.raise_for_status()
        if raw.startswith(b"%PDF"):
            with pdfplumber.open(io.BytesIO(raw)) as document:
                pages = [{"page": i, "text": p.extract_text() or ""} for i, p in enumerate(document.pages, 1)]
            write(folder / "pages.json", pages)
            text = "\n\n".join(p["text"] for p in pages)
            result["pages"] = len(pages)
        else:
            response.encoding = response.apparent_encoding or "utf-8"
            text = response.text
            if "<html" in text[:4000].lower() or "<!doctype" in text[:4000].lower():
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
        (folder / "text.txt").write_text(text, encoding="utf-8")
        result.update(status="SAVED_RESPONSE_PENDING_USE_REVIEW", text_characters=len(text))
    except Exception as error:
        result.update(status="SOURCE_REQUEST_OR_EXTRACTION_FAILED", error_type=type(error).__name__, error=str(error))
    finally:
        session.close()
        result["ended_at"] = now()
        write(folder / "receipt.json", result)
    print(f"{sid}：{result['status']}，正文字符{result.get('text_characters', 0)}。", flush=True)
    return result


def collect():
    read(OUT / "protocol.json")
    if (OUT / "collection_started.json").exists():
        raise RuntimeError("本机归档已经开始，不重试或覆盖。")
    write(OUT / "collection_started.json", {"at": now(), "requests": 6, "financial_runs": 0})
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, SOURCES))
    write(OUT / "collection_receipt.json", {"at": now(), "local_archive_requests": 6,
          "saved_extractable_responses": sum(r["status"].startswith("SAVED") for r in receipts),
          "results": receipts, "new_market_bar_requests": 0, "new_financial_runs": 0})


def verify_frozen(protocol):
    for row in protocol["frozen_sources"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("原事实文件发生变化：" + row["path"])
    if digest(Path(__file__)) != protocol["review_code_sha256"]:
        raise RuntimeError("登记后的审查代码发生变化。")


def review():
    protocol = read(OUT / "protocol.json")
    verify_frozen(protocol)
    if (OUT / "summary.json").exists() or (OUT / "review_started.json").exists():
        raise RuntimeError("完整审查已经开始或完成。")
    collection = read(OUT / "collection_receipt.json")
    annotations = read(OUT / "source_use_annotations.json")
    if set(annotations) != {x[0] for x in SOURCES}:
        raise ValueError("六个固定来源必须完整保留，包括失败和未知。")
    write(OUT / "review_started.json", {"at": now(), "financial_runs": 0})
    nodes = pd.read_parquet(NODES)
    points = pd.read_parquet(POINTS)
    trades = pd.read_parquet(TRADES)
    cases = trades.loc[trades.cost.eq("STRESS")].copy()
    if len(nodes) != 65 or len(points) != 12 or len(trades) != 10 or len(cases) != 5:
        raise ValueError("原65节点、12点位、5真实补充范围不同。")
    if cases.entry_origin.duplicated().any():
        raise ValueError("费用复本误当独立机会。")
    cases = cases.merge(points, left_on="entry_origin", right_on="origin", validate="one_to_one", suffixes=("", "_point"))
    if len(cases) != 5 or cases.point_id.isna().any():
        raise ValueError("五真实成交没有完整对应原决定信息。")
    keep = ["point_id", "entry_origin", "decision_time", "source_node_ids", "source_available_upper", "support_setup_date",
            "support_setup_low", "support_setup_high", "accepted_cash_close", "daily_hist", "known_es95",
            "pmi_original_level", "pmi_original_month_change", "entry_date", "entry_quantity", "exit_date", "net_pnl", "net_return"]
    cases = cases[keep].copy()
    cases["same_tool_prerelease_quantitative_expectation"] = "NOT_ESTABLISHED"
    cases["causal_explanation"] = "NOT_IDENTIFIED"
    cases["historical_role"] = "ALREADY_SEEN_DEVELOPMENT_NOT_NEW_LABEL"
    table("全部五真实补充_原决定技术宏观值与实际盈亏分开", cases)
    point_review = points.copy()
    point_review["prerelease_consensus_for_underlying_policy_tool"] = "NOT_ESTABLISHED"
    point_review["new_numeric_financial_admission"] = "NOT_ADMITTED"
    table("全部12原支持点位_数量预期与政策支持不同", point_review)
    node_review = nodes.copy()
    node_review["expectation_variable_status"] = "NOT_ESTABLISHED_IN_ORIGINAL_RECEIPT_SCHEMA"
    node_review["same_common_source_is_not_independent_vote"] = True
    table("全部65原节点_实际公告和数量预期用途区分", node_review)
    source_code = (ROOT / "research/broker_cycle_sources_v1.py").read_text(encoding="utf-8-sig")
    parsed = ast.parse(source_code)
    catalog = next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", "") == "SOURCES" for t in n.targets))
    catalog = {x[0]: x for x in catalog}
    review_rows, pairs = [], []
    for check in read(BROKERS / "source_author_and_method_review.json")["checks"]:
        sid = check["source_id"]
        receipt = read(BROKERS / "sources" / sid / "receipt.json")
        declared = "2023-06-25" if sid == "GJ_MAINLINE_PDF" else catalog[sid][3]
        title = receipt["title"]
        row = {"id": sid, "broker": receipt["broker"], "title": title, "internal_report_date": declared,
               "url": receipt["requested_url"], "provenance": check["provenance_after_body_check"],
               "historical_first_vintage": "NOT_CERTIFIED", "use": "METHOD_INSPIRATION",
               "same_tool_preannouncement_quantitative_consensus": "NOT_ESTABLISHED",
               "date_limitation": "内部/目录报告日期不自动证明公开上界；招商目标现有作者转载2026-06-03，26页原PDF未取得。"}
        review_rows.append(row)
        for _, case in cases.iterrows():
            before = pd.Timestamp(declared) < pd.Timestamp(case.entry_origin)
            pairs.append({"report_id": sid, "case_id": case.point_id, "report_internal_date": declared,
                          "original_decision_date": case.entry_origin, "internal_date_precedes_decision": bool(before),
                          "historically_authenticated_forecast_before_policy": "NOT_ESTABLISHED",
                          "historically_authenticated_comment_before_entry": "NOT_ESTABLISHED",
                          "role": "DEVELOPMENT_METHOD_ONLY", "case_net_pnl_not_used_for_source_selection": float(case.net_pnl)})
    table("七券商报告_方法用途与历史预测分开", pd.DataFrame(review_rows))
    table("全部35报告点位组合_内部日期与公开钟分开", pd.DataFrame(pairs))
    annotated = []
    for receipt in collection["results"]:
        sid = receipt["id"]
        annotation = annotations[sid]
        if not receipt["status"].startswith("SAVED") and annotation["local_body_verified"]:
            raise ValueError("失败响应不能标为本机正文已验证。")
        annotated.append({**receipt, **annotation})
    write(OUT / "new_source_use_review.json", {"at": now(), "sources": annotated,
          "note": "一份报告的市场预测、另一工具预期与政策实际变化不可拼成同工具事前超预期。"})
    families = []
    for name, path in FAMILIES.items():
        value = read(path)
        families.append({"name": name, "path": rel(path), "status": value.get("status", value.get("account_status", "REFER_TO_SAVED_RESULT")),
                         "current_verified_result": value})
    write(OUT / "saved_family_adjudications.json", {"at": now(), "families": families,
          "stage_correction": "早期6个月、来源扩展61对和后来合并77个月为不同已保存阶段；最终V2使用77个月，不混作新增来源。"})
    v2 = read(FAMILIES["77个月共识增量"])
    nonlinear = read(FAMILIES["非线性预期差联合状态"])
    primary = v2["adjudications"][0]
    if primary["MSE_gate_pass"] or nonlinear["message_increment_primary"]:
        raise ValueError("旧预期检验裁决与已阅读事实不符，停止归档。")
    new_forecasts = sum(a.get("same_tool_prerelease_numeric_admitted", False) for a in annotated)
    if new_forecasts:
        raise ValueError("本描述审查不能默默准入新数量因子，须另立用途。")
    result = {"at": now(), "registration": REGISTER, "decision": DECISION,
              "status": "COMPLETED_SOURCE_REVIEW_NO_NEW_MATCHED_EXPECTATION_OR_ETF_TRANSMISSION_FINANCIAL_ADMISSION",
              "verified_broker_reports": 7, "report_case_pairs": len(pairs), "original_policy_nodes": len(nodes),
              "original_support_points": len(points), "actual_unique_complements": len(cases), "saved_source_families": len(families),
              "local_archive_requests": 6, "saved_extractable_responses": collection["saved_extractable_responses"],
              "new_same_tool_prerelease_numeric_forecasts": 0, "new_complete_etf_transmission_inputs": 0,
              "old_consensus_admitted_months": v2["admitted_consensus_months"], "old_consensus_evaluation_events": primary["evaluation_events"],
              "old_consensus_relative_mse_improvement": primary["relative_MSE_improvement"],
              "old_nonlinear_full_account_sharpe": nonlinear["full_continuous_account_net_sharpe"],
              "old_nonlinear_full_account_cagr": nonlinear["full_continuous_account_cagr"],
              "old_nonlinear_primary_profit_concentration": nonlinear["largest_cycle_share_of_primary_net_profit"],
              "new_fits": 0, "new_labels": 0, "new_accounts": 0, "new_financial_runs": 0, "new_market_bar_requests": 0,
              "minute_bars": 0, "adaptive_strategy_direction": "ACCEPTED_RESEARCH_DIRECTION_NOT_VALIDATED_STRATEGY",
              "financial_admission": "NOT_ADMITTED", "causal_attribution": "NOT_IDENTIFIED",
              "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
              "search_selection_correction": "NOT_COMPUTED", "goal_achieved": False}
    write(OUT / "summary.json", result)
    write(OUT / "next_source_matching_proposal.json", {
        "at": now(), "status": "PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN",
        "question": "能否取得同政策工具的事前数量预期或带真实公布钟的日度ETF份额/成分传导数据？",
        "priority": "核对已有全部65节点对应原文，不按五笔输赢选择；先取得不同于M1/M2预期差的工具/公布时点匹配信息。",
        "required_expectation_fields": protocol["expectation_match"],
        "fund_fields": ["ETF份额观察日", "实际公开上界", "原始数值与单位", "申赎净变化定义", "修订身份"],
        "controls": "若来源够用，再唯一固定支持背景与传导/失效动作；相同共同池、价格基准、既有支持机制和完整费用账户比较。",
        "stop": "重复旧共识代理、全样本政策冲击、只有事后评论或缺真实钟，则该输入NOT_ADMITTED；不据此否定所有适应性策略。",
        "new_financial_runs": 0, "new_accounts": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False})
    render_report(result, cases, annotated)
    verify_frozen(protocol)
    print("R242完整审查已保存：7报告、35配对、65节点、12点位、5真实成交、8旧家族；0新金融。", flush=True)


def render_report(result, cases, sources):
    lines = ["# 政策预期、券商框架与510300传导：来源审查结果", "", "2026-10-06；TECH.R241—R242。",
             "", "市场变化时允许另立适应性策略；本轮确认该研究方向，尚未得到收益或夏普提高的新策略。最新实际金融仍R240固定失败。",
             "", "## 券商启发如何落到本项目", "",
             "招商的多维周期框架可转成当时支持、价格接受、参与扩散、继续持有价值和结构失效的条件关系。阶段必须由当时信息确定；已经读过的历史只能作开发，不能事后按每段赢家切换。行业主线对股票组合的相对收益也不能直接搬到510300。",
             "", "现有7份券商内容对应35个报告与真实点位组合，全部保留。报告内部日期、作者转载时间与首次公开上界分开；部分报告在案例之后出版，可供今天设计方法，不能倒填成当时的政策预测。原目标报告现存作者转载，原26页PDF未取得。",
             "", "## 必须避免重复的旧研究", "",
             f"早期6个月代理之后，最终共识V2已扩展到77个月、76个成熟标签。旧口径24次评价加入预期差的相对MSE改善为{result['old_consensus_relative_mse_improvement']:.6%}，两半改善均负；该固定用途REJECTED_FROZEN。2025新M1只有16个成熟事件，未过60群门；资金C/D没有足量公开钟，仍NOT_RUN。这里拒绝的是该固定用途，不代表所有预期信息无用。",
             "", f"非线性联合评分也已经做过：完整2021—2025账户净Sharpe {result['old_nonlinear_full_account_sharpe']:.6f}、CAGR {result['old_nonlinear_full_account_cagr']:.4%}；2024—2025局部Sharpe约1.24不能替代完整结果，主要期单笔占净利润{result['old_nonlinear_primary_profit_concentration']:.2%}。主要期模型没有用到预期差分裂，不能把收益归给新消息增量。",
             "", "446次作者政策因子使用全样本PCA，原角色是历史测量；不是可按当时版本生成的实时预期。另有CPI/FOMC历史锚、信用预期差、成分队列与行业领涨失效家族，原结果均链接保留，不将其更名为未尝试方向。",
             "", "## 五个真实补充：当时信息与后来结果", "",
             "以下只复用R238原决定值与R240压力费用实际完整周期，费用复本不算独立样本。PMI为原已公布值，没有按输赢反向设门。",
             "", "| 原决定日 | 原源 | 已接受收盘 / 原锚上沿 | 日MACD柱 | 当时PMI水平 / 变动 | 实际进入份额 | 实际净盈亏（元） |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in cases.iterrows():
        lines.append(f"| {pd.Timestamp(row.entry_origin).date()} | {row.source_node_ids} | {row.accepted_cash_close:.4f} / {row.support_setup_high:.4f} | {row.daily_hist:.6f} | {row.pmi_original_level:.1f} / {row.pmi_original_month_change:+.1f} | {row.entry_quantity:.0f} | {row.net_pnl:+.2f} |")
    lines.extend(["", "五例均有当时支持与价格接受，结果仍是3盈2亏。这说明现有条件没有区分持续与短反弹；并不能从五例推断一个可靠概率，也不能仅给两亏损追加新门。具体结果及所有12原接受点、65来源节点和原版本保存在完整表。政策导致其盈亏的因果关系尚未识别。",
                  "", "## 新查来源的实际用途", "", "| 来源 | 用途结论 | 主要限制 |", "|---|---|---|"])
    for source in sources:
        lines.append(f"| [{source['title']}]({source['url']}) | {source['use']} | {source['limitation']} |")
    lines.extend(["", f"本机固定6次GET，成功提取{result['saved_extractable_responses']}个响应，失败如实保留；之前11条网页搜索与8次打开另记，不伪计零。",
                  "", "政策的方向指引只能说明此前存在支持倾向，不能凭它算出某次操作幅度相对市场预期的数量差。政策出台后的机构点评也不属于出台前共识。PDF内部日期、文件路径日期和服务器Last-Modified都不单独证明历史首次公开；报告中的国债利率预测不能替代7天逆回购利率预期。",
                  "", "主线传导仍有明确缺口：旧行业表7536个available_at由生效日程序生成，不能当历史首发；成员/回报直接日期只有2823/2855日，缺32日；日度行业贡献/加权宽度现成表不存在。之前队列与行业领涨金融用途已经失败，本轮不重新运行或更改其失效阈值。",
                  "", "## 下一步与停止条件", "",
                  "优先取得不同信息：同政策工具、同公布时点的事前数量预期，或有真实公开钟的日度ETF份额与成分传导。对全部65节点核对，成功与失败一同保留；依赖旧M1/M2代理、全样本冲击或事后评论的部分不准入。没有这种输入时，不按五笔输赢补阈值。",
                  "", "取得合格输入后才另登记一个完整适应性用途：固定输入如何改变进入、确认、持有与失效；与同共同池价格基准、已有支持机制比较，同20万元、风险、股息、下一开盘、T+1、整手、两时期两费用。先检验全部实际完整账户净CAGR、Sharpe、DD、p×B和标准期望、集中度与有效次数；不拼接时期或升级事后赢家对照。",
                  "", "本轮0新拟合、0新标签、0新账户、0新金融运行、0分钟柱；新增同工具事前数量预期及完整ETF传导输入均未准入。独立验证、首版认证和选择校正未建立。原E03前瞻13字段精确保持，最早2026-10-08 15:05，日历并不等于已经有新观察。",
                  "", "复算入口：research/source_expectation_transmission_review_v1.py；来源协议protocol.json、原响应sources、new_source_use_review.json、saved_family_adjudications.json、summary.json及results全部表。",
                  "", "本分支关闭的是此次来源审查；策略收益/夏普目标仍未实现，旧原策略和固定拒绝结果保持。"])
    with (OUT / "政策预期_券商框架与ETF传导_完整来源结论.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def close():
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    verify_frozen(protocol)
    receipt = OUT / "project_state_update_receipt.json"
    if receipt.exists():
        raise RuntimeError("四份长期事实和项目状态已经更新，不重复。")
    service = read(OUT / "goal_service_status_after_review.json")
    if service["goal"]["status"] != "active":
        raise ValueError("目标服务没有确认active，不能写入虚假状态。")
    raw = STATE.read_bytes()
    state = json.loads(raw.decode("utf-8-sig"))
    for keys, name in ((FORWARD, "forward_before"), (FINANCE, "financial_before")):
        if {k: state[k] for k in keys} != protocol[name]:
            raise ValueError("原独立前瞻或实际金融字段改变。")
    report = rel(OUT / "政策预期_券商框架与ETF传导_完整来源结论.md")
    summary_path = rel(OUT / "summary.json")
    note = f"""### TECH.R241—R242：政策预期、券商方法和ETF传导完整来源审查（2026-10-06）

已完成7份原券商内容/35报告与真实点位配对、65原来源节点、12原支持点、5实际共同账户补充和8旧家族核对；固定6本机归档、{summary['saved_extractable_responses']}正文可提取，失败保存。登记前已经看到全部旧盈亏及网页内容，本轮描述审查不是盲测。新增金融/账户/拟合/标签/行情柱/分钟柱均0。

当前预期数据不是只有6个月：较新V2准入77个月、成熟76、旧口径24评价MSE相对改善−4.9830%，固定用途拒绝；2025新口径16个成熟未过60门，资金C/D仍NOT_RUN。非线性预期差与指数状态也已做过，完整2021—2025账户净Sharpe .699294、CAGR .7394%；主要期局部1.24155与82.263%单笔净利润集中不能证明新消息增量。原446全样本PCA不准实时信号。旧预期锚/信用/成分队列/行业领涨结果继续保持，不更名重跑。

券商适应性框架接受为研究方向，尚未验证为有效策略；方向指引、政策实际操作、同工具事前数量共识、出台后评论和ETF传播分开。新来源未提供准入的同工具事前数量预期或完整ETF传导输入。内部日期/路径日期/Last-Modified不能单独当首次公開；作者事件层政策冲击只作方法与历史对照，没有采集分钟线。五真实补充3盈2亏均有原支持及价格接受；不能按两亏追加门或声称政策已被因果识别。

下一提案为同工具预期/真实钟ETF份额或成分传导来源合同，PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN、0待跑金融；旧R240/R232/R236不救改、不升格对照、不拼时期。最新实际金融仍TECH.R240完整固定拒绝，收益/夏普目标未达；本轮PROGRESS、目标服务active、受阻0，原13独立前瞻精确保持。独立NOT_ESTABLISHED、首版NOT_CERTIFIED、DSR/PBO未算。

依据：[完整来源结论](../{report})、[完整实际统计](../{summary_path})、[旧家族逐项结果](../{rel(OUT/'saved_family_adjudications.json')})、[下一来源提案](../{rel(OUT/'next_source_matching_proposal.json')})。
"""
    decisions = f"""### TECH.R241—R242：新增信息与旧失败的用途裁决（2026-10-06）

本次描述来源审查完整结束，0新金融；看过原结果后登记，未声称独立验证。最新实际金融R240及13前瞻字段保持，目标active未完成，受阻0。

| 方向 | 假设 | 验证方法 | 结果 | 为什么接受/拒绝 | 是否重新验证 |
|---|---|---|---|---|---|
| 券商多维适应性框架 | 市场状态变化时动作可变化 | 7原内容与35报告点位时间/用途配对 | 可供开发方法，未给出本项目认证逐日状态 | 接受新完整适应性策略研究，不接受直接获利结论 | 是，固定当时分类及全部动作后完整账户与独立检验 |
| 再加原M1/M2预期差 | 尚未尝试的共识能提供消息增量 | 核对早6、扩展61、最终77三个阶段及V2原评价 | 24评价误差增加约4.98%，两半均负；新口径样本不足 | 拒绝把旧固定失败包装新因子；不是否定一切预期 | 本配置不重跑救援，不同工具信息或新样本另登记 |
| 原预期差非线性打分 | 局部高夏普说明消息有效 | 原3模型族、47评分、6账户及完整期结果 | 完整Sharpe .699294/CAGR .7394%，主要期集中82.263%，主要期预期差没有树分裂 | 拒绝局部与集中收益晋升、拒绝消息增量归因 | 不调旧树；不同机制与独立样本另检验 |
| 政策指引等于超预期 | 此前支持倾向可当数量共识 | 原65节点、6新归档、政策前与交易前两时钟及工具/单位匹配 | 新同工具数量预期0准入，方向指引不是市场共识数值 | 接受历史背景，拒绝精确超预期因子 | 是，必须取得相同工具的事前原数量/公开钟/版本 |
| 原PDF日期证明事前可知 | 报告内部日期早于交易即可填历史 | 7报告及新点评公开上界分别核对 | 内部/目录/路径日期不证明首发；后政策点评不是前政策预测 | 拒绝将事后叙述转换历史预测 | 可补原公开钟证据，原未知保持 |
| 446政策因子与作者冲击实时化 | 学术冲击估计直接适合实盘日线 | 原全样本PCA说明与另一作者2006—2020事件层资料 | 历史测量可解释方法，没有本项目实时版本认证 | 接受方法/历史对照，拒绝倒填实时惊喜 | 先完成实时重建信息合同；不新增分钟线 |
| 主线必传导510300 | 行业主线相对上涨足以指导宽基 | 原成员/权重/行业钟与R216/R222结果 | 7536钟程序生效生成、2823/2855直接日期，旧两配置已失败 | 拒绝现有表直接准入和旧阈值重跑；不否定所有传导机制 | 是，真实钟/版本和不同信息来源先准入 |
| 五补充解释即因果 | 支持与价格接受解释所有持续 | 全部12原点、五真实压力交易及全部盈亏保留 | 3盈2亏，原技术/PMI值完整复用 | 接受描述事实；因果和可靠概率未识别 | 不按已知输赢补门，须完整机会集与独立验证 |

依据：[完整来源报告](../{report})、[实际结果](../{summary_path})、[旧家族裁决](../{rel(OUT/'saved_family_adjudications.json')})。本研究结束是分支关闭，不是项目停止或目标达成。
"""
    prepared = []
    for name, text in (("PROJECT_STATE", note), ("PROJECT_STATE_TECHNICAL_LINE", note),
                       ("RESEARCH_DECISIONS", decisions), ("RESEARCH_DECISIONS_TECHNICAL_LINE", decisions)):
        p = ROOT / f"docs/{name}.md"
        old = p.read_bytes()
        title, sep, body = old.partition(b"\n")
        if not sep:
            raise ValueError("事实正文没有标题分隔。")
        new = title + sep + ("\n" + text.strip() + "\n\n").replace("\n", "\r\n").encode("utf-8") + body
        prepared.append((p, old, new, body))
    write(OUT / "state_before_TECH_R242.json", state)
    state.update({"updated_at": now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
                  "latest_technical_decision": DECISION, "latest_registration_decision": REGISTER,
                  "latest_result": summary_path, "latest_report": report, "latest_completed_study": rel(OUT),
                  "latest_research_status": summary["status"], "current_study": "510300_SOURCE_EXPECTATION_TRANSMISSION_REVIEW_V1",
                  "current_phase": "SOURCE_EXPECTATION_TRANSMISSION_REVIEW_COMPLETE_NO_NEW_NUMERIC_ADMISSION",
                  "latest_progress": "核对77月与既有非线性真实裁决，完整7报告/65节点/12点/5成交及新源两时钟用途；0新增金融。",
                  "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
                  "current_financial_candidate_admission": "R240_FIXED_REJECTION_R242_NO_NEW_MATCHED_SOURCE_ADMITTED",
                  "new_accounts_in_current_phase": 0, "current_phase_trial_accounting": summary,
                  "goal_turn_classification": "PROGRESS_R241_R242_COMPLETE_SOURCE_AND_OLD_FAMILY_INFORMATION_REVIEW",
                  "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0,
                  "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": service["observed_at_utc"],
                  "latest_goal_tool_status_receipt": rel(OUT / "goal_service_status_after_review.json"),
                  "current_goal_turn_actual_work": {"source_review_completed": True, "archive_requests": 6,
                                                    "reports": 7, "nodes": 65, "points": 12, "actual_cases": 5, "new_financial_runs": 0},
                  "next_source_expectation_transmission_review_proposal": {**state["next_source_expectation_transmission_review_proposal"],
                      "status": "COMPLETED_SOURCE_REVIEW_NOT_FINANCIAL_ADMISSION", "registration": REGISTER, "decision": DECISION, "result": summary_path},
                  "next_source_matching_proposal": {**read(OUT / "next_source_matching_proposal.json"),
                      "proposal_path": rel(OUT / "next_source_matching_proposal.json")}})
    documents = []
    for p, old, new, body in prepared:
        if p.read_bytes() != old:
            raise RuntimeError("准备之后事实文档发生变化。")
        with (OUT / f"{p.stem}_before_TECH_R242.md").open("xb") as stream:
            stream.write(old)
        p.write_bytes(new)
        if p.read_bytes() != new or not new.endswith(body):
            raise AssertionError("原正文未保持。")
        documents.append({"path": rel(p), "old_body_preserved": True})
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    saved = read(STATE)
    for keys, name in ((FORWARD, "forward_before"), (FINANCE, "financial_before")):
        if {k: saved[k] for k in keys} != protocol[name]:
            raise AssertionError("原前瞻或金融字段未保持。")
    write(receipt, {"at": now(), "decision": DECISION, "documents": documents, "actual_state_updates": 1,
                   "forward_fields_preserved_exact": list(FORWARD), "financial_fields_preserved_exact": list(FINANCE),
                   "new_financial_runs": 0, "goal_status": "active", "goal_achieved": False})
    print("R241—R242四份长期事实与状态已一次更新；原13前瞻及R240金融字段精确保持。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="政策预期与ETF传导隔离来源审查")
    parser.add_argument("action", choices=("register", "collect", "review", "close"))
    args = parser.parse_args()
    {"register": register, "collect": collect, "review": review, "close": close}[args.action]()


if __name__ == "__main__":
    main()
