"""固定六个官方汇总日期请求，保存原件、缺失和字段，不运行模型。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research import macro_2026_source_coverage_intake_v1 as source
from research.finalize_macro_funding_source_contract_v2 import FORWARD

ROOT = source.ROOT
OUT = ROOT / "reports/research/510300_market_margin_official_feasibility_2026_v1"
CARD = ROOT / "docs/510300_2026_MARKET_MARGIN_OFFICIAL_FEASIBILITY_V1.md"
PARENT = ROOT / "reports/research/510300_macro_2026_source_coverage_intake_v1_clock_adapter"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
DATES = ["2026-01-05", "2026-08-14", "2026-09-30"]
SSE = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
SZSE = "https://www.szse.cn/api/report/ShowReport/data"


def specification(exchange, day):
    if exchange == "SSE":
        return SSE, {"isPagination": "true", "beginDate": day.replace("-", ""), "endDate": day.replace("-", ""),
                     "tabType": "", "stockCode": "", "pageHelp.pageSize": "5000", "pageHelp.pageNo": "1",
                     "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "5"}, "https://www.sse.com.cn/"
    return SZSE, {"SHOWTYPE": "JSON", "CATALOGID": "1837_xxpl", "txtDate": day, "tab1PAGENO": "1"}, \
           "https://www.szse.cn/disclosure/margin/margin/index.html"


def number(value):
    result = float(str(value).replace(",", ""))
    source.study.require(np.isfinite(result) and result > 0, "官方金额为空、零或非正，未准入。")
    return result


def decode(exchange, day, payload):
    if exchange == "SSE":
        source.study.require(isinstance(payload, dict) and isinstance(payload.get("result"), list), "沪市无汇总result列表。")
        matching = [x for x in payload["result"] if any(str(v).replace("-", "") == day.replace("-", "") for v in x.values())]
        source.study.require(len(matching) == 1, "沪市目标日期汇总不唯一或缺失。")
        row = matching[0]
        balance = next((k for k in ["rzye", "RZYE"] if k in row), None)
        buy = next((k for k in ["rzmrje", "rzmre", "RZMRJE"] if k in row), None)
        source.study.require(balance is not None and buy is not None, "沪市余额/买入键未按既有定义识别。")
        return {"balance_raw": number(row[balance]), "buy_raw": number(row[buy]), "balance_key": balance,
                "buy_key": buy, "source_unit": "原汇总接口金额，单位解释继承本地既有原定义，未认证首版",
                "raw_row_keys": "|".join(row), "date_identity_exact": True}
    source.study.require(isinstance(payload, list) and len(payload) > 0, "深市无汇总页。")
    main = payload[0]
    metadata, rows = main["metadata"], main["data"]
    source.study.require(metadata["subname"] == day and len(rows) == 1, "深市汇总统计日不一致或不唯一。")
    cols = metadata["cols"]
    source.study.require("融资买入" in cols["jrrzmr"] and "融资余额" in cols["jrrzye"], "深市字段语义不同。")
    source.study.require("亿" in cols["jrrzmr"] and "亿" in cols["jrrzye"], "深市金额单位未知。")
    return {"balance_raw": number(rows[0]["jrrzye"]), "buy_raw": number(rows[0]["jrrzmr"]),
            "balance_key": "jrrzye", "buy_key": "jrrzmr", "source_unit": "亿元；原显示两位小数",
            "raw_row_keys": "|".join(rows[0]), "date_identity_exact": True}


def freeze():
    source.study.require(not OUT.exists(), "有限官方来源申请已存在，不覆盖。")
    sources = [CARD, Path(__file__), PARENT / "summary.json", PARENT / "protocol.json",
               ROOT / "research/factor96_margin_source_repair_v1.py", ROOT / "scripts/collect_market_margin_leverage_v0.py"]
    (OUT / "raw").mkdir(parents=True)
    (OUT / "results").mkdir()
    source.save_json(OUT / "protocol.json", {"at": source.study.now(), "study": "510300_MARKET_MARGIN_OFFICIAL_FEASIBILITY_2026_V1",
        "decision": "TECH.R201", "sources": [{"path": source.rel(p), "sha256": source.h(p)} for p in sources],
        "dates": DATES, "date_selection": "覆盖端点而非事后收益", "exchanges": ["SSE", "SZSE"],
        "maximum_data_gets": 6, "maximum_official_definition_page_opens": 2, "automatic_retry": False,
        "alternate_source_or_domain": False, "timeout_seconds": 20, "tls_verification": True, "uses_credentials": False,
        "new_financial_candidate": False, "new_models": 0, "new_accounts": 0, "new_labels": 0,
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED"})
    print("TECH.R201固定六日期官方GET已登记，0金融候选。", flush=True)


def run():
    source.study.require(not (OUT / "RUN_STARTED.json").exists(), "来源已开始，不重复请求。")
    protocol = source.study.read(OUT / "protocol.json")
    for item in protocol["sources"]:
        source.study.require(source.h(ROOT / item["path"]) == item["sha256"], "申请源改变。")
    source.save_json(OUT / "RUN_STARTED.json", {"at": source.study.now(), "maximum_data_gets": 6})
    session = requests.Session()
    session.trust_env = False
    records = []
    try:
        for day in DATES:
            for exchange in ["SSE", "SZSE"]:
                url, params, referer = specification(exchange, day)
                record = {"exchange": exchange, "stat_date": day, "url": url, "params": params,
                          "requested_at": source.study.now(), "attempts": 1, "first_vintage": "UNKNOWN",
                          "historical_available_at": None, "historical_available_at_status": "UNKNOWN_NOT_THIS_RETRIEVAL_TIME"}
                filename = f"{exchange}_{day}"
                try:
                    response = session.get(url, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": referer},
                                           timeout=(20, 20), allow_redirects=False)
                    raw = OUT / "raw" / (filename + ".response")
                    raw.write_bytes(response.content)
                    record.update(received_at=source.study.now(), http_status=response.status_code,
                                  server_date=response.headers.get("Date"), response_bytes=len(response.content),
                                  raw_path=source.rel(raw), raw_sha256=source.h(raw))
                    response.raise_for_status()
                    record.update(decode(exchange, day, response.json()), status="PASS_OFFICIAL_DATE_AND_AMOUNT_SCHEMA")
                except Exception as exc:
                    record.update(status="SOURCE_NOT_ADMITTED_THIS_REQUEST", error_type=type(exc).__name__, error=str(exc),
                                  finished_at=source.study.now())
                source.save_json(OUT / "raw" / (filename + ".receipt.json"), record)
                records.append(record)
                print(f"官方来源：{exchange}/{day}/{record['status']}。", flush=True)
    finally:
        session.close()
    frame = pd.DataFrame(records)
    frame.to_parquet(OUT / "results/六固定官方请求及全部缺失.parquet", index=False)
    frame.to_csv(OUT / "results/六固定官方请求及全部缺失.csv", index=False, encoding="utf-8-sig")
    passed = frame.status.eq("PASS_OFFICIAL_DATE_AND_AMOUNT_SCHEMA")
    matched = [day for day in DATES if frame.loc[frame.stat_date.eq(day), "status"].eq("PASS_OFFICIAL_DATE_AND_AMOUNT_SCHEMA").all()]
    summary = {"at": source.study.now(), "study": protocol["study"], "registration_decision": "TECH.R201", "decision": "TECH.R202",
        "status": "COMPLETED_BOUNDED_OFFICIAL_MARGIN_SOURCE_FEASIBILITY", "fixed_data_gets": len(records),
        "successful_date_and_schema_responses": int(passed.sum()), "matched_two_market_dates": matched,
        "all_six_passed": bool(passed.all()), "request_retries": 0, "alternate_domains_or_credentials": False,
        "full_2026_calendar_collected": False, "historical_first_publication_receipts": "NOT_ESTABLISHED",
        "source_value_vintage": "RETRIEVED_NOW_DEVELOPMENT_ONLY", "new_fits": 0, "new_accounts": 0, "new_labels": 0,
        "new_net_sharpe": "NOT_COMPUTED", "new_net_cagr": "NOT_COMPUTED", "goal_achieved": False,
        "overfitting_removed": False, "independent_validation": "NOT_ESTABLISHED", "latest_actual_financial_decision": "TECH.R198",
        "financial_candidate_admitted": False,
        "next_scope": "仅在双市场全部固定日期字段成功且单位/发布时间合同核清后另申请2026完整范围，不自动收集；否则保持本实际失败，明确可用源缺口。"}
    source.save_json(OUT / "summary.json", summary)
    report = "# 2026两市汇总融资官方源：六固定请求的实际结果\n\n"
    report += "TECH.R201先登记，TECH.R202执行六次官方GET，0重试/换域/模型/账户/标签；日期来自覆盖端点，未按股票收益选择。\n\n"
    display = ["exchange", "stat_date", "status", "http_status", "balance_raw", "buy_raw", "source_unit", "error_type"]
    report += frame[[x for x in display if x in frame]].to_markdown(index=False) + "\n\n"
    report += f"有效字段响应{int(passed.sum())}/6；两市同日共同支持{matched}。HTTP成功、JSON字段成功与历史公布时点/首版成功分别判断。原始响应和错误全部保存，网页服务器Date与本次接收钟不是历史公布钟。\n\n"
    report += "样本来源成功不代表全年已补齐，不能合并进模型或宣称收益提高；当前金融仍R198固定拒绝。若请求失败，只说明本次域和日期请求的结果，不证明全部机构都无数据，也不改原失败或退回自身ETF融资。\n\n"
    report += summary["next_scope"] + "\n\n[全部请求、原件路径及错误](results/六固定官方请求及全部缺失.csv)、[登记](protocol.json)、[实际结果](summary.json)。\n"
    report_path = OUT / "六官方来源请求与准入结论.md"
    report_path.write_bytes(report.encode("utf-8"))
    before = STATE.read_bytes()
    state = json.loads(before.decode("utf-8-sig"))
    (OUT / "state_before_TECH_R202.json").write_bytes(before)
    forward = {k: copy.deepcopy(state[k]) for k in FORWARD}
    state.update(updated_at=source.study.now(), latest_technical_decision="TECH.R202", latest_information_intake_technical_decision="TECH.R202",
        latest_prior_review_technical_decision="TECH.R202", latest_registration_decision="TECH.R201",
        latest_completed_study=source.rel(OUT), latest_result=source.rel(OUT / "summary.json"), latest_report=source.rel(report_path),
        latest_research_status=summary["status"], latest_official_2026_margin_source_feasibility=source.rel(OUT / "summary.json"),
        current_study=summary["study"], current_phase="BOUNDED_OFFICIAL_MARGIN_SOURCE_FEASIBILITY_COMPLETED",
        current_phase_information_scope="TWO_EXCHANGE_OFFICIAL_AGGREGATE_MARGIN_ONLY_SIX_FIXED_REQUESTS",
        current_phase_definition_browsing="EXISTING_LOCAL_OFFICIAL_COLLECTION_CODE_AND_UP_TO_TWO_OFFICIAL_PAGES",
        current_phase_source_freeze_count=len(protocol["sources"]), next_information_source_proposal=summary["next_scope"],
        next_research_question=summary["next_scope"], current_priority=summary["next_scope"],
        latest_continuation_receipt=source.rel(OUT / "project_state_update_receipt.json"),
        latest_progress_check=source.rel(OUT / "summary.json"), current_goal_turn_classification="progress",
        current_goal_turn_classification_reason="R200实际155资金槽/1原点与本R202真实官方响应改变下一源动作。",
        consecutive_blocked_goal_turns=0, blocked_audit_count=0, new_market_requests_this_continuation=6,
        current_admitted_unrun_numeric_candidates=0, next_financial_experiment="NOT_REGISTERED_OFFICIAL_SOURCE_FEASIBILITY_ONLY",
        new_financial_result_computed_this_continuation=False, return_and_sharpe_improved_this_continuation=False,
        latest_progress=f"TECH.R200源覆盖实际完成，TECH.R202六官方GET完成：字段成功{int(passed.sum())}/6，共同日期{matched}；原件/缺失全保留，0金融，最近实际仍R198。",
        goal_achieved=False, whole_model_overfitting_removed=False)
    source.study.require(all(state[k] == v for k, v in forward.items()), "原十三项前瞻改变。")
    STATE.write_bytes((json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
    block = ("> 最新官方来源事实 TECH.R202（2026-10-05，金融仍TECH.R198）：固定6官方GET实际完成，"+
             f"字段成功{int(passed.sum())}/6，两市同日{matched}，无重试/换域；原件及失败全保存。"+
             "2026全年尚未补齐，历史公布钟/首版未知，0模型/账户/标签，新收益夏普NOT_COMPUTED；金融用途未准入。"+
             "本轮R200资料及例外和R202实际响应属于进展，连续受阻0，目标active未达，原13前瞻值保持。\n\n"+
             "下一步："+summary["next_scope"]+"\n\n依据：[全部六请求与实际裁决](../"+source.rel(report_path)+")。\n\n")
    decision = ("\n## TECH.R201—TECH.R202：2026两市汇总融资有限官方可获得性\n\n"+
        "- 假设：原汇总融资2026缺口可由原沪深官方接口补齐，须先证明日期/工具/字段成功。\n"+
        "- 方法：固定首观察日1月5日、资金截止8月14日、ETF截止9月30日，各市场一次GET，共6；先登记、不读收益选样、不重试/换域，保存全部原件及状态。\n"+
        f"- 结果：有效日期及金额字段{int(passed.sum())}/6，共同日期{matched}，全部缺失和HTTP/字段失败详见原件台账；全年未收集。\n"+
        "- 接受/拒绝原因：仅判当前官方来源可获得性，样本成功不能替代全年、公布钟或首次版本；金融候选未准入、最近R198拒绝保持，0新模型/账户/标签。来源失败不证明全市场无数据。\n"+
        "- 重验条件：本次6请求结束，不按失败追加尝试。"+summary["next_scope"]+"原冻结策略、失败和独立验证边界保持。\n")
    backups = OUT / "documents_before_TECH_R202"
    backups.mkdir()
    documents = []
    for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        p = ROOT / "docs" / name
        old = p.read_bytes()
        (backups / name).write_bytes(old)
        cut = old.index(b"\n") + 1
        newline = "\r\n" if old[:cut].endswith(b"\r\n") else "\n"
        insertion = ("\n"+block).replace("\n", newline).encode("utf-8")
        tail = decision.replace("\n", newline).encode("utf-8") if name.startswith("RESEARCH_DECISIONS") else b""
        p.write_bytes(old[:cut]+insertion+old[cut:]+tail)
        documents.append({"path": source.rel(p), "original_body_preserved": True, "sha256": source.h(p)})
    source.save_json(OUT / "project_state_update_receipt.json", {"at": source.study.now(), "status": "PASS_OFFICIAL_SOURCE_AND_DURABLE_FACTS",
        "documents": documents, "state_sha256": source.h(STATE), "latest_source_decision": "TECH.R202",
        "latest_actual_financial_decision": "TECH.R198", "original_forward_thirteen_unchanged": True,
        "progress_is_actual_source_responses_not_document_updates": True, "consecutive_blocked_goal_turns": 0,
        "goal_status": "active", "goal_achieved": False})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026汇总融资官方来源固定六请求")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
