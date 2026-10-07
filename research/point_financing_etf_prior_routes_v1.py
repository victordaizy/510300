"""合并剩余融资与ETF份额的七项有限旧用途路由，不重查已关闭原始响应。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_financing_etf_prior_routes_v1"
INTAKE = "reports/research/510300_point_next_information_intake_20261002"
PREVIOUS = "reports/research/510300_point_f01_f03_prior_source_routing_v1"
PROGRAM = "reports/research/510300_factor96_program_v1"
STRUCTURE = "reports/research/510300_factor96_rapid_structure_v1"
CLOSURE = "reports/research/510300_fund_share_publication_receipts_closure_v1/result.json"
IDS = ["F04", "F05", "F06", "G01", "G02", "G03", "G04"]


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def sha(relative: str) -> str:
    with (ROOT / relative).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name: str, value) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sources() -> list[str]:
    path = f"{INTAKE}/F04_F06_G01_G04_prior_source_routing_plan.json"
    plan = read(path)
    for entry in plan["sources"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"原提案来源已改变：{entry['path']}")
    paths = [path] + [entry["path"] for entry in plan["sources"]] + [
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        f"{PROGRAM}/factor_progress.json", f"{PROGRAM}/strategy_progress.json", f"{PROGRAM}/status.json",
        "reports/research/510300_factor96_crowding_overlay_v1/source_evidence/overlap_and_source_decisions.json",
        "reports/research/510300_factor96_free_float_source_probe_v1/single_auth_validation_receipt.json",
        "config/510300_etf_share_premium_level_binary_screen_v1.yaml",
        "config/510300_etf_share_premium_level_binary_screen_v1_candidates.yaml",
        "reports/research/510300_etf_share_premium_level_binary_screen_v1.json",
        "research/factor96_rapid_structure_v1.py", f"{STRUCTURE}/protocol.json",
        f"{STRUCTURE}/result.json", f"{STRUCTURE}/coverage.json", f"{STRUCTURE}/saved_verification_receipt.json",
        f"{PREVIOUS}/cases.json", f"{PREVIOUS}/saved_old_primary_account_quotes.json",
    ]
    return list(dict.fromkeys(paths))


def quote_rows() -> list[dict]:
    previous = read(f"{PREVIOUS}/saved_old_primary_account_quotes.json")
    selected = [row for row in previous if "saved_row" in row and
                (("crowding_overlay" in row["source"] and row["saved_row"]["policy"] == "FULL") or
                 row["saved_row"]["policy"] == "T05")]
    if len(selected) != 2:
        raise ValueError("F05/F06两原主行不完整")
    path = f"{STRUCTURE}/result.json"
    rows = [row for row in read(path)["primary_results"] if row["group"] == "ETF_MIGRATION_PROXY"
            and row["capital"] == 200000 and row["cost"] == "STRESS"
            and row["policy"] in ("ETF_MIGRATION_PROXY_HIGH", "ETF_MIGRATION_PROXY_LOW", "PRICE_CONTROL")]
    if len(rows) != 3:
        raise ValueError("ETF迁移两方向及匹配对照不完整")
    return selected + [{"source": path, "annual_days": 242, "saved_row": row} for row in rows]


def facts() -> dict:
    closure = read(CLOSURE)
    fields = ["status", "source_rows", "source_first_date", "source_last_date", "actual_trading_dates_covered",
              "extra_non_trading_dates", "extra_dates_already_excluded_in_old_merge",
              "historical_raw_response_count", "historical_publication_fields_found",
              "historical_raw_file_timestamp_meaning", "extension_rows",
              "extension_does_not_backfill_earlier_publication_proof", "decision", "scope_limit"]
    binary = read("reports/research/510300_etf_share_premium_level_binary_screen_v1.json")
    structure = read(f"{STRUCTURE}/result.json")
    old_gate = read("reports/research/510300_factor96_crowding_overlay_v1/source_evidence/overlap_and_source_decisions.json")
    status = read(f"{PROGRAM}/status.json")
    return {"saved_share_closure": {key: closure[key] for key in fields},
            "closed_receipt_handling": "直接引用旧2220响应核对结论；本轮未重扫、未重新计为2220新进展，没有新首版证据。",
            "original_T07_gate": old_gate["T07"], "original_T06_scope": old_gate["T06"],
            "saved_free_float_gate": status["free_float_source_blocker"],
            "saved_free_float_receipt": read("reports/research/510300_factor96_free_float_source_probe_v1/single_auth_validation_receipt.json"),
            "credential_claim_limit": "只引用2026-09-28保存的过期/0行凭证，本轮未测试当前凭据或请求接口。",
            "old_binary": {key: binary[key] for key in ("status", "scope", "candidate_count", "result_rows",
                            "passing_candidates", "positive_shadow_candidates", "decision")},
            "old_binary_eight_saved_summaries": binary["candidate_summaries"],
            "old_binary_units": "原50万元/现金年利率1.5%/无最低佣金、年化242及H00300超额双20门；不是当前20万元原252现金0净收益夏普口径，320结果行不是320独立策略。",
            "old_migration": {key: structure[key] for key in ("classification", "publication_clock",
                               "original_G04_T07_validated", "qualified_candidates", "worth_followup")},
            "migration_coverage": read(f"{STRUCTURE}/coverage.json")["ETF_MIGRATION_PROXY"],
            "migration_population": "固定510300/510310/510330三只沪市产品，2021起周度，原始收盘价估值；非动态全体系日频NAV。",
            "current_1507_support": "NOT_COMPUTED", "current_115_support": "NOT_COMPUTED"}


def cases() -> list[dict]:
    plan = read(f"{INTAKE}/F04_F06_G01_G04_prior_source_routing_plan.json")
    originals = {row["id"]: row for row in plan["registered_cards"]}
    states = {row["id"]: row for row in read(f"{PROGRAM}/factor_progress.json") if row["id"] in IDS}
    items = [
        ("F04", "融资需求从少数股票扩散到多数，是否支持修复。", "原卡NOT_RUN；有限定位源是沪深汇总，没有绑定历史有效融资股票集合、明细及覆盖合同；不以汇总代成员、不填无资格0。", "HISTORICAL_ELIGIBLE_STOCK_FINANCING_PANEL_NOT_ESTABLISHED"),
        ("F05", "融资扩张但价格不推进是否形成退出风险。", "旧T06拥挤FULL已使用融资高分位与价格停滞，主账户CAGR−1.188573%、夏普−0.636625，失败保持；余额/匹配自由流通市值未测，旧源验证0行，不能用总股本换自由流通补救。", "OLD_T06_COMPONENT_FAILED_MATCHED_FREE_FLOAT_NOT_ADMITTED"),
        ("F06", "高偿还活动后价格抗跌是否显示承接。", "旧T05用隐含偿还和价格修复，CAGR−1.001076%、夏普−1.058448、18闭合周期；原卡直报偿还与E02配对未完成，R108源差额原因未知保持。", "OLD_T05_IMPLIED_REPAY_FAILED_DIRECT_REPAY_E02_PURPOSE_NOT_ADMITTED"),
        ("G01", "份额净创造是否提供价格以外需求信息。", "旧八条份额/收盘折溢价规则均拒绝；原单基金完整卡NOT_RUN。旧1216缓存/1211交易日及2220官方响应都缺历史首版，局部补源已关闭，原始响应本轮不重扫。", "RELATED_OLD_EIGHT_RULES_REJECTED_CLOSED_PUBLICATION_GATE_RETAINED"),
        ("G02", "下跌中份额增加与压力缓和是否显示需求吸收。", "原条件组合NOT_RUN，G01资格是必要输入；份额/NAV公布与拆分未形成当前合同，不把缺数据当0、价值代理当现金流或国家队。", "G01_SOURCE_CLOCK_AND_SPLIT_DEPENDENCY_NOT_ADMITTED"),
        ("G03", "动态同指数体系需求是否比单产品更有信息。", "旧T07日频体系源门NOT_RUN；固定十只沪市周度中的三只同指数子池不能代动态全体系，历史上市退出、拆分、前日NAV及发布时间仍缺合同。", "T07_DAILY_DYNAMIC_SYSTEM_FLOW_SOURCE_GATE_RETAINED"),
        ("G04", "单产品相对全体系变化是否识别产品迁移。", "旧三产品周度迁移HIGH/LOW均高于匹配价格对照，CAGR1.191156%/0.312628%、夏普0.370320/0.139443、入场14/16；预定继续门要求至少20入场、CAGR5%、夏普0.8及两期正，未过。完整日频NAV原卡仍未验证。", "OLD_WEEKLY_THREE_PRODUCT_PROXY_NOT_ACCEPTED_FULL_DAILY_CARD_NOT_VALIDATED"),
    ]
    return [{"id": factor, "name": originals[factor]["name"], "hypothesis": hypothesis,
             "verification_method": "合并读取七张原卡/程序状态、旧直接源门和固定结果；引用已关闭份额回执结论、八原规则摘要和五原主行，不复扫2220响应或运行模型/账户。",
             "result": result, "routing_status": status, "original_card": originals[factor],
             "saved_program_status": states[factor],
             "acceptance_or_rejection": "接受旧用途/来源/实际正负结果的区分；当前完整观测资格未建立，字段和模型不准入。旧固定失败与原完整卡NOT_RUN分开，不否定全部经济机制。",
             "revalidation": "只有新增独立首版、资格/拆分/动态体系/NAV或匹配分母合同及不同用途才另立固定比较；不重扫已关闭回执、改池/频率/窗口/符号/阈值/费用救旧失败。",
             "current_1507_support": "NOT_COMPUTED", "current_115_support": "NOT_COMPUTED",
             "field_bound": False, "field_admitted": False} for factor, hypothesis, result, status in items]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError("输出已存在，请用verify，不覆盖原结果")
    paths = sources()
    save("evidence_manifest.json", [{"path": path, "sha256": sha(path)} for path in paths])
    save("protocol.json", {"at": now(), "technical_decision": "TECH.R109", "cards": IDS,
         "classification": "FINITE_PRIOR_SOURCE_ROUTING_OLD_OUTCOMES_KNOWN_NOT_NEW_RETURN_EXPERIMENT",
         "purpose": "用户要求尽快，七项合并一次，直接复用已完成源门，不继续局部补充或改旧规则。",
         "new_configurations": 0, "new_fits": 0, "new_labels": 0, "new_accounts": 0,
         "closed_raw_responses_rescanned": 0, "network_requests": 0,
         "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False})
    save("cases.json", cases())
    save("source_and_old_purpose_facts.json", facts())
    save("saved_old_primary_account_quotes.json", quote_rows())
    save("summary.json", {"at": now(), "study_id": "510300_POINT_FINANCING_ETF_PRIOR_ROUTES_V1",
         "technical_decision": "TECH.R109", "status": "NOT_ADMITTED_SEVEN_FINANCING_AND_ETF_PRIOR_ROUTES_COMPLETE",
         "source_count": len(paths), "completed_prior_routes": 7, "saved_old_primary_account_rows": 5,
         "saved_old_binary_rule_summaries": 8, "closed_share_raw_response_rescans": 0,
         "current_member_support": "NOT_COMPUTED", "new_field_bound": False, "new_field_admitted": False,
         "new_configurations": 0, "new_ols_or_model_fits": 0, "new_labels": 0,
         "new_predictions": 0, "new_accounts": 0, "new_collection": 0,
         "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
         "independent_validation": "NOT_ESTABLISHED", "overall_overfit_removed": False,
         "goal_achieved": False, "goal_turn_classification": "progress", "blocked_audit_count": 0})
    text = "# 七项融资与ETF份额有限旧用途路由（TECH.R109）\n\n七项当前不准入；本轮是实际旧用途/来源决策，不是七个新回测或金融改进。\n\n"
    text += "| 原卡 | 保存事实 | 路由 |\n| --- | --- | --- |\n"
    text += "".join(f"| {row['id']} {row['name']} | {row['result']} | `{row['routing_status']}` |\n" for row in cases())
    text += "\n周度迁移两方向均高于对应旧价格对照；原HIGH早段为负、LOW两段为正，保留全部事实，不能只挑HIGH后段夏普1.0688恢复旧候选。两方向入场14/16低于原20门，收益/夏普均低于原继续门；固定三沪市产品周度原始价估值不是原卡动态日频前日NAV体系。\n\n"
    text += "旧份额局部核对已关闭。本轮仅读结论，未重扫2220原响应、声称新增公布证据或进行网络请求。旧五条周度总流量、八条单基金份额/收盘NAV折溢价、两条三产品周度迁移分别保留为不同旧用途。NAV折溢价不是同步IOPV，NAV×份额变化仅价值代理。旧八条的50万元、现金1.5%、最低佣金0、242年化及超额双20门不是当前20万元口径，不比较排名。\n\n"
    text += "F04有限指定来源尚未绑定股票资格/明细合同，不声称全项目或外部没有任何面板。F05只引用旧2026-09-28的过期凭证/0行，不宣称本轮验证了当前接口。F06保留直报与隐含偿还差额和E02未配对，不把其解释成强平。G03/G04的原T07源门保持，不以固定池或今日基金池回填。\n\n"
    text += "七路、五原主行、八旧规则摘要及旧源门保存核对一次；当前1507/115支持NOT_COMPUTED，新字段/配置/拟合/标签/预测/账户/采集0、新CAGR/夏普NOT_COMPUTED。独立验证未建立、过拟合未去除、完整目标未达。下一合并检查96因子卡E01—E06成分参与面的旧表达和当前保存源支持资格；它们不是TECH.E01目标传递方法。原E03不重置、原策略及其他线保持。\n\n"
    text += "直接证据：[七项假设/路由](cases.json)、[源合同与旧八规则](source_and_old_purpose_facts.json)、[五原账户](saved_old_primary_account_quotes.json)、[保存核对](saved_output_verification_receipt.json)。\n"
    with (OUT / "研究结论与下一步.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("七项融资/份额旧用途合并路由已保存；已关闭2220响应未重查，新拟合/账户0。")


def verify() -> None:
    manifest = json.loads((OUT / "evidence_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"冻结来源变化：{entry['path']}")
    expected = {"cases.json": cases(), "source_and_old_purpose_facts.json": facts(),
                "saved_old_primary_account_quotes.json": quote_rows()}
    for name, value in expected.items():
        if json.loads((OUT / name).read_text(encoding="utf-8")) != value:
            raise ValueError(f"保存摘录不一致：{name}")
    save("saved_output_verification_receipt.json", {"at": now(),
         "status": "PASS_SAVED_SEVEN_ROUTE_AND_OLD_SOURCE_FACT_VERIFICATION",
         "prior_routes": 7, "old_account_quotes": 5, "old_binary_rule_summaries": 8,
         "closed_raw_response_rescans": 0, "new_model_or_account_calls": 0,
         "limit": "保存旧裁决/源合同的对应一致，不证明当前全部来源或金融改进。"})
    print("七路/五旧账户/八旧规则保存核对通过，没有重新核2220响应或计算收益。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="七项融资与ETF份额旧用途合并路由")
    parser.add_argument("operation", choices=("run", "verify"))
    args = parser.parse_args()
    run() if args.operation == "run" else verify()
