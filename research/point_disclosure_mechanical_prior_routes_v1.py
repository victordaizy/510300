"""八项公告与机械供需的有限旧用途核对；只摘录保存事实，不重跑金融实验。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_disclosure_mechanical_prior_routes_v1"
PLAN = "reports/research/510300_point_next_information_intake_20261002/M01_M04_M06_N03_N06_O05_prior_source_plan.json"
BASE = "reports/research"
JOIN = f"{BASE}/510300_factor96_repurchase_execution_join_v1"
PURPOSE = f"{BASE}/510300_factor96_repurchase_purpose_v1"
CHANGE = f"{BASE}/510300_factor96_repurchase_change_chain_v1"
CONTEXT = f"{BASE}/510300_factor96_repurchase_9_plan_context_v1"
SUPPLY = f"{BASE}/510300_factor96_supply_version_ledger_v1"
RIGHTS = f"{BASE}/510300_factor96_rights_event_chains_v1"
HOLDER = f"{BASE}/510300_shareholder_disclosure_breadth_v1"
DRIVER = f"{BASE}/510300_shareholder_driver_diagnostic_v1"
CHANNEL = f"{BASE}/510300_historical_index_participation_channels_v1"
ALLOCATION = f"{BASE}/510300_historical_index_allocation_scope_v1"
OFFICIAL = "A_SHARE_HS_CSI300_OFFICIAL_ADDITION_FORCED_DEMAND_V1"
SCHEMAS = tuple(f"{HOLDER}/{p}" for p in (
    "inputs/provider_records.parquet", "inputs/events.parquet", "daily_features.parquet"))
IDS = ("M01", "M02", "M03", "M04", "M06", "N03", "N06", "O05")


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path: str) -> dict | list:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path: str) -> str:
    h = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(name: str, value: object) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def pick(path: str, keys: tuple[str, ...]) -> dict:
    value = read(path)
    if not isinstance(value, dict):
        raise TypeError(f"保存结果应为对象：{path}")
    return {"source": path, **{key: value[key] for key in keys}}


def sources() -> list[str]:
    paths = [PLAN, *[r["path"] for r in read(PLAN)["sources"]],
             Path(__file__).relative_to(ROOT).as_posix(),
             *[f"{p}/result.json" for p in (JOIN, PURPOSE, CHANGE, CONTEXT, SUPPLY,
                 RIGHTS, HOLDER, DRIVER, CHANNEL, ALLOCATION)],
             f"{JOIN}/protocol.json", f"{SUPPLY}/protocol.json", f"{RIGHTS}/protocol.json",
             f"{HOLDER}/summary.json", f"{HOLDER}/concentration.json",
             "research/finalize_shareholder_breadth_saved_v1.py",
             "research/shareholder_driver_diagnostic_v1.py",
             "research/historical_index_participation_channels_v1.py",
             "research/historical_index_allocation_scope_v1.py",
             f"{CHANNEL}/source_manifest.json", f"{CHANNEL}/web_evidence.json",
             f"{BASE}/510300_funding_repurchase_demand_daily_v1/result.json",
             f"{BASE}/510300_repurchase_fixed_maturity_account_v1/result.json",
             f"{BASE}/510300_unlock_announcement_increment_v1/result.json",
             f"config/a_share_hs_csi300_official_addition_forced_demand_v1.yaml",
             f"reports/data_quality/{OFFICIAL}_SOURCE_ADMISSION.json",
             f"reports/data_quality/{OFFICIAL}_D4_D5_ADMISSION.json",
             f"reports/audit/{OFFICIAL}_PRE_RETURN_FREEZE.json", *SCHEMAS]
    return list(dict.fromkeys(paths))


def cases() -> list[dict]:
    cards = {r["id"]: r for r in read(PLAN)["registered_cards"]}
    rows = [
        ("M01", "原公告实际回购新增金额/事件前自由流通市值是否改善剩余持有价值。",
         "948执行行合成946公开批次；严格M01非空行0，自由流通分母未建立。较晚9方案背景已复核21未知正批次，其中19批准用途、2用途变更待批准；同文档未知21及原现金时钟不改。旧回购需求、固定到期账户均已拒绝。",
         "STRICT_REPURCHASE_INTENSITY_SOURCE_NOT_ADMITTED_OLD_PROXY_ACCOUNTS_REJECTED"),
        ("M02", "注销与激励库存、已实施进度和剩余期限是否区分未来影响。",
         "逐公告明确用途与54变更对象的部分审批链已建；后续9方案背景不能覆盖完整更改/终止史。多用途没有分配量，原文未知不跨文件填成同文档明确；严格T12仍NOT_RUN。",
         "PURPOSE_PROGRESS_PARTIAL_CHAIN_NOT_ADMITTED"),
        ("M03", "已披露内部人实际净交易金额/自由流通市值是否构成新信息。",
         "现有源是普通股东增减持交易公告，非季度前十大快照，也非严格内部人金额。供应商NOTICE_DATE日末代理不是first_seen；原32次非重叠毛均值1.900355%、胜率59.375%，扣固定摩擦代理1.620355%非真实账户。最大一次贡献原算术和36.542885%；后续三个已看收益的机制案例未量化预期差，账户仍NOT_RUN，不写为账户失败。",
         "SHAREHOLDER_DISCLOSURE_GROSS_CLUE_RETAINED_STRICT_INSIDER_NOT_ADMITTED"),
        ("M04", "事前公告未来20日解禁量/此前成交额变化是否有持有价值增量。",
         "1177字段候选不等于事件全集；32公告角色、10显式引用及三版延期链只是固定部分，去重与完整历史版本未建立。日期加两自然日是保守假设、不是历史首次HTTP证明。另一个既有解禁增量模型1599配对预测误差增0.364559%、区间跨0，固定账户已拒绝；完整原卡不同用途仍未准入。",
         "PARTIAL_UNLOCK_VERSION_SOURCE_NOT_ADMITTED_OLD_INCREMENT_REJECTED"),
        ("M06", "当时已公告A股发行、缴款、上市及新增股份是否形成新供给压力。",
         "固定范围126原文形成36个A股配股事件，35有范围内首公告，36有结果及上市；不代表全M06。历史首次发布、所有中间版本、自由流通分母均未建立；H超额配股、A转H和子公司发行分开，最终量不倒填计划时点，T13仍NOT_RUN。",
         "FIXED_A_RIGHTS_SCOPE_COMPLETE_FULL_SUPPLY_CARD_NOT_ADMITTED"),
        ("N03", "可核实旧新权重乘跟踪规模的理论调仓额/成交额是否解释ETF价值。",
         "旧个股官方调入研究21期425事件的名单门通过，但D4涨跌停/公司行为覆盖失败、D5五退出未解决，明确停止收益检验。它不是510300理论调仓需求特征；原卡旧新权重与同时点跟踪规模未绑定。另有规模/估值分母描述研究，未识别调样因果贡献或策略。",
         "OTHER_SECURITY_ADDITION_DATA_GATE_STOPPED_ETF_DEMAND_CARD_NOT_BOUND"),
        ("N06", "融资标的调整的事前受影响CSI成员权重与融资覆盖是否提供增量。",
         "已有2022-10-21两市扩围公告的历史解释，10-24生效；上交所保存原页、深交所原机抓取失败但留网页正文证据，历史首发未证明。现有观察没有完整纳入/剔除版本、当期受影响成员权重和原1507状态/115训练月共同合同；制度供给变化不等于实际新增买入。",
         "ONE_HISTORICAL_SCOPE_EVENT_NOT_FULL_MEMBER_SOURCE_ADMISSION"),
        ("O05", "当时已公布的未来5日公告风险覆盖是否改变持有价值。",
         "本次计划定位器rapid_orders_inventory实际是PMI新订单/库存月度资料，不能当财报预约日历。有限research/config关键词核对未绑定财报预约历史版本；已有数据发布或议息日期也不自动构成完整组合卡。当前预约表、改期首版及成员权重合同未建立，不据搜索无命中声称全repo或全部市场不存在。",
         "MISROUTED_PMI_LOCATOR_CORRECTED_ANNOUNCEMENT_CALENDAR_CARD_NOT_BOUND"),
    ]
    return [{"id": code, "name": cards[code]["name"], "hypothesis": hypothesis,
             "verification_method": "有限原卡、实际代码、保存来源合同及最终裁决合并；不读取新收益或重跑模型。",
             "result": result, "decision": decision, "field_admitted": False,
             "why": "完整原信息用途、事前版本和当前全部成熟成员资格未同时成立。旧终态按原口径保持，不否定整个信息家族。",
             "revalidation": "需实质不同用途及合格原来源、历史版本和可得时钟，先核原全部成员，再另冻比较；不得更换旧阈值、窗口、方向、费用或选期营救。"}
            for code, hypothesis, result, decision in rows]


def account_quotes() -> list[dict]:
    result = []
    for name in ("510300_funding_repurchase_demand_daily_v1",
                 "510300_repurchase_fixed_maturity_account_v1"):
        path = f"{BASE}/{name}/result.json"
        result.append({"source": path, "role": "原主压力账户，非本轮新账户",
                       "annualization": "原来源口径，不重新年化", "saved_row": read(path)["primary"]})
    path = f"{BASE}/510300_unlock_announcement_increment_v1/result.json"
    selected = [r for r in read(path)["accounts"] if r["model"] == "M1" and r["cost"] == "STRESS"]
    if len(selected) != 1:
        raise ValueError("原解禁M1压力账户应恰有一行")
    result.append({"source": path, "role": "原解禁M1压力账户，非完整M04卡",
                   "annualization": "原来源口径，不重新年化", "saved_row": selected[0]})
    return result


def facts() -> dict:
    return {
        "repurchase_batches": pick(f"{JOIN}/result.json", ("status", "execution_rows", "publication_batches",
            "positive_increment_batches", "batch_purpose_known", "batch_purpose_unknown", "strict_M01_non_null_rows")),
        "later_nine_plan_context": pick(f"{CONTEXT}/result.json", ("status", "target_batches", "scheme_roots",
            "target_context_states", "same_document_unknowns_remaining", "same_document_purpose_consensus_changed",
            "full_change_coverage_established", "free_float_denominator_established", "strict_M01_non_null_rows")),
        "purpose": pick(f"{PURPOSE}/result.json", ("status", "execution_records", "execution_purpose_statuses",
            "full_purpose_termination_chain_established", "free_float_denominator_established", "T12")),
        "changes": pick(f"{CHANGE}/result.json", ("status", "change_documents", "documents_with_confirmed_target_root",
            "distinct_confirmed_target_roots", "full_purpose_termination_chain_established", "T12")),
        "supply_versions": pick(f"{SUPPLY}/result.json", ("status", "role_documents", "explicit_links",
            "schedule_versions", "all_events_deduplicated", "full_M06_calendar_established", "T13")),
        "rights": pick(f"{RIGHTS}/result.json", ("status", "reviewed_documents", "identified_A_rights_events",
            "events_with_initial_notice_in_scope", "events_with_result_and_listing", "historical_first_publication_verified",
            "full_intermediate_version_coverage", "free_float_denominator_established", "full_M06_calendar_established")),
        "shareholder_screen": pick(f"{HOLDER}/result.json", ("status", "primary", "concentration", "account_status",
            "net_sharpe", "net_cagr", "strict_M03_run", "historical_first_vintage_verified")),
        "shareholder_driver": pick(f"{DRIVER}/result.json", ("status", "scope", "practical_disposition",
            "post_policy_price_path_is_causal_attribution", "policy_detail_known_at_original_entry")),
        "other_security_source_admission": read(f"reports/data_quality/{OFFICIAL}_SOURCE_ADMISSION.json"),
        "other_security_final_data_admission": read(f"reports/data_quality/{OFFICIAL}_D4_D5_ADMISSION.json"),
        "other_security_pre_return_status": read(f"reports/audit/{OFFICIAL}_PRE_RETURN_FREEZE.json"),
        "margin_scope_sources": [r for r in read(f"{CHANNEL}/source_manifest.json")
                                 if r["id"] in ("sse_expansion", "szse_expansion")],
        "margin_scope_web_evidence": [r for r in read(f"{CHANNEL}/web_evidence.json")
                                      if r["id"] in ("sse_expansion", "szse_expansion")],
        "historical_participation": pick(f"{CHANNEL}/result.json", ("status", "classification", "discovery", "new_full_accounts")),
        "index_scope_limits": pick(f"{ALLOCATION}/result.json", ("status", "classification", "limitations", "new_full_accounts")),
        "schemas": [{"path": p, "rows": pq.ParquetFile(ROOT / p).metadata.num_rows,
                     "columns": pq.ParquetFile(ROOT / p).schema_arrow.names} for p in SCHEMAS],
        "schema_scope": "只读取三份保存表模式与行数，没有将CLOSE_PRICE/FREE_SHARES等字段认证为历史交易金额或当时自由流通分母。",
        "calendar_locator_correction": "rapid_orders_inventory为月度PMI，非原O05财报预约。有限代码搜索没有绑定合同，非全项目不存在证明。",
        "current_membership_support_test": "NOT_RUN_NO_COMPLETE_FIELD_BOUND",
        "current_1507_states_or_115_model_months_verified": False,
        "closed_prior_M05_rescanned": False,
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError("目录已存在，禁止覆盖或重复核对")
    for row in read(PLAN)["sources"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"提案源变化：{row['path']}")
    manifest = [{"path": p, "sha256": sha(p), "bytes": (ROOT / p).stat().st_size} for p in sources()]
    outputs = {"cases.json": cases(), "saved_old_primary_account_quotes.json": account_quotes(),
               "source_and_old_purpose_facts.json": facts()}
    if tuple(r["id"] for r in outputs["cases.json"]) != IDS:
        raise ValueError("八项有限范围改变")
    OUT.mkdir(parents=True)
    save("evidence_manifest.json", manifest)
    save("protocol.json", {"at": now(), "study_id": "510300_POINT_DISCLOSURE_MECHANICAL_PRIOR_ROUTES_V1",
         "decision": "TECH.R114", "plan": PLAN, "scope": "有限八项旧用途及源合同，三表模式，三原主压力账户摘录。",
         "new_configs_fits_labels_predictions_accounts_collection": 0, "old_frozen_files_modified": False,
         "not_a_new_financial_test": True, "current_full_member_support_not_computed": True})
    for name, value in outputs.items():
        save(name, value)
    save("summary.json", {"at": now(), "study_id": "510300_POINT_DISCLOSURE_MECHANICAL_PRIOR_ROUTES_V1",
         "technical_decision": "TECH.R114", "status": "NOT_ADMITTED_EIGHT_DISCLOSURE_MECHANICAL_PRIOR_SOURCE_ROUTES",
         "completed_prior_routes": len(IDS), "source_count": len(manifest), "saved_old_account_quotes": 3,
         "saved_schema_tables": 3, "field_bound": False, "field_admitted": False,
         "current_full_member_support": "NOT_RUN_NO_COMPLETE_FIELD_BOUND",
         "new_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
         "new_predictions": 0, "new_accounts": 0, "network_requests": 0,
         "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
         "latest_actual_prediction_decision_unchanged": "TECH.R113_REJECTED_FIXED_ENTRY_ATR_INCREMENT",
         "independent_validation": "NOT_ESTABLISHED", "overall_overfit_removed": False,
         "goal_achieved": False, "goal_turn_classification": "progress", "blocked_audit_count": 0})
    lines = "\n\n".join(f"**{r['id']} {r['name']}**\n\n{r['result']}" for r in outputs["cases.json"])
    report = ("# 八项公告与机械供需来源核对\n\n"
              "八项当前完整卡均未准入。这是旧用途和来源核对，没有新增预测、账户或夏普提升结果。"
              "上一实际模型仍是TECH.R113：入场ATR尺度回撤的单一固定增量双期失败。\n\n"
              + lines + "\n\n原冻结策略、原E03前瞻账户及其独立验证门保持。M05既已结束，本轮未重扫。"
              "其他分支允许有限历史入场/持有/退出研发的权限不被本线来源裁决覆盖。"
              "下一实际实验需先说明不同信息为何改变入场或持有价值，并绑定可得数据；"
              "仅批量换指标、阈值、年度和费用不能算去过拟合。\n")
    with (OUT / "研究结论与下一步.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    print(f"八项来源核对完成：{len(manifest)}个证据对象，未准入；三原账户摘录，新增拟合/账户0。")


def verify() -> None:
    for row in read((OUT / "evidence_manifest.json").relative_to(ROOT).as_posix()):
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"冻结来源变化：{row['path']}")
    for name, expected in {"cases.json": cases(), "saved_old_primary_account_quotes.json": account_quotes(),
                           "source_and_old_purpose_facts.json": facts()}.items():
        if json.loads((OUT / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"摘录复算不一致：{name}")
    save("saved_output_verification_receipt.json", {"at": now(),
         "status": "PASS_SAVED_EIGHT_DISCLOSURE_MECHANICAL_PRIOR_SOURCE_ROUTE_VERIFICATION",
         "case_count": len(IDS), "saved_original_account_rows": 3, "saved_schema_tables": 3,
         "new_fits_labels_predictions_accounts_collection": 0,
         "limit": "只核原来源及摘录，不证明历史首次可得、当前字段支持或策略有效。"})
    print("八分项、三原主账户及三表模式核对通过；冻结策略保持。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="公告与机械供需八项有限来源核对")
    parser.add_argument("operation", choices=("run", "verify"))
    args = parser.parse_args()
    run() if args.operation == "run" else verify()
