"""合并核对十一项期货日线及盈利信息的旧用途、真实来源合同与研究终态。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_if_earnings_prior_routes_v1"
INTAKE = "reports/research/510300_point_next_information_intake_20261002"
PROGRAM = "reports/research/510300_factor96_program_v1"
IF = "reports/research/510300_if_open_interest_increment_v1"
STRUCTURE = "reports/research/510300_factor96_rapid_structure_v1"
EPS = "reports/research/510300_eps_growth_disagreement_increment_v1"
NOVELTY = "reports/research/510300_eps_explicit_revision_novelty_diagnostic_v1"
MEASURE = "reports/research/510300_factor96_earnings_cashflow_measurement_v1"
COHORT = "reports/research/510300_factor96_t11_financial_cohorts_v1_0_1"
T11 = "reports/research/510300_factor96_t11_date_proxy_account_v1"
BREADTH = "reports/research/510300_original_earnings_breadth_v1"
DIVIDEND = "reports/research/510300_constituent_dividend_calendar_v1"
BINARY = "reports/research/510300_if_true_term_structure_binary_screen_v1_0_1.json"
VAL = "reports/research/normalized_valuation_5y_family_predictive_screen_v1.json"
PLAN = f"{INTAKE}/H01_H05_L01_L06_prior_source_routing_plan.json"
IDS = ("H01", "H02", "H03", "H04", "H05", "L01", "L02", "L03", "L04", "L05", "L06")
SCHEMAS = (
    "data/raw/futures/cffex_if_contract_history_v1_0_1/cffex_if_contract_daily.parquet",
    "data/raw/futures/cffex_if_contract_history_v1_0_1/cffex_if_contract_expiry.parquet",
    f"{MEASURE}/daily_coverage.parquet",
    f"{COHORT}/daily_financial_cohorts.parquet",
    f"{COHORT}/inputs/repaired_member_report_measurements.parquet",
    f"{BREADTH}/daily_earnings_breadth.parquet",
    f"{DIVIDEND}/inputs/payment_schedules.parquet",
)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path: str) -> dict | list:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def save(name: str, value: object) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def sources() -> list[str]:
    plan = read(PLAN)
    paths = [PLAN, *[r["path"] for r in plan["sources"]],
             Path(__file__).relative_to(ROOT).as_posix(),
             f"{PROGRAM}/strategy_progress.json",
             "config/510300_if_open_interest_increment_v1.json",
             f"{IF}/result.json", f"{IF}/data_preflight.json",
             f"{STRUCTURE}/protocol.json", f"{STRUCTURE}/result.json", f"{STRUCTURE}/coverage.json",
             "research/if_true_term_structure_binary_screen_v1_0_1.py", BINARY,
             "config/510300_if_true_term_structure_binary_screen_v1_0_1.yaml",
             f"{EPS}/result.json", f"{EPS}/account_stage_result.json", f"{NOVELTY}/result.json",
             f"{MEASURE}/protocol.json", f"{MEASURE}/result.json", f"{MEASURE}/source_invalidation_addendum.json",
             f"{COHORT}/protocol.json", f"{COHORT}/result.json",
             f"{T11}/protocol.json", f"{T11}/result.json", f"{T11}/source_receipt.json",
             f"{T11}/measurement_result.json", f"{T11}/saved_verification_receipt.json",
             f"{BREADTH}/result.json", f"{BREADTH}/source_receipt.json",
             "research/original_earnings_breadth_v1.py", "research/normalized_valuation_5y_models_v1.py",
             "config/normalized_valuation_5y_models_v1.yaml", "config/normalized_valuation_5y_predictive_screen_v1.yaml",
             VAL, "reports/research/val01_norm_ey_5y_v1_predictive_diagnostics.json",
             "reports/research/val02_norm_ey_spread_5y_v1_predictive_diagnostics.json",
             "research/constituent_dividend_calendar_v1.py", "research/constituent_dividend_calendar_account_v1.py",
             f"{DIVIDEND}/result.json", f"{DIVIDEND}/source_receipt.json", f"{DIVIDEND}/completed_round.json",
             f"{DIVIDEND}/account_stage/result.json", *SCHEMAS]
    return list(dict.fromkeys(paths))


def cases() -> list[dict]:
    original = {r["id"]: r for r in read(PLAN)["registered_cards"]}
    progress = {r["id"]: r for r in read(f"{PROGRAM}/factor_progress.json")}
    items = [
        ("H01", "已知分红和融资成本扣除后的异常基差是否包含增量信息。",
         "旧八个名义年化基差/期限曲线规则已拒绝；实际旧公式log(F/S)*365/DTE，没有扣除成分待分红及融资成本。原卡分红指数点数、历史权重版本和时钟源门仍未过，不能把510300基金分红当000300现货指数待分红。",
         "NOMINAL_BASIS_RULES_REJECTED_CARRY_ADJUSTED_SOURCE_GATE_RETAINED"),
        ("H02", "不同期限扣成本基差的压力迁移是否有增量信息。",
         "旧近次月/近远月名义曲线已包含在八规则失败中，完整扣成本差尚未运行。各期限分红暴露不同，依赖H01及事前实际到期/选择规则；不换月、不改分位营救。",
         "OLD_NOMINAL_CURVE_REJECTED_H01_DEPENDENCIES_NOT_ADMITTED"),
        ("H03", "总持仓变化与价格、基差组合是否区分调整性质。",
         "旧同模型M0/M1已加入总OI单日对数变化及其与ETF五日方向的交互；1599对成熟预测，MSE改善−0.065659%，95%区间跨零、三个时期仅一段正，账户NOT_RUN。原五日OI四象限及完整扣成本基差并非该精确定义，但不能重命名单日失败或把OI当净多资金。",
         "OLD_ONE_DAY_OI_PAIRED_INCREMENT_REJECTED_FIVE_DAY_CARD_NOT_VALIDATED"),
        ("H04", "换月完成后扣成本基差恢复是否改变后续价值。",
         "完整原卡尚未验证，近次月OI迁移、总OI变化与H01/H02恢复需同一时点合同；当前H01/H02必要资料未建立。旧换月控制、期限曲线和持仓集中度不等于原组合，更不能用未来主力或最后出现日选择合约。",
         "ROLL_RECOVERY_H01_H02_SOURCE_DEPENDENCIES_NOT_ADMITTED"),
        ("H05", "按剩余日数比较的合约持仓分布是否有风险增量。",
         "旧名义第三周五剩余日数5日桶、前两日历年同桶的HHI高低方向已检验：20万元STRESS高CAGR−0.776238%/夏普−0.532676，低CAGR0.326475%/夏普0.123218，低方向后期为负；两方向均未过固定继续门。原实际到期完整卡没有据此认证。",
         "FIXED_HHI_HIGH_LOW_SCREEN_FAILED_ORIGINAL_CARD_NOT_VALIDATED"),
        ("L01", "同机构分析师、预测年及股本口径真实修正是否形成新信息。",
         "两份显式上调/下调叙述复用既有数值，新增原数值0；美的既有同对已保存，兴业因利润名称别名不准入保留空值。诊断仅七份已选报告、两事件，时点差异未测，未证明同分析师/股本全历史匹配。另一个两机构已报告年度EPS增长定义分歧增量已失败，不能写成真实预测修正本身已全面否定。",
         "TWO_NARRATIVES_NOT_NEW_NUMBERS_FULL_MATCHED_REVISION_CONTRACT_NOT_ESTABLISHED"),
        ("L02", "原公告单季盈利相对前两年同季的意外是否有增量价值。",
         "单季拆分/前季资产/同季历史意外已测；最初测量有七个确认错字段，后有独立修复及财务篮子版本。应读取最终T11日期代理：120固定账户、主20万元STRESS年化−0.270499%/夏普−0.351761，7次入场。严格first_seen及更正链原T11仍NOT_RUN，不能只停留在早期未回测状态。",
         "REPAIRED_DATE_PROXY_T11_ACCOUNT_FAILED_STRICT_CLOCK_CARD_NOT_ADMITTED"),
        ("L03", "同一已披露成员集中的盈利扩散减价格扩散是否形成残余信息。",
         "旧每日原财报YTD盈利广度模型主STRESS年化0.296111%/夏普0.087205、四笔交易，目标失败；不是同已披露集合盈利减价格扩散的完整新定义。当前没有绑定同成员价格、已披露权重覆盖及首次版本的共同合同，不替换为晚披露未来值、今成分或简单等权。",
         "OLD_YTD_EARNINGS_BREADTH_TARGET_FAILED_MATCHED_SPREAD_CARD_NOT_BOUND"),
        ("L04", "非金融真正TTM现金质量及同比变化是否有增量价值。",
         "真正TTM测量及按前两年同业参考标准化、新披露篮子已进入修复后的T11日期代理失败账户；不是全部现金质量信息无效。金融排除、少数股东/合并范围口径、供应商行业生效时钟和first_seen局限仍在，不能以YTD乘4/季数代TTM或以组合失败宣称单指标独立检验。",
         "REPAIRED_TTM_COMPOSITE_ACCOUNT_FAILED_STRICT_SOURCE_CLOCK_NOT_ADMITTED"),
        ("L05", "多年度正常化盈利收益率与国债利差是否提供慢状态增量。",
         "VAL01正常化EY与VAL02利差的预注册242日预测检验都拒绝；前者桶样本、双半段及单调性门未过，后者桶样本、单调性及便宜端差门未过，账户未运行。显著相关或某项p值不是整体通过；原60日卡并非242日定义，不改变旧标签期限救援。",
         "VAL01_VAL02_REGISTERED_PREDICTIVE_GATES_FAILED_SIXTY_DAY_CARD_NOT_VALIDATED"),
        ("L06", "已公告现金分红估值配合现金流覆盖是否形成增量价值。",
         "已有分红日历广度代理毛筛通过，但之后16固定账户主20万元STRESS日历年化−0.556401%/夏普−0.291530、66次入场，已拒绝；不能只引用毛筛PASS。645缓存6153日历没有完整原公告版本和股本金额，原L06分红/市值及现金质量并非日历代理，且基金分红与成分分红含义不同。",
         "DIVIDEND_CALENDAR_GROSS_PASS_ACCOUNT_REJECTED_L06_CASH_COVERAGE_NOT_BOUND"),
    ]
    return [{"id": code, "name": original[code]["name"], "hypothesis": hypothesis,
             "verification_method": "原卡、实际代码公式、来源模式、最终固定裁决合并比对；只摘录保存值，不重新计算预测或账户。",
             "registered_prior_status": progress[code].get("current_status"), "result": result,
             "decision": decision, "accepted": False,
             "why": "接受精确旧用途及其终态；本线不同用途、当前完整成员和入场前来源资格未同时成立，当前不准入。不是整个信息家族永久无效。",
             "revalidation": "只有实质不同信息用途与新合格原源、历史版本/发布时间、原全部成熟成员资格成立，另立固定比较；不改旧方向、窗口、阈值、费用或挑子期救援。"}
            for code, hypothesis, result, decision in items]


def account_quotes() -> list[dict]:
    selected = [r for r in read(f"{STRUCTURE}/result.json")["primary_results"]
                if r["group"] == "IF_HHI" and r["capital"] == 200000 and r["cost"] == "STRESS"
                and r["policy"] in ("IF_HHI_HIGH", "IF_HHI_LOW", "PRICE_CONTROL")]
    if len(selected) != 3:
        raise ValueError("IF集中度两方向及价格对照原行缺失")
    rows = [{"source": f"{STRUCTURE}/result.json", "annual_days": 242, "saved_row": r} for r in selected]
    rows.append({"source": f"{T11}/result.json", "annual_days": 242,
                 "saved_row": read(f"{T11}/result.json")["primary_account"]})
    stress = [r for r in read(f"{BREADTH}/result.json")["primary"] if r["cost"] == "STRESS"]
    if len(stress) != 1:
        raise ValueError("原盈利广度主压力行缺失")
    rows.append({"source": f"{BREADTH}/result.json", "annual_days": 242, "saved_row": stress[0]})
    rows.append({"source": f"{DIVIDEND}/account_stage/result.json", "annual_days": 242,
                 "cagr_clock": "calendar_cagr是该原主裁决，legacy_cagr_242单独保留，不拼成本线252结果。",
                 "saved_row": read(f"{DIVIDEND}/account_stage/result.json")["primary"]})
    return rows


def source_facts() -> dict:
    schemas = []
    for path in SCHEMAS:
        table = pq.ParquetFile(ROOT / path)
        schemas.append({"path": path, "rows": table.metadata.num_rows, "columns": table.schema_arrow.names})
    oi = read(f"{IF}/result.json")
    eps = read(f"{EPS}/result.json")
    novelty = read(f"{NOVELTY}/result.json")
    t11 = read(f"{T11}/result.json")
    calendar = read(f"{DIVIDEND}/result.json")
    return {
        "bounded_saved_schemas": schemas,
        "IF_raw_date_coverage": {k: read(f"{IF}/data_preflight.json")[k]
                                 for k in ("if_rows", "if_contracts", "if_days", "if_first", "if_last",
                                           "same_day_publication_timestamp_proven", "historical_lag_is_assumption")},
        "old_IF_paired_increment": {k: oi[k] for k in ("status", "evaluation", "account_stage", "net_sharpe",
                                                          "availability_timestamp_proven", "availability_assumption")},
        "nominal_basis_formula": "log(IF close/spot close)*365/calendar_days_to_expiry；旧实际代码未扣融资成本和成分待分红。",
        "old_eight_nominal_rules": {k: read(BINARY)[k] for k in ("status", "scope", "candidate_count",
                                                                 "result_rows", "passing_candidates", "decision")},
        "H01_H02_source_gate": [r for r in read(f"{PROGRAM}/factor_progress.json") if r["id"] in ("H01", "H02")],
        "H05_actual_protocol": {k: read(f"{STRUCTURE}/protocol.json")[k]
                                for k in ("hypotheses", "expiry_rule", "source_clock", "next_stage_filter")},
        "EPS_growth_definition_disagreement": {k: eps[k] for k in ("status", "evaluation", "account_stage")},
        "EPS_label_limit": "两机构报告年度EPS增长定义之差，非纯同分析师预测不确定性或真实修正。",
        "explicit_revision_two_event_limits": {k: novelty[k] for k in ("status", "census_reports", "diagnostic_events",
                                              "new_original_numeric_forecast_values", "all_possible_report_text_features_rejected",
                                              "timing_alternative_tested", "label_alias_admitted")},
        "initial_measurement_known_invalidations": read(f"{MEASURE}/source_invalidation_addendum.json"),
        "later_repaired_cohort_contract": read(f"{COHORT}/protocol.json"),
        "later_repaired_cohort_measurement": read(f"{COHORT}/result.json"),
        "final_T11_not_early_measurement": {k: t11[k] for k in ("status", "accounts", "signals", "original_T11", "scope", "source_limits")},
        "final_T11_clock": read(f"{T11}/source_receipt.json")["financial_clock"],
        "T11_supersession_limit": "引用后续独立修复版本及最终日期代理检验；原错误测量保留，未声称所有修复范围外字段或历史首版已验证。",
        "normalization_family": read(VAL),
        "VAL01_gate": read("reports/research/val01_norm_ey_5y_v1_predictive_diagnostics.json")["primary_predictive_gate"],
        "VAL02_gate": read("reports/research/val02_norm_ey_spread_5y_v1_predictive_diagnostics.json")["primary_predictive_gate"],
        "dividend_calendar_original_gross_stage": {k: calendar[k] for k in ("status", "source_gate", "numeric_gate", "primary", "account_status")},
        "dividend_calendar_final_account_status": read(f"{DIVIDEND}/account_stage/result.json")["status"],
        "dividend_calendar_source_contract": read(f"{DIVIDEND}/source_receipt.json"),
        "current_1507_member_support": "NOT_COMPUTED_NO_COMPLETE_PURPOSE_SOURCE_ADMITTED",
        "current_115_mature_month_support": "NOT_COMPUTED_NO_COMPLETE_PURPOSE_SOURCE_ADMITTED",
        "limits": "原注册NOT_RUN、旧组件失败、部分代理已测及毛筛到最终账户分别保留；未重新拟合或回测，未认定因果，也未宣称其他数据不存在。",
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError("本轮目录已存在，禁止覆盖或重复运行")
    for row in read(PLAN)["sources"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"提案来源已变：{row['path']}")
    paths = sources()
    manifest = [{"path": p, "sha256": sha(p), "bytes": (ROOT / p).stat().st_size} for p in paths]
    outputs = {"cases.json": cases(), "saved_old_primary_account_quotes.json": account_quotes(),
               "source_and_old_purpose_facts.json": source_facts()}
    OUT.mkdir(parents=True)
    save("evidence_manifest.json", manifest)
    save("protocol.json", {"at": now(), "study_id": "510300_POINT_IF_EARNINGS_PRIOR_ROUTES_V1",
         "decision": "TECH.R111", "scope": "有限十一旧用途及原源合同，七份保存表只读取模式/行数；六主行摘录，非新实验。",
         "plan": PLAN, "field_bound": False, "field_admitted": False,
         "same_day_or_first_publication_proven": False, "old_frozen_files_modified": False,
         "old_study_reopened": False, "new_predictions_accounts_collection": 0})
    for name, value in outputs.items():
        save(name, value)
    save("summary.json", {"at": now(), "study_id": "510300_POINT_IF_EARNINGS_PRIOR_ROUTES_V1",
         "technical_decision": "TECH.R111", "status": "NOT_ADMITTED_ELEVEN_IF_EARNINGS_PRIOR_SOURCE_ROUTES",
         "completed_prior_routes": 11, "source_count": len(manifest), "old_account_quotes": 6,
         "saved_schema_tables_checked": len(SCHEMAS), "field_bound": False, "field_admitted": False,
         "new_strategy_configurations": 0, "new_ols_or_return_fits": 0, "new_return_labels": 0,
         "new_predictions": 0, "new_accounts": 0, "network_requests": 0,
         "current_prediction_stage": "NOT_RUN_NO_COMPLETE_PURPOSE_SOURCE_ADMITTED",
         "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
         "independent_validation": "NOT_ESTABLISHED", "overall_overfit_removed": False,
         "goal_achieved": False, "goal_turn_classification": "progress", "blocked_audit_count": 0})
    text = "# 期货日线与盈利信息十一项核对（TECH.R111）\n\n十一项当前均未准入新的点位模型。核心事实：同模型IF单日持仓增量已失败，HHI两个固定方向没有继续资格；盈利现金质量的后续日期代理账户也已失败；分红日历毛筛PASS之后的完整账户已拒绝，不能停留在中间状态。\n\n"
    text += "| 旧用途，20万元压力成本 | 原年化收益 | 原净夏普 | 含义 |\n| --- | ---: | ---: | --- |\n"
    text += "| IF持仓集中度高方向 | −0.776238% | −0.532676 | 固定初筛未通过 |\n| IF持仓集中度低方向 | 0.326475% | 0.123218 | 后段收益负，未通过 |\n| 盈利意外×现金质量×首轮反应T11 | −0.270499% | −0.351761 | 最终日期代理账户拒绝 |\n| 原财报YTD盈利广度 | 0.296111% | 0.087205 | 原目标未达 |\n| 分红日历广度 | −0.556401% | −0.291530 | 日历年化，毛筛后账户拒绝 |\n\n"
    text += "这些为保存的原结果，原年化242或原日历时钟各自保留，不拼接或排序成本线252账户。IF总OI配对和EPS增长定义分歧失败后账户未运行，没有它们的新收益夏普。\n\n"
    for row in outputs["cases.json"]:
        text += f"- **{row['id']} {row['name']}**：{row['result']}\n\n"
    text += "原公告测量的七个错字段与后续修复、原YTD广度与真正TTM、完整原卡与日期代理分别保存。HHI是合约持仓分布；OI不是净多/净空。IF名义基差不是已扣成分分红/资金成本的基差，ETF基金分红不是指数成分待分红。实际IF保存源到2026-08-12，取得时间与保守滞后均不证明历史首次可得。\n\n"
    text += "本轮十一假设和六原主行、七保存模式核对，不计算新1507/115覆盖值，也不以稀疏公告补全原训练成员。新字段/配置/标签/拟合/预测/账户/采集均0，净收益和夏普NOT_COMPUTED，独立验证未建立，整体过拟合未去除、完整目标未达。下一优先检查可从原日线计算的P03持仓路径时间信息和N04季末交互；先排除旧同用途，P04与现有等待价值目标的定义关系只作核对。通过后才注册唯一固定预测比较，原八特征及目标、原成员与成熟时钟、双期门和经济门保持。\n\n"
    text += "直接证据：[十一分项](cases.json)、[六主账户原行](saved_old_primary_account_quotes.json)、[来源与最终旧裁决](source_and_old_purpose_facts.json)、[保存核对](saved_output_verification_receipt.json)。\n"
    with (OUT / "研究结论与下一步.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print("十一项期货日线/盈利旧用途与源合同已保存；六旧主行，新增模型与账户0。")


def verify() -> None:
    manifest = read((OUT / "evidence_manifest.json").relative_to(ROOT).as_posix())
    for row in manifest:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError(f"冻结来源变化：{row['path']}")
    expected = {"cases.json": cases(), "saved_old_primary_account_quotes.json": account_quotes(),
                "source_and_old_purpose_facts.json": source_facts()}
    for name, value in expected.items():
        if json.loads((OUT / name).read_text(encoding="utf-8")) != value:
            raise ValueError(f"保存事实不一致：{name}")
    save("saved_output_verification_receipt.json", {"at": now(),
         "status": "PASS_SAVED_ELEVEN_IF_EARNINGS_PRIOR_SOURCE_ROUTE_VERIFICATION",
         "source_count": len(manifest), "case_count": 11, "saved_primary_rows": 6,
         "saved_schema_tables_checked": len(SCHEMAS), "new_fits_labels_predictions_accounts": 0,
         "network_requests": 0, "limit": "只核保存来源和摘录；不证明首次可得、当前字段资格或金融目标改善。"})
    print("十一分项、六原主行及七保存表模式核对通过；原策略与未知状态保持。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="期货日线与盈利信息十一项有限核对")
    parser.add_argument("operation", choices=("run", "verify"))
    args = parser.parse_args()
    run() if args.operation == "run" else verify()
