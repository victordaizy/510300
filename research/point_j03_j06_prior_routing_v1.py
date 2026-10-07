"""J03—J06有限旧用途路由：只摘录保存事实，不运行字段、模型或账户。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_j03_j06_prior_source_routing_v1"
NEXT = ROOT / "reports/research/510300_point_next_information_intake_20261002"
PLAN = NEXT / "J03_J06_cross_market_prior_source_routing_plan.json"
STUDY = "510300_POINT_J03_J06_PRIOR_SOURCE_ROUTING_V1"
PREFIX = "reports/research/"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write(path: Path, value: dict | list) -> None:
    require(not path.exists(), f"已保存文件不覆盖：{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def subset(value: dict, keys: list[str]) -> dict:
    require(all(key in value for key in keys), "旧记录字段变化，停止摘录，不推断缺项。")
    return {key: value[key] for key in keys}


def extract() -> list[dict]:
    """固定四项旧主目的；不选择事后最好账户，不把原卡片与实现混为一谈。"""
    j03 = read(ROOT / (PREFIX + "510300_factor96_crossborder_absorption_v1/result.json"))
    j03_source = read(ROOT / (PREFIX + "510300_factor96_crossborder_absorption_v1/source_receipt.json"))
    j03_rows = [r for r in j03["primary_rows"] if r["capital"] == 200000 and r["cost"] == "STRESS"
               and r["period"] == "MAIN" and r["lag"] == 0 and r["policy"] == "FULL"]
    require(len(j03_rows) == 1 and j03["historical_joint_point_pass"] is False, "J03旧主用途或裁决变化。")
    j04 = read(ROOT / (PREFIX + "510300_ah_premium_increment_v1/result.json"))
    j04_config = read(ROOT / "config/510300_ah_premium_increment_v1.json")
    j04_rows = [r for r in j04["accounts"] if r["cost"] == "STRESS" and r["model"] in ["M0", "M1"]]
    require(len(j04_rows) == 2 and j04["status"] == "REJECTED_FROZEN_NO_RELIABLE_AH_INCREMENT", "J04旧裁决变化。")
    j05 = read(ROOT / (PREFIX + "510300_copper_gold_monthly_vintage_v1/result.json"))
    require(j05["status"] == "PARTIAL_GROSS_SCREEN_FAILED_WITH_SOURCE_GAPS"
            and j05["account_status"] == "NOT_RUN", "J05旧数值/来源/账户裁决变化。")
    j06 = read(ROOT / (PREFIX + "510300_factor96_rapid_external_resilience_v1/result.json"))
    j06_increment = read(ROOT / (PREFIX + "510300_factor96_rapid_external_resilience_v1/increment_diagnostic.json"))
    j06_rows = [r for r in j06["primary_results"] if r["capital"] == 200000 and r["cost"] == "STRESS" and r["candidate"]]
    require(len(j06_rows) == 3 and j06["qualified_candidates"] == 0 and not j06["worth_followup"], "J06旧裁决变化。")
    common = {"current_field_bound": False, "current_field_admitted": False, "current_model_registered": False,
              "current_member_support": "NOT_COMPUTED", "current_predictive_increment": "NOT_RUN",
              "current_financial_result": "NOT_COMPUTED", "old_failures_reopened": False}
    return [
        {**common, "id": "J03", "name": "海外中国资产消息与A股反应差",
         "old_purpose": "ASHR正信息、A股完整首日仍上涨但比OLS预测低一个残差标准差；16:00确认后次开盘入场。",
         "original_card_boundary": "原卡为次日开盘/首段反应；旧开盘残差只保存描述，没有完成原卡或当前自然继续目标的检验。",
         "routing_status": "OLD_FULL_FIRST_DAY_FAILED_OPENING_PURPOSE_NOT_ADMITTED",
         "old_facts": subset(j03, ["new_accounts", "new_candidates", "historical_joint_point_pass", "response_models", "increments"]),
         "saved_old_primary_accounts": j03_rows, "old_account_annual_days": 242,
         "source_facts": subset(j03_source, ["ashr_rows", "ashr_cash_distributions", "ashr_adjusted_close_used", "fx_rows",
                                             "historical_publication_clock", "first_public_version_authenticated", "cash_distribution_values_source"]),
         "accept_reject_reason": "保留旧完整首日失败；原开盘描述与当前15:05源/用途没有绑定，不将16:00计算当成此前已发布。中间价不是离岸交易汇率，ASHR权益首版未逐笔认证。",
         "revalidation": "不同明确经济用途及合格源才另立；不改旧一倍标准差、符号、时段或退出规则重试。"},
        {**common, "id": "J04", "name": "A/H相对定价压力",
         "old_purpose": "公开恒生A/H溢价聚合指数的对数水平及五日变化补充ETF/HSI，预测五日独立净收益。",
         "original_card_boundary": "原卡要求历史双重上市公司匹配、同币价格与当时权重或明确等权；现有指数来源不证明具备该公司面板。",
         "routing_status": "OLD_AGGREGATE_INCREMENT_REJECTED_MATCHED_PANEL_NOT_ESTABLISHED",
         "old_facts": subset(j04, ["status", "evaluation", "economic_increment", "model_fits", "refit_months", "new_accounts",
                                   "continuation_gate_pass", "historical_stress_point_target_pass"]),
         "saved_old_primary_accounts": j04_rows, "old_account_annual_days": j04_config["annual_days"],
         "source_facts": subset(j04, ["historical_first_delivery_timestamp_proven", "historical_availability_assumption", "limited_source_history_start"]),
         "accept_reject_reason": "629配对原点的MSE扩大2.612%，预测增量门失败。M1压力账户点值高于M0，同时暴露更高；增量区间跨0，不能写成账户所有数值都变差或稳定独立改善。",
         "revalidation": "保留公开聚合指数用途拒绝；不得以换窗、符号、指数供应商或事后年份救回。原卡完整匹配面板需独立的实际来源合同。"},
        {**common, "id": "J05", "name": "铜金价格比的条件变化",
         "old_purpose": "每月原版Pink Sheet已完成月铜金比环比走强且ETF原20日趋势正，固定20开盘间隔。",
         "original_card_boundary": "旧协议明确月度原版是不同用途，不能等同原卡过去20日铜金日频变化；缺六个月没有被检验。",
         "routing_status": "OLD_MONTHLY_PARTIAL_NUMERIC_FAILED_DAILY_TWENTY_DAY_PURPOSE_NOT_BOUND",
         "old_facts": subset(j05, ["status", "registered_months", "source_months", "missing_months", "statuses", "primary_available_subsample",
                                   "price_control_available_subsample", "numeric_gates_on_available_subsample", "source_complete_gate",
                                   "partial_numeric_screen_passed", "account_status", "net_sharpe", "net_cagr", "maximum_drawdown", "new_accounts"]),
         "saved_old_primary_accounts": [], "old_account_annual_days": None,
         "source_facts": {"source": "World Bank Pink Sheet逐月原版官方PDF", "proxy_clock": "PDF表头发布日期后2自然日23:59中国时间",
                          "actual_historical_first_seen_proven": False, "full_source_complete": j05["source_complete_gate"]},
         "accept_reject_reason": "102/108原版可用，26主事件均值高于费用代理/价格对照，但后段均值不为正，部分数值与完整来源门均失败。费用代理不是完整CNY账户，净夏普/CAGR未算。",
         "revalidation": "不通过改频率、铜金/趋势窗口、持有期或象限重试旧失败；只有具体新官方缺失原文才补源，原日频卡仍未准入。"},
        {**common, "id": "J06", "name": "外部冲击的抗跌程度",
         "old_purpose": "预定GSPC负向冲击后，以完整A股首日响应残差和绝对不跌固定三个候选，随后五开盘间隔。",
         "original_card_boundary": "这是已完成的GSPC负冲击/完整首日用途；不把T10正信息当成此用途，不宣称所有美债事件或抗跌机制均已检验。",
         "routing_status": "OLD_THREE_RAPID_CANDIDATES_NOT_ACCEPTED_RELATIVE_FILTER_REDUNDANT",
         "old_facts": subset(j06, ["classification", "account_scenarios", "candidate_rules", "source_known_days", "shock_events", "event_counts",
                                   "worth_followup", "joint_historical_target_count", "qualified_candidates"]),
         "saved_old_primary_accounts": j06_rows, "old_account_annual_days": 242,
         "source_facts": subset(j06, ["causal_prefix_check", "historical_vendor_first_delivery_proven"]),
         "saved_old_increment_diagnostic": j06_increment,
         "accept_reject_reason": "三个候选均未通过预定继续筛选。绝对不跌与联合候选信号和四份账本完全相同，相对条件未带来额外筛选，不能算两份独立证据。算法前缀通过不证明供应商历史首版。",
         "revalidation": "不改冲击阈值、两年模型、五日、首日条件、费用或样本救回；不同机制/实际独立证据需另立，旧三候选终态保持。"},
    ]


def route() -> None:
    require(not OUT.exists(), "本批路由已建立，不覆盖、不重复运行。")
    plan = read(PLAN)
    require(plan["candidate_ids"] == ["J03", "J04", "J05", "J06"], "有限提案对象变化。")
    for source in plan["sources"]:
        require(identity(ROOT / source["path"])["sha256"] == source["sha256"], f"既有提案来源改变：{source['path']}")
    paths = {PLAN, Path(__file__).resolve()}
    paths.update(ROOT / s["path"] for s in plan["sources"])
    old_files = {
        "510300_factor96_crossborder_absorption_v1": ["protocol.json", "result.json", "source_receipt.json", "source_evidence/overlap_and_source_decisions.json"],
        "510300_ah_premium_increment_v1": ["result.json", "data_preflight.json", "deduplication.json"],
        "510300_copper_gold_monthly_vintage_v1": ["protocol.json", "result.json", "source_qa_receipt.json", "source_gap_registration.json"],
        "510300_factor96_rapid_external_resilience_v1": ["protocol.json", "result.json", "increment_diagnostic.json"],
    }
    paths.update(ROOT / PREFIX / folder / name for folder, files in old_files.items() for name in files)
    paths.update([ROOT / "config/510300_ah_premium_increment_v1.json", ROOT / "docs/510300_AH_PREMIUM_INCREMENT_V1_PROTOCOL.md"])
    require(all(p.is_file() for p in paths), "有限来源文件缺失，停止，不搜索替代版本。")
    manifest = [identity(p) for p in sorted(paths)]
    cases = extract()
    OUT.mkdir(parents=True)
    stamp = datetime.now().astimezone().isoformat()
    protocol = {"study": STUDY, "at": stamp, "prior_plan": PLAN.relative_to(ROOT).as_posix(),
                "classification": "FINITE_READ_ONLY_PRIOR_PURPOSE_AND_SOURCE_ROUTING",
                "old_results_already_read_before_this_evidence_seal": True,
                "evidence_seal_role": "旧事实摘录与保存身份，非新实验预注册、物理首版认证或金融验证。",
                "current_prediction_states": "NOT_READ_NOT_COMPUTED", "current_member_support": "NOT_COMPUTED",
                "candidate_ids": plan["candidate_ids"], "original_registry_and_program_unchanged": True,
                "old_account_selector": "只引用各旧协议保存的20万元压力主账户；J03固定MAIN/lag0/FULL，AH固定M0/M1，J06全三个候选；不择最好。",
                "new_strategy_configurations": 0, "new_model_fits": 0, "new_return_labels": 0,
                "new_strategy_accounts": 0, "new_market_collection": 0, "independent_validation": "NOT_ESTABLISHED",
                "history_role": "DEVELOPMENT_CALIBRATION", "goal_achieved": False}
    write(OUT / "protocol.json", protocol)
    write(OUT / "evidence_manifest.json", manifest)
    write(OUT / "cases.json", cases)
    with (OUT / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "old_purpose", "original_card_boundary", "routing_status", "accept_reject_reason", "revalidation"])
        writer.writeheader()
        writer.writerows({key: item[key] for key in writer.fieldnames} for item in cases)
    accounts = [{"card_id": item["id"], "old_account_annual_days": item["old_account_annual_days"], "saved_old_account": row}
                for item in cases for row in item["saved_old_primary_accounts"]]
    write(OUT / "saved_old_primary_account_quotes.json", accounts)
    summary = {**protocol, "status": "COMPLETED_FINITE_FOUR_CASE_ROUTING_NO_CURRENT_FIELD_ADMITTED",
               "reviewed_cases": len(cases), "evidence_files": len(manifest), "saved_old_account_rows_quoted": len(accounts),
               "current_fields_admitted": 0, "new_account_return_sharpe": "NOT_COMPUTED",
               "cases": [{"id": c["id"], "status": c["routing_status"]} for c in cases],
               "next_question": "先做有限一次E10旧仓位/效用/目标单位和保存方差角色核对；不预先绑定新公式或风险系数。"}
    write(OUT / "summary.json", summary)
    lines = ["# J03—J06四项跨市场旧用途与来源路由", "",
             "本轮完成四项旧研究的直接协议、结果和来源核对。没有新增可进入当前模型的字段，也没有新策略收益或夏普结果。", "",
             "原卡片、实际旧用途、来源门和金融门分别保留。旧结果已经看过；本次保存来源身份不构成新实验预注册或独立验证。", "",
             "| 卡片 | 实际旧用途结果 | 当前接续判断 |", "|---|---|---|",
             "| J03 海外反应差 | 完整首日策略20万元压力主账户：CAGR−0.173894%，夏普−0.306678，仅1已闭周期 | 原开盘卡只存描述；未绑定当前用途与源 |",
             "| J04 A/H压力 | 629配对原点，MSE扩大2.612%；原聚合指数增量REJECTED | 没有证明具备历史匹配公司、同币价及权重面板 |",
             "| J05 铜金条件 | 原版102/108；26主事件毛均值1.278899%，比例费用代理后0.998899%；后段不为正 | 完整账户NOT_RUN；原日频20日卡未绑定 |",
             "| J06 外部抗跌 | 3候选/32旧账户全部未接受；相对候选夏普−0.106649，绝对/联合0.255778 | 绝对与联合信号/账本相同；不计两份独立证据 |", "",
             "上述已有账户数字采用各自原242日口径、时期和风险合同，不能与当前252日A账户直接排名。J05比例费用代理不包括完整现金、T+1、整手、最低佣金和路径风险，不能据此生成账户夏普。", "",
             "A/H需要特别区分：M0压力年化1.313437%、夏普0.346948；M1为2.228376%、0.420020，暴露由12.117%增至19.539%，回撤绝对值由3.503%增至4.708%。账户点值有所提高，但原预测门失败，配对收益增量区间跨0且不是暴露中性的比较。不能记录为稳定改善，也不能笼统写作所有账户指标恶化。", ""]
    for item in cases:
        lines += [f"## {item['id']} {item['name']}", "", f"旧用途：{item['old_purpose']}", "",
                  f"原卡边界：{item['original_card_boundary']}", "", f"裁决理由：{item['accept_reject_reason']}", "",
                  f"重验条件：{item['revalidation']}", ""]
    lines += ["## 下一步", "",
              "暂停逐项扩展这四种旧来源。先核当前原预测的经济单位、已做仓位方法和可知方差角色，判断有没有实质不同的资金分配问题。旧adaptive_allocation已使用均值−风险−交易费用效用，不能将同一方法重新包装成新研究；E08随机截距方差来自带周期成员数的拟合协方差，也不能直接充当下一日标的方差。", "",
              "下一E10只有有限旧用途/单位核对提案。若没有不同用途或无法建立实际资金/时间尺度映射，即结束该项；未绑定γ、Kelly比例、置信阈值、交易规则或账户。原A、E03前瞻日历与各失败保持，当前完整收益和夏普目标未达。", "",
              "直接机器事实：[四项完整记录](cases.json)、[六行旧主账户引用](saved_old_primary_account_quotes.json)、[当前状态](summary.json)、[来源身份](evidence_manifest.json)。"]
    (OUT / "研究结论与下一步.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"结果": summary["status"], "四项": len(cases), "旧账户引用": len(accounts), "新拟合": 0, "新账户": 0}, ensure_ascii=False))


def verify() -> None:
    receipt_path = OUT / "saved_output_verification_receipt.json"
    require(not receipt_path.exists(), "保存核对已完成，不重复执行。")
    manifest = read(OUT / "evidence_manifest.json")
    for source in manifest:
        require(identity(ROOT / source["path"]) == source, f"引用来源改变：{source['path']}")
    require(read(OUT / "cases.json") == extract(), "四项保存旧事实不相同。")
    expected = [{"card_id": item["id"], "old_account_annual_days": item["old_account_annual_days"], "saved_old_account": row}
                for item in read(OUT / "cases.json") for row in item["saved_old_primary_accounts"]]
    require(read(OUT / "saved_old_primary_account_quotes.json") == expected, "旧主账户引用不相同。")
    with (OUT / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_csv = list(csv.DictReader(stream))
    expected_csv = [{key: item[key] for key in saved_csv[0]} for item in read(OUT / "cases.json")]
    require(saved_csv == expected_csv, "四项表格与完整记录不相同。")
    summary = read(OUT / "summary.json")
    require(summary["reviewed_cases"] == 4 and summary["saved_old_account_rows_quoted"] == 6
            and summary["current_fields_admitted"] == 0 and summary["new_strategy_accounts"] == 0, "路由计数错误。")
    receipt = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_PRIOR_FACTS_AND_SOURCE_IDENTITY",
               "evidence_files_checked": len(manifest), "cases_checked": 4, "saved_old_account_rows_checked": 6,
               "current_member_support": "NOT_COMPUTED", "new_model_fits": 0, "new_account_evaluations": 0,
               "old_models_or_accounts_reexecuted": False, "claim_limit": "只核保存事实与来源身份，不证明策略有效或历史首次公开。"}
    write(receipt_path, receipt)
    print(json.dumps({"核对": receipt["status"], "来源": len(manifest), "四项": 4, "旧账户引用": 6}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="J03—J06有限旧事实路由，不执行研究模型或账户。")
    parser.add_argument("--verify", action="store_true", help="只核保存结果和所读文件身份一次。")
    args = parser.parse_args()
    verify() if args.verify else route()


if __name__ == "__main__":
    main()
