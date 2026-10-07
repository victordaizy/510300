"""有限核对旧融资用途与来源，仅读保存结果，不运行收益模型或账户。"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_f01_f03_prior_source_routing_v1"
NEXT = "reports/research/510300_point_next_information_intake_20261002"
BATCH = "reports/research/510300_factor96_mechanism_batch_v1"
CROWD = "reports/research/510300_factor96_crowding_overlay_v1_0_1"
RAPID = "reports/research/510300_factor96_rapid_feasibility_v1"
DAILY = "reports/research/510300_factor96_daily_state_shrink_v1"
CONTRACT = "reports/research/510300_historical_index_margin_contract_v1"
STUDY = "510300_POINT_F01_F03_PRIOR_SOURCE_ROUTING_V1"
STATUS = "NOT_ADMITTED_FINANCING_PRIOR_AND_SAVED_SOURCE_ROUTING_COMPLETE"


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name: str, value) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv_save(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_paths() -> list[str]:
    plan = read(f"{NEXT}/F01_F03_financing_prior_source_routing_plan.json")
    paths = [entry["path"] for entry in plan["sources"]]
    for entry in plan["sources"]:
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"原提案来源已变化：{entry['path']}")
    paths += [
        str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        f"{NEXT}/F01_F03_financing_prior_source_routing_plan.json",
        f"{BATCH}/completed_run/protocol.json", f"{BATCH}/completed_run/result.json",
        f"{BATCH}/completed_run/data_admission.json", f"{BATCH}/completed_run/inputs/margin.parquet",
        f"{BATCH}/run_failure_01.json", f"{BATCH}/data_repair/source_receipt.json",
        f"{CROWD}/protocol.json", f"{CROWD}/result.json", f"{CROWD}/source_receipt.json",
        f"{CROWD}/inputs/margin.parquet", "research/factor96_rapid_feasibility_v1.py",
        f"{RAPID}/protocol.json", f"{RAPID}/result.json", f"{RAPID}/all_account_metrics.csv",
        "research/factor96_daily_state_shrink_v1.py", f"{DAILY}/protocol.json",
        f"{DAILY}/result.json", f"{DAILY}/source_receipt.json",
        f"{CONTRACT}/result.json", f"{CONTRACT}/necessary_checks.json",
        f"{CONTRACT}/sources/sse_margin_202601.json",
        f"{CONTRACT}/sse_margin_source_receipt.json",
        f"{CONTRACT}/sources/sse_definition.html",
        f"{CONTRACT}/supplementary_source_receipts.json",
    ]
    return list(dict.fromkeys(paths))


def old_account_quotes() -> list[dict]:
    """选取提案所涉及用途的原主账户；不读账本或重新计算收益。"""
    output = []
    selections = [
        (f"{BATCH}/completed_run/result.json", "main_stress_candidate_rows", ("T03", "T05")),
        (f"{CROWD}/result.json", "primary_rows", ("FULL",)),
        (f"{DAILY}/result.json", "primary_accounts", ("FULL",)),
    ]
    for path, key, policies in selections:
        rows = [row for row in read(path)[key]
                if row["capital"] == 200000 and row["cost"] == "STRESS"
                and row["policy"] in policies and row.get("lag", 1) == 1]
        if len(rows) != len(policies):
            raise ValueError(f"旧主账户选择数不符：{path}")
        for row in rows:
            output.append({"source": path, "annual_days": 242, "saved_row": row})
    path = f"{RAPID}/all_account_metrics.csv"
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["policy"] in ("F03_HIGH", "F03_LOW")
                and Decimal(row["capital"]) == 200000 and row["cost"] == "STRESS"]
    if len(rows) != 2:
        raise ValueError("F03两个预定旧主账户未完整找到")
    # 保留CSV原字符串，避免把保存数值的小数表示换成另一个结果版本。
    output.extend({"source": path, "annual_days": 242, "saved_csv_row": row} for row in rows)
    return output


def margin_source_facts() -> dict:
    tables = []
    for path in (f"{BATCH}/completed_run/inputs/margin.parquet", f"{CROWD}/inputs/margin.parquet"):
        frame = pd.read_parquet(ROOT / path)
        dates = pd.DatetimeIndex(frame["date"])
        tables.append({"path": path, "rows": len(frame), "columns": list(frame.columns),
                       "date_start": str(dates.min().date()), "date_end": str(dates.max().date()),
                       "dates_unique": dates.is_unique, "dates_sorted": dates.is_monotonic_increasing,
                       "direct_reported_repayment_present": "rzche" in frame.columns,
                       "source_values": sorted(frame["source"].dropna().unique().tolist()),
                       "publication_rule_values": sorted(frame["publication_rule"].dropna().unique().tolist())})
    admission = read(f"{BATCH}/completed_run/data_admission.json")
    repair = read(f"{BATCH}/data_repair/source_receipt.json")
    return {"saved_tables": tables, "original_admission": admission,
            "old_repair": {key: repair[key] for key in (
                "missing_dates", "records_added", "unchanged_existing_balance_and_buy_rows",
                "complete_days", "strategy_or_parameter_changes", "new_strategy_results_seen_before_repair")},
            "repair_role": "原11深市统计日补缺发生于收益评估前，不是本轮新采集或营救失败策略。",
            "amount_unit": "CNY；深市官方旧修复为亿元两位小数，约100万元精度。",
            "population": "两交易所融资汇总，不等同CSI300成员或股票专属资金流；细项与汇总的调出标的范围不同。",
            "implied_repayment": "买入额减余额变化；不是直接报告偿还、强平或净主力资金。",
            "misleading_column_note": "market_financing_balance_identity_residual保存的是余额变化减买入，即负隐含偿还；未独立使用直报偿还，不能当三字段恒等审计。",
            "physical_first_publication": "NOT_ESTABLISHED",
            "current_1507_natural_support": "NOT_COMPUTED",
            "current_115_mature_month_support": "NOT_COMPUTED",
            "support_boundary": "2015—2025两市表不自动支持2014成熟训练和2026当前全部成员；不缩样本或以沪市2026单月代替两市完整源。"}


def repayment_counterexample() -> list[dict]:
    """只核原官方JSON的会计差额；不计算F5、收益标签、事件收益。"""
    raw = sorted(read(f"{CONTRACT}/sources/sse_margin_202601.json")["result"], key=lambda row: row["opDate"])
    if len(raw) != 20 or len({row["opDate"] for row in raw}) != 20:
        raise ValueError("原2026年1月沪市来源行数或唯一日期与保存描述不符")
    output, previous = [], None
    for row in raw:
        balance, buy, reported = (Decimal(str(row[key])) for key in ("rzye", "rzmre", "rzche"))
        implied = None if previous is None else buy - (balance - previous)
        residual = None if implied is None else implied - reported
        output.append({"date": datetime.strptime(str(row["opDate"]), "%Y%m%d").date().isoformat(),
                       "balance_cny": str(balance), "buy_cny": str(buy), "reported_repayment_cny": str(reported),
                       "implied_repayment_cny": None if implied is None else str(implied),
                       "implied_minus_reported_cny": None if residual is None else str(residual),
                       "status": "NOT_COMPUTED_NO_PRIOR_ROW_IN_THIS_SOURCE" if previous is None else "COMPUTED_SOURCE_IDENTITY_ONLY"})
        previous = balance
    known = [row for row in output if row["implied_minus_reported_cny"] is not None]
    extreme = max(known, key=lambda row: abs(Decimal(row["implied_minus_reported_cny"])))
    original = read(f"{CONTRACT}/necessary_checks.json")
    if extreme["date"] != original["max_daily_residual_date"] or Decimal(extreme["implied_minus_reported_cny"]) != Decimal(str(original["max_daily_residual_cny"])):
        raise ValueError("原官方JSON差额与保存必要核对不符")
    return output


def implementation_evidence() -> dict:
    path = "research/factor96_rapid_feasibility_v1.py"
    source = (ROOT / path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    residual = ast.get_source_segment(source, functions["prior_residual"])
    if not all(token in residual for token in ("pd.DateOffset(years=2)", "[-252:]", "len(eligible) < 126", "np.linalg.lstsq")):
        raise ValueError("F03旧源码的已读实现身份不符")
    lines = source.splitlines()
    return {"path": path, "prior_residual_first_line": functions["prior_residual"].lineno,
            "prior_residual_source": residual,
            "financing_lines": [{"line": index + 1, "text": text} for index, text in enumerate(lines)
                                if 139 <= index + 1 <= 143],
            "actual_fit_clock": "仅当前日前、先前两日历年、最近至多252个合格日；最少126，不是严格满252。",
            "original_card_clock": "此前252个合格日。",
            "difference_role": "初筛实现与原卡完整定义有差异，保留原NOT_RUN/完整验证false；不修改最少126、两年或方向重试。",
            "lag": "先计算完整统计日同期收益/波动与余额残差，再整体后移1交易日。",
            "high_low_direction": "HIGH/LOW均为旧做多入场值域筛选，未回测空头。"}


def cases() -> list[dict]:
    plan = read(f"{NEXT}/F01_F03_financing_prior_source_routing_plan.json")
    program = read("reports/research/510300_factor96_program_v1/factor_progress.json")
    cards = {row["id"]: row for row in plan["registered_cards"]}
    states = {row["id"]: row for row in program if row["id"] in cards}
    facts = [
        ("F01", "融资净扩张率", "同价格背景下额外融资需求是否提供新信息。",
         "旧F01为余额五日变化/起点余额；T18日更A01/E01/F01组件和拥挤FULL已测，不能称F01从未使用。直接买入减直报偿还版尚未完整验证。",
         "日更FULL主账户夏普−0.0070148；拥挤FULL−0.6366248；均旧242口径、目标未达，不能把多组件失败当F01独立无效。",
         "OLD_BALANCE_COMPONENTS_TESTED_DIRECT_FLOW_AND_CURRENT_SOURCE_NOT_ADMITTED"),
        ("F02", "融资收缩的缓和速度", "净收缩仍负但减速是否有利于已冻结价格修复事件。",
         "T03已联合使用F5<0及F5−前非重叠五日F5>0；固定负向冲击后第1—3日确认与原退出完整保留。",
         "原主期20万压力lag1闭合周期0、CAGR0、夏普null；没有事件不能称胜率0或夏普0，不放宽确认/融资条件创造交易。",
         "OLD_FIXED_T03_ZERO_EVENTS_RETAINED_CURRENT_SOURCE_NOT_ADMITTED"),
        ("F03", "价格无法解释的融资变化", "价格/波动之外的融资残差是否有可交易增量。",
         "旧初筛实际两年内最近126—252合格日OLS，余额变化代理，滞后1；两年30/70分位、十开盘间隔；并非严格原卡252满窗或因果识别。",
         "HIGH/LOW旧20万压力做多筛选CAGR−0.523029%/−0.827645%，夏普−0.234242/−0.331546，112/105闭合周期，两个都未获继续资格。",
         "OLD_RAPID_TWO_LONG_DIRECTIONS_FAILED_FULL_CARD_NOT_VALIDATED_SOURCE_NOT_ADMITTED"),
    ]
    return [{"id": factor, "name": name, "hypothesis": hypothesis,
             "verification_method": "有限读取旧直接协议/源码/主结果及保存官方原文，摘录六原账户行，核两余额表与20官方源记录；不运行OLS、收益模型或账户。",
             "original_card": cards[factor], "saved_program_status": states[factor],
             "old_purpose": purpose, "result": result, "routing_status": status,
             "acceptance_or_rejection": "接受旧用途及源口径事实；当前两市直报偿还、逐日首版/修订和原全部成员资格未建立，当前字段/模型不准入；不是经济机制永远无效的证明。",
             "revalidation": "只在不同用途和独立合格来源成立时另立唯一固定比较；不改旧窗口、方向、阈值、费用、样本或供应商营救。",
             "field_bound": False, "field_admitted": False,
             "current_1507_support": "NOT_COMPUTED", "current_115_support": "NOT_COMPUTED"}
            for factor, name, hypothesis, purpose, result, status in facts]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError("本轮输出已存在，请执行verify核保存结果，不覆盖原结果")
    paths = source_paths()
    save("evidence_manifest.json", [{"path": path, "sha256": digest(ROOT / path)} for path in paths])
    save("protocol.json", {"at": now(), "study_id": STUDY, "technical_decision": "TECH.R108",
         "classification": "FINITE_PRIOR_AND_SOURCE_ROUTING_OLD_OUTCOMES_KNOWN_NOT_NEW_PREREGISTERED_RETURN_EXPERIMENT",
         "cards": ["F01", "F02", "F03"], "old_primary_selection": "各相关用途原MAIN/20万/STRESS/lag1主行；初筛为完整2017—2025两方向。不筛最好账户。",
         "authority": "允许新增隔离实验代码，保留原冻结策略。",
         "operations": "来源口径/实现/旧裁决摘录；官方JSON20行和19相邻来源记录会计差额；首行缺先前源保留未知。",
         "new_fields_admitted": 0, "new_configurations": 0, "new_fits": 0, "new_return_labels": 0,
         "new_accounts": 0, "network_requests": 0, "history_role": "DEVELOPMENT_CALIBRATION",
         "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False})
    save("cases.json", cases())
    save("saved_old_primary_account_quotes.json", old_account_quotes())
    save("source_facts.json", margin_source_facts())
    save("implementation_evidence.json", implementation_evidence())
    counter = repayment_counterexample()
    save("repayment_identity_rows.json", counter)
    csv_save("repayment_identity_rows.csv", counter)
    save("summary.json", {"at": now(), "study_id": STUDY, "status": STATUS,
         "technical_decision": "TECH.R108", "source_count": len(paths), "completed_prior_routes": 3,
         "saved_old_primary_account_rows": 6, "source_tables_checked": 2,
         "official_sse_january_rows": 20, "adjacent_source_identity_comparisons": 19,
         "maximum_absolute_residual_date": "2026-01-22", "implied_minus_reported_cny": "-168023619",
         "residual_cause": "NOT_IDENTIFIED；官方偿还定义已含权益调整，不擅自把额外差额解释为该项或强平。",
         "f03_actual_clock": "先前两年内最多252合格日、最少126；原卡严格252完整验证未建立。",
         "new_field_bound": False, "new_field_admitted": False, "current_member_support": "NOT_COMPUTED",
         "new_strategy_configurations": 0, "new_field_ols": 0, "new_model_fits": 0,
         "new_return_labels": 0, "new_predictions": 0, "new_accounts": 0, "new_collection": 0,
         "net_cagr": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
         "independent_validation": "NOT_ESTABLISHED", "overall_overfit_removed": False,
         "goal_achieved": False, "goal_turn_classification": "progress", "blocked_audit_count": 0})
    with (OUT / "研究结论与下一步.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report())
    print("三项融资旧用途/来源核对已保存；未新增模型或账户。")


def report() -> str:
    return """# F01—F03 融资旧用途与来源有限核对（TECH.R108）

三卡当前不准入。已有余额代理、净收缩缓和和融资残差用途都实际研究过；完整原卡的直报流量版本和当前全部成员来源未建立。现有失败不能通过改窗口或源替换重开，也不能据此声称融资信息永远无效。本轮没有新收益率或夏普率结果。

| 卡片/相关旧用途 | 保存结果（20万元压力，旧原年化242） | 接续判断 |
| --- | --- | --- |
| F01：T18日更A01/E01/F01组合 | CAGR −0.043612%，夏普 −0.007015，21闭合周期 | 多组件目标失败保留，不等于F01独立无效；原季度T18仍NOT_RUN |
| F01/F05相关：拥挤FULL减仓 | CAGR −1.188573%，夏普 −0.636625，130闭合周期 | 固定旧覆盖层未通过，不改分位/窗口/恢复规则 |
| F02：T03融资收缩减速联合修复 | 主期零交易、CAGR 0、夏普未定义 | 固定种子无机会，不能把夏普/胜率写0或放宽条件造交易 |
| F06相关：T05高隐含偿还联合修复 | CAGR −1.001076%，夏普 −1.058448，18闭合周期 | 是相关隐含偿还用途，不是F02独立收益；原失败保持 |
| F03 HIGH：高残差做多筛选 | CAGR −0.523029%，夏普 −0.234242，112闭合周期 | 旧初筛无继续资格 |
| F03 LOW：低残差做多筛选 | CAGR −0.827645%，夏普 −0.331546，105闭合周期 | 同为做多筛选，没有空头检验 |

F03的协议概述为252个先前合格日。实际源码`prior_residual`只取先前两年内最近最多252个合格日，达到126个即计算；再将整列后移一交易日。这是已存在的初筛差异，原卡完整验证false/NOT_RUN保持，不把最少126改252后重新筛方向。

两份保存融资表均2674日（2015-01-05至2025-12-31），沪深汇总，无直接偿还列。原11深市缺项在收益运行前修复，原2663行买入/余额未改，策略及参数改动0，补缺前新收益结果0；不把合理补源描述成事后调参。沪市主要官方、深市有第三方交叉核对及官方补缺；首次发布/修订逐日凭证仍NOT_ESTABLISHED。T+1早上发布的规则和算法滞后不是历史真实首版证明。深市亿元两位小数约100万元精度，未形成精确双交易所实际资金流。

按官方定义，偿还包括现金还款、卖券还款、强平和正负权益调整；汇总还含已调出融资资格标的余额，明细仅含现有标的。买入减余额变化是隐含偿还。旧`market_financing_balance_identity_residual`列实际是余额变化减买入，不是使用第三个直接报告字段核恒等。不能命名为主力净流入、纯卖压或CSI300特有需求。

已保存官方沪市2026年1月JSON的20个统计日提供直接偿还。以十进制精确核19对相邻来源记录，2026-01-22的“隐含偿还−直报偿还”为−168,023,619元，与原必要核对一致。首行缺本源先前记录保持未知。差额原因NOT_IDENTIFIED；官方定义已含权益调整，不能凭猜测归因。沪市一个月不是当前沪深全段来源，不计算F5/策略标签或事件收益。

本轮只完成三路旧用途/源事实核对，六旧主账户原行摘录及一次保存核对。1507原状态/115成熟月完整支持NOT_COMPUTED。新字段、OLS/模型、配置、标签、预测、账户、网络采集均0，当前净CAGR/夏普NOT_COMPUTED，独立验证NOT_ESTABLISHED。旧64/56/112/88账户属于不同原研究，未重算、相加成当前试验或与原252口径A排名。没有新增可执行策略，完整目标未达。

下一步合并检查剩余F04—F06及日频ETF份额G01—G04的既有用途/源合同；高偿还及拥挤已有旧用途直接复用，股票融资覆盖需历史资格/范围，份额需拆分与真实公布时钟。同指数体系不拿今天基金池回填，NAV×份额变化仅价值代理。先以有限保存来源决定能否进入固定比较，缺资格就停止；不先构造方向、窗口或供应商搜索。原E03按已冻结日历保持，未来真实观察为0。

直接事实：[三项路由](cases.json)、[原六主账户行](saved_old_primary_account_quotes.json)、[来源口径](source_facts.json)、[F03实际源码](implementation_evidence.json)、[官方源差额表](repayment_identity_rows.csv)、[保存核对](saved_output_verification_receipt.json)。官方定义来源：[上海证券交易所融资融券汇总说明](https://www.sse.com.cn/market/othersdata/margin/sum/)，本轮依据已保存原页，没有重新请求或证明历史首版。
"""


def verify() -> None:
    for row in read(str((OUT / "evidence_manifest.json").relative_to(ROOT))):
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"冻结源变化：{row['path']}")
    checks = {"cases.json": cases(), "saved_old_primary_account_quotes.json": old_account_quotes(),
              "source_facts.json": margin_source_facts(), "implementation_evidence.json": implementation_evidence(),
              "repayment_identity_rows.json": repayment_counterexample()}
    for name, expected in checks.items():
        if json.loads((OUT / name).read_text(encoding="utf-8-sig")) != expected:
            raise ValueError(f"保存摘录或来源核对不符：{name}")
    with (OUT / "repayment_identity_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    normalized = [{key: "" if value is None else str(value) for key, value in row.items()}
                  for row in checks["repayment_identity_rows.json"]]
    if csv_rows != normalized:
        raise ValueError("保存差额CSV与JSON不符")
    save("saved_output_verification_receipt.json", {"at": now(),
         "status": "PASS_SAVED_FINANCING_PRIOR_AND_SOURCE_FACT_VERIFICATION",
         "checked_json_outputs": list(checks), "checked_source_identity_csv_rows": 20,
         "source_identity_comparisons": 19, "old_account_quotes_checked": 6,
         "new_ols_or_model_calls": 0, "new_labels_or_accounts": 0, "network_requests": 0,
         "limit": "保存摘录/原官方源差额一致，不证明全部历史首版、因果、当前成员准入或金融改进。"})
    print("六旧账户、两融资表、20官方源行及F03实现保存核对通过；新拟合/账户0。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有限融资旧用途与来源保存核对")
    parser.add_argument("operation", choices=("run", "verify"))
    arguments = parser.parse_args()
    run() if arguments.operation == "run" else verify()
