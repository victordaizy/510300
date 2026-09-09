"""交付第29、30轮中文规则和结果，登记已停止的无改善路线。"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300权重与退出效率检验_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def metric(result, key, cost="BASE", earlier=False):
    return next(m for m in result["earlier_diagnostics" if earlier else "all_metrics"] if m["model"] == key and m["cost"] == cost)


def number(value):
    return "未计算" if value is None else f"{value:.3f}"


def pct(value):
    return "未计算" if value is None else f"{value:.2%}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r29 = read("reports/research/510300_simple_pair_risk_v1/result.json")
    r30 = read("reports/research/510300_simple_intraday_protection_v1/result.json")
    old = read("reports/backtest/round5_three_strategy_comparison.json")["summaries"]["t_only"]["intraday_t"]
    diagnostic = {"recorded_at": now(), "source": "reports/backtest/round5_three_strategy_comparison.json",
                  "source_sha256": digest(ROOT / "reports/backtest/round5_three_strategy_comparison.json"),
                  "original_t_statistics": old,
                  "same_trades_zero_commission_pnl": old["raw_gross_pnl_cny"] - old["slippage_and_tick_cost_cny"],
                  "same_trades_zero_all_friction_pnl": old["raw_gross_pnl_cny"],
                  "decision": "STOP_SIZE_AND_MINIMUM_FEE_ONLY_REPAIR_OF_OLD_T_PATH",
                  "scope": "旧2万元账户已发生的研究交易路径；不是20万元重新模拟，也不否定所有日内机制。",
                  "new_accounts_generated": 0, "goal_achieved": False, "position_impact": 0}
    write_json(ROOT / "reports/research/510300_old_inventory_cost_diagnostic_20260907/result.json", diagnostic)

    files = []
    for prefix, study in [("权重", "510300_simple_pair_risk_v1"), ("退出", "510300_simple_intraday_protection_v1")]:
        for source, name in [("metrics.csv", "全部主评价"), ("earlier_diagnostics.csv", "全部较早诊断"),
                             ("yearly_metrics.csv", "逐年表现"), ("era_metrics.csv", "分阶段表现")]:
            target = OUT / f"{prefix}_{name}.csv"
            target.write_bytes((ROOT / "reports/research" / study / source).read_bytes())
            files.append(target.name)
    target = OUT / "盘中保护_交易及受阻次数.csv"
    target.write_bytes((ROOT / "reports/research/510300_simple_intraday_protection_v1/execution_statistics.csv").read_bytes())
    files.append(target.name)

    lines = ["# 510300：权重与退出改动的结果，以及下一步停止投入什么", "", f"更新时间：{now()}。", "",
             "**完整账户净夏普1.2仍未达到。前瞻每股收益资料补齐继续暂停。** 本次完成第29、30轮：14个新增策略设置、46个主评价记录，其中34个新账户、12个复用对照；另46个较早历史诊断记录，其中40个新账户、6个复用对照。策略数量与成交情景、费用情景、对照账户分别计数。", "",
             "## 给管理者看的结论", "",
             "1. 继续补资料不再是主线。现有前瞻每股收益方法来源修正后没有出现足够的局部改善，因此继续暂停研报、盈利、估值和申购来源补齐；直接用已经持有的免费行情做完整账户比较。", 
             "2. 调整两个信号的权重、压低高波动时期仓位，没有改善主评价。十二个设置中，最高仍是原来的两个信号各半、不另外缩减：夏普0.534。预定的15%波动缩减主方案为0.514。较早历史有不同排序，不能按年份事后换成各年的赢家。", 
             "3. 提前到盘中止损也没有解决问题。趋势切换夏普从0.481降至0.368，均值回升从0.370降至0.176。日内强弱从0.465提高至0.542，但使用当日低价的不利成交情景后只有0.449；较早时期原规则0.500，保护价退出0.482，低价退出0.178。改善缺乏跨时期和成交情景的一致性。", 
             "4. 旧日内交易不能只靠减佣金修好。旧固定底仓交易在扣费前已亏887.90元；扣除佣金和滑点后亏11550.40元。即使把同一批交易的佣金全部免去，仍亏5480.40元。应停止只提高金额或降低最低佣金的修补方案。", "",
             "这两轮的具体决定：停止本组邻近权重、波动缩减和保护止损幅度的细调；保留原结果，转向不同的决策方法。最新证据没有支持可以交付使用的夏普1.2策略。历史最高点估计1.118仍来自此前仅三次完整反弹交易，原参数在较早时期亏损，不能作为已达标结果。", "",
             "## 相同口径怎样比较", "",
             "主评价为2020年1月2日至2026年8月14日开盘，1604个账户日。较早诊断为2015年1月5日至2019年12月31日开盘，1219个账户日。两段分别从20万元现金开始；较早历史也已被研究过，不能称为独立验证或拼成一条实际连续净值。", "",
             "全部记录保留现金、持仓、分红权益与到账、整手、方向性涨跌停、买入次日才可卖、交易费用、未成交和终点退出。现金及无风险收益按零，年化242个交易日。基础费用为单边佣金万分之二、最低5元、滑点万分之五；压力费用为佣金万分之四、最低5元、滑点千分之一。夏普使用每日净收益的算术均值及样本标准差，不能只挑持仓日计算。", "",
             "下表年化收益为复合年化收益。较早诊断、费用压力和盘中成交情景是不同检验，不把它们当作新的独立策略。", "",
             "## 第29轮：十二个权重与风险设置的完整结果", "",
             "|设置|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|主评价基础年化|主评价基础最大回撤|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in r29["all_metrics"]:
        if row["cost"] != "BASE" or row["model"] == "BUY_HOLD":
            continue
        key = row["model"]
        values = [number(metric(r29, key, cost, earlier)["net_sharpe"]) for earlier, cost in [(False, "BASE"), (False, "STRESS"), (True, "BASE"), (True, "STRESS")]]
        lines.append("|" + row["name"] + "|" + "|".join(values + [pct(row["annualized_return"]), pct(row["max_drawdown"])]) + "|")
    lines += ["", "## 第30轮：每个原入场信号的三种退出方式", "",
              "|信号与退出方式|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|主评价基础年化|主评价基础最大回撤|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    assumption_names = {"TRIGGER": "盘中按保护价条件成交", "DAY_LOW": "盘中按当日低价不利成交", "CLOSE_NEXT_OPEN": "原收盘判断、次日开盘退出"}
    for row in r30["all_metrics"]:
        if row["cost"] != "BASE" or row["model"] == "BUY_HOLD":
            continue
        key = row["model"]
        values = [number(metric(r30, key, cost, earlier)["net_sharpe"]) for earlier, cost in [(False, "BASE"), (False, "STRESS"), (True, "BASE"), (True, "STRESS")]]
        label = row["name"] + "；" + assumption_names[row["assumption"]]
        lines.append("|" + label + "|" + "|".join(values + [pct(row["annualized_return"]), pct(row["max_drawdown"])]) + "|")
    lines += ["", "盘中保护的最高主评价点估计来自日内强弱信号，共18次完整买卖；保护价情景中7次盘中卖出、1次开盘保护卖出。更高的夏普并非日内真实成交的证明。", "",
              "## 旧底仓日内交易的成本诊断", "",
              "核查依据是已保存的旧三策略比较结果，共607次底仓日内回合，佣金6070.00元、滑点及价格单位损失4592.50元。扣费前亏损887.90元，三项相加为扣费后亏损11550.40元。", "",
              "将同一已保存交易路径的佣金归零，算术上仍亏5480.40元；佣金、滑点全部归零，仍亏887.90元。这是旧2万元账户的事后成本分解，没有模拟一个新的20万元策略。新的交易金额可能改变成交量、现金和路径，不能简单按十倍外推；本结论只用于停止“仅靠摊薄最低佣金修复旧路径”的方向，不能证明所有日内交易都无效。", "",
              "## 下一步如何提高效率", "",
              "下一条拟检验的机制是：保持少量固定入场条件，用已经结束且在当时可知的历史持仓周期，学习什么时候继续持有、什么时候退出。先核对旧研究是否已覆盖这种以持仓周期为单位的退出决策；若已覆盖且失败，直接跳过。若未覆盖，先登记极小候选集、同时写明进出场，用同样的账户和费用比较，不重新补齐EPS。这里是后续研究方向，尚没有新训练或新结果。", "",
              "允许其他ETF组合计入目标的问题仍待用户回复；当前实际评价范围保持510300与现金。不会把一个近年最高点估计、较短窗口或只持仓期间的夏普代替完整账户目标。", "",
              "## 全部因子、进入、持有、退出与重新进入规则", ""]
    for path in ["docs/510300_SIMPLE_PAIR_RISK_V1.md", "docs/510300_INTRADAY_PROTECTIVE_EXIT_V1.md"]:
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            lines.append("#" + line if line.startswith("#") else line)
        lines.append("")
    lines += ["## 必要验证与普通结果文件", "",
              "新增盘中退出的8项必要测试通过：关闭新增机制后的旧账户一致性、买入当天不可卖、跳空跌破按开盘退出、当天高点不抬升事前保护价、跌停退出受阻后持续请求、除息保护价与分红到账、不同成交情景和终点不读取盘中数据。新账户逐日财富恒等式、完整日期及已完成交易至少间隔一个交易日的检查通过。", "",
              "没有真实盘中排队及成交验证，没有开展安全审计，没有生成GPT数值审阅包。普通结果文件如下：", ""]
    lines += [f"- [{name}]({name})" for name in files]
    document = OUT / "权重与退出改动_全部中文规则和历史结果.md"
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")

    index = read("reports/research/510300_sharpe_1_2_latest_research.json")
    for number_, result, title, folder in [(29, r29, "双信号权重与风险缩减", "510300_simple_pair_risk_v1"),
                                           (30, r30, "已有持仓盘中保护退出", "510300_simple_intraday_protection_v1")]:
        count = result["candidate_configurations"]
        record = {"round": number_, "study": result["study_id"], "title": title, "status": "COMPLETED_TARGET_NOT_MET",
                  "result": f"reports/research/{folder}/result.json", "candidate_configurations": count,
                  "evaluated_candidate_source_runs": count,
                  **{k: result[k] for k in ["evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "post_selected_best_base"]},
                  "primary_base": next(x for x in result["primary"] if x["cost"] == "BASE"),
                  "primary_stress": next(x for x in result["primary"] if x["cost"] == "STRESS")}
        if not any(r["round"] == number_ for r in index["completed_rounds"]):
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += count
        write_json(ROOT / "reports/research" / folder / "acceptance_outcome.json",
                   {"recorded_at": now(), "status": "COMPLETED_TARGET_NOT_MET", "historical_point_target_met": result["historical_point_target_met"],
                    "goal_achieved": False, "decision": "STOP_NEARBY_WEIGHT_RISK_OR_PROTECTION_TUNING", "position_impact": 0})
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [29, 30]]
    index.update({"updated_at": now(), "status": "ROUNDS29_30_COMPLETE_NEW_EXIT_DECISION_RESEARCH_CONTINUES",
                  "latest_completed_round": record, "running_studies": [], "goal_achieved": False,
                  "process_state_note": "第29、30轮及旧底仓成本诊断完成，当前没有这些研究的运行进程。",
                  "count_warning": "完成30轮，287个不同配置或范围、299个已评价来源版本、734个主评价记录。登记来源版本304，包含5个旧未运行绑定。第29、30轮另46个较早诊断记录不计入734。",
                  "next_work": ["停止本组邻近权重、波动缩减、止损幅度和旧底仓仅摊薄佣金的修补。",
                                "先核对旧研究是否覆盖按已结束持仓周期学习持有或退出；未覆盖再登记极小候选集，固定入场、仅用当时成熟历史学习退出。",
                                "全部EPS及研报来源补齐保持暂停；其他ETF验收范围问题保持待回复，不重复询问。"],
                  "checks": "第30轮8项必要机制测试通过；两轮新账户已核对财富恒等式、完整日期，盘中退出保留成交假设及T+1状态。",
                  "old_inventory_cost_diagnostic": "reports/research/510300_old_inventory_cost_diagnostic_20260907/result.json",
                  "prepare_gpt_numerical_review_package": False})
    delivery = {"created_at": now(), "type": "PAIR_RISK_AND_PROTECTIVE_EXIT_CHINESE_RESULTS", "rounds": [29, 30],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    if not any(d.get("type") == delivery["type"] for d in index["deliveries"]):
        index["deliveries"].append(delivery)
    require(index["evaluated_configurations_in_this_resumption"] == 287, "完成配置计数未对齐")
    require(index["evaluation_accounts_in_this_resumption"] == 734, "主评价记录计数未对齐")
    write_json(INDEX, index)
    write_json(OUT / "delivery_receipt.json", {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_SAVED_RESULTS",
               "document": str(document), "new_candidate_configurations": 14, "main_evaluation_records": 46,
               "new_main_accounts": 34, "reused_main_controls": 12, "earlier_diagnostic_records": 46,
               "new_earlier_accounts": 40, "reused_earlier_controls": 6, "necessary_tests_passed": 8,
               "files": [{"path": p.name, "sha256": digest(p)} for p in [document] + [OUT / f for f in files]],
               "goal_achieved": False, "new_gpt_review_archive_created": False, "security_audit_performed": False})
    print(json.dumps({"中文报告": str(document), "已完成轮数": len(index["completed_rounds"]), "主评价记录": 734}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
