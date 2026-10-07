"""保存有限选择诊断的结论与接续事实，不重新计算分割或交易账户。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.point_comparable_selection_v1 import OUT, digest, now, require, write_json

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD_KEYS = ["forward_protocol", "forward_registry", "new_prospective_observations",
                "earliest_future_exchange_session", "registered_candidate_intents", "new_prospective_completed_points",
                "next_new_close_eligible_at", "current_validated_candidates", "forward_account_comparison_protocol",
                "latest_forward_account_check", "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation"]


def main():
    require(not (OUT / "fact_recording_receipt.json").exists(), "当前诊断已归档，不重复更新事实。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    inventory = pd.read_parquet(OUT / "results/候选准入与完整重复身份.parquet")
    selections = pd.read_parquet(OUT / "results/逐候选入选与其余半样本表现.parquet")
    crossings = pd.read_parquet(OUT / "results/实际周期跨块依赖.parquet")
    checks = []
    for scenario in result["scenarios"]:
        period, cost = scenario["period"], scenario["cost"]
        rows = pd.read_parquet(OUT / "results" / f"{period}_{cost}_全部选择分割.parquet")
        require(rows.split.nunique() == 12870, "保存分割数量不符。")
        denominator = rows.split.nunique()
        for name, column in [("pbo_le_zero", "pbo_le_zero"), ("strictly_below_median_fraction", "strictly_below_median"),
                             ("median_mass_fraction", "at_median"), ("selected_oos_negative_fraction", "oos_negative_score")]:
            actual = float(np.sum(rows.selection_weight * rows[column]) / denominator)
            require(abs(actual - scenario[name]) < 1e-14, "保存选择比例不一致：" + name)
        np.testing.assert_allclose(rows.groupby("split").selection_weight.sum(), 1., atol=1e-14, rtol=0)
        require(abs(scenario["pbo_le_zero"] - scenario["strictly_below_median_fraction"] - scenario["median_mass_fraction"]) < 1e-14,
                "中位边界与严格低于中位未还原总比例。")
        checks.append({"period": period, "cost": cost, "saved_split_count": denominator,
                       "selection_weights_and_rates": "EXACT_SAVED_RECOMPUTATION", "new_split_recalculations": 0})
    write_json(OUT / "saved_result_verification.json", {"at": now(), "checks": checks,
               "new_accounts": 0, "new_model_fits": 0, "new_cscv_runs": 0}, exclusive=True)

    def fraction(period, candidate):
        row = selections.loc[selections.period.eq(period) & selections.cost.eq("STRESS") & selections.winner.eq(candidate)]
        return float(row.selection_fraction.iloc[0]) if len(row) else 0.

    early = next(r for r in result["scenarios"] if r["period"] == "2015_2019" and r["cost"] == "STRESS")
    recent = next(r for r in result["scenarios"] if r["period"] == "2020_2026" and r["cost"] == "STRESS")
    headline = (f"2026-10-04技术线TECH.R155—R156完成有限可比账户选择诊断：38原账户、4真实日历矩阵、6必要统计测试，"
                f"16块/每场景12870对称分割，0新账户/模型/训练标签。较早/近期压力有限PBO分别"
                f"{early['pbo_le_zero']:.2%}/{recent['pbo_le_zero']:.2%}，严格低于中位"
                f"{early['strictly_below_median_fraction']:.2%}/{recent['strictly_below_median_fraction']:.2%}，"
                f"入选者其余半样本净均值负比例{early['selected_oos_negative_fraction']:.2%}/{recent['selected_oos_negative_fraction']:.2%}。"
                f"原A入选份额{fraction('2015_2019', 'A_SAVED_WEIGHT'):.2%}/{fraction('2020_2026', 'A_SAVED_WEIGHT'):.2%}；"
                "较早6/近期11排序身份，非全项目搜索宇宙，不据此切换或恢复旧失败。"
                "全局DSR/PBO仍NOT_COMPUTED、独立及去过拟合未成立；最新实际金融裁决仍R154，收益夏普未提高、目标active。")

    lines = ["# 技术线有限选择稳定性：结果与下一步", "", headline, "",
             "本轮回答：在目前能直接比较的完整技术账户里，某些历史时间块的最优方案，在其余历史时间块是否仍有相对优势。"
             "这是已有净日收益的选择诊断，不是新策略回测、未来收益概率或全项目过拟合概率。", "",
             "共同口径：20万元、原50%股票上限及ES/跳空/回撤预算、现金0、252日、原基础与压力成本。"
             "2015—2019和2020—2026-09-30分别启动，1219/1636真实交易日完整保留；不拼接成新账户。", "",
             "| 时期 | 成本 | 排序身份 | 中位或以下：有限PBO | 严格低于中位 | 正好中位 | 入选者其余半样本净均值负 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in result["scenarios"]:
        lines.append(f"| {r['period']} | {r['cost']} | {len(r['candidates'])} | {r['pbo_le_zero']:.2%} | "
                     f"{r['strictly_below_median_fraction']:.2%} | {r['median_mass_fraction']:.2%} | {r['selected_oos_negative_fraction']:.2%} |")
    lines.extend(["", "**近期有限候选内部的相对排名较稳定，但尚未建立全局独立优势。** 近期压力4.71%中，"
                  "只有0.34%严格低于中位，其余4.37%是正好处于中位；不能把所有中位情况都解释成明确失败。"
                  "较早压力约29.04%的入选者在其余半样本净均值为负，说明较早开发表现仍有明显风险。", "",
                  "| 压力成本时期 | A保留权重入选份额 | B保留权重入选份额 | 其余原固定身份 |",
                  "|---|---:|---:|---:|"])
    for period in ["2015_2019", "2020_2026"]:
        a, b = fraction(period, "A_SAVED_WEIGHT"), fraction(period, "B_SAVED_WEIGHT")
        lines.append(f"| {period} | {a:.2%} | {b:.2%} | {1-a-b:.2%} |")
    lines.extend(["", "这些是分割中的统计选择份额，不是资金仓位或可以执行的历史切换策略。"
                  "近期A相对较强、较早主要选中B，说明固定A不是两个时期中都最优；"
                  "时期结果和新分割排名不能作为按年份选A/B、调混合权重或恢复已拒绝方案的交易规则。", "",
                  "## 范围和方法", "",
                  "有限范围为六项技术线研究的11个既存身份，包括A/B的不同仓位表达、NR7、缩量回调和方向确认的固定版本。"
                  "较早3个身份没有同日历原账户，保持未提供而不补跑；缩量回调单独账户全零，单列不排序；"
                  "缩量回调组合与原A在两档费用下全路径完全一致，归并一个排序身份。所有原身份和失败均保留在原始矩阵及清单。", "",
                  "每个时期划为16个连续近等长块，最多差1日，不裁掉末端数据；枚举12870种8块对8块分割。"
                  "半样本用真实净日收益的均值/样本标准差×sqrt252排序，完全空仓全零时排序分数取0；"
                  "并列最优平分选择权重，另一半使用平均名次。名次除以N+1，低于或等于0.5定义有限PBO，"
                  "严格低于和中位边界分别报告。这个比例反映本矩阵和本分割规则的排名变化，不能代替实际完成交易的pB。", "",
                  "方法参考[The Probability of Backtest Overfitting原论文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。"
                  "原文讨论从若干历史配置中选最优时的排名失稳，并指出遗漏真实试验会削弱诊断。"
                  "本项目的完整可比搜索宇宙仍未建立，故这里明确只报告有限范围；全局PBO/DSR保持未计算。", "",
                  "## 不能据此宣称去除过拟合", "",
                  "51,480个场景分割重复使用同一批历史，不是51,480个独立样本。既有模型不按分割重新拟合，"
                  "日收益保留原训练和持仓路径；这不是按时间前行的独立验证。两时期排序身份数不同，"
                  "因此PBO大小不能直接作同集合的时期质量比较。", "",
                  "| 原A压力成本时期 | 实际周期（含未完成） | 跨16块边界的周期 | 这些周期覆盖的日历日 | 占日历比例 |",
                  "|---|---:|---:|---:|---:|"])
    for r in crossings.loc[crossings.cost.eq("STRESS") & crossings.candidate.eq("A_SAVED_WEIGHT")].itertuples(index=False):
        lines.append(f"| {r.period} | {r.actual_cycles} | {r.cross_block_cycles} | {r.return_days_in_cross_block_cycles} | "
                     f"{r.fraction_of_calendar_in_cross_block_cycles:.2%} |")
    lines.extend(["", "跨块周期说明两边还共享持有路径；本轮只披露依赖，不删除这些交易重新制造更好结果。"
                  "上述日历占比不是独立样本比例，原A较早23个周期包括1个未完成周期，不能当23次完成交易。", "",
                  "## 对下一步的实际影响", "",
                  "接受的只是有限选择稳定性事实。近期原A在这批候选中较稳定的相对优势，不支持继续随机增加信号；"
                  "较早弱表现仍需解释。下一项研究应先明确新信息或完整机制解决的是新增互补机会还是持有优势衰减。"
                  "A/B之间的上下文差异需要先核对旧路由、调权和风险预算研究；若仍只是相同来源的再组合，不能登记为新机制。"
                  "本轮未指定或准入新数值策略，待跑候选0；原退出MSE/成员门不通用化，原冻结失败保持。", "",
                  "现有上涨图谱、长空仓和持仓路径诊断已经完成，不重复计数。形态学习V3也已完成6账户，"
                  "其压力夏普0.617、价量相对背景夏普区间跨0、利润高度集中，不能再称未运行或恢复参数搜索。"
                  "直接依据为[V3已完成结论](../510300_pattern_daily_state_learning_v3/历史发现_指数价量状态增量.md)。", "",
                  "本轮没有新账户、模型拟合、训练标签或行情采集，没有生成更高收益/夏普。最新实际金融裁决仍TECH.R154；"
                  "原A较早压力净年化1.84%/夏普0.438/净pB0.611、近期3.99%/1.217/1.073保持，"
                  "独立验证及去过拟合未完成，目标active。", "",
                  "直接文件：[唯一协议](protocol.json)、[6必要统计测试](tests_receipt.json)、[实际结果](summary.json)、"
                  "[候选完整身份](results/候选准入与完整重复身份.csv)、[四场景](results/四场景有限选择稳定性.csv)、"
                  "[逐身份入选](results/逐候选入选与其余半样本表现.csv)、[真实周期跨块](results/实际周期跨块依赖.csv)、"
                  "[保存比例核对](saved_result_verification.json)。", ""])
    report = OUT / "研究结果与下一步.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    write_json(OUT / "state_before_recording_R156.json", state, exclusive=True)
    preserved = {key: state.get(key) for key in FORWARD_KEYS}
    state.update({"current_study": protocol["study"], "current_phase": "COMPLETED_RESTRICTED_COMPARABLE_SELECTION_DIAGNOSTIC",
                  "latest_technical_decision": "TECH.R156", "latest_registration_decision": "TECH.R155",
                  "latest_actual_model_decision": "TECH.R154", "latest_model_decision": "TECH.R154",
                  "latest_actual_prediction_model_decision": "TECH.R145", "latest_financial_strategy_decision": "TECH.R154",
                  "latest_selection_diagnostic": "reports/research/510300_point_comparable_selection_v1/summary.json",
                  "latest_result": "reports/research/510300_point_comparable_selection_v1/summary.json",
                  "latest_report": "reports/research/510300_point_comparable_selection_v1/研究结果与下一步.md",
                  "status": "research_active", "goal_status": "active", "goal_achieved": False, "overfitting_removed": False,
                  "goal_turn_classification": "progress", "previous_goal_turn_classification": "progress",
                  "previous_goal_turn_classification_reason": "R153—R154完成8个实际新账户及19个真实持仓重叠日期解释。",
                  "current_goal_turn_classification_reason": "第一次计算有限可比完整账户的对称选择统计，新增早期不稳定与近期局部优势的定量边界。",
                  "blocked_audit_count": 0, "blocking_decision": None, "live_own_process_handle": None, "verified_wait": False,
                  "new_accounts_in_current_phase": 0, "new_accounts_this_continuation": 0,
                  "new_strategy_configurations_this_continuation": 0, "saved_account_controls_replayed_this_continuation": 0,
                  "new_model_training_labels_this_continuation": 0, "new_market_requests_this_continuation": 0,
                  "return_and_sharpe_changed_this_continuation": False, "return_and_sharpe_improved_this_continuation": False,
                  "new_account_return_sharpe": "NOT_GENERATED_BY_STATISTICAL_DIAGNOSTIC",
                  "current_goal_turn_actual_work": {"new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0,
                                                    "new_market_requests": 0, "saved_accounts_read": 38, "return_matrices": 4,
                                                    "necessary_tests_passed": 6, "new_statistical_split_evaluations": 51480},
                  "restricted_selection_pbo": {"status": "COMPUTED_FINITE_SCOPE_NOT_GLOBAL", "source": "reports/research/510300_point_comparable_selection_v1/summary.json",
                                               "earlier_stress": early["pbo_le_zero"], "recent_stress": recent["pbo_le_zero"]},
                  "global_pbo": result["global_pbo"], "global_dsr": result["global_dsr"],
                  "current_admitted_unrun_numeric_candidates": 0,
                  "next_available_action": "明确不同完整技术机制及新增信息用途；A/B上下文提案须先核对旧调权和路由，不按年份或此次分割胜者切换，不恢复已失败策略。独立前瞻保持。",
                  "code_files_added_this_continuation": ["research/point_comparable_selection_v1.py", "tests/test_point_comparable_selection_v1.py",
                                                         "research/finalize_point_comparable_selection_v1.py"],
                  "literature_lookup_this_continuation": {"page_open_attempts": 2, "paper_find_requests": 3, "market_data_requests": 0,
                                                         "scope": "过拟合方法原论文背景，底层HTTP次数未统计"}})
    require({key: state.get(key) for key in FORWARD_KEYS} == preserved, "前瞻状态受到本诊断影响。")
    write_json(STATE, state)
    relative = "../reports/research/510300_point_comparable_selection_v1/研究结果与下一步.md"
    banner = f"> 技术线最新统计诊断：{headline} [完整结果]({relative})。\n\n"
    section = ("\n\n## TECH.R155—R156：有限可比账户选择稳定性\n\n" + headline + "\n\n"
               "**假设**：历史局部最优是否在其余半样本仍有相对优势，能否支持继续围绕原A改进。\n\n"
               "**验证方法**：原38账户只读形成4真实日历矩阵；按两成本完整重复去重、缺账户不补跑，较早6/近期11身份。"
               "6可手算边界测试后冻结89来源，16块全部12870对称分割，IS最优平分并列、OOS平均名次，零半样本排序分数0。"
               "另报严格低于与中位边界、负净均值和实际跨块周期。\n\n"
               "**结果**：较早/近期压力有限PBO21.24%/4.71%，严格低于21.24%/0.34%；"
               "入选者其余半样本净均值负比例29.04%/0%。原A入选5.13%/50.05%，B76.72%/42.57%。"
               "这些不是未来亏损概率或执行策略；新账户、模型和训练标签0，没有收益夏普改善。\n\n"
               "**接受/拒绝原因**：接受有限集合选择稳定性及时期差异事实；拒绝低局部PBO即已去过拟合、"
               "以分割/年份选A/B或恢复失败方案、把51480重组当独立样本。原A跨块周期较早8/23、近期3/32，依赖保持。\n\n"
               "**是否需要重新验证**：固定诊断已完成，不修改块数或候选集合争取更好比例；"
               "全局DSR/PBO仍未计算，独立验证缺失。下一金融实验仍须不同完整机制/信息用途，旧原退出MSE门不通用化，"
               "原前瞻与全部旧失败保持。V3已完成失败，不再当待跑策略。\n\n"
               f"直接依据：[完整报告]({relative})、[唯一协议](../reports/research/510300_point_comparable_selection_v1/protocol.json)、"
               "[实际统计](../reports/research/510300_point_comparable_selection_v1/summary.json)。\n")
    for rel in ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md", "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]:
        path = ROOT / rel
        data = path.read_bytes()
        require("## TECH.R155—R156：有限可比账户选择稳定性".encode("utf-8") not in data, "事实条目已经存在。")
        first, separator, remaining = data.partition(b"\n")
        path.write_bytes(first + separator + b"\n" + banner.encode("utf-8") + remaining + section.encode("utf-8"))
    write_json(OUT / "fact_recording_receipt.json", {"at": now(), "headline": headline, "latest_technical_decision": "TECH.R156",
               "latest_actual_financial_decision": "TECH.R154", "preserved_forward_values": preserved,
               "new_accounts": 0, "restricted_pbo_computed": True, "global_pbo_computed": False,
               "goal_status": "active", "goal_achieved": False, "report_sha256": digest(report)}, exclusive=True)
    print(headline, flush=True)


if __name__ == "__main__":
    main()
