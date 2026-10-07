"""把1.3新目标下的旧模型点值、跨期差异和原裁决并列呈现。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_saved_candidates_sharpe13_v1"
OLD = ROOT / "reports/research/510300_sparse_opportunity_mandate_v1"


def write(name, value):
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def main():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    old = json.loads((OLD / "result.json").read_text(encoding="utf-8"))
    comparisons = pd.read_csv(OUT / "deduplicated_candidate_comparison.csv")
    paired = comparisons[comparisons.main_both_costs_numerical_pass].copy()
    old_models = {r["model"] for r in old["candidate_results"] if r["main_both_costs_numerical_pass"]}
    paired["new_main_both_cost_pass_after_target_revision"] = ~paired.model.isin(old_models)
    inputs = OUT / "inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copy2(OLD / "result.json", inputs / "previous_1_5_saved_screen.json")
    paired.to_csv(OUT / "main_pass_14_with_target_change.csv", index=False, encoding="utf-8-sig")
    labels = {
        "CORE_AUXILIARY_DRAWDOWN_GATE": "回撤时撤去辅助仓位",
        "DRAWDOWN_GATE_TWO_CLOSE_RECOVERY": "回撤恢复后延迟恢复辅助仓位",
        "DUAL_CONFIRMED_RUNS_AUXILIARY": "双方向确认辅助仓位",
        "EITHER_CONFIRMED_RUNS_AUXILIARY": "任一方向确认辅助仓位",
        "EPISODE_START_CONFIRMED_AUXILIARY": "周期起点确认辅助仓位",
        "EPISODE_WAIT_CONFIRMED_AUXILIARY": "等待确认后加入辅助仓位",
        "EXPOSURE_EXPANSION_150": "原预算扩大至1.5倍并封顶",
        "LAG_CONFIRMED_RUNS_AUXILIARY": "相关方向确认辅助仓位",
        "RUNS_CLOSED_CYCLE_BUDGET": "已结束周期风险预算",
        "RUNS_COVARIANCE_BUDGET": "月度协方差风险预算",
        "RUNS_OPPORTUNITY_CAPPED_SUM": "两类机会相加后封顶",
        "RUNS_OPPORTUNITY_MAX": "两类机会取较大仓位",
        "RUNS_REFERENCE_BLEND": "固定各半参考组合",
        "SIGN_CONFIRMED_RUNS_AUXILIARY": "上涨方向确认辅助仓位",
    }
    table = ["| 原模型 | 本次新过线 | 主时期压力净夏普 | 较早时期压力净夏普 | 主时期复合年化 | 主时期最大回撤幅度 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in paired.sort_values("model").itertuples():
        table.append(f"| {labels[row.model]} | {'是' if row.new_main_both_cost_pass_after_target_revision else '旧1.5已过线'} | {row.main_stress_sharpe:.4f} | {row.earlier_stress_sharpe:.4f} | {row.main_stress_cagr:.2%} | {abs(row.main_stress_drawdown):.2%} |")
    change = {
        "previous_target": 1.5, "current_target": 1.3,
        "previous_main_both_cost_pass_models": len(old_models),
        "current_main_both_cost_pass_models": len(paired),
        "new_main_both_cost_pass_models": int(paired.new_main_both_cost_pass_after_target_revision.sum()),
        "earlier_period_is_preexisting_diagnostic_not_new_user_requirement": True,
        "earlier_stress_sharpe_min": float(paired.earlier_stress_sharpe.min()),
        "earlier_stress_sharpe_max": float(paired.earlier_stress_sharpe.max()),
        "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0,
        "goal_achieved": False, "orders_authorized": False,
    }
    (OUT / "target_change_result.json").write_text(json.dumps(change, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write("研究结论.md", f"""
# 510300：净夏普1.3下的早期训练档案重评

**新目标确实增加了历史点值过线的候选。** 对214份一级研究指标表、2710行保存记录按用户最新的净夏普1.3及最大回撤10%重新计算判定，14个原模型在2020-01-02至2026-08-14主要历史期的基础和压力成本都满足这两项数值条件，比1.5门槛增加8个。这些策略主时期平均每年约4.2—4.7个持仓周期，说明稀疏参与与数值目标在历史上可以同时成立。但原有2015—2019对照期的压力成本净夏普降至0.823—1.089，独立验证仍未建立；不能据此认定已经找到高置信度、可持续的机会。

## 当前要求

仅交易510300与现金，20万元完整账户，净夏普至少1.3、最大回撤幅度不超过10%。空仓日保留；一年四五次是可接受频率，不是配额。用户要求不用采集，当前配置已据此暂停采集并使用已有本地数据。新目标没有要求年化收益必须10%，因此本次不使用旧研究的年化10%门槛来淘汰候选。

## 14个主时期双成本点值过线的模型

下表全部为20万元、压力成本保存结果，按模型标识排序，完整列出14项。主时期2020-01-02至2026-08-14；较早时期2015-01-05至2019-12-31。两段沿用各自原账户，均包含全部空仓日，未拼接为一个连续账户。

{chr(10).join(table)}

“原预算扩大至1.5倍并封顶”是原模型名称，原规则将总仓位封顶100%，不是授权杠杆交易。表内1.4998对应保存值1.4997866，旧1.5门槛下没有将它四舍五入成通过。

这里明确区分两件事：**14个模型的主时期确实满足当前数值要求；其稳定性与独立证据仍不足。** 较早时期是原研究已保存的对照，不是临时增加“每个任意子区间都必须超过1.3”的用户要求，也不宣称较早时期略低于1.3就必然不可用。需要面对的是同类规则经过大量反复搜索、主时期较好而另一整段历史系统性下降的证据。

## 频率、风险与收益

14项主时期包含28—31个从空仓开始的保存持仓周期，包含研究期末强制平仓周期；按1604交易日和242日年化，约4.22—4.68次/年。收盘时点空仓比例72.38%—75.69%，不是精确的盘中空仓时长。压力成本主时期最大回撤幅度1.56%—4.87%，复合年化收益3.51%—8.42%。这些数字只描述历史记录。

多个模型共享核心规则、辅助信号和实际交易，部分较早时期结果完全相同，因此14个名称不是14份相互独立的成功证据。没有把多次仓位调整当成多个独立机会，也没有恢复已终止的SELECTED_MIX_BAND10_SIMPLE2。

## 完整筛查口径

共214个文件、2710行，包含重复引用的对照及两档成本。128行在保存数值上过线，按模型、成本和指标去重后36条记录、22个模型至少一档成本过线；其中14个主要区间两档都过线。22个模型均已定位到声明其为候选的原研究，保留原规则、原裁决、较早时期、逐年指标和完整保存周期。

本次只覆盖reports/research/510300_*/metrics.csv这一命名范围，不宣称穷尽全仓库。上轮近期7项研究使用另一种目录结构，原结论仍成立：那7项没有当前数值达标策略；本轮14项属于更早的训练档案，二者不矛盾。

## 结论对下一步的影响

降低目标已经应用到近期账户和早期训练档案，未遗漏这8个因门槛变化而新增的主时期过线模型。后续研究无需再以凑交易次数或单纯降低门槛来解释困难。尚未解决的是可重复的事前信息优势和跨期证据；不把重新筛出的旧高分记录重新称为未见数据验证。

本次新增拟合0、新回测账户0、新下载0。原研究的关闭裁决保留为历史记录，不冒充当前数值判定。当前目标仍未证明完成，候选没有自动转为当前买入信号。

## 可以复算什么

包内直接来源足够重新筛查全部2710行、定位原模型、计算目标变化、检查保存周期利润总和与原累计利润的一致性。这里没有重跑原始行情到策略账户的整个引擎，也没有新增独立前向数据或完成外部GPT审阅。
""")
    write("00_README_FIRST.md", """
# 交付导航

先读研究结论.md，再看main_pass_14_with_target_change.csv和target_change_result.json。完整22模型对照为deduplicated_candidate_comparison.csv；所有2710行和214个来源分别在all_saved_metric_rows.csv、metric_source_inventory.csv。

source_snapshot保留本轮实际使用的原配置、指标、裁决、较早时期和保存周期，身份见source_snapshot_manifest.json。inputs/previous_1_5_saved_screen.json用于6至14的目标变化比较。用户当前1.3指令快照见source_snapshot/config/510300_existing_data_training_mandate_v1.json。

可用Python运行source_snapshot/scripts/review_510300_saved_candidates_sharpe13_v1.py，从包内来源生成同一筛查结果。该入口只重算保存表格，不拟合模型、运行策略账户或下载数据。输出位于source_snapshot/reports/research/510300_saved_candidates_sharpe13_v1；若该目录已经存在完成结果，入口会拒绝覆盖。

FILE_INDEX.csv覆盖ZIP内全部其他成员。根目录delivery_receipt.json在ZIP外生成，记录新解压目录复算与压缩包身份。finish脚本是原工作区报告生成工具，不是便携入口。
""")
    write("用户需求.md", """
# 用户最新要求

只操作510300，20万元，最大回撤10%，寻找低频且有充分事前依据的机会，其他时间空仓。用户明确说“请加速干”“不用那么严格，直接进入训练”“不用采集”，随后将目标改为“夏普1.3就行”。

本次动作是按最新1.3目标重判此前已完成的训练档案，避免仅检查最近七项而遗漏更早的点值过线结果。没有重复训练已完成模型，没有新增采集，没有把原1.5结果覆盖为1.3。
""")
    write("GPT审阅提示词.md", """
请依据包内数据审阅本次目标变更后的历史结果重评，重点核对：

1. 用户目标已从净夏普1.5改为1.3，资本20万元、最大回撤10%不变。14个主时期双成本点值过线、其中8个新增是否算对？是否错误沿用年化10%旧要求？
2. 是否清楚区分数值条件满足、较早对照明显下降、原裁决和独立证据缺失？不要额外假设用户要求每个子区间夏普都必须1.3。
3. 对2710条保存记录的去重、候选原研究定位、22至14的成本配对，以及周期利润恒等式是否可复算？
4. 是否把共有规则和相同路径误当独立证据？机会次数是否误把调仓成交笔数当周期？
5. 本包的验证范围是否只是保存结果重算，是否有把它冒充完整市场到账户重建、独立验证或外部审阅？

请给出问题位置、影响和下一步最有信息量的动作。不要建议删除空仓日、按事后最好区间裁剪样本或恢复已终止策略来达到目标。
""")
    write("EXCLUSIONS.md", """
# 包的范围

此包自含本轮保存指标筛查的直接输入，不含上游每个策略的全部原始行情、模型训练文件、订单级逐日账户或本机虚拟环境。可以复算筛查和周期汇总，不足以完整从市场数据重建全部模型。214份指标表不是全仓库所有历史研究的穷尽集合。

不包括新解压验证目录、缓存或本轮根交付回执；这些不参与科学结论。原关闭裁决、已使用历史及缺少独立验证均保留。没有上传、外部GPT审阅、当前交易信号或真实订单。
""")
    print(json.dumps(change, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
