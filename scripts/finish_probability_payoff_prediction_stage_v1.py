"""整理先训练概率与幅度的实际结果，不增加节点或账户。"""
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_probability_payoff_prediction_stage_v1"


def write(name, value):
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUT / "results/分项预测误差.csv")
    comparisons = pd.read_csv(OUT / "results/固定对照比较.csv")
    full = metrics.loc[metrics.segment == "ALL"]

    def row(scope, model, head):
        return full.loc[(full.scope == scope) & (full.model == model) & (full['head'] == head)].iloc[0]

    table = ["| 评价对象 | 评价样本 | 旧幅度模型 | 新直接均值模型 | 历史条件均值 |",
             "|---|---:|---:|---:|---:|"]
    for scope, head, label in [("MONTHLY", "GAIN", "月末盈利幅度"), ("MONTHLY", "LOSS", "月末亏损幅度"),
                               ("NODE", "GAIN", "节点盈利幅度"), ("NODE", "LOSS", "节点亏损幅度")]:
        old, new, simple = [row(scope, model, head) for model in ["ORIGINAL_LOG_IF", "GAMMA_IF", "EMPIRICAL_ALL"]]
        table.append(f"| {label} | {old.n} | {old.rmse*100:.3f} | {new.rmse*100:.3f} | {simple.rmse*100:.3f} |")
    monthly = row("MONTHLY", "GAMMA_IF", "EXPECTATION")
    monthly_simple = row("MONTHLY", "IF_PROB_EMPIRICAL_AMPLITUDES", "EXPECTATION")
    small_increment = 1-monthly.mse/monthly_simple.mse
    node_old, node_new = row("NODE", "ORIGINAL_LOG_IF", "EXPECTATION"), row("NODE", "GAMMA_IF", "EXPECTATION")
    new_loss_vs_old = 1-row("MONTHLY", "GAMMA_IF", "LOSS").mse/row("MONTHLY", "ORIGINAL_LOG_IF", "LOSS").mse
    new_loss_vs_mean = row("MONTHLY", "GAMMA_IF", "LOSS").mse/row("MONTHLY", "EMPIRICAL_ALL", "LOSS").mse-1
    p_month, p_month_mean = row("MONTHLY", "ORIGINAL_LOG_IF", "PROBABILITY"), row("MONTHLY", "EMPIRICAL_ALL", "PROBABILITY")
    p_node, p_node_mean = row("NODE", "ORIGINAL_LOG_IF", "PROBABILITY"), row("NODE", "EMPIRICAL_ALL", "PROBABILITY")
    write("训练结果.md", f"""
# 先训练胜率和盈亏幅度：本轮已完成444次条件均值拟合

研究顺序已按用户要求改为“先训练概率与盈亏幅度，再逐步增加节点”。每完整自然年至少5次保留为最终策略验收条件，不再阻止单个节点和预测模型训练。本轮保留已训练的概率模型，实际新增{summary['new_conditional_mean_fits']}次条件幅度拟合；节点仍是原10个，没有新增账户或采集。结果表明概率部分有有限的历史评分改善，复杂幅度部分尚未稳定胜过简单历史条件均值。

## 本轮实际做了什么

沿用原124个月末训练样本、10个IF节点、10日收益期限、24bp名义往返压力成本代理和原特征。IF只作510300的辅助信息，交易资产范围保持510300与现金。按原时间顺序，标签结束后才可加入下一次训练，标准化只使用当时成熟样本。

概率部分沿用上一轮保存的222份逻辑模型参数（含最终参考），本轮新增概率拟合为0，不重复拟合同一模型冒充进展。幅度部分新增220份逐期模型组合和2份最终组合，每份分别拟合盈利和亏损方向，共444次新拟合。最终输出6个固定模型/对照、660条预测：100个月末原点×6和10个节点×6，不是660个独立事件。

原幅度模型先回归对数，再用一个训练残差还原系数转回平均幅度。新模型使用正值均值回归，直接学习条件平均盈利和条件平均亏损；惩罚固定1、标准化截断固定±3，没有窗口或参数搜索。增加“IF概率配历史条件均值”的幅度去特征对照，用来判断复杂幅度预测是否有独立贡献；本轮不拼接任何交易账户。

## 盈亏幅度的分项结果

下表是收益率误差RMSE，单位为百分点，越小越好。按实际盈利/亏损分组只发生在事后评价，预测时并不知道未来方向。

{chr(10).join(table)}

新模型的月末亏损幅度MSE比旧模型下降{new_loss_vs_old:.2%}，但仍比历史条件均值高{new_loss_vs_mean:.2%}。盈利幅度也未胜过简单均值。两个亏损节点上的误差反而变大；这类小样本不足以说明真实尾部风险已学好。

组合成预期净收益代理后，月末100次RMSE为{monthly.rmse:.4%}，简单“IF概率+历史幅度”对照为{monthly_simple.rmse:.4%}，新幅度模型的MSE只额外改善{small_increment:.3%}。这个很小的差距不足以证明复杂模型有可靠增量。

10个节点上的整体预期收益RMSE由{node_old.rmse:.4%}降到{node_new.rmse:.4%}，MSE下降{1-node_new.mse/node_old.mse:.2%}。但前后半段并不一致：新模型主要改善前5个节点，后5个节点的RMSE略高于原模型；完整分段表没有删除这一反例。只看总体误差会掩盖亏损幅度预测变差的问题。

## 胜率预测单独评价

概率不因本轮幅度训练而改变。月末100次Brier误差为{p_month.mse:.6f}，成熟经验概率为{p_month_mean.mse:.6f}，改善{1-p_month.mse/p_month_mean.mse:.2%}；节点10次分别为{p_node.mse:.6f}与{p_node_mean.mse:.6f}，改善{1-p_node.mse/p_node_mean.mse:.2%}。两类样本分别报告，没有把月末记录算成独立节点。

月末平均预测获利概率{p_month.mean_prediction:.2%}、实际代理获利比例{p_month.mean_actual:.2%}；节点平均预测{p_node.mean_prediction:.2%}、实际{p_node.mean_actual:.2%}。10个节点的事后80%不能直接写成模型事前有80%把握，也不能据此回填提高预测概率。

## 当前研究决定

保留概率评分的有限改善作为探索证据。复杂条件幅度模型尚未证明优于简单基准，本轮不将其升级为已验证预测器，也不据此运行新账户或增加节点。原账户失败和旧策略终止记录保留；单个节点每年不足5次现在只意味着它不能独立完成最终频率要求，不再作为禁止预测训练的理由。

后续仍先检查概率、盈利幅度、亏损幅度各自的误差。节点扩展时每轮加入一种事前定义的节点，分别报告原节点与新增节点的预测质量；最后再检查组合策略每年实际至少5次、完整账户净夏普1.3和最大回撤10%。不能通过挑选事后盈利日期或放宽同一节点阈值凑数量。

## 证据范围

全部标签是既有固定10日净收益代理，不是含最低佣金和账户提前退出的实际周期利润。现有历史此前已经被反复查看，本次方法选择也受旧分项误差启发，所以是历史探索，不是独立前向验证。新节点0、新账户0、新下载0；本轮没有新夏普或回撤结果，也不使用上一轮的账户数值冒充本轮表现。
""")
    decision = {"status": "PREDICTION_STAGE_COMPLETE_AMPLITUDE_INCREMENT_NOT_ESTABLISHED",
                "probability": "已有IF概率较经验概率的Brier点值改善保留，未建立可靠校准或独立证据。",
                "amplitude": "Gamma条件均值较旧幅度模型部分改善，但总体分项尚未胜过历史条件均值。",
                "monthly_expectation_mse_increment_vs_same_probability_empirical_amplitudes": small_increment,
                "future_node_expansion": "后续每次扩展一种预先定义的节点，分别披露旧/新节点预测，不按事后收益挑日期。",
                "annual_frequency_as_training_gate": False, "minimum_annual_cycles_final_strategy": 5,
                "new_amplitude_fits": 444, "new_probability_fits": 0, "new_nodes": 0, "new_accounts": 0,
                "no_historical_parameter_rescue": True, "goal_achieved": False, "orders_authorized": False}
    (OUT / "stage_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write("00_README_FIRST.md", """
# 阅读导航

先读训练结果.md和stage_decision.json。当前优先顺序见authority_update.json：先概率与盈亏幅度，之后逐步增加节点；每年至少5次仍是最终策略要求。

protocol.json及freeze.json固定本轮训练和比较规则。inputs含124个月末样本、10节点、上一轮保存参数及预测、标签复算所需原行情和来源回执。models为本轮新增的444个条件均值拟合结果，results提供660条完整预测、分项误差和固定对照的前后半段比较。

运行code/probability_payoff_prediction_stage_v1.py的verify命令，--root指定解压目录，可检查成熟标签、保存模型最优化方程、固定概率及所有预测误差；该入口不拟合、不增加节点、不运行账户。依赖版本见requirements.txt。

本轮仅评价预测，不生成账户、频率、夏普或回撤新结果。原已关闭账户仍原样保留，不能将本轮误差下降自动转换成账户达标。
""")
    write("用户要求.md", """
# 最新用户要求

用户最新明确：“先训练胜率与盈亏幅度，在逐步增加节点”。已纠正上一轮先以年度节点数量筛选训练对象的顺序。

保持仅510300和现金、20万元、最终净夏普1.3、最大回撤10%、每完整自然年至少5次交易、使用已有本地数据不采集。胜率和盈亏幅度先训练及评价，频率放在完整策略最终验收。
""")
    write("GPT审阅提示词.md", """
请审阅这一轮先训练概率与幅度、暂不增加节点的结果。用户仍要求最终每年至少5次、20万元、净夏普1.3、回撤10%，但这些不应取代当前预测训练。

请检查：

1. 概率是否确实固定并分别评价，是否将10节点事后80%误写为预测概率？
2. 条件均值回归、正负样本分组、成熟标签和训练期标准化是否正确？444次是实际新增幅度拟合，是否没有冒称新增概率拟合或账户？
3. Gamma幅度较旧模型改善，但月末两项幅度仍未胜过简单经验；在相同IF概率下，组合期望MSE只改善约0.079%，是否足以支持复杂度？
4. 是否完整披露两个亏损节点变差、前后半段不一致，以及大量既有历史试验造成的限制？
5. 如何进一步区分概率与幅度的可预测部分，以及之后每次增加一种节点时最值得做的检验？请给优先级、具体证据和停止条件，避免推荐据本批收益再挑窗口或阈值。

先给结论，再列具体文件中的问题。本包尚未经过外部审阅，没有交易授权。
""")
    write("EXCLUSIONS.md", """
# 复算范围

包含本轮直接数值输入：原训练/节点特征、十日标签复算行情、旧概率和幅度参数、原预测及本轮完整代码和新参数。原IF原始合约表及获取过程不重复打包，本轮不重建IF因子；上轮交付回执、协议和关闭记录提供来源。

包内verify复算标签、检查旧/新模型的训练时钟及方程，并重算全部预测误差；不重新拟合、不采集、不增加节点、不运行账户。未建立独立前向证据、外部审阅或当前交易信号。

排除缓存、验证解压目录和包外最终回执。FILE_INDEX.csv列出其余所有成员的大小及SHA-256。
""")
    status = ROOT / "RESEARCH_STATUS.md"
    text = status.read_text(encoding="utf-8-sig")
    marker = "# 510300 研究权威状态"
    entry = "> 2026-09-24 用户最新调整顺序：“先训练胜率与盈亏幅度，再逐步增加节点”。当前年度至少5次只用于最终策略验收，不再作为预测训练准入条件。完成 `510300_PROBABILITY_PAYOFF_PREDICTION_STAGE_V1`：沿用124月末和10节点，概率参数固定，新拟合444个条件均值模型，6固定对照共660条预测；0新概率拟合/节点/账户/下载。新幅度月末亏损MSE较旧下降15.12%，仍较历史条件均值高14.77%；同IF概率下整体期望MSE较经验幅度仅改善0.079%，节点亏损幅度反而变差，未建立复杂幅度增量。先概率/幅度、再逐类增加节点的顺序已写入权威配置；1.3/10%/年度5次目标保持，整体未完成。见[训练结果](reports/research/510300_probability_payoff_prediction_stage_v1/训练结果.md)、[研究顺序变更](reports/research/510300_probability_payoff_prediction_stage_v1/authority_update.json)。"
    if marker not in text:
        raise ValueError("未找到研究状态插入点。")
    if entry not in text:
        status.write_text(text.replace(marker, marker + "\n\n" + entry, 1), encoding="utf-8")
    print("预测阶段实际训练结果和新顺序已记录；年度频率保持最终验收条件。")


if __name__ == "__main__":
    main()
