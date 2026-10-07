"""整理压力危险率已有数据训练的保存结果；不拟合、不回测。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_stress_hazard_existing_training_v1"


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    models = json.loads((OUT / "models/季度已训练模型.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUT / "results/预测指标.csv")
    comparisons = pd.read_csv(OUT / "results/模型增量比较.csv")
    predictions = pd.read_csv(OUT / "results/逐期风险分数.csv")
    valid = predictions[predictions.risk_score.notna()]
    wide = valid.pivot(index="origin_date", columns="model", values="risk_score")
    descriptions = {"B0": "事件加权历史基准", "B1": "价格风险模型", "B2": "压力传导模型", "B3": "压力传导加宏观状态"}
    diagnostics = []
    for name in ["B1", "B2", "B3"]:
        selected = [model for model in models if model["model"] == name]
        diagnostics.append({"model": name, "quarterly_fits": len(selected),
            "all_slopes_zero_quarters": sum(max(model["slopes"]) <= 1e-10 for model in selected),
            "maximum_slope": max(max(model["slopes"]) for model in selected),
            "maximum_score_difference_from_B0": float((wide[name] - wide.B0).abs().max())})
    pd.DataFrame(diagnostics).to_csv(OUT / "results/模型近常数诊断.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    write_json("branch_decision.json", {
        "status": "STOP_CURRENT_FIXED_HAZARD_REPRESENTATION_NO_PARAMETER_RESCUE",
        "basis": "实际季度训练后B2没有胜过价格风险B1，B3没有改善B2；分数几乎等同成熟训练基准率。",
        "scope": "只否定当前固定特征、非负约束、原L2=1和事件权重表示，不宣称所有压力信息无价值。",
        "numerical_effect_is_tiny": True, "no_opposite_signal_claim": True,
        "old_sample_and_era_failures_preserved": True, "portfolio_status": "NOT_RUN_RISK_SCORE_IS_NOT_RETURN_OR_ENTRY_MODEL",
        "goal_achieved": False, "new_collection_enabled": False, "orders_authorized": False,
    })
    lines = ["# 压力危险率：已有数据实际训练", "",
        "本轮已经训练完成，但没有得到可用于高置信机会筛选的预测改善。87次固定逻辑回归拟合覆盖29个季度，B2压力传导模型接近历史基准率，B3再加入宏观状态也没有带来可辨认的改善。当前表示停止调整；20万元、净夏普1.5、最大回撤10%的总目标仍未完成。", "",
        "用户要求直接训练、放宽限制、不要采集。原研究在历史修补后有31个可识别压力事件，但仅2个时期达到原跨时期样本门，模型一直未拟合。本轮另建探索产物，保留原阶段失败，使用完全相同的修补数据和原模型表达；没有把旧门槛改写成通过。", "",
        f"原始2,813个标签全部从原版510300行情和分红重新核对。共同输入可用原点{summary['common_origins']}个，含31个压力事件；季度训练成熟后，每个模型有{summary['scored_origins_per_model']}个可评价原点、29个完整压力事件。另104个共同原点处于训练预热，保持无分数；其余1,534个原点不满足原共同输入可用性，未填值。有效预测日期为{wide.index.min()}至{wide.index.max()}，不是2026-09-22当前信号。", "",
        "目标BAD10是：下一交易日实际开盘进入后的10个交易日内，任一收盘加有权分红的累计损失达到4%。它不是10日期末收益，也不是未来盘中最低价。这里只判断不利路径，低分数不能自动解释为上涨机会。", "",
        "B0为同一成熟训练集的事件加权历史基准；B1使用自身波动、回撤和近期跌幅；B2使用内部扩散及资金压力T和T×内部脆弱性F；B3再加T×宏观状态M。B2替换B1的模型表示，B3是在B2上增加一项。所有斜率非负，截距不惩罚，原始分位值不标准化，L2固定为1，未搜索窗口、惩罚或方向。", "",
        "每季度首个交易日更新；正类整个事件的最后标签结束早于版本日才能训练，负类标签也必须先成熟。正类所有重叠原点使用事件首次原点时适用的同一模型，不能把一场压力分到多个模型版本。事件归组使用完整历史标签，只用于预先固定的切分、去重及事后评价，不是可交易事件识别器。", "",
        "关键解释边界：原模型把同一压力事件的总权重设为1，每个非事件日的权重也为1。评价样本的加权正类率约2.79%，实际原点中正类占比约13.87%。因此输出必须称事件加权风险分数，不能将2.5%的分数直接说成未来十日真实下跌概率；当前也没有另做概率校准。", "",
        "|模型|共同评价原点|加权Log Loss，越小越好|加权ROC AUC，仅描述|", "|---|---:|---:|---:|"]
    for row in metrics[metrics.segment == "ALL"].itertuples():
        lines.append(f"|{descriptions[row.model]}|{row.origins}|{row.weighted_log_loss:.9f}|{row.weighted_roc_auc:.4f}|")
    lines += ["", "|比较|相对前者的损失改善，正数为好|", "|---|---:|"]
    for row in comparisons[comparisons.segment == "ALL"].itertuples():
        lines.append(f"|{row.candidate}相对{row.reference}|{row.relative_log_loss_improvement:.5%}|")
    lines += ["", "这些损失差仅约百万分之一量级；没有统计或经济上有用的增量，不把细小负差解释成可靠的反向信号。B2在18/29个季度的全部系数都为零，全部季度相对B0的最大分数差只有约0.00148个百分点。数值优化在设定容差内收敛，不依据保存的微小差异继续调整惩罚或标准化。", "",
        "四个原固定时期的可评价压力事件依次为0、3、19、7。无预测时期原样保留；本轮允许小样本探索，没有补出早期证据。按事件和年份两种单位各固定重抽样5,000次，报告全部40,000个损失差及抽样计数。年份只有8组，区间不是独立前向验证；事件抽样中负类按单日分组，也不能消除其标签重叠相关性。", "",
        "没有运行新账户：这份原模型尚未产生收益预测、事前买入条件和持有规则，而且风险增量检验无效。夏普和回撤保持未计算，没有把零交易或低风险分数算成目标通过。没有网络采集、重新启用采集任务、下单或恢复已终止策略。", ""]
    (OUT / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")
    documents = {
        "00_阅读导航.md": """# 阅读导航

先看训练结果.md、summary.json和branch_decision.json。本轮完成实际训练，不能作为达标投资策略。

- protocol.json和freeze.json：本轮训练前固定的输入与模型规则。
- inputs/samples.parquet：完整2813行原修补样本，包括原不合格行；实际模型只用1279行共同可用样本。
- inputs/market.parquet和dividends.csv：原BAD10标签所需行情和分红。
- inputs/parent_*：旧失败、原模型设定、数据修补回执、版本锁及原跨时期计数。
- code：完整本轮代码、继承的纯逻辑回归函数及原父代码参考。
- models：87个逻辑回归参数快照、29个历史基准率；末次季度模型引用没有重复拟合。
- results：共同样本、全部事件、季度计数、逐期分数、全部指标、40,000条固定重抽样结果及原始抽样计数。
- FILE_INDEX.csv：包内其他成员的字节数和SHA-256。

在包根目录使用requirements.txt列出的环境，运行 `python -X utf8 code/stress_hazard_existing_training_v1.py verify --root .`。复核会重算BAD10标签、成熟训练集合、模型一阶条件、保存预测、指标和固定抽样结果，不重新拟合、不新建账户、不生成随机样本、不下载数据。

交付回执位于研究目录包外，记录ZIP身份和新解压保存结果复算。ZIP可以交给GPT评议，本次没有上传或声称已经外部审阅。
""",
        "用户需求与本轮边界.md": """# 用户需求与本轮边界

用户目标：只操作510300；本金20万元；全账户扣成本净夏普至少1.5、最大回撤10%；允许一年只有四五次机会，其余时间空仓，次数不是配额。

当前执行要求：请加速干，不用那么严格，直接进入训练，不用采集。本轮因此跳过旧样本数量和时期数量准入限制，实际拟合此前未拟合的原定模型，旧失败结果保留。

这一步检查机会研究中的风险识别是否有信息增量，是总目标的一个研究步骤。它不替代最终账户验收，也没有把风险分数直接当成买入策略。前一轮IF卖压缓解训练和账户属于已完成进展；本轮新增87次拟合及其否定性证据，总目标仍ACTIVE。
""",
        "GPT审阅提示词.md": """# 可复制审阅提示词

请评议这份510300压力危险率实际训练包。用户只允许510300与现金，20万元，净夏普1.5、回撤10%，接受少数机会和长期空仓；本轮明确不采集，允许放宽旧样本准入后直接训练。

先检查是否确实完成了87次季度逻辑回归和29份基准率快照，旧31事件/时期不足的失败是否保留。请重点审查成熟标签、整事件去重和季度版本分配、共同样本以及104个预热原点；非负约束、不标准化和固定L2=1的解释边界；事件权重导致输出不能视作真实每日概率；微小损失差与数值精度；无预测时期和重抽样相关性。

请说明本轮证据究竟否定了什么、没有否定什么，以及是否值得继续分配研究资源。不要把低风险分数直接变成上涨机会，不要把本轮没有账户误称为夏普达标。给出少量有优先级的下一步、验证要求与停止条件，区分已有数据可完成的工作和必须有新增数据才能完成的工作。
""",
        "EXCLUSIONS.md": """# 包范围与排除

包内包括本轮模型的全部直接输入、2813个原始样本标签所需的原版行情/分红、父版本锁与修补回执、实际参数、预测、指标和固定抽样计数。

不复制原股票级四状态数据构建的整个历史仓库、全部原始财报或停牌资料、虚拟环境以及无关试验；包内父回执和版本锁保留这些上游来源定位。因此新解压复核覆盖本轮标签、模型和保存预测，不声称独立重建了全部上游股票级数据。没有新账户、现实成交或外部审阅。

verification、新解压副本和delivery_receipt.json留在包外，避免循环。Python编译缓存不收入ZIP。
""",
    }
    for name, content in documents.items():
        (OUT / name).write_text(content, encoding="utf-8")
    print("训练结论、风险分数解释边界和停止条件已保存。")


if __name__ == "__main__":
    main()
