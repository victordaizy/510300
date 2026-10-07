"""完成节点联合评价说明，并记录用户最新的研究方向。"""
from datetime import datetime
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sparse_node_joint_quality_v1"
REQUEST = "我们只需要交易某几个节点，高盈亏比，包括变盘信号，高胜率，高夏普，这些结合起来，拒绝过拟合"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(name, value):
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    accounts = pd.read_csv(OUT / "results/完整账户联合指标.csv")
    timestamp = datetime.now().astimezone().isoformat()
    if summary["account_metric_records"] != 170:
        raise ValueError("联合计算范围不正确，不能形成结论。")
    contract = {
        "contract_id": "510300_SPARSE_NODE_TRAINING_CONTRACT_V1", "recorded_at": timestamp,
        "user_instruction": REQUEST, "capital_cny": 200000, "assets": ["510300.SH", "CASH_CNY"],
        "target_net_sharpe": 1.3, "target_max_drawdown_magnitude": 0.1,
        "frequency": "少数入场前可识别的节点；每年四五次可接受，不设置交易配额。",
        "joint_preferences": ["高胜率", "高实际盈亏比", "正的成本后单次期望", "完整账户高夏普", "有限尾部损失"],
        "minimum_win_rate": None, "minimum_payoff_ratio": None,
        "threshold_reason": "用户没有指定这两项数值，先联合报告，不事后反推门槛。",
        "node_fields_before_entry": ["节点机制及完整事件母集", "可用信息与时点", "失衡或变盘的顺序触发条件",
                                     "可执行入场时点", "预期收益来源及估计不确定性", "逻辑失效条件及可执行退出",
                                     "最晚退出时点", "预计成本与仓位风险"],
        "model_target": "在事前固定的节点、持有和退出规则下，估计获利概率及条件盈利/亏损分布；账户夏普用于评价结果。",
        "signal_definition": "每一日的触发只依赖该日决策前已知数据；未来峰谷只能作为训练标签，不能回填信号时间。",
        "direction": "只有510300多头和现金；向下变盘只用于不入场或退出。",
        "preentry_reward_risk": "事前预期盈利空间/失效风险与事后平均盈亏比分列；预计止损幅度不保证成交损失。",
        "training_unit": "完整候选事件，保留未触发和失败事件；不能仅使用已经筛出的盈利成交。",
        "splits": "按时间逐期训练，只使用预测前已成熟的标签；同一市场事件及重叠持有期不跨训练评价边界泄漏。",
        "baseline": "相同节点、日期、成本、持有规则下的简洁价格基准与全部节点规则，检查新信息有无增量。",
        "search_budget": "每次明确一个机制增量与固定的小模型；记录全部尝试，不因看见测试损益再换窗口、止盈止损或特征。",
        "exploration_allowed": "样本小可以训练并披露不确定性，不用繁重准入流程替代训练；探索结果不自动等于稳定优势。",
        "stability": "查看原时间分段、完整事件集中度及预先声明的成本情形；不新增每个任意子期都必须夏普1.3的条件。",
        "independence": "多指标同源或同一事件的多个交易日不算独立证据；已反复查看的历史保留发现样本身份。",
        "collection_enabled": False, "orders_authorized": False,
        "terminated_strategy": "SELECTED_MIX_BAND10_SIMPLE2",
        "current_round": "联合质量测量和训练目标明确化；不把本轮测量说成新模型训练。",
    }
    write_json(ROOT / "config/510300_sparse_node_training_contract_v1.json", contract)
    write_json(OUT / "node_training_contract.json", contract)

    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = json.loads(mandate_path.read_text(encoding="utf-8-sig"))
    if mandate["target_net_sharpe"] != 1.3 or mandate["capital_cny"] != 200000:
        raise ValueError("当前权威目标与本轮要求不同，不能覆盖。")
    mandate.update({"latest_user_instruction": REQUEST, "joint_node_objective": contract["joint_preferences"],
                    "node_definition_required_before_entry": True, "minimum_win_rate": None,
                    "minimum_payoff_ratio": None, "node_direction_revision_at": timestamp,
                    "node_training_contract": "config/510300_sparse_node_training_contract_v1.json",
                    "node_direction_revision_receipt": "reports/research/510300_sparse_node_joint_quality_v1/authority_update.json"})
    write_json(mandate_path, mandate)
    write_json(OUT / "mandate_after.json", mandate)
    write_json(OUT / "authority_update.json", {"recorded_at": timestamp, "instruction": REQUEST,
               "before": "inputs/mandate_before.json", "after": "mandate_after.json",
               "contract": "node_training_contract.json", "target_net_sharpe": 1.3,
               "capital_cny": 200000, "target_max_drawdown": 0.1,
               "new_market_data_collection_enabled": False, "goal_achieved": False})

    examples = [
        ("IF耗竭：加入模型", "510300_if_exhaustion_existing_training_v1", "B_IF_EXHAUSTION", "NET_POSITIVE_DIAGNOSTIC"),
        ("IF耗竭：所有原节点", "510300_if_exhaustion_existing_training_v1", "EVENT_ONLY", "EVENT_RULE_DIAGNOSTIC"),
        ("政策节点：原主要时点", "510300_equity_support_policy_event_v1", "REVIEW_CLOSE_NEXT_OPEN_PRIMARY", "ORIGINAL_SINGLE_POLICY"),
        ("分红支付模型", "510300_dividend_payment_event_training_v1", "B_DIVIDEND", "NET_POSITIVE_DIAGNOSTIC"),
        ("私募意向与空间", "510300_private_manager_exploratory_training_v1", "C_INTENT_SPACE", "NET_POSITIVE_DIAGNOSTIC"),
        ("吸收率模型", "510300_absorption_available_members_training_v1", "B_ABSORPTION", "NET_POSITIVE_DIAGNOSTIC"),
        ("新版M1预期差", "510300_new_m1_consensus_training_v1", "B_PRICE_SURPRISE", "NET_POSITIVE_DIAGNOSTIC"),
    ]
    table = ["| 已保存节点规则 | 完整周期 | 胜率 | 金额盈亏比 | 投入归一化盈亏比 | 全账户净夏普 | 最大回撤 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, study, model, policy in examples:
        subset = accounts.loc[(accounts["study"] == study) & (accounts["model"] == model) &
                              (accounts["policy"] == policy) & (accounts["cost"] == "STRESS")]
        if len(subset) != 1:
            raise ValueError("示例账户不唯一。")
        row = subset.iloc[0]
        cash_ratio = "未定义" if pd.isna(row.realized_cash_payoff_ratio) else f"{row.realized_cash_payoff_ratio:.2f}"
        return_ratio = "未定义" if pd.isna(row.realized_return_payoff_ratio) else f"{row.realized_return_payoff_ratio:.2f}"
        table.append(f"| {label} | {row.cycles} | {row.win_rate:.1%} | {cash_ratio} | {return_ratio} | {row.full_calendar_net_sharpe:.3f} | {row.max_drawdown_magnitude:.2%} |")
    old = accounts.loc[(accounts["model"] == "RUNS_REFERENCE_BLEND") & (accounts["cost"] == "STRESS")]
    old_table = ["| 原方案的保存时期 | 周期 | 胜率 | 金额盈亏比 | 投入归一化盈亏比 | 全账户净夏普 |",
                 "|---|---:|---:|---:|---:|---:|"]
    for period, label in [("evaluation", "2020-01-02至2026-08-14"), ("earlier_diagnostic", "2015-01-05至2019-12-31")]:
        row = old.loc[old.period == period].iloc[0]
        old_table.append(f"| {label} | {row.cycles} | {row.win_rate:.1%} | {row.realized_cash_payoff_ratio:.2f} | {row.realized_return_payoff_ratio:.2f} | {row.full_calendar_net_sharpe:.3f} |")
    write_text("研究结论.md", f"""
# 少数节点的胜率、盈亏比与全账户夏普联合评价

研究方向已按用户最新要求更新：只做入场前能识别的少数510300节点，把变盘依据、胜率、盈亏比及完整账户夏普放在同一规则下评价。资金20万元，成本后净夏普目标1.3，最大回撤10%，允许一年四五次机会，其余时间现金；现有资料继续使用，不新增采集。本轮完成联合测量与训练任务明确化，没有得到可直接认定稳定达标的新节点。

## 已有实际节点给出的信息

下面全部取各研究原STRESS成本，保留原交易、退出和全部空仓交易日。表中模型交易为原较宽松的净预测为正诊断规则；它们与原误差缓冲主要规则的全空仓结果均保留在完整表。胜率按完整交易周期计算，不使用每日涨跌命中率。

{chr(10).join(table)}

IF模型9次中7次盈利，但高胜率没有转化为1.3的账户夏普；所有原IF节点的10次交易有8次盈利，夏普反而稍高，不能宣称加模型已带来增量。政策节点3次全赢，但亏损样本为空，平均盈亏比无法估计；其中最大一笔约占净利润81.2%，不能据此认定真实胜率就是100%。分红模型实际金额盈亏比接近2，但胜率只有三分之一，整体净利润仍为负。LPR候选全部未成交，相关胜率、盈亏比和夏普保持未定义，完整表未将它们删除。

金额盈亏比是盈利周期平均净利润除以亏损周期平均净亏损。投入归一化盈亏比先将每个周期净利润除以该周期累计买入支出，再计算平均赢/平均亏；旧周期可能多次加减仓，因此此指标也不是按失效风险归一化的R倍数。两项差异反映收益分布与投入规模的共同影响，不能只展示较大的那个数。

## 看起来达标的旧历史为什么仍需谨慎解释

以此前已列入14方案的RUNS_REFERENCE_BLEND为例：

{chr(10).join(old_table)}

这个旧方案在主历史中同时出现较高胜率、金额盈亏比和夏普，较早历史则明显不同。本轮原样保留14方案全部两时期、两档成本：28条主历史记录满足夏普1.3和回撤10%的数值要求；28条较早记录均未同时满足。较早时期是已保存的对照，不新增“任意子期都必须达标”的要求。14个相关变体共用大量历史和信号，不能当作14次独立成功，也不因这张表重新启用原关闭策略。

旧STRESS单边佣金4bp、滑点10bp；近期STRESS单边佣金2bp、滑点10bp。最低佣金均5元。各自按原设定复算，表间不作相同成本、相同日历的直接排名。每个旧时期均包含原期末清仓周期，完整表另列强制或截尾退出次数。

## 如何落实用户要求

节点先定义“为什么此处可能改变收益分布、什么信息触发、何时能成交、出现什么情况说明判断失效”，再学习这些节点上的获利概率和条件盈利/亏损。事前预期收益空间与失效风险单列，不能用事后平均盈亏比冒充开仓前已有把握。用户未指定胜率和盈亏比的数值，不从这批结果反推一个恰好过线的门槛。

将概率、盈利幅度、亏损幅度和账户结果分开估计、共同检验。若以W表示平均盈利、L表示平均亏损绝对值，单次平均净期望为p_win×W−p_loss×L（零收益周期另算）；本轮交易利润已经扣除原成本，不再重复扣费。没有观察到亏损时，L未知，不能把公式中这一项当作未来必为零。

训练使用完整候选事件，而非只拿已成交或盈利的节点。保持时间顺序与成熟标签，同一事件及重叠持有期避免跨训练/评价边界。每轮只检验明确的机制增量，保留简洁基准；不用测试期收益选择阈值、窗口和止盈止损。小样本允许探索训练，结论如实反映不确定性，不把繁重准入流程当成训练工作的替代。

详细可执行口径见《节点训练任务书.md》和node_training_contract.json。这个任务书是后续训练遵循的口径，并不表示已经训练了新的获利概率或条件盈亏模型。

## 计算范围与交付

近期7个事件研究的114条账户记录（86策略、28对照），以及此前已列明14方案的56条账户记录，共170条；复算{summary['recomputed_calendar_rows']:,}条逐日权益记录，统一{summary['saved_cycle_rows']:,}条成交周期记录。跨模型与成本重复记录不合并为独立交易样本。全部来源和逐项结果入包。

完整逐日夏普与原值最大差异{summary['maximum_sharpe_recomputation_error']:.3g}；周期净利润与账户净利润最大差异{summary['maximum_cycle_profit_identity_error_cny']:.3g}元。近期86条策略记录中，满足当前夏普和回撤数值要求的仍为0。

本轮新增拟合0、新账户0、下载0；完成的是已有实际训练账户的联合质量测量和目标更新。整体“稀疏高置信度且拒绝过拟合”的目标仍未证明完成。当前资料的结果不构成某个当下买点。
""")
    write_text("节点训练任务书.md", """
# 稀疏节点训练任务书

目标是识别少数有条件优势的交易节点。变盘信号是入场前的信息，胜率和盈亏比分别描述是否容易赚、赚亏多大，夏普和回撤评价整个20万元账户。四者来自同一套可执行规则，不通过拼接各自最漂亮的回测来满足。

## 一次机会的完整定义

准备状态 → 当时可见的触发 → 最早可执行入场 → 逻辑失效或时限退出 → 恢复现金。

每个节点记录机制、原始信息可用时间、触发日期、预计收益来源、失效条件、最晚退出日期和成本。失效条件可以是市场状态不再支持原假说，不能只写“亏到不舒服”。未来极值、事后ZigZag低点及完整行情才知道的变盘点，不得回填成当时可见的触发。

“卖压缓解”“趋势转强”“公开事件后的重新定价”只作为机制分类。每类先写具体且可反驳的条件；这些名称本身不代表存在优势，也不自动重新开启此前失败分支。510300只做多或现金，向下变盘用于回避或退出。

## 训练和评价对象

1. 从原始条件生成全部候选事件，包括没有触发、触发后失败和资料不足的事件，保留原因。相邻日期属于同一行情事件时不能当成大量独立样本。
2. 在查看本轮评价结果前，确定事件定义、退出期限、成本和仓位规则。未来收益可以是监督标签，但触发特征只能来自决策前。
3. 优先用低维、固定复杂度的模型学习“在这个节点盈利的概率、盈利时的幅度、亏损时的幅度”。资料不足以估计某一项时，报告未知和样本数，允许探索结果继续输出。
4. 按时间推进，只用当时已成熟标签拟合。保留相同事件、日期和成本下的价格基准与直接节点规则，判断加入信息是否真正改善表现。
5. 输出完整账户净夏普、最大回撤、周期数、胜率、两种实际盈亏比、单次期望、最大盈利贡献和尾部损失。空仓时间计入账户评价。

输入胜率的预测概率须检查校准；分类打分、加权分类器的原始输出不能直接冒充概率。按累计买入支出归一化的收益亦不是按预先失效风险计算的R倍数。

## 拒绝过拟合的最小约束

每轮只比较明确的机制增量与固定基准，保存所有尝试。已经看过的历史仍是发现资料，不因为换切分名称就成为独立证据。同一事件跨模型、成本的结果不能堆成独立样本；多个同源价格指标也不视为多重独立确认。

对同一批结果，不反复换观察窗口、改变入场阈值、缩短亏损持有期或删掉失败节点。查看分期表现和最大事件贡献，用于判断依赖关系；不把少赚少波动、没有交易或没有亏损样本自动说成高确定性。

用户要求快，现有资料足以计算的探索就直接运行；不再堆叠无必要的审批或样本准入门。最少保留时间顺序、实际成本、完整账簿和失败记录。

## 当前已有证据

本轮完成的是170条保存账户的联合测量，尚未新拟合概率与条件盈亏模型。近期IF节点展示了高历史胜率但账户夏普低的情形；旧14方案展示了主历史较好而较早阶段下降的情形。当前没有已确认为稳定满足全部偏好的策略。

用户当前数值约束为净夏普至少1.3、最大回撤不超过10%；胜率及盈亏比暂未指定硬阈值。不得从已见结果倒推一个看起来过线的数字，不要求每年必须凑够四五次交易。
""")
    write_text("指标口径.md", """
# 指标口径

- 胜率：净盈利完整周期数/全部完整周期数。盈亏分界容差为0.000001元，未成交账户不填胜率0。
- 实际金额盈亏比：平均盈利金额/平均亏损金额绝对值。无盈利或无亏损时保留未定义及原因，不填写无限大。
- 投入归一化盈亏比：周期净利润/周期累计买入支出，再计算盈利均值与亏损均值绝对值之比。不是最大仓位收益、资金时间加权收益或R倍数。
- 利润因子：盈利总额/亏损总额绝对值。它与平均盈亏比不同；存在亏损且没有盈利时为0，没有亏损时未定义。
- 单次净期望：所有保存周期的净损益平均值。只描述历史样本，不当作已知未来期望。
- 夏普：初始20万元逐日权益变动计算的收益均值/样本标准差×sqrt(242)，含原账户窗口中所有现金日，现金与无风险收益均按原假设为0。无波动账户夏普未定义。
- 回撤：从初始资金开始的权益历史高点回撤绝对值。完整表统一为正的幅度。
- 集中度：最大单笔盈利/全部盈利，以及最大单笔盈利/总净利润。后者仅总利润为正时计算，允许大于100%。扣去最大单笔仅是金额归因，不是另一个可执行账户。
- 期末退出：保留原期末强平或截尾，另列次数，不通过删除不利期末交易美化指标。
- 事前收益风险比：这次周期账簿复算不建立该证据；不能从事后平均盈亏比推定开仓时已知。
- original scope：旧14个模型来自上一轮已公开范围，仍带有历史筛选，不是本轮重新找到的14个独立策略。不同模型/成本共享周期，1496行不是1496个独立行情事件。

原7项协议中的1.5以及旧配置中的1.2均保留为历史来源。当前判定统一使用用户最新1.3，源文件历史目标不覆盖当前目标。
""")
    write_text("00_README_FIRST.md", """
# 交付导航

先读研究结论.md，其次读节点训练任务书.md。完整账户联合指标、统一周期与原14方案跨时期表在results目录。指标口径.md区分金额盈亏比、投入归一化盈亏比、利润因子和事前收益风险。

node_training_contract.json是当前节点研究口径，mandate_after.json是更新后的用户授权。inputs/mandate_before.json保留更新前版本，authority_update.json记录变更。

inputs含直接使用的原完整逐日账户、交易周期、指标、协议或配置，以及旧14方案范围。freeze.json固定这些来源和计算代码；FILE_INDEX.csv索引包内文件。使用包内code/sparse_node_joint_quality_v1.py的verify命令并以--root指定解压目录，即可从保存账户复算联合指标。该入口不拟合、不下载，也不生成交易账户。

requirements.txt列出本次环境依赖。完整市场数据、原训练特征和父模型拟合代码不在本包，因本轮不重建训练或执行原回测。范围见EXCLUSIONS.md。
""")
    write_text("用户需求.md", f"""
# 用户需求

最新原话：{REQUEST}

此前要求继续有效：仅510300和现金，20万元，最大回撤10%，净夏普已从1.5调整为1.3；一年四五次可以，其他时间空仓，不凑交易次数；不用采集，使用已有本地资料加速探索训练，减少不必要审计。

本轮执行的是最新方向的明确化，以及现有实际训练账户在胜率、盈亏比、夏普和回撤上的联合测量。未自行增加胜率或盈亏比硬门槛。
""")
    write_text("GPT审阅提示词.md", """
请围绕用户要少数节点、高胜率、高盈亏比、净夏普1.3及拒绝过拟合的目标审阅：

1. 三类指标是否来自同一套保存周期和完整每日账户？现金日、最低佣金、压力成本差异是否保留？
2. 是否区分实际金额盈亏比、投入归一化盈亏比、利润因子和入场前预计收益风险？3次全赢的亏损分布是否保持未知？
3. IF加入模型是否比同一批原节点规则更好？旧14个相关变体和两历史时期是否被错误当作独立验证？
4. 节点训练任务是否覆盖全部候选事件、时点、逻辑失效和标签成熟？是否存在用未来峰谷回填信号或用评价收益倒推门槛的空间？
5. 请指出最值得继续检验的具体机制缺口，以及可以停止的重复调参；优先具体、低成本的验证，不添加与问题无关的流程。

请引用包内文件。先给结论，再给错误、优先级、验证办法与停止条件。本包尚未经过外部GPT审阅。
""")
    write_text("EXCLUSIONS.md", """
# 范围与排除

本包自含保存账户及联合指标复算所需的全部直接数值输入。它不重建原数据采集、模型拟合、新闻抽取或完整历史回测，因此不重复纳入原市场数据库、公告原文及父级全部代码。各输入协议、来源路径和已有交付回执用于溯源。

近期研究保留全部114条策略及基准账户；旧档案范围固定为上一轮已经列明的14方案56账户，不宣称覆盖全部历史研究。父级指标表中其它模型保留在原表中，但不越过本轮范围新增选优。

不包含运行缓存、解压目录或ZIP外最终交付回执。根FILE_INDEX.csv自身不列入自己的哈希；其余包内成员均列入。交付结构和算术复算通过，不代表独立市场验证或外部审阅。

本轮0新拟合、0新账户、0新下载，不创建订单或恢复已终止策略，也不修改系统目标状态。
""")
    entry = "> 2026-09-22 用户新增明确方向“少数节点、高盈亏比、变盘信号、高胜率、高夏普、拒绝过拟合”，已更新当前研究授权及节点训练任务书。完成 `510300_SPARSE_NODE_JOINT_QUALITY_V1`：170条原账户、249594条保存日线、1496条含重复的周期记录复算胜率、两种盈亏比、单次期望与完整账户夏普。近期86条策略记录当前数值通过0；旧14方案主历史28条通过、较早28条未同时通过，保留历史筛选和相关性。IF模型9次7胜、金额盈亏比1.84、净夏普0.348；政策3次全赢但盈亏比未定义。新增拟合/账户/下载均0，此次进展不等于整体目标完成；现有系统BLOCKED回执保留，最新方向不被旧状态禁止。见[联合结论](reports/research/510300_sparse_node_joint_quality_v1/研究结论.md)、[节点训练任务书](reports/research/510300_sparse_node_joint_quality_v1/节点训练任务书.md)。当前目标仍为20万元、净夏普1.3、回撤10%、仅510300与现金，不采集。"
    status_path = ROOT / "RESEARCH_STATUS.md"
    status = status_path.read_text(encoding="utf-8-sig")
    marker = "# 510300 研究权威状态"
    if marker not in status:
        raise ValueError("未找到研究权威状态插入点。")
    if entry not in status:
        status_path.write_text(status.replace(marker, marker + "\n\n" + entry, 1), encoding="utf-8")
    print("联合质量结论、节点训练任务书和最新研究方向已保存；夏普目标保持1.3。")


if __name__ == "__main__":
    main()
