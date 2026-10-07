# 阅读导航

本轮已经实际训练，先读训练结果.md和summary.json，再看results/账户联合指标.csv、逐节点交易判断.csv和联合筛选与原节点对照.csv。

用户在训练完成后新增“每年至少5次交易”，当前以annual_frequency_requirement.json及最新频率要求与逐年评价.md为准。逐年完整持仓周期.csv和频率夏普回撤联合评价.csv包含188条保存账户的新增频率评价；首尾不完整年不年化凑数。

- protocol.json和freeze.json：本轮拟合前固定的模型、比较规则、成本及来源身份；明确历史早已被查看。
- inputs：原市场、分红、IF状态和上游已有输入；原关闭结果；父级月末/事件表与基准日线；最新用户要求。
- code：完整本轮训练代码和所用原账户、标签计算依赖。
- models：330份逐期快照、3份最终快照；每份包含训练月份、最晚标签时间和全部参数。
- results：124个月末样本、全部10节点、330条预测、概率分组、18账户及每个节点的进入/拒绝原因。
- figures：相同完整日历上的账户权益和回撤图。
- branch_decision.json：本固定表示停止，不进行结果驱动的参数补救。
- FILE_INDEX.csv：除自身外所有包内文件的大小与SHA-256。

用Python运行code/if_node_joint_payoff_training_v1.py的verify命令，--root指定解压目录，即可代入保存模型、检查成熟标签、最优化方程和完整账簿。verify新增拟合、账户、下载均为0。train入口拒绝覆盖已有summary.json。
