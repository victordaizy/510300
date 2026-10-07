# 阅读顺序

先读研究结论.md及四张figures图；再读protocol.json和results/summary.json；随后检查inputs、sources与evidence。旧口径B的MSE比A高4.98%，本轮拒绝；新口径样本不足，C/D资金资料未准入。20万与2万账户、回撤退出均未运行。

完整104个月选择表、27个月缺失表、76个成熟事件、24次评价与48组模型均已保留。来源是历史重建，不是严格前向证据。来源扩展经过四个固定阶段，全部记录在sources/search_history。

在Windows PowerShell安装requirements.txt依赖后，从解压根目录可执行以下命令（脚本位于code目录）：

- 保存结果核对：`python code/verify_money_consensus_increment_v2.py --study-dir .`
- 重算相同历史模型至新目录：`python code/money_consensus_increment_v2.py analyse --study-dir . --output-dir replay_results`
- 重绘图表：`python code/plot_money_consensus_increment_v2.py --study-dir . --output-dir replay_figures`

第一项不下载、不新增研究候选；重算模型只使用本包冻结输入。原报告整篇PDF未附，无法离线重做PDF解析；已附必要数字、页码、URL和哈希，外部原文件可能变更。来源抓取代码用于记录原流程，路径依赖原仓库，不属于上述离线重算命令。FILE_INDEX.csv覆盖除自身外全部包成员。未进行外部GPT评审。
