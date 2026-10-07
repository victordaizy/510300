# 当前交付：2021—2026样本门槛放宽V2

先读研究结论.md和2021至2026账户对比.png，再读user_request.md、protocol.md。V2只把样本数量6/4改为3/2；没有放宽收益、风险、成本、状态映射或交易执行。

comparison.csv包含全部16条完整账户指标，annual.csv包含逐年结果。baseline是2021年起原规则对照，relaxed是V2，仅近期启停增加1笔亏损，完整启停仍0笔。relaxed/eligibility_changes.csv、trigger_decisions.csv保留启停变化和每次机会的判定；不能将47条新增资格记录当作47次交易。

inputs自含重新计算所需价格、股息、前序特征、完整形态信号、事件标签、普通时点对照和原资格。sources包含直接来源材料。baseline_freeze.json和relaxation_freeze.json分别固定两阶段输入与代码。前序统计除两个资格门槛外完全相同。

prior/V1_GPT_review.zip是经过SHA-256身份核对的原始完整V1快照，包含三类247个完整过程、87次确认、原36账户、机制测试、原用户请求和分红来源；它是旧快照，不可用旧近期指标替代本轮2021年起指标。当前ZIP根目录FILE_INDEX.csv是本包权威索引。

在解压目录安装requirements.txt后，用Python运行 `python code/verify_sequential_patterns_relaxed_v2.py --root .` 只读复算。本包结构核对和保存计算一致性不等于已完成外部GPT审阅、独立前向验证或达到收益目标。
