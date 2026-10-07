# R05｜一个跨市场通道：真实IF持有成本

假设：已知融资成本与已公告现金分红调整后的真实合约偏离，是否有额外跨日信息。

旧用途及结果：旧八条真实期限结构规则冻结拒绝；总OI增量失败、旧全球隔夜主压力夏普0.1308、年化0.8417%。本批不同时搜索汇率/股债/海外风险。

现有字段：旧真实IF合约/现货覆盖存在，但严格成本调整需要同合约日历、当时可知利率及预期分红映射。

未解决：未建立区别于旧基差水平的完整点时持有成本输入；事后实现分红和总OI不能分别充当事前预期与净多头。

本次处置：`INTAKE_COMPLETE_CARRY_SOURCE_NOT_ADMITTED`。

下一最小步骤与停止线：只有新增的可核验成本成分成立才登记单一残差，不换旧期限尾部分位救回。

若准入，主对照为同期限D日线字段；先固定信息可得原点，再比较共同日期预测及完整日历净账户。不得把D-native原二日模型直接移植为不同期限的已验证对照。模型和主目标见experiment_specs.json；当前没有新拟合、标签或账户。

直接证据：

- reports\research\510300_if_true_term_structure_binary_screen_v1_0_1.json
- reports\research\510300_if_open_interest_increment_v1\result.json
- reports\research\510300_overnight_global_information_v1\result.json
