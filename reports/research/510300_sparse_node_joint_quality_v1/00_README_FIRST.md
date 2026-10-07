# 交付导航

先读研究结论.md，其次读节点训练任务书.md。完整账户联合指标、统一周期与原14方案跨时期表在results目录。指标口径.md区分金额盈亏比、投入归一化盈亏比、利润因子和事前收益风险。

node_training_contract.json是当前节点研究口径，mandate_after.json是更新后的用户授权。inputs/mandate_before.json保留更新前版本，authority_update.json记录变更。

inputs含直接使用的原完整逐日账户、交易周期、指标、协议或配置，以及旧14方案范围。freeze.json固定这些来源和计算代码；FILE_INDEX.csv索引包内文件。使用包内code/sparse_node_joint_quality_v1.py的verify命令并以--root指定解压目录，即可从保存账户复算联合指标。该入口不拟合、不下载，也不生成交易账户。

requirements.txt列出本次环境依赖。完整市场数据、原训练特征和父模型拟合代码不在本包，因本轮不重建训练或执行原回测。范围见EXCLUSIONS.md。
