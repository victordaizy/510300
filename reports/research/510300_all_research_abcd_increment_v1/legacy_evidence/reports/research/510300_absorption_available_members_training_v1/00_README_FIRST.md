# 阅读导航

本轮完成3109日吸收率计算、393次逐期岭拟合、3个期末岭模型及20个账户情景。固定穿越事件只有4次；126次完整月末评价中，B比价格基准A的MSE高8.45%，相对价格加覆盖率C仅降低0.12%，后半段没有改善。费用覆盖账户B与A/C相同，压力净夏普-0.211、最大回撤8.91%、期末189209.97元；误差缓冲主规则无交易。

1. `训练结果.md`、`summary.json`：预测、事件和完整账户。
2. `training_amendment.json`：必须结合原`protocol.json`阅读。因穿越事件只有4次，原12事件训练入口在拟合前停止；首次模型拟合前改用固定月末训练样本，穿越规则保持。原冻结字节保留，不隐去修订。
3. `inputs/parent_data_failure.json`：旧V1要求95%覆盖失败的记录。本轮按用户放宽要求用90%探索，不将旧失败改成通过，也不宣称独立新家族。
4. `results/月末训练特征与标签.csv`、`results/月末逐期校准预测.csv`、`results/候选事件与20日标签.csv`、`results/逐期预测.csv`：139个月末样本、508行月末预测（其中4行标签未成熟）、16行穿越预测。
5. `models/`、`results/完整逐日账户.csv`、`results/机会成交账簿.csv`：528份模型/均值快照及完整现金流。
6. `inputs/constituent_prices.parquet`、成员输入、`results/本轮使用的逐日成分.parquet`：可复算输入。成分股只作为信息，执行资产仅510300与现金。

本轮实际运行入口为`code/absorption_available_members_monthly_training_v1.py`。旧事件训练入口及`parent_builder_reference_only.py`仅作历史代码参考；后者含旧采集能力，本轮没有调用。

独立解压后，只读核对命令为`python -X utf8 code/absorption_available_members_monthly_training_v1.py verify --root .`。该命令检查保存公式、模型、标签与分红到账，不重新生成日度协方差、不拟合预测模型、不重跑账户、不下载。

主规则的RMSE来自过去月末顺序预测，不能解释成对4个稀疏事件校准过的置信区间。历史已见，来源为回溯重建，没有独立前向证据。ZIP验证不等于外部GPT审阅。
