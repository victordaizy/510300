# 510300第二轮价量机制与IF环境研究

## 治理边界

- 本轮由第一轮价量候选后验派生，只能做机制拆解和候选收缩，不能提供独立确认。
- 成交量冲击严格使用此前20日均值和标准差，并固定截断至[-4,4]。
- 高成交量趋势交互只在成交量冲击为正时生效，修正旧乘积把缩量下跌记成正值的问题。
- IF主连相对强弱受换月影响，本轮仅作环境探索；原始持仓量变化未注册。

- 首次输出后追加统一审计：pseudo-OOS折算独立观测少于20时，WEAK/STRONG封顶为WEAK_SAMPLE_LIMITED；该规则不伪装成事前登记。

- 另报告第一、二轮合计28个日线假设的研究级BH q值；轮内q值继续保留，避免覆盖原始记录。

## 单因子结果

|ID|因子|目标|开发Spearman|pseudo-OOS Spearman|成本后有利-不利|轮内BH q|全研究BH q|得分|评级|
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|R2H001|510300二十日价格主效应|exec_total_return_20d_net|0.0477|-0.2747|-5.01%|0.1940|0.2716|1/7|REJECTED|
|R2H002|510300相对前二十日成交量冲击|exec_total_return_20d_net|0.0524|0.1281|2.91%|0.0724|0.1520|7/7|WEAK_SAMPLE_LIMITED|
|R2H003|510300高成交量趋势交互|exec_total_return_20d_net|-0.0349|-0.1493|-6.30%|0.0207|0.0485|0/7|REJECTED|
|R2H004|IF主连相对沪深300五日领先|exec_total_return_20d_net|0.0035|0.0522|0.67%|0.0724|0.1283|5/7|WEAK_SAMPLE_LIMITED|
|R2H005|510300二十日价格主效应|exec_total_return_5d_net|0.0425|-0.2209|-2.15%|0.1527|0.2291|1/7|REJECTED|
|R2H006|510300相对前二十日成交量冲击|exec_total_return_5d_net|0.0573|-0.0730|1.59%|0.5221|0.5847|2/7|INCONCLUSIVE|
|R2H007|510300高成交量趋势交互|exec_total_return_5d_net|-0.0102|-0.0694|-1.23%|0.0000|0.0000|0/7|REJECTED|
|R2H008|IF主连相对沪深300五日领先|exec_total_return_5d_net|0.0816|-0.0238|-0.32%|0.0288|0.0728|2/7|INCONCLUSIVE|

## 增量回归

|目标|模型|开发调整R²|相对前一模型增量|pseudo-OOS冻结模型R²|预测相关|
|---|---|---:|---:|---:|---:|
|exec_total_return_20d_net|PRICE_ONLY|-0.0007||-0.1329|-0.1909|
|exec_total_return_20d_net|PRICE_AND_VOLUME|0.0022|0.0029|-0.1307|-0.0351|
|exec_total_return_20d_net|PRICE_VOLUME_INTERACTION|0.0041|0.0019|-0.1273|0.0304|
|exec_total_return_5d_net|PRICE_ONLY|-0.0002||-0.0339|-0.1715|
|exec_total_return_5d_net|PRICE_AND_VOLUME|0.0088|0.0090|-0.0330|-0.0294|
|exec_total_return_5d_net|PRICE_VOLUME_INTERACTION|0.0172|0.0084|-0.0008|0.1679|

## 交互项判读

- `exec_total_return_20d_net`：开发期交互β=-0.30%、p=0.2865；pseudo-OOS重拟合诊断β=-0.77%、p=0.1142。
- `exec_total_return_5d_net`：开发期交互β=-0.26%、p=0.0420；pseudo-OOS重拟合诊断β=-1.07%、p=0.0000。

完整HAC、非重叠样本、Bootstrap、年度稳定性和冻结模型诊断保存在同名JSON中。
