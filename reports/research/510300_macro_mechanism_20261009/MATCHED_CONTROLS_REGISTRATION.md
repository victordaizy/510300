# 宏观增量的同覆盖纯价格对照：运行前登记

主批56规则的两时期账户已经观察。本补充用于区分宏观变量增量、可用日期覆盖和训练池差异，不称新的独立确认，不用于选择新的最好策略。

对VALUE、ORDERS_FUNDING、REAL_CREDIT、NONGOV_CREDIT、COST_FUNDING、HOUSING六组及21/63日两个期限，各建立一个纯价格模型。只删除宏观特征列，完全复用对应主模型的月度训练原点、成熟日、压力收益标签、重叠权重、预测覆盖、标准化方法、Ridge(alpha=12,solver=svd)和缺失回退。纯价格特征仍为21/63/252日趋势及126日回撤。每个拟合仅用当前训练池重新估计价格列均值方差与系数。

12个配方×3个原仓位映射×两时期×两费用=144个匹配对照账户。全部原资金、交易成本、风险限制、分红和空仓日保持，禁止把重复控制当144个独立市场证据。主候选及早期选择不改变，失败结果全部保留。

执行代码matched_price_controls.py SHA256=fb5ac7867878b85c00cc82f235dbaf71906490b071cea89f44f95e55dd4d4d8f。

goal_achieved=false；independent_validation=NOT_ESTABLISHED；orders_authorized=false。