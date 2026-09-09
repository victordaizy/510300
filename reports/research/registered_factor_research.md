# 510300预注册因子时间序列研究

## 治理口径

- 这是单标的时间序列研究，不使用横截面Rank IC。
- D20为主目标，D5为敏感性目标；标签包含分红、滑点、佣金、最低佣金和整手约束。
- 同时报告重叠样本、错位非重叠样本、HAC/Newey-West和移动区块Bootstrap。
- 历史时间顺序验证仅称pseudo-OOS；没有严格未见历史留出集。
- 评级是证据强度，不是实盘批准。

## 假设结果

|ID|因子|目标|预期方向|开发Spearman|pseudo-OOS Spearman|非重叠中位数|成本后有利-不利|BH q值|得分|评级|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
|H001|510300过去5日含分红累计收益|exec_total_return_20d_net|negative|0.0387|-0.2625|-0.1636|0.0487|0.0519|6|STRONG|
|H002|沪深300过去20日累计收益|exec_total_return_20d_net|positive|0.0364|-0.2782|-0.2485|-0.0479|0.2612|1|REJECTED|
|H003|沪深300二十日均线相对六十日均线|exec_total_return_20d_net|positive|-0.0426|-0.4001|-0.4424||0.1461|0|REJECTED|
|H004|沪深300六十日价格区间位置|exec_total_return_20d_net|positive|0.1024|-0.4624|-0.4788|-0.1094|0.0642|1|REJECTED|
|H005|沪深300二十日实现波动率|exec_total_return_20d_net|negative|0.1924|0.0224|0.0424|0.0730|0.5587|1|REJECTED|
|H006|沪深300短中期波动率比|exec_total_return_20d_net|negative|-0.0581|0.0939|-0.0061|-0.0309|0.2612|2|INCONCLUSIVE|
|H007|510300二十日价格与成交量确认|exec_total_return_20d_net|positive|-0.0998|0.0865|0.2121|0.0055|0.4163|5|WEAK|
|H008|510300收盘价相对当日VWAP|exec_total_return_20d_net|positive|-0.0218|-0.0206|-0.0545|-0.0267|0.0665|0|REJECTED|
|H009|沪深300官方PE五年滚动百分位|exec_total_return_20d_net|negative|-0.2802|-0.4999|-0.4788|0.2452|0.0520|6|WEAK|
|H010|沪深300PB五年滚动百分位|exec_total_return_20d_net|negative|-0.2741|-0.4979|-0.4788|0.1989|0.0520|6|WEAK|
|H011|510300过去5日含分红累计收益|exec_total_return_5d_net|negative|-0.0245|-0.1393|-0.1321|0.0063|0.2984|6|STRONG|
|H012|沪深300过去20日累计收益|exec_total_return_5d_net|positive|0.0313|-0.2031|-0.1930|-0.0184|0.2984|1|REJECTED|
|H013|沪深300二十日均线相对六十日均线|exec_total_return_5d_net|positive|-0.0173|-0.1035|-0.0820||0.5574|0|REJECTED|
|H014|沪深300六十日价格区间位置|exec_total_return_5d_net|positive|0.0366|-0.1833|-0.2015|-0.0153|0.2984|1|REJECTED|
|H015|沪深300二十日实现波动率|exec_total_return_5d_net|negative|0.1033|0.1221|0.1471|0.0011|0.8831|1|REJECTED|
|H016|沪深300短中期波动率比|exec_total_return_5d_net|negative|-0.0550|-0.1932|-0.1697|0.0045|0.5574|6|STRONG|
|H017|510300二十日价格与成交量确认|exec_total_return_5d_net|positive|-0.0893|0.0738|0.0735|0.0013|0.2984|5|WEAK|
|H018|510300收盘价相对当日VWAP|exec_total_return_5d_net|positive|-0.0349|0.0105|0.0181|0.0064|0.8831|3|INCONCLUSIVE|
|H019|沪深300官方PE五年滚动百分位|exec_total_return_5d_net|negative|-0.1541|-0.2680|-0.2488|0.0296|0.2984|5|WEAK|
|H020|沪深300PB五年滚动百分位|exec_total_return_5d_net|negative|-0.1331|-0.2579|-0.2446|0.0817|0.2984|5|WEAK|

## D20证据候选

- `ETF_TR_5D`：510300过去5日含分红累计收益，STRONG（6/7）。
- `CSI300_PE_PCTL_5Y`：沪深300官方PE五年滚动百分位，WEAK（6/7）。
- `CSI300_PB_PCTL_5Y`：沪深300PB五年滚动百分位，WEAK（6/7）。
- `ETF_PRICE_VOLUME_CONFIRM_20D`：510300二十日价格与成交量确认，WEAK（5/7）。

详细条件分桶、年度稳定性、HAC和Bootstrap结果保存在同名JSON报告中。
