# R03｜同指数ETF总需求与迁移

假设：510300份额减少时，体系流出与产品迁移是否对相同合法跨日合同有不同增量。

旧用途及结果：旧官方周度10产品2,870行/287快照的五规则冻结失败；旧三只沪市300产品迁移HIGH/LOW主压力夏普0.3703/0.1394、年化1.1912%/0.3126%，未过原深化门。

现有字段：现有2021-01-08至2026-08-14周度份额；本次920只沪市ETF的2026-09-30份额，含510300/510310/510330。

未解决：920不是920只沪深300产品；名单须按跟踪指数和上市史核对。完整同指数产品范围、深市份额、同日可比NAV和历史发布时间仍缺；TOT_VOL是份额，不是净资金或NAV。

本次处置：`INTAKE_COMPLETE_FULL_BASKET_SOURCE_NOT_ADMITTED`。

下一最小步骤与停止线：补齐事前产品全集、NAV/份额调整及真实接收连续记录，再注册整体需求相对单产品的唯一增量；不直接重复旧三只迁移。

若准入，主对照为同期限D日线字段；先固定信息可得原点，再比较共同日期预测及完整日历净账户。不得把D-native原二日模型直接移植为不同期限的已验证对照。模型和主目标见experiment_specs.json；当前没有新拟合、标签或账户。

直接证据：

- reports\research\510300_factor96_rapid_structure_v1\protocol.json
- reports\research\510300_factor96_rapid_structure_v1\result.json
- reports\research\510300_cross_etf_forced_flow_binary_screen_v1.json
- reports\research\510300_fund_share_publication_receipts_closure_v1\result.json
