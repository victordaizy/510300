# 行业预期差—ETF 估值研究 V1 首期输出

- 截止日：`2026-08-18`
- 信息截止：`2026-08-18T16:30:00+08:00`
- 最终状态：`NO_VIEW`
- 当前阶段：`M0_FULL_RECORD`
- 边界：`RESEARCH_ONLY / SHADOW_ONLY / NO_POSITION_CHANGE`

## 结论

行业预期差覆盖指数权重 `67.632%`，净加权贡献 `2.369%`，未越过冻结方向阈值 `8.0%`；行业聚合状态为 `BALANCED_NO_EDGE`。

市场流动性门控为 `NO_VIEW`；国家队披露持仓为 `UNOBSERVED`。

ETF 当日观察为 `PASS`，但前瞻成熟度为 `INSUFFICIENT_FORWARD_HISTORY`，因此只允许 `OBSERVE_ONLY`。

失败/停止原因：`NET_EXPECTATION_GAP_BELOW_FROZEN_DIRECTION_THRESHOLD`；`ONE_OR_MORE_LIQUIDITY_DIMENSIONS_UNOBSERVED`；`NATIONAL_TEAM_DISCLOSED_HOLDINGS_UNOBSERVED`；`ETF_FORWARD_HISTORY_INSUFFICIENT`

## 行业底图

| 行业 | 权重 | 60日经营 | 120日经营 | 预期差 | 加权贡献 | 置信度 |
|---|---:|---|---|---|---:|---|
| 电子 | 17.492% | STRONG_IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | MEDIUM |
| 银行 | 11.330% | IMPROVEMENT | IMPROVEMENT | UNOBSERVED | 未观察 | LOW |
| 非银行金融 | 8.737% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | LOW |
| 通信 | 8.549% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | MEDIUM |
| 电力设备及新能源 | 8.129% | MIXED | IMPROVEMENT | UNOBSERVED | 未观察 | LOW |
| 有色金属 | 5.924% | STRONG_IMPROVEMENT | IMPROVEMENT | POSITIVE | 5.924% | MEDIUM |
| 食品饮料 | 5.781% | DETERIORATION | MIXED | UNOBSERVED | 未观察 | LOW |
| 医药 | 4.354% | MIXED | MIXED | UNOBSERVED | 未观察 | LOW |
| 计算机 | 3.757% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | MEDIUM |
| 汽车 | 3.142% | DETERIORATION | MIXED | NEGATIVE | -3.142% | MEDIUM |
| 家电 | 3.017% | DETERIORATION | MIXED | BALANCED | 0.000% | LOW |
| 交通运输 | 2.909% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | LOW |
| 电力及公用事业 | 2.885% | MIXED | MIXED | BALANCED | 0.000% | LOW |
| 机械 | 2.763% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | MEDIUM |
| 基础化工 | 2.223% | STRONG_IMPROVEMENT | IMPROVEMENT | POSITIVE | 2.223% | MEDIUM |
| 石油石化 | 1.537% | IMPROVEMENT | MIXED | BALANCED | 0.000% | LOW |
| 建筑 | 1.330% | STRONG_DETERIORATION | DETERIORATION | NEGATIVE | -1.330% | MEDIUM |
| 国防军工 | 1.274% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | LOW |
| 煤炭 | 1.087% | MIXED | MIXED | UNOBSERVED | 未观察 | LOW |
| 农林牧渔 | 0.917% | DETERIORATION | MIXED | UNOBSERVED | 未观察 | LOW |
| 传媒 | 0.787% | IMPROVEMENT | IMPROVEMENT | BALANCED | 0.000% | LOW |
| 建材 | 0.520% | STRONG_DETERIORATION | DETERIORATION | NEGATIVE | -0.520% | HIGH |
| 钢铁 | 0.457% | STRONG_DETERIORATION | DETERIORATION | NEGATIVE | -0.457% | MEDIUM |
| 房地产 | 0.329% | STRONG_DETERIORATION | STRONG_DETERIORATION | NEGATIVE | -0.329% | HIGH |
| 商贸零售 | 0.135% | MIXED | MIXED | UNOBSERVED | 未观察 | LOW |

## 可审计的行业论点与失效条件

### 电子

- 论点：电子信息制造业产出、收入和利润均高增，指数成分盈利增速也强；但行业盈利收益率偏低，强改善可能已有较多计价。
- 失效条件：后续电子制造业利润或集成电路产量增速显著回落，且指数成分收入增速同步转负。
- 估值方法：行业盈利收益率结合经营增速，不做跨行业统一PE比较
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727, MIIT_ELECTRONICS_4M_20260529

### 银行

- 论点：金融业增加值保持增长且行业估值锚较低，但缺少银行关键经营变量的同口径点时证据，不能判断低估是否为正预期差。
- 失效条件：净息差、资产质量或信用成本数据确认盈利方向与当前改善判断相反。
- 估值方法：盈利收益率和净资产锚仅作初筛，缺少同期净息差与不良生成率
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_GDP_INDUSTRY_20260716, CHINABOND_MOF_CURVE_20260731

### 非银行金融

- 论点：金融业增加值与指数成分盈利均改善，但缺少券商、保险分行业点时经营数据和当前风险承接证据。
- 失效条件：资本市场活跃度和保险负债端数据同步恶化。
- 估值方法：盈利收益率与净资产锚结合金融业增加值，暂不拆券商和保险
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_GDP_INDUSTRY_20260716

### 通信

- 论点：流量、5G连接和信息服务需求增长，指数成分盈利增速强；但电信业务收入承压且估值锚偏高，暂判改善已部分计价。
- 失效条件：数据流量增长不能转化为收入或利润，且成分盈利增速连续回落。
- 估值方法：业务量、收入与行业盈利收益率联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, MIIT_COMMUNICATIONS_5M_20260626, NBS_H1_GDP_INDUSTRY_20260716

### 电力设备及新能源

- 论点：用电与新能源装机增长提供需求，但电气设备利润和利用小时承压，供给扩张与价格压力尚未分清。
- 失效条件：设备利润、价格或利用小时没有企稳，或新增装机不能改善成分企业现金流。
- 估值方法：需求、装机、利用小时和制造业利润联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727, NEA_H1_ELECTRICITY_20260715, NEA_H1_CAPACITY_20260722

### 有色金属

- 论点：行业利润高增且指数成分盈利、收入均改善，盈利收益率仍提供一定定价缓冲，形成首期正预期差候选。
- 失效条件：主要金属价格、库存或工业利润转弱，并使成分盈利预期连续下修。
- 估值方法：行业利润增速与盈利收益率联合判断，后续需补金属价格和库存
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 食品饮料

- 论点：宏观食品饮料零售仍增长，但指数成分盈利与收入同比为负，结构分化使当前定价是否过度悲观无法确认。
- 失效条件：成分收入利润恢复并得到终端量价数据确认，或零售增速明显转负。
- 估值方法：零售需求与指数成分盈利、收入方向交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_RETAIL_20260715

### 医药

- 论点：中西药零售和成分收入仍增长，但成分利润同比明显下降，缺少集采、创新药和器械的分项证据。
- 失效条件：利润降幅持续扩大或分项经营数据确认收入增长质量显著改善。
- 估值方法：药品零售与指数成分收入、利润交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_RETAIL_20260715

### 计算机

- 论点：软件和信息技术服务收入保持增长，指数成分盈利与收入改善；但利润增速弱于收入且盈利收益率偏低。
- 失效条件：软件利润与成分收入同步转弱，或应收和现金流证据显示增长质量恶化。
- 估值方法：软件收入利润与行业盈利收益率联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, MIIT_SOFTWARE_H1_20260730, NBS_H1_GDP_INDUSTRY_20260716

### 汽车

- 论点：汽车零售与制造业利润均明显承压，虽然指数成分盈利仍增长，但行业整体恶化尚未被成分数据充分确认。
- 失效条件：终端销量、价格和行业利润同步企稳，指数成分盈利继续上修。
- 估值方法：终端零售、工业利润与指数成分盈利交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_RETAIL_20260715, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 家电

- 论点：家电零售转弱而指数成分盈利收入接近持平，较高盈利收益率可能已反映部分压力，暂不判负预期差。
- 失效条件：终端零售继续恶化并传导至成分利润，或消费需求显著恢复。
- 估值方法：家电零售、成分盈利与盈利收益率联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_RETAIL_20260715

### 交通运输

- 论点：交通运输业增加值增长，但指数成分利润基本持平，缺少货运、客运和运价分项的及时证据。
- 失效条件：运输量价或成分盈利同步转弱。
- 估值方法：交通业增加值、成分盈利和盈利收益率联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_GDP_INDUSTRY_20260716

### 电力及公用事业

- 论点：全社会用电和新型负荷增长，但电力行业利润及指数成分盈利略降，需求增长尚未转化为清晰利润改善。
- 失效条件：燃料成本和电价变化推动利润方向明确转正或转负。
- 估值方法：用电需求、行业利润与成分估值交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NEA_H1_ELECTRICITY_20260715, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 机械

- 论点：装备制造增加值和指数成分盈利强，但通用、专用设备行业利润接近停滞，改善的广度仍不足。
- 失效条件：订单、行业利润和成分收入共同回落，或利润广度明显改善。
- 估值方法：装备制造增加值、机械行业利润与成分盈利联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_ECON_20260715, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 基础化工

- 论点：化工行业利润和指数成分盈利、收入同步改善，盈利收益率未显示极端昂贵，形成首期正预期差候选。
- 失效条件：产品价差、行业利润或成分收入连续转弱。
- 估值方法：行业利润、成分盈利收入与盈利收益率联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 石油石化

- 论点：油气行业利润改善、成分利润增长，但成分收入下降，价格与销量贡献尚未拆清。
- 失效条件：油价、炼化价差或成分收入利润同步转弱。
- 估值方法：油气行业利润、成分收入利润和盈利收益率联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 建筑

- 论点：建筑业增加值、房地产投资和新开工均弱，指数成分利润收入也下降，低估值尚不足以证明压力已充分计价。
- 失效条件：新开工、资金来源、建筑业增加值和成分收入共同企稳。
- 估值方法：建筑业增加值、地产投资与行业盈利估值交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_GDP_INDUSTRY_20260716, NBS_H1_REAL_ESTATE_20260715

### 国防军工

- 论点：指数成分盈利收入强且高技术制造景气较好，但缺少军工订单、交付和现金流的公开点时证据，盈利收益率也偏低。
- 失效条件：订单、交付或现金流数据与成分盈利方向不一致。
- 估值方法：成分盈利与高技术制造景气作弱交叉，缺少订单交付证据
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_ECON_20260715

### 煤炭

- 论点：宏观煤炭行业利润增长与指数成分利润收入下降相互冲突，缺少煤价、产量和库存同口径证据。
- 失效条件：煤价、库存和成分利润形成一致方向。
- 估值方法：工业行业利润与指数成分盈利收入交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 农林牧渔

- 论点：农业增加值增长但农副食品加工和指数成分盈利偏弱，缺少主要农产品价格与成本的及时证据。
- 失效条件：产品价格、养殖成本和成分利润形成一致改善或恶化。
- 估值方法：农业增加值、农副食品加工利润与成分盈利交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_ECON_20260715, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 传媒

- 论点：指数成分盈利收入改善并受信息服务景气支持，但缺少广告、游戏、影视等分项经营证据。
- 失效条件：行业分项收入或成分盈利连续转弱。
- 估值方法：信息服务增加值与指数成分盈利收入作弱交叉
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_GDP_INDUSTRY_20260716

### 建材

- 论点：地产投资、新开工、建筑装潢零售和非金属矿物制品利润同步走弱，指数成分也未显示足够反转。
- 失效条件：地产施工销售、建材零售和行业利润至少两项连续企稳。
- 估值方法：地产投资、建材零售、非金属材料利润与成分盈利联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_REAL_ESTATE_20260715, NBS_H1_RETAIL_20260715, NBS_H1_INDUSTRIAL_PROFIT_20260727

### 钢铁

- 论点：钢铁行业及指数成分利润均明显下降，建筑和地产需求偏弱，低估值不足以确认盈利已见底。
- 失效条件：钢材价差、库存、建筑需求和行业利润形成一致改善。
- 估值方法：钢铁行业利润、建筑需求与成分盈利联合判断
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_INDUSTRIAL_PROFIT_20260727, NBS_H1_GDP_INDUSTRY_20260716

### 房地产

- 论点：开发投资、销售、新开工和资金来源全面下降，行业增加值偏弱，指数成分盈利收入也恶化。
- 失效条件：销售、资金来源和新开工至少两项持续转正，且成分盈利同步修复。
- 估值方法：投资、销售、新开工、资金来源与成分盈利联合判断，不使用负PE
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_REAL_ESTATE_20260715, NBS_H1_GDP_INDUSTRY_20260716

### 商贸零售

- 论点：批零增加值增长但社会消费品零售偏弱，且沪深300该行业只有一个成分，无法把宏观行业直接映射为指数暴露。
- 失效条件：消费总量、品类广度和唯一成分盈利形成一致方向。
- 估值方法：社会消费品零售、批零增加值与成分盈利交叉核对
- 来源：LOCAL_SECTOR_PIT_20260731, NBS_H1_RETAIL_20260715, NBS_H1_GDP_INDUSTRY_20260716

## 市场与尾部边界

- `funding_liquidity`：`NO_VIEW`；只有月末国债曲线，缺少同一截止时点的资金价格、信用和融资条件，不能判为宽松或收紧。
- `equity_liquidity`：`NO_VIEW`；基金份额、两融和成分资金流只到2026-08-12，不能代表2026-08-18。
- `risk_bearing`：`NO_VIEW`；虽有510300价格和成交额，但缺少同口径宽度、冲击成本和风险承接证据，价格上涨不能替代流动性判断。
- `national_team`：`UNOBSERVED`；没有带报告期和公开日的当前国家队持仓库存；不得从ETF份额、成交或折溢价反推主体身份。

## 安全开关

- `position_mapping_enabled`：`false`
- `order_generation_enabled`：`false`
- `broker_connection_enabled`：`false`
- `live_trading_enabled`：`false`
- `current_holdings_read_enabled`：`false`

首期输出是冻结待验证的行业预期差底图；NO_VIEW 不等于看空，不构成仓位、订单或实盘建议。
