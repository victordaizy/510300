# 国家统计局全国月度数据采集

本任务按2026年10月4日用户确认的范围执行：全部可获取历史，先只做全国。工作内容是官方数据采集、整理及行业与经济观察。

## 数据范围

数据来源为新版国家数据全国月度目录，以及国家统计局全国数据发布栏目。全国月度目录包含14类、605张末级数据表、14,604个指标条目。指标条目涵盖金额、数量、同比、累计、比较基数和不同年份的统计分类版本；条目数量不是独立经济变量数量。

工业数据包括工业增加值、产品产量、企业数量、收入、成本、费用、利润、资产、负债、应收及库存等官方表内字段。其他类别包括价格、能源、固定资产投资、服务业、就业、房地产、国内贸易、对外经济、交通运输、邮电通信、PMI、财政和金融。金融目录目前提供M0、M1、M2的期末值及同比增速。

GDP、居民收入等季度或年度发布材料若出现在同一全国发布目录，保留完整原文和附表，并标记原始频率；本次没有扩展成季度、年度、分省或城市数据库。全国房价发布稿中的城市附表随原稿完整保存。

## 文件入口

最新完成快照由 `C:\Users\戴周阳\Documents\New project 8\data\nbs_monthly\LATEST.json` 指向；本次目录为 `C:\Users\戴周阳\Documents\New project 8\data\nbs_monthly\runs\20261004_national_full`。

- `阅读说明.md`：覆盖范围、读取方式和实际缺口。
- `国家统计局月度.sqlite`：指标定义、月度数据、历史覆盖、发布稿索引及附表单元格。
- `全国月度数据.parquet`：完整月度数据，适合Python或数据库分析。
- `全国月度数据.csv.gz`：压缩CSV完整导出；较大文件建议按指标查询。
- `指标字典.csv`、`历史覆盖.csv`、`分类覆盖.csv`：定义及逐项覆盖。
- `最新指标快照.csv`：每个指标最新有值月份、原始口径及是否早于该类别最新月份。
- `官方发布索引.csv`、`release_text`、`release_tables`、`attachments`：官方发布文本、原始表格结构和附件。
- `附件来源索引.csv`、`官方附件缺口.csv`：各附件实际保存结果及官网失效链接；失效链接不计入成功保存数量。
- `全国经济观察.png`、`工业行业趋势.png`：本地观察图；对应数据另存CSV。

`monthly_with_definitions` 是SQLite中方便查询的视图。发布附表单元格包含坐标、合并跨度、原始文本及可识别数字；保留表头原文，未将所有单元格强行解释成同口径时间序列。

## 更新与查询

📁 C:\Users\戴周阳\Documents\New project 8\scripts\collect_nbs_monthly.py

```powershell
Set-Location -LiteralPath 'C:\Users\戴周阳\Documents\New project 8'
.\.venv\Scripts\python.exe -X utf8 .\scripts\collect_nbs_monthly.py --phase all
```

日期默认采用北京时间，新的一次运行生成独立快照。若中断，可用同一个 `--run-id` 继续，已成功保存的数据直接复用。HTTP原始响应在raw目录压缩保存，每条请求有对应来源记录。若官方返回要求启用JavaScript的页面，程序使用正常Microsoft Edge浏览器读取；没有登录要求的公开接口才进入数据采集。

📁 C:\Users\戴周阳\Documents\New project 8\scripts\query_nbs_monthly.py

```powershell
Set-Location -LiteralPath 'C:\Users\戴周阳\Documents\New project 8'
.\.venv\Scripts\python.exe -X utf8 .\scripts\query_nbs_monthly.py --keyword 新订单
.\.venv\Scripts\python.exe -X utf8 .\scripts\query_nbs_monthly.py --category 金融
.\.venv\Scripts\python.exe -X utf8 .\scripts\query_nbs_monthly.py --indicator-id ef1b1765960d45a29b4d7c4ca91be916 --start 202101
```

📁 C:\Users\戴周阳\Documents\New project 8\scripts\plot_nbs_monthly.py

```powershell
Set-Location -LiteralPath 'C:\Users\戴周阳\Documents\New project 8'
.\.venv\Scripts\python.exe -X utf8 .\scripts\plot_nbs_monthly.py
```

采集依赖requests、beautifulsoup4、pandas、pyarrow；浏览器读取需要playwright和已安装的Microsoft Edge；绘图需要matplotlib。当前项目环境已具备这些依赖。

## 客观数据分析

2026年10月4日按用户要求新增客观分析，输出到 `C:\Users\戴周阳\Documents\New project 8\data\nbs_monthly\analyses\20261004_national_full_objective_v1`。报告仅使用数据与统计定义，分别列示原始事实、可重算计算和解释边界，不引用机构或官方文字解读的景气结论。

宏观部分按25项主要指标覆盖14类；工业部分比较全部41个行业同一累计区间的增加值、营业收入、利润。公布稿和数据库版本分别标记。利润同比为负与利润总额为负分别处理；缺失的可比基数不填补，M1和M2差值分别分解到两个指标。

📁 C:\Users\戴周阳\Documents\New project 8\scripts\analyze_nbs_monthly.py

```powershell
Set-Location -LiteralPath 'C:\Users\戴周阳\Documents\New project 8'
.\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_nbs_monthly.py --run-id 20261004_national_full
```

`客观数据分析.md` 是分析入口；所有41行业、宏观近12个月读数及PMI算术分解均有对应CSV或JSON。脚本只读来源数据库，分析结果单独保存。

## 多年同比、环比和行业分析

2026年10月4日根据用户对比较深度的要求，新增多年分析，作为当前分析入口。结果位于 `C:\Users\戴周阳\Documents\New project 8\data\nbs_monthly\analyses\20261004_national_full_multiyear_v2`，工作簿位于 `C:\Users\戴周阳\Documents\New project 8\outputs\nbs_multiyear_20261004\全国月度多年同比环比.xlsx`。

全指标计算保留全部可获取历史，包括14,604个目录条目及1,237,804个有值读数；空目录条目仍列在历史覆盖文件内。主要指标比较覆盖14类中的61个系列，默认展示2017—2026年同月读数、逐年1—12月路径，以及原口径的同比、环比和三个月变化。原始金额、数量的多年复合变化与官方可比增速分别列示。

工业、零售和投资的环比来自官方发布稿季调环比表，保存2,082个公布版本读数，按数据月份选择214个最新已保存修订值，不从同比读数相除得到环比。CPI、PPI环比由官方上月=100指数减100得到。累计总量只在同年相邻月份机械差分，不跨年，不将1—2月合并数据拆成单月。

行业分析覆盖2018年至今相同分类的全部41个行业、95个有值历史月份，比较增加值、收入、利润、利润率、应收与库存。利润普通同比要求上年可比利润基数为正，其他情况单列，数据库原百分比保留。CPI总指数跨基期只并列公布读数；M1旧口径和2025年起的新口径分别标注。

📁 C:\Users\戴周阳\Documents\New project 8\scripts\analyze_nbs_multiyear.py

```powershell
Set-Location -LiteralPath 'C:\Users\戴周阳\Documents\New project 8'
.\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_nbs_multiyear.py --run-id 20261004_national_full --display-years 10
```

`多年同比环比分析.md` 是报告入口，`同比环比多年同月综合比较.csv` 同时列出各年相同月份的官方同比与真实环比。报告另附多年宏观曲线、六年同月路径、季调环比与三个月复合变化、41行业多年同期图。全指标历史计算结果保存为Parquet，便于进一步按类别或指标查询。分析脚本只读采集快照，全部结果存到独立分析目录。

## 口径和时间

下载版本与最初公布版本分别处理。官方数据库可能修订历史数值，当前接口没有提供所有历史数值的首次公布时间和历次版本；相应字段保留未知。发布稿从原页PubDate或发布页眉读取公布时间，2023年网站迁移后的URL日期不代替原公布日期。部分旧信息公开页只有成文日期，此字段单列保存，原页公布时间仍保留未知。

明确空字符串、接口省略的单元格和零值分别记录；不前向填充，不从累计增速反推当月增速。旧行业分类和价格分类目录分别保存。最新快照中的相邻数值差只在连续月份之间计算，含义取决于指标自身口径。

新闻稿可先于数据库更新。国家统计局2026年发布日程说明，数据库通常在发布后约三个工作日更新进度指标。因此原稿已有新月份而数据库暂时空缺时，两种来源分别保存，避免覆盖版本身份。

官网当前分页目录的历史范围有限；较早原稿来自本项目已经定位的官方原页。本次数据库历史覆盖和发布稿覆盖分别列示，不把有限发布稿目录称为完整历史首次发布档案。数据库查询全空的条目保留清单，未自行补值。

旧信息公开页的部分图片已迁移。对应的现行官方原稿也列入补充索引：[2018年全国工业企业利润](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900227.html)、[2020年1—2月工业企业利润](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900667.html)。旧链接的404记录和迁移页实际保存的原件各自保留。

官方入口：[新版国家数据](https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData)、[全国数据发布](https://www.stats.gov.cn/sj/zxfb/)、[2026年发布日程](https://www.stats.gov.cn/xw/tjxw/tzgg/202512/t20251224_1962137.html)。
