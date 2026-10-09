# 最终交付身份

文件：510300_dynamic_million_20261009.zip

字节数：139255490（约132.8 MiB）

SHA-256：d4001627d731e56b8a26aeac30d75f1ef0939b48b3c9eec4bc368b73d2344056

155个内容文件与DELIVERY_MANIFEST.json；ZIP逐文件CRC核验通过。独立复制的交付目录再次运行32个重点账户，全部daily/orders/cycles数组与首次结果一致。此复跑不算新策略或新独立市场样本。

包内包括一百万场景的全部21项指标（四个250000行NPY）、全部128位持仓指纹、32个重点账户完整日账/订单/周期、全部动态路由与季度元选择代码、必要原数据和PMI视图、预处理输入、季度选择记录、块置信区间及本批选择修正、故障记录、核验。大体积原始压力块分片未重复打包，保留1000分片身份清单，完整重跑会生成。没有声称保存了一百万份逐日日账。

默认核对32个重点账户：python dynamic_million/replay.py --output verification_runs/check_01

完整重跑一百万：python dynamic_million/replay.py --output verification_runs/million_01 --full-search

导出全部指标为CSV：python dynamic_million/export_metrics.py --output exports/check_01

要求新输出路径，不覆盖首次结果。依赖固定在dynamic_million/requirements.txt。原数据/CSV均在包内，不联网、不连接券商、不下单。

另附510300_dynamic_million_selected_results.csv、510300_dynamic_million_confidence.csv、510300_dynamic_million_report_20261009.md。

当前0项通过10%年化、暂用1.2夏普、10%回撤及可靠性门，goal_achieved=false。所有历史仍为development。GitHub保存本轮登记、早期选择、结果及交付身份；新代码和完整指标在会话压缩包，未冒称全部已提交Git主树。