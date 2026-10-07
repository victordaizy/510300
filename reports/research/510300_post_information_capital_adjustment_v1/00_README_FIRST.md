# 阅读顺序

先看《研究结论.md》和figures下四组图，再看《需求完成与缺口.csv》、protocol.json、results及verification.json。

本包包含独立复算所需完整日行情、104月正式发布及原始HTML、六例信息/新旧收益、政策背景及原政策事件表、历史裁决、冻结文件清单和代码。六份周报仅保留必要数字摘录、公开链接、页码及原PDF哈希，不分发全文。资金2220条原始响应未重复装入；该项仅据直接关联的既有来源裁决判断缺口，不宣称重新核查全部响应。

可在已安装依赖的Python环境中从解压目录运行code/post_information_capital_adjustment_v1.py，参数--study-dir为解压目录、--output-dir为新的复算目录；验证命令使用code/verify_post_information_capital_adjustment_v1.py --study-dir 解压目录。绘图加--plot。不得对已有冻结结果目录运行建库脚本。冻结/打包脚本用于原仓库环境，离线数值复算与核对脚本不依赖原仓库。

模型、账户和退出比较均为NOT_RUN，代码没有把这些未运行阶段伪装成已实现的交易系统。当前没有订单、外部上传或自动收集任务。

背景图可由code/plot_post_information_context_v1.py --study-dir 解压目录 --output-dir 新图目录重绘。sources/104个月原始页面定位.csv为本包的官方来源路径；冻结CSV保留旧路径只作历史来源记录。
