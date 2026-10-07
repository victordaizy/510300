阅读顺序：研究结论.md → 同样本散点图及走势对照 → protocol.json → 逐事件特征与随后收益.csv → 固定逐时点预测.csv及固定模型记录.json → 月度意向人工准入.csv、全部月份覆盖与缺口.csv和来源链接与下载记录.json → 审阅提示词.md。

本包数值验证命令：python research/private_manager_intent_delivery_v1.py verify --root .
绘图：python research/private_manager_intent_diagnostic_v1.py plots
固定历史重算：python research/private_manager_intent_diagnostic_v1.py analyze（会重新计算模型；不是只读验证）。

原始抓取全文和PDF仅本地保存，商业全文及其长摘录不进入审阅包。提供原链接、哈希、日期和抽取事实；离线可复核从已保存字段到标签、预测和误差的计算，不能离线逐份认证原网页历史版本。代码的fetch/extract/curate属于源资料准备阶段，需要包外原文，不是完整离线步骤。

早期案例仅附摘录，包内未重新重建持有人研究。源资料缺口和当前研究未完成均明确保留。
