# 交付导航

先读研究结论.md；实际输出在results/保存模型的历史重建输出.csv，特征在results/补齐的历史事件特征.csv。data_boundary.json给出三日增量、所需五日窗口及四个缺失日期。

inputs包含原保存模型、原待处理事件、原16个成熟事件、两份本地行情、分红和日历，以及原关闭结论与来源回执。freeze.json固定计算输入。protocol.json说明本轮是事后历史重建，不是事前已发布的预测。

用本地Python环境运行code/local_pending_m1_completion_v1.py的verify子命令，并以--root指定解压目录，即可只从包内文件复算。该入口不拟合、不下载、不运行账户。requirements.txt列出环境依赖。文件大小和SHA-256见FILE_INDEX.csv。

local_runtime_observation.json记录检查时任务禁用状态与匹配研究入口的Python进程，不是持久后台等待句柄。goal_continuation_state.json说明当前进度及未解决的数据条件；它不会修改系统目标状态。
