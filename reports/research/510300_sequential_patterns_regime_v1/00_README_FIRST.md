# 三类连续形态与启停 V1：阅读导航

先读研究结论.md，再看账户对比.png、protocol.md、user_request_full.md（用户原文）与user_request.md（结构摘要）。用户目标是改研究主线，当前目标年化10%、夏普1.2、20万元、回撤风险目标10%，年度次数下限取消。

关键结果：仅形态近期基础夏普1.281，压力1.185；完整机制未开仓，夏普未定义。最大盈利周期贡献约四分之三，无法据此确认当前有效。历史回放不是独立样本，当前行情仍止于9月16日。

results/episodes.csv及steps.csv包含所有准备、失败、超时与确认；signals.csv、trigger_decisions.csv、training_events.csv说明当时为何允许或拒绝。accounts目录保存三时段×两成本×六方案共36账户，不是36种搜索策略。annual.csv披露次数和逐年收益，无次数下限。

results/paired_bootstrap_valid.csv是最终可解释的统计区间表；paired_bootstrap.csv仅保留原始数值痕迹，其3项夏普差异由于FULL零波动已作废，详见statistical_validity.json。不能把作废值抄成研究发现。

freeze.json固定本轮结果运行前的协议、输入与代码。source_manifest.json对应原始路径；inputs与sources提供直接数值输入、来源回执和14份官方分红文件。code保存引擎、七项测试、结果整理、复核与每日观察脚本。

每日观察说明.md与forward/scheduler_receipt.json说明真实Windows任务。无当前资料时只留下NO_VIEW，不代表已新增有效观察。观察入口不会恢复行情采集。

可复制01_GPT_REVIEW_PROMPT.md让外部审阅者批评研究。ZIP根目录FILE_INDEX.csv是本包成员大小与哈希的权威索引，独立交付回执在ZIP外。结构校验和保存结果复算不等于已完成外部GPT审阅。

只读复算：在解压目录使用安装requirements.txt依赖的Python运行 `python code/verify_sequential_patterns_delivery_v1.py --root .`。该命令不拟合、不下载、不新建账户、不重新抽样。年度完整性说明见results/annual_metadata_correction.json；年度次数和收益未改变。
