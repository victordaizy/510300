# 阅读导航

本包交付“资料扩样及样本可行性”阶段。不是已通过策略，也不是账户回测。

1. `研究结论.md`：本次补齐10个月、仍然不能进入预测验收的具体原因。
2. `用户需求.md`、`mandate.json`：510300、20万元、夏普1.5、回撤10%、低频与空仓约束。
3. `source_plan.json`：继承原协议，只补来源；固定追加检索范围与停止条件。
4. `新增月份与来源.csv`、`new_admitted_sources.json`：新增10个月的数值、原文链接、哈希和疑点。
5. `combined_admitted_sources.json`、`全部月份覆盖与缺口.csv`：43个事件与152个月的完整母集。
6. `parent_snapshot/`：此前V1准入资料、冻结协议和结果，保留原33个事件不变。
7. `result.json`、`source_freeze_receipt.json`、`来源下载回执_无正文.json`：状态、证据身份、成功与失败记录。
8. `GPT审阅提示词.md`、`EXCLUSIONS.md`：复核任务及范围限制。

`code/package_private_manager_intent_extension_v2.py --verify-root <解压目录>` 只依赖Python标准库，可离线核对月份母集、旧记录保持、计数上限和索引。来源抓取/数值核对脚本另在code目录，依赖requests和beautifulsoup4；在线页面可能变化，全文缓存保留本地未打包，离线复核不声称重新验证网页原文。
