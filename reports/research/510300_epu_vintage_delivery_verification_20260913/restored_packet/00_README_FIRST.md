# EPU历史版本信息增量V1：阅读导航

本轮已完成并拒绝这一固定用途：压力年化9.687%、净夏普0.916，整体目标尚未实现。请先读02_研究结论与下一步.md、01_GPT_REVIEW_PROMPT.md，再查docs/510300_EPU_VINTAGE_INCREMENT_V1_PROTOCOL.md和配置。

主要证据在reports/research/510300_epu_vintage_increment_v1：USER_REQUEST.md、deduplication.json、data_preflight.json、freeze_manifest.json、models.json、predictions.parquet、accounts完整账本及saved_verification_receipt.json。图为EPU压力账户与历史信息.png。原始EPU下载与308个历史值交叉核对在data/raw/market/510300_epu_vintages_source_v1/20260913T231935_0800。

本包包含本轮全部直接输入及独立冻结快照、代码、测试、全部模型与账户、原始EPU公开表单与ZIP、官方方法、分红直接凭据、价格原始值与更正。旧IF/AH只包含去重和结果背景，不提供整个旧研究闭包，也不要求重跑它们。排除虚拟环境、缓存和全项目其余研究。

FILE_INDEX.csv列出除索引自身以外所有成员的字节数与SHA-256；外部同名.delivery.json记录最终ZIP哈希。包CRC、哈希、索引和解压后核对只证明所述保存材料一致，不证明科学有效或已获外部GPT审阅。

Windows复查：在解压目录准备Python3.13并安装REQUIREMENTS_REVIEW.txt依赖，然后执行`python scripts/run_510300_epu_vintage_increment_v1.py verify`。该命令从已存模型、预测、历史版本、账本和月份抽样重算，不训练、不新建账户、不重新抽样、不联网。合成测试命令是`python -m pytest tests/test_epu_vintage_increment_v1.py -q`。

run已经消耗唯一机会，禁止删claim或改冻结文件重跑。retrieve_510300_epu_public_vintages_v1.py是此次公开表单请求的可复现来源说明代码，未在本轮追加运行；它只可创建新的时间戳目录，不改变本包冻结输入，也不表示已启动未来每日任务。本包未上传、未获外部审阅，不包含交易指令。
