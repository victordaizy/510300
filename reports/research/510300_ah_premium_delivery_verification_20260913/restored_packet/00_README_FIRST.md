# A/H信息增量V1：先读这里

本轮已完成并拒绝：未确认新信息增量，压力账户年化2.228%、夏普0.420，10%/1.2尚未实现。

阅读顺序：02_研究结论与下一步.md → 01_GPT_REVIEW_PROMPT.md → reports/research/510300_ah_premium_increment_v1/USER_REQUEST.md → docs/510300_AH_PREMIUM_INCREMENT_V1_PROTOCOL.md及config同名配置 → 数据预检/冻结/模型预测/账户/核对回执。图在该研究目录的压力账户与持有比例.png。

本包包括本轮全部直接数据与代码、模型、完整账户、原始A/H公开网页数据和来源失败回执、14次分红凭据、原价格与三处更正、必要近邻背景。旧IF及全球信息资料仅供去重背景，不包含其完整旧数据闭包，也不要求重跑。排除整个项目的其余研究、虚拟环境、缓存和机器配置。

根FILE_INDEX.csv索引除自身外的每个成员，记录字节数和SHA-256。同名外部.delivery.json记录ZIP最终哈希；结构通过不等于科学有效或外部GPT审阅。本次没有上传、外部审阅或安全审计。

Windows核对：在解压目录准备Python3.13和REQUIREMENTS_REVIEW.txt列出的依赖，然后执行`python scripts/run_510300_ah_premium_increment_v1.py verify`。该模式只检查保存模型/预测/账本/抽样，不拟合、不下载、不新增账户。合成测试命令为`python -m pytest tests/test_ah_premium_increment_v1.py -q`。

run已使用一次机会，run_claim.json必须保留；禁止删除claim、修改冻结输入或调参重跑。本包的采集脚本是来源说明，不要运行它们覆盖冻结输入。retrieve_510300_ah_official_source_snapshot_v1.py仅能另存新时间戳目录，存在脚本不代表已启动每日采集。图与报告根据保存结果生成，没有新增策略变体。
