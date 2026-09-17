# 510300增长分歧增量试验：预测门失败，账户未运行

阅读顺序：reports/research/510300_eps_growth_disagreement_increment_v1/研究结论与下一步.md → GPT审阅提示.md → docs/510300_EPS_GROWTH_DISAGREEMENT_INCREMENT_V1_PROTOCOL.md → result.json、全部预测和训练收据。原始用户附件与本轮范围在同一结果目录。

一个新变量、72次拟合、36个共同预测、33个成熟评价原点。M1均方误差增加5.97%，没有改善涨跌识别，拒绝本用途。账户NOT_RUN_PREDICTION_GATE，年化和夏普未计算，总目标未达成。

包含1906份报告的事实及采集元数据，但完整原PDF仅6份固定样本；83400行旧机构输入与本轮全部来源配对、标签、保存模型、预测、固定抽样、ETF价格分红日历等均包含。能够从保存输入独立复算本轮，不重建上游全量原始研报提取。不要把事实文件哈希当成会计口径完全统一或预测有效的证明。

Windows离线复核：完整解压到新目录，用已经安装requirements-review.txt所列运行库的Python，在解压根目录PowerShell运行：python .\scripts\verify_510300_eps_growth_disagreement_packet_20260914.py --receipt-directory .\本轮离线复核。该新回执目录必须不存在。脚本先核对FILE_INDEX，再重算来源配对、标签、全部月末状态、保存模型与固定区块结果；不联网、不重新拟合、不新增随机抽样、不运行账户。不要执行run模式或旧账户入口。

FILE_INDEX.csv列出除自身外全部成员的路径、大小和SHA-256；protocol_freeze.json另绑定研究开始前的输入、代码和协议。源检查、合成测试、未运行账户、局限与下一步都保留。本包尚未上传或获得外部GPT评审，复核通过不表示夏普1.2和年化10%目标实现。
