# 510300 EPS明示修正去重与增长分歧来源检查

先读 reports/research/510300_eps_explicit_revision_novelty_diagnostic_v1/研究结论与下一步.md 和 GPT审阅提示.md。两份明示前值叙述未增加既有原表之外的预测数字；两机构增长分歧在50个源月份满足预定覆盖条件。预测价值未计算，0新拟合、0新账户，夏普1.2和年化10%的共同目标仍未达成。

包含7份完整原PDF（94页），本轮重新提取其中用于核实的14页；包含83400行保存机构输入、41700行配对结果和139行月度统计。自包含范围为本轮原件对应与从冻结保存输入重算统计，不重建全部上游券商原件事实库。

Windows离线复核：完整解压到新目录，在解压目录的PowerShell中，使用已安装numpy、pandas、pyarrow、pdfplumber的Python运行：python .\scripts\verify_510300_eps_novelty_and_disagreement_packet_20260914.py --receipt-directory .\本轮离线复核。回执目录必须尚不存在。脚本核对FILE_INDEX，复算两项研究，不联网、不拟合、不生成账户。不要执行run模式或旧模型入口。

FILE_INDEX.csv列出除自身外全部成员的路径、大小和SHA-256。两套代码与配置均由各自claim冻结，0新收益标签。原始用户附件、范围边界、旧缺失、查重检索缺失文件说明、下一步和停止条件均保留。未上传，未收到外部GPT评审；数值复算和ZIP检查不是交易目标证明。
