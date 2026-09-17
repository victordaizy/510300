# 510300两对研报归母口径核实V1

先读 reports/research/510300_eps_profit_attribution_pair_adjudication_v1/研究结论与下一步.md，再读同目录GPT审阅提示.md。两对特定归母语义可以核实，条件覆盖63.07%→66.20%，整体修正方向仍不确定。夏普1.2和年化10%的共同目标尚未达成。

本包自包含本轮四份原件、两对语义核对以及从保存基线到条件覆盖的增量计算。包含原PDF、HTML、平台目录、旧事实和日期、新20页文本、8页原图、337行条件结果、990行原保存基线、代码与冻结claim。附三份基金原年报及上轮实际解压复算回执。完整659份券商原件的整体基线重建不在本包范围；其完整上轮包名、SHA-256和大小见upstream_archive_identity.json，不将外部引用当作本包文件已经齐全。

Windows离线复核：完整解压到一个新目录，在解压目录打开PowerShell。使用已有包含numpy、pandas、pyarrow、pdfplumber的Python，实际版本在研究目录requirements-review.txt。运行 python .\scripts\verify_510300_eps_profit_pair_packet_v1_1.py --receipt .\本轮离线复核回执.json。此入口核对FILE_INDEX及冻结输入，重新提取四份原PDF共20页，按原顺序和业务键逐值核对337行及保存结果。不要调用run模式；它不是复核入口。原V1验证器的行索引失败和V1.1处理依据都保留。

FILE_INDEX.csv覆盖除自身外的所有成员，含相对路径、字节数、SHA-256。结构与数值复核不是外部GPT评审、交易收益证明或新的仓位操作。本包不包含虚拟环境、Git对象、凭据或其他无关项目材料。
