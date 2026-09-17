# 510300 EPS实际持仓覆盖诊断V1

先读 reports/research/510300_eps_disclosed_holdings_diagnostic_v1/研究结论与下一步.md，再读同目录GPT审阅提示.md。当前目标仍未达成；本轮0新策略、0新收益标签、0拟合、0账户。三个旧持仓报告与已有盈利事实揭示重要资产覆盖缺口。

本包包含三个基金原始完整年报及659份直接相关券商原件、原始HTML和目录响应，990条股票持仓和1800条机构公司原点证据；包含旧EPS账户及训练保存物供交叉审阅。旧EPS目录全集的完整性和所有缺失分类没有在本轮重新证明。这里有历史日期的旧基金持仓，不是当日指数权重；三个诊断原点不能外推全历史或实时状态。

Windows复核：把ZIP完整解压到任意新目录，在该目录打开PowerShell。使用已有包含numpy、pandas、pyarrow、pdfplumber的Python；所用版本见研究目录requirements-review.txt。运行 python .\scripts\verify_510300_eps_holdings_packet_v1.py --receipt .\离线复核回执.json。该脚本不联网、不拟合、不生成账户，先验证FILE_INDEX，再从基金PDF提取相关页面，重算保存盈利事实与覆盖边界。不要运行采集器或旧回测入口来“验证”本包。

FILE_INDEX.csv覆盖除索引自身以外全部ZIP成员，提供相对路径、字节数与SHA-256。ZIP检查和数值复核不等于外部研究评审或交易授权。用户请求和原始附件已在研究目录保存。
