# 510300官方日频份额版本观察：固定使用合同

2026-10-06，TECH.R243—R244。用途为新官方数据版本观察，未准入金融模型，没有交易动作或收益验证。只观察510300.SH，日线信息；原E03独立。

已完成一次freeze、十项必要测试和一次seed，不重复这些动作。请求直接来自上交所当前官方API，原字段TOT_VOL单位万份；available_at只等于本机实际收到保存时刻，不能用STAT_DATE或网页总规模23:00冒充历史首次公布。

种子为2026-09-30统计值，实际2026-10-06取得，明确排除前瞻和后续份额变化。正式观察从2026-10-08开始；一个统计日一次，只有当前实际交易日23:05以后可调用。缺失/错误日/请求失败保留，不插值、不补旧日期。

📁 C:\Users\戴周阳\Documents\New project 8\research\fund_share_daily_snapshot_observer_v1.py（PowerShell运行入口）

```powershell
Set-Location -LiteralPath 'C:\Users\戴周阳\Documents\New project 8'
.\.venv\Scripts\python.exe -X utf8 -m research.fund_share_daily_snapshot_observer_v1 observe --date 2026-10-08
```

上述命令只在实际2026-10-08 23:05以后执行，本轮没有提前调用。每个新统计日有独立目录、原响应、实际钟和规范值；已经创建的目录禁止覆盖/重跑。未创建自动采集任务。

两份相邻官方交易日的新版本均在使用时点前取得，才能计算原份额增量；一份基线与一份新版本不能配对。首次可能配对10月8日与9日，是否实际取得需看真实回执。10月12日等后续使用时点仍必须晚于双方available_at。

拆分/人民币资金流/投资者身份未建立，不把原份额变化解释为现金净流入或国家队。原G01—G04与八失败规则保持。新的适应性策略还需事前固定完整输入/动作和对照，再比较完整账户净收益/夏普/实际pB与风险，不能由观察器晋升为策略。

当前日历仅2026年，缺日历时停止。当前没有新前瞻份额变化、模型预测或交易信号；目标未达，独立验证未建立。

证据：[实际来源与基线结果](../reports/research/510300_current_official_share_source_v1/当前官方份额_字段时钟与独立日频观察结果.md)、[冻结观察协议](../reports/research/510300_current_official_share_source_v1/daily_observer_protocol.json)、[必要测试回执](../reports/research/510300_current_official_share_source_v1/observer_tests_receipt.json)。
