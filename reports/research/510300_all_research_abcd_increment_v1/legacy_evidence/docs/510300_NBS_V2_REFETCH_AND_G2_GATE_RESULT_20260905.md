# NBS V2：异常月份重取完成，原数据门仍未通过

本轮按用户“请继续，做完数据后继续往前推进”的要求，使用新提供的有效临时凭据完成全部异常月份的来源重取，并准备、冻结和检查了 G2 执行入口。

当前结论：`FULL_DEFECT_MONTH_REFETCH_COMPLETE_G0_FAILED_G2_ENTRY_BLOCKED_BEFORE_RETURN_READ`。

## 数据处理结果

| 项目 | 结果 |
| --- | --- |
| 教程 | 已动态读取用户提供的语雀完整正文 |
| 新凭据状态 | 供应商确认有效，未禁用，具备 stk_mins 独立权限 |
| 供应商有效期 | 北京时间 2026-09-05 20:05:10，到期毫秒见状态回执 |
| 供应商速率 | 120 次/分钟；实际最短请求起点间隔 0.650312 秒 |
| 分钟请求 | 84 次：83 次成功，1 次读取超时 |
| 异常月份重取 | 77 / 77 完成，覆盖全部原异常所在月份 |
| 本轮成功重取的月份数据 | 373,791 根；不含重复的域名探测日计数 |
| 完整候选数据 | 566,350 根、2,350 个开放日 |
| 原始字段变化 | open/high/low/close/vol/amount 全部为 0 |
| 完整候选与原版本关系 | Parquet 字节 SHA-256 完全相同 |

原 tt 域名在 2017 年 7 月分区发生读取超时，先前 6 个月份的成功结果与失败回执保留。随后依用户教程使用同供应商 fast 域名，并先核验三个异常日的全部字段一致，再完成其余 71 个月份。域名、超时与有限传输重试均单独冻结，质量门槛没有变化。

来源数据未发生更正：正成交分钟 VWAP 越界仍有 **263 根**；零成交量、零成交额分钟仍有 **232 根**。后者不能单独定义 VWAP，仍被原实现的可定义性检查阻断。未将两类问题混写为相同的数据错误。

其他硬门仍通过：2,350 个开放日全部 241 根、88 个合格事件主窗口完整、重复主键为 0、非法时间为 0、量额日汇总误差远低于原 0.1% 门槛、主四窗口的既有同源独立采集比对为 4,844/4,844。

完整候选数据 SHA-256：`f39a9f66e0c4bc533cb8323b55b223af8ac218d744b9fcfa92ee1a76aebbfefa`。

## 下一阶段准备和实际执行边界

已经完成 NBS B0/B1 严格前序 G2 实现：36 个初始训练事件、此后每个事件前仅用成熟样本扩展重估、带截距 OLS、HC3 协方差、1.645 均值下界、固定 10,000 次事件 Bootstrap、两个安慰剂的成对重采样比较。

实际日期账本已冻结：86 个模型事件，36/17/17/16 切分；真实事件和安慰剂 B 使用严格此前 60 个非事件日，安慰剂 A 匹配最近此前完整非事件日、不复用，并使用它自身此前 60 日基线。此账本仅包含日期和身份，不包含价格、收益或模型特征。

24 项来源与传输测试、10 项 G2 合成测试均通过。HC3 通过逐点删除系数变化的独立数值验证；测试同时覆盖当期及未来标签不改变当期预测、训练标签成熟时钟、同一安慰剂不能被判为显著更弱、退化 Bootstrap 不重抽，以及 G0 失败时绝不调用标签读取器。

实际运行 G2 入口后返回：`NOT_RUN_BLOCKED_BY_G0`。没有创建收益读取 claim，没有生成真实标签、预测或 Bootstrap 结果文件。

| 阶段 | 当前状态 |
| --- | --- |
| G0 数据源准入 | FAIL_BAR_VWAP_CONTRACT |
| G1 样本与时代 | PASS_36_17_17_16 |
| G2 真实事件预测 | NOT_RUN_BLOCKED_BY_G0 |
| G3 42bp 信号密度和净边际 | NOT_RUN |
| G4 20 万元账户测试 | NOT_RUN |
| 真实事件收益读取 / 标签生成 | 0 / 0 |
| 实际 NBS 模型训练 / 安慰剂收益读取 | FALSE / FALSE |
| 持仓影响 / 实盘授权 | 0 / FALSE |

这是来源准入未通过的结果，尚无 NBS Alpha 有效或无效的实证结论。

## 可继续推进的前提

当前有效合同仍为原 V2 逐分钟硬门。继续推进需要可核验的供应商更正或精确字段解释使原硬门满足；先前准备的五分钟用途合同仍为未生效草案，不能据其诊断通过结果自动放行。

供应商复现材料已整理，尚未发送。它列出可复现请求参数和异常原始字段，不包含凭据。原 V1/V2 和 DSV5 文件未覆盖：原来源 385 个冻结身份、NBS V1 31 个文件及 DSV5 23 个文件已复核无漂移。

## 文件入口

- [来源重取与审核回执](<C:/Users/戴周阳/Documents/New project 8/reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/source_adjudication.json>)
- [全月份重取明细](<C:/Users/戴周阳/Documents/New project 8/reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/acquisition_receipt.json>)
- [供应商复现材料](<C:/Users/戴周阳/Documents/New project 8/reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/SUPPLIER_REPRO_CASES.md>)
- [G2 入口停止回执](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_nbs_v2_g2_execution_v1/gate_check_receipt.json>)
- [从数据到预测的本轮总回执](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_nbs_v2_g2_execution_v1/data_to_prediction_execution_receipt.json>)
- [无收益事件、基线和安慰剂账本](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_nbs_v2_g2_execution_v1/event_baseline_placebo_plan_pre_return.csv>)
- [G2 完整执行代码](<C:/Users/戴周阳/Documents/New project 8/research/nbs_v2_g2_engine.py>)

方法参考：[用户指定的代理使用说明](https://www.yuque.com/aimiao-ke7gj/vosp6x/tsdwcwttw15976iv)、[HC3 官方公式](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html)。
