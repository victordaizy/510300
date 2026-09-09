# 510300 沪深300横截面收益离散度波动预算 V1

- 项目标识：`510300_CSI300_RETURN_DISPERSION_VOLATILITY_V1R1`
- 最终状态：`REJECTED_FROZEN_CSI300_RETURN_DISPERSION_VOLATILITY_GATE_FAILED_NO_RESCUE`
- 机制门：`FAIL`
- 组合收益是否获准计算：`False`
- RETURN_EVALUATION：`NOT_ALLOWED`
- NET_SHARPE：`NOT_COMPUTED`
- 实盘授权：`false`

## 冻结候选

唯一活跃因子是沪深300点时成分股月度总收益的等权横截面样本标准差。模型以本月实现方差为基线控制项，扩展模型只增加本月离散度平方的对数；不设阈值、不反转方向、不组合既有被拒候选。

2015年起逐月扩展窗样本外预测下一自然月000300价格指数实现方差。只有扩展模型相对基线通过系数方向、HAC显著性、移动块自助区间、QLIKE和结构分段全部门槛，才允许读取510300组合收益。

## 机制门结果

- 完整样本外目标月：139
- 离散度系数：0.3965862510501378
- Newey-West t值：3.5958227203460646
- 90%移动块自助区间：[0.20520071485895364, 0.5734531178539024]
- 平均QLIKE改善：0.0193902609813728
- QLIKE DM t值：1.186320068091537

机制门明细：

- `minimum_complete_oos_target_months`：`True`
- `full_sample_beta_rd_positive`：`True`
- `full_sample_beta_rd_one_sided_t`：`True`
- `bootstrap_90pct_lower_beta_rd_positive`：`True`
- `full_sample_qlike_improvement_positive`：`True`
- `full_sample_qlike_dm_one_sided_t`：`False`
- `every_structural_period_beta_rd_positive`：`True`
- `every_structural_period_qlike_improvement_positive`：`True`

## 组合评价

机制门失败，按冻结协议禁止计算策略收益、夏普率或仅用波动基线生成替代仓位。

## 证据边界

历史成员与行情为回溯重建，不能充当历史月末可得性时间戳证明；历史通过也只能进入独立前瞻确认，不能生成订单、连接券商或改变仓位。
