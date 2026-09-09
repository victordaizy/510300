# 510300 非对称压力危险率 V1 外部复核裁决

## 裁决身份

- 裁决日期：2026-09-03
- 裁决类型：追加式外部复核，不覆盖 V1 原始状态、结果、代码、配置或收据
- 来源文件：`E:\CodexData\.codex\attachments\8712825e-d589-4c9f-90dd-dcac55d9f808\pasted-text-1.txt`
- 来源 SHA-256：`bcca9e24481ac3acb780e619550571aeeda7fce57a7b0b84108f879c1705f876`
- 来源字节数：22,559

## 权威裁决

```text
CURRENT_VALIDATED_HIGH_SHARPE_STRATEGY = NONE

510300_ASYMMETRIC_STRESS_HAZARD_V1
ORIGINAL_OUTPUT = PRESERVE_IMMUTABLE
G2_AUTHORITATIVE_RESULT = INVALIDATED
ECONOMIC_HYPOTHESIS = NOT_CLEANLY_ADJUDICATED
PORTFOLIO_EVALUATION = NOT_ALLOWED
LIVE_TRADING_AUTHORIZED = FALSE
POSITION_IMPACT = 0

NEXT_MAINLINE
= 510300_STRESS_TRANSMISSION_HAZARD_V2
= AUTHORIZED_FOR_PROTOCOL_FREEZE_AND_DISCOVERY_ONLY
```

`INVALIDATED` 表示现有 G2 不能承担正式预测裁决，不表示把原输出删除、改写成
失败或把数值取反。原始结果只保留为当时实现及数据口径的历史证据。

## 两项 P0 原因

1. 正式 G2 依据行业分类的历史生效区间回填到 2015 年，但
   `effective_from`、`record_updated_at` 与投资者当时可知的 `known_at` 不是同一
   时钟。历史行业可得来源无法证明，因此行业字段不得继续承担预测用途。
2. 成分收益链在证券首次出现后把普通缺失日填为 0，混淆了真实不变、官方停牌、
   供应商缺失以及未解决公司行动。缺少可靠历史证据时，无法把既有 0 值诚实地
   重分为四态，也就不能声称已有一个“修正后的 G2 AUC”。

## V1 保留边界

- 不修改任何 `510300_ASYMMETRIC_STRESS_HAZARD_V1` 原始代码、配置、测试、结果、
  数据质量报告、收据和交付压缩包。
- 以单独的 SHA-256 清单检测后续漂移；哈希清单不把 V1 结果提升为有效模型。
- 不再次关闭行业时点门运行 G2，不反转低 AUC，不修改 BAD10 阈值或期限，不增加
  变量、窗口、权重或恢复规则进行救援。
- `NO_VIEW`、`RETURN_EVALUATION=NOT_ALLOWED`、`DISCOVERY_ONLY` 与
  `POSITION_IMPACT=0` 保持为状态语义，不翻译成现金仓位、买卖建议或订单。

## V2 许可边界

V2 当前只获准进行协议冻结、数据契约修复、无行业特征的确定性实现与事件账本
建设。G0—G4 全部通过前，禁止读取或生成新的 AUC、组合收益、夏普、回撤、仓位
或订单。若未来进入组合阶段，仍须另行满足冻结协议中的授权门，当前文件不授予
Paper、Shadow、券商连接或实盘权限。

