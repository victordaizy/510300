# V2 M/F/T 执行 V1.0.1：持久化 schema 规范化修正

## 保留的失败

V1 首次特征构建通过内存断言并成功写入，但全新进程重放在第一张受影响表 `internal_features` 上封闭失败：持久化 Parquet 的 `tail_diffusion5` 为 `float64`，内存重建值为 `object`。G0 重放收据和 G0 状态均未写入。

对全部10张表做写后语义摘要复核后，8张表与 V1 收据完全一致；只有同时包含 `tail_diffusion5` 的 `internal_features` 和 `mft_feature_panel` 不一致。根因是 V1 在 Parquet 写入前计算语义摘要，而 Parquet 将由布尔左尾标记求和产生的“浮点值加缺失”的 object 列规范化为 `float64`。

## 唯一修正

V1.0.1 不重新生成、不覆盖任何 V1 输出，也不修改 V1 收据。全新进程重算后，仅允许以下两个精确转换：

```text
internal_features.tail_diffusion5: object -> float64
mft_feature_panel.tail_diffusion5: object -> float64
```

转换后必须与不可变 Parquet 的列顺序、dtype、行顺序和每个值完全一致，并使用写后持久化语义摘要。任何其他 dtype 差异、任何值差异、行顺序差异或列顺序差异仍立即失败。

特征公式、经济方向、输入身份、信息时钟、覆盖门、新成分规则、`NO_VIEW` 链和研究/交易权限全部不变。

## 权限边界

修正重放仍不读取实际 BAD10、未来收益、模型结果、警报或组合绩效。只有修正后的全新进程重放和全部定向测试通过，才允许写入追加式 G0 裁决；G1 仍不在本修正内裁决。
