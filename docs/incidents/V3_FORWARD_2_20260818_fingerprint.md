# V3_FORWARD_2 2026-08-18 冻结指纹事故

## 处置结论

- 事故状态：`FAILED_FINGERPRINT_SEMANTIC_CHANGE_REQUIRES_NEW_MODEL_ID`
- 处置日期：2026-08-19
- 受影响运行：2026-08-18 日终周期，开始于 `2026-08-18T20:07:06.382395+08:00`
- 旧模型：`V3_FORWARD_2`
- 建议后继模型 ID：`V3_FORWARD_3`，但本事故处置不创建其 manifest，也不授权运行。
- 等价性结论：**不得出具等价性证书。**11 个冻结对象中有 3 个不匹配，其中行业分类输入发生了字段、行数、分类用途和来源变化，会进入正常化盈利方法选择，属于模型输入语义变化。
- 信号结论：失败运行没有产生可消费信号、Outcome、Shadow 仓位或监控报告；不得回填 2026-08-18。
- 实盘边界：`live_trading_authorized=false`，本事故不产生订单、真实仓位映射或券商连接。

旧 `config/v3_forward_2_manifest.json` 未被修改。调查时其 SHA-256 为：

```text
7622574c550cf9993bbd3736c6030df5fbcd8960bb801f9a22af6d97d9e88d35
```

该文件与 `v3-forward-2-frozen-baseline` 所在提交及当前 `HEAD` 的版本均一致。

## 事故时间线

1. `2026-08-13T22:06:17.642518+08:00`：创建 `V3_FORWARD_2` manifest，冻结 11 个对象。
2. `2026-08-16T02:58:31+08:00`：提交 `bab9147d8695cf9919d6a3aa5b398eaeebcb6276`，标签 `v3-forward-2-frozen-baseline` 指向该提交。
3. `2026-08-16T03:25:21+08:00`：提交 `ffcb02cb7ff2d54722f0a424394ae58ea5a612eb` 修改两份已被冻结的 V3 脚本，增加进度、耗时、原子状态及财务断点遥测，但未修改旧 manifest。
4. `2026-08-16T14:24:14.141099+08:00`：`scripts/download_return_tail_auxiliary_data.py` 完成行业区间采集。该脚本在第 290—293 行先保存备份，再把点时行业区间写入 V3 已冻结的共享路径 `data/raw/reference/a_share_sw_industry_static.parquet`。
5. `2026-08-18T20:07:06.382395+08:00`：日终运行开始。输入刷新用时约 1482.837 秒并成功。
6. 进入 `ridge_and_forward_output` 后，冻结校验首先发现 `scripts/refresh_v3_forward_2_inputs.py` 指纹不一致；总运行约 1482.954 秒后以 `FAILED` 结束。

## 指纹审计

| 冻结对象 | manifest 预期 SHA-256 | 调查时实际 SHA-256 | 分类 |
|---|---|---|---|
| `scripts/refresh_v3_forward_2_inputs.py` | `803a179788a16f0c28905904bfc54c16d627376c1df58e3e7c14441244128852` | `332e77accb7bb529619c2a4000dec7a0045eadf766ea03bc9ed08f5d2295741d` | 采集工程语义变化：状态持久化、分步计时和断点遥测；不是仅换行或编码变化 |
| `scripts/run_v3_forward_2_daily.py` | `4753694d8769fabe025428a3b3f845448a6405f901e1abec24c2ed240ba9434a` | `999d7fedeabd2297ffcf7bb0e9a90c5ddf7ca84e98848eb905007582a318f2f5` | 运行工程语义变化：外层周期状态、分步计时和失败传播；不是仅换行或编码变化 |
| `data/raw/reference/a_share_sw_industry_static.parquet` | `ffb84191ea84a3a86b7acfb6df5242267630ad54cf17146a8570352c69649bd3` | `dac8502137bc6dd651300eb86d11d5412cec9ef2de700286513207f7c11b52b6` | **模型输入语义变化** |

其余 8 个冻结对象全部与 manifest 匹配。没有发现“只有 Parquet 压缩、元数据、换行或编码发生变化但表值等价”的冻结对象。

## 模型输入语义变化证据

冻结预期对象仍以备份形式存在：

```text
data/raw/reference/a_share_sw_industry_static_pre_pit.parquet
SHA-256 = ffb84191ea84a3a86b7acfb6df5242267630ad54cf17146a8570352c69649bd3
```

该哈希与 manifest 预期精确一致，因此可以确定原冻结字节没有丢失，但活动路径已经被另一个研究采集流程覆盖。

| 属性 | 冻结输入/备份 | 2026-08-16 覆盖后活动文件 |
|---|---:|---:|
| 行数 × 列数 | 4,430 × 10 | 1,298 × 13 |
| 唯一证券数 | 4,430 | 947 |
| 同一证券多行 | 0 行 | 631 行 |
| `classification_usage` | `EXPLANATORY_STATIC_CLASSIFICATION` | `POINT_IN_TIME_INTERVAL` |
| 来源 | GitHub 文件，声明源为 Tushare Pro | `tushare_proxy.index_member_all` |
| 关键日期字段 | 无 `in_date/out_date` | 有 `in_date/out_date/is_current` |

对两份文件按当前分类比较：共同证券 886 只，其中 41 只一级行业不同；在 2026-07-31 最新 300 只权重截面中，共同覆盖 276 只，其中 6 只一级行业不同，旧文件另有 24 只在新文件中无映射。

这会改变模型结果，而不是只改变审计元数据：

- `research/v3_forward_validation.py` 第 270、322 行读取该文件并传给 `derive_industry_aware_normalization`。
- `research/valuation_v2_expected_return.py` 第 86—94 行按一级行业选择 `FINANCIAL_ROE`、`CYCLICAL_MARGIN` 或 `OTHER_BLEND`。
- 同文件第 139—143 行按 `con_code` 保留最后一条行业记录。覆盖后文件包含历史区间且同一证券可有多行，但该调用没有按信号日筛选区间；保留哪一条记录可能变化。

因此即使绕过指纹校验，2026-08-18 的正常化盈利、增长、Fair PE、Edge 和 Shadow 映射也不再能被视为原 `V3_FORWARD_2` 的输出。

## 根因与促成因素

### 直接根因

1. 冻结后的提交 `ffcb02c` 直接修改了两份 manifest 已列名的脚本。
2. 独立的 return-tail 采集脚本复用了 V3 的冻结行业文件路径，并在通过覆盖率门槛后原子覆盖该路径。

### 治理根因

- manifest 记录了内容哈希，但冻结数据位于被 `.gitignore` 排除的 `data/raw/`，没有独立只读存储或内容寻址路径。
- 多个研究项目共享一个可写文件名；“模型冻结输入”和“其他研究最新数据”没有命名空间隔离。
- 指纹校验发生在约 24 分 43 秒输入刷新之后，而不是运行开始前的只读 preflight。系统最终正确失败关闭，但浪费了采集时间，并扩大了中间输入与失败状态被误读的机会。
- 监控功能直接加入冻结脚本，而不是放在 manifest 外的观察包装层。

## 失败传播与无可消费信号证明

`research/v3_forward_validation.py` 第 950 行在构造状态、读取已有信号或追加输出之前调用 `validate_frozen_manifest()`；实际异常发生在该函数第 159—166 行。信号、Shadow 和 Outcome 的写入分别位于后续第 988、992、999 行。

调查时下列路径均不存在：

```text
data/forward/v3_forward_2/
paper/v3_forward_2_latest_status.json
reports/forward/v3_forward_2/
```

外层失败证据：

- `paper/v3_forward_2_daily_run_status.json`
  - SHA-256：`0856752613fcc978d45810d8145297e426506782950f42bc78c6af769da9fbdb`
  - `status=FAILED`
  - `failed_step=ridge_and_forward_output`
  - 精确错误：`ValueError: 冻结文件指纹不一致：scripts/refresh_v3_forward_2_inputs.py；必须创建新模型版本`
- `paper/v3_forward_2_input_refresh_status.json`
  - SHA-256：`6644502d81f3f7dc2818f5e945156c2669f93badb27ffe1d0bfeeedb1d88cdf2`
  - 只证明五个输入刷新步骤在 2026-08-18 成功，不证明模型运行成功。

结论：刷新后的原始/中间输入可以作为失败事故证据保留，但不是信号，任何 Edge、Shadow 目标仓位或订单都不存在。

## 安全处置

本次不实施代码或数据修复，理由是活动行业文件已经发生模型语义变化。以下动作明确禁止：

- 不修改或重写 `config/v3_forward_2_manifest.json`。
- 不把备份复制回活动路径后声称 2026-08-18 是原模型成功运行。
- 不把冻结后两份脚本的遥测变化认证为整个模型等价。
- 不绕过指纹校验，不回填 2026-08-18 信号。
- 不创建伪等价证书，不沿用 `V3_FORWARD_2` 模型 ID 接受新行业语义。

`V3_FORWARD_2` 保持失败。若要继续，必须建立新模型 ID `V3_FORWARD_3`，并在首个可捕获市场数据之前完成新的协议和 manifest。最低要求：

1. 为模型输入使用独立、不可被其他研究覆盖的内容寻址路径；不得再引用“最新文件”共享路径。
2. 明确选择静态分类还是按信号日生效的点时行业区间；若选择后者，必须冻结区间筛选规则及重叠/缺口处理。
3. 将运行进度、耗时和状态包装器移到模型冻结边界之外，或把它作为新实现的一部分完整冻结。
4. 在任何网络采集前先执行只读 manifest preflight；preflight 失败时不得刷新输入或启动模型。
5. 新实现必须用固定历史夹具证明确定性，并验证 manifest、代码、配置、依赖及输入路径全部唯一绑定。
6. 新模型前瞻起点只能是完整冻结后的下一可捕获交易日，不能追溯到 2026-08-18。

## 调查验收

- 11 个冻结对象已全部重算 SHA-256；结果为 8 个匹配、3 个不匹配。
- 指纹失败反馈环连续重放 3 次，均在同一首个对象上以同一 `ValueError` 失败。
- 冻结行业文件的原始备份哈希与 manifest 预期精确相等。
- 已核对覆盖脚本、采集报告、Git 提交和模型消费调用点。
- 已核对失败状态和所有 V3 输出路径，不存在 2026-08-18 可消费信号。
- 事故调查没有修改旧 manifest、冻结对象、paper 状态或任何原始/中间输入。

